"""Stage 1 Step 3: Direction A — Anchor Symmetry, Unsupervised/Learned Dynamic Anchor, and Soft Agreement Boundaries."""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agree_phrase_asr.data_bridge import load_frozen_dev_bundle, SEEDS
from agree_phrase_asr.align import build_projection_bundle, make_projector, lev_dist
from agree_phrase_asr.partition import (
    agreement_partition,
    soft_agreement_partition,
    validate_partition,
)
from agree_phrase_asr.features import build_phrase_graph, nonspace_len
from agree_phrase_asr.eval import edit_counts, evaluate_predictions, paired_cluster_bootstrap_diff
from agree_phrase_asr.models import (
    DEFAULT_HGB_PARAMS,
    run_grouped_oof,
    prepare_fit_targets,
    fit_dual_regressor,
    predict_block_costs,
    choose_from_block_costs,
)


def utterance_anchor_features(b0: dict, char_lm, word_lm) -> np.ndarray:
    """Symmetric reference-free utterance features for predicting cleaner alignment anchor."""
    g1, g2 = b0["g1"], b0["g2"]
    p1, p2 = b0["p1"], b0["p2"]
    c4_1 = float(char_lm(g1)) / (len(g1) + 1)
    c4_2 = float(char_lm(g2)) / (len(g2) + 1)
    w2_1 = np.asarray(word_lm.scores(g1), dtype=np.float64)
    w2_2 = np.asarray(word_lm.scores(g2), dtype=np.float64)

    nw1, nw2 = max(1, len(g1.split())), max(1, len(g2.split()))
    nc1, nc2 = max(1, nonspace_len(g1)), max(1, nonspace_len(g2))
    cpw1, cpw2 = nc1 / nw1, nc2 / nw2

    wdiv = lev_dist(g1.split(), g2.split()) / max(1, 0.5 * (nw1 + nw2))
    cdiv = lev_dist("".join(g1.split()), "".join(g2.split())) / max(1, 0.5 * (nc1 + nc2))

    def _ent(d):
        v = np.array(list(d.values()), dtype=np.float64)
        v = np.clip(v, 1e-12, 1.0)
        return float(-(v * np.log(v)).sum())

    return np.array([
        c4_1, c4_2, c4_2 - c4_1,
        w2_1[0], w2_2[0], w2_2[0] - w2_1[0],
        w2_1[1], w2_2[1], w2_2[1] - w2_1[1],
        w2_1[2], w2_2[2], w2_2[2] - w2_1[2],
        float(nw1), float(nw2), float(nw2) / float(nw1),
        float(nc1), float(nc2), float(nc2) / float(nc1),
        cpw1, cpw2, cpw2 - cpw1,
        wdiv, cdiv, wdiv - cdiv,
        p1.get(g1, 0.0), p2.get(g2, 0.0), p2.get(g2, 0.0) - p1.get(g1, 0.0),
        _ent(p1), _ent(p2), float(len(p1)), float(len(p2)),
    ], dtype=np.float64)


def run_learned_dynamic_anchor_oof(
    pb_omni: dict[str, dict],
    pb_mms: dict[str, dict],
    g_omni: dict[str, tuple],
    g_mms: dict[str, tuple],
    utt_feats: dict[str, np.ndarray],
    meta: dict[str, dict],
    folds_by_seed: dict[int, dict],
    fleurs_uids: list[str],
    kathbath_uids: list[str],
    alpha_grid: tuple[float, ...] = (1.0, 0.5),
    need_whole: bool = True,
) -> dict:
    """Strictly leakage-free grouped OOF where the utterance anchor router is fit inside each training fold!"""
    ft_omni = prepare_fit_targets(g_omni, meta, fleurs_uids)
    ft_mms = prepare_fit_targets(g_mms, meta, fleurs_uids)

    # Precompute utterance anchor training labels on FLEURS (never touched for val/test!)
    utt_y = {}
    utt_w = {}
    for u in fleurs_uids:
        ref = meta[u]["reference"]
        w1, c1 = edit_counts(ref, pb_omni[u]["g1"])
        w2, c2 = edit_counts(ref, pb_omni[u]["g2"])
        # Target: 1 if MMS greedy has fewer word edits (or tie with fewer char edits)
        diff_w = w1["E"] - w2["E"]
        diff_c = c1["E"] - c2["E"]
        utt_y[u] = 1 if (diff_w > 0 or (diff_w == 0 and diff_c > 0)) else 0
        utt_w[u] = float(max(0.5, abs(diff_w) / max(1, w1["N"]) * 10.0))

    def fit_anchor_router(tr_ids):
        X = np.stack([utt_feats[u] for u in tr_ids], axis=0)
        y = np.array([utt_y[u] for u in tr_ids], dtype=np.int32)
        w = np.array([utt_w[u] for u in tr_ids], dtype=np.float64)
        # Equal total weight per language
        lang_cnt = defaultdict(float)
        for i, u in enumerate(tr_ids):
            lang_cnt[meta[u]["language"]] += w[i]
        for i, u in enumerate(tr_ids):
            w[i] /= lang_cnt[meta[u]["language"]]
        w *= len(w) / w.sum()
        clf = HistGradientBoostingClassifier(
            max_iter=80, learning_rate=0.05, max_leaf_nodes=7, max_depth=3,
            min_samples_leaf=25, l2_regularization=2.0, early_stopping=False, random_state=1701,
        )
        clf.fit(X, y, sample_weight=w)
        return clf

    def predict_anchor_choice(clf, uids, threshold=0.5):
        X = np.stack([utt_feats[u] for u in uids], axis=0)
        probs = clf.predict_proba(X)[:, 1]
        return {u: int(p >= threshold) for u, p in zip(uids, probs)}

    all_preds, all_whole = {}, {}
    router_stats = {s: {"FLEURS_mms_frac": [], "Kathbath_mms_frac": 0.0} for s in SEEDS}

    for seed in SEEDS:
        fs = folds_by_seed[seed]
        preds, whole_preds = {}, {}
        for f in range(6):
            tr_ids = [u for u in fleurs_uids if f == 5 or fs[u]["outer"] != f]
            ev_ids = kathbath_uids if f == 5 else [u for u in fleurs_uids if fs[u]["outer"] == f]

            def inner_id(u: str) -> int:
                return fs[u]["full"] if f == 5 else fs[u]["inner"][str(f)]

            # Inner CV: fit router on inner train, get OOF anchor choices on inner val, select alpha
            inner_preds = {a: {} for a in alpha_grid}
            for j in range(3):
                in_tr = [u for u in tr_ids if inner_id(u) != j]
                in_va = [u for u in tr_ids if inner_id(u) == j]
                clf_in = fit_anchor_router(in_tr)
                # Use OOF/predicted anchor on in_tr (or router choice on in_tr) to pick graph for training!
                tr_choice = predict_anchor_choice(clf_in, in_tr)
                va_choice = predict_anchor_choice(clf_in, in_va)
                g_in_tr = {u: (g_mms[u] if tr_choice[u] == 1 else g_omni[u]) for u in in_tr}
                ft_in_tr = {u: (ft_mms[u] if tr_choice[u] == 1 else ft_omni[u]) for u in in_tr if (u in ft_mms if tr_choice[u] == 1 else u in ft_omni)}
                mod_in = fit_dual_regressor("hgb", in_tr, ft_in_tr, meta, seed=1701)

                g_in_va = {u: (g_mms[u] if va_choice[u] == 1 else g_omni[u]) for u in in_va}
                b_in_va = {u: (pb_mms[u] if va_choice[u] == 1 else pb_omni[u]) for u in in_va}
                sc_va = predict_block_costs(g_in_va, in_va, mod_in)
                for a in alpha_grid:
                    for u in in_va:
                        h, _ = choose_from_block_costs(g_in_va[u], sc_va[u], b_in_va[u], alpha_c=a)
                        inner_preds[a][u] = h

            inner_ev = {a: (evaluate_predictions(inner_preds[a], meta, sorted(inner_preds[a]))["macro_WER"],
                            evaluate_predictions(inner_preds[a], meta, sorted(inner_preds[a]))["macro_CER"]) for a in alpha_grid}
            winner = min(alpha_grid, key=lambda a: (inner_ev[a][0], inner_ev[a][1], -a))

            # Fit outer router & outer phrase GBDT on tr_ids
            clf_out = fit_anchor_router(tr_ids)
            tr_ch = predict_anchor_choice(clf_out, tr_ids)
            ev_ch = predict_anchor_choice(clf_out, ev_ids)
            if f < 5:
                router_stats[seed]["FLEURS_mms_frac"].append(float(np.mean(list(ev_ch.values()))))
            else:
                router_stats[seed]["Kathbath_mms_frac"] = float(np.mean(list(ev_ch.values())))

            ft_out = {u: (ft_mms[u] if tr_ch[u] == 1 else ft_omni[u]) for u in tr_ids if (u in ft_mms if tr_ch[u] == 1 else u in ft_omni)}
            mod_out = fit_dual_regressor("hgb", tr_ids, ft_out, meta, seed=1701)

            g_ev = {u: (g_mms[u] if ev_ch[u] == 1 else g_omni[u]) for u in ev_ids}
            b_ev = {u: (pb_mms[u] if ev_ch[u] == 1 else pb_omni[u]) for u in ev_ids}
            sc_ev = predict_block_costs(g_ev, ev_ids, mod_out)
            for u in ev_ids:
                h, wh = choose_from_block_costs(g_ev[u], sc_ev[u], b_ev[u], alpha_c=winner, need_whole=need_whole)
                preds[u] = h
                if need_whole and wh is not None:
                    whole_preds[u] = wh

        all_preds[seed] = preds
        all_whole[seed] = whole_preds

    summary = {}
    for ds_name, ds_uids in [("FLEURS", fleurs_uids), ("Kathbath", kathbath_uids)]:
        wers, cers = [], []
        per_lang = defaultdict(lambda: {"WER": [], "CER": []})
        for seed in SEEDS:
            ev = evaluate_predictions(all_preds[seed], meta, ds_uids)
            wers.append(ev["macro_WER"])
            cers.append(ev["macro_CER"])
            for l, lv in ev["per_lang"].items():
                per_lang[l]["WER"].append(lv["WER"])
                per_lang[l]["CER"].append(lv["CER"])
        summary[ds_name] = {
            "macro_WER_mean": float(np.mean(wers)),
            "macro_WER_sd": float(np.std(wers, ddof=1)),
            "macro_CER_mean": float(np.mean(cers)),
            "macro_CER_sd": float(np.std(cers, ddof=1)),
            "per_lang_mean": {l: {"WER": float(np.mean(v["WER"])), "CER": float(np.mean(v["CER"]))} for l, v in sorted(per_lang.items())},
        }
    return {"summary": summary, "preds": all_preds, "whole_preds": all_whole, "router_stats": router_stats}


def main():
    t0 = time.monotonic()
    bundle = load_frozen_dev_bundle()
    all_data = bundle["all_data"]
    meta = bundle["meta"]
    folds = bundle["folds_by_seed"]
    fleurs_uids = bundle["fleurs_uids"]
    kathbath_uids = bundle["kathbath_uids"]
    all_uids = fleurs_uids + kathbath_uids
    char_lm = bundle["char_lm_fn"]
    word_lm = bundle["word_lm"]
    norm_fn = bundle["normalize_fn"]
    proj_fn = make_projector(bundle["align_fn"])

    print(f"[{time.monotonic()-t0:.1f}s] Building Omni-anchor and MMS-anchor projection bundles...", flush=True)
    pb_omni = {u: build_projection_bundle(all_data[u][0], all_data[u][1], anchor_idx=0, normalize_fn=norm_fn, project_fn=proj_fn) for u in all_uids}
    pb_mms = {u: build_projection_bundle(all_data[u][0], all_data[u][1], anchor_idx=1, normalize_fn=norm_fn, project_fn=proj_fn) for u in all_uids}

    # Baseline A (Omni anchor)
    g_A = {u: build_phrase_graph(pb_omni[u], agreement_partition(pb_omni[u]), char_lm, word_lm) for u in all_uids}
    res_A = run_grouped_oof(g_A, pb_omni, meta, folds, fleurs_uids, kathbath_uids, need_whole=True)

    results = {"baseline_A": res_A["summary"], "A0_swapped": {}, "A1_dynamic_anchor": {}, "A2_soft_agreement": {}, "A3_combined": {}, "bootstrap_vs_A": {}}

    # ---- A0: Swapped Anchor (MMS as Anchor, Omni as Donor) ----
    print("\n=== A0: Swapped Anchor (MMS -> Omni) ===", flush=True)
    for order in ["fixed_id", "anchor_first"]:
        g_swap = {u: build_phrase_graph(pb_mms[u], agreement_partition(pb_mms[u]), char_lm, word_lm, source_order=order) for u in all_uids}
        r_swap = run_grouped_oof(g_swap, pb_mms, meta, folds, fleurs_uids, kathbath_uids)
        results["A0_swapped"][order] = r_swap["summary"]
        sf, sk = r_swap["summary"]["FLEURS"], r_swap["summary"]["Kathbath"]
        print(f"  Swap ({order:12s}) | FLEURS {sf['macro_WER_mean']:6.3f}±{sf['macro_WER_sd']:.3f} / {sf['macro_CER_mean']:5.3f} | Kathbath {sk['macro_WER_mean']:6.3f} / {sk['macro_CER_mean']:5.3f}", flush=True)

    # ---- A1: Dynamic Anchor Selection ----
    print("\n=== A1: Dynamic Anchor Selection ===", flush=True)
    # A1-a: Unsupervised Char4-LM Dynamic Anchor (fixed_id + anchor_id feat)
    dyn_char_choice = {}
    for u in all_uids:
        g1, g2 = pb_omni[u]["g1"], pb_omni[u]["g2"]
        s1 = float(char_lm(g1)) / (len(g1) + 1)
        s2 = float(char_lm(g2)) / (len(g2) + 1)
        dyn_char_choice[u] = 1 if s2 > s1 else 0
    f_mms_frac = np.mean([dyn_char_choice[u] for u in fleurs_uids])
    k_mms_frac = np.mean([dyn_char_choice[u] for u in kathbath_uids])
    print(f"  DynAnchor_CharLM MMS-anchor frac: FLEURS={f_mms_frac:.3f}, Kathbath={k_mms_frac:.3f}", flush=True)

    for order in ["fixed_id", "anchor_first"]:
        g_dyn_lm = {
            u: build_phrase_graph(
                pb_mms[u] if dyn_char_choice[u] == 1 else pb_omni[u],
                agreement_partition(pb_mms[u] if dyn_char_choice[u] == 1 else pb_omni[u]),
                char_lm, word_lm, source_order=order, add_anchor_id_feat=True,
            )
            for u in all_uids
        }
        pb_dyn_lm = {u: (pb_mms[u] if dyn_char_choice[u] == 1 else pb_omni[u]) for u in all_uids}
        r_dlm = run_grouped_oof(g_dyn_lm, pb_dyn_lm, meta, folds, fleurs_uids, kathbath_uids, need_whole=True)
        results["A1_dynamic_anchor"][f"CharLM_{order}"] = r_dlm["summary"]
        sf, sk = r_dlm["summary"]["FLEURS"], r_dlm["summary"]["Kathbath"]
        print(f"  DynAnchor_CharLM ({order:12s}) | FLEURS {sf['macro_WER_mean']:6.3f}±{sf['macro_WER_sd']:.3f} / {sf['macro_CER_mean']:5.3f} | Kathbath {sk['macro_WER_mean']:6.3f} / {sk['macro_CER_mean']:5.3f}", flush=True)

    # A1-b: Learned Leakage-Free Utterance Anchor Router
    utt_feats = {u: utterance_anchor_features(pb_omni[u], char_lm, word_lm) for u in all_uids}
    g_omni_id = {u: build_phrase_graph(pb_omni[u], agreement_partition(pb_omni[u]), char_lm, word_lm, source_order="fixed_id", add_anchor_id_feat=True) for u in all_uids}
    g_mms_id = {u: build_phrase_graph(pb_mms[u], agreement_partition(pb_mms[u]), char_lm, word_lm, source_order="fixed_id", add_anchor_id_feat=True) for u in all_uids}
    r_learned = run_learned_dynamic_anchor_oof(pb_omni, pb_mms, g_omni_id, g_mms_id, utt_feats, meta, folds, fleurs_uids, kathbath_uids)
    results["A1_dynamic_anchor"]["Learned_fixed_id"] = r_learned["summary"]
    results["A1_dynamic_anchor"]["Learned_router_stats"] = r_learned["router_stats"]
    sf, sk = r_learned["summary"]["FLEURS"], r_learned["summary"]["Kathbath"]
    print(f"  DynAnchor_Learned (fixed_id)   | FLEURS {sf['macro_WER_mean']:6.3f}±{sf['macro_WER_sd']:.3f} / {sf['macro_CER_mean']:5.3f} | Kathbath {sk['macro_WER_mean']:6.3f} / {sk['macro_CER_mean']:5.3f}", flush=True)

    # ---- A2: Soft Agreement Boundaries (tau sweep on Omni anchor) ----
    print("\n=== A2: Soft Agreement Boundaries (Omni Anchor) ===", flush=True)
    soft_configs = [
        ("min_source", 0.99),
        ("min_source", 0.95),
        ("min_source", 0.90),
        ("min_source", 0.85),
        ("min_source", 0.75),
        ("min_source", 0.60),
        ("mean_source", 0.90),
        ("mean_source", 0.80),
        ("mean_source", 0.70),
        ("greedy_agree", 1.0),
    ]
    best_soft = None
    best_soft_wer = 999.0
    best_soft_preds = None
    for mode, tau in soft_configs:
        cname = f"{mode}_tau{tau:.2f}" if mode != "greedy_agree" else "greedy_agree"
        g_soft = {}
        f_vblocks = []
        for u in all_uids:
            part = soft_agreement_partition(pb_omni[u], tau=tau, mode=mode)
            validate_partition(pb_omni[u], part)
            g_soft[u] = build_phrase_graph(pb_omni[u], part, char_lm, word_lm)
            if u in bundle["fleurs"]:
                f_vblocks.append(len(g_soft[u][1]))
        r_s = run_grouped_oof(g_soft, pb_omni, meta, folds, fleurs_uids, kathbath_uids)
        sf, sk = r_s["summary"]["FLEURS"], r_s["summary"]["Kathbath"]
        r_s["summary"]["FLEURS_mean_var_blocks"] = float(np.mean(f_vblocks))
        results["A2_soft_agreement"][cname] = r_s["summary"]
        print(f"  {cname:22s} (vb={np.mean(f_vblocks):.2f}) | FLEURS {sf['macro_WER_mean']:6.3f}±{sf['macro_WER_sd']:.3f} / {sf['macro_CER_mean']:5.3f} | Kathbath {sk['macro_WER_mean']:6.3f} / {sk['macro_CER_mean']:5.3f}", flush=True)
        if sf["macro_WER_mean"] < best_soft_wer:
            best_soft_wer = sf["macro_WER_mean"]
            best_soft = (mode, tau, cname)
            best_soft_preds = r_s["preds"]

    # ---- A3: Combined Dynamic Anchor + Best Soft Agreement ----
    print(f"\n=== A3: Combined Direction A (Dynamic Anchor + Soft Agreement {best_soft}) ===", flush=True)
    b_mode, b_tau, b_cname = best_soft
    # Test Learned Dynamic Anchor + Soft Agreement (top 2 soft settings: min_source 0.85, 0.75, greedy_agree, plus best_soft)
    combo_candidates = list(dict.fromkeys([(b_mode, b_tau), ("min_source", 0.85), ("min_source", 0.75), ("greedy_agree", 1.0)]))
    best_combo_name = None
    best_combo_preds = None
    best_combo_wer = 999.0
    for mode, tau in combo_candidates:
        cname = f"LearnedDynAnchor+{mode}_tau{tau:.2f}" if mode != "greedy_agree" else "LearnedDynAnchor+greedy_agree"
        g_o_s = {u: build_phrase_graph(pb_omni[u], soft_agreement_partition(pb_omni[u], tau=tau, mode=mode), char_lm, word_lm, source_order="fixed_id", add_anchor_id_feat=True) for u in all_uids}
        g_m_s = {u: build_phrase_graph(pb_mms[u], soft_agreement_partition(pb_mms[u], tau=tau, mode=mode), char_lm, word_lm, source_order="fixed_id", add_anchor_id_feat=True) for u in all_uids}
        r_c = run_learned_dynamic_anchor_oof(pb_omni, pb_mms, g_o_s, g_m_s, utt_feats, meta, folds, fleurs_uids, kathbath_uids)
        results["A3_combined"][cname] = r_c["summary"]
        sf, sk = r_c["summary"]["FLEURS"], r_c["summary"]["Kathbath"]
        print(f"  {cname:42s} | FLEURS {sf['macro_WER_mean']:6.3f}±{sf['macro_WER_sd']:.3f} / {sf['macro_CER_mean']:5.3f} | Kathbath {sk['macro_WER_mean']:6.3f} / {sk['macro_CER_mean']:5.3f}", flush=True)
        if sf["macro_WER_mean"] < best_combo_wer:
            best_combo_wer = sf["macro_WER_mean"]
            best_combo_name = cname
            best_combo_preds = r_c["preds"]

    # Also test CharLM Dynamic Anchor + Best Soft Agreement
    g_dlm_s = {}
    pb_dlm_s = {}
    for u in all_uids:
        pb_u = pb_mms[u] if dyn_char_choice[u] == 1 else pb_omni[u]
        pb_dlm_s[u] = pb_u
        g_dlm_s[u] = build_phrase_graph(pb_u, soft_agreement_partition(pb_u, tau=b_tau, mode=b_mode), char_lm, word_lm, source_order="fixed_id", add_anchor_id_feat=True)
    r_dlm_s = run_grouped_oof(g_dlm_s, pb_dlm_s, meta, folds, fleurs_uids, kathbath_uids)
    results["A3_combined"][f"CharLMDynAnchor+{b_cname}"] = r_dlm_s["summary"]
    sf, sk = r_dlm_s["summary"]["FLEURS"], r_dlm_s["summary"]["Kathbath"]
    print(f"  {'CharLMDynAnchor+'+b_cname:42s} | FLEURS {sf['macro_WER_mean']:6.3f}±{sf['macro_WER_sd']:.3f} / {sf['macro_CER_mean']:5.3f} | Kathbath {sk['macro_WER_mean']:6.3f} / {sk['macro_CER_mean']:5.3f}", flush=True)

    # Paired cluster bootstrap vs Baseline A (on seed 17 representative + print deltas)
    print("\n=== Paired Cluster Bootstrap vs Baseline A (seed=17, 10k draws) ===", flush=True)
    for label, pdict in [
        ("DynAnchor_Learned", r_learned["preds"]),
        (f"BestSoft({b_cname})", best_soft_preds),
        (f"BestCombo({best_combo_name})", best_combo_preds),
    ]:
        ci_f = paired_cluster_bootstrap_diff(pdict[17], res_A["preds"][17], meta, fleurs_uids, n_draws=10000, stratify_by_language=False)
        ci_k = paired_cluster_bootstrap_diff(pdict[17], res_A["preds"][17], meta, kathbath_uids, n_draws=10000, stratify_by_language=True)
        results["bootstrap_vs_A"][label] = {"FLEURS": ci_f, "Kathbath": ci_k}
        print(f"  {label:38s} | FLEURS dWER={ci_f['delta_WER_pp']:+.3f} {ci_f['delta_WER_CI']} dCER={ci_f['delta_CER_pp']:+.3f} {ci_f['delta_CER_CI']}")
        print(f"  {'':38s} | Kathb  dWER={ci_k['delta_WER_pp']:+.3f} {ci_k['delta_WER_CI']} dCER={ci_k['delta_CER_pp']:+.3f} {ci_k['delta_CER_CI']}", flush=True)

    out_path = ROOT / "results" / "stage1_direction_a.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n[{time.monotonic()-t0:.1f}s] Saved Direction A results to {out_path}", flush=True)


if __name__ == "__main__":
    main()
