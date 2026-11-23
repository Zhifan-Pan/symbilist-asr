"""Stage 1 Gate: Symmetric Bidirectional Agreement-Anchored Phrase Recombination (SymBiPhrase) + Direction A+B Synthesis."""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agree_phrase_asr.data_bridge import load_frozen_dev_bundle, SEEDS
from agree_phrase_asr.align import build_projection_bundle, make_projector, flatten_cells, splice_cells, lev_dist
from agree_phrase_asr.partition import agreement_partition, validate_partition
from agree_phrase_asr.features import build_phrase_graph, nonspace_len
from agree_phrase_asr.eval import edit_counts, evaluate_predictions, paired_cluster_bootstrap_diff
from agree_phrase_asr.models import (
    DEFAULT_HGB_PARAMS,
    prepare_fit_targets,
    build_training_arrays,
    predict_block_costs,
    choose_from_block_costs,
)


def fit_regressor_clean(kind: str, tr_ids, fit_targets, meta, seed: int = 1701):
    X, Y, W, _ = build_training_arrays(tr_ids, fit_targets, meta)
    if kind == "hgb":
        p = dict(DEFAULT_HGB_PARAMS, random_state=seed)
        return [HistGradientBoostingRegressor(**p).fit(X, Y[:, i], sample_weight=W) for i in (0, 1)]
    elif kind == "hgb_deep":
        p = dict(DEFAULT_HGB_PARAMS, max_iter=150, max_leaf_nodes=15, max_depth=4, min_samples_leaf=20, random_state=seed)
        return [HistGradientBoostingRegressor(**p).fit(X, Y[:, i], sample_weight=W) for i in (0, 1)]
    elif kind == "mlp":
        # Converged, well-regularized MLP ensemble (3 seeds averaged for low variance!)
        models = []
        for i in (0, 1):
            ens = []
            for mseed in (seed + i * 10, seed + i * 10 + 1, seed + i * 10 + 2):
                pipe = make_pipeline(
                    StandardScaler(),
                    MLPRegressor(
                        hidden_layer_sizes=(64, 32),
                        activation="relu",
                        alpha=2e-2,
                        learning_rate_init=1e-3,
                        max_iter=200,
                        early_stopping=True,
                        n_iter_no_change=15,
                        validation_fraction=0.1,
                        random_state=mseed,
                    ),
                )
                pipe.fit(X, Y[:, i])
                ens.append(pipe)
            models.append(ens)
        return models
    elif kind == "hgb_mlp_blend":
        m_hgb = fit_regressor_clean("hgb_deep", tr_ids, fit_targets, meta, seed=seed)
        m_mlp = fit_regressor_clean("mlp", tr_ids, fit_targets, meta, seed=seed)
        return [m_hgb, m_mlp]
    else:
        raise ValueError(kind)


def predict_clean(kind: str, graphs, uids, models):
    all_x = [bl["X"] for u in uids for bl in graphs[u][1]]
    if not all_x:
        return {u: [] for u in uids}
    stacked = np.vstack(all_x)
    if kind in ("hgb", "hgb_deep"):
        pred = np.stack([m.predict(stacked) for m in models], axis=1)
    elif kind == "mlp":
        pred = np.stack([np.mean([m.predict(stacked) for m in ens], axis=0) for ens in models], axis=1)
    elif kind == "hgb_mlp_blend":
        p1 = np.stack([m.predict(stacked) for m in models[0]], axis=1)
        p2 = np.stack([np.mean([m.predict(stacked) for m in ens], axis=0) for ens in models[1]], axis=1)
        pred = 0.5 * (p1 + p2)
    else:
        raise ValueError(kind)
    pos = 0
    out = {}
    for u in uids:
        zz = []
        for bl in graphs[u][1]:
            n = len(bl["X"])
            zz.append(np.vstack((np.zeros((1, 2), dtype=np.float64), pred[pos : pos + n])))
            pos += n
        out[u] = zz
    return out


def symmetric_sentence_score_feats(h: str, b0: dict, char_lm, word_lm) -> np.ndarray:
    """Purely symmetric sentence-level quality features for candidate h against BOTH source lists (p1, p2)."""
    hw = h.split()
    hc = "".join(hw)
    nw = max(1, len(hw))
    nc = max(1, len(hc))
    c4_raw = float(char_lm(h))
    c4_per = c4_raw / (len(h) + 1)
    w2 = np.asarray(word_lm.scores(h), dtype=np.float64)

    # Dual-source exact word and char expected edit risk (normalized by mean source greedy len)
    g1w, g2w = b0["g1"].split(), b0["g2"].split()
    g1c, g2c = "".join(g1w), "".join(g2w)
    A_sym = max(1.0, 0.5 * (len(g1w) + len(g2w)))
    C_sym = max(1.0, 0.5 * (len(g1c) + len(g2c)))

    r1w = sum(wt * lev_dist(hw, k.split()) for k, wt in b0["p1"].items()) / A_sym
    r2w = sum(wt * lev_dist(hw, k.split()) for k, wt in b0["p2"].items()) / A_sym
    r1c = sum(wt * lev_dist(hc, "".join(k.split())) for k, wt in b0["p1"].items()) / C_sym
    r2c = sum(wt * lev_dist(hc, "".join(k.split())) for k, wt in b0["p2"].items()) / C_sym

    return np.array([
        c4_per,
        c4_raw / (C_sym + 1.0),
        w2[0], w2[1], w2[2],
        r1w, r2w, 0.5 * (r1w + r2w),
        r1c, r2c, 0.5 * (r1c + r2c),
        float(nw) / A_sym,
        float(nc) / C_sym,
        float(h in b0["p1"]),
        float(h in b0["p2"]),
    ], dtype=np.float64)


def run_symmetric_bidirectional_oof(
    kind: str,
    add_ctx: bool,
    pb_omni: dict,
    pb_mms: dict,
    meta: dict,
    folds_by_seed: dict,
    fleurs_uids: list[str],
    kathbath_uids: list[str],
    char_lm,
    word_lm,
    proj_fn,
    alpha_grid: tuple[float, ...] = (1.0, 0.5),
):
    """Run dedicated Omni-anchor (M1) and MMS-anchor (M2) models + Symmetric Bidirectional Reconciliation (SymBiPhrase)."""
    all_uids = fleurs_uids + kathbath_uids
    g1 = {u: build_phrase_graph(pb_omni[u], agreement_partition(pb_omni[u]), char_lm, word_lm, add_block_context_feats=add_ctx) for u in all_uids}
    g2 = {u: build_phrase_graph(pb_mms[u], agreement_partition(pb_mms[u]), char_lm, word_lm, add_block_context_feats=add_ctx) for u in all_uids}

    ft1 = prepare_fit_targets(g1, meta, fleurs_uids)
    ft2 = prepare_fit_targets(g2, meta, fleurs_uids)

    preds_dir1 = {s: {} for s in SEEDS}
    preds_dir2 = {s: {} for s in SEEDS}
    preds_sym_sent = {s: {} for s in SEEDS}
    preds_sym_phrase = {s: {} for s in SEEDS}
    preds_whole_dir1 = {s: {} for s in SEEDS}

    for seed in SEEDS:
        fs = folds_by_seed[seed]
        for f in range(6):
            tr_ids = [u for u in fleurs_uids if f == 5 or fs[u]["outer"] != f]
            ev_ids = kathbath_uids if f == 5 else [u for u in fleurs_uids if fs[u]["outer"] == f]

            def inner_id(u):
                return fs[u]["full"] if f == 5 else fs[u]["inner"][str(f)]

            # Step 1: Inner CV for M1 and M2 independently -> also collect clean inner OOF (y1, y2) for training symmetric arbiter!
            in_p1 = {a: {} for a in alpha_grid}
            in_p2 = {a: {} for a in alpha_grid}
            for j in range(3):
                in_tr = [u for u in tr_ids if inner_id(u) != j]
                in_va = [u for u in tr_ids if inner_id(u) == j]
                m1_in = fit_regressor_clean(kind, in_tr, ft1, meta, seed=1701)
                m2_in = fit_regressor_clean(kind, in_tr, ft2, meta, seed=1701)
                s1_in = predict_clean(kind, g1, in_va, m1_in)
                s2_in = predict_clean(kind, g2, in_va, m2_in)
                for a in alpha_grid:
                    for u in in_va:
                        in_p1[a][u], _ = choose_from_block_costs(g1[u], s1_in[u], pb_omni[u], alpha_c=a)
                        in_p2[a][u], _ = choose_from_block_costs(g2[u], s2_in[u], pb_mms[u], alpha_c=a)

            ev1 = {a: (evaluate_predictions(in_p1[a], meta, sorted(in_p1[a]))["macro_WER"],
                       evaluate_predictions(in_p1[a], meta, sorted(in_p1[a]))["macro_CER"]) for a in alpha_grid}
            ev2 = {a: (evaluate_predictions(in_p2[a], meta, sorted(in_p2[a]))["macro_WER"],
                       evaluate_predictions(in_p2[a], meta, sorted(in_p2[a]))["macro_CER"]) for a in alpha_grid}
            a1 = min(alpha_grid, key=lambda a: (ev1[a][0], ev1[a][1], -a))
            a2 = min(alpha_grid, key=lambda a: (ev2[a][0], ev2[a][1], -a))

            # Step 2: Fit symmetric sentence & residual-span arbiter STRICTLY on inner OOF (in_p1[a1], in_p2[a2])!
            # Symmetry enforced by data augmentation: for every disagreeing pair (y1, y2), add BOTH (f(y2)-f(y1), err(y2)-err(y1)) AND (f(y1)-f(y2), err(y1)-err(y2))!
            per_lang = defaultdict(list)
            for u in tr_ids:
                rw = meta[u]["reference"].split()
                per_lang[meta[u]["language"]].append([len(rw), len("".join(rw))])
            lscale = {l: np.mean(v, axis=0) for l, v in per_lang.items()}

            X_arb_sent, Y_arb_sent, W_arb_sent = [], [], []
            X_arb_span, Y_arb_span, W_arb_span = [], [], []
            for u in tr_ids:
                y1, y2 = in_p1[a1][u], in_p2[a2][u]
                if y1 == y2:
                    continue
                ref = meta[u]["reference"]
                sc = lscale[meta[u]["language"]]
                f1 = symmetric_sentence_score_feats(y1, pb_omni[u], char_lm, word_lm)
                f2 = symmetric_sentence_score_feats(y2, pb_omni[u], char_lm, word_lm)
                e1 = np.array([edit_counts(ref, y1)[0]["E"], edit_counts(ref, y1)[1]["E"]], dtype=np.float64) / sc
                e2 = np.array([edit_counts(ref, y2)[0]["E"], edit_counts(ref, y2)[1]["E"]], dtype=np.float64) / sc
                # Symmetric augmentation: predict (e_cand - e_ref) from (f_cand - f_ref, f_cand, f_ref)
                X_arb_sent.append(np.concatenate([f2 - f1, 0.5 * (f1 + f2)]))
                Y_arb_sent.append(e2 - e1)
                W_arb_sent.append(1.0)
                X_arb_sent.append(np.concatenate([f1 - f2, 0.5 * (f1 + f2)]))
                Y_arb_sent.append(e1 - e2)
                W_arb_sent.append(1.0)

                # Residual agreement span reconciliation training pairs between y1 and y2
                cells_r = proj_fn(y1, y1)
                proj_y2 = proj_fn(y1, y2)
                # Find unanimous words between y1 and y2
                rranges = []
                lo = 0
                for idx_c, cval in enumerate(cells_r):
                    if idx_c % 2 == 1 and proj_y2[idx_c] == cval:
                        if idx_c > lo:
                            rranges.append((lo, idx_c))
                        lo = idx_c + 1
                if lo < len(cells_r):
                    rranges.append((lo, len(cells_r)))
                for lo, hi in rranges:
                    p_a = flatten_cells(cells_r[lo:hi])
                    p_b = flatten_cells(proj_y2[lo:hi])
                    if p_a == p_b:
                        continue
                    h_b = splice_cells(cells_r, [(lo, hi, p_b)])
                    fb = symmetric_sentence_score_feats(h_b, pb_omni[u], char_lm, word_lm)
                    eb = np.array([edit_counts(ref, h_b)[0]["E"], edit_counts(ref, h_b)[1]["E"]], dtype=np.float64) / sc
                    X_arb_span.append(np.concatenate([fb - f1, 0.5 * (f1 + fb)]))
                    Y_arb_span.append(eb - e1)
                    W_arb_span.append(1.0)
                    X_arb_span.append(np.concatenate([f1 - fb, 0.5 * (f1 + fb)]))
                    Y_arb_span.append(e1 - eb)
                    W_arb_span.append(1.0)

            arb_params = dict(max_iter=60, learning_rate=0.05, max_leaf_nodes=7, max_depth=3, min_samples_leaf=20, l2_regularization=5.0, early_stopping=False, random_state=1701)
            Xas, Yas, Was = np.asarray(X_arb_sent), np.asarray(Y_arb_sent), np.asarray(W_arb_sent)
            arb_sent = [HistGradientBoostingRegressor(**arb_params).fit(Xas, Yas[:, i], sample_weight=Was) for i in (0, 1)]
            Xap, Yap, Wap = np.asarray(X_arb_span), np.asarray(Y_arb_span), np.asarray(W_arb_span)
            arb_span = [HistGradientBoostingRegressor(**arb_params).fit(Xap, Yap[:, i], sample_weight=Wap) for i in (0, 1)]

            # Step 3: Fit outer dedicated M1 and M2 on full tr_ids and decode ev_ids!
            m1_out = fit_regressor_clean(kind, tr_ids, ft1, meta, seed=1701)
            m2_out = fit_regressor_clean(kind, tr_ids, ft2, meta, seed=1701)
            s1_ev = predict_clean(kind, g1, ev_ids, m1_out)
            s2_ev = predict_clean(kind, g2, ev_ids, m2_out)

            # Collect directional outputs
            ev_y1, ev_y2 = {}, {}
            for u in ev_ids:
                h1, wh1 = choose_from_block_costs(g1[u], s1_ev[u], pb_omni[u], alpha_c=a1, need_whole=True)
                h2, _ = choose_from_block_costs(g2[u], s2_ev[u], pb_mms[u], alpha_c=a2, need_whole=False)
                ev_y1[u], ev_y2[u] = h1, h2
                preds_dir1[seed][u] = h1
                preds_dir2[seed][u] = h2
                preds_whole_dir1[seed][u] = wh1

            # Reconcile disagreeing pairs symmetrically
            a_sym = 0.5 * (a1 + a2)
            for u in ev_ids:
                y1, y2 = ev_y1[u], ev_y2[u]
                if y1 == y2:
                    preds_sym_sent[seed][u] = y1
                    preds_sym_phrase[seed][u] = y1
                    continue
                f1 = symmetric_sentence_score_feats(y1, pb_omni[u], char_lm, word_lm)
                f2 = symmetric_sentence_score_feats(y2, pb_omni[u], char_lm, word_lm)
                # Exact anti-symmetric evaluation: 0.5 * (f(y2 vs y1) - f(y1 vs y2))!
                x_fwd = np.concatenate([f2 - f1, 0.5 * (f1 + f2)])[None, :]
                x_rev = np.concatenate([f1 - f2, 0.5 * (f1 + f2)])[None, :]
                d_w = 0.5 * (arb_sent[0].predict(x_fwd)[0] - arb_sent[0].predict(x_rev)[0])
                d_c = 0.5 * (arb_sent[1].predict(x_fwd)[0] - arb_sent[1].predict(x_rev)[0])
                # Choose y2 iff minimax regret prefers y2 over y1 (equivalently: max(d_w, a_sym*d_c) < max(-d_w, -a_sym*d_c) -> d_w + a_sym*d_c < 0 for 2 options!)
                # Wait: for 2 options {0, d}: r(0)=max(max(0,-dw), a*max(0,-dc)), r(1)=max(max(0,dw), a*max(0,dc))
                r0 = max(max(0.0, -d_w), a_sym * max(0.0, -d_c))
                r1 = max(max(0.0, d_w), a_sym * max(0.0, d_c))
                preds_sym_sent[seed][u] = y2 if r1 < r0 else y1

                # Pass-2 Symmetric Residual Phrase Reconciliation!
                cells_r = proj_fn(y1, y1)
                proj_y2 = proj_fn(y1, y2)
                rranges = []
                lo = 0
                for idx_c, cval in enumerate(cells_r):
                    if idx_c % 2 == 1 and proj_y2[idx_c] == cval:
                        if idx_c > lo:
                            rranges.append((lo, idx_c))
                        lo = idx_c + 1
                if lo < len(cells_r):
                    rranges.append((lo, len(cells_r)))
                acts = []
                for lo, hi in rranges:
                    p_a = flatten_cells(cells_r[lo:hi])
                    p_b = flatten_cells(proj_y2[lo:hi])
                    if p_a == p_b:
                        continue
                    h_b = splice_cells(cells_r, [(lo, hi, p_b)])
                    fb = symmetric_sentence_score_feats(h_b, pb_omni[u], char_lm, word_lm)
                    xf = np.concatenate([fb - f1, 0.5 * (f1 + fb)])[None, :]
                    xr = np.concatenate([f1 - fb, 0.5 * (f1 + fb)])[None, :]
                    sw = 0.5 * (arb_span[0].predict(xf)[0] - arb_span[0].predict(xr)[0])
                    sc_val = 0.5 * (arb_span[1].predict(xf)[0] - arb_span[1].predict(xr)[0])
                    sr0 = max(max(0.0, -sw), a_sym * max(0.0, -sc_val))
                    sr1 = max(max(0.0, sw), a_sym * max(0.0, sc_val))
                    acts.append((lo, hi, p_b if sr1 < sr0 else p_a))
                preds_sym_phrase[seed][u] = splice_cells(cells_r, acts)

    def _summ(pd):
        out = {}
        for ds_name, ds_uids in [("FLEURS", fleurs_uids), ("Kathbath", kathbath_uids)]:
            ws, cs = [], []
            pl = defaultdict(lambda: {"WER": [], "CER": []})
            for s in SEEDS:
                ev = evaluate_predictions(pd[s], meta, ds_uids)
                ws.append(ev["macro_WER"])
                cs.append(ev["macro_CER"])
                for l, lv in ev["per_lang"].items():
                    pl[l]["WER"].append(lv["WER"])
                    pl[l]["CER"].append(lv["CER"])
            out[ds_name] = {
                "macro_WER_mean": float(np.mean(ws)),
                "macro_WER_sd": float(np.std(ws, ddof=1)),
                "macro_CER_mean": float(np.mean(cs)),
                "macro_CER_sd": float(np.std(cs, ddof=1)),
                "per_lang_mean": {l: {"WER": float(np.mean(v["WER"])), "CER": float(np.mean(v["CER"]))} for l, v in sorted(pl.items())},
            }
        return out

    return {
        "dir1_omni": {"summary": _summ(preds_dir1), "preds": preds_dir1},
        "dir2_mms": {"summary": _summ(preds_dir2), "preds": preds_dir2},
        "sym_sent": {"summary": _summ(preds_sym_sent), "preds": preds_sym_sent},
        "sym_phrase": {"summary": _summ(preds_sym_phrase), "preds": preds_sym_phrase},
        "whole_dir1": {"summary": _summ(preds_whole_dir1), "preds": preds_whole_dir1},
    }


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

    pb_omni = {u: build_projection_bundle(all_data[u][0], all_data[u][1], anchor_idx=0, normalize_fn=norm_fn, project_fn=proj_fn) for u in all_uids}
    pb_mms = {u: build_projection_bundle(all_data[u][0], all_data[u][1], anchor_idx=1, normalize_fn=norm_fn, project_fn=proj_fn) for u in all_uids}

    results = {}
    baseline_A_preds = None

    for cfg_name, kind, add_ctx in [
        ("HGB_24d", "hgb", False),
        ("HGB_Context_32d", "hgb", True),
        ("HGB_deep_Context_32d", "hgb_deep", True),
        ("MLP_ens_Context_32d", "mlp", True),
        ("HGB+MLP_Context_32d", "hgb_mlp_blend", True),
    ]:
        print(f"\n=== Running SymBiPhrase Suite: {cfg_name} ===", flush=True)
        res = run_symmetric_bidirectional_oof(kind, add_ctx, pb_omni, pb_mms, meta, folds, fleurs_uids, kathbath_uids, char_lm, word_lm, proj_fn)
        if cfg_name == "HGB_24d":
            baseline_A_preds = res["dir1_omni"]["preds"]
        results[cfg_name] = {}
        for sub in ["dir1_omni", "dir2_mms", "sym_sent", "sym_phrase"]:
            s = res[sub]["summary"]
            sf, sk = s["FLEURS"], s["Kathbath"]
            ci_f = paired_cluster_bootstrap_diff(res[sub]["preds"][17], baseline_A_preds[17], meta, fleurs_uids, n_draws=10000)
            ci_k = paired_cluster_bootstrap_diff(res[sub]["preds"][17], baseline_A_preds[17], meta, kathbath_uids, n_draws=10000, stratify_by_language=True)
            results[cfg_name][sub] = {"summary": s, "ci_vs_A": {"FLEURS": ci_f, "Kathbath": ci_k}}
            print(
                f"  {sub:12s} | FLEURS {sf['macro_WER_mean']:6.3f}±{sf['macro_WER_sd']:.3f} / {sf['macro_CER_mean']:5.3f} "
                f"[dW={ci_f['delta_WER_pp']:+.3f} {ci_f['delta_WER_CI']}] | "
                f"Kathb {sk['macro_WER_mean']:6.3f} / {sk['macro_CER_mean']:5.3f} "
                f"[dW={ci_k['delta_WER_pp']:+.3f} {ci_k['delta_WER_CI']}]",
                flush=True,
            )

    out_path = ROOT / "results" / "stage1_sym_gate.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n[{time.monotonic()-t0:.1f}s] Saved SymBiPhrase Gate results to {out_path}", flush=True)


if __name__ == "__main__":
    main()
