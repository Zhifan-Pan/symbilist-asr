"""Stage 2A: Cross-Architecture (CTC + AED) & 3-System Combination (OmniASR-CTC + MMS-CTC + Whisper-v3-turbo AED) on FLEURS-12 & Kathbath valid."""
from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from agree_phrase_asr.data_bridge import load_frozen_dev_bundle, SEEDS
from agree_phrase_asr.align import (
    build_projection_bundle,
    collapse_candidates,
    make_projector,
    flatten_cells,
    splice_cells,
    lev_dist,
)
from agree_phrase_asr.partition import agreement_partition, fixed_partition, validate_partition
from agree_phrase_asr.features import build_phrase_graph, nonspace_len
from agree_phrase_asr.eval import edit_counts, evaluate_predictions, paired_cluster_bootstrap_diff
from agree_phrase_asr.models import (
    DEFAULT_HGB_PARAMS,
    run_grouped_oof,
    prepare_fit_targets,
    fit_dual_regressor,
    predict_block_costs,
    choose_from_block_costs,
    predict_exact_nbest_mbr,
)
from run_stage2_cv26 import fit_listwise_ens, pred_listwise_ens, fit_and_apply_sym_reconciler

# 9 of 12 FLEURS languages natively supported by Whisper-large-v3-turbo (excluding ast_es, ckb_iq, kea_cv)
WHISPER_NATIVE_9 = {"am_et", "bn_in", "ha_ng", "he_il", "hy_am", "ka_ge", "kn_in", "ta_in", "uk_ua"}


def build_3way_bundle(r1, r2, r3, anchor_idx: int, norm_fn, proj_fn):
    """3-System projection bundle over S=3 heterogeneous recognizers (OmniASR-CTC, MMS-CTC, Whisper-AED)."""
    g1, p1 = collapse_candidates(r1, mode="weighted", temperature=0.5, normalize_fn=norm_fn)
    g2, p2 = collapse_candidates(r2, mode="weighted", temperature=0.5, normalize_fn=norm_fn)
    g3, p3 = collapse_candidates(r3, mode="weighted", temperature=0.5, normalize_fn=norm_fn)
    greedys = [g1, g2, g3]
    dists = [p1, p2, p3]
    base = greedys[anchor_idx]
    texts = list(dict.fromkeys([base, *p1.keys(), *p2.keys(), *p3.keys()]))
    proj = {h: proj_fn(base, h) for h in texts}
    return {
        "anchor_idx": anchor_idx,
        "base": base,
        "greedys": greedys,
        "dists": dists,
        "texts": texts,
        "proj": proj,
        "cells": proj[base],
    }


def build_3way_graph(bundle, char_lm, word_lm):
    """Agreement-anchored phrase graph for S=3 recognizers (30 base + 8 context = 38 features)."""
    base = bundle["base"]
    texts = bundle["texts"]
    proj = bundle["proj"]
    cells = bundle["cells"]
    proj_vals = list(proj.values())
    ranges, lo = [], 0
    for i, cell in enumerate(cells):
        if i % 2 == 1 and all(z[i] == cell for z in proj_vals):
            if i > lo:
                ranges.append((lo, i))
            ranges.append((i, i + 1))
            lo = i + 1
    if lo < len(cells):
        ranges.append((lo, len(cells)))

    A = max(1, len(base.split()))
    C_base = max(1, nonspace_len(base))
    divs = [lev_dist(base.split(), g.split()) / A for i, g in enumerate(bundle["greedys"]) if i != bundle["anchor_idx"]]
    div_mean = float(np.mean(divs))
    base_ll = float(char_lm(base))
    base_wlm = np.asarray(word_lm.scores(base), dtype=np.float64)

    blocks = []
    for lo, hi in ranges:
        phrases = {h: flatten_cells(proj[h][lo:hi]) for h in texts}
        v0 = phrases[base]
        alts = list(dict.fromkeys(phrases.values()))
        if len(alts) < 2:
            continue
        p_dists = []
        for weights in bundle["dists"]:
            d = Counter()
            for h, wt in weights.items():
                d[phrases[h]] += wt
            p_dists.append(d)

        xs_all, full_texts = [], []
        for v in alts:
            h = splice_cells(cells, [(lo, hi, v)])
            full_texts.append(h)
            ll = float(char_lm(h))
            wlm_d = np.asarray(word_lm.scores(h), dtype=np.float64) - base_wlm
            feats = [
                div_mean, lo / (2 * A + 1), len(alts), len(v0), len(v),
                lev_dist(v0, v) / A, len("".join(v)) - len("".join(v0)),
                ll / (len(h) + 1) - base_ll / (len(base) + 1),
                (ll - base_ll) / (len(base) + 1),
                *wlm_d.tolist(),
            ]
            vs, v0s = "".join(v), "".join(v0)
            for d in p_dists:
                dw = sum(wt * (lev_dist(v, a) - lev_dist(v0, a)) for a, wt in d.items())
                dc = sum(wt * (lev_dist(vs, "".join(a)) - lev_dist(v0s, "".join(a))) for a, wt in d.items())
                feats.extend([d[v], d[v0], dw / A, dc / C_base, len(d), max(d.values())])
            xs_all.append(feats)

        X_full = np.asarray(xs_all, dtype=np.float64)
        # Intra-block context over 3 sources (mass cols: 12,18,24; wr cols: 14,20,26; cr cols: 15,21,27)
        c4 = X_full[:, 7]
        w2 = X_full[:, 9]
        tot_m = (X_full[:, 12] + X_full[:, 18] + X_full[:, 24]) / 3.0
        tot_wr = (X_full[:, 14] + X_full[:, 20] + X_full[:, 26]) / 3.0
        tot_cr = (X_full[:, 15] + X_full[:, 21] + X_full[:, 27]) / 3.0
        ctx = np.column_stack([
            c4 - c4.max(), w2 - w2.max(), tot_m - tot_m.max(),
            tot_wr - tot_wr.min(), tot_cr - tot_cr.min(),
            (tot_wr == tot_wr.min()).astype(np.float64),
            (c4 == c4.max()).astype(np.float64),
            np.full(len(alts), np.std(tot_wr)),
        ])
        X_full = np.hstack([X_full, ctx])
        blocks.append({"lo": lo, "hi": hi, "alts": alts, "full": full_texts, "X": X_full[1:], "X_all": X_full})
    return cells, blocks


def run_2way_pair(name, rows_s1, rows_s2, meta, folds, fleurs_uids, kathbath_uids, char_lm, word_lm, norm_fn, proj_fn):
    """Evaluate a 2-system pair on FLEURS (seed 17 grouped 5-fold OOF) + Kathbath valid (full FLEURS fit)."""
    all_uids = fleurs_uids + kathbath_uids
    pb1 = {u: build_projection_bundle(rows_s1[u], rows_s2[u], anchor_idx=0, normalize_fn=norm_fn, project_fn=proj_fn) for u in all_uids}
    pb2 = {u: build_projection_bundle(rows_s1[u], rows_s2[u], anchor_idx=1, normalize_fn=norm_fn, project_fn=proj_fn) for u in all_uids}
    g1 = {u: build_phrase_graph(pb1[u], agreement_partition(pb1[u]), char_lm, word_lm, add_block_context_feats=True) for u in all_uids}
    g2 = {u: build_phrase_graph(pb2[u], agreement_partition(pb2[u]), char_lm, word_lm, add_block_context_feats=True) for u in all_uids}
    gF = {u: build_phrase_graph(pb1[u], fixed_partition(pb1[u], 3), char_lm, word_lm) for u in all_uids}

    fs = folds[17]
    cols24 = list(range(24))
    ft1 = prepare_fit_targets(g1, meta, fleurs_uids)
    ft2 = prepare_fit_targets(g2, meta, fleurs_uids)
    ftF = prepare_fit_targets(gF, meta, fleurs_uids)

    preds = {k: {} for k in ["src1_greedy", "src2_greedy", "Whole_fixed_HGB", "Fixed_L3_HGB", "Phrase_A_HGB", "SymBiPhrase_HGB", "Listwise_Whole_fixed", "Listwise_dir1", "SymBiList"]}
    for u in all_uids:
        preds["src1_greedy"][u] = pb1[u]["g1"]
        preds["src2_greedy"][u] = pb1[u]["g2"]

    for f in range(6):
        tr = [u for u in fleurs_uids if f == 5 or fs[u]["outer"] != f]
        ev = kathbath_uids if f == 5 else [u for u in fleurs_uids if fs[u]["outer"] == f]
        def inner_id(u): return fs[u]["full"] if f == 5 else fs[u]["inner"][str(f)]

        # Standard 24d HGB models (alpha_c=0.5)
        m1 = fit_dual_regressor("hgb", tr, ft1, meta, col_indices=cols24, seed=1701)
        m2 = fit_dual_regressor("hgb", tr, ft2, meta, col_indices=cols24, seed=1701)
        mF = fit_dual_regressor("hgb", tr, ftF, meta, seed=1701)
        s1 = predict_block_costs(g1, ev, m1, col_indices=cols24)
        s2 = predict_block_costs(g2, ev, m2, col_indices=cols24)
        sF = predict_block_costs(gF, ev, mF)
        ev_A1, ev_A2 = {}, {}
        for u in ev:
            preds["Phrase_A_HGB"][u], preds["Whole_fixed_HGB"][u] = choose_from_block_costs(g1[u], s1[u], pb1[u], alpha_c=0.5, need_whole=True)
            ev_A1[u] = preds["Phrase_A_HGB"][u]
            ev_A2[u], _ = choose_from_block_costs(g2[u], s2[u], pb2[u], alpha_c=0.5)
            preds["Fixed_L3_HGB"][u], _ = choose_from_block_costs(gF[u], sF[u], pb1[u], alpha_c=0.5)

        # Inner 3-fold OOF for symmetric reconcilers
        in_A1, in_A2, in_L1, in_L2 = {}, {}, {}, {}
        for j in range(3):
            itr = [u for u in tr if inner_id(u) != j]
            iva = [u for u in tr if inner_id(u) == j]
            im1 = fit_dual_regressor("hgb", itr, ft1, meta, col_indices=cols24, seed=1701)
            im2 = fit_dual_regressor("hgb", itr, ft2, meta, col_indices=cols24, seed=1701)
            is1 = predict_block_costs(g1, iva, im1, col_indices=cols24)
            is2 = predict_block_costs(g2, iva, im2, col_indices=cols24)
            for u in iva:
                in_A1[u], _ = choose_from_block_costs(g1[u], is1[u], pb1[u], alpha_c=0.5)
                in_A2[u], _ = choose_from_block_costs(g2[u], is2[u], pb2[u], alpha_c=0.5)
            il1, isc1 = fit_listwise_ens(itr, g1, meta, alpha_c=0.5, seed=1701 + j, n_ens=1, epochs=35)
            il2, isc2 = fit_listwise_ens(itr, g2, meta, alpha_c=0.5, seed=1701 + j, n_ens=1, epochs=35)
            p1, _ = pred_listwise_ens(g1, pb1, iva, il1, isc1)
            p2, _ = pred_listwise_ens(g2, pb2, iva, il2, isc2)
            in_L1.update(p1); in_L2.update(p2)

        preds["SymBiPhrase_HGB"].update(fit_and_apply_sym_reconciler(tr, in_A1, in_A2, pb1, ev, ev_A1, ev_A2, pb1, meta, char_lm, word_lm, proj_fn, use_lm=True, alpha_c=0.5))

        ml1, msc1 = fit_listwise_ens(tr, g1, meta, alpha_c=0.5, seed=1701, n_ens=3, epochs=40)
        ml2, msc2 = fit_listwise_ens(tr, g2, meta, alpha_c=0.5, seed=1701, n_ens=3, epochs=40)
        ev_L1, ev_Lwh = pred_listwise_ens(g1, pb1, ev, ml1, msc1, need_whole=True)
        ev_L2, _ = pred_listwise_ens(g2, pb2, ev, ml2, msc2, need_whole=False)
        preds["Listwise_dir1"].update(ev_L1)
        preds["Listwise_Whole_fixed"].update(ev_Lwh)
        preds["SymBiList"].update(fit_and_apply_sym_reconciler(tr, in_L1, in_L2, pb1, ev, ev_L1, ev_L2, pb1, meta, char_lm, word_lm, proj_fn, use_lm=True, alpha_c=0.5))

    fl9_uids = [u for u in fleurs_uids if meta[u]["language"] in WHISPER_NATIVE_9]
    out = {}
    print(f"\n=== 2-Way Pair: {name} ===", flush=True)
    for k, pr in preds.items():
        ef12 = evaluate_predictions(pr, meta, fleurs_uids)
        ef9 = evaluate_predictions(pr, meta, fl9_uids)
        ek = evaluate_predictions(pr, meta, kathbath_uids)
        out[k] = {
            "FLEURS12": {"macro_WER": ef12["macro_WER"], "macro_CER": ef12["macro_CER"]},
            "FLEURS9_whisper_native": {"macro_WER": ef9["macro_WER"], "macro_CER": ef9["macro_CER"]},
            "Kathbath": {"macro_WER": ek["macro_WER"], "macro_CER": ek["macro_CER"]},
        }
        print(f"  {k:22s} | FL12 {ef12['macro_WER']:6.3f}/{ef12['macro_CER']:5.3f} | FL9 {ef9['macro_WER']:6.3f}/{ef9['macro_CER']:5.3f} | Kathb {ek['macro_WER']:6.3f}/{ek['macro_CER']:5.3f}", flush=True)
    # Bootstrap CIs for SymBiList vs Whole_fixed_HGB, Listwise_Whole_fixed, and Phrase_A_HGB
    out["bootstrap"] = {}
    for comp in ["Whole_fixed_HGB", "Listwise_Whole_fixed", "Phrase_A_HGB", "Fixed_L3_HGB", "src1_greedy"]:
        out["bootstrap"][f"SymBiList_vs_{comp}"] = {
            "FLEURS12": paired_cluster_bootstrap_diff(preds["SymBiList"], preds[comp], meta, fleurs_uids, n_draws=10000),
            "FLEURS9": paired_cluster_bootstrap_diff(preds["SymBiList"], preds[comp], meta, fl9_uids, n_draws=10000),
            "Kathbath": paired_cluster_bootstrap_diff(preds["SymBiList"], preds[comp], meta, kathbath_uids, n_draws=10000, stratify_by_language=True),
        }
    return out


def run_3way_fusion(rows_omni, rows_mms, rows_whis, meta, folds, fleurs_uids, kathbath_uids, char_lm, word_lm, norm_fn, proj_fn):
    """3-System Combination (S=3: OmniASR-CTC + MMS-CTC + Whisper-v3-turbo AED)."""
    all_uids = fleurs_uids + kathbath_uids
    pb3_o = {u: build_3way_bundle(rows_omni[u], rows_mms[u], rows_whis[u], 0, norm_fn, proj_fn) for u in all_uids}
    pb3_m = {u: build_3way_bundle(rows_omni[u], rows_mms[u], rows_whis[u], 1, norm_fn, proj_fn) for u in all_uids}
    g3_o = {u: build_3way_graph(pb3_o[u], char_lm, word_lm) for u in all_uids}
    g3_m = {u: build_3way_graph(pb3_m[u], char_lm, word_lm) for u in all_uids}

    fs = folds[17]
    cols30 = list(range(30))
    ft_o = prepare_fit_targets(g3_o, meta, fleurs_uids)
    preds = {k: {} for k in ["3way_Whole_HGB", "3way_Phrase_HGB", "3way_Whole_Listwise", "3way_Listwise_OmniAnc", "3way_Listwise_MMSAnc", "3way_SymBiList"]}
    for f in range(6):
        tr = [u for u in fleurs_uids if f == 5 or fs[u]["outer"] != f]
        ev = kathbath_uids if f == 5 else [u for u in fleurs_uids if fs[u]["outer"] == f]
        def inner_id(u): return fs[u]["full"] if f == 5 else fs[u]["inner"][str(f)]

        mh = fit_dual_regressor("hgb", tr, ft_o, meta, col_indices=cols30, seed=1701)
        sh = predict_block_costs(g3_o, ev, mh, col_indices=cols30)
        for u in ev:
            preds["3way_Phrase_HGB"][u], preds["3way_Whole_HGB"][u] = choose_from_block_costs(g3_o[u], sh[u], pb3_o[u], alpha_c=0.5, need_whole=True)

        # Inner 3-fold OOF for 3-way symmetric Pass-2 reconciler (Omni anchor vs MMS anchor)
        in_Lo, in_Lm = {}, {}
        for j in range(3):
            itr = [u for u in tr if inner_id(u) != j]
            iva = [u for u in tr if inner_id(u) == j]
            ilo, isco = fit_listwise_ens(itr, g3_o, meta, alpha_c=0.5, seed=1701 + j, n_ens=1, epochs=35)
            ilm, iscm = fit_listwise_ens(itr, g3_m, meta, alpha_c=0.5, seed=1701 + j, n_ens=1, epochs=35)
            p_o, _ = pred_listwise_ens(g3_o, pb3_o, iva, ilo, isco)
            p_m, _ = pred_listwise_ens(g3_m, pb3_m, iva, ilm, iscm)
            in_Lo.update(p_o); in_Lm.update(p_m)

        ml_o, sc_o = fit_listwise_ens(tr, g3_o, meta, alpha_c=0.5, seed=1701, n_ens=3, epochs=40)
        ml_m, sc_m = fit_listwise_ens(tr, g3_m, meta, alpha_c=0.5, seed=1701, n_ens=3, epochs=40)
        po, pwh = pred_listwise_ens(g3_o, pb3_o, ev, ml_o, sc_o, need_whole=True)
        pm, _ = pred_listwise_ens(g3_m, pb3_m, ev, ml_m, sc_m, need_whole=False)
        preds["3way_Listwise_OmniAnc"].update(po)
        preds["3way_Whole_Listwise"].update(pwh)
        preds["3way_Listwise_MMSAnc"].update(pm)
        # Reconcile 3-way Omni-anchored and MMS-anchored predictions using 2-way bundle of (Omni, MMS) for Pass-2 features
        pb_pair_tr = {u: build_projection_bundle(rows_omni[u], rows_mms[u], anchor_idx=0, normalize_fn=norm_fn, project_fn=proj_fn) for u in tr}
        pb_pair_ev = {u: build_projection_bundle(rows_omni[u], rows_mms[u], anchor_idx=0, normalize_fn=norm_fn, project_fn=proj_fn) for u in ev}
        preds["3way_SymBiList"].update(fit_and_apply_sym_reconciler(tr, in_Lo, in_Lm, pb_pair_tr, ev, po, pm, pb_pair_ev, meta, char_lm, word_lm, proj_fn, use_lm=True, alpha_c=0.5))

    fl9_uids = [u for u in fleurs_uids if meta[u]["language"] in WHISPER_NATIVE_9]
    out = {}
    print("\n=== 3-System Combination (OmniASR-CTC + MMS-CTC + Whisper-v3-turbo AED) ===", flush=True)
    for k, pr in preds.items():
        ef12 = evaluate_predictions(pr, meta, fleurs_uids)
        ef9 = evaluate_predictions(pr, meta, fl9_uids)
        ek = evaluate_predictions(pr, meta, kathbath_uids)
        out[k] = {
            "FLEURS12": {"macro_WER": ef12["macro_WER"], "macro_CER": ef12["macro_CER"]},
            "FLEURS9_whisper_native": {"macro_WER": ef9["macro_WER"], "macro_CER": ef9["macro_CER"]},
            "Kathbath": {"macro_WER": ek["macro_WER"], "macro_CER": ek["macro_CER"]},
        }
        print(f"  {k:24s} | FL12 {ef12['macro_WER']:6.3f}/{ef12['macro_CER']:5.3f} | FL9 {ef9['macro_WER']:6.3f}/{ef9['macro_CER']:5.3f} | Kathb {ek['macro_WER']:6.3f}/{ek['macro_CER']:5.3f}", flush=True)
    out["bootstrap"] = {}
    for comp in ["3way_Whole_HGB", "3way_Whole_Listwise", "3way_Phrase_HGB"]:
        out["bootstrap"][f"3way_SymBiList_vs_{comp}"] = {
            "FLEURS12": paired_cluster_bootstrap_diff(preds["3way_SymBiList"], preds[comp], meta, fleurs_uids, n_draws=10000),
            "FLEURS9": paired_cluster_bootstrap_diff(preds["3way_SymBiList"], preds[comp], meta, fl9_uids, n_draws=10000),
            "Kathbath": paired_cluster_bootstrap_diff(preds["3way_SymBiList"], preds[comp], meta, kathbath_uids, n_draws=10000, stratify_by_language=True),
        }
    return out


def main():
    t0 = time.monotonic()
    bundle = load_frozen_dev_bundle()
    meta = bundle["meta"]
    folds = bundle["folds_by_seed"]
    fleurs_uids = bundle["fleurs_uids"]
    kathbath_uids = bundle["kathbath_uids"]
    all_uids = fleurs_uids + kathbath_uids
    char_lm = bundle["char_lm_fn"]
    word_lm = bundle["word_lm"]
    norm_fn = bundle["normalize_fn"]
    proj_fn = make_projector(bundle["align_fn"])

    whis_fl = {r["uid"]: r for r in (json.loads(l) for l in (ROOT / "cache/whisper_v3t_fleurs_4078.jsonl").open())}
    whis_kb = {r["uid"]: r for r in (json.loads(l) for l in (ROOT / "cache/whisper_v3t_kathbath_4920.jsonl").open())}
    assert set(whis_fl) == set(fleurs_uids) and set(whis_kb) == set(kathbath_uids)

    rows_omni = {u: bundle["all_data"][u][0] for u in all_uids}
    rows_mms = {u: bundle["all_data"][u][1] for u in all_uids}
    rows_whis = whis_fl | whis_kb

    results = {}
    results["pair_Omni_Whisper"] = run_2way_pair("OmniASR-CTC + Whisper-v3-turbo (AED)", rows_omni, rows_whis, meta, folds, fleurs_uids, kathbath_uids, char_lm, word_lm, norm_fn, proj_fn)
    results["pair_MMS_Whisper"] = run_2way_pair("MMS-CTC + Whisper-v3-turbo (AED)", rows_mms, rows_whis, meta, folds, fleurs_uids, kathbath_uids, char_lm, word_lm, norm_fn, proj_fn)
    results["three_way_Omni_MMS_Whisper"] = run_3way_fusion(rows_omni, rows_mms, rows_whis, meta, folds, fleurs_uids, kathbath_uids, char_lm, word_lm, norm_fn, proj_fn)

    p_out = ROOT / "results/stage2_fleurs_kb_cross_arch_3way.json"
    p_out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n[{time.monotonic()-t0:.1f}s] Saved Stage 2A results to {p_out}", flush=True)


if __name__ == "__main__":
    main()
