"""Stage 1 Step 2: Bit-exact baseline verification + Zero-inference strong baselines + Complete feature/position/T/sample ablations."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agree_phrase_asr.data_bridge import load_frozen_dev_bundle, SEEDS
from agree_phrase_asr.align import build_projection_bundle, make_projector
from agree_phrase_asr.partition import (
    agreement_partition,
    random_varcount_matched_partition,
    shifted_agreement_partition,
    validate_partition,
)
from agree_phrase_asr.features import BASE_FEATURE_NAMES, FEATURE_GROUPS, build_phrase_graph
from agree_phrase_asr.eval import evaluate_predictions, paired_cluster_bootstrap_diff
from agree_phrase_asr.models import (
    run_grouped_oof,
    predict_exact_nbest_mbr,
    predict_unsupervised_block_mbr,
    predict_alignment_cell_vote,
    predict_block_costs,
    choose_from_block_costs,
)


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

    print(f"[{time.monotonic()-t0:.1f}s] Loaded frozen dev bundle: FLEURS={len(fleurs_uids)}, Kathbath={len(kathbath_uids)}", flush=True)

    # 1. Build standard baseline A projection bundles & graphs
    proj_bundles = {
        u: build_projection_bundle(all_data[u][0], all_data[u][1], anchor_idx=0, mode="weighted", temperature=0.5, normalize_fn=norm_fn, project_fn=proj_fn)
        for u in all_uids
    }
    graphs_A = {}
    for u in all_uids:
        part = agreement_partition(proj_bundles[u])
        validate_partition(proj_bundles[u], part)
        graphs_A[u] = build_phrase_graph(proj_bundles[u], part, char_lm, word_lm)

    print(f"[{time.monotonic()-t0:.1f}s] Built baseline A graphs.", flush=True)

    # 2. Verify BIT-EXACT reproduction of frozen baseline A!
    res_A = run_grouped_oof(graphs_A, proj_bundles, meta, folds, fleurs_uids, kathbath_uids, need_whole=True)
    exact_matches = {}
    for seed in SEEDS:
        bp = bundle["baseline_preds"][seed]
        m_a = sum(res_A["preds"][seed][u] == bp[u]["CTC_phrase"] for u in all_uids)
        m_b = sum(res_A["whole_preds"][seed][u] == bp[u]["CTC_whole_fixed"] for u in all_uids)
        exact_matches[str(seed)] = {"A_match": m_a, "B_match": m_b, "total": len(all_uids)}
    print("Bit-exact baseline check:", exact_matches, flush=True)
    print(f"Baseline A FLEURS:   WER={res_A['summary']['FLEURS']['macro_WER_mean']:.3f}±{res_A['summary']['FLEURS']['macro_WER_sd']:.3f}  CER={res_A['summary']['FLEURS']['macro_CER_mean']:.3f}±{res_A['summary']['FLEURS']['macro_CER_sd']:.3f}")
    print(f"Baseline A Kathbath: WER={res_A['summary']['Kathbath']['macro_WER_mean']:.3f}±{res_A['summary']['Kathbath']['macro_WER_sd']:.3f}  CER={res_A['summary']['Kathbath']['macro_CER_mean']:.3f}±{res_A['summary']['Kathbath']['macro_CER_sd']:.3f}", flush=True)

    results = {
        "bit_exact_check": exact_matches,
        "baseline_A": res_A["summary"],
        "unsupervised_baselines": {},
        "feature_ablations": {},
        "permutation_importance": {},
        "boundary_position_controls": {},
        "temperature_sweep": {},
        "sample_efficiency": {},
    }

    # 3. Unsupervised Strong Baselines (Exact Multi-Source N-best MBR, Block MBR, Alignment Cell Vote)
    unsup_configs = [
        ("MBR_word", lambda: predict_exact_nbest_mbr(proj_bundles, all_uids, metric="word")),
        ("MBR_char", lambda: predict_exact_nbest_mbr(proj_bundles, all_uids, metric="char")),
        ("MBR_dual", lambda: predict_exact_nbest_mbr(proj_bundles, all_uids, metric="dual")),
        ("MBR_word_charLM0.05", lambda: predict_exact_nbest_mbr(proj_bundles, all_uids, metric="word", char_lm_fn=char_lm, lm_weight=0.05)),
        ("MBR_word_charLM0.10", lambda: predict_exact_nbest_mbr(proj_bundles, all_uids, metric="word", char_lm_fn=char_lm, lm_weight=0.10)),
        ("MBR_word_charLM0.20", lambda: predict_exact_nbest_mbr(proj_bundles, all_uids, metric="word", char_lm_fn=char_lm, lm_weight=0.20)),
        ("Block_MBR_word", lambda: predict_unsupervised_block_mbr(graphs_A, all_uids, metric="word")),
        ("Block_MBR_char", lambda: predict_unsupervised_block_mbr(graphs_A, all_uids, metric="char")),
        ("Align_Cell_Vote", lambda: predict_alignment_cell_vote(proj_bundles, all_uids)),
    ]
    print("\n=== Unsupervised Baselines ===")
    for name, fn in unsup_configs:
        pr = fn()
        ev_f = evaluate_predictions(pr, meta, fleurs_uids)
        ev_k = evaluate_predictions(pr, meta, kathbath_uids)
        results["unsupervised_baselines"][name] = {
            "FLEURS": {"macro_WER": ev_f["macro_WER"], "macro_CER": ev_f["macro_CER"]},
            "Kathbath": {"macro_WER": ev_k["macro_WER"], "macro_CER": ev_k["macro_CER"]},
        }
        print(f"  {name:22s} | FLEURS {ev_f['macro_WER']:6.3f} / {ev_f['macro_CER']:5.3f} | Kathbath {ev_k['macro_WER']:6.3f} / {ev_k['macro_CER']:5.3f}", flush=True)

    # 4. Complete 4-Family Feature Ablations (Leave-One-Group-Out + Single-Group-Only)
    all_cols = list(range(len(BASE_FEATURE_NAMES)))
    feat_exps = {
        "wo_geom": [c for c in all_cols if c not in FEATURE_GROUPS["geom"]],
        "wo_char_lm": [c for c in all_cols if c not in FEATURE_GROUPS["char_lm"]],
        "wo_word_lm": [c for c in all_cols if c not in FEATURE_GROUPS["word_lm"]],
        "wo_all_lm": [c for c in all_cols if c not in FEATURE_GROUPS["all_lm"]],
        "wo_src_dist": [c for c in all_cols if c not in FEATURE_GROUPS["src_dist"]],
        "wo_src_risk_eq4": [c for c in all_cols if c not in FEATURE_GROUPS["src_risk_eq4"]],
        "only_lm_plus_geom": [c for c in all_cols if c not in FEATURE_GROUPS["src_dist"]],
        "only_src_dist": FEATURE_GROUPS["src_dist"],
        "only_all_lm": FEATURE_GROUPS["all_lm"],
    }
    print("\n=== Complete Feature-Group Ablations ===")
    for name, cols in feat_exps.items():
        if name == "only_lm_plus_geom":
            continue  # identical to wo_src_dist
        r = run_grouped_oof(graphs_A, proj_bundles, meta, folds, fleurs_uids, kathbath_uids, col_indices=cols)
        results["feature_ablations"][name] = r["summary"]
        sf, sk = r["summary"]["FLEURS"], r["summary"]["Kathbath"]
        print(f"  {name:20s} ({len(cols):2d}d) | FLEURS {sf['macro_WER_mean']:6.3f}±{sf['macro_WER_sd']:.3f} / {sf['macro_CER_mean']:5.3f} | Kathbath {sk['macro_WER_mean']:6.3f} / {sk['macro_CER_mean']:5.3f}", flush=True)

    # 5. Permutation Feature Importance (using seed 17 outer models on FLEURS OOF & full model on Kathbath)
    print("\n=== Permutation Feature Importance (WER increase in pp when feature permuted) ===")
    rng = np.random.Generator(np.random.PCG64(20261123))
    seed0 = 17
    fs0 = folds[seed0]
    for ds_name, ds_uids in [("FLEURS", fleurs_uids), ("Kathbath", kathbath_uids)]:
        base_wer = evaluate_predictions(res_A["preds"][seed0], meta, ds_uids)["macro_WER"]
        base_cer = evaluate_predictions(res_A["preds"][seed0], meta, ds_uids)["macro_CER"]
        # Pre-stack X per fold
        fold_map = {f: (ds_uids if f == 5 else [u for u in ds_uids if fs0[u]["outer"] == f]) for f in ([5] if ds_name == "Kathbath" else range(5))}
        imp = {}
        for col_idx, fname in enumerate(BASE_FEATURE_NAMES):
            dw_list, dc_list = [], []
            for rep in range(3):
                perm_preds = {}
                for f, fuids in fold_map.items():
                    mod = res_A["models"][seed0][f]["models"]
                    alpha = res_A["models"][seed0][f]["alpha"]
                    all_x = [bl["X"] for u in fuids for bl in graphs_A[u][1]]
                    stacked = np.vstack(all_x).copy()
                    rng.shuffle(stacked[:, col_idx])
                    pred = np.stack([m.predict(stacked) for m in mod], axis=1)
                    pos = 0
                    for u in fuids:
                        zz = []
                        for bl in graphs_A[u][1]:
                            n = len(bl["X"])
                            zz.append(np.vstack((np.zeros((1, 2)), pred[pos : pos + n])))
                            pos += n
                        perm_preds[u], _ = choose_from_block_costs(graphs_A[u], zz, proj_bundles[u], alpha_c=alpha)
                ev = evaluate_predictions(perm_preds, meta, ds_uids)
                dw_list.append(ev["macro_WER"] - base_wer)
                dc_list.append(ev["macro_CER"] - base_cer)
            imp[fname] = {"delta_WER_pp": float(np.mean(dw_list)), "delta_CER_pp": float(np.mean(dc_list))}
        results["permutation_importance"][ds_name] = imp
        top5 = sorted(imp.items(), key=lambda kv: -kv[1]["delta_WER_pp"])[:8]
        print(f"  Top-8 ({ds_name}): " + ", ".join(f"{k}:{v['delta_WER_pp']:+.3f}" for k, v in top5), flush=True)

    # 6. Pure Boundary Position Controls: Q_var (variable-block-count matched) & Q_shift (shifted agreement cuts)
    print("\n=== Pure Boundary Position Controls (Q_var & Q_shift) ===")
    for cname, part_fn_factory in [
        ("Q_var_3seeds", lambda s: (lambda b, u: random_varcount_matched_partition(b, u, s))),
        ("Q_shift2_3seeds", lambda s: (lambda b, u: shifted_agreement_partition(b, u, s, shift=2))),
    ]:
        layout_seeds = (101, 211, 307)
        sub_res = {}
        f_wers, f_cers, k_wers, k_cers, f_vblocks = [], [], [], [], []
        for ls in layout_seeds:
            pfn = part_fn_factory(ls)
            g_ctrl = {}
            vbc = []
            for u in all_uids:
                part = pfn(proj_bundles[u], u)
                validate_partition(proj_bundles[u], part)
                g_ctrl[u] = build_phrase_graph(proj_bundles[u], part, char_lm, word_lm)
                if u in bundle["fleurs"]:
                    vbc.append(len(g_ctrl[u][1]))
            r = run_grouped_oof(g_ctrl, proj_bundles, meta, folds, fleurs_uids, kathbath_uids)
            sub_res[str(ls)] = r["summary"]
            f_wers.append(r["summary"]["FLEURS"]["macro_WER_mean"])
            f_cers.append(r["summary"]["FLEURS"]["macro_CER_mean"])
            k_wers.append(r["summary"]["Kathbath"]["macro_WER_mean"])
            k_cers.append(r["summary"]["Kathbath"]["macro_CER_mean"])
            f_vblocks.append(float(np.mean(vbc)))
        agg = {
            "FLEURS_WER": float(np.mean(f_wers)),
            "FLEURS_CER": float(np.mean(f_cers)),
            "Kathbath_WER": float(np.mean(k_wers)),
            "Kathbath_CER": float(np.mean(k_cers)),
            "FLEURS_mean_var_blocks": float(np.mean(f_vblocks)),
            "by_layout": sub_res,
        }
        results["boundary_position_controls"][cname] = agg
        print(f"  {cname:18s} (var_blocks={agg['FLEURS_mean_var_blocks']:.2f}) | FLEURS {agg['FLEURS_WER']:6.3f} / {agg['FLEURS_CER']:5.3f} | Kathbath {agg['Kathbath_WER']:6.3f} / {agg['Kathbath_CER']:5.3f}", flush=True)

    # 7. Temperature T sensitivity sweep
    print("\n=== Temperature T Sweep ===")
    for T in [0.25, 0.5, 1.0, 2.0, float("inf")]:
        tname = "inf_uniform" if np.isinf(T) else f"T_{T}"
        if T == 0.5:
            results["temperature_sweep"][tname] = res_A["summary"]
            sf, sk = res_A["summary"]["FLEURS"], res_A["summary"]["Kathbath"]
        else:
            pb_t = {
                u: build_projection_bundle(all_data[u][0], all_data[u][1], anchor_idx=0, mode="uniform" if np.isinf(T) else "weighted", temperature=T, normalize_fn=norm_fn, project_fn=proj_fn)
                for u in all_uids
            }
            g_t = {u: build_phrase_graph(pb_t[u], agreement_partition(pb_t[u]), char_lm, word_lm) for u in all_uids}
            r_t = run_grouped_oof(g_t, pb_t, meta, folds, fleurs_uids, kathbath_uids)
            results["temperature_sweep"][tname] = r_t["summary"]
            sf, sk = r_t["summary"]["FLEURS"], r_t["summary"]["Kathbath"]
        print(f"  {tname:12s} | FLEURS {sf['macro_WER_mean']:6.3f}±{sf['macro_WER_sd']:.3f} / {sf['macro_CER_mean']:5.3f} | Kathbath {sk['macro_WER_mean']:6.3f} / {sk['macro_CER_mean']:5.3f}", flush=True)

    # 8. Sample Efficiency Curve (10%, 25%, 50%, 100% training prompt groups)
    print("\n=== Sample Efficiency Curve ===")
    for frac in [0.10, 0.25, 0.50, 1.00]:
        fname = f"frac_{int(frac*100)}"
        if frac == 1.0:
            results["sample_efficiency"][fname] = res_A["summary"]
            sf, sk = res_A["summary"]["FLEURS"], res_A["summary"]["Kathbath"]
        else:
            r_s = run_grouped_oof(graphs_A, proj_bundles, meta, folds, fleurs_uids, kathbath_uids, train_frac=frac)
            results["sample_efficiency"][fname] = r_s["summary"]
            sf, sk = r_s["summary"]["FLEURS"], r_s["summary"]["Kathbath"]
        print(f"  {fname:10s} | FLEURS {sf['macro_WER_mean']:6.3f}±{sf['macro_WER_sd']:.3f} / {sf['macro_CER_mean']:5.3f} | Kathbath {sk['macro_WER_mean']:6.3f} / {sk['macro_CER_mean']:5.3f}", flush=True)

    out_path = ROOT / "results" / "stage1_baselines_and_ablations.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n[{time.monotonic()-t0:.1f}s] Saved complete Stage 1 baselines & ablations to {out_path}", flush=True)


if __name__ == "__main__":
    main()
