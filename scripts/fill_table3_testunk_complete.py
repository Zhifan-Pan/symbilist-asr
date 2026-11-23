#!/usr/bin/env python3
"""Fill all 11 remaining Table III rows on Kathbath clean/testunk (3,030 unseen-speaker utterances).

Strictly read-only on frozen evidence; zero fitting or hyperparameter selection on testunk.
Computes:
  1. Hystoc-ROVER (official hystoc + NIST SCTK rover, T=1, redirected to upgrade tmp dir)
  2. Exact MBR (word, whole)
  3. Block MBR (word, agree)
  4. Exact MBR + Char4 LM (lambda=0.20)
  5. Whole (fixed Listwise) from frozen_symbilist_fleurs12.pkl
  6. Fixed length (F, L=3) from frozen full-FLEURS-12 HGB fit
  7. Random partition (Q, mean of 5 layouts 101/211/307/401/503) from frozen full-FLEURS-12 HGB fits
  8. Agree Phrase A0 (MMS anchor, HGB 24d) fit on full FLEURS-12
  9. SymBiPhrase (HGB, 24d) fit on full FLEURS-12
 10. Whole list oracle (H_whole) exact min word/char edits
 11. Agreement phrase oracle (H_agree) exact DP min word/char edits (same mechanism.oracle as FLEURS/KB valid)
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
import tarfile
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from agree_phrase_asr.data_bridge import load_frozen_dev_bundle, FROZEN_BASE
from agree_phrase_asr.align import build_projection_bundle, make_projector, flatten_cells, splice_cells, lev_dist
from agree_phrase_asr.partition import agreement_partition, fixed_partition, random_length_matched_partition
from agree_phrase_asr.features import build_phrase_graph
from agree_phrase_asr.eval import evaluate_predictions
from agree_phrase_asr.models import (
    prepare_fit_targets,
    fit_dual_regressor,
    predict_block_costs,
    choose_from_block_costs,
    predict_exact_nbest_mbr,
    predict_unsupervised_block_mbr,
)
from run_stage1_symbilist import ListwiseBlockNet, predict_listwise_ensemble
from run_stage2_cv26 import fit_and_apply_sym_reconciler

# Import official oracle DP helper read-only
sys.path.insert(0, str(FROZEN_BASE / "source_frozen/dev_acceptance_20261118/scripts"))
from mechanism import oracle as dp_oracle


def run_official_hystoc_rover(omni, mms, uids, tmp_dir: Path) -> dict[str, str]:
    """Run exact official Hystoc + NIST SCTK ROVER (T=1) redirected to tmp_dir."""
    rp = FROZEN_BASE / "rover"
    sys.path.insert(0, str(rp))
    import decode as official

    tmp_dir.mkdir(parents=True, exist_ok=True)
    (tmp_dir / "data").mkdir(parents=True, exist_ok=True)
    (tmp_dir / "logs").mkdir(parents=True, exist_ok=True)
    official.HERE = tmp_dir
    official.DATA = tmp_dir / "data"

    pairs = {u: (omni[u], mms[u]) for u in uids}
    mapping = [{"uid": u, "seg": f"u{i:05d}"} for i, u in enumerate(uids)]
    for j, name in enumerate(("omni", "mms")):
        masses = {r["seg"]: official.collapsed(pairs[r["uid"]][j])[1] for r in mapping}
        ht = tmp_dir / "data" / f"{name}.txt"
        hs = ht.with_suffix(".score")
        with ht.open("w") as f, hs.open("w") as v:
            for seg, d in masses.items():
                for idx, (h, score) in enumerate(d.items()):
                    f.write(f"{seg}-{idx} {h}\n")
                    v.write(f"{seg}-{idx} {score!r}\n")
        official.invoke_hystoc((name, 1))
    official.fuse_temperature(1, mapping)
    import gzip
    rhy = {}
    with gzip.open(tmp_dir / "data/fused_T1.jsonl.gz", "rt") as f:
        for line in f:
            row = json.loads(line)
            rhy[row["uid"]] = row["prediction"]
    return rhy


def main():
    t0 = time.monotonic()
    bundle = load_frozen_dev_bundle()
    meta_dev = bundle["meta"]
    fleurs_uids = bundle["fleurs_uids"]
    folds = bundle["folds_by_seed"][17]
    char_lm = bundle["char_lm_fn"]
    word_lm = bundle["word_lm"]
    norm_fn = bundle["normalize_fn"]
    proj_fn = make_projector(bundle["align_fn"])

    # Load Kathbath clean/testunk (3,030 utterances)
    kb_dir = FROZEN_BASE / "source_frozen/kathbath_confirm_20261120"
    roster = [json.loads(l) for l in (kb_dir / "inputs/roster.jsonl").open()]
    omni_raw = {r["recording_uid"]: r for r in (json.loads(l) for l in (kb_dir / "native/omni.jsonl").open())}
    mms_raw = {r["recording_uid"]: r for r in (json.loads(l) for l in (kb_dir / "native/mms.jsonl").open())}
    sealed = {r["uid"]: r["hypotheses"] for r in (json.loads(l) for l in (kb_dir / "sealed/predictions.jsonl").open())}

    tar_p = Path(os.environ["SYMBILIST_KATHBATH_TRANSCRIPTS"])
    refs = {}
    with tarfile.open(tar_p, "r:") as t:
        for lang in ["tamil", "kannada"]:
            with t.extractfile(f"kb_data_clean_m4a/{lang}/test/transcription_n2w.txt") as fh:
                for line in fh:
                    fn, txt = line.rstrip(b"\r\n").split(b"\t", 1)
                    u = f"kb_{lang}_{fn.decode('ascii')[:-4]}"
                    refs[u] = norm_fn(txt.decode("utf-8"))

    omni, mms, meta_test = {}, {}, {}
    uids = []
    for r in roster:
        u = r["uid"]
        opaque = hashlib.sha256(u.encode()).hexdigest()
        omni[u] = omni_raw[opaque]
        mms[u] = mms_raw[opaque]
        uids.append(u)
        meta_test[u] = {
            "dataset": "Kathbath_testunk",
            "language": r["language"],
            "cluster": f"{r['language']}:{r['speaker_id']}",
            "reference": refs[u],
        }
    uids = sorted(uids)
    print(f"Loaded testunk: {len(uids)} utterances", flush=True)

    results = {}

    def record(name: str, preds: dict[str, str]):
        ev = evaluate_predictions(preds, meta_test, uids)
        results[name] = {
            "macro_WER": ev["macro_WER"],
            "macro_CER": ev["macro_CER"],
            "per_lang": {lg: {"WER": v["WER"], "CER": v["CER"]} for lg, v in ev["per_lang"].items()},
        }
        print(f"  {name:26s} | WER={ev['macro_WER']:6.3f}  CER={ev['macro_CER']:5.3f}", flush=True)

    # 1. Official Hystoc-ROVER (T=1)
    print("[1/7] Running official Hystoc-ROVER (T=1)...", flush=True)
    rover_preds = run_official_hystoc_rover(omni, mms, uids, ROOT / "tmp_testunk_rover")
    record("Hystoc_ROVER", rover_preds)

    # Build testunk projection bundles
    pb1_te = {u: build_projection_bundle(omni[u], mms[u], anchor_idx=0, normalize_fn=norm_fn, project_fn=proj_fn) for u in uids}
    pb2_te = {u: build_projection_bundle(omni[u], mms[u], anchor_idx=1, normalize_fn=norm_fn, project_fn=proj_fn) for u in uids}

    # 2. Unsupervised MBR baselines & Whole (fixed Listwise)
    print("[2/7] Computing Exact MBR, Block MBR, and Exact MBR + Char4 LM...", flush=True)
    g1_te_ctx = {u: build_phrase_graph(pb1_te[u], agreement_partition(pb1_te[u]), char_lm, word_lm, add_block_context_feats=True) for u in uids}
    record("Exact_MBR_word", predict_exact_nbest_mbr(pb1_te, uids, metric="word"))
    record("Block_MBR_word", predict_unsupervised_block_mbr(g1_te_ctx, uids, metric="word"))
    record("Exact_MBR_Char4LM", predict_exact_nbest_mbr(pb1_te, uids, metric="word", char_lm_fn=char_lm, lm_weight=0.20))

    # 3. Whole (fixed Listwise) from frozen FLEURS-12 SymBiList checkpoint
    print("[3/7] Computing Whole (fixed Listwise) from frozen FLEURS-12 checkpoint...", flush=True)
    ckpt = pickle.loads((ROOT / "results/frozen_symbilist_fleurs12.pkl").read_bytes())
    D = ckpt["sc1"].n_features_in_
    m1_list = []
    for sd in ckpt["m1_state"]:
        m = ListwiseBlockNet(D, hidden=48)
        m.load_state_dict(sd)
        m.eval()
        m1_list.append(m)
    _, wh1_map = predict_listwise_ensemble(g1_te_ctx, pb1_te, uids, m1_list, ckpt["sc1"], need_whole=True)
    record("Whole_fixed_Listwise", wh1_map)

    # 4. Fixed length (F, L=3) and Random partition (Q, 5 seeds) fit on full FLEURS-12 (alpha_c=0.5 matching frozen A/F/Q)
    print("[4/7] Fitting full-FLEURS-12 Fixed (F, L=3) and Random (Q, 5 seeds) and scoring testunk...", flush=True)
    pb1_fl = {u: build_projection_bundle(bundle["all_data"][u][0], bundle["all_data"][u][1], anchor_idx=0, normalize_fn=norm_fn, project_fn=proj_fn) for u in fleurs_uids}
    pb2_fl = {u: build_projection_bundle(bundle["all_data"][u][0], bundle["all_data"][u][1], anchor_idx=1, normalize_fn=norm_fn, project_fn=proj_fn) for u in fleurs_uids}

    gF_fl = {u: build_phrase_graph(pb1_fl[u], fixed_partition(pb1_fl[u], 3), char_lm, word_lm) for u in fleurs_uids}
    gF_te = {u: build_phrase_graph(pb1_te[u], fixed_partition(pb1_te[u], 3), char_lm, word_lm) for u in uids}
    ftF = prepare_fit_targets(gF_fl, meta_dev, fleurs_uids)
    mF = fit_dual_regressor("hgb", fleurs_uids, ftF, meta_dev, seed=1701)
    sF = predict_block_costs(gF_te, uids, mF)
    record("Fixed_L3", {u: choose_from_block_costs(gF_te[u], sF[u], pb1_te[u], alpha_c=0.5)[0] for u in uids})

    q_seeds = [101, 211, 307, 401, 503]
    q_wers, q_cers = [], []
    q_kn_w, q_kn_c, q_ta_w, q_ta_c = [], [], [], []
    for qs in q_seeds:
        gQ_fl = {u: build_phrase_graph(pb1_fl[u], random_length_matched_partition(pb1_fl[u], u, qs), char_lm, word_lm) for u in fleurs_uids}
        gQ_te = {u: build_phrase_graph(pb1_te[u], random_length_matched_partition(pb1_te[u], u, qs), char_lm, word_lm) for u in uids}
        ftQ = prepare_fit_targets(gQ_fl, meta_dev, fleurs_uids)
        mQ = fit_dual_regressor("hgb", fleurs_uids, ftQ, meta_dev, seed=1701)
        sQ = predict_block_costs(gQ_te, uids, mQ)
        prQ = {u: choose_from_block_costs(gQ_te[u], sQ[u], pb1_te[u], alpha_c=0.5)[0] for u in uids}
        evQ = evaluate_predictions(prQ, meta_test, uids)
        q_wers.append(evQ["macro_WER"]); q_cers.append(evQ["macro_CER"])
        q_kn_w.append(evQ["per_lang"]["kn_in"]["WER"]); q_kn_c.append(evQ["per_lang"]["kn_in"]["CER"])
        q_ta_w.append(evQ["per_lang"]["ta_in"]["WER"]); q_ta_c.append(evQ["per_lang"]["ta_in"]["CER"])
    results["Random_Q_5seeds"] = {
        "macro_WER": float(np.mean(q_wers)),
        "macro_CER": float(np.mean(q_cers)),
        "per_lang": {
            "kn_in": {"WER": float(np.mean(q_kn_w)), "CER": float(np.mean(q_kn_c))},
            "ta_in": {"WER": float(np.mean(q_ta_w)), "CER": float(np.mean(q_ta_c))},
        },
        "by_layout_WER": q_wers,
        "by_layout_CER": q_cers,
    }
    print(f"  {'Random_Q_5seeds':26s} | WER={np.mean(q_wers):6.3f}  CER={np.mean(q_cers):5.3f}", flush=True)

    # 5. Agree Phrase A0 (MMS anchor, HGB 24d) & SymBiPhrase (HGB 24d)
    print("[5/7] Fitting full-FLEURS-12 Agree A0 (MMS anc.) and SymBiPhrase (HGB 24d)...", flush=True)
    g1_fl = {u: build_phrase_graph(pb1_fl[u], agreement_partition(pb1_fl[u]), char_lm, word_lm) for u in fleurs_uids}
    g2_fl = {u: build_phrase_graph(pb2_fl[u], agreement_partition(pb2_fl[u]), char_lm, word_lm) for u in fleurs_uids}
    g1_te = {u: build_phrase_graph(pb1_te[u], agreement_partition(pb1_te[u]), char_lm, word_lm) for u in uids}
    g2_te = {u: build_phrase_graph(pb2_te[u], agreement_partition(pb2_te[u]), char_lm, word_lm) for u in uids}
    ft1 = prepare_fit_targets(g1_fl, meta_dev, fleurs_uids)
    ft2 = prepare_fit_targets(g2_fl, meta_dev, fleurs_uids)
    m1 = fit_dual_regressor("hgb", fleurs_uids, ft1, meta_dev, seed=1701)
    m2 = fit_dual_regressor("hgb", fleurs_uids, ft2, meta_dev, seed=1701)
    s1 = predict_block_costs(g1_te, uids, m1)
    s2 = predict_block_costs(g2_te, uids, m2)
    te_A1 = {u: choose_from_block_costs(g1_te[u], s1[u], pb1_te[u], alpha_c=0.5)[0] for u in uids}
    te_A2 = {u: choose_from_block_costs(g2_te[u], s2[u], pb2_te[u], alpha_c=0.5)[0] for u in uids}
    record("Agree_A_check", te_A1)
    record("Agree_A0_MMS_anc", te_A2)

    # Inner 3-fold OOF on FLEURS-12 for Pass-2 SymBiPhrase reconciler
    in_A1, in_A2 = {}, {}
    for j in range(3):
        itr = [u for u in fleurs_uids if folds[u]["full"] != j]
        iva = [u for u in fleurs_uids if folds[u]["full"] == j]
        im1 = fit_dual_regressor("hgb", itr, ft1, meta_dev, seed=1701)
        im2 = fit_dual_regressor("hgb", itr, ft2, meta_dev, seed=1701)
        is1 = predict_block_costs(g1_fl, iva, im1)
        is2 = predict_block_costs(g2_fl, iva, im2)
        for u in iva:
            in_A1[u], _ = choose_from_block_costs(g1_fl[u], is1[u], pb1_fl[u], alpha_c=0.5)
            in_A2[u], _ = choose_from_block_costs(g2_fl[u], is2[u], pb2_fl[u], alpha_c=0.5)
    te_sym = fit_and_apply_sym_reconciler(fleurs_uids, in_A1, in_A2, pb1_fl, uids, te_A1, te_A2, pb1_te, meta_dev, char_lm, word_lm, proj_fn, use_lm=True, alpha_c=0.5)
    record("SymBiPhrase_HGB_24d", te_sym)

    # 6. Exact Oracles (Whole list oracle & Agreement phrase oracle via official DP)
    print("[6/7] Computing exact DP Whole and Agreement Phrase Oracles on testunk...", flush=True)
    for space_name in ["Oracle_Whole", "Oracle_Agree"]:
        by_lang = {"kn_in": [0, 0, 0, 0], "ta_in": [0, 0, 0, 0]}
        for u in uids:
            ref = meta_test[u]["reference"]
            lg = meta_test[u]["language"]
            pb = pb1_te[u]
            if space_name == "Oracle_Whole":
                ss = [[tuple(h.split()) for h in pb["texts"]]]
            else:
                rr = agreement_partition(pb)
                ss = [list(dict.fromkeys(flatten_cells(z[i:j]) for z in pb["proj"].values())) for i, j in rr]
            we, _ = dp_oracle(ss, ref, False)
            ce, _ = dp_oracle(ss, ref, True)
            by_lang[lg][0] += int(we)
            by_lang[lg][1] += len(ref.split())
            by_lang[lg][2] += int(ce)
            by_lang[lg][3] += len(ref.replace(" ", ""))
        per_l = {lg: {"WER": 100.0 * v[0] / v[1], "CER": 100.0 * v[2] / v[3]} for lg, v in by_lang.items()}
        mw = float(np.mean([x["WER"] for x in per_l.values()]))
        mc = float(np.mean([x["CER"] for x in per_l.values()]))
        results[space_name] = {"macro_WER": mw, "macro_CER": mc, "per_lang": per_l}
        print(f"  {space_name:26s} | WER={mw:6.3f}  CER={mc:5.3f}", flush=True)

    p_out = ROOT / "results/testunk_complete_table3.json"
    p_out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"[7/7] Completed in {time.monotonic()-t0:.1f}s -> {p_out}", flush=True)


if __name__ == "__main__":
    main()
