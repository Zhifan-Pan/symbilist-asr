"""Post-lock evaluation of FROZEN FLEURS-12 SymBiList on Kathbath clean/testunk (3,030 unseen-speaker utterances).

Strictly read-only on frozen artifacts; zero fitting or selection on testunk.
Explicitly disclosed as post-lock evaluation alongside the original locked 4-endpoint family.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from agree_phrase_asr.data_bridge import load_frozen_dev_bundle, FROZEN_BASE
from agree_phrase_asr.align import build_projection_bundle, make_projector, flatten_cells, splice_cells
from agree_phrase_asr.partition import agreement_partition
from agree_phrase_asr.features import build_phrase_graph
from agree_phrase_asr.eval import evaluate_predictions, paired_cluster_bootstrap_diff
from run_stage1_symbilist import ListwiseBlockNet, predict_listwise_ensemble, symmetric_sentence_feats


def main():
    t0 = time.monotonic()
    bundle = load_frozen_dev_bundle()
    char_lm = bundle["char_lm_fn"]
    word_lm = bundle["word_lm"]
    norm_fn = bundle["normalize_fn"]
    proj_fn = make_projector(bundle["align_fn"])

    import hashlib, tarfile
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
        assert opaque in omni_raw and opaque in mms_raw and u in sealed and u in refs
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
    print(f"Loaded Kathbath clean/testunk: {len(uids)} utterances", flush=True)

    # Verify locked baseline numbers from sealed predictions
    sealed_keys = list(sealed[uids[0]].keys())
    print("Sealed systems:", sealed_keys, flush=True)
    sealed_preds = {k: {u: sealed[u][k] for u in uids} for k in sealed_keys}
    for k in sealed_keys:
        ev = evaluate_predictions(sealed_preds[k], meta_test, uids)
        print(f"  [Locked] {k:20s}: WER={ev['macro_WER']:6.3f}  CER={ev['macro_CER']:5.3f}", flush=True)

    # Build bidirectional graphs on testunk
    pb1 = {u: build_projection_bundle(omni[u], mms[u], anchor_idx=0, normalize_fn=norm_fn, project_fn=proj_fn) for u in uids}
    pb2 = {u: build_projection_bundle(omni[u], mms[u], anchor_idx=1, normalize_fn=norm_fn, project_fn=proj_fn) for u in uids}
    g1 = {u: build_phrase_graph(pb1[u], agreement_partition(pb1[u]), char_lm, word_lm, add_block_context_feats=True) for u in uids}
    g2 = {u: build_phrase_graph(pb2[u], agreement_partition(pb2[u]), char_lm, word_lm, add_block_context_feats=True) for u in uids}

    # Load frozen FLEURS-12 SymBiList artifact
    ckpt = pickle.loads((ROOT / "results/frozen_symbilist_fleurs12.pkl").read_bytes())
    D = ckpt["sc1"].n_features_in_
    m1_list, m2_list = [], []
    for sd in ckpt["m1_state"]:
        m = ListwiseBlockNet(D, hidden=48); m.load_state_dict(sd); m.eval(); m1_list.append(m)
    for sd in ckpt["m2_state"]:
        m = ListwiseBlockNet(D, hidden=48); m.load_state_dict(sd); m.eval(); m2_list.append(m)

    t_inf = time.monotonic()
    y1_map, wh1_map = predict_listwise_ensemble(g1, pb1, uids, m1_list, ckpt["sc1"], need_whole=True)
    y2_map, _ = predict_listwise_ensemble(g2, pb2, uids, m2_list, ckpt["sc2"], need_whole=False)

    arb_sent = ckpt["arb_sent"]
    arb_span = ckpt["arb_span"]
    a_sym = 0.5 * (ckpt["a1"] + ckpt["a2"])

    sym_sent, sym_phrase = {}, {}
    for u in uids:
        y1, y2 = y1_map[u], y2_map[u]
        if y1 == y2:
            sym_sent[u] = y1
            sym_phrase[u] = y1
            continue
        f1 = symmetric_sentence_feats(y1, pb1[u], char_lm, word_lm)
        f2 = symmetric_sentence_feats(y2, pb1[u], char_lm, word_lm)
        xf = np.concatenate([f2 - f1, 0.5 * (f1 + f2)])[None, :]
        xr = np.concatenate([f1 - f2, 0.5 * (f1 + f2)])[None, :]
        dw = 0.5 * (arb_sent[0].predict(xf)[0] - arb_sent[0].predict(xr)[0])
        dc = 0.5 * (arb_sent[1].predict(xf)[0] - arb_sent[1].predict(xr)[0])
        r0 = max(max(0.0, -dw), a_sym * max(0.0, -dc))
        r1 = max(max(0.0, dw), a_sym * max(0.0, dc))
        sym_sent[u] = y2 if r1 < r0 else y1

        cells_r = proj_fn(y1, y1); proj_y2 = proj_fn(y1, y2)
        rranges, lo = [], 0
        for ic, cv in enumerate(cells_r):
            if ic % 2 == 1 and proj_y2[ic] == cv:
                if ic > lo:
                    rranges.append((lo, ic))
                lo = ic + 1
        if lo < len(cells_r):
            rranges.append((lo, len(cells_r)))
        acts = []
        for lo, hi in rranges:
            pa, pb = flatten_cells(cells_r[lo:hi]), flatten_cells(proj_y2[lo:hi])
            if pa == pb:
                continue
            hb = splice_cells(cells_r, [(lo, hi, pb)])
            fb = symmetric_sentence_feats(hb, pb1[u], char_lm, word_lm)
            xf = np.concatenate([fb - f1, 0.5 * (f1 + fb)])[None, :]
            xr = np.concatenate([f1 - fb, 0.5 * (f1 + fb)])[None, :]
            sw = 0.5 * (arb_span[0].predict(xf)[0] - arb_span[0].predict(xr)[0])
            sc_v = 0.5 * (arb_span[1].predict(xf)[0] - arb_span[1].predict(xr)[0])
            sr0 = max(max(0.0, -sw), a_sym * max(0.0, -sc_v))
            sr1 = max(max(0.0, sw), a_sym * max(0.0, sc_v))
            acts.append((lo, hi, pb if sr1 < sr0 else pa))
        sym_phrase[u] = splice_cells(cells_r, acts)
    dt_inf = time.monotonic() - t_inf
    print(f"SymBiList inference on 3,030 testunk utterances took {dt_inf:.2f}s ({1000*dt_inf/len(uids):.2f} ms/utt)", flush=True)

    out = {"role": "post_lock_frozen_fleurs12_readout", "inference_ms_per_utt": 1000.0 * dt_inf / len(uids), "systems": {}, "bootstrap": {}}
    for name, pr in [
        ("Listwise_dir1_omni", y1_map),
        ("Listwise_dir2_mms", y2_map),
        ("Listwise_whole_fixed", wh1_map),
        ("SymBiList_sent", sym_sent),
        ("SymBiList_phrase", sym_phrase),
    ]:
        ev = evaluate_predictions(pr, meta_test, uids)
        out["systems"][name] = ev
        print(f"  [PostLock] {name:22s}: WER={ev['macro_WER']:6.3f}  CER={ev['macro_CER']:5.3f} | ta={ev['per_lang']['ta_in']['WER']:.3f}/{ev['per_lang']['ta_in']['CER']:.3f} kn={ev['per_lang']['kn_in']['WER']:.3f}/{ev['per_lang']['kn_in']['CER']:.3f}", flush=True)

    # Speaker-stratified paired cluster bootstrap (50,000 draws) vs locked baselines!
    for comp in ["CTC_phrase", "CTC_whole", "CTC_whole_fixed", "slot", "cn"]:
        if comp in sealed_preds:
            ci = paired_cluster_bootstrap_diff(sym_phrase, sealed_preds[comp], meta_test, uids, n_draws=50000, stratify_by_language=True)
            out["bootstrap"][f"SymBiList_vs_{comp}"] = ci
            print(f"  CI SymBiList vs {comp:16s}: dWER={ci['delta_WER_pp']:+.3f} {ci['delta_WER_CI']} | dCER={ci['delta_CER_pp']:+.3f} {ci['delta_CER_CI']}", flush=True)

    p_out = ROOT / "results/testunk_postlock_symbilist.json"
    p_out.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[{time.monotonic()-t0:.1f}s] Saved to {p_out}", flush=True)


if __name__ == "__main__":
    main()
