"""Stage 2B: Completely Untouched Cross-Architecture (OmniASR-CTC + Whisper-v3-turbo AED) & 7-Unseen-Language Confirmation on Common Voice 26 (2,000 clips, 8 langs)."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
import datasets
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agree_phrase_asr.data_bridge import load_frozen_dev_bundle
from agree_phrase_asr.align import build_projection_bundle, make_projector, normalize_text, flatten_cells, splice_cells, lev_dist
from agree_phrase_asr.partition import agreement_partition, fixed_partition, random_length_matched_partition, validate_partition
from agree_phrase_asr.features import build_phrase_graph, FEATURE_GROUPS, BASE_FEATURE_NAMES
from agree_phrase_asr.lm import CharNgramLM, WordBigramLM
from agree_phrase_asr.eval import edit_counts, evaluate_predictions, paired_cluster_bootstrap_diff
from agree_phrase_asr.models import (
    DEFAULT_HGB_PARAMS,
    prepare_fit_targets,
    build_training_arrays,
    fit_dual_regressor,
    predict_block_costs,
    choose_from_block_costs,
    predict_exact_nbest_mbr,
    predict_unsupervised_block_mbr,
    predict_alignment_cell_vote,
)


class ListwiseBlockNet(nn.Module):
    def __init__(self, d_in: int, hidden: int = 48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def fit_listwise_ens(train_uids, graphs, meta, col_indices=None, alpha_c=0.5, seed=1701, n_ens=3, epochs=35):
    per_lang = defaultdict(list)
    active = defaultdict(int)
    for u in train_uids:
        rw = meta[u]["reference"].split()
        per_lang[meta[u]["language"]].append([len(rw), len("".join(rw))])
        if graphs[u][1]:
            active[meta[u]["language"]] += 1
    scale = {l: np.mean(v, axis=0) for l, v in per_lang.items()}
    all_rows, block_recs = [], []
    for u in train_uids:
        _, blocks = graphs[u]
        if not blocks:
            continue
        ref = meta[u]["reference"]
        lang = meta[u]["language"]
        sc = scale[lang]
        wt = 1.0 / (active[lang] * len(blocks))
        for bl in blocks:
            xa = bl["X_all"] if col_indices is None else bl["X_all"][:, col_indices]
            errs = np.array([[edit_counts(ref, h)[0]["E"], edit_counts(ref, h)[1]["E"]] for h in bl["full"]], dtype=np.float64) / sc
            ce = errs[:, 0] + alpha_c * errs[:, 1]
            ce = ce - ce.min()
            all_rows.append(xa)
            block_recs.append((xa, ce, wt))
    scaler = StandardScaler().fit(np.vstack(all_rows))
    w_sum = sum(r[2] for r in block_recs)
    max_m = max(len(r[0]) for r in block_recs)
    B, D = len(block_recs), block_recs[0][0].shape[1]
    X_pad = np.zeros((B, max_m, D), dtype=np.float32)
    Y_pad = np.zeros((B, max_m), dtype=np.float32)
    M_pad = np.zeros((B, max_m), dtype=np.bool_)
    W_arr = np.zeros(B, dtype=np.float32)
    for i, (xa, ye, wt) in enumerate(block_recs):
        m = len(xa)
        X_pad[i, :m] = scaler.transform(xa).astype(np.float32)
        Y_pad[i, :m] = ye.astype(np.float32)
        M_pad[i, :m] = True
        W_arr[i] = float(wt * B / w_sum)
    Xt, Yt, Mt, Wt = torch.from_numpy(X_pad), torch.from_numpy(Y_pad), torch.from_numpy(M_pad), torch.from_numpy(W_arr)
    models = []
    for ei in range(n_ens):
        torch.manual_seed(seed + ei * 17)
        np.random.seed(seed + ei * 17)
        mod = ListwiseBlockNet(D, hidden=48)
        opt = torch.optim.AdamW(mod.parameters(), lr=3e-3, weight_decay=2e-2)
        mod.train()
        for _ in range(epochs):
            perm = torch.randperm(B)
            for s in range(0, B, 512):
                idx = perm[s : s + 512]
                logits = mod(Xt[idx]).masked_fill(~Mt[idx], -1e9)
                probs = torch.softmax(logits, dim=-1)
                loss = ((probs * Yt[idx]).sum(dim=-1) * Wt[idx]).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        mod.eval()
        models.append(mod)
    return models, scaler


def pred_listwise_ens(graphs, bundles, uids, models, scaler, col_indices=None, need_whole=False):
    out, whole_out = {}, {}
    with torch.no_grad():
        for u in uids:
            cells, blocks = graphs[u]
            acts = []
            pool = bundles[u]["texts"]
            proj = bundles[u]["proj"]
            wscore = np.zeros(len(pool), dtype=np.float64)
            for bl in blocks:
                xa = bl["X_all"] if col_indices is None else bl["X_all"][:, col_indices]
                x = torch.from_numpy(scaler.transform(xa).astype(np.float32))
                probs = np.mean([torch.softmax(m(x), dim=-1).numpy() for m in models], axis=0)
                j = int(np.argmax(probs))
                acts.append((bl["lo"], bl["hi"], bl["alts"][j]))
                if need_whole:
                    logp = np.log(np.clip(probs, 1e-12, 1.0))
                    for pi, h in enumerate(pool):
                        wscore[pi] += float(logp[bl["alts"].index(flatten_cells(proj[h][bl["lo"]:bl["hi"]]))])
            out[u] = splice_cells(cells, acts)
            if need_whole:
                whole_out[u] = pool[int(np.argmax(wscore))]
    return out, whole_out


def sym_sent_feats(h, b0, char_lm_fn, word_lm, use_lm=True):
    hw = h.split(); hc = "".join(hw)
    g1w, g2w = b0["g1"].split(), b0["g2"].split()
    g1c, g2c = "".join(g1w), "".join(g2w)
    A_sym = max(1.0, 0.5 * (len(g1w) + len(g2w)))
    C_sym = max(1.0, 0.5 * (len(g1c) + len(g2c)))
    r1w = sum(wt * lev_dist(hw, k.split()) for k, wt in b0["p1"].items()) / A_sym
    r2w = sum(wt * lev_dist(hw, k.split()) for k, wt in b0["p2"].items()) / A_sym
    r1c = sum(wt * lev_dist(hc, "".join(k.split())) for k, wt in b0["p1"].items()) / C_sym
    r2c = sum(wt * lev_dist(hc, "".join(k.split())) for k, wt in b0["p2"].items()) / C_sym
    base = [r1w, r2w, 0.5 * (r1w + r2w), r1c, r2c, 0.5 * (r1c + r2c), float(len(hw)) / A_sym, float(len(hc)) / C_sym, float(h in b0["p1"]), float(h in b0["p2"])]
    if use_lm:
        c4_raw = float(char_lm_fn(h))
        w2 = word_lm.scores(h)
        base = [c4_raw / (len(h) + 1), c4_raw / (C_sym + 1.0), w2[0], w2[1], w2[2]] + base
    return np.array(base, dtype=np.float64)


def fit_and_apply_sym_reconciler(tr_ids, tr_y1, tr_y2, tr_pb, ev_ids, ev_y1, ev_y2, ev_pb, meta, char_lm_fn, word_lm, proj_fn, use_lm=True, alpha_c=0.5):
    per_lang = defaultdict(list)
    for u in tr_ids:
        rw = meta[u]["reference"].split()
        per_lang[meta[u]["language"]].append([len(rw), len("".join(rw))])
    lscale = {l: np.mean(v, axis=0) for l, v in per_lang.items()}
    Xap, Yap = [], []
    for u in tr_ids:
        y1, y2 = tr_y1[u], tr_y2[u]
        if y1 == y2:
            continue
        ref = meta[u]["reference"]; sc = lscale[meta[u]["language"]]
        f1 = sym_sent_feats(y1, tr_pb[u], char_lm_fn, word_lm, use_lm=use_lm)
        e1 = np.array([edit_counts(ref, y1)[0]["E"], edit_counts(ref, y1)[1]["E"]], dtype=np.float64) / sc
        cells_r = proj_fn(y1, y1); proj_y2 = proj_fn(y1, y2)
        rranges, lo = [], 0
        for ic, cv in enumerate(cells_r):
            if ic % 2 == 1 and proj_y2[ic] == cv:
                if ic > lo:
                    rranges.append((lo, ic))
                lo = ic + 1
        if lo < len(cells_r):
            rranges.append((lo, len(cells_r)))
        for lo, hi in rranges:
            pa, pb = flatten_cells(cells_r[lo:hi]), flatten_cells(proj_y2[lo:hi])
            if pa == pb:
                continue
            hb = splice_cells(cells_r, [(lo, hi, pb)])
            fb = sym_sent_feats(hb, tr_pb[u], char_lm_fn, word_lm, use_lm=use_lm)
            eb = np.array([edit_counts(ref, hb)[0]["E"], edit_counts(ref, hb)[1]["E"]], dtype=np.float64) / sc
            Xap.append(np.concatenate([fb - f1, 0.5 * (f1 + fb)])); Yap.append(eb - e1)
            Xap.append(np.concatenate([f1 - fb, 0.5 * (f1 + fb)])); Yap.append(e1 - eb)
    arb_p = dict(max_iter=60, learning_rate=0.05, max_leaf_nodes=7, max_depth=3, min_samples_leaf=20, l2_regularization=5.0, early_stopping=False, random_state=1701)
    Xap_a, Yap_a = np.asarray(Xap), np.asarray(Yap)
    arb = [HistGradientBoostingRegressor(**arb_p).fit(Xap_a, Yap_a[:, i]) for i in (0, 1)]

    preds = {}
    for u in ev_ids:
        y1, y2 = ev_y1[u], ev_y2[u]
        if y1 == y2:
            preds[u] = y1
            continue
        f1 = sym_sent_feats(y1, ev_pb[u], char_lm_fn, word_lm, use_lm=use_lm)
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
            fb = sym_sent_feats(hb, ev_pb[u], char_lm_fn, word_lm, use_lm=use_lm)
            xf = np.concatenate([fb - f1, 0.5 * (f1 + fb)])[None, :]
            xr = np.concatenate([f1 - fb, 0.5 * (f1 + fb)])[None, :]
            sw = 0.5 * (arb[0].predict(xf)[0] - arb[0].predict(xr)[0])
            sc_v = 0.5 * (arb[1].predict(xf)[0] - arb[1].predict(xr)[0])
            sr0 = max(max(0.0, -sw), alpha_c * max(0.0, -sc_v))
            sr1 = max(max(0.0, sw), alpha_c * max(0.0, sc_v))
            acts.append((lo, hi, pb if sr1 < sr0 else pa))
        preds[u] = splice_cells(cells_r, acts)
    return preds


def build_slot_graph(bundle, char_lm_fn, word_lm):
    """Exact learned word-slot graph matching original protocol."""
    proj = bundle["proj"]
    base = bundle["base"]
    donor = bundle["donor"]
    texts = bundle["texts"]
    n = len(bundle["cells"])
    coords = [(i, j) for i in range(n) for j in range(max(len(z[i]) for z in proj.values()))]
    sproj = {h: tuple((z[i][j],) if j < len(z[i]) else () for i, j in coords) for h, z in proj.items()}
    scells = sproj[base]
    A = max(1, len(base.split()))
    C_base = max(1, len("".join(base.split())))
    div = lev_dist(base.split(), donor.split()) / A
    base_ll = float(char_lm_fn(base))
    base_wlm = np.asarray(word_lm.scores(base), dtype=np.float64)
    blocks = []
    for pos_idx in range(len(scells)):
        phrases = {h: flatten_cells(sproj[h][pos_idx : pos_idx + 1]) for h in texts}
        v0 = phrases[base]
        alts = list(dict.fromkeys(phrases.values()))
        if len(alts) < 2:
            continue
        dists = []
        for weights in [bundle["p1"], bundle["p2"]]:
            d = defaultdict(float)
            for h, wt in weights.items():
                d[phrases[h]] += wt
            dists.append(d)
        xs, full = [], []
        for v in alts:
            h = splice_cells(scells, [(pos_idx, pos_idx + 1, v)])
            full.append(h)
            ll = float(char_lm_fn(h))
            wlm_d = np.asarray(word_lm.scores(h), dtype=np.float64) - base_wlm
            feats = [
                div, coords[pos_idx][0] / (2 * A + 1), len(alts), len(v0), len(v),
                lev_dist(v0, v) / A, len("".join(v)) - len("".join(v0)),
                ll / (len(h) + 1) - base_ll / (len(base) + 1),
                (ll - base_ll) / (len(base) + 1),
                *wlm_d.tolist(),
            ]
            vs, v0s = "".join(v), "".join(v0)
            for d in dists:
                dw = sum(wt * (lev_dist(v, a) - lev_dist(v0, a)) for a, wt in d.items())
                dc = sum(wt * (lev_dist(vs, "".join(a)) - lev_dist(v0s, "".join(a))) for a, wt in d.items())
                feats.extend([d.get(v, 0.0), d.get(v0, 0.0), dw / A, dc / C_base, len(d), max(d.values())])
            xs.append(feats)
        X_full = np.asarray(xs, dtype=np.float64)
        blocks.append({"lo": pos_idx, "hi": pos_idx + 1, "alts": alts, "full": full, "X": X_full[1:], "X_all": X_full})
    return scells, blocks


def summarize_cv26(preds: dict[str, str], cv_meta: dict, all_uids: list[str], seen_uids: list[str], unseen_uids: list[str]):
    ev_all = evaluate_predictions(preds, cv_meta, all_uids)
    ev_seen = evaluate_predictions(preds, cv_meta, seen_uids)
    ev_unseen = evaluate_predictions(preds, cv_meta, unseen_uids)
    return {
        "all8": {"macro_WER": ev_all["macro_WER"], "macro_CER": ev_all["macro_CER"]},
        "seen_ta": {"macro_WER": ev_seen["macro_WER"], "macro_CER": ev_seen["macro_CER"]},
        "unseen7": {"macro_WER": ev_unseen["macro_WER"], "macro_CER": ev_unseen["macro_CER"]},
        "per_lang": {l: {"WER": v["WER"], "CER": v["CER"]} for l, v in ev_all["per_lang"].items()},
    }


def main():
    t0 = time.monotonic()
    # 1. Load CV26 sealed refs + OmniASR + Whisper-v3-turbo caches
    cv_refs = [json.loads(l) for l in (ROOT / "manifests/cv26_2000_refs_SEALED.jsonl").open()]
    cv_meta = {r["uid"]: {"dataset": "CV26", "language": r["language"], "cluster": r["cluster"], "reference": r["reference"], "seen": r["seen_in_fleurs12"]} for r in cv_refs}
    omni_cv = {r["uid"]: r for r in (json.loads(l) for l in (ROOT / "cache/omni1B_cv26_2000.jsonl").open())}
    whis_cv = {r["uid"]: r for r in (json.loads(l) for l in (ROOT / "cache/whisper_v3t_cv26_2000.jsonl").open())}
    cv_uids = sorted(cv_meta.keys())
    assert set(omni_cv) == set(whis_cv) == set(cv_uids) and len(cv_uids) == 2000
    seen_uids = [u for u in cv_uids if cv_meta[u]["seen"]]
    unseen_uids = [u for u in cv_uids if not cv_meta[u]["seen"]]

    proj_fn = make_projector()
    # Build CV26 projections: Source 1 = OmniASR-1B-CTC, Source 2 = Whisper-v3-turbo (AED)
    # Note: Whisper sequences_scores are mean token log-probs (~ -0.05 to -0.5); T=0.5 works cleanly, and we also preserve T=0.5 consistency!
    pb_cv_s1 = {u: build_projection_bundle(omni_cv[u], whis_cv[u], anchor_idx=0, normalize_fn=normalize_text, project_fn=proj_fn) for u in cv_uids}
    pb_cv_s2 = {u: build_projection_bundle(omni_cv[u], whis_cv[u], anchor_idx=1, normalize_fn=normalize_text, project_fn=proj_fn) for u in cv_uids}

    results = {"single_and_unsupervised": {}, "Z1_zero_shot_lang_agnostic": {}, "Z3_cv26_5fold_cross_arch": {}, "bootstrap_Z3": {}}

    # ---- Single models & Unsupervised baselines (ZERO labels, ZERO LM) ----
    print("=== Common Voice 26 (2,000 clips, 8 langs: 1 seen ta + 7 unseen) — Single & Unsupervised ===", flush=True)
    singles = {
        "OmniASR_CTC_greedy": {u: pb_cv_s1[u]["g1"] for u in cv_uids},
        "Whisper_v3t_AED_greedy": {u: pb_cv_s1[u]["g2"] for u in cv_uids},
        "MBR_word_unsup": predict_exact_nbest_mbr(pb_cv_s1, cv_uids, metric="word"),
        "MBR_dual_unsup": predict_exact_nbest_mbr(pb_cv_s1, cv_uids, metric="dual"),
        "Align_Cell_Vote_unsup": predict_alignment_cell_vote(pb_cv_s1, cv_uids),
    }
    # Dummy zero-LM for unsupervised block MBR
    zero_char = lambda s: 0.0
    class ZeroWordLM:
        def scores(self, s): return (0.0, 0.0, 0.0)
    zlm = ZeroWordLM()
    g_cv_nolm_s1 = {u: build_phrase_graph(pb_cv_s1[u], agreement_partition(pb_cv_s1[u]), zero_char, zlm, add_block_context_feats=True) for u in cv_uids}
    g_cv_nolm_s2 = {u: build_phrase_graph(pb_cv_s2[u], agreement_partition(pb_cv_s2[u]), zero_char, zlm, add_block_context_feats=True) for u in cv_uids}
    singles["Block_MBR_word_unsup"] = predict_unsupervised_block_mbr(g_cv_nolm_s1, cv_uids, metric="word")
    for name, pr in singles.items():
        s = summarize_cv26(pr, cv_meta, cv_uids, seen_uids, unseen_uids)
        results["single_and_unsupervised"][name] = s
        print(f"  {name:24s} | All8 {s['all8']['macro_WER']:6.3f}/{s['all8']['macro_CER']:5.3f} | Seen(ta) {s['seen_ta']['macro_WER']:6.3f}/{s['seen_ta']['macro_CER']:5.3f} | Unseen7 {s['unseen7']['macro_WER']:6.3f}/{s['unseen7']['macro_CER']:5.3f}", flush=True)

    # ---- Protocol Z1: Strictly Zero-Shot Language-Agnostic Transfer (Trained ONLY on FLEURS-12 Omni+MMS, NO LM, NO CV26 labels!) ----
    print(f"\n[{time.monotonic()-t0:.1f}s] === Protocol Z1: Zero-Shot Cross-Lingual + Cross-Architecture Transfer (FLEURS-12 -> CV26, ZERO target labels/LM) ===", flush=True)
    fl_bundle = load_frozen_dev_bundle()
    fl_uids = fl_bundle["fleurs_uids"]
    fl_meta = fl_bundle["meta"]
    fl_proj = make_projector(fl_bundle["align_fn"])
    pb_fl_s1 = {u: build_projection_bundle(fl_bundle["fleurs"][u][0], fl_bundle["fleurs"][u][1], anchor_idx=0, normalize_fn=fl_bundle["normalize_fn"], project_fn=fl_proj) for u in fl_uids}
    pb_fl_s2 = {u: build_projection_bundle(fl_bundle["fleurs"][u][0], fl_bundle["fleurs"][u][1], anchor_idx=1, normalize_fn=fl_bundle["normalize_fn"], project_fn=fl_proj) for u in fl_uids}
    g_fl_nolm_s1 = {u: build_phrase_graph(pb_fl_s1[u], agreement_partition(pb_fl_s1[u]), zero_char, zlm, add_block_context_feats=True) for u in fl_uids}
    g_fl_nolm_s2 = {u: build_phrase_graph(pb_fl_s2[u], agreement_partition(pb_fl_s2[u]), zero_char, zlm, add_block_context_feats=True) for u in fl_uids}

    # Language-agnostic feature columns: drop char_lm(7,8), word_lm(9,10,11) and LM context cols(24,25,30) -> keep geom(0..6), src_dist(12..23), non-LM ctx(26,27,28,29,31)
    nolm_cols_24 = [c for c in range(24) if c not in FEATURE_GROUPS["all_lm"]]
    nolm_cols_32 = nolm_cols_24 + [26, 27, 28, 29, 31]

    ft_fl_s1 = prepare_fit_targets(g_fl_nolm_s1, fl_meta, fl_uids)
    ft_fl_s2 = prepare_fit_targets(g_fl_nolm_s2, fl_meta, fl_uids)
    m_z1_hgb1 = fit_dual_regressor("hgb", fl_uids, ft_fl_s1, fl_meta, col_indices=nolm_cols_24, seed=1701)
    sc_z1_1 = predict_block_costs(g_cv_nolm_s1, cv_uids, m_z1_hgb1, col_indices=nolm_cols_24)
    z1_A, z1_B = {}, {}
    for u in cv_uids:
        z1_A[u], z1_B[u] = choose_from_block_costs(g_cv_nolm_s1[u], sc_z1_1[u], pb_cv_s1[u], alpha_c=0.5, need_whole=True)

    # Zero-shot SymBiList (LM-free) trained strictly on FLEURS-12!
    m_z1_lw1, sc_lw1 = fit_listwise_ens(fl_uids, g_fl_nolm_s1, fl_meta, col_indices=nolm_cols_32, alpha_c=0.5, seed=1701, n_ens=3, epochs=35)
    m_z1_lw2, sc_lw2 = fit_listwise_ens(fl_uids, g_fl_nolm_s2, fl_meta, col_indices=nolm_cols_32, alpha_c=0.5, seed=1701, n_ens=3, epochs=35)
    z1_lw1, z1_lw_wh = pred_listwise_ens(g_cv_nolm_s1, pb_cv_s1, cv_uids, m_z1_lw1, sc_lw1, col_indices=nolm_cols_32, need_whole=True)
    z1_lw2, _ = pred_listwise_ens(g_cv_nolm_s2, pb_cv_s2, cv_uids, m_z1_lw2, sc_lw2, col_indices=nolm_cols_32, need_whole=False)
    # Use FLEURS-12 training predictions to fit LM-free symmetric span reconciler
    fl_y1, _ = pred_listwise_ens(g_fl_nolm_s1, pb_fl_s1, fl_uids, m_z1_lw1, sc_lw1, col_indices=nolm_cols_32)
    fl_y2, _ = pred_listwise_ens(g_fl_nolm_s2, pb_fl_s2, fl_uids, m_z1_lw2, sc_lw2, col_indices=nolm_cols_32)
    z1_symbilist = fit_and_apply_sym_reconciler(fl_uids, fl_y1, fl_y2, pb_fl_s1, cv_uids, z1_lw1, z1_lw2, pb_cv_s1, fl_meta, zero_char, zlm, proj_fn, use_lm=False, alpha_c=0.5)

    for name, pr in [
        ("Z1_Whole_fixed_HGB", z1_B),
        ("Z1_Phrase_A_HGB", z1_A),
        ("Z1_Whole_fixed_Listwise", z1_lw_wh),
        ("Z1_Listwise_OmniAnchor", z1_lw1),
        ("Z1_Listwise_WhisAnchor", z1_lw2),
        ("Z1_SymBiList_ZeroShot", z1_symbilist),
    ]:
        s = summarize_cv26(pr, cv_meta, cv_uids, seen_uids, unseen_uids)
        results["Z1_zero_shot_lang_agnostic"][name] = s
        print(f"  {name:26s} | All8 {s['all8']['macro_WER']:6.3f}/{s['all8']['macro_CER']:5.3f} | Seen(ta) {s['seen_ta']['macro_WER']:6.3f}/{s['seen_ta']['macro_CER']:5.3f} | Unseen7 {s['unseen7']['macro_WER']:6.3f}/{s['unseen7']['macro_CER']:5.3f}", flush=True)

    # ---- Protocol Z3: Full Cross-Architecture CTC+AED 5-Fold OOF on CV26 (with text-only LM from CV26 train split!) ----
    print(f"\n[{time.monotonic()-t0:.1f}s] === Protocol Z3: Cross-Architecture CTC+AED on CV26 (text-only train LM, 5-fold OOF) ===", flush=True)
    cv_train_texts = []
    cv_dir = Path(os.environ["SYMBILIST_CV26_DIR"])
    for lang in ["ta", "ar", "tr", "sw", "ru", "gl", "es", "en"]:
        ds_tr = datasets.load_from_disk(str(cv_dir / lang / "train"))
        # Up to 5,000 train transcripts per language (strictly train split, zero overlap with eval!)
        for idx_r in range(min(5000, len(ds_tr))):
            t = normalize_text(ds_tr[idx_r]["text"])
            if t:
                cv_train_texts.append(t)
    print(f"  Fitting CV26 text-only Char4 + Word2 LMs on {len(cv_train_texts)} train transcripts...", flush=True)
    c4_cv = CharNgramLM(order=4, alpha=0.1).fit(cv_train_texts)
    w2_cv = WordBigramLM(alpha=0.1).fit(cv_train_texts)

    # Build graphs with CV26 text-only LM
    g_cv_s1 = {u: build_phrase_graph(pb_cv_s1[u], agreement_partition(pb_cv_s1[u]), c4_cv.score, w2_cv, add_block_context_feats=True) for u in cv_uids}
    g_cv_s2 = {u: build_phrase_graph(pb_cv_s2[u], agreement_partition(pb_cv_s2[u]), c4_cv.score, w2_cv, add_block_context_feats=True) for u in cv_uids}
    g_cv_F = {u: build_phrase_graph(pb_cv_s1[u], fixed_partition(pb_cv_s1[u], 3), c4_cv.score, w2_cv) for u in cv_uids}
    g_cv_Q = {u: build_phrase_graph(pb_cv_s1[u], random_length_matched_partition(pb_cv_s1[u], u, 101), c4_cv.score, w2_cv) for u in cv_uids}
    g_cv_slots = {u: build_slot_graph(pb_cv_s1[u], c4_cv.score, w2_cv) for u in cv_uids}

    # Stratified 5-fold outer split by deterministic SHA256 hash within each language
    fold_of = {}
    for lang in ["ta", "ar", "tr", "sw", "ru", "gl", "es", "en"]:
        luids = sorted([u for u in cv_uids if cv_meta[u]["language"] == lang], key=lambda u: hashlib.sha256(u.encode()).hexdigest())
        for idx_u, u in enumerate(luids):
            fold_of[u] = idx_u % 5

    ft_s1 = prepare_fit_targets(g_cv_s1, cv_meta, cv_uids)
    ft_s2 = prepare_fit_targets(g_cv_s2, cv_meta, cv_uids)
    ft_F = prepare_fit_targets(g_cv_F, cv_meta, cv_uids)
    ft_Q = prepare_fit_targets(g_cv_Q, cv_meta, cv_uids)
    ft_slots = prepare_fit_targets(g_cv_slots, cv_meta, cv_uids)

    cols24 = list(range(24))
    z3_preds = {k: {} for k in [
        "Whole_fixed_HGB", "Fixed_L3_HGB", "Random_Q_HGB", "Learned_Slots_HGB",
        "Baseline_A_HGB_OmniAnc", "Baseline_A_HGB_WhisAnc", "SymBiPhrase_HGB",
        "Listwise_Whole_fixed", "Listwise_OmniAnc", "Listwise_WhisAnc", "SymBiList_Ours"
    ]}

    for f in range(5):
        tr = [u for u in cv_uids if fold_of[u] != f]
        ev = [u for u in cv_uids if fold_of[u] == f]

        # Standard 24d HGB models (alpha_c=0.5 fixed from FLEURS protocol to avoid over-tuning on CV26!)
        mA1 = fit_dual_regressor("hgb", tr, ft_s1, cv_meta, col_indices=cols24, seed=1701)
        mA2 = fit_dual_regressor("hgb", tr, ft_s2, cv_meta, col_indices=cols24, seed=1701)
        mF = fit_dual_regressor("hgb", tr, ft_F, cv_meta, seed=1701)
        mQ = fit_dual_regressor("hgb", tr, ft_Q, cv_meta, seed=1701)
        mSl = fit_dual_regressor("hgb", tr, ft_slots, cv_meta, seed=1701)

        sA1_ev = predict_block_costs(g_cv_s1, ev, mA1, col_indices=cols24)
        sA2_ev = predict_block_costs(g_cv_s2, ev, mA2, col_indices=cols24)
        sF_ev = predict_block_costs(g_cv_F, ev, mF)
        sQ_ev = predict_block_costs(g_cv_Q, ev, mQ)
        sSl_ev = predict_block_costs(g_cv_slots, ev, mSl)

        ev_A1, ev_A2 = {}, {}
        for u in ev:
            z3_preds["Baseline_A_HGB_OmniAnc"][u], z3_preds["Whole_fixed_HGB"][u] = choose_from_block_costs(g_cv_s1[u], sA1_ev[u], pb_cv_s1[u], alpha_c=0.5, need_whole=True)
            z3_preds["Baseline_A_HGB_WhisAnc"][u], _ = choose_from_block_costs(g_cv_s2[u], sA2_ev[u], pb_cv_s2[u], alpha_c=0.5)
            z3_preds["Fixed_L3_HGB"][u], _ = choose_from_block_costs(g_cv_F[u], sF_ev[u], pb_cv_s1[u], alpha_c=0.5)
            z3_preds["Random_Q_HGB"][u], _ = choose_from_block_costs(g_cv_Q[u], sQ_ev[u], pb_cv_s1[u], alpha_c=0.5)
            z3_preds["Learned_Slots_HGB"][u], _ = choose_from_block_costs(g_cv_slots[u], sSl_ev[u], pb_cv_s1[u], alpha_c=0.5)
            ev_A1[u] = z3_preds["Baseline_A_HGB_OmniAnc"][u]
            ev_A2[u] = z3_preds["Baseline_A_HGB_WhisAnc"][u]

        # Inner 2-fold OOF on tr for SymBiPhrase_HGB & SymBiList reconcilers (strictly leakage-free!)
        tr_sub0 = [u for i_u, u in enumerate(tr) if i_u % 2 == 0]
        tr_sub1 = [u for i_u, u in enumerate(tr) if i_u % 2 == 1]
        in_A1, in_A2, in_L1, in_L2 = {}, {}, {}, {}
        for sub_tr, sub_va in [(tr_sub0, tr_sub1), (tr_sub1, tr_sub0)]:
            im1 = fit_dual_regressor("hgb", sub_tr, ft_s1, cv_meta, col_indices=cols24, seed=1701)
            im2 = fit_dual_regressor("hgb", sub_tr, ft_s2, cv_meta, col_indices=cols24, seed=1701)
            is1 = predict_block_costs(g_cv_s1, sub_va, im1, col_indices=cols24)
            is2 = predict_block_costs(g_cv_s2, sub_va, im2, col_indices=cols24)
            for u in sub_va:
                in_A1[u], _ = choose_from_block_costs(g_cv_s1[u], is1[u], pb_cv_s1[u], alpha_c=0.5)
                in_A2[u], _ = choose_from_block_costs(g_cv_s2[u], is2[u], pb_cv_s2[u], alpha_c=0.5)
            il1, isc1 = fit_listwise_ens(sub_tr, g_cv_s1, cv_meta, alpha_c=0.5, seed=1701, n_ens=1, epochs=35)
            il2, isc2 = fit_listwise_ens(sub_tr, g_cv_s2, cv_meta, alpha_c=0.5, seed=1701, n_ens=1, epochs=35)
            p1, _ = pred_listwise_ens(g_cv_s1, pb_cv_s1, sub_va, il1, isc1)
            p2, _ = pred_listwise_ens(g_cv_s2, pb_cv_s2, sub_va, il2, isc2)
            in_L1.update(p1); in_L2.update(p2)

        sym_hgb_ev = fit_and_apply_sym_reconciler(tr, in_A1, in_A2, pb_cv_s1, ev, ev_A1, ev_A2, pb_cv_s1, cv_meta, c4_cv.score, w2_cv, proj_fn, use_lm=True, alpha_c=0.5)
        z3_preds["SymBiPhrase_HGB"].update(sym_hgb_ev)

        ml1, msc1 = fit_listwise_ens(tr, g_cv_s1, cv_meta, alpha_c=0.5, seed=1701, n_ens=3, epochs=40)
        ml2, msc2 = fit_listwise_ens(tr, g_cv_s2, cv_meta, alpha_c=0.5, seed=1701, n_ens=3, epochs=40)
        ev_L1, ev_Lwh = pred_listwise_ens(g_cv_s1, pb_cv_s1, ev, ml1, msc1, need_whole=True)
        ev_L2, _ = pred_listwise_ens(g_cv_s2, pb_cv_s2, ev, ml2, msc2, need_whole=False)
        z3_preds["Listwise_OmniAnc"].update(ev_L1)
        z3_preds["Listwise_WhisAnc"].update(ev_L2)
        z3_preds["Listwise_Whole_fixed"].update(ev_Lwh)
        sym_lw_ev = fit_and_apply_sym_reconciler(tr, in_L1, in_L2, pb_cv_s1, ev, ev_L1, ev_L2, pb_cv_s1, cv_meta, c4_cv.score, w2_cv, proj_fn, use_lm=True, alpha_c=0.5)
        z3_preds["SymBiList_Ours"].update(sym_lw_ev)
        print(f"  CV26 fold {f}/5 done ({time.monotonic()-t0:.1f}s)", flush=True)

    print("\n=== Protocol Z3 Results (CV26 8 langs, CTC OmniASR + AED Whisper-v3-turbo) ===", flush=True)
    for name, pr in z3_preds.items():
        s = summarize_cv26(pr, cv_meta, cv_uids, seen_uids, unseen_uids)
        results["Z3_cv26_5fold_cross_arch"][name] = s
        print(f"  {name:24s} | All8 {s['all8']['macro_WER']:6.3f}/{s['all8']['macro_CER']:5.3f} | Seen(ta) {s['seen_ta']['macro_WER']:6.3f}/{s['seen_ta']['macro_CER']:5.3f} | Unseen7 {s['unseen7']['macro_WER']:6.3f}/{s['unseen7']['macro_CER']:5.3f}", flush=True)

    # Paired bootstrap CIs for SymBiList_Ours on CV26
    for comp_name, comp_pr in [
        ("Whole_fixed_HGB", z3_preds["Whole_fixed_HGB"]),
        ("Listwise_Whole_fixed", z3_preds["Listwise_Whole_fixed"]),
        ("Baseline_A_HGB_OmniAnc", z3_preds["Baseline_A_HGB_OmniAnc"]),
        ("Fixed_L3_HGB", z3_preds["Fixed_L3_HGB"]),
        ("Learned_Slots_HGB", z3_preds["Learned_Slots_HGB"]),
        ("Whisper_v3t_AED_greedy", singles["Whisper_v3t_AED_greedy"]),
        ("OmniASR_CTC_greedy", singles["OmniASR_CTC_greedy"]),
    ]:
        ci_all = paired_cluster_bootstrap_diff(z3_preds["SymBiList_Ours"], comp_pr, cv_meta, cv_uids, n_draws=10000, stratify_by_language=True)
        results["bootstrap_Z3"][f"SymBiList_vs_{comp_name}"] = ci_all
        print(f"  CI SymBiList vs {comp_name:22s}: dWER={ci_all['delta_WER_pp']:+.3f} {ci_all['delta_WER_CI']} | dCER={ci_all['delta_CER_pp']:+.3f} {ci_all['delta_CER_CI']}", flush=True)

    out_path = ROOT / "results" / "stage2_cv26_cross_arch.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n[{time.monotonic()-t0:.1f}s] Saved Stage 2B CV26 results to {out_path}", flush=True)


if __name__ == "__main__":
    main()
