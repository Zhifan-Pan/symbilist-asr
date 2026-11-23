"""Metrics, paired cluster bootstrap simultaneous CIs, and DP oracle utilities."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any
import numpy as np
from rapidfuzz.distance import Levenshtein


def edit_counts(ref: str, hyp: str) -> tuple[dict[str, int], dict[str, int]]:
    def _cnt(a, b):
        ops = Counter(x.tag for x in Levenshtein.editops(a, b))
        return {
            "S": ops["replace"],
            "D": ops["delete"],
            "I": ops["insert"],
            "E": ops["replace"] + ops["delete"] + ops["insert"],
            "N": len(a),
        }

    rw, hw = ref.split(), hyp.split()
    return _cnt(rw, hw), _cnt("".join(rw), "".join(hw))


def evaluate_predictions(
    preds: dict[str, str],
    meta: dict[str, dict[str, Any]],
    uids: list[str] | None = None,
) -> dict[str, Any]:
    if uids is None:
        uids = sorted(preds.keys())
    by_lang: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(4, dtype=np.int64))
    by_lang_sdi: dict[str, Counter] = defaultdict(Counter)
    for u in uids:
        m = meta[u]
        w, c = edit_counts(m["reference"], preds[u])
        lang = m["language"]
        by_lang[lang] += np.array([w["E"], w["N"], c["E"], c["N"]], dtype=np.int64)
        by_lang_sdi[lang].update(wS=w["S"], wD=w["D"], wI=w["I"], cS=c["S"], cD=c["D"], cI=c["I"], n=1)

    langs = sorted(by_lang.keys())
    mat = np.stack([by_lang[l] for l in langs], axis=0)
    per_lang = {
        l: {
            "WER": float(100.0 * by_lang[l][0] / by_lang[l][1]),
            "CER": float(100.0 * by_lang[l][2] / by_lang[l][3]),
            "wE": int(by_lang[l][0]),
            "wN": int(by_lang[l][1]),
            "cE": int(by_lang[l][2]),
            "cN": int(by_lang[l][3]),
            **dict(by_lang_sdi[l]),
        }
        for l in langs
    }
    macro_wer = float(100.0 * np.mean(mat[:, 0] / mat[:, 1]))
    macro_cer = float(100.0 * np.mean(mat[:, 2] / mat[:, 3]))
    tot = mat.sum(axis=0)
    micro_wer = float(100.0 * tot[0] / tot[1])
    micro_cer = float(100.0 * tot[2] / tot[3])
    return {
        "macro_WER": macro_wer,
        "macro_CER": macro_cer,
        "micro_WER": micro_wer,
        "micro_CER": micro_cer,
        "wE": int(tot[0]),
        "wN": int(tot[1]),
        "cE": int(tot[2]),
        "cN": int(tot[3]),
        "per_lang": per_lang,
    }


def paired_cluster_bootstrap_diff(
    preds_a: dict[str, str],
    preds_b: dict[str, str],
    meta: dict[str, dict[str, Any]],
    uids: list[str],
    n_draws: int = 10000,
    bonferroni_k: int = 1,
    seed: int = 20261123,
    stratify_by_language: bool = False,
) -> dict[str, Any]:
    """Paired cluster bootstrap for (A - B) macro WER and macro CER in percentage points."""
    langs = sorted({meta[u]["language"] for u in uids})
    lang_idx = {l: i for i, l in enumerate(langs)}
    n_l = len(langs)

    # Precompute per-cluster sums of [wE_a, wN, cE_a, cN, wE_b, cE_b] per language
    clusters_by_stratum: dict[str, list[str]] = defaultdict(list)
    cluster_tensors: dict[str, np.ndarray] = {}
    for u in uids:
        m = meta[u]
        cl = m["cluster"]
        stratum = m["language"] if stratify_by_language else "__all__"
        if cl not in cluster_tensors:
            clusters_by_stratum[stratum].append(cl)
            cluster_tensors[cl] = np.zeros((n_l, 6), dtype=np.int64)
        wa, ca = edit_counts(m["reference"], preds_a[u])
        wb, cb = edit_counts(m["reference"], preds_b[u])
        li = lang_idx[m["language"]]
        cluster_tensors[cl][li] += [wa["E"], wa["N"], ca["E"], ca["N"], wb["E"], cb["E"]]

    rng = np.random.Generator(np.random.PCG64(seed))
    wer_diffs = np.zeros(n_draws, dtype=np.float64)
    cer_diffs = np.zeros(n_draws, dtype=np.float64)

    # Vectorized sampling per stratum
    strata_arrays = []
    for st, clist in sorted(clusters_by_stratum.items()):
        arr = np.stack([cluster_tensors[c] for c in clist], axis=0)  # (C, L, 6)
        strata_arrays.append(arr)

    batch = 1000
    for start in range(0, n_draws, batch):
        bs = min(batch, n_draws - start)
        totals = np.zeros((bs, n_l, 6), dtype=np.float64)
        for arr in strata_arrays:
            c_cnt = arr.shape[0]
            # Multinomial counts per draw: (bs, C)
            counts_mat = rng.multinomial(c_cnt, np.full(c_cnt, 1.0 / c_cnt), size=bs)
            totals += np.einsum("bc,clk->blk", counts_mat, arr)
        # Guard against zero denom (rare in tiny strata)
        wn = np.maximum(totals[:, :, 1], 1.0)
        cn = np.maximum(totals[:, :, 3], 1.0)
        wer_a = 100.0 * (totals[:, :, 0] / wn).mean(axis=1)
        wer_b = 100.0 * (totals[:, :, 4] / wn).mean(axis=1)
        cer_a = 100.0 * (totals[:, :, 2] / cn).mean(axis=1)
        cer_b = 100.0 * (totals[:, :, 5] / cn).mean(axis=1)
        wer_diffs[start : start + bs] = wer_a - wer_b
        cer_diffs[start : start + bs] = cer_a - cer_b

    alpha = 0.05 / max(1, bonferroni_k)
    lo_q, hi_q = 100.0 * (alpha / 2.0), 100.0 * (1.0 - alpha / 2.0)

    ev_a = evaluate_predictions(preds_a, meta, uids)
    ev_b = evaluate_predictions(preds_b, meta, uids)
    return {
        "delta_WER_pp": ev_a["macro_WER"] - ev_b["macro_WER"],
        "delta_WER_CI": [float(np.percentile(wer_diffs, lo_q)), float(np.percentile(wer_diffs, hi_q))],
        "delta_CER_pp": ev_a["macro_CER"] - ev_b["macro_CER"],
        "delta_CER_CI": [float(np.percentile(cer_diffs, lo_q)), float(np.percentile(cer_diffs, hi_q))],
        "n_draws": n_draws,
        "bonferroni_k": bonferroni_k,
    }
