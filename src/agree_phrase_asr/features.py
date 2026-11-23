"""Explicit 24-feature schema and extended context/symmetric feature extractors."""
from __future__ import annotations

from collections import Counter
from typing import Any, Callable
import numpy as np
from .align import flatten_cells, splice_cells, lev_dist

BASE_FEATURE_NAMES = [
    # Group 0: Edit geometry & position (7)
    "anchor_donor_word_div",
    "norm_first_cell_pos",
    "num_block_alts",
    "anchor_phrase_word_len",
    "cand_phrase_word_len",
    "norm_phrase_word_dist",
    "delta_nonspace_char_len",
    # Group 1: Char 4-gram LM (2)
    "char4_delta_per_char",
    "char4_delta_norm_sum",
    # Group 2: Word 2-gram LM (3)
    "word2_delta_per_tok",
    "word2_delta_per_char",
    "word2_delta_unk_frac",
    # Group 3: Source-1 (anchor in default) projected mass & expected edit risk (6)
    "src1_cand_mass",
    "src1_anchor_mass",
    "src1_exp_word_risk",
    "src1_exp_char_risk",
    "src1_num_distinct",
    "src1_max_mass",
    # Group 4: Source-2 (donor in default) projected mass & expected edit risk (6)
    "src2_cand_mass",
    "src2_anchor_mass",
    "src2_exp_word_risk",
    "src2_exp_char_risk",
    "src2_num_distinct",
    "src2_max_mass",
]

FEATURE_GROUPS = {
    "geom": list(range(0, 7)),
    "char_lm": [7, 8],
    "word_lm": [9, 10, 11],
    "all_lm": [7, 8, 9, 10, 11],
    "src_dist": list(range(12, 24)),
    "src_risk_eq4": [14, 15, 20, 21],
    "src_mass_only": [12, 13, 16, 17, 18, 19, 22, 23],
}


def nonspace_len(text: str) -> int:
    return len("".join(text.split()))


def build_phrase_graph(
    bundle: dict,
    ranges: list[tuple[int, int]],
    char_lm_fn: Callable[[str], float],
    word_lm: Any,
    source_order: str = "fixed_id",
    add_anchor_id_feat: bool = False,
    add_block_context_feats: bool = False,
) -> tuple[tuple[tuple[str, ...], ...], list[dict]]:
    """Extract block options, full single-replacement sentences, and feature matrices.

    When anchor_idx=0 and source_order='fixed_id', add_anchor_id_feat=False,
    add_block_context_feats=False, the 24-dim feature matrix matches frozen baseline A bit-for-bit.
    """
    base = bundle["base"]
    donor = bundle["donor"]
    texts = bundle["texts"]
    proj = bundle["proj"]
    cells = bundle["cells"]
    anchor_idx = bundle["anchor_idx"]

    A = max(1, len(base.split()))
    C_base = max(1, nonspace_len(base))
    div = lev_dist(base.split(), donor.split()) / A
    base_ll = float(char_lm_fn(base))
    base_wlm = np.asarray(word_lm.scores(base), dtype=np.float64)

    if source_order == "fixed_id":
        dist_sources = [bundle["p1"], bundle["p2"]]
    elif source_order == "anchor_first":
        dist_sources = [bundle["p_anchor"], bundle["p_donor"]]
    else:
        raise ValueError(f"Unknown source_order: {source_order}")

    blocks: list[dict] = []
    for lo, hi in ranges:
        phrases = {h: flatten_cells(proj[h][lo:hi]) for h in texts}
        v0 = phrases[base]
        alts = list(dict.fromkeys(phrases.values()))
        if len(alts) < 2:
            continue

        distributions = []
        for weights in dist_sources:
            d: Counter = Counter()
            for h, wt in weights.items():
                d[phrases[h]] += wt
            distributions.append(d)

        xs_all = []
        full_texts = []
        word_risk_votes = []
        char_risk_votes = []
        for v in alts:
            h = splice_cells(cells, [(lo, hi, v)])
            full_texts.append(h)
            ll = float(char_lm_fn(h))
            wlm_delta = np.asarray(word_lm.scores(h), dtype=np.float64) - base_wlm
            feats = [
                div,
                lo / (2 * A + 1),
                len(alts),
                len(v0),
                len(v),
                lev_dist(v0, v) / A,
                len("".join(v)) - len("".join(v0)),
                ll / (len(h) + 1) - base_ll / (len(base) + 1),
                (ll - base_ll) / (len(base) + 1),
                *wlm_delta.tolist(),
            ]
            w_risks = []
            c_risks = []
            v_str = "".join(v)
            v0_str = "".join(v0)
            for d in distributions:
                dw = sum(wt * (lev_dist(v, a) - lev_dist(v0, a)) for a, wt in d.items())
                dc = sum(
                    wt * (lev_dist(v_str, "".join(a)) - lev_dist(v0_str, "".join(a)))
                    for a, wt in d.items()
                )
                feats.extend([d[v], d[v0], dw / A, dc / C_base, len(d), max(d.values())])
                w_risks.append(dw)
                c_risks.append(dc)
            if add_anchor_id_feat:
                feats.append(float(anchor_idx))
            xs_all.append(feats)
            word_risk_votes.append(sum(w_risks))
            char_risk_votes.append(sum(c_risks))

        X_full = np.asarray(xs_all, dtype=np.float64)
        if add_block_context_feats:
            # Intra-block relative margin & rank features (computed over ALL options incl. anchor v0)
            # Key signals: char4_delta_per_char(7), word2_delta_per_tok(9),
            # sum_mass(12+18), sum_word_risk(14+20), sum_char_risk(15+21)
            c4 = X_full[:, 7]
            w2 = X_full[:, 9]
            tot_mass = 0.5 * (X_full[:, 12] + X_full[:, 18])
            tot_wr = 0.5 * (X_full[:, 14] + X_full[:, 20])
            tot_cr = 0.5 * (X_full[:, 15] + X_full[:, 21])
            ctx = np.column_stack([
                c4 - c4.max(),               # gap from best char4 in block
                w2 - w2.max(),               # gap from best word2 in block
                tot_mass - tot_mass.max(),   # gap from highest mass in block
                tot_wr - tot_wr.min(),       # excess word risk over best in block
                tot_cr - tot_cr.min(),       # excess char risk over best in block
                (tot_wr == tot_wr.min()).astype(np.float64),
                (c4 == c4.max()).astype(np.float64),
                np.full(len(alts), np.std(tot_wr)),
            ])
            X_full = np.hstack([X_full, ctx])

        blocks.append({
            "lo": lo,
            "hi": hi,
            "alts": alts,
            "full": full_texts,
            "X": X_full[1:],       # non-anchor alternatives (for baseline compatibility)
            "X_all": X_full,       # including anchor row 0 (for listwise/pairwise models)
            "word_risk": word_risk_votes,
            "char_risk": char_risk_votes,
        })
    return cells, blocks
