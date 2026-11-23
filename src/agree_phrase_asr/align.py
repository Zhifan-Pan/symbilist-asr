"""Candidate list normalization, score collapsing, DP alignment, and 2n+1 anchor projection."""
from __future__ import annotations

from functools import lru_cache
from typing import Callable, Iterable, Sequence
import unicodedata
import numpy as np
from rapidfuzz.distance import Levenshtein


def normalize_text(text: str) -> str:
    """Project-standard NFC -> lower -> NFC -> Unicode punctuation to space -> whitespace collapse."""
    s = unicodedata.normalize("NFC", text).lower()
    s = unicodedata.normalize("NFC", s)
    out = []
    for ch in s:
        cat = unicodedata.category(ch)
        if cat.startswith("P"):
            out.append(" ")
        else:
            out.append(ch)
    return " ".join("".join(out).split())


def collapse_candidates(
    row: dict,
    mode: str = "weighted",
    temperature: float = 0.5,
    shrink: float = 0.0,
    normalize_fn: Callable[[str], str] = normalize_text,
) -> tuple[str, dict[str, float]]:
    """Merge duplicate normalized hypotheses by log-sum-exp and compute list-conditional mass."""
    hyps = [normalize_fn(h) for h in row["hypotheses"]]
    greedy = hyps[row["greedy_index"]]
    if mode == "greedy":
        return greedy, {greedy: 1.0}

    logp: dict[str, float] = {}
    scores = row.get("exact_ctc_logprobs", row.get("logprobs"))
    for h, s in zip(hyps, scores):
        if h in logp:
            logp[h] = float(np.logaddexp(logp[h], s))
        else:
            logp[h] = float(s)

    vals = np.array(list(logp.values()), dtype=np.float64)
    if mode == "uniform" or np.isinf(temperature):
        weights = np.ones_like(vals)
    else:
        shifted = (vals - vals.max()) / temperature
        weights = np.exp(shifted)
    weights = weights / weights.sum()
    dist = dict(zip(logp.keys(), weights.tolist()))
    if shrink > 0.0:
        k = len(dist)
        dist = {h: (1.0 - shrink) * w + shrink / k for h, w in dist.items()}
    return greedy, dist


def dp_word_alignment(a: Sequence[str], b: Sequence[str]) -> list[tuple[str | None, str | None]]:
    """Unit word edit distance alignment with exact tie order: diagonal, deletion, insertion."""
    n, m = len(a), len(b)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    bt = np.zeros((n + 1, m + 1), dtype=np.int8)  # 0=diag, 1=del(up), 2=ins(left)
    for i in range(1, n + 1):
        dp[i, 0] = i
        bt[i, 0] = 1
    for j in range(1, m + 1):
        dp[0, j] = j
        bt[0, j] = 2
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost_diag = dp[i - 1, j - 1] + (0 if ai == b[j - 1] else 1)
            cost_del = dp[i - 1, j] + 1
            cost_ins = dp[i, j - 1] + 1
            best = cost_diag
            op = 0
            if cost_del < best:
                best = cost_del
                op = 1
            if cost_ins < best:
                best = cost_ins
                op = 2
            dp[i, j] = best
            bt[i, j] = op
    pairs: list[tuple[str | None, str | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        op = bt[i, j]
        if op == 0:
            pairs.append((a[i - 1], b[j - 1]))
            i -= 1
            j -= 1
        elif op == 1:
            pairs.append((a[i - 1], None))
            i -= 1
        else:
            pairs.append((None, b[j - 1]))
            j -= 1
    pairs.reverse()
    return pairs


def make_projector(align_fn: Callable = dp_word_alignment):
    @lru_cache(maxsize=200000)
    def project(base: str, text: str) -> tuple[tuple[str, ...], ...]:
        words = base.split()
        cells: list[list[str]] = [[] for _ in range(2 * len(words) + 1)]
        if text == base:
            for idx, w in enumerate(words):
                cells[2 * idx + 1].append(w)
        else:
            i = 0
            for left, right in align_fn(words, text.split()):
                if left is None:
                    cells[2 * i].append(right)
                else:
                    if right is not None:
                        cells[2 * i + 1].append(right)
                    i += 1
            assert i == len(words), f"Alignment mismatch: {i} != {len(words)}"
        res = tuple(tuple(c) for c in cells)
        return res

    return project


def flatten_cells(cells: Sequence[tuple[str, ...]]) -> tuple[str, ...]:
    return tuple(w for cell in cells for w in cell)


def splice_cells(
    cells: Sequence[tuple[str, ...]],
    replacements: Sequence[tuple[int, int, tuple[str, ...]]],
) -> str:
    out: list[str] = []
    last = 0
    for lo, hi, phrase in replacements:
        out.extend(flatten_cells(cells[last:lo]))
        out.extend(phrase)
        last = hi
    out.extend(flatten_cells(cells[last:]))
    return " ".join(out)


def lev_dist(a: Sequence, b: Sequence) -> int:
    return int(Levenshtein.distance(a, b))


def build_projection_bundle(
    row1: dict,
    row2: dict,
    anchor_idx: int = 0,
    mode: str = "weighted",
    temperature: float = 0.5,
    shrink: float = 0.0,
    normalize_fn: Callable[[str], str] = normalize_text,
    project_fn: Callable | None = None,
) -> dict:
    """Build anchor-aligned projection bundle for two scored candidate lists.

    anchor_idx=0 uses source 1 (Omni) as anchor; anchor_idx=1 uses source 2 (MMS) as anchor.
    Always stores p1 (source 1) and p2 (source 2) alongside anchor-relative references.
    """
    if project_fn is None:
        project_fn = make_projector()
    g1, p1 = collapse_candidates(row1, mode=mode, temperature=temperature, shrink=shrink, normalize_fn=normalize_fn)
    g2, p2 = collapse_candidates(row2, mode=mode, temperature=temperature, shrink=shrink, normalize_fn=normalize_fn)

    if anchor_idx == 0:
        base, donor = g1, g2
        p_anchor, p_donor = p1, p2
    else:
        base, donor = g2, g1
        p_anchor, p_donor = p2, p1

    texts = list(dict.fromkeys([base, *p_anchor.keys(), *p_donor.keys()]))
    proj = {h: project_fn(base, h) for h in texts}
    return {
        "anchor_idx": anchor_idx,
        "base": base,
        "donor": donor,
        "g1": g1,
        "g2": g2,
        "p1": p1,
        "p2": p2,
        "p_anchor": p_anchor,
        "p_donor": p_donor,
        "texts": texts,
        "proj": proj,
        "cells": proj[base],
    }
