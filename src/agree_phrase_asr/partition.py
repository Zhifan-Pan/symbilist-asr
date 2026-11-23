"""Partition policies over 2n+1 anchor projection cells.

Every partition strictly covers [0, 2n+1) without gaps or overlaps, guaranteeing
that every original list member is exactly reconstructible (H_whole subseteq H_part).
"""
from __future__ import annotations

import hashlib
import numpy as np
from .align import flatten_cells


def validate_partition(bundle: dict, ranges: list[tuple[int, int]]) -> None:
    n_cells = len(bundle["cells"])
    assert ranges and ranges[0][0] == 0 and ranges[-1][1] == n_cells
    for k, (lo, hi) in enumerate(ranges):
        assert lo < hi
        if k + 1 < len(ranges):
            assert hi == ranges[k + 1][0]
    for h, z in bundle["proj"].items():
        rebuilt = " ".join(w for lo, hi in ranges for w in flatten_cells(z[lo:hi]))
        assert rebuilt == h, f"Reconstruction failed: {rebuilt!r} != {h!r}"


def agreement_partition(bundle: dict) -> list[tuple[int, int]]:
    """Original strict unanimous agreement partition (A)."""
    cells = bundle["cells"]
    proj_vals = list(bundle["proj"].values())
    ranges: list[tuple[int, int]] = []
    lo = 0
    for i, cell in enumerate(cells):
        if i % 2 == 1 and all(z[i] == cell for z in proj_vals):
            if i > lo:
                ranges.append((lo, i))
            ranges.append((i, i + 1))
            lo = i + 1
    if lo < len(cells):
        ranges.append((lo, len(cells)))
    return ranges


def soft_agreement_partition(
    bundle: dict,
    tau: float = 0.90,
    mode: str = "min_source",
) -> list[tuple[int, int]]:
    """Probability-weighted consensus boundary partition (Direction A2).

    Places block boundaries around anchor word cell i (odd index) whenever
    consensus mass >= tau, while STILL retaining all observed phrases in (i, i+1)
    if a rare tail hypothesis disagreed. Strictly guarantees H_whole subseteq H_tau,
    and reduces identically to strict agreement when tau = 1.0 (with positive masses).

    Modes:
      - 'min_source': min(P_1(cell_i == anchor_i), P_2(cell_i == anchor_i)) >= tau
      - 'mean_source': 0.5 * (P_1 + P_2) >= tau
      - 'greedy_agree': both greedy hypotheses agree on anchor_i (equivalent to anchor-donor greedy consensus)
    """
    cells = bundle["cells"]
    proj = bundle["proj"]
    p1 = bundle["p1"]
    p2 = bundle["p2"]
    g1 = bundle["g1"]
    g2 = bundle["g2"]

    ranges: list[tuple[int, int]] = []
    lo = 0
    for i, cell in enumerate(cells):
        if i % 2 == 1:
            is_anchor_cut = False
            if mode == "greedy_agree":
                is_anchor_cut = (proj[g1][i] == cell) and (proj[g2][i] == cell)
            else:
                m1 = sum(w for h, w in p1.items() if proj[h][i] == cell)
                m2 = sum(w for h, w in p2.items() if proj[h][i] == cell)
                score = min(m1, m2) if mode == "min_source" else 0.5 * (m1 + m2)
                if tau >= 1.0:
                    is_anchor_cut = all(z[i] == cell for z in proj.values())
                else:
                    is_anchor_cut = score >= tau - 1e-12
            if is_anchor_cut:
                if i > lo:
                    ranges.append((lo, i))
                ranges.append((i, i + 1))
                lo = i + 1
    if lo < len(cells):
        ranges.append((lo, len(cells)))
    return ranges


def whole_partition(bundle: dict) -> list[tuple[int, int]]:
    return [(0, len(bundle["cells"]))]


def fixed_partition(bundle: dict, block_len: int = 3) -> list[tuple[int, int]]:
    n = len(bundle["cells"])
    return [(i, min(n, i + block_len)) for i in range(0, n, block_len)]


def random_length_matched_partition(bundle: dict, uid: str, seed: int) -> list[tuple[int, int]]:
    """Original Q control: permutes all agreement block lengths (including invariant 1-cells)."""
    a = agreement_partition(bundle)
    sizes = [hi - lo for lo, hi in a]
    key = f"random_length_matched|{seed}|{uid}".encode("utf-8")
    rng = np.random.Generator(np.random.PCG64(int.from_bytes(hashlib.sha256(key).digest(), "big")))
    rng.shuffle(sizes)
    ends = np.cumsum([0, *sizes]).tolist()
    return list(zip(ends, ends[1:]))


def random_varcount_matched_partition(bundle: dict, uid: str, seed: int) -> list[tuple[int, int]]:
    """Decision-count-matched random partition (Q_var): isolates pure boundary position from block count.

    Counts the number K of variable blocks in A for this utterance, then places exactly
    K-1 random cut points among cells [1, n_cells-1] so the partition has at most K
    contiguous blocks (closely matching variable decision count without cutting at unanimous words).
    """
    a = agreement_partition(bundle)
    proj_vals = list(bundle["proj"].values())
    k_var = sum(
        1 for lo, hi in a if len({flatten_cells(z[lo:hi]) for z in proj_vals}) > 1
    )
    n = len(bundle["cells"])
    if k_var <= 1 or n <= 1:
        return [(0, n)]
    k_cuts = min(k_var - 1, n - 1)
    key = f"random_varcount_matched|{seed}|{uid}".encode("utf-8")
    rng = np.random.Generator(np.random.PCG64(int.from_bytes(hashlib.sha256(key).digest(), "big")))
    cuts = sorted(rng.choice(np.arange(1, n), size=k_cuts, replace=False).tolist())
    pts = [0, *cuts, n]
    return list(zip(pts[:-1], pts[1:]))


def shifted_agreement_partition(bundle: dict, uid: str, seed: int, shift: int = 2) -> list[tuple[int, int]]:
    """Boundary-shift control (Q_shift): shifts every unanimous anchor cut by +/- shift cells.

    Preserves the exact topology and number of cut points of A while moving boundaries off
    the unanimous anchor words.
    """
    a = agreement_partition(bundle)
    n = len(bundle["cells"])
    internal_cuts = sorted({lo for lo, _ in a if 0 < lo < n})
    if not internal_cuts:
        return [(0, n)]
    key = f"shifted_agreement|{shift}|{seed}|{uid}".encode("utf-8")
    rng = np.random.Generator(np.random.PCG64(int.from_bytes(hashlib.sha256(key).digest(), "big")))
    new_cuts = set()
    for c in internal_cuts:
        delta = int(rng.choice([-shift, shift]))
        nc = min(max(1, c + delta), n - 1)
        new_cuts.add(nc)
    pts = [0, *sorted(new_cuts), n]
    return list(zip(pts[:-1], pts[1:]))
