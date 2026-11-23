"""Unit tests for core invariants: alignment invertibility, partition containment H_whole subseteq H_part, symmetry, and metric aggregation."""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agree_phrase_asr.align import (
    normalize_text,
    dp_word_alignment,
    make_projector,
    flatten_cells,
    splice_cells,
    build_projection_bundle,
)
from agree_phrase_asr.partition import (
    agreement_partition,
    soft_agreement_partition,
    fixed_partition,
    random_length_matched_partition,
    random_varcount_matched_partition,
    shifted_agreement_partition,
    whole_partition,
    validate_partition,
)
from agree_phrase_asr.lm import CharNgramLM, WordBigramLM
from agree_phrase_asr.features import build_phrase_graph, BASE_FEATURE_NAMES
from agree_phrase_asr.eval import edit_counts, evaluate_predictions


def _sample_rows():
    r1 = {
        "hypotheses": ["two red birds sing", "two red birds sang", "three red birds sing"],
        "greedy_index": 0,
        "exact_ctc_logprobs": [-1.0, -2.5, -3.2],
    }
    r2 = {
        "hypotheses": ["three red birds fly", "two red birds fly"],
        "greedy_index": 0,
        "exact_ctc_logprobs": [-1.2, -1.8],
    }
    return r1, r2


def test_normalization_preserves_combining_marks():
    assert normalize_text("Hello, World!  It's 2026.") == "hello world it s 2026"
    assert normalize_text("தமிழ், கன்னடம்!") == "தமிழ் கன்னடம்"


def test_projection_exact_reconstruction():
    proj = make_projector(dp_word_alignment)
    base = "two red birds sing"
    for cand in ["two red birds sing", "three red birds fly", "red birds", "a two big red birds sing loudly", ""]:
        cells = proj(base, cand)
        assert len(cells) == 2 * len(base.split()) + 1
        assert " ".join(flatten_cells(cells)) == cand


def test_all_partitions_preserve_whole_containment():
    r1, r2 = _sample_rows()
    proj = make_projector()
    for anchor_idx in (0, 1):
        b = build_projection_bundle(r1, r2, anchor_idx=anchor_idx, project_fn=proj)
        for part in [
            agreement_partition(b),
            whole_partition(b),
            fixed_partition(b, 3),
            random_length_matched_partition(b, "utt_1", 101),
            random_varcount_matched_partition(b, "utt_1", 101),
            shifted_agreement_partition(b, "utt_1", 101, shift=2),
            soft_agreement_partition(b, tau=0.85, mode="min_source"),
            soft_agreement_partition(b, tau=0.85, mode="mean_source"),
            soft_agreement_partition(b, tau=1.0, mode="greedy_agree"),
        ]:
            validate_partition(b, part)


def test_agreement_phrase_space_admits_novel_recombination():
    r1, r2 = _sample_rows()
    b = build_projection_bundle(r1, r2, anchor_idx=0)
    part = agreement_partition(b)
    # "red" and "birds" are unanimous anchor words -> cells 3 and 5 are invariant
    # Block before "red" has {"two", "three"}, block after "birds" has {"sing", "sang", "fly"}
    c4 = CharNgramLM(order=4).fit(["two red birds fly", "three red birds sing"])
    w2 = WordBigramLM().fit(["two red birds fly", "three red birds sing"])
    cells, blocks = build_phrase_graph(b, part, c4.score, w2, add_block_context_feats=True)
    assert len(blocks) == 2
    assert blocks[0]["X"].shape[1] == len(BASE_FEATURE_NAMES) + 8
    # Recombining v0=("two",) in block 0 and ("fly",) in block 1 yields "two red birds fly"
    recombined = splice_cells(cells, [(blocks[0]["lo"], blocks[0]["hi"], ("two",)), (blocks[1]["lo"], blocks[1]["hi"], ("fly",))])
    assert recombined == "two red birds fly"


def test_soft_agreement_tau1_matches_strict_agreement():
    r1, r2 = _sample_rows()
    b = build_projection_bundle(r1, r2, anchor_idx=0)
    assert soft_agreement_partition(b, tau=1.0, mode="min_source") == agreement_partition(b)


def test_macro_wer_cer_exactness():
    meta = {
        "u1": {"language": "L1", "reference": "a b c d"},
        "u2": {"language": "L2", "reference": "x y"},
    }
    preds = {"u1": "a b c e", "u2": "x y"}
    ev = evaluate_predictions(preds, meta)
    assert np.isclose(ev["per_lang"]["L1"]["WER"], 25.0)
    assert np.isclose(ev["per_lang"]["L2"]["WER"], 0.0)
    assert np.isclose(ev["macro_WER"], 12.5)
    assert np.isclose(ev["micro_WER"], 100.0 / 6.0)
