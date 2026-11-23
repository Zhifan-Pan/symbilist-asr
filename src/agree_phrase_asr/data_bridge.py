"""Read-only bridge to frozen FLEURS-12 dev (4,078) and Kathbath valid (4,920) data, folds, and LMs."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

FROZEN_BASE = Path(os.environ.get("SYMBILIST_BUNDLE", Path(__file__).resolve().parents[2] / "local-data" / "frozen")).expanduser().resolve()
SEEDS = (17, 42, 2026)


def _ensure_frozen_imports():
    scripts_dir = str(FROZEN_BASE / "scripts")
    if not Path(scripts_dir, "boundaries.py").is_file():
        raise FileNotFoundError("Set SYMBILIST_BUNDLE to a compatible local frozen research bundle; see docs/reproducibility.md.")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import boundaries as fb  # type: ignore
    return fb


def read_jsonl(path: Path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def load_frozen_dev_bundle() -> dict[str, Any]:
    """Load exact frozen FLEURS dev (4,078), Kathbath valid (4,920), folds, metadata, and LMs.

    Strictly development/validation scopes only; never loads locked official test references.
    """
    fb = _ensure_frozen_imports()
    tr, ext = fb.native()
    rr = fb.original_labels()

    # Build unified metadata: uid -> dict(dataset, language, cluster, reference)
    meta: dict[str, dict[str, Any]] = {}
    for u, r in rr.items():
        meta[u] = {
            "dataset": "FLEURS",
            "language": r["language"],
            "cluster": r["group"],
            "reference": r["reference"],
        }
    for r in read_jsonl(fb.ART / "roster.jsonl.gz"):
        u = r["recording_uid"]
        assert u not in meta and r["official_split"] == "valid"
        meta[u] = {
            "dataset": "Kathbath",
            "language": r["language"],
            "cluster": f"{r['language']}:{r['speaker_id']}",
            "reference": fb.ex.old.ev.normalize_text(r["reference"]),
        }

    folds_by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    for seed in SEEDS:
        fdict = {r["uid"]: r for r in read_jsonl(fb.rootrun(seed) / "outer_inner_memberships.jsonl")}
        folds_by_seed[seed] = fdict

    word_lm = fb.prior.WordLM()
    char_lm_fn = fb.old.lmscore
    normalize_fn = fb.old.ev.normalize_text
    align_fn = fb.old.local.alignment

    # Load frozen baseline sealed predictions for bit-exact verification
    baseline_preds: dict[int, dict[str, dict[str, str]]] = {}
    for seed in SEEDS:
        p = fb.rootrun(seed) / "CTC_phrase" / "sealed.jsonl.gz"
        baseline_preds[seed] = {r["uid"]: r["hypotheses"] for r in read_jsonl(p)}

    return {
        "fleurs": tr,
        "kathbath": ext,
        "all_data": tr | ext,
        "fleurs_refs": rr,
        "meta": meta,
        "folds_by_seed": folds_by_seed,
        "word_lm": word_lm,
        "char_lm_fn": char_lm_fn,
        "normalize_fn": normalize_fn,
        "align_fn": align_fn,
        "baseline_preds": baseline_preds,
        "fleurs_uids": sorted(tr.keys()),
        "kathbath_uids": sorted(ext.keys()),
    }


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()
