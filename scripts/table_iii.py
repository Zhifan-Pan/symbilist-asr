#!/usr/bin/env python3
"""Render the two-encoder test comparison from the archived aggregate JSONs."""
import argparse
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
FL = "fp_fleurs12_official_test"
KB = "testunk_complete_table3"
KCONF = "kathbath_original_readout"
MBR = "asr_rigor_upgrade"
FQ = "p1_fleurs12_test_q17"
KC = "p1_symbilist_crc"
KQ = "rp1_symbilist_qwen17b_and_sem"
KWHOLEQ = "p1_nlm_qwen3_1_7b"


def metrics(file, path, wer_key="macro_WER", cer_key="macro_CER"):
    return file, path.split("/") if path else [], [wer_key, cer_key]


def fleurs(system):
    return metrics(FL, "systems/" + system)


def kath(system):
    return metrics(KB, system)


def mbr(scope):
    return metrics(MBR, "unsupervised_block_mbr_char4/" + scope,
                   "Block_MBR_Char4_WER", "Block_MBR_Char4_CER")


# Groups mark comparable compute budgets; oracle rows use the references.
ROWS = [
    ("Single", "OmniASR (greedy)", fleurs("OmniASR_1B_v2"), metrics(KCONF, "metrics/anchor")),
    ("Single", "MMS (greedy)", fleurs("MMS_1B_all"), metrics(KCONF, "metrics/donor")),
    ("Unsupervised", "Hystoc--ROVER", fleurs("Hystoc_ROVER"), kath("Hystoc_ROVER")),
    ("Unsupervised", "Block MBR + char LM", mbr("FLEURS12_test"), mbr("Kathbath_test")),
    ("GPU", "Qwen3-8B GER", metrics("p1_fleurs12_test_q8b", ""), metrics("rp1_qwen8b_kathbath", "Kathbath_testunk")),
    ("GPU", "Whole + 1.7B + n-grams", metrics(FQ, "systems/Whole_Listwise_Qwen3_1.7B_plus_ngrams"),
     metrics(KWHOLEQ, "FLEURS_Kathbath/Whole_Listwise_Qwen3_1.7B_plus_ngrams/Kathbath_testunk")),
    ("GPU", r"\SymBiList{} + 1.7B", metrics(FQ, "systems/SymBiList_Qwen3_1.7B"),
     metrics(KQ, "Phase1_SymBiList_Qwen3_1.7B/Kathbath_testunk/SymBiList_Qwen3_1.7B", "WER", "CER")),
    ("Control", "Bidirectional Whole", fleurs("Whole_sym_Listwise"), metrics(KC, "FLEURS_Kathbath/Kathbath_testunk/Whole_sym")),
    ("Control", "Fixed length ($L=3$)", fleurs("Fixed_L3_HGB24"), kath("Fixed_L3")),
    ("Phrase", "Agree A (HGB)", fleurs("Agree_Phrase_A_Omni_HGB24"), kath("Agree_A_check")),
    ("Phrase", r"\SymBiPhrase{} (HGB)", fleurs("SymBiPhrase_HGB24"), kath("SymBiPhrase_HGB_24d")),
    ("Phrase", r"\SymBiList{} (ours)", fleurs("SymBiList_frozen"), metrics("testunk_postlock_symbilist", "systems/SymBiList_phrase")),
    ("Phrase", r"\SymBiList{} + \crc{} (.06)", fleurs("SymBiList_CRC_a0.06"), metrics(KC, "FLEURS_Kathbath/Kathbath_testunk/CRC_a0.06")),
    ("Oracle", "Whole oracle", fleurs("Oracle_H_whole"), kath("Oracle_Whole")),
    ("Oracle", "Phrase oracle", fleurs("Oracle_H_agree"), kath("Oracle_Agree")),
]


def build(inputs, output, provenance):
    files = {}
    manifest = {}
    for _, _, *sources in ROWS:
        for name, _, _ in sources:
            if name not in files:
                path = inputs / (name + ".json")
                data = path.read_bytes()
                files[name] = json.loads(data)
                manifest[name + ".json"] = hashlib.sha256(data).hexdigest()

    rows = []
    for group, label, *sources in ROWS:
        rates = []
        cells = []
        for name, path, keys in sources:
            record = files[name]
            for k in path:
                record = record[k]
            for key in keys:
                value = record[key]
                assert math.isfinite(value) and value >= 0
                rates.append(value)
                cells.append({"file": name + ".json", "path": [*path, key], "rate": value})
        rows.append((group, label, rates, cells))

    overall = [min(r[2][j] for r in rows if r[0] != "Oracle") for j in range(4)]
    cpu = [min(r[2][j] for r in rows if r[0] not in ("GPU", "Oracle")) for j in range(4)]
    lines = [r"\begin{table}[t]", r"\caption{OmniASR + MMS on the full test splits. Language-macro WER and CER (\%, $\downarrow$). Bold and underline mark minima among CPU and all deployable systems, respectively. Oracles minimize each metric separately.}\label{tab:main}",
             r"\centering\footnotesize\setlength{\tabcolsep}{1.6pt}",
             r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lcccc@{}}\toprule",
             r"\multirow{2}{*}{Method} & \multicolumn{2}{c}{FLEURS-12} & \multicolumn{2}{c}{Kathbath}\\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r"& WER & CER & WER & CER\\\midrule"]
    previous = rows[0][0]
    for group, label, values, _ in rows:
        if group != previous:
            lines.append(r"\midrule")
        previous = group
        cols = []
        for j, val in enumerate(values):
            cell = f"{val:.3f}"
            if group != "Oracle" and val == cpu[j]:
                cell = r"\textbf{" + cell + "}"
            if group != "Oracle" and val == overall[j]:
                cell = r"\underline{" + cell + "}"
            cols.append(cell)
        lines.append(" & ".join([label, *cols]) + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular*}", r"\end{table}"])
    output.write_text("\n".join(lines) + "\n")
    provenance.write_text(json.dumps({
        "notes": ["All reported rates are full-split language macro averages in percent.",
                  "GPU rows use Qwen models. Single and oracle rows retain their original input budgets.",
                  "The Kathbath single-source controls come from kathbath_confirm_20261120/results/RESULTS.json.",
                  "All other JSON files originate from research_upgrade_20261123/results with their original filenames."],
        "files_sha256": manifest,
        "columns": ["FLEURS12_test_WER", "FLEURS12_test_CER", "Kathbath_testunk_WER", "Kathbath_testunk_CER"],
        "cells": {r[1]: r[3] for r in rows}
    }, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, default=HERE / "inputs")
    parser.add_argument("--output", type=Path, default=HERE / "iii.tex")
    parser.add_argument("--provenance", type=Path, default=HERE / "iii_sources.json")
    args = parser.parse_args()
    build(args.inputs, args.output, args.provenance)
