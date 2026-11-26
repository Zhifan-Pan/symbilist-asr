"""Generate cross-architecture and matched three-source results from archived metrics."""
import argparse
import json
from pathlib import Path


def render(results_dir):
    def load(stem):
        return json.loads((results_dir / (stem + ".json")).read_text())

    def cv(row, prefix=""):
        return tuple(row[subset][prefix + metric]
                     for subset in ("all8", "unseen7") for metric in ("WER", "CER"))

    def three_source(row_fl, row_kb):
        return tuple(row[metric] for row in (row_fl, row_kb) for metric in ("WER", "CER"))

    def row(name, values, best=()):
        def cell(index, value):
            text = f"{value:.3f}"
            return r"\textbf{" + text + "}" if index in best else text
        return " & ".join([name, *(cell(i, value) for i, value in enumerate(values))]) + r"\\"

    def best_in_panel(rows):
        minimum = tuple(map(min, zip(*(values for _, values in rows))))
        return [row(name, values, [i for i in range(4) if values[i] == minimum[i]])
                for name, values in rows]

    def panel_header(name, left, right):
        if r"\\" in name:
            left_cell = r"\multirow{2}{*}{\shortstack[c]{" + name + "}}"
        else:
            left_cell = r"\multirow{2}{*}[-0.3em]{" + name + "}"
        return [
            left_cell + " & " +
            r"\multicolumn{2}{c}{\textbf{" + left + "}} & " +
            r"\multicolumn{2}{c}{\textbf{" + right + r"}}\\",
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
            r" & WER & CER & WER & CER\\\midrule",
        ]

    def subhead(text):
        return r"\multicolumn{5}{@{}l}{\emph{" + text + r"}}\\"

    cv_all = load("stage2_cv26_cross_arch")
    z3 = load("p2_cv26_z3_3seed")["summary_3seed"]
    upper_zero = [
        (r"OmniASR-CTC", cv(cv_all["single_and_unsupervised"]["OmniASR_CTC_greedy"], "macro_")),
        (r"Whisper-AED", cv(cv_all["single_and_unsupervised"]["Whisper_v3t_AED_greedy"], "macro_")),
        (r"Qwen3-8B GER", cv(load("p1_qwen8b_fair_cv26"), "macro_")),
        (r"Whole (Z1, CPU)", cv(cv_all["Z1_zero_shot_lang_agnostic"]["Z1_Whole_fixed_Listwise"], "macro_")),
        (r"\SymBiList{} (Z1)", cv(cv_all["Z1_zero_shot_lang_agnostic"]["Z1_SymBiList_ZeroShot"], "macro_")),
    ]
    upper_adapted = [
        (name, cv(z3[key])) for name, key in [
            ("Whole + Qwen3-1.7B", "Whole_Listwise_Qwen3_1.7B"),
            ("Whole (CPU)", "Listwise_Whole_CPU"),
            (r"\SymBiPhrase{} (CPU)", "SymBiPhrase_HGB"),
            (r"\SymBiList{} (CPU)", "SymBiList_unconstrained"),
            (r"\SymBiList{} + \crc{} (CPU)", "SymBiList_CRC_a0.01_CPU"),
            (r"\SymBiList{} + 1.7B + \crc{}", "SymBiList_Qwen3_1.7B_CRC_a0.01"),
            (r"\quad + repetition guard", "SymBiList_Qwen3_1.7B_CRC_a0.01_RepGuard"),
        ]
    ]

    # Unified single matched 3-source run with fair Bidirectional Whole (3-source)
    e1 = load("solid_enhancement_checks")["E1_3way_fair_bidirectional_whole"]["modes"]
    fl_ms = load("cscwd_3way_fleurs12_test")["modes"]["mass_soft"]["SymBiList"]
    kb_ms = load("fp_3way_kathbath_test")["modes"]["mass_soft"]["SymBiList"]
    three_rows = [
        ("Bidirectional Whole (3-src)", three_source(e1["quorum_2of3"]["FLEURS_Whole_Bidirectional"],
                                                      e1["quorum_2of3"]["Kathbath_Whole_Bidirectional"])),
        ("Unanimity (3/3)", three_source(e1["strict_3of3"]["FLEURS_SymBiList"],
                                         e1["strict_3of3"]["Kathbath_SymBiList"])),
        ("Majority, hard lock (2/3)", three_source(e1["quorum_hard_lock"]["FLEURS_SymBiList"],
                                                   e1["quorum_hard_lock"]["Kathbath_SymBiList"])),
        ("Majority, selectable (ours)", three_source(e1["quorum_2of3"]["FLEURS_SymBiList"],
                                                     e1["quorum_2of3"]["Kathbath_SymBiList"])),
        (r"Mass-soft ($\tau=0.65$)", three_source(fl_ms, kb_ms)),
    ]

    lines = [
        r"\begin{table}[t]",
        r"\caption{Two-source CV26 evaluation and matched three-source test results. "
        r"Language-macro WER and CER (\%, $\downarrow$). Bold marks the minimum within each group, excluding the post-hoc guard.}"
        r"\label{tab:stage2}",
        r"\centering\footnotesize\setlength{\tabcolsep}{1.4pt}",
        r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lcccc@{}}\toprule",
        *panel_header(r"Method\\(OmniASR + Whisper)", "CV26 (all 8)", "CV26 (other 7)"),
        subhead("No CV26 training labels"),
        *best_in_panel(upper_zero), r"\midrule",
        subhead("Z3, mean over three recording-level five-fold splits"),
        *best_in_panel(upper_adapted[:-1]), row(*upper_adapted[-1]),
        r"\midrule",
        *panel_header("Three-source fusion", "FLEURS-12", "Kathbath"),
        subhead("Matched OmniASR + MMS + Whisper official tests"),
        *best_in_panel(three_rows),
        r"\bottomrule\end{tabular*}",
        r"\par\smallskip\raggedright\footnotesize "
        r"The seven-language subset excludes Tamil, the only CV26 language in the FLEURS fit. "
        r"CV26 \crc{} uses $\alpha=0.01$; speaker-disjoint Z3 check is in Sec.~\ref{sec:cross}.",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(render(args.results), encoding="utf-8")
