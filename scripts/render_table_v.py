"""Official-test component ablations (incl. param-free Pass-2 & no-LM control) and noise stress tests."""
from pathlib import Path
import argparse
import json


def records(directory, stem):
    return json.loads((directory / f'{stem}.json').read_text())


def group_header(label, left, right):
    return [
        r'\multirow{2}{*}{\textbf{' + label + r'}} & \multicolumn{2}{c}{\textbf{' + left
        + r'}} & \multicolumn{2}{c}{\textbf{' + right + r'}}\\',
        r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',
        r'& WER & CER & WER & CER\\\midrule',
    ]


def fmt_best_rows(entries):
    mins = [min(r[1][j] for r in entries) for j in range(4)]
    out = []
    for label, vals in entries:
        cells = [
            (r'\textbf{' + f'{v:.3f}' + '}' if v == mins[j] else f'{v:.3f}')
            for j, v in enumerate(vals)
        ]
        out.append(' & '.join([label, *cells]) + r'\\')
    return out


def render(directory):
    fl = records(directory, 'fp_fleurs12_official_test')['systems']
    kb = records(directory, 'testunk_postlock_symbilist')['systems']
    kb_crc = records(directory, 'p1_symbilist_crc')['FLEURS_Kathbath']['Kathbath_testunk']
    enh = records(directory, 'solid_enhancement_checks')
    e2 = enh['E2_paramfree_and_asymmetry']
    e3 = enh['E3_no_external_LM_control']
    noise = records(directory, 'cscwd_noise_robustness_1200')
    assert noise['n_utts'] == 1200

    rows = [
        r'\begin{table}[t]',
        r'\caption{Frozen two-source ablations on the full test splits (top; bold marks minima) and noise stress test on 1,200 FLEURS-12 test clips (bottom). Macro rates (\%, $\downarrow$).}\label{tab:ablation}',
        r'\centering\footnotesize\setlength{\tabcolsep}{1.8pt}',
        r'\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lcccc@{}}\toprule',
        *group_header('Variant', 'FLEURS-12', 'Kathbath'),
    ]

    arms = [
        ('Omni anchor only', [fl['SymBiList_Pass1_Omni']['macro_WER'], fl['SymBiList_Pass1_Omni']['macro_CER'],
                              kb['Listwise_dir1_omni']['macro_WER'], kb['Listwise_dir1_omni']['macro_CER']]),
        ('MMS anchor only', [fl['SymBiList_Pass1_MMS']['macro_WER'], fl['SymBiList_Pass1_MMS']['macro_CER'],
                             kb['Listwise_dir2_mms']['macro_WER'], kb['Listwise_dir2_mms']['macro_CER']]),
        ('Bidirectional, learned Pass-2', [fl['SymBiList_frozen']['macro_WER'], fl['SymBiList_frozen']['macro_CER'],
                                           kb['SymBiList_phrase']['macro_WER'], kb['SymBiList_phrase']['macro_CER']]),
        (r'\quad + \crc{} ($\alpha=0.04$)', [fl['SymBiList_CRC_a0.04']['macro_WER'], fl['SymBiList_CRC_a0.04']['macro_CER'],
                                             kb_crc['CRC_a0.04']['macro_WER'], kb_crc['CRC_a0.04']['macro_CER']]),
        (r'\quad + \crc{} ($\alpha=0.06$)', [fl['SymBiList_CRC_a0.06']['macro_WER'], fl['SymBiList_CRC_a0.06']['macro_CER'],
                                             kb_crc['CRC_a0.06']['macro_WER'], kb_crc['CRC_a0.06']['macro_CER']]),
        (r'\quad + \crc{} ($\alpha=0.08$)', [fl['SymBiList_CRC_a0.08']['macro_WER'], fl['SymBiList_CRC_a0.08']['macro_CER'],
                                             kb_crc['CRC_a0.08']['macro_WER'], kb_crc['CRC_a0.08']['macro_CER']]),
        ('Bidirectional, param-free Pass-2', [e2['FLEURS12_test']['Bidirectional_paramfree_phrase_RiskOnly']['WER'],
                                              e2['FLEURS12_test']['Bidirectional_paramfree_phrase_RiskOnly']['CER'],
                                              e2['Kathbath_testunk']['Bidirectional_paramfree_phrase_RiskOnly']['WER'],
                                              e2['Kathbath_testunk']['Bidirectional_paramfree_phrase_RiskOnly']['CER']]),
    ]
    rows.extend(fmt_best_rows(arms))

    # Zero-external-text-LM decoupled control sub-block
    rows.append(r'\midrule')
    rows.append(r'\multicolumn{5}{@{}l}{\emph{Zero external text LMs (24 acoustic/geometry features only)}}\\')
    nolm_arms = [
        ('Bidirectional Whole (no LM)', [e3['FLEURS12_test']['Bidirectional_Whole_noLM']['WER'],
                                         e3['FLEURS12_test']['Bidirectional_Whole_noLM']['CER'],
                                         e3['Kathbath_testunk']['Bidirectional_Whole_noLM']['WER'],
                                         e3['Kathbath_testunk']['Bidirectional_Whole_noLM']['CER']]),
        (r'\SymBiList{} (no LM)', [e3['FLEURS12_test']['SymBiList_noLM']['WER'],
                                   e3['FLEURS12_test']['SymBiList_noLM']['CER'],
                                   e3['Kathbath_testunk']['SymBiList_noLM']['WER'],
                                   e3['Kathbath_testunk']['SymBiList_noLM']['CER']]),
    ]
    rows.extend(fmt_best_rows(nolm_arms))

    rows.extend([r'\midrule', *group_header('Noise stress test', 'Bidirectional Whole', r'\SymBiList{}')])
    for key, label in [('clean', 'Clean'), ('snr_15db', 'Babble + pink, 15 dB'),
                       ('snr_10db', 'Babble + pink, 10 dB'), ('snr_5db', 'Babble + pink, 5 dB')]:
        rates = noise['conditions'][key]['systems']
        w_wer, w_cer = rates['Whole_Listwise']['WER'], rates['Whole_Listwise']['CER']
        s_wer, s_cer = rates['SymBiList']['WER'], rates['SymBiList']['CER']
        rows.append(
            f"{label} & {w_wer:.3f} & {w_cer:.3f} & "
            f"\\textbf{{{s_wer:.3f}}} & \\textbf{{{s_cer:.3f}}}\\\\"
        )
    rows.extend([r'\bottomrule\end{tabular*}', r'\end{table}'])
    return '\n'.join(rows) + '\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(render(args.results))
