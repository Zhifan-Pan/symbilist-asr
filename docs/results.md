# Results index

The files below contain stored aggregate results from the experiments reported in the paper. Consult the corresponding table captions for populations, units, baselines and uncertainty. Lower WER and CER mean fewer edits relative to the reference under the stated normalization.

| Current paper | Source |
|---|---|
| Table I: two-recognizer FLEURS-12 and Kathbath test results | [Table generator](../scripts/table_iii.py); [file, JSON key and full-precision rate for each cell](table_iii_sources.json) |
| Table II: Common Voice 26 and matched three-recognizer fusion | [Table generator](../scripts/render_table_iv.py); [matched comparisons](../results/solid_enhancement_checks.json) |
| Table III: ablations and noise stress test | [Table generator](../scripts/render_table_v.py); [noise conditions](../results/cscwd_noise_robustness_1200.json) |
| Figure 2: language differences, latency and deferral | [Figure generator](../scripts/generate_figures.py); [per-language effects](../results/asr_rigor_upgrade.json), [deferral](../results/p1_symbilist_crc.json) and [cascade](../results/cscwd_cascade_and_q17_ablation.json) |

[Word errors by whole-list membership](novel_vs_whole.json) distinguish the fixed and bidirectional whole-list comparators. The [seed record](seeds.json) distinguishes development partitions from the frozen test predictions. The public CPU analysis and CER regression summaries retain their aggregate metrics; the per-record qualitative examples and top-ten transcript list are omitted.

| Setting or analysis | Files |
|---|---|
| FLEURS-12 and Kathbath validation: ranking, anchor orientation, deferral and ablations | [Baselines](../results/stage1_baselines_and_ablations.json), [anchor controls](../results/stage1_direction_a.json), [ranking and deferral](../results/stage1_direction_b.json), [SymBiList](../results/stage1_symbilist.json) |
| Earlier language and support analyses | [Language summaries](../results/fleurs_per_lang_summary.json), [complete-transcript oracle summaries](../results/current_support_oracles.json) |
| Kathbath clean/testunk | [Fixed-model readout](../results/testunk_postlock_symbilist.json), [matched table controls](../results/testunk_complete_table3.json) |
| Common Voice 26: two recognizer architectures | [Zero-shot and grouped development protocols](../results/stage2_cv26_cross_arch.json) |
| FLEURS-12 and Kathbath: cross-architecture pairs and three-system fusion | [Two- and three-system results](../results/stage2_fleurs_kb_cross_arch_3way.json) |

The Common Voice zero-shot and grouped settings answer different questions. Reference-driven oracle values describe the available transcript space and are not deployable predictions.
