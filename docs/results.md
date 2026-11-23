# Results index

The files below contain stored aggregate results from the experiments reported in the paper. Consult the corresponding table captions for populations, units, baselines and uncertainty. Lower WER and CER mean fewer edits relative to the reference under the stated normalization.

| Setting or analysis | Files |
|---|---|
| FLEURS-12 and Kathbath validation: ranking, anchor orientation, deferral and ablations | [Baselines](../results/stage1_baselines_and_ablations.json), [anchor controls](../results/stage1_direction_a.json), [ranking and deferral](../results/stage1_direction_b.json), [SymBiList](../results/stage1_symbilist.json) |
| Language-level comparison and diagnostic figure | [Language summaries](../results/fleurs_per_lang_summary.json), [complete-transcript oracle summaries](../results/current_support_oracles.json) |
| Kathbath clean/testunk | [Fixed-model readout](../results/testunk_postlock_symbilist.json), [matched table controls](../results/testunk_complete_table3.json) |
| Common Voice 26: two recognizer architectures | [Zero-shot and grouped development protocols](../results/stage2_cv26_cross_arch.json) |
| FLEURS-12 and Kathbath: cross-architecture pairs and three-system fusion | [Two- and three-system results](../results/stage2_fleurs_kb_cross_arch_3way.json) |

The Common Voice zero-shot and grouped settings answer different questions. Reference-driven oracle values describe the available transcript space and are not deployable predictions.
