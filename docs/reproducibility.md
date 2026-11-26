# Build and data requirements

## Paper and figures

From the repository root, `bash scripts/build_paper.sh` uses Python 3 to regenerate Tables I–III, then runs latexmk, pdfLaTeX and BibTeX and writes `paper/main.pdf`. The IEEEtran class and bibliography style are included. The TeX installation needs the packages loaded by [main.tex](../manuscript/main.tex), including microtype, multirow, booktabs and balance. With the authors’ TeX toolchain the build is byte-reproducible; `sha256sum -c SHA256SUMS` checks the published files.

The manuscript embeds the [supplied Figure 1 PNG](../manuscript/figures/SYMBILIST_Figure1_final.png) directly; it includes the Hausa development example and the fixed-versus-agreement inset. The older `method_overview` artwork and its SVG script are historical. [Figure 2](../manuscript/figures/mechanism_and_risk.pdf) is also included and can be redrawn with `python3 scripts/generate_figures.py` (NumPy and Matplotlib) from the aggregate inputs in `results/`.

## Code and experiments

- [Alignment](../src/agree_phrase_asr/align.py) and [partitions](../src/agree_phrase_asr/partition.py): normalize hypotheses, project them onto an anchor, and construct agreement, fixed, random or whole-sentence blocks.
- [Features](../src/agree_phrase_asr/features.py), [language models](../src/agree_phrase_asr/lm.py) and [models](../src/agree_phrase_asr/models.py): construct and score local phrase alternatives.
- [SymBiList](../scripts/run_stage1_symbilist.py): fit the listwise neural ranker and perform bidirectional recombination. The other stage drivers expose the comparisons and cross-architecture settings.
- [Evaluation](../src/agree_phrase_asr/eval.py): word and character edits, macro/micro rates and paired cluster bootstrap utilities.

Running a corpus experiment requires the corresponding licensed speech assets or previously decoded candidate lists, language-model resources, and exact split memberships. The development adapter takes `SYMBILIST_BUNDLE`, pointing to a compatible local frozen research bundle with its original data-loading code, scored hypotheses, group folds and fitted LMs. Some stage drivers also use local `cache/` and `manifests/` inputs. The Common Voice driver takes `SYMBILIST_CV26_DIR` for the language-specific training splits; the Kathbath test drivers take `SYMBILIST_KATHBATH_TRANSCRIPTS` for the publisher's transcript archive. These are data-dependent research entry points; installing the Python package alone does not recreate those experiments.

Acquire FLEURS, Kathbath and Mozilla Common Voice through the respective data providers and follow their applicable terms. The repository contains aggregate metrics and the illustrated excerpt in the paper. It does not distribute corpus recordings, per-utterance reference or prediction files, internal rosters, acoustic model weights or training checkpoints.
