# Build and data requirements

## Paper and figures

From the repository root, `bash scripts/build_paper.sh` runs latexmk with pdfLaTeX and BibTeX on `manuscript/main.tex` and copies the result to `paper/main.pdf`. Install the usual IEEEtran, AMS, multirow, booktabs, cite and hyperref TeX packages. The checked-in vector figures are the ones used by the manuscript.

Figure 1 has an editable [SVG](../manuscript/figures/method_overview.svg) and a [CairoSVG build script](../scripts/build_fig1_svg.py). Figure 2 can be redrawn with `python scripts/generate_figures.py` (NumPy and Matplotlib); its corpus-level inputs are in `results/`. Neither figure needs raw audio to render.

## Code and experiments

- [Alignment](../src/agree_phrase_asr/align.py) and [partitions](../src/agree_phrase_asr/partition.py): normalize hypotheses, project them onto an anchor, and construct agreement, fixed, random or whole-sentence blocks.
- [Features](../src/agree_phrase_asr/features.py), [language models](../src/agree_phrase_asr/lm.py) and [models](../src/agree_phrase_asr/models.py): construct and score local phrase alternatives.
- [SymBiList](../scripts/run_stage1_symbilist.py): fit the listwise neural ranker and perform bidirectional recombination. The other stage drivers expose the comparisons and cross-architecture settings.
- [Evaluation](../src/agree_phrase_asr/eval.py): word and character edits, macro/micro rates and paired cluster bootstrap utilities.

Running a corpus experiment requires the corresponding licensed speech assets or previously decoded candidate lists, language-model resources, and exact split memberships. The development adapter takes `SYMBILIST_BUNDLE`, pointing to a compatible local frozen research bundle with its original data-loading code, scored hypotheses, group folds and fitted LMs. Some stage drivers also use local `cache/` and `manifests/` inputs. The Common Voice driver takes `SYMBILIST_CV26_DIR` for the language-specific training splits; the Kathbath test drivers take `SYMBILIST_KATHBATH_TRANSCRIPTS` for the publisher's transcript archive. These are data-dependent research entry points; installing the Python package alone does not recreate those experiments.

Acquire FLEURS, Kathbath and Mozilla Common Voice through the respective data providers and follow their applicable terms. The release contains the reported aggregate metrics. It does not redistribute recordings, per-utterance references or predictions, internal rosters, acoustic model weights or a training checkpoint.
