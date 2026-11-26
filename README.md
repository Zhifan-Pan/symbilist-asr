# SymBiList

**Symmetric Agreement-Anchored Phrase Recombination for Frozen Multilingual ASR**

Zhifan Pan, Maihemuti Maimaiti (corresponding author), Yingxiong Gao, Yifeng Yu, and Aishan Wumaier

[Read the paper (PDF)](paper/main.pdf) · [LaTeX source](manuscript/main.tex) · [Results index](docs/results.md) · [Build and data requirements](docs/reproducibility.md)

SymBiList combines scored N-best transcripts from frozen speech recognizers. It aligns the same lists to each of two anchors and ranks intact phrases between agreement boundaries. With two recognizers, a boundary requires unanimity; with three, two agreeing recognizers establish a boundary and the dissenting word remains selectable. The two views are reconciled to produce a transcript that can combine phrases no recognizer returned together.

The [core package](src/agree_phrase_asr/) implements alignment, partitioning, phrase features, language models and edit-count evaluation. The [SymBiList experiment](scripts/run_stage1_symbilist.py) contains the listwise ranker and bidirectional selection. The [aggregate results](results/) accompany the tables and figures in the manuscript.

## Build the paper

With Python 3, pdfLaTeX, BibTeX and latexmk installed, run the following from the repository root:

```sh
bash scripts/build_paper.sh
```

The build regenerates Tables I–III from aggregate results and writes the six-page PDF to `paper/main.pdf`. [Figure 1](manuscript/figures/SYMBILIST_Figure1_final.png) and [Figure 2](manuscript/figures/mechanism_and_risk.pdf) are included. No GPU, speech corpus or trained model is required to build the paper.

## Run the core tests

In a Python 3.10+ environment, run:

```sh
python -m pip install -e '.[dev]'
python -m pytest -q tests/
```

The tests exercise alignment, partitioning, features and scoring using local fixtures. For the inputs needed by the research drivers, see [reproducibility](docs/reproducibility.md).

Original software is available under the [MIT License](LICENSE). The manuscript, figures and third-party TeX files are covered separately in [NOTICES.md](NOTICES.md).
