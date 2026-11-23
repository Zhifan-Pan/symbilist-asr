# SymBiList

**Symmetric Agreement-Anchored Phrase Recombination for Frozen Multilingual ASR**

Zhifan Pan, Yingxiong Gao, Maihemuti Maimaiti, Yifeng Yu, and Aishan Wumaier

[Read the paper (PDF)](paper/main.pdf) · [LaTeX source](manuscript/main.tex) · [Results index](docs/results.md) · [Build and data requirements](docs/reproducibility.md)

SymBiList combines scored N-best transcripts from frozen speech recognizers. It aligns the lists to an anchor, treats unanimously agreed words as boundaries, and selects an observed phrase between successive boundaries. The symmetric model runs in both anchor orientations and reconciles the remaining disagreements. This allows combinations that no recognizer returned as a whole sentence.

The [core package](src/agree_phrase_asr/) implements alignment, partitioning, phrase features, language models and edit-count evaluation. The [SymBiList experiment](scripts/run_stage1_symbilist.py) contains the listwise ranker and bidirectional selection. The [aggregate results](results/) accompany the tables and figures in the manuscript.

## Build the paper

With pdfLaTeX, BibTeX and latexmk installed, run the following from the repository root:

```sh
bash scripts/build_paper.sh
```

The PDF is written to `paper/main.pdf`. The figure PDFs are included; building the paper does not require a GPU, a speech corpus or a trained model.

## Run the core tests

In a Python 3.10+ environment, run:

```sh
python -m pip install -e '.[dev]'
python -m pytest -q tests/
```

The tests exercise alignment, partitioning, features and scoring using local fixtures. For the inputs needed by the research drivers, see [reproducibility](docs/reproducibility.md).

Original software is available under the [MIT License](LICENSE). The manuscript, figures and third-party TeX files are covered separately in [NOTICES.md](NOTICES.md).
