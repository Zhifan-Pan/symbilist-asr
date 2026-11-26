#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 scripts/rebuild_tables.py
mkdir -p build paper
(cd manuscript && SOURCE_DATE_EPOCH=1795564800 FORCE_SOURCE_DATE=1 latexmk -pdf -usepretex='\pdftrailerid{}' -interaction=nonstopmode -halt-on-error -outdir=../build main.tex)
cp build/main.pdf paper/main.pdf
