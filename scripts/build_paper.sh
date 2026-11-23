#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build paper
(cd manuscript && SOURCE_DATE_EPOCH=1795449600 FORCE_SOURCE_DATE=1 latexmk -g -pdf -usepretex='\pdftrailerid{}' -interaction=nonstopmode -halt-on-error -outdir=../build main.tex)
cp build/main.pdf paper/main.pdf
