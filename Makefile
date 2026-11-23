.PHONY: paper test

paper:
	bash scripts/build_paper.sh

test:
	python -m pytest -q tests/
