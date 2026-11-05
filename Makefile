UV ?= uv
RUN_PYTHON = $(UV) run --no-project --managed-python --python 3.12 python

.DEFAULT_GOAL := help

.PHONY: all help check compile test runtime vendor-sync clean lab-test

# `all` is what a content installer runs: it must not compile llama.cpp and
# it must not download weights.
all: compile
	@test -x ./kilix-llm || { echo 'kilix-llm launcher is not executable'; exit 1; }

help:
	@printf '%s\n' \
		'kilix-llm' \
		'' \
		'  make all          compile the Python tree (installer target)' \
		'  make check        compile + the suite' \
		'  make test         the suite' \
		'  make runtime      build vendored llama.cpp, CPU only' \
		'  make vendor-sync  fetch the pinned llama.cpp commit' \
		'  make lab-test     weight-edit and interpretability suite' \
		'' \
		'  KILIX_LLM_BUILD_DIR=$(KILIX_LLM_BUILD_DIR)'

compile:
	$(RUN_PYTHON) -m compileall -q src tests kilix-llm

test:
	PYTHONPATH=src $(RUN_PYTHON) -m unittest discover -s tests -v

check: compile test

lab-test:
	PYTHONPATH=lab/src $(RUN_PYTHON) -m unittest discover -s lab/tests -v

runtime:
	./scripts/build-runtime.sh

vendor-sync:
	./scripts/vendor-sync.sh

clean:
	rm -rf build src/kilix_llm/__pycache__ tests/__pycache__
