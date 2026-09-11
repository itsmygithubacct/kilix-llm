PYTHON ?= python3

.DEFAULT_GOAL := help

.PHONY: all help check compile test runtime vendor-sync clean

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
		'' \
		'  KILIX_LLM_BUILD_DIR=$(KILIX_LLM_BUILD_DIR)'

compile:
	$(PYTHON) -m compileall -q src tests kilix-llm

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v

check: compile test

runtime:
	./scripts/build-runtime.sh

vendor-sync:
	./scripts/vendor-sync.sh

clean:
	rm -rf build src/kilix_llm/__pycache__ tests/__pycache__
