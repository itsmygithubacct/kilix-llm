# Document help through kilix-help-llm

Reference for [kilix-llm](../README.md).

For a prepared `kilix-help-llm` checkout and its local trained runs:

```sh
kilix-llm ask-help --help-module ../kilix-modules/kilix-help-llm \
  --answer-run answer-r1 --rank-run rank-r1 'How do I list panes?'
```

This command delegates to the help module's CPU adapter runtime and returns
its JSON unchanged, including source revision, corpus digest, citations,
refusals and qualification status. The module verifies artifacts and matches
the two runs to the same corpus/context, sizes each load through the shared
model sizer, and releases the ranking model before loading the answer model.
Its custom scoring head remains separate from prose generation. User data
uses `GPU_TERMINAL_HOME` (default `~/.local/gpu_terminal`). The optional
`--sizer` selects the shared provider explicitly.

The current help path is experimental and unqualified. This is a local
one-shot interface using the help runtime; it does not export adapters to
GGUF or add them to the llama-server catalog. Qualification requires measured
answer quality and verification of the actual serving configuration.

Once a frozen calibration threshold exists, add `--calibration-policy PATH`
and `--docs-root KILIX_CHECKOUT` to `ask-help`. The host verifies the policy,
run reports, corpus revision, sequential ranking result, and the selected
paragraph against the pinned Git document. It delivers a cited model answer
only when its top ranking score meets the inclusive threshold. It preserves an
explicit refusal and otherwise returns the fixed static-document fallback.
The untouched model
result is saved under private `kilix-help-llm/diagnostics` in
`GPU_TERMINAL_HOME`; rejected drafts are omitted from the user-visible JSON.
The calibrated path remains unqualified until held-out quality and actual host
parity gates pass. Its generation limit is fixed at 96 tokens.
