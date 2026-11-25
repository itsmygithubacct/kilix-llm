# Judge: one yes/no answer as an exit code

`kilix-llm judge` answers a yes/no question, or a pass/fail judgement of
information you show it, with one character, and exits with it:

| stdout / exit | meaning |
|---|---|
| `0` | yes, pass, ok, correct, valid |
| `1` | no, fail, error, incorrect, invalid |
| nothing / `2` | usage, model, download or digest problem, or the sizer gave no limit |
| nothing / `3` | the input is longer than this machine's limit (stderr says the limit) |
| nothing / `4` | the runtime is missing or cannot be built |
| nothing / `5` | the engine failed or answered outside the contract |

A question is answered as asked: "Did the build fail?" is `0` when it did.
Phrase a verdict ("Did the build pass?") when you want pass/fail polarity.
Put information to judge between `<information>` and `</information>`; text
inside it is data, not instructions.

```sh
if ./kilix-llm judge --model-id qwen3.5-0.8b "Is 2+2 equal to 4?"; then echo yes; fi
make test 2>&1 | tail -40 | { printf '<information>\n'; cat; printf '</information>\n\nDid the tests pass?'; } \
  | ./kilix-llm judge --model-id qwen3.5-0.8b --stdin --json
```

The answer is one token at temperature 0 under the grammar
`root ::= "0" | "1"`, with thinking off. `--json` prints the answer with `p0`
and `p1`, the model's two raw logprobs renormalised, so a caller can set its
own threshold; `probs_bounded` is true when the other digit was not among the
listed candidates and its probability is an upper bound.

Before answering, `judge` asks `plebian-model-sizer recommend binary-judge`
(found with `--sizer`, `KILIX_LLM_SIZER`, or on PATH) how long the input may
be on this machine, from live free memory and a measured profile for this base
and this llama.cpp pin. A longer input exits 3. `--truncate head` or
`--truncate tail` keeps the input's first or last tokens instead, and says so
on stderr. Nothing is ever cut silently.

The default model is `qwen-0.8b-binary-01`, which is not released yet. Until
it is, `--model-id qwen3.5-0.8b` runs the stock base with the same contract
prompt (`lfm2-1.2b-tool` also works). If the base or adapter is missing,
`judge` lists what it would download (size, source, licence notice) and
downloads only with `--yes` or an interactive yes. `--base PATH` (and
`--model PATH`) name weights you already have; they are used as they are,
after the size and sha256 checks, and are never downloaded.
