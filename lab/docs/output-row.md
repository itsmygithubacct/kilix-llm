# Output-row edit

`output-row` is the Ask command-model edit. It changes one row of
`output.weight` from frozen activations. It does not run a backward pass,
and it does not modify any other tensor. The column command `weight` is a
different tool and still refuses `lm_head`.

The loss is a weighted squared hinge on the logit shift, plus an L2 penalty
on the row change. Each feature is divided by its root-mean-square, with a
floor of 0.01. The penalty applies to the unscaled change. The solver is
L-BFGS with that analytic gradient.

## What can be fit

Every state needs a case id. These prefixes are sealed and are refused
before the solver runs:

| Prefix | Set |
|---|---|
| `copy-fresh-` | Fresh 160, including the four V5 wins and the three false abstentions |
| `copy-operational-` | Operational 32 |
| `copy-heldout-` | Held-out 128 |

Development ids, including `copy-probe-` and `known-ask-`, are accepted.
A capture without ids is refused by the command.

## Capture

`rows.json` uses schema `kilix.llm.output-row/v1`:

```json
{
  "schema": "kilix.llm.output-row/v1",
  "row": 8104,
  "parent_version": "lfm2-kilix-v5e1-wse-20261104.8",
  "regularization": 1e-5,
  "states": [[0.1, 0.2]],
  "sign": [1],
  "bound": [0.25],
  "weights": [100],
  "case_ids": ["copy-probe-010"],
  "old_row": [0.0, 0.0]
}
```

`sign` is 1 when this row should win the state and -1 when a guard should
keep its current decision. `bound` is the margin that shift must clear.
`weights` defaults to 1. An `.npz` file carries the same arrays. `meta` in
that file is a JSON object with `regularization`, `row`, and
`parent_version`.

```sh
lab/scripts/lab.sh output-row --capture rows.json
lab/scripts/lab.sh output-row --capture rows.json --write-correction correction.json
```

The manifest does not include the correction vector. `--write-correction`
writes it to a new file. A failed solve does not write that file and does
not write a model. The command exits 1 when the solver does not converge.

This path uses the lab virtual environment from `lab/scripts/uv-venv.sh`,
which provides torch and numpy. It does not download a model. An `.npz`
capture and a quantized row write both use that numpy.

## A new model file

```sh
lab/scripts/lab.sh output-row \
  --capture rows.json \
  --parent-gguf parent.gguf \
  --expected-sha256 "$PARENT_SHA" \
  --output candidate.gguf \
  --row 8104
```

`--parent-gguf` is read only. The command copies it to `--output`, which
must not already exist, and replaces one quantized `output.weight` row.
The source sha256 has to match `--expected-sha256`. Bytes outside that row
stay as they were. The published `.8` file can be the parent of a new
candidate. Passing the same path as both parent and output is refused, so
the published bytes are not replaced in place.

The manifest records `correction_sha256`, `float_relative_l2` when an old
row was supplied, and `quantized_relative_l2` after a model write. The
quantized change is the value that landed in the file. It is not claimed to
be bit-identical to any earlier solver.

## Tests

`make lab-test` covers the closed form, the gradient, the sealed ids, and
the copy rules. It does not download a checkpoint. The Q8 block check runs
when numpy can import the vendored reader.
