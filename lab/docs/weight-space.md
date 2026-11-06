# Weight-space column edits

`kilix_llm_lab.weight` edits selected columns of one loaded writing matrix.
The command is `weight`. It does not train, and it does not rewrite the
checkpoint file. The 0.2.3 lab uses it to try a small, recorded change and
then score how the answer changes.

Abliteration is a different edit. It writes `P = I - r rᵀ` into both
`self_attn.o_proj` and `mlp.down_proj`, and it projects the bias when one
exists. A column edit touches one matrix, leaves the bias alone, and can
leave the other columns alone.

## What a column is

Hugging Face stores a linear weight as `(out_features, in_features)`.
One column is one input's contribution to the output. For `mlp.down_proj`
that input is one MLP neuron. For `self_attn.o_proj` it is one slice of
the attention output. Both matrices write the residual stream.

The editable leaves are `down_proj` and `o_proj`. The embedding, the
output projection (`lm_head`), and a final `norm` are refused. Any other
leaf is refused.

## Operations

`u` is the supplied direction after it is normalized. A zero direction is
refused.

| Operation | Effect | Notes |
|---|---|---|
| `remove` | `column := column - scale * u (u · column)` | Scale 1 removes `u` from the chosen columns. Scale 1.5 subtracts the direction one and a half times. Scale 0 is refused. |
| `scale` | `column := factor * column` | `--factor` is required. |
| `zero` | `column := 0` | The same rule as scale with factor 0. |

`--columns` is a comma-separated list of indexes. `--top N` ignores that
list and keeps the N columns with the largest squared dot against `u`.
An equal score keeps the lower index. Passing both is refused.

`remove` with neither `--columns` nor `--top` edits every column of that
one matrix. `scale` and `zero` must name columns. A constant factor on
every column of a tensor is refused.

No operation runs a backward pass or an optimizer.

## Manifest

The command prints one JSON object.

| Field | Meaning |
|---|---|
| `op` | `remove`, `scale`, or `zero` |
| `module` | `layers.<index>.mlp.down_proj` or `layers.<index>.self_attn.o_proj`; null for a JSON matrix |
| `layer` | Decoder index, or null for a JSON matrix |
| `scale` | Remove strength, or null |
| `factor` | Column multiplier, or null. Zero uses 0 |
| `targets` | Column indexes the edit was asked to change |
| `shape` | `[out_features, in_features]` |
| `changed_entries` | Entries that differ |
| `changed_columns` | Columns with any differing entry |
| `changed_rows` | Rows with any differing entry |
| `fraction` | `changed_entries` divided by the number of entries |
| `relative_change` | Frobenius norm of the difference divided by the Frobenius norm of the original. Null when the original norm is 0 and the edit is not |

On a loaded module the counts are taken after the new values are cast
back to the matrix dtype. A touch that rounds back to the stored value
is not counted. `--quant 4bit` is refused because the edit needs a dense
floating-point matrix.

`--prompt` adds three fields to that object: `before`, `after`, and
`changed`. Those are the greedy continuations. That comparison is the
score. The manifest does not report download time or tokens per second.

## A matrix, with no checkpoint

`rows.json`:

```json
{"rows": [[3.0, 1.0], [4.0, 0.0]]}
```

```sh
lab/scripts/lab.sh weight \
  --matrix rows.json \
  --op remove \
  --direction 1,0 \
  --columns 0
```

`--write-matrix out.json` saves the edited rows. `--direction-file` reads
a JSON object whose `direction` array is the vector. A difference-of-means
file has that field.

This path uses system Python. It does not create the lab virtual
environment and it does not download a model.

## A loaded checkpoint

```sh
lab/scripts/lab.sh weight \
  --layer 15 \
  --module down_proj \
  --op remove \
  --top 64 \
  --direction-file direction.json \
  --allow-other \
  --prompt 'Tell me a story.'
```

`--module` is `down_proj` or `o_proj`. The default is `down_proj`.
`--layer` is a 0-based decoder index. Model commands go through
`lab/scripts/uv-venv.sh`. The checkpoint file on disk is not modified.
Reload the model to drop the edit.

The same score used by the other lab edits applies here. `before` and
`after` are the greedy continuations of `--prompt`.

An Ask output row is a different edit. `output-row` solves one
`output.weight` row from frozen activations. This column command still
refuses `lm_head`. See [output-row.md](output-row.md).

## Tests

`make lab-test` covers the arithmetic, the refusals, and the JSON command.
It does not download a checkpoint. The loaded-module check runs only when
torch is already installed.
