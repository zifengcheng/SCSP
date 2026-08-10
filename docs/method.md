# SCOUT Method

SCOUT is a training-free document encoder with four stages: semantic chunking,
semantic compression prompting, prompt-guided token selection, and selective
intermediate-layer pooling.

## 1. Semantic Chunks

The document is segmented into sentences with spaCy. Consecutive sentences are
packed into chunks up to a soft token budget. A sentence longer than the budget
is retained as one unit, while the complete document is still capped by
`model.max_body_tokens`.

For chunks `C_1, ..., C_m` and compression prompt `P`, SCOUT constructs:

```text
[BOS, C_1, P_1, C_2, P_2, ..., C_m, P_m]
```

The default prompt is `\nBrief summary:"`. Its final token is the semantic
compression anchor by default.

## 2. Prompt Isolation

Real-text rows preserve their original causal attention and cannot read any
compression prompt. Under the default `local` scope, prompt `P_i` reads:

- the common prefix, such as BOS;
- real tokens in `C_i`;
- preceding tokens inside `P_i`.

It cannot read other chunks or prompts. The optional `cumulative` scope allows
`P_i` to read real text through `C_i`, while prompts remain mutually hidden.
Position IDs remain continuous in both modes.

## 3. Attention Scoring

For layer `l`, head `h`, prompt anchor `a`, and token `j` in chunk `C_i`, SCOUT
first computes the standard scaled dot-product attention distribution:

```text
p[l,h,a,j] = softmax_j(q[l,h,a] k[l,h,j]^T / sqrt(d)).
```

It then retains the strongest probability received by each token:

```text
r[i,j] = max_l max_h max_a p[l,h,a,j].
```

The implementation applies a final chunk-wise L1 normalization:

```text
s[i,j] = r[i,j] / sum_t r[i,t].
```

This is implemented by `maximum_attention_distribution`. Softmax and
aggregation use FP32 even when model inference uses lower precision.

## 4. Token Selection And Pooling

A token is retained when its normalized attention exceeds a multiple of the
uniform attention level in its chunk:

```text
s[i,j] >= alpha / |C_i|.
```

`alpha` is `scout.selection_ratio`; at least one token is retained in every
non-empty chunk. Let `S` be the union of retained real tokens. The final
document representation from readout layer `r` is:

```text
e = L2Normalize(mean({h_j^(r) : j in S})).
```

Attention scores determine only the selection mask. They do not weight the
final mean. Structural prompts, BOS, padding, PST, and B-PST are excluded.

## Composition

SCOUT controls which real-token representations are pooled. Echo controls the
document input through repetition, while HTP controls hidden-state propagation
through structural slots and layer rewiring. The concerns remain separate:

1. Construct the complete parent-method input.
2. Append SCOUT prompts without moving parent tokens.
3. Preserve every parent attention row.
4. Compute one SCOUT selection mask over eligible real tokens.
5. Pool only selected real-token hidden states.

This invariant is covered by the model-integration tests.
