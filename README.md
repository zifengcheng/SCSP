# SCOUT

SCOUT builds a single embedding for a long document by using semantic
compression prompts to identify informative tokens before pooling. It is
training-free and composes with standard mean pooling, Echo, and Hierarchical
Token Prepending (HTP).

## How It Works

1. Pack sentence boundaries into token-budgeted semantic chunks.
2. Append a short semantic compression prompt to every chunk.
3. Isolate each prompt so it reads its own chunk without changing real-text
   attention rows.
4. Score tokens using the strongest prompt attention across all layers and
   heads.
5. Mean-pool selected real-token representations from an intermediate layer.

The default configuration uses 512-token chunks, the prompt
`\nBrief summary:"`, a selection ratio of `1.75`, and the third-to-last model
layer. See [the method definition](docs/method.md) for the exact equations and
mask semantics.

## Installation

SCOUT requires Python 3.10 or newer. Install the PyTorch build appropriate for
your platform, then run:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m spacy download en_core_web_sm
```

For development:

```bash
pip install -e '.[dev]'
ruff check src tests
pytest
```

Model checkpoints and datasets are not bundled. Model access is managed by
the Hugging Face Hub and remains subject to each checkpoint's license.

## Quick Start

Evaluate SCOUT on the datasets listed in its configuration:

```bash
scout-eval --config configs/scout.yaml
```

Run all included method configurations:

```bash
bash scripts/run_longembed.sh
```

Each command loads the model once and evaluates all configured datasets. It
writes one JSON file per dataset and a `summary.csv` containing retrieval
metrics, retained-token fraction, sequence statistics, and wall-clock time.

## Included Methods

| Configuration | Document input | Readout |
|---|---|---|
| `vanilla.yaml` | Original document | Real-token mean |
| `scout.yaml` | Semantic chunks and compression prompts | SCOUT-selected mean |
| `echo.yaml` | Two document copies | Second-copy mean |
| `echo_scout.yaml` | Echo input with SCOUT prompts | Selected second-copy mean |
| `htp.yaml` | HTP sentence blocks | Real-token mean |
| `htp_scout.yaml` | HTP input with SCOUT prompts | Selected real-token mean |

Queries follow the parent encoder and do not use token selection. Echo and HTP
inputs are constructed first; SCOUT prompts are then appended as readout-only
structures. Parent token positions and parent attention rows remain unchanged.

## Configuration

Experiments are defined in YAML. The principal SCOUT fields are:

```yaml
scout:
  chunk_size: 512
  chunking: sentence_aware
  sentence_splitter: en_core_web_sm
  compression_prompt: "\nBrief summary:\""
  prompt_anchor_tokens: 1
  prompt_attention_scope: local
  position_mode: continuous
  attention_aggregation: maximum
  selection_ratio: 1.75
  min_tokens_per_chunk: 1
```

`max_body_tokens` limits real document tokens. Compression prompts and HTP
structural tokens are tracked separately and never enter selection or pooling.

## Custom Retrieval Data

A local dataset directory must contain:

```text
queries.jsonl   {"qid": "q1", "text": "..."}
corpus.jsonl    {"doc_id": "d1", "text": "..."}
qrels.jsonl     {"qid": "q1", "doc_id": "d1", "score": 1}
```

Register it at runtime and include its name in `evaluation.datasets`:

```bash
scout-eval --config my_config.yaml \
  --local-dataset MyTask=path/to/MyTask
```

## Repository Layout

```text
configs/                         Reproducible method configurations
docs/method.md                   Equations and architectural invariants
scripts/                         Portable evaluation entry points
src/scout_embedding/
  chunking.py                    Fixed and sentence-aware chunking
  input_construction.py          SCOUT inputs and prompt isolation
  attention_scoring.py           All-layer attention collection and scoring
  token_selection.py             Attention-threshold selection rules
  selective_pooling.py           Masked mean pooling and normalization
  baselines/                     Echo input and independent HTP components
  modeling.py                    Model adapter and method composition
  data.py                        LongEmbed and local JSONL loading
  evaluation.py                  nDCG evaluation
  cli.py                         Command-line interface
tests/                           Algorithm and model-integration tests
```

## Reproducibility

- Attention logits, softmax, and score aggregation are computed in FP32.
- Real-text tokens cannot attend to semantic compression prompts.
- Local prompt isolation changes only prompt rows, not real-text rows.
- BOS, padding, prompts, PST, and B-PST are never pooled.
- Model and dataset revisions can be pinned in YAML.

The adapter targets decoder models exposing Hugging Face-style decoder layers,
Q/K projections, and rotary embeddings. Architecture-specific behavior should
be validated before reporting results on a new model family.

## Acknowledgements

The evaluation pipeline builds on LongEmbed. Echo and HTP are independently
implemented compositional baselines; their original repositories and licenses
remain authoritative. See [NOTICE.md](NOTICE.md) for attribution boundaries.
