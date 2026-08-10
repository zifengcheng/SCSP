# Contributing

Contributions should preserve three invariants:

1. Structural prompt, PST, B-PST, BOS, and padding tokens are never pooled.
2. SCOUT combined with Echo or HTP cannot alter parent token positions or
   parent attention rows.
3. Every new score or selection rule is implemented as a pure function and
   covered by a CPU test before it is exposed in the model adapter.

Run the local checks before opening a pull request:

```bash
ruff format --check src tests
ruff check src tests
pytest
```

Keep experiment settings in YAML. Do not add model weights, downloaded
datasets, private paths, access tokens, raw logs, or machine-specific launch
scripts.
