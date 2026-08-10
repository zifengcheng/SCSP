#!/usr/bin/env bash
set -euo pipefail

# Copy configs/scout.yaml, change one field at a time, and list the files
# here. Every run records its complete resolved configuration in result JSON.
for config in "$@"; do
  scout-eval --config "$config"
done
