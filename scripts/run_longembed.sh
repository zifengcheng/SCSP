#!/usr/bin/env bash
set -euo pipefail

METHODS=(vanilla scout echo echo_scout htp htp_scout)

for method in "${METHODS[@]}"; do
  scout-eval --config "configs/${method}.yaml" "$@"
done
