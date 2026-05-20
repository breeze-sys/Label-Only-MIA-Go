#!/usr/bin/env bash
set -euo pipefail

if [ ! -f shadow_train_data.json ]; then
  echo "shadow_train_data.json is required to rebuild thresholds."
  echo "Generate it from the relabel pipeline or place the real JSON file at the repository root."
  exit 1
fi

docker compose run --rm threshold-generator
