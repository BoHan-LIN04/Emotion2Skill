#!/usr/bin/env bash
set -euo pipefail
bash scripts/run_pipeline.sh webshop "${1:-0,1}" configs/default.yaml
