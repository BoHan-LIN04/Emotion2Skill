#!/usr/bin/env bash
set -euo pipefail
# Usage: bash scripts/run_qwen3_14b.sh [webshop|alfworld|math|mbpp] [GPU_IDS]
bash scripts/run_pipeline.sh "${1:-webshop}" "${2:-0,1}" configs/qwen3_14b.yaml
