#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${1:-0,1}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG="${2:-configs/default.yaml}"
for BENCHMARK in webshop alfworld; do
  for CONDITION in full no_diagnostics; do
    "${PYTHON_BIN}" evaluate.py --config "${CONFIG}" --env "${BENCHMARK}" \
      --ablation "${CONDITION}" --evaluate_each_round --n_runs 3 \
      --evolution --evolution_rounds 3 --episode_per_round 50 --eta 0.4 \
      --out_dir "results/round_analysis/${BENCHMARK}/${CONDITION}"
  done
done
