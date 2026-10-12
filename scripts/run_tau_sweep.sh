#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${1:-0,1}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
# Choose a threshold from disjoint validation data, then report all test settings.
"${PYTHON_BIN}" evaluate.py --config configs/default.yaml --env webshop --tau_sweep \
  --n_episodes 100 --n_runs 3 --out_dir results/tau_validation
for BENCHMARK in webshop alfworld; do
  for TAU in 0.1 0.3 0.6 0.9; do
    "${PYTHON_BIN}" evaluate.py --config configs/default.yaml --env "${BENCHMARK}" --tau "${TAU}" \
      --n_runs 3 --out_dir "results/tau_sensitivity/${BENCHMARK}/${TAU}"
  done
done
