#!/usr/bin/env bash
# ============================================================
# τ sensitivity sweep (Table 2, bottom section)
# ============================================================
# Usage: bash scripts/run_tau_sweep.sh [GPU_IDS]
# ============================================================

set -euo pipefail

GPU_IDS="${1:-0,1}"
MODEL="Qwen/Qwen3-8B"
EXTRACTOR="checkpoints/extractor_qwen3_8b.pkl"
ENCODER_WS="checkpoints/encoder_webshop_8b.pt"
ENCODER_ALF="checkpoints/encoder_alfworld_8b.pt"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

for ENV in webshop alfworld; do
  ENC="${ENCODER_WS}"
  [ "${ENV}" == "alfworld" ] && ENC="${ENCODER_ALF}"
  echo ""
  echo "=== τ sweep on ${ENV} ==="
  python evaluate.py \
    --model     "${MODEL}" \
    --env       "${ENV}" \
    --extractor "${EXTRACTOR}" \
    --encoder   "${ENC}" \
    --n_runs    1 \
    --tau_sweep \
    --out_dir   "results/tau_sweep/${ENV}"
done

echo "τ sweep complete."
