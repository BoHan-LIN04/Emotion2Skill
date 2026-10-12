#!/usr/bin/env bash
# ============================================================
# Ablation study sweep (Table 2)
# ============================================================
# Runs all five ablation conditions on WebShop and ALFWorld.
# Requires pre-built extractor + encoder checkpoints.
# Usage: bash scripts/run_ablations.sh [GPU_IDS]
# ============================================================

set -euo pipefail

GPU_IDS="${1:-0,1}"
MODEL="Qwen/Qwen3-8B"
MODEL_KEY="qwen3-8b"
EXTRACTOR="checkpoints/extractor_qwen3_8b.pkl"
ENCODER_WS="checkpoints/encoder_webshop_8b.pt"
ENCODER_ALF="checkpoints/encoder_alfworld_8b.pt"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

ABLATIONS=("no_extraction" "no_encoder" "selection_only" "no_diagnostics" "full")

for ENV in webshop alfworld; do
  ENC="${ENCODER_WS}"
  [ "${ENV}" == "alfworld" ] && ENC="${ENCODER_ALF}"

  for ABL in "${ABLATIONS[@]}"; do
    echo ""
    echo "=== Ablation: ${ABL} | Env: ${ENV} ==="
    python evaluate.py \
      --model       "${MODEL}" \
      --model_key   "${MODEL_KEY}" \
      --env         "${ENV}" \
      --extractor   "${EXTRACTOR}" \
      --encoder     "${ENC}" \
      --tau         0.3 \
      --ablation    "${ABL}" \
      --n_runs      1 \
      --out_dir     "results/ablations/${ENV}"
  done
done

echo ""
echo "All ablations complete."
