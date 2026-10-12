#!/usr/bin/env bash
# Refit control encoders with matching features; keep library/data/order fixed.
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${1:-0,1}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
EXTRACTOR=checkpoints/extractor_qwen3_8b.pkl
for CONDITION in no_extraction no_encoder selection_only no_diagnostics generic_pca shuffled full; do
  FEATURE=contrastive
  CONTROL_EXTRACTOR="${EXTRACTOR}"
  ENCODER=checkpoints/encoder_qwen3_8b.pt
  EXTRA=()
  if [[ "${CONDITION}" == no_extraction ]]; then FEATURE=class_mean; fi
  if [[ "${CONDITION}" == generic_pca ]]; then
    FEATURE=generic_pca
    CONTROL_EXTRACTOR=checkpoints/extractor_qwen3_8b_generic_pca.pkl
    EXTRA+=(--extractor_out "${CONTROL_EXTRACTOR}")
  fi
  if [[ "${CONDITION}" == shuffled ]]; then EXTRA+=(--shuffle_context); fi
  if [[ "${CONDITION}" == no_extraction || "${CONDITION}" == generic_pca || "${CONDITION}" == shuffled ]]; then
    ENCODER="checkpoints/encoder_qwen3_8b_${CONDITION}.pt"
    if [[ ! -f "${ENCODER}" ]]; then
      "${PYTHON_BIN}" train_encoder.py --env mixed --warmup 250 --epochs 50 --batch_size 64 \
        --lr 0.001 --weight_decay 0.0001 --patience 5 --extractor "${EXTRACTOR}" \
        --feature_mode "${FEATURE}" --out "${ENCODER}" "${EXTRA[@]}"
    fi
  fi
  for BENCHMARK in webshop alfworld; do
    "${PYTHON_BIN}" evaluate.py --config configs/default.yaml --env "${BENCHMARK}" \
      --extractor "${CONTROL_EXTRACTOR}" --encoder "${ENCODER}" --ablation "${CONDITION}" \
      --n_runs 3 --out_dir "results/ablations/${BENCHMARK}/${CONDITION}"
  done
done
