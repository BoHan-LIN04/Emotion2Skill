#!/usr/bin/env bash
set -euo pipefail
BENCHMARK="${1:-webshop}"
GPU_IDS="${2:-0,1}"
CONFIG="${3:-configs/default.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
EXTRACTOR="$(${PYTHON_BIN} -c 'from emotion2skill.config import load_config; import sys; print(load_config(sys.argv[1])["extractor"]["checkpoint"])' "${CONFIG}")"
ENCODER="$(${PYTHON_BIN} -c 'from emotion2skill.config import load_config; import sys; print(load_config(sys.argv[1])["encoder"]["checkpoint"])' "${CONFIG}")"
if [[ ! -f "${EXTRACTOR}" ]]; then
  "${PYTHON_BIN}" scripts/build_extractor.py --config "${CONFIG}"
fi
if [[ ! -f "${ENCODER}" ]]; then
  "${PYTHON_BIN}" train_encoder.py --config "${CONFIG}" --env mixed \
    --warmup 250 --epochs 50 --batch_size 64 --lr 0.001 \
    --weight_decay 0.0001 --patience 5 --extractor "${EXTRACTOR}" --out "${ENCODER}"
fi
"${PYTHON_BIN}" evaluate.py --config "${CONFIG}" --env "${BENCHMARK}" \
  --extractor "${EXTRACTOR}" --encoder "${ENCODER}" --n_runs 3 \
  --evolution --evolution_rounds 3 --episode_per_round 50 --eta 0.4
