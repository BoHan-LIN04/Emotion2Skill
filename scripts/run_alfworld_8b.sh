#!/usr/bin/env bash
# ============================================================
# Run the full Emotion2Skill pipeline on ALFWorld (Qwen3-8B)
# ============================================================
# Usage: bash scripts/run_alfworld_8b.sh [GPU_IDS]
# ============================================================

set -euo pipefail

GPU_IDS="${1:-0,1}"
MODEL="Qwen/Qwen3-8B"
MODEL_KEY="qwen3-8b"
EXTRACTOR="checkpoints/extractor_qwen3_8b.pkl"
ENCODER_ALF="checkpoints/encoder_alfworld_8b.pt"
RESULTS="results/alfworld_8b"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-data/alfworld}"

echo "============================================================"
echo " Emotion2Skill — ALFWorld — Qwen3-8B"
echo " GPUs: ${GPU_IDS}"
echo "============================================================"

# ── Step 1: Extractor ────────────────────────────────────────────────────────
if [ ! -f "${EXTRACTOR}" ]; then
  echo "[Step 1] Building extractor..."
  python -c "
from emotion2skill.extraction import EmotionExtractor
ext = EmotionExtractor('${MODEL}', '${MODEL_KEY}', 'cuda')
ext.build(
    stories_path='data/emotion_stories/stories.json',
    goemotions_test_path='data/goemotions/test.json',
    save_path='${EXTRACTOR}',
)
"
else
  echo "[Step 1] Extractor cached. Skipping."
fi

# ── Step 2: Encoder ──────────────────────────────────────────────────────────
if [ ! -f "${ENCODER_ALF}" ]; then
  echo "[Step 2] Training encoder on ALFWorld warm-up episodes..."
  python train_encoder.py \
    --model       "${MODEL}" \
    --model_key   "${MODEL_KEY}" \
    --env         alfworld \
    --warmup      50 \
    --epochs      30 \
    --extractor   "${EXTRACTOR}" \
    --out         "${ENCODER_ALF}" \
    --log_dir     logs/warmup_alfworld_8b
else
  echo "[Step 2] Encoder cached. Skipping."
fi

# ── Step 3: Evaluate ─────────────────────────────────────────────────────────
echo "[Step 3] Evaluating (3 runs × 134 episodes)..."
mkdir -p "${RESULTS}"
python evaluate.py \
  --model       "${MODEL}" \
  --model_key   "${MODEL_KEY}" \
  --env         alfworld \
  --extractor   "${EXTRACTOR}" \
  --encoder     "${ENCODER_ALF}" \
  --tau         0.3 \
  --evolution \
  --n_runs      3 \
  --out_dir     "${RESULTS}"

echo "Done. Results → ${RESULTS}/"
