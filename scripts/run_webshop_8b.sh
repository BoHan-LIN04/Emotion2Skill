#!/usr/bin/env bash
# ============================================================
# Run the full Emotion2Skill pipeline on WebShop (Qwen3-8B)
# ============================================================
# Usage: bash scripts/run_webshop_8b.sh [GPU_IDS]
# Example: bash scripts/run_webshop_8b.sh 0,1
# ============================================================

set -euo pipefail

GPU_IDS="${1:-0,1}"
MODEL="Qwen/Qwen3-8B"
MODEL_KEY="qwen3-8b"
EXTRACTOR="checkpoints/extractor_qwen3_8b.pkl"
ENCODER_WS="checkpoints/encoder_webshop_8b.pt"
RESULTS="results/webshop_8b"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

echo "============================================================"
echo " Emotion2Skill — WebShop — Qwen3-8B"
echo " GPUs: ${GPU_IDS}"
echo "============================================================"

# ── Step 1: Build emotion extractor (if not cached) ─────────────────────────
if [ ! -f "${EXTRACTOR}" ]; then
  echo ""
  echo "[Step 1] Building emotion extractor (W_e, L*=24)..."
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
  echo "[Step 1] Extractor already cached at ${EXTRACTOR}. Skipping."
fi

# ── Step 2: Train emotion encoder ────────────────────────────────────────────
if [ ! -f "${ENCODER_WS}" ]; then
  echo ""
  echo "[Step 2] Training emotion encoder f_θ (warm-up: 50 episodes)..."
  python train_encoder.py \
    --model       "${MODEL}" \
    --model_key   "${MODEL_KEY}" \
    --env         webshop \
    --warmup      50 \
    --epochs      30 \
    --lr          3e-4 \
    --lam         0.5 \
    --tau         0.3 \
    --extractor   "${EXTRACTOR}" \
    --out         "${ENCODER_WS}" \
    --log_dir     logs/warmup_webshop_8b
else
  echo "[Step 2] Encoder already cached at ${ENCODER_WS}. Skipping."
fi

# ── Step 3: Evaluate (3 runs, with evolution) ────────────────────────────────
echo ""
echo "[Step 3] Running evaluation (3 runs × 500 episodes, with evolution)..."
mkdir -p "${RESULTS}"
python evaluate.py \
  --model       "${MODEL}" \
  --model_key   "${MODEL_KEY}" \
  --env         webshop \
  --extractor   "${EXTRACTOR}" \
  --encoder     "${ENCODER_WS}" \
  --tau         0.3 \
  --eta         0.5 \
  --evolution \
  --evolution_rounds    5 \
  --episode_per_round  20 \
  --n_runs      3 \
  --out_dir     "${RESULTS}"

echo ""
echo "Done. Results written to ${RESULTS}/"
