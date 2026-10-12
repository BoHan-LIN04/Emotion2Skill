# Emotion2Skill

**Grounding Agent Skill Selection in LLM-Internal Emotion Vectors**

AAAI 2027 (under review)

---

## Overview

Emotion2Skill is an agent framework that extracts **emotion vectors** directly from a Qwen3 model's residual stream and uses them to improve skill selection and skill evolution—without any external emotion classifier or fine-tuning of the backbone model.

Three named contributions:

| Component | Role |
|-----------|------|
| **Emotion Vectors** | Contrastive directions extracted offline from the residual stream at layer L* |
| **Emotion-Augmented Skill Selection** | Confidence-gated emotion summary injected into the routing prompt |
| **Emotion-Driven Skill Evolution** | Transition-point diagnostics used to localise and rewrite underperforming SOPs |

### Key results (Qwen3-8B)

| Benchmark | Metric | Zero-Shot | MASA | **Emotion2Skill** |
|-----------|--------|-----------|------|-------------------|
| ALFWorld  | Avg.%  | 21.9      | 43.3 | **47.4 ± 1.6**    |
| WebShop   | Succ.% | 2.8       | 27.1 | **29.7 ± 1.6**    |
| MATH      | Acc.%  | 54.8      | —    | **69.2** (+14.4 pp) |
| MBPP      | Pass@1 | 60.0      | —    | **71.8** (+11.8 pp) |

---

## Repository Structure

```
emotion2skill/
├── emotion2skill/              # Main package
│   ├── extraction.py           # §3.2 — Emotion vector extraction (W_e, L*)
│   ├── encoder.py              # §3.2 — Emotion encoder f_θ (MLP + losses)
│   ├── selection.py            # §3.3 — Emotion-augmented skill selection
│   ├── evolution.py            # §3.4 — Emotion-driven skill evolution
│   ├── agent.py                # Algorithm 1 — full agent loop
│   ├── skills/
│   │   └── __init__.py         # Skill libraries (WebShop, ALFWorld, MATH, MBPP)
│   └── envs/
│       ├── webshop.py          # WebShop environment wrapper
│       └── alfworld.py         # ALFWorld environment wrapper
├── train_encoder.py            # Encoder warm-up training script
├── evaluate.py                 # Full evaluation + ablation script
├── scripts/
│   ├── run_webshop_8b.sh       # End-to-end WebShop pipeline
│   ├── run_alfworld_8b.sh      # End-to-end ALFWorld pipeline
│   ├── run_ablations.sh        # Table 2 ablation sweep
│   └── run_tau_sweep.sh        # τ sensitivity sweep
├── configs/
│   ├── default.yaml            # Qwen3-8B default config
│   └── qwen3_14b.yaml          # Qwen3-14B overrides
├── tests/
│   └── test_core.py            # Unit tests (pytest)
├── data/
│   ├── emotion_stories/        # Generated stories (auto-built)
│   └── goemotions/             # GoEmotions test split (JSON)
├── requirements.txt
└── setup.py
```

---

## Installation

```bash
git clone <repo-url>
cd emotion2skill
pip install -e ".[dev]"
```

### Install benchmark environments

**WebShop:**
```bash
git clone https://github.com/princeton-nlp/WebShop
cd WebShop && pip install -e .
```

**ALFWorld:**
```bash
pip install alfworld
alfworld-download
export ALFWORLD_DATA=/path/to/alfworld/data
```

---

## Quick Start

### 1. Build the emotion extractor (offline, one-time)

```python
from emotion2skill.extraction import EmotionExtractor

ext = EmotionExtractor("Qwen/Qwen3-8B", model_key="qwen3-8b", device="cuda")
ext.build(
    stories_path         = "data/emotion_stories/stories.json",
    goemotions_test_path = "data/goemotions/test.json",
    save_path            = "checkpoints/extractor_qwen3_8b.pkl",
)
# Prints selected L* (should be 24) and GoEmotions accuracy (37.2%)
```

### 2. Train the emotion encoder

```bash
python train_encoder.py \
    --model   Qwen/Qwen3-8B \
    --env     webshop \
    --warmup  50 \
    --epochs  30 \
    --extractor checkpoints/extractor_qwen3_8b.pkl \
    --out     checkpoints/encoder_webshop_8b.pt
```

### 3. Evaluate

```bash
python evaluate.py \
    --model     Qwen/Qwen3-8B \
    --env       webshop \
    --extractor checkpoints/extractor_qwen3_8b.pkl \
    --encoder   checkpoints/encoder_webshop_8b.pt \
    --tau       0.3 \
    --evolution \
    --n_runs    3
```

Or use the end-to-end script (runs Steps 1–3 automatically):

```bash
bash scripts/run_webshop_8b.sh 0,1   # GPU IDs
bash scripts/run_alfworld_8b.sh 0,1
```

### 4. Ablations (Table 2)

```bash
bash scripts/run_ablations.sh 0,1
bash scripts/run_tau_sweep.sh 0,1
```

---

## Method Summary

### Emotion Vector Extraction (§3.2, Eq. 1–3)

For each of K=27 GoEmotions concepts, we generate N=100 short stories with Qwen3 and record residual-stream activations at layer ℓ.  The contrastive direction for concept k is:

```
v_k^(ℓ) = (1/N) Σ_n h̄_{k,n}^(ℓ)  −  (1/KN) Σ_{k',n} h̄_{k',n}^(ℓ)
```

PCA-based denoising projects out the top 5 neutral principal components.  The optimal layer L* is selected by maximising GoEmotions 27-class accuracy.  For Qwen3-8B, L*=24 (accuracy 37.2%; random baseline 8.2%).

At inference, the emotion state is:

```
e_t = W_e h_T^(L*) / ‖W_e h_T^(L*)‖₂  ∈ R^K
```

### Emotion Encoder f_θ (§3.2, Eq. 4–8)

A 3-layer MLP (K → 64 → 128 → 128, <27K parameters) maps e_t to:
- **Template index** T_t* (Eq. 4): cosine similarity over 12 learnable prototypes
- **Confidence** c_t ∈ (0,1) (Eq. 5): sigmoid head

Training loss (Eq. 8):  `L = L_ctr + λ * L_cls`
- `L_ctr`: supervised contrastive (positives = same skill ∧ successful episode)
- `L_cls`: cross-entropy over k-means pseudo-labels (k=12)

### Emotion-Augmented Skill Selection (§3.3, Eq. 9–11)

```
P_t = [x_t; {D_i}; T_t*; c_t]   if c_t ≥ τ
      [x_t; {D_i}]               otherwise
```

Setting τ=1 recovers standard text-only selection.

### Emotion-Driven Skill Evolution (§3.4, Eq. 12–17)

Step-to-step emotion shift:
```
δ_t = 1 − cosine_similarity(e_t, e_{t−1})
```

Transition points where δ_t > δ̄ + σ_δ are used to extract 3-step diagnostic windows d_t.  The rewriter prompt includes the current SOP, diagnostics, and episode outcome.

---

## Hyper-parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| K | 27 | GoEmotions emotion dimensions |
| C | 12 | Emotion template slots |
| d' | 128 | Encoder output dimension |
| L* (8B) | 24 | Optimal extraction layer (Qwen3-8B) |
| L* (14B) | 26 | Optimal extraction layer (Qwen3-14B) |
| τ | 0.3 | Confidence gate threshold |
| κ | 0.07 | Contrastive temperature |
| κ' | 0.10 | Classification temperature |
| λ | 0.5 | L_cls weight |
| η | 0.5 | Underperformance threshold for evolution |
| N_warmup | 50 | Warm-up episodes for encoder training |

---

## Skill Libraries

| Benchmark | Skills | Notes |
|-----------|--------|-------|
| WebShop | 8 | ProductSearch, QueryRephrase, ProductCompare, PriceCompare, ReviewAnalysis, OptionSelect, CartManage, PurchaseConfirm |
| ALFWorld | 6 | PickObject, LookObject, CleanObject, HeatObject, CoolObject, PickTwoObjects |
| MATH | 6 | DirectSolve, AlgebraManip, Decompose, Casework, GeomVisualize, VerifyCheck |
| MBPP | 5 | DirectImplement, DecomposeGuard, BruteForceOptimize, PatternRecognize, TestDriven |

---

## Tests

```bash
pytest tests/ -v
```

All tests run on CPU and do not require Qwen3 to be downloaded.

---

## Reproducing Paper Results

All experiments use **Qwen3-8B / Qwen3-14B in non-thinking mode, greedy decoding, 2×H800 GPUs**.

| Script | Reproduces |
|--------|-----------|
| `scripts/run_webshop_8b.sh` | Table 1, Qwen3-8B WebShop row |
| `scripts/run_alfworld_8b.sh` | Table 1, Qwen3-8B ALFWorld row |
| `scripts/run_ablations.sh` | Table 2 ablation rows |
| `scripts/run_tau_sweep.sh` | Table 2 τ sensitivity section |

Replace `Qwen/Qwen3-8B` with `Qwen/Qwen3-14B` and use `configs/qwen3_14b.yaml` for 14B results.

---

## Citation

```bibtex
@inproceedings{emotion2skill2027,
  title     = {Emotion2Skill: Grounding Agent Skill Selection in
               LLM-Internal Emotion Vectors},
  booktitle = {Proceedings of the 41st AAAI Conference on Artificial Intelligence},
  year      = {2027},
}
```

---

## License

This code is released under the **MIT License**.  
The Emotion Stories dataset is released under **CC BY 4.0**.
