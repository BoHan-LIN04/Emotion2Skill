"""
Train the Emotion Encoder  f_θ
================================
Implements the warm-up phase from §3.2 of the paper.

Procedure
---------
1. Run the agent for N_WARMUP episodes under a ReAct baseline policy
   (no emotion signals) to collect training tuples:
       (e_t, s_t^sel, y_t)
   where e_t is the emotion state, s_t^sel is the chosen skill, and
   y_t is the episode outcome.

2. Compute k-means pseudo-labels on the collected e_t vectors (k=C=12).

3. Train the 3-layer MLP encoder with:
       L = L_ctr + λ * L_cls
   - L_ctr : supervised contrastive loss (positives = same skill ∧ y=1)
   - L_cls : cross-entropy over template-prototype cosine similarities

Training converges in under 10 minutes on a single GPU.

Usage
-----
python train_encoder.py \
    --model    Qwen/Qwen3-8B \
    --env      webshop \
    --warmup   50 \
    --epochs   30 \
    --out      checkpoints/encoder_webshop.pt \
    --extractor checkpoints/extractor_qwen3_8b.pkl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from emotion2skill.agent import AgentConfig, Emotion2SkillAgent
from emotion2skill.encoder import (
    C,
    D_PRIME,
    EmotionEncoder,
    EmotionEncoderMLP,
    kmeans_pseudo_labels,
    supervised_contrastive_loss,
    template_classification_loss,
)
from emotion2skill.skills import get_library


# ─────────────────────────────────────────────────────────────────────────────
# Warm-up data collection
# ─────────────────────────────────────────────────────────────────────────────

def collect_warmup_data(
    agent: Emotion2SkillAgent,
    env,
    library,
    n_episodes: int = 50,
    verbose: bool = False,
    format_context_fn=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run episodes WITHOUT emotion augmentation (baseline policy) to collect
    training tuples (e_t, skill_label, outcome_flag).

    Returns
    -------
    e_vectors    : (N, K) float32 — raw emotion states
    skill_labels : (N,) long     — skill indices
    success_flags : (N,) bool    — y_t
    """
    e_list: list[torch.Tensor] = []
    s_list: list[int]          = []
    y_list: list[bool]         = []

    skill_names = library.names()
    skill2idx   = {name: i for i, name in enumerate(skill_names)}

    logs = agent.run_episodes(
        env=env,
        library=library,
        n_episodes=n_episodes,
        use_emotion=False,      # baseline policy
        evolution=False,        # no evolution during warm-up
        verbose=verbose,
        format_context_fn=format_context_fn,
    )

    for ep_log in logs:
        for step in ep_log.steps:
            e_list.append(step.emotion_state)
            s_list.append(skill2idx.get(step.skill_selected, 0))
            y_list.append(ep_log.success)

    if not e_list:
        raise RuntimeError("No warm-up data collected. Check environment setup.")

    e_vectors     = torch.stack(e_list).float()                    # (N, K)
    skill_labels  = torch.tensor(s_list, dtype=torch.long)         # (N,)
    success_flags = torch.tensor(y_list, dtype=torch.bool)         # (N,)

    return e_vectors, skill_labels, success_flags


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_encoder(
    e_vectors:     torch.Tensor,
    skill_labels:  torch.Tensor,
    success_flags: torch.Tensor,
    out_path:      str,
    epochs:        int   = 30,
    batch_size:    int   = 256,
    lr:            float = 3e-4,
    kappa:         float = 0.07,
    kappa_prime:   float = 0.10,
    lam:           float = 0.5,
    device:        str   = "cuda",
    seed:          int   = 42,
) -> EmotionEncoderMLP:
    """
    Train the emotion encoder f_θ.

    Parameters
    ----------
    e_vectors, skill_labels, success_flags : warm-up data
    out_path     : where to save the trained state dict
    epochs       : number of training epochs
    batch_size   : mini-batch size
    lr           : Adam learning rate
    kappa        : contrastive temperature (Eq. 6)
    kappa_prime  : classification temperature (Eq. 7)
    lam          : λ — weight of L_cls (Eq. 8)
    device       : "cuda" | "cpu"
    seed         : random seed

    Returns
    -------
    Trained EmotionEncoderMLP
    """
    torch.manual_seed(seed)
    random.seed(seed)

    # ── k-means pseudo-labels ─────────────────────────────────────────────────
    print("Computing k-means pseudo-labels…")
    pseudo_labels = kmeans_pseudo_labels(e_vectors, n_clusters=C, seed=seed)

    # ── dataset / loader ─────────────────────────────────────────────────────
    dataset = TensorDataset(e_vectors, skill_labels, success_flags, pseudo_labels)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # ── model + optimiser ─────────────────────────────────────────────────────
    model = EmotionEncoderMLP().to(device)
    opt   = optim.Adam(model.parameters(), lr=lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.1)

    # ── training ─────────────────────────────────────────────────────────────
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = total_ctr = total_cls = 0.0
        n_batches  = 0

        for e_b, sk_b, suc_b, ps_b in loader:
            e_b   = e_b.to(device)
            sk_b  = sk_b.to(device)
            suc_b = suc_b.to(device)
            ps_b  = ps_b.to(device)

            r, _ = model(e_b)   # (B, d'), ignores confidence during training

            # Eq. (6): L_ctr
            l_ctr = supervised_contrastive_loss(r, sk_b, suc_b, temperature=kappa)

            # Eq. (7): L_cls
            l_cls = template_classification_loss(
                r, ps_b, model.prototypes, temperature=kappa_prime
            )

            # Eq. (8): L = L_ctr + λ * L_cls
            loss = l_ctr + lam * l_cls

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += loss.item()
            total_ctr  += l_ctr.item()
            total_cls  += l_cls.item()
            n_batches  += 1

        sched.step()
        avg = lambda x: x / max(n_batches, 1)
        row = {
            "epoch": epoch,
            "loss":  round(avg(total_loss), 4),
            "ctr":   round(avg(total_ctr), 4),
            "cls":   round(avg(total_cls), 4),
        }
        history.append(row)
        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{epochs} | "
                f"loss={row['loss']:.4f} "
                f"(ctr={row['ctr']:.4f}, cls={row['cls']:.4f})"
            )

    # ── save ─────────────────────────────────────────────────────────────────
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"Encoder saved to {out_path}")

    # save training history
    hist_path = str(out_path).replace(".pt", "_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    return model


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the Emotion2Skill encoder f_θ")
    p.add_argument("--model",      type=str, default="Qwen/Qwen3-8B",
                   help="HuggingFace model id or local path")
    p.add_argument("--model_key",  type=str, default="qwen3-8b",
                   choices=["qwen3-8b", "qwen3-14b"])
    p.add_argument("--env",        type=str, default="webshop",
                   choices=["webshop", "alfworld", "math", "mbpp"],
                   help="Benchmark environment")
    p.add_argument("--warmup",     type=int, default=50,
                   help="Number of warm-up episodes")
    p.add_argument("--epochs",     type=int, default=30)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--lam",        type=float, default=0.5,
                   help="λ: weight of L_cls")
    p.add_argument("--tau",        type=float, default=0.3,
                   help="Confidence gate threshold")
    p.add_argument("--device",     type=str, default="cuda")
    p.add_argument("--extractor",  type=str, default=None,
                   help="Path to precomputed extractor .pkl")
    p.add_argument("--out",        type=str, default="checkpoints/encoder.pt",
                   help="Output path for encoder state dict")
    p.add_argument("--log_dir",    type=str, default="logs/warmup")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--verbose",    action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    # ── build agent (loads extractor) ─────────────────────────────────────────
    config = AgentConfig(
        model_name_or_path = args.model,
        model_key          = args.model_key,
        device             = args.device,
        extractor_path     = args.extractor,
        encoder_path       = None,   # not yet trained
        tau                = args.tau,
        log_dir            = args.log_dir,
    )
    agent = Emotion2SkillAgent(config)
    agent.load()

    # ── initialise environment ────────────────────────────────────────────────
    library = get_library(args.env)

    if args.env == "webshop":
        from emotion2skill.envs.webshop import WebShopEnv
        env = WebShopEnv(split="train", max_steps=15)
        fmt_fn = lambda tg, h, o: env.format_context(tg, h, o)
    elif args.env == "alfworld":
        from emotion2skill.envs.alfworld import ALFWorldEnv
        env = ALFWorldEnv()
        fmt_fn = lambda tg, h, o: env.format_context(tg, h, o)
    else:
        # MATH / MBPP: environment must be provided externally
        raise NotImplementedError(
            f"Warm-up environment for '{args.env}' not yet implemented. "
            "Use webshop or alfworld for encoder training."
        )

    # ── collect warm-up data ──────────────────────────────────────────────────
    print(f"\nCollecting {args.warmup} warm-up episodes on '{args.env}'…")
    e_vectors, skill_labels, success_flags = collect_warmup_data(
        agent       = agent,
        env         = env,
        library     = library,
        n_episodes  = args.warmup,
        verbose     = args.verbose,
        format_context_fn = fmt_fn,
    )
    print(f"Collected {len(e_vectors)} step records.")

    # ── train encoder ─────────────────────────────────────────────────────────
    print(f"\nTraining encoder for {args.epochs} epochs…")
    train_encoder(
        e_vectors     = e_vectors,
        skill_labels  = skill_labels,
        success_flags = success_flags,
        out_path      = args.out,
        epochs        = args.epochs,
        batch_size    = args.batch_size,
        lr            = args.lr,
        lam           = args.lam,
        device        = args.device,
        seed          = args.seed,
    )


if __name__ == "__main__":
    main()
