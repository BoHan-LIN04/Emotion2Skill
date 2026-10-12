"""
Emotion Encoder  f_θ
=====================
Implements §3.2 (Emotion encoder) of the paper.

Architecture
------------
Input : e_t ∈ R^K  (K=27, ℓ2-normalised emotion state)
Hidden: 3-layer MLP  K → 64 → 128 → d' (d'=128), ReLU activations
Output:
  r_t ∈ R^{d'} — encoded representation
  T_t* ∈ {T_1,...,T_C} — template index via cosine similarity (C=12)
  c_t ∈ (0,1) — confidence scalar via sigmoid head

Training objective (Eq. 6–8)
------------------------------
L = L_ctr + λ * L_cls

L_ctr : supervised-contrastive loss (Khosla et al., 2020)
        positives = same effective skill AND successful episode
L_cls : cross-entropy over template prototype similarities
        pseudo-labels from k-means (k=C) on raw emotion vectors

Parameters: < 27 K  (tiny; trains in < 10 min on one GPU)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .extraction import K, GOEMOTIONS_LABELS

# ── Constants ──────────────────────────────────────────────────────────────────
C  = 12    # number of emotion-state template slots
D_PRIME = 128   # encoder output dim


# ─────────────────────────────────────────────────────────────────────────────
# 12 natural-language emotion-state templates
# (designed to span the GoEmotions valence/arousal space)
# ─────────────────────────────────────────────────────────────────────────────
EMOTION_TEMPLATES: list[str] = [
    # T1 – high arousal positive
    "The agent is curious and eager, actively exploring new possibilities.",
    # T2 – approach / desire
    "The agent has a clear desire toward an identified goal and is reaching for it.",
    # T3 – uncertainty / confusion
    "The agent is confused; the current information is ambiguous or contradictory.",
    # T4 – frustration / annoyance
    "The agent is frustrated; repeated attempts have not resolved the situation.",
    # T5 – approval / confidence
    "The agent feels confident and approving; the current path looks correct.",
    # T6 – relief / satisfaction
    "The agent senses relief; a difficulty has been resolved.",
    # T7 – nervousness / low-confidence
    "The agent is nervous; uncertainty is high and stakes feel significant.",
    # T8 – disappointment
    "The agent is disappointed; an expected outcome did not materialise.",
    # T9 – optimism / forward-looking
    "The agent is optimistic; progress is evident and the goal feels reachable.",
    # T10 – caring / careful attention
    "The agent is attending carefully; precision and correctness feel important.",
    # T11 – low-signal (neutral)
    "The agent's internal state is neutral; no strong affective signal is present.",
    # T12 – surprise / realization
    "The agent is surprised or has just had a realisation that changes its plan.",
]

assert len(EMOTION_TEMPLATES) == C


# ─────────────────────────────────────────────────────────────────────────────
# MLP Encoder
# ─────────────────────────────────────────────────────────────────────────────

class EmotionEncoderMLP(nn.Module):
    """
    3-layer MLP:  K → 64 → 128 → d'

    Also holds:
    - C learnable prototype vectors q_j ∈ R^{d'} (Eq. 4)
    - a linear confidence head w_c, b_c (Eq. 5)
    """

    def __init__(self, k: int = K, d_prime: int = D_PRIME, n_templates: int = C):
        super().__init__()
        self.k          = k
        self.d_prime    = d_prime
        self.n_templates = n_templates

        # MLP backbone
        self.mlp = nn.Sequential(
            nn.Linear(k, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, d_prime),
        )

        # Template prototypes q_j ∈ R^{d'}  (Eq. 4)
        self.prototypes = nn.Parameter(torch.randn(n_templates, d_prime))
        nn.init.xavier_uniform_(self.prototypes.unsqueeze(0)).squeeze(0)

        # Confidence head  (Eq. 5)
        self.w_c = nn.Linear(d_prime, 1, bias=True)

    def forward(self, e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        e : (B, K) ℓ2-normalised emotion state vectors

        Returns
        -------
        r : (B, d') encoded representations (ℓ2-normalised)
        c : (B,)   confidence scores in (0, 1)
        """
        r_raw = self.mlp(e)                         # (B, d')
        r     = F.normalize(r_raw, dim=-1)          # (B, d')
        c     = torch.sigmoid(self.w_c(r)).squeeze(-1)  # (B,)
        return r, c

    def template_scores(self, r: torch.Tensor) -> torch.Tensor:
        """
        Cosine similarities between r and all prototype vectors.

        Returns
        -------
        scores : (B, C)
        """
        proto_norm = F.normalize(self.prototypes, dim=-1)   # (C, d')
        return r @ proto_norm.T                              # (B, C)

    def select_template(self, r: torch.Tensor) -> torch.Tensor:
        """
        Eq. (4): T_t* = argmax_j  r_t · q_j / (‖r_t‖ ‖q_j‖)

        Parameters
        ----------
        r : (B, d')

        Returns
        -------
        indices : (B,) LongTensor
        """
        return self.template_scores(r).argmax(dim=-1)   # (B,)


# ─────────────────────────────────────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────────────────────────────────────

def supervised_contrastive_loss(
    r: torch.Tensor,
    skill_labels: torch.Tensor,
    success_flags: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Eq. (6): L_ctr — supervised contrastive loss.

    Positive pairs: same skill_label AND both episodes succeeded (y=1).

    Parameters
    ----------
    r             : (B, d') ℓ2-normalised representations
    skill_labels  : (B,) LongTensor — skill index s_t^sel
    success_flags : (B,) BoolTensor — y_t ∈ {0,1}
    temperature   : κ in the paper

    Returns
    -------
    loss : scalar
    """
    B = r.shape[0]
    device = r.device

    # Pairwise cosine similarities (r is already ℓ2-normalised)
    sim = (r @ r.T) / temperature   # (B, B)

    # Mask out self
    self_mask = torch.eye(B, dtype=torch.bool, device=device)
    sim = sim.masked_fill(self_mask, float("-inf"))

    # Positive mask: same skill AND both succeeded
    sk_match = skill_labels.unsqueeze(1) == skill_labels.unsqueeze(0)  # (B, B)
    both_succ = success_flags.unsqueeze(1) & success_flags.unsqueeze(0)
    pos_mask  = sk_match & both_succ & ~self_mask                       # (B, B)

    n_pos = pos_mask.sum(dim=1).clamp(min=1)  # (B,)

    # Guard: rows where all non-self entries are -inf produce NaN in log_softmax.
    # Replace such rows with zeros before the softmax so they evaluate to a
    # uniform distribution; they will be excluded from the mean anyway.
    all_masked = sim.isinf() & (sim < 0)   # True where entries are -inf
    rows_all_inf = all_masked.all(dim=1, keepdim=True)   # (B, 1)
    sim_safe = sim.masked_fill(rows_all_inf, 0.0)

    log_probs = F.log_softmax(sim_safe, dim=1)     # (B, B)

    # IEEE 754: -inf * 0 = NaN.  The diagonal has log_prob=-inf and pos_mask=0,
    # so we must zero those out explicitly before summing.
    # Use torch.where to avoid the multiply-then-NaN path entirely.
    pos_log = torch.where(pos_mask, log_probs, torch.zeros_like(log_probs))
    loss_per_anchor = -pos_log.sum(dim=1) / n_pos

    # Only average over anchors that have at least one positive
    has_pos = pos_mask.any(dim=1)
    if has_pos.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return loss_per_anchor[has_pos].mean()


def template_classification_loss(
    r: torch.Tensor,
    pseudo_labels: torch.Tensor,
    prototypes: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Eq. (7): L_cls — cross-entropy over prototype cosine similarities.

    Parameters
    ----------
    r             : (B, d') ℓ2-normalised representations
    pseudo_labels : (B,) LongTensor — k-means cluster assignments
    prototypes    : (C, d') learnable prototype matrix
    temperature   : κ' in the paper

    Returns
    -------
    loss : scalar
    """
    proto_norm = F.normalize(prototypes, dim=-1)
    logits = (r @ proto_norm.T) / temperature   # (B, C)
    return F.cross_entropy(logits, pseudo_labels)


# ─────────────────────────────────────────────────────────────────────────────
# k-means pseudo-label generation
# ─────────────────────────────────────────────────────────────────────────────

def kmeans_pseudo_labels(
    e_vectors: torch.Tensor,
    n_clusters: int = C,
    n_iters: int = 100,
    seed: int = 42,
) -> torch.Tensor:
    """
    Simple Lloyd's k-means on emotion vectors to generate pseudo-labels.

    Parameters
    ----------
    e_vectors  : (N, K) raw emotion vectors
    n_clusters : k = C = 12

    Returns
    -------
    labels : (N,) LongTensor
    """
    torch.manual_seed(seed)
    N, D = e_vectors.shape
    # Random init from data points
    idx = torch.randperm(N)[:n_clusters]
    centroids = e_vectors[idx].clone()   # (C, D)

    labels = torch.zeros(N, dtype=torch.long)
    for _ in range(n_iters):
        # Assign
        dists  = torch.cdist(e_vectors, centroids)   # (N, C)
        labels = dists.argmin(dim=1)                  # (N,)
        # Update
        new_centroids = torch.zeros_like(centroids)
        for c in range(n_clusters):
            mask = labels == c
            if mask.any():
                new_centroids[c] = e_vectors[mask].mean(dim=0)
            else:
                new_centroids[c] = centroids[c]
        if torch.allclose(new_centroids, centroids, atol=1e-6):
            break
        centroids = new_centroids

    return labels


# ─────────────────────────────────────────────────────────────────────────────
# Encoded emotion summary (used at inference time)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EmotionSummary:
    """
    φ_t = (T_t*, c_t)

    Attributes
    ----------
    template_idx  : int — index into EMOTION_TEMPLATES (0-based)
    template_text : str — the natural-language template string
    confidence    : float — scalar in (0, 1)
    emotion_state : torch.Tensor of shape (K,) — raw e_t for logging
    is_admitted   : bool — True if c_t >= τ (injected into prompt)
    """
    template_idx:  int
    template_text: str
    confidence:    float
    emotion_state: torch.Tensor
    is_admitted:   bool

    def to_prompt_str(self) -> str:
        """Human-readable string for prompt injection (when admitted)."""
        return (
            f"[Emotion State] {self.template_text} "
            f"(confidence: {self.confidence:.2f})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# High-level encoder wrapper (used by the agent at inference)
# ─────────────────────────────────────────────────────────────────────────────

class EmotionEncoder:
    """
    Wraps EmotionEncoderMLP for inference use.

    Parameters
    ----------
    checkpoint_path : str — path to saved encoder state dict
    tau             : float — confidence gate threshold (default 0.3)
    device          : str
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        tau: float = 0.3,
        device: str = "cuda",
    ):
        self.tau    = tau
        self.device = device
        self.model  = EmotionEncoderMLP().to(device)

        if checkpoint_path is not None:
            state = torch.load(checkpoint_path, map_location=device)
            self.model.load_state_dict(state)
        self.model.eval()

    @torch.no_grad()
    def encode(self, e_t: torch.Tensor) -> EmotionSummary:
        """
        Produce the emotion summary φ_t = (T_t*, c_t) from a raw emotion
        state vector e_t ∈ R^K.

        Parameters
        ----------
        e_t : (K,) ℓ2-normalised emotion state (output of EmotionExtractor)

        Returns
        -------
        EmotionSummary
        """
        e = e_t.float().unsqueeze(0).to(self.device)   # (1, K)
        r, c = self.model(e)                            # (1, d'), (1,)
        tidx  = self.model.select_template(r).item()   # int
        conf  = c.item()                                # float

        return EmotionSummary(
            template_idx  = tidx,
            template_text = EMOTION_TEMPLATES[tidx],
            confidence    = conf,
            emotion_state = e_t.cpu(),
            is_admitted   = conf >= self.tau,
        )

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()
