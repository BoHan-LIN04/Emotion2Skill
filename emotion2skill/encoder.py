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

The MLP backbone has 26,624 parameters; including prototypes and the
confidence head the encoder has 28,289 trainable parameters.
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
TEMPLATE_BANK_VERSION = "appendix-12-v1"
EMOTION_TEMPLATES: list[str] = [
    "The agent is curious; it is actively seeking new information.",
    "The agent desires a specific goal and strongly anticipates reaching it.",
    "The agent is confused or uncertain; the current strategy may fail.",
    "The agent feels approving and confident in the current direction.",
    "The agent is optimistic and expects upcoming actions to make progress.",
    "The agent feels proud and self-assured, with high internal confidence.",
    "The agent is frustrated or annoyed by repeated obstacles.",
    "The agent is disappointed; its expectations have not been met.",
    "The agent is nervous or apprehensive and perceives risk.",
    "The agent has a sudden realization or insight that changes its strategy.",
    "The agent feels lightly amused in a low-stakes interaction.",
    "The agent's internal state is neutral; no strong affective signal is present.",
]

# The association is explicit and stored in checkpoints. Cluster numbers have
# no inherent meaning: centroid-to-profile assignment establishes the mapping.
TEMPLATE_EMOTION_WEIGHTS: list[dict[str, float]] = [
    {"curiosity": 1.0}, {"desire": 1.0}, {"confusion": 1.0},
    {"approval": 1.0}, {"optimism": 1.0}, {"pride": 1.0},
    {"annoyance": 0.7, "anger": 0.3}, {"disappointment": 1.0},
    {"nervousness": 0.7, "fear": 0.3},
    {"realization": 0.8, "surprise": 0.2}, {"amusement": 1.0}, {},
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

    Positive examples j have the same skill label and y_j=1. Failed anchors
    can still be compared to successful examples of the same skill. Self pairs
    are excluded; anchors without a positive do not enter the average.

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

    # Match the paper's A+(i): a successful positive j, regardless of y_i.
    sk_match = skill_labels.unsqueeze(1) == skill_labels.unsqueeze(0)  # (B, B)
    pos_mask = sk_match & success_flags.unsqueeze(0) & ~self_mask

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
    return_centroids: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
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
    if e_vectors.ndim != 2 or not torch.isfinite(e_vectors).all():
        raise ValueError("k-means expects a finite (N, K) feature matrix")
    N, D = e_vectors.shape
    if N < n_clusters:
        raise ValueError(f"Need at least {n_clusters} training vectors, got {N}")
    if torch.unique(e_vectors, dim=0).shape[0] < n_clusters:
        raise ValueError("Too few distinct feature vectors for the template clusters")
    generator = torch.Generator(device=e_vectors.device).manual_seed(seed)
    # k-means++ prevents duplicate centroid initialisation.
    first = torch.randint(N, (1,), generator=generator, device=e_vectors.device)
    idx_list = [int(first.item())]
    for _ in range(1, n_clusters):
        distances = torch.cdist(e_vectors, e_vectors[idx_list]).square().amin(dim=1)
        next_idx = torch.multinomial(distances, 1, generator=generator).item()
        idx_list.append(int(next_idx))
    idx = torch.tensor(idx_list, device=e_vectors.device)
    centroids = e_vectors[idx].clone()   # (C, D)

    labels = torch.zeros(N, dtype=torch.long, device=e_vectors.device)
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

    labels = torch.cdist(e_vectors, centroids).argmin(dim=1)
    return (labels, centroids) if return_centroids else labels


def cluster_template_mapping(
    semantic_centroids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match clusters to the appendix bank by GoEmotions semantic profiles.

    For a generic-PCA control, these centroids must be computed from the
    corresponding *emotion* states, not from arbitrary principal components.
    The neutral profile prefers the least concentrated absolute activation.
    This is an auditable semantic heuristic, not an annotated ground truth.
    """
    from scipy.optimize import linear_sum_assignment

    if semantic_centroids.shape != (C, K):
        raise ValueError(f"Expected semantic centroids of shape {(C, K)}")
    if not torch.isfinite(semantic_centroids).all():
        raise ValueError("Semantic centroids contain nonfinite values")
    profiles = torch.zeros(C, K, dtype=semantic_centroids.dtype,
                           device=semantic_centroids.device)
    for j, weights in enumerate(TEMPLATE_EMOTION_WEIGHTS):
        for emotion, weight in weights.items():
            profiles[j, GOEMOTIONS_LABELS.index(emotion)] = weight
    profiles = F.normalize(profiles, dim=1)
    scores = semantic_centroids @ profiles.T
    scores[:, -1] = -semantic_centroids.abs().amax(dim=1)
    cluster_ids, template_ids = linear_sum_assignment(-scores.detach().cpu().numpy())
    mapping = torch.empty(C, dtype=torch.long, device=semantic_centroids.device)
    mapping[torch.as_tensor(cluster_ids, device=mapping.device)] = torch.as_tensor(
        template_ids, device=mapping.device)
    return mapping, scores


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
        expected_extractor_mode: Optional[str] = None,
        expected_shuffle_context: Optional[bool] = None,
        require_trained: bool = False,
    ):
        if not 0 <= tau <= 1:
            raise ValueError("tau must be in [0, 1]")
        self.tau    = tau
        self.device = device
        self.model  = EmotionEncoderMLP().to(device)
        self.metadata: dict = {"trained": False, "confidence_trained": False,
                               "template_bank": EMOTION_TEMPLATES,
                               "template_bank_version": TEMPLATE_BANK_VERSION}
        self.expected_extractor_mode = expected_extractor_mode
        self.expected_shuffle_context = expected_shuffle_context
        self.require_trained = require_trained

        if checkpoint_path is not None:
            self.load(checkpoint_path)
        elif require_trained:
            raise ValueError("A trained encoder checkpoint is required for evaluation")
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
        if e_t.shape != (K,) or not torch.isfinite(e_t).all():
            raise ValueError(f"Expected a finite emotion vector of shape ({K},)")
        e = e_t.float().unsqueeze(0).to(self.device)   # (1, K)
        r, c = self.model(e)                            # (1, d'), (1,)
        tidx  = self.model.select_template(r).item()   # int
        conf  = c.item()                                # float

        return EmotionSummary(
            template_idx  = tidx,
            template_text = EMOTION_TEMPLATES[tidx],
            confidence    = conf,
            emotion_state = e_t.cpu(),
            is_admitted   = self.tau < 1.0 and conf >= self.tau,
        )

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"format_version": 2, "state_dict": self.model.state_dict(),
                    "metadata": self.metadata}, path)

    def load(self, path: str) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=True)
        if "state_dict" not in payload:
            if self.require_trained or self.expected_extractor_mode is not None:
                raise ValueError("Legacy encoder lacks training/mapping metadata; retrain it")
            self.model.load_state_dict(payload)
            self.metadata = {"legacy": True}
            self.model.eval()
            return
        metadata = payload.get("metadata", {})
        if metadata.get("template_bank") != EMOTION_TEMPLATES:
            raise ValueError("Encoder template bank differs from the canonical appendix bank")
        if metadata.get("template_bank_version") != TEMPLATE_BANK_VERSION:
            raise ValueError("Unsupported encoder template-bank version")
        if self.require_trained and (not metadata.get("trained") or
                                     not metadata.get("confidence_trained")):
            raise ValueError("Encoder checkpoint has no verified confidence training metadata")
        if self.expected_extractor_mode is not None and metadata.get(
                "extractor_mode") != self.expected_extractor_mode:
            raise ValueError("Encoder was trained for a different extractor feature mode")
        if self.expected_shuffle_context is not None and metadata.get(
                "shuffle_context") != self.expected_shuffle_context:
            raise ValueError("Encoder training shuffle control does not match evaluation")
        mapping = metadata.get("cluster_to_template")
        if self.require_trained and (not isinstance(mapping, list) or
                                     sorted(mapping) != list(range(C))):
            raise ValueError("Encoder checkpoint lacks a valid semantic cluster mapping")
        self.metadata = metadata
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
