"""Encode normalized emotion vectors into template choices and confidence."""

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

C  = 12
D_PRIME = 128


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

# Cluster IDs have no inherent template semantics; persist their assignment.
TEMPLATE_EMOTION_WEIGHTS: list[dict[str, float]] = [
    {"curiosity": 1.0}, {"desire": 1.0}, {"confusion": 1.0},
    {"approval": 1.0}, {"optimism": 1.0}, {"pride": 1.0},
    {"annoyance": 0.7, "anger": 0.3}, {"disappointment": 1.0},
    {"nervousness": 0.7, "fear": 0.3},
    {"realization": 0.8, "surprise": 0.2}, {"amusement": 1.0}, {},
]

assert len(EMOTION_TEMPLATES) == C


class EmotionEncoderMLP(nn.Module):
    """Normalized embeddings with learnable template prototypes and a sigmoid confidence head."""

    def __init__(self, k: int = K, d_prime: int = D_PRIME, n_templates: int = C):
        super().__init__()
        self.k          = k
        self.d_prime    = d_prime
        self.n_templates = n_templates

        self.mlp = nn.Sequential(
            nn.Linear(k, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, d_prime),
        )

        self.prototypes = nn.Parameter(torch.randn(n_templates, d_prime))
        nn.init.xavier_uniform_(self.prototypes.unsqueeze(0)).squeeze(0)

        self.w_c = nn.Linear(d_prime, 1, bias=True)

    def forward(self, e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map (B, K) states to normalized (B, d') representations and (B,) confidence scores."""
        r_raw = self.mlp(e)
        r     = F.normalize(r_raw, dim=-1)
        c     = torch.sigmoid(self.w_c(r)).squeeze(-1)
        return r, c

    def template_scores(self, r: torch.Tensor) -> torch.Tensor:
        """Return (B, C) cosine scores for normalized representations and prototypes."""
        proto_norm = F.normalize(self.prototypes, dim=-1)
        return r @ proto_norm.T

    def select_template(self, r: torch.Tensor) -> torch.Tensor:
        """Return one zero-based template index per representation."""
        return self.template_scores(r).argmax(dim=-1)


def supervised_contrastive_loss(
    r: torch.Tensor,
    skill_labels: torch.Tensor,
    success_flags: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Contrast normalized (B, d') representations using same-skill successful positives.

    Positive eligibility depends on y_j, not the anchor outcome y_i.
    Self pairs and anchors without positives are excluded from the mean.
    """
    B = r.shape[0]
    device = r.device

    sim = (r @ r.T) / temperature

    self_mask = torch.eye(B, dtype=torch.bool, device=device)
    sim = sim.masked_fill(self_mask, float("-inf"))

    sk_match = skill_labels.unsqueeze(1) == skill_labels.unsqueeze(0)
    pos_mask = sk_match & success_flags.unsqueeze(0) & ~self_mask

    n_pos = pos_mask.sum(dim=1).clamp(min=1)

    # All-masked rows would yield NaN in log_softmax; they do not enter the mean.
    all_masked = sim.isinf() & (sim < 0)
    rows_all_inf = all_masked.all(dim=1, keepdim=True)
    sim_safe = sim.masked_fill(rows_all_inf, 0.0)

    log_probs = F.log_softmax(sim_safe, dim=1)

    # Mask before reduction: multiplying masked -inf values by zero yields NaN.
    pos_log = torch.where(pos_mask, log_probs, torch.zeros_like(log_probs))
    loss_per_anchor = -pos_log.sum(dim=1) / n_pos

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
    """Cross-entropy over normalized (B, d') representations and prototype cosine scores."""
    proto_norm = F.normalize(prototypes, dim=-1)
    logits = (r @ proto_norm.T) / temperature
    return F.cross_entropy(logits, pseudo_labels)


def kmeans_pseudo_labels(
    e_vectors: torch.Tensor,
    n_clusters: int = C,
    n_iters: int = 100,
    seed: int = 42,
    return_centroids: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Cluster (N, K) features with Lloyd k-means; optionally return centroids with labels."""
    if e_vectors.ndim != 2 or not torch.isfinite(e_vectors).all():
        raise ValueError("k-means expects a finite (N, K) feature matrix")
    N, D = e_vectors.shape
    if N < n_clusters:
        raise ValueError(f"Need at least {n_clusters} training vectors, got {N}")
    if torch.unique(e_vectors, dim=0).shape[0] < n_clusters:
        raise ValueError("Too few distinct feature vectors for the template clusters")
    generator = torch.Generator(device=e_vectors.device).manual_seed(seed)
    # k-means++ avoids duplicate initial centroids.
    first = torch.randint(N, (1,), generator=generator, device=e_vectors.device)
    idx_list = [int(first.item())]
    for _ in range(1, n_clusters):
        distances = torch.cdist(e_vectors, e_vectors[idx_list]).square().amin(dim=1)
        next_idx = torch.multinomial(distances, 1, generator=generator).item()
        idx_list.append(int(next_idx))
    idx = torch.tensor(idx_list, device=e_vectors.device)
    centroids = e_vectors[idx].clone()

    labels = torch.zeros(N, dtype=torch.long, device=e_vectors.device)
    for _ in range(n_iters):
        dists  = torch.cdist(e_vectors, centroids)
        labels = dists.argmin(dim=1)
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
    """Assign clusters to templates using GoEmotions centroid profiles.

    Generic-PCA clusters require centroids from aligned emotion states.
    The neutral profile favors the smallest maximum absolute activation.
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


@dataclass
class EmotionSummary:
    """Zero-based template choice, confidence, admission flag, and logged (K,) emotion vector."""
    template_idx:  int
    template_text: str
    confidence:    float
    emotion_state: torch.Tensor
    is_admitted:   bool

    def to_prompt_str(self) -> str:
        """Format the summary for prompt insertion; the caller checks admission."""
        return (
            f"[Emotion State] {self.template_text} "
            f"(confidence: {self.confidence:.2f})"
        )


class EmotionEncoder:
    """Validate checkpoint metadata and encode emotion states for confidence-gated routing."""

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
        """Map a finite, normalized (K,) emotion vector to a template and confidence summary."""
        if e_t.shape != (K,) or not torch.isfinite(e_t).all():
            raise ValueError(f"Expected a finite emotion vector of shape ({K},)")
        e = e_t.float().unsqueeze(0).to(self.device)
        r, c = self.model(e)
        tidx  = self.model.select_template(r).item()
        conf  = c.item()

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
