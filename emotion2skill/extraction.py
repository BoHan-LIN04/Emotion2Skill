"""
Emotion Vector Extraction
=========================
Implements the contrastive-averaging procedure from §3.2 of the paper.

Pipeline
--------
1. Generate N=100 short stories per GoEmotions concept (K=27).
2. Run a forward pass through Qwen3; hook residual-stream activations.
3. Mean-pool over token positions t >= PREFIX_SKIP.
4. Subtract global mean to obtain contrastive directions v_k^(ℓ).
5. PCA-based denoising: project out top components of neutral activations.
6. Stack into emotion extractor matrix W_e ∈ R^{K×d}.
7. Select optimal layer L* via GoEmotions classification accuracy.

At inference time, project the "Assistant:" delimiter token activation
through W_e and l2-normalise to get e_t ∈ R^K.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── GoEmotions taxonomy (27 concepts) ────────────────────────────────────────
GOEMOTIONS_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise",
]
K = len(GOEMOTIONS_LABELS)          # 27

# ── Layer candidate grids ─────────────────────────────────────────────────────
LAYER_GRID = {
    "qwen3-8b":  [16, 18, 20, 22, 24, 26, 28],   # 36-layer model; L*=24
    "qwen3-14b": [18, 20, 22, 24, 26, 28, 30],   # 40-layer model; L*=26
}
OPTIMAL_LAYER = {
    "qwen3-8b":  24,
    "qwen3-14b": 26,
}

PREFIX_SKIP  = 50   # ignore first 50 token positions when mean-pooling
N_STORIES    = 100  # stories per emotion concept
N_NEUTRAL    = 200  # neutral stories for PCA denoising
PCA_COMPONENTS = 5  # number of neutral principal components to project out


# ─────────────────────────────────────────────────────────────────────────────
# Story-generation prompt
# ─────────────────────────────────────────────────────────────────────────────

STORY_SYSTEM = (
    "You are a creative writer. Write a short story (3–5 sentences) that "
    "naturally embodies the following emotion WITHOUT naming the emotion word "
    "directly. Be subtle and show the emotion through character actions and "
    "inner state."
)

STORY_USER_TEMPLATE = "Emotion concept: {emotion}\n\nWrite the story now:"

NEUTRAL_SYSTEM = (
    "Write a factual description (3–5 sentences) of an everyday activity "
    "such as cooking, commuting, or reading. Be neutral and informative, "
    "with no emotional content."
)


# ─────────────────────────────────────────────────────────────────────────────
# Activation hooking
# ─────────────────────────────────────────────────────────────────────────────

class ResidualStreamHook:
    """Hooks the residual stream output of a specified decoder layer."""

    def __init__(self, model: AutoModelForCausalLM, layer_idx: int):
        self.layer_idx = layer_idx
        self.activation: Optional[torch.Tensor] = None
        self._hook = model.model.layers[layer_idx].register_forward_hook(
            self._capture
        )

    def _capture(self, module, inp, out):
        # out is (hidden_states, ...) for Qwen3 decoder layers
        hidden = out[0] if isinstance(out, tuple) else out
        self.activation = hidden.detach()   # (1, seq_len, d)

    def remove(self):
        self._hook.remove()


# ─────────────────────────────────────────────────────────────────────────────
# Core extraction class
# ─────────────────────────────────────────────────────────────────────────────

class EmotionExtractor:
    """
    Extracts emotion directions from a Qwen3 model's residual stream.

    Parameters
    ----------
    model_name_or_path : str
        HuggingFace model id or local path (e.g. "Qwen/Qwen3-8B").
    model_key : str
        One of "qwen3-8b" | "qwen3-14b" — used to pick layer grid.
    device : str
        "cuda" | "cpu".
    cache_dir : str | None
        Where to save/load precomputed directions.
    """

    def __init__(
        self,
        model_name_or_path: str,
        model_key: str = "qwen3-8b",
        device: str = "cuda",
        cache_dir: Optional[str] = None,
    ):
        self.model_key = model_key.lower()
        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir else None

        # Load model in non-thinking mode, float16 for memory efficiency
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True,
        ).eval()

        self.d_model = self.model.config.hidden_size

        # Will be set after build()
        self.W_e: Optional[torch.Tensor] = None   # (K, d)
        self.L_star: Optional[int] = None
        self._neutral_basis: Optional[torch.Tensor] = None  # (PCA_COMPONENTS, d)

    # ── private helpers ───────────────────────────────────────────────────────

    def _generate_text(self, system: str, user: str, temperature: float = 0.0) -> str:
        """Generate a single text completion with the model."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            ids = self.model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else 1.0,
                top_p=0.9 if temperature > 0 else 1.0,
            )
        generated = ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    @torch.no_grad()
    def _get_layer_activation(self, text: str, layer_idx: int) -> torch.Tensor:
        """
        Forward-pass `text` through the model and return the mean-pooled
        residual-stream activation at `layer_idx` over positions >= PREFIX_SKIP.

        Returns
        -------
        torch.Tensor of shape (d,)
        """
        hook = ResidualStreamHook(self.model, layer_idx)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        _ = self.model(**inputs)
        hook.remove()

        # activation: (1, seq_len, d)
        act = hook.activation.float().squeeze(0)   # (seq_len, d)
        seq_len = act.shape[0]
        start = min(PREFIX_SKIP, seq_len - 1)
        pooled = act[start:].mean(dim=0)            # (d,)
        return pooled.cpu()

    # ── public API ────────────────────────────────────────────────────────────

    def generate_stories(
        self,
        stories_path: str,
        n_stories: int = N_STORIES,
        temperature: float = 0.7,
    ) -> dict[str, list[str]]:
        """
        Generate or load the Emotion Stories dataset.

        Returns
        -------
        dict mapping emotion label -> list of story strings (len = n_stories)
        """
        p = Path(stories_path)
        if p.exists():
            with open(p) as f:
                return json.load(f)

        p.parent.mkdir(parents=True, exist_ok=True)
        stories: dict[str, list[str]] = {}

        for emotion in tqdm(GOEMOTIONS_LABELS, desc="Generating stories"):
            collected: list[str] = []
            attempts = 0
            while len(collected) < n_stories and attempts < n_stories * 5:
                attempts += 1
                story = self._generate_text(
                    STORY_SYSTEM,
                    STORY_USER_TEMPLATE.format(emotion=emotion),
                    temperature=temperature,
                )
                # Quality filter (i): 2–8 sentences
                sentences = [s.strip() for s in story.split(".") if s.strip()]
                if not (2 <= len(sentences) <= 8):
                    continue
                # Quality filter (ii): deduplication by cosine sim > 0.95
                # (simplified: just check exact-ish duplicates here;
                #  full cosine dedup requires embeddings — done offline)
                if story not in collected:
                    collected.append(story)
            stories[emotion] = collected[:n_stories]

        with open(p, "w") as f:
            json.dump(stories, f, indent=2, ensure_ascii=False)

        return stories

    def compute_directions(
        self,
        stories: dict[str, list[str]],
        layer_idx: int,
    ) -> torch.Tensor:
        """
        Compute contrastive emotion directions at a single layer.

        Eq. (2): v_k^(ℓ) = (1/N) Σ_n h̄_{k,n}^(ℓ) - (1/KN) Σ_{k',n} h̄_{k',n}^(ℓ)

        Returns
        -------
        directions : torch.Tensor of shape (K, d)
        """
        all_means: list[torch.Tensor] = []   # (K, d) list of per-class means

        for emotion in tqdm(GOEMOTIONS_LABELS, desc=f"Layer {layer_idx} directions"):
            story_acts: list[torch.Tensor] = []
            for story in stories[emotion]:
                act = self._get_layer_activation(story, layer_idx)
                story_acts.append(act)
            # within-class mean: (d,)
            class_mean = torch.stack(story_acts).mean(dim=0)
            all_means.append(class_mean)

        # Stack: (K, d)
        means = torch.stack(all_means)
        # Global mean: (d,)
        global_mean = means.mean(dim=0)
        # Contrastive directions
        directions = means - global_mean   # (K, d)
        return directions

    def compute_neutral_basis(
        self,
        layer_idx: int,
        n_neutral: int = N_NEUTRAL,
        temperature: float = 0.7,
    ) -> torch.Tensor:
        """
        Compute PCA basis of emotionally neutral activations.

        Returns
        -------
        basis : torch.Tensor of shape (PCA_COMPONENTS, d)
        """
        neutral_acts: list[torch.Tensor] = []
        for _ in tqdm(range(n_neutral), desc="Neutral activations"):
            text = self._generate_text(NEUTRAL_SYSTEM, "", temperature=temperature)
            act = self._get_layer_activation(text, layer_idx)
            neutral_acts.append(act)

        neutral_matrix = torch.stack(neutral_acts)   # (N_NEUTRAL, d)
        # Subtract mean
        neutral_matrix -= neutral_matrix.mean(dim=0, keepdim=True)
        # SVD for PCA
        _, _, Vt = torch.linalg.svd(neutral_matrix, full_matrices=False)
        basis = Vt[:PCA_COMPONENTS]  # (PCA_COMPONENTS, d)
        return basis

    def denoise_directions(
        self,
        directions: torch.Tensor,
        neutral_basis: torch.Tensor,
    ) -> torch.Tensor:
        """
        Project out the top neutral principal components from emotion directions.

        v̂_k = Π⊥ v_k = v_k - Σ_j (v_k · u_j) u_j

        Parameters
        ----------
        directions : (K, d)
        neutral_basis : (PCA_COMPONENTS, d)  — rows are unit vectors

        Returns
        -------
        denoised : (K, d)
        """
        # Ensure unit rows in basis
        basis = F.normalize(neutral_basis, dim=1)   # (PCA_COMPONENTS, d)
        # Projection: (K, PCA_COMPONENTS) @ (PCA_COMPONENTS, d)
        projections = (directions @ basis.T) @ basis   # (K, d)
        return directions - projections

    def classify_goemotions(
        self,
        W_e: torch.Tensor,
        test_texts: list[str],
        test_labels: list[list[int]],
        layer_idx: int,
    ) -> float:
        """
        Evaluate GoEmotions hit-any accuracy for a given layer.

        A prediction is correct if the highest-similarity direction
        matches any annotated label (multi-label hit-any criterion).

        Parameters
        ----------
        W_e : (K, d) emotion directions at this layer
        test_texts : list of input texts
        test_labels : list of lists of correct label indices (multi-label)
        layer_idx : int

        Returns
        -------
        accuracy : float in [0, 1]
        """
        W_e_norm = F.normalize(W_e, dim=1)   # (K, d)
        correct = 0
        for text, gold in zip(test_texts, test_labels):
            act = self._get_layer_activation(text, layer_idx).to(W_e.device)
            act_norm = F.normalize(act.unsqueeze(0), dim=1)        # (1, d)
            sims = (act_norm @ W_e_norm.T).squeeze(0)              # (K,)
            pred = sims.argmax().item()
            if pred in gold:
                correct += 1
        return correct / len(test_texts)

    def build(
        self,
        stories_path: str,
        goemotions_test_path: str,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Full offline build pipeline:
        1. Generate stories.
        2. For each candidate layer: compute directions → denoise → evaluate.
        3. Select L* = argmax accuracy.
        4. Store self.W_e, self.L_star.
        """
        stories = self.generate_stories(stories_path)

        # Load GoEmotions test split for layer selection
        with open(goemotions_test_path) as f:
            go_data = json.load(f)   # list of {"text": str, "labels": [int]}
        test_texts  = [d["text"]   for d in go_data]
        test_labels = [d["labels"] for d in go_data]

        layer_grid = LAYER_GRID[self.model_key]
        best_acc   = -1.0
        best_layer = layer_grid[0]
        best_W_e   = None

        for layer_idx in layer_grid:
            directions     = self.compute_directions(stories, layer_idx)
            neutral_basis  = self.compute_neutral_basis(layer_idx)
            directions_hat = self.denoise_directions(directions, neutral_basis)
            acc = self.classify_goemotions(
                directions_hat, test_texts, test_labels, layer_idx
            )
            print(f"  Layer {layer_idx}: GoEmotions accuracy = {acc:.4f}")
            if acc > best_acc:
                best_acc   = acc
                best_layer = layer_idx
                best_W_e   = directions_hat
                self._neutral_basis = neutral_basis

        self.L_star = best_layer
        self.W_e    = best_W_e   # (K, d)
        print(f"Selected L* = {self.L_star} (accuracy = {best_acc:.4f})")

        if save_path is not None:
            self.save(save_path)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "W_e":           self.W_e.numpy() if self.W_e is not None else None,
            "L_star":        self.L_star,
            "neutral_basis": self._neutral_basis.numpy()
                             if self._neutral_basis is not None else None,
            "model_key":     self.model_key,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        print(f"Saved emotion extractor to {path}")

    @classmethod
    def load(
        cls,
        path: str,
        model_name_or_path: str,
        device: str = "cuda",
    ) -> "EmotionExtractor":
        """Load precomputed W_e and L* without re-running the extraction pipeline."""
        with open(path, "rb") as f:
            payload = pickle.load(f)
        extractor = cls(model_name_or_path, payload["model_key"], device)
        extractor.W_e    = torch.tensor(payload["W_e"], dtype=torch.float32)
        extractor.L_star = payload["L_star"]
        extractor._neutral_basis = (
            torch.tensor(payload["neutral_basis"], dtype=torch.float32)
            if payload["neutral_basis"] is not None else None
        )
        return extractor

    # ── inference-time projection ─────────────────────────────────────────────

    @torch.no_grad()
    def get_emotion_state(
        self,
        prompt_text: str,
        assistant_token: str = "Assistant:",
    ) -> torch.Tensor:
        """
        Eq. (3): e_t = W_e h_T^(L*) / ‖W_e h_T^(L*)‖₂

        Extracts the emotion state at the "Assistant:" delimiter token
        position, then projects and normalises.

        Returns
        -------
        e_t : torch.Tensor of shape (K,)
        """
        assert self.W_e is not None and self.L_star is not None, \
            "Extractor not built. Call build() or load() first."

        hook   = ResidualStreamHook(self.model, self.L_star)
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)

        # Find position of the last "Assistant:" token
        token_ids   = inputs["input_ids"][0]
        assistant_ids = self.tokenizer.encode(
            assistant_token, add_special_tokens=False
        )
        T = self._find_last_subsequence(token_ids.tolist(), assistant_ids)

        _ = self.model(**inputs)
        hook.remove()

        h_T = hook.activation[0, T, :].float().cpu()   # (d,)

        W_e = self.W_e.float()                          # (K, d)
        proj = W_e @ h_T                                # (K,)
        e_t  = F.normalize(proj, dim=0)                 # (K,)
        return e_t

    @staticmethod
    def _find_last_subsequence(token_ids: list[int], subseq: list[int]) -> int:
        """Return start index of last occurrence of subseq in token_ids."""
        n, m = len(token_ids), len(subseq)
        for i in range(n - m, -1, -1):
            if token_ids[i : i + m] == subseq:
                return i + m - 1
        # fallback: last token
        return n - 1
