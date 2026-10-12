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
import copy
import hashlib
import re
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

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
N_NEUTRAL    = 100  # appendix extraction protocol
PCA_COMPONENTS = 10
EXTRACTOR_MODES = {"contrastive", "class_mean", "generic_pca"}
STORY_FILTER_VERSION = "top3-cosine095-prefix50-v1"


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
        model=None,
        tokenizer=None,
        revision: Optional[str] = None,
    ):
        self.model_key = model_key.lower()
        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir else None

        # Resolve a moving remote ref once, then use the same immutable commit
        # for tokenizer and weights. Shared models already carry that commit.
        if revision is None and not Path(model_name_or_path).is_dir():
            if model is not None:
                revision = getattr(model.config, "_commit_hash", None)
            else:
                configuration = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
                revision = getattr(configuration, "_commit_hash", None)
                if not revision or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
                    raise ValueError("Could not resolve an immutable remote model commit")

        # Load model in non-thinking mode, float16 for memory efficiency
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True, revision=revision,
        )
        self.model = model or AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float32 if device == "cpu" else torch.float16,
            device_map=device,
            trust_remote_code=True,
            revision=revision,
        ).eval()

        self.d_model = self.model.config.hidden_size
        self.input_device = getattr(self.model, "device", self.device)
        self.model_name_or_path = model_name_or_path
        self.model_revision = getattr(self.model.config, "_commit_hash", None)
        self.model_identity = getattr(self.model, "_emotion2skill_asset_identity", None)
        if Path(model_name_or_path).is_dir() and self.model_identity is None:
            from .provenance import local_model_identity
            self.model_identity = local_model_identity(model_name_or_path)
        if revision and re.fullmatch(r"[0-9a-fA-F]{40}", revision) and self.model_revision != revision:
            raise ValueError("Loaded model commit does not match the pinned extractor revision")

        # Will be set after build()
        self.W_e: Optional[torch.Tensor] = None   # (K, d)
        self.L_star: Optional[int] = None
        self._neutral_basis: Optional[torch.Tensor] = None  # (PCA_COMPONENTS, d)
        self.mode = "contrastive"
        self.matrices: dict[str, torch.Tensor] = {}
        self.generic_mean: Optional[torch.Tensor] = None
        self.metadata: dict = {}

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
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(self.input_device)
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
        try:
            inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(self.input_device)
            self.model(**inputs)
            act = hook.activation.float().squeeze(0)   # (seq_len, d)
        finally:
            hook.remove()
        seq_len = act.shape[0]
        if seq_len <= PREFIX_SKIP:
            raise ValueError("Story has no token positions beyond the pooling prefix")
        pooled = act[PREFIX_SKIP:].mean(dim=0)       # (d,)
        return pooled.cpu()

    @torch.no_grad()
    def _get_position_activation(self, prompt_text, layer_idx, position=None):
        """Read a specified position of an already formatted generation prefix."""
        hook = ResidualStreamHook(self.model, layer_idx)
        try:
            inputs = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(self.input_device)
            n = inputs["input_ids"].shape[1]
            position = n - 1 if position is None else position
            if not 0 <= position < n:
                raise ValueError(f"Activation position {position} is outside {n} tokens")
            self.model(**inputs)
            return hook.activation[0, position, :].float().cpu()
        finally:
            hook.remove()

    def _story_classifier(self, story):
        output = self._generate_text(
            "Classify the emotions conveyed by a story. Choose exactly three labels "
            "from this taxonomy, ordered from most to least likely: " +
            ", ".join(GOEMOTIONS_LABELS) + ". Return only three comma-separated labels.",
            story, temperature=0.0)
        labels = []
        pattern = r"\b(?:" + "|".join(GOEMOTIONS_LABELS) + r")\b"
        for match in re.finditer(pattern, output.lower()):
            if match.group() not in labels:
                labels.append(match.group())
        return labels[:3] if len(labels) >= 3 else []

    def _story_embedding(self, story):
        return F.normalize(self._get_position_activation(
            story, OPTIMAL_LAYER[self.model_key]), dim=0)

    # ── public API ────────────────────────────────────────────────────────────

    def generate_stories(
        self,
        stories_path: str,
        n_stories: int = N_STORIES,
        temperature: float = 0.0,
        max_attempts_per_class: int = 5000,
    ) -> dict[str, list[str]]:
        """
        Generate or load the Emotion Stories dataset.

        Returns
        -------
        dict mapping emotion label -> list of story strings (len = n_stories)
        """
        p = Path(stories_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        cached = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        if self.model_key == "qwen3-14b":
            # The appendix corpus is generated by 8B and reused by both
            # backbones. Do not silently regenerate or relabel it with 14B.
            source_metadata = cached.get("metadata", {})
            source_stories = cached.get("stories", {})
            if (source_metadata.get("model_key") != "qwen3-8b" or
                source_metadata.get("filter_version") != STORY_FILTER_VERSION or
                not source_metadata.get("complete")):
                raise ValueError("Build the complete canonical Qwen3-8B story corpus before 14B extraction")
            for emotion in GOEMOTIONS_LABELS:
                rows = source_stories.get(emotion, [])
                evidence = {r.get("sha256"): r for r in source_metadata.get("filter_records", {}).get(emotion, [])}
                if len(rows) != n_stories:
                    raise ValueError("Canonical story corpus has an incorrect per-concept count")
                for story in rows:
                    record = evidence.get(hashlib.sha256(story.encode()).hexdigest(), {})
                    if len(record.get("top3", [])) != 3 or emotion not in record["top3"] or record.get("tokens", 0) <= PREFIX_SKIP:
                        raise ValueError("Canonical story corpus lacks matching quality-filter evidence")
            return source_stories
        legacy = "stories" not in cached
        stories = cached if legacy else cached["stories"]
        previous_meta = {} if legacy else cached.get("metadata", {})
        trusted = (previous_meta.get("filter_version") == STORY_FILTER_VERSION and
                   previous_meta.get("temperature") == temperature and
                   previous_meta.get("model_key") == self.model_key)
        attempts_by_emotion = previous_meta.get("attempts", {}) if trusted else {}
        records = previous_meta.get("filter_records", {}) if trusted else {}
        metadata = {"filter_version": STORY_FILTER_VERSION, "model_key": self.model_key,
                    "model_name_or_path": getattr(self, "model_name_or_path", self.model_key),
                    "model_revision": getattr(self, "model_revision", None),
                    "temperature": temperature, "target_per_class": n_stories,
                    "classifier": "same-backbone zero-shot top-3",
                    "embedding": "same-backbone residual last-token cosine",
                    "embedding_layer": OPTIMAL_LAYER[self.model_key],
                    "cosine_threshold": 0.95, "prefix_skip": PREFIX_SKIP,
                    "diversity": "vary scenario details and independent scenario index; greedy decoding",
                    "attempts": attempts_by_emotion, "filter_records": records}

        def save_progress():
            payload = {"schema_version": 2, "metadata": metadata, "stories": stories}
            temporary = p.with_suffix(p.suffix + ".tmp")
            temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            temporary.replace(p)

        settings = ["a library", "a train station", "a garden", "a workshop", "a classroom",
                    "a kitchen", "a mountain trail", "a concert hall", "a market", "a harbor"]
        roles = ["a student", "a gardener", "a musician", "a visitor", "a teacher",
                 "a cook", "an athlete", "a scientist", "an artist", "a shopkeeper"]
        for emotion_idx, emotion in enumerate(tqdm(GOEMOTIONS_LABELS, desc="Generating stories")):
            collected, embeddings, filter_records = [], [], []
            prior_records = {r.get("sha256"): r for r in records.get(emotion, [])}
            for story in stories.get(emotion, []):
                # Unverified legacy caches are rechecked, never silently certified.
                if re.search(r"\b" + re.escape(emotion) + r"\b", story, flags=re.IGNORECASE):
                    continue
                story_hash = hashlib.sha256(story.encode()).hexdigest()
                prior = prior_records.get(story_hash, {})
                labels = prior.get("top3", []) if trusted else []
                if len(labels) != 3:
                    labels = self._story_classifier(story)
                if emotion not in labels or self._story_token_count(story) <= PREFIX_SKIP:
                    continue
                emb = self._story_embedding(story)
                if self._is_duplicate(emb, embeddings):
                    continue
                if self._sentence_count(story) < 2 or self._sentence_count(story) > 8:
                    continue
                collected.append(story); embeddings.append(emb)
                filter_records.append({"sha256": hashlib.sha256(story.encode()).hexdigest(),
                                       "top3": labels, "tokens": self._story_token_count(story)})
                if len(collected) == n_stories:
                    break
            stories[emotion] = collected
            records[emotion] = filter_records
            attempts = int(attempts_by_emotion.get(emotion, 0))
            budget_end = attempts + max_attempts_per_class
            while len(collected) < n_stories and attempts < budget_end:
                attempts += 1
                scenario = (f"Use {settings[(attempts + emotion_idx) % len(settings)]} as the setting "
                            f"and {roles[(attempts // len(settings) + emotion_idx) % len(roles)]} as the protagonist. "
                            f"This is independent scenario {attempts}; vary the event and its outcome.")
                story = self._generate_text(
                    STORY_SYSTEM,
                    STORY_USER_TEMPLATE.format(emotion=emotion) + "\n" + scenario,
                    temperature=temperature,
                )
                attempts_by_emotion[emotion] = attempts
                if not 2 <= self._sentence_count(story) <= 8 or self._story_token_count(story) <= PREFIX_SKIP:
                    continue
                if re.search(r"\b" + re.escape(emotion) + r"\b", story, flags=re.IGNORECASE):
                    continue
                labels = self._story_classifier(story)
                if emotion not in labels:
                    continue
                emb = self._story_embedding(story)
                if story in collected or self._is_duplicate(emb, embeddings):
                    continue
                collected.append(story); embeddings.append(emb)
                filter_records.append({"sha256": hashlib.sha256(story.encode()).hexdigest(),
                                       "top3": labels, "tokens": self._story_token_count(story)})
                save_progress()
            save_progress()
            if len(collected) != n_stories:
                raise RuntimeError(f"Only {len(collected)}/{n_stories} passing {emotion} stories; "
                                   f"partial cache saved to {p}. Resume with a larger attempt budget.")
        metadata["complete"] = True
        save_progress()
        return stories

    @staticmethod
    def _sentence_count(story):
        return len([part for part in re.split(r"[.!?]+(?:\s+|$)", story.strip()) if part.strip()])

    def _story_token_count(self, story):
        return len(self.tokenizer.encode(story, add_special_tokens=False))

    @staticmethod
    def _is_duplicate(embedding, accepted):
        return bool(accepted) and bool((torch.stack(accepted) @ embedding).max() > 0.95)

    def compute_class_means(
        self,
        stories: dict[str, list[str]],
        layer_idx: int,
    ) -> torch.Tensor:
        """
        Compute plain class means at a single layer, retained for the ablation.

        Returns
        -------
        directions : torch.Tensor of shape (K, d)
        """
        all_means: list[torch.Tensor] = []   # (K, d) list of per-class means

        for emotion in tqdm(GOEMOTIONS_LABELS, desc=f"Layer {layer_idx} directions"):
            if len(stories.get(emotion, [])) != N_STORIES:
                raise ValueError(f"Need exactly {N_STORIES} passing stories for {emotion}")
            story_acts: list[torch.Tensor] = []
            for story in stories[emotion]:
                act = self._get_layer_activation(story, layer_idx)
                story_acts.append(act)
            # within-class mean: (d,)
            class_mean = torch.stack(story_acts).mean(dim=0)
            all_means.append(class_mean)

        return torch.stack(all_means)

    def compute_directions(self, stories, layer_idx):
        means = self.compute_class_means(stories, layer_idx)
        return means - means.mean(dim=0)

    def compute_neutral_basis(
        self,
        layer_idx: int,
        n_neutral: int = N_NEUTRAL,
        temperature: float = 0.0,
    ) -> torch.Tensor:
        """
        Compute PCA basis of emotionally neutral activations.

        Returns
        -------
        basis : torch.Tensor of shape (PCA_COMPONENTS, d)
        """
        neutral_acts: list[torch.Tensor] = []
        for i in tqdm(range(n_neutral * 10), desc="Neutral activation attempts"):
            text = self._generate_text(NEUTRAL_SYSTEM,
                f"Describe everyday activity scenario {i+1}: focus on a different routine detail.",
                temperature=temperature)
            if self._story_token_count(text) <= PREFIX_SKIP:
                continue
            neutral_acts.append(self._get_layer_activation(text, layer_idx))
            if len(neutral_acts) == n_neutral:
                break
        if len(neutral_acts) != n_neutral:
            raise RuntimeError("Could not obtain enough neutral stories beyond the token prefix")

        neutral_matrix = torch.stack(neutral_acts)   # (N_NEUTRAL, d)
        # Subtract mean
        neutral_matrix -= neutral_matrix.mean(dim=0, keepdim=True)
        # SVD for PCA
        _, singular_values, Vt = torch.linalg.svd(neutral_matrix, full_matrices=False)
        if (singular_values > 1e-7).sum() < PCA_COMPONENTS:
            raise ValueError("Neutral activations have insufficient rank for top-10 denoising")
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
        if not test_texts or len(test_texts) != len(test_labels):
            raise ValueError("Probe texts/labels must be equally sized and nonempty")
        for text, gold in zip(test_texts, test_labels):
            prompt = self.tokenizer.apply_chat_template([
                {"role": "system", "content": "Read the text and consider its emotional state."},
                {"role": "user", "content": text}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            act = self._get_position_activation(prompt, layer_idx).to(W_e.device)
            act_norm = F.normalize(act.unsqueeze(0), dim=1)        # (1, d)
            sims = (act_norm @ W_e_norm.T).squeeze(0)              # (K,)
            pred = sims.argmax().item()
            if pred in gold:
                correct += 1
        return correct / len(test_texts)

    @staticmethod
    def _load_probe_data(path, expected_split):
        source = Path(path)
        data = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            split = data.get("split")
            acceptable = {expected_split, "dev"} if expected_split == "validation" else {expected_split}
            if split not in acceptable:
                raise ValueError(f"Expected {expected_split} GoEmotions data, got {split!r}")
            records = data.get("records", [])
        else:
            records = data
            # Old lists have no source provenance. At least reject obvious test
            # filenames in the validation slot; new preparation writes metadata.
            if expected_split == "validation" and "test" in source.stem.lower():
                raise ValueError("The GoEmotions test split cannot be used for layer selection")
        if not records:
            raise ValueError(f"Empty GoEmotions {expected_split} split")
        for row in records:
            if not isinstance(row.get("text"), str) or not row.get("labels"):
                raise ValueError("Invalid GoEmotions text/labels record")
            if any(not isinstance(label, int) or not 0 <= label < K for label in row["labels"]):
                raise ValueError("Probe labels must use the 27-emotion taxonomy; remove neutral-only rows")
        return records

    def build(self, stories_path, goemotions_validation_path=None, save_path=None,
              *, goemotions_test_path=None, validation_path=None, test_path=None,
              mode="contrastive"):
        """Select L* using validation only; optionally evaluate fixed L* on test.

        The test path is a separate post-selection evaluation input. A cached
        extractor keeps both denoised contrastive directions and plain means,
        so the class-mean control uses exactly the same stories and layer.
        """
        validation_path = validation_path or goemotions_validation_path
        test_path = test_path or goemotions_test_path
        if not validation_path:
            raise ValueError("Provide a GoEmotions validation path for layer selection")
        if test_path and Path(validation_path).resolve() == Path(test_path).resolve():
            raise ValueError("GoEmotions validation and test paths must differ")
        if mode not in {"contrastive", "class_mean"}:
            raise ValueError("Build emotion directions first; fit generic PCA on train prefixes separately")
        validation = self._load_probe_data(validation_path, "validation")
        test = self._load_probe_data(test_path, "test") if test_path else None
        if test:
            validation_ids = {row["id"] for row in validation if row.get("id")}
            if any(row.get("id") in validation_ids for row in test):
                raise ValueError("GoEmotions validation/test example IDs overlap")
        stories = self.generate_stories(stories_path)
        best_acc, best_layer, best_matrices = -1.0, None, None
        sweep = []
        for layer_idx in LAYER_GRID[self.model_key]:
            means = self.compute_class_means(stories, layer_idx)
            neutral_basis = self.compute_neutral_basis(layer_idx)
            denoised = self.denoise_directions(means - means.mean(0), neutral_basis)
            accuracy = self.classify_goemotions(denoised,
                [row["text"] for row in validation], [row["labels"] for row in validation], layer_idx)
            sweep.append({"layer": layer_idx, "validation_hit_any": accuracy})
            if accuracy > best_acc:
                best_acc, best_layer = accuracy, layer_idx
                best_matrices = {"contrastive": denoised, "class_mean": means}
                self._neutral_basis = neutral_basis
        self.L_star, self.matrices, self.mode = best_layer, best_matrices, mode
        self.W_e = self.matrices[mode]
        self.metadata = {"selection_split": "validation", "validation_path": str(Path(validation_path).resolve()),
            "model_name_or_path": getattr(self, "model_name_or_path", self.model_key),
            "model_revision": getattr(self, "model_revision", None),
            "model_identity": getattr(self, "model_identity", None),
            "validation_sha256": hashlib.sha256(Path(validation_path).read_bytes()).hexdigest(),
            "validation_records": len(validation), "layer_sweep": sweep,
            "pca_components": PCA_COMPONENTS, "neutral_stories": N_NEUTRAL,
            "prefix_skip": PREFIX_SKIP, "story_sha256": hashlib.sha256(Path(stories_path).read_bytes()).hexdigest(),
            "probe_position": "last token of non-thinking chat generation prefix",
            "layer_indexing": "zero-based decoder-layer output"}
        if test:
            self.metadata["test_path"] = str(Path(test_path).resolve())
            self.metadata["test_sha256"] = hashlib.sha256(Path(test_path).read_bytes()).hexdigest()
            self.metadata["test_records"] = len(test)
            self.metadata["post_selection_test_hit_any"] = self.classify_goemotions(
                self.matrices["contrastive"], [row["text"] for row in test],
                [row["labels"] for row in test], self.L_star)
        print(f"Selected L*={self.L_star} using validation hit-any={best_acc:.4f}")
        if save_path:
            self.save(save_path)

    def with_mode(self, mode):
        """Return a shallow extractor view sharing the model and offline assets."""
        if mode not in EXTRACTOR_MODES:
            raise ValueError(f"Unknown extractor mode {mode}")
        if mode not in self.matrices:
            raise ValueError(f"No {mode} matrix in this extractor; build/fit the control first")
        result = copy.copy(self)
        result.mode, result.W_e = mode, self.matrices[mode]
        return result

    @torch.no_grad()
    def fit_generic_pca(self, prompt_texts, layer_idx=None, *, split="train"):
        """Fit PCA-27 on training routing-prefix activations, never test episodes."""
        if split != "train":
            raise ValueError("Generic PCA may only be fitted on training activations")
        layer_idx = self.L_star if layer_idx is None else layer_idx
        if layer_idx is None or (self.L_star is not None and layer_idx != self.L_star):
            raise ValueError("Generic PCA must use the fixed selected extraction layer")
        if len(prompt_texts) <= K:
            raise ValueError("PCA-27 needs at least 28 training activations")
        activations = torch.stack([self._get_position_activation(prompt, layer_idx)
                                   for prompt in prompt_texts])
        mean = activations.mean(0)
        centered = activations - mean
        if min(centered.shape) < K:
            raise ValueError("Activation matrix is too small for PCA-27")
        _, singular_values, components = torch.linalg.svd(centered, full_matrices=False)
        if (singular_values[:K] > 1e-7).sum() < K:
            raise ValueError("Training activations do not have rank 27 for the generic control")
        self.matrices["generic_pca"] = components[:K].cpu()
        self.generic_mean = mean.cpu()
        self.metadata["generic_pca"] = {"fit_split": split, "fit_records": len(prompt_texts),
            "layer": layer_idx, "components": K,
            "prompt_sha256": hashlib.sha256("\n".join(prompt_texts).encode()).hexdigest(),
            "singular_values": singular_values[:K].tolist()}

    @torch.no_grad()
    def export_probe_activations(self, probe_path, output_path, *, split="validation", layer_idx=None):
        """Export actual fixed-layer activations for reproducible probe analyses.

        This does not create a figure or substitute paper values. The archive
        contains hidden activations, emotion projections, multi-hot labels,
        original example IDs, and the exact split/layer/source provenance.
        """
        if split not in {"validation", "test"}:
            raise ValueError("Probe exports must specify validation or test")
        layer_idx = self.L_star if layer_idx is None else layer_idx
        if layer_idx is None:
            raise ValueError("Select an extraction layer before exporting activations")
        records = self._load_probe_data(probe_path, split)
        prompts = [self.tokenizer.apply_chat_template([
            {"role": "system", "content": "Read the text and consider its emotional state."},
            {"role": "user", "content": row["text"]}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False) for row in records]
        activations = torch.stack([self._get_position_activation(prompt, layer_idx) for prompt in prompts])
        labels = torch.zeros(len(records), K, dtype=torch.int64)
        for i, row in enumerate(records):
            labels[i, row["labels"]] = 1
        provenance = {"split": split, "layer": layer_idx, "model_key": self.model_key,
            "mode": self.mode, "layer_indexing": "zero-based decoder-layer output",
            "source_sha256": hashlib.sha256(Path(probe_path).read_bytes()).hexdigest(),
            "source_path": str(Path(probe_path).resolve()), "extractor_metadata": self.metadata}
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"activations": activations.numpy(), "labels": labels.numpy(),
                  "ids": np.asarray([str(row.get("id", i)) for i, row in enumerate(records)]),
                  "metadata_json": np.asarray(json.dumps(provenance, ensure_ascii=False))}
        if layer_idx == self.L_star and self.W_e is not None:
            centered = activations - self.generic_mean if self.mode == "generic_pca" else activations
            arrays["emotion_states"] = F.normalize(centered @ self.W_e.T, dim=1).numpy()
        np.savez_compressed(output, **arrays)
        return provenance

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {"format_version": 2, "mode": self.mode,
            "W_e": self.W_e.cpu().numpy() if self.W_e is not None else None,
            "matrices": {name: matrix.cpu().numpy() for name, matrix in self.matrices.items()},
            "L_star": self.L_star,
            "neutral_basis": self._neutral_basis.cpu().numpy() if self._neutral_basis is not None else None,
            "generic_mean": self.generic_mean.cpu().numpy() if self.generic_mean is not None else None,
            "model_key": self.model_key, "metadata": self.metadata}
        with open(path, "wb") as stream:
            pickle.dump(payload, stream)

    @classmethod
    def load(cls, path, model_name_or_path, device="cuda", *, model=None, tokenizer=None, revision=None):
        with open(path, "rb") as stream:
            payload = pickle.load(stream)
        metadata = payload.get("metadata", {})
        saved_revision = metadata.get("model_revision")
        if revision and saved_revision and revision != saved_revision:
            raise ValueError("Requested revision differs from the extractor checkpoint")
        extractor = cls(model_name_or_path, payload["model_key"], device,
                        model=model, tokenizer=tokenizer, revision=saved_revision or revision)
        expected_identity = metadata.get("model_identity")
        if expected_identity and extractor.model_identity != expected_identity:
            raise ValueError("Local model weights/tokenizer differ from the extractor checkpoint identity")
        extractor.L_star = payload["L_star"]
        extractor.mode = payload.get("mode", "contrastive")
        matrices = payload.get("matrices") or {"contrastive": payload["W_e"]}
        extractor.matrices = {name: torch.tensor(matrix, dtype=torch.float32)
                              for name, matrix in matrices.items()}
        extractor.W_e = extractor.matrices[extractor.mode]
        extractor._neutral_basis = (torch.tensor(payload["neutral_basis"], dtype=torch.float32)
                                    if payload.get("neutral_basis") is not None else None)
        extractor.generic_mean = (torch.tensor(payload["generic_mean"], dtype=torch.float32)
                                  if payload.get("generic_mean") is not None else None)
        extractor.metadata = payload.get("metadata", {"legacy": True})
        if extractor.W_e.shape != (K, extractor.d_model) or not torch.isfinite(extractor.W_e).all():
            raise ValueError("Invalid extractor matrix shape or values")
        return extractor

    @torch.no_grad()
    def get_emotion_state(self, prompt_text, assistant_token=None, *, position=None):
        """Project an actual routing chat-prefix activation at an explicit index.

        With no position, use the final token of the supplied generation prefix.
        ``assistant_token`` is a compatibility option: if explicitly requested,
        absence is an error rather than an observation-tail fallback.
        """
        if self.W_e is None or self.L_star is None:
            raise ValueError("Extractor is not built")
        if assistant_token is not None:
            token_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
            delimiter = self.tokenizer.encode(assistant_token, add_special_tokens=False)
            found = self._find_last_subsequence(token_ids, delimiter)
            if found < 0:
                raise ValueError(f"Requested assistant delimiter {assistant_token!r} is absent")
            if position is not None and position != found:
                raise ValueError("Explicit extraction position disagrees with assistant delimiter")
            position = found
        activation = self._get_position_activation(prompt_text, self.L_star, position)
        if self.mode == "generic_pca":
            if self.generic_mean is None:
                raise ValueError("Generic PCA has not been fitted")
            activation = activation - self.generic_mean
        projection = self.W_e.float().cpu() @ activation
        if not torch.isfinite(projection).all() or projection.norm() < 1e-12:
            raise ValueError("Emotion projection is zero or nonfinite")
        return F.normalize(projection, dim=0)

    @staticmethod
    def _find_last_subsequence(token_ids: list[int], subseq: list[int]) -> int:
        """Return the final token index of the last occurrence, or -1."""
        n, m = len(token_ids), len(subseq)
        for i in range(n - m, -1, -1):
            if token_ids[i : i + m] == subseq:
                return i + m - 1
        return -1
