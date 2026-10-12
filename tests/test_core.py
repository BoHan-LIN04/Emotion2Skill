"""
Unit tests for Emotion2Skill core components.
Run: pytest tests/ -v
"""

import torch
import pytest

from emotion2skill.encoder import (
    C, D_PRIME, EmotionEncoderMLP, EmotionEncoder,
    supervised_contrastive_loss, template_classification_loss,
    kmeans_pseudo_labels, EMOTION_TEMPLATES,
)
from emotion2skill.extraction import K, GOEMOTIONS_LABELS
from emotion2skill.evolution import (
    EpisodeLog, StepRecord,
    compute_shift_magnitudes, find_transition_points,
    extract_diagnostic_windows,
)
from emotion2skill.selection import Skill, SkillLibrary, SkillSelector
from emotion2skill.skills import get_library


# ─────────────────────────────────────────────────────────────────────────────
# Encoder tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEmotionEncoderMLP:

    def test_forward_shapes(self):
        model = EmotionEncoderMLP()
        e = torch.randn(8, K)
        r, c = model(e)
        assert r.shape == (8, D_PRIME)
        assert c.shape == (8,)
        assert c.min() >= 0.0 and c.max() <= 1.0

    def test_r_l2_normalised(self):
        model = EmotionEncoderMLP()
        e = torch.randn(4, K)
        r, _ = model(e)
        norms = r.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(4), atol=1e-5)

    def test_template_scores_shape(self):
        model = EmotionEncoderMLP()
        r = torch.randn(3, D_PRIME)
        scores = model.template_scores(r)
        assert scores.shape == (3, C)

    def test_select_template_in_range(self):
        model = EmotionEncoderMLP()
        e = torch.randn(5, K)
        r, _ = model(e)
        idxs = model.select_template(r)
        assert all(0 <= i.item() < C for i in idxs)


class TestLossFunctions:

    def test_contrastive_loss_scalar(self):
        B = 16
        r   = torch.nn.functional.normalize(torch.randn(B, D_PRIME), dim=-1)
        sk  = torch.randint(0, 4, (B,))
        suc = torch.randint(0, 2, (B,)).bool()
        # Guarantee at least one positive pair so the loss is non-trivial
        sk[0]  = sk[1]  = 0
        suc[0] = suc[1] = True
        loss = supervised_contrastive_loss(r, sk, suc)
        assert loss.ndim == 0
        assert not loss.isnan(), f"Loss should not be NaN, got {loss}"
        assert loss.item() >= 0.0

    def test_cls_loss_scalar(self):
        B   = 8
        r   = torch.nn.functional.normalize(torch.randn(B, D_PRIME), dim=-1)
        pl  = torch.randint(0, C, (B,))
        proto = torch.randn(C, D_PRIME)
        loss = template_classification_loss(r, pl, proto)
        assert loss.ndim == 0
        assert loss.item() >= 0.0

    def test_kmeans_labels_range(self):
        e = torch.randn(50, K)
        labels = kmeans_pseudo_labels(e, n_clusters=C)
        assert labels.shape == (50,)
        assert labels.min() >= 0
        assert labels.max() < C


class TestEmotionEncoder:

    def test_encode_without_checkpoint(self):
        enc = EmotionEncoder(checkpoint_path=None, tau=0.3, device="cpu")
        e_t = torch.randn(K)
        summary = enc.encode(e_t)
        assert 0 <= summary.template_idx < C
        assert 0.0 <= summary.confidence <= 1.0
        assert isinstance(summary.is_admitted, bool)
        assert summary.template_text == EMOTION_TEMPLATES[summary.template_idx]

    def test_confidence_gate(self):
        enc = EmotionEncoder(checkpoint_path=None, tau=0.5, device="cpu")
        # Force confidence by mocking
        e_t = torch.randn(K)
        summary = enc.encode(e_t)
        assert summary.is_admitted == (summary.confidence >= 0.5)

    def test_prompt_str(self):
        enc = EmotionEncoder(checkpoint_path=None, tau=0.3, device="cpu")
        e_t = torch.randn(K)
        summary = enc.encode(e_t)
        ps = summary.to_prompt_str()
        assert "confidence" in ps.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Evolution tests
# ─────────────────────────────────────────────────────────────────────────────

def _make_episode(n_steps: int = 6, success: bool = False) -> EpisodeLog:
    ep = EpisodeLog(episode_id="test_ep", success=success, outcome=float(success))
    for t in range(n_steps):
        ep.add_step(StepRecord(
            step           = t,
            observation    = f"obs_{t}",
            skill_selected = "ProductSearch" if t % 2 == 0 else "QueryRephrase",
            template_idx   = t % C,
            confidence     = 0.3 + 0.05 * t,
            emotion_state  = torch.nn.functional.normalize(torch.randn(K), dim=0),
            action_taken   = f"action_{t}",
            reward         = 0.0,
        ))
    return ep


class TestEvolution:

    def test_shift_magnitudes_length(self):
        ep = _make_episode(6)
        deltas = compute_shift_magnitudes(ep)
        assert len(deltas) == 5   # T-1

    def test_shift_magnitudes_in_range(self):
        ep = _make_episode(8)
        deltas = compute_shift_magnitudes(ep)
        for d in deltas:
            assert 0.0 <= d <= 2.0   # cosine distance ∈ [0, 2]

    def test_transition_points_type(self):
        ep = _make_episode(10)
        deltas = compute_shift_magnitudes(ep)
        pts = find_transition_points(ep, deltas)
        assert isinstance(pts, list)

    def test_diagnostic_windows(self):
        ep = _make_episode(10)
        deltas = compute_shift_magnitudes(ep)
        pts = find_transition_points(ep, deltas)
        windows = extract_diagnostic_windows(ep, pts, "ProductSearch", deltas)
        for w in windows:
            assert len(w.observations) == 3
            assert len(w.skills)       == 3
            assert len(w.templates)    == 3
            assert len(w.shift_mags)   == 3


# ─────────────────────────────────────────────────────────────────────────────
# Skill library tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSkillLibrary:

    def test_webshop_library_count(self):
        lib = get_library("webshop")
        assert len(lib) == 8

    def test_alfworld_library_count(self):
        lib = get_library("alfworld")
        assert len(lib) == 6

    def test_math_library_count(self):
        lib = get_library("math")
        assert len(lib) == 6

    def test_mbpp_library_count(self):
        lib = get_library("mbpp")
        assert len(lib) == 5

    def test_skill_lookup(self):
        lib = get_library("webshop")
        s   = lib["ProductSearch"]
        assert s.name == "ProductSearch"
        assert len(s.sop) > 50

    def test_skill_replace(self):
        lib = get_library("webshop")
        new = Skill("ProductSearch", "New SOP.", "New description.")
        lib.replace("ProductSearch", new)
        assert lib["ProductSearch"].sop == "New SOP."

    def test_format_menu(self):
        lib = get_library("webshop")
        menu = lib.format_menu()
        assert "ProductSearch" in menu
        assert "PurchaseConfirm" in menu

    def test_unknown_benchmark(self):
        with pytest.raises(ValueError):
            get_library("nonexistent")


# ─────────────────────────────────────────────────────────────────────────────
# Skill selector tests (mock LLM)
# ─────────────────────────────────────────────────────────────────────────────

class TestSkillSelector:

    def _mock_llm(self, system: str, user: str) -> str:
        return "ProductSearch"

    def test_select_returns_valid_skill(self):
        lib = get_library("webshop")
        sel = SkillSelector(self._mock_llm)
        result = sel.select("Find a shirt under $50.", lib)
        assert result.skill_name in lib.names()

    def test_parse_fallback(self):
        lib = get_library("webshop")
        # LLM returns garbage
        sel = SkillSelector(lambda s, u: "UNKNOWN SKILL XYZ")
        result = sel.select("test", lib)
        assert result.skill_name in lib.names()   # fallback to first skill

    def test_emotion_not_admitted(self):
        from emotion2skill.encoder import EmotionSummary
        lib = get_library("webshop")
        sel = SkillSelector(self._mock_llm)
        # emotion with c < τ → not admitted → emotion_used should be False
        summary = EmotionSummary(
            template_idx=0,
            template_text=EMOTION_TEMPLATES[0],
            confidence=0.1,   # below τ=0.3
            emotion_state=torch.zeros(K),
            is_admitted=False,
        )
        result = sel.select("test", lib, emotion=summary)
        assert result.emotion_used is False
