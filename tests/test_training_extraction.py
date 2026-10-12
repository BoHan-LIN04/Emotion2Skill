import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from emotion2skill.encoder import (
    C, K, EMOTION_TEMPLATES, GOEMOTIONS_LABELS, TEMPLATE_EMOTION_WEIGHTS,
    EmotionEncoder, EmotionEncoderMLP, cluster_template_mapping,
    kmeans_pseudo_labels, supervised_contrastive_loss,
)
from emotion2skill.extraction import EmotionExtractor
from train_encoder import collect_warmup_data, parse_args, split_training_indices, train_encoder


def synthetic_training_data():
    generator = torch.Generator().manual_seed(10)
    features = torch.nn.functional.normalize(torch.randn(48, K, generator=generator), dim=1)
    skills = torch.arange(48) % 4
    outcomes = torch.arange(48) % 3 != 0
    groups = [f"episode-{i // 6}" for i in range(48)]
    return features, skills, outcomes, groups


def test_warmup_records_real_states_without_injection_and_filters_low_signal():
    observed = {}
    def run(**kwargs):
        observed.update(kwargs)
        steps = [SimpleNamespace(emotion_state=torch.zeros(K), skill_selected="S", extraction_prompt="zero"),
                 SimpleNamespace(emotion_state=torch.eye(K)[0], skill_selected="S", extraction_prompt="prefix")]
        return [SimpleNamespace(steps=steps, success=True, episode_id="ep", task_id="task"),
                SimpleNamespace(steps=[steps[0]], success=False, episode_id="filtered-ep", task_id="filtered")]
    agent = SimpleNamespace(run_episodes=run)
    library = SimpleNamespace(names=lambda: ["S"])
    e, skills, y, metadata = collect_warmup_data(
        agent, object(), library, n_episodes=2, return_metadata=True, namespace="webshop", skill_offset=8)
    assert observed["use_emotion"] is False
    assert observed["collect_emotion"] is True
    assert observed["evolution"] is False
    assert observed["phase"] == "warmup"
    assert e.shape == (1, K) and torch.count_nonzero(e) == 1
    assert skills.tolist() == [8] and y.tolist() == [True]
    assert metadata["filtered_steps"] == 2
    assert metadata["extraction_prompts"] == ["prefix"]
    assert metadata["episode_task_ids"] == ["task", "filtered"]
    assert metadata["benchmark"] == "webshop"


def test_kmeans_rejects_degenerate_zero_warmup():
    with pytest.raises(ValueError, match="distinct"):
        kmeans_pseudo_labels(torch.zeros(50, K))


def test_semantic_mapping_is_stable_under_cluster_permutation():
    centroids = torch.zeros(C, K)
    for i, weights in enumerate(TEMPLATE_EMOTION_WEIGHTS):
        for emotion, value in weights.items():
            centroids[i, GOEMOTIONS_LABELS.index(emotion)] = value
    permutation = torch.tensor([6, 2, 11, 0, 7, 4, 8, 1, 10, 5, 9, 3])
    mapping, _ = cluster_template_mapping(centroids[permutation])
    assert mapping.tolist() == permutation.tolist()
    assert "approving" in EMOTION_TEMPLATES[3]
    assert "frustrated" in EMOTION_TEMPLATES[6]
    assert "neutral" in EMOTION_TEMPLATES[11]


def test_failed_anchor_can_have_a_successful_same_skill_positive():
    r = torch.tensor([[1., 0.], [1., 0.], [-1., 0.]], requires_grad=True)
    loss = supervised_contrastive_loss(r, torch.tensor([0, 0, 1]),
                                       torch.tensor([False, True, False]), temperature=1.)
    assert loss.item() == pytest.approx(np.log1p(np.exp(-2)), rel=1e-5)
    loss.backward()
    assert r.grad is not None and torch.isfinite(r.grad).all()


def test_group_validation_keeps_whole_episodes_separate():
    _, _, _, groups = synthetic_training_data()
    train, validation = split_training_indices(48, group_ids=groups, seed=42)
    assert {groups[i] for i in train} .isdisjoint({groups[i] for i in validation})
    assert sorted(train.tolist() + validation.tolist()) == list(range(48))


def test_partial_batch_trains_confidence_and_saves_auditable_checkpoint(tmp_path):
    e, skills, success, groups = synthetic_training_data()
    torch.manual_seed(42)
    initial = EmotionEncoderMLP().w_c.weight.detach().clone()
    output = tmp_path / "encoder.pt"
    model = train_encoder(e, skills, success, output, device="cpu", epochs=2,
                          batch_size=256, group_ids=groups)
    assert not torch.equal(initial, model.w_c.weight.detach())
    checkpoint = torch.load(output, weights_only=True)
    metadata = checkpoint["metadata"]
    assert metadata["optimizer_updates"] == 2
    assert metadata["confidence_trained"]
    assert sorted(metadata["cluster_to_template"]) == list(range(C))
    assert metadata["split_unit"] == "episode"
    assert metadata["hyperparameters"]["confidence_weight"] == 1.
    encoder = EmotionEncoder(str(output), device="cpu", require_trained=True,
                              expected_extractor_mode="contrastive", expected_shuffle_context=False)
    assert 0 <= encoder.encode(e[0]).confidence <= 1
    with pytest.raises(ValueError, match="feature mode"):
        EmotionEncoder(str(output), device="cpu", expected_extractor_mode="class_mean")
    with pytest.raises(ValueError, match="shuffle"):
        EmotionEncoder(str(output), device="cpu", expected_shuffle_context=True)


def test_untrained_legacy_encoder_is_rejected_for_formal_use(tmp_path):
    path = tmp_path / "legacy.pt"
    torch.save(EmotionEncoderMLP().state_dict(), path)
    with pytest.raises(ValueError, match="Legacy"):
        EmotionEncoder(str(path), device="cpu", require_trained=True)
    with pytest.raises(ValueError, match="required"):
        EmotionEncoder(device="cpu", require_trained=True)


def test_shuffled_control_preserves_feature_semantics_and_records_permutation(tmp_path):
    e, skills, y, groups = synthetic_training_data()
    model = train_encoder(e, skills, y, tmp_path / "shuffle.pt", device="cpu", epochs=1,
                          group_ids=groups, shuffle_context=True)
    metadata = model.training_metadata
    permutation = metadata["context_permutation"]
    assert sorted(permutation) == list(range(len(metadata["train_indices"])))
    assert permutation != sorted(permutation)
    assert metadata["shuffle_context"] is True
    assert metadata["template_bank"] == EMOTION_TEMPLATES
    # Context shuffling must preserve the centroids' emotion semantics.
    mapping, _ = cluster_template_mapping(torch.tensor(metadata["semantic_centroids"]))
    assert mapping.tolist() == metadata["cluster_to_template"]


def make_offline_extractor():
    ext = object.__new__(EmotionExtractor)
    ext.model_key = "qwen3-8b"
    ext.mode = "contrastive"
    ext.L_star = None
    ext.W_e = None
    ext.matrices = {}
    ext._neutral_basis = None
    ext.generic_mean = None
    ext.metadata = {}
    ext.d_model = 30
    return ext


def write_probe(path, split, count=1):
    path.write_text(json.dumps({"split": split, "records": [
        {"id": f"{split}-{i}", "text": f"text {split}-{i}", "labels": [i % K]}
        for i in range(count)]}), encoding="utf-8")


def test_layer_selection_never_uses_test_accuracy(tmp_path, monkeypatch):
    import emotion2skill.extraction as extraction
    monkeypatch.setitem(extraction.LAYER_GRID, "qwen3-8b", [1, 2])
    val, test, stories = tmp_path / "validation.json", tmp_path / "test.json", tmp_path / "stories.json"
    write_probe(val, "validation", 1); write_probe(test, "test", 2)
    stories.write_text("{}")
    ext = make_offline_extractor()
    ext.generate_stories = lambda path: {}
    ext.compute_class_means = lambda data, layer: torch.arange(K * 30).reshape(K, 30).float() + layer
    ext.compute_neutral_basis = lambda layer: torch.eye(30)[:10]
    ext.denoise_directions = lambda directions, basis: directions
    calls = []
    def classify(matrix, texts, labels, layer):
        calls.append((len(texts), layer))
        return ({1: .9, 2: .1}[layer] if len(texts) == 1 else .2)
    ext.classify_goemotions = classify
    ext.build(str(stories), goemotions_validation_path=str(val), goemotions_test_path=str(test))
    assert ext.L_star == 1
    assert calls == [(1, 1), (1, 2), (2, 1)]
    assert ext.metadata["selection_split"] == "validation"
    assert ext.metadata["post_selection_test_hit_any"] == .2
    class_mean = ext.with_mode("class_mean")
    assert not torch.equal(class_mean.W_e, ext.W_e)
    with pytest.raises(ValueError, match="validation"):
        ext.build(str(stories), goemotions_test_path=str(test))
    with pytest.raises(ValueError, match="Expected validation"):
        ext.build(str(stories), goemotions_validation_path=str(test))


class FakeBatch(dict):
    def to(self, device):
        return self


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(char) % 60 + 1 for char in text]
    def __call__(self, text, return_tensors=None, **kwargs):
        return FakeBatch(input_ids=torch.tensor([self.encode(text)]))
    def apply_chat_template(self, messages, **kwargs):
        return " ".join(message["content"] for message in messages) + "<assistant>"


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SimpleNamespace(layers=nn.ModuleList([nn.Identity()]))
        self.config = SimpleNamespace(hidden_size=4)
    def forward(self, input_ids):
        x = input_ids.float()
        hidden = torch.stack([x, x.square(), torch.ones_like(x), -x], -1)
        return self.model.layers[0](hidden)


def test_online_extraction_uses_prefix_position_and_missing_delimiter_errors():
    ext = EmotionExtractor("unused", device="cpu", model=FakeModel(), tokenizer=FakeTokenizer())
    ext.L_star = 0
    ext.W_e = torch.arange(K * 4).reshape(K, 4).float() + 1
    ext.matrices = {"contrastive": ext.W_e}
    prefix = "Task<assistant>"
    first = ext._get_position_activation(prefix, 0, position=0)
    last = ext._get_position_activation(prefix, 0)
    assert not torch.equal(first, last)
    expected = torch.nn.functional.normalize(ext.W_e @ last, dim=0)
    assert torch.allclose(ext.get_emotion_state(prefix), expected)
    with pytest.raises(ValueError, match="absent"):
        ext.get_emotion_state(prefix, assistant_token="Assistant:")
    with pytest.raises(ValueError, match="outside"):
        ext.get_emotion_state(prefix, position=999)


def test_generic_pca_requires_train_data_and_persists_centering(tmp_path):
    ext = make_offline_extractor(); ext.L_star = 1
    activations = torch.randn(40, 30, generator=torch.Generator().manual_seed(9))
    ext._get_position_activation = lambda prompt, layer, position=None: activations[int(prompt)]
    with pytest.raises(ValueError, match="training"):
        ext.fit_generic_pca([str(i) for i in range(40)], split="test")
    ext.fit_generic_pca([str(i) for i in range(40)], split="train")
    generic = ext.with_mode("generic_pca")
    assert generic.W_e.shape == (K, 30)
    assert torch.allclose(generic.generic_mean, activations.mean(0))
    expected = torch.nn.functional.normalize(generic.W_e @ (activations[0] - generic.generic_mean), dim=0)
    assert torch.allclose(generic.get_emotion_state("0"), expected)
    assert generic.metadata["generic_pca"]["fit_split"] == "train"


def story_extractor(monkeypatch):
    import emotion2skill.extraction as extraction
    monkeypatch.setattr(extraction, "GOEMOTIONS_LABELS", ["curiosity"])
    ext = make_offline_extractor()
    ext._story_token_count = lambda story: 60
    ext._story_classifier = lambda story: ["anger", "fear", "sadness"] if "wrong" in story else ["curiosity", "desire", "realization"]
    ext._story_embedding = lambda story: torch.tensor([0., 1.]) if "second" in story else torch.tensor([1., 0.])
    return ext


def test_story_filters_deduplicate_and_reentrant_cache_keeps_real_top3(tmp_path, monkeypatch):
    ext = story_extractor(monkeypatch)
    candidates = iter(["Short.", "First event. First reaction.", "First event. First reaction.",
                       "A wrong event. An emotional reaction.", "A second event. Another reaction."])
    calls = []
    def generate(system, user, temperature):
        calls.append((user, temperature))
        return next(candidates)
    ext._generate_text = generate
    path = tmp_path / "stories.json"
    result = ext.generate_stories(str(path), n_stories=2, max_attempts_per_class=5)
    assert len(result["curiosity"]) == 2 and len(calls) == 5
    assert all(temp == 0 for _, temp in calls)
    payload = json.loads(path.read_text())
    assert payload["metadata"]["complete"]
    assert all(len(record["top3"]) == 3 for record in payload["metadata"]["filter_records"]["curiosity"])
    ext._generate_text = lambda *args, **kwargs: pytest.fail("Completed cache should not regenerate stories")
    assert ext.generate_stories(str(path), n_stories=2) == result


def test_story_exhaustion_saves_partial_cache_instead_of_underfilled_success(tmp_path, monkeypatch):
    ext = story_extractor(monkeypatch)
    ext._generate_text = lambda *args, **kwargs: "First event. First reaction."
    path = tmp_path / "partial.json"
    with pytest.raises(RuntimeError, match="partial cache"):
        ext.generate_stories(str(path), n_stories=2, max_attempts_per_class=1)
    assert len(json.loads(path.read_text())["stories"]["curiosity"]) == 1


def load_preparation_module():
    spec = importlib.util.spec_from_file_location("prepare_goemotions", Path(__file__).parents[1] / "scripts" / "prepare_goemotions.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_goemotions_preparation_removes_neutral_and_keeps_official_ids(tmp_path):
    module = load_preparation_module()
    source = tmp_path / "source"; source.mkdir()
    (source / "emotions.txt").write_text("\n".join(module.EMOTION_LABELS + ["neutral"]))
    for filename, suffix in (("train.tsv", "tr"), ("dev.tsv", "va"), ("test.tsv", "te")):
        (source / filename).write_text(f"Good\t0,27\t{suffix}-1\nNeutral\t27\t{suffix}-2\n")
    manifest = module.prepare(source, tmp_path / "prepared", "fixed-commit")
    assert manifest["splits"]["validation"]["records"] == 1
    payload = json.loads((tmp_path / "prepared" / "validation.json").read_text())
    assert payload["records"][0] == {"id": "va-1", "text": "Good", "labels": [0]}
    assert payload["filter"]["neutral_only_rows_removed"] == 1


def test_training_config_inheritance_and_explicit_cli_precedence(tmp_path, monkeypatch):
    parent = tmp_path / "base.yaml"
    parent.write_text("model:\n  name_or_path: Qwen/Qwen3-14B\ntraining:\n  warmup_episodes_per_env: 250\n  batch_size: 64\nextractor:\n  checkpoint: fixed.pkl\n")
    child = tmp_path / "child.yaml"
    child.write_text("extends: base.yaml\ntraining:\n  epochs: 50\n")
    monkeypatch.setattr("sys.argv", ["train_encoder.py", "--config", str(child), "--batch_size", "16"])
    args = parse_args()
    assert args.batch_size == 16 and args.epochs == 50 and args.warmup == 250
    assert args.extractor == "fixed.pkl" and args.model_key == "qwen3-14b"


def test_14b_reuses_verified_8b_corpus_without_regenerating(tmp_path, monkeypatch):
    ext = story_extractor(monkeypatch)
    candidates = iter(["First event. First reaction.", "A second event. Another reaction."])
    ext._generate_text = lambda *args, **kwargs: next(candidates)
    path = tmp_path / "canonical.json"
    expected = ext.generate_stories(str(path), n_stories=2, max_attempts_per_class=2)
    original_bytes = path.read_bytes()
    ext.model_key = "qwen3-14b"
    ext._generate_text = lambda *args, **kwargs: pytest.fail("14B must reuse the canonical 8B corpus")
    assert ext.generate_stories(str(path), n_stories=2) == expected
    assert path.read_bytes() == original_bytes
    missing = tmp_path / "missing.json"
    with pytest.raises(ValueError, match="canonical Qwen3-8B"):
        ext.generate_stories(str(missing), n_stories=2)


def test_tau_one_withholds_even_a_numerically_saturated_confidence():
    encoder = EmotionEncoder(tau=1., device="cpu")
    with torch.no_grad():
        encoder.model.w_c.weight.zero_()
        encoder.model.w_c.bias.fill_(100.)
    summary = encoder.encode(torch.eye(K)[0])
    assert summary.confidence == 1.
    assert not summary.is_admitted


def test_extractor_rejects_a_shared_model_from_another_commit(tmp_path):
    model = FakeModel()
    model.config._commit_hash = "a" * 40
    ext = EmotionExtractor("unused", device="cpu", model=model, tokenizer=FakeTokenizer(), revision="a" * 40)
    ext.L_star = 0
    ext.W_e = torch.ones(K, 4)
    ext.matrices = {"contrastive": ext.W_e}
    ext.metadata = {"model_revision": "a" * 40}
    path = tmp_path / "extractor.pkl"
    ext.save(path)
    other = FakeModel()
    other.config._commit_hash = "b" * 40
    with pytest.raises(ValueError, match="commit"):
        EmotionExtractor.load(path, "unused", device="cpu", model=other, tokenizer=FakeTokenizer())


def test_standalone_extractor_resolves_one_remote_commit_for_tokenizer_and_model(monkeypatch):
    import emotion2skill.extraction as extraction
    sha = "c" * 40
    calls = []
    model = FakeModel(); model.config._commit_hash = sha
    monkeypatch.setattr(extraction.AutoConfig, "from_pretrained", lambda *a, **k: SimpleNamespace(_commit_hash=sha))
    def tokenizer_loader(*args, **kwargs):
        calls.append(("tokenizer", kwargs["revision"]))
        return FakeTokenizer()
    def model_loader(*args, **kwargs):
        calls.append(("model", kwargs["revision"]))
        return model
    monkeypatch.setattr(extraction.AutoTokenizer, "from_pretrained", tokenizer_loader)
    monkeypatch.setattr(extraction.AutoModelForCausalLM, "from_pretrained", model_loader)
    ext = EmotionExtractor("fake/remote-model", device="cpu")
    assert calls == [("tokenizer", sha), ("model", sha)]
    assert ext.model_revision == sha
