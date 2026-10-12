import json
import math
from types import SimpleNamespace

import pytest

from emotion2skill.agent import AgentConfig, Emotion2SkillAgent, Qwen3LLM
from emotion2skill.evolution import EpisodeLog, StepRecord, SkillEvolutionEngine, SkillPerformanceTracker
from emotion2skill.selection import Skill, SkillLibrary, SkillSelector


class FakeLLM:
    def __init__(self):
        self.calls = []
        self.scorings = []

    def chat_prefix(self, system, user):
        return f"<system>{system}<user>{user}<assistant-generation>"

    def score_candidates(self, system, user, candidates):
        self.scorings.append((system, user, candidates))
        return {name: -index - 1.0 for index, name in enumerate(candidates)}

    def __call__(self, system, user):
        self.calls.append((system, user))
        if "Use ReAct reasoning" in system:
            return "Thought: Check the appliance.\nAction: look"
        return "look"


class FakeEnv:
    benchmark = "alfworld"
    backend = "synthetic"
    split = "train"
    action_instruction = "Use take object N from receptacle N."

    def reset(self, task_idx=None):
        self.task_id = f"task-{task_idx}"
        self.task_type = "heat" if task_idx % 2 else "pick"
        self.steps = 0
        return "Visible microwave and mug.", f"Heat mug for task {task_idx} and put it on counter."

    def step(self, action):
        self.steps += 1
        return SimpleNamespace(observation="The door is now open.", reward=float(self.steps == 2),
            done=self.steps == 2, success=self.steps == 2, info={"prediction": action})

    def metadata(self):
        return {"benchmark": self.benchmark, "split": self.split, "backend": self.backend,
                "synthetic": True, "dataset_revision": "mock-v1"}


def library():
    return SkillLibrary([Skill("HeatObject", "Open and heat the mug.", "Heat an object."),
                         Skill("PickObject", "Take and place.", "Pick an object.")])


def agent(tmp_path, **kwargs):
    return Emotion2SkillAgent(AgentConfig(log_dir=str(tmp_path), run_id="test", mock=True,
                              max_steps=3, **kwargs), llm=FakeLLM())


def test_reset_goal_and_full_observation_reach_every_action_and_disk(tmp_path):
    runner = agent(tmp_path)
    logs = runner.run_episodes(FakeEnv(), library(), n_episodes=2, task_order=[0, 1], use_emotion=False)
    assert [item.task_type for item in logs] == ["pick", "heat"]
    for log in logs:
        assert log.task_goal in log.steps[0].action_prompt
        assert log.task_goal in log.steps[1].action_prompt
        assert "Visible microwave and mug." in log.steps[0].action_prompt
        assert "take object N from receptacle N" in log.steps[0].action_prompt
        assert log.final_info == {"prediction": "look"}
    saved = [json.loads(line) for line in (tmp_path / "test/evaluation_000/episode_logs.jsonl").read_text().splitlines()]
    assert saved[0]["benchmark"] == "alfworld"
    assert saved[0]["synthetic"] is True
    assert saved[1]["task_type"] == "heat"
    assert saved[0]["steps"][0]["selection_raw_output"] == "HeatObject"
    assert saved[0]["steps"][0]["observation"] == "Visible microwave and mug."
    assert (tmp_path / "test/evaluation_000/library_round_000.json").is_file()
    with pytest.raises(FileExistsError):
        agent(tmp_path).run_episodes(FakeEnv(), library(), n_episodes=1, use_emotion=False)


def test_warmup_collects_real_states_on_routing_chat_prefix_without_encoder(tmp_path):
    runner = agent(tmp_path)
    prompts = []
    runner.extractor = SimpleNamespace(get_emotion_state=lambda prefix: prompts.append(prefix) or [1.0] + [0.0] * 26)
    logs = runner.run_episodes(FakeEnv(), library(), n_episodes=1, use_emotion=False,
                              collect_emotion=True, phase="warmup")
    assert all(prompt.endswith("<assistant-generation>") for prompt in prompts)
    assert "HeatObject" in prompts[0]
    assert "Skill SOPs" in prompts[0]
    assert logs[0].steps[0].emotion_state == [1.0] + [0.0] * 26
    assert logs[0].steps[0].template_idx == -1
    assert not logs[0].steps[0].emotion_used
    assert "Agent Internal State" not in runner.llm.scorings[0][1]
    assert logs[0].steps[0].extraction_prompt == prompts[0]
    assert "Previous ReAct reasoning" in prompts[1]
    assert "Thought: Check the appliance." in prompts[1]


@pytest.mark.parametrize("admitted", [False, True])
def test_confidence_gate_is_applied_to_selection_and_execution(tmp_path, admitted):
    runner = agent(tmp_path)
    runner.extractor = SimpleNamespace(get_emotion_state=lambda prefix: [1.0] + [0.0] * 26)
    runner.encoder = SimpleNamespace(encode=lambda state: SimpleNamespace(template_idx=8,
        template_text="Nervous before checking the appliance.", confidence=.4 if admitted else .2,
        is_admitted=admitted, to_prompt_str=lambda: "Nervous before checking the appliance."))
    logs = runner.run_episodes(FakeEnv(), library(), n_episodes=1, use_emotion=True)
    assert ("Agent Internal State" in runner.llm.scorings[0][1]) is admitted
    assert ("Nervous before" in logs[0].steps[0].action_prompt) is admitted
    assert logs[0].steps[0].emotion_used is admitted


def test_round_limit_and_frozen_evaluation(tmp_path):
    runner = agent(tmp_path, evolution_rounds=2)
    calls = []
    runner.evolver = SimpleNamespace(run_round=lambda *args, **kwargs: calls.append(args), last_evidence={})
    runner.run_episodes(FakeEnv(), library(), n_episodes=5, use_emotion=False,
                       evolution=True, episode_per_round=1, phase="evolution")
    assert len(calls) == 2
    assert sorted(path.name for path in (tmp_path / "test/evolution_000").glob("library*")) == [
        "library_round_000.json", "library_round_001.json", "library_round_002.json"]
    with pytest.raises(ValueError, match="frozen"):
        runner.run_episodes(FakeEnv(), library(), n_episodes=1, evolution=True)


def test_strict_mock_selection_and_scored_argmax():
    lib = library()
    assert SkillSelector(lambda *args: "2").select("goal", lib).skill_name == "PickObject"
    assert SkillSelector(lambda *args: "heatobject").select("goal", lib).skill_name == "HeatObject"
    for raw in ("Avoid HeatObject; choose PickObject", "unknown", "0", "3"):
        with pytest.raises(ValueError):
            SkillSelector(lambda *args: raw).select("goal", lib)
    scored = FakeLLM()
    scored.score_candidates = lambda *args: {"HeatObject": -10.0, "PickObject": -.2}
    assert SkillSelector(scored).select("goal", lib).skill_name == "PickObject"


def test_qwen_candidate_scoring_uses_full_sequence_probability():
    import torch
    class Tokens:
        def __call__(self, text, **kwargs):
            class Batch(dict):
                def to(self, device):
                    return self
            return Batch(input_ids=torch.tensor([[1, 2]]))
        def apply_chat_template(self, messages, **kwargs):
            return "actual-generation-prefix"
        def encode(self, text, **kwargs):
            return {"A": [3], "BB": [4, 5]}[text]
    class Model:
        device = "cpu"
        def __call__(self, input_ids, attention_mask):
            logits = torch.full((1, input_ids.shape[1], 6), -100.0)
            logits[0, 1, 3], logits[0, 1, 4] = math.log(.4), math.log(.6)
            logits[0, 2, 5], logits[0, 2, 0] = math.log(.7), math.log(.3)
            return SimpleNamespace(logits=logits)
    llm = Qwen3LLM.__new__(Qwen3LLM)
    llm.tokenizer, llm.model, llm.max_context_tokens = Tokens(), Model(), None
    scores = llm.score_candidates("system", "user", ["A", "BB"])
    assert scores["A"] == pytest.approx(math.log(.4))
    assert scores["BB"] == pytest.approx(math.log(.6 * .7))
    assert scores["BB"] > scores["A"]


def test_evolution_calls_real_verbalizer_preserves_context_action_and_outcomes():
    episodes = []
    for index in range(5):
        episode = EpisodeLog(f"episode{index}", task_id=f"task{index}", outcome=.1 * index)
        for step, state in enumerate(([1., 0.], [1., 0.], [0., 1.], [0., 1.], [0., 1.])):
            episode.add_step(StepRecord(step, "full observation " * 30, "HeatObject", 8, .4, state,
                                       f"action {step}", 0.0, context=f"unique goal {index}; full context {step}",
                                       template_text="Nervous"))
        episodes.append(episode)
    calls = []
    def llm(system, prompt):
        calls.append((system, prompt))
        return "The appliance precondition needs checking." if "analyze" in system else "Check the appliance, then heat and place."
    engine = SkillEvolutionEngine(llm, SkillPerformanceTracker(eta=.4))
    changed = engine.run_round(library(), episodes)
    assert "HeatObject" in changed
    assert len(calls) == 2
    assert "unique goal 0" in calls[0][1]
    assert "action 2" in calls[0][1]
    assert "episode4" in calls[1][1]
    assert "The appliance precondition needs checking." in calls[1][1]
    assert engine.last_evidence["skills"]["HeatObject"]["accepted"]


def test_tiny_token_budget_covers_every_long_window_and_bounds_final_rewrite():
    class TinyLLM:
        max_context_tokens = 1900
        max_new_tokens = 120
        def count_prompt_tokens(self, system, user):
            # A character tokenizer makes the small budget exact and reproducible.
            return len(system) + len(user)
        def __call__(self, system, user):
            assert self.count_prompt_tokens(system, user) <= self.max_context_tokens - self.max_new_tokens
            if "analyze" in system:
                return "Check the observed appliance precondition and avoid unsupported retries."
            return "Check the appliance, then heat and place the object."
    episodes = []
    for index in range(5):
        episode = EpisodeLog(f"episode{index}", task_id=f"task{index}", outcome=0.0)
        for step, state in enumerate(([1., 0.], [1., 0.], [0., 1.], [0., 1.], [0., 1.])):
            episode.add_step(StepRecord(step, "complete observation " * 50, "HeatObject", 8, .4, state,
                f"action {step}", 0.0, context=f"Task goal {index}. " + "Detailed recorded context. " * 75,
                template_text="Nervous"))
        episodes.append(episode)
    llm = TinyLLM()
    engine = SkillEvolutionEngine(llm, SkillPerformanceTracker(eta=.4))
    assert engine.run_round(library(), episodes)
    evidence = engine.last_evidence["skills"]["HeatObject"]
    assert set(evidence["covered_window_ids"]) == {f"window_{i:04d}" for i in range(5)}
    assert evidence["rewriter_diagnostics_mode"] == "hierarchical_summary_all_sources"
    assert evidence["rewrite_input_tokens"] <= llm.max_context_tokens - llm.max_new_tokens
    records = evidence["verbalization_records"]
    assert all(record["input_tokens"] <= record["prompt_budget"] for record in records)
    from emotion2skill.evolution import DiagnosticWindow
    for index, window in enumerate(evidence["windows"]):
        source = f"window_{index:04d}"
        fragments = [piece for record in records if record["stage"] == "window_fragment"
                     for piece in record["input_pieces"] if source in piece["sources"]]
        assert fragments
        cursor = 0
        for fragment in fragments:
            assert fragment["character_range"][0] == cursor
            cursor = fragment["character_range"][1]
        original = DiagnosticWindow(**window).verbalize()
        assert cursor == len(original)
        assert "".join(fragment["text"] for fragment in fragments) == original


def test_extractor_preflight_rejects_wrong_width_and_impossible_layer_before_model(tmp_path):
    import pickle
    import numpy as np
    from emotion2skill.agent import validate_inference_assets
    path = tmp_path / "extractor.pkl"
    config = AgentConfig(device="cpu", extractor_path=str(path))
    payload = {"format_version": 2, "model_key": "qwen3-8b", "L_star": 24,
        "matrices": {"contrastive": np.ones((27, 2))},
        "metadata": {"selection_split": "validation", "model_revision": "a" * 40}}
    path.write_bytes(pickle.dumps(payload))
    with pytest.raises(ValueError, match="shape"):
        validate_inference_assets(config, require_encoder=False)
    payload["matrices"]["contrastive"] = np.ones((27, 4096))
    payload["L_star"] = 999
    path.write_bytes(pickle.dumps(payload))
    with pytest.raises(ValueError, match="candidate grid"):
        validate_inference_assets(config, require_encoder=False)
    payload["L_star"] = 24
    path.write_bytes(pickle.dumps(payload))
    result = validate_inference_assets(config, require_encoder=False)
    assert result["model_revision"] == "a" * 40
    assert config.model_revision == "a" * 40


def test_qwen_pins_tokenizer_and_weights_to_same_resolved_commit(monkeypatch):
    import transformers
    commit = "a" * 40
    calls = []
    class Model:
        config = SimpleNamespace(_commit_hash=commit)
        def eval(self): return self
    def tokenizer_load(*args, **kwargs):
        calls.append(("tokenizer", kwargs["revision"]))
        return SimpleNamespace()
    def model_load(*args, **kwargs):
        calls.append(("model", kwargs["revision"]))
        return Model()
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", tokenizer_load)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", model_load)
    llm = Qwen3LLM("Qwen/Qwen3-8B", "cpu", revision=commit)
    assert calls == [("tokenizer", commit), ("model", commit)]
    assert llm.resolved_model_revision == commit
    with pytest.raises(ValueError, match="commit"):
        Qwen3LLM("Qwen/Qwen3-8B", "cpu", revision="b" * 40)


def test_local_identity_changes_when_tokenizer_or_weights_change(tmp_path):
    from emotion2skill.provenance import local_model_identity
    (tmp_path / "model.safetensors").write_bytes(b"synthetic unit-test weights")
    (tmp_path / "config.json").write_text('{"hidden_size":4096}')
    (tmp_path / "tokenizer.json").write_text('{"vocab":{"mug":1}}')
    before = local_model_identity(tmp_path)
    (tmp_path / "tokenizer.json").write_text('{"vocab":{"mug":2}}')
    tokenizer_changed = local_model_identity(tmp_path)
    assert tokenizer_changed["digest"] != before["digest"]
    (tmp_path / "model.safetensors").write_bytes(b"different synthetic unit-test weights")
    assert local_model_identity(tmp_path)["digest"] != tokenizer_changed["digest"]
