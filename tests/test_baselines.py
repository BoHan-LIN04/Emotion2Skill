import json
from types import SimpleNamespace

import pytest

from emotion2skill.agent import AgentConfig
from emotion2skill.baselines import BM25, build_baseline, build_expel_memory, parse_react_action
from emotion2skill.evolution import EpisodeLog, StepRecord
from emotion2skill.selection import Skill, SkillLibrary


class Env:
    benchmark, backend, split, task_type = "webshop", "synthetic", "test", "webshop"
    action_instruction = "Use search[query] or click[buy now]."
    def __init__(self):
        self.resets = []
    def reset(self, task_idx=None):
        self.task_id = str(task_idx)
        self.resets.append(task_idx)
        return "Search page visible.", "Buy a blue shirt."
    def step(self, action):
        success = len(self.resets) % 2 == 0
        return SimpleNamespace(observation="Feedback.", reward=float(success), done=True,
                               success=success, info={})


class LLM:
    def __init__(self): self.calls = []
    def __call__(self, system, user):
        self.calls.append((system, user))
        if "Reflect" in system:
            return "Next attempt: use the color constraint in the query."
        if "learn general" in system:
            return "1. Check the color before buying."
        if "Reason" in system:
            return "Thought: Need a matching product.\nAction: search[blue shirt]"
        return "search[blue shirt]"


def cfg(tmp_path, **kwargs):
    return AgentConfig(log_dir=str(tmp_path), mock=True, max_steps=2, **kwargs)


def lib():
    return SkillLibrary([Skill("SECRET_SKILL", "SECRET_SOP", "A skill.")])


def train_episode(task_id="train-1"):
    return EpisodeLog("train", benchmark="webshop", task_id=task_id, split="train", backend="synthetic",
        task_goal="Buy a red shirt.", success=True, outcome=1.0,
        steps=[StepRecord(0, "Red shirt visible.", "", -1, 0.0, [], "click[buy now]", 1.0)])


def test_zero_shot_uses_no_skills_or_reasoning_and_loads_no_emotion(tmp_path):
    llm = LLM()
    baseline = build_baseline("ZeroShot", cfg(tmp_path), llm=llm)
    baseline.load()
    log = baseline.run_episodes(Env(), lib(), n_episodes=1)[0]
    assert baseline.encoder is baseline.extractor is None
    assert "SECRET_SKILL" not in llm.calls[0][1]
    assert "SECRET_SOP" not in llm.calls[0][1]
    assert "Thought:" not in llm.calls[0][1]
    assert "Buy a blue shirt." in log.steps[0].action_prompt


def test_react_retains_reasoning_but_executes_only_action(tmp_path):
    baseline = build_baseline("ReAct", cfg(tmp_path), llm=LLM())
    log = baseline.run_episodes(Env(), lib(), n_episodes=1)[0]
    assert log.steps[0].action_taken == "search[blue shirt]"
    assert "Thought: Need a matching product." in log.steps[0].action_raw_output
    assert "SECRET_SOP" not in log.steps[0].action_prompt
    with pytest.raises(ValueError):
        parse_react_action("Thought: decide later")
    with pytest.raises(ValueError):
        parse_react_action("Action: search[x]\nAction: click[y]")


def test_reflexion_retries_same_task_and_clears_cross_task_memory(tmp_path):
    llm, env = LLM(), Env()
    baseline = build_baseline("Reflexion", cfg(tmp_path), llm=llm)
    logs = baseline.run_episodes(env, lib(), n_episodes=2, task_order=["test-a", "test-b"])
    assert env.resets == ["test-a", "test-a", "test-b", "test-b"]
    assert all(len(log.attempts) == 2 for log in logs)
    assert all(log.success for log in logs)
    assert all(len(log.reflections) == 1 for log in logs)
    action_calls = [user for system, user in llm.calls if "Reason" in system]
    assert "self-reflections" not in action_calls[0]
    assert "self-reflections" in action_calls[1]
    assert "self-reflections" not in action_calls[2]


def test_bm25_ranks_query_relevant_experience():
    retrieval = BM25(["heat mug in microwave", "buy cotton blue shirt", "place apple on shelf"])
    assert retrieval.retrieve("blue cotton shirt", 1) == [1]


def test_expel_extracts_insights_from_actual_train_data_and_uses_memory(tmp_path):
    path, llm = tmp_path / "memory.json", LLM()
    artifact = build_expel_memory(llm, [train_episode()], path, allow_mock=True)
    assert artifact["extraction_records"][0]["task_ids"] == ["train-1"]
    baseline = build_baseline("ExpeL", cfg(tmp_path), llm=llm, artifact_path=path)
    log = baseline.run_episodes(Env(), lib(), n_episodes=1, task_order=["test-1"])[0]
    assert "Check the color before buying" in log.steps[0].action_prompt
    assert "Red shirt visible" in log.steps[0].action_prompt
    env = Env()
    obs, goal = env.reset(task_idx="train-1")
    with pytest.raises(ValueError, match="overlap"):
        baseline.run_episode(env, lib(), initial_observation=obs, task_goal=goal)
    with pytest.raises(ValueError, match="Synthetic"):
        build_baseline("ExpeL", AgentConfig(log_dir=str(tmp_path)), llm=llm, artifact_path=path)


def test_missing_upstream_skill_implementations_fail_before_loading_models(tmp_path):
    for method in ("MASA", "EvoSkill", "Trace2Skill"):
        with pytest.raises(FileNotFoundError, match="genuine"):
            build_baseline(method, cfg(tmp_path))


def test_text_similarity_is_named_separately_from_masa(tmp_path):
    baseline = build_baseline("TextSimilarity", cfg(tmp_path), llm=LLM())
    library = SkillLibrary([Skill("Heat", "microwave heat mug", "heat"),
                            Skill("Shopping", "buy blue shirt", "shopping")])
    log = baseline.run_episodes(Env(), library, n_episodes=1)[0]
    assert log.steps[0].skill_selected == "Shopping"
    assert baseline.method_name == "TextSimilarity"
