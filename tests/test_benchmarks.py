import json
from types import SimpleNamespace
import pytest
from emotion2skill.envs import build_env
from emotion2skill.envs.common import load_manifest
from emotion2skill.envs.webshop import WebShopEnv
from emotion2skill.envs.alfworld import ALFWorldEnv
from emotion2skill.envs.math import extract_answer, answers_equivalent, MATHEnv
from emotion2skill.envs.mbpp import execute_code, MBPPEnv

def manifest(tmp_path, benchmark, splits):
    path = tmp_path / f"{benchmark}.json"
    path.write_text(json.dumps({"schema_version": 1, "benchmark": benchmark,
        "dataset_revision": "fixture-commit", "splits": splits}))
    return path

def test_manifests_reject_leakage_even_with_renamed_ids(tmp_path):
    path = manifest(tmp_path, "webshop", {"train": [{"task_id": "a", "index": 3}],
        "dev": [], "test": [{"task_id": "b", "index": 3}]})
    with pytest.raises(ValueError, match="Overlapping source"):
        load_manifest(path, "webshop")

def test_formal_benchmark_requires_manifest():
    for name in ("webshop", "alfworld", "math", "mbpp"):
        with pytest.raises(ValueError, match="manifest"):
            build_env(name)

def test_official_webshop_reset_contract_and_purchase(tmp_path):
    path = manifest(tmp_path, "webshop", {"train": [], "dev": [],
        "test": [{"task_id": "webshop:7", "index": 7}]})
    class OfficialFixture:
        def reset(self, session=None, instruction_text=None):
            assert session == 7
            return "Instruction: select a shirt", None
        def get_instruction_text(self): return "Select a shirt"
        def step(self, action):
            assert action == "click[buy now]"
            return "Purchased", 1.0, True, None
    env = WebShopEnv(path, env=OfficialFixture())
    assert env.reset("webshop:7") == ("Instruction: select a shirt", "Select a shirt")
    assert env.task_id == "webshop:7"
    assert env.step("click[buy now]").success
    assert env.backend == "webshop"

def test_no_implicit_webshop_fallback(tmp_path, monkeypatch):
    path = manifest(tmp_path, "webshop", {"train": [], "dev": [],
        "test": [{"task_id": "webshop:0", "index": 0}]})
    import builtins
    real_import = builtins.__import__
    def blocked(name, *args, **kwargs):
        if name == "gym": raise ImportError("fixture missing gym")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(RuntimeError, match="initialization failed"):
        WebShopEnv(path)

def test_alfworld_success_uses_won_and_pinned_game(tmp_path):
    game = tmp_path / "game.tw-pddl"
    game.write_text("{}")
    path = manifest(tmp_path, "alfworld", {"train": [], "dev": [],
        "test": [{"task_id": "alfworld:heat-fixture", "gamefile": str(game), "task_type": "heat"}]})
    class Wrapper:
        def __init__(self, config, train_eval):
            assert train_eval == "eval_out_of_distribution"
            assert config["env"]["task_types"] == [1, 2, 3, 4, 5, 6]
            assert config["general"]["training_method"] == "dagger"
            self.game_files = [str(game)]
        def init_env(self, batch_size): return self
        def reset(self): return ["Kitchen\nYour task is to heat a mug."], {}
        def step(self, actions): return ["Finished"], [0.0], [True], {"won": [True]}
    env = ALFWorldEnv(path, env_factory=Wrapper)
    obs, goal = env.reset()
    assert goal == "Your task is to heat a mug."
    step = env.step("heat mug 1 with microwave 1")
    assert step.success and step.reward == 0.0
    assert env.task_type == "heat"

def test_alfworld_per_type_uses_episode_metadata():
    metrics = ALFWorldEnv.compute_metrics([{"task_type": "heat", "success": True},
        {"task_type": "cool", "success": False}])
    assert metrics["heat"] == 100 and metrics["cool"] == 0 and metrics["avg"] == 50
    assert metrics["pick"] is None

def test_math_boxed_nested_fractions_and_exact_equivalence():
    assert extract_answer(r"First \boxed{1}. Finally \boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert answers_equivalent(r"\boxed{\frac{1}{2}}", "0.5")
    assert answers_equivalent(r"\sqrt{4}", "2")
    assert answers_equivalent("(x+1)(x-1)", "x^2-1")
    assert answers_equivalent(r"(\frac{1}{2},2)", "(0.5,2)")
    assert answers_equivalent(r"\{1,2\}", r"\{2,1\}")
    assert not answers_equivalent("__import__('os').system('false')", "0")
    assert not answers_equivalent("0.499999", "0.5")

def test_math_fixed_task_manifest(tmp_path):
    path = manifest(tmp_path, "math", {"train": [], "dev": [],
        "test": [{"task_id": "math:fixture", "problem": "Compute 1/2", "answer": r"\boxed{\frac{1}{2}}"}]})
    env = MATHEnv(path)
    assert env.reset()[1] == "Compute 1/2"
    step = env.step(r"final[\boxed{0.5}]")
    assert step.success and step.info["prediction"] == r"final[\boxed{0.5}]"

def test_mbpp_candidate_tests_failure_and_timeout():
    assert execute_code("def add(a,b): return a+b", ["assert add(1,2)==3"])["passed"]
    failed = execute_code("def add(a,b): return 0", ["assert add(1,2)==3"])
    assert not failed["passed"] and "AssertionError" in failed["error"]
    timeout = execute_code("while True: pass", ["assert True"], timeout=0.3)
    assert not timeout["passed"] and timeout["timed_out"]
    assert not execute_code("raise SystemExit(0)", ["assert False"])["passed"]

def test_mbpp_no_test_feedback_retry(tmp_path):
    path = manifest(tmp_path, "mbpp", {"train": [], "dev": [],
        "test": [{"task_id": "mbpp:fixture", "text": "Write add(a,b)", "test_list": ["assert add(1,2)==3"]}]})
    env = MBPPEnv(path)
    env.reset()
    step = env.step("```python\ndef add(a,b): return 0\n```")
    assert step.done and not step.success
    assert step.info["test_list"] == ["assert add(1,2)==3"]
    with pytest.raises(RuntimeError, match="one graded candidate"):
        env.step("final[def add(a,b): return a+b]")
