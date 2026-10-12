import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from emotion2skill.config import DEFAULT_CONFIG, load_config, resolve_config
from emotion2skill.envs import build_env

_SPEC = importlib.util.spec_from_file_location("evaluation_entry", Path(__file__).parents[1] / "evaluate.py")
evaluation = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(evaluation)

def test_yaml_inheritance_and_cli_override(tmp_path):
    (tmp_path / "base.yaml").write_text("evaluation:\n  n_runs: 2\nencoder:\n  tau: 0.6\n")
    (tmp_path / "child.yaml").write_text("extends: base.yaml\nmodel:\n  name_or_path: Qwen/Qwen3-14B\n  key: qwen3-14b\n")
    config = resolve_config(tmp_path / "child.yaml", {"evaluation.n_runs": 3, "encoder.tau": 0.3})
    assert config["evaluation"]["n_runs"] == 3
    assert config["encoder"]["tau"] == 0.3
    assert config["model"]["key"] == "qwen3-14b"
    with pytest.raises(ValueError, match="does not match"):
        resolve_config(None, {"model.name_or_path": "Qwen/Qwen3-14B", "model.key": "qwen3-8b"})

def test_task_subset_fixed_order_varies_and_reserves_warmup():
    env = build_env("webshop", split="train", mock=True)
    order_a = evaluation.task_order(env, 10, 1, reserved=250)
    order_b = evaluation.task_order(env, 10, 2, reserved=250)
    assert set(order_a) == set(order_b) and order_a != order_b
    assert set(order_a).isdisjoint({task["task_id"] for task in env.tasks[:250]})
    with pytest.raises(ValueError, match="eligible"):
        evaluation.task_order(env, 251, 1, reserved=250)

def test_sample_std_and_synthetic_rejection():
    runs = [{"synthetic": False, "metrics": {"accuracy": v}} for v in [10, 20, 30]]
    assert evaluation.summarize_runs(runs)["accuracy"]["sample_std"] == 10
    with pytest.raises(ValueError, match="non-synthetic"):
        evaluation.summarize_runs([{"synthetic": True, "metrics": {"accuracy": 100}}])
    with pytest.raises(ValueError, match="Synthetic"):
        evaluation.compute_metrics("math", [SimpleNamespace(backend="synthetic", success=True)])

def test_tau_selection_never_uses_test_split():
    with pytest.raises(SystemExit):
        evaluation.parse_args(["--tau_sweep", "--split", "test"])
    _, config = evaluation.parse_args(["--tau_sweep"])
    assert config["evaluation"]["split"] == "dev"

def test_mixed_evolution_then_frozen_test_with_independent_runs(tmp_path, monkeypatch):
    calls = []
    class Runner:
        def __init__(self):
            self.config = SimpleNamespace(max_steps=15)
            self.llm = None
        def run_episodes(self, env, library, **kwargs):
            calls.append((env.benchmark, env.split, kwargs["phase"], kwargs["evolution"], list(kwargs["task_order"]), kwargs["run_id"]))
            result = []
            for task_id in kwargs["task_order"]:
                result.append(SimpleNamespace(backend="synthetic", success=False,
                    to_dict=lambda task_id=task_id: {"task_id": task_id, "synthetic": True, "steps": []}))
            return result
    monkeypatch.setattr(evaluation, "_agent", lambda *args: Runner())
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["evaluation"].update(mock=True, n_episodes=2, n_runs=2)
    config["evolution"].update(rounds=2, episode_per_round=2)
    config["training"]["warmup_episodes_per_env"] = 3
    result = evaluation.run_evaluation(config, tmp_path)
    assert result["synthetic"] and result["summary"] is None and not result["paper_eligible"]
    for run_index in range(2):
        run = calls[run_index * 5:(run_index + 1) * 5]
        assert [entry[:4] for entry in run[:4]] == [
            ("webshop", "train", "evolution", True), ("alfworld", "train", "evolution", True),
            ("webshop", "train", "evolution", True), ("alfworld", "train", "evolution", True)]
        assert run[-1][:4] == ("webshop", "test", "evaluation", False)
        for benchmark in ("webshop", "alfworld"):
            training_ids = [task for entry in run[:4] if entry[0] == benchmark for task in entry[4]]
            assert len(set(training_ids)) == 4
    assert len({entry[5] for entry in calls}) == len(calls)
    assert len(list(tmp_path.glob("run_*/evaluation_metadata.json"))) == 2

def test_round_curves_use_frozen_same_test_tasks_without_feedback(tmp_path, monkeypatch):
    calls = []
    class Runner:
        def __init__(self):
            self.config = SimpleNamespace(max_steps=15)
            self.llm = None
        def run_episodes(self, env, library, **kwargs):
            calls.append((env.split, kwargs["phase"], kwargs["evolution"], list(kwargs["task_order"])))
            return [SimpleNamespace(backend="synthetic", success=False,
                to_dict=lambda task_id=task_id: {"task_id": task_id, "synthetic": True, "steps": []})
                for task_id in kwargs["task_order"]]
    monkeypatch.setattr(evaluation, "_agent", lambda *args: Runner())
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["evaluation"].update(mock=True, n_episodes=2, n_runs=1, evaluate_each_round=True)
    config["evolution"].update(rounds=2, episode_per_round=2)
    config["training"]["warmup_episodes_per_env"] = 3
    result = evaluation.run_evaluation(config, tmp_path)
    round_calls = [entry for entry in calls if entry[1] == "round_evaluation"]
    assert len(round_calls) == 3
    assert all(entry[:3] == ("test", "round_evaluation", False) for entry in round_calls)
    assert round_calls[0][3] == round_calls[1][3] == round_calls[2][3] == calls[-1][3]
    assert calls[0][:3] == ("test", "round_evaluation", False)
    assert [record["round"] for record in result["runs"][0]["rounds"]] == [0, 1, 2]
    assert result["round_summaries"] is None  # synthetic outputs stay excluded
    assert len(list(tmp_path.glob("run_*/round_evaluation/round_*/metrics.json"))) == 3
    assert all(not record["feeds_evolution"] for record in result["runs"][0]["rounds"])

def _provenance_fixture(tmp_path):
    import torch
    from emotion2skill.envs.common import load_manifest
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["training"]["warmup_episodes_per_env"] = 2
    entries = []
    for benchmark in ("webshop", "alfworld"):
        def record(index):
            result = {"task_id": f"{benchmark}:{index}"}
            if benchmark == "webshop": result["index"] = index
            else:
                result.update(gamefile=str(tmp_path / f"game_{index}.tw-pddl"), task_type="heat")
            return result
        path = tmp_path / f"{benchmark}.json"
        path.write_text(json.dumps({"schema_version": 1, "benchmark": benchmark,
            "dataset_revision": "fixture-revision", "splits": {
                "train": [record(i) for i in range(6)], "dev": [record(100)], "test": [record(101)]}}))
        config["datasets"][benchmark]["manifest"] = str(path)
        manifest = load_manifest(path, benchmark)
        entries.append({"namespace": benchmark, "benchmark": benchmark,
            "episodes": 2, "episode_task_ids": [f"{benchmark}:2", f"{benchmark}:4"],
            "manifest_sha256": manifest["manifest_sha256"]})
    checkpoint = tmp_path / "encoder.pt"
    source = {"environment_split": "train", "warmup": entries}
    torch.save({"metadata": {"source": source}}, checkpoint)
    config["encoder"]["checkpoint"] = str(checkpoint)
    return config, source

def test_exact_warmup_ids_bound_to_manifest_and_excluded(tmp_path):
    config, source = _provenance_fixture(tmp_path)
    provenance = evaluation.validate_protocol_provenance(config)
    assert provenance["warmup_task_ids"]["webshop"] == ["webshop:2", "webshop:4"]
    env = build_env("webshop", config["datasets"]["webshop"]["manifest"], split="train", mock=True)
    order = evaluation.task_order(env, 3, 42,
        excluded_task_ids=provenance["warmup_task_ids"]["webshop"])
    assert set(order) == {"webshop:0", "webshop:1", "webshop:3"}

def test_checkpoint_manifest_reordering_fails_before_agent_construction(tmp_path, monkeypatch):
    config, source = _provenance_fixture(tmp_path)
    path = Path(config["datasets"]["webshop"]["manifest"])
    data = json.loads(path.read_text())
    data["splits"]["train"].reverse()
    path.write_text(json.dumps(data))
    monkeypatch.setattr(evaluation, "_agent", lambda *args: pytest.fail("Backbone construction must not occur"))
    with pytest.raises(ValueError, match="manifest hash differs"):
        evaluation.run_evaluation(config, tmp_path / "unused-output")

def test_encoder_needs_all_episodes_and_both_domain_provenance(tmp_path):
    import torch
    config, source = _provenance_fixture(tmp_path)
    del source["warmup"][0]["episode_task_ids"]
    torch.save({"metadata": {"source": source}}, config["encoder"]["checkpoint"])
    with pytest.raises(ValueError, match="complete unique"):
        evaluation.validate_protocol_provenance(config)
    source["warmup"] = source["warmup"][1:]
    torch.save({"metadata": {"source": source}}, config["encoder"]["checkpoint"])
    with pytest.raises(ValueError, match="both WebShop and ALFWorld"):
        evaluation.validate_protocol_provenance(config)

def test_warmup_ids_cannot_enter_dev_test_even_with_declared_hash(tmp_path):
    import torch
    config, source = _provenance_fixture(tmp_path)
    source["warmup"][0]["episode_task_ids"] = ["webshop:100", "webshop:4"]
    torch.save({"metadata": {"source": source}}, config["encoder"]["checkpoint"])
    with pytest.raises(ValueError, match="overlap current dev/test"):
        evaluation.validate_protocol_provenance(config)

def test_external_baseline_source_tasks_bound_to_explicit_namespace(tmp_path):
    config, source = _provenance_fixture(tmp_path)
    config["evaluation"]["method"] = "MASA"
    artifact = tmp_path / "external.json"
    config["evaluation"]["baseline_artifact"] = str(artifact)
    data = {"benchmark": "webshop", "source_split": "train", "source_task_ids": ["webshop:0"]}
    artifact.write_text(json.dumps(data))
    assert evaluation.validate_protocol_provenance(config)["baseline_source_task_ids"] == ["webshop:0"]
    data["benchmark"] = "alfworld"
    artifact.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="target benchmark namespace"):
        evaluation.validate_protocol_provenance(config)
    data.update(benchmark="webshop", source_task_ids=["webshop:101"])
    artifact.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="overlap current dev/test"):
        evaluation.validate_protocol_provenance(config)
    data["source_task_ids"] = ["alfworld:0"]
    artifact.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="namespace alfworld"):
        evaluation.validate_protocol_provenance(config)

def test_mock_protocol_does_not_claim_training_provenance():
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["evaluation"]["mock"] = True
    assert evaluation.validate_protocol_provenance(config) == {
        "synthetic": True, "warmup_task_ids": {}, "manifests": {}}
