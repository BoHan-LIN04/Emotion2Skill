#!/usr/bin/env python3
"""Evaluate declared methods on fixed manifests, with train-only evolution.

Each run records its effective config, task order, provenance and full episode
logs. Synthetic smoke-test results are never aggregated into paper metrics.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import random
import statistics
from datetime import datetime, timezone
from pathlib import Path
from emotion2skill.config import resolve_config
from emotion2skill.envs import build_env, load_manifest
from emotion2skill.envs.alfworld import ALFWorldEnv
from emotion2skill.envs.webshop import WebShopEnv

METHODS = ["Emotion2Skill", "ZeroShot", "ReAct", "Reflexion", "ExpeL", "TextSimilarity", "MASA", "EvoSkill", "Trace2Skill"]
ABLATIONS = ["full", "no_extraction", "no_encoder", "selection_only", "no_diagnostics", "generic_pca", "shuffled"]
DEFAULT_N_EPISODES = {"webshop": 500, "alfworld": 134, "math": 500, "mbpp": 374}

CLI_KEYS = {"model": "model.name_or_path", "model_key": "model.key", "model_revision": "model.revision", "device": "model.device",
    "env": "evaluation.env", "method": "evaluation.method", "ablation": "evaluation.ablation",
    "split": "evaluation.split", "n_episodes": "evaluation.n_episodes", "n_runs": "evaluation.n_runs",
    "extractor": "extractor.checkpoint", "encoder": "encoder.checkpoint", "tau": "encoder.tau",
    "eta": "evolution.eta", "evolution": "evolution.enabled", "evolution_rounds": "evolution.rounds",
    "episode_per_round": "evolution.episode_per_round", "seed": "evaluation.seed", "mock": "evaluation.mock",
    "max_steps": "evaluation.max_steps", "task_order": "evaluation.task_order",
    "evolution_task_order": "evaluation.evolution_task_order", "baseline_artifact": "evaluation.baseline_artifact",
    "external_adapter": "evaluation.external_adapter", "evaluate_each_round": "evaluation.evaluate_each_round",
    "out_dir": "logging.results_dir", "run_id": "logging.run_id"}

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    for flag in ("model", "model_key", "model_revision", "device", "extractor", "encoder", "task_order",
                 "evolution_task_order", "baseline_artifact", "external_adapter", "out_dir", "run_id"):
        parser.add_argument("--" + flag, default=None)
    parser.add_argument("--env", choices=list(DEFAULT_N_EPISODES))
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--ablation", choices=ABLATIONS)
    parser.add_argument("--split", choices=["dev", "test"])
    for flag in ("n_episodes", "n_runs", "evolution_rounds", "episode_per_round", "seed", "max_steps"):
        parser.add_argument("--" + flag, type=int, default=None)
    for flag in ("tau", "eta"):
        parser.add_argument("--" + flag, type=float, default=None)
    parser.add_argument("--evolution", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--mock", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--evaluate_each_round", action=argparse.BooleanOptionalAction, default=None,
                        help="Frozen evaluation after each mixed evolution round; never feeds the evolution tracker")
    parser.add_argument("--manifest", help="Overrides the selected benchmark manifest")
    parser.add_argument("--tau_sweep", action="store_true", help="Validation-only sweep; does not select a threshold from test data")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    overrides = {key: getattr(args, flag) for flag, key in CLI_KEYS.items()}
    config = resolve_config(args.config, overrides)
    config["evaluation"].setdefault("split", "test")
    if args.manifest:
        config["datasets"][config["evaluation"]["env"]]["manifest"] = args.manifest
    if args.tau_sweep:
        if args.split == "test":
            parser.error("--tau_sweep is validation-only; use individual --tau settings for test sensitivity reporting")
        config["evaluation"]["split"] = "dev"
    if config["evaluation"]["method"] not in METHODS or config["evaluation"]["ablation"] not in ABLATIONS:
        parser.error("Unknown method or ablation in config")
    if config["evaluation"]["method"] != "Emotion2Skill" and config["evaluation"]["ablation"] != "full":
        parser.error("Component ablations apply to Emotion2Skill only")
    return args, config

def task_order(env, n_episodes, seed, order_file=None, reserved=0, excluded_task_ids=None):
    available = [record["task_id"] for record in env.tasks]
    excluded = set(excluded_task_ids or [])
    candidates = [task_id for task_id in available[reserved:] if task_id not in excluded]
    if order_file:
        document = json.loads(Path(order_file).read_text())
        order = document["task_ids"] if isinstance(document, dict) else document
        if len(order) != n_episodes or len(set(order)) != len(order):
            raise ValueError("Task-order file must contain exactly n_episodes unique task IDs")
        if any(item not in candidates for item in order):
            raise ValueError("Task-order file contains another split or a reserved warm-up task")
        return order
    if n_episodes > len(candidates) or n_episodes < 1:
        raise ValueError(f"Requested {n_episodes} tasks but {len(candidates)} eligible {env.split} tasks exist")
    # Fix the subset across runs; vary ordering only, matching the stated protocol.
    order = candidates[:n_episodes]
    random.Random(seed).shuffle(order)
    return order

def _protocol_manifest(config, benchmark):
    path = config["datasets"].get(benchmark, {}).get("manifest")
    if not path or not Path(path).is_file():
        raise FileNotFoundError(f"Training provenance validation requires the current {benchmark} manifest")
    return load_manifest(path, benchmark)

def _validate_train_task_ids(task_ids, manifest, *, source):
    """Bind source IDs to an explicit benchmark namespace and original split."""
    benchmark = manifest["benchmark"]
    if (not isinstance(task_ids, list) or not task_ids or
            any(not isinstance(item, str) or not item for item in task_ids) or
            len(set(task_ids)) != len(task_ids)):
        raise ValueError(f"{source}: complete unique episode task IDs are required")
    for task_id in task_ids:
        declared_prefix = next((name for name in DEFAULT_N_EPISODES if task_id.startswith(name + ":")), None)
        if declared_prefix is not None and declared_prefix != benchmark:
            raise ValueError(f"{source}: task namespace {declared_prefix} does not match {benchmark}")
    train = {record["task_id"] for record in manifest["splits"]["train"]}
    held_out = {record["task_id"] for split in ("dev", "test") for record in manifest["splits"][split]}
    ids = set(task_ids)
    if ids & held_out:
        raise ValueError(f"{source}: training task IDs overlap current dev/test tasks")
    if not ids <= train:
        raise ValueError(f"{source}: source task IDs are not in the declared {benchmark} training manifest")
    return ids

def validate_protocol_provenance(config):
    """Close the checkpoint/data lineage before any backbone is constructed."""
    evaluation = config["evaluation"]
    if evaluation["mock"]:
        return {"synthetic": True, "warmup_task_ids": {}, "manifests": {}}
    method, ablation = evaluation["method"], evaluation["ablation"]
    provenance = {"synthetic": False, "warmup_task_ids": {}, "manifests": {}}
    if method == "Emotion2Skill" and ablation != "no_encoder":
        import torch
        path = config["encoder"].get("checkpoint")
        if not path or not Path(path).is_file():
            raise FileNotFoundError("Formal learned emotion inference requires an encoder checkpoint with training provenance")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        source = payload.get("metadata", {}).get("source", {})
        entries = source.get("warmup")
        if source.get("environment_split") != "train" or not isinstance(entries, list):
            raise ValueError("Encoder checkpoint is missing complete train-only warm-up provenance")
        by_benchmark = {}
        for entry in entries:
            benchmark = entry.get("benchmark", entry.get("namespace"))
            if benchmark not in {"webshop", "alfworld"} or benchmark in by_benchmark:
                raise ValueError("Shared encoder requires one explicit warm-up namespace per WebShop/ALFWorld domain")
            by_benchmark[benchmark] = entry
        if set(by_benchmark) != {"webshop", "alfworld"}:
            raise ValueError("Shared encoder warm-up provenance must include both WebShop and ALFWorld")
        for benchmark, entry in by_benchmark.items():
            manifest = _protocol_manifest(config, benchmark)
            if entry.get("manifest_sha256") != manifest["manifest_sha256"]:
                raise ValueError(f"Encoder warm-up {benchmark} manifest hash differs from the current task manifest")
            ids = entry.get("episode_task_ids")
            _validate_train_task_ids(ids, manifest, source=f"encoder warm-up {benchmark}")
            if len(ids) != entry.get("episodes") or len(ids) != config["training"]["warmup_episodes_per_env"]:
                raise ValueError(f"Encoder warm-up {benchmark} episode count differs from the declared protocol")
            provenance["warmup_task_ids"][benchmark] = list(ids)
            provenance["manifests"][benchmark] = manifest["manifest_sha256"]
    elif method in {"MASA", "EvoSkill", "Trace2Skill"}:
        path = evaluation.get("baseline_artifact")
        if not path or not Path(path).is_file():
            raise FileNotFoundError(f"{method} requires an external baseline artifact manifest")
        artifact = json.loads(Path(path).read_text())
        benchmark = artifact.get("benchmark")
        if benchmark != evaluation["env"] or artifact.get("source_split") != "train":
            raise ValueError("External baseline must declare the target benchmark namespace and train source split")
        if artifact.get("synthetic"):
            raise ValueError("Synthetic baseline artifacts cannot enter a formal evaluation")
        manifest = _protocol_manifest(config, benchmark)
        _validate_train_task_ids(artifact.get("source_task_ids"), manifest, source=method)
        provenance["baseline_source_task_ids"] = artifact["source_task_ids"]
        provenance["manifests"][benchmark] = manifest["manifest_sha256"]
    return provenance

def compute_metrics(env_name, logs):
    if not logs:
        raise ValueError("No episodes to evaluate")
    if any(log.backend in {"synthetic", "mock"} for log in logs):
        raise ValueError("Synthetic episodes cannot enter paper metrics")
    if env_name == "webshop":
        return WebShopEnv.compute_metrics([log.outcome for log in logs])
    if env_name == "alfworld":
        return ALFWorldEnv.compute_metrics([{"task_type": log.task_type, "success": log.success} for log in logs])
    key = "accuracy" if env_name == "math" else "pass_at_1"
    return {key: 100 * sum(log.success for log in logs) / len(logs)}

def summarize_runs(runs):
    if not runs or any(run.get("synthetic") for run in runs):
        raise ValueError("Paper aggregation requires non-synthetic runs")
    keys = sorted(set().union(*(run["metrics"] for run in runs)))
    summary = {}
    for key in keys:
        values = [run["metrics"].get(key) for run in runs]
        if any(value is None for value in values):
            summary[key] = {"mean": None, "sample_std": None, "n_runs": len(runs)}
        else:
            summary[key] = {"mean": statistics.mean(values),
                "sample_std": statistics.stdev(values) if len(values) > 1 else None, "n_runs": len(values)}
    return summary

def _hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if path and Path(path).is_file() else None

class SyntheticLLM:
    """Deterministic local plumbing probe; never an actual model prediction."""
    def __init__(self, benchmark): self.benchmark = benchmark
    def chat_prefix(self, system, user): return system + "\n" + user
    def __call__(self, system, user):
        if "updated numbered insight" in user.lower() or "learn general action insights" in system.lower():
            return "1. Inspect the current observation and use a valid environment action. [Synthetic plumbing test]"
        if "reflect" in system.lower():
            return "Synthetic reflection: inspect the observation before choosing the next action."
        if "expert agent designer" in system.lower() or "improve agent skill sops" in system.lower():
            return "Inspect the current observation. Choose a valid action that advances the stated task. Verify completion and stop. [Synthetic SOP]"
        if "diagnostic" in system.lower() or "analyze recorded agent trajectories" in system.lower():
            return "Synthetic diagnostic summary: no scientific interpretation is inferred."
        if "skill name" in user.lower() or "choose exactly one skill" in system.lower():
            import re
            match = re.search(r"\d+\. (\w+):", user)
            return match.group(1) if match else "ProductSearch"
        if self.benchmark == "math": action = r"final[\boxed{2}]"
        elif self.benchmark == "mbpp": action = "```python\ndef add(a, b):\n    return a + b\n```"
        elif self.benchmark == "alfworld": action = "look"
        else: action = "click[buy now]"
        if "Output 'Thought:" in user:
            return "Thought: Synthetic local interface probe.\nAction: " + action
        return action

def _agent(config, run_dir, seed):
    from emotion2skill.agent import AgentConfig, Emotion2SkillAgent, validate_inference_assets
    from emotion2skill.baselines import build_baseline
    evaluation = config["evaluation"]
    ablation = evaluation["ablation"]
    mode = "class_mean" if ablation == "no_extraction" else "generic_pca" if ablation == "generic_pca" else config["extractor"].get("mode", "contrastive")
    fields = {"model_name_or_path": config["model"]["name_or_path"], "model_key": config["model"]["key"],
        "model_revision": config["model"].get("revision"),
        "device": config["model"]["device"], "extractor_path": config["extractor"].get("checkpoint"),
        "encoder_path": config["encoder"].get("checkpoint"), "tau": config["encoder"]["tau"],
        "eta": config["evolution"]["eta"], "evolution_rounds": config["evolution"]["rounds"],
        "evolution_window": config["evolution"].get("window", 100),
        "max_steps": evaluation.get("max_steps") or (50 if evaluation["env"] == "alfworld" else 15),
        "max_context_tokens": 8192 if config["model"]["key"] == "qwen3-14b" else 4096,
        "log_dir": str(run_dir), "run_id": run_dir.name, "seed": seed, "mock": evaluation["mock"],
        "benchmark": evaluation["env"], "extractor_mode": mode,
        "encoder_mode": "top3" if ablation == "no_encoder" else "learned",
        "shuffle_context": ablation == "shuffled", "ablation_mode": "full",
        "evolution_mode": "outcome_only" if ablation == "no_diagnostics" else "emotion",
        "evolution_diagnostics": ablation != "no_diagnostics"}
    agent_config = AgentConfig(**fields)
    llm = SyntheticLLM(evaluation["env"]) if evaluation["mock"] else None
    if evaluation["method"] == "Emotion2Skill":
        if not evaluation["mock"]:
            validate_inference_assets(agent_config, require_encoder=ablation != "no_encoder")
        agent = Emotion2SkillAgent(agent_config, llm=llm)
        agent.load(load_emotion=not evaluation["mock"], require_encoder=ablation != "no_encoder")
    else:
        agent = build_baseline(evaluation["method"], agent_config, llm=llm,
            artifact_path=evaluation.get("baseline_artifact"), external_adapter=evaluation.get("external_adapter"))
    return agent

def _environment(config, name, split):
    settings = config["datasets"][name]
    evaluation = config["evaluation"]
    manifest = settings.get("manifest")
    if evaluation["mock"] and manifest and not Path(manifest).exists():
        manifest = None
    extra = {}
    if name == "alfworld" and settings.get("data_path"):
        extra["data_path"] = settings["data_path"]
    if name == "mbpp":
        extra["timeout"] = evaluation.get("mbpp_timeout", 5.0)
    # An override applies to the target benchmark; the other evolution domain
    # retains its own action budget (15 WebShop, 50 ALFWorld).
    max_steps = evaluation.get("max_steps") if name == evaluation["env"] else None
    return build_env(name, manifest, split, mock=evaluation["mock"], max_steps=max_steps, **extra)

def _evaluate_frozen_round(agent, env, library, order, run_dir, seed, round_index,
                           use_emotion, mock, verbose):
    if hasattr(agent, "config"):
        agent.config.max_steps = env.max_steps
    if isinstance(getattr(agent, "llm", None), SyntheticLLM):
        agent.llm.benchmark = env.benchmark
    logs = agent.run_episodes(env, library, n_episodes=len(order),
        use_emotion=use_emotion, collect_emotion=use_emotion, evolution=False,
        task_order=order, seed=seed, phase="round_evaluation",
        run_id=f"{run_dir.name}_round_{round_index}_evaluation", verbose=verbose)
    if len(logs) != len(order):
        raise RuntimeError("Runner returned an incomplete round evaluation")
    directory = run_dir / "round_evaluation" / f"round_{round_index:02d}"
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / "episodes.jsonl").open("w") as handle:
        for log in logs:
            log.round_id = round_index
            handle.write(json.dumps(log.to_dict(), ensure_ascii=False) + "\n")
    snapshot = [{"name": skill.name, "sop": skill.sop, "description": skill.description}
                for skill in library.skills]
    (directory / "frozen_library.json").write_text(json.dumps(snapshot, indent=2))
    record = {"round": round_index, "run_id": run_dir.name, "benchmark": env.benchmark,
        "phase": "round_evaluation", "split": env.split, "seed": seed,
        "synthetic": mock, "environment": env.metadata(), "task_ids": order,
        "feeds_evolution": False,
        "metrics": None if mock else compute_metrics(env.benchmark, logs)}
    (directory / "metrics.json").write_text(json.dumps(record, indent=2))
    return record

def run_evaluation(config, output, verbose=False):
    from emotion2skill.skills import get_library
    evaluation = config["evaluation"]
    provenance = validate_protocol_provenance(config)
    env_name, split = evaluation["env"], evaluation.get("split", "test")
    env = _environment(config, env_name, split)
    n_episodes = evaluation.get("n_episodes") or (min(100, len(env.tasks)) if split == "dev" else DEFAULT_N_EPISODES[env_name])
    run_count = evaluation["n_runs"]
    runs = []
    for index in range(run_count):
        seed = evaluation["seed"] + index
        run_dir = output / f"run_{index:02d}_seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=False)
        library = get_library(env_name)
        agent = _agent(config, run_dir, seed)
        order = task_order(env, n_episodes, seed, evaluation.get("task_order"))
        round_results = []
        use_emotion = evaluation["method"] == "Emotion2Skill" and not evaluation["mock"]
        evolution_config = config["evolution"]
        evolve = evaluation["method"] == "Emotion2Skill" and evolution_config["enabled"] and evolution_config["rounds"] > 0 and evaluation["ablation"] != "selection_only" and env_name in {"webshop", "alfworld"}
        evolution_ids = {}
        if evaluation.get("evaluate_each_round", False):
            round_results.append(_evaluate_frozen_round(agent, env, library, order,
                run_dir, seed, 0, use_emotion, evaluation["mock"], verbose))
        if evolve:
            libraries = {name: library if name == env_name else get_library(name)
                         for name in ("webshop", "alfworld")}
            training_envs = {name: _environment(config, name, "train") for name in libraries}
            count = evolution_config["rounds"] * evolution_config["episode_per_round"]
            # Learned inference is bound to the exact checkpoint source IDs.
            # Synthetic and top-3/no-encoder plumbing reserves the same nominal
            # warm-up subset conservatively, without claiming it was trained.
            reserved = config["training"]["warmup_episodes_per_env"] if not provenance["warmup_task_ids"] else 0
            for name, training in training_envs.items():
                order_file = evaluation.get("evolution_task_order")
                if isinstance(order_file, dict):
                    order_file = order_file.get(name)
                elif order_file:
                    raise ValueError("Mixed evolution requires evaluation.evolution_task_order mapping per benchmark")
                evolution_ids[name] = task_order(training, count, seed, order_file, reserved=reserved,
                    excluded_task_ids=provenance["warmup_task_ids"].get(name, []))
            batch = evolution_config["episode_per_round"]
            for round_index in range(evolution_config["rounds"]):
                for name, training in training_envs.items():
                    # Keep the domain-specific step budget while sharing model,
                    # extractor and trained encoder across both skill libraries.
                    if hasattr(agent, "config"):
                        agent.config.max_steps = training.max_steps
                    if isinstance(getattr(agent, "llm", None), SyntheticLLM):
                        agent.llm.benchmark = name
                    agent.run_episodes(training, libraries[name], n_episodes=batch,
                        use_emotion=use_emotion, collect_emotion=use_emotion, evolution=True,
                        episode_per_round=batch, evolution_rounds=1,
                        task_order=evolution_ids[name][round_index * batch:(round_index + 1) * batch],
                        seed=seed, run_id=f"{run_dir.name}_evolution_round_{round_index + 1}_{name}",
                        phase="evolution", verbose=verbose)
                if evaluation.get("evaluate_each_round", False):
                    round_results.append(_evaluate_frozen_round(agent, env, library, order,
                        run_dir, seed, round_index + 1, use_emotion, evaluation["mock"], verbose))
            for name, evolved in libraries.items():
                (run_dir / f"frozen_library_{name}.json").write_text(json.dumps([
                    {"name": skill.name, "sop": skill.sop, "description": skill.description}
                    for skill in evolved.skills], indent=2))
            for training in training_envs.values():
                if callable(getattr(training, "close", None)):
                    training.close()
        if hasattr(agent, "config"):
            agent.config.max_steps = env.max_steps
        if isinstance(getattr(agent, "llm", None), SyntheticLLM):
            agent.llm.benchmark = env_name
        frozen_library = [{"name": skill.name, "sop": skill.sop, "description": skill.description} for skill in library.skills]
        (run_dir / "frozen_library.json").write_text(json.dumps(frozen_library, indent=2))
        logs = agent.run_episodes(env, library, n_episodes=n_episodes, use_emotion=use_emotion,
            collect_emotion=use_emotion, evolution=False, task_order=order, seed=seed,
            run_id=f"{run_dir.name}_evaluation", phase="evaluation", verbose=verbose)
        if len(logs) != n_episodes:
            raise RuntimeError("Runner returned an incomplete evaluation")
        metadata = {"run_id": run_dir.name, "seed": seed, "benchmark": env_name,
            "phase": "evaluation", "method": evaluation["method"],
            "ablation": evaluation["ablation"], "synthetic": evaluation["mock"],
            "paper_eligible": not evaluation["mock"] and split == "test", "environment": env.metadata(),
            "task_ids": order, "evolution_task_ids": evolution_ids, "effective_config": config,
            "data_provenance": provenance,
            "checkpoint_hashes": {"extractor": _hash_file(config["extractor"].get("checkpoint")),
                "encoder": _hash_file(config["encoder"].get("checkpoint")),
                "baseline_artifact": _hash_file(evaluation.get("baseline_artifact"))}}
        (run_dir / "evaluation_metadata.json").write_text(json.dumps(metadata, indent=2))
        # Complete per-task grading evidence even for external runners.
        with (run_dir / "episodes.jsonl").open("w") as handle:
            for log in logs:
                handle.write(json.dumps(log.to_dict(), ensure_ascii=False) + "\n")
        metrics = None if evaluation["mock"] else compute_metrics(env_name, logs)
        run = {"run_id": run_dir.name, "seed": seed, "synthetic": evaluation["mock"],
               "metrics": metrics, "rounds": round_results}
        (run_dir / "metrics.json").write_text(json.dumps(run, indent=2))
        runs.append(run)
    if callable(getattr(env, "close", None)):
        env.close()
    result = {"schema_version": 1, "benchmark": env_name, "phase": "evaluation",
              "synthetic": evaluation["mock"], "split": split,
              "paper_eligible": not evaluation["mock"] and split == "test", "runs": runs,
              "summary": None if evaluation["mock"] else summarize_runs(runs)}
    if evaluation.get("evaluate_each_round"):
        if evaluation["mock"]:
            result["round_summaries"] = None
        elif all(run["rounds"] for run in runs):
            round_numbers = [record["round"] for record in runs[0]["rounds"]]
            result["round_summaries"] = {
                str(number): summarize_runs([next(record for record in run["rounds"]
                    if record["round"] == number) for run in runs]) for number in round_numbers}
        else:
            result["round_summaries"] = {}
    (output / "results.json").write_text(json.dumps(result, indent=2))
    return result

def main(argv=None):
    args, config = parse_args(argv)
    session = config["logging"].get("run_id") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path(config["logging"]["results_dir"]) / session
    output.mkdir(parents=True, exist_ok=False)
    if args.tau_sweep:
        sweep = {}
        for tau in (0.1, 0.3, 0.6, 0.9):
            condition = copy.deepcopy(config)
            condition["encoder"]["tau"] = tau
            directory = output / f"tau_{tau}"
            directory.mkdir()
            sweep[str(tau)] = run_evaluation(condition, directory, args.verbose)
        (output / "validation_tau_sweep.json").write_text(json.dumps(sweep, indent=2))
        print(f"Validation sweep saved: {output}; select tau using validation metrics only.")
    else:
        result = run_evaluation(config, output, args.verbose)
        print("SYNTHETIC smoke test; paper aggregation disabled." if result["synthetic"] else json.dumps(result["summary"], indent=2))
        print(f"Full results: {output}")
if __name__ == "__main__":
    main()
