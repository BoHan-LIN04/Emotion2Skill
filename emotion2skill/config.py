"""Load inherited YAML configuration and apply CLI overrides."""
from __future__ import annotations
import copy
from pathlib import Path

DEFAULT_CONFIG = {
    "model": {"name_or_path": "Qwen/Qwen3-8B", "key": "qwen3-8b", "device": "cuda"},
    "extractor": {"checkpoint": "checkpoints/extractor_qwen3_8b.pkl", "mode": "contrastive",
        "stories_path": "data/emotion_stories/stories.json", "goemotions_validation": "data/goemotions/validation.json",
        "goemotions_test": "data/goemotions/test.json", "pca_components": 10, "n_neutral": 100},
    "encoder": {"checkpoint": "checkpoints/encoder_qwen3_8b.pt", "tau": 0.3},
    "training": {"warmup_episodes_per_env": 250, "epochs": 50, "batch_size": 64,
        "lr": 1e-3, "weight_decay": 1e-4, "patience": 5, "seed": 42},
    "evolution": {"enabled": True, "rounds": 3, "eta": 0.4, "window": 20, "episode_per_round": 50},
    "evaluation": {"env": "webshop", "method": "Emotion2Skill", "ablation": "full", "n_runs": 3,
        "n_episodes": None, "max_steps": None, "seed": 42, "mock": False,
        "mbpp_timeout": 5.0, "baseline_artifact": None, "external_adapter": None,
        "task_order": None, "evolution_task_order": None, "evaluate_each_round": False},
    "datasets": {name: {"manifest": f"manifests/{name}.json"} for name in ("webshop", "alfworld", "math", "mbpp")},
    "logging": {"results_dir": "results", "run_id": None},
}

def deep_merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result

def load_config(path=None, _seen=None):
    if path is None:
        return copy.deepcopy(DEFAULT_CONFIG)
    import yaml
    path = Path(path).resolve()
    seen = set() if _seen is None else set(_seen)
    if path in seen:
        raise ValueError("Circular YAML config inheritance")
    seen.add(path)
    document = yaml.safe_load(path.read_text()) or {}
    if not isinstance(document, dict):
        raise ValueError("Config root must be a mapping")
    parent = document.pop("extends", None)
    base = load_config(path.parent / parent, seen) if parent else copy.deepcopy(DEFAULT_CONFIG)
    return deep_merge(base, document)

def resolve_config(path=None, overrides=None):
    config = load_config(path)
    for dotted, value in (overrides or {}).items():
        if value is None:
            continue
        keys = dotted.split(".")
        destination = config
        for key in keys[:-1]:
            destination = destination.setdefault(key, {})
        destination[keys[-1]] = value
    name = config["model"]["name_or_path"].lower()
    explicit_key = (overrides or {}).get("model.key")
    detected = "qwen3-14b" if "qwen3-14b" in name else "qwen3-8b" if "qwen3-8b" in name else None
    if detected:
        if explicit_key and explicit_key != detected:
            raise ValueError("--model_key does not match --model")
        config["model"]["key"] = detected
        if detected == "qwen3-14b":
            for section in ("extractor", "encoder"):
                current = config[section].get("checkpoint")
                if current == DEFAULT_CONFIG[section]["checkpoint"] and not (overrides or {}).get(section + ".checkpoint"):
                    config[section]["checkpoint"] = current.replace("qwen3_8b", "qwen3_14b")
    if config["model"]["key"] not in {"qwen3-8b", "qwen3-14b"}:
        raise ValueError("Unsupported backbone key")
    if config["evaluation"]["n_runs"] < 1:
        raise ValueError("n_runs must be positive")
    if not 0 <= config["encoder"]["tau"] <= 1:
        raise ValueError("tau must be in [0,1]")
    if config["evolution"]["rounds"] < 0 or config["evolution"]["episode_per_round"] < 1:
        raise ValueError("Evolution rounds must be nonnegative and batch size positive")
    if config["evaluation"].get("n_episodes") is not None and config["evaluation"]["n_episodes"] < 1:
        raise ValueError("n_episodes must be positive")
    if config["evaluation"].get("max_steps") is not None and config["evaluation"]["max_steps"] < 1:
        raise ValueError("max_steps must be positive")
    return config
