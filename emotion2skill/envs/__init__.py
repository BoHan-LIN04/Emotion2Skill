"""Benchmark adapters share reset -> (observation, goal) and BenchmarkStep."""
from .common import BenchmarkStep, load_manifest
from .webshop import WebShopEnv
from .alfworld import ALFWorldEnv
from .math import MATHEnv
from .mbpp import MBPPEnv

def build_env(name, manifest_path=None, split="test", mock=False, max_steps=None, **kwargs):
    classes = {"webshop": WebShopEnv, "alfworld": ALFWorldEnv, "math": MATHEnv, "mbpp": MBPPEnv}
    if name not in classes:
        raise ValueError(f"Unknown benchmark: {name}")
    if max_steps is not None:
        kwargs["max_steps"] = max_steps
    return classes[name](manifest_path=manifest_path, split=split, mock=mock, **kwargs)
