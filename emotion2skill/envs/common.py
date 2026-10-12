"""Benchmark contracts and explicit, disjoint task manifests."""
from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class BenchmarkStep:
    observation: str
    reward: float
    done: bool
    success: bool = False
    info: dict = field(default_factory=dict)

def load_manifest(path, benchmark):
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    if data.get("schema_version") != 1 or data.get("benchmark") != benchmark:
        raise ValueError(f"Invalid {benchmark} manifest schema: {path}")
    if not data.get("dataset_revision"):
        raise ValueError("A pinned dataset_revision is required.")
    splits = data.get("splits", {})
    if set(splits) != {"train", "dev", "test"}:
        raise ValueError("Manifest must explicitly contain train, dev and test splits.")
    seen_ids, seen_sources = set(), set()
    for split, records in splits.items():
        if not isinstance(records, list):
            raise ValueError(f"{split} must be a list of task records.")
        for record in records:
            task_id = record.get("task_id")
            if not isinstance(task_id, str) or not task_id or task_id in seen_ids:
                raise ValueError(f"Missing or overlapping task_id: {task_id}")
            seen_ids.add(task_id)
            if benchmark == "webshop":
                if not isinstance(record.get("index"), int) or record["index"] < 0:
                    raise ValueError("WebShop records require a nonnegative canonical index.")
                source = (benchmark, record["index"])
            elif benchmark == "alfworld":
                if not record.get("gamefile"):
                    raise ValueError("ALFWorld records require a gamefile.")
                gamefile = Path(record["gamefile"]).expanduser()
                if not gamefile.is_absolute():
                    gamefile = Path(data.get("source_path", path.parent)) / gamefile
                source = (benchmark, str(gamefile.resolve()))
            elif benchmark == "math":
                if not isinstance(record.get("problem"), str) or "answer" not in record:
                    raise ValueError("MATH records require problem and answer.")
                source = (benchmark, record["problem"].strip())
            elif benchmark == "mbpp":
                if not isinstance(record.get("text"), str) or not record.get("test_list"):
                    raise ValueError("MBPP records require text and nonempty test_list.")
                source = (benchmark, record["text"].strip())
            else:
                raise ValueError(f"Unsupported benchmark: {benchmark}")
            if source in seen_sources:
                raise ValueError(f"Overlapping source task across manifest splits: {task_id}")
            seen_sources.add(source)
    data["manifest_path"] = str(path)
    data["manifest_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return data

class ManifestEnv:
    def _configure(self, manifest_path, split, mock, max_steps):
        if split not in {"train", "dev", "test"}:
            raise ValueError("split must be train, dev or test")
        self.split, self.mock, self.max_steps = split, bool(mock), max_steps
        if manifest_path is None:
            if not mock:
                raise ValueError("Formal benchmark runs require an explicit task manifest.")
            self.manifest = {"schema_version": 1, "benchmark": self.benchmark,
                "dataset_revision": "synthetic-v1", "manifest_sha256": None,
                "splits": {s: [{"task_id": f"synthetic:{self.benchmark}:{s}:{i}", "index": i,
                    "task_type": "heat" if self.benchmark == "alfworld" else self.benchmark}
                    for i in range(500)] for s in ("train", "dev", "test")}}
        else:
            self.manifest = load_manifest(manifest_path, self.benchmark)
        self.tasks = self.manifest["splits"][split]
        if not self.tasks:
            raise ValueError(f"Manifest has no {split} tasks for {self.benchmark}.")
        self.backend = "synthetic" if mock else self.benchmark
        self._cursor, self._step_count = -1, 0
        self.task_id, self.task_type = "", self.benchmark
        self._current_obs, self._goal = "", ""
        self._task = {}

    def _choose_task(self, task_idx=None):
        if task_idx is None:
            self._cursor = (self._cursor + 1) % len(self.tasks)
            record = self.tasks[self._cursor]
        elif isinstance(task_idx, int):
            if not 0 <= task_idx < len(self.tasks):
                raise IndexError(f"Manifest task index out of range: {task_idx}")
            record = self.tasks[task_idx]
        else:
            record = next((r for r in self.tasks if r["task_id"] == str(task_idx)), None)
            if record is None:
                raise KeyError(f"Task ID is not in {self.split}: {task_idx}")
        self._task = record
        self.task_id = record["task_id"]
        self.task_type = record.get("task_type", self.benchmark)
        self._step_count = 0
        return record

    def task_instruction(self):
        return self._goal

    def format_context(self, task_goal, history, current_obs):
        previous = "\n".join(f"Action: {a}\nObservation: {o}" for a, o in history[-6:])
        return f"Task: {task_goal}\n\nPrevious steps:\n{previous}\n\nCurrent observation:\n{current_obs}"

    def metadata(self):
        return {"benchmark": self.benchmark, "split": self.split, "backend": self.backend,
            "synthetic": self.mock, "dataset_revision": self.manifest["dataset_revision"],
            "manifest_sha256": self.manifest.get("manifest_sha256"),
            "environment_options": self.manifest.get("environment", {})}
