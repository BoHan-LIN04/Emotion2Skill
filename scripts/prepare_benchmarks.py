#!/usr/bin/env python3
"""Create split manifests from local benchmark records.

WebShop input: JSON with explicit train/dev/test lists of canonical goal indices.
MATH/MBPP input: train/dev/test records with task_id or id.
ALFWorld input: a data root containing json_2.1.1 game files.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emotion2skill.envs.common import load_manifest
from emotion2skill.envs.alfworld import OFFICIAL_TYPES

def prepare(benchmark, source, revision, output):
    source, output = Path(source).resolve(), Path(output)
    splits = {}
    if benchmark == "alfworld":
        for split, directory in {"train": "train", "dev": "valid_seen", "test": "valid_unseen"}.items():
            records = []
            for game in sorted((source / "json_2.1.1" / directory).rglob("game.tw-pddl")):
                metadata_path = game.parent / "traj_data.json"
                game_data = json.loads(game.read_text())
                if not metadata_path.exists() or not game_data.get("solvable", False):
                    continue
                metadata = json.loads(metadata_path.read_text())
                task_type = OFFICIAL_TYPES.get(metadata.get("task_type"))
                if task_type is None or "movable" in str(game) or "Sliced" in str(game):
                    continue
                records.append({"task_id": "alfworld:" + str(game.relative_to(source)),
                                "gamefile": str(game.relative_to(source)), "task_type": task_type,
                                "gamefile_sha256": hashlib.sha256(game.read_bytes()).hexdigest()})
            splits[split] = records
    else:
        raw = json.loads(source.read_text())
        for split in ("train", "dev", "test"):
            if split not in raw:
                raise ValueError("Input must explicitly provide train/dev/test; implicit random re-splitting is forbidden.")
            records = []
            for record in raw[split]:
                if benchmark == "webshop":
                    index = record if isinstance(record, int) else record["index"]
                    records.append({"task_id": f"webshop:{index}", "index": index})
                else:
                    record = dict(record)
                    source_id = record.get("task_id", record.get("id"))
                    if source_id is None:
                        raise ValueError("MATH/MBPP input records require their original task_id/id.")
                    record["task_id"] = f"{benchmark}:{source_id}"
                    if benchmark == "math" and "answer" not in record:
                        if "solution" not in record:
                            raise ValueError("MATH requires answer or original solution")
                        record["answer"] = record["solution"]
                    if benchmark == "mbpp" and "text" not in record and "prompt" in record:
                        record["text"] = record["prompt"]
                    records.append(record)
            splits[split] = records
    data = {"schema_version": 1, "benchmark": benchmark, "dataset_revision": revision,
            "source_path": str(source), "splits": splits}
    if benchmark == "webshop":
        data["environment"] = raw.get("environment", {})
    if source.is_file():
        data["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing manifest: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    try:
        load_manifest(output, benchmark)
    except Exception:
        output.unlink()
        raise
    return data

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, choices=["webshop", "alfworld", "math", "mbpp"])
    parser.add_argument("--source", required=True)
    parser.add_argument("--dataset-revision", required=True, help="Exact source commit/dataset revision")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data = prepare(args.benchmark, args.source, args.dataset_revision, args.output)
    print({split: len(records) for split, records in data["splits"].items()})
if __name__ == "__main__":
    main()
