#!/usr/bin/env python3
"""Prepare official GoEmotions train/validation/test data for the 27-way probe.

Use --source_dir for a local official TSV snapshot, or --download to obtain and
record a fixed google-research commit. Neutral labels are removed; neutral-only
examples are excluded and counted. Original split IDs are kept separate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization", "relief",
    "remorse", "sadness", "surprise",
]
SOURCE_REPO = "google-research/google-research"


def convert_split(tsv_path, emotion_names, split, revision):
    records, neutral_only, removed_labels = [], 0, 0
    mapping = {i: EMOTION_LABELS.index(name) for i, name in enumerate(emotion_names)
               if name in EMOTION_LABELS}
    if set(emotion_names) != set(EMOTION_LABELS + ["neutral"]):
        raise ValueError("Official emotions.txt must contain 27 concepts plus neutral")
    ids = set()
    with Path(tsv_path).open(encoding="utf-8", newline="") as stream:
        for line, row in enumerate(csv.reader(stream, delimiter="\t"), 1):
            if len(row) != 3:
                raise ValueError(f"Expected text, labels, ID at {tsv_path}:{line}")
            text, label_string, example_id = row
            source_labels = [int(value) for value in label_string.split(",")]
            if any(not 0 <= label < len(emotion_names) for label in source_labels):
                raise ValueError(f"Invalid emotion index at {tsv_path}:{line}")
            labels = sorted({mapping[label] for label in source_labels if label in mapping})
            removed_labels += sum(label not in mapping for label in source_labels)
            if not labels:
                neutral_only += 1
                continue
            if example_id in ids:
                raise ValueError(f"Duplicate example ID {example_id} in {split}")
            ids.add(example_id)
            records.append({"id": example_id, "text": text, "labels": labels})
    return {"schema_version": 1, "dataset": "GoEmotions simplified official split",
            "split": split, "taxonomy": EMOTION_LABELS, "records": records,
            "source": {"repository": SOURCE_REPO, "revision": revision,
                       "file": Path(tsv_path).name,
                       "sha256": hashlib.sha256(Path(tsv_path).read_bytes()).hexdigest()},
            "filter": {"remove_neutral_label": True, "exclude_neutral_only": True,
                       "neutral_only_rows_removed": neutral_only,
                       "neutral_labels_removed": removed_labels}}


def download_snapshot(source_dir, revision):
    # Resolve moving branch references to a concrete commit before downloading.
    if len(revision) != 40 or any(ch not in "0123456789abcdef" for ch in revision.lower()):
        request = Request(f"https://api.github.com/repos/{SOURCE_REPO}/commits/{revision}",
                          headers={"User-Agent": "Emotion2Skill-data-preparation"})
        with urlopen(request, timeout=60) as response:
            revision = json.load(response)["sha"]
    source_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("train.tsv", "dev.tsv", "test.tsv", "emotions.txt"):
        url = f"https://raw.githubusercontent.com/{SOURCE_REPO}/{revision}/goemotions/data/{filename}"
        with urlopen(url, timeout=60) as response:
            content = response.read()
        if not content:
            raise RuntimeError(f"Empty official source file {filename}")
        (source_dir / filename).write_bytes(content)
    (source_dir / "source_revision.txt").write_text(revision + "\n", encoding="utf-8")
    return revision


def prepare(source_dir, out_dir, revision):
    source_dir, out_dir = Path(source_dir), Path(out_dir)
    emotion_names = (source_dir / "emotions.txt").read_text(encoding="utf-8").splitlines()
    payloads = {target: convert_split(source_dir / filename, emotion_names, target, revision)
                for filename, target in (("train.tsv", "train"), ("dev.tsv", "validation"),
                                         ("test.tsv", "test"))}
    split_ids = {split: {r["id"] for r in data["records"]} for split, data in payloads.items()}
    for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if split_ids[first] & split_ids[second]:
            raise ValueError(f"Official example IDs overlap between {first} and {second}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, payload in payloads.items():
        (out_dir / (split + ".json")).write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                                  encoding="utf-8")
    manifest = {"schema_version": 1, "repository": SOURCE_REPO, "revision": revision,
                "taxonomy": EMOTION_LABELS, "splits": {
                    split: {"records": len(data["records"]), "file": split + ".json",
                            "sha256": hashlib.sha256((out_dir / (split + ".json")).read_bytes()).hexdigest(),
                            "filter": data["filter"]}
                    for split, data in payloads.items()}}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source_dir", default="data/goemotions/source")
    p.add_argument("--out_dir", default="data/goemotions")
    p.add_argument("--download", action="store_true")
    p.add_argument("--revision", default="master")
    args = p.parse_args()
    source = Path(args.source_dir)
    if args.download:
        revision = download_snapshot(source, args.revision)
    elif (source / "source_revision.txt").exists():
        revision = (source / "source_revision.txt").read_text(encoding="utf-8").strip()
    else:
        revision = args.revision
    manifest = prepare(source, args.out_dir, revision)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
