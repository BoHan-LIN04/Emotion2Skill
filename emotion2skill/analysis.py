"""Episode metrics, paired comparisons, and activation plots."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_episodes(paths, *, phase="evaluation", split="test"):
    episodes = []
    for path in paths:
        with Path(path).open(encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("synthetic") or record.get("backend") in {"mock", "synthetic"}:
                    raise ValueError(f"Synthetic trace cannot enter analysis: {path}:{line_no}")
                required = {"task_id", "run_id", "backend", "phase", "split", "success", "steps"}
                if required - record.keys():
                    raise ValueError(f"Missing trace metadata: {required - record.keys()}")
                if record["phase"] == phase:
                    if record["split"] != split:
                        raise ValueError(f"Expected {split} evidence, found {record['split']}; select the split explicitly")
                    episodes.append(record)
    if not episodes:
        raise ValueError(f"No measured {phase} episodes found")
    return episodes


def aggregate_runs(episodes, *, split="test"):
    """Summarize runs using sample standard deviation (ddof=1)."""
    by_run = defaultdict(list)
    seen = set()
    for ep in episodes:
        if ep.get("split") != split:
            raise ValueError("Run aggregation cannot mix dataset partitions")
        key = (ep.get("benchmark", "unknown"), ep["run_id"], str(ep["task_id"]))
        if key in seen:
            raise ValueError(f"Duplicate task in run: {key}")
        seen.add(key)
        by_run[key[:2]].append(ep)
    metrics = defaultdict(list)
    counts = {}
    for (benchmark, run_id), rows in sorted(by_run.items()):
        values = {"success_rate": 100 * np.mean([bool(r["success"]) for r in rows])}
        if benchmark == "webshop":
            values["score"] = 100 * np.mean([r["outcome"] for r in rows])
        if benchmark == "alfworld":
            for task_type in sorted({r["task_type"] for r in rows}):
                values[task_type] = 100 * np.mean([
                    bool(r["success"]) for r in rows if r["task_type"] == task_type
                ])
            values["avg"] = values["success_rate"]
        metrics[benchmark].append({"run_id": run_id, **values})
        counts[f"{benchmark}/{run_id}"] = len(rows)
    summary = {}
    for benchmark, rows in metrics.items():
        if len({counts[f"{benchmark}/{r['run_id']}"] for r in rows}) != 1:
            raise ValueError("Runs contain different task counts")
        task_sets = [{str(ep["task_id"]) for ep in by_run[(benchmark, r["run_id"])]} for r in rows]
        if any(ids != task_sets[0] for ids in task_sets[1:]):
            raise ValueError("Independent runs must use identical task sets")
        fields = set.intersection(*(set(r) - {"run_id"} for r in rows))
        summary[benchmark] = {
            "n_runs": len(rows), "runs": rows,
            "metrics": {
                field: {"mean": float(np.mean([r[field] for r in rows])),
                        "std": float(np.std([r[field] for r in rows], ddof=1))
                        if len(rows) > 1 else None}
                for field in sorted(fields)
            },
        }
    return summary


def paired_comparison(baseline, candidate, *, seed=42, resamples=10000):
    """Task-paired bootstrap and sign-flip test after averaging repeated runs."""
    if resamples < 100:
        raise ValueError("Use at least 100 resamples")
    def keyed(rows):
        result = {}
        for row in rows:
            key = (row.get("benchmark", "unknown"), row["run_id"], str(row["task_id"]))
            if key in result:
                raise ValueError(f"Duplicate paired observation: {key}")
            result[key] = float(bool(row["success"]))
        return result
    a, b = keyed(baseline), keyed(candidate)
    if not a or a.keys() != b.keys():
        raise ValueError("Paired comparison requires identical benchmark/run/task IDs")
    task_differences = defaultdict(list)
    for key in sorted(a):
        task_differences[(key[0], key[2])].append(b[key] - a[key])
    if len({key[0] for key in task_differences}) != 1:
        raise ValueError("Compare one benchmark at a time")
    diffs = np.array([np.mean(values) for values in task_differences.values()])
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(resamples)
    permuted = np.empty(resamples)
    for i in range(resamples):
        bootstrap[i] = np.mean(rng.choice(diffs, size=len(diffs), replace=True))
        permuted[i] = np.mean(diffs * rng.choice([-1, 1], size=len(diffs)))
    observed = float(diffs.mean())
    return {
        "n_tasks": len(diffs), "n_paired_episodes": len(a),
        "difference_pp": observed * 100,
        "bootstrap_95ci_pp": (np.quantile(bootstrap, [0.025, 0.975]) * 100).tolist(),
        "paired_sign_flip_p": float((1 + np.sum(np.abs(permuted) >= abs(observed))) / (resamples + 1)),
        "seed": seed, "resamples": resamples,
    }


def coactivation(episodes):
    from .extraction import K
    groups = defaultdict(list)
    for ep in episodes:
        if not ep["success"]:
            continue
        for step in ep["steps"]:
            if "emotion_state" not in step:
                raise ValueError("Trace is missing raw emotion_state")
            e = np.asarray(step["emotion_state"], dtype=float)
            if e.shape != (K,) or not np.isfinite(e).all():
                raise ValueError("Invalid raw emotion vector")
            groups[step.get("skill", step.get("skill_selected"))].append(e)
    if not groups:
        raise ValueError("No successful emotion-bearing events")
    skills = sorted(groups)
    return skills, np.vstack([np.mean(groups[skill], axis=0) for skill in skills])


def transition_points(steps):
    e = np.asarray([s["emotion_state"] for s in steps], dtype=float)
    if len(e) < 2:
        return [], []
    norms = np.linalg.norm(e, axis=1)
    if np.any(norms <= 0):
        raise ValueError("Zero vectors cannot define emotion transitions")
    e /= norms[:, None]
    deltas = 1 - np.clip(np.sum(e[1:] * e[:-1], axis=1), -1, 1)
    points = (np.flatnonzero(deltas > deltas.mean() + deltas.std(ddof=0)) + 1).tolist()
    return deltas.tolist(), points


def annotation_summary(judgments, human_annotations):
    """Compute coherence and Cohen's kappa from event annotations."""
    from sklearn.metrics import cohen_kappa_score
    def keyed(rows):
        values = {}
        for r in rows:
            if r["event_id"] in values or not isinstance(r["coherent"], bool):
                raise ValueError("Annotations need unique event IDs and boolean labels")
            values[r["event_id"]] = r["coherent"]
        return values
    machine, human = keyed(judgments), keyed(human_annotations)
    if not machine or not human or not human.keys() <= machine.keys():
        raise ValueError("Human subset must match supplied model judgments")
    ids = sorted(human)
    kappa = float(cohen_kappa_score([machine[i] for i in ids], [human[i] for i in ids]))
    return {"n_judged": len(machine), "n_human": len(human),
            "coherent_percent": 100 * sum(machine.values()) / len(machine),
            "cohen_kappa": kappa if np.isfinite(kappa) else None}


def export_annotation_events(episodes, *, count=200, seed=42):
    from .extraction import GOEMOTIONS_LABELS
    rows = []
    for ep in episodes:
        for index, step in enumerate(ep["steps"]):
            e = np.asarray(step["emotion_state"])
            rows.append({"event_id": f"{ep['run_id']}/{ep['task_id']}/{index}",
                         "skill": step.get("skill", step.get("skill_selected")),
                         "context": step["context"],
                         "top_emotions": [GOEMOTIONS_LABELS[i] for i in np.argsort(e)[-3:][::-1]],
                         "peak_activation": float(e.max())})
    if len(rows) < count:
        raise ValueError(f"Requested {count} events; only {len(rows)} available")
    rng = np.random.default_rng(seed)
    return [rows[i] for i in rng.choice(len(rows), count, replace=False)]


def plot_traces(episodes, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .extraction import GOEMOTIONS_LABELS
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    skills, values = coactivation(episodes)
    fig, ax = plt.subplots(figsize=(8, max(3, .35 * len(skills))))
    im = ax.imshow(values, aspect="auto", cmap="coolwarm")
    ax.set_xticks(range(len(GOEMOTIONS_LABELS)), GOEMOTIONS_LABELS, rotation=60, ha="right")
    ax.set_yticks(range(len(skills)), skills)
    fig.colorbar(im, ax=ax, label="Mean normalized activation")
    fig.tight_layout()
    fig.savefig(out_dir / "heatmap.pdf")
    plt.close(fig)
    confidence = [s["confidence"] for ep in episodes for s in ep["steps"]]
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.hist(confidence, bins=np.linspace(0, 1, 21))
    ax.axvline(.3, linestyle="--", color="black", label=r"$\tau=0.3$")
    ax.set(xlabel="Confidence", ylabel="Measured steps")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "confidence_hist.pdf")
    plt.close(fig)


def plot_trajectory(episode, out_dir):
    import matplotlib.pyplot as plt
    steps = episode["steps"]
    deltas, transitions = transition_points(steps)
    fig, axes = plt.subplots(3, 1, figsize=(5, 5), sharex=True)
    axes[0].plot(range(len(steps)), [s["confidence"] for s in steps], marker="o")
    axes[0].axhline(.3, color="black", linestyle="--")
    axes[0].set_ylabel("Confidence")
    axes[1].step(range(len(steps)), [s["template_idx"] + 1 for s in steps], where="mid")
    axes[1].set_ylabel("Template ID")
    axes[2].plot(range(1, len(steps)), deltas, marker="o")
    for t in transitions:
        axes[2].axvline(t, color="red", alpha=.3)
    axes[2].set(xlabel="Decision step (0-based)", ylabel="Cosine shift")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "trajectory.pdf")
    plt.close(fig)


def plot_generalization(baseline, candidate, out_dir):
    import matplotlib.pyplot as plt
    split = candidate[0]["split"]
    a, b = aggregate_runs(baseline, split=split), aggregate_runs(candidate, split=split)
    if not {"math", "mbpp"} <= a.keys() or not {"math", "mbpp"} <= b.keys():
        raise ValueError("Generalization plot requires measured MATH and MBPP runs for both methods")
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    for ax, benchmark, metric in zip(axes, ["math", "mbpp"], ["Accuracy (%)", "pass@1 (%)"]):
        values = [source[benchmark]["metrics"]["success_rate"] for source in (a, b)]
        ax.bar(["Baseline", "Candidate"], [r["mean"] for r in values],
               yerr=[r["std"] or 0 for r in values], capsize=3)
        ax.set(title=benchmark.upper(), ylabel=metric, ylim=(0, 100))
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "generalization.pdf")
    plt.close(fig)


def plot_layer_probes(probe_files, out_dir):
    """Plot t-SNE and Fisher diagnostics from activation exports."""
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    records = []
    for path in probe_files:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata_json"]))
            if meta.get("split") not in {"validation", "test"}:
                raise ValueError("Probe plots require held-out validation/test activations")
            records.append((data["activations"].copy(), data["labels"].copy(), meta))
    if not records:
        raise ValueError("No measured activation files")
    first = records[0][2]
    def identity(meta):
        extractor = meta["extractor_metadata"]
        return (meta.get("model_key"), meta.get("split"), meta.get("source_sha256"),
                extractor.get("model_revision"), extractor.get("validation_sha256"),
                extractor.get("story_sha256"), json.dumps(extractor.get("layer_sweep"), sort_keys=True))
    if any(identity(meta) != identity(first) for _, _, meta in records[1:]):
        raise ValueError("Layer probes mix models, revisions, sources or dataset partitions")
    if len({meta["layer"] for _, _, meta in records}) != len(records):
        raise ValueError("Duplicate layer probe archives")
    sweep = first["extractor_metadata"].get("layer_sweep")
    if isinstance(sweep, list):
        sweep = {str(r["layer"]): r["validation_hit_any"] for r in sweep}
    if not sweep:
        raise ValueError("Extractor metadata is missing measured validation layer scores")
    layers = sorted(int(k) for k in sweep)
    def fisher(x, labels):
        memberships = np.asarray(labels, dtype=bool)
        if memberships.ndim != 2 or memberships.shape[0] != len(x):
            raise ValueError("Probe labels must be a multi-hot class matrix")
        sizes = memberships.sum(axis=0)
        means = [x[memberships[:, k]].mean(axis=0) for k in range(memberships.shape[1]) if sizes[k]]
        weights = sizes[sizes > 0]
        global_mean = np.average(np.vstack(means), axis=0, weights=weights)
        between = sum(w * np.sum((m - global_mean) ** 2) for w, m in zip(weights, means))
        within = sum(np.sum((x[memberships[:, k]] - x[memberships[:, k]].mean(axis=0)) ** 2)
                     for k in range(memberships.shape[1]) if sizes[k])
        return float(between / within) if within > 0 else None
    diagnostic = {int(meta["layer"]): fisher(x, labels) for x, labels, meta in records}
    selected_layer = max(layers, key=lambda k: float(sweep.get(str(k), sweep.get(k))))
    selected = next((r for r in records if r[2]["layer"] == selected_layer), None)
    if selected is None:
        raise ValueError(f"Missing probe archive for validation-selected layer {selected_layer}")
    x, labels, _ = selected
    if len(x) < 4:
        raise ValueError("t-SNE needs at least four measured observations")
    xy = TSNE(n_components=2, perplexity=min(30, len(x) - 1), random_state=42, init="pca").fit_transform(x)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].plot(layers, [100 * float(sweep.get(str(k), sweep.get(k))) for k in layers], marker="o")
    axes[0].set(xlabel="Layer", ylabel="Validation hit-any (%)")
    axes[1].scatter(xy[:, 0], xy[:, 1], c=np.asarray(labels).argmax(axis=1), cmap="tab20", s=3, alpha=.5)
    axes[1].set(xlabel="t-SNE 1", ylabel="t-SNE 2")
    valid = sorted(k for k, v in diagnostic.items() if v is not None)
    axes[2].plot(valid, [diagnostic[k] for k in valid], marker="o")
    axes[2].set(xlabel="Layer", ylabel="Fisher ratio")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "preliminary_validation.pdf")
    plt.close(fig)


def plot_round_results(paths, out_dir, *, split="test"):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 3))
    for condition, path in enumerate(paths, 1):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        groups = defaultdict(list)
        for run in data["runs"]:
            if run.get("synthetic"):
                raise ValueError("Synthetic round curves are not paper evidence")
            for record in run.get("rounds", []):
                if record["phase"] != "round_evaluation" or record["feeds_evolution"] or record["split"] != split:
                    raise ValueError("Round curves require frozen held-out evaluation")
                metric = "avg" if record["benchmark"] == "alfworld" else "success_rate"
                groups[record["round"]].append(record["metrics"][metric])
        if not groups:
            raise ValueError("No measured held-out round evaluations")
        rounds = sorted(groups)
        means = np.array([np.mean(groups[r]) for r in rounds])
        std = np.array([np.std(groups[r], ddof=1) if len(groups[r]) > 1 else 0 for r in rounds])
        ax.plot(rounds, means, marker="o", label=f"Condition {condition}")
        ax.fill_between(rounds, means-std, means+std, alpha=.15)
    ax.set(xlabel="Evolution round", ylabel="Held-out success (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "evolution_rounds.pdf")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", nargs="+")
    parser.add_argument("--baseline_logs", nargs="+")
    parser.add_argument("--out", default="results/analysis.json")
    parser.add_argument("--figures")
    parser.add_argument("--annotation_events")
    parser.add_argument("--judgments")
    parser.add_argument("--human_annotations")
    parser.add_argument("--probe_files", nargs="+")
    parser.add_argument("--case_task_id")
    parser.add_argument("--generalization", action="store_true")
    parser.add_argument("--round_results", nargs="+")
    parser.add_argument("--split", choices=["test", "dev"], default="test")
    args = parser.parse_args(argv)
    if not args.logs and not args.probe_files:
        parser.error("Supply measured episode logs or exported layer probes")
    rows = read_episodes(args.logs, split=args.split) if args.logs else []
    result = {"run_summaries": aggregate_runs(rows, split=args.split)} if rows else {"probe_artifacts": args.probe_files}
    result.update({"split": args.split, "paper_eligible": args.split == "test"})
    if args.baseline_logs:
        baseline = read_episodes(args.baseline_logs, split=args.split)
        benchmarks = {r.get("benchmark", "unknown") for r in rows}
        result["paired_comparison"] = {
            name: paired_comparison([r for r in baseline if r.get("benchmark", "unknown") == name],
                                    [r for r in rows if r.get("benchmark", "unknown") == name])
            for name in sorted(benchmarks)
        }
    if args.judgments or args.human_annotations:
        if not (args.judgments and args.human_annotations):
            parser.error("Both annotation files are required")
        result["annotations"] = annotation_summary(
            json.loads(Path(args.judgments).read_text(encoding="utf-8")),
            json.loads(Path(args.human_annotations).read_text(encoding="utf-8")),
        )
    if args.annotation_events:
        target = Path(args.annotation_events)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(export_annotation_events(rows), indent=2), encoding="utf-8")
    if args.figures:
        Path(args.figures).mkdir(parents=True, exist_ok=True)
        if rows:
            plot_traces(rows, args.figures)
        if args.probe_files:
            plot_layer_probes(args.probe_files, args.figures)
        if args.case_task_id:
            matches = [r for r in rows if str(r["task_id"]) == args.case_task_id]
            if len(matches) != 1:
                parser.error("Case task ID must identify exactly one logged episode")
            plot_trajectory(matches[0], args.figures)
        if args.generalization:
            if not args.baseline_logs:
                parser.error("Generalization needs --baseline_logs")
            plot_generalization(read_episodes(args.baseline_logs, split=args.split), rows, args.figures)
        if args.round_results:
            plot_round_results(args.round_results, args.figures, split=args.split)
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
