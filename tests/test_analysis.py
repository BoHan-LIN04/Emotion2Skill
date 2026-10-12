import json
import pytest

from emotion2skill.analysis import read_episodes, aggregate_runs, paired_comparison, annotation_summary


def episode(run, task, success, **kwargs):
    return {"benchmark": "webshop", "run_id": run, "task_id": task,
            "backend": "webshop", "phase": "evaluation", "split": "test", "success": success,
            "outcome": float(success), "steps": [], **kwargs}


def test_synthetic_results_cannot_enter_paper_analysis(tmp_path):
    path = tmp_path / "episodes.jsonl"
    path.write_text(json.dumps(episode("run1", "task1", True, backend="mock")))
    with pytest.raises(ValueError, match="Synthetic"):
        read_episodes([path])


def test_independent_run_std_and_duplicate_detection():
    rows = [episode("r1", "t1", True), episode("r2", "t1", False)]
    result = aggregate_runs(rows)["webshop"]["metrics"]["success_rate"]
    assert result["mean"] == 50
    assert result["std"] == pytest.approx(70.710678)
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate_runs(rows + [rows[0]])
    with pytest.raises(ValueError, match="identical task sets"):
        aggregate_runs([episode("r1", "t1", True), episode("r2", "other", True)])


def test_dev_results_require_explicit_partition(tmp_path):
    path = tmp_path / "dev.jsonl"
    path.write_text(json.dumps(episode("r1", "t1", True, split="dev")))
    with pytest.raises(ValueError, match="Expected test"):
        read_episodes([path])
    assert len(read_episodes([path], split="dev")) == 1


def test_pairing_rejects_mismatched_tasks_and_clusters_repeated_runs():
    a = [episode("r1", "t1", False), episode("r2", "t1", False)]
    b = [episode("r1", "t1", True), episode("r2", "t1", True)]
    result = paired_comparison(a, b, resamples=100)
    assert result["n_tasks"] == 1
    assert result["difference_pp"] == 100
    assert result["bootstrap_95ci_pp"] == [100, 100]
    with pytest.raises(ValueError, match="identical"):
        paired_comparison(a, [episode("r1", "other", True)], resamples=100)


def test_annotations_are_matched_by_event_id():
    judgments = [{"event_id": str(i), "coherent": value} for i, value in enumerate([True, False, True, False])]
    result = annotation_summary(judgments, judgments[::-1])
    assert result["cohen_kappa"] == 1
    assert result["coherent_percent"] == 50


def test_plot_pipeline_uses_complete_logged_vectors(tmp_path):
    from emotion2skill.analysis import plot_traces, plot_trajectory
    steps = [{"emotion_state": [1.0 if j == i else 0.0 for j in range(27)],
              "skill_selected": "ProductSearch", "confidence": .4 + .1*i,
              "template_idx": i} for i in range(3)]
    record = episode("r1", "t1", True, steps=steps)
    plot_traces([record], tmp_path)
    plot_trajectory(record, tmp_path)
    assert {p.name for p in tmp_path.iterdir()} == {"heatmap.pdf", "confidence_hist.pdf", "trajectory.pdf"}


def test_layer_probe_pipeline_reads_export_schema(tmp_path):
    import numpy as np
    from emotion2skill.analysis import plot_layer_probes
    path = tmp_path / "probe.npz"
    labels = np.zeros((6, 27), dtype=int)
    labels[:3, 0] = 1
    labels[3:, 1] = 1
    meta = {"split": "test", "layer": 24, "model_key": "qwen3-8b", "source_sha256": "fixture", "extractor_metadata": {
        "layer_sweep": [{"layer": 24, "validation_hit_any": .5}]}}
    np.savez(path, activations=np.arange(24).reshape(6, 4).astype(float), labels=labels,
             metadata_json=np.asarray(json.dumps(meta)))
    plot_layer_probes([path], tmp_path)
    assert (tmp_path / "preliminary_validation.pdf").is_file()
