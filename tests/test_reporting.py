from __future__ import annotations

import csv
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from nepra.cli import main
from nepra.config import ExperimentConfig
from nepra.evaluation import BenchmarkResult
from nepra.reporting import METRIC_COLUMNS, build_summary, validate_run, write_run

ATTACK_METRICS = (
    "balanced_accuracy",
    "macro_f1",
    "advantage_over_chance",
    "session_accuracy",
)


def _write_json(path: Path, data: dict[str, object]) -> None:
    path.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")


def test_run_artifact_round_trip(
    temporary_output_config: ExperimentConfig,
    benchmark_result: BenchmarkResult,
) -> None:
    run_dir = write_run(
        temporary_output_config,
        benchmark_result,
        created_at=datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
    )

    validate_run(run_dir)
    assert main(["validate-run", str(run_dir)]) == 0
    with (run_dir / "manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert manifest["artifact_schema_version"] == 2
    assert manifest["datasets"] == [
        {
            "name": "synthetic",
            "subjects": [1, 2, 3, 4],
            "classes": ["left_hand", "right_hand", "feet", "tongue"],
            "session_split": "fixed",
        }
    ]
    assert manifest["protected_session"] == "1test"
    assert manifest["status"] == "completed"

    with (run_dir / "metrics.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert set(reader.fieldnames or ()) == METRIC_COLUMNS
        rows = list(reader)
        assert len(rows) == 612
        assert {row["dataset"] for row in rows} == {"synthetic"}
        assert {row["scope"] for row in rows} == {
            "utility",
            "identity_attack",
            "open_set_identity_attack",
            "repeated_release_identity_attack",
        }

    with (run_dir / "summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    assert summary["artifact_schema_version"] == 2
    assert summary["datasets"] == ["synthetic"]
    assert (
        summary["hierarchical_summary"]["method"] == "dataset_subject_seed_hierarchical_bootstrap"
    )
    assert len(summary["hierarchical_summary"]["entries"]) == 6
    assert summary["dataset_summaries"][0]["dataset"] == "synthetic"
    assert len(summary["entries"]) == 6
    assert {entry["dataset"] for entry in summary["entries"]} == {"synthetic"}
    assert all("strongest" in entry["identity_attack"] for entry in summary["entries"])
    assert all("strongest" in entry["open_set_identity_attack"] for entry in summary["entries"])
    assert all(
        "group_sizes" in entry["repeated_release_identity_attack"] for entry in summary["entries"]
    )
    assert all("privacy_composition_basic" in entry for entry in summary["entries"])
    for entry in summary["entries"]:
        for candidate in entry["identity_attack"]["all"]:
            assert candidate["observations"] == 1
            for metric in ATTACK_METRICS:
                assert f"{metric}_mean" in candidate
                assert candidate[f"{metric}_ci95"] is None
        for candidate in entry["open_set_identity_attack"]["all"]:
            assert candidate["observations"] == 1
            assert "auroc_mean" in candidate
            assert candidate["auroc_ci95"] is None
        assert len(entry["repeated_release_identity_attack"]["group_sizes"]) == 5
        assert len(entry["privacy_composition_basic"]) == 5

    for plot_name in (
        "privacy-utility.png",
        "privacy-utility-by-dataset.png",
        "hierarchical-ci.png",
        "repeated-release-leakage.png",
        "open-set-auroc.png",
    ):
        plot = run_dir / "plots" / plot_name
        assert plot.is_file()
        assert plot.stat().st_size > 0


def test_published_v01_run_validates() -> None:
    validate_run("docs/results/v0.1/run")


def test_identity_attack_summary_reports_seed_uncertainty(
    smoke_config: ExperimentConfig,
    benchmark_result: BenchmarkResult,
) -> None:
    original = next(
        score
        for score in benchmark_result.attack_scores
        if score.condition == "analytic_gaussian"
        and score.epsilon == 1.0
        and score.regime == "clean_auxiliary"
        and score.attacker == "logistic_regression"
    )
    extra_observation = replace(
        original,
        seed=123,
        balanced_accuracy=original.balanced_accuracy + 0.1,
        macro_f1=original.macro_f1 + 0.1,
        advantage_over_chance=original.advantage_over_chance + 0.1,
        session_accuracy=min(1.0, original.session_accuracy + 0.1),
    )
    result = replace(
        benchmark_result,
        attack_scores=(*benchmark_result.attack_scores, extra_observation),
    )

    summary = build_summary(smoke_config, result)
    entry = next(
        item
        for item in summary["entries"]
        if item["condition"] == "analytic_gaussian" and item["epsilon"] == 1.0
    )
    candidate = next(
        item
        for item in entry["identity_attack"]["all"]
        if item["regime"] == "clean_auxiliary" and item["attacker"] == "logistic_regression"
    )

    assert candidate["observations"] == 2
    for metric in ATTACK_METRICS:
        interval = candidate[f"{metric}_ci95"]
        assert interval is not None
        assert len(interval) == 2
        assert interval[0] <= candidate[f"{metric}_mean"] <= interval[1]


def test_validation_rejects_incomplete_run(tmp_path: Path) -> None:
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    try:
        validate_run(incomplete)
    except ValueError as error:
        assert "missing required files" in str(error)
    else:
        raise AssertionError("incomplete run should fail validation")


def test_validation_rejects_manifest_session_mismatch(
    temporary_output_config: ExperimentConfig,
    benchmark_result: BenchmarkResult,
) -> None:
    run_dir = write_run(
        temporary_output_config,
        benchmark_result,
        created_at=datetime(2026, 1, 2, 3, 4, 6, tzinfo=UTC),
    )
    manifest_path = run_dir / "manifest.json"
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest["protected_session"] = "0train"
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="protected session"):
        validate_run(run_dir)


def test_validation_rejects_empty_summary(
    temporary_output_config: ExperimentConfig,
    benchmark_result: BenchmarkResult,
) -> None:
    run_dir = write_run(
        temporary_output_config,
        benchmark_result,
        created_at=datetime(2026, 1, 2, 3, 4, 7, tzinfo=UTC),
    )
    summary_path = run_dir / "summary.json"
    with summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    summary["entries"] = []
    _write_json(summary_path, summary)

    with pytest.raises(ValueError, match="benchmark entries"):
        validate_run(run_dir)


def test_validation_rejects_invalid_metrics(
    temporary_output_config: ExperimentConfig,
    benchmark_result: BenchmarkResult,
) -> None:
    run_dir = write_run(
        temporary_output_config,
        benchmark_result,
        created_at=datetime(2026, 1, 2, 3, 4, 8, tzinfo=UTC),
    )
    with (run_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(METRIC_COLUMNS))
        writer.writeheader()
        writer.writerow(
            {
                "scope": "identity_attack",
                "dataset": "synthetic",
                "condition": "clean",
                "epsilon": "",
                "seed": "",
                "subject": "",
                "regime": "",
                "model": "logistic_regression",
                "balanced_accuracy": "0.5",
                "macro_f1": "0.5",
                "advantage_over_chance": "0.0",
                "session_accuracy": "0.5",
            }
        )

    with pytest.raises(ValueError, match="attacker metadata"):
        validate_run(run_dir)


def test_validation_rejects_missing_tradeoff_plot(
    temporary_output_config: ExperimentConfig,
    benchmark_result: BenchmarkResult,
) -> None:
    run_dir = write_run(
        temporary_output_config,
        benchmark_result,
        created_at=datetime(2026, 1, 2, 3, 4, 9, tzinfo=UTC),
    )
    (run_dir / "plots" / "privacy-utility.png").unlink()

    with pytest.raises(ValueError, match="privacy-utility plot"):
        validate_run(run_dir)
