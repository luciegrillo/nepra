from __future__ import annotations

import csv
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

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
    assert manifest["protected_session"] == "1test"
    assert manifest["status"] == "completed"

    with (run_dir / "metrics.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert set(reader.fieldnames or ()) == METRIC_COLUMNS
        assert len(list(reader)) == 72

    with (run_dir / "summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    assert len(summary["entries"]) == 6
    assert all("strongest" in entry["identity_attack"] for entry in summary["entries"])
    for entry in summary["entries"]:
        for candidate in entry["identity_attack"]["all"]:
            assert candidate["observations"] == 1
            for metric in ATTACK_METRICS:
                assert f"{metric}_mean" in candidate
                assert candidate[f"{metric}_ci95"] is None


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
