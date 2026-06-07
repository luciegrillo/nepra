from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

from nepra.cli import main
from nepra.config import ExperimentConfig
from nepra.evaluation import BenchmarkResult
from nepra.reporting import METRIC_COLUMNS, validate_run, write_run


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


def test_validation_rejects_incomplete_run(tmp_path: Path) -> None:
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    try:
        validate_run(incomplete)
    except ValueError as error:
        assert "missing required files" in str(error)
    else:
        raise AssertionError("incomplete run should fail validation")
