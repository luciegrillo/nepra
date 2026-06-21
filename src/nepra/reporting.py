"""Versioned benchmark artifacts, summaries, and validation."""

from __future__ import annotations

import csv
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from nepra.config import ExperimentConfig
from nepra.data import CALIBRATION_SESSION, HELDOUT_SESSION
from nepra.evaluation import BenchmarkResult, bootstrap_mean_interval

ARTIFACT_SCHEMA_VERSION = 1
REQUIRED_FILES = {
    "resolved-config.yaml",
    "manifest.json",
    "metrics.csv",
    "summary.json",
    "environment.json",
}
METRIC_COLUMNS = {
    "scope",
    "condition",
    "epsilon",
    "seed",
    "subject",
    "regime",
    "model",
    "balanced_accuracy",
    "macro_f1",
    "advantage_over_chance",
    "session_accuracy",
}
ATTACK_METRICS = (
    "balanced_accuracy",
    "macro_f1",
    "advantage_over_chance",
    "session_accuracy",
)


def _git_commit() -> str | None:
    git = shutil.which("git")
    if git is None:
        return None
    try:
        # The executable is resolved to an absolute path and all arguments are constant.
        return subprocess.run(  # noqa: S603
            [git, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _metric_rows(result: BenchmarkResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for score in result.task_scores:
        rows.append(
            {
                "scope": "utility",
                "condition": score.condition,
                "epsilon": score.epsilon,
                "seed": score.seed,
                "subject": score.subject,
                "regime": None,
                "model": "personalized_logistic_regression",
                "balanced_accuracy": score.balanced_accuracy,
                "macro_f1": score.macro_f1,
                "advantage_over_chance": None,
                "session_accuracy": None,
            }
        )
    for score in result.attack_scores:
        rows.append(
            {
                "scope": "identity_attack",
                "condition": score.condition,
                "epsilon": score.epsilon,
                "seed": score.seed,
                "subject": None,
                "regime": score.regime,
                "model": score.attacker,
                "balanced_accuracy": score.balanced_accuracy,
                "macro_f1": score.macro_f1,
                "advantage_over_chance": score.advantage_over_chance,
                "session_accuracy": score.session_accuracy,
            }
        )
    return rows


def _condition_key(condition: str, epsilon: float | None) -> tuple[str, float | None]:
    return condition, epsilon


def _utility_summary(
    result: BenchmarkResult, config: ExperimentConfig
) -> dict[tuple[str, float | None], dict[str, Any]]:
    grouped: dict[tuple[str, float | None], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    f1_grouped: dict[tuple[str, float | None], list[float]] = defaultdict(list)
    for score in result.task_scores:
        key = _condition_key(score.condition, score.epsilon)
        grouped[key][score.subject].append(score.balanced_accuracy)
        f1_grouped[key].append(score.macro_f1)

    summary: dict[tuple[str, float | None], dict[str, Any]] = {}
    for key, subjects in grouped.items():
        subject_means = np.asarray(
            [np.mean(values) for values in subjects.values()], dtype=np.float64
        )
        low, high = bootstrap_mean_interval(
            subject_means,
            samples=config.evaluation.bootstrap_samples,
            seed=config.privacy.seeds[0],
        )
        summary[key] = {
            "balanced_accuracy_mean": float(np.mean(subject_means)),
            "balanced_accuracy_ci95": [low, high],
            "macro_f1_mean": float(np.mean(f1_grouped[key])),
            "subjects": len(subject_means),
        }
    return summary


def _mean_and_interval(
    values: list[float],
    *,
    config: ExperimentConfig,
) -> tuple[float, list[float] | None]:
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, None
    low, high = bootstrap_mean_interval(
        values,
        samples=config.evaluation.bootstrap_samples,
        seed=config.privacy.seeds[0],
    )
    return mean, [low, high]


def _attack_summary(
    result: BenchmarkResult,
    config: ExperimentConfig,
) -> dict[tuple[str, float | None], dict[str, Any]]:
    grouped: dict[tuple[str, float | None], dict[tuple[str, str], list[Any]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for score in result.attack_scores:
        grouped[_condition_key(score.condition, score.epsilon)][
            (score.regime, score.attacker)
        ].append(score)

    summary: dict[tuple[str, float | None], dict[str, Any]] = {}
    for key, attacks in grouped.items():
        candidates: list[dict[str, Any]] = []
        for (regime, attacker), scores in attacks.items():
            candidate: dict[str, Any] = {
                "regime": regime,
                "attacker": attacker,
                "observations": len(scores),
            }
            for metric in ATTACK_METRICS:
                values = [float(getattr(score, metric)) for score in scores]
                mean, interval = _mean_and_interval(values, config=config)
                candidate[f"{metric}_mean"] = mean
                candidate[f"{metric}_ci95"] = interval
            candidates.append(candidate)
        strongest = max(candidates, key=lambda item: item["balanced_accuracy_mean"])
        summary[key] = {"strongest": strongest, "all": candidates}
    return summary


def build_summary(config: ExperimentConfig, result: BenchmarkResult) -> dict[str, Any]:
    """Aggregate utility by subject and identity leakage by strongest attack."""
    utility = _utility_summary(result, config)
    attacks = _attack_summary(result, config)
    entries: list[dict[str, Any]] = []
    keys = sorted(
        utility,
        key=lambda item: (
            item[0],
            float("-inf") if item[1] is None else item[1],
        ),
    )
    for condition, epsilon in keys:
        key = (condition, epsilon)
        entries.append(
            {
                "condition": condition,
                "epsilon": epsilon,
                "utility": utility[key],
                "identity_attack": attacks[key],
            }
        )
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_name": config.run.name,
        "entries": entries,
        "clipping": {
            "calibration": asdict(result.calibration_clipping),
            "heldout": asdict(result.heldout_clipping),
            "heldout_to_calibration_p95_ratio": (
                result.heldout_clipping.norm_p95 / result.calibration_clipping.norm_p95
            ),
        },
        "representation": {
            "feature_dimensions": result.feature_dimensions,
            "calibration_trials": result.calibration_trials,
            "heldout_trials": result.heldout_trials,
        },
        "task_weights": {
            "minimum": float(np.min(result.task_weights.weights)),
            "maximum": float(np.max(result.task_weights.weights)),
            "rms": float(np.sqrt(np.mean(np.square(result.task_weights.weights)))),
        },
        "elapsed_seconds": result.elapsed_seconds,
    }


def _plot_tradeoff(summary: dict[str, Any], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    markers = {
        "clean": "o",
        "clipped": "s",
        "analytic_gaussian": "^",
        "task_weighted_empirical": "D",
    }
    fig, axis = plt.subplots(figsize=(7.2, 5.0))
    for entry in summary["entries"]:
        condition = entry["condition"]
        utility = entry["utility"]["balanced_accuracy_mean"]
        leakage = entry["identity_attack"]["strongest"]["balanced_accuracy_mean"]
        epsilon = entry["epsilon"]
        label = condition if epsilon is None else f"{condition}, eps={epsilon:g}"
        axis.scatter(
            leakage,
            utility,
            marker=markers[condition],
            s=64,
            label=label,
        )
    axis.set_xlabel("Strongest identity balanced accuracy (lower is better)")
    axis.set_ylabel("Motor-task balanced accuracy (higher is better)")
    axis.set_title("NEPRA privacy-utility benchmark")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _environment() -> dict[str, Any]:
    packages = [
        "diffprivlib",
        "matplotlib",
        "moabb",
        "numpy",
        "pandas",
        "pyriemann",
        "scikit-learn",
        "scipy",
    ]
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {package: importlib.metadata.version(package) for package in packages},
        "git_commit": _git_commit(),
    }


def write_run(
    config: ExperimentConfig,
    result: BenchmarkResult,
    *,
    created_at: datetime | None = None,
) -> Path:
    """Write a complete versioned run directory and return its path."""
    timestamp = (created_at or datetime.now(UTC)).strftime("%Y%m%dT%H%M%S%fZ")
    run_id = f"{timestamp}-{config.run.name}"
    run_dir = config.run.output_dir / run_id
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=False)

    with (run_dir / "resolved-config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)

    rows = _metric_rows(result)
    with (run_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = build_summary(config, result)
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "environment.json", _environment())
    _write_json(
        run_dir / "manifest.json",
        {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "run_id": run_id,
            "status": "completed",
            "dataset": config.dataset.name,
            "subjects": list(config.dataset.subjects),
            "classes": list(config.dataset.classes),
            "protected_session": "1test",
            "public_calibration_session": "0train",
            "git_commit": _git_commit(),
        },
    )
    _plot_tradeoff(summary, plots_dir / "privacy-utility.png")
    validate_run(run_dir)
    return run_dir


def _read_json_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return data


def _load_resolved_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("resolved-config.yaml must contain a mapping")
    return data


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _finite_metric(row: dict[str, str], column: str) -> bool:
    try:
        return np.isfinite(float(row[column]))
    except (KeyError, TypeError, ValueError):
        return False


def _validate_manifest(manifest: dict[str, Any], resolved_config: dict[str, Any]) -> None:
    if manifest.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("unsupported artifact schema version")
    if manifest.get("status") != "completed":
        raise ValueError("run manifest is not completed")
    if not _nonempty_string(manifest.get("run_id")):
        raise ValueError("run manifest has an invalid run_id")
    if manifest.get("public_calibration_session") != CALIBRATION_SESSION:
        raise ValueError("run manifest has an invalid public calibration session")
    if manifest.get("protected_session") != HELDOUT_SESSION:
        raise ValueError("run manifest has an invalid protected session")

    dataset = resolved_config.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("resolved-config.yaml has no dataset section")
    if manifest.get("dataset") != dataset.get("name"):
        raise ValueError("run manifest dataset does not match resolved configuration")
    if manifest.get("subjects") != dataset.get("subjects"):
        raise ValueError("run manifest subjects do not match resolved configuration")
    if manifest.get("classes") != dataset.get("classes"):
        raise ValueError("run manifest classes do not match resolved configuration")
    git_commit = manifest.get("git_commit")
    if git_commit is not None and not _nonempty_string(git_commit):
        raise ValueError("run manifest has an invalid git_commit")


def _validate_environment(environment: dict[str, Any]) -> None:
    if not _nonempty_string(environment.get("python")):
        raise ValueError("environment.json has an invalid python value")
    if not _nonempty_string(environment.get("platform")):
        raise ValueError("environment.json has an invalid platform value")
    packages = environment.get("packages")
    if not isinstance(packages, dict) or not packages:
        raise ValueError("environment.json has no package versions")
    invalid_versions = (
        not _nonempty_string(name) or not _nonempty_string(version)
        for name, version in packages.items()
    )
    if any(invalid_versions):
        raise ValueError("environment.json has invalid package versions")


def _validate_summary(summary: dict[str, Any]) -> None:
    if summary.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("summary.json has an unsupported artifact schema version")
    if not _nonempty_string(summary.get("run_name")):
        raise ValueError("summary.json has an invalid run_name")
    entries = summary.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("run summary has no benchmark entries")
    for entry in entries:
        if not isinstance(entry, dict) or not _nonempty_string(entry.get("condition")):
            raise ValueError("summary.json contains an invalid benchmark entry")
        utility = entry.get("utility")
        identity_attack = entry.get("identity_attack")
        if not isinstance(utility, dict) or "balanced_accuracy_mean" not in utility:
            raise ValueError("summary.json contains an invalid utility summary")
        if not isinstance(identity_attack, dict):
            raise ValueError("summary.json contains an invalid identity attack summary")
        strongest = identity_attack.get("strongest")
        all_attacks = identity_attack.get("all")
        if not isinstance(strongest, dict) or not isinstance(all_attacks, list) or not all_attacks:
            raise ValueError("summary.json contains an invalid identity attack summary")


def _validate_metrics(path: Path) -> None:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or ()) != METRIC_COLUMNS:
            raise ValueError("metrics.csv has an invalid schema")
        rows = list(reader)
    if not rows:
        raise ValueError("metrics.csv contains no rows")
    for row in rows:
        scope = row.get("scope")
        if scope not in {"utility", "identity_attack"}:
            raise ValueError("metrics.csv contains an invalid scope")
        if not _nonempty_string(row.get("condition")):
            raise ValueError("metrics.csv contains an invalid condition")
        if not _finite_metric(row, "balanced_accuracy") or not _finite_metric(row, "macro_f1"):
            raise ValueError("metrics.csv contains invalid performance metrics")
        if scope == "utility" and not _nonempty_string(row.get("subject")):
            raise ValueError("metrics.csv contains a utility row without a subject")
        if scope == "identity_attack":
            if not _nonempty_string(row.get("regime")) or not _nonempty_string(row.get("model")):
                raise ValueError("metrics.csv contains an identity row without attacker metadata")
            if not _finite_metric(row, "advantage_over_chance") or not _finite_metric(
                row, "session_accuracy"
            ):
                raise ValueError("metrics.csv contains invalid identity metrics")


def validate_run(path: str | Path) -> None:
    """Validate the structure and minimum contents of a run directory."""
    run_dir = Path(path)
    if not run_dir.is_dir():
        raise ValueError(f"run directory does not exist: {run_dir}")
    present = {item.name for item in run_dir.iterdir() if item.is_file()}
    missing = REQUIRED_FILES - present
    if missing:
        raise ValueError(f"run is missing required files: {sorted(missing)}")

    resolved_config = _load_resolved_config(run_dir / "resolved-config.yaml")
    _validate_manifest(_read_json_object(run_dir / "manifest.json"), resolved_config)
    _validate_environment(_read_json_object(run_dir / "environment.json"))
    _validate_summary(_read_json_object(run_dir / "summary.json"))
    _validate_metrics(run_dir / "metrics.csv")

    plot = run_dir / "plots" / "privacy-utility.png"
    if not plot.is_file() or plot.stat().st_size == 0:
        raise ValueError("run contains no privacy-utility plot")
