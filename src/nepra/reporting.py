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

ARTIFACT_SCHEMA_VERSION = 2
SUPPORTED_ARTIFACT_SCHEMA_VERSIONS = frozenset({1, ARTIFACT_SCHEMA_VERSION})
REQUIRED_FILES = {
    "resolved-config.yaml",
    "manifest.json",
    "metrics.csv",
    "summary.json",
    "environment.json",
}
METRIC_COLUMNS_V1 = {
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
METRIC_COLUMNS = METRIC_COLUMNS_V1 | {
    "dataset",
    "auroc",
    "enrolled_subjects",
    "unknown_subjects",
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
                "dataset": score.dataset,
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
                "auroc": None,
                "enrolled_subjects": None,
                "unknown_subjects": None,
            }
        )
    for score in result.attack_scores:
        rows.append(
            {
                "dataset": score.dataset,
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
                "auroc": None,
                "enrolled_subjects": None,
                "unknown_subjects": None,
            }
        )
    for score in result.open_set_scores:
        rows.append(
            {
                "dataset": score.dataset,
                "scope": "open_set_identity_attack",
                "condition": score.condition,
                "epsilon": score.epsilon,
                "seed": score.seed,
                "subject": None,
                "regime": score.regime,
                "model": score.attacker,
                "balanced_accuracy": None,
                "macro_f1": None,
                "advantage_over_chance": None,
                "session_accuracy": None,
                "auroc": score.auroc,
                "enrolled_subjects": score.enrolled_subjects,
                "unknown_subjects": score.unknown_subjects,
            }
        )
    return rows


def _condition_key(
    dataset: str,
    condition: str,
    epsilon: float | None,
) -> tuple[str, str, float | None]:
    return dataset, condition, epsilon


def _utility_summary(
    result: BenchmarkResult, config: ExperimentConfig
) -> dict[tuple[str, str, float | None], dict[str, Any]]:
    grouped: dict[tuple[str, str, float | None], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    f1_grouped: dict[tuple[str, str, float | None], list[float]] = defaultdict(list)
    for score in result.task_scores:
        key = _condition_key(score.dataset, score.condition, score.epsilon)
        grouped[key][score.subject].append(score.balanced_accuracy)
        f1_grouped[key].append(score.macro_f1)

    summary: dict[tuple[str, str, float | None], dict[str, Any]] = {}
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
) -> dict[tuple[str, str, float | None], dict[str, Any]]:
    grouped: dict[tuple[str, str, float | None], dict[tuple[str, str], list[Any]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for score in result.attack_scores:
        grouped[_condition_key(score.dataset, score.condition, score.epsilon)][
            (score.regime, score.attacker)
        ].append(score)

    summary: dict[tuple[str, str, float | None], dict[str, Any]] = {}
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


def _open_set_summary(
    result: BenchmarkResult,
    config: ExperimentConfig,
) -> dict[tuple[str, str, float | None], dict[str, Any]]:
    grouped: dict[tuple[str, str, float | None], dict[tuple[str, str], list[Any]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for score in result.open_set_scores:
        grouped[_condition_key(score.dataset, score.condition, score.epsilon)][
            (score.regime, score.attacker)
        ].append(score)

    summary: dict[tuple[str, str, float | None], dict[str, Any]] = {}
    for key, attacks in grouped.items():
        candidates: list[dict[str, Any]] = []
        for (regime, attacker), scores in attacks.items():
            values = [float(score.auroc) for score in scores]
            mean, interval = _mean_and_interval(values, config=config)
            candidates.append(
                {
                    "regime": regime,
                    "attacker": attacker,
                    "observations": len(scores),
                    "enrolled_subjects": scores[0].enrolled_subjects,
                    "unknown_subjects": scores[0].unknown_subjects,
                    "auroc_mean": mean,
                    "auroc_ci95": interval,
                }
            )
        strongest = max(candidates, key=lambda item: item["auroc_mean"])
        summary[key] = {"strongest": strongest, "all": candidates}
    return summary


def build_summary(config: ExperimentConfig, result: BenchmarkResult) -> dict[str, Any]:
    """Aggregate utility by subject and identity leakage by strongest attack."""
    utility = _utility_summary(result, config)
    attacks = _attack_summary(result, config)
    open_set = _open_set_summary(result, config)
    entries: list[dict[str, Any]] = []
    keys = sorted(
        utility,
        key=lambda item: (
            item[0],
            item[1],
            float("-inf") if item[2] is None else item[2],
        ),
    )
    for dataset, condition, epsilon in keys:
        key = (dataset, condition, epsilon)
        entries.append(
            {
                "dataset": dataset,
                "condition": condition,
                "epsilon": epsilon,
                "utility": utility[key],
                "identity_attack": attacks[key],
                "open_set_identity_attack": open_set[key],
            }
        )
    dataset_summaries = [
        {
            "dataset": dataset_result.dataset,
            "clipping": {
                "calibration": asdict(dataset_result.calibration_clipping),
                "heldout": asdict(dataset_result.heldout_clipping),
                "heldout_to_calibration_p95_ratio": (
                    dataset_result.heldout_clipping.norm_p95
                    / dataset_result.calibration_clipping.norm_p95
                ),
            },
            "representation": {
                "feature_dimensions": dataset_result.feature_dimensions,
                "calibration_trials": dataset_result.calibration_trials,
                "heldout_trials": dataset_result.heldout_trials,
            },
            "task_weights": {
                "minimum": float(np.min(dataset_result.task_weights.weights)),
                "maximum": float(np.max(dataset_result.task_weights.weights)),
                "rms": float(np.sqrt(np.mean(np.square(dataset_result.task_weights.weights)))),
            },
            "elapsed_seconds": dataset_result.elapsed_seconds,
        }
        for dataset_result in result.dataset_results
    ]
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_name": config.run.name,
        "datasets": [dataset.name for dataset in config.datasets],
        "entries": entries,
        "dataset_summaries": dataset_summaries,
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
    datasets = {entry.get("dataset") for entry in summary["entries"]}
    multi_dataset = len(datasets) > 1
    for entry in summary["entries"]:
        condition = entry["condition"]
        utility = entry["utility"]["balanced_accuracy_mean"]
        leakage = entry["identity_attack"]["strongest"]["balanced_accuracy_mean"]
        epsilon = entry["epsilon"]
        condition_label = condition if epsilon is None else f"{condition}, eps={epsilon:g}"
        label = f"{entry['dataset']}: {condition_label}" if multi_dataset else condition_label
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
    manifest: dict[str, Any] = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_id": run_id,
        "status": "completed",
        "datasets": [
            {
                "name": dataset.name,
                "subjects": list(dataset.subjects),
                "classes": list(dataset.classes),
                "session_split": dataset.session_split,
            }
            for dataset in config.datasets
        ],
        "protected_session": HELDOUT_SESSION,
        "public_calibration_session": CALIBRATION_SESSION,
        "git_commit": _git_commit(),
    }
    if len(config.datasets) == 1:
        manifest.update(
            {
                "dataset": config.dataset.name,
                "subjects": list(config.dataset.subjects),
                "classes": list(config.dataset.classes),
            }
        )
    _write_json(
        run_dir / "manifest.json",
        manifest,
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


def _artifact_schema_version(data: dict[str, Any], source: str) -> int:
    version = data.get("artifact_schema_version")
    if version not in SUPPORTED_ARTIFACT_SCHEMA_VERSIONS:
        raise ValueError(f"{source} has an unsupported artifact schema version")
    return int(version)


def _resolved_datasets(resolved_config: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(resolved_config.get("datasets"), list):
        datasets = resolved_config["datasets"]
    elif isinstance(resolved_config.get("dataset"), dict):
        datasets = [resolved_config["dataset"]]
    else:
        raise ValueError("resolved-config.yaml has no dataset section")
    if not all(isinstance(dataset, dict) for dataset in datasets):
        raise ValueError("resolved-config.yaml contains an invalid dataset section")
    return datasets


def _validate_manifest(manifest: dict[str, Any], resolved_config: dict[str, Any]) -> int:
    schema_version = _artifact_schema_version(manifest, "run manifest")
    if manifest.get("status") != "completed":
        raise ValueError("run manifest is not completed")
    if not _nonempty_string(manifest.get("run_id")):
        raise ValueError("run manifest has an invalid run_id")
    if manifest.get("public_calibration_session") != CALIBRATION_SESSION:
        raise ValueError("run manifest has an invalid public calibration session")
    if manifest.get("protected_session") != HELDOUT_SESSION:
        raise ValueError("run manifest has an invalid protected session")

    resolved_datasets = _resolved_datasets(resolved_config)
    if schema_version == 1:
        dataset = resolved_datasets[0]
        if manifest.get("dataset") != dataset.get("name"):
            raise ValueError("run manifest dataset does not match resolved configuration")
        if manifest.get("subjects") != dataset.get("subjects"):
            raise ValueError("run manifest subjects do not match resolved configuration")
        if manifest.get("classes") != dataset.get("classes"):
            raise ValueError("run manifest classes do not match resolved configuration")
    else:
        manifest_datasets = manifest.get("datasets")
        if not isinstance(manifest_datasets, list) or not manifest_datasets:
            raise ValueError("run manifest has no datasets")
        expected = [
            {
                "name": dataset.get("name"),
                "subjects": dataset.get("subjects"),
                "classes": dataset.get("classes"),
                "session_split": dataset.get("session_split", "fixed"),
            }
            for dataset in resolved_datasets
        ]
        if manifest_datasets != expected:
            raise ValueError("run manifest datasets do not match resolved configuration")
    git_commit = manifest.get("git_commit")
    if git_commit is not None and not _nonempty_string(git_commit):
        raise ValueError("run manifest has an invalid git_commit")
    return schema_version


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


def _validate_summary(summary: dict[str, Any]) -> int:
    schema_version = _artifact_schema_version(summary, "summary.json")
    if not _nonempty_string(summary.get("run_name")):
        raise ValueError("summary.json has an invalid run_name")
    if schema_version == 2:
        datasets = summary.get("datasets")
        if not isinstance(datasets, list) or not all(_nonempty_string(item) for item in datasets):
            raise ValueError("summary.json contains invalid datasets")
        dataset_summaries = summary.get("dataset_summaries")
        if not isinstance(dataset_summaries, list) or not dataset_summaries:
            raise ValueError("summary.json contains invalid dataset summaries")
    entries = summary.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("run summary has no benchmark entries")
    for entry in entries:
        if not isinstance(entry, dict) or not _nonempty_string(entry.get("condition")):
            raise ValueError("summary.json contains an invalid benchmark entry")
        if schema_version == 2 and not _nonempty_string(entry.get("dataset")):
            raise ValueError("summary.json contains an entry without dataset")
        utility = entry.get("utility")
        identity_attack = entry.get("identity_attack")
        open_set_identity_attack = entry.get("open_set_identity_attack")
        if not isinstance(utility, dict) or "balanced_accuracy_mean" not in utility:
            raise ValueError("summary.json contains an invalid utility summary")
        if not isinstance(identity_attack, dict):
            raise ValueError("summary.json contains an invalid identity attack summary")
        if schema_version == 2 and not isinstance(open_set_identity_attack, dict):
            raise ValueError("summary.json contains an invalid open-set identity attack summary")
        strongest = identity_attack.get("strongest")
        all_attacks = identity_attack.get("all")
        if not isinstance(strongest, dict) or not isinstance(all_attacks, list) or not all_attacks:
            raise ValueError("summary.json contains an invalid identity attack summary")
    return schema_version


def _validate_metrics(path: Path) -> int:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        if fieldnames == METRIC_COLUMNS:
            schema_version = 2
        elif fieldnames == METRIC_COLUMNS_V1:
            schema_version = 1
        else:
            raise ValueError("metrics.csv has an invalid schema")
        rows = list(reader)
    if not rows:
        raise ValueError("metrics.csv contains no rows")
    for row in rows:
        if schema_version == 2 and not _nonempty_string(row.get("dataset")):
            raise ValueError("metrics.csv contains a row without dataset")
        scope = row.get("scope")
        if scope not in {"utility", "identity_attack", "open_set_identity_attack"}:
            raise ValueError("metrics.csv contains an invalid scope")
        if not _nonempty_string(row.get("condition")):
            raise ValueError("metrics.csv contains an invalid condition")
        if scope != "open_set_identity_attack" and (
            not _finite_metric(row, "balanced_accuracy") or not _finite_metric(row, "macro_f1")
        ):
            raise ValueError("metrics.csv contains invalid performance metrics")
        if scope == "utility" and not _nonempty_string(row.get("subject")):
            raise ValueError("metrics.csv contains a utility row without a subject")
        if scope in {"identity_attack", "open_set_identity_attack"} and (
            not _nonempty_string(row.get("regime")) or not _nonempty_string(row.get("model"))
        ):
            raise ValueError("metrics.csv contains an identity row without attacker metadata")
        if scope == "identity_attack" and (
            not _finite_metric(row, "advantage_over_chance")
            or not _finite_metric(row, "session_accuracy")
        ):
            raise ValueError("metrics.csv contains invalid identity metrics")
        if scope == "open_set_identity_attack":
            if not _finite_metric(row, "auroc"):
                raise ValueError("metrics.csv contains invalid open-set AUROC")
            try:
                enrolled_subjects = int(row["enrolled_subjects"])
                unknown_subjects = int(row["unknown_subjects"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("metrics.csv contains invalid open-set subject counts") from error
            if enrolled_subjects <= 0 or unknown_subjects <= 0:
                raise ValueError("metrics.csv contains invalid open-set subject counts")
    return schema_version


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
    manifest_version = _validate_manifest(
        _read_json_object(run_dir / "manifest.json"), resolved_config
    )
    _validate_environment(_read_json_object(run_dir / "environment.json"))
    summary_version = _validate_summary(_read_json_object(run_dir / "summary.json"))
    metrics_version = _validate_metrics(run_dir / "metrics.csv")
    if len({manifest_version, summary_version, metrics_version}) != 1:
        raise ValueError("run artifact schema versions do not match")

    plot = run_dir / "plots" / "privacy-utility.png"
    if not plot.is_file() or plot.stat().st_size == 0:
        raise ValueError("run contains no privacy-utility plot")
