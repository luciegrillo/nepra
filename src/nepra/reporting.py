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
    "group_size",
    "group_accuracy",
    "groups",
    "max_releases",
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
                "group_size": None,
                "group_accuracy": None,
                "groups": None,
                "max_releases": None,
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
                "group_size": None,
                "group_accuracy": None,
                "groups": None,
                "max_releases": None,
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
                "group_size": None,
                "group_accuracy": None,
                "groups": None,
                "max_releases": None,
            }
        )
    for score in result.repeated_release_scores:
        rows.append(
            {
                "dataset": score.dataset,
                "scope": "repeated_release_identity_attack",
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
                "auroc": None,
                "enrolled_subjects": None,
                "unknown_subjects": None,
                "group_size": score.group_size,
                "group_accuracy": score.group_accuracy,
                "groups": score.groups,
                "max_releases": score.max_releases,
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


def _repeated_release_summary(
    result: BenchmarkResult,
    config: ExperimentConfig,
) -> dict[tuple[str, str, float | None], dict[str, Any]]:
    grouped: dict[tuple[str, str, float | None], dict[str, dict[tuple[str, str], list[Any]]]] = (
        defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )
    for score in result.repeated_release_scores:
        grouped[_condition_key(score.dataset, score.condition, score.epsilon)][score.group_size][
            (score.regime, score.attacker)
        ].append(score)

    summary: dict[tuple[str, str, float | None], dict[str, Any]] = {}
    for key, group_sizes in grouped.items():
        entries: list[dict[str, Any]] = []
        for group_size, attacks in sorted(
            group_sizes.items(), key=lambda item: _group_size_key(item[0])
        ):
            candidates: list[dict[str, Any]] = []
            for (regime, attacker), scores in attacks.items():
                values = [float(score.group_accuracy) for score in scores]
                mean, interval = _mean_and_interval(values, config=config)
                candidates.append(
                    {
                        "regime": regime,
                        "attacker": attacker,
                        "observations": len(scores),
                        "groups": int(np.sum([score.groups for score in scores])),
                        "max_releases": int(max(score.max_releases for score in scores)),
                        "group_accuracy_mean": mean,
                        "group_accuracy_ci95": interval,
                    }
                )
            strongest = max(candidates, key=lambda item: item["group_accuracy_mean"])
            entries.append({"group_size": group_size, "strongest": strongest, "all": candidates})
        summary[key] = {"group_sizes": entries}
    return summary


def _group_size_key(group_size: str) -> tuple[int, int | str]:
    if group_size == "all":
        return (1, group_size)
    return (0, int(group_size))


def _basic_composition(
    repeated_release: dict[str, Any],
    *,
    epsilon: float | None,
    delta: float,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for item in repeated_release["group_sizes"]:
        max_releases = item["strongest"]["max_releases"]
        entries.append(
            {
                "group_size": item["group_size"],
                "max_releases": max_releases,
                "epsilon_basic": None if epsilon is None else float(max_releases * epsilon),
                "delta_basic": None if epsilon is None else float(max_releases * delta),
            }
        )
    return entries


def _hierarchical_mean_interval(
    groups: list[list[float]],
    *,
    config: ExperimentConfig,
) -> tuple[float, list[float] | None]:
    cleaned = [np.asarray(group, dtype=np.float64) for group in groups if group]
    if not cleaned:
        raise ValueError("hierarchical summary requires at least one group")
    mean = float(np.mean([np.mean(group) for group in cleaned]))
    if len(cleaned) < 2:
        return mean, None
    rng = np.random.default_rng(config.privacy.seeds[0])
    sampled_means: list[float] = []
    for _ in range(config.evaluation.bootstrap_samples):
        sampled_groups = rng.integers(0, len(cleaned), size=len(cleaned))
        values: list[float] = []
        for group_index in sampled_groups:
            group = cleaned[int(group_index)]
            sampled_values = group[rng.integers(0, len(group), size=len(group))]
            values.append(float(np.mean(sampled_values)))
        sampled_means.append(float(np.mean(values)))
    low, high = np.quantile(sampled_means, [0.025, 0.975])
    return mean, [float(low), float(high)]


def _mean_ci_from_values(
    values: list[float],
    *,
    config: ExperimentConfig,
) -> dict[str, Any]:
    mean, interval = _mean_and_interval(values, config=config)
    return {"mean": mean, "ci95": interval, "datasets": len(values)}


def _paired_delta_summary(
    *,
    utility_means: dict[tuple[str, str, float | None], float],
    identity_means: dict[tuple[str, str, float | None], float],
    datasets: list[str],
    condition: str,
    epsilon: float | None,
    baseline: str,
    config: ExperimentConfig,
) -> dict[str, Any]:
    utility_changes: list[float] = []
    identity_reductions: list[float] = []
    for dataset in datasets:
        key = (dataset, condition, epsilon)
        baseline_key = (dataset, baseline, None)
        if key not in utility_means or baseline_key not in utility_means:
            continue
        if key not in identity_means or baseline_key not in identity_means:
            continue
        utility_changes.append(utility_means[key] - utility_means[baseline_key])
        identity_reductions.append(identity_means[baseline_key] - identity_means[key])
    return {
        "utility_change": _mean_ci_from_values(utility_changes, config=config),
        "identity_reduction": _mean_ci_from_values(identity_reductions, config=config),
    }


def _hierarchical_scientific_summary(
    config: ExperimentConfig,
    result: BenchmarkResult,
) -> dict[str, Any]:
    dataset_names = [dataset.name for dataset in config.datasets]
    utility_groups: dict[tuple[str, float | None], list[list[float]]] = defaultdict(list)
    utility_means: dict[tuple[str, str, float | None], float] = {}
    utility_by_dataset: dict[tuple[str, str, float | None], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for score in result.task_scores:
        utility_by_dataset[(score.dataset, score.condition, score.epsilon)][score.subject].append(
            score.balanced_accuracy
        )
    for key, subjects in utility_by_dataset.items():
        subject_means = [float(np.mean(values)) for values in subjects.values()]
        utility_means[key] = float(np.mean(subject_means))
        utility_groups[(key[1], key[2])].append(subject_means)

    identity_groups, identity_means = _strongest_metric_groups(
        result.attack_scores,
        metric="balanced_accuracy",
    )
    open_set_groups, _ = _strongest_metric_groups(result.open_set_scores, metric="auroc")
    repeated_groups, _ = _strongest_repeated_groups(result)

    entries: list[dict[str, Any]] = []
    for condition, epsilon in sorted(
        utility_groups,
        key=lambda item: (item[0], float("-inf") if item[1] is None else item[1]),
    ):
        utility_mean, utility_ci = _hierarchical_mean_interval(
            utility_groups[(condition, epsilon)], config=config
        )
        identity_mean, identity_ci = _hierarchical_mean_interval(
            identity_groups[(condition, epsilon)], config=config
        )
        open_mean, open_ci = _hierarchical_mean_interval(
            open_set_groups[(condition, epsilon)], config=config
        )
        repeated_entries = []
        for group_size, groups in repeated_groups[(condition, epsilon)].items():
            mean, interval = _hierarchical_mean_interval(groups, config=config)
            repeated_entries.append(
                {
                    "group_size": group_size,
                    "group_accuracy_mean": mean,
                    "group_accuracy_ci95": interval,
                }
            )
        entries.append(
            {
                "condition": condition,
                "epsilon": epsilon,
                "datasets": len(utility_groups[(condition, epsilon)]),
                "utility": {
                    "balanced_accuracy_mean": utility_mean,
                    "balanced_accuracy_ci95": utility_ci,
                },
                "identity_attack": {
                    "balanced_accuracy_mean": identity_mean,
                    "balanced_accuracy_ci95": identity_ci,
                },
                "open_set_identity_attack": {
                    "auroc_mean": open_mean,
                    "auroc_ci95": open_ci,
                },
                "repeated_release_identity_attack": {
                    "group_sizes": sorted(
                        repeated_entries, key=lambda item: _group_size_key(item["group_size"])
                    )
                },
                "paired_deltas": {
                    "vs_clean": _paired_delta_summary(
                        utility_means=utility_means,
                        identity_means=identity_means,
                        datasets=dataset_names,
                        condition=condition,
                        epsilon=epsilon,
                        baseline="clean",
                        config=config,
                    ),
                    "vs_clipped": _paired_delta_summary(
                        utility_means=utility_means,
                        identity_means=identity_means,
                        datasets=dataset_names,
                        condition=condition,
                        epsilon=epsilon,
                        baseline="clipped",
                        config=config,
                    ),
                },
            }
        )
    return {
        "method": "dataset_subject_seed_hierarchical_bootstrap",
        "entries": entries,
    }


def _strongest_metric_groups(
    scores: tuple[Any, ...],
    *,
    metric: str,
) -> tuple[
    dict[tuple[str, float | None], list[list[float]]], dict[tuple[str, str, float | None], float]
]:
    grouped: dict[tuple[str, str, float | None], dict[tuple[str, str], list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for score in scores:
        grouped[(score.dataset, score.condition, score.epsilon)][
            (score.regime, score.attacker)
        ].append(float(getattr(score, metric)))

    aggregate: dict[tuple[str, float | None], list[list[float]]] = defaultdict(list)
    means: dict[tuple[str, str, float | None], float] = {}
    for key, candidates in grouped.items():
        values = max(candidates.values(), key=lambda item: float(np.mean(item)))
        means[key] = float(np.mean(values))
        aggregate[(key[1], key[2])].append(values)
    return aggregate, means


def _strongest_repeated_groups(
    result: BenchmarkResult,
) -> tuple[dict[tuple[str, float | None], dict[str, list[list[float]]]], dict[str, float]]:
    grouped: dict[tuple[str, str, float | None, str], dict[tuple[str, str], list[float]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    for score in result.repeated_release_scores:
        grouped[(score.dataset, score.condition, score.epsilon, score.group_size)][
            (score.regime, score.attacker)
        ].append(float(score.group_accuracy))

    aggregate: dict[tuple[str, float | None], dict[str, list[list[float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for key, candidates in grouped.items():
        values = max(candidates.values(), key=lambda item: float(np.mean(item)))
        aggregate[(key[1], key[2])][key[3]].append(values)
    return aggregate, {}


def build_summary(config: ExperimentConfig, result: BenchmarkResult) -> dict[str, Any]:
    """Aggregate utility by subject and identity leakage by strongest attack."""
    utility = _utility_summary(result, config)
    attacks = _attack_summary(result, config)
    open_set = _open_set_summary(result, config)
    repeated_release = _repeated_release_summary(result, config)
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
                "repeated_release_identity_attack": repeated_release[key],
                "privacy_composition_basic": _basic_composition(
                    repeated_release[key],
                    epsilon=epsilon,
                    delta=config.privacy.delta,
                ),
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
        "hierarchical_summary": _hierarchical_scientific_summary(config, result),
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


def _plot_hierarchical_ci(summary: dict[str, Any], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    entries = summary["hierarchical_summary"]["entries"]
    fig, axis = plt.subplots(figsize=(7.2, 5.0))
    for entry in entries:
        utility = entry["utility"]["balanced_accuracy_mean"]
        leakage = entry["identity_attack"]["balanced_accuracy_mean"]
        utility_ci = entry["utility"]["balanced_accuracy_ci95"]
        leakage_ci = entry["identity_attack"]["balanced_accuracy_ci95"]
        xerr = (
            None if leakage_ci is None else [[leakage - leakage_ci[0]], [leakage_ci[1] - leakage]]
        )
        yerr = (
            None if utility_ci is None else [[utility - utility_ci[0]], [utility_ci[1] - utility]]
        )
        label = (
            entry["condition"]
            if entry["epsilon"] is None
            else f"{entry['condition']}, eps={entry['epsilon']:g}"
        )
        axis.errorbar(leakage, utility, xerr=xerr, yerr=yerr, fmt="o", capsize=3, label=label)
    axis.set_xlabel("Hierarchical strongest identity balanced accuracy")
    axis.set_ylabel("Hierarchical motor-task balanced accuracy")
    axis.set_title("Aggregate privacy-utility with hierarchical CI")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_repeated_release(summary: dict[str, Any], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    positions = {"1": 1, "4": 4, "16": 16, "64": 64, "all": 128}
    fig, axis = plt.subplots(figsize=(7.2, 5.0))
    for entry in summary["hierarchical_summary"]["entries"]:
        group_entries = entry["repeated_release_identity_attack"]["group_sizes"]
        x = [positions[item["group_size"]] for item in group_entries]
        y = [item["group_accuracy_mean"] for item in group_entries]
        label = (
            entry["condition"]
            if entry["epsilon"] is None
            else f"{entry['condition']}, eps={entry['epsilon']:g}"
        )
        axis.plot(x, y, marker="o", linewidth=1.2, label=label)
    axis.set_xscale("log", base=2)
    axis.set_xticks(list(positions.values()), labels=list(positions))
    axis.set_xlabel("Repeated releases grouped per subject")
    axis.set_ylabel("Strongest group identification accuracy")
    axis.set_title("Repeated-release identity leakage")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_open_set(summary: dict[str, Any], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    entries = summary["hierarchical_summary"]["entries"]
    labels = [
        entry["condition"]
        if entry["epsilon"] is None
        else f"{entry['condition']}\neps={entry['epsilon']:g}"
        for entry in entries
    ]
    values = [entry["open_set_identity_attack"]["auroc_mean"] for entry in entries]
    fig, axis = plt.subplots(figsize=(max(7.2, 0.55 * len(entries)), 5.0))
    axis.bar(range(len(entries)), values, color="#4c78a8")
    axis.axhline(0.5, color="black", linestyle="--", linewidth=1)
    axis.set_xticks(range(len(entries)), labels=labels, rotation=45, ha="right")
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Strongest open-set AUROC")
    axis.set_title("Open-set identity recognition")
    axis.grid(axis="y", alpha=0.25)
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
    _plot_tradeoff(summary, plots_dir / "privacy-utility-by-dataset.png")
    _plot_hierarchical_ci(summary, plots_dir / "hierarchical-ci.png")
    _plot_repeated_release(summary, plots_dir / "repeated-release-leakage.png")
    _plot_open_set(summary, plots_dir / "open-set-auroc.png")
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
        hierarchical = summary.get("hierarchical_summary")
        if not isinstance(hierarchical, dict) or not isinstance(hierarchical.get("entries"), list):
            raise ValueError("summary.json contains invalid hierarchical summary")
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
        repeated_release_identity_attack = entry.get("repeated_release_identity_attack")
        composition = entry.get("privacy_composition_basic")
        if not isinstance(utility, dict) or "balanced_accuracy_mean" not in utility:
            raise ValueError("summary.json contains an invalid utility summary")
        if not isinstance(identity_attack, dict):
            raise ValueError("summary.json contains an invalid identity attack summary")
        if schema_version == 2 and not isinstance(open_set_identity_attack, dict):
            raise ValueError("summary.json contains an invalid open-set identity attack summary")
        if schema_version == 2 and not isinstance(repeated_release_identity_attack, dict):
            raise ValueError("summary.json contains an invalid repeated-release summary")
        if schema_version == 2 and not isinstance(composition, list):
            raise ValueError("summary.json contains an invalid basic composition table")
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
        if scope not in {
            "utility",
            "identity_attack",
            "open_set_identity_attack",
            "repeated_release_identity_attack",
        }:
            raise ValueError("metrics.csv contains an invalid scope")
        if not _nonempty_string(row.get("condition")):
            raise ValueError("metrics.csv contains an invalid condition")
        if scope in {"utility", "identity_attack"} and (
            not _finite_metric(row, "balanced_accuracy") or not _finite_metric(row, "macro_f1")
        ):
            raise ValueError("metrics.csv contains invalid performance metrics")
        if scope == "utility" and not _nonempty_string(row.get("subject")):
            raise ValueError("metrics.csv contains a utility row without a subject")
        if scope in {
            "identity_attack",
            "open_set_identity_attack",
            "repeated_release_identity_attack",
        } and (not _nonempty_string(row.get("regime")) or not _nonempty_string(row.get("model"))):
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
        if scope == "repeated_release_identity_attack":
            if not _nonempty_string(row.get("group_size")) or not _finite_metric(
                row, "group_accuracy"
            ):
                raise ValueError("metrics.csv contains invalid repeated-release metrics")
            try:
                groups = int(row["groups"])
                max_releases = int(row["max_releases"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("metrics.csv contains invalid repeated-release counts") from error
            if groups <= 0 or max_releases <= 0:
                raise ValueError("metrics.csv contains invalid repeated-release counts")
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
    if manifest_version == 2:
        for plot_name in (
            "privacy-utility-by-dataset.png",
            "hierarchical-ci.png",
            "repeated-release-leakage.png",
            "open-set-auroc.png",
        ):
            plot_path = run_dir / "plots" / plot_name
            if not plot_path.is_file() or plot_path.stat().st_size == 0:
                raise ValueError(f"run contains no {plot_name} plot")
