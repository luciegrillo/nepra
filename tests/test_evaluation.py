from __future__ import annotations

from collections import Counter
from dataclasses import replace

import numpy as np
from sklearn.dummy import DummyClassifier

from nepra import evaluation
from nepra.config import ExperimentConfig
from nepra.data import EEGDataset
from nepra.evaluation import BenchmarkResult, bootstrap_mean_interval, run_benchmark


def test_benchmark_trains_personalized_tasks_and_pooled_attackers(
    benchmark_result: BenchmarkResult,
) -> None:
    task_counts = Counter(score.condition for score in benchmark_result.task_scores)
    attack_counts = Counter(score.condition for score in benchmark_result.attack_scores)
    open_set_counts = Counter(score.condition for score in benchmark_result.open_set_scores)
    repeated_counts = Counter(score.condition for score in benchmark_result.repeated_release_scores)

    assert task_counts == {
        "clean": 4,
        "clipped": 4,
        "analytic_gaussian": 8,
        "task_weighted_empirical": 8,
    }
    assert attack_counts == {
        "clean": 14,
        "clipped": 14,
        "analytic_gaussian": 28,
        "task_weighted_empirical": 28,
    }
    assert open_set_counts == attack_counts
    assert repeated_counts == {condition: count * 5 for condition, count in attack_counts.items()}
    assert {score.subject for score in benchmark_result.task_scores} == {
        "1",
        "2",
        "3",
        "4",
    }
    assert {score.dataset for score in benchmark_result.task_scores} == {"synthetic"}
    assert {score.dataset for score in benchmark_result.attack_scores} == {"synthetic"}
    assert {score.dataset for score in benchmark_result.open_set_scores} == {"synthetic"}
    assert {score.dataset for score in benchmark_result.repeated_release_scores} == {"synthetic"}
    assert {score.regime for score in benchmark_result.attack_scores} == {
        "clean_auxiliary",
        "mechanism_aware",
    }
    assert {score.attacker for score in benchmark_result.attack_scores} == {
        "logistic_regression",
        "rbf_svm",
        "knn",
        "random_forest",
        "extra_trees",
        "hist_gradient_boosting",
        "mlp",
    }
    assert {score.attacker for score in benchmark_result.open_set_scores} == {
        "logistic_regression",
        "rbf_svm",
        "knn",
        "random_forest",
        "extra_trees",
        "hist_gradient_boosting",
        "mlp",
    }
    assert all(0.0 <= score.auroc <= 1.0 for score in benchmark_result.open_set_scores)
    assert {score.enrolled_subjects for score in benchmark_result.open_set_scores} == {3}
    assert {score.unknown_subjects for score in benchmark_result.open_set_scores} == {1}
    assert {score.group_size for score in benchmark_result.repeated_release_scores} == {
        "1",
        "4",
        "16",
        "64",
        "all",
    }
    assert all(
        0.0 <= score.group_accuracy <= 1.0 for score in benchmark_result.repeated_release_scores
    )
    assert all(score.groups > 0 for score in benchmark_result.repeated_release_scores)
    assert all(score.max_releases > 0 for score in benchmark_result.repeated_release_scores)


def test_benchmark_aggregates_configured_dataset_mapping(
    smoke_config: ExperimentConfig,
    synthetic_dataset: EEGDataset,
    monkeypatch,
) -> None:
    def dummy_attackers(*args, **kwargs) -> dict[str, DummyClassifier]:
        return {"dummy": DummyClassifier(strategy="most_frequent")}

    monkeypatch.setattr(evaluation, "build_identity_attackers", dummy_attackers)
    second_dataset_config = replace(smoke_config.dataset, name="BNCI2014_004")
    config = replace(
        smoke_config,
        datasets=(smoke_config.dataset, second_dataset_config),
    )

    result = run_benchmark(
        config,
        {
            "synthetic": synthetic_dataset,
            "BNCI2014_004": synthetic_dataset,
        },
    )

    assert [dataset.dataset for dataset in result.dataset_results] == [
        "synthetic",
        "BNCI2014_004",
    ]
    assert {score.dataset for score in result.task_scores} == {
        "synthetic",
        "BNCI2014_004",
    }
    assert {score.dataset for score in result.attack_scores} == {
        "synthetic",
        "BNCI2014_004",
    }
    assert {score.dataset for score in result.open_set_scores} == {
        "synthetic",
        "BNCI2014_004",
    }
    assert {score.dataset for score in result.repeated_release_scores} == {
        "synthetic",
        "BNCI2014_004",
    }


def test_bootstrap_interval_is_deterministic() -> None:
    values = np.asarray([0.5, 0.6, 0.7, 0.8])
    first = bootstrap_mean_interval(values, samples=200, seed=42)
    second = bootstrap_mean_interval(values, samples=200, seed=42)

    assert first == second
    assert first[0] <= np.mean(values) <= first[1]
