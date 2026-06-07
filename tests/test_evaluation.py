from __future__ import annotations

from collections import Counter

import numpy as np

from nepra.evaluation import BenchmarkResult, bootstrap_mean_interval


def test_benchmark_trains_personalized_tasks_and_pooled_attackers(
    benchmark_result: BenchmarkResult,
) -> None:
    task_counts = Counter(score.condition for score in benchmark_result.task_scores)
    attack_counts = Counter(score.condition for score in benchmark_result.attack_scores)

    assert task_counts == {
        "clean": 4,
        "clipped": 4,
        "analytic_gaussian": 8,
        "task_weighted_empirical": 8,
    }
    assert attack_counts == {
        "clean": 8,
        "clipped": 8,
        "analytic_gaussian": 16,
        "task_weighted_empirical": 16,
    }
    assert {score.subject for score in benchmark_result.task_scores} == {
        "1",
        "2",
        "3",
        "4",
    }
    assert {score.regime for score in benchmark_result.attack_scores} == {
        "clean_auxiliary",
        "mechanism_aware",
    }
    assert {score.attacker for score in benchmark_result.attack_scores} == {
        "logistic_regression",
        "rbf_svm",
        "random_forest",
        "mlp",
    }


def test_bootstrap_interval_is_deterministic() -> None:
    values = np.asarray([0.5, 0.6, 0.7, 0.8])
    first = bootstrap_mean_interval(values, samples=200, seed=42)
    second = bootstrap_mean_interval(values, samples=200, seed=42)

    assert first == second
    assert first[0] <= np.mean(values) <= first[1]
