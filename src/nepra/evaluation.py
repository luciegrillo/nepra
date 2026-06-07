"""Strict cross-session privacy-utility evaluation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

from nepra.config import ExperimentConfig
from nepra.data import EEGDataset
from nepra.geometry import FeatureDataset, RiemannianRepresentation
from nepra.mechanisms import (
    AnalyticGaussianRandomizer,
    ClippingDiagnostics,
    L2Clipper,
    TaskAwareEmpiricalRandomizer,
    TaskWeightModel,
    fit_task_weights,
)
from nepra.models import build_identity_attackers, build_task_decoder

FloatArray = NDArray[np.float64]
LabelArray = NDArray[np.str_]


@dataclass(frozen=True)
class Condition:
    """One train/test representation pair evaluated by the benchmark."""

    name: str
    epsilon: float | None
    seed: int | None
    calibration: FeatureDataset
    heldout: FeatureDataset


@dataclass(frozen=True)
class TaskScore:
    condition: str
    epsilon: float | None
    seed: int | None
    subject: str
    balanced_accuracy: float
    macro_f1: float


@dataclass(frozen=True)
class AttackScore:
    condition: str
    epsilon: float | None
    seed: int | None
    regime: str
    attacker: str
    balanced_accuracy: float
    macro_f1: float
    advantage_over_chance: float
    session_accuracy: float


@dataclass(frozen=True)
class BenchmarkResult:
    task_scores: tuple[TaskScore, ...]
    attack_scores: tuple[AttackScore, ...]
    calibration_clipping: ClippingDiagnostics
    heldout_clipping: ClippingDiagnostics
    task_weights: TaskWeightModel
    feature_dimensions: int
    calibration_trials: int
    heldout_trials: int
    elapsed_seconds: float


@dataclass(frozen=True)
class FittedAttacker:
    model: Any
    label_encoder: LabelEncoder

    def predict(self, features: FloatArray) -> LabelArray:
        encoded = np.asarray(self.model.predict(features), dtype=np.int64)
        return np.asarray(self.label_encoder.inverse_transform(encoded), dtype=str)


def _derived_seed(seed: int, stream: int) -> int:
    return int(np.random.SeedSequence([seed, stream]).generate_state(1)[0])


def _session_accuracy(y_true: LabelArray, y_pred: LabelArray) -> float:
    correct = 0
    subjects = np.unique(y_true)
    for subject in subjects:
        predictions = y_pred[y_true == subject]
        values, counts = np.unique(predictions, return_counts=True)
        majority = values[np.argmax(counts)]
        correct += int(majority == subject)
    return correct / len(subjects)


def _evaluate_task(
    condition: Condition,
    config: ExperimentConfig,
    model_seed: int,
) -> list[TaskScore]:
    scores: list[TaskScore] = []
    train_subjects = set(condition.calibration.subject_labels)
    test_subjects = set(condition.heldout.subject_labels)
    if train_subjects != test_subjects:
        raise ValueError("task evaluation requires equal subjects in both sessions")

    for subject in sorted(train_subjects):
        train = condition.calibration.select_subject(subject)
        test = condition.heldout.select_subject(subject)
        decoder = build_task_decoder(config.models, seed=model_seed)
        decoder.fit(train.features, train.task_labels)
        predictions = decoder.predict(test.features)
        scores.append(
            TaskScore(
                condition=condition.name,
                epsilon=condition.epsilon,
                seed=condition.seed,
                subject=str(subject),
                balanced_accuracy=float(balanced_accuracy_score(test.task_labels, predictions)),
                macro_f1=float(f1_score(test.task_labels, predictions, average="macro")),
            )
        )
    return scores


def _fit_attackers(
    calibration: FeatureDataset,
    config: ExperimentConfig,
    model_seed: int,
) -> dict[str, FittedAttacker]:
    fitted: dict[str, FittedAttacker] = {}
    encoder = LabelEncoder().fit(calibration.subject_labels)
    encoded_labels = encoder.transform(calibration.subject_labels)
    for name, attacker in build_identity_attackers(
        config.models, model_seed, config.evaluation.n_jobs
    ).items():
        model = clone(attacker)
        model.fit(calibration.features, encoded_labels)
        fitted[name] = FittedAttacker(model=model, label_encoder=encoder)
    return fitted


def _score_attackers(
    *,
    attackers: dict[str, FittedAttacker],
    heldout: FeatureDataset,
    condition: Condition,
    regime: str,
) -> list[AttackScore]:
    chance = 1.0 / len(np.unique(heldout.subject_labels))
    scores: list[AttackScore] = []
    for name, attacker in attackers.items():
        predictions = attacker.predict(heldout.features)
        balanced_accuracy = float(balanced_accuracy_score(heldout.subject_labels, predictions))
        scores.append(
            AttackScore(
                condition=condition.name,
                epsilon=condition.epsilon,
                seed=condition.seed,
                regime=regime,
                attacker=name,
                balanced_accuracy=balanced_accuracy,
                macro_f1=float(f1_score(heldout.subject_labels, predictions, average="macro")),
                advantage_over_chance=balanced_accuracy - chance,
                session_accuracy=float(_session_accuracy(heldout.subject_labels, predictions)),
            )
        )
    return scores


def _conditions(
    *,
    clean_calibration: FeatureDataset,
    clean_heldout: FeatureDataset,
    clipped_calibration: FeatureDataset,
    clipped_heldout: FeatureDataset,
    clipping_norm: float,
    task_weights: TaskWeightModel,
    config: ExperimentConfig,
) -> list[Condition]:
    conditions = [
        Condition(
            name="clean",
            epsilon=None,
            seed=None,
            calibration=clean_calibration,
            heldout=clean_heldout,
        ),
        Condition(
            name="clipped",
            epsilon=None,
            seed=None,
            calibration=clipped_calibration,
            heldout=clipped_heldout,
        ),
    ]

    for epsilon in config.privacy.epsilons:
        for seed in config.privacy.seeds:
            gaussian_train = AnalyticGaussianRandomizer(
                epsilon=epsilon,
                delta=config.privacy.delta,
                clipping_norm=clipping_norm,
                seed=_derived_seed(seed, 10),
            )
            gaussian_test = AnalyticGaussianRandomizer(
                epsilon=epsilon,
                delta=config.privacy.delta,
                clipping_norm=clipping_norm,
                seed=_derived_seed(seed, 11),
            )
            conditions.append(
                Condition(
                    name="analytic_gaussian",
                    epsilon=epsilon,
                    seed=seed,
                    calibration=gaussian_train.transform(clipped_calibration),
                    heldout=gaussian_test.transform(clipped_heldout),
                )
            )

            weighted_train = TaskAwareEmpiricalRandomizer(
                standard_deviation=gaussian_train.standard_deviation,
                clipping_norm=clipping_norm,
                weights=task_weights.weights,
                seed=_derived_seed(seed, 20),
            )
            weighted_test = TaskAwareEmpiricalRandomizer(
                standard_deviation=gaussian_test.standard_deviation,
                clipping_norm=clipping_norm,
                weights=task_weights.weights,
                seed=_derived_seed(seed, 21),
            )
            conditions.append(
                Condition(
                    name="task_weighted_empirical",
                    epsilon=epsilon,
                    seed=seed,
                    calibration=weighted_train.transform(clipped_calibration),
                    heldout=weighted_test.transform(clipped_heldout),
                )
            )
    return conditions


def bootstrap_mean_interval(
    values: FloatArray | list[float],
    *,
    samples: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return a percentile bootstrap interval for a mean."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not array.size:
        raise ValueError("bootstrap values must be a non-empty vector")
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(samples, array.size))
    means = np.mean(array[indices], axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return float(low), float(high)


def run_benchmark(config: ExperimentConfig, dataset: EEGDataset) -> BenchmarkResult:
    """Run the complete strict cross-session benchmark in memory."""
    started = time.perf_counter()
    calibration, heldout = dataset.split_calibration_heldout()

    representation = RiemannianRepresentation()
    clean_calibration = representation.fit_transform(calibration)
    clean_heldout = representation.transform(heldout)

    clipper = L2Clipper(config.privacy.clipping_percentile).fit(clean_calibration)
    clipped_calibration_result = clipper.transform(clean_calibration)
    clipped_heldout_result = clipper.transform(clean_heldout)

    model_seed = config.privacy.seeds[0]
    task_weights = fit_task_weights(
        clean_calibration,
        task_c=config.models.task_c,
        weight_floor=config.privacy.weight_floor,
        weight_exponent=config.privacy.weight_exponent,
        max_iter=config.models.max_iter,
        seed=model_seed,
    )
    conditions = _conditions(
        clean_calibration=clean_calibration,
        clean_heldout=clean_heldout,
        clipped_calibration=clipped_calibration_result.dataset,
        clipped_heldout=clipped_heldout_result.dataset,
        clipping_norm=clipper.clipping_norm_,
        task_weights=task_weights,
        config=config,
    )

    clean_attackers = _fit_attackers(clean_calibration, config, model_seed)
    task_scores: list[TaskScore] = []
    attack_scores: list[AttackScore] = []
    for condition in conditions:
        task_scores.extend(_evaluate_task(condition, config, model_seed))
        attack_scores.extend(
            _score_attackers(
                attackers=clean_attackers,
                heldout=condition.heldout,
                condition=condition,
                regime="clean_auxiliary",
            )
        )
        mechanism_aware = (
            clean_attackers
            if condition.name == "clean"
            else _fit_attackers(condition.calibration, config, model_seed)
        )
        attack_scores.extend(
            _score_attackers(
                attackers=mechanism_aware,
                heldout=condition.heldout,
                condition=condition,
                regime="mechanism_aware",
            )
        )

    return BenchmarkResult(
        task_scores=tuple(task_scores),
        attack_scores=tuple(attack_scores),
        calibration_clipping=clipped_calibration_result.diagnostics,
        heldout_clipping=clipped_heldout_result.diagnostics,
        task_weights=task_weights,
        feature_dimensions=clean_calibration.features.shape[1],
        calibration_trials=clean_calibration.features.shape[0],
        heldout_trials=clean_heldout.features.shape[0],
        elapsed_seconds=time.perf_counter() - started,
    )
