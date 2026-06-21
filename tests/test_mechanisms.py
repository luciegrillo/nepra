from __future__ import annotations

import numpy as np
import pytest
from diffprivlib.mechanisms import GaussianAnalytic

from nepra.config import ExperimentConfig
from nepra.geometry import FeatureDataset
from nepra.mechanisms import (
    AnalyticGaussianRandomizer,
    L2Clipper,
    TaskAwareEmpiricalRandomizer,
    fit_task_weights,
)


def test_clipping_enforces_replace_one_bound(
    feature_datasets: tuple[FeatureDataset, FeatureDataset],
) -> None:
    calibration, heldout = feature_datasets
    clipper = L2Clipper(95).fit(calibration)
    clipped_calibration = clipper.transform(calibration).dataset
    clipped_heldout = clipper.transform(heldout).dataset
    all_features = np.vstack([clipped_calibration.features, clipped_heldout.features])

    assert np.max(np.linalg.norm(all_features, axis=1)) <= clipper.clipping_norm_ + 1e-12
    differences = all_features[:32, None, :] - all_features[None, -32:, :]
    assert np.max(np.linalg.norm(differences, axis=2)) <= clipper.replace_one_sensitivity


def test_clipper_rejects_heldout_calibration(
    feature_datasets: tuple[FeatureDataset, FeatureDataset],
) -> None:
    _, heldout = feature_datasets
    with pytest.raises(ValueError, match="public session"):
        L2Clipper(95).fit(heldout)


def test_gaussian_scale_agrees_with_diffprivlib(
    feature_datasets: tuple[FeatureDataset, FeatureDataset],
    smoke_config: ExperimentConfig,
) -> None:
    calibration, _ = feature_datasets
    clipper = L2Clipper(95).fit(calibration)
    randomizer = AnalyticGaussianRandomizer(
        epsilon=1.0,
        delta=smoke_config.privacy.delta,
        clipping_norm=clipper.clipping_norm_,
        seed=42,
    )
    reference = GaussianAnalytic(
        epsilon=1.0,
        delta=smoke_config.privacy.delta,
        sensitivity=clipper.replace_one_sensitivity,
        random_state=42,
    )

    assert randomizer.sensitivity == clipper.replace_one_sensitivity
    assert randomizer.standard_deviation == float(reference._scale)


def test_gaussian_scale_compatibility_failure_is_explicit() -> None:
    randomizer = AnalyticGaussianRandomizer(
        epsilon=1.0,
        delta=1e-5,
        clipping_norm=2.0,
        seed=42,
    )
    randomizer._mechanism._scale = None

    with pytest.raises(RuntimeError, match="compatibility layer"):
        _ = randomizer.standard_deviation


def test_secure_gaussian_mode_rejects_seed() -> None:
    with pytest.raises(ValueError, match="does not accept"):
        AnalyticGaussianRandomizer(
            epsilon=1.0,
            delta=1e-5,
            clipping_norm=2.0,
            seed=42,
            secure=True,
        )


def test_task_weights_are_positive_and_energy_matched(
    feature_datasets: tuple[FeatureDataset, FeatureDataset],
    smoke_config: ExperimentConfig,
) -> None:
    calibration, _ = feature_datasets
    weights = fit_task_weights(
        calibration,
        task_c=smoke_config.models.task_c,
        weight_floor=smoke_config.privacy.weight_floor,
        weight_exponent=smoke_config.privacy.weight_exponent,
        max_iter=smoke_config.models.max_iter,
        seed=42,
    ).weights
    sigma = 3.5

    assert np.all(weights > 0)
    assert np.isclose(np.mean(np.square(weights)), 1.0)
    assert np.isclose(np.sum(np.square(sigma * weights)), weights.size * sigma**2)


def test_empirical_randomizer_requires_clipped_input(
    feature_datasets: tuple[FeatureDataset, FeatureDataset],
) -> None:
    calibration, _ = feature_datasets
    randomizer = TaskAwareEmpiricalRandomizer(
        standard_deviation=1.0,
        clipping_norm=0.1,
        weights=np.ones(calibration.features.shape[1]),
        seed=42,
    )
    with pytest.raises(ValueError, match="clipped"):
        randomizer.transform(calibration)
