from __future__ import annotations

import numpy as np
import pytest

from nepra.data import EEGDataset
from nepra.geometry import FeatureDataset, RiemannianRepresentation


def test_representation_is_standardized_on_calibration(
    feature_datasets: tuple[FeatureDataset, FeatureDataset],
) -> None:
    calibration, heldout = feature_datasets

    assert calibration.features.shape == heldout.features.shape == (128, 36)
    assert np.allclose(calibration.features.mean(axis=0), 0.0, atol=1e-10)
    assert np.allclose(calibration.features.std(axis=0), 1.0, atol=1e-10)


def test_representation_rejects_heldout_data_during_fit(
    synthetic_dataset: EEGDataset,
) -> None:
    with pytest.raises(ValueError, match="requires only"):
        RiemannianRepresentation().fit(synthetic_dataset)


def test_transform_requires_fit(synthetic_dataset: EEGDataset) -> None:
    calibration, _ = synthetic_dataset.split_calibration_heldout()
    with pytest.raises(RuntimeError, match="fitted"):
        RiemannianRepresentation().transform(calibration)
