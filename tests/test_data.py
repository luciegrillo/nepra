from __future__ import annotations

import numpy as np
import pytest

from nepra.config import ExperimentConfig
from nepra.data import (
    CALIBRATION_SESSION,
    HELDOUT_SESSION,
    EEGDataset,
    download_dataset,
    generate_synthetic_eeg,
)


def test_synthetic_data_is_deterministic_and_balanced(
    smoke_config: ExperimentConfig,
) -> None:
    first = generate_synthetic_eeg(smoke_config.dataset)
    second = generate_synthetic_eeg(smoke_config.dataset)

    assert np.array_equal(first.epochs, second.epochs)
    assert first.epochs.shape == (256, 8, 128)
    assert set(first.session_labels) == {CALIBRATION_SESSION, HELDOUT_SESSION}
    calibration, heldout = first.split_calibration_heldout()
    assert calibration.epochs.shape == heldout.epochs.shape == (128, 8, 128)


def test_dataset_rejects_misaligned_labels() -> None:
    with pytest.raises(ValueError, match="align"):
        EEGDataset(
            epochs=np.zeros((2, 3, 4)),
            task_labels=np.asarray(["left"]),
            subject_labels=np.asarray(["1", "1"]),
            session_labels=np.asarray(["0train", "0train"]),
            sample_rate=64,
        )


def test_synthetic_download_is_rejected(smoke_config: ExperimentConfig) -> None:
    with pytest.raises(ValueError, match="nothing to download"):
        download_dataset(smoke_config.dataset)
