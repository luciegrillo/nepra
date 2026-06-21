from __future__ import annotations

import numpy as np
import pytest

from nepra.config import ExperimentConfig
from nepra.data import (
    CALIBRATION_SESSION,
    HELDOUT_SESSION,
    MOABB_DATASET_REGISTRY,
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


def test_first_last_session_split_uses_natural_session_order() -> None:
    raw = EEGDataset(
        epochs=np.arange(12 * 2 * 3, dtype=np.float64).reshape(12, 2, 3),
        task_labels=np.asarray(["left", "right"] * 6),
        subject_labels=np.asarray(["1"] * 6 + ["2"] * 6),
        session_labels=np.asarray(
            [
                "session_1",
                "session_2",
                "session_10",
                "session_1",
                "session_2",
                "session_10",
                "session_1",
                "session_2",
                "session_10",
                "session_1",
                "session_2",
                "session_10",
            ]
        ),
        sample_rate=64,
    )

    split = raw.with_session_split("first_last")

    assert split.epochs.shape[0] == 8
    assert set(split.session_labels) == {CALIBRATION_SESSION, HELDOUT_SESSION}
    assert np.array_equal(
        split.session_labels,
        np.asarray(
            [
                CALIBRATION_SESSION,
                HELDOUT_SESSION,
                CALIBRATION_SESSION,
                HELDOUT_SESSION,
                CALIBRATION_SESSION,
                HELDOUT_SESSION,
                CALIBRATION_SESSION,
                HELDOUT_SESSION,
            ]
        ),
    )


def test_first_last_session_split_requires_two_sessions() -> None:
    raw = EEGDataset(
        epochs=np.zeros((2, 2, 3)),
        task_labels=np.asarray(["left", "right"]),
        subject_labels=np.asarray(["1", "1"]),
        session_labels=np.asarray(["only", "only"]),
        sample_rate=64,
    )

    with pytest.raises(ValueError, match="at least two sessions"):
        raw.with_session_split("first_last")


def test_first_last_session_split_does_not_truncate_short_raw_labels() -> None:
    raw = EEGDataset(
        epochs=np.zeros((4, 2, 3)),
        task_labels=np.asarray(["left", "right", "left", "right"]),
        subject_labels=np.asarray(["1", "1", "2", "2"]),
        session_labels=np.asarray(["0", "1", "0", "1"]),
        sample_rate=64,
    )

    split = raw.with_session_split("first_last")

    assert set(split.session_labels) == {CALIBRATION_SESSION, HELDOUT_SESSION}


def test_v0_2_core_datasets_are_registered(v0_2_core_config: ExperimentConfig) -> None:
    configured = {dataset.name for dataset in v0_2_core_config.datasets}

    assert configured <= set(MOABB_DATASET_REGISTRY)
