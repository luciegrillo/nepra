"""Reproducible EEG access and deterministic synthetic data."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from nepra.config import DatasetConfig

FloatArray = NDArray[np.float64]
LabelArray = NDArray[np.str_]

CALIBRATION_SESSION = "0train"
HELDOUT_SESSION = "1test"
SUPPORTED_SESSIONS = frozenset({CALIBRATION_SESSION, HELDOUT_SESSION})


@dataclass(frozen=True)
class EEGDataset:
    """Epoched EEG and aligned task, subject, and session labels."""

    epochs: FloatArray
    task_labels: LabelArray
    subject_labels: LabelArray
    session_labels: LabelArray
    sample_rate: float

    def __post_init__(self) -> None:
        n_trials = self.epochs.shape[0]
        if self.epochs.ndim != 3:
            raise ValueError("epochs must have shape (trials, channels, samples)")
        if any(len(labels) != n_trials for labels in self.label_arrays):
            raise ValueError("all label arrays must align with epochs")
        if not np.isfinite(self.epochs).all():
            raise ValueError("epochs must contain only finite values")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")

    @property
    def label_arrays(self) -> tuple[LabelArray, LabelArray, LabelArray]:
        return self.task_labels, self.subject_labels, self.session_labels

    def select_session(self, session: str) -> EEGDataset:
        """Return an independent view containing one session."""
        mask = self.session_labels == session
        if not mask.any():
            raise ValueError(f"session {session!r} is absent")
        return EEGDataset(
            epochs=self.epochs[mask],
            task_labels=self.task_labels[mask],
            subject_labels=self.subject_labels[mask],
            session_labels=self.session_labels[mask],
            sample_rate=self.sample_rate,
        )

    def split_calibration_heldout(self) -> tuple[EEGDataset, EEGDataset]:
        """Return the public calibration and protected held-out sessions."""
        return (
            self.select_session(CALIBRATION_SESSION),
            self.select_session(HELDOUT_SESSION),
        )


def generate_synthetic_eeg(config: DatasetConfig, seed: int = 2026) -> EEGDataset:
    """Generate deterministic two-session EEG with task and identity structure."""
    rng = np.random.default_rng(seed)
    subjects = tuple(str(subject) for subject in config.subjects)
    classes = config.classes
    channels = config.channels
    samples = round((config.tmax - config.tmin) * config.resample_hz)
    time = np.arange(samples, dtype=np.float64) / config.resample_hz

    task_topographies = rng.normal(size=(len(classes), channels))
    task_topographies /= np.linalg.norm(task_topographies, axis=1, keepdims=True)
    subject_topographies = rng.normal(size=(len(subjects), channels))
    subject_topographies /= np.linalg.norm(subject_topographies, axis=1, keepdims=True)

    epochs: list[FloatArray] = []
    task_labels: list[str] = []
    subject_labels: list[str] = []
    session_labels: list[str] = []

    for subject_index, subject in enumerate(subjects):
        identity_frequency = 18.0 + subject_index
        identity_phase = rng.uniform(0, 2 * np.pi)
        for session_index, session in enumerate((CALIBRATION_SESSION, HELDOUT_SESSION)):
            drift = rng.normal(scale=0.06, size=(channels, channels))
            mixing = np.eye(channels) + session_index * drift
            for class_index, task_class in enumerate(classes):
                task_frequency = 9.0 + 1.5 * class_index
                for _ in range(config.trials_per_class):
                    task_phase = rng.uniform(0, 2 * np.pi)
                    task_wave = np.sin(2 * np.pi * task_frequency * time + task_phase)
                    identity_wave = np.sin(2 * np.pi * identity_frequency * time + identity_phase)
                    noise = rng.normal(scale=0.65, size=(channels, samples))
                    epoch = (
                        1.25 * task_topographies[class_index, :, None] * task_wave
                        + 0.8 * subject_topographies[subject_index, :, None] * identity_wave
                        + noise
                    )
                    epoch = mixing @ epoch
                    epochs.append(epoch.astype(np.float64, copy=False))
                    task_labels.append(task_class)
                    subject_labels.append(subject)
                    session_labels.append(session)

    return EEGDataset(
        epochs=np.stack(epochs),
        task_labels=np.asarray(task_labels, dtype=str),
        subject_labels=np.asarray(subject_labels, dtype=str),
        session_labels=np.asarray(session_labels, dtype=str),
        sample_rate=config.resample_hz,
    )


def _configure_bnci_cache(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    resolved = cache_dir.resolve()
    os.environ["MNE_DATASETS_BNCI_PATH"] = str(resolved)
    return resolved


def _bnci_dataset(config: DatasetConfig):
    from moabb.datasets import BNCI2014_001

    dataset = BNCI2014_001(subjects=list(config.subjects))
    available = set(dataset.subject_list)
    missing = set(config.subjects) - available
    if missing:
        raise ValueError(f"BNCI2014_001 does not contain subjects: {sorted(missing)}")
    if set(config.classes) != set(dataset.event_id):
        raise ValueError(
            f"configured classes do not match BNCI2014_001 events: {sorted(dataset.event_id)}"
        )
    return dataset


def download_dataset(config: DatasetConfig, force: bool = False) -> Path:
    """Download the configured real dataset and return its cache directory."""
    if config.name == "synthetic":
        raise ValueError("synthetic data is generated locally and has nothing to download")
    cache_dir = _configure_bnci_cache(config.cache_dir)
    dataset = _bnci_dataset(config)
    dataset.download(
        subject_list=list(config.subjects),
        path=str(cache_dir),
        force_update=force,
        update_path=False,
        verbose=False,
    )
    return cache_dir


def load_bnci2014_001(config: DatasetConfig) -> EEGDataset:
    """Load four-class BNCI2014_001 epochs through MOABB."""
    from moabb.paradigms import MotorImagery

    _configure_bnci_cache(config.cache_dir)
    dataset = _bnci_dataset(config)
    paradigm = MotorImagery(
        n_classes=len(config.classes),
        events=list(config.classes),
        resample=config.resample_hz,
        fmin=config.fmin,
        fmax=config.fmax,
        tmin=config.tmin,
        tmax=config.tmax,
    )
    epochs, task_labels, metadata = paradigm.get_data(
        dataset=dataset,
        subjects=list(config.subjects),
    )
    session_labels = metadata["session"].astype(str).to_numpy()
    observed_sessions = set(session_labels)
    if observed_sessions != SUPPORTED_SESSIONS:
        raise ValueError(
            "unexpected BNCI2014_001 sessions: "
            f"{sorted(observed_sessions)}; expected {sorted(SUPPORTED_SESSIONS)}"
        )
    observed_classes = set(np.asarray(task_labels, dtype=str))
    if observed_classes != set(config.classes):
        raise ValueError(f"loaded classes {sorted(observed_classes)} do not match configuration")
    return EEGDataset(
        epochs=np.asarray(epochs, dtype=np.float64),
        task_labels=np.asarray(task_labels, dtype=str),
        subject_labels=metadata["subject"].astype(str).to_numpy(),
        session_labels=session_labels,
        sample_rate=config.resample_hz,
    )


def load_dataset(config: DatasetConfig) -> EEGDataset:
    """Load synthetic data or the configured MOABB benchmark."""
    if config.name == "synthetic":
        return generate_synthetic_eeg(config)
    if config.name == "BNCI2014_001":
        return load_bnci2014_001(config)
    raise ValueError(f"unsupported dataset: {config.name}")
