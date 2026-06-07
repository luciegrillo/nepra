"""Leakage-resistant Riemannian EEG representation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nepra.data import CALIBRATION_SESSION, EEGDataset

FloatArray = NDArray[np.float64]
LabelArray = NDArray[np.str_]


@dataclass(frozen=True)
class FeatureDataset:
    """Tangent features with labels retained for grouped evaluation."""

    features: FloatArray
    task_labels: LabelArray
    subject_labels: LabelArray
    session_labels: LabelArray

    def __post_init__(self) -> None:
        n_trials = self.features.shape[0]
        if self.features.ndim != 2:
            raise ValueError("features must have shape (trials, dimensions)")
        if any(
            len(labels) != n_trials
            for labels in (self.task_labels, self.subject_labels, self.session_labels)
        ):
            raise ValueError("feature labels must align with rows")
        if not np.isfinite(self.features).all():
            raise ValueError("features must contain only finite values")

    def select_subject(self, subject: str) -> FeatureDataset:
        """Select one subject while preserving aligned labels."""
        mask = self.subject_labels == str(subject)
        if not mask.any():
            raise ValueError(f"subject {subject!r} is absent")
        return FeatureDataset(
            features=self.features[mask],
            task_labels=self.task_labels[mask],
            subject_labels=self.subject_labels[mask],
            session_labels=self.session_labels[mask],
        )

    def with_features(self, features: FloatArray) -> FeatureDataset:
        """Replace features without altering labels."""
        return FeatureDataset(
            features=np.asarray(features, dtype=np.float64),
            task_labels=self.task_labels.copy(),
            subject_labels=self.subject_labels.copy(),
            session_labels=self.session_labels.copy(),
        )


class RiemannianRepresentation:
    """OAS covariance, Riemannian tangent projection, and standardization."""

    def __init__(self) -> None:
        self.geometry = Pipeline(
            [
                ("covariance", Covariances(estimator="oas")),
                ("tangent_space", TangentSpace(metric="riemann")),
            ]
        )
        self.scaler = StandardScaler()

    def fit(self, calibration: EEGDataset) -> RiemannianRepresentation:
        """Fit every data-dependent transform on public calibration data."""
        sessions = set(calibration.session_labels)
        if sessions != {CALIBRATION_SESSION}:
            raise ValueError(
                "representation fitting requires only the public calibration "
                f"session {CALIBRATION_SESSION!r}; received {sorted(sessions)}"
            )
        tangent = self.geometry.fit_transform(calibration.epochs)
        self.scaler.fit(tangent)
        self.n_features_in_ = tangent.shape[1]
        self.calibration_trials_ = tangent.shape[0]
        return self

    def transform(self, dataset: EEGDataset) -> FeatureDataset:
        """Transform epochs using frozen calibration-session parameters."""
        if not hasattr(self, "n_features_in_"):
            raise RuntimeError("representation must be fitted before transform")
        tangent = self.geometry.transform(dataset.epochs)
        features = self.scaler.transform(tangent)
        return FeatureDataset(
            features=np.asarray(features, dtype=np.float64),
            task_labels=dataset.task_labels.copy(),
            subject_labels=dataset.subject_labels.copy(),
            session_labels=dataset.session_labels.copy(),
        )

    def fit_transform(self, calibration: EEGDataset) -> FeatureDataset:
        """Fit on and transform the public calibration session."""
        return self.fit(calibration).transform(calibration)
