"""Clipping and inference-time Gaussian randomization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from diffprivlib.mechanisms import GaussianAnalytic
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression

from nepra.data import CALIBRATION_SESSION
from nepra.geometry import FeatureDataset

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ClippingDiagnostics:
    """Observed norm distribution and clipping rate for one dataset."""

    clipping_norm: float
    clipped_fraction: float
    norm_mean: float
    norm_p95: float
    norm_max: float


@dataclass(frozen=True)
class ClippingResult:
    """Clipped features and their diagnostics."""

    dataset: FeatureDataset
    diagnostics: ClippingDiagnostics


def clip_l2(features: FloatArray, clipping_norm: float) -> tuple[FloatArray, FloatArray]:
    """Clip every row to an L2 ball and return original norms."""
    if clipping_norm <= 0:
        raise ValueError("clipping_norm must be positive")
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("features must be a two-dimensional matrix")
    norms = np.linalg.norm(values, axis=1)
    scales = np.maximum(1.0, norms / clipping_norm)
    return values / scales[:, None], norms


class L2Clipper:
    """Estimate an L2 clipping threshold from public calibration features."""

    def __init__(self, percentile: float = 95.0) -> None:
        if not 0 < percentile <= 100:
            raise ValueError("percentile must be in (0, 100]")
        self.percentile = float(percentile)

    def fit(self, calibration: FeatureDataset) -> L2Clipper:
        """Fit the clipping threshold on calibration features only."""
        sessions = set(calibration.session_labels)
        if sessions != {CALIBRATION_SESSION}:
            raise ValueError(
                "clipping calibration requires only the public session "
                f"{CALIBRATION_SESSION!r}; received {sorted(sessions)}"
            )
        norms = np.linalg.norm(calibration.features, axis=1)
        self.clipping_norm_ = float(np.percentile(norms, self.percentile))
        if not np.isfinite(self.clipping_norm_) or self.clipping_norm_ <= 0:
            raise ValueError("calibrated clipping norm must be finite and positive")
        return self

    @property
    def replace_one_sensitivity(self) -> float:
        """Return the replace-one L2 sensitivity bound, 2C."""
        if not hasattr(self, "clipping_norm_"):
            raise RuntimeError("clipper must be fitted before reading sensitivity")
        return 2.0 * self.clipping_norm_

    def transform(self, dataset: FeatureDataset) -> ClippingResult:
        """Clip a feature dataset using the frozen threshold."""
        if not hasattr(self, "clipping_norm_"):
            raise RuntimeError("clipper must be fitted before transform")
        clipped, norms = clip_l2(dataset.features, self.clipping_norm_)
        diagnostics = ClippingDiagnostics(
            clipping_norm=self.clipping_norm_,
            clipped_fraction=float(np.mean(norms > self.clipping_norm_)),
            norm_mean=float(np.mean(norms)),
            norm_p95=float(np.percentile(norms, 95)),
            norm_max=float(np.max(norms)),
        )
        return ClippingResult(dataset=dataset.with_features(clipped), diagnostics=diagnostics)


def _analytic_gaussian_scale(mechanism: GaussianAnalytic) -> float:
    """Return diffprivlib's scale with an explicit compatibility failure."""
    scale = getattr(mechanism, "_scale", None)
    if scale is None:
        raise RuntimeError(
            "diffprivlib GaussianAnalytic no longer exposes a calibrated _scale; "
            "update NEPRA's compatibility layer before reporting benchmark noise scales"
        )
    try:
        value = float(scale)
    except (TypeError, ValueError) as error:
        raise RuntimeError("diffprivlib GaussianAnalytic returned a non-numeric _scale") from error
    if not np.isfinite(value) or value <= 0:
        raise RuntimeError("diffprivlib GaussianAnalytic returned an invalid _scale")
    return value


class AnalyticGaussianRandomizer:
    """Isotropic analytic Gaussian randomization for clipped vectors.

    Seeded mode is intended only for reproducible benchmark runs. Secure mode
    delegates every draw to diffprivlib with fresh secret randomness.
    """

    def __init__(
        self,
        *,
        epsilon: float,
        delta: float,
        clipping_norm: float,
        seed: int | None = None,
        secure: bool = False,
    ) -> None:
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if not 0 < delta < 1:
            raise ValueError("delta must be in (0, 1)")
        if clipping_norm <= 0:
            raise ValueError("clipping_norm must be positive")
        if secure and seed is not None:
            raise ValueError("secure mode does not accept a deterministic seed")
        if not secure and seed is None:
            raise ValueError("benchmark mode requires an explicit seed")

        self.epsilon = float(epsilon)
        self.delta = float(delta)
        self.clipping_norm = float(clipping_norm)
        self.seed = seed
        self.secure = secure
        self.sensitivity = 2.0 * self.clipping_norm
        self._mechanism = GaussianAnalytic(
            epsilon=self.epsilon,
            delta=self.delta,
            sensitivity=self.sensitivity,
            random_state=None if secure else seed,
        )

    @property
    def standard_deviation(self) -> float:
        """Return diffprivlib's analytically calibrated Gaussian scale."""
        return _analytic_gaussian_scale(self._mechanism)

    def transform(self, dataset: FeatureDataset) -> FeatureDataset:
        """Randomize an already-clipped feature dataset."""
        norms = np.linalg.norm(dataset.features, axis=1)
        tolerance = np.finfo(np.float64).eps * max(1.0, self.clipping_norm) * 16
        if np.any(norms > self.clipping_norm + tolerance):
            raise ValueError("Gaussian randomization requires features clipped to norm C")

        if self.secure:
            flat = np.fromiter(
                (self._mechanism.randomise(float(value)) for value in dataset.features.flat),
                dtype=np.float64,
                count=dataset.features.size,
            )
            randomized = flat.reshape(dataset.features.shape)
        else:
            rng = np.random.default_rng(self.seed)
            noise = rng.normal(
                loc=0.0,
                scale=self.standard_deviation,
                size=dataset.features.shape,
            )
            randomized = dataset.features + noise
        return dataset.with_features(randomized)


@dataclass(frozen=True)
class TaskWeightModel:
    """Task importance and RMS-normalized empirical noise weights."""

    importance: FloatArray
    weights: FloatArray

    def __post_init__(self) -> None:
        if self.importance.ndim != 1 or self.weights.ndim != 1:
            raise ValueError("importance and weights must be one-dimensional")
        if self.importance.shape != self.weights.shape:
            raise ValueError("importance and weights must have equal shape")
        if np.any(self.weights <= 0):
            raise ValueError("all empirical noise weights must be positive")
        if not np.isclose(np.mean(np.square(self.weights)), 1.0):
            raise ValueError("empirical weights must have unit RMS")


def fit_task_weights(
    calibration: FeatureDataset,
    *,
    task_c: float,
    weight_floor: float,
    weight_exponent: float,
    max_iter: int,
    seed: int,
) -> TaskWeightModel:
    """Fit task-aware empirical noise weights on public calibration data."""
    sessions = set(calibration.session_labels)
    if sessions != {CALIBRATION_SESSION}:
        raise ValueError(
            "task weights require only the public calibration session "
            f"{CALIBRATION_SESSION!r}; received {sorted(sessions)}"
        )
    if not 0 < weight_floor <= 1:
        raise ValueError("weight_floor must be in (0, 1]")
    if weight_exponent <= 0:
        raise ValueError("weight_exponent must be positive")

    classifier = LogisticRegression(
        C=task_c,
        solver="lbfgs",
        max_iter=max_iter,
        random_state=seed,
    )
    classifier.fit(calibration.features, calibration.task_labels)
    importance = np.linalg.norm(classifier.coef_, axis=0)
    max_importance = float(np.max(importance))
    if max_importance <= np.finfo(np.float64).eps:
        relative_weights = np.ones_like(importance)
    else:
        normalized_importance = importance / max_importance
        relative_weights = 1.0 - (1.0 - weight_floor) * normalized_importance
    curved_weights = np.power(relative_weights, weight_exponent)
    rms = float(np.sqrt(np.mean(np.square(curved_weights))))
    weights = curved_weights / rms
    return TaskWeightModel(
        importance=np.asarray(importance, dtype=np.float64),
        weights=np.asarray(weights, dtype=np.float64),
    )


class TaskAwareEmpiricalRandomizer:
    """Dimension-weighted Gaussian perturbation with matched expected energy."""

    def __init__(
        self,
        *,
        standard_deviation: float,
        clipping_norm: float,
        weights: FloatArray,
        seed: int,
    ) -> None:
        if standard_deviation <= 0:
            raise ValueError("standard_deviation must be positive")
        if clipping_norm <= 0:
            raise ValueError("clipping_norm must be positive")
        values = np.asarray(weights, dtype=np.float64)
        if values.ndim != 1 or np.any(values <= 0):
            raise ValueError("weights must be a positive one-dimensional vector")
        if not np.isclose(np.mean(np.square(values)), 1.0):
            raise ValueError("weights must have unit RMS")
        self.standard_deviation = float(standard_deviation)
        self.clipping_norm = float(clipping_norm)
        self.weights = values
        self.seed = int(seed)

    def transform(self, dataset: FeatureDataset) -> FeatureDataset:
        """Add task-weighted Gaussian noise to clipped features."""
        if dataset.features.shape[1] != self.weights.shape[0]:
            raise ValueError("weight count does not match feature dimensions")
        norms = np.linalg.norm(dataset.features, axis=1)
        tolerance = np.finfo(np.float64).eps * max(1.0, self.clipping_norm) * 16
        if np.any(norms > self.clipping_norm + tolerance):
            raise ValueError("empirical randomization requires features clipped to norm C")
        rng = np.random.default_rng(self.seed)
        noise = rng.normal(
            loc=0.0,
            scale=self.standard_deviation,
            size=dataset.features.shape,
        )
        return dataset.with_features(dataset.features + noise * self.weights[None, :])
