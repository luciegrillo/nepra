"""Motor-task decoders and subject-identity attackers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from nepra.config import ModelConfig


def build_task_decoder(config: ModelConfig, seed: int) -> Pipeline:
    """Build a personalized linear motor-imagery decoder."""
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classify",
                LogisticRegression(
                    C=config.task_c,
                    solver="lbfgs",
                    class_weight="balanced",
                    max_iter=config.max_iter,
                    random_state=seed,
                ),
            ),
        ]
    )


def build_identity_attackers(config: ModelConfig, seed: int, n_jobs: int) -> Mapping[str, Any]:
    """Build the documented closed-set identity attack suite."""
    return {
        "logistic_regression": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classify",
                    LogisticRegression(
                        C=config.attacker_c,
                        solver="lbfgs",
                        class_weight="balanced",
                        max_iter=config.max_iter,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "rbf_svm": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classify",
                    SVC(
                        C=config.attacker_c,
                        kernel="rbf",
                        gamma="scale",
                        class_weight="balanced",
                        probability=True,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "knn": Pipeline(
            [
                ("scale", StandardScaler()),
                ("classify", KNeighborsClassifier(n_neighbors=5)),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=config.rf_estimators,
            class_weight="balanced_subsample",
            n_jobs=n_jobs,
            random_state=seed,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=config.rf_estimators,
            class_weight="balanced",
            n_jobs=n_jobs,
            random_state=seed,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=config.max_iter,
            learning_rate=0.08,
            random_state=seed,
        ),
        "mlp": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classify",
                    MLPClassifier(
                        hidden_layer_sizes=config.mlp_hidden,
                        activation="relu",
                        early_stopping=True,
                        max_iter=config.max_iter,
                        n_iter_no_change=20,
                        tol=1e-3,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }
