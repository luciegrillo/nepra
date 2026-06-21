"""Versioned experiment configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

CURRENT_SCHEMA_VERSION = 2
SUPPORTED_SCHEMA_VERSIONS = frozenset({1, CURRENT_SCHEMA_VERSION})
SUPPORTED_DATASETS = frozenset({"synthetic", "BNCI2014_001", "BNCI2014_004", "BNCI2015_001"})
SESSION_SPLITS = frozenset({"fixed", "first_last"})


class ConfigError(ValueError):
    """Raised when an experiment configuration is invalid."""


def _require_keys(data: dict[str, Any], expected: set[str], section: str) -> None:
    missing = expected - data.keys()
    extra = data.keys() - expected
    if missing:
        raise ConfigError(f"{section} is missing keys: {sorted(missing)}")
    if extra:
        raise ConfigError(f"{section} has unknown keys: {sorted(extra)}")


@dataclass(frozen=True)
class RunConfig:
    name: str
    output_dir: Path


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    subjects: tuple[int, ...]
    classes: tuple[str, ...]
    cache_dir: Path
    resample_hz: float
    fmin: float
    fmax: float
    tmin: float
    tmax: float
    trials_per_class: int
    channels: int
    session_split: str = "fixed"


@dataclass(frozen=True)
class PrivacyConfig:
    epsilons: tuple[float, ...]
    delta: float
    clipping_percentile: float
    weight_floor: float
    weight_exponent: float
    seeds: tuple[int, ...]


@dataclass(frozen=True)
class ModelConfig:
    task_c: float
    attacker_c: float
    rf_estimators: int
    mlp_hidden: tuple[int, ...]
    max_iter: int


@dataclass(frozen=True)
class EvaluationConfig:
    bootstrap_samples: int
    n_jobs: int


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: int
    run: RunConfig
    dataset: DatasetConfig
    datasets: tuple[DatasetConfig, ...]
    privacy: PrivacyConfig
    models: ModelConfig
    evaluation: EvaluationConfig

    def to_dict(self) -> dict[str, Any]:
        """Return a YAML-serializable resolved configuration."""
        data = asdict(self)
        data["run"]["output_dir"] = str(self.run.output_dir)
        if self.schema_version == 1:
            data.pop("datasets")
            data["dataset"]["cache_dir"] = str(self.dataset.cache_dir)
            data["dataset"].pop("session_split")
        else:
            data.pop("dataset")
            for index, dataset in enumerate(self.datasets):
                data["datasets"][index]["cache_dir"] = str(dataset.cache_dir)
        return data


def _positive(value: float, name: str) -> None:
    if value <= 0:
        raise ConfigError(f"{name} must be positive")


def _build_dataset_config(raw: dict[str, Any], *, default_session_split: str) -> DatasetConfig:
    _require_keys(
        raw,
        {
            "name",
            "subjects",
            "classes",
            "cache_dir",
            "resample_hz",
            "fmin",
            "fmax",
            "tmin",
            "tmax",
            "trials_per_class",
            "channels",
            "session_split",
        },
        "dataset",
    )
    return DatasetConfig(
        name=str(raw["name"]),
        subjects=tuple(int(value) for value in raw["subjects"]),
        classes=tuple(str(value) for value in raw["classes"]),
        cache_dir=Path(raw["cache_dir"]).expanduser(),
        resample_hz=float(raw["resample_hz"]),
        fmin=float(raw["fmin"]),
        fmax=float(raw["fmax"]),
        tmin=float(raw["tmin"]),
        tmax=float(raw["tmax"]),
        trials_per_class=int(raw["trials_per_class"]),
        channels=int(raw["channels"]),
        session_split=str(raw.get("session_split", default_session_split)),
    )


def _build_config(raw: dict[str, Any]) -> ExperimentConfig:
    schema_version = raw.get("schema_version")
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ConfigError(
            f"unsupported schema_version={schema_version}; expected one of "
            f"{sorted(SUPPORTED_SCHEMA_VERSIONS)}"
        )
    root_keys = (
        {"schema_version", "run", "dataset", "privacy", "models", "evaluation"}
        if schema_version == 1
        else {"schema_version", "run", "datasets", "privacy", "models", "evaluation"}
    )
    _require_keys(raw, root_keys, "root")

    run = raw["run"]
    privacy = raw["privacy"]
    models = raw["models"]
    evaluation = raw["evaluation"]
    if schema_version == 1:
        dataset_values = (raw["dataset"],)
    else:
        raw_datasets = raw["datasets"]
        if not isinstance(raw_datasets, list | tuple):
            raise ConfigError("datasets must be a sequence")
        dataset_values = tuple(raw_datasets)
    for value, name in (
        (run, "run"),
        (privacy, "privacy"),
        (models, "models"),
        (evaluation, "evaluation"),
    ):
        if not isinstance(value, dict):
            raise ConfigError(f"{name} must be a mapping")
    if not dataset_values:
        raise ConfigError("datasets cannot be empty")
    for dataset in dataset_values:
        if not isinstance(dataset, dict):
            raise ConfigError("dataset must be a mapping")

    _require_keys(run, {"name", "output_dir"}, "run")
    if schema_version == 1:
        dataset_values = ({**dataset_values[0], "session_split": "fixed"},)
    _require_keys(
        privacy,
        {
            "epsilons",
            "delta",
            "clipping_percentile",
            "weight_floor",
            "weight_exponent",
            "seeds",
        },
        "privacy",
    )
    _require_keys(
        models,
        {"task_c", "attacker_c", "rf_estimators", "mlp_hidden", "max_iter"},
        "models",
    )
    _require_keys(evaluation, {"bootstrap_samples", "n_jobs"}, "evaluation")
    datasets = tuple(
        _build_dataset_config(dataset, default_session_split="fixed") for dataset in dataset_values
    )

    config = ExperimentConfig(
        schema_version=int(schema_version),
        run=RunConfig(name=str(run["name"]), output_dir=Path(run["output_dir"])),
        dataset=datasets[0],
        datasets=datasets,
        privacy=PrivacyConfig(
            epsilons=tuple(float(value) for value in privacy["epsilons"]),
            delta=float(privacy["delta"]),
            clipping_percentile=float(privacy["clipping_percentile"]),
            weight_floor=float(privacy["weight_floor"]),
            weight_exponent=float(privacy["weight_exponent"]),
            seeds=tuple(int(value) for value in privacy["seeds"]),
        ),
        models=ModelConfig(
            task_c=float(models["task_c"]),
            attacker_c=float(models["attacker_c"]),
            rf_estimators=int(models["rf_estimators"]),
            mlp_hidden=tuple(int(value) for value in models["mlp_hidden"]),
            max_iter=int(models["max_iter"]),
        ),
        evaluation=EvaluationConfig(
            bootstrap_samples=int(evaluation["bootstrap_samples"]),
            n_jobs=int(evaluation["n_jobs"]),
        ),
    )
    validate_config(config)
    return config


def validate_config(config: ExperimentConfig) -> None:
    """Validate cross-field configuration constraints."""
    if not config.run.name.strip():
        raise ConfigError("run.name cannot be empty")
    if not config.datasets:
        raise ConfigError("datasets cannot be empty")
    names = [dataset.name for dataset in config.datasets]
    if len(set(names)) != len(names):
        raise ConfigError("dataset names must be unique within one run")
    for dataset in config.datasets:
        if dataset.name not in SUPPORTED_DATASETS:
            raise ConfigError(f"dataset.name must be one of {sorted(SUPPORTED_DATASETS)}")
        if dataset.session_split not in SESSION_SPLITS:
            raise ConfigError(f"dataset.session_split must be one of {sorted(SESSION_SPLITS)}")
        if len(dataset.subjects) < 2:
            raise ConfigError("at least two subjects are required for identity attacks")
        if len(set(dataset.subjects)) != len(dataset.subjects):
            raise ConfigError("dataset.subjects must be unique")
        if len(dataset.classes) < 2:
            raise ConfigError("at least two task classes are required")
        if not 0 < dataset.fmin < dataset.fmax:
            raise ConfigError("dataset frequencies must satisfy 0 < fmin < fmax")
        if dataset.tmin >= dataset.tmax:
            raise ConfigError("dataset times must satisfy tmin < tmax")
        _positive(dataset.resample_hz, "dataset.resample_hz")
        if dataset.name == "synthetic":
            _positive(dataset.trials_per_class, "dataset.trials_per_class")
            _positive(dataset.channels, "dataset.channels")
    if not config.privacy.epsilons:
        raise ConfigError("privacy.epsilons cannot be empty")
    for epsilon in config.privacy.epsilons:
        _positive(epsilon, "privacy epsilon")
    if not 0 < config.privacy.delta < 1:
        raise ConfigError("privacy.delta must be in (0, 1)")
    if not 0 < config.privacy.clipping_percentile <= 100:
        raise ConfigError("privacy.clipping_percentile must be in (0, 100]")
    if not 0 < config.privacy.weight_floor <= 1:
        raise ConfigError("privacy.weight_floor must be in (0, 1]")
    _positive(config.privacy.weight_exponent, "privacy.weight_exponent")
    if not config.privacy.seeds:
        raise ConfigError("privacy.seeds cannot be empty")
    _positive(config.models.task_c, "models.task_c")
    _positive(config.models.attacker_c, "models.attacker_c")
    _positive(config.models.rf_estimators, "models.rf_estimators")
    _positive(config.models.max_iter, "models.max_iter")
    _positive(config.evaluation.bootstrap_samples, "evaluation.bootstrap_samples")


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate a YAML experiment configuration."""
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ConfigError("configuration root must be a mapping")
    return _build_config(raw)
