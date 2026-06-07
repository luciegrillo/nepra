from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from nepra.config import ExperimentConfig, load_config
from nepra.data import EEGDataset, load_dataset
from nepra.evaluation import BenchmarkResult, run_benchmark
from nepra.geometry import FeatureDataset, RiemannianRepresentation


@pytest.fixture(scope="session")
def smoke_config() -> ExperimentConfig:
    return load_config("configs/smoke.yaml")


@pytest.fixture(scope="session")
def synthetic_dataset(smoke_config: ExperimentConfig) -> EEGDataset:
    return load_dataset(smoke_config.dataset)


@pytest.fixture(scope="session")
def feature_datasets(
    synthetic_dataset: EEGDataset,
) -> tuple[FeatureDataset, FeatureDataset]:
    calibration, heldout = synthetic_dataset.split_calibration_heldout()
    representation = RiemannianRepresentation()
    return (
        representation.fit_transform(calibration),
        representation.transform(heldout),
    )


@pytest.fixture(scope="session")
def benchmark_result(
    smoke_config: ExperimentConfig,
    synthetic_dataset: EEGDataset,
) -> BenchmarkResult:
    return run_benchmark(smoke_config, synthetic_dataset)


@pytest.fixture
def temporary_output_config(
    smoke_config: ExperimentConfig,
    tmp_path: Path,
) -> ExperimentConfig:
    return replace(
        smoke_config,
        run=replace(smoke_config.run, output_dir=tmp_path),
    )
