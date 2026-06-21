from __future__ import annotations

from importlib import resources
from pathlib import Path

import pytest

from nepra.config import ConfigError, load_config


def test_smoke_and_demo_configs_are_valid() -> None:
    smoke = load_config("configs/smoke.yaml")
    demo = load_config("configs/demo.yaml")

    assert smoke.schema_version == 1
    assert smoke.dataset.name == "synthetic"
    assert smoke.datasets == (smoke.dataset,)
    assert smoke.dataset.session_split == "fixed"
    assert "datasets" not in smoke.to_dict()
    assert "session_split" not in smoke.to_dict()["dataset"]
    assert demo.schema_version == 1
    assert demo.dataset.name == "BNCI2014_001"
    assert demo.datasets == (demo.dataset,)
    assert demo.dataset.subjects == tuple(range(1, 10))
    assert len(demo.dataset.classes) == 4
    assert demo.privacy.epsilons == (0.5, 1.0, 2.0, 4.0, 8.0)


def test_v0_2_core_config_is_valid_schema_v2() -> None:
    config = load_config("configs/v0.2/core.yaml")
    resolved = config.to_dict()

    assert config.schema_version == 2
    assert config.dataset == config.datasets[0]
    assert [dataset.name for dataset in config.datasets] == [
        "BNCI2014_001",
        "BNCI2014_004",
        "BNCI2015_001",
    ]
    assert all(dataset.session_split == "first_last" for dataset in config.datasets)
    assert "dataset" not in resolved
    assert "datasets" in resolved
    assert len(resolved["datasets"]) == 3
    assert resolved["datasets"][0]["session_split"] == "first_last"


def test_packaged_smoke_config_matches_repository_config() -> None:
    repository = load_config("configs/smoke.yaml")
    packaged_resource = resources.files("nepra").joinpath("resources/smoke.yaml")

    with resources.as_file(packaged_resource) as packaged_path:
        packaged = load_config(packaged_path)

    assert packaged.to_dict() == repository.to_dict()


def test_unknown_config_keys_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(
        """
schema_version: 1
run: {name: invalid, output_dir: artifacts}
dataset: {}
privacy: {}
models: {}
evaluation: {}
unexpected: true
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="unknown keys"):
        load_config(path)


def test_schema_v2_rejects_empty_datasets(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(
        """
schema_version: 2
run: {name: invalid, output_dir: artifacts}
datasets: []
privacy:
  epsilons: [1.0]
  delta: 0.00001
  clipping_percentile: 95.0
  weight_floor: 0.25
  weight_exponent: 1.0
  seeds: [42]
models:
  task_c: 1.0
  attacker_c: 1.0
  rf_estimators: 10
  mlp_hidden: [8]
  max_iter: 100
evaluation:
  bootstrap_samples: 10
  n_jobs: 1
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="datasets cannot be empty"):
        load_config(path)


def test_schema_v2_rejects_non_sequence_datasets(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(
        """
schema_version: 2
run: {name: invalid, output_dir: artifacts}
datasets: {}
privacy: {}
models: {}
evaluation: {}
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="datasets must be a sequence"):
        load_config(path)
