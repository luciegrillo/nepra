from __future__ import annotations

from importlib import resources
from pathlib import Path

import pytest

from nepra.config import ConfigError, load_config


def test_smoke_and_demo_configs_are_valid() -> None:
    smoke = load_config("configs/smoke.yaml")
    demo = load_config("configs/demo.yaml")

    assert smoke.dataset.name == "synthetic"
    assert demo.dataset.name == "BNCI2014_001"
    assert demo.dataset.subjects == tuple(range(1, 10))
    assert len(demo.dataset.classes) == 4
    assert demo.privacy.epsilons == (0.5, 1.0, 2.0, 4.0, 8.0)


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
