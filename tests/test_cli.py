from __future__ import annotations

from pathlib import Path

from nepra import cli
from nepra.config import DatasetConfig


def test_data_download_preserves_single_dataset_output(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    def fake_download(dataset: DatasetConfig, *, force: bool = False) -> Path:
        assert dataset.name == "BNCI2014_001"
        assert not force
        return tmp_path / "cache"

    monkeypatch.setattr(cli, "download_dataset", fake_download)

    exit_code = cli.main(["data", "download", "--config", "configs/demo.yaml"])

    assert exit_code == 0
    assert capsys.readouterr().out == f"{tmp_path / 'cache'}\n"


def test_data_download_visits_all_schema_v2_datasets(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    downloaded: list[str] = []

    def fake_download(dataset: DatasetConfig, *, force: bool = False) -> Path:
        downloaded.append(dataset.name)
        assert force
        return tmp_path / dataset.name

    monkeypatch.setattr(cli, "download_dataset", fake_download)

    exit_code = cli.main(["data", "download", "--config", "configs/v0.2/core.yaml", "--force"])

    assert exit_code == 0
    assert downloaded == ["BNCI2014_001", "BNCI2014_004", "BNCI2015_001"]
    assert capsys.readouterr().out == (
        f"BNCI2014_001: {tmp_path / 'BNCI2014_001'}\n"
        f"BNCI2014_004: {tmp_path / 'BNCI2014_004'}\n"
        f"BNCI2015_001: {tmp_path / 'BNCI2015_001'}\n"
    )
