"""NEPRA command-line interface."""

from __future__ import annotations

import argparse
import sys
from importlib import resources
from pathlib import Path

from nepra.config import ConfigError, load_config
from nepra.data import download_dataset, load_dataset
from nepra.evaluation import run_benchmark
from nepra.reporting import validate_run, write_run


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nepra",
        description="Reproducible identity-leakage benchmark for Riemannian EEG.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke", help="run the offline synthetic benchmark")
    smoke.add_argument("--config")

    data = subparsers.add_parser("data", help="manage benchmark datasets")
    data_subparsers = data.add_subparsers(dest="data_command", required=True)
    download = data_subparsers.add_parser("download", help="download configured EEG data")
    download.add_argument("--config", required=True)
    download.add_argument("--force", action="store_true")

    run = subparsers.add_parser("run", help="run a configured benchmark")
    run.add_argument("--config", required=True)

    validate = subparsers.add_parser("validate-run", help="validate a run artifact directory")
    validate.add_argument("path")
    return parser


def _execute_benchmark(config_path: str) -> Path:
    config = load_config(config_path)
    datasets = {dataset.name: load_dataset(dataset) for dataset in config.datasets}
    result = run_benchmark(config, datasets)
    return write_run(config, result)


def _execute_smoke(config_path: str | None) -> Path:
    if config_path is not None:
        return _execute_benchmark(config_path)
    packaged_config = resources.files("nepra").joinpath("resources/smoke.yaml")
    with resources.as_file(packaged_config) as path:
        return _execute_benchmark(str(path))


def main(argv: list[str] | None = None) -> int:
    """Execute the NEPRA CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "smoke":
            output = _execute_smoke(args.config)
            print(output)
            return 0
        if args.command == "run":
            output = _execute_benchmark(args.config)
            print(output)
            return 0
        if args.command == "data" and args.data_command == "download":
            config = load_config(args.config)
            caches = [
                (dataset.name, download_dataset(dataset, force=args.force))
                for dataset in config.datasets
            ]
            if len(caches) == 1:
                print(caches[0][1])
            else:
                for dataset_name, cache in caches:
                    print(f"{dataset_name}: {cache}")
            return 0
        if args.command == "validate-run":
            validate_run(args.path)
            print(Path(args.path).resolve())
            return 0
    except (ConfigError, OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    parser.error("unhandled command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
