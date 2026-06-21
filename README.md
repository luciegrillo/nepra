# NEPRA

NEPRA is a reproducible benchmark for studying identity leakage mitigation in
Riemannian EEG representations. It compares motor-imagery utility with
subject-identification attacks under clipping, isotropic Gaussian
randomization, and task-aware empirical perturbation.

The benchmark deliberately separates two kinds of evidence:

- a conditionally formal, trial-level local differential privacy baseline for
  inference-time tangent vectors;
- empirical identity-obfuscation results against a documented attack suite.

These statements are not interchangeable. Read the
[threat model](docs/threat-model.md) and [methodology](docs/methodology.md)
before interpreting results.

## Documentation

- [Methodology and claim boundaries](docs/methodology.md)
- [Threat model](docs/threat-model.md)
- [Experiment protocol](docs/experiment-protocol.md)
- [v0.1 benchmark results](docs/results.md)
- [Dataset provenance and access](docs/data-provenance.md)
- [Known limitations](docs/limitations.md)
- [Scientific references](docs/references.md)
- [Security policy](SECURITY.md)

## Quick Start

NEPRA targets Python 3.12 and uses [uv](https://docs.astral.sh/uv/) for
environment and dependency management.

```bash
uv sync --frozen
uv run nepra smoke
```

The smoke command uses synthetic EEG and requires no network access. It prints
the generated run directory, which can be checked independently:

```bash
uv run nepra validate-run artifacts/runs/<run-id>
```

Run the nine-subject benchmark:

```bash
uv run nepra data download --config configs/demo.yaml
uv run nepra run --config configs/demo.yaml
```

Raw EEG is cached outside the repository at `~/.cache/nepra`.

CI runs linting, tests, dependency audit, SBOM validation, secret scanning, and
synthetic smoke checks. See the [security policy](SECURITY.md) for scope and
reporting guidance.

## Result Snapshot

The first full run used all nine `BNCI2014_001` subjects and both sessions.

| Condition | Epsilon | Motor task | Strongest identity attack |
|---|---:|---:|---:|
| Clean | - | 61.23% | 97.07% |
| Clipped | - | 61.11% | 97.18% |
| Analytic Gaussian | 8.0 | 25.46% | 16.71% |
| Task-aware empirical | 8.0 | 25.33% | 16.67% |

The tested noise levels reduced identity leakage, but they also reduced the
four-class motor task to approximately its 25% chance level. The adaptive
heuristic did not improve the observed trade-off. This negative result is part
of the benchmark, not hidden behind a selected operating point.

See the [complete results](docs/results.md), the validated
[v0.1 run artifacts](docs/results/v0.1/run/), and the generated
[privacy-utility plot](docs/results/v0.1/privacy-utility.png).

## Intended Use

NEPRA is research and educational software. It is not a medical device,
an anonymization service, or a production privacy system.

## License

Licensed under the [Apache License 2.0](LICENSE).
