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
- [Known limitations](docs/limitations.md)
- [Scientific references](docs/references.md)

## Development

NEPRA targets Python 3.12 and uses [uv](https://docs.astral.sh/uv/) for
environment and dependency management.

```bash
uv sync --frozen
```

The executable benchmark is introduced progressively before the first public
release.

## Intended Use

NEPRA is research and educational software. It is not a medical device,
an anonymization service, or a production privacy system.

## License

Licensed under the [Apache License 2.0](LICENSE).

