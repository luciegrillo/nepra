# Security Policy

## Scope

NEPRA is research and educational software for reproducible EEG privacy-utility
benchmarks. It is not a production anonymization service, a medical device, or a
privacy certification system.

Security work in this repository covers:

- source-code vulnerabilities in the NEPRA package and CLI;
- dependency and supply-chain risk in the Python environment;
- accidental secret or raw-data inclusion in versioned files;
- integrity checks for generated benchmark artifacts.

Raw EEG is downloaded through MOABB into the configured local cache and is not
redistributed by this repository. Published run artifacts contain derived
metrics, environment metadata, manifests, summaries, and plots.

## Reporting a Vulnerability

Do not paste secrets, private datasets, access tokens, or sensitive health data
into public issues. If GitHub private vulnerability reporting is available for
the repository, use it. Otherwise, open a minimal public issue asking for a
private contact path and include only non-sensitive context.

Please include:

- affected version or commit;
- operating system and Python version;
- concise reproduction steps;
- expected and observed behavior;
- whether raw data, credentials, or private artifacts could be exposed.

## Automated Checks

The CI workflow runs:

- Ruff linting, including security rules for `src`;
- `pip-audit` against dependencies exported from the locked environment;
- CycloneDX SBOM generation and validation;
- `detect-secrets` scanning with documented false-positive filters;
- unit tests, coverage, and synthetic smoke benchmarks;
- installed-wheel smoke execution.

These checks are safeguards, not a guarantee that the software is secure or that
the privacy claims extend beyond the documented threat model.

## Privacy-Sensitive Use

Benchmark mode uses deterministic seeds for reproducibility and must not be used
as a protected release mechanism. Secure randomization mode rejects deterministic
seeds and delegates fresh randomness to `diffprivlib`, but NEPRA v0.1 still does
not provide user-level privacy accounting or repeated-release composition.
