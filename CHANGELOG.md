# Changelog

All notable changes to NEPRA are documented in this file.

## 0.1.0 - 2026-06-07

### Added

- Versioned Python 3.12 environment managed with uv.
- Offline synthetic smoke benchmark.
- Reproducible `BNCI2014_001` access through MOABB.
- Strict cross-session Riemannian EEG representation.
- Analytic Gaussian trial-level local DP baseline with explicit assumptions.
- Task-aware empirical perturbation with matched expected noise energy.
- Personalized motor-task decoders and four global identity attackers.
- Clean-auxiliary and mechanism-aware attack regimes.
- Versioned run artifacts, validation, summaries, and plots.
- Automated linting, tests, coverage, and smoke execution.
- Full nine-subject benchmark results and limitations.

### Result

The tested Gaussian conditions reduced subject-identification performance but
also reduced four-class motor-task utility to chance. The task-aware empirical
mechanism did not improve the measured trade-off over isotropic noise.

