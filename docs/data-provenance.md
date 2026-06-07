# Dataset Provenance

## Main Benchmark

NEPRA v0.1 uses
[MOABB `BNCI2014_001`](https://moabb.neurotechx.com/docs/generated/moabb.datasets.BNCI2014_001.html),
also known as BCI Competition IV Dataset 2a.

The dataset contains:

- nine healthy participants;
- two recording sessions on different days;
- 22 EEG channels;
- four motor-imagery classes: left hand, right hand, both feet, and tongue;
- 288 trials per subject and session.

The original study is:

Tangermann, M., et al. "Review of the BCI Competition IV." *Frontiers in
Neuroscience*, 2012.
[doi:10.3389/fnins.2012.00055](https://doi.org/10.3389/fnins.2012.00055)

## Access

NEPRA downloads data through MOABB and does not redistribute raw or processed
EEG. The default cache is `~/.cache/nepra`, configurable in the experiment
YAML.

```bash
nepra data download --config configs/demo.yaml
```

Users are responsible for reviewing the current source-dataset terms before
downloading or redistributing derivatives. The Apache-2.0 license in this
repository applies to NEPRA source code, not to third-party EEG data.

## Session Semantics

MOABB exposes the original training session as `0train` and the original
evaluation session as `1test`.

- `0train` is treated as public calibration and auxiliary attacker data.
- `1test` is held out from all fitting and parameter selection.

The loader fails closed if these session identifiers or the configured task
classes are absent.

## Synthetic Data

The smoke workflow generates deterministic artificial epochs with task,
subject, and session structure. Synthetic output tests software behavior only
and must not be used for scientific claims.

