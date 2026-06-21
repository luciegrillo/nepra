# NEPRA v0.2 Scientific Protocol

This document defines the v0.2 analysis before publishing v0.2 results. It is a
protocol for a benchmark, not a clinical study, privacy certification, or
anonymization claim.

## Research Questions

NEPRA v0.2 asks whether the v0.1 privacy-utility findings remain stable across
multiple motor-imagery datasets and under stronger empirical identity analyses.

Primary questions:

1. Do perturbations that reduce closed-set subject identification preserve
   useful motor-imagery decoding performance?
2. Does task-aware empirical weighting improve the privacy-utility trade-off
   over isotropic analytic Gaussian noise at matched expected noise energy?
3. Does repeated release increase empirical identity leakage even when
   single-trial leakage appears low?
4. Are aggregate conclusions consistent across more than one dataset?

## Hypotheses

- H1: Gaussian perturbation reduces subject-identification performance relative
  to clean and clipped representations.
- H2: The same perturbation can destroy task utility at conservative trial-level
  local privacy budgets.
- H3: Task-aware weighting is supported only if it improves utility while
  producing equal or lower identity leakage than isotropic noise at the same
  epsilon.
- H4: Repeated-release aggregation increases identity recovery relative to
  single-trial release.
- H5: Conclusions are treated as robust only when they hold in the aggregate and
  are not driven by a single dataset.

## Datasets

The v0.2 core benchmark uses three multi-session MOABB motor-imagery datasets:

| Dataset | Subjects | Classes | Split |
|---|---:|---|---|
| `BNCI2014_001` | 9 | left hand, right hand, feet, tongue | first session to last session |
| `BNCI2014_004` | 9 | left hand, right hand | first session to last session |
| `BNCI2015_001` | 12 | feet, right hand | first session to last session |

The first observed session is public calibration and auxiliary attacker data.
The last observed session is the protected held-out evaluation session. Datasets
with fewer than two observed sessions are not part of the v0.2 core benchmark.

## Conditions

The benchmark keeps the v0.1 conditions:

1. clean standardized tangent features;
2. L2-clipped features;
3. analytic Gaussian perturbation;
4. task-aware empirical perturbation at matched expected noise energy.

The analytic Gaussian condition remains the only condition eligible for the
conditional trial-level local differential privacy statement.

## Primary Endpoints

- Motor-task balanced accuracy.
- Motor-task normalized advantage over chance.
- Strongest closed-set identity balanced accuracy.
- Open-set identity AUROC for enrolled-vs-unknown subject recognition.
- Repeated-release identity accuracy for group sizes `1`, `4`, `16`, `64`, and
  all available held-out trials.
- Utility loss and identity reduction paired against clean and clipped baselines.

## Identity Analyses

Closed-set attackers are trained on enrolled calibration subjects and tested on
held-out trials from the same enrolled subject set.

Open-set evaluation uses a deterministic subject split per dataset: the last
20% of sorted subjects, with at least one subject, are held out as unknown. The
attacker is trained only on enrolled subjects. Enrolled-vs-unknown AUROC uses
the attacker's maximum predicted enrolled-subject probability as the membership
score. This is empirical open-set recognition, not formal membership inference.

Repeated-release evaluation groups held-out predictions within each subject and
reports subject recovery after aggregating multiple trial releases. This measures
empirical composition risk and does not imply a formal user-level guarantee.

## Composition Accounting

For randomized analytic Gaussian releases, v0.2 reports the conservative basic
composition values:

- `epsilon_basic = k * epsilon`
- `delta_basic = k * delta`

where `k` is the repeated-release group size. This is not RDP, advanced
composition, or user-level differential privacy accounting.

## Statistical Reporting

Reports include per-dataset metrics and an aggregate hierarchical bootstrap.
The aggregate bootstrap resamples datasets, then subjects within datasets, and
seeds for randomized conditions. Intervals are descriptive and are not treated
as proof of population-level performance.

The primary comparison between task-aware and isotropic perturbation is paired
within dataset, epsilon, seed, and subject where applicable.

## Publication Criteria

The v0.2 result is published only if all three core datasets run successfully and
the generated artifact directory validates. Partial real-dataset runs may be
kept as local debugging artifacts but must not be documented as the v0.2 result.

If a dataset download, license term, metadata shape, or session layout prevents a
complete run, the blocker is documented and v0.2 results are not published.

## Explicit Limits

NEPRA v0.2 still does not claim:

- anonymization;
- user-level or session-level differential privacy;
- protection under repeated-release composition;
- formal membership-inference resistance;
- open-set recognition resistance beyond the tested attack suite;
- clinical or production readiness.
