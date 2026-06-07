# Experiment Protocol

## Dataset

The v0.1 demo uses all nine subjects and all four motor-imagery classes from
`BNCI2014_001`. Session `0train` is public calibration data. Session `1test`
is held out until final evaluation.

Signals are filtered to 8-30 Hz, resampled to 160 Hz, and epoched from 0 to
4 seconds through MOABB.

## Calibration

The following operations are fitted once on pooled `0train` trials:

1. OAS covariance estimation per trial;
2. one global Riemannian tangent-space reference;
3. one global standard scaler;
4. the 95th percentile L2 clipping threshold;
5. task-importance coefficients for empirical noise allocation.

No `1test` trial contributes to these parameters.

## Conditions

- **Clean:** standardized tangent features.
- **Clipped:** clean features projected onto the calibrated L2 ball.
- **Analytic Gaussian:** clipped features with isotropic Gaussian noise
  calibrated by `diffprivlib.GaussianAnalytic`.
- **Task-aware empirical:** clipped features with dimension-weighted Gaussian
  noise and the same expected total squared noise as the isotropic condition.

The randomized conditions use epsilon values `0.5, 1, 2, 4, 8`, delta
`1e-5`, and five independent benchmark seeds. Calibration and held-out
representations receive independent noise draws.

## Motor-Task Evaluation

Each subject receives a separate logistic-regression decoder. The model is
trained on that subject's transformed `0train` trials and evaluated on the
same subject's transformed `1test` trials.

Utility is reported as balanced accuracy and macro F1. For each condition and
epsilon, seed-level scores are averaged per subject before computing the
overall mean and a percentile bootstrap interval over subjects.

## Identity Evaluation

Identity attackers are trained globally across all nine subjects:

- logistic regression;
- RBF SVM;
- random forest;
- compact MLP.

Two regimes are evaluated:

1. a clean auxiliary attacker trained on clean `0train` features;
2. a mechanism-aware attacker trained on independently transformed `0train`
   features.

Both are tested on transformed `1test` features. The attack with the highest
mean balanced accuracy is reported as the strongest observed attacker.

Reported attack metrics are balanced accuracy, macro F1, advantage over the
`1/9` chance rate, and majority-vote session identification accuracy.

## Reproducibility

Benchmark mode fixes noise and model seeds. These deterministic outputs test
engineering behavior and enable comparison, but they are not private releases.
The analytic mechanism also exposes a secure API that rejects deterministic
seeds and delegates fresh randomness to diffprivlib.

Every run writes the resolved configuration, environment versions, raw metric
table, aggregate summary, manifest, and plots. `nepra validate-run` checks the
artifact schema.

