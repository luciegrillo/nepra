# Methodology

## Research Question

NEPRA asks how much motor-imagery utility remains when tangent-space EEG
representations are perturbed to reduce subject-identification performance.
It reports a privacy-utility benchmark; it does not claim to anonymize EEG.

## Representation

Each epoched multichannel trial is represented by an OAS covariance matrix.
Covariances are projected into a Euclidean tangent space around a reference
estimated exclusively from the public calibration session. Tangent features
are standardized with calibration-session statistics before clipping or
randomization.

The reference, scaler, and clipping threshold are frozen before the protected
session is transformed.

## Compared Conditions

1. **Clean:** standardized tangent vectors without clipping or noise.
2. **Clipped:** vectors constrained to an L2 norm of `C`.
3. **Analytic Gaussian:** clipped vectors randomized with an isotropic analytic
   Gaussian mechanism.
4. **Task-aware empirical:** clipped vectors receive dimension-weighted
   Gaussian noise based on calibration-session task coefficients.

The analytic Gaussian condition is the only mechanism eligible for a formal
privacy statement. The task-aware condition is evaluated empirically.

## Conditional Privacy Statement

For the inference release, preprocessing parameters are public and fixed. A
single standardized tangent vector is clipped to norm `C`, giving replace-one
L2 sensitivity at most `2C`. Isotropic noise is calibrated with the analytic
Gaussian mechanism for the configured `(epsilon, delta)`.

Under those assumptions, with fresh secret randomness, the released vector is
described as a trial-level approximate local differential privacy mechanism.
This statement excludes the public calibration data and does not imply
user-level protection across repeated trials.

Seeded benchmark runs reproduce draws for engineering verification. Their
outputs are not presented as private releases.

## Empirical Perturbation

Task importance is estimated from the L2 norm of multiclass logistic-regression
coefficients fitted only on calibration data. Inverse importance becomes a
positive per-dimension noise weight. Weights use a floor and root-mean-square
normalization, so expected total squared noise matches the isotropic condition.

Equal expected noise energy is an experimental control, not a differential
privacy proof. Anisotropic Gaussian privacy depends on worst-case sensitivity
under the inverse covariance metric.

## Evaluation Principles

- Motor-task decoders are personalized: one model per subject.
- Identity attackers are pooled across subjects.
- The calibration session is used for all fitting and parameter selection.
- The held-out session is used only for final transformation and evaluation.
- Both clean-trained and mechanism-aware identity attackers are tested.
- The strongest observed identity attack is reported for each condition.
- Results use descriptive metrics and subject-bootstrap intervals rather than
  significance tests over dependent cross-validation folds.

