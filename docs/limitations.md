# Limitations

NEPRA is a small benchmark, not a privacy certification.

## Scientific Scope

- The formal privacy statement applies to one inference-time tangent vector,
  not to a session, a complete user history, or continuous BCI operation.
- The public calibration session is outside the protected set. It is used to
  estimate the tangent reference, feature scaling, and clipping threshold.
- Repeated releases compose. NEPRA v0.1 does not provide a user-level privacy
  accountant.
- The task-aware mechanism is an empirical perturbation heuristic. Matching
  expected noise energy does not prove differential privacy.
- Identity-attack performance measures the attacks included in this benchmark.
  A failed classifier is not proof of anonymity.
- Identity-attack intervals summarize variation across benchmark seeds for the
  tested attack suite. They do not measure population-level uncertainty or
  protection against stronger adversaries.

## Experimental Scope

- The main dataset has nine subjects. Subject-level uncertainty estimates are
  therefore broad and should not be treated as population-level conclusions.
- Cross-session evaluation introduces genuine recording drift. Clean
  cross-session baselines and clipping diagnostics are required to separate
  drift from perturbation effects.
- Closed-set subject identification does not measure open-set identification,
  membership inference, or sensitive-attribute inference.
- Hyperparameters are fixed before the held-out session is evaluated. The
  benchmark does not search for a universal optimal privacy-utility point.

## Engineering Scope

- Seeded benchmark mode exists for reproducibility and must not be used as a
  protected release mechanism.
- Secure mode requires fresh secret randomness for every release.
- Raw EEG data remains under its original dataset terms and is never
  redistributed by this repository.
