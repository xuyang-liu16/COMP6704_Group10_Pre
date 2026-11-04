# Supplementary experiment results

Original member solvers, unchanged. Full-gradient target 1e-6; lambda=0.01; zero initialization; maximum 10,000 updates; single-thread numerical libraries.

Timing: 1 warm-up per configuration; randomized solver order within each repeat. Full solver call includes setup, logging and final metrics. Data loading, imports, endpoint verification and gradient tracing are excluded. Error bars are IQR, not confidence intervals.

Gradient traces come from a separate replay, checked against the timed trajectory. Time plots use internal timestamps from the repeat nearest the upper median wall time; they are representative runs, not pointwise median curves.

## Same-precision comparison (seed 0, 20 timed repeats)

| Dataset | Method | Updates | Gradient norm | Target met | Median ms | IQR ms | Test accuracy |
|---|---|---:|---:|---|---:|---:|---:|
| synthetic | gd | 136 | 9.813e-07 | True | 12.12 | 12.02–12.18 | 73.80% |
| synthetic | nesterov | 138 | 8.685e-07 | True | 17.99 | 17.75–18.27 | 73.80% |
| synthetic | fista | 138 | 8.685e-07 | True | 18.16 | 17.80–18.42 | 73.80% |
| synthetic | admm | 195 | 9.683e-07 | True | 110.64 | 109.55–111.67 | 73.80% |
| breast_cancer | gd | 2450 | 9.987e-07 | True | 65.56 | 64.24–66.49 | 96.50% |
| breast_cancer | nesterov | 1172 | 9.877e-07 | True | 47.53 | 47.09–48.11 | 96.50% |
| breast_cancer | fista | 1172 | 9.877e-07 | True | 48.32 | 47.87–48.77 | 96.50% |
| breast_cancer | admm | 738 | 9.893e-07 | True | 117.87 | 116.29–120.04 | 96.50% |

## ADMM rho sensitivity (seed 0, 5 repeats)

| Dataset | rho | Updates | Gradient norm | Target met | Median ms | Inner Newton steps |
|---|---:|---:|---:|---|---:|---:|
| synthetic | 0.01 | 18 | 2.273e-12 | True | 4.08 | 5 |
| synthetic | 0.1 | 25 | 9.556e-07 | True | 18.22 | 53 |
| synthetic | 1.0 | 195 | 9.683e-07 | True | 111.21 | 322 |
| synthetic | 10.0 | 1894 | 9.945e-07 | True | 964.76 | 2609 |
| breast_cancer | 0.01 | 19 | 9.269e-13 | True | 1.69 | 8 |
| breast_cancer | 0.1 | 78 | 9.102e-07 | True | 14.80 | 137 |
| breast_cancer | 1.0 | 738 | 9.893e-07 | True | 118.27 | 1023 |
| breast_cancer | 10.0 | 7336 | 9.998e-07 | True | 1085.30 | 8497 |

## Seed sensitivity

Seeds 0–4. Breast-cancer seeds change the stratified split; synthetic seeds change both the generated dataset and split. Timing repetitions are not independent datasets.

| Dataset | Method | Converged seeds / 5 | Updates min–max | Test accuracy min–max |
|---|---|---:|---:|---:|
| synthetic | gd | 5/5 | 136–198 | 73.80–83.60% |
| synthetic | nesterov | 5/5 | 138–168 | 73.80–83.60% |
| synthetic | fista | 5/5 | 138–168 | 73.80–83.60% |
| synthetic | admm | 5/5 | 195–233 | 73.80–83.60% |
| breast_cancer | gd | 5/5 | 2339–2450 | 96.50–97.90% |
| breast_cancer | nesterov | 5/5 | 1127–1192 | 96.50–97.90% |
| breast_cancer | fista | 5/5 | 1127–1192 | 96.50–97.90% |
| breast_cancer | admm | 5/5 | 700–738 | 96.50–97.90% |

## Reproduce

`python experiments/supplement.py`

Use the project requirements in your Python environment. Raw repetitions, histories, weights, baseline 400-update checks, package versions, source commit and thread settings are in raw.json. Original saved results and slide files are not overwritten.

These timings describe these implementations on this machine. rho was swept for optimization efficiency using training convergence, not selected using test accuracy. The default rho=1 remains the main comparison; tuned ADMM is a separate sensitivity result.
