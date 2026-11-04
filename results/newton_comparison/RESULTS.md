# Direct Newton controls and conditioning

Same batch: all four original solvers, ADMM rho=lambda, and two direct Newton controls. 20 repetitions after warm-up, shuffled method order. Training target 1e-6. Newton reuses the existing ADMM damped-Newton kernel with zero center and rho=lambda. It is a controlled component comparison, not an independently implemented solver. The 1e-10 control matches ADMM inner tolerance; the 1e-6 control uses the common target. Newton logs only its endpoints; ADMM also logs outer iterations, residuals and metrics. Thus the time difference includes bookkeeping, repeated checks and split-variable updates; it is not a pure estimate of a mathematical splitting operation.

| Dataset | Method | Updates | Median ms | IQR ms | Gradient |
|---|---|---:|---:|---:|---:|
| synthetic | gd | 136 | 12.269 | 11.967–12.503 | 9.813e-07 |
| synthetic | nesterov | 138 | 18.046 | 17.684–18.292 | 8.685e-07 |
| synthetic | fista | 138 | 18.083 | 17.833–18.132 | 8.685e-07 |
| synthetic | admm | 195 | 114.304 | 112.167–115.771 | 9.683e-07 |
| synthetic | admm_matched | 18 | 3.856 | 3.720–4.129 | 2.273e-12 |
| synthetic | newton | 4 | 1.243 | 1.205–1.438 | 8.205e-07 |
| synthetic | newton_strict | 5 | 1.474 | 1.431–1.598 | 2.273e-12 |
| breast_cancer | gd | 2450 | 66.817 | 66.030–67.750 | 9.987e-07 |
| breast_cancer | nesterov | 1172 | 48.765 | 48.172–51.229 | 9.877e-07 |
| breast_cancer | fista | 1172 | 49.497 | 48.727–51.071 | 9.877e-07 |
| breast_cancer | admm | 738 | 122.958 | 119.546–140.132 | 9.893e-07 |
| breast_cancer | admm_matched | 19 | 1.804 | 1.723–2.093 | 9.269e-13 |
| breast_cancer | newton | 7 | 0.754 | 0.713–0.810 | 2.703e-07 |
| breast_cancer | newton_strict | 8 | 0.795 | 0.708–0.888 | 9.269e-13 |

Newton counts are Newton steps; ADMM counts are outer iterations. Do not compare counts as equal work.

## Conditioning (seed 0, standardized training design including bias)

| Dataset | L | mu lower bound | L/mu upper bound | Hessian condition at optimum |
|---|---:|---:|---:|---:|
| synthetic | 0.74970 | 0.01 | 74.97 | 47.80 |
| breast_cancer | 3.34609 | 0.01 | 334.61 | 21.04 |

L/mu is a conservative global bound, not a measured Hessian condition number. Local conditioning is evaluated at the strict Newton solution. The local condition number is actually smaller on breast cancer. Its global step bound is more conservative near the solution: L divided by the largest local Hessian eigenvalue is 15.90 versus 1.57. This is not a causal ablation. The momentum schedule is general-convex and has no restart; its objective-error guarantee does not prescribe the number of updates to a gradient target.

The rho sweep and seed checks remain the earlier separate experiments, explicitly labeled. All main four-method timing tables and curves are refreshed from this new batch.

Reproduce: `python experiments/newton_comparison.py`. Raw samples and environment are in raw.json.
