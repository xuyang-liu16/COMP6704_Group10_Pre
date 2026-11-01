# Experimental Results (ZHAO Jixin)

## 4.1 Protocol and evidence

The shared experiment uses L2-regularized binary logistic regression with
`lam=0.01`, a zero initialization, and at most 400 outer iterations. The
synthetic dataset contains 2,000 examples and 50 features. A stratified 75/25
split with seed 0 gives 1,500 training and 500 test examples. Standardization
fits the training split only. The added bias column gives 51 parameters, all
regularized by `problem.objective`.

Snapshot: 1 November 2026, file modification time 19:34:56 local. The table
below transcribes `results/summary.json`, whose
configuration matches the protocol above. Each row passes the runner's
gradient check at `1e-6`. The separate penalty sweep in Section 4.3 comes
from standalone ADMM validation in the existing `.venv`. Its timings belong
to a different invocation.

Snapshot SHA-256:
`cdce2dd2a0820bc5ede2404730a7dc9af109d9967a53139f3ea3e1f29f812487`.
Reruns overwrite this artifact and can change timing without changing the
deterministic optimization results.

| Method | Run status | Final objective | Outer iterations | Wall time, s | Train accuracy | Test accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| GD | OK; gradient check passed | 0.493214040493 | 136 | 0.007084 | 76.67% | 73.80% |
| Nesterov | OK; gradient check passed | 0.493214040491 | 138 | 0.008519 | 76.67% | 73.80% |
| FISTA | OK; gradient check passed | 0.493214040491 | 138 | 0.008109 | 76.67% | 73.80% |
| ADMM | OK; gradient and residual checks passed | 0.493214040493 | 195 | 0.054900 | 76.67% | 73.80% |

For later runs, refresh rows from `final_obj`, `n_iter`, `wall_s`,
`train_acc`, and `test_acc`. Preserve `ok=False` rows as failures with their
error message. The runner's `ok=True` means that the call returned and its
history passed validation. Numerical convergence requires the separate
`gradient_converged` check and, for ADMM, its residual checks. Missing
metrics remain unavailable, never zero.

All methods reach essentially the same terminal objective. GD takes fewer
updates than the accelerated methods at this stopping tolerance. On this
single timing run, it also finishes first, while ADMM takes about 0.055 s.
Repeated measurements are required to assess the timing differences.

Nesterov and default proximal FISTA have matching trajectories up to
roundoff in this implementation. Let `L_f` be the logistic smoothness bound.
For the FISTA step `eta=1/L_f`, its L2 update is
`(v-eta*grad f(v))/(1+eta*lam)`.
This equals `v-grad F(v)/(L_f+lam)`, the Nesterov update at the same
extrapolated point. Both use the same momentum recurrence. Their agreement
is expected from these particular choices.

## 4.2 ADMM and stopping criteria

Write the mean logistic loss as `ell(w)` and introduce a copy `z`:

```text
minimize ell(w) + (lam/2) ||z||^2, subject to w = z

w[k+1] = argmin_w ell(w) + (rho/2) ||w - z[k] + u[k]||^2
z[k+1] = rho/(rho + lam) * (w[k+1] + u[k])
u[k+1] = u[k] + w[k+1] - z[k+1]
```

`u` is the scaled dual variable. The logistic subproblem uses warm-started
Newton iterations with Armijo backtracking. The L2 update is closed form.
The default penalty is `rho=1`, with at most 50 inner Newton updates and an
inner gradient tolerance of `1e-10`. This implementation performs a dense
Newton solve, so an outer iteration can cost substantially more than a
gradient step.

For `d` parameters, the recorded residuals and thresholds are

```text
r[k] = ||w[k] - z[k]||
s[k] = rho * ||z[k] - z[k-1]||
eps_primal[k] = tol * (sqrt(d) + max(||w[k]||, ||z[k]||))
eps_dual[k] = tol * (sqrt(d) + ||rho * u[k]||).
```

Early stopping requires both residual tests, a successful inner solve, and
`||grad F(w[k])|| <= tol`, where `F` is the original regularized objective.
The default outer tolerance is `1e-6`. The original gradient check prevents
a feasible zero start from being mistaken for an optimum. `tol=0` requests
the full iteration budget.

Every reported objective and accuracy uses the returned `w`. History index
zero is the initial point, with objective `log(2)` and time zero. Subsequent
times include validation, initialization, inner solves, and diagnostics.
The final accuracy computation lies outside the last history timestamp.
The runner's external `wall_s` includes the entire solver call, but excludes
data loading. Without a held-out pair, `test_acc` is `None`.

## 4.3 Standalone measurements

One local validation run used Python 3.14.5, NumPy 2.5.4, SciPy 1.18.1, and
scikit-learn 1.9.1. All rows use the synthetic setup above, `tol=1e-6`, and
the same 400-iteration cap. Times are single-call measurements, not repeated
benchmark estimates. The command in the README reproduces this sweep.

| rho | Outer iterations | Converged | Final objective | Wall time, s | Train accuracy | Test accuracy |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 0.01 | 18 | Yes | 0.493214040485 | 0.0026 | 76.67% | 73.80% |
| 0.1 | 25 | Yes | 0.493214040493 | 0.0088 | 76.67% | 73.80% |
| 1.0 | 195 | Yes | 0.493214040493 | 0.0558 | 76.67% | 73.80% |
| 10.0 | 400 | No: budget exhausted | 0.493455451589 | 0.1311 | 76.67% | 73.80% |

A SciPy BFGS reference solve reaches objective `0.493214040484765`, with
gradient norm `2.66e-8`. Default ADMM differs by `8.33e-12` in objective.
Its final primal residual is `9.68e-9`, dual residual is `9.68e-7`, and
original gradient norm is `9.68e-7`. All inner solves in this sweep satisfy
their tolerance.

At `rho=0.01`, the penalty equals `lam`. With the zero start, the first
logistic subproblem is already the original regularized problem. The
subsequent consensus iterations mainly reduce the split residuals. This
special choice explains the low total of five Newton updates across 18
outer iterations. It should be disclosed when interpreting this sweep.

At `rho=10`, the gradient norm remains `5.42e-3`, despite identical
classification accuracy. Accuracy alone therefore cannot certify numerical
convergence. Across the converged rows, the training-to-test gap is 2.87
percentage points on this single split.

An additional correctness check on `breast_cancer`, seed 0, uses the same
`lam`, `rho=1`, and a larger cap of 1,000. ADMM converges after 738 iterations
to objective `0.094167338333`, with 99.06% training and 96.50% test accuracy.
This check uses a different dataset and budget, so it is outside the shared
400-iteration comparison.

## 4.4 Figure and analysis template

### Convergence by iteration

The existing figures `figures/obj_vs_iter.png` and `figures/obj_vs_time.png`
plot the raw training objective on a linear vertical axis. They show the
early decrease and the runtime difference, but compress the terminal errors.
The gap plots and residual panels below are planned refinements.

Figure 1 refinement: plot `F(w[k]) - F_ref` against outer iteration on a
logarithmic vertical axis. Obtain `F_ref` from a separate high-accuracy solve for each
dataset, split, and `lam`. Inspect its gradient norm. Label it a numerical
reference, not an exact optimum. Clip gaps to `1e-14` for display only and
retain signed differences in the numerical audit.

Fill-in analysis: "[METHOD] first reaches objective gap [THRESHOLD] after
[K] iterations. Over [K_START, K_END], the measured gap decreases by
[FACTOR]. [METHOD] remains above the threshold at the budget limit."

To discuss the observed rate, inspect semilog and log-log curves away from
the numerical floor. Approximately constant successive gap ratios suggest
geometric decay over the selected interval. A log-log slope estimates a
power law over that interval. Report the fit window and residuals. Neither
visual fit proves a worst-case convergence rate, and raw ADMM objective
values need not decrease every iteration.

### Convergence by elapsed time

Figure 2 refinement: plot the same gap against `time_s`, then compare time to common
thresholds `1e-4`, `1e-6`, and `1e-8`. Mark an unreached target explicitly.
Include terminal `wall_s` as a separate summary, because different stopping
rules can produce different final accuracies.

Fill-in analysis: "[METHOD] reaches gap [THRESHOLD] in [SECONDS], compared
with [SECONDS] for [BASELINE]. Median times over [REPEATS] runs are [VALUES],
with interquartile ranges [VALUES]."

ADMM uses dense Hessian formation and factorization in each Newton update.
Report `sum(inner_iterations)` alongside outer counts. For example, default
ADMM in the standalone sweep uses 322 inner updates. A comparison based only
on outer iterations would omit this work.

### Prediction quality

Figure 3: show paired training/test accuracy bars for every successful
method, with the final objective and convergence flag next to each pair.
Report differences in percentage points. The existing shared history stores
terminal accuracy only, so an accuracy trajectory requires additional
instrumentation by the experiment owner.

Fill-in analysis: "At comparable optimization error, train accuracy spans
[RANGE] and test accuracy spans [RANGE]. The training-to-test gap is [GAP].
Across [SEEDS] splits, mean test accuracy is [MEAN] with standard deviation
[SD]."

For a fixed split and positive `lam`, the objective has a unique minimizer.
Different solvers should approach that solution when run accurately enough.
Differences near a classification boundary can still affect terminal
accuracy. A single split cannot establish a generalization advantage.

### Stability

Figure 4: plot ADMM primal/dual residuals and their thresholds against
iteration, with a separate panel for each `rho`. Omit the artificial zero
residuals at index zero from logarithmic plots. Add a penalty-versus-time
panel and a matched step-size panel once the other solvers support it.

Fill-in analysis: "[SETTING] converges in [COUNT/SEEDS] runs. [SETTING]
exhausts its budget or reports an inner failure. The residual curves show
[OBSERVATION], while the original gradient norm ends at [VALUE]."

The present sweep supports sensitivity to `rho` on one synthetic instance.
It provides no measured evidence about other algorithms' step-size stability.
`rho` is a penalty coefficient. ADMM's Newton line-search step is a separate
quantity, with initial value 1 and repeated halving.

## 4.5 Ablation plan

| Factor | Proposed settings | Measurements and controls |
| --- | --- | --- |
| Regularization `lam` | `1e-4, 1e-3, 1e-2, 1e-1` | Recompute the reference optimum and smoothness bound for every setting; report weight norm and accuracy |
| First-order step | `c/L`, with `c=0.25, 0.5, 1`; GD initial-step stress tests at `1.5, 2.5` | Use `L_f` for proximal FISTA and `L_f+lam` for GD/Nesterov; Nesterov/FISTA reject oversized steps, while GD backtracks |
| ADMM penalty `rho` | `0.001, 0.01, 0.1, 1, 10` | Keep the inner tolerance fixed; report residuals, gradient norm, Newton work, and budget exhaustion |
| Inner accuracy | `1e-6, 1e-8, 1e-10` | Check whether cheaper inner solves preserve outer convergence; keep `inner_max_iter=50` |
| Data and timing | Both datasets; seeds `0, 1, 2, 3, 4`; one warm-up and five timed repeats | Fix hardware and BLAS threads; use identical data per paired comparison; report median time and IQR |

First sweep each factor separately around `lam=0.01`, then examine the
`lam`/`rho` interaction. Report the `rho=lam` case explicitly. Use equal
outer budgets for trajectory plots and a common objective-gap target for
efficiency comparisons. Count failed and nonconverged settings.

Choose `lam` on a validation split within the training set, with scaling
fitted only on that inner training partition. Keep the final test set out
of tuning. Solver step and penalty selection should use training
optimization criteria. The current runner does not implement this grid or
validation protocol; implementation belongs to the experiment owner.

## 4.6 Current conclusion

ADMM matches an independent numerical optimizer on the tested problems and
exposes unsuccessful termination through its convergence flag. The shared
run reaches the same predictive accuracy for all four methods, with ADMM
slower on this small dense problem. Penalty choice changes convergence within
a fixed budget even when accuracy stays constant. Repeated timing estimates,
multi-seed conclusions, and regularization ablations remain pending.
