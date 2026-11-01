# Conclusions (LIU Xuyang)

Author: LIU Xuyang, 26123832r.

## What the comparison establishes

L2-logistic regression gives all four solvers the same training target.
With positive regularization on every coordinate, its minimizer is unique.
The comparison asks how much computation each solver needs to approach
that solution, how stable its progress is, and whether remaining
optimization differences affect held-out predictions.

The verified single-run artifact reaches the same objective to the
reported precision with all four methods. GD finishes first in that
record; all solvers achieve 73.80% test accuracy. These observations do
not establish a stable runtime ranking across repetitions or datasets.

## Strengths, weaknesses, and selection

| Method | Strengths | Weaknesses | When to use it |
| --- | --- | --- | --- |
| Gradient descent, GD | Simple full-gradient baseline; valid steps give monotone descent in exact arithmetic | Poor conditioning can require many iterations | Small or well-conditioned smooth problems; debugging and baseline comparisons |
| Nesterov accelerated gradient | Look-ahead momentum improves the general-convex objective-gap bound to $O(1/k^2)$ | Can oscillate; requires a valid step; the implemented schedule does not exploit known strong convexity | Smooth centralized training where first-order iteration cost is low |
| Fast iterative shrinkage-thresholding algorithm, FISTA | Supports a smooth loss plus an inexpensive proximal term; L2 prox has closed form | Can oscillate; benefit over Nesterov depends on the split and schedule | Composite objectives, particularly future L1 or constraint-indicator extensions |
| Alternating direction method of multipliers, ADMM | Splits loss and regularizer; supports consensus or constraint-based decompositions | Penalty tuning, extra variables, and expensive subproblem solves | Problems where splitting or data distribution offsets its overhead |

For a smooth convex objective with a minimizer, fixed-step GD has an
$O(1/k)$ objective-gap guarantee. The implemented general-convex Nesterov
sequence and standard FISTA have $O(1/k^2)$ guarantees under their
respective step-size assumptions. These rates concern iterations, not
wall-clock time or prediction accuracy.

Positive $\lambda$ gives a strong-convexity bound $\mu=\lambda$. With
$\kappa=L/\mu$, fixed-step GD can achieve
$O((1-1/\kappa)^k)$ objective-gap decay. A separately configured
strongly convex Nesterov variant can achieve
$O((1-1/\sqrt{\kappa})^k)$. Our Nesterov uses the general-convex
$t_k$ sequence, so this latter linear rate is a possible extension,
not a guarantee claimed for the executed code.

ADMM guarantees depend on the split, convexity assumptions, and accuracy
of the subproblem solves. An ergodic convergence result must not be
presented as a last-iterate objective-gap rate without its corresponding
assumptions and metric. A logistic subproblem solved by inner Newton
iterations also makes one ADMM iteration more expensive than one
first-order update.

FISTA and Nesterov are closely related on this smooth objective. Placing
the whole objective in FISTA's smooth term and using the identity prox,
the same step, and the same momentum schedule can produce identical
iterates. Splitting out the L2 penalty instead uses
$\operatorname{prox}_{\alpha g}(v)=v/(1+\alpha\lambda)$.
In the default configuration, this proximal step with $\alpha=1/L_f$
equals a full-gradient step of size $1/(L_f+\lambda)$. With their shared
momentum sequence and initialization, Nesterov and FISTA therefore have
matching trajectories up to roundoff. The observed overlap is expected,
not independent evidence of two different acceleration mechanisms.
Timing differences can still reflect implementation overhead and noise.

For centralized L2-logistic training, start with GD as a correctness
baseline and evaluate Nesterov as an accelerated alternative. Choose
FISTA when proximal structure is useful. Consider ADMM when constraints
or decomposition justify its additional work. The fastest implementation
for the tested data must be determined by measurements.

## Verified single-run results

Use identical data splits, zero initialization, regularization, and a
declared budget. Compare time to a common solution-quality target, as
well as fixed-budget outcomes. A numerical reference solution must be
labeled approximate; matching its objective alone is not an exact proof.

Source: `results/summary.json`, checked on 1 November 2026, with SHA-256
`cdce2dd2a0820bc5ede2404730a7dc9af109d9967a53139f3ea3e1f29f812487`,
the same snapshot used in the results section. It uses synthetic data,
seed 0, $\lambda=0.01$, 1,500 training and 500 test observations,
51 parameters, a 400-update cap, and gradient tolerance $10^{-6}$.
All four rows pass the common gradient check.

| Method | Final objective | Updates | Final gradient norm | Full-call seconds | Parameter norm |
| --- | ---: | ---: | ---: | ---: | ---: |
| GD | 0.493214040493 | 136 | $9.813\times10^{-7}$ | 0.007084 | 1.633161 |
| Nesterov | 0.493214040491 | 138 | $8.685\times10^{-7}$ | 0.008519 | 1.633165 |
| FISTA | 0.493214040491 | 138 | $8.685\times10^{-7}$ | 0.008109 | 1.633165 |
| ADMM | 0.493214040493 | 195 | $9.683\times10^{-7}$ | 0.054900 | 1.633161 |

Every method has 76.67% training accuracy and 73.80% test accuracy.
ADMM uses $\rho=1$, an inner tolerance of $10^{-10}$, and at most 50
inner Newton updates. Its final primal and dual residuals are
$9.684\times10^{-9}$ and $9.684\times10^{-7}$, respectively. The original
gradient and both residual stopping checks pass.

GD reaches the shared gradient threshold with fewer updates than either
accelerated implementation in this run. Acceleration's worst-case
guarantee does not predict a win on every instance or stopping metric.
The reported seconds describe one call per method, including setup and
logging but excluding data loading and module imports. They are not
repeated timing estimates or a measured scaling law.

Still needed: `[TBD_HARDWARE_THREADS_REPEATS]`,
`[TBD_MULTI_SEED_STATISTICS]`, `[TBD_GRADIENT_AND_INNER_WORK_COUNTS]`,
and `[TBD_TIME_TO_TARGET_DISTRIBUTION]`. Exact hardware/thread provenance
and timing variation are not stored in this summary. Preserve failed or
budget-limited rows in later comparisons; a valid return alone is not a
convergence certificate. Reruns can overwrite the source artifact, so
refresh this table and the results section together.

## Limitations

- Small dense binary-classification datasets do not establish performance
  on large sparse data, distributed systems, or nonconvex neural networks.
- One random seed or train/test split cannot establish statistical
  stability. Lower training risk need not yield higher test accuracy.
- Standardization and $\lambda$ affect both conditioning and the fitted
  model. Changing either across solvers confounds the comparison.
  Hyperparameter selection must not use the test labels.
- The spectral norm has setup cost. Nesterov evaluates an additional
  gradient for monitoring; ADMM can perform multiple inner steps.
  Count this work and use consistent full-call timing boundaries.
- The implementation regularizes the intercept and clips sigmoid inputs.
  External library comparisons must match these conventions and the
  loss normalization.
- Algorithm names do not specify implementation details. The final
  report must match the owners' actual FISTA mode, GD line search, and
  ADMM stopping rules. A single-machine run does not measure distributed
  speedup, communication overhead, or privacy.

## Future work

1. Complete common-target comparisons across seeds and repeated timings.
   Validate solutions with an independent numerical reference and report
   residuals, including failed and budget-limited cases.
2. Vary $\lambda$, sample size, and feature correlation. Compare
   iterations and full-call time to separate conditioning effects from
   predictive differences.
3. Evaluate strong-convexity-aware momentum, backtracking, and adaptive
   restart with tuning cost included. Add L1 or elastic-net regularization
   to test where the proximal formulation becomes useful.
4. Study ADMM penalty sensitivity and inner-solve accuracy, then evaluate
   a genuine consensus deployment with communication and local work
   included.

The current evidence supports comparable solution quality and equal
classification accuracy on one synthetic split. GD has the lowest
recorded runtime there. A broader selection recommendation requires
repeated measurements on the intended data and explicit accounting of
each method's work.
