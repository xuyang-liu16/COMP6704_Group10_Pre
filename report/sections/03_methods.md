# Methodology (WU Aoqi)

Author: WU Aoqi, 26027031r.

## Shared objective

All four algorithms minimize

$$
F(w)=f(w)+g(w),\quad
f(w)=\frac1n\sum_{i=1}^{n}\log(1+\exp(-y_i x_i^\top w)),\quad
g(w)=\frac{\lambda}{2}\|w\|_2^2.
$$

Here \(X\in\mathbb{R}^{n\times d}\) has rows \(x_i^\top\),
\(y_i\in\{-1,+1\}\), and \(w\in\mathbb{R}^d\) includes the bias.
Every coordinate receives the L2 penalty. With predictions
\(p_i=\sigma(x_i^\top w)\), where \(\sigma(a)=1/(1+\exp(-a))\),

$$
\nabla F(w)=X^\top\left(p-\frac{y+\mathbf1}{2}\right)/n+\lambda w.
$$

Since \(0\le p_i(1-p_i)\le1/4\), valid gradient Lipschitz constants
are \(L_f=\|X\|_2^2/(4n)\) for the loss and \(L=L_f+\lambda\) for
the full objective. For \(\lambda>0\), strong convexity with
\(\mu=\lambda\) gives a unique minimizer \(w^\star\).
`problem.lipschitz_hint` computes the spectral norm.

## Update rules

### Gradient descent

Gradient descent uses \(w_{k+1}=w_k-\nabla F(w_k)/L\).
The module starts with this step and includes Armijo backtracking.
For a convex smooth objective, the objective gap is \(O(1/k)\).
With strong convexity and this step size,
\(F(w_k)-F(w^\star)\le(1-\mu/L)^k[F(w_0)-F(w^\star)]\).

### Nesterov accelerated gradient

The implemented general-convex version uses \(z_0=w_0=0\), \(t_0=1\):

$$
w_{k+1}=z_k-\nabla F(z_k)/L,\quad
t_{k+1}=(1+\sqrt{1+4t_k^2})/2,\quad
z_{k+1}=w_{k+1}+\frac{t_k-1}{t_{k+1}}(w_{k+1}-w_k).
$$

Its objective-gap guarantee is \(O(1/k^2)\). A different strongly
convex schedule uses fixed momentum
\(\beta=(\sqrt L-\sqrt\mu)/(\sqrt L+\sqrt\mu)\) and achieves
\(O((1-\sqrt{\mu/L})^k)\), up to a problem-dependent constant.
That schedule is not the current experimental implementation.

### FISTA and proximal gradient

The fast iterative shrinkage-thresholding algorithm, FISTA, uses the
split \(F=f+g\). Both terms are smooth here, and the quadratic penalty
has an exact proximal operator:

$$
\operatorname{prox}_{\eta g}(v)
=\arg\min_u\left\{g(u)+\frac{\|u-v\|^2}{2\eta}\right\}
=\frac{v}{1+\eta\lambda}.
$$

Initialize \(w_0=z_0=0\), \(t_0=1\), and \(\eta=1/L_f\). Repeat

$$
w_{k+1}=\frac{z_k-\eta\nabla f(z_k)}{1+\eta\lambda},\qquad
t_{k+1}=\frac{1+\sqrt{1+4t_k^2}}2,
$$

$$
z_{k+1}=w_{k+1}+\frac{t_k-1}{t_{k+1}}(w_{k+1}-w_k).
$$

This is the default `mode="prox"`. The gradient contains only logistic
loss, so the penalty enters once through shrinkage. Setting the
extrapolation coefficient to zero gives proximal gradient.
The optional `mode="smooth"` applies the same momentum to the full
gradient, with \(\eta=1/L\); both modes minimize the same objective.
An explicit `step` must be positive and no greater than the relevant
inverse Lipschitz bound.

For this quadratic penalty, the default prox update can be rewritten as

$$
\frac{z-\nabla f(z)/L_f}{1+\lambda/L_f}
=z-\frac{\nabla f(z)+\lambda z}{L_f+\lambda}.
$$

Thus FISTA and the implemented Nesterov solver have the same mathematical
iterates under their default steps, initialization, and momentum.
Their objective curves should overlap up to roundoff. Runtime differences
reflect implementation and measurement costs; these curves do not compare
two distinct acceleration schedules.

For \(L_f>0\), convex FISTA satisfies

$$
F(w_k)-F(w^\star)\le
\frac{2L_f\|w_0-w^\star\|^2}{(k+1)^2},\qquad k\ge1.
$$

The implemented momentum does not explicitly exploit strong convexity.
We claim \(O(1/k^2)\), without a linear-rate claim for this schedule.
Individual objective values can increase. If the design matrix is zero,
the zero initial point is optimal; the code uses a finite fallback step.

FISTA stops when \(\|\nabla F(w_k)\|_2\le\texttt{tol}\), or when
the update budget is exhausted. Setting `tol=0` disables early stopping.
For \(\lambda>0\), the full-gradient norm gives the certificate
\(F(w_k)-F(w^\star)\le\|\nabla F(w_k)\|^2/(2\lambda)\).

### ADMM

The alternating direction method of multipliers, ADMM, solves
\(\min_{w,z}f(w)+g(z)\) subject to \(w=z\).
With scaled dual variable \(u\) and penalty \(\rho>0\),

$$
w_{k+1}=\arg\min_w\left\{f(w)+\frac\rho2\|w-z_k+u_k\|^2\right\},
\quad z_{k+1}=\frac{\rho}{\rho+\lambda}(w_{k+1}+u_k),
$$

$$
u_{k+1}=u_k+w_{k+1}-z_{k+1}.
$$

The current ADMM module uses warm-started Newton steps with Armijo
backtracking for the logistic subproblem. Defaults are \(\rho=1\),
at most 50 inner steps, and inner gradient tolerance \(10^{-10}\).
Primal and
dual residuals are \(r_{k+1}=w_{k+1}-z_{k+1}\) and
\(s_{k+1}=\rho(z_{k+1}-z_k)\). Standard convex convergence requires
a saddle point and accurate subproblem solutions. Inexact solves need
additional error control. Ergodic \(O(1/k)\) results concern averages
and appropriate gap or feasibility measures, not an arbitrary
last-iterate objective curve.

## Computational cost

For dense data, each gradient costs \(O(nd)\). GD, Nesterov, and
FISTA use \(O(nd)\) work per update and \(O(n+d)\) working memory,
excluding the design matrix. The L2 prox costs \(O(d)\).
FISTA also evaluates the objective and full-gradient norm after each
update; these extra passes affect measured runtime. A scalar history
over \(K\) updates costs \(O(K)\) storage.

Spectral-norm setup typically costs \(O(nd\min(n,d))\) for dense
linear algebra. ADMM's outer cost depends on its inner solver:
\(m\) gradient steps cost \(O(mnd)\); a dense Newton step can require
\(O(nd^2+d^3)\) work. Equal outer-iteration budgets therefore do not
imply equal computation.

## Experimental protocol

| Item | Setting |
| --- | --- |
| Synthetic dataset | 2,000 samples, 50 features, 20 informative and 5 redundant |
| Breast cancer dataset | scikit-learn built-in data: 569 samples, 30 features |
| Split | Stratified 75% train / 25% test, seed 0 |
| Scaling | Fit `StandardScaler` on training rows only; apply it to both splits |
| Bias | Append ones; regularize the added coordinate |
| Training shape | Synthetic: 1,500 x 51; breast cancer: 426 x 31 |
| Regularization | \(\lambda=0.01\) |
| Budget | 400 updates; requested tolerance \(10^{-6}\) for every method |
| Initialization | All four current modules start from zero |
| Metrics | Training objective, full-gradient norm, train/test accuracy, parameter norm, updates, wall time |

Test labels affect evaluation only. FISTA logs iteration zero and every
update. Its cumulative times include validation, spectral-norm setup,
updates, objective logging, and stationarity checks. `wall_s` covers the
complete solver call, including final metric packing, but excludes
data loading, module imports, and the runner's independent validation.
GD and Nesterov start their cumulative clocks after input validation;
ADMM starts before validation but labels its initial history entry zero.
These small timing differences and diagnostic costs remain in the curves.

The runner independently imports GD, Nesterov, FISTA, and ADMM.
It validates finite weights, aligned histories, nondecreasing timestamps,
and agreement between the last objective and the returned iterate.
Endpoint metrics are recomputed through the shared helpers.
`summary.json` retains histories and solver diagnostics; `curves.json`
provides a compact curve-only export. `table.md` distinguishes valid
returns from satisfying the common gradient tolerance.
Failed or unfinished methods remain explicit rows and are omitted
from plots with an availability annotation.

```bash
.venv/bin/python experiments/run_compare.py
.venv/bin/python scripts/make_figures.py
.venv/bin/python experiments/run_compare.py --dataset breast_cancer --out results/breast_cancer --figure-dir figures/breast_cancer
```

The comparison command also calls the standalone figure script.
`figures/obj_vs_iter` and `figures/obj_vs_time` are saved as PNG and PDF;
the second dataset uses its separate output directories.
Plots show the raw training objective without subtracting an estimated
optimum. Single-seed runs check reproducibility. A speed ranking needs
working implementations, comparable stopping rules, controlled thread
settings, and repeated timings. An unavailable method supplies no
evidence of inferior optimization performance.
