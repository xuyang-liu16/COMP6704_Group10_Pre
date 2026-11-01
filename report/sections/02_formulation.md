# Problem Formulation (LIU Xuyang)

Author: LIU Xuyang, 26123832r.

## Learning task and variables

We train a linear probabilistic classifier on fixed features. For observation
$i \in \{1,\ldots,n\}$, let $a_i \in \mathbb{R}^{d-1}$ be the standardized
features and $y_i \in \{-1,+1\}$ its label. Append a constant to obtain
$x_i=(a_i^\top,1)^\top \in \mathbb{R}^d$. The rows of
$X \in \mathbb{R}^{n \times d}$ are $x_i^\top$, and
$y=(y_1,\ldots,y_n)^\top$. The optimization variable is
$w=(\theta^\top,b)^\top \in \mathbb{R}^d$, where $\theta$ contains feature
weights and $b$ is the intercept. There are no constraints on $w$.

With $\sigma(s)=1/(1+\exp(-s))$, the model assigns
$P(y_i=+1\mid x_i;w)=\sigma(x_i^\top w)$. Its predicted class is $+1$ when
$x_i^\top w\geq0$ and $-1$ otherwise, matching `metrics.accuracy`.
The negative log-likelihood of one labeled observation is
$\ell_i(w)=\log(1+\exp(-y_i x_i^\top w))$.

## Regularized empirical risk

For fixed regularization $\lambda\geq0$, training minimizes

$$
F_\lambda(w)
=\underbrace{\frac{1}{n}\sum_{i=1}^n
\log(1+\exp(-y_i x_i^\top w))}_{f(w)}
+\underbrace{\frac{\lambda}{2}\|w\|_2^2}_{g(w)},
\qquad w\in\mathbb{R}^d.
$$

Write $w^\star$ for a minimizer when one exists. For $\lambda>0$ it exists
and is unique, as proved below; for $\lambda=0$ a finite minimizer need
not exist.

The averaging factor $1/n$ keeps the loss scale comparable across sample
sizes. Regularization discourages large coefficients; $\lambda$ controls
the trade-off between fitting labels and shrinking parameters. In this
repository, the intercept is also regularized:
$\|w\|_2^2=\|\theta\|_2^2+b^2$. Excluding $b$ would change the problem and
remove the global strong-convexity bound $\mu=\lambda$.

This formulation connects convex optimization to machine learning through
a specified training objective. With identical features, labels, and
$\lambda$, the solvers fit the same model. We can therefore study
optimization efficiency separately from model design. Held-out accuracy
measures generalization; a reduction in training objective need not
increase test accuracy. The convexity arguments apply to fixed-feature
logistic regression, not to jointly trained nonlinear neural networks.

## Gradient and Hessian

Define $q_i=(y_i+1)/2$, $q=(q_1,\ldots,q_n)^\top$,
$p_i(w)=\sigma(x_i^\top w)$, and $p(w)=(p_1(w),\ldots,p_n(w))^\top$.
Differentiation gives

$$
\nabla F_\lambda(w)
=-\frac{1}{n}\sum_{i=1}^n
y_i x_i\,\sigma(-y_i x_i^\top w)+\lambda w
=\frac{1}{n}X^\top(p(w)-q)+\lambda w.
$$

Let $D(w)=\operatorname{diag}(p_i(w)(1-p_i(w)))_{i=1}^n$.
The Hessian is

$$
\nabla^2 F_\lambda(w)=\frac{1}{n}X^\top D(w)X+\lambda I_d,
$$

where $I_d$ is the $d$-dimensional identity matrix. Since
$0<p_i(w)(1-p_i(w))\leq1/4$ for finite scores, any
$v\in\mathbb{R}^d$ satisfies

$$
\lambda\|v\|_2^2
\leq v^\top\nabla^2 F_\lambda(w)v
=\frac{1}{n}\sum_i p_i(w)(1-p_i(w))(x_i^\top v)^2
 +\lambda\|v\|_2^2
\leq\left(\frac{\|X\|_2^2}{4n}+\lambda\right)\|v\|_2^2.
$$

Here $\|X\|_2$ is the spectral norm, its largest singular value.
The nonnegative lower bound proves convexity for $\lambda\geq0$.
For $\lambda>0$, the function is $\mu$-strongly convex with
$\mu=\lambda$. It is coercive because
$F_\lambda(w)\geq\lambda\|w\|_2^2/2$, so a minimizer exists. Strong
convexity proves uniqueness. Every stationary point is the global optimum.

Without regularization, rank deficiency can prevent strict convexity.
Even full column rank does not ensure an attained optimum: on separable
data, the logistic loss can approach its infimum as $\|w\|_2$ diverges.

## Lipschitz gradient and acceleration

Integrating the Hessian along the segment between any
$u,v\in\mathbb{R}^d$ gives

$$
\|\nabla F_\lambda(u)-\nabla F_\lambda(v)\|_2
\leq L\|u-v\|_2,\qquad
L=\frac{\|X\|_2^2}{4n}+\lambda.
$$

Thus $F_\lambda$ is globally $L$-smooth. The descent lemma reads

$$
F_\lambda(u)\leq F_\lambda(v)
+\nabla F_\lambda(v)^\top(u-v)+\frac{L}{2}\|u-v\|_2^2.
$$

For $L>0$, a gradient step of size $1/L$ is valid.
`problem.lipschitz_hint` computes the spectral norm using
`np.linalg.norm(X, ord=2)`, despite a nearby comment mentioning a
Frobenius proxy. The logistic loss alone has bound
$L_f=\|X\|_2^2/(4n)$.

The implemented Nesterov method uses the general-convex acceleration
sequence. Set $w_0=z_0=0$, $t_0=1$, and choose $0<\alpha\leq1/L$:

$$
\begin{aligned}
w_{k+1}&=z_k-\alpha\nabla F_\lambda(z_k),\\
t_{k+1}&=\frac{1+\sqrt{1+4t_k^2}}{2},\\
z_{k+1}&=w_{k+1}+
\frac{t_k-1}{t_{k+1}}(w_{k+1}-w_k).
\end{aligned}
$$

The vector $z_k$ is the look-ahead point, separate from labels $y$.
The default is $\alpha=1/L$. With an attained minimizer and exact
gradients, this sequence has objective-gap guarantee
$F_\lambda(w_k)-F_\lambda(w^\star)
=O(\|w_0-w^\star\|_2^2/(\alpha k^2))$ for $k\geq1$.
Individual objective values need not decrease monotonically.
The code does not use the strongly convex fixed-momentum variant
$\beta=(1-\sqrt{\mu/L})/(1+\sqrt{\mu/L})$; that variant's linear rate
must not be attributed to this implementation.

When $L=0$, both $X$ and $\lambda$ are zero, the objective is constant,
and the initial vector is returned. This handles degenerate synthetic
inputs, although the usual appended bias column makes $L>0$.

For $\lambda>0$, minimizing the strong-convexity lower model also yields
an a posteriori certificate:

$$
0\leq F_\lambda(w)-F_\lambda(w^\star)
\leq\frac{\|\nabla F_\lambda(w)\|_2^2}{2\lambda}.
$$

`nesterov.run` checks the gradient norm at the accepted iterate against
`tol`. Each update evaluates one look-ahead gradient and an additional
gradient for monitoring. `tol=0` disables early stopping; `max_iter`
limits updates. Reaching that budget alone does not certify convergence.

## Equivalent solver formulations

FISTA can split $F_\lambda=f+g$ into logistic loss and L2 penalty.
For $\alpha>0$ and $v\in\mathbb{R}^d$, its proximal map is

$$
\operatorname{prox}_{\alpha g}(v)
:=\underset{u\in\mathbb{R}^d}{\operatorname{argmin}}
\left\{\frac{\lambda}{2}\|u\|_2^2+
\frac{1}{2\alpha}\|u-v\|_2^2\right\}
=\frac{v}{1+\alpha\lambda}.
$$

This scales all coordinates, including the bias. Soft-thresholding
would correspond to an L1 penalty and a different training objective.
For the default FISTA step $\alpha=1/L_f$ with $L_f>0$, this update is
algebraically identical to the full-gradient Nesterov step:

$$
\frac{v-\nabla f(v)/L_f}{1+\lambda/L_f}
=v-\frac{\nabla f(v)+\lambda v}{L_f+\lambda}
=v-\frac{\nabla F_\lambda(v)}{L}.
$$

Both implementations use the same zero initialization and momentum
sequence. Their default trajectories therefore coincide up to roundoff,
even though the code expresses the updates differently. This equality
depends on the L2 split and these matched steps; it does not hold for
arbitrary composite penalties.

ADMM instead introduces an auxiliary $z\in\mathbb{R}^d$ and solves
$\min_{w,z}f(w)+g(z)$ subject to $w=z$. At feasibility this is the same
original problem. Its augmented Lagrangian is not the objective to use
for the shared convergence plot.

## Data and measurement contract

`data.load_problem` uses a stratified 75/25 train/test split. It fits
standardization on training features only, transforms the test data with
those statistics, then appends the bias column. Its synthetic configuration
has 2,000 observations and 50 original features; the training matrix is
$1500\times51$. These are generator settings, not measured outcomes.
The loader also supports breast-cancer data; results on that dataset
require a separate completed run.

Training and stopping use only training data. The common return fields are
`objective`, `time_s`, `final_obj`, `train_acc`, `test_acc`, and `w_norm`.
Nesterov records initialization and every accepted iterate, recomputes
`final_obj` at the returned vector, and uses `test_acc=None` when no
held-out pair is supplied. It also exposes `n_iter`, `step`, `grad_norm`,
`converged`, and `stop_reason`.

Its internal cumulative timer includes spectral-norm setup, updates, and
monitoring, but excludes input validation and final metric packaging.
The initial timestamp can therefore be positive. The driver's full-call
`wall_s` is the appropriate boundary for end-to-end runtime comparison;
other solvers' internal timers must be audited before comparing `time_s`.

The objective uses `np.logaddexp` for stable loss evaluation. The shared
sigmoid clips scores to $[-40,40]$, so the computed gradient approximates
the analytic expression at extreme margins. The theoretical statements
assume exact gradients and arithmetic; numerical checks allow roundoff.
