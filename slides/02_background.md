<!--
Background segment: WANG Zeqing, 26029011r.
Seven slides, 7:00 total. Speaker notes are not projected.
Timing: 45 + 60 + 65 + 70 + 65 + 55 + 60 = 420 seconds.
-->

## 1. One model, four optimization strategies

- Binary classification: features $x_i$, labels $y_i \in \{-1,+1\}$.
- Logistic regression fits a linear score and a class probability.
- L2 regularization controls parameter magnitude.
- Same training objective; compare GD, Nesterov, FISTA, and ADMM.
- Question: how much computation is needed to fit the same model?

<!-- Speaker notes | 0:00-0:45 | 45 seconds
Use a tumor-feature classifier as a concrete example: measurements enter the
model, and its score determines a binary label. Training means selecting the
weights from labeled examples. Our comparison keeps this model fixed and changes
the optimizer. That makes the computational question precise: how quickly does
each method approach the same objective value? State the four solver names, but
leave their update details for the methods segment.
Transition: first define exactly what all four solvers minimize.
-->

---

## 2. What does training minimize?

$$
F(w)=\frac{1}{n}\sum_{i=1}^{n}
\log\!\left(1+\exp(-y_i x_i^\top w)\right)
+\frac{\lambda}{2}\|w\|_2^2.
$$

- $n$ examples; $d$ features; $w,x_i \in \mathbb{R}^d$; $\lambda \geq 0$.
- Logistic loss penalizes confidently wrong predictions.
- $\lambda$ balances data fit and parameter magnitude.
- Project convention: the appended intercept is regularized too.
- Standardization uses training statistics for both splits.

<!-- Speaker notes | 0:45-1:45 | 60 seconds
Read the equation from left to right. The average loss measures agreement with
the observed labels. A large negative signed margin means a confident mistake
and receives a large penalty. The quadratic term discourages large weights.
Lambda changes the optimization problem, so every solver must receive the same
value. Our bias is a column of ones and receives the same penalty as the other
coordinates. This detail matters when comparing with library defaults. Scaling
must be fitted only on training data to keep the held-out evaluation separate.
Transition: this particular objective gives us useful convergence guarantees.
-->

---

## 3. Why is the problem convex?

- For scalar margin $t$, $\ell(t)=\log(1+\exp(-t))$ has $0\leq\ell''(t)\leq1/4$.
- Affine composition and averaging preserve convexity.
- L2 adds curvature $\lambda I$, where $I$ is the identity matrix.
- For $\lambda>0$: strong convexity and a unique minimizer $w^\star$.
- A global smoothness bound is
  $L=\|X\|_2^2/(4n)+\lambda$, with $X$ the feature matrix and $\|X\|_2$ its spectral norm.

Source: Boyd and Vandenberghe, 2004; Nesterov, 2004.

<!-- Speaker notes | 1:45-2:50 | 65 seconds
Convexity says a local minimum is global. To see it here, the scalar logistic
loss has nonnegative second derivative. Substituting a linear score and
averaging samples preserves that property. The quadratic penalty adds at least
lambda curvature in every direction because we regularize the full vector.
Positive lambda therefore gives a unique finite optimum. The upper curvature
bound is equally useful: it tells us how large a gradient step can safely be.
Correlated or poorly scaled features can increase the ratio L over lambda and
make progress slow. At lambda zero, uniqueness and finite attainment are no
longer guaranteed, particularly for separable data.
Transition: gradient descent uses this curvature bound directly.
-->

---

## 4. Gradient descent: the baseline

$$
w_{k+1}=w_k-\alpha_k\nabla F(w_k),\qquad w_0=0.
$$

- $k$ indexes updates; start with $\alpha_k=1/L$, with backtracking if needed.
- Stop when $\|\nabla F(w_k)\|_2\leq\texttt{tol}$ or the update budget is exhausted.
- Dense gradient cost: $O(nd)$; no Hessian matrix is formed.
- Fixed $1/L$ step: smooth convex objective gap $O(1/k)$.
- With $\lambda>0$: geometric bound proportional to $(1-\lambda/L)^k$.

Source: Nesterov, 2004.

<!-- Speaker notes | 2:50-4:00 | 70 seconds
Start from zero and move opposite the gradient. The implementation takes the
objective and gradient from the shared problem module, so it solves the same
problem as the other methods. Its default step starts at one over the global
Lipschitz bound; Armijo backtracking can shorten it to enforce sufficient
decrease. Each gradient uses matrix-vector products, and objective logging
requires another data pass. We include the initial point in the history and
time the spectral-norm setup as well as the iterations. Explain that a rate
describes the objective gap, F at the current weights minus F at the optimum.
For the fixed theoretical step, positive regularization permits geometric
convergence. Its factor approaches one when L is much larger than lambda.
Transition: acceleration tries to reduce the number of such data passes.
-->

---

## 5. Acceleration and proximal structure

- Nesterov uses an extrapolated point to guide the next update.
- Classical smooth-convex objective gap: $O(1/k^2)$ with the prescribed schedule.
- FISTA applies acceleration to a split $F=f+g$.
- Here $f$ is logistic loss and $g(w)=\lambda\|w\|_2^2/2$:
  $\operatorname{prox}_{\alpha g}(v)=v/(1+\alpha\lambda)$, for step $\alpha>0$ and vector $v$.
- L2 shrinks smoothly; L1 soft-thresholding can create exact zeros.

Sources: Nesterov, 2004; Beck and Teboulle, 2009.

<!-- Speaker notes | 4:00-5:05 | 65 seconds
Acceleration adds a history-dependent extrapolation. Its classical rate improves
the dependence on the iteration count, but the step and momentum schedule still
matter. Objective values can temporarily rise during an accelerated run. FISTA
separates a differentiable term from a term handled by a proximal map. In this
project that map is simply division by one plus alpha times lambda. It is not
the soft-thresholding operator associated with an L1 penalty. Since our whole
objective is smooth, FISTA and accelerated gradient are related methods. We are
comparing their particular splitting and schedules, not unrelated model classes.
Transition: ADMM separates the two terms using a constraint instead.
-->

---

## 6. ADMM and the cost of an iteration

- Introduce $z \in \mathbb{R}^d$ and solve $\min_{w,z} f(w)+g(z)$ subject to $w=z$.
- ADMM alternates primal subproblems and a dual update.
- The quadratic regularizer gives a closed-form subproblem.
- The logistic subproblem generally needs an inner iterative solve.
- Report wall time, inner work, and consensus residuals alongside outer iterations.

Source: Boyd et al., 2011.

<!-- Speaker notes | 5:05-6:00 | 55 seconds
The equality constraint ensures the split problem represents the original
objective. ADMM alternates between the two parameter vectors and updates a dual
variable to enforce agreement. The regularization update is inexpensive, but
the logistic update generally takes several inner steps. One ADMM iteration
therefore cannot be treated as equivalent to one GD iteration. The penalty
parameter and inner-solver tolerance influence both speed and accuracy. The
distributed motivation in the literature does not establish a speedup for our
single-machine implementation. We need actual timings and residual checks.
Transition: finish by specifying what evidence makes the comparison meaningful.
-->

---

## 7. What will make the comparison fair?

- Fix data splits, preprocessing, $\lambda$, and initial weights.
- Plot objective versus updates and elapsed wall time.
- Report final objective, training/test accuracy, and parameter norm.
- Check stationarity; also check ADMM primal and dual residuals.
- Full-batch scope: stochastic scalability requires separate experiments.

Sources: Bottou, 2010; Bottou, Curtis, and Nocedal, 2018.

<!-- Speaker notes | 6:00-7:00 | 60 seconds
The stochastic-gradient literature explains why a full pass can be too expensive
for very large datasets. Our manageable full-batch setting instead lets us
compare deterministic convergence. Hold the data and model choices fixed. Use
time curves because setup costs and inner solves differ. A numerical reference
optimum is approximate and should come with a tolerance. Accuracy answers a
different question: score signs can stay unchanged while the objective improves,
and a lower training objective need not improve held-out accuracy. Check the
same stationarity criterion when possible, while retaining ADMM residuals.
Handoff: with the objective and comparison criteria established, the next
segment can explain the detailed algorithm updates and experimental protocol.
-->
