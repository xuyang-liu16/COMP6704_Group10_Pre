# Methods (WU Aoqi)

Speaker: WU Aoqi, 26027031r. Total speaking time: 8 minutes.

## 1. One objective, four update rules

Time: 0:00 to 0:45.

- \(F(w)=f(w)+\lambda\|w\|^2/2\), where \(f\) is mean logistic loss.
- \(\lambda=0.01\); the bias receives the same penalty.
- GD, Nesterov, FISTA, and ADMM target the same unique minimizer.

Speaker notes:
- Define \(w\) as the classifier weights and \(n\) as the training count.
- Compare the update rules, their cost, and their convergence assumptions.

---

## 2. Gradient descent and Nesterov

Time: 0:45 to 1:45.

- Full-objective smoothness: \(L=\|X\|_2^2/(4n)+\lambda\).
- GD moves along the negative gradient.
- Nesterov evaluates it at a look-ahead point.

```text
GD:
    w_next = w - grad_F(w)/L

Implemented Nesterov, initialized with z = w = 0 and t = 1:
    w_next = z - grad_F(z)/L
    t_next = (1 + sqrt(1 + 4*t*t))/2
    z = w_next + ((t - 1)/t_next)*(w_next - w)
    w = w_next; t = t_next
```

Speaker notes:
- Momentum changes where the gradient is evaluated.
- This implementation uses the general-convex momentum schedule.

---

## 3. The L2 proximal operator

Time: 1:45 to 2:45.

- Split \(F=f+g\), with \(g(w)=\lambda\|w\|^2/2\).
- Differentiate the loss; minimize the quadratic penalty exactly.

$$
\operatorname{prox}_{\eta g}(v)
=\arg\min_u\left\{\frac{\lambda}{2}\|u\|^2+
\frac1{2\eta}\|u-v\|^2\right\}
=\frac{v}{1+\eta\lambda}.
$$

- Shrinkage costs \(O(d)\); the data gradient costs \(O(nd)\).
- L2 shrinkage has no coordinate thresholding or sparse-solution guarantee.

Speaker notes:
- Set the derivative \(\lambda u+(u-v)/\eta\) to zero.
- The objective and regularization remain the same across all methods.

---

## 4. FISTA pseudo-code

Time: 2:45 to 4:15.

```text
w = z = zeros(d); t = 1
Lf = spectral_norm(X)^2/(4*n)
eta = 1/Lf                       # use 1 when Lf = 0
record initial objective and elapsed time

repeat at most 400 times:
    if norm(grad_F(w)) <= 1e-6: stop
    v = z - eta*grad_loss(z)
    w_next = v/(1 + eta*lambda)
    t_next = (1 + sqrt(1 + 4*t*t))/2
    z = w_next + ((t - 1)/t_next)*(w_next - w)
    w = w_next; t = t_next
    record F(w), iteration, and elapsed time
return w and shared history metrics
```

- Apply the penalty once, through the prox.
- The stopping check uses the full gradient at \(w\).
- Objective values may increase during acceleration.

Speaker notes:
- Explain the loss step, shrinkage, and momentum in sequence.
- Adding \(\lambda z\) to the loss gradient would double-count the penalty.
- Setting extrapolation to zero recovers proximal gradient.
- An optional full-gradient mode uses step \(1/(L_f+\lambda)\).
- With the default steps, the prox update equals
  \(z-[\nabla f(z)+\lambda z]/(L_f+\lambda)\).
  Its iterates therefore match the current Nesterov solver up to roundoff.

---

## 5. ADMM uses a constrained split

Time: 4:15 to 5:15.

- Minimize \(f(w)+g(z)\) subject to \(w=z\).
- \(\rho>0\) is the algorithm penalty; \(u\) is the scaled dual variable.

```text
w = inner_solve: min_v f(v) + (rho/2)*norm(v - z + u)^2
z = rho*(w + u)/(rho + lambda)
u = u + w - z
check primal and dual residuals
```

- The loss subproblem needs an inner numerical solve.
- One outer iteration can involve several data passes.

Speaker notes:
- Explain the dual update as enforcing agreement between the copies.
- Distinguish algorithm parameter \(\rho\) from model parameter \(\lambda\).
- Inner tolerances affect both convergence and runtime.

---

## 6. Convergence and cost

Time: 5:15 to 6:30. Let \(\mu=\lambda>0\).

| Method | Guarantee under stated assumptions | Dense update cost |
| --- | --- | --- |
| GD, step \(1/L\) | Objective gap \(O((1-\mu/L)^k)\) | \(O(nd)\) |
| Implemented Nesterov | Objective gap \(O(1/k^2)\) | \(O(nd)\) |
| Implemented FISTA | Composite objective gap \(O(1/k^2)\) | \(O(nd)\) |
| ADMM | Convex convergence with controlled inner solves; ergodic rates concern averages | Depends on inner solver |

- Optimality certificate: \(F(w)-F^\star\le\|\nabla F(w)\|^2/(2\lambda)\).
- Spectral-norm setup and diagnostic passes contribute to runtime.
- Plot objective against iterations and against time.

Speaker notes:
- The FISTA schedule here does not establish a linear-rate claim.
- A strongly convex Nesterov schedule can exploit \(\mu\), but that
  variant is outside this implementation.
- These guarantees do not constitute a measured speed ranking.
- Equal outer iterations do not imply equal work for ADMM.

---

## 7. Protocol and results handoff

Time: 6:30 to 8:00.

- Synthetic: 2,000 x 50; breast cancer: 569 x 30.
- Stratified 75/25 split, seed 0; training-only scaling; regularized bias.
- \(\lambda=0.01\), 400 updates, requested tolerance \(10^{-6}\).
- Record objective, gradient norm, train/test accuracy, and time.
- `summary.json` retains histories; `table.md` distinguishes success from convergence.
- Failed or unfinished methods remain visible.

```bash
.venv/bin/python experiments/run_compare.py
.venv/bin/python scripts/make_figures.py
```

Speaker notes:
- All four current methods start at zero.
- Test labels do not guide optimization.
- Timing includes solver setup and logging; dataset loading is excluded.
- Show plots with their current availability labels.
- Explain the expected Nesterov/FISTA curve overlap for this L2 split.
- Single runs establish functionality. Repeated timing trials and
  comparable stopping rules are needed for a speed ranking.
- Hand over to the results speaker for measured values and limitations.
