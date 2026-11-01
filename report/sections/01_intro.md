# 1. Introduction and background

<!-- Section owner: WANG Zeqing, 26029011r. -->

## Convex model training

A binary classifier assigns a label from measured features, such as the attributes of a tumor sample. Training selects model parameters by minimizing a loss over observed examples. The resulting predictions depend on both the statistical model and the numerical procedure used to fit it. Convex empirical risk minimization makes this distinction useful: every local minimum is global, so algorithm comparisons can focus on the cost of approaching a common optimum. Boyd and Vandenberghe [2004] develop the convex analysis and optimality conditions that support this viewpoint.

L2-regularized logistic regression provides a tractable setting for studying these costs. It combines a probabilistic linear classifier with a differentiable convex loss and a quadratic penalty. The model supports efficient gradient evaluations while retaining practical issues such as correlated features, unequal feature scales, and sensitivity to regularization. Our project compares gradient descent, Nesterov accelerated gradient, the fast iterative shrinkage-thresholding algorithm, or FISTA, and the alternating direction method of multipliers, or ADMM. The comparison asks how their update rules affect convergence and computational work on the same training objective.

## Geometry of logistic regression

Let $X \in \mathbb{R}^{n \times d}$ contain $n$ training examples, with row $x_i^\top$, and let $y_i \in \{-1,+1\}$ denote each label. For parameters $w \in \mathbb{R}^d$ and regularization strength $\lambda \geq 0$, we minimize

$$
F(w) = \frac{1}{n}\sum_{i=1}^{n}
\log\!\left(1+\exp(-y_i x_i^\top w)\right)
+ \frac{\lambda}{2}\|w\|_2^2.
$$

The sigmoid $\sigma(t)=1/(1+\exp(-t))$ maps a score to the modeled probability of label $+1$. The logistic term is the negative log-likelihood of the observed labels. Its second derivative with respect to the score lies between zero and $1/4$. Consequently, the Hessian satisfies

$$
\nabla^2 F(w)=\frac{1}{n}X^\top D(w)X+\lambda I,
\qquad
\lambda I \preceq \nabla^2 F(w)
\preceq \left(\frac{\|X\|_2^2}{4n}+\lambda\right)I.
$$

Here $D(w)$ is diagonal with entries $\sigma(x_i^\top w)[1-\sigma(x_i^\top w)]$, $I$ is the $d$-dimensional identity, and $\preceq$ denotes positive-semidefinite ordering. The matrix norm $\|X\|_2$ is the spectral norm. These bounds establish convexity and a global gradient Lipschitz bound $L=\|X\|_2^2/(4n)+\lambda$. For $\lambda>0$, the objective is strongly convex with parameter at least $\lambda$ and has a unique minimizer, denoted $w^\star$.

Regularization controls parameter magnitude and supplies curvature in directions that the data constrain weakly. It also prevents the diverging parameter sequences that can occur for unregularized logistic regression on strictly separable data. Our implementation appends an intercept column and penalizes every coordinate, including that intercept; the strong-convexity statement therefore applies to the entire vector. Feature standardization changes the curvature seen by the optimizer. It should use training-set statistics and remain identical across solvers.

## First-order methods

Gradient descent uses the update $w_{k+1}=w_k-\nabla F(w_k)/L$, where $k$ indexes iterations. It requires matrix-vector products and avoids constructing a dense Hessian. For dense data, a gradient costs $O(nd)$ arithmetic operations, with $O(n+d)$ auxiliary storage beyond the dataset. Nesterov [2004] gives the classical $O(1/k)$ bound on the objective gap $F(w_k)-F(w^\star)$ for smooth convex optimization. With $\lambda>0$, the fixed-step method has a geometric bound proportional to $(1-\lambda/L)^k$. Thus, slow progress can reflect a large curvature ratio $L/\lambda$, even when each update is inexpensive. Our implementation starts with the theoretical step and supports Armijo backtracking to enforce sufficient decrease.

Nesterov acceleration introduces an extrapolated point whose gradient influences the next iterate. The classical smooth-convex guarantee improves the objective-gap rate to $O(1/k^2)$; variants tuned to strong convexity improve the dependence on the curvature ratio. These guarantees require the corresponding parameter schedules and assumptions. Momentum can produce nonmonotone objective trajectories, so a temporary increase alone does not demonstrate divergence. Measuring elapsed time alongside iteration count tests whether the reduction in updates offsets the additional computation.

Beck and Teboulle [2009] extend acceleration to composite objectives through FISTA. Writing $F=f+g$, with $f$ the mean logistic loss and $g(w)=\lambda\|w\|_2^2/2$, yields the proximal map $\operatorname{prox}_{\alpha g}(v)=v/(1+\alpha\lambda)$ for step size $\alpha>0$ and input vector $v$. This is smooth shrinkage; it does not create the exact zeros associated with L1 soft-thresholding. The relevant smoothness bound for the forward step is that of $f$. Since both terms are differentiable here, FISTA and accelerated gradient are closely related; their comparison isolates the choice of splitting and update schedule.

Boyd et al. [2011] survey ADMM for separable convex problems and distributed statistical learning. A consensus split introduces a second parameter vector $z$ and the constraint $w=z$, separating the logistic loss from the quadratic penalty. The method alternates primal subproblems with a dual update. The regularization subproblem has a closed-form solution, while the logistic subproblem generally requires an iterative solver. Its inner iterations, stopping accuracy, and penalty parameter affect runtime. A count of outer ADMM iterations therefore does not measure the same work as a count of gradient evaluations.

## Experimental perspective

Bottou [2010] and Bottou, Curtis, and Nocedal [2018] explain why stochastic gradients are important when full passes through large datasets become costly. Their analyses distinguish optimization accuracy from statistical accuracy under finite computational budgets. We use full-batch solvers to study deterministic convergence on manageable datasets. This scope supports controlled comparisons of acceleration and splitting; it leaves stochastic sampling and distributed scalability for separate experiments.

A fair evaluation holds the objective, regularization, preprocessing, data split, and initialization fixed. Objective traces describe progress toward the training optimum; wall-clock traces account for setup, logging, and inner solves. A numerical reference value can support an empirical objective-gap plot, provided its solver tolerance is reported and the value is identified as approximate. Gradient norms provide a common stationarity diagnostic, while ADMM also requires primal and dual residual checks. Training and test accuracy describe predictive behavior, and the parameter norm records the scale of the fitted model.

These measurements answer different questions. Classification accuracy can remain unchanged while the logistic objective improves because the predicted labels depend only on score signs. A lower training objective also does not guarantee better held-out accuracy. We therefore assess convergence, computational efficiency, and predictive performance separately, and reserve algorithm rankings for the measured results. Repeated timings and sensitivity to $\lambda$ help determine whether observed differences persist across runs and conditioning regimes.

## References

- Beck, Amir, and Marc Teboulle. 2009. A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems. *SIAM Journal on Imaging Sciences*. DOI: `10.1137/080716542`.
- Bottou, Leon. 2010. Large-Scale Machine Learning with Stochastic Gradient Descent. In *Proceedings of COMPSTAT'2010*. DOI: `10.1007/978-3-7908-2604-3_16`.
- Bottou, Leon, Frank E. Curtis, and Jorge Nocedal. 2018. Optimization Methods for Large-Scale Machine Learning. *SIAM Review*. DOI: `10.1137/16M1080173`.
- Boyd, Stephen, Neal Parikh, Eric Chu, Borja Peleato, and Jonathan Eckstein. 2011. Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers. *Foundations and Trends in Machine Learning*. DOI: `10.1561/2200000016`.
- Boyd, Stephen, and Lieven Vandenberghe. 2004. *Convex Optimization*. Cambridge University Press. DOI: `10.1017/CBO9780511804441`.
- Nesterov, Yurii. 2004. *Introductory Lectures on Convex Optimization: A Basic Course*. Kluwer Academic Publishers. DOI: `10.1007/978-1-4419-8853-9`.
