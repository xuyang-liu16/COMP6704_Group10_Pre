# COMP6704 Group 10: English speaking scripts

32 visible slides, 2 hidden Q&A backups. Talk: 30 minutes, followed by 10 minutes of Q&A.

| Part | Speaker | Slides | Duration | Clock |
| --- | --- | --- | --- | --- |
| Opening | LIU Xuyang | 1 | 0m 25s | 00:00-00:25 |
| Background and shared formulation | WANG Zeqing | 2-6 | 3m 55s | 00:25-04:20 |
| Four-method overview | LIU Xuyang | 7 | 0m 40s | 04:20-05:00 |
| Gradient Descent | WANG Zeqing | 8-11 | 3m 0s | 05:00-08:00 |
| Nesterov | LIU Xuyang | 12-15 | 4m 0s | 08:00-12:00 |
| FISTA and equivalence | WU Aoqi | 16-19 | 4m 0s | 12:00-16:00 |
| ADMM | ZHAO Jixin | 20-23 | 4m 0s | 16:00-20:00 |
| Experiments and comparison | ZHAO Jixin | 24-31 | 9m 0s | 20:00-29:00 |
| Closing | LIU Xuyang | 32 | 1m 0s | 29:00-30:00 |

Each member presents one solver. WANG owns the shared formulation, WU owns the Nesterov-FISTA equivalence, and ZHAO owns all experiments and limitations. LIU gives the opening, overview and closing.

Method handoffs: 08:00, 12:00, 16:00. Experiments begin at 20:00. Closing begins at 29:00. Bracketed cues are actions, not spoken words. Hidden slides 33-34 carry no planned talk time.

### Slide 1: Convex Optimization for Machine Learning

Time: 00:00-00:25. Speaker: LIU Xuyang. Duration: 25s.

We use four methods, GD, Nesterov, FISTA and ADMM, to train the same model. Each of us presents one.

Good evening. We are Group 10. I am LIU Xuyang, with WANG Zeqing, WU Aoqi and ZHAO Jixin. Our model is L2-regularized logistic regression. Zeqing will introduce the classification problem and our shared setup.

### Slide 2: Binary classification with L2 regularization

Time: 00:25-01:00. Speaker: WANG Zeqing. Duration: 35s.

I am WANG Zeqing. Our task is binary classification: use measured features to predict one of two labels. Logistic regression assigns a linear score and converts it to a probability. We learn the weights by minimizing average logistic loss with an L2 penalty. The model stays fixed throughout this comparison. Only the solver changes. We measure both training precision and accuracy on separate test data.

### Slide 3: One objective for four solvers

Time: 01:00-01:50. Speaker: WANG Zeqing. Duration: 50s.

This is the objective that all four methods share.

Capital F is the average logistic loss f plus the L2 penalty g. Each input x has a label y equal to minus one or plus one. The score is x transpose w. A score that supports the correct label gives a smaller logistic loss.

[Point to the two terms.]

The penalty is lambda over two times the squared weight norm. We fix lambda at zero point zero one. The bias is an extra coordinate, and we penalize it too.

Every solver uses this exact objective, so we can compare their progress toward the same answer.

### Slide 4: Predictions and the full gradient

Time: 01:50-02:40. Speaker: WANG Zeqing. Duration: 50s.

The sigmoid turns each score into a probability p. For this formula, b rewrites the label as zero or one.

[Point to p minus b, then X transpose.]

The difference p minus b measures the prediction error in probability. Multiplying by X transpose combines the examples, and dividing by n takes the average. Adding lambda w gives the gradient of the complete objective.

GD moves against this vector. Nesterov also differentiates the full objective, while FISTA handles the penalty separately.

For class prediction, a nonnegative score means plus one. One dense gradient costs order n times d. We optimize the training objective and evaluate accuracy on separate test data.

### Slide 5: Strong convexity and a unique minimizer

Time: 02:40-03:30. Speaker: WANG Zeqing. Duration: 50s.

The Hessian H describes how the gradient changes. Logistic curvature lies between zero and one quarter for each example.

[Point to the lower and upper matrix bounds.]

Adding lambda I gives positive curvature in every direction, including the bias. The objective is strongly convex with mu equal to lambda, so it has a unique minimizer.

The upper bound L limits how quickly the full gradient changes. We obtain it by adding lambda to the loss-only bound L f.

GD and Nesterov use one over L. FISTA uses one over L f before its separate penalty step. These bounds therefore give the four implementations a common mathematical starting point.

### Slide 6: A common convergence certificate

Time: 03:30-04:20. Speaker: WANG Zeqing. Duration: 50s.

Our common stopping test uses the norm of the full gradient at the returned weights. The target is ten to the minus six.

Strong convexity bounds the objective gap by the squared gradient norm divided by twice lambda. Passing our test therefore certifies an objective gap at most five times ten to the minus eleven.

[Point to the certificate, then the update budget.]

We cap every run at four hundred updates. A run can exhaust that budget while already making useful predictions. Equal test accuracy alone does not certify convergence.

ADMM adds residual and inner-solve checks, which Jixin will explain. Xuyang will give the four-method overview before I return to gradient descent.

### Slide 7: Contents: four methods, four speakers

Time: 04:20-05:00. Speaker: LIU Xuyang. Duration: 40s.

[Point to the four rows.]

Each member presents one solver. Zeqing presents GD, and I present Nesterov. Aoqi presents FISTA and explains its equivalence to Nesterov under our settings. Jixin presents ADMM and all experimental comparisons. I return for the closing. Each algorithm has four slides. We keep the same model, starting weights and convergence target throughout. Zeqing, please continue with GD.

### Slide 8: GD: a full-gradient baseline

Time: 05:00-05:45. Speaker: WANG Zeqing. Duration: 45s.

My solver is full-batch gradient descent. It starts with zero weights and subtracts a scaled full gradient.

The input is the same standardized training data and lambda used by the other solvers. We initialize the weights at zero.

At each accepted point, we check the full gradient norm. If the target has not been met, we form a candidate using the negative gradient. Backtracking can require several objective evaluations.

### Slide 9: GD: update and backtracking

Time: 05:45-06:30. Speaker: WANG Zeqing. Duration: 45s.

[Point to the update and its step size.]

The initial trial step is one over L. The implementation evaluates the candidate objective and applies the Armijo condition shown here. The required decrease depends on the step size and the squared gradient norm.

If the candidate fails, the code halves the step and tries again. It accepts a weight vector only after that decrease check passes. The counter records accepted updates, so trial evaluations add work without adding an iteration.

For an L-smooth objective, the standard one-over-L step supplies sufficient decrease. Backtracking also protects the implementation when a caller supplies a different trial step.

### Slide 10: GD: convergence rate

Time: 06:30-07:15. Speaker: WANG Zeqing. Duration: 45s.

[Point to the rate bound.]

With this standard step and mu-strong convexity, the objective-error bound contracts by one minus mu over L at each update. If L is large relative to mu, that factor is close to one and the guarantee can be slow.

This gives us a concrete baseline: a simple update, a descent check, and a rate tied to the curvature bounds.

### Slide 11: GD: implementation and work

Time: 07:15-08:00. Speaker: WANG Zeqing. Duration: 45s.

After acceptance, the history records the original objective and cumulative time. It includes initialization, so a history with one hundred thirty-seven points represents one hundred thirty-six updates.

The main dense work is order n times d, plus any trial evaluations and shared setup. GD has no momentum state and needs no inner matrix solve. That makes its behavior easy to audit.

Its limitation is the conservative step set by the global curvature bound. LIU Xuyang will now show how Nesterov uses previous movement to build a look-ahead point.

[Hand over to LIU Xuyang at 08:00.]

### Slide 12: Nesterov: a look-ahead point

Time: 08:00-09:00. Speaker: LIU Xuyang. Duration: 60s.

Thank you. I am LIU Xuyang, and my solver is Nesterov accelerated gradient.

GD evaluates its gradient at the current weights. Nesterov keeps a second vector, v, which looks ahead using recent movement. The gradient step starts from that look-ahead point.

At initialization, both vectors are zero. The first momentum coefficient is zero, so the algorithm starts with an ordinary gradient step. Later coefficients retain more of the recent direction.

[Point to the current weights and the look-ahead point. The displacement uses accepted weights from the two most recent steps.]

### Slide 13: Nesterov: the accelerated update

Time: 09:00-10:00. Speaker: LIU Xuyang. Duration: 60s.

[Point to the first equation, then the momentum equations.]

The scalar t starts at one. The second equation updates it, and the ratio in the final line controls the momentum. We add a multiple of the latest displacement to the accepted weights to form the next v.

Our implementation uses the general-convex schedule shown here. We keep the step at one over L and use neither adaptive restart nor a schedule tuned to strong convexity.

### Slide 14: Nesterov: rate and stopping

Time: 10:00-11:00. Speaker: LIU Xuyang. Duration: 60s.

The familiar one-over-k-squared objective-error rate applies to this schedule. An individual step can still increase the objective because the look-ahead point may overshoot.

First, the theoretical rate concerns the objective gap. Our experiment stops on the full-gradient norm. A faster objective-error bound does not directly determine how many updates are needed to cross a particular gradient threshold.

Third, the basic momentum schedule can oscillate. The recorded objective need not decrease at every update. We therefore retain the common gradient test instead of stopping merely because a plotted curve looks flat.

### Slide 15: Nesterov: work and trade-offs

Time: 11:00-12:00. Speaker: LIU Xuyang. Duration: 60s.

[Point to the update and monitoring rows.]

The update gradient is evaluated at v, while the stopping gradient is evaluated at the accepted w. Our implementation computes both. Each dense evaluation costs order n times d, so the asymptotic order stays the same but the constant work differs from a single-gradient update.

These details help interpret the synthetic result later: GD takes one hundred thirty-six updates and Nesterov takes one hundred thirty-eight. That observation is consistent with the different stopping metric and this particular momentum schedule.

We have used the full objective in one gradient. WU Aoqi will now present FISTA, which keeps the same momentum schedule and handles the penalty through a proximal step.

[Hand over to WU Aoqi at 12:00.]

### Slide 16: FISTA: separate loss and penalty

Time: 12:00-12:40. Speaker: WU Aoqi. Duration: 40s.

I am WU Aoqi, and I present FISTA, the Fast Iterative Shrinkage-Thresholding Algorithm.

The objective still equals f plus g. We differentiate only the logistic loss f, then apply the L2 penalty g through a proximal operation.

We use eta equal to one over L f, the smoothness bound for the loss alone. The penalty enters only through this shrink step. Including lambda v in the gradient as well would count it twice.

[Point to the two terms. The loss gradient and the penalty enter through different operations.]

### Slide 17: FISTA: the L2 proximal step

Time: 12:40-13:30. Speaker: WU Aoqi. Duration: 50s.

[Point to the proximal minimization.]

The proximal problem balances the penalty against the squared distance from a trial point a. For our quadratic penalty, its derivative is lambda w plus w minus a divided by eta. Setting that expression to zero gives w equal to a divided by one plus eta lambda.

Now let a be the look-ahead point minus eta times the loss gradient. Substitution gives the update shown below. The denominator exceeds one, so every coordinate shrinks by the same factor.

### Slide 18: FISTA: acceleration and work

Time: 13:30-14:20. Speaker: WU Aoqi. Duration: 50s.

FISTA uses the same t recurrence and extrapolation as Xuyang's implementation. Its composite objective-error bound also falls as one over k squared.

Each update evaluates the loss gradient at v, applies the closed-form shrinkage, and checks the full gradient at the accepted weights. Dense gradients cost order n times d. Shrinkage and momentum require only vector operations. We use the same general-convex momentum schedule without restart. The rate concerns the composite objective gap, while our stopping rule checks the full gradient.

For this quadratic penalty, the shrink step has a closed form. The next slide shows exactly why our default FISTA and Nesterov iterates coincide.

### Slide 19: Nesterov-FISTA equivalence

Time: 14:20-16:00. Speaker: WU Aoqi. Duration: 100s.

This equivalence follows directly from the update, so we can predict the overlapping curves before looking at the results.

[Follow the four equalities slowly.]

Start with the FISTA step using one over L f. Multiply numerator and denominator by L f. The numerator becomes L f times v minus the loss gradient, and the denominator becomes L f plus lambda.

Rewrite the numerator by adding and subtracting lambda v. This produces v minus a fraction. Its numerator is the loss gradient plus lambda v, and its denominator is L f plus lambda.

The bracketed gradient is the gradient of the full objective. The denominator is L. We have therefore recovered the Nesterov gradient step exactly.

To obtain identical sequences, we also need identical initial weights and the same momentum recurrence. Both conditions hold in our code. The equality then carries from one update to the next, apart from floating-point rounding.

The conclusion depends on this isotropic L2 penalty and these step sizes. It does not establish an equivalence for arbitrary penalties or parameter choices.

ADMM also separates the loss and penalty, but introduces an equality constraint and a dual variable. ZHAO Jixin will explain that method next.

[Hand over to ZHAO Jixin at 16:00.]

### Slide 20: ADMM: two copies of the weights

Time: 16:00-17:00. Speaker: ZHAO Jixin. Duration: 60s.

I am ZHAO Jixin, and my solver is ADMM, the Alternating Direction Method of Multipliers. I will also compare the experimental results.

We give the loss and penalty separate copies of the weights. The loss uses w and the penalty uses z. The constraint w equals z preserves the original optimization problem.

We fix rho at one. Rho controls the equality penalty inside the algorithm, while lambda defines the model's regularization. They have different roles, and the Newton line-search step is another separate quantity.

[Point to w = z. At agreement, the split formulation is the original regularized logistic objective.]

### Slide 21: ADMM: the three updates

Time: 17:00-18:00. Speaker: ZHAO Jixin. Duration: 60s.

[Point to the constraint and then the three updates.]

The first update minimizes the logistic loss plus a quadratic term. That term pulls w toward z minus u. Logistic loss makes this subproblem nonlinear, so our implementation uses a warm-started Newton solve with backtracking.

The second update handles z. Both terms are quadratic in this variable, giving the direct shrinkage formula rho divided by rho plus lambda, multiplied by w plus u.

The third update adds the disagreement w minus z to u. This scaled dual variable carries constraint information into the next iteration. Repeatedly adjusting it encourages the copies to agree.

### Slide 22: ADMM: work inside an outer update

Time: 18:00-19:00. Speaker: ZHAO Jixin. Duration: 60s.

For dense data, GD and the accelerated methods use order n times d work per gradient evaluation. Momentum adds vector operations, and the accelerated implementations also evaluate the full gradient for monitoring.

ADMM's Newton step forms a d-by-d Hessian and solves a linear system. The dense cost is order n times d squared plus d cubed per inner step. One outer update may need several such steps.

[Point to the inner-solve limits.]

Our implementation allows up to fifty Newton steps per outer update and uses an inner gradient tolerance of ten to the minus ten. A limit of fifty does not mean every update uses all fifty.

An ADMM outer update therefore contains a nonlinear inner solve followed by inexpensive vector operations. Even when the outer iteration count is modest, that inner work can dominate elapsed time.

### Slide 23: ADMM: residuals and stopping

Time: 19:00-20:00. Speaker: ZHAO Jixin. Duration: 60s.

To judge convergence, we must check agreement between the copies as well as stationarity of the original objective. This slide specifies those checks.

The primal residual r measures disagreement between w and z. The dual residual s tracks the change in z, scaled by rho. We require both residual checks, a successful inner solve, and the common full-gradient test.

All reported objectives evaluate the original F at the returned w. This keeps the comparison consistent across solvers.

I will now compare the four implementations on the same data, using both update counts and elapsed time.

[Continue as ZHAO Jixin into the experiments at 20:00.]

### Slide 24: Experimental setup

Time: 20:00-20:55. Speaker: ZHAO Jixin. Duration: 55s.

We use two datasets: synthetic classification data and the
breast-cancer dataset included with scikit-learn.

[point to the training and test columns]

We keep seventy-five percent for training and twenty-five percent for
testing. The split keeps similar class proportions.
The synthetic problem has fifty-one weights, including the bias.
The second has thirty-one.

We learn feature scaling from training data only.
All methods use the same split, zero start, penalty and stopping target.
The shared code checks the returned results and calculates the final
scores again.

Our timer covers the solver call, including setup and logging.
It excludes data loading and imports.
These choices tell us what the reported time includes.

### Slide 25: Synthetic: all four reach the target

Time: 20:55-22:10. Speaker: ZHAO Jixin. Duration: 75s.

All four methods pass on the synthetic data.
They reach the same objective, about zero point four nine three two.
Test accuracy is seventy-three point eight percent for every method.

[point to the updates column]

GD needs one hundred thirty-six updates.
Nesterov and FISTA each need one hundred thirty-eight.
ADMM needs one hundred ninety-five outer updates and passes its extra checks.

So GD reaches our target in slightly fewer updates here.
The word accelerated does not promise fewer steps on every dataset.
Our stopping rule checks the gradient, while the rate formula describes
the objective error.

[point to the wall-time column]

Times range from about seven to fifty-five milliseconds.
These are single recorded runs.
The small differences among the first three need repeated measurements
before we can call one reliably faster.

### Slide 26: Synthetic objective versus iteration

Time: 22:10-23:00. Speaker: ZHAO Jixin. Duration: 50s.

The horizontal axis counts updates. The vertical axis shows the original
objective, including the penalty.
Lower values mean a better fit to this training objective.

[point to the axes, then the start of the curves]

Every method starts with zero weights.
At that point, the objective is log two.
The curves drop early and then approach the same level.

Nesterov and FISTA overlap. Aoqi showed that their update formulas match
with our settings, so this is what we expect.

[point to the overlapping curves and the tail]

The vertical axis uses an ordinary linear scale.
Tiny differences near the end are hard to see.
We use the gradient values in the table to decide whether a run has
reached the target.

### Slide 27: Synthetic objective versus time

Time: 23:00-23:55. Speaker: ZHAO Jixin. Duration: 55s.

This plot keeps the same objective values, but changes the horizontal
axis from updates to time.

[point to the time axis and the ADMM curve]

ADMM takes longer in this run because it performs inner Newton solves.
The other three finish close together.
Counting outer updates alone would miss this extra work.

The curve uses timestamps recorded inside the solver.
The table times the complete call from outside.
That includes work at the end, so the two time measurements can differ
slightly.

Checks and logging also take time.
On a small dataset, they can affect the result.
We would repeat these measurements with fixed thread settings before
ranking small differences. For now, this plot describes the saved run.

### Slide 28: Breast cancer: the budget ends first

Time: 23:55-25:10. Speaker: ZHAO Jixin. Duration: 75s.

The second dataset gives a different result.
Every method uses all four hundred updates.
None reaches our gradient target of ten to the minus six.

[point to the updates and gradient columns]

GD has a gradient size of about one point seven times ten to the minus
three. Nesterov and FISTA are around four point two times ten to the
minus five. ADMM is close to those two, but also above the target.

All four still have the same test accuracy: ninety-six point five percent.
Their class predictions can agree even while training is unfinished.

Nesterov and FISTA reach a lower objective than GD within this budget.
That is a useful measured difference, even though none passes the
requested stopping check.

The report also has a longer ADMM check.
It uses a different update budget, so we keep it outside this table.
Here, we compare all four under the same limit.

### Slide 29: Breast-cancer objective versus iteration

Time: 25:10-25:55. Speaker: ZHAO Jixin. Duration: 45s.

The objective drops quickly and then looks almost flat.
Looking at the graph alone, we might think the methods have finished.

[point to the flat tail]

The gradient sizes in the previous table are still above our target.
The scale hides the small changes that remain.

The graph helps us see the broad pattern.
The table answers the more precise question: did the stopping check pass?
We need both views to describe these runs.

A future plot could show the gap from a carefully checked reference
value on a logarithmic scale.
That would make small differences near the end easier to see.

### Slide 30: Comparison and solver choices

Time: 25:55-27:30. Speaker: ZHAO Jixin. Duration: 95s.

[Point to the solver rows.]

All four solvers pass the gradient check on the synthetic data. Nesterov and FISTA overlap because their default updates match for this penalty.

On breast cancer, equal test accuracy appears before any method reaches the requested gradient target. Those runs describe progress within a fixed budget.

ADMM also shows why outer update counts alone miss important work. Its inner solves take time.

GD is a useful starting point for this problem.
Its update is easy to understand, and it solves the synthetic case well.

Nesterov uses recent movement to look ahead.
FISTA handles the penalty with a cheap shrink step.
With our settings, they follow the same path.
So these runs cannot show a difference in their progress toward the answer.

ADMM separates the loss and penalty.
This can help when separate parts are easier to handle.
Here, its inner Newton solve adds work.

[point to the practical conclusion]

We should compare time at the same training precision.
We should also check the work inside each update.
A method with fewer outer updates may still take longer.
The formulas help us understand the methods, and the measured results
show what happens in this setup.

### Slide 31: Limitations and next experiments

Time: 27:30-29:00. Speaker: ZHAO Jixin. Duration: 90s.

Our comparison uses two small datasets, one split per dataset, and one timing per method. The breast-cancer runs exhaust the budget before reaching the common target.

[Point to the limitations.]

The next comparison should give every method enough updates to reach the same precision. We could then measure time to that target.

For timing, we would fix the thread settings used by matrix operations, repeat each run, and report the variation. Additional random splits would test whether the pattern persists.

We could also vary step sizes and rho, or evaluate momentum restart. These are future experiments, not settings used in the results shown today.

The current evidence supports conclusions about these saved runs. It does not establish a universally fastest solver.

LIU Xuyang will close with our main finding and invite questions.

[Hand over to LIU Xuyang at 29:00.]

### Slide 32: Conclusion and questions

Time: 29:00-30:00. Speaker: LIU Xuyang. Duration: 60s.

Thank you. Each of us presented one solver for the same model: Zeqing on GD, myself on Nesterov, Aoqi on FISTA, and Jixin on ADMM.

All four pass the gradient check on the synthetic problem, with different amounts of computational work. On breast cancer, matching test accuracy comes before the training target is met.

The comparison therefore needs a clear optimization certificate as well as prediction accuracy. Runtime must account for the work inside each update.

Thank you for listening. We welcome your questions. Each member can answer questions about their own solver, and Jixin can discuss the shared results.

[Finish at 30:00. Keep this slide visible for Q&A. Open hidden slides only when needed.]

### Slide 33: Backup: ADMM residual thresholds

Time: Q&A only. Speaker: ZHAO Jixin. Duration: Q&A only.

[point to the relevant threshold]

The first check asks whether w and z agree closely enough.
The second checks how much z moves, scaled by rho.
We also require the full gradient check and a successful inner solve.
The final objective uses the returned w.

### Slide 34: Backup: references and code

Time: Q&A only. Speaker: LIU Xuyang. Duration: Q&A only.

These references cover convex optimization, FISTA and ADMM.
Our local repository has the four methods and shared experiment code.
