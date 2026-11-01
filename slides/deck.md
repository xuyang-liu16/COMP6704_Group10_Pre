# Full deck outline (ZHAO Jixin assemble)

COMP6704 Group 10: Machine Learning Model Training via Convex Optimization

Tuesday, 3 November 2026. Repository schedule: approximately 20:30 HKT.
Prepared speaking time: 30 minutes. The following 10-minute Q&A is separate.

## Section assembly

| Order | Source | Presenter | Talk interval | Duration |
| --- | --- | --- | --- | --- |
| 01 | [Opening](01_open.md) | LIU Xuyang | 00:00 to 04:00 | 4:00 |
| 02 | [Background](02_background.md) | WANG Zeqing | 04:00 to 11:00 | 7:00 |
| 03 | [Methods](03_methods.md) | WU Aoqi | 11:00 to 19:00 | 8:00 |
| 04 | [Results](04_results.md) | ZHAO Jixin | 19:00 to 27:00 | 8:00 |
| 05 | [Closing](05_close.md) | LIU Xuyang | 27:00 to 30:00 | 3:00 |
| | Total | | | 30:00 |

Assembly status on 1 November 2026: sections 02 and 03 contain seven slides
each, with owner-specified durations of 7 and 8 minutes. Section 04 contains
five result slides and two existing objective figures. Sections 01 and 05
remain TODO placeholders; their slots below are proposals for LIU Xuyang.
The outline preserves each owner's source file and does not replace
`TALK_SCRIPT.md`. The assembled sequence has 23 slides and totals 30 minutes.

## Slide sequence

| Slide | Title | Main visual or content | Presenter | Interval |
| ---: | --- | --- | --- | --- |
| 1 | Convex model training | Title, Group 10, names and student IDs | LIU | 00:00 to 01:00 |
| 2 | Comparison questions | Proposed opening: objective error, time, accuracy | LIU | 01:00 to 04:00 |
| 3 | One model, four optimization strategies | Background 1: fixed model, changing optimizer | WANG | 04:00 to 04:45 |
| 4 | What does training minimize? | Background 2: objective and regularized bias | WANG | 04:45 to 05:45 |
| 5 | Why is the problem convex? | Background 3: curvature and smoothness bound | WANG | 05:45 to 06:50 |
| 6 | Gradient descent: the baseline | Background 4: update, step, and convergence | WANG | 06:50 to 08:00 |
| 7 | Acceleration and proximal structure | Background 5: momentum and L2 shrinkage | WANG | 08:00 to 09:05 |
| 8 | ADMM and the cost of an iteration | Background 6: splitting and inner work | WANG | 09:05 to 10:00 |
| 9 | What will make the comparison fair? | Background 7: controls and metrics | WANG | 10:00 to 11:00 |
| 10 | One objective, four update rules | Methods 1: update comparison | WU | 11:00 to 11:45 |
| 11 | Gradient descent and Nesterov | Methods 2: schedule requires reconciliation below | WU | 11:45 to 12:45 |
| 12 | The L2 proximal operator | Methods 3: closed-form derivation | WU | 12:45 to 13:45 |
| 13 | FISTA pseudo-code | Methods 4: gradient, shrinkage, extrapolation | WU | 13:45 to 15:15 |
| 14 | ADMM uses a constrained split | Methods 5: Newton subproblem and dual update | WU | 15:15 to 16:15 |
| 15 | Convergence and cost | Methods 6: actual schedules and dense work | WU | 16:15 to 17:30 |
| 16 | Protocol and results handoff | Methods 7: settings and stopping checks | WU | 17:30 to 19:00 |
| 17 | Experimental setup and coverage | R1: data split and run-status table | ZHAO | 19:00 to 20:00 |
| 18 | Objective convergence | R2: existing objective-versus-iteration figure | ZHAO | 20:00 to 22:00 |
| 19 | Wall-clock cost | R3: existing objective-versus-time figure | ZHAO | 22:00 to 23:30 |
| 20 | Training and test accuracy | R4: accuracy table; paired bars planned | ZHAO | 23:30 to 25:00 |
| 21 | Stability and ablations | R5: penalty sweep; residual panels planned | ZHAO | 25:00 to 27:00 |
| 22 | Findings and limitations | Proposed closing: measured findings and scope | LIU | 27:00 to 29:00 |
| 23 | Conclusion | Proposed closing: answers to opening questions | LIU | 29:00 to 30:00 |

Source reconciliation before delivery: methods slides 2 and 6 currently show
a strongly convex Nesterov schedule and its linear-rate bound. The actual
`algorithms/nesterov.py` uses
`t_next=(1+sqrt(1+4*t*t))/2` and momentum `(t-1)/t_next`.
WU and LIU should align those slides with the implemented general-convex
schedule or clearly label the fixed-momentum variant as theory-only.
These owner files remain unchanged by this assembly.

Default FISTA uses the logistic/L2 proximal split. With the current steps,
its update equals the implemented Nesterov update, explaining the overlapping
curves. The background introduces concepts; the methods section supplies
equations and implementation details within its existing timebox.

## Speaker transitions

- LIU to WANG at 04:00: introduce the shared objective and why it is convex.
- WANG to WU at 11:00: move from the gradient baseline to acceleration and splitting.
- WU to ZHAO at 19:00: compare the measured cost of those updates.
- ZHAO to LIU at 27:00: summarize what the evidence supports and what remains open.

Durations include these handovers. WU presents the method comparison;
algorithm owners validate their equations and answer implementation questions.

## Evidence required before the oral

- Refresh shared-result numbers from `results/summary.json` after reruns.
  Retain failed or nonconverged rows. All four current synthetic runs pass
  the gradient check; the result section records single-run timings.
- Put dataset, seed, `lam`, step or `rho`, stopping rule, and budget in figure
  captions. Keep terminal accuracy separate from optimization error.
- Use `figures/obj_vs_iter.png` and `figures/obj_vs_time.png` for the current
  raw-objective plots. Render proposed gap, accuracy, and residual panels
  separately and check labels at presentation size.
- Rehearse the timed sequence. Keep single-run ADMM findings labeled if
  repeated timings remain unavailable.
- Use the source sections in numerical order when assembling final slides.
  Preserve the owners' edits when their placeholders are replaced.

## Q&A backup [30:00 to 40:00, outside the talk]

| Question | Supporting material | First responder |
| --- | --- | --- |
| Why is the minimizer unique for positive `lam`? | Hessian and formulation | LIU |
| Which step bound does GD use? | Actual step API and smoothness bound | WANG |
| How do Nesterov and FISTA differ here? | Actual updates and objective split | LIU / WU |
| Why use an inner Newton solve in ADMM? | Logistic subproblem and work count | ZHAO |
| What happens when `rho=lam`? | First subproblem and residual trajectory | ZHAO |
| How reliable are runtime and accuracy differences? | Seeds, repeats, convergence flags | WU / ZHAO |

Optional backup visuals: update equations, full parameter table, numerical
reference checks, breast-cancer validation, and failed-setting diagnostics.
