# Optimization comparison

Dataset: `synthetic`; seed: 0; lambda: 0.01; budget: 400; tolerance: 1e-06.
Train/test: 1500/500 rows; 51 parameters including regularized bias.

OK denotes a valid return; gradient convergence is checked separately.
Wall time includes solver setup and logging, excluding data loading and module imports.

| Method | Status | Final obj | Gradient norm | Grad <= tol | Train acc | Test acc | Wall (s) | Iters |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| gd | OK | 0.4932140405 | 9.813e-07 | yes | 0.7667 | 0.7380 | 0.007084 | 136 |
| nesterov | OK | 0.4932140405 | 8.685e-07 | yes | 0.7667 | 0.7380 | 0.008519 | 138 |
| fista | OK | 0.4932140405 | 8.685e-07 | yes | 0.7667 | 0.7380 | 0.008109 | 138 |
| admm | OK | 0.4932140405 | 9.683e-07 | yes | 0.7667 | 0.7380 | 0.054900 | 195 |
