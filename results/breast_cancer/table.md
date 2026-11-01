# Optimization comparison

Dataset: `breast_cancer`; seed: 0; lambda: 0.01; budget: 400; tolerance: 1e-06.
Train/test: 426/143 rows; 31 parameters including regularized bias.

OK denotes a valid return; gradient convergence is checked separately.
Wall time includes solver setup and logging, excluding data loading and module imports.

| Method | Status | Final obj | Gradient norm | Grad <= tol | Train acc | Test acc | Wall (s) | Iters |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| gd | OK | 0.0942636788 | 1.673e-03 | no | 0.9906 | 0.9650 | 0.007959 | 400 |
| nesterov | OK | 0.0941673920 | 4.230e-05 | no | 0.9906 | 0.9650 | 0.013232 | 400 |
| fista | OK | 0.0941673920 | 4.230e-05 | no | 0.9906 | 0.9650 | 0.012638 | 400 |
| admm | OK | 0.0941674245 | 4.500e-05 | no | 0.9906 | 0.9650 | 0.054912 | 400 |
