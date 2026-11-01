"""Owned by 武澳奇 (WU Aoqi, 26027031r). FISTA / proximal gradient for L2-logistic.

The default composite mode applies prox_{eta*(lam/2)||.||^2}(u)=u/(1+eta*lam)
after a logistic-loss gradient step. The optional smooth mode accelerates
the full objective. With their default steps and identical momentum, both
forms generate the same iterates up to floating-point roundoff.
"""
from __future__ import annotations

from time import perf_counter

import numpy as np

from metrics import pack_history
from problem import grad, lipschitz_hint, objective, sigmoid


def _grad_logistic_only(w, X, y):
    p = sigmoid(X @ w)
    target = (y + 1.0) / 2.0
    return (X.T @ (p - target)) / X.shape[0]


def run(X, y, lam: float, max_iter: int = 500, tol: float = 1e-6, **kwargs):
    """Zero-start FISTA; tol=0 disables early stopping.

    History includes iteration zero and setup time. In prox mode the gradient
    contains only logistic loss; smooth mode differentiates the full objective.
    """
    start = perf_counter()
    X_test = kwargs.pop("X_test", None)
    y_test = kwargs.pop("y_test", None)
    step = kwargs.pop("step", None)
    mode = kwargs.pop("mode", "prox")  # prox | smooth
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if kwargs:
        raise TypeError(f"Unexpected FISTA options: {', '.join(sorted(kwargs))}")
    if X.ndim != 2 or 0 in X.shape or y.shape != (X.shape[0],):
        raise ValueError("X must be nonempty (n, d), and y must have shape (n,)")
    if not np.isfinite(X).all() or not np.isin(y, [-1.0, 1.0]).all():
        raise ValueError("X must be finite and y must contain only -1/+1")
    if not np.isfinite(lam) or lam < 0:
        raise ValueError("lam must be finite and nonnegative")
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be finite and nonnegative")
    if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)) or max_iter < 0:
        raise ValueError("max_iter must be a nonnegative integer")
    if mode not in {"prox", "smooth"}:
        raise ValueError("mode must be 'prox' or 'smooth'")
    if (X_test is None) != (y_test is None):
        raise ValueError("Supply X_test and y_test together")
    has_test = X_test is not None and y_test is not None
    if has_test:
        X_test = np.asarray(X_test, dtype=float)
        y_test = np.asarray(y_test, dtype=float)
        if (X_test.ndim != 2 or X_test.shape[0] == 0
                or X_test.shape[1] != X.shape[1] or y_test.shape != (X_test.shape[0],)
                or not np.isfinite(X_test).all() or not np.isin(y_test, [-1.0, 1.0]).all()):
            raise ValueError("Test data must be finite, dimension-compatible, with +/-1 labels")

    w = np.zeros(X.shape[1], dtype=float)
    yk = w.copy()
    t = 1.0
    # The smooth variant must include lam in its gradient Lipschitz bound.
    L = lipschitz_hint(X, lam if mode == "smooth" else 0.0)
    if not np.isfinite(L):
        raise ValueError("Nonfinite Lipschitz bound; rescale X and lam")
    if step is None:
        step = 1.0 / L if L > 0 else 1.0
    if not np.isfinite(step) or step <= 0 or (L > 0 and step * L > 1.0 + 1e-12):
        raise ValueError("step must be finite, positive, and at most 1/L")

    obj_hist = [objective(w, X, y, lam)]
    grad_norm = float(np.linalg.norm(grad(w, X, y, lam)))
    time_hist = [perf_counter() - start]
    converged = tol > 0 and grad_norm <= tol
    for _ in range(max_iter):
        if converged:
            break
        if mode == "smooth":
            g = grad(yk, X, y, lam)
            w_new = yk - step * g
        else:
            g = _grad_logistic_only(yk, X, y)
            u = yk - step * g
            w_new = u / (1.0 + step * lam)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        yk = w_new + ((t - 1.0) / t_new) * (w_new - w)
        w, t = w_new, t_new
        value = objective(w, X, y, lam)
        grad_norm = float(np.linalg.norm(grad(w, X, y, lam)))
        if not np.isfinite(value) or not np.isfinite(grad_norm):
            raise FloatingPointError("FISTA produced a nonfinite objective or gradient")
        obj_hist.append(value)
        time_hist.append(perf_counter() - start)
        converged = tol > 0 and grad_norm <= tol

    hist = pack_history(
        obj_hist,
        time_hist,
        w,
        X,
        y,
        X_test if has_test else X,
        y_test if has_test else y,
        lam,
    )
    if not has_test:
        hist["test_acc"] = None
    hist["n_iter"] = len(obj_hist) - 1
    hist["step"] = float(step)
    hist["mode"] = mode
    hist["iteration"] = list(range(len(obj_hist)))
    hist["final_grad_norm"] = grad_norm
    hist["converged"] = bool(converged)
    hist["stop_reason"] = "gradient_tolerance" if converged else "max_iter"
    return w, hist
