"""Owned by 刘旭洋 (LIU Xuyang, 26123832r). Nesterov accelerated gradient for L2-logistic."""
from __future__ import annotations

from time import perf_counter

import numpy as np

from metrics import pack_history
from problem import grad, lipschitz_hint, objective


def run(X, y, lam: float, max_iter: int = 500, tol: float = 1e-6, **kwargs):
    """Run general-convex Nesterov acceleration from a zero start.

    Optional kwargs: paired X_test/y_test and a positive step <= 1/L.
    Other kwargs are ignored for compatibility with the common runner.
    History includes initialization and each accepted iterate; time_s
    includes spectral-norm setup and monitoring, but not validation or
    final metric packaging. Missing test data give test_acc=None.
    Stop when the accepted point's gradient norm <= tol; tol=0 disables
    early stopping. Monitoring costs an extra gradient per update.
    """
    X_test = kwargs.pop("X_test", None)
    y_test = kwargs.pop("y_test", None)
    step = kwargs.pop("step", None)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2 or min(X.shape) == 0 or y.shape != (X.shape[0],):
        raise ValueError("X must be nonempty (n, d) and y must have shape (n,)")
    if not np.isfinite(X).all() or not np.isin(y, [-1.0, 1.0]).all():
        raise ValueError("X must be finite and y must contain only -1/+1")
    if not np.isfinite(lam) or lam < 0:
        raise ValueError("lam must be finite and nonnegative")
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 0:
        raise ValueError("max_iter must be a nonnegative integer")
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be finite and nonnegative")
    if (X_test is None) != (y_test is None):
        raise ValueError("Supply X_test and y_test together")
    has_test = X_test is not None and y_test is not None
    if has_test:
        X_test = np.asarray(X_test, dtype=float)
        y_test = np.asarray(y_test, dtype=float)
        if (X_test.ndim != 2 or X_test.shape[0] == 0
                or X_test.shape[1] != X.shape[1]
                or y_test.shape != (X_test.shape[0],)):
            raise ValueError("Test data must have shape (m, d) and (m,) with m > 0")
        if not np.isfinite(X_test).all() or not np.isin(y_test, [-1.0, 1.0]).all():
            raise ValueError("X_test must be finite and y_test must be -1/+1")

    start = perf_counter()
    w = np.zeros(X.shape[1], dtype=float)
    v = w.copy()  # extrapolated point
    t = 1.0
    L = lipschitz_hint(X, lam)
    if not np.isfinite(L):
        raise ValueError("Nonfinite Lipschitz bound; rescale X and lam")
    if step is None:
        step = 1.0 / L if L > 0 else 1.0
    if not np.isfinite(step) or step <= 0 or (L > 0 and step > 1.0 / L):
        raise ValueError("step must be finite, positive, and <= 1/L")

    gradient_norm = float(np.linalg.norm(grad(w, X, y, lam)))
    obj_hist = [objective(w, X, y, lam)]
    time_hist = [perf_counter() - start]
    grad_hist = [gradient_norm]
    for _ in range(max_iter):
        # L == 0 means X == 0 and lam == 0, so every point is optimal.
        if L == 0 or (tol > 0 and gradient_norm <= tol):
            break
        g = grad(v, X, y, lam)
        w_new = v - step * g
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        v = w_new + ((t - 1.0) / t_new) * (w_new - w)
        w, t = w_new, t_new
        gradient_norm = float(np.linalg.norm(grad(w, X, y, lam)))
        obj_hist.append(objective(w, X, y, lam))
        grad_hist.append(gradient_norm)
        time_hist.append(perf_counter() - start)

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
    hist["grad_norm"] = grad_hist
    hist["converged"] = bool(L == 0 or (tol > 0 and gradient_norm <= tol))
    hist["stop_reason"] = "tolerance" if hist["converged"] else "max_iter"
    return w, hist
