"""Owned by 王泽卿 (WANG Zeqing, 26029011r). Gradient descent for L2-logistic."""
from __future__ import annotations

from time import perf_counter

import numpy as np

from metrics import pack_history
from problem import grad, lipschitz_hint, objective


def run(X, y, lam: float, max_iter: int = 500, tol: float = 1e-6, **kwargs):
    """Run full-batch GD from zero with a 1/L step and Armijo backtracking.

    ``tol`` bounds the absolute Euclidean gradient norm. History includes
    the initial point and every accepted update. ``time_s`` records cumulative
    wall time including step-size setup and objective logging, excluding final
    accuracy calculations. ``max_iter`` bounds accepted updates.

    Supply both ``X_test`` and ``y_test`` for held-out accuracy; otherwise
    ``test_acc`` is None. An optional positive ``step`` sets the initial
    backtracking step. Other keyword arguments are ignored.
    """
    X_test = kwargs.pop("X_test", None)
    y_test = kwargs.pop("y_test", None)
    step = kwargs.pop("step", None)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2 or min(X.shape) == 0 or y.shape != (X.shape[0],):
        raise ValueError("X must be nonempty 2D and y must have shape (n_samples,)")
    if not np.all(np.isfinite(X)) or not np.all(np.isin(y, (-1.0, 1.0))):
        raise ValueError("X must be finite and y must contain only -1 and +1")
    if not np.isfinite(lam) or lam < 0:
        raise ValueError("lam must be finite and nonnegative")
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 0:
        raise ValueError("max_iter must be a nonnegative integer")
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be finite and nonnegative")
    if (X_test is None) != (y_test is None):
        raise ValueError("X_test and y_test must be supplied together")
    has_test = X_test is not None and y_test is not None
    if has_test:
        X_test = np.asarray(X_test, dtype=float)
        y_test = np.asarray(y_test, dtype=float)
        if (
            X_test.ndim != 2
            or X_test.shape[0] == 0
            or X_test.shape[1] != X.shape[1]
            or y_test.shape != (X_test.shape[0],)
        ):
            raise ValueError("Test data must be nonempty and match training dimensions")
        if not np.all(np.isfinite(X_test)) or not np.all(np.isin(y_test, (-1.0, 1.0))):
            raise ValueError("X_test must be finite and y_test must contain only -1 and +1")

    start = perf_counter()
    w = np.zeros(X.shape[1], dtype=float)
    if step is None:
        L = lipschitz_hint(X, lam)
        if not np.isfinite(L) or L < 0:
            raise ValueError("The Lipschitz bound must be finite and nonnegative")
        step = 1.0 / max(L, 1e-12)
    if not np.isfinite(step) or step <= 0:
        raise ValueError("step must be finite and positive")

    obj_hist = [objective(w, X, y, lam)]
    time_hist = [perf_counter() - start]
    for _ in range(max_iter):
        g = grad(w, X, y, lam)
        if np.linalg.norm(g) <= tol:
            break
        # Armijo backtracking
        eta = step
        f0 = obj_hist[-1]
        for _bt in range(60):
            w_new = w - eta * g
            f_new = objective(w_new, X, y, lam)
            if f_new <= f0 - 1e-4 * eta * float(g @ g):
                break
            eta *= 0.5
        else:
            raise RuntimeError("GD backtracking failed to find a descent step")
        w = w_new
        obj_hist.append(f_new)
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
    return w, hist
