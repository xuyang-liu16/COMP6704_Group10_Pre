"""L2-logistic ADMM. Owned by 赵纪鑫 (ZHAO Jixin, 26009105r)."""
from __future__ import annotations

from time import perf_counter

import numpy as np
from scipy.linalg import solve
from scipy.special import expit

from metrics import pack_history
from problem import grad, objective


def _logistic_update(w, X, y, center, rho, max_iter, tol):
    """Warm-started Newton solve of loss(w) + rho/2 * ||w-center||^2."""
    n, d = X.shape
    identity = np.eye(d)
    for iteration in range(max_iter + 1):
        margins = y * (X @ w)
        prob = expit(-margins)
        delta = w - center
        gradient = -(X.T @ (y * prob)) / n + rho * delta
        if np.linalg.norm(gradient) <= tol:
            return w, iteration, True
        if iteration == max_iter:
            break
        curvature = prob * expit(margins)
        hessian = (X.T @ (curvature[:, None] * X)) / n + rho * identity
        direction = solve(hessian, -gradient, assume_a="pos")
        value = np.mean(np.logaddexp(0.0, -margins)) + 0.5 * rho * delta @ delta
        slope = gradient @ direction
        step = 1.0
        # Armijo backtracking protects Newton steps on poorly scaled data.
        for _ in range(40):
            candidate = w + step * direction
            offset = candidate - center
            candidate_value = (
                np.mean(np.logaddexp(0.0, -y * (X @ candidate)))
                + 0.5 * rho * offset @ offset
            )
            roundoff = 10.0 * np.finfo(float).eps * max(1.0, abs(value))
            if np.isfinite(candidate_value) and (
                candidate_value <= value + 1e-4 * step * slope + roundoff
            ):
                w = candidate
                break
            step *= 0.5
        else:
            return w, iteration, False
    return w, max_iter, False


def run(X, y, lam: float, max_iter: int = 500, tol: float = 1e-6, **kwargs):
    """Return (w, history) with the shared metrics plus ADMM diagnostics.

    Keyword options: X_test/y_test, rho=1, inner_max_iter=50, inner_tol=1e-10.
    Without a test pair, test_acc is None. History index 0 is the zero start.
    tol controls absolute/relative residual tests and the original gradient
    norm. tol=0 disables early stopping; max_iter=0 returns the zero start.
    The bias column is regularized, matching problem.objective.
    """
    start = perf_counter()
    X_test = kwargs.pop("X_test", None)
    y_test = kwargs.pop("y_test", None)
    rho = float(kwargs.pop("rho", 1.0))
    inner_max_iter = kwargs.pop("inner_max_iter", 50)
    inner_tol = float(kwargs.pop("inner_tol", min(1e-10, tol * 0.1) if tol > 0 else 1e-10))
    if kwargs:
        raise TypeError(f"Unexpected ADMM options: {', '.join(sorted(kwargs))}")
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    if X.ndim != 2 or min(X.shape) == 0 or y.shape != (X.shape[0],):
        raise ValueError("X must be nonempty (n, d) and y must have shape (n,)")
    if not np.isfinite(X).all() or not np.isin(y, [-1.0, 1.0]).all():
        raise ValueError("X must be finite and y must contain only -1/+1")
    if not np.isfinite(lam) or lam < 0:
        raise ValueError("lam must be finite and nonnegative")
    if not np.isfinite(rho) or rho <= 0:
        raise ValueError("rho must be finite and positive")
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be finite and nonnegative")
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 0:
        raise ValueError("max_iter must be a nonnegative integer")
    if not isinstance(inner_max_iter, (int, np.integer)) or inner_max_iter < 1:
        raise ValueError("inner_max_iter must be a positive integer")
    if not np.isfinite(inner_tol) or inner_tol <= 0:
        raise ValueError("inner_tol must be finite and positive")
    has_test = X_test is not None and y_test is not None
    if (X_test is None) != (y_test is None):
        raise ValueError("Supply X_test and y_test together")
    if has_test:
        X_test, y_test = np.asarray(X_test, dtype=float), np.asarray(y_test, dtype=float)
        if (
            X_test.ndim != 2 or X_test.shape[0] == 0
            or X_test.shape[1] != X.shape[1] or y_test.shape != (X_test.shape[0],)
        ):
            raise ValueError("Test data must have shape (m, d) and (m,) with m > 0")
        if not np.isfinite(X_test).all() or not np.isin(y_test, [-1.0, 1.0]).all():
            raise ValueError("X_test must be finite and y_test must contain only -1/+1")

    # Split min loss(w) + lam/2 * ||z||^2 subject to w = z.
    # Scaled ADMM:
    # w <- argmin loss(w) + rho/2 * ||w-z+u||^2
    # z <- rho/(rho+lam) * (w+u), u <- u+w-z.
    # rho is a fixed penalty, distinct from the inner Newton line-search step.
    w = np.zeros(X.shape[1])
    z, u = w.copy(), w.copy()
    obj_hist, time_hist = [objective(w, X, y, lam)], [0.0]
    gradient_norm = float(np.linalg.norm(grad(w, X, y, lam)))
    diagnostics = {
        "primal_residual": [0.0],
        "dual_residual": [0.0],
        "eps_primal": [float(np.sqrt(w.size) * tol)],
        "eps_dual": [float(np.sqrt(w.size) * tol)],
        "grad_norm": [gradient_norm],
        "inner_iterations": [0],
        "inner_converged": [True],
    }
    converged = tol > 0 and gradient_norm <= tol
    for _ in range(max_iter):
        if converged:
            break
        w, inner_steps, inner_ok = _logistic_update(
            w, X, y, z - u, rho, inner_max_iter, inner_tol
        )
        previous_z = z
        z = (rho / (rho + lam)) * (w + u)
        u = u + w - z
        primal = float(np.linalg.norm(w - z))
        dual = float(rho * np.linalg.norm(z - previous_z))
        eps_primal = float(tol * (np.sqrt(w.size) + max(np.linalg.norm(w), np.linalg.norm(z))))
        eps_dual = float(tol * (np.sqrt(w.size) + np.linalg.norm(rho * u)))
        gradient_norm = float(np.linalg.norm(grad(w, X, y, lam)))
        # Evaluate the original objective at the returned w, never the
        # augmented Lagrangian or the infeasible split objective loss(w)+g(z).
        obj_hist.append(objective(w, X, y, lam))
        diagnostics["primal_residual"].append(primal)
        diagnostics["dual_residual"].append(dual)
        diagnostics["eps_primal"].append(eps_primal)
        diagnostics["eps_dual"].append(eps_dual)
        diagnostics["grad_norm"].append(gradient_norm)
        diagnostics["inner_iterations"].append(inner_steps)
        diagnostics["inner_converged"].append(inner_ok)
        time_hist.append(perf_counter() - start)
        converged = bool(
            tol > 0 and inner_ok and primal <= eps_primal
            and dual <= eps_dual and gradient_norm <= tol
        )

    history = pack_history(
        obj_hist, time_hist, w, X, y,
        X_test if has_test else X, y_test if has_test else y, lam,
    )
    if not has_test:
        history["test_acc"] = None
    history.update(diagnostics)
    history.update({
        "n_iter": len(obj_hist) - 1,
        "converged": bool(converged),
        "stop_reason": "tolerance" if converged else "max_iter",
        "rho": rho,
        "inner_tol": inner_tol,
        "inner_max_iter": int(inner_max_iter),
    })
    return w, history
