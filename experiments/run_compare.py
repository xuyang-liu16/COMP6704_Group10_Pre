"""Compare GD / Nesterov / FISTA / ADMM on the shared convex problem."""
from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import load_problem
from metrics import pack_history
from problem import grad

OUT = ROOT / "results"
FIG = ROOT / "figures"
METHODS = ("gd", "nesterov", "fista", "admm")


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _validated_history(w, hist, Xtr, ytr, Xte, yte, lam, tol):
    """Check curves and recompute endpoint metrics from the returned weights."""
    w = np.asarray(w, dtype=float)
    if w.shape != (Xtr.shape[1],) or not np.isfinite(w).all():
        raise ValueError("returned w has invalid shape or nonfinite entries")
    hist = dict(hist)
    values = np.asarray(hist["objective"], dtype=float)
    times = np.asarray(hist["time_s"], dtype=float)
    if (values.ndim != 1 or not values.size or times.shape != values.shape
            or not np.isfinite(values).all() or not np.isfinite(times).all()
            or np.any(times < 0) or np.any(np.diff(times) < 0)):
        raise ValueError("objective/time_s must be finite, aligned, nonempty with monotone time")
    start = 0 if hist.get("n_iter") == len(values) - 1 else 1
    iteration = np.asarray(hist.get("iteration", np.arange(start, len(values) + start)))
    if (iteration.shape != values.shape or not np.isfinite(iteration).all()
            or np.any(iteration < 0) or np.any(iteration != np.floor(iteration))
            or np.any(np.diff(iteration) <= 0)):
        raise ValueError("iteration must contain aligned, increasing nonnegative integers")
    measured = pack_history(values, times, w, Xtr, ytr, Xte, yte, lam)
    if not np.isclose(values[-1], measured["final_obj"], rtol=1e-7, atol=1e-10):
        raise ValueError("last objective does not match returned w")
    hist.update(measured)
    hist["iteration"] = iteration.astype(int).tolist()
    hist["n_iter"] = int(iteration[-1])
    hist["final_grad_norm"] = float(np.linalg.norm(grad(w, Xtr, ytr, lam)))
    hist["gradient_converged"] = bool(tol > 0 and hist["final_grad_norm"] <= tol)
    return hist


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("synthetic", "breast_cancer"), default="synthetic")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lam", type=float, default=1e-2)
    parser.add_argument("--max-iter", type=int, default=400)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--figure-dir", type=Path, default=FIG)
    args = parser.parse_args(argv)
    if not np.isfinite(args.lam) or args.lam < 0:
        parser.error("--lam must be finite and nonnegative")
    if not np.isfinite(args.tol) or args.tol < 0 or args.max_iter < 0:
        parser.error("--tol must be finite and nonnegative; --max-iter must be nonnegative")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be in [0, 2**32)")
    Xtr, ytr, Xte, yte = load_problem(args.dataset, seed=args.seed)
    config = {
        "dataset": args.dataset, "seed": args.seed, "lam": args.lam,
        "max_iter": args.max_iter, "tol": args.tol,
        "n_train": len(ytr), "n_test": len(yte), "n_features": Xtr.shape[1],
    }
    summary = {}
    curves = {}
    for name in METHODS:
        t0 = time.perf_counter()
        try:
            # Import gd/nesterov/fista/admm separately: one unfinished module
            # must not prevent the other owners' methods from running.
            fn = importlib.import_module(f"algorithms.{name}").run
            t0 = time.perf_counter()
            w, hist = fn(Xtr, ytr, lam=args.lam, max_iter=args.max_iter,
                         tol=args.tol, X_test=Xte, y_test=yte)
            wall_s = time.perf_counter() - t0
            hist = _validated_history(w, hist, Xtr, ytr, Xte, yte, args.lam, args.tol)
            hist.update(wall_s=wall_s, ok=True, status="ok", config=config)
            hist["n_obj"] = len(hist["objective"])
            json.dumps(hist, default=_json_default, allow_nan=False)
            curves[name] = {key: hist[key] for key in ("objective", "time_s", "iteration")}
            summary[name] = hist
            print(name, "obj", hist["final_obj"], "acc", hist["test_acc"], "ok", True)
        except Exception as e:
            summary[name] = {
                "ok": False, "error": repr(e), "config": config,
                "status": "not_implemented" if isinstance(e, NotImplementedError) else "error",
                "wall_s": time.perf_counter() - t0,
            }
            print(name, "FAIL", e)
    args.out.mkdir(parents=True, exist_ok=True)
    for filename, payload in (("summary.json", summary), ("curves.json", curves)):
        (args.out / filename).write_text(
            json.dumps(payload, indent=2, default=_json_default, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    lines = [
        "# Optimization comparison", "",
        f"Dataset: `{args.dataset}`; seed: {args.seed}; lambda: {args.lam:g}; "
        f"budget: {args.max_iter}; tolerance: {args.tol:g}.",
        f"Train/test: {len(ytr)}/{len(yte)} rows; {Xtr.shape[1]} parameters including regularized bias.",
        "",
        "OK denotes a valid return; gradient convergence is checked separately.",
        "Wall time includes solver setup and logging, excluding data loading and module imports.",
        "",
        "| Method | Status | Final obj | Gradient norm | Grad <= tol | Train acc | Test acc | Wall (s) | Iters |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for name, h in summary.items():
        if not h.get("ok"):
            error = h["error"].replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {name} | {h['status']}: {error} | - | - | - | - | - | {h['wall_s']:.6f} | - |")
        else:
            lines.append(
                f"| {name} | OK | {h['final_obj']:.10f} | {h['final_grad_norm']:.3e} | "
                f"{'yes' if h['gradient_converged'] else 'no'} | {h['train_acc']:.4f} | "
                f"{h['test_acc']:.4f} | {h['wall_s']:.6f} | {h['n_iter']} |"
            )
    (args.out / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", args.out / "summary.json", args.out / "table.md")
    if curves:
        from scripts.make_figures import main as make_figures
        make_figures(["--input", str(args.out / "summary.json"),
                      "--output-dir", str(args.figure_dir)])
    else:
        print("No successful methods; no figures generated.")
    return summary


if __name__ == "__main__":
    main()
