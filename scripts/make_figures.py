"""Plot successful solver histories from experiments/run_compare.py."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METHODS = {
    "gd": ("GD", "#4285F4", "o"),
    "nesterov": ("Nesterov", "#DB4437", "s"),
    "fista": ("FISTA", "#0F9D58", "^"),
    "admm": ("ADMM", "#B88600", "D"),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results" / "summary.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    args = parser.parse_args(argv)
    summary = json.loads(args.input.read_text(encoding="utf-8"))
    curves, omitted = {}, []
    for name in METHODS:
        hist = summary.get(name, {})
        if not hist.get("ok", False):
            omitted.append(name)
            print(f"Skipping {name}: {hist.get('error', 'no successful result')}", file=sys.stderr)
            continue
        values = np.asarray(hist["objective"], dtype=float)
        times = np.asarray(hist["time_s"], dtype=float)
        start = 0 if hist.get("n_iter") == len(values) - 1 else 1
        iterations = np.asarray(hist.get("iteration", np.arange(start, len(values) + start)), dtype=float)
        if (values.ndim != 1 or values.size == 0 or times.shape != values.shape
                or iterations.shape != values.shape or not np.isfinite(values).all()
                or not np.isfinite(times).all() or not np.isfinite(iterations).all()
                or np.any(times < 0) or np.any(np.diff(times) < 0)
                or np.any(iterations < 0) or np.any(np.diff(iterations) <= 0)):
            raise ValueError(f"{name}: invalid objective/time/iteration history")
        curves[name] = (iterations, times, values)
    if not curves:
        parser.error("no successful histories to plot")

    # Palatino/STIX and categorical colors adapted from the local plot template.
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Palatino", "TeX Gyre Pagella", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "legend.frameon": False,
    })
    config = summary[next(iter(curves))].get("config", {})
    title = "L2-logistic regression"
    if config:
        title += f"\n{config['dataset']}, lambda = {config['lam']:g}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, suffix, xlabel in [
        (0, "iter", "Iteration"),
        (1, "time", "Cumulative solver time (s)"),
    ]:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for method_index, (name, curve) in enumerate(curves.items()):
            label, color, marker = METHODS[name]
            spacing = max(1, len(curve[2]) // 10)
            ax.plot(curve[index], curve[2], label=label, color=color,
                    linestyle="--" if name == "fista" else "-",
                    marker=marker, markevery=(method_index % spacing, spacing),
                    linewidth=1.7, markersize=4)
        ax.set(xlabel=xlabel, ylabel="Training objective F(w)", title=title)
        ax.legend()
        fig.tight_layout(rect=(0, 0.07 if omitted else 0, 1, 1))
        if omitted:
            fig.text(0.5, 0.025, "Unavailable: " + ", ".join(METHODS[n][0] for n in omitted),
                     ha="center", fontsize=9, color="#555555")
        for extension in ("png", "pdf"):
            path = args.output_dir / f"obj_vs_{suffix}.{extension}"
            fig.savefig(path, dpi=200)
            print(f"Saved {path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
