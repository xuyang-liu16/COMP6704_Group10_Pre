"""Reproducible supplementary experiments; does not modify member solvers.

Run: python experiments/supplement.py
Outputs: results/supplement/ (raw JSON, figures, and review notes).
Gradient tracing is a separate, untimed run. Timing always uses original solvers.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import importlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import scipy
import sklearn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits, threadpool_info
from data import load_problem
from problem import grad, objective
from experiments.run_compare import _validated_history, _json_default

METHODS = ('gd', 'nesterov', 'fista', 'admm')
LABELS = dict(zip(METHODS, ('GD', 'Nesterov', 'FISTA', 'ADMM')))
COLORS = dict(zip(METHODS, ('#0072B2', '#D55E00', '#009E73', '#CC79A7')))


def call(module, data, lam, budget, tol, options):
    X, y, Xt, yt = data
    start = time.perf_counter()
    w, h = module.run(X, y, lam=lam, max_iter=budget, tol=tol,
                      X_test=Xt, y_test=yt, **options)
    elapsed = time.perf_counter() - start
    h = _validated_history(w, h, X, y, Xt, yt, lam, tol)
    if module.__name__.endswith('admm') and h['gradient_converged']:
        assert h['converged'] and h['inner_converged'][-1]
        assert h['primal_residual'][-1] <= h['eps_primal'][-1]
        assert h['dual_residual'][-1] <= h['eps_dual'][-1]
    return w, h, elapsed


def experiment(dataset, seed, lam, budget, tol, repeats, rho=1., trace=False,
               methods=METHODS):
    data = load_problem(dataset, seed)
    modules = {m: importlib.import_module('algorithms.' + m) for m in methods}
    options = {m: ({'rho': rho} if m == 'admm' else {}) for m in methods}
    for m in methods:  # warm-up excluded from statistics
        call(modules[m], data, lam, budget, tol, options[m])
    runs = {m: [] for m in methods}
    rng = np.random.default_rng(7301 + seed)
    for rep in range(repeats):
        for m in rng.permutation(methods):
            w, h, elapsed = call(modules[m], data, lam, budget, tol, options[m])
            runs[m].append((elapsed, w, h))
    result = {}
    for m in methods:
        times = np.array([r[0] for r in runs[m]])
        representative = int(np.argsort(times)[len(times)//2])
        elapsed, w, h = runs[m][representative]
        assert all(r[2]['n_iter'] == h['n_iter'] for r in runs[m])
        assert all(np.allclose(r[1], w, atol=1e-12, rtol=1e-12) for r in runs[m])
        if trace:
            # Every solver logs the original objective once per accepted iterate.
            # Capture those points in an untimed replay; no tracing in timed runs.
            original = modules[m].objective
            gradients = []
            def traced(point, X, y, reg):
                gradients.append(float(np.linalg.norm(grad(point, X, y, reg))))
                return original(point, X, y, reg)
            modules[m].objective = traced
            try:
                wt, ht, _ = call(modules[m], data, lam, budget, tol, options[m])
            finally:
                modules[m].objective = original
            # GD trial objective calls could include rejected line-search steps.
            # Its default 1/L step must accept every trial for this trace to align.
            assert len(gradients) == len(h['objective']), (m, 'trace alignment')
            assert np.allclose(wt, w, atol=1e-12, rtol=1e-12)
            assert np.allclose(ht['objective'], h['objective'], atol=1e-12)
            assert np.isclose(gradients[-1], h['final_grad_norm'])
            h['grad_norm'] = gradients
        result[m] = {
            'wall_samples_s': times.tolist(),
            'median_ms': float(np.median(times)*1000),
            'q25_ms': float(np.quantile(times,.25)*1000),
            'q75_ms': float(np.quantile(times,.75)*1000),
            'representative_wall_s': elapsed,
            'weights': w.tolist(), 'history': h,
        }
    if 'nesterov' in result and 'fista' in result:
        a, b = result['nesterov'], result['fista']
        assert a['history']['n_iter'] == b['history']['n_iter']
        assert np.allclose(a['history']['objective'], b['history']['objective'], atol=1e-12)
        assert np.allclose(a['weights'], b['weights'], atol=1e-10)
    return dict(dataset=dataset, seed=seed, lam=lam, budget=budget, tol=tol,
                repeats=repeats, rho=rho, methods=result)


def savefig(fig, path):
    fig.savefig(path.with_suffix('.png'), dpi=220, bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def figures(payload, out):
    plt.rcParams.update({'font.size': 13, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    for case in payload['precision']:
        fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
        for m, r in case['methods'].items():
            h = r['history']
            for ax, x in zip(axs, (h['iteration'], np.array(h['time_s'])*1000)):
                ax.semilogy(x, h['grad_norm'], label=LABELS[m], color=COLORS[m],
                            linestyle='--' if m == 'fista' else '-', linewidth=2,
                            zorder=4 if m == 'fista' else 3)
        for ax, label in zip(axs, ('Accepted / outer updates', 'Internal elapsed time (ms)')):
            ax.axhline(case['tol'], color='#555555', ls=':', label='Stopping target')
            ax.set(xlabel=label, ylabel='Full gradient norm')
            ax.grid(alpha=.2, which='both')
        axs[0].legend(fontsize=10)
        fig.suptitle(case['dataset'].replace('_',' ').title() + ': convergence to the same gradient target')
        savefig(fig, out / (case['dataset'] + '_gradient'))
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for ax, case in zip(axs, payload['precision']):
        for i, (m, r) in enumerate(case['methods'].items()):
            ax.bar(i, r['median_ms'], color=COLORS[m], width=.6)
            ax.errorbar(i, r['median_ms'], yerr=[[r['median_ms']-r['q25_ms']],
                        [r['q75_ms']-r['median_ms']]], color='black', capsize=5)
        ax.set(xticks=range(4), xticklabels=[LABELS[m] for m in METHODS],
               ylabel='Complete solver wall time (ms)', title=case['dataset'].replace('_',' ').title())
        ax.grid(axis='y',alpha=.2)
    fig.suptitle('Same gradient target: median and interquartile range (20 runs)')
    savefig(fig, out/'timing')
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for ax, dataset in zip(axs, ('synthetic','breast_cancer')):
        cases = [c for c in payload['rho'] if c['dataset']==dataset]
        ax.plot([c['rho'] for c in cases], [c['methods']['admm']['median_ms'] for c in cases], 'o-', color=COLORS['admm'])
        for c in cases:
            r = c['methods']['admm']; h = r['history']
            ax.annotate(str(h['n_iter']) + ('*' if not h['gradient_converged'] else ''),
                        (c['rho'],r['median_ms']), xytext=(4,7),textcoords='offset points')
        ax.set(xscale='log', yscale='log', xlabel='ADMM penalty rho',
               ylabel='Median solver wall time (ms)', title=dataset.replace('_',' ').title())
        ax.grid(alpha=.2,which='both')
    fig.suptitle('ADMM penalty sensitivity (5 repeats; labels show outer updates)')
    savefig(fig,out/'admm_rho')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'results'/'supplement')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    payload = {'precision': [], 'rho': [], 'splits': [], 'baseline_400': []}
    with threadpool_limits(limits=1):
        payload['environment'] = dict(python=sys.version, platform=platform.platform(),
            numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__,
            matplotlib=matplotlib.__version__, threadpools=threadpool_info(),
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
        for ds in ('synthetic', 'breast_cancer'):
            payload['baseline_400'].append(experiment(ds,0,.01,400,1e-6,1))
            payload['precision'].append(experiment(ds,0,.01,10000,1e-6,20,trace=True))
            print(ds, 'precision complete', flush=True)
            for rho in (.01,.1,1.,10.):
                payload['rho'].append(experiment(ds,0,.01,10000,1e-6,5,rho=rho,methods=('admm',)))
            print(ds, 'rho complete', flush=True)
            for seed in range(1,5):
                payload['splits'].append(experiment(ds,seed,.01,10000,1e-6,3))
            print(ds, 'seeds complete', flush=True)
    (args.out/'raw.json').write_text(json.dumps(payload,indent=2,default=_json_default,allow_nan=False)+'\n')
    figures(payload,args.out)
    lines = ['# Supplementary experiment results', '',
        'Original member solvers, unchanged. Full-gradient target 1e-6; lambda=0.01; '
        'zero initialization; maximum 10,000 updates; single-thread numerical libraries.', '',
        'Timing: 1 warm-up per configuration; randomized solver order within each repeat. '
        'Full solver call includes setup, logging and final metrics. Data loading, imports, '
        'endpoint verification and gradient tracing are excluded. Error bars are IQR, not confidence intervals.', '',
        'Gradient traces come from a separate replay, checked against the timed trajectory. '
        'Time plots use internal timestamps from the repeat nearest the upper median wall time; '
        'they are representative runs, not pointwise median curves.', '',
        '## Same-precision comparison (seed 0, 20 timed repeats)', '',
        '| Dataset | Method | Updates | Gradient norm | Target met | Median ms | IQR ms | Test accuracy |',
        '|---|---|---:|---:|---|---:|---:|---:|']
    for c in payload['precision']:
        for m,r in c['methods'].items():
            h=r['history']
            lines.append(f"| {c['dataset']} | {m} | {h['n_iter']} | {h['final_grad_norm']:.3e} | {h['gradient_converged']} | {r['median_ms']:.2f} | {r['q25_ms']:.2f}–{r['q75_ms']:.2f} | {100*h['test_acc']:.2f}% |")
    lines += ['', '## ADMM rho sensitivity (seed 0, 5 repeats)', '',
        '| Dataset | rho | Updates | Gradient norm | Target met | Median ms | Inner Newton steps |',
        '|---|---:|---:|---:|---|---:|---:|']
    for c in payload['rho']:
        r=c['methods']['admm']; h=r['history']
        lines.append(f"| {c['dataset']} | {c['rho']} | {h['n_iter']} | {h['final_grad_norm']:.3e} | {h['gradient_converged']} | {r['median_ms']:.2f} | {sum(h['inner_iterations'])} |")
    lines += ['', '## Seed sensitivity', '',
        'Seeds 0–4. Breast-cancer seeds change the stratified split; synthetic seeds change both '
        'the generated dataset and split. Timing repetitions are not independent datasets.', '',
        '| Dataset | Method | Converged seeds / 5 | Updates min–max | Test accuracy min–max |',
        '|---|---|---:|---:|---:|']
    for ds in ('synthetic','breast_cancer'):
        cases=[c for c in payload['precision']+payload['splits'] if c['dataset']==ds]
        for m in METHODS:
            hs=[c['methods'][m]['history'] for c in cases]
            its=[h['n_iter'] for h in hs]; acc=[h['test_acc']*100 for h in hs]
            lines.append(f"| {ds} | {m} | {sum(h['gradient_converged'] for h in hs)}/5 | {min(its)}–{max(its)} | {min(acc):.2f}–{max(acc):.2f}% |")
    lines += ['', '## Reproduce', '', '`python experiments/supplement.py`', '',
        'Use the project requirements in your Python environment. Raw repetitions, histories, weights, '
        'baseline 400-update checks, package versions, source commit and thread settings are in raw.json. '
        'Original saved results and slide files are not overwritten.', '',
        'These timings describe these implementations on this machine. rho was swept for optimization '
        'efficiency using training convergence, not selected using test accuracy. The default rho=1 '
        'remains the main comparison; tuned ADMM is a separate sensitivity result.', '']
    (args.out/'RESULTS.md').write_text('\n'.join(lines))
    print('Wrote',args.out,flush=True)


if __name__ == '__main__':
    main()
