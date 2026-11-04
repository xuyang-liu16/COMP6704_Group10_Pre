"""Direct Newton controls and conditioning diagnostics for the presentation.

Uses the existing ADMM Newton kernel without changing any member algorithm.
The strict Newton control isolates outer splitting overhead with the same kernel
and tolerance. The target Newton control uses the common 1e-6 tolerance.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import sys,json,time,platform,subprocess,importlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import scipy,sklearn
from scipy.special import expit
from threadpoolctl import threadpool_limits,threadpool_info
from data import load_problem
from problem import grad,objective,lipschitz_hint
from metrics import pack_history
from algorithms.admm import _logistic_update
from experiments.run_compare import _validated_history,_json_default
from experiments.supplement import figures,savefig
import matplotlib.pyplot as plt


def direct_newton(X,y,lam,max_iter=10000,tol=1e-6,X_test=None,y_test=None,strict=False):
    X=np.asarray(X,dtype=float);y=np.asarray(y,dtype=float)
    if X.ndim!=2 or y.shape!=(len(X),) or not np.isfinite(X).all() or not np.isin(y,[-1,1]).all():
        raise ValueError('Invalid training data')
    t=time.perf_counter();zero=np.zeros(X.shape[1]);v0=objective(zero,X,y,lam)
    w,steps,ok=_logistic_update(zero,X,y,zero,lam,min(max_iter,50),1e-10 if strict else tol)
    h=pack_history([v0,objective(w,X,y,lam)],[0,time.perf_counter()-t],w,X,y,X_test,y_test,lam)
    h.update(iteration=[0,steps],n_iter=steps,converged=ok,newton_steps=steps)
    if not ok:raise RuntimeError('Newton did not converge')
    return w,h


def main():
    out=ROOT/'results'/'newton_comparison';out.mkdir(exist_ok=True,parents=True)
    prior=json.loads((ROOT/'results'/'supplement'/'raw.json').read_text())
    methods={m:importlib.import_module('algorithms.'+m).run for m in ('gd','nesterov','fista','admm')}
    methods.update(admm_matched=methods['admm'],newton=direct_newton,newton_strict=direct_newton)
    payload={'precision':[],'rho':prior['rho'],'splits':prior['splits'],'conditioning':{},'newton_controls':[]}
    with threadpool_limits(limits=1):
        payload['environment']=dict(platform=platform.platform(),cpu=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),
            python=sys.version,numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,threadpools=threadpool_info(),
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
        for ds in ('synthetic','breast_cancer'):
            X,y,Xt,yt=load_problem(ds,0);runs={m:[] for m in methods}
            def run(m):
                kw={'rho':.01} if m=='admm_matched' else ({'strict':True} if m=='newton_strict' else {})
                t=time.perf_counter();w,h=methods[m](X,y,lam=.01,max_iter=10000,tol=1e-6,X_test=Xt,y_test=yt,**kw);elapsed=time.perf_counter()-t
                h=_validated_history(w,h,X,y,Xt,yt,.01,1e-6)
                assert h['gradient_converged']
                if m.startswith('admm'):
                    assert h['converged'] and h['inner_converged'][-1]
                    assert h['primal_residual'][-1]<=h['eps_primal'][-1] and h['dual_residual'][-1]<=h['eps_dual'][-1]
                return elapsed,w,h
            for m in methods:run(m)
            rng=np.random.default_rng(82345)
            for rep in range(20):
                for m in rng.permutation(list(methods)):runs[m].append(run(m))
            case=dict(dataset=ds,seed=0,lam=.01,budget=10000,tol=1e-6,repeats=20,rho=1,methods={})
            for m,rr in runs.items():
                times=np.array([r[0] for r in rr]);elapsed,w,h=rr[np.argsort(times)[10]]
                assert all(r[2]['n_iter']==h['n_iter'] and np.allclose(r[1],w,atol=1e-12,rtol=1e-12) for r in rr)
                case['methods'][m]=dict(wall_samples_s=times.tolist(),median_ms=float(np.median(times)*1000),
                    q25_ms=float(np.quantile(times,.25)*1000),q75_ms=float(np.quantile(times,.75)*1000),
                    representative_wall_s=elapsed,weights=w.tolist(),history=h)
            ref=np.array(case['methods']['newton_strict']['weights'])
            for m,r in case['methods'].items():
                w=np.array(r['weights']);gap=objective(w,X,y,.01)-objective(ref,X,y,.01)
                assert -1e-14<=gap<=5e-11+1e-14
                r['objective_gap_to_strict_newton']=float(gap)
            matched=case['methods']['admm_matched'];strict=case['methods']['newton_strict']
            assert np.allclose(matched['weights'],strict['weights'],atol=1e-12,rtol=1e-12)
            assert sum(matched['history']['inner_iterations'])==strict['history']['n_iter']
            assert all(n==0 for n in matched['history']['inner_iterations'][2:])
            pr=expit(X@ref);H=X.T@((pr*(1-pr))[:,None]*X)/len(y)+.01*np.eye(X.shape[1]);ev=np.linalg.eigvalsh(H)
            payload['conditioning'][ds]=dict(L=lipschitz_hint(X,.01),mu_bound=.01,L_over_mu=lipschitz_hint(X,.01)/.01,
                local_hessian_min=float(ev[0]),local_hessian_max=float(ev[-1]),local_condition_number=float(ev[-1]/ev[0]))
            payload['newton_controls'].append(case)
            subset={**case,'methods':{m:case['methods'][m] for m in ('gd','nesterov','fista','admm')}}
            old=next(c for c in prior['precision'] if c['dataset']==ds)
            for m,r in subset['methods'].items():
                assert np.allclose(r['history']['objective'],old['methods'][m]['history']['objective'],atol=1e-12)
                r['history']['grad_norm']=old['methods'][m]['history']['grad_norm']
            payload['precision'].append(subset)
            print(ds,[(m,round(r['median_ms'],3),r['history']['n_iter']) for m,r in case['methods'].items()],flush=True)
    (out/'raw.json').write_text(json.dumps(payload,indent=2,default=_json_default,allow_nan=False)+'\n')
    figures(payload,out)
    fig,axs=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
    for ax,c in zip(axs,payload['newton_controls']):
        for i,m in enumerate(('newton','newton_strict','admm_matched')):
            r=c['methods'][m];ax.bar(i,r['median_ms'],color=['#0072B2','#009E73','#CC79A7'][i],width=.6)
            ax.errorbar(i,r['median_ms'],yerr=[[r['median_ms']-r['q25_ms']],[r['q75_ms']-r['median_ms']]],color='black',capsize=4)
        ax.set(xticks=[0,1,2],xticklabels=['Newton\n1e-6','Newton\n1e-10','ADMM\nrho = lambda'],ylabel='Full solver time (ms)',title=c['dataset'].replace('_',' ').title())
        ax.grid(axis='y',alpha=.2)
    fig.suptitle('Direct Newton controls: same inner kernel, 20 timed repeats (median / IQR)')
    savefig(fig,out/'newton_controls')
    lines=['# Direct Newton controls and conditioning','',
      'Same batch: all four original solvers, ADMM rho=lambda, and two direct Newton controls. '
      '20 repetitions after warm-up, shuffled method order. Training target 1e-6. '
      'Newton reuses the existing ADMM damped-Newton kernel with zero center and rho=lambda. '
      'It is a controlled component comparison, not an independently implemented solver. '
      'The 1e-10 control matches ADMM inner tolerance; the 1e-6 control uses the common target. '
      'Newton logs only its endpoints; ADMM also logs outer iterations, residuals and metrics. '
      'Thus the time difference includes bookkeeping, repeated checks and split-variable updates; '
      'it is not a pure estimate of a mathematical splitting operation.','',
      '| Dataset | Method | Updates | Median ms | IQR ms | Gradient |',
      '|---|---|---:|---:|---:|---:|']
    for c in payload['newton_controls']:
        for m,r in c['methods'].items():
            h=r['history'];lines.append(f"| {c['dataset']} | {m} | {h['n_iter']} | {r['median_ms']:.3f} | {r['q25_ms']:.3f}–{r['q75_ms']:.3f} | {h['final_grad_norm']:.3e} |")
    lines+=['','Newton counts are Newton steps; ADMM counts are outer iterations. Do not compare counts as equal work.','',
      '## Conditioning (seed 0, standardized training design including bias)','',
      '| Dataset | L | mu lower bound | L/mu upper bound | Hessian condition at optimum |','|---|---:|---:|---:|---:|']
    for ds,c in payload['conditioning'].items():lines.append(f"| {ds} | {c['L']:.5f} | 0.01 | {c['L_over_mu']:.2f} | {c['local_condition_number']:.2f} |")
    lines+=['','L/mu is a conservative global bound, not a measured Hessian condition number. '
      'Local conditioning is evaluated at the strict Newton solution and is smaller on breast cancer. '
      'Its global step bound is more conservative near the solution: L divided by the largest local '
      'Hessian eigenvalue is 15.90 versus 1.57. This is not a causal ablation. '
      'The momentum schedule is general-convex and has no restart; its objective-error guarantee '
      'does not prescribe the number of updates to a gradient target.','',
      'The rho sweep and seed checks remain the earlier separate experiments, explicitly labeled. '
      'All main four-method timing tables and curves are refreshed from this new batch.','',
      'Reproduce: `python experiments/newton_comparison.py`. Raw samples and environment are in raw.json.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print('Conditioning:',payload['conditioning'],flush=True)


if __name__=='__main__':main()
