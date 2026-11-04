"""Check saved supplement against original results and an independent Newton reference."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.special import expit
from data import load_problem


def main():
    out = ROOT/'results'/'newton_comparison'
    payload = json.loads((out/'raw.json').read_text())
    checks = []
    for c in payload['newton_controls']+payload['rho']+payload['splits']:
        X,y,_,_=load_problem(c['dataset'],c['seed']); lam=c['lam']
        w=np.zeros(X.shape[1])
        def value(w):
            return np.logaddexp(0,-y*(X@w)).mean()+lam*(w@w)/2
        for _ in range(50):
            p=expit(X@w)
            g=X.T@(p-(y+1)/2)/len(y)+lam*w
            if np.linalg.norm(g)<1e-12: break
            H=X.T@((p*(1-p))[:,None]*X)/len(y)+lam*np.eye(len(w))
            d=np.linalg.solve(H,-g); step=1.
            while value(w+step*d)>value(w)+1e-4*step*(g@d)+1e-15:
                step*=.5
                assert step>1e-12
            w=w+step*d
        assert np.linalg.norm(g)<1e-11
        for m,r in c['methods'].items():
            wr=np.array(r['weights']); h=r['history']
            measured=X.T@(expit(X@wr)-(y+1)/2)/len(y)+lam*wr
            gap=float(value(wr)-value(w))
            assert np.linalg.norm(measured)<=c['tol']
            assert -1e-14<=gap<=c['tol']**2/(2*lam)+1e-14
            checks.append(dict(dataset=c['dataset'],seed=c['seed'],rho=c['rho'],
                               method=m,objective_gap=gap,reference_gradient=float(np.linalg.norm(g))))
    for c in payload['rho']:
        if c['rho']==c['lam']:
            h=c['methods']['admm']['history']
            assert h['grad_norm'][1]<1e-10
            assert all(n==0 for n in h['inner_iterations'][2:])
    (out/'verification.json').write_text(json.dumps(dict(
        independent_newton_checks=checks,rho_equals_lambda_first_subproblem_verified=True),indent=2)+'\n')
    print('PASS: independent Newton references for',len(checks),
          'endpoints; rho=lambda first-subproblem check.')


if __name__=='__main__': main()
