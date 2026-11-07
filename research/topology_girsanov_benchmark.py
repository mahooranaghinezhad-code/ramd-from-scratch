"""Out-of-topology feedback-RAMD toy benchmark with discrete path reweighting.

This is a controlled 2D OVERDAMPED Euler-Maruyama (EM) experiment, not a
protein-ligand simulation, a replication of marginal Girsanov reweighting,
or a validated tauRAMD residence-time correction.

Three independent questions:
  (1) Does a frozen switch classifier transfer to DIFFERENT 1D potential shapes?
  (2) Can exact EM Gaussian path-likelihood ratios estimate the F=0 RESTRICTED
      first-passage mean from feedback-accelerated trajectories?
  (3) How does such path weighting compare with a deliberately heuristic
      force-to-zero extrapolation for the SAME restricted-time estimand?

Run from repository root:
    python research/topology_girsanov_benchmark.py --quick --output results_topology
    python research/topology_girsanov_benchmark.py --output results_topology

All quantities are dimensionless. Results are exploratory. High force/long
horizon Girsanov weights often have extremely low effective sample size.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from .state_conditioned_switching import RidgeLogistic, metrics
except ImportError:
    from state_conditioned_switching import RidgeLogistic, metrics


@dataclass(frozen=True)
class Design:
    source: str = 'double_well'
    targets: tuple[str, ...] = ('harmonic', 'tilted_double', 'two_barriers', 'gaussian_gate')
    train_forces: tuple[float, ...] = (0.6, 1.2)
    test_forces: tuple[float, ...] = (0.7, 1.4, 2.0)
    train_seeds: tuple[int, ...] = (2026, 2027, 2028)
    test_seeds: tuple[int, ...] = (2031, 2032, 2033)
    replicas: int = 128
    dt: float = .01
    horizon: float = 20.
    check_interval: float = .1
    min_progress: float = .42
    kbt: float = 1.
    transverse_k: float = 2.
    left: float = -2.
    start: float = -1.
    right: float = .8
    penalty: float = .002


def energy(x, shape):
    x=np.asarray(x, dtype=float)
    if shape == 'flat': return np.zeros_like(x)
    if shape == 'harmonic': return .5*(x+1.)**2
    if shape == 'double_well': return 2.5*(x*x-1.)**2
    if shape == 'tilted_double': return 1.9*(x*x-1.)**2 + .38*x
    if shape == 'two_barriers': return 9.0*(x*(x*x-1.))**2
    if shape == 'gaussian_gate':
        zz=(x-.08)/.23
        return .38*(x+1.)**2 + 2.1*np.exp(-.5*zz*zz)
    raise ValueError('Unknown potential topology: '+str(shape))


def grad_u(x, shape):
    x=np.asarray(x, dtype=float)
    if shape == 'flat': return np.zeros_like(x)
    if shape == 'harmonic': return x+1.
    if shape == 'double_well': return 10.*x*(x*x-1.)
    if shape == 'tilted_double': return 7.6*x*(x*x-1.)+.38
    if shape == 'two_barriers': return 18.*(x*x*x-x)*(3*x*x-1.)
    if shape == 'gaussian_gate':
        zz=(x-.08)/.23
        return .76*(x+1.) - (2.1/.23)*zz*np.exp(-.5*zz*zz)
    raise ValueError('Unknown potential topology')


def validate(d:Design):
    choices={'flat','harmonic','double_well','tilted_double','two_barriers','gaussian_gate'}
    if not d.targets or any(s not in choices for s in (d.source,*d.targets)):
        raise ValueError('Unknown/empty potential set')
    if d.source in d.targets or len(set(d.targets)) != len(d.targets):
        raise ValueError('No source shape may appear in held-out targets')
    if len(d.test_forces)<3 or not d.train_forces or set(d.test_forces)&set(d.train_forces):
        raise ValueError('Need >=3 strictly held-out test forces')
    if not d.train_seeds or not d.test_seeds or set(d.train_seeds)&set(d.test_seeds):
        raise ValueError('Training and test seeds must be strictly disjoint')
    if (len(set(d.train_seeds))!=len(d.train_seeds) or
        len(set(d.test_seeds))!=len(d.test_seeds)):
        raise ValueError('Duplicate seed')
    if d.replicas<2 or d.dt<=0 or d.horizon<=d.dt or d.kbt<=0 or d.check_interval<d.dt:
        raise ValueError('Invalid dynamics/time parameters')
    if d.left>=d.start or d.start>=d.right or d.min_progress<0:
        raise ValueError('Invalid boundaries/feedback threshold')
    for t in (d.check_interval,d.horizon):
        if abs(round(t/d.dt)*d.dt-t)>1e-9:
            raise ValueError('Time intervals must divide EM dt')


def seed_for(seed, shape_no, force_no, stream):
    return int(np.random.SeedSequence([seed,shape_no,force_no,stream]).generate_state(1)[0])


def full_mfpt(shape, d:Design, n=4001):
    """Exact continuum, F=0, reflecting-left/absorbing-right MFPT quadrature.

    The y harmonic variable is separable and does not affect x first passage.
    T(x0) = (1/D) int_x0^b exp(U(y)/kBT) int_a^y exp(-U(z)/kBT) dz dy.
    """
    x=np.sort(np.unique(np.r_[np.linspace(d.left,d.right,n), d.start]))
    u=energy(x,shape)/d.kbt
    q=np.exp(-u)
    inner=np.zeros(len(x))
    inner[1:]=np.cumsum(np.diff(x)*.5*(q[:-1]+q[1:]))
    keep=x>=d.start
    return float(np.trapezoid(np.exp(u[keep])*inner[keep]/d.kbt,x[keep]))


def tridiagonal_solve(lower, diag, upper, rhs):
    """Thomas algorithm, copy inputs, no SciPy dependency."""
    b=np.asarray(diag,float).copy(); f=np.asarray(rhs,float).copy()
    a=np.asarray(lower,float);c=np.asarray(upper,float)
    for j in range(1,len(b)):
        q=a[j]/b[j-1]
        b[j]-=q*c[j-1]
        f[j]-=q*f[j-1]
    ans=np.empty_like(f)
    ans[-1]=f[-1]/b[-1]
    for j in range(len(b)-2,-1,-1):
        ans[j]=(f[j]-c[j]*ans[j+1])/b[j]
    return ans


def restricted_mfpt(shape, d:Design, grid_n=351, time_steps=400):
    """Compute F=0 E[min(T,horizon)] via backward Kolmogorov PDE.

    ∂_t v = 1 + D∂²_x v - U'(x)∂_x v, v(x,0)=0,
    reflecting v_x(left,t)=0, absorbing v(right,t)=0.
    Uses backward implicit Euler in time and centered finite differences
    on x (grid/time discretization errors remain and are tested).
    """
    if grid_n<51 or time_steps<4: raise ValueError('Grid too coarse')
    grid=np.linspace(d.left,d.right,grid_n)
    dx=grid[1]-grid[0]
    n=grid_n-1
    b=-grad_u(grid[:n],shape)
    D=d.kbt
    lower=np.zeros(n);diag=np.full(n,-2*D/dx**2);upper=np.zeros(n)
    upper[0]=2*D/dx**2
    lower[1:]=D/dx**2-b[1:]/(2*dx)
    upper[1:]=D/dx**2+b[1:]/(2*dx)
    upper[-1]=0
    dt_t=d.horizon/time_steps
    A=np.zeros(n); B=1-dt_t*diag; C=np.zeros(n)
    A[1:]=-dt_t*lower[1:]
    C[:-1]=-dt_t*upper[:-1]
    val=np.zeros(n)
    for _ in range(time_steps):
        val=tridiagonal_solve(A,B,C,val+dt_t)
    return float(np.interp(d.start,grid[:n],val))


def gaussian_log_increment(force, kbt, dt, n1, n2, z1, z2):
    """Exact conditional *discrete EM* log density p_0/p_F for raw noise.

    Under q: δr = [b(x)+F n]dt + sqrt(2*kBT*dt)*z.
    Under p: δr = b(x)dt + sqrt(2*kBT*dt)*z_0.
    log(p/q) = -(F sqrt(dt)/(sqrt(2*kBT)))*(n⋅z)
               - F²dt/(4*kBT).
    The force is predictable from PAST state / direction, never z_{t+1}.
    Original Gaussian *pre-reflection* increments are used: the deterministic
    x-reflection mapping makes this an augmented-path likelihood ratio.
    """
    return (-force*np.sqrt(dt/(2*kbt))*(n1*z1+n2*z2)
            -force*force*dt/(4*kbt))


def logweight_stats(logw, truncated_times):
    """Self-normalized path importance sampling and weight degeneracy."""
    logw=np.asarray(logw,dtype=float)
    t=np.asarray(truncated_times,dtype=float)
    if len(logw)!=len(t) or not len(logw) or not np.isfinite(logw).all():
        raise ValueError('Invalid terminal log weights')
    a=logw.max()
    w=np.exp(logw-a)
    s=w.sum()
    ess=float(s*s/np.dot(w,w))
    snis=float(np.dot(w,t)/s)
    logmean=float(a+np.log(s/len(w)))
    raw=float(np.exp(a)*np.dot(w,t)/len(w)) if abs(a)<700 else float('nan')
    return dict(snis_rmst=snis,raw_rmst=raw,ess=ess,
                ess_fraction=ess/len(w),log_mean_weight=logmean,
                max_weight_fraction=float(np.max(w)/s))


def simulate(shape, d:Design, force:float, seed:int, collect=False):
    """2D feedback simulation, with log likelihood ratio until first exit.

    The force-direction feedback policy is the SAME under p0 and pF: given
    an observed state/history, every checkpoint uses the same displacement
    threshold and new uniform direction law; these conditional factors
    cancel in the joint path density ratio. Direction is selected before
    the step's thermal innovations, so the force is nonanticipative.
    """
    if force<0 or not np.isfinite(force):raise ValueError('Invalid force')
    rng=np.random.default_rng(seed)
    n=d.replicas
    x=np.full(n,d.start);y=np.zeros(n)
    theta=rng.uniform(0,2*np.pi,n)
    nx=np.cos(theta);ny=np.sin(theta)
    start_x=x.copy();start_y=y.copy();start_nx=nx.copy();start_ny=ny.copy()
    ref_x=x.copy();ref_y=y.copy()
    alive=np.ones(n,bool)
    stopped=np.full(n,d.horizon)
    censored=np.ones(n,dtype=bool)
    logw=np.zeros(n)
    reset_count=0
    exposure=0.
    logs=[]
    sigma=np.sqrt(2*d.kbt*d.dt)
    steps=round(d.horizon/d.dt)
    every=round(d.check_interval/d.dt)
    for step in range(1,steps+1):
        ids=np.flatnonzero(alive)
        if not len(ids):break
        z1=rng.normal(size=len(ids));z2=rng.normal(size=len(ids))
        exposure+=len(ids)*d.dt
        logw[ids]+=gaussian_log_increment(force,d.kbt,d.dt,
                              nx[ids],ny[ids],z1,z2)
        proposal_x=(x[ids]+d.dt*(-grad_u(x[ids],shape)+force*nx[ids])
                    +sigma*z1)
        x[ids]=np.where(proposal_x<d.left,2*d.left-proposal_x,proposal_x)
        y[ids]+=d.dt*(-d.transverse_k*y[ids]+force*ny[ids])+sigma*z2
        escaped=x[ids]>=d.right
        if np.any(escaped):
            gone=ids[escaped]
            alive[gone]=False
            censored[gone]=False
            stopped[gone]=step*d.dt
        rem=ids[~escaped]
        if not len(rem) or step%every:continue
        progress=np.hypot(x[rem]-ref_x[rem],y[rem]-ref_y[rem])
        swap=progress<d.min_progress
        if collect:
            k=len(rem)
            pre_features=np.column_stack((np.full(k,(step-every)*d.dt),
                                          start_x[rem],start_y[rem],
                                          start_nx[rem],start_ny[rem]))
            logs.append((pre_features,swap.astype(float)))
        ref_x[rem]=x[rem];ref_y[rem]=y[rem]
        changed=rem[swap]
        reset_count+=len(changed)
        if len(changed):
            new=rng.uniform(0,2*np.pi,len(changed))
            nx[changed]=np.cos(new);ny[changed]=np.sin(new)
        start_x[rem]=x[rem];start_y[rem]=y[rem]
        start_nx[rem]=nx[rem];start_ny[rem]=ny[rem]
    result=dict(stop=stopped,censored=int(censored.sum()),logw=logw,
                reset_count=reset_count,persistence=(exposure/reset_count if reset_count else None),
                rmst=float(stopped.mean()),
                full_mfpt_if_complete=(float(stopped.mean()) if not censored.any() else None))
    if force>0:result.update(logweight_stats(logw,stopped))
    if collect:
        X=np.concatenate([a for a,_ in logs],axis=0) if logs else np.empty((0,5))
        Y=np.concatenate([b for _,b in logs],axis=0) if logs else np.empty(0)
        result['X']=X;result['y']=Y
    return result


def features(X, shape, force, mode):
    """Only state at interval START; the future switching label is excluded."""
    X=np.asarray(X,dtype=float)
    if X.ndim!=2 or X.shape[1]!=5: raise ValueError('Invalid start-state array')
    t,x,y,nx,ny=X.T
    if mode=='time':return np.column_stack((t,t*t))
    if mode!='state':raise ValueError('Invalid feature mode')
    b=-grad_u(x,shape)
    return np.column_stack((t,t*t,x,y,nx,ny,b,b*nx,force*nx,force*ny))


def fit_source(d:Design):
    Xtime=[];Xstate=[];Ys=[];count=0
    for i,F in enumerate(d.train_forces):
        for seed in d.train_seeds:
            r=simulate(d.source,d,F,seed_for(seed,0,i,11),collect=True)
            if not len(r['y']):raise RuntimeError('Empty training labels')
            Xtime.append(features(r['X'],d.source,F,'time'))
            Xstate.append(features(r['X'],d.source,F,'state'))
            Ys.append(r['y']);count+=len(r['y'])
    target=np.concatenate(Ys)
    model_t=RidgeLogistic(d.penalty).fit(np.concatenate(Xtime),target)
    model_s=RidgeLogistic(d.penalty).fit(np.concatenate(Xstate),target)
    return model_t,model_s,count


def extrapolate_alpha_rmstr(force_alpha, forced_restricted_mean):
    """HEURISTIC log RMST_F=a+b*alpha, predict at alpha=1."""
    a=np.asarray(force_alpha,float);y=np.asarray(forced_restricted_mean,float)
    if len(a)<3 or a.shape!=y.shape or not np.isfinite(a).all() or not np.isfinite(y).all():
        return None
    if min(y)<=0 or np.ptp(a)<.01:return None
    slope, intercept=np.polyfit(a,np.log(y),1)
    ans=float(np.exp(intercept+slope))
    return ans if np.isfinite(ans) and ans>0 else None


def write_csv(path, records):
    if not records:raise ValueError('Cannot save empty data')
    cols=list(dict.fromkeys(k for r in records for k in r))
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=cols)
        w.writeheader();w.writerows(records)


def evaluate(d:Design, output:Path):
    validate(d)
    output.mkdir(parents=True,exist_ok=True)
    mdl_t,mdl_s,n_train=fit_source(d)
    np.savez(output/'frozen_source_models.npz',time_coef=mdl_t.w,state_coef=mdl_s.w,
             time_center=mdl_t.center,time_scale=mdl_t.scale,
             state_center=mdl_s.center,state_scale=mdl_s.scale,
             training_shape=d.source,training_forces=np.array(d.train_forces))
    reference=[];force_rows=[];seed_rows=[];shape_rows=[]
    for i,shape in enumerate(d.targets):
        exact_full=full_mfpt(shape,d)
        exact_restricted=restricted_mfpt(shape,d)
        ref=dict(shape=shape,exact_full_mfpt=exact_full,exact_rmst=exact_restricted,
                 fraction_of_full=exact_restricted/exact_full)
        reference.append(ref)
        for seed in d.test_seeds:
            control=simulate(shape,d,0.,seed_for(seed,i+50,0,22))
            alpha=[];restricted_means=[];weighted=[];ess=[];gains=[]
            for j,F in enumerate(d.test_forces):
                pilot=simulate(shape,d,F,seed_for(seed,i+50,j,0))
                tau=pilot['persistence']
                if tau is None:raise RuntimeError('Pilot had no feedback switches')
                r=simulate(shape,d,F,seed_for(seed,i+50,j,1),collect=True)
                pred_t=mdl_t.predict(features(r['X'],shape,F,'time'))
                pred_s=mdl_s.predict(features(r['X'],shape,F,'state'))
                lt,_=metrics(r['y'],pred_t)
                ls,_=metrics(r['y'],pred_s)
                gain=lt-ls
                a=d.kbt/(d.kbt+F**2*tau/2.0)
                alpha.append(a);restricted_means.append(r['rmst'])
                weighted.append(r['snis_rmst']);ess.append(r['ess_fraction']);gains.append(gain)
                force_rows.append(dict(shape=shape,seed=seed,force=F,
                    alpha_proxy=a,tau_pilot=tau,n_events=len(r['y']),
                    logloss_gain_state_vs_time=gain,
                    biased_rmst=r['rmst'],direct_unbiased_rmst=control['rmst'],
                    exact_unbiased_rmst=exact_restricted,
                    path_snis_rmst=r['snis_rmst'],path_raw_rmst=r['raw_rmst'],
                    path_weight_log_mean=r['log_mean_weight'],
                    path_ess=r['ess'],path_ess_fraction=r['ess_fraction'],
                    max_weight_fraction=r['max_weight_fraction'],
                    n_censored_biased=r['censored']))
            heuristic=extrapolate_alpha_rmstr(alpha,restricted_means)
            mid=len(d.test_forces)//2
            reweighted=weighted[mid]
            e_heuristic=(abs(np.log(heuristic/exact_restricted)) if heuristic else None)
            e_path=abs(np.log(reweighted/exact_restricted))
            seed_rows.append(dict(shape=shape,seed=seed,exact_full_mfpt=exact_full,
                exact_rmst=exact_restricted,
                source_state_gain=float(np.mean(gains)),
                direct_unbiased_rmst=control['rmst'],n_censored_direct=control['censored'],
                heuristic_rmst=heuristic,heuristic_abs_log_error=e_heuristic,
                path_reweighted_rmst=reweighted,path_abs_log_error=e_path,
                path_force=d.test_forces[mid],path_ess_fraction=ess[mid],
                status=('low_ess' if ess[mid]<.05 else 'exploratory')))
        records=[r for r in seed_rows if r['shape']==shape]
        shape_rows.append(dict(shape=shape,exact_full_mfpt=exact_full,
            exact_rmst=exact_restricted,ratio_rmst_to_full=exact_restricted/exact_full,
            n_seed_batches=len(records),
            mean_state_gain=float(np.mean([r['source_state_gain'] for r in records])),
            mean_heuristic_logerror=float(np.mean([r['heuristic_abs_log_error'] for r in records])),
            mean_path_logerror=float(np.mean([r['path_abs_log_error'] for r in records])),
            mean_path_ess_fraction=float(np.mean([r['path_ess_fraction'] for r in records]))))
        print(f'{shape:15s}  reference RMST={exact_restricted:7.3f} '
              f'path err={shape_rows[-1]["mean_path_logerror"]:.3f} '
              f'heuristic err={shape_rows[-1]["mean_heuristic_logerror"]:.3f} '
              f'ESS fraction={shape_rows[-1]["mean_path_ess_fraction"]:.3f}')
    write_csv(output/'exact_references.csv',reference)
    write_csv(output/'per_force.csv',force_rows)
    write_csv(output/'per_seed.csv',seed_rows)
    write_csv(output/'topology_summary.csv',shape_rows)
    info=dict(source=d.source,heldout_shapes=d.targets,train_forces=d.train_forces,
              test_forces=d.test_forces,train_seeds=d.train_seeds,test_seeds=d.test_seeds,
              n_training_checkpoint_events=n_train,source_fit_count=2,
              target_refit_count=0, estimand='E_0[min(T, horizon)]',
              error_definition='absolute log(estimate/exact restricted mean)',
              path_weighting='exact discrete EM Gaussian likelihood ratio on augmented stopped path',
              heuristic='fit log biased RMST vs effective-alpha proxy, extrapolate to alpha=1',
              no_claim_of_new_method=True)
    (output/'overview.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    fig,axs=plt.subplots(1,2,figsize=(11,4.5))
    ix=np.arange(len(d.targets));width=.34
    axs[0].bar(ix-width/2,[r['mean_path_logerror'] for r in shape_rows],width,label='Path-weighted')
    axs[0].bar(ix+width/2,[r['mean_heuristic_logerror'] for r in shape_rows],width,label='Heuristic extrapolation')
    axs[0].set_xticks(ix,[r['shape'] for r in shape_rows],rotation=22,ha='right')
    axs[0].set_ylabel('|log(estimate / exact restricted MFPT)|')
    axs[0].set_title('Same F=0 restricted first-passage estimand')
    axs[0].legend()
    axs[1].bar(ix,[r['mean_path_ess_fraction'] for r in shape_rows])
    axs[1].set_xticks(ix,[r['shape'] for r in shape_rows],rotation=22,ha='right')
    axs[1].axhline(.05,color='black',ls='--',lw=1,label='5% ESS warning')
    axs[1].set_ylabel('Mean path importance ESS / trajectories')
    axs[1].set_title('Weight degeneracy of standard Girsanov')
    axs[1].legend()
    fig.tight_layout();fig.savefig(output/'topology_and_reweighting.png',dpi=160);plt.close(fig)
    (output/'LIMITATIONS.txt').write_text(
        'The target is restricted MFPT/RMST, not unrestricted ligand residence time.\n'
        'Exact F=0 RMST uses a time-discretized backward PDE, not a closed-form rate.\n'
        'Path likelihood weights are exact for the specified augmented EM step with known Gaussian noise and predictable feedback; not a general underdamped OpenMM formula.\n'
        'Reflection handled as deterministic function of pre-reflection Gaussian increments; weights include latent steps.\n'
        'Conventional Girsanov has potentially catastrophic weight variance; inspect ESS and weight means.\n'
        'Heuristic alpha extrapolation is NOT a validated kinetic correction or a result of the Roux paper.\n'
        'Source classifier frozen from one topology; known analytic target physical force enters feature input.\n'
        'Each held-out topology is a different 1D U(x), yet all are separable 2D toy potentials.\n'
        'Check numerical convergence, independent seeds, model uncertainty, and topology diversity before scientific claims.\n',encoding='utf-8')
    return info,reference,force_rows,seed_rows,shape_rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--quick',action='store_true',help='Tiny CI smoke, no scientific interpretation')
    p.add_argument('--replicas',type=int,default=128)
    p.add_argument('--horizon',type=float,default=20.)
    p.add_argument('--output',type=Path,default=Path('results_topology'))
    a=p.parse_args()
    d=Design(replicas=a.replicas,horizon=a.horizon)
    if a.quick:
        d=Design(replicas=32,horizon=6.,
                 train_seeds=(2026,2027),test_seeds=(2031,2032),
                 targets=('harmonic','two_barriers'))
    evaluate(d,a.output)


if __name__=='__main__':main()
