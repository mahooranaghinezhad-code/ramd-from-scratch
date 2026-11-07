"""Predict the NEXT tauRAMD-like checkpoint decision from CURRENT state only.

Educational, dimensionless 2D overdamped Langevin model. At time t immediately
AFTER a checkpoint, record (position, direction, time); predict whether the
force will be reset at t+Delta, conditional on survival to that checkpoint.
The displacement over [t,t+Delta] is the LABEL-generating future event,
never a predictor. Splits are by independent RNG seeds, NOT event rows.

Compare constant-rate, time-only and state+time logistic models for actual
feedback and randomized-checkpoint controls.  Results do NOT reconstruct
unbiased ligand residence times; they evaluate next-checkpoint prediction.

Run: python research/state_conditioned_switching.py --quick
Full exploratory: python research/state_conditioned_switching.py --output results_state_switch
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

try:
    from .feedback_vs_exponential import Config, randomized_checkpoint_mask, simulate
except ImportError:
    from feedback_vs_exponential import Config, randomized_checkpoint_mask, simulate


COLUMNS = ('seed', 'trajectory', 'checkpoint', 'time_start', 'x_start', 'y_start',
           'nx_start', 'ny_start', 'switched', 'future_progress')


@dataclass(frozen=True)
class Study:
    conditions: tuple[tuple[float, float], ...] = ((1.0, 0.55), (3.0, 0.25))
    train_seeds: tuple[int, ...] = (2026, 2027, 2028, 2029, 2030)
    test_seeds: tuple[int, ...] = (2031, 2032, 2033, 2034)
    replicas: int = 192
    dt: float = 0.005
    checkpoint_time: float = 0.1
    horizon: float = 16.0
    penalty: float = 0.002


def seed_for(seed: int, cell: int, stream: int) -> int:
    return int(np.random.SeedSequence([seed, cell, stream]).generate_state(1)[0])


def make_config(study: Study, force: float, threshold: float) -> Config:
    if study.replicas < 2 or study.dt <= 0 or study.checkpoint_time < study.dt:
        raise ValueError('Invalid simulation size or timing')
    steps = round(study.horizon/study.dt)
    checks = round(study.checkpoint_time/study.dt)
    if not steps or not checks or abs(steps*study.dt-study.horizon)>1e-9 or abs(checks*study.dt-study.checkpoint_time)>1e-9:
        raise ValueError('Horizon and checkpoint times must be integer multiples of dt')
    return Config(replicas=study.replicas, steps=steps, dt=study.dt, force=force,
                  min_progress=threshold, check_steps=checks, trace_replicas=0)


def record_checkpoints(cfg: Config, protocol: str, seed: int,
                       switch_probability: float | None = None):
    """Instrument exact force-reset dynamics with *pre-interval* features.

    Output checkpoint rows only for trajectories alive at the END checkpoint.
    Escapes during the interval do NOT receive an artificial negative label.
    This is a survivor-conditioned decision process, not an escape predictor.
    No new RNG calls are added relative to simulate() for the same protocol.
    """
    if protocol not in ('feedback', 'checkpoint_random'):
        raise ValueError('Only comparable checkpoint protocols supported')
    if protocol == 'checkpoint_random' and (switch_probability is None or
            not np.isfinite(switch_probability) or not 0 <= switch_probability <= 1):
        raise ValueError('Need a finite independent-pilot reset fraction')
    rng=np.random.default_rng(seed)
    n=cfg.replicas
    x=np.full(n,cfg.start,dtype=float);y=np.zeros(n)
    theta=rng.uniform(0,2*np.pi,n)
    nx,ny=np.cos(theta),np.sin(theta)
    ref_x=x.copy(); ref_y=y.copy()
    start_x=x.copy(); start_y=y.copy()
    start_nx=nx.copy(); start_ny=ny.copy()
    alive=np.ones(n,bool)
    times=np.full(n,np.nan)
    changed=0;checkpoint_opportunities=0;intervals_ended_early=0
    collected=[]
    for step in range(1,cfg.steps+1):
        ids=np.flatnonzero(alive)
        if not len(ids):break
        sigma=np.sqrt(2*cfg.kbt*cfg.dt)
        x[ids]+=cfg.dt*(-4*cfg.height*x[ids]*(x[ids]**2-1)+cfg.force*nx[ids])+sigma*rng.normal(size=len(ids))
        y[ids]+=cfg.dt*(-cfg.transverse_k*y[ids]+cfg.force*ny[ids])+sigma*rng.normal(size=len(ids))
        reflected=x[ids]<cfg.left
        x[ids[reflected]]=2*cfg.left-x[ids[reflected]]
        escaped=x[ids]>=cfg.right
        left_now=ids[escaped]
        if len(left_now):
            times[left_now]=step*cfg.dt;alive[left_now]=False
            intervals_ended_early+=len(left_now)
        remaining=ids[~escaped]
        if not len(remaining):continue
        if step%cfg.check_steps:
            continue
        if protocol=='feedback':
            progress=np.hypot(x[remaining]-ref_x[remaining],y[remaining]-ref_y[remaining])
            swap=progress<cfg.min_progress
        else:
            # For the randomized control the selection ignores displacement.
            # future_progress is still recorded for diagnostic tests only.
            progress=np.hypot(x[remaining]-ref_x[remaining],y[remaining]-ref_y[remaining])
            swap=randomized_checkpoint_mask(len(remaining),switch_probability,
                        checkpoint_opportunities,changed,rng)
        size=len(remaining)
        row=np.column_stack([
           np.full(size,seed),remaining,np.full(size,step//cfg.check_steps),
           np.full(size,(step-cfg.check_steps)*cfg.dt),
           start_x[remaining],start_y[remaining],start_nx[remaining],start_ny[remaining],
           swap.astype(int),progress,
        ])
        collected.append(row)
        checkpoint_opportunities+=size
        ref_x[remaining]=x[remaining];ref_y[remaining]=y[remaining]
        updating=remaining[swap]
        changed+=len(updating)
        if len(updating):
            new_angle=rng.uniform(0,2*np.pi,len(updating))
            nx[updating]=np.cos(new_angle);ny[updating]=np.sin(new_angle)
        # At t=checkpoint end the future-label features are recorded: reset
        # decisions have already happened, and the next interval now starts.
        start_x[remaining]=x[remaining];start_y[remaining]=y[remaining]
        start_nx[remaining]=nx[remaining];start_ny[remaining]=ny[remaining]
    raw=np.concatenate(collected,axis=0) if collected else np.empty((0,len(COLUMNS)))
    return dict(raw=raw, events=len(raw), escapes=int(np.isfinite(times).sum()),
                escape_times=times, n_changed=changed,
                opportunities=checkpoint_opportunities,
                unobserved_interval_outcomes=intervals_ended_early)


def features(raw: np.ndarray, cfg: Config, mode: str):
    """Only information available at the START of an observation interval.

    Index 9 (the future displacement) is DELIBERATELY excluded.
    """
    if raw.ndim!=2 or raw.shape[1]!=len(COLUMNS):
        raise ValueError('Invalid event matrix')
    t=raw[:,3]
    if mode=='time':
        return np.column_stack((t,t*t))
    if mode!='state':
        raise ValueError('Unknown feature mode')
    x,y,nx,ny=raw[:,4],raw[:,5],raw[:,6],raw[:,7]
    fx=-4*cfg.height*x*(x*x-1)
    return np.column_stack((t,t*t,x,y,nx,ny,fx,fx*nx,cfg.force*nx,cfg.force*ny))


def sigmoid(z):
    z=np.clip(z,-40.,40.)
    return 1.0/(1.0+np.exp(-z))


class RidgeLogistic:
    """Standardized, intercept-unpenalized, NumPy-only logistic regression."""
    def __init__(self,penalty=.002):
        self.penalty=penalty

    def fit(self,X,y):
        X=np.asarray(X,dtype=float);y=np.asarray(y,dtype=float)
        if not (X.ndim==2 and len(X)==len(y) and len(y) and np.isfinite(X).all()):
            raise ValueError('Invalid or empty training data')
        if not np.all((y==0)|(y==1)):
            raise ValueError('Binary outcomes required')
        if self.penalty<0: raise ValueError('Penalty must be nonnegative')
        self.center=X.mean(axis=0)
        self.scale=X.std(axis=0);self.scale[self.scale<1e-12]=1.
        Z=np.column_stack((np.ones(len(X)),(X-self.center)/self.scale))
        mean_rate=np.clip(y.mean(),1e-5,1-1e-5)
        w=np.zeros(Z.shape[1]);w[0]=np.log(mean_rate/(1-mean_rate))
        reg=np.eye(len(w))*self.penalty
        reg[0,0]=0
        for _ in range(50):
            p=sigmoid(Z@w)
            g=Z.T@(p-y)/len(y)+reg@w
            a=np.maximum(p*(1-p),1e-9)
            h=(Z.T*a)@Z/len(y)+reg
            delta=np.linalg.solve(h+np.eye(len(w))*1e-9,g)
            w-=delta
            if np.max(np.abs(delta))<1e-7:break
        self.w=w
        return self

    def predict(self,X):
        z=np.column_stack((np.ones(len(X)),(X-self.center)/self.scale))
        return sigmoid(z@self.w)


def metrics(y,p):
    p=np.clip(np.asarray(p,dtype=float),1e-12,1-1e-12)
    y=np.asarray(y,dtype=float)
    return (float(np.mean(-y*np.log(p)-(1-y)*np.log(1-p))),
            float(np.mean((y-p)**2)))


def _write_csv(path,rows):
    if not rows:raise ValueError('No output records')
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        wr=csv.DictWriter(f,fieldnames=fields);wr.writeheader();wr.writerows(rows)


def evaluate(study:Study,output:Path):
    overlap=set(study.train_seeds)&set(study.test_seeds)
    if overlap:raise ValueError('Training and test seed batches must be disjoint')
    if not study.conditions or not study.train_seeds or not study.test_seeds:
        raise ValueError('Empty experimental design')
    output.mkdir(parents=True,exist_ok=True)
    by_seed=[];summary=[];calibration=[]
    for case_id,(force,threshold) in enumerate(study.conditions):
        cfg=make_config(study,force,threshold)
        pilot=record_checkpoints(cfg,'feedback',seed_for(9999,case_id,0))
        if not pilot['opportunities'] or not pilot['n_changed']:
            raise RuntimeError('Pilot had no decisions; increase run length')
        p_pilot=pilot['n_changed']/pilot['opportunities']
        for ptype,stream in [('feedback',1),('checkpoint_random',2)]:
            train=[];test={}
            for seed in study.train_seeds+study.test_seeds:
                result=record_checkpoints(cfg,ptype,seed_for(seed,case_id,stream),
                    switch_probability=p_pilot if ptype=='checkpoint_random' else None)
                if result['events']==0:raise RuntimeError('No eligible labeled checkpoints')
                result['raw'][:,0]=seed  # preserve human-readable batch ID
                if seed in study.train_seeds:train.append(result['raw'])
                else:test[seed]=result
            training=np.concatenate(train)
            y=training[:,8]
            mean_p=float(np.clip(y.mean(),1e-6,1-1e-6))
            tm=RidgeLogistic(study.penalty).fit(features(training,cfg,'time'),y)
            sm=RidgeLogistic(study.penalty).fit(features(training,cfg,'state'),y)
            for seed,result in test.items():
                r=result['raw'];target=r[:,8]
                pred_const=np.full(len(r),mean_p)
                pred_time=tm.predict(features(r,cfg,'time'))
                pred_state=sm.predict(features(r,cfg,'state'))
                l0,b0=metrics(target,pred_const)
                lt,bt=metrics(target,pred_time)
                ls,bs=metrics(target,pred_state)
                by_seed.append(dict(condition=case_id,force=force,threshold=threshold,
                    protocol=ptype,seed=seed,n_event=len(r),
                    n_escaped=result['escapes'],n_unlabeled_due_to_escape=result['unobserved_interval_outcomes'],
                    train_event_rate=mean_p,test_event_rate=float(target.mean()),
                    pilot_reset_rate=p_pilot,
                    logloss_constant=l0,logloss_time=lt,logloss_state=ls,
                    gain_state_vs_time=lt-ls,gain_state_vs_constant=l0-ls,
                    brier_constant=b0,brier_time=bt,brier_state=bs))
                for k in range(5):
                    keep=(pred_state>=k/5)&(pred_state<(k+1)/5 if k<4 else pred_state<=1.)
                    if keep.any():
                        calibration.append(dict(condition=case_id,protocol=ptype,seed=seed,
                             bin=k,n=int(keep.sum()),mean_prediction=float(pred_state[keep].mean()),
                             observed_rate=float(target[keep].mean())))
            batch=[v for v in by_seed if v['condition']==case_id and v['protocol']==ptype]
            summary.append(dict(condition=case_id,force=force,threshold=threshold,protocol=ptype,
                n_train_batches=len(study.train_seeds),n_heldout_batches=len(study.test_seeds),
                total_test_events=sum(b['n_event'] for b in batch),
                mean_gain_vs_time=float(np.mean([b['gain_state_vs_time'] for b in batch])),
                mean_gain_vs_constant=float(np.mean([b['gain_state_vs_constant'] for b in batch])),
                n_batches_state_beats_time=sum(b['gain_state_vs_time']>0 for b in batch)))
            print(f'F={force:g}, rmin={threshold:g}, {ptype}: test events='
                  f'{summary[-1]["total_test_events"]}, state - time logloss gain='
                  f'{summary[-1]["mean_gain_vs_time"]:+.5f}, positive seeds='
                  f'{summary[-1]["n_batches_state_beats_time"]}/{len(study.test_seeds)}')
    _write_csv(output/'heldout_seed_metrics.csv',by_seed)
    _write_csv(output/'summary.csv',summary)
    _write_csv(output/'state_calibration.csv',calibration)
    fig,ax=plt.subplots(figsize=(8,4.6))
    cases=[];gains=[];colors=[]
    for s in summary:
        cases.append(f'F={s["force"]:g}, r={s["threshold"]:g}\n{s["protocol"]}')
        gains.append(s['mean_gain_vs_time'])
    ax.bar(np.arange(len(cases)),gains)
    ax.axhline(0,color='black',ls='--',lw=1)
    ax.set_xticks(np.arange(len(cases)),cases)
    ax.set_ylabel('Held-out log loss gain: time-only minus state+time')
    ax.set_title('Next-checkpoint switch prediction (by independent seed)')
    fig.tight_layout();fig.savefig(output/'heldout_model_comparison.png',dpi=160)
    plt.close(fig)
    (output/'LIMITATIONS.txt').write_text(
      'This predicts a switch at the next checkpoint GIVEN that the trajectory survives until then.\n'
      'Features are sampled at interval START; future displacement is recorded for audit, NOT used as input.\n'
      'Independent RNG seed batches are held out; events from the same trajectory are correlated.\n'
      'Random checkpoint control uses pilot-count-matched quotas, not true iid Bernoulli switching.\n'
      'Better held-out logloss does NOT establish a new kinetics correction or valid residence-time recovery.\n'
      'Additional held-out force/threshold/potential conditions, seed-batch uncertainty and timestep convergence are required.\n',encoding='utf-8')
    return by_seed,summary,calibration


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--quick',action='store_true',help='Tiny CI smoke, NOT a scientific result')
    p.add_argument('--output',type=Path,default=Path('results_state_conditioned'))
    p.add_argument('--replicas',type=int,default=192)
    p.add_argument('--horizon',type=float,default=16)
    args=p.parse_args()
    study=Study(replicas=args.replicas,horizon=args.horizon)
    if args.quick:
        study=Study(replicas=48,horizon=5.,train_seeds=(2026,2027),test_seeds=(2030,2031))
    evaluate(study,args.output)


if __name__=='__main__':
    main()
