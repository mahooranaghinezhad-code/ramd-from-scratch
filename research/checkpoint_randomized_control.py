"""Checkpoint-randomized control isolating displacement feedback in toy RAMD.

Compare three direction-update policies on one dimensionless 2D Langevin
potential: (1) genuine displacement feedback at checkpoints, (2) randomized
checkpoint decisions with a PILOT-calibrated global reset count, (3) Poisson
per-step exponential persistence as in the effective-theory approximation.

The randomized control retains the same physical checkpoint schedule and
matches the *pilot* global reset fraction to within one reset event; it draws
which living trajectories reset uniformly, without looking at displacement.
It does not match the full force autocorrelation or a per-time feedback
hazard. This cannot by itself validate any unbiased kinetic reconstruction.

Run from repository root:
 python research/checkpoint_randomized_control.py --quick
 python research/checkpoint_randomized_control.py --output results_checkpoint_control
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

if __package__:
    from .feedback_vs_exponential import Config, exact_unbiased_mfpt, simulate
else:
    from feedback_vs_exponential import Config, exact_unbiased_mfpt, simulate


@dataclass(frozen=True)
class Plan:
    forces: tuple[float, ...] = (1.0, 3.0)
    thresholds: tuple[float, ...] = (0.25, 0.55)
    seeds: tuple[int, ...] = (2026, 2027, 2028, 2029, 2030)
    dt: float = 0.005
    check_interval: float = 0.1
    horizon: float = 30.0
    replicas: int = 256


def config_for(plan: Plan, force: float, threshold: float) -> Config:
    if plan.dt <= 0 or plan.horizon <= 0 or plan.check_interval < plan.dt:
        raise ValueError('Invalid dt, horizon, or checkpoint interval')
    if plan.replicas < 2 or force < 0 or threshold < 0:
        raise ValueError('Invalid force, replicas or threshold')
    steps = round(plan.horizon / plan.dt)
    check = round(plan.check_interval / plan.dt)
    if (abs(steps*plan.dt-plan.horizon) > 1e-10 or
            abs(check*plan.dt-plan.check_interval) > 1e-10):
        raise ValueError('Physical check interval and horizon must divide dt')
    return Config(replicas=plan.replicas, steps=steps, dt=plan.dt,
                  check_steps=check, force=force, min_progress=threshold,
                  trace_replicas=min(plan.replicas, 24))


def seed_for(master: int, cell: int, replicate_kind: int) -> int:
    """Deterministically independent RNG streams across protocols and pilots."""
    return int(np.random.SeedSequence([master, cell, replicate_kind]).generate_state(1)[0])


def student_interval(values):
    """Pointwise t interval across independent seed batches; n<=10 table."""
    a=np.asarray(values,dtype=float)
    a=a[np.isfinite(a)]
    if not len(a):
        return (np.nan,np.nan,np.nan)
    avg=float(np.mean(a))
    if len(a)<2:
        return (avg,np.nan,np.nan)
    critical={1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,
              6:2.447,7:2.365,8:2.306,9:2.262}
    half=critical.get(len(a)-1,1.96)*float(np.std(a,ddof=1)/np.sqrt(len(a)))
    return (avg,avg-half,avg+half)


def run_batches(plan: Plan):
    if not plan.forces or not plan.thresholds or not plan.seeds:
        raise ValueError('Scan axes must not be empty')
    if len(set(plan.seeds))!=len(plan.seeds):
        raise ValueError('Independent seeds must be unique')
    rows=[]
    controls=[]
    analytic=exact_unbiased_mfpt(config_for(plan,0,plan.thresholds[0]))
    for k, seed in enumerate(plan.seeds):
        control=simulate(config_for(plan,0,plan.thresholds[0]),'unbiased',
                         seed_for(seed,5000,9))
        controls.append(dict(seed=seed, n=plan.replicas,
                             unbiased_rmst=control['rmst'],
                             n_censored=control['n_censored'],
                             exact_unbiased_mfpt=analytic))
    for cell, (force, threshold) in enumerate(product(plan.forces,plan.thresholds)):
        cfg=config_for(plan,force,threshold)
        for seed in plan.seeds:
            pilot=simulate(cfg,'feedback',seed_for(seed,cell,0))
            attempts=pilot['checkpoint_opportunities']
            if not attempts:
                raise RuntimeError('No eligible feedback checkpoint; extend horizon')
            p=pilot['n_resets']/attempts
            tau=pilot['tau_empirical']
            # No invention of a rate when no resets occurred.
            if tau is None:
                rows.append(dict(force=force,threshold=threshold,seed=seed,
                                 status='no_pilot_resets',n=plan.replicas,dt=cfg.dt,
                                 checkpoint_time=cfg.dt*cfg.check_steps,
                                 p_pilot=p,pilot_eligible=attempts,pilot_resets=0))
                continue
            f=simulate(cfg,'feedback',seed_for(seed,cell,1))
            randomized=simulate(cfg,'checkpoint_random',seed_for(seed,cell,2),
                                switch_probability=p)
            exponential=simulate(cfg,'exponential',seed_for(seed,cell,3),tau=tau)
            r=dict(force=force,threshold=threshold,seed=seed,status='valid',
                   n=plan.replicas,dt=cfg.dt,checkpoint_time=cfg.dt*cfg.check_steps,
                   horizon=cfg.dt*cfg.steps,
                   p_pilot=p,pilot_eligible=attempts,pilot_resets=pilot['n_resets'],
                   tau_pilot=tau)
            for label,obs in (('fb',f),('rand',randomized),('exp',exponential)):
                r[label+'_rmst']=obs['rmst']
                r[label+'_censored']=obs['n_censored']
                r[label+'_reset_fraction']=obs['checkpoint_reset_fraction']
                r[label+'_eligible']=obs['checkpoint_opportunities']
                r[label+'_resets']=obs['n_resets']
                r[label+'_out_in']=obs['persistence_ratio_out_in']
                r[label+'_basin_fx']=obs['mean_force_x_in_basin']
            r['delta_fb_rand_rmst']=f['rmst']-randomized['rmst']
            r['delta_fb_exp_rmst']=f['rmst']-exponential['rmst']
            r['delta_rand_exp_rmst']=randomized['rmst']-exponential['rmst']
            r['difference_persistence_ratio']=(f['persistence_ratio_out_in']-randomized['persistence_ratio_out_in']
                    if f['persistence_ratio_out_in'] is not None and randomized['persistence_ratio_out_in'] is not None else np.nan)
            r['rand_quota_error']=(randomized['n_resets']-
                                np.floor(p*randomized['checkpoint_opportunities']+0.5))
            rows.append(r)
    return rows,controls


def summarize(rows):
    grouped={}
    for r in rows:
        grouped.setdefault((r['force'],r['threshold']),[]).append(r)
    summary=[]
    for (force,threshold),rs in sorted(grouped.items()):
        good=[r for r in rs if r['status']=='valid']
        out=dict(force=force,threshold=threshold,n_batches=len(rs),n_valid=len(good))
        for key in ('delta_fb_rand_rmst','delta_fb_exp_rmst','delta_rand_exp_rmst',
                    'difference_persistence_ratio','fb_out_in','rand_out_in',
                    'fb_reset_fraction','rand_reset_fraction'):
            avg,lo,hi=student_interval([v[key] for v in good if v.get(key) is not None])
            out[key+'_mean']=avg;out[key+'_lo']=lo;out[key+'_hi']=hi
        out['fb_total_censored']=sum(v['fb_censored'] for v in good)
        out['rand_total_censored']=sum(v['rand_censored'] for v in good)
        out['exp_total_censored']=sum(v['exp_censored'] for v in good)
        out['max_abs_random_reset_quota_error']=(max(abs(r['rand_quota_error']) for r in good)
                                                  if good else np.nan)
        summary.append(out)
    return summary


def save_csv(path,rows):
    if not rows:
        raise ValueError('Empty result dataset')
    fields=list(dict.fromkeys(field for r in rows for field in r))
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def plot(summary,output: Path):
    fig,axes=plt.subplots(1,2,figsize=(11,4.3))
    for threshold in sorted(set(s['threshold'] for s in summary)):
        rs=sorted([s for s in summary if s['threshold']==threshold and s['n_valid']>=2],
                  key=lambda s:s['force'])
        if not rs:continue
        xx=np.array([r['force'] for r in rs])
        for ax,field in ((axes[0],'delta_fb_rand_rmst'),(axes[1],'difference_persistence_ratio')):
            mm=np.array([r[field+'_mean'] for r in rs])
            lo=np.array([r[field+'_lo'] for r in rs]);hi=np.array([r[field+'_hi'] for r in rs])
            ax.errorbar(xx,mm,yerr=np.maximum(0,np.stack([mm-lo,hi-mm])),marker='o',capsize=4,
                        label='r_min = '+str(threshold))
    for ax in axes:
        ax.axhline(0,ls='--',color='black',lw=1)
        ax.grid(alpha=.2);ax.legend()
        ax.set_xlabel('Force magnitude (dimensionless)')
    axes[0].set_ylabel('RMST feedback - checkpoint-random')
    axes[0].set_title('Effect of state-dependent switching (95% pointwise CI)')
    axes[1].set_ylabel('Out/in persistence ratio difference')
    axes[1].set_title('Direction-dependent persistence (pointwise CI)')
    fig.tight_layout();fig.savefig(output/'checkpoint_randomized_comparison.png',dpi=170)
    plt.close(fig)


def run(plan: Plan, output: Path):
    rows,controls=run_batches(plan)
    summary=summarize(rows)
    output.mkdir(parents=True,exist_ok=True)
    save_csv(output/'per_seed.csv',rows)
    save_csv(output/'conditions.csv',summary)
    save_csv(output/'unbiased_control.csv',controls)
    plot(summary,output)
    (output/'LIMITATIONS.txt').write_text(
      'This compares synthetic accelerated PROTOCOLS, not unbiased kinetic reconstruction errors.\n'
      'The randomized control matches pilot global reset frequency at identical checkpoints within one event, not time-varying hazards.\n'
      'Switch decisions are uniform random among living trajectories, not based on individual displacements.\n'
      'An exponential process is a separate time-schedule control and is not equivalent to randomized checkpoints.\n'
      'RMST includes censored paths as min(T,tmax); unrestricted MFPT is not inferred when censored.\n'
      'Outward/inward ratio is force-x sign, not actual velocity, intrinsic outward drift, or causality.\n'
      'T intervals across few seeds are uncorrected for multiple comparisons and are exploratory only.\n',
      encoding='utf-8')
    for s in summary:
        print(f"F={s['force']:.2f} rmin={s['threshold']:.2f} "
              f"FB-RAND RMST={s['delta_fb_rand_rmst_mean']:.3f} "
              f"[CI {s['delta_fb_rand_rmst_lo']:.3f}, {s['delta_fb_rand_rmst_hi']:.3f}], "
              f"OUT/IN difference={s['difference_persistence_ratio_mean']:.3f}, "
              f"censored FB/RAND={s['fb_total_censored']}/{s['rand_total_censored']}")
    return rows,summary,controls


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--quick',action='store_true',help='Tiny regression smoke, NOT a scientific result')
    p.add_argument('--forces',default='1,3')
    p.add_argument('--thresholds',default='0.25,0.55')
    p.add_argument('--seeds',default='2026,2027,2028,2029,2030')
    p.add_argument('--replicas',type=int,default=256)
    p.add_argument('--dt',type=float,default=0.005)
    p.add_argument('--check-interval',type=float,default=0.1)
    p.add_argument('--horizon',type=float,default=30.0)
    p.add_argument('--output',type=Path,default=Path('results_checkpoint_control'))
    args=p.parse_args()
    if args.quick:
        args.replicas=48;args.horizon=6.;args.seeds='2026,2027'
    parse=lambda x,typ:tuple(typ(v) for v in x.split(',') if v.strip())
    plan=Plan(forces=parse(args.forces,float),thresholds=parse(args.thresholds,float),
              seeds=parse(args.seeds,int),replicas=args.replicas,dt=args.dt,
              check_interval=args.check_interval,horizon=args.horizon)
    run(plan,args.output)


if __name__=='__main__':
    main()
