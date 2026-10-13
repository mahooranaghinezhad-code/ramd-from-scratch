"""Experimental research-proposal benchmark, not a validated RAMD method.

2D coupled potential U(x,y)=a(x²-1)² + k_y(y-c*x)²/2.
Independent overdamped mobilities let y become a hidden slow coordinate.
Estimate unbiased reference MFPT, force-accelerated MFPT, a simple
effective-temperature extrapolation, and committor-projection quality.
Stochastic force directions reset with exponential-time persistence,
NOT the displacement-triggered tauRAMD protocol.

Run: python research/memory_rc_validity_benchmark.py --quick
Full: python research/memory_rc_validity_benchmark.py --replicas 400 --max-steps 250000
Requires numpy, matplotlib.
"""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

KBT, A, KY, COUPLING = 1., 3., 3., 1.
DT, TAU, BOUNDARY = .008, .12, .8
FORCES = (1.5, 2.5, 3.5, 4.5)


def gradient(x, y):
    delta = y - COUPLING*x
    return 4*A*x*(x*x-1) - KY*COUPLING*delta, KY*delta


def simulate(n, max_steps, dt, force, mobility_y, seed):
    """EM overdamped dynamics with geometric approximations to exp dwell times."""
    rng = np.random.default_rng(seed)
    x = np.full(n, -1.)
    y = np.full(n, -COUPLING)
    angles = rng.uniform(0, 2*np.pi, n)
    alive = np.ones(n, bool)
    escape = np.full(n, np.nan)
    for step in range(1, max_steps+1):
        idx = np.flatnonzero(alive)
        if not len(idx):
            break
        flips = rng.random(len(idx)) < -np.expm1(-dt/TAU)
        if flips.any():
            angles[idx[flips]] = rng.uniform(0, 2*np.pi, flips.sum())
        gx, gy = gradient(x[idx], y[idx])
        x[idx] += (-gx + force*np.cos(angles[idx]))*dt + np.sqrt(2*KBT*dt)*rng.normal(size=len(idx))
        y[idx] += mobility_y*(-gy + force*np.sin(angles[idx]))*dt + np.sqrt(2*KBT*mobility_y*dt)*rng.normal(size=len(idx))
        hit = x[idx] >= BOUNDARY
        escape[idx[hit]] = step*dt
        alive[idx[hit]] = False
    return escape


def mfpt_if_complete(times):
    """Do not take an escaped-only mean when observations are right-censored."""
    return float(np.mean(times)) if np.isfinite(times).all() else np.nan


def alpha(force):
    """Paper 1 Eq. 7-inspired baseline (d=2, gamma_x=1); anisotropic caveat."""
    return KBT/(KBT + force*force*TAU/2)


def extrapolate(forces, means):
    """Extrapolate log mean escape time linearly to alpha=1."""
    if len(means)<3 or not np.isfinite(means).all():
        return np.nan, np.nan
    slope, intercept = np.polyfit([alpha(f) for f in forces], np.log(means), 1)
    return float(np.exp(slope+intercept)), float(slope)


def committor_quality(mobility_y, seed, shots=32, max_steps=4500):
    """Independent unbiased shooting trajectories; smaller bin variance is better."""
    rng = np.random.default_rng(seed)
    grid = np.array([(x, y) for x in np.linspace(-.6, .6, 7)
                             for y in np.linspace(-1.3, 1.3, 5)])
    p = []
    for x0, y0 in grid:
        x = np.full(shots, x0)
        y = np.full(shots, y0)
        outcomes = np.full(shots, -1, int)
        for _ in range(max_steps):
            ids = np.flatnonzero(outcomes < 0)
            if not len(ids):
                break
            gx, gy = gradient(x[ids], y[ids])
            x[ids] += -gx*DT + np.sqrt(2*KBT*DT)*rng.normal(size=len(ids))
            y[ids] += -mobility_y*gy*DT + np.sqrt(2*KBT*mobility_y*DT)*rng.normal(size=len(ids))
            outcomes[ids[x[ids] <= -BOUNDARY]] = 0
            outcomes[ids[x[ids] >= BOUNDARY]] = 1
        resolved = outcomes >= 0
        p.append(np.mean(outcomes[resolved]) if resolved.any() else np.nan)
    p = np.asarray(p)
    def loss(coordinate):
        edges = np.linspace(coordinate.min()-1e-9, coordinate.max()+1e-9, 8)
        b = np.digitize(coordinate, edges) - 1
        error = 0.
        count = 0
        for j in range(7):
            q = p[(b == j) & np.isfinite(p)]
            if len(q)>1:
                error += len(q)*np.var(q)
                count += len(q)
        return error/count if count else np.nan
    return loss(grid[:,0]), loss(grid[:,1]), float(np.isfinite(p).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--replicas', type=int, default=180)
    ap.add_argument('--max-steps', type=int, default=100000)
    ap.add_argument('--dt', type=float, default=DT)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--output', type=Path, default=Path('results_validity'))
    args = ap.parse_args()
    if args.replicas < 2 or args.max_steps < 1 or args.dt <= 0:
        ap.error('Invalid counts or timestep')
    if args.quick:
        args.replicas, args.max_steps = 24, 8000
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    summary = []
    for condition, mu_y in [('fast_hidden', 2.), ('medium_hidden', .5),
                             ('slow_hidden', .08)]:
        seed = args.seed + int(mu_y*10000)
        t0 = simulate(args.replicas, args.max_steps, args.dt, 0, mu_y, seed)
        true_t = mfpt_if_complete(t0)
        rows.append((condition,mu_y,0,1,np.isfinite(t0).sum(),len(t0),true_t))
        means = []
        for j, force in enumerate(FORCES):
            t = simulate(args.replicas,args.max_steps,args.dt,force,mu_y,seed+100+j)
            means.append(mfpt_if_complete(t))
            rows.append((condition,mu_y,force,alpha(force),np.isfinite(t).sum(),len(t),means[-1]))
        estimated, slope = extrapolate(FORCES,means)
        error = abs(np.log(estimated/true_t)) if np.isfinite(estimated) and np.isfinite(true_t) else np.nan
        qx,qy,coverage = committor_quality(
            mu_y, seed+500, shots=12 if args.quick else 32,
            max_steps=1500 if args.quick else 4500)
        summary.append((condition,mu_y,1/(KY*mu_y),true_t,estimated,error,slope,qx,qy,coverage))
        print(condition,'reference',true_t,'reconstructed',estimated,'error',error)
    with open(args.output/'benchmark.csv','w',newline='') as f:
        w=csv.writer(f)
        w.writerow(['condition','mobility_y','force','alpha_baseline','n_escaped','n_total','mfpt_if_complete'])
        w.writerows(rows)
    with open(args.output/'summary.csv','w',newline='') as f:
        w=csv.writer(f)
        w.writerow(['condition','mobility_y','approx_tau_y','unbiased_mfpt','reconstructed_mfpt','abs_log_error','fit_slope','committor_loss_x','committor_loss_y','committor_coverage'])
        w.writerows(summary)
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for condition in [r[0] for r in summary]:
        rr=[r for r in rows if r[0]==condition and r[2]>0 and np.isfinite(r[6])]
        if rr:
            axs[0].plot([r[3] for r in rr],np.log([r[6] for r in rr]),'o-',label=condition)
    axs[0].set(xlabel='alpha = T/T_eff (baseline)',ylabel='log MFPT',title='RAMD force extrapolation')
    axs[0].legend()
    axs[1].bar([r[0] for r in summary],[r[5] for r in summary])
    axs[1].set(ylabel='|log(t_predicted/t_reference)|',title='Error vs hidden-coordinate mobility')
    axs[1].tick_params(axis='x',rotation=20)
    fig.tight_layout()
    fig.savefig(args.output/'validity_map.png',dpi=200)
    plt.close(fig)
    (args.output/'NOTES.txt').write_text(
        'Research prototype, NOT a validated RAMD correction.\n'
        'Censored paths yield NaN, never an escaped-only mean.\n'
        'Committor shooting and within-bin scores are noisy preliminary diagnostics.\n'
        'The effective-temperature extrapolation is an approximation for anisotropic mobility.\n'
        'Held-out systems, convergence, censoring-aware estimators, and uncertainties are needed.\n')
if __name__=='__main__':
    main()
