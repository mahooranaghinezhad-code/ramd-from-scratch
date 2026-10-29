"""
Paper 1 — Ding, Ghidini, Cossio & Roux (2026)
A theoretical framework for random accelerated molecular dynamics simulations

Educational numerical experiments with the article's *simplified* overdamped
model. NOT a tauRAMD displacement-triggered algorithm, all-atom reproduction,
or validated correction for ligand residence times.

Models a force of fixed magnitude whose direction is reset after exponentially
distributed persistence times (Poisson resetting at rate 1/tau).

In dimensionless units kBT = 1:
   gamma dr/dt = -grad W + F_RAMD n(t) + Gaussian thermal force

Paper Eq. (5):
   D_eff = D + F_RAMD^2 * tau / (d * gamma^2)
where D = kBT / gamma.

Eq. (7): kB T_eff = kBT + F_RAMD^2 * tau/(d gamma).

We compare MSD-based free diffusion, harmonic-well fluctuations, and measured
first-passage escape times versus a simple effective-Smoluchowski prediction.
The escape comparison is deliberately a *diagnostic*: it is not the exact
barrier model or ligand-specific estimate used in the article.
"""
import argparse
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

KBT = 1.0
GAMMA = 1.0
TAU = 0.08
DIM = 2
FORCES = (0., 1.5, 3., 4.5, 6.)
DT = 0.002
D0 = KBT / GAMMA


def unit_directions(rng, n, dim):
    z = rng.normal(size=(n, dim))
    return z / np.linalg.norm(z, axis=1, keepdims=True)


def theory(force, tau=TAU, dim=DIM):
    diffusion = D0 + force**2 * tau/(dim*GAMMA**2)
    teff = KBT + force**2 * tau/(dim*GAMMA)
    return diffusion, teff


def step(x, direction, force, dt, rng, drift=None, tau=TAU):
    """Vectorized overdamped Langevin with exponential direction persistence."""
    switches = rng.random(len(x)) < -np.expm1(-dt/tau)
    if np.any(switches):
        direction[switches] = unit_directions(rng, switches.sum(), x.shape[1])
    physical = 0. if drift is None else drift(x)
    x += dt*(physical+force*direction)/GAMMA + np.sqrt(2*D0*dt)*rng.normal(size=x.shape)
    return x


def free_diffusion(force, rng, n=1800, steps=5000, dt=DT):
    """Estimate D from end-to-end MSD: <|r(t)-r(0)|²> / (2 d t)."""
    x = np.zeros((n, DIM))
    direction = unit_directions(rng, n, DIM)
    for _ in range(steps):
        step(x, direction, force, dt, rng)
    t = steps*dt
    return float(np.mean(np.sum(x*x, axis=1))/(2*DIM*t))


def harmonic_temperature(force, rng, k=2.0, n=1200, steps=6500, burn=1500, dt=DT):
    """Use Eq. (20): kBT_eff ~= k <|r-r_mean|²> / d for harmonic W."""
    x = np.zeros((n, DIM))
    direction = unit_directions(rng, n, DIM)
    samples = []
    for i in range(steps):
        step(x, direction, force, dt, rng, drift=lambda q: -k*q)
        if i >= burn and i % 40 == 0:
            samples.append(x.copy())
    arr = np.concatenate(samples, axis=0)
    return float(k*np.mean(np.sum((arr-arr.mean(axis=0))**2, axis=1))/DIM)


def barrier_energy(x, height=5.0, transverse_k=2.0):
    """A double-well in x and harmonic confinement in y."""
    return height*(x[..., 0]**2-1)**2 + 0.5*transverse_k*x[..., 1]**2


def barrier_drift(q, height=5.0, transverse_k=2.0):
    out = np.empty_like(q)
    out[:, 0] = -4*height*q[:, 0]*(q[:, 0]**2-1)
    out[:, 1] = -transverse_k*q[:, 1]
    return out


def escape_simulation(force, rng, n=450, steps=16000, dt=DT):
    """First hitting of x>=0.85 starting near x=-1; right-censor unresolved."""
    x = np.zeros((n, DIM))
    x[:, 0] = -1.
    direction = unit_directions(rng, n, DIM)
    alive = np.ones(n, bool)
    times = np.full(n, np.nan)
    for i in range(steps):
        inds = np.flatnonzero(alive)
        if len(inds) == 0:
            break
        subset = x[inds].copy()
        dirs = direction[inds].copy()
        step(subset, dirs, force, dt, rng, drift=barrier_drift)
        x[inds] = subset
        direction[inds] = dirs
        reached = subset[:, 0] >= 0.85
        times[inds[reached]] = (i+1)*dt
        alive[inds[reached]] = False
    return times


def effective_smolu_mfpt(force, height=5., left=-2., start=-1., right=0.85):
    """
    The standard 1D reflecting-left / absorbing-right Smoluchowski MFPT:
      MFPT(start) = integral_start^right dy exp(beta_eff U(y))/D_eff
                    * integral_left^y dz exp(-beta_eff U(z)).
    Numerical quadrature by cumulative trapezoids. Only a model comparison;
    transverse coordinate and finite directional-memory effects are neglected.
    """
    diffusion, teff = theory(force)
    xs = np.linspace(left, right, 3501)
    U = height*(xs*xs-1.)**2
    # Improve stability by removing a constant offset in each Boltzmann factor.
    # U is bounded on this finite grid.
    a = np.exp(-U/teff)
    inner = np.zeros(len(xs))
    inner[1:] = np.cumsum((a[1:]+a[:-1])*np.diff(xs)/2)
    outer = np.exp(U/teff)*inner/diffusion
    mask = xs >= start
    return float(np.trapezoid(outer[mask], xs[mask]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--output',type=Path,default=Path('results_paper1'))
    args = p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    rng = np.random.default_rng(args.seed)
    records = []
    for f in FORCES:
        d_model,t_model = theory(f)
        d_obs = free_diffusion(f,rng)
        temp_obs = harmonic_temperature(f,rng)
        t_escape = escape_simulation(f,rng)
        n_escaped = int(np.isfinite(t_escape).sum())
        observed_mean = np.nan if n_escaped < len(t_escape) else float(np.mean(t_escape))
        prediction = effective_smolu_mfpt(f)
        records.append((f,d_model,d_obs,t_model,temp_obs,n_escaped,
                        len(t_escape),observed_mean,prediction))
        print(f'F={f:4.1f}  D: theory={d_model:.3f}, MSD={d_obs:.3f}; '
              f'kBT_eff: theory={t_model:.3f}, well={temp_obs:.3f}; '
              f'escaped={n_escaped}/{len(t_escape)}')
        if n_escaped < len(t_escape):
            print('  Note: observed escape mean omitted because some paths were censored.')
    header=('force,D_eff_theory,D_eff_MSD,kBT_eff_theory,kBT_eff_harmonic,'
            'n_escaped,n_total,mean_escape_if_all_resolved,Smoluchowski_MFPT')
    with open(args.output/'results.csv','w',newline='',encoding='utf-8') as fh:
        writer=csv.writer(fh)
        writer.writerow(header.split(','))
        writer.writerows(records)
    a=np.array(records,dtype=float)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    axes[0].plot(a[:,0],a[:,1],'-o',label='Eq. 5')
    axes[0].plot(a[:,0],a[:,2],'s',label='Simulated MSD')
    axes[0].set(xlabel='F_RAMD',ylabel='D_eff',title='Effective diffusion')
    axes[1].plot(a[:,0],a[:,3],'-o',label='Eq. 7')
    axes[1].plot(a[:,0],a[:,4],'s',label='Harmonic fluctuations')
    axes[1].set(xlabel='F_RAMD',ylabel='kBT_eff',title='Effective temperature')
    axes[2].semilogy(a[:,0],a[:,8],'-o',label='Effective Smoluchowski MFPT')
    good=np.isfinite(a[:,7])
    axes[2].semilogy(a[good,0],a[good,7],'s',label='Simulated mean (fully resolved only)')
    axes[2].set(xlabel='F_RAMD',ylabel='Time',title='Escape: approximation vs simulation')
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(args.output/'ramd_theory_comparison.png',dpi=200)
    plt.close(fig)
    print('Saved',args.output)
    print('LIMITATION: Finite tau, numerical timestep, censoring, and reaction-coordinate '
          'projection can all affect kinetics; this is not a validated unbiasing method.')


if __name__=='__main__':
    main()
