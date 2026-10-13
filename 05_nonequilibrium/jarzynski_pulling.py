"""Educational numerical test of the Jarzynski equality (Paper 6).

Overdamped Brownian particle in a harmonic trap of stiffness k, whose center
moves from 0 to displacement L during protocol time tau:
 U(x,lambda) = 0.5*k*(x-lambda)^2, lambda(t)=L*t/tau.

Initial x is sampled from the exact canonical equilibrium at lambda=0.
Inclusive work is accumulated from changes in the externally controlled
parameter lambda at fixed x, followed by overdamped Langevin integration.
Because only the *center* moves and stiffness is constant, exact Delta F=0.
The simulation is dimensionless: kB=T=mobility=1 by default.
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def run_protocol(n_traj, tau, dt, k, L, beta, mobility, seed):
    if any(v <= 0 for v in (n_traj, tau, dt, k, beta, mobility)):
        raise ValueError("Trajectories, tau, dt, k, beta and mobility must be positive")
    steps = max(1, round(tau/dt))
    h = tau/steps
    rng = np.random.default_rng(seed)
    x = rng.normal(0, np.sqrt(1/(beta*k)), size=n_traj)
    W = np.zeros(n_traj)
    noise = np.sqrt(2*mobility*h/beta)
    for step in range(steps):
        lam0 = L*step/steps
        lam1 = L*(step+1)/steps
        # Work due to protocol change at FIXED position.
        W += 0.5*k*((x-lam1)**2-(x-lam0)**2)
        # Euler-Maruyama evolution at the new parameter value.
        x += -mobility*k*(x-lam1)*h + noise*rng.normal(size=n_traj)
    return W


def free_energy_from_work(work, beta):
    z = -beta*work
    zmax = np.max(z)
    return -(zmax+np.log(np.mean(np.exp(z-zmax))))/beta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-traj',type=int,default=3000)
    ap.add_argument('--dt',type=float,default=0.005)
    ap.add_argument('--seed',type=int,default=2026)
    ap.add_argument('--output',type=Path,default=Path('results_paper6'))
    args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    beta, k, L, mobility = 1.0, 4.0, 2.0, 1.0
    protocols=[('fast',0.15),('medium',1.0),('slow',6.0)]
    rows=[]
    fig, ax=plt.subplots(figsize=(8,5))
    for idx,(name,tau) in enumerate(protocols):
        W=run_protocol(args.n_traj,tau,args.dt,k,L,beta,mobility,args.seed+idx)
        delta_f=free_energy_from_work(W,beta)
        mean_w=float(np.mean(W))
        exp_mean=float(np.mean(np.exp(-beta*W)))
        rows.append((name,tau,mean_w,delta_f,exp_mean))
        np.savetxt(args.output/f'work_{name}.csv',W,delimiter=',',header='work',comments='')
        ax.hist(W,bins=70,density=True,histtype='step',label=f'{name} (tau={tau})')
    ax.axvline(0,color='black',ls='--',label='Exact Delta F = 0')
    ax.set(xlabel='Nonequilibrium work W',ylabel='Probability density',title='Moving harmonic trap: work distributions')
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output/'work_distributions.png',dpi=200)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    labels=[r[0] for r in rows]
    positions=np.arange(len(rows))
    ax.plot(positions,[r[2] for r in rows],'o-',label='Mean work <W>')
    ax.plot(positions,[r[3] for r in rows],'s-',label='Jarzynski estimate')
    ax.axhline(0,color='black',ls='--',label='Exact Delta F = 0')
    ax.set_xticks(positions,labels)
    ax.set(ylabel='Energy',title='Dissipated mean work versus Jarzynski free energy')
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output/'free_energy_comparison.png',dpi=200)
    plt.close(fig)
    with open(args.output/'summary.csv','w',encoding='utf-8') as f:
        f.write('protocol,tau,mean_work,jarzynski_deltaF,mean_exp_minus_betaW\n')
        for row in rows:
            f.write(','.join(map(str,row))+'\n')
    for row in rows:
        print(f'{row[0]:6s} tau={row[1]:.2f}, <W>={row[2]:.3f}, Jarzynski DeltaF={row[3]:.3f}, <exp(-beta W)>={row[4]:.3f}')
    print('Exact Delta F = 0 (translation of a harmonic trap at fixed stiffness).')
    print('Finite-sample Jarzynski estimates can be noisy; rare low-work trajectories matter.')


if __name__=='__main__':
    main()
