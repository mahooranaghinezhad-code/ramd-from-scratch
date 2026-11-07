"""ESS decay and published-style weight truncation in genuinely coupled 2D RAMD.

Compares conventional Gaussian stopped-path Girsanov importance weights to
an *Ionides (2008)-inspired* deterministic sqrt(N) weight truncation followed
by self-normalization. This stabilized estimator is BIASED at finite N, and
is not the Marginal Girsanov Reweighting (MGR) algorithm. We compare both
against a large, separate direct F=0 Monte Carlo reference for THE SAME
restricted mean first-passage estimand E[min(T, horizon)].

Coupled dynamics U(x,y) = H*(x*x-1)**2 + k/2*(y - coupling*x)**2,
with a slow y coordinate (mobility_y=0.4). Unlike the previous project
benchmarks, the x force depends directly on y when coupling is nonzero.
All paths are dimensionless overdamped Euler-Maruyama trajectories.
The force history is displacement-feedback (tauRAMD-like) and predictable.

Run from the repository root, with numpy and matplotlib:
    python research/ess_scaling_coupled.py --quick
    python research/ess_scaling_coupled.py --output results_ess_coupled

Scientific scope: an exploratory numerical stress test, NOT an all-atom
ligand dissociation time estimate or a validated kinetic correction.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Plan:
    couplings: tuple[float, ...] = (0.0, 1.2)
    forces: tuple[float, ...] = (0.0, 0.6, 1.2, 2.0)
    horizons: tuple[float, ...] = (2.0, 4.0, 8.0)
    test_seeds: tuple[int, ...] = (2026, 2027, 2028)
    reference_seeds: tuple[int, ...] = (50261, 50262, 50263)
    replicas: int = 192
    reference_replicas: int = 3500
    dt: float = 0.01
    check_time: float = 0.1
    min_progress: float = 0.42
    H: float = 2.5
    ky: float = 2.0
    mobility_y: float = 0.4
    kbt: float = 1.0
    left: float = -2.0
    start: float = -1.0
    right: float = 0.8


def validate(p: Plan):
    if not p.couplings or not p.forces or not p.horizons:
        raise ValueError("All physics axes must be nonempty")
    if (len(set(p.test_seeds)) != len(p.test_seeds)
        or len(set(p.reference_seeds)) != len(p.reference_seeds)
        or set(p.test_seeds) & set(p.reference_seeds)):
        raise ValueError("Independent seed splits must be disjoint")
    if min(p.forces) < 0 or not np.isfinite(p.forces).all():
        raise ValueError("Forces must be finite and nonnegative")
    if any(not np.isfinite(x) for x in (*p.horizons, *p.couplings)):
        raise ValueError("Finite inputs only")
    if (p.replicas < 2 or p.reference_replicas < 2 or p.dt <= 0
        or p.kbt <= 0 or p.H < 0 or p.ky <= 0 or p.mobility_y <= 0
        or p.min_progress < 0 or p.check_time < p.dt
        or p.left >= p.start or p.start >= p.right):
        raise ValueError("Invalid stochastic model parameters")
    if len(set(p.horizons)) != len(p.horizons) or min(p.horizons) <= 0:
        raise ValueError("Positive distinct horizons needed")
    for time_value in (*p.horizons, p.check_time):
        if abs(round(time_value / p.dt)*p.dt-time_value) > 1e-8:
            raise ValueError("Horizon and feedback checkpoint must divide dt")


def seed_for(seed: int, coupled_i: int, force_i: int, stream: int) -> int:
    return int(np.random.SeedSequence([seed, coupled_i, force_i, stream])
               .generate_state(1)[0])


def potential(x, y, coupling, p: Plan):
    q = y - coupling*x
    return p.H*(x*x-1)**2 + .5*p.ky*q*q


def gradient(x, y, coupling, p: Plan):
    q = y - coupling*x
    return (4*p.H*x*(x*x-1) - p.ky*coupling*q,
            p.ky*q)


def log_gaussian_ratio(force, dt, kbt, mobility_y, nx, ny, zx, zy):
    """EXACT Gaussian p_(F=0)/q_F log-density for one raw 2D EM step.

    x has mobility 1; y has mobility mobility_y. The force is a
    predictable history-dependent vector F*(nx,ny) chosen BEFORE
    the independent step's normal draws. Reflection of the raw x
    proposal is deterministic; this ratio is for the AUGMENTED latent
    Gaussian history (not automatically the marginal reflected path).
    """
    return (-force*np.sqrt(dt/(2*kbt))*(nx*zx+np.sqrt(mobility_y)*ny*zy)
            - force*force*dt*(nx*nx + mobility_y*ny*ny)/(4*kbt))


def importance_estimates(log_weights, restricted_times):
    """Full-path SNIS vs deterministic sqrt(N)-truncated SNIS.

    For *known normalized* p/q weights, E_q[w]=1. Ionides (2008)
    motivates a nonrandom sqrt(N) truncation rate for importance
    sampling. We cap the RAW weights at sqrt(N) and subsequently
    self-normalize; this is a simple, explicitly labeled variant,
    NOT a verbatim implementation of all of Ionides's estimators or
    a guaranteed variance improvement for any finite dataset.

    The clipping is predeclared; it is NOT tuned on test outcomes.
    """
    lw = np.asarray(log_weights, dtype=float)
    tt = np.asarray(restricted_times, dtype=float)
    if (len(lw) < 2 or lw.shape != tt.shape or
        not np.isfinite(lw).all() or not np.isfinite(tt).all()
        or np.any(tt < 0)):
        raise ValueError("Need finite matched path outcomes and weights")
    n=len(lw)
    a=float(np.max(lw))
    scaled=np.exp(lw-a)
    sum_w=float(np.sum(scaled))
    if sum_w <= 0:
        raise ValueError("All path weights vanished")
    normalized=scaled/sum_w
    snis=float(np.dot(normalized,tt))
    ess=float(1/np.dot(normalized,normalized))
    log_mean_w=float(a + np.log(sum_w/n))

    # min(exp(logw), sqrt(n)), stably scaled by exp(-a).
    cutoff=.5*np.log(n)
    clipped=np.exp(np.minimum(lw,cutoff)-max(a,cutoff))
    clip_norm=clipped/np.sum(clipped)
    clipped_est=float(np.dot(clip_norm,tt))
    clipped_ess=float(1/np.dot(clip_norm,clip_norm))
    fraction_capped=float(np.mean(lw > cutoff))
    # Raw IS remains a useful control but can have very high variance.
    raw=float(np.exp(log_mean_w)*snis) if log_mean_w < 690 else float('nan')
    return dict(raw_is_rmst=raw,snis_rmst=snis,ionides_style_rmst=clipped_est,
                ess=ess,ess_fraction=ess/n,
                clipped_ess=clipped_ess,clipped_ess_fraction=clipped_ess/n,
                capped_fraction=fraction_capped,log_mean_weight=log_mean_w,
                max_weight_share=float(np.max(normalized)))


def simulate_snapshots(p: Plan, coupling: float, force: float, seed: int,
                       replicas: int | None = None):
    """Run feedback trajectories; store stopped path stats at each horizon.

    If a particle escapes before a horizon, its time AND path likelihood
    ratio are frozen at first passage. Other particles are right-censored
    at each horizon; RMST uses min(T,h), never escaped-only averages.

    Feedback direction is identical under p0 and qF, conditional on the
    observed path and state; direction-transition factors cancel in
    augmented-path change of measure. This is NOT an underdamped MD
    path-reweighting implementation.
    """
    validate(p)
    n=p.replicas if replicas is None else int(replicas)
    if n < 2:
        raise ValueError("Need at least two independent paths")
    rng=np.random.default_rng(seed)
    x=np.full(n,p.start,dtype=float)
    y=np.full(n,-coupling,dtype=float)
    theta=rng.uniform(0,2*np.pi,n)
    nx=np.cos(theta)
    ny=np.sin(theta)
    refx=x.copy()
    refy=y.copy()
    logw=np.zeros(n)
    stop=np.full(n,np.inf)
    alive=np.ones(n,dtype=bool)
    steps=round(max(p.horizons)/p.dt)
    checkpoint_steps=round(p.check_time/p.dt)
    snapshot_steps={round(h/p.dt):h for h in p.horizons}
    saved={}
    sigma_x=np.sqrt(2*p.kbt*p.dt)
    sigma_y=np.sqrt(2*p.kbt*p.mobility_y*p.dt)
    for step in range(1,steps+1):
        ids=np.flatnonzero(alive)
        if len(ids):
            zx=rng.normal(size=len(ids))
            zy=rng.normal(size=len(ids))
            gx,gy=gradient(x[ids],y[ids],coupling,p)
            # IMPORTANT: old orientation and noise before step.
            logw[ids]+=log_gaussian_ratio(force,p.dt,p.kbt,p.mobility_y,
                                          nx[ids],ny[ids],zx,zy)
            rawx=x[ids]+p.dt*(-gx+force*nx[ids])+sigma_x*zx
            x[ids]=np.where(rawx<p.left,2*p.left-rawx,rawx)
            y[ids]+=p.dt*p.mobility_y*(-gy+force*ny[ids])+sigma_y*zy
            escaped=x[ids]>=p.right
            gone=ids[escaped]
            if len(gone):
                alive[gone]=False
                stop[gone]=step*p.dt
            remain=ids[~escaped]
            if len(remain) and step%checkpoint_steps==0:
                progress=np.hypot(x[remain]-refx[remain], y[remain]-refy[remain])
                switch=progress<p.min_progress
                refx[remain]=x[remain]
                refy[remain]=y[remain]
                changing=remain[switch]
                if len(changing):
                    angles=rng.uniform(0,2*np.pi,len(changing))
                    nx[changing],ny[changing]=np.cos(angles),np.sin(angles)
        if step in snapshot_steps:
            horizon=snapshot_steps[step]
            saved[horizon]=dict(stop=np.minimum(stop,horizon).copy(),
                                logw=logw.copy(),
                                censored=int(np.sum(stop>horizon)),
                                escaped=int(np.sum(stop<=horizon)))
    if len(saved)!=len(p.horizons):
        raise RuntimeError("Missing observation checkpoint")
    return saved


def _write_csv(path: Path, rows):
    if not rows:
        raise ValueError("No output rows")
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def mean_sem(values):
    a=np.asarray(values,dtype=float)
    if not len(a) or not np.isfinite(a).all():
        raise ValueError("Invalid reference paths")
    return float(a.mean()),float(np.std(a,ddof=1)/np.sqrt(len(a)))


def run(p:Plan, output:Path):
    validate(p)
    output.mkdir(parents=True,exist_ok=True)
    references={}
    reference_rows=[]
    for ic,coupling in enumerate(p.couplings):
        # Large independent F=0 reference; the F=0 force-direction process
        # is a spectator. Pooled independent trajectories estimate RMST and SEM.
        pools={h:[] for h in p.horizons}
        for seed in p.reference_seeds:
            ss=simulate_snapshots(p,coupling,0,seed_for(seed,ic,0,77),
                                  replicas=p.reference_replicas)
            for h in p.horizons:
                pools[h].append(ss[h]["stop"])
        for h in p.horizons:
            all_times=np.concatenate(pools[h])
            exact,sem=mean_sem(all_times)
            references[(coupling,h)]=(exact,sem)
            reference_rows.append(dict(coupling=coupling,horizon=h,
                reference_rmst=exact,reference_sem=sem,n_reference=len(all_times),
                n_censored=int(np.sum(all_times>=h)),
                reference_method="independent direct F=0 MC; NOT analytic PDE"))
    samples=[]
    for ic,coupling in enumerate(p.couplings):
        for jf,F in enumerate(p.forces):
            for seed in p.test_seeds:
                ss=simulate_snapshots(p,coupling,F,seed_for(seed,ic,jf,22))
                for h in p.horizons:
                    st=ss[h]
                    estimates=importance_estimates(st["logw"],st["stop"])
                    reference,refsem=references[(coupling,h)]
                    observed=float(st["stop"].mean())
                    row=dict(coupling=coupling,force=F,horizon=h,seed=seed,
                        n_path=p.replicas,reference_rmst=reference,reference_sem=refsem,
                        n_censored=st["censored"],biased_rmst=observed,**estimates)
                    for method,key in (("snis","snis_rmst"),
                                        ("ionides_style","ionides_style_rmst"),
                                        ("raw","raw_is_rmst"),
                                        ("biased","biased_rmst")):
                        val=row[key]
                        row["abs_error_"+method]=abs(val-reference) if np.isfinite(val) else None
                        row["abs_logerror_"+method]=(abs(np.log(val/reference))
                            if np.isfinite(val) and val>0 else None)
                    samples.append(row)
    grouped={}
    for row in samples:
        k=(row["coupling"],row["force"],row["horizon"])
        grouped.setdefault(k,[]).append(row)
    summary=[]
    for (c,f,h),items in sorted(grouped.items()):
        out=dict(coupling=c,force=f,horizon=h,n_seed_batches=len(items))
        fields=("ess_fraction","clipped_ess_fraction","log_mean_weight",
                "max_weight_share","capped_fraction","abs_error_snis",
                "abs_error_ionides_style","abs_error_raw","abs_logerror_snis",
                "abs_logerror_ionides_style","abs_logerror_biased")
        for key in fields:
            a=[float(r[key]) for r in items if r[key] is not None and np.isfinite(r[key])]
            out["mean_"+key]=(float(np.mean(a)) if a else None)
            out["sd_"+key]=(float(np.std(a,ddof=1)) if len(a)>=2 else None)
        out["n_ionides_improves_abs_error"]=sum(
            r["abs_error_ionides_style"]<r["abs_error_snis"] for r in items)
        out["reference_rmst"]=items[0]["reference_rmst"]
        out["reference_sem"]=items[0]["reference_sem"]
        out["n_censored_total"]=sum(r["n_censored"] for r in items)
        summary.append(out)
        print(f"c={c:.1f} F={f:.1f} horizon={h:.1f} "
              f"ESS/N={out['mean_ess_fraction']:.4f} "
              f"cap-ESS/N={out['mean_clipped_ess_fraction']:.4f} "
              f"mean abs err SNIS={out['mean_abs_error_snis']:.3f} "
              f"truncated={out['mean_abs_error_ionides_style']:.3f}")
    _write_csv(output/"reference_F0.csv",reference_rows)
    _write_csv(output/"per_seed.csv",samples)
    _write_csv(output/"conditions.csv",summary)
    cfg=dict(couplings=p.couplings,forces=p.forces,horizons=p.horizons,
             seeds=p.test_seeds,reference_seeds=p.reference_seeds,
             n_path=p.replicas,n_reference_paths_per_coupling=(
                 len(p.reference_seeds)*p.reference_replicas),
             dt=p.dt,check_time=p.check_time,mobility_y=p.mobility_y,
             dataset="coupled vs separable overdamped 2D",
             method="SNIS vs raw sqrt(N) truncated then self-normalized IS",
             citation="Ionides (2008), Truncated Importance Sampling, doi:10.1198/106186008X320456",
             note="not original unnormalized Ionides estimator, not MGR or PSIS")
    (output/"design.json").write_text(json.dumps(cfg,indent=2),encoding="utf-8")
    fig,ax=plt.subplots(1,2,figsize=(11,4.3))
    for coupling in p.couplings:
        for F in p.forces:
            filt=sorted((r for r in summary
                if r["coupling"]==coupling and r["force"]==F),
                key=lambda r:r["horizon"])
            ax[0].plot([r["horizon"] for r in filt],
                       [r["mean_ess_fraction"] for r in filt],
                       marker="o",label=f"c={coupling:g}, F={F:g}")
    ax[0].set(xlabel="Observation horizon",ylabel="Standard ESS / N",
              title="ESS decay with force and horizon")
    ax[0].legend(fontsize=7)
    for c in p.couplings:
        filt=sorted((r for r in summary if r["coupling"]==c
                 and r["force"]==max(p.forces)),key=lambda r:r["horizon"])
        ax[1].plot([r["horizon"] for r in filt],
            [r["mean_abs_error_snis"] for r in filt],marker="o",
            label=f"SNIS, c={c:g}")
        ax[1].plot([r["horizon"] for r in filt],
            [r["mean_abs_error_ionides_style"] for r in filt],marker="s",
            label=f"Truncated, c={c:g}")
    ax[1].set(xlabel="Observation horizon",ylabel="Mean absolute RMST error",
              title="Variance-reduction tradeoff (highest force)")
    ax[1].legend(fontsize=7)
    for a in ax:a.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output/"ess_scaling_coupled.png",dpi=170)
    plt.close(fig)
    (output/"LIMITATIONS.txt").write_text(
        "Genuinely coupled 2D x-y potential, but synthetic overdamped Euler-Maruyama dynamics.\n"
        "F=0 reference from independent direct Monte Carlo, NOT an exact PDE solution; report its SEM.\n"
        "All methods estimate SAME finite-horizon restricted E_0[min(T,h)] target.\n"
        "Gaussian likelihood ratios are exact for the specified augmented pre-reflection EM noise path.\n"
        "Truncated importance sampling with a deterministic sqrt(N) raw-weight cap is Ionides-inspired,\n"
        "with self-normalization, and is BIASED. It is NOT Marginal Girsanov or Pareto Smoothed IS.\n"
        "Truncation can reduce dispersion but INCREASE mean squared error. Never claim universal improvement.\n"
        "ESS itself is a diagnostic, NOT a proof of accuracy; small seed and path samples here are exploratory.\n"
        "No kinetic inversion from tauRAMD or experimental protein-ligand residence time is established.\n",
        encoding="utf-8")
    return reference_rows,samples,summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick",action="store_true",help="Small numerical smoke test")
    parser.add_argument("--output",type=Path,default=Path("results_ess_coupled"))
    args=parser.parse_args()
    p=Plan()
    if args.quick:
        p=Plan(couplings=(0.,1.2),forces=(0.,1.2,2.),
               horizons=(1.,2.,4.),test_seeds=(2026,2027),
               reference_seeds=(50261,50262),replicas=48,
               reference_replicas=240)
    run(p,args.output)


if __name__=="__main__":
    main()
