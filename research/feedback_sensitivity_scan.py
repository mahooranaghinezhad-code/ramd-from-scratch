"""Sensitivity study for the dimensionless feedback-vs-exponential RAMD toy.

Scan force, displacement threshold and integration timestep while holding
physical checkpoint interval and observation horizon fixed. Independently
calibrate the exponential orientation-reset rate from a feedback pilot at
EACH parameter combination, then use separate random seeds to compare
restricted first-passage means, censoring and direction-persistence proxies.

This program is a *controlled diagnostic experiment*, NOT a validated
ligand residence-time estimator or evidence of a new kinetic correction.

Quick sanity run (~8 conditions x 2 batches, few trajectories):
 python research/feedback_sensitivity_scan.py --quick --output results_sensitivity

Exploratory full scan (2 forces x 2 thresholds x 2 dt x 5 seeds):
 python research/feedback_sensitivity_scan.py --output results_sensitivity
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if __package__:
    from .feedback_vs_exponential import Config, exact_unbiased_mfpt, simulate
else:
    from feedback_vs_exponential import Config, exact_unbiased_mfpt, simulate


@dataclass(frozen=True)
class ScanPlan:
    forces: tuple[float, ...] = (1.0, 3.0)
    thresholds: tuple[float, ...] = (0.25, 0.55)
    timesteps: tuple[float, ...] = (0.005, 0.0025)
    seeds: tuple[int, ...] = (2026, 2027, 2028, 2029, 2030)
    replicas: int = 256
    horizon: float = 30.0
    check_interval: float = 0.1
    trace_interval: float = 0.025


def config_for(plan: ScanPlan, force: float, threshold: float, dt: float) -> Config:
    """Hold *physical* time scales constant when refining the discretization."""
    if not 0 < dt < plan.horizon:
        raise ValueError("Timestep must be positive and smaller than horizon")
    if plan.check_interval < dt:
        raise ValueError("Checkpoint interval must not be smaller than dt")
    return Config(replicas=plan.replicas,
                  steps=max(1, round(plan.horizon / dt)), dt=dt,
                  force=force, min_progress=threshold,
                  check_steps=max(1, round(plan.check_interval / dt)),
                  trace_stride=max(1, round(plan.trace_interval / dt)),
                  trace_replicas=min(24, plan.replicas))


def validate(plan: ScanPlan):
    if not all((plan.forces, plan.thresholds, plan.timesteps, plan.seeds)):
        raise ValueError("At least one value is required for every scan axis")
    if (min(plan.forces) < 0 or min(plan.thresholds) < 0
            or min(plan.timesteps) <= 0 or plan.replicas < 2
            or plan.horizon <= 0 or plan.check_interval <= 0
            or plan.trace_interval <= 0):
        raise ValueError("Invalid scan parameters")
    if len(set(plan.seeds)) != len(plan.seeds):
        raise ValueError("Seeds must be distinct independent batches")
    for dt in plan.timesteps:
        cfg = config_for(plan, plan.forces[0], plan.thresholds[0], dt)
        if abs(cfg.steps*dt-plan.horizon) > dt/2 + 1e-10:
            raise ValueError("Observation horizon cannot be discretized")
        if abs(cfg.check_steps*dt-plan.check_interval) > dt/2 + 1e-10:
            raise ValueError("Physical checkpoint interval cannot be discretized")


def ci95_student(values):
    """Two-sided Student-t CI of the mean across independent seed batches.

    The lookup supports n=2..10, with a normal approximation for n>10.
    It is not a replacement for convergence or a hierarchical error model.
    """
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return (float("nan"), float("nan"), float("nan"))
    mean = float(np.mean(a))
    if len(a) < 2:
        return (mean, float("nan"), float("nan"))
    t95 = {1:12.706, 2:4.303, 3:3.182, 4:2.776, 5:2.571,
           6:2.447, 7:2.365, 8:2.306, 9:2.262}
    sem = float(np.std(a, ddof=1) / np.sqrt(len(a)))
    half = t95.get(len(a)-1, 1.96)*sem
    return (mean, mean-half, mean+half)


def scan(plan: ScanPlan):
    """Return row-wise independent realizations and unbiased baseline controls.

    All raw per-seed records are retained; a missing reset rate is an explicit
    skipped condition rather than a fabricated exponential calibration.
    """
    validate(plan)
    rows = []
    ctrl = []
    exact = exact_unbiased_mfpt(Config())
    for j, dt in enumerate(plan.timesteps):
        cfg = config_for(plan, 0.0, plan.thresholds[0], dt)
        for k, seed in enumerate(plan.seeds):
            r = simulate(cfg, "unbiased", seed + 900000 + j*1000000)
            ctrl.append(dict(dt=dt, seed=seed, n=cfg.replicas,
                             tmax=cfg.steps*cfg.dt, escaped=r["n_escaped"],
                             censored=r["n_censored"], rmst=r["rmst"],
                             exact_unrestricted_mfpt=exact))
    for condition_id, (force, threshold, dt) in enumerate(product(plan.forces, plan.thresholds, plan.timesteps)):
        cfg = config_for(plan, force, threshold, dt)
        for k, seed in enumerate(plan.seeds):
            # All seeds are distinct across conditions, calibration and measurements.
            base = seed*13 + condition_id*1_000_003
            pilot = simulate(cfg, "feedback", base+1)
            tau = pilot["tau_empirical"]
            row = dict(condition_id=condition_id, seed=seed, force=force,
                       min_progress=threshold, dt=dt,
                       check_steps=cfg.check_steps,
                       physical_checkpoint_time=cfg.dt*cfg.check_steps,
                       horizon=cfg.dt*cfg.steps, n=cfg.replicas,
                       pilot_resets=pilot["n_resets"], tau_pilot=tau,
                       status="valid" if tau is not None else "no_feedback_resets")
            if tau is not None:
                fb = simulate(cfg, "feedback", base+2)
                exp = simulate(cfg, "exponential", base+3, tau=tau)
                row.update(fb_rmst=fb["rmst"], exp_rmst=exp["rmst"],
                           delta_rmst=fb["rmst"]-exp["rmst"],
                           delta_over_exact_unbiased_mfpt=(fb["rmst"]-exp["rmst"])/exact,
                           fb_escaped=fb["n_escaped"], exp_escaped=exp["n_escaped"],
                           fb_censored=fb["n_censored"], exp_censored=exp["n_censored"],
                           fb_tau=fb["tau_empirical"], exp_tau=exp["tau_empirical"],
                           fb_out_in=fb["persistence_ratio_out_in"],
                           exp_out_in=exp["persistence_ratio_out_in"],
                           fb_basin_force_x=fb["mean_force_x_in_basin"],
                           exp_basin_force_x=exp["mean_force_x_in_basin"],
                           fb_resets=fb["n_resets"], exp_resets=exp["n_resets"])
            rows.append(row)
    return rows, ctrl


def aggregate(rows):
    grouped = {}
    for row in rows:
        key = (row["force"], row["min_progress"], row["dt"])
        grouped.setdefault(key, []).append(row)
    results = []
    for (force, threshold, dt), values in sorted(grouped.items()):
        successful = [v for v in values if v["status"] == "valid"]
        result = dict(force=force, min_progress=threshold, dt=dt,
                      n_seed_batches=len(values), n_valid_batches=len(successful),
                      n_skipped_batches=len(values)-len(successful))
        for source, prefix in (("delta_rmst","delta_rmst"),
                               ("delta_over_exact_unbiased_mfpt","normalized_delta"),
                               ("fb_out_in","feedback_out_in"),
                               ("exp_out_in","exponential_out_in"),
                               ("fb_basin_force_x","feedback_basin_fx"),
                               ("exp_basin_force_x","exponential_basin_fx")):
            triplet = ci95_student([v[source] for v in successful
                                    if v.get(source) is not None])
            result[prefix+"_mean"],result[prefix+"_ci_low"],result[prefix+"_ci_high"] = triplet
        result["fb_total_censored"] = sum(v.get("fb_censored", 0) for v in successful)
        result["exp_total_censored"] = sum(v.get("exp_censored", 0) for v in successful)
        results.append(result)
    return results


def _save_csv(path: Path, rows):
    if not rows:
        raise ValueError("No rows to write")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w",newline="",encoding="utf-8") as fh:
        wr=csv.DictWriter(fh,fieldnames=fields)
        wr.writeheader();wr.writerows(rows)


def plot_results(aggregates, plan: ScanPlan, output: Path):
    """Plots descriptive differences with across-seed t intervals, not proof."""
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.3))
    for threshold, dt in product(plan.thresholds, plan.timesteps):
        r = sorted([v for v in aggregates if v["min_progress"] == threshold
                    and v["dt"] == dt and v["n_valid_batches"] >= 2],
                   key=lambda v:v["force"])
        if not r: continue
        x=np.array([v["force"] for v in r])
        label=f"rmin={threshold:g}, dt={dt:g}"
        for ax, key in ((axs[0], "delta_rmst"), (axs[1], "feedback_out_in")):
            m=np.array([v[key+"_mean"] for v in r]);low=np.array([v[key+"_ci_low"] for v in r]);high=np.array([v[key+"_ci_high"] for v in r])
            ax.errorbar(x,m,yerr=np.maximum(0,np.vstack((m-low,high-m))),marker="o",capsize=3,label=label)
    axs[0].axhline(0, color="black", linewidth=1, linestyle="--")
    axs[0].set(xlabel="Force (dimensionless)",ylabel="Feedback - exponential RMST",
               title="Protocol difference (t interval across seeds)")
    axs[1].axhline(1,color="black", linewidth=1,linestyle="--")
    axs[1].set(xlabel="Force (dimensionless)",ylabel="Feedback outward/inward exposure per reset",
               title="Directional persistence proxy")
    for ax in axs:
        ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(output/"sensitivity_scan.png",dpi=170);plt.close(fig)


def run(plan: ScanPlan, output: Path):
    rows, ctrl = scan(plan)
    agg=aggregate(rows)
    output.mkdir(parents=True,exist_ok=True)
    _save_csv(output/"per_seed.csv",rows)
    _save_csv(output/"conditions.csv",agg)
    _save_csv(output/"unbiased_control.csv",ctrl)
    plot_results(agg,plan,output)
    for r in agg:
        print(f"F={r['force']:.2g} rmin={r['min_progress']:.2g} dt={r['dt']:.4g} "
              f"delta_RMST={r['delta_rmst_mean']:.3f} "
              f"95%CI=[{r['delta_rmst_ci_low']:.3f},{r['delta_rmst_ci_high']:.3f}] "
              f"censored F/E={r['fb_total_censored']}/{r['exp_total_censored']} ")
    (output/"LIMITATIONS.txt").write_text(
        "Synthetic 2D overdamped Langevin model only.\n"
        "The reported difference compares accelerated-protocol restricted escape times; "
        "it is NOT an error of unbiased kinetic extrapolation.\n"
        "The exact analytic MFPT is valid ONLY for the F=0 control.\n"
        "95% Student-t intervals across independent seed batches are fragile for small n.\n"
        "All censored paths contribute as min(T,tmax) to RMST; unrestricted MFPT "
        "is not estimated from censored data.\n"
        "Matching global reset frequency does not match orientation autocorrelations.\n"
        "No convergence or generalization claim is warranted without further tests.\n",
        encoding="utf-8")
    return rows, agg, ctrl


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick",action="store_true",help="Very small CI smoke, not a scientific result")
    p.add_argument("--forces",default="1,3")
    p.add_argument("--thresholds",default="0.25,0.55")
    p.add_argument("--dts",default="0.005,0.0025")
    p.add_argument("--seeds",default="2026,2027,2028,2029,2030")
    p.add_argument("--replicas",type=int,default=256)
    p.add_argument("--horizon",type=float,default=30.0)
    p.add_argument("--check-interval",type=float,default=.1)
    p.add_argument("--output",type=Path,default=Path("results_feedback_sweep"))
    args=p.parse_args()
    if args.quick:
        args.replicas=48;args.horizon=8.0;args.seeds="2026,2027"
    parse=lambda string,typ:tuple(typ(x) for x in string.split(",") if x.strip())
    plan=ScanPlan(forces=parse(args.forces,float),thresholds=parse(args.thresholds,float),
                  timesteps=parse(args.dts,float),seeds=parse(args.seeds,int),
                  replicas=args.replicas,horizon=args.horizon,
                  check_interval=args.check_interval)
    run(plan,args.output)


if __name__=="__main__":
    main()
