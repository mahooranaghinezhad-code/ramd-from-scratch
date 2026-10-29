"""Protocol-controlled feedback vs exponential-persistence RAMD toy experiment.

Research prototype (2D overdamped Langevin, dimensionless units), NOT an
atomistic tauRAMD implementation and NOT a kinetic unbiasing method.

Run from repo root:
  python research/feedback_vs_exponential.py --quick --output results_feedback
  python research/feedback_vs_exponential.py --replicas 384 --output results_feedback

Physical U(x,y) = H (x^2-1)^2 + k_y y^2 / 2; reflective left x boundary,
absorbing escape at right x boundary. An independent numerical quadrature
computes the exact unbiased x first-passage mean for the stated boundaries.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Config:
    replicas: int = 384
    steps: int = 14000
    dt: float = 0.005
    height: float = 2.5
    transverse_k: float = 2.0
    kbt: float = 1.0
    force: float = 2.0
    left: float = -2.0
    right: float = 0.8
    start: float = -1.0
    check_steps: int = 20
    min_progress: float = 0.42
    trace_stride: int = 5
    trace_replicas: int = 32


def exact_unbiased_mfpt(cfg: Config, n_grid: int = 16001) -> float:
    """Backward Smoluchowski solution with reflecting left and absorbing right.

    T(x0) = 1/D * integral_x0^b exp(U(y)/kBT)
                          * [integral_a^y exp(-U(z)/kBT) dz] dy.
    This formula is exact for F=0 and separable U(x,y), with constant D=kBT.
    """
    if n_grid < 101 or cfg.start <= cfg.left or cfg.start >= cfg.right:
        raise ValueError("Invalid quadrature grid or initial position")
    grid = np.linspace(cfg.left, cfg.right, n_grid)
    energy = cfg.height * (grid * grid - 1.0) ** 2 / cfg.kbt
    w = np.exp(-energy)
    inner = np.zeros_like(grid)
    inner[1:] = np.cumsum(np.diff(grid) * (w[1:] + w[:-1]) / 2.0)
    keep = grid >= cfg.start
    return float(np.trapezoid(np.exp(energy[keep]) * inner[keep] / cfg.kbt, grid[keep]))


def feedback_switch(displacement: np.ndarray, min_progress: float) -> np.ndarray:
    """Original tauRAMD-style decision at periodic checkpoints, not each dt."""
    return np.asarray(displacement) < min_progress


def _correlation(trace: np.ndarray, dt_between: float, max_lags: int = 45):
    """Raw direction autocorrelation using only uncensored time-origin pairs."""
    values = []
    lags = np.arange(min(max_lags + 1, len(trace)))
    for lag in lags:
        first = trace[:len(trace)-lag] if lag else trace
        second = trace[lag:]
        dots = np.sum(first * second, axis=2)
        ok = np.isfinite(dots)
        values.append(float(np.mean(dots[ok])) if np.any(ok) else np.nan)
    return lags * dt_between, np.asarray(values)


def randomized_checkpoint_mask(n_eligible: int, probability: float,
                               n_seen: int, n_resets: int, rng):
    """Choose a uniform random subset at each checkpoint, without displacement.

    The number selected makes cumulative resets approximately round(p *
    cumulative eligible checkpoint opportunities), so the *global* switch
    frequency is held within one reset of the independent-pilot target.
    This is a count-matched randomized control, not a Poisson process.
    """
    if n_eligible < 0 or not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Invalid checkpoint opportunities or reset probability")
    target = int(np.floor(probability * (n_seen + n_eligible) + 0.5))
    count = target - n_resets
    if count < 0 or count > n_eligible:
        raise RuntimeError("Cumulative reset quota cannot be satisfied")
    mask = np.zeros(n_eligible, dtype=bool)
    if count:
        mask[rng.choice(n_eligible, size=count, replace=False)] = True
    return mask


def simulate(cfg: Config, protocol: str, seed: int, tau: float | None = None,
             switch_probability: float | None = None):
    """Matched-potential independent escapes; orientation reset *after* step.

    feedback: at each checkpoint, retain orientation if 2D COM displacement
              since previous checkpoint >= min_progress.
    exponential: memoryless per-step reset probability 1-exp(-dt/tau).
    checkpoint_random: at the SAME periodic checkpoints as feedback,
              uniformly randomize which alive trajectories reset, with a
              cumulative count calibrated from an INDEPENDENT feedback pilot.
              No progress/displacement is consulted for the decision.
    unbiased: zero external force; no direction resets.

    End-of-path orientation segments are right-censored; reset-rate estimates
    use total at-risk exposure / completed reset events, not completed-only means.
    """
    if protocol not in ("feedback", "exponential", "checkpoint_random", "unbiased"):
        raise ValueError("Invalid direction protocol")
    if protocol == "exponential" and (tau is None or not np.isfinite(tau) or tau <= 0):
        raise ValueError("Exponential persistence tau must be finite and positive")
    if protocol == "checkpoint_random" and (switch_probability is None or
            not np.isfinite(switch_probability) or not 0 <= switch_probability <= 1):
        raise ValueError("checkpoint_random needs a pilot reset probability in [0,1]")
    if cfg.replicas < 2 or cfg.steps < 1 or cfg.dt <= 0 or cfg.kbt <= 0 or cfg.check_steps < 1:
        raise ValueError("Invalid integration settings")
    if cfg.min_progress < 0 or cfg.right <= cfg.start or cfg.start <= cfg.left:
        raise ValueError("Invalid first-passage domain or feedback threshold")
    rng = np.random.default_rng(seed)
    n = cfg.replicas
    x = np.full(n, cfg.start)
    y = np.zeros(n)
    theta = rng.uniform(0.0, 2.0*np.pi, size=n)
    dx, dy = np.cos(theta), np.sin(theta)
    ref_x, ref_y = x.copy(), y.copy()
    alive = np.ones(n, dtype=bool)
    escape = np.full(n, np.nan)
    reset_count = np.zeros(n, dtype=int)
    exposure = np.zeros(2)
    resets = np.zeros(2, dtype=int)
    checkpoint_opportunities = 0
    basin_nx_sum = 0.0
    basin_frames = 0
    trace_n = min(cfg.trace_replicas, n)
    trace = np.full(((cfg.steps-1)//cfg.trace_stride+1, trace_n, 2), np.nan, dtype=np.float32)
    for step in range(1, cfg.steps+1):
        ids = np.flatnonzero(alive)
        if ids.size == 0:
            break
        if (step-1) % cfg.trace_stride == 0:
            j = (step-1)//cfg.trace_stride
            valid = alive[:trace_n]
            trace[j, valid, 0] = dx[:trace_n][valid]
            trace[j, valid, 1] = dy[:trace_n][valid]
        group = (dx[ids] >= 0).astype(int)  # 1 = force toward +x escape boundary.
        exposure[0] += cfg.dt*np.count_nonzero(group == 0)
        exposure[1] += cfg.dt*np.count_nonzero(group == 1)
        within_basin = (x[ids] > -1.5) & (x[ids] < -0.3)
        basin_nx_sum += float(np.sum(dx[ids][within_basin]))
        basin_frames += int(np.count_nonzero(within_basin))
        force = cfg.force if protocol != "unbiased" else 0.0
        noise = np.sqrt(2.0*cfg.kbt*cfg.dt)
        x[ids] += cfg.dt*(-4.0*cfg.height*x[ids]*(x[ids]**2-1.0) + force*dx[ids]) + noise*rng.normal(size=len(ids))
        y[ids] += cfg.dt*(-cfg.transverse_k*y[ids] + force*dy[ids]) + noise*rng.normal(size=len(ids))
        # Reflect on x=left: keep the same deterministic no-flux boundary
        # in simulations and analytic backward-equation reference.
        below = x[ids] < cfg.left
        x[ids[below]] = 2.0*cfg.left - x[ids[below]]
        escaped = x[ids] >= cfg.right
        escape[ids[escaped]] = step*cfg.dt
        alive[ids[escaped]] = False
        remaining = ids[~escaped]
        if not len(remaining) or protocol == "unbiased":
            continue
        if protocol == "exponential":
            swap = rng.random(len(remaining)) < -np.expm1(-cfg.dt/tau)
        else:
            swap = np.zeros(len(remaining), bool)
            if step % cfg.check_steps == 0:
                if protocol == "feedback":
                    progress = np.hypot(x[remaining]-ref_x[remaining], y[remaining]-ref_y[remaining])
                    swap = feedback_switch(progress, cfg.min_progress)
                else:
                    # Exact cumulative global reset QUOTA from the independent
                    # pilot, but random assignment to individuals. Timing
                    # opportunities and force/potential stay identical.
                    swap = randomized_checkpoint_mask(
                        len(remaining), switch_probability,
                        checkpoint_opportunities, int(resets.sum()), rng)
                checkpoint_opportunities += len(remaining)
                ref_x[remaining] = x[remaining]
                ref_y[remaining] = y[remaining]
        changing = remaining[swap]
        if len(changing):
            old_group = (dx[changing] >= 0).astype(int)
            resets += np.bincount(old_group, minlength=2)
            reset_count[changing] += 1
            new_theta = rng.uniform(0.0, 2.0*np.pi, len(changing))
            dx[changing], dy[changing] = np.cos(new_theta), np.sin(new_theta)
    tmax = cfg.steps*cfg.dt
    # Every trajectory is either an observed escape or administratively
    # right-censored at tmax. RMST includes both and is not unrestricted MFPT.
    observed = np.isfinite(escape)
    restricted = float(np.mean(np.where(observed, escape, tmax)))
    mfpt = float(np.mean(escape)) if observed.all() else None
    sem = float(np.std(escape, ddof=1)/np.sqrt(n)) if observed.all() else None
    tau_empirical = float(exposure.sum()/resets.sum()) if resets.sum() else None
    persistence = np.divide(exposure, resets, out=np.full(2,np.nan), where=resets>0)
    ratio = float(persistence[1]/persistence[0]) if np.isfinite(persistence).all() and persistence[0] > 0 else None
    corr_time, corr = _correlation(trace, cfg.dt*cfg.trace_stride)
    return {
        "protocol":protocol, "seed":seed, "times":escape, "n_escaped":int(observed.sum()),
        "n_censored":int((~observed).sum()), "rmst":restricted,"mfpt_if_complete":mfpt,
        "sem_if_complete":sem,"n_resets":int(resets.sum()), "reset_counts":resets,
        "exposure":exposure,"tau_empirical":tau_empirical,
        "checkpoint_opportunities":checkpoint_opportunities,
        "checkpoint_reset_fraction": (float(resets.sum()/checkpoint_opportunities)
                                      if checkpoint_opportunities else None),
        "persistence_ratio_out_in":ratio,
        "mean_force_x_in_basin": force*basin_nx_sum/max(basin_frames,1),
        "corr_t":corr_time,"corr":corr,
    }


def run(cfg: Config, seed: int, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    pilot = simulate(cfg, "feedback", seed+1)
    matched_tau = pilot["tau_empirical"]
    if matched_tau is None:
        raise RuntimeError("Feedback pilot had no direction resets; increase min_progress")
    experimental = simulate(cfg, "feedback", seed+2)
    exponential = simulate(cfg, "exponential", seed+3, tau=matched_tau)
    unbiased = simulate(cfg, "unbiased", seed+4)
    exact = exact_unbiased_mfpt(cfg)
    rows=[]
    for r in (experimental, exponential, unbiased):
        rows.append({
            "protocol":r["protocol"], "seed":r["seed"], "n_trajectories":cfg.replicas,
            "n_escaped":r["n_escaped"], "n_censored":r["n_censored"],
            "restricted_mean_to_tmax":r["rmst"],
            "mfpt_only_if_all_resolved":r["mfpt_if_complete"],
            "mfpt_standard_error_if_all_resolved":r["sem_if_complete"],
            "resets":r["n_resets"], "mean_orientation_time_per_reset":r["tau_empirical"],
            "persistence_ratio_out_in":r["persistence_ratio_out_in"],
            "mean_applied_force_x_in_basin":r["mean_force_x_in_basin"],
            "calibrated_exponential_tau":matched_tau,
            "exact_unbiased_mfpt":exact,
        })
        print(f'{r["protocol"]:11s}: escaped={r["n_escaped"]}/{cfg.replicas}, '
              f'RMST={r["rmst"]:.3f}, full MFPT={r["mfpt_if_complete"]}, '
              f'tau={r["tau_empirical"]}, outward/inward persistence={r["persistence_ratio_out_in"]}')
    with (output/"summary.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (output/"escape_times.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.writer(f);writer.writerow(["protocol","trajectory_index","escape_time","censored_at"])
        for r in (experimental, exponential, unbiased):
            for j,t in enumerate(r["times"]):
                writer.writerow([r["protocol"],j,"" if not np.isfinite(t) else t,
                                 cfg.steps*cfg.dt if not np.isfinite(t) else ""])
    with (output/"orientation_correlation.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.writer(f);writer.writerow(["lag_time","feedback","exponential","ideal_exponential"])
        for t,cf,ce in zip(experimental["corr_t"],experimental["corr"],exponential["corr"]):
            writer.writerow([t,cf,ce,np.exp(-t/matched_tau)])
    fig,axes=plt.subplots(1,2,figsize=(10.5,4))
    for r in (experimental, exponential):
        axes[0].plot(r["corr_t"],r["corr"],marker=".",label=r["protocol"])
    axes[0].plot(exponential["corr_t"],np.exp(-exponential["corr_t"]/matched_tau),"k--",label="Poisson theory")
    axes[0].set(xlabel="Lag (time units)",ylabel="Direction correlation",title="Force-direction persistence")
    axes[0].legend();axes[0].grid(alpha=0.3)
    xlabels=[r["protocol"] for r in (experimental,exponential,unbiased)]
    means=[r["rmst"] for r in (experimental,exponential,unbiased)]
    axes[1].bar(xlabels,means)
    axes[1].axhline(exact,ls="--",color="black",label="Exact unbiased MFPT")
    axes[1].set(ylabel="Restricted mean up to tmax",title="Escape (censoring-safe descriptive statistic)")
    axes[1].legend();fig.tight_layout();fig.savefig(output/"comparison.png",dpi=160);plt.close(fig)
    with (output/"README.txt").open("w",encoding="utf-8") as f:
        f.write("Educational 2D Langevin protocol comparison; not atomistic RAMD or a kinetic correction.\n")
        f.write(f"Exact analytic unbiased MFPT = {exact:.8f} from backward Smoluchowski quadrature.\n")
        f.write(f"Exponential tau calibrated from independent feedback pilot's reset exposure = {matched_tau:.8f}.\n")
        f.write("Restricted mean is NOT the unconditional MFPT if censored trajectories exist.\n")
        f.write("Outward/inward is defined by FORCE x sign, not actual progress; ratio ~1 alone does not validate a kinetic theory.\n")
        f.write("A feedback-vs-exponential mismatch is not proof of theoretical failure or ligand-specific behavior.\n")
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick",action="store_true",help="Short CI smoke test, not converged scientific result")
    p.add_argument("--replicas",type=int,default=384)
    p.add_argument("--steps",type=int,default=14000)
    p.add_argument("--force",type=float,default=2.0)
    p.add_argument("--seed",type=int,default=2026)
    p.add_argument("--output",type=Path,default=Path("results_feedback"))
    args=p.parse_args()
    if args.quick:args.replicas,args.steps=96,4500
    if args.force<0 or args.replicas<2 or args.steps<1:p.error("Invalid force or sample counts")
    cfg=Config(replicas=args.replicas,steps=args.steps,force=args.force)
    run(cfg,args.seed,args.output)


if __name__=="__main__":
    main()
