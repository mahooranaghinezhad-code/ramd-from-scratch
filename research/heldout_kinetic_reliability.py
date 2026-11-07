"""Frozen switching classifier transfer and kinetic extrapolation on held-out wells.

Train the next-checkpoint state and time-only logistic models ONCE on an
H=2.5 double well; evaluate without refitting on other barrier heights and
force magnitudes. Separately, measure whether a heuristic effective-temperature
force-extrapolation can recover a known exact F=0 mean first-passage time.

Crucial: the extrapolation is a deliberately *tested heuristic*,
NOT a validated Roux/Ding inversion formula or a new kinetic estimator.
The classifier predicts a switch conditional on survival; its log-loss
gain does not automatically measure kinetic-reconstruction reliability.

Run from repository root:
  python research/heldout_kinetic_reliability.py --quick
  python research/heldout_kinetic_reliability.py --output results_heldout
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if __package__:
    from .feedback_vs_exponential import Config, exact_unbiased_mfpt, simulate
    from .state_conditioned_switching import (
        RidgeLogistic, record_checkpoints, features, metrics,
    )
else:
    from feedback_vs_exponential import Config, exact_unbiased_mfpt, simulate
    from state_conditioned_switching import (
        RidgeLogistic, record_checkpoints, features, metrics,
    )


@dataclass(frozen=True)
class Plan:
    source_height: float = 2.5
    training_forces: tuple[float, ...] = (0.7, 1.6, 2.7)
    target_heights: tuple[float, ...] = (1.5, 2.0, 3.0, 3.5)
    target_forces: tuple[float, ...] = (1.0, 1.9, 2.8)
    threshold: float = 0.42
    training_seeds: tuple[int, ...] = (2026, 2027, 2028, 2029)
    testing_seeds: tuple[int, ...] = (2031, 2032, 2033)
    replicas: int = 96
    dt: float = 0.01
    horizon: float = 80.0
    checkpoint_time: float = 0.1
    penalty: float = 0.002


def validate(plan: Plan):
    if not plan.training_forces or len(plan.target_forces) < 3 or len(plan.target_heights) < 2:
        raise ValueError("Need nonempty training forces and >=3 test forces, >=2 target heights")
    if not plan.training_seeds or not plan.testing_seeds:
        raise ValueError("Both seed splits are required")
    if (set(plan.training_seeds) & set(plan.testing_seeds)
        or len(set(plan.training_seeds)) != len(plan.training_seeds)
        or len(set(plan.testing_seeds)) != len(plan.testing_seeds)):
        raise ValueError("Training and testing seed batches must be disjoint and unique")
    if (plan.source_height in plan.target_heights
        or set(plan.training_forces) & set(plan.target_forces)
        or len(set(plan.target_heights)) != len(plan.target_heights)):
        raise ValueError("Physical target heights and force levels must be held out")
    if (plan.replicas < 2 or plan.dt <= 0 or plan.horizon <= plan.dt
        or plan.checkpoint_time < plan.dt or plan.threshold < 0
        or plan.source_height <= 0 or min(plan.target_heights) <= 0
        or min(plan.training_forces) < 0 or min(plan.target_forces) <= 0
        or plan.penalty < 0):
        raise ValueError("Invalid physics or experiment parameters")
    for time_value in (plan.horizon, plan.checkpoint_time):
        if abs(round(time_value / plan.dt) * plan.dt - time_value) > 1e-8:
            raise ValueError("Horizon and checkpoint period must divide timestep")


def cfg_for(plan: Plan, height: float, force: float) -> Config:
    return Config(height=float(height), force=float(force),
                  min_progress=plan.threshold, replicas=plan.replicas,
                  dt=plan.dt, steps=round(plan.horizon / plan.dt),
                  check_steps=round(plan.checkpoint_time / plan.dt),
                  trace_replicas=0)


def seed_for(seed: int, a: int, b: int, stream: int) -> int:
    """Independent deterministic RNG streams, including calibration pilots."""
    return int(np.random.SeedSequence([seed, a, b, stream]).generate_state(1)[0])


def fit_source_models(plan: Plan):
    """Fit exactly ONCE using only source H and training forces/seeds."""
    collected = []
    n_events = 0
    for force_index, force in enumerate(plan.training_forces):
        cfg = cfg_for(plan, plan.source_height, force)
        for seed in plan.training_seeds:
            r = record_checkpoints(cfg, "feedback",
                                   seed_for(seed, 0, force_index, 1))
            if not r["events"]:
                raise RuntimeError("No source checkpoint outcomes; extend horizon")
            X = r["raw"]
            collected.append((features(X, cfg, "time"),
                              features(X, cfg, "state"), X[:, 8]))
            n_events += len(X)
    x_time = np.concatenate([item[0] for item in collected])
    x_state = np.concatenate([item[1] for item in collected])
    labels = np.concatenate([item[2] for item in collected])
    time_model = RidgeLogistic(plan.penalty).fit(x_time, labels)
    state_model = RidgeLogistic(plan.penalty).fit(x_state, labels)
    constant_rate = float(np.clip(np.mean(labels), 1e-6, 1 - 1e-6))
    return time_model, state_model, constant_rate, n_events


def effective_alpha(force: float, pilot_tau: float, kbt=1.0) -> float:
    """Dimensionless effective-temperature proxy in d=2, gamma=1.

    alpha = T/T_eff = kBT / (kBT + F^2 * tau/(2*gamma)).
    For feedback switching, inserting an empirical tau is NOT an
    established fluctuation-dissipation or kinetic-unbiasing relation.
    """
    if force < 0 or not np.isfinite(pilot_tau) or pilot_tau <= 0 or kbt <= 0:
        raise ValueError("Invalid alpha parameters")
    return float(kbt / (kbt + force * force * pilot_tau / 2.0))


def kinetic_extrapolation(alphas, mean_forced_times):
    """Heuristic: log(mean T_F) = intercept + slope * alpha(F).

    Extrapolate to alpha=1 (F=0). Not an exact result of RAMD theory.
    Requires ALL trajectories in ALL force ensembles to have escaped,
    so inputs with censoring cannot be used for unrestricted MFPT.
    """
    a = np.asarray(alphas, dtype=float)
    t = np.asarray(mean_forced_times, dtype=float)
    if len(a) < 3 or a.shape != t.shape or not np.isfinite(a).all():
        return None
    if not np.isfinite(t).all() or np.any(t <= 0) or np.ptp(a) < 0.015:
        return None
    slope, intercept = np.polyfit(a, np.log(t), 1)
    estimate = float(np.exp(intercept + slope))
    return estimate if np.isfinite(estimate) and estimate > 0 else None


def rank(values):
    """Rank with mean ranks for ties (no SciPy dependency)."""
    a = np.asarray(values, dtype=float)
    ix = np.argsort(a, kind="stable")
    s = a[ix]
    r = np.empty(len(a), dtype=float)
    i = 0
    while i < len(a):
        j = i + 1
        while j < len(a) and s[j] == s[i]:
            j += 1
        r[ix[i:j]] = (i + j - 1) / 2 + 1
        i = j
    return r


def spearman(x, y):
    """Descriptive rank association across landscapes, no p-value."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return None
    a, b = rank(x[valid]), rank(y[valid])
    if np.std(a) < 1e-14 or np.std(b) < 1e-14:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def write_csv(path: Path, rows):
    if not rows:
        raise ValueError("No output to write")
    names = list(dict.fromkeys(key for r in rows for key in r))
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(plan: Plan, output: Path):
    validate(plan)
    output.mkdir(parents=True, exist_ok=True)
    time_model, state_model, constant_rate, n_source_events = fit_source_models(plan)
    np.savez(output / "frozen_source_models.npz",
             time_coef=time_model.w, time_center=time_model.center,
             time_scale=time_model.scale, state_coef=state_model.w,
             state_center=state_model.center, state_scale=state_model.scale,
             training_height=plan.source_height,
             training_forces=np.asarray(plan.training_forces))
    event_rows = []
    seed_rows = []
    landscape_rows = []

    for landscape_id, height in enumerate(plan.target_heights):
        reference_cfg = cfg_for(plan, height, 0.0)
        exact = exact_unbiased_mfpt(reference_cfg)
        if not np.isfinite(exact) or exact <= 0:
            raise RuntimeError("Invalid exact unbiased MFPT reference")
        for seed in plan.testing_seeds:
            force_rows = []
            for force_id, force in enumerate(plan.target_forces):
                cfg = cfg_for(plan, height, force)
                # The independent pilot sets only a physical protocol
                # persistence parameter. Neither logistic model is re-fitted.
                pilot = simulate(cfg, "feedback",
                                 seed_for(seed, landscape_id+100, force_id, 0))
                tau = pilot["tau_empirical"]
                if tau is None:
                    raise RuntimeError("Feedback pilot has no resets; change threshold")
                fb = record_checkpoints(cfg, "feedback",
                                        seed_for(seed, landscape_id+100, force_id, 1))
                exp = simulate(cfg, "exponential",
                               seed_for(seed, landscape_id+100, force_id, 2), tau=tau)
                data = fb["raw"]
                if not len(data):
                    raise RuntimeError("No held-out checkpoint outcomes")
                y = data[:, 8]
                pred_t = time_model.predict(features(data, cfg, "time"))
                pred_s = state_model.predict(features(data, cfg, "state"))
                ll_t, br_t = metrics(y, pred_t)
                ll_s, br_s = metrics(y, pred_s)
                ll_const, _ = metrics(y, np.full(len(y), constant_rate))
                alpha = effective_alpha(force, tau, cfg.kbt)
                # IMPORTANT: None when even one escape is right-censored.
                fb_all = bool(np.isfinite(fb["escape_times"]).all())
                exp_all = bool(np.isfinite(exp["times"]).all())
                fb_mean = (float(np.mean(fb["escape_times"])) if fb_all else None)
                exp_mean = (float(np.mean(exp["times"])) if exp_all else None)
                row = dict(height=height, force=force, seed=seed,
                           n_checkpoint_events=len(data),
                           n_fb_censored=cfg.replicas-fb["escapes"],
                           n_exp_censored=exp["n_censored"],
                           tau_feedback_pilot=tau, alpha_effective_proxy=alpha,
                           logloss_constant=ll_const, logloss_time=ll_t,
                           logloss_state=ll_s,
                           gain_state_vs_time=ll_t-ll_s,
                           gain_state_vs_constant=ll_const-ll_s,
                           brier_time=br_t, brier_state=br_s,
                           mean_fb_if_complete=fb_mean,
                           mean_exp_if_complete=exp_mean,
                           exact_unbiased_mfpt=exact)
                event_rows.append(row)
                force_rows.append(row)
            alpha_vals = [r["alpha_effective_proxy"] for r in force_rows]
            fb_pred = kinetic_extrapolation(alpha_vals,
                         [r["mean_fb_if_complete"] for r in force_rows])
            exp_pred = kinetic_extrapolation(alpha_vals,
                         [r["mean_exp_if_complete"] for r in force_rows])
            gain = float(np.mean([r["gain_state_vs_time"] for r in force_rows]))
            efb = float(abs(np.log(fb_pred / exact))) if fb_pred is not None else None
            eexp = float(abs(np.log(exp_pred / exact))) if exp_pred is not None else None
            seed_rows.append(dict(height=height, seed=seed, exact_unbiased_mfpt=exact,
                                  switching_gain_state_vs_time=gain,
                                  reconstruction_status=("complete" if efb is not None
                                                         and eexp is not None else "censored_or_invalid"),
                                  predicted_unbiased_feedback=fb_pred,
                                  predicted_unbiased_exponential=exp_pred,
                                  error_abs_log_feedback=efb,
                                  error_abs_log_exponential=eexp,
                                  excess_feedback_error=(efb-eexp if efb is not None
                                                         and eexp is not None else None),
                                  total_fb_censored=sum(r["n_fb_censored"] for r in force_rows),
                                  total_exp_censored=sum(r["n_exp_censored"] for r in force_rows)))
        rs = [r for r in seed_rows if r["height"] == height]
        complete = [r for r in rs if r["reconstruction_status"] == "complete"]
        landscape_rows.append(dict(height=height, exact_unbiased_mfpt=exact,
                    n_seed_batches=len(rs), n_usable_reconstructions=len(complete),
                    mean_heldout_switch_gain=float(np.mean(
                        [r["switching_gain_state_vs_time"] for r in rs])),
                    feedback_reconstruction_logerror=(
                        float(np.mean([r["error_abs_log_feedback"] for r in complete]))
                        if complete else None),
                    exponential_reconstruction_logerror=(
                        float(np.mean([r["error_abs_log_exponential"] for r in complete]))
                        if complete else None),
                    excess_feedback_error=(
                        float(np.mean([r["excess_feedback_error"] for r in complete]))
                        if complete else None)))
        s = landscape_rows[-1]
        print(f"H={height:.2f}: exact={exact:.4f}, "
              f"gain={s['mean_heldout_switch_gain']:+.5f}, "
              f"complete={len(complete)}/{len(rs)}, "
              f"FB log-error={s['feedback_reconstruction_logerror']}, "
              f"EXP log-error={s['exponential_reconstruction_logerror']}")

    write_csv(output / "per_force_heldout.csv", event_rows)
    write_csv(output / "per_seed_reconstruction.csv", seed_rows)
    write_csv(output / "landscape_summary.csv", landscape_rows)
    corr_error = spearman(
        [r["mean_heldout_switch_gain"] for r in landscape_rows],
        [np.nan if r["feedback_reconstruction_logerror"] is None
         else r["feedback_reconstruction_logerror"] for r in landscape_rows])
    corr_excess = spearman(
        [r["mean_heldout_switch_gain"] for r in landscape_rows],
        [np.nan if r["excess_feedback_error"] is None
         else r["excess_feedback_error"] for r in landscape_rows])
    overview = dict(source_height=plan.source_height,
                    source_forces=plan.training_forces,
                    target_heights=plan.target_heights,
                    target_forces=plan.target_forces,
                    training_seeds=plan.training_seeds,
                    testing_seeds=plan.testing_seeds,
                    n_training_checkpoint_events=n_source_events,
                    classifier_refitted_on_targets=False,
                    source_model_fit_count=2,
                    heuristic_reconstruction="log MFPT_F linear in alpha_eff; extrapolated to alpha=1",
                    usable_landscapes=sum(r["n_usable_reconstructions"]>0 for r in landscape_rows),
                    spearman_switch_gain_vs_feedback_logerror=corr_error,
                    spearman_switch_gain_vs_excess_feedback_error=corr_excess,
                    association_status=("descriptive_only" if len(plan.target_heights)>=3
                                        else "insufficient_landscapes"))
    (output / "overview.json").write_text(json.dumps(overview, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.2))
    hs = np.asarray([r["height"] for r in landscape_rows])
    fb_e = [np.nan if r["feedback_reconstruction_logerror"] is None
            else r["feedback_reconstruction_logerror"] for r in landscape_rows]
    exp_e = [np.nan if r["exponential_reconstruction_logerror"] is None
             else r["exponential_reconstruction_logerror"] for r in landscape_rows]
    gains = np.asarray([r["mean_heldout_switch_gain"] for r in landscape_rows])
    axes[0].plot(hs, fb_e, "o-", label="Feedback")
    axes[0].plot(hs, exp_e, "s--", label="Exponential")
    axes[0].set(xlabel="Held-out barrier height H", ylabel="|log(predicted/exact)|",
                title="Heuristic kinetic reconstruction error")
    axes[0].legend()
    axes[1].plot(hs, gains, "o-", label="State - time gain")
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set(xlabel="Held-out barrier height H",
                ylabel="Time-only minus state log loss",
                title="Frozen model switching prediction")
    axes[2].scatter(gains, fb_e)
    for h, x, y in zip(hs, gains, fb_e):
        if np.isfinite(y):
            axes[2].annotate(f"H={h:g}", (x, y), fontsize=8, xytext=(3, 3),
                             textcoords="offset points")
    axes[2].set(xlabel="State vs time log-loss gain",
                ylabel="Feedback |log kinetic error|",
                title="Exploratory association (NOT prediction)")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "heldout_reliability.png", dpi=170)
    plt.close(fig)
    (output / "LIMITATIONS.txt").write_text(
        "Models were frozen after fit on H=2.5; only H and force values held out.\n"
        "Known test potential gradient is used as a feature, not fitted from training data.\n"
        "Switch labels are conditional on surviving until next checkpoint (selection).\n"
        "The kinetic extrapolation is an illustrative model inspired by effective-temperature ideas; NOT a validated Ding/Roux formula.\n"
        "Only complete uncensored force-ensemble means can be extrapolated; no escaped-only mean substitution.\n"
        "Effective alpha assumes free 2D isotropic overdamped behavior; feedback breaks its independence assumptions.\n"
        "Any cross-landscape association uses very few independent H values, without hypothesis-test significance.\n"
        "Same separable double-well family with different H is a limited OOD test, not generalization to independent potential topologies.\n"
        "No production atomistic ligand-residence-time validation or new reliable kinetic correction.\n",
        encoding="utf-8")
    return overview, event_rows, seed_rows, landscape_rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true", help="Tiny CI smoke, not physics")
    p.add_argument("--replicas", type=int, default=96)
    p.add_argument("--horizon", type=float, default=80.0)
    p.add_argument("--output", type=Path, default=Path("results_heldout"))
    args = p.parse_args()
    plan = Plan(replicas=args.replicas, horizon=args.horizon)
    if args.quick:
        plan = Plan(replicas=24, horizon=30.0, training_seeds=(2026, 2027),
                    testing_seeds=(2031, 2032),
                    target_heights=(1.7, 3.1))
    evaluate(plan, args.output)


if __name__ == "__main__":
    main()
