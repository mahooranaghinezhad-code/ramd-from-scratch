"""Paper 3: Reaction coordinates and rates from transition paths (Best & Hummer).

Educational 2D overdamped-Langevin experiment, not a reproduction of the
systems or the formal variational optimization used by Best & Hummer.

Estimate the committor p_B(x,y): fraction of independent stochastic shooting
trajectories that reach product B before reactant A. Compare two candidate
reaction coordinates, x and y, against that committor.
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def potential(x, y, barrier=4.0, transverse=2.0):
    return barrier * (x*x - 1.0)**2 + 0.5 * transverse * (y - 0.4*x)**2


def force(x, y, barrier=4.0, transverse=2.0):
    """Return minus the gradient of the two-dimensional potential."""
    fx = -4.0*barrier*x*(x*x-1.0) + 0.4*transverse*(y-0.4*x)
    fy = -transverse*(y-0.4*x)
    return fx, fy


def shoot_committor(points, n_shots=100, dt=0.003, max_steps=14000,
                    beta=1.0, mobility=1.0, seed=2026, boundary=0.75):
    """Estimate p_B with independent overdamped Langevin shooting trajectories.

    A: x <= -boundary. B: x >= boundary. Unresolved shots are excluded,
    and the resolved fraction is reported separately.
    """
    rng = np.random.default_rng(seed)
    p_b = np.full(len(points), np.nan)
    resolved_fraction = np.zeros(len(points))
    sigma = np.sqrt(2.0*mobility*dt/beta)
    for i, (x0, y0) in enumerate(points):
        x = np.full(n_shots, x0, dtype=float)
        y = np.full(n_shots, y0, dtype=float)
        active = np.ones(n_shots, dtype=bool)
        outcomes = np.full(n_shots, -1, dtype=np.int8)
        for _ in range(max_steps):
            ids = np.flatnonzero(active)
            if not len(ids):
                break
            fx, fy = force(x[ids], y[ids])
            x[ids] += mobility*fx*dt + sigma*rng.standard_normal(len(ids))
            y[ids] += mobility*fy*dt + sigma*rng.standard_normal(len(ids))
            a = x[ids] <= -boundary
            b = x[ids] >= boundary
            outcomes[ids[a]] = 0
            outcomes[ids[b]] = 1
            active[ids[a | b]] = False
        resolved = outcomes >= 0
        resolved_fraction[i] = resolved.mean()
        if resolved.any():
            p_b[i] = outcomes[resolved].mean()
    return p_b, resolved_fraction


def coordinate_quality(values, p_b, n_bins=12):
    """Within-bin variance of committor values: lower is more predictive.

    Pedagogical diagnostic only, not the paper's variational RC estimator.
    """
    edges = np.linspace(values.min()-1e-8, values.max()+1e-8, n_bins+1)
    digitized = np.digitize(values, edges) - 1
    weighted_var = 0.0
    total = 0
    for j in range(n_bins):
        samples = p_b[(digitized == j) & np.isfinite(p_b)]
        if len(samples) >= 2:
            weighted_var += len(samples) * np.var(samples)
            total += len(samples)
    return weighted_var / total if total else np.nan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=14000)
    parser.add_argument("--dt", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("results_paper3"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    # x separates A from B; y is mainly a transverse fluctuation.
    xs = np.linspace(-0.65, 0.65, 13)
    ys = np.linspace(-1.4, 1.4, 9)
    points = np.array([(x, y) for x in xs for y in ys], dtype=float)
    p_b, resolved = shoot_committor(
        points, n_shots=args.shots, max_steps=args.max_steps,
        dt=args.dt, seed=args.seed,
    )

    np.savetxt(
        args.output / "committor.csv",
        np.column_stack((points, p_b, resolved)),
        delimiter=",", header="x,y,p_B,resolved_fraction", comments="",
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, coordinate, label in zip(
        axes, (points[:, 0], points[:, 1]),
        ("x: candidate good RC", "y: candidate poor RC")
    ):
        sc = ax.scatter(coordinate, p_b, c=points[:, 0], s=24, alpha=0.8)
        ax.set_xlabel(label)
        ax.set_ylabel("Estimated forward committor p_B")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.2)
    fig.colorbar(sc, ax=axes, label="starting x", shrink=0.8)
    fig.suptitle("Same configurations, different RC projections")
    fig.savefig(
        args.output / "coordinate_vs_committor.png",
        dpi=180, bbox_inches="tight"
    )
    plt.close(fig)

    xline = np.linspace(-1.5, 1.5, 250)
    yline = np.linspace(-1.8, 1.8, 250)
    X, Y = np.meshgrid(xline, yline)
    plt.figure(figsize=(7, 5.3))
    plt.contourf(
        X, Y, potential(X, Y),
        levels=np.linspace(0, 8, 28), extend="max"
    )
    plt.colorbar(label="Potential energy U / (kBT)")
    plt.scatter(
        points[:, 0], points[:, 1], c=p_b, cmap="coolwarm",
        vmin=0, vmax=1, s=20, edgecolor="k", linewidth=0.15
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("2D potential and sampled committor")
    plt.savefig(
        args.output / "landscape_and_committor.png",
        dpi=180, bbox_inches="tight"
    )
    plt.close()

    x_quality = coordinate_quality(points[:, 0], p_b)
    y_quality = coordinate_quality(points[:, 1], p_b)
    with open(args.output / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Number of configurations: {len(points)}\n")
        f.write(f"Shots per configuration: {args.shots}\n")
        f.write(f"Mean resolved fraction: {resolved.mean():.3f}\n")
        f.write(f"Within-bin committor variance (x): {x_quality:.6f}\n")
        f.write(f"Within-bin committor variance (y): {y_quality:.6f}\n")
        f.write("Lower within-bin variance indicates a more predictive projection.\n")
        f.write("This does not establish kinetics or a formally optimal RC.\n")
    print((args.output / "summary.txt").read_text())


if __name__ == "__main__":
    main()
