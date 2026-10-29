"""
Paper 4 implementation:
Leimkuhler & Matthews, "Rational Construction of Stochastic Numerical
Methods for Molecular Sampling".

Scientific question
-------------------
Does a time-discretized stochastic integrator reproduce the intended
configurational Gibbs-Boltzmann distribution in long-time sampling?

This script compares:
1) Euler-Maruyama for overdamped Langevin dynamics.
2) The modified correlated-noise scheme discussed in Eq. (3) of the paper.

It uses a 1D double-well potential, removes an initial relaxation period,
compares sampled histograms with the exact Gibbs density, and measures
sampling error as a function of timestep.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


KB = 1.0
T = 1.0
BETA = 1.0 / (KB * T)
MASS = 1.0


def potential(x):
    """Symmetric double-well potential U(x) = (x^2 - 1)^2."""
    return (x**2 - 1.0) ** 2


def grad_potential(x):
    """dU/dx = 4 x (x^2 - 1)."""
    return 4.0 * x * (x**2 - 1.0)


def exact_gibbs_density(grid):
    """rho(x) = Z^{-1} exp[-beta U(x)]."""
    unnormalized = np.exp(-BETA * potential(grid))
    Z = np.trapezoid(unnormalized, grid)
    return unnormalized / Z


def euler_maruyama(x0, timestep, n_steps, seed=1):
    """Euler-Maruyama discretization of overdamped Langevin dynamics."""
    rng = np.random.default_rng(seed)
    x = float(x0)
    trajectory = np.empty(n_steps)

    noise_scale = np.sqrt(2.0 * KB * T * timestep / MASS)

    for step in range(n_steps):
        deterministic_drift = -timestep * grad_potential(x) / MASS
        random_kick = noise_scale * rng.normal()
        x = x + deterministic_drift + random_kick
        trajectory[step] = x

    return trajectory


def modified_colored_noise(x0, timestep, n_steps, seed=1):
    """
    1D version of the modified scheme discussed in Eq. (3):

        x_(n+1) = x_n - h grad U(x_n)/M
                  + sqrt(kBT h/(2M)) (R_n + R_(n+1))

    Neighboring random perturbations are correlated because one Gaussian
    variate is shared between adjacent timesteps.
    """
    rng = np.random.default_rng(seed)
    x = float(x0)
    trajectory = np.empty(n_steps)

    noise_scale = np.sqrt(KB * T * timestep / (2.0 * MASS))
    r_current = rng.normal()

    for step in range(n_steps):
        r_next = rng.normal()

        deterministic_drift = -timestep * grad_potential(x) / MASS
        correlated_random_kick = noise_scale * (r_current + r_next)

        x = x + deterministic_drift + correlated_random_kick
        trajectory[step] = x

        r_current = r_next

    return trajectory


def discard_relaxation(trajectory, burn_fraction=0.20):
    """Remove the early portion that still remembers the initial condition."""
    start = int(burn_fraction * len(trajectory))
    return trajectory[start:]


def sampled_density(samples, grid):
    """Convert a long trajectory into a normalized histogram density."""
    edges = np.linspace(grid[0], grid[-1], len(grid) + 1)
    hist, _ = np.histogram(samples, bins=edges, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, hist


def interpolate_exact_density(centers, exact_grid, exact_density):
    return np.interp(centers, exact_grid, exact_density)


def l1_error(sampled, exact, x):
    """Integral |rho_sampled - rho_exact| dx."""
    return np.trapezoid(np.abs(sampled - exact), x)


def kl_divergence(sampled, exact, x):
    """D_KL(sampled || exact), with a small epsilon for empty bins."""
    eps = 1.0e-12
    p = sampled + eps
    q = exact + eps
    return np.trapezoid(p * np.log(p / q), x)


def observable_x2(samples):
    """Example configurational observable <x^2>."""
    return np.mean(samples**2)


def run_single_experiment(timestep, n_steps, seed, exact_grid, exact_density):
    em_trajectory = euler_maruyama(
        x0=0.0,
        timestep=timestep,
        n_steps=n_steps,
        seed=seed,
    )

    modified_trajectory = modified_colored_noise(
        x0=0.0,
        timestep=timestep,
        n_steps=n_steps,
        seed=seed,
    )

    em_samples = discard_relaxation(em_trajectory)
    modified_samples = discard_relaxation(modified_trajectory)

    em_x, em_density = sampled_density(em_samples, exact_grid)
    mod_x, mod_density = sampled_density(modified_samples, exact_grid)

    exact_at_em = interpolate_exact_density(em_x, exact_grid, exact_density)
    exact_at_mod = interpolate_exact_density(mod_x, exact_grid, exact_density)

    return {
        "timestep": timestep,
        "em_samples": em_samples,
        "modified_samples": modified_samples,
        "em_x": em_x,
        "em_density": em_density,
        "mod_x": mod_x,
        "mod_density": mod_density,
        "em_l1": l1_error(em_density, exact_at_em, em_x),
        "mod_l1": l1_error(mod_density, exact_at_mod, mod_x),
        "em_kl": kl_divergence(em_density, exact_at_em, em_x),
        "mod_kl": kl_divergence(mod_density, exact_at_mod, mod_x),
        "em_x2": observable_x2(em_samples),
        "mod_x2": observable_x2(modified_samples),
    }


def plot_distribution_comparison(result, exact_grid, exact_density, output_directory):
    timestep = result["timestep"]

    plt.figure(figsize=(9, 6))
    plt.plot(
        exact_grid,
        exact_density,
        linewidth=3,
        label="Exact Gibbs distribution",
    )
    plt.plot(
        result["em_x"],
        result["em_density"],
        label="Euler-Maruyama",
    )
    plt.plot(
        result["mod_x"],
        result["mod_density"],
        label="Modified correlated-noise",
    )

    plt.xlabel("Position x")
    plt.ylabel("Probability density")
    plt.title(f"Configurational sampling, h = {timestep}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_directory / f"distribution_h_{timestep}.png",
        dpi=200,
    )
    plt.close()


def timestep_study():
    output_directory = Path("results_paper4")
    output_directory.mkdir(exist_ok=True)

    exact_grid = np.linspace(-2.5, 2.5, 400)
    exact_density = exact_gibbs_density(exact_grid)

    timesteps = np.array([0.0025, 0.005, 0.010, 0.020, 0.040])
    n_steps = 1_000_000
    seed = 2026

    results = []

    for timestep in timesteps:
        print(f"\nRunning timestep h = {timestep}")

        result = run_single_experiment(
            timestep=timestep,
            n_steps=n_steps,
            seed=seed,
            exact_grid=exact_grid,
            exact_density=exact_density,
        )
        results.append(result)

        print("Euler-Maruyama")
        print(f"  L1 error       = {result['em_l1']:.6f}")
        print(f"  KL divergence  = {result['em_kl']:.6f}")
        print(f"  <x^2>          = {result['em_x2']:.6f}")

        print("Modified correlated-noise")
        print(f"  L1 error       = {result['mod_l1']:.6f}")
        print(f"  KL divergence  = {result['mod_kl']:.6f}")
        print(f"  <x^2>          = {result['mod_x2']:.6f}")

        plot_distribution_comparison(
            result,
            exact_grid,
            exact_density,
            output_directory,
        )

    em_errors = np.array([r["em_l1"] for r in results])
    modified_errors = np.array([r["mod_l1"] for r in results])

    plt.figure(figsize=(8, 6))
    plt.loglog(timesteps, em_errors, marker="o", label="Euler-Maruyama")
    plt.loglog(
        timesteps,
        modified_errors,
        marker="o",
        label="Modified correlated-noise",
    )
    plt.xlabel("Timestep h")
    plt.ylabel("L1 sampling error")
    plt.title("Long-time configurational sampling error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_directory / "sampling_error_vs_timestep.png",
        dpi=200,
    )
    plt.close()

    result_file = output_directory / "summary.csv"

    with open(result_file, "w", encoding="utf-8") as file:
        file.write(
            "timestep,em_l1,modified_l1,em_kl,modified_kl,em_x2,modified_x2\n"
        )

        for result in results:
            file.write(
                f"{result['timestep']},"
                f"{result['em_l1']},"
                f"{result['mod_l1']},"
                f"{result['em_kl']},"
                f"{result['mod_kl']},"
                f"{result['em_x2']},"
                f"{result['mod_x2']}\n"
            )

    print("\nFinished.")
    print(f"Results saved in: {output_directory}")


if __name__ == "__main__":
    timestep_study()
