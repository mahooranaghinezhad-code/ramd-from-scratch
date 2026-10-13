"""
Paper 2 implementation:
Torrie & Valleau, "Nonphysical Sampling Distributions in Monte Carlo
Free-Energy Estimation: Umbrella Sampling" (1977).

Scientific question
-------------------
What can we do when ordinary equilibrium sampling rarely visits important
high-free-energy regions?

This script demonstrates the umbrella-sampling idea with a 1D double-well:
1) Run conventional Metropolis Monte Carlo.
2) Run several overlapping biased simulations using harmonic umbrellas.
3) Save every window's raw samples for later multistate analysis.
4) Reconstruct an unbiased PMF.

Important literature note
-------------------------
The umbrella-sampling concept and reweighting motivation come from
Torrie & Valleau.  The multi-window reconstruction below uses a compact
WHAM-style self-consistent estimator as a practical modern analysis layer.
WHAM itself is not claimed to be part of the 1977 paper.  In the next
repository module, these same window samples can be analyzed with MBAR
(Paper 5).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


KB = 1.0
T = 1.0
BETA = 1.0 / (KB * T)

# A higher barrier than in the Paper 4 toy model makes the rare-region
# problem visually obvious.
BARRIER_SCALE = 5.0

# Umbrella settings.
WINDOW_CENTERS = np.array(
    [-1.50, -1.00, -0.50, 0.00, 0.50, 1.00, 1.50]
)
K_BIAS = 20.0

# Monte Carlo settings.
N_STEPS = 250_000
BURN_IN = 25_000
STRIDE = 10
PROPOSAL_SIGMA = 0.22
SEED = 2027

# PMF grid.
X_MIN = -2.2
X_MAX = 2.2
N_BINS = 180


def potential(x):
    """
    Physical double-well potential:

        U(x) = a (x^2 - 1)^2

    where a = BARRIER_SCALE.

    The minima are near x = +/-1 and the barrier is at x = 0.
    """
    return BARRIER_SCALE * (x**2 - 1.0) ** 2


def umbrella_bias(x, center, k_bias=K_BIAS):
    """
    Harmonic umbrella potential:

        W_i(x) = 1/2 k (x - center_i)^2
    """
    return 0.5 * k_bias * (x - center) ** 2


def total_biased_energy(x, center):
    """Physical potential plus one umbrella restraint."""
    return potential(x) + umbrella_bias(x, center)


def metropolis_trajectory(
    energy_function,
    x0,
    n_steps,
    proposal_sigma,
    rng,
):
    """
    Generic 1D Metropolis Monte Carlo sampler.

    Proposal:
        x' = x + Normal(0, proposal_sigma)

    Acceptance:
        min(1, exp[-beta (E_new - E_old)])
    """
    x = float(x0)
    energy = float(energy_function(x))

    trajectory = np.empty(n_steps)
    accepted = 0

    for step in range(n_steps):
        trial = x + rng.normal(0.0, proposal_sigma)
        trial_energy = float(energy_function(trial))

        delta_energy = trial_energy - energy

        if delta_energy <= 0.0:
            accept = True
        else:
            probability = np.exp(-BETA * delta_energy)
            accept = rng.random() < probability

        if accept:
            x = trial
            energy = trial_energy
            accepted += 1

        trajectory[step] = x

    acceptance_ratio = accepted / n_steps

    return trajectory, acceptance_ratio


def production_samples(trajectory):
    """
    Remove burn-in and subsample the trajectory.

    The stride here is used only to reduce storage and short-range correlation.
    It is not a full statistical-inefficiency analysis.
    """
    return trajectory[BURN_IN::STRIDE]


def run_conventional_sampling(rng):
    """Unbiased Monte Carlo trajectory under U(x) only."""
    trajectory, acceptance = metropolis_trajectory(
        energy_function=potential,
        x0=-1.0,
        n_steps=N_STEPS,
        proposal_sigma=PROPOSAL_SIGMA,
        rng=rng,
    )

    return production_samples(trajectory), acceptance


def run_umbrella_windows():
    """
    Run one independent biased simulation for each umbrella center.

    A different random-number stream is used for every window.
    """
    windows = []

    for i, center in enumerate(WINDOW_CENTERS):
        rng = np.random.default_rng(SEED + 100 * i)

        energy_function = lambda x, c=center: total_biased_energy(x, c)

        trajectory, acceptance = metropolis_trajectory(
            energy_function=energy_function,
            x0=center,
            n_steps=N_STEPS,
            proposal_sigma=PROPOSAL_SIGMA,
            rng=rng,
        )

        samples = production_samples(trajectory)

        windows.append(
            {
                "index": i,
                "center": center,
                "k_bias": K_BIAS,
                "samples": samples,
                "acceptance": acceptance,
            }
        )

    return windows


def exact_equilibrium_density(grid):
    """Exact 1D Gibbs density for comparison."""
    unnormalized = np.exp(-BETA * potential(grid))
    normalization = np.trapz(unnormalized, grid)
    return unnormalized / normalization


def histogram_counts(samples, edges):
    counts, _ = np.histogram(samples, bins=edges)
    return counts.astype(float)


def build_window_histograms(windows, edges):
    """Return an array counts[i, j] for umbrella window i and bin j."""
    counts = []

    for window in windows:
        counts.append(
            histogram_counts(
                window["samples"],
                edges,
            )
        )

    return np.asarray(counts)


def bias_matrix(centers):
    """
    bias_ij = W_i(x_j)

    Rows correspond to umbrella windows.
    Columns correspond to histogram-bin centers.
    """
    matrix = np.empty((len(WINDOW_CENTERS), len(centers)))

    for i, center in enumerate(WINDOW_CENTERS):
        matrix[i, :] = umbrella_bias(
            centers,
            center,
        )

    return matrix


def wham_reconstruct(
    counts,
    bin_centers,
    tolerance=1.0e-10,
    max_iterations=100_000,
):
    """
    Histogram-based self-consistent reconstruction of the unbiased
    probability mass P_j.

    For harmonic biases W_i(x), a compact WHAM-style iteration is

        P_j =
        [sum_i n_ij]
        /
        [sum_i N_i exp(beta (f_i - W_i(x_j)))]

    followed by

        exp(-beta f_i)
        =
        sum_j P_j exp(-beta W_i(x_j))

    P_j is normalized to sum to 1 each iteration.

    The free-energy offsets f_i are defined only up to an additive constant.
    """
    n_windows, n_bins = counts.shape

    if n_bins != len(bin_centers):
        raise ValueError("Histogram and bin-center sizes do not match.")

    total_per_window = counts.sum(axis=1)
    total_per_bin = counts.sum(axis=0)

    W = bias_matrix(bin_centers)

    # Start with equal free-energy offsets.
    f = np.zeros(n_windows)

    # Avoid division by exactly zero in bins never visited by any window.
    visited = total_per_bin > 0.0

    for iteration in range(max_iterations):
        denominator = np.zeros(n_bins)

        for i in range(n_windows):
            denominator += total_per_window[i] * np.exp(
                BETA * (f[i] - W[i, :])
            )

        probability_mass = np.zeros(n_bins)

        valid = visited & (denominator > 0.0)

        probability_mass[valid] = (
            total_per_bin[valid]
            / denominator[valid]
        )

        normalization = probability_mass.sum()

        if normalization == 0.0:
            raise RuntimeError(
                "WHAM reconstruction failed: no probability mass was recovered."
            )

        probability_mass /= normalization

        new_f = np.empty_like(f)

        for i in range(n_windows):
            z_i = np.sum(
                probability_mass
                * np.exp(-BETA * W[i, :])
            )

            if z_i <= 0.0:
                raise RuntimeError(
                    f"WHAM reconstruction failed in window {i}."
                )

            new_f[i] = -np.log(z_i) / BETA

        # Remove the arbitrary additive constant.
        new_f -= new_f[0]

        change = np.max(np.abs(new_f - f))
        f = new_f

        if change < tolerance:
            print(
                f"WHAM converged after {iteration + 1} iterations "
                f"(max delta f = {change:.3e})."
            )
            break
    else:
        print(
            "Warning: WHAM reached max_iterations before the requested tolerance."
        )

    return probability_mass, f


def probability_mass_to_pmf(probability_mass):
    """
    Convert discrete probability mass to a PMF up to an additive constant:

        G(x) = -kBT ln P(x) + C
    """
    pmf = np.full_like(probability_mass, np.nan, dtype=float)

    positive = probability_mass > 0.0

    pmf[positive] = (
        -KB * T * np.log(probability_mass[positive])
    )

    pmf[positive] -= np.nanmin(pmf[positive])

    return pmf


def exact_pmf(bin_centers):
    """
    For this 1D coordinate with a flat measure, the exact PMF differs from U(x)
    only by an additive constant.
    """
    values = potential(bin_centers)
    return values - np.min(values)


def overlap_matrix(counts):
    """
    Simple histogram-overlap diagnostic.

    Each row is normalized into a discrete distribution and the overlap between
    windows i and j is sum_k min(p_i(k), p_j(k)).
    """
    row_sums = counts.sum(axis=1, keepdims=True)

    probabilities = np.divide(
        counts,
        row_sums,
        out=np.zeros_like(counts),
        where=row_sums > 0.0,
    )

    n = counts.shape[0]
    overlap = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            overlap[i, j] = np.minimum(
                probabilities[i],
                probabilities[j],
            ).sum()

    return overlap


def save_raw_window_data(windows, output_directory):
    """
    Save raw samples so Paper 5 can reuse the same data for MBAR.
    """
    metadata_path = output_directory / "umbrella_metadata.csv"

    with open(metadata_path, "w", encoding="utf-8") as file:
        file.write(
            "window_index,center,k_bias,n_samples,acceptance_ratio\n"
        )

        for window in windows:
            file.write(
                f"{window['index']},"
                f"{window['center']},"
                f"{window['k_bias']},"
                f"{len(window['samples'])},"
                f"{window['acceptance']}\n"
            )

            sample_path = output_directory / (
                f"window_{window['index']:02d}_samples.csv"
            )

            np.savetxt(
                sample_path,
                window["samples"],
                delimiter=",",
                header="x",
                comments="",
            )


def plot_sampling_problem(
    conventional_samples,
    windows,
    output_directory,
):
    """Show why ordinary sampling struggles and how umbrellas cover the barrier."""
    x_plot = np.linspace(X_MIN, X_MAX, 1000)

    plt.figure(figsize=(10, 6))

    plt.hist(
        conventional_samples,
        bins=120,
        range=(X_MIN, X_MAX),
        density=True,
        alpha=0.45,
        label="Conventional sampling",
    )

    exact_density = exact_equilibrium_density(x_plot)

    plt.plot(
        x_plot,
        exact_density,
        linewidth=2.5,
        label="Exact equilibrium density",
    )

    plt.xlabel("Reaction coordinate x")
    plt.ylabel("Probability density")
    plt.title("Rare-region problem in conventional sampling")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_directory / "conventional_sampling.png",
        dpi=220,
    )
    plt.close()

    plt.figure(figsize=(10, 6))

    for window in windows:
        plt.hist(
            window["samples"],
            bins=100,
            range=(X_MIN, X_MAX),
            density=True,
            histtype="step",
            linewidth=1.5,
            label=f"center={window['center']:.2f}",
        )

    plt.xlabel("Reaction coordinate x")
    plt.ylabel("Biased probability density")
    plt.title("Overlapping umbrella windows")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(
        output_directory / "umbrella_windows.png",
        dpi=220,
    )
    plt.close()


def plot_pmf(
    bin_centers,
    reconstructed_pmf,
    output_directory,
):
    exact = exact_pmf(bin_centers)

    plt.figure(figsize=(9, 6))

    plt.plot(
        bin_centers,
        exact,
        linewidth=3,
        label="Exact PMF",
    )

    plt.plot(
        bin_centers,
        reconstructed_pmf,
        linewidth=2,
        label="Umbrella reconstruction",
    )

    plt.xlabel("Reaction coordinate x")
    plt.ylabel("Free energy / kBT")
    plt.title("Unbiased PMF reconstructed from umbrella windows")
    plt.ylim(0.0, min(10.0, np.nanmax(exact) + 0.5))
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_directory / "pmf_reconstruction.png",
        dpi=220,
    )
    plt.close()


def plot_overlap(overlap, output_directory):
    plt.figure(figsize=(7, 6))

    image = plt.imshow(
        overlap,
        origin="lower",
        vmin=0.0,
        vmax=1.0,
        aspect="equal",
    )

    plt.colorbar(
        image,
        label="Histogram overlap",
    )

    plt.xticks(
        range(len(WINDOW_CENTERS)),
        [f"{x:.2f}" for x in WINDOW_CENTERS],
        rotation=45,
    )
    plt.yticks(
        range(len(WINDOW_CENTERS)),
        [f"{x:.2f}" for x in WINDOW_CENTERS],
    )

    plt.xlabel("Umbrella center")
    plt.ylabel("Umbrella center")
    plt.title("Window-overlap diagnostic")
    plt.tight_layout()
    plt.savefig(
        output_directory / "window_overlap.png",
        dpi=220,
    )
    plt.close()


def save_pmf_csv(
    bin_centers,
    probability_mass,
    reconstructed_pmf,
    output_directory,
):
    exact = exact_pmf(bin_centers)

    data = np.column_stack(
        [
            bin_centers,
            probability_mass,
            reconstructed_pmf,
            exact,
        ]
    )

    np.savetxt(
        output_directory / "pmf.csv",
        data,
        delimiter=",",
        header="x,probability_mass,reconstructed_pmf,exact_pmf",
        comments="",
    )


def main():
    output_directory = Path("results_paper2")
    output_directory.mkdir(exist_ok=True)

    print("Running conventional sampling...")
    conventional_rng = np.random.default_rng(SEED)

    conventional_samples, conventional_acceptance = (
        run_conventional_sampling(conventional_rng)
    )

    print(
        f"Conventional acceptance ratio: "
        f"{conventional_acceptance:.3f}"
    )

    print("\nRunning umbrella windows...")
    windows = run_umbrella_windows()

    for window in windows:
        print(
            f"window {window['index']:02d} | "
            f"center={window['center']:+.2f} | "
            f"acceptance={window['acceptance']:.3f} | "
            f"samples={len(window['samples'])}"
        )

    save_raw_window_data(
        windows,
        output_directory,
    )

    edges = np.linspace(
        X_MIN,
        X_MAX,
        N_BINS + 1,
    )

    bin_centers = 0.5 * (
        edges[:-1] + edges[1:]
    )

    counts = build_window_histograms(
        windows,
        edges,
    )

    probability_mass, window_free_energies = wham_reconstruct(
        counts,
        bin_centers,
    )

    reconstructed_pmf = probability_mass_to_pmf(
        probability_mass
    )

    overlap = overlap_matrix(counts)

    plot_sampling_problem(
        conventional_samples,
        windows,
        output_directory,
    )

    plot_pmf(
        bin_centers,
        reconstructed_pmf,
        output_directory,
    )

    plot_overlap(
        overlap,
        output_directory,
    )

    save_pmf_csv(
        bin_centers,
        probability_mass,
        reconstructed_pmf,
        output_directory,
    )

    np.savetxt(
        output_directory / "window_free_energy_offsets.csv",
        np.column_stack(
            [WINDOW_CENTERS, window_free_energies]
        ),
        delimiter=",",
        header="window_center,free_energy_offset",
        comments="",
    )

    print("\nScientific outputs written to:")
    print(output_directory)

    print("\nNext module:")
    print(
        "Reuse the saved umbrella-window samples in Paper 5 "
        "to replace histogram reconstruction with MBAR."
    )


if __name__ == "__main__":
    main()
