"""Numerical regression checks for the linked Paper 2 → Paper 5 pipeline.

These are limited toy-model sanity checks, NOT validation of experimental
ligand residence times or the proposed RAMD-unbiasing research hypothesis.
Run after the Paper 2 and Paper 5 scripts from the repository root.
"""
from pathlib import Path
import numpy as np


def pmf_error(path: Path, x_cutoff: float):
    values = np.genfromtxt(path, delimiter=",", names=True)
    x = values["x"]
    if path.name == "pmf.csv":
        pmf = values["reconstructed_pmf"]
    else:
        pmf = values["mbar_pmf"]
    exact = values["exact_pmf"]
    probs = values["probability_mass"] if path.name == "pmf.csv" else values["unbiased_probability_mass"]
    assert len(x) > 100, f"Insufficient PMF bins in {path}"
    assert np.isfinite(probs).all() and np.all(probs >= 0), path
    assert abs(float(np.sum(probs)) - 1.0) < 1e-8, path
    selected = (np.abs(x) <= x_cutoff) & np.isfinite(pmf) & (probs > 0)
    assert selected.sum() >= 80, f"Insufficient coverage in core region: {path}"
    rmse = float(np.sqrt(np.mean((pmf[selected] - exact[selected]) ** 2)))
    print(f"{path}: central PMF RMSE={rmse:.4f} (dimensionless kBT units); {selected.sum()} bins")
    return rmse


def main():
    p2 = Path("results_paper2/pmf.csv")
    p5 = Path("results_paper5/mbar_pmf.csv")
    assert p2.exists() and p5.exists(), "Run Paper 2 and Paper 5 first"
    rmse_wham = pmf_error(p2, 1.3)
    rmse_mbar = pmf_error(p5, 1.3)
    # These generous limits are sanity thresholds for the deterministic toy
    # example and fixed random seed, NOT claims of generalized accuracy.
    assert rmse_wham < 0.25, f"WHAM accuracy regression: {rmse_wham:.3f}"
    assert rmse_mbar < 0.85, f"MBAR accuracy regression: {rmse_mbar:.3f}"
    window_meta = np.genfromtxt("results_paper2/umbrella_metadata.csv",
                                delimiter=",", names=True)
    assert len(window_meta) == 7
    assert np.all(window_meta["n_samples"] >= 100), "Missing umbrella samples"
    for required in ("results_paper2/window_overlap.png",
                     "results_paper5/mbar_overlap.png"):
        assert Path(required).exists(), required
    print("PASS: WHAM/MBAR probability normalization, core PMF accuracy and window data.")


if __name__ == "__main__":
    main()
