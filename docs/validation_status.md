# Validation status and known limitations

**Last reviewed:** 2026-10-28

**Scope:** numerical checks of the educational toy models. This is **not**
a claim of experimentally validated protein–ligand residence times or a new
reliable RAMD kinetic correction.

## Prior CI failure and fix

- [October 13 scientific-validation run](https://github.com/mahooranaghinezhad-code/ramd-from-scratch/actions/runs/38350850919): the `cpp` and `openmm` jobs passed, and `python-smoke` failed in the Paper 2 → Paper 5 chain.
- The retrieved Python job log reported
  `AttributeError: module 'numpy' has no attribute 'trapz'`
  from `02_umbrella_sampling/umbrella_sampling.py`, during exact equilibrium-density plotting.
- Replaced five uses of `np.trapz` with the supported `np.trapezoid` across Paper 2, Paper 4 and Paper 1 modules.
- Extended CI with `tests/scientific_checks.py`, including analytic PMF and probability-normalization checks. Also added a bounded `--quick` mode for the RAMD theory module.
- **Do not infer a new GitHub Actions pass until the updated workflow has actually completed with green status.** See the [live CI badge](../README.md) and [runs](../.github/workflows/validate.yml).

## Local independent numerical check

In a local run of the **equivalent mathematical kernels** (not a complete end-to-end execution of all repository scripts), with NumPy 2.3.5 and the same double-well and umbrella parameters:

| Metric | Result |
|---|---|
| Number of umbrella windows | 7 |
| Stored Monte Carlo samples per window | 22,500 |
| WHAM iterations to convergence | 404 |
| WHAM probability normalization | 1.0 (within numerical precision) |
| WHAM central-region PMF RMSE, |x| <= 1.3 | approx. 0.034 kBT |
| MBAR subsample | 500 samples per window |
| MBAR self-consistency iterations | approx. 452 |
| MBAR target-weight normalization | 1.0 (within numerical precision) |
| MBAR target weight effective sample size | approx. 2,873 |
| MBAR central-region PMF RMSE, |x| <= 1.3 | approx. 0.379 kBT |

These are **dimensionless educational simulations** against a known exact
toy potential. Edge bins are sparsely sampled and have much larger errors.
Finite sampling and correlation affect precision. The new automated checks
are deliberately *sanity tests*, not publication-grade convergence studies.

## What is and is not independently verified

- **C++ tauRAMD toy**: compiled and smoke-executed in the previous CI run.
  Synthetic effective-particle model; no atomistic protein, solvent or
  calibrated dissociation time.
- **OpenMM toy**: smoke-executed successfully in the previous CI run.
  This is not a biochemical benchmark and does not validate production RAMD.
- **Paper 4 numerics**: Python syntax and small integrator run passed in
  prior CI; a thorough timestep-order/convergence study is still needed.
- **Paper 2 and Paper 5**: independently checked against known toy PMF
  as detailed above; updated end-to-end CI pending explicit confirmation.
- **Paper 3 and Paper 6**: smoke checks are in CI; quantitative replication
  of published findings has not been established.
- **Paper 1**: effective-model demonstration, not a quantitative validation
  of the full paper or experimental residence times.
- **MDAnalysis pipeline**: needs actual user-supplied trajectory/topology,
  checked PBC conventions, atom selections and mass assignments for data testing.
- **OpenMM-Torch**: PyTorch model demo exists; optional OpenMM-Torch
  integration is not established by the default GitHub workflow.
- **RC/memory hypothesis benchmark**: hypothesis-generating only. Multiple
  independent runs, time-step checks, censoring-aware MFPT estimates,
  error bars and held-out potentials are needed before conclusions.

## Validation philosophy

1. Confirm the script actually runs in the stated environment.
2. Confirm conservation/normalization and exact-analytic toy controls.
3. Check sensitivity to timestep, sample count, random seed and force.
4. Estimate uncertainty and account for correlated/censored trajectories.
5. Only then report a comparison to literature or an experimental value.

Do not present a green CI badge as proof that a kinetic model is physically
correct. It certifies only the explicit tests that were run.
