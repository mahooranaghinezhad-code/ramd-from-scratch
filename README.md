# RAMD from Scratch — Literature-to-Code Portfolio

An exploratory computational-biophysics project connecting **statistical mechanics, molecular sampling, rare events, reaction coordinates, and accelerated ligand dissociation**.

This repository contains educational model implementations and a **testable RAMD research hypothesis**. It is **not** a validated method for recovering experimental residence times, nor a reproduction of full protein–ligand molecular dynamics.

## Scientific narrative

The guiding question is:

> When can biased or accelerated molecular simulations recover reliable equilibrium information or kinetic estimates, and when do the underlying assumptions break down?

We build the argument step-by-step:

| Step | Literature focus | Code / deliverable |
|---|---|---|
| 1 (Paper 4) | Gibbs–Boltzmann sampling and numerical integrators | [Equilibrium sampling](01_equilibrium_sampling/sampling_integrators.py) |
| 2 (Paper 2) | Umbrella sampling across rare regions | [Umbrella windows + PMF](02_umbrella_sampling/umbrella_sampling.py) |
| 3 (Paper 5) | Multistate reweighting and free energies | [MBAR](03_mbar/mbar_analysis.py) |
| 4 (Paper 3) | Committor and reaction-coordinate quality | [Committor shooting](04_reaction_coordinates/committor_analysis.py) |
| 5 (Paper 6) | Nonequilibrium work and free energies | [Jarzynski experiment](05_nonequilibrium/jarzynski_pulling.py) |
| 6 (Paper 10) | Random-acceleration ligand escape protocol | [tauRAMD-style C++ toy](06_tau_ramd/tau_ramd.cpp) |
| 7 (Paper 1) | Effective diffusion / temperature of RAMD | [RAMD theory experiment](07_ramd_theory/ramd_effective_dynamics.py) |
| 8 (Paper 7) | Molecular trajectory observables | [MDAnalysis workflow](08_trajectory_analysis/analyze_mdanalysis.py) |
| 9 (Paper 8) | Custom forces using a molecular simulation engine | [OpenMM demonstration](09_openmm_ramd/openmm_ramd.py) |
| 10 (Paper 9) | Differentiable potential integration | [PyTorch / OpenMM-Torch demo](10_ml_extension/torchforce_demo.py) |

The original C++ learning steps are retained in the repository.

## Independent research direction

**[Read the research hypothesis](docs/research_gap.md)**

*Can reaction-coordinate quality and dynamical memory predict failure of RAMD kinetic unbiasing?*

[Exploratory benchmark source](research/memory_rc_validity_benchmark.py).

The aim is to compare reference **unbiased mean-first-passage times** with accelerated estimates across controlled landscapes and levels of hidden-coordinate relaxation. The hypothesis is **not yet established**. A thorough benchmark requires simulation convergence, censored-trajectory handling, uncertainty estimation and independent validation systems.

## Reproducibility

Requirements vary by module:
- NumPy and Matplotlib for numerical demonstrations.
- A C++17 compiler for the tauRAMD-style model.
- MDAnalysis plus a real topology and trajectory for molecular analysis.
- OpenMM for the custom-force example.
- PyTorch, and optional OpenMM-Torch, for the potential-integration example.

Run from the repository root, for example:

```bash
pip install numpy matplotlib
python 02_umbrella_sampling/umbrella_sampling.py
python 03_mbar/mbar_analysis.py --max-samples-per-window 500
python research/memory_rc_validity_benchmark.py --quick
```

**Scientific testing and continuous integration:** [validation workflow](.github/workflows/validate.yml). A workflow's presence does not imply that every scientific result has passed validation; check GitHub Actions results and the scripts' assumptions.

## Limitations worth noting

- Umbrella/MBAR estimate equilibrium quantities; they do **not** directly recover unbiased RAMD kinetics.
- The simplified tauRAMD logic and exponentially persistent force model correspond to **different protocols**.
- A threshold crossing does not necessarily prove irreversible ligand dissociation.
- Some modules require environment-specific dependencies and real trajectory inputs.
- A plausible research hypothesis is not a verified contribution until it survives controlled numerical and physical validation.

## Primary inspiration

Ding, Ghidini, Cossio & Roux (2026), *A theoretical framework for random accelerated molecular dynamics simulations* (J. Chem. Phys.), together with the sampling, free-energy and reaction-coordinate papers listed in the relevant scripts.
