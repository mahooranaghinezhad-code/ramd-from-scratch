# Research hypothesis: When can RAMD unbiasing be trusted?

**Status:** research proposal / untested hypothesis — not a demonstrated discovery.

## One-sentence pitch

Build a quantitative **validity map for kinetic unbiasing of accelerated ligand escape**, conditioned on the quality of the chosen reaction coordinate and on the memory timescale of the projected dynamics.

## Motivation from the literature

- **Paper 4 — Leimkuhler & Matthews:** numerical integrators can change long-run configurational sampling, so discretization bias needs a control before extracting physics.
- **Paper 2 — Torrie & Valleau:** rare regions can be sampled deliberately with a nonphysical distribution.
- **Paper 5 — Shirts & Chodera:** multiple biased **equilibrium** ensembles can be reweighted to estimate equilibrium free energies and observables, provided overlap and correlation are addressed. This is **not automatically a kinetic unbiasing method**.
- **Paper 3 — Best & Hummer:** projecting onto a poor reaction coordinate can yield strongly non-Markovian dynamics; good coordinates better describe transition outcomes via splitting/committor probabilities.
- **Paper 6 — Jarzynski:** certain equilibrium free-energy differences can be reconstructed from specified nonequilibrium work ensembles. This equality is **not a direct formula for RAMD residence times**.
- **Paper 10 — Kokh et al.:** displacement-triggered random-force directions accelerate ligand dissociation in tauRAMD.
- **Paper 1 — Ding, Ghidini, Cossio & Roux (2026):** approximate RAMD as overdamped diffusion driven by a fixed-amplitude random force with exponentially distributed persistence. Their analysis predicts an effective diffusion coefficient and effective temperature; the article flags sensitivity to memory and relaxation times, and provides corrections/limitations when the persistence time is not short.
- **Papers 7, 8, 9:** MDAnalysis, OpenMM 7, and OpenMM 8 provide tooling for trajectory analysis, custom dynamics, and possible future ML-potential extensions.

**Crucial novelty boundary:** Paper 1 **already** defines eta = tau / tau_harm and discusses its validity limitations. Simply rediscovering this ratio is **not novel**. Our proposed new contribution is to jointly test **reaction-coordinate quality and dynamical memory as predictors of *kinetic reconstruction error*** across multiple landscapes and realistic protocols.

## Main research question

When the observed reaction coordinate omits slow degrees of freedom, does a RAMD effective-diffusion/temperature-based extrapolation produce systematically incorrect *unbiased mean first-passage times* even after accurate force-dependent measurements?

## Falsifiable hypotheses

H1. At matched force magnitude and persistence time, projected reaction coordinates with poorer committor predictivity produce larger errors in reconstructed unbiased escape times.

H2. The error is not predicted by eta = tau/tau_harm alone: hidden-coordinate memory, measured from conditional history dependence or a projected memory/correlation diagnostic, adds predictive power.

H3. A predeclared validity classifier trained on dimensionless physical quantities generalizes to held-out potential landscapes better than a baseline classifier using only force magnitude and eta.

**Null/failure outcome:** If adding committor quality and memory makes no held-out improvement, the proposed framework is not supported. This negative result would itself constrain where the simpler theory is sufficient.

## Proposed experimental design (not completed)

### Stage A — trusted ground truth

Use 2D and 3D analytical toy potentials with a controllable hidden slow coordinate:
- Double-well escape coordinate x with a tunable transverse coordinate y.
- Tune coupling, hidden-coordinate relaxation rate, barrier height, thermal temperature, and friction.
- Obtain **unbiased** reference mean-first-passage time (MFPT), its confidence interval, and committor estimates from independent unaccelerated trajectories.
- Verify integration convergence with timestep and enough independent transitions.

### Stage B — controlled acceleration

For the same potential, run independent accelerated ensembles over a grid:
- RAMD force magnitude F.
- Force-direction persistence tau with **exponential** update times, as in the approximation studied in Paper 1.
- Friction gamma and landscape curvature.
- Reaction coordinates: true x; distance-like projection; intentionally poor combination of x and y.

Keep separate experiments for **displacement-triggered tauRAMD** (Paper 10) versus **exponential-persistence model** (Paper 1). They are not the same stochastic process.

### Stage C — compare reconstructions

1. Apply the paper's effective-dynamics assumptions and extrapolation, where applicable.
2. Quantify error against ground-truth unbiased MFPT:
   E = abs(log(tau_predicted/tau_reference)).
3. Compare models with and without committor-quality and memory features.
4. Use held-out systems for validation; report confidence intervals and censoring.
5. Explicitly report parameter regions where the method **cannot** produce a reliable estimate.

### Stage D — scientific deliverables

- Plots: D_eff versus F²; effective temperature versus F²; logarithmic escape time versus alpha = T/T_eff.
- Committor-calibration and history-dependence plots for each candidate RC.
- Validity map: error E over bias strength, eta, and RC-quality strata.
- Reproducible scripts, test fixtures, fixed random seeds, and robust uncertainty analysis.
- One-to-two-page research note explaining what fails, why, and what is new relative to Paper 1.

## Candidate explanatory variables

- eta = tau / tau_harm (**already in Paper 1; not new**).
- chi = F * ell / (kBT), a proposed dimensionless bias strength, where ell is a *predefined* characteristic landscape length.
- Q_RC = a predeclared measured RC-quality score (e.g. committor prediction loss; direction must be documented).
- A memory/history feature M, estimated separately from independent trajectories.

A combined parameter such as chi*eta can be inspected as an exploratory feature but **must not be claimed as a validated universal criterion** without derivation and cross-system tests.

## What counts as success?

A useful result is not just a colorful scatter plot. It is a verified prediction of **when a kinetic estimate is unreliable**, with a clear baseline, held-out validation, and uncertainty. The result must be honest about excluded/censored trajectories, solver discretization, and force-protocol differences.

## What has already been implemented in this repository?

- Numerical sampling, umbrella windows, MBAR, reaction-coordinate shooting, Jarzynski pulling, toy tauRAMD, toy effective RAMD, MDAnalysis scripts, OpenMM demos and an OpenMM-Torch example.
- **Not yet completed:** an integrated benchmark that measures RC-quality/memory features and compares RAMD kinetic extrapolations to ground-truth MFPT on matched landscapes.

**Therefore this repository is currently an educational literature-to-code portfolio, with a concrete hypothesis for further research, not yet a validated novel RAMD unbiasing technique.**

## Key references

1. Ding Y, Ghidini A, Cossio P, Roux B. *A theoretical framework for random accelerated molecular dynamics simulations*. J Chem Phys (2026).
2. Best RB, Hummer G. *Reaction coordinates and rates from transition paths*. PNAS (2005).
3. Kokh DB et al. *Estimation of Drug-Target Residence Times by tau-Random Acceleration Molecular Dynamics Simulations*. JCTC (2018).
4. Shirts MR, Chodera JD. *Statistically optimal analysis of samples from multiple equilibrium states*. J Chem Phys (2008).
5. Leimkuhler B, Matthews C. *Rational Construction of Stochastic Numerical Methods for Molecular Sampling*. AMRX (2013).
