# October 2026 literature update — feedback-aware RAMD validity

**Status: proposal; no new kinetic correction or scientific validation is claimed.**

## Working title

**Feedback-Aware RAMD: Testing Whether Force-Direction History and Hidden-Coordinate Memory Predict Kinetic Extrapolation Failure**

### Simple motivation

RAMD accelerates ligand exit using a randomly oriented force. In *real* tauRAMD, the direction is changed when the ligand COM has made insufficient progress during a prescribed interval. In the simplified effective-theory model, direction resets instead occur after exponentially distributed persistence times. If those two protocols create different correlations, and a poor reaction coordinate ignores slow protein motions, when is recovering an unbiased dissociation time still credible?

## Prior art: what is already known (DO NOT claim as our discovery)

1. **Kokh et al., JCTC 2018** developed the COM-displacement-triggered tauRAMD protocol and relative residence-time analysis. [DOI](https://doi.org/10.1021/acs.jctc.8b00230).
2. **Ding, Ghidini, Cossio & Roux, JCP 2026** derived effective diffusion and temperature for an exponentially persistent random-direction force, addressed relaxation-time effects (eta = tau/tau_harm), and *already studied* persistence-time distributions versus force, including departures at high force. **Neither eta nor noticing non-exponential dwell times is novel.** [DOI](https://doi.org/10.1063/5.0309358).
3. **Best & Hummer, PNAS 2005** connected good reaction coordinates to splitting/committor probabilities and showed how poor projections can exhibit memory. [Article](https://doi.org/10.1073/pnas.0408098102).
4. **Brünig et al., EPL 2025** showed that a popular maximum transition-path-probability criterion need not remain valid under non-Markovian memory. Do not use the p(TP|s) = 1/2 criterion alone to label a coordinate as good. [DOI](https://doi.org/10.1209/0295-5075/adda75).
5. **Keller & Bolhuis, Annual Review of Physical Chemistry 2024** reviewed dynamical (not just equilibrium) reweighting. [DOI](https://doi.org/10.1146/annurev-physchem-083122-124538).
6. **Schäfer & Keller, JPCB 2024** implemented integrator-specific Girsanov path reweighting using OpenMM and Deeptime. [DOI](https://doi.org/10.1021/acs.jpcb.4c01702).
7. **Zhang, Wang, Keller & Wu, JCP August 2026** introduced pi-Girsanov for biased / nonequilibrium multiensemble Markov models. [DOI](https://doi.org/10.1063/5.0335247).
8. **Wang et al., JCTC September 2026** introduced Marginal Girsanov Reweighting (MGR), targeting the variance explosion of long-time path weights. [DOI](https://doi.org/10.1021/acs.jctc.6c00777).
9. **Breebaart & Bolhuis, JCP September 2026** studied diagnostics and validation of committors trained on reweighted path ensembles. [DOI](https://doi.org/10.1063/5.0350496).
10. **Salvalaglio, Tiwary & Parrinello, JCTC 2014** proposed statistical trustworthiness checks for kinetics recovered from infrequent metadynamics. A trustworthiness test alone is not a new concept. [DOI](https://doi.org/10.1021/ct500040r).

**Novelty boundary:** This reading set does not prove that the combined application proposed below is novel. Conduct a targeted citation and keyword survey before making originality claims.

## Narrow research gap to TEST

The original feedback-triggered direction-update protocol is a *path-dependent control policy*, whereas the analytical baseline uses exponentially persistent random directions. It is not yet established by this project's benchmark whether **(i) deviations in force-direction correlations**, **(ii) reaction-coordinate committor prediction loss**, and **(iii) conditional history dependence** can jointly predict **errors in unbiased MFPT extrapolation under the original tauRAMD feedback protocol**, beyond force magnitude and eta.

This is a proposal for a *reliability diagnostic* and quantitative failure map, rather than another generic force-residence-time regression or a brand-new path-reweighting theorem.

## One analytic control (known consequence of correlation theory, NOT novel)

For a free overdamped particle with an independent stationary isotropic orientation process n(t),

    dr/dt = (F/gamma) n(t) + sqrt(2D) noise(t) ,
    D = kBT/gamma.

Define C_n(t) = E[n(t) . n(0)] and dimension d. The long-time diffusion is

    D_eff = D + (F^2/(d gamma^2)) * integral_0^infinity C_n(t) dt.

For C_n(t) = exp(-t/tau), this reduces to Paper 1's expression:

    D_eff = D + F^2 tau/(d gamma^2).

**Important:** In displacement-feedback tauRAMD, n(t) depends on observed progress, which itself depends on position and noise. The independence and stationarity assumptions can fail. Therefore inserting an empirical correlation time into this formula is **a benchmark hypothesis/diagnostic only**, not an exact correction for feedback-driven barrier crossing.

## Falsifiable hypotheses

- **H1 (protocol):** The difference between the true switching protocol's force correlation C_n(t) and the exponential model predicts a portion of the error in effective diffusion/kinetic extrapolation.
- **H2 (hidden variables):** After controlling for force, curvature and characteristic persistence, out-of-sample committor prediction loss and conditional-history dependence predict additional errors in reconstructed unbiased MFPT.
- **H3 (generalization):** A predeclared simple reliability score combining H1 and H2 identifies regimes with MFPT errors exceeding a predeclared factor of two on **held-out** landscapes better than a baseline with only force and eta.
- **Null outcome:** If H1/H2 do not add held-out predictive performance, report the negative finding. Do not fit a universal correction law.

## Concrete implementation milestones

**A. Protocol controls** (adapt the existing `06_tau_ramd/`, `07_ramd_theory/` and `research/` scripts):

- Exponential direction persistence (Paper 1).
- Fixed-interval direction resets (control).
- Displacement-feedback direction changes with recorded checkpoint progress (original tauRAMD logic).
- Same potential, initial-condition distribution, force magnitudes, and integration discretization for fair comparisons.
- Log each integration step's state, force vector, random draw/seed, direction-update decision, elapsed persistence, and whether escape occurred. Full stochastic paths are necessary for any later Girsanov check.

**B. Analytical unit tests:**

- F = 0 reduces to unbiased diffusion; compare MSD/(2 d t) with kBT/gamma.
- Exponential persistence reproduces the analytical D_eff formula in a free-particle test as dt decreases.
- Independent harmonic-well fluctuations check the effective-temperature baseline.
- Test two or more dt and several seeds; include confidence intervals.
- Do not interpret a successful smoke test as quantitative convergence.

**C. Hidden-variable landscapes:**

- Use 2D double wells with a slow transverse y-coordinate, controlling coupling and its mobility. Include at least one distinct held-out potential.
- For each system, run unbiased independent escapes for reference MFPT (or sufficient survival-curve observation) and estimate the committor with independent unbiased shooting.
- Compare reaction coordinates x, x + c*y, and a distance-like proxy, using held-out committor-prediction loss.
- Diagnose history dependence at fixed reaction-coordinate bins (e.g., does previous progress alter the next-state outcome?).

**D. Kinetic reconstruction and error:**

- Apply the effective-temperature/force extrapolation of Paper 1 only where its assumptions are stated and testable.
- Measure E = abs(log(MFPT_estimate / MFPT_unbiased)).
- Record unescaped trajectories as right-censored. Do NOT use an escaped-only mean as an unbiased MFPT.
- Compare explanatory power of force, eta, force-correlation diagnostics, committor loss and memory metrics on held-out landscapes.
- Prefer a transparent model with uncertainty and failure/abstain option over a complex black-box predictor.

**E. Independent baseline and scope:**

- For a compatible *fully specified stochastic integrator*, compare a path-reweighting approach such as Girsanov as a separate control. Long-time weight variance and the endogenous feedback rule require special care; MBAR or Jarzynski alone are not valid shortcuts for reconstructing RAMD residence times.
- Only after success in synthetic landscapes consider an atomistic protein–ligand example; distinguish protocol-relative times from experimentally validated absolute residence times.

## First figure worth creating

Plot measured versus predicted unbiased MFPT, grouped by **protocol** (exponential vs feedback), **slow-coordinate mobility**, and **reaction-coordinate quality**. Display confidence intervals and marked censored/unresolved conditions. Color or stratify by the empirical force-correlation mismatch.

A real scientific result requires held-out validation and a simple baseline. A colorful diagram without data is only a research proposal.

## Current implementation status

The repository's `research/memory_rc_validity_benchmark.py` currently implements **only exponentially distributed direction resets**, uses a simple approximate alpha formula under anisotropic mobility, and omits MFPT estimates when any trajectory is censored. It does *not* implement the full feedback-vs-exponential protocol comparison or a validated path-reweighting benchmark. These are next steps, not completed work.
