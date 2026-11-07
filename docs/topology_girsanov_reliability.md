# Out-of-topology RAMD and path-reweighting reliability benchmark

**Exploratory numerical benchmark • November 7, 2026. NOT an experimentally validated ligand residence-time prediction or a new kinetic correction.**

## What changed from our previous work?

Our previous transfer study only changed the height H of one double-well potential. This experiment instead **trains once on a symmetric double well** and measures behavior on four different potential shapes with unrelated analytic derivatives:

| Source/target | \(U(x)\) | Rationale |
|---|---|---|
| SOURCE: symmetric double well | \(2.5(x^2-1)^2\) | Previously used training landscape |
| Harmonic | \((x+1)^2/2\) | Single confining well, no double-well saddle |
| Tilted double | \(1.9(x^2-1)^2+0.38x\) | Asymmetric well |
| Two barriers | \(9[x(x^2-1)]^2\) | Two barriers on escape path |
| Gaussian gate | \(0.38(x+1)^2+2.1e^{-(x-0.08)^2/(2(0.23)^2)}\) | Localized narrow gate |

All remain **separable, dimensionless 2D overdamped toy models**:
\(U(x,y)=U(x)+k_y y^2/2,\ k_y=2\), with reflecting \(x=-2\),
absorbing \(x=0.8\) and start \(x=-1\). These are more diverse than
varying H, but are NOT atomistic protein landscapes or fully coupled 2D barriers.

## Two different kinetic reconstruction procedures — SAME estimand

A core scientific correction is that unrestricted mean first passage
time and a finite-horizon restricted mean are NOT interchangeable. We
compare both methods against the **same** restricted F=0 observable:

\[
R_0(t_{\max})=\mathbb{E}_0[\min(T_{\mathrm{exit}},t_{\max})].
\]

The default observation horizon is 20 dimensionless time units.

**Independent reference**: Solve the time-dependent backward Kolmogorov
equation on a 1D grid, since the target F=0 x dynamics is separable:

\[
\partial_t v=1-U'(x)\partial_xv+k_BT\partial_{xx}v,\quad
v(-2,t)'=0,\quad v(0.8,t)=0,\quad v(x,0)=0.
\]

The value \(v(-1,t_{\max})\) is the reference *restricted* mean,
computed with finite-difference space / implicit-Euler time discretization.
An independent high-resolution double-integral gives the **unrestricted**
F=0 MFPT for diagnostic comparison, but is NOT used as the restricted target.
The PDE solution, quadrature and EM integrator retain finite-resolution
errors; our unit tests verify refinement and a direct F=0 Monte Carlo check.

### Method A — heuristic effective-temperature extrapolation

As a **stress-tested hypothesis, not a theorem of Ding et al.**, calibrate
\(\tau_F\) from an independent feedback pilot at each force; set
\(\alpha_F=k_BT/(k_BT+F^2\tau_F/2)\) for our dimensionless d=2 model.
Fit \(\log R_F(t_{\max})\) linearly against \(\alpha_F\) at three
out-of-training forces \(F=0.7,1.4,2.0\), then extrapolate to
\(\alpha=1\). This is not a validated tauRAMD kinetic-unbiasing formula;
it can be biased even when numerical fits are excellent.

### Method B — conventional discrete Girsanov / path importance weights

For the **specified overdamped Euler–Maruyama** update

\[
\Delta\mathbf r = [\mathbf b(\mathbf r)+F\mathbf n_t]\Delta t+
  \sqrt{2k_BT\Delta t}\,\boldsymbol\xi_t,\quad
\boldsymbol\xi_t\sim \mathcal N(0,I),
\]

record noise and the *already chosen, predictable* force direction
BEFORE each step. The conditional log target/proposal Gaussian
likelihood ratio at one step is

\[
\Delta\log w_t =
  -\frac{F\sqrt{\Delta t}}{\sqrt{2k_BT}}\,
   \mathbf n_t\cdot\boldsymbol\xi_t
  -\frac{F^2\Delta t}{4k_BT}.
\]

Accumulate until first exit or the finite observation horizon. The
feedback policy is defined identically under the biased and unbiased
measures, conditioned on the state/history, so its conditional
direction-reset factors cancel. We preserve raw pre-reflection
Gaussian increments in the *augmented* path measure, which is
important at the reflecting boundary.

The **raw** importance-sampling estimator of the restricted time is
\(\frac1N\sum_iw_i\min(T_i,t_{\max})\). We also report the
self-normalized estimate \(\sum_iw_i\min(T_i,t_{\max})/\sum_iw_i\);
the latter introduces finite-sample normalization bias but is often
numerically more stable.

**Weight degeneracy diagnostic:**

\[
N_\mathrm{eff}=\frac{(\sum_iw_i)^2}{\sum_iw_i^2}.
\]

If the effective sample fraction is tiny, a *formally correct*
change-of-measure estimator can be extremely inaccurate in finite data.
This is exactly why recent research studies, including Marginal
Girsanov Reweighting (2026), seek lower-variance alternatives.
**This code implements conventional full-path weighting only; it
does not reproduce MGR or OpenMM underdamped MD.**

## Frozen switching classifier transfer

Two L2 logistic predictors (time-only and state+time) are fitted
exactly **once** on the source symmetric double well at forces
0.6 and 1.2, using seeds 2026, 2027 and 2028. They are evaluated,
without retraining, on the target shapes and forces 0.7, 1.4 and 2.0,
using independent seeds 2031, 2032 and 2033. The target **known
analytic force** enters the current-state feature representation.
The future checkpoint displacement or switch outcome never does.

The label means **a switch at the next checkpoint, conditional on
survival to that checkpoint**; it does not represent a complete
escape hazard, nor does a predictive score imply a kinetic correction.

## Actual local pilot results (128 trajectories / force / seed)

All four target topologies were held out from training. The following
is a mean across **three independent test seed batches**; do not
treat the thousands of within-trajectory checkpoint observations as
independent scientific replicates.

| Held-out shape | Exact F=0 restricted mean | Heuristic absolute log error | Path-weighted absolute log error | Path ESS / N | Frozen-state log-loss gain vs time-only |
|---|---:|---:|---:|---:|---:|
| Harmonic | 5.8623 | 0.0325 | 0.4250 | 0.0533 | -0.0586 |
| Tilted double | 6.3061 | 0.0819 | 0.2920 | 0.0633 | +0.0068 |
| Two barriers | 2.6913 | 0.1516 | 0.3490 | 0.2343 | -0.0500 |
| Gaussian gate | 9.7677 | 0.1084 | 0.3242 | 0.0174 | -0.0577 |

The path column is self-normalized **at the middle predeclared
force F=1.4**, not a retrospectively selected best force. In the
Gaussian-gate case, an ESS fraction of 0.0174 with 128 trajectories
corresponds to roughly 2.2 effective samples in an average batch.
This is too small for credible quantitative kinetic inference.

The original (source-fitted) state classifier has a *negative*
transfer gain for three of four unseen topologies, suggesting the
earlier within-topology switching gains **do not transfer reliably**.
The gain is descriptive, not a cross-topology reliability classifier.

Importantly, the conventional likelihood-ratio method is correct
**for the specified augmented EM stochastic process in the
infinite-sample limit**; its poor *finite-sample* performance
does not invalidate Girsanov's theorem. The heuristic having lower
error in these twelve batches does NOT establish that it is
scientifically unbiased or better in general.

The force-averaged ESS fractions across all shapes were approximately:
F=0.7: **0.348**; F=1.4: **0.092**; F=2.0: **0.097**.
These are noisy sample estimates, not a monotonic law.

### Caveats about different estimands

For the localized Gaussian gate, the exact unrestricted F=0 mean
was 11.971 while its restricted mean at horizon 20 was 9.768,
a substantial difference. It would be fundamentally incorrect
to evaluate the restricted path estimator against 11.971 as if
both target the same quantity.

## Code and reproducibility

[Topologies and path weighting source](../research/topology_girsanov_benchmark.py)

[Eleven numerical/scientific regression checks](../tests/test_topology_girsanov_benchmark.py)

[Archived four-shape summary](../research/topology_girsanov_pilot_2026.csv)

From repository root with NumPy and Matplotlib installed:

    python -m unittest discover -s tests -p 'test_topology_girsanov_benchmark.py' -v
    python research/topology_girsanov_benchmark.py --quick --output results_topology_quick
    python research/topology_girsanov_benchmark.py --output results_topology_pilot

The main PR GitHub Actions job runs a full numerical pilot and
archives the **full per-force and per-seed CSV files**, frozen
logistic parameters, both analytic references, and the visualization.
The quick smoke is never adequate for scientific inference.

The scientific tests cover numerical force gradients, exact flat-well
first-passage formula, grid/time refinement, Gaussian density-ratio
identity, a Monte Carlo reweighting identity, zero-force weights,
ESS, independent zero-force control, frozen fit count and split
separation, and the complete output pipeline.

## Why this does NOT yet prove a novel RAMD reliability criterion

- Four different 1D landscapes, but still separable two-dimensional
  toy systems; finite numbers of seeds and trajectories.
- No convergence analysis with the numerical EM timestep or
  different horizons, and no bootstrap confidence bounds.
- The state classifier transfers poorly on most new shapes.
- Full-path importance weights suffer poor overlap and often
  virtually zero useful effective sample size.
- The F=0 PDE is solved approximately on a finite grid.
- The heuristic log-linear extrapolation is a demonstration, not a
  published RAMD kinetic inversion derived for our feedback protocol.
- No protein, solvent, underdamped real MD, or experimentally calibrated
  dissociation times are present.

**Next falsifiable work:** demonstrate ESS-versus-force/time scaling,
test low-variance *published* path reweighting baselines such as
Marginal Girsanov or shorter-time MSM estimators if realistically
implementable, include coupled/rotated 2D reaction coordinates and
independent topology families, then test a predeclared reliability
rule on untouched held-out landscapes.

## Literature grounding

- Schäfer & Keller (2024), *Implementation of Girsanov Reweighting in
  OpenMM and Deeptime*, J. Phys. Chem. B.
  https://doi.org/10.1021/acs.jpcb.4c01702
- Wang, Wu & Olsson (2026), *Marginal Girsanov Reweighting: Stable
  Variance Reduction for Long-Time-Scale Dynamics from Biased
  Simulation*, JCTC.
  https://doi.org/10.1021/acs.jctc.6c00777
- Ding, Ghidini, Cossio & Roux (2026), *A theoretical framework for
  random accelerated molecular dynamics simulations*, JCP.
  https://doi.org/10.1063/5.0309358

**These methods already exist.** The present code is an educational
cross-protocol reliability stress test, not a claim to invent
Girsanov reweighting or a new molecular dynamics algorithm.
