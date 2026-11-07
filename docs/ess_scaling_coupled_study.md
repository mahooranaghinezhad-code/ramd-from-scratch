# Effective Sample Size vs force/time in coupled feedback RAMD

**Status: controlled numerical pilot / not a validated ligand kinetic method.**

## The specific question

Earlier feedback RAMD benchmarks showed severe **path-weight degeneracy**
when conventional full-path Girsanov reweighting was applied to biased
Euler–Maruyama escape trajectories. This extension asks:

1. Does path-weight ESS decrease with force F and observation horizon?
2. Does a **predeclared** published-style truncated importance weighting
   improve finite-sample accuracy compared to ordinary self-normalized IS?
3. Are these problems present in **genuinely coupled 2D** dynamics, not
   just separable 1D-plus-harmonic toy potentials?

## Physical model (2D, dimensionless)

    U(x,y) = H (x²-1)² + (k_y/2)*(y-c*x)²
    d x = [-∂_x U + F*n_x] dt + sqrt(2*kBT*dt) dW_x
    d y = mu_y*[-∂_y U + F*n_y] dt
                 + sqrt(2*kBT*mu_y*dt) dW_y

H=2.5; ky=2; kBT=1; mu_y=0.4; reflecting x=-2 and absorbing
x=0.8. Start x=-1, y=-c.

At c=0, the model is separable. At c=1.2, the x drift has a true y term:

    dU/dx = 4H*x*(x²-1) - ky*c*(y-c*x)

and mixed Hessian ∂²U/(∂x∂y) = -ky*c != 0.

Force direction follows the displacement-triggered feedback protocol;
checkpoint interval remains physically fixed at 0.1.

## Same time-limited target for all estimators

For each observation horizon h, estimate the **unbiased restricted**
first-passage time

    R0(h) = E_0[min(T_exit,h)].

Each run records the stopped path likelihood and min(T,h) at h=2,4,8
from the same parent trajectory, so estimates at different horizons
are correlated. Reference values come from a **separate, independent,
large F=0 Monte Carlo experiment** using identical coupled dynamics:
three independent batches of 3500 trajectories per coupling.
These references have Monte Carlo uncertainty (SEM); they are NOT
exact backward-PDE results.

## Raw Gaussian path weights

For our anisotropic overdamped Euler–Maruyama update, before applying
the deterministic x-reflection, let z_x,z_y be independent unit
Gaussian innovations. With force F*n applied to both components and
y mobility mu_y, the one-step likelihood ratio satisfies

    log[p0(dz | history)/pF(dz | history)]
      = -F*sqrt(dt/(2*kBT))*(n_x*z_x + sqrt(mu_y)*n_y*z_y)
        -F²*dt*(n_x² + mu_y*n_y²)/(4*kBT).

The force direction at this step is predictable from previous
positions/checkpoints (no lookahead). The direction-update rule is
the SAME under p0 and pF for a given augmented history; its
conditional factors cancel. We stop weights at first exit or horizon.
These weights are exact for the specified **augmented latent Gaussian
Euler–Maruyama path measure**, not automatically an atomistic,
underdamped, continuous-time or marginal reflected-trajectory model.

## Two estimators plus diagnostics

**Method A: conventional full-path SNIS**

    Rhat_A = sum_i w_i * min(T_i,h) / sum_i w_i
    ESS_A = (sum_i w_i)^2 / sum_i (w_i)^2.

**Method B: predeclared deterministic truncated weights**

    w_i^clip = min(w_i, sqrt(N))
    Rhat_B = sum_i w_i^clip * min(T_i,h) / sum_i w_i^clip.

This **Ionides (2008)-inspired** cutoff uses E_q[w]=1 for
**normalized** true likelihood ratios, so sqrt(N) is specified
*before seeing the outcomes*. We then self-normalize the truncated
weights. It is NOT a verbatim reproduction of the original
unnormalized Ionides estimator, nor Pareto Smoothed Importance
Sampling (PSIS), nor Marginal Girsanov Reweighting (MGR).
Truncation can reduce weight instability **at the cost of finite-N
bias** and does not guarantee lower Monte Carlo mean-squared error.

Also report raw IS, normalized weight mean (should be 1 in
expectation), maximum normalized weight share, fraction capped, and
ESS after clipping. ESS is not a guarantee of estimator accuracy.

Source paper:
Edward L. Ionides, *Truncated Importance Sampling*,
J. Comput. Graph. Stat. 17 (2008), 295–311.
https://doi.org/10.1198/106186008X320456

Related but **not implemented**: Vehtari et al.,
*Pareto Smoothed Importance Sampling*, JMLR 25 (2024).
https://www.jmlr.org/papers/v25/19-556.html

## Prespecified experiment grid

| Dimension | Values |
|---|---|
| Coupling c | 0.0 (separable), 1.2 (genuinely coupled) |
| RAMD force F | 0, 0.6, 1.2, 2.0 |
| Finite observation horizon h | 2, 4, 8 |
| Independent test seeds | 2026–2028 |
| Test paths | 192 per force/coupling/seed |
| Separate F=0 reference | 3500 paths per reference seed, 3 independent seeds |
| EM dt / fixed checkpoint interval | 0.01 / 0.1 |

The short --quick mode is for CI smoke testing only.

## Reproduction

From the repository root (Python 3.11+, NumPy and Matplotlib):

    python -m unittest discover -s tests -p 'test_ess_scaling_coupled.py' -v
    python research/ess_scaling_coupled.py --quick --output results_ess_quick
    python research/ess_scaling_coupled.py --output results_ess_pilot

Output files:

- reference_F0.csv — independent F=0 restricted-time estimates and SEM;
- per_seed.csv — individual test batches, all estimators, ESS/weight diagnostics;
- conditions.csv — means and between-batch SD by coupling × F × horizon;
- design.json — all settings;
- ess_scaling_coupled.png — ESS decay and high-force error comparison;
- LIMITATIONS.txt — scientific caveats.

The full pilot is executed and archived by a separate GitHub Actions
pull-request job so the numerical evidence is not a claimed simulation
that was never run.

## Actual independent GitHub Actions pilot results — November 7, 2026

The full prespecified pilot **completed successfully**, using the
[PR workflow run #42386043645](https://github.com/mahooranaghinezhad-code/ramd-from-scratch/actions/runs/42386043645)
and the downloadable **coupled-ess-pilot** artifact. This report
is grounded in the outputs of the actual simulation, not projected
or fabricated numbers.

The exact archived numerical data are:
- [24-condition summary](../research/ess_coupled_conditions_2026.csv)
- [72 independently seeded measurements](../research/ess_coupled_per_seed_2026.csv)
- [6 high-sample independent F=0 references](../research/ess_coupled_F0_reference_2026.csv)

### First validation of a genuinely coupled system

With no external random force and observation horizon 8, the
**independent** F=0 Monte Carlo reference gave

| Coupling c | E_0[min(T,8)] | Independent MC standard error | Total direct reference paths |
|---:|---:|---:|---:|
| 0 (separable) | 4.6588 | 0.0269 | 10,500 |
| 1.2 (coupled) | 6.0685 | 0.0244 | 10,500 |

Thus the same x barrier but a slow cross-coupled coordinate changes
the restricted escape-time statistic. This is *numerical*, not a
proof of a specific protein gating mechanism.

### Force and time versus ESS

All following values are averages across three **independent**
seed batches, each with 192 paths. They are normalized ESS/N,
so 1.0 means equal weights. The specified comparison uses c=1.2.

| Force F | Horizon 2: ESS/N | Horizon 4: ESS/N | Horizon 8: ESS/N |
|---:|---:|---:|---:|
| 0.0 | 1.000 | 1.000 | 1.000 |
| 0.6 | 0.800 | 0.598 | 0.505 |
| 1.2 | 0.313 | 0.162 | 0.128 |
| 2.0 | 0.107 | 0.133 | 0.027 |

ESS generally deteriorated with strong forces and longer histories,
though the F=2 row is **not monotonic** between horizons 2 and 4
in this small finite-seed experiment. The F=2, h=8 result
ESS/N≈0.0267 corresponds to roughly **5.1 effective paths**
from 192 simulated paths per seed.

### Does published-style truncation help?

| Coupling | Force | Horizon | Standard ESS/N | Truncated ESS/N | Standard mean abs RMST error | Truncated mean abs RMST error |
|---:|---:|---:|---:|---:|---:|---:|
| 1.2 | 1.2 | 8 | 0.1285 | 0.1768 | 0.4088 | 0.5415 |
| 1.2 | 2.0 | 8 | 0.0267 | 0.0802 | 0.8884 | 0.8243 |
| 0.0 | 2.0 | 4 | 0.0467 | 0.1322 | 0.3093 | 0.0355 |
| 0.0 | 2.0 | 8 | 0.1061 | 0.1515 | 0.7183 | 0.8186 |

The finite-N estimator with truncated weights can sometimes substantially
decrease **absolute error**, but can also **increase error** while
raising ESS. This is a diagnostic, not proof of unbiasedness
or reproducible improvement. There is only one predeclared cutoff,
a small number of independent batches, and no independent validation
of this particular truncation strategy on different potentials.
ESS alone is **not** a sufficient proxy for correctness.

The conventional and truncated estimators use the same *restricted
time* target as each independently estimated F=0 reference.
Our direct Monte Carlo controls have nonzero standard errors
and the test paths include administrative censoring correctly
through min(T,h).

### Additional scientific caution

The available experiments show strong *weight overlap loss* in
long biased paths under this toy process; they do **not** establish
that a higher-ESS estimator fixes the inverse kinetics problem.
The Ionides-inspired cutoff introduces bias and is not PSIS or MGR.
In particular, in the coupled F=1.2,h=8 condition, clipping made
the estimator's mean absolute error worse by ~0.133 despite higher ESS.
This negative result is relevant to choosing meaningful reliability
criteria rather than assuming large ESS implies correct kinetics.

## Interpretation and what is NOT established

- Strict increase of ESS decay with every parameter is not guaranteed
  for finite samples: report measurements, not a monotonic law.
- Estimator errors are compared to a **Monte Carlo reference** that
  has its own uncertainty; a single short run cannot establish bias.
- Within-horizon data from the same trajectory and different horizon
  snapshots are correlated.
- Even if truncation raises ESS, it may *worsen* kinetic accuracy.
- The result assesses F=0 **restricted** escape-time recovery in a
  synthetic overdamped model. It is NOT a validated tauRAMD residence
  time inversion, experimental drug binding kinetics, or a replication
  of MGR.
- Future stronger low-variance controls could include a properly
  validated, published Pareto-smoothed or marginalized path estimator,
  with held-out tuning and the same target estimand.
