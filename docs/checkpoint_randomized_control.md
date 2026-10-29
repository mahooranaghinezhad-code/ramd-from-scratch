# Displacement feedback vs count-matched randomized checkpoints

**Status:** independently reproducible *two-dimensional stochastic toy*
experiment, not an atomistic tauRAMD simulation, a new kinetic correction, or
evidence of bias-free residence-time recovery.

**Research question:** Does conditioning a RAMD direction switch on the
*individual* particle's displacement matter above and beyond the number
and periodic schedule of direction changes?

## Why a third control is necessary

Our earlier study contrasted feedback-triggered switches at fixed checkpoints
and a Poisson (exponential-persistence) process that may switch at *any* step.
Those two processes differ in **both** the information used for switching
and the schedule on which switching is allowed.

The new control changes only the first feature: at the same physical
checkpoints, select a random subset of still-active trajectories for reset,
without inspecting each trajectory's displacement or progress. The fraction
selected is calibrated from an **independent feedback pilot** on the same
potential, force, threshold, integration dt and checkpoint interval.

This is a stricter, but not perfect, counterfactual for the feedback rule.
It matches the **pilot-calibrated global rate**, rather than the exact
time-varying feedback hazard or all higher-order force correlations.
The randomization quota depends on the number of trajectories still at risk
(alive), so it must not be described as entirely independent of survival.

## Mathematical model

Use the same overdamped Langevin double well as our previous experiments:

    U(x,y) = H (x² - 1)² + (k_y / 2) y²

    dx = [-∂_x U + F n_x(t)] dt + sqrt(2 k_B T dt) ξ_x
    dy = [-∂_y U + F n_y(t)] dt + sqrt(2 k_B T dt) ξ_y

with reflecting x=-2, absorbing x=0.8, start x=-1. All quantities are
dimensionless. The integration scheme, potential, noise, initial position,
force and checkpoint times are identical across accelerated protocols.

Define, in an independent feedback pilot:

    p_pilot = N_resets_pilot / N_eligible_checkpoint_visits_pilot

At each checkpoint in the randomized control, let M be the cumulative
number of eligible trajectory-checkpoint visits and R the cumulative
reset count. Select exactly

    max(0, round(p_pilot * M) - R)

of the presently eligible trajectories **uniformly without replacement**.
(Any infeasible quota is treated as an error.) Thus the total realized
count remains within one half event of p_pilot * M. No displacement
information chooses *which* individual changes direction.

We compare three policies:

| Protocol | When reset can occur | What determines reset |
|---|---|---|
| feedback | every fixed checkpoint | observed 2D displacement below r_min |
| checkpoint_random | every fixed checkpoint | uniform random selection under pilot reset quota |
| exponential | any numerical step | memoryless random reset probability 1-exp(-dt/tau) |

The exponential mean tau comes from the separate feedback pilot's at-risk
exposure divided by resets, as in our previous baseline.

### Measurement caveats

Use the **restricted mean first-passage time** (RMST) over a fixed horizon
rather than silently averaging only trajectories that escaped:

    RMST = E[min(T_escape, t_max)].

The observed difference is

    Delta = RMST_feedback - RMST_checkpoint_random.

This **is not** an error in reconstructed *unbiased* MFPT: no such
reconstruction is performed here. Right-censored paths remain accounted
for in RMST. The unrestricted true MFPT is not identifiable from this
finite-horizon mean without additional assumptions.

A secondary descriptor, the outward/inward persistence ratio, compares
at-risk force-direction exposure per reset according to the **sign of n_x**.
It does not directly measure progress, physical drift or causality.

For the F=0 separable control, the exact *unrestricted* mean of the
1D first-passage problem is calculated using the backward Smoluchowski
integral. The finite-horizon simulated RMST is not expected to equal that
value precisely, especially with right-censored paths.

## Reproducible parameters

- Four conditions: F = 1 or 3; r_min = 0.25 or 0.55.
- Physical checkpoint interval = 0.1, dt = 0.005.
- Horizon = 30 dimensionless time units.
- Five independent RNG seed batches (2026–2030).
- Per condition and seed: 256 independent trajectories for pilot,
  feedback, randomized checkpoint and exponential simulations.
- Additionally, independent F=0 controls (256 trajectories per seed).
- Twenty independent *condition × seed* batches in total. Randomness for
  pilot and measured runs is separated by NumPy SeedSequence streams.

Run from the repository root (NumPy and Matplotlib are required):

    python -m unittest discover -s tests -p 'test_randomized_checkpoint_control.py' -v
    python research/checkpoint_randomized_control.py --quick --output results_checkpoint_quick
    python research/checkpoint_randomized_control.py --output results_checkpoint_control

Outputs: per_seed.csv, conditions.csv, unbiased_control.csv,
checkpoint_randomized_comparison.png and LIMITATIONS.txt. The CI quick mode
has a short horizon and must **never** be used for quantitative inference.

## Exploratory local pilot (October 29, 2026)

The source and tests were executed locally with Python and NumPy. In the
table, the uncertainty is a pointwise two-sided Student-t 95% interval
across **five seed-batch differences**, not across individual correlated
trajectory samples.

| F | r_min | Feedback - randomized RMST | Pointwise 95% CI | p (uncorrected) | Bonferroni p across 4 | Feedback vs randomized outward/inward |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.25 | -0.113 | [-0.682, 0.456] | 0.611 | 1.000 | 1.069 / 1.000 |
| 1 | 0.55 | +0.010 | [-0.369, 0.389] | 0.946 | 1.000 | 1.030 / 0.996 |
| 3 | 0.25 | -0.382 | [-0.952, 0.189] | 0.137 | 0.546 | 1.219 / 1.052 |
| 3 | 0.55 | -0.336 | [-0.578, -0.094] | 0.0183 | 0.0733 | 1.107 / 1.009 |

The [archived rounded per-seed pilot CSV](../research/checkpoint_randomized_pilot_20batches.csv)
contains all 20 condition × seed summaries. Values in that public CSV are
rounded for readability; the Python code generates full-precision per_seed.csv
and conditions.csv. No individual trajectory data are archived in the
summary file.

**No robust kinetic conclusion:** One uncorrected time-difference CI does
not cross zero, but the Bonferroni-adjusted p is 0.0733. Also, four
force/threshold combinations and five independent batches are not enough
to establish generality or timestep convergence.

There is a more suggestive *descriptive* change in the ratio of force
orientation exposures per reset. That proxy is survival-conditioned and
force-sign-defined; it does not yet establish an emergent physical drift
or a reliability condition for kinetic unbiasing. It is exploratory and
was not pre-registered as a primary endpoint.

**Control of the matching rate:** Every randomized batch had exactly
zero reset-count quota error, and the maximum absolute difference between
the randomized realized reset fraction and its independent pilot value was
about 0.000052 (across 20 batches). The corresponding *feedback comparison
batch* can fluctuate around the pilot rate; this does not match its exact
realized direction-change count.

The five F=0 controls had a mean simulated RMST ~6.123, with 9 censored
trajectories of 1280, versus the exact *unrestricted* MFPT ~6.424. This
sanity comparison does not show integrator/finite-horizon convergence.

## Limitations to resolve before claiming a paper contribution

1. The count-matching quota couples decisions across alive particles;
   a second, independent Bernoulli-at-checkpoints control is useful.
2. Condition on particle position and time-before-escape to separate
   survivor selection from the effect of feedback on direction persistence.
3. Repeat with smaller dt, fixed physical checkpoints and multiple
   independent seed batches; compare empirical force-direction
   autocorrelation and residence-time distributions.
4. Estimate *kinetic reconstruction error* against an unbiased reference
   in several held-out potential landscapes, with explicit assumptions
   and proper censoring-aware uncertainty. This study measures only
   accelerated-time differences.
5. The effective diffusion/temperature theory of Ding et al. already
   addresses important persistence limitations. This work is a
   **proposed complement**, not a refutation or a demonstrated new formula.

## Literature

- Ding, Ghidini, Cossio & Roux, *A theoretical framework for random
  accelerated molecular dynamics simulations*, J. Chem. Phys. (2026).
- Kokh et al., *Estimation of Drug-Target Residence Times by
  tau-Random Acceleration Molecular Dynamics Simulations*, JCTC (2018).
- Best & Hummer, *Reaction coordinates and rates from transition paths*,
  PNAS (2005).
