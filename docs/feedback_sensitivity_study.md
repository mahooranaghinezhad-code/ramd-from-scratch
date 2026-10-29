# Feedback vs exponential RAMD: force, threshold and timestep sensitivity

**Status (2026-10-29): exploratory numerical pilot on a synthetic two-dimensional potential.**
No protein–ligand simulations, unbiased residence-time reconstruction, or validated new kinetic estimator are claimed.

## Precise question

Does the **displacement-based direction update** used in a tauRAMD-like toy
model produce different force-direction statistics or escape times from
the exponentially persistent direction process used as an approximation
in Ding–Ghidini–Cossio–Roux, *J. Chem. Phys.* (2026)?

A larger outward/inward **force-direction persistence ratio** is a
candidate mechanism, not a proof of extra drift or a reliable correction
to experimentally measured ligand dissociation rates.

## Predefined controlled grid

The accompanying program is
[research/feedback_sensitivity_scan.py](../research/feedback_sensitivity_scan.py).

| Variable | Values |
|---|---|
| Force magnitude F | 1, 3 |
| Checkpoint displacement r_min | 0.25, 0.55 |
| Integration timestep dt | 0.005, 0.0025 |
| *Physical* check interval | 0.1 (20 or 40 steps respectively) |
| Observation horizon | 30 dimensionless time units |
| Independent seed batches | 2026, 2027, 2028, 2029, 2030 |
| Trajectories per pilot, feedback and exponential batch | 256 each |
| Fixed potential | H(x²−1)² + y² (H=2.5), left reflection at x=−2, right absorption at x=0.8 |
| Unbiased control | additional 256 independent trajectories per seed and dt, F=0 |

Every cell has its **own independent feedback pilot** whose total at-risk
exposure divided by its number of direction changes gives the matching
mean exponential persistence time. The pilot data are not reused for
the test comparison. The exponential control and feedback run have equal
force magnitude, potential, integrator, dt and observation horizon.

Matching reset frequencies DOES NOT match the full orientation
correlation, event-time distribution, or effective kinetics.

## Results of one local run

The response is the difference in **restricted** mean first-passage time:

    Delta_RMST = E[min(T_feedback, 30)] - E[min(T_exponential, 30)].

A negative Delta means faster *restricted* escape for feedback within the
simulation horizon; it does **not** mean a reliable estimate of the
unbiased natural mean residence time.

Uncertainty is a pointwise, two-sided 95% Student-t interval across five
independent random-seed batches. This is exploratory: no correction for
eight comparisons, and five batches are insufficient to establish
reliable scientific inference.

| F | r_min | dt | Delta_RMST [pointwise 95% CI] | Feedback outward/inward persistence ratio | Censored (F/E), of 1280 each |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.25 | 0.0050 | −0.195 [−0.757, 0.367] | 1.061 | 3 / 6 |
| 1 | 0.25 | 0.0025 | −0.454 [−0.890, −0.018] | 1.079 | 5 / 8 |
| 1 | 0.55 | 0.0050 | +0.124 [−0.584, 0.833] | 1.028 | 11 / 7 |
| 1 | 0.55 | 0.0025 | −0.121 [−0.736, 0.494] | 1.032 | 5 / 8 |
| 3 | 0.25 | 0.0050 | −0.073 [−0.419, 0.273] | 1.196 | 0 / 0 |
| 3 | 0.25 | 0.0025 | −0.380 [−0.751, −0.009] | 1.259 | 0 / 0 |
| 3 | 0.55 | 0.0050 | −0.234 [−0.571, 0.103] | 1.113 | 0 / 1 |
| 3 | 0.55 | 0.0025 | −0.232 [−0.502, 0.037] | 1.101 | 0 / 0 |

**Interpretation:** Outward force orientations have longer mean
exposure per direction-reset event than inward ones in these feedback
runs, especially for the smaller displacement threshold and stronger
force. However, the escape-time differences are inconsistent across
some settings and statistical intervals overlap zero in six of eight
uncorrected comparisons. The two intervals that do not overlap zero
cannot justify a mechanistic or generalizable claim: the comparison
grid is exploratory, corrections for multiple testing are absent,
and discretization convergence has not been established.

The outward/inward ratio groups orientation by the **sign of n_x**,
NOT by actual x velocity or an independently estimated local drift.
Because groups are sampled from survival-conditioned trajectories,
this diagnostic should not be interpreted as an unbiased causal
drift estimate without additional control experiments.

### Unbiased analytical sanity check

At zero external force the exact *unrestricted* backward-equation MFPT
is about **6.424** time units. The observed restricted means of the
five seed batches, with 256 trajectories each, were:

| dt | Observed F=0 RMST (mean of seed batches) | Censored out of 1280 |
|---:|---:|---:|
| 0.0050 | 6.049 | 9 |
| 0.0025 | 6.413 | 24 |

These are finite-sample, finitely censored *restricted* means. They
cannot be equated to the exact unrestricted MFPT; their differences
also motivate repeating the analysis with additional trajectories
and finer timesteps before quantitative interpretation.

## Reproduction

From the repository root, with Python 3.11+, NumPy and Matplotlib:

    python -m unittest discover -s tests -p 'test_feedback_sensitivity_scan.py' -v
    python research/feedback_sensitivity_scan.py --quick --output results_sensitivity_quick
    python research/feedback_sensitivity_scan.py --output results_feedback_sweep

The quick CI run uses 48 trajectories per batch and an 8-unit horizon
and exists **only to catch regressions**, not to support scientific
conclusions.

The full script writes:
- per_seed.csv: each cell and independent seed, including matched
  persistence, reset counts and censored event counts
- conditions.csv: seed-batch means and t intervals
- unbiased_control.csv: independent F=0 consistency observations
- sensitivity_scan.png: visualization of protocol differences
- LIMITATIONS.txt: cautions carried with the data

The analysis may be re-run with --forces, --thresholds, --dts,
--seeds, --replicas, --horizon, and --check-interval to
probe stability; do not cherry-pick the best-looking condition.

## What must happen before we pitch a new correction

1. Increase independent batches and reduce dt at *fixed physical*
   check interval; determine whether both persistence ratio and
   Delta_RMST converge.
2. Include a **randomized decision control** preserving the observed
   global switching rate while deliberately breaking the
   displacement-to-switch decision link.
3. Compare force-correlation integrals and effective diffusion
   in a free-particle control against the Green–Kubo limit, where
   assumptions can be checked explicitly.
4. Only then compare **unbiased MFPT reconstruction errors**
   using defined force-extrapolation models, on independently chosen
   held-out potential landscapes. Raw Delta_RMST is not reconstruction error.
5. Use survival/censoring-aware estimates and report uncertainty,
   including appropriate adjustment for testing multiple hypotheses.
6. If a proposed reliability diagnostic cannot beat simple baselines
   out of sample, report that negative finding and do not claim
   a universal RAMD correction.

## Research novelty boundary

Paper 1 *already* treats persistence effects, eta and deviations from
its short-correlation assumptions. The distinguishing prospective
contribution here is a **quantitative protocol-aware reliability test**
for displacement-feedback switching, provided it survives controlled
independent benchmarks. The present data do not establish novelty.
