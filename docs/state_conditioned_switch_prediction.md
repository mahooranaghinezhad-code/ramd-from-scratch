# State-conditioned direction-switch prediction: held-out seed benchmark

**Status:** exploratory toy-model research, 2026-11-07. These results are
NOT a newly validated RAMD residence-time correction, a new theorem,
or a full atomistic τRAMD calculation.

## Motivation and scientific question

The displacement-feedback rule in the τRAMD-like toy changes force
direction when the particle's displacement during an interval fails to
reach the predefined threshold. That decision is **deterministic
given the displacement**; predicting it *after* the interval would be
tautological and would constitute label leakage.

Instead, at the **start** of each interval, after any previous reset,
we predict the **next** checkpoint decision using only the current
position, force direction and current time:

    P(S_{t+Δ}=1 | x_t, y_t, n_x(t), n_y(t), t,
                      trajectory survives to t+Δ).

The survival qualification matters. A trajectory that exits before
the checkpoint does not get a "no switch" label. Therefore this
experiment does NOT yet model unconditional switching/escape hazards.

We test the state dependence by comparing models trained on strictly
independent random seed batches, and by running a **negative control**
with checkpoint decisions randomized across live particles at the same
pilot-matched global reset rate (see the earlier control study).

## Three competing predictors

1. **Constant switching rate:** training-data event fraction.
2. **Time-only predictor:** regularized logistic regression using t and t².
   This controls simple survivor-composition/time effects.
3. **State + time predictor:** same logistic-regression optimizer, using
   t, t², x, y, n_x, n_y, physical force_x from the known potential,
   force_x*n_x, F*n_x and F*n_y.

Only the physical model's **current state** is used. The future
interval displacement and the future switching label are NEVER included
in the predictors. This is explicitly asserted in unit tests.

**Important:** every checkpoint outcome is correlated with other
checkpoints on the same trajectory. Held-out splitting is by
**independent random seed batches**, not by individual event rows.
Pointwise confidence intervals below use the four independent test
seed means, NOT the thousands of correlated events as independent
statistical samples.

## Experiment design

- Potential: H (x²-1)² + (k_y/2) y², with H=2.5 and k_y=2;
  reflecting x=-2, absorbing x=0.8.
- Fixed physical time step: dt=0.005; checkpoint interval Δ=0.1;
  horizon=16 dimensionless time units.
- Conditions: (F=1, r_min=0.55) and (F=3, r_min=0.25).
- 192 independent stochastic replicas per seed and protocol.
- **Training seed batches:** 2026, 2027, 2028, 2029, 2030.
- **Held-out seed batches:** 2031, 2032, 2033, 2034.
- A fully independent feedback pilot with seed stream 9999
  calibrates the randomized checkpoint reset probability.
- The code repeats the same study for feedback and randomized controls.
- Regularization coefficient: 0.002; features are standardized using
  training data only. There is no model or hyperparameter search
  on the held-out batches.

### Criterion

Use cross-entropy log loss and Brier score on the held-out batches.
Define

    ΔLL = LL(time-only predictor) - LL(state+time predictor).

A **positive** ΔLL means current-state information improves prediction
of the *next switching decision* over a time-only predictor. This is
not an improvement in estimated natural dissociation time.

## Preliminary independent-seed results

The local pilot produced the following **descriptive** means over four
held-out seed batches. The 95% intervals use t(3) = 3.182 on the
four per-seed ΔLL values, with no adjustment for multiple comparisons.

| Force / threshold | Policy | Held-out checkpoint events | Mean ΔLL | Pointwise seed-level 95% interval | Positive seed batches |
|---|---|---:|---:|---:|---:|
| F=1, r_min=0.55 | Feedback | 41,044 | +0.01017 | [+0.00792, +0.01242] | 4 / 4 |
| F=1, r_min=0.55 | Randomized checkpoints | 44,449 | -0.000014 | [-0.000110, +0.000082] | 2 / 4 |
| F=3, r_min=0.25 | Feedback | 25,088 | +0.00444 | [+0.00366, +0.00523] | 4 / 4 |
| F=3, r_min=0.25 | Randomized checkpoints | 27,067 | -0.000037 | [-0.000120, +0.000045] | 1 / 4 |

The two feedback conditions correspond to a roughly **1.56% and
0.92% relative decrease** in mean held-out log loss versus the
time-only baseline, respectively. The randomized controls do not show
a consistent improvement from state features.

This provides **preliminary, narrow evidence** that start-of-interval
state contains predictive information for the next switch under
displacement feedback, beyond time alone, while the same state
features do not consistently improve prediction when the switching
decision is randomized.

### Why this is NOT yet a discovery

A dependence on the state is physically *plausible and expected*
because the displacement-based feedback rule was designed to depend
on motion. Our empirical result is a controlled demonstration, not
proof that this feature is novel, generalizable, or useful for
reconstructing unbiased kinetics.

Additionally:

- Test batches use independent RNGs but **the same two potential
  and force/threshold conditions as training**. No held-out
  landscapes, dt extrapolation or protein–ligand systems have
  been tested.
- The primary comparison was formulated after exploratory work,
  not pre-registered in an independent prospective study.
- Repeated events are correlated, and selection on surviving to
  a checkpoint can create additional dependencies. A full analysis
  should model escape/censoring and switching jointly.
- A small improvement in log loss is NOT a reliability diagnostic for
  a kinetic extrapolation. That would require known unbiased
  MFPTs and an explicit reconstruction procedure.

## Code and reproducibility

Implementation:
[research/state_conditioned_switching.py](../research/state_conditioned_switching.py)

Ten unit/regression tests:
[tests/test_state_conditioned_switching.py](../tests/test_state_conditioned_switching.py)

Run from repository root:

    python -m unittest discover -s tests -p 'test_state_conditioned_switching.py' -v
    python research/state_conditioned_switching.py --quick --output results_state_switch
    python research/state_conditioned_switching.py --output results_state_switch_full

The script produces:
- heldout_seed_metrics.csv (each held-out independent seed and model),
- summary.csv (per-condition and protocol summary),
- state_calibration.csv (calibration-bin aggregates),
- heldout_model_comparison.png (prediction comparison),
- LIMITATIONS.txt (qualification for downstream readers).

The archived
[full-pilot summary](../research/state_conditioned_switch_summary_2026.csv)
records the mean scores from the independent local simulation.
The archived [16 held-out seed-level metrics](../research/state_conditioned_switch_heldout_seeds_2026.csv)
include log loss and Brier score for each seed and protocol. For full
calibration bins and reproducibility, run the script as documented.

Calibration is imperfect in some sparsely populated bins; improved
aggregate log loss is not evidence of a calibrated kinetic hazard.

The unit tests verify: equality of instrumented and baseline
dynamics for both protocols with identical RNG seeds; no future
displacement/label in model input; correct event labels in
feedback; unique/disjoint seeds; finite probability outputs; and
a small full-pipeline smoke execution.

## Next falsifiable steps

1. **Held-out landscapes:** train on H=2.5 then evaluate on a
   distinct barrier height and distinct force/threshold regimes
   without retraining or selecting conditions after viewing tests.
2. **Timestep, survival and calibration:** refine dt with a fixed
   physical checkpoint interval and perform survival-aware modeling,
   not only event classification among survivors.
3. **Kinetic reliability (a separate study):** calculate unbiased
   MFPT or full survival distributions under an independently
   validated model; define a specific forced-to-unbiased kinetic
   reconstruction and directly test whether the state-dependent
   switching diagnostic predicts *reconstruction error* on held-out
   potentials. Until then the claim is about switching decisions,
   NOT residence-time recovery.

Prior art includes Ding et al. (JCP 2026), Kokh et al. (JCTC
2018), and state-dependent run-and-tumble switching in active matter.
State-dependent switching as a general phenomenon is NOT claimed new.
