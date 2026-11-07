# Held-out landscape transfer and kinetic reconstruction error

**Exploratory research benchmark; not a validated correction for real ligand residence times.**

## What this experiment adds

Previous repository analyses showed that state information can help predict
the NEXT displacement-triggered force switch **on the same potential**.
That is expected from the feedback rule and is not by itself a new physical law.
This follow-up deliberately separates two distinct questions:

1. **Classifier transfer:** If we freeze the predictor fitted on one
   potential, does it still predict a next force-direction switch on
   *previously unseen barrier heights and force values*, without retraining?
2. **Actual kinetic reconstruction error:** When a specific, predeclared
   force-extrapolation model estimates the natural (F=0) mean first-passage
   time, how different is it from the **independently known exact F=0 answer**?
   Does the classifier's held-out log-loss gain track those errors?

**A statistically predictive switching model is not automatically a
kinetic unbiasing technique.** These are separate endpoints.

## Experimental controls and locked split

All runs share one separable two-dimensional overdamped Langevin model:

    U_H(x, y) = H (x² - 1)² + (k_y / 2) y², k_y=2.

Reflecting boundary at x=-2; absorbing exit at x=0.8; start at x=-1.
Dimensionless kBT=gamma=1. dt=0.01, checkpoint interval=0.1,
simulation horizon=80 in the exploratory benchmark.

**Frozen source training:**

- Barrier height H=2.5 ONLY;
- training force magnitudes F=(0.7, 1.6, 2.7) ONLY;
- training random seeds 2026–2029; physical threshold r_min=0.42;
- two models fitted once: time-only logistic and state+time logistic.

**Held-out evaluation:**

- Heights H=(1.5, 2.0, 3.0, 3.5): none equal the training height;
- force levels F=(1.0, 1.9, 2.8): none equal a training force;
- independent test RNG seeds (2031, 2032, 2033);
- initial states, force, dt, noise and force-reset semantics otherwise fixed.

For each H/F/seed, we use a separate feedback **pilot** to calibrate
the force persistence time tau_F. That is NOT a refit of either
next-switch classifier. The target potential height enters the known
physical force-gradient feature; this is an explicit physical covariate,
not learning coefficients from target outcomes.

At a start-of-checkpoint interval, the classifier receives only
current x, y, direction, time and known current physical-force terms.
The future displacement that defines a force reset is never an input.
Labels only exist for trajectories surviving to the next checkpoint.
This is **conditional next-switch prediction**, not competing-risk or
unconditional escape-hazard prediction.

We report separate held-out scores for each seed, force and landscape:
log loss, Brier score, censoring, and a frozen classifier log-loss gain

    G(H,F) = LogLoss(time-only) − LogLoss(state+time).

Positive G means the state model predicts *switching* more accurately
than the time-only model. This is not a kinetic error.

## Exact ground truth for F=0

For the separable U_H and constant diffusion D=kBT/gamma=1, the
backward Smoluchowski boundary-value problem has a known quadrature:

    -1 = -(U'_H(x)/gamma) T'(x) + D T''(x),
    T'(-2)=0, T(0.8)=0,

so the unrestricted unbiased MFPT from x0=-1 is

    T_0 = 1/D * ∫_(x0)^0.8 exp(U_H(y)/kBT)
                [∫_(-2)^y exp(-U_H(z)/kBT) dz] dy.

The code computes the reference independently by fine trapezoidal
quadrature. This analytical control is exact for **the stated F=0
separable toy potential**, not for the biased feedback protocol.

## Explicit and deliberately limited reconstruction baseline

For each target H and each independent testing seed, obtain a *complete*
ensemble of force-accelerated escape times at F=(1.0,1.9,2.8).
At each F, estimate the persistence tau_F from an independent
feedback pilot. Define the *effective-temperature proxy*:

    alpha_F = 1/(1 + F² tau_F / 2),   d=2, kBT=gamma=1,

and fit the **hypothesis** (not a theorem):

    log(mean T_F) = b0 + b1 alpha_F.

Extrapolate to alpha=1 (zero forcing) to get

    T_hat_0 = exp(b0+b1)

and measure reconstruction error:

    E_H = abs(log(T_hat_0 / T_0_exact)).

Repeat the same procedure with a *separate exponential-persistence
control* whose reset time is the feedback pilot tau_F. Compare
feedback and exponential reconstruction errors descriptively.
Because the feedback force direction depends on the path,
a free-particle effective-temperature relation is NOT guaranteed
to hold, and a linear log-MFPT extrapolation is NOT a formula derived
or validated for tauRAMD in Ding et al. This is a deliberately
stress-tested **toy baseline**, NOT a reproduction of that paper's
kinetic correction.

**Censoring rule:** If ANY trajectory in a required forced ensemble is
right-censored, the entire extrapolation for that H/seed/protocol is
marked unusable. We do not compute an "escaped-only" mean and pretend
it is the unrestricted MFPT.

## Association is exploratory, not predictive validation

For each held-out H, average the frozen switching model's log-loss gain
across target forces and testing seeds. Separately average the
absolute log-error of the toy kinetic extrapolation from usable seeds.
Only where at least three distinct H have complete reconstructed times
do we compute a *descriptive* Spearman rank correlation:

- switching gain versus error of feedback extrapolation;
- switching gain versus feedback-minus-exponential extrapolation error.

This correlation must NOT be reported as a validated predictor:
only four distinct test landscapes, all members of the **same
double-well family**, are included, and the extrapolation model may
be structurally misspecified. No p-value for this exploratory
correlation is supplied. If too many paths are censored, the association
is reported as unavailable rather than manufactured.

## How to run

From repository root (Python 3.11+, NumPy, Matplotlib):

    python -m unittest discover -s tests -p 'test_heldout_kinetic_reliability.py' -v
    python research/heldout_kinetic_reliability.py --quick --output results_heldout_quick
    python research/heldout_kinetic_reliability.py --replicas 72 --horizon 80 --output results_heldout_pilot

The PR CI runs the 72-replica exploratory pilot and archives the
full results. Do not use the short "quick" mode for scientific inference.

Outputs:

- frozen_source_models.npz — coefficients, means/scales and source design;
- per_force_heldout.csv — held-out switching metrics and force escape data;
- per_seed_reconstruction.csv — independent-seed kinetic reconstructions;
- landscape_summary.csv — one row per held-out barrier height;
- overview.json — data split, usable counts, descriptive correlations;
- heldout_reliability.png — errors, gains and exploratory scatter;
- LIMITATIONS.txt — limitations carried with outputs.

## Measured exploratory pilot — GitHub Actions run 42383181464

The independent numerical job completed on November 7, 2026. Its output
artifact is **heldout-kinetic-pilot**, from the
[actual PR workflow run](https://github.com/mahooranaghinezhad-code/ramd-from-scratch/actions/runs/42383181464).
The frozen classifiers were trained once, using **40,293 source checkpoint
records**; both model coefficient arrays and source scaling factors are
saved in the workflow artifact.

There were **four held-out barrier heights**, three unseen test force
magnitudes, and three independent evaluation RNG seeds per height.
For this **72-replica/force** pilot, **all 12 height × seed kinetic
extrapolations were usable** and no forced escape trajectory was censored.
There is still considerable independent-seed sampling uncertainty.

The table lists mean absolute log reconstruction errors over the
three independent evaluation seeds. These are dimensionless, not percent
error, and the reported error is NOT the switching classifier's error:

| Held-out H | Exact F=0 MFPT | Frozen switching gain (time minus state log loss) | Feedback kinetic log-error | Exponential kinetic log-error |
|---:|---:|---:|---:|---:|
| 1.5 | 3.8903 | +0.00139 | 0.07757 | 0.07320 |
| 2.0 | 4.9002 | +0.00714 | 0.12020 | 0.08570 |
| 3.0 | 8.6960 | +0.01116 | 0.18386 | 0.09704 |
| 3.5 | 12.0847 | +0.01100 | 0.25856 | 0.09466 |

The table is archived as
[heldout_landscape_pilot_2026.csv](../research/heldout_landscape_pilot_2026.csv).
The artifact includes all per-force and per-seed measurements and
a plot, not just aggregated values.

Across **only four** held-out heights, the descriptive Spearman rank
association between classifier gain and feedback kinetic log-error was
rho≈+0.8. The association with the feedback-minus-exponential
reconstruction error was also rho≈+0.8. **Neither value establishes
significance, causation, or out-of-family predictive generalization.**
Both gain and kinetic extrapolation error may depend on the same
increasing barrier height, making H a serious confounder.

**Scientific reading:** this pilot shows that a frozen switch predictor
can be evaluated without retraining on unseen H and F, and that the
chosen heuristic extrapolation has measurable, sometimes substantial
errors against a true reference. It does **not** show that our
switch-prediction gain independently predicts error in a real RAMD
kinetic correction. To test that harder claim we need independent
potential topologies, stronger uncertainty quantification, and an
externally validated reconstruction procedure.

## Before claiming novelty or scientific reliability

A useful **positive** result would require convincing predictive value
for kinetic errors on many independent potential *topologies*, not merely
changes in H; robust timestep convergence; appropriate survival and
censoring treatment; independent validation; and comparison to a
correctly derived kinetic method rather than this heuristic baseline.

A **negative** result is equally informative: if switching prediction
does not transfer, or its gain does not track kinetic error, we should
document that failure and not assert a universal "RAMD reliability
diagnostic."

This project is a falsifiable computational study, not yet a claim of a
new theoretical result, experimental residence-time estimate, or
publication-ready correction.
