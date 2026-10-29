# First controlled experiment: feedback vs exponential-direction RAMD

**Status:** an educational, testable computational experiment; not a demonstrated
new kinetic unbiasing formula and not a production tauRAMD simulation.

## Purpose

The analytical model of Ding, Ghidini, Cossio and Roux (JCP 2026)
uses random-force directions with exponentially distributed lifetimes.
The tauRAMD method used in practice (Kokh et al., JCTC 2018) checks
the ligand displacement at fixed intervals and changes direction when
progress is insufficient. We want a **fair, explicit numerical control**
of this difference before studying protein memory or committor quality.

This experiment isolates the switching protocol, holding the physical
potential, force magnitude, stochastic integrator, boundaries and noise
parameters fixed. It does **not** assert that feedback inevitably creates
an outward drift; that must be measured rather than assumed.

## Model

Two-dimensional overdamped Brownian motion, with gamma = 1:

    U(x,y) = H*(x^2 - 1)^2 + (k_y/2)*y^2
    dx = [-4*H*x*(x^2 - 1) + F*n_x(t)]*dt + sqrt(2*kBT*dt)*dW_x
    dy = [-k_y*y + F*n_y(t)]*dt + sqrt(2*kBT*dt)*dW_y

The reflecting boundary is x=a=-2; the absorbing exit is x=b=0.8;
the particle starts at x0=-1. For F=0 the y coordinate is separable,
so the **exact unbiased mean first-passage time** is computed from the
1D backward Smoluchowski equation:

    -1 = -U'(x)*T'(x) + D*T''(x),  D=kBT,
    T'(a)=0,  T(b)=0.

Using an integrating factor gives

    T(x0) = (1/D) * integral_(x0)^b exp[U(y)/kBT]
                      * { integral_a^y exp[-U(z)/kBT] dz } dy.

The code uses a fine deterministic trapezoidal quadrature.
This analytic control is valid for **F=0**, not a reference exact kinetic
solution for the externally forced protocols.

## Protocols (each in independent runs)

- **Feedback:** redraw an isotropic force direction only at the end of
  a checkpoint interval of 20 integration steps when the **Euclidean 2D
  displacement** since the last checkpoint is below r_min. This captures
  the central toy tauRAMD feedback logic, not all atomistic details.
- **Exponential:** redraw direction at each integration step with
  probability 1-exp(-dt/tau), the time-discretized memoryless law.
- **Unbiased control:** set F=0 and compare its first-passage time to
  the exact backward-equation result.

A separate feedback **pilot**, with an independent random seed,
sets the exponential mean persistence:

    tau_match = total duration at risk / number of direction resets.

This is a **matching of an empirical average reset frequency**; it does
not make the feedback process exponential, equalize the full dwell-time
distributions, or guarantee identical actual switching rates.
The reported comparison run itself does not determine tau_match.

## Measurements and meaning

1. **Escapes and censoring:** records every trajectory's exit time or
   administrative right censoring at the maximum simulated time.
   Unrestricted MFPT is reported only when **all trajectories escaped**.
   Otherwise report the restricted mean, E[min(T,T_max)], and the
   number censored; **do not** average only the escaped subset.
2. **Force-direction autocorrelation:**
   C_n(lag) = mean[n(t) dot n(t+lag)] for pairs where both samples
   belong to an unfinished trajectory. Exponential theoretical control:
   exp(-lag/tau_match). Time-origin availability varies with lag and
   escape, and the feedback process is not guaranteed stationary.
3. **Outward/inward persistence ratio:** group segments by the sign of
   the **applied force's x component**, with x increasing toward the
   absorbing exit. Define mean-exposure-per-reset for each group,
   then ratio M_out/M_in. This is NOT based on actual observed velocity,
   is NOT a splitting probability, and is not proof of theoretical
   validity when close to one.
4. **Average applied x-force in the bound basin** -1.5 < x < -0.3:
   a descriptive signal of conditioning on the state and survival,
   not an exact effective drift or causal decomposition.

## Run it

From the repository root, with Python 3.11+, NumPy and Matplotlib:

    python research/feedback_vs_exponential.py --quick --output results_feedback
    python -m unittest discover -s tests -p 'test_feedback_protocol.py' -v

For a larger but still preliminary sample:

    python research/feedback_vs_exponential.py --replicas 384 --steps 14000 --seed 2026 --output results_feedback

For a sensitivity study, change --seed, --replicas, --steps and --force
and preserve all output directories separately. To assess uncertainty,
replicate several **independent seeds** and vary timestep, force,
checkpoint interval and threshold in future code revisions.

The script writes:
- summary.csv (three protocols plus exact F=0 control, censoring and reset statistics)
- escape_times.csv (per trajectory, with explicit censoring)
- orientation_correlation.csv (the two protocols and ideal exponential)
- comparison.png (autocorrelation and restricted escape times)
- README.txt (methodological warnings)

The CI "quick" mode uses 96 trajectories per protocol and is a **smoke
test only**. It must never be presented as a convergence study.

## Preliminary local reproducibility check, not a discovery

For one locally executed Python sample (384 trajectories/protocol,
seed 2026, F=2, 14,000 steps):
- The F=0 exact backward-equation MFPT was about **6.424** time units.
  The unbiased sampled MFPT was about **6.132**; all 384 escaped.
- Feedback sampled escape mean was about **4.413**; exponential sampled
  escape mean about **4.688**; all 384 escaped in both.
- The independently calibrated exponential tau was about **0.228**,
  and realized feedback and exponential reset-time proxies were
  approximately **0.228** and **0.223**, respectively.
- The feedback outward/inward persistence proxy was about **1.080**;
  the exponential proxy was approximately **0.991**.

**These differences are not significance-tested, robust across seeds,
or corrected for timestep bias.** In a smaller 96-trajectory local
run the relative escape-time ordering reversed. Thus NO new physical
conclusion about feedback direction, accuracy of Roux's theory,
or protein residence times is claimed.

## Replication across independent random seeds (local exploratory check)

With 384 trajectories **per protocol per seed**, identical physical settings,
and seeds 2026–2030, the feedback-minus-exponential differences in
restricted mean escape time were, in time units:

| Seed | Feedback | Exponential | Feedback - exponential | Feedback out/in |
|---|---:|---:|---:|---:|
| 2026 | 4.413 | 4.688 | -0.275 | 1.080 |
| 2027 | 4.352 | 4.572 | -0.219 | 1.100 |
| 2028 | 4.531 | 4.472 | +0.060 | 1.094 |
| 2029 | 4.556 | 4.605 | -0.049 | 1.087 |
| 2030 | 4.617 | 4.350 | +0.267 | 1.089 |

All trajectories escaped before the 70-unit time limit in these runs.
Across these **five** independently calibrated batches, the mean difference
was about **-0.043** with an across-batch SEM of about **0.098** time
units. The sign reversal makes clear that **these data do not establish
a meaningful change in mean escape time due to feedback** at this
parameter point. The feedback out/in persistence proxy remained above
one, but that does not yet demonstrate a useful kinetic-error diagnostic.

These summaries are exploratory (small number of independent batches,
no grid of force strengths, no timestep extrapolation, and no
multiplicity-controlled uncertainty). Never claim these tables are a
publication-ready statistical study or a validation of ligand residence
times.

## Planned next controls

- Multi-seed confidence intervals and empirical censoring curves.
- Same integrated autocorrelation time, not only event frequency,
  plus explicit dwell-time distribution matching where possible.
- Force F=0 and small-force limits; dt and checkpoint-threshold sweeps.
- A direct comparison to the *predicted* effective diffusion for
  free particles, where analytical assumptions apply.
- A carefully formulated failure diagnostic that correlates measured
  history effects with errors in kinetic extrapolation across
  **held-out potentials**, not just with raw escape times.
- Extend to protein–ligand trajectories only after these controls.

## Literature

- Ding et al., *A theoretical framework for random accelerated molecular
  dynamics simulations*, J Chem Phys (2026).
- Kokh et al., *Estimation of Drug-Target Residence Times by
  tau-Random Acceleration Molecular Dynamics Simulations*, JCTC (2018).
- Best and Hummer, *Reaction coordinates and rates from transition paths*,
  PNAS (2005).
