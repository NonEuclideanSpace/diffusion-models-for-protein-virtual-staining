# Module M1, pre-registered

Written before any model is trained on real data, so that the gate cannot be adjusted after
seeing the answer. Every threshold below is derived from simulation at the sample sizes the
data actually provides, not chosen for convenience. The simulation is in `m1-feasibility.md`
and is reproducible from `pvs.eval.calibration`.

## Claim

Conditional diffusion virtual staining reproduces a sharpened, dominant-mode version of the
compartment distribution rather than the distribution itself, and the sharpening increases
with classifier-free guidance weight.

## What is measured

For each protein `g` in the study set:

    pi_emp[g]  = SubCell's mean predicted compartment distribution over the real cells of g
    pi_hat[g,w] = the same over N generated cells of g at guidance weight w
    D[g,w]     = total variation between them

The reported statistic is the **population mean over proteins**, `D_bar[w]`, as a function of
guidance weight. The claim is about the slope of `D_bar` against `w`.

## Fixed in advance

| choice | value | why |
|---|---|---|
| divergence | **total variation** | Highest detection rate at the real parameters, and bounded. Plug-in KL is not merely weak: over the measured empirical distributions its detection rate **falls** as proteins are added, from 14.3% at 50 to 5.3% at 448, which is below the 4.2% expected by chance. Averaging concentrates its bias rather than cancelling it, so more data actively makes it worse. |
| class vocabulary | SubCell `Grouping 3`, **7 groups + Other** | At 31 classes with 75 cells per protein, every divergence sits below its own noise floor. The eight ungrouped mitotic annotations go to Other rather than being dropped, so probability mass is conserved. |
| cells per protein | **at least 50**, pooled across cell lines | Conditioning on cell line gives a median of 18, which is unusable. Pooling gives 75. The cost is that the claim is scoped to "pooled across the lines HPA imaged this protein in" and cannot be stated per line. |
| proteins | **at least 150**; 448 are available | Simulation over the measured empirical distributions gives 99% detection at 50 and 100% at 150. The earlier figure of 50 came from assuming Dirichlet-distributed references; the real ones are more concentrated, so 50 is no longer comfortable. |
| generated samples | **100 per protein per weight** | |
| guidance weights | at least four, spanning w=1 upward | A monotone trend needs at least three points to be meaningful and four to be convincing. |

## Gate

**Pass** requires all three:

1. `D_bar[w]` is strictly increasing across the guidance weights.
2. A paired permutation test over proteins rejects a flat trend at p < 0.01. Under simulation
   a perfectly calibrated model produces a monotone sequence 2.0% of the time by chance, so
   monotonicity alone is not sufficient evidence.
3. Per-sample fidelity does not decrease across the sweep, by a metric fixed before the run.
   Without this the module is trading quality for diversity, which is trivial.

**Fail** is any of: a non-monotone or decreasing population trend, a permutation test that
does not reject, or fidelity that falls with `w`.

A failure is reported as a negative result with these thresholds quoted, per `INTERNAL.md`
section 9 item 2. It is not grounds for changing the estimator and re-running.

## What cannot be claimed, whatever the outcome

**No per-protein number.** At K=7 and 75 cells the best estimator clears its own 95th-percentile
noise floor by a factor of 1.03. A single protein's D is not resolvable and must not be quoted,
plotted per protein, or used to rank proteins.

**pi_emp is a prediction, not a measurement.** No public per-cell compartment labels exist;
SubCell's heads were trained on labels broadcast identically across each field of view. If
that training regressed cells toward the consensus, both `pi_emp` and `pi_hat` are compressed
and D is biased downward. The direction of that bias is toward accepting the null, so a
**positive** result is conservative and a **negative** result is not interpretable without
the calibration evidence from probe A.

**Cell line is not controlled.** Pooling was necessary to reach a usable sample size.

## Dependencies

Probe A must pass first. It asks whether SubCell's predicted per-cell distributions carry any
heterogeneity signal at all, tested against HPA's own independently curated
`Single-cell variation spatial` flag. If predicted entropy does not separate flagged proteins
from matched controls, `pi_emp` carries no within-protein information and M1 has no target,
whatever the diffusion model does.

## Pre-registration record

The parameter table and the gate above were fixed on 2026-08-21, before any real-data model
existed. The simulation supporting them ran at K=7 (seven groups plus Other), n_real=75,
n_gen=100, over 300 trials per configuration, **resampling reference distributions from the 40
measured in probe A** rather than from a parametric family. Those measured references have a
dominant component of 0.413 at the median and an effective support of 4.55 groups, both more
concentrated than a Dirichlet(1.2) would give, which is why the protein-count requirement rose
from 50 to 150.

---

## Amendment 1, 2026-08-21 — selection and pooling

Made **before any model output exists**, on evidence from probe A alone. Recorded here rather
than by editing the table above, so the original commitments stay legible.

### What forced it

`heterogeneity-selector.md`: HPA's single-cell variation annotation explains **2%** of the
between-gene variance in measured localization heterogeneity. Dominant cell line explains
**23.5%**. The 51 genes of the expanded probe that had never been looked at put the
annotation's effect at 0.438 [0.257, 0.622] — below the null, direction reversed.

Two commitments above rest on that annotation without saying so.

### A1.1 — Protein selection is by measured spread, not by annotation

The row `proteins | at least 150; 448 are available` counted the annotation-flagged pool.
That pool is not enriched for the quantity M1 measures, so the count was never the relevant
one. **Proteins are selected by measured empirical spread**, ranked within the screen, taking
the top of the distribution rather than the top of the label.

This is a selection rule, not an estimator change, and it is fixed now — before any generated
sample exists. `INTERNAL.md` section 9 item 2 forbids changing the estimator after a failure;
it does not require keeping a selector that the data has shown to be uninformative.

### A1.2 — Pooling across cell lines is dropped

The row `cells per protein | at least 50, pooled across cell lines` accepted a scoping caveat
to reach 75 cells. Pooling is now the single largest identified confound, and it is no longer
necessary:

| requirement | U2OS alone |
|---|---|
| genes at >=40 cells | 1,608 |
| **genes at >=50 cells** | **661** |
| genes at >=75 cells | 96 |

661 is more than four times the 150-protein requirement. **M1 runs in U2OS only.** The claim
gains a cleaner scope — one cell line, stated — and loses the pooled caveat entirely.

The cost is real: **n_real drops from 75 to a median of 58**, and the power table above was
simulated at 75.

### A1.3 — The power calculation is void and must be re-run

Two of its inputs are now wrong:

1. It resampled reference distributions from **the 40 measured in probe A**, which were
   annotation-selected. Section A1.1 rejects that sample as unrepresentative of what M1 will
   actually be run on.
2. It assumed `n_real=75`. Section A1.2 sets 50-58.

Until it is re-run on the U2OS screen's reference distributions at n_real=50, **the
150-protein figure and the "99% detection at 50" claim carry no weight and must not be
quoted.** The gate in the `Gate` section is unchanged and remains binding.

### What is not changed

Divergence (total variation), class vocabulary (Grouping 3, 7+Other), generated samples per
protein (100), guidance weights (at least four), and all three gate conditions stand exactly
as pre-registered on 2026-08-21.

### Dependency

Blocks on the U2OS screen (600 genes x 40 cells, `experiments/supervise.sh` stage `u2os`),
which supplies both the selection ranking and the reference distributions the re-run needs.

---

## Amendment 2, 2026-08-22 — the power calculation, re-run

Amendment 1.3 voided the original. This replaces it.

### Inputs

Reference distributions are the first **100 genes of the U2OS screen**, measured rather than
annotation-selected, at 7 groups plus Other. They are more concentrated than the voided
calculation assumed:

| | this calculation | the voided one |
|---|---|---|
| dominant component, median | 0.428 | 0.413 |
| effective support, median | **3.47 groups** | 4.55 groups |

More concentrated means harder, so this is not a favourable substitution.

Collapse model: the reference mixed toward a point mass on its dominant bin
(`collapse_toward_mode`) at weights 0, 0.08, 0.16, 0.24 across the sweep. 100 generated samples
per protein per weight. Gate: mean trend positive **and** a sign-flip permutation test at
alpha 0.01. 60 trials per cell, 600 permutations.

### Result

| n_real | 50 proteins | 100 | 150 | 300 |
|---|---|---|---|---|
| **50** (U2OS alone) | 85.0% | **100%** | **100%** | **100%** |
| 58 (U2OS median) | 93.3% | 100% | 100% | 100% |
| 75 (the old pooled figure) | 95.0% | 100% | 100% | 100% |

**150 proteins at 50 real cells each gives 100% detection.** Amendment 1.2's decision to drop
cross-cell-line pooling — which removed the largest identified confound — therefore costs
nothing in power. The pre-registered 150-protein requirement stands, now on its own evidence.

### Two traps found while running it

1. **A permutation test cannot report p below `1/(draws+1)`.** At 80 draws the floor is 0.0123,
   above alpha 0.01, so detection was 0% at every configuration regardless of effect size. The
   script now refuses to run when the floor cannot clear alpha.
2. **`trend_statistic` scored a perfectly flat sweep as +1.000**, because ordinal ranking on
   ties hands back the original order (`m1-rehearsal.md`). A sweep that did nothing would have
   passed gate condition 1. Fixed with average ranks; a sweep with no rank variance now scores
   0.0.

The whole table above was re-run after the second fix. Only one cell moved — 50 proteins at
n_real 50, from 86.7% to 85.0%, which is one trial in sixty. Every 100/150/300-protein row is
identical. **The conclusion does not depend on the bug.**

Reproduce:

    uv run python experiments/m1_power.py --trials 60 --permutations 600 \
        --proteins 50 100 150 300 --n-real 50 58 75 --divergence total_variation
