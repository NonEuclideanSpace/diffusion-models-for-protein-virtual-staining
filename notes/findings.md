# Findings

Everything established so far, with the numbers, ordered by how much it changes the project.
Each entry says what was believed, what was measured, and what it forces.

---

## A. Results from real data

### A1. RETRACTED — the annotation does not select for heterogeneity, but heterogeneity is real

> **Superseded by `heterogeneity-selector.md`.** The completed expanded run's 77
> never-before-seen genes put the effect at **0.433 [0.303, 0.564]** — below 0.5, direction
> reversed, with balanced groups and the cell-line composition distance down to 0.232. The
> discovery result was a discovery artefact. What survives is the quantity itself: split-half
> reliability 1.000, independent-sample reliability 0.923, kurtosis 9.4, visible in head-free
> embedding space at rho +0.591. Cell line explains **0.236** of the between-gene variance; the
> annotation explains **0.004**, and the heterogeneity top quintile is 48% flagged against a 48%
> base rate — enrichment 0.99x. M1's selector is replaced by a measured-heterogeneity screen
> within U2OS; see `m1-protocol.md` Amendment 1.

The original entry follows.

#### A1-original. Probe A is inconclusive — the apparent effect is confounded by cell line

40 genes, 2400 real HPA cells. The primary statistic is the between-cell mutual information
`H(mean prediction) − mean(per-cell entropy)`, which removes each cell's own uncertainty and
leaves only variation between cells.

| statistic | flagged | control | effect [95% CI] | z |
|---|---|---|---|---|
| **between-cell heterogeneity** | 0.0995 | 0.0660 | **0.693 [0.525, 0.848]** | +2.08 |
| entropy of the mean | 2.3127 | 2.4627 | 0.450 [0.265, 0.630] | −0.54 |
| **embedding spread** | 2.7787 | 2.4315 | **0.725 [0.555, 0.870]** | +2.43 |

**But controls were not matched on cell line, and that breaks it.** Suspension and
round-morphology lines (HEL, REH, SK-MEL-30) appear almost only among controls, and genes
dominated by them show a median heterogeneity of 0.0179 against 0.0978 for the rest — a 5.5x
difference driven by cell shape rather than protein. Removing the nine affected genes:

| statistic | all 40 | confound removed (19 flagged, 12 control) |
|---|---|---|
| heterogeneity | 0.693 [0.517, 0.845] | **0.601 [0.395, 0.794]** |
| embedding spread | 0.725 [0.560, 0.873] | **0.645 [0.421, 0.842]** |

Both intervals now include 0.5. **The direction survives, the significance does not.**

**Consequence:** whether `pi_emp` carries within-protein information is unsettled, and M1
cannot proceed on this evidence. A 116-gene run with cell-line-matched controls is the test.

**The embedding result carries the most weight** because it never passes through a classifier
head, and the heads are where the circularity lives — they were trained on labels broadcast
identically across each field of view. The encoder's objective was protein-level, not
per-cell, so its representation separating annotator-flagged genes is evidence the
heterogeneity is real rather than an artefact of the label pipeline.

**Caveats:** 20 genes per group, lower CI bound 0.525 barely clears 0.5, and the magnitude is
small (median 0.074 nats where uniform over 31 classes is 3.434). An expanded run over 116
genes is in progress. Details in `probe-a.md`.

### A2. SubCell agrees with HPA on which compartment, 82.5% of the time

Top-1 group matches an annotated location for 33 of 40 genes; the annotation is in the top two
for 95%. Chance, weighted by annotation frequency, is 22.9%. Three of the seven misses involve
lipid droplets, a rare class; most of the rest are low confidence. The pipeline works before
any question about heterogeneity is asked.

---

### A3. CDC20's two populations are cell-cycle populations, and the prediction was stated first

The heterogeneity measure split CDC20's cells into Nucleoplasm (27) and Negative (13). CDC20 is
destroyed by the APC/C in G1, so if that split is real the Negative cells should carry less DNA.
DNA was measured as DAPI inside the cell mask — **a different channel, which the classifier
never sees**.

| | Nucleoplasm 27 | Negative 13 | ratio | p |
|---|---|---|---|---|
| integrated DAPI | 26086 | 17652 | 1.48x | 0.0021 |
| **nuclear area** | 44816 | 42620 | 1.05x | **0.396** |
| **mean DAPI** | 0.6 | 0.4 | **1.50x** | **<0.0001** |

Same-sized nuclei, 1.5x the chromatin density. That is the G1 against G2/M signature and cell
size cannot produce it. Mann-Whitney effect 0.909 on mean DAPI.

INCENP separates too (p 0.0060) but its signal is mostly nuclear area (1.7x), consistent with
mitotic envelope breakdown and also with simply larger cells — **suggestive only**. FAF2, the
negative control, does not separate on total DNA (p 0.105) as predicted, though its two
populations do differ in area and density in compensating directions, so they are not
morphologically identical. Localization heterogeneity and cell-state heterogeneity are
entangled, and M1 will have to handle that.

---

## B. Findings that change what the project can claim

### B1. Module M1's measurement plan does not resolve as originally specified

At the sample sizes HPA provides — 7 compartment groups, 75 cells per protein — **no
divergence resolves a single protein**. The best clears its own 95th-percentile noise floor by
a factor of 1.03.

The population trend across guidance weights does resolve. Simulated over the 40 empirical
distributions measured in probe A:

| estimator | N=50 | N=150 | N=300 | N=448 |
|---|---|---|---|---|
| total variation | 99.0% | **100%** | 100% | 100% |
| Jensen-Shannon | 96.0% | 100% | 100% | 100% |
| **plug-in KL** | 14.3% | 9.3% | 9.0% | **5.3%** |

**Plug-in KL degrades as proteins are added**, ending below the 4.2% a four-point monotone
sequence reaches by chance. Its bias varies with each protein's sparsity, so averaging
concentrates it. Over these references KL is anti-informative.

**Consequence:** M1 must be restated as a population trend over at least 150 proteins using a
bounded divergence, never as a per-protein number. Pre-registered in `m1-protocol.md`.

### B2. There are no public per-cell localization labels

HPA's labels are per-antibody and are broadcast identically to every cell in a field of view —
verified by reading a crop manifest, where all eight cells carry the same string. The only
genuine per-cell annotations, 41,597 expert-labelled cells from the 2021 Kaggle competition,
appear never to have been released; the authors' scoring script reads them from a local path.

**Consequence:** `pi_emp` is an estimate, not a measurement. Probe A exists because of this.

### B3. Pooling across cell lines is what makes the study possible

Per (protein, cell line) the median is 18 cells, which is unusable. Pooling raises it to 75.
The cost is that no claim can be conditioned on cell line.

---

## C. Findings about the instrument

### C1. Sampling and inversion want opposite terminal SNR

Sampling from `N(0, I)` is only correct when `alpha_bar_T` reaches zero; otherwise the sample
scale is biased (measured 1.83 against a target of 2.0). Inversion divides by
`sqrt(alpha_bar)` on its final step, so as `alpha_bar_T` approaches zero it amplifies model
error — a 1% error became a 38% round trip, and the final step alone contributed 93% of total
error. Flooring `sqrt(alpha_bar_T)` at 1e-2 satisfies both. No paper appears to address the
interaction.

### C2. BDIA beats EDICT on every axis measured

| | evaluations per step | round trip | at 200 steps |
|---|---|---|---|
| EDICT | 2 | 3.5e-16 | degrades to 3.2e-12 |
| **BDIA** | **1** | **3.8e-17** | **holds at 2.3e-17** |

Given both initial states the BDIA recurrence is exact to 7e-18, and it *attenuates* an error
in its bootstrap by about 28x rather than amplifying it — the parasitic root leapfrog schemes
classically carry is not a practical problem here. Neither point appears in the paper.

### C3. A perfect round trip does not certify the latent

Below its safe parameter range BDIA still reconstructs at machine precision while returning a
latent with standard deviation 12.9, and 1e11 further down. Round-trip error alone must never
be the acceptance criterion.

### C4. M2's estimator needs common random numbers

`Delta` is a difference of two nearly equal expectations. Drawing the timestep and the noise
once and evaluating both conditions on them cuts the standard deviation 6.3x, a 40x variance
reduction. Without it the signal is buried.

---

## D. Things that would have silently corrupted everything

Each of these produces plausible output rather than an error.

### D1. SubCell's input convention differs from the diffusion model's

SubCell expects [0, 1] and channels ordered (microtubules, ER, nucleus, protein). Fed [-1, 1]
in on-disk order, it assigned four proteins with completely different annotations **the same
class at 0.8 confidence**, with a largest pairwise distance of 0.074 against 0.725 once
corrected. The first probe A run produced a clean, wrong, publishable-looking null.

Caught by a positive control, not by anything failing. See `subcell-preprocessing.md`.

### D2. Channel order on disk

(nucleus, ER, microtubules, protein). Established by rendering the channels and looking at
them, after two rounds of statistical inference gave the wrong answer. Pinned by tests against
four committed fixtures; correlation of the protein channel with the nucleus channel runs
+0.738 for a nucleoplasm protein, +0.343 for Golgi, −0.190 for plasma membrane.

### D3. The SubCell checkpoints do not load into current transformers

Parameter names changed upstream. A direct load leaves 192 tensors uninitialized **and raises
nothing**. The remapping is in `eval/subcell.py`.

### D4. Storage estimated from one plate is wrong by 2.8x

4.27 MB per cell measured across 2400 cells and 40 genes; a single plate suggested 1.54. The
full probe-A selection is 44,831 cells and **187 GB**, not 51.

---

### D5. Module M3 fails its kill gate

Spot detection at an honestly calibrated threshold does not distinguish vesicular from
cytosolic compartments — AUC 0.438 on density, 0.271 on clustering. The encouraging clustering
number from an earlier pass (0.729) was measured at a threshold where **89% of detections
survived phase scrambling**, i.e. it was measuring texture. Full record in `m3-gate.md`.

Cost: about four CPU hours, no GPU, no training run. M3 is out of the main line.

---

### D6. Module M2's estimator recovers the right structure but its noise floor is half the signal

On synthetic data where each compartment's landmark dependence is known by construction, the
`Delta_k[compartment]` matrix comes out exactly as designed — DNA feeding nuclear compartments,
microtubules and ER feeding the cell boundary. But the built-in null, vesicles, which depend on
no landmark, reads **-0.0009 to -0.0007 and is flagged as more than two standard errors from
zero**. A negative information is meaningless, so that is the noise floor, and the real signal
is only 0.002 to 0.004.

Quadrupling the evaluation batches changed nothing (0.000867 to 0.000995), so it is not sampling
noise. It is structural: for a channel that carries no information, the masked input — a single
learned constant plane — has lower variance than the real one, and a model that is not perfectly
invariant does slightly better on it. **`Delta_k` is not lower-bounded at zero**, so testing it
against zero is testing the wrong null. Fixed by reporting every delta against an empirical
uninformative-channel baseline, decided before any real-data number existed. Full record in
`m2-rehearsal.md`.

---

## E. Open

- Expanded probe A over 116 genes, in progress.
- Full-scale probe A over 497 genes, needs the 187 GB download.
- Synthetic M1 and M2 rehearsals — written, too slow on two cloud cores.
- Phase 1 training — needs real GPUs.
- The Kaggle per-cell labels — needs an email; draft in `outbox/`.
