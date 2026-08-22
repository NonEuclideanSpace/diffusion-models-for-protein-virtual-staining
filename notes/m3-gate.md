# Module M3's kill gate

`INTERNAL.md` section 6: *"Spot detection must be validated on **real** images before touching
generated ones. If the detector is unreliable at this image quality, the module measures
detection noise and is dead."*

M3 had no code at all before this. `src/pvs/eval/pointprocess.py` is now written and tested
(16 tests): multi-scale Laplacian-of-Gaussian detection with non-maximum suppression across
both space and scale, nearest-neighbour distances, and Ripley's K with the isotropic edge
correction — the per-pair weight estimated by sampling each circle against the cell mask,
because cells are arbitrary shapes and the rectangular-window formulas do not apply. Pure
torch, so it runs on the same device as the model and adds no dependency.

The decisive test is `test_ripley_l_separates_clustered_random_and_regular`: the statistic has
to tell a clustered point pattern from a Poisson one from a lattice, or it cannot tell anything.

---

## The first run failed, and the failure was mine

| gate | 1024px | verdict |
|---|---|---|
| 1. spot density separates punctate from diffuse annotations | AUC **0.301** | dead, and backwards |
| 2. one gene's density reproducible across disjoint halves | r 0.974 | usable |
| 3. survives the resolution a model generates at (256px) | r 0.966 | usable |

Backwards is the interesting part. Checking the group membership explained it: 7 of my 14
"punctate" genes were **nuclear speckles, nuclear bodies or centrosomes**. Those are punctate,
but they sit inside the nucleus — roughly a tenth of the cell mask — and density was normalised
by *whole-cell* area, so they scored low by construction. A test design error, not a detector
failure.

## The corrected run: density fails, clustering appears to work — it does not (see below)

Vesicles and peroxisomes against cytosol, which occupies the same territory:

| gate | 1024px | 256px | 128px |
|---|---|---|---|
| 1. spot **density** | AUC 0.271 **dead** | 0.319 dead | 0.347 dead |
| 1b. **Ripley's L**, no area normalisation | AUC **0.729** weak | 0.722 weak | 0.708 weak |
| 2. split-half reproducibility | r **0.974** | 0.964 | 0.941 |
| 3. rank agreement against 1024px | — | r **0.907** | r 0.789 |

Density fails in the same direction at every resolution, and the reason is mechanical: a
cytosolic protein fills the cell with granular texture and the detector fires on all of it,
while a vesicular protein gives fewer, brighter, better-separated puncta. **Counting maxima
rewards texture.** Ripley's L asks how the detections are arranged rather than how many there
are, and it points the right way at every resolution — weakly, at 12 genes per group.

Gate 3 matters more than it looks: 256px keeps the ranking (r 0.907) and 128px starts to lose
it (0.789). A model generating at 128px would not support this module. 256px does.

## The threshold calibration, and what it did to the clustering result

Threshold fixed by the pre-registered label-free rule: smallest value whose phase-scrambled
false-positive share is under 5%. 24 genes, 10 cells each, 256px.

| threshold | real density | scrambled | **FP share** | density AUC |
|---|---|---|---|---|
| 0.02 | 274.6 | 235.7 | **86%** | 0.347 |
| **0.04** (the first run) | 239.7 | 213.6 | **89%** | 0.361 |
| 0.08 | 189.4 | 79.7 | 42% | 0.347 |
| 0.15 | 83.9 | 7.2 | 9% | 0.375 |
| **0.30** | 5.0 | 0.0 | **0%** | 0.472 | <- chosen by the rule |
| 0.60 | 0.0 | 0.0 | 0% | *degenerate* |

At the threshold the first two gate runs used, **89% of every detection survived phase
scrambling** — the detector was almost entirely reporting texture. That explains the backwards
density result completely, and it invalidates the one encouraging number in the table above:
the Ripley's L AUC of 0.729 was measuring how *texture* is arranged, not how vesicles are.

The AUC of 1.000 at thresholds 0.60 and 1.20 is not a result. Both groups detect zero spots
there; the value comes from ties in degenerate data. Worth stating because a sweep that reports
it without comment is how a project talks itself into a dead module.

Re-running the gate at the calibrated 0.30:

| gate | 256px | verdict |
|---|---|---|
| 1. spot density | AUC 0.438 | **dead** |
| 1b. Ripley's L | AUC **0.271** | **dead, and now reversed** |
| 2. split-half reproducibility | r 0.780 | usable, down from 0.974 |

## Verdict: M3 fails its kill gate

`INTERNAL.md` section 6 states the consequence: *"If the detector is unreliable at this image
quality, the module measures detection noise and is dead."* At an honestly calibrated threshold
it does not separate vesicular from cytosolic compartments on either statistic, and the one
statistic that appeared to work was reading texture.

**Recorded as a negative result. M3 is out of the main line.**

### Why it failed, as a hypothesis for anyone reviving it

Not a reason to keep tuning now — the gate has been called and re-tuning after a failure is
what the pre-registration forbids. But the diagnosis is specific and worth writing down.

At 0.30 the vesicular genes give a median density of 2.36 and the cytosolic ones 15.31: the
supposedly punctate group detects *fewer* spots than the diffuse group. `normalize()` divides
every channel by the microtubule channel's 99.5th percentile, so a dim protein stays dim in
absolute terms. A vesicular protein with a handful of bright puncta and no background can end
up with a lower absolute LoG response than a bright cytosolic protein's texture.

**An absolute LoG threshold is confounded with expression level.** A revival would need a
per-cell relative criterion — a quantile of the within-cell LoG response, or peak prominence
against local background — and it would need its own label-free calibration before any gate is
re-run.

### What this cost

About four CPU-hours, no GPU, no training run. That is the entire point of a kill gate: M3 was
one of three modules the project could have spent Phase 4 on.

## Original verdict, superseded

The text below was written before the threshold calibration and is kept only to show what
the module looked like one step earlier.




**Not dead, but not passed either.** The detector is reliable; the statistic was wrong. Before
M3 can run for real, two things have to be fixed *in advance*:

1. **The primary statistic is Ripley's L, not density.** Fixed here.
2. **The detection threshold**, which was an arbitrary 0.04 and is what lets texture in.

Point 2 is a trap. Sweeping the threshold and keeping whichever value makes the gate pass is
exactly the forking path that produced this project's first retracted result. So the threshold
is calibrated against a null that never sees the labels: **phase scrambling**, which replaces
each cell's Fourier phases with random ones, preserving the whole power spectrum — all the
texture statistics — while destroying every localised structure. Anything the detector still
finds there is a false positive by construction.

**The rule, fixed before the numbers were looked at:** the threshold is the smallest value at
which the scrambled false-positive share falls under 5%. `experiments/m3_threshold.py`.

## What is still open, if anyone revives it

- A per-cell relative detection criterion, with its own label-free calibration.
- More genes per group; 12 v 12 cannot resolve an AUC near 0.5 either way.
- Nuclear punctate structures — speckles, bodies, centrosomes — need the nuclear mask as their
  window rather than the cell mask. Out of scope for M3 as specified, but the first run here
  shows what happens if that is forgotten.
