# Probe A: preliminary result

**Verdict: inconclusive. The apparent effect is substantially confounded by cell line.**

An earlier version of this file said "passes". That was wrong, and the correction is the most
important thing in it. See "The confound" below.

## Question

Module M1 needs a per-protein distribution over compartments across individual cells. No
public per-cell labels exist, so that distribution has to come from SubCell — whose classifier
heads were trained on labels broadcast identically across each field of view. If that training
regressed cells toward the consensus, the predicted distributions carry no within-protein
information and M1 has no target.

HPA independently flags 186 genes as varying spatially at the single-cell level. That flag is
curated by annotators and owes nothing to SubCell. So: does SubCell see more heterogeneity in
flagged genes than in controls matched on both reliability and main location?

Matching on main location is what makes the test mean anything. Without it, a positive result
would be consistent with nothing more than "multi-located proteins get higher entropy", which
is true by construction. With it, the two groups have nearly identical multi-localization
rates (55.4% against 53.4%).

## Setup

20 flagged genes and 20 matched controls, 60 real HPA cells each, 2400 cells total. SubCell
ViT-ProtS-Pool encoder with all ten released classifier heads, averaged in probability space.

## Result

| statistic | flagged | control | effect size [95% bootstrap] | z |
|---|---|---|---|---|
| **between-cell heterogeneity** | 0.0995 | 0.0660 | **0.693 [0.525, 0.848]** | **+2.08** |
| entropy of the mean prediction | 2.3127 | 2.4627 | 0.450 [0.265, 0.630] | −0.54 |
| **embedding spread** | 2.7787 | 2.4315 | **0.725 [0.555, 0.870]** | **+2.43** |

Effect size is the probability that a randomly chosen flagged gene exceeds a randomly chosen
control; 0.5 is no separation.

Six of the seven most heterogeneous genes are flagged: ZNF560 0.416, TERT 0.271, CENPC 0.196,
PMS1 0.187, RYK 0.169, RBL2 0.161.

## An independent check: does SubCell agree with HPA at all

Before asking whether the classifier resolves *within*-protein variation, it is worth asking
whether it gets the protein right. Collapsing both SubCell's prediction and HPA's `Main
location` onto the same seven groups:

| | |
|---|---|
| top-1 group matches an annotated location | **82.5%** (33 of 40) |
| annotated location in the top two groups | **95.0%** |
| chance, weighting by annotation frequency | 22.9% |

Flagged genes 90%, controls 75%.

The seven disagreements are informative rather than alarming. Three involve lipid droplets, a
rare class; most of the rest are low confidence (0.21 to 0.37). Only PNPLA2 is confidently
wrong, at 0.83 for Nucleus against an annotation of Lipid droplets. GAPDH predicted as Nucleus
against `Cytosol;Plasma membrane` is defensible — GAPDH is a textbook moonlighting protein
with a real nuclear role.

So the pipeline works and `pi_emp` is meaningful before any question about heterogeneity is
asked. That matters because the heterogeneity signal is small, and a small signal from a
classifier that could not even name the compartment would not be worth interpreting.

## The statistic matters more than the result

**Plain entropy gives the wrong answer.** It is the obvious choice, it separates nothing
(0.450), and its point estimate points the wrong way. Using it, probe A fails and M1 dies.

Entropy of a protein's mean prediction is high when cells genuinely differ *or* when the
classifier is uniformly uncertain about every cell. Only the decomposition separates them:

    I(cell ; compartment) = H(mean prediction) − mean(per-cell entropy)

That gap is the between-cell variability with each cell's own uncertainty removed, and it is
what M1 actually needs. Both quantities were computed from the start; the gap is the primary
statistic on theoretical grounds, and naming it as primary in advance would have been better
practice than arriving at it after seeing that entropy did not work.

## The embedding result matters most

Embedding spread separates the groups more strongly than the classifier-derived statistic
does, at 0.725 against 0.693, and it never passes through a classifier head. The heads are
where the circularity lives — they were trained on broadcast labels. The encoder was trained
with a protein-level objective, not a per-cell one, so its representation seeing more spread
in genes that annotators independently flagged is evidence the heterogeneity is real rather
than an artefact of the label pipeline.

This is the strongest single piece of evidence that `pi_emp` is worth estimating at all.

## The confound

Controls were matched on reliability and main location. **They were not matched on cell line**,
and the two groups turn out to differ sharply in which lines their cells come from — a total
variation distance of 0.580 between the compositions.

The direction of that mismatch is exactly wrong. Suspension and round-morphology lines appear
almost only among controls: HEL 15% of control cells against 0% of flagged, REH 10% against 0%,
SK-MEL-30 10% against 0%. Cells of those lines are round and morphologically uniform, so they
should show less apparent heterogeneity whatever protein is imaged.

They do:

| genes dominated by | n | median heterogeneity | median embedding spread |
|---|---|---|---|
| suspension or round lines | 7 | **0.0179** | **1.548** |
| everything else | 33 | **0.0978** | **2.758** |

A 5.5x difference, driven by cell line rather than by protein, and every one of those seven
genes is a control.

**Removing them removes the result.** Dropping the nine genes dominated by HEL, REH,
SK-MEL-30, THP-1 or Rh30 leaves 19 flagged and 12 control:

| statistic | all 40 genes | cell-line-confounded genes removed |
|---|---|---|
| heterogeneity | 0.693 [0.517, 0.845] — separates | **0.601 [0.395, 0.794] — not resolved** |
| embedding spread | 0.725 [0.560, 0.873] — separates | **0.645 [0.421, 0.842] — not resolved** |

Both point estimates still lean the right way, and neither interval excludes 0.5 any more.

**What this means.** The honest reading is that the effect direction survives and its
significance does not. Twelve control genes cannot resolve an effect of this size. The result
is not refuted; it is unsupported at this sample size once the confound is controlled.

**Where the imbalance came from.** Not from the matching logic. At full scale the same matcher
gives a composition distance of 0.099 between 175 flagged genes and 311 controls, with
suspension lines at 6.3% and 8.7%. The 0.580 belongs to the 40-gene development subset alone,
which was built by ranking genes on cell count — and because imaging volume varies by line,
that ranking pushed the suspension-line genes into the control group.

So the lesson is narrower and sharper than "the matcher was broken": **a subset drawn by
ranking on any quantity correlated with cell line will reintroduce the confound even from a
balanced parent selection.** Composition distance has to be checked on the set actually used,
not on the set it was drawn from.

**What is fixed.** `matched_controls` now accepts a cell-line map and matches on it, bringing
the full-scale distance to 0.083 and equalizing the suspension fraction at 6.3% against 6.4%.
`composition_distance` exists so any future selection can be checked in one line. The expanded
116-gene run underway sits at 0.246 — better than the subset, worse than full scale — and its
reading should be treated accordingly.

## Caveats, all of which matter

**Twenty genes per group.** The bootstrap interval on the primary statistic is [0.525, 0.848]
and its lower bound barely clears 0.5. This is a preliminary result, not a confirmed one.

**Forty of 497 available genes.** The full index supports 175 flagged and 322 matched control
genes with at least 50 cells each. Re-running there is the confirmation, and it costs nothing
but the download and a few GPU-minutes.

**Sixty cells per gene**, against a median of 75 available.

**The magnitude is small.** Median heterogeneity is 0.074 nats where a uniform distribution
over 31 classes would give 3.434. Cells of the same protein receive similar predictions. That
is consistent with real biology — most proteins do have one dominant localization — but it
also means the quantity M1 measures is small and the model's sharpening of it will be smaller
still. This tightens rather than loosens the sample-size requirements in `m1-protocol.md`.

## What this unblocks, and what it does not

**The classifier is validated**: 82.5% top-1 agreement with HPA annotation against 22.9%
chance does not depend on the flagged-versus-control comparison at all, and is not affected by
the confound. SubCell names the right compartment.

**Whether it resolves within-protein variation is unsettled.** That is what probe A was for,
and after controlling for cell line the sample is too small to say. M1 cannot proceed on this
evidence.

**The next run decides it.** 116 genes, matched on cell line, is the test. If the effect
survives there it is real; if it does not, `pi_emp` carries no usable within-protein signal and
M1 has no target, exactly as `data-availability.md` warned.

## Reproducing

    ./run_local.sh weights
    DRIVE=/Volumes/<drive>/hpa-subset ./run_local.sh subset
    DRIVE=/Volumes/<drive>/hpa-subset ./run_local.sh probe
    uv run python experiments/probe_a_report.py

Raw per-gene output is in `probe-a-results.json`.
