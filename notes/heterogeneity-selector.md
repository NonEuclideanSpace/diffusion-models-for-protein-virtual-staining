# The annotation is not the selector

**Status:** the discovery-set effect did not replicate. The quantity it was measuring is real;
the label used to select for it is not predictive. This note records the retraction and the
replacement design.

---

## 1. What was claimed

Probe A's discovery run (40 genes, 2400 cells) reported that genes HPA flags with
single-cell variation carry more between-cell localization heterogeneity than matched
controls: effect 0.693 [0.517, 0.845], z +2.08. `findings.md` A1 already downgraded this to
"inconclusive" after the cell-line confound was found. The expanded run settles it.

## 2. What replicated

The expanded run shares 36 genes with the discovery set. The 51 genes it does not share are a
genuine independent replication and were analysed first.

| set | genes | effect | 95% CI | verdict |
|---|---|---|---|---|
| discovery | 20 v 20 | 0.693 | [0.517, 0.845] | separates |
| **independent replication** | 37 v 40 | **0.433** | [0.303, 0.564] | null, direction reversed |
| expanded, all | 56 v 60 | 0.508 | [0.401, 0.615] | null, on the line |
| expanded, round lines removed | 50 v 50 | 0.451 | [0.337, 0.566] | null |

Every estimate outside the discovery set sits at or below 0.5: flagged genes were marginally
*less* heterogeneous than controls. The groups are balanced (56 v 60, and 50 v 50 after the
morphology filter) and the cell-line composition distance between them fell from 0.650 in the
discovery set to **0.232**, so the matching worked and the null is not a confound artefact.
The discovery result was a discovery artefact and is retracted.

## 3. Why this is a real null and not underpowering

The obvious escape is that 40 cells per gene is too few to measure heterogeneity. It is not.

- **Split-half reliability, 200 random splits within each of 90 genes: 1.000.** Two disjoint
  halves of the same 40 cells rank genes identically.
- **Independent-sample reliability, 39 genes measured at 60 and at 40 cells: 0.923.** This is
  the honest number — different cells, sometimes different plates — and it is still high.

Per-gene heterogeneity is a reproducible property of the gene. The measurement is not noise.
It simply does not track the annotation.

## 4. What the measurement does track

116 genes, heterogeneity as `H(mean prediction) − mean(per-cell entropy)`:

    mean 0.0847  sd 0.0644  p50 0.0696  p90 0.1552  max 0.4006
    skew +2.09   kurtosis 9.40           (gaussian is 3.0)

A heavy tail — the top gene is 5.8 times the median. Variance in that spread, explained:

| factor | R² |
|---|---|
| dominant cell line (20 lines) | **0.236** |
| predicted compartment (13) | 0.101 |
| annotated compartment (19) | 0.086 |
| **HPA variable annotation** | **0.004** |

Cell line explains **sixty times** what the annotation does. The top quintile is 48% flagged
against a 48% base rate — **enrichment 0.99x**. The annotation is exactly as informative about
which genes are heterogeneous as a coin flip.

## 5. The tail is not a classifier artefact

Three attempts to make it go away, all failed:

1. **Head-free.** Rank correlation between the classifier-based information and raw embedding
   spread is **+0.591** over 90 genes. The 1536-d representation sees the same genes as
   heterogeneous.
2. **Coarse grouping.** Collapsing 31 classes to 8 groups — so nucleoplasm/nucleoli boundary
   wobble cannot count — retains 65% of the mean at a **+0.918** rank correlation.
3. **Genuine minority states.** 9 genes have a top-quintile score *and* under 70% of cells
   agreeing with the modal call. These are two-population genes:

       INCENP    Nucleoplasm 23   Cytosol 17
       FAF2      Endoplasmic reticulum 24   Vesicles 15
       DECR1     Plasma membrane 18   Cytosol 10   Mitochondria 8
       IRS1      Nucleoplasm 22   Cytosol 16
       GJB2      Negative 26   Mitochondria 14

   **Seven of the nine are annotation controls.** INCENP is the clearest case: the inner
   centromere protein relocates from centromere to spindle midzone during mitosis, which is
   textbook two-state localization, and HPA has it as a control.

## 5b. Which annotation, exactly — the columns are not interchangeable

HPA carries three separate single-cell columns and the retraction above is only meaningful if
it is about the right one.

| column | genes in HPA | in the 116 | meaning |
|---|---|---|---|
| Single-cell variation **spatial** | 186 | 56 | the protein sits in different *places* between cells |
| Single-cell variation **intensity** | 3,650 | 44 | the protein is present at different *levels* |
| **Cell cycle dependency** | 479 | 12 | localization tracks the cell cycle |

Only spatial is the construct measured here. Each tested on its own, against everything it does
not mark:

| column | n | effect | 95% CI | verdict |
|---|---|---|---|---|
| **spatial** | 56 v 60 | **0.508** | [0.401, 0.615] | null |
| intensity | 44 v 72 | 0.487 | [0.383, 0.598] | null |
| cell cycle | 12 v 104 | 0.578 | [0.422, 0.728] | suggestive, underpowered |

**The null is about spatial variation specifically**, which is the strongest form the
retraction can take: the correct column, tested directly, separates nothing. Median rank of a
spatial-marked gene is 55 of 116 — dead centre.

Cell cycle dependency is the one column pointing the right way, but twelve genes cannot settle
it. It is worth testing properly, and section 6's screen now does: all 95 U2OS genes carrying a
spatial or cell-cycle mark are taken whole rather than sampled, giving 28 spatial and 68
cell-cycle out of 600.

One thing this changed immediately: `build_u2os_plan.py` originally OR-ed spatial with
intensity, exactly the mistake this section is about. Because intensity is twenty times more
common, the flagged group would have been 96% intensity genes. Each flag is now recorded
separately in the plan (`S`, `I`, `C`) and the comparison is made at analysis time.

## 5c. Looking at the images raised an alarm, and the test half-confirmed it

Rendering the top of the U2OS tail (`outputs/figures/u2os_tail.png`) showed the same shape in
all four genes: one population with a clear protein signal and one with almost none.

    NR3C1   Nucleoplasm 19 / Negative 18     glucocorticoid receptor
    CDK1    Negative 21 / Nucleoplasm 18     cell-cycle master kinase
    ARF6    Negative 19 / Cytosol 18         vesicle trafficking
    NEK2    Negative 24 / Nucleoplasm 16     centrosome kinase

SubCell's `Negative` class means *no detectable protein*. A gene expressed in half its cells and
silent in the other half produces a large `H(mean p) - mean H(p_i)` **without ever changing
compartment** — that is abundance variation, HPA's `intensity` column, not the `spatial` one
this project is about. If the tail were built out of it, the whole selector would be measuring
the wrong thing.

### The measurement

Across 280 U2OS genes, `P(Negative)` correlates with measured heterogeneity at **pearson +0.587,
spearman +0.447**. The median gene puts 0.007 of its mass there; the top of the tail puts
0.13-0.44. So the alarm was real in magnitude.

Recomputing heterogeneity with the Negative class dropped and the remaining mass renormalised,
on 53 genes with per-cell data:

| | mean | median |
|---|---|---|
| with Negative | 0.2069 | 0.0978 |
| without Negative | 0.1656 | 0.0978 |

- **80% of the mean survives**, and the median is unchanged — for most genes Negative is
  irrelevant and the action is entirely in the tail.
- **Rank correlation between the two statistics: +0.902.**
- **All 15 of the top 15 stay above the median** once Negative is removed.

**The ranking is not driven by detection dropout.** Magnitude is inflated in the tail; order is
not. The selector survives.

### What it does force: a decision that has to be made now

What `Negative` *means* is not resolvable from these images. For CDC20 it is real biology and
was validated independently — same-sized nuclei, 1.5x the chromatin density, so those cells are
in G1 and the protein has genuinely been degraded (`findings.md` A3). For NR3C1, a receptor that
is always expressed, a Negative call more likely means *present but below detection* than
*absent*.

These are different things and M1 treats them identically. A cell whose protein is undetectable
has no defined position in a distribution over compartments.

**Fixed here, before any model output exists:** the empirical distribution `pi_emp` is computed
over the 7 compartment groups **plus Other**, as pre-registered, and `Negative` is folded into
`Other` rather than dropped — probability mass is conserved and the class stays visible. Genes
whose modal call is `Negative` in more than half their cells are **excluded from M1's target
set**, because for those the conditional over compartments is not identified. On the current
280 genes that excludes none, and on the full screen it should exclude very few.

That rule is recorded now so it cannot be chosen later to suit a result.

## 5d. The definitive test: 588 U2OS genes, zero cell-line confound

The screen finished. 588 genes, one cell line, so the factor that explained 23.6% of the
variance in the mixed-line probe is eliminated by construction rather than controlled for.

| column | marked | AUC | z | R² |
|---|---|---|---|---|
| **spatial variation** | 28 v 560 | **0.598** | +1.76 | **0.010** |
| intensity variation | 225 v 363 | 0.553 | +2.15 | 0.006 |
| cell cycle dependency | 68 v 520 | 0.532 | +0.87 | 0.000 |
| dominant cell line | — | — | — | 0.000 (one line) |

Top-quintile enrichment: **spatial 1.62x**, intensity 1.09x, cell cycle 0.74x.

**The retraction stands and this is its final form.** Spatial is the best of the three and the
only one pointing anywhere: AUC 0.598, enrichment 1.62x, and it moved in the right direction
from the mixed-line estimate of 0.508. But z is +1.76, which does not clear 0.05 two-sided, and
it accounts for **1% of the between-gene variance**.

Intensity's z of +2.15 is not evidence of anything — with 225 marked genes, an AUC of 0.553
reaches z > 2 while explaining 0.6% of the variance. Reporting it as significant would be a
sample-size artefact, and it is recorded here so nobody later mistakes it for a result.

Cell cycle dependency is at 0.74x enrichment: the marked genes are *less* likely to be in the
heterogeneity tail than chance.

### What survives at 588 genes

    heterogeneity   mean 0.1141  median 0.0881  max 0.5104
    skew +1.47   kurtosis 5.31        (gaussian is 3.0)
    split-half reliability 1.000

Heavy-tailed, reliably measured, and unpredicted by any of the three annotation columns. The
quantity is real; the labels are not the way to find it. **A data-driven ranking over 588 U2OS
genes is the deliverable, and it now exists.**

## 5e. Why the cell-cycle column points the wrong way

Cell cycle dependency came out at **0.74x enrichment** — genes it marks are *less* likely to be
in the heterogeneity tail than chance. That is strange next to CDK1, CDC20 and NEK2 all sitting
near the top, so the column's contents were checked:

    158  Cytokinetic bridge (biological definition)
     79  Mitotic chromosome (biological definition)
     78  Cytokinetic bridge;Mitotic spindle
     77  Mitotic spindle (biological definition)
     44  Midbody (biological definition)
     23  Midbody ring (biological definition)

**It is a location annotation for mitotic cells, not a variability annotation.** It says *which
mitotic structure this protein sits in*, not *this protein's position changes with the cycle*.

Mitotic cells are a few percent of an asynchronous population, so a protein annotated
"cytokinetic bridge" looks uniform in the ~95% of cells that are not dividing. The column marks
proteins that are **uniform except in a rare state** — close to the opposite of what the name
suggests, and exactly why the enrichment is below one.

This is the third time on this project that an HPA column has not meant what its name implies,
after `Single-cell variation spatial` against `intensity` (section 5b) and the `Negative` class
(section 5c). None of them is HPA's fault; all three are cases of a name being read instead of a
definition.

## 5f. The deliverable

`outputs/u2os_heterogeneity_ranking.tsv` — 588 U2OS genes ranked by measured heterogeneity, with
per-gene cell count, predicted compartment and confidence, coarse-group entropy, the probability
mass on `Negative`, all three HPA flags, and the HPA location string.

    rank  gene      heterogeneity  predicted     hpa_location
       1  NR3C1            0.5104  Negative      Cytosol;Nucleoplasm
       2  ENTREP2          0.4814  Negative      Nucleoplasm
       3  ARF6             0.4778  Negative      Cytosol
       4  CDK1             0.4482  Negative      Cytosol;Nucleoplasm
       5  ZNF560           0.4055  Nucleoplasm   Nucleoplasm

    median 0.0874   p90 0.2321   max 0.5105

Anyone selecting proteins whose localization genuinely varies between cells currently has no
way to do it from annotation. This is that list, and the three sections above are why it had to
be measured.

    uv run python experiments/publish_ranking.py

## 6. What this does to M1

M1's premise survives and its selector does not.

The premise needs a ground-truth conditional spread for a diffusion model to fail to
reproduce. Section 4 and 5 establish that spread exists, is heavy-tailed, is reliably
measurable, and is visible without the classifier head. That is the thing M1 is about.

What fails is selecting the target genes by HPA annotation. `m1-protocol.md` pre-registered
that selector; it is replaced, and the replacement is fixed here before any model output has
been looked at.

**Replacement design — the U2OS screen.** Rank genes by *measured* heterogeneity within a
single cell line, and take the tail.

- Holding cell line fixed removes the 0.235 term by construction, which is the largest
  identified confound.
- U2OS has 295,893 indexed cells over 10,921 genes; **1,564 genes have >=40 cells and an
  Enhanced/Supported/Approved annotation**. Data is not the constraint.
- 600 genes x 40 cells = 24,000 cells, streamed, ~6.7 h on 2 cores. No download to the user's
  disk is required — the streaming runner deletes each crop after embedding it.
- The screen also re-tests the annotation null at 6.7x the sample size with the cell-line term
  eliminated, so section 2's retraction gets a decisive confirmation for free.

Plan: `/home/claude/data/subset/u2os.txt`, built by `experiments/build_u2os_plan.py`
(227 flagged, 373 control). Runner: `experiments/supervise.sh`, which restarts on reap.

## 7. What this cost and what it saved

Cost: two probe runs, 8,240 cells of inference, about four hours.

Saved: the pre-registered M1 protocol would have trained a conditional diffusion model on a
gene set selected by a label that explains 2% of the variance in the quantity being measured.
The failure would have appeared after the training run, not before it, and would have been
indistinguishable from the model being fine.

## 8. Reproduce

    uv run python experiments/probe_a_combined.py        # section 2
    uv run python experiments/heterogeneity_tail.py      # sections 3, 4
    uv run python experiments/tail_checks.py             # section 5
    uv run python experiments/build_u2os_plan.py         # section 6
