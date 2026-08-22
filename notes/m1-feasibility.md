# M1 feasibility: what the numbers say before any model is trained

Two independent problems threaten module M1. Neither is about compute. One is about labels
and might be solved by asking. The other is about sample size and cannot be.

## Problem 1 — no per-cell ground truth

Covered in `data-availability.md`. HPA labels are per-antibody and broadcast to every cell in
a field of view. `pi_emp` has to be predicted by SubCell, whose classifier heads were trained
on those same broadcast labels.

## Problem 2 — the estimator inverts sign at HPA's sample sizes

Measured from 25 sampled SubCell crop manifests (10,694 cells):

| quantity | value |
|---|---|
| U2OS share of the corpus | 27.1% |
| distinct cell lines | 14 in sample, 39 reported corpus-wide |
| **cells per (gene, cell line) in U2OS** | **median 18, mean 20, max 92** |
| bytes per cell, measured on plate 1 | 1.54 MB (image + mask) |

Then simulate the plug-in KL estimator at those sizes. Under the null, where the model is
perfectly calibrated and all divergence is sampling noise:

| K | n_real | n_gen | median D_cal | 90th pct |
|---|---|---|---|---|
| 5 | 20 | 100 | 0.115 | 0.313 |
| 5 | 200 | 100 | 0.025 | 0.060 |
| 10 | 20 | 100 | **1.841** | 3.916 |
| 10 | 200 | 100 | 0.064 | 0.113 |

For comparison, the signal from a model that moves 40% of the mass onto the dominant mode is
0.31 at K=10. **At n_real=20 the noise floor is six times the signal.**

Worse, the failure is not loss of power. Averaging over 300 proteins at K=10, n_real=20:

| guidance | measured mean D_cal |
|---|---|
| w = 0.0 | 2.026 |
| w = 0.2 | 1.771 |
| w = 0.4 | 1.632 |
| w = 0.6 | 1.584 |

**D_cal falls as the model collapses modes — the sign is inverted.** With 20 cells spread over
10 bins the empirical estimate is itself sparse, so a concentrated generated distribution
matches it better than the true spread-out one does. A mode-collapsing model would be scored
as better calibrated, and the study would report a confident, wrong result.

## Detection rate of the monotone trend, by configuration

Fraction of trials where population-mean D_cal is strictly increasing in guidance weight,
which is the hypothesis `INTERNAL.md` section 4 actually states:

| K | n_real | 30 proteins | 100 proteins | 300 proteins |
|---|---|---|---|---|
| 5 | 20 | 8.8% | 2.2% | 0.2% |
| 5 | 50 | 78.5% | 89.5% | **98.5%** |
| 10 | 20 | 0.0% | 0.0% | 0.0% |
| 10 | 50 | 41.0% | 39.0% | 36.8% |

Note that at K=5, n_real=20, **more proteins makes it worse** — averaging concentrates a
biased estimator onto its bias rather than averaging noise away.

## Requirements this imposes

1. **K <= 5.** Collapse the 35 HPA annotations to a coarse grouping.
   `data/subcell/location_group_mapping.tsv` already carries four hierarchical groupings.
2. **n_real >= 50 cells per conditioning group.** HPA gives a median of 18 per
   (protein, cell line), so the per-cell-line conditioning has to be dropped and cells pooled
   per protein across lines (roughly 95 per protein corpus-wide), or the study restricted to
   the high-count tail, or both.
3. **N >= 300 proteins**, and the statistic must be the trend across guidance weights, not any
   single absolute D_cal.
4. **Pre-register the estimator against synthetic data.** The table above is that validation.
   Re-run it with the final K, the final n distribution and the final estimator before
   touching real images.

## Estimators worth trying instead of plug-in KL

Plug-in KL is the worst choice at these sizes: unbounded, positively biased, and its bias
depends on the sparsity of both arguments, which is what produced the sign inversion.

- **Jensen-Shannon divergence** — bounded by log 2, symmetric, far better behaved when either
  argument has empty bins.
- **Total variation** — no log, so no exploding term from empty bins, and directly
  interpretable as fraction of misplaced mass.
- **Miller-Madow or NSB bias correction** on the entropy terms.
- **Paired permutation test on the trend**, using the protein as its own control, which
  removes the shared `pi_emp` noise by construction rather than by averaging.

Whichever is chosen, the null-distribution table has to be recomputed for it. That is cheap
and requires no GPU.

## The decisive calculation, at the parameters the data actually provides

The numbers above used K=10 and 20 cells per gene, which is what conditioning on cell line
gives. Two changes move the problem: pooling across cell lines raises the median to **75 cells
per gene**, and SubCell's own `Grouping 3` collapses its 31 classes to **seven** (plus an
Other bin for eight rare mitotic annotations, kept rather than dropped so mass is conserved).

**Per gene, it is still not measurable.** At K=7, n_real=75, n_gen=100, against a 40% collapse:

| estimator | 95th-percentile null floor | signal | ratio |
|---|---|---|---|
| KL | 0.829 | 0.132 | 0.16 |
| Jensen-Shannon | 0.0387 | 0.0355 | 0.92 |
| total variation | 0.200 | 0.207 | **1.03** |
| Hellinger | 0.215 | 0.194 | 0.90 |

Not one clears its floor by a useful margin. No single-gene D_cal can be reported.

**As a population trend it works, and comfortably.** Detection rate is the fraction of trials
where the population mean rises monotonically across four guidance weights, each gene carrying
one noisy reference shared across all of them, exactly as in the real design:

| estimator | N=50 genes | N=150 | N=300 | N=448 |
|---|---|---|---|---|
| **total variation** | **100%** | 100% | 100% | 100% |
| Jensen-Shannon | 99.3% | 100% | 100% | 100% |
| Hellinger | 99.3% | 100% | 100% | 100% |
| KL | 51.7% | 57.7% | 58.3% | **64.0%** |

Under the null — a perfectly calibrated model at every weight — total variation gives a 2.0%
false-positive rate and Jensen-Shannon 3.5%, against the 4.2% expected by chance for a
four-point monotone sequence. The test is properly calibrated.

Three things follow.

**Fifty genes suffice.** The complete index gives 448 genes with at least 50 cells, so the
design has close to an order of magnitude in hand.

**KL fails even as a trend**, at 64% with 448 genes. This is not low power. Its bias depends
on the sparsity of each gene's empirical estimate, so averaging over genes concentrates the
bias rather than cancelling it. Use a bounded divergence; total variation was best here and is
also the easiest to interpret, being the fraction of probability mass in the wrong compartment.

**The claim has to be a trend, not a level.** No single number for one protein can be
defended. What can be defended is the population slope against guidance weight.

## Redone over the measured references

The table above resampled references from a Dirichlet. Probe A supplies 40 real ones, and they
are more concentrated: dominant component 0.413 at the median against 0.309, effective support
4.55 groups against 5.94. Resampling from those instead:

| estimator | N=50 | N=150 | N=300 | N=448 |
|---|---|---|---|---|
| total variation | 99.0% | **100%** | 100% | 100% |
| Jensen-Shannon | 96.0% | 100% | 100% | 100% |
| Hellinger | 94.7% | 100% | 100% | 100% |
| KL | 14.3% | 9.3% | 9.0% | **5.3%** |

False-positive rate under a calibrated model is 5.5% for both bounded divergences.

Two revisions follow. **The protein-count floor rises from 50 to 150** — at 50 the bounded
divergences fall to between 94.7% and 99%, which is no longer comfortable. And **plug-in KL
degrades as proteins are added**, from 14.3% to 5.3%, dropping below the 4.2% a monotone
four-point sequence achieves by chance. It is not weak; over these references it is
anti-informative, because averaging concentrates a bias that varies with each protein's
sparsity rather than cancelling it.

## Where this leaves M1

The claim survives, but the measurement plan does not. Restated so it is achievable:

> Over at least 50 proteins with at least 50 single cells each, pooled across the cell lines
> HPA imaged them in, with SubCell's 31 localization classes collapsed to the seven groups of
> its own `Grouping 3`, the population mean total-variation distance between generated and
> predicted-empirical compartment fractions increases monotonically with classifier-free
> guidance weight.

Every quantity in that sentence is one the data supports and the estimator can resolve, and
the simulation above is the pre-registration. The original claim, stated per protein with
plug-in KL over 31 classes, is not measurable at any sample size HPA provides.
