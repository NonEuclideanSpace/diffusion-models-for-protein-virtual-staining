# If the CELL-Diff probe says D_cal is small

Written **before** the probe runs, so the fallbacks cannot be chosen to flatter whatever comes
back. `INTERNAL.md` section 4 says a small `D_cal` kills module M1 *as specified*. It does not
kill the project, and the branches below are not consolation prizes — one of them is a sharper
claim than the original.

---

## What "D_cal is small" would and would not establish

`D_cal = KL(pi_hat || pi_emp)` compares two distributions over compartments, each **marginalised
over cells**. Small means the model reproduces the *population* mixture. Three things it does
not mean:

1. **It does not mean the fine structure matches.** The grouping is 7 groups plus Other, chosen
   for statistical power. A model can match coarse compartments while collapsing the 31-class
   distinction inside them.
2. **It does not mean the tail matches.** `D_cal` averaged over proteins can be small while
   being large exactly on the high-heterogeneity proteins M1 is about. The probe selects those
   proteins, but the *contrast* between high- and low-heterogeneity proteins is a separate
   measurement and a more pointed one.
3. **It does not mean the conditional is right.** This is the big one and it gets its own
   section.

---

## Plan B — the per-cell conditional test

The strongest version of the spine claim is not distributional at all.

Given a specific cell's landmark channels, the model should place the protein where **that
cell** puts it. `D_cal` cannot see this: a model that samples from the correct population
mixture while ignoring the cell in front of it scores perfectly.

    D_cal        compares  pi_hat(compartment | protein)  to  pi_emp(compartment | protein)
    the per-cell test compares, for each cell i,  phi(generated_i)  to  phi(real_i)

Concretely: generate one image per real cell, conditioned on that cell's own landmarks, and
measure paired agreement — accuracy against the cell's own SubCell call, and the mutual
information `I(generated ; real)` over compartments, against a null built by shuffling the
pairing.

**"Reproduces the marginal, fails the conditional" is a cleaner statement of "learns a consensus
localization" than the guidance story ever was**, and it is testable on exactly the images the
probe already generates. No extra compute, no new model.

It also survives the discovery that CELL-Diff uses no classifier-free guidance
(`celldiff-probe.md`), which the original mechanism claim does not.

---

## Plan C — if the per-cell conditional also matches

Then conditional diffusion virtual staining is doing better than the field says it does, and
saying so with measurements is a contribution. `INTERNAL.md` section 2 is built on limitations
ProtiCelli *named*; a paper that takes a named limitation, measures it properly with controls,
and reports that it is not there is a real result — provided the measurement is strong enough to
have found it. That is what the reliability numbers, the null calibration and the power
calculation are for.

The risk to watch: this is the outcome most likely to be a false negative. Before claiming it,
the measurement has to be shown capable of detecting a collapse that is *injected* — which
`m1_power.py` already does at 100% detection for 150 proteins, and which should be repeated on
CELL-Diff's own outputs with a synthetic collapse mixed in.

---

## Plan D — the measurement and resource paper, which stands regardless

Already in hand and independent of every branch above:

- **The annotation negative result.** HPA's spatial single-cell variation annotation does not
  predict measured localization heterogeneity: 0.433 [0.303, 0.564] on 77 never-seen genes,
  enrichment 0.99x, split-half reliability 1.000, head-free confirmation, cell-line confound
  controlled by construction.
- **A reliable measure and a data-driven ranking.** Independent-sample reliability 0.923,
  heavy-tailed, and it recovers known biology — CDC20's two populations differ 1.5x in
  chromatin density at constant nuclear area, which is the G1/G2M signature, predicted before
  it was measured.
- **The U2OS ranking itself**, 600 genes, as a resource for anyone selecting heterogeneous
  proteins.

This is a smaller paper than Plan A or B but it is not contingent on anything.

---

## Plan E — self-contained methods results

Each of these is small and each is finished:

- **Sampling and inversion want opposite terminal SNR** (`findings.md` C1). Driving
  `alpha_bar_T` to zero is required for unbiased sampling and makes exact inversion amplify
  model error 38x; a floor at 1e-2 satisfies both. No paper appears to address the interaction.
- **BDIA against EDICT** at equal budget: exact inversion at 1 NFE, 3.8e-17, stable where EDICT
  degrades four orders of magnitude at 200 steps.
- **Label-free threshold calibration by phase scrambling** (`m3-gate.md`). Randomise Fourier
  phase, keep the power spectrum: whatever a detector still finds is a false positive by
  construction. It showed 89% of our spot detections were texture.

---

## What changes with A800s

Every branch above is currently scoped to one 8 GB card. With A800s:

- **Plan A stops being a small-model paper.** The `mid` preset — 130M parameters, 256px, 8-pixel
  patches — is 0.89 A100-days, and `large` is 2.94. Either is a model reviewers will not dismiss
  next to CELL-Diff.
- **The full M1 sweep becomes affordable**: 150 proteins x 4 guidance weights x 100 samples is
  60,000 generations, which is hours rather than a week.
- **Plan B gets its proper sample size.** The per-cell test wants one generation per real cell
  across thousands of cells, not dozens.
- **M2's magnitudes become quotable.** `m2-rehearsal.md` shows the estimator's noise floor is
  half its signal at this compute; more evaluation batches and a better model separate them.

The order does not change. The probe still comes first, because it is what tells us which branch
we are on, and it costs an hour on hardware we already have.
