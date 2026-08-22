# Data availability

Surveyed 2026-08-21. Every claim below was verified against a primary source or a live
request; items that could not be verified are marked.

## Summary

| Asset | Status | Where |
|---|---|---|
| HPA images, 4 channels | available | per-image HTTPS, no bulk endpoint, CC BY 4.0, v25.1 |
| HPA pre-segmented single-cell crops | available | `s3://czi-subcell-public/hpa-processed/`, ~1.22M cells, ~5 TB, CC BY-SA 3.0 |
| **Per-cell compartment labels** | **not found** | see below |
| OpenCell imaging | available | `s3://czb-opencell`, 1,310 lines, REST API, CC BY-SA 4.0 |
| CELL-Diff weights | available | `s3://czi-celldiff-public/v2/`, MIT, 137 GB incl. training LMDBs |
| ProtiCelli code and weights | available | GitHub, MIT; checkpoints 6.88 GB + 0.13 GB VAE |
| SubCell encoder and classifiers | available | `s3://czi-subcell-public/models/`, 88 files, 3.1 GB, MIT |

Everything the project needs is public **except the one thing the central claim rests on.**

## The blocker

`INTERNAL.md` section 4 defines the empirical mixture weights as

```
pi_emp[k] = Pr_{x ~ D}[ phi(x) = k | protein, cell_line ]     from real single cells
```

and argues that HPA's measurability of these weights is "the only genuinely original element
in the project". That measurability does not hold as stated.

**HPA labels are per-antibody, and are broadcast to every cell in a field of view.** Verified
directly by reading a crop manifest, `hpa-processed/cell_crops/1/1_A1_1_cell_bbox.csv`:
all eight cells in that field carry the identical string `Golgi apparatus`. The
OpenCell-processed metadata behaves the same way. Per-cell *geometry* is real; per-cell
*labels* are not.

The only fields in `subcellular_location.tsv` that touch single cells are coarse flags, not
distributions: `Single-cell variation intensity` on ~3,650 of 13,603 genes, `Single-cell
variation spatial` on **186 genes**, `Cell cycle dependency` on 479.

**The one genuine per-cell annotation set appears unreleased.** The HPA Single-Cell
Classification Kaggle competition (2021) had 41,597 single cells annotated per-cell by HPA
experts, 10% double-annotated at 90% inter-annotator consistency, across 19 classes. Those
are test-set labels and Kaggle withholds test labels; the official analysis repository
contains scoring code and the winning solution but no ground-truth file. Whether the data
remains downloadable at all could not be verified without an account.

## What this does to M1

`pi_emp` has to be **predicted** by SubCell rather than looked up. That turns the reference
distribution from a measurement into a model output, and introduces a circularity:
SubCell's classifier heads were themselves trained against the same broadcast image-level
labels. A classifier trained to map every cell in a field to one consensus label may well
regress individual cells toward that consensus, which would systematically **understate**
exactly the heterogeneity M1 exists to measure.

If both `pi_emp` and `pi_hat` are understated, `D_cal` can come out small for the wrong
reason, and the module's kill gate fires as a false negative.

## Probe A — the cheapest decisive test, and it needs no GPU

The original probe in `INTERNAL.md` section 4 samples CELL-Diff and compares to `pi_emp`.
That probe silently assumes `pi_emp` is trustworthy. It is not, so the assumption has to be
tested first, and it can be tested for almost nothing.

> **Does SubCell detect single-cell heterogeneity that is known to be real?**

HPA flags 186 genes as spatially variable at the single-cell level and ~3,650 as
intensity-variable. Those flags are curated by HPA annotators and are **independent of
SubCell**. OpenCell's `heterogeneous_gfp` flag (89 lines) is a second independent source.

Procedure: draw the flagged genes and a control set matched on reliability, cell line and
main location. Run SubCell over their crops. Compute per-(protein, cell line) entropy of the
predicted compartment distribution. Test whether the flagged group separates from the control.

- **Separation present** — SubCell recovers real heterogeneity, `pi_emp` is usable with a
  stated confidence interval, and this becomes the calibration figure that licenses M1.
- **No separation** — the predicted distribution carries no heterogeneity signal, `pi_emp`
  is unavailable, and the M1 claim has no measurable target. Fall back to M2 or M3, and
  record the negative result, which `INTERNAL.md` section 9 item 2 already commits to doing.

This is a stronger gate than the original probe because it fails earlier and costs less. It
also produces the classifier calibration number that any reviewer would ask for.

## Fallback if probe A fails

Redefine the mode variable so it does not pass through a label head. `INTERNAL.md` section 4
already allows this: cluster the protein's single-cell SubCell **embeddings** rather than
classifying them. Modes then become data-defined, and the circularity through the classifier
is removed, though the encoder's protein-level supervision remains. The cost is that modes
lose their biological names, which weakens the write-up but not the measurement.

## Practical hazards

- **Three canonical pixel sizes** in play: HPA 0.0801276, SubCell 0.0800885,
  ProtiCelli 0.1067 um/px. Resampling errors here are silent and corrupt texture metrics.
- **Licence heterogeneity.** HPA itself is CC BY 4.0, the CZI redistributions are CC BY-SA
  (3.0 for HPA crops, 4.0 for OpenCell), and the ProtiCelli preprint is CC BY-NC. Share-alike
  is viral; mixing these into one derived dataset propagates the strictest terms. All code is
  MIT, so the constraint is on data, not software.
- **OpenCell is not channel-compatible with HPA**: protein plus nucleus only, no microtubule
  or ER channel, and 0.206 um/px against HPA's 0.080. Cross-domain work loses two of the four
  conditioning channels and needs the reduced `DNA-Protein` SubCell variant.
- **HPA's per-topic TSVs are unlisted** on the v25.1 download page though still served, and
  two of them already return 404. Pin a local copy rather than fetching at runtime. Done:
  `data/hpa/subcellular_location.tsv`.
- **Volume.** ~5 TB for the full SubCell HPA crop set, +121 GB OpenCell-processed, +137 GB
  for CELL-Diff weights and LMDBs. Needs a decision before Phase 1.
