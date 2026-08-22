# Data plan and how to run the download

## What changed the picture

Pooling cells across cell lines instead of conditioning on cell line moves the sample size
from unusable to workable. Measured from the crop index at roughly half coverage:

| grouping | cells per gene |
|---|---|
| (gene, cell line) | median 18 |
| gene, pooled over cell lines | **median 65** |

At half index coverage, 187 of the 275 probe-A genes present already have at least 50 cells,
which is the threshold the calibration estimator needs (`m1-feasibility.md`). The cost is
losing cell-line conditioning, so any claim has to be scoped to "pooled across the cell lines
HPA imaged this protein in" rather than to one line.

## Size

Measured across 2400 cells spanning 40 genes: **4.27 MB per cell**, median 4.35, range 0.37
to 7.59, and per-gene means from 1.0 to 6.0 MB.

An earlier estimate of 1.54 MB came from sampling a single plate and was wrong by a factor of
2.8. Estimating storage from one plate is not safe; the spread across genes is large.

| selection | cells | size |
|---|---|---|
| probe A, 200 cells per gene | 34,115 | **146 GB** |
| probe A, 100 cells per gene | ~17,000 | **73 GB** |
| development subset actually fetched | 2,400 | 9.6 GB |

A 1 TB disk holds either comfortably. The full 1.9 TB corpus is not needed.

## Gene selection

`src/pvs/data/hpa.py` builds it. 186 genes carry HPA's `Single-cell variation spatial` flag;
controls are drawn to match each positive on **both** reliability and main location.

Matching on main location is deliberate and load-bearing. Without it, a positive result would
be consistent with nothing more than "multi-located proteins get higher predicted entropy",
which is true by construction and says nothing about cell-to-cell variation. With it, the two
groups have nearly identical multi-localization rates (55.4% against 53.4%), so any remaining
separation is heterogeneity that the consensus label does not already imply.

## Running it

The index is built in a networked environment; the download runs where the disk is. Nothing
about the download needs a GPU.

```
# once, wherever there is network: builds plates/*.csv, resumable
python data/build_index.py

# then, wherever the disk is: check the plan first
python data/fetch_crops.py --index plates --tsv hpa/subcellular_location.tsv \
                           --out /Volumes/DRIVE/hpa-crops --dry-run

# and run it, interruptible and resumable
python data/fetch_crops.py --index plates --tsv hpa/subcellular_location.tsv \
                           --out /Volumes/DRIVE/hpa-crops --workers 12
```

`--max-cells-per-gene` caps the per-gene draw; 200 is generous against a 50-cell requirement
and leaves room to discard cells that fail segmentation quality checks later.

## Crop format, verified against real files

Crops are **1024x1024 RGBA PNG, 8-bit** — not 16-bit as some documentation suggests — with a
separate binary mask marking the cell the crop is centred on. Neighbouring cells are in the
frame and must be masked out for any per-cell measurement.

**Channel order is (nucleus, ER, microtubules, protein).** This was checked rather than
assumed, because an error here corrupts every downstream result without raising anything: the
model would learn to predict the wrong channel from the wrong conditions and the loss would
look perfectly healthy.

The check that settles it is biological. Fetching one crop each for proteins with very
different annotations and correlating the fourth channel against the first:

| gene | annotation | corr(ch3, ch0) | corr(ch3, ch2) |
|---|---|---|---|
| DCAF11 | nucleoplasm | **0.738** | 0.019 |
| GOLGA5 | Golgi | 0.343 | 0.342 |
| HSPA9 | mitochondria | -0.016 | **0.476** |
| CD2AP | plasma membrane | **-0.190** | 0.408 |

Channel 3 tracks the annotation, channel 0 is a single compact bright blob in every cell, and
channel 2 is unmistakably filamentous on inspection. Four downsampled fixtures are committed
under `tests/fixtures/` so the assertion runs in CI rather than living in a document.

Two earlier guesses were both wrong, which is the reason for keeping the evidence: correlation
statistics alone pointed at the wrong assignment, and only looking at the images settled it.

## Normalization

`INTERNAL.md` section 7 item 2 specifies the ProtiCelli protocol and it is worth stating why
it is not the obvious one. Each channel is clipped at its own 99.5th percentile, then **every**
channel is divided by the microtubule channel's percentile rather than by its own. Per-channel
scaling maps all four maxima to 1.0 and destroys exactly the relative brightness that separates
a compact bright structure from a dim diffuse one. `normalize(per_channel=True)` exists only so
the difference can be measured rather than argued about.

## Still to decide

- Where the Phase 1 training set lands. If training runs on a rented or lab machine, download
  there rather than to the laptop and copying.
- Whether to fetch OpenCell at all. It is 121 GB processed, has no microtubule or ER channel,
  and only matters for the domain-adaptation phase that `INTERNAL.md` section 8 already
  recommends cutting.
