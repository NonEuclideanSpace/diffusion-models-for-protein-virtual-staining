# The SubCell preprocessing bug

Cost: one invalid probe A run, caught by a positive control rather than by anything failing.

## What happened

Probe A produced numbers that looked like a finding. Predicted entropy was about 0.64 nats
against a 31-class ceiling of 3.43, and between-cell heterogeneity — the gap between the
entropy of the mean prediction and the mean per-cell entropy — was 0.001 to 0.003 nats across
every gene, with no separation between the flagged and control groups.

Read at face value that says SubCell cannot see single-cell heterogeneity, which is exactly
the circularity `data-availability.md` warns about, and it would have killed module M1.

It was a bug in my preprocessing.

## The control that caught it

Four proteins with completely different annotations — nucleoplasm, mitochondria, plasma
membrane, Golgi — all received **the same top-1 class at about 0.8 confidence**, with a
largest pairwise total-variation distance of 0.074. A classifier that cannot separate the
nucleus from the plasma membrane is not measuring anything.

The lesson generalizes: before believing a null result from a model you did not train, check
that it can produce the positive result it obviously should.

## The two faults

A sweep over channel order, input range and normalization located both.

| input range | channel order | distinct top-1 across the four | largest pairwise TV |
|---|---|---|---|
| [-1, 1] | any | **1** | 0.03 – 0.11 |
| [0, 1] | nucleus, ER, microtubules, protein | **1** | 0.44 – 0.47 |
| [0, 1] | **microtubules, ER, nucleus, protein** | **4** | **0.725** |

**Range.** SubCell expects [0, 1]. I fed it [-1, 1], which is what a diffusion model wants —
two consumers of the same crop with different conventions, and I had conflated them. This was
the dominant fault: every [-1, 1] row collapses regardless of channel order.

**Channel order.** SubCell takes channels in the order its own `path_list.csv` names them:
r, y, b, g — microtubules, ER, nucleus, protein. Crops are stored nucleus, ER, microtubules,
protein, so the first and third must be exchanged.

## Fix

`pvs.eval.subcell.subcell_input` now performs both, and raises rather than degrading if given
a crop already in the diffusion range. `crops.normalize` still produces [-1, 1] because that
is right for its own consumer. Two tests in `test_crops.py` pin the arrangement.

## After the fix

| protein | annotation | top-1 class | confidence | entropy |
|---|---|---|---|---|
| DCAF11 | nucleoplasm | 26 | 0.590 | 1.880 |
| HSPA9 | mitochondria | 17 | 0.652 | 1.568 |
| CD2AP | plasma membrane | 7 | 0.265 | 2.473 |
| GOLGA5 | Golgi | 11 | 0.505 | 2.028 |

Four proteins, four classes, entropies between 1.57 and 2.47 nats rather than 0.60 to 0.63.
Only now is there enough spread for a heterogeneity measurement to mean anything.

---

## bfloat16 is 3x faster here and gives a different answer

This container's Xeon has AMX (`amx_bf16`, `amx_tile`), so bf16 matmuls are much cheaper than
fp32 and `torch.autocast("cpu", dtype=torch.bfloat16)` is an obvious speedup to reach for.

Measured on 12 real cached cells:

| | ms/cell | argmax agreement | heterogeneity statistic |
|---|---|---|---|
| fp32 | 3185 | — | **0.310** |
| autocast bf16 | 1037 | **0.25** | **0.085** |

3.07x faster, and the statistic the whole project measures moves by a factor of **3.6**. Only
a quarter of cells keep the same predicted class.

The cause is structural rather than a bug: 31 classes with several near-synonymous pairs
(nucleoplasm / nucleoli, cytosol / plasma membrane) means the top two logits are often close,
bf16's 8 mantissa bits are enough to reorder them, and gated-attention MIL pooling upstream
compounds it. **SubCell inference stays in fp32.**

A first attempt at this benchmark used random noise as input and reported 0.00 agreement, which
looks alarming and means nothing — on noise the model is unconfident everywhere and any
perturbation flips the argmax. The number above is on real cells.

### What is left after that

| change | effect |
|---|---|
| batch 1 -> 8 | 2868 -> 1531 ms/cell. Already what the runner uses. |
| batch 8 -> 16 | worse, 2271 ms/cell |
| channels_last | no gain for a ViT |
| bf16 | 3x, **rejected on accuracy** |

ViT-B/16 at 448x448 is 784 patch tokens, roughly 100 GFLOP per cell. Two 2.1 GHz Xeon cores
deliver about 60 GFLOPS in fp32, so ~1.7 s per cell is arithmetic, not inefficiency. There is
no fp32 speedup left on this machine — the work needs different hardware, and
`run_local.sh screen` runs the same job on Apple silicon via MPS.
