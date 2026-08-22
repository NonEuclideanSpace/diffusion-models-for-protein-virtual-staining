# Localization as a cell-state readout

A proposed upgrade to the project's spine claim, with the pilot that motivates it and the
reasons it is not yet established.

---

## 1. The observation

The project's spine is *"conditional diffusion virtual staining learns a consensus localization,
not a conditional distribution."* That is a claim about a model's failure. The data suggests a
claim about the target instead.

**A protein's position in a single cell is often not a property of the protein. It is a readout
of that cell's state.** CDC20 sits in the nucleoplasm through G2/M and is destroyed by the APC/C
in G1. INCENP moves from centromere to spindle midzone during mitosis. NR3C1 stays cytosolic
until it is liganded. For proteins like these, what `pi_emp` measures is not a conditional
distribution at all — it is a **marginal over an unobserved state variable**.

That splits the quantity module M1 treats as its target:

    H(compartment | protein) = I(compartment ; state | protein) + H(compartment | protein, state)
                               ^ recoverable by conditioning      ^ genuinely aleatoric

The first term is not noise. It is information a model could have used and did not — and part of
it is sitting in the **landmark channels the model already receives**. DNA content, read from the
nucleus channel, is a standard cell-cycle proxy; nuclear and cell area carry size and confluency.

## 2. Why this would be an upgrade rather than a rewrite

| | spine as written | this |
|---|---|---|
| shape of the claim | a deficiency: the model collapses | a quantity: how much of the spread is recoverable |
| what it predicts | `D_cal` rises with guidance weight | `D_cal` falls when state is conditioned on |
| survives CELL-Diff having no guidance? | **no** — the mechanism claim is about a knob the SOTA model never turns | yes — it says nothing about samplers |
| gives the field a number it lacks | no | **yes**: an information-theoretic ceiling |

That last row is the part worth having. Virtual staining is evaluated with pixel correlation,
SSIM and FID, **none of which has a known optimum** — a reported 0.87 tells you nothing about how
much was achievable. `I(compartment ; state | protein)` is computable from data alone, with no
model, and it says how much of the remaining spread is in principle reachable.

## 3. The pilot

`experiments/state_decomposition.py`, 24 U2OS genes, 40 cells each, half drawn from the heavy
tail of measured heterogeneity and half from the middle so the answer is not read off the tail
alone. Covariates come only from channels a virtual staining model is already conditioned on.
The plug-in mutual information estimator is biased upward at 40 cells, so a permutation null is
subtracted.

**The genes at the top:**

| gene | H | DNA | cell area | chromatin density | what it is |
|---|---|---|---|---|---|
| CDK1 | 0.3815 | +60% | −0% | **+78%** | cell-cycle master kinase |
| ABITRAM | 0.2747 | +37% | −2% | **+47%** | |
| CTSC | 0.2412 | +17% | +9% | **+38%** | cathepsin, lysosomal |
| ARF6 | 0.3697 | +6% | **+33%** | **+33%** | vesicle trafficking |
| SKIC2 | 0.0694 | +30% | +7% | +25% | |
| NR3C1 | 0.4422 | **+11%** | +2% | −1% | glucocorticoid receptor |

**The controls behave:**

| gene | DNA | cell area | chromatin density |
|---|---|---|---|
| ZNF560 | −2% | −2% | +2% |
| ACAD9 | +1% | −2% | −1% |
| NEK2 | +1% | −1% | −1% |
| RAB23 | −2% | −3% | −1% |
| LIN37 | +1% | +1% | −1% |

**The distribution across all 24:**

    dna                 median +2.5%   mean  +9.1%    9/24 genes clear 2 null SD
    cell_area           median +0.5%   mean  +4.5%    8/24
    chromatin_density   median +3.7%   mean +12.6%   12/24

## 4. What that actually says, stated carefully

**The median gene is barely explained.** Median 3.7% is not "localization heterogeneity is mostly
cell state". The mean is three times the median, which is the signature of a heavy tail: **a
minority of genes are almost entirely state readouts and most are not.**

So the honest claim is not a universal law. It is:

> A measurable subset of proteins have localization that is largely determined by cell state,
> that state is recoverable from the landmark channels, and the fraction is quantifiable per
> gene.

That is still worth having, because **the quantity is a selector**: it ranks genes by how much of
their position is a state readout, which is a list of candidate endogenous state reporters. CDK1
at 78% is what the top of such a list looks like.

## 4b. The representative draw, 130 genes — and a correction

The pilot in section 3 deliberately over-sampled the tail. This is a uniform random draw over
the 588 screened genes, `--sample random --seed 0`.

| covariate | median share | mean share | clear 2 null SD |
|---|---|---|---|
| integrated DNA | +2.9% | +6.7% | 50/130 |
| cell area | +3.0% | +5.8% | 53/130 |
| chromatin density | **+3.7%** | **+9.0%** | **60/130** |
| **any of the three** | **+11.7%** | — | **97/130 (75%)** |

Two things change.

**The means came down, as they had to.** Chromatin density went from 12.6% on the stratified
pilot to 9.0% here. The pilot's mean was an artefact of selecting half its genes for being
extreme, which is exactly why the work order now requires `--sample random`.

**The effect is far more common than the pilot suggested.** 75% of genes have at least one
covariate clearing two null standard deviations, against about 2.3% expected by chance. This is
not a rare property of a few cell-cycle proteins. It is a small effect in most proteins.

### The correction: it is *not* concentrated in the tail

I said earlier that the ablation would be justified because M1 selects the high-heterogeneity
tail, where the explained share is large. **That is wrong.** By stratum:

| stratum | n | median H | best-covariate share |
|---|---|---|---|
| top decile | 13 | 0.2412 | **8.9%** |
| top quartile | 32 | 0.1578 | 11.3% |
| top half | 65 | 0.0968 | 13.3% |
| bottom half | 65 | 0.0333 | 10.1% |
| all | 130 | 0.0632 | 11.7% |

The explained share is **roughly flat**, and the top decile is if anything slightly *below* the
overall median. Correlation between heterogeneity and explained share is only +0.328.

CDK1 at 78% is real and it is in the tail, but the tail's *median* is 8.9%. **The tail is
heterogeneous in why it is heterogeneous**: it contains cell-cycle proteins whose position is
almost entirely state, and proteins like ZNF560 and ACAD9 whose heterogeneity has nothing to do
with any covariate measured here.

That is itself a finding worth stating: **high localization heterogeneity has several distinct
causes, and cell-state is only one of them.**

### What it does to the expected ablation effect

A state-conditioned model can, at best, remove about **10% of the conditional entropy on a
typical gene** — not the 78% the pilot's headline implied. At 150 proteins the M1 machinery
detects considerably smaller effects than that (`m1-protocol.md` Amendment 2), so the ablation
is still worth running and its result will be interpretable.

But the honest pre-registration of the expected effect is **around 10%, uniform across the
heterogeneity range**, and the ablation should therefore be evaluated on **all** proteins rather
than on the tail. Written down now so that a 10% result is read as the predicted outcome rather
than a disappointment, and a 40% result is treated with suspicion.

## 5. What would make it real

- **200+ genes, not 24.** The pilot is half tail by construction; the distribution over a
  representative sample is the number that matters. This is Task 1 of the server work order and
  it gates everything else.
- **The controls must stay near zero at scale.** If they do not, the permutation correction is
  insufficient and the whole thing is estimator bias.
- **DNA and chromatin density are correlated** and may be measuring one thing. CDK1 gives +60%
  and +78% respectively; that gap needs explaining rather than reporting.
- **A model has to actually improve.** Nothing here shows that conditioning on state helps a
  diffusion model. That is the ablation, and it is the only part that needs a GPU.

## 6. What was built for it

- `StateEmbedding` in `src/pvs/models/conditioning.py` — covariates through the same adaLN path
  as the timestep, 4,544 parameters, output layer zero-initialised so **adding it does not
  change an existing model at initialisation**. That makes the ablation exact rather than
  approximate: at step 0 the two arms are numerically identical.
- `CachedCrops(state=True)` and `state_covariates()` in `src/pvs/data/cache.py`.
- `build_cache.py --masks`, because without a cell mask the nuclear statistics pick up every
  neighbour in the frame.
- `experiments/train.py --state`.
- Nine tests, including one whose only job is to defeat the adaLN-Zero trap that has already
  produced two false negatives on this project.

## 7. The version that would be novel rather than merely sensible

Hand-picking DNA content and area is the obvious move. The interesting one is to **learn the
state variable**: a low-dimensional `z = f(landmarks)` trained to minimise
`H(compartment | protein, z)` subject to a bottleneck on `z`.

That turns virtual staining into **state inference** — the model returns not only an image but an
estimate of where the cell sits in state space — and it makes the recoverable term something the
model discovers rather than something we hand it. It needs its own training run, which is the
third card in `gpu-request.md`.

## 8. Reproduce

    uv run python experiments/state_decomposition.py --genes 24
