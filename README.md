# pvs — reference implementation

Working implementation for the protein virtual-staining project. Written to be read: every
module exists because something in `INTERNAL.md` needs it, and the reasoning that is not
obvious from the code lives in `notes/`.

Nothing here is the learning repository. That is a separate tree and is written only by hand.

```
uv sync
uv run pytest          # 177 tests, no GPU, about a minute
```

## Layout

```
src/pvs/
  diffusion/     schedule, forward process, DDPM, DDIM, EDICT, BDIA, guidance,
                 eps/x0/v parameterization
  models/        conditional U-Net baseline, DiT with adaLN-Zero, shared blocks,
                 conditioning with channel dropout
  train/         resumable loop, EMA
  data/          synthetic task, HPA gene selection, crop loading and normalization
  eval/          SubCell encoder, compartment classifier, calibration divergences,
                 module M2 information matrix, image and distribution metrics
data/            build_index.py and fetch_crops.py — run these where the disk is
experiments/     toy_2d.py, synthetic_m1.py, synthetic_m2.py
notes/           findings; read data-availability.md and m1-feasibility.md first
```

## What to read first

**`notes/findings.md`** — everything established so far, with the numbers, ordered by how much
it changes the project. Start here.

`notes/m1-feasibility.md` — the estimator that module M1 depends on inverts sign at the
sample sizes HPA actually provides. This is the single most consequential finding so far and
it changes what the study can claim.

`notes/data-availability.md` — everything the project needs is public except per-cell
localization labels, which appear never to have been released.

`notes/inversion.md` — sampling wants zero terminal SNR and inversion wants the opposite.
Where the Phase 0 gate should actually be set, and why.

`notes/data-plan.md` — crop format, verified channel order, normalization, and the download.

`notes/progress.md` — dated log.

## Choices worth knowing about

**Inversion.** Both EDICT and BDIA are implemented. BDIA wins on every axis measured: one
network evaluation per step instead of two, round trip at 3.8e-17 instead of 3.5e-16, and no
degradation at 200 steps where EDICT loses four digits to mixing dilation. EDICT is kept as
the independent check that catches algebra errors in the other.

**A perfect round trip does not mean a usable latent.** Below its safe parameter range BDIA
still reconstructs at machine precision while returning a latent with standard deviation 12.9.
Round-trip error alone must never be the acceptance criterion.

**Channel dropout and classifier-free guidance are one mechanism.** OpenCell has no
microtubule or ER channel and guidance needs an unconditional pass; implementing them
separately would mean two ways for a condition to be absent.

**The synthetic task exists to validate the measurement, not the model.** Its conditional
distributions are known by construction, including a vesicular compartment that no landmark
predicts — the failure mode every published model shows, reproduced deliberately so the
estimators can be checked against a known answer before meeting real data.
