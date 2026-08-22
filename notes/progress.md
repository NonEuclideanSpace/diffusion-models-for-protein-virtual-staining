# Progress log

Reference implementation for the protein virtual staining project. Developed in a cloud
container, mirrored to `dmfpvs-v0` on local disk after each verified chunk. The learning
repository is a separate tree and is never written to from here.

---

## 2026-08-21

### Done
- Package scaffold, `uv` environment, src layout under `src/pvs`.
- `diffusion/schedule.py` — linear and cosine beta schedules, all derived constants
  (including the posterior mean and variance), `extract` with broadcasting guards.
- `diffusion/forward.py` — `q_sample`, both prediction inverses, `q_posterior`, SNR.
- `diffusion/ddpm.py` — noise-prediction loss, ancestral sampler.
- `diffusion/ddim.py` — DDIM sampling with eta, inversion, optional fixed-point refinement,
  explicit timestep grids on both directions.
- `diffusion/edict.py` — exactly invertible coupled sampler. Round trip at 3.5e-16.
- `models/embeddings.py` — sinusoidal timestep embedding and its MLP wrapper.
- `models/toy.py` — small MLP predictor plus an analytic oracle used as a test fixture.
- `experiments/toy_2d.py` — 8-mode ring, trains and samples end to end on CPU.
- 60 tests, deterministic under a seeded autouse fixture.
- Data fetch started: `subcellular_location.tsv` (13,603 genes), SubCell encoder and
  classifier heads, SubCell label vocabulary, OpenCell line metadata.

### Findings written up
- `phase0-findings.md` — terminal SNR, sampler validation, mode coverage.
- `inversion.md` — why plain DDIM inversion fails and what fixes it.
- `data-availability.md` — **the per-cell label blocker. Read this before Phase 3.**

### Next
1. Run probe A from `data-availability.md`: does SubCell detect real single-cell
   heterogeneity? No GPU, decides whether M1 has a measurable target at all.
2. Conditional U-Net baseline, completing Phase 0.
3. v-prediction, required before any zero-terminal-SNR rescaling.
4. BDIA as a 1-NFE alternative to EDICT, validated against it.

### Added later on 2026-08-21
- `notes/m1-feasibility.md` — **the D_cal estimator inverts sign at HPA's real sample sizes.**
  Measured 18 cells per (gene, cell line) in U2OS; simulated the plug-in KL null distribution;
  found the only usable configuration is K<=5, n_real>=50, N>=300 proteins on the trend.
- `notes/outbox/kaggle-labels-request.md` — verified contact path and a draft, unsent.
- Sized storage from 25 sampled crop manifests: 1.54 MB per cell measured, U2OS is 27.1% of
  the corpus. A scoped subset is tens of GB, not terabytes.

### Phase 0 completed, Phase 1 model built
- `diffusion/parameterization.py` — eps / x0 / v conversions, and `as_eps_predictor` so the
  samplers stay parameterization-agnostic.
- `schedule.enforce_zero_terminal_snr(floor)` — the Lin et al. rescaling with a floor, since
  driving alpha_bar_T to exactly zero breaks every exact inversion scheme.
- `diffusion/bdia.py` — exact inversion at **1 network evaluation per step**, half of EDICT.
- `models/blocks.py`, `models/conditioning.py`, `models/unet.py` — conditional U-Net baseline
  with channel dropout and a learned null plane.
- `diffusion/guidance.py` — classifier-free guidance, with the dropped condition selectable
  because dropping the label and dropping the landmarks are different interventions.
- `models/dit.py` — **DiT with adaLN-Zero**, plus the mode-embedding hook module M1 needs.
- 111 tests.

### Measured, and better than the literature reports
BDIA dominates EDICT on every axis tested:

| | NFE per step | round trip | at 200 steps | safe parameter range |
|---|---|---|---|---|
| EDICT | 2 | 3.5e-16 | degrades to 3.2e-12 | p in [0.9, 0.97] |
| BDIA | **1** | **3.8e-17** | **holds at 2.3e-17** | gamma in [0.9, 1.0] |

Two findings the papers do not state:
- The BDIA recurrence given both initial states is exact to **7e-18**, and it *attenuates* an
  error in the bootstrap by about 28x rather than amplifying it. The parasitic root that
  leapfrog schemes classically carry is not a practical problem at these step counts.
- **A perfect round trip does not imply a usable latent.** At gamma 0.8 the round trip is
  still 1.2e-15 while the recovered latent has standard deviation 12.9, and at gamma 0.5 it
  is 1e11. Round-trip error alone must never be the acceptance criterion.

### Phase 1 infrastructure and module M2
- `models/dit.py` — DiT with adaLN-Zero, mode-embedding hook included so Phase 3 needs no
  model change. `models/unet.py` — the baseline it has to beat.
- `diffusion/guidance.py` — classifier-free guidance with a selectable dropped condition.
- `train/loop.py`, `train/ema.py` — resumable training: optimizer, scheduler, EMA, step count
  and RNG state travel in one atomically written checkpoint. A resumed run reproduces the
  uninterrupted trajectory exactly, which is the only guarantee that matters on rented compute.
- `data/synthetic.py` — a virtual-staining task whose conditional distributions are known by
  construction, including a vesicular compartment whose position no landmark predicts. That
  reproduces, deliberately, the failure mode every published model shows.
- `eval/calibration.py` — bounded divergences and the null-distribution machinery, because
  plug-in KL inverts sign at HPA's sample sizes.
- `eval/compartment.py` — the learned classifier standing in for SubCell.
- `eval/information.py` — **module M2**, the landmark-by-compartment information matrix.
- `data/hpa.py`, `data/fetch_crops.py`, `data/build_index.py` — gene selection and a
  resumable scoped download.
- 156 tests.

### Measured
- **Pairing the randomness in M2 cuts the estimator's standard deviation 6.3x** (40x in
  variance). Delta is a difference of two nearly equal expectations; drawing the timestep and
  the noise once and evaluating both conditions on them is what makes it measurable.
  `INTERNAL.md` section 5 does not mention this and the estimator is close to useless without it.
- **Pooling cells across cell lines raises the median from 18 per gene to 65**, past the
  50-cell threshold the calibration estimator needs. See `data-plan.md`.
- The probe-A download is about 31 GB at half index coverage, so roughly 60 GB complete.

### First result from real data (2026-08-21)
**Probe A passes on a 40-gene subset.** Between-cell heterogeneity separates HPA-flagged genes
from matched controls with effect size 0.693 [0.525, 0.848], z +2.08. Embedding spread, which
never touches a classifier head, separates more strongly still at 0.725, z +2.43 — evidence
that the signal is not an artefact of the broadcast-label pipeline. See `probe-a.md`.

Plain entropy, the obvious statistic, separates nothing (0.450) and points the wrong way.
Probe A would have failed on it.

**A preprocessing bug invalidated the first run.** SubCell expects [0, 1] and channels in
(microtubules, ER, nucleus, protein) order; I fed it [-1, 1] in on-disk order. Every protein
then received the same class at 0.8 confidence, which read as a finding rather than a defect.
Caught by a positive control, not by anything failing. See `subcell-preprocessing.md`.

**M1 is measurable, but only restated.** At the real parameters — 7 groups, 75 cells per
protein — no per-protein divergence clears its own noise floor. The population trend across
guidance weights does, at 100% detection with 50 proteins using total variation, against 64%
for plug-in KL even with 448. Pre-registered in `m1-protocol.md`.

### Open questions for Euclid
- The Kaggle 2021 per-cell expert labels (41,597 cells) are the only real per-cell ground
  truth and appear unreleased. Worth writing to CellProfiling to ask.
- Storage: the full SubCell HPA crop set is roughly 5 TB. Decide where it lands before
  Phase 1.
