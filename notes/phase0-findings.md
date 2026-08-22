# Phase 0 findings

Status: forward process, DDPM, DDIM and DDIM inversion implemented and tested (49 tests).
Validated against an analytic oracle (exact score for isotropic Gaussian data) and a trained
model on an 8-mode 2-D mixture. No GPU used.

## 1. Terminal SNR governs sampling bias

Samplers start from `N(0, I)`, which is only the correct marginal when `alpha_bar_T` reaches 0.
Linear betas tuned for T=1000 do not reach it at shorter T, and the sample scale is biased.

| T | schedule | alpha_bar_T | true std(x_T) | ddpm std | ddim std |
|---|---|---|---|---|---|
| 200 | linear | 1.3e-1 | 1.182 | 1.831 | 1.665 |
| 1000 | linear | 4.0e-5 | 1.000 | 2.000 | 1.983 |
| 200 | cosine | 6.1e-8 | 1.000 | 1.991 | 1.942 |

Target std was 2.0. Feeding the correct terminal marginal to the T=200 linear case recovers
1.978, so the sampler is right and the schedule is the problem. Matches Lin et al. 2024.

## 2. Terminal SNR also governs inversion conditioning, in the opposite direction

The last inversion step divides by `sqrt(alpha_bar)`. As `alpha_bar -> 0` the coefficient on
the noise-estimate difference rises from ~0.02 to ~0.98, so model error passes through
unattenuated. Profiling a 100-step inversion: the final step contributed 93% of total error.

Perturbing an exact oracle by `0.01 * sin(3x)` and measuring the round trip:

| model error | truncated at alpha_bar 6e-3 | full range to alpha_bar 6e-8 | amplification |
|---|---|---|---|
| 0 | 3.9e-5 | 3.7e-4 | 10x |
| 0.001 | 3.8e-5 | 4.1e-2 | 1075x |
| 0.01 | 4.2e-5 | 3.8e-1 | 9089x |

**This is the central Phase 0 constraint.** Sampling wants zero terminal SNR; inversion wants
terminal SNR bounded away from zero. The INTERNAL.md Phase 0 gate ("DDIM inversion round-trip
error below a threshold") is only meaningful once the terminal timestep is fixed. Proposal:
state the gate as round-trip error over a grid truncated at `alpha_bar >= 1e-2`.

## 3. Fixed-point inversion is worth its cost where the problem is well posed

Reusing the current point's noise estimate for the target point is the dominant error in
plain DDIM inversion. Re-solving the sampling update for its own input removes it.

| steps | plain | 2 fixed-point iterations |
|---|---|---|
| 50 | 4.8e-2 | 1.5e-3 |
| 200 | 1.2e-2 | 8.8e-5 |

On the truncated grid this reaches 1e-4 with a trained model and preserves mode identity for
100% of samples. Across the singular final step it diverges instead, which is consistent with
the iteration being a contraction only while the coefficient stays below 1.

## 4. Deterministic sampling redistributes mass between modes

On the 8-mode mixture, with an adequately trained model:

| sampler | KL(sampled modes || real modes) |
|---|---|
| ddpm ancestral | 0.0004 |
| ddim eta=0.7 | 0.0006 |
| ddim eta=0.3 | 0.0074 |
| ddim eta=0.0 | 0.0267 |

Step count also matters at eta=0 (0.046 at 25 steps, 0.018 at 200). An undertrained model
collapsed four of eight modes outright (KL 0.68), so this measurement is only meaningful
once the loss has converged.

This is a miniature of the M1 quantity `D_cal`: a categorical divergence between generated
and empirical mode fractions, with ground-truth weights available by construction. Worth
keeping as the smoke test for the M1 estimator before it meets real data.

## Open

- v-prediction, needed if the zero-terminal-SNR rescaling of Lin et al. is adopted:
  eps-prediction divides by `sqrt(alpha_bar)` and cannot survive `alpha_bar = 0`.
- Conditional U-Net baseline (Phase 0 remainder).
- Data and weight availability, which bounds M1 and is not a code question.
