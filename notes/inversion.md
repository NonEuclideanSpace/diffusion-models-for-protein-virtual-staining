# Inversion

## The problem

DDIM sampling steps from noise level `a` to `a'` using the noise estimate at the current
point. Inverting that step needs the estimate at the *target* point, which is what we are
solving for. Plain DDIM inversion breaks the circle by reusing the current point's estimate.
That lag is an approximation, and its error compounds.

Two separate things go wrong, and they were easy to confuse:

1. **The lag itself.** Fixable by iteration or by restructuring the step.
2. **Conditioning as `alpha_bar -> 0`.** The step divides by `sqrt(alpha_bar)`, so the
   multiplier `a_t = sqrt(alpha_bar_prev / alpha_bar)` diverges. Profiling a 100-step
   inversion under a cosine schedule: the final step alone contributed 93% of total error.

## What each remedy actually buys

Measured on a 2-D fixture with an analytic score, perturbed by `0.01 * sin(3x)` to stand in
for model error. Relative round-trip error, 50 steps unless stated.

| method | linear (`alpha_bar_T` 1.3e-1) | cosine (`alpha_bar_T` 6.1e-8) |
|---|---|---|
| plain DDIM inversion | 4.8e-2 | 7.2e-2 |
| + 2 fixed-point steps | 1.5e-3 | diverges |
| EDICT, float32 model | 9.4e-9 | 6.5e-3 |
| EDICT, float64 model | **3.5e-16** | 4.0e-12 |

**Fixed-point iteration** re-solves the sampling update for its own input. It converges only
while `|shift/scale| * Lip(model)` stays below 1, which fails across the singular final step —
that is why it appeared to diverge on trained models before the cause was isolated.

**EDICT** carries two sequences and steps each from the other's noise estimate, making every
step an affine coupling layer and therefore invertible by construction. Model error cancels
because the round trip reproduces the same estimates. Cost is 2x the network evaluations.

## Three numerical constraints, all measured

**Float64 for the coupling.** Casting the coupling state down to float32 at each model call
quantizes the input, so the reverse pass sees different values and exactness is lost:
3.5e-16 becomes 8.1e-9. With a real float32 network, 1e-8 is the realistic floor.

**Floor the terminal SNR.** Round-trip error tracks `a_T`. Dropping a single step from the
end of a cosine grid moves the error from 4.0e-12 to 7.2e-16:

| last t | alpha_bar | round trip |
|---|---|---|
| 199 | 6.1e-08 | 3.95e-12 |
| 195 | 9.7e-04 | 7.21e-16 |
| 191 | 3.9e-03 | 4.72e-16 |
| 179 | 2.4e-02 | 4.71e-16 |

**Mixing coefficient and step count.** Un-mixing divides by `p` once per step, dilating any
difference between the two sequences geometrically.

| p | round trip | pair drift |
|---|---|---|
| 1.00 | 2.3e-16 | 1.6e-04 |
| 0.93 | 3.6e-16 | 3.7e-03 |
| 0.90 | 5.3e-16 | 4.0e-02 |
| 0.80 | 3.5e-14 | 1.7e+00 |
| 0.60 | 8.5e-03 | 1.6e+00 |

The same dilation limits step count: 25/50/100 steps all land at ~4e-16, but 200 steps costs
four digits (3.2e-12). Keep `p` in [0.9, 0.97] and prefer fewer, larger steps.

## Consequence for the Phase 0 gate

`INTERNAL.md` section 8 sets the gate at "DDIM inversion round-trip reconstruction error
below a self-set threshold". The measurements above show the number is governed almost
entirely by the terminal timestep, so the gate is not well defined without it. Proposed
restatement:

> EDICT round-trip relative error below 1e-6 on a timestep grid truncated at
> `alpha_bar >= 1e-2`, with the coupling arithmetic in float64.

That threshold is loose enough to survive a float32 network and tight enough that any error
in the schedule constants, the timestep spacing or the coupling algebra shows up immediately.

## What the literature does and does not say

EDICT (Wallace et al., CVPR 2023) reports reconstruction pinned to the autoencoder floor at
50, 200 and 1000 steps, conditional and unconditional. It warns that repeated dilation can
exaggerate floating-point differences, but only in the context of small `p`.

Lin et al. (WACV 2024) state the terminal singularity directly and prescribe converting to
`x0` form — which removes the affine structure EDICT depends on. So zero terminal SNR and
exact coupling are not simultaneously achievable, and the compromise is to floor
`sqrt(alpha_bar_T)` rather than drive it to zero.

**No paper appears to evaluate an exactly-invertible inversion method under a zero terminal
SNR schedule.** The interaction is unaddressed in the literature; the numbers above are ours.

Null-text inversion targets the error classifier-free guidance introduces, not the error in
plain DDIM inversion, and its optimized object is a text-token embedding. It does not port to
image-plus-class conditioning and should not be attempted here.

ReNoise is a better fixed-point scheme with averaging. It is more accurate, never exact, and
at 50 deterministic steps costs the same 2 NFE as EDICT for a weaker guarantee. Worth
revisiting only for few-step inversion.

BDIA (Zhang et al., ECCV 2024) achieves exact invertibility at 1 NFE per step instead of 2,
by making the update a linear combination of the two previous states. Worth implementing
after EDICT, and worth validating against it, since its leapfrog recurrence carries a
parasitic root whose stability is not analysed in the paper.
