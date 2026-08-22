# The synthetic M1 rehearsal, and what it agrees with

`experiments/synthetic_m1.py` runs M1's whole chain on data whose conditional distribution is
known by construction: train a compartment classifier, train a conditional diffusion model,
sweep classifier-free guidance, and ask whether the calibration statistic detects the mode
collapse guidance is claimed to cause. A null here is a statement about the measurement rather
than about biology — which is the point of running it.

Reduced configuration, 2 CPU cores: 32px, 16 proteins, 2,500 steps, 4.18M-parameter DiT, 32
samples per protein per weight, 20 inference steps.

## The setup is sound

    compartment classifier accuracy 1.000, confusion matrix diagonal
    probe proteins by entropy: 1.504, 1.452, 1.445, 1.441, 1.432, 1.360, 1.277, 1.250 nats
    diffusion training loss 0.0735, held-out 0.0724

The classifier is perfect, so `phi` is not the weak link. The probe proteins carry 1.25-1.50
nats over five compartments, so the conditional really is multi-modal. Training and held-out
loss agree, so the model is not overfitting.

## The result: D_cal is large, and completely flat in guidance

| w | KL | Jensen-Shannon | total variation | Hellinger |
|---|---|---|---|---|
| 1.0 | 1.0791 | 0.1536 | 0.3677 | 0.4177 |
| 2.0 | 1.1125 | 0.1571 | 0.3690 | 0.4225 |
| 3.0 | 1.0898 | 0.1499 | 0.3671 | 0.4063 |
| 5.0 | 1.1058 | 0.1534 | 0.3723 | 0.4227 |

    trend statistics: kl +0.40   jensen_shannon -0.60   total_variation +0.40   hellinger +0.40

`D_cal` is **large** — total variation 0.368 means the model's compartment distribution is far
from the empirical one — and it **does not move with guidance**, changing in the fourth decimal
place while w goes from 1 to 5. The four divergences do not even agree on the sign of the trend.

M1's pre-registered gate requires a strictly increasing `D_bar[w]`. On this run it is flat.

## Two readings, and this run cannot separate them

1. **The model is too weak.** 2,500 steps on a 4.18M-parameter DiT at 32px is small. A model
   that has not learned the conditional well may not respond to guidance the way a trained one
   would, and `D_cal` of 0.368 could be fitting error rather than mode collapse.
2. **Guidance is not the mechanism.** The collapse is there — `D_cal` is large at every w — but
   guidance does not modulate it, which would mean it comes from the training objective rather
   than the sampler.

## Why reading 2 is worth taking seriously

It is the second independent hint in the same direction. `notes/celldiff-probe.md`: **CELL-Diff,
the SOTA model this project positions against, does not use classifier-free guidance at all.**
Its `sequence_to_image` takes no guidance weight and the `forward_with_cfg` in its DiT file is
vendored dead code nothing calls.

So: the SOTA model has no guidance, and a synthetic model with guidance shows a divergence that
does not respond to it. If both hold up, M1's mechanism claim — *guidance absorbs minor modes* —
is about a knob that neither the field's best model nor a working rehearsal actually turns.

That would not kill M1. `D_cal` being large is the interesting part and it survives. What it
kills is the framing, and the proposed fix: mode conditioning at sample time addresses a
sampler, and the cause would be in the loss.

## What resolves it

Train the synthetic model properly. This run took three hours on two CPU cores; on a GPU it is
minutes. **Do the rehearsal on the CUDA machine before the CELL-Diff probe** — it is cheaper,
it is fully controlled, and if `D_cal` still refuses to move with guidance on a well-trained
model, that reframes what the probe is even measuring.

    uv run python experiments/synthetic_m1.py --size 32 --proteins 32 --steps 30000 \
        --classifier-steps 4000 --samples 100 --inference-steps 50 --models dit unet \
        --out outputs/synthetic_m1_gpu

Report `D_bar[w]`, the trend statistic per divergence, and the held-out loss. The held-out loss
is what distinguishes reading 1 from reading 2: if it drops substantially below 0.072 and the
trend is still flat, reading 1 is out.

## A defect the rehearsal exposed in the gate statistic itself

Running the rehearsal at a deliberately tiny configuration produced a sweep that was *exactly*
flat — every guidance weight giving the same value, because the model was too small to respond
at all. `trend_statistic` scored it **+1.000**, identical to a perfectly rising sweep.

The cause: it ranked with `argsort().argsort()`, which is ordinal. On ties `argsort` is stable
and hands back the original order, so a constant sequence gets ranks 0,1,2,3 and correlates
perfectly with position.

M1's gate condition 1 is *"`D_bar[w]` is strictly increasing"*. **A sweep that did nothing would
have passed it.** The real run reported +0.40 on a nearly-flat sweep, which is close enough to
the failure mode to be uncomfortable.

Fixed: ties now take average ranks, and a sweep with no rank variance returns 0.0. Four tests
cover it, including the flat case and the sign symmetry of partial ties. This was worth more
than the rehearsal's own result.

## Separately: the measurement itself is validated

The flat trend above is not evidence that the estimator cannot see a collapse. `m1_power.py`
injects a *known* collapse into 100 real U2OS reference distributions and measures whether the
gate fires: **100% detection at 150 proteins, 50 real cells each**. The machinery works. What is
undetermined is whether guidance produces the thing it is able to detect.
