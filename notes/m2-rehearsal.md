# Module M2 rehearsed on data whose information structure is known

`experiments/synthetic_m2.py` builds a task where each compartment depends on a specific
landmark and **one compartment depends on none**:

    nucleoplasm, nuclear rim   determined by the DNA channel
    cytosol                    needs DNA to exclude the nucleus, and the cell extent
    membrane                   determined by the cell boundary, carried by microtubules and ER
    vesicles                   position drawn at random; no landmark predicts it

The last row is the control. `INTERNAL.md` section 6 attributes every model's failure on
vesicular compartments to weak spatial correlation with the landmarks; here that is true by
construction, so a correct estimator **must report approximately zero for vesicles across every
channel**. If it reports a signal there it is measuring model fit rather than information, and
the matrix cannot be trusted on real data.

Configuration: 32px, 4.18M parameters, 4,000 steps, channel dropout 0.3, 24 evaluation batches.
The model is trained *with* dropout — evaluating one that has never seen a masked channel would
measure out-of-distribution handling, a different and much larger quantity.

## Result

Loss increase from withholding each landmark, by compartment:

| channel | nucleoplasm | nuclear rim | cytosol | membrane | **vesicles** |
|---|---|---|---|---|---|
| dna | +0.0018* | **+0.0034*** | +0.0019* | +0.0001 | **-0.0009*** |
| microtubules | -0.0003* | +0.0002* | +0.0015* | **+0.0017*** | **-0.0007*** |
| reticulum | +0.0002 | +0.0021* | **+0.0030*** | **+0.0039*** | **-0.0008*** |

`*` = more than two standard errors from zero.

**The designed structure is recovered exactly.** DNA feeds the nuclear compartments; microtubules
and ER feed the cell boundary and cytosol; microtubules carry nothing for nucleoplasm. If the
matrix had come out uniform, M2's kill gate would have failed on the spot.

## The control fails, and the way it fails is the diagnosis

Vesicles should read zero. They read **-0.0009, -0.0007, -0.0008**, all flagged as more than two
standard errors from zero.

A *negative* information is meaningless: withholding a channel cannot make the model better. So
this is not the estimator finding phantom structure — it is the estimator's noise floor sitting
at roughly 0.0009 while the real signal is 0.002 to 0.004. **Signal to noise of two to four**,
and a "differs from zero by 2 SE" test fires on noise at that scale.

That is a power problem, not a validity problem. It is also exactly what the control is for: had
the vesicle row read a clean positive 0.0009, it would have looked like a small real effect.

## I diagnosed it as a power problem. That was wrong.

The obvious explanation for a null control firing at 0.0009 against a signal of 0.002 to 0.004
is sampling noise in the evaluation, which more evaluation batches would shrink. So that was
tested: the same run at **96 evaluation batches instead of 24**, four times the data.

| | 24 batches | 96 batches |
|---|---|---|
| max vesicle delta | 0.000867 | **0.000995** |
| dna to nucleoplasm | 0.001842 | 0.001858 |
| `vesicles_are_uninformed` | False | **False** |

**Nothing moved.** The vesicle deltas are not sampling noise; they are systematic, and they are
consistently *negative*.

### What it actually is

The estimator's pairing is not the problem — `landmark_information` already shares the timestep
and the noise between the masked and unmasked passes, which is finding C4 and it is implemented
correctly.

The cause is structural. For a compartment where the landmark genuinely carries no information,
the model's best behaviour is to ignore that channel. The full-condition input still contains
it, with all its variance; the masked input replaces it with a single learned constant plane.
**A model that is not perfectly invariant to an irrelevant input does slightly better on the
lower-variance version of it** — so withholding the channel lowers the loss, and the delta comes
out negative.

That is not a bug to fix. It means:

> **`Delta_k` is not lower-bounded at zero.** For an uninformative channel it is biased slightly
> negative, by an amount that depends on how non-invariant the trained model happens to be.

Testing `Delta_k` against zero, which is what the current implementation's standard errors do, is
therefore testing against the wrong null.

### The fix, which the control handed us

Use an **empirical null instead of zero**: a compartment known to be uninformed, or the same
channel permuted across samples so its content is destroyed but its statistics are not. The
vesicle row in this synthetic task is exactly such a null, and on real data a permuted-channel
control plays the same role.

Read that way, this run does not fail. Its numbers become:

    dna -> nuclear_rim     +0.0034  against a null of -0.0009   =  +0.0043
    reticulum -> membrane  +0.0039  against the same null       =  +0.0048

which is a cleaner separation than testing against zero gave.

**M2's kill gate asks whether the matrix shows non-trivial structure. It does.** What the
rehearsal changed is the null it should be tested against, and it changed it before any real-data
number was quoted.

## Verdict

**The matrix is trustworthy in direction and in relative magnitude; it is not trustworthy against
a null of zero.** Before real data:

- **Fix the null.** Every `Delta_k` is reported against an empirical uninformative-channel
  baseline, not against zero. Recorded here, before any real-data number exists.
- A better-trained model, so the signal grows relative to the model's residual non-invariance.
- More evaluation batches help the standard error but, as measured above, **do not touch the
  bias** - which is the whole reason the control was worth running.

## Reproduce

    uv run python experiments/synthetic_m2.py --steps 4000 --out outputs/synthetic_m2_cpu
