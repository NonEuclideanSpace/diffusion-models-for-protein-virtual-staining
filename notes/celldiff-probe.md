# The pre-registered M1 probe, and what reading CELL-Diff's code changed

`INTERNAL.md` section 4 specifies a probe to run **before writing any training code**: sample
public CELL-Diff weights for high-entropy proteins, estimate `pi_hat`, compare to `pi_emp`.
"D_cal already small -> module is dead."

It was never run. This note records why it is still the right next step, what it costs, and one
thing about it that has to change.

## 1. The weights are public and small enough

`s3://czi-celldiff-public/v2/`, MIT.

| object | size |
|---|---|
| `checkpoints/cell_diff/hpa_pretrained_all.bin` | 6.69 GB |
| `checkpoints/vae/hpa_pretrained.bin` | 0.17 GB |
| training LMDBs | 101 GB — **not needed** |

6.9 GB, about three minutes at the bandwidth measured here. Code: `BoHuangLab/CELL-Diff`,
clones and reads cleanly.

## 2. The API is exactly the one this project needs

    cell_img_latent = cat(vae.encode(nucleus), vae.encode(ER), vae.encode(microtubule))
    sample = model.sequence_to_image(sequence_token, cell_img_latent, sampling_strategy="ddim")
    image  = vae.decode(sample).sample

Three landmark channels in, protein channel out, conditioned on the ESM-2 embedding of the
protein sequence. That is the same task and the same conditioning structure as Phase 1 here, so
`pi_hat` from CELL-Diff and `pi_emp` from HPA are directly comparable through the same SubCell
classifier.

## 3. **CELL-Diff does not use classifier-free guidance**

`sequence_to_image` takes no guidance weight, and steps the full 200 timesteps with a single
network call each. `forward_with_cfg` exists in `modules/dit.py` but that file is vendored from
the original Meta DiT repo, keyed on class labels, and **is never called** — `self.net` is
`CELLDiff`, not `DiT`.

This matters more than it looks.

M1's hypothesis is that `D_cal` **rises monotonically with CFG weight w**, and that the
mechanism is guidance absorbing minor modes. The SOTA model the project positions itself
against does not use guidance at all. Two consequences:

- **The guidance sweep cannot be run on CELL-Diff.** It requires a model trained with
  conditioning dropout, i.e. ours. The probe on CELL-Diff answers a different and prior
  question: is `D_cal` large *at the model's native operating point*.
- **If `D_cal` is large without any guidance, guidance is not the mechanism.** Mode collapse
  would then be a property of conditional diffusion training itself rather than of the sampler,
  which is a broader claim than the one pre-registered — and it would mean a sample-time fix
  (mode conditioning) is addressing a symptom whose cause sits in the loss.

Either outcome is worth knowing before a training run, and neither is knowable without the
probe.

## 4. Cost, measured rather than guessed

The UNet is Stable-Diffusion-1.5 scale: `block_out_channels` 320/640/1280/1280, 2 layers per
block, 64x64 latent, 4 latent channels, ~340 GFLOP per forward.

| where | per 200-step sample | 20 proteins x 100 samples |
|---|---|---|
| 2 CPU cores here | ~1 hour | ~2,000 hours — **infeasible** |
| one A100, batch 16 | ~0.7 s | **~25 minutes** |

**The decisive experiment costs well under one A100-hour.** That is a different order of ask
from the ~1 A100-day Phase 1 training run, and it comes first: it is what tells us whether the
training run is worth paying for.

## 4b. The dependency list in `install.sh` is not what inference needs

`install.sh` installs twelve packages including `wandb`, `lmdb`, `frc` and `pytorch-fid`. Those
are for training and FID evaluation; the inference path needs far less. What actually imports:

    diffusers loguru einops timm fair-esm torchvision
    requests urllib3 charset-normalizer idna platformdirs click

83 MB, installed with `--no-deps` into a separate directory so the project's own environment is
untouched. Installing without `--no-deps` pulls a second full CUDA torch — 4.7 GB.

`wandb` is imported at module scope in `logging/loggers.py` and `utils/cli_utils.py` but only
ever called as `wandb.init`, `wandb.log`, `wandb.run`. The real package drags in pydantic,
sentry and gitpython and conflicts with this environment. A nine-line stub with those three
names is enough, and `celldiff_probe.py` writes it itself.

## 4c. Two things that make the run cheaper than expected

- **`timestep_respacing` is already a config field**, wired to the standard `SpacedDiffusion`
  from guided-diffusion. `sequence_to_image` steps `range(self.diffusion.num_timesteps)`, so
  setting it to `"50"` cuts 200 steps to 50 with no change to their code.
- **CELL-Diff's own example inputs are 256x256**, the same resolution as this project's
  training cache. `sample_size=64` in the shipped config refers to the latent grid of a 512px
  variant; the VAE encodes whatever it is given.

## 5. What to run

1. Pick 20 proteins by measured `H(pi_emp)` from the U2OS screen, not by annotation
   (`heterogeneity-selector.md`).
2. For each, 100 samples conditioned on real HPA landmark channels from held-out cells.
3. Classify every generated image with the same SubCell ensemble used for `pi_emp`.
4. Report `D_cal` against the null distribution in `eval/calibration.py`, at 7 groups
   (`Grouping 3`), per the protocol.
5. Gate: `D_cal` indistinguishable from the null -> M1 is dead, say so, move to M2/M3.

Steps 1 and 3-5 are written. Step 2 is the GPU hour.

## 6. Status of the other modules, for the same accounting

| module | estimator | harness | run |
|---|---|---|---|
| M1 | done, tested | needs a model | blocked; power calculation void |
| M2 | done, tested (`eval/information.py`) | needs a model | blocked |
| M3 | **nothing written** | — | its kill gate — validating spot detection on *real* images — is CPU-only and could be done now |
