"""Phase 1 training entry point.

One command from the cache to a checkpoint. The defaults are the configuration this project
intends to run on 4-8 A100s; --smoke shrinks every dimension so the whole path can be exercised
on a laptop CPU in under a minute, which is the only way to know the GPU run will start.
"""
from __future__ import annotations

import argparse
import json
from itertools import cycle
from pathlib import Path

import torch

from pvs.data.cache import STATE_NAMES, CachedCrops, ShardShuffleSampler
from pvs.diffusion.schedule import NoiseSchedule, cosine_beta_schedule, enforce_zero_terminal_snr
from pvs.models.dit import DiT
from pvs.train.loop import TrainConfig, Trainer, resolve_device

# A100-days at 200k steps x batch 64, assuming 40% MFU in bf16. The patch size is the number
# that matters: at 256px a 16-pixel patch is too coarse to resolve punctate structure, which is
# most of what distinguishes vesicles from cytosol, so "mid" and not "base" is the target.
PRESETS = {
    "smoke": dict(image_size=32, patch_size=4, hidden_size=64, depth=2, heads=2),
    "small": dict(image_size=128, patch_size=8, hidden_size=384, depth=12, heads=6),     # 0.05 d
    "base": dict(image_size=256, patch_size=16, hidden_size=768, depth=12, heads=12),    # 0.17 d
    "mid": dict(image_size=256, patch_size=8, hidden_size=768, depth=12, heads=12),      # 0.89 d
    "large": dict(image_size=256, patch_size=8, hidden_size=1024, depth=24, heads=16),   # 2.94 d
}


def batches(dataset:CachedCrops, batch_size:int, workers:int, seed:int):
    sampler = ShardShuffleSampler(dataset, seed=seed)
    epoch = 0
    while True:
        sampler.set_epoch(epoch)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, sampler=sampler, num_workers=workers,
            drop_last=True, pin_memory=torch.cuda.is_available(),
            persistent_workers=False,
        )
        yield from loader
        epoch += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path("data/cache"))
    parser.add_argument("--preset", choices=sorted(PRESETS), default="mid")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("outputs/phase1"))
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--target", default="v")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--genes", type=int, default=0)
    parser.add_argument("--state", action="store_true",
                        help="condition on cell-state covariates taken from the landmark channels")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.smoke:
        # Explicit flags win over the smoke preset, so --smoke --steps 20 means twenty steps.
        given = {a.lstrip("-").replace("-", "_") for a in __import__("sys").argv if a.startswith("--")}
        for name, value in (("preset", "smoke"), ("steps", 12), ("batch", 4), ("workers", 0),
                            ("timesteps", 40), ("genes", 3)):
            if name not in given:
                setattr(args, name, value)
        if "out" not in given:
            args.out = args.out.with_name(args.out.name + "-smoke")

    preset = dict(PRESETS[args.preset])
    size = preset.pop("image_size")

    names = sorted(p.stem for p in (args.cache / "genes").glob("*.npz"))
    if args.genes:
        names = names[: args.genes]
    dataset = CachedCrops(args.cache, size=size, genes=names or None, state=args.state)

    model = DiT(image_size=size, target_channels=1, condition_channels=3,
                num_classes=None, num_state=len(STATE_NAMES) if args.state else None,
                **preset)
    parameters = sum(p.numel() for p in model.parameters())

    betas = enforce_zero_terminal_snr(cosine_beta_schedule(args.timesteps))
    schedule = NoiseSchedule(betas)

    config = TrainConfig(
        steps=args.steps, batch_size=args.batch, learning_rate=args.lr,
        accumulation=args.accumulation, target=args.target, device=args.device,
        seed=args.seed, output_dir=str(args.out),
        log_every=4 if args.smoke else 100,
        checkpoint_every=8 if args.smoke else 5_000,
        amp=not args.smoke,
        extra={"preset": args.preset, "image_size": size, "genes": len(dataset.genes),
               "cells": len(dataset), "parameters": parameters, "state": args.state},
    )

    device = resolve_device(args.device)
    print(json.dumps({"preset": args.preset, "image_size": size, "parameters": parameters,
                      "genes": len(dataset.genes), "cells": len(dataset),
                      "device": str(device), "target": args.target,
                      "state": args.state}), flush=True)

    trainer = Trainer(model, schedule, batches(dataset, args.batch, args.workers, args.seed), config)
    if args.resume and Path(args.resume).exists():
        trainer.load(args.resume)
        print(f"resumed at step {trainer.step}", flush=True)
    elif (args.out / "latest.pt").exists():
        trainer.load(args.out / "latest.pt")
        print(f"resumed at step {trainer.step}", flush=True)
    trainer.run()
    print(f"done at step {trainer.step}, checkpoint {args.out / 'latest.pt'}")


if __name__ == "__main__":
    main()
