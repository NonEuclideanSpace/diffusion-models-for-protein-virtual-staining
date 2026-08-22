"""Probe A: does a compartment classifier see heterogeneity that HPA annotators flagged?

Module M1 needs a per-protein distribution over compartments across individual cells. HPA
publishes no per-cell labels, so that distribution has to be predicted by SubCell — whose
classifier heads were themselves trained on labels broadcast identically to every cell in a
field of view. If that training regressed cells toward the consensus, the predicted
distributions will understate exactly the heterogeneity M1 exists to measure, and the module
has no measurable target.

HPA independently flags 186 genes as varying spatially at the single-cell level. That flag is
curated by annotators and owes nothing to SubCell. So: run SubCell over flagged genes and
over controls matched on both reliability and main location, and test whether the predicted
per-gene entropy separates them.

Matching on main location is what makes the test mean something. Without it a positive result
would be consistent with nothing more than "multi-located proteins get higher entropy", which
is true by construction.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch

from pvs.data.crops import load_crop, resize
from pvs.eval.subcell import IMAGE_SIZE, SubCellEnsemble, subcell_input


def entropy(probabilities: torch.Tensor) -> torch.Tensor:
    p = probabilities.clamp_min(1e-12)
    return -(p * p.log()).sum(-1)


def mann_whitney(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    """Rank-sum test. Returns the common-language effect size and a normal-approximation z."""
    combined = torch.cat([a, b])
    ranks = combined.argsort().argsort().float() + 1
    rank_sum = ranks[: len(a)].sum().item()
    n, m = len(a), len(b)
    u = rank_sum - n * (n + 1) / 2
    mean, sd = n * m / 2, (n * m * (n + m + 1) / 12) ** 0.5
    return u / (n * m), (u - mean) / max(sd, 1e-9)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crops", type=Path, default=Path("/home/claude/data/subset/raw"))
    parser.add_argument("--plan", type=Path, default=Path("/home/claude/data/subset/plan.txt"))
    parser.add_argument("--encoder", type=Path,
                        default=Path("/home/claude/data/subcell/all_channels_ViT-ProtS-Pool.pth"))
    parser.add_argument("--classifiers", type=Path,
                        default=Path("/home/claude/data/subcell/classifiers"))
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit-per-gene", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("outputs/probe_a"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    handle = (args.out / "log.txt").open("a")

    def log(message: str) -> None:
        print(message, flush=True)
        handle.write(message + "\n")
        handle.flush()

    torch.set_num_threads(args.threads)
    kind, location = {}, {}
    for line in args.plan.read_text().splitlines():
        gene, group, loc, _, _ = line.split("\t")
        kind[gene], location[gene] = group, loc

    files = defaultdict(list)
    for path in sorted(args.crops.glob("*/*_cell_image.png")):
        files[path.parent.name].append(path)
    if args.limit_per_gene:
        files = {g: p[: args.limit_per_gene] for g, p in files.items()}
    total = sum(len(v) for v in files.values())
    log(f"=== probe A === {len(files)} genes, {total} cells, device {args.device}")

    model = SubCellEnsemble.from_directory(args.encoder, args.classifiers).to(args.device).eval()

    results, started, done = {}, time.time(), 0
    for gene, paths in sorted(files.items()):
        probabilities, embeddings = [], []
        for start in range(0, len(paths), args.batch):
            batch = torch.stack([
                resize(subcell_input(load_crop(p)), IMAGE_SIZE) for p in paths[start:start + args.batch]
            ]).to(args.device)
            with torch.no_grad():
                embedding, probability = model(batch)
            probabilities.append(probability.cpu())
            embeddings.append(embedding.cpu())
            done += len(batch)
        probability = torch.cat(probabilities)
        results[gene] = {
            "kind": kind.get(gene, "unknown"),
            "location": location.get(gene, ""),
            "cells": len(paths),
            "mean_probability": probability.mean(0).tolist(),
            "entropy_of_mean": float(entropy(probability.mean(0))),
            "mean_cell_entropy": float(entropy(probability).mean()),
            "argmax_fractions": torch.bincount(probability.argmax(-1),
                                               minlength=probability.shape[1]).float().div(len(paths)).tolist(),
            "embedding_spread": float((torch.cat(embeddings) - torch.cat(embeddings).mean(0)).norm(dim=1).mean()),
        }
        rate = done / (time.time() - started)
        log(f"  {gene:<10} {kind.get(gene,'?'):<8} n={len(paths):<4} "
            f"H(mean)={results[gene]['entropy_of_mean']:.3f} "
            f"H_cell={results[gene]['mean_cell_entropy']:.3f} "
            f"| {done}/{total} {rate:.2f} cells/s eta {(total-done)/max(rate,1e-9)/60:.0f} min")

    positive = torch.tensor([r["entropy_of_mean"] for r in results.values() if r["kind"] == "positive"])
    control = torch.tensor([r["entropy_of_mean"] for r in results.values() if r["kind"] == "control"])
    within_positive = torch.tensor([r["mean_cell_entropy"] for r in results.values() if r["kind"] == "positive"])
    within_control = torch.tensor([r["mean_cell_entropy"] for r in results.values() if r["kind"] == "control"])

    log("\n=== does the flag separate the groups ===")
    verdict = {}
    for name, a, b in (("entropy_of_mean", positive, control),
                       ("mean_cell_entropy", within_positive, within_control)):
        effect, z = mann_whitney(a, b)
        verdict[name] = {"positive_median": float(a.median()), "control_median": float(b.median()),
                         "effect_size": effect, "z": z, "n_positive": len(a), "n_control": len(b)}
        log(f"  {name:<20} positive {a.median():.4f}  control {b.median():.4f}  "
            f"effect {effect:.3f}  z {z:+.2f}")
    log("\n  effect size 0.5 means no separation. Above about 0.65 with |z| over 2 would say")
    log("  SubCell detects heterogeneity the annotators independently saw.")

    (args.out / "results.json").write_text(json.dumps({"genes": results, "verdict": verdict}, indent=2))
    log(f"\nwritten to {args.out / 'results.json'}")


if __name__ == "__main__":
    main()
