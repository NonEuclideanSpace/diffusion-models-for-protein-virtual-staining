"""Probe A over a larger gene set, streaming so disk is not the constraint.

Crops are 4.27 MB each and the container has 14 GB free, so a 4640-cell run cannot be
downloaded first and processed afterwards. Downloads run ahead of inference in a bounded
queue, each crop is deleted once embedded, and per-gene results are written as they complete
so a killed process loses one gene rather than the run — this container has reaped every
long-lived background job so far.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time
from collections import defaultdict
from pathlib import Path
from urllib.request import urlopen

import torch

from pvs.data.crops import load_crop, resize
from pvs.eval.subcell import IMAGE_SIZE, SubCellEnsemble, subcell_input

S3 = "https://czi-subcell-public.s3.us-west-2.amazonaws.com"


def download(key: str, destination: Path, retries: int = 3) -> bool:
    for attempt in range(retries):
        try:
            with urlopen(f"{S3}/{key}", timeout=60) as response:
                destination.write_bytes(response.read())
            return True
        except Exception:
            if attempt == retries - 1:
                return False
            time.sleep(2 * (attempt + 1))
    return False


def producer(genes: list[tuple[str, list[str]]], scratch: Path, pending: queue.Queue, workers: int) -> None:
    from concurrent.futures import ThreadPoolExecutor

    for gene, stems in genes:
        folder = scratch / gene
        folder.mkdir(parents=True, exist_ok=True)
        jobs = [(f"{stem}_cell_image.png", folder / f"{os.path.basename(stem)}.png") for stem in stems]
        with ThreadPoolExecutor(workers) as pool:
            ok = list(pool.map(lambda j: download(*j), jobs))
        pending.put((gene, [path for (_, path), good in zip(jobs, ok) if good]))
    pending.put(None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=Path("data/plans/plan2.txt"))
    parser.add_argument("--scratch", type=Path, default=Path("data/stream"))
    parser.add_argument("--out", type=Path, default=Path("outputs/probe_a_large"))
    parser.add_argument("--encoder", type=Path,
                        default=Path("data/subcell/all_channels_ViT-ProtS-Pool.pth"))
    parser.add_argument("--classifiers", type=Path, default=Path("data/subcell/classifiers"))
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--download-workers", type=int, default=12)
    parser.add_argument("--queue", type=int, default=2)
    parser.add_argument("--save-embeddings", action="store_true", default=True)
    parser.add_argument("--device", default="auto",
                        help="auto picks mps on Apple silicon, then cuda, then cpu")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    args.scratch.mkdir(parents=True, exist_ok=True)
    handle = (args.out / "log.txt").open("a")

    def log(message: str) -> None:
        print(message, flush=True)
        handle.write(message + "\n")
        handle.flush()

    torch.set_num_threads(args.threads)
    if args.device == "auto":
        args.device = ("mps" if torch.backends.mps.is_available()
                       else "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(args.device)
    order, meta, stems = [], {}, defaultdict(list)
    for line in args.plan.read_text().splitlines():
        gene, kind, location, _, stem = line.split("\t")
        if gene not in meta:
            order.append(gene)
            meta[gene] = (kind, location)
        stems[gene].append(stem)

    results_path = args.out / "results.json"
    done = json.loads(results_path.read_text()) if results_path.exists() else {}
    remaining = [(g, stems[g]) for g in order if g not in done]
    total = sum(len(s) for _, s in remaining)
    log(f"=== probe A, streaming === {len(remaining)} genes to do "
        f"({len(done)} already), {total} cells")

    model = SubCellEnsemble.from_directory(args.encoder, args.classifiers).eval().to(device)
    log(f"device: {device}")

    pending: queue.Queue = queue.Queue(maxsize=args.queue)
    threading.Thread(target=producer, args=(remaining, args.scratch, pending, args.download_workers),
                     daemon=True).start()

    started, processed = time.time(), 0
    while True:
        item = pending.get()
        if item is None:
            break
        gene, paths = item
        probabilities, embeddings = [], []
        for start in range(0, len(paths), args.batch):
            chunk = paths[start:start + args.batch]
            try:
                batch = torch.stack([resize(subcell_input(load_crop(p)), IMAGE_SIZE)
                                     for p in chunk]).to(device)
            except Exception as error:
                log(f"  skip batch in {gene}: {error}")
                continue
            with torch.no_grad():
                embedding, probability = model(batch)
            probabilities.append(probability.cpu())
            embeddings.append(embedding.cpu())
        for path in paths:
            path.unlink(missing_ok=True)
        (args.scratch / gene).rmdir()
        if not probabilities:
            continue

        probability = torch.cat(probabilities)
        embedding = torch.cat(embeddings)
        kind, location = meta[gene]
        if args.save_embeddings:
            # The embeddings are the escape route from the classifier head, where the
            # circularity lives: modes can be defined by clustering them instead of by
            # classifying. 40 cells by 1536 dimensions is 250 KB per gene.
            store = args.out / "embeddings"
            store.mkdir(exist_ok=True)
            torch.save({"embedding": embedding.half(), "probability": probability.half(),
                        "kind": kind, "location": location}, store / f"{gene}.pt")
        entropy = lambda p: float(-(p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).sum(-1).mean())
        done[gene] = {
            "kind": kind, "location": location, "cells": len(probability),
            "mean_probability": probability.mean(0).tolist(),
            "entropy_of_mean": entropy(probability.mean(0)[None]),
            "mean_cell_entropy": entropy(probability),
            "embedding_spread": float((embedding - embedding.mean(0)).norm(dim=1).mean()),
        }
        results_path.write_text(json.dumps(done, indent=1))
        processed += len(probability)
        rate = processed / (time.time() - started)
        gap = done[gene]["entropy_of_mean"] - done[gene]["mean_cell_entropy"]
        log(f"  {gene:<11}{kind:<9}n={len(probability):<3} between-cell={gap:.4f} "
            f"| {len(done)}/{len(order)} genes  {rate:.2f} cells/s "
            f"eta {(total - processed) / max(rate, 1e-9) / 60:.0f} min")

    log(f"\ncomplete: {len(done)} genes")


if __name__ == "__main__":
    main()
