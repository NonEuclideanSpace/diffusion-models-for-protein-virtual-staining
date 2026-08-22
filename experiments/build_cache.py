"""Build the training cache.

The original plan required downloading 187 GB of 1024x1024 crops onto local disk before any
training could start. At the resolution a diffusion model actually trains at, the same cells
cost 78 KB each instead of 4.27 MB - 55x less - so the whole set fits in under 5 GB and takes
about an hour to fetch. Nothing needs to touch the user's disk.

Downloads run wide because the bottleneck is bandwidth, not compute. Per-gene shards make the
run resumable, which this container requires.
"""
from __future__ import annotations

import argparse
import io
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from PIL import Image

S3 = "https://czi-subcell-public.s3.us-west-2.amazonaws.com"


def fetch_one(key: str, size: int, resample, retries: int = 3) -> np.ndarray | None:
    for attempt in range(retries):
        try:
            with urlopen(f"{S3}/{key}", timeout=90) as response:
                blob = response.read()
            image = Image.open(io.BytesIO(blob))
            if image.size != (size, size):
                image = image.resize((size, size), resample)
            return np.array(image)
        except Exception:
            if attempt == retries - 1:
                return None
            time.sleep(2 * (attempt + 1))
    return None


def fetch(stem: str, size: int, masks: bool = False):
    """The crop, and optionally its cell mask.

    Masks are what make per-cell state covariates possible: without one, integrated nuclear
    intensity picks up every neighbouring cell in the frame. They cost about a fifth again in
    download and use nearest-neighbour resampling so the boundary stays binary.
    """
    image = fetch_one(f"{stem}_cell_image.png", size, Image.BILINEAR)
    if image is None or not masks:
        return image, None
    mask = fetch_one(f"{stem}_cell_mask.png", size, Image.NEAREST)
    return image, (None if mask is None else (mask > 0).astype(np.uint8))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=Path("data/plans/u2os.txt"))
    parser.add_argument("--out", type=Path, default=Path("data/cache"))
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--masks", action="store_true",
                        help="also fetch cell masks, needed for per-cell state covariates")
    parser.add_argument("--cells-per-gene", type=int, default=0,
                        help="cap cells per gene; the full plan is 100+ GB of source crops")
    parser.add_argument("--max-genes", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    shards = args.out / "genes"
    shards.mkdir(exist_ok=True)

    order, meta, stems = [], {}, defaultdict(list)
    for line in args.plan.read_text().splitlines():
        gene, kind, location, line_name, stem = line.split("\t")
        if gene not in meta:
            order.append(gene)
            meta[gene] = (kind, location, line_name)
        stems[gene].append(stem)

    if args.cells_per_gene or args.max_genes:
        # a first training run does not need every cell, and the full plan would pull well over
        # 100 GB of source images before a single step
        rng = random.Random(args.seed)
        if args.cells_per_gene:
            for gene in order:
                rng.shuffle(stems[gene])
                stems[gene] = stems[gene][:args.cells_per_gene]
        if args.max_genes:
            order = order[:args.max_genes]
        print(f"subset: {len(order)} genes, {sum(len(stems[g]) for g in order)} cells", flush=True)

    remaining = [g for g in order if not (shards / f"{g}.npz").exists()]
    print(f"{len(order)} genes, {len(order) - len(remaining)} cached, {len(remaining)} to fetch",
          flush=True)

    started, cells, failures = time.time(), 0, 0
    with ThreadPoolExecutor(args.workers) as pool:
        for done, gene in enumerate(remaining, 1):
            arrays = list(pool.map(lambda s: fetch(s, args.size, args.masks), stems[gene]))
            pairs = [(a, m) for a, m in arrays if a is not None and (m is not None or not args.masks)]
            failures += len(arrays) - len(pairs)
            if not pairs:
                continue
            kind, location, line_name = meta[gene]
            payload = {"x": np.stack([a for a, _ in pairs]),
                       "meta": np.array([kind, location, line_name, str(args.size)])}
            if args.masks:
                payload["mask"] = np.stack([m for _, m in pairs])
            np.savez_compressed(shards / f"{gene}.npz", **payload)
            cells += len(pairs)
            if done % 10 == 0 or done == len(remaining):
                rate = cells / max(time.time() - started, 1e-9)
                size = sum(p.stat().st_size for p in shards.glob("*.npz")) / 1e9
                print(f"  {done}/{len(remaining)} genes  {cells} cells  {rate:.1f} cells/s  "
                      f"{size:.2f} GB  {failures} failed  "
                      f"eta {(len(remaining) - done) * 40 / max(rate, 1e-9) / 3600:.1f} h",
                      flush=True)

    total = sum(p.stat().st_size for p in shards.glob("*.npz")) / 1e9
    print(f"cache complete: {len(list(shards.glob('*.npz')))} genes, {total:.2f} GB")


if __name__ == "__main__":
    main()
