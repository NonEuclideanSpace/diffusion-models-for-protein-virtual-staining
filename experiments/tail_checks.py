"""Three checks on the heavy tail of measured heterogeneity.

The classifier head can manufacture heterogeneity: nucleoplasm and nucleoli are adjacent
classes and a boundary that wobbles cell to cell looks like biological variability. Each check
below is an attempt to make the tail go away.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch

from pvs.eval.localization import CLASS_NAMES, GROUPING_3, to_groups


def correlation(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-12))


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    rank = lambda x: x.argsort().argsort().float()
    return correlation(rank(a), rank(b))


def information(probability: torch.Tensor) -> float:
    entropy = lambda p: -(p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).sum(-1)
    return float(entropy(probability.mean(0)) - entropy(probability).mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("outputs/probe_a_large"))
    args = parser.parse_args()

    records = json.loads((args.out / "results.json").read_text())
    store = args.out / "embeddings"
    genes = sorted(p.stem for p in store.glob("*.pt"))

    head, spread, coarse, agreement = [], [], [], []
    for gene in genes:
        blob = torch.load(store / f"{gene}.pt", map_location="cpu")
        probability = blob["probability"].float()
        embedding = blob["embedding"].float()
        head.append(information(probability))
        centred = embedding - embedding.mean(0)
        spread.append(float(centred.norm(dim=1).mean() / embedding.norm(dim=1).mean()))
        coarse.append(information(to_groups(probability)))
        label = probability.argmax(1)
        agreement.append(float((label == label.mode().values).float().mean()))
    head = torch.tensor(head)
    spread = torch.tensor(spread)
    coarse = torch.tensor(coarse)
    agreement = torch.tensor(agreement)

    print(f"=== check 1: is the tail visible without the classifier head? ===")
    print(f"  {len(genes)} genes, head-based information against embedding spread")
    print(f"  pearson {correlation(head, spread):+.3f}   spearman {spearman(head, spread):+.3f}")
    print(f"  {'the embedding sees it too' if spearman(head, spread) > 0.4 else 'the head is inventing it'}")

    print(f"\n=== check 2: does it survive coarse grouping? ===")
    print(f"  31 classes collapsed to {len(set(GROUPING_3.values())) + 1} groups, "
          f"so nucleoplasm/nucleoli wobble cannot count")
    print(f"  fine   mean {head.mean():.4f}  p90 {head.quantile(0.9):.4f}  max {head.max():.4f}")
    print(f"  coarse mean {coarse.mean():.4f}  p90 {coarse.quantile(0.9):.4f}  max {coarse.max():.4f}")
    print(f"  retained {float(coarse.mean() / head.mean()):.0%} of the mean, "
          f"rank correlation {spearman(head, coarse):+.3f}")

    print(f"\n=== check 3: is it one population or two? ===")
    print(f"  fraction of cells agreeing with the gene's modal call")
    order = head.argsort(descending=True)
    print(f"  top decile   {agreement[order[:len(genes)//10]].mean():.2f}")
    print(f"  bottom decile{agreement[order[-len(genes)//10:]].mean():.2f}")
    print(f"  correlation with heterogeneity {correlation(head, agreement):+.3f}")
    split = [g for g, a, h in zip(genes, agreement.tolist(), head.tolist())
             if a < 0.7 and h > float(head.quantile(0.8))]
    print(f"  genes with a genuine minority state (<70% modal, top-quintile "
          f"heterogeneity): {len(split)}")
    for gene in split[:10]:
        blob = torch.load(store / f"{gene}.pt", map_location="cpu")
        counts = Counter(blob["probability"].float().argmax(1).tolist())
        top = [(CLASS_NAMES[c], n) for c, n in counts.most_common(3)]
        print(f"    {gene:<10} {records[gene]['kind']:<9} "
              f"{'  '.join(f'{name} {n}' for name, n in top)}")


if __name__ == "__main__":
    main()
