"""Read probe A's output and decide whether the flag separates the groups.

The headline quantity is not entropy. It is the gap

    I(cell ; compartment) = H(mean prediction) - mean(per-cell entropy)

which is the mutual information between which cell was imaged and which compartment was
predicted — that is, the **between-cell** variability, with each cell's own uncertainty
removed. A classifier that is uncertain in the same way about every cell has a high entropy
and zero heterogeneity; only the gap distinguishes them, and heterogeneity is what module M1
needs.

Effect sizes come with bootstrap intervals because 20 proteins per group is few, and a point
estimate at that size invites over-reading.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from pvs.eval.localization import GROUP_NAMES, to_groups


def mann_whitney(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    ranks = torch.cat([a, b]).argsort().argsort().float() + 1
    n, m = len(a), len(b)
    u = ranks[:n].sum().item() - n * (n + 1) / 2
    return u / (n * m), (u - n * m / 2) / max((n * m * (n + m + 1) / 12) ** 0.5, 1e-9)


def bootstrap_effect(a: torch.Tensor, b: torch.Tensor, draws: int = 4000, seed: int = 0) -> tuple[float, float]:
    generator = torch.Generator().manual_seed(seed)
    values = []
    for _ in range(draws):
        ra = a[torch.randint(len(a), (len(a),), generator=generator)]
        rb = b[torch.randint(len(b), (len(b),), generator=generator)]
        values.append(mann_whitney(ra, rb)[0])
    sorted_values = torch.tensor(values).sort().values
    return float(sorted_values[int(0.025 * draws)]), float(sorted_values[int(0.975 * draws)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("outputs/probe_a/results.json"))
    args = parser.parse_args()
    payload = json.loads(args.results.read_text())
    genes = payload["genes"]

    rows = []
    for gene, record in genes.items():
        probability = torch.tensor(record["mean_probability"])
        heterogeneity = record["entropy_of_mean"] - record["mean_cell_entropy"]
        grouped = to_groups(probability)
        rows.append({
            "gene": gene, "kind": record["kind"], "location": record["location"],
            "cells": record["cells"], "heterogeneity": heterogeneity,
            "entropy_of_mean": record["entropy_of_mean"],
            "top_group": GROUP_NAMES[int(grouped.argmax())],
            "top_group_p": float(grouped.max()),
            "spread": record["embedding_spread"],
        })
    rows.sort(key=lambda r: -r["heterogeneity"])

    print(f"{'gene':<11}{'group':<10}{'between-cell':>13}{'H(mean)':>9}{'top compartment':>22}{'p':>7}")
    for row in rows:
        print(f"{row['gene']:<11}{row['kind']:<10}{row['heterogeneity']:>13.4f}"
              f"{row['entropy_of_mean']:>9.3f}{row['top_group']:>22}{row['top_group_p']:>7.2f}")

    print("\n=== does the HPA flag separate the groups ===")
    for field in ("heterogeneity", "entropy_of_mean", "spread"):
        positive = torch.tensor([r[field] for r in rows if r["kind"] == "positive"])
        control = torch.tensor([r[field] for r in rows if r["kind"] == "control"])
        effect, z = mann_whitney(positive, control)
        low, high = bootstrap_effect(positive, control)
        verdict = "separates" if low > 0.5 else ("separates the other way" if high < 0.5 else "no separation")
        print(f"  {field:<18} positive {positive.median():.4f}  control {control.median():.4f}  "
              f"effect {effect:.3f} [{low:.3f}, {high:.3f}]  z {z:+.2f}   {verdict}")

    print("\n  An effect size of 0.5 is no separation. The interval is a 95% bootstrap over")
    print("  proteins; with 20 per group it is wide, and a point estimate alone means little.")

    heterogeneity = torch.tensor([r["heterogeneity"] for r in rows])
    print(f"\n  between-cell heterogeneity overall: median {heterogeneity.median():.4f}, "
          f"max {heterogeneity.max():.4f} nats")
    print("  For scale, a uniform distribution over the 31 classes has entropy 3.434 nats.")
    print("  If this is near zero for every protein, SubCell reports the same distribution for")
    print("  every cell and pi_emp carries no within-protein information, whatever the flag says.")


if __name__ == "__main__":
    main()
