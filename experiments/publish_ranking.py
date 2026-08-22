"""The screen's deliverable: 588 U2OS genes ranked by measured localization heterogeneity.

The point of the ranking is that it is data-driven. None of HPA's three single-cell annotation
columns predicts it (`notes/heterogeneity-selector.md`), so a list of genes whose position
genuinely varies between cells has to be measured rather than looked up. This writes that list
in a form somebody else can use.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from pvs.eval.localization import CLASS_NAMES, to_groups

FLAGS=(("S","spatial"),("I","intensity"),("C","cell_cycle"))


def entropy(p:torch.Tensor)->torch.Tensor:
    return -(p.clamp_min(1e-12)*p.clamp_min(1e-12).log()).sum(-1)


def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--results",type=Path,default=Path("outputs/u2os_screen/results.json"))
    parser.add_argument("--out",type=Path,default=Path("outputs/u2os_heterogeneity_ranking.tsv"))
    args=parser.parse_args()

    records=json.loads(args.results.read_text())
    rows=[]
    for gene,record in records.items():
        mean=torch.tensor(record["mean_probability"])
        grouped=to_groups(mean[None])[0]
        grouped=grouped/grouped.sum().clamp_min(1e-12)
        marks=record["kind"]
        rows.append({
            "gene":gene,
            "heterogeneity":round(record["entropy_of_mean"]-record["mean_cell_entropy"],5),
            "cells":record["cells"],
            "predicted":CLASS_NAMES[int(mean.argmax())],
            "predicted_confidence":round(float(mean.max()),4),
            "coarse_entropy":round(float(entropy(grouped)),4),
            "negative_mass":round(float(mean[CLASS_NAMES.index("Negative")]),4),
            "hpa_location":record["location"],
            **{name:int(letter in marks) for letter,name in FLAGS},
        })
    rows.sort(key=lambda r:-r["heterogeneity"])
    for rank,row in enumerate(rows,1):
        row["rank"]=rank

    columns=["rank","gene","heterogeneity","cells","predicted","predicted_confidence",
             "coarse_entropy","negative_mass","spatial","intensity","cell_cycle","hpa_location"]
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=columns,delimiter="\t")
        writer.writeheader()
        writer.writerows({k:r[k] for k in columns} for r in rows)

    values=torch.tensor([r["heterogeneity"] for r in rows])
    print(f"wrote {len(rows)} genes -> {args.out}")
    print(f"  heterogeneity  median {values.median():.4f}  p90 {values.quantile(0.9):.4f}  "
          f"max {values.max():.4f}")
    print(f"  top 10:")
    for row in rows[:10]:
        print(f"    {row['rank']:>3}  {row['gene']:<10} {row['heterogeneity']:.4f}  "
              f"{row['predicted']:<16} {row['hpa_location'][:38]}")


if __name__=="__main__":
    main()
