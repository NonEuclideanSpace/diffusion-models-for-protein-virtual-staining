"""Probe A's flagged group was built by OR-ing two HPA columns that mean different things.

`Single-cell variation spatial` marks proteins that sit in different *places* in different
cells. `Single-cell variation intensity` marks proteins present at different *levels*. Only the
first is the construct this project measures, and it is twenty times rarer, so the OR produces
a group that is roughly 95% intensity genes. This asks whether the null in
`heterogeneity-selector.md` is a null about localization variability or an artefact of that OR.

`Cell cycle dependency` is a third column, tested here as an independent candidate selector.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch


def mann_whitney(a:torch.Tensor,b:torch.Tensor)->tuple[float,float]:
    ranks=torch.cat([a,b]).argsort().argsort().float()+1
    n,m=len(a),len(b)
    u=ranks[:n].sum().item()-n*(n+1)/2
    return u/(n*m),(u-n*m/2)/max((n*m*(n+m+1)/12)**0.5,1e-9)


def bootstrap(a:torch.Tensor,b:torch.Tensor,draws:int=4000,seed:int=0)->tuple[float,float]:
    generator=torch.Generator().manual_seed(seed)
    values=torch.tensor([
        mann_whitney(a[torch.randint(len(a),(len(a),),generator=generator)],
                     b[torch.randint(len(b),(len(b),),generator=generator)])[0]
        for _ in range(draws)]).sort().values
    return float(values[int(0.025*draws)]),float(values[int(0.975*draws)])


def compare(name:str,flagged:torch.Tensor,control:torch.Tensor)->None:
    if len(flagged)<3 or len(control)<3:
        print(f"  {name:<30} {len(flagged):>3} v {len(control):>3}   too few to test")
        return
    effect,z=mann_whitney(flagged,control)
    low,high=bootstrap(flagged,control)
    verdict=("separates" if low>0.5 else "reversed" if high<0.5 else "not resolved")
    print(f"  {name:<30} {len(flagged):>3} v {len(control):>3}   "
          f"{flagged.mean():.4f} v {control.mean():.4f}   "
          f"effect {effect:.3f} [{low:.3f}, {high:.3f}]  z {z:+.2f}   {verdict}")


def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--results",type=Path,default=Path("outputs/probe_a_large/results.json"))
    parser.add_argument("--hpa",type=Path,
                        default=Path("/home/claude/data/hpa/subcellular_location.tsv"))
    args=parser.parse_args()

    columns={}
    with args.hpa.open() as handle:
        for row in csv.DictReader(handle,delimiter="\t"):
            columns[row["Gene name"]]=(
                bool(row.get("Single-cell variation spatial","").strip()),
                bool(row.get("Single-cell variation intensity","").strip()),
                bool(row.get("Cell cycle dependency","").strip()),
            )

    records=json.loads(args.results.read_text())
    genes=[g for g in records if g in columns]
    score={g:records[g]["entropy_of_mean"]-records[g]["mean_cell_entropy"] for g in genes}
    spatial={g for g in genes if columns[g][0]}
    intensity={g for g in genes if columns[g][1]}
    cycle={g for g in genes if columns[g][2]}
    flagged=spatial|intensity

    print(f"=== {len(genes)} genes with annotation ===")
    print(f"  spatial {len(spatial)}   intensity {len(intensity)}   cell cycle {len(cycle)}")
    print(f"  spatial as a share of the flagged group: "
          f"{len(spatial)/max(len(flagged),1):.0%}  ({len(spatial)} of {len(flagged)})")

    pick=lambda members:torch.tensor([score[g] for g in genes if g in members])
    drop=lambda members:torch.tensor([score[g] for g in genes if g not in members])

    print(f"\n=== each column against everything it does not mark ===")
    compare("spatial variation",pick(spatial),drop(spatial))
    compare("intensity variation",pick(intensity),drop(intensity))
    compare("cell cycle dependency",pick(cycle),drop(cycle))
    compare("the OR used in probe A",pick(flagged),drop(flagged))

    print(f"\n=== against a clean control: marked by no column ===")
    clean={g for g in genes if g not in flagged and g not in cycle}
    compare("spatial v clean",pick(spatial),pick(clean))
    compare("intensity v clean",pick(intensity),pick(clean))
    compare("cell cycle v clean",pick(cycle),pick(clean))

    print(f"\n=== where the marked genes sit in the ranking ===")
    order=sorted(genes,key=lambda g:-score[g])
    for name,members in (("spatial",spatial),("cell cycle",cycle),("intensity",intensity)):
        positions=[i for i,g in enumerate(order) if g in members]
        if not positions:
            print(f"  {name:<12} none present")
            continue
        median=sorted(positions)[len(positions)//2]
        top=sum(1 for p in positions if p<len(order)//5)
        print(f"  {name:<12} n={len(positions):>3}  median rank {median+1:>3}/{len(order)}"
              f"  top quintile {top}/{len(positions)} = {top/len(positions):.0%}  (base 20%)")

    print(f"\n=== the ten most heterogeneous, with every column ===")
    for gene in order[:10]:
        marks="".join(("S" if gene in spatial else "-",
                       "I" if gene in intensity else "-",
                       "C" if gene in cycle else "-"))
        print(f"  {score[gene]:.4f}  {marks}  {gene:<10} {records[gene]['location'][:46]}")
    print(f"\n  S=spatial variation  I=intensity variation  C=cell cycle dependency")


if __name__=="__main__":
    main()
