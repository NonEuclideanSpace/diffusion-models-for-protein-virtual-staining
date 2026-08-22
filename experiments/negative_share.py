"""How much of the measured heterogeneity is about *where* the protein is, and how much is
about whether it is there at all.

Rendering the top of the U2OS tail showed the same shape in every gene: one population with a
clear signal and one with almost none. SubCell has a `Negative` class meaning no detectable
protein, and a gene that is expressed in half its cells and silent in the other half produces a
large `H(mean p) - mean H(p_i)` without ever changing compartment.

That is abundance variation, not localization variation - HPA's own distinction between
`Single-cell variation intensity` and `Single-cell variation spatial`. M1 measures a
distribution over compartments, so heterogeneity that lives entirely in the Negative class is
the wrong target and has to be measured before it is selected on.

This recomputes each gene's heterogeneity with the Negative class removed and the remaining
mass renormalised, and reports what survives.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from pvs.eval.localization import CLASS_NAMES


def information(probability:torch.Tensor)->float:
    entropy=lambda p:-(p.clamp_min(1e-12)*p.clamp_min(1e-12).log()).sum(-1)
    return float(entropy(probability.mean(0))-entropy(probability).mean())


def correlation(a:torch.Tensor,b:torch.Tensor)->float:
    a,b=a-a.mean(),b-b.mean()
    return float((a*b).sum()/(a.norm()*b.norm()).clamp_min(1e-12))


def spearman(a:torch.Tensor,b:torch.Tensor)->float:
    rank=lambda x:x.argsort().argsort().float()
    return correlation(rank(a),rank(b))


def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,default=Path("outputs/u2os_screen"))
    parser.add_argument("--top",type=int,default=15)
    args=parser.parse_args()

    negative=CLASS_NAMES.index("Negative")
    records=json.loads((args.out/"results.json").read_text())
    keep=[i for i in range(len(CLASS_NAMES)) if i!=negative]

    genes,full,without,share=[],[],[],[]
    for gene,record in records.items():
        mean=torch.tensor(record["mean_probability"])
        genes.append(gene)
        full.append(record["entropy_of_mean"]-record["mean_cell_entropy"])
        share.append(float(mean[negative]))
    full=torch.tensor(full)
    share=torch.tensor(share)

    store=args.out/"embeddings"
    index={g:i for i,g in enumerate(genes)}
    # The screen writes embeddings as it goes, so a snapshot of results.json can lag behind the
    # embedding directory. Only genes present in both are comparable.
    available=sorted(p.stem for p in store.glob("*.pt") if p.stem in index) if store.exists() else []
    print(f"=== {len(genes)} genes, mean probability mass on the Negative class ===")
    print(f"  median {share.median():.3f}  p90 {share.quantile(0.9):.3f}  max {share.max():.3f}")
    print(f"  correlation with measured heterogeneity: pearson {correlation(share,full):+.3f}"
          f"   spearman {spearman(share,full):+.3f}")

    order=full.argsort(descending=True)
    print(f"\n=== the top {args.top}, and how much of them is the Negative class ===")
    print(f"{'gene':<12}{'heterogeneity':>14}{'P(Negative)':>13}")
    for rank in order[:args.top].tolist():
        print(f"{genes[rank]:<12}{full[rank]:>14.4f}{share[rank]:>13.3f}")

    if not available:
        print(f"\n  no per-cell embeddings under {store}; run the screen with --save-embeddings "
              f"to recompute heterogeneity without the Negative class")
        return

    for gene in available:
        probability=torch.load(store/f"{gene}.pt",map_location="cpu")["probability"].float()
        trimmed=probability[:,keep]
        trimmed=trimmed/trimmed.sum(-1,keepdim=True).clamp_min(1e-12)
        without.append(information(trimmed))
    without=torch.tensor(without)
    paired=torch.tensor([full[index[g]] for g in available])

    print(f"\n=== heterogeneity with the Negative class removed, {len(available)} genes ===")
    print(f"  with Negative     mean {paired.mean():.4f}  median {paired.median():.4f}")
    print(f"  without Negative  mean {without.mean():.4f}  median {without.median():.4f}")
    print(f"  retained {float(without.mean()/paired.mean().clamp_min(1e-9)):.0%} of the mean")
    print(f"  rank correlation between the two: {spearman(paired,without):+.3f}")
    print(f"  {'the ranking is about localization' if spearman(paired,without)>0.7 else 'THE RANKING IS DRIVEN BY DETECTION, NOT LOCALIZATION'}")

    top=paired.argsort(descending=True)[:args.top].tolist()
    survivors=sum(1 for i in top if without[i]>float(without.median()))
    print(f"\n  of the top {args.top} by the full statistic, {survivors} stay above the median "
          f"once Negative is removed")


if __name__=="__main__":
    main()
