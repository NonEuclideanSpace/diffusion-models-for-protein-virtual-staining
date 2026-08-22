"""Were we testing the annotations against the wrong statistic?

HPA's own method page defines single-cell variation as what an annotator sees "between cells
within the same image". The observation unit of the ground truth is therefore one imaging field.
The statistic we tested it against - total heterogeneity over a gene's 40 cells - puts about 85%
of its mass *between* fields, which is exactly the part an annotator never compares. That would
explain an AUC of 0.55 without the annotations being wrong and without the measurement being
wrong: they are simply not the same quantity.

Prediction, written before running: the within-field component should score higher against
`spatial` than the total does. Two further consequences follow if the story is right.

  - the field-count bias disappears. Total heterogeneity rose with the number of fields a gene
    spans (rho = +0.269), which is why the earlier AUC needed a correction at all. The within
    component is an average over fields, so it should be flat in field count and need no
    correction.
  - it should hold in the embedding space too. Compartment probabilities pass through the same
    classifier that defines the annotation vocabulary; distances between raw SubCell embeddings
    do not. If the effect is real it survives that change of readout.

No new downloads: the screen already saved per-cell probabilities and embeddings, and the plan
file carries the field identity. A gene is used only when the saved row count equals its plan
length, which is the condition under which the two are known to be aligned.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from pvs.eval.localization import to_groups

FIELD=lambda stem:"_".join(Path(stem).name.split("_")[:3])


def entropy(p:torch.Tensor)->torch.Tensor:
    return -(p.clamp_min(1e-12)*p.clamp_min(1e-12).log()).sum(-1)


def decompose(p:torch.Tensor,field:list[str])->tuple[float,float,float]:
    """total = within + between, exactly (chain rule for mutual information)."""
    members=defaultdict(list)
    for index,name in enumerate(field):
        members[name].append(index)
    total=float(entropy(p.mean(0))-entropy(p).mean())
    within=pooled=0.0
    for rows in members.values():
        block=p[rows]; weight=len(rows)/len(p)
        within+=weight*float(entropy(block.mean(0))-entropy(block).mean())
        pooled+=weight*float(entropy(block.mean(0)))
    return total,within,float(entropy(p.mean(0)))-pooled


def spread(e:torch.Tensor,field:list[str])->tuple[float,float]:
    """Dispersion of raw embeddings, total and averaged inside fields."""
    members=defaultdict(list)
    for index,name in enumerate(field):
        members[name].append(index)
    total=float((e-e.mean(0)).norm(dim=1).mean())
    within=sum(len(r)/len(e)*float((e[r]-e[r].mean(0)).norm(dim=1).mean()) for r in members.values())
    return total,within


def auc(score:list[float],label:list[int])->float:
    """Mann-Whitney U over ranks, ties averaged."""
    order=sorted(range(len(score)),key=lambda i:score[i])
    rank=[0.0]*len(score); i=0
    while i<len(order):
        j=i
        while j+1<len(order) and score[order[j+1]]==score[order[i]]:
            j+=1
        for k in range(i,j+1):
            rank[order[k]]=(i+j)/2+1
        i=j+1
    positive=sum(label); negative=len(label)-positive
    if positive==0 or negative==0:
        return float("nan")
    return (sum(r for r,l in zip(rank,label) if l)-positive*(positive+1)/2)/(positive*negative)


def interval(score:list[float],label:list[int],draws:int,seed:int)->tuple[float,float]:
    generator=torch.Generator().manual_seed(seed)
    values=[]
    for _ in range(draws):
        pick=torch.randint(len(label),(len(label),),generator=generator).tolist()
        values.append(auc([score[i] for i in pick],[label[i] for i in pick]))
    values=torch.tensor([v for v in values if v==v]).sort().values
    return float(values[int(0.025*len(values))]),float(values[int(0.975*len(values))])


def spearman(a:list[float],b:list[float])->float:
    rank=lambda v:[sorted(range(len(v)),key=lambda i:v[i]).index(i) for i in range(len(v))]
    x,y=torch.tensor(rank(a),dtype=torch.float),torch.tensor(rank(b),dtype=torch.float)
    x,y=x-x.mean(),y-y.mean()
    return float((x*y).sum()/(x.norm()*y.norm()))


def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--plan",type=Path,default=Path("data/plans/u2os.txt"))
    parser.add_argument("--embeddings",type=Path,default=Path("outputs/u2os_screen/embeddings"))
    parser.add_argument("--ranking",type=Path,default=Path("outputs/u2os_heterogeneity_ranking.tsv"))
    parser.add_argument("--draws",type=int,default=4000)
    parser.add_argument("--out",type=Path,default=Path("outputs/annotation_within_field.json"))
    args=parser.parse_args()

    stems=defaultdict(list)
    for line in args.plan.read_text().splitlines():
        gene,_,_,_,stem=line.split("\t")
        stems[gene].append(stem)

    header,*body=args.ranking.read_text().splitlines()
    column={name:index for index,name in enumerate(header.split("\t"))}
    annotation={}
    for line in body:
        cell=line.split("\t")
        annotation[cell[column["gene"]]]={k:int(cell[column[k]]) for k in ("spatial","intensity","cell_cycle")}

    record,skipped={},0
    for path in sorted(args.embeddings.glob("*.pt")):
        gene=path.stem
        if gene not in stems or gene not in annotation:
            continue
        blob=torch.load(path,map_location="cpu",weights_only=False)
        p=blob["probability"].float()
        if len(p)!=len(stems[gene]):
            skipped+=1
            continue
        field=[FIELD(s) for s in stems[gene]]
        total,within,between=decompose(to_groups(p),field)
        e_total,e_within=spread(blob["embedding"].float(),field)
        record[gene]={"total":total,"within":within,"between":between,
                      "embedding_total":e_total,"embedding_within":e_within,
                      "fields":len(set(field)),"cells":len(p),**annotation[gene]}

    genes=sorted(record)
    print(f"{len(genes)} genes usable, {skipped} skipped for row/plan mismatch\n")

    print(f"{'statistic':<24}{'rho with field count':>22}")
    counts=[float(record[g]["fields"]) for g in genes]
    for key in ("total","within","between","embedding_total","embedding_within"):
        print(f"{key:<24}{spearman([record[g][key] for g in genes],counts):>22.3f}")

    print(f"\nAUC against HPA annotations, {len(genes)} genes, {args.draws} bootstrap draws")
    print(f"{'statistic':<24}{'annotation':<12}{'n_pos':>6}{'AUC':>8}{'95% interval':>20}")
    summary={}
    for key in ("total","within","between","embedding_total","embedding_within"):
        for flag in ("spatial","intensity","cell_cycle"):
            score=[record[g][key] for g in genes]
            label=[record[g][flag] for g in genes]
            value=auc(score,label); lo,hi=interval(score,label,args.draws,hash(key+flag)%2**31)
            print(f"{key:<24}{flag:<12}{sum(label):>6}{value:>8.3f}   [{lo:>6.3f}, {hi:>6.3f}]")
            summary[f"{key}|{flag}"]={"auc":value,"lo":lo,"hi":hi,"positives":sum(label)}
        print()

    args.out.write_text(json.dumps({"per_gene":record,"auc":summary},indent=1))
    print(f"wrote {args.out}")


if __name__=="__main__":
    main()
