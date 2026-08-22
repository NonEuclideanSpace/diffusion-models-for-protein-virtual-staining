"""Split-half reliability of the per-gene heterogeneity, split by field rather than by cell.

Everything downstream rests on a split-half reliability of 1.000 that was computed by splitting
*cells* at random. A gene's cells sit in 4-6 imaging fields, so a random cell split puts the same
fields on both sides: it measures re-measurement of the same fields, not agreement between two
fresh samples of cells.

Three controls decide whether the answer means anything:
  - matched n: the cell split is re-run at the sample sizes the field split produces, so a drop
    cannot be blamed on the field halves being smaller
  - permuted fields: field labels are shuffled inside a gene, giving the between-field value
    finite samples produce with no field structure at all
  - permuted state: the same for the state covariates. This one is not a formality - binning 40
    cells into three quantile bins buys 23% of the within-field heterogeneity from pure noise,
    which is most of what an uncorrected decomposition would report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

BINS=3


def entropy(p:torch.Tensor)->torch.Tensor:
    return -(p.clamp_min(1e-12)*p.clamp_min(1e-12).log()).sum(-1)


def hetero(p:torch.Tensor)->float:
    """I(cell; class) in nats: how much a cell's identity says about its predicted class."""
    return float(entropy(p.mean(0))-entropy(p).mean())


def conditional(p:torch.Tensor,label:torch.Tensor)->float:
    """I(cell; class | group): the heterogeneity that survives holding the group fixed."""
    pooled=sum(float((label==v).float().mean())*float(entropy(p[label==v].mean(0)))
               for v in label.unique())
    return pooled-float(entropy(p).mean())


def quantile_bins(x:torch.Tensor,k:int)->torch.Tensor:
    return torch.bucketize(x,torch.quantile(x,torch.linspace(0,1,k+1)[1:-1]))


def explained_within(p:torch.Tensor,x:torch.Tensor,field:torch.Tensor)->float:
    """I(state; class | field): bin the covariate inside each field, so field is held fixed."""
    gain=0.0
    for value in field.unique():
        take=field==value
        block=p[take]
        gain+=float(take.float().mean())*(hetero(block)-conditional(block,quantile_bins(x[take],BINS)))
    return gain


def split_fields(field:torch.Tensor,generator)->tuple[torch.Tensor,torch.Tensor]|None:
    """Put whole fields on two sides, keeping the cell counts as even as the fields allow."""
    unique=field.unique()
    if len(unique)<2:
        return None
    order=unique[torch.randperm(len(unique),generator=generator)].tolist()
    sizes={f:int((field==f).sum()) for f in order}
    half=sum(sizes.values())/2
    left,running=[],0
    for name in order:
        if not left or running+sizes[name]<=half:
            left.append(name); running+=sizes[name]
    if len(left)==len(unique):
        left=left[:-1]
    mask=torch.isin(field,torch.tensor(left))
    return mask,~mask


def correlate(a:list[float],b:list[float])->tuple[float,float]:
    x,y=torch.tensor(a),torch.tensor(b)
    rank=lambda v:v.argsort().argsort().float()
    return (float(torch.corrcoef(torch.stack([x,y]))[0,1]),
            float(torch.corrcoef(torch.stack([rank(x),rank(y)]))[0,1]))


def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--cells",type=Path,default=Path("outputs/audit_cells"))
    parser.add_argument("--draws",type=int,default=400)
    parser.add_argument("--permutations",type=int,default=200)
    parser.add_argument("--out",type=Path,default=Path("outputs/field_audit.json"))
    args=parser.parse_args()

    genes={}
    for path in sorted(args.cells.glob("*.pt")):
        blob=torch.load(path,map_location="cpu",weights_only=False)
        names=sorted(set(blob["field"]))
        genes[path.stem]={"grouped":blob["grouped"].float(),
                          "field":torch.tensor([names.index(f) for f in blob["field"]]),
                          "density":blob["density"],"area":blob["area"]}
    names=sorted(genes)
    report={"genes":len(names),"per_gene":{}}
    generator=torch.Generator().manual_seed(0)

    print(f"{len(names)} genes\n")
    print(f"{'gene':<10}{'cells':>6}{'flds':>5}{'total':>9}{'between':>9}{'within':>9}{'btw_null':>10}")
    grand=torch.zeros(4)
    for name in names:
        g=genes[name]; p=g["grouped"]; field=g["field"]
        total=hetero(p); within=conditional(p,field)
        null=torch.tensor([total-conditional(p,field[torch.randperm(len(p),generator=generator)])
                           for _ in range(args.permutations)]).mean()
        grand+=torch.tensor([total,total-within,within,float(null)])
        fields=int(field.max())+1
        report["per_gene"][name]={"total":total,"between":total-within,"within":within,
                                 "between_null":float(null),"fields":fields,"cells":len(p)}
        print(f"{name:<10}{len(p):>6}{fields:>5}{total:>9.4f}{total-within:>9.4f}"
              f"{within:>9.4f}{float(null):>10.4f}")
    print(f"\n{'MEAN':<21}{grand[0]/len(names):>9.4f}{grand[1]/len(names):>9.4f}"
          f"{grand[2]/len(names):>9.4f}{grand[3]/len(names):>10.4f}")
    share=float(grand[1]/grand[0]); excess=float((grand[1]-grand[3])/grand[0])
    print(f"between-field share {share:.3f}   after subtracting the permutation null {excess:.3f}")
    report["between_share"]=share; report["between_share_excess"]=excess

    stores={key:[] for key in ("field_p","field_s","cell_p","cell_s","win_p","win_s")}
    for draw in range(args.draws):
        generator=torch.Generator().manual_seed(1000+draw)
        halves={key:[] for key in ("fa","fb","ca","cb","wa","wb")}
        for name in names:
            g=genes[name]; p=g["grouped"]; field=g["field"]
            split=split_fields(field,generator)
            if split is None:
                continue
            left,right=split
            halves["fa"].append(hetero(p[left])); halves["fb"].append(hetero(p[right]))
            halves["wa"].append(conditional(p[left],field[left]))
            halves["wb"].append(conditional(p[right],field[right]))
            order=torch.randperm(len(p),generator=generator)
            pick=torch.zeros(len(p),dtype=torch.bool); pick[order[:int(left.sum())]]=True
            halves["ca"].append(hetero(p[pick])); halves["cb"].append(hetero(p[~pick]))
        for prefix,(a,b) in (("field",("fa","fb")),("cell",("ca","cb")),("win",("wa","wb"))):
            pearson,spearman=correlate(halves[a],halves[b])
            stores[f"{prefix}_p"].append(pearson); stores[f"{prefix}_s"].append(spearman)

    def summarise(label:str,values:list[float])->None:
        v=torch.tensor(values).sort().values
        median=float(v[len(v)//2]); lo=float(v[int(0.025*len(v))]); hi=float(v[int(0.975*len(v))])
        print(f"{label:<52}{median:>8.3f}  [{lo:>6.3f}, {hi:>6.3f}]   SB={2*median/(1+median):>6.3f}")
        report[label]={"median":median,"lo":lo,"hi":hi,"spearman_brown":2*median/(1+median)}

    print(f"\nsplit-half reliability over {len(names)} genes, {args.draws} random splits")
    summarise("total heterogeneity, split by FIELD (pearson)",stores["field_p"])
    summarise("total heterogeneity, split by FIELD (spearman)",stores["field_s"])
    summarise("total heterogeneity, split by CELL, matched n (pearson)",stores["cell_p"])
    summarise("total heterogeneity, split by CELL, matched n (spearman)",stores["cell_s"])
    summarise("WITHIN-field heterogeneity, split by FIELD (pearson)",stores["win_p"])
    summarise("WITHIN-field heterogeneity, split by FIELD (spearman)",stores["win_s"])

    generator=torch.Generator().manual_seed(7)
    shuffle=lambda v:v[torch.randperm(len(v),generator=generator)]
    print(f"\nshare of heterogeneity explained by cell state, observed minus permutation null")
    print(f"{'covariate':<22}{'of TOTAL':>12}{'of WITHIN':>12}{'nulls':>18}")
    for label in ("chromatin_density","nuclear_area","dna_integral"):
        sums=torch.zeros(6)
        for name in names:
            g=genes[name]; p=g["grouped"]; field=g["field"]
            x={"chromatin_density":g["density"],"nuclear_area":g["area"],
               "dna_integral":g["density"]*g["area"]}[label]
            state=quantile_bins(x,BINS)
            sums+=torch.tensor([
                hetero(p)-conditional(p,state), hetero(p),
                explained_within(p,x,field), conditional(p,field),
                sum(hetero(p)-conditional(p,shuffle(state)) for _ in range(args.permutations))/args.permutations,
                sum(explained_within(p,shuffle(x),field) for _ in range(args.permutations))/args.permutations])
        total=float((sums[0]-sums[4])/sums[1]); within=float((sums[2]-sums[5])/sums[3])
        print(f"{label:<22}{total:>11.1%}{within:>12.1%}"
              f"{float(sums[4]/sums[1]):>12.1%} /{float(sums[5]/sums[3]):>6.1%}")
        report[label]={"total":total,"within":within,
                       "null_total":float(sums[4]/sums[1]),"null_within":float(sums[5]/sums[3])}

    args.out.write_text(json.dumps(report,indent=1))
    print(f"\nwrote {args.out}")


if __name__=="__main__":
    main()
