"""Is between-cell localization heterogeneity biology, or is it imaging?

Everything measured so far treats a gene's 40 cells as exchangeable. They are not. Cells share
an imaging field with some of their neighbours and not others, and cells in the same field share
illumination, focus, staining batch and segmentation pass. If most of the measured heterogeneity
sits *between* fields, a large part of what this project calls biological variability is
microscopy, and the ceiling for any virtual staining model is lower than it looks.

The statistic decomposes exactly. With p_i the compartment distribution for cell i, fields f,
and w_f the fraction of cells in field f:

    total   = H(mean_all p) - mean_all H(p_i)
    within  = sum_f w_f [ H(mean_{i in f} p) - mean_{i in f} H(p_i) ]
    between = H(mean_all p) - sum_f w_f H(mean_{i in f} p)
    total   = within + between                                        (identity, not approximation)

`between` is an **upper** bound on the technical contribution: fields also differ biologically,
in confluency and local density. `within` is therefore a **lower** bound on the biological
contribution, because cells sharing a field share every imaging condition there is.

Both terms use a plug-in estimator that is biased upward at these sample sizes, so each is
reported against a permutation null that shuffles field labels among the gene's own cells.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

import torch

from pvs.data.crops import load_crop, resize
from pvs.eval.localization import to_groups
from pvs.eval.subcell import IMAGE_SIZE, SubCellEnsemble, subcell_input

S3="https://czi-subcell-public.s3.us-west-2.amazonaws.com"


def fetch(stem:str,destination:Path)->Path|None:
    for attempt in range(3):
        try:
            with urlopen(f"{S3}/{stem}_cell_image.png",timeout=90) as response:
                destination.write_bytes(response.read())
            return destination
        except Exception:
            time.sleep(2*(attempt+1))
    return None


def entropy(p:torch.Tensor)->torch.Tensor:
    return -(p.clamp_min(1e-12)*p.clamp_min(1e-12).log()).sum(-1)


def decompose(probability:torch.Tensor,field:list[str])->tuple[float,float,float]:
    total=float(entropy(probability.mean(0))-entropy(probability).mean())
    groups=defaultdict(list)
    for index,name in enumerate(field):
        groups[name].append(index)
    within=0.0
    pooled=0.0
    for members in groups.values():
        block=probability[members]
        weight=len(members)/len(probability)
        within+=weight*float(entropy(block.mean(0))-entropy(block).mean())
        pooled+=weight*float(entropy(block.mean(0)))
    between=float(entropy(probability.mean(0)))-pooled
    return total,within,between


def permuted_between(probability:torch.Tensor,field:list[str],draws:int,seed:int)->float:
    generator=torch.Generator().manual_seed(seed)
    values=[]
    for _ in range(draws):
        order=torch.randperm(len(field),generator=generator).tolist()
        values.append(decompose(probability,[field[i] for i in order])[2])
    return float(torch.tensor(values).mean())


def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--plan",type=Path,default=Path("/home/claude/data/subset/u2os.txt"))
    parser.add_argument("--rank",type=Path,default=Path("outputs/u2os_screen/results.json"))
    parser.add_argument("--genes",type=int,default=40)
    parser.add_argument("--min-fields",type=int,default=3)
    parser.add_argument("--draws",type=int,default=64)
    parser.add_argument("--scratch",type=Path,default=Path("/home/claude/data/field"))
    parser.add_argument("--out",type=Path,default=Path("outputs/field_decomposition.json"))
    parser.add_argument("--encoder",type=Path,
                        default=Path("/home/claude/data/subcell/all_channels_ViT-ProtS-Pool.pth"))
    parser.add_argument("--classifiers",type=Path,
                        default=Path("/home/claude/data/subcell/classifiers"))
    parser.add_argument("--device",default="auto")
    parser.add_argument("--seed",type=int,default=0)
    args=parser.parse_args()

    if args.device=="auto":
        args.device=("cuda" if torch.cuda.is_available()
                     else "mps" if torch.backends.mps.is_available() else "cpu")
    device=torch.device(args.device)
    torch.set_num_threads(2)

    stems=defaultdict(list)
    for line in args.plan.read_text().splitlines():
        gene,_,_,_,stem=line.split("\t")
        stems[gene].append(stem)
    # field identity is plate_well_sample; the trailing number is the cell within the field
    field_of=lambda stem:"_".join(Path(stem).name.split("_")[:3])

    ranked=json.loads(args.rank.read_text())
    order=sorted((g for g in ranked if g in stems),
                 key=lambda g:-(ranked[g]["entropy_of_mean"]-ranked[g]["mean_cell_entropy"]))
    # Spread across the range: the question is whether the split differs for heterogeneous genes.
    step=max(len(order)//args.genes,1)
    chosen=[g for g in order[::step] if len({field_of(s) for s in stems[g]})>=args.min_fields]
    chosen=chosen[:args.genes]

    model=SubCellEnsemble.from_directory(args.encoder,args.classifiers).eval().to(device)
    args.scratch.mkdir(parents=True,exist_ok=True)
    results=json.loads(args.out.read_text()) if args.out.exists() else {}
    print(f"device: {device}   {len(chosen)} genes\n")
    print(f"{'gene':<11}{'fields':>7}{'total':>9}{'within':>9}{'between':>9}"
          f"{'null btw':>10}{'within share':>14}")

    for gene in chosen:
        if gene in results:
            continue
        folder=args.scratch/gene; folder.mkdir(exist_ok=True)
        jobs=[(s,folder/f"{Path(s).name}.png") for s in stems[gene]]
        with ThreadPoolExecutor(24) as pool:
            list(pool.map(lambda j:fetch(*j),jobs))
        kept=[(s,p) for s,p in jobs if p.exists()]
        probabilities=[]
        for start in range(0,len(kept),8):
            chunk=kept[start:start+8]
            batch=torch.stack([resize(subcell_input(load_crop(p)),IMAGE_SIZE)
                               for _,p in chunk]).to(device)
            with torch.no_grad():
                _,probability=model(batch)
            probabilities.append(probability.cpu())
        for _,p in kept:
            p.unlink(missing_ok=True)
        folder.rmdir()
        if not probabilities:
            continue

        probability=to_groups(torch.cat(probabilities))
        probability=probability/probability.sum(-1,keepdim=True).clamp_min(1e-12)
        field=[field_of(s) for s,_ in kept]
        total,within,between=decompose(probability,field)
        null=permuted_between(probability,field,args.draws,args.seed)
        corrected=between-null
        share=(total-corrected)/max(total,1e-9)
        results[gene]={"cells":len(probability),"fields":len(set(field)),
                       "total":total,"within":within,"between":between,
                       "null_between":null,"corrected_between":corrected,
                       "within_share":share}
        args.out.write_text(json.dumps(results,indent=2))
        print(f"{gene:<11}{len(set(field)):>7}{total:>9.4f}{within:>9.4f}{between:>9.4f}"
              f"{null:>10.4f}{share:>13.0%}",flush=True)

    if not results:
        return
    total=torch.tensor([r["total"] for r in results.values()])
    share=torch.tensor([r["within_share"] for r in results.values()])
    corrected=torch.tensor([r["corrected_between"] for r in results.values()])
    print(f"\n=== {len(results)} genes ===")
    print(f"  within-field share of heterogeneity: median {share.median():.0%}  "
          f"mean {share.mean():.0%}  p10 {share.quantile(0.1):.0%}  p90 {share.quantile(0.9):.0%}")
    print(f"  bias-corrected between-field term:   median {corrected.median():+.4f}  "
          f"against a median total of {total.median():.4f}")
    print(f"\n  within-field share is a LOWER bound on the biological fraction: cells sharing a")
    print(f"  field share illumination, focus, staining batch and segmentation pass.")


if __name__=="__main__":
    main()
