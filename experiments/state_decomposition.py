"""How much of a protein's localization uncertainty is unmodelled cell state, not noise.

M1 treats the spread in `pi_emp` as the conditional distribution a model should reproduce. But a
protein's position in a single cell is often not a property of the protein - it is a readout of
the cell's state. CDC20 sits in the nucleoplasm in G2/M and is degraded in G1; NR3C1 sits in the
cytosol until it is liganded. For those, "the conditional distribution over compartments" is
mostly a marginal over an unobserved state variable.

That splits the quantity M1 measures in two:

    H(compartment | protein) = I(compartment ; state | protein) + H(compartment | protein, state)
                               ^ recoverable by conditioning     ^ genuinely aleatoric

The first term is not noise. It is information the model could have used and did not - and
crucially, part of it is sitting in the landmark channels the model already receives. DNA
content, read from the nucleus channel, is a standard cell-cycle proxy; nuclear and cell area
carry size and confluency.

This estimates the first term per gene, using covariates taken only from channels a virtual
staining model is already conditioned on. Nothing here needs a trained model.
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

from pvs.data.crops import NUCLEUS, load_crop, load_mask, resize
from pvs.eval.localization import to_groups
from pvs.eval.subcell import IMAGE_SIZE, SubCellEnsemble, subcell_input

S3="https://czi-subcell-public.s3.us-west-2.amazonaws.com"


def fetch(key:str,destination:Path)->Path|None:
    for attempt in range(3):
        try:
            with urlopen(f"{S3}/{key}",timeout=90) as response:
                destination.write_bytes(response.read())
            return destination
        except Exception:
            time.sleep(2*(attempt+1))
    return None


def entropy(p:torch.Tensor)->torch.Tensor:
    return -(p.clamp_min(1e-12)*p.clamp_min(1e-12).log()).sum(-1)


def conditional_information(probability:torch.Tensor,covariate:torch.Tensor,bins:int=2)->float:
    """I(compartment ; binned covariate) for one gene, plug-in, bias-corrected by permutation.

    Cells are split at covariate quantiles. Within a bin the compartment distribution is the bin
    mean; the information is the entropy of the pooled mean minus the bin-weighted mean of bin
    entropies. The plug-in estimator is biased upward at these sample sizes, so the same
    statistic is recomputed on shuffled covariates and subtracted.
    """
    edges=torch.quantile(covariate,torch.linspace(0,1,bins+1)[1:-1]) if bins>1 else covariate[:0]
    assign=torch.bucketize(covariate,edges)
    def statistic(groups:torch.Tensor)->float:
        pooled=probability.mean(0)
        total=0.0
        for b in range(bins):
            mask=groups==b
            if not mask.any():
                continue
            total+=float(mask.float().mean())*float(entropy(probability[mask].mean(0)))
        return float(entropy(pooled))-total
    observed=statistic(assign)
    generator=torch.Generator().manual_seed(0)
    null=torch.tensor([statistic(assign[torch.randperm(len(assign),generator=generator)])
                       for _ in range(64)])
    return observed-float(null.mean()),float(null.std())


def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--plan",type=Path,default=Path("data/plans/u2os.txt"))
    parser.add_argument("--genes",type=int,default=24)
    parser.add_argument("--rank",type=Path,default=Path("outputs/u2os_screen/results.json"))
    parser.add_argument("--cells",type=int,default=40)
    parser.add_argument("--scratch",type=Path,default=Path("data/state"))
    parser.add_argument("--out",type=Path,default=Path("outputs/state_decomposition.json"))
    parser.add_argument("--sample",choices=("stratified","random"),default="stratified",
                        help="stratified halves the sample between the tail and the middle, "
                             "which is right for a pilot; random is what the distribution needs")
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--device",default="auto",
                        help="auto picks cuda, then mps, then cpu")
    parser.add_argument("--threads",type=int,default=2)
    parser.add_argument("--encoder",type=Path,
                        default=Path("data/subcell/all_channels_ViT-ProtS-Pool.pth"))
    parser.add_argument("--classifiers",type=Path,
                        default=Path("data/subcell/classifiers"))
    args=parser.parse_args()

    torch.set_num_threads(args.threads)
    if args.device=="auto":
        args.device=("cuda" if torch.cuda.is_available()
                     else "mps" if torch.backends.mps.is_available() else "cpu")
    device=torch.device(args.device)
    print(f"device: {device}",flush=True)
    stems=defaultdict(list)
    for line in args.plan.read_text().splitlines():
        gene,_,location,_,stem=line.split("\t")
        stems[gene].append((stem,location))

    ranked=json.loads(args.rank.read_text())
    order=[g for g in sorted(ranked,key=lambda g:-(ranked[g]["entropy_of_mean"]
                                                   -ranked[g]["mean_cell_entropy"]))
           if g in stems]
    if args.sample=="random":
        # The pilot deliberately over-sampled the tail. Reporting a distribution needs a
        # representative draw, or the median is meaningless.
        import random
        rng=random.Random(args.seed)
        chosen=order[:]
        rng.shuffle(chosen)
        chosen=chosen[:args.genes]
    else:
        chosen=order[:args.genes//2]
        middle=len(order)//2
        chosen+=order[middle:middle+args.genes//2]

    model=SubCellEnsemble.from_directory(args.encoder,args.classifiers).eval().to(device)
    args.scratch.mkdir(parents=True,exist_ok=True)
    results=json.loads(args.out.read_text()) if args.out.exists() else {}

    for gene in chosen:
        if gene in results:
            continue
        folder=args.scratch/gene; folder.mkdir(exist_ok=True)
        jobs=[]
        for stem,_ in stems[gene][:args.cells]:
            name=Path(stem).name
            jobs.append((f"{stem}_cell_image.png",folder/f"{name}.png"))
            jobs.append((f"{stem}_cell_mask.png",folder/f"{name}_m.png"))
        with ThreadPoolExecutor(24) as pool:
            list(pool.map(lambda j:fetch(*j),jobs))

        probabilities,dna,area,density=[],[],[],[]
        images=sorted(p for p in folder.glob("*.png") if not p.name.endswith("_m.png"))
        for start in range(0,len(images),8):
            chunk=[p for p in images[start:start+8] if p.with_name(p.stem+"_m.png").exists()]
            if not chunk:
                continue
            crops=[load_crop(p) for p in chunk]
            masks=[load_mask(p.with_name(p.stem+"_m.png")) for p in chunk]
            batch=torch.stack([resize(subcell_input(c),IMAGE_SIZE) for c in crops]).to(device)
            with torch.no_grad():
                _,probability=model(batch)
            probabilities.append(probability.cpu())
            for crop,mask in zip(crops,masks):
                nucleus=crop[NUCLEUS]*mask[0]
                lit=nucleus>nucleus[nucleus>0].median() if (nucleus>0).any() else nucleus>0
                dna.append(float(nucleus.sum()))
                area.append(float(mask.sum()))
                density.append(float(nucleus[lit].mean()) if lit.any() else 0.0)
        for path in folder.glob("*.png"):
            path.unlink(missing_ok=True)
        folder.rmdir()
        if not probabilities:
            continue

        probability=to_groups(torch.cat(probabilities))
        probability=probability/probability.sum(-1,keepdim=True).clamp_min(1e-12)
        total=float(entropy(probability.mean(0)))
        within=float(entropy(probability).mean())
        heterogeneity=total-within
        covariates={"dna":torch.tensor(dna),"cell_area":torch.tensor(area),
                    "chromatin_density":torch.tensor(density)}
        explained={}
        for name,values in covariates.items():
            value,noise=conditional_information(probability,values)
            explained[name]={"information":value,"null_sd":noise,
                             "share":value/max(heterogeneity,1e-9)}
        results[gene]={"cells":len(probability),"heterogeneity":heterogeneity,
                       "explained":explained}
        best=max(explained,key=lambda k:explained[k]["information"])
        print(f"  {gene:<11} n={len(probability):<3} H={heterogeneity:.4f}  "
              +"  ".join(f"{k}={v['information']:+.4f}({v['share']:+.0%})"
                         for k,v in explained.items()),flush=True)
        args.out.write_text(json.dumps(results,indent=2))

    if not results:
        return
    print(f"\n=== {len(results)} genes ===")
    for name in ("dna","cell_area","chromatin_density"):
        shares=torch.tensor([r["explained"][name]["share"] for r in results.values()])
        values=torch.tensor([r["explained"][name]["information"] for r in results.values()])
        sds=torch.tensor([r["explained"][name]["null_sd"] for r in results.values()])
        clears=int((values>2*sds).sum())
        print(f"  {name:<19} median share {shares.median():+.1%}  mean {shares.mean():+.1%}"
              f"   {clears}/{len(shares)} genes clear 2 null SD")


if __name__=="__main__":
    main()
