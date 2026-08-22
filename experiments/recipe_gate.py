"""How much of the between-field heterogeneity is the encoder reading the neighbours?

`subcell_input` feeds SubCell the whole 1024-pixel crop. The centre cell covers about a sixth of
it and roughly seven in ten bright nucleus pixels belong to other cells, so the encoder sees the
neighbourhood. SubCell's own inference masks the crop first, in the paper's words "to remove the
effects of surrounding cell regions" - and cells that share an imaging field share neighbours.

That is a third explanation for the 85% between-field share, alongside microscopy and biology,
and unlike those two it is a property of our pipeline rather than of the data. It is also the
only one that can be switched off, so it goes first.

Four recipes on the same cells, so every difference is preprocessing and nothing else:

    ours            percentile clip anchored on microtubules, resized to 448, unmasked
    ours_masked     the same, with the dilated single-cell mask applied
    official_448    mask, then global min-max over four channels, resized to 448
    official_640    mask, then global min-max, centre-cropped to 640 with no resize

`official_640` is the published recipe. If the between-field share falls when the mask goes on,
the neighbourhood was inside the measurement; if it survives, the 85% is microscopy or biology
and the audit stands as written.
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

from pvs.data.crops import load_crop, load_mask, resize
from pvs.eval.context import FIELDS, covariates
from pvs.eval.localization import to_groups
from pvs.eval.subcell import IMAGE_SIZE, SubCellEnsemble, subcell_input, subcell_official

S3="https://czi-subcell-public.s3.us-west-2.amazonaws.com"
RECIPES=("ours","ours_masked","official_448","official_640")


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


def decompose(p:torch.Tensor,field:list[str])->tuple[float,float,float]:
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


def prepare(recipe:str,crops:list[torch.Tensor],masks:list[torch.Tensor])->torch.Tensor:
    if recipe=="ours":
        return torch.stack([resize(subcell_input(c),IMAGE_SIZE) for c in crops])
    if recipe=="ours_masked":
        from pvs.eval.subcell import dilate
        return torch.stack([resize(subcell_input(c)*dilate(m),IMAGE_SIZE) for c,m in zip(crops,masks)])
    if recipe=="official_448":
        return torch.stack([resize(subcell_official(c,m),IMAGE_SIZE) for c,m in zip(crops,masks)])
    return torch.stack([subcell_official(c,m) for c,m in zip(crops,masks)])


def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--plan",type=Path,default=Path("data/plans/u2os.txt"))
    parser.add_argument("--genes",type=Path,default=Path("outputs/audit_cells"),
                        help="reuse exactly the genes the field audit ran on")
    parser.add_argument("--scratch",type=Path,default=Path("data/recipe"))
    parser.add_argument("--out",type=Path,default=Path("outputs/recipe_gate"))
    parser.add_argument("--encoder",type=Path,
                        default=Path("data/subcell/all_channels_ViT-ProtS-Pool.pth"))
    parser.add_argument("--classifiers",type=Path,default=Path("data/subcell/classifiers"))
    parser.add_argument("--batch",type=int,default=4)
    parser.add_argument("--device",default="auto")
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
    chosen=sorted(p.stem for p in args.genes.glob("*.pt"))
    print(f"device: {device}   {len(chosen)} genes   {len(RECIPES)} recipes",flush=True)

    model=SubCellEnsemble.from_directory(args.encoder,args.classifiers).eval().to(device)
    args.out.mkdir(parents=True,exist_ok=True)
    args.scratch.mkdir(parents=True,exist_ok=True)

    for index,gene in enumerate(chosen,1):
        target=args.out/f"{gene}.pt"
        if target.exists():
            continue
        clock=time.perf_counter()
        folder=args.scratch/gene; folder.mkdir(exist_ok=True)
        jobs=[]
        for stem in stems[gene]:
            name=Path(stem).name
            jobs.append((f"{stem}_cell_image.png",folder/f"{name}.png"))
            jobs.append((f"{stem}_cell_mask.png",folder/f"{name}_m.png"))
        with ThreadPoolExecutor(24) as pool:
            list(pool.map(lambda j:fetch(*j),jobs))

        spent={"download":time.perf_counter()-clock}
        images=sorted(p for p in folder.glob("*.png") if not p.name.endswith("_m.png"))
        images=[p for p in images if p.with_name(p.stem+"_m.png").exists()]
        collected={recipe:[] for recipe in RECIPES}
        fields,wells,context=[],[],[]
        for start in range(0,len(images),args.batch):
            chunk=images[start:start+args.batch]
            crops=[load_crop(p) for p in chunk]
            masks=[load_mask(p.with_name(p.stem+"_m.png")) for p in chunk]
            for recipe in RECIPES:
                mark=time.perf_counter()
                batch=prepare(recipe,crops,masks).to(device)
                with torch.no_grad():
                    _,probability=model(batch)
                collected[recipe].append(probability.cpu())
                spent[recipe]=spent.get(recipe,0.0)+time.perf_counter()-mark
            mark=time.perf_counter()
            fields.extend("_".join(p.stem.split("_")[:3]) for p in chunk)
            wells.extend("_".join(p.stem.split("_")[:2]) for p in chunk)
            context.extend([covariates(c,m)[key] for key in FIELDS] for c,m in zip(crops,masks))
            spent["context"]=spent.get("context",0.0)+time.perf_counter()-mark
        for path in folder.glob("*.png"):
            path.unlink(missing_ok=True)
        folder.rmdir()
        if not fields:
            continue
        torch.save({recipe:torch.cat(values).half() for recipe,values in collected.items()}
                   |{"field":fields,"well":wells,"context":torch.tensor(context),
                     "context_keys":list(FIELDS)},target)
        print(f"  {index}/{len(chosen)} {gene:<11} {len(fields)} cells   "
              +"  ".join(f"{k} {v:.0f}s" for k,v in spent.items()),flush=True)

    genes={p.stem:torch.load(p,map_location="cpu",weights_only=False)
           for p in sorted(args.out.glob("*.pt"))}
    # a field is plate_well_sample, so the well it sits in is recoverable from the field alone -
    # which keeps files written before `well` was recorded usable without re-running them
    for blob in genes.values():
        blob.setdefault("well",["_".join(f.split("_")[:2]) for f in blob["field"]])
    print(f"\nagreement with the published recipe (official_640), over {len(genes)} genes")
    print(f"{'recipe':<16}{'argmax agree':>14}{'mean TV':>10}")
    reference={g:blob["official_640"].float() for g,blob in genes.items()}
    summary={}
    for recipe in RECIPES[:-1]:
        agree=total_tv=count=0
        for gene,blob in genes.items():
            p=blob[recipe].float(); q=reference[gene]
            agree+=int((p.argmax(-1)==q.argmax(-1)).sum()); count+=len(p)
            total_tv+=float((p-q).abs().sum(-1).mul(0.5).sum())
        print(f"{recipe:<16}{agree/count:>14.3f}{total_tv/count:>10.3f}")
        summary[recipe]={"argmax_agreement":agree/count,"mean_tv":total_tv/count}

    print(f"\nvariance decomposition under each recipe")
    print(f"{'recipe':<16}{'total':>9}{'within':>9}{'between':>9}{'between share':>15}")
    for recipe in RECIPES:
        grand=torch.zeros(3)
        for gene,blob in genes.items():
            grand+=torch.tensor(decompose(to_groups(blob[recipe].float()),blob["field"]))
        share=float(grand[2]/grand[0])
        print(f"{recipe:<16}{grand[0]/len(genes):>9.4f}{grand[1]/len(genes):>9.4f}"
              f"{grand[2]/len(genes):>9.4f}{share:>15.3f}")
        summary.setdefault(recipe,{})["decomposition"]={
            "total":float(grand[0]/len(genes)),"within":float(grand[1]/len(genes)),
            "between":float(grand[2]/len(genes)),"between_share":share}

    print(f"\nwhere does population context vary? A covariate is only usable in a stratum that "
          f"holds imaging fixed AND still lets the covariate move.")
    print(f"{'covariate':<20}{'mean':>10}{'sd':>9}{'within-FOV':>12}{'within-well':>13}")
    keys=next(iter(genes.values()))["context_keys"]
    for position,key in enumerate(keys):
        pooled,inside_fov,inside_well=[],[],[]
        for blob in genes.values():
            column=blob["context"][:,position]
            keep=torch.isfinite(column)
            pooled.append(column[keep])
            for level,store in (("field",inside_fov),("well",inside_well)):
                for name in set(blob[level]):
                    take=[i for i,f in enumerate(blob[level]) if f==name and bool(keep[i])]
                    if len(take)>1:
                        block=column[take]; store.append(block-block.mean())
        pooled=torch.cat(pooled)
        spread=pooled.var().clamp_min(1e-12)
        fov=float(torch.cat(inside_fov).var()/spread) if inside_fov else float("nan")
        well=float(torch.cat(inside_well).var()/spread) if inside_well else float("nan")
        print(f"{key:<20}{float(pooled.mean()):>10.3f}{float(pooled.std()):>9.3f}"
              f"{fov:>12.3f}{well:>13.3f}")
        summary.setdefault("context",{})[key]={"mean":float(pooled.mean()),"sd":float(pooled.std()),
                                               "within_fov":fov,"within_well":well}

    print(f"\nnested decomposition of the heterogeneity, published recipe")
    print(f"{'level':<28}{'nats':>9}{'share':>9}")
    grand=torch.zeros(3)
    for blob in genes.values():
        p=to_groups(blob["official_640"].float())
        total,within_fov,_=decompose(p,blob["field"])
        _,within_well,_=decompose(p,blob["well"])
        grand+=torch.tensor([within_fov,within_well-within_fov,total-within_well])
    labels=("within FOV","between FOV, same well","between well")
    for label,value in zip(labels,grand):
        print(f"{label:<28}{value/len(genes):>9.4f}{float(value/grand.sum()):>9.3f}")

    (args.out/"summary.json").write_text(json.dumps(summary,indent=1))
    print(f"\nwrote {args.out/'summary.json'}")


if __name__=="__main__":
    main()
