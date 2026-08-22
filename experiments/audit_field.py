"""Method audit: is the per-gene heterogeneity a property of the gene or of its imaging fields?

Every downstream claim rests on one number: split-half reliability of 1.000, quoted as evidence
that heterogeneity is "a real per-gene quantity". That split was **random over cells**, and a
gene's cells are spread over about four imaging fields. A random split puts the same fields on
both sides, so it measures whether the same fields can be re-measured - not whether a fresh
sample of cells would agree.

`field_decomposition.py` found roughly 85% of the statistic sits between fields. If reliability
collapses when the split is by *field* instead of by cell, then the reliable quantity is the
imaging layout and not the gene, and the annotation result, the ranking and the state
decomposition are all measuring that.

This makes one download pass and saves per-cell probabilities and field identity, so every audit
question afterwards is offline.
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


def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--plan",type=Path,default=Path("data/plans/u2os.txt"))
    parser.add_argument("--rank",type=Path,default=Path("outputs/u2os_screen/results.json"))
    parser.add_argument("--genes",type=int,default=44)
    parser.add_argument("--min-fields",type=int,default=4)
    parser.add_argument("--scratch",type=Path,default=Path("data/audit"))
    parser.add_argument("--out",type=Path,default=Path("outputs/audit_cells"))
    parser.add_argument("--encoder",type=Path,
                        default=Path("data/subcell/all_channels_ViT-ProtS-Pool.pth"))
    parser.add_argument("--classifiers",type=Path,
                        default=Path("data/subcell/classifiers"))
    parser.add_argument("--device",default="auto")
    args=parser.parse_args()

    if args.device=="auto":
        args.device=("cuda" if torch.cuda.is_available()
                     else "mps" if torch.backends.mps.is_available() else "cpu")
    device=torch.device(args.device)
    torch.set_num_threads(2)
    print(f"device: {device}",flush=True)
    stems=defaultdict(list)
    for line in args.plan.read_text().splitlines():
        gene,_,_,_,stem=line.split("\t")
        stems[gene].append(stem)
    field_of=lambda stem:"_".join(Path(stem).name.split("_")[:3])

    ranked=json.loads(args.rank.read_text())
    order=sorted((g for g in ranked if g in stems),
                 key=lambda g:-(ranked[g]["entropy_of_mean"]-ranked[g]["mean_cell_entropy"]))
    step=max(len(order)//args.genes,1)
    chosen=[g for g in order[::step]
            if len({field_of(s) for s in stems[g]})>=args.min_fields][:args.genes]

    model=SubCellEnsemble.from_directory(args.encoder,args.classifiers).eval().to(device)
    args.out.mkdir(parents=True,exist_ok=True)
    args.scratch.mkdir(parents=True,exist_ok=True)
    print(f"{len(chosen)} genes with at least {args.min_fields} fields",flush=True)

    for index,gene in enumerate(chosen,1):
        target=args.out/f"{gene}.pt"
        if target.exists():
            continue
        folder=args.scratch/gene; folder.mkdir(exist_ok=True)
        jobs=[]
        for stem in stems[gene]:
            name=Path(stem).name
            jobs.append((f"{stem}_cell_image.png",folder/f"{name}.png"))
            jobs.append((f"{stem}_cell_mask.png",folder/f"{name}_m.png"))
        with ThreadPoolExecutor(24) as pool:
            list(pool.map(lambda j:fetch(*j),jobs))

        images=sorted(p for p in folder.glob("*.png") if not p.name.endswith("_m.png"))
        probabilities,density,area,fields=[],[],[],[]
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
            for path,crop,mask in zip(chunk,crops,masks):
                nucleus=crop[NUCLEUS]*mask[0]
                lit=nucleus>nucleus[nucleus>0].median() if bool((nucleus>0).any()) else nucleus>0
                density.append(float(nucleus[lit].mean()) if bool(lit.any()) else 0.0)
                area.append(float(lit.sum()))
                fields.append("_".join(path.stem.split("_")[:3]))
        for path in folder.glob("*.png"):
            path.unlink(missing_ok=True)
        folder.rmdir()
        if not probabilities:
            continue
        torch.save({"probability":torch.cat(probabilities).half(),
                    "grouped":to_groups(torch.cat(probabilities)).half(),
                    "density":torch.tensor(density),"area":torch.tensor(area),
                    "field":fields},target)
        print(f"  {index}/{len(chosen)} {gene:<11} {len(fields)} cells, "
              f"{len(set(fields))} fields",flush=True)


if __name__=="__main__":
    main()
