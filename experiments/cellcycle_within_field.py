"""Does CDC20's cell-cycle result survive blocking on the imaging field?

`findings.md` A3 is this project's strongest biological claim: the cells a classifier calls
Negative for CDC20 carry 1.5x less chromatin density than the ones it calls Nucleoplasm, at
identical nuclear area, p < 0.0001. It was validated against the nucleus channel, which the
classifier never sees.

`field_decomposition.py` now shows that roughly 85% of a gene's measured heterogeneity sits
between imaging fields rather than between cells sharing one. Fields differ in exposure, focus
and staining batch, and a 4.6x change in protein-channel brightness flips 29% of SubCell calls
(`intensity_sensitivity.py`). The nucleus channel shares the field's focus and exposure too, so
a field-level artefact could in principle produce both halves of the A3 result at once.

The test: recompute the same comparison **within each field**, where every imaging condition is
shared by construction, and pool across fields. If the chromatin-density difference survives
blocking, the result is about cells. If it vanishes, it was about fields.
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

import torch

from pvs.data.crops import NUCLEUS, load_crop, load_mask, resize
from pvs.eval.localization import CLASS_NAMES
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


def mann_whitney(a:torch.Tensor,b:torch.Tensor)->float:
    ranks=torch.cat([a,b]).argsort().argsort().float()+1
    n,m=len(a),len(b)
    return (ranks[:n].sum().item()-n*(n+1)/2)/(n*m)


def stratified_effect(values:torch.Tensor,label:torch.Tensor,block:list[str])->tuple[float,int,int]:
    """Van Elteren style: Mann-Whitney within each block, weighted by block information."""
    groups=defaultdict(list)
    for index,name in enumerate(block):
        groups[name].append(index)
    weighted,weight,used=0.0,0.0,0
    for members in groups.values():
        left=values[[i for i in members if label[i]]]
        right=values[[i for i in members if not label[i]]]
        if len(left)<2 or len(right)<2:
            continue
        w=len(left)*len(right)/(len(left)+len(right)+1)
        weighted+=w*mann_whitney(left,right)
        weight+=w
        used+=1
    return (weighted/weight if weight else float("nan")),used,len(groups)


def permutation_within(values:torch.Tensor,label:torch.Tensor,block:list[str],
                       draws:int,seed:int)->float:
    """Null: shuffle the label *inside* each field, so field effects cannot contribute."""
    observed=abs(stratified_effect(values,label,block)[0]-0.5)
    groups=defaultdict(list)
    for index,name in enumerate(block):
        groups[name].append(index)
    generator=torch.Generator().manual_seed(seed)
    hits=0
    for _ in range(draws):
        shuffled=label.clone()
        for members in groups.values():
            order=torch.randperm(len(members),generator=generator).tolist()
            shuffled[members]=label[[members[i] for i in order]]
        value=stratified_effect(values,shuffled,block)[0]
        hits+=(not value!=value) and abs(value-0.5)>=observed
    return (hits+1)/(draws+1)


def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--plan",type=Path,default=Path("data/plans/u2os.txt"))
    parser.add_argument("--fallback",type=Path,default=Path("data/plans/plan2.txt"))
    parser.add_argument("--genes",nargs="+",default=["CDC20","INCENP","FAF2"])
    parser.add_argument("--draws",type=int,default=4000)
    parser.add_argument("--scratch",type=Path,default=Path("data/cyclefield"))
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
    torch.set_num_threads(1)
    stems=defaultdict(list)
    for path in (args.plan,args.fallback):
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            gene,_,_,_,stem=line.split("\t")
            if gene in args.genes and stem not in stems[gene]:
                stems[gene].append(stem)

    model=SubCellEnsemble.from_directory(args.encoder,args.classifiers).eval().to(device)
    args.scratch.mkdir(parents=True,exist_ok=True)
    field_of=lambda stem:"_".join(Path(stem).name.split("_")[:3])

    print("A3 asks whether two classifier-defined populations differ in chromatin density.")
    print("Blocking on field removes exposure, focus and staining batch by construction.\n")

    for gene in args.genes:
        if not stems[gene]:
            print(f"{gene}: not in either plan"); continue
        folder=args.scratch/gene; folder.mkdir(exist_ok=True)
        jobs=[]
        for stem in stems[gene]:
            name=Path(stem).name
            jobs.append((f"{stem}_cell_image.png",folder/f"{name}.png"))
            jobs.append((f"{stem}_cell_mask.png",folder/f"{name}_m.png"))
        with ThreadPoolExecutor(24) as pool:
            list(pool.map(lambda j:fetch(*j),jobs))

        density,area,probabilities,fields=[],[],[],[]
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

        probability=torch.cat(probabilities)
        calls=probability.argmax(1)
        counts=torch.bincount(calls,minlength=len(CLASS_NAMES))
        first,second=counts.argsort(descending=True)[:2].tolist()
        keep=[i for i,c in enumerate(calls.tolist()) if c in (first,second)]
        label=torch.tensor([calls[i].item()==first for i in keep])
        block=[fields[i] for i in keep]
        if int(label.sum())<2 or int((~label).sum())<2:
            # One population only. That is a finding about the gene in this line, not an error.
            print(f"{gene}   {CLASS_NAMES[first]} {int(label.sum())} v "
                  f"{CLASS_NAMES[second]} {int((~label).sum())}   "
                  f"single population in this cell line, nothing to block on\n",flush=True)
            continue
        print(f"{gene}   {CLASS_NAMES[first]} {int(label.sum())} v "
              f"{CLASS_NAMES[second]} {int((~label).sum())}   "
              f"{len(set(block))} fields, {len(set(fields))} total")
        # how the two populations are distributed over fields
        spread=defaultdict(lambda:[0,0])
        for i,name in zip(keep,block):
            spread[name][0 if calls[i].item()==first else 1]+=1
        pure=sum(1 for v in spread.values() if 0 in v)
        print(f"  fields containing only one of the two populations: {pure}/{len(spread)}")
        for name,measure in (("chromatin density",torch.tensor([density[i] for i in keep])),
                             ("nuclear area",torch.tensor([area[i] for i in keep]))):
            crude=mann_whitney(measure[label],measure[~label])
            within,used,total=stratified_effect(measure,label,block)
            value=permutation_within(measure,label,block,args.draws,0) if used else float("nan")
            print(f"  {name:<18} unblocked {crude:.3f}   within-field {within:.3f}"
                  f"  ({used}/{total} fields usable)   p {value:.4f}"
                  f"   {'survives' if value==value and value<0.05 else 'does not survive'}")
        print(flush=True)


if __name__=="__main__":
    main()
