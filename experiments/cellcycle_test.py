"""Turn a recognition into a prediction, and check it.

CDC20 activates the anaphase-promoting complex and is destroyed by it in G1, so a population of
cells with no CDC20 signal should not be spread at random across the cell cycle - it should be
the low-DNA-content population. Integrated DAPI inside the cell mask is a standard proxy for DNA
content: G1 cells sit near 2N and G2/M cells near 4N.

The prediction is fixed before the measurement: cells the classifier calls Negative for CDC20
carry *less* DNA than cells it calls Nucleoplasm. Nothing about the classifier knows about DNA -
it sees the protein channel through a representation trained on localization, and the nucleus
channel is read here directly from the image.
"""
from __future__ import annotations

import argparse
import time
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


def mann_whitney(a:torch.Tensor,b:torch.Tensor)->tuple[float,float]:
    ranks=torch.cat([a,b]).argsort().argsort().float()+1
    n,m=len(a),len(b)
    u=ranks[:n].sum().item()-n*(n+1)/2
    return u/(n*m),(u-n*m/2)/max((n*m*(n+m+1)/12)**0.5,1e-9)


def permutation(a:torch.Tensor,b:torch.Tensor,draws:int=20000,seed:int=0)->float:
    observed=abs(mann_whitney(a,b)[0]-0.5)
    pool=torch.cat([a,b])
    generator=torch.Generator().manual_seed(seed)
    hits=0
    for _ in range(draws):
        order=torch.randperm(len(pool),generator=generator)
        left,right=pool[order[:len(a)]],pool[order[len(a):]]
        hits+=abs(mann_whitney(left,right)[0]-0.5)>=observed
    return (hits+1)/(draws+1)


def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--plan",type=Path,default=Path("/home/claude/data/subset/plan2.txt"))
    parser.add_argument("--genes",nargs="+",default=["CDC20","INCENP","FAF2"])
    parser.add_argument("--scratch",type=Path,default=Path("/home/claude/data/cycle"))
    parser.add_argument("--encoder",type=Path,
                        default=Path("/home/claude/data/subcell/all_channels_ViT-ProtS-Pool.pth"))
    parser.add_argument("--classifiers",type=Path,
                        default=Path("/home/claude/data/subcell/classifiers"))
    args=parser.parse_args()

    torch.set_num_threads(1)
    wanted=set(args.genes)
    stems={g:[] for g in args.genes}
    for line in args.plan.read_text().splitlines():
        gene,_,_,_,stem=line.split("\t")
        if gene in wanted:
            stems[gene].append(stem)

    model=SubCellEnsemble.from_directory(args.encoder,args.classifiers).eval()
    args.scratch.mkdir(parents=True,exist_ok=True)

    print("prediction, fixed before measuring: for a cell-cycle-degraded protein the cells the")
    print("classifier calls Negative carry LESS DNA than the cells it calls by a compartment.\n")

    for gene in args.genes:
        folder=args.scratch/gene
        folder.mkdir(exist_ok=True)
        jobs=[]
        for stem in stems[gene]:
            name=Path(stem).name
            jobs.append((f"{stem}_cell_image.png",folder/f"{name}.png"))
            jobs.append((f"{stem}_cell_mask.png",folder/f"{name}_mask.png"))
        with ThreadPoolExecutor(24) as pool:
            list(pool.map(lambda j:fetch(*j),jobs))

        dna,probabilities,kept=[],[],[]
        images=sorted(p for p in folder.glob("*.png") if not p.name.endswith("_mask.png"))
        for start in range(0,len(images),8):
            chunk=images[start:start+8]
            crops,masks=[],[]
            for path in chunk:
                mask_path=path.with_name(path.stem+"_mask.png")
                if not mask_path.exists():
                    continue
                crops.append(load_crop(path)); masks.append(load_mask(mask_path))
                kept.append(path.stem)
            if not crops:
                continue
            batch=torch.stack([resize(subcell_input(c),IMAGE_SIZE) for c in crops])
            with torch.no_grad():
                _,probability=model(batch)
            probabilities.append(probability)
            for crop,mask in zip(crops,masks):
                nucleus=crop[NUCLEUS]*mask[0]
                # Integrated DAPI scales with both DNA content and nuclear size. A G2/M nucleus
                # holds twice the DNA without doubling its area, so the mean over the stained
                # region separates the two explanations.
                stained=(nucleus>nucleus[nucleus>0].median()) if (nucleus>0).any() else nucleus>0
                area=float(stained.sum())
                dna.append((float(nucleus.sum()),area,
                            float(nucleus[stained].mean()) if area else 0.0))
        for path in folder.glob("*.png"):
            path.unlink(missing_ok=True)
        folder.rmdir()

        probability=torch.cat(probabilities)
        dna=torch.tensor(dna)
        calls=probability.argmax(1)
        counts=torch.bincount(calls,minlength=len(CLASS_NAMES))
        first,second=counts.argsort(descending=True)[:2].tolist()
        left=dna[calls==first]
        right=dna[calls==second]
        if min(len(left),len(right))<5:
            print(f"{gene}: one population too small to test"); continue
        print(f"{gene}   {CLASS_NAMES[first]} n={len(left)}  v  "
              f"{CLASS_NAMES[second]} n={len(right)}")
        for column,label in enumerate(("integrated DAPI","nuclear area","mean DAPI")):
            a,b=left[:,column],right[:,column]
            effect,z=mann_whitney(a,b)
            value=permutation(a,b)
            print(f"  {label:<17} {a.median():9.1f} v {b.median():9.1f}"
                  f"   ratio {float(a.median()/max(b.median(),1e-9)):.2f}x"
                  f"   effect {effect:.3f}  p {value:.4f}   "
                  f"{'separates' if value<0.05 else 'not resolved'}")
        print(flush=True)


if __name__=="__main__":
    main()
