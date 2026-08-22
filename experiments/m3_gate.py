"""Module M3's kill gate: is spot detection reliable on real images at this quality.

INTERNAL.md section 6: "Spot detection must be validated on real images before touching
generated ones. If the detector is unreliable at this image quality, the module measures
detection noise and is dead."

Three things have to hold. The detector must separate compartments that are punctate by
annotation from compartments that are diffuse; the number it produces for one gene must be
reproducible across disjoint halves of that gene's cells; and it has to still work at the
resolution a diffusion model will actually generate, which is not the resolution HPA stores.
That last one is the part nobody would notice until after the training run.
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

import torch

from pvs.data.crops import PROTEIN, load_crop, load_mask, normalize, resize
from pvs.eval.pointprocess import detect_spots, nearest_neighbour, ripley_l

S3="https://czi-subcell-public.s3.us-west-2.amazonaws.com"

# A first run put nuclear speckles, nuclear bodies and centrosomes in the punctate group. They
# are punctate, but they sit inside the nucleus - roughly a tenth of the cell mask - while
# density here is normalised by whole-cell area, so they scored low by construction and the gate
# failed in the wrong direction. M3 is about vesicular compartments, where the condition
# genuinely lacks the information; the comparison is against cytosol, which occupies the same
# territory. Nuclear structures need their own window and are out of scope.
PUNCTATE={"Vesicles","Peroxisomes","Endosomes","Lysosomes","Lipid droplets"}
DIFFUSE={"Cytosol"}


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


def correlation(a:torch.Tensor,b:torch.Tensor)->float:
    a,b=a-a.mean(),b-b.mean()
    return float((a*b).sum()/(a.norm()*b.norm()).clamp_min(1e-12))


def measure(image:torch.Tensor,mask:torch.Tensor,size:int,threshold:float)->dict:
    scale=size/image.shape[-1]
    plane=resize(image[None],size)[0] if scale!=1 else image
    window=(resize(mask,size)[0]>0.5).float() if scale!=1 else mask[0]
    sigmas=tuple(max(s*scale,0.8) for s in (1.5,2.5,4.0,6.5))
    positions,radii,_=detect_spots(plane,window,sigmas=sigmas,threshold=threshold)
    area=float(window.sum())
    neighbours=nearest_neighbour(positions)
    grid=torch.linspace(2,0.06*size,8)
    return {"density":len(positions)/max(area,1.0)*1e4,
            "count":len(positions),
            "nearest":float(neighbours.median()) if len(neighbours) else 0.0,
            "clustering":float(ripley_l(positions,window,grid)[1:5].mean())
                         if len(positions)>4 else 0.0}


def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--plan",type=Path,default=Path("/home/claude/data/subset/u2os_full.txt"))
    parser.add_argument("--genes-per-group",type=int,default=14)
    parser.add_argument("--cells",type=int,default=12)
    parser.add_argument("--sizes",type=int,nargs="+",default=[1024,256,128])
    parser.add_argument("--threshold",type=float,default=0.04)
    parser.add_argument("--scratch",type=Path,default=Path("/home/claude/data/m3"))
    args=parser.parse_args()

    groups=defaultdict(list)
    seen=set()
    for line in args.plan.read_text().splitlines():
        gene,_,location,_,stem=line.split("\t")
        primary=location.split(";")[0]
        kind=("punctate" if primary in PUNCTATE else "diffuse" if primary in DIFFUSE else None)
        if kind is None:
            continue
        if gene not in seen:
            if len([g for g,_ in groups[kind]])>=args.genes_per_group:
                continue
            seen.add(gene)
            groups[kind].append((gene,[]))
        for name,stems in groups[kind]:
            if name==gene and len(stems)<args.cells:
                stems.append(stem)

    args.scratch.mkdir(parents=True,exist_ok=True)
    results={size:defaultdict(list) for size in args.sizes}
    clustering={size:defaultdict(list) for size in args.sizes}
    halves={size:[] for size in args.sizes}

    for kind in ("punctate","diffuse"):
        for gene,stems in groups[kind]:
            folder=args.scratch/gene
            folder.mkdir(exist_ok=True)
            jobs=[]
            for stem in stems:
                name=Path(stem).name
                jobs.append((f"{stem}_cell_image.png",folder/f"{name}.png"))
                jobs.append((f"{stem}_cell_mask.png",folder/f"{name}_m.png"))
            with ThreadPoolExecutor(24) as pool:
                list(pool.map(lambda j:fetch(*j),jobs))
            per_size=defaultdict(list)
            for path in sorted(p for p in folder.glob("*.png") if not p.name.endswith("_m.png")):
                mask_path=path.with_name(path.stem+"_m.png")
                if not mask_path.exists():
                    continue
                crop=normalize(load_crop(path))
                image=crop[PROTEIN]*load_mask(mask_path)[0]
                mask=load_mask(mask_path)
                for size in args.sizes:
                    per_size[size].append(measure(image,mask,size,args.threshold))
            for path in folder.glob("*.png"):
                path.unlink(missing_ok=True)
            folder.rmdir()
            for size in args.sizes:
                cells=per_size[size]
                if len(cells)<4:
                    continue
                density=torch.tensor([c["density"] for c in cells])
                results[size][kind].append(float(density.median()))
                clustering[size][kind].append(
                    float(torch.tensor([c["clustering"] for c in cells]).median()))
                halves[size].append((float(density[::2].median()),float(density[1::2].median())))
            print(f"  {kind:<9} {gene:<10} "
                  +"  ".join(f"{s}px n={len(per_size[s])} "
                             f"d={torch.tensor([c['density'] for c in per_size[s]]).median():.2f}"
                             for s in args.sizes),flush=True)

    print(f"\n=== gate 1: does spot density separate punctate from diffuse annotations ===")
    for size in args.sizes:
        punctate=torch.tensor(results[size]["punctate"])
        diffuse=torch.tensor(results[size]["diffuse"])
        if min(len(punctate),len(diffuse))<3:
            print(f"  {size:>5}px  too few genes"); continue
        effect,z=mann_whitney(punctate,diffuse)
        print(f"  {size:>5}px  {len(punctate)} v {len(diffuse)} genes   "
              f"median density {punctate.median():6.2f} v {diffuse.median():6.2f}   "
              f"AUC {effect:.3f}  z {z:+.2f}   "
              f"{'usable' if effect>0.75 else 'weak' if effect>0.65 else 'DEAD'}")

    print(f"\n=== gate 1b: does clustering separate them (no area normalisation) ===")
    for size in args.sizes:
        punctate=torch.tensor(clustering[size]["punctate"])
        diffuse=torch.tensor(clustering[size]["diffuse"])
        if min(len(punctate),len(diffuse))<3:
            print(f"  {size:>5}px  too few genes"); continue
        effect,z=mann_whitney(punctate,diffuse)
        print(f"  {size:>5}px  {len(punctate)} v {len(diffuse)} genes   "
              f"median Ripley L {punctate.median():6.2f} v {diffuse.median():6.2f}   "
              f"AUC {effect:.3f}  z {z:+.2f}   "
              f"{'usable' if effect>0.75 else 'weak' if effect>0.65 else 'DEAD'}")

    print(f"\n=== gate 2: is one gene's density reproducible across disjoint halves ===")
    for size in args.sizes:
        pairs=torch.tensor(halves[size])
        if len(pairs)<4:
            print(f"  {size:>5}px  too few genes"); continue
        r=correlation(pairs[:,0],pairs[:,1])
        print(f"  {size:>5}px  {len(pairs)} genes   half against half r={r:.3f}   "
              f"{'usable' if r>0.7 else 'weak' if r>0.5 else 'DEAD'}")

    print(f"\n=== gate 3: does it survive the resolution a model will generate at ===")
    base=args.sizes[0]
    for size in args.sizes[1:]:
        shared=min(len(results[base]['punctate']),len(results[size]['punctate']))
        a=torch.tensor(results[base]["punctate"][:shared])
        b=torch.tensor(results[size]["punctate"][:shared])
        if shared<3:
            print(f"  {size:>5}px  too few"); continue
        print(f"  {base}px against {size:>5}px   rank agreement on punctate genes "
              f"r={correlation(a,b):.3f}   density retained "
              f"{float(b.median()/a.median().clamp_min(1e-9)):.2f}x")


if __name__=="__main__":
    main()
