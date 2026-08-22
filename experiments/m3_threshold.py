"""Fix M3's detection threshold without looking at the labels it will later be judged on.

The first gate run used threshold 0.04, chosen arbitrarily, and spot density separated
vesicular from cytosolic genes in the wrong direction: the detector was firing on the granular
texture that fills a cytosolic cell rather than on puncta. Sweeping the threshold and keeping
whichever value makes the gate pass is the forking path that has already produced one retracted
result on this project.

So the threshold is calibrated against a null that never sees the labels. Phase scrambling
replaces each cell's Fourier phases with random ones, which preserves the power spectrum - all
the texture statistics - and destroys every localised structure. Whatever the detector still
finds there is a false positive by construction. The threshold is the smallest value at which
the scrambled false-positive rate falls under a fixed budget, and that rule is fixed here,
before the discrimination is measured again.
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
from pvs.eval.pointprocess import detect_spots

S3="https://czi-subcell-public.s3.us-west-2.amazonaws.com"
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


def phase_scramble(image:torch.Tensor,generator:torch.Generator)->torch.Tensor:
    """Randomise Fourier phase, keep the amplitude spectrum. Texture survives, structure does not."""
    spectrum=torch.fft.rfft2(image)
    phase=torch.rand(spectrum.shape,generator=generator)*2*torch.pi
    scrambled=spectrum.abs()*torch.exp(1j*phase)
    scrambled[0,0]=spectrum[0,0]
    return torch.fft.irfft2(scrambled,s=image.shape)


def mann_whitney(a:torch.Tensor,b:torch.Tensor)->tuple[float,float]:
    ranks=torch.cat([a,b]).argsort().argsort().float()+1
    n,m=len(a),len(b)
    u=ranks[:n].sum().item()-n*(n+1)/2
    return u/(n*m),(u-n*m/2)/max((n*m*(n+m+1)/12)**0.5,1e-9)


def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--plan",type=Path,default=Path("/home/claude/data/subset/u2os_full.txt"))
    parser.add_argument("--genes-per-group",type=int,default=10)
    parser.add_argument("--cells",type=int,default=8)
    parser.add_argument("--size",type=int,default=1024)
    parser.add_argument("--thresholds",type=float,nargs="+",
                        default=[0.02,0.04,0.08,0.15,0.30,0.60,1.20])
    parser.add_argument("--budget",type=float,default=0.05,
                        help="allowed share of detections that survive phase scrambling")
    parser.add_argument("--scratch",type=Path,default=Path("/home/claude/data/m3t"))
    parser.add_argument("--seed",type=int,default=0)
    args=parser.parse_args()

    groups=defaultdict(list); seen=set()
    for line in args.plan.read_text().splitlines():
        gene,_,location,_,stem=line.split("\t")
        primary=location.split(";")[0]
        kind=("punctate" if primary in PUNCTATE else "diffuse" if primary in DIFFUSE else None)
        if kind is None:
            continue
        if gene not in seen:
            if len(groups[kind])>=args.genes_per_group:
                continue
            seen.add(gene); groups[kind].append((gene,[]))
        for name,stems in groups[kind]:
            if name==gene and len(stems)<args.cells:
                stems.append(stem)

    args.scratch.mkdir(parents=True,exist_ok=True)
    generator=torch.Generator().manual_seed(args.seed)
    real=defaultdict(lambda:defaultdict(list))
    scrambled=defaultdict(list)

    for kind in ("punctate","diffuse"):
        for gene,stems in groups[kind]:
            folder=args.scratch/gene; folder.mkdir(exist_ok=True)
            jobs=[]
            for stem in stems:
                name=Path(stem).name
                jobs.append((f"{stem}_cell_image.png",folder/f"{name}.png"))
                jobs.append((f"{stem}_cell_mask.png",folder/f"{name}_m.png"))
            with ThreadPoolExecutor(24) as pool:
                list(pool.map(lambda j:fetch(*j),jobs))
            counts=defaultdict(list); noise=defaultdict(list)
            for path in sorted(p for p in folder.glob("*.png") if not p.name.endswith("_m.png")):
                mask_path=path.with_name(path.stem+"_m.png")
                if not mask_path.exists():
                    continue
                mask=load_mask(mask_path)
                plane=(normalize(load_crop(path))[PROTEIN]*mask[0])
                if args.size!=plane.shape[-1]:
                    plane=resize(plane[None],args.size)[0]
                    mask=resize(mask,args.size)
                window=(mask[0]>0.5).float()
                fake=phase_scramble(plane,generator)*window
                area=float(window.sum())
                for threshold in args.thresholds:
                    counts[threshold].append(
                        len(detect_spots(plane,window,threshold=threshold)[0])/max(area,1.0)*1e4)
                    noise[threshold].append(
                        len(detect_spots(fake,window,threshold=threshold)[0])/max(area,1.0)*1e4)
            for path in folder.glob("*.png"):
                path.unlink(missing_ok=True)
            folder.rmdir()
            for threshold in args.thresholds:
                if counts[threshold]:
                    real[threshold][kind].append(float(torch.tensor(counts[threshold]).median()))
                    scrambled[threshold].append(float(torch.tensor(noise[threshold]).median()))
            print(f"  {kind:<9} {gene}",flush=True)

    print(f"\n=== label-free calibration: what survives phase scrambling ===")
    print(f"  budget: false-positive share below {args.budget:.0%}\n")
    print(f"{'threshold':>10}{'real':>10}{'scrambled':>11}{'FP share':>10}"
          f"{'AUC':>8}{'z':>7}   verdict")
    chosen=None
    for threshold in args.thresholds:
        genuine=torch.tensor(real[threshold]["punctate"]+real[threshold]["diffuse"])
        fake=torch.tensor(scrambled[threshold])
        share=float(fake.median()/genuine.median().clamp_min(1e-9))
        punctate=torch.tensor(real[threshold]["punctate"])
        diffuse=torch.tensor(real[threshold]["diffuse"])
        auc,z=mann_whitney(punctate,diffuse) if min(len(punctate),len(diffuse))>2 else (0.5,0.0)
        passes=share<args.budget
        if passes and chosen is None:
            chosen=threshold
        print(f"{threshold:>10.2f}{genuine.median():>10.1f}{fake.median():>11.1f}{share:>9.0%}"
              f"{auc:>8.3f}{z:>+7.2f}   {'within budget' if passes else 'texture-driven'}"
              f"{'  <- chosen' if threshold==chosen and passes else ''}")

    print(f"\n  the rule was fixed before this table: smallest threshold whose scrambled")
    print(f"  false-positive share is under {args.budget:.0%}. That is "
          f"{chosen if chosen else 'none of the values tried'}.")
    if chosen is not None:
        punctate=torch.tensor(real[chosen]["punctate"])
        diffuse=torch.tensor(real[chosen]["diffuse"])
        auc,z=mann_whitney(punctate,diffuse)
        print(f"  at that threshold, density AUC is {auc:.3f} (z {z:+.2f}) on "
              f"{len(punctate)} v {len(diffuse)} genes")


if __name__=="__main__":
    main()
