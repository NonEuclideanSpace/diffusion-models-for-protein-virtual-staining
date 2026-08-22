"""Render the two-population genes as images.

The heterogeneity measure is a number per gene. This turns the number back into cells: for each
gene the classifier splits into two modal calls, and the extreme cells of each are drawn side by
side. If the split is real the two rows should look different to the eye, with no statistics
involved.
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import torch

from pvs.data.crops import MICROTUBULES, NUCLEUS, PROTEIN, RETICULUM, load_crop
from pvs.eval.localization import CLASS_NAMES
from pvs.eval.subcell import IMAGE_SIZE, SubCellEnsemble, subcell_input
from pvs.data.crops import resize

S3="https://czi-subcell-public.s3.us-west-2.amazonaws.com"


def download(stem:str,destination:Path)->Path|None:
    for attempt in range(3):
        try:
            with urlopen(f"{S3}/{stem}_cell_image.png",timeout=90) as response:
                destination.write_bytes(response.read())
            return destination
        except Exception:
            time.sleep(2*(attempt+1))
    return None


def stretch(plane:np.ndarray,low:float=1.0,high:float=99.7)->np.ndarray:
    a,b=np.percentile(plane,low),np.percentile(plane,high)
    return np.clip((plane-a)/max(b-a,1e-8),0,1)


def composite(crop:torch.Tensor)->np.ndarray:
    """Protein in green over the nucleus in blue and microtubules in dim red."""
    array=crop.numpy()
    rgb=np.zeros(array.shape[1:]+(3,),dtype=np.float32)
    rgb[...,2]=stretch(array[NUCLEUS])*0.75
    rgb[...,0]=stretch(array[MICROTUBULES])*0.40
    rgb[...,1]=stretch(array[PROTEIN])
    return np.clip(rgb,0,1)


def protein_only(crop:torch.Tensor)->np.ndarray:
    return stretch(crop[PROTEIN].numpy())


def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--plan",type=Path,default=Path("/home/claude/data/subset/plan2.txt"))
    parser.add_argument("--genes",nargs="+",default=["INCENP","FAF2","DECR1","CDC20"])
    parser.add_argument("--scratch",type=Path,default=Path("/home/claude/data/figure"))
    parser.add_argument("--out",type=Path,default=Path("outputs/figures/populations.png"))
    parser.add_argument("--per-row",type=int,default=4)
    parser.add_argument("--encoder",type=Path,
                        default=Path("/home/claude/data/subcell/all_channels_ViT-ProtS-Pool.pth"))
    parser.add_argument("--classifiers",type=Path,
                        default=Path("/home/claude/data/subcell/classifiers"))
    args=parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    torch.set_num_threads(1)
    wanted=set(args.genes)
    stems={g:[] for g in args.genes}
    for line in args.plan.read_text().splitlines():
        gene,_,location,_,stem=line.split("\t")
        if gene in wanted:
            stems[gene].append((stem,location))

    model=SubCellEnsemble.from_directory(args.encoder,args.classifiers).eval()
    args.scratch.mkdir(parents=True,exist_ok=True)
    args.out.parent.mkdir(parents=True,exist_ok=True)

    panels={}
    for gene in args.genes:
        entries=stems[gene]
        if not entries:
            print(f"  {gene}: not in the plan"); continue
        folder=args.scratch/gene
        folder.mkdir(exist_ok=True)
        with ThreadPoolExecutor(24) as pool:
            paths=list(pool.map(lambda e:download(e[0],folder/f"{Path(e[0]).name}.png"),entries))
        paths=[p for p in paths if p is not None]
        crops,probabilities=[],[]
        for start in range(0,len(paths),8):
            chunk=paths[start:start+8]
            loaded=[load_crop(p) for p in chunk]
            batch=torch.stack([resize(subcell_input(c),IMAGE_SIZE) for c in loaded])
            with torch.no_grad():
                _,probability=model(batch)
            crops.extend(loaded)
            probabilities.append(probability)
        probability=torch.cat(probabilities)
        calls=probability.argmax(1)
        counts=torch.bincount(calls,minlength=len(CLASS_NAMES))
        first,second=counts.argsort(descending=True)[:2].tolist()
        pick=lambda cls:probability[:,cls].argsort(descending=True)[:args.per_row].tolist()
        panels[gene]=dict(location=entries[0][1],crops=crops,
                          rows=[(CLASS_NAMES[first],int(counts[first]),pick(first)),
                                (CLASS_NAMES[second],int(counts[second]),pick(second))])
        for path in paths:
            path.unlink(missing_ok=True)
        folder.rmdir()
        print(f"  {gene}: {CLASS_NAMES[first]} {int(counts[first])} / "
              f"{CLASS_NAMES[second]} {int(counts[second])}  of {len(calls)}",flush=True)

    genes=[g for g in args.genes if g in panels]
    rows=len(genes)*2
    figure,axes=plt.subplots(rows,args.per_row,figsize=(args.per_row*2.05,rows*2.12),
                             facecolor="#0A0E14")
    for index,gene in enumerate(genes):
        panel=panels[gene]
        for side,(name,count,picks) in enumerate(panel["rows"]):
            for column in range(args.per_row):
                ax=axes[index*2+side,column]
                ax.set_facecolor("#0A0E14")
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_color("#3DD68C" if side==0 else "#FF6B6F")
                    spine.set_linewidth(1.4)
                if column<len(picks):
                    ax.imshow(composite(panel["crops"][picks[column]]))
                else:
                    ax.axis("off")
                if column==0:
                    ax.set_ylabel(f"{name}\n{count} cells",color="#E7ECF3",fontsize=8,
                                  rotation=0,ha="right",va="center",labelpad=42)
            if side==0:
                axes[index*2,0].set_title(
                    f"{gene}   HPA: {panel['location']}",color="#E7ECF3",fontsize=10,
                    loc="left",pad=8)
    figure.suptitle("Two populations, found by measurement and then looked at",
                    color="#E7ECF3",fontsize=12,y=0.997)
    figure.text(0.5,0.004,"green protein   blue nucleus   red microtubules",
                color="#7A8695",fontsize=8,ha="center")
    figure.tight_layout(rect=(0.045,0.014,1,0.985))
    figure.savefig(args.out,dpi=155,facecolor="#0A0E14")
    print(f"wrote {args.out}")


if __name__=="__main__":
    main()
