"""Validate the CELL-Diff compositing path with real protein channels."""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

import torch

from pvs.data.crops import load_crop,resize
from pvs.eval.calibration import null_distribution,total_variation
from pvs.eval.localization import to_groups
from pvs.eval.subcell import IMAGE_SIZE,SubCellEnsemble,subcell_input

S3="https://czi-subcell-public.s3.us-west-2.amazonaws.com"


def download(stem:str,destination:Path,retries:int=3)->bool:
    for attempt in range(retries):
        try:
            with urlopen(f"{S3}/{stem}_cell_image.png",timeout=90) as response:
                destination.write_bytes(response.read())
            return True
        except Exception:
            if attempt==retries-1:
                return False
            time.sleep(2*(attempt+1))
    return False


def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--ranking",type=Path,default=Path("outputs/u2os_screen/tail.json"))
    parser.add_argument("--empirical",type=Path,default=Path("outputs/u2os_screen/results.json"))
    parser.add_argument("--plan",type=Path,default=Path("data/plans/u2os.txt"))
    parser.add_argument("--num-genes",type=int,default=12)
    parser.add_argument("--size",type=int,default=256)
    parser.add_argument("--batch",type=int,default=32)
    parser.add_argument("--workers",type=int,default=24)
    parser.add_argument("--null-trials",type=int,default=2000)
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--encoder",type=Path,
                        default=Path("data/subcell/all_channels_ViT-ProtS-Pool.pth"))
    parser.add_argument("--classifiers",type=Path,default=Path("data/subcell/classifiers"))
    parser.add_argument("--scratch",type=Path,default=Path("data/control_stream"))
    parser.add_argument("--out",type=Path,
                        default=Path("outputs/celldiff_validity/positive_control.json"))
    parser.add_argument("--device",default="auto")
    args=parser.parse_args()

    if args.device=="auto":
        args.device="cuda" if torch.cuda.is_available() else "cpu"
    device=torch.device(args.device)
    ranking=json.loads(args.ranking.read_text())
    empirical=json.loads(args.empirical.read_text())
    order=sorted(zip(ranking["genes"],ranking["heterogeneity"]),key=lambda pair:-pair[1])
    genes=[gene for gene,_ in order[:args.num_genes]]

    stems=defaultdict(list)
    for line in args.plan.read_text().splitlines():
        gene,_,_,_,stem=line.split("\t")
        stems[gene].append(stem)
    missing=[gene for gene in genes if gene not in stems or gene not in empirical]
    if missing:
        raise ValueError(f"selected genes missing from the screen inputs: {missing}")

    args.scratch.mkdir(parents=True,exist_ok=True)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    classifier=SubCellEnsemble.from_directory(args.encoder,args.classifiers).eval().to(device)
    torch.manual_seed(args.seed)
    results={}
    print(f"positive control: {len(genes)} genes on {device}",flush=True)

    for gene in genes:
        folder=args.scratch/gene
        folder.mkdir(parents=True,exist_ok=True)
        paths=[folder/f"{os.path.basename(stem)}.png" for stem in stems[gene]]
        with ThreadPoolExecutor(args.workers) as pool:
            ok=list(pool.map(lambda pair:download(*pair),zip(stems[gene],paths)))
        paths=[path for path,good in zip(paths,ok) if good]
        if not paths:
            raise RuntimeError(f"no crops downloaded for {gene}")

        probabilities=[]
        intensities=[]
        for start in range(0,len(paths),args.batch):
            crops=[resize(load_crop(path),args.size) for path in paths[start:start+args.batch]]
            intensities.extend(crop[3].flatten() for crop in crops)
            composed=[]
            for crop in crops:
                value=crop.clone()
                value[3]=crop[3]
                composed.append(resize(subcell_input(value),IMAGE_SIZE))
            batch=torch.stack(composed).to(device)
            with torch.no_grad():
                _,probability=classifier(batch)
            probabilities.append(probability.cpu())

        for path in paths:
            path.unlink(missing_ok=True)
        folder.rmdir()
        probability=torch.cat(probabilities)
        control=to_groups(probability).mean(0)
        reference=to_groups(torch.tensor(empirical[gene]["mean_probability"]))
        observed=float(total_variation(control[None],reference[None])[0])
        null=null_distribution(
            reference,
            n_reference=int(empirical[gene]["cells"]),
            n_generated=len(probability),
            name="total_variation",
            trials=args.null_trials,
        )
        intensity=torch.cat(intensities)
        threshold=float(null.quantile(0.95))
        results[gene]={
            "cells":len(probability),
            "control_grouped":control.tolist(),
            "empirical_grouped":reference.tolist(),
            "total_variation":observed,
            "null_median":float(null.median()),
            "null_p95":threshold,
            "passes":observed<=threshold,
            "real_median":float(intensity.median()),
            "real_p995":float(intensity.quantile(0.995)),
        }
        print(
            f"  {gene:<11} TV={observed:.4f} null95={threshold:.4f} "
            f"{'pass' if observed<=threshold else 'FAIL'}",
            flush=True,
        )

    report={
        "genes":genes,
        "size":args.size,
        "null_trials":args.null_trials,
        "passes":all(result["passes"] for result in results.values()),
        "results":results,
    }
    args.out.write_text(json.dumps(report,indent=2))
    print(f"wrote {args.out}; gate={'pass' if report['passes'] else 'FAIL'}",flush=True)


if __name__=="__main__":
    main()
