"""Does the generated protein channel's intensity scale change what SubCell calls it?

CELL-Diff's output for TUBB4B peaks at 55/255 where the real landmark channels reach 255.
`subcell_input` normalises against a percentile anchored on the microtubule channel, so a
uniformly dimmer protein channel does not normalise away - it stays dim relative to the anchor.
If classification moves with that scale, then any D_cal measured against real cells is partly a
brightness artefact rather than a localization result.

This answers it without generating anything: take real cached crops, scale only the protein
channel, and watch the classification. The answer is known in advance for the identity scale,
which is what makes it a control.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import torch

from pvs.data.crops import PROTEIN, resize
from pvs.eval.localization import CLASS_NAMES, to_groups
from pvs.eval.subcell import IMAGE_SIZE, SubCellEnsemble, subcell_input


def entropy(p:torch.Tensor)->torch.Tensor:
    return -(p.clamp_min(1e-12)*p.clamp_min(1e-12).log()).sum(-1)


def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--cache",type=Path,default=Path("/home/claude/data/cache256"))
    parser.add_argument("--genes",type=int,default=6)
    parser.add_argument("--cells",type=int,default=8)
    parser.add_argument("--scales",type=float,nargs="+",
                        default=[1.0,0.6,0.4,0.216,0.1,1.6])
    parser.add_argument("--encoder",type=Path,
                        default=Path("/home/claude/data/subcell/all_channels_ViT-ProtS-Pool.pth"))
    parser.add_argument("--classifiers",type=Path,
                        default=Path("/home/claude/data/subcell/classifiers"))
    args=parser.parse_args()

    torch.set_num_threads(2)
    model=SubCellEnsemble.from_directory(args.encoder,args.classifiers).eval()
    shards=sorted(glob.glob(str(args.cache/"genes"/"*.npz")))[:args.genes]

    crops=[]
    for shard in shards:
        with np.load(shard,allow_pickle=False) as blob:
            for array in blob["x"][:args.cells]:
                crops.append(torch.from_numpy(array.astype("float32")/255.0).permute(2,0,1))
    print(f"{len(crops)} real cells from {len(shards)} genes\n")

    reference=None
    print(f"{'protein scale':>14}{'agreement':>11}{'mean |dp|':>11}"
          f"{'P(Negative)':>13}{'heterogeneity':>15}")
    for scale in args.scales:
        probabilities=[]
        for start in range(0,len(crops),8):
            batch=[]
            for crop in crops[start:start+8]:
                scaled=crop.clone()
                scaled[PROTEIN]=(scaled[PROTEIN]*scale).clamp(0,1)
                batch.append(resize(subcell_input(scaled),IMAGE_SIZE))
            with torch.no_grad():
                _,probability=model(torch.stack(batch))
            probabilities.append(probability)
        probability=torch.cat(probabilities)
        if reference is None:
            reference=probability
        agreement=float((probability.argmax(1)==reference.argmax(1)).float().mean())
        grouped=to_groups(probability)
        grouped=grouped/grouped.sum(-1,keepdim=True).clamp_min(1e-12)
        information=float(entropy(grouped.mean(0))-entropy(grouped).mean())
        print(f"{scale:>14.3f}{agreement:>11.2f}"
              f"{float((probability-reference).abs().mean()):>11.4f}"
              f"{float(probability[:,CLASS_NAMES.index('Negative')].mean()):>13.4f}"
              f"{information:>15.4f}",flush=True)


if __name__=="__main__":
    main()
