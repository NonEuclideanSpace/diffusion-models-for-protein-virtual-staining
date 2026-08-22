"""Population context measured from the landmark channels of a single-cell crop.

Snijder and Pelkmans showed that much of what looks like cell-to-cell variability in cultured
cells is set by a cell's neighbourhood - local density, position at a colony edge, how much room
it has to spread - rather than by anything intrinsic. That work measured expression levels and
phenotype strength. Nobody has asked the same question of *where a protein sits inside the cell*,
and nobody has asked it at proteome scale.

The data to ask it already exists. An HPA single-cell crop is 1024 pixels wide, the centre cell
covers about a sixth of it, and roughly seven in ten bright nucleus pixels belong to neighbours.
Every crop carries its own neighbourhood; the pipeline has been discarding it.

Definitions follow published ones so the measurement is citable rather than invented: neighbour
count and local density after Snijder 2009, the angular-gap edge rule after DBSCAN-CellX
(Kuchenhoff 2023, optimum 120-160 degrees), spreading area as the crowding readout of Frechin
2015, and boundary crowding standing in for the neighbour fraction of Sero 2015, which would
otherwise need the neighbours segmented.

Nuclei are found as maxima of a heavily blurred nucleus channel rather than by the multi-scale
LoG in `pointprocess`. LoG is tuned for vesicles and fires on chromatin texture at this scale -
it reported 29 objects inside a single nucleus. Blurring to the nucleus scale first gives one
maximum per nucleus, stable across a wide threshold range.

The centre cell's mask is free ground truth: the detector must find exactly one nucleus inside
it. `centre_hit` reports that per crop, so a bad threshold surfaces as a number rather than as a
silently wrong density.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from pvs.data.crops import NUCLEUS
from pvs.eval.pointprocess import blur

DOWNSAMPLE=8
NUCLEUS_BLUR=4.0
MIN_DISTANCE=9
THRESHOLD=0.20
RING_SCALE=4
RING=8
RADII=(256.0,384.0)
EDGE_GAP=math.radians(140.0)
PIXEL_MICRONS=0.0800885


def nuclei(plane:Tensor,threshold:float=THRESHOLD)->Tensor:
    """Nucleus centroids in full-crop pixel coordinates, row-column order."""
    small=F.avg_pool2d(plane[None,None],DOWNSAMPLE)[0,0]
    smoothed=blur(small/small.max().clamp_min(1e-6),NUCLEUS_BLUR)
    window=2*MIN_DISTANCE+1
    peak=((smoothed==F.max_pool2d(smoothed[None,None],window,stride=1,padding=MIN_DISTANCE)[0,0])
          &(smoothed>threshold))
    return peak.nonzero().to(plane.dtype)*DOWNSAMPLE


def covariates(crop:Tensor,mask:Tensor,threshold:float=THRESHOLD)->dict[str,float]:
    """Population context for the cell at the centre of a crop."""
    plane=crop[NUCLEUS]
    sheet=mask[0] if mask.ndim==3 else mask
    points=nuclei(plane,threshold)
    inside=torch.zeros(len(points),dtype=torch.bool)
    if len(points):
        rows=points[:,0].long().clamp(0,sheet.shape[0]-1)
        columns=points[:,1].long().clamp(0,sheet.shape[1]-1)
        inside=sheet[rows,columns]>0

    centre=(points[inside].mean(0) if bool(inside.any())
            else torch.tensor([sheet.shape[0]/2,sheet.shape[1]/2],dtype=plane.dtype))
    others=points[~inside]
    offset=others-centre
    distance=offset.norm(dim=1) if len(others) else torch.zeros(0)

    result={"centre_hit":float(inside.sum()),"detected":float(len(points))}
    for radius in RADII:
        near=int((distance<radius).sum())
        result[f"neighbours_{int(radius)}"]=float(near)
        result[f"density_{int(radius)}"]=near/(math.pi*(radius*PIXEL_MICRONS)**2)
    result["nearest"]=float(distance.min()*PIXEL_MICRONS) if len(distance) else float("nan")

    if len(others)>=2:
        angle=torch.atan2(offset[:,0],offset[:,1]).sort().values
        gaps=torch.cat([angle[1:]-angle[:-1],(angle[0]+2*math.pi-angle[-1])[None]])
        result["angular_gap"]=float(gaps.max())
    else:
        result["angular_gap"]=2*math.pi
    result["edge"]=float(result["angular_gap"]>EDGE_GAP)

    result["spread_area"]=float(sheet.sum())*PIXEL_MICRONS**2
    # A 31-wide dilation at full resolution costs 945 ms per cell, which is 38 s per gene for a
    # number that a quarter-resolution mask reproduces. Everything below the nucleus detector
    # works on the downsampled planes for the same reason.
    thumb=F.avg_pool2d(plane[None,None],RING_SCALE)[0,0]
    small=F.avg_pool2d(sheet[None,None].float(),RING_SCALE)[0,0]
    grown=F.max_pool2d(small[None,None],2*(RING//2)+1,stride=1,padding=RING//2)[0,0]
    ring=(grown>0)&(small==0)
    result["boundary_crowding"]=float(thumb[ring].mean()) if bool(ring.any()) else 0.0

    # Detector-free replicates. Every quantity above depends on nucleus detection, and detection
    # is hardest exactly where it matters most - in crowded fields, where nuclei merge and a
    # local-maximum detector under-counts. These two need no detection at all and are monotone in
    # how much other-cell nuclear material surrounds the centre cell, so they replicate any
    # context result without sharing its failure mode.
    outside=small==0
    result["outside_signal"]=float(thumb[outside].mean()) if bool(outside.any()) else 0.0
    bright=thumb>thumb.flatten().quantile(0.90)
    result["outside_bright"]=float((bright&outside).float().mean())

    # A crop taken near the edge of its field of view is padded, which truncates the neighbourhood
    # and fakes a low density. Flag it rather than silently measuring it.
    result["padding"]=float((F.avg_pool2d(crop[None],RING_SCALE)[0].abs().sum(0)==0).float().mean())
    return result


FIELDS=("centre_hit","detected","neighbours_256","density_256","neighbours_384","density_384",
        "nearest","angular_gap","edge","spread_area","boundary_crowding",
        "outside_signal","outside_bright","padding")

DETECTOR_FREE=("outside_signal","outside_bright","spread_area","boundary_crowding")
