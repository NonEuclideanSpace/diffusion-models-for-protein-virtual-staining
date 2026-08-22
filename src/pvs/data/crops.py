"""Loading and normalizing SubCell single-cell crops.

**Channel order is (nucleus, ER, microtubules, protein).** Verified visually and numerically
against real crops rather than inferred: the nucleus channel is a single compact blob in
every cell, the microtubule channel is unmistakably filamentous, and the fourth channel
tracks the gene's annotation — correlation with the nucleus channel is 0.74 for a
nucleoplasm protein, -0.19 for a plasma-membrane protein and 0.34 for a Golgi protein.
Getting this wrong corrupts everything downstream without raising an error, so the check is
kept as a test.

Crops are 1024x1024 RGBA PNGs, 8-bit, with a separate binary mask marking the cell that the
crop is centred on. Neighbouring cells are present in the frame and have to be masked out for
any per-cell measurement.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

NUCLEUS,RETICULUM,MICROTUBULES,PROTEIN=0,1,2,3
CHANNEL_NAMES=("nucleus","reticulum","microtubules","protein")
LANDMARKS=(NUCLEUS,RETICULUM,MICROTUBULES)
NATIVE_SIZE=1024
NATIVE_PIXEL_SIZE_UM=0.0800885


def load_crop(path:str|Path)->Tensor:
    """Read one crop as a (4, H, W) float tensor scaled to [0, 1]."""
    import numpy as np
    from PIL import Image

    array=np.array(Image.open(path))
    if array.ndim!=3 or array.shape[2]!=4:
        raise ValueError(f"expected a 4-channel crop, got shape {array.shape} at {path}")
    scale=255.0 if array.dtype==np.uint8 else float(np.iinfo(array.dtype).max)
    return torch.from_numpy(array.astype("float32")/scale).permute(2,0,1)


def load_mask(path:str|Path)->Tensor:
    import numpy as np
    from PIL import Image

    return torch.from_numpy((np.array(Image.open(path))>0).astype("float32"))[None]


def normalize(
    crop:Tensor,
    percentile:float=99.5,
    anchor:int=MICROTUBULES,
    per_channel:bool=False,
)->Tensor:
    """Clip outliers, scale by a biological anchor, and map to [-1, 1].

    Following ProtiCelli: each channel is clipped at its own high percentile to remove
    saturated speckle, then **all** channels are divided by the anchor channel's percentile
    rather than by their own. Per-channel scaling would normalize away exactly the relative
    intensity that distinguishes a bright compact structure from a dim diffuse one; the
    microtubule channel is a housekeeping structure with fairly stable abundance, so it acts
    as a common reference across cells.

    `per_channel=True` restores the naive behaviour, kept only so the difference can be
    measured rather than assumed.
    """
    flat=crop.flatten(1)
    limits=torch.quantile(flat,percentile/100.0,dim=1).clamp_min(1e-6)
    clipped=torch.minimum(crop,limits[:,None,None])
    scale=limits[:,None,None] if per_channel else limits[anchor].clamp_min(1e-6)
    return (clipped/scale).clamp(0,1)*2-1


def resize(crop:Tensor,size:int)->Tensor:
    if crop.shape[-1]==size:
        return crop
    return F.interpolate(crop[None],size=(size,size),mode="area" if crop.shape[-1]>size else "bilinear",
                         align_corners=None if crop.shape[-1]>size else False)[0]


def split(crop:Tensor)->tuple[Tensor,Tensor]:
    """Separate the expensive channel from the cheap ones."""
    return crop[PROTEIN:PROTEIN+1],crop[list(LANDMARKS)]


@dataclass
class CropRecord:
    path:Path
    gene:str


class HpaCrops(torch.utils.data.Dataset):
    """Crops on disk, laid out as <root>/<gene>/<plate>_<position>_<sample>_<cell>_cell_image.png."""

    def __init__(
        self,
        root:str|Path,
        size:int=128,
        genes:list[str]|None=None,
        percentile:float=99.5,
        apply_mask:bool=False,
    )->None:
        self.root=Path(root)
        self.size=size
        self.percentile=percentile
        self.apply_mask=apply_mask
        allowed=set(genes) if genes else None
        self.records=[
            CropRecord(path,path.parent.name)
            for path in sorted(self.root.glob("*/*_cell_image.png"))
            if allowed is None or path.parent.name in allowed
        ]
        self.genes=sorted({record.gene for record in self.records})
        self.gene_index={gene:i for i,gene in enumerate(self.genes)}

    def __len__(self)->int:
        return len(self.records)

    def __getitem__(self,index:int)->dict:
        record=self.records[index]
        crop=load_crop(record.path)
        if self.apply_mask:
            mask_path=Path(str(record.path).replace("_cell_image.png","_cell_mask.png"))
            if mask_path.exists():
                crop=crop*load_mask(mask_path)
        crop=resize(normalize(crop,self.percentile),self.size)
        protein,landmarks=split(crop)
        return {"x0":protein,"condition":landmarks,
                "labels":torch.tensor(self.gene_index[record.gene]),
                "gene":record.gene,"path":str(record.path)}
