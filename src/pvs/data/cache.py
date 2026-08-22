"""Crops cached at training resolution, one compressed shard per gene.

A 1024x1024 crop costs 4.27 MB and a 256x256 one costs 78 KB, so the whole U2OS selection
fits in under 5 GB instead of 187 GB. The saving is only useful if random access stays cheap:
each shard decompresses as a unit, so an LRU of whole shards plus a sampler that keeps a batch
inside a handful of shards turns one decompression into forty samples.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from .crops import NUCLEUS, normalize, resize, split

STATE_NAMES=("dna","nucleus_area","chromatin_density","cell_area")


def state_covariates(crop:Tensor,mask:Tensor|None=None)->Tensor:
    """Cell-state descriptors, from channels the model is already conditioned on.

    Integrated DNA and chromatin density are standard cell-cycle proxies; a G2/M nucleus holds
    twice the DNA without doubling its area, which is why both are kept. Measured on real cells
    these account for up to 78% of a gene's between-cell localization heterogeneity
    (`notes/state-decomposition.md`), so they are not decoration.

    Without a mask the numbers pick up neighbouring cells in the frame, which is why
    `build_cache.py --masks` exists. The unmasked version is still ordered correctly within a
    field and is accepted rather than refused, so a cache built without masks stays usable.
    """
    window=mask if mask is not None else torch.ones_like(crop[NUCLEUS])
    nucleus=crop[NUCLEUS]*window
    lit=nucleus>nucleus[nucleus>0].median() if bool((nucleus>0).any()) else nucleus>0
    area=lit.sum().float()
    return torch.stack([
        nucleus.sum(),
        area,
        nucleus[lit].mean() if bool(lit.any()) else torch.zeros((),dtype=crop.dtype),
        window.sum().float(),
    ])


@dataclass
class Shard:
    gene:str
    path:Path
    cells:int
    kind:str
    location:str


def read_manifest(root:str|Path)->list[Shard]:
    root=Path(root)
    shards=[]
    for path in sorted((root/"genes").glob("*.npz")):
        with np.load(path,allow_pickle=False) as blob:
            meta=blob["meta"]
            shards.append(Shard(path.stem,path,int(blob["x"].shape[0]),str(meta[0]),str(meta[1])))
    return shards


class ShardCache:
    def __init__(self,capacity:int=24)->None:
        self.capacity=capacity
        self.entries:OrderedDict[Path,tuple]=OrderedDict()

    def get(self,path:Path)->tuple[np.ndarray,np.ndarray|None]:
        if path in self.entries:
            self.entries.move_to_end(path)
            return self.entries[path]
        with np.load(path,allow_pickle=False) as blob:
            entry=(blob["x"],blob["mask"] if "mask" in blob.files else None)
        self.entries[path]=entry
        if len(self.entries)>self.capacity:
            self.entries.popitem(last=False)
        return entry


class CachedCrops(torch.utils.data.Dataset):
    """Same contract as HpaCrops, backed by per-gene shards instead of loose PNGs."""

    def __init__(
        self,
        root:str|Path,
        size:int=128,
        genes:list[str]|None=None,
        percentile:float=99.5,
        capacity:int=24,
        state:bool=False,
    )->None:
        allowed=set(genes) if genes else None
        self.shards=[s for s in read_manifest(root) if allowed is None or s.gene in allowed]
        if not self.shards:
            raise ValueError(f"no shards under {root}")
        self.size=size
        self.percentile=percentile
        self.state=state
        self.cache=ShardCache(capacity)
        self.genes=sorted(s.gene for s in self.shards)
        self.gene_index={gene:i for i,gene in enumerate(self.genes)}
        self.offsets=np.cumsum([0]+[s.cells for s in self.shards])

    def __len__(self)->int:
        return int(self.offsets[-1])

    def locate(self,index:int)->tuple[Shard,int]:
        shard=int(np.searchsorted(self.offsets,index,side="right")-1)
        return self.shards[shard],index-int(self.offsets[shard])

    def __getitem__(self,index:int)->dict:
        shard,offset=self.locate(index)
        arrays,masks=self.cache.get(shard.path)
        crop=torch.from_numpy(arrays[offset].astype("float32")/255.0).permute(2,0,1)
        item={"labels":torch.tensor(self.gene_index[shard.gene]),"gene":shard.gene}
        if self.state:
            mask=(torch.from_numpy(masks[offset].astype("float32"))
                  if masks is not None else None)
            item["state"]=state_covariates(crop,mask)
        crop=resize(normalize(crop,self.percentile),self.size)
        protein,landmarks=split(crop)
        return {**item,"x0":protein,"condition":landmarks}


class ShardShuffleSampler(torch.utils.data.Sampler):
    """Shuffle shard order and cells within each shard, never across the whole set.

    Uniform shuffling would decompress one shard per sample. This keeps every decompression
    paying for all of its cells while still giving the optimizer a different order each epoch,
    which is the property that actually matters.
    """

    def __init__(self,dataset:CachedCrops,block:int=8,seed:int=0)->None:
        self.dataset=dataset
        self.block=block
        self.seed=seed
        self.epoch=0

    def set_epoch(self,epoch:int)->None:
        self.epoch=epoch

    def __len__(self)->int:
        return len(self.dataset)

    def __iter__(self):
        generator=torch.Generator().manual_seed(self.seed+self.epoch)
        order=torch.randperm(len(self.dataset.shards),generator=generator).tolist()
        for start in range(0,len(order),self.block):
            indices=[
                int(self.dataset.offsets[shard])+offset
                for shard in order[start:start+self.block]
                for offset in range(self.dataset.shards[shard].cells)
            ]
            picks=torch.randperm(len(indices),generator=generator).tolist()
            yield from (indices[p] for p in picks)
