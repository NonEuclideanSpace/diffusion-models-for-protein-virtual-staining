"""Embeddings shared by every conditional backbone."""
from __future__ import annotations

import math

import torch
from torch import Tensor,nn


def timestep_embedding(t:Tensor,dim:int,max_period:float=10000.0)->Tensor:
    """Sinusoidal position embedding of a diffusion timestep."""
    half=dim//2
    freqs=torch.exp(-math.log(max_period)*torch.arange(half,device=t.device,dtype=torch.float32)/half)
    args=t.float()[:,None]*freqs[None]
    embedding=torch.cat([args.cos(),args.sin()],dim=-1)
    if dim%2:
        embedding=torch.cat([embedding,embedding.new_zeros(embedding.shape[0],1)],dim=-1)
    return embedding


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep features passed through a small MLP."""

    def __init__(self,hidden_size:int,frequency_size:int=256)->None:
        super().__init__()
        self.frequency_size=frequency_size
        self.mlp=nn.Sequential(
            nn.Linear(frequency_size,hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size,hidden_size),
        )

    def forward(self,t:Tensor)->Tensor:
        return self.mlp(timestep_embedding(t,self.frequency_size))
