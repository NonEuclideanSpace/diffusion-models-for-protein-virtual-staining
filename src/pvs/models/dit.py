"""Diffusion transformer with adaLN-Zero conditioning.

Conditioning splits by type rather than by convenience. Landmark channels are spatial and
enter by concatenation before patchify, the way every landmark-to-protein model does it.
Everything without spatial extent — timestep, cell line, protein identity, and later the
discrete mode variable of module M1 — is summed into one vector that modulates every block
through adaLN-Zero.

adaLN-Zero produces per-block shift, scale and gate from that vector, with the gate zero
initialized so each block begins as the identity and the network starts as a clean residual
path. Peebles and Xie report this beating cross-attention and in-context conditioning, and it
is also what makes adding a new conditioning source later a matter of one more summand.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor,nn

from pvs.models.conditioning import ChannelDropout,LabelEmbedding,StateEmbedding
from pvs.models.embeddings import TimestepEmbedder


def sincos_position_embedding(dim:int,height:int,width:int)->Tensor:
    """Fixed two-dimensional sine-cosine embedding, half the channels per axis."""
    if dim%4:
        raise ValueError(f"embedding dim {dim} must be divisible by 4 for a 2-D grid")
    quarter=dim//4
    frequencies=torch.exp(-math.log(10000.0)*torch.arange(quarter,dtype=torch.float32)/quarter)
    rows=torch.arange(height,dtype=torch.float32)[:,None]*frequencies[None]
    cols=torch.arange(width,dtype=torch.float32)[:,None]*frequencies[None]
    row=torch.cat([rows.sin(),rows.cos()],dim=-1)[:,None].expand(height,width,2*quarter)
    col=torch.cat([cols.sin(),cols.cos()],dim=-1)[None].expand(height,width,2*quarter)
    return torch.cat([row,col],dim=-1).reshape(height*width,dim)


def modulate(x:Tensor,shift:Tensor,scale:Tensor)->Tensor:
    return x*(1+scale.unsqueeze(1))+shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """Pre-norm attention and MLP, each modulated and gated by the conditioning vector."""

    def __init__(self,hidden_size:int,heads:int,mlp_ratio:float=4.0)->None:
        super().__init__()
        self.norm_attention=nn.LayerNorm(hidden_size,elementwise_affine=False,eps=1e-6)
        self.attention=nn.MultiheadAttention(hidden_size,heads,batch_first=True)
        self.norm_mlp=nn.LayerNorm(hidden_size,elementwise_affine=False,eps=1e-6)
        inner=int(hidden_size*mlp_ratio)
        self.mlp=nn.Sequential(nn.Linear(hidden_size,inner),nn.GELU(approximate="tanh"),nn.Linear(inner,hidden_size))
        self.modulation=nn.Sequential(nn.SiLU(),nn.Linear(hidden_size,6*hidden_size))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self,x:Tensor,conditioning:Tensor)->Tensor:
        shift_a,scale_a,gate_a,shift_m,scale_m,gate_m=self.modulation(conditioning).chunk(6,dim=-1)
        h=modulate(self.norm_attention(x),shift_a,scale_a)
        x=x+gate_a.unsqueeze(1)*self.attention(h,h,h,need_weights=False)[0]
        h=modulate(self.norm_mlp(x),shift_m,scale_m)
        return x+gate_m.unsqueeze(1)*self.mlp(h)


class FinalLayer(nn.Module):
    def __init__(self,hidden_size:int,patch_size:int,out_channels:int)->None:
        super().__init__()
        self.norm=nn.LayerNorm(hidden_size,elementwise_affine=False,eps=1e-6)
        self.modulation=nn.Sequential(nn.SiLU(),nn.Linear(hidden_size,2*hidden_size))
        self.projection=nn.Linear(hidden_size,patch_size*patch_size*out_channels)
        for module in (self.modulation[-1],self.projection):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self,x:Tensor,conditioning:Tensor)->Tensor:
        shift,scale=self.modulation(conditioning).chunk(2,dim=-1)
        return self.projection(modulate(self.norm(x),shift,scale))


class DiT(nn.Module):
    """Diffusion transformer over patches of a landmark-conditioned image."""

    def __init__(
        self,
        image_size:int=64,
        patch_size:int=4,
        target_channels:int=1,
        condition_channels:int=3,
        hidden_size:int=384,
        depth:int=12,
        heads:int=6,
        mlp_ratio:float=4.0,
        num_classes:int|None=None,
        num_modes:int|None=None,
        num_state:int|None=None,
        channel_dropout:float=0.15,
        label_dropout:float=0.1,
        state_dropout:float=0.1,
    )->None:
        super().__init__()
        if image_size%patch_size:
            raise ValueError(f"image size {image_size} must be divisible by patch size {patch_size}")
        self.image_size=image_size
        self.patch_size=patch_size
        self.target_channels=target_channels
        self.condition_channels=condition_channels
        self.grid=image_size//patch_size

        self.patchify=nn.Conv2d(target_channels+condition_channels,hidden_size,patch_size,stride=patch_size)
        self.register_buffer("positions",sincos_position_embedding(hidden_size,self.grid,self.grid),persistent=False)

        self.time=TimestepEmbedder(hidden_size)
        self.channels=ChannelDropout(condition_channels,hidden_size,channel_dropout) if condition_channels else None
        self.labels=LabelEmbedding(num_classes,hidden_size,label_dropout) if num_classes else None
        self.state=StateEmbedding(num_state,hidden_size,state_dropout) if num_state else None
        self.modes=nn.Embedding(num_modes,hidden_size) if num_modes else None
        if self.modes is not None:
            nn.init.normal_(self.modes.weight,std=0.02)

        self.blocks=nn.ModuleList(DiTBlock(hidden_size,heads,mlp_ratio) for _ in range(depth))
        self.final=FinalLayer(hidden_size,patch_size,target_channels)

    def unpatchify(self,x:Tensor)->Tensor:
        b=x.shape[0]
        x=x.reshape(b,self.grid,self.grid,self.patch_size,self.patch_size,self.target_channels)
        return x.permute(0,5,1,3,2,4).reshape(b,self.target_channels,self.image_size,self.image_size)

    def forward(
        self,
        x:Tensor,
        t:Tensor,
        condition:Tensor|None=None,
        labels:Tensor|None=None,
        channel_mask:Tensor|None=None,
        modes:Tensor|None=None,
        state:Tensor|None=None,
        state_mask:Tensor|None=None,
    )->Tensor:
        conditioning=self.time(t)
        if self.channels is not None:
            if condition is None:
                raise ValueError(f"model expects {self.condition_channels} conditioning channels")
            condition,pattern=self.channels(condition,channel_mask)
            conditioning=conditioning+pattern
            x=torch.cat([x,condition],dim=1)
        if self.labels is not None:
            conditioning=conditioning+self.labels(labels,x.shape[0],x.device)
        if self.state is not None:
            conditioning=conditioning+self.state(state,x.shape[0],x.device,state_mask)
        if self.modes is not None and modes is not None:
            conditioning=conditioning+self.modes(modes)

        tokens=self.patchify(x).flatten(2).transpose(1,2)+self.positions
        for block in self.blocks:
            tokens=block(tokens,conditioning)
        return self.unpatchify(self.final(tokens,conditioning))


def dit_small(**kwargs)->DiT:
    return DiT(hidden_size=384,depth=12,heads=6,**kwargs)


def dit_base(**kwargs)->DiT:
    return DiT(hidden_size=768,depth=12,heads=12,**kwargs)


def dit_large(**kwargs)->DiT:
    return DiT(hidden_size=1024,depth=24,heads=16,**kwargs)
