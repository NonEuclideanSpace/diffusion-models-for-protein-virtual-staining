"""Residual and attention blocks shared by the convolutional backbones."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor,nn


def normalization(channels:int,groups:int=32)->nn.GroupNorm:
    """GroupNorm with the largest group count that divides `channels`.

    Skip concatenation produces widths like 48 that no fixed group count divides, and
    GroupNorm raises rather than adapting, so the divisor has to be chosen here.
    """
    for count in range(min(groups,channels),0,-1):
        if channels%count==0:
            return nn.GroupNorm(count,channels)
    raise ValueError(f"cannot normalize {channels} channels")


class ResidualBlock(nn.Module):
    """Two convolutions with the conditioning vector injected as a scale and shift.

    Scale-shift rather than a plain additive bias: it lets the conditioning modulate the
    activation's gain, which is what adaLN does in the transformer and is markedly stronger
    than addition when the conditioning has to change texture rather than offset it.
    """

    def __init__(self,in_channels:int,out_channels:int,conditioning_dim:int,dropout:float=0.0)->None:
        super().__init__()
        self.norm_in=normalization(in_channels)
        self.conv_in=nn.Conv2d(in_channels,out_channels,3,padding=1)
        self.projection=nn.Linear(conditioning_dim,2*out_channels)
        self.norm_out=normalization(out_channels)
        self.dropout=nn.Dropout(dropout)
        self.conv_out=nn.Conv2d(out_channels,out_channels,3,padding=1)
        self.skip=nn.Conv2d(in_channels,out_channels,1) if in_channels!=out_channels else nn.Identity()
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self,x:Tensor,conditioning:Tensor)->Tensor:
        h=self.conv_in(F.silu(self.norm_in(x)))
        scale,shift=self.projection(F.silu(conditioning))[:,:,None,None].chunk(2,dim=1)
        h=self.norm_out(h)*(1+scale)+shift
        return self.skip(x)+self.conv_out(self.dropout(F.silu(h)))


class SelfAttention(nn.Module):
    """Multi-head self-attention over spatial positions, with a zeroed output projection."""

    def __init__(self,channels:int,heads:int=4)->None:
        super().__init__()
        if channels%heads:
            raise ValueError(f"channels {channels} must be divisible by heads {heads}")
        self.heads=heads
        self.norm=normalization(channels)
        self.qkv=nn.Conv1d(channels,3*channels,1)
        self.out=nn.Conv1d(channels,channels,1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self,x:Tensor)->Tensor:
        b,c,h,w=x.shape
        flat=self.norm(x).reshape(b,c,h*w)
        q,k,v=self.qkv(flat).reshape(b,self.heads,3*c//self.heads,h*w).chunk(3,dim=2)
        attended=F.scaled_dot_product_attention(q.transpose(-1,-2),k.transpose(-1,-2),v.transpose(-1,-2))
        return x+self.out(attended.transpose(-1,-2).reshape(b,c,h*w)).reshape(b,c,h,w)


class Downsample(nn.Module):
    def __init__(self,channels:int)->None:
        super().__init__()
        self.op=nn.Conv2d(channels,channels,3,stride=2,padding=1)

    def forward(self,x:Tensor)->Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self,channels:int)->None:
        super().__init__()
        self.op=nn.Conv2d(channels,channels,3,padding=1)

    def forward(self,x:Tensor)->Tensor:
        return self.op(F.interpolate(x,scale_factor=2,mode="nearest"))
