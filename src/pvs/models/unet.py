"""Conditional U-Net baseline.

The point of this model is not to win. It is the reference the diffusion transformer has to
beat in Phase 1; if the transformer cannot match a competent U-Net on held-out data, the
transformer has a bug, and every downstream conclusion would rest on it.
"""
from __future__ import annotations

import torch
from torch import Tensor,nn

from pvs.models.blocks import Downsample,ResidualBlock,SelfAttention,Upsample,normalization
from pvs.models.conditioning import ChannelDropout,LabelEmbedding
from pvs.models.embeddings import TimestepEmbedder


class ConditionalUNet(nn.Module):
    """Predicts the diffusion target for `target_channels`, conditioned on landmark channels.

    Conditioning enters three ways: landmark channels concatenated to the input, a timestep
    embedding, and an optional class label. The latter two are summed into one vector that
    modulates every residual block through a scale and shift.
    """

    def __init__(
        self,
        target_channels:int=1,
        condition_channels:int=3,
        base_width:int=64,
        width_multipliers:tuple[int,...]=(1,2,4,4),
        blocks_per_level:int=2,
        attention_resolutions:tuple[int,...]=(16,8),
        num_classes:int|None=None,
        channel_dropout:float=0.15,
        label_dropout:float=0.1,
        dropout:float=0.0,
        image_size:int=64,
    )->None:
        super().__init__()
        self.target_channels=target_channels
        self.condition_channels=condition_channels
        conditioning_dim=base_width*4

        self.time=TimestepEmbedder(conditioning_dim)
        self.channels=ChannelDropout(condition_channels,conditioning_dim,channel_dropout) if condition_channels else None
        self.labels=LabelEmbedding(num_classes,conditioning_dim,label_dropout) if num_classes else None

        self.stem=nn.Conv2d(target_channels+condition_channels,base_width,3,padding=1)

        widths=[base_width*m for m in width_multipliers]
        self.down=nn.ModuleList()
        skip_widths=[base_width]
        width=base_width
        resolution=image_size
        for level,target in enumerate(widths):
            for _ in range(blocks_per_level):
                stage=nn.ModuleList([ResidualBlock(width,target,conditioning_dim,dropout)])
                width=target
                if resolution in attention_resolutions:
                    stage.append(SelfAttention(width))
                self.down.append(stage)
                skip_widths.append(width)
            if level<len(widths)-1:
                self.down.append(nn.ModuleList([Downsample(width)]))
                skip_widths.append(width)
                resolution//=2

        self.middle=nn.ModuleList([
            ResidualBlock(width,width,conditioning_dim,dropout),
            SelfAttention(width),
            ResidualBlock(width,width,conditioning_dim,dropout),
        ])

        self.up=nn.ModuleList()
        for level,target in reversed(list(enumerate(widths))):
            for block in range(blocks_per_level+1):
                stage=nn.ModuleList([ResidualBlock(width+skip_widths.pop(),target,conditioning_dim,dropout)])
                width=target
                if resolution in attention_resolutions:
                    stage.append(SelfAttention(width))
                if level and block==blocks_per_level:
                    stage.append(Upsample(width))
                    resolution*=2
                self.up.append(stage)

        self.head=nn.Sequential(normalization(width),nn.SiLU(),nn.Conv2d(width,target_channels,3,padding=1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self,
        x:Tensor,
        t:Tensor,
        condition:Tensor|None=None,
        labels:Tensor|None=None,
        channel_mask:Tensor|None=None,
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

        h=self.stem(x)
        skips=[h]
        for stage in self.down:
            for module in stage:
                h=module(h,conditioning) if isinstance(module,ResidualBlock) else module(h)
            skips.append(h)
        for module in self.middle:
            h=module(h,conditioning) if isinstance(module,ResidualBlock) else module(h)
        for stage in self.up:
            h=torch.cat([h,skips.pop()],dim=1)
            for module in stage:
                h=module(h,conditioning) if isinstance(module,ResidualBlock) else module(h)
        return self.head(h)
