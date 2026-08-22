"""Minimal noise predictors used to validate the diffusion machinery."""
from __future__ import annotations

import torch
from torch import Tensor,nn

from pvs.diffusion.schedule import NoiseSchedule,extract
from pvs.models.embeddings import TimestepEmbedder


class ToyMLP(nn.Module):
    """Noise predictor for low-dimensional data, conditioned on t by addition."""

    def __init__(self,data_dim:int=2,hidden_size:int=256,depth:int=4)->None:
        super().__init__()
        self.time=TimestepEmbedder(hidden_size)
        self.input=nn.Linear(data_dim,hidden_size)
        self.blocks=nn.ModuleList(
            nn.Sequential(nn.Linear(hidden_size,hidden_size),nn.SiLU()) for _ in range(depth)
        )
        self.output=nn.Linear(hidden_size,data_dim)

    def forward(self,xt:Tensor,t:Tensor)->Tensor:
        h=self.input(xt)+self.time(t)
        for block in self.blocks:
            h=h+block(h)
        return self.output(h)


class AnalyticGaussianPredictor:
    """Exact E[noise | x_t] when the data is an isotropic Gaussian of a known scale.

    Gives the diffusion code an oracle to be tested against, with no training involved.
    """

    def __init__(self,schedule:NoiseSchedule,data_std:float=1.0)->None:
        self.schedule=schedule
        self.data_std=data_std

    def __call__(self,xt:Tensor,t:Tensor)->Tensor:
        alpha_bar=extract(self.schedule.alpha_bars,t,xt.ndim)
        variance=alpha_bar*self.data_std**2+(1-alpha_bar)
        return (1-alpha_bar).sqrt()*xt/variance
