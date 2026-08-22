"""DDPM noise-prediction objective and ancestral sampler."""
from __future__ import annotations

from typing import Protocol

import torch
import torch.nn.functional as F
from torch import Tensor

from pvs.diffusion.forward import predict_x0_from_noise,q_posterior,q_sample
from pvs.diffusion.schedule import NoiseSchedule


class NoisePredictor(Protocol):
    def __call__(self,xt:Tensor,t:Tensor)->Tensor: ...


def ddpm_loss(model:NoisePredictor,schedule:NoiseSchedule,x0:Tensor,t:Tensor|None=None)->Tensor:
    """Simplified objective of Ho et al. (2020): MSE between true and predicted noise."""
    if t is None:
        t=torch.randint(0,len(schedule),(x0.shape[0],),device=x0.device)
    noise=torch.randn_like(x0)
    return F.mse_loss(model(q_sample(schedule,x0,t,noise),t),noise)


@torch.no_grad()
def ddpm_sample(
    model:NoisePredictor,
    schedule:NoiseSchedule,
    shape:tuple[int,...]|None=None,
    latent:Tensor|None=None,
    clip:tuple[float,float]|None=None,
    generator:torch.Generator|None=None,
)->Tensor:
    """Ancestral sampling: walk the full reverse chain from pure noise to x_0."""
    device=schedule.device
    if latent is None:
        if shape is None:
            raise ValueError("pass either shape or latent")
        latent=torch.randn(shape,device=device,generator=generator)
    x=latent
    for step in reversed(range(len(schedule))):
        t=torch.full((x.shape[0],),step,device=device,dtype=torch.long)
        x0=predict_x0_from_noise(schedule,x,t,model(x,t))
        if clip is not None:
            x0=x0.clamp(*clip)
        mean,log_variance=q_posterior(schedule,x0,x,t)
        if step==0:
            x=mean
        else:
            x=mean+(0.5*log_variance).exp()*torch.randn(x.shape,device=device,generator=generator)
    return x
