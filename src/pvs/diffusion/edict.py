"""EDICT: exactly invertible sampling through coupled latent sequences.

Plain DDIM inversion is only an approximate inverse of the sampling step, so model error
leaks into the round trip and is amplified wherever alpha_bar is small. Carrying two
sequences and updating each from the other makes every step an affine coupling layer,
which is invertible by construction: the round trip reproduces the same noise estimates
and returns to the input up to floating-point error, whatever the model predicts.

Two numerical constraints, both load-bearing:

The coupling arithmetic runs in float64. Every step multiplies by a_t on the way down and
divides by it on the way up; in float32 the cancellation leaves ~1e-8 instead of ~1e-15.

The multiplier a_t = sqrt(alpha_bar_prev / alpha_bar) diverges as alpha_bar reaches zero,
and round-trip error scales as a_T times machine epsilon. Exactness is algebraic, not
numerical: at a literal zero terminal SNR the method detonates. Keep sqrt(alpha_bar_T)
floored near 1e-3 to 1e-2, which costs almost nothing in terms of leaked signal.

Wallace, Gokul and Naik, CVPR 2023.
"""
from __future__ import annotations

import torch
from torch import Tensor

from pvs.diffusion.ddim import uniform_timesteps
from pvs.diffusion.ddpm import NoisePredictor
from pvs.diffusion.schedule import NoiseSchedule


def _coefficients(alpha_bar:Tensor,alpha_bar_next:Tensor)->tuple[Tensor,Tensor]:
    scale=(alpha_bar_next/alpha_bar).sqrt()
    return scale,(1-alpha_bar_next).sqrt()-scale*(1-alpha_bar).sqrt()


def _mix(leading:Tensor,trailing:Tensor,mixing:float)->tuple[Tensor,Tensor]:
    leading=mixing*leading+(1-mixing)*trailing
    return leading,mixing*trailing+(1-mixing)*leading


def _unmix(leading:Tensor,trailing:Tensor,mixing:float)->tuple[Tensor,Tensor]:
    trailing=(trailing-(1-mixing)*leading)/mixing
    return (leading-(1-mixing)*trailing)/mixing,trailing


@torch.no_grad()
def edict_sample(
    model:NoisePredictor,
    schedule:NoiseSchedule,
    shape:tuple[int,...]|None=None,
    latent:Tensor|None=None,
    latents:tuple[Tensor,Tensor]|None=None,
    num_inference_steps:int=50,
    timesteps:Tensor|None=None,
    mixing:float=0.93,
    generator:torch.Generator|None=None,
)->Tensor:
    device=schedule.device
    if latents is None:
        if latent is None:
            if shape is None:
                raise ValueError("pass one of shape, latent or latents")
            latent=torch.randn(shape,device=device,generator=generator)
        latents=(latent,latent)
    grid=(uniform_timesteps(len(schedule),num_inference_steps) if timesteps is None else timesteps).to(device)
    one=schedule.alpha_bars.new_ones(())
    dtype=latents[0].dtype
    x,y=latents[0].double(),latents[1].double()
    for i in reversed(range(len(grid))):
        t=grid[i].expand(x.shape[0])
        scale,shift=_coefficients(schedule.alpha_bars[grid[i]].double(),
                                  (schedule.alpha_bars[grid[i-1]] if i>0 else one).double())
        x_inter=scale*x+shift*model(y.to(dtype),t).double()
        y_inter=scale*y+shift*model(x_inter.to(dtype),t).double()
        x,y=_mix(x_inter,y_inter,mixing)
    return ((x+y)/2).to(dtype)


@torch.no_grad()
def edict_invert(
    model:NoisePredictor,
    schedule:NoiseSchedule,
    x0:Tensor,
    num_inference_steps:int=50,
    timesteps:Tensor|None=None,
    mixing:float=0.93,
)->tuple[Tensor,Tensor]:
    """Return the coupled latent pair that edict_sample transports back to x0."""
    device=schedule.device
    grid=(uniform_timesteps(len(schedule),num_inference_steps) if timesteps is None else timesteps).to(device)
    one=schedule.alpha_bars.new_ones(())
    dtype=x0.dtype
    x=y=x0.double()
    for i in range(len(grid)):
        t=grid[i].expand(x0.shape[0])
        scale,shift=_coefficients(schedule.alpha_bars[grid[i]].double(),
                                  (schedule.alpha_bars[grid[i-1]] if i>0 else one).double())
        x_inter,y_inter=_unmix(x,y,mixing)
        y=(y_inter-shift*model(x_inter.to(dtype),t).double())/scale
        x=(x_inter-shift*model(y.to(dtype),t).double())/scale
    return x.to(dtype),y.to(dtype)
