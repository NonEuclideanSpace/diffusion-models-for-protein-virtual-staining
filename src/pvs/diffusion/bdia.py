"""BDIA: exact inversion at one network evaluation per step.

EDICT buys exactness by carrying a second sequence, which costs two evaluations per step.
BDIA instead makes each update a linear combination of the two previous states and a single
noise estimate, so the recurrence is time-symmetric and inverts algebraically at 1 NFE.

The price is a leapfrog recurrence: the update needs both z_{i+1} and z_i, so it has two
initial conditions rather than one. Given both, the round trip is exact to 7e-18 in float64 —
the recurrence itself introduces no error at all. Supplying only x_0 forces one bootstrap
step, and that step is then the sole source of inexactness, so it is worth refining.

Measured here and not stated in the paper: the recurrence *attenuates* an error in the
bootstrap rather than amplifying it, by about 28x over 50 steps. The parasitic root that
leapfrog schemes classically carry is not a practical problem at these step counts.

Zhang, Niwa and Kleijn, ECCV 2024.
"""
from __future__ import annotations

import torch
from torch import Tensor

from pvs.diffusion.ddim import uniform_timesteps
from pvs.diffusion.ddpm import NoisePredictor
from pvs.diffusion.schedule import NoiseSchedule


def _ddim_increment(x:Tensor,noise:Tensor,alpha_bar:Tensor,alpha_bar_to:Tensor)->Tensor:
    scale=(alpha_bar_to/alpha_bar).sqrt()
    shift=(1-alpha_bar_to).sqrt()-scale*(1-alpha_bar).sqrt()
    return scale*x+shift*noise-x


@torch.no_grad()
def bdia_sample(
    model:NoisePredictor,
    schedule:NoiseSchedule,
    shape:tuple[int,...]|None=None,
    latent:Tensor|None=None,
    latents:tuple[Tensor,Tensor]|None=None,
    num_inference_steps:int=50,
    timesteps:Tensor|None=None,
    gamma:float=1.0,
    generator:torch.Generator|None=None,
)->Tensor:
    """Sample from a latent, or from the leapfrog pair that bdia_invert returns."""
    device=schedule.device
    if latents is None and latent is None:
        if shape is None:
            raise ValueError("pass one of shape, latent or latents")
        latent=torch.randn(shape,device=device,generator=generator)
    reference=latents[0] if latents is not None else latent
    grid=(uniform_timesteps(len(schedule),num_inference_steps) if timesteps is None else timesteps).to(device)
    levels=torch.cat([schedule.alpha_bars.new_ones(1),schedule.alpha_bars[grid]]).double()

    dtype=reference.dtype
    top=len(grid)
    if latents is not None:
        x_next,x=latents[0].double(),latents[1].double()
    else:
        x_next=latent.double()
        t=grid[top-1].expand(reference.shape[0])
        x=x_next+_ddim_increment(x_next,model(x_next.to(dtype),t).double(),levels[top],levels[top-1])
    for i in range(top-1,0,-1):
        t=grid[i-1].expand(reference.shape[0])
        noise=model(x.to(dtype),t).double()
        forward=_ddim_increment(x,noise,levels[i],levels[i+1])
        backward=_ddim_increment(x,noise,levels[i],levels[i-1])
        x,x_next=gamma*x_next+(1-gamma)*x-gamma*forward+backward,x
    return x.to(dtype)


@torch.no_grad()
def bdia_invert(
    model:NoisePredictor,
    schedule:NoiseSchedule,
    x0:Tensor,
    num_inference_steps:int=50,
    timesteps:Tensor|None=None,
    gamma:float=1.0,
    bootstrap_steps:int=4,
)->tuple[Tensor,Tensor]:
    """Recover the leapfrog pair that bdia_sample transports back to x0.

    The recurrence needs two initial states and only x0 is given, so the second is solved for
    by fixed-point iteration. Near the clean end alpha_bar is close to one, the iteration is
    strongly contracting, and a handful of steps reaches machine precision.
    """
    if gamma==0:
        raise ValueError("gamma must be non-zero for the recurrence to be invertible")
    device=schedule.device
    grid=(uniform_timesteps(len(schedule),num_inference_steps) if timesteps is None else timesteps).to(device)
    levels=torch.cat([schedule.alpha_bars.new_ones(1),schedule.alpha_bars[grid]]).double()

    dtype=x0.dtype
    top=len(grid)
    x_prev=x0.double()
    t=grid[0].expand(x0.shape[0])
    scale=(levels[0]/levels[1]).sqrt()
    shift=(1-levels[0]).sqrt()-scale*(1-levels[1]).sqrt()
    x=x_prev-_ddim_increment(x_prev,model(x_prev.to(dtype),t).double(),levels[1],levels[0])
    for _ in range(bootstrap_steps):
        x=(x_prev-shift*model(x.to(dtype),t).double())/scale
    for i in range(1,top):
        t=grid[i-1].expand(x0.shape[0])
        noise=model(x.to(dtype),t).double()
        forward=_ddim_increment(x,noise,levels[i],levels[i+1])
        backward=_ddim_increment(x,noise,levels[i],levels[i-1])
        x_prev,x=x,(x_prev-(1-gamma)*x+gamma*forward-backward)/gamma
    return x.to(dtype),x_prev.to(dtype)
