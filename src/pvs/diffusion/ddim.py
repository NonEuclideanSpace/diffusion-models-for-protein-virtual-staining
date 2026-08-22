"""DDIM sampling and inversion."""
from __future__ import annotations

import torch
from torch import Tensor

from pvs.diffusion.ddpm import NoisePredictor
from pvs.diffusion.schedule import NoiseSchedule


def uniform_timesteps(num_steps:int,num_inference_steps:int)->Tensor:
    if not 0<num_inference_steps<=num_steps:
        raise ValueError(f"num_inference_steps must lie in (0, {num_steps}], got {num_inference_steps}")
    return torch.linspace(0,num_steps-1,num_inference_steps).round().long()


def _transport(
    x:Tensor,
    noise:Tensor,
    alpha_bar:Tensor,
    alpha_bar_next:Tensor,
    eta:float,
    generator:torch.Generator|None,
)->Tensor:
    """Move x from noise level alpha_bar to alpha_bar_next. Deterministic when eta is 0."""
    x0=(x-(1-alpha_bar).sqrt()*noise)/alpha_bar.sqrt()
    if eta==0:
        return alpha_bar_next.sqrt()*x0+(1-alpha_bar_next).sqrt()*noise
    sigma=eta*((1-alpha_bar_next)/(1-alpha_bar)).sqrt()*(1-alpha_bar/alpha_bar_next).sqrt()
    direction=(1-alpha_bar_next-sigma**2).clamp_min(0).sqrt()*noise
    return alpha_bar_next.sqrt()*x0+direction+sigma*torch.randn(x.shape,device=x.device,generator=generator)


@torch.no_grad()
def ddim_sample(
    model:NoisePredictor,
    schedule:NoiseSchedule,
    shape:tuple[int,...]|None=None,
    latent:Tensor|None=None,
    num_inference_steps:int=50,
    timesteps:Tensor|None=None,
    eta:float=0.0,
    generator:torch.Generator|None=None,
)->Tensor:
    """Sample from noise, or transport a supplied latent back to data."""
    device=schedule.device
    if latent is None:
        if shape is None:
            raise ValueError("pass either shape or latent")
        latent=torch.randn(shape,device=device,generator=generator)
    grid=(uniform_timesteps(len(schedule),num_inference_steps) if timesteps is None else timesteps).to(device)
    one=schedule.alpha_bars.new_ones(())
    x=latent
    for i in reversed(range(len(grid))):
        t=grid[i].expand(x.shape[0])
        alpha_bar=schedule.alpha_bars[grid[i]]
        alpha_bar_next=schedule.alpha_bars[grid[i-1]] if i>0 else one
        x=_transport(x,model(x,t),alpha_bar,alpha_bar_next,eta,generator)
    return x


@torch.no_grad()
def ddim_invert(
    model:NoisePredictor,
    schedule:NoiseSchedule,
    x0:Tensor,
    num_inference_steps:int=50,
    timesteps:Tensor|None=None,
    fixed_point_steps:int=0,
)->Tensor:
    """Map a sample back to the latent that ddim_sample would transport to it.

    The plain form reuses the noise estimate at the current point as a stand-in for the
    one at the target point. That lag is the dominant error and it compounds. Each
    fixed-point step re-solves the sampling update for its own input instead.
    """
    device=schedule.device
    grid=(uniform_timesteps(len(schedule),num_inference_steps) if timesteps is None else timesteps).to(device)
    x=x0
    alpha_bar=schedule.alpha_bars.new_ones(())
    for i in range(len(grid)):
        t_here=(grid[i-1] if i>0 else grid.new_zeros(())).expand(x0.shape[0])
        t_next=grid[i].expand(x0.shape[0])
        alpha_bar_next=schedule.alpha_bars[grid[i]]
        target=_transport(x,model(x,t_here),alpha_bar,alpha_bar_next,0.0,None)
        scale=(alpha_bar/alpha_bar_next).sqrt()
        offset=(1-alpha_bar).sqrt()-scale*(1-alpha_bar_next).sqrt()
        for _ in range(fixed_point_steps):
            target=(x-offset*model(target,t_next))/scale
        x=target
        alpha_bar=alpha_bar_next
    return x
