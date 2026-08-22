"""Conversions between the eps, x0 and v prediction targets.

A network can be trained to predict any of the three; they carry the same information but
behave very differently at the ends of the schedule. eps-prediction divides by sqrt(alpha_bar)
to recover x0 and so degenerates as the terminal SNR reaches zero, which is exactly where
x0-prediction is well conditioned, and vice versa at t=0. v-prediction interpolates between
them and stays well conditioned at both ends, which is why it is the prerequisite for any
zero-terminal-SNR schedule.

Salimans and Ho, Progressive Distillation, 2022.
"""
from __future__ import annotations

from typing import Literal

from torch import Tensor

from pvs.diffusion.schedule import NoiseSchedule,extract

Target=Literal["eps","x0","v"]
TARGETS=("eps","x0","v")


def _coefficients(schedule:NoiseSchedule,t:Tensor,ndim:int)->tuple[Tensor,Tensor]:
    return (extract(schedule.sqrt_alpha_bars,t,ndim),
            extract(schedule.sqrt_one_minus_alpha_bars,t,ndim))


def v_target(schedule:NoiseSchedule,x0:Tensor,noise:Tensor,t:Tensor)->Tensor:
    signal,noise_scale=_coefficients(schedule,t,x0.ndim)
    return signal*noise-noise_scale*x0


def x0_from_v(schedule:NoiseSchedule,xt:Tensor,v:Tensor,t:Tensor)->Tensor:
    signal,noise_scale=_coefficients(schedule,t,xt.ndim)
    return signal*xt-noise_scale*v


def eps_from_v(schedule:NoiseSchedule,xt:Tensor,v:Tensor,t:Tensor)->Tensor:
    signal,noise_scale=_coefficients(schedule,t,xt.ndim)
    return noise_scale*xt+signal*v


def to_eps(schedule:NoiseSchedule,prediction:Tensor,xt:Tensor,t:Tensor,target:Target)->Tensor:
    """Normalize any prediction target to eps, which every sampler here consumes."""
    if target=="eps":
        return prediction
    if target=="v":
        return eps_from_v(schedule,xt,prediction,t)
    if target=="x0":
        signal,noise_scale=_coefficients(schedule,t,xt.ndim)
        return (xt-signal*prediction)/noise_scale
    raise ValueError(f"unknown target {target!r}, expected one of {TARGETS}")


def training_target(schedule:NoiseSchedule,x0:Tensor,noise:Tensor,t:Tensor,target:Target)->Tensor:
    if target=="eps":
        return noise
    if target=="x0":
        return x0
    if target=="v":
        return v_target(schedule,x0,noise,t)
    raise ValueError(f"unknown target {target!r}, expected one of {TARGETS}")


def as_eps_predictor(model,schedule:NoiseSchedule,target:Target):
    """Wrap a model so the samplers can stay parameterization-agnostic."""
    if target=="eps":
        return model
    return lambda xt,t,*a,**k:to_eps(schedule,model(xt,t,*a,**k),xt,t,target)
