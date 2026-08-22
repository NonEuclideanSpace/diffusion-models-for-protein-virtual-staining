"""Closed-form forward process and the identities derived from it."""
from __future__ import annotations

import torch
from torch import Tensor

from pvs.diffusion.schedule import NoiseSchedule,extract


def q_sample(schedule:NoiseSchedule,x0:Tensor,t:Tensor,noise:Tensor|None=None)->Tensor:
    """Draw x_t ~ q(x_t | x_0) directly, without iterating the chain."""
    if noise is None:
        noise=torch.randn_like(x0)
    return (extract(schedule.sqrt_alpha_bars,t,x0.ndim)*x0
            +extract(schedule.sqrt_one_minus_alpha_bars,t,x0.ndim)*noise)


def predict_x0_from_noise(schedule:NoiseSchedule,xt:Tensor,t:Tensor,noise:Tensor)->Tensor:
    """Invert q_sample for x_0 given a noise estimate."""
    return ((xt-extract(schedule.sqrt_one_minus_alpha_bars,t,xt.ndim)*noise)
            /extract(schedule.sqrt_alpha_bars,t,xt.ndim))


def predict_noise_from_x0(schedule:NoiseSchedule,xt:Tensor,t:Tensor,x0:Tensor)->Tensor:
    return ((xt-extract(schedule.sqrt_alpha_bars,t,xt.ndim)*x0)
            /extract(schedule.sqrt_one_minus_alpha_bars,t,xt.ndim))


def q_posterior(schedule:NoiseSchedule,x0:Tensor,xt:Tensor,t:Tensor)->tuple[Tensor,Tensor]:
    """Mean and log-variance of the tractable posterior q(x_{t-1} | x_t, x_0)."""
    mean=(extract(schedule.posterior_mean_coef_x0,t,xt.ndim)*x0
          +extract(schedule.posterior_mean_coef_xt,t,xt.ndim)*xt)
    return mean,extract(schedule.posterior_log_variance,t,xt.ndim)


def signal_to_noise_ratio(schedule:NoiseSchedule)->Tensor:
    return schedule.alpha_bars/(1-schedule.alpha_bars)
