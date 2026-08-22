"""Noise schedules and the constants derived from them."""
from __future__ import annotations

import math

import torch
from torch import Tensor


def linear_beta_schedule(num_steps:int,beta_start:float=1e-4,beta_end:float=0.02)->Tensor:
    """Linear schedule from Ho et al. (2020). Endpoints are tuned for num_steps=1000."""
    return torch.linspace(beta_start,beta_end,num_steps,dtype=torch.float64)


def cosine_beta_schedule(num_steps:int,offset:float=0.008)->Tensor:
    """Cosine schedule from Nichol & Dhariwal (2021), derived by differencing alpha_bar."""
    t=torch.linspace(0,num_steps,num_steps+1,dtype=torch.float64)/num_steps
    alpha_bars=torch.cos((t+offset)/(1+offset)*math.pi/2)**2
    alpha_bars=alpha_bars/alpha_bars[0]
    return (1-alpha_bars[1:]/alpha_bars[:-1]).clip(0,0.999)


SCHEDULES={"linear":linear_beta_schedule,"cosine":cosine_beta_schedule}


class NoiseSchedule:
    """Constants of a Gaussian diffusion forward process, indexed by t in [0, T).

    Derived in float64 and stored in float32: alpha_bar is a product of T terms and
    single precision accumulates visible error in its tail.
    """

    def __init__(self,betas:Tensor)->None:
        betas=betas.double()
        if betas.ndim!=1 or not bool(((betas>0)&(betas<1)).all()):
            raise ValueError("betas must be 1-D with every value in the open interval (0, 1)")
        alphas=1-betas
        alpha_bars=alphas.cumprod(0)
        alpha_bars_prev=torch.cat([alpha_bars.new_ones(1),alpha_bars[:-1]])
        posterior_variance=betas*(1-alpha_bars_prev)/(1-alpha_bars)
        self.betas=betas.float()
        self.alphas=alphas.float()
        self.alpha_bars=alpha_bars.float()
        self.alpha_bars_prev=alpha_bars_prev.float()
        self.sqrt_alpha_bars=alpha_bars.sqrt().float()
        self.sqrt_one_minus_alpha_bars=(1-alpha_bars).sqrt().float()
        self.posterior_variance=posterior_variance.float()
        self.posterior_log_variance=posterior_variance.clamp_min(1e-20).log().float()
        self.posterior_mean_coef_x0=(betas*alpha_bars_prev.sqrt()/(1-alpha_bars)).float()
        self.posterior_mean_coef_xt=((1-alpha_bars_prev)*alphas.sqrt()/(1-alpha_bars)).float()

    @classmethod
    def make(cls,num_steps:int=1000,kind:str="linear",terminal_snr_floor:float|None=None,**kwargs)->NoiseSchedule:
        if kind not in SCHEDULES:
            raise ValueError(f"unknown schedule {kind!r}, expected one of {sorted(SCHEDULES)}")
        betas=SCHEDULES[kind](num_steps,**kwargs)
        if terminal_snr_floor is not None:
            betas=enforce_zero_terminal_snr(betas,terminal_snr_floor)
        return cls(betas)

    def to(self,device:torch.device|str)->NoiseSchedule:
        for name,value in vars(self).items():
            if isinstance(value,Tensor):
                setattr(self,name,value.to(device))
        return self

    @property
    def device(self)->torch.device:
        return self.betas.device

    def __len__(self)->int:
        return self.betas.shape[0]


def enforce_zero_terminal_snr(betas:Tensor,floor:float=1e-2)->Tensor:
    """Rescale betas so sqrt(alpha_bar) descends linearly to `floor` at the final step.

    Lin et al. (WACV 2024) show that schedules which never reach zero terminal SNR leak
    signal into x_T, so sampling from N(0, I) is starting from the wrong marginal. Their
    fix drives sqrt(alpha_bar_T) to exactly zero, which makes eps-prediction degenerate and
    makes any exact inversion scheme divide by zero on its last step.

    `floor` keeps sqrt(alpha_bar_T) small but non-zero. At 1e-2 the leaked signal is already
    an order of magnitude below a stock linear schedule while the inversion multiplier
    sqrt(alpha_bar_prev / alpha_bar_T) stays order 1.
    """
    if not 0<=floor<1:
        raise ValueError(f"floor must lie in [0, 1), got {floor}")
    sqrt_alpha_bars=(1-betas.double()).cumprod(0).sqrt()
    first,last=sqrt_alpha_bars[0].clone(),sqrt_alpha_bars[-1].clone()
    sqrt_alpha_bars=(sqrt_alpha_bars-last)*(first-floor)/(first-last)+floor
    alpha_bars=sqrt_alpha_bars**2
    alphas=torch.cat([alpha_bars[:1],alpha_bars[1:]/alpha_bars[:-1]])
    return 1-alphas


def extract(values:Tensor,t:Tensor,ndim:int)->Tensor:
    """Gather per-sample constants and reshape to (B, 1, ..., 1) for broadcasting."""
    if t.dtype!=torch.long:
        raise TypeError(f"timesteps must be int64, got {t.dtype}")
    if t.ndim!=1:
        raise ValueError(f"timesteps must be 1-D, got shape {tuple(t.shape)}")
    return values.gather(0,t).reshape(-1,*(1,)*(ndim-1))
