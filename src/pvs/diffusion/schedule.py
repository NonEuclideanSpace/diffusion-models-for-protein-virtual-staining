"""Noise schedules for the diffusion forward process."""
import torch
from torch import Tensor
def linear_beta_schedule(
        num_steps:int,
        beta_start:float=1e-4,
        beta_end:float=0.02,
)->Tensor:
    """Linear beta schedule from DDPM (Ho et al., 2020)."""
    return torch.linspace(beta_start,beta_end,num_steps,dtype=torch.float64)
