"""Module M2: how much does each landmark channel tell the model about each compartment.

The quantity is the increase in denoising loss when a conditioning channel is withheld:

    delta[k, compartment] = E[ L(x | c without k) ] - E[ L(x | c) ]

read as a landmark-by-compartment matrix. Biologically it answers which reference stain is
worth acquiring for which class of target protein.

Two things have to be right or the number is meaningless.

**It is not mutual information.** L is an upper bound on the negative log likelihood, and the
difference of two upper bounds bounds nothing. Delta is a proxy, and it is confounded by how
well the model fits each condition. The partial mitigation is to use one model trained with
channel dropout for both terms, so capacity is matched; a residual bias remains and its
direction has to be stated wherever the matrix is reported.

**The two expectations must share their randomness.** Delta is a difference of two large,
nearly equal numbers. Estimating them from independent noise draws buries the signal in
Monte Carlo error; drawing the timestep and the noise once and evaluating both conditions on
them makes the estimator paired, and the variance falls by more than an order of magnitude.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor,nn

from pvs.diffusion.forward import q_sample
from pvs.diffusion.parameterization import Target,training_target
from pvs.diffusion.schedule import NoiseSchedule


@dataclass
class InformationMatrix:
    delta:Tensor
    standard_error:Tensor
    counts:Tensor
    channel_names:tuple[str,...]
    compartment_names:tuple[str,...]

    def significant(self,sigma:float=2.0)->Tensor:
        return self.delta.abs()>sigma*self.standard_error

    def report(self)->str:
        header="channel".ljust(14)+"".join(name[:11].rjust(13) for name in self.compartment_names)
        lines=[header]
        for row,name in enumerate(self.channel_names):
            cells=[]
            for column in range(self.delta.shape[1]):
                mark="*" if self.significant()[row,column] else " "
                cells.append(f"{self.delta[row,column]:+.4f}{mark}".rjust(13))
            lines.append(name.ljust(14)+"".join(cells))
        lines.append("* differs from zero by more than two standard errors")
        return "\n".join(lines)


@torch.no_grad()
def landmark_information(
    model:nn.Module,
    schedule:NoiseSchedule,
    batches,
    num_batches:int,
    num_channels:int,
    num_compartments:int,
    channel_names:tuple[str,...],
    compartment_names:tuple[str,...],
    target:Target="eps",
    seed:int=0,
)->InformationMatrix:
    """Paired estimate of the loss increase from withholding each conditioning channel."""
    device=next(model.parameters()).device
    totals=torch.zeros(num_channels,num_compartments,device=device)
    squares=torch.zeros(num_channels,num_compartments,device=device)
    counts=torch.zeros(num_compartments,device=device)
    generator=torch.Generator(device=device).manual_seed(seed)

    for _ in range(num_batches):
        batch=next(batches)
        x0=batch["x0"].to(device)
        condition=batch["condition"].to(device)
        labels=batch["labels"].to(device)
        compartments=batch["compartments"].to(device)
        size=x0.shape[0]

        t=torch.randint(0,len(schedule),(size,),device=device,generator=generator)
        noise=torch.randn(x0.shape,device=device,generator=generator)
        xt=q_sample(schedule,x0,t,noise)
        goal=training_target(schedule,x0,noise,t,target)

        full_mask=torch.ones(size,num_channels,device=device)
        per_sample=lambda mask:F.mse_loss(model(xt,t,condition,labels,mask),goal,
                                          reduction="none").flatten(1).mean(1)
        baseline=per_sample(full_mask)

        for channel in range(num_channels):
            mask=full_mask.clone()
            mask[:,channel]=0.0
            delta=per_sample(mask)-baseline
            totals[channel].index_add_(0,compartments,delta)
            squares[channel].index_add_(0,compartments,delta**2)
        counts.index_add_(0,compartments,torch.ones(size,device=device))

    counts=counts.clamp_min(1)
    mean=totals/counts
    variance=(squares/counts-mean**2).clamp_min(0)
    return InformationMatrix(
        delta=mean.cpu(),
        standard_error=(variance/counts).sqrt().cpu(),
        counts=counts.cpu(),
        channel_names=channel_names,
        compartment_names=compartment_names,
    )
