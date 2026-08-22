"""Conditioning inputs and the dropout that makes them optional.

Two requirements turn out to be one mechanism. OpenCell has no microtubule or ER channel, so
the model must accept a subset of the landmark channels. Classifier-free guidance needs an
unconditional pass, which is the same thing taken to its limit. Implementing them separately
would mean two ways for a condition to be absent, and two chances to get it wrong.
"""
from __future__ import annotations

import torch
from torch import Tensor,nn


class ChannelDropout(nn.Module):
    """Replace absent conditioning channels with a learned constant plane.

    A zero plane would be ambiguous: a genuinely dark channel and an absent one would look
    identical. A learned value per channel lets the network tell them apart, and the presence
    pattern is additionally signalled through the conditioning vector so it never has to
    infer absence from pixel statistics.
    """

    def __init__(self,num_channels:int,conditioning_dim:int,dropout:float=0.15)->None:
        super().__init__()
        self.num_channels=num_channels
        self.dropout=dropout
        self.null_plane=nn.Parameter(torch.zeros(num_channels))
        self.pattern=nn.Embedding(2**num_channels,conditioning_dim)
        nn.init.normal_(self.pattern.weight,std=0.02)

    def sample_mask(self,batch:int,device:torch.device,generator:torch.Generator|None=None)->Tensor:
        keep=torch.rand(batch,self.num_channels,device=device,generator=generator)>=self.dropout
        return keep.float()

    def pattern_index(self,mask:Tensor)->Tensor:
        weights=2**torch.arange(self.num_channels,device=mask.device)
        return (mask.long()*weights).sum(-1)

    def forward(self,condition:Tensor,mask:Tensor|None=None)->tuple[Tensor,Tensor]:
        """Return the masked condition and the embedding of which channels survived."""
        if mask is None:
            mask=torch.ones(condition.shape[0],self.num_channels,device=condition.device)
        plane=mask[:,:,None,None]
        masked=condition*plane+self.null_plane[None,:,None,None]*(1-plane)
        return masked,self.pattern(self.pattern_index(mask))


class LabelEmbedding(nn.Module):
    """Class embedding with a reserved null index, so guidance and absence share a path."""

    def __init__(self,num_classes:int,conditioning_dim:int,dropout:float=0.1)->None:
        super().__init__()
        self.num_classes=num_classes
        self.dropout=dropout
        self.embedding=nn.Embedding(num_classes+1,conditioning_dim)
        nn.init.normal_(self.embedding.weight,std=0.02)

    @property
    def null_index(self)->int:
        return self.num_classes

    def drop(self,labels:Tensor,generator:torch.Generator|None=None)->Tensor:
        drop=torch.rand(labels.shape,device=labels.device,generator=generator)<self.dropout
        return torch.where(drop,torch.full_like(labels,self.null_index),labels)

    def forward(self,labels:Tensor|None,batch:int=0,device:torch.device|None=None)->Tensor:
        if labels is None:
            labels=torch.full((batch,),self.null_index,device=device,dtype=torch.long)
        return self.embedding(labels)


class StateEmbedding(nn.Module):
    """Cell-state covariates, taken from the landmark channels the model already receives.

    A protein's position in one cell is often a readout of that cell's state rather than a
    property of the protein: CDC20 sits in the nucleoplasm through G2/M and is degraded in G1,
    NR3C1 stays cytosolic until it is liganded. Measured on real cells, integrated DNA and cell
    area account for 6% to 33% of a gene's between-cell localization heterogeneity
    (`notes/state-decomposition.md`), which is information the model is given and cannot easily
    use: extracting integrated nuclear intensity out of a pixel grid is possible for a
    convolutional or attention stack but nothing about the architecture privileges it.

    Handing those scalars in directly through the same adaLN path as the timestep costs a few
    hundred parameters and makes the ablation clean - the same model with and without.

    Values arrive raw and in wildly different units, so they are standardised against running
    statistics rather than assumed pre-scaled. Dropout uses a learned null vector so the state
    can be absent exactly the way a channel or a label can, which keeps guidance over state on
    the same mechanism as everything else.
    """

    def __init__(self,num_covariates:int,conditioning_dim:int,dropout:float=0.1,
                 momentum:float=0.01)->None:
        super().__init__()
        self.num_covariates=num_covariates
        self.dropout=dropout
        self.momentum=momentum
        self.project=nn.Sequential(
            nn.Linear(num_covariates,conditioning_dim),
            nn.SiLU(),
            nn.Linear(conditioning_dim,conditioning_dim),
        )
        self.null_state=nn.Parameter(torch.zeros(conditioning_dim))
        self.register_buffer("running_mean",torch.zeros(num_covariates))
        self.register_buffer("running_var",torch.ones(num_covariates))
        nn.init.zeros_(self.project[-1].weight)
        nn.init.zeros_(self.project[-1].bias)

    def standardise(self,state:Tensor)->Tensor:
        if self.training:
            with torch.no_grad():
                mean=state.mean(0)
                var=state.var(0,unbiased=False).clamp_min(1e-8)
                self.running_mean.mul_(1-self.momentum).add_(self.momentum*mean)
                self.running_var.mul_(1-self.momentum).add_(self.momentum*var)
        return (state-self.running_mean)/self.running_var.clamp_min(1e-8).sqrt()

    def sample_mask(self,batch:int,device:torch.device,
                    generator:torch.Generator|None=None)->Tensor:
        return (torch.rand(batch,device=device,generator=generator)>=self.dropout).float()

    def forward(self,state:Tensor|None,batch:int,device:torch.device,
                mask:Tensor|None=None)->Tensor:
        if state is None:
            return self.null_state[None].expand(batch,-1)
        if state.shape[-1]!=self.num_covariates:
            raise ValueError(f"expected {self.num_covariates} covariates, "
                             f"got {state.shape[-1]}")
        embedded=self.project(self.standardise(state.float()))
        if mask is None:
            mask=self.sample_mask(len(state),device) if self.training \
                else torch.ones(len(state),device=device)
        keep=mask[:,None]
        return embedded*keep+self.null_state[None]*(1-keep)
