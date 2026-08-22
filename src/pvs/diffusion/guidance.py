"""Classifier-free guidance.

Guidance is the independent variable of module M1: the claim under test is that pushing the
weight up sharpens a single dominant mode rather than sampling the conditional distribution.
So the implementation has to make the weight easy to sweep, and has to be explicit about
which conditions are dropped for the unconditional pass — dropping the label and dropping the
landmark channels are different interventions and will not produce the same curve.
"""
from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

Dropped=Literal["labels","channels","both"]


def classifier_free_guidance(
    model,
    weight:float,
    drop:Dropped="labels",
    batched:bool=True,
):
    """Wrap a conditional model so it returns the guided prediction.

    weight 0 is the unconditional model, 1 is the conditional model unchanged, and above 1
    extrapolates away from the unconditional prediction.
    """
    if drop not in ("labels","channels","both"):
        raise ValueError(f"unknown drop mode {drop!r}, expected labels, channels or both")

    def unconditional_arguments(condition,labels,channel_mask,batch,device):
        if drop in ("labels","both"):
            labels=None
        if drop in ("channels","both") and condition is not None:
            channel_mask=torch.zeros(batch,condition.shape[1],device=device)
        return condition,labels,channel_mask

    def guided(x:Tensor,t:Tensor,condition:Tensor|None=None,labels:Tensor|None=None,
               channel_mask:Tensor|None=None)->Tensor:
        if weight==1.0:
            return model(x,t,condition,labels,channel_mask)
        null_condition,null_labels,null_mask=unconditional_arguments(
            condition,labels,channel_mask,x.shape[0],x.device)
        if batched and labels is not None:
            twice=lambda v:None if v is None else torch.cat([v,v],dim=0)
            merged_labels=torch.cat([labels,torch.full_like(labels,_null_index(model))],dim=0)
            merged_mask=(torch.cat([channel_mask,null_mask],dim=0)
                         if channel_mask is not None and null_mask is not None else None)
            if merged_mask is None and null_mask is not None:
                ones=torch.ones_like(null_mask)
                merged_mask=torch.cat([ones,null_mask],dim=0)
            both=model(twice(x),twice(t),twice(condition),merged_labels,merged_mask)
            conditional,unconditional=both.chunk(2,dim=0)
        else:
            conditional=model(x,t,condition,labels,channel_mask)
            unconditional=model(x,t,null_condition,null_labels,null_mask)
        return unconditional+weight*(conditional-unconditional)

    return guided


def _null_index(model)->int:
    labels=getattr(model,"labels",None)
    if labels is None:
        raise ValueError("model has no label embedding, so labels cannot be dropped")
    return labels.null_index
