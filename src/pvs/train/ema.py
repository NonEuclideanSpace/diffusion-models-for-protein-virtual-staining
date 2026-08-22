"""Exponential moving average of model parameters."""
from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn


class EMA:
    """Shadow copy of the weights, averaged over training.

    Diffusion sample quality depends on this much more than on the last raw checkpoint;
    evaluating the online weights understates a model by a wide margin.
    """

    def __init__(self,model:nn.Module,decay:float=0.9999,warmup:int=0)->None:
        if not 0<decay<1:
            raise ValueError(f"decay must lie in (0, 1), got {decay}")
        self.decay=decay
        self.warmup=warmup
        self.steps=0
        self.shadow=deepcopy(model).eval().requires_grad_(False)

    def rate(self)->float:
        if self.steps<self.warmup:
            return min(self.decay,(1+self.steps)/(10+self.steps))
        return self.decay

    @torch.no_grad()
    def update(self,model:nn.Module)->None:
        rate=self.rate()
        for shadow,live in zip(self.shadow.state_dict().values(),model.state_dict().values()):
            if shadow.dtype.is_floating_point:
                shadow.mul_(rate).add_(live.detach(),alpha=1-rate)
            else:
                shadow.copy_(live)
        self.steps+=1

    def state_dict(self)->dict:
        return {"decay":self.decay,"warmup":self.warmup,"steps":self.steps,
                "shadow":self.shadow.state_dict()}

    def load_state_dict(self,state:dict)->None:
        self.decay=state["decay"]; self.warmup=state["warmup"]; self.steps=state["steps"]
        self.shadow.load_state_dict(state["shadow"])
