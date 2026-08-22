"""Divergences between compartment distributions, and the noise floor of each estimator.

Plug-in KL is the obvious choice and the wrong one. It is unbounded, positively biased, and
its bias grows with the sparsity of either argument — so at the sample sizes HPA actually
provides it does not merely lose power, it inverts: a model that collapses modes scores as
better calibrated than one that does not. Any estimator used here has to have its null
distribution measured at the sample size it will meet, which is what `null_distribution` is
for.
"""
from __future__ import annotations

from typing import Callable,Literal

import torch
from torch import Tensor

Divergence=Literal["kl","jensen_shannon","total_variation","hellinger"]
EPS=1e-10


def _normalize(counts:Tensor)->Tensor:
    total=counts.sum(-1,keepdim=True).clamp_min(EPS)
    return counts/total


def categorical_kl(p:Tensor,q:Tensor)->Tensor:
    """KL(p || q). Unbounded, and explodes wherever q has an empty bin that p does not."""
    p,q=_normalize(p)+EPS,_normalize(q)+EPS
    return (p*(p/q).log()).sum(-1)


def jensen_shannon(p:Tensor,q:Tensor)->Tensor:
    """Symmetric, bounded above by log 2, finite even when the supports differ."""
    p,q=_normalize(p),_normalize(q)
    m=0.5*(p+q)
    return 0.5*(categorical_kl(p,m)+categorical_kl(q,m))


def total_variation(p:Tensor,q:Tensor)->Tensor:
    """Half the L1 distance: the fraction of probability mass in the wrong bin."""
    return 0.5*(_normalize(p)-_normalize(q)).abs().sum(-1)


def hellinger(p:Tensor,q:Tensor)->Tensor:
    return ((_normalize(p).sqrt()-_normalize(q).sqrt())**2).sum(-1).sqrt()/(2**0.5)


DIVERGENCES:dict[str,Callable[[Tensor,Tensor],Tensor]]={
    "kl":categorical_kl,
    "jensen_shannon":jensen_shannon,
    "total_variation":total_variation,
    "hellinger":hellinger,
}


def divergence(name:Divergence)->Callable[[Tensor,Tensor],Tensor]:
    if name not in DIVERGENCES:
        raise ValueError(f"unknown divergence {name!r}, expected one of {sorted(DIVERGENCES)}")
    return DIVERGENCES[name]


def counts_from_labels(labels:Tensor,num_classes:int)->Tensor:
    return torch.bincount(labels,minlength=num_classes).float()


def null_distribution(
    reference:Tensor,
    n_reference:int,
    n_generated:int,
    name:Divergence="jensen_shannon",
    trials:int=2000,
    generator:torch.Generator|None=None,
)->Tensor:
    """Divergence when both samples come from the same distribution.

    This is the floor a measured value has to clear before it means anything, and it depends
    on sample size and class count far more than on the divergence being used.
    """
    reference=_normalize(reference)
    draw=lambda n:torch.distributions.Multinomial(n,reference).sample((trials,))
    return divergence(name)(draw(n_generated),draw(n_reference))


def detection_threshold(
    reference:Tensor,
    n_reference:int,
    n_generated:int,
    name:Divergence="jensen_shannon",
    quantile:float=0.95,
    trials:int=2000,
)->float:
    """Value the null exceeds only `1 - quantile` of the time."""
    return float(null_distribution(reference,n_reference,n_generated,name,trials).quantile(quantile))


def collapse_toward_mode(reference:Tensor,weight:float)->Tensor:
    """Mix the reference toward a point mass on its dominant bin, as guidance is claimed to."""
    reference=_normalize(reference)
    peak=torch.zeros_like(reference).scatter_(-1,reference.argmax(-1,keepdim=True),1.0)
    return (1-weight)*reference+weight*peak


def average_ranks(values:Tensor)->Tensor:
    """Ranks with ties averaged, so equal values cannot be ordered by accident."""
    order=values.argsort()
    ranks=torch.empty(values.numel(),dtype=torch.float32)
    ranks[order]=torch.arange(values.numel(),dtype=torch.float32)
    sorted_values=values[order]
    start=0
    for index in range(1,values.numel()+1):
        if index==values.numel() or sorted_values[index]!=sorted_values[start]:
            if index-start>1:
                ranks[order[start:index]]=ranks[order[start:index]].mean()
            start=index
    return ranks


def trend_statistic(values:Tensor)->float:
    """Spearman-style monotone trend score over a guidance sweep, in [-1, 1].

    The M1 hypothesis is about a trend, not a level, and a trend statistic keeps the shared
    noise in the reference distribution from dominating, because the same reference is used
    at every guidance weight.
    """
    if values.ndim!=1 or values.numel()<3:
        raise ValueError("need a one-dimensional sweep of at least three points")
    n=values.numel()
    ranks=average_ranks(values)
    positions=torch.arange(n,dtype=ranks.dtype)
    centred=lambda v:v-v.mean()
    # A flat sweep has no rank variance. Ordinal ranking would silently return +1 for it,
    # because argsort is stable and hands back the original order - which would let a sweep
    # that does nothing pass a gate that asks for a strictly increasing trend.
    if float(centred(ranks).norm())<EPS:
        return 0.0
    return float((centred(ranks)*centred(positions)).sum()/
                 (centred(ranks).norm()*centred(positions).norm()).clamp_min(EPS))
