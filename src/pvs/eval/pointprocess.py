"""Module M3: treat punctate compartments as a point process rather than an image.

Landmark channels carry almost no information about where an individual vesicle sits, so a
per-pixel loss asks the model for something the condition does not determine, and rewards the
blurred average that satisfies it. The alternative is to score the *statistics* of the pattern:
how many puncta, how big, how clustered.

Everything here is torch so that the detector runs on the same device as the model and nothing
is added to the dependency list. Ripley's K uses the isotropic edge correction, with the
per-pair weight estimated by sampling each circle against the cell mask, because cells are
arbitrary shapes and the rectangular-window formulas do not apply to them.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def gaussian_kernel(sigma:float,device:torch.device,dtype:torch.dtype)->Tensor:
    radius=max(int(math.ceil(3*sigma)),1)
    grid=torch.arange(-radius,radius+1,device=device,dtype=dtype)
    kernel=torch.exp(-0.5*(grid/sigma)**2)
    return kernel/kernel.sum()


def blur(image:Tensor,sigma:float)->Tensor:
    kernel=gaussian_kernel(sigma,image.device,image.dtype)
    n=len(kernel)
    padded=F.pad(image[None,None],(n//2,n//2,n//2,n//2),mode="reflect")
    out=F.conv2d(padded,kernel.view(1,1,1,n))
    return F.conv2d(out,kernel.view(1,1,n,1))[0,0]


def laplacian_of_gaussian(image:Tensor,sigma:float)->Tensor:
    """Scale-normalised LoG, so responses are comparable across sigma."""
    smoothed=blur(image,sigma)
    kernel=torch.tensor([[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]],
                        device=image.device,dtype=image.dtype)
    padded=F.pad(smoothed[None,None],(1,1,1,1),mode="reflect")
    return -(sigma**2)*F.conv2d(padded,kernel.view(1,1,3,3))[0,0]


def detect_spots(
    image:Tensor,
    mask:Tensor|None=None,
    sigmas:tuple[float,...]=(1.0,1.6,2.5,4.0),
    threshold:float=0.05,
    min_distance:int=2,
)->tuple[Tensor,Tensor,Tensor]:
    """Multi-scale LoG blob detection.

    Returns (positions as (n, 2) in row-column order, radii, response strengths). A maximum must
    beat its neighbours in space *and* across neighbouring scales, which is what stops one blob
    being reported once per sigma.
    """
    if image.ndim!=2:
        raise ValueError(f"expected a single plane, got shape {tuple(image.shape)}")
    stack=torch.stack([laplacian_of_gaussian(image,s) for s in sigmas])
    window=2*min_distance+1
    spatial=F.max_pool2d(stack[None],window,stride=1,padding=min_distance)[0]
    across=F.max_pool1d(spatial.permute(1,2,0).reshape(-1,1,len(sigmas)),
                        3,stride=1,padding=1)[:,0].reshape(*spatial.shape[1:],len(sigmas))
    peak=(stack==spatial)&(stack==across.permute(2,0,1))&(stack>threshold)
    if mask is not None:
        peak=peak&(mask>0)[None]
    index=peak.nonzero()
    if not len(index):
        empty=torch.zeros((0,2),device=image.device,dtype=image.dtype)
        return empty,empty[:,0],empty[:,0]
    scale=torch.tensor(sigmas,device=image.device,dtype=image.dtype)[index[:,0]]
    return (index[:,1:].to(image.dtype),scale*math.sqrt(2),
            stack[index[:,0],index[:,1],index[:,2]])


def nearest_neighbour(points:Tensor)->Tensor:
    if len(points)<2:
        return torch.zeros(0,device=points.device,dtype=points.dtype)
    distance=torch.cdist(points,points)
    distance.fill_diagonal_(float("inf"))
    return distance.min(dim=1).values


def _circle_weights(points:Tensor,radii:Tensor,mask:Tensor,samples:int=32)->Tensor:
    """Fraction of each circle that falls inside the mask, for Ripley's isotropic correction."""
    angle=torch.arange(samples,device=points.device,dtype=points.dtype)*(2*math.pi/samples)
    offset=torch.stack([angle.sin(),angle.cos()],dim=1)
    probe=points[:,None,None,:]+radii[None,:,None,None]*offset[None,None,:,:]
    rows=probe[...,0].round().long().clamp(0,mask.shape[0]-1)
    columns=probe[...,1].round().long().clamp(0,mask.shape[1]-1)
    inside=(mask[rows,columns]>0).to(points.dtype)
    outside=((probe[...,0]<0)|(probe[...,0]>mask.shape[0]-1)|
             (probe[...,1]<0)|(probe[...,1]>mask.shape[1]-1))
    return (inside*(~outside).to(points.dtype)).mean(dim=2).clamp_min(1.0/samples)


def ripley_k(points:Tensor,mask:Tensor,radii:Tensor,samples:int=32)->Tensor:
    """Isotropic-corrected K(r) over the region the mask defines."""
    n=len(points)
    area=float((mask>0).sum())
    if n<2 or area<=0:
        return torch.zeros_like(radii)
    distance=torch.cdist(points,points)
    distance.fill_diagonal_(float("inf"))
    weights=_circle_weights(points,radii,mask,samples)
    counted=[]
    for column,radius in enumerate(radii):
        within=(distance<=radius).to(points.dtype)/weights[:,column:column+1]
        counted.append(within.sum())
    return area/(n*(n-1))*torch.stack(counted)


def ripley_l(points:Tensor,mask:Tensor,radii:Tensor,samples:int=32)->Tensor:
    """L(r) - r. Zero under complete spatial randomness, positive when clustered."""
    return (ripley_k(points,mask,radii,samples)/math.pi).clamp_min(0).sqrt()-radii


def summarize(image:Tensor,mask:Tensor|None=None,radii:Tensor|None=None,**kwargs)->dict:
    positions,sizes,strengths=detect_spots(image,mask,**kwargs)
    window=mask if mask is not None else torch.ones_like(image)
    radii=radii if radii is not None else torch.linspace(2,20,10,device=image.device,
                                                         dtype=image.dtype)
    neighbours=nearest_neighbour(positions)
    area=float((window>0).sum())
    return {
        "count":len(positions),
        "density":len(positions)/max(area,1.0),
        "median_radius":float(sizes.median()) if len(sizes) else 0.0,
        "median_strength":float(strengths.median()) if len(strengths) else 0.0,
        "median_nearest":float(neighbours.median()) if len(neighbours) else 0.0,
        "ripley_l":ripley_l(positions,window,radii,**{}),
        "radii":radii,
    }
