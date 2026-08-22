"""A synthetic virtual-staining task with known conditional distributions.

The real question — does a conditional diffusion model reproduce the distribution over
compartments, or collapse onto the dominant one — cannot be answered on real data until the
estimator is known to work. On real data the reference distribution is itself a prediction.
Here it is known by construction, so the whole measurement chain can be validated first.

Structure deliberately mirrors HPA: three landmark channels that are cheap to acquire, one
protein channel that is not, and a protein identity whose localization is stochastic across
cells. Each protein carries a fixed categorical distribution over compartments; a cell is
rendered by drawing one compartment from it. The distribution is the ground truth that
`pi_emp` stands in for on real data.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

COMPARTMENTS=("nucleoplasm","nuclear_rim","cytosol","membrane","vesicles")
NUM_COMPARTMENTS=len(COMPARTMENTS)


def _grid(size:int,device:torch.device)->tuple[Tensor,Tensor]:
    axis=torch.linspace(-1,1,size,device=device)
    return torch.meshgrid(axis,axis,indexing="ij")


def _ellipse(yy:Tensor,xx:Tensor,centre:Tensor,radii:Tensor,angle:Tensor)->Tensor:
    """Signed radial coordinate: below 1 inside the ellipse, 1 on its boundary."""
    cos,sin=angle.cos()[:,None,None],angle.sin()[:,None,None]
    dy=yy[None]-centre[:,0,None,None]
    dx=xx[None]-centre[:,1,None,None]
    u=(cos*dx+sin*dy)/radii[:,1,None,None]
    v=(-sin*dx+cos*dy)/radii[:,0,None,None]
    return (u*u+v*v).sqrt()


def _soft(x:Tensor,edge:float=0.06)->Tensor:
    return torch.sigmoid(-x/edge)


@dataclass
class SyntheticConfig:
    image_size:int=64
    num_proteins:int=64
    concentration:float=0.6
    noise:float=0.05
    vesicle_count:int=14


class SyntheticCells:
    """Renders cells and knows the exact compartment distribution of every protein."""

    def __init__(self,config:SyntheticConfig|None=None,seed:int=0,device:str|torch.device="cpu")->None:
        self.config=config or SyntheticConfig()
        self.device=torch.device(device)
        generator=torch.Generator(device="cpu").manual_seed(seed)
        weights=torch.distributions.Dirichlet(
            torch.full((NUM_COMPARTMENTS,),self.config.concentration)
        ).sample((self.config.num_proteins,))
        self.distributions=weights.to(self.device)
        self.yy,self.xx=_grid(self.config.image_size,self.device)

    def entropy(self)->Tensor:
        """Per-protein entropy of the compartment distribution, in nats."""
        p=self.distributions.clamp_min(1e-12)
        return -(p*p.log()).sum(-1)

    def sample_compartments(self,labels:Tensor,generator:torch.Generator|None=None)->Tensor:
        probabilities=self.distributions[labels]
        return torch.multinomial(probabilities,1,generator=generator).squeeze(-1)

    def _geometry(self,batch:int,generator:torch.Generator|None)->dict[str,Tensor]:
        randn=lambda *s:torch.randn(*s,device=self.device,generator=generator)
        rand=lambda *s:torch.rand(*s,device=self.device,generator=generator)
        centre=randn(batch,2)*0.08
        cell_radii=0.62+rand(batch,2)*0.16
        nucleus_radii=cell_radii*(0.36+rand(batch,2)*0.10)
        angle=rand(batch)*torch.pi
        cell=_ellipse(self.yy,self.xx,centre,cell_radii,angle)
        nucleus=_ellipse(self.yy,self.xx,centre,nucleus_radii,angle)
        return {"centre":centre,"angle":angle,"cell":cell,"nucleus":nucleus}

    def _landmarks(self,geometry:dict[str,Tensor],generator:torch.Generator|None)->Tensor:
        cell,nucleus=geometry["cell"],geometry["nucleus"]
        inside_cell=_soft(cell-1.0)
        dna=_soft(nucleus-1.0)
        radial=(1-(cell-1).abs().clamp(0,1))*inside_cell
        spokes=(torch.atan2(self.yy-geometry["centre"][:,0,None,None],
                            self.xx-geometry["centre"][:,1,None,None])*6).cos().abs()
        microtubules=(radial*spokes*(1-dna)).clamp(0,1)
        reticulum=((nucleus*3).sin().abs()*inside_cell*(1-dna)).clamp(0,1)
        stack=torch.stack([dna,microtubules,reticulum],dim=1)
        return (stack+torch.randn(stack.shape,device=self.device,generator=generator)*self.config.noise).clamp(0,1)

    def _protein(self,geometry:dict[str,Tensor],compartments:Tensor,generator:torch.Generator|None)->Tensor:
        cell,nucleus=geometry["cell"],geometry["nucleus"]
        inside_cell=_soft(cell-1.0)
        inside_nucleus=_soft(nucleus-1.0)
        cytoplasm=(inside_cell*(1-inside_nucleus)).clamp(0,1)
        masks=[
            inside_nucleus,
            _soft((nucleus-1.0).abs()-0.10),
            cytoplasm,
            _soft((cell-1.0).abs()-0.10)*inside_cell,
            self._vesicles(cytoplasm,generator),
        ]
        selected=torch.stack(masks,dim=1)[torch.arange(len(compartments),device=self.device),compartments]
        noise=torch.randn(selected.shape,device=self.device,generator=generator)*self.config.noise
        return (selected+noise).clamp(0,1)[:,None]

    def _vesicles(self,cytoplasm:Tensor,generator:torch.Generator|None)->Tensor:
        batch=cytoplasm.shape[0]
        spots=torch.zeros_like(cytoplasm)
        for _ in range(self.config.vesicle_count):
            centre=(torch.rand(batch,2,device=self.device,generator=generator)*2-1)*0.55
            radius=torch.full((batch,2),0.07,device=self.device)
            spots=torch.maximum(spots,_soft(_ellipse(self.yy,self.xx,centre,radius,torch.zeros(batch,device=self.device))-1.0,0.03))
        return (spots*cytoplasm).clamp(0,1)

    def batch(self,size:int,generator:torch.Generator|None=None)->dict[str,Tensor]:
        labels=torch.randint(0,self.config.num_proteins,(size,),device=self.device,generator=generator)
        compartments=self.sample_compartments(labels,generator)
        geometry=self._geometry(size,generator)
        return {
            "x0":self._protein(geometry,compartments,generator)*2-1,
            "condition":self._landmarks(geometry,generator)*2-1,
            "labels":labels,
            "compartments":compartments,
        }

    def stream(self,size:int,seed:int=0):
        generator=torch.Generator(device=self.device).manual_seed(seed)
        while True:
            yield self.batch(size,generator)
