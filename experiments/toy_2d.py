"""Train a small noise predictor on a 2-D mixture and validate the Phase 0 loop."""
from __future__ import annotations

import argparse
import math

import torch
from torch import Tensor

from pvs.diffusion import NoiseSchedule,ddim_invert,ddim_sample,ddpm_loss,ddpm_sample
from pvs.models.toy import ToyMLP

MODES=8
RADIUS=2.0


def mode_centers()->Tensor:
    angles=torch.arange(MODES)*(2*math.pi/MODES)
    return torch.stack([angles.cos(),angles.sin()],dim=1)*RADIUS


def eight_gaussians(n:int,std:float=0.12)->Tensor:
    centers=mode_centers()
    return centers[torch.randint(0,MODES,(n,))]+torch.randn(n,2)*std


def mode_fractions(x:Tensor)->Tensor:
    nearest=torch.cdist(x,mode_centers()).argmin(dim=1)
    return torch.bincount(nearest,minlength=MODES).float()/len(x)


def categorical_kl(p:Tensor,q:Tensor,eps:float=1e-8)->float:
    p,q=p+eps,q+eps
    return float((p*(p/q).log()).sum())


def train(model:torch.nn.Module,schedule:NoiseSchedule,steps:int,batch:int,lr:float)->list[float]:
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    history=[]
    for step in range(steps):
        loss=ddpm_loss(model,schedule,eight_gaussians(batch))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history.append(loss.item())
        if (step+1)%(steps//8)==0:
            window=history[-steps//8:]
            print(f"  step {step+1:>6}  loss {sum(window)/len(window):.4f}")
    return history


def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--steps",type=int,default=4000)
    parser.add_argument("--batch",type=int,default=512)
    parser.add_argument("--lr",type=float,default=2e-3)
    parser.add_argument("--timesteps",type=int,default=200)
    parser.add_argument("--seed",type=int,default=0)
    args=parser.parse_args()

    torch.manual_seed(args.seed)
    schedule=NoiseSchedule.make(args.timesteps,"cosine")
    model=ToyMLP()

    print(f"training {sum(p.numel() for p in model.parameters()):,} parameters")
    train(model,schedule,args.steps,args.batch,args.lr)
    model.eval()

    real=eight_gaussians(8192)
    with torch.no_grad():
        ancestral=ddpm_sample(model,schedule,(8192,2))
        deterministic=ddim_sample(model,schedule,(8192,2),num_inference_steps=100)

    print("\nmode coverage")
    print(f"  real          {mode_fractions(real).numpy().round(3)}")
    print(f"  ddpm          {mode_fractions(ancestral).numpy().round(3)}  KL={categorical_kl(mode_fractions(ancestral),mode_fractions(real)):.4f}")
    print(f"  ddim          {mode_fractions(deterministic).numpy().round(3)}  KL={categorical_kl(mode_fractions(deterministic),mode_fractions(real)):.4f}")

    print("\nddim inversion round-trip on a trained model")
    x0=eight_gaussians(512)
    for n in (25,50,100,200):
        with torch.no_grad():
            latent=ddim_invert(model,schedule,x0,num_inference_steps=n)
            recon=ddim_sample(model,schedule,latent=latent,num_inference_steps=n,eta=0.0)
        print(f"  steps {n:>4}  rel err {((recon-x0).norm()/x0.norm()).item():.4f}"
              f"  latent std {latent.std().item():.3f}")


if __name__=="__main__":
    main()
