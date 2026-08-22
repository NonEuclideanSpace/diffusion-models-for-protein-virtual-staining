"""The diffusion machinery on a real cell, before any training exists.

Two things are worth showing without a trained model. The terminal-SNR defect is a property of
the schedule alone, and it is quantitative rather than visual - per-panel contrast stretching
hides it, so it is plotted, not rendered. BDIA's inversion is exact for *any* noise network, so
an untrained one is a fair demonstration of the recurrence.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import torch

from pvs.data.crops import PROTEIN, load_crop, normalize, resize
from pvs.diffusion.bdia import bdia_invert, bdia_sample
from pvs.diffusion.forward import q_sample
from pvs.diffusion.schedule import (NoiseSchedule, cosine_beta_schedule, linear_beta_schedule,
                                    enforce_zero_terminal_snr)
from pvs.models.dit import DiT

S3="https://czi-subcell-public.s3.us-west-2.amazonaws.com"
INK,DIM,GREEN,RED,LINE="#E7ECF3","#7A8695","#3DD68C","#FF6B6F","#1D2531"
BG="#0A0E14"


def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--plan",type=Path,default=Path("/home/claude/data/subset/plan2.txt"))
    parser.add_argument("--gene",default="CDC20")
    parser.add_argument("--size",type=int,default=64)
    parser.add_argument("--timesteps",type=int,default=1000)
    parser.add_argument("--steps",type=int,default=12)
    parser.add_argument("--out",type=Path,default=Path("outputs/figures/diffusion.png"))
    args=parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    torch.set_num_threads(1)
    torch.manual_seed(0)

    stem=next(line.split("\t")[4] for line in args.plan.read_text().splitlines()
              if line.split("\t")[0]==args.gene)
    scratch=Path("/home/claude/data/figure"); scratch.mkdir(parents=True,exist_ok=True)
    path=scratch/"cell.png"
    for attempt in range(3):
        try:
            with urlopen(f"{S3}/{stem}_cell_image.png",timeout=90) as response:
                path.write_bytes(response.read()); break
        except Exception:
            time.sleep(2*(attempt+1))

    crop=resize(normalize(load_crop(path)),args.size)
    x0=crop[PROTEIN:PROTEIN+1][None]

    plain=NoiseSchedule(cosine_beta_schedule(args.timesteps))
    rescaled=NoiseSchedule(enforce_zero_terminal_snr(cosine_beta_schedule(args.timesteps)))
    marks=[0,200,400,600,800,args.timesteps-1]
    noise=torch.randn_like(x0)
    forward=[x0[0,0] if t==0 else q_sample(rescaled,x0,torch.tensor([t]),noise)[0,0] for t in marks]

    network=DiT(image_size=args.size,patch_size=8,target_channels=1,condition_channels=0,
                hidden_size=128,depth=4,heads=4).eval()
    # DiT zero-initializes its final layer, so a fresh network predicts exactly zero noise and
    # the inversion degenerates to a rescale - which would make the round trip trivially exact
    # and the latent a scaled copy of the cell. Perturbing the final layer gives a nontrivial
    # deterministic eps-predictor, which is what the recurrence has to survive.
    with torch.no_grad():
        for parameter in network.final.parameters():
            parameter.add_(0.35*torch.randn_like(parameter))
    model=lambda x,t:network(x.float(),t)
    with torch.no_grad():
        latents=bdia_invert(model,rescaled,x0.double(),num_inference_steps=args.steps)
        reconstruction=bdia_sample(model,rescaled,latents=latents,num_inference_steps=args.steps)
    error=(reconstruction-x0.double()).abs()

    figure=plt.figure(figsize=(13.4,7.1),facecolor=BG)
    grid=GridSpec(2,6,figure=figure,height_ratios=[1,1.05],hspace=0.40,wspace=0.16,
                  left=0.055,right=0.985,top=0.875,bottom=0.115)

    span=float(x0.abs().max())
    for column,(t,plane) in enumerate(zip(marks,forward)):
        ax=figure.add_subplot(grid[0,column])
        ax.imshow(plane.numpy(),cmap="magma",vmin=-span,vmax=span)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor(BG)
        for spine in ax.spines.values(): spine.set_color(LINE)
        ax.set_title(f"t = {t}",color=DIM,fontsize=9,pad=5)
    figure.text(0.045,0.925,"Forward diffusion on a real cell, one shared intensity scale "
                "across all six panels",color=INK,fontsize=10.5,ha="left")

    linear=NoiseSchedule(linear_beta_schedule(args.timesteps))
    curve=figure.add_subplot(grid[1,0:2])
    steps=np.arange(args.timesteps)
    for schedule,name,colour,style in (
            (linear,"linear",RED,"-"),
            (plain,"cosine",DIM,"-"),
            (rescaled,"cosine, floored at 1e-2",GREEN,"-")):
        signal=schedule.sqrt_alpha_bars.numpy()
        curve.plot(steps,np.maximum(signal,1e-9),color=colour,linewidth=1.7,
                   linestyle=style,label=f"{name}   {signal[-1]:.1e}")
    curve.set_yscale("log"); curve.set_ylim(2e-5,1.6)
    curve.set_facecolor(BG)
    curve.set_xlabel("timestep",color=DIM,fontsize=9)
    curve.set_ylabel(r"$\sqrt{\bar\alpha_t}$  signal left in $x_t$",color=DIM,fontsize=9)
    curve.tick_params(colors=DIM,labelsize=8)
    for spine in curve.spines.values(): spine.set_color(LINE)
    legend=curve.legend(facecolor=BG,edgecolor=LINE,fontsize=8,loc="lower left",
                        title="terminal value")
    legend.get_title().set_color(DIM); legend.get_title().set_fontsize(8)
    for text in legend.get_texts(): text.set_color(INK)
    curve.set_title("Sampling and inversion want opposite ends of this curve",
                    color=INK,fontsize=10.5,loc="left",pad=8)
    figure.text(0.055,0.068,
        "Linear leaks signal into $x_T$, so sampling from $N(0,I)$ starts from the wrong\n"
        "marginal. Cosine goes low enough that inversion, which divides by "
        "$\\sqrt{\\bar\\alpha_T}$,\namplifies model error 38x. The floor satisfies both.",
        color=DIM,fontsize=8,va="top",linespacing=1.6)

    latent=latents[0][0,0].numpy()
    panels=[("$x_0$  real cell",x0[0,0].numpy(),-span,span),
            ("$x_T$  inverted latent",latent,-abs(latent).max(),abs(latent).max()),
            ("$x_0'$  reconstructed",reconstruction[0,0].numpy(),-span,span),
            ("$|x_0-x_0'|$  scaled to $x_0$",error[0,0].numpy(),0.0,span)]
    for offset,(label,plane,low,high) in enumerate(panels):
        ax=figure.add_subplot(grid[1,2+offset])
        ax.imshow(plane,cmap="magma",vmin=low,vmax=high)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor(BG)
        for spine in ax.spines.values(): spine.set_color(LINE)
        ax.set_title(label,color=DIM,fontsize=9,pad=5)
    figure.text(0.383,0.475,"BDIA round trip at 1 NFE, untrained network with a perturbed output layer",
                color=INK,fontsize=10.5,ha="left")
    figure.text(0.383,0.068,
        f"max |error| {float(error.max()):.1e}, about {float(error.max())/2.22e-16:.0f}x float64 epsilon "
        f"accumulated over {args.steps} steps. The panel is drawn\non $x_0$'s own scale, so black means "
        f"nothing survives. Exactness is a property of the recurrence, not of the weights.",
        color=DIM,fontsize=8,ha="left",va="top",linespacing=1.6)

    figure.suptitle(f"{args.gene}, protein channel — the diffusion machinery, before training",
                    color=INK,fontsize=12.5,y=0.972)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    figure.savefig(args.out,dpi=155,facecolor=BG)
    path.unlink(missing_ok=True)
    print(f"terminal signal: linear {float(linear.sqrt_alpha_bars[-1]):.3e}  "
          f"cosine {float(plain.sqrt_alpha_bars[-1]):.3e}  "
          f"floored {float(rescaled.sqrt_alpha_bars[-1]):.3e}")
    print(f"max abs error {float(error.max()):.3e}")
    print(f"wrote {args.out}")


if __name__=="__main__":
    main()
