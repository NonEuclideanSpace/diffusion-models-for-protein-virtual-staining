"""Re-run M1's power calculation, which Amendment A1.3 voided.

The original used reference distributions resampled from the 40 annotation-selected genes of
probe A, and assumed n_real=75 from pooling across cell lines. Both inputs are now wrong: the
annotation does not select for the quantity M1 measures, and the study runs in U2OS alone at
50-58 cells. This redoes it on the U2OS screen's own reference distributions at n_real=50.

The question is not "can a divergence detect a difference" but "how many proteins does the gate
need". The gate is a monotone trend in D_bar across guidance weights plus a permutation test, so
the simulation runs the whole sweep rather than a single comparison.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch import Tensor

import torch

from pvs.eval.calibration import DIVERGENCES, collapse_toward_mode, divergence, trend_statistic
from pvs.eval.localization import to_groups


def multinomial(probabilities:Tensor,n:int,generator:torch.Generator)->Tensor:
    """torch.distributions.Multinomial ignores an explicit generator, so draw and count."""
    draws=torch.multinomial(probabilities,n,replacement=True,generator=generator)
    return torch.bincount(draws,minlength=len(probabilities)).float()


def sweep(reference:Tensor,weights:list[float],n_real:int,n_generated:int,
          name:str,generator:torch.Generator)->torch.Tensor:
    """One protein's D_bar across the guidance sweep, under a model that collapses linearly."""
    measure=divergence(name)
    empirical=multinomial(reference,n_real,generator)
    values=[]
    for weight in weights:
        collapsed=collapse_toward_mode(reference,weight)
        generated=multinomial(collapsed,n_generated,generator)
        values.append(measure(generated[None],empirical[None])[0])
    return torch.stack(values)


def permutation_p(trends:torch.Tensor,draws:int,generator:torch.Generator)->float:
    """Null: the per-protein sweeps carry no order information."""
    observed=float(trends.mean())
    hits=0
    for _ in range(draws):
        signs=torch.randint(0,2,(len(trends),),generator=generator).float()*2-1
        hits+=float((trends*signs).mean())>=observed
    return (hits+1)/(draws+1)


def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--results",type=Path,
                        default=Path("/mnt/user-data/uploads/dmfpvs-v0/outputs/u2os_partial.json"))
    parser.add_argument("--weights",type=float,nargs="+",default=[0.0,0.08,0.16,0.24])
    parser.add_argument("--n-real",type=int,nargs="+",default=[50,58,75])
    parser.add_argument("--n-generated",type=int,default=100)
    parser.add_argument("--proteins",type=int,nargs="+",default=[50,100,150,300])
    parser.add_argument("--divergence",nargs="+",default=["total_variation","jensen_shannon"])
    parser.add_argument("--trials",type=int,default=300)
    parser.add_argument("--permutations",type=int,default=400)
    parser.add_argument("--alpha",type=float,default=0.01)
    parser.add_argument("--seed",type=int,default=0)
    args=parser.parse_args()

    # A permutation test can never report p below 1/(draws+1). With alpha at 0.01 and 80 draws
    # the floor is 0.0123 and detection is 0% whatever the effect - a trap worth failing loudly.
    floor=1.0/(args.permutations+1)
    if floor>=args.alpha:
        raise SystemExit(f"--permutations {args.permutations} gives a p-value floor of {floor:.4f}, "
                         f"which can never clear alpha {args.alpha}. Use at least "
                         f"{int(1/args.alpha)+1}.")

    records=json.loads(args.results.read_text())
    fine=torch.tensor([r["mean_probability"] for r in records.values()])
    references=to_groups(fine)
    references=references/references.sum(-1,keepdim=True).clamp_min(1e-12)
    dominant=references.max(-1).values
    support=(1.0/(references**2).sum(-1))
    print(f"=== {len(references)} U2OS reference distributions, "
          f"{references.shape[1]} groups ===")
    print(f"  dominant component: median {dominant.median():.3f}  "
          f"p10 {dominant.quantile(0.1):.3f}  p90 {dominant.quantile(0.9):.3f}")
    print(f"  effective support:  median {support.median():.2f} groups")
    print(f"  the voided calculation used 0.413 dominant and 4.55 effective support, from "
          f"annotation-selected genes\n")

    generator=torch.Generator().manual_seed(args.seed)
    print(f"  collapse weights {args.weights}, {args.n_generated} generated per protein, "
          f"alpha {args.alpha}")
    print(f"\n{'divergence':<18}{'n_real':>8}{'proteins':>10}{'detection':>12}   verdict")
    for name in args.divergence:
        for n_real in args.n_real:
            for proteins in args.proteins:
                detected=0
                for _ in range(args.trials):
                    picks=torch.randint(len(references),(proteins,),generator=generator)
                    trends=torch.stack([
                        torch.tensor(trend_statistic(
                            sweep(references[p],args.weights,n_real,args.n_generated,
                                  name,generator)))
                        for p in picks])
                    monotone=float(trends.mean())>0
                    detected+=monotone and permutation_p(trends,args.permutations,
                                                         generator)<args.alpha
                rate=detected/args.trials
                verdict=("sufficient" if rate>=0.95 else
                         "marginal" if rate>=0.8 else "underpowered")
                print(f"{name:<18}{n_real:>8}{proteins:>10}{rate:>11.1%}   {verdict}",flush=True)


if __name__=="__main__":
    main()
