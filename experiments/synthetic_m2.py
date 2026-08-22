"""Module M2 rehearsed on data whose information structure is known.

The synthetic task is built so that each compartment depends on a specific landmark, and one
compartment depends on none of them:

    nucleoplasm, nuclear rim   determined by the DNA channel
    cytosol                    needs DNA to exclude the nucleus, and the cell extent
    membrane                   determined by the cell boundary, carried by microtubules and ER
    vesicles                   position is drawn at random; no landmark predicts it

That last row is the point. Every published model fails on vesicular compartments, and
`INTERNAL.md` section 6 attributes it to weak spatial correlation with the landmarks. Here
that is true by construction, so a correct estimator must report approximately zero
information for vesicles across every channel. If it reports a signal there, the estimator is
measuring model fit rather than information, and the M2 matrix cannot be trusted on real data.

The model must be trained *with* channel dropout. Evaluating a model that has never seen a
masked channel measures how badly it handles an out-of-distribution input, which is a
different and much larger quantity.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from pvs.data.synthetic import COMPARTMENTS,NUM_COMPARTMENTS,SyntheticCells,SyntheticConfig
from pvs.diffusion import NoiseSchedule,ddpm_loss
from pvs.eval.information import landmark_information
from pvs.models.dit import DiT

CHANNELS=("dna","microtubules","reticulum")


def train(model,dataset,schedule,steps,batch,lr,dropout,log):
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr)
    lr_schedule=torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=lr,total_steps=steps,pct_start=0.1)
    stream=dataset.stream(batch,seed=5)
    started=time.time()
    for step in range(steps):
        b=next(stream)
        mask=model.channels.sample_mask(b["x0"].shape[0],b["x0"].device)
        wrapped=lambda xt,t:model(xt,t,b["condition"],model.labels.drop(b["labels"]),mask)
        loss=ddpm_loss(wrapped,schedule,b["x0"])
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step(); lr_schedule.step()
        if (step+1)%max(1,steps//8)==0:
            log(f"    step {step+1}/{steps}  loss {loss.item():.4f}  {(step+1)/(time.time()-started):.2f} it/s")
    return model.eval()


def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--size",type=int,default=32)
    parser.add_argument("--proteins",type=int,default=32)
    parser.add_argument("--steps",type=int,default=2500)
    parser.add_argument("--batch",type=int,default=48)
    parser.add_argument("--lr",type=float,default=3e-4)
    parser.add_argument("--timesteps",type=int,default=200)
    parser.add_argument("--dropout",type=float,default=0.3)
    parser.add_argument("--eval-batches",type=int,default=24)
    parser.add_argument("--out",default="outputs/synthetic_m2")
    args=parser.parse_args()

    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    handle=(out/"log.txt").open("a")
    def log(message):
        print(message,flush=True); handle.write(message+"\n"); handle.flush()

    torch.manual_seed(0)
    torch.set_num_threads(2)
    dataset=SyntheticCells(SyntheticConfig(image_size=args.size,num_proteins=args.proteins),seed=0)
    schedule=NoiseSchedule.make(args.timesteps,"cosine")
    model=DiT(image_size=args.size,patch_size=4,target_channels=1,condition_channels=3,
              hidden_size=192,depth=6,heads=6,num_classes=args.proteins,
              channel_dropout=args.dropout,label_dropout=0.15)

    log(f"=== module M2 rehearsal, {args.size}px, channel dropout {args.dropout} ===")
    log(f"  parameters {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    train(model,dataset,schedule,args.steps,args.batch,args.lr,args.dropout,log)

    matrix=landmark_information(model,schedule,dataset.stream(args.batch,seed=999),
                                num_batches=args.eval_batches,num_channels=3,
                                num_compartments=NUM_COMPARTMENTS,
                                channel_names=CHANNELS,compartment_names=COMPARTMENTS)
    log("\n  loss increase from withholding each landmark, by compartment")
    for line in matrix.report().splitlines():
        log("  "+line)
    log(f"\n  cells per compartment: {dict(zip(COMPARTMENTS,[int(c) for c in matrix.counts]))}")

    vesicles=COMPARTMENTS.index("vesicles")
    nucleus=COMPARTMENTS.index("nucleoplasm")
    verdict={
        "vesicles_are_uninformed":bool(~matrix.significant()[:,vesicles].any()),
        "dna_informs_nucleoplasm":bool(matrix.significant()[0,nucleus]),
        "max_vesicle_delta":float(matrix.delta[:,vesicles].abs().max()),
        "dna_nucleoplasm_delta":float(matrix.delta[0,nucleus]),
    }
    log("\n  ground-truth checks")
    for key,value in verdict.items():
        log(f"    {key}: {value}")
    (out/"results.json").write_text(json.dumps({
        "delta":matrix.delta.tolist(),
        "standard_error":matrix.standard_error.tolist(),
        "counts":matrix.counts.tolist(),
        "channels":list(CHANNELS),
        "compartments":list(COMPARTMENTS),
        "verdict":verdict,
    },indent=2))
    log(f"\n  written to {out/'results.json'}")


if __name__=="__main__":
    main()
