"""End-to-end rehearsal of module M1 on data whose conditional distribution is known.

Runs the whole chain: train a compartment classifier, train a conditional diffusion model,
sweep classifier-free guidance, and measure whether the calibration statistic detects the
mode collapse that guidance is claimed to cause. Everything the real study will do, on a task
where the answer is known by construction, so a null result here is a statement about the
measurement rather than about biology.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from pvs.data.synthetic import COMPARTMENTS,NUM_COMPARTMENTS,SyntheticCells,SyntheticConfig
from pvs.diffusion import NoiseSchedule,ddim_sample,ddpm_loss
from pvs.diffusion.guidance import classifier_free_guidance
from pvs.eval.calibration import DIVERGENCES,detection_threshold,divergence,trend_statistic
from pvs.eval.compartment import classifier_accuracy,confusion,train_classifier
from pvs.models.dit import DiT
from pvs.models.unet import ConditionalUNet


def build(kind:str,size:int,proteins:int):
    if kind=="dit":
        return DiT(image_size=size,patch_size=4,target_channels=1,condition_channels=3,
                   hidden_size=192,depth=6,heads=6,num_classes=proteins,label_dropout=0.15)
    return ConditionalUNet(target_channels=1,condition_channels=3,base_width=48,
                           width_multipliers=(1,2,2),blocks_per_level=1,
                           attention_resolutions=(8,),num_classes=proteins,
                           label_dropout=0.15,image_size=size)


def to_device(batch:dict,device)->dict:
    return {k:(v.to(device) if torch.is_tensor(v) else v) for k,v in batch.items()}


def train(model,dataset,schedule,steps,batch,lr,log,device=None):
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=0.0)
    lr_schedule=torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=lr,total_steps=steps,pct_start=0.1)
    stream=dataset.stream(batch,seed=5)
    started=time.time()
    for step in range(steps):
        b=next(stream)
        if device is not None:
            b=to_device(b,device)
        labels=model.labels.drop(b["labels"])
        wrapped=lambda xt,t:model(xt,t,b["condition"],labels)
        loss=ddpm_loss(wrapped,schedule,b["x0"])
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step(); lr_schedule.step()
        if (step+1)%max(1,steps//10)==0:
            log(f"    step {step+1}/{steps}  loss {loss.item():.4f}  {(step+1)/(time.time()-started):.2f} it/s")
    return model.eval()


@torch.no_grad()
def held_out_loss(model,dataset,schedule,batches=8,batch=64,seed=1234,device=None):
    generator=torch.Generator().manual_seed(seed)
    total=0.0
    for _ in range(batches):
        b=dataset.batch(batch,generator)
        if device is not None:
            b=to_device(b,device)
        wrapped=lambda xt,t:model(xt,t,b["condition"],b["labels"])
        total+=float(ddpm_loss(wrapped,schedule,b["x0"]))
    return total/batches


@torch.no_grad()
def sample_fractions(model,classifier,dataset,schedule,protein,weight,samples,steps,seed,
                     device=None):
    # The dataset generates on CPU, so it keeps a CPU generator; sampling draws noise on the
    # model's device and needs a generator that lives there.
    generator=torch.Generator().manual_seed(seed)
    reference=dataset.batch(samples,generator)
    if device is not None:
        reference=to_device(reference,device)
    condition=reference["condition"]
    labels=torch.full((samples,),protein,dtype=torch.long,device=condition.device)
    sampling=torch.Generator(device=condition.device).manual_seed(seed)
    guided=classifier_free_guidance(model,weight,drop="labels",batched=False)
    predictor=lambda xt,t:guided(xt,t,condition,labels)
    images=ddim_sample(predictor,schedule,(samples,1,condition.shape[-1],condition.shape[-1]),
                       num_inference_steps=steps,eta=0.0,generator=sampling)
    return classifier.fractions(images.clamp(-1,1),condition,NUM_COMPARTMENTS)


def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--size",type=int,default=32)
    parser.add_argument("--proteins",type=int,default=32)
    parser.add_argument("--steps",type=int,default=4000)
    parser.add_argument("--classifier-steps",type=int,default=1500)
    parser.add_argument("--batch",type=int,default=48)
    parser.add_argument("--lr",type=float,default=3e-4)
    parser.add_argument("--timesteps",type=int,default=200)
    parser.add_argument("--samples",type=int,default=64)
    parser.add_argument("--inference-steps",type=int,default=40)
    parser.add_argument("--weights",type=float,nargs="+",default=[1.0,2.0,3.0,5.0])
    parser.add_argument("--probes",type=int,default=8)
    parser.add_argument("--models",nargs="+",default=["dit","unet"])
    parser.add_argument("--out",default="outputs/synthetic_m1")
    parser.add_argument("--device",default="auto",
                        help="auto picks cuda, then mps, then cpu")
    args=parser.parse_args()

    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    logfile=(out/"log.txt").open("a")
    def log(message):
        print(message,flush=True); logfile.write(message+"\n"); logfile.flush()

    torch.manual_seed(0)
    torch.set_num_threads(2)
    if args.device=="auto":
        args.device=("cuda" if torch.cuda.is_available()
                     else "mps" if torch.backends.mps.is_available() else "cpu")
    device=torch.device(args.device)
    dataset=SyntheticCells(SyntheticConfig(image_size=args.size,num_proteins=args.proteins),seed=0)
    schedule=NoiseSchedule.make(args.timesteps,"cosine").to(device)
    log(f"=== synthetic M1 rehearsal, {args.size}px, {args.proteins} proteins, {device} ===")

    log("  training compartment classifier")
    classifier=train_classifier(dataset,steps=args.classifier_steps,log=log)
    classifier=classifier.to(device)
    accuracy=classifier_accuracy(classifier,dataset)
    log(f"  classifier accuracy {accuracy:.3f}")
    log("  confusion (rows = truth):")
    for name,row in zip(COMPARTMENTS,confusion(classifier,dataset)):
        log(f"    {name:<12} "+" ".join(f"{v:.2f}" for v in row.tolist()))

    entropy=dataset.entropy()
    probes=entropy.argsort(descending=True)[:args.probes].tolist()
    log(f"  probe proteins (highest entropy): {probes}")
    log(f"  their entropies: {[round(float(entropy[p]),3) for p in probes]}")

    results={"accuracy":accuracy,"probes":probes,
             "entropy":[float(entropy[p]) for p in probes],"models":{}}

    for kind in args.models:
        log(f"\n  training {kind}")
        model=build(kind,args.size,args.proteins).to(device)
        log(f"    parameters {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
        train(model,dataset,schedule,args.steps,args.batch,args.lr,log,device)
        loss=held_out_loss(model,dataset,schedule,device=device)
        log(f"    held-out loss {loss:.4f}")

        sweep={}
        for weight in args.weights:
            per_protein=[]
            for protein in probes:
                counts=sample_fractions(model,classifier,dataset,schedule,protein,weight,
                                        args.samples,args.inference_steps,seed=1000+protein,
                                        device=device)
                counts=counts.cpu()
                truth=dataset.distributions[protein]
                per_protein.append({
                    "protein":protein,
                    "generated":(counts/counts.sum()).tolist(),
                    "truth":truth.tolist(),
                    **{name:float(fn(counts,truth*args.samples)) for name,fn in DIVERGENCES.items()},
                })
            sweep[str(weight)]=per_protein
            means={name:sum(p[name] for p in per_protein)/len(per_protein) for name in DIVERGENCES}
            log(f"    w={weight:<4} "+"  ".join(f"{k}={v:.4f}" for k,v in means.items()))
        results["models"][kind]={"held_out_loss":loss,"sweep":sweep}

        for name in DIVERGENCES:
            curve=torch.tensor([sum(p[name] for p in sweep[str(w)])/len(probes) for w in args.weights])
            results["models"][kind].setdefault("trend",{})[name]=trend_statistic(curve)
        log(f"    trend statistics: "+
            "  ".join(f"{k}={v:+.2f}" for k,v in results['models'][kind]['trend'].items()))

    (out/"results.json").write_text(json.dumps(results,indent=2))
    log(f"\nwritten to {out/'results.json'}")


if __name__=="__main__":
    main()
