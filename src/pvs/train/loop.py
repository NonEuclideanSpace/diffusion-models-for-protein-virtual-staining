"""Training loop with resumable checkpoints.

Written for rented compute: any run must survive being killed at an arbitrary step and
resume to the same trajectory. That means the optimizer, the scheduler, the EMA, the step
counter and the RNG state all travel together in one checkpoint, and the checkpoint is
written atomically so a kill during the write cannot corrupt it.
"""
from __future__ import annotations

import inspect
import json
import time
from dataclasses import asdict,dataclass,field
from pathlib import Path
from typing import Callable,Iterator

import torch
from torch import Tensor,nn

from pvs.diffusion.parameterization import Target,training_target
from pvs.diffusion.forward import q_sample
from pvs.diffusion.schedule import NoiseSchedule
from pvs.train.ema import EMA


def resolve_device(preference:str="auto")->torch.device:
    if preference!="auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class TrainConfig:
    steps:int=100_000
    batch_size:int=64
    learning_rate:float=1e-4
    weight_decay:float=0.0
    warmup_steps:int=1_000
    grad_clip:float|None=1.0
    accumulation:int=1
    ema_decay:float=0.9999
    target:Target="eps"
    device:str="auto"
    amp:bool=True
    log_every:int=100
    checkpoint_every:int=5_000
    seed:int=0
    output_dir:str="outputs/run"
    extra:dict=field(default_factory=dict)


class Trainer:
    def __init__(
        self,
        model:nn.Module,
        schedule:NoiseSchedule,
        batches:Iterator[dict],
        config:TrainConfig,
        loss_inputs:Callable[[nn.Module,dict,Tensor,Tensor,Tensor],Tensor]|None=None,
    )->None:
        self.config=config
        self.device=resolve_device(config.device)
        self.model=model.to(self.device)
        self.schedule=schedule.to(self.device)
        self.batches=batches
        self.loss_inputs=loss_inputs or self._default_forward
        self.ema=EMA(self.model,config.ema_decay,config.warmup_steps)
        self.optimizer=torch.optim.AdamW(self.model.parameters(),lr=config.learning_rate,
                                         weight_decay=config.weight_decay)
        self.scheduler=torch.optim.lr_scheduler.LambdaLR(self.optimizer,self._warmup)
        self.scaler=torch.amp.GradScaler(enabled=config.amp and self.device.type=="cuda")
        self.step=0
        self.output=Path(config.output_dir)
        self.output.mkdir(parents=True,exist_ok=True)
        (self.output/"config.json").write_text(json.dumps(asdict(config),indent=2))
        torch.manual_seed(config.seed)

    def _warmup(self,step:int)->float:
        return min(1.0,(step+1)/max(1,self.config.warmup_steps))

    CONDITION_KEYS=("condition","labels","channel_mask","modes","state","state_mask")

    @staticmethod
    def _accepted(model)->tuple[str,...]:
        """Which conditioning inputs this model's forward actually takes.

        Passing the full set positionally means every new conditioning key breaks every model
        with a shorter signature, which is exactly what adding `state` did. Asking the model
        instead keeps the two independent.
        """
        target=getattr(model,"forward",model)
        try:
            parameters=inspect.signature(target).parameters
        except (TypeError,ValueError):
            return Trainer.CONDITION_KEYS
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
            return Trainer.CONDITION_KEYS
        return tuple(k for k in Trainer.CONDITION_KEYS if k in parameters)

    @staticmethod
    def _default_forward(model,batch,xt,t,target)->Tensor:
        keys=Trainer._accepted(model)
        prediction=model(xt,t,**{k:batch.get(k) for k in keys})
        return torch.nn.functional.mse_loss(prediction,target)

    def loss(self,batch:dict)->Tensor:
        x0=batch["x0"]
        t=torch.randint(0,len(self.schedule),(x0.shape[0],),device=x0.device)
        noise=torch.randn_like(x0)
        xt=q_sample(self.schedule,x0,t,noise)
        target=training_target(self.schedule,x0,noise,t,self.config.target)
        return self.loss_inputs(self.model,batch,xt,t,target)

    def to_device(self,batch:dict)->dict:
        return {k:(v.to(self.device) if isinstance(v,Tensor) else v) for k,v in batch.items()}

    def run(self)->None:
        self.model.train()
        started=time.time()
        running=0.0
        while self.step<self.config.steps:
            self.optimizer.zero_grad(set_to_none=True)
            for _ in range(self.config.accumulation):
                batch=self.to_device(next(self.batches))
                with torch.autocast(self.device.type,enabled=self.scaler.is_enabled()):
                    loss=self.loss(batch)/self.config.accumulation
                self.scaler.scale(loss).backward()
                running+=loss.item()
            if self.config.grad_clip:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.config.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.ema.update(self.model)
            self.step+=1

            if self.step%self.config.log_every==0:
                mean=running/self.config.log_every
                rate=self.step/(time.time()-started)
                self.log({"step":self.step,"loss":round(mean,5),
                          "lr":round(self.scheduler.get_last_lr()[0],8),
                          "steps_per_second":round(rate,2)})
                running=0.0
            if self.step%self.config.checkpoint_every==0:
                self.save()
        self.save()

    def log(self,record:dict)->None:
        line=json.dumps(record)
        with (self.output/"log.jsonl").open("a") as f:
            f.write(line+"\n")
        print(line,flush=True)

    def save(self,name:str="latest.pt")->Path:
        path=self.output/name
        temporary=path.with_suffix(".tmp")
        torch.save({
            "step":self.step,
            "model":self.model.state_dict(),
            "ema":self.ema.state_dict(),
            "optimizer":self.optimizer.state_dict(),
            "scheduler":self.scheduler.state_dict(),
            "scaler":self.scaler.state_dict(),
            "rng":torch.get_rng_state(),
            "config":asdict(self.config),
        },temporary)
        temporary.replace(path)
        return path

    def load(self,path:str|Path)->None:
        state=torch.load(path,map_location=self.device,weights_only=False)
        self.step=state["step"]
        self.model.load_state_dict(state["model"])
        self.ema.load_state_dict(state["ema"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        torch.set_rng_state(state["rng"].cpu().to(torch.uint8))
