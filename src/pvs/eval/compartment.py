"""A compartment classifier standing in for SubCell.

On real data phi is a learned classifier trained on labels that were broadcast across a whole
field of view, so its per-cell predictions carry error of unknown sign. Using a ground-truth
oracle on synthetic data would hide exactly that, and the point of the synthetic task is to
find out how classifier error propagates into the calibration measurement. So phi is learned
here too, from pixels alone.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor,nn

from pvs.models.blocks import normalization


class CompartmentClassifier(nn.Module):
    def __init__(self,in_channels:int=4,num_classes:int=5,width:int=32,depth:int=3)->None:
        super().__init__()
        layers=[]
        channels=in_channels
        for level in range(depth):
            out=width*2**level
            layers+=[nn.Conv2d(channels,out,3,padding=1),normalization(out),nn.SiLU(),
                     nn.Conv2d(out,out,3,padding=1),normalization(out),nn.SiLU(),
                     nn.AvgPool2d(2)]
            channels=out
        self.features=nn.Sequential(*layers)
        self.head=nn.Linear(channels,num_classes)

    def forward(self,protein:Tensor,condition:Tensor)->Tensor:
        h=self.features(torch.cat([protein,condition],dim=1))
        return self.head(h.mean(dim=(2,3)))

    @torch.no_grad()
    def predict(self,protein:Tensor,condition:Tensor)->Tensor:
        return self(protein,condition).argmax(-1)

    @torch.no_grad()
    def fractions(self,protein:Tensor,condition:Tensor,num_classes:int|None=None)->Tensor:
        classes=num_classes or self.head.out_features
        return torch.bincount(self.predict(protein,condition),minlength=classes).float()


def train_classifier(
    dataset,
    steps:int=1500,
    batch_size:int=64,
    learning_rate:float=2e-3,
    device:str|torch.device="cpu",
    log:callable=print,
)->CompartmentClassifier:
    from pvs.data.synthetic import NUM_COMPARTMENTS
    model=CompartmentClassifier(num_classes=NUM_COMPARTMENTS).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=steps)
    stream=dataset.stream(batch_size,seed=17)
    for step in range(steps):
        batch=next(stream)
        loss=F.cross_entropy(model(batch["x0"],batch["condition"]),batch["compartments"])
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); schedule.step()
        if (step+1)%(steps//5)==0:
            log(f"    classifier step {step+1}/{steps} loss {loss.item():.4f}")
    return model.eval()


@torch.no_grad()
def classifier_accuracy(model:CompartmentClassifier,dataset,samples:int=2048,seed:int=99)->float:
    generator=torch.Generator(device=dataset.device).manual_seed(seed)
    batch=dataset.batch(samples,generator)
    device=model.head.weight.device
    predicted=model.predict(batch["x0"].to(device),batch["condition"].to(device)).cpu()
    return float((predicted==batch["compartments"].cpu()).float().mean())


@torch.no_grad()
def confusion(model:CompartmentClassifier,dataset,samples:int=2048,seed:int=99)->Tensor:
    from pvs.data.synthetic import NUM_COMPARTMENTS
    generator=torch.Generator(device=dataset.device).manual_seed(seed)
    batch=dataset.batch(samples,generator)
    device=model.head.weight.device
    predicted=model.predict(batch["x0"].to(device),batch["condition"].to(device)).cpu()
    matrix=torch.zeros(NUM_COMPARTMENTS,NUM_COMPARTMENTS)
    for truth,guess in zip(batch["compartments"].cpu(),predicted):
        matrix[truth,guess]+=1
    return matrix/matrix.sum(-1,keepdim=True).clamp_min(1)
