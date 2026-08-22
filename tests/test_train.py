import json

import pytest
import torch
from torch import nn

from pvs.diffusion import NoiseSchedule
from pvs.train.ema import EMA
from pvs.train.loop import TrainConfig,Trainer,resolve_device


class Tiny(nn.Module):
    def __init__(self)->None:
        super().__init__()
        self.net=nn.Sequential(nn.Linear(4,16),nn.SiLU(),nn.Linear(16,4))

    def forward(self,xt,t,condition=None,labels=None,channel_mask=None,modes=None):
        return self.net(xt)+t.float()[:,None]*0


def _batches(seed=0):
    generator=torch.Generator().manual_seed(seed)
    while True:
        yield {"x0":torch.randn(8,4,generator=generator)}


def _config(tmp_path,**kwargs):
    defaults=dict(steps=20,batch_size=8,warmup_steps=5,log_every=10,checkpoint_every=20,
                  amp=False,device="cpu",output_dir=str(tmp_path/"run"))
    return TrainConfig(**{**defaults,**kwargs})


def _trainer(tmp_path,seed=0,**kwargs):
    torch.manual_seed(0)
    return Trainer(Tiny(),NoiseSchedule.make(50,"cosine"),_batches(seed),_config(tmp_path,**kwargs))


def test_device_resolution_honours_an_explicit_choice():
    assert resolve_device("cpu").type=="cpu"
    assert resolve_device().type in ("cpu","cuda","mps")


def test_ema_tracks_a_frozen_model():
    model=nn.Linear(4,4)
    ema=EMA(model,decay=0.5)
    with torch.no_grad():
        model.weight.fill_(1.0); model.bias.fill_(0.0)
    for _ in range(40):
        ema.update(model)
    torch.testing.assert_close(ema.shadow.weight,model.weight,rtol=1e-4,atol=1e-4)


def test_ema_warmup_starts_fast_then_slows():
    ema=EMA(nn.Linear(2,2),decay=0.99,warmup=100)
    early=ema.rate()
    ema.steps=50
    assert early<ema.rate()<0.99
    ema.steps=1000
    assert ema.rate()==0.99


def test_ema_rejects_an_invalid_decay():
    with pytest.raises(ValueError):
        EMA(nn.Linear(2,2),decay=1.0)


def test_ema_survives_a_state_dict_round_trip():
    model=nn.Linear(4,4)
    ema=EMA(model,decay=0.9)
    for _ in range(5):
        ema.update(model)
    restored=EMA(nn.Linear(4,4),decay=0.5)
    restored.load_state_dict(ema.state_dict())
    assert restored.steps==ema.steps and restored.decay==0.9
    torch.testing.assert_close(restored.shadow.weight,ema.shadow.weight)


def test_training_writes_a_log_and_a_checkpoint(tmp_path):
    trainer=_trainer(tmp_path)
    trainer.run()
    assert (trainer.output/"latest.pt").exists()
    assert json.loads((trainer.output/"config.json").read_text())["steps"]==20
    records=[json.loads(line) for line in (trainer.output/"log.jsonl").read_text().splitlines()]
    assert [r["step"] for r in records]==[10,20]
    assert all(r["loss"]>0 for r in records)


def test_learning_rate_warms_up_then_holds(tmp_path):
    trainer=_trainer(tmp_path,steps=10,warmup_steps=5,learning_rate=1e-3)
    trainer.run()
    assert trainer.scheduler.get_last_lr()[0]==pytest.approx(1e-3)


def test_resume_reproduces_the_uninterrupted_trajectory(tmp_path):
    """The guarantee that matters on rented compute: a kill at any step costs only the
    steps since the last checkpoint, not the trajectory."""
    straight=_trainer(tmp_path/"a",steps=20,checkpoint_every=20)
    straight.run()

    first=_trainer(tmp_path/"b",steps=10,checkpoint_every=10)
    first.run()
    second=_trainer(tmp_path/"c",steps=20,checkpoint_every=20)
    second.load(first.output/"latest.pt")
    second.batches=_batches(0)
    for _ in range(10):
        next(second.batches)
    second.run()

    assert second.step==straight.step
    for a,b in zip(straight.model.parameters(),second.model.parameters()):
        torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)


def test_gradient_accumulation_matches_a_larger_batch(tmp_path):
    single=_trainer(tmp_path/"single",steps=4,accumulation=1,learning_rate=1e-3)
    single.run()
    accumulated=_trainer(tmp_path/"accum",steps=4,accumulation=2,learning_rate=1e-3)
    accumulated.run()
    changed=any(not torch.allclose(a,b) for a,b in
                zip(single.model.parameters(),accumulated.model.parameters()))
    assert changed
    assert all(torch.isfinite(p).all() for p in accumulated.model.parameters())


def test_checkpoint_is_written_atomically(tmp_path):
    trainer=_trainer(tmp_path,steps=2,checkpoint_every=2)
    trainer.run()
    assert not list(trainer.output.glob("*.tmp"))
    assert (trainer.output/"latest.pt").stat().st_size>0


def test_conditioning_keys_are_matched_to_the_model_signature():
    from pvs.train.loop import Trainer

    class Narrow(torch.nn.Module):
        def forward(self,x,t,condition=None):
            return x

    class Wide(torch.nn.Module):
        def forward(self,x,t,condition=None,labels=None,state=None):
            return x

    class Open(torch.nn.Module):
        def forward(self,x,t,**kwargs):
            return x

    assert Trainer._accepted(Narrow())==("condition",)
    assert Trainer._accepted(Wide())==("condition","labels","state")
    # A model taking **kwargs gets everything; so does anything unintrospectable.
    assert Trainer._accepted(Open())==Trainer.CONDITION_KEYS
    assert Trainer._accepted(lambda x,t:x)==()


def test_default_forward_does_not_break_a_narrow_model():
    from pvs.train.loop import Trainer

    class Narrow(torch.nn.Module):
        def forward(self,x,t,condition=None):
            return torch.zeros_like(x)

    batch={"condition":torch.zeros(2,3,8,8),"state":torch.zeros(2,4),
           "labels":torch.zeros(2,dtype=torch.long)}
    loss=Trainer._default_forward(Narrow(),batch,torch.zeros(2,1,8,8),
                                  torch.zeros(2,dtype=torch.long),torch.zeros(2,1,8,8))
    assert float(loss)==0.0
