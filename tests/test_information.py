import torch
import torch.nn.functional as F
from torch import nn

from pvs.diffusion import NoiseSchedule
from pvs.diffusion.forward import q_sample
from pvs.eval.information import landmark_information

CHANNELS=("a","b","c")
COMPARTMENTS=("x","y")


class KnownDependence(nn.Module):
    """A model whose dependence on each channel is fixed by construction.

    Predicts the noise with a timestep-dependent error, as any real model does, and degrades
    by a known extra amount when the channel a compartment depends on is masked. Compartment
    x depends on channel 0 only; compartment y depends on nothing, standing in for the
    vesicular case where the landmarks carry no information about position.

    The timestep-dependent wobble is what makes this a fair test of the paired estimator: it
    is common to both terms and so cancels under pairing, exactly like the real nuisance
    variation the estimator has to see through.
    """

    def __init__(self,schedule)->None:
        super().__init__()
        self.schedule=schedule
        self.unused=nn.Parameter(torch.zeros(1))

    def forward(self,xt,t,condition=None,labels=None,channel_mask=None,modes=None):
        from pvs.diffusion.schedule import extract
        signal=extract(self.schedule.sqrt_alpha_bars,t,xt.ndim)
        noise_scale=extract(self.schedule.sqrt_one_minus_alpha_bars,t,xt.ndim)
        exact=(xt-signal*condition[:,:1])/noise_scale.clamp_min(1e-4)
        wobble=0.6*torch.sin(t.float()/7.0)[:,None,None,None]
        if channel_mask is None:
            penalty=torch.zeros(xt.shape[0],device=xt.device)
        else:
            penalty=(1-channel_mask[:,0])*labels.float()
        return exact*(1+wobble)+penalty[:,None,None,None]*0.5+self.unused


def _batches(schedule,size=64,seed=0):
    generator=torch.Generator().manual_seed(seed)
    while True:
        compartments=torch.randint(0,2,(size,),generator=generator)
        condition=torch.randn(size,3,8,8,generator=generator)
        yield {"x0":condition[:,:1].clone(),"condition":condition,
               "labels":(compartments==0).long(),"compartments":compartments}


def test_recovers_a_known_dependence_structure():
    schedule=NoiseSchedule.make(50,"cosine")
    model=KnownDependence(schedule)
    matrix=landmark_information(model,schedule,_batches(schedule),num_batches=6,
                                num_channels=3,num_compartments=2,
                                channel_names=CHANNELS,compartment_names=COMPARTMENTS)
    assert matrix.delta.shape==(3,2)
    assert matrix.delta[0,0]>0.02
    assert matrix.delta[0,1].abs()<1e-6
    assert matrix.delta[1].abs().max()<1e-6
    assert matrix.delta[2].abs().max()<1e-6


def test_significance_marks_only_the_real_dependence():
    schedule=NoiseSchedule.make(50,"cosine")
    matrix=landmark_information(KnownDependence(schedule),schedule,_batches(schedule),
                                num_batches=6,num_channels=3,num_compartments=2,
                                channel_names=CHANNELS,compartment_names=COMPARTMENTS)
    significant=matrix.significant()
    assert bool(significant[0,0])
    assert not bool(significant[0,1])
    assert not significant[1:].any()


def test_report_renders_every_cell():
    schedule=NoiseSchedule.make(50,"cosine")
    text=landmark_information(KnownDependence(schedule),schedule,_batches(schedule),
                              num_batches=2,num_channels=3,num_compartments=2,
                              channel_names=CHANNELS,compartment_names=COMPARTMENTS).report()
    for name in CHANNELS+COMPARTMENTS:
        assert name[:11] in text


def test_pairing_the_randomness_cuts_the_variance():
    """The estimator is a difference of two nearly equal expectations. Sharing the timestep
    and the noise between them is what makes it measurable at all."""
    schedule=NoiseSchedule.make(50,"cosine")
    model=KnownDependence(schedule)
    batch=next(_batches(schedule,size=256,seed=3))
    x0,condition,labels=batch["x0"],batch["condition"],batch["labels"]
    full=torch.ones(256,3)
    dropped=full.clone(); dropped[:,0]=0

    def loss(mask,t,noise):
        xt=q_sample(schedule,x0,t,noise)
        return F.mse_loss(model(xt,t,condition,labels,mask),noise,reduction="none").flatten(1).mean(1)

    paired,independent=[],[]
    for trial in range(24):
        generator=torch.Generator().manual_seed(trial)
        t=torch.randint(0,50,(256,),generator=generator)
        noise=torch.randn(x0.shape,generator=generator)
        paired.append((loss(dropped,t,noise)-loss(full,t,noise)).mean())
        other=torch.Generator().manual_seed(trial+1000)
        t2=torch.randint(0,50,(256,),generator=other)
        noise2=torch.randn(x0.shape,generator=other)
        independent.append((loss(dropped,t,noise)-loss(full,t2,noise2)).mean())
    reduction=(torch.stack(independent).std()/torch.stack(paired).std()).item()
    assert reduction>3.0,f"pairing only reduced the standard deviation by {reduction:.1f}x"
