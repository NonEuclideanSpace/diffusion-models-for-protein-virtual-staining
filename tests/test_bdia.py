import pytest
import torch

from pvs.diffusion import NoiseSchedule,bdia_invert,bdia_sample,ddim_invert,ddim_sample
from pvs.diffusion.edict import edict_invert,edict_sample

T=200


def _oracle64(schedule,calls=None):
    alpha_bars=schedule.alpha_bars.double()

    def model(xt,t):
        if calls is not None:
            calls[0]+=1
        alpha_bar=alpha_bars.gather(0,t).reshape(-1,*(1,)*(xt.ndim-1))
        return (1-alpha_bar).sqrt()*xt+0.01*torch.sin(3*xt)

    return model


def _round_trip(schedule,x0,steps=50,gamma=1.0,bootstrap_steps=4)->float:
    model=_oracle64(schedule)
    pair=bdia_invert(model,schedule,x0,num_inference_steps=steps,gamma=gamma,
                     bootstrap_steps=bootstrap_steps)
    recon=bdia_sample(model,schedule,latents=pair,num_inference_steps=steps,gamma=gamma)
    return ((recon-x0).norm()/x0.norm()).item()


def test_round_trip_is_exact_on_a_well_conditioned_schedule():
    s=NoiseSchedule.make(T,"linear")
    assert _round_trip(s,torch.randn(128,2,dtype=torch.float64))<1e-10


@pytest.mark.parametrize("gamma",[0.9,0.92,0.95,1.0])
def test_round_trip_holds_across_the_safe_gamma_range(gamma):
    s=NoiseSchedule.make(T,"linear")
    assert _round_trip(s,torch.randn(64,2,dtype=torch.float64),gamma=gamma)<1e-12


def test_exactness_is_flat_in_step_count():
    """EDICT loses four digits at 200 steps to mixing dilation. BDIA has no mixing layer."""
    s=NoiseSchedule.make(T,"linear")
    x0=torch.randn(64,2,dtype=torch.float64)
    errors=[_round_trip(s,x0,steps=n) for n in (25,50,100,200)]
    assert max(errors)<1e-14


def test_a_perfect_round_trip_does_not_imply_a_usable_latent():
    """The failure mode below the safe gamma range is the latent, not the reconstruction.

    At gamma 0.8 the round trip is still at machine precision while the recovered latent has
    a standard deviation over ten times what a Gaussian latent should have. Anything that
    consumes the latent — editing, or estimating a distribution from it — is already broken
    at that point, so the round trip alone must never be used as the acceptance criterion.
    """
    s=NoiseSchedule.make(T,"linear")
    model=_oracle64(s)
    x0=torch.randn(64,2,dtype=torch.float64)
    safe=bdia_invert(model,s,x0,num_inference_steps=50,gamma=1.0)
    unsafe=bdia_invert(model,s,x0,num_inference_steps=50,gamma=0.8)
    assert _round_trip(s,x0,gamma=0.8)<1e-12
    assert 0.8<safe[0].std().item()<1.3
    assert unsafe[0].std().item()>5.0


def test_costs_one_network_evaluation_per_step():
    """The whole point over EDICT: same exactness, half the compute."""
    s=NoiseSchedule.make(T,"linear")
    x0=torch.randn(16,2,dtype=torch.float64)
    steps=20
    bdia_calls=[0]
    bdia_invert(_oracle64(s,bdia_calls),s,x0,num_inference_steps=steps,bootstrap_steps=0)
    edict_calls=[0]
    edict_invert(_oracle64(s,edict_calls),s,x0,num_inference_steps=steps)
    assert bdia_calls[0]<=steps+1
    assert edict_calls[0]==2*steps
    assert bdia_calls[0]*2<=edict_calls[0]+2


def test_matches_edict_accuracy_at_half_the_cost():
    s=NoiseSchedule.make(T,"linear")
    x0=torch.randn(64,2,dtype=torch.float64)
    model=_oracle64(s)
    pair=edict_invert(model,s,x0,num_inference_steps=50)
    edict_error=((edict_sample(model,s,latents=pair,num_inference_steps=50)-x0).norm()/x0.norm()).item()
    assert _round_trip(s,x0)<edict_error*100


def test_the_pair_is_what_buys_exactness_not_the_bootstrap():
    """The recurrence is a bijection on pairs, so whichever second state the bootstrap picks,
    sampling from the returned pair lands back on x0. Refining the bootstrap changes which
    latent is recovered, not whether the round trip closes."""
    s=NoiseSchedule.make(T,"linear")
    model=_oracle64(s)
    x0=torch.randn(64,2,dtype=torch.float64)
    coarse=bdia_invert(model,s,x0,num_inference_steps=50,bootstrap_steps=0)
    refined=bdia_invert(model,s,x0,num_inference_steps=50,bootstrap_steps=4)
    assert _round_trip(s,x0,bootstrap_steps=0)<1e-14
    assert _round_trip(s,x0,bootstrap_steps=4)<1e-14
    assert 0<((coarse[0]-refined[0]).norm()/refined[0].norm()).item()<1e-6


def test_beats_plain_ddim_inversion_by_orders_of_magnitude():
    s=NoiseSchedule.make(T,"linear")
    x0=torch.randn(64,2,dtype=torch.float64)
    model=_oracle64(s)
    latent=ddim_invert(model,s,x0,num_inference_steps=50)  # noqa
    ddim_error=((ddim_sample(model,s,latent=latent,num_inference_steps=50,eta=0.0)-x0).norm()/x0.norm()).item()
    assert _round_trip(s,x0)*1e4<ddim_error


def test_zero_gamma_is_rejected():
    s=NoiseSchedule.make(T,"linear")
    with pytest.raises(ValueError):
        bdia_invert(_oracle64(s),s,torch.randn(8,2,dtype=torch.float64),gamma=0.0)


def test_sample_requires_shape_or_latent():
    s=NoiseSchedule.make(T,"linear")
    with pytest.raises(ValueError):
        bdia_sample(_oracle64(s),s,num_inference_steps=10)
