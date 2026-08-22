import pytest
import torch

from pvs.diffusion import NoiseSchedule,ddim_invert,ddim_sample,uniform_timesteps
from pvs.diffusion.edict import edict_invert,edict_sample

T=200


def _oracle64(schedule):
    """Exact score for unit-variance Gaussian data, kept in float64 throughout.

    EDICT reproduces the forward pass's noise estimates only if the model sees bit-identical
    inputs. Casting the float64 coupling state down to float32 at each call quantizes it and
    costs seven orders of magnitude, so the precision test needs a float64 model.
    """
    alpha_bars=schedule.alpha_bars.double()

    def model(xt,t):
        alpha_bar=alpha_bars.gather(0,t).reshape(-1,*(1,)*(xt.ndim-1))
        return (1-alpha_bar).sqrt()*xt+0.01*torch.sin(3*xt)

    return model


def _round_trip(model,schedule,x0,**kwargs)->float:
    latents=edict_invert(model,schedule,x0,**kwargs)
    recon=edict_sample(model,schedule,latents=latents,**kwargs)
    return ((recon-x0).norm()/x0.norm()).item()


def test_round_trip_reaches_machine_precision():
    """Exactness is algebraic, so it does not depend on the model being any good."""
    s=NoiseSchedule.make(T,"linear")
    assert _round_trip(_oracle64(s),s,torch.randn(128,2,dtype=torch.float64),num_inference_steps=50)<1e-13


def test_exactness_does_not_degrade_with_step_count():
    """The DDIM signature is error falling with steps. The EDICT signature is flatness."""
    s=NoiseSchedule.make(T,"linear")
    model=_oracle64(s)
    x0=torch.randn(128,2,dtype=torch.float64)
    errors=[_round_trip(model,s,x0,num_inference_steps=n) for n in (25,50,100)]
    assert max(errors)<1e-13
    assert max(errors)/min(errors)<5


def test_mixing_dilation_erodes_precision_at_many_steps():
    """Un-mixing divides by p once per step, so floating-point differences between the two
    sequences are dilated geometrically. At 200 steps that costs four digits."""
    s=NoiseSchedule.make(T,"linear")
    model=_oracle64(s)
    x0=torch.randn(128,2,dtype=torch.float64)
    assert _round_trip(model,s,x0,num_inference_steps=100)*100<_round_trip(model,s,x0,num_inference_steps=200)


def test_beats_fixed_point_ddim_by_orders_of_magnitude():
    s=NoiseSchedule.make(T,"linear")
    model=_oracle64(s)
    x0=torch.randn(128,2,dtype=torch.float64)
    latent=ddim_invert(model,s,x0,num_inference_steps=50,fixed_point_steps=2)
    ddim_error=((ddim_sample(model,s,latent=latent,num_inference_steps=50,eta=0.0)-x0).norm()/x0.norm()).item()
    assert _round_trip(model,s,x0,num_inference_steps=50)*1000<ddim_error


def test_terminal_snr_sets_the_precision_floor():
    """Round-trip error scales as a_T = sqrt(alpha_bar_prev / alpha_bar_T) times epsilon.
    Cosine drives alpha_bar_T to 6e-8, which costs about eleven digits."""
    x0=torch.randn(128,2,dtype=torch.float64)
    errors={}
    for kind in ("linear","cosine"):
        s=NoiseSchedule.make(T,kind)
        errors[kind]=_round_trip(_oracle64(s),s,x0,num_inference_steps=50)
    assert errors["linear"]<1e-13
    assert errors["cosine"]>errors["linear"]*1e3


def test_truncating_before_the_singularity_restores_precision():
    s=NoiseSchedule.make(T,"cosine")
    model=_oracle64(s)
    x0=torch.randn(128,2,dtype=torch.float64)
    full=uniform_timesteps(T,50)
    assert _round_trip(model,s,x0,timesteps=full[:-1])*1e3<_round_trip(model,s,x0,timesteps=full)


@pytest.mark.parametrize("mixing",[0.90,0.93,0.97])
def test_mixing_inside_the_safe_envelope_stays_exact(mixing):
    s=NoiseSchedule.make(T,"linear")
    x0=torch.randn(128,2,dtype=torch.float64)
    assert _round_trip(_oracle64(s),s,x0,num_inference_steps=50,mixing=mixing)<1e-13


def test_low_mixing_diverges_the_coupled_pair():
    """Below the safe envelope the mixing layer dilates instead of averaging."""
    s=NoiseSchedule.make(T,"linear")
    model=_oracle64(s)
    x0=torch.randn(128,2,dtype=torch.float64)
    safe=edict_invert(model,s,x0,num_inference_steps=50,mixing=0.93)
    unsafe=edict_invert(model,s,x0,num_inference_steps=50,mixing=0.60)
    drift=lambda pair:((pair[0]-pair[1]).norm()/pair[0].norm()).item()
    assert drift(safe)<0.05
    assert drift(unsafe)>1.0
    assert _round_trip(model,s,x0,num_inference_steps=50,mixing=0.60)>1e-6


def test_sample_accepts_shape_latent_or_pair():
    s=NoiseSchedule.make(T,"linear")
    model=_oracle64(s)
    assert edict_sample(model,s,(4,2),num_inference_steps=10).shape==(4,2)
    z=torch.randn(4,2)
    torch.testing.assert_close(
        edict_sample(model,s,latent=z,num_inference_steps=10),
        edict_sample(model,s,latents=(z,z),num_inference_steps=10),
    )
    with pytest.raises(ValueError):
        edict_sample(model,s,num_inference_steps=10)
