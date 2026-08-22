import pytest
import torch

from pvs.diffusion import NoiseSchedule,ddpm_loss,ddpm_sample,q_sample
from pvs.models.toy import AnalyticGaussianPredictor,ToyMLP

T=200


def test_loss_is_scalar_and_finite():
    s=NoiseSchedule.make(T)
    loss=ddpm_loss(ToyMLP(),s,torch.randn(8,2))
    assert loss.ndim==0 and torch.isfinite(loss)


def test_oracle_beats_predicting_zero():
    s=NoiseSchedule.make(T,"cosine")
    x0=torch.randn(4096,2)
    t=torch.randint(0,T,(4096,))
    oracle=AnalyticGaussianPredictor(s,1.0)
    assert ddpm_loss(oracle,s,x0,t)<ddpm_loss(lambda xt,t:torch.zeros_like(xt),s,x0,t)


def test_loss_gradient_flows_to_every_parameter():
    model=ToyMLP()
    ddpm_loss(model,NoiseSchedule.make(T),torch.randn(16,2)).backward()
    for name,p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(),name


def test_sample_shape_and_finiteness():
    s=NoiseSchedule.make(20)
    out=ddpm_sample(AnalyticGaussianPredictor(s,1.0),s,(6,3,4,4))
    assert out.shape==(6,3,4,4) and torch.isfinite(out).all()


def test_sample_recovers_gaussian_scale_when_terminal_snr_is_zero():
    s=NoiseSchedule.make(T,"cosine")
    out=ddpm_sample(AnalyticGaussianPredictor(s,2.0),s,(4096,2))
    assert out.std().item()==pytest.approx(2.0,rel=0.05)
    assert out.mean().abs().item()<0.1


def test_nonzero_terminal_snr_biases_the_sample_scale():
    """Linear betas at T=200 stop at alpha_bar=0.13, so N(0, I) is the wrong start."""
    s=NoiseSchedule.make(T,"linear")
    model=AnalyticGaussianPredictor(s,2.0)
    assert s.alpha_bars[-1].item()>0.1
    assert ddpm_sample(model,s,(4096,2)).std().item()<1.9
    alpha_bar=s.alpha_bars[-1]
    latent=torch.randn(4096,2)*(alpha_bar*4+1-alpha_bar).sqrt()
    assert ddpm_sample(model,s,latent=latent).std().item()==pytest.approx(2.0,rel=0.05)


def test_generator_makes_sampling_reproducible():
    s=NoiseSchedule.make(20)
    model=AnalyticGaussianPredictor(s,1.0)
    a=ddpm_sample(model,s,(8,2),generator=torch.Generator().manual_seed(7))
    b=ddpm_sample(model,s,(8,2),generator=torch.Generator().manual_seed(7))
    torch.testing.assert_close(a,b)


def test_clip_bounds_the_x0_estimate():
    s=NoiseSchedule.make(50,"cosine")
    out=ddpm_sample(AnalyticGaussianPredictor(s,5.0),s,(512,2),clip=(-1.0,1.0))
    assert out.abs().max().item()<1.5


def test_sample_requires_shape_or_latent():
    s=NoiseSchedule.make(10)
    with pytest.raises(ValueError):
        ddpm_sample(AnalyticGaussianPredictor(s,1.0),s)
