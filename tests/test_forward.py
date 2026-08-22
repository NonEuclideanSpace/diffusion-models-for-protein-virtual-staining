import pytest
import torch

from pvs.diffusion import (
    NoiseSchedule,
    predict_noise_from_x0,
    predict_x0_from_noise,
    q_posterior,
    q_sample,
    signal_to_noise_ratio,
)

T=200


def test_q_sample_is_deterministic_given_noise():
    s=NoiseSchedule.make(T)
    x0,noise=torch.randn(16,3,8,8),torch.randn(16,3,8,8)
    t=torch.randint(0,T,(16,))
    torch.testing.assert_close(q_sample(s,x0,t,noise),q_sample(s,x0,t,noise))


def test_q_sample_matches_closed_form_moments():
    s=NoiseSchedule.make(T)
    step=140
    x0=torch.full((40000,1),2.0)
    xt=q_sample(s,x0,torch.full((40000,),step,dtype=torch.long))
    assert xt.mean().item()==pytest.approx(2.0*s.sqrt_alpha_bars[step].item(),abs=0.02)
    assert xt.std().item()==pytest.approx(s.sqrt_one_minus_alpha_bars[step].item(),abs=0.02)


def test_q_sample_endpoints():
    s=NoiseSchedule.make(1000)
    n=16384
    x0=torch.randn(n,2)
    early=q_sample(s,x0,torch.zeros(n,dtype=torch.long))
    late=q_sample(s,x0,torch.full((n,),999,dtype=torch.long))
    correlation=lambda a,b:torch.corrcoef(torch.stack([a.flatten(),b.flatten()]))[0,1].item()
    assert correlation(early,x0)>0.99
    assert abs(correlation(late,x0))<0.05


def test_x0_and_noise_predictions_are_mutual_inverses():
    s=NoiseSchedule.make(T)
    x0,noise=torch.randn(32,4),torch.randn(32,4)
    t=torch.randint(0,T,(32,))
    xt=q_sample(s,x0,t,noise)
    torch.testing.assert_close(predict_x0_from_noise(s,xt,t,noise),x0,rtol=1e-3,atol=1e-3)
    torch.testing.assert_close(predict_noise_from_x0(s,xt,t,x0),noise,rtol=1e-3,atol=1e-3)


def test_posterior_sample_reproduces_forward_marginal():
    s=NoiseSchedule.make(T)
    step,scale,n=120,3.0,40000
    x0=torch.randn(n,1)*scale
    t=torch.full((n,),step,dtype=torch.long)
    mean,log_variance=q_posterior(s,x0,q_sample(s,x0,t),t)
    x_prev=mean+(0.5*log_variance).exp()*torch.randn_like(mean)
    expected=(s.alpha_bars[step-1]*scale**2+1-s.alpha_bars[step-1]).sqrt()
    assert x_prev.std().item()==pytest.approx(expected.item(),rel=0.02)


def test_posterior_at_step_zero_is_the_clean_sample():
    s=NoiseSchedule.make(T)
    x0=torch.randn(64,2)
    t=torch.zeros(64,dtype=torch.long)
    mean,_=q_posterior(s,x0,q_sample(s,x0,t),t)
    torch.testing.assert_close(mean,x0,rtol=1e-3,atol=1e-3)


def test_snr_decreases_monotonically():
    snr=signal_to_noise_ratio(NoiseSchedule.make(T))
    assert bool((snr[1:]<=snr[:-1]).all())
