import math

import pytest
import torch

from pvs.diffusion import NoiseSchedule,cosine_beta_schedule,extract,linear_beta_schedule

T=200
KINDS=["linear","cosine"]


@pytest.mark.parametrize("fn",[linear_beta_schedule,cosine_beta_schedule])
def test_beta_schedules_are_1d_float64(fn):
    betas=fn(T)
    assert betas.shape==(T,)
    assert betas.dtype==torch.float64


def test_linear_endpoints_and_monotonicity():
    betas=linear_beta_schedule(T,1e-4,0.02)
    assert math.isclose(betas[0].item(),1e-4)
    assert math.isclose(betas[-1].item(),0.02)
    assert bool((betas[1:]>=betas[:-1]).all())


def test_cosine_is_clipped_below_one():
    assert cosine_beta_schedule(T).max().item()<=0.999+1e-9


def test_cosine_retains_more_signal_at_midpoint():
    assert NoiseSchedule.make(1000,"cosine").alpha_bars[500]>NoiseSchedule.make(1000,"linear").alpha_bars[500]


@pytest.mark.parametrize("kind",KINDS)
def test_alpha_identities(kind):
    s=NoiseSchedule.make(T,kind)
    torch.testing.assert_close(s.alphas,1-s.betas)
    torch.testing.assert_close(s.alpha_bars.double(),(1-s.betas.double()).cumprod(0),rtol=1e-5,atol=1e-7)
    torch.testing.assert_close(s.sqrt_alpha_bars**2+s.sqrt_one_minus_alpha_bars**2,torch.ones(T),rtol=1e-4,atol=1e-5)
    assert bool((s.alpha_bars[1:]<=s.alpha_bars[:-1]).all())


@pytest.mark.parametrize("kind",KINDS)
def test_alpha_bar_prev_is_shifted_with_leading_one(kind):
    s=NoiseSchedule.make(T,kind)
    assert math.isclose(s.alpha_bars_prev[0].item(),1.0,rel_tol=1e-6)
    torch.testing.assert_close(s.alpha_bars_prev[1:],s.alpha_bars[:-1])


@pytest.mark.parametrize("kind",KINDS)
def test_posterior_variance_vanishes_at_zero_and_is_positive_after(kind):
    s=NoiseSchedule.make(T,kind)
    assert s.posterior_variance[0].item()<1e-8
    assert bool((s.posterior_variance[1:]>0).all())


@pytest.mark.parametrize("kind",KINDS)
def test_constants_stored_as_float32(kind):
    for name,value in vars(NoiseSchedule.make(T,kind)).items():
        if isinstance(value,torch.Tensor):
            assert value.dtype==torch.float32,name


def test_rejects_invalid_betas():
    with pytest.raises(ValueError):
        NoiseSchedule(torch.tensor([0.1,0.5,1.5]))
    with pytest.raises(ValueError):
        NoiseSchedule(torch.tensor([0.0,0.1,0.2]))
    with pytest.raises(ValueError):
        NoiseSchedule(torch.rand(3,4)*0.1+0.01)


def test_make_rejects_unknown_kind_and_forwards_kwargs():
    with pytest.raises(ValueError,match="cosine"):
        NoiseSchedule.make(T,"sigmoid")
    s=NoiseSchedule.make(T,"linear",beta_start=0.001,beta_end=0.01)
    assert math.isclose(s.betas[0].item(),0.001,rel_tol=1e-4)


def test_terminal_noise_is_nearly_pure_at_default_length():
    assert NoiseSchedule.make(1000,"linear").alpha_bars[-1].item()<1e-3


def test_short_schedule_with_default_endpoints_barely_noises():
    assert NoiseSchedule.make(50,"linear").alpha_bars[-1].item()>0.5


def test_len_and_device_and_to():
    s=NoiseSchedule.make(T,"linear")
    assert len(s)==T
    assert s.to("cpu") is s
    assert s.device.type=="cpu"


def test_extract_shapes_values_and_guards():
    s=NoiseSchedule.make(T,"linear")
    t=torch.tensor([0,7,199,50])
    out=extract(s.sqrt_alpha_bars,t,4)
    assert out.shape==(4,1,1,1)
    torch.testing.assert_close(out.flatten(),s.sqrt_alpha_bars[t])
    assert extract(s.betas,torch.tensor([3,4]),2).shape==(2,1)
    with pytest.raises(TypeError):
        extract(s.betas,torch.tensor([1.0,2.0]),4)
    with pytest.raises(ValueError):
        extract(s.betas,torch.tensor([[1,2],[3,4]]),4)


def test_extract_broadcasts_along_batch_not_width():
    s=NoiseSchedule.make(T,"linear")
    n=8
    out=extract(s.sqrt_alpha_bars,torch.arange(n),4)*torch.ones(n,1,n,n)
    for i in range(n):
        assert out[i].unique().numel()==1
        assert out[i].flatten()[0].item()==pytest.approx(s.sqrt_alpha_bars[i].item(),rel=1e-5)
