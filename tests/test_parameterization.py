import pytest
import torch

from pvs.diffusion import (
    TARGETS,
    NoiseSchedule,
    as_eps_predictor,
    enforce_zero_terminal_snr,
    eps_from_v,
    linear_beta_schedule,
    q_sample,
    to_eps,
    training_target,
    v_target,
    x0_from_v,
)

T=200


def _fixture():
    s=NoiseSchedule.make(T,"cosine")
    x0,noise=torch.randn(32,3,8,8),torch.randn(32,3,8,8)
    t=torch.randint(0,T,(32,))
    return s,x0,noise,t,q_sample(s,x0,t,noise)


def test_v_recovers_both_endpoints():
    s,x0,noise,t,xt=_fixture()
    v=v_target(s,x0,noise,t)
    torch.testing.assert_close(x0_from_v(s,xt,v,t),x0,rtol=1e-3,atol=1e-4)
    torch.testing.assert_close(eps_from_v(s,xt,v,t),noise,rtol=1e-3,atol=1e-4)


@pytest.mark.parametrize("target",TARGETS)
def test_to_eps_agrees_across_parameterizations(target):
    s,x0,noise,t,xt=_fixture()
    prediction=training_target(s,x0,noise,t,target)
    torch.testing.assert_close(to_eps(s,prediction,xt,t,target),noise,rtol=1e-3,atol=1e-4)


def test_unknown_target_is_rejected():
    s,x0,noise,t,xt=_fixture()
    with pytest.raises(ValueError,match="v"):
        to_eps(s,noise,xt,t,"score")
    with pytest.raises(ValueError,match="v"):
        training_target(s,x0,noise,t,"score")


def test_v_is_bounded_where_eps_and_x0_are_not():
    """The reason v exists: its recovery coefficients stay bounded at both ends of the
    schedule, while eps-prediction blows up as alpha_bar goes to zero."""
    s=NoiseSchedule.make(1000,"cosine")
    late=torch.tensor([999])
    assert (1/s.sqrt_alpha_bars[late]).item()>100
    assert s.sqrt_alpha_bars[late].item()<1 and s.sqrt_one_minus_alpha_bars[late].item()<=1


def test_as_eps_predictor_wraps_a_v_model():
    s,x0,noise,t,xt=_fixture()
    v_model=lambda x,step:v_target(s,x0,noise,step)
    torch.testing.assert_close(as_eps_predictor(v_model,s,"v")(xt,t),noise,rtol=1e-3,atol=1e-4)
    identity=lambda x,step:noise
    assert as_eps_predictor(identity,s,"eps") is identity


def test_terminal_snr_rescaling_hits_its_floor_and_stays_valid():
    for floor in (1e-2,1e-3):
        betas=enforce_zero_terminal_snr(linear_beta_schedule(T),floor)
        s=NoiseSchedule(betas)
        assert s.alpha_bars[-1].sqrt().item()==pytest.approx(floor,rel=1e-6)
        assert bool(((s.betas>0)&(s.betas<1)).all())
        assert bool((s.alpha_bars[1:]<=s.alpha_bars[:-1]).all())


def test_terminal_snr_rescaling_removes_the_sampling_bias():
    """A schedule that stops at alpha_bar 0.13 biases the sample scale; rescaling fixes it."""
    from pvs.diffusion import ddpm_sample
    from pvs.models.toy import AnalyticGaussianPredictor
    plain=NoiseSchedule.make(T,"linear")
    fixed=NoiseSchedule.make(T,"linear",terminal_snr_floor=1e-2)
    assert plain.alpha_bars[-1].item()>0.1
    biased=ddpm_sample(AnalyticGaussianPredictor(plain,2.0),plain,(4096,2)).std().item()
    corrected=ddpm_sample(AnalyticGaussianPredictor(fixed,2.0),fixed,(4096,2)).std().item()
    assert biased<1.9
    assert corrected==pytest.approx(2.0,rel=0.05)


def test_rescaling_rejects_an_out_of_range_floor():
    with pytest.raises(ValueError):
        enforce_zero_terminal_snr(linear_beta_schedule(T),1.5)
