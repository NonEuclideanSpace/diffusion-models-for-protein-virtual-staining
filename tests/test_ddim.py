import pytest
import torch

from pvs.diffusion import NoiseSchedule,ddim_invert,ddim_sample,uniform_timesteps
from pvs.models.toy import AnalyticGaussianPredictor

T=200


def test_uniform_timesteps_spans_the_schedule():
    grid=uniform_timesteps(1000,50)
    assert grid.dtype==torch.long and len(grid)==50
    assert grid[0].item()==0 and grid[-1].item()==999
    assert bool((grid[1:]>grid[:-1]).all())


@pytest.mark.parametrize("n",[0,-1,1001])
def test_uniform_timesteps_rejects_out_of_range(n):
    with pytest.raises(ValueError):
        uniform_timesteps(1000,n)


def test_eta_zero_is_deterministic_and_eta_one_is_not():
    s=NoiseSchedule.make(T,"cosine")
    model=AnalyticGaussianPredictor(s,1.0)
    latent=torch.randn(32,2)
    a=ddim_sample(model,s,latent=latent,num_inference_steps=20,eta=0.0)
    b=ddim_sample(model,s,latent=latent,num_inference_steps=20,eta=0.0)
    torch.testing.assert_close(a,b)
    c=ddim_sample(model,s,latent=latent,num_inference_steps=20,eta=1.0,
                  generator=torch.Generator().manual_seed(0))
    assert (c-a).abs().max().item()>1e-3


def test_sample_recovers_gaussian_scale():
    s=NoiseSchedule.make(T,"cosine")
    out=ddim_sample(AnalyticGaussianPredictor(s,2.0),s,(4096,2),num_inference_steps=100)
    assert out.std().item()==pytest.approx(2.0,rel=0.06)


def test_inversion_round_trip_is_accurate():
    """The sharpest available correctness check on the sampler pair."""
    s=NoiseSchedule.make(T,"cosine")
    model=AnalyticGaussianPredictor(s,1.0)
    x0=torch.randn(256,2)
    latent=ddim_invert(model,s,x0,num_inference_steps=100)
    recon=ddim_sample(model,s,latent=latent,num_inference_steps=100,eta=0.0)
    assert ((recon-x0).norm()/x0.norm()).item()<0.05


def test_inversion_error_falls_first_order_in_step_count():
    """Halving the step size should roughly halve the error. Catches wrong alpha indexing."""
    s=NoiseSchedule.make(T,"cosine")
    model=AnalyticGaussianPredictor(s,1.0)
    x0=torch.randn(256,2)
    errors=[]
    for n in (25,50,100,200):
        latent=ddim_invert(model,s,x0,num_inference_steps=n)
        recon=ddim_sample(model,s,latent=latent,num_inference_steps=n,eta=0.0)
        errors.append(((recon-x0).norm()/x0.norm()).item())
    for coarse,fine in zip(errors,errors[1:]):
        assert 1.6<coarse/fine<2.6,errors


def test_inversion_leaves_a_standard_normal_latent():
    s=NoiseSchedule.make(T,"cosine")
    latent=ddim_invert(AnalyticGaussianPredictor(s,1.0),s,torch.randn(4096,2),num_inference_steps=100)
    assert latent.std().item()==pytest.approx(1.0,rel=0.06)
    assert latent.mean().abs().item()<0.1


def _round_trip(model,schedule,x0,grid,fixed_point_steps=0)->float:
    latent=ddim_invert(model,schedule,x0,timesteps=grid,fixed_point_steps=fixed_point_steps)
    recon=ddim_sample(model,schedule,latent=latent,timesteps=grid,eta=0.0)
    return ((recon-x0).norm()/x0.norm()).item()


def test_explicit_timestep_grid_matches_the_uniform_default():
    s=NoiseSchedule.make(T,"cosine")
    model=AnalyticGaussianPredictor(s,1.0)
    latent=torch.randn(16,2)
    a=ddim_sample(model,s,latent=latent,num_inference_steps=40,eta=0.0)
    b=ddim_sample(model,s,latent=latent,timesteps=uniform_timesteps(T,40),eta=0.0)
    torch.testing.assert_close(a,b)


def test_fixed_point_refinement_sharpens_inversion():
    """Two iterations should buy at least an order of magnitude on a well-posed range."""
    s=NoiseSchedule.make(T,"cosine")
    model=AnalyticGaussianPredictor(s,1.0)
    x0=torch.randn(256,2)
    grid=uniform_timesteps(T,80)
    assert _round_trip(model,s,x0,grid,2)*10<_round_trip(model,s,x0,grid,0)


def test_inversion_amplifies_model_error_as_alpha_bar_reaches_zero():
    """The final step divides by sqrt(alpha_bar). Once that underflows, any error in the
    noise estimate passes through at full strength: a 1% model error becomes a 38% round
    trip. Truncating the grid one alpha_bar decade earlier removes the amplification."""
    s=NoiseSchedule.make(T,"cosine")
    oracle=AnalyticGaussianPredictor(s,1.0)
    perturbed=lambda xt,t:oracle(xt,t)+0.01*torch.sin(3*xt)
    x0=torch.randn(256,2)
    full=uniform_timesteps(T,100)
    assert s.alpha_bars[full[-1]].item()<1e-6
    assert s.alpha_bars[full[-6]].item()>1e-3
    assert _round_trip(perturbed,s,x0,full[:-5],2)*100<_round_trip(perturbed,s,x0,full,2)


def test_linear_schedule_inverts_cleanly_over_its_whole_range():
    """Linear betas never reach zero terminal SNR, which is exactly what inversion wants."""
    s=NoiseSchedule.make(T,"linear")
    model=AnalyticGaussianPredictor(s,1.0)
    assert s.alpha_bars[-1].item()>0.1
    assert _round_trip(model,s,torch.randn(256,2),uniform_timesteps(T,100),2)<1e-3
