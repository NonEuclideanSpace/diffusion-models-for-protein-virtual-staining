import pytest
import torch

from pvs.eval.calibration import (
    DIVERGENCES,
    categorical_kl,
    collapse_toward_mode,
    detection_threshold,
    divergence,
    hellinger,
    jensen_shannon,
    null_distribution,
    total_variation,
    trend_statistic,
)


@pytest.mark.parametrize("name",list(DIVERGENCES))
def test_identical_distributions_have_zero_divergence(name):
    p=torch.tensor([0.3,0.25,0.2,0.15,0.1])
    assert divergence(name)(p,p).item()==pytest.approx(0.0,abs=1e-6)


@pytest.mark.parametrize("name",["jensen_shannon","total_variation","hellinger"])
def test_symmetric_divergences_are_symmetric(name):
    p,q=torch.tensor([0.5,0.3,0.2]),torch.tensor([0.2,0.3,0.5])
    fn=divergence(name)
    assert fn(p,q).item()==pytest.approx(fn(q,p).item(),abs=1e-6)


def test_kl_is_not_symmetric():
    p,q=torch.tensor([0.8,0.15,0.05]),torch.tensor([0.4,0.3,0.3])
    assert abs(categorical_kl(p,q).item()-categorical_kl(q,p).item())>1e-3


def test_bounded_divergences_respect_their_bounds():
    p,q=torch.tensor([1.0,0.0,0.0]),torch.tensor([0.0,0.0,1.0])
    assert jensen_shannon(p,q).item()<=torch.tensor(2.0).log().item()+1e-6
    assert total_variation(p,q).item()==pytest.approx(1.0,abs=1e-6)
    assert hellinger(p,q).item()<=1.0+1e-6


def test_kl_explodes_on_an_empty_bin_where_the_others_do_not():
    """This is why plug-in KL fails at small samples: one empty bin dominates the value."""
    p,q=torch.tensor([0.5,0.5,0.0]),torch.tensor([0.34,0.33,0.33])
    assert categorical_kl(q,p)>5.0
    assert jensen_shannon(q,p)<0.4
    assert total_variation(q,p)<0.4


def test_counts_are_normalized_before_comparison():
    p=torch.tensor([30.0,20.0,10.0])
    torch.testing.assert_close(jensen_shannon(p,p*7),torch.tensor(0.0),atol=1e-6,rtol=0)


def test_null_floor_falls_as_the_sample_grows():
    reference=torch.ones(5)/5
    floors=[detection_threshold(reference,n,100,"jensen_shannon",trials=400) for n in (20,100,500)]
    assert floors[0]>floors[1]>floors[2]


def test_kl_floor_is_far_worse_than_the_bounded_alternatives_at_small_n():
    reference=torch.ones(10)/10
    kl=detection_threshold(reference,20,100,"kl",trials=400)
    js=detection_threshold(reference,20,100,"jensen_shannon",trials=400)
    assert kl>10*js


def test_null_distribution_is_centred_below_a_real_signal():
    reference=torch.tensor([0.3,0.25,0.2,0.15,0.1])
    floor=null_distribution(reference,200,100,"jensen_shannon",trials=400).median()
    collapsed=collapse_toward_mode(reference,0.5)
    signal=jensen_shannon(collapsed,reference)
    assert signal>5*floor


def test_collapse_moves_mass_to_the_dominant_bin():
    reference=torch.tensor([0.3,0.25,0.2,0.15,0.1])
    for weight in (0.25,0.5,1.0):
        collapsed=collapse_toward_mode(reference,weight)
        assert collapsed[0].item()>reference[0].item()
        assert collapsed.sum().item()==pytest.approx(1.0,abs=1e-6)
    torch.testing.assert_close(collapse_toward_mode(reference,1.0),
                               torch.tensor([1.0,0.0,0.0,0.0,0.0]))


def test_trend_statistic_is_signed_and_bounded():
    assert trend_statistic(torch.tensor([1.0,2.0,3.0,4.0]))==pytest.approx(1.0)
    assert trend_statistic(torch.tensor([4.0,3.0,2.0,1.0]))==pytest.approx(-1.0)
    assert abs(trend_statistic(torch.tensor([1.0,3.0,2.0,4.0])))<1.0


def test_trend_statistic_rejects_a_short_sweep():
    with pytest.raises(ValueError):
        trend_statistic(torch.tensor([1.0,2.0]))


def test_unknown_divergence_is_rejected():
    with pytest.raises(ValueError,match="jensen_shannon"):
        divergence("wasserstein")


def test_average_ranks_ties_are_averaged():
    from pvs.eval.calibration import average_ranks
    assert average_ranks(torch.tensor([5.,5.,1.,9.])).tolist()==pytest.approx([1.5,1.5,0.0,3.0])
    assert average_ranks(torch.tensor([2.,2.,2.,2.])).tolist()==pytest.approx([1.5]*4)
    assert average_ranks(torch.tensor([1.,2.,3.])).tolist()==pytest.approx([0.0,1.0,2.0])


def test_trend_statistic_scores_a_flat_sweep_as_no_trend():
    # Ordinal ranking returns +1 here, because argsort is stable and gives back the original
    # order. M1's gate asks for a strictly increasing D_bar, so a sweep that does nothing must
    # not score the same as one that rises.
    assert trend_statistic(torch.ones(5))==pytest.approx(0.0)
    assert trend_statistic(torch.full((4,),3.7))==pytest.approx(0.0)


def test_trend_statistic_endpoints_and_partial_ties():
    assert trend_statistic(torch.tensor([1.,2.,3.,4.]))==pytest.approx(1.0)
    assert trend_statistic(torch.tensor([4.,3.,2.,1.]))==pytest.approx(-1.0)
    rising=trend_statistic(torch.tensor([1.,1.,1.,2.]))
    falling=trend_statistic(torch.tensor([2.,2.,2.,1.]))
    assert 0.0<rising<1.0 and falling==pytest.approx(-rising)


def test_trend_statistic_is_unchanged_by_a_monotone_rescaling():
    values=torch.tensor([0.11,0.14,0.13,0.19])
    assert trend_statistic(values)==pytest.approx(trend_statistic(values*100+7))
