import pytest
import torch
import torch.nn.functional as F

from pvs.eval.metrics import (
    coverage_statistics,
    frechet_distance,
    glcm_features,
    multiscale_ssim,
    mutual_information,
    pearson_correlation,
    precision_recall,
    ssim,
)


@pytest.fixture
def images():
    return torch.randn(8,1,32,32).clamp(-1,1)


def test_pearson_spans_its_range(images):
    assert pearson_correlation(images,images).mean().item()==pytest.approx(1.0,abs=1e-5)
    assert pearson_correlation(images,-images).mean().item()==pytest.approx(-1.0,abs=1e-5)
    assert abs(pearson_correlation(images,torch.randn_like(images)).mean().item())<0.1


def test_pearson_ignores_affine_rescaling(images):
    assert pearson_correlation(images,3*images+0.5).mean().item()==pytest.approx(1.0,abs=1e-5)


def test_mutual_information_is_highest_with_itself(images):
    other=torch.randn_like(images).clamp(-1,1)
    assert mutual_information(images,images).mean()>mutual_information(images,other).mean()


def test_mutual_information_catches_a_monotone_nonlinearity(images):
    """Pearson misses a strong but non-linear relationship; MI is there to catch it."""
    warped=images.sign()*images.abs().pow(3)
    assert mutual_information(images,warped).mean()>mutual_information(images,torch.randn_like(images)).mean()


def test_ssim_and_msssim_saturate_on_identity(images):
    assert ssim(images,images).mean().item()==pytest.approx(1.0,abs=1e-4)
    assert multiscale_ssim(images,images).mean().item()==pytest.approx(1.0,abs=1e-4)


def test_ssim_falls_as_noise_grows(images):
    values=[ssim(images,images+torch.randn_like(images)*level).mean().item() for level in (0.1,0.5,1.0)]
    assert values[0]>values[1]>values[2]


def test_texture_features_detect_over_smoothing(images):
    """The failure mode of a regression baseline: pixel metrics survive, texture does not."""
    smoothed=F.interpolate(F.avg_pool2d(images,4),size=32,mode="bilinear",align_corners=False)
    sharp_features=glcm_features(images).mean(0)
    smooth_features=glcm_features(smoothed).mean(0)
    assert smooth_features[0]<sharp_features[0]/10
    assert smooth_features[1]>sharp_features[1]*2


def test_frechet_distance_is_zero_for_a_set_against_itself():
    embeddings=torch.randn(300,8)
    assert frechet_distance(embeddings,embeddings)==pytest.approx(0.0,abs=1e-4)


def test_frechet_distance_grows_with_a_mean_shift():
    real=torch.randn(400,8)
    distances=[frechet_distance(real,real+shift) for shift in (0.5,1.0,2.0)]
    assert distances[0]<distances[1]<distances[2]


def test_frechet_distance_requires_matching_dimensions():
    with pytest.raises(ValueError):
        frechet_distance(torch.randn(10,4),torch.randn(10,8))


def test_precision_and_recall_separate_collapse_from_a_shift():
    """The distinguishing test for the project's claim. A model that produces canonical
    outputs scores high precision and near-zero recall; a model that is simply wrong scores
    low on both."""
    torch.manual_seed(0)
    real=torch.randn(400,16)
    collapsed_precision,collapsed_recall=precision_recall(real,torch.randn(400,16)*0.3)
    shifted_precision,shifted_recall=precision_recall(real,torch.randn(400,16)+1.5)
    faithful_precision,faithful_recall=precision_recall(real,torch.randn(400,16))
    assert collapsed_precision>0.9 and collapsed_recall<0.1
    assert shifted_precision<0.3 and shifted_recall<0.1
    assert faithful_precision>0.5 and faithful_recall>0.5


def test_coverage_statistics_expose_variance_collapse():
    real=torch.randn(300,8)
    stats=coverage_statistics(real,torch.randn(300,8)*0.25)
    assert stats["variance_ratio"]<0.2
    assert stats["generated_distance_to_centroid"]<stats["real_distance_to_centroid"]
    faithful=coverage_statistics(real,torch.randn(300,8))
    assert 0.8<faithful["variance_ratio"]<1.25
