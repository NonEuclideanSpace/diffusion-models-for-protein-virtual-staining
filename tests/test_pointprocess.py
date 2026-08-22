import math

import pytest
import torch

from pvs.eval.pointprocess import (blur, detect_spots, laplacian_of_gaussian, nearest_neighbour,
                                   ripley_k, ripley_l, summarize)


def plant(positions,size=96,sigma=2.0,amplitude=1.0)->torch.Tensor:
    rows,columns=torch.meshgrid(torch.arange(float(size)),torch.arange(float(size)),indexing="ij")
    image=torch.zeros(size,size)
    for row,column in positions:
        image=image+amplitude*torch.exp(-((rows-row)**2+(columns-column)**2)/(2*sigma**2))
    return image


def test_blur_preserves_total_mass():
    image=torch.rand(48,48)
    assert blur(image,2.0).sum()==pytest.approx(float(image.sum()),rel=0.02)


def test_log_peaks_at_a_matching_scale():
    image=plant([(24,24)],size=48,sigma=3.0)
    responses=[float(laplacian_of_gaussian(image,s)[24,24]) for s in (1.0,3.0,9.0)]
    assert responses[1]>responses[0] and responses[1]>responses[2]


def test_detects_every_planted_spot_and_no_others():
    positions=[(12,12),(20,60),(70,25),(60,70)]
    found,_,_=detect_spots(plant(positions),threshold=0.05)
    assert len(found)==len(positions)
    for row,column in positions:
        assert any(abs(p[0]-row)<=1 and abs(p[1]-column)<=1 for p in found.tolist())


def test_one_blob_is_not_reported_once_per_scale():
    found,_,_=detect_spots(plant([(48,48)],sigma=4.0),sigmas=(1.0,2.0,4.0,8.0),threshold=0.02)
    assert len(found)==1


def test_radius_tracks_blob_size():
    small=detect_spots(plant([(48,48)],sigma=1.5),threshold=0.02)[1]
    large=detect_spots(plant([(48,48)],sigma=5.0),sigmas=(1.0,2.5,5.0,8.0),threshold=0.02)[1]
    assert float(small.median())<float(large.median())


def test_mask_excludes_spots_outside_it():
    mask=torch.zeros(96,96); mask[:48]=1
    found,_,_=detect_spots(plant([(12,12),(70,70)]),mask=mask,threshold=0.05)
    assert len(found)==1 and found[0,0]<48


def test_empty_image_returns_no_spots():
    found,radii,strength=detect_spots(torch.zeros(48,48),threshold=0.05)
    assert len(found)==0 and len(radii)==0 and len(strength)==0


def test_rejects_multichannel_input():
    with pytest.raises(ValueError):
        detect_spots(torch.zeros(3,48,48))


def test_nearest_neighbour_matches_a_hand_computation():
    points=torch.tensor([[0.,0.],[3.,4.],[0.,10.]])
    # (0,10) is 10 from the origin but sqrt(45) from (3,4), so sqrt(45) is its neighbour.
    assert nearest_neighbour(points).tolist()==pytest.approx([5.0,5.0,math.sqrt(45)],abs=1e-5)


def test_nearest_neighbour_needs_two_points():
    assert len(nearest_neighbour(torch.tensor([[1.,1.]])))==0


def _uniform(n,size,seed):
    generator=torch.Generator().manual_seed(seed)
    return torch.rand(n,2,generator=generator)*size


def _clustered(centres,per,size,spread,seed):
    generator=torch.Generator().manual_seed(seed)
    anchors=torch.rand(centres,2,generator=generator)*size
    points=anchors.repeat_interleave(per,0)+torch.randn(centres*per,2,generator=generator)*spread
    return points.clamp(0,size-1)


def _regular(side,size):
    step=size/side
    grid=torch.arange(side,dtype=torch.float32)*step+step/2
    rows,columns=torch.meshgrid(grid,grid,indexing="ij")
    return torch.stack([rows.flatten(),columns.flatten()],dim=1)


RADII=torch.linspace(4,24,8)


def test_ripley_l_separates_clustered_random_and_regular():
    size=128
    mask=torch.ones(size,size)
    clustered=ripley_l(_clustered(8,12,size,4.0,0),mask,RADII)
    random=ripley_l(_uniform(96,size,1),mask,RADII)
    regular=ripley_l(_regular(10,size),mask,RADII)
    middle=slice(1,5)
    assert clustered[middle].mean()>3.0
    assert abs(float(random[middle].mean()))<3.0
    assert regular[middle].mean()<float(random[middle].mean())


def test_ripley_l_of_a_random_pattern_is_near_zero():
    mask=torch.ones(160,160)
    values=torch.stack([ripley_l(_uniform(140,160,seed),mask,RADII) for seed in range(6)])
    assert abs(float(values.mean(0)[1:5].mean()))<2.5


def test_ripley_k_grows_with_radius():
    mask=torch.ones(128,128)
    k=ripley_k(_uniform(80,128,3),mask,RADII)
    assert all(k[i]<=k[i+1]+1e-6 for i in range(len(k)-1))


def test_ripley_k_is_empty_without_two_points():
    assert float(ripley_k(torch.zeros(1,2),torch.ones(32,32),RADII).abs().sum())==0.0


def test_edge_correction_raises_k_for_points_near_a_boundary():
    mask=torch.ones(128,128)
    corner=torch.tensor([[2.,2.],[2.,10.],[10.,2.],[10.,10.]])
    centre=corner+58.0
    assert float(ripley_k(corner,mask,RADII).sum())>float(ripley_k(centre,mask,RADII).sum())


def test_summary_reports_count_density_and_a_curve():
    mask=torch.zeros(96,96); mask[8:88,8:88]=1
    report=summarize(plant([(20,20),(30,60),(60,30),(70,70)]),mask,threshold=0.05)
    assert report["count"]==4
    assert report["density"]==pytest.approx(4/float((mask>0).sum()))
    assert report["median_radius"]>0 and report["median_nearest"]>0
    assert len(report["ripley_l"])==len(report["radii"])
