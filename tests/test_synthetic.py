import pytest
import torch

from pvs.data.synthetic import COMPARTMENTS,NUM_COMPARTMENTS,SyntheticCells,SyntheticConfig


@pytest.fixture(scope="module")
def dataset():
    return SyntheticCells(SyntheticConfig(image_size=32,num_proteins=16),seed=0)


def test_batch_shapes_and_range(dataset):
    b=dataset.batch(8,torch.Generator().manual_seed(0))
    assert b["x0"].shape==(8,1,32,32)
    assert b["condition"].shape==(8,3,32,32)
    assert b["labels"].shape==(8,) and b["compartments"].shape==(8,)
    for key in ("x0","condition"):
        assert -1.001<=b[key].min() and b[key].max()<=1.001


def test_declared_distribution_matches_what_is_drawn(dataset):
    """The whole point of the synthetic task: pi_emp is known, not estimated."""
    protein=int(dataset.entropy().argmax())
    draws=dataset.sample_compartments(torch.full((20000,),protein),torch.Generator().manual_seed(1))
    empirical=torch.bincount(draws,minlength=NUM_COMPARTMENTS).float()/20000
    torch.testing.assert_close(empirical,dataset.distributions[protein],rtol=0,atol=0.02)


def test_entropy_spans_a_useful_range(dataset):
    entropy=dataset.entropy()
    assert entropy.shape==(16,)
    assert entropy.min()<entropy.max()
    assert entropy.max()<torch.tensor(float(NUM_COMPARTMENTS)).log()+1e-6


def test_compartments_render_differently(dataset):
    """If two compartments produced the same image the task would be degenerate."""
    generator=torch.Generator().manual_seed(3)
    geometry=dataset._geometry(1,generator)
    rendered=[dataset._protein(geometry,torch.tensor([k]),torch.Generator().manual_seed(3))
              for k in range(NUM_COMPARTMENTS)]
    for i in range(NUM_COMPARTMENTS):
        for j in range(i+1,NUM_COMPARTMENTS):
            assert (rendered[i]-rendered[j]).abs().mean().item()>0.01,(COMPARTMENTS[i],COMPARTMENTS[j])


def test_landmarks_depend_on_geometry_not_on_the_protein(dataset):
    """Landmarks are the cheap channels; they must not leak the answer."""
    generator=torch.Generator().manual_seed(4)
    geometry=dataset._geometry(2,generator)
    a=dataset._landmarks(geometry,torch.Generator().manual_seed(7))
    b=dataset._landmarks(geometry,torch.Generator().manual_seed(7))
    torch.testing.assert_close(a,b)


def test_stream_is_reproducible(dataset):
    first=next(dataset.stream(4,seed=11))
    second=next(dataset.stream(4,seed=11))
    torch.testing.assert_close(first["x0"],second["x0"])
    torch.testing.assert_close(first["condition"],second["condition"])
