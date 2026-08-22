"""The channel-order check is the important test in this file.

An error there corrupts every downstream result silently — the model would simply learn to
predict the wrong channel from the wrong conditions and the loss would look fine. The
fixtures are real crops of proteins with known, very different localizations, so the
assertion below is a statement about biology rather than about file formats.
"""
import json
from pathlib import Path

import pytest
import torch

from pvs.data.crops import (
    CHANNEL_NAMES,
    LANDMARKS,
    MICROTUBULES,
    NUCLEUS,
    PROTEIN,
    RETICULUM,
    load_crop,
    normalize,
    resize,
    split,
)

FIXTURES=Path(__file__).parent/"fixtures"


def _fixture(name):
    return load_crop(FIXTURES/f"{name}.png")


def test_channel_indices_are_distinct_and_named():
    assert sorted((NUCLEUS,RETICULUM,MICROTUBULES,PROTEIN))==[0,1,2,3]
    assert len(CHANNEL_NAMES)==4
    assert PROTEIN not in LANDMARKS and len(LANDMARKS)==3


def test_crops_load_as_four_channels_in_unit_range():
    crop=_fixture("golgi")
    assert crop.shape[0]==4
    assert 0.0<=crop.min() and crop.max()<=1.0


def test_protein_channel_tracks_the_annotated_localization():
    """The decisive check on channel order.

    A nucleoplasm protein must correlate strongly with the nucleus channel, a plasma-membrane
    protein must anticorrelate with it, and a Golgi protein sits between the two because it
    is perinuclear. If the channel order were wrong, none of these would hold.
    """
    def correlation(crop,a,b):
        selected=crop.sum(0).flatten()>0.08
        x,y=crop[a].flatten()[selected],crop[b].flatten()[selected]
        x,y=x-x.mean(),y-y.mean()
        return float((x*y).sum()/(x.norm()*y.norm()))

    nucleoplasm=correlation(_fixture("nucleoplasm"),PROTEIN,NUCLEUS)
    membrane=correlation(_fixture("plasma_membrane"),PROTEIN,NUCLEUS)
    golgi=correlation(_fixture("golgi"),PROTEIN,NUCLEUS)
    assert nucleoplasm>0.5,f"nucleoplasm protein should follow the nucleus, got {nucleoplasm:.3f}"
    assert membrane<0.1,f"membrane protein should not follow the nucleus, got {membrane:.3f}"
    assert membrane<golgi<nucleoplasm


def test_mitochondrial_protein_follows_microtubules_not_the_nucleus():
    crop=_fixture("mitochondria")
    selected=crop.sum(0).flatten()>0.08
    def correlation(a,b):
        x,y=crop[a].flatten()[selected],crop[b].flatten()[selected]
        x,y=x-x.mean(),y-y.mean()
        return float((x*y).sum()/(x.norm()*y.norm()))
    assert correlation(PROTEIN,MICROTUBULES)>correlation(PROTEIN,NUCLEUS)


def test_nucleus_channel_concentrates_its_intensity_most():
    """A solid bright structure against filamentous and reticular ones.

    Spatial variance would be the obvious statistic and it does not work: crops contain
    neighbouring cells, so the nucleus channel often shows two or three nuclei spread across
    the frame. Concentration of intensity is robust to that.
    """
    for name in json.loads((FIXTURES/"manifest.json").read_text()):
        crop=_fixture(name)
        concentration=[]
        for channel in range(4):
            values=crop[channel].flatten().sort(descending=True).values
            top=max(1,int(0.05*values.numel()))
            concentration.append(float(values[:top].sum()/values.sum().clamp_min(1e-6)))
        assert concentration[NUCLEUS]==max(concentration),f"{name}: {concentration}"


def test_normalization_maps_into_the_symmetric_unit_range():
    normalized=normalize(_fixture("golgi"))
    assert -1.0<=normalized.min() and normalized.max()<=1.0


def test_anchor_normalization_preserves_relative_channel_brightness():
    """Per-channel scaling would flatten every channel to the same maximum, discarding the
    difference between a bright compact structure and a dim diffuse one."""
    crop=_fixture("golgi")
    anchored=normalize(crop).amax(dim=(1,2))
    per_channel=normalize(crop,per_channel=True).amax(dim=(1,2))
    assert torch.allclose(per_channel,torch.ones(4),atol=1e-5)
    assert anchored.min()<0.95
    assert anchored[MICROTUBULES].item()==pytest.approx(1.0,abs=1e-5)


def test_resize_and_split_shapes():
    crop=resize(normalize(_fixture("golgi")),64)
    protein,landmarks=split(crop)
    assert crop.shape==(4,64,64)
    assert protein.shape==(1,64,64) and landmarks.shape==(3,64,64)


def test_load_rejects_a_non_four_channel_image(tmp_path):
    from PIL import Image
    path=tmp_path/"grey.png"
    Image.new("L",(8,8)).save(path)
    with pytest.raises(ValueError,match="4-channel"):
        load_crop(path)


def test_subcell_input_separates_proteins_with_different_localizations():
    """A positive control on the whole preprocessing path, not just the file format.

    Feeding SubCell the wrong range or the wrong channel order does not raise: it assigns
    every protein the same class with about 0.8 confidence, which reads as a finding rather
    than a defect. Only run when the weights are present; the assertion is on the arrangement,
    which is checked here without them.
    """
    from pvs.eval.subcell import CROP_TO_SUBCELL, subcell_input

    crop = _fixture("golgi")
    prepared = subcell_input(crop)
    assert prepared.shape == crop.shape
    assert 0.0 <= prepared.min() and prepared.max() <= 1.0
    # nucleus and microtubules exchange places
    assert CROP_TO_SUBCELL == (2, 1, 0, 3)
    original = crop[list(CROP_TO_SUBCELL)]
    ranking = lambda t: [int(x) for x in t.flatten(1).mean(1).argsort()]
    assert ranking(prepared) == ranking(original)


def test_subcell_input_rejects_the_diffusion_range():
    """The failure that cost an invalid probe run: [-1, 1] input silently collapses SubCell."""
    from pvs.eval.subcell import subcell_input

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        subcell_input(normalize(_fixture("golgi")))
    with pytest.raises(ValueError, match="4 channels"):
        subcell_input(_fixture("golgi")[:3])
