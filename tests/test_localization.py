import pytest
import torch

from pvs.eval.localization import (
    CLASS_NAMES,
    GROUP_NAMES,
    GROUPING_3,
    NUM_CLASSES,
    NUM_GROUPS,
    grouping_matrix,
    name,
    to_groups,
)


def test_the_class_table_matches_the_positive_control():
    """These three indices are what SubCell assigns to proteins annotated as nucleoplasm,
    mitochondria and Golgi in tests/fixtures. The table has to agree or one of them is wrong."""
    assert name(26)=="Nucleoplasm"
    assert name(17)=="Mitochondria"
    assert name(11)=="Golgi apparatus"
    assert NUM_CLASSES==31


def test_every_class_maps_somewhere():
    matrix=grouping_matrix()
    assert matrix.shape==(NUM_CLASSES,NUM_GROUPS)
    assert bool((matrix.sum(1)==1).all()),"a class must land in exactly one group"


def test_grouping_conserves_probability_mass():
    probabilities=torch.softmax(torch.randn(16,NUM_CLASSES),-1)
    torch.testing.assert_close(to_groups(probabilities).sum(-1),torch.ones(16),rtol=0,atol=1e-6)


def test_grouping_sends_related_classes_together():
    grouped=lambda label:GROUP_NAMES[int(to_groups(
        torch.nn.functional.one_hot(torch.tensor(CLASS_NAMES.index(label)),NUM_CLASSES).float()
    ).argmax())]
    assert grouped("Nucleoplasm")==grouped("Nuclear speckles")=="Nucleus"
    assert grouped("Golgi apparatus")==grouped("Lysosomes")=="Endomembrane system"
    assert grouped("Nucleoli")=="Nucleoli"
    assert grouped("Plasma membrane")==grouped("Cell Junctions")=="Plasma membrane"


def test_ungrouped_classes_are_kept_not_dropped():
    """Eight rare mitotic and aggregate annotations have no upstream group. Dropping them
    would silently remove probability mass and bias every distribution."""
    assert "Other" in GROUP_NAMES
    unmapped=[n for n in CLASS_NAMES if n not in GROUPING_3]
    assert unmapped
    onehot=torch.nn.functional.one_hot(torch.tensor(CLASS_NAMES.index(unmapped[0])),NUM_CLASSES).float()
    assert GROUP_NAMES[int(to_groups(onehot).argmax())]=="Other"


def test_group_count_is_small_enough_to_measure():
    """Calibration at 75 cells per gene is only measurable at a small class count; 31 puts
    every divergence below its own noise floor."""
    assert NUM_GROUPS<=8


def test_wrong_class_count_is_rejected():
    with pytest.raises(ValueError):
        to_groups(torch.rand(4,20))
