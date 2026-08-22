import pytest
import torch

from pvs.data.synthetic import NUM_COMPARTMENTS,SyntheticCells,SyntheticConfig
from pvs.eval.compartment import CompartmentClassifier,classifier_accuracy,confusion


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA is unavailable")
def test_cpu_dataset_with_cuda_classifier():
    dataset=SyntheticCells(SyntheticConfig(image_size=16,num_proteins=4),seed=0)
    classifier=CompartmentClassifier().cuda().eval()
    accuracy=classifier_accuracy(classifier,dataset,samples=8)
    matrix=confusion(classifier,dataset,samples=8)

    assert 0<=accuracy<=1
    assert matrix.shape==(NUM_COMPARTMENTS,NUM_COMPARTMENTS)
    assert matrix.device.type=="cpu"
