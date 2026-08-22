import pytest
import torch


@pytest.fixture(autouse=True)
def deterministic_seed():
    """Statistical assertions need a fixed stream, otherwise thresholds flake at 3 sigma."""
    torch.manual_seed(0)
