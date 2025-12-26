"""
Unit tests for FAMIC model.

TODO: Add comprehensive tests once model architecture is finalized.
"""

import torch
import pytest
from src.model import FAMIC


def test_model_initialization():
    """Test that model can be initialized."""
    model = FAMIC(input_dim=512, hidden_dim=256, num_classes=2)
    assert model is not None


def test_model_forward():
    """Test model forward pass."""
    model = FAMIC(input_dim=512, hidden_dim=256, num_classes=2)
    batch_size = 32
    x = torch.randn(batch_size, 512)
    output = model(x)
    
    assert output.shape == (batch_size, 2)


# TODO: Add more tests:
# - Test model loading from weights
# - Test model on actual data
# - Test edge cases

