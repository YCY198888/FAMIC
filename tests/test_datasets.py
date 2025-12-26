"""
Unit tests for dataset loading.

TODO: Add comprehensive tests once dataset loading is implemented.
"""

import pytest
from src.datasets import load_dataset, get_dataset_info


def test_get_dataset_info():
    """Test dataset info retrieval."""
    info = get_dataset_info("dataset1")
    assert "num_classes" in info
    assert "input_dim" in info


# TODO: Add more tests:
# - Test dataset loading
# - Test data transformations
# - Test DataLoader creation

