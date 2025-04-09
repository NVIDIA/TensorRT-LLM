"""Tests for elimination of redundant transpose operations."""

import pytest
import torch
import torch.nn as nn
from _graph_test_helpers import run_test
from torch.fx import GraphModule

from tensorrt_llm._torch.auto_deploy.transformations.library.eliminate_redundant_transposes import (
    _is_transpose_op,
    eliminate_redundant_transposes,
)


class RedundantTransposeModel(nn.Module):
    """Model with redundant transpose operations (same dimensions)."""

    def __init__(self):
        super().__init__()
        self.expected_remaining_transposes = 0

    def forward(self, x):
        # Apply two consecutive transpose operations with the same dimensions
        # These should cancel each other out
        x = x.transpose(1, 2)
        x = x.transpose(1, 2)
        return x


class NonRedundantTransposeModel(nn.Module):
    """Model with non-redundant transpose operations (different dimensions)."""

    def __init__(self):
        super().__init__()
        self.expected_remaining_transposes = 2

    def forward(self, x):
        # Apply two transpose operations with different dimensions
        # These should not be eliminated
        x = x.transpose(1, 2)
        x = x.transpose(0, 2)
        return x


class MixedTransposeModel(nn.Module):
    """Model with both redundant and non-redundant transpose operations."""

    def __init__(self):
        super().__init__()
        self.expected_remaining_transposes = 1

    def forward(self, x):
        # Apply a mix of transpose operations
        x = x.transpose(1, 2)  # First transpose
        y = x.transpose(1, 2)  # Redundant with first (should be eliminated)
        z = y.transpose(0, 1)  # Non-redundant (should remain)
        return z


class InterleavedTransposeModel(nn.Module):
    """Model with interleaved operations between redundant transposes."""

    def __init__(self):
        super().__init__()
        self.expected_remaining_transposes = 2

    def forward(self, x):
        # Apply transpose, then another operation, then another transpose
        x = x.transpose(1, 2)
        x = torch.relu(x)  # Some operation in between
        x = x.transpose(1, 2)  # Should not be eliminated due to operation in between
        return x


class MultipleRedundantTransposeModel(nn.Module):
    """Model with multiple pairs of redundant transpose operations."""

    def __init__(self):
        super().__init__()
        self.expected_remaining_transposes = 0

    def forward(self, x):
        # First pair of redundant transposes
        x = x.transpose(1, 2)
        x = x.transpose(1, 2)

        # Some computation
        x = torch.relu(x)

        # Second pair of redundant transposes
        x = x.transpose(0, 1)
        x = x.transpose(0, 1)
        return x


def count_transpose_ops(gm: GraphModule) -> int:
    """Count the number of transpose operations in the graph."""
    return sum(1 for node in gm.graph.nodes if _is_transpose_op(node))


def check_transpose_count(gm: GraphModule, expected_count: int) -> bool:
    """Check if the number of transpose operations matches the expected count."""
    actual_count = count_transpose_ops(gm)
    return actual_count == expected_count


@pytest.mark.parametrize(
    "model_class",
    [
        RedundantTransposeModel,
        NonRedundantTransposeModel,
        MixedTransposeModel,
        InterleavedTransposeModel,
        MultipleRedundantTransposeModel,
    ],
)
@torch.inference_mode()
def test_eliminate_redundant_transposes(model_class):
    """Test elimination of redundant transpose operations using run_test."""
    # Setup model and input
    model = model_class().cuda()
    x = torch.randn(2, 3, 4).cuda()

    # Create a check function for this specific model
    expected_count = model.expected_remaining_transposes

    # Run the test using the helper
    run_test(
        model=model,
        x=x,
        transform=eliminate_redundant_transposes,
        check_transformed_graph=lambda gm: check_transpose_count(gm, expected_count),
        _get_expected_num_params=lambda num_p: num_p,  # Parameter count shouldn't change
        test_load_hook=False,  # Our transformation doesn't affect parameters, no need for loading tests
    )
