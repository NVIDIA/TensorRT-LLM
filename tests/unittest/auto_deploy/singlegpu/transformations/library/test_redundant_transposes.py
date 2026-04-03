"""Tests for elimination of redundant transpose operations."""

import pytest
import torch
import torch.nn as nn
from _graph_test_helpers import run_test_transformed_gm
from torch.fx import GraphModule

from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.transform.library.eliminate_redundant_transposes import (
    _is_contiguous_op,
    _is_transpose_op,
)
from tensorrt_llm._torch.auto_deploy.transform.optimizer import InferenceOptimizer


class RedundantTransposeModel(nn.Module):
    """Model with redundant transpose operations (same dimensions)."""

    def __init__(self):
        super().__init__()
        self.expected_remaining_transposes = 0
        self.expected_remaining_contiguous = 0

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
        self.expected_remaining_contiguous = 0

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
        self.expected_remaining_contiguous = 0

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
        self.expected_remaining_contiguous = 0

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
        self.expected_remaining_contiguous = 0

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


class ContiguousBetweenTransposesModel(nn.Module):
    """Model with contiguous() call between redundant transpose operations."""

    def __init__(self):
        super().__init__()
        self.expected_remaining_transposes = 0
        self.expected_remaining_contiguous = 1

    def forward(self, x):
        # Apply transpose -> contiguous -> transpose pattern
        # This should be replaced with a single contiguous call
        x = x.transpose(1, 2)
        x = x.contiguous()
        x = x.transpose(1, 2)
        return x


class ContiguousBeforeTransposesModel(nn.Module):
    """Model with contiguous() call before redundant transpose operations."""

    def __init__(self):
        super().__init__()
        self.expected_remaining_transposes = 0
        self.expected_remaining_contiguous = 0

    def forward(self, x):
        # Apply contiguous -> transpose -> transpose pattern
        # This should be replaced with a single contiguous call
        x = x.contiguous()  # torch.export will eliminate this one ...
        x = x.transpose(1, 2)
        x = x.transpose(1, 2)
        return x


class ContiguousAfterTransposesModel(nn.Module):
    """Model with contiguous() call after redundant transpose operations."""

    def __init__(self):
        super().__init__()
        self.expected_remaining_transposes = 0
        self.expected_remaining_contiguous = 0

    def forward(self, x):
        # Apply transpose -> transpose -> contiguous pattern
        # This should be replaced with a single contiguous call
        x = x.transpose(1, 2)
        x = x.transpose(1, 2)
        x = x.contiguous()  # torch.export will eliminate this one ...
        return x


class ContiguousManyTransposesModel(nn.Module):
    """Model with contiguous() call after redundant transpose operations."""

    def __init__(self):
        super().__init__()
        self.expected_remaining_transposes = 0
        self.expected_remaining_contiguous = 2

    def forward(self, x):
        # Apply transpose -> transpose -> contiguous pattern
        # This should be replaced with a single contiguous call
        x = x.contiguous()  # torch.export will eliminate this one ...
        x = x.transpose(1, 2)
        x = x.contiguous()
        x = x.transpose(1, 2)
        x = x.contiguous()
        return x


def count_transpose_ops(gm: GraphModule) -> int:
    """Count the number of transpose operations in the graph."""
    return sum(1 for node in gm.graph.nodes if _is_transpose_op(node))


def count_contiguous_ops(gm: GraphModule) -> int:
    """Count the number of contiguous operations in the graph."""
    return sum(1 for node in gm.graph.nodes if _is_contiguous_op(node))


def check_transpose_and_contiguous_count(
    gm: GraphModule, expected_transpose_count: int, expected_contiguous_count: int
) -> bool:
    """Check if the number of transpose and contiguous operations match the expected counts."""
    actual_transpose_count = count_transpose_ops(gm)
    actual_contiguous_count = count_contiguous_ops(gm)
    print(
        f"actual_transpose_count: {actual_transpose_count}, expected_transpose_count: {expected_transpose_count}"
    )
    print(
        f"actual_contiguous_count: {actual_contiguous_count}, expected_contiguous_count: {expected_contiguous_count}"
    )
    return (
        actual_transpose_count == expected_transpose_count
        and actual_contiguous_count == expected_contiguous_count
    )


@pytest.mark.parametrize(
    "model_class",
    [
        RedundantTransposeModel,
        NonRedundantTransposeModel,
        MixedTransposeModel,
        InterleavedTransposeModel,
        MultipleRedundantTransposeModel,
        ContiguousBetweenTransposesModel,
        ContiguousBeforeTransposesModel,
        ContiguousAfterTransposesModel,
        ContiguousManyTransposesModel,
    ],
)
@torch.inference_mode()
def test_eliminate_redundant_transposes_with_contiguous(model_class):
    """Test elimination of redundant transpose operations with contiguous calls."""
    # Setup model and input
    model = model_class().cuda()
    x = torch.randn(2, 3, 4).cuda()
    gm = torch_export_to_gm(model, args=(x,), clone=True)
    gm_transformed = InferenceOptimizer(
        None,
        {
            "eliminate_redundant_transposes": {
                "stage": "pattern_matcher",
            },
        },
    )(None, gm)

    # Create a check function for this specific model
    expected_transpose_count = model.expected_remaining_transposes
    expected_contiguous_count = model.expected_remaining_contiguous

    # Run the test using the helper
    run_test_transformed_gm(
        model=model,
        x=x,
        gm_transformed=gm_transformed,
        check_transformed_graph=lambda gm: check_transpose_and_contiguous_count(
            gm, expected_transpose_count, expected_contiguous_count
        ),
        _get_expected_num_params=lambda num_p: num_p,  # Parameter count shouldn't change
        test_load_hook=False,  # Our transformation doesn't affect parameters, no need for loading tests
    )
