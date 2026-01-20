"""Test dist_backend configuration for sharding transformations."""

import sys
from pathlib import Path
from typing import Optional

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add parent directory to path for test utilities
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "_utils_test"))

from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.transform.optimizer import InferenceOptimizer
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op


class SimpleMLP(nn.Module):
    """Simple MLP model for testing dist_backend."""

    def __init__(self, in_features=32, hidden_features=64, out_features=32, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.linear2 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x):
        y = F.relu(self.linear1(x))
        return self.linear2(y)


def _create_and_transform_model(
    model: nn.Module,
    dist_backend: Optional[str] = None,
    world_size: int = 2,
):
    """Helper to create model, export to graph, and apply sharding transforms.

    Args:
        model: The model to transform
        dist_backend: The distributed backend ('torch', 'trtllm', 'auto', or None for default)
        world_size: World size for multi-GPU setting

    Returns:
        Transformed GraphModule
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device=device, dtype=torch.float32)
    x = torch.randn(2, 32, device=device, dtype=torch.float32)

    # Export to graph module
    gm = torch_export_to_gm(model, args=(x,), clone=True)

    # Build config
    config = {
        "detect_sharding": {
            "stage": "sharding",
        },
        "sharding_transform_executor": {
            "stage": "sharding",
        },
    }

    # Add dist_backend if specified
    if dist_backend is not None:
        config["detect_sharding"]["dist_backend"] = dist_backend

    # Run optimizer
    optimizer = InferenceOptimizer(None, config)
    optimizer.shared_config.local_rank = 0
    optimizer.shared_config.world_size = world_size
    return optimizer(None, gm)


def _check_dist_ops(gm, expected_backend: str):
    """Helper to check which distributed ops are present in the graph.

    Args:
        gm: GraphModule to check
        expected_backend: Expected backend ('torch', 'trtllm', or 'any')

    Returns:
        tuple: (has_torch_ops, has_trtllm_ops)
    """
    has_torch_all_reduce = any(
        is_op(n, torch.ops.auto_deploy.torch_dist_all_reduce) for n in gm.graph.nodes
    )
    has_trtllm_all_reduce = any(
        is_op(n, torch.ops.auto_deploy.trtllm_dist_all_reduce) for n in gm.graph.nodes
    )
    has_torch_all_gather = any(
        is_op(n, torch.ops.auto_deploy.torch_dist_all_gather) for n in gm.graph.nodes
    )
    has_trtllm_all_gather = any(
        is_op(n, torch.ops.auto_deploy.trtllm_dist_all_gather) for n in gm.graph.nodes
    )

    has_torch_ops = has_torch_all_reduce or has_torch_all_gather
    has_trtllm_ops = has_trtllm_all_reduce or has_trtllm_all_gather

    if expected_backend == "torch":
        assert has_torch_ops, (
            f"Expected torch distributed ops when dist_backend='{expected_backend}'"
        )
        assert not has_trtllm_ops, (
            f"Did not expect trtllm distributed ops when dist_backend='{expected_backend}'"
        )
    elif expected_backend == "trtllm":
        assert has_trtllm_ops, (
            f"Expected trtllm distributed ops when dist_backend='{expected_backend}'"
        )
        assert not has_torch_ops, (
            f"Did not expect torch distributed ops when dist_backend='{expected_backend}'"
        )
    elif expected_backend == "any":
        assert has_torch_ops or has_trtllm_ops, (
            "Expected at least one type of distributed op in graph"
        )

    return has_torch_ops, has_trtllm_ops


@pytest.mark.parametrize(
    "dist_backend,expected",
    [
        ("torch", "torch"),
        ("trtllm", "trtllm"),
    ],
)
def test_dist_backend_explicit(dist_backend, expected):
    """Test that explicit dist_backend forces the correct distributed ops."""
    model = SimpleMLP()
    gm_transformed = _create_and_transform_model(model, dist_backend=dist_backend, world_size=2)
    _check_dist_ops(gm_transformed, expected_backend=expected)


@pytest.mark.parametrize("dist_backend", ["auto", None])
def test_dist_backend_auto_and_default(dist_backend):
    """Test that dist_backend='auto' or omitting it uses auto-selection."""
    model = SimpleMLP()
    gm_transformed = _create_and_transform_model(model, dist_backend=dist_backend, world_size=2)
    _check_dist_ops(gm_transformed, expected_backend="any")


@pytest.mark.parametrize("dist_backend", ["torch", "trtllm"])
def test_dist_backend_all_gather(dist_backend):
    """Test dist_backend with all_gather operations (column sharding with single Linear)."""
    model = nn.Linear(32, 64, bias=False)
    gm_transformed = _create_and_transform_model(model, dist_backend=dist_backend, world_size=2)
    _check_dist_ops(gm_transformed, expected_backend=dist_backend)


if __name__ == "__main__":
    # Run tests directly for debugging
    print("Testing explicit backends...")
    test_dist_backend_explicit("torch", "torch")
    test_dist_backend_explicit("trtllm", "trtllm")
    print("✓ Passed\n")

    print("Testing auto and default backends...")
    test_dist_backend_auto_and_default("auto")
    test_dist_backend_auto_and_default(None)
    print("✓ Passed\n")

    print("Testing all_gather with different backends...")
    test_dist_backend_all_gather("torch")
    test_dist_backend_all_gather("trtllm")
    print("✓ Passed\n")

    print("\nAll tests passed!")
