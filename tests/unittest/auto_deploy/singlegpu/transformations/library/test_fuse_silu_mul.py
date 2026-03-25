import pytest
import torch
from torch.export import Dim

import tensorrt_llm._torch.auto_deploy.custom_ops.linear.silu_mul as _silu_mul  # noqa: F401 — registers custom op
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.transform.interface import SharedConfig, TransformRegistry
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op


class SwiGLUMLP(torch.nn.Module):
    """SwiGLU MLP with separate gate and up projections."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class TestModel(torch.nn.Module):
    """Test model with SwiGLU MLP."""

    def __init__(self, hidden_size: int = 256, intermediate_size: int = 512):
        super().__init__()
        self.mlp = SwiGLUMLP(hidden_size, intermediate_size).to(device="cuda", dtype=torch.float16)

    def forward(self, x):
        return self.mlp(x)


class MultiLayerTestModel(torch.nn.Module):
    """Test model with multiple SwiGLU MLP layers."""

    def __init__(self, hidden_size: int = 256, intermediate_size: int = 512, num_layers: int = 2):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                SwiGLUMLP(hidden_size, intermediate_size).to(device="cuda", dtype=torch.float16)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def _count_ops(gm, op):
    return sum(1 for n in gm.graph.nodes if is_op(n, op))


def _export_model(model, batch_size=2, hidden_size=256):
    x = torch.randn(batch_size, hidden_size, device="cuda", dtype=torch.float16)
    dynamic_shapes = {"x": {0: Dim("batch", min=1, max=16)}}
    gm = torch_export_to_gm(model, (x,), dynamic_shapes=dynamic_shapes)
    return gm, x


def _run_transforms(gm, transform_specs):
    """Run transforms directly on a graph module without InferenceOptimizer.

    Args:
        gm: The graph module to transform.
        transform_specs: Dict mapping transform names to config dicts
            (e.g. {"fuse_silu_mul": {"stage": "post_load_fusion", "enabled": True}}).

    Returns:
        The transformed graph module.
    """
    shared_config = SharedConfig(local_rank=0, world_size=1)
    for name, config_kwargs in transform_specs.items():
        config_cls = TransformRegistry.get_config_class(name)
        config = config_cls(**config_kwargs)
        transform = TransformRegistry.get(name)(config)
        gm, _ = transform._apply(gm, cm=None, factory=None, shared_config=shared_config)
    return gm


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fuse_silu_mul_basic():
    """Test that silu+mul is fused after GEMM fusion merges gate+up projections."""
    model = TestModel()
    gm, x = _export_model(model)

    # Step 1: Run GEMM fusion to merge gate+up projections
    gm = _run_transforms(
        gm,
        {
            "fuse_gemms_mixed_children": {"stage": "post_load_fusion", "enabled": True},
        },
    )

    # Verify GEMM fusion happened: should have narrow ops now
    assert _count_ops(gm, torch.narrow) >= 2, "Expected narrow ops after GEMM fusion"
    # silu and mul should still exist
    assert _count_ops(gm, torch.ops.aten.silu.default) >= 1
    assert _count_ops(gm, torch.ops.aten.mul.Tensor) >= 1

    # Step 2: Run silu+mul fusion
    gm = _run_transforms(
        gm,
        {
            "fuse_silu_mul": {"stage": "post_load_fusion", "enabled": True},
        },
    )

    # Verify silu+mul fusion happened
    assert _count_ops(gm, torch.ops.auto_deploy.silu_and_mul.default) >= 1
    # Original silu and mul ops should be gone
    assert _count_ops(gm, torch.ops.aten.silu.default) == 0
    assert _count_ops(gm, torch.ops.aten.mul.Tensor) == 0

    # Verify correctness
    ref_output = model(x)
    fused_output = gm(x)
    torch.testing.assert_close(fused_output, ref_output, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fuse_silu_mul_multi_layer():
    """Test that silu+mul fusion works across multiple layers."""
    num_layers = 3
    model = MultiLayerTestModel(num_layers=num_layers)
    gm, x = _export_model(model)

    # Run both fusions in sequence
    gm = _run_transforms(
        gm,
        {
            "fuse_gemms_mixed_children": {"stage": "post_load_fusion", "enabled": True},
            "fuse_silu_mul": {"stage": "post_load_fusion", "enabled": True},
        },
    )

    # Should have one silu_and_mul per layer
    assert _count_ops(gm, torch.ops.auto_deploy.silu_and_mul.default) == num_layers
    assert _count_ops(gm, torch.ops.aten.silu.default) == 0
    assert _count_ops(gm, torch.ops.aten.mul.Tensor) == 0

    # Verify correctness
    ref_output = model(x)
    fused_output = gm(x)
    torch.testing.assert_close(fused_output, ref_output, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fuse_silu_mul_disabled():
    """Test that fusion is skipped when disabled."""
    model = TestModel()
    gm, x = _export_model(model)

    gm = _run_transforms(
        gm,
        {
            "fuse_gemms_mixed_children": {"stage": "post_load_fusion", "enabled": True},
            "fuse_silu_mul": {"stage": "post_load_fusion", "enabled": False},
        },
    )

    # silu_and_mul should NOT be present when disabled
    assert _count_ops(gm, torch.ops.auto_deploy.silu_and_mul.default) == 0
    # Original ops should still be there
    assert _count_ops(gm, torch.ops.aten.silu.default) >= 1
