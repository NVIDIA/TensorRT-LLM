import pytest
import torch
from torch.export import Dim

from tensorrt_llm._torch.auto_deploy.custom_ops.linear.swiglu import *  # noqa
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.transform.optimizer import InferenceOptimizer
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op


class SwiGLUMLP(torch.nn.Module):
    """SwiGLU MLP module: silu(x @ gate.T) * (x @ up.T) @ down.T"""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class SwiGLUMLPWithBias(torch.nn.Module):
    """SwiGLU MLP module with biases."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=True)
        self.up_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=True)
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=True)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class SwiGLUTestModel(torch.nn.Module):
    """Test model with SwiGLU MLP sandwiched between linear layers."""

    def __init__(
        self,
        hidden_size: int = 256,
        intermediate_size: int = 512,
        with_bias: bool = False,
    ):
        super().__init__()
        self.linear1 = torch.nn.Linear(hidden_size, hidden_size, device="cuda", dtype=torch.float16)
        if with_bias:
            self.mlp = SwiGLUMLPWithBias(hidden_size, intermediate_size)
        else:
            self.mlp = SwiGLUMLP(hidden_size, intermediate_size)
        self.mlp = self.mlp.to(device="cuda", dtype=torch.float16)
        self.linear2 = torch.nn.Linear(hidden_size, hidden_size, device="cuda", dtype=torch.float16)

    def forward(self, x):
        x = self.linear1(x)
        x = self.mlp(x)
        x = self.linear2(x)
        return x


class SwiGLUTestModelMultipleMLP(torch.nn.Module):
    """Test model with multiple SwiGLU MLPs to test multiple pattern matches."""

    def __init__(
        self,
        hidden_size: int = 256,
        intermediate_size: int = 512,
        num_layers: int = 2,
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                torch.nn.ModuleDict(
                    {
                        "linear": torch.nn.Linear(
                            hidden_size, hidden_size, device="cuda", dtype=torch.float16
                        ),
                        "mlp": SwiGLUMLP(hidden_size, intermediate_size).to(
                            device="cuda", dtype=torch.float16
                        ),
                    }
                )
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer["linear"](x)
            x = layer["mlp"](x)
        return x


def _run_fusion_test(model, expected_op, expected_num_matches=1):
    """Run the SwiGLU fusion test.

    Args:
        model: The test model to transform.
        expected_op: The expected fused op to find in the transformed graph.
        expected_num_matches: Expected number of fused ops.
    """
    x = torch.randn(2, 256, device="cuda", dtype=torch.float16)
    dynamic_shapes = {0: Dim.DYNAMIC}
    gm = torch_export_to_gm(model, args=(x,), dynamic_shapes=(dynamic_shapes,), clone=True)

    # Apply transforms
    gm_transformed = InferenceOptimizer(
        None,
        {
            "match_swiglu_pattern": {
                "stage": "pattern_matcher",
            },
            "fuse_swiglu": {
                "stage": "post_load_fusion",
                "enabled": True,
            },
        },
    )(None, gm)

    # Move to CUDA if needed
    gm_transformed = gm_transformed.to("cuda")

    # Check that the expected op is present
    count = sum(1 for n in gm_transformed.graph.nodes if is_op(n, expected_op))
    assert count == expected_num_matches, (
        f"Expected {expected_num_matches} {expected_op} ops, got {count}"
    )

    # Verify numerical correctness
    y_transformed = gm_transformed(x)
    y_model = model(x)
    torch.testing.assert_close(y_transformed, y_model, atol=1e-2, rtol=1e-2)

    # Test with a different batch size
    new_input = torch.randn(4, 256, device="cuda", dtype=torch.float16)
    y_transformed_2 = gm_transformed(new_input)
    y_model_2 = model(new_input)
    torch.testing.assert_close(y_transformed_2, y_model_2, atol=1e-2, rtol=1e-2)


def test_swiglu_fusion_basic():
    """Test basic SwiGLU fusion without biases."""
    model = SwiGLUTestModel(with_bias=False)
    _run_fusion_test(model, torch.ops.auto_deploy.fused_swiglu_mlp.default)


def test_swiglu_fusion_with_bias():
    """Test SwiGLU fusion with biases."""
    model = SwiGLUTestModel(with_bias=True)
    _run_fusion_test(model, torch.ops.auto_deploy.fused_swiglu_mlp.default)


@pytest.mark.parametrize("num_layers", [2, 3])
def test_swiglu_fusion_multiple_layers(num_layers):
    """Test that multiple SwiGLU patterns are fused correctly."""
    model = SwiGLUTestModelMultipleMLP(num_layers=num_layers)
    _run_fusion_test(
        model, torch.ops.auto_deploy.fused_swiglu_mlp.default, expected_num_matches=num_layers
    )


def test_swiglu_pattern_match_only():
    """Test pattern matching stage only (without fusion)."""
    model = SwiGLUTestModel()
    x = torch.randn(2, 256, device="cuda", dtype=torch.float16)
    dynamic_shapes = {0: Dim.DYNAMIC}
    gm = torch_export_to_gm(model, args=(x,), dynamic_shapes=(dynamic_shapes,), clone=True)

    # Only run pattern matching, not fusion
    gm_matched = InferenceOptimizer(
        None,
        {
            "match_swiglu_pattern": {
                "stage": "pattern_matcher",
            },
        },
    )(None, gm)

    # Check that the intermediate op is present
    has_swiglu_op = any(
        is_op(n, torch.ops.auto_deploy.torch_swiglu_mlp.default) for n in gm_matched.graph.nodes
    )
    assert has_swiglu_op, "Pattern matcher should produce torch_swiglu_mlp op"

    # Verify numerical correctness
    y_matched = gm_matched(x)
    y_model = model(x)
    torch.testing.assert_close(y_matched, y_model, atol=1e-3, rtol=1e-3)
