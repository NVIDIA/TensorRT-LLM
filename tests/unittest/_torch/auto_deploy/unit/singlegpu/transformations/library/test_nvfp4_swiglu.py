"""Tests for NVFP4 quantized SwiGLU pattern matching and fusion transforms.

Tests the parallel NVFP4 SwiGLU path:
1. match_nvfp4_swiglu_pattern: Matches torch_fake_quant_nvfp4_linear SwiGLU -> torch_nvfp4_swiglu_mlp
2. fuse_nvfp4_swiglu: Fuses gate+up FP4 weights -> fused_nvfp4_swiglu_mlp
"""

import pytest
import torch
import torch.nn as nn
from _torch_test_utils import fp4_compatible, trtllm_ops_available

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.transform.optimizer import InferenceOptimizer
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op
from tensorrt_llm._torch.auto_deploy.utils.quantization_utils import fp4_global_scale

_skip_reason = "Requires NVFP4 (Blackwell+) and TRT-LLM ops"
_skip_condition = not (fp4_compatible() and trtllm_ops_available())


class NVFP4SwiGLUMLP(nn.Module):
    """SwiGLU MLP using NVFP4 quantized linear ops.

    Mimics the graph structure produced by quantize_nvfp4_linear_from_config
    applied to a standard SwiGLU MLP: silu(gate(x)) * up(x) -> down(hidden).
    """

    def __init__(self, hidden_size: int = 128, intermediate_size: int = 128):
        super().__init__()
        device = torch.device("cuda")
        scaling_vector_size = 16

        # Create random weights and quantize them to FP4
        gate_weight = (
            torch.randn(intermediate_size, hidden_size, dtype=torch.half, device=device) * 0.05
        )
        up_weight = (
            torch.randn(intermediate_size, hidden_size, dtype=torch.half, device=device) * 0.05
        )
        down_weight = (
            torch.randn(hidden_size, intermediate_size, dtype=torch.half, device=device) * 0.05
        )

        # Quantize gate projection
        s_w_gate = fp4_global_scale(gate_weight)
        gate_fp4, gate_cutlass = torch.ops.trtllm.fp4_quantize(
            gate_weight, s_w_gate, scaling_vector_size, False
        )
        # Use a shared input scale for gate and up (same input x)
        s_in = fp4_global_scale(torch.randn(1, hidden_size, dtype=torch.half, device=device))
        gate_alpha = (1.0 / (s_in * s_w_gate)).to(torch.float32)

        self.register_buffer("gate_weight", gate_fp4)
        self.register_buffer("gate_input_scale", s_in.to(torch.float32))
        self.register_buffer("gate_weight_scale", gate_cutlass)
        self.register_buffer("gate_alpha", gate_alpha)

        # Quantize up projection (same input scale as gate)
        s_w_up = fp4_global_scale(up_weight)
        up_fp4, up_cutlass = torch.ops.trtllm.fp4_quantize(
            up_weight, s_w_up, scaling_vector_size, False
        )
        up_alpha = (1.0 / (s_in * s_w_up)).to(torch.float32)

        self.register_buffer("up_weight", up_fp4)
        self.register_buffer("up_input_scale", s_in.to(torch.float32))
        self.register_buffer("up_weight_scale", up_cutlass)
        self.register_buffer("up_alpha", up_alpha)

        # Quantize down projection (different input: the hidden state)
        s_in_down = fp4_global_scale(
            torch.randn(1, intermediate_size, dtype=torch.half, device=device)
        )
        s_w_down = fp4_global_scale(down_weight)
        down_fp4, down_cutlass = torch.ops.trtllm.fp4_quantize(
            down_weight, s_w_down, scaling_vector_size, False
        )
        down_alpha = (1.0 / (s_in_down * s_w_down)).to(torch.float32)

        self.register_buffer("down_weight", down_fp4)
        self.register_buffer("down_input_scale", s_in_down.to(torch.float32))
        self.register_buffer("down_weight_scale", down_cutlass)
        self.register_buffer("down_alpha", down_alpha)

    def forward(self, x):
        gate_out = torch.ops.auto_deploy.torch_fake_quant_nvfp4_linear(
            x,
            self.gate_weight,
            None,
            [self.gate_input_scale],
            [self.gate_weight_scale, self.gate_alpha],
            [],
            [],
        )
        up_out = torch.ops.auto_deploy.torch_fake_quant_nvfp4_linear(
            x,
            self.up_weight,
            None,
            [self.up_input_scale],
            [self.up_weight_scale, self.up_alpha],
            [],
            [],
        )
        hidden = torch.nn.functional.silu(gate_out) * up_out
        return torch.ops.auto_deploy.torch_fake_quant_nvfp4_linear(
            hidden,
            self.down_weight,
            None,
            [self.down_input_scale],
            [self.down_weight_scale, self.down_alpha],
            [],
            [],
        )


class NVFP4SwiGLUTestModel(nn.Module):
    """Test model wrapping NVFP4 SwiGLU MLP between linear layers."""

    def __init__(self, hidden_size: int = 128, intermediate_size: int = 128):
        super().__init__()
        device = torch.device("cuda")
        self.linear_in = nn.Linear(hidden_size, hidden_size, device=device, dtype=torch.float16)
        self.mlp = NVFP4SwiGLUMLP(hidden_size, intermediate_size)
        self.linear_out = nn.Linear(hidden_size, hidden_size, device=device, dtype=torch.float16)

    def forward(self, x):
        x = self.linear_in(x)
        x = self.mlp(x)
        x = self.linear_out(x)
        return x


class NVFP4SwiGLUMultiLayerModel(nn.Module):
    """Test model with multiple NVFP4 SwiGLU MLP layers."""

    def __init__(
        self,
        hidden_size: int = 128,
        intermediate_size: int = 128,
        num_layers: int = 2,
    ):
        super().__init__()
        device = torch.device("cuda")
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "linear": nn.Linear(
                            hidden_size,
                            hidden_size,
                            device=device,
                            dtype=torch.float16,
                        ),
                        "mlp": NVFP4SwiGLUMLP(hidden_size, intermediate_size),
                    }
                )
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer["linear"](x)
            x = layer["mlp"](x)
        return x


# -- Test helpers --------------------------------------------------------------


def _count_ops(gm, op):
    """Count how many nodes in the graph match the given op."""
    return sum(1 for n in gm.graph.nodes if is_op(n, op))


def _has_no_fake_quant_nvfp4(gm):
    """Verify no torch_fake_quant_nvfp4_linear ops remain."""
    return _count_ops(gm, torch.ops.auto_deploy.torch_fake_quant_nvfp4_linear) == 0


# -- Tests ---------------------------------------------------------------------


@pytest.mark.skipif(_skip_condition, reason=_skip_reason)
def test_nvfp4_swiglu_pattern_match_only():
    """Test that match_nvfp4_swiglu_pattern produces torch_nvfp4_swiglu_mlp op."""
    torch.manual_seed(0)
    model = NVFP4SwiGLUMLP().to("cuda")
    x = torch.randn(2, 128, device="cuda", dtype=torch.float16)

    gm = torch_export_to_gm(model, args=(x,), clone=True)

    # Verify the graph has torch_fake_quant_nvfp4_linear ops before transform
    assert _count_ops(gm, torch.ops.auto_deploy.torch_fake_quant_nvfp4_linear) == 3, (
        "Expected 3 torch_fake_quant_nvfp4_linear ops (gate, up, down) before transform"
    )

    # Apply only pattern matching
    gm_matched = InferenceOptimizer(
        None,
        {
            "match_nvfp4_swiglu_pattern": {
                "stage": "pattern_matcher",
            },
        },
    )(None, gm)

    # Check the intermediate op is present
    nvfp4_swiglu_count = _count_ops(
        gm_matched, torch.ops.auto_deploy.torch_nvfp4_swiglu_mlp.default
    )
    assert nvfp4_swiglu_count == 1, (
        f"Expected 1 torch_nvfp4_swiglu_mlp op, got {nvfp4_swiglu_count}"
    )

    # All 3 fake_quant_nvfp4 ops should be consumed
    assert _has_no_fake_quant_nvfp4(gm_matched), (
        "torch_fake_quant_nvfp4_linear ops should be consumed by pattern matcher"
    )

    # Verify numerical correctness
    gm_matched = gm_matched.to("cuda")
    y_matched = gm_matched(x)
    y_model = model(x)
    torch.testing.assert_close(y_matched, y_model, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(_skip_condition, reason=_skip_reason)
def test_nvfp4_swiglu_full_fusion():
    """Test full pipeline: pattern match -> fuse -> fused_nvfp4_swiglu_mlp."""
    torch.manual_seed(0)
    model = NVFP4SwiGLUTestModel().to("cuda")
    x = torch.randn(2, 128, device="cuda", dtype=torch.float16)

    gm = torch_export_to_gm(model, args=(x,), clone=True)

    # Apply pattern matching + fusion
    gm_fused = InferenceOptimizer(
        None,
        {
            "match_nvfp4_swiglu_pattern": {
                "stage": "pattern_matcher",
            },
            "fuse_nvfp4_swiglu": {
                "stage": "post_load_fusion",
            },
        },
    )(None, gm)

    gm_fused = gm_fused.to("cuda")

    # Check the fused op is present
    fused_count = _count_ops(gm_fused, torch.ops.auto_deploy.fused_nvfp4_swiglu_mlp.default)
    assert fused_count == 1, f"Expected 1 fused_nvfp4_swiglu_mlp op, got {fused_count}"

    # No intermediate or unfused ops should remain
    assert _count_ops(gm_fused, torch.ops.auto_deploy.torch_nvfp4_swiglu_mlp.default) == 0, (
        "Intermediate torch_nvfp4_swiglu_mlp should be replaced by fused version"
    )
    assert _has_no_fake_quant_nvfp4(gm_fused), (
        "No torch_fake_quant_nvfp4_linear ops should remain after fusion"
    )

    # Verify numerical correctness (fused uses TRT-LLM kernel, allow wider tolerance)
    y_fused = gm_fused(x)
    y_model = model(x)
    torch.testing.assert_close(y_fused, y_model, atol=0.15, rtol=0.05)

    # Test with a different batch size to verify dynamic shapes work
    x2 = torch.randn(4, 128, device="cuda", dtype=torch.float16)
    y_fused_2 = gm_fused(x2)
    y_model_2 = model(x2)
    torch.testing.assert_close(y_fused_2, y_model_2, atol=0.15, rtol=0.05)


@pytest.mark.skipif(_skip_condition, reason=_skip_reason)
@pytest.mark.parametrize("num_layers", [2, 3])
def test_nvfp4_swiglu_fusion_multiple_layers(num_layers):
    """Test that multiple NVFP4 SwiGLU patterns are fused correctly."""
    torch.manual_seed(0)
    model = NVFP4SwiGLUMultiLayerModel(num_layers=num_layers).to("cuda")
    x = torch.randn(2, 128, device="cuda", dtype=torch.float16)

    gm = torch_export_to_gm(model, args=(x,), clone=True)

    # Apply pattern matching + fusion
    gm_fused = InferenceOptimizer(
        None,
        {
            "match_nvfp4_swiglu_pattern": {
                "stage": "pattern_matcher",
            },
            "fuse_nvfp4_swiglu": {
                "stage": "post_load_fusion",
            },
        },
    )(None, gm)

    gm_fused = gm_fused.to("cuda")

    # Check that all layers are fused
    fused_count = _count_ops(gm_fused, torch.ops.auto_deploy.fused_nvfp4_swiglu_mlp.default)
    assert fused_count == num_layers, (
        f"Expected {num_layers} fused_nvfp4_swiglu_mlp ops, got {fused_count}"
    )

    # Verify numerical correctness
    y_fused = gm_fused(x)
    y_model = model(x)
    torch.testing.assert_close(y_fused, y_model, atol=0.2, rtol=0.1)


@pytest.mark.skipif(_skip_condition, reason=_skip_reason)
def test_nvfp4_swiglu_does_not_match_non_swiglu():
    """Test that the NVFP4 SwiGLU matcher does not match non-SwiGLU NVFP4 linears."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    hidden_size = 128

    # Model with two sequential NVFP4 linears + relu (NOT a SwiGLU pattern)
    class NonSwiGLUModel(nn.Module):
        def __init__(self):
            super().__init__()
            w1 = torch.randn(hidden_size, hidden_size, dtype=torch.half, device=device) * 0.05
            w2 = torch.randn(hidden_size, hidden_size, dtype=torch.half, device=device) * 0.05

            s_in = fp4_global_scale(torch.randn(1, hidden_size, dtype=torch.half, device=device))
            s_w1 = fp4_global_scale(w1)
            s_w2 = fp4_global_scale(w2)

            w1_fp4, w1_cutlass = torch.ops.trtllm.fp4_quantize(w1, s_w1, 16, False)
            w2_fp4, w2_cutlass = torch.ops.trtllm.fp4_quantize(w2, s_w2, 16, False)

            self.register_buffer("w1", w1_fp4)
            self.register_buffer("w1_is", s_in.to(torch.float32))
            self.register_buffer("w1_ws", w1_cutlass)
            self.register_buffer("w1_a", (1.0 / (s_in * s_w1)).to(torch.float32))

            self.register_buffer("w2", w2_fp4)
            self.register_buffer("w2_is", s_in.to(torch.float32))
            self.register_buffer("w2_ws", w2_cutlass)
            self.register_buffer("w2_a", (1.0 / (s_in * s_w2)).to(torch.float32))

        def forward(self, x):
            # Sequential linears without SwiGLU pattern
            y = torch.ops.auto_deploy.torch_fake_quant_nvfp4_linear(
                x, self.w1, None, [self.w1_is], [self.w1_ws, self.w1_a], [], []
            )
            y = torch.nn.functional.relu(y)
            return torch.ops.auto_deploy.torch_fake_quant_nvfp4_linear(
                y, self.w2, None, [self.w2_is], [self.w2_ws, self.w2_a], [], []
            )

    model = NonSwiGLUModel().to("cuda")
    x = torch.randn(2, hidden_size, device="cuda", dtype=torch.float16)

    gm = torch_export_to_gm(model, args=(x,), clone=True)

    gm_result = InferenceOptimizer(
        None,
        {
            "match_nvfp4_swiglu_pattern": {
                "stage": "pattern_matcher",
            },
        },
    )(None, gm)

    # No SwiGLU ops should be found
    assert _count_ops(gm_result, torch.ops.auto_deploy.torch_nvfp4_swiglu_mlp.default) == 0, (
        "Non-SwiGLU NVFP4 pattern should not match"
    )

    # Original NVFP4 linear ops should still be present
    assert _count_ops(gm_result, torch.ops.auto_deploy.torch_fake_quant_nvfp4_linear) == 2, (
        "Original NVFP4 linear ops should be unchanged"
    )
