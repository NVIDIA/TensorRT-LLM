# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for FineGrained FP8 quantized SwiGLU pattern matching and fusion transforms.

Tests the FineGrained FP8 SwiGLU path:
1. match_finegrained_fp8_swiglu_pattern: Matches torch_fake_quant_finegrained_fp8_linear
   SwiGLU -> torch_finegrained_fp8_swiglu_mlp
2. fuse_finegrained_fp8_swiglu: Fuses gate+up FP8 weights -> fused_finegrained_fp8_swiglu_mlp
"""

import pytest
import torch
import torch.nn as nn
from _torch_test_utils import fp8_compatible, trtllm_ops_available
from torch.export import Dim

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.transform.optimizer import InferenceOptimizer
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op

_skip_reason = "Requires FP8 (Hopper+) and TRT-LLM ops"
_skip_condition = not (fp8_compatible() and trtllm_ops_available())

# Block size for finegrained FP8 quantization (128x128)
_BLOCK_SIZE = 128


def _quantize_fp8_block(weight: torch.Tensor) -> tuple:
    """Quantize a weight tensor to FP8 with per-128x128-block scales.

    Args:
        weight: Float weight tensor of shape [N, K] where N and K are multiples of 128.

    Returns:
        (weight_fp8, weight_scale_inv) where:
        - weight_fp8: [N, K] float8_e4m3fn
        - weight_scale_inv: [N/128, K/128] float32 per-block scale
    """
    N, K = weight.shape
    assert N % _BLOCK_SIZE == 0 and K % _BLOCK_SIZE == 0, (
        f"Dimensions must be multiples of {_BLOCK_SIZE}, got ({N}, {K})"
    )
    FP8_MAX = torch.finfo(torch.float8_e4m3fn).max

    # Reshape into blocks of [128, 128]
    blocks = weight.reshape(N // _BLOCK_SIZE, _BLOCK_SIZE, K // _BLOCK_SIZE, _BLOCK_SIZE)
    blocks = blocks.permute(0, 2, 1, 3)  # [N/128, K/128, 128, 128]

    # Compute per-block scale: amax / FP8_MAX
    block_amax = blocks.abs().amax(dim=(-2, -1))  # [N/128, K/128]
    scale = (block_amax / FP8_MAX).clamp(min=1e-12).to(torch.float32)

    # Quantize: divide by scale, clamp, cast to fp8
    scale_expanded = scale[:, :, None, None]  # [N/128, K/128, 1, 1]
    blocks_scaled = (blocks.float() / scale_expanded).clamp(-FP8_MAX, FP8_MAX)
    blocks_fp8 = blocks_scaled.to(torch.float8_e4m3fn)

    # Reshape back to [N, K]
    weight_fp8 = blocks_fp8.permute(0, 2, 1, 3).reshape(N, K)

    return weight_fp8, scale


class FineGrainedFP8SwiGLUMLP(nn.Module):
    """SwiGLU MLP using FineGrained FP8 quantized linear ops.

    Mimics the graph structure produced by quantize_finegrained_fp8_linear_from_config
    applied to a standard SwiGLU MLP: silu(gate(x)) * up(x) -> down(hidden).
    """

    def __init__(self, hidden_size: int = 256, intermediate_size: int = 256):
        super().__init__()
        device = torch.device("cuda")

        # Create random weights and quantize them to FP8
        gate_weight = (
            torch.randn(intermediate_size, hidden_size, dtype=torch.bfloat16, device=device) * 0.05
        )
        up_weight = (
            torch.randn(intermediate_size, hidden_size, dtype=torch.bfloat16, device=device) * 0.05
        )
        down_weight = (
            torch.randn(hidden_size, intermediate_size, dtype=torch.bfloat16, device=device) * 0.05
        )

        # Quantize each projection
        gate_fp8, gate_scale = _quantize_fp8_block(gate_weight)
        up_fp8, up_scale = _quantize_fp8_block(up_weight)
        down_fp8, down_scale = _quantize_fp8_block(down_weight)

        self.register_buffer("gate_weight", gate_fp8)
        self.register_buffer("gate_weight_scale", gate_scale)
        self.register_buffer("up_weight", up_fp8)
        self.register_buffer("up_weight_scale", up_scale)
        self.register_buffer("down_weight", down_fp8)
        self.register_buffer("down_weight_scale", down_scale)

    def forward(self, x):
        gate_out = torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear(
            x,
            self.gate_weight,
            None,
            [],
            [self.gate_weight_scale],
            [],
            [],
        )
        up_out = torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear(
            x,
            self.up_weight,
            None,
            [],
            [self.up_weight_scale],
            [],
            [],
        )
        hidden = torch.nn.functional.silu(gate_out) * up_out
        return torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear(
            hidden,
            self.down_weight,
            None,
            [],
            [self.down_weight_scale],
            [],
            [],
        )


class FineGrainedFP8SwiGLUTestModel(nn.Module):
    """Test model wrapping FineGrained FP8 SwiGLU MLP between linear layers."""

    def __init__(self, hidden_size: int = 256, intermediate_size: int = 256):
        super().__init__()
        device = torch.device("cuda")
        self.linear_in = nn.Linear(hidden_size, hidden_size, device=device, dtype=torch.bfloat16)
        self.mlp = FineGrainedFP8SwiGLUMLP(hidden_size, intermediate_size)
        self.linear_out = nn.Linear(hidden_size, hidden_size, device=device, dtype=torch.bfloat16)

    def forward(self, x):
        x = self.linear_in(x)
        x = self.mlp(x)
        x = self.linear_out(x)
        return x


class FineGrainedFP8SwiGLUMultiLayerModel(nn.Module):
    """Test model with multiple FineGrained FP8 SwiGLU MLP layers."""

    def __init__(
        self,
        hidden_size: int = 256,
        intermediate_size: int = 256,
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
                            dtype=torch.bfloat16,
                        ),
                        "mlp": FineGrainedFP8SwiGLUMLP(hidden_size, intermediate_size),
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


def _has_no_fake_quant_finegrained_fp8(gm):
    """Verify no torch_fake_quant_finegrained_fp8_linear ops remain."""
    return _count_ops(gm, torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear) == 0


# -- Tests ---------------------------------------------------------------------


@pytest.mark.skipif(_skip_condition, reason=_skip_reason)
def test_finegrained_fp8_swiglu_pattern_match_only():
    """Test that match_finegrained_fp8_swiglu_pattern produces torch_finegrained_fp8_swiglu_mlp."""
    torch.manual_seed(0)
    model = FineGrainedFP8SwiGLUMLP().to("cuda")
    x = torch.randn(2, 256, device="cuda", dtype=torch.bfloat16)

    gm = torch_export_to_gm(model, args=(x,), clone=True)

    # Verify the graph has torch_fake_quant_finegrained_fp8_linear ops before transform
    assert _count_ops(gm, torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear) == 3, (
        "Expected 3 torch_fake_quant_finegrained_fp8_linear ops (gate, up, down) before transform"
    )

    # Apply only pattern matching
    gm_matched = InferenceOptimizer(
        None,
        {
            "match_finegrained_fp8_swiglu_pattern": {
                "stage": "pattern_matcher",
            },
        },
    )(None, gm)

    # Check the intermediate op is present
    fp8_swiglu_count = _count_ops(
        gm_matched, torch.ops.auto_deploy.torch_finegrained_fp8_swiglu_mlp.default
    )
    assert fp8_swiglu_count == 1, (
        f"Expected 1 torch_finegrained_fp8_swiglu_mlp op, got {fp8_swiglu_count}"
    )

    # All 3 fake_quant_finegrained_fp8 ops should be consumed
    assert _has_no_fake_quant_finegrained_fp8(gm_matched), (
        "torch_fake_quant_finegrained_fp8_linear ops should be consumed by pattern matcher"
    )

    # Verify numerical correctness
    gm_matched = gm_matched.to("cuda")
    y_matched = gm_matched(x)
    y_model = model(x)
    torch.testing.assert_close(y_matched, y_model, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(_skip_condition, reason=_skip_reason)
def test_finegrained_fp8_swiglu_full_fusion():
    """Test full pipeline: pattern match -> fuse -> fused_finegrained_fp8_swiglu_mlp."""
    torch.manual_seed(0)
    model = FineGrainedFP8SwiGLUTestModel().to("cuda")
    x = torch.randn(2, 256, device="cuda", dtype=torch.bfloat16)

    gm = torch_export_to_gm(model, args=(x,), clone=True, dynamic_shapes=({0: Dim.DYNAMIC},))

    # Apply pattern matching + fusion
    gm_fused = InferenceOptimizer(
        None,
        {
            "match_finegrained_fp8_swiglu_pattern": {
                "stage": "pattern_matcher",
            },
            "fuse_finegrained_fp8_swiglu": {
                "stage": "post_load_fusion",
            },
        },
    )(None, gm)

    gm_fused = gm_fused.to("cuda")

    # Check the fused op is present
    fused_count = _count_ops(
        gm_fused, torch.ops.auto_deploy.fused_finegrained_fp8_swiglu_mlp.default
    )
    assert fused_count == 1, f"Expected 1 fused_finegrained_fp8_swiglu_mlp op, got {fused_count}"

    # No intermediate or unfused ops should remain
    assert (
        _count_ops(gm_fused, torch.ops.auto_deploy.torch_finegrained_fp8_swiglu_mlp.default) == 0
    ), "Intermediate torch_finegrained_fp8_swiglu_mlp should be replaced by fused version"
    assert _has_no_fake_quant_finegrained_fp8(gm_fused), (
        "No torch_fake_quant_finegrained_fp8_linear ops should remain after fusion"
    )

    # Verify numerical correctness (fused uses TRT-LLM kernel, allow wider tolerance)
    y_fused = gm_fused(x)
    y_model = model(x)
    torch.testing.assert_close(y_fused, y_model, atol=0.15, rtol=0.05)

    # Test with a different batch size to verify dynamic shapes work
    x2 = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16)
    y_fused_2 = gm_fused(x2)
    y_model_2 = model(x2)
    torch.testing.assert_close(y_fused_2, y_model_2, atol=0.15, rtol=0.05)


@pytest.mark.skipif(_skip_condition, reason=_skip_reason)
@pytest.mark.parametrize("num_layers", [2, 3])
def test_finegrained_fp8_swiglu_fusion_multiple_layers(num_layers):
    """Test that multiple FineGrained FP8 SwiGLU patterns are fused correctly."""
    torch.manual_seed(0)
    model = FineGrainedFP8SwiGLUMultiLayerModel(num_layers=num_layers).to("cuda")
    x = torch.randn(2, 256, device="cuda", dtype=torch.bfloat16)

    gm = torch_export_to_gm(model, args=(x,), clone=True)

    # Apply pattern matching + fusion
    gm_fused = InferenceOptimizer(
        None,
        {
            "match_finegrained_fp8_swiglu_pattern": {
                "stage": "pattern_matcher",
            },
            "fuse_finegrained_fp8_swiglu": {
                "stage": "post_load_fusion",
            },
        },
    )(None, gm)

    gm_fused = gm_fused.to("cuda")

    # Check that all layers are fused
    fused_count = _count_ops(
        gm_fused, torch.ops.auto_deploy.fused_finegrained_fp8_swiglu_mlp.default
    )
    assert fused_count == num_layers, (
        f"Expected {num_layers} fused_finegrained_fp8_swiglu_mlp ops, got {fused_count}"
    )

    # Verify numerical correctness
    y_fused = gm_fused(x)
    y_model = model(x)
    torch.testing.assert_close(y_fused, y_model, atol=0.2, rtol=0.1)


@pytest.mark.skipif(_skip_condition, reason=_skip_reason)
def test_finegrained_fp8_swiglu_does_not_match_non_swiglu():
    """Test that the FP8 SwiGLU matcher does not match non-SwiGLU FP8 linears."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    hidden_size = 256

    # Model with two sequential FP8 linears + relu (NOT a SwiGLU pattern)
    class NonSwiGLUModel(nn.Module):
        def __init__(self):
            super().__init__()
            w1 = torch.randn(hidden_size, hidden_size, dtype=torch.bfloat16, device=device) * 0.05
            w2 = torch.randn(hidden_size, hidden_size, dtype=torch.bfloat16, device=device) * 0.05

            w1_fp8, w1_scale = _quantize_fp8_block(w1)
            w2_fp8, w2_scale = _quantize_fp8_block(w2)

            self.register_buffer("w1", w1_fp8)
            self.register_buffer("w1_scale", w1_scale)
            self.register_buffer("w2", w2_fp8)
            self.register_buffer("w2_scale", w2_scale)

        def forward(self, x):
            # Sequential linears without SwiGLU pattern
            y = torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear(
                x, self.w1, None, [], [self.w1_scale], [], []
            )
            y = torch.nn.functional.relu(y)
            return torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear(
                y, self.w2, None, [], [self.w2_scale], [], []
            )

    model = NonSwiGLUModel().to("cuda")
    x = torch.randn(2, hidden_size, device="cuda", dtype=torch.bfloat16)

    gm = torch_export_to_gm(model, args=(x,), clone=True)

    gm_result = InferenceOptimizer(
        None,
        {
            "match_finegrained_fp8_swiglu_pattern": {
                "stage": "pattern_matcher",
            },
        },
    )(None, gm)

    # No SwiGLU ops should be found
    assert (
        _count_ops(gm_result, torch.ops.auto_deploy.torch_finegrained_fp8_swiglu_mlp.default) == 0
    ), "Non-SwiGLU FP8 pattern should not match"

    # Original FP8 linear ops should still be present
    assert (
        _count_ops(gm_result, torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear) == 2
    ), "Original FP8 linear ops should be unchanged"
