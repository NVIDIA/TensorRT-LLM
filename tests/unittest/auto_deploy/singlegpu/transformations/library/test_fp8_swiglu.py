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

"""Tests for FP8 quantized SwiGLU pattern matching and fusion transforms."""

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
_FP8_MAX = torch.finfo(torch.float8_e4m3fn).max


def _quantize_fp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a weight tensor to per-tensor FP8."""
    scale = torch.clamp(weight.abs().amax().to(torch.float32) / _FP8_MAX, min=1e-12)
    return (weight / scale).to(torch.float8_e4m3fn), scale


class FP8SwiGLUMLP(nn.Module):
    """SwiGLU MLP using FP8 fake-quant linear ops."""

    def __init__(self, hidden_size: int = 256, intermediate_size: int = 512):
        super().__init__()
        device = torch.device("cuda")

        gate_weight = (
            torch.randn(intermediate_size, hidden_size, dtype=torch.bfloat16, device=device) * 0.05
        )
        up_weight = (
            torch.randn(intermediate_size, hidden_size, dtype=torch.bfloat16, device=device) * 0.05
        )
        down_weight = (
            torch.randn(hidden_size, intermediate_size, dtype=torch.bfloat16, device=device) * 0.05
        )

        gate_fp8, gate_scale = _quantize_fp8(gate_weight)
        up_fp8, up_scale = _quantize_fp8(up_weight)
        down_fp8, down_scale = _quantize_fp8(down_weight)

        self.register_buffer("gate_weight", gate_fp8)
        self.register_buffer("gate_weight_scale", gate_scale)
        self.register_buffer("up_weight", up_fp8)
        self.register_buffer("up_weight_scale", up_scale)
        self.register_buffer("down_weight", down_fp8)
        self.register_buffer("down_weight_scale", down_scale)
        self.register_buffer(
            "gate_input_scale", torch.tensor(1.0, dtype=torch.float32, device=device)
        )
        self.register_buffer(
            "up_input_scale", torch.tensor(1.0, dtype=torch.float32, device=device)
        )
        self.register_buffer(
            "down_input_scale", torch.tensor(1.0, dtype=torch.float32, device=device)
        )

    def forward(self, x):
        gate_out = torch.ops.auto_deploy.torch_fake_quant_fp8_linear.default(
            x,
            self.gate_weight,
            None,
            [self.gate_input_scale],
            [self.gate_weight_scale],
            [],
            [],
        )
        up_out = torch.ops.auto_deploy.torch_fake_quant_fp8_linear.default(
            x,
            self.up_weight,
            None,
            [self.up_input_scale],
            [self.up_weight_scale],
            [],
            [],
        )
        hidden = torch.nn.functional.silu(gate_out) * up_out
        return torch.ops.auto_deploy.torch_fake_quant_fp8_linear.default(
            hidden,
            self.down_weight,
            None,
            [self.down_input_scale],
            [self.down_weight_scale],
            [],
            [],
        )


class FP8SwiGLUTestModel(nn.Module):
    """Test model wrapping FP8 SwiGLU MLP between dense layers."""

    def __init__(self, hidden_size: int = 256, intermediate_size: int = 512):
        super().__init__()
        device = torch.device("cuda")
        self.linear_in = nn.Linear(hidden_size, hidden_size, device=device, dtype=torch.bfloat16)
        self.mlp = FP8SwiGLUMLP(hidden_size, intermediate_size)
        self.linear_out = nn.Linear(hidden_size, hidden_size, device=device, dtype=torch.bfloat16)

    def forward(self, x):
        x = self.linear_in(x)
        x = self.mlp(x)
        x = self.linear_out(x)
        return x


def _count_ops(gm, op):
    return sum(1 for node in gm.graph.nodes if is_op(node, op))


@pytest.mark.skipif(_skip_condition, reason=_skip_reason)
def test_fp8_swiglu_pattern_match_only():
    torch.manual_seed(0)
    model = FP8SwiGLUMLP().to("cuda")
    x = torch.randn(2, 256, device="cuda", dtype=torch.bfloat16)

    gm = torch_export_to_gm(model, args=(x,), clone=True)
    assert _count_ops(gm, torch.ops.auto_deploy.torch_fake_quant_fp8_linear.default) == 3

    gm_matched = InferenceOptimizer(
        None,
        {
            "match_fp8_swiglu_pattern": {
                "stage": "pattern_matcher",
            },
        },
    )(None, gm)

    assert _count_ops(gm_matched, torch.ops.auto_deploy.torch_fp8_swiglu_mlp.default) == 1
    assert _count_ops(gm_matched, torch.ops.auto_deploy.torch_fake_quant_fp8_linear.default) == 0

    gm_matched = gm_matched.to("cuda")
    torch.testing.assert_close(gm_matched(x), model(x), atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(_skip_condition, reason=_skip_reason)
def test_fp8_swiglu_full_fusion():
    torch.manual_seed(0)
    model = FP8SwiGLUTestModel().to("cuda")
    x = torch.randn(2, 256, device="cuda", dtype=torch.bfloat16)

    gm = torch_export_to_gm(model, args=(x,), clone=True, dynamic_shapes=({0: Dim.DYNAMIC},))
    gm_fused = InferenceOptimizer(
        None,
        {
            "match_fp8_swiglu_pattern": {
                "stage": "pattern_matcher",
            },
            "fuse_fp8_swiglu": {
                "stage": "post_load_fusion",
                "enabled": True,
            },
        },
    )(None, gm)
    gm_fused = gm_fused.to("cuda")

    assert _count_ops(gm_fused, torch.ops.auto_deploy.fused_fp8_swiglu_mlp.default) == 1
    assert _count_ops(gm_fused, torch.ops.auto_deploy.torch_fp8_swiglu_mlp.default) == 0
    assert _count_ops(gm_fused, torch.ops.auto_deploy.torch_fake_quant_fp8_linear.default) == 0

    torch.testing.assert_close(gm_fused(x), model(x), atol=0.15, rtol=0.05)

    x2 = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(gm_fused(x2), model(x2), atol=0.15, rtol=0.05)


@pytest.mark.skipif(_skip_condition, reason=_skip_reason)
def test_fp8_swiglu_does_not_match_non_swiglu():
    torch.manual_seed(0)
    device = torch.device("cuda")
    hidden_size = 256

    class NonSwiGLUModel(nn.Module):
        def __init__(self):
            super().__init__()
            w1_fp8, w1_scale = _quantize_fp8(
                torch.randn(hidden_size, hidden_size, dtype=torch.bfloat16, device=device) * 0.05
            )
            w2_fp8, w2_scale = _quantize_fp8(
                torch.randn(hidden_size, hidden_size, dtype=torch.bfloat16, device=device) * 0.05
            )
            self.register_buffer("w1", w1_fp8)
            self.register_buffer("w1_scale", w1_scale)
            self.register_buffer("w2", w2_fp8)
            self.register_buffer("w2_scale", w2_scale)
            self.register_buffer(
                "input_scale", torch.tensor(1.0, dtype=torch.float32, device=device)
            )

        def forward(self, x):
            y = torch.ops.auto_deploy.torch_fake_quant_fp8_linear.default(
                x, self.w1, None, [self.input_scale], [self.w1_scale], [], []
            )
            y = torch.nn.functional.relu(y)
            return torch.ops.auto_deploy.torch_fake_quant_fp8_linear.default(
                y, self.w2, None, [self.input_scale], [self.w2_scale], [], []
            )

    model = NonSwiGLUModel().to("cuda")
    x = torch.randn(2, hidden_size, device="cuda", dtype=torch.bfloat16)
    gm = torch_export_to_gm(model, args=(x,), clone=True)

    gm_result = InferenceOptimizer(
        None,
        {
            "match_fp8_swiglu_pattern": {
                "stage": "pattern_matcher",
            },
        },
    )(None, gm)

    assert _count_ops(gm_result, torch.ops.auto_deploy.torch_fp8_swiglu_mlp.default) == 0
    assert _count_ops(gm_result, torch.ops.auto_deploy.torch_fake_quant_fp8_linear.default) == 2
