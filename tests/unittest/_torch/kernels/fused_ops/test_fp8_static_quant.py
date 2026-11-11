# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Triton static FP8 quantization must match the C++ op bit for bit."""

import pytest
import torch
from utils.util import skip_pre_ada

import tensorrt_llm  # noqa: F401  (registers torch.ops.tensorrt_llm)
from tensorrt_llm._torch.kernels.fused_ops.fp8_static_quant import fp8_static_quant
from tensorrt_llm._torch.modules import linear as linear_module
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

pytestmark = skip_pre_ada

_MIN_NUMEL = linear_module._TRITON_STATIC_FP8_QUANT_MIN_NUMEL


def _cpp_quant(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(x, scale)[0]


def _assert_bitwise_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.dtype == torch.float8_e4m3fn
    torch.testing.assert_close(actual.view(torch.uint8), expected.view(torch.uint8), atol=0, rtol=0)


@pytest.fixture
def triton_calls(monkeypatch):
    """Records the inputs that the Linear dispatch sends to the Triton kernel."""
    calls = []

    def recording_quant(input, scale):
        calls.append(input.shape)
        return fp8_static_quant(input, scale)

    monkeypatch.setattr(linear_module, "fp8_static_quant", recording_quant)
    return calls


@pytest.mark.parametrize("shape", [(1, 128), (3, 5, 4096), (3, 16385)])
@pytest.mark.parametrize("scale_value", [0.0213, 1.0, 3.7e-3])
def test_fp8_static_quant_matches_cpp_op(shape, scale_value):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.bfloat16, device="cuda") * 3
    # Some values far outside the FP8 range exercise saturation.
    x.view(-1)[::97] *= 1000
    scale = torch.tensor([scale_value], dtype=torch.float32, device="cuda")

    _assert_bitwise_equal(fp8_static_quant(x, scale), _cpp_quant(x, scale))


def test_fp8_static_quant_special_values():
    values = [0.0, -0.0, 1e-30, -1e-30, 1e-5, 448.0, 449.0, 1e30, -1e30]
    values += [float("inf"), float("-inf")]
    x = torch.tensor(values, dtype=torch.bfloat16, device="cuda").repeat(64, 1)
    scale = torch.tensor([0.5], dtype=torch.float32, device="cuda")

    _assert_bitwise_equal(fp8_static_quant(x, scale), _cpp_quant(x, scale))


@pytest.mark.parametrize(
    "rows,dtype,expect_triton",
    [
        (_MIN_NUMEL // 4096 - 1, torch.bfloat16, False),
        (_MIN_NUMEL // 4096, torch.bfloat16, True),
        (_MIN_NUMEL // 4096, torch.float16, False),
    ],
)
def test_static_quantize_dispatch(triton_calls, rows, dtype, expect_triton):
    torch.manual_seed(1)
    x = torch.randn((rows, 4096), dtype=dtype, device="cuda")
    scale = torch.tensor([0.0213], dtype=torch.float32, device="cuda")

    out = linear_module._static_quantize_e4m3_per_tensor(x, scale)

    assert bool(triton_calls) == expect_triton
    _assert_bitwise_equal(out, _cpp_quant(x, scale))


def test_fp8_linear_output_unchanged_by_triton_quant(triton_calls, monkeypatch):
    torch.manual_seed(2)
    in_features = 4096
    fp8 = QuantConfig(quant_algo=QuantAlgo.FP8)
    linear = Linear(in_features, 512, bias=False, dtype=torch.bfloat16, quant_config=fp8).cuda()
    linear.weight.copy_(torch.randn(linear.weight.shape, device="cuda").to(torch.float8_e4m3fn))
    linear.input_scale.fill_(0.0213)
    x = torch.randn((_MIN_NUMEL // in_features, in_features), dtype=torch.bfloat16, device="cuda")

    with torch.inference_mode():
        triton_out = linear(x)
        assert triton_calls
        monkeypatch.setattr(linear_module, "_TRITON_STATIC_FP8_QUANT_MIN_NUMEL", 1 << 62)
        cpp_out = linear(x)

    torch.testing.assert_close(triton_out, cpp_out, atol=0, rtol=0)
