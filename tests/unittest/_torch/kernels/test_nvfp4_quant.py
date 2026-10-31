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
"""Byte-level parity with the linear-layout CUDA NVFP4 quantizer."""

import pytest
import torch
from utils.util import skip_pre_blackwell

from tensorrt_llm._torch.kernels.nvfp4_quant import can_use_nvfp4_quant, nvfp4_quant

pytestmark = skip_pre_blackwell


def _check(x: torch.Tensor, scale: torch.Tensor) -> None:
    actual = nvfp4_quant(x, scale)
    expected = torch.ops.trtllm.fp4_quantize(x, scale, 16, False, False)
    for result, reference in zip(actual, expected):
        assert result.dtype == reference.dtype == torch.uint8
        torch.testing.assert_close(result, reference, rtol=0, atol=0)


@pytest.mark.parametrize("rows", [1, 3, 4, 5, 129, 1025, 4097])
@pytest.mark.parametrize("hidden", [1024, 2048, 4096, 7168])
@pytest.mark.parametrize("global_scale", [1e-6, 1.0, 100.0, 1e6])
def test_nvfp4_quant_parity(rows: int, hidden: int, global_scale: float) -> None:
    torch.manual_seed(42)
    x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    scale = torch.tensor([global_scale], device="cuda", dtype=torch.float32)
    _check(x, scale)


@pytest.mark.parametrize("global_scale", [1e-6, 1.0, 100.0, 1e6])
def test_nvfp4_quant_finite_bf16_patterns(global_scale: float) -> None:
    # All finite BF16 bit patterns, including signed zero and subnormals.
    x = torch.arange(65536, device="cuda").to(torch.int16).view(torch.bfloat16)
    x = torch.where(x.isfinite(), x, 0).reshape(64, 1024)
    scale = torch.tensor([global_scale], device="cuda", dtype=torch.float32)
    _check(x, scale)


@pytest.mark.parametrize("fill", [0.0, -0.0, 0.25, -0.75, 1.25, 2.5, 5.0, 6.0])
def test_nvfp4_quant_rounding_and_zero(fill: float) -> None:
    x = torch.full((5, 1024), fill, device="cuda", dtype=torch.bfloat16)
    # Max=6, global_scale=1 makes the E4M3 block scale exactly 1.
    x[:, 15::16] = 6.0
    _check(x, torch.ones(1, device="cuda"))


def test_nvfp4_quant_graph_replay() -> None:
    x = torch.randn(5, 4096, device="cuda", dtype=torch.bfloat16)
    scale = torch.ones(1, device="cuda")
    _check(x, scale)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = nvfp4_quant(x, scale)
    for value in [0.0, 1.0, -2.5]:
        x.fill_(value)
        graph.replay()
        expected = torch.ops.trtllm.fp4_quantize(x, scale, 16, False, False)
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference, rtol=0, atol=0)


def test_nvfp4_quant_eligibility() -> None:
    x = torch.ones(5, 2048, device="cuda", dtype=torch.bfloat16)
    scale = torch.ones(1, device="cuda")
    assert can_use_nvfp4_quant(x, scale)
    for excluded in [x[:0], x[:, :0], x[:, :1024], x[:, ::2], x.float(), x.half(), x[0]]:
        assert not can_use_nvfp4_quant(excluded, scale)
    for excluded_scale in [scale.cpu(), scale.half(), scale.expand(2)]:
        assert not can_use_nvfp4_quant(x, excluded_scale)


@pytest.mark.parametrize("global_scale", [1.0, 100.0])
@pytest.mark.parametrize(
    "value", [float("inf"), -float("inf"), float("nan")], ids=["inf", "neg_inf", "nan"]
)
def test_nvfp4_quant_nonfinite_inputs(global_scale: float, value: float) -> None:
    scale = torch.tensor([global_scale], device="cuda", dtype=torch.float32)
    # Exercise each lane of a scale block, mixed and entirely non-finite blocks.
    for lane in range(16):
        x = torch.linspace(-6, 6, 1024, device="cuda", dtype=torch.bfloat16).repeat(5, 1)
        x[:, lane::16] = value
        _check(x, scale)
    _check(torch.full((5, 1024), value, device="cuda", dtype=torch.bfloat16), scale)
