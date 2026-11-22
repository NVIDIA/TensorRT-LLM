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
"""Parity test: fused gelu_tanh+mul+FP8 quantize vs flashinfer_gelu_tanh_and_mul + static quantize."""

import pytest
import torch

import tensorrt_llm  # noqa: F401  (registers trtllm torch ops)
import tensorrt_llm._torch.custom_ops.flashinfer_custom_ops  # noqa: F401
from tensorrt_llm._torch.kernels.fused_ops.gelu_tanh_mul_fp8_quant import gelu_tanh_mul_fp8_quant


def _reference(x, qs):
    act = torch.ops.trtllm.flashinfer_gelu_tanh_and_mul(x)
    return torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(act, qs)[0]


# 10240 is the Gemma4-E4B intermediate size. The input is a row slice of a wider buffer.
@pytest.mark.parametrize("n_tokens", [1, 333, 4096])
@pytest.mark.parametrize("intermediate", [10240, 2112])
@pytest.mark.parametrize("quant_scale", [0.03, 0.002])  # 0.03: FP8 ties; 0.002 saturates
def test_gelu_tanh_mul_fp8_quant_parity(n_tokens, intermediate, quant_scale):
    torch.manual_seed(9)
    buf = torch.randn((n_tokens, 2 * intermediate + 256), dtype=torch.bfloat16, device="cuda") * 2
    x = buf[:, : 2 * intermediate]
    qs = torch.tensor([quant_scale], dtype=torch.float32, device="cuda")
    fused = gelu_tanh_mul_fp8_quant(x, qs)
    assert fused.dtype == torch.float8_e4m3fn and fused.shape == (n_tokens, intermediate)
    ref = _reference(x.contiguous(), qs)
    assert torch.equal(fused.view(torch.uint8), ref.view(torch.uint8))
