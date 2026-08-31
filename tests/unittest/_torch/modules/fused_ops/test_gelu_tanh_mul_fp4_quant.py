# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Parity test: fused gelu_tanh+mul+NVFP4-quantize (modules/fused_ops/
gelu_tanh_mul_fp4_quant) vs the unfused reference chain
(trtllm::flashinfer_gelu_tanh_and_mul -> trtllm::fp4_quantize).

Both the packed E2M1 payload and the swizzled E4M3 scale factors are compared
at the byte level; the scale comparison is restricted to the valid (row, kvec)
region because the unfused op leaves its 128-row padding uninitialized (the
fused op zero-fills it). Byte-exact on SM100 (measured mismatch 0.0); the
tolerance below only guards hypothetical cross-arch fp contraction drift.
"""

import pytest
import torch
from utils.util import skip_pre_blackwell

import tensorrt_llm  # noqa: F401  (registers trtllm torch ops)
import tensorrt_llm._torch.custom_ops.flashinfer_custom_ops  # noqa: F401
from tensorrt_llm._torch.modules.fused_ops.gelu_tanh_mul_fp4_quant import (
    gelu_tanh_mul_fp4_quant,
    sf_swizzled_offsets,
)

pytestmark = skip_pre_blackwell


def _reference(x, gs):
    h = torch.ops.trtllm.flashinfer_gelu_tanh_and_mul(x)
    return torch.ops.trtllm.fp4_quantize(h, gs, 16, False)


def _check(x, gs):
    fq, fsf = gelu_tanh_mul_fp4_quant(x, gs)
    rq, rsf = _reference(x, gs)
    assert fq.shape == rq.shape and fsf.numel() == rsf.numel()
    mm_fp4 = (fq != rq).float().mean().item()
    valid = sf_swizzled_offsets(x.shape[0], x.shape[1] // 2 // 16, x.device)
    mm_sf = (fsf[valid] != rsf[valid]).float().mean().item()
    assert mm_fp4 < 1e-4, f"fp4 payload mismatch fraction {mm_fp4}"
    assert mm_sf < 1e-4, f"scale-factor mismatch fraction {mm_sf}"


# 21504 is the Gemma4-31B intermediate size; 228 the pinned decode batch;
# 6455 the profiled serving prefill size; 333 exercises masked tail rows.
@pytest.mark.parametrize("n_tokens", [1, 7, 228, 333, 6455])
@pytest.mark.parametrize("intermediate", [21504, 1024])
def test_gelu_tanh_mul_fp4_quant_parity(n_tokens, intermediate):
    torch.manual_seed(1234)
    x = (
        torch.randn((n_tokens, 2 * intermediate), dtype=torch.bfloat16, device="cuda")
        * torch.rand((n_tokens, 1), dtype=torch.bfloat16, device="cuda")
        * 3
    )
    h = torch.ops.trtllm.flashinfer_gelu_tanh_and_mul(x)
    gs = (448.0 * 6.0 / h.abs().max().float()).reshape(1)
    _check(x, gs)


@pytest.mark.parametrize("gs_value", [1e6, 1.0, 1e-6])
def test_gelu_tanh_mul_fp4_quant_extreme_scales(gs_value):
    """Saturating / degenerate global scales must match the unfused op."""
    torch.manual_seed(7)
    x = torch.randn((333, 2 * 21504), dtype=torch.bfloat16, device="cuda")
    gs = torch.tensor([gs_value], dtype=torch.float32, device="cuda")
    _check(x, gs)


def test_gelu_tanh_mul_fp4_quant_strided_rows():
    """Row-strided input (a view of a wider buffer)."""
    torch.manual_seed(3)
    n, i = 65, 21504
    buf = torch.randn((n, 2 * i + 512), dtype=torch.bfloat16, device="cuda")
    x = buf[:, : 2 * i]
    h = torch.ops.trtllm.flashinfer_gelu_tanh_and_mul(x.contiguous())
    gs = (448.0 * 6.0 / h.abs().max().float()).reshape(1)
    fq, fsf = gelu_tanh_mul_fp4_quant(x, gs)
    rq, rsf = torch.ops.trtllm.fp4_quantize(h, gs, 16, False)
    valid = sf_swizzled_offsets(n, i // 16, x.device)
    assert (fq != rq).float().mean().item() < 1e-4
    assert (fsf[valid] != rsf[valid]).float().mean().item() < 1e-4


if __name__ == "__main__":
    test_gelu_tanh_mul_fp4_quant_parity(6455, 21504)
    test_gelu_tanh_mul_fp4_quant_parity(228, 21504)
    test_gelu_tanh_mul_fp4_quant_extreme_scales(1e6)
    test_gelu_tanh_mul_fp4_quant_strided_rows()
    print("ALL PARITY CHECKS PASSED")
