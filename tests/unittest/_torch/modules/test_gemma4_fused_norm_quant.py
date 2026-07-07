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
"""Parity test: fused Gemma4 RMSNorm+NVFP4-quantize vs the unfused reference
chain (trtllm::flashinfer_rmsnorm -> trtllm::fp4_quantize).

Both the packed E2M1 payload and the swizzled E4M3 scale factors are compared
at the byte level; the scale comparison is restricted to the valid (row, kvec)
region because the unfused op leaves its 128-row padding uninitialized (the
fused op zero-fills it). Unlike the gelu+quant sibling, the producer here
contains a row reduction (the norm's sum of squares), so the fp32
reduction-order difference can flip an occasional bf16 pre-quant value and
with it a payload byte / block scale - the tolerance below covers that
(~5e-6 class, same as the fused-tail kernels).
"""

import pytest
import torch

import tensorrt_llm  # noqa: F401  (registers trtllm torch ops)
import tensorrt_llm._torch.custom_ops.flashinfer_custom_ops  # noqa: F401
from tensorrt_llm._torch.models.gemma4_fused_gelu_quant import sf_swizzled_offsets
from tensorrt_llm._torch.models.gemma4_fused_norm_quant import gemma4_fused_norm_fp4


def _reference(x, w, eps, gs):
    n = torch.ops.trtllm.flashinfer_rmsnorm(x, w, eps)
    return torch.ops.trtllm.fp4_quantize(n, gs, 16, False)


def _check(x, w, eps, gs):
    fq, fsf = gemma4_fused_norm_fp4(x, w, eps, gs)
    rq, rsf = _reference(x.contiguous(), w, eps, gs)
    assert fq.shape == rq.shape and fsf.numel() == rsf.numel()
    mm_fp4 = (fq != rq).float().mean().item()
    valid = sf_swizzled_offsets(x.shape[0], x.shape[1] // 16, x.device)
    mm_sf = (fsf[valid] != rsf[valid]).float().mean().item()
    assert mm_fp4 < 1e-4, f"fp4 payload mismatch fraction {mm_fp4}"
    assert mm_sf < 1e-4, f"scale-factor mismatch fraction {mm_sf}"


# 5376 is the Gemma4-31B hidden size; 228 the pinned decode batch; 7700 the
# profiled serving prefill size; 333 exercises masked tail rows.
@pytest.mark.parametrize("n_tokens", [1, 7, 228, 333, 7700])
@pytest.mark.parametrize("hidden", [5376, 512])
def test_fused_norm_quant_parity(n_tokens, hidden):
    torch.manual_seed(1234)
    x = (
        torch.randn((n_tokens, hidden), dtype=torch.bfloat16, device="cuda")
        * torch.rand((n_tokens, 1), dtype=torch.bfloat16, device="cuda")
        * 3
    )
    w = torch.rand((hidden,), dtype=torch.bfloat16, device="cuda") + 0.5
    n = torch.ops.trtllm.flashinfer_rmsnorm(x, w, 1e-6)
    gs = (448.0 * 6.0 / n.abs().max().float()).reshape(1)
    _check(x, w, 1e-6, gs)


@pytest.mark.parametrize("gs_value", [1e6, 1.0, 1e-6])
def test_fused_norm_quant_extreme_scales(gs_value):
    """Saturating / degenerate global scales must match the unfused op."""
    torch.manual_seed(7)
    x = torch.randn((333, 5376), dtype=torch.bfloat16, device="cuda")
    w = torch.rand((5376,), dtype=torch.bfloat16, device="cuda") + 0.5
    gs = torch.tensor([gs_value], dtype=torch.float32, device="cuda")
    _check(x, w, 1e-6, gs)


def test_fused_norm_quant_strided_rows():
    """Row-strided input (a view of a wider buffer)."""
    torch.manual_seed(3)
    n, h = 65, 5376
    buf = torch.randn((n, h + 512), dtype=torch.bfloat16, device="cuda")
    x = buf[:, :h]
    w = torch.rand((h,), dtype=torch.bfloat16, device="cuda") + 0.5
    normed = torch.ops.trtllm.flashinfer_rmsnorm(x.contiguous(), w, 1e-6)
    gs = (448.0 * 6.0 / normed.abs().max().float()).reshape(1)
    _check(x, w, 1e-6, gs)


def test_fused_norm_quant_zero_row():
    """An all-zero row must produce zero scales (the vmax==0 branch)."""
    torch.manual_seed(5)
    x = torch.randn((9, 5376), dtype=torch.bfloat16, device="cuda")
    x[4] = 0
    w = torch.rand((5376,), dtype=torch.bfloat16, device="cuda") + 0.5
    gs = torch.tensor([100.0], dtype=torch.float32, device="cuda")
    _check(x, w, 1e-6, gs)


if __name__ == "__main__":
    test_fused_norm_quant_parity(7700, 5376)
    test_fused_norm_quant_parity(228, 5376)
    test_fused_norm_quant_extreme_scales(1e6)
    test_fused_norm_quant_strided_rows()
    test_fused_norm_quant_zero_row()
    print("ALL PARITY CHECKS PASSED")
