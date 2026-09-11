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
"""Unit tests for ``trtllm::cute_dsl_bf16_bmm_fp8out_blackwell``.

BF16 x BF16 batched GEMM (FP32 accumulate) whose epilogue stores FP8 E4M3 at
unit scale, possibly into a strided column slice of a wider buffer. It replaces
the bf16 absorb bmm + standalone quantizeCopyInputToFp8Kernel pair on the
GLM 5.2 / DSv3.2 MLA context path (MLA.forward_absorption_context), writing the
nope columns of the FP8 Q buffer directly.

Reference: torch.bmm in fp32 -> .to(float8_e4m3fn). Tensor-core accumulation
order differs from torch's, so values sitting on an FP8 rounding boundary can
land on the other side; we require >= 99% bit-exact FP8 bytes and a bounded
dequantized error everywhere (one E4M3 step = 12.5% relative).
"""

import pytest
import torch

import tensorrt_llm  # noqa: F401  (loads the custom ops)
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._utils import is_sm_100f

skip_unless_available = pytest.mark.skipif(
    not is_sm_100f() or not IS_CUTLASS_DSL_AVAILABLE
    or not hasattr(torch.ops.trtllm, "cute_dsl_bf16_bmm_fp8out_blackwell"),
    reason="Requires SM100 family, CuTe DSL and the cute_dsl_bf16_bmm_fp8out_blackwell op",
)


def _ref_fp8(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # a: [B, M, K], b: [B, N, K] -> [B, M, N]
    return torch.bmm(a.float(), b.float().transpose(1, 2)).to(torch.float8_e4m3fn)


def _check(out: torch.Tensor, ref: torch.Tensor, ctx: str, min_match: float = 0.99):
    assert out.dtype == torch.float8_e4m3fn
    exact = (out.view(torch.uint8) == ref.view(torch.uint8)).float().mean().item()
    assert exact >= min_match, f"FP8 bit-exact rate {exact:.4f} < {min_match} ({ctx})"
    o, r = out.float(), ref.float()
    # Every element within one E4M3 step of the reference (plus an absolute
    # floor for values near zero where the step is tiny but accumulation noise is not).
    tol = 0.125 * r.abs() + 1e-2 * r.abs().amax()
    bad = (o - r).abs() > tol
    assert not bad.any(), f"{int(bad.sum())} elements beyond one FP8 step ({ctx})"
    assert torch.isfinite(o).all(), ctx


@skip_unless_available
@pytest.mark.parametrize("batch", [1, 4, 64])
@pytest.mark.parametrize("m", [128, 1000, 4096])
@pytest.mark.parametrize("n,k", [(512, 192), (512, 128), (256, 512)])
def test_bf16_bmm_fp8out_contiguous(batch, m, n, k):
    torch.manual_seed(0)
    device = torch.device("cuda")
    # Scale inputs so outputs span the FP8 range without saturating (|out| < 448).
    a = torch.randn(batch, m, k, dtype=torch.bfloat16, device=device)
    b = torch.randn(batch, n, k, dtype=torch.bfloat16, device=device) * (4.0 / k**0.5)
    out = torch.empty(batch, m, n, dtype=torch.float8_e4m3fn, device=device)
    torch.ops.trtllm.cute_dsl_bf16_bmm_fp8out_blackwell(a, b, out)
    torch.cuda.synchronize()
    _check(out, _ref_fp8(a, b), f"contiguous b={batch} m={m} n={n} k={k}")


@skip_unless_available
@pytest.mark.parametrize("num_tokens", [64, 1000, 8192])
def test_bf16_bmm_fp8out_mla_absorb_column_slice(num_tokens):
    """The production call: q_nope [tokens, H, 192] (transposed view of the
    q_b_proj output) x W_UK^T [H, 512, 192] written into columns 0..511 of the
    FP8 Q buffer [tokens, H, 576]; columns 512..575 (the RoPE segment the RoPE
    kernel fills) must be left untouched."""
    torch.manual_seed(1)
    device = torch.device("cuda")
    heads, nope, lora, rope = 64, 192, 512, 64
    q = torch.randn(num_tokens, heads, nope + rope, dtype=torch.bfloat16, device=device)
    q_nope = q[..., :nope]  # [tokens, H, 192], row-strided view
    w_uk_t = torch.randn(heads, lora, nope, dtype=torch.bfloat16, device=device) * (4.0 / nope**0.5)
    quant_q = torch.empty(num_tokens, heads, lora + rope, dtype=torch.float8_e4m3fn, device=device)
    sentinel = torch.full_like(quant_q[..., lora:], 3.0).to(torch.float8_e4m3fn)
    quant_q[..., lora:] = sentinel

    torch.ops.trtllm.cute_dsl_bf16_bmm_fp8out_blackwell(
        q_nope.transpose(0, 1),  # [H, tokens, 192], K contiguous
        w_uk_t,  # [H, 512, 192]
        quant_q[..., :lora].transpose(0, 1),  # [H, tokens, 512], N contiguous, row stride 576
    )
    torch.cuda.synchronize()
    ref = _ref_fp8(q_nope.transpose(0, 1).contiguous(), w_uk_t)  # [H, tokens, 512]
    _check(quant_q[..., :lora].transpose(0, 1), ref, f"absorb slice tokens={num_tokens}")
    assert torch.equal(
        quant_q[..., lora:].view(torch.uint8), sentinel.view(torch.uint8)
    ), "rope columns were overwritten"


@skip_unless_available
def test_bf16_bmm_fp8out_matches_standalone_quant_path():
    """Equivalence with the path it replaces: bf16 bmm (torch) followed by an
    fp8 cast is what quantizeCopyInputToFp8Kernel does at unit scale."""
    torch.manual_seed(2)
    device = torch.device("cuda")
    heads, m, k, n = 8, 512, 192, 512
    a = torch.randn(heads, m, k, dtype=torch.bfloat16, device=device)
    b = torch.randn(heads, n, k, dtype=torch.bfloat16, device=device) * (4.0 / k**0.5)
    bf16_out = torch.bmm(a, b.transpose(1, 2))
    two_pass = bf16_out.to(torch.float8_e4m3fn)
    fused = torch.empty_like(two_pass)
    torch.ops.trtllm.cute_dsl_bf16_bmm_fp8out_blackwell(a, b, fused)
    torch.cuda.synchronize()
    # The fused path quantizes from the fp32 accumulator (no bf16 round trip),
    # so it is at least as close to the fp32 product as the two-pass result.
    exact_fp32 = torch.bmm(a.float(), b.float().transpose(1, 2))
    err_fused = (fused.float() - exact_fp32).abs().mean()
    err_two_pass = (two_pass.float() - exact_fp32).abs().mean()
    assert err_fused <= err_two_pass * 1.05, f"fused err {err_fused} > two-pass err {err_two_pass}"


@skip_unless_available
def test_bf16_bmm_fp8out_rejects_noncontiguous_n():
    device = torch.device("cuda")
    a = torch.randn(2, 128, 64, dtype=torch.bfloat16, device=device)
    b = torch.randn(2, 128, 64, dtype=torch.bfloat16, device=device)
    out = torch.empty(2, 128, 128, dtype=torch.float8_e4m3fn, device=device).transpose(1, 2)
    with pytest.raises(AssertionError):
        torch.ops.trtllm.cute_dsl_bf16_bmm_fp8out_blackwell(a, b, out)
