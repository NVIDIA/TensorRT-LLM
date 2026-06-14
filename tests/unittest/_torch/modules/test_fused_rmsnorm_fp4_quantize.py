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
"""Unit tests for the fused (add +) RMSNorm + NVFP4-quantize kernels.

Both ops fold the standalone NVFP4 input-quantize (FP16/BF16 -> NVFP4) into its
producing RMSNorm at DeepSeek-V3.2 / Kimi-K2.5 MLA sites:

  * ``fused_add_rmsnorm_fp4_quantize`` - layer-boundary
    ``residual_add -> rms_norm -> nvfp4_quantize`` feeding the next layer's
    ``kv_a_proj``.
  * ``fused_rmsnorm_fp4_quantize`` - the residual-less intra-MLA
    ``q_a_layernorm -> q_b_proj`` input quant, which additionally supports a
    row-strided (column-slice) input read so the ``q`` slice from
    ``kv_a_proj_with_mqa(...).split(...)`` needs no ``.contiguous()`` copy.

Accuracy is checked against the unfused reference: a PyTorch RMSNorm followed by
``torch.ops.trtllm.fp4_quantize`` (the exact kernel the fusion replaces). The
fused kernel computes the RMSNorm in registers while already reading the tensor
to quantize it, so a handful of values right at quantization boundaries can fall
on the other side of a step due to FMA-vs-separate rounding; we require a >= 99%
match on the packed FP4 bytes, mirroring ``test_fused_activation_quant.py``.
"""

import pytest
import torch

from tests.unittest.utils.util import getSMVersion


def fused_add_rmsnorm_fp4_quantize_available():
    return hasattr(torch.ops, "trtllm") and hasattr(
        torch.ops.trtllm, "fused_add_rmsnorm_fp4_quantize"
    )


def fused_rmsnorm_fp4_quantize_available():
    return hasattr(torch.ops, "trtllm") and hasattr(torch.ops.trtllm, "fused_rmsnorm_fp4_quantize")


def fp4_quantize_available():
    return hasattr(torch.ops, "trtllm") and hasattr(torch.ops.trtllm, "fp4_quantize")


skip_unless_add_rmsnorm = pytest.mark.skipif(
    getSMVersion() < 100
    or not fused_add_rmsnorm_fp4_quantize_available()
    or not fp4_quantize_available(),
    reason="Requires SM100+ (Blackwell) and trtllm "
    "fused_add_rmsnorm_fp4_quantize + fp4_quantize ops",
)

skip_unless_rmsnorm = pytest.mark.skipif(
    getSMVersion() < 100
    or not fused_rmsnorm_fp4_quantize_available()
    or not fp4_quantize_available(),
    reason="Requires SM100+ (Blackwell) and trtllm fused_rmsnorm_fp4_quantize + fp4_quantize ops",
)


def rms_norm_ref(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Reference RMSNorm matching tensorrt_llm/_torch/modules/rms_norm.py:
    fp32 accumulate, rsqrt(var + eps), then weight * normed cast back to the
    input dtype."""
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return weight * hidden_states.to(input_dtype)


def fp4_quantize_ref(normed: torch.Tensor, sf_scale: torch.Tensor):
    """Unfused baseline: the standalone NVFP4 quantize the fusion replaces."""
    return torch.ops.trtllm.fp4_quantize(
        normed.contiguous(),
        sf_scale,
        16,
        False,  # use_ue8m0
        True,  # is_sf_swizzled_layout
    )


def make_sf_scale(normed: torch.Tensor) -> torch.Tensor:
    """Per-tensor SF scale (amax / (6 * 448)), same scalar fed to both the
    fused op and fp4_quantize — both forward it to cvt_warp_fp16_to_fp4."""
    return (normed.abs().amax().float() / (6.0 * 448.0)).to(normed.device).view(1)


def assert_fp4_match(fp4_fused: torch.Tensor, fp4_ref: torch.Tensor, ctx: str):
    assert fp4_fused.shape == fp4_ref.shape, (
        f"shape mismatch {tuple(fp4_fused.shape)} vs {tuple(fp4_ref.shape)} ({ctx})"
    )
    match_rate = (fp4_fused == fp4_ref).float().mean().item()
    assert match_rate >= 0.99, f"FP4 packed match rate {match_rate:.4f} < 0.99 ({ctx})"


# --------------------------------------------------------------------------- #
# fused_add_rmsnorm_fp4_quantize: residual_add -> rms_norm -> nvfp4_quantize    #
# --------------------------------------------------------------------------- #
@skip_unless_add_rmsnorm
@pytest.mark.parametrize("m", [1, 16, 64, 128])
@pytest.mark.parametrize("n", [32, 128, 512, 7168])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_add_rmsnorm_fp4_quantize_vs_separate(m, n, dtype):
    torch.manual_seed(42)
    device = torch.device("cuda")
    eps = 1e-6

    hidden = torch.randn(m, n, dtype=dtype, device=device)
    residual = torch.randn(m, n, dtype=dtype, device=device)
    weight = torch.randn(n, dtype=dtype, device=device)

    # Reference: add in fp32 (matches kernel), then rmsnorm, then quantize.
    added = hidden.float() + residual.float()
    normed_ref = rms_norm_ref(added.to(dtype), weight, eps)
    sf_scale = make_sf_scale(normed_ref)
    fp4_ref, _ = fp4_quantize_ref(normed_ref, sf_scale)

    # The op returns residual_out as a fresh tensor and must NOT mutate
    # hidden_states (required for torch.compile functionalization).
    hidden_fused = hidden.clone()
    hidden_before = hidden_fused.clone()
    quant_out, scale_out, residual_out = torch.ops.trtllm.fused_add_rmsnorm_fp4_quantize(
        hidden_fused, residual, weight, sf_scale, eps, False
    )

    assert_fp4_match(quant_out, fp4_ref, f"add-rmsnorm m={m} n={n} {dtype}")

    # hidden_states must be left untouched (no in-place alias).
    torch.testing.assert_close(hidden_fused, hidden_before, rtol=0, atol=0)
    # residual_out must equal hidden + residual (the next layer's residual).
    torch.testing.assert_close(residual_out.float(), added, rtol=2e-2, atol=2e-2)


@skip_unless_add_rmsnorm
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_add_rmsnorm_fp4_quantize_return_norm_out(dtype):
    """return_norm_out=True must also yield the BF16/FP16 post-RMSNorm value
    (consumed by the DSA indexer's pre_indexer_proj)."""
    torch.manual_seed(7)
    device = torch.device("cuda")
    m, n, eps = 64, 7168, 1e-6

    hidden = torch.randn(m, n, dtype=dtype, device=device)
    residual = torch.randn(m, n, dtype=dtype, device=device)
    weight = torch.randn(n, dtype=dtype, device=device)

    added = hidden.float() + residual.float()
    normed_ref = rms_norm_ref(added.to(dtype), weight, eps)
    sf_scale = make_sf_scale(normed_ref)
    fp4_ref, _ = fp4_quantize_ref(normed_ref, sf_scale)

    hidden_fused = hidden.clone()
    norm_out, quant_out, scale_out, residual_out = torch.ops.trtllm.fused_add_rmsnorm_fp4_quantize(
        hidden_fused, residual, weight, sf_scale, eps, True
    )

    assert_fp4_match(quant_out, fp4_ref, f"add-rmsnorm return_norm_out {dtype}")
    torch.testing.assert_close(norm_out, normed_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(residual_out.float(), added, rtol=2e-2, atol=2e-2)


# --------------------------------------------------------------------------- #
# fused_rmsnorm_fp4_quantize: residual-less rms_norm -> nvfp4_quantize          #
# --------------------------------------------------------------------------- #
@skip_unless_rmsnorm
@pytest.mark.parametrize("m", [1, 16, 64, 128])
@pytest.mark.parametrize("n", [32, 128, 512, 1536])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_rmsnorm_fp4_quantize_vs_separate(m, n, dtype):
    torch.manual_seed(42)
    device = torch.device("cuda")
    eps = 1e-6

    hidden = torch.randn(m, n, dtype=dtype, device=device)
    weight = torch.randn(n, dtype=dtype, device=device)

    normed_ref = rms_norm_ref(hidden, weight, eps)
    sf_scale = make_sf_scale(normed_ref)
    fp4_ref, _ = fp4_quantize_ref(normed_ref, sf_scale)

    quant_out, scale_out = torch.ops.trtllm.fused_rmsnorm_fp4_quantize(
        hidden, weight, sf_scale, eps, False
    )

    assert_fp4_match(quant_out, fp4_ref, f"rmsnorm m={m} n={n} {dtype}")


@skip_unless_rmsnorm
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_rmsnorm_fp4_quantize_return_norm_out(dtype):
    torch.manual_seed(11)
    device = torch.device("cuda")
    m, n, eps = 64, 1536, 1e-6

    hidden = torch.randn(m, n, dtype=dtype, device=device)
    weight = torch.randn(n, dtype=dtype, device=device)

    normed_ref = rms_norm_ref(hidden, weight, eps)
    sf_scale = make_sf_scale(normed_ref)
    fp4_ref, _ = fp4_quantize_ref(normed_ref, sf_scale)

    norm_out, quant_out, scale_out = torch.ops.trtllm.fused_rmsnorm_fp4_quantize(
        hidden, weight, sf_scale, eps, True
    )

    assert_fp4_match(quant_out, fp4_ref, f"rmsnorm return_norm_out {dtype}")
    torch.testing.assert_close(norm_out, normed_ref, rtol=2e-2, atol=2e-2)


@skip_unless_rmsnorm
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_rmsnorm_fp4_quantize_does_not_mutate_input(dtype):
    """Residual=false path must not write back into hidden_states."""
    torch.manual_seed(3)
    device = torch.device("cuda")
    m, n, eps = 32, 512, 1e-6

    hidden = torch.randn(m, n, dtype=dtype, device=device)
    weight = torch.randn(n, dtype=dtype, device=device)
    hidden_before = hidden.clone()
    sf_scale = make_sf_scale(rms_norm_ref(hidden, weight, eps))

    torch.ops.trtllm.fused_rmsnorm_fp4_quantize(hidden, weight, sf_scale, eps, False)
    torch.testing.assert_close(hidden, hidden_before, rtol=0, atol=0)


# m starts at 2: a single-row (m=1) column slice is still contiguous in PyTorch
# (the row-pitch stride is irrelevant when there is only one row), so it would
# not exercise the input_row_stride read path at all.
@skip_unless_rmsnorm
@pytest.mark.parametrize("m", [2, 16, 64])
@pytest.mark.parametrize("n", [512, 1536])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_rmsnorm_fp4_quantize_strided_input(m, n, dtype):
    """The key F1 feature: read a row-strided column slice in place (no copy).

    Mimics ``q = kv_a_proj_with_mqa(x).split(...)[0]``: a leading column slice
    of a wider projection whose last dim is unit-stride but whose row pitch is
    the full projection width. The fused op must read this strided layout
    directly and match the contiguous reference exactly (the read offset is the
    only thing the stride changes; all outputs are written packed)."""
    torch.manual_seed(42)
    device = torch.device("cuda")
    eps = 1e-6
    extra = 2 * 16  # trailing columns (e.g. kv_lora + rope), keep 16-aligned

    wide = torch.randn(m, n + extra, dtype=dtype, device=device)
    q = wide[:, :n]  # column slice: unit-stride last dim, row pitch n+extra
    assert not q.is_contiguous()
    assert q.stride(0) == n + extra and q.stride(1) == 1

    weight = torch.randn(n, dtype=dtype, device=device)
    normed_ref = rms_norm_ref(q.contiguous(), weight, eps)
    sf_scale = make_sf_scale(normed_ref)
    fp4_ref, _ = fp4_quantize_ref(normed_ref, sf_scale)

    quant_out, scale_out = torch.ops.trtllm.fused_rmsnorm_fp4_quantize(
        q, weight, sf_scale, eps, False
    )

    assert quant_out.is_contiguous(), "outputs must be packed"
    assert_fp4_match(quant_out, fp4_ref, f"strided m={m} n={n} {dtype}")


@skip_unless_rmsnorm
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_rmsnorm_fp4_quantize_contiguous_matches_strided(dtype):
    """A contiguous tensor (input_row_stride==0 path) and the same data viewed
    as a column slice (input_row_stride>0 path) must produce identical FP4."""
    torch.manual_seed(99)
    device = torch.device("cuda")
    m, n, eps = 48, 1536, 1e-6

    base = torch.randn(m, n, dtype=dtype, device=device)
    weight = torch.randn(n, dtype=dtype, device=device)
    sf_scale = make_sf_scale(rms_norm_ref(base, weight, eps))

    # Packed path.
    fp4_packed, _ = torch.ops.trtllm.fused_rmsnorm_fp4_quantize(
        base.contiguous(), weight, sf_scale, eps, False
    )

    # Strided path: embed the same rows into a wider buffer as a column slice.
    wide = torch.randn(m, n + 32, dtype=dtype, device=device)
    wide[:, :n] = base
    strided = wide[:, :n]
    fp4_strided, _ = torch.ops.trtllm.fused_rmsnorm_fp4_quantize(
        strided, weight, sf_scale, eps, False
    )

    torch.testing.assert_close(fp4_packed, fp4_strided, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# RMSNorm (M, N) kernel-selection gate (CPU-only logic; no kernel launch).
# Verifies the routing in RMSNorm._ws_kernel_eligible: the warp-specialized
# fused_add_rms_norm_quant ("ws") serves only the large, contiguous, rank-2
# residual edge with an in-range hidden dim; everything else (small M,
# residual-less, row-strided, out-of-range N) uses the reduce_fusion kernel.
# ---------------------------------------------------------------------------


def _make_rmsnorm(n: int):
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm

    # device="meta" so no allocation / GPU is needed; the gate only reads
    # tensor shapes and contiguity, never data.
    return RMSNorm(hidden_size=n, eps=1e-6, dtype=torch.bfloat16, device="meta")


@pytest.mark.parametrize(
    "m,n,residual,contiguous,expect_ws",
    [
        # Large, contiguous, in-range N, M >= threshold (4096) -> ws.
        (4096, 7168, True, True, True),
        (8192, 8192, True, True, True),
        # M below the threshold -> reduce_fusion.
        (2048, 7168, True, True, False),
        (1, 7168, True, True, False),
        # Residual-less edge (q_a) -> reduce_fusion (no ws equivalent).
        (8192, 7168, False, True, False),
        # N below ws minimum (2048) -> reduce_fusion (q_a's 1536 lands here).
        (8192, 1536, True, True, False),
        # N above ws maximum -> reduce_fusion.
        (8192, 20480, True, True, False),
        # Row-strided (non-contiguous) input -> reduce_fusion.
        (8192, 7168, True, False, False),
    ],
)
def test_rmsnorm_ws_kernel_gate(m, n, residual, contiguous, expect_ws):
    norm = _make_rmsnorm(n)
    hs = torch.empty((m, n), dtype=torch.bfloat16, device="meta")
    res = torch.empty((m, n), dtype=torch.bfloat16, device="meta") if residual else None
    if not contiguous:
        # A column slice of a wider buffer: unit-stride last dim, larger pitch.
        hs = torch.empty((m, n + 32), dtype=torch.bfloat16, device="meta")[:, :n]
        assert not hs.is_contiguous()
    assert norm._ws_kernel_eligible(hs, res) is expect_ws
