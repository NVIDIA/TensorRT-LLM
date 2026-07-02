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
"""
Tests for trtllm::fused_dit_layernorm_shift_scale[_quant].

Covers all 6 compile-time instantiations:
  (has_ln_affine=F, has_modulation=F, has_quant=F) -- plain LN -> bf16
  (has_ln_affine=F, has_modulation=F, has_quant=T) -- plain LN -> FP4
  (has_ln_affine=T, has_modulation=F, has_quant=F) -- LN+affine -> bf16  (norm2)
  (has_ln_affine=T, has_modulation=F, has_quant=T) -- LN+affine -> FP4  (norm2)
  (has_ln_affine=F, has_modulation=T, has_quant=F) -- LN+AdaLN -> bf16  (norm1, norm3)
  (has_ln_affine=F, has_modulation=T, has_quant=T) -- LN+AdaLN -> FP4  (norm1, norm3)
"""

import pytest
import torch
import torch.nn.functional as F
from utils.util import skip_pre_blackwell

D = 5120
EPS = 1e-6


# ---------------------------------------------------------------------------
# Float32 reference implementations
# ---------------------------------------------------------------------------


@torch.inference_mode()
def _ref_layernorm_adaln(x, scale_msa, shift_msa, seq_len_per_batch, eps):
    """y = (1 + scale_msa) * x_hat + shift_msa, broadcast over seq."""
    M, _ = x.shape
    xf = x.float()
    mean = xf.mean(dim=-1, keepdim=True)
    var = xf.var(dim=-1, keepdim=True, unbiased=False)
    x_hat = (xf - mean) / (var + eps).sqrt()
    batch_idx = torch.arange(M, device=x.device) // seq_len_per_batch
    return (1.0 + scale_msa.float()[batch_idx]) * x_hat + shift_msa.float()[batch_idx]


@torch.inference_mode()
def _ref_layernorm_affine(x, ln_weight, ln_bias, eps):
    """y = ln_weight * x_hat + ln_bias."""
    xf = x.float()
    mean = xf.mean(dim=-1, keepdim=True)
    var = xf.var(dim=-1, keepdim=True, unbiased=False)
    x_hat = (xf - mean) / (var + eps).sqrt()
    return ln_weight.float() * x_hat + ln_bias.float()


@torch.inference_mode()
def _ref_layernorm_plain(x, eps):
    xf = x.float()
    mean = xf.mean(dim=-1, keepdim=True)
    var = xf.var(dim=-1, keepdim=True, unbiased=False)
    return (xf - mean) / (var + eps).sqrt()


def _make_inputs(M, B, has_ln_affine, has_modulation, seed=42):
    torch.manual_seed(seed)
    device = torch.device("cuda")
    x = torch.randn(M, D, device=device).to(torch.bfloat16)
    ln_weight = ln_bias = scale_msa = shift_msa = None
    seq_len_per_batch = M  # default (B=1 case)
    if has_ln_affine:
        ln_weight = (torch.ones(D, device=device) + torch.randn(D, device=device) * 0.1).to(
            torch.bfloat16
        )
        ln_bias = (torch.randn(D, device=device) * 0.1).to(torch.bfloat16)
        ref = _ref_layernorm_affine(x, ln_weight, ln_bias, EPS)
    elif has_modulation:
        seq_len_per_batch = M // B
        scale_msa = (torch.randn(B, D, device=device) * 0.2).to(torch.bfloat16)
        shift_msa = (torch.randn(B, D, device=device) * 0.2).to(torch.bfloat16)
        ref = _ref_layernorm_adaln(x, scale_msa, shift_msa, seq_len_per_batch, EPS)
    else:
        ref = _ref_layernorm_plain(x, EPS)
    return x, ln_weight, ln_bias, scale_msa, shift_msa, seq_len_per_batch, ref


# ---------------------------------------------------------------------------
# BF16 correctness: all modes × several (M, B) shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "M,B",
    [
        (1, 1),  # single row edge case
        (4, 4),  # tpb=1: single token per batch element
        (32, 2),  # non-multiple of warp; B=2
        (128, 4),  # nominal Wan batch
        (512, 1),  # large M, B=1
    ],
)
@pytest.mark.parametrize(
    "has_ln_affine,has_modulation",
    [
        (False, False),
        (True, False),
        (False, True),
    ],
)
def test_bf16_correctness(M, B, has_ln_affine, has_modulation):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if has_modulation and M % B != 0:
        pytest.skip("M not divisible by B")

    x, ln_w, ln_b, scale_msa, shift_msa, seq_len, ref = _make_inputs(
        M, B, has_ln_affine, has_modulation
    )
    out = torch.ops.trtllm.fused_dit_layernorm_shift_scale(
        x, ln_w, ln_b, scale_msa, shift_msa, seq_len, EPS
    )

    assert out.shape == (M, D)
    assert out.dtype == torch.bfloat16

    # bf16 accumulation gives ~1e-3 max absolute error vs fp32 reference.
    torch.testing.assert_close(out.float(), ref.to(torch.bfloat16).float(), rtol=2e-2, atol=2e-2)


# ---------------------------------------------------------------------------
# Batch modulation correctness: each batch element gets its own scale/shift.
# With very different modulators the output rows must differ significantly.
# ---------------------------------------------------------------------------


def test_adaln_batch_modulation_correctness():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    M, B = 128, 4
    S = M // B
    device = torch.device("cuda")
    torch.manual_seed(0)

    x = torch.randn(M, D, device=device).to(torch.bfloat16)
    # Very distinct scale/shift per batch element.
    scale_msa = torch.stack([torch.ones(D, device=device) * (i * 0.5) for i in range(B)]).to(
        torch.bfloat16
    )
    shift_msa = torch.stack([torch.ones(D, device=device) * (i * 1.0) for i in range(B)]).to(
        torch.bfloat16
    )

    out = torch.ops.trtllm.fused_dit_layernorm_shift_scale(
        x, None, None, scale_msa, shift_msa, S, EPS
    )
    ref = _ref_layernorm_adaln(x, scale_msa, shift_msa, S, EPS).to(torch.bfloat16)

    # Cross-batch: rows from different batch elements should be measurably different.
    out_b0 = out[:S].float().mean()
    out_b1 = out[S : 2 * S].float().mean()
    assert abs(out_b0.item() - out_b1.item()) > 0.1, (
        "Batch elements have indistinguishable outputs — modulation broadcast may be broken"
    )

    # Still matches reference row-by-row.
    torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=2e-2)


# ---------------------------------------------------------------------------
# FP4 correctness: feed quant output through nvfp4_gemm, cosine_sim > 0.98.
# Pattern from Yiyun's LTX-2 test_fused_dit_gate_resid_norm_shift_scale.py.
# ---------------------------------------------------------------------------


@skip_pre_blackwell
@pytest.mark.parametrize(
    "M,B",
    [
        (1, 1),  # M=1: maximum SF over-allocation (1 row padded to 128)
        (128, 4),
        (256, 1),
    ],
)
@pytest.mark.parametrize(
    "has_ln_affine,has_modulation",
    [
        (False, False),
        (True, False),
        (False, True),
    ],
)
def test_fp4_quant_gemm_correctness(M, B, has_ln_affine, has_modulation):
    if has_modulation and M % B != 0:
        pytest.skip("M not divisible by B")

    device = torch.device("cuda")
    out_dim = 1024
    x, ln_w, ln_b, scale_msa, shift_msa, seq_len, _ = _make_inputs(
        M, B, has_ln_affine, has_modulation, seed=7
    )
    # Scale x to moderate magnitude so FP4 clipping is minimal.
    x = (x * 0.5).to(torch.bfloat16)

    W = torch.randn(out_dim, D, dtype=torch.bfloat16, device=device) * 0.05

    # bf16 reference: unfused LN + F.linear.
    if has_ln_affine:
        out_bf16 = torch.ops.trtllm.fused_dit_layernorm_shift_scale(
            x, ln_w, ln_b, None, None, seq_len, EPS
        )
    elif has_modulation:
        out_bf16 = torch.ops.trtllm.fused_dit_layernorm_shift_scale(
            x, None, None, scale_msa, shift_msa, seq_len, EPS
        )
    else:
        out_bf16 = torch.ops.trtllm.fused_dit_layernorm_shift_scale(
            x, None, None, None, None, seq_len, EPS
        )
    D_ref = F.linear(out_bf16, W)

    sf_scale_x = (448.0 * 6.0) / out_bf16.abs().max().float()
    sf_scale_w = (448.0 * 6.0) / W.abs().max().float()

    y_fp4, y_sf = torch.ops.trtllm.fused_dit_layernorm_shift_scale_quant(
        x, ln_w, ln_b, scale_msa, shift_msa, sf_scale_x, seq_len, EPS
    )

    assert y_fp4.shape == (M, D // 2)
    assert y_fp4.dtype == torch.uint8
    assert y_sf.dtype == torch.uint8

    W_fp4, W_sf = torch.ops.trtllm.fp4_quantize(W, sf_scale_w, 16)
    alpha = 1.0 / (sf_scale_x * sf_scale_w).float()
    C = torch.ops.trtllm.nvfp4_gemm(y_fp4, W_fp4, y_sf, W_sf, alpha, torch.bfloat16)

    cos_sim = F.cosine_similarity(C.flatten().float(), D_ref.flatten().float(), dim=0).item()
    assert cos_sim > 0.98, (
        f"FP4 GEMM cosine similarity {cos_sim:.4f} < 0.98 "
        f"(M={M} B={B} has_ln_affine={has_ln_affine} has_modulation={has_modulation})"
    )


# ---------------------------------------------------------------------------
# Input validation: the op must reject invalid argument combinations.
# ---------------------------------------------------------------------------


def test_validation_mutual_exclusivity():
    """Both ln_weight and scale_msa provided → TORCH_CHECK error."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = torch.device("cuda")
    x = torch.randn(32, D, device=device).to(torch.bfloat16)
    ln_w = torch.ones(D, device=device).to(torch.bfloat16)
    ln_b = torch.zeros(D, device=device).to(torch.bfloat16)
    scale = torch.zeros(1, D, device=device).to(torch.bfloat16)
    shift = torch.zeros(1, D, device=device).to(torch.bfloat16)
    with pytest.raises(RuntimeError, match="mutually exclusive"):
        torch.ops.trtllm.fused_dit_layernorm_shift_scale(x, ln_w, ln_b, scale, shift, 32, EPS)


def test_validation_wrong_hidden_dim():
    """D != 5120 must raise a descriptive error."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = torch.device("cuda")
    x = torch.randn(32, 4096, device=device).to(torch.bfloat16)
    with pytest.raises(RuntimeError):
        torch.ops.trtllm.fused_dit_layernorm_shift_scale(x, None, None, None, None, 32, EPS)


def test_validation_wrong_dtype():
    """float32 input must raise a descriptive error."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = torch.device("cuda")
    x = torch.randn(32, D, device=device)  # float32
    with pytest.raises(RuntimeError):
        torch.ops.trtllm.fused_dit_layernorm_shift_scale(x, None, None, None, None, 32, EPS)


def test_validation_non_divisible_seq_len():
    """M not divisible by seq_len_per_batch must raise for AdaLN path."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = torch.device("cuda")
    x = torch.randn(33, D, device=device).to(torch.bfloat16)
    scale = torch.zeros(3, D, device=device).to(torch.bfloat16)
    shift = torch.zeros(3, D, device=device).to(torch.bfloat16)
    # seq_len_per_batch=11 → B=3, 3*11=33 ✓; seq_len_per_batch=10 → 33%10≠0 ✗
    with pytest.raises(RuntimeError):
        torch.ops.trtllm.fused_dit_layernorm_shift_scale(x, None, None, scale, shift, 10, EPS)


def test_validation_non_contiguous_x():
    """Non-contiguous x must raise a RuntimeError mentioning 'contiguous'."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = torch.device("cuda")
    # Create a strided (non-contiguous) view by slicing a wider tensor.
    base = torch.randn(32, D * 2, device=device).to(torch.bfloat16)
    x_strided = base[:, :D]
    assert not x_strided.is_contiguous()
    with pytest.raises(RuntimeError, match=r"contiguous"):
        torch.ops.trtllm.fused_dit_layernorm_shift_scale(x_strided, None, None, None, None, 32, EPS)


def test_validation_ln_weight_without_ln_bias():
    """ln_weight provided without ln_bias must raise (partial affine pair)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = torch.device("cuda")
    x = torch.randn(32, D, device=device).to(torch.bfloat16)
    ln_w = torch.ones(D, device=device).to(torch.bfloat16)
    with pytest.raises(RuntimeError, match="ln_weight and ln_bias must both be provided together"):
        torch.ops.trtllm.fused_dit_layernorm_shift_scale(x, ln_w, None, None, None, 32, EPS)


def test_validation_scale_msa_without_shift_msa():
    """scale_msa provided without shift_msa must raise (partial modulation pair)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = torch.device("cuda")
    x = torch.randn(32, D, device=device).to(torch.bfloat16)
    scale = torch.zeros(1, D, device=device).to(torch.bfloat16)
    with pytest.raises(
        RuntimeError, match="scale_msa and shift_msa must both be provided together"
    ):
        torch.ops.trtllm.fused_dit_layernorm_shift_scale(x, None, None, scale, None, 32, EPS)


def test_validation_non_positive_seq_len():
    """seq_len_per_batch <= 0 on the AdaLN path must raise."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = torch.device("cuda")
    x = torch.randn(32, D, device=device).to(torch.bfloat16)
    scale = torch.zeros(1, D, device=device).to(torch.bfloat16)
    shift = torch.zeros(1, D, device=device).to(torch.bfloat16)
    with pytest.raises(RuntimeError, match="seq_len_per_batch must be positive"):
        torch.ops.trtllm.fused_dit_layernorm_shift_scale(x, None, None, scale, shift, 0, EPS)


@skip_pre_blackwell
def test_validation_sf_scale_wrong_dtype():
    """sf_scale with a non-float32 dtype must raise a descriptive error."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = torch.device("cuda")
    x = torch.randn(32, D, device=device).to(torch.bfloat16)
    sf_scale = torch.ones(1, device=device).to(torch.bfloat16)  # wrong dtype
    with pytest.raises(RuntimeError, match="dtype"):
        torch.ops.trtllm.fused_dit_layernorm_shift_scale_quant(
            x, None, None, None, None, sf_scale, 32, EPS
        )


@skip_pre_blackwell
def test_validation_sf_scale_non_scalar():
    """sf_scale with more than one element must raise a descriptive error."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = torch.device("cuda")
    x = torch.randn(32, D, device=device).to(torch.bfloat16)
    sf_scale = torch.ones(2, device=device)  # non-scalar
    with pytest.raises(RuntimeError, match="scalar tensor"):
        torch.ops.trtllm.fused_dit_layernorm_shift_scale_quant(
            x, None, None, None, None, sf_scale, 32, EPS
        )


# ---------------------------------------------------------------------------
# Production shapes: Wan 2.2 T2V-A14B at both default resolutions.
#
# Model: 14B, hidden_size=5120, patch_size=[1,2,2], vae_spatial=8, vae_temporal=4.
# Default resolutions (pipeline_wan.py): 480×832 and 720×1280, 81 frames.
#
# Token counts (latent → patchify):
#   480p / 81-frame: latent=(21, 60, 104) → 21×30×52 = 32760 tokens  ← real default
#   720p / 81-frame: latent=(21, 90, 160) → 21×45×80 = 75600 tokens  ← real default
#   480p /  1-frame: latent=( 1, 60, 104) →  1×30×52 =  1560 tokens  ← CI-speed subset
#
# Three modes match the three norm sites in WanTransformerBlock:
#   plain LN          — standalone norm (no modulation)
#   AdaLN (no affine) — norm1 (pre-attn) and norm3 (pre-cross-attn)
#   affine LN         — norm2 (pre-FFN, learned weight/bias)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "M,B,has_ln_affine,has_modulation",
    [
        # 720p 81-frame — all three kernel modes
        (75600, 1, False, False),  # plain LN
        (75600, 1, False, True),  # AdaLN  (norm1 / norm3 sites)
        (75600, 1, True, False),  # affine (norm2 site)
        # 480p 81-frame (real default) — the two modulated modes
        (32760, 1, False, True),  # AdaLN  480p full
        (32760, 1, True, False),  # affine 480p full
        # 480p 1-frame (CI-speed subset)
        (1560, 1, False, True),  # AdaLN  480p 1-frame
        (1560, 1, True, False),  # affine 480p 1-frame
        # 480p 1-frame, B=2 — batch inference
        (3120, 2, False, True),  # AdaLN  B=2
    ],
)
def test_wan14b_production_shape_bf16(M, B, has_ln_affine, has_modulation):
    """BF16 correctness at Wan 14B production token counts across all norm-site modes."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    x, ln_w, ln_b, scale_msa, shift_msa, seq_len, ref = _make_inputs(
        M, B, has_ln_affine, has_modulation, seed=99
    )
    out = torch.ops.trtllm.fused_dit_layernorm_shift_scale(
        x, ln_w, ln_b, scale_msa, shift_msa, seq_len, EPS
    )

    assert out.shape == (M, D)
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), ref.to(torch.bfloat16).float(), rtol=2e-2, atol=2e-2)


@skip_pre_blackwell
@pytest.mark.parametrize(
    "M,B,has_ln_affine,has_modulation",
    [
        # 480p 1-frame subset: avoids a slow M=32760 GEMM reference in CI while
        # still exercising real patchified token counts.
        (1560, 1, False, True),  # AdaLN  (norm1/norm3) — the dominant quantized site
        (1560, 1, True, False),  # affine (norm2)
        (3120, 2, False, True),  # AdaLN  B=2
    ],
)
def test_fp4_quant_production_scale(M, B, has_ln_affine, has_modulation):
    """FP4 GEMM quality at 480p/1-frame token count.

    Mirrors Yiyun's KA/KC-quant approach: feed fused-kernel fp4 output through
    nvfp4_gemm and verify cosine_similarity > 0.98 vs a bf16 F.linear reference.
    Full 480p (32760 tokens) and 720p shape/SF checks are in
    test_fp4_quant_720p_smoke and test_fp4_sf_allocation_boundaries.
    """
    device = torch.device("cuda")
    out_dim = 1024
    x, ln_w, ln_b, scale_msa, shift_msa, seq_len, _ = _make_inputs(
        M, B, has_ln_affine, has_modulation, seed=7
    )
    x = (x * 0.5).to(torch.bfloat16)
    W = torch.randn(out_dim, D, dtype=torch.bfloat16, device=device) * 0.05

    out_bf16 = torch.ops.trtllm.fused_dit_layernorm_shift_scale(
        x, ln_w, ln_b, scale_msa, shift_msa, seq_len, EPS
    )
    D_ref = F.linear(out_bf16, W)

    sf_scale_x = (448.0 * 6.0) / out_bf16.abs().max().float()
    sf_scale_w = (448.0 * 6.0) / W.abs().max().float()

    y_fp4, y_sf = torch.ops.trtllm.fused_dit_layernorm_shift_scale_quant(
        x, ln_w, ln_b, scale_msa, shift_msa, sf_scale_x, seq_len, EPS
    )

    W_fp4, W_sf = torch.ops.trtllm.fp4_quantize(W, sf_scale_w, 16)
    alpha = 1.0 / (sf_scale_x * sf_scale_w).float()
    C = torch.ops.trtllm.nvfp4_gemm(y_fp4, W_fp4, y_sf, W_sf, alpha, torch.bfloat16)

    cos_sim = F.cosine_similarity(C.flatten().float(), D_ref.flatten().float(), dim=0).item()
    assert cos_sim > 0.98, (
        f"FP4 GEMM cosine {cos_sim:.4f} < 0.98 "
        f"(M={M} B={B} has_ln_affine={has_ln_affine} has_modulation={has_modulation})"
    )


# ---------------------------------------------------------------------------
# Production modulator pattern: Wan extracts scale/shift via
#   .chunk(6, dim=2) → .squeeze(2) → .reshape(B, -1)
# The reshape on a non-contiguous chunk output returns a contiguous copy,
# so scale_msa/shift_msa are always contiguous by the time they reach our op.
# This test verifies the full extraction pipeline produces correct results
# and that the op rejects genuinely non-contiguous modulators.
# ---------------------------------------------------------------------------


def test_adaln_production_modulator_extraction():
    """AdaLN with scale/shift extracted via the Wan chunk→squeeze→reshape pipeline."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    B, T, S = 2, 1, 64
    M = B * S
    K = 6  # Wan uses 6 modulation slots
    device = torch.device("cuda")
    torch.manual_seed(5)

    x = torch.randn(M, D, device=device).to(torch.bfloat16)

    # Mimic Wan WanBlock: temb [B, T, K, D] → chunk → squeeze → reshape
    temb = torch.randn(B, T, K, D, device=device).to(torch.bfloat16)
    chunks = temb.chunk(K, dim=2)  # K tensors of [B, T, 1, D]
    shift_msa_raw = chunks[0].squeeze(2)  # [B, T, D], non-contiguous strides
    scale_msa_raw = chunks[1].squeeze(2)
    # reshape may return a non-contiguous view; .contiguous() mirrors the wrapper behavior
    scale_msa = scale_msa_raw.reshape(B, -1).contiguous()
    shift_msa = shift_msa_raw.reshape(B, -1).contiguous()
    assert scale_msa.is_contiguous()
    assert scale_msa.shape == (B, D)  # T=1 so T*D == D

    out = torch.ops.trtllm.fused_dit_layernorm_shift_scale(
        x, None, None, scale_msa, shift_msa, S, EPS
    )
    ref = _ref_layernorm_adaln(x, scale_msa, shift_msa, S, EPS).to(torch.bfloat16)

    assert out.shape == (M, D)
    torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=2e-2)


def test_adaln_rejects_non_contiguous_modulators():
    """Op must raise when scale_msa/shift_msa are non-contiguous (before reshape)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    B, S = 4, 32
    M = B * S
    device = torch.device("cuda")
    x = torch.randn(M, D, device=device).to(torch.bfloat16)

    # chunk produces non-contiguous views
    temb = torch.randn(B, 6, D, device=device).to(torch.bfloat16)
    chunks = temb.chunk(6, dim=1)
    scale_nc = chunks[0].squeeze(1)  # [B, D], non-contiguous
    shift_nc = chunks[1].squeeze(1)
    assert not scale_nc.is_contiguous()

    with pytest.raises(RuntimeError, match=r"contiguous"):
        torch.ops.trtllm.fused_dit_layernorm_shift_scale(x, None, None, scale_nc, shift_nc, S, EPS)


# ---------------------------------------------------------------------------
# FP4 smoke tests at full production shapes — shape / SF-size checks only.
# The [M, D] × [out_dim, D]^T GEMM reference is prohibitively slow at M=75600,
# so these tests verify the kernel runs without error and emits the correct
# output shapes; cosine correctness is covered by test_fp4_quant_production_scale.
# ---------------------------------------------------------------------------


@skip_pre_blackwell
@pytest.mark.parametrize(
    "M,B,has_ln_affine,has_modulation",
    [
        # 720p 81-frame — dominant FP4 quantized sites
        (75600, 1, False, True),  # AdaLN  (norm1/norm3)
        (75600, 1, True, False),  # affine (norm2)
        # 480p 81-frame — full default resolution
        (32760, 1, False, True),  # AdaLN  480p full
        (32760, 1, True, False),  # affine 480p full
    ],
)
def test_fp4_quant_production_shape_smoke(M, B, has_ln_affine, has_modulation):
    """FP4 kernel output shapes are correct at full Wan 2.2 production token counts."""
    device = torch.device("cuda")
    x, ln_w, ln_b, scale_msa, shift_msa, seq_len, _ = _make_inputs(
        M, B, has_ln_affine, has_modulation, seed=11
    )
    x = (x * 0.5).to(torch.bfloat16)

    sf_scale = torch.tensor(1.0, dtype=torch.float32, device=device)
    y_fp4, y_sf = torch.ops.trtllm.fused_dit_layernorm_shift_scale_quant(
        x, ln_w, ln_b, scale_msa, shift_msa, sf_scale, seq_len, EPS
    )

    assert y_fp4.shape == (M, D // 2), f"y_fp4 shape mismatch: {y_fp4.shape}"
    assert y_fp4.dtype == torch.uint8

    sf_cols = D // 16  # 320 for D=5120
    expected_sf_numel = (M + 127) // 128 * 128 * sf_cols
    assert y_sf.numel() == expected_sf_numel, (
        f"y_sf numel {y_sf.numel()} != expected {expected_sf_numel} (M={M})"
    )


# ---------------------------------------------------------------------------
# SF allocation boundary test: verifies the (M+127)/128*128 * (D/16) formula
# at the 128-row tile boundary.  M=128 is the only case with no padding;
# M=127 and M=129 straddle it on either side.
# ---------------------------------------------------------------------------


@skip_pre_blackwell
@pytest.mark.parametrize("M", [1, 127, 128, 129, 255, 256])
def test_fp4_sf_allocation_boundaries(M):
    """SF tensor size matches the inline formula at every 128-row tile boundary."""
    device = torch.device("cuda")
    torch.manual_seed(M)
    x = (torch.randn(M, D, device=device) * 0.5).to(torch.bfloat16)
    sf_scale = torch.tensor(1.0, dtype=torch.float32, device=device)

    y_fp4, y_sf = torch.ops.trtllm.fused_dit_layernorm_shift_scale_quant(
        x, None, None, None, None, sf_scale, M, EPS
    )

    assert y_fp4.shape == (M, D // 2)

    sf_cols = D // 16  # 320 for D=5120; already divisible by 4
    expected_sf_numel = (M + 127) // 128 * 128 * sf_cols
    assert y_sf.numel() == expected_sf_numel, (
        f"M={M}: y_sf.numel()={y_sf.numel()} expected={expected_sf_numel}"
    )
