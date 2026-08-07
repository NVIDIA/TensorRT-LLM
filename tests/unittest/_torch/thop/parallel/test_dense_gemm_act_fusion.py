# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the dense NVFP4 GEMM + activation epilogue-fusion kernels.

Covers the dense (non-MoE) CuteDSL act-fusion ops, which share one kernel
(``dense_blockscaled_gemm_act_fusion.py``) dispatched on the activation type:

  GELU(tanh)  (MLP, non-gated, optional per-N bias):
    - ``cute_dsl_nvfp4_dense_gemm_gelu_blackwell``         (bf16 out)
    - ``cute_dsl_nvfp4_dense_gemm_gelu_fp4out_blackwell``  (fp4 out + SFC)
  SwiGLU      (GatedMLP, gated, bias-free):
    - ``cute_dsl_nvfp4_dense_gemm_swiglu_blackwell``        (bf16 out)
    - ``cute_dsl_nvfp4_dense_gemm_swiglu_fp4out_blackwell`` (fp4 out + SFC)

The GELU kernel computes ``gelu_tanh(alpha * (A @ B.T) + bias)``; SwiGLU computes
``up * silu(gate)`` over ``alpha * (A @ B.T)`` (gate/up interleaved at group 64).
The fp4-out variants additionally quantize to NVFP4 with scale-factor generation.
The reference uses ``nvfp4_gemm`` (fp4-precision GEMM matching the kernel operands)
followed by the activation. fp4-out cases use M >= 128 (the kernel's _FP4OUT_MIN_M
floor); the small-M (m=64) bf16-out cases exercise the path the MLP/GatedMLP
fall back to below that floor.
"""

import pytest
import torch
import torch.nn.functional as F
from utils.util import skip_pre_blackwell

from tensorrt_llm._torch.modules.fused_moe.quantization import interleave_linear_and_gate
from tensorrt_llm._torch.utils import swizzle_sf, unswizzle_sf
from tensorrt_llm.math_utils import pad_up

FP4_E2M1_MAX = 6.0
FP8_E4M3_MAX = 448.0
SF_VEC = 16


def _quantize_nvfp4(x_bf16: torch.Tensor):
    """Quantize [., K] bf16 -> (fp4 packed, swizzled SF, global_sf scalar)."""
    global_sf = x_bf16.abs().max().float() / (FP8_E4M3_MAX * FP4_E2M1_MAX)
    fp4, sf = torch.ops.trtllm.fp4_quantize(x_bf16, 1.0 / global_sf, SF_VEC, False, True)
    return fp4, sf, global_sf


def _gemm_ref(a, b, a_sf, b_sf, alpha):
    """alpha * (A @ B.T) in fp4 precision (bf16 out), full N."""
    return torch.ops.trtllm.nvfp4_gemm(
        a.view(torch.uint8), b.view(torch.uint8), a_sf, b_sf, alpha, torch.bfloat16
    ).float()


def _nibble_match(c, ref_fp4):
    """Fraction of fp4 nibbles within +/-1 (two nibbles per uint8 byte)."""
    c_u8 = c.view(torch.uint8).flatten().int()
    r_u8 = ref_fp4.view(torch.uint8).flatten().int()
    lo = ((c_u8 & 0xF) - (r_u8 & 0xF)).abs()
    hi = ((c_u8 >> 4) - (r_u8 >> 4)).abs()
    return ((lo <= 1).sum().item() + (hi <= 1).sum().item()) / (c_u8.numel() * 2)


def _sf_match(c_sf, ref_sf, m, n):
    """Fraction of scale factors within +/-1 on the valid (first m) rows."""
    c_un = unswizzle_sf(c_sf.view(-1), pad_up(m, 128), n)[:m, :].view(torch.uint8)
    r_un = unswizzle_sf(ref_sf.view(-1), pad_up(m, 128), n)[:m, :].view(torch.uint8)
    return ((c_un.int() - r_un.int()).abs() <= 1).sum().item() / c_un.numel()


# ---------------------------------------------------------------------------
# GELU (non-gated, optional bias)
# ---------------------------------------------------------------------------
@skip_pre_blackwell
@pytest.mark.parametrize("use_bias", [False, True])
@pytest.mark.parametrize("m, k, n", [(64, 256, 512), (128, 256, 512), (256, 512, 2048)])
def test_dense_gemm_gelu_bf16out(m: int, k: int, n: int, use_bias: bool):
    """bf16-out fused up-GEMM + bias + GELU(tanh). m=64 < _FP4OUT_MIN_M (fallback)."""
    torch.manual_seed(0)
    a_bf16 = torch.randint(-5, 5, (m, k), dtype=torch.int32, device="cuda").to(torch.bfloat16)
    b_bf16 = torch.randint(-5, 5, (n, k), dtype=torch.int32, device="cuda").to(torch.bfloat16)
    a, a_sf, a_gsf = _quantize_nvfp4(a_bf16)
    b, b_sf, b_gsf = _quantize_nvfp4(b_bf16)
    alpha = (a_gsf * b_gsf).view(1)
    bias = (torch.randn(n, dtype=torch.bfloat16, device="cuda") * 0.1) if use_bias else None

    gemm = _gemm_ref(a, b, a_sf, b_sf, alpha)
    ref = F.gelu(gemm + (bias.float() if bias is not None else 0.0), approximate="tanh")

    c = torch.ops.trtllm.cute_dsl_nvfp4_dense_gemm_gelu_blackwell(
        a, b, a_sf, b_sf, alpha, torch.bfloat16, bias
    )[..., :n]

    assert c.shape == (m, n) and c.dtype == torch.bfloat16
    ratio = torch.isclose(c.float(), ref, rtol=0.1, atol=0.05).float().mean().item()
    assert ratio > 0.90, f"bf16-out GELU match {ratio * 100:.2f}% < 90%"


@skip_pre_blackwell
@pytest.mark.parametrize("use_bias", [False, True])
@pytest.mark.parametrize("m, k, n", [(128, 256, 512), (256, 512, 2048)])
def test_dense_gemm_gelu_fp4out(m: int, k: int, n: int, use_bias: bool):
    """fp4-out fused up-GEMM + bias + GELU(tanh) + NVFP4 quant (m >= _FP4OUT_MIN_M)."""
    torch.manual_seed(0)
    a_bf16 = torch.randint(-5, 5, (m, k), dtype=torch.int32, device="cuda").to(torch.bfloat16)
    b_bf16 = torch.randint(-5, 5, (n, k), dtype=torch.int32, device="cuda").to(torch.bfloat16)
    a, a_sf, a_gsf = _quantize_nvfp4(a_bf16)
    b, b_sf, b_gsf = _quantize_nvfp4(b_bf16)
    alpha = (a_gsf * b_gsf).view(1)
    bias = (torch.randn(n, dtype=torch.bfloat16, device="cuda") * 0.1) if use_bias else None

    gemm = _gemm_ref(a, b, a_sf, b_sf, alpha)
    ref = F.gelu(gemm + (bias.float() if bias is not None else 0.0), approximate="tanh").to(
        torch.bfloat16
    )
    norm_const = (1.0 / (ref.abs().max().float() / (FP8_E4M3_MAX * FP4_E2M1_MAX))).view(1)
    ref_fp4, ref_sf = torch.ops.trtllm.fp4_quantize(ref, norm_const, SF_VEC, False, True)

    c, c_sf = torch.ops.trtllm.cute_dsl_nvfp4_dense_gemm_gelu_fp4out_blackwell(
        a, b, a_sf, b_sf, alpha, norm_const, bias
    )

    assert c.shape == (m, n // 2)
    assert _nibble_match(c, ref_fp4) > 0.95, "fp4-out GELU nibble match < 95%"
    assert _sf_match(c_sf, ref_sf, m, n) > 0.95, "fp4-out GELU SF match < 95%"


# ---------------------------------------------------------------------------
# SwiGLU (gated, bias-free). Weight is [2*inter, k] (linear||gate); the kernel
# takes it gate/up-interleaved at group 64 (as GatedMLP stores it), the
# reference uses the non-interleaved layout and chunks it.
# ---------------------------------------------------------------------------
def _swiglu_weight(inter: int, k: int):
    """Build [2*inter, k] NVFP4 weight; return (non-interleaved, interleaved) forms."""
    w = torch.randint(-5, 5, (2 * inter, k), dtype=torch.int32, device="cuda").to(torch.bfloat16)
    b, b_sf, b_gsf = _quantize_nvfp4(w)
    n2 = 2 * inter
    b_il = interleave_linear_and_gate(b.view(torch.uint8), group_size=64, dim=0).view(b.dtype)
    b_sf_un = unswizzle_sf(b_sf, n2, k)
    b_sf_il = swizzle_sf(interleave_linear_and_gate(b_sf_un, group_size=64, dim=0), n2, k)
    return b, b_sf, b_il, b_sf_il, b_gsf


def _swiglu_ref(a, b, a_sf, b_sf, alpha):
    """up * silu(gate) over alpha*(A@B.T); [linear, gate] = chunk(gemm, 2)."""
    gemm = _gemm_ref(a, b, a_sf, b_sf, alpha)
    up, gate = gemm.chunk(2, dim=-1)
    return up * F.silu(gate)


@skip_pre_blackwell
@pytest.mark.parametrize("m, k, inter", [(64, 256, 512), (128, 256, 512), (256, 512, 1024)])
def test_dense_gemm_swiglu_bf16out(m: int, k: int, inter: int):
    """bf16-out fused gate-up GEMM + SwiGLU. m=64 < _FP4OUT_MIN_M (fallback)."""
    torch.manual_seed(0)
    a_bf16 = torch.randint(-5, 5, (m, k), dtype=torch.int32, device="cuda").to(torch.bfloat16)
    a, a_sf, a_gsf = _quantize_nvfp4(a_bf16)
    b, b_sf, b_il, b_sf_il, b_gsf = _swiglu_weight(inter, k)
    alpha = (a_gsf * b_gsf).view(1)

    ref = _swiglu_ref(a, b, a_sf, b_sf, alpha)

    c = torch.ops.trtllm.cute_dsl_nvfp4_dense_gemm_swiglu_blackwell(
        a, b_il, a_sf, b_sf_il, alpha, torch.bfloat16
    )[..., :inter]

    assert c.shape == (m, inter) and c.dtype == torch.bfloat16
    ratio = torch.isclose(c.float(), ref, rtol=0.1, atol=0.05).float().mean().item()
    assert ratio > 0.90, f"bf16-out SwiGLU match {ratio * 100:.2f}% < 90%"


@skip_pre_blackwell
@pytest.mark.parametrize("m, k, inter", [(128, 256, 512), (256, 512, 1024)])
def test_dense_gemm_swiglu_fp4out(m: int, k: int, inter: int):
    """fp4-out fused gate-up GEMM + SwiGLU + NVFP4 quant (m >= _FP4OUT_MIN_M)."""
    torch.manual_seed(0)
    a_bf16 = torch.randint(-5, 5, (m, k), dtype=torch.int32, device="cuda").to(torch.bfloat16)
    a, a_sf, a_gsf = _quantize_nvfp4(a_bf16)
    b, b_sf, b_il, b_sf_il, b_gsf = _swiglu_weight(inter, k)
    alpha = (a_gsf * b_gsf).view(1)

    ref = _swiglu_ref(a, b, a_sf, b_sf, alpha).to(torch.bfloat16)
    norm_const = (1.0 / (ref.abs().max().float() / (FP8_E4M3_MAX * FP4_E2M1_MAX))).view(1)
    ref_fp4, ref_sf = torch.ops.trtllm.fp4_quantize(ref, norm_const, SF_VEC, False, True)

    c, c_sf = torch.ops.trtllm.cute_dsl_nvfp4_dense_gemm_swiglu_fp4out_blackwell(
        a, b_il, a_sf, b_sf_il, alpha, norm_const
    )

    assert c.shape == (m, inter // 2)
    assert _nibble_match(c, ref_fp4) > 0.95, "fp4-out SwiGLU nibble match < 95%"
    assert _sf_match(c_sf, ref_sf, m, inter) > 0.95, "fp4-out SwiGLU SF match < 95%"


# ---------------------------------------------------------------------------
# fp4-out fallback / switching condition (module-level dispatch in MLP.forward).
# fp4-out is selected only when m >= _FP4OUT_MIN_M; below it the (still-fused)
# bf16-out path is used (the kernel's SFC epilogue is unsafe on a partial tile).
# This is HW-independent (no kernel is launched), so it is not Blackwell-gated.
# ---------------------------------------------------------------------------
def test_mlp_fp4out_min_m_switch():
    """MLP.forward picks fp4-out only at m >= _FP4OUT_MIN_M, else bf16-out fallback."""
    import types

    from tensorrt_llm._torch.modules.mlp import MLP
    from tensorrt_llm._torch.utils import gelu_tanh

    mlp = MLP(hidden_size=64, intermediate_size=128, bias=True, activation=gelu_tanh)
    # forward() reaches the fp4-out vs bf16-out switch only when the static
    # eligibility flags AND a runtime NVFP4 quant_method (exposing _input_prepare)
    # are present on up/down. Force all of that here; _fused_gelu is stubbed, so no
    # kernel runs and the test stays HW-independent (hence no skip_pre_blackwell).
    fake_qm = types.SimpleNamespace(_input_prepare=lambda *a, **k: None)
    mlp._use_fused_gelu = True
    mlp._use_fused_gelu_fp4out = True
    mlp.up_proj.quant_method = fake_qm
    down = torch.nn.Identity()  # skip the real down GEMM
    down.quant_method = fake_qm
    mlp.down_proj = down
    seen = {}

    def _spy(x, fp4_out=False):
        seen["fp4_out"] = fp4_out
        return torch.zeros(MLP._token_count(x), mlp.intermediate_size)

    mlp._fused_gelu = _spy

    assert MLP._FP4OUT_MIN_M == 128
    for m, expected in [(64, False), (127, False), (128, True), (256, True)]:
        mlp.forward(torch.randn(m, 64))
        assert seen["fp4_out"] == expected, (
            f"m={m}: expected fp4_out={expected}, got {seen['fp4_out']}"
        )


# ---------------------------------------------------------------------------
# The fused MLP flattens a rank-3 Fp4QuantizedTensor to 2D before the (2D-only)
# GEMM. A fused rmsnorm hands it [B, S, D/2]; the path previously flattened only
# plain tensors, tripping `assert inputs[0].dim() == 2`.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["gelu", "swiglu"])
def test_fused_act_flattens_3d_fp4_input(kind):
    """Fused NVFP4 MLP flattens a rank-3 Fp4QuantizedTensor to 2D before the GEMM."""
    import types

    from tensorrt_llm._torch.utils import Fp4QuantizedTensor, gelu_tanh

    B, S, K = 2, 8, 64
    if kind == "gelu":
        from tensorrt_llm._torch.modules.mlp import MLP

        mod = MLP(hidden_size=K, intermediate_size=128, bias=False, activation=gelu_tanh)
        proj = mod.up_proj
        run = lambda x: mod._fused_gelu(x)  # noqa: E731
    else:
        from tensorrt_llm._torch.modules.gated_mlp import GatedMLP

        mod = GatedMLP(hidden_size=K, intermediate_size=128, bias=False)
        proj = mod.gate_up_proj
        run = lambda x: mod._fused_gate_up_swiglu(x)  # noqa: E731

    captured = {}

    class _Stop(Exception):
        pass

    def fake_input_prepare(module, x):
        captured["dim"] = x.fp4_tensor.dim() if isinstance(x, Fp4QuantizedTensor) else x.dim()
        captured["shape"] = (
            tuple(x.fp4_tensor.shape) if isinstance(x, Fp4QuantizedTensor) else tuple(x.shape)
        )
        raise _Stop  # short-circuit before the real (HW-only) GEMM

    proj.quant_method = types.SimpleNamespace(_input_prepare=fake_input_prepare)

    fp4_3d = Fp4QuantizedTensor(
        fp4_tensor=torch.zeros(B, S, K // 2, dtype=torch.uint8),
        scaling_factor=torch.zeros(B * S, K // 16, dtype=torch.uint8),
    )
    with pytest.raises(_Stop):
        run(fp4_3d)

    assert captured["dim"] == 2, (
        f"{kind}: rank-3 Fp4QuantizedTensor not flattened before the GEMM "
        f"(kernel would see rank {captured['dim']})"
    )
    assert captured["shape"] == (B * S, K // 2)
