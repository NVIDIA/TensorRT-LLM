# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1's shared expert on FP8 block-scale GEMMs, as stock runs it (`FP8BlockScalesLinearMethod`).

The checkpoint stores the shared expert as E4M3 with 32x32 UE8M0 scales; stock requantizes each weight to 128x128 block
scales at load (`_requantize_block128`, copied here as `requantize_block128`: power-of-two tile scales, so E4M3 values
in the normal range are kept exactly) and runs `x -> fp8 1x128 -> GEMM(gate | up) -> clamped SwiGLU -> fp8 1x128 ->
GEMM(down)` on its SM100 DeepGEMM path (`shared_expert_fp8_deep_gemm`), instead of bf16 GEMMs over a dequantized copy.
Few-row inputs take stock's small-M GEMMs there, as its `FP8BlockScalesLinearMethod` and `GatedMLP` pick them.
"""

from __future__ import annotations

import functools
from typing import Tuple

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 -- registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ..activation.clamped_swiglu import clamped_swiglu

_BLOCK = 128
_E4M3_MAX = 448.0


def _pow2_tile_scale(amax: torch.Tensor) -> torch.Tensor:
    """Per-tile multiplier `S = 2**ceil(log2(amax / 448))` (1 for an all-zero tile)."""
    ratio = amax / _E4M3_MAX
    return torch.where(ratio > 0, torch.exp2(torch.ceil(torch.log2(ratio))), torch.ones_like(ratio))


def requantize_block128(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """A dense `[M, N]` weight (any float dtype) -> `(E4M3 [M, N], fp32 [M/128, N/128] multipliers)`, `w ~= q * S`.

    A setup helper, not an entry (stock's `_requantize_block128`).
    """
    m, n = weight.shape
    tm, tn = -(-m // _BLOCK), -(-n // _BLOCK)
    w = torch.nn.functional.pad(weight.float(), (0, tn * _BLOCK - n, 0, tm * _BLOCK - m))
    scale = _pow2_tile_scale(w.reshape(tm, _BLOCK, tn, _BLOCK).abs().amax(dim=(1, 3)))
    q = (w / scale.repeat_interleave(_BLOCK, 0).repeat_interleave(_BLOCK, 1))[:m, :n].to(
        torch.float8_e4m3fn
    )
    return q.contiguous(), scale.contiguous()


def _dequant(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    m, n = q.shape
    return q.float() * s.repeat_interleave(_BLOCK, 0)[:m].repeat_interleave(_BLOCK, 1)[:, :n]


_CELLS = (
    Cell(
        why="a decode-sized step: 8 rows, H 5120 / I 2304",
        spec=dict(n=8, hidden=5120, inter=2304, limit=10.0),
    ),
    Cell(why="a context chunk: 4096 rows", spec=dict(n=4096, hidden=5120, inter=2304, limit=10.0)),
    Cell(
        why="a large context step: 16384 rows",
        spec=dict(n=16384, hidden=5120, inter=2304, limit=10.0),
    ),
)
_NOTE = """
From the context chunk on, both activations are quantized to FP8 per 1x128 block, as stock's linear does; the
decode-sized step takes stock's small-M GEMMs (bf16 or MXFP8 1x32 activations). The gate is a relative RMS error of 3%
against the fp32 product over the dequantized weights and unquantized activations.
"""


def _reference(x: torch.Tensor, w13, s13, w2, s2, limit: float) -> torch.Tensor:
    """`y = W2 (silu(clamp(W1 x)) * clamp(W3 x))` in fp32 over the dequantized weights (`requantize_block128`'s
    multipliers `s13` / `s2`), bf16 out: the shared-expert entries' reference."""
    inter = w2.shape[1]
    gu = x.float() @ _dequant(w13, s13).t()
    h = clamped_swiglu.reference(gu[:, :inter], gu[:, inter:], limit).float()
    return (h @ _dequant(w2, s2).t()).to(torch.bfloat16)


@functools.cache
def _low_m_range(n: int, k: int):
    """Stock's `Fp8BlockScaleLowMGemmRange` of an `[n, k]` weight (`FP8BlockScalesLinearMethod.transform_weights`)."""
    from tensorrt_llm._torch.modules.linear import Fp8BlockScaleLowMGemmRange

    return Fp8BlockScaleLowMGemmRange.for_weight(n, k)


def _linear(x: torch.Tensor, w: torch.Tensor, sf: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """bf16 `x @ (w * s).T` as stock's `FP8BlockScalesLinearMethod` runs a bf16 input on SM100 with DeepGEMM operands:
    the autotuned race of its small-M kernel (FP32 block scales `s`) where that race ran, else the low-M GEMM within
    the weight's low-M range, else `fp8_swap_ab_gemm`."""
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import fp8_block_scale_gemm_autotuned
    from tensorrt_llm._torch.modules.linear import (
        _fp8_block_scale_small_m_tuned,
        fp8_block_scale_ue8m0_gemm,
    )
    from tensorrt_llm._torch.utils import is_torch_compiling

    if not is_torch_compiling() and _fp8_block_scale_small_m_tuned(x, w, sf, s):
        return fp8_block_scale_gemm_autotuned(x, w, sf, s)
    return fp8_block_scale_ue8m0_gemm(x, w, sf, _low_m_range(*w.shape))


def deep_gemm_weight_scale(scale: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """fp32 power-of-two `[n/128, k/128]` multipliers (`requantize_block128`) -> the packed UE8M0 int32 layout DeepGEMM
    reads (stock's `transform_sf_into_required_layout(..., recipe=(1, 128, 128))`). A setup helper, not an entry."""
    from tensorrt_llm.quantization.utils.fp8_utils import transform_sf_into_required_layout

    return transform_sf_into_required_layout(scale, mn=n, k=k, recipe=(1, 128, 128), is_sfa=False)


class SharedExpertFp8DeepGemm(OpWrapper):
    """`y = W2 (silu(clamp(W1 x)) * clamp(W3 x))`, bf16 `[n, H]` in and out, as stock's `GatedMLP` runs it on its SM100
    DeepGEMM path (`FP8BlockScalesLinearMethod` with the DeepGEMM layout): `GEMM(gate | up)` on the bf16 rows, then
    either the clamped SwiGLU fused with the fp8 1x128 quantization and `fp8_prequantized_swap_ab_gemm(down)`, or, where
    the autotuner picked stock's small-M kernel for the down projection (`fp8_block_scale_swiglu_prefers_bf16`), a bf16
    SwiGLU and that kernel. Each bf16 GEMM takes stock's choice for its row count (`_linear`): the small-M kernel or the
    low-M GEMM on few rows (bf16 / MXFP8 1x32 activations), else the fp8 1x128 (UE8M0) quantization and DeepGEMM.

    `w13` E4M3 `[2I, H]` is `[w1 (gate) | w3 (up)]`, `w2` E4M3 `[H, I]` (`requantize_block128`); `s13` / `s2` their fp32
    power-of-two 128x128 multipliers (stock's `weight_scale_block128`), `sf13` / `sf2` the same in DeepGEMM's layout
    (`deep_gemm_weight_scale`). `limit` is the checkpoint's SwiGLU limit (0: unclamped).
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS = _CELLS
    note = _NOTE

    def __call__(self, x: torch.Tensor, w13, sf13, s13, w2, sf2, s2, limit: float) -> torch.Tensor:
        from tensorrt_llm._torch.custom_ops.torch_custom_ops import (
            fp8_block_scale_swiglu_prefers_bf16,
        )
        from tensorrt_llm._torch.modules.swiglu import swiglu

        lim = limit if limit > 0 else None
        gu = _linear(x if x.is_contiguous() else x.contiguous(), w13, sf13, s13)
        if fp8_block_scale_swiglu_prefers_bf16(gu, w2, sf2, s2, lim):
            return _linear(swiglu(gu, swiglu_limit=lim), w2, sf2, s2)
        h, h_sf = torch.ops.trtllm.silu_and_mul_fp8_quantize_1x128_packed_ue8m0(gu, lim, False)
        return torch.ops.trtllm.fp8_prequantized_swap_ab_gemm(
            h, h_sf, w2, sf2, disable_ue8m0_cast=True
        )

    def reference(self, x: torch.Tensor, w13, sf13, s13, w2, sf2, s2, limit: float) -> torch.Tensor:
        return _reference(x, w13, s13, w2, s2, limit)

    def is_valid(self, x: torch.Tensor, w13, sf13, s13, w2, sf2, s2, limit: float) -> None:
        n, h = x.shape
        i2 = w13.shape[0]
        assert x.dtype == torch.bfloat16 and h % _BLOCK == 0 and i2 % (2 * _BLOCK) == 0
        assert (
            w13.dtype == w2.dtype == torch.float8_e4m3fn
            and w13.shape == (i2, h)
            and w2.shape == (h, i2 // 2)
        )
        assert sf13.dtype == sf2.dtype == torch.int32
        assert s13.dtype == s2.dtype == torch.float32
        assert s13.shape == (i2 // _BLOCK, h // _BLOCK) and s2.shape == (
            h // _BLOCK,
            i2 // 2 // _BLOCK,
        )


shared_expert_fp8_deep_gemm = SharedExpertFp8DeepGemm()


class SharedExpertFp8DeepGemmPrequant(OpWrapper):
    """`shared_expert_fp8_deep_gemm` from an input already quantized to FP8 1x128 by its producer (`a` E4M3 `[n, H]`,
    `a_sf` int32 packed UE8M0 in DeepGEMM's MN-major layout, e.g. from `nvfp4_fp8_quantize`), as stock's `GatedMLP`
    takes the mHC boundary's `fp8_1x128` input."""

    ARCHS = frozenset({Arch.SM_103})
    # Its producer quantizes from `nvfp4_fp8_quantize.MIN_ROWS` (1024) rows on: no decode-sized step reaches it.
    CELLS = (
        Cell(why="the smallest step the FFN boundary quantizes to FP8 1x128: 1024 rows, H 5120 / I 2304",
             spec=dict(n=1024, hidden=5120, inter=2304, limit=10.0)),
        Cell(why="a context chunk: 4096 rows", spec=dict(n=4096, hidden=5120, inter=2304, limit=10.0)),
    )  # fmt: skip
    note = """
Both activations are FP8 per 1x128 block (the input as its producer quantized it), as stock's linear on the mHC
boundary's `fp8_1x128` input; bitwise `shared_expert_fp8_deep_gemm` of the bf16 input from 1024 rows on. The gate is a
relative RMS error of 6% against the fp32 product over the dequantized weights and unquantized activations.
"""

    def __call__(
        self, a: torch.Tensor, a_sf: torch.Tensor, w13, sf13, w2, sf2, limit: float
    ) -> torch.Tensor:
        gu = torch.ops.trtllm.fp8_prequantized_swap_ab_gemm(
            a, a_sf, w13, sf13, disable_ue8m0_cast=True
        )
        h, h_sf = torch.ops.trtllm.silu_and_mul_fp8_quantize_1x128_packed_ue8m0(
            gu, limit if limit > 0 else None, False
        )
        return torch.ops.trtllm.fp8_prequantized_swap_ab_gemm(
            h, h_sf, w2, sf2, disable_ue8m0_cast=True
        )

    def reference(
        self, a, a_sf, w13, sf13, w2, sf2, limit: float, x=None, s13=None, s2=None
    ) -> torch.Tensor:
        """`x`: the bf16 input `a` was quantized from; `s13` / `s2` the fp32 weight multipliers."""
        return _reference(x, w13, s13, w2, s2, limit)

    def is_valid(self, a, a_sf, w13, sf13, w2, sf2, limit: float) -> None:
        n, h = a.shape
        assert a.dtype == w13.dtype == w2.dtype == torch.float8_e4m3fn and a_sf.dtype == torch.int32
        assert (
            w13.shape[1] == h
            and a_sf.shape == (n, h // 512)
            and sf13.dtype == sf2.dtype == torch.int32
        )


shared_expert_fp8_deep_gemm_prequant = SharedExpertFp8DeepGemmPrequant()
