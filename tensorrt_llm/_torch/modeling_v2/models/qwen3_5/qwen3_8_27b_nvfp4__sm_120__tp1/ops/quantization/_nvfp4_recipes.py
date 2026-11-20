# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load-time NVFP4 weight recipes: the copies and layouts ``weights.py`` builds.

Plain functions, not op entries: each runs once per weight at load, on the device
the inputs are on, deterministically. The GEMM entries document which of these
layouts they take.

Two NVFP4 recipes exist for an FP8 per-tensor weight ``W = w8 * s`` and they are
deliberately different (the copies of one weight need not hold the same E2M1/SF
values):

* decode (W4A16; Marlin and the cuda-core GEMV): the per-tensor scale is folded
  into ``fp4_quantize``'s global scale, so ``w8 * s`` is never rounded to BF16
  (`fp8_to_nvfp4_values`);
* W4A4 (``nvfp4_gemm`` at rows above the decode caps): ``BF16(w8 * s)`` is
  quantized (`fp8_to_nvfp4_w4a4`).

The NVFP4 lm_head of the checkpoint is used as stored: repacked for Marlin
(`lm_head_marlin`) and, unchanged, as the W4A4 operand (`lm_head_w4a4`).
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn.functional as F

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm.quantization.utils import marlin_utils

from ._nvfp4_reference import pad_up

_SF_VEC = 16
_E4M3_MAX = 448.0
_E2M1_MAX = 6.0


class Nvfp4Values(NamedTuple):
    """The NVFP4 values both decode layouts share.

    ``packed`` uint8 ``[N, K/2]`` (element ``2i`` in the low nibble), ``sf`` uint8
    ``[N, K/16]`` linear E4M3 bits, ``global_scale`` fp32 0-dim: ``W ~= e2m1 * sf *
    global_scale``.
    """

    packed: torch.Tensor
    sf: torch.Tensor
    global_scale: torch.Tensor


class MarlinNvfp4Weight(NamedTuple):
    """What the target's ``marlin_nvfp4_w4a16_gemm`` takes for the weight: Marlin-tiled int32
    ``weight`` ``[size_k/16, 2*size_n]``, S0E5M3 ``weight_scale`` ``[size_k/16, size_n]``
    (float8_e4m3fn storage), BF16 ``weight_global_scale`` ``[1]`` (global scale times
    2^119), and the padded ``size_n`` (multiple of 128) / ``size_k`` (multiple of 64)
    to pass as the op's sizes."""

    weight: torch.Tensor
    weight_scale: torch.Tensor
    weight_global_scale: torch.Tensor
    size_n: int
    size_k: int


class GemvNvfp4Weight(NamedTuple):
    """What ``native_ops.nvfp4_w4a16_gemv[_conv]`` takes: ``weight`` uint8 ``[N, K/2]``,
    ``weight_sf`` uint8 ``[N, K/16]`` linear E4M3, ``global_scale`` a Python float."""

    weight: torch.Tensor
    weight_sf: torch.Tensor
    global_scale: float


class W4a4Weight(NamedTuple):
    """What ``nvfp4_gemm`` takes for a W4A4 weight: ``weight`` uint8 ``[N, K/2]``,
    ``weight_sf`` the swizzled E4M3 buffer, ``act_global`` fp32 ``[1]`` (the
    activation's ``fp4_quantize`` global scale), ``alpha`` fp32 ``[1]`` (the GEMM's
    output multiplier)."""

    weight: torch.Tensor
    weight_sf: torch.Tensor
    act_global: torch.Tensor
    alpha: torch.Tensor


def fp8_to_nvfp4_values(w_fp8: torch.Tensor, weight_scale: torch.Tensor) -> Nvfp4Values:
    """NVFP4 decode values of an FP8 per-tensor weight ``W = w8 * s``.

    Recipe: ``g = 2^ceil(log2(amax(|w8|) * s / 2688))`` (2688 = 448 * 6, so the largest
    block scale fits E4M3 and ``g`` is a power of two that Marlin's BF16 global-scale
    multiply carries exactly); ``fp4_quantize(BF16(w8), s / g, 16, swizzled)`` -- FP8
    values are exact in BF16, so folding ``s`` into the quantizer's global scale gives
    ``SF = e4m3(blockmax(W) / (6 g))`` and ``W ~= e2m1 * SF * g`` without ever rounding
    ``w8 * s`` to BF16. The swizzled scales are read back to the linear order.
    """
    n, k = w_fp8.shape
    assert w_fp8.dtype == torch.float8_e4m3fn and k % _SF_VEC == 0
    s = weight_scale.float().reshape(())
    w_vals = w_fp8.to(torch.bfloat16)
    amax = w_vals.abs().amax().float() * s
    amax = torch.clamp(amax, min=torch.finfo(torch.float32).tiny)
    g = torch.exp2(torch.ceil(torch.log2(amax / (_E4M3_MAX * _E2M1_MAX))))
    quant_global = (s / g).reshape(1).to(torch.float32)
    packed, sf_swizzled = torch.ops.trtllm.fp4_quantize(w_vals, quant_global, _SF_VEC, False, True)
    del w_vals
    sf = torch.ops.trtllm.block_scale_interleave_reverse(sf_swizzled.view(pad_up(n, 128), -1))
    sf = sf[:n, : k // _SF_VEC].contiguous().view(torch.uint8)
    return Nvfp4Values(packed.view(torch.uint8), sf, g.reshape(()).to(torch.float32))


def marlin_pack(values: Nvfp4Values) -> MarlinNvfp4Weight:
    """Pack NVFP4 values for the target's ``marlin_nvfp4_w4a16_gemm`` (W4A16).

    ``gptq_marlin_repack`` of the codes (``[N, K/2]`` viewed int32 and transposed),
    ``marlin_permute_scales`` + ``nvfp4_marlin_process_scales`` of the ``[K/16, N]``
    scales, ``nvfp4_marlin_process_global_scale(BF16(global_scale))``. N is padded to
    128 and K to 64; K padding gets scale 0x08 (Marlin's S0E5M3 dequantization needs a
    non-zero scale for zero weights), N padding scale 0. A non-finite or zero global
    scale becomes 1.0.
    """
    weight, size_n, size_k = values.packed, values.packed.shape[0], values.packed.shape[1] * 2
    num_groups = size_k // _SF_VEC
    size_k_pad = pad_up(size_k, 64)
    size_n_pad = pad_up(size_n, 128)
    scale_2d = values.sf.view(torch.uint8).reshape(size_n, num_groups)
    if size_k_pad != size_k or size_n_pad != size_n:
        weight = F.pad(weight, (0, (size_k_pad - size_k) // 2, 0, size_n_pad - size_n))
        scale_2d = F.pad(scale_2d, (0, size_k_pad // _SF_VEC - num_groups), value=0x08)
        scale_2d = F.pad(scale_2d, (0, 0, 0, size_n_pad - size_n), value=0)

    qweight_int32 = weight.contiguous().view(torch.int32).T.contiguous()
    marlin_weight = torch.ops.trtllm.gptq_marlin_repack(
        b_q_weight=qweight_int32,
        perm=torch.empty(0, dtype=torch.int32, device=weight.device),
        size_k=size_k_pad,
        size_n=size_n_pad,
        num_bits=4,
        is_a_8bit=False,
    )
    scale_2d = scale_2d.view(torch.float8_e4m3fn).T.contiguous()
    marlin_scale = marlin_utils.marlin_permute_scales(
        scale_2d.to(torch.half), size_k_pad, size_n_pad, group_size=_SF_VEC
    )
    marlin_scale = marlin_utils.nvfp4_marlin_process_scales(marlin_scale)

    global_scale = values.global_scale.float().reshape(1)
    if (
        global_scale.numel() == 0
        or not torch.isfinite(global_scale).all()
        or global_scale.item() == 0
    ):
        global_scale = torch.tensor([1.0], dtype=torch.float32, device=weight.device)
    weight_global_scale = marlin_utils.nvfp4_marlin_process_global_scale(
        global_scale.to(torch.bfloat16)
    )
    return MarlinNvfp4Weight(
        marlin_weight, marlin_scale, weight_global_scale, size_n_pad, size_k_pad
    )


def gemv_layout(values: Nvfp4Values) -> GemvNvfp4Weight:
    """The same NVFP4 values in the cuda-core GEMV's layout: packed E2M1 rows, linear
    E4M3 scales, the global scale as a Python float."""
    return GemvNvfp4Weight(
        values.packed.view(torch.uint8).contiguous(),
        values.sf.view(torch.uint8).contiguous(),
        float(values.global_scale.item()),
    )


def fp8_to_nvfp4_w4a4(
    w_fp8: torch.Tensor, weight_scale: torch.Tensor, input_scale: torch.Tensor
) -> W4a4Weight:
    """NVFP4 W4A4 copy of an FP8 per-tensor weight with a static FP8 input scale ``a``.

    Recipe: ``W = BF16(w8 * s)`` (rounded to BF16 first, unlike `fp8_to_nvfp4_values`),
    ``g = 2^ceil(log2(amax(|W|) / 2688))``, ``fp4_quantize(W, 1/g, 16)`` with swizzled
    scales. The activation is quantized with global scale ``6 / a`` so NVFP4 saturates
    at the amax the FP8 static quantization it replaces saturates at, and the GEMM's
    ``alpha = (a / 6) * g``. ``g``, ``6 / a`` and ``alpha`` are formed in Python doubles
    and stored as fp32 ``[1]``.
    """
    assert w_fp8.dtype == torch.float8_e4m3fn and w_fp8.dim() == 2
    assert weight_scale.numel() == 1 and input_scale.numel() == 1
    with torch.no_grad():
        w_bf16 = (w_fp8.float() * weight_scale.float().reshape(())).to(torch.bfloat16)
        amax = float(w_bf16.abs().amax().float().item())
        assert math.isfinite(amax) and amax > 0.0, f"degenerate weight amax {amax}"
        g = 2.0 ** math.ceil(math.log2(amax / (_E4M3_MAX * _E2M1_MAX)))
        w_global = torch.tensor([1.0 / g], dtype=torch.float32, device=w_fp8.device)
        weight, weight_sf = torch.ops.trtllm.fp4_quantize(w_bf16, w_global, _SF_VEC, False)
        del w_bf16
        a = float(input_scale.float().reshape(()).item())
        act_global = torch.tensor([_E2M1_MAX / a], dtype=torch.float32, device=w_fp8.device)
        alpha = torch.tensor([(a / _E2M1_MAX) * g], dtype=torch.float32, device=w_fp8.device)
    return W4a4Weight(weight.view(torch.uint8), weight_sf, act_global, alpha)


def gate_up_swiglu_row_permutation(n: int) -> torch.Tensor:
    """Row order of the ``[gate | up]`` weight for the SM120 SwiGLU-epilogue GEMM.

    Permuted row ``r`` holds source row ``perm[r]``. Inside every 128-row tile ``t``,
    row ``c = 32a + 16nw + 8b + 2lq + p`` is gate (``p = 0``) or up (``p = 1``) of output
    ``j = 64t + 32nw + 16(a // 2) + 4lq + 2(a % 2) + b`` -- the epilogue's register
    layout, so each thread holds a gate/up pair. Every GEMM column stays the same dot
    product, only reordered; the permuted SwiGLU+NVFP4 kernel reads it back.
    """
    assert n % 128 == 0
    d = n // 2
    c = torch.arange(128)
    a, nw = c // 32, (c % 32) // 16
    b, lq, p = (c % 16) // 8, (c % 8) // 2, c % 2
    jl = 32 * nw + 4 * lq + 16 * (a // 2) + 2 * (a % 2) + b
    j = 64 * torch.arange(n // 128)[:, None] + jl[None, :]
    return torch.where(p[None, :] == 0, j, d + j).reshape(-1)


def permute_gate_up_for_swiglu_epilogue(
    weight: torch.Tensor, weight_sf: torch.Tensor, n: int, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(weight[perm], swizzle(unswizzle(weight_sf)[perm]))`` for the gate_up NVFP4
    weight ``[n, k/2]`` and its swizzled scales, ``perm`` =
    `gate_up_swiglu_row_permutation`. Returns new tensors (scales in ``weight_sf``'s
    shape); the caller copies them over the loaded ones."""
    assert n % 128 == 0 and k % 128 == 0
    assert tuple(weight.shape) == (n, k // 2) and weight_sf.numel() == n * (k // _SF_VEC)
    perm = gate_up_swiglu_row_permutation(n).to(weight.device)
    with torch.no_grad():
        new_weight = weight[perm].contiguous()
        sf2d = torch.ops.trtllm.block_scale_interleave_reverse(
            weight_sf.view(1, n, k // _SF_VEC)
        ).view(n, k // _SF_VEC)
        new_sf = torch.ops.trtllm.block_scale_interleave(sf2d[perm].contiguous()).view_as(weight_sf)
    return new_weight, new_sf


def lm_head_marlin(
    weight: torch.Tensor, weight_scale: torch.Tensor, weight_scale_2: torch.Tensor
) -> MarlinNvfp4Weight:
    """The checkpoint's NVFP4 lm_head (packed ``[V, K/2]``, E4M3 ``[V, K/16]`` linear,
    fp32 ``weight_scale_2``) repacked for the Marlin W4A16 GEMM with its own values:
    `marlin_pack` with global scale ``weight_scale_2`` (Marlin carries it in BF16)."""
    values = Nvfp4Values(
        weight.view(torch.uint8),
        weight_scale.view(torch.uint8).reshape(weight.shape[0], -1),
        weight_scale_2.float().reshape(()),
    )
    return marlin_pack(values)


def lm_head_w4a4(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    input_scale: torch.Tensor,
) -> W4a4Weight:
    """The checkpoint's NVFP4 lm_head as the W4A4 operand, in the checkpoint's own mode:
    the weight as stored, ``block_scale_interleave`` of its E4M3 scales, activation
    global scale ``1 / input_scale`` (the checkpoint stores ``amax / 2688``), ``alpha =
    input_scale * weight_scale_2``."""
    weight_sf = torch.ops.trtllm.block_scale_interleave(
        weight_scale.view(torch.uint8).reshape(weight.shape[0], -1).contiguous()
    )
    act_global = (1.0 / input_scale.float()).reshape(1).contiguous()
    alpha = (input_scale.float() * weight_scale_2.float()).reshape(1).contiguous()
    return W4a4Weight(weight.view(torch.uint8), weight_sf, act_global, alpha)
