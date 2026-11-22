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
"""Fused post-norm residual kernels: ``residual + rmsnorm(x)`` variants.

One Triton kernel body (specialized via constexpr flags) serves two fusions
on a post-norm residual stream (norm the branch output, then add it to the
residual - the Gemma-style decoder-layer pattern):

1. ``rmsnorm_residual_add_scale``: RMSNorm + residual add + fp32 scalar mul
   + bf16 cast in one kernel, replacing

       flashinfer_rmsnorm(x)                (read + write [M, H] bf16)
       residual + normed                    (2 reads + write, bf16)
       hidden * scale                       (fp32[1] buffer promotes the mul:
                                             read bf16 + WRITE FP32 [M, H])
       .to(bf16) at the consumer's entry    (read fp32 + write bf16)

   Optionally (``next_norm_weight``) the kernel also emits the RMSNorm of
   the result as a second output (e.g. the next decoder layer's input norm),
   so the consumer skips its standalone norm pass entirely (one extra [M, H]
   write here replaces a read + write there).

2. ``rmsnorm_residual_add``: the same body without the scalar stage,
   replacing

       flashinfer_rmsnorm(x)                (read + write [M, H] bf16)
       residual + normed                    (2 reads + write, bf16)

   with one kernel (2 bf16 reads + 1 bf16 write).

3. ``rmsnorm_fp8_quant``: the body without residual or scalar, emitting the
   norm as static per-tensor FP8 for an FP8 Linear consumer. The secondary
   norm output of (1) can likewise be emitted as FP8.

Numerics replicate the unfused chains op-for-op: each norm mirrors
flashinfer's RMSNormKernel (fp32 sum of squares,
`rsqrt.approx.ftz.f32(ssq/d + eps)`, `(x * rcp) * w`, bf16 round), the add
rounds to bf16 exactly like the aten bf16 add (fp32 compute, single round),
the fp32 scalar multiply and final bf16 round match the promoted aten mul +
`.to(bfloat16)`, and the secondary norm reads the bf16-rounded primary
output exactly as a standalone downstream norm would. The only residual
difference is the fp32 reduction order inside the sums of squares (measured
~5e-6 one-step bf16 flips at Gemma4 serving shapes).

Callers own enablement and keep the unfused chains as fallbacks for
configurations these kernels do not support (non-bf16 norms, torch.compile,
...); see the residual-chain fusions in modeling_gemma4.py.
"""

from typing import Optional, Tuple, Union

import torch
import triton
import triton.language as tl

from tensorrt_llm._utils import get_sm_version

from ...flashinfer_utils import get_env_enable_pdl
from ...modules.fla.op import gdc_wait_with_memory_clobber


@triton.jit
def _quant_fp8(v, inv):
    # Static per-tensor E4M3 quantize, as static_quantize_e4m3_per_tensor:
    # fp32(v) * inv, inv = 1 / scale (IEEE reciprocal), saturated to +-448.
    return tl.clamp(v.to(tl.float32) * inv, -448.0, 448.0).to(tl.float8e4nv)


@triton.jit
def _rmsnorm_residual_add_kernel(
    x_ptr,  # [M, H] bf16 kernel input (the branch output being normed)
    r_ptr,  # [M, H] bf16 residual
    w_ptr,  # [H] bf16 RMSNorm weight for x
    sc_ptr,  # [1] fp32 scale (unused when HAS_SCALE == False)
    o_ptr,  # [M, H] bf16 out
    w2_ptr,  # [H] bf16 secondary norm weight (unused when HAS_NORM2 == False)
    n2_ptr,  # [M, H] bf16/fp8 secondary norm out (unused when HAS_NORM2 == False)
    o_qs_ptr,  # [1] fp32 FP8 quant scale of o (unused when O_FP8 == False)
    n2_qs_ptr,  # [1] fp32 FP8 quant scale of n2 (unused when N2_FP8 == False)
    M,
    SX,
    SR,
    SO,
    SN2,
    EPS,
    EPS2,
    H: tl.constexpr,
    BH: tl.constexpr,  # next_power_of_2(H)
    BM: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    HAS_NORM2: tl.constexpr,
    O_FP8: tl.constexpr,
    N2_FP8: tl.constexpr,
    PDL: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM).to(tl.int64)
    rmask = rows < M
    cols = tl.arange(0, BH)
    cmask = cols < H
    mask = rmask[:, None] & cmask[None, :]

    # Weights and scalars are not written by the previous kernel: load them
    # before the PDL wait so their latency overlaps the producer's tail.
    w = tl.load(w_ptr + cols, mask=cmask, other=0.0).to(tl.float32)
    if HAS_NORM2:
        w2 = tl.load(w2_ptr + cols, mask=cmask, other=0.0).to(tl.float32)
    if HAS_SCALE:
        sc = tl.load(sc_ptr)
    if O_FP8:
        o_inv = tl.div_rn(1.0, tl.load(o_qs_ptr))
    if N2_FP8:
        n2_inv = tl.div_rn(1.0, tl.load(n2_qs_ptr))
    if PDL:
        gdc_wait_with_memory_clobber()

    x = tl.load(x_ptr + rows[:, None] * SX + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    # flashinfer RMSNormKernel: rms_rcp = rsqrt.approx.ftz(sum_sq/d + eps);
    # out = (x * rms_rcp) * (weight_bias=0 + w), bf16 round on store.
    ssq = tl.sum(x * x, axis=1)
    rcp = tl.inline_asm_elementwise(
        "rsqrt.approx.ftz.f32 $0, $1;",
        "=f,f",
        [ssq / H + EPS],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    normed = ((x * rcp[:, None]) * w[None, :]).to(tl.bfloat16)
    if HAS_RESIDUAL:
        # aten bf16 add computes in fp32 and rounds once.
        r = tl.load(r_ptr + rows[:, None] * SR + cols[None, :], mask=mask, other=0.0).to(tl.float32)
        s = (r + normed.to(tl.float32)).to(tl.bfloat16)
    else:
        s = normed
    if HAS_SCALE:
        # aten mul(bf16, fp32[1]) promotes to fp32; the consumer's dtype
        # guard rounds back to bf16.
        out = (s.to(tl.float32) * sc).to(tl.bfloat16)
    else:
        out = s
    if O_FP8:
        tl.store(o_ptr + rows[:, None] * SO + cols[None, :], _quant_fp8(out, o_inv), mask=mask)
    else:
        tl.store(o_ptr + rows[:, None] * SO + cols[None, :], out, mask=mask)

    if HAS_NORM2:
        # The secondary RMSNorm, computed on the bf16-rounded primary
        # output exactly as a standalone flashinfer_rmsnorm would read it.
        # Masked columns hold zeros (x, r and w all load 0 there), so they
        # contribute nothing to the sum of squares.
        outf = out.to(tl.float32)
        ssq2 = tl.sum(outf * outf, axis=1)
        rcp2 = tl.inline_asm_elementwise(
            "rsqrt.approx.ftz.f32 $0, $1;",
            "=f,f",
            [ssq2 / H + EPS2],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        n2 = ((outf * rcp2[:, None]) * w2[None, :]).to(tl.bfloat16)
        if N2_FP8:
            n2 = _quant_fp8(n2, n2_inv)
        tl.store(n2_ptr + rows[:, None] * SN2 + cols[None, :], n2, mask=mask)


def _launch_norm_add(
    x: torch.Tensor,
    residual: Optional[torch.Tensor],
    norm_weight: torch.Tensor,
    scale: Optional[torch.Tensor],
    eps: float,
    next_norm_weight: Optional[torch.Tensor],
    next_norm_eps: float,
    out_quant_scale: Optional[torch.Tensor] = None,
    next_norm_quant_scale: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    assert x.dim() == 2 and x.stride(-1) == 1
    assert x.dtype == torch.bfloat16
    has_residual = residual is not None
    if has_residual:
        assert residual.dtype == torch.bfloat16
        assert residual.shape == x.shape and residual.stride(-1) == 1
    m, h = x.shape
    assert norm_weight.shape == (h,)
    if scale is not None:
        assert scale.dtype == torch.float32 and scale.numel() == 1
    has_norm2 = next_norm_weight is not None
    if has_norm2:
        assert next_norm_weight.shape == (h,)
    for qs in (out_quant_scale, next_norm_quant_scale):
        if qs is not None:
            assert qs.dtype == torch.float32 and qs.numel() == 1
    assert next_norm_quant_scale is None or has_norm2
    fp8 = torch.float8_e4m3fn
    o = torch.empty_like(x, dtype=fp8 if out_quant_scale is not None else x.dtype)
    n2 = (
        torch.empty_like(x, dtype=fp8 if next_norm_quant_scale is not None else x.dtype)
        if has_norm2
        else None
    )
    if m == 0:
        return (o, n2) if has_norm2 else o

    # B200-tuned; fixed (no runtime autotune) so launches stay deterministic
    # under CUDA-graph capture. Measured 239 -> 39 us vs the 4-op chain at
    # [7700, 5376]; 34 -> 16 us at the 228-token decode shape; ~6 TB/s
    # effective (memory roofline). A block-size/warp sweep confirmed
    # BM=1/num_warps=4 optimal for all three variants. On SM120, 16 warps
    # per row cut the latency-bound small-M launches by ~25% (2.8 -> 2.1 us
    # at 40 rows); 4 stay faster from ~8k rows.
    num_warps = 16 if get_sm_version() >= 120 and m <= 2048 else 4
    grid = (triton.cdiv(m, 1),)
    pdl = get_env_enable_pdl() and get_sm_version() >= 90
    _rmsnorm_residual_add_kernel[grid](
        x,
        residual if has_residual else x,
        norm_weight,
        scale if scale is not None else norm_weight,
        o,
        next_norm_weight if has_norm2 else norm_weight,
        n2 if has_norm2 else o,
        out_quant_scale if out_quant_scale is not None else norm_weight,
        next_norm_quant_scale if next_norm_quant_scale is not None else norm_weight,
        m,
        x.stride(0),
        residual.stride(0) if has_residual else 0,
        o.stride(0),
        n2.stride(0) if has_norm2 else 0,
        eps,
        next_norm_eps,
        H=h,
        BH=triton.next_power_of_2(h),
        BM=1,
        HAS_RESIDUAL=has_residual,
        HAS_SCALE=scale is not None,
        HAS_NORM2=has_norm2,
        O_FP8=out_quant_scale is not None,
        N2_FP8=next_norm_quant_scale is not None,
        PDL=pdl,
        num_warps=num_warps,
        launch_pdl=pdl,
    )
    return (o, n2) if has_norm2 else o


def rmsnorm_residual_add_scale(
    x: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    scale: torch.Tensor,
    eps: float,
    next_norm_weight: Optional[torch.Tensor] = None,
    next_norm_eps: float = 0.0,
    next_norm_quant_scale: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Fused ``bf16(fp32(residual + rmsnorm(x)) * scale)``.

    Args:
        x: [M, H] bf16 - the branch output (e.g. an MLP down_proj output).
        residual: [M, H] bf16 - the residual stream.
        norm_weight: [H] bf16 RMSNorm weight for ``x`` (plain multiplier
            convention, ``use_gemma=False``).
        scale: [1] fp32 tensor (e.g. a checkpoint-loaded per-layer scalar;
            any value).
        eps: RMSNorm epsilon.
        next_norm_weight: optional [H] bf16 - a second RMSNorm weight. When
            given, the kernel also returns ``rmsnorm(out, next_norm_weight)``
            computed on the bf16-rounded primary output, replacing the
            consumer's standalone norm (e.g. the next decoder layer's input
            norm).
        next_norm_eps: epsilon for the secondary norm.
        next_norm_quant_scale: optional [1] fp32 static FP8 input scale of the
            secondary norm's consumer (an FP8 per-tensor Linear). When given,
            the secondary output is emitted as float8_e4m3fn, quantized from
            its bf16 value exactly as static_quantize_e4m3_per_tensor would.

    Returns:
        [M, H] bf16 output - or ``(out, normed_next)`` when
        ``next_norm_weight`` is given. The intermediate fp32 tensor of the
        unfused chain is never materialized.
    """
    return _launch_norm_add(
        x,
        residual,
        norm_weight,
        scale,
        eps,
        next_norm_weight,
        next_norm_eps,
        next_norm_quant_scale=next_norm_quant_scale,
    )


def rmsnorm_residual_add(
    x: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    next_norm_weight: Optional[torch.Tensor] = None,
    next_norm_eps: float = 0.0,
    next_norm_quant_scale: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Fused ``bf16(fp32(residual) + fp32(rmsnorm(x)))``.

    The post-norm half of a residual chain: RMSNorm on the branch output
    (e.g. an attention o_proj output) followed by the aten bf16 residual
    add, in one kernel.

    Args:
        x: [M, H] bf16 - the branch output.
        residual: [M, H] bf16 - the residual stream.
        norm_weight: [H] bf16 RMSNorm weight for ``x`` (plain multiplier
            convention, ``use_gemma=False``).
        eps: RMSNorm epsilon.
        next_norm_weight: optional [H] bf16 - a second RMSNorm weight. When
            given, the kernel also returns ``rmsnorm(out, next_norm_weight)``
            computed on the bf16-rounded sum (e.g. the pre-feedforward norm).
        next_norm_eps: epsilon for the secondary norm.
        next_norm_quant_scale: optional [1] fp32 static FP8 input scale of the
            secondary norm's consumer; the secondary output is then
            float8_e4m3fn, as ``rmsnorm_fp8_quant`` would emit it.

    Returns:
        [M, H] bf16 - identical (modulo fp32 reduction order in the sum of
        squares) to ``residual + flashinfer_rmsnorm(x, norm_weight, eps)`` -
        or ``(out, normed_next)`` when ``next_norm_weight`` is given.
    """
    return _launch_norm_add(
        x,
        residual,
        norm_weight,
        None,
        eps,
        next_norm_weight,
        next_norm_eps,
        next_norm_quant_scale=next_norm_quant_scale,
    )


def rmsnorm_fp8_quant(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    quant_scale: torch.Tensor,
) -> torch.Tensor:
    """Fused RMSNorm + static per-tensor FP8 quantize.

    Replaces ``flashinfer_rmsnorm`` followed by the consuming FP8 Linear's
    ``static_quantize_e4m3_per_tensor`` (a bf16 [M, H] write and read back).
    The norm is rounded to bf16 before quantizing, as the unfused pair does.

    Args:
        x: [M, H] bf16.
        norm_weight: [H] bf16 RMSNorm weight (``use_gemma=False``).
        eps: RMSNorm epsilon.
        quant_scale: [1] fp32 static input scale of the consumer Linear.

    Returns:
        [M, H] float8_e4m3fn.
    """
    out = _launch_norm_add(x, None, norm_weight, None, eps, None, 0.0, out_quant_scale=quant_scale)
    assert isinstance(out, torch.Tensor)
    return out
