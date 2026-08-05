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


@triton.jit
def _rmsnorm_residual_add_kernel(
    x_ptr,  # [M, H] bf16 kernel input (the branch output being normed)
    r_ptr,  # [M, H] bf16 residual
    w_ptr,  # [H] bf16 RMSNorm weight for x
    sc_ptr,  # [1] fp32 scale (unused when HAS_SCALE == False)
    o_ptr,  # [M, H] bf16 out
    w2_ptr,  # [H] bf16 secondary norm weight (unused when HAS_NORM2 == False)
    n2_ptr,  # [M, H] bf16 secondary norm out (unused when HAS_NORM2 == False)
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
    HAS_SCALE: tl.constexpr,
    HAS_NORM2: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM).to(tl.int64)
    rmask = rows < M
    cols = tl.arange(0, BH)
    cmask = cols < H
    mask = rmask[:, None] & cmask[None, :]

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
    w = tl.load(w_ptr + cols, mask=cmask, other=0.0).to(tl.float32)
    normed = ((x * rcp[:, None]) * w[None, :]).to(tl.bfloat16)
    # aten bf16 add computes in fp32 and rounds once.
    r = tl.load(r_ptr + rows[:, None] * SR + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    s = (r + normed.to(tl.float32)).to(tl.bfloat16)
    if HAS_SCALE:
        # aten mul(bf16, fp32[1]) promotes to fp32; the consumer's dtype
        # guard rounds back to bf16.
        sc = tl.load(sc_ptr)
        out = (s.to(tl.float32) * sc).to(tl.bfloat16)
    else:
        out = s
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
        w2 = tl.load(w2_ptr + cols, mask=cmask, other=0.0).to(tl.float32)
        n2 = ((outf * rcp2[:, None]) * w2[None, :]).to(tl.bfloat16)
        tl.store(n2_ptr + rows[:, None] * SN2 + cols[None, :], n2, mask=mask)


def _launch_norm_add(
    x: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    scale: Optional[torch.Tensor],
    eps: float,
    next_norm_weight: Optional[torch.Tensor],
    next_norm_eps: float,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    assert x.dim() == 2 and x.stride(-1) == 1
    assert x.dtype == torch.bfloat16 and residual.dtype == torch.bfloat16
    assert residual.shape == x.shape and residual.stride(-1) == 1
    m, h = x.shape
    assert norm_weight.shape == (h,)
    if scale is not None:
        assert scale.dtype == torch.float32 and scale.numel() == 1
    has_norm2 = next_norm_weight is not None
    if has_norm2:
        assert next_norm_weight.shape == (h,)
    o = torch.empty_like(x)
    n2 = torch.empty_like(x) if has_norm2 else None
    if m == 0:
        return (o, n2) if has_norm2 else o

    # B200-tuned; fixed (no runtime autotune) so launches stay deterministic
    # under CUDA-graph capture. Measured 239 -> 39 us vs the 4-op chain at
    # [7700, 5376]; 34 -> 16 us at the 228-token decode shape; ~6 TB/s
    # effective (memory roofline). A block-size/warp sweep confirmed
    # BM=1/num_warps=4 optimal for all three variants.
    grid = (triton.cdiv(m, 1),)
    _rmsnorm_residual_add_kernel[grid](
        x,
        residual,
        norm_weight,
        scale if scale is not None else norm_weight,
        o,
        next_norm_weight if has_norm2 else norm_weight,
        n2 if has_norm2 else o,
        m,
        x.stride(0),
        residual.stride(0),
        o.stride(0),
        n2.stride(0) if has_norm2 else 0,
        eps,
        next_norm_eps,
        H=h,
        BH=triton.next_power_of_2(h),
        BM=1,
        HAS_SCALE=scale is not None,
        HAS_NORM2=has_norm2,
        num_warps=4,
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

    Returns:
        [M, H] bf16 output - or ``(out, normed_next)`` when
        ``next_norm_weight`` is given. The intermediate fp32 tensor of the
        unfused chain is never materialized.
    """
    return _launch_norm_add(x, residual, norm_weight, scale, eps, next_norm_weight, next_norm_eps)


def rmsnorm_residual_add(
    x: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
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

    Returns:
        [M, H] bf16 - identical (modulo fp32 reduction order in the sum of
        squares) to ``residual + flashinfer_rmsnorm(x, norm_weight, eps)``.
    """
    out = _launch_norm_add(x, residual, norm_weight, None, eps, None, 0.0)
    assert isinstance(out, torch.Tensor)
    return out
