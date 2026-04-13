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

"""Fused Gemma4 post-layer ops: RMSNorm + residual_add [+ layer_scalar].

These kernels eliminate 2-3 separate kernel launches per decoder layer by
fusing the post-attention and post-feedforward normalization, residual add,
and optional layer scalar multiply into single Triton kernels.

Savings at c=1 (batch=1): ~2µs saved per fused occurrence × ~12 occurrences
per decode step = ~24µs = ~1% of 2273µs baseline TPOT.

The fused kernels are:
  - gemma4_post_norm_add: norm(x, w) + residual  (post-attention pattern)
  - gemma4_post_norm_add_scale: (norm(x, w) + residual) * scalar  (post-ffn pattern)
"""

from typing import Tuple

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 4096}, num_warps=4),
        triton.Config({"BLOCK_H": 4096}, num_warps=8),
        triton.Config({"BLOCK_H": 4096}, num_warps=16),
    ],
    key=["H"],
)
@triton.jit
def _post_norm_add_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    out_ptr,
    H: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Fused: out[row] = rms_norm(x[row], weight, eps) + residual[row]"""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H

    x = tl.load(x_ptr + row * H + offs, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + row * H + offs, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + offs, mask=mask, other=1.0).to(tl.float32)

    # RMSNorm: normalize x then apply learned weight
    var = tl.sum(x * x) / H
    inv_rms = tl.rsqrt(var + eps)
    normed = x * inv_rms * weight

    out = normed + residual
    tl.store(out_ptr + row * H + offs, out.to(tl.bfloat16), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 4096}, num_warps=4),
        triton.Config({"BLOCK_H": 4096}, num_warps=8),
        triton.Config({"BLOCK_H": 4096}, num_warps=16),
    ],
    key=["H"],
)
@triton.jit
def _post_norm_add_scale_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    scalar_ptr,
    out_ptr,
    H: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Fused: out[row] = (rms_norm(x[row], weight, eps) + residual[row]) * scalar"""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H

    x = tl.load(x_ptr + row * H + offs, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + row * H + offs, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    scalar = tl.load(scalar_ptr).to(tl.float32)

    var = tl.sum(x * x) / H
    inv_rms = tl.rsqrt(var + eps)
    normed = x * inv_rms * weight

    out = (normed + residual) * scalar
    tl.store(out_ptr + row * H + offs, out.to(tl.bfloat16), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 4096}, num_warps=4),
        triton.Config({"BLOCK_H": 4096}, num_warps=8),
        triton.Config({"BLOCK_H": 4096}, num_warps=16),
    ],
    key=["H"],
)
@triton.jit
def _post_norm_add_and_pre_ff_norm_kernel(
    attn_out_ptr,
    residual_ptr,
    w_post_attn_ptr,
    w_pre_ff_ptr,
    out_hs_ptr,
    out_pre_ff_ptr,
    H: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Fused post-attention norm and pre-feedforward norm sharing one kernel launch.

    Computes two sequential RMSNorms in a single kernel:
      out_hs[row]     = rms_norm(attn_out[row], w_post_attn, eps) + residual[row]
      out_pre_ff[row] = rms_norm(out_hs[row],   w_pre_ff,    eps)

    The intermediate out_hs stays in registers between the two norms, avoiding
    a round-trip through global memory and eliminating one kernel launch per layer.
    """
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H

    attn_out = tl.load(attn_out_ptr + row * H + offs, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + row * H + offs, mask=mask, other=0.0).to(tl.float32)
    w1 = tl.load(w_post_attn_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    w2 = tl.load(w_pre_ff_ptr + offs, mask=mask, other=1.0).to(tl.float32)

    # Step 1: post-attention RMSNorm + residual add → hidden_states
    var1 = tl.sum(attn_out * attn_out) / H
    inv_rms1 = tl.rsqrt(var1 + eps)
    hs = attn_out * inv_rms1 * w1 + residual

    # Step 2: pre-feedforward RMSNorm of hidden_states (in registers — no extra load)
    var2 = tl.sum(hs * hs) / H
    inv_rms2 = tl.rsqrt(var2 + eps)
    pre_ff = hs * inv_rms2 * w2

    tl.store(out_hs_ptr + row * H + offs, hs.to(tl.bfloat16), mask=mask)
    tl.store(out_pre_ff_ptr + row * H + offs, pre_ff.to(tl.bfloat16), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 4096}, num_warps=4),
        triton.Config({"BLOCK_H": 4096}, num_warps=8),
        triton.Config({"BLOCK_H": 4096}, num_warps=16),
    ],
    key=["H"],
)
@triton.jit
def _post_norm_add_and_pre_ff_2norm_kernel(
    attn_out_ptr,
    residual_ptr,
    w_post_attn_ptr,
    w_pre_ff_ptr,
    w_pre_ff_2_ptr,
    out_hs_ptr,
    out_pre_ff_ptr,
    out_pre_ff_2_ptr,
    H: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Fused post-attention norm + two pre-feedforward norms in one kernel.

    Computes:
      out_hs[row]       = rms_norm(attn_out[row], w_post_attn, eps) + residual[row]
      out_pre_ff[row]   = rms_norm(out_hs[row],   w_pre_ff,    eps)   (dense MLP input)
      out_pre_ff_2[row] = rms_norm(out_hs[row],   w_pre_ff_2,  eps)   (MoE input)

    Both pre_ff and pre_ff_2 share the same variance (both normalize out_hs), so
    inv_rms2 is computed once and applied twice with different weight vectors.
    Eliminates 1 extra kernel launch vs the 2-output version (pre_feedforward_layernorm_2).
    """
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H

    attn_out = tl.load(attn_out_ptr + row * H + offs, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + row * H + offs, mask=mask, other=0.0).to(tl.float32)
    w1 = tl.load(w_post_attn_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    w2 = tl.load(w_pre_ff_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    w3 = tl.load(w_pre_ff_2_ptr + offs, mask=mask, other=1.0).to(tl.float32)

    # Step 1: post-attention RMSNorm + residual add → hidden_states
    var1 = tl.sum(attn_out * attn_out) / H
    inv_rms1 = tl.rsqrt(var1 + eps)
    hs = attn_out * inv_rms1 * w1 + residual

    # Step 2: pre-feedforward norms — both use the same variance over hs
    var2 = tl.sum(hs * hs) / H
    inv_rms2 = tl.rsqrt(var2 + eps)
    pre_ff = hs * inv_rms2 * w2  # dense MLP input
    pre_ff_2 = hs * inv_rms2 * w3  # MoE input (frees pre_feedforward_layernorm_2 kernel)

    tl.store(out_hs_ptr + row * H + offs, hs.to(tl.bfloat16), mask=mask)
    tl.store(out_pre_ff_ptr + row * H + offs, pre_ff.to(tl.bfloat16), mask=mask)
    tl.store(out_pre_ff_2_ptr + row * H + offs, pre_ff_2.to(tl.bfloat16), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 4096}, num_warps=4),
        triton.Config({"BLOCK_H": 4096}, num_warps=8),
        triton.Config({"BLOCK_H": 4096}, num_warps=16),
    ],
    key=["H"],
)
@triton.jit
def _post_add_norm_add_scale_kernel(
    a_ptr,
    b_ptr,
    residual_ptr,
    weight_ptr,
    scalar_ptr,
    out_ptr,
    H: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Fused: out[row] = (rms_norm(a[row]+b[row], weight, eps) + residual[row]) * scalar.

    Combines the dense+MoE element-wise add with the subsequent post-feedforward
    norm+residual+scale, saving 1 kernel launch per MoE layer (30 layers = 30 savings).
    Safe to fuse: a and b are already synchronized (multi_stream_moe join point).
    """
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H

    a = tl.load(a_ptr + row * H + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + row * H + offs, mask=mask, other=0.0).to(tl.float32)
    x = a + b  # dense MLP + MoE combine
    residual = tl.load(residual_ptr + row * H + offs, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    scalar = tl.load(scalar_ptr).to(tl.float32)

    var = tl.sum(x * x) / H
    inv_rms = tl.rsqrt(var + eps)
    normed = x * inv_rms * weight

    out = (normed + residual) * scalar
    tl.store(out_ptr + row * H + offs, out.to(tl.bfloat16), mask=mask)


# ---------------------------------------------------------------------------
# Custom op registration for torch.export / FakeTensor tracing
# ---------------------------------------------------------------------------

# Threshold: use Triton fused kernel for T ≤ this value; use flashinfer for T > this.
# iter103 finding: CUDA-graph benchmark (torch.cuda.CUDAGraph) shows Triton is
# 2.5-3.8× faster than FlashInfer for ALL T from 1 to 256, due to node dispatch
# overhead dominating over device compute time at all batch sizes.
# - post_norm_add_pre_ff_2norm at T=256: Triton=2.17µs vs FI=5.86µs → 2.7× faster
# - 3-norm-scale at T=256: Triton=3.33µs vs FI=11.0µs → 3.3× faster
# Raised from 8 → 512 to cover all decode batch sizes (max_batch_size=512).
# FlashInfer fallback kept for very large T (prefill ISL > 512) where Triton's
# BLOCK_H=4096 wastes >31% bandwidth relative to H=2816 hidden size.
# Original threshold=8 was set based on eager-mode (non-graph) benchmarks at
# iter92 which measure Python/CUDA-runtime overhead, not CUDA-graph node overhead.
_TRITON_T_THRESHOLD = 512


@torch.library.custom_op("auto_deploy::gemma4_post_norm_add", mutates_args=(), device_types="cuda")
def gemma4_post_norm_add(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Adaptive post-attention: rms_norm(x, weight, eps) + residual.

    Dispatches to a fused Triton kernel for small T (≤ _TRITON_T_THRESHOLD) where
    reducing kernel-launch count in the CUDA graph pays off, and falls back to
    flashinfer RMSNorm + elementwise add for larger T where flashinfer's vectorised
    kernel achieves higher bandwidth utilisation.
    """
    import flashinfer

    H = x.shape[-1]
    out = torch.empty_like(x)
    x_2d = x.view(-1, H)
    residual_2d = residual.view(-1, H)
    T_flat = x_2d.shape[0]
    out_2d = out.view(-1, H)

    if T_flat <= _TRITON_T_THRESHOLD:
        _post_norm_add_kernel[(T_flat,)](x_2d, residual_2d, weight, out_2d, H=H, eps=eps)
    else:
        # Write rmsnorm directly to out (no norm_out temp), then add residual in-place.
        # Saves 1 T×H intermediate allocation and 2T×H memory traffic vs 2-step approach.
        flashinfer.norm.rmsnorm(x_2d, weight, eps, out=out_2d)
        out_2d.add_(residual_2d)

    return out


@gemma4_post_norm_add.register_fake
def _gemma4_post_norm_add_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op(
    "auto_deploy::gemma4_post_norm_add_scale", mutates_args=(), device_types="cuda"
)
def gemma4_post_norm_add_scale(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    scalar: torch.Tensor,
) -> torch.Tensor:
    """Adaptive post-feedforward: (rms_norm(x, weight, eps) + residual) * scalar.

    Same adaptive dispatch as gemma4_post_norm_add: Triton for T ≤ threshold
    (saves 2 kernel launches in CUDA graph), flashinfer + elementwise for T > threshold.
    """
    import flashinfer

    H = x.shape[-1]
    out = torch.empty_like(x)
    x_2d = x.view(-1, H)
    residual_2d = residual.view(-1, H)
    T_flat = x_2d.shape[0]
    out_2d = out.view(-1, H)

    if T_flat <= _TRITON_T_THRESHOLD:
        _post_norm_add_scale_kernel[(T_flat,)](
            x_2d, residual_2d, weight, scalar.view(-1), out_2d, H=H, eps=eps
        )
    else:
        # Write rmsnorm directly to out (no norm_out temp), then add+scale in-place.
        # Eliminates 2 intermediate T×H allocations and 2T×H memory traffic vs 3-step approach.
        flashinfer.norm.rmsnorm(x_2d, weight, eps, out=out_2d)
        out_2d.add_(residual_2d)
        out_2d.mul_(scalar)

    return out


@gemma4_post_norm_add_scale.register_fake
def _gemma4_post_norm_add_scale_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    scalar: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op(
    "auto_deploy::gemma4_post_add_norm_add_scale", mutates_args=(), device_types="cuda"
)
def gemma4_post_add_norm_add_scale(
    a: torch.Tensor,
    b: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    scalar: torch.Tensor,
) -> torch.Tensor:
    """Fused MoE combine + post-norm: (rms_norm(a+b, weight, eps) + residual) * scalar.

    Merges the dense+MoE element-wise add with post_norm_add_scale into one kernel,
    saving 1 launch per MoE layer (30 layers × 1 = 30 launches at T≤8).
    Safe: a and b are already synchronized at the multi_stream_moe join point.
    """
    import flashinfer

    H = a.shape[-1]
    out = torch.empty_like(a)
    a_2d = a.view(-1, H)
    b_2d = b.view(-1, H)
    residual_2d = residual.view(-1, H)
    T_flat = a_2d.shape[0]
    out_2d = out.view(-1, H)

    if T_flat <= _TRITON_T_THRESHOLD:
        _post_add_norm_add_scale_kernel[(T_flat,)](
            a_2d, b_2d, residual_2d, weight, scalar.view(-1), out_2d, H=H, eps=eps
        )
    else:
        # fused_add_rmsnorm(a, b): 1 kernel instead of 2 (separate add + rmsnorm).
        # Modifies in-place: b_2d ← a+b, a_2d ← rmsnorm(a+b).
        # Both a and b are consumed here and not used downstream — safe to mutate.
        flashinfer.norm.fused_add_rmsnorm(a_2d, b_2d, weight, eps)
        torch.add(a_2d, residual_2d, out=out_2d)
        out_2d.mul_(scalar)

    return out


@gemma4_post_add_norm_add_scale.register_fake
def _gemma4_post_add_norm_add_scale_fake(
    a: torch.Tensor,
    b: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    scalar: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(a)


# ---------------------------------------------------------------------------
# 3-norm fusion: post_ff_ln_1(a) + post_ff_ln_2(b) → combined → post_ff_ln + residual + scale
# Replaces 3 kernel dispatches with 1; saves 2 launches × 30 layers = 60 launches ≈ 78µs at c=1.
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 4096}, num_warps=4),
        triton.Config({"BLOCK_H": 4096}, num_warps=8),
        triton.Config({"BLOCK_H": 4096}, num_warps=16),
    ],
    key=["H"],
)
@triton.jit
def _fused_post_3norm_add_scale_kernel(
    a_ptr,
    b_ptr,
    residual_ptr,
    w_a_ptr,
    w_b_ptr,
    w_ptr,
    scalar_ptr,
    out_ptr,
    H: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Fused 3-norm: out = (rms_norm(rms_norm(a,w_a) + rms_norm(b,w_b), w) + residual) * scalar.

    Absorbs post_feedforward_layernorm_1(a) and post_feedforward_layernorm_2(b) into
    the post_add_norm_add_scale kernel, eliminating 2 kernel launches per MoE layer.
    """
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H

    a = tl.load(a_ptr + row * H + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + row * H + offs, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + row * H + offs, mask=mask, other=0.0).to(tl.float32)
    w_a = tl.load(w_a_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    w_b = tl.load(w_b_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    scalar = tl.load(scalar_ptr).to(tl.float32)

    # RMSNorm a with w_a
    var_a = tl.sum(a * a) / H
    a_normed = a * tl.rsqrt(var_a + eps) * w_a

    # RMSNorm b with w_b
    var_b = tl.sum(b * b) / H
    b_normed = b * tl.rsqrt(var_b + eps) * w_b

    # Combined = normed_a + normed_b, then RMSNorm with w
    combined = a_normed + b_normed
    var_c = tl.sum(combined * combined) / H
    normed = combined * tl.rsqrt(var_c + eps) * w

    out = (normed + residual) * scalar
    tl.store(out_ptr + row * H + offs, out.to(tl.bfloat16), mask=mask)


@torch.library.custom_op(
    "auto_deploy::gemma4_fused_post_3norm_add_scale", mutates_args=(), device_types="cuda"
)
def gemma4_fused_post_3norm_add_scale(
    a: torch.Tensor,
    b: torch.Tensor,
    residual: torch.Tensor,
    w_a: torch.Tensor,
    w_b: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    scalar: torch.Tensor,
) -> torch.Tensor:
    """3-norm fusion: (rms_norm(rms_norm(a,w_a) + rms_norm(b,w_b), weight) + residual) * scalar.

    Absorbs post_feedforward_layernorm_1 and post_feedforward_layernorm_2 into the
    combined post_norm_add_scale op, saving 2 kernel launches per MoE layer (all 30 layers
    are MoE). Expected savings: 2 × 30 × 1.3µs = 78µs at c=1.

    Small T (≤ _TRITON_T_THRESHOLD): fused Triton kernel (3 norms + residual + scale in 1 pass).
    Large T: flashinfer rmsnorm calls (bandwidth-optimal).
    """
    import flashinfer

    H = a.shape[-1]
    out = torch.empty_like(a)
    a_2d = a.view(-1, H)
    b_2d = b.view(-1, H)
    residual_2d = residual.view(-1, H)
    T_flat = a_2d.shape[0]
    out_2d = out.view(-1, H)

    if T_flat <= _TRITON_T_THRESHOLD:
        _fused_post_3norm_add_scale_kernel[(T_flat,)](
            a_2d, b_2d, residual_2d, w_a, w_b, weight, scalar.view(-1), out_2d, H=H, eps=eps
        )
    else:
        # norm a → a_normed, norm b → b_normed, then fused_add_rmsnorm(a_n, b_n) + residual + scale
        a_normed = torch.empty_like(a_2d)
        b_normed = torch.empty_like(b_2d)
        flashinfer.norm.rmsnorm(a_2d, w_a, eps, out=a_normed)
        flashinfer.norm.rmsnorm(b_2d, w_b, eps, out=b_normed)
        flashinfer.norm.fused_add_rmsnorm(a_normed, b_normed, weight, eps)
        torch.add(a_normed, residual_2d, out=out_2d)
        out_2d.mul_(scalar)

    return out


@gemma4_fused_post_3norm_add_scale.register_fake
def _gemma4_fused_post_3norm_add_scale_fake(
    a: torch.Tensor,
    b: torch.Tensor,
    residual: torch.Tensor,
    w_a: torch.Tensor,
    w_b: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    scalar: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(a)


# ---------------------------------------------------------------------------
# iter102: 3-norm + input_ln fusion — pre-compute next layer's input_layernorm
# Adds one more norm to _fused_post_3norm_add_scale while `out` is still in registers.
# Saves 1 kernel launch per layer × 29 inter-layer transitions ≈ 38µs at c=1.
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 4096}, num_warps=4),
        triton.Config({"BLOCK_H": 4096}, num_warps=8),
        triton.Config({"BLOCK_H": 4096}, num_warps=16),
    ],
    key=["H"],
)
@triton.jit
def _fused_post_3norm_add_scale_and_input_ln_kernel(
    a_ptr,
    b_ptr,
    residual_ptr,
    w_a_ptr,
    w_b_ptr,
    w_ptr,
    scalar_ptr,
    w_input_ln_ptr,  # next layer's input_layernorm weight
    out_ptr,  # (normed + residual) * scalar
    out_input_ptr,  # rms_norm(out, w_input_ln) — next layer's pre-normed input
    H: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """3-norm + input_ln: fuses next layer's input_layernorm while out is in registers.

    Computes everything from _fused_post_3norm_add_scale_kernel, then additionally:
      out_input[row] = rms_norm(out[row], w_input_ln, eps)

    Since out is already in registers after the residual add + scale, this extra norm
    costs only 1 load (w_input_ln) + 1 store (out_input_ptr) with no extra memory
    round-trip for the input tensor — eliminating 1 flashinfer kernel per layer.
    """
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H

    a = tl.load(a_ptr + row * H + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + row * H + offs, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + row * H + offs, mask=mask, other=0.0).to(tl.float32)
    w_a = tl.load(w_a_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    w_b = tl.load(w_b_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    scalar = tl.load(scalar_ptr).to(tl.float32)

    # RMSNorm a with w_a
    var_a = tl.sum(a * a) / H
    a_normed = a * tl.rsqrt(var_a + eps) * w_a

    # RMSNorm b with w_b
    var_b = tl.sum(b * b) / H
    b_normed = b * tl.rsqrt(var_b + eps) * w_b

    # Combined = normed_a + normed_b, then RMSNorm with w
    combined = a_normed + b_normed
    var_c = tl.sum(combined * combined) / H
    normed = combined * tl.rsqrt(var_c + eps) * w

    out = (normed + residual) * scalar
    tl.store(out_ptr + row * H + offs, out.to(tl.bfloat16), mask=mask)

    # Pre-compute next layer's input_layernorm.
    # Round-trip through bfloat16 so the variance matches what the next layer would
    # see (flashinfer operates on the bfloat16-stored out, not the float32 intermediate).
    out_bf16 = out.to(tl.bfloat16).to(tl.float32)
    w_iln = tl.load(w_input_ln_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    var_out = tl.sum(out_bf16 * out_bf16) / H
    input_normed = out_bf16 * tl.rsqrt(var_out + eps) * w_iln
    tl.store(out_input_ptr + row * H + offs, input_normed.to(tl.bfloat16), mask=mask)


@torch.library.custom_op(
    "auto_deploy::gemma4_fused_post_3norm_add_scale_and_input_ln",
    mutates_args=(),
    device_types="cuda",
)
def gemma4_fused_post_3norm_add_scale_and_input_ln(
    a: torch.Tensor,
    b: torch.Tensor,
    residual: torch.Tensor,
    w_a: torch.Tensor,
    w_b: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    scalar: torch.Tensor,
    w_input_ln: torch.Tensor,
) -> torch.Tensor:
    """3-norm fusion + next-layer input_layernorm fused into one kernel.

    Returns packed tensor of shape (2, *a.shape) where:
      packed[0] = (rms_norm(rms_norm(a,w_a) + rms_norm(b,w_b), weight) + residual) * scalar
      packed[1] = rms_norm(packed[0], w_input_ln)   # next layer's input_layernorm output

    Small T (≤ _TRITON_T_THRESHOLD): fused Triton kernel (4 norms + residual + scale in 1 pass).
    Large T: flashinfer for the 3-norm part, then one extra flashinfer rmsnorm for input_ln.
    Saves 1 kernel launch per layer × 29 inter-layer transitions ≈ 38µs at c=1 (decode).
    """
    import flashinfer

    H = a.shape[-1]
    a_2d = a.view(-1, H)
    b_2d = b.view(-1, H)
    residual_2d = residual.view(-1, H)
    T_flat = a_2d.shape[0]

    # packed_flat[0] = out, packed_flat[1] = input_normed
    packed_flat = torch.empty(2, T_flat, H, dtype=torch.bfloat16, device=a.device)
    out_2d = packed_flat[0]
    input_normed_2d = packed_flat[1]

    if T_flat <= _TRITON_T_THRESHOLD:
        _fused_post_3norm_add_scale_and_input_ln_kernel[(T_flat,)](
            a_2d,
            b_2d,
            residual_2d,
            w_a,
            w_b,
            weight,
            scalar.view(-1),
            w_input_ln,
            out_2d,
            input_normed_2d,
            H=H,
            eps=eps,
        )
    else:
        # Large-T: reuse existing 3-norm logic then one extra rmsnorm for input_ln
        a_normed = torch.empty_like(a_2d)
        b_normed = torch.empty_like(b_2d)
        flashinfer.norm.rmsnorm(a_2d, w_a, eps, out=a_normed)
        flashinfer.norm.rmsnorm(b_2d, w_b, eps, out=b_normed)
        flashinfer.norm.fused_add_rmsnorm(a_normed, b_normed, weight, eps)
        torch.add(a_normed, residual_2d, out=out_2d)
        out_2d.mul_(scalar)
        # Pre-compute next layer's input_layernorm
        flashinfer.norm.rmsnorm(out_2d, w_input_ln, eps, out=input_normed_2d)

    return packed_flat.reshape(2, *a.shape)


@gemma4_fused_post_3norm_add_scale_and_input_ln.register_fake
def _gemma4_fused_post_3norm_add_scale_and_input_ln_fake(
    a: torch.Tensor,
    b: torch.Tensor,
    residual: torch.Tensor,
    w_a: torch.Tensor,
    w_b: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    scalar: torch.Tensor,
    w_input_ln: torch.Tensor,
) -> torch.Tensor:
    return torch.empty((2, *a.shape), dtype=a.dtype, device=a.device)


@torch.library.custom_op(
    "auto_deploy::gemma4_post_norm_add_and_pre_ff_norm", mutates_args=(), device_types="cuda"
)
def gemma4_post_norm_add_and_pre_ff_norm(
    attn_out: torch.Tensor,
    residual: torch.Tensor,
    post_attn_weight: torch.Tensor,
    pre_ff_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Fused post-attention norm + pre-feedforward norm: saves 1 kernel launch per layer.

    Returns a packed tensor of shape (2, *attn_out.shape) where:
      packed[0] = rms_norm(attn_out, post_attn_weight, eps) + residual  (hidden_states)
      packed[1] = rms_norm(packed[0], pre_ff_weight, eps)               (pre-feedforward input)

    Dispatches to the fused Triton kernel for T ≤ _TRITON_T_THRESHOLD (decode path),
    saving one round-trip through global memory and one kernel launch per decoder layer.
    Falls back to two flashinfer rmsnorm calls for large T (prefill path).
    """
    import flashinfer

    H = attn_out.shape[-1]
    T_flat = attn_out.numel() // H
    attn_2d = attn_out.reshape(T_flat, H)
    residual_2d = residual.reshape(T_flat, H)

    # packed_flat: (2, T_flat, H) — packed[0]=hidden_states, packed[1]=pre_ff_in
    packed_flat = torch.empty(2, T_flat, H, dtype=torch.bfloat16, device=attn_out.device)

    if T_flat <= _TRITON_T_THRESHOLD:
        _post_norm_add_and_pre_ff_norm_kernel[(T_flat,)](
            attn_2d,
            residual_2d,
            post_attn_weight,
            pre_ff_weight,
            packed_flat[0],
            packed_flat[1],
            H=H,
            eps=eps,
        )
    else:
        # Write rmsnorm directly to packed_flat[0], add residual in-place, then
        # write second rmsnorm directly to packed_flat[1].
        # Saves 2 T×H intermediate allocations and 3T×H memory traffic vs prior 4-op approach.
        flashinfer.norm.rmsnorm(attn_2d, post_attn_weight, eps, out=packed_flat[0])
        packed_flat[0].add_(residual_2d)
        flashinfer.norm.rmsnorm(packed_flat[0], pre_ff_weight, eps, out=packed_flat[1])

    return packed_flat.reshape(2, *attn_out.shape)


@gemma4_post_norm_add_and_pre_ff_norm.register_fake
def _gemma4_post_norm_add_and_pre_ff_norm_fake(
    attn_out: torch.Tensor,
    residual: torch.Tensor,
    post_attn_weight: torch.Tensor,
    pre_ff_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return torch.empty(2, *attn_out.shape, dtype=torch.bfloat16, device=attn_out.device)


@torch.library.custom_op(
    "auto_deploy::gemma4_post_norm_add_and_pre_ff_2norm", mutates_args=(), device_types="cuda"
)
def gemma4_post_norm_add_and_pre_ff_2norm(
    attn_out: torch.Tensor,
    residual: torch.Tensor,
    post_attn_weight: torch.Tensor,
    pre_ff_weight: torch.Tensor,
    pre_ff_2_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Fused post-attention norm + two pre-feedforward norms: saves 2 kernel launches per layer.

    Returns a packed tensor of shape (3, *attn_out.shape) where:
      packed[0] = rms_norm(attn_out, post_attn_weight, eps) + residual  (hidden_states)
      packed[1] = rms_norm(packed[0], pre_ff_weight, eps)               (dense MLP input)
      packed[2] = rms_norm(packed[0], pre_ff_2_weight, eps)             (MoE input)

    Both packed[1] and packed[2] share the same variance computation over packed[0],
    so the second norm is nearly free. Eliminates one extra kernel vs the 2-output version
    (the separate pre_feedforward_layernorm_2 call in the MoE path).

    Expected savings vs iter100: 1 kernel × 30 layers × 1.3µs = 39µs at c=1.
    """
    import flashinfer

    H = attn_out.shape[-1]
    T_flat = attn_out.numel() // H
    attn_2d = attn_out.reshape(T_flat, H)
    residual_2d = residual.reshape(T_flat, H)

    packed_flat = torch.empty(3, T_flat, H, dtype=torch.bfloat16, device=attn_out.device)

    if T_flat <= _TRITON_T_THRESHOLD:
        _post_norm_add_and_pre_ff_2norm_kernel[(T_flat,)](
            attn_2d,
            residual_2d,
            post_attn_weight,
            pre_ff_weight,
            pre_ff_2_weight,
            packed_flat[0],
            packed_flat[1],
            packed_flat[2],
            H=H,
            eps=eps,
        )
    else:
        flashinfer.norm.rmsnorm(attn_2d, post_attn_weight, eps, out=packed_flat[0])
        packed_flat[0].add_(residual_2d)
        flashinfer.norm.rmsnorm(packed_flat[0], pre_ff_weight, eps, out=packed_flat[1])
        flashinfer.norm.rmsnorm(packed_flat[0], pre_ff_2_weight, eps, out=packed_flat[2])

    return packed_flat.reshape(3, *attn_out.shape)


@gemma4_post_norm_add_and_pre_ff_2norm.register_fake
def _gemma4_post_norm_add_and_pre_ff_2norm_fake(
    attn_out: torch.Tensor,
    residual: torch.Tensor,
    post_attn_weight: torch.Tensor,
    pre_ff_weight: torch.Tensor,
    pre_ff_2_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return torch.empty(3, *attn_out.shape, dtype=torch.bfloat16, device=attn_out.device)


# ---------------------------------------------------------------------------
# Fused Q/K/V RMSNorm — replaces 3 separate torch_rmsnorm launches with 1
# ---------------------------------------------------------------------------

# Use Triton fused kernel when total QKV head-rows ≤ this value.
# iter103: CUDA-graph benchmark shows Triton is 2.1-4.6× faster than FlashInfer
# for ALL batch sizes due to graph node dispatch overhead dominating compute.
# Raised from 256 → 20000 to cover all decode batch sizes up to max_batch_size=512:
#   - Local (H=256,NQ=16,NKV=8) BS=512: total_rows=512×16+2×512×8=16384; Triton 2.1× faster
#   - Global (H=512,NQ=16,NKV=2) BS=512: total_rows=512×16+2×512×2=10240; Triton 2.2× faster
# FlashInfer fallback kept for very large T prefills (total_rows > 20000).
_QKV_TRITON_THRESHOLD = 20000


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 512}, num_warps=4),
        triton.Config({"BLOCK_H": 512}, num_warps=8),
        triton.Config({"BLOCK_H": 512}, num_warps=16),
    ],
    key=["H"],
)
@triton.jit
def _qkv_norm_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    w_q_ptr,
    w_k_ptr,
    w_v_ptr,
    q_out_ptr,
    k_out_ptr,
    v_out_ptr,
    N_Q,
    N_KV,
    H: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Single-kernel fused RMSNorm for Q, K, V head tensors.

    Grid: (N_Q + 2*N_KV,) — one program instance per head row.
    Programs [0, N_Q)         process q rows with w_q weights.
    Programs [N_Q, N_Q+N_KV)  process k rows with w_k weights.
    Programs [N_Q+N_KV, ...)   process v rows with w_v weights.

    Inactive tensor loads are predicated (mask=False → other=0.0 → no mem access).
    Because the masks are mutually exclusive, x = xq + xk + xv equals the active
    tensor's data for every thread. Same logic applies to the weight vector.
    OOB pointer arithmetic (negative kv_row / v_row for q/k blocks) is safe because
    CUDA predicated loads/stores never dereference the address when the mask is False.
    """
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H

    is_q = row < N_Q
    is_k = (row >= N_Q) & (row < N_Q + N_KV)
    is_v = ~is_q & ~is_k  # row >= N_Q + N_KV

    kv_row = row - N_Q  # index into k; may be negative for q-blocks (predicated out)
    v_row = row - N_Q - N_KV  # index into v; may be negative for q/k-blocks (predicated out)

    # Load input — exactly one of the three is non-zero for a given block
    xq = tl.load(q_ptr + row * H + offs, mask=mask & is_q, other=0.0).to(tl.float32)
    xk = tl.load(k_ptr + kv_row * H + offs, mask=mask & is_k, other=0.0).to(tl.float32)
    xv = tl.load(v_ptr + v_row * H + offs, mask=mask & is_v, other=0.0).to(tl.float32)
    x = xq + xk + xv

    # Load weight — one of w_q, w_k, w_v; inactive tensors contribute 0 via other=0.0
    wq = tl.load(w_q_ptr + offs, mask=mask & is_q, other=0.0).to(tl.float32)
    wk = tl.load(w_k_ptr + offs, mask=mask & is_k, other=0.0).to(tl.float32)
    wv = tl.load(w_v_ptr + offs, mask=mask & is_v, other=0.0).to(tl.float32)
    w = wq + wk + wv

    # RMSNorm: normalize x, then scale by the learned weight
    var = tl.sum(x * x) / H
    inv_rms = tl.rsqrt(var + eps)
    normed = (x * inv_rms * w).to(tl.bfloat16)

    # Store to the correct output tensor (predicated)
    tl.store(q_out_ptr + row * H + offs, normed, mask=mask & is_q)
    tl.store(k_out_ptr + kv_row * H + offs, normed, mask=mask & is_k)
    tl.store(v_out_ptr + v_row * H + offs, normed, mask=mask & is_v)


@torch.library.custom_op("auto_deploy::gemma4_qkv_norm", mutates_args=(), device_types="cuda")
def gemma4_qkv_norm(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused per-head RMSNorm for Q, K, V: one Triton kernel instead of 3.

    Replaces three separate torch_rmsnorm calls (one per Q/K/V head group) with a
    single fused Triton kernel, saving 2 kernel-launch overheads per decoder layer.
    At 30 layers the savings are ~30µs at c=1 (30 × ~1µs avoided CG dispatch).

    Small T (total QKV rows ≤ _QKV_TRITON_THRESHOLD): fused Triton kernel.
    Large T: three separate flashinfer.norm.rmsnorm calls (bandwidth-optimal).

    K=V sharing (global layers where v = k before normalization): safe — the kernel
    reads k_ptr and v_ptr independently (both point to the same pre-norm data) and
    writes to separate output buffers using distinct weights (w_k vs w_v).
    """
    import flashinfer as _flashinfer

    H = q.shape[-1]
    q_2d = q.reshape(-1, H)
    k_2d = k.reshape(-1, H)
    v_2d = v.reshape(-1, H)
    N_Q = q_2d.shape[0]
    N_KV = k_2d.shape[0]

    q_out = torch.empty_like(q_2d)
    k_out = torch.empty_like(k_2d)
    v_out = torch.empty_like(v_2d)

    total_rows = N_Q + 2 * N_KV
    if total_rows <= _QKV_TRITON_THRESHOLD:
        _qkv_norm_kernel[(total_rows,)](
            q_2d,
            k_2d,
            v_2d,
            w_q,
            w_k,
            w_v,
            q_out,
            k_out,
            v_out,
            N_Q=N_Q,
            N_KV=N_KV,
            H=H,
            eps=eps,
        )
    else:
        _flashinfer.norm.rmsnorm(q_2d, w_q, eps, out=q_out)
        _flashinfer.norm.rmsnorm(k_2d, w_k, eps, out=k_out)
        _flashinfer.norm.rmsnorm(v_2d, w_v, eps, out=v_out)

    return q_out.reshape(q.shape), k_out.reshape(k.shape), v_out.reshape(v.shape)


@gemma4_qkv_norm.register_fake
def _gemma4_qkv_norm_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


# ---------------------------------------------------------------------------
# Fused QKV RMSNorm + RoPE — eliminates the separate rope kernel launch
# ---------------------------------------------------------------------------


@triton.jit
def _qkv_norm_rope_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    w_q_ptr,
    w_k_ptr,
    w_v_ptr,
    cos_ptr,
    sin_ptr,
    q_out_ptr,
    k_out_ptr,
    v_out_ptr,
    N_Q,
    N_KV,
    N_QH: tl.constexpr,
    N_KVH: tl.constexpr,
    H: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Single-kernel fused RMSNorm + RoPE for Q, K, V.

    Grid: (N_Q + 2*N_KV,) — one program per head row.
    Programs [0, N_Q)        → Q rows: RMSNorm(w_q) + RoPE rotation.
    Programs [N_Q, N_Q+N_KV) → K rows: RMSNorm(w_k) + RoPE rotation.
    Programs [N_Q+N_KV, ...)  → V rows: RMSNorm(w_v), no rotation.

    BLOCK_H = H // 2: each program loads the first and second halves of the head
    separately so that rotate_half can be computed as a direct in-register operation:
      out_lo = n_lo * cos_lo - n_hi * sin_lo
      out_hi = n_hi * cos_hi + n_lo * sin_hi
    For V rows, cos/sin loads use mask=False → other=1.0/0.0, giving cos=1, sin=0
    so the rotation formula passes the normed value through unchanged.

    This kernel NEVER calls another custom op — all computation is inline Triton,
    avoiding the eager-kernel explosion caused by nested custom op calls.
    """
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)  # [0..H/2 - 1]

    is_q = row < N_Q
    is_k = (row >= N_Q) & (row < N_Q + N_KV)
    is_v = ~is_q & ~is_k

    kv_row = row - N_Q
    v_row = row - N_Q - N_KV
    kv_row_safe = tl.where(is_k, kv_row, 0)
    v_row_safe = tl.where(is_v, v_row, 0)

    # ---- Load first and second halves of each input ----
    xq_lo = tl.load(q_ptr + row * H + offs, mask=is_q, other=0.0).to(tl.float32)
    xq_hi = tl.load(q_ptr + row * H + offs + BLOCK_H, mask=is_q, other=0.0).to(tl.float32)
    xk_lo = tl.load(k_ptr + kv_row_safe * H + offs, mask=is_k, other=0.0).to(tl.float32)
    xk_hi = tl.load(k_ptr + kv_row_safe * H + offs + BLOCK_H, mask=is_k, other=0.0).to(tl.float32)
    xv_lo = tl.load(v_ptr + v_row_safe * H + offs, mask=is_v, other=0.0).to(tl.float32)
    xv_hi = tl.load(v_ptr + v_row_safe * H + offs + BLOCK_H, mask=is_v, other=0.0).to(tl.float32)
    x_lo = xq_lo + xk_lo + xv_lo
    x_hi = xq_hi + xk_hi + xv_hi

    # ---- Load weights (half each) ----
    wq_lo = tl.load(w_q_ptr + offs, mask=is_q, other=0.0).to(tl.float32)
    wq_hi = tl.load(w_q_ptr + offs + BLOCK_H, mask=is_q, other=0.0).to(tl.float32)
    wk_lo = tl.load(w_k_ptr + offs, mask=is_k, other=0.0).to(tl.float32)
    wk_hi = tl.load(w_k_ptr + offs + BLOCK_H, mask=is_k, other=0.0).to(tl.float32)
    wv_lo = tl.load(w_v_ptr + offs, mask=is_v, other=0.0).to(tl.float32)
    wv_hi = tl.load(w_v_ptr + offs + BLOCK_H, mask=is_v, other=0.0).to(tl.float32)
    w_lo = wq_lo + wk_lo + wv_lo
    w_hi = wq_hi + wk_hi + wv_hi

    # ---- RMSNorm over all H = BLOCK_H * 2 elements ----
    var = (tl.sum(x_lo * x_lo) + tl.sum(x_hi * x_hi)) / H
    inv_rms = tl.rsqrt(var + eps)
    n_lo = x_lo * inv_rms * w_lo
    n_hi = x_hi * inv_rms * w_hi

    # ---- RoPE for Q and K rows; identity for V ----
    token = tl.where(is_q, row // N_QH, kv_row_safe // N_KVH)
    # For V rows: mask=False → other=1.0 (cos) / 0.0 (sin) → rotation is identity
    need_rope = is_q | is_k
    cos_lo = tl.load(cos_ptr + token * H + offs, mask=need_rope, other=1.0).to(tl.float32)
    cos_hi = tl.load(cos_ptr + token * H + offs + BLOCK_H, mask=need_rope, other=1.0).to(tl.float32)
    sin_lo = tl.load(sin_ptr + token * H + offs, mask=need_rope, other=0.0).to(tl.float32)
    sin_hi = tl.load(sin_ptr + token * H + offs + BLOCK_H, mask=need_rope, other=0.0).to(tl.float32)

    # rotate_half: rh_lo = -n_hi, rh_hi = n_lo
    # out_lo = n_lo*cos_lo - n_hi*sin_lo  (same formula works for V: cos=1,sin=0 → out=n)
    # out_hi = n_hi*cos_hi + n_lo*sin_hi
    out_lo = n_lo * cos_lo - n_hi * sin_lo
    out_hi = n_hi * cos_hi + n_lo * sin_hi

    # ---- Store ----
    tl.store(q_out_ptr + row * H + offs, out_lo.to(tl.bfloat16), mask=is_q)
    tl.store(q_out_ptr + row * H + offs + BLOCK_H, out_hi.to(tl.bfloat16), mask=is_q)
    tl.store(k_out_ptr + kv_row_safe * H + offs, out_lo.to(tl.bfloat16), mask=is_k)
    tl.store(k_out_ptr + kv_row_safe * H + offs + BLOCK_H, out_hi.to(tl.bfloat16), mask=is_k)
    tl.store(v_out_ptr + v_row_safe * H + offs, out_lo.to(tl.bfloat16), mask=is_v)
    tl.store(v_out_ptr + v_row_safe * H + offs + BLOCK_H, out_hi.to(tl.bfloat16), mask=is_v)


@triton.jit
def _rope_qk_kernel(
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    q_out_ptr,
    k_out_ptr,
    N_Q,
    N_KV,
    N_QH: tl.constexpr,
    N_KVH: tl.constexpr,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Apply RoPE rotation to pre-normed Q and K (no V).

    Grid: (N_Q + N_KV,) — one program per q or k head row.
    Used in the large-T fallback of gemma4_qkv_norm_rope after flashinfer
    has already applied the per-head RMSNorm to q and k.
    Same half-split approach as _qkv_norm_rope_kernel.
    """
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)

    is_q = row < N_Q
    kv_row = row - N_Q
    kv_row_safe = tl.where(is_q, 0, kv_row)

    # Load first/second half of q or k
    xq_lo = tl.load(q_ptr + row * H + offs, mask=is_q, other=0.0).to(tl.float32)
    xq_hi = tl.load(q_ptr + row * H + offs + BLOCK_H, mask=is_q, other=0.0).to(tl.float32)
    xk_lo = tl.load(k_ptr + kv_row_safe * H + offs, mask=~is_q, other=0.0).to(tl.float32)
    xk_hi = tl.load(k_ptr + kv_row_safe * H + offs + BLOCK_H, mask=~is_q, other=0.0).to(tl.float32)
    x_lo = xq_lo + xk_lo
    x_hi = xq_hi + xk_hi

    # Token index and cos/sin
    token = tl.where(is_q, row // N_QH, kv_row_safe // N_KVH)
    cos_lo = tl.load(cos_ptr + token * H + offs).to(tl.float32)
    cos_hi = tl.load(cos_ptr + token * H + offs + BLOCK_H).to(tl.float32)
    sin_lo = tl.load(sin_ptr + token * H + offs).to(tl.float32)
    sin_hi = tl.load(sin_ptr + token * H + offs + BLOCK_H).to(tl.float32)

    out_lo = x_lo * cos_lo - x_hi * sin_lo
    out_hi = x_hi * cos_hi + x_lo * sin_hi

    tl.store(q_out_ptr + row * H + offs, out_lo.to(tl.bfloat16), mask=is_q)
    tl.store(q_out_ptr + row * H + offs + BLOCK_H, out_hi.to(tl.bfloat16), mask=is_q)
    tl.store(k_out_ptr + kv_row_safe * H + offs, out_lo.to(tl.bfloat16), mask=~is_q)
    tl.store(k_out_ptr + kv_row_safe * H + offs + BLOCK_H, out_hi.to(tl.bfloat16), mask=~is_q)


@torch.library.custom_op("auto_deploy::gemma4_qkv_norm_rope", mutates_args=(), device_types="cuda")
def gemma4_qkv_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused per-head RMSNorm + RoPE for Q, K, V: one Triton kernel instead of ~3.

    Replaces three torch_rmsnorm calls and the torch_rope_with_explicit_cos_sin call
    with a single fused Triton kernel. Expected savings at c=1 (decode T=1):
    2 fewer kernel dispatches per layer × 30 layers = 60 × 1.3µs = 78µs ≈ 1.4%.

    Small T (total QKV rows ≤ _QKV_TRITON_THRESHOLD): pure inline Triton kernel —
    both RMSNorm and RoPE rotation computed in a single pass, no nested custom op calls.

    Large T: gemma4_qkv_norm (flashinfer, 3 kernels) + Triton rope-only kernel (1 kernel).
    Large-T rope uses a dedicated _rope_qk_kernel to avoid the eager-explosion
    caused by calling torch_rope_with_explicit_cos_sin from inside a custom op body.

    cos/sin shape: [BS, seq_len, head_dim] (already indexed by position_ids).
    Both tensors are 2D-flattened to [T, H] inside the kernel (T = BS*seq_len).
    """
    import flashinfer as _flashinfer

    H = q.shape[-1]
    q_2d = q.reshape(-1, H)
    k_2d = k.reshape(-1, H)
    v_2d = v.reshape(-1, H)
    N_Q = q_2d.shape[0]
    N_KV = k_2d.shape[0]
    N_QH = N_Q // max(cos.reshape(-1, H).shape[0], 1)  # q heads per token
    N_KVH = N_KV // max(cos.reshape(-1, H).shape[0], 1)  # kv heads per token
    cos_2d = cos.reshape(-1, H)
    sin_2d = sin.reshape(-1, H)

    q_out = torch.empty_like(q_2d)
    k_out = torch.empty_like(k_2d)
    v_out = torch.empty_like(v_2d)

    total_rows = N_Q + 2 * N_KV
    if total_rows <= _QKV_TRITON_THRESHOLD:
        H_half = H // 2
        _qkv_norm_rope_kernel[(total_rows,)](
            q_2d,
            k_2d,
            v_2d,
            w_q,
            w_k,
            w_v,
            cos_2d,
            sin_2d,
            q_out,
            k_out,
            v_out,
            N_Q=N_Q,
            N_KV=N_KV,
            N_QH=N_QH,
            N_KVH=N_KVH,
            H=H,
            eps=eps,
            BLOCK_H=H_half,
            num_warps=8,
        )
    else:
        # Large T: flashinfer per-head RMSNorm then Triton rope (1 kernel for q+k)
        _flashinfer.norm.rmsnorm(q_2d, w_q, eps, out=q_out)
        _flashinfer.norm.rmsnorm(k_2d, w_k, eps, out=k_out)
        _flashinfer.norm.rmsnorm(v_2d, w_v, eps, out=v_out)
        # Apply RoPE to q_out and k_out in-place via Triton kernel (avoids eager explosion)
        q_roped = torch.empty_like(q_out)
        k_roped = torch.empty_like(k_out)
        H_half = H // 2
        _rope_qk_kernel[(N_Q + N_KV,)](
            q_out,
            k_out,
            cos_2d,
            sin_2d,
            q_roped,
            k_roped,
            N_Q=N_Q,
            N_KV=N_KV,
            N_QH=N_QH,
            N_KVH=N_KVH,
            H=H,
            BLOCK_H=H_half,
            num_warps=8,
        )
        q_out = q_roped
        k_out = k_roped

    return q_out.reshape(q.shape), k_out.reshape(k.shape), v_out.reshape(v.shape)


@gemma4_qkv_norm_rope.register_fake
def _gemma4_qkv_norm_rope_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
