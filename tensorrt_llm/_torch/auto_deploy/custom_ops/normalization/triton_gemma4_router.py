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

"""Fused Gemma4 MoE Router kernel: RMSNorm + scale + proj + softmax + topk + normalize.

Fuses 9 separate PyTorch kernel launches into a single Triton kernel per token.
Key design decisions:
- W is transposed to W_T[H, E] so inner-H loop accesses W_T[h, 0:E] contiguously (coalesced).
- Single pass over H: compute RMSNorm variance AND (x*scale*W) simultaneously.
  This works because score[e] = norm_factor * sum_h(x[h]*scale[h]*W[e,h]), so we
  accumulate raw (un-normalized) scores and multiply by norm_factor at the end.
- Softmax and topk done in registers after the H loop.
- TopK masking: use tl.max + sentinel pattern (from triton_routing.py) to correctly
  handle ties and 0-d index tensors across all T values.
- Multi-token programs: N_TOKENS tokens per program reuse the W_T tile across tokens,
  reducing W_T bandwidth by N_TOKENS for large-T prefill shapes.
- Fast softmax: use exp2(x * log2e) instead of exp(x) for ~10% faster exp.

Multi-SM router for small decode batch sizes:
- Problem: at T=1..32, a single SM does [T, H=2816] @ [H, E=128] with H reading bottleneck.
  The H=2816 weight bandwidth limits throughput to ~20µs on one SM.
- Solution: split H across NUM_SPLITS SMs (parallel partial reduction), then one finalize kernel.
  Two-kernel approach; in CUDA-graph mode the launch overhead is ~1µs vs ~20µs GPU time.
- CUDA-graph benchmarks (H100):
    T=1:  single=20.7µs vs multi8=7.7µs  (2.7× faster)
    T=16: single=21.3µs vs multi8=8.7µs  (2.5× faster)
    T=64: single=21.9µs vs multi4=13.5µs (1.6× faster)
    T=256: single=43.8µs vs multi4=37.5µs (1.2× faster)
- Threshold: T ≤ 32 → splits=8; 32 < T ≤ _MULTI_SM_MAX_T → splits=4; else single-SM.
"""

import math as _math
import weakref
from typing import Dict, Tuple

import torch
import triton
import triton.language as tl

# Cache transposed router weights (proj_weight [E, H] → proj_T [H, E]).
# Avoids a per-call .contiguous() copy that would otherwise appear as an extra
# CUDA graph node on every decode step.  Keyed by id(proj_weight).
# PyTorch tensors define __eq__ returning a tensor, so we cannot use them directly
# as WeakKeyDictionary keys — we key by integer id instead and register a weakref
# finalizer so the entry is evicted automatically when the weight tensor is freed.
_proj_T_cache: Dict[int, torch.Tensor] = {}
_proj_weight_refs: Dict[int, "weakref.ref[torch.Tensor]"] = {}


@triton.jit
def _topk_and_store(
    probs,  # [E] softmax probabilities
    e_offs,  # [E] expert offsets
    weights_ptr,
    indices_ptr,
    out_offset,  # base offset for this token's outputs
    E: tl.constexpr,
    K: tl.constexpr,
):
    """Iterative top-K selection with sentinel pattern; stores K weights and indices."""
    k_offs = tl.arange(0, K)
    top_vals = tl.zeros([K], dtype=tl.float32)
    top_idxs = tl.zeros([K], dtype=tl.int32)
    top_sum = tl.zeros([1], dtype=tl.float32)

    for i in tl.static_range(K):
        max_val = tl.max(probs, 0)
        is_max = probs == max_val
        candidate = tl.where(is_max, e_offs, E)
        max_idx = tl.min(candidate, 0)

        ki_mask = k_offs == i
        top_vals = tl.where(ki_mask, max_val, top_vals)
        top_idxs = tl.where(ki_mask, max_idx.to(tl.int32), top_idxs)
        top_sum += max_val
        probs = tl.where(e_offs == max_idx, float("-inf"), probs)

    tl.store(weights_ptr + out_offset + k_offs, top_vals / top_sum)
    tl.store(indices_ptr + out_offset + k_offs, top_idxs)


# ---------------------------------------------------------------------------
# Multi-SM router kernels
# Two-kernel approach: parallel partial H-reduction across NUM_SPLITS SMs,
# then a single finalize kernel that reduces partials + softmax + topk.
# ---------------------------------------------------------------------------

# Thresholds for adaptive multi-SM dispatch.  Tuned for H100 (132 SMs):
#   T ≤ _MULTI_SM_T8: use NUM_SPLITS=8  (best for T=1..32, ~2.5-2.7× speedup)
#   T ≤ _MULTI_SM_T4: use NUM_SPLITS=4  (best for T=33..512, ~1.2-1.6× speedup)
#   T >  _MULTI_SM_T4: single-SM (T is large enough to fill the GPU)
_MULTI_SM_T8: int = 32
_MULTI_SM_T4: int = 512


@triton.jit
def _router_partial_fwd(
    hidden_ptr,
    scale_ptr,
    proj_T_ptr,
    partial_var_ptr,
    partial_scores_ptr,
    stride_th,
    stride_hE,
    T,
    H: tl.constexpr,
    E: tl.constexpr,
    H_PER_SPLIT: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    """Partial router: each program computes one split of one token.

    grid = (T * NUM_SPLITS,); prog_id encodes (token, split) as prog_id = t*NS + split_id.
    Accumulates partial variance and partial score vector [E] for its H slice,
    writing to partial_var[t, split] and partial_scores[t, split, :].
    """
    prog_id = tl.program_id(0)
    split_id = prog_id % NUM_SPLITS
    t = prog_id // NUM_SPLITS

    h_offset = split_id * H_PER_SPLIT
    h_offs = tl.arange(0, BLOCK_H)
    e_offs = tl.arange(0, E)

    var = tl.zeros([1], dtype=tl.float32)
    scores = tl.zeros([E], dtype=tl.float32)

    for h_base in tl.range(0, H_PER_SPLIT, BLOCK_H):
        h_idx = h_offset + h_base + h_offs
        split_mask = (h_base + h_offs) < H_PER_SPLIT
        global_mask = h_idx < H
        mask = split_mask & global_mask
        token_mask = (t < T) & mask

        x = tl.load(hidden_ptr + t * stride_th + h_idx, mask=token_mask, other=0.0).to(tl.float32)
        s = tl.load(scale_ptr + h_idx, mask=mask, other=0.0).to(tl.float32)

        var += tl.sum(x * x, 0)

        w_T = tl.load(
            proj_T_ptr + h_idx[:, None] * stride_hE + e_offs[None, :],
            mask=mask[:, None],
            other=0.0,
        ).to(tl.float32)

        xs = (x * s)[:, None]
        scores += tl.sum(xs * w_T, 0)

    tl.store(partial_var_ptr + t * NUM_SPLITS + split_id, tl.sum(var, 0))
    tl.store(partial_scores_ptr + (t * NUM_SPLITS + split_id) * E + e_offs, scores)


@triton.jit
def _router_finalize_fwd(
    partial_var_ptr,
    partial_scores_ptr,
    weights_ptr,
    indices_ptr,
    T,
    H: tl.constexpr,
    E: tl.constexpr,
    K: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    root_size,
    eps,
    LOG2E: tl.constexpr,
):
    """Finalize: sum partials, apply RMSNorm + softmax + topk for one token.

    grid = (T,); each program processes one token's partial results.
    """
    t = tl.program_id(0)
    e_offs = tl.arange(0, E)
    s_offs = tl.arange(0, NUM_SPLITS)

    # Reduce partial vars [NUM_SPLITS] -> scalar
    var_parts = tl.load(partial_var_ptr + t * NUM_SPLITS + s_offs)
    var = tl.sum(var_parts, 0)

    # Reduce partial scores [NUM_SPLITS, E] -> [E]
    scores = tl.zeros([E], dtype=tl.float32)
    for s in tl.static_range(NUM_SPLITS):
        sp = tl.load(partial_scores_ptr + (t * NUM_SPLITS + s) * E + e_offs)
        scores = scores + sp

    # Apply RMSNorm scaling
    norm_factor = tl.rsqrt(var / H + eps) * root_size
    scores = scores * norm_factor

    # Softmax using exp2 for speed
    max_s = tl.max(scores, 0)
    exp_s = tl.exp2((scores - max_s) * LOG2E)
    sum_exp = tl.sum(exp_s, 0)
    probs = exp_s / sum_exp

    _topk_and_store(probs, e_offs, weights_ptr, indices_ptr, t * K, E, K)


def gemma4_router_multi_sm(
    hidden: torch.Tensor,  # [T, H]
    scale: torch.Tensor,  # [H]
    proj_T: torch.Tensor,  # [H, E]
    root_size: float,
    eps: float,
    top_k: int,
    num_splits: int = 8,
    block_h: int = 256,
    num_warps_partial: int = 4,
) -> tuple:
    """Multi-SM router: parallelises H across num_splits SMs per token.

    Faster than single-SM for T ≤ 512 on H100.  Returns the same
    (top_k_weights [T,K], top_k_indices [T,K]) as gemma4_router_triton.
    """
    T, H = hidden.shape
    E = proj_T.shape[1]
    K = top_k
    log2e = _math.log2(_math.e)

    assert H % num_splits == 0, f"H={H} must be divisible by num_splits={num_splits}"
    H_per_split = H // num_splits

    out_weights = torch.empty((T, K), dtype=torch.float32, device=hidden.device)
    out_indices = torch.empty((T, K), dtype=torch.int32, device=hidden.device)
    partial_var = torch.empty((T, num_splits), dtype=torch.float32, device=hidden.device)
    partial_scores = torch.empty((T, num_splits, E), dtype=torch.float32, device=hidden.device)

    grid1 = (T * num_splits,)
    _router_partial_fwd[grid1](
        hidden,
        scale,
        proj_T,
        partial_var,
        partial_scores,
        hidden.stride(0),
        proj_T.stride(0),
        T=T,
        H=H,
        E=E,
        H_PER_SPLIT=H_per_split,
        BLOCK_H=block_h,
        NUM_SPLITS=num_splits,
        num_warps=num_warps_partial,
        num_stages=2,
    )

    grid2 = (T,)
    _router_finalize_fwd[grid2](
        partial_var,
        partial_scores,
        out_weights,
        out_indices,
        T=T,
        H=H,
        E=E,
        K=K,
        NUM_SPLITS=num_splits,
        root_size=root_size,
        eps=eps,
        LOG2E=log2e,
        num_warps=2,
        num_stages=1,
    )

    return out_weights, out_indices


@triton.jit
def _gemma4_router_fwd(
    hidden_ptr,  # [T, H] input hidden states
    scale_ptr,  # [H]    per-dim router scale
    proj_T_ptr,  # [H, E] transposed proj weight (W.T of nn.Linear weight [E, H])
    weights_ptr,  # [T, K] output top-k weights (normalized)
    indices_ptr,  # [T, K] output top-k expert indices (int32)
    stride_th,  # stride of hidden along token dim (= H)
    stride_hE,  # stride of proj_T along H dim (= E)
    T,  # total number of tokens (runtime, for boundary check)
    H: tl.constexpr,
    E: tl.constexpr,  # must be power of 2 (128)
    K: tl.constexpr,  # must be power of 2 (8)
    root_size,
    eps,
    BLOCK_H: tl.constexpr,  # tile size along H
    N_TOKENS: tl.constexpr,  # tokens per program (1 for decode, 2-4 for prefill)
    LOG2E: tl.constexpr,  # tl.math.log2e() as constexpr for fast exp2
):
    prog_id = tl.program_id(0)
    t_base = prog_id * N_TOKENS

    h_offs = tl.arange(0, BLOCK_H)
    e_offs = tl.arange(0, E)

    # For multi-token programs: loop over N_TOKENS tokens per program.
    # When T is not a multiple of N_TOKENS, the last program may have fewer tokens.
    # We mask out-of-range tokens by conditioning on t < T.
    for n in tl.static_range(N_TOKENS):
        t = t_base + n

        # Accumulators (scalar for var, vector [E] for scores)
        var = tl.zeros([1], dtype=tl.float32)
        scores = tl.zeros([E], dtype=tl.float32)

        # Single pass over H: accumulate variance and (x*scale) @ W simultaneously
        for h_base in tl.range(0, H, BLOCK_H):
            h_idx = h_base + h_offs
            mask = h_idx < H

            # Load hidden [BLOCK_H] and scale [BLOCK_H]
            # Mask out-of-range tokens with other=0.0 (var and scores stay 0)
            token_mask = (t < T) & mask
            x = tl.load(hidden_ptr + t * stride_th + h_idx, mask=token_mask, other=0.0).to(
                tl.float32
            )
            s = tl.load(scale_ptr + h_idx, mask=mask, other=0.0).to(tl.float32)

            # Accumulate variance (without applying norm yet)
            var += tl.sum(x * x, 0)

            # Load W_T[h_base:h_base+BLOCK_H, 0:E] as [BLOCK_H, E] tile
            w_T = tl.load(
                proj_T_ptr + h_idx[:, None] * stride_hE + e_offs[None, :],
                mask=mask[:, None],
                other=0.0,
            ).to(tl.float32)  # [BLOCK_H, E]

            # Accumulate scores: scores += (x * s) @ w_T  -> [E]
            xs = (x * s)[:, None]  # [BLOCK_H, 1]
            scores += tl.sum(xs * w_T, 0)  # [E]

        # Only write outputs for valid tokens
        if t < T:
            # Apply RMSNorm scaling
            norm_factor = tl.rsqrt(var / H + eps) * root_size
            scores = scores * norm_factor  # [E]

            # Softmax over E experts using exp2 for speed
            max_s = tl.max(scores, 0)
            exp_s = tl.exp2((scores - max_s) * LOG2E)
            sum_exp = tl.sum(exp_s, 0)
            probs = exp_s / sum_exp  # [E]

            # TopK and store
            _topk_and_store(probs, e_offs, weights_ptr, indices_ptr, t * K, E, K)


def gemma4_router_triton(
    hidden: torch.Tensor,  # [T, H]
    scale: torch.Tensor,  # [H]
    proj_T: torch.Tensor,  # [H, E] transposed proj weight
    root_size: float,
    eps: float,
    top_k: int,
    num_warps: int = 8,
    num_stages: int = 2,
    block_h: int = 512,
    n_tokens: int = 1,
) -> tuple:
    """Fused Gemma4 router: returns (top_k_weights [T,K], top_k_indices [T,K]).

    Default params nw=8, ns=2, bh=512, n_tokens=1 are tuned for H100 decode.
    For large prefill (T>=256), use n_tokens=2-4 to amortize W_T bandwidth.
    """
    T, H = hidden.shape
    E = proj_T.shape[1]
    K = top_k

    out_weights = torch.empty((T, K), dtype=torch.float32, device=hidden.device)
    out_indices = torch.empty((T, K), dtype=torch.int32, device=hidden.device)

    log2e = _math.log2(_math.e)

    grid = (triton.cdiv(T, n_tokens),)
    _gemma4_router_fwd[grid](
        hidden,
        scale,
        proj_T,
        out_weights,
        out_indices,
        hidden.stride(0),  # stride_th = H
        proj_T.stride(0),  # stride_hE = E
        T=T,
        H=H,
        E=E,
        K=K,
        root_size=root_size,
        eps=eps,
        BLOCK_H=block_h,
        N_TOKENS=n_tokens,
        LOG2E=log2e,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out_weights, out_indices


# ---------------------------------------------------------------------------
# Custom op registration — needed for FakeTensor / meta tensor tracing in AD
# ---------------------------------------------------------------------------


@torch.library.custom_op("auto_deploy::gemma4_router", mutates_args=(), device_types="cuda")
def gemma4_router(
    hidden: torch.Tensor,
    scale: torch.Tensor,
    proj_weight: torch.Tensor,
    root_size: float,
    eps: float,
    top_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused Gemma4 router custom op: RMSNorm + proj + softmax + topk.

    Accepts ``proj_weight`` in its natural ``[E, H]`` (nn.Linear) layout and
    transposes it to ``[H, E]`` internally so callers can pass ``self.proj.weight``
    directly without maintaining a separate transposed buffer.

    Registered as a torch custom op so that FakeTensor / meta-tensor tracing
    during torch.export sees only shape inference (via ``register_fake``) rather
    than the actual Triton kernel, which cannot execute on meta tensors.

    Adaptive multi-SM dispatch based on batch size T:
    - T ≤ _MULTI_SM_T8 (32):  2-kernel multi-SM with NUM_SPLITS=8  (~2.7× faster at T=1)
    - T ≤ _MULTI_SM_T4 (512): 2-kernel multi-SM with NUM_SPLITS=4  (~1.2-1.6× faster)
    - T > _MULTI_SM_T4:        single-SM (H is fully parallelised by enough tokens)

    The proj_weight [E, H] is cached in transposed [H, E] form in ``_proj_T_cache``
    so the per-call .contiguous() copy is a one-time cost at first use rather
    than a repeated CUDA graph node.
    """
    key = id(proj_weight)
    proj_T = _proj_T_cache.get(key)
    if proj_T is None:
        proj_T = proj_weight.t().contiguous()
        _proj_T_cache[key] = proj_T

        # Register a finalizer so the cache entry is evicted when proj_weight is freed,
        # preventing stale hits if a new tensor is later allocated at the same address.
        def _evict(ref, k=key):
            _proj_T_cache.pop(k, None)
            _proj_weight_refs.pop(k, None)

        _proj_weight_refs[key] = weakref.ref(proj_weight, _evict)
    T = hidden.shape[0]
    if T <= _MULTI_SM_T8:
        return gemma4_router_multi_sm(hidden, scale, proj_T, root_size, eps, top_k, num_splits=8)
    elif T <= _MULTI_SM_T4:
        return gemma4_router_multi_sm(hidden, scale, proj_T, root_size, eps, top_k, num_splits=4)
    else:
        # Single-SM for very large T (GPU is already fully loaded by token parallelism)
        return gemma4_router_triton(hidden, scale, proj_T, root_size, eps, top_k)


@gemma4_router.register_fake
def _gemma4_router_fake(
    hidden: torch.Tensor,
    scale: torch.Tensor,
    proj_weight: torch.Tensor,
    root_size: float,
    eps: float,
    top_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    T = hidden.shape[0]
    return (
        torch.empty((T, top_k), dtype=torch.float32, device=hidden.device),
        torch.empty((T, top_k), dtype=torch.int32, device=hidden.device),
    )


# ---------------------------------------------------------------------------
# Router fence: piecewise CUDA-graph partition boundary
#
# Inserted by the multi_stream_moe transform immediately after the router's
# weights output.  Being a registered dynamic op (see piecewise_utils.py),
# it forces the FX-graph splitter to create a new partition here.  The
# partition BEFORE the fence contains only the router (and the preceding
# 3-norm fusion) — a pure main-stream region with no @dynamo.disable
# stream-switch calls — so it is captured as a CUDA graph.  The partition
# AFTER the fence inherits begin_aux_stream_passthrough and friends and is
# reclassified as dynamic (eager) as before.
#
# The op is a mathematical no-op (identity): weights_fenced == weights.
# ---------------------------------------------------------------------------


@torch._dynamo.disable
def gemma4_router_fence(weights: torch.Tensor) -> torch.Tensor:
    """No-op identity used as a piecewise-CUDA-graph partition boundary.

    Decorated with @torch._dynamo.disable so torch.compile cannot optimize
    this call away.  Being opaque to dynamo, it appears as a ``call_function``
    node in the FX graph, which the piecewise splitter recognises as a dynamic
    op (via the "gemma4_router_fence" name in _PARTITION_BOUNDARY_OPS) and
    uses as a cut point.  The partition BEFORE the fence (containing the router
    and the preceding 3-norm fusion) is captured as a CUDA graph; the partition
    AFTER (containing begin_aux_stream_passthrough and the aux-stream MoE block)
    is reclassified as dynamic (eager).
    """
    return weights
