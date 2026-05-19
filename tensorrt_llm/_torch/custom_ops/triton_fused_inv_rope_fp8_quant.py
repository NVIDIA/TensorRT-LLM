# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from vLLM:
#   https://github.com/vllm-project/vllm/blob/ecd0b60aad2f4e28dd00ababfc1402690d88cbed/vllm/v1/attention/ops/deepseek_v4_ops/fused_inv_rope_fp8_quant.py
#
# Modifications relative to upstream vLLM (Apache-2.0, licensed identically to
# TensorRT-LLM):
#  * Added a NEOX RoPE branch (`IS_NEOX` constexpr). Upstream is interleaved
#    (GPT-J style) only; DeepSeek-V4 in TensorRT-LLM uses NEOX layout for the
#    inverse RoPE applied before `o_a_proj` (see
#    `cpp/tensorrt_llm/kernels/mlaKernels.cu:1190-1276`).
#  * Output FP8 buffer is returned without the trailing `transpose(0, 1)` so
#    the result is directly consumable by `cute_dsl_fp8_bmm_blackwell` whose
#    input layout is `[batch=n_groups, m=num_tokens, k=d]`. Likewise the scale
#    buffer is returned as a contiguous `[n_groups, num_scale_blocks, pad_up(num_tokens, 4)]`
#    FP32 tensor (matching the output of `fp8_batched_quantize_1x128_permute102`)
#    rather than the strided `[T, G, num_scale_blocks]` view used by vLLM.
#  * Drops the `tma_aligned_scales`/SM90 split — TensorRT-LLM's consumer is
#    SM100 with FP32 a-side scales, so the upstream UE8M0-INT32 pack path is
#    not needed here.
#  * Rebound the cos/sin cache to TensorRT-LLM's NEOX layout
#    `[max_pos, 2, half_rope]` (cos block then sin block per position) by
#    advertising the contract as a flat `[max_pos, rope_dim]` view; the
#    indexing math is identical for both kernels.
#  * Replaces vLLM's `direct_register_custom_op` with `torch.library.custom_op`,
#    matching the rest of `tensorrt_llm/_torch/custom_ops/`.
#
# Engaged from `tensorrt_llm/_torch/modules/attention.py:_deepseek_v4_o_proj`
# when `use_cute_dsl_blockscaling_bmm` is enabled in `extra_llm_api_options` on
# SM100; falls back to the legacy `mla_rope_inplace + permute102_quant` pair
# otherwise.
"""
Fused inverse-RoPE + 1x128 block-scaled FP8 quantize, for DeepSeek-V4 ``o_a_proj``.

Replaces the (``mla_rope_inplace`` -> ``fp8_batched_quantize_1x128_permute102``)
two-kernel pair in ``_deepseek_v4_o_proj``. The output is consumed verbatim by
``cute_dsl_fp8_bmm_blackwell``.
"""

from __future__ import annotations

import os

import torch
import triton  # type: ignore[import]
import triton.language as tl  # type: ignore[import]

# Lazy import of the CuTe DSL backend. Triton is the default path; CuTe DSL
# is a perf-tied alternative (microbench-verified within ~0.05% of Triton at
# M >= 2048, and within the cross-allocation noise floor at M <= 1024). Opt
# in via `TLLM_USE_CUTE_DSL_FUSED_INV_ROPE=1` for future kernel work (TMA /
# warp-spec V2 will build on this scaffold). Import failure is non-fatal.
try:
    from ..cute_dsl_kernels.blackwell import (  # noqa: F401
        fused_inv_rope_fp8_quant as _cute_dsl_backend,
    )
except Exception:
    _cute_dsl_backend = None  # type: ignore[assignment]


@triton.jit
def _fused_inv_rope_fp8_quant_per_head(
    o_ptr,
    positions_ptr,
    cos_sin_cache_ptr,
    fp8_ptr,
    scale_ptr,
    num_tokens,
    heads_per_group: tl.constexpr,
    o_stride_token,
    o_stride_head,
    cache_stride_pos,
    fp8_stride_group,
    fp8_stride_token,
    scale_stride_group,
    scale_stride_k,
    fp8_max: tl.constexpr,
    eps: tl.constexpr,
    QUANT_GROUP_SIZE: tl.constexpr,
    CHUNKS_PER_HEAD: tl.constexpr,
    ROPE_START: tl.constexpr,
    HALF_ROPE: tl.constexpr,
    IS_NEOX: tl.constexpr,
):
    # Original 1-token-per-block kernel. Used by the Python dispatcher when
    # M < threshold (small-M / GEN-phase shapes), where wrapping the body in
    # an outer loop measurably hurts perf even with BLOCK_TOKENS_M=1 (the if/
    # else around the bulk creates predicated execution that compiles less
    # tightly than a straight return path). See `_fused_inv_rope_fp8_quant_
    # per_head_mblock` for the multi-token path used at M >= 1024.
    # int64: stride multiply overflows int32 past num_tokens=32768 (IMA).
    pid_token = tl.program_id(0).to(tl.int64)
    pid_gh = tl.program_id(1).to(tl.int64)

    g = pid_gh // heads_per_group
    head_in_group = pid_gh % heads_per_group
    global_head = pid_gh
    qb_start = head_in_group * CHUNKS_PER_HEAD

    # Padding rows (M aligned up to 4): zero out the scale and skip quant.
    if pid_token >= num_tokens:
        block_offsets = tl.arange(0, CHUNKS_PER_HEAD)
        qb_indices = qb_start + block_offsets
        scale_addrs = scale_ptr + g * scale_stride_group + pid_token + qb_indices * scale_stride_k
        tl.store(scale_addrs, tl.zeros((CHUNKS_PER_HEAD,), dtype=tl.float32))
        return

    input_base = o_ptr + pid_token * o_stride_token + global_head * o_stride_head

    HEAD_DIM: tl.constexpr = CHUNKS_PER_HEAD * QUANT_GROUP_SIZE
    offsets = tl.arange(0, HEAD_DIM)
    x = tl.load(input_base + offsets).to(tl.float32)

    rope_abs_start: tl.constexpr = (CHUNKS_PER_HEAD - 1) * QUANT_GROUP_SIZE + ROPE_START
    pos = tl.load(positions_ptr + pid_token)
    cache_base = cos_sin_cache_ptr + pos * cache_stride_pos
    is_rope = offsets >= rope_abs_start
    rope_local = offsets - rope_abs_start

    if IS_NEOX:
        # NEOX layout: rope half = [x1[0..H), x2[0..H)] where H = HALF_ROPE.
        # For inverse RoPE (coef = (cos, -sin)):
        #   new_x1[i] = cos[i] * x1[i] + sin[i] * x2[i]
        #   new_x2[i] = cos[i] * x2[i] - sin[i] * x1[i]
        # In a single per-element formulation:
        #   partner_offset = (i < H) ? i + H : i - H
        #   sign = (i < H) ? +1 : -1
        #   new = cos * x[i] + sign * sin * x[partner]
        is_first_half = rope_local < HALF_ROPE
        partner_local = tl.where(is_first_half, rope_local + HALF_ROPE, rope_local - HALF_ROPE)
        partner_abs = rope_abs_start + partner_local
        x_partner = tl.load(input_base + partner_abs, mask=is_rope, other=0.0).to(tl.float32)
        cs_idx = tl.where(is_first_half, rope_local, rope_local - HALF_ROPE)
        cos_v = tl.load(cache_base + cs_idx, mask=is_rope, other=1.0)
        sin_v = tl.load(cache_base + HALF_ROPE + cs_idx, mask=is_rope, other=0.0)
        sign = tl.where(is_first_half, 1.0, -1.0)
        rotated = x * cos_v + sign * sin_v * x_partner
    else:
        # GPT-J / interleaved layout: pairs (x[2i], x[2i+1]).
        x_partner = tl.load(input_base + (offsets ^ 1), mask=is_rope, other=0.0).to(tl.float32)
        cs_idx = tl.maximum(rope_local >> 1, 0)
        cos_v = tl.load(cache_base + cs_idx, mask=is_rope, other=1.0)
        sin_v = tl.load(cache_base + HALF_ROPE + cs_idx, mask=is_rope, other=0.0)
        x_add = x * cos_v + x_partner * sin_v
        x_sub = x * cos_v - x_partner * sin_v
        is_even = (rope_local & 1) == 0
        rotated = tl.where(is_even, x_add, x_sub)

    x = tl.where(is_rope, rotated, x)

    x_2d = tl.reshape(tl.abs(x), (CHUNKS_PER_HEAD, QUANT_GROUP_SIZE))
    block_absmax = tl.maximum(tl.max(x_2d, axis=1), eps)
    # Match `fp8_batched_quantize_1x128_permute102`'s raw FP32 scale layout
    # (no UE8M0 round-up): scales = absmax / fp8_max so that quant value =
    # x / scale = x * fp8_max / absmax. This keeps `cute_dsl_fp8_bmm_blackwell`
    # bit-for-bit on parity with the legacy permute102 quant in the same gate.
    scales = block_absmax * (1.0 / fp8_max)

    scales_exp = tl.reshape(
        tl.broadcast_to(
            tl.reshape(scales, (CHUNKS_PER_HEAD, 1)),
            (CHUNKS_PER_HEAD, QUANT_GROUP_SIZE),
        ),
        (HEAD_DIM,),
    )
    x_quant = tl.clamp(x / scales_exp, -fp8_max, fp8_max).to(tl.float8e4nv)

    fp8_base = (
        fp8_ptr + g * fp8_stride_group + pid_token * fp8_stride_token + qb_start * QUANT_GROUP_SIZE
    )
    tl.store(fp8_base + offsets, x_quant)

    block_offsets = tl.arange(0, CHUNKS_PER_HEAD)
    qb_indices = qb_start + block_offsets
    scale_addrs = scale_ptr + g * scale_stride_group + pid_token + qb_indices * scale_stride_k
    tl.store(scale_addrs, scales)


@triton.jit
def _fused_inv_rope_fp8_quant_per_head_mblock(
    o_ptr,
    positions_ptr,
    cos_sin_cache_ptr,
    fp8_ptr,
    scale_ptr,
    num_tokens,
    heads_per_group: tl.constexpr,
    o_stride_token,
    o_stride_head,
    cache_stride_pos,
    fp8_stride_group,
    fp8_stride_token,
    scale_stride_group,
    scale_stride_k,
    fp8_max: tl.constexpr,
    eps: tl.constexpr,
    QUANT_GROUP_SIZE: tl.constexpr,
    CHUNKS_PER_HEAD: tl.constexpr,
    ROPE_START: tl.constexpr,
    HALF_ROPE: tl.constexpr,
    IS_NEOX: tl.constexpr,
    BLOCK_TOKENS_M: tl.constexpr,
):
    # Multi-token-per-block variant for M >= 1024. Grid X = ceil(M / BTM).
    # Compared to the BTM=1 kernel: fewer total blocks → higher SM occupancy
    # (each SM gets more useful work between launch-issue overheads), with
    # the trade-off of inner-loop unrolling growing the compiled binary by
    # BTM×. At BTM∈{8,16} the unroll cost is comfortably amortized.
    # int64: stride multiply overflows int32 past num_tokens=32768 (IMA).
    pid_x = tl.program_id(0).to(tl.int64)
    pid_gh = tl.program_id(1).to(tl.int64)

    g = pid_gh // heads_per_group
    head_in_group = pid_gh % heads_per_group
    global_head = pid_gh
    qb_start = head_in_group * CHUNKS_PER_HEAD

    HEAD_DIM: tl.constexpr = CHUNKS_PER_HEAD * QUANT_GROUP_SIZE
    rope_abs_start: tl.constexpr = (CHUNKS_PER_HEAD - 1) * QUANT_GROUP_SIZE + ROPE_START
    offsets = tl.arange(0, HEAD_DIM)
    is_rope = offsets >= rope_abs_start
    rope_local = offsets - rope_abs_start
    block_offsets = tl.arange(0, CHUNKS_PER_HEAD)
    qb_indices = qb_start + block_offsets

    # Inner loop over BLOCK_TOKENS_M tokens. tl.range (vs tl.static_range)
    # generates a runtime loop instead of a fully-unrolled body. The
    # num_stages arg here controls cross-iteration software pipelining
    # (load_next overlap with compute_current_and_store) — distinct from
    # the launch-site num_stages which only matters for warpgroup GEMM
    # kernels. Depth=2 is enough to overlap one load with the previous
    # iter's store.
    for m_in_block in tl.range(0, BLOCK_TOKENS_M, num_stages=2):
        pid_token = pid_x * BLOCK_TOKENS_M + m_in_block

        if pid_token >= num_tokens:
            scale_addrs = (
                scale_ptr + g * scale_stride_group + pid_token + qb_indices * scale_stride_k
            )
            tl.store(scale_addrs, tl.zeros((CHUNKS_PER_HEAD,), dtype=tl.float32))
        else:
            input_base = o_ptr + pid_token * o_stride_token + global_head * o_stride_head
            x = tl.load(input_base + offsets).to(tl.float32)

            pos = tl.load(positions_ptr + pid_token)
            cache_base = cos_sin_cache_ptr + pos * cache_stride_pos

            if IS_NEOX:
                is_first_half = rope_local < HALF_ROPE
                partner_local = tl.where(
                    is_first_half, rope_local + HALF_ROPE, rope_local - HALF_ROPE
                )
                partner_abs = rope_abs_start + partner_local
                x_partner = tl.load(input_base + partner_abs, mask=is_rope, other=0.0).to(
                    tl.float32
                )
                cs_idx = tl.where(is_first_half, rope_local, rope_local - HALF_ROPE)
                cos_v = tl.load(cache_base + cs_idx, mask=is_rope, other=1.0)
                sin_v = tl.load(cache_base + HALF_ROPE + cs_idx, mask=is_rope, other=0.0)
                sign = tl.where(is_first_half, 1.0, -1.0)
                rotated = x * cos_v + sign * sin_v * x_partner
            else:
                x_partner = tl.load(input_base + (offsets ^ 1), mask=is_rope, other=0.0).to(
                    tl.float32
                )
                cs_idx = tl.maximum(rope_local >> 1, 0)
                cos_v = tl.load(cache_base + cs_idx, mask=is_rope, other=1.0)
                sin_v = tl.load(cache_base + HALF_ROPE + cs_idx, mask=is_rope, other=0.0)
                x_add = x * cos_v + x_partner * sin_v
                x_sub = x * cos_v - x_partner * sin_v
                is_even = (rope_local & 1) == 0
                rotated = tl.where(is_even, x_add, x_sub)

            x = tl.where(is_rope, rotated, x)

            x_2d = tl.reshape(tl.abs(x), (CHUNKS_PER_HEAD, QUANT_GROUP_SIZE))
            block_absmax = tl.maximum(tl.max(x_2d, axis=1), eps)
            scales = block_absmax * (1.0 / fp8_max)

            scales_exp = tl.reshape(
                tl.broadcast_to(
                    tl.reshape(scales, (CHUNKS_PER_HEAD, 1)),
                    (CHUNKS_PER_HEAD, QUANT_GROUP_SIZE),
                ),
                (HEAD_DIM,),
            )
            x_quant = tl.clamp(x / scales_exp, -fp8_max, fp8_max).to(tl.float8e4nv)

            fp8_base = (
                fp8_ptr
                + g * fp8_stride_group
                + pid_token * fp8_stride_token
                + qb_start * QUANT_GROUP_SIZE
            )
            tl.store(fp8_base + offsets, x_quant)

            scale_addrs = (
                scale_ptr + g * scale_stride_group + pid_token + qb_indices * scale_stride_k
            )
            tl.store(scale_addrs, scales)


def _tma_aligned_size(x: int, tma_align_size_in_elems: int = 4) -> int:
    return (x + tma_align_size_in_elems - 1) // tma_align_size_in_elems * tma_align_size_in_elems


def _choose_block_tokens_m(num_tokens: int) -> int:
    """Pick BLOCK_TOKENS_M based on M.

    Microbench (DEP8 shape, n_groups=8, heads_per_group=16) on GB300:
    - At M < 1024 the single-token-per-block path is the winner — BTM>1 adds
      static-unroll overhead that's not amortized at small block count.
    - At M >= 1024 multi-token blocks reduce grid size (fewer single-warp
      launch costs) and improve SM occupancy. Measured wins:
        M=1024  : BTM=8  → 63 µs vs 70 µs (BTM=1)  ≈ 10% faster
        M=2048+ : BTM=16 → 113 µs vs 137 µs (BTM=1) ≈ 18% faster
        M=8192  : BTM=16 → 448 µs vs 533 µs (BTM=1) ≈ 16% faster
    """
    if num_tokens >= 4096:
        return 32
    if num_tokens >= 2048:
        return 16
    if num_tokens >= 1024:
        return 8
    return 1


def _fused_inv_rope_fp8_quant_impl(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    quant_group_size: int,
    is_neox: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Optional CuTe DSL backend. Microbench-tied with the Triton kernel
    # below (within 0.05 % at M >= 2048 and within the cross-allocation
    # noise floor at M <= 1024 — 4-allocation median on B200 GB200). Kept
    # in tree as an alternative implementation that future TMA / warp-spec
    # work (V2) can build on. Opt in via `TLLM_USE_CUTE_DSL_FUSED_INV_ROPE=1`.
    if (
        _cute_dsl_backend is not None
        and _cute_dsl_backend._fused_inv_rope_fp8_quant_impl_cute_dsl is not None
        and os.environ.get("TLLM_USE_CUTE_DSL_FUSED_INV_ROPE", "0") == "1"
    ):
        return _cute_dsl_backend._fused_inv_rope_fp8_quant_impl_cute_dsl(
            o,
            positions,
            cos_sin_cache,
            n_groups,
            heads_per_group,
            nope_dim,
            rope_dim,
            quant_group_size,
            is_neox,
        )
    num_tokens, num_heads, head_dim = o.shape
    assert num_heads == n_groups * heads_per_group, (
        f"num_heads={num_heads} != n_groups({n_groups}) * heads_per_group({heads_per_group})"
    )
    assert head_dim == nope_dim + rope_dim, (
        f"head_dim={head_dim} != nope_dim({nope_dim}) + rope_dim({rope_dim})"
    )
    assert head_dim % quant_group_size == 0
    expected_nope_mod = quant_group_size - rope_dim
    assert nope_dim % quant_group_size == expected_nope_mod, (
        f"Layout requires nope_dim%{quant_group_size}=="
        f"{quant_group_size}-rope_dim={expected_nope_mod}, "
        f"got {nope_dim % quant_group_size}"
    )
    assert rope_dim % 2 == 0
    assert cos_sin_cache.dtype == torch.float32, (
        f"cos_sin_cache must be fp32, got {cos_sin_cache.dtype}"
    )

    d = heads_per_group * head_dim
    num_scale_blocks = d // quant_group_size
    chunks_per_head = head_dim // quant_group_size

    fp8_dtype = torch.float8_e4m3fn
    fp8_max = torch.finfo(fp8_dtype).max

    block_tokens_m = _choose_block_tokens_m(num_tokens)
    # Pad scale-buffer M dim to max(BLOCK_TOKENS_M, 4): TMA consumer needs
    # 4-alignment, and grid-X = ceil(M / BTM) blocks each writes BTM scale
    # rows — so M must align to BTM too. lcm(BTM, 4) = max(BTM, 4) for
    # BTM ∈ {1, 2, 4, 8, 16, ...} (powers of 2).
    tma_aligned_T = _tma_aligned_size(num_tokens, max(block_tokens_m, 4))

    # FP8 output buffer: fully contiguous [n_groups, num_tokens, d] — must
    # match `fp8_batched_quantize_1x128_permute102`'s contiguous output exactly,
    # because `cute_dsl_fp8_bmm_blackwell` consumes the FP8 buffer via
    # `make_ptr(.data_ptr())` and assumes contiguous `(G, T, K)` strides
    # `(T*K, K, 1)`. Padding the M dim (tma_aligned_T) leaves stride[0] =
    # tma_aligned_T*K, which silently shifts every batch's data when
    # num_tokens % 4 != 0 — manifested as a catastrophic gsm8k regression
    # (96.25 -> 37.38) on Pro TEP8 c128 in the first attempt.
    fp8_buf = torch.empty(
        (n_groups, num_tokens, d),
        dtype=fp8_dtype,
        device=o.device,
    )

    # Scale buffer: contiguous [n_groups, num_scale_blocks, tma_aligned_T] FP32.
    scale_buf = torch.empty(
        (n_groups, num_scale_blocks, tma_aligned_T),
        dtype=torch.float32,
        device=o.device,
    )
    # The kernel addresses scales via `scale_ptr + g * scale_stride_group +
    # token_idx + qb_indices * scale_stride_k`, i.e. layout
    # `[n_groups, num_scale_blocks, tma_aligned_T]` with the inner stride 1
    # along tokens.  scale_stride_group = num_scale_blocks * tma_aligned_T,
    # scale_stride_k = tma_aligned_T.

    # Pass cos_sin_cache as a flat [max_pos, rope_dim] view. TensorRT-LLM
    # stores it as [max_pos, 2, rope_dim/2] (cos block then sin block per
    # position) which is bit-identical layout — just a different shape.
    cos_sin_view = cos_sin_cache.view(cos_sin_cache.shape[0], -1)
    assert cos_sin_view.shape[-1] == rope_dim, (
        f"cos_sin_cache last dim {cos_sin_view.shape[-1]} != rope_dim {rope_dim}"
    )

    # positions can be int32 or int64; the kernel does an opaque tl.load.
    if positions.dtype != torch.int32 and positions.dtype != torch.int64:
        positions = positions.to(torch.int64)

    common_kwargs = dict(
        heads_per_group=heads_per_group,
        o_stride_token=o.stride(0),
        o_stride_head=o.stride(1),
        cache_stride_pos=cos_sin_view.stride(0),
        fp8_stride_group=fp8_buf.stride(0),
        fp8_stride_token=fp8_buf.stride(1),
        scale_stride_group=scale_buf.stride(0),
        scale_stride_k=scale_buf.stride(1),
        fp8_max=fp8_max,
        eps=1e-10,
        QUANT_GROUP_SIZE=quant_group_size,
        CHUNKS_PER_HEAD=chunks_per_head,
        ROPE_START=nope_dim % quant_group_size,
        HALF_ROPE=rope_dim // 2,
        IS_NEOX=is_neox,
    )

    if block_tokens_m == 1:
        # Small-M / GEN path: single-token-per-block kernel (no outer loop).
        # Bit-for-bit equivalent to the original V1 kernel.
        grid = (tma_aligned_T, n_groups * heads_per_group)
        _fused_inv_rope_fp8_quant_per_head[grid](
            o,
            positions,
            cos_sin_view,
            fp8_buf,
            scale_buf,
            num_tokens,
            **common_kwargs,
            num_warps=1,
            num_stages=1,
        )
    else:
        # Large-M / CTX path: multi-token-per-block kernel. num_stages=2
        # lets Triton interleave the next iter's TMA load with the current
        # iter's compute+store across the unrolled BLOCK_TOKENS_M body.
        # num_warps tuning: per-block work scales with BTM × HEAD_DIM.
        # V2.3 microbench at M=8192 DEP8 shape:
        #   BTM=16 nw=1 → 448 µs (3.6 TB/s, 45% peak)
        #   BTM=16 nw=2 → 420 µs (3.8 TB/s, 48% peak)  ← best
        #   BTM=16 nw=4 → 555 µs (2.9 TB/s, 36% peak)  ← reg-pressure regression
        # So nw=2 for BTM>=8 (more threads help amortize the larger per-block
        # load fan-out), nw=1 for BTM=1 (V1 ruled out nw>1 there).
        nw = 2 if block_tokens_m >= 8 else 1
        # num_stages: software-pipelining depth across the inner loop.
        # BTM=32 has more iterations to pipeline so stages=3 has more value.
        ns = 3 if block_tokens_m >= 32 else 2
        grid = (tma_aligned_T // block_tokens_m, n_groups * heads_per_group)
        _fused_inv_rope_fp8_quant_per_head_mblock[grid](
            o,
            positions,
            cos_sin_view,
            fp8_buf,
            scale_buf,
            num_tokens,
            **common_kwargs,
            BLOCK_TOKENS_M=block_tokens_m,
            num_warps=nw,
            num_stages=ns,
        )
    return fp8_buf, scale_buf


def _fused_inv_rope_fp8_quant_fake(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    quant_group_size: int,
    is_neox: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens, num_heads, head_dim = o.shape
    d = heads_per_group * head_dim
    num_scale_blocks = d // quant_group_size
    block_tokens_m = _choose_block_tokens_m(num_tokens)
    tma_aligned_T = _tma_aligned_size(num_tokens, max(block_tokens_m, 4))
    fp8_buf = torch.empty(
        (n_groups, num_tokens, d),
        dtype=torch.float8_e4m3fn,
        device=o.device,
    )
    scale_buf = torch.empty(
        (n_groups, num_scale_blocks, tma_aligned_T),
        dtype=torch.float32,
        device=o.device,
    )
    return fp8_buf, scale_buf


# Register as a torch custom op so dynamo / cudagraph see an opaque boundary.
torch.library.define(
    "trtllm::fused_inv_rope_fp8_quant_vllm_port",
    "(Tensor o, Tensor positions, Tensor cos_sin_cache, "
    "int n_groups, int heads_per_group, int nope_dim, int rope_dim, "
    "int quant_group_size, bool is_neox) -> (Tensor, Tensor)",
)
torch.library.impl(
    "trtllm::fused_inv_rope_fp8_quant_vllm_port",
    "CUDA",
)(_fused_inv_rope_fp8_quant_impl)
torch.library.register_fake(
    "trtllm::fused_inv_rope_fp8_quant_vllm_port",
)(_fused_inv_rope_fp8_quant_fake)
