# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 DSA indexer: the q side and the K-cache update in one launch (MTP3 / MTP5 wide decode).

After its projections an indexer layer's decode step runs two short kernels back to back:

* ``self_attn_indexer_q_side_fused``: rope + e4m3 pack of the indexer q and the finalised
  head weights, 16 (token, head) rows of 8 lanes per 128-thread CTA;
* ``self_attn_indexer_k_norm_rope_fp8_cache_update``: LayerNorm + rope + e4m3 pack of this
  step's keys, scattered into the paged K cache, 16 lanes per token.

Neither reads what the other writes. At the wide decode rows (65..256 tokens) each is a
few microseconds, much of it the PDL hand-off and the tail. Here one grid runs both: CTAs
``[0, q_blocks)`` are the q side's (``indexer_q_side_rows``, 16 rows each) and CTAs
``[q_blocks, q_blocks + k_blocks)`` the K update's (``indexer_k_cache_update_rows``, 8 tokens
each: the 128 threads of a q-side CTA). Every CTA runs its kernel's own row body, so every
byte written is that kernel's (ue8m0 scales, 8-byte key loads). Both bodies run their decode
variants: bf16 ``[T, 128]`` keys (``F32_INPUT=False``) and packed ``[T, 32]`` head weights
(``WQ_PITCH=32``), not the context projection's in-place fp32 reads.

PDL: every thread waits (``griddepcontrol.wait``) before its first global read, the launch
carries the programmatic-serialisation attribute, and the dependents are released at
completion -- as both kernels do.

ABI: :func:`compile_indexer_q_side_k_cache` returns a callable taking ``(q, position_ids,
cos_sin, wq, q_fp8, weights, indexer_k, norm_weight, norm_bias, rotary_cos_sin, k_cache,
slot_mapping_fp8, slot_mapping_scale, num_tokens, weight_scale_factor, cache_stride_0,
cache_stride_1, cache_stride_2, cache_stride_3, stream)``; each tensor as its kernel takes it
(``position_ids`` is both kernels'; ``cos_sin`` ``[max_pos, 64]`` and ``rotary_cos_sin``
``[max_pos, 2, 32]`` are two views of one table), ``indexer_k`` 8-byte aligned and the slot
mappings ``num_tokens`` long.

Kernel name: ``glm_self_attn_indexer_q_side_k_cache_<neox|interleave>_<contig|strided>``.
"""

from __future__ import annotations

from collections.abc import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream, make_fake_tensor

from . import self_attn_indexer_k_norm_rope_fp8_cache_update as _k
from . import self_attn_indexer_q_side_fused as _q

#: The q side's rows per CTA (its served launch) and the CTA size they fix.
Q_ROWS_PER_BLOCK = 16
THREADS = Q_ROWS_PER_BLOCK // _q._ROWS_PER_WARP * _q._WARP_SIZE
#: The K update's tokens per CTA at that size (16 lanes per token).
K_ROWS_PER_BLOCK = THREADS // _k._FUSED_CAT_THREADS_PER_ROW
_NUM_HEADS = _q._NUM_HEADS
_HEAD_DIM = _q._HEAD_DIM
_PE_DIM = _q._PE_DIM
KERNEL_NAME_PREFIX = "glm_self_attn_indexer_q_side_k_cache"

_COMPILE_CACHE: dict[tuple[int, bool, bool], Callable] = {}


@cute.kernel
def indexer_q_side_k_cache_kernel(
    q: cute.Tensor,
    position_ids: cute.Tensor,
    cos_sin: cute.Tensor,
    wq: cute.Tensor,
    q_fp8: cute.Tensor,
    weights: cute.Tensor,
    q_rows: cutlass.Int32,
    weight_scale_factor: cutlass.Float32,
    indexer_k: cute.Tensor,
    norm_weight: cute.Tensor,
    norm_bias: cute.Tensor,
    rotary_cos_sin: cute.Tensor,
    k_cache: cute.Tensor,
    slot_mapping_fp8: cute.Tensor,
    slot_mapping_scale: cute.Tensor,
    num_tokens: cutlass.Int32,
    cache_dim_0: cutlass.Int32,
    cache_dim_1: cutlass.Int32,
    cache_dim_2: cutlass.Int32,
    cache_dim_3: cutlass.Int32,
    cache_stride_0: cutlass.Int64,
    cache_stride_1: cutlass.Int64,
    cache_stride_2: cutlass.Int64,
    cache_stride_3: cutlass.Int64,
    q_blocks: cutlass.Int32,
    IS_NEOX: cutlass.Constexpr[bool],
    CACHE_CONTIGUOUS: cutlass.Constexpr[bool],
) -> None:
    """CTA ``b < q_blocks``: the q side's CTA ``b``; else the K update's CTA ``b - q_blocks``."""
    block, _, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    # Nothing above touches global memory: every thread waits before its first read of the
    # projections' outputs (no CTA barrier on this path, as both kernels).
    cute.arch.griddepcontrol_wait()
    if block < q_blocks:
        _q.indexer_q_side_rows(
            block,
            tidx,
            q,
            position_ids,
            cos_sin,
            wq,
            q_fp8,
            weights,
            q_rows,
            weight_scale_factor,
            IS_NEOX,
            True,
            Q_ROWS_PER_BLOCK,
            _NUM_HEADS,
        )
    else:
        _k.indexer_k_cache_update_rows(
            block - q_blocks,
            tidx,
            indexer_k,
            norm_weight,
            norm_bias,
            position_ids,
            rotary_cos_sin,
            k_cache,
            slot_mapping_fp8,
            slot_mapping_scale,
            num_tokens,
            cache_dim_0,
            cache_dim_1,
            cache_dim_2,
            cache_dim_3,
            cache_stride_0,
            cache_stride_1,
            cache_stride_2,
            cache_stride_3,
            True,
            IS_NEOX,
            K_ROWS_PER_BLOCK,
            True,
            CACHE_CONTIGUOUS,
            False,
        )


@cute.jit
def _launch_indexer_q_side_k_cache(
    q: cute.Tensor,
    position_ids: cute.Tensor,
    cos_sin: cute.Tensor,
    wq: cute.Tensor,
    q_fp8: cute.Tensor,
    weights: cute.Tensor,
    indexer_k: cute.Tensor,
    norm_weight: cute.Tensor,
    norm_bias: cute.Tensor,
    rotary_cos_sin: cute.Tensor,
    k_cache: cute.Tensor,
    slot_mapping_fp8: cute.Tensor,
    slot_mapping_scale: cute.Tensor,
    num_tokens: cutlass.Int32,
    weight_scale_factor: cutlass.Float32,
    cache_stride_0: cutlass.Int64,
    cache_stride_1: cutlass.Int64,
    cache_stride_2: cutlass.Int64,
    cache_stride_3: cutlass.Int64,
    stream: cuda.CUstream,
    IS_NEOX: cutlass.Constexpr[bool],
    CACHE_CONTIGUOUS: cutlass.Constexpr[bool],
) -> None:
    q_rows = num_tokens * cutlass.Int32(_NUM_HEADS)
    q_blocks = cute.ceil_div(q_rows, Q_ROWS_PER_BLOCK)
    k_blocks = cute.ceil_div(num_tokens, K_ROWS_PER_BLOCK)
    indexer_q_side_k_cache_kernel(
        q,
        position_ids,
        cos_sin,
        wq,
        q_fp8,
        weights,
        q_rows,
        weight_scale_factor,
        indexer_k,
        norm_weight,
        norm_bias,
        rotary_cos_sin,
        k_cache,
        slot_mapping_fp8,
        slot_mapping_scale,
        num_tokens,
        cute.size(k_cache, mode=[0]),
        cute.size(k_cache, mode=[1]),
        cute.size(k_cache, mode=[2]),
        cute.size(k_cache, mode=[3]),
        cache_stride_0,
        cache_stride_1,
        cache_stride_2,
        cache_stride_3,
        q_blocks,
        IS_NEOX,
        CACHE_CONTIGUOUS,
    ).launch(
        grid=(q_blocks + k_blocks, 1, 1),
        block=(THREADS, 1, 1),
        stream=stream,
        use_pdl=True,
    )


def compile_indexer_q_side_k_cache(
    device_id: int, is_neox: bool, cache_contiguous: bool
) -> Callable:
    """Compile (or fetch) one specialisation (rope mode, cache addressing); allocates
    nothing. ``cache_contiguous`` addresses the cache as dense (only for a dense view)."""
    key = (int(device_id), bool(is_neox), bool(cache_contiguous))
    cached = _COMPILE_CACHE.get(key)
    if cached is not None:
        return cached
    cutlass.cuda.initialize_cuda_context(device_id)
    sym_tokens = cute.sym_int32()
    sym_positions = cute.sym_int32()
    sym_rotary_positions = cute.sym_int32()
    sym_blocks = cute.sym_int32()
    sym_block_size = cute.sym_int32()
    sym_token_bytes = cute.sym_int32()
    sym_stride_0 = cute.sym_int64(symbol="indexer_q_side_k_cache_stride_0")
    sym_stride_1 = cute.sym_int64(symbol="indexer_q_side_k_cache_stride_1")
    sym_stride_2 = cute.sym_int64(symbol="indexer_q_side_k_cache_stride_2")
    fp32, i32, i64 = cutlass.Float32, cutlass.Int32, cutlass.Int64
    fakes = (
        make_fake_compact_tensor(
            cutlass.BFloat16, (sym_tokens, _NUM_HEADS, _HEAD_DIM), stride_order=(2, 1, 0),
            assumed_align=16,
        ),  # q
        make_fake_compact_tensor(i32, (sym_tokens,), assumed_align=4),  # position_ids
        make_fake_compact_tensor(
            fp32, (sym_positions, _PE_DIM), stride_order=(1, 0), assumed_align=16
        ),  # cos_sin
        make_fake_compact_tensor(fp32, (sym_tokens, _NUM_HEADS), stride_order=(1, 0), assumed_align=4),
        make_fake_compact_tensor(
            cutlass.Float8E4M3FN, (sym_tokens, _NUM_HEADS, _HEAD_DIM), stride_order=(2, 1, 0),
            assumed_align=16,
        ),  # q_fp8
        make_fake_compact_tensor(fp32, (sym_tokens, _NUM_HEADS), stride_order=(1, 0), assumed_align=4),
        make_fake_compact_tensor(
            cutlass.BFloat16, (sym_tokens, _HEAD_DIM), stride_order=(1, 0), assumed_align=8
        ),  # indexer_k
        make_fake_compact_tensor(fp32, (_HEAD_DIM,), assumed_align=4),  # norm_weight
        make_fake_compact_tensor(fp32, (_HEAD_DIM,), assumed_align=4),  # norm_bias
        make_fake_compact_tensor(
            fp32, (sym_rotary_positions, 2, _PE_DIM // 2), stride_order=(2, 1, 0), assumed_align=4
        ),  # rotary_cos_sin
        make_fake_tensor(
            cutlass.Uint8,
            (sym_blocks, sym_block_size, 1, sym_token_bytes),
            stride=(sym_stride_0, sym_stride_1, sym_stride_2, 1),
            assumed_align=4,
        ),  # k_cache
        make_fake_compact_tensor(i64, (sym_tokens,), assumed_align=8),  # slot_mapping_fp8
        make_fake_compact_tensor(i64, (sym_tokens,), assumed_align=8),  # slot_mapping_scale
    )  # fmt: skip
    mode = "neox" if is_neox else "interleave"
    addressing = "contig" if cache_contiguous else "strided"
    indexer_q_side_k_cache_kernel.set_name_prefix(
        f"{KERNEL_NAME_PREFIX}_{mode}_{addressing}", remove_cutlass_symbol=True
    )
    compiled = cute.compile(
        _launch_indexer_q_side_k_cache,
        *fakes,
        cutlass.Int32(0),
        cutlass.Float32(1.0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(1),
        make_fake_stream(),
        bool(is_neox),
        bool(cache_contiguous),
        options="--enable-tvm-ffi",
    )
    _COMPILE_CACHE[key] = compiled
    return compiled
