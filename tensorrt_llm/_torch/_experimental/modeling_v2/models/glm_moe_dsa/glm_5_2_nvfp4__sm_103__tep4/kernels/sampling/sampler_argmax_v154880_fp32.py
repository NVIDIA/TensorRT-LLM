# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Greedy token selection over the GLM-5.2 vocabulary: row-wise argmax.

The MTP draft loop picks each draft token as ``argmax`` of the full-vocabulary
FP32 logits ``[M, 154880]`` (one row per request).  This kernel computes

    tokens[r] = argmax_c logits[r, c]        (int32)

with ``torch.argmax`` semantics: the **first** index of the maximum wins ties,
``-0.0`` and ``+0.0`` compare equal, and any NaN counts as the maximum (the
first NaN of a row wins).

Every value is mapped to an order-preserving int32 key -- the IEEE bits, with
the magnitude bits flipped for negative values, ``-0.0`` folded onto ``+0.0``
and every NaN raised to ``INT32_MAX`` -- so the whole reduction is integer
compares on ``(key, index)`` pairs: larger key wins, equal keys keep the smaller
index.  That makes the result independent of the reduction order, which the
CTA and warp trees below are free to choose.

One cluster of ``_CTAS_PER_ROW`` CTAs per row, each CTA scanning one
contiguous segment.  Each thread streams 16-byte vectors (four FP32) with
``_VECTORS_IN_FLIGHT`` loads issued before any compare and keeps its own best
pair; a butterfly warp reduction and a shared-memory pass across warps give the
CTA's pair, which it stores into cluster rank 0's shared memory (DSMEM) once
every CTA of the cluster has started; one more cluster barrier, then rank 0
reduces the segment pairs and writes the token.  ``M`` is symbolic; the logits
must be contiguous rows of ``VOCAB`` elements on 16-byte-aligned storage.

"""

from collections.abc import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.experimental import primitives as prims

VOCAB = 154880
# One cluster of _CTAS_PER_ROW CTAs per row, each scanning a contiguous
# VOCAB / _CTAS_PER_ROW segment: one CTA per row is latency-bound at ~15 us
# for any M on B300, since a single SM cannot keep enough of the
# 620 KB row in flight.
_CTAS_PER_ROW = 8
_SEGMENT = VOCAB // _CTAS_PER_ROW
_THREADS = 256
_WARP_SIZE = 32
_WARPS = _THREADS // _WARP_SIZE
_FP32_PER_VECTOR = 4
_VECTORS_IN_FLIGHT = 4
_ALIGNMENT_BYTES = 16
_INT32_MAX = 0x7FFFFFFF
_SIGN_BIT = -0x80000000  # bit pattern of -0.0 as int32
assert VOCAB % (_CTAS_PER_ROW * _FP32_PER_VECTOR) == 0

CompiledKernel = Callable[[torch.Tensor, torch.Tensor, cuda.CUstream], object]
_COMPILED: dict[bool, CompiledKernel] = {}


@cute.jit
def _order_key(value: cutlass.Float32) -> cutlass.Int32:
    """Order-preserving int32 key of an FP32 value (NaN -> INT32_MAX, -0 -> +0)."""

    bits = value.bitcast(cutlass.Int32)
    key = bits
    if bits < cutlass.Int32(0):
        key = bits ^ cutlass.Int32(_INT32_MAX)
    if bits == cutlass.Int32(_SIGN_BIT):
        key = cutlass.Int32(0)
    if value != value:
        key = cutlass.Int32(_INT32_MAX)
    return key


@cute.jit
def _better(key, idx, best_key, best_idx):
    """Whether ``(key, idx)`` beats ``(best_key, best_idx)``."""

    return (key > best_key) | ((key == best_key) & (idx < best_idx))


@cute.kernel
def _argmax_rows_kernel(
    logits: cute.Tensor,
    tokens: cute.Tensor,
    use_pdl: cutlass.Constexpr[bool],
) -> None:
    bidx, _, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    cta_rank = cute.arch.block_idx_in_cluster()
    row = bidx // _CTAS_PER_ROW
    lane = tidx % _WARP_SIZE
    warp = tidx // _WARP_SIZE

    warp_keys = cutlass.Array(
        cutlass.Int32,
        _WARPS,
        space=cutlass.AddressSpace.smem,
        alignment=_ALIGNMENT_BYTES,
    )
    warp_idxs = cutlass.Array(
        cutlass.Int32,
        _WARPS,
        space=cutlass.AddressSpace.smem,
        alignment=_ALIGNMENT_BYTES,
    )
    # Cluster rank 0's copies are the ones read; every CTA writes its own slot.
    cluster_keys = cutlass.Array(
        cutlass.Int32,
        _CTAS_PER_ROW,
        space=cutlass.AddressSpace.smem,
        alignment=_ALIGNMENT_BYTES,
    )
    cluster_idxs = cutlass.Array(
        cutlass.Int32,
        _CTAS_PER_ROW,
        space=cutlass.AddressSpace.smem,
        alignment=_ALIGNMENT_BYTES,
    )

    # Every CTA of the cluster must have started before another one stores into its
    # shared memory: arrive now, wait right before the DSMEM stores below.
    prims.barrier_cluster_arrive_relaxed()
    if cutlass.const_expr(use_pdl):
        cute.arch.griddepcontrol_wait()

    seg_begin = cutlass.Int32(cta_rank) * cutlass.Int32(_SEGMENT)
    seg_end = seg_begin + cutlass.Int32(_SEGMENT)
    row_ptr = logits.iterator.raw_ptr() + cutlass.Int64(row) * cutlass.Int64(VOCAB)
    best_key = cutlass.Int32(-_INT32_MAX - 1)
    best_idx = cutlass.Int32(_INT32_MAX)

    STEP = _THREADS * _FP32_PER_VECTOR
    CHUNK = STEP * _VECTORS_IN_FLIGHT
    num_chunks = (_SEGMENT + CHUNK - 1) // CHUNK
    for chunk in cutlass.range(num_chunks):
        base = seg_begin + cutlass.Int32(chunk * CHUNK) + tidx * _FP32_PER_VECTOR
        # Issue every load of the chunk before the first compare.
        vectors = []
        for k in cutlass.range_constexpr(_VECTORS_IN_FLIGHT):
            col = base + cutlass.Int32(k * STEP)
            values = cute.make_rmem_tensor((_FP32_PER_VECTOR,), cutlass.Float32)
            values.fill(0.0)
            if col < seg_end:
                loaded = (row_ptr + cute.assume(col, _FP32_PER_VECTOR)).load(
                    count=_FP32_PER_VECTOR, alignment=_ALIGNMENT_BYTES
                )
                for lane_k in cutlass.range_constexpr(_FP32_PER_VECTOR):
                    values[lane_k] = loaded[lane_k]
            vectors.append(values)
        # Columns visited in increasing order: strict ">" keeps the first.
        for k in cutlass.range_constexpr(_VECTORS_IN_FLIGHT):
            col = base + cutlass.Int32(k * STEP)
            if col < seg_end:
                for lane_k in cutlass.range_constexpr(_FP32_PER_VECTOR):
                    key = _order_key(vectors[k][lane_k])
                    if key > best_key:
                        best_key = key
                        best_idx = col + cutlass.Int32(lane_k)

    # Butterfly over the warp: every lane ends with the warp's best pair.
    for offset in (16, 8, 4, 2, 1):
        other_key = cute.arch.shuffle_sync_bfly(best_key, offset)
        other_idx = cute.arch.shuffle_sync_bfly(best_idx, offset)
        if _better(other_key, other_idx, best_key, best_idx):
            best_key = other_key
            best_idx = other_idx
    if lane == 0:
        warp_keys[warp] = best_key
        warp_idxs[warp] = best_idx
    cute.arch.barrier(barrier_id=0, number_of_threads=_THREADS)

    prims.barrier_cluster_wait()
    # This CTA's best pair -> its slot in cluster rank 0's shared memory.
    if tidx == 0:
        cta_key = warp_keys[0]
        cta_idx = warp_idxs[0]
        for w in cutlass.range_constexpr(1, _WARPS):
            if _better(warp_keys[w], warp_idxs[w], cta_key, cta_idx):
                cta_key = warp_keys[w]
                cta_idx = warp_idxs[w]
        for slot in cutlass.range_constexpr(_CTAS_PER_ROW):
            if cta_rank == slot:
                prims.mapa(cluster_keys.data_ptr() + slot, cutlass.Int32(0)).store(cta_key)
                prims.mapa(cluster_idxs.data_ptr() + slot, cutlass.Int32(0)).store(cta_idx)
    # arrive.release publishes the DSMEM stores, wait.acquire makes them
    # visible to rank 0; no CTA touches another's shared memory afterwards.
    prims.barrier_cluster_arrive()
    prims.barrier_cluster_wait()

    if (cta_rank == 0) & (tidx == 0):
        row_key = cluster_keys[0]
        row_idx = cluster_idxs[0]
        for slot in cutlass.range_constexpr(1, _CTAS_PER_ROW):
            if _better(cluster_keys[slot], cluster_idxs[slot], row_key, row_idx):
                row_key = cluster_keys[slot]
                row_idx = cluster_idxs[slot]
        tokens[row] = row_idx

    if cutlass.const_expr(use_pdl):
        if tidx == 0:
            cute.arch.griddepcontrol_launch_dependents()


@cute.jit
def _launch_argmax_rows(
    logits: cute.Tensor,
    tokens: cute.Tensor,
    stream: cuda.CUstream,
    use_pdl: cutlass.Constexpr[bool],
) -> None:
    rows = cute.size(logits, mode=[0])
    _argmax_rows_kernel(logits, tokens, use_pdl).launch(
        grid=(rows * _CTAS_PER_ROW, 1, 1),
        block=(_THREADS, 1, 1),
        cluster=(_CTAS_PER_ROW, 1, 1),
        stream=stream,
        use_pdl=use_pdl,
    )


def compile(use_pdl: bool = False) -> CompiledKernel:  # noqa: A001
    """Compile (once per PDL mode) the symbolic-M argmax:
    ``fn(logits[M, VOCAB] fp32, tokens[M] int32, stream)``."""

    if use_pdl not in _COMPILED:
        rows = cute.sym_int32()
        fake_logits = make_fake_compact_tensor(
            cutlass.Float32,
            (rows, VOCAB),
            stride_order=(1, 0),
            assumed_align=_ALIGNMENT_BYTES,
        )
        fake_tokens = make_fake_compact_tensor(cutlass.Int32, (rows,), assumed_align=4)
        _COMPILED[use_pdl] = cute.compile(
            _launch_argmax_rows,
            fake_logits,
            fake_tokens,
            make_fake_stream(),
            use_pdl,
            options="--enable-tvm-ffi",
        )
    return _COMPILED[use_pdl]


def _require_inputs(logits: torch.Tensor, tokens: torch.Tensor) -> None:
    """Validate the production ABI before dispatch."""

    if not logits.is_cuda or logits.device != tokens.device:
        raise ValueError("logits and tokens must be CUDA tensors on one device")
    if logits.dtype != torch.float32 or tokens.dtype != torch.int32:
        raise TypeError(
            f"expected fp32 logits and int32 tokens, got {logits.dtype}, {tokens.dtype}"
        )
    if logits.ndim != 2 or logits.shape[1] != VOCAB or logits.shape[0] < 1:
        raise ValueError(f"expected logits [M>=1, {VOCAB}], got {tuple(logits.shape)}")
    if tokens.shape != (logits.shape[0],) or not tokens.is_contiguous():
        raise ValueError(
            f"expected contiguous tokens [{logits.shape[0]}], got {tuple(tokens.shape)}"
        )
    if not logits.is_contiguous() or logits.data_ptr() % _ALIGNMENT_BYTES != 0:
        raise ValueError("logits must be contiguous and 16-byte aligned")


def run(compiled_fn: CompiledKernel, logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Launch a precompiled variant into caller-owned ``tokens``."""

    _require_inputs(logits, tokens)
    compiled_fn(logits, tokens, torch.cuda.current_stream(logits.device).cuda_stream)
    return tokens
