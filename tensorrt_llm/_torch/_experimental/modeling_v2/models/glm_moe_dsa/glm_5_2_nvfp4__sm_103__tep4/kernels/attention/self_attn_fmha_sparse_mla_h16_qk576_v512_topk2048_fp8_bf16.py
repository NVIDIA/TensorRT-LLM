# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sparse MLA FMHA for GLM generation, built on CuTe DSL primitives: the shared ABI and parts.

This module held the original decode kernel, which gathered every Kv128 tile twice (K for
QK, then the latent again as V for PV). The kernel the target runs is
``self_attn_fmha_sparse_mla_wide_h16_qk576_v512_topk2048_fp8_bf16`` (H16, PV reading V
from the tile's own K slots, bitwise the same outputs); the original kernel, its launcher,
compile cache and runner were removed with it, as nothing served called them. What remains
is what the wide kernel builds on: the ABI below, the tile constants, the device helpers
(gather, softmax fragments, masking, the split-K reducers), and the host side (split-K and
reducer selection, workspaces, the runner and specialization records, validation). The
description below is the original kernel's; the wide module's docstring says what it
changes (the SMEM ring and the producer schedule). ``prepare_runner``,
``precompile_for_serving`` and ``compile`` are the wide module's, and it serves H16 only.

Direct post-preprocessing ABI
-----------------------------
The callable consumes packed FP8 Q, sparse-row indices, the selected layer's
flattened FP8 KV-pool view, flattened-head ``cu_q_seqlens``, runtime sequence
lengths, and caller-owned BF16 output. The indices are interpreted per the
host ``index_kind``: ``INDEX_KIND_GLOBAL`` (default) takes global physical
pool rows from TRT-LLM's ``convert_req_index_to_global`` step as-is;
``INDEX_KIND_BASE_WITH_LAYER_OFFSET`` with ``local_layer_idx`` takes
the layer-independent base map of ``convert_req_index_to_base_rawptr`` and
adds ``local_layer_idx * 32`` on the device while staging the indices, so one
base map serves three consecutive shared layers. Both kinds compute the same
attention; GLOBAL is the unchanged kernel. This is a source-level semantic adapter ABI; it
is not TensorRT-LLM's binary ``KernelParams`` layout. Do not register the compiled
artifact as a TRT cubin or pass an unchanged ``KernelParams*`` to it.

Within every sparse row, positions before the sequence-derived valid-prefix
length must contain in-range global pool rows. The remaining positions are the
contiguous ``-1`` tail. An interior ``-1`` followed by a later valid row is
outside this ABI: the kernel masks the positional tail, not arbitrary holes.

H in {8,16,32}, QK576=(latent512,rope64), V512, top-k 2048, validity mode,
``q_scaling``, split-K CTA count, and reducer kind are compile-time choices.
Batch size, packed query count, maximum Sq in [1,8], per-request Sq in
[0,maximum Sq], pool size, sequence lengths, global sparse rows, tensor
addresses, and stream remain runtime values. The two TRT scale tensors are
accepted and shape-checked as outputs of the fixed GLM RoPE preprocessing,
but this generation device kernel uses the corresponding model-fixed
``q_scaling`` specialization rather than loading those arrays.

Recommended TensorRT-LLM integration
------------------------------------
Invoke this core from the existing generation hook after
``mla_rope_generation`` has filled FP8 Q and after
``transform_local_topk_and_prepare_pool_view(..., is_generation=True)`` has
produced the global indices and layer pool view. Route only supported GLM
generation requests; all other phases and shapes must fall back. H8 and H16
are exercised by the current serving adapter. H32 is independently validated
by this module but needs an explicit production dispatch extension before use.

The tensor-view adaptation needs no extra device kernel beyond TRT-LLM's
existing local-to-global conversion. It is not literally free on the host:
Each ``prepare_runner`` call repeats Python validation and may allocate split-K workspaces.
For CUDA Graph replay or a stable eager path, call ``prepare_runner`` once,
retain its runner and workspaces while tensor addresses remain stable, and
invoke the retained runner on each replay. Re-prepare only when an address,
specialization, or launch shape changes. This avoids host work in the replay
hot path without pretending binary ``KernelParams`` compatibility.

Before accepting dynamic mixed-batch traffic, call ``precompile_for_serving``
once per device with the scheduler's maximum batch size. The split-K heuristic
derives a constexpr CTA count from the runtime batch size; precompiling its
unique reachable variants keeps a new mixed-batch bucket from running
``cute.compile`` on the request/TTFT critical path. This mirrors TRT-LLM's
ahead-of-traffic kernel instantiation rather than changing the device ABI.

TRT-LLM's generation heuristic selects one to eight CTA1 compute pipelines per
query from runtime batch, max Sq, max KV length, SM count, and cluster
occupancy. Each CTA owns a contiguous block of Kv128 tiles. W9 loads Q and
stages sparse rows; W12-W15 issue direct TMA gather4 transactions into a shared
K5/V4 nine-slot ring. W8 computes QK, W0-W3 perform online softmax and write
FP8 P, W11 issues SwapsAb VP, and W4-W7 apply online-softmax correction. A
single-wave split uses a CGA merge; larger launches use BF16/FP32 GMEM partials
and TRT-LLM's atomic completion-counter reduction protocol.

Two constexpr validity specializations share the source. StaticTokenSparse
indices follow TRT-LLM's positional contract: each query has a contiguous valid
prefix followed by ``-1`` rows. The device derives that prefix length from the
runtime sequence length and query position, then masks the tail QK scores before
the online max and exponential. For production H16, split CTAs partition only
the runtime-valid Kv128 tiles when that removes at least one complete tile and
every CTA can retain a nonempty pipeline. This avoids computing the fixed
2048-slot invalid tail without changing the split-reducer ABI. The all-valid
specialization removes those predicates when all 2048 positions are known
valid. Probabilities are scaled by 448 before FP8 E4M3 conversion, matching
TRT-LLM's MMA operand protocol; the same constant factor is carried in the
online-softmax sum and therefore cancels during normalization and split-K
merge. Q scaling is constexpr because the GLM model config is fixed; the public
scale tensors are shape-checked as the coupled output of the preceding RoPE
specialization and are not reloaded by this device kernel.

For cluster (DSMEM) reducers, the TMEM dealloc and its full-CTA sync come
after the reducer. Single-CTA and GMEM reducers dealloc before the reducer.
All head counts store the P tiles through ``.apply_swizzle(s128b)`` on the
absolute SMEM address.

PDL is always enabled. Only the predecessor-reading W9 producer waits, after
independent barrier/TMEM setup has started. Every output-store thread reaches a
CTA publication barrier before the CTA releases dependent work; cluster
teardown remains after that release. Hardware operations use
``cutlass.experimental.primitives`` directly. The kernel does not use CuTe
copy/GEMM helpers, tiled-MMA helpers, or the SM100 utility module.

The production wrapper accepts H8/H16/H32 and runtime maximum Sq 1..8, then
matches TRT-LLM's split-K selection across one to eight CTAs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import torch
from cutlass.experimental import primitives as prims

from ..common.cuda_stream import raw_stream as _raw_stream

QK_HEAD_DIM = 256
_MAX_ACTIVE_CLUSTERS_CACHE: dict[tuple[int, int], int] = {}
# Split-KV completion counters, one zeroed int32 buffer per device, reused across
# eager launches. The kernel's `atomicrmw INC` with limit `num_ctas_kv - 1` wraps
# each row's counter back to 0 when its last split arrives, so a buffer that
# starts at zero is zero again after every completed launch; allocating (and
# filling) a fresh one per call cost one PyTorch fill launch per layer on every
# eager mixed step (13 per context step in a served profile). Grown on demand.
_COMPLETION_COUNTER_CACHE: dict[int, torch.Tensor] = {}


def _completion_counters(total_q: int, device: torch.device) -> torch.Tensor:
    """Zeroed split-KV completion counters for ``total_q`` query rows.

    Under CUDA-graph capture this returns a fresh allocation, as before: a
    capture-pool tensor must not outlive its graph in a module cache.
    """
    if torch.cuda.is_current_stream_capturing():
        return torch.zeros(total_q, dtype=torch.int32, device=device)
    index = device.index if device.index is not None else torch.cuda.current_device()
    cached = _COMPLETION_COUNTER_CACHE.get(index)
    if cached is None or cached.numel() < total_q:
        cached = torch.zeros(max(total_q, 1024), dtype=torch.int32, device=device)
        _COMPLETION_COUNTER_CACHE[index] = cached
    return cached[:total_q]


IO_DTYPE = cutlass.Float8E4M3FN
OUT_DTYPE = cutlass.BFloat16
ACC_DTYPE = cutlass.Float32

NUM_HEADS: int = 16
SUPPORTED_NUM_HEADS: tuple[int, ...] = (8, 16, 32)
L: int = 512
R: int = 64
DEFAULT_MAX_SQ: int = 2
SUPPORTED_MAX_SQ: tuple[int, ...] = tuple(range(1, 9))
TOPK: int = 2048
KV_STRIDE: int = L + R
LOG2_E: float = 1.4426950408889634
FP8_P_SCALE: float = 448.0
LOG2_FP8_P_SCALE: float = math.log2(FP8_P_SCALE)
# A 16x256b TMEM load groups the H16 softmax statistics by fragment column:
# [0, 1, 8, 9, 2, 3, 10, 11, ...].  The PV accumulator and reducer ABIs are
# instead in logical head order.  Keep the online-softmax state packed, then
# apply this inverse permutation exactly once at the correction handoff.
H16_LOGICAL_TO_PACKED_STATS: tuple[int, ...] = (
    0,
    1,
    4,
    5,
    8,
    9,
    12,
    13,
    2,
    3,
    6,
    7,
    10,
    11,
    14,
    15,
)

# One tcgen05 M=128 tile computes all heads for one query token. H is an
# H8/H16/H32 constexpr specialization; remaining rows are pipeline padding.
M_TILE: int = 128

QK_N: int = 128
QK_K: int = 128
PV_M: int = 128
N_V: int = L // PV_M

CTA_N: int = QK_N
N_LAT: int = L // QK_K
GRAN: int = 32
EB: int = 1

LBO_K: int = 16
SBO_LAT: int = 8 * QK_K * EB
SBO_ROPE: int = 8 * R * EB
SBO_P: int = 8 * QK_N * EB
V_GROUP: int = 128
LBO_V: int = (QK_N // 2) * V_GROUP * EB
SBO_V: int = 8 * V_GROUP * EB
V_KSTEP: int = GRAN * V_GROUP * EB

# H8/H16 use 32 score columns followed by 64 output columns. H32 doubles both
# regions. The concrete values are selected as constexprs inside the kernel.
TMEM_O_OFF: int = 32
NUM_TMEM_COLS: int = 128

# TRTLLM stages K576 as five K128 slices and V512 as four V128 slices. A
# shared nine-slot ring lets next-tile K reuse slots as soon as QK releases
# them, independently of the longer-lived current-tile V/PV slots.
K_HDIM_STAGES: int = 5
V_HDIM_STAGES: int = 4
ST_I: int = 6
I_TILES_PER_STAGE: int = 2
I_STAGE_ELTS: int = I_TILES_PER_STAGE * QK_N
I_TOTAL_STAGES: int = TOPK // I_STAGE_ELTS
NUM_KV_TILES: int = TOPK // QK_N
MAX_CTAS_KV: int = I_TOTAL_STAGES
ST_S: int = 2
ST_P: int = 2
ST_C: int = 2

W_CORR0: int = 4
W_MMA_QK: int = 8
W_LOAD_KQ: int = 9
W_MMA_PV: int = 11
W_LOAD_GATHER0: int = 12
NUM_GATHER_WARPS: int = 4
THREADS: int = 16 * 32

BAR_SM0: int = 1  # g0 intra-group (128 thr)

KV_STAGE_BYTES: int = QK_N * QK_K * EB

# Interpretation of ``sparse_attn_indices`` (host contract ``index_kind``).
# GLOBAL: rows are global KV-pool rows (TRT-LLM's convert_req_index_to_global
# output); the device uses them as-is. BASE_WITH_LAYER_OFFSET: rows are
# the layer-independent base map of ``convert_req_index_to_base_rawptr``
# and the index-staging warp adds ``local_layer_idx * KV_POOL_TOKENS_PER_BLOCK``
# to every nonnegative row (``-1`` tails stay ``-1``), so one base map serves
# three consecutive shared layers. GLOBAL compiles the unchanged kernel.
INDEX_KIND_GLOBAL = "global"
INDEX_KIND_BASE_WITH_LAYER_OFFSET = "base_with_layer_offset"
INDEX_KINDS: tuple[str, ...] = (
    INDEX_KIND_GLOBAL,
    INDEX_KIND_BASE_WITH_LAYER_OFFSET,
)
# GLM paged KV cache: 32 tokens per block; inside a physical block row the
# layers are laid out one block of tokens apart (the ``layer_id * 32`` term of
# ``convert_req_index_to_base_rawptr``).
KV_POOL_TOKENS_PER_BLOCK: int = 32


@cute.jit
def _add_layer_offset_vector4(values, layer_offset: cutlass.Int32):
    """Apply the local-layer offset only to nonnegative base rows."""
    return cutlass.Vector.from_elements(
        tuple(
            values[index] + layer_offset if values[index] >= cutlass.Int32(0) else cutlass.Int32(-1)
            for index in range(4)
        ),
        dtype=cutlass.Int32,
    )


@cute.jit
def _fmax(lhs: cutlass.Float32, rhs: cutlass.Float32) -> cutlass.Float32:
    return cutlass.Float32(prims.dialect.fmax(lhs, rhs))


@cute.jit
def _tma_gather4_cta1(
    smem_dst,
    tma_desc,
    mbarrier,
    col,
    row0,
    row1,
    row2,
    row3,
) -> None:
    """Issue one public-primitive CTA1 TMA gather4 transaction per warp."""
    local_barrier = prims.cvta_to(mbarrier, prims.CvtaSpace.SHARED)
    prims.inline_ptx_hl(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "elect.sync _|leader, -1;\n\t"
        "@leader cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
        ".mbarrier::complete_tx::bytes.L2::cache_hint"
        " [{$r0}], [{$r1}, {{$r2}, {$r3}, {$r4}, {$r5}, {$r6}}],"
        " [{$r7}], {$r8};\n\t"
        "}",
        read_only_args=[
            smem_dst,
            tma_desc,
            col,
            row0,
            row1,
            row2,
            row3,
            local_barrier,
            cutlass.Int64(0),
        ],
    )


@cute.jit
def _fragment_max4(values):
    return cutlass.Vector.from_elements(
        (
            cute.math.max(
                cute.math.max(values[0], values[2]),
                cute.math.max(values[8], values[10]),
            ),
            cute.math.max(
                cute.math.max(values[1], values[3]),
                cute.math.max(values[9], values[11]),
            ),
            cute.math.max(
                cute.math.max(values[4], values[6]),
                cute.math.max(values[12], values[14]),
            ),
            cute.math.max(
                cute.math.max(values[5], values[7]),
                cute.math.max(values[13], values[15]),
            ),
        ),
        dtype=ACC_DTYPE,
    )


@cute.jit
def _fragment_sum4(values):
    sum01 = prims.add_packed_f32x2(
        (values[0], values[1]),
        (values[2], values[3]),
        rnd=prims.FPRoundingMode.RN,
    )
    sum01 = prims.add_packed_f32x2(
        sum01,
        (values[8], values[9]),
        rnd=prims.FPRoundingMode.RN,
    )
    sum01 = prims.add_packed_f32x2(
        sum01,
        (values[10], values[11]),
        rnd=prims.FPRoundingMode.RN,
    )
    sum23 = prims.add_packed_f32x2(
        (values[4], values[5]),
        (values[6], values[7]),
        rnd=prims.FPRoundingMode.RN,
    )
    sum23 = prims.add_packed_f32x2(
        sum23,
        (values[12], values[13]),
        rnd=prims.FPRoundingMode.RN,
    )
    sum23 = prims.add_packed_f32x2(
        sum23,
        (values[14], values[15]),
        rnd=prims.FPRoundingMode.RN,
    )
    return cutlass.Vector.from_elements(
        (sum01[0], sum01[1], sum23[0], sum23[1]),
        dtype=ACC_DTYPE,
    )


@cute.jit
def _scale_scores_packed(scores, new_max, scale_log2, num_heads: cutlass.Constexpr[int]):
    """Apply the online-softmax affine transform with packed FP32 FMA."""
    scaled_values = cutlass.Array(ACC_DTYPE, num_heads, space=cutlass.AddressSpace.rmem)
    scale_pair = (scale_log2, scale_log2)
    neg_scale_pair = (-scale_log2, -scale_log2)
    for i in cutlass.range_constexpr(0, num_heads, 2):
        if cutlass.const_expr(num_heads == 32):
            max_idx0: cutlass.Constexpr[int] = ((i % 16) // 4) * 2
            max_idx1: cutlass.Constexpr[int] = max_idx0 + 1
        else:
            max_idx0: cutlass.Constexpr[int] = (i // 16) * 4 + (((i % 8) // 4) * 2 + (i % 2))
            max_idx1: cutlass.Constexpr[int] = ((i + 1) // 16) * 4 + (
                (((i + 1) % 8) // 4) * 2 + ((i + 1) % 2)
            )
        bias = prims.fma_packed_f32x2(
            (new_max[max_idx0], new_max[max_idx1]),
            neg_scale_pair,
            (LOG2_FP8_P_SCALE, LOG2_FP8_P_SCALE),
            rnd=prims.FPRoundingMode.RN,
        )
        scaled_pair = prims.fma_packed_f32x2(
            (scores[i], scores[i + 1]),
            scale_pair,
            bias,
            rnd=prims.FPRoundingMode.RN,
        )
        scaled_values[i] = scaled_pair[0]
        scaled_values[i + 1] = scaled_pair[1]
    return cutlass.Vector.from_elements(
        tuple(scaled_values[i] for i in range(num_heads)), dtype=ACC_DTYPE
    )


@cute.jit
def _mask_scores_after_valid_prefix_h16(
    scores,
    tile_start,
    valid_kv,
    warp_idx,
    row_idx,
    num_heads: cutlass.Constexpr[int],
):
    """Mask H16 in its native score-fragment order."""
    masked = cutlass.Array(ACC_DTYPE, num_heads, space=cutlass.AddressSpace.rmem)
    row_base = tile_start + warp_idx * cutlass.Int32(32) + row_idx
    for i in cutlass.range_constexpr(num_heads):
        fragment_index: cutlass.Constexpr[int] = i % 16
        token_offset: cutlass.Constexpr[int] = (
            ((fragment_index % 4) // 2) + 2 * (fragment_index // 8)
        ) * 8
        value = scores[i]
        if row_base + cutlass.Int32(token_offset) >= valid_kv:
            value = ACC_DTYPE(-3.0e38)
        masked[i] = value
    return cutlass.Vector.from_elements(tuple(masked[i] for i in range(num_heads)), dtype=ACC_DTYPE)


@cute.jit
def _warpgroup_row_max(value: cutlass.Float32) -> cutlass.Float32:
    for offset in (4, 8):
        value = _fmax(
            value,
            prims.shfl_sync(
                thread_mask=0xFFFFFFFF,
                val=value,
                offset=offset,
                mask_and_clamp=0x1F,
                kind=prims.Shfl.BFLY,
            ),
        )
    return value


@cute.jit
def _warpgroup_row_sum(value: cutlass.Float32) -> cutlass.Float32:
    for offset in (16, 8, 4):
        value += prims.shfl_sync(
            thread_mask=0xFFFFFFFF,
            val=value,
            offset=offset,
            mask_and_clamp=0x1F,
            kind=prims.Shfl.BFLY,
        )
    return value


@cute.jit
def _reduce_gmem_output(
    partial_o,
    partial_stats,
    partial_o_workspace: cute.Tensor,
    partial_stats_workspace: cute.Tensor,
    completion_counters: cute.Tensor,
    split_scratch,
    o: cute.Tensor,
    tidx,
    packed_q,
    cta,
    scale_log2,
    num_heads: cutlass.Constexpr[int],
    num_ctas_kv: cutlass.Constexpr[int],
) -> None:
    partial_o_ptr = partial_o_workspace.iterator.raw_ptr()
    partial_stats_ptr = partial_stats_workspace.iterator.raw_ptr()
    workspace_cta = packed_q * cutlass.Int32(num_ctas_kv) + cta
    workspace_o_base = workspace_cta * cutlass.Int32(num_heads * L)
    workspace_stats_base = workspace_cta * cutlass.Int32(num_heads * 2)

    if cutlass.const_expr(num_heads == 32):
        if tidx < cutlass.Int32(16 * 32):
            head_in_block = tidx // cutlass.Int32(32)
            vec = tidx % cutlass.Int32(32)
            for head_block in cutlass.range_constexpr(2):
                head = head_in_block + cutlass.Int32(head_block * 16)
                for half in cutlass.range_constexpr(2):
                    col = cutlass.Int32(half * (L // 2)) + vec * cutlass.Int32(8)
                    local_v = (partial_o.data_ptr() + head * cutlass.Int32(L) + col).load(
                        count=8, alignment=16
                    )
                    (partial_o_ptr + workspace_o_base + head * cutlass.Int32(L) + col).store(
                        local_v, alignment=16
                    )
        if tidx < cutlass.Int32(num_heads):
            stat_offset = tidx * cutlass.Int32(2)
            stats = cutlass.Vector.from_elements(
                (partial_stats[tidx], partial_stats[num_heads + tidx]),
                dtype=ACC_DTYPE,
            )
            (partial_stats_ptr + workspace_stats_base + stat_offset).store(stats, alignment=8)
    else:
        if tidx < cutlass.Int32(num_heads * 32):
            head = tidx // cutlass.Int32(32)
            vec = tidx % cutlass.Int32(32)
            for half in cutlass.range_constexpr(2):
                col = cutlass.Int32(half * (L // 2)) + vec * cutlass.Int32(8)
                local_v = (partial_o.data_ptr() + head * cutlass.Int32(L) + col).load(
                    count=8, alignment=16
                )
                (partial_o_ptr + workspace_o_base + head * cutlass.Int32(L) + col).store(
                    local_v, alignment=16
                )
        if tidx < cutlass.Int32(num_heads):
            stat_offset = tidx * cutlass.Int32(2)
            stats = (partial_stats.data_ptr() + stat_offset).load(count=2, alignment=8)
            (partial_stats_ptr + workspace_stats_base + stat_offset).store(stats, alignment=8)

    # Every producer thread publishes its GMEM partial before thread 0
    # releases this CTA through the completion counter.
    prims.memory_barrier(prims.MemScope.GPU)
    prims.barrier_cta_sync(0)
    counter_ptr = completion_counters.iterator.raw_ptr() + packed_q
    if tidx == 0:
        old_count = prims.atomicrmw(
            prims.AtomicOp.INC,
            counter_ptr,
            cutlass.Int32(num_ctas_kv - 1),
            mem_order=prims.MemOrder.ACQ_REL,
            syncscope=prims.MemScope.GPU,
        )
        split_scratch[0] = cutlass.Int32(num_ctas_kv - 1) - old_count
        if old_count != cutlass.Int32(num_ctas_kv - 1):
            counter_value = prims.load_ext(
                counter_ptr,
                dtype=cutlass.Int32,
                order=prims.MemOrder.ACQUIRE,
                scope=prims.MemScope.GPU,
            )
            while counter_value != cutlass.Int32(0):
                counter_value = prims.load_ext(
                    counter_ptr,
                    dtype=cutlass.Int32,
                    order=prims.MemOrder.ACQUIRE,
                    scope=prims.MemScope.GPU,
                )
    prims.barrier_cta_sync(0)

    reducer_rank = split_scratch[0]
    total_vectors: cutlass.Constexpr[int] = num_heads * L // 8
    first_vector = (reducer_rank * cutlass.Int32(total_vectors)) // cutlass.Int32(num_ctas_kv)
    end_vector = (
        (reducer_rank + cutlass.Int32(1)) * cutlass.Int32(total_vectors)
    ) // cutlass.Int32(num_ctas_kv)
    vectors_per_thread: cutlass.Constexpr[int] = (total_vectors + num_ctas_kv * THREADS - 1) // (
        num_ctas_kv * THREADS
    )
    for vector_iter in cutlass.range_constexpr(vectors_per_thread):
        flat_vector = first_vector + tidx + cutlass.Int32(vector_iter * THREADS)
        if flat_vector < end_vector:
            head = flat_vector // cutlass.Int32(L // 8)
            vec = flat_vector % cutlass.Int32(L // 8)
            col = vec * cutlass.Int32(8)
            global_max = ACC_DTYPE(-3.0e38)
            for peer_rank in cutlass.range_constexpr(num_ctas_kv):
                peer_cta = packed_q * cutlass.Int32(num_ctas_kv) + cutlass.Int32(peer_rank)
                peer_stat = peer_cta * cutlass.Int32(num_heads * 2) + head * cutlass.Int32(2)
                global_max = _fmax(global_max, partial_stats_ptr[peer_stat])

            global_mass = ACC_DTYPE(0.0)
            merged_v = cutlass.Vector.from_elements(
                tuple(ACC_DTYPE(0.0) for _ in range(8)), dtype=ACC_DTYPE
            )
            nonzero_peers = cutlass.Int32(0)
            sole_peer_o = cutlass.Int32(0)
            for peer_rank in cutlass.range_constexpr(num_ctas_kv):
                peer_cta = packed_q * cutlass.Int32(num_ctas_kv) + cutlass.Int32(peer_rank)
                peer_stat = peer_cta * cutlass.Int32(num_heads * 2) + head * cutlass.Int32(2)
                peer_max = partial_stats_ptr[peer_stat]
                peer_sum = partial_stats_ptr[peer_stat + cutlass.Int32(1)]
                peer_weight = cute.math.exp2((peer_max - global_max) * scale_log2, fastmath=True)
                peer_mass = peer_sum * peer_weight
                global_mass += peer_mass
                peer_o = peer_cta * cutlass.Int32(num_heads * L) + head * cutlass.Int32(L) + col
                if cutlass.const_expr(num_heads == 16):
                    if peer_sum != ACC_DTYPE(0.0):
                        nonzero_peers += cutlass.Int32(1)
                        sole_peer_o = peer_o
                peer_v = (partial_o_ptr + peer_o).load(count=8, alignment=16).to(ACC_DTYPE)
                merged_v += peer_v * peer_mass

            out_row = (packed_q * cutlass.Int32(num_heads) + head) * cutlass.Int32(L)
            safe_mass = _fmax(global_mass, ACC_DTYPE(1.0e-30))
            merged = (merged_v / safe_mass).to(OUT_DTYPE)
            if cutlass.const_expr(num_heads == 16):
                # H16 empty split participants (sum == 0): a single surviving
                # partial is copied unchanged (bitwise split-N == split-1).
                if nonzero_peers == cutlass.Int32(1):
                    merged = (partial_o_ptr + sole_peer_o).load(count=8, alignment=16)
            (o.iterator.raw_ptr() + out_row + col).store(merged, alignment=16)


@cute.jit
def _store_single_cta_output(
    partial_o,
    o: cute.Tensor,
    tidx,
    packed_q,
    num_heads: cutlass.Constexpr[int],
) -> None:
    if cutlass.const_expr(num_heads == 32):
        if tidx < cutlass.Int32(16 * 32):
            head_in_block = tidx // cutlass.Int32(32)
            vec = tidx % cutlass.Int32(32)
            o_ptr = o.iterator.raw_ptr()
            for head_block in cutlass.range_constexpr(2):
                head = head_in_block + cutlass.Int32(head_block * 16)
                out_row = (packed_q * cutlass.Int32(num_heads) + head) * cutlass.Int32(L)
                for half in cutlass.range_constexpr(2):
                    col = cutlass.Int32(half * (L // 2)) + vec * cutlass.Int32(8)
                    local_v = (partial_o.data_ptr() + head * cutlass.Int32(L) + col).load(
                        count=8, alignment=16
                    )
                    (o_ptr + out_row + col).store(local_v, alignment=16)
    else:
        if tidx < cutlass.Int32(num_heads * 32):
            head = tidx // cutlass.Int32(32)
            vec = tidx % cutlass.Int32(32)
            o_ptr = o.iterator.raw_ptr()
            out_row = (packed_q * cutlass.Int32(num_heads) + head) * cutlass.Int32(L)
            for half in cutlass.range_constexpr(2):
                col = cutlass.Int32(half * (L // 2)) + vec * cutlass.Int32(8)
                local_v = (partial_o.data_ptr() + head * cutlass.Int32(L) + col).load(
                    count=8, alignment=16
                )
                (o_ptr + out_row + col).store(local_v, alignment=16)


@cute.jit
def _reduce_two_cta_output(
    partial_o,
    partial_stats,
    o: cute.Tensor,
    tidx,
    packed_q,
    cta,
    scale_log2,
    num_heads: cutlass.Constexpr[int],
) -> None:
    # Release-publish both CTA partials, then each CTA pulls the peer's
    # SMEM and writes one disjoint 256-column half of the final output.
    prims.barrier_cluster_arrive()
    prims.barrier_cluster_wait()
    if cutlass.const_expr(num_heads == 32):
        if tidx < cutlass.Int32(16 * 32):
            head_in_block = tidx // cutlass.Int32(32)
            vec = tidx % cutlass.Int32(32)
            col = cta * cutlass.Int32(L // 2) + vec * cutlass.Int32(8)
            peer = cta ^ cutlass.Int32(1)
            peer_o = prims.mapa(partial_o.data_ptr(), peer)
            peer_stats = prims.mapa(partial_stats.data_ptr(), peer)
            o_ptr = o.iterator.raw_ptr()
            for head_block in cutlass.range_constexpr(2):
                head = head_in_block + cutlass.Int32(head_block * 16)
                local_max = partial_stats[head]
                local_sum = partial_stats[cutlass.Int32(num_heads) + head]
                peer_max = peer_stats[head]
                peer_sum = peer_stats[cutlass.Int32(num_heads) + head]
                global_max = _fmax(local_max, peer_max)
                local_weight = cute.math.exp2((local_max - global_max) * scale_log2, fastmath=True)
                peer_weight = cute.math.exp2((peer_max - global_max) * scale_log2, fastmath=True)
                local_mass = local_sum * local_weight
                peer_mass = peer_sum * peer_weight
                safe_mass = _fmax(local_mass + peer_mass, ACC_DTYPE(1.0e-30))
                inv_mass = ACC_DTYPE(1.0) / safe_mass
                out_row = (packed_q * cutlass.Int32(num_heads) + head) * cutlass.Int32(L)
                local_v = (
                    (partial_o.data_ptr() + head * cutlass.Int32(L) + col)
                    .load(count=8, alignment=16)
                    .to(ACC_DTYPE)
                )
                peer_v = (
                    (peer_o + head * cutlass.Int32(L) + col)
                    .load(count=8, alignment=16)
                    .to(ACC_DTYPE)
                )
                merged = ((local_v * local_mass + peer_v * peer_mass) * inv_mass).to(OUT_DTYPE)
                (o_ptr + out_row + col).store(merged, alignment=16)
    else:
        if tidx < cutlass.Int32(num_heads * 32):
            head = tidx // cutlass.Int32(32)
            vec = tidx % cutlass.Int32(32)
            col = cta * cutlass.Int32(L // 2) + vec * cutlass.Int32(8)
            peer = cta ^ cutlass.Int32(1)
            peer_o = prims.mapa(partial_o.data_ptr(), peer)
            peer_stats = prims.mapa(partial_stats.data_ptr(), peer)

            local_max = partial_stats[head * cutlass.Int32(2)]
            local_sum = partial_stats[head * cutlass.Int32(2) + cutlass.Int32(1)]
            peer_max = peer_stats[head * cutlass.Int32(2)]
            peer_sum = peer_stats[head * cutlass.Int32(2) + cutlass.Int32(1)]
            global_max = _fmax(local_max, peer_max)
            local_weight = cute.math.exp2((local_max - global_max) * scale_log2, fastmath=True)
            peer_weight = cute.math.exp2((peer_max - global_max) * scale_log2, fastmath=True)
            local_mass = local_sum * local_weight
            peer_mass = peer_sum * peer_weight
            safe_mass = _fmax(local_mass + peer_mass, ACC_DTYPE(1.0e-30))
            inv_mass = ACC_DTYPE(1.0) / safe_mass
            o_ptr = o.iterator.raw_ptr()
            out_row = (packed_q * cutlass.Int32(num_heads) + head) * cutlass.Int32(L)
            local_raw = (partial_o.data_ptr() + head * cutlass.Int32(L) + col).load(
                count=8, alignment=16
            )
            peer_raw = (peer_o + head * cutlass.Int32(L) + col).load(count=8, alignment=16)
            merged = (
                (local_raw.to(ACC_DTYPE) * local_mass + peer_raw.to(ACC_DTYPE) * peer_mass)
                * inv_mass
            ).to(OUT_DTYPE)
            if cutlass.const_expr(num_heads == 16):
                # H16 empty split participant (sum == 0): the surviving partial
                # is already normalized, so copy it unchanged; v * mass *
                # (1 / mass) would not be bitwise v. Keeps split-N bitwise equal
                # to split-1 when only one CTA holds valid tiles.
                if peer_sum == ACC_DTYPE(0.0):
                    merged = local_raw
                elif local_sum == ACC_DTYPE(0.0):
                    merged = peer_raw
            (o_ptr + out_row + col).store(merged, alignment=16)


@cute.jit
def _reduce_cga_output(
    partial_o,
    partial_stats,
    o: cute.Tensor,
    tidx,
    packed_q,
    cta,
    scale_log2,
    num_heads: cutlass.Constexpr[int],
    num_ctas_kv: cutlass.Constexpr[int],
) -> None:
    # TRT-LLM CGA reduction partitions contiguous flattened output slices
    # across the CTA ranks, including non-power-of-two cluster sizes.
    prims.barrier_cluster_arrive()
    prims.barrier_cluster_wait()
    total_vectors: cutlass.Constexpr[int] = num_heads * L // 8
    first_vector = (cta * cutlass.Int32(total_vectors)) // cutlass.Int32(num_ctas_kv)
    end_vector = ((cta + cutlass.Int32(1)) * cutlass.Int32(total_vectors)) // cutlass.Int32(
        num_ctas_kv
    )
    vectors_per_thread: cutlass.Constexpr[int] = (total_vectors + num_ctas_kv * THREADS - 1) // (
        num_ctas_kv * THREADS
    )
    for vector_iter in cutlass.range_constexpr(vectors_per_thread):
        flat_vector = first_vector + tidx + cutlass.Int32(vector_iter * THREADS)
        if flat_vector < end_vector:
            head = flat_vector // cutlass.Int32(L // 8)
            vec = flat_vector % cutlass.Int32(L // 8)
            col = vec * cutlass.Int32(8)
            stat_offset = head * cutlass.Int32(2)
            global_max = ACC_DTYPE(-3.0e38)
            for peer_rank in cutlass.range_constexpr(num_ctas_kv):
                peer_stats = prims.mapa(partial_stats.data_ptr(), cutlass.Int32(peer_rank))
                if cutlass.const_expr(num_heads == 32):
                    global_max = _fmax(global_max, peer_stats[head])
                else:
                    global_max = _fmax(global_max, peer_stats[stat_offset])

            global_mass = ACC_DTYPE(0.0)
            merged_v = cutlass.Vector.from_elements(
                tuple(ACC_DTYPE(0.0) for _ in range(8)), dtype=ACC_DTYPE
            )
            nonzero_peers = cutlass.Int32(0)
            sole_peer = cutlass.Int32(0)
            for peer_rank in cutlass.range_constexpr(num_ctas_kv):
                peer_stats = prims.mapa(partial_stats.data_ptr(), cutlass.Int32(peer_rank))
                if cutlass.const_expr(num_heads == 32):
                    peer_max = peer_stats[head]
                    peer_sum = peer_stats[cutlass.Int32(num_heads) + head]
                else:
                    peer_max = peer_stats[stat_offset]
                    peer_sum = peer_stats[stat_offset + cutlass.Int32(1)]
                if cutlass.const_expr(num_heads == 16):
                    if peer_sum != ACC_DTYPE(0.0):
                        nonzero_peers += cutlass.Int32(1)
                        sole_peer = cutlass.Int32(peer_rank)
                peer_weight = cute.math.exp2((peer_max - global_max) * scale_log2, fastmath=True)
                global_mass += peer_sum * peer_weight
                peer_o = prims.mapa(partial_o.data_ptr(), cutlass.Int32(peer_rank))
                peer_v = (
                    (peer_o + head * cutlass.Int32(L) + col)
                    .load(count=8, alignment=16)
                    .to(ACC_DTYPE)
                )
                merged_v += peer_v * peer_sum * peer_weight

            o_ptr = o.iterator.raw_ptr()
            out_row = (packed_q * cutlass.Int32(num_heads) + head) * cutlass.Int32(L)
            safe_mass = _fmax(global_mass, ACC_DTYPE(1.0e-30))
            merged = (merged_v / safe_mass).to(OUT_DTYPE)
            if cutlass.const_expr(num_heads == 16):
                # H16 empty split participants (sum == 0): with a single
                # surviving partial copy it unchanged (already normalized) so
                # split-N stays bitwise equal to split-1.
                if nonzero_peers == cutlass.Int32(1):
                    sole_o = prims.mapa(partial_o.data_ptr(), sole_peer)
                    merged = (sole_o + head * cutlass.Int32(L) + col).load(count=8, alignment=16)
            (o_ptr + out_row + col).store(merged, alignment=16)


@dataclass(frozen=True)
class MlaFmhaRuntimeShape:
    """Launch-varying shape metadata."""

    total_q: int
    batch_size: int
    pool_rows: int
    max_sq: int

    def validate(self) -> None:
        if self.total_q <= 0:
            raise ValueError(f"total_q must be positive, got {self.total_q}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.max_sq not in SUPPORTED_MAX_SQ:
            raise ValueError(f"max_sq must be one of {SUPPORTED_MAX_SQ}, got {self.max_sq}")
        if self.total_q > self.batch_size * self.max_sq:
            raise ValueError(
                f"packed total_q={self.total_q} exceeds batch_size * max_sq bound "
                f"({self.batch_size} * {self.max_sq})"
            )
        if self.pool_rows <= 0:
            raise ValueError(f"pool_rows must be positive, got {self.pool_rows}")


@dataclass(frozen=True)
class MlaFmhaSpecialization:
    """Values compiled as constexpr because they change generated device code."""

    all_sparse_indices_valid: bool
    num_heads: int = NUM_HEADS
    q_scaling: float = 1.0
    num_ctas_kv: int = 2
    use_gmem_reduction: bool = False
    index_kind: str = INDEX_KIND_GLOBAL

    def validate(self) -> None:
        if self.index_kind not in INDEX_KINDS:
            raise ValueError(f"index_kind must be one of {INDEX_KINDS}, got {self.index_kind!r}")
        if self.num_heads not in SUPPORTED_NUM_HEADS:
            raise ValueError(
                f"num_heads must be one of {SUPPORTED_NUM_HEADS}, got {self.num_heads}"
            )
        if not math.isfinite(self.q_scaling) or self.q_scaling <= 0.0:
            raise ValueError(f"q_scaling must be finite and positive, got {self.q_scaling}")
        if not 1 <= self.num_ctas_kv <= MAX_CTAS_KV:
            raise ValueError(f"num_ctas_kv must be in [1, {MAX_CTAS_KV}], got {self.num_ctas_kv}")
        if self.num_ctas_kv == 1 and self.use_gmem_reduction:
            raise ValueError("single-CTA specialization does not need GMEM reduction")

    @property
    def scale_log2(self) -> float:
        return LOG2_E / (self.q_scaling * math.sqrt(float(QK_HEAD_DIM)))

    @property
    def add_layer_offset(self) -> bool:
        return self.index_kind == INDEX_KIND_BASE_WITH_LAYER_OFFSET


def _device_index(tensor: torch.Tensor) -> int:
    device_index = tensor.device.index
    return device_index if device_index is not None else torch.cuda.current_device()


def _current_stream(device_index: int) -> cuda_driver.CUstream:
    return cuda_driver.CUstream(_raw_stream(device_index))


def _select_num_ctas_kv(
    *,
    batch_size: int,
    max_sq: int,
    max_seq_len_kv: int,
    multiprocessor_count: int,
) -> int:
    """TRT-LLM's sparse-generation split-K heuristic, restricted to power-of-two splits.

    TRT-LLM picks ``min(MAX_CTAS_KV, ceil(kv / 2*QK_N), SMs // query_ctas)``. On a
    148-SM B300 that yields 3 for the 48-row decode bucket (24 requests x MTP 2),
    and a 3-CTA split can only run as a cluster of 3 or through the GMEM reducer.
    Both lose to a 2-CTA cluster: the cluster-of-3 specialization is slower
    standalone, and the GMEM-3 path is slower in the decode attention PDL chain
    because its completion-counter spin is exposed under real co-scheduling
    (measured on B300). Any
    non-power-of-two split is therefore rounded DOWN to the largest power of two
    below it (3 -> 2, 5..7 -> 4); power-of-two splits (1, 2, 4, 8) are unchanged,
    so the 2/4/8/16/32/64-row buckets keep their previous specialization and the
    GMEM decision in :func:`_select_gmem_reduction` is untouched.

    Design decision: the rounding trades SM fill for a reducer that does not
    spin; it is not a statement that odd splits are wrong. At 48 rows a 3-way
    split is 144 CTAs on 148 SMs, the best fill, and it should win once the
    reducer does not spin (a last-arriving-CTA GMEM reducer where the other
    CTAs exit, or a cluster-of-3 reduce tree that packs well). Revisit when
    such a reducer exists: re-compare 2- vs 3-way at 48 rows and 4- vs 3-way
    at 32 rows in-chain, and drop the rounding if the odd split wins.
    """
    if max_seq_len_kv <= 0:
        raise ValueError(f"max_seq_len_kv must be positive, got {max_seq_len_kv}")
    query_ctas = batch_size * max_sq
    split_k_granularity = 2 * QK_N
    effective_kv = min(max_seq_len_kv, TOPK)
    max_ctas_for_kv = (effective_kv + split_k_granularity - 1) // split_k_granularity
    num_ctas_kv = min(
        MAX_CTAS_KV,
        max_ctas_for_kv,
        max(multiprocessor_count // query_ctas, 1),
    )
    return _largest_power_of_two_at_most(num_ctas_kv)


def _largest_power_of_two_at_most(value: int) -> int:
    """Return the largest power of two <= ``value`` (``value`` >= 1)."""
    if value < 1:
        raise ValueError(f"split factor must be >= 1, got {value}")
    return 1 << (value.bit_length() - 1)


def _max_active_clusters(device_index: int, cluster_size: int) -> int:
    """Cache TRT-equivalent one-CTA-per-SM cluster occupancy by device."""
    key = (device_index, cluster_size)
    cached = _MAX_ACTIVE_CLUSTERS_CACHE.get(key)
    if cached is not None:
        return cached

    cutlass.cuda.initialize_cuda_context(device_index)
    with torch.cuda.device(device_index):
        active_clusters = cutlass.utils.HardwareInfo(device_index).get_max_active_clusters(
            cluster_size
        )
    _MAX_ACTIVE_CLUSTERS_CACHE[key] = active_clusters
    return active_clusters


def _select_gmem_reduction(*, batch_size: int, max_sq: int, max_active_clusters: int) -> bool:
    """Use CGA only when every query cluster fits in one occupancy wave."""
    return batch_size * max_sq > max_active_clusters


@dataclass
class PreparedMlaFmhaRunner:
    """Validated, prebound launcher for the graph replay hot path."""

    compiled: object
    args: tuple[torch.Tensor | int, ...]
    device_index: int
    stream: cuda_driver.CUstream | None
    runtime_shape: MlaFmhaRuntimeShape
    specialization: MlaFmhaSpecialization

    def __call__(self) -> None:
        stream = self.stream
        if stream is None:
            stream = _current_stream(self.device_index)
        self.compiled(*self.args, stream)


def _fake_tensors(num_heads: int, num_ctas_kv: int, use_gmem_reduction: bool) -> tuple[object, ...]:
    make_fake = cute.runtime.make_fake_compact_tensor
    total_q = cute.sym_int64()
    pool_rows = cute.sym_int64()
    cu_q_count = cute.sym_int64()
    batch_size = cute.sym_int64()
    partial_o_shape = (total_q, num_ctas_kv, num_heads, L) if use_gmem_reduction else (1,)
    partial_stats_shape = (total_q, num_ctas_kv, num_heads, 2) if use_gmem_reduction else (1,)
    counter_shape = (total_q,) if use_gmem_reduction else (1,)
    return (
        make_fake(
            cutlass.Uint8,
            (total_q, num_heads, KV_STRIDE),
            stride_order=(2, 1, 0),
            assumed_align=16,
        ),
        make_fake(
            cutlass.Uint8,
            (pool_rows, KV_STRIDE),
            stride_order=(1, 0),
            assumed_align=16,
        ),
        make_fake(
            cutlass.Int32,
            (total_q, TOPK),
            stride_order=(1, 0),
            assumed_align=16,
        ),
        make_fake(
            OUT_DTYPE,
            (total_q, num_heads, L),
            stride_order=(2, 1, 0),
            assumed_align=16,
        ),
        make_fake(
            cutlass.Int32,
            (cu_q_count,),
            stride_order=(0,),
            assumed_align=16,
        ),
        make_fake(
            cutlass.Int32,
            (batch_size,),
            stride_order=(0,),
            # TRT-LLM may pass a request-subset view whose starting element
            # is not 16-byte aligned. This tensor is only scalar-loaded.
            assumed_align=4,
        ),
        make_fake(
            OUT_DTYPE,
            partial_o_shape,
            stride_order=tuple(reversed(range(len(partial_o_shape)))),
            assumed_align=16,
        ),
        make_fake(
            ACC_DTYPE,
            partial_stats_shape,
            stride_order=tuple(reversed(range(len(partial_stats_shape)))),
            assumed_align=16,
        ),
        make_fake(
            cutlass.Int32,
            counter_shape,
            stride_order=(0,),
            assumed_align=16,
        ),
    )


def _compile_options() -> str:
    return "--enable-tvm-ffi --opt-level 2"


def _require_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    dtype: torch.dtype,
    shape: tuple[int, ...] | None = None,
    device_type: str = "cuda",
    contiguous: bool = True,
) -> None:
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must be {dtype}, got {tensor.dtype}")
    if shape is not None and tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if tensor.device.type != device_type:
        raise ValueError(f"{name} must live on {device_type}, got {tensor.device}")
    if contiguous and not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _as_uint8_storage(tensor: torch.Tensor, *, name: str) -> torch.Tensor:
    if tensor.dtype == torch.uint8:
        return tensor
    if tensor.dtype == torch.float8_e4m3fn:
        return tensor.view(torch.uint8)
    raise TypeError(f"{name} must be uint8 storage or torch.float8_e4m3fn, got {tensor.dtype}")


def _require_same_device(
    tensors: tuple[tuple[str, torch.Tensor], ...],
    device: torch.device,
) -> None:
    for name, tensor in tensors:
        if tensor.device != device:
            raise ValueError(f"{name} must live on {device}, got {tensor.device}")


def _validate_scale_metadata(
    tensor: torch.Tensor | None,
    *,
    name: str,
    shape: tuple[int, ...],
    device: torch.device,
) -> None:
    if tensor is None:
        return
    _require_tensor(tensor, name=name, dtype=torch.float32, shape=shape)
    if tensor.device != device:
        raise ValueError(f"{name} must live on {device}, got {tensor.device}")
