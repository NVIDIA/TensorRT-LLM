# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sparse MLA FMHA for GLM generation, built on CuTe DSL primitives.

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
import operator
from dataclasses import dataclass

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda
import torch
from cutlass.experimental import primitives as prims

from ..common import compat48 as _compat48

QK_HEAD_DIM = 256
_COMPILE_CACHE: dict[tuple[object, ...], object] = {}
_MAX_ACTIVE_CLUSTERS_CACHE: dict[tuple[int, int], int] = {}

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
# H32 softmax lanes own a two-head column group and keep eight logical heads
# contiguous. PV output is instead logical-head major across the four column
# groups. Apply this inverse permutation at the correction/final handoff.
H32_LOGICAL_TO_PACKED_STATS: tuple[int, ...] = (
    0,
    1,
    8,
    9,
    16,
    17,
    24,
    25,
    2,
    3,
    10,
    11,
    18,
    19,
    26,
    27,
    4,
    5,
    12,
    13,
    20,
    21,
    28,
    29,
    6,
    7,
    14,
    15,
    22,
    23,
    30,
    31,
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
H32_TMEM_O_OFF: int = 64
H32_NUM_TMEM_COLS: int = 256

# TRTLLM stages K576 as five K128 slices and V512 as four V128 slices. A
# shared nine-slot ring lets next-tile K reuse slots as soon as QK releases
# them, independently of the longer-lived current-tile V/PV slots.
K_HDIM_STAGES: int = 5
V_HDIM_STAGES: int = 4
ST_KV: int = K_HDIM_STAGES + V_HDIM_STAGES
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
def _atomic_fmax_cta(ptr, value: cutlass.Float32) -> None:
    """Implement SMEM f32 atomic max with integer primitives.

    NVVM has no native f32 atomic max. IEEE-754 ordering allows the same
    lowering used by CUDA: unsigned min for negative values and signed max
    for non-negative values.
    """
    bits = value.bitcast(cutlass.Int32)
    if bits < cutlass.Int32(0):
        prims.atomicrmw(
            prims.AtomicOp.MIN,
            ptr,
            bits.bitcast(cutlass.Uint32),
            syncscope=prims.MemScope.CTA,
        )
    else:
        prims.atomicrmw(
            prims.AtomicOp.MAX,
            ptr,
            bits,
            syncscope=prims.MemScope.CTA,
        )


@cute.jit
def _float_to_u32_for_atomic_max(value: cutlass.Float32) -> cutlass.Uint32:
    bits = prims.mov_b32(value, target_type=cutlass.Int32)
    mask = (bits >> cutlass.Int32(31)) | cutlass.Int32(0x80000000)
    encoded = bits ^ mask
    return prims.mov_b32(encoded, target_type=cutlass.Uint32)


@cute.jit
def _u32_to_float_for_atomic_max(value: cutlass.Uint32) -> cutlass.Float32:
    encoded = prims.mov_b32(value, target_type=cutlass.Int32)
    mask = (~(encoded >> cutlass.Int32(31))) | cutlass.Int32(0x80000000)
    bits = encoded ^ mask
    return prims.mov_b32(bits, target_type=cutlass.Float32)


@cute.jit
def _atomic_max_u32_cta(ptr, value: cutlass.Uint32) -> None:
    prims.atomicrmw(
        prims.AtomicOp.MAX,
        ptr,
        value,
        syncscope=prims.MemScope.CTA,
        space=prims.SharedSpace.shared_cta,
    )


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
def _mask_scores_after_valid_prefix(
    scores,
    tile_start,
    valid_kv,
    warp_idx,
    row_idx,
    col_group,
    num_heads: cutlass.Constexpr[int],
):
    """Set TRT-LLM's positional sparse tail to -FLT_MAX before softmax."""
    masked = cutlass.Array(ACC_DTYPE, num_heads, space=cutlass.AddressSpace.rmem)
    for i in cutlass.range_constexpr(num_heads):
        if cutlass.const_expr(num_heads == 8):
            # Natural 16x256b fragment coordinate: registers 0..3 come from
            # the first 16-row load (rows lane//4 and +8), 4..7 from the
            # second (+16). With the P store swizzled on the absolute SMEM
            # address, PV consumes exactly this token coordinate.
            token_offset: cutlass.Constexpr[int] = 8 * ((i % 4) // 2) + 16 * (i // 4)
            token_in_tile = warp_idx * cutlass.Int32(32) + row_idx + cutlass.Int32(token_offset)
        elif cutlass.const_expr(num_heads == 32):
            # Natural 16x256b x4 fragment coordinate: registers 0..15 come
            # from the first 16-row load (rows lane//4 and +8), 16..31 from
            # the second (+16); the head is 8*((i%16)//4) + 2*col_group + i%2.
            token_offset: cutlass.Constexpr[int] = 8 * ((i % 4) // 2) + 16 * (i // 16)
            token_in_tile = warp_idx * cutlass.Int32(32) + row_idx + cutlass.Int32(token_offset)
        else:
            fragment_index: cutlass.Constexpr[int] = i % 16
            token_offset: cutlass.Constexpr[int] = (
                ((fragment_index % 4) // 2) + 2 * (fragment_index // 8)
            ) * 8
            token_in_tile = warp_idx * cutlass.Int32(32) + row_idx + cutlass.Int32(token_offset)
        value = scores[i]
        if tile_start + token_in_tile >= valid_kv:
            value = ACC_DTYPE(-3.0e38)
        masked[i] = value
    return cutlass.Vector.from_elements(tuple(masked[i] for i in range(num_heads)), dtype=ACC_DTYPE)


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
def _warpgroup_row_sum8_packed(v0, v1, v2, v3, v4, v5, v6, v7):
    """Reduce eight columns with TRTLLM-style paired FP32 adds."""
    sum01 = (v0, v1)
    sum23 = (v2, v3)
    sum45 = (v4, v5)
    sum67 = (v6, v7)
    for offset in (16, 8, 4):
        other01 = (
            ACC_DTYPE(
                prims.shfl_sync(
                    thread_mask=0xFFFFFFFF,
                    val=sum01[0],
                    offset=offset,
                    mask_and_clamp=0x1F,
                    kind=prims.Shfl.BFLY,
                )
            ),
            ACC_DTYPE(
                prims.shfl_sync(
                    thread_mask=0xFFFFFFFF,
                    val=sum01[1],
                    offset=offset,
                    mask_and_clamp=0x1F,
                    kind=prims.Shfl.BFLY,
                )
            ),
        )
        other23 = (
            ACC_DTYPE(
                prims.shfl_sync(
                    thread_mask=0xFFFFFFFF,
                    val=sum23[0],
                    offset=offset,
                    mask_and_clamp=0x1F,
                    kind=prims.Shfl.BFLY,
                )
            ),
            ACC_DTYPE(
                prims.shfl_sync(
                    thread_mask=0xFFFFFFFF,
                    val=sum23[1],
                    offset=offset,
                    mask_and_clamp=0x1F,
                    kind=prims.Shfl.BFLY,
                )
            ),
        )
        other45 = (
            ACC_DTYPE(
                prims.shfl_sync(
                    thread_mask=0xFFFFFFFF,
                    val=sum45[0],
                    offset=offset,
                    mask_and_clamp=0x1F,
                    kind=prims.Shfl.BFLY,
                )
            ),
            ACC_DTYPE(
                prims.shfl_sync(
                    thread_mask=0xFFFFFFFF,
                    val=sum45[1],
                    offset=offset,
                    mask_and_clamp=0x1F,
                    kind=prims.Shfl.BFLY,
                )
            ),
        )
        other67 = (
            ACC_DTYPE(
                prims.shfl_sync(
                    thread_mask=0xFFFFFFFF,
                    val=sum67[0],
                    offset=offset,
                    mask_and_clamp=0x1F,
                    kind=prims.Shfl.BFLY,
                )
            ),
            ACC_DTYPE(
                prims.shfl_sync(
                    thread_mask=0xFFFFFFFF,
                    val=sum67[1],
                    offset=offset,
                    mask_and_clamp=0x1F,
                    kind=prims.Shfl.BFLY,
                )
            ),
        )
        sum01 = prims.add_packed_f32x2(sum01, other01, rnd=prims.FPRoundingMode.RN)
        sum23 = prims.add_packed_f32x2(sum23, other23, rnd=prims.FPRoundingMode.RN)
        sum45 = prims.add_packed_f32x2(sum45, other45, rnd=prims.FPRoundingMode.RN)
        sum67 = prims.add_packed_f32x2(sum67, other67, rnd=prims.FPRoundingMode.RN)
    return cutlass.Vector.from_elements(
        (
            sum01[0],
            sum01[1],
            sum23[0],
            sum23[1],
            sum45[0],
            sum45[1],
            sum67[0],
            sum67[1],
        ),
        dtype=ACC_DTYPE,
    )


@cute.jit
def _h32_group_max(value: cutlass.Float32) -> cutlass.Float32:
    """Reduce one H32 logical head across its eight same-column lanes."""
    for offset in (16, 8, 4):
        value = _fmax(
            value,
            cutlass.Float32(
                prims.shfl_sync(
                    thread_mask=0xFFFFFFFF,
                    val=value,
                    offset=offset,
                    mask_and_clamp=0x1F,
                    kind=prims.Shfl.BFLY,
                )
            ),
        )
    return value


@cute.jit
def _q8_group_max(value: cutlass.Float32) -> cutlass.Float32:
    for offset in (16, 8):
        value = _fmax(
            value,
            cutlass.Float32(
                prims.shfl_sync(
                    thread_mask=0xFFFFFFFF,
                    val=value,
                    offset=offset,
                    mask_and_clamp=0x1F,
                    kind=prims.Shfl.BFLY,
                )
            ),
        )
    return value


@cute.jit
def _q8_group_sum(value: cutlass.Float32) -> cutlass.Float32:
    # A fixed H8 head pair spans all eight lanes with the same ``lane % 4``.
    # The final xor-4 step merges the two four-lane halves before lanes 0..3
    # publish the denominator; omitting it drops exactly half of the mass.
    for offset in (16, 8, 4):
        value += cutlass.Float32(
            prims.shfl_sync(
                thread_mask=0xFFFFFFFF,
                val=value,
                offset=offset,
                mask_and_clamp=0x1F,
                kind=prims.Shfl.BFLY,
            )
        )
    return value


@cute.jit
def _store_transposed_p_q8(
    smem_base,
    stage: cutlass.Int32,
    reg0: cutlass.Int32,
    reg1: cutlass.Int32,
    warp_group_thread: cutlass.Int32,
    num_heads: cutlass.Constexpr[int],
) -> None:
    """Store one Q8 FP8 P fragment in the SwapsAb operand layout.

    Lanes 0..7 address head rows 0..7 of the first 16-token chunk (reg0,
    tokens 32*warp + [0, 16)), lanes 8..15 the second chunk (reg1). The
    SWIZZLE_128B pattern the PV descriptor expects is a function of the
    absolute SMEM address, so it is applied with ``apply_swizzle`` on the
    linear address (sP is only 128-byte aligned; a base-relative XOR
    permutes the 16-token chunks of 7 of every 8 rows).
    """
    warp = warp_group_thread >> cutlass.Int32(5)
    lane = warp_group_thread & cutlass.Int32(0x1F)
    thread_row = lane & cutlass.Int32(0x7)
    matrix_col = (lane >> cutlass.Int32(3)) & cutlass.Int32(1)
    byte_offset = (
        stage * cutlass.Int32(num_heads * QK_N)
        + thread_row * cutlass.Int32(QK_N)
        + (warp * cutlass.Int32(2) + matrix_col) * cutlass.Int32(16)
    )
    smem_swizzle = cutlass.Swizzle.from_name("s128b")
    prims.stmatrix(
        (smem_base.data_ptr() + byte_offset).apply_swizzle(smem_swizzle),
        [reg0, reg1],
        prims.MMALayout.COL,
        shape=prims.StoreShape.M16N8,
    )


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


@cute.kernel
def _fmha_sparse_mla_kernel(
    tma_ql: cutlass.GridConstant[cuda.TensorMap],
    tma_qr: cutlass.GridConstant[cuda.TensorMap],
    tma_kv: cutlass.GridConstant[cuda.TensorMap],
    tma_kr: cutlass.GridConstant[cuda.TensorMap],
    kv: cute.Tensor,
    sparse_indices: cute.Tensor,
    o: cute.Tensor,
    cu_q_seqlens: cute.Tensor,
    sequence_length: cute.Tensor,
    partial_o_workspace: cute.Tensor,
    partial_stats_workspace: cute.Tensor,
    completion_counters: cute.Tensor,
    layer_offset: cutlass.Int32,
    scale_log2: cutlass.Float32,
    num_heads: cutlass.Constexpr[int],
    all_sparse_indices_valid: cutlass.Constexpr[bool],
    num_ctas_kv: cutlass.Constexpr[int],
    use_gmem_reduction: cutlass.Constexpr[bool],
    add_layer_offset: cutlass.Constexpr[bool],
) -> None:
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.warp_idx()
    task_x, _, batch = cute.arch.block_idx()
    if cutlass.const_expr(use_gmem_reduction):
        cta = task_x % cutlass.Int32(num_ctas_kv)
    else:
        cta = cute.arch.block_idx_in_cluster()
    query_in_batch = task_x // cutlass.Int32(num_ctas_kv)

    index_buffer_stages: cutlass.Constexpr[int] = TOPK // I_STAGE_ELTS if num_ctas_kv == 1 else ST_I
    num_kt_per_cta: cutlass.Constexpr[int] = (NUM_KV_TILES + num_ctas_kv - 1) // num_ctas_kv
    used_index_stages: cutlass.Constexpr[int] = (
        num_kt_per_cta + I_TILES_PER_STAGE - 1
    ) // I_TILES_PER_STAGE
    needs_tail_mask: cutlass.Constexpr[bool] = (
        not all_sparse_indices_valid or NUM_KV_TILES % num_ctas_kv != 0
    )
    is_h8: cutlass.Constexpr[bool] = num_heads == 8
    is_h16: cutlass.Constexpr[bool] = num_heads == 16
    is_h32: cutlass.Constexpr[bool] = num_heads == 32
    num_scale_groups: cutlass.Constexpr[int] = num_heads // 4
    q_repeats: cutlass.Constexpr[int] = num_heads // 8
    scale_alignment: cutlass.Constexpr[int] = 16 if num_scale_groups >= 4 else 8
    defer_h32_sum: cutlass.Constexpr[bool] = is_h32 and not needs_tail_mask
    h32_max_stages: cutlass.Constexpr[int] = num_kt_per_cta if defer_h32_sum else ST_S
    balanced_h8_tail: cutlass.Constexpr[bool] = is_h8 and (num_ctas_kv == 6 or num_ctas_kv == 7)
    tmem_o_offset: cutlass.Constexpr[int] = H32_TMEM_O_OFF if is_h32 else TMEM_O_OFF
    num_tmem_cols: cutlass.Constexpr[int] = H32_NUM_TMEM_COLS if is_h32 else NUM_TMEM_COLS
    # N1 executes all 16 Kv128 tiles in one CTA. Four-way partial unrolling
    # removes uniform loop-control overhead without growing the code enough to
    # cause instruction-cache misses; split-K specializations retain their original
    # code generation, including the default N2 path.
    kv_loop_unroll: cutlass.Constexpr[int] = (
        1 if is_h32 and num_ctas_kv == 1 else 4 if num_ctas_kv == 1 else 1
    )

    # ---- SMEM ------------------------------------------------------------
    sQ = cutlass.Array(IO_DTYPE, num_heads * L, space=cutlass.AddressSpace.smem, alignment=128)
    sQr = cutlass.Array(IO_DTYPE, num_heads * R, space=cutlass.AddressSpace.smem, alignment=128)
    sKv = cutlass.Array(
        IO_DTYPE,
        ST_KV * QK_N * QK_K,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )
    sparse_stage = cutlass.Array(
        cutlass.Int32,
        index_buffer_stages * I_STAGE_ELTS,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )
    reduction_scratch = cutlass.Array(
        cutlass.Int32, 1, space=cutlass.AddressSpace.smem, alignment=32
    )
    sP = cutlass.Array(
        IO_DTYPE,
        ST_P * QK_N * num_heads,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )
    max_warp_rows: cutlass.Constexpr[int] = 4 if is_h16 else 1
    softmax_max_entries: cutlass.Constexpr[int] = 1 if is_h32 else ST_S * max_warp_rows * num_heads
    softmax_max = cutlass.Array(
        ACC_DTYPE,
        softmax_max_entries,
        space=cutlass.AddressSpace.smem,
        alignment=16,
    )
    softmax_max_u32_entries: cutlass.Constexpr[int] = h32_max_stages * num_heads if is_h32 else 1
    softmax_max_u32 = cutlass.Array(
        cutlass.Uint32,
        softmax_max_u32_entries,
        space=cutlass.AddressSpace.smem,
        alignment=16,
    )
    softmax_sum = cutlass.Array(
        ACC_DTYPE,
        4 * num_heads,
        space=cutlass.AddressSpace.smem,
        alignment=16,
    )
    compact_stats = cutlass.Array(
        ACC_DTYPE,
        ST_C * num_heads * 3,
        space=cutlass.AddressSpace.smem,
        alignment=16,
    )
    partial_o = cutlass.Array(
        OUT_DTYPE,
        num_heads * L,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )
    partial_stats = cutlass.Array(
        ACC_DTYPE,
        num_heads * 2,
        space=cutlass.AddressSpace.smem,
        alignment=16,
    )
    q_full = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem)
    kv_full = cutlass.Array(cutlass.Int64, ST_KV, space=cutlass.AddressSpace.smem)
    kv_empty = cutlass.Array(cutlass.Int64, ST_KV, space=cutlass.AddressSpace.smem)
    index_full = cutlass.Array(cutlass.Int64, index_buffer_stages, space=cutlass.AddressSpace.smem)
    s_full = cutlass.Array(cutlass.Int64, ST_S, space=cutlass.AddressSpace.smem)
    s_empty = cutlass.Array(cutlass.Int64, ST_S, space=cutlass.AddressSpace.smem)
    p_full = cutlass.Array(cutlass.Int64, ST_P, space=cutlass.AddressSpace.smem)
    p_empty = cutlass.Array(cutlass.Int64, ST_P, space=cutlass.AddressSpace.smem)
    c_full = cutlass.Array(cutlass.Int64, ST_C, space=cutlass.AddressSpace.smem)
    c_empty = cutlass.Array(cutlass.Int64, ST_C, space=cutlass.AddressSpace.smem)
    o_full = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem)
    o_empty = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem)
    tmem_i32 = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem)
    query_bounds = cutlass.Array(cutlass.Int32, 3, space=cutlass.AddressSpace.smem, alignment=16)

    if warp_idx == 0:
        if prims.elect_sync():
            prims.mbarrier_init(q_full, 1)
            for i in cutlass.range_constexpr(ST_KV):
                prims.mbarrier_init(kv_full.data_ptr() + i, 1)
                prims.mbarrier_init(kv_empty.data_ptr() + i, 1)
            for i in cutlass.range_constexpr(index_buffer_stages):
                prims.mbarrier_init(index_full.data_ptr() + i, 1)
            for i in cutlass.range_constexpr(ST_S):
                prims.mbarrier_init(s_full.data_ptr() + i, 1)
                prims.mbarrier_init(s_empty.data_ptr() + i, 1)
            for i in cutlass.range_constexpr(ST_P):
                prims.mbarrier_init(p_full.data_ptr() + i, 4)
                prims.mbarrier_init(p_empty.data_ptr() + i, 1)
            for i in cutlass.range_constexpr(ST_C):
                prims.mbarrier_init(c_full.data_ptr() + i, 1)
                prims.mbarrier_init(c_empty.data_ptr() + i, 4)
            prims.mbarrier_init(o_full, 1)
            prims.mbarrier_init(o_empty, 4)
    prims.fence_mbarrier_init()
    prims.barrier_cluster_arrive_relaxed()

    # Each CTA owns an independent CTA1 TMEM allocation. The cluster is used
    # only for the final partial-O/statistics reduction, not for CTA2 MMA.
    if warp_idx == W_MMA_PV:
        prims.tcgen05_alloc(tmem_i32, num_tmem_cols, group="cta_1")
    if warp_idx == W_LOAD_KQ:
        # Only W9 dereferences the predecessor-owned query prefix. Its wait
        # overlaps barrier initialization and CTA1 TMEM allocation by the
        # other roles; the following CTA barrier publishes both bounds.
        if prims.elect_sync():
            prims.griddepcontrol(prims.GridDepAction.WAIT)
            query_bounds[0] = cu_q_seqlens[batch] // cutlass.Int32(num_heads)
            query_bounds[1] = cu_q_seqlens[batch + cutlass.Int32(1)] // cutlass.Int32(num_heads)
            query_bounds[2] = sequence_length[batch]
    prims.barrier_cluster_wait()
    prims.barrier_cta_sync(0)
    tmem_raw = tmem_i32.load()
    tmem_ptr = cutlass.inttoptr(tmem_raw, 6, cutlass.Float32)
    if warp_idx == W_MMA_PV:
        prims.tcgen05_relinquish_alloc_permit(group="cta_1")

    q_begin = query_bounds[0]
    q_end = query_bounds[1]
    if query_in_batch >= q_end - q_begin:
        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
        prims.barrier_cta_sync(0)
        if warp_idx == W_MMA_PV:
            prims.tcgen05_dealloc(tmem_ptr, num_tmem_cols, group="cta_1")
        if tidx == 0:
            prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)
        prims.exit()
    packed_q = q_begin + query_in_batch
    valid_kv = query_bounds[2] - (q_end - q_begin) + query_in_batch + cutlass.Int32(1)
    if valid_kv < cutlass.Int32(0):
        valid_kv = cutlass.Int32(0)
    if valid_kv > cutlass.Int32(TOPK):
        valid_kv = cutlass.Int32(TOPK)
    if cutlass.const_expr(is_h32):
        if tidx < cutlass.Int32(softmax_max_u32_entries):
            softmax_max_u32[tidx] = _float_to_u32_for_atomic_max(ACC_DTYPE(-3.0e38))
    else:
        if tidx < cutlass.Int32(ST_S * num_heads):
            softmax_max[tidx] = ACC_DTYPE(-3.0e38)
    prims.barrier_cta_sync(0)

    # The all-valid specialization consumes exactly TOPK entries. The active
    # KV sequence may be larger than TOPK; it must not change the MMA count.
    if cutlass.const_expr(balanced_h8_tail):
        # Ceil partitioning gives the final H8 N6/N7 CTA nominal tiles beyond
        # tile 16. Floor partitioning keeps each CTA's staged index range
        # inside TOPK; padding within its final index stage remains masked.
        first_kv_tile = (cta * cutlass.Int32(NUM_KV_TILES)) // cutlass.Int32(num_ctas_kv)
        end_kv_tile = ((cta + cutlass.Int32(1)) * cutlass.Int32(NUM_KV_TILES)) // cutlass.Int32(
            num_ctas_kv
        )
        num_kt = end_kv_tile - first_kv_tile
    else:
        first_kv_tile = cta * cutlass.Int32(num_kt_per_cta)
        num_kt = cutlass.Int32(num_kt_per_cta)
    if cutlass.const_expr(num_heads == 16 and not all_sparse_indices_valid):
        # StaticTokenSparse has a contiguous valid prefix, so fully-invalid
        # trailing Kv128 tiles do not participate in softmax. Repartition the
        # runtime-valid tiles across H16 split CTAs. A CTA whose share of the
        # valid tiles is empty (num_kt == 0) is an EMPTY PARTICIPANT: it skips
        # the gather / QK / softmax / PV pipeline entirely and contributes the
        # neutral partial (max = -3.0e38, sum = 0, O = 0) through the unchanged
        # reducer protocol. The fixed 16 / num_ctas_kv-tile partition would make
        # every split CTA scan the whole masked row (about 5x slower at 64 rows,
        # split 2, KV <= 128).
        valid_tiles = (valid_kv + cutlass.Int32(QK_N - 1)) // cutlass.Int32(QK_N)
        if valid_tiles < cutlass.Int32(1):
            valid_tiles = cutlass.Int32(1)
        if cutlass.const_expr(num_ctas_kv == 1):
            num_kt = valid_tiles
        else:
            # Keep the original fixed bounds when the final Kv128 tile may be
            # valid. This removes runtime division from the full/near-full hot
            # path while retaining tail skipping when at least one whole tile
            # is invalid.
            if valid_kv <= cutlass.Int32(TOPK - QK_N):
                first_kv_tile = (cta * valid_tiles) // cutlass.Int32(num_ctas_kv)
                num_kt = ((cta + cutlass.Int32(1)) * valid_tiles) // cutlass.Int32(
                    num_ctas_kv
                ) - first_kv_tile
    # Empty split participants exist only on the H16 StaticTokenSparse split
    # path; every other specialization keeps num_kt >= 1 and its codegen.
    empty_split_participants: cutlass.Constexpr[bool] = (
        num_heads == 16 and not all_sparse_indices_valid and num_ctas_kv > 1
    )
    q_bytes = (num_heads * L + num_heads * R) * EB

    # Configure donor warpgroup WG3 before WG2 requests its H32 register
    # budget. PTX requires all four warps in a warpgroup to execute the same
    # setmaxnreg operation.
    if (warp_idx >= cutlass.Int32(W_LOAD_GATHER0)) and (
        warp_idx < cutlass.Int32(W_LOAD_GATHER0 + NUM_GATHER_WARPS)
    ):
        prims.setmaxregister(80, prims.SetMaxRegisterAction.DECREASE)

    # WG2 contains W8 QK, W9 Q/page-offset, idle W10, and W11 PV. H32 needs
    # the otherwise-unused register budget for its wider MMA descriptors.
    if (warp_idx >= cutlass.Int32(8)) and (warp_idx < cutlass.Int32(12)):
        if cutlass.const_expr(is_h32):
            prims.setmaxregister(104, prims.SetMaxRegisterAction.DECREASE)
        else:
            prims.setmaxregister(80, prims.SetMaxRegisterAction.DECREASE)

    # =====================================================================
    # W9 - Q TMA plus six-stage sparse-offset producer. One 256-element
    # offset stage feeds two consecutive Kv128 payload tiles.
    # =====================================================================
    if warp_idx == W_LOAD_KQ:
        if prims.elect_sync():
            prims.mbarrier_arrive_expect_tx(q_full, q_bytes)
        if prims.elect_sync():
            for c in cutlass.range_constexpr(N_LAT):
                prims.cp_async_bulk_tensor_shared_cta_global(
                    sQ.data_ptr() + c * (num_heads * QK_K),
                    tma_ql.get_ptr(),
                    (c * QK_K, 0, packed_q),
                    q_full,
                )
            prims.cp_async_bulk_tensor_shared_cta_global(
                sQr,
                tma_qr.get_ptr(),
                (0, 0, packed_q),
                q_full,
            )

        sparse_ptr = sparse_indices.iterator.raw_ptr()
        lane = tidx % cutlass.Int32(32)
        query_base = packed_q * cutlass.Int32(TOPK) + first_kv_tile * cutlass.Int32(QK_N)
        for i_stage in cutlass.range_constexpr(used_index_stages):
            for half in cutlass.range_constexpr(I_STAGE_ELTS // (32 * 4)):
                row = cutlass.Int32(half * 32 * 4) + lane * cutlass.Int32(4)
                if cutlass.const_expr(needs_tail_mask):
                    values = cutlass.Vector.from_elements(
                        tuple(cutlass.Int32(-1) for _ in range(4)),
                        dtype=cutlass.Int32,
                    )
                    sparse_offset = (
                        first_kv_tile * cutlass.Int32(QK_N)
                        + cutlass.Int32(i_stage * I_STAGE_ELTS)
                        + row
                    )
                    in_bounds = sparse_offset < cutlass.Int32(TOPK)
                    if cutlass.const_expr(balanced_h8_tail):
                        local_sparse_offset = cutlass.Int32(i_stage * I_STAGE_ELTS) + row
                        in_bounds = in_bounds and (
                            local_sparse_offset < num_kt * cutlass.Int32(QK_N)
                        )
                    if in_bounds:
                        values = (
                            sparse_ptr + query_base + cutlass.Int32(i_stage * I_STAGE_ELTS) + row
                        ).load(count=4, alignment=16)
                else:
                    values = (
                        sparse_ptr + query_base + cutlass.Int32(i_stage * I_STAGE_ELTS) + row
                    ).load(count=4, alignment=16)
                if cutlass.const_expr(add_layer_offset):
                    # Base rows -> this layer's pool rows (-1 stays -1).
                    values = _add_layer_offset_vector4(values, layer_offset)
                (sparse_stage.data_ptr() + cutlass.Int32(i_stage * I_STAGE_ELTS) + row).store(
                    values, alignment=16
                )
            prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
            if prims.elect_sync():
                prims.mbarrier_arrive(index_full.data_ptr() + i_stage, relaxed=True)

    # =====================================================================
    # W12-W15 - four independent direct sparse gather4 producers. Each warp
    # owns one quarter of both gather trees, matching TRTLLM's sparse Load WG.
    # =====================================================================
    if (warp_idx >= cutlass.Int32(W_LOAD_GATHER0)) and (
        warp_idx < cutlass.Int32(W_LOAD_GATHER0 + NUM_GATHER_WARPS)
    ):
        producer_rank = warp_idx - cutlass.Int32(W_LOAD_GATHER0)
        routed_rows = cutlass.Array(cutlass.Int32, 32, space=cutlass.AddressSpace.rmem)

        # Prime K(0). Every steady-state iteration then fills K(i+1) into
        # panels released by QK(i) before filling V(i). This exposes the same
        # QK-next/PV-current overlap as the H64 schedule. An empty split
        # participant (num_kt == 0) must not prime: nobody would consume K(0).
        prime_tiles = cutlass.Int32(1)
        if cutlass.const_expr(empty_split_participants):
            prime_tiles = num_kt
        if prime_tiles > cutlass.Int32(0):
            while not prims.mbarrier_try_wait_parity(index_full.data_ptr(), 0):
                pass
            staged_base = producer_rank * cutlass.Int32(CTA_N // NUM_GATHER_WARPS)
            for g in cutlass.range_constexpr(CTA_N // (NUM_GATHER_WARPS * 4)):
                values = (sparse_stage.data_ptr() + staged_base + cutlass.Int32(g * 4)).load(
                    count=4, alignment=16
                )
                for j in cutlass.range_constexpr(4):
                    routed_rows[g * 4 + j] = values[j]
            for c in cutlass.range_constexpr(K_HDIM_STAGES):
                while not prims.mbarrier_try_wait_parity(kv_empty.data_ptr() + c, cutlass.Int32(1)):
                    pass
                if producer_rank == cutlass.Int32(0):
                    if prims.elect_sync():
                        prims.mbarrier_arrive_expect_tx(kv_full.data_ptr() + c, KV_STAGE_BYTES)
                for g in cutlass.range_constexpr(CTA_N // (NUM_GATHER_WARPS * 4)):
                    _tma_gather4_cta1(
                        sKv.data_ptr()
                        + c * cutlass.Int32(CTA_N * QK_K)
                        + (
                            producer_rank * cutlass.Int32(CTA_N // NUM_GATHER_WARPS)
                            + cutlass.Int32(g * 4)
                        )
                        * cutlass.Int32(QK_K),
                        tma_kv.get_ptr(),
                        kv_full.data_ptr() + c,
                        cutlass.Int32(c * QK_K),
                        routed_rows[g * 4],
                        routed_rows[g * 4 + 1],
                        routed_rows[g * 4 + 2],
                        routed_rows[g * 4 + 3],
                    )

        for kt in cutlass.range(0, num_kt, 1, unroll=1):
            if kt + cutlass.Int32(1) < num_kt:
                next_kt = kt + cutlass.Int32(1)
                next_ist = next_kt // cutlass.Int32(I_TILES_PER_STAGE)
                while not prims.mbarrier_try_wait_parity(index_full.data_ptr() + next_ist, 0):
                    pass
                staged_base = (
                    next_ist * cutlass.Int32(I_STAGE_ELTS)
                    + (next_kt % cutlass.Int32(I_TILES_PER_STAGE)) * cutlass.Int32(QK_N)
                    + producer_rank * cutlass.Int32(CTA_N // NUM_GATHER_WARPS)
                )
                for g in cutlass.range_constexpr(CTA_N // (NUM_GATHER_WARPS * 4)):
                    values = (sparse_stage.data_ptr() + staged_base + cutlass.Int32(g * 4)).load(
                        count=4, alignment=16
                    )
                    for j in cutlass.range_constexpr(4):
                        routed_rows[g * 4 + j] = values[j]

                next_phase = next_kt % cutlass.Int32(2)
                for c in cutlass.range_constexpr(K_HDIM_STAGES):
                    while not prims.mbarrier_try_wait_parity(
                        kv_empty.data_ptr() + c, next_phase ^ cutlass.Int32(1)
                    ):
                        pass
                    if producer_rank == cutlass.Int32(0):
                        if prims.elect_sync():
                            prims.mbarrier_arrive_expect_tx(kv_full.data_ptr() + c, KV_STAGE_BYTES)
                    for g in cutlass.range_constexpr(CTA_N // (NUM_GATHER_WARPS * 4)):
                        _tma_gather4_cta1(
                            sKv.data_ptr()
                            + c * cutlass.Int32(CTA_N * QK_K)
                            + (
                                producer_rank * cutlass.Int32(CTA_N // NUM_GATHER_WARPS)
                                + cutlass.Int32(g * 4)
                            )
                            * cutlass.Int32(QK_K),
                            tma_kv.get_ptr(),
                            kv_full.data_ptr() + c,
                            cutlass.Int32(c * QK_K),
                            routed_rows[g * 4],
                            routed_rows[g * 4 + 1],
                            routed_rows[g * 4 + 2],
                            routed_rows[g * 4 + 3],
                        )

            # Reload the current tile's offsets after the K-ahead issue.
            i_stage = kt // cutlass.Int32(I_TILES_PER_STAGE)
            while not prims.mbarrier_try_wait_parity(index_full.data_ptr() + i_stage, 0):
                pass
            staged_base = (
                i_stage * cutlass.Int32(I_STAGE_ELTS)
                + (kt % cutlass.Int32(I_TILES_PER_STAGE)) * cutlass.Int32(QK_N)
                + producer_rank * cutlass.Int32(CTA_N // NUM_GATHER_WARPS)
            )
            for g in cutlass.range_constexpr(CTA_N // (NUM_GATHER_WARPS * 4)):
                values = (sparse_stage.data_ptr() + staged_base + cutlass.Int32(g * 4)).load(
                    count=4, alignment=16
                )
                for j in cutlass.range_constexpr(4):
                    routed_rows[g * 4 + j] = values[j]

            current_phase = kt % cutlass.Int32(2)
            for s in cutlass.range_constexpr(V_HDIM_STAGES):
                slot: cutlass.Constexpr[int] = K_HDIM_STAGES + s
                while not prims.mbarrier_try_wait_parity(
                    kv_empty.data_ptr() + slot, current_phase ^ cutlass.Int32(1)
                ):
                    pass
                if producer_rank == cutlass.Int32(0):
                    if prims.elect_sync():
                        prims.mbarrier_arrive_expect_tx(kv_full.data_ptr() + slot, KV_STAGE_BYTES)
                col = cutlass.Int32(s * PV_M)
                for g in cutlass.range_constexpr(QK_N // (NUM_GATHER_WARPS * 4)):
                    row = producer_rank * cutlass.Int32(QK_N // NUM_GATHER_WARPS) + cutlass.Int32(
                        g * 4
                    )
                    _tma_gather4_cta1(
                        sKv.data_ptr()
                        + slot * cutlass.Int32(QK_N * QK_K)
                        + row * cutlass.Int32(PV_M),
                        tma_kv.get_ptr(),
                        kv_full.data_ptr() + slot,
                        col,
                        routed_rows[g * 4],
                        routed_rows[g * 4 + 1],
                        routed_rows[g * 4 + 2],
                        routed_rows[g * 4 + 3],
                    )

    # =====================================================================
    # W8 — independent CTA1 QK for this CTA's 1024-token split
    # =====================================================================
    if warp_idx == W_MMA_QK:
        mma_heads: cutlass.Constexpr[int] = num_heads
        head_blocks: cutlass.Constexpr[int] = 1
        idesc_qk = prims.Tcgen05InstrDesc.build(
            c_dtype=ACC_DTYPE,
            a_dtype=IO_DTYPE,
            b_dtype=IO_DTYPE,
            n_dim=mma_heads,
            m_dim=M_TILE,
        )
        while not prims.mbarrier_try_wait_parity(q_full, 0):
            pass
        dq = _compat48.tcgen05_smem_desc(
            start_address=sQ,
            leading_byte_offset=LBO_K,
            stride_byte_offset=SBO_LAT,
            layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
        )
        dqr = _compat48.tcgen05_smem_desc(
            start_address=sQr,
            leading_byte_offset=LBO_K,
            stride_byte_offset=SBO_ROPE,
            layout=prims.Tcgen05SmemSwizzle.SWIZZLE_64B,
        )
        dq_hi = dq
        dqr_hi = dqr
        if cutlass.const_expr(is_h32):
            dq_hi = _compat48.tcgen05_smem_desc(
                start_address=sQ.data_ptr() + 16 * QK_K,
                leading_byte_offset=LBO_K,
                stride_byte_offset=SBO_LAT,
                layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
            )
            dqr_hi = _compat48.tcgen05_smem_desc(
                start_address=sQr.data_ptr() + 16 * R,
                leading_byte_offset=LBO_K,
                stride_byte_offset=SBO_ROPE,
                layout=prims.Tcgen05SmemSwizzle.SWIZZLE_64B,
            )
        dk_root = _compat48.tcgen05_smem_desc(
            start_address=sKv,
            leading_byte_offset=LBO_K,
            stride_byte_offset=SBO_LAT,
            layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
        )

        kvph = cutlass.Int32(0)
        sst = cutlass.Int32(0)
        sph = cutlass.Int32(0)
        for kt in cutlass.range(0, num_kt, 1, unroll=kv_loop_unroll):
            while not prims.mbarrier_try_wait_parity(
                s_empty.data_ptr() + sst, sph ^ cutlass.Int32(1)
            ):
                pass
            for c in cutlass.range_constexpr(K_HDIM_STAGES):
                while not prims.mbarrier_try_wait_parity(kv_full.data_ptr() + c, kvph):
                    pass
                ko: cutlass.Constexpr[int] = c * CTA_N * QK_K * EB
                for head_block in cutlass.range_constexpr(head_blocks):
                    stmem = cutlass.inttoptr(
                        tmem_raw + sst * cutlass.Int32(num_heads) + head_block * mma_heads,
                        6,
                        cutlass.Float32,
                    )
                    dq_block = dq
                    dqr_block = dqr
                    if cutlass.const_expr(head_block == 1):
                        dq_block = dq_hi
                        dqr_block = dqr_hi
                    if c < N_LAT:
                        qo: cutlass.Constexpr[int] = c * num_heads * QK_K * EB
                        for kb in cutlass.range_constexpr(QK_K // GRAN):
                            g: cutlass.Constexpr[int] = kb * GRAN * EB
                            if prims.elect_sync():
                                prims.tcgen05_mma(
                                    prims.Tcgen05MMAKind.F8F6F4,
                                    prims.CTAGroup.CTA_1,
                                    stmem,
                                    dk_root.advance_start_address(ko + g),
                                    dq_block.advance_start_address(qo + g),
                                    idesc_qk,
                                    (c != 0) or (kb != 0),
                                )
                    else:
                        for kb in cutlass.range_constexpr(R // GRAN):
                            g: cutlass.Constexpr[int] = kb * GRAN * EB
                            if prims.elect_sync():
                                prims.tcgen05_mma(
                                    prims.Tcgen05MMAKind.F8F6F4,
                                    prims.CTAGroup.CTA_1,
                                    stmem,
                                    dk_root.advance_start_address(ko + g),
                                    dqr_block.advance_start_address(g),
                                    idesc_qk,
                                    True,
                                )
                if prims.elect_sync():
                    if kt + cutlass.Int32(1) < num_kt:
                        prims.tcgen05_commit(
                            kv_empty.data_ptr() + c,
                            group=prims.CTAGroup.CTA_1,
                        )
            if prims.elect_sync():
                prims.tcgen05_commit(
                    s_full.data_ptr() + sst,
                    group=prims.CTAGroup.CTA_1,
                )
            kvph ^= cutlass.Int32(1)
            sst += cutlass.Int32(1)
            if sst == cutlass.Int32(ST_S):
                sst = cutlass.Int32(0)
                sph ^= cutlass.Int32(1)

    # =====================================================================
    # W11 — CTA1 SwapsAb VP; retain all four V128 stripes in TMEM
    # =====================================================================
    if warp_idx == W_MMA_PV:
        mma_heads: cutlass.Constexpr[int] = num_heads
        head_blocks: cutlass.Constexpr[int] = 1
        idesc_pv = prims.Tcgen05InstrDesc.build(
            c_dtype=ACC_DTYPE,
            a_dtype=IO_DTYPE,
            b_dtype=IO_DTYPE,
            n_dim=mma_heads,
            m_dim=PV_M,
            a_major=1,
            b_major=0,
        )
        if cutlass.const_expr(is_h8):
            dp_root_lo = _compat48.tcgen05_smem_desc(
                start_address=sP,
                leading_byte_offset=num_heads * QK_N * EB,
                stride_byte_offset=SBO_P,
                version=1,
                layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
            )
        else:
            dp_root_lo = _compat48.tcgen05_smem_desc(
                start_address=sP,
                leading_byte_offset=16,
                stride_byte_offset=SBO_P,
                layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
            )
        dp_root_hi = dp_root_lo
        if cutlass.const_expr(is_h32):
            dp_root_hi = _compat48.tcgen05_smem_desc(
                start_address=sP.data_ptr() + 16 * QK_N,
                leading_byte_offset=16,
                stride_byte_offset=SBO_P,
                layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
            )
        dv_root = _compat48.tcgen05_smem_desc(
            start_address=sKv,
            leading_byte_offset=LBO_V,
            stride_byte_offset=SBO_V,
            layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
        )
        kvph = cutlass.Int32(0)
        pst = cutlass.Int32(0)
        pph = cutlass.Int32(0)
        for kt in cutlass.range(0, num_kt, 1, unroll=kv_loop_unroll):
            while not prims.mbarrier_try_wait_parity(p_full.data_ptr() + pst, pph):
                pass
            oph = kt % cutlass.Int32(2)
            while not prims.mbarrier_try_wait_parity(o_empty, oph ^ cutlass.Int32(1)):
                pass
            pbase = pst * cutlass.Int32(QK_N * num_heads)
            for s in cutlass.range_constexpr(V_HDIM_STAGES):
                slot: cutlass.Constexpr[int] = K_HDIM_STAGES + s
                while not prims.mbarrier_try_wait_parity(kv_full.data_ptr() + slot, kvph):
                    pass
                vso: cutlass.Constexpr[int] = slot * QK_N * QK_K * EB
                for head_block in cutlass.range_constexpr(head_blocks):
                    otp = cutlass.inttoptr(
                        tmem_raw + tmem_o_offset + s * num_heads + head_block * mma_heads,
                        6,
                        cutlass.Float32,
                    )
                    dp_root = dp_root_lo
                    if cutlass.const_expr(head_block == 1):
                        dp_root = dp_root_hi
                    for kb in cutlass.range_constexpr(QK_N // GRAN):
                        po: cutlass.Constexpr[int] = kb * GRAN * EB
                        vo: cutlass.Constexpr[int] = kb * V_KSTEP
                        if prims.elect_sync():
                            prims.tcgen05_mma(
                                prims.Tcgen05MMAKind.F8F6F4,
                                prims.CTAGroup.CTA_1,
                                otp,
                                dv_root.advance_start_address(vso + vo),
                                dp_root.advance_start_address(pbase + po),
                                idesc_pv,
                                (kt > 0) or (kb != 0),
                            )
                if prims.elect_sync():
                    if kt + cutlass.Int32(1) < num_kt:
                        prims.tcgen05_commit(
                            kv_empty.data_ptr() + slot,
                            group=prims.CTAGroup.CTA_1,
                        )
            if prims.elect_sync():
                prims.tcgen05_commit(o_full, group=prims.CTAGroup.CTA_1)
            if prims.elect_sync():
                if kt + cutlass.Int32(ST_P) < num_kt:
                    prims.tcgen05_commit(
                        p_empty.data_ptr() + pst,
                        group=prims.CTAGroup.CTA_1,
                    )
            kvph ^= cutlass.Int32(1)
            pst += cutlass.Int32(1)
            if pst == cutlass.Int32(ST_P):
                pst = cutlass.Int32(0)
                pph ^= cutlass.Int32(1)

    # =====================================================================
    # W0-W3 — TRTLLM-style wide-tcgen05.ld online softmax warpgroup
    # =====================================================================
    if warp_idx < cutlass.Int32(4):
        prims.setmaxregister(152, prims.SetMaxRegisterAction.INCREASE)
        lane = tidx % cutlass.Int32(32)
        col_group = lane % cutlass.Int32(4)
        row_idx = lane // cutlass.Int32(4)
        local_row_idx = row_idx % cutlass.Int32(4)
        base = prims.TmemAddr(tmem_raw)
        row0 = base.row_id + warp_idx * cutlass.Int32(32)
        addr0 = (row0 << cutlass.Int32(16)) | base.col_id
        addr1 = ((row0 + cutlass.Int32(16)) << cutlass.Int32(16)) | base.col_id
        prev_max = cutlass.Vector.from_elements(
            tuple(ACC_DTYPE(-3.0e38) for _ in range(num_scale_groups)),
            dtype=ACC_DTYPE,
        )
        prev_sum = cutlass.Vector.from_elements(
            tuple(ACC_DTYPE(0.0) for _ in range(num_scale_groups)), dtype=ACC_DTYPE
        )
        stage = cutlass.Int32(0)
        phase = cutlass.Int32(0)

        for kt in cutlass.range(0, num_kt, 1, unroll=kv_loop_unroll):
            while not prims.mbarrier_try_wait_parity(s_full.data_ptr() + stage, phase):
                pass
            frag0 = prims.tcgen05_ld(
                "16x256b",
                cutlass.inttoptr(addr0 + stage * cutlass.Int32(num_heads), 6, ACC_DTYPE),
                num=q_repeats,
            )
            frag1 = prims.tcgen05_ld(
                "16x256b",
                cutlass.inttoptr(addr1 + stage * cutlass.Int32(num_heads), 6, ACC_DTYPE),
                num=q_repeats,
            )
            prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
            scores = cutlass.Vector.from_elements(
                tuple(frag0[i] for i in range(4 * q_repeats))
                + tuple(frag1[i] for i in range(4 * q_repeats)),
                dtype=ACC_DTYPE,
            )
            if cutlass.const_expr(needs_tail_mask):
                if cutlass.const_expr(is_h16):
                    # Keep H16 on its original helper signature. Passing the
                    # H8/H32-only column group through this hot path perturbs
                    # register allocation despite being constexpr-dead.
                    scores = _mask_scores_after_valid_prefix_h16(
                        scores,
                        (first_kv_tile + kt) * cutlass.Int32(QK_N),
                        valid_kv,
                        warp_idx,
                        row_idx,
                        num_heads,
                    )
                else:
                    scores = _mask_scores_after_valid_prefix(
                        scores,
                        (first_kv_tile + kt) * cutlass.Int32(QK_N),
                        valid_kv,
                        warp_idx,
                        row_idx,
                        col_group,
                        num_heads,
                    )

            if cutlass.const_expr(is_h8):
                local_max = cutlass.Vector.from_elements(
                    (
                        cute.math.max(
                            cute.math.max(scores[0], scores[2]),
                            cute.math.max(scores[4], scores[6]),
                        ),
                        cute.math.max(
                            cute.math.max(scores[1], scores[3]),
                            cute.math.max(scores[5], scores[7]),
                        ),
                    ),
                    dtype=ACC_DTYPE,
                )
                reduced_max = cutlass.Vector.from_elements(
                    (
                        _q8_group_max(local_max[0]),
                        _q8_group_max(local_max[1]),
                    ),
                    dtype=ACC_DTYPE,
                )
                if lane < cutlass.Int32(8):
                    max_slot = stage * cutlass.Int32(num_heads) + col_group * cutlass.Int32(2)
                    _atomic_fmax_cta(
                        softmax_max.data_ptr() + max_slot,
                        reduced_max[0],
                    )
                    _atomic_fmax_cta(
                        softmax_max.data_ptr() + max_slot + cutlass.Int32(1),
                        reduced_max[1],
                    )
            elif cutlass.const_expr(is_h16):
                thread_max = _fragment_max4(scores)
                reduced_max = cutlass.Vector.from_elements(
                    tuple(_warpgroup_row_max(thread_max[i]) for i in range(4)),
                    dtype=ACC_DTYPE,
                )
                atomic_value = reduced_max[0]
                if local_row_idx == cutlass.Int32(1):
                    atomic_value = reduced_max[1]
                elif local_row_idx == cutlass.Int32(2):
                    atomic_value = reduced_max[2]
                elif local_row_idx == cutlass.Int32(3):
                    atomic_value = reduced_max[3]
                atomic_value = _fmax(
                    atomic_value,
                    prims.shfl_sync(
                        thread_mask=0xFFFFFFFF,
                        val=atomic_value,
                        offset=16,
                        mask_and_clamp=0x1F,
                        kind=prims.Shfl.BFLY,
                    ),
                )
                if lane < cutlass.Int32(16):
                    max_head = col_group * cutlass.Int32(4) + local_row_idx
                    max_slot = (
                        stage * cutlass.Int32(4 * num_heads)
                        + warp_idx * cutlass.Int32(num_heads)
                        + max_head
                    )
                    softmax_max[max_slot] = atomic_value
            else:
                max_stage_idx = kt if defer_h32_sum else stage
                local_max = cutlass.Array(
                    ACC_DTYPE, num_scale_groups, space=cutlass.AddressSpace.rmem
                )
                for head_group in cutlass.range_constexpr(4):
                    for head_parity in cutlass.range_constexpr(2):
                        register_base: cutlass.Constexpr[int] = head_group * 4 + head_parity
                        local_max[head_group * 2 + head_parity] = cute.math.max(
                            cute.math.max(scores[register_base], scores[register_base + 2]),
                            cute.math.max(
                                scores[register_base + 16],
                                scores[register_base + 18],
                            ),
                        )
                reduced_max = cutlass.Vector.from_elements(
                    tuple(_h32_group_max(local_max[i]) for i in range(num_scale_groups)),
                    dtype=ACC_DTYPE,
                )
                if lane < cutlass.Int32(4):
                    max_slot = max_stage_idx * cutlass.Int32(num_heads) + col_group * cutlass.Int32(
                        num_scale_groups
                    )
                    for max_idx in cutlass.range_constexpr(num_scale_groups):
                        _atomic_max_u32_cta(
                            softmax_max_u32.data_ptr() + max_slot + max_idx,
                            _float_to_u32_for_atomic_max(reduced_max[max_idx]),
                        )
            prims.barrier_cta_sync(BAR_SM0, thread_count=128)
            if cutlass.const_expr(is_h32):
                encoded_max = (
                    softmax_max_u32.data_ptr()
                    + max_stage_idx * cutlass.Int32(num_heads)
                    + col_group * cutlass.Int32(num_scale_groups)
                ).load(count=num_scale_groups, alignment=scale_alignment)
                decoded_max = cutlass.Array(
                    ACC_DTYPE,
                    num_scale_groups,
                    space=cutlass.AddressSpace.rmem,
                )
                for max_idx in cutlass.range_constexpr(num_scale_groups):
                    decoded_max[max_idx] = _u32_to_float_for_atomic_max(encoded_max[max_idx])
                tile_max = cutlass.Vector.from_elements(
                    tuple(decoded_max[i] for i in range(num_scale_groups)),
                    dtype=ACC_DTYPE,
                )
            elif cutlass.const_expr(not is_h16):
                tile_max = (
                    softmax_max.data_ptr()
                    + stage * cutlass.Int32(num_heads)
                    + col_group * cutlass.Int32(num_scale_groups)
                ).load(count=num_scale_groups, alignment=scale_alignment)
            else:
                max_group = col_group * cutlass.Int32(num_scale_groups)
                max_stage = stage * cutlass.Int32(4 * num_heads)
                tile_max = (softmax_max.data_ptr() + max_stage + max_group).load(
                    count=num_scale_groups, alignment=scale_alignment
                )
                for max_warp in cutlass.range_constexpr(1, 4):
                    tile_max = cute.math.max(
                        tile_max,
                        (
                            softmax_max.data_ptr() + max_stage + max_warp * num_heads + max_group
                        ).load(count=num_scale_groups, alignment=scale_alignment),
                    )
            if (warp_idx == cutlass.Int32(0)) and (lane == cutlass.Int32(0)):
                prims.mbarrier_arrive(s_empty.data_ptr() + stage, relaxed=True)

            while not prims.mbarrier_try_wait_parity(
                c_empty.data_ptr() + stage, phase ^ cutlass.Int32(1)
            ):
                pass
            while not prims.mbarrier_try_wait_parity(
                p_empty.data_ptr() + stage, phase ^ cutlass.Int32(1)
            ):
                pass

            new_max = cute.math.max(prev_max, tile_max)
            corr = cute.math.exp2((prev_max - new_max) * scale_log2, fastmath=True)
            if cutlass.const_expr(is_h8):
                scaled = cutlass.Vector.from_elements(
                    tuple(
                        scores[i] * scale_log2
                        - new_max[i % 2] * scale_log2
                        + ACC_DTYPE(LOG2_FP8_P_SCALE)
                        for i in range(8)
                    ),
                    dtype=ACC_DTYPE,
                )
            else:
                scaled = _scale_scores_packed(scores, new_max, scale_log2, num_heads)
            probs = cute.math.exp2(scaled, fastmath=True)
            if cutlass.const_expr(is_h8):
                thread_sum = cutlass.Vector.from_elements(
                    (
                        probs[0] + probs[2] + probs[4] + probs[6],
                        probs[1] + probs[3] + probs[5] + probs[7],
                    ),
                    dtype=ACC_DTYPE,
                )
                warp_sum = cutlass.Vector.from_elements(
                    (
                        _q8_group_sum(thread_sum[0]),
                        _q8_group_sum(thread_sum[1]),
                    ),
                    dtype=ACC_DTYPE,
                )
            elif cutlass.const_expr(is_h16):
                thread_sum = _fragment_sum4(probs)
                warp_sum = cutlass.Vector.from_elements(
                    tuple(_warpgroup_row_sum(thread_sum[i]) for i in range(4)),
                    dtype=ACC_DTYPE,
                )
            else:
                local_sum = cutlass.Array(
                    ACC_DTYPE, num_scale_groups, space=cutlass.AddressSpace.rmem
                )
                for head_group in cutlass.range_constexpr(4):
                    register_base: cutlass.Constexpr[int] = head_group * 4
                    sum_lo = prims.add_packed_f32x2(
                        (probs[register_base], probs[register_base + 1]),
                        (probs[register_base + 2], probs[register_base + 3]),
                        rnd=prims.FPRoundingMode.RN,
                    )
                    sum_hi = prims.add_packed_f32x2(
                        (probs[register_base + 16], probs[register_base + 17]),
                        (probs[register_base + 18], probs[register_base + 19]),
                        rnd=prims.FPRoundingMode.RN,
                    )
                    sum_pair = prims.add_packed_f32x2(sum_lo, sum_hi, rnd=prims.FPRoundingMode.RN)
                    local_sum[head_group * 2] = sum_pair[0]
                    local_sum[head_group * 2 + 1] = sum_pair[1]
                thread_sum = cutlass.Vector.from_elements(
                    tuple(local_sum[i] for i in range(num_scale_groups)),
                    dtype=ACC_DTYPE,
                )
                warp_sum = thread_sum
                if cutlass.const_expr(defer_h32_sum):
                    prev_sum = prev_sum * corr + thread_sum
                    if kt + cutlass.Int32(1) == num_kt:
                        warp_sum = _warpgroup_row_sum8_packed(
                            prev_sum[0],
                            prev_sum[1],
                            prev_sum[2],
                            prev_sum[3],
                            prev_sum[4],
                            prev_sum[5],
                            prev_sum[6],
                            prev_sum[7],
                        )
                        if lane < cutlass.Int32(4):
                            (
                                softmax_sum.data_ptr()
                                + warp_idx * cutlass.Int32(num_heads)
                                + lane * cutlass.Int32(num_scale_groups)
                            ).store(warp_sum, alignment=scale_alignment)
                else:
                    warp_sum = _warpgroup_row_sum8_packed(
                        thread_sum[0],
                        thread_sum[1],
                        thread_sum[2],
                        thread_sum[3],
                        thread_sum[4],
                        thread_sum[5],
                        thread_sum[6],
                        thread_sum[7],
                    )
            if cutlass.const_expr(not defer_h32_sum):
                if lane < cutlass.Int32(4):
                    (
                        softmax_sum.data_ptr()
                        + warp_idx * cutlass.Int32(num_heads)
                        + lane * cutlass.Int32(num_scale_groups)
                    ).store(warp_sum, alignment=scale_alignment)
            if cutlass.const_expr(defer_h32_sum):
                if kt + cutlass.Int32(1) == num_kt:
                    prims.barrier_cta_sync(BAR_SM0, thread_count=128)
            else:
                # Reused max stages need all consumers to finish before reset.
                prims.barrier_cta_sync(BAR_SM0, thread_count=128)
            if cutlass.const_expr(not is_h16):
                if (warp_idx == cutlass.Int32(0)) and (lane < cutlass.Int32(num_heads)):
                    if cutlass.const_expr(is_h32 and not defer_h32_sum):
                        softmax_max_u32[stage * cutlass.Int32(num_heads) + lane] = (
                            _float_to_u32_for_atomic_max(ACC_DTYPE(-3.0e38))
                        )
                    elif cutlass.const_expr(not is_h32):
                        softmax_max[stage * cutlass.Int32(num_heads) + lane] = ACC_DTYPE(-3.0e38)
            sum_base = col_group * cutlass.Int32(num_scale_groups)
            if cutlass.const_expr(not defer_h32_sum):
                tile_sum = (softmax_sum.data_ptr() + sum_base).load(
                    count=num_scale_groups, alignment=scale_alignment
                )
                for sum_warp in cutlass.range_constexpr(1, 4):
                    tile_sum += (
                        softmax_sum.data_ptr() + sum_warp * cutlass.Int32(num_heads) + sum_base
                    ).load(count=num_scale_groups, alignment=scale_alignment)
                prev_sum = prev_sum * corr + tile_sum
            elif kt + cutlass.Int32(1) == num_kt:
                prev_sum = (softmax_sum.data_ptr() + sum_base).load(
                    count=num_scale_groups, alignment=scale_alignment
                )
                for sum_warp in cutlass.range_constexpr(1, 4):
                    prev_sum += (
                        softmax_sum.data_ptr() + sum_warp * cutlass.Int32(num_heads) + sum_base
                    ).load(count=num_scale_groups, alignment=scale_alignment)
            prev_max = new_max

            stat_base = stage * cutlass.Int32(num_heads * 3)
            head_base = lane * cutlass.Int32(num_scale_groups)

            if cutlass.const_expr(is_h8):
                packed_p = probs.to(IO_DTYPE).bitcast(cutlass.Int32)
                _store_transposed_p_q8(
                    sP,
                    stage,
                    packed_p[0],
                    packed_p[1],
                    tidx,
                    num_heads,
                )
            elif cutlass.const_expr(is_h16):
                smem_swizzle = cutlass.Swizzle.from_name("s128b")
                for half in cutlass.range_constexpr(2):
                    half_probs = cutlass.Vector.from_elements(
                        tuple(probs[half * 8 + i] for i in range(8)),
                        dtype=ACC_DTYPE,
                    )
                    packed_p = half_probs.to(IO_DTYPE).bitcast(cutlass.Int32)
                    address_row = lane % cutlass.Int32(8)
                    address_segment = (lane // cutlass.Int32(8)) % cutlass.Int32(2)
                    row_start = (
                        stage * cutlass.Int32(num_heads * QK_N)
                        + (address_segment * cutlass.Int32(8) + address_row) * cutlass.Int32(QK_N)
                        + (warp_idx * cutlass.Int32(2) + cutlass.Int32(half)) * cutlass.Int32(16)
                    )
                    prims.stmatrix(
                        (sP.data_ptr() + row_start).apply_swizzle(smem_swizzle),
                        [packed_p[0], packed_p[1]],
                        prims.MMALayout.COL,
                        shape=prims.StoreShape.M16N8,
                    )
            else:
                # One x4 store per 16-token half (registers 4*half..+3 are the
                # four 8-head repeats of that half); lane l addresses head row
                # l. The 128B swizzle is applied on the absolute SMEM address,
                # as the H16 store does: sP is only 128-byte aligned, so a
                # base-relative XOR permutes the token chunks of 7 of 8 rows.
                smem_swizzle = cutlass.Swizzle.from_name("s128b")
                packed_p = probs.to(IO_DTYPE).bitcast(cutlass.Int32)
                for half in cutlass.range_constexpr(2):
                    row_start = (
                        stage * cutlass.Int32(num_heads * QK_N)
                        + lane * cutlass.Int32(QK_N)
                        + (warp_idx * cutlass.Int32(2) + cutlass.Int32(half)) * cutlass.Int32(16)
                    )
                    prims.stmatrix(
                        (sP.data_ptr() + row_start).apply_swizzle(smem_swizzle),
                        [
                            packed_p[half * 4],
                            packed_p[half * 4 + 1],
                            packed_p[half * 4 + 2],
                            packed_p[half * 4 + 3],
                        ],
                        prims.MMALayout.COL,
                        shape=prims.StoreShape.M16N8,
                    )
            if cutlass.const_expr(is_h32):
                if kt + cutlass.Int32(1) == num_kt:
                    if (warp_idx == cutlass.Int32(0)) and (lane < cutlass.Int32(4)):
                        (compact_stats.data_ptr() + stat_base + head_base).store(
                            prev_sum, alignment=scale_alignment
                        )
                        (
                            compact_stats.data_ptr()
                            + stat_base
                            + cutlass.Int32(num_heads)
                            + head_base
                        ).store(prev_max, alignment=scale_alignment)
            prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
            if prims.elect_sync():
                prims.mbarrier_arrive(p_full.data_ptr() + stage)
            if cutlass.const_expr(is_h32):
                if (warp_idx == cutlass.Int32(0)) and (lane < cutlass.Int32(4)):
                    (
                        compact_stats.data_ptr()
                        + stat_base
                        + cutlass.Int32(2 * num_heads)
                        + head_base
                    ).store(corr, alignment=scale_alignment)
                if (warp_idx == cutlass.Int32(0)) and (lane == cutlass.Int32(0)):
                    prims.mbarrier_arrive(c_full.data_ptr() + stage)
            if cutlass.const_expr(not is_h32):
                if (warp_idx == cutlass.Int32(0)) and (lane < cutlass.Int32(4)):
                    (compact_stats.data_ptr() + stat_base + head_base).store(
                        prev_sum, alignment=scale_alignment
                    )
                    (
                        compact_stats.data_ptr() + stat_base + cutlass.Int32(num_heads) + head_base
                    ).store(prev_max, alignment=scale_alignment)
                    (
                        compact_stats.data_ptr()
                        + stat_base
                        + cutlass.Int32(2 * num_heads)
                        + head_base
                    ).store(corr, alignment=scale_alignment)
                if (warp_idx == cutlass.Int32(0)) and (lane == cutlass.Int32(0)):
                    prims.mbarrier_arrive(c_full.data_ptr() + stage)
            stage += cutlass.Int32(1)
            if stage == cutlass.Int32(ST_S):
                stage = cutlass.Int32(0)
                phase ^= cutlass.Int32(1)

    # =====================================================================
    # W4-7 — correction / rescale / epilogue
    # =====================================================================
    if (warp_idx >= cutlass.Int32(W_CORR0)) and (warp_idx < cutlass.Int32(W_CORR0 + 4)):
        prims.setmaxregister(176, prims.SetMaxRegisterAction.INCREASE)
        lane = tidx % cutlass.Int32(32)
        corr_warp = warp_idx - cutlass.Int32(W_CORR0)
        out_row = corr_warp * cutlass.Int32(32) + lane
        base = prims.TmemAddr(tmem_raw)
        my_addr = ((base.row_id + corr_warp * cutlass.Int32(32)) << 16) | base.col_id

        if cutlass.const_expr(is_h32):
            cst = cutlass.Int32(0)
            cph = cutlass.Int32(0)
            for kt in cutlass.range(0, num_kt, 1, unroll=kv_loop_unroll):
                while not prims.mbarrier_try_wait_parity(c_full.data_ptr() + cst, cph):
                    pass
                stat_base = cst * cutlass.Int32(num_heads * 3)
                corr_base = stat_base + cutlass.Int32(2 * num_heads)
                corr_lane = lane % cutlass.Int32(16)
                skip_corr_lo = prims.vote_sync(
                    0xFFFFFFFF,
                    compact_stats[corr_base + corr_lane] == ACC_DTYPE(1.0),
                    prims.VoteSync.ALL,
                )
                skip_corr_hi = prims.vote_sync(
                    0xFFFFFFFF,
                    compact_stats[corr_base + cutlass.Int32(16) + corr_lane] == ACC_DTYPE(1.0),
                    prims.VoteSync.ALL,
                )
                # Each logical 16-head PV block draws from both packed
                # statistics halves, so either non-identity half requires the
                # full inverse permutation.
                skip_corr = skip_corr_lo and skip_corr_hi
                packed_corr = cutlass.Array(ACC_DTYPE, num_heads, space=cutlass.AddressSpace.rmem)
                if not skip_corr:
                    loaded_corr_lo = (compact_stats.data_ptr() + corr_base).load(
                        count=16, alignment=16
                    )
                    loaded_corr_hi = (
                        compact_stats.data_ptr() + corr_base + cutlass.Int32(16)
                    ).load(count=16, alignment=16)
                    for corr_idx in cutlass.range_constexpr(16):
                        packed_corr[corr_idx] = loaded_corr_lo[corr_idx]
                        packed_corr[16 + corr_idx] = loaded_corr_hi[corr_idx]
                if prims.elect_sync():
                    prims.mbarrier_arrive(c_empty.data_ptr() + cst, relaxed=True)

                oph = kt % cutlass.Int32(2)
                if kt > cutlass.Int32(0):
                    while not prims.mbarrier_try_wait_parity(o_full, oph ^ cutlass.Int32(1)):
                        pass
                    for head_block in cutlass.range_constexpr(2):
                        if not skip_corr:
                            corr_block = cutlass.Vector.from_elements(
                                tuple(
                                    packed_corr[
                                        H32_LOGICAL_TO_PACKED_STATS[head_block * 16 + logical_head]
                                    ]
                                    for logical_head in range(16)
                                ),
                                dtype=ACC_DTYPE,
                            )
                            for s in cutlass.range_constexpr(N_V):
                                h32_corr_tmem_ptr = cutlass.inttoptr(
                                    my_addr + tmem_o_offset + s * num_heads + head_block * 16,
                                    6,
                                    ACC_DTYPE,
                                )
                                h32_corr_tile = prims.tcgen05_ld(
                                    "32x32b", h32_corr_tmem_ptr, num=16
                                )
                                prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                                prims.tcgen05_st(
                                    "32x32b",
                                    h32_corr_tmem_ptr,
                                    h32_corr_tile * corr_block,
                                )
                    prims.tcgen05_wait(prims.Tcgen05Wait.STORE)
                    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                    if prims.elect_sync():
                        prims.mbarrier_arrive(o_empty, relaxed=True)
                cst += cutlass.Int32(1)
                if cst == cutlass.Int32(ST_C):
                    cst = cutlass.Int32(0)
                    cph ^= cutlass.Int32(1)

            final_phase = (num_kt - cutlass.Int32(1)) % cutlass.Int32(2)
            while not prims.mbarrier_try_wait_parity(o_full, final_phase):
                pass
            final_stat_base = ((num_kt - cutlass.Int32(1)) % cutlass.Int32(ST_C)) * cutlass.Int32(
                num_heads * 3
            )
            packed_final_sum = (compact_stats.data_ptr() + final_stat_base).load(
                count=num_heads, alignment=16
            )
            packed_final_max = (compact_stats.data_ptr() + final_stat_base + num_heads).load(
                count=num_heads, alignment=16
            )
            logical_final_sum = cutlass.Vector.from_elements(
                tuple(packed_final_sum[packed_head] for packed_head in H32_LOGICAL_TO_PACKED_STATS),
                dtype=ACC_DTYPE,
            )
            logical_final_max = cutlass.Vector.from_elements(
                tuple(packed_final_max[packed_head] for packed_head in H32_LOGICAL_TO_PACKED_STATS),
                dtype=ACC_DTYPE,
            )
            for head_block in cutlass.range_constexpr(2):
                head_offset: cutlass.Constexpr[int] = head_block * 16
                final_sum = cutlass.Vector.from_elements(
                    tuple(logical_final_sum[head_offset + head] for head in range(16)),
                    dtype=ACC_DTYPE,
                )
                final_max = cutlass.Vector.from_elements(
                    tuple(logical_final_max[head_offset + head] for head in range(16)),
                    dtype=ACC_DTYPE,
                )
                if cutlass.const_expr(needs_tail_mask):
                    safe_sum = cutlass.Vector.from_elements(
                        tuple(_fmax(final_sum[head], ACC_DTYPE(1.0e-30)) for head in range(16)),
                        dtype=ACC_DTYPE,
                    )
                    inv = ACC_DTYPE(1.0) / safe_sum
                else:
                    inv = ACC_DTYPE(1.0) / final_sum
                for s in cutlass.range_constexpr(N_V):
                    h32_final_tmem_ptr = cutlass.inttoptr(
                        my_addr + tmem_o_offset + s * num_heads + head_offset,
                        6,
                        ACC_DTYPE,
                    )
                    h32_final_tile = prims.tcgen05_ld("32x32b", h32_final_tmem_ptr, num=16)
                    prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                    h32_final_tile = h32_final_tile * inv
                    out_col = s * PV_M + out_row
                    for head in cutlass.range_constexpr(16):
                        partial_o[(head_offset + head) * cutlass.Int32(L) + out_col] = OUT_DTYPE(
                            h32_final_tile[head]
                        )
                if (corr_warp == cutlass.Int32(0)) and (lane == cutlass.Int32(0)):
                    (partial_stats.data_ptr() + head_offset).store(final_max, alignment=16)
                    (partial_stats.data_ptr() + num_heads + head_offset).store(
                        final_sum, alignment=16
                    )
        else:
            r_sum = cutlass.Vector.from_elements(
                tuple(ACC_DTYPE(0.0) for _ in range(num_heads)), dtype=ACC_DTYPE
            )
            r_max = cutlass.Vector.from_elements(
                tuple(ACC_DTYPE(-3.0e38) for _ in range(num_heads)), dtype=ACC_DTYPE
            )
            cst = cutlass.Int32(0)
            cph = cutlass.Int32(0)
            for kt in cutlass.range(0, num_kt, 1, unroll=kv_loop_unroll):
                while not prims.mbarrier_try_wait_parity(c_full.data_ptr() + cst, cph):
                    pass
                stat_base = cst * cutlass.Int32(num_heads * 3)
                packed_sum = (compact_stats.data_ptr() + stat_base).load(
                    count=num_heads, alignment=16
                )
                if cutlass.const_expr(is_h16):
                    # compact_stats follows the 16x256b QK fragment layout,
                    # while PV TMEM columns and partial_stats use logical
                    # heads. Reorder all three online-softmax quantities:
                    # corr rescales prior PV tiles; max/sum feed split-K.
                    r_sum = cutlass.Vector.from_elements(
                        tuple(
                            packed_sum[packed_head] for packed_head in H16_LOGICAL_TO_PACKED_STATS
                        ),
                        dtype=ACC_DTYPE,
                    )
                else:
                    r_sum = packed_sum
                packed_max = (compact_stats.data_ptr() + stat_base + num_heads).load(
                    count=num_heads, alignment=16
                )
                if cutlass.const_expr(is_h16):
                    r_max = cutlass.Vector.from_elements(
                        tuple(
                            packed_max[packed_head] for packed_head in H16_LOGICAL_TO_PACKED_STATS
                        ),
                        dtype=ACC_DTYPE,
                    )
                else:
                    r_max = packed_max
                packed_corr = (compact_stats.data_ptr() + stat_base + 2 * num_heads).load(
                    count=num_heads, alignment=16
                )
                if cutlass.const_expr(is_h16):
                    corr = cutlass.Vector.from_elements(
                        tuple(
                            packed_corr[packed_head] for packed_head in H16_LOGICAL_TO_PACKED_STATS
                        ),
                        dtype=ACC_DTYPE,
                    )
                else:
                    corr = packed_corr
                if prims.elect_sync():
                    prims.mbarrier_arrive(c_empty.data_ptr() + cst, relaxed=True)

                oph = kt % cutlass.Int32(2)
                if kt > cutlass.Int32(0):
                    while not prims.mbarrier_try_wait_parity(o_full, oph ^ cutlass.Int32(1)):
                        pass
                    for s in cutlass.range_constexpr(N_V):
                        tp = cutlass.inttoptr(my_addr + tmem_o_offset + s * num_heads, 6, ACC_DTYPE)
                        tile_v = prims.tcgen05_ld("32x32b", tp, num=num_heads)
                        prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                        prims.tcgen05_st("32x32b", tp, tile_v * corr)
                    prims.tcgen05_wait(prims.Tcgen05Wait.STORE)
                    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                    if prims.elect_sync():
                        prims.mbarrier_arrive(o_empty, relaxed=True)
                cst += cutlass.Int32(1)
                if cst == cutlass.Int32(ST_C):
                    cst = cutlass.Int32(0)
                    cph ^= cutlass.Int32(1)

            finalize_tiles = cutlass.Int32(1)
            if cutlass.const_expr(empty_split_participants):
                finalize_tiles = num_kt
                if num_kt == cutlass.Int32(0):
                    # Empty split participant: PV never ran, so TMEM holds no
                    # accumulator and o_full never completes. Publish the
                    # neutral partial explicitly (O = 0, max = -3.0e38, sum = 0):
                    # its reducer weight exp2((-3.0e38 - m) * scale) is 0.
                    for s in cutlass.range_constexpr(N_V):
                        out_col = s * PV_M + out_row
                        for head in cutlass.range_constexpr(num_heads):
                            partial_o[head * cutlass.Int32(L) + out_col] = OUT_DTYPE(0.0)
                    if (corr_warp == cutlass.Int32(0)) and (lane < cutlass.Int32(num_heads)):
                        partial_stats[lane * cutlass.Int32(2)] = ACC_DTYPE(-3.0e38)
                        partial_stats[lane * cutlass.Int32(2) + cutlass.Int32(1)] = ACC_DTYPE(0.0)
            if finalize_tiles > cutlass.Int32(0):
                final_phase = (num_kt - cutlass.Int32(1)) % cutlass.Int32(2)
                while not prims.mbarrier_try_wait_parity(o_full, final_phase):
                    pass
                if cutlass.const_expr(needs_tail_mask):
                    safe_sum = cutlass.Vector.from_elements(
                        tuple(_fmax(r_sum[head], ACC_DTYPE(1.0e-30)) for head in range(num_heads)),
                        dtype=ACC_DTYPE,
                    )
                    inv = ACC_DTYPE(1.0) / safe_sum
                else:
                    inv = ACC_DTYPE(1.0) / r_sum
                for s in cutlass.range_constexpr(N_V):
                    tp = cutlass.inttoptr(my_addr + tmem_o_offset + s * num_heads, 6, ACC_DTYPE)
                    tile_v = prims.tcgen05_ld("32x32b", tp, num=num_heads)
                    prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                    tile_v = tile_v * inv
                    out_col = s * PV_M + out_row
                    for head in cutlass.range_constexpr(num_heads):
                        partial_o[head * cutlass.Int32(L) + out_col] = OUT_DTYPE(tile_v[head])
                if (corr_warp == cutlass.Int32(0)) and (lane < cutlass.Int32(num_heads)):
                    partial_stats[lane * cutlass.Int32(2)] = r_max[lane]
                    partial_stats[lane * cutlass.Int32(2) + cutlass.Int32(1)] = r_sum[lane]

    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    # For cluster reducers the TMEM dealloc and its full-CTA sync come after
    # the reducer; the reducer's cluster barrier already orders every
    # thread's partial_o / partial_stats stores before any peer read.
    if cutlass.const_expr(num_ctas_kv == 1 or use_gmem_reduction):
        prims.barrier_cta_sync(0)
        if warp_idx == W_MMA_PV:
            prims.tcgen05_dealloc(tmem_ptr, num_tmem_cols, group="cta_1")

    if cutlass.const_expr(use_gmem_reduction):
        _reduce_gmem_output(
            partial_o,
            partial_stats,
            partial_o_workspace,
            partial_stats_workspace,
            completion_counters,
            reduction_scratch,
            o,
            tidx,
            packed_q,
            cta,
            scale_log2,
            num_heads,
            num_ctas_kv,
        )
    elif cutlass.const_expr(num_ctas_kv == 1):
        _store_single_cta_output(partial_o, o, tidx, packed_q, num_heads)
    elif cutlass.const_expr(num_ctas_kv == 2):
        _reduce_two_cta_output(
            partial_o,
            partial_stats,
            o,
            tidx,
            packed_q,
            cta,
            scale_log2,
            num_heads,
        )
    else:
        _reduce_cga_output(
            partial_o,
            partial_stats,
            o,
            tidx,
            packed_q,
            cta,
            scale_log2,
            num_heads,
            num_ctas_kv,
        )
    # Every output-store thread publishes before this CTA releases dependents.
    # Cluster teardown follows the release so the next grid's independent
    # setup can overlap it while peer SMEM remains alive.
    prims.barrier_cta_sync(0)
    if cutlass.const_expr(num_ctas_kv > 1 and not use_gmem_reduction):
        if warp_idx == W_MMA_PV:
            prims.tcgen05_dealloc(tmem_ptr, num_tmem_cols, group="cta_1")
    if tidx == 0:
        prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)
    if cutlass.const_expr(num_ctas_kv > 1 and not use_gmem_reduction):
        prims.barrier_cluster_arrive()
        prims.barrier_cluster_wait()


@cute.jit
def _launch_fmha_sparse_mla(
    q_storage: cute.Tensor,
    kv_storage: cute.Tensor,
    sparse_indices: cute.Tensor,
    output: cute.Tensor,
    cu_q_seqlens: cute.Tensor,
    sequence_length: cute.Tensor,
    partial_o_workspace: cute.Tensor,
    partial_stats_workspace: cute.Tensor,
    completion_counters: cute.Tensor,
    layer_offset: cutlass.Int32,
    max_sq,
    stream: cuda_driver.CUstream,
    scale_log2: cutlass.Constexpr[float],
    num_heads: cutlass.Constexpr[int],
    all_sparse_indices_valid: cutlass.Constexpr[bool],
    num_ctas_kv: cutlass.Constexpr[int],
    use_gmem_reduction: cutlass.Constexpr[bool],
    add_layer_offset: cutlass.Constexpr[bool],
) -> None:
    def q_tensor_map(offset, inner, box_inner, swizzle):
        return cuda.create_tensor_map_tiled(
            global_address=(q_storage.iterator + offset).toint(),
            dtype=IO_DTYPE,
            global_dims=[inner, num_heads, q_storage.shape[0]],
            global_strides=[
                KV_STRIDE * EB // 16,
                num_heads * KV_STRIDE * EB // 16,
            ],
            box_dims=[box_inner, num_heads, 1],
            swizzle=swizzle,
        )

    def kv_tensor_map(offset, inner, box_inner, swizzle):
        return cuda.create_tensor_map_tiled(
            global_address=(kv_storage.iterator + offset).toint(),
            dtype=IO_DTYPE,
            global_dims=[inner, kv_storage.shape[0]],
            global_strides=[KV_STRIDE * EB // 16],
            box_dims=[box_inner, 1],
            swizzle=swizzle,
        )

    tma_ql = q_tensor_map(0, L, QK_K, cuda.TensorMapSwizzle.s128b)
    tma_qr = q_tensor_map(L, R, R, cuda.TensorMapSwizzle.s64b)
    tma_kv = kv_tensor_map(0, KV_STRIDE, QK_K, cuda.TensorMapSwizzle.s128b)
    tma_kr = kv_tensor_map(L, R, R, cuda.TensorMapSwizzle.s64b)
    _fmha_sparse_mla_kernel(
        tma_ql,
        tma_qr,
        tma_kv,
        tma_kr,
        kv_storage,
        sparse_indices,
        output,
        cu_q_seqlens,
        sequence_length,
        partial_o_workspace,
        partial_stats_workspace,
        completion_counters,
        layer_offset,
        cutlass.Float32(scale_log2),
        num_heads,
        all_sparse_indices_valid,
        num_ctas_kv,
        use_gmem_reduction,
        add_layer_offset,
    ).launch(
        grid=(num_ctas_kv * max_sq, 1, cu_q_seqlens.shape[0] - 1),
        block=(THREADS, 1, 1),
        cluster=(1 if use_gmem_reduction else num_ctas_kv, 1, 1),
        stream=stream,
        min_blocks_per_mp=1,
        use_pdl=True,
    )


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
    return cuda_driver.CUstream(torch.cuda.current_stream(device_index).cuda_stream)


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


def _compile_for(
    specialization: MlaFmhaSpecialization,
    device_index: int,
) -> object:
    specialization.validate()
    key = (
        device_index,
        specialization.num_heads,
        specialization.all_sparse_indices_valid,
        specialization.q_scaling,
        specialization.num_ctas_kv,
        specialization.use_gmem_reduction,
        specialization.index_kind,
    )
    cached = _COMPILE_CACHE.get(key)
    if cached is not None:
        return cached

    cutlass.cuda.initialize_cuda_context(device_index)
    compiled = cute.compile(
        _launch_fmha_sparse_mla,
        *_fake_tensors(
            specialization.num_heads,
            specialization.num_ctas_kv,
            specialization.use_gmem_reduction,
        ),
        cutlass.Int32(0),
        cute.sym_int32(),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        specialization.scale_log2,
        specialization.num_heads,
        specialization.all_sparse_indices_valid,
        specialization.num_ctas_kv,
        specialization.use_gmem_reduction,
        specialization.add_layer_offset,
        options=_compile_options(),
    )
    _COMPILE_CACHE[key] = compiled
    return compiled


def compile(  # noqa: A001
    *,
    all_sparse_indices_valid: bool = True,
    num_heads: int = NUM_HEADS,
    q_scaling: float = 1.0,
    num_ctas_kv: int = 2,
    use_gmem_reduction: bool = False,
    device_index: int = 0,
    index_kind: str = INDEX_KIND_GLOBAL,
) -> object:
    """Compile one production specialization with symbolic runtime shapes."""
    return _compile_for(
        MlaFmhaSpecialization(
            all_sparse_indices_valid=all_sparse_indices_valid,
            num_heads=num_heads,
            q_scaling=q_scaling,
            num_ctas_kv=num_ctas_kv,
            use_gmem_reduction=use_gmem_reduction,
            index_kind=index_kind,
        ),
        device_index,
    )


def precompile_for_serving(
    *,
    max_batch_size: int,
    all_sparse_indices_valid: bool,
    num_heads: int = NUM_HEADS,
    q_scaling: float = 1.0,
    max_sq: int = DEFAULT_MAX_SQ,
    max_seq_len_kv: int = TOPK,
    device_index: int = 0,
    index_kind: str = INDEX_KIND_GLOBAL,
) -> tuple[MlaFmhaSpecialization, ...]:
    """Compile every split-K variant reachable by a dynamic serving batch.

    Call this during model/CUDA-graph warmup, before benchmark timing or live
    requests. Runtime batch sizes from one through ``max_batch_size`` are
    reduced to unique ``(num_ctas_kv, use_gmem_reduction)`` specializations;
    symbolic tensor shapes and addresses remain runtime launch arguments.
    ``index_kind`` selects the sparse-index interpretation the serving route
    will pass to the prepared runner (callers that pass layer-independent base rows use
    BASE_WITH_LAYER_OFFSET).
    """
    if max_batch_size <= 0:
        raise ValueError(f"max_batch_size must be positive, got {max_batch_size}")
    if max_sq not in SUPPORTED_MAX_SQ:
        raise ValueError(f"max_sq must be one of {SUPPORTED_MAX_SQ}, got {max_sq}")
    if max_seq_len_kv <= 0:
        raise ValueError(f"max_seq_len_kv must be positive, got {max_seq_len_kv}")

    multiprocessor_count = torch.cuda.get_device_properties(device_index).multi_processor_count
    specializations: list[MlaFmhaSpecialization] = []
    for batch_size in range(1, max_batch_size + 1):
        num_ctas_kv = _select_num_ctas_kv(
            batch_size=batch_size,
            max_sq=max_sq,
            max_seq_len_kv=max_seq_len_kv,
            multiprocessor_count=multiprocessor_count,
        )
        use_gmem_reduction = num_ctas_kv > 1 and _select_gmem_reduction(
            batch_size=batch_size,
            max_sq=max_sq,
            max_active_clusters=_max_active_clusters(device_index, num_ctas_kv),
        )
        specialization = MlaFmhaSpecialization(
            all_sparse_indices_valid=all_sparse_indices_valid,
            num_heads=num_heads,
            q_scaling=q_scaling,
            num_ctas_kv=num_ctas_kv,
            use_gmem_reduction=use_gmem_reduction,
            index_kind=index_kind,
        )
        specialization.validate()
        if specialization not in specializations:
            specializations.append(specialization)

    for specialization in specializations:
        _compile_for(specialization, device_index)
    return tuple(specializations)


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


def prepare_runner(
    quant_q_buffer: torch.Tensor,
    kv_pool_view: torch.Tensor,
    sparse_attn_indices: torch.Tensor,
    output: torch.Tensor,
    cu_q_seqlens: torch.Tensor,
    sequence_length: torch.Tensor,
    *,
    mla_bmm1_scale: torch.Tensor | None = None,
    mla_bmm2_scale: torch.Tensor | None = None,
    host_past_key_value_lengths: torch.Tensor | None = None,
    all_sparse_indices_valid: bool,
    q_scaling: float = 1.0,
    max_sq: int = DEFAULT_MAX_SQ,
    max_seq_len_kv: int | None = None,
    num_ctas_kv: int | None = None,
    use_gmem_reduction: bool | None = None,
    refresh_stream: bool = False,
    index_kind: str = INDEX_KIND_GLOBAL,
    local_layer_idx: int | None = None,
) -> PreparedMlaFmhaRunner:
    """Prepare the production direct-ABI launch.

    sparse_attn_indices contains global physical pool rows
    (``index_kind=INDEX_KIND_GLOBAL``, the default) or, with
    ``index_kind=INDEX_KIND_BASE_WITH_LAYER_OFFSET`` and a nonnegative
    ``local_layer_idx``, the layer-independent base map of
    ``convert_req_index_to_base_rawptr`` to which the kernel adds
    ``local_layer_idx * KV_POOL_TOKENS_PER_BLOCK``. ``local_layer_idx``
    is rejected with the GLOBAL kind so a caller cannot pass base rows to the
    unchanged kernel by mistake.
    cu_q_seqlens follows TRT-LLM's MLA-generation convention and contains
    cumulative flattened query-head rows, not cumulative token rows.
    max_sq is the scheduler-provided upper bound for every request's query
    length and must be one of SUPPORTED_MAX_SQ. Individual requests may still
    have Sq=0. Only the aggregate total_q <= batch_size * max_sq relation can
    be checked here without synchronizing to read cu_q_seqlens.
    max_seq_len_kv is host metadata used only by TRT-LLM's split-K heuristic;
    when omitted it comes from host_past_key_value_lengths or defaults to TOPK.
    all_sparse_indices_valid is a scheduler/indexer contract, not inferred by
    synchronizing and scanning the CUDA index tensor. Scale tensors are only
    shape-checked because the preceding fixed GLM RoPE specialization writes
    their known values asynchronously.
    """
    if index_kind not in INDEX_KINDS:
        raise ValueError(f"index_kind must be one of {INDEX_KINDS}, got {index_kind!r}")
    if index_kind == INDEX_KIND_BASE_WITH_LAYER_OFFSET:
        # ``int()`` would truncate 2.5 -> 2 and silently address the wrong
        # KV rows; require an integral value (operator.index rejects floats).
        try:
            local_layer_idx_int = operator.index(local_layer_idx)
        except TypeError as exc:
            raise ValueError(
                "index_kind=BASE_WITH_LAYER_OFFSET requires a nonnegative "
                f"integral local_layer_idx, got {local_layer_idx!r}"
            ) from exc
        if local_layer_idx_int < 0:
            raise ValueError(
                "index_kind=BASE_WITH_LAYER_OFFSET requires a nonnegative "
                f"integral local_layer_idx, got {local_layer_idx!r}"
            )
        layer_offset = local_layer_idx_int * KV_POOL_TOKENS_PER_BLOCK
    else:
        if local_layer_idx is not None:
            raise ValueError(
                "local_layer_idx is only accepted with "
                "index_kind=BASE_WITH_LAYER_OFFSET (global rows carry the layer)"
            )
        layer_offset = 0
    if quant_q_buffer.device.type != "cuda":
        raise ValueError("quant_q_buffer must live on cuda")
    if not quant_q_buffer.is_contiguous():
        raise ValueError("quant_q_buffer must be contiguous")
    if not kv_pool_view.is_contiguous():
        raise ValueError("kv_pool_view must be contiguous")
    q_storage = _as_uint8_storage(quant_q_buffer, name="quant_q_buffer")
    kv_storage = _as_uint8_storage(kv_pool_view, name="kv_pool_view")

    if q_storage.ndim != 3 or int(q_storage.shape[2]) != KV_STRIDE:
        raise ValueError(
            "quant_q_buffer must have shape "
            f"[total_q, H, {KV_STRIDE}], got {tuple(q_storage.shape)}"
        )
    num_heads = int(q_storage.shape[1])
    if num_heads not in SUPPORTED_NUM_HEADS:
        raise ValueError(f"quant_q_buffer H must be one of {SUPPORTED_NUM_HEADS}, got {num_heads}")
    if kv_storage.numel() % KV_STRIDE:
        raise ValueError(
            f"kv_pool_view byte count must be divisible by {KV_STRIDE}, got {kv_storage.numel()}"
        )

    total_q = int(q_storage.shape[0])
    if sequence_length.ndim != 1:
        raise ValueError(f"sequence_length must be rank 1, got {tuple(sequence_length.shape)}")
    batch_size = int(sequence_length.shape[0])
    pool_rows = kv_storage.numel() // KV_STRIDE
    runtime_shape = MlaFmhaRuntimeShape(total_q, batch_size, pool_rows, max_sq)
    runtime_shape.validate()

    _require_tensor(
        sparse_attn_indices,
        name="sparse_attn_indices",
        dtype=torch.int32,
        shape=(total_q, TOPK),
    )
    _require_tensor(
        cu_q_seqlens,
        name="cu_q_seqlens",
        dtype=torch.int32,
        shape=(batch_size + 1,),
    )
    _require_tensor(
        sequence_length,
        name="sequence_length",
        dtype=torch.int32,
        shape=(batch_size,),
    )
    if output.dtype != torch.bfloat16 or not output.is_contiguous():
        raise TypeError("output must be contiguous torch.bfloat16")
    expected_output_elements = total_q * num_heads * L
    if output.numel() != expected_output_elements:
        raise ValueError(
            f"output must contain {expected_output_elements} BF16 elements, got {output.numel()}"
        )

    device = quant_q_buffer.device
    _require_same_device(
        (
            ("kv_pool_view", kv_pool_view),
            ("sparse_attn_indices", sparse_attn_indices),
            ("output", output),
            ("cu_q_seqlens", cu_q_seqlens),
            ("sequence_length", sequence_length),
        ),
        device,
    )
    for name, tensor in (
        ("quant_q_buffer", q_storage),
        ("kv_pool_view", kv_storage),
        ("sparse_attn_indices", sparse_attn_indices),
        ("output", output),
    ):
        if tensor.data_ptr() % 16:
            raise ValueError(f"{name} must be at least 16-byte aligned")

    _validate_scale_metadata(
        mla_bmm1_scale,
        name="mla_bmm1_scale",
        shape=(2,),
        device=device,
    )
    _validate_scale_metadata(
        mla_bmm2_scale,
        name="mla_bmm2_scale",
        shape=(1,),
        device=device,
    )
    if host_past_key_value_lengths is not None:
        _require_tensor(
            host_past_key_value_lengths,
            name="host_past_key_value_lengths",
            dtype=torch.int32,
            shape=(batch_size,),
            device_type="cpu",
        )
        if all_sparse_indices_valid and int(host_past_key_value_lengths.min().item()) < TOPK:
            raise ValueError(
                f"all-valid specialization requires every host KV length to be at least TOPK={TOPK}"
            )
        if max_seq_len_kv is None:
            max_seq_len_kv = int(host_past_key_value_lengths.max().item())

    if max_seq_len_kv is None:
        max_seq_len_kv = TOPK
    if max_seq_len_kv <= 0:
        raise ValueError(f"max_seq_len_kv must be positive, got {max_seq_len_kv}")

    device_index = _device_index(quant_q_buffer)
    if num_ctas_kv is None:
        multiprocessor_count = torch.cuda.get_device_properties(device_index).multi_processor_count
        num_ctas_kv = _select_num_ctas_kv(
            batch_size=batch_size,
            max_sq=max_sq,
            max_seq_len_kv=max_seq_len_kv,
            multiprocessor_count=multiprocessor_count,
        )
    if not 1 <= num_ctas_kv <= MAX_CTAS_KV:
        raise ValueError(f"num_ctas_kv must be in [1, {MAX_CTAS_KV}], got {num_ctas_kv}")
    if use_gmem_reduction is None:
        use_gmem_reduction = num_ctas_kv > 1 and _select_gmem_reduction(
            batch_size=batch_size,
            max_sq=max_sq,
            max_active_clusters=_max_active_clusters(device_index, num_ctas_kv),
        )
    specialization = MlaFmhaSpecialization(
        all_sparse_indices_valid=all_sparse_indices_valid,
        num_heads=num_heads,
        q_scaling=q_scaling,
        num_ctas_kv=num_ctas_kv,
        use_gmem_reduction=use_gmem_reduction,
        index_kind=index_kind,
    )
    compiled = _compile_for(specialization, device_index)
    if use_gmem_reduction:
        partial_o_workspace = torch.empty(
            total_q,
            num_ctas_kv,
            num_heads,
            L,
            dtype=torch.bfloat16,
            device=device,
        )
        partial_stats_workspace = torch.empty(
            total_q,
            num_ctas_kv,
            num_heads,
            2,
            dtype=torch.float32,
            device=device,
        )
        completion_counters = torch.zeros(
            total_q,
            dtype=torch.int32,
            device=device,
        )
    else:
        # These constexpr-dead parameters retain one launch ABI across modes.
        partial_o_workspace = output.reshape(-1)[:1]
        partial_stats_workspace = sparse_attn_indices.view(torch.float32).reshape(-1)[:1]
        completion_counters = cu_q_seqlens[:1]
    stream = None if refresh_stream else _current_stream(device_index)
    return PreparedMlaFmhaRunner(
        compiled=compiled,
        args=(
            q_storage,
            kv_storage.reshape(pool_rows, KV_STRIDE),
            sparse_attn_indices,
            output.reshape(total_q, num_heads, L),
            cu_q_seqlens,
            sequence_length,
            partial_o_workspace,
            partial_stats_workspace,
            completion_counters,
            layer_offset,
            max_sq,
        ),
        device_index=device_index,
        stream=stream,
        runtime_shape=runtime_shape,
        specialization=specialization,
    )
