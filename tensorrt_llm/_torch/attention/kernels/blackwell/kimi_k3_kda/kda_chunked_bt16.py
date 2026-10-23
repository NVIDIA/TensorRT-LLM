# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""KDA chunked BT=16 forward kernel using direct CUTLASS primitives.

The current body implements the chunk-size 16 KDA schedule:

  load q/k/v/gate
  L2-normalize q/k
  exp2(g), exp2(-g), stage final-token exp2(g) as exp2(g_last)
  auxiliary MMA warp: kk/qk/blockwise inverse + apply beta
  tcgen05-MMA: state*k/state*q/new-v/kv-update/qkv
  store o

Inputs use a packed variable-length ABI: q/k/v/gate/BF16 beta logits
have a singleton batch dimension and `cu_seqlens` partitions their token
dimension. Initial and final states retain one batch entry per logical sequence
and may use BF16 or FP32 storage. State presence (initial / final / periodic
checkpoints) is a compile-time specialization, but the returned callable
DERIVES it per call from which state tensors are actually passed (`None` vs a
real tensor) and transparently uses the matching build -- the `has_state_*`
compile flags only choose which build is compiled eagerly. The gate input dtype
is a compile-time parameter `gate_dtype`; it DEFAULTS to BF16, matching
FlashKDA's published ABI ("g: Gate before activation, bf16, shape [B,T,H,K]"),
so every tensor of the default ABI agrees with FlashKDA (q/k/v/beta/out BF16,
A_log/dt_bias FP32).  FP32 gates remain fully supported by passing
`gate_dtype=cutlass.Float32`; that build is bit-for-bit the historical kernel.
Only the MEMORY FORMAT of the gate changes: a 16-bit raw logit is widened to
FP32 on the SMEM read and the whole activation/prefix/exp2 chain stays FP32.
The real branch point is 32-bit vs 16-bit, not one dtype per path: under FP32
the `exp2(g_prefix)` exchange tile stays fused inside the raw-gate SMEM ring
(which is FP32 and therefore big enough), while a 16-bit gate shrinks that ring
to a 16-bit landing buffer and gives the FP32 exchange its own 4-deep ring
(`gate_exchange_smem`).  Scale is a runtime FP32 argument. The
torch reference below is the numerical contract used by the verify path.

On top of the plain engine kernel this file adds a two-kernel decomposition of
the same math.  A chunk-parallel factor-prep kernel (kernel 1) computes the
rank-16 chunk factors from q/k/gate/beta and writes them, pre-permuted, to a
user-allocated workspace; a serial chain kernel (kernel 2, DV2 split: two CTAs
per sequence-head, each owning half the value dimension) runs the state
recursion, reading those factors by TMA.  Decomp always runs PREP-FIRST on one
stream -- k1 completes its full grid, then k2 reads a workspace k1 has fully
written, so the kernel boundary is the only synchronization.  A single host
`compile()` returns one callable that routes the plain engine or the
decomposition per launch shape by one occupancy rule (decomp iff its doubled k2
grid fits one wave: `n_seq * heads * 2 <= sm_count`); the decomp route reads an
opaque workspace whose byte size is queried with `workspace_size()` (0 for the
engine route).  All device kernels keep the packed varlen ABI.

Optimization notes (each expanded at its definition site):

* Beta TMA transport -- token-major beta makes the per-chunk beta read a
  strided gather (16 lanes at `heads * 2B`; up to 16 distinct 32B sectors where
  a head-major layout touches 1, ~1us per extra sector at H96).  A 2D TMA
  descriptor over the same memory moves the gather onto the TMA engine, where
  it is free; `g = 8 / gcd(heads, 8)` pair-packs rows so the descriptor stride
  stays a legal 16B multiple for EVERY head count (H12 = the TP8 shard of H96
  uses g=2).  See the constants block above `tma_stage_load_inputs`.
* Prep-first decomposition -- a co-resident overlap of k1/k2 was measured and
  removed: it forces a per-chunk flag/fence ring onto both kernels and caps
  k1's grid to the leftover SMs, which together cost more than the concurrency
  returns.  Sequential prep runs its full grid with zero cross-kernel
  synchronization.  See the route rule in `host_unified`.
* K1_CPC = 4 -- each prep CTA walks four contiguous chunks (measured ladder);
  amortizes launch/preamble and enables mid-chunk prefetch of the next chunk's
  raw tiles.
* Recurrent state lives in TMEM as an fp32 accumulator across the whole chunk
  walk; the final store, the optional periodic checkpoints, and the DV2 halves
  all drain the same accumulator (`tcgen05_store_final_state_tmem`).

State checkpoints (opt-in): `compile(has_state_ckpt=True)` returns a callable
taking `state_ckpt`, `checkpoint_cu_starts`, and `ckpt_interval` (a positive
multiple of BT).  Without `checkpoint_state_indices`, each sequence stores its
initial state first, followed by states at interior interval boundaries.  With
indices, completed boundaries (including an aligned final boundary, excluding
the initial state) are written directly to the selected rows of `state_ckpt`.

Reproducing the results::

    # single-shape correctness vs the torch MMA reference below
    python kda_chunked_bt16.py --batch 1 --heads 32 --seqlen 8192

    # timing protocol used for the reported numbers: engine route =
    # CUPTI/Kineto device-time sum; decomp route = back-to-back wall clock
    # with one synchronize at the end (per-launch syncs both add ~10us of
    # launch overhead per sample and serialize the two kernels' tail).
"""

import argparse
import weakref
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Protocol, cast

import torch
import cuda.bindings.driver as cuda_driver

import cutlass
import cutlass.experimental.cuda as cuda
import cutlass.experimental.primitives as prims
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream


LOG2_E: float = 1.4426950408889634
DEFAULT_GATE_LOWER_BOUND: float = -5.0


class CompiledKDA(Protocol):
    """Callable returned by ``compile`` with its workspace query attached."""

    workspace_size: Callable[..., int]
    workspace_size_from_total_chunks: Callable[..., int]

    def __call__(self, *args, **kwargs) -> None: ...


def packed_f32x2_binary(
    op: Callable,
    lhs: tuple[cutlass.Float32, cutlass.Float32],
    rhs: tuple[cutlass.Float32, cutlass.Float32],
) -> tuple[cutlass.Float32, cutlass.Float32]:
    """Apply a CUTLASS packed-FP32 primitive to two scalar pairs."""

    lhs_vec = cutlass.Vector.from_elements(lhs, cutlass.Float32)
    rhs_vec = cutlass.Vector.from_elements(rhs, cutlass.Float32)
    result = op(lhs_vec, rhs_vec, ftz=False, rnd="rn")
    return cutlass.Float32(result[0]), cutlass.Float32(result[1])


def fadd2(lhs, rhs):
    return packed_f32x2_binary(prims.add_packed_f32x2, lhs, rhs)


def fmul2(lhs, rhs):
    return packed_f32x2_binary(prims.mul_packed_f32x2, lhs, rhs)


@cute.jit
def movmatrix_b16(value: cutlass.Int32) -> cutlass.Int32:
    """Transpose one packed m8n8 b16 register fragment."""

    return prims.inline_ptx_hl(
        "movmatrix.sync.aligned.m8n8.trans.b16 {$w0}, {$r0};",
        write_only_types=[cutlass.Int32],
        read_only_args=[value],
    )


@cute.jit
def mul_f16x2(value: cutlass.Int32, scale: cutlass.Int32) -> cutlass.Int32:
    """Multiply two packed FP16 pairs."""

    return prims.inline_ptx_hl(
        "mul.f16x2 {$w0}, {$r0}, {$r1};",
        write_only_types=[cutlass.Int32],
        read_only_args=[value, scale],
    )


@cute.jit
def sub_b16x2_input_dtype(
    lhs: cutlass.Int32,
    rhs: cutlass.Int32,
    input_dtype: cutlass.Constexpr,
) -> cutlass.Int32:
    """Subtract two packed pairs using the compile-time input dtype."""

    if cutlass.const_expr(input_dtype is cutlass.BFloat16):
        return prims.inline_ptx_hl(
            "sub.bf16x2 {$w0}, {$r0}, {$r1};",
            write_only_types=[cutlass.Int32],
            read_only_args=[lhs, rhs],
        )
    return prims.inline_ptx_hl(
        "sub.f16x2 {$w0}, {$r0}, {$r1};",
        write_only_types=[cutlass.Int32],
        read_only_args=[lhs, rhs],
    )


@cute.jit
def mul_b16x2_input_dtype(
    lhs: cutlass.Int32,
    rhs: cutlass.Int32,
    input_dtype: cutlass.Constexpr,
) -> cutlass.Int32:
    """Multiply two packed pairs using the compile-time input dtype."""

    if cutlass.const_expr(input_dtype is cutlass.BFloat16):
        return prims.mul_bf16x2(lhs, rhs)
    return mul_f16x2(lhs, rhs)


@cute.jit
def safe_gate_log2_increment_prehalved(
    half_scaled_gate: cutlass.Float32,
    SAFE_GATE: cutlass.Constexpr,
    GATE_SCALE_LOG2: cutlass.Constexpr,
) -> cutlass.Float32:
    """Apply the safe-gate transform after folding its exact half scales."""

    if cutlass.const_expr(SAFE_GATE):
        half_scale = cutlass.Float32(GATE_SCALE_LOG2 * 0.5)
        tanh_value = cute.math.tanh(half_scaled_gate, approx=True)
        return tanh_value * half_scale + half_scale
    return half_scaled_gate


@cute.jit
def softplus_log2_f32(value: cutlass.Float32) -> cutlass.Float32:
    """Compute ``softplus(value) * log2(e)`` with predicated MUFU ops.

    Evaluates the FLA non-safe activation directly in the log2 domain so the
    prefix scan never leaves base-2 units.  Matches Triton's ``x > 20``
    overflow guard: above the threshold ``softplus(x) ~= x`` and the result is
    just ``x * log2(e)`` (the ``@!p`` ops are skipped).
    """

    return prims.inline_ptx_hl(
        """
        {
            .reg .pred p;
            setp.gt.f32 p, {$r0}, 20.0;
            mul.f32 {$w0}, {$r0}, 1.4426950408889634;
            @!p ex2.approx.ftz.f32 {$w0}, {$w0};
            @!p add.f32 {$w0}, {$w0}, 1.0;
            @!p lg2.approx.ftz.f32 {$w0}, {$w0};
        }
        """,
        write_only_types=[cutlass.Float32],
        read_only_args=[value],
    )


@cute.jit
def pack_input_b16x2_to_i32(
    value0: cutlass.Float32,
    value1: cutlass.Float32,
    input_dtype: cutlass.Constexpr,
):
    """Pack two FP32 values through the compile-time input 16-bit dtype."""

    return (
        cutlass.Vector.from_elements(
            (value0, value1),
            cutlass.Float32,
        )
        .to(input_dtype)
        .bitcast(cutlass.Int32)[0]
    )


# ---------------------------------------------------------------------------
# Fixed KDA BT=16 shape
# ---------------------------------------------------------------------------

BT: int = 16
DK: int = 128
DV: int = 128
DEFAULT_SEQLEN: int = 8192
DEFAULT_BATCH: int = 1
DEFAULT_HEADS: int = 32
DEFAULT_SCALE: float = 1.0 / (DK**0.5)
L2_NORM_EPS: float = 1.0e-12
VERIFY_RTOL: float = 1.0e-2
VERIFY_ATOL: float = 1.0e-3

THREADS_PER_WARP: int = 32
THREADS_PER_CTA: int = 16 * THREADS_PER_WARP
CG0_GROUP_COUNT: int = 2
CG0_WARPS_PER_GROUP: int = 4
CG0_THREADS_PER_GROUP: int = CG0_WARPS_PER_GROUP * THREADS_PER_WARP
NBAR_CG0_GROUP0_ID: int = 1
TMEM_USER_WARP_COUNT: int = 5
TMEM_USER_THREADS: int = TMEM_USER_WARP_COUNT * THREADS_PER_WARP
NBAR_TMEM_LIFECYCLE_ID: int = 2
NBAR_CG0_GROUP1_ID: int = 3
# Third CG0 producer group (kernel 1 runs warps 8-11 as group 2).
NBAR_CG0_GROUP2_ID: int = 4
KDA_CG0_REGS: int = 160
KDA_CG1_REGS: int = 136
KDA_SERVICE_REGS: int = 56
TCGEN05_VALID_ALLOC_COLS: tuple[int, ...] = (32, 64, 128, 256, 512)
TCGEN05_F16_K_ATOM: int = 16
TCGEN05_F16_ELEM_BYTES: int = 2
TCGEN05_F16_A_TMEM_PAIR_XOR: int = 4
TCGEN05_SW128_BYTES: int = 128
TCGEN05_SW128_K_PHASES_PER_SLICE: int = 4
# `state` and decay operands use K-box-major SW128 staging.  K-major F16 SMEM
# descriptors use a 16B leading offset and 1024B stride; the decay store applies
# a row-group key xor so tcgen05 B reads logical `[DK, BT]` correctly.
TCGEN05_STATE_K_B_LEADING_BYTES: int = 16
TCGEN05_STATE_K_B_STRIDE_BYTES: int = 1024
TCGEN05_STATE_K_B_K_STEP_BYTES: int = TCGEN05_F16_K_ATOM * TCGEN05_F16_ELEM_BYTES
TCGEN05_SW32_BT_HALF_XOR: int = BT // 2
PAIRWISE_SW32_ROW_STRIDE: int = BT
PAIRWISE_SW32_COL_XOR: int = TCGEN05_SW32_BT_HALF_XOR
PAIRWISE_SW32_TILE_ELEMS: int = BT * BT
TCGEN05_TMEM_LOAD_COLS: int = BT
TCGEN05_STATE_INPUT_LOAD_COLS: int = 16
TCGEN05_STATE_INPUT_PACKED_COLS: int = TCGEN05_STATE_INPUT_LOAD_COLS // 2
TCGEN05_STATE_K_TMEM_ROW_BLOCKS: int = DV // THREADS_PER_WARP


def _tcgen05_accumulator_tmem_cols(n_dim: int) -> int:
    """Return TMEM columns for an FP32 `[128, n_dim]` accumulator tile."""

    if n_dim <= 0:
        raise ValueError(f"n_dim must be positive, got {n_dim}")
    if n_dim % 8 != 0:
        raise ValueError(f"n_dim must be a multiple of 8, got {n_dim}")
    return n_dim


def _tcgen05_f16_input_tmem_cols(k_dim: int) -> int:
    """Return TMEM columns for an F16 `[128, k_dim]` A-input staging tile."""

    if k_dim <= 0:
        raise ValueError(f"k_dim must be positive, got {k_dim}")
    if k_dim % 2 != 0:
        raise ValueError(f"k_dim must be even for packed F16 TMEM, got {k_dim}")
    return k_dim // 2


def _tcgen05_allocation_tmem_cols(required_cols: int) -> int:
    """Round required TMEM columns up to a tcgen05.alloc-supported count."""

    for alloc_cols in TCGEN05_VALID_ALLOC_COLS:
        if required_cols <= alloc_cols:
            return alloc_cols
    raise ValueError(f"required_cols must be <= 512, got {required_cols}")


# Value-side TMEM layout for the BT=16 KDA schedule.  Gate TMEM is intentionally
# omitted until that path is implemented.  The state slot holds the recurrent VK
# state across chunks and is reused in-place as the final_state accumulator late
# in each chunk.  The two shared_input slots stage F16 A operands, while
# shared_acc is the runtime-selected accumulator pool used by state*k and update.
#
KDA_TMEM_N16_ACC_COLS: int = _tcgen05_accumulator_tmem_cols(BT)
KDA_TMEM_N128_ACC_COLS: int = _tcgen05_accumulator_tmem_cols(DK)
KDA_TMEM_STATE_COLS: int = KDA_TMEM_N128_ACC_COLS
KDA_TMEM_STATE_AS_INPUT_COLS: int = _tcgen05_f16_input_tmem_cols(DK)
KDA_TMEM_SHARED_INPUT_COLS: int = _tcgen05_f16_input_tmem_cols(BT)
KDA_TMEM_SHARED_INPUT_STAGE_COUNT: int = 2
KDA_TMEM_QSTATE_ACC_STAGE_COUNT: int = 2
KDA_TMEM_SHARED_ACC_STAGE_COUNT: int = 2


KDA_TMEM_STATE_COL_OFFSET: int = 0
KDA_TMEM_FINAL_STATE_ACC_COL_OFFSET: int = KDA_TMEM_STATE_COL_OFFSET
KDA_TMEM_STATE_AS_INPUT_COL_OFFSET: int = (
    KDA_TMEM_STATE_COL_OFFSET + KDA_TMEM_STATE_COLS
)
KDA_TMEM_SHARED_INPUT_COL_OFFSET: int = (
    KDA_TMEM_STATE_AS_INPUT_COL_OFFSET + KDA_TMEM_STATE_AS_INPUT_COLS
)
KDA_TMEM_QSTATE_ACC_COL_OFFSET: int = (
    KDA_TMEM_SHARED_INPUT_COL_OFFSET
    + KDA_TMEM_SHARED_INPUT_STAGE_COUNT * KDA_TMEM_SHARED_INPUT_COLS
)
KDA_TMEM_SHARED_ACC_COL_OFFSET: int = (
    KDA_TMEM_QSTATE_ACC_COL_OFFSET + KDA_TMEM_N16_ACC_COLS
)
KDA_TMEM_QSTATE_ACC_STAGE1_COL_OFFSET: int = (
    KDA_TMEM_SHARED_ACC_COL_OFFSET
    + KDA_TMEM_SHARED_ACC_STAGE_COUNT * KDA_TMEM_N16_ACC_COLS
)
KDA_TMEM_QSTATE_ACC_STAGE_STRIDE_COLS: int = (
    KDA_TMEM_QSTATE_ACC_STAGE1_COL_OFFSET - KDA_TMEM_QSTATE_ACC_COL_OFFSET
)


KDA_TMEM_LAYOUT_COLS: int = (
    KDA_TMEM_QSTATE_ACC_STAGE1_COL_OFFSET + KDA_TMEM_N16_ACC_COLS
)
KDA_TMEM_ALLOC_COLS: int = _tcgen05_allocation_tmem_cols(KDA_TMEM_LAYOUT_COLS)


@cute.jit
def cta_sync() -> None:
    """Synchronize all threads in the CTA."""

    prims.barrier_cta_sync(0, thread_count=THREADS_PER_CTA)


@cute.jit
def cg0_sync(cg0_group_id) -> None:
    """Synchronize one four-warp CG0 producer group."""

    if cg0_group_id == 0:
        prims.barrier_cta_sync(
            NBAR_CG0_GROUP0_ID,
            thread_count=CG0_THREADS_PER_GROUP,
        )
    elif cg0_group_id == 1:
        prims.barrier_cta_sync(
            NBAR_CG0_GROUP1_ID,
            thread_count=CG0_THREADS_PER_GROUP,
        )
    else:
        prims.barrier_cta_sync(
            NBAR_CG0_GROUP2_ID,
            thread_count=CG0_THREADS_PER_GROUP,
        )


@cute.jit
def tmem_user_sync() -> None:
    """Named barrier for CG1 plus the tcgen05 warp during TMEM lifecycle setup."""

    prims.barrier_cta_sync(NBAR_TMEM_LIFECYCLE_ID, thread_count=TMEM_USER_THREADS)


@cute.jit
def is_compute_group0_warp(warp_idx) -> cutlass.Boolean:
    """Return whether this warp belongs to CG0 preprocessing."""

    return (warp_idx >= ROLES.compute_group0_first) & (
        warp_idx <= ROLES.compute_group0_last
    )


@cute.jit
def is_compute_group1_warp(warp_idx) -> cutlass.Boolean:
    """Return whether this warp belongs to CG1 value/final-state work."""

    return (warp_idx >= ROLES.compute_group1_first) & (
        warp_idx <= ROLES.compute_group1_last
    )


@cute.jit
def is_tmem_user_warp(warp_idx) -> cutlass.Boolean:
    """Return whether this warp needs the allocated TMEM base pointer."""

    return is_compute_group1_warp(warp_idx) | (warp_idx == ROLES.tcgen05_mma)


@cute.jit
def is_service_warpgroup(warp_idx) -> cutlass.Boolean:
    """Return whether this warp belongs to the non-CG0/CG1 service warpgroup."""

    return (warp_idx >= ROLES.super_mma) & (warp_idx <= ROLES.epilogue)


@cute.jit
def warp_group_sum_8(value: cutlass.Float32) -> cutlass.Float32:
    """Reduce independent 8-lane row groups inside one warp."""

    value = value + cutlass.Float32(
        prims.shfl_sync(cute.arch.FULL_MASK, value, 4, 0x1F, prims.Shfl.BFLY)
    )
    value = value + cutlass.Float32(
        prims.shfl_sync(cute.arch.FULL_MASK, value, 2, 0x1F, prims.Shfl.BFLY)
    )
    return value + cutlass.Float32(
        prims.shfl_sync(cute.arch.FULL_MASK, value, 1, 0x1F, prims.Shfl.BFLY)
    )


@cute.jit
def mma_input_dtype(
    value: cutlass.Float32,
    input_dtype: cutlass.Constexpr,
) -> cutlass.Float32:
    """Round a scalar through the compile-time input dtype for MMA reuse."""

    return value.to(input_dtype).to(cutlass.Float32)


@cute.jit
def pairwise_eye(row_coord, col_coord) -> cutlass.Float32:
    """Return the 16x16 identity value used by inverse product factors."""

    return cutlass.Float32(1.0) if row_coord == col_coord else cutlass.Float32(0.0)


@cute.jit
def pairwise_sw32_smem_index(offset: cutlass.Constexpr, row_coord, col_coord):
    """Return the SW32 physical index for a logical pairwise tile element."""

    storage_col_coord = col_coord ^ PAIRWISE_SW32_COL_XOR
    return offset + tcgen05_swizzle_32b_elem_index(
        row_coord * PAIRWISE_SW32_ROW_STRIDE + storage_col_coord,
        TCGEN05_F16_ELEM_BYTES,
    )


@cute.jit
def pairwise_stmatrix_m8n8x4_ptr(
    pairwise_smem,
    offset: cutlass.Constexpr,
    lane,
):
    """Return the row-start pointer for a 16x16 F16 pairwise STSM store."""

    matrix_id = lane // 8
    row_coord = lane & 7
    col_coord = cutlass.Int32(0)
    if matrix_id & 1:
        row_coord = row_coord + cutlass.Int32(8)
    if matrix_id >= 2:
        col_coord = cutlass.Int32(8)
    return pairwise_smem.subview(
        pairwise_sw32_smem_index(offset, row_coord, col_coord)
    ).data_ptr()


@cute.jit
def tcgen05_swizzle_128b_elem_index(
    linear_elem_idx,
    elem_bytes: cutlass.Constexpr,
    rows: cutlass.Constexpr,
):
    """Return the K-box-major SW128 physical element index."""

    elems_per_128b = 128 // elem_bytes
    row_coord = linear_elem_idx // DK
    col_coord = linear_elem_idx - row_coord * DK
    slice_coord = col_coord // elems_per_128b
    col_in_slice = col_coord - slice_coord * elems_per_128b
    slice_linear_idx = row_coord * elems_per_128b + col_in_slice
    byte_offset = slice_linear_idx * elem_bytes
    swizzle_mask = ((byte_offset >> 7) & 0x7) << 4
    return slice_coord * rows * elems_per_128b + (
        (byte_offset ^ swizzle_mask) // elem_bytes
    )


@cute.jit
def tcgen05_swizzle_32b_elem_index(
    linear_elem_idx,
    elem_bytes: cutlass.Constexpr,
):
    """Return the SW32 physical element index for a row-major logical tile."""

    byte_offset = linear_elem_idx * elem_bytes
    # SW32 is CuTe Swizzle<1,4,3>: xor address bit 7 into bit 4.
    swizzle_mask = ((byte_offset >> 7) & 0x1) << 4
    return (byte_offset ^ swizzle_mask) // elem_bytes


@cute.jit
def tcgen05_decay_b_key_storage_dim_runtime(token_coord, key_dim):
    """Return the runtime key coordinate for tcgen05 SW128 decay operands.

    Only the constant half-atom interleave applies.  The former extra
    token-dependent XOR (``(token & 2) * K_ATOM``) was matched by the warp
    ldmatrix reader (self-consistent, so the intra-chunk path stayed clean)
    but NOT by the tcgen05 state MMA, which reads this tile through a
    standard-layout SMEM descriptor: tokens with ``t % 4 in {2, 3}`` read a
    decay row displaced by 32 key channels every chunk.  The error scales
    with state persistence (large-negative ``dt_bias``), which is why random
    small-gate tests never caught it.
    """

    key_mask = cutlass.Int32(TCGEN05_F16_K_ATOM // 2)
    return key_dim ^ key_mask


# Token-major staged output tiles consumed by the epilogue role.  RHS/update
# inputs are now packed by CG1 directly into shared_input TMEM, and value-side B
# operands that need tcgen05-specific layouts live in distinct
# `tcgen05_*_smem` buffers.
O_STAGE_COUNT: int = 2
O_STAGE_COLS: int = DV
O_OUT_OFFSET: int = 0
O_ELEM_BYTES: int = TCGEN05_F16_ELEM_BYTES
O_TMA_SWIZZLE_BYTES: int = 128
O_TMA_SWIZZLE_ELEMS: int = O_TMA_SWIZZLE_BYTES // O_ELEM_BYTES
O_TMA_SWIZZLE_GROUP_BYTES: int = 16
O_TMA_SWIZZLE_GROUP_ELEMS: int = O_TMA_SWIZZLE_GROUP_BYTES // O_ELEM_BYTES
O_TMA_SWIZZLE_ROW_MASK: int = (O_TMA_SWIZZLE_ELEMS // O_TMA_SWIZZLE_GROUP_ELEMS) - 1
O_TMA_SEGMENTS: int = DV // O_TMA_SWIZZLE_ELEMS
O_TMA_SWIZZLE_ALIGNMENT_BYTES: int = O_TMA_SWIZZLE_BYTES * (
    O_TMA_SWIZZLE_ELEMS // O_TMA_SWIZZLE_GROUP_ELEMS
)
O_SMEM_STAGE_SIZE: int = BT * O_STAGE_COLS
O_SMEM_TILE_SIZE: int = O_STAGE_COUNT * O_SMEM_STAGE_SIZE

TCGEN05_VALUE_PAIRWISE_B_LEADING_BYTES: int = 16
TCGEN05_VALUE_PAIRWISE_B_STRIDE_BYTES: int = 8 * BT * TCGEN05_F16_ELEM_BYTES
TCGEN05_FINAL_STATE_B_N_GROUP_ELEMS: int = TCGEN05_SW128_BYTES // TCGEN05_F16_ELEM_BYTES
TCGEN05_FINAL_STATE_B_LEADING_BYTES: int = (
    BT * TCGEN05_FINAL_STATE_B_N_GROUP_ELEMS * TCGEN05_F16_ELEM_BYTES
)
TCGEN05_FINAL_STATE_B_STRIDE_BYTES: int = (
    8 * TCGEN05_FINAL_STATE_B_N_GROUP_ELEMS * TCGEN05_F16_ELEM_BYTES
)
TCGEN05_FINAL_STATE_TMEM_LOAD_COLS: int = 32
DECAY_STAGE_COUNT: int = 2
Q_K_RESTORE_READY_STAGE_COUNT: int = 3
TCGEN05_K_DECAY_STAGE_SIZE: int = DK * BT
TCGEN05_Q_DECAY_STAGE_SIZE: int = DK * BT
TCGEN05_K_RESTORE_STAGE_SIZE: int = DK * BT
TCGEN05_K_DECAY_SMEM_TILE_SIZE: int = DECAY_STAGE_COUNT * TCGEN05_K_DECAY_STAGE_SIZE
TCGEN05_Q_DECAY_SMEM_TILE_SIZE: int = DECAY_STAGE_COUNT * TCGEN05_Q_DECAY_STAGE_SIZE
TCGEN05_K_RESTORE_SMEM_TILE_SIZE: int = DECAY_STAGE_COUNT * TCGEN05_K_RESTORE_STAGE_SIZE

# Per-chunk q/k/v/gate/beta inputs staged by the TMA/load role.  "raw" means
# pre-normalization/pre-decay input; the 16-bit raw tiles still use SW128.
RAW_STAGE_COUNT: int = 8
# Depth of the TMA-completion mbarrier ring in the ENGINE-CLASS kernels
# (engine / decomp-k1 prep; consumer-direct-wait).  Chunk c's
# transaction is issued against tma_mbar[c % D] and CONSUMERS wait that
# slot's parity directly (no raw_ready relay), so D must equal the raw SMEM
# ring depth: consumers reuse the raw-slot index and flip phase on the same
# 8-chunk wrap.  Warp 14 never waits completions; raw_consumed alone
# throttles issue to <= D chunks in flight.  (The decomp k2 chain kernels
# keep their own relay ring, K2_TMA_MBAR_STAGE_COUNT below.)
TMA_MBAR_STAGE_COUNT: int = RAW_STAGE_COUNT
if TMA_MBAR_STAGE_COUNT != RAW_STAGE_COUNT:
    raise ValueError("consumer-direct TMA ring must match the raw SMEM ring depth")

RAW_F16_TMA_SWIZZLE_BYTES: int = 128
RAW_F16_TMA_SWIZZLE_ELEMS: int = RAW_F16_TMA_SWIZZLE_BYTES // TCGEN05_F16_ELEM_BYTES
RAW_F16_TMA_SWIZZLE_GROUP_BYTES: int = 16
RAW_F16_TMA_SWIZZLE_GROUP_ELEMS: int = (
    RAW_F16_TMA_SWIZZLE_GROUP_BYTES // TCGEN05_F16_ELEM_BYTES
)
RAW_F16_TMA_SWIZZLE_ROW_MASK: int = (
    RAW_F16_TMA_SWIZZLE_ELEMS // RAW_F16_TMA_SWIZZLE_GROUP_ELEMS
) - 1
RAW_F16_TMA_SEGMENTS: int = DK // RAW_F16_TMA_SWIZZLE_ELEMS
RAW_F16_TMA_SEGMENT_ELEMS: int = BT * RAW_F16_TMA_SWIZZLE_ELEMS
RAW_F16_TMA_SWIZZLE_ALIGNMENT_BYTES: int = RAW_F16_TMA_SWIZZLE_BYTES * (
    RAW_F16_TMA_SWIZZLE_ELEMS // RAW_F16_TMA_SWIZZLE_GROUP_ELEMS
)
RAW_F16_TAIL_ZERO_LANES: int = DK // RAW_F16_TMA_SWIZZLE_GROUP_ELEMS

RAW_F32_ELEM_BYTES: int = 4
RAW_F32_TMA_SWIZZLE_BYTES: int = 128
RAW_F32_TMA_SWIZZLE_ELEMS: int = RAW_F32_TMA_SWIZZLE_BYTES // RAW_F32_ELEM_BYTES
RAW_F32_TMA_SWIZZLE_GROUP_BYTES: int = 16
RAW_F32_TMA_SWIZZLE_GROUP_ELEMS: int = (
    RAW_F32_TMA_SWIZZLE_GROUP_BYTES // RAW_F32_ELEM_BYTES
)
CG0_TOKEN_ROWS_PER_WARP: int = BT // CG0_WARPS_PER_GROUP
RAW_F32_TMA_SWIZZLE_ROW_MASK: int = (
    RAW_F32_TMA_SWIZZLE_ELEMS // RAW_F32_TMA_SWIZZLE_GROUP_ELEMS
) - 1
RAW_F32_TMA_SEGMENTS: int = DK // RAW_F32_TMA_SWIZZLE_ELEMS
RAW_F32_TMA_SEGMENT_ELEMS: int = BT * RAW_F32_TMA_SWIZZLE_ELEMS
RAW_F32_TMA_SWIZZLE_ALIGNMENT_BYTES: int = RAW_F32_TMA_SWIZZLE_BYTES * (
    RAW_F32_TMA_SWIZZLE_ELEMS // RAW_F32_TMA_SWIZZLE_GROUP_ELEMS
)
RAW_F32_TAIL_ZERO_LANES: int = DK // RAW_F32_TMA_SWIZZLE_GROUP_ELEMS

RAW_Q_STAGE_SIZE: int = DK * BT
RAW_K_STAGE_SIZE: int = DK * BT
RAW_V_STAGE_SIZE: int = DV * BT
RAW_GATE_STAGE_SIZE: int = DK * BT
# FP32 gate-prefix exchange ring.  Sizes are ELEMENT counts, so the raw-gate
# ring above is dtype-agnostic (its byte size follows `gate_dtype`); this ring
# is always FP32 and is only ALLOCATED on the 16-bit gate path -- with an FP32
# gate the exchange keeps aliasing the raw-gate stage exactly as it always did.
#
# CG0 writes exp2(g_prefix) for one chunk, then CG0 (materialize + pack_rescale)
# and CG1 (publish_projection_then_rescale) read it in the SAME chunk.  There is
# NO consumed mbarrier for this tile (it used to be covered by living inside the
# 8-deep raw ring under `raw_consumed`), so the depth alone must order producer
# chunk c+D against the readers of chunk c.  D=2 does NOT close: CG0 group g's
# write of c+2 and CG1's read of c are both gated only by
# k_restore_consumed(c-1).  D=4 closes through k_restore_consumed_l(c+1) ->
# update_ready(c+1) -> CG1 chunk c+1 -> k_restore_consumed(c) -> update_ready(c)
# -> CG0 group g's pack_rescale(c), and the two CG0 groups own disjoint stage
# parities.
GATE_EXCHANGE_STAGE_COUNT: int = 4
GATE_EXCHANGE_STAGE_SIZE: int = DK * BT
RAW_BETA_STAGE_SIZE: int = BT
RAW_DT_BIAS_STAGE_SIZE: int = DK
RAW_DT_BIAS_A_LOG_EXP_OFFSET: int = RAW_DT_BIAS_STAGE_SIZE
RAW_Q_SMEM_TILE_SIZE: int = RAW_STAGE_COUNT * RAW_Q_STAGE_SIZE
RAW_K_SMEM_TILE_SIZE: int = RAW_STAGE_COUNT * RAW_K_STAGE_SIZE
RAW_V_SMEM_TILE_SIZE: int = RAW_STAGE_COUNT * RAW_V_STAGE_SIZE
RAW_GATE_SMEM_TILE_SIZE: int = RAW_STAGE_COUNT * RAW_GATE_STAGE_SIZE
GATE_EXCHANGE_SMEM_TILE_SIZE: int = GATE_EXCHANGE_STAGE_COUNT * GATE_EXCHANGE_STAGE_SIZE
RAW_BETA_SMEM_TILE_SIZE: int = RAW_STAGE_COUNT * RAW_BETA_STAGE_SIZE
RAW_DT_BIAS_SMEM_TILE_SIZE: int = RAW_DT_BIAS_STAGE_SIZE + 1


# The public gate ABI is BF16. Float32 preserves compatibility with callers of
# the historical kernel; other dtypes are intentionally rejected so they do not
# silently multiply the compile-specialization matrix.
GATE_DTYPES_SUPPORTED: tuple = (cutlass.BFloat16, cutlass.Float32)


def gate_dtype_is_f32(gate_dtype) -> bool:
    """True when the gate rides the historical FP32 memory format."""

    return gate_dtype is cutlass.Float32


def validate_gate_dtype(gate_dtype):
    """Reject any gate dtype the kernel has no lowering for."""

    if gate_dtype not in GATE_DTYPES_SUPPORTED:
        names = ", ".join(getattr(d, "__name__", str(d)) for d in GATE_DTYPES_SUPPORTED)
        raise TypeError(
            "KDA gate_dtype must be one of {"
            + names
            + "}, got "
            + str(getattr(gate_dtype, "__name__", gate_dtype))
        )
    return gate_dtype


K_INV_STAGE_SIZE: int = BT * DK
K_INV_SMEM_TILE_SIZE: int = DECAY_STAGE_COUNT * K_INV_STAGE_SIZE
PAIRWISE_SMEM_QK_OFFSET: int = 0
PAIRWISE_SMEM_AINV_OFFSET: int = PAIRWISE_SMEM_QK_OFFSET + PAIRWISE_SW32_TILE_ELEMS
PAIRWISE_SMEM_STAGE_SIZE: int = PAIRWISE_SMEM_AINV_OFFSET + PAIRWISE_SW32_TILE_ELEMS
PAIRWISE_STAGE_COUNT: int = 2
PAIRWISE_SMEM_TILE_SIZE: int = PAIRWISE_STAGE_COUNT * PAIRWISE_SMEM_STAGE_SIZE

SUPER_MMA_ATOM_N: int = 8
SUPER_MMA_ATOM_K: int = 16
SUPER_MMA_K_BLOCKS: int = DK // SUPER_MMA_ATOM_K
SUPER_MMA_ACCUMULATORS_PER_LANE: int = 4


@dataclass(frozen=True)
class WarpRoles:
    """Warp assignment for the BT=16 fully fused KDA kernel."""

    compute_group0_first: int = 0
    compute_group0_last: int = 7
    compute_group1_first: int = 8
    compute_group1_last: int = 11
    super_mma: int = 12
    tcgen05_mma: int = 13
    tma_load: int = 14
    epilogue: int = 15


ROLES = WarpRoles()


DTYPE_MAP: dict[type, torch.dtype] = {
    cutlass.Float16: torch.float16,
    cutlass.BFloat16: torch.bfloat16,
    cutlass.Float32: torch.float32,
}

CLI_DTYPES: dict[str, type] = {
    "fp16": cutlass.Float16,
    "bf16": cutlass.BFloat16,
}

CLI_STATE_DTYPES: dict[str, type] = {
    "bf16": cutlass.BFloat16,
    "fp32": cutlass.Float32,
}

# Gate ABI dtype selectable from the CLI harness.  BF16 is the default (see the
# module docstring); FP32 selects the historical build.
CLI_GATE_DTYPES: dict[str, type] = {
    "bf16": cutlass.BFloat16,
    "fp32": cutlass.Float32,
}


# ---------------------------------------------------------------------------
# Torch reference: current numerical contract
# ---------------------------------------------------------------------------


_LN_2: float = 0.6931471805599453


def _torch_exp2(x: torch.Tensor) -> torch.Tensor:
    """``torch.exp2`` via the ``exp`` identity.

    ``torch.exp2`` lowers through the CUDA jiterator (a runtime NVRTC
    compile), which fails when the pip NVRTC build predates the running
    device's architecture; ``torch.exp`` is precompiled ATen and never
    touches NVRTC.  The identity costs at most ~1 ulp on the FP32 reference
    path, far below the kernel comparison tolerances.
    """

    return torch.exp(x * _LN_2)


def _torch_l2_normalize_qk(
    q: torch.Tensor, k: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """L2-normalize q/k on the head dimension (rsqrt of the squared sum)."""

    return (
        torch.nn.functional.normalize(q.float(), p=2.0, dim=-1, eps=L2_NORM_EPS),
        torch.nn.functional.normalize(k.float(), p=2.0, dim=-1, eps=L2_NORM_EPS),
    )


def _torch_mma_input(x: torch.Tensor, mma_dtype: torch.dtype) -> torch.Tensor:
    """Round tensors as 16-bit MMA inputs while keeping FP32 accumulation sites."""

    if mma_dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"mma_dtype must be fp16 or bf16, got {mma_dtype}")
    return x.to(mma_dtype).float()


def _torch_mma_input_mul(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    mma_dtype: torch.dtype,
) -> torch.Tensor:
    """Model packed 16-bit multiply used when materializing decay operands."""

    return _torch_mma_input(
        _torch_mma_input(lhs, mma_dtype) * _torch_mma_input(rhs, mma_dtype),
        mma_dtype,
    )


def _torch_safe_gate_log2_increment(
    raw_gate: torch.Tensor,
    a_log: torch.Tensor | None,
    dt_bias: torch.Tensor | None,
    safe_gate: bool,
    gate_lower_bound: float,
) -> torch.Tensor:
    """Apply FLA-compatible KDA gate preprocessing in log2-domain units.

    safe_gate=True -> the bounded sigmoid gate;
    safe_gate=False -> FLA's unbounded ``-exp(A_log) * softplus(g + dt_bias)``
    activation, both returned as base-2 (log2) decay increments.
    """

    if a_log is None:
        a_log = torch.zeros(
            raw_gate.shape[1], dtype=torch.float32, device=raw_gate.device
        )
    if dt_bias is None:
        dt_bias = torch.zeros(
            raw_gate.shape[1],
            raw_gate.shape[-1],
            dtype=torch.float32,
            device=raw_gate.device,
        )
    if a_log.shape != (raw_gate.shape[1],):
        raise ValueError(f"a_log must be [H], got {a_log.shape}")
    if dt_bias.shape != (raw_gate.shape[1], raw_gate.shape[-1]):
        raise ValueError(
            f"dt_bias must be [H,{raw_gate.shape[-1]}], got {dt_bias.shape}"
        )

    a_log_exp = _torch_exp2(a_log.float() * LOG2_E).view(1, raw_gate.shape[1], 1, 1)
    biased_gate = raw_gate.float() + dt_bias.float().view(
        1, raw_gate.shape[1], 1, raw_gate.shape[-1]
    )
    if safe_gate:
        sigmoid = 0.5 * torch.tanh(0.5 * a_log_exp * biased_gate) + 0.5
        return (gate_lower_bound * LOG2_E) * sigmoid
    return -a_log_exp * torch.nn.functional.softplus(biased_gate) * LOG2_E


def _state_vk_to_kv(state_vk: torch.Tensor) -> torch.Tensor:
    """Interpret a VK `[B, H, DV, DK]` state tensor with the reference KV axes."""

    if state_vk.shape[-2:] != (DV, DK):
        raise ValueError(f"VK state must end with [{DV},{DK}], got {state_vk.shape}")
    return state_vk.transpose(-1, -2)


def _state_kv_to_vk(state_kv: torch.Tensor) -> torch.Tensor:
    """Interpret a KV `[B, H, DK, DV]` reference state with kernel VK axes."""

    if state_kv.shape[-2:] != (DK, DV):
        raise ValueError(f"KV state must end with [{DK},{DV}], got {state_kv.shape}")
    return state_kv.transpose(-1, -2)


def _blockwise_inverse_unit_lower_bt16_mma(
    strict_lower: torch.Tensor,
    mma_dtype: torch.dtype,
) -> torch.Tensor:
    """BT=16 blockwise inverse mirroring the device auxiliary-MMA stage.

    Hierarchical (GDN-style) block inverse expressed entirely as MMAs:

        D    = blockdiag(L11, L22)        (strict-lower 8x8 diagonal blocks)
        Binv = (I - D)(I + D^2)(I + D^4)  (exact: blocks nilpotent at 8)
        A^-1 = Binv - Binv @ A21hat @ Binv  (exact: (Binv @ A21hat)^2 = 0)

    Chain and combine operands are FP16 (FP32 accumulate) regardless of the
    kernel input dtype, matching the device stage; only the staged A_inv
    tile is rounded to `mma_dtype`.
    """

    if strict_lower.shape[-2:] != (BT, BT):
        raise ValueError(
            f"strict_lower must end with [{BT},{BT}], got {strict_lower.shape}"
        )

    chain_dtype = torch.float16
    strict_lower = strict_lower.float()
    eye = torch.eye(BT, dtype=torch.float32, device=strict_lower.device)

    d = torch.zeros_like(strict_lower)
    d[..., :8, :8] = strict_lower[..., :8, :8]
    d[..., 8:, 8:] = strict_lower[..., 8:, 8:]
    d16 = _torch_mma_input(d, chain_dtype)

    d2 = _torch_mma_input(d16 @ d16, chain_dtype)
    d4 = _torch_mma_input(d2 @ d2, chain_dtype)

    b01_lhs = _torch_mma_input(eye - d, chain_dtype)
    b01_rhs = _torch_mma_input(eye + d2, chain_dtype)
    b01 = _torch_mma_input(b01_lhs @ b01_rhs, chain_dtype)
    binv = b01 @ _torch_mma_input(eye + d4, chain_dtype)  # FP32 accumulator

    a21hat = torch.zeros_like(strict_lower)
    a21hat[..., 8:, :8] = strict_lower[..., 8:, :8]
    t1 = _torch_mma_input(binv, chain_dtype) @ _torch_mma_input(a21hat, chain_dtype)
    x21 = -(_torch_mma_input(t1, chain_dtype) @ _torch_mma_input(binv, chain_dtype))

    ainv = binv.clone()
    ainv[..., 8:, :8] = x21[..., 8:, :8]
    return _torch_mma_input(ainv, mma_dtype)


@dataclass(frozen=True)
class PairwiseF16MmaTiles:
    """Input-dtype-rounded tiles owned by the KDA auxiliary-MMA warp."""

    kk: torch.Tensor
    qk: torch.Tensor
    strict_lower: torch.Tensor
    a_inv: torch.Tensor


@cute.jit
def raw_f16_s128_smem_index(token_coord, dim):
    """Return the physical s128 SMEM index for raw F16 q/k/v staging."""

    segment = dim // RAW_F16_TMA_SWIZZLE_ELEMS
    segment_dim = dim - segment * RAW_F16_TMA_SWIZZLE_ELEMS
    col_group = segment_dim // RAW_F16_TMA_SWIZZLE_GROUP_ELEMS
    col_in_group = segment_dim - col_group * RAW_F16_TMA_SWIZZLE_GROUP_ELEMS
    row_swizzle = token_coord & RAW_F16_TMA_SWIZZLE_ROW_MASK
    return (
        segment * RAW_F16_TMA_SEGMENT_ELEMS
        + token_coord * RAW_F16_TMA_SWIZZLE_ELEMS
        + ((col_group ^ row_swizzle) * RAW_F16_TMA_SWIZZLE_GROUP_ELEMS)
        + col_in_group
    )


@cute.jit
def raw_f32_s128_smem_index(token_coord, dim):
    """Return the physical s128 SMEM index for raw F32 gate staging."""

    segment = dim // RAW_F32_TMA_SWIZZLE_ELEMS
    segment_dim = dim - segment * RAW_F32_TMA_SWIZZLE_ELEMS
    col_group = segment_dim // RAW_F32_TMA_SWIZZLE_GROUP_ELEMS
    col_in_group = segment_dim - col_group * RAW_F32_TMA_SWIZZLE_GROUP_ELEMS
    row_swizzle = token_coord & RAW_F32_TMA_SWIZZLE_ROW_MASK
    return (
        segment * RAW_F32_TMA_SEGMENT_ELEMS
        + token_coord * RAW_F32_TMA_SWIZZLE_ELEMS
        + ((col_group ^ row_swizzle) * RAW_F32_TMA_SWIZZLE_GROUP_ELEMS)
        + col_in_group
    )


def raw_f32_exchange_smem_index(token_coord, dim):
    """Return a conflict-free SMEM index for the CG0 gate-prefix exchange.

    The TMA s128 swizzle XORs banks only within one 128-byte segment, so the
    vectorized prefix re-reads hit the same bank group once per segment
    (2-way conflicts). The exchange scratch is FP32 and either reuses the
    raw-gate stage after its raw values are consumed (FP32 gate) or is a
    dedicated `gate_exchange_smem` buffer (16-bit gate), so both sides of the
    exchange can add a segment XOR that spreads segments across bank groups;
    column-wise prefix stores stay conflict-free because the segment term is
    constant per warp.

    NOTE the layout math stays on the RAW_F32 s128 geometry on BOTH gate paths:
    this buffer is FP32 and independent of the TMA landing layout.
    """

    segment = dim // RAW_F32_TMA_SWIZZLE_ELEMS
    segment_dim = dim - segment * RAW_F32_TMA_SWIZZLE_ELEMS
    col_group = segment_dim // RAW_F32_TMA_SWIZZLE_GROUP_ELEMS
    col_in_group = segment_dim - col_group * RAW_F32_TMA_SWIZZLE_GROUP_ELEMS
    row_swizzle = token_coord & RAW_F32_TMA_SWIZZLE_ROW_MASK
    return (
        segment * RAW_F32_TMA_SEGMENT_ELEMS
        + token_coord * RAW_F32_TMA_SWIZZLE_ELEMS
        + ((col_group ^ row_swizzle ^ segment) * RAW_F32_TMA_SWIZZLE_GROUP_ELEMS)
        + col_in_group
    )


@cute.jit
def k_inv_s128_smem_index(token_coord, key_dim):
    """Return the physical s128 SMEM index for the K-inverse auxiliary-MMA RHS."""

    return raw_f16_s128_smem_index(token_coord, key_dim)


@cute.jit
def o_smem_swizzle_128b_elem_index(
    o_stage_base,
    value_dim,
    token_coord,
):
    """Return the physical W128 SMEM index for one staged output element."""

    segment = value_dim // O_TMA_SWIZZLE_ELEMS
    segment_value_dim = value_dim - segment * O_TMA_SWIZZLE_ELEMS
    col_group = segment_value_dim // O_TMA_SWIZZLE_GROUP_ELEMS
    col_in_group = segment_value_dim - col_group * O_TMA_SWIZZLE_GROUP_ELEMS
    row_swizzle = token_coord & O_TMA_SWIZZLE_ROW_MASK
    return (
        o_stage_base
        + O_OUT_OFFSET
        + segment * BT * O_TMA_SWIZZLE_ELEMS
        + token_coord * O_TMA_SWIZZLE_ELEMS
        + ((col_group ^ row_swizzle) * O_TMA_SWIZZLE_GROUP_ELEMS)
        + col_in_group
    )


@cute.jit
def o_smem_stmatrix_128b_ptr(
    o_smem,
    o_stage_base,
    value_dim_base,
    lane,
):
    """Return the per-lane W128 row-start pointer for one 16x16 STSM.T tile."""

    matrix_id = lane // 8
    row_in_matrix = lane & 7
    token_block = matrix_id // 2
    value_block = matrix_id & 1
    token_coord = token_block * 8 + row_in_matrix
    value_dim = value_dim_base + value_block * 8
    smem_idx = o_smem_swizzle_128b_elem_index(
        o_stage_base,
        value_dim,
        token_coord,
    )
    return o_smem.subview(smem_idx).data_ptr()


@cute.jit
def raw_v_ldmatrix_trans_ptr(raw_v_smem, value_dim_base, lane):
    """Return the per-lane row-start pointer for raw V `ldmatrix.x4.trans`."""

    matrix_id = lane // 8
    row_in_matrix = lane & 7
    token_block = matrix_id // 2
    value_block = matrix_id & 1
    token_coord = token_block * 8 + row_in_matrix
    value_dim = value_dim_base + value_block * 8
    smem_idx = raw_f16_s128_smem_index(token_coord, value_dim)
    return raw_v_smem.subview(smem_idx).data_ptr()


def _kda_pairwise_mma_tiles(
    q_decay: torch.Tensor,
    k_decay: torch.Tensor,
    k_inv: torch.Tensor,
    beta_chunk: torch.Tensor,
    mma_dtype: torch.dtype,
) -> PairwiseF16MmaTiles:
    """Reference contract for the BT=16 KK/QK/inverse auxiliary-MMA path."""

    expected_tile_shape = (BT, DK)
    if q_decay.shape[-2:] != expected_tile_shape:
        raise ValueError(
            f"q_decay must end with {expected_tile_shape}, got {q_decay.shape}"
        )
    if k_decay.shape != q_decay.shape:
        raise ValueError(f"k_decay must match q_decay, got {k_decay.shape}")
    if k_inv.shape != q_decay.shape:
        raise ValueError(f"k_inv must match q_decay, got {k_inv.shape}")
    if beta_chunk.shape != q_decay.shape[:-1]:
        raise ValueError(f"beta_chunk must be [...,{BT}], got {beta_chunk.shape}")

    tril_strict = torch.tril(
        torch.ones(BT, BT, dtype=torch.bool, device=q_decay.device), diagonal=-1
    )
    tril_inclusive = torch.tril(
        torch.ones(BT, BT, dtype=torch.bool, device=q_decay.device), diagonal=0
    )

    kk = _torch_mma_input(
        torch.einsum("...ik,...jk->...ij", k_decay, k_inv),
        mma_dtype,
    )
    qk_full = _torch_mma_input(
        torch.einsum("...ik,...jk->...ij", q_decay, k_inv),
        mma_dtype,
    )
    kk_lower = torch.where(tril_strict, kk, torch.zeros_like(kk))
    qk = _torch_mma_input(
        torch.where(tril_inclusive, qk_full, torch.zeros_like(qk_full)),
        mma_dtype,
    )
    strict_lower = _torch_mma_input(kk_lower * beta_chunk[..., :, None], mma_dtype)
    a_inv = _blockwise_inverse_unit_lower_bt16_mma(strict_lower, mma_dtype)
    return PairwiseF16MmaTiles(
        kk=kk,
        qk=qk,
        strict_lower=strict_lower,
        a_inv=a_inv,
    )


def _pad_chunk_to_bt(value: torch.Tensor, valid_tokens: int) -> torch.Tensor:
    """Pad a `[B,H,T,...]` chunk to BT tokens with trailing zeros."""

    if valid_tokens == BT:
        return value
    pad_shape = list(value.shape)
    pad_shape[2] = BT - valid_tokens
    return torch.cat([value, value.new_zeros(pad_shape)], dim=2)


def _pad_beta_chunk_to_bt(value: torch.Tensor, valid_tokens: int) -> torch.Tensor:
    """Pad a `[B,H,T]` beta chunk to BT tokens with trailing zeros."""

    if valid_tokens == BT:
        return value
    pad_shape = list(value.shape)
    pad_shape[2] = BT - valid_tokens
    return torch.cat([value, value.new_zeros(pad_shape)], dim=2)


def kda_chunked_mma_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_gate: torch.Tensor,
    beta_logits: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    scale: float | None = None,
    mma_dtype: torch.dtype | None = None,
    a_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    safe_gate: bool = True,
    gate_lower_bound: float = DEFAULT_GATE_LOWER_BOUND,
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunk KDA reference with dtype-rounded operands at MMA input boundaries."""

    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(f"packed q batch must be 1, got {q.shape[0]}")
        boundaries = cu_seqlens.tolist()
        num_sequences = len(boundaries) - 1
        if initial_state is not None and initial_state.shape[0] != num_sequences:
            raise ValueError("packed initial_state must have one row per sequence")
        packed_out = torch.empty_like(v)
        final_states: list[torch.Tensor] = []
        for sequence_idx, (start, end) in enumerate(
            zip(boundaries[:-1], boundaries[1:], strict=True)
        ):
            sequence_out, sequence_state = kda_chunked_mma_reference(
                q[:, :, start:end, :],
                k[:, :, start:end, :],
                v[:, :, start:end, :],
                raw_gate[:, :, start:end, :],
                beta_logits[:, :, start:end],
                None
                if initial_state is None
                else initial_state[sequence_idx : sequence_idx + 1],
                scale,
                mma_dtype,
                a_log,
                dt_bias,
                safe_gate,
                gate_lower_bound,
            )
            packed_out[:, :, start:end, :] = sequence_out
            final_states.append(sequence_state)
        return packed_out, torch.cat(final_states, dim=0)

    if q.shape != k.shape:
        raise ValueError(
            f"q and k must have the same shape, got {q.shape} and {k.shape}"
        )
    if v.shape[:-1] != q.shape[:-1] or v.shape[-1] != DV:
        raise ValueError(f"v must be [B,H,T,{DV}], got {v.shape}")
    if raw_gate.shape != q.shape:
        raise ValueError(f"raw_gate must be [B,H,T,{DK}], got {raw_gate.shape}")
    if beta_logits.shape != q.shape[:-1]:
        raise ValueError(f"beta_logits must be [B,H,T], got {beta_logits.shape}")

    batch, heads, seqlen, d_k = q.shape
    if d_k != DK:
        raise ValueError(f"DK must be {DK}, got {d_k}")
    if scale is None:
        scale = 1.0 / (DK**0.5)
    if mma_dtype is None:
        mma_dtype = q.dtype

    state = (
        torch.zeros(batch, heads, DK, DV, dtype=torch.float32, device=q.device)
        if initial_state is None
        else initial_state.float().clone()
    )
    out = torch.empty(batch, heads, seqlen, DV, dtype=q.dtype, device=q.device)

    q_f, k_f = _torch_l2_normalize_qk(q, k)
    v_f = v.float()
    gate_f = _torch_safe_gate_log2_increment(
        raw_gate,
        a_log,
        dt_bias,
        safe_gate,
        gate_lower_bound,
    )
    beta_f = torch.tanh(beta_logits.float() * 0.5) * 0.5 + 0.5
    for chunk_start in range(0, seqlen, BT):
        chunk_end = min(chunk_start + BT, seqlen)
        valid_tokens = chunk_end - chunk_start
        q_chunk = _pad_chunk_to_bt(q_f[:, :, chunk_start:chunk_end, :], valid_tokens)
        k_chunk = _pad_chunk_to_bt(k_f[:, :, chunk_start:chunk_end, :], valid_tokens)
        v_chunk = _pad_chunk_to_bt(v_f[:, :, chunk_start:chunk_end, :], valid_tokens)
        gate_chunk = _pad_chunk_to_bt(
            gate_f[:, :, chunk_start:chunk_end, :],
            valid_tokens,
        )
        beta_chunk = _pad_beta_chunk_to_bt(
            beta_f[:, :, chunk_start:chunk_end],
            valid_tokens,
        )

        g_prefix = torch.cumsum(gate_chunk, dim=2)
        exp_g = _torch_exp2(g_prefix)
        exp_neg_g = _torch_exp2(-g_prefix)
        exp_g_last = _torch_exp2(g_prefix[:, :, -1, :])

        q_decay = _torch_mma_input_mul(q_chunk, exp_g, mma_dtype)
        k_decay = _torch_mma_input_mul(k_chunk, exp_g, mma_dtype)
        k_inv = _torch_mma_input(k_chunk * exp_neg_g, mma_dtype)
        k_restore = _torch_mma_input_mul(
            k_inv,
            exp_g_last[:, :, None, :],
            mma_dtype,
        )

        pairwise_tiles = _kda_pairwise_mma_tiles(
            q_decay,
            k_decay,
            k_inv,
            beta_chunk,
            mma_dtype,
        )

        state_mma = _torch_mma_input(state, mma_dtype)
        state_k = torch.einsum("bhtk,bhkv->bhtv", k_decay, state_mma)
        rhs = _torch_mma_input(
            beta_chunk[:, :, :, None] * (v_chunk - state_k),
            mma_dtype,
        )
        update = torch.einsum("bhij,bhjv->bhiv", pairwise_tiles.a_inv, rhs)
        update_mma = _torch_mma_input(update, mma_dtype)

        state_q = torch.einsum("bhtk,bhkv->bhtv", q_decay, state_mma)
        intra = torch.einsum("bhij,bhjv->bhiv", pairwise_tiles.qk, update_mma)
        out[:, :, chunk_start:chunk_end, :] = ((state_q + intra) * scale).to(q.dtype)[
            :, :, :valid_tokens, :
        ]

        state = state * exp_g_last[:, :, :, None] + torch.einsum(
            "bhtk,bhtv->bhkv", k_restore, update_mma
        )

    return out, state


# ---------------------------------------------------------------------------
# Device-stage helpers.  These keep each warp group's numerical contract
# explicit while the heavy MMA/TMEM lowering lands stage by stage.
# ---------------------------------------------------------------------------


# --- beta TMA transport ------------------------------------------------------
# beta arrives by 2D TMA instead of a per-lane strided LSU gather (which cost
# ~1.07us per extra 32B sector at H96: 16 lanes x heads*2B stride = 16 sectors
# vs 1).  One descriptor family, runtime-selected by g = 8/gcd(heads, 8):
#   g == 1 (heads % 8 == 0): view (heads, T) stride (1, heads); box = (8, BT)
#     -- 16B rows, the CTA fetches only its 8-head group.  Verified H96/64/32,
#     H96-T8192 engine 450.6 -> 434.2us (beats head-major 438.0; NCU: the
#     +786K-sector LSU signature vanishes exactly).
#   g > 1: PAIR-PACKED view (g*heads, ceil(T/g)) over the same memory -- one
#     packed row = g consecutive tokens x all heads, so the row stride
#     g*heads*2B is a legal 16B multiple even when heads % 8 != 0 (H12 = TP8
#     of H96: g=2, 48B).  Box = one packed row x (BT/g + 1) rows; the +1
#     parity row lets a chunk whose start token is not g-aligned round DOWN
#     to the packed grain and consume at offset (start % g).  Verified H12
#     bitwise incl. odd-bos varlen; requires heads <= 14 when heads % 8 != 0
#     (SMEM stage cap + Int8 box dims), which covers every TP split of
#     production head counts.
# The descriptor needs beta.data_ptr() 16B-aligned: guaranteed for allocator-
# fresh tensors (512B granule) and checked loudly in the wrapper; only an
# odd-token-offset *view* at g > 1 can violate it (see test_beta_alignment).
BETA_TILE_STAGE_ELEMS: int = 256  # bf16; >= max(8*BT group, heads*(BT+g) pair)
BETA_TILE_STAGE_COUNT: int = 8
BETA_TMA_LOOKAHEAD: int = 4


@cute.jit
def tma_stage_load_inputs(
    tma_desc_q: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_k: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_v: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_gate: cutlass.GridConstant[cuda.TensorMap],
    beta_tile_stage,
    beta_g,
    heads32,
    raw_q_smem,
    raw_k_smem,
    raw_v_smem,
    raw_gate_smem,
    raw_beta_smem,
    sequence_start,
    head_idx,
    lane,
    chunk_start,
    seqlen,
    tma_mbar,
    tma_tx_bytes: cutlass.Constexpr,
    gate_dtype: cutlass.Constexpr,
) -> None:
    """Issue tensor-operand TMA + sigmoid(beta logits) for one chunk (no wait).

    Consumers wait the chunk's tma_mbar slot parity directly, so the
    beta scalar store MUST precede the mbarrier_arrive_expect_tx: the
    arrive's release (after the warp sync) is the only edge that publishes
    beta to the consumers' acquire (there is no raw_ready relay any more).
    """

    global_chunk_start = sequence_start + chunk_start
    if lane < BT:
        token_idx = chunk_start + lane
        beta_value = cutlass.Float32(0.0)
        if token_idx < seqlen:
            # The tile is a contiguous [token][head-slot] slab staged by TMA:
            # g == 1 -> 8-head group rows (slot = head % 8, token = lane);
            # g > 1  -> whole packed rows from token (start - start % g)
            #           (slot = head, token = start % g + lane).
            b_idx = lane * cutlass.Int32(8) + head_idx % cutlass.Int32(8)
            if beta_g > cutlass.Int32(1):
                b_idx = (sequence_start % beta_g + lane) * heads32 + head_idx
            beta_logit = beta_tile_stage[b_idx].to(cutlass.Float32)
            half = cutlass.Float32(0.5)
            beta_value = cute.math.tanh(beta_logit * half, approx=True) * half + half
        raw_beta_smem[lane] = beta_value
    prims.bar_warp_sync(cute.arch.FULL_MASK)
    if prims.elect_sync():
        prims.mbarrier_arrive_expect_tx(tma_mbar, tma_tx_bytes)
    if prims.elect_sync():
        for segment in cutlass.range_constexpr(RAW_F16_TMA_SEGMENTS):
            tma_coord = (
                cutlass.Int32(segment * RAW_F16_TMA_SWIZZLE_ELEMS),
                global_chunk_start,
                head_idx,
                cutlass.Int32(0),
            )
            smem_offset = segment * RAW_F16_TMA_SEGMENT_ELEMS
            prims.cp_async_bulk_tensor_shared_cta_global(
                raw_q_smem.subview(smem_offset),
                tma_desc_q.get_ptr(),
                tma_coord,
                tma_mbar,
            )
            prims.cp_async_bulk_tensor_shared_cta_global(
                raw_k_smem.subview(smem_offset),
                tma_desc_k.get_ptr(),
                tma_coord,
                tma_mbar,
            )
            prims.cp_async_bulk_tensor_shared_cta_global(
                raw_v_smem.subview(smem_offset),
                tma_desc_v.get_ptr(),
                tma_coord,
                tma_mbar,
            )
            if cutlass.const_expr(not gate_dtype_is_f32(gate_dtype)):
                # 16-bit gate: it rides the SAME 16-bit s128 box/swizzle family
                # as q/k/v, so it folds into this loop (2 x 128 B segments
                # instead of the FP32 path's 4 x 128 B segments).
                prims.cp_async_bulk_tensor_shared_cta_global(
                    raw_gate_smem.subview(smem_offset),
                    tma_desc_gate.get_ptr(),
                    tma_coord,
                    tma_mbar,
                )

        if cutlass.const_expr(gate_dtype_is_f32(gate_dtype)):
            for segment in cutlass.range_constexpr(RAW_F32_TMA_SEGMENTS):
                tma_coord = (
                    cutlass.Int32(segment * RAW_F32_TMA_SWIZZLE_ELEMS),
                    global_chunk_start,
                    head_idx,
                    cutlass.Int32(0),
                )
                smem_offset = segment * RAW_F32_TMA_SEGMENT_ELEMS
                prims.cp_async_bulk_tensor_shared_cta_global(
                    raw_gate_smem.subview(smem_offset),
                    tma_desc_gate.get_ptr(),
                    tma_coord,
                    tma_mbar,
                )


@cute.jit
def tma_transfer_wait(tma_mbar, tma_phase) -> None:
    """Spin until one tma_mbar ring slot's expect_tx transaction completes."""

    while not prims.mbarrier_wait_parity(
        tma_mbar,
        tma_phase,
        prims.MBarrierWait.TRY,
    ):
        pass


@cute.jit
def cg0_zero_tail_raw_operands(
    raw_q_smem,
    raw_k_smem,
    raw_v_smem,
    raw_gate_smem,
    lane,
    chunk_start,
    seqlen,
    input_dtype: cutlass.Constexpr,
    gate_dtype: cutlass.Constexpr,
) -> None:
    """Zero padded tail rows after TMA has produced the raw input tile.

    `gate_dtype` is carried separately from `input_dtype` because the gate has
    its own ABI dtype: a 16-bit gate shares the q/k/v s128 geometry (and can be
    zeroed in the same lane block), an FP32 gate keeps its own wider geometry.
    """

    raw_q_ptr = raw_q_smem.data_ptr()
    raw_k_ptr = raw_k_smem.data_ptr()
    raw_v_ptr = raw_v_smem.data_ptr()
    raw_gate_ptr = raw_gate_smem.data_ptr()
    f16_zero = input_dtype(0.0)
    f16_zero_vec = cutlass.Vector.from_elements(
        (
            f16_zero,
            f16_zero,
            f16_zero,
            f16_zero,
            f16_zero,
            f16_zero,
            f16_zero,
            f16_zero,
        ),
        input_dtype,
    )
    if cutlass.const_expr(gate_dtype_is_f32(gate_dtype)):
        f32_zero = cutlass.Float32(0.0)
        gate_zero_vec = cutlass.Vector.from_elements(
            (f32_zero, f32_zero, f32_zero, f32_zero),
            cutlass.Float32,
        )
    else:
        gate_zero = gate_dtype(0.0)
        gate_zero_vec = cutlass.Vector.from_elements(
            (
                gate_zero,
                gate_zero,
                gate_zero,
                gate_zero,
                gate_zero,
                gate_zero,
                gate_zero,
                gate_zero,
            ),
            gate_dtype,
        )
    for row in cutlass.range_constexpr(BT):
        token_idx = chunk_start + cutlass.Int32(row)
        if token_idx >= seqlen:
            if lane < RAW_F16_TAIL_ZERO_LANES:
                f16_dim_base = lane * RAW_F16_TMA_SWIZZLE_GROUP_ELEMS
                f16_idx = raw_f16_s128_smem_index(row, f16_dim_base)
                (raw_q_ptr + f16_idx).store(
                    f16_zero_vec,
                    alignment=RAW_F16_TMA_SWIZZLE_GROUP_BYTES,
                )
                (raw_k_ptr + f16_idx).store(
                    f16_zero_vec,
                    alignment=RAW_F16_TMA_SWIZZLE_GROUP_BYTES,
                )
                (raw_v_ptr + f16_idx).store(
                    f16_zero_vec,
                    alignment=RAW_F16_TMA_SWIZZLE_GROUP_BYTES,
                )
                if cutlass.const_expr(not gate_dtype_is_f32(gate_dtype)):
                    (raw_gate_ptr + f16_idx).store(
                        gate_zero_vec,
                        alignment=RAW_F16_TMA_SWIZZLE_GROUP_BYTES,
                    )
            if cutlass.const_expr(gate_dtype_is_f32(gate_dtype)):
                if lane < RAW_F32_TAIL_ZERO_LANES:
                    f32_dim_base = lane * RAW_F32_TMA_SWIZZLE_GROUP_ELEMS
                    f32_idx = raw_f32_s128_smem_index(row, f32_dim_base)
                    (raw_gate_ptr + f32_idx).store(
                        gate_zero_vec,
                        alignment=RAW_F32_TMA_SWIZZLE_GROUP_BYTES,
                    )


@cute.jit
def cg0_materialize_decay_operands(
    raw_q_smem,
    raw_k_smem,
    raw_gate_smem,
    gate_exchange_smem,
    a_log_exp,
    dt_bias_value,
    k_inv_smem,
    tcgen05_k_decay_smem,
    tcgen05_q_decay_smem,
    tcgen05_k_restore_smem,
    cg0_k_ready_stage_mbar,
    cg0_k_half_ready_stage_mbar,
    diag_ready_stage_mbar,
    operand_smem_consumed_stage_mbar,
    k_restore_consumed_stage_mbar,
    chunk,
    seqlen,
    input_dtype: cutlass.Constexpr,
    gate_dtype: cutlass.Constexpr,
    SAFE_GATE: cutlass.Constexpr,
    GATE_SCALE_LOG2: cutlass.Constexpr,
    FULL_CHUNKS: cutlass.Constexpr,
    cg0_group_id,
    cg0_local_warp,
    lane,
) -> None:
    """Materialize safe-gate KDA decay operands for one key dimension.

    `FULL_CHUNKS` is the engine peel's per-call selector (True for a guard-free
    interior chunk, False for the peeled partial tail), NOT a compile-time
    specialization: the guard-free interior skips the masked-tail fixup below.
    """

    row_group_start = cg0_local_warp * CG0_TOKEN_ROWS_PER_WARP
    lane_row_group = lane // RAW_F16_TMA_SWIZZLE_GROUP_ELEMS
    lane_in_row_group = lane - lane_row_group * RAW_F16_TMA_SWIZZLE_GROUP_ELEMS
    decay_row = row_group_start + lane_row_group

    raw_q_ptr = raw_q_smem.data_ptr()
    raw_k_ptr = raw_k_smem.data_ptr()
    # The FP32 gate-prefix exchange buffer.  With an FP32 gate the caller
    # passes the raw-gate stage itself (the historical aliasing); with a 16-bit
    # gate it is the dedicated `gate_exchange_smem` tile.
    g_prefix_ptr = gate_exchange_smem.data_ptr()
    k_inv_ptr = k_inv_smem.data_ptr()
    tcgen05_k_decay_ptr = tcgen05_k_decay_smem.data_ptr()
    tcgen05_q_decay_ptr = tcgen05_q_decay_smem.data_ptr()
    tcgen05_k_restore_ptr = tcgen05_k_restore_smem.data_ptr()

    prefix_dim = cg0_local_warp * THREADS_PER_WARP + lane
    g_prefix_regs = cutlass.Array(
        cutlass.Float32,
        BT,
        alignment=16,
    )
    if cutlass.const_expr(SAFE_GATE):
        # Fold tanh's exact 0.5 scale into the chunk-uniform coefficient.
        a_log_exp_half = a_log_exp * cutlass.Float32(0.5)
        for row_pair in cutlass.range_constexpr(BT // 2):
            row0 = row_pair * 2
            row1 = row0 + 1
            # Only the MEMORY FORMAT of the raw gate depends on gate_dtype:
            # a 16-bit gate lands in the q/k s128 geometry and is widened to
            # FP32 right here -- ALL gate arithmetic below stays FP32.
            if cutlass.const_expr(gate_dtype_is_f32(gate_dtype)):
                prefix_idx0 = raw_f32_s128_smem_index(row0, prefix_dim)
                prefix_idx1 = raw_f32_s128_smem_index(row1, prefix_dim)
                gate0 = raw_gate_smem[prefix_idx0]
                gate1 = raw_gate_smem[prefix_idx1]
            else:
                prefix_idx0 = raw_f16_s128_smem_index(row0, prefix_dim)
                prefix_idx1 = raw_f16_s128_smem_index(row1, prefix_dim)
                gate0 = raw_gate_smem[prefix_idx0].to(cutlass.Float32)
                gate1 = raw_gate_smem[prefix_idx1].to(cutlass.Float32)
            gate0 = a_log_exp_half * (gate0 + dt_bias_value)
            gate1 = a_log_exp_half * (gate1 + dt_bias_value)
            gate0 = safe_gate_log2_increment_prehalved(
                gate0,
                SAFE_GATE,
                GATE_SCALE_LOG2,
            )
            gate1 = safe_gate_log2_increment_prehalved(
                gate1,
                SAFE_GATE,
                GATE_SCALE_LOG2,
            )
            gate_pair = cutlass.Vector.from_elements((gate0, gate1), cutlass.Float32)
            g_prefix_regs[row0] = gate_pair[0]
            g_prefix_regs[row1] = gate_pair[1]
    else:
        # FLA-compatible non-safe gate: accept raw gate logits + A_log +
        # dt_bias and compute the log2-domain decay increment in-kernel as
        # ``-exp(A_log) * softplus(raw_gate + dt_bias) * log2(e)``.  a_log_exp
        # already carries ``exp2(A_log * log2(e)) == exp(A_log)`` and the
        # softplus is evaluated in base-2 units (softplus_log2_f32).
        for row in cutlass.range_constexpr(BT):
            if cutlass.const_expr(gate_dtype_is_f32(gate_dtype)):
                prefix_idx = raw_f32_s128_smem_index(row, prefix_dim)
                gate = raw_gate_smem[prefix_idx]
            else:
                prefix_idx = raw_f16_s128_smem_index(row, prefix_dim)
                gate = raw_gate_smem[prefix_idx].to(cutlass.Float32)
            gate = -a_log_exp * softplus_log2_f32(gate + dt_bias_value)
            g_prefix_regs[row] = gate

    if cutlass.const_expr(not FULL_CHUNKS):
        # tail-block peeling: interior (full) chunks run the lean
        # unconditional gate stream above; only a sequence's genuinely
        # partial tail chunk (warp-uniform runtime condition, one compare
        # per chunk) pays the masked zeroing fixup below.
        tail_valid_rows = seqlen - chunk * cutlass.Int32(BT)
        if tail_valid_rows < cutlass.Int32(BT):
            tail_mask_pt = cutlass.vector.create_mask([BT], [tail_valid_rows])
            for row_pair_pt in cutlass.range_constexpr(BT // 2):
                row0_pt = row_pair_pt * 2
                row1_pt = row0_pt + 1
                gate_pair_pt = cutlass.Vector.from_elements(
                    (g_prefix_regs[row0_pt], g_prefix_regs[row1_pt]),
                    cutlass.Float32,
                )
                gate_pair_pt = cutlass.vector.where(
                    tail_mask_pt[row0_pt : row1_pt + 1], gate_pair_pt, 0.0
                )
                g_prefix_regs[row0_pt] = gate_pair_pt[0]
                g_prefix_regs[row1_pt] = gate_pair_pt[1]

    prefix_acc = cutlass.Float32(0.0)
    for row_pair in cutlass.range_constexpr(BT // 2):
        row0 = row_pair * 2
        row1 = row0 + 1
        # The scalar scan has the same dependency depth as packed fadd2 but
        # avoids the register copy needed to construct its first input pair.
        prefix0 = prefix_acc + g_prefix_regs[row0]
        prefix1 = prefix0 + g_prefix_regs[row1]
        g_prefix_regs[row0] = prefix0
        g_prefix_regs[row1] = prefix1
        prefix_acc = prefix1

    for row in cutlass.range_constexpr(BT):
        g_prefix_regs[row] = cute.math.exp2(g_prefix_regs[row], fastmath=True)

    for row in cutlass.range_constexpr(BT):
        prefix_idx = raw_f32_exchange_smem_index(row, prefix_dim)
        gate_exchange_smem[prefix_idx] = g_prefix_regs[row]

    cg0_sync(cg0_group_id)
    diag_ready_arrive(diag_ready_stage_mbar)

    k_inv_regs = cutlass.Array(
        input_dtype,
        2 * RAW_F16_TMA_SWIZZLE_GROUP_ELEMS,
        alignment=16,
    )
    k_restore_all_regs = cutlass.Array(
        input_dtype,
        2 * RAW_F16_TMA_SWIZZLE_GROUP_ELEMS,
        alignment=16,
    )
    raw_q_regs = cutlass.Array(
        cutlass.Float32,
        2 * RAW_F16_TMA_SWIZZLE_GROUP_ELEMS,
        alignment=16,
    )
    raw_k_regs = cutlass.Array(
        cutlass.Float32,
        2 * RAW_F16_TMA_SWIZZLE_GROUP_ELEMS,
        alignment=16,
    )
    q_sum_sq = cutlass.Float32(0.0)
    k_sum_sq = cutlass.Float32(0.0)
    for dim_half in cutlass.range_constexpr(2):
        dim_base = (
            dim_half * (DK // 2) + lane_in_row_group * RAW_F16_TMA_SWIZZLE_GROUP_ELEMS
        )
        reg_base = dim_half * RAW_F16_TMA_SWIZZLE_GROUP_ELEMS
        raw_f16_idx = raw_f16_s128_smem_index(decay_row, dim_base)
        raw_q_vec = (raw_q_ptr + raw_f16_idx).load(
            count=RAW_F16_TMA_SWIZZLE_GROUP_ELEMS,
            alignment=RAW_F16_TMA_SWIZZLE_GROUP_BYTES,
        )
        raw_k_vec = (raw_k_ptr + raw_f16_idx).load(
            count=RAW_F16_TMA_SWIZZLE_GROUP_ELEMS,
            alignment=RAW_F16_TMA_SWIZZLE_GROUP_BYTES,
        )
        raw_q_vec_f32 = raw_q_vec.to(cutlass.Float32)
        raw_k_vec_f32 = raw_k_vec.to(cutlass.Float32)
        for dim_offset in cutlass.range_constexpr(RAW_F16_TMA_SWIZZLE_GROUP_ELEMS):
            q_val = raw_q_vec_f32[dim_offset]
            k_val = raw_k_vec_f32[dim_offset]
            raw_q_regs[reg_base + dim_offset] = q_val
            raw_k_regs[reg_base + dim_offset] = k_val
            q_sum_sq = q_sum_sq + q_val * q_val
            k_sum_sq = k_sum_sq + k_val * k_val

    q_sum_sq = warp_group_sum_8(q_sum_sq)
    k_sum_sq = warp_group_sum_8(k_sum_sq)
    norm_floor_sq = cutlass.Float32(L2_NORM_EPS * L2_NORM_EPS)
    q_inv_norm = cute.math.rsqrt(
        cute.math.max(q_sum_sq, norm_floor_sq, ftz=True),
        fastmath=True,
    )
    k_inv_norm = cute.math.rsqrt(
        cute.math.max(k_sum_sq, norm_floor_sq, ftz=True),
        fastmath=True,
    )

    exp_g_regs = cutlass.Array(
        cutlass.Float32,
        2 * RAW_F16_TMA_SWIZZLE_GROUP_ELEMS,
        alignment=16,
    )
    exp_g_last_regs = cutlass.Array(
        cutlass.Float32,
        2 * RAW_F16_TMA_SWIZZLE_GROUP_ELEMS,
        alignment=16,
    )
    for dim_half in cutlass.range_constexpr(2):
        dim_base = (
            dim_half * (DK // 2) + lane_in_row_group * RAW_F16_TMA_SWIZZLE_GROUP_ELEMS
        )
        reg_base = dim_half * RAW_F16_TMA_SWIZZLE_GROUP_ELEMS
        exp_neg_g_regs = cutlass.Array(
            cutlass.Float32,
            RAW_F16_TMA_SWIZZLE_GROUP_ELEMS,
            alignment=16,
        )
        for f32_group in cutlass.range_constexpr(
            RAW_F16_TMA_SWIZZLE_GROUP_ELEMS // RAW_F32_TMA_SWIZZLE_GROUP_ELEMS
        ):
            f32_dim_base = dim_base + f32_group * RAW_F32_TMA_SWIZZLE_GROUP_ELEMS
            g_prefix_idx = raw_f32_exchange_smem_index(decay_row, f32_dim_base)
            exp_g_vec = (g_prefix_ptr + g_prefix_idx).load(
                count=RAW_F32_TMA_SWIZZLE_GROUP_ELEMS,
                alignment=RAW_F32_TMA_SWIZZLE_GROUP_BYTES,
            )
            exp_g_last_idx = raw_f32_exchange_smem_index(BT - 1, f32_dim_base)
            exp_g_last_vec = (g_prefix_ptr + exp_g_last_idx).load(
                count=RAW_F32_TMA_SWIZZLE_GROUP_ELEMS,
                alignment=RAW_F32_TMA_SWIZZLE_GROUP_BYTES,
            )
            half_reg_base = f32_group * RAW_F32_TMA_SWIZZLE_GROUP_ELEMS
            f32_reg_base = reg_base + half_reg_base
            exp_g_regs[f32_reg_base] = exp_g_vec[0]
            exp_g_regs[f32_reg_base + 1] = exp_g_vec[1]
            exp_g_regs[f32_reg_base + 2] = exp_g_vec[2]
            exp_g_regs[f32_reg_base + 3] = exp_g_vec[3]
            exp_neg_g_regs[half_reg_base] = cute.math.rcp(
                exp_g_vec[0], approx=True, ftz=True
            )
            exp_neg_g_regs[half_reg_base + 1] = cute.math.rcp(
                exp_g_vec[1], approx=True, ftz=True
            )
            exp_neg_g_regs[half_reg_base + 2] = cute.math.rcp(
                exp_g_vec[2], approx=True, ftz=True
            )
            exp_neg_g_regs[half_reg_base + 3] = cute.math.rcp(
                exp_g_vec[3], approx=True, ftz=True
            )
            exp_g_last_regs[f32_reg_base] = exp_g_last_vec[0]
            exp_g_last_regs[f32_reg_base + 1] = exp_g_last_vec[1]
            exp_g_last_regs[f32_reg_base + 2] = exp_g_last_vec[2]
            exp_g_last_regs[f32_reg_base + 3] = exp_g_last_vec[3]

        k_decay_vec_regs = cutlass.Array(
            input_dtype,
            RAW_F16_TMA_SWIZZLE_GROUP_ELEMS,
            alignment=RAW_F16_TMA_SWIZZLE_GROUP_BYTES,
        )
        for pair_idx in cutlass.range_constexpr(RAW_F16_TMA_SWIZZLE_GROUP_ELEMS // 2):
            dim0 = pair_idx * 2
            dim1 = dim0 + 1
            raw_reg_idx0 = reg_base + dim0
            raw_reg_idx1 = reg_base + dim1
            k_value0, k_value1 = fmul2(
                (raw_k_regs[raw_reg_idx0], raw_k_regs[raw_reg_idx1]),
                (k_inv_norm, k_inv_norm),
            )
            k_decay0, k_decay1 = fmul2(
                (k_value0, k_value1),
                (exp_g_regs[raw_reg_idx0], exp_g_regs[raw_reg_idx1]),
            )
            k_inv0, k_inv1 = fmul2(
                (k_value0, k_value1),
                (exp_neg_g_regs[dim0], exp_neg_g_regs[dim1]),
            )
            k_inv_regs[reg_base + dim0] = k_inv0.to(input_dtype)
            k_inv_regs[reg_base + dim1] = k_inv1.to(input_dtype)
            k_restore0, k_restore1 = fmul2(
                (k_inv0, k_inv1),
                (
                    exp_g_last_regs[reg_base + dim0],
                    exp_g_last_regs[reg_base + dim1],
                ),
            )
            k_restore_all_regs[reg_base + dim0] = k_restore0.to(input_dtype)
            k_restore_all_regs[reg_base + dim1] = k_restore1.to(input_dtype)
            k_decay_vec_regs[dim0] = k_decay0.to(input_dtype)
            k_decay_vec_regs[dim1] = k_decay1.to(input_dtype)

        k_inv_vec = cutlass.Vector.from_elements(
            (
                k_inv_regs[reg_base],
                k_inv_regs[reg_base + 1],
                k_inv_regs[reg_base + 2],
                k_inv_regs[reg_base + 3],
                k_inv_regs[reg_base + 4],
                k_inv_regs[reg_base + 5],
                k_inv_regs[reg_base + 6],
                k_inv_regs[reg_base + 7],
            ),
            input_dtype,
        )
        k_decay_vec = cutlass.Vector.from_elements(
            (
                k_decay_vec_regs[0],
                k_decay_vec_regs[1],
                k_decay_vec_regs[2],
                k_decay_vec_regs[3],
                k_decay_vec_regs[4],
                k_decay_vec_regs[5],
                k_decay_vec_regs[6],
                k_decay_vec_regs[7],
            ),
            input_dtype,
        )
        if cutlass.const_expr(dim_half == 0):
            operand_smem_consumed_phase = ((chunk // DECAY_STAGE_COUNT) + 1) % 2
            operand_smem_consumed_wait(
                operand_smem_consumed_stage_mbar,
                operand_smem_consumed_phase,
            )
        k_inv_swizzled_idx = k_inv_s128_smem_index(decay_row, dim_base)
        (k_inv_ptr + k_inv_swizzled_idx).store(
            k_inv_vec,
            alignment=RAW_F16_TMA_SWIZZLE_GROUP_BYTES,
        )
        decay_storage_dim_base = tcgen05_decay_b_key_storage_dim_runtime(
            decay_row,
            dim_base,
        )
        decay_linear_idx_base = decay_row * DK + decay_storage_dim_base
        decay_swizzled_idx_base = tcgen05_swizzle_128b_elem_index(
            decay_linear_idx_base,
            TCGEN05_F16_ELEM_BYTES,
            BT,
        )
        (tcgen05_k_decay_ptr + decay_swizzled_idx_base).store(
            k_decay_vec,
            alignment=RAW_F16_TMA_SWIZZLE_GROUP_BYTES,
        )
        if cutlass.const_expr(dim_half == 0):
            # Publish the first half-DK of k_inv/k_decay early: warp 12's
            # KK K-blocks 0..3 only read key dims [0, 64).
            cg0_k_ready_arrive(cg0_k_half_ready_stage_mbar)
    cg0_k_ready_arrive(cg0_k_ready_stage_mbar)

    for dim_half in cutlass.range_constexpr(2):
        dim_base = (
            dim_half * (DK // 2) + lane_in_row_group * RAW_F16_TMA_SWIZZLE_GROUP_ELEMS
        )
        reg_base = dim_half * RAW_F16_TMA_SWIZZLE_GROUP_ELEMS
        q_decay_vec_regs = cutlass.Array(
            input_dtype,
            RAW_F16_TMA_SWIZZLE_GROUP_ELEMS,
            alignment=RAW_F16_TMA_SWIZZLE_GROUP_BYTES,
        )
        for pair_idx in cutlass.range_constexpr(RAW_F16_TMA_SWIZZLE_GROUP_ELEMS // 2):
            dim0 = pair_idx * 2
            dim1 = dim0 + 1
            raw_reg_idx0 = reg_base + dim0
            raw_reg_idx1 = reg_base + dim1
            q_value0, q_value1 = fmul2(
                (raw_q_regs[raw_reg_idx0], raw_q_regs[raw_reg_idx1]),
                (q_inv_norm, q_inv_norm),
            )
            q_decay0, q_decay1 = fmul2(
                (q_value0, q_value1),
                (exp_g_regs[raw_reg_idx0], exp_g_regs[raw_reg_idx1]),
            )
            q_decay_vec_regs[dim0] = q_decay0.to(input_dtype)
            q_decay_vec_regs[dim1] = q_decay1.to(input_dtype)

        q_decay_vec = cutlass.Vector.from_elements(
            (
                q_decay_vec_regs[0],
                q_decay_vec_regs[1],
                q_decay_vec_regs[2],
                q_decay_vec_regs[3],
                q_decay_vec_regs[4],
                q_decay_vec_regs[5],
                q_decay_vec_regs[6],
                q_decay_vec_regs[7],
            ),
            input_dtype,
        )
        decay_storage_dim_base = tcgen05_decay_b_key_storage_dim_runtime(
            decay_row,
            dim_base,
        )
        decay_linear_idx_base = decay_row * DK + decay_storage_dim_base
        decay_swizzled_idx_base = tcgen05_swizzle_128b_elem_index(
            decay_linear_idx_base,
            TCGEN05_F16_ELEM_BYTES,
            BT,
        )
        (tcgen05_q_decay_ptr + decay_swizzled_idx_base).store(
            q_decay_vec,
            alignment=RAW_F16_TMA_SWIZZLE_GROUP_BYTES,
        )

    k_restore_consumed_phase = ((chunk // DECAY_STAGE_COUNT) + 1) % 2
    operand_smem_consumed_wait(
        k_restore_consumed_stage_mbar,
        k_restore_consumed_phase,
    )

    for dim_half in cutlass.range_constexpr(2):
        dim_base = (
            dim_half * (DK // 2) + lane_in_row_group * RAW_F16_TMA_SWIZZLE_GROUP_ELEMS
        )
        reg_base = dim_half * RAW_F16_TMA_SWIZZLE_GROUP_ELEMS
        storage_row = decay_row ^ TCGEN05_SW32_BT_HALF_XOR
        k_restore_idx = raw_f16_s128_smem_index(storage_row, dim_base)
        k_restore_vec = cutlass.Vector.from_elements(
            (
                k_restore_all_regs[reg_base],
                k_restore_all_regs[reg_base + 1],
                k_restore_all_regs[reg_base + 2],
                k_restore_all_regs[reg_base + 3],
                k_restore_all_regs[reg_base + 4],
                k_restore_all_regs[reg_base + 5],
                k_restore_all_regs[reg_base + 6],
                k_restore_all_regs[reg_base + 7],
            ),
            input_dtype,
        )
        (tcgen05_k_restore_ptr + k_restore_idx).store(
            k_restore_vec,
            alignment=RAW_F16_TMA_SWIZZLE_GROUP_BYTES,
        )


@cute.jit
def ptx_mma_m16n8k16_b16_f32(
    a0,
    a1,
    a2,
    a3,
    b0,
    b1,
    c0,
    c1,
    c2,
    c3,
    input_dtype: cutlass.Constexpr,
):
    """Issue `mma.sync.aligned.m16n8k16.row.col.f32.{f16|bf16}.{f16|bf16}.f32`."""

    if cutlass.const_expr(
        input_dtype != cutlass.Float16 and input_dtype != cutlass.BFloat16
    ):
        raise TypeError(f"Invalid auxiliary-MMA input dtype: {input_dtype}")
    input_tag = "f16" if cutlass.const_expr(input_dtype == cutlass.Float16) else "bf16"

    return cute.arch.inline_ptx(
        f"mma.sync.aligned.m16n8k16.row.col.f32.{input_tag}.{input_tag}.f32"
        " {$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13};",
        write_only_types=[
            cutlass.Float32,
            cutlass.Float32,
            cutlass.Float32,
            cutlass.Float32,
        ],
        read_only_args=[a0, a1, a2, a3, b0, b1, c0, c1, c2, c3],
    )


@cute.jit
def super_mma_accumulator_row(lane, accum_idx: cutlass.Constexpr) -> cutlass.Int32:
    """Row coordinate for one `m16n8k16` accumulator element."""

    row = lane // 4
    if cutlass.const_expr(accum_idx >= 2):
        row = row + cutlass.Int32(8)
    return row


@cute.jit
def super_mma_accumulator_col(
    lane,
    n_block: cutlass.Constexpr,
    accum_idx: cutlass.Constexpr,
) -> cutlass.Int32:
    """Column coordinate for one `m16n8k16` accumulator element."""

    col = n_block * SUPER_MMA_ATOM_N + 2 * (lane % 4)
    if cutlass.const_expr((accum_idx % 2) == 1):
        col = col + cutlass.Int32(1)
    return col


@cute.jit
def super_mma_store_pairwise_tile_stmatrix_x4(
    pairwise_smem,
    dst_offset: cutlass.Constexpr,
    lane,
    n0_acc,
    n1_acc,
    input_dtype: cutlass.Constexpr,
) -> None:
    """Store one 16x16 pairwise tile through `stmatrix.m8n8.x4.b16`."""

    prims.stmatrix(
        pairwise_stmatrix_m8n8x4_ptr(pairwise_smem, dst_offset, lane),
        [
            pack_input_b16x2_to_i32(n0_acc[0], n0_acc[1], input_dtype),
            pack_input_b16x2_to_i32(n0_acc[2], n0_acc[3], input_dtype),
            pack_input_b16x2_to_i32(n1_acc[0], n1_acc[1], input_dtype),
            pack_input_b16x2_to_i32(n1_acc[2], n1_acc[3], input_dtype),
        ],
        prims.MMALayout.ROW,
        shape=prims.StoreShape.M8N8,
    )


@cute.jit
def super_mma_strict_lower_beta_value(
    value: cutlass.Float32,
    row_coord,
    col_coord,
    beta_scale: cutlass.Float32,
) -> cutlass.Float32:
    """Apply `tril(x, -1) * beta[row]` to one pairwise accumulator value."""

    lower = value if row_coord > col_coord else cutlass.Float32(0.0)
    return lower * beta_scale


@cute.jit
def super_mma_build_l_fragment(
    raw_beta_smem,
    lane,
    n0_acc,
    n1_acc,
    l_frag,
    input_dtype: cutlass.Constexpr,
) -> None:
    """Build the packed `L = beta * tril(KK, -1)` registers into `l_frag`."""

    row_lo = super_mma_accumulator_row(lane, 0)
    row_hi = super_mma_accumulator_row(lane, 2)
    n0_col0 = super_mma_accumulator_col(lane, 0, 0)
    n0_col1 = super_mma_accumulator_col(lane, 0, 1)
    n0_col2 = super_mma_accumulator_col(lane, 0, 2)
    n0_col3 = super_mma_accumulator_col(lane, 0, 3)
    n1_col0 = super_mma_accumulator_col(lane, 1, 0)
    n1_col1 = super_mma_accumulator_col(lane, 1, 1)
    n1_col2 = super_mma_accumulator_col(lane, 1, 2)
    n1_col3 = super_mma_accumulator_col(lane, 1, 3)
    beta_lo = raw_beta_smem[row_lo].to(cutlass.Float32)
    beta_hi = raw_beta_smem[row_hi].to(cutlass.Float32)

    l_frag[0] = pack_input_b16x2_to_i32(
        super_mma_strict_lower_beta_value(n0_acc[0], row_lo, n0_col0, beta_lo),
        super_mma_strict_lower_beta_value(n0_acc[1], row_lo, n0_col1, beta_lo),
        input_dtype,
    )
    l_frag[1] = pack_input_b16x2_to_i32(
        super_mma_strict_lower_beta_value(n0_acc[2], row_hi, n0_col2, beta_hi),
        super_mma_strict_lower_beta_value(n0_acc[3], row_hi, n0_col3, beta_hi),
        input_dtype,
    )
    l_frag[2] = pack_input_b16x2_to_i32(
        super_mma_strict_lower_beta_value(n1_acc[0], row_lo, n1_col0, beta_lo),
        super_mma_strict_lower_beta_value(n1_acc[1], row_lo, n1_col1, beta_lo),
        input_dtype,
    )
    l_frag[3] = pack_input_b16x2_to_i32(
        super_mma_strict_lower_beta_value(n1_acc[2], row_hi, n1_col2, beta_hi),
        super_mma_strict_lower_beta_value(n1_acc[3], row_hi, n1_col3, beta_hi),
        input_dtype,
    )


@cute.jit
def super_mma_sw128_decay_ldmatrix_index(
    row_coord,
    col_offset,
    k_block: cutlass.Constexpr,
):
    """Return the SW128 row-segment index used by the auxiliary-MMA warp.

    The decay tile is stored in the tcgen05 B-operand K-box-major layout with
    only the constant half-atom interleave; the tcgen05 descriptor reads the
    standard SW128 layout, so no row-dependent storage K-block xor exists (the
    former ``(row & 2) * K_ATOM`` term here matched a write-side xor that the
    tcgen05 state MMA never saw — both are removed together).  ldmatrix reads
    the 16B row segment at that physical K-block.
    """

    key_mask = cutlass.Int32(TCGEN05_F16_K_ATOM // 2)
    logical_key = k_block * TCGEN05_F16_K_ATOM + col_offset
    storage_key = logical_key ^ key_mask
    elems_per_128b = cutlass.Int32(TCGEN05_SW128_BYTES // TCGEN05_F16_ELEM_BYTES)
    storage_slice = storage_key // elems_per_128b
    key_in_slice = storage_key - storage_slice * elems_per_128b
    storage_phase = key_in_slice // cutlass.Int32(TCGEN05_F16_K_ATOM)
    storage_col_offset = key_in_slice - storage_phase * cutlass.Int32(
        TCGEN05_F16_K_ATOM
    )
    row_byte = row_coord * cutlass.Int32(TCGEN05_SW128_BYTES)
    phase_byte = storage_phase * cutlass.Int32(
        TCGEN05_F16_K_ATOM * TCGEN05_F16_ELEM_BYTES
    )
    col_byte = storage_col_offset * cutlass.Int32(TCGEN05_F16_ELEM_BYTES)
    byte_in_slice = row_byte + phase_byte + col_byte
    swizzle_mask = (row_coord & cutlass.Int32(7)) << 4
    elems_per_slice = cutlass.Int32(BT) * elems_per_128b
    return storage_slice * elems_per_slice + (
        (byte_in_slice ^ swizzle_mask) // cutlass.Int32(TCGEN05_F16_ELEM_BYTES)
    )


@cute.jit
def super_mma_load_decay_lhs_fragment(
    tcgen05_decay_smem,
    lane,
    k_block: cutlass.Constexpr,
):
    """Load the A operand fragment for one `m16n8k16` K phase.

    A is logically `[BT, DK]` and physically stored in the tcgen05 SW128 decay
    operand layout. The register order matches the direct-Q fragment order used
    by the CUTLASS primitives FMHA examples: rows 0/8 crossed with K cols 0/8.
    """

    lane_div8 = lane // 8
    lane_mod8 = lane % 8
    row_offset = cutlass.Int32(8) if (lane_div8 % 2) else cutlass.Int32(0)
    col_offset = cutlass.Int32(8) if (lane_div8 // 2) else cutlass.Int32(0)
    row_coord = lane_mod8 + row_offset
    swizzled_idx = super_mma_sw128_decay_ldmatrix_index(
        row_coord,
        col_offset,
        k_block,
    )
    ptr = tcgen05_decay_smem.subview(swizzled_idx)
    return prims.ldmatrix(ptr.data_ptr(), 4, prims.MMALayout.ROW)


@cute.jit
def super_mma_load_k_inv_rhs_fragment(
    k_inv_smem,
    lane,
    k_block: cutlass.Constexpr,
):
    """Load the B operand fragment for one `m16n8k16` K phase.

    B is stored token-major `[BT, DK]`, which is column-major for the logical
    `[DK, BT]` MMA RHS. The `ldmatrix.x4` return is split into two N fragments:
    registers 0/1 for columns 0..7, and registers 2/3 for columns 8..15.
    """

    lane_div8 = lane // 8
    lane_div16 = lane // 16
    lane_mod8 = lane % 8
    row_offset = cutlass.Int32(8) if lane_div16 else cutlass.Int32(0)
    col_offset = cutlass.Int32(8) if (lane_div8 % 2) else cutlass.Int32(0)
    row_coord = lane_mod8 + row_offset
    col_coord = k_block * SUPER_MMA_ATOM_K + col_offset
    ptr = k_inv_smem.subview(k_inv_s128_smem_index(row_coord, col_coord))
    return prims.ldmatrix(ptr.data_ptr(), 4, prims.MMALayout.ROW)


@cute.jit
def super_mma_qk_causal_value(
    value: cutlass.Float32,
    lane,
    n_block: cutlass.Constexpr,
    accum_idx: cutlass.Constexpr,
) -> cutlass.Float32:
    """Zero an accumulator outside the inclusive-lower QK tile."""

    row_coord = super_mma_accumulator_row(lane, accum_idx)
    col_coord = super_mma_accumulator_col(lane, n_block, accum_idx)
    return value if row_coord >= col_coord else cutlass.Float32(0.0)


@cute.jit
def super_mma_stage_kk_blocks(
    tcgen05_k_decay_smem,
    k_inv_smem,
    lane,
    input_dtype: cutlass.Constexpr,
    k_block_lo: cutlass.Constexpr,
    k_block_hi: cutlass.Constexpr,
    kk_n0_acc,
    kk_n1_acc,
):
    """Accumulate KK m16n8k16 K-blocks [k_block_lo, k_block_hi).

    The KK product is split at the half-DK boundary so warp 12 can run
    K-blocks 0..3 on the `cg0_k_half_ready` arrival (CG0's dim_half==0
    stores cover key dims [0, 64): k_inv segment 0 and, because the decay
    storage-key xor mask is 8 or 40, k_decay SW128 slice 0) while CG0 is
    still staging the second half.
    """

    for k_block_off in cutlass.range_constexpr(k_block_hi - k_block_lo):
        k_block = k_block_lo + k_block_off
        rhs_vec = super_mma_load_k_inv_rhs_fragment(
            k_inv_smem,
            lane,
            k_block,
        )
        kk_lhs_vec = super_mma_load_decay_lhs_fragment(
            tcgen05_k_decay_smem,
            lane,
            k_block,
        )

        kk_n0_d0, kk_n0_d1, kk_n0_d2, kk_n0_d3 = ptx_mma_m16n8k16_b16_f32(
            kk_lhs_vec[0],
            kk_lhs_vec[1],
            kk_lhs_vec[2],
            kk_lhs_vec[3],
            rhs_vec[0],
            rhs_vec[1],
            kk_n0_acc[0],
            kk_n0_acc[1],
            kk_n0_acc[2],
            kk_n0_acc[3],
            input_dtype,
        )
        kk_n0_acc[0] = kk_n0_d0
        kk_n0_acc[1] = kk_n0_d1
        kk_n0_acc[2] = kk_n0_d2
        kk_n0_acc[3] = kk_n0_d3
        kk_n1_d0, kk_n1_d1, kk_n1_d2, kk_n1_d3 = ptx_mma_m16n8k16_b16_f32(
            kk_lhs_vec[0],
            kk_lhs_vec[1],
            kk_lhs_vec[2],
            kk_lhs_vec[3],
            rhs_vec[2],
            rhs_vec[3],
            kk_n1_acc[0],
            kk_n1_acc[1],
            kk_n1_acc[2],
            kk_n1_acc[3],
            input_dtype,
        )
        kk_n1_acc[0] = kk_n1_d0
        kk_n1_acc[1] = kk_n1_d1
        kk_n1_acc[2] = kk_n1_d2
        kk_n1_acc[3] = kk_n1_d3


@cute.jit
def super_mma_stage_qk(
    tcgen05_q_decay_smem,
    k_inv_smem,
    pairwise_smem,
    lane,
    input_dtype: cutlass.Constexpr,
) -> None:
    """Produce the causal QK tile consumed by the qkv tcgen05 MMA."""

    qk_n0_acc = cutlass.Array(
        cutlass.Float32,
        SUPER_MMA_ACCUMULATORS_PER_LANE,
        alignment=16,
    )
    qk_n1_acc = cutlass.Array(
        cutlass.Float32,
        SUPER_MMA_ACCUMULATORS_PER_LANE,
        alignment=16,
    )
    for accum_idx in cutlass.range_constexpr(SUPER_MMA_ACCUMULATORS_PER_LANE):
        qk_n0_acc[accum_idx] = cutlass.Float32(0.0)
        qk_n1_acc[accum_idx] = cutlass.Float32(0.0)

    for k_block in cutlass.range_constexpr(SUPER_MMA_K_BLOCKS):
        rhs_vec = super_mma_load_k_inv_rhs_fragment(
            k_inv_smem,
            lane,
            k_block,
        )
        qk_lhs_vec = super_mma_load_decay_lhs_fragment(
            tcgen05_q_decay_smem,
            lane,
            k_block,
        )

        qk_n0_d0, qk_n0_d1, qk_n0_d2, qk_n0_d3 = ptx_mma_m16n8k16_b16_f32(
            qk_lhs_vec[0],
            qk_lhs_vec[1],
            qk_lhs_vec[2],
            qk_lhs_vec[3],
            rhs_vec[0],
            rhs_vec[1],
            qk_n0_acc[0],
            qk_n0_acc[1],
            qk_n0_acc[2],
            qk_n0_acc[3],
            input_dtype,
        )
        qk_n0_acc[0] = qk_n0_d0
        qk_n0_acc[1] = qk_n0_d1
        qk_n0_acc[2] = qk_n0_d2
        qk_n0_acc[3] = qk_n0_d3
        qk_n1_d0, qk_n1_d1, qk_n1_d2, qk_n1_d3 = ptx_mma_m16n8k16_b16_f32(
            qk_lhs_vec[0],
            qk_lhs_vec[1],
            qk_lhs_vec[2],
            qk_lhs_vec[3],
            rhs_vec[2],
            rhs_vec[3],
            qk_n1_acc[0],
            qk_n1_acc[1],
            qk_n1_acc[2],
            qk_n1_acc[3],
            input_dtype,
        )
        qk_n1_acc[0] = qk_n1_d0
        qk_n1_acc[1] = qk_n1_d1
        qk_n1_acc[2] = qk_n1_d2
        qk_n1_acc[3] = qk_n1_d3

    qk_n0_acc[0] = super_mma_qk_causal_value(qk_n0_acc[0], lane, 0, 0)
    qk_n0_acc[1] = super_mma_qk_causal_value(qk_n0_acc[1], lane, 0, 1)
    qk_n0_acc[2] = super_mma_qk_causal_value(qk_n0_acc[2], lane, 0, 2)
    qk_n0_acc[3] = super_mma_qk_causal_value(qk_n0_acc[3], lane, 0, 3)
    qk_n1_acc[0] = super_mma_qk_causal_value(qk_n1_acc[0], lane, 1, 0)
    qk_n1_acc[1] = super_mma_qk_causal_value(qk_n1_acc[1], lane, 1, 1)
    qk_n1_acc[2] = super_mma_qk_causal_value(qk_n1_acc[2], lane, 1, 2)
    qk_n1_acc[3] = super_mma_qk_causal_value(qk_n1_acc[3], lane, 1, 3)

    super_mma_store_pairwise_tile_stmatrix_x4(
        pairwise_smem,
        PAIRWISE_SMEM_QK_OFFSET,
        lane,
        qk_n0_acc,
        qk_n1_acc,
        input_dtype,
    )


@cute.jit
def super_mma_pairwise_product_from_ab_regs(
    lhs_frag,
    rhs_frag,
    n0_out,
    n1_out,
    input_dtype: cutlass.Constexpr,
) -> None:
    """Compute `lhs @ rhs` accumulators from packed A-layout fragments.

    Results land in the caller-provided `n0_out`/`n1_out` accumulator arrays.
    """

    rhs_b0 = movmatrix_b16(rhs_frag[0])
    rhs_b1 = movmatrix_b16(rhs_frag[1])
    rhs_b2 = movmatrix_b16(rhs_frag[2])
    rhs_b3 = movmatrix_b16(rhs_frag[3])
    zero_acc = cutlass.Array(
        cutlass.Float32,
        SUPER_MMA_ACCUMULATORS_PER_LANE,
        alignment=16,
    )
    for accum_idx in cutlass.range_constexpr(SUPER_MMA_ACCUMULATORS_PER_LANE):
        zero_acc[accum_idx] = cutlass.Float32(0.0)
    n0_out[0], n0_out[1], n0_out[2], n0_out[3] = ptx_mma_m16n8k16_b16_f32(
        lhs_frag[0],
        lhs_frag[1],
        lhs_frag[2],
        lhs_frag[3],
        rhs_b0,
        rhs_b1,
        zero_acc[0],
        zero_acc[1],
        zero_acc[2],
        zero_acc[3],
        input_dtype,
    )
    n1_out[0], n1_out[1], n1_out[2], n1_out[3] = ptx_mma_m16n8k16_b16_f32(
        lhs_frag[0],
        lhs_frag[1],
        lhs_frag[2],
        lhs_frag[3],
        rhs_b2,
        rhs_b3,
        zero_acc[0],
        zero_acc[1],
        zero_acc[2],
        zero_acc[3],
        input_dtype,
    )


@cute.jit
def super_mma_initial_inverse_from_fragment(
    l_values,
    lane,
    n_block: cutlass.Constexpr,
    accum_idx: cutlass.Constexpr,
) -> cutlass.Float32:
    """Return one accumulator element of `I - L` from packed L registers."""

    row_coord = super_mma_accumulator_row(lane, accum_idx)
    col_coord = super_mma_accumulator_col(lane, n_block, accum_idx)
    l_value_idx: cutlass.Constexpr[int] = (
        n_block * SUPER_MMA_ACCUMULATORS_PER_LANE + accum_idx
    )
    return pairwise_eye(row_coord, col_coord) - l_values[l_value_idx]


@cute.jit
def super_mma_pack_pairwise_accumulator(
    n0_acc,
    n1_acc,
    out_frag,
    input_dtype: cutlass.Constexpr,
) -> None:
    """Round one 16x16 pairwise accumulator tile into an A-layout b16 fragment.

    The packed registers land in the caller-provided `out_frag` array.
    """

    out_frag[0] = pack_input_b16x2_to_i32(n0_acc[0], n0_acc[1], input_dtype)
    out_frag[1] = pack_input_b16x2_to_i32(n0_acc[2], n0_acc[3], input_dtype)
    out_frag[2] = pack_input_b16x2_to_i32(n1_acc[0], n1_acc[1], input_dtype)
    out_frag[3] = pack_input_b16x2_to_i32(n1_acc[2], n1_acc[3], input_dtype)


@cute.jit
def super_mma_square_pairwise_fragment(
    src_frag,
    out_frag,
    input_dtype: cutlass.Constexpr,
) -> None:
    """Pack b16 registers for `src @ src` into `out_frag` without SMEM roundtrip."""

    n0_acc = cutlass.Array(
        cutlass.Float32,
        SUPER_MMA_ACCUMULATORS_PER_LANE,
        alignment=16,
    )
    n1_acc = cutlass.Array(
        cutlass.Float32,
        SUPER_MMA_ACCUMULATORS_PER_LANE,
        alignment=16,
    )
    super_mma_pairwise_product_from_ab_regs(
        src_frag,
        src_frag,
        n0_acc,
        n1_acc,
        input_dtype,
    )
    super_mma_pack_pairwise_accumulator(
        n0_acc,
        n1_acc,
        out_frag,
        input_dtype,
    )


@cute.jit
def super_mma_update_inverse_with_power_regs(
    rhs_frag,
    n0_acc,
    n1_acc,
    input_dtype: cutlass.Constexpr,
) -> None:
    """Update `inv += inv @ Lpow` in place, keeping `inv` in registers."""

    a_frag = cutlass.Array(
        cutlass.Int32,
        SUPER_MMA_ACCUMULATORS_PER_LANE,
        alignment=16,
    )
    super_mma_pack_pairwise_accumulator(
        n0_acc,
        n1_acc,
        a_frag,
        input_dtype,
    )
    p0_acc = cutlass.Array(
        cutlass.Float32,
        SUPER_MMA_ACCUMULATORS_PER_LANE,
        alignment=16,
    )
    p1_acc = cutlass.Array(
        cutlass.Float32,
        SUPER_MMA_ACCUMULATORS_PER_LANE,
        alignment=16,
    )
    super_mma_pairwise_product_from_ab_regs(
        a_frag,
        rhs_frag,
        p0_acc,
        p1_acc,
        input_dtype,
    )

    for accum_idx in cutlass.range_constexpr(SUPER_MMA_ACCUMULATORS_PER_LANE):
        n0_acc[accum_idx] = (
            mma_input_dtype(n0_acc[accum_idx], input_dtype) + p0_acc[accum_idx]
        )
    for accum_idx in cutlass.range_constexpr(SUPER_MMA_ACCUMULATORS_PER_LANE):
        n1_acc[accum_idx] = (
            mma_input_dtype(n1_acc[accum_idx], input_dtype) + p1_acc[accum_idx]
        )


@cute.jit
def super_mma_stage_blockwise_inverse(
    pairwise_smem,
    lane,
    l_frag,
    input_dtype: cutlass.Constexpr,
) -> None:
    """Produce `A_inv` via the blockwise inverse, register-carried, all-MMA.

    Hierarchical (GDN-style) block inverse of the unit-lower `A = I + L`,
    expressed entirely as pairwise tensor-core MMA operations:

        D    = blockdiag(L11, L22)        (strict-lower 8x8 diagonal blocks)
        Binv = (I - D)(I + D^2)(I + D^4)  -- EXACT: each 8x8 unit-lower
                                             block is nilpotent at index 8
        A^-1 = Binv - Binv @ A21hat @ Binv,  A21hat = [[0,0],[A21,0]]
                                          -- EXACT: (Binv @ A21hat)^2 = 0

    Numerics: with near-identical keys and high beta the 16x16 Neumann
    chain's intermediates L^2/L^4/L^8 grow combinatorially large and 16-bit
    rounding destroys the alternating-sign cancellation that keeps the true
    inverse bounded (geometric state growth on repeated tokens).  The
    block-diagonal chain never materializes them — D^2/D^4 stay O(10) — so
    the result sits at the exact-inverse floor: adversarial d_state vs the
    fp32 pipeline drops from 0.33-0.55 (16x16 fp16 Neumann) to 0.05-0.07.
    Chain and combine operands are FP16 with FP32 accumulation, never bf16;
    only the final staged tile is rounded to the input dtype, matching a
    bf16 INV store.

    Cost: 12 MMA operations (2 squares + 2 updates + 2 combine products) — the same
    count as the 16x16 Neumann chain with one fewer pack/round/add update
    stage; measured engine-kernel time is ~0.4% BELOW the Neumann baseline
    (issue-bound auxiliary-MMA warp region, so avoiding non-MMA work is what
    matters).
    """

    l_vec = cutlass.Vector.from_elements(
        (l_frag[0], l_frag[1], l_frag[2], l_frag[3]),
        cutlass.Int32,
    )
    l_values = l_vec.bitcast(input_dtype).to(cutlass.Float32)

    zero_pair = cutlass.Int32(0)
    # D fragment = the L fragment with the A21 quadrant zeroed, repacked to
    # FP16.  Slot 1 holds rows r+8 / cols c,c+1 (A21); slot 2 (rows r,
    # cols c+8,c+9) is already the zero upper-right quadrant of the
    # strictly-lower L.
    d_frag = cutlass.Array(
        cutlass.Int32,
        SUPER_MMA_ACCUMULATORS_PER_LANE,
        alignment=16,
    )
    d_frag[0] = pack_input_b16x2_to_i32(l_values[0], l_values[1], cutlass.Float16)
    d_frag[1] = zero_pair
    d_frag[2] = zero_pair
    d_frag[3] = pack_input_b16x2_to_i32(l_values[6], l_values[7], cutlass.Float16)

    # Accumulators start at I - D (= I - L with the A21 quadrant zeroed).
    n0_acc = cutlass.Array(
        cutlass.Float32,
        SUPER_MMA_ACCUMULATORS_PER_LANE,
        alignment=16,
    )
    n1_acc = cutlass.Array(
        cutlass.Float32,
        SUPER_MMA_ACCUMULATORS_PER_LANE,
        alignment=16,
    )
    for accum_idx in cutlass.range_constexpr(SUPER_MMA_ACCUMULATORS_PER_LANE):
        n0_acc[accum_idx] = super_mma_initial_inverse_from_fragment(
            l_values,
            lane,
            0,
            accum_idx,
        )
        n1_acc[accum_idx] = super_mma_initial_inverse_from_fragment(
            l_values,
            lane,
            1,
            accum_idx,
        )
    n0_acc[2] = cutlass.Float32(0.0)
    n0_acc[3] = cutlass.Float32(0.0)

    d2_frag = cutlass.Array(
        cutlass.Int32,
        SUPER_MMA_ACCUMULATORS_PER_LANE,
        alignment=16,
    )
    super_mma_square_pairwise_fragment(
        d_frag,
        d2_frag,
        cutlass.Float16,
    )
    super_mma_update_inverse_with_power_regs(
        d2_frag,
        n0_acc,
        n1_acc,
        cutlass.Float16,
    )

    d4_frag = cutlass.Array(
        cutlass.Int32,
        SUPER_MMA_ACCUMULATORS_PER_LANE,
        alignment=16,
    )
    super_mma_square_pairwise_fragment(
        d2_frag,
        d4_frag,
        cutlass.Float16,
    )
    super_mma_update_inverse_with_power_regs(
        d4_frag,
        n0_acc,
        n1_acc,
        cutlass.Float16,
    )

    # The accumulators now hold Binv: A11inv in n0[0:2], A22inv in n1[2:4];
    # the off-diagonal quadrants (n0[2:4], n1[0:2]) stay exactly zero
    # through the chain (block-diagonal times block-diagonal).
    binv_frag = cutlass.Array(
        cutlass.Int32,
        SUPER_MMA_ACCUMULATORS_PER_LANE,
        alignment=16,
    )
    binv_frag[0] = pack_input_b16x2_to_i32(n0_acc[0], n0_acc[1], cutlass.Float16)
    binv_frag[1] = zero_pair
    binv_frag[2] = zero_pair
    binv_frag[3] = pack_input_b16x2_to_i32(n1_acc[2], n1_acc[3], cutlass.Float16)

    a21_frag = cutlass.Array(
        cutlass.Int32,
        SUPER_MMA_ACCUMULATORS_PER_LANE,
        alignment=16,
    )
    a21_frag[0] = zero_pair
    a21_frag[1] = pack_input_b16x2_to_i32(l_values[2], l_values[3], cutlass.Float16)
    a21_frag[2] = zero_pair
    a21_frag[3] = zero_pair

    # MMA pair 1: T1 = Binv @ A21hat = [[0,0],[A22inv A21, 0]]
    t1_n0_acc = cutlass.Array(
        cutlass.Float32,
        SUPER_MMA_ACCUMULATORS_PER_LANE,
        alignment=16,
    )
    t1_n1_acc = cutlass.Array(
        cutlass.Float32,
        SUPER_MMA_ACCUMULATORS_PER_LANE,
        alignment=16,
    )
    super_mma_pairwise_product_from_ab_regs(
        binv_frag,
        a21_frag,
        t1_n0_acc,
        t1_n1_acc,
        cutlass.Float16,
    )

    # MMA pair 2: X21hat = (-T1) @ Binv = [[0,0],[-A22inv A21 A11inv, 0]]
    t1n_frag = cutlass.Array(
        cutlass.Int32,
        SUPER_MMA_ACCUMULATORS_PER_LANE,
        alignment=16,
    )
    t1n_frag[0] = zero_pair
    t1n_frag[1] = pack_input_b16x2_to_i32(
        cutlass.Float32(0.0) - t1_n0_acc[2],
        cutlass.Float32(0.0) - t1_n0_acc[3],
        cutlass.Float16,
    )
    t1n_frag[2] = zero_pair
    t1n_frag[3] = zero_pair
    x21_n0_acc = cutlass.Array(
        cutlass.Float32,
        SUPER_MMA_ACCUMULATORS_PER_LANE,
        alignment=16,
    )
    x21_n1_acc = cutlass.Array(
        cutlass.Float32,
        SUPER_MMA_ACCUMULATORS_PER_LANE,
        alignment=16,
    )
    super_mma_pairwise_product_from_ab_regs(
        t1n_frag,
        binv_frag,
        x21_n0_acc,
        x21_n1_acc,
        cutlass.Float16,
    )

    # A_inv = Binv with the A21 quadrant replaced by X21.
    n0_acc[2] = x21_n0_acc[2]
    n0_acc[3] = x21_n0_acc[3]

    super_mma_store_pairwise_tile_stmatrix_x4(
        pairwise_smem,
        PAIRWISE_SMEM_AINV_OFFSET,
        lane,
        n0_acc,
        n1_acc,
        input_dtype,
    )


@cute.jit
def super_mma_stage_pairwise_pipeline(
    tcgen05_k_decay_smem,
    k_inv_smem,
    pairwise_smem,
    raw_beta_smem,
    cg0_k_ready_stage_mbar,
    cg0_k_ready_phase,
    lane,
    input_dtype: cutlass.Constexpr,
) -> None:
    """Run the KK/L/inverse sequence inside the auxiliary-MMA warp.

    The caller has already waited `cg0_k_half_ready` (first half-DK of
    k_inv/k_decay staged); the full `cg0_k_ready` wait happens here, between
    the two KK half-products.
    """

    kk_n0_acc = cutlass.Array(
        cutlass.Float32, SUPER_MMA_ACCUMULATORS_PER_LANE, alignment=16
    )
    kk_n1_acc = cutlass.Array(
        cutlass.Float32, SUPER_MMA_ACCUMULATORS_PER_LANE, alignment=16
    )
    for acc_idx in cutlass.range_constexpr(SUPER_MMA_ACCUMULATORS_PER_LANE):
        kk_n0_acc[acc_idx] = cutlass.Float32(0.0)
        kk_n1_acc[acc_idx] = cutlass.Float32(0.0)
    super_mma_stage_kk_blocks(
        tcgen05_k_decay_smem,
        k_inv_smem,
        lane,
        input_dtype,
        0,
        SUPER_MMA_K_BLOCKS // 2,
        kk_n0_acc,
        kk_n1_acc,
    )
    cg0_k_ready_wait(
        cg0_k_ready_stage_mbar,
        cg0_k_ready_phase,
    )
    super_mma_stage_kk_blocks(
        tcgen05_k_decay_smem,
        k_inv_smem,
        lane,
        input_dtype,
        SUPER_MMA_K_BLOCKS // 2,
        SUPER_MMA_K_BLOCKS,
        kk_n0_acc,
        kk_n1_acc,
    )
    l_frag = cutlass.Array(cutlass.Int32, SUPER_MMA_ACCUMULATORS_PER_LANE, alignment=16)
    super_mma_build_l_fragment(
        raw_beta_smem,
        lane,
        kk_n0_acc,
        kk_n1_acc,
        l_frag,
        input_dtype,
    )
    super_mma_stage_blockwise_inverse(
        pairwise_smem,
        lane,
        l_frag,
        input_dtype,
    )


@cute.jit
def pairwise_ready_arrive(pairwise_ready_mbar) -> None:
    """Signal that the auxiliary-MMA pairwise/A_inv workspace is ready."""

    prims.fence_proxy(
        prims.Proxy.ASYNC_SHARED,
        space=prims.SharedSpace.shared_cta,
    )
    if prims.elect_sync():
        prims.mbarrier_arrive(pairwise_ready_mbar)


@cute.jit
def pairwise_ready_wait(pairwise_ready_mbar, pairwise_ready_phase):
    """Wait for the auxiliary-MMA pairwise/A_inv workspace."""

    while not prims.mbarrier_wait_parity(
        pairwise_ready_mbar,
        pairwise_ready_phase,
        prims.MBarrierWait.TRY,
    ):
        pass
    return pairwise_ready_phase ^ cutlass.Int32(1)


@cute.jit
def pairwise_consumed_arrive(pairwise_consumed_mbar) -> None:
    """Release a pairwise SMEM stage after tcgen05 consumption."""

    if prims.elect_sync():
        prims.tcgen05_commit(pairwise_consumed_mbar, group=prims.CTAGroup.CTA_1)


@cute.jit
def pairwise_consumed_wait(pairwise_consumed_mbar, pairwise_consumed_phase):
    """Wait until one pairwise stage can be overwritten."""

    while not prims.mbarrier_wait_parity(
        pairwise_consumed_mbar,
        pairwise_consumed_phase,
        prims.MBarrierWait.TRY,
    ):
        pass
    return pairwise_consumed_phase ^ cutlass.Int32(1)


@cute.jit
def output_ready_arrive(output_ready_mbar) -> None:
    """Signal that this CG1 warp has finished staging output SMEM."""

    if prims.elect_sync():
        prims.mbarrier_arrive(output_ready_mbar)


@cute.jit
def output_ready_wait(output_ready_mbar, output_ready_phase):
    """Wait until all CG1 warps have staged the output tile."""

    while not prims.mbarrier_wait_parity(
        output_ready_mbar,
        output_ready_phase,
        prims.MBarrierWait.TRY,
    ):
        pass
    return output_ready_phase ^ cutlass.Int32(1)


@cute.jit
def q_k_restore_ready_arrive(q_k_restore_ready_mbar) -> None:
    """Signal that this CG0 warp has staged q_decay and k_restore."""

    prims.fence_proxy(
        prims.Proxy.ASYNC_SHARED,
        space=prims.SharedSpace.shared_cta,
    )
    if prims.elect_sync():
        prims.mbarrier_arrive(q_k_restore_ready_mbar)


@cute.jit
def q_k_restore_ready_wait(q_k_restore_ready_mbar, q_k_restore_ready_phase):
    """Wait until CG0 has staged q_decay and k_restore."""

    while not prims.mbarrier_wait_parity(
        q_k_restore_ready_mbar,
        q_k_restore_ready_phase,
        prims.MBarrierWait.TRY,
    ):
        pass
    return q_k_restore_ready_phase ^ cutlass.Int32(1)


@cute.jit
def cg0_k_ready_arrive(cg0_k_ready_mbar) -> None:
    """Publish k_decay and k_inv before CG0 finishes Q operands."""

    prims.fence_proxy(
        prims.Proxy.ASYNC_SHARED,
        space=prims.SharedSpace.shared_cta,
    )
    if prims.elect_sync():
        prims.mbarrier_arrive(cg0_k_ready_mbar)


@cute.jit
def cg0_k_ready_wait(cg0_k_ready_mbar, cg0_k_ready_phase):
    """Wait until CG0 has staged k_decay and k_inv."""

    while not prims.mbarrier_wait_parity(
        cg0_k_ready_mbar,
        cg0_k_ready_phase,
        prims.MBarrierWait.TRY,
    ):
        pass
    return cg0_k_ready_phase ^ cutlass.Int32(1)


@cute.jit
def diag_ready_arrive(diag_ready_mbar) -> None:
    """Publish the FP32 final-prefix diagonal before K/Q materialization."""

    prims.fence_proxy(
        prims.Proxy.ASYNC_SHARED,
        space=prims.SharedSpace.shared_cta,
    )
    if prims.elect_sync():
        prims.mbarrier_arrive(diag_ready_mbar)


@cute.jit
def diag_ready_wait(diag_ready_mbar, diag_ready_phase):
    """Wait until CG0 has staged the FP32 final-prefix diagonal."""

    while not prims.mbarrier_wait_parity(
        diag_ready_mbar,
        diag_ready_phase,
        prims.MBarrierWait.TRY,
    ):
        pass
    return diag_ready_phase ^ cutlass.Int32(1)


@cute.jit
def raw_ready_arrive(raw_ready_mbar) -> None:
    """Signal that raw SMEM inputs are ready (k2-chain relay only)."""

    if prims.elect_sync():
        prims.mbarrier_arrive(raw_ready_mbar)


@cute.jit
def raw_ready_wait(raw_ready_mbar, raw_ready_phase):
    """Generic mbarrier parity spin-wait.

    The engine-class kernels observe raw readiness directly on the
    chunk's tma_mbar ring slot (consumer-direct wait); the k2 chain
    kernels still wait their raw_ready relay ring, and the ws_stored /
    The deltas_issued relay ring uses this helper too.
    """

    while not prims.mbarrier_wait_parity(
        raw_ready_mbar,
        raw_ready_phase,
        prims.MBarrierWait.TRY,
    ):
        pass
    return raw_ready_phase ^ cutlass.Int32(1)


@cute.jit
def checkpoint_read_done_arrive(checkpoint_read_done_mbar) -> None:
    """Signal that one producer warp finished reading checkpoint state."""

    if prims.elect_sync():
        prims.mbarrier_arrive(checkpoint_read_done_mbar)


@cute.jit
def checkpoint_read_done_wait(checkpoint_read_done_mbar, phase) -> None:
    """Protect a checkpoint read from the next in-place state update."""

    while not prims.mbarrier_wait_parity(
        checkpoint_read_done_mbar,
        phase,
        prims.MBarrierWait.TRY,
    ):
        pass


@cute.jit
def raw_consumed_arrive(raw_consumed_mbar) -> None:
    """Signal that one participating warp has finished raw-slot reads."""

    if prims.elect_sync():
        prims.mbarrier_arrive(raw_consumed_mbar)


@cute.jit
def raw_consumed_wait(raw_consumed_mbar, raw_consumed_phase):
    """Wait until the previous chunk no longer reads raw input SMEM."""

    while not prims.mbarrier_wait_parity(
        raw_consumed_mbar,
        raw_consumed_phase,
        prims.MBarrierWait.TRY,
    ):
        pass
    return raw_consumed_phase ^ cutlass.Int32(1)


@cute.jit
def state_input_ready_arrive(state_input_ready_mbar) -> None:
    """Signal that this CG1 warp has packed state_as_input TMEM."""

    if prims.elect_sync():
        prims.mbarrier_arrive(state_input_ready_mbar)


@cute.jit
def state_input_ready_wait(state_input_ready_mbar, state_input_ready_phase):
    """Wait until all CG1 warps have packed state_as_input TMEM."""

    while not prims.mbarrier_wait_parity(
        state_input_ready_mbar,
        state_input_ready_phase,
        prims.MBarrierWait.TRY,
    ):
        pass
    return state_input_ready_phase ^ cutlass.Int32(1)


@cute.jit
def operand_smem_consumed_arrive(operand_smem_consumed_mbar) -> None:
    """Signal that this warp no longer needs the current operand SMEM."""

    if prims.elect_sync():
        prims.mbarrier_arrive(operand_smem_consumed_mbar)


@cute.jit
def operand_smem_consumed_wait(
    operand_smem_consumed_mbar,
    operand_smem_consumed_phase,
):
    """Wait until previous chunk consumers released operand SMEM."""

    while not prims.mbarrier_wait_parity(
        operand_smem_consumed_mbar,
        operand_smem_consumed_phase,
        prims.MBarrierWait.TRY,
    ):
        pass
    return operand_smem_consumed_phase ^ cutlass.Int32(1)


@cute.jit
def rhs_ready_arrive(rhs_ready_mbar) -> None:
    """Signal that this CG1 warp has staged RHS for update MMA."""

    if prims.elect_sync():
        prims.mbarrier_arrive(rhs_ready_mbar)


@cute.jit
def rhs_ready_wait(rhs_ready_mbar, rhs_ready_phase):
    """Wait until all CG1 warps have staged RHS for update MMA."""

    while not prims.mbarrier_wait_parity(
        rhs_ready_mbar,
        rhs_ready_phase,
        prims.MBarrierWait.TRY,
    ):
        pass
    return rhs_ready_phase ^ cutlass.Int32(1)


@cute.jit
def update_ready_arrive(update_ready_mbar) -> None:
    """Signal that this CG1 warp has staged update for qkv MMA."""

    if prims.elect_sync():
        prims.mbarrier_arrive(update_ready_mbar)


@cute.jit
def update_ready_wait(update_ready_mbar, update_ready_phase):
    """Wait until all CG1 warps have staged update for qkv MMA."""

    while not prims.mbarrier_wait_parity(
        update_ready_mbar,
        update_ready_phase,
        prims.MBarrierWait.TRY,
    ):
        pass
    return update_ready_phase ^ cutlass.Int32(1)


@cute.jit
def output_consumed_arrive(output_consumed_mbar) -> None:
    """Signal that the epilogue warp has drained current output SMEM."""

    if prims.elect_sync():
        prims.mbarrier_arrive(output_consumed_mbar)


@cute.jit
def output_consumed_wait(output_consumed_mbar, output_consumed_phase):
    """Wait until previous output SMEM contents have been stored."""

    while not prims.mbarrier_wait_parity(
        output_consumed_mbar,
        output_consumed_phase,
        prims.MBarrierWait.TRY,
    ):
        pass
    return output_consumed_phase ^ cutlass.Int32(1)


@cute.jit
def final_state_stored_arrive(final_state_stored_mbar) -> None:
    """Signal that this CG1 warp has finished draining final_state TMEM."""

    if prims.elect_sync():
        prims.mbarrier_arrive(final_state_stored_mbar)


@cute.jit
def final_state_stored_wait(final_state_stored_mbar, final_state_stored_phase):
    """Wait until CG1 has drained final_state before TMEM deallocation."""

    while not prims.mbarrier_wait_parity(
        final_state_stored_mbar,
        final_state_stored_phase,
        prims.MBarrierWait.TRY,
    ):
        pass
    return final_state_stored_phase ^ cutlass.Int32(1)


@cute.jit
def pack_output_b16x2_to_i32(
    value0: cutlass.Float32,
    value1: cutlass.Float32,
    output_dtype: cutlass.Constexpr,
):
    """Pack two FP32 output values through the compile-time 16-bit output dtype."""

    return (
        cutlass.Vector.from_elements(
            (value0, value1),
            cutlass.Float32,
        )
        .to(output_dtype)
        .bitcast(cutlass.Int32)[0]
    )


@cute.jit
def tcgen05_qstate_acc_tmem_col_offset(qstate_acc_stage):
    """Return the runtime TMEM column offset for one qstate acc stage."""

    return (
        KDA_TMEM_QSTATE_ACC_COL_OFFSET
        + qstate_acc_stage * KDA_TMEM_QSTATE_ACC_STAGE_STRIDE_COLS
    )


@cute.jit
def tcgen05_shared_acc_tmem_col_offset(shared_acc_stage):
    """Return the runtime TMEM column offset for one shared_acc stage."""

    return KDA_TMEM_SHARED_ACC_COL_OFFSET + shared_acc_stage * KDA_TMEM_N16_ACC_COLS


@cute.jit
def tcgen05_shared_acc_stage_from_event(shared_acc_event_id):
    """Map one shared-acc event to its ring-buffer stage."""

    return shared_acc_event_id % KDA_TMEM_SHARED_ACC_STAGE_COUNT


@cute.jit
def tcgen05_shared_acc_phase_from_event(shared_acc_event_id):
    """Map one shared-acc event to the selected stage's barrier phase."""

    return (shared_acc_event_id // KDA_TMEM_SHARED_ACC_STAGE_COUNT) % 2


@cute.jit
def tcgen05_shared_input_tmem_col_offset(shared_input_stage):
    """Return the runtime TMEM column offset for one shared input stage."""

    return (
        KDA_TMEM_SHARED_INPUT_COL_OFFSET
        + shared_input_stage * KDA_TMEM_SHARED_INPUT_COLS
    )


@cute.jit
def advance_ring_stage(
    stage,
    step: cutlass.Constexpr,
    stage_count: cutlass.Constexpr,
):
    """Advance a runtime ring index without division or a wrap branch."""

    next_stage = stage + cutlass.Int32(step)
    wrapped = cutlass.Int32(next_stage >= cutlass.Int32(stage_count))
    return next_stage - wrapped * cutlass.Int32(stage_count), wrapped


@cute.jit
def tcgen05_store_initial_state_tmem(
    tmem_raw_addr,
    initial_state: cute.Tensor | None,
    state_ckpt: cute.Tensor | None,
    ckpt_slot,
    state_slot,
    bidy,
    dv_half,
    warp_idx,
    lane,
    HALF: cutlass.Constexpr,
) -> None:
    """Initialize recurrent TMEM from the optional external VK state.

    HALF=True is the DV2 chain form (Layout F): this warp's quadrant lanes
    0..15 hold state rows dv_half*64 + 16q .. +15; the alignment-16 lanes are
    zero-filled junk (kept written so the first pack reads defined data).
    """

    base_col_id = tmem_raw_addr & 0xFFFF
    base_row_id = tmem_raw_addr >> 16
    tmem_sp = warp_idx % TCGEN05_STATE_K_TMEM_ROW_BLOCKS

    row_id = base_row_id + tmem_sp * THREADS_PER_WARP
    if cutlass.const_expr(HALF):
        value_dim = (
            dv_half * cutlass.Int32(DV_HALF)
            + tmem_sp * ROWS_PER_WARP
            + (lane % ROWS_PER_WARP)
        )
        valid_lane = lane < ROWS_PER_WARP
    else:
        value_dim = tmem_sp * THREADS_PER_WARP + lane
    for key_block_start in cutlass.range_constexpr(
        0,
        DK,
        TCGEN05_FINAL_STATE_TMEM_LOAD_COLS,
    ):
        state_block = cutlass.Array(
            cutlass.Float32,
            TCGEN05_FINAL_STATE_TMEM_LOAD_COLS,
            alignment=16,
        )
        for col in cutlass.range_constexpr(TCGEN05_FINAL_STATE_TMEM_LOAD_COLS):
            key_dim = key_block_start + col
            state_value = cutlass.Float32(0.0)
            if cutlass.const_expr(initial_state is not None):
                if cutlass.const_expr(HALF):
                    if valid_lane:
                        state_value = initial_state[
                            state_slot, bidy, value_dim, key_dim
                        ].to(cutlass.Float32)
                else:
                    state_value = initial_state[
                        state_slot, bidy, value_dim, key_dim
                    ].to(cutlass.Float32)
            state_block[col] = state_value
            if cutlass.const_expr(state_ckpt is not None):
                if (ckpt_slot >= cutlass.Int32(0)) & (
                    ckpt_slot < cutlass.Int32(state_ckpt.shape[0])
                ):
                    if cutlass.const_expr(HALF):
                        if valid_lane:
                            state_ckpt[ckpt_slot, bidy, value_dim, key_dim] = (
                                state_value.to(state_ckpt.element_type)
                            )
                    else:
                        state_ckpt[ckpt_slot, bidy, value_dim, key_dim] = (
                            state_value.to(state_ckpt.element_type)
                        )

        projection_col_id = base_col_id + KDA_TMEM_STATE_COL_OFFSET + key_block_start
        block_addr = (row_id << 16) | projection_col_id
        block_ptr = cutlass.inttoptr(block_addr, 6, cutlass.Float32)
        prims.tcgen05_st(
            "32x32b",
            block_ptr,
            state_block[0:TCGEN05_FINAL_STATE_TMEM_LOAD_COLS],
        )

    prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)


@cute.jit
def pack_state_input_x16(state_block, input_dtype: cutlass.Constexpr):
    """Pack one 16-column FP32 state fragment into an 8-column A fragment."""

    packed_state = cutlass.Array(
        cutlass.Int32,
        TCGEN05_STATE_INPUT_PACKED_COLS,
        alignment=16,
    )
    for packed_col in cutlass.range_constexpr(TCGEN05_STATE_INPUT_PACKED_COLS):
        source_pair = packed_col ^ TCGEN05_F16_A_TMEM_PAIR_XOR
        key_dim0 = source_pair * 2
        key_dim1 = key_dim0 + 1
        packed_state[packed_col] = pack_input_b16x2_to_i32(
            state_block[key_dim0],
            state_block[key_dim1],
            input_dtype,
        )
    return packed_state


@cute.jit
def tcgen05_stage_state_input_tmem(
    tmem_raw_addr,
    warp_idx,
    output_consumed_mbar,
    output_consumed_phase,
    input_dtype: cutlass.Constexpr,
    WAIT_OUTPUT_CONSUMED: cutlass.Constexpr,
    HALF: cutlass.Constexpr,
    DEFER_STORE_WAIT: cutlass.Constexpr,
):
    """Pack one 64-column half of the recurrent state as a tcgen05 A operand.

    ``HALF`` selects state columns ``[HALF*64, HALF*64+64)``. Splitting the
    pack lets the left half start as soon as the left half of the previous
    chunk's final-state delta MMA has committed, overlapping with the right
    delta half. Steady-state chunks overlap the prior output-stage reuse
    wait with the asynchronous state loads.
    """

    base_col_id = tmem_raw_addr & 0xFFFF
    base_row_id = tmem_raw_addr >> 16
    tmem_sp = warp_idx % TCGEN05_STATE_K_TMEM_ROW_BLOCKS

    row_addr = (base_row_id + tmem_sp * THREADS_PER_WARP) << 16
    state_col_id = (
        base_col_id
        + KDA_TMEM_STATE_COL_OFFSET
        + HALF * 4 * TCGEN05_STATE_INPUT_LOAD_COLS
    )
    state_ptr0 = cutlass.inttoptr(row_addr | state_col_id, 6, cutlass.Float32)
    state_ptr1 = cutlass.inttoptr(
        row_addr | (state_col_id + TCGEN05_STATE_INPUT_LOAD_COLS),
        6,
        cutlass.Float32,
    )
    state_ptr2 = cutlass.inttoptr(
        row_addr | (state_col_id + 2 * TCGEN05_STATE_INPUT_LOAD_COLS),
        6,
        cutlass.Float32,
    )
    state_ptr3 = cutlass.inttoptr(
        row_addr | (state_col_id + 3 * TCGEN05_STATE_INPUT_LOAD_COLS),
        6,
        cutlass.Float32,
    )

    state0 = prims.tcgen05_ld("32x32b", state_ptr0, num=TCGEN05_STATE_INPUT_LOAD_COLS)
    state1 = prims.tcgen05_ld("32x32b", state_ptr1, num=TCGEN05_STATE_INPUT_LOAD_COLS)
    state2 = prims.tcgen05_ld("32x32b", state_ptr2, num=TCGEN05_STATE_INPUT_LOAD_COLS)
    state3 = prims.tcgen05_ld("32x32b", state_ptr3, num=TCGEN05_STATE_INPUT_LOAD_COLS)
    if cutlass.const_expr(WAIT_OUTPUT_CONSUMED):
        output_consumed_wait(
            output_consumed_mbar,
            output_consumed_phase,
        )
    prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)

    packed_col_id = (
        base_col_id
        + KDA_TMEM_STATE_AS_INPUT_COL_OFFSET
        + HALF * 4 * TCGEN05_STATE_INPUT_PACKED_COLS
    )
    packed_row_addr = base_row_id << 16
    packed_ptr0 = prims.make_tmem_ptr(packed_row_addr | packed_col_id, cutlass.Int8)
    packed_ptr1 = prims.make_tmem_ptr(
        packed_row_addr + packed_col_id + TCGEN05_STATE_INPUT_PACKED_COLS,
        cutlass.Int8,
    )
    packed_ptr2 = prims.make_tmem_ptr(
        packed_row_addr + packed_col_id + 2 * TCGEN05_STATE_INPUT_PACKED_COLS,
        cutlass.Int8,
    )
    packed_ptr3 = prims.make_tmem_ptr(
        packed_row_addr + packed_col_id + 3 * TCGEN05_STATE_INPUT_PACKED_COLS,
        cutlass.Int8,
    )
    packed0 = pack_state_input_x16(state0, input_dtype)
    prims.tcgen05_st("32x32b", packed_ptr0, packed0[0:TCGEN05_STATE_INPUT_PACKED_COLS])
    packed1 = pack_state_input_x16(state1, input_dtype)
    prims.tcgen05_st("32x32b", packed_ptr1, packed1[0:TCGEN05_STATE_INPUT_PACKED_COLS])
    packed2 = pack_state_input_x16(state2, input_dtype)
    prims.tcgen05_st("32x32b", packed_ptr2, packed2[0:TCGEN05_STATE_INPUT_PACKED_COLS])
    packed3 = pack_state_input_x16(state3, input_dtype)
    prims.tcgen05_st("32x32b", packed_ptr3, packed3[0:TCGEN05_STATE_INPUT_PACKED_COLS])

    if cutlass.const_expr(not DEFER_STORE_WAIT):
        prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    return state0, state1, state2, state3


@cute.jit
def tcgen05_scale_state_x16_regs(
    state_block,
    state_scale_f32_ptr,
    key_block_start: cutlass.Constexpr,
    EXCHANGE_LAYOUT: cutlass.Constexpr,
):
    """Build one true-FP32 scaled state fragment without issuing its store."""

    scaled_state = cutlass.Array(
        cutlass.Float32,
        TCGEN05_STATE_INPUT_LOAD_COLS,
        alignment=16,
    )
    for vec_idx in cutlass.range_constexpr(TCGEN05_STATE_INPUT_LOAD_COLS // 4):
        reg_base = vec_idx * 4
        scale_dim: cutlass.Constexpr = key_block_start + reg_base
        scale_idx = cutlass.Int32(scale_dim)
        if cutlass.const_expr(EXCHANGE_LAYOUT):
            scale_idx = raw_f32_exchange_smem_index(BT - 1, scale_dim)
        scale = (state_scale_f32_ptr + scale_idx).load(count=4, alignment=16)
        scaled_state[reg_base], scaled_state[reg_base + 1] = fmul2(
            (state_block[reg_base], state_block[reg_base + 1]),
            (scale[0], scale[1]),
        )
        scaled_state[reg_base + 2], scaled_state[reg_base + 3] = fmul2(
            (state_block[reg_base + 2], state_block[reg_base + 3]),
            (scale[2], scale[3]),
        )
    return scaled_state


@cute.jit
def tcgen05_rescale_state_x32(
    state_block,
    state_block_ptr,
    state_scale_f32_ptr,
    key_block_start: cutlass.Constexpr,
    EXCHANGE_LAYOUT: cutlass.Constexpr,
) -> None:
    """Apply one 32-element FP32 decay fragment and store the state block."""

    block_cols: cutlass.Constexpr = 2 * TCGEN05_STATE_INPUT_LOAD_COLS
    scaled_state = cutlass.Array(cutlass.Float32, block_cols, alignment=16)
    for vec_idx in cutlass.range_constexpr(block_cols // 4):
        reg_base = vec_idx * 4
        scale_dim: cutlass.Constexpr = key_block_start + reg_base
        scale_idx = cutlass.Int32(scale_dim)
        if cutlass.const_expr(EXCHANGE_LAYOUT):
            scale_idx = raw_f32_exchange_smem_index(BT - 1, scale_dim)
        scale = (state_scale_f32_ptr + scale_idx).load(count=4, alignment=16)
        scaled_state[reg_base], scaled_state[reg_base + 1] = fmul2(
            (state_block[reg_base], state_block[reg_base + 1]),
            (scale[0], scale[1]),
        )
        scaled_state[reg_base + 2], scaled_state[reg_base + 3] = fmul2(
            (state_block[reg_base + 2], state_block[reg_base + 3]),
            (scale[2], scale[3]),
        )
    prims.tcgen05_st("32x32b", state_block_ptr, scaled_state[0:block_cols])


@cute.jit
def tcgen05_publish_projection_then_rescale_state_regs(
    tmem_raw_addr,
    state_scale_f32_smem,
    state_input_ready_mbar,
    warp_idx,
    state0,
    state1,
    state2,
    state3,
    HALF: cutlass.Constexpr,
    EXCHANGE_LAYOUT: cutlass.Constexpr,
) -> None:
    """Hide a projection-input store behind true-FP32 state scaling.

    The packed projection columns and live recurrent-state columns do not
    alias.  Build the scaled fragments while the preceding packed stores are
    outstanding, then publish those packed columns before issuing the scaled
    live-state stores.  The final wait/fence remains a separate update join.
    """

    base_col_id = tmem_raw_addr & 0xFFFF
    base_row_id = tmem_raw_addr >> 16
    tmem_sp = warp_idx % TCGEN05_STATE_K_TMEM_ROW_BLOCKS
    row_addr = (base_row_id + tmem_sp * THREADS_PER_WARP) << 16
    state_col_id = (
        base_col_id
        + KDA_TMEM_STATE_COL_OFFSET
        + HALF * 4 * TCGEN05_STATE_INPUT_LOAD_COLS
    )
    state_ptr0 = cutlass.inttoptr(row_addr | state_col_id, 6, cutlass.Float32)
    state_ptr1 = cutlass.inttoptr(
        row_addr | (state_col_id + TCGEN05_STATE_INPUT_LOAD_COLS),
        6,
        cutlass.Float32,
    )
    state_ptr2 = cutlass.inttoptr(
        row_addr | (state_col_id + 2 * TCGEN05_STATE_INPUT_LOAD_COLS),
        6,
        cutlass.Float32,
    )
    state_ptr3 = cutlass.inttoptr(
        row_addr | (state_col_id + 3 * TCGEN05_STATE_INPUT_LOAD_COLS),
        6,
        cutlass.Float32,
    )
    state_scale_f32_ptr = state_scale_f32_smem.data_ptr()
    key_half_start: cutlass.Constexpr = HALF * 4 * TCGEN05_STATE_INPUT_LOAD_COLS
    scaled0 = tcgen05_scale_state_x16_regs(
        state0,
        state_scale_f32_ptr,
        key_half_start,
        EXCHANGE_LAYOUT,
    )
    scaled1 = tcgen05_scale_state_x16_regs(
        state1,
        state_scale_f32_ptr,
        key_half_start + TCGEN05_STATE_INPUT_LOAD_COLS,
        EXCHANGE_LAYOUT,
    )
    scaled2 = tcgen05_scale_state_x16_regs(
        state2,
        state_scale_f32_ptr,
        key_half_start + 2 * TCGEN05_STATE_INPUT_LOAD_COLS,
        EXCHANGE_LAYOUT,
    )
    scaled3 = tcgen05_scale_state_x16_regs(
        state3,
        state_scale_f32_ptr,
        key_half_start + 3 * TCGEN05_STATE_INPUT_LOAD_COLS,
        EXCHANGE_LAYOUT,
    )

    # This wait retires only the already-issued packed projection stores.
    prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    state_input_ready_arrive(state_input_ready_mbar)

    prims.tcgen05_st(
        "32x32b",
        state_ptr0,
        scaled0[0:TCGEN05_STATE_INPUT_LOAD_COLS],
    )
    prims.tcgen05_st(
        "32x32b",
        state_ptr1,
        scaled1[0:TCGEN05_STATE_INPUT_LOAD_COLS],
    )
    prims.tcgen05_st(
        "32x32b",
        state_ptr2,
        scaled2[0:TCGEN05_STATE_INPUT_LOAD_COLS],
    )
    prims.tcgen05_st(
        "32x32b",
        state_ptr3,
        scaled3[0:TCGEN05_STATE_INPUT_LOAD_COLS],
    )
    prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)


@cute.jit
def tcgen05_pack_rescale_state_half_tmem(
    tmem_raw_addr,
    state_scale_f32_smem,
    state_input_ready_mbar,
    warp_idx,
    input_dtype: cutlass.Constexpr,
    HALF: cutlass.Constexpr,
    EXCHANGE_LAYOUT: cutlass.Constexpr,
) -> None:
    """Pack and FP32-decay one state half from a single pair of TMEM loads.

    The owning CG0 group publishes the packed left projection operand and the
    decayed live state together.  This removes both CG1's duplicate left-half
    load/pack and CG0's former scale-only reload.
    """

    base_col_id = tmem_raw_addr & 0xFFFF
    base_row_id = tmem_raw_addr >> 16
    tmem_sp = warp_idx % TCGEN05_STATE_K_TMEM_ROW_BLOCKS
    row_addr = (base_row_id + tmem_sp * THREADS_PER_WARP) << 16
    block_cols: cutlass.Constexpr = 2 * TCGEN05_STATE_INPUT_LOAD_COLS
    state_col_id = (
        base_col_id
        + KDA_TMEM_STATE_COL_OFFSET
        + HALF * 4 * TCGEN05_STATE_INPUT_LOAD_COLS
    )
    state_ptr0 = cutlass.inttoptr(row_addr | state_col_id, 6, cutlass.Float32)
    state_ptr1 = cutlass.inttoptr(
        row_addr | (state_col_id + block_cols), 6, cutlass.Float32
    )

    packed_col_id = (
        base_col_id
        + KDA_TMEM_STATE_AS_INPUT_COL_OFFSET
        + HALF * 4 * TCGEN05_STATE_INPUT_PACKED_COLS
    )
    packed_row_addr = base_row_id << 16
    packed_ptr0 = prims.make_tmem_ptr(packed_row_addr | packed_col_id, cutlass.Int8)
    packed_ptr1 = prims.make_tmem_ptr(
        packed_row_addr | (packed_col_id + TCGEN05_STATE_INPUT_PACKED_COLS),
        cutlass.Int8,
    )
    packed_ptr2 = prims.make_tmem_ptr(
        packed_row_addr | (packed_col_id + 2 * TCGEN05_STATE_INPUT_PACKED_COLS),
        cutlass.Int8,
    )
    packed_ptr3 = prims.make_tmem_ptr(
        packed_row_addr | (packed_col_id + 3 * TCGEN05_STATE_INPUT_PACKED_COLS),
        cutlass.Int8,
    )

    state0 = prims.tcgen05_ld("32x32b", state_ptr0, num=block_cols)
    state1 = prims.tcgen05_ld("32x32b", state_ptr1, num=block_cols)
    prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
    packed0 = pack_state_input_x16(state0[0:TCGEN05_STATE_INPUT_LOAD_COLS], input_dtype)
    packed1 = pack_state_input_x16(
        state0[TCGEN05_STATE_INPUT_LOAD_COLS:block_cols], input_dtype
    )
    prims.tcgen05_st("32x32b", packed_ptr0, packed0[0:TCGEN05_STATE_INPUT_PACKED_COLS])
    prims.tcgen05_st("32x32b", packed_ptr1, packed1[0:TCGEN05_STATE_INPUT_PACKED_COLS])
    packed2 = pack_state_input_x16(state1[0:TCGEN05_STATE_INPUT_LOAD_COLS], input_dtype)
    packed3 = pack_state_input_x16(
        state1[TCGEN05_STATE_INPUT_LOAD_COLS:block_cols], input_dtype
    )
    prims.tcgen05_st("32x32b", packed_ptr2, packed2[0:TCGEN05_STATE_INPUT_PACKED_COLS])
    prims.tcgen05_st("32x32b", packed_ptr3, packed3[0:TCGEN05_STATE_INPUT_PACKED_COLS])
    prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    state_input_ready_arrive(state_input_ready_mbar)

    state_scale_f32_ptr = state_scale_f32_smem.data_ptr()
    key_half_start: cutlass.Constexpr = HALF * 4 * TCGEN05_STATE_INPUT_LOAD_COLS
    tcgen05_rescale_state_x32(
        state0,
        state_ptr0,
        state_scale_f32_ptr,
        key_half_start,
        EXCHANGE_LAYOUT,
    )
    tcgen05_rescale_state_x32(
        state1,
        state_ptr1,
        state_scale_f32_ptr,
        key_half_start + block_cols,
        EXCHANGE_LAYOUT,
    )
    prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)


@cute.jit
def tcgen05_stage_state_input_dv2_half_tmem(
    tmem_raw_addr,
    warp_idx,
    input_dtype: cutlass.Constexpr,
    HALF: cutlass.Constexpr,
) -> cutlass.Array:
    """Load one FP32 state half and asynchronously pack its MMA-A image."""

    base_col_id = tmem_raw_addr & 0xFFFF
    base_row_id = tmem_raw_addr >> 16
    tmem_sp = warp_idx % TCGEN05_STATE_K_TMEM_ROW_BLOCKS
    row_addr = (base_row_id + tmem_sp * THREADS_PER_WARP) << 16
    state_half_col = (
        base_col_id
        + KDA_TMEM_STATE_COL_OFFSET
        + HALF * 4 * TCGEN05_STATE_INPUT_LOAD_COLS
    )
    state_ptr = cutlass.inttoptr(
        row_addr | state_half_col,
        6,
        cutlass.Float32,
    )
    state = prims.tcgen05_ld("16x256b", state_ptr, num=8)
    prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)

    packed_state = cutlass.Array(cutlass.Int32, 16, alignment=16)
    for dst_repeat in cutlass.range_constexpr(8):
        src_base = (dst_repeat ^ 1) * 4
        dst_base = dst_repeat * 2
        packed_state[dst_base] = pack_input_b16x2_to_i32(
            state[src_base],
            state[src_base + 1],
            input_dtype,
        )
        packed_state[dst_base + 1] = pack_input_b16x2_to_i32(
            state[src_base + 2],
            state[src_base + 3],
            input_dtype,
        )

    packed_col = (
        base_col_id
        + KDA_TMEM_STATE_AS_INPUT_COL_OFFSET
        + HALF * 4 * TCGEN05_STATE_INPUT_PACKED_COLS
    )
    packed_ptr = prims.make_tmem_ptr(
        (base_row_id << 16) | packed_col,
        cutlass.Int8,
    )
    prims.tcgen05_st("16x128b", packed_ptr, packed_state[0:16])
    return state


@cute.jit
def tcgen05_rescale_state_dv2_half_regs(
    tmem_raw_addr,
    state_scale_f32_smem,
    warp_idx,
    state,
    state_input_ready_mbar,
    HALF: cutlass.Constexpr,
) -> None:
    """Overlap true-FP32 decay with the outstanding packed-state store."""

    base_col_id = tmem_raw_addr & 0xFFFF
    base_row_id = tmem_raw_addr >> 16
    tmem_sp = warp_idx % TCGEN05_STATE_K_TMEM_ROW_BLOCKS
    row_addr = (base_row_id + tmem_sp * THREADS_PER_WARP) << 16
    state_half_col = (
        base_col_id
        + KDA_TMEM_STATE_COL_OFFSET
        + HALF * 4 * TCGEN05_STATE_INPUT_LOAD_COLS
    )
    state_ptr = cutlass.inttoptr(
        row_addr | state_half_col,
        6,
        cutlass.Float32,
    )
    state_scale_f32_ptr = state_scale_f32_smem.data_ptr()
    key_half_start: cutlass.Constexpr = HALF * 4 * TCGEN05_STATE_INPUT_LOAD_COLS
    scaled_state = cutlass.Array(cutlass.Float32, 32, alignment=16)
    lane_col_pair = (cute.arch.lane_idx() % 4) * 2
    for repeat in cutlass.range_constexpr(8):
        reg_base = repeat * 4
        scale_dim = key_half_start + repeat * 8 + lane_col_pair
        scale = (state_scale_f32_ptr + scale_dim).load(count=2, alignment=8)
        scaled_state[reg_base], scaled_state[reg_base + 1] = fmul2(
            (state[reg_base], state[reg_base + 1]),
            (scale[0], scale[1]),
        )
        scaled_state[reg_base + 2], scaled_state[reg_base + 3] = fmul2(
            (state[reg_base + 2], state[reg_base + 3]),
            (scale[0], scale[1]),
        )

    prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    state_input_ready_arrive(state_input_ready_mbar)
    prims.tcgen05_st("16x256b", state_ptr, scaled_state[0:32])
    prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)


@cute.jit
def tcgen05_issue_state_projection_mma(
    tcgen05_decay_smem,
    tmem_raw_addr,
    acc_ready_mbar,
    tmem_col_offset,
    input_dtype: cutlass.Constexpr,
    K_BLOCK_BEGIN: cutlass.Constexpr,
    K_BLOCK_END: cutlass.Constexpr,
    INITIAL_SCALE_D: cutlass.Constexpr,
    COMMIT: cutlass.Constexpr,
    M_DIM: cutlass.Constexpr,
) -> None:
    """Issue state*decay K-slices through tcgen05, optionally committing."""

    tmem_ptr = cutlass.inttoptr(
        tmem_raw_addr + tmem_col_offset,
        6,
        cutlass.Float32,
    )
    idesc = prims.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=input_dtype,
        b_dtype=input_dtype,
        n_dim=BT,
        m_dim=M_DIM,
        b_major=0,
    )
    desc_k_decay = prims.Tcgen05SmemDesc.build(
        tcgen05_decay_smem.subview(0),
        leading_byte_offset=TCGEN05_STATE_K_B_LEADING_BYTES,
        stride_byte_offset=TCGEN05_STATE_K_B_STRIDE_BYTES,
        layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
    )

    for k_block in cutlass.range_constexpr(K_BLOCK_BEGIN, K_BLOCK_END):
        scale_d = INITIAL_SCALE_D or k_block != K_BLOCK_BEGIN
        k_decay_offset = (
            k_block % TCGEN05_SW128_K_PHASES_PER_SLICE
        ) * TCGEN05_STATE_K_B_K_STEP_BYTES + (
            k_block // TCGEN05_SW128_K_PHASES_PER_SLICE
        ) * BT * TCGEN05_SW128_BYTES
        state_a_tmem = prims.make_tmem_ptr(tmem_raw_addr, cutlass.Int8).subview(
            KDA_TMEM_STATE_AS_INPUT_COL_OFFSET + k_block * (TCGEN05_F16_K_ATOM // 2)
        )
        if prims.elect_sync():
            prims.tcgen05_mma(
                prims.Tcgen05MMAKind.F16,
                prims.CTAGroup.CTA_1,
                tmem_ptr,
                state_a_tmem,
                desc_k_decay.advance_start_address(k_decay_offset),
                idesc,
                scale_d,
            )

    if cutlass.const_expr(COMMIT):
        if prims.elect_sync():
            prims.tcgen05_commit(acc_ready_mbar, group=prims.CTAGroup.CTA_1)


@cute.jit
def tcgen05_issue_state_k_mma(
    tcgen05_k_decay_smem,
    tmem_raw_addr,
    acc_ready_mbar,
    shared_acc_stage,
    input_dtype: cutlass.Constexpr,
    K_BLOCK_BEGIN: cutlass.Constexpr,
    K_BLOCK_END: cutlass.Constexpr,
    INITIAL_SCALE_D: cutlass.Constexpr,
    COMMIT: cutlass.Constexpr,
    M_DIM: cutlass.Constexpr,
) -> None:
    """Issue a K-slice range of state*k into a scheduled shared_acc stage."""

    tcgen05_issue_state_projection_mma(
        tcgen05_k_decay_smem,
        tmem_raw_addr,
        acc_ready_mbar,
        tcgen05_shared_acc_tmem_col_offset(shared_acc_stage),
        input_dtype,
        K_BLOCK_BEGIN,
        K_BLOCK_END,
        INITIAL_SCALE_D,
        COMMIT,
        M_DIM,
    )


@cute.jit
def tcgen05_issue_state_q_mma(
    tcgen05_q_decay_smem,
    tmem_raw_addr,
    operand_smem_consumed_mbar,
    qstate_acc_stage,
    input_dtype: cutlass.Constexpr,
) -> None:
    """Issue state*q and release q_decay when the tensor core consumes it."""

    tcgen05_issue_state_projection_mma(
        tcgen05_q_decay_smem,
        tmem_raw_addr,
        operand_smem_consumed_mbar,
        tcgen05_qstate_acc_tmem_col_offset(qstate_acc_stage),
        input_dtype,
        0,
        DK // TCGEN05_F16_K_ATOM,
        False,
        True,
        DV,
    )


@cute.jit
def tcgen05_wait_acc_buffer_ready(
    acc_ready_mbar,
    acc_ready_phase,
):
    """Wait until a producer commit has filled the corresponding TMEM acc tile."""

    while not prims.mbarrier_wait_parity(
        acc_ready_mbar,
        acc_ready_phase,
        prims.MBarrierWait.TRY,
    ):
        pass
    return acc_ready_phase ^ cutlass.Int32(1)


@cute.jit
def tcgen05_rhs_token_pair_from_16x256b_fragment(
    fragment,
    reg_idx: cutlass.Constexpr,
):
    """Select the lane-local state*k pair needed by the RHS TMEM store."""

    if cutlass.const_expr(reg_idx == 0):
        return fragment[4], fragment[5]
    if cutlass.const_expr(reg_idx == 1):
        return fragment[6], fragment[7]
    if cutlass.const_expr(reg_idx == 2):
        return fragment[0], fragment[1]
    return fragment[2], fragment[3]


@cute.jit
def tcgen05_stage_rhs_input_tmem(
    tmem_raw_addr,
    raw_v_smem,
    raw_beta_smem,
    warp_idx,
    lane,
    shared_acc_stage,
    shared_input_stage,
    input_dtype: cutlass.Constexpr,
) -> None:
    """Produce RHS = beta * (v - state*k) into shared_input TMEM by 16-row tiles."""

    base_col_id = tmem_raw_addr & 0xFFFF
    base_row_id = tmem_raw_addr >> 16
    tmem_sp = warp_idx % TCGEN05_STATE_K_TMEM_ROW_BLOCKS

    projection_col_id = base_col_id + tcgen05_shared_acc_tmem_col_offset(
        shared_acc_stage
    )
    input_col_id = base_col_id + tcgen05_shared_input_tmem_col_offset(
        shared_input_stage
    )
    value_dim_base = tmem_sp * THREADS_PER_WARP

    row_id0 = base_row_id + value_dim_base
    block_addr0 = (row_id0 << 16) | projection_col_id
    block_ptr0 = cutlass.inttoptr(block_addr0, 6, cutlass.Float32)
    state_k0 = prims.tcgen05_ld(
        "16x256b",
        block_ptr0,
        num=2,
    )

    row_id1 = row_id0 + 16
    block_addr1 = (row_id1 << 16) | projection_col_id
    block_ptr1 = cutlass.inttoptr(block_addr1, 6, cutlass.Float32)
    state_k1 = prims.tcgen05_ld(
        "16x256b",
        block_ptr1,
        num=2,
    )

    raw_v_regs0 = prims.ldmatrix(
        raw_v_ldmatrix_trans_ptr(
            raw_v_smem,
            value_dim_base,
            lane,
        ),
        4,
        prims.MMALayout.COL,
    )
    raw_v_regs1 = prims.ldmatrix(
        raw_v_ldmatrix_trans_ptr(
            raw_v_smem,
            value_dim_base + 16,
            lane,
        ),
        4,
        prims.MMALayout.COL,
    )
    prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)

    packed_rhs0 = cutlass.Array(
        cutlass.Int32,
        4,
        space=cutlass.AddressSpace.rmem,
    )
    for reg_idx in cutlass.range_constexpr(4):
        packed_col = (reg_idx // 2) * 4 + (lane & 3)
        source_pair = packed_col ^ TCGEN05_F16_A_TMEM_PAIR_XOR
        token0 = source_pair * 2
        token1 = token0 + 1
        beta0 = raw_beta_smem[token0].to(cutlass.Float32)
        beta1 = raw_beta_smem[token1].to(cutlass.Float32)
        raw_matrix = (1 - (reg_idx // 2)) * 2 + (reg_idx & 1)
        state_k_val0, state_k_val1 = tcgen05_rhs_token_pair_from_16x256b_fragment(
            state_k0,
            reg_idx,
        )
        beta_pair = pack_input_b16x2_to_i32(beta0, beta1, input_dtype)
        state_k_pair = pack_input_b16x2_to_i32(
            state_k_val0,
            state_k_val1,
            input_dtype,
        )
        diff_pair = sub_b16x2_input_dtype(
            raw_v_regs0[raw_matrix],
            state_k_pair,
            input_dtype,
        )
        packed_rhs0[reg_idx] = mul_b16x2_input_dtype(
            beta_pair,
            diff_pair,
            input_dtype,
        )

    packed_rhs1 = cutlass.Array(
        cutlass.Int32,
        4,
        space=cutlass.AddressSpace.rmem,
    )
    for reg_idx in cutlass.range_constexpr(4):
        packed_col = (reg_idx // 2) * 4 + (lane & 3)
        source_pair = packed_col ^ TCGEN05_F16_A_TMEM_PAIR_XOR
        token0 = source_pair * 2
        token1 = token0 + 1
        beta0 = raw_beta_smem[token0].to(cutlass.Float32)
        beta1 = raw_beta_smem[token1].to(cutlass.Float32)
        raw_matrix = (1 - (reg_idx // 2)) * 2 + (reg_idx & 1)
        state_k_val0, state_k_val1 = tcgen05_rhs_token_pair_from_16x256b_fragment(
            state_k1,
            reg_idx,
        )
        beta_pair = pack_input_b16x2_to_i32(beta0, beta1, input_dtype)
        state_k_pair = pack_input_b16x2_to_i32(
            state_k_val0,
            state_k_val1,
            input_dtype,
        )
        diff_pair = sub_b16x2_input_dtype(
            raw_v_regs1[raw_matrix],
            state_k_pair,
            input_dtype,
        )
        packed_rhs1[reg_idx] = mul_b16x2_input_dtype(
            beta_pair,
            diff_pair,
            input_dtype,
        )

    input_block_addr0 = (base_row_id << 16) | input_col_id
    input_block_ptr0 = prims.make_tmem_ptr(input_block_addr0, cutlass.Int8)
    prims.tcgen05_st("16x128b", input_block_ptr0, packed_rhs0[0:4])

    input_block_addr1 = ((base_row_id + 16) << 16) | input_col_id
    input_block_ptr1 = prims.make_tmem_ptr(input_block_addr1, cutlass.Int8)
    prims.tcgen05_st("16x128b", input_block_ptr1, packed_rhs1[0:4])

    prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)


@cute.jit
def tcgen05_stage_update_input_tmem(
    tmem_raw_addr,
    warp_idx,
    shared_acc_stage,
    shared_input_stage,
    input_dtype: cutlass.Constexpr,
) -> None:
    """Move update from shared_acc TMEM into packed shared_input TMEM."""

    base_col_id = tmem_raw_addr & 0xFFFF
    base_row_id = tmem_raw_addr >> 16
    tmem_sp = warp_idx % TCGEN05_STATE_K_TMEM_ROW_BLOCKS

    row_id = base_row_id + tmem_sp * THREADS_PER_WARP
    projection_col_id = base_col_id + tcgen05_shared_acc_tmem_col_offset(
        shared_acc_stage
    )
    block_addr = (row_id << 16) | projection_col_id
    block_ptr = cutlass.inttoptr(block_addr, 6, cutlass.Float32)
    update = prims.tcgen05_ld(
        "32x32b",
        block_ptr,
        num=TCGEN05_TMEM_LOAD_COLS,
    )
    prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)

    packed_update = cutlass.Array(
        cutlass.Int32,
        KDA_TMEM_SHARED_INPUT_COLS,
        alignment=16,
    )
    for packed_col in cutlass.range_constexpr(KDA_TMEM_SHARED_INPUT_COLS):
        source_pair = packed_col ^ TCGEN05_F16_A_TMEM_PAIR_XOR
        token0 = source_pair * 2
        token1 = token0 + 1
        packed_update[packed_col] = pack_input_b16x2_to_i32(
            update[token0],
            update[token1],
            input_dtype,
        )

    col_id = base_col_id + tcgen05_shared_input_tmem_col_offset(shared_input_stage)
    input_block_addr = (base_row_id << 16) | col_id
    input_block_ptr = prims.make_tmem_ptr(input_block_addr, cutlass.Int8)
    prims.tcgen05_st(
        "32x32b",
        input_block_ptr,
        packed_update[0:KDA_TMEM_SHARED_INPUT_COLS],
    )
    prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)


@cute.jit
def tcgen05_issue_value_pairwise_mma(
    pairwise_stage_smem,
    pairwise_tile_offset: cutlass.Constexpr,
    tmem_raw_addr,
    acc_ready_mbar,
    shared_input_stage,
    tmem_col_offset,
    scale_d: cutlass.Constexpr,
    input_dtype: cutlass.Constexpr,
) -> None:
    """Issue a `[DV,BT] @ [BT,BT]` value-side tcgen05 MMA.

    The A operand is a staged value tile (`rhs` or `update`) in the scheduled
    shared_input TMEM slot.  The B operand is a pairwise tile staged as
    `[N=token_i, K=token_j]` for tcgen05.
    """

    tmem_ptr = cutlass.inttoptr(
        tmem_raw_addr + tmem_col_offset,
        6,
        cutlass.Float32,
    )
    idesc = prims.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=input_dtype,
        b_dtype=input_dtype,
        n_dim=BT,
        m_dim=DV,
        b_major=0,
    )
    lhs_tmem = prims.make_tmem_ptr(tmem_raw_addr, cutlass.Int8).subview(
        tcgen05_shared_input_tmem_col_offset(shared_input_stage)
    )
    desc_pairwise = prims.Tcgen05SmemDesc.build(
        pairwise_stage_smem.subview(pairwise_tile_offset),
        leading_byte_offset=TCGEN05_VALUE_PAIRWISE_B_LEADING_BYTES,
        stride_byte_offset=TCGEN05_VALUE_PAIRWISE_B_STRIDE_BYTES,
        layout=prims.Tcgen05SmemSwizzle.SWIZZLE_32B,
    )

    if prims.elect_sync():
        prims.tcgen05_mma(
            prims.Tcgen05MMAKind.F16,
            prims.CTAGroup.CTA_1,
            tmem_ptr,
            lhs_tmem,
            desc_pairwise,
            idesc,
            scale_d,
        )
        prims.tcgen05_commit(acc_ready_mbar, group=prims.CTAGroup.CTA_1)


@cute.jit
def tcgen05_issue_update_mma(
    pairwise_stage_smem,
    tmem_raw_addr,
    acc_ready_mbar,
    shared_acc_stage,
    shared_input_stage,
    input_dtype: cutlass.Constexpr,
) -> None:
    """Issue update = A_inv @ rhs into a scheduled shared_acc TMEM stage."""

    tcgen05_issue_value_pairwise_mma(
        pairwise_stage_smem,
        PAIRWISE_SMEM_AINV_OFFSET,
        tmem_raw_addr,
        acc_ready_mbar,
        shared_input_stage,
        tcgen05_shared_acc_tmem_col_offset(shared_acc_stage),
        False,
        input_dtype,
    )


@cute.jit
def tcgen05_issue_qkv_mma(
    pairwise_stage_smem,
    tmem_raw_addr,
    acc_ready_mbar,
    shared_input_stage,
    qstate_acc_stage,
    input_dtype: cutlass.Constexpr,
) -> None:
    """Accumulate qkv = qk @ update into the live qstate_acc TMEM slot."""

    tcgen05_issue_value_pairwise_mma(
        pairwise_stage_smem,
        PAIRWISE_SMEM_QK_OFFSET,
        tmem_raw_addr,
        acc_ready_mbar,
        shared_input_stage,
        tcgen05_qstate_acc_tmem_col_offset(qstate_acc_stage),
        True,
        input_dtype,
    )


@cute.jit
def tcgen05_load_qstate_output_tmem(
    tmem_raw_addr,
    o_smem,
    warp_idx,
    lane,
    o_stage_base,
    qstate_acc_stage,
    scale: cutlass.Float32,
    output_dtype: cutlass.Constexpr,
) -> None:
    """Drain `state_q + qkv` from qstate_acc TMEM with STSM.T output staging."""

    base_col_id = tmem_raw_addr & 0xFFFF
    base_row_id = tmem_raw_addr >> 16
    tmem_sp = warp_idx % TCGEN05_STATE_K_TMEM_ROW_BLOCKS

    projection_col_id = base_col_id + tcgen05_qstate_acc_tmem_col_offset(
        qstate_acc_stage
    )
    value_dim_base = tmem_sp * THREADS_PER_WARP

    row_id0 = base_row_id + value_dim_base
    row_id1 = row_id0 + 16
    block_addr0 = (row_id0 << 16) | projection_col_id
    block_addr1 = (row_id1 << 16) | projection_col_id
    block_ptr0 = cutlass.inttoptr(block_addr0, 6, cutlass.Float32)
    block_ptr1 = cutlass.inttoptr(block_addr1, 6, cutlass.Float32)
    loaded0 = prims.tcgen05_ld(
        "16x256b",
        block_ptr0,
        num=2,
    )
    loaded1 = prims.tcgen05_ld(
        "16x256b",
        block_ptr1,
        num=2,
    )
    prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)

    stsm_regs0 = cutlass.Array(
        cutlass.Int32,
        4,
        space=cutlass.AddressSpace.rmem,
    )
    stsm_regs1 = cutlass.Array(
        cutlass.Int32,
        4,
        space=cutlass.AddressSpace.rmem,
    )
    for reg_idx in cutlass.range_constexpr(4):
        scaled0_0, scaled0_1 = fmul2(
            (loaded0[2 * reg_idx], loaded0[2 * reg_idx + 1]),
            (scale, scale),
        )
        scaled1_0, scaled1_1 = fmul2(
            (loaded1[2 * reg_idx], loaded1[2 * reg_idx + 1]),
            (scale, scale),
        )
        stsm_regs0[reg_idx] = pack_output_b16x2_to_i32(
            scaled0_0,
            scaled0_1,
            output_dtype,
        )
        stsm_regs1[reg_idx] = pack_output_b16x2_to_i32(
            scaled1_0,
            scaled1_1,
            output_dtype,
        )

    smem_dst0 = o_smem_stmatrix_128b_ptr(
        o_smem,
        o_stage_base,
        value_dim_base,
        lane,
    )
    smem_dst1 = o_smem_stmatrix_128b_ptr(
        o_smem,
        o_stage_base,
        value_dim_base + 16,
        lane,
    )
    prims.stmatrix(
        smem_dst0,
        stsm_regs0.data_ptr().load(count=4, alignment=4),
        prims.MMALayout.COL,
        shape=prims.StoreShape.M8N8,
    )
    prims.stmatrix(
        smem_dst1,
        stsm_regs1.data_ptr().load(count=4, alignment=4),
        prims.MMALayout.COL,
        shape=prims.StoreShape.M8N8,
    )
    cute.arch.fence_view_async_shared()


@cute.jit
def epilogue_stage_store(
    tma_desc_o: cutlass.GridConstant[cuda.TensorMap],
    o_smem,
    sequence_start,
    head_idx,
    chunk_start,
    o_stage_base,
) -> None:
    """Store the staged `[BT, DV]` output tile to global memory with TMA."""

    global_chunk_start = sequence_start + chunk_start
    if prims.elect_sync():
        for value_segment in cutlass.range_constexpr(O_TMA_SEGMENTS):
            segment_base = (
                o_stage_base + O_OUT_OFFSET + value_segment * BT * O_TMA_SWIZZLE_ELEMS
            )
            o_coord = (
                cutlass.Int32(value_segment * O_TMA_SWIZZLE_ELEMS),
                global_chunk_start,
                head_idx,
                cutlass.Int32(0),
            )
            prims.cp_async_bulk_tensor_global_shared_cta(
                tma_desc_o.get_ptr(),
                o_smem.subview(segment_base),
                o_coord,
            )
        prims.cp_async_bulk_commit_group()
        prims.cp_async_bulk_wait_group(0, read=True)
    prims.bar_warp_sync(cute.arch.FULL_MASK)


@cute.jit
def epilogue_tail_store(
    out,
    o_smem,
    sequence_start,
    head_idx,
    chunk_start,
    seqlen,
    o_stage_base,
    lane,
) -> None:
    """Store a partial packed-sequence tail without crossing its boundary."""

    valid_tokens = seqlen - chunk_start
    for elem_iter in cutlass.range_constexpr((BT * DV) // THREADS_PER_WARP):
        linear_idx = elem_iter * THREADS_PER_WARP + lane
        token_coord = linear_idx // DV
        value_dim = linear_idx - token_coord * DV
        if token_coord < valid_tokens:
            smem_idx = o_smem_swizzle_128b_elem_index(
                o_stage_base,
                value_dim,
                token_coord,
            )
            out[0, sequence_start + chunk_start + token_coord, head_idx, value_dim] = (
                o_smem[smem_idx]
            )
    prims.bar_warp_sync(cute.arch.FULL_MASK)


@cute.jit
def epilogue_wait_and_store_full_output(
    tma_desc_o: cutlass.GridConstant[cuda.TensorMap],
    o_smem,
    output_ready_mbar,
    output_consumed_mbar,
    sequence_start,
    head_idx,
    output_chunk,
    O_STAGES: cutlass.Constexpr,
):
    """Drain one full staged output chunk from SMEM with TMA."""

    output_chunk_start = output_chunk * BT
    o_stage = output_chunk % O_STAGES
    o_stage_base = o_stage * O_SMEM_STAGE_SIZE
    output_ready_wait(
        output_ready_mbar.subview(o_stage),
        (output_chunk // O_STAGES) % 2,
    )
    epilogue_stage_store(
        tma_desc_o,
        o_smem,
        sequence_start,
        head_idx,
        output_chunk_start,
        o_stage_base,
    )
    output_consumed_arrive(output_consumed_mbar.subview(o_stage))


@cute.jit
def epilogue_wait_and_store_final_output(
    tma_desc_o: cutlass.GridConstant[cuda.TensorMap],
    out,
    o_smem,
    output_ready_mbar,
    output_consumed_mbar,
    sequence_start,
    head_idx,
    seqlen,
    output_chunk,
    lane,
    O_STAGES: cutlass.Constexpr,
):
    """Drain the final output chunk, guarding a partial packed tail."""

    if seqlen % BT == 0:
        epilogue_wait_and_store_full_output(
            tma_desc_o,
            o_smem,
            output_ready_mbar,
            output_consumed_mbar,
            sequence_start,
            head_idx,
            output_chunk,
            O_STAGES,
        )
    else:
        output_chunk_start = output_chunk * BT
        o_stage = output_chunk % O_STAGES
        o_stage_base = o_stage * O_SMEM_STAGE_SIZE
        output_ready_wait(
            output_ready_mbar.subview(o_stage),
            (output_chunk // O_STAGES) % 2,
        )
        epilogue_tail_store(
            out,
            o_smem,
            sequence_start,
            head_idx,
            output_chunk_start,
            seqlen,
            o_stage_base,
            lane,
        )
        output_consumed_arrive(output_consumed_mbar.subview(o_stage))


@cute.jit
def tcgen05_issue_final_state_delta_mma(
    tcgen05_k_restore_smem,
    tmem_raw_addr,
    k_restore_consumed_l_mbar,
    k_restore_consumed_mbar,
    shared_input_stage,
    input_dtype: cutlass.Constexpr,
) -> None:
    """Accumulate final_state += update @ k_restore in two N=64 halves.

    The left half commits to its own mbarrier so the next chunk's left
    state pack can begin while the right half is still in flight; the
    right commit (which orders after every prior MMA) keeps the original
    full-completion contract for CG0 and the right pack.
    """

    half_n = DK // 2
    tmem_ptr_l = cutlass.inttoptr(
        tmem_raw_addr + KDA_TMEM_FINAL_STATE_ACC_COL_OFFSET,
        6,
        cutlass.Float32,
    )
    tmem_ptr_r = cutlass.inttoptr(
        tmem_raw_addr + KDA_TMEM_FINAL_STATE_ACC_COL_OFFSET + half_n,
        6,
        cutlass.Float32,
    )
    idesc = prims.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=input_dtype,
        b_dtype=input_dtype,
        n_dim=half_n,
        m_dim=DV,
        b_major=1,
    )
    update_tmem = prims.make_tmem_ptr(tmem_raw_addr, cutlass.Int8).subview(
        tcgen05_shared_input_tmem_col_offset(shared_input_stage)
    )
    desc_k_restore = prims.Tcgen05SmemDesc.build(
        tcgen05_k_restore_smem.subview(0),
        leading_byte_offset=TCGEN05_FINAL_STATE_B_LEADING_BYTES,
        stride_byte_offset=TCGEN05_FINAL_STATE_B_STRIDE_BYTES,
        layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
    )

    if prims.elect_sync():
        prims.tcgen05_mma(
            prims.Tcgen05MMAKind.F16,
            prims.CTAGroup.CTA_1,
            tmem_ptr_l,
            update_tmem,
            desc_k_restore,
            idesc,
            True,
        )
        prims.tcgen05_commit(
            k_restore_consumed_l_mbar,
            group=prims.CTAGroup.CTA_1,
        )
        prims.tcgen05_mma(
            prims.Tcgen05MMAKind.F16,
            prims.CTAGroup.CTA_1,
            tmem_ptr_r,
            update_tmem,
            desc_k_restore.advance_start_address(TCGEN05_FINAL_STATE_B_LEADING_BYTES),
            idesc,
            True,
        )
        prims.tcgen05_commit(
            k_restore_consumed_mbar,
            group=prims.CTAGroup.CTA_1,
        )


@cute.jit
def tcgen05_store_final_state_tmem(
    tmem_raw_addr,
    state_col_offset,
    final_state: cute.Tensor,
    bidx,
    bidy,
    dv_half,
    warp_idx,
    lane,
    HALF: cutlass.Constexpr,
) -> None:
    """Store the live recurrent TMEM state to the VK final_state tensor.

    HALF=True is the DV2 chain form: each M=64 half stores its own rows
    (Layout F -- only the quadrant lanes 0..15 carry state).
    """

    base_col_id = tmem_raw_addr & 0xFFFF
    base_row_id = tmem_raw_addr >> 16
    tmem_sp = warp_idx % TCGEN05_STATE_K_TMEM_ROW_BLOCKS

    row_id = base_row_id + tmem_sp * THREADS_PER_WARP
    if cutlass.const_expr(HALF):
        value_dim = (
            dv_half * cutlass.Int32(DV_HALF)
            + tmem_sp * ROWS_PER_WARP
            + (lane % ROWS_PER_WARP)
        )
        valid_lane = lane < ROWS_PER_WARP
    else:
        value_dim = tmem_sp * THREADS_PER_WARP + lane
    for key_block_start in cutlass.range_constexpr(
        0,
        DK,
        TCGEN05_FINAL_STATE_TMEM_LOAD_COLS,
    ):
        projection_col_id = base_col_id + state_col_offset + key_block_start
        block_addr = (row_id << 16) | projection_col_id
        block_ptr = cutlass.inttoptr(block_addr, 6, cutlass.Float32)
        loaded = prims.tcgen05_ld(
            "32x32b",
            block_ptr,
            num=TCGEN05_FINAL_STATE_TMEM_LOAD_COLS,
        )
        prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)

        for col in cutlass.range_constexpr(TCGEN05_FINAL_STATE_TMEM_LOAD_COLS):
            key_dim = key_block_start + col
            if cutlass.const_expr(HALF):
                if valid_lane:
                    final_state[bidx, bidy, value_dim, key_dim] = loaded[col].to(
                        final_state.element_type
                    )
            else:
                final_state[bidx, bidy, value_dim, key_dim] = loaded[col].to(
                    final_state.element_type
                )


PACKED_METADATA_THREADS: int = 256


@cute.kernel
def packed_metadata_kernel(
    cu_seqlens: cute.Tensor,
    seq_order: cute.Tensor,
    cu_chunks: cute.Tensor,
    generate_seq_order: cutlass.Int32,
    generate_cu_chunks: cutlass.Int32,
) -> None:
    """Build packed scheduling metadata from device-resident prefix offsets."""

    tidx, _, _ = cute.arch.thread_idx()
    n_seq = cutlass.Int32(seq_order.shape[0])
    if generate_seq_order != cutlass.Int32(0):
        for index in cutlass.range(tidx, n_seq, PACKED_METADATA_THREADS):
            seq_order[index] = cutlass.Int32(index)
        cute.arch.sync_threads()

        # Stable descending-length order.  The issue workload has at most 64
        # sequences; odd-even transposition keeps the implementation compact
        # and supports a runtime sequence count without temporary storage.
        for phase in cutlass.range(n_seq):
            parity = phase & cutlass.Int32(1)
            pair_count = (n_seq - parity) // cutlass.Int32(2)
            for pair in cutlass.range(tidx, pair_count, PACKED_METADATA_THREADS):
                left = parity + pair * cutlass.Int32(2)
                right = left + cutlass.Int32(1)
                left_seq = cutlass.Int32(seq_order[left])
                right_seq = cutlass.Int32(seq_order[right])
                left_len = cu_seqlens[left_seq + 1] - cu_seqlens[left_seq]
                right_len = cu_seqlens[right_seq + 1] - cu_seqlens[right_seq]
                swap = (right_len > left_len) | (
                    (right_len == left_len) & (right_seq < left_seq)
                )
                if swap:
                    seq_order[left] = right_seq
                    seq_order[right] = left_seq
            cute.arch.sync_threads()

    if (generate_cu_chunks != cutlass.Int32(0)) & (tidx == cutlass.Int32(0)):
        cu_chunks[0] = cutlass.Int32(0)
        total_chunks = cutlass.Int32(0)
        for index in cutlass.range(n_seq):
            seq_len = cutlass.Int32(cu_seqlens[index + 1] - cu_seqlens[index])
            total_chunks += (seq_len + cutlass.Int32(BT - 1)) // cutlass.Int32(BT)
            cu_chunks[index + 1] = total_chunks


@cute.kernel
def kernel(
    tma_desc_q: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_k: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_v: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_gate: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_beta: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_o: cutlass.GridConstant[cuda.TensorMap],
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    raw_gate: cute.Tensor,
    a_log: cute.Tensor,
    dt_bias: cute.Tensor,
    beta: cute.Tensor,
    cu_seqlens: cute.Tensor,
    seq_order: cute.Tensor,
    state_indices: cute.Tensor | None,
    initial_state: cute.Tensor | None,
    out: cute.Tensor,
    final_state: cute.Tensor | None,
    SCALE: cutlass.Float32,
    state_ckpt: cute.Tensor | None,
    cu_ckpts: cute.Tensor | None,
    checkpoint_state_indices: cute.Tensor | None,
    checkpoint_stride_chunks: cutlass.Int32,
    SAFE_GATE: cutlass.Constexpr,
    GATE_SCALE_LOG2: cutlass.Constexpr,
    gate_dtype: cutlass.Constexpr,
) -> None:
    """BT=16 KDA forward kernel.

    Grid: `(heads, num_sequences, 1)`. Each CTA owns one packed sequence/head
    and iterates over `ceil(sequence_length / 16)` chunks in order.
    """

    tidx, _, _ = cute.arch.thread_idx()
    # Put heads in grid-x so every sequence's head CTAs are contiguous in the
    # linear launch order. sequence_slot follows the automatic longest-first
    # permutation; bidx/bidy remain the original sequence/head indices below.
    bidy, sequence_slot, _ = cute.arch.block_idx()
    bidx = cutlass.Int32(seq_order[sequence_slot])
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = tidx % THREADS_PER_WARP

    sequence_start = cutlass.Int32(cu_seqlens[bidx])
    # beta TMA transport shape: heads from grid-x; g = 8/gcd(heads, 8) via the
    # lowest set bit (g==1 -> 8-head-group box, g>1 -> pair-packed rows).
    heads32 = cutlass.Int32(cute.arch.grid_dim()[0])
    beta_lsb = heads32 & (-heads32)
    beta_g = cutlass.Int32(8) // cutlass.min(beta_lsb, cutlass.Int32(8))
    sequence_end = cutlass.Int32(cu_seqlens[bidx + 1])
    seqlen = sequence_end - sequence_start
    num_chunks = cute.ceil_div(seqlen, BT)
    input_dtype = q.element_type
    if cutlass.const_expr(
        k.element_type != input_dtype or v.element_type != input_dtype
    ):
        raise TypeError(
            "KDA CUTLASS primitives kernel expects q/k/v to use the same 16-bit dtype"
        )
    if cutlass.const_expr(raw_gate.element_type != gate_dtype):
        raise TypeError(
            "KDA CUTLASS primitives kernel expects raw_gate to match gate_dtype"
        )
    if cutlass.const_expr(beta.element_type != cutlass.BFloat16):
        raise TypeError("KDA CUTLASS primitives kernel expects beta logits to use BF16")
    if cutlass.const_expr(cu_seqlens.element_type != cutlass.Int64):
        raise TypeError("KDA CUTLASS primitives kernel expects cu_seqlens to use int64")
    if cutlass.const_expr(initial_state is not None):
        if cutlass.const_expr(
            initial_state.element_type not in (cutlass.BFloat16, cutlass.Float32)
        ):
            raise TypeError(
                "KDA CUTLASS primitives kernel input state dtype must be BF16 or FP32"
            )
    if cutlass.const_expr(final_state is not None):
        if cutlass.const_expr(
            final_state.element_type not in (cutlass.BFloat16, cutlass.Float32)
        ):
            raise TypeError(
                "KDA CUTLASS primitives kernel output state dtype must be BF16 or FP32"
            )
    if cutlass.const_expr(initial_state is not None and final_state is not None):
        if cutlass.const_expr(initial_state.element_type != final_state.element_type):
            raise TypeError(
                "KDA CUTLASS primitives kernel expects matching state input/output dtypes"
            )
    # Buffers are declaration-ordered and intentionally non-aliased.
    tma_mbar = cutlass.Array(
        cutlass.Int64,
        TMA_MBAR_STAGE_COUNT,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    qstate_acc_ready_mbar = cutlass.Array(
        cutlass.Int64,
        1,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    output_ready_mbar = cutlass.Array(
        cutlass.Int64,
        O_STAGE_COUNT,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    q_k_restore_ready_mbar = cutlass.Array(
        cutlass.Int64,
        Q_K_RESTORE_READY_STAGE_COUNT,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    cg0_k_ready_mbar = cutlass.Array(
        cutlass.Int64,
        DECAY_STAGE_COUNT,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    cg0_k_half_ready_mbar = cutlass.Array(
        cutlass.Int64,
        DECAY_STAGE_COUNT,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    diag_ready_mbar = cutlass.Array(
        cutlass.Int64,
        RAW_STAGE_COUNT,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    raw_consumed_mbar = cutlass.Array(
        cutlass.Int64,
        RAW_STAGE_COUNT,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    state_input_ready_mbar = cutlass.Array(
        cutlass.Int64,
        1,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    state_input_ready_l_mbar = cutlass.Array(
        cutlass.Int64,
        1,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    initial_state_ready_mbar = cutlass.Array(
        cutlass.Int64,
        1,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    checkpoint_read_done_mbar = cutlass.Array(
        cutlass.Int64,
        1,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    operand_smem_consumed_mbar = cutlass.Array(
        cutlass.Int64,
        DECAY_STAGE_COUNT,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    rhs_ready_mbar = cutlass.Array(
        cutlass.Int64,
        1,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    update_ready_mbar = cutlass.Array(
        cutlass.Int64,
        1,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    output_consumed_mbar = cutlass.Array(
        cutlass.Int64,
        O_STAGE_COUNT,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    final_state_stored_mbar = cutlass.Array(
        cutlass.Int64,
        1,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    pairwise_ready_mbar = cutlass.Array(
        cutlass.Int64,
        PAIRWISE_STAGE_COUNT,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    qk_ready_mbar = cutlass.Array(
        cutlass.Int64,
        PAIRWISE_STAGE_COUNT,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    pairwise_consumed_mbar = cutlass.Array(
        cutlass.Int64,
        PAIRWISE_STAGE_COUNT,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    k_restore_consumed_mbar = cutlass.Array(
        cutlass.Int64,
        DECAY_STAGE_COUNT,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    k_restore_consumed_l_mbar = cutlass.Array(
        cutlass.Int64,
        DECAY_STAGE_COUNT,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    shared_acc_ready_mbar = cutlass.Array(
        cutlass.Int64,
        KDA_TMEM_SHARED_ACC_STAGE_COUNT,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    # The hand-written K-box-major SW128 mapping is normalized to phase 0,
    # so both tcgen05 and ldmatrix can share 1KB-aligned operand buffers.
    tmem_ptr_i32 = cutlass.Array(
        cutlass.Int32,
        1,
        space=cutlass.AddressSpace.smem,
        alignment=4,
    )
    tcgen05_k_decay_smem = cutlass.Array(
        input_dtype,
        TCGEN05_K_DECAY_SMEM_TILE_SIZE,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )
    tcgen05_q_decay_smem = cutlass.Array(
        input_dtype,
        TCGEN05_Q_DECAY_SMEM_TILE_SIZE,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )
    tcgen05_k_restore_smem = cutlass.Array(
        input_dtype,
        TCGEN05_K_RESTORE_SMEM_TILE_SIZE,
        space=cutlass.AddressSpace.smem,
        alignment=RAW_F16_TMA_SWIZZLE_ALIGNMENT_BYTES,
    )
    pairwise_smem = cutlass.Array(
        input_dtype,
        PAIRWISE_SMEM_TILE_SIZE,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )
    raw_q_smem = cutlass.Array(
        q.element_type,
        RAW_Q_SMEM_TILE_SIZE,
        space=cutlass.AddressSpace.smem,
        alignment=RAW_F16_TMA_SWIZZLE_ALIGNMENT_BYTES,
    )
    raw_k_smem = cutlass.Array(
        k.element_type,
        RAW_K_SMEM_TILE_SIZE,
        space=cutlass.AddressSpace.smem,
        alignment=RAW_F16_TMA_SWIZZLE_ALIGNMENT_BYTES,
    )
    raw_v_smem = cutlass.Array(
        v.element_type,
        RAW_V_SMEM_TILE_SIZE,
        space=cutlass.AddressSpace.smem,
        alignment=RAW_F16_TMA_SWIZZLE_ALIGNMENT_BYTES,
    )
    # Gate SMEM.  Sizes above are ELEMENT counts, so the ring's byte size
    # follows gate_dtype (FP32: 8 x 8 KB = 64 KB; BF16: 8 x 4 KB = 32 KB).
    # Under FP32 the exp2(g_prefix) exchange keeps aliasing this ring exactly
    # as it always did and NOTHING extra is allocated; a 16-bit gate cannot
    # host an FP32 exchange, so it gets a dedicated 4-deep FP32 ring (16 KB).
    if cutlass.const_expr(gate_dtype_is_f32(gate_dtype)):
        raw_gate_smem = cutlass.Array(
            cutlass.Float32,
            RAW_GATE_SMEM_TILE_SIZE,
            space=cutlass.AddressSpace.smem,
            alignment=RAW_F32_TMA_SWIZZLE_ALIGNMENT_BYTES,
        )
        gate_exchange_smem = None
    else:
        raw_gate_smem = cutlass.Array(
            gate_dtype,
            RAW_GATE_SMEM_TILE_SIZE,
            space=cutlass.AddressSpace.smem,
            alignment=RAW_F16_TMA_SWIZZLE_ALIGNMENT_BYTES,
        )
        gate_exchange_smem = cutlass.Array(
            cutlass.Float32,
            GATE_EXCHANGE_SMEM_TILE_SIZE,
            space=cutlass.AddressSpace.smem,
            alignment=RAW_F32_TMA_SWIZZLE_ALIGNMENT_BYTES,
        )
    k_inv_smem = cutlass.Array(
        input_dtype,
        K_INV_SMEM_TILE_SIZE,
        space=cutlass.AddressSpace.smem,
        alignment=RAW_F16_TMA_SWIZZLE_ALIGNMENT_BYTES,
    )
    o_smem = cutlass.Array(
        out.element_type,
        O_SMEM_TILE_SIZE,
        space=cutlass.AddressSpace.smem,
        # The scalar CG1 store computes W128 offsets relative to this buffer.
        # Align to the full s128b period so absolute SMEM address bits do not
        # add a hidden phase to the TMA store-side swizzle.
        alignment=O_TMA_SWIZZLE_ALIGNMENT_BYTES,
    )
    raw_dt_bias_smem = cutlass.Array(
        cutlass.Float32,
        RAW_DT_BIAS_SMEM_TILE_SIZE,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )
    raw_beta_smem = cutlass.Array(
        cutlass.Float32,
        RAW_BETA_SMEM_TILE_SIZE,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )
    beta_tile_smem = cutlass.Array(
        cutlass.BFloat16,
        BETA_TILE_STAGE_COUNT * BETA_TILE_STAGE_ELEMS,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )
    beta_tile_mbar = cutlass.Array(
        cutlass.Int64,
        BETA_TILE_STAGE_COUNT,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    # Actual bytes one beta tile transfers (the ring stage is padded to a
    # fixed stride; expect_tx must be the real box size).
    beta_tile_tx = cutlass.Int32(8 * BT * 2)
    if beta_g > cutlass.Int32(1):
        beta_tile_tx = heads32 * (cutlass.Int32(BT) + beta_g) * cutlass.Int32(2)
    tma_tx_bytes = cutlass.const_expr(
        DK * BT * q.element_type.width // 8
        + DK * BT * k.element_type.width // 8
        + DV * BT * v.element_type.width // 8
        + DK * BT * raw_gate.element_type.width // 8
    )

    if warp_idx == ROLES.tma_load:
        if prims.elect_sync():
            for stage in cutlass.range_constexpr(TMA_MBAR_STAGE_COUNT):
                prims.mbarrier_init(tma_mbar.subview(stage), 1)
            for stage in cutlass.range_constexpr(RAW_STAGE_COUNT):
                # Four owner-CG0 warps consume q/k/gate, four CG1 warps
                # consume v/beta, and the standalone auxiliary-MMA warp consumes
                # beta while constructing the pairwise tile.
                prims.mbarrier_init(raw_consumed_mbar.subview(stage), 9)
            for stage in cutlass.range_constexpr(BETA_TILE_STAGE_COUNT):
                prims.mbarrier_init(beta_tile_mbar.subview(stage), 1)
    elif warp_idx == ROLES.tcgen05_mma:
        if prims.elect_sync():
            prims.mbarrier_init(qstate_acc_ready_mbar, 1)
            for stage in cutlass.range_constexpr(KDA_TMEM_SHARED_ACC_STAGE_COUNT):
                prims.mbarrier_init(shared_acc_ready_mbar.subview(stage), 1)
            prims.mbarrier_init(state_input_ready_mbar, 4)
            prims.mbarrier_init(state_input_ready_l_mbar, 4)
            prims.mbarrier_init(initial_state_ready_mbar, 4)
            if cutlass.const_expr(state_ckpt is not None):
                prims.mbarrier_init(checkpoint_read_done_mbar, 4)
            for stage in cutlass.range_constexpr(DECAY_STAGE_COUNT):
                prims.mbarrier_init(operand_smem_consumed_mbar.subview(stage), 3)
                prims.mbarrier_init(k_restore_consumed_mbar.subview(stage), 1)
                prims.mbarrier_init(k_restore_consumed_l_mbar.subview(stage), 1)
            prims.mbarrier_init(rhs_ready_mbar, 4)
            prims.mbarrier_init(update_ready_mbar, 8)
            prims.mbarrier_init(final_state_stored_mbar, 4)
    elif warp_idx == ROLES.super_mma:
        if prims.elect_sync():
            for stage in cutlass.range_constexpr(PAIRWISE_STAGE_COUNT):
                prims.mbarrier_init(pairwise_ready_mbar.subview(stage), 1)
                prims.mbarrier_init(qk_ready_mbar.subview(stage), 1)
                prims.mbarrier_init(pairwise_consumed_mbar.subview(stage), 1)
            for stage in cutlass.range_constexpr(Q_K_RESTORE_READY_STAGE_COUNT):
                prims.mbarrier_init(q_k_restore_ready_mbar.subview(stage), 4)
            for stage in cutlass.range_constexpr(DECAY_STAGE_COUNT):
                prims.mbarrier_init(cg0_k_ready_mbar.subview(stage), 4)
                prims.mbarrier_init(cg0_k_half_ready_mbar.subview(stage), 4)
            for stage in cutlass.range_constexpr(RAW_STAGE_COUNT):
                prims.mbarrier_init(diag_ready_mbar.subview(stage), 4)
    elif warp_idx == ROLES.epilogue:
        if prims.elect_sync():
            for stage in cutlass.range_constexpr(O_STAGE_COUNT):
                prims.mbarrier_init(output_ready_mbar.subview(stage), 4)
                prims.mbarrier_init(output_consumed_mbar.subview(stage), 1)
    prims.fence_mbarrier_init()
    cta_sync()
    if is_tmem_user_warp(warp_idx):
        if warp_idx == ROLES.tcgen05_mma:
            prims.tcgen05_alloc(tmem_ptr_i32, KDA_TMEM_ALLOC_COLS, group="cta_1")
        tmem_user_sync()
        if warp_idx == ROLES.tcgen05_mma:
            prims.tcgen05_relinquish_alloc_permit(group="cta_1")
        tmem_user_sync()
    qstate_acc_ready_phase = cutlass.Int32(0)
    state_input_ready_phase = cutlass.Int32(0)
    state_input_ready_l_phase = cutlass.Int32(0)
    rhs_ready_phase = cutlass.Int32(0)
    update_ready_phase = cutlass.Int32(0)
    final_state_stored_phase = cutlass.Int32(0)

    # Actual SMEM/TMEM buffers for the BT=16 schedule:
    #   q/k/v              : 16 x 128 each
    #   gate_log2          : 16 x 128
    #   beta               : 16
    #   q/k inverse norm   : 16 each, staged once per decay stage
    #   exp_g_last         : 128, CG0-local and staged once per decay stage
    #   state decay        : row 15 of the fp32 gate-prefix exchange tile
    #   q_decay/k_decay    : tcgen05 SW128 operands shared with auxiliary MMA
    #   k_restore          : tcgen05 SW128 N-major final-state operand
    #   k_inv              : 16 x 128 token-major for auxiliary-MMA RHS
    #   A inverse/QK       : 16 x 16 each, plus transposed tcgen05 operands
    #   state              : external/kernel ABI is VK `[DV, DK]`; reference
    #                        math can view it as KV `[DK, DV]` by transposing.
    #                        The TS A-staging path keeps VK in TMEM so state*k
    #                        is `[DV, DK] @ [DK, BT] -> [DV, BT]` with M=128.

    if is_service_warpgroup(warp_idx):
        prims.setmaxregister(KDA_SERVICE_REGS, prims.SetMaxRegisterAction.DECREASE)
        if warp_idx == ROLES.tma_load:
            # Gate constants are needed by BOTH the safe sigmoid and the
            # FLA non-safe softplus activations (a_log_exp == exp(A_log),
            # dt_bias per key dim).  Materialize unconditionally; for
            # SAFE_GATE=True this traces exactly as before (byte-identical
            # generated code), the non-safe path now reads the same staged constants.
            if lane == 0:
                raw_dt_bias_smem[RAW_DT_BIAS_A_LOG_EXP_OFFSET] = cute.math.exp2(
                    a_log[bidy].to(cutlass.Float32) * LOG2_E,
                    fastmath=True,
                )
            for dim_group in cutlass.range_constexpr(DK // THREADS_PER_WARP):
                dim = dim_group * THREADS_PER_WARP + lane
                raw_dt_bias_smem[dim] = dt_bias[bidy, dim].to(cutlass.Float32)

            # Prime the beta tile ring: LOOKAHEAD tiles in flight before the
            # first consumption so the wait below is never exposed.  Group
            # coords walk (head-group, token); pair coords walk packed rows
            # from the g-aligned floor of the sequence start.
            beta_row0 = sequence_start // beta_g
            beta_rows_per_chunk = cutlass.Int32(BT) // beta_g
            beta_head_group = (bidy // cutlass.Int32(8)) * cutlass.Int32(8)
            for pre in cutlass.range_constexpr(BETA_TMA_LOOKAHEAD):
                if cutlass.Int32(pre) < num_chunks:
                    if prims.elect_sync():
                        prims.mbarrier_arrive_expect_tx(
                            beta_tile_mbar.subview(pre % BETA_TILE_STAGE_COUNT),
                            beta_tile_tx,
                        )
                        beta_c0 = beta_head_group
                        beta_c1 = sequence_start + cutlass.Int32(pre * BT)
                        if beta_g > cutlass.Int32(1):
                            beta_c0 = cutlass.Int32(0)
                            beta_c1 = (
                                beta_row0 + cutlass.Int32(pre) * beta_rows_per_chunk
                            )
                        prims.cp_async_bulk_tensor_shared_cta_global(
                            beta_tile_smem.subview(
                                (pre % BETA_TILE_STAGE_COUNT) * BETA_TILE_STAGE_ELEMS
                            ),
                            tma_desc_beta.get_ptr(),
                            (beta_c0, beta_c1),
                            beta_tile_mbar.subview(pre % BETA_TILE_STAGE_COUNT),
                        )
            raw_stage = cutlass.Int32(0)
            raw_consumed_phase = cutlass.Int32(1)
            for chunk in cutlass.range(num_chunks, unroll=1):
                chunk_start = chunk * BT
                beta_tile_slot = chunk % BETA_TILE_STAGE_COUNT
                tma_transfer_wait(
                    beta_tile_mbar.subview(beta_tile_slot),
                    (chunk // BETA_TILE_STAGE_COUNT) % 2,
                )
                raw_q_stage = raw_q_smem.subview(raw_stage * RAW_Q_STAGE_SIZE)
                raw_k_stage = raw_k_smem.subview(raw_stage * RAW_K_STAGE_SIZE)
                raw_v_stage = raw_v_smem.subview(raw_stage * RAW_V_STAGE_SIZE)
                raw_gate_stage = raw_gate_smem.subview(raw_stage * RAW_GATE_STAGE_SIZE)
                raw_beta_stage = raw_beta_smem.subview(raw_stage * RAW_BETA_STAGE_SIZE)
                raw_consumed_wait(
                    raw_consumed_mbar.subview(raw_stage),
                    raw_consumed_phase,
                )
                # TMA issues q/k/v/gate into typed SMEM against the chunk's
                # tma_mbar ring slot (consumers wait that slot's parity
                # directly; this warp never waits completions -- raw_consumed
                # alone throttles issue to <= RAW_STAGE_COUNT in flight).
                tma_stage_load_inputs(
                    tma_desc_q,
                    tma_desc_k,
                    tma_desc_v,
                    tma_desc_gate,
                    beta_tile_smem.subview(beta_tile_slot * BETA_TILE_STAGE_ELEMS),
                    beta_g,
                    heads32,
                    raw_q_stage,
                    raw_k_stage,
                    raw_v_stage,
                    raw_gate_stage,
                    raw_beta_stage,
                    sequence_start,
                    bidy,
                    lane,
                    chunk_start,
                    seqlen,
                    tma_mbar.subview(raw_stage),
                    tma_tx_bytes,
                    gate_dtype,
                )
                beta_next = chunk + BETA_TMA_LOOKAHEAD
                if beta_next < num_chunks:
                    beta_next_slot = beta_next % BETA_TILE_STAGE_COUNT
                    if prims.elect_sync():
                        prims.mbarrier_arrive_expect_tx(
                            beta_tile_mbar.subview(beta_next_slot),
                            beta_tile_tx,
                        )
                        beta_c0 = beta_head_group
                        beta_c1 = sequence_start + beta_next * BT
                        if beta_g > cutlass.Int32(1):
                            beta_c0 = cutlass.Int32(0)
                            beta_c1 = beta_row0 + beta_next * beta_rows_per_chunk
                        prims.cp_async_bulk_tensor_shared_cta_global(
                            beta_tile_smem.subview(
                                beta_next_slot * BETA_TILE_STAGE_ELEMS
                            ),
                            tma_desc_beta.get_ptr(),
                            (beta_c0, beta_c1),
                            beta_tile_mbar.subview(beta_next_slot),
                        )
                raw_stage, raw_wrapped = advance_ring_stage(
                    raw_stage,
                    1,
                    RAW_STAGE_COUNT,
                )
                raw_consumed_phase = raw_consumed_phase ^ raw_wrapped

        elif warp_idx == ROLES.super_mma:
            raw_stage = cutlass.Int32(0)
            for chunk in cutlass.range(num_chunks, unroll=1):
                decay_stage = chunk % DECAY_STAGE_COUNT
                pairwise_stage = chunk % PAIRWISE_STAGE_COUNT
                raw_beta_stage = raw_beta_smem.subview(raw_stage * RAW_BETA_STAGE_SIZE)
                k_inv_stage = k_inv_smem.subview(decay_stage * K_INV_STAGE_SIZE)
                tcgen05_k_decay_stage = tcgen05_k_decay_smem.subview(
                    decay_stage * TCGEN05_K_DECAY_STAGE_SIZE
                )
                pairwise_stage_smem = pairwise_smem.subview(
                    pairwise_stage * PAIRWISE_SMEM_STAGE_SIZE
                )

                pairwise_consumed_wait(
                    pairwise_consumed_mbar.subview(pairwise_stage),
                    ((chunk // PAIRWISE_STAGE_COUNT) + 1) % 2,
                )
                cg0_k_ready_wait(
                    cg0_k_half_ready_mbar.subview(decay_stage),
                    (chunk // DECAY_STAGE_COUNT) % 2,
                )
                # KDA schedule owner: standalone auxiliary-MMA warp.  The first
                # KK half runs on the half-DK arrival; the pipeline waits the
                # full cg0_k_ready on its own before the second half.
                super_mma_stage_pairwise_pipeline(
                    tcgen05_k_decay_stage,
                    k_inv_stage,
                    pairwise_stage_smem,
                    raw_beta_stage,
                    cg0_k_ready_mbar.subview(decay_stage),
                    (chunk // DECAY_STAGE_COUNT) % 2,
                    lane,
                    input_dtype,
                )
                pairwise_ready_arrive(pairwise_ready_mbar.subview(pairwise_stage))
                operand_smem_consumed_arrive(
                    operand_smem_consumed_mbar.subview(decay_stage)
                )
                raw_consumed_arrive(raw_consumed_mbar.subview(raw_stage))
                raw_stage, _ = advance_ring_stage(raw_stage, 1, RAW_STAGE_COUNT)

        elif warp_idx == ROLES.tcgen05_mma:
            tmem_raw_addr = tmem_ptr_i32.load()
            shared_acc_event_id = cutlass.Int32(0)
            late_operand_stage = cutlass.Int32(0)
            q_k_restore_ready_phase = cutlass.Int32(0)
            for chunk in cutlass.range(num_chunks, unroll=1):
                o_stage = chunk % O_STAGE_COUNT
                qstate_acc_stage = chunk % KDA_TMEM_QSTATE_ACC_STAGE_COUNT
                decay_stage = chunk % DECAY_STAGE_COUNT
                q_k_restore_ready_stage = late_operand_stage
                pairwise_stage = chunk % PAIRWISE_STAGE_COUNT
                shared_input_stage = o_stage
                tcgen05_k_decay_stage = tcgen05_k_decay_smem.subview(
                    decay_stage * TCGEN05_K_DECAY_STAGE_SIZE
                )
                tcgen05_q_decay_stage = tcgen05_q_decay_smem.subview(
                    decay_stage * TCGEN05_Q_DECAY_STAGE_SIZE
                )
                tcgen05_k_restore_stage = tcgen05_k_restore_smem.subview(
                    decay_stage * TCGEN05_K_RESTORE_STAGE_SIZE
                )
                pairwise_stage_smem = pairwise_smem.subview(
                    pairwise_stage * PAIRWISE_SMEM_STAGE_SIZE
                )

                cg0_k_ready_wait(
                    cg0_k_ready_mbar.subview(decay_stage),
                    (chunk // DECAY_STAGE_COUNT) % 2,
                )

                state_input_ready_l_phase = state_input_ready_wait(
                    state_input_ready_l_mbar,
                    state_input_ready_l_phase,
                )
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                state_k_acc_stage = tcgen05_shared_acc_stage_from_event(
                    shared_acc_event_id
                )
                tcgen05_issue_state_k_mma(
                    tcgen05_k_decay_stage,
                    tmem_raw_addr,
                    shared_acc_ready_mbar.subview(state_k_acc_stage),
                    state_k_acc_stage,
                    input_dtype,
                    0,
                    (DK // TCGEN05_F16_K_ATOM) // 2,
                    False,
                    False,
                    DV,
                )
                state_input_ready_phase = state_input_ready_wait(
                    state_input_ready_mbar,
                    state_input_ready_phase,
                )
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                tcgen05_issue_state_k_mma(
                    tcgen05_k_decay_stage,
                    tmem_raw_addr,
                    shared_acc_ready_mbar.subview(state_k_acc_stage),
                    state_k_acc_stage,
                    input_dtype,
                    (DK // TCGEN05_F16_K_ATOM) // 2,
                    DK // TCGEN05_F16_K_ATOM,
                    True,
                    True,
                    DV,
                )
                shared_acc_event_id += cutlass.Int32(1)

                q_k_restore_ready_wait(
                    q_k_restore_ready_mbar.subview(q_k_restore_ready_stage),
                    q_k_restore_ready_phase,
                )

                qstate_acc_reuse_phase = (
                    chunk // KDA_TMEM_QSTATE_ACC_STAGE_COUNT + cutlass.Int32(1)
                ) % cutlass.Int32(2)
                output_ready_wait(
                    output_ready_mbar.subview(qstate_acc_stage),
                    qstate_acc_reuse_phase,
                )

                # State*Q tile producer.  This uses the dedicated qstate_acc slot
                # because the tile stays live until qkv is fused into output.
                tcgen05_issue_state_q_mma(
                    tcgen05_q_decay_stage,
                    tmem_raw_addr,
                    operand_smem_consumed_mbar.subview(decay_stage),
                    qstate_acc_stage,
                    input_dtype,
                )
                pairwise_ready_wait(
                    pairwise_ready_mbar.subview(pairwise_stage),
                    (chunk // PAIRWISE_STAGE_COUNT) % 2,
                )
                rhs_ready_phase = rhs_ready_wait(
                    rhs_ready_mbar,
                    rhs_ready_phase,
                )
                # KDA schedule owner: tcgen05-MMA warp.
                #
                # Update tile producer:
                #   update = A_inv @ rhs
                update_acc_stage = tcgen05_shared_acc_stage_from_event(
                    shared_acc_event_id
                )
                tcgen05_issue_update_mma(
                    pairwise_stage_smem,
                    tmem_raw_addr,
                    shared_acc_ready_mbar.subview(update_acc_stage),
                    update_acc_stage,
                    shared_input_stage,
                    input_dtype,
                )
                shared_acc_event_id += cutlass.Int32(1)

                update_ready_phase = update_ready_wait(
                    update_ready_mbar,
                    update_ready_phase,
                )
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                # Final-state producer:
                #   final_state_acc += update @ k_restore
                tcgen05_issue_final_state_delta_mma(
                    tcgen05_k_restore_stage,
                    tmem_raw_addr,
                    k_restore_consumed_l_mbar.subview(decay_stage),
                    k_restore_consumed_mbar.subview(decay_stage),
                    shared_input_stage,
                    input_dtype,
                )

                pairwise_ready_wait(
                    qk_ready_mbar.subview(pairwise_stage),
                    (chunk // PAIRWISE_STAGE_COUNT) % 2,
                )
                # Output tile producer:
                #   staged_o = (state_q + qk @ update) * scale
                tcgen05_issue_qkv_mma(
                    pairwise_stage_smem,
                    tmem_raw_addr,
                    qstate_acc_ready_mbar,
                    shared_input_stage,
                    qstate_acc_stage,
                    input_dtype,
                )
                pairwise_consumed_arrive(pairwise_consumed_mbar.subview(pairwise_stage))
                late_operand_stage, late_operand_wrapped = advance_ring_stage(
                    late_operand_stage,
                    1,
                    Q_K_RESTORE_READY_STAGE_COUNT,
                )
                q_k_restore_ready_phase = q_k_restore_ready_phase ^ late_operand_wrapped

            final_state_stored_phase = final_state_stored_wait(
                final_state_stored_mbar,
                final_state_stored_phase,
            )
            tmem_ptr = cutlass.inttoptr(tmem_raw_addr, 6, cutlass.Float32)
            prims.tcgen05_dealloc(tmem_ptr, KDA_TMEM_ALLOC_COLS, group="cta_1")

        elif warp_idx == ROLES.epilogue:
            q_k_restore_ready_stage = cutlass.Int32(0)
            q_k_restore_ready_phase = cutlass.Int32(0)
            for chunk in cutlass.range(num_chunks, unroll=1):
                decay_stage = chunk % DECAY_STAGE_COUNT
                pairwise_stage = chunk % PAIRWISE_STAGE_COUNT
                k_inv_stage = k_inv_smem.subview(decay_stage * K_INV_STAGE_SIZE)
                tcgen05_q_decay_stage = tcgen05_q_decay_smem.subview(
                    decay_stage * TCGEN05_Q_DECAY_STAGE_SIZE
                )
                pairwise_stage_smem = pairwise_smem.subview(
                    pairwise_stage * PAIRWISE_SMEM_STAGE_SIZE
                )

                pairwise_consumed_wait(
                    pairwise_consumed_mbar.subview(pairwise_stage),
                    ((chunk // PAIRWISE_STAGE_COUNT) + 1) % 2,
                )
                q_k_restore_ready_wait(
                    q_k_restore_ready_mbar.subview(q_k_restore_ready_stage),
                    q_k_restore_ready_phase,
                )
                super_mma_stage_qk(
                    tcgen05_q_decay_stage,
                    k_inv_stage,
                    pairwise_stage_smem,
                    lane,
                    input_dtype,
                )
                pairwise_ready_arrive(qk_ready_mbar.subview(pairwise_stage))
                operand_smem_consumed_arrive(
                    operand_smem_consumed_mbar.subview(decay_stage)
                )
                q_k_restore_ready_stage, q_k_restore_wrapped = advance_ring_stage(
                    q_k_restore_ready_stage,
                    1,
                    Q_K_RESTORE_READY_STAGE_COUNT,
                )
                q_k_restore_ready_phase = q_k_restore_ready_phase ^ q_k_restore_wrapped

                if chunk > 0:
                    output_chunk = chunk - cutlass.Int32(1)
                    epilogue_wait_and_store_full_output(
                        tma_desc_o,
                        o_smem,
                        output_ready_mbar,
                        output_consumed_mbar,
                        sequence_start,
                        bidy,
                        output_chunk,
                        O_STAGE_COUNT,
                    )
            if num_chunks > 0:
                output_chunk = num_chunks - cutlass.Int32(1)
                epilogue_wait_and_store_final_output(
                    tma_desc_o,
                    out,
                    o_smem,
                    output_ready_mbar,
                    output_consumed_mbar,
                    sequence_start,
                    bidy,
                    seqlen,
                    output_chunk,
                    lane,
                    O_STAGE_COUNT,
                )
    elif is_compute_group0_warp(warp_idx):
        prims.setmaxregister(KDA_CG0_REGS, prims.SetMaxRegisterAction.INCREASE)
        cg0_warp = warp_idx - ROLES.compute_group0_first
        cg0_group_id = cg0_warp // CG0_WARPS_PER_GROUP
        cg0_local_warp = cg0_warp % CG0_WARPS_PER_GROUP
        cg0_a_log_exp = cutlass.Float32(1.0)
        cg0_dt_bias_value = cutlass.Float32(0.0)
        tmem_raw_addr = cutlass.Int32(0)
        if num_chunks > 0:
            raw_ready_wait(tma_mbar.subview(0), 0)
            prefix_dim = cg0_local_warp * THREADS_PER_WARP + lane
            cg0_a_log_exp = raw_dt_bias_smem[RAW_DT_BIAS_A_LOG_EXP_OFFSET]
            cg0_dt_bias_value = raw_dt_bias_smem[prefix_dim]
            state_input_ready_wait(initial_state_ready_mbar, cutlass.Int32(0))
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            # The allocator publishes one immutable TMEM base for the CTA.
            # Cache it after the initial-state handoff instead of reloading
            # the SMEM pointer in every steady and peeled CG0 iteration.
            tmem_raw_addr = tmem_ptr_i32.load()
        raw_stage = cg0_group_id
        raw_ready_phase = cutlass.Int32(0)
        late_operand_stage = cg0_group_id
        cg0_ckpt_stride = cutlass.Int32(0)
        if cutlass.const_expr(state_ckpt is not None):
            cg0_ckpt_stride = checkpoint_stride_chunks
        # v27_peel: STRUCTURAL in-kernel tail-peel of the CG0 producer loop.
        # The interior mainloop runs the GUARD-FREE body over chunks
        # [cg0_group_id, num_chunks-1): NO per-chunk `chunk_start+BT>seqlen`
        # compare and NO tail zeroing (materialize called with the FULL branch).
        # The single genuinely-partial tail chunk (num_chunks-1) is peeled out
        # once below and run with the masked tail fixup.  This makes ONE kernel
        # correct for both aligned and ragged inputs (the peeled tail is a
        # runtime no-op when the sequence is BT-aligned).  Ring-state locals
        # (raw_stage / phase, late_operand_stage / phase) are loop-carried and
        # left positioned for the peeled chunk by the mainloop.
        for chunk in cutlass.range(
            cg0_group_id,
            num_chunks - cutlass.Int32(1),
            CG0_GROUP_COUNT,
            unroll=1,
        ):
            decay_stage = chunk % DECAY_STAGE_COUNT
            q_k_restore_ready_stage = late_operand_stage
            raw_q_stage = raw_q_smem.subview(raw_stage * RAW_Q_STAGE_SIZE)
            raw_k_stage = raw_k_smem.subview(raw_stage * RAW_K_STAGE_SIZE)
            raw_gate_stage = raw_gate_smem.subview(raw_stage * RAW_GATE_STAGE_SIZE)
            # FP32 exp2(g_prefix) exchange tile.  With an FP32 gate it IS the
            # raw-gate stage (historical aliasing, zero cost); with a 16-bit gate
            # it comes from the dedicated 4-deep gate_exchange ring.
            if cutlass.const_expr(gate_dtype_is_f32(gate_dtype)):
                gate_exchange_stage = raw_gate_stage
            else:
                gate_exchange_stage = gate_exchange_smem.subview(
                    (chunk % GATE_EXCHANGE_STAGE_COUNT) * GATE_EXCHANGE_STAGE_SIZE
                )
            k_inv_stage = k_inv_smem.subview(decay_stage * K_INV_STAGE_SIZE)
            tcgen05_k_decay_stage = tcgen05_k_decay_smem.subview(
                decay_stage * TCGEN05_K_DECAY_STAGE_SIZE
            )
            tcgen05_q_decay_stage = tcgen05_q_decay_smem.subview(
                decay_stage * TCGEN05_Q_DECAY_STAGE_SIZE
            )
            tcgen05_k_restore_stage = tcgen05_k_restore_smem.subview(
                decay_stage * TCGEN05_K_RESTORE_STAGE_SIZE
            )
            raw_ready_wait(
                tma_mbar.subview(raw_stage),
                raw_ready_phase,
            )

            # KDA schedule owner: compute warp group 0.
            #
            # The decay materialization stages full gate prefixes once, then
            # issues the two gate MUFU families for the full KDA operand set:
            #   exp2(g_prefix)      : 16 * 128
            #   exp2(-g_prefix)     : 16 * 128
            cg0_materialize_decay_operands(
                raw_q_stage,
                raw_k_stage,
                raw_gate_stage,
                gate_exchange_stage,
                cg0_a_log_exp,
                cg0_dt_bias_value,
                k_inv_stage,
                tcgen05_k_decay_stage,
                tcgen05_q_decay_stage,
                tcgen05_k_restore_stage,
                cg0_k_ready_mbar.subview(decay_stage),
                cg0_k_half_ready_mbar.subview(decay_stage),
                diag_ready_mbar.subview(raw_stage),
                operand_smem_consumed_mbar.subview(decay_stage),
                k_restore_consumed_mbar.subview(decay_stage),
                chunk,
                seqlen,
                input_dtype,
                gate_dtype,
                SAFE_GATE,
                GATE_SCALE_LOG2,
                True,
                cg0_group_id,
                cg0_local_warp,
                lane,
            )
            q_k_restore_ready_arrive(
                q_k_restore_ready_mbar.subview(q_k_restore_ready_stage)
            )
            if chunk > 0:
                prev_state_chunk = chunk - cutlass.Int32(1)
                tcgen05_wait_acc_buffer_ready(
                    k_restore_consumed_l_mbar.subview(
                        prev_state_chunk % DECAY_STAGE_COUNT
                    ),
                    (prev_state_chunk // DECAY_STAGE_COUNT) % 2,
                )
            if cutlass.const_expr(state_ckpt is not None):
                if (chunk > cutlass.Int32(0)) & (chunk % cg0_ckpt_stride == 0):
                    checkpoint_read_done_wait(
                        checkpoint_read_done_mbar,
                        ((chunk // cg0_ckpt_stride) + cutlass.Int32(1)) % 2,
                    )
                    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            tcgen05_pack_rescale_state_half_tmem(
                tmem_raw_addr,
                gate_exchange_stage,
                state_input_ready_l_mbar,
                warp_idx,
                input_dtype,
                0,
                True,
            )
            raw_consumed_arrive(raw_consumed_mbar.subview(raw_stage))
            update_ready_arrive(update_ready_mbar)
            raw_stage, raw_wrapped = advance_ring_stage(
                raw_stage,
                CG0_GROUP_COUNT,
                RAW_STAGE_COUNT,
            )
            raw_ready_phase = raw_ready_phase ^ raw_wrapped
            late_operand_stage, late_operand_wrapped = advance_ring_stage(
                late_operand_stage,
                CG0_GROUP_COUNT,
                Q_K_RESTORE_READY_STAGE_COUNT,
            )

        # Peeled final (tail) chunk num_chunks-1, executed once by the CG0 group
        # that owns it.  num_chunks==1 -> the single chunk IS the tail (group 0);
        # num_chunks==0 -> no work.  materialize runs the tail branch (its
        # built-in `tail_valid<BT` runtime test is a no-op when BT-aligned, so
        # aligned output is bit-identical to the guard-free FULL body).
        if num_chunks > cutlass.Int32(0):
            if (num_chunks - cutlass.Int32(1)) % CG0_GROUP_COUNT == cg0_group_id:
                chunk = num_chunks - cutlass.Int32(1)
                chunk_start = chunk * BT
                decay_stage = chunk % DECAY_STAGE_COUNT
                q_k_restore_ready_stage = late_operand_stage
                raw_q_stage = raw_q_smem.subview(raw_stage * RAW_Q_STAGE_SIZE)
                raw_k_stage = raw_k_smem.subview(raw_stage * RAW_K_STAGE_SIZE)
                raw_v_stage = raw_v_smem.subview(raw_stage * RAW_V_STAGE_SIZE)
                raw_gate_stage = raw_gate_smem.subview(raw_stage * RAW_GATE_STAGE_SIZE)
                # FP32 exp2(g_prefix) exchange tile.  With an FP32 gate it IS the
                # raw-gate stage (historical aliasing, zero cost); with a 16-bit gate
                # it comes from the dedicated 4-deep gate_exchange ring.
                if cutlass.const_expr(gate_dtype_is_f32(gate_dtype)):
                    gate_exchange_stage = raw_gate_stage
                else:
                    gate_exchange_stage = gate_exchange_smem.subview(
                        (chunk % GATE_EXCHANGE_STAGE_COUNT) * GATE_EXCHANGE_STAGE_SIZE
                    )
                k_inv_stage = k_inv_smem.subview(decay_stage * K_INV_STAGE_SIZE)
                tcgen05_k_decay_stage = tcgen05_k_decay_smem.subview(
                    decay_stage * TCGEN05_K_DECAY_STAGE_SIZE
                )
                tcgen05_q_decay_stage = tcgen05_q_decay_smem.subview(
                    decay_stage * TCGEN05_Q_DECAY_STAGE_SIZE
                )
                tcgen05_k_restore_stage = tcgen05_k_restore_smem.subview(
                    decay_stage * TCGEN05_K_RESTORE_STAGE_SIZE
                )
                raw_ready_wait(
                    tma_mbar.subview(raw_stage),
                    raw_ready_phase,
                )

                if chunk_start + cutlass.Int32(BT) > seqlen:
                    if cg0_local_warp == 0:
                        cg0_zero_tail_raw_operands(
                            raw_q_stage,
                            raw_k_stage,
                            raw_v_stage,
                            raw_gate_stage,
                            lane,
                            chunk_start,
                            seqlen,
                            q.element_type,
                            gate_dtype,
                        )
                    cg0_sync(cg0_group_id)

                cg0_materialize_decay_operands(
                    raw_q_stage,
                    raw_k_stage,
                    raw_gate_stage,
                    gate_exchange_stage,
                    cg0_a_log_exp,
                    cg0_dt_bias_value,
                    k_inv_stage,
                    tcgen05_k_decay_stage,
                    tcgen05_q_decay_stage,
                    tcgen05_k_restore_stage,
                    cg0_k_ready_mbar.subview(decay_stage),
                    cg0_k_half_ready_mbar.subview(decay_stage),
                    diag_ready_mbar.subview(raw_stage),
                    operand_smem_consumed_mbar.subview(decay_stage),
                    k_restore_consumed_mbar.subview(decay_stage),
                    chunk,
                    seqlen,
                    input_dtype,
                    gate_dtype,
                    SAFE_GATE,
                    GATE_SCALE_LOG2,
                    False,
                    cg0_group_id,
                    cg0_local_warp,
                    lane,
                )
                q_k_restore_ready_arrive(
                    q_k_restore_ready_mbar.subview(q_k_restore_ready_stage)
                )
                if chunk > 0:
                    prev_state_chunk = chunk - cutlass.Int32(1)
                    tcgen05_wait_acc_buffer_ready(
                        k_restore_consumed_l_mbar.subview(
                            prev_state_chunk % DECAY_STAGE_COUNT
                        ),
                        (prev_state_chunk // DECAY_STAGE_COUNT) % 2,
                    )
                if cutlass.const_expr(state_ckpt is not None):
                    if (chunk > cutlass.Int32(0)) & (chunk % cg0_ckpt_stride == 0):
                        checkpoint_read_done_wait(
                            checkpoint_read_done_mbar,
                            ((chunk // cg0_ckpt_stride) + cutlass.Int32(1)) % 2,
                        )
                        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                tcgen05_pack_rescale_state_half_tmem(
                    tmem_raw_addr,
                    gate_exchange_stage,
                    state_input_ready_l_mbar,
                    warp_idx,
                    input_dtype,
                    0,
                    True,
                )
                raw_consumed_arrive(raw_consumed_mbar.subview(raw_stage))
                update_ready_arrive(update_ready_mbar)

    elif is_compute_group1_warp(warp_idx):
        prims.setmaxregister(KDA_CG1_REGS, prims.SetMaxRegisterAction.INCREASE)
        tmem_raw_addr = tmem_ptr_i32.load()
        # Only CG1 touches recurrent state. Keep the pool lookup outside the
        # chunk loop and out of producer/MMAs warps.
        state_slot = bidx
        if cutlass.const_expr(state_indices is not None):
            state_slot = cutlass.Int32(state_indices[bidx])
        ckpt_slot = cutlass.Int32(0)
        if cutlass.const_expr(state_ckpt is not None):
            ckpt_stride = checkpoint_stride_chunks
            ckpt_next = ckpt_stride
            ckpt_slot = cutlass.Int32(cu_ckpts[bidx])
        if cutlass.const_expr(checkpoint_state_indices is not None):
            tcgen05_store_initial_state_tmem(
                tmem_raw_addr,
                initial_state,
                None,
                cutlass.Int32(0),
                state_slot,
                bidy,
                cutlass.Int32(0),
                warp_idx,
                lane,
                HALF=False,
            )
        else:
            tcgen05_store_initial_state_tmem(
                tmem_raw_addr,
                initial_state,
                state_ckpt,
                ckpt_slot,
                state_slot,
                bidy,
                cutlass.Int32(0),
                warp_idx,
                lane,
                HALF=False,
            )
        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
        state_input_ready_arrive(initial_state_ready_mbar)
        shared_acc_event_id = cutlass.Int32(0)
        if cutlass.const_expr(
            state_ckpt is not None and checkpoint_state_indices is None
        ):
            # cu_ckpts supplies the per-sequence base slot offsets.  A
            # loop-carried counter replaces a per-chunk div/mod.
            ckpt_slot += cutlass.Int32(1)

        if num_chunks > 0:
            shared_input_stage = cutlass.Int32(0)
            raw_v_stage = raw_v_smem.subview(0)
            raw_beta_stage = raw_beta_smem.subview(0)
            raw_gate_stage = raw_gate_smem.subview(0)
            # FP32 exp2(g_prefix) exchange tile for the CG1 prologue (chunk 0).
            # With an FP32 gate it IS the raw-gate stage (historical aliasing,
            # zero cost); with a 16-bit gate it is stage 0 of the dedicated
            # 4-deep gate_exchange ring (chunk 0 -> stage 0).
            if cutlass.const_expr(gate_dtype_is_f32(gate_dtype)):
                gate_exchange_stage = raw_gate_stage
            else:
                gate_exchange_stage = gate_exchange_smem.subview(0)

            state0, state1, state2, state3 = tcgen05_stage_state_input_tmem(
                tmem_raw_addr,
                warp_idx,
                output_consumed_mbar.subview(0),
                cutlass.Int32(0),
                input_dtype,
                False,
                1,
                True,
            )
            # Decay the retained right half as soon as CG0 publishes the FP32
            # diagonal; K/Q materialization continues independently.
            diag_ready_wait(diag_ready_mbar.subview(0), 0)
            tcgen05_publish_projection_then_rescale_state_regs(
                tmem_raw_addr,
                gate_exchange_stage,
                state_input_ready_mbar,
                warp_idx,
                state0,
                state1,
                state2,
                state3,
                1,
                True,
            )

            state_k_acc_stage = tcgen05_shared_acc_stage_from_event(shared_acc_event_id)
            state_k_acc_phase = tcgen05_shared_acc_phase_from_event(shared_acc_event_id)
            tcgen05_wait_acc_buffer_ready(
                shared_acc_ready_mbar.subview(state_k_acc_stage),
                state_k_acc_phase,
            )
            rhs_lane = cute.arch.lane_idx()
            tcgen05_stage_rhs_input_tmem(
                tmem_raw_addr,
                raw_v_stage,
                raw_beta_stage,
                warp_idx,
                rhs_lane,
                state_k_acc_stage,
                shared_input_stage,
                input_dtype,
            )
            shared_acc_event_id += cutlass.Int32(1)
            rhs_ready_arrive(rhs_ready_mbar)

            raw_consumed_arrive(raw_consumed_mbar.subview(0))
            post_scale_lane = cute.arch.lane_idx()

            update_acc_stage = tcgen05_shared_acc_stage_from_event(shared_acc_event_id)
            update_acc_phase = tcgen05_shared_acc_phase_from_event(shared_acc_event_id)
            tcgen05_wait_acc_buffer_ready(
                shared_acc_ready_mbar.subview(update_acc_stage),
                update_acc_phase,
            )
            tcgen05_stage_update_input_tmem(
                tmem_raw_addr,
                warp_idx,
                update_acc_stage,
                shared_input_stage,
                input_dtype,
            )
            shared_acc_event_id += cutlass.Int32(1)
            update_ready_arrive(update_ready_mbar)

            if cutlass.const_expr(state_ckpt is not None):
                # Peeled chunk 0's checkpoint (fires only for stride 1, i.e. a
                # checkpoint every BT tokens).
                checkpoint_due = ckpt_next == cutlass.Int32(1)
                if cutlass.const_expr(checkpoint_state_indices is not None):
                    checkpoint_due = checkpoint_due & (cutlass.Int32(BT) <= seqlen)
                else:
                    checkpoint_due = checkpoint_due & (cutlass.Int32(1) < num_chunks)
                if checkpoint_due:
                    tcgen05_wait_acc_buffer_ready(k_restore_consumed_mbar.subview(0), 0)
                    checkpoint_output_slot = ckpt_slot
                    if cutlass.const_expr(checkpoint_state_indices is not None):
                        checkpoint_output_slot = cutlass.Int32(
                            checkpoint_state_indices[ckpt_slot]
                        )
                    if (checkpoint_output_slot >= cutlass.Int32(0)) & (
                        checkpoint_output_slot < cutlass.Int32(state_ckpt.shape[0])
                    ):
                        tcgen05_store_final_state_tmem(
                            tmem_raw_addr,
                            KDA_TMEM_FINAL_STATE_ACC_COL_OFFSET,
                            state_ckpt,
                            checkpoint_output_slot,
                            bidy,
                            cutlass.Int32(0),
                            warp_idx,
                            post_scale_lane,
                            HALF=False,
                        )
                        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                    checkpoint_read_done_arrive(checkpoint_read_done_mbar)
                    ckpt_slot += cutlass.Int32(1)
                    ckpt_next += ckpt_stride

        # Peel chunk 0 so the steady-state loop always drains a prior output.
        # Each actual shared-acc commit advances the event ring independently
        # of which GEMM produced it.
        raw_stage = cutlass.Int32(1)
        for chunk in cutlass.range(1, num_chunks, 1, unroll=1):
            o_stage = chunk % O_STAGE_COUNT
            decay_stage = chunk % DECAY_STAGE_COUNT
            shared_input_stage = o_stage
            raw_v_stage = raw_v_smem.subview(raw_stage * RAW_V_STAGE_SIZE)
            raw_beta_stage = raw_beta_smem.subview(raw_stage * RAW_BETA_STAGE_SIZE)
            raw_gate_stage = raw_gate_smem.subview(raw_stage * RAW_GATE_STAGE_SIZE)
            # FP32 exp2(g_prefix) exchange tile.  With an FP32 gate it IS the
            # raw-gate stage (historical aliasing, zero cost); with a 16-bit gate
            # it comes from the dedicated 4-deep gate_exchange ring.
            if cutlass.const_expr(gate_dtype_is_f32(gate_dtype)):
                gate_exchange_stage = raw_gate_stage
            else:
                gate_exchange_stage = gate_exchange_smem.subview(
                    (chunk % GATE_EXCHANGE_STAGE_COUNT) * GATE_EXCHANGE_STAGE_SIZE
                )

            prev_output_chunk = chunk - cutlass.Int32(1)
            prev_o_stage = prev_output_chunk % O_STAGE_COUNT
            prev_qstate_acc_stage = prev_output_chunk % KDA_TMEM_QSTATE_ACC_STAGE_COUNT
            prev_o_stage_base = prev_o_stage * O_SMEM_STAGE_SIZE
            prev_decay_stage = prev_output_chunk % DECAY_STAGE_COUNT
            prev_k_restore_phase = (prev_output_chunk // DECAY_STAGE_COUNT) % 2
            tcgen05_wait_acc_buffer_ready(
                k_restore_consumed_mbar.subview(prev_decay_stage),
                prev_k_restore_phase,
            )
            state0, state1, state2, state3 = tcgen05_stage_state_input_tmem(
                tmem_raw_addr,
                warp_idx,
                output_consumed_mbar.subview(prev_o_stage),
                ((prev_output_chunk // O_STAGE_COUNT) + 1) % 2,
                input_dtype,
                True,
                1,
                True,
            )
            # The diagonal is complete before k_decay/q_decay/k_restore; use
            # that narrower dependency to hide right-half FP32 scaling.
            diag_ready_wait(
                diag_ready_mbar.subview(raw_stage),
                (chunk // RAW_STAGE_COUNT) % 2,
            )
            tcgen05_publish_projection_then_rescale_state_regs(
                tmem_raw_addr,
                gate_exchange_stage,
                state_input_ready_mbar,
                warp_idx,
                state0,
                state1,
                state2,
                state3,
                1,
                True,
            )
            pre_scale_lane = cute.arch.lane_idx()
            qstate_acc_ready_phase = tcgen05_wait_acc_buffer_ready(
                qstate_acc_ready_mbar,
                qstate_acc_ready_phase,
            )
            tcgen05_load_qstate_output_tmem(
                tmem_raw_addr,
                o_smem,
                warp_idx,
                pre_scale_lane,
                prev_o_stage_base,
                prev_qstate_acc_stage,
                SCALE,
                out.element_type,
            )
            output_ready_arrive(output_ready_mbar.subview(prev_o_stage))

            state_k_acc_stage = tcgen05_shared_acc_stage_from_event(shared_acc_event_id)
            state_k_acc_phase = tcgen05_shared_acc_phase_from_event(shared_acc_event_id)
            tcgen05_wait_acc_buffer_ready(
                shared_acc_ready_mbar.subview(state_k_acc_stage),
                state_k_acc_phase,
            )
            # KDA schedule owner: compute warp group 1.
            #
            # Epilogue for tcgen05 state*k, staged directly as the next
            # tcgen05 A operand:
            #   shared_input = beta_i * (v_i - state*k_i)
            tcgen05_stage_rhs_input_tmem(
                tmem_raw_addr,
                raw_v_stage,
                raw_beta_stage,
                warp_idx,
                pre_scale_lane,
                state_k_acc_stage,
                shared_input_stage,
                input_dtype,
            )
            shared_acc_event_id += cutlass.Int32(1)
            rhs_ready_arrive(rhs_ready_mbar)

            raw_consumed_arrive(raw_consumed_mbar.subview(raw_stage))
            post_scale_lane = cute.arch.lane_idx()

            update_acc_stage = tcgen05_shared_acc_stage_from_event(shared_acc_event_id)
            update_acc_phase = tcgen05_shared_acc_phase_from_event(shared_acc_event_id)
            tcgen05_wait_acc_buffer_ready(
                shared_acc_ready_mbar.subview(update_acc_stage),
                update_acc_phase,
            )
            tcgen05_stage_update_input_tmem(
                tmem_raw_addr,
                warp_idx,
                update_acc_stage,
                shared_input_stage,
                input_dtype,
            )
            shared_acc_event_id += cutlass.Int32(1)
            update_ready_arrive(update_ready_mbar)

            if cutlass.const_expr(state_ckpt is not None):
                # State checkpoint: at every ckpt_stride-th chunk boundary the
                # TMEM accumulator holds exactly the state a run truncated here
                # would emit as final_state.  Wait the SAME parity slot the
                # post-loop final store waits (the chunk's state-update MMA has
                # committed), then reuse the final-state store routine with the
                # flat checkpoint slot standing in for the sequence index.
                checkpoint_due = chunk + cutlass.Int32(1) == ckpt_next
                if cutlass.const_expr(checkpoint_state_indices is not None):
                    checkpoint_due = checkpoint_due & (
                        (chunk + cutlass.Int32(1)) * cutlass.Int32(BT) <= seqlen
                    )
                else:
                    checkpoint_due = checkpoint_due & (
                        chunk + cutlass.Int32(1) < num_chunks
                    )
                if checkpoint_due:
                    tcgen05_wait_acc_buffer_ready(
                        k_restore_consumed_mbar.subview(chunk % DECAY_STAGE_COUNT),
                        (chunk // DECAY_STAGE_COUNT) % 2,
                    )
                    checkpoint_output_slot = ckpt_slot
                    if cutlass.const_expr(checkpoint_state_indices is not None):
                        checkpoint_output_slot = cutlass.Int32(
                            checkpoint_state_indices[ckpt_slot]
                        )
                    if (checkpoint_output_slot >= cutlass.Int32(0)) & (
                        checkpoint_output_slot < cutlass.Int32(state_ckpt.shape[0])
                    ):
                        tcgen05_store_final_state_tmem(
                            tmem_raw_addr,
                            KDA_TMEM_FINAL_STATE_ACC_COL_OFFSET,
                            state_ckpt,
                            checkpoint_output_slot,
                            bidy,
                            cutlass.Int32(0),
                            warp_idx,
                            post_scale_lane,
                            HALF=False,
                        )
                        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                    checkpoint_read_done_arrive(checkpoint_read_done_mbar)
                    ckpt_slot += cutlass.Int32(1)
                    ckpt_next += ckpt_stride

            raw_stage, _ = advance_ring_stage(raw_stage, 1, RAW_STAGE_COUNT)

        if num_chunks > 0:
            final_lane = cute.arch.lane_idx()
            output_chunk = num_chunks - cutlass.Int32(1)
            last_decay_stage = output_chunk % DECAY_STAGE_COUNT
            tcgen05_wait_acc_buffer_ready(
                k_restore_consumed_mbar.subview(last_decay_stage),
                (output_chunk // DECAY_STAGE_COUNT) % 2,
            )
            final_o_stage = output_chunk % O_STAGE_COUNT
            final_qstate_acc_stage = output_chunk % KDA_TMEM_QSTATE_ACC_STAGE_COUNT
            final_o_stage_base = final_o_stage * O_SMEM_STAGE_SIZE
            output_consumed_wait(
                output_consumed_mbar.subview(final_o_stage),
                ((output_chunk // O_STAGE_COUNT) + 1) % 2,
            )

            qstate_acc_ready_phase = tcgen05_wait_acc_buffer_ready(
                qstate_acc_ready_mbar,
                qstate_acc_ready_phase,
            )
            tcgen05_load_qstate_output_tmem(
                tmem_raw_addr,
                o_smem,
                warp_idx,
                final_lane,
                final_o_stage_base,
                final_qstate_acc_stage,
                SCALE,
                out.element_type,
            )
            output_ready_arrive(output_ready_mbar.subview(final_o_stage))

        if cutlass.const_expr(final_state is not None):
            final_lane = cute.arch.lane_idx()
            tcgen05_store_final_state_tmem(
                tmem_raw_addr,
                KDA_TMEM_FINAL_STATE_ACC_COL_OFFSET,
                final_state,
                state_slot,
                bidy,
                cutlass.Int32(0),
                warp_idx,
                final_lane,
                HALF=False,
            )
        final_state_stored_arrive(final_state_stored_mbar)


@cute.jit
def host(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    raw_gate: cute.Tensor,
    a_log: cute.Tensor,
    dt_bias: cute.Tensor,
    beta: cute.Tensor,
    cu_seqlens: cute.Tensor,
    seq_order: cute.Tensor,
    state_indices: cute.Tensor | None,
    initial_state: cute.Tensor | None,
    out: cute.Tensor,
    final_state: cute.Tensor | None,
    stream,
    SCALE: cutlass.Float32,
    state_ckpt: cute.Tensor | None,
    cu_ckpts: cute.Tensor | None,
    checkpoint_state_indices: cute.Tensor | None,
    checkpoint_stride_chunks: cutlass.Int32,
    SAFE_GATE: cutlass.Constexpr,
    GATE_SCALE_LOG2: cutlass.Constexpr,
    THREADS: cutlass.Constexpr,
    gate_dtype: cutlass.Constexpr,
) -> None:
    packed_batch = q.shape[0]
    num_sequences = cu_seqlens.shape[0] - 1
    seqlen = q.shape[1]
    heads = q.shape[2]
    # Token-major activations: memory order [T, H, D] (token stride = D*heads),
    # so packed [1, T_total, H, D] / batched [B, T, H, D] callers are read
    # directly with no transpose.
    # Keep the unused outer stride representable by CuTe IR for packed inputs.
    # CuTe layouts currently lower strides through signed Int32, so the natural
    # D*T*H batch stride overflows once a singleton packed token buffer exceeds
    # that range even though the batch coordinate is always zero.
    qk_batch_stride = DK
    v_batch_stride = DV
    if packed_batch != cutlass.Int32(1):
        qk_batch_stride = DK * seqlen * heads
        v_batch_stride = DV * seqlen * heads
    qk_layout = cute.make_layout(
        (DK, seqlen, heads, packed_batch),
        stride=(1, DK * heads, DK, qk_batch_stride),
    )
    v_layout = cute.make_layout(
        (DV, seqlen, heads, packed_batch),
        stride=(1, DV * heads, DV, v_batch_stride),
    )
    q_tma = cute.make_tensor(q.iterator, qk_layout)
    k_tma = cute.make_tensor(k.iterator, qk_layout)
    v_tma = cute.make_tensor(v.iterator, v_layout)
    gate_tma = cute.make_tensor(raw_gate.iterator, qk_layout)
    out_tma = cute.make_tensor(out.iterator, v_layout)
    raw_f16_tma_box = (RAW_F16_TMA_SWIZZLE_ELEMS, BT, 1, 1)
    raw_f32_tma_box = (RAW_F32_TMA_SWIZZLE_ELEMS, BT, 1, 1)
    o_tma_box = (O_TMA_SWIZZLE_ELEMS, BT, 1, 1)
    tma_desc_q = cuda.create_tensor_map_tiled_from_view(
        q_tma,
        box_dims=raw_f16_tma_box,
        stride_order=(0, 1, 2, 3),
        swizzle=cuda.TensorMapSwizzle.s128b,
    )
    tma_desc_k = cuda.create_tensor_map_tiled_from_view(
        k_tma,
        box_dims=raw_f16_tma_box,
        stride_order=(0, 1, 2, 3),
        swizzle=cuda.TensorMapSwizzle.s128b,
    )
    tma_desc_v = cuda.create_tensor_map_tiled_from_view(
        v_tma,
        box_dims=raw_f16_tma_box,
        stride_order=(0, 1, 2, 3),
        swizzle=cuda.TensorMapSwizzle.s128b,
    )
    # The gate box/swizzle family follows `gate_dtype`: FP32 keeps the wide
    # 4 x 128 B box, a 16-bit gate uses the same family as q/k/v.
    gate_tma_box = raw_f32_tma_box if gate_dtype_is_f32(gate_dtype) else raw_f16_tma_box
    tma_desc_gate = cuda.create_tensor_map_tiled_from_view(
        gate_tma,
        box_dims=gate_tma_box,
        stride_order=(0, 1, 2, 3),
        swizzle=cuda.TensorMapSwizzle.s128b,
    )
    # beta transport: one descriptor family over the SAME contiguous [T, H]
    # memory, runtime-shaped by g = 8/gcd(heads, 8) (see the constants block).
    # g == 1: view (heads, T) box (8, BT) -- per-head-group 16B rows.
    # g > 1 : packed view (g*heads, ceil(T/g)) box (g*heads, BT/g + 1).
    beta_heads = cutlass.Int32(beta.shape[2])
    beta_lsb = beta_heads & (-beta_heads)
    beta_g = cutlass.Int32(8) // cutlass.min(beta_lsb, cutlass.Int32(8))
    beta_is_pair = cutlass.Int32(1)
    if beta_g == cutlass.Int32(1):
        beta_is_pair = cutlass.Int32(0)
    beta_rows = (cutlass.Int32(beta.shape[1]) + beta_g - cutlass.Int32(1)) // beta_g
    beta_inner = beta_g * beta_heads
    beta_box_inner = (
        cutlass.Int32(8) * (cutlass.Int32(1) - beta_is_pair) + beta_inner * beta_is_pair
    )
    beta_box_outer = (
        cutlass.Int32(BT) * (cutlass.Int32(1) - beta_is_pair)
        + (cutlass.Int32(BT) // beta_g + cutlass.Int32(1)) * beta_is_pair
    )
    beta_tma_view = cute.make_tensor(
        beta.iterator,
        cute.make_layout((beta_inner, beta_rows), stride=(1, beta_inner)),
    )
    tma_desc_beta = cuda.create_tensor_map_tiled_from_view(
        beta_tma_view,
        box_dims=(beta_box_inner, beta_box_outer),
        stride_order=(0, 1),
        swizzle=cuda.TensorMapSwizzle.none,
    )
    tma_desc_o = cuda.create_tensor_map_tiled_from_view(
        out_tma,
        box_dims=o_tma_box,
        stride_order=(0, 1, 2, 3),
        swizzle=cuda.TensorMapSwizzle.s128b,
    )
    kernel(
        tma_desc_q,
        tma_desc_k,
        tma_desc_v,
        tma_desc_gate,
        tma_desc_beta,
        tma_desc_o,
        q,
        k,
        v,
        raw_gate,
        a_log,
        dt_bias,
        beta,
        cu_seqlens,
        seq_order,
        state_indices,
        initial_state,
        out,
        final_state,
        SCALE,
        state_ckpt,
        cu_ckpts,
        checkpoint_state_indices,
        checkpoint_stride_chunks,
        SAFE_GATE,
        GATE_SCALE_LOG2,
        gate_dtype,
    ).launch(
        grid=(heads, num_sequences, 1),
        block=(THREADS, 1, 1),
        stream=stream,
        min_blocks_per_mp=1,
    )


# ---------------------------------------------------------------------------
# Host-side plan/decision cache keys: every cu_seqlens-keyed cache
# keys on the tensor CONTENTS, obtained sync-free through the memo below.
# ---------------------------------------------------------------------------

# data_ptr -> (weakref(tensor), version, contents). See _cu_seqlens_contents.
_CU_CONTENTS_MEMO: dict = {}


def _cu_seqlens_contents(cu_seqlens: torch.Tensor) -> tuple:
    """cu_seqlens contents as a host tuple, sync-free on the steady path.

    Every plan/decision cache keys on the CONTENTS of cu_seqlens: keying on
    buffer identity (data_ptr) is unsafe because torch's caching allocator
    reuses freed blocks, so a recycled same-shape buffer with different
    contents would hit a stale plan (wrong partition boundaries => wrong
    output rows). But reading the contents is a D2H copy that synchronizes
    the current stream — done per call it serializes the host into every
    call and costs 5-14% wall time on the cached routes (measured, session
    29). This memo removes the read when the SAME live tensor object is
    passed again (the steady case): a hit requires the memoized weakref to
    resolve to the argument tensor ITSELF (`wr() is cu_seqlens`), so a
    recycled allocator block can never produce a false hit — its previous
    owner is a different (dead or dying) object; and an unchanged
    `_version` (catches in-place mutation, which shares the version
    counter across views). Tensors without version counters (inference
    mode) memoize with version None and still get the identity protection.
    """

    key = cu_seqlens.data_ptr()
    try:
        ver = cu_seqlens._version
    except Exception:  # inference tensors track no version counter
        ver = None
    ent = _CU_CONTENTS_MEMO.get(key)
    if ent is not None:
        wr, memo_ver, contents = ent
        if wr() is cu_seqlens and memo_ver == ver:
            return contents
    contents = tuple(int(x) for x in cu_seqlens.cpu().tolist())
    _CU_CONTENTS_MEMO[key] = (weakref.ref(cu_seqlens), ver, contents)
    return contents


# ---------------------------------------------------------------------------
# Single host entry: ONE callable that routes the plain engine vs
# the two-kernel decomposition (DV2 chain) by a one-line occupancy rule
# (_occupancy_pick_route).  The engine is compiled eagerly, the decomp route
# lazily on first use.  The decomp route takes a user-allocated opaque
# workspace (query bytes with workspace_size(); engine needs none).
# ---------------------------------------------------------------------------

_SM_COUNT_CACHE: dict = {}


def _device_sm_count(device) -> int:
    key = str(device)
    count = _SM_COUNT_CACHE.get(key)
    if count is None:
        count = torch.cuda.get_device_properties(device).multi_processor_count
        _SM_COUNT_CACHE[key] = count
    return count


# The route is a one-line occupancy rule: run the decomp DV2 split (2 CTAs
# per sequence-head walk) iff the doubled k2 grid still fits one CTA-per-SM
# wave, else the plain engine.  DV2 is the only decomp variant, so decomp
# always dispatches at dv=2.
def _occupancy_pick_route(
    cu_list: tuple[int, ...], heads: int, sm_count: int
) -> tuple[str, int]:
    """Pick ("decomp", 2) or ("engine", 0) by a simple occupancy rule.

    n_seq * heads is the number of sequence-head walks (B*H for uniform
    seqlen; n_seq = len(cu_list) - 1).  DV2 doubles the k2 grid to 2 CTAs
    per walk; take it iff that grid still fits one wave (<= sm_count), else
    the plain engine.
    """

    n_seq = len(cu_list) - 1
    if n_seq * heads * 2 <= sm_count:
        return ("decomp", 2)
    return ("engine", 0)


# =============================================================================
# Two-kernel decomposition of the same math.
#
# Kernel 1 (`kernel_prep`) is chunk-parallel prep: it reuses the engine's CG0
# (gate prefix / L2 norm / decay operand staging) and warp-12 (KK / L / A_inv
# A_inv) code paths, then composes and stores the rank-16 factors.
# Kernel 2 (`kernel_chain_dv2`) is the lean serial recursion.
#
# Factor math:
#   Ainvb = A_inv (.) beta-col                       [16, 16]
#   W     = k_restore^T @ Ainvb                      [DK, 16]
#   QK'   = QK @ Ainvb                               [16, 16]
#   S_next = D (.) S + W @ (v - X),   X = k_decay @ round_bf16(S)
#   o      = (q_decay @ round_bf16(S) + QK' @ (v - X)) * scale
# The value-side u_v / o_v tiles are algebraically folded into W and QK'
# (k_restore^T @ u_v == W @ v and QK @ u_v == QK' @ v exactly), so kernel 2
# reads the ORIGINAL v tensor as the value-side A operand and issues one extra
# 16x16 qkv MMA instead of moving 8 KB/chunk of extra factors.
#
# Workspace pre-permutations (so kernel 2's raw s128 TMA loads land the exact
# SMEM images the engine's tcgen05 descriptors expect):
#   ws_kd/ws_qd[.., c*16 + t, k ^ 8 ^ ((t & 2) * 16)] = decay[t, k]
#     (the K-box-major SW128 storage key of tcgen05_decay_b_key_storage_dim_
#      runtime; the TMA s128b row pattern then reproduces
#      tcgen05_swizzle_128b_elem_index staging exactly)
#   ws_w[.., c*16 + (j ^ 8), k] = W^T[j, k]
#     (the engine's k_restore staging is raw_f16_s128 with token rows ^8)
#   ws_qk[.., c, pairwise_sw32_smem_index(0, i, j)] = QK'[i, j]
#   ws_diag[.., c, k] = 2^{g_last[k]}  (fp32; kernel 2's warp 15 scatters it
#     into the engine's SW32 block-diagonal ring off-chain)
# =============================================================================

K2_RAW_STAGE_COUNT: int = 8
# TMA completion-mbar ring depth (< raw ring depth: at depth 8 the raw slot
# recycles before w14 republishes, serializing slot reuse on consumer latency).
K2_TMA_MBAR_STAGE_COUNT: int = 6
if K2_TMA_MBAR_STAGE_COUNT > K2_RAW_STAGE_COUNT:
    raise ValueError("TMA mbar ring cannot be deeper than the raw ring")
TILE_ELEMS: int = BT * DK
DIAG_REC_ELEMS: int = DK  # per-chunk fp32 diag record
QK_REC_ELEMS: int = BT * BT  # per-chunk SW32-permuted QK' record
# kernel-2 TMEM: engine layout (cols 0..271) + 2-slot packed V A-operand ring.
TMEM_V_INPUT_COL: int = 272
TMEM_ALLOC_COLS: int = 512
# kernel-2 qstate/output acc ring depth.  Stages keep the engine's 48-col
# stride (208 / 256 / ...); with the drain on its own warpgroup (below) a
# shallow ring keeps the issuer from running ahead of the drain.
K2_QSTATE_STAGE_COUNT: int = 2
# The kernel-2 qstate output drain (TMEM read -> o_smem stage) runs on the
# otherwise-idle warps 0-3 (a lone warp cannot drain a 128-lane accumulator
# because tcgen05 lane access is restricted to the warp's `warp_id % 4`
# quadrant).
# --- kernel-2 DV split -----------------------------------------------------
# mode="decomp" routes through kernel_chain_dv2: one CTA per (sequence, head,
# DV-half), M=64 MMAs throughout (PTX Layout F).  DV2 is the only decomp split.
DV_HALF: int = DV // 2
ROWS_PER_WARP: int = DV_HALF // 4  # Layout F: 16 rows per quadrant
V_TILE_ELEMS: int = BT * DV_HALF  # one 64-elem s128 TMA segment
K2_O_SMEM_STAGE_SIZE: int = BT * DV_HALF
K2_O_SMEM_TILE_SIZE: int = K2_QSTATE_STAGE_COUNT * K2_O_SMEM_STAGE_SIZE
K2_TX_BYTES: int = (
    3 * TILE_ELEMS * 2  # kd + w + qd (bf16, full DK)
    + V_TILE_ELEMS * 2  # v (bf16, DV half)
    + DIAG_REC_ELEMS * 4  # diag (fp32)
    + QK_REC_ELEMS * 2  # qk' (bf16)
)
_K2_QSTATE_LAST_COL = (
    KDA_TMEM_QSTATE_ACC_COL_OFFSET
    + (K2_QSTATE_STAGE_COUNT - 1) * KDA_TMEM_QSTATE_ACC_STAGE_STRIDE_COLS
    + KDA_TMEM_N16_ACC_COLS
)
if _K2_QSTATE_LAST_COL > TMEM_ALLOC_COLS:
    raise ValueError("qstate acc ring exceeds the TMEM allocation")
if (
    K2_QSTATE_STAGE_COUNT > 2
    and KDA_TMEM_QSTATE_ACC_COL_OFFSET + 2 * KDA_TMEM_QSTATE_ACC_STAGE_STRIDE_COLS
    < TMEM_V_INPUT_COL + 2 * KDA_TMEM_SHARED_INPUT_COLS
):
    raise ValueError("qstate acc stage 2 overlaps the packed V ring")


@cute.jit
def super_mma_beta_col_scale_a_frags(
    a_frags,
    raw_beta_smem,
    lane,
    ab_frag,
    input_dtype: cutlass.Constexpr,
) -> None:
    """Scale a packed 16x16 A-layout fragment by beta per COLUMN.

    A-layout m16n8k16 fragments hold columns 2*(lane%4)/+1 in regs 0/1 and
    columns 8+2*(lane%4)/+1 in regs 2/3, so one packed beta pair per column
    half scales the whole fragment (Ainvb = A_inv (.) beta-col).  The scaled
    registers land in the caller-provided `ab_frag` array.
    """

    col_lo = 2 * (lane % 4)
    beta_lo = pack_input_b16x2_to_i32(
        raw_beta_smem[col_lo].to(cutlass.Float32),
        raw_beta_smem[col_lo + 1].to(cutlass.Float32),
        input_dtype,
    )
    beta_hi = pack_input_b16x2_to_i32(
        raw_beta_smem[col_lo + 8].to(cutlass.Float32),
        raw_beta_smem[col_lo + 9].to(cutlass.Float32),
        input_dtype,
    )
    ab_frag[0] = mul_b16x2_input_dtype(a_frags[0], beta_lo, input_dtype)
    ab_frag[1] = mul_b16x2_input_dtype(a_frags[1], beta_lo, input_dtype)
    ab_frag[2] = mul_b16x2_input_dtype(a_frags[2], beta_hi, input_dtype)
    ab_frag[3] = mul_b16x2_input_dtype(a_frags[3], beta_hi, input_dtype)


@cute.jit
def super_mma_load_ainv_a_frags(pairwise_stage_smem, lane):
    """Reload the staged A_inv pairwise tile as an m16n8k16 A-layout fragment."""

    return prims.ldmatrix(
        pairwise_stmatrix_m8n8x4_ptr(
            pairwise_stage_smem,
            PAIRWISE_SMEM_AINV_OFFSET,
            lane,
        ),
        4,
        prims.MMALayout.ROW,
    )


@cute.jit
def super_mma_load_k_restore_a_fragment(
    tcgen05_k_restore_smem,
    lane,
    n_group: cutlass.Constexpr,
):
    """Load an A-layout fragment of the k_restore tile (tokens x 16 key cols).

    The engine stages k_restore in the raw s128 layout with token rows XORed
    by TCGEN05_SW32_BT_HALF_XOR (see cg0_materialize_decay_operands), so the
    per-lane row coordinate applies the same xor before the s128 index.
    """

    lane_div8 = lane // 8
    lane_mod8 = lane % 8
    row_offset = cutlass.Int32(8) if (lane_div8 % 2) else cutlass.Int32(0)
    col_offset = cutlass.Int32(8) if (lane_div8 // 2) else cutlass.Int32(0)
    row_coord = (lane_mod8 + row_offset) ^ cutlass.Int32(TCGEN05_SW32_BT_HALF_XOR)
    col_coord = n_group * 16 + col_offset
    ptr = tcgen05_k_restore_smem.subview(raw_f16_s128_smem_index(row_coord, col_coord))
    return prims.ldmatrix(ptr.data_ptr(), 4, prims.MMALayout.ROW)


@cute.jit
def w_out_stmatrix_ptr(
    w_out_smem,
    n_group: cutlass.Constexpr,
    lane,
):
    """Per-lane row-segment pointer for one 16x16 STSM tile of the W image.

    The target SMEM image is the engine's N-major k_restore staging:
    raw_f16_s128 with token rows XORed by 8; a TMA bulk store of this tile
    then lands the pre-permuted ws_w global rows directly.
    """

    matrix_id = lane // 8
    row_coord = lane & 7
    col_coord = cutlass.Int32(n_group * 16)
    if matrix_id & 1:
        row_coord = row_coord + cutlass.Int32(8)
    if matrix_id >= 2:
        col_coord = col_coord + cutlass.Int32(8)
    storage_row = row_coord ^ cutlass.Int32(TCGEN05_SW32_BT_HALF_XOR)
    return w_out_smem.subview(
        raw_f16_s128_smem_index(storage_row, col_coord)
    ).data_ptr()


@cute.jit
def super_mma_stage_qk_prime(
    tcgen05_q_decay_smem,
    k_inv_smem,
    pairwise_stage_smem,
    raw_beta_smem,
    ws_qk: cute.Tensor,
    bidy,
    ws_chunk,
    lane,
    input_dtype: cutlass.Constexpr,
) -> None:
    """Produce QK' = QK @ Ainvb and store it in SW32 pairwise order to ws_qk.

    The QK accumulation matches super_mma_stage_qk (16 m16n8k16 K-blocks +
    causal mask) but keeps the tile in registers instead of staging it; the
    packed causal QK fragment is then contracted with the beta-scaled A_inv
    fragment and the f32 product is stored bf16-rounded, element-addressed
    with pairwise_sw32_smem_index so kernel 2's linear TMA copy of the
    512-byte record IS the engine's pairwise SMEM tile.
    """

    qk_n0_acc = cutlass.Array(
        cutlass.Float32,
        SUPER_MMA_ACCUMULATORS_PER_LANE,
        alignment=16,
    )
    qk_n1_acc = cutlass.Array(
        cutlass.Float32,
        SUPER_MMA_ACCUMULATORS_PER_LANE,
        alignment=16,
    )
    for accum_idx in cutlass.range_constexpr(SUPER_MMA_ACCUMULATORS_PER_LANE):
        qk_n0_acc[accum_idx] = cutlass.Float32(0.0)
        qk_n1_acc[accum_idx] = cutlass.Float32(0.0)

    for k_block in cutlass.range_constexpr(SUPER_MMA_K_BLOCKS):
        rhs_vec = super_mma_load_k_inv_rhs_fragment(
            k_inv_smem,
            lane,
            k_block,
        )
        qk_lhs_vec = super_mma_load_decay_lhs_fragment(
            tcgen05_q_decay_smem,
            lane,
            k_block,
        )
        qk_n0_d0, qk_n0_d1, qk_n0_d2, qk_n0_d3 = ptx_mma_m16n8k16_b16_f32(
            qk_lhs_vec[0],
            qk_lhs_vec[1],
            qk_lhs_vec[2],
            qk_lhs_vec[3],
            rhs_vec[0],
            rhs_vec[1],
            qk_n0_acc[0],
            qk_n0_acc[1],
            qk_n0_acc[2],
            qk_n0_acc[3],
            input_dtype,
        )
        qk_n0_acc[0] = qk_n0_d0
        qk_n0_acc[1] = qk_n0_d1
        qk_n0_acc[2] = qk_n0_d2
        qk_n0_acc[3] = qk_n0_d3
        qk_n1_d0, qk_n1_d1, qk_n1_d2, qk_n1_d3 = ptx_mma_m16n8k16_b16_f32(
            qk_lhs_vec[0],
            qk_lhs_vec[1],
            qk_lhs_vec[2],
            qk_lhs_vec[3],
            rhs_vec[2],
            rhs_vec[3],
            qk_n1_acc[0],
            qk_n1_acc[1],
            qk_n1_acc[2],
            qk_n1_acc[3],
            input_dtype,
        )
        qk_n1_acc[0] = qk_n1_d0
        qk_n1_acc[1] = qk_n1_d1
        qk_n1_acc[2] = qk_n1_d2
        qk_n1_acc[3] = qk_n1_d3

    qk_n0_acc[0] = super_mma_qk_causal_value(qk_n0_acc[0], lane, 0, 0)
    qk_n0_acc[1] = super_mma_qk_causal_value(qk_n0_acc[1], lane, 0, 1)
    qk_n0_acc[2] = super_mma_qk_causal_value(qk_n0_acc[2], lane, 0, 2)
    qk_n0_acc[3] = super_mma_qk_causal_value(qk_n0_acc[3], lane, 0, 3)
    qk_n1_acc[0] = super_mma_qk_causal_value(qk_n1_acc[0], lane, 1, 0)
    qk_n1_acc[1] = super_mma_qk_causal_value(qk_n1_acc[1], lane, 1, 1)
    qk_n1_acc[2] = super_mma_qk_causal_value(qk_n1_acc[2], lane, 1, 2)
    qk_n1_acc[3] = super_mma_qk_causal_value(qk_n1_acc[3], lane, 1, 3)

    qk_frag = cutlass.Array(
        cutlass.Int32,
        SUPER_MMA_ACCUMULATORS_PER_LANE,
        alignment=16,
    )
    super_mma_pack_pairwise_accumulator(
        qk_n0_acc,
        qk_n1_acc,
        qk_frag,
        input_dtype,
    )
    ainv = super_mma_load_ainv_a_frags(pairwise_stage_smem, lane)
    ab_frag = cutlass.Array(
        cutlass.Int32,
        SUPER_MMA_ACCUMULATORS_PER_LANE,
        alignment=16,
    )
    super_mma_beta_col_scale_a_frags(
        ainv,
        raw_beta_smem,
        lane,
        ab_frag,
        input_dtype,
    )
    p0_acc = cutlass.Array(
        cutlass.Float32,
        SUPER_MMA_ACCUMULATORS_PER_LANE,
        alignment=16,
    )
    p1_acc = cutlass.Array(
        cutlass.Float32,
        SUPER_MMA_ACCUMULATORS_PER_LANE,
        alignment=16,
    )
    super_mma_pairwise_product_from_ab_regs(
        qk_frag,
        ab_frag,
        p0_acc,
        p1_acc,
        input_dtype,
    )

    row_lo = super_mma_accumulator_row(lane, 0)
    row_hi = super_mma_accumulator_row(lane, 2)
    n0_col0 = super_mma_accumulator_col(lane, 0, 0)
    n0_col1 = super_mma_accumulator_col(lane, 0, 1)
    n1_col0 = super_mma_accumulator_col(lane, 1, 0)
    n1_col1 = super_mma_accumulator_col(lane, 1, 1)
    ws_qk[0, bidy, ws_chunk, pairwise_sw32_smem_index(0, row_lo, n0_col0)] = p0_acc[
        0
    ].to(input_dtype)
    ws_qk[0, bidy, ws_chunk, pairwise_sw32_smem_index(0, row_lo, n0_col1)] = p0_acc[
        1
    ].to(input_dtype)
    ws_qk[0, bidy, ws_chunk, pairwise_sw32_smem_index(0, row_hi, n0_col0)] = p0_acc[
        2
    ].to(input_dtype)
    ws_qk[0, bidy, ws_chunk, pairwise_sw32_smem_index(0, row_hi, n0_col1)] = p0_acc[
        3
    ].to(input_dtype)
    ws_qk[0, bidy, ws_chunk, pairwise_sw32_smem_index(0, row_lo, n1_col0)] = p1_acc[
        0
    ].to(input_dtype)
    ws_qk[0, bidy, ws_chunk, pairwise_sw32_smem_index(0, row_lo, n1_col1)] = p1_acc[
        1
    ].to(input_dtype)
    ws_qk[0, bidy, ws_chunk, pairwise_sw32_smem_index(0, row_hi, n1_col0)] = p1_acc[
        2
    ].to(input_dtype)
    ws_qk[0, bidy, ws_chunk, pairwise_sw32_smem_index(0, row_hi, n1_col1)] = p1_acc[
        3
    ].to(input_dtype)


@cute.jit
def chunk_coords(cu_seqlens, cu_chunks, gchunk):
    """Return coordinates for a global chunk using the cumulative chunk prefix."""

    # Binary-search the compact [num_sequences + 1] prefix instead of carrying
    # a redundant dense chunk_to_seq[total_chunks] inverse map.
    lo = cutlass.Int32(0)
    hi = cutlass.Int32(cu_chunks.shape[0] - 1)
    while lo + cutlass.Int32(1) < hi:
        mid = (lo + hi) // cutlass.Int32(2)
        if cutlass.Int32(cu_chunks[mid]) <= gchunk:
            lo = mid
        else:
            hi = mid
    seq = lo
    sequence_start = cutlass.Int32(cu_seqlens[seq])
    seqlen = cutlass.Int32(cu_seqlens[seq + 1]) - sequence_start
    chunk_start_tok = (gchunk - cutlass.Int32(cu_chunks[seq])) * cutlass.Int32(BT)
    return sequence_start, seqlen, chunk_start_tok


@cute.jit
def tma_prep_stage_load_inputs(
    tma_desc_q: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_k: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_gate: cutlass.GridConstant[cuda.TensorMap],
    beta_logits: cute.Tensor,
    raw_q_smem,
    raw_k_smem,
    raw_gate_smem,
    raw_beta_smem,
    sequence_start,
    head_idx,
    lane,
    chunk_start_tok,
    seqlen,
    tma_mbar,
    tma_tx_bytes: cutlass.Constexpr,
    gate_dtype: cutlass.Constexpr,
) -> None:
    """Kernel-1 chunk input transaction: q/k/gate TMA + beta sigmoid (no v)."""

    global_chunk_start = sequence_start + chunk_start_tok
    if prims.elect_sync():
        prims.mbarrier_arrive_expect_tx(tma_mbar, tma_tx_bytes)
    if prims.elect_sync():
        for segment in cutlass.range_constexpr(RAW_F16_TMA_SEGMENTS):
            tma_coord = (
                cutlass.Int32(segment * RAW_F16_TMA_SWIZZLE_ELEMS),
                global_chunk_start,
                head_idx,
                cutlass.Int32(0),
            )
            smem_offset = segment * RAW_F16_TMA_SEGMENT_ELEMS
            prims.cp_async_bulk_tensor_shared_cta_global(
                raw_q_smem.subview(smem_offset),
                tma_desc_q.get_ptr(),
                tma_coord,
                tma_mbar,
            )
            prims.cp_async_bulk_tensor_shared_cta_global(
                raw_k_smem.subview(smem_offset),
                tma_desc_k.get_ptr(),
                tma_coord,
                tma_mbar,
            )
            if cutlass.const_expr(not gate_dtype_is_f32(gate_dtype)):
                # 16-bit gate: shares the q/k 16-bit box/swizzle family.
                prims.cp_async_bulk_tensor_shared_cta_global(
                    raw_gate_smem.subview(smem_offset),
                    tma_desc_gate.get_ptr(),
                    tma_coord,
                    tma_mbar,
                )
        if cutlass.const_expr(gate_dtype_is_f32(gate_dtype)):
            for segment in cutlass.range_constexpr(RAW_F32_TMA_SEGMENTS):
                tma_coord = (
                    cutlass.Int32(segment * RAW_F32_TMA_SWIZZLE_ELEMS),
                    global_chunk_start,
                    head_idx,
                    cutlass.Int32(0),
                )
                smem_offset = segment * RAW_F32_TMA_SEGMENT_ELEMS
                prims.cp_async_bulk_tensor_shared_cta_global(
                    raw_gate_smem.subview(smem_offset),
                    tma_desc_gate.get_ptr(),
                    tma_coord,
                    tma_mbar,
                )

    if lane < BT:
        token_idx = chunk_start_tok + lane
        beta_value = cutlass.Float32(0.0)
        if token_idx < seqlen:
            beta_logit = beta_logits[0, sequence_start + token_idx, head_idx].to(
                cutlass.Float32
            )
            half = cutlass.Float32(0.5)
            beta_value = cute.math.tanh(beta_logit * half, approx=True) * half + half
        raw_beta_smem[lane] = beta_value


@cute.jit
def host_prep(
    q: cute.Tensor,
    k: cute.Tensor,
    raw_gate: cute.Tensor,
    a_log: cute.Tensor,
    dt_bias: cute.Tensor,
    beta: cute.Tensor,
    cu_seqlens: cute.Tensor,
    cu_chunks: cute.Tensor,
    ws_kd: cute.Tensor,
    ws_qd: cute.Tensor,
    ws_w: cute.Tensor,
    ws_qk: cute.Tensor,
    ws_diag: cute.Tensor,
    stream,
    num_ctas: cutlass.Int32,
    chunks_per_cta: cutlass.Int32,
    head_base: cutlass.Int32,
    launch_heads: cutlass.Int32,
    SAFE_GATE: cutlass.Constexpr,
    GATE_SCALE_LOG2: cutlass.Constexpr,
    THREADS: cutlass.Constexpr,
    gate_dtype: cutlass.Constexpr,
) -> None:
    seqlen = q.shape[1]
    heads = q.shape[2]
    # Token-major user activations (see host()).
    qk_layout = cute.make_layout(
        (DK, seqlen, heads, 1),
        stride=(1, DK * heads, DK, DK * seqlen * heads),
    )
    raw_f16_tma_box = (RAW_F16_TMA_SWIZZLE_ELEMS, BT, 1, 1)
    raw_f32_tma_box = (RAW_F32_TMA_SWIZZLE_ELEMS, BT, 1, 1)
    tma_desc_q = cuda.create_tensor_map_tiled_from_view(
        cute.make_tensor(q.iterator, qk_layout),
        box_dims=raw_f16_tma_box,
        stride_order=(0, 1, 2, 3),
        swizzle=cuda.TensorMapSwizzle.s128b,
    )
    tma_desc_k = cuda.create_tensor_map_tiled_from_view(
        cute.make_tensor(k.iterator, qk_layout),
        box_dims=raw_f16_tma_box,
        stride_order=(0, 1, 2, 3),
        swizzle=cuda.TensorMapSwizzle.s128b,
    )
    gate_tma_box = raw_f32_tma_box if gate_dtype_is_f32(gate_dtype) else raw_f16_tma_box
    tma_desc_gate = cuda.create_tensor_map_tiled_from_view(
        cute.make_tensor(raw_gate.iterator, qk_layout),
        box_dims=gate_tma_box,
        stride_order=(0, 1, 2, 3),
        swizzle=cuda.TensorMapSwizzle.s128b,
    )
    ws_rows = ws_kd.shape[2]
    # Canonicalize the unused head stride for H=1.  Leaving it proportional to
    # ws_rows reaches the grid-constant TensorMap launch boundary at 1M tokens,
    # even though a singleton mode never contributes to an address.
    ws_head_stride = DK
    if heads != cutlass.Int32(1):
        ws_head_stride = DK * ws_rows
    ws_tile_layout = cute.make_layout(
        (DK, ws_rows, heads, 1),
        stride=(1, DK, ws_head_stride, DK),
    )
    tma_desc_ws_kd = cuda.create_tensor_map_tiled_from_view(
        cute.make_tensor(ws_kd.iterator, ws_tile_layout),
        box_dims=raw_f16_tma_box,
        stride_order=(0, 1, 2, 3),
        swizzle=cuda.TensorMapSwizzle.s128b,
    )
    tma_desc_ws_qd = cuda.create_tensor_map_tiled_from_view(
        cute.make_tensor(ws_qd.iterator, ws_tile_layout),
        box_dims=raw_f16_tma_box,
        stride_order=(0, 1, 2, 3),
        swizzle=cuda.TensorMapSwizzle.s128b,
    )
    tma_desc_ws_w = cuda.create_tensor_map_tiled_from_view(
        cute.make_tensor(ws_w.iterator, ws_tile_layout),
        box_dims=raw_f16_tma_box,
        stride_order=(0, 1, 2, 3),
        swizzle=cuda.TensorMapSwizzle.s128b,
    )
    # per-chunk-tile kernel 1 (4-warp CTAs; THREADS is the prep ABI
    # placeholder and is ignored — the launch uses K1_THREADS with a
    # min-CTAs/SM occupancy floor so several CTAs co-reside per SM).
    kernel_prep(
        tma_desc_q,
        tma_desc_k,
        tma_desc_gate,
        tma_desc_ws_kd,
        tma_desc_ws_qd,
        tma_desc_ws_w,
        q,
        k,
        raw_gate,
        a_log,
        dt_bias,
        beta,
        cu_seqlens,
        cu_chunks,
        ws_kd,
        ws_qd,
        ws_w,
        ws_qk,
        ws_diag,
        chunks_per_cta,
        head_base,
        SAFE_GATE,
        GATE_SCALE_LOG2,
        gate_dtype,
    ).launch(
        grid=(num_ctas, launch_heads, 1),
        block=(K1_THREADS, 1, 1),
        stream=stream,
        min_blocks_per_mp=K1_MIN_BLOCKS,
        # Explicit carveout: (a) several 38-KB CTAs per SM need the max
        # SMEM split anyway, (b) leaving it None makes min_blocks_per_mp>1
        # query the device at COMPILE time, which breaks the login-node
        # CUTE_DSL_DRYRUN flow.
        preferred_smem_carveout=100,
    )


_PLAN_CACHE: dict = {}


def _plan(cu_seqlens: torch.Tensor) -> dict:
    """Cached cumulative chunk-prefix tensor and host metadata.

    Keyed by the cu_seqlens CONTENTS (data_ptr identity is unsafe under
    allocator block reuse — the s27 stale-plan hazard; the read is
    memoized sync-free by `_cu_seqlens_contents`).
    """

    cu_list = list(_cu_seqlens_contents(cu_seqlens))
    key = (tuple(cu_list), str(cu_seqlens.device))
    plan = _PLAN_CACHE.get(key)
    if plan is None:
        chunks = [
            (cu_list[i + 1] - cu_list[i] + BT - 1) // BT
            for i in range(len(cu_list) - 1)
        ]
        cu_chunks = [0]
        for c in chunks:
            cu_chunks.append(cu_chunks[-1] + c)
        total_chunks = cu_chunks[-1]
        plan = {
            "total_chunks": total_chunks,
            "cu_chunks": torch.tensor(
                cu_chunks, dtype=torch.int32, device=cu_seqlens.device
            ),
            "cu_list": cu_list,
        }
        _PLAN_CACHE[key] = plan
    return plan


# ---------------------------------------------------------------------------
# CUTLASS-style user-allocated decomp workspace (single opaque uint8 buffer).
#
# The decomp route needs the kernel-1 -> kernel-2 factor tiles plus a tiny
# flag array.  Instead of a hidden per-shape cache, the caller queries
# `workspace_size(cu_seqlens, heads[, mode])` and allocates one raw uint8
# buffer; `host` partitions it itself and does ALL init (it zeroes the
# flag region each call).  The buffer is kernel-managed: pass a raw/uninitialized
# buffer, reuse it freely across calls, never touch it otherwise.
#
# Layout (each region rounded up to WS_REGION_ALIGN so every view is
# >=16B aligned for TMA):
#   [ ws_kd | ws_qd | ws_w | ws_qk | ws_diag ]
#   ws_kd/ws_qd/ws_w : bf16 (1, heads, total_chunks*BT, DK)
#   ws_qk            : bf16 (1, heads, total_chunks, QK_REC_ELEMS)
#   ws_diag          : f32  (1, heads, total_chunks, DIAG_REC_ELEMS)
# ---------------------------------------------------------------------------

WS_REGION_ALIGN: int = 256


def _align_up(n: int, a: int) -> int:
    return -(-int(n) // a) * a


def _decomp_ws_regions(heads: int, total_chunks: int) -> list[tuple]:
    """Region byte sizes and shapes for the decomp workspace, in buffer order.

    Each entry is (nbytes, torch_dtype, shape).
    """

    rows = total_chunks * BT
    tile = (heads * rows * DK * 2, torch.bfloat16, (1, heads, rows, DK))
    return [
        tile,
        tile,
        tile,
        (
            heads * total_chunks * QK_REC_ELEMS * 2,
            torch.bfloat16,
            (1, heads, total_chunks, QK_REC_ELEMS),
        ),
        (
            heads * total_chunks * DIAG_REC_ELEMS * 4,
            torch.float32,
            (1, heads, total_chunks, DIAG_REC_ELEMS),
        ),
    ]


def _decomp_ws_bytes(heads: int, total_chunks: int) -> int:
    return sum(
        _align_up(nbytes, WS_REGION_ALIGN)
        for nbytes, _, _ in _decomp_ws_regions(heads, total_chunks)
    )


def _partition_workspace(workspace, heads: int, total_chunks: int) -> dict:
    """View the user uint8 buffer as the six factor/flag tensors (no copy)."""

    if workspace.dtype != torch.uint8 or workspace.dim() != 1:
        raise ValueError("workspace must be a 1-D uint8 tensor")
    need = _decomp_ws_bytes(heads, total_chunks)
    if workspace.numel() < need:
        raise ValueError(
            f"workspace too small: {workspace.numel()} bytes < {need} required; "
            "call workspace_size() to size it"
        )
    keys = ("kd", "qd", "w", "qk", "diag")
    out: dict = {}
    off = 0
    for key, (nbytes, dtype, shape) in zip(
        keys, _decomp_ws_regions(heads, total_chunks), strict=True
    ):
        seg = workspace.narrow(0, off, nbytes)
        out[key] = seg.view(dtype).view(*shape)
        off += _align_up(nbytes, WS_REGION_ALIGN)
    return out


def _route_for_workspace(num_sequences: int, heads: int, device, mode: str) -> str:
    if mode == "engine":
        return "engine"
    if mode == "decomp":
        return "decomp"
    if mode != "auto":
        raise ValueError(f"Unknown mode: {mode}")
    sm_count = _device_sm_count(device)
    return "decomp" if num_sequences * heads * 2 <= sm_count else "engine"


def workspace_size(cu_seqlens, heads: int, mode: str = "auto") -> int:
    """Bytes for the opaque decomp workspace of this problem shape.

    Returns 0 for the engine route (engine needs no workspace), else the total
    partitioned + 256B-region-aligned byte count.  Uses the same CTA-count
    occupancy rule as `host`, so the query and the launch agree.
    ``cu_seqlens`` is the int64 cumulative-length tensor (or a host sequence).
    """

    heads = int(heads)
    if isinstance(cu_seqlens, torch.Tensor):
        device = cu_seqlens.device
        num_sequences = cu_seqlens.numel() - 1
    else:
        cu_list = tuple(int(x) for x in cu_seqlens)
        device = torch.cuda.current_device()
        num_sequences = len(cu_list) - 1
    if _route_for_workspace(num_sequences, heads, device, mode) == "engine":
        return 0
    if isinstance(cu_seqlens, torch.Tensor):
        cu_list = _cu_seqlens_contents(cu_seqlens)
    total_chunks = sum(
        (cu_list[i + 1] - cu_list[i] + BT - 1) // BT for i in range(len(cu_list) - 1)
    )
    return _decomp_ws_bytes(heads, total_chunks)


def _k1_grid(total_chunks: int, heads: int) -> tuple[int, int]:
    """Return (num_ctas, chunks_per_cta) for the per-chunk-tile grid.

    One CTA per K1_CPC contiguous chunks (the measured ladder puts the
    optimum at 4); `heads` is the grid's y dimension and does not shape x.
    """

    if total_chunks <= 0:
        return 1, 1
    chunks_per_cta = max(1, K1_CPC)
    num_ctas = -(-total_chunks // chunks_per_cta)
    return num_ctas, chunks_per_cta


# =============================================================================
# kernel 1: FLA-style per-chunk-tile factor prep.
#
# One small CTA per (chunk-group, head) tile; 4 warps, no warp
# specialization, no cross-chunk rings.  All factor math and workspace
# pre-permutations are the ring-based donors (byte-identical output); only the
# CTA structure changes.  Warp roles inside the CTA:
#   warp 0: TMA issue + beta, CG0 rows 0-3,  KK -> L -> A_inv,
#           overlap-flag publish
#   warp 1: CG0 rows 4-7,   W fold columns [0, 64)  + W TMA store (seg 0)
#   warp 2: CG0 rows 8-11,  QK' fold (plain global stores)
#   warp 3: CG0 rows 12-15, kd/qd TMA stores, W fold columns [64, 128)
#           + W TMA store (seg 1)
# =============================================================================

K1_WARPS: int = 4
K1_THREADS: int = K1_WARPS * THREADS_PER_WARP
# Chunks walked per CTA (grid x = ceil(total_chunks / CPC)).  1 = pure
# per-chunk-tile grid; >1 amortizes launch/preamble cost per CTA and
# enables the mid-chunk prefetch of the next chunk's raw tiles (measured
# ladder: CPC=4 optimal).
K1_CPC: int = 4
# Launch occupancy floor (min CTAs per SM): caps regs/thread so several
# small CTAs co-reside per SM — the entire point of this kernel shape.
# Measured: 5 (~102 regs/thread) beats 4 (128 regs) and 8 (64 regs).
K1_MIN_BLOCKS: int = 5
K1_W_N_GROUPS: int = DK // 16
K1_W_N_SPLIT: int = K1_W_N_GROUPS // 2
# Which warp issues the CPC>1 mid-chunk prefetch of the next chunk's raw
# tiles (must be a warp that waits q_k_restore_ready: 1 = W-half warp
# (lighter tail), 2 = QK' warp).
K1_PREFETCH_WARP: int = 1


@cute.jit
def prep_cta_sync() -> None:
    """Synchronize the 4-warp kernel-1 CTA."""

    prims.barrier_cta_sync(0, thread_count=K1_THREADS)


@cute.jit
def prefetch_next_chunk(
    tma_desc_q: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_k: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_gate: cutlass.GridConstant[cuda.TensorMap],
    beta: cute.Tensor,
    raw_q_smem,
    raw_k_smem,
    raw_gate_smem,
    raw_beta_smem,
    cu_seqlens: cute.Tensor,
    cu_chunks: cute.Tensor,
    local_chunk,
    my_chunks,
    gchunk,
    gchunk_stride,
    bidy,
    lane,
    tma_mbar,
    tma_tx_bytes: cutlass.Constexpr,
    gate_dtype: cutlass.Constexpr,
) -> None:
    """CPC>1 prefetch: overwrite the raw tiles with chunk c+1's transaction.

    Caller contract: issued only AFTER this chunk's q_k_restore_ready
    completed (all 4 warps arrived), i.e. every raw q/k/gate read of the
    chunk in flight is done.  The FP32 gate-prefix exchange is produced AND
    consumed inside materialize, either aliased into raw_gate_smem (FP32 gate)
    or in its own gate_exchange_smem tile (16-bit gate); this prefetch only
    rewrites the raw tiles, so it cannot disturb it either way.  Beta lands in the
    other half of the double-buffered record; the transaction completes
    tma_mbar[(c+1) % 2].
    """

    if local_chunk + cutlass.Int32(1) < my_chunks:
        next_gchunk = gchunk + gchunk_stride
        (
            next_sequence_start,
            next_seqlen,
            next_chunk_start_tok,
        ) = chunk_coords(
            cu_seqlens,
            cu_chunks,
            next_gchunk,
        )
        tma_prep_stage_load_inputs(
            tma_desc_q,
            tma_desc_k,
            tma_desc_gate,
            beta,
            raw_q_smem,
            raw_k_smem,
            raw_gate_smem,
            raw_beta_smem.subview(((local_chunk + 1) % 2) * RAW_BETA_STAGE_SIZE),
            next_sequence_start,
            bidy,
            lane,
            next_chunk_start_tok,
            next_seqlen,
            tma_mbar.subview((local_chunk + 1) % 2),
            tma_tx_bytes,
            gate_dtype,
        )


@cute.jit
def cg0_materialize_prep_operands(
    raw_q_smem,
    raw_k_smem,
    raw_gate_smem,
    gate_exchange_smem,
    a_log_exp,
    dt_bias_value,
    k_inv_smem,
    tcgen05_k_decay_smem,
    tcgen05_q_decay_smem,
    tcgen05_k_restore_smem,
    cg0_k_ready_mbar,
    cg0_k_half_ready_mbar,
    ws_diag: cute.Tensor,
    bidy,
    ws_chunk,
    chunk_start_tok,
    seqlen,
    input_dtype: cutlass.Constexpr,
    gate_dtype: cutlass.Constexpr,
    SAFE_GATE: cutlass.Constexpr,
    GATE_SCALE_LOG2: cutlass.Constexpr,
    cg0_local_warp,
    lane,
) -> None:
    """One-shot ring-free CG0 materializer.

    All 4 CTA warps form a single CG0 group (cg0_local_warp = warp index);
    the ring-stage consumed waits of the ring-based version are gone (single
    stage SMEM, chunk serialization by the caller's CTA sync).  The
    cg0_k_half/full ready arrives are kept so warp 0's KK K-blocks 0..3
    can start while the second half-DK is still being staged.
    """

    row_group_start = cg0_local_warp * CG0_TOKEN_ROWS_PER_WARP
    lane_row_group = lane // RAW_F16_TMA_SWIZZLE_GROUP_ELEMS
    lane_in_row_group = lane - lane_row_group * RAW_F16_TMA_SWIZZLE_GROUP_ELEMS
    decay_row = row_group_start + lane_row_group

    raw_q_ptr = raw_q_smem.data_ptr()
    raw_k_ptr = raw_k_smem.data_ptr()
    # The FP32 gate-prefix exchange buffer.  With an FP32 gate the caller
    # passes the raw-gate stage itself (the historical aliasing); with a 16-bit
    # gate it is the dedicated `gate_exchange_smem` tile.
    g_prefix_ptr = gate_exchange_smem.data_ptr()
    k_inv_ptr = k_inv_smem.data_ptr()
    tcgen05_k_decay_ptr = tcgen05_k_decay_smem.data_ptr()
    tcgen05_q_decay_ptr = tcgen05_q_decay_smem.data_ptr()
    tcgen05_k_restore_ptr = tcgen05_k_restore_smem.data_ptr()

    prefix_dim = cg0_local_warp * THREADS_PER_WARP + lane
    g_prefix_regs = cutlass.Array(
        cutlass.Float32,
        BT,
        alignment=16,
    )
    if cutlass.const_expr(SAFE_GATE):
        # Fold tanh's exact 0.5 scale into the chunk-uniform coefficient.
        a_log_exp_half = a_log_exp * cutlass.Float32(0.5)
        for row_pair in cutlass.range_constexpr(BT // 2):
            row0 = row_pair * 2
            row1 = row0 + 1
            # Only the MEMORY FORMAT of the raw gate depends on gate_dtype:
            # a 16-bit gate lands in the q/k s128 geometry and is widened to
            # FP32 right here -- ALL gate arithmetic below stays FP32.
            if cutlass.const_expr(gate_dtype_is_f32(gate_dtype)):
                prefix_idx0 = raw_f32_s128_smem_index(row0, prefix_dim)
                prefix_idx1 = raw_f32_s128_smem_index(row1, prefix_dim)
                gate0 = raw_gate_smem[prefix_idx0]
                gate1 = raw_gate_smem[prefix_idx1]
            else:
                prefix_idx0 = raw_f16_s128_smem_index(row0, prefix_dim)
                prefix_idx1 = raw_f16_s128_smem_index(row1, prefix_dim)
                gate0 = raw_gate_smem[prefix_idx0].to(cutlass.Float32)
                gate1 = raw_gate_smem[prefix_idx1].to(cutlass.Float32)
            gate0 = a_log_exp_half * (gate0 + dt_bias_value)
            gate1 = a_log_exp_half * (gate1 + dt_bias_value)
            gate0 = safe_gate_log2_increment_prehalved(
                gate0,
                SAFE_GATE,
                GATE_SCALE_LOG2,
            )
            gate1 = safe_gate_log2_increment_prehalved(
                gate1,
                SAFE_GATE,
                GATE_SCALE_LOG2,
            )
            gate_pair = cutlass.Vector.from_elements((gate0, gate1), cutlass.Float32)
            g_prefix_regs[row0] = gate_pair[0]
            g_prefix_regs[row1] = gate_pair[1]
    else:
        # FLA-compatible non-safe gate: accept raw gate logits + A_log +
        # dt_bias and compute the log2-domain decay increment in-kernel as
        # ``-exp(A_log) * softplus(raw_gate + dt_bias) * log2(e)``.  a_log_exp
        # already carries ``exp2(A_log * log2(e)) == exp(A_log)`` and the
        # softplus is evaluated in base-2 units (softplus_log2_f32).
        for row in cutlass.range_constexpr(BT):
            if cutlass.const_expr(gate_dtype_is_f32(gate_dtype)):
                prefix_idx = raw_f32_s128_smem_index(row, prefix_dim)
                gate = raw_gate_smem[prefix_idx]
            else:
                prefix_idx = raw_f16_s128_smem_index(row, prefix_dim)
                gate = raw_gate_smem[prefix_idx].to(cutlass.Float32)
            gate = -a_log_exp * softplus_log2_f32(gate + dt_bias_value)
            g_prefix_regs[row] = gate

    # Masked-zeroing tail fixup: a warp-uniform runtime compare (one per chunk)
    # applies only to a sequence's genuinely partial tail chunk; full chunks
    # skip it via the `tail_valid_rows < BT` test.
    tail_valid_rows = seqlen - chunk_start_tok
    if tail_valid_rows < cutlass.Int32(BT):
        tail_mask_pt = cutlass.vector.create_mask([BT], [tail_valid_rows])
        for row_pair_pt in cutlass.range_constexpr(BT // 2):
            row0_pt = row_pair_pt * 2
            row1_pt = row0_pt + 1
            gate_pair_pt = cutlass.Vector.from_elements(
                (g_prefix_regs[row0_pt], g_prefix_regs[row1_pt]),
                cutlass.Float32,
            )
            gate_pair_pt = cutlass.vector.where(
                tail_mask_pt[row0_pt : row1_pt + 1], gate_pair_pt, 0.0
            )
            g_prefix_regs[row0_pt] = gate_pair_pt[0]
            g_prefix_regs[row1_pt] = gate_pair_pt[1]

    prefix_acc = cutlass.Float32(0.0)
    for row_pair in cutlass.range_constexpr(BT // 2):
        row0 = row_pair * 2
        row1 = row0 + 1
        # The scalar scan has the same dependency depth as packed fadd2 but
        # avoids the register copy needed to construct its first input pair.
        prefix0 = prefix_acc + g_prefix_regs[row0]
        prefix1 = prefix0 + g_prefix_regs[row1]
        g_prefix_regs[row0] = prefix0
        g_prefix_regs[row1] = prefix1
        prefix_acc = prefix1

    for row in cutlass.range_constexpr(BT):
        g_prefix_regs[row] = cute.math.exp2(g_prefix_regs[row], fastmath=True)

    exp_g_last = g_prefix_regs[BT - 1]
    for row in cutlass.range_constexpr(BT):
        prefix_idx = raw_f32_exchange_smem_index(row, prefix_dim)
        gate_exchange_smem[prefix_idx] = g_prefix_regs[row]

    ws_diag[0, bidy, ws_chunk, prefix_dim] = exp_g_last

    cg0_sync(cutlass.Int32(0))

    k_inv_regs = cutlass.Array(
        input_dtype,
        2 * RAW_F16_TMA_SWIZZLE_GROUP_ELEMS,
        alignment=16,
    )
    k_restore_all_regs = cutlass.Array(
        input_dtype,
        2 * RAW_F16_TMA_SWIZZLE_GROUP_ELEMS,
        alignment=16,
    )
    raw_q_regs = cutlass.Array(
        cutlass.Float32,
        2 * RAW_F16_TMA_SWIZZLE_GROUP_ELEMS,
        alignment=16,
    )
    raw_k_regs = cutlass.Array(
        cutlass.Float32,
        2 * RAW_F16_TMA_SWIZZLE_GROUP_ELEMS,
        alignment=16,
    )
    q_sum_sq = cutlass.Float32(0.0)
    k_sum_sq = cutlass.Float32(0.0)
    for dim_half in cutlass.range_constexpr(2):
        dim_base = (
            dim_half * (DK // 2) + lane_in_row_group * RAW_F16_TMA_SWIZZLE_GROUP_ELEMS
        )
        reg_base = dim_half * RAW_F16_TMA_SWIZZLE_GROUP_ELEMS
        raw_f16_idx = raw_f16_s128_smem_index(decay_row, dim_base)
        raw_q_vec = (raw_q_ptr + raw_f16_idx).load(
            count=RAW_F16_TMA_SWIZZLE_GROUP_ELEMS,
            alignment=RAW_F16_TMA_SWIZZLE_GROUP_BYTES,
        )
        raw_k_vec = (raw_k_ptr + raw_f16_idx).load(
            count=RAW_F16_TMA_SWIZZLE_GROUP_ELEMS,
            alignment=RAW_F16_TMA_SWIZZLE_GROUP_BYTES,
        )
        raw_q_vec_f32 = raw_q_vec.to(cutlass.Float32)
        raw_k_vec_f32 = raw_k_vec.to(cutlass.Float32)
        for dim_offset in cutlass.range_constexpr(RAW_F16_TMA_SWIZZLE_GROUP_ELEMS):
            q_val = raw_q_vec_f32[dim_offset]
            k_val = raw_k_vec_f32[dim_offset]
            raw_q_regs[reg_base + dim_offset] = q_val
            raw_k_regs[reg_base + dim_offset] = k_val
            q_sum_sq = q_sum_sq + q_val * q_val
            k_sum_sq = k_sum_sq + k_val * k_val

    q_sum_sq = warp_group_sum_8(q_sum_sq)
    k_sum_sq = warp_group_sum_8(k_sum_sq)
    norm_floor_sq = cutlass.Float32(L2_NORM_EPS * L2_NORM_EPS)
    q_inv_norm = cute.math.rsqrt(
        cute.math.max(q_sum_sq, norm_floor_sq, ftz=True),
        fastmath=True,
    )
    k_inv_norm = cute.math.rsqrt(
        cute.math.max(k_sum_sq, norm_floor_sq, ftz=True),
        fastmath=True,
    )

    exp_g_regs = cutlass.Array(
        cutlass.Float32,
        2 * RAW_F16_TMA_SWIZZLE_GROUP_ELEMS,
        alignment=16,
    )
    exp_g_last_regs = cutlass.Array(
        cutlass.Float32,
        2 * RAW_F16_TMA_SWIZZLE_GROUP_ELEMS,
        alignment=16,
    )
    for dim_half in cutlass.range_constexpr(2):
        dim_base = (
            dim_half * (DK // 2) + lane_in_row_group * RAW_F16_TMA_SWIZZLE_GROUP_ELEMS
        )
        reg_base = dim_half * RAW_F16_TMA_SWIZZLE_GROUP_ELEMS
        exp_neg_g_regs = cutlass.Array(
            cutlass.Float32,
            RAW_F16_TMA_SWIZZLE_GROUP_ELEMS,
            alignment=16,
        )
        for f32_group in cutlass.range_constexpr(
            RAW_F16_TMA_SWIZZLE_GROUP_ELEMS // RAW_F32_TMA_SWIZZLE_GROUP_ELEMS
        ):
            f32_dim_base = dim_base + f32_group * RAW_F32_TMA_SWIZZLE_GROUP_ELEMS
            g_prefix_idx = raw_f32_exchange_smem_index(decay_row, f32_dim_base)
            exp_g_vec = (g_prefix_ptr + g_prefix_idx).load(
                count=RAW_F32_TMA_SWIZZLE_GROUP_ELEMS,
                alignment=RAW_F32_TMA_SWIZZLE_GROUP_BYTES,
            )
            exp_g_last_idx = raw_f32_exchange_smem_index(BT - 1, f32_dim_base)
            exp_g_last_vec = (g_prefix_ptr + exp_g_last_idx).load(
                count=RAW_F32_TMA_SWIZZLE_GROUP_ELEMS,
                alignment=RAW_F32_TMA_SWIZZLE_GROUP_BYTES,
            )
            half_reg_base = f32_group * RAW_F32_TMA_SWIZZLE_GROUP_ELEMS
            f32_reg_base = reg_base + half_reg_base
            exp_g_regs[f32_reg_base] = exp_g_vec[0]
            exp_g_regs[f32_reg_base + 1] = exp_g_vec[1]
            exp_g_regs[f32_reg_base + 2] = exp_g_vec[2]
            exp_g_regs[f32_reg_base + 3] = exp_g_vec[3]
            exp_neg_g_regs[half_reg_base] = cute.math.rcp(
                exp_g_vec[0], approx=True, ftz=True
            )
            exp_neg_g_regs[half_reg_base + 1] = cute.math.rcp(
                exp_g_vec[1], approx=True, ftz=True
            )
            exp_neg_g_regs[half_reg_base + 2] = cute.math.rcp(
                exp_g_vec[2], approx=True, ftz=True
            )
            exp_neg_g_regs[half_reg_base + 3] = cute.math.rcp(
                exp_g_vec[3], approx=True, ftz=True
            )
            exp_g_last_regs[f32_reg_base] = exp_g_last_vec[0]
            exp_g_last_regs[f32_reg_base + 1] = exp_g_last_vec[1]
            exp_g_last_regs[f32_reg_base + 2] = exp_g_last_vec[2]
            exp_g_last_regs[f32_reg_base + 3] = exp_g_last_vec[3]

        k_decay_vec_regs = cutlass.Array(
            input_dtype,
            RAW_F16_TMA_SWIZZLE_GROUP_ELEMS,
            alignment=RAW_F16_TMA_SWIZZLE_GROUP_BYTES,
        )
        for pair_idx in cutlass.range_constexpr(RAW_F16_TMA_SWIZZLE_GROUP_ELEMS // 2):
            dim0 = pair_idx * 2
            dim1 = dim0 + 1
            raw_reg_idx0 = reg_base + dim0
            raw_reg_idx1 = reg_base + dim1
            k_value0, k_value1 = fmul2(
                (raw_k_regs[raw_reg_idx0], raw_k_regs[raw_reg_idx1]),
                (k_inv_norm, k_inv_norm),
            )
            k_decay0, k_decay1 = fmul2(
                (k_value0, k_value1),
                (exp_g_regs[raw_reg_idx0], exp_g_regs[raw_reg_idx1]),
            )
            k_inv0, k_inv1 = fmul2(
                (k_value0, k_value1),
                (exp_neg_g_regs[dim0], exp_neg_g_regs[dim1]),
            )
            k_inv_regs[reg_base + dim0] = k_inv0.to(input_dtype)
            k_inv_regs[reg_base + dim1] = k_inv1.to(input_dtype)
            k_restore0, k_restore1 = fmul2(
                (k_inv0, k_inv1),
                (
                    exp_g_last_regs[reg_base + dim0],
                    exp_g_last_regs[reg_base + dim1],
                ),
            )
            k_restore_all_regs[reg_base + dim0] = k_restore0.to(input_dtype)
            k_restore_all_regs[reg_base + dim1] = k_restore1.to(input_dtype)
            k_decay_vec_regs[dim0] = k_decay0.to(input_dtype)
            k_decay_vec_regs[dim1] = k_decay1.to(input_dtype)

        k_inv_vec = cutlass.Vector.from_elements(
            (
                k_inv_regs[reg_base],
                k_inv_regs[reg_base + 1],
                k_inv_regs[reg_base + 2],
                k_inv_regs[reg_base + 3],
                k_inv_regs[reg_base + 4],
                k_inv_regs[reg_base + 5],
                k_inv_regs[reg_base + 6],
                k_inv_regs[reg_base + 7],
            ),
            input_dtype,
        )
        k_decay_vec = cutlass.Vector.from_elements(
            (
                k_decay_vec_regs[0],
                k_decay_vec_regs[1],
                k_decay_vec_regs[2],
                k_decay_vec_regs[3],
                k_decay_vec_regs[4],
                k_decay_vec_regs[5],
                k_decay_vec_regs[6],
                k_decay_vec_regs[7],
            ),
            input_dtype,
        )
        k_inv_swizzled_idx = k_inv_s128_smem_index(decay_row, dim_base)
        (k_inv_ptr + k_inv_swizzled_idx).store(
            k_inv_vec,
            alignment=RAW_F16_TMA_SWIZZLE_GROUP_BYTES,
        )
        decay_storage_dim_base = tcgen05_decay_b_key_storage_dim_runtime(
            decay_row,
            dim_base,
        )
        decay_linear_idx_base = decay_row * DK + decay_storage_dim_base
        decay_swizzled_idx_base = tcgen05_swizzle_128b_elem_index(
            decay_linear_idx_base,
            TCGEN05_F16_ELEM_BYTES,
            BT,
        )
        (tcgen05_k_decay_ptr + decay_swizzled_idx_base).store(
            k_decay_vec,
            alignment=RAW_F16_TMA_SWIZZLE_GROUP_BYTES,
        )
        if cutlass.const_expr(dim_half == 0):
            # Publish the first half-DK of k_inv/k_decay early: warp 0's
            # KK K-blocks 0..3 only read key dims [0, 64).
            cg0_k_ready_arrive(cg0_k_half_ready_mbar)
    cg0_k_ready_arrive(cg0_k_ready_mbar)

    for dim_half in cutlass.range_constexpr(2):
        dim_base = (
            dim_half * (DK // 2) + lane_in_row_group * RAW_F16_TMA_SWIZZLE_GROUP_ELEMS
        )
        reg_base = dim_half * RAW_F16_TMA_SWIZZLE_GROUP_ELEMS
        q_decay_vec_regs = cutlass.Array(
            input_dtype,
            RAW_F16_TMA_SWIZZLE_GROUP_ELEMS,
            alignment=RAW_F16_TMA_SWIZZLE_GROUP_BYTES,
        )
        for pair_idx in cutlass.range_constexpr(RAW_F16_TMA_SWIZZLE_GROUP_ELEMS // 2):
            dim0 = pair_idx * 2
            dim1 = dim0 + 1
            raw_reg_idx0 = reg_base + dim0
            raw_reg_idx1 = reg_base + dim1
            q_value0, q_value1 = fmul2(
                (raw_q_regs[raw_reg_idx0], raw_q_regs[raw_reg_idx1]),
                (q_inv_norm, q_inv_norm),
            )
            q_decay0, q_decay1 = fmul2(
                (q_value0, q_value1),
                (exp_g_regs[raw_reg_idx0], exp_g_regs[raw_reg_idx1]),
            )
            q_decay_vec_regs[dim0] = q_decay0.to(input_dtype)
            q_decay_vec_regs[dim1] = q_decay1.to(input_dtype)

        q_decay_vec = cutlass.Vector.from_elements(
            (
                q_decay_vec_regs[0],
                q_decay_vec_regs[1],
                q_decay_vec_regs[2],
                q_decay_vec_regs[3],
                q_decay_vec_regs[4],
                q_decay_vec_regs[5],
                q_decay_vec_regs[6],
                q_decay_vec_regs[7],
            ),
            input_dtype,
        )
        decay_storage_dim_base = tcgen05_decay_b_key_storage_dim_runtime(
            decay_row,
            dim_base,
        )
        decay_linear_idx_base = decay_row * DK + decay_storage_dim_base
        decay_swizzled_idx_base = tcgen05_swizzle_128b_elem_index(
            decay_linear_idx_base,
            TCGEN05_F16_ELEM_BYTES,
            BT,
        )
        (tcgen05_q_decay_ptr + decay_swizzled_idx_base).store(
            q_decay_vec,
            alignment=RAW_F16_TMA_SWIZZLE_GROUP_BYTES,
        )

    for dim_half in cutlass.range_constexpr(2):
        dim_base = (
            dim_half * (DK // 2) + lane_in_row_group * RAW_F16_TMA_SWIZZLE_GROUP_ELEMS
        )
        reg_base = dim_half * RAW_F16_TMA_SWIZZLE_GROUP_ELEMS
        storage_row = decay_row ^ TCGEN05_SW32_BT_HALF_XOR
        k_restore_idx = raw_f16_s128_smem_index(storage_row, dim_base)
        k_restore_vec = cutlass.Vector.from_elements(
            (
                k_restore_all_regs[reg_base],
                k_restore_all_regs[reg_base + 1],
                k_restore_all_regs[reg_base + 2],
                k_restore_all_regs[reg_base + 3],
                k_restore_all_regs[reg_base + 4],
                k_restore_all_regs[reg_base + 5],
                k_restore_all_regs[reg_base + 6],
                k_restore_all_regs[reg_base + 7],
            ),
            input_dtype,
        )
        (tcgen05_k_restore_ptr + k_restore_idx).store(
            k_restore_vec,
            alignment=RAW_F16_TMA_SWIZZLE_GROUP_BYTES,
        )


@cute.jit
def super_mma_stage_w_half(
    tcgen05_k_restore_smem,
    pairwise_stage_smem,
    raw_beta_smem,
    w_out_smem,
    lane,
    input_dtype: cutlass.Constexpr,
    n_group_lo: cutlass.Constexpr,
    n_group_hi: cutlass.Constexpr,
) -> None:
    """W^T = Ainvb^T @ k_restore for key-column groups [n_lo, n_hi).

    Column-split variant of super_mma_stage_w: groups [0, 4) cover key
    dims [0, 64) = W TMA segment 0, groups [4, 8) segment 1, so two warps
    stage and bulk-store their halves fully independently.  The Ainvb^T
    fragment build is duplicated per warp (cheap: one ldmatrix + 4
    movmatrix + the beta packs).
    """

    ainv = super_mma_load_ainv_a_frags(pairwise_stage_smem, lane)
    ab_frag = cutlass.Array(
        cutlass.Int32,
        SUPER_MMA_ACCUMULATORS_PER_LANE,
        alignment=16,
    )
    super_mma_beta_col_scale_a_frags(
        ainv,
        raw_beta_smem,
        lane,
        ab_frag,
        input_dtype,
    )
    # A-layout fragment of Ainvb^T: transpose quadrants, swap off-diagonals.
    t0 = movmatrix_b16(ab_frag[0])
    t1 = movmatrix_b16(ab_frag[2])
    t2 = movmatrix_b16(ab_frag[1])
    t3 = movmatrix_b16(ab_frag[3])

    # Literal FP32 operands into the inline-PTX MMA are rejected downstream; route the
    # zero accumulators through an array exactly like the donor code does.
    zero_acc = cutlass.Array(
        cutlass.Float32,
        SUPER_MMA_ACCUMULATORS_PER_LANE,
        alignment=16,
    )
    for accum_idx in cutlass.range_constexpr(SUPER_MMA_ACCUMULATORS_PER_LANE):
        zero_acc[accum_idx] = cutlass.Float32(0.0)

    for n_group_off in cutlass.range_constexpr(n_group_hi - n_group_lo):
        n_group = n_group_lo + n_group_off
        kr = super_mma_load_k_restore_a_fragment(
            tcgen05_k_restore_smem,
            lane,
            n_group,
        )
        b0 = movmatrix_b16(kr[0])
        b1 = movmatrix_b16(kr[1])
        b2 = movmatrix_b16(kr[2])
        b3 = movmatrix_b16(kr[3])
        n0_d0, n0_d1, n0_d2, n0_d3 = ptx_mma_m16n8k16_b16_f32(
            t0,
            t1,
            t2,
            t3,
            b0,
            b1,
            zero_acc[0],
            zero_acc[1],
            zero_acc[2],
            zero_acc[3],
            input_dtype,
        )
        n1_d0, n1_d1, n1_d2, n1_d3 = ptx_mma_m16n8k16_b16_f32(
            t0,
            t1,
            t2,
            t3,
            b2,
            b3,
            zero_acc[0],
            zero_acc[1],
            zero_acc[2],
            zero_acc[3],
            input_dtype,
        )
        prims.stmatrix(
            w_out_stmatrix_ptr(w_out_smem, n_group, lane),
            [
                pack_input_b16x2_to_i32(n0_d0, n0_d1, input_dtype),
                pack_input_b16x2_to_i32(n0_d2, n0_d3, input_dtype),
                pack_input_b16x2_to_i32(n1_d0, n1_d1, input_dtype),
                pack_input_b16x2_to_i32(n1_d2, n1_d3, input_dtype),
            ],
            prims.MMALayout.ROW,
            shape=prims.StoreShape.M8N8,
        )


@cute.kernel
def kernel_prep(
    tma_desc_q: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_k: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_gate: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_ws_kd: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_ws_qd: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_ws_w: cutlass.GridConstant[cuda.TensorMap],
    q: cute.Tensor,
    k: cute.Tensor,
    raw_gate: cute.Tensor,
    a_log: cute.Tensor,
    dt_bias: cute.Tensor,
    beta: cute.Tensor,
    cu_seqlens: cute.Tensor,
    cu_chunks: cute.Tensor,
    ws_kd: cute.Tensor,
    ws_qd: cute.Tensor,
    ws_w: cute.Tensor,
    ws_qk: cute.Tensor,
    ws_diag: cute.Tensor,
    chunks_per_cta: cutlass.Int32,
    head_base: cutlass.Int32,
    SAFE_GATE: cutlass.Constexpr,
    GATE_SCALE_LOG2: cutlass.Constexpr,
    gate_dtype: cutlass.Constexpr,
) -> None:
    """kernel 1: per-chunk-tile factor prep (4-warp CTA, no rings).

    Grid `(ceil(total_chunks / chunks_per_cta), heads, 1)`; each CTA walks a
    CONTIGUOUS range of chunks_per_cta chunks (K1_CPC), which keeps its TMA
    loads and workspace stores cache-local.
    """

    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    bidy = bidy + head_base
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = tidx % THREADS_PER_WARP

    # The workspace extent is a graph-static capacity.  The device-generated
    # prefix carries the actual count for the current replay.
    total_chunks = cutlass.Int32(cu_chunks[cu_chunks.shape[0] - 1])
    gchunk_stride = cutlass.Int32(1)
    chunk_lo = bidx * chunks_per_cta
    my_chunks = total_chunks - chunk_lo
    if my_chunks > chunks_per_cta:
        my_chunks = chunks_per_cta
    if my_chunks < 0:
        my_chunks = cutlass.Int32(0)
    input_dtype = q.element_type

    # 2 slots: chunk c completes tma_mbar[c % 2] (parity (c//2) % 2), so the
    # CPC>1 prefetch of chunk c+1 (issued mid-chunk by warp 2, see below)
    # never aliases the wait slot of the chunk in flight.
    tma_mbar = cutlass.Array(
        cutlass.Int64,
        2,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    cg0_k_ready_mbar = cutlass.Array(
        cutlass.Int64,
        1,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    cg0_k_half_ready_mbar = cutlass.Array(
        cutlass.Int64,
        1,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    q_k_restore_ready_mbar = cutlass.Array(
        cutlass.Int64,
        1,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    pairwise_ready_mbar = cutlass.Array(
        cutlass.Int64,
        1,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )

    tcgen05_k_decay_smem = cutlass.Array(
        input_dtype,
        TCGEN05_K_DECAY_STAGE_SIZE,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )
    tcgen05_q_decay_smem = cutlass.Array(
        input_dtype,
        TCGEN05_Q_DECAY_STAGE_SIZE,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )
    tcgen05_k_restore_smem = cutlass.Array(
        input_dtype,
        TCGEN05_K_RESTORE_STAGE_SIZE,
        space=cutlass.AddressSpace.smem,
        alignment=RAW_F16_TMA_SWIZZLE_ALIGNMENT_BYTES,
    )
    pairwise_smem = cutlass.Array(
        input_dtype,
        PAIRWISE_SMEM_STAGE_SIZE,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )
    raw_q_smem = cutlass.Array(
        q.element_type,
        RAW_Q_STAGE_SIZE,
        space=cutlass.AddressSpace.smem,
        alignment=RAW_F16_TMA_SWIZZLE_ALIGNMENT_BYTES,
    )
    raw_k_smem = cutlass.Array(
        k.element_type,
        RAW_K_STAGE_SIZE,
        space=cutlass.AddressSpace.smem,
        alignment=RAW_F16_TMA_SWIZZLE_ALIGNMENT_BYTES,
    )
    # Same policy as the engine, ring-free: k1's exchange is produced AND
    # consumed inside cg0_materialize_prep_operands by the same 4-warp CTA,
    # and the CPC>1 prefetch (which only overwrites raw_gate_smem) is issued
    # after q_k_restore_ready, so one stage suffices on the 16-bit path.
    if cutlass.const_expr(gate_dtype_is_f32(gate_dtype)):
        raw_gate_smem = cutlass.Array(
            cutlass.Float32,
            RAW_GATE_STAGE_SIZE,
            space=cutlass.AddressSpace.smem,
            alignment=RAW_F32_TMA_SWIZZLE_ALIGNMENT_BYTES,
        )
        gate_exchange_smem = raw_gate_smem
    else:
        raw_gate_smem = cutlass.Array(
            gate_dtype,
            RAW_GATE_STAGE_SIZE,
            space=cutlass.AddressSpace.smem,
            alignment=RAW_F16_TMA_SWIZZLE_ALIGNMENT_BYTES,
        )
        gate_exchange_smem = cutlass.Array(
            cutlass.Float32,
            GATE_EXCHANGE_STAGE_SIZE,
            space=cutlass.AddressSpace.smem,
            alignment=RAW_F32_TMA_SWIZZLE_ALIGNMENT_BYTES,
        )
    k_inv_smem = cutlass.Array(
        input_dtype,
        K_INV_STAGE_SIZE,
        space=cutlass.AddressSpace.smem,
        alignment=RAW_F16_TMA_SWIZZLE_ALIGNMENT_BYTES,
    )
    # Beta is double-buffered (2 x 64 B): the CPC>1 prefetch writes chunk
    # c+1's beta while chunk c's consumers still read theirs.
    raw_beta_smem = cutlass.Array(
        cutlass.Float32,
        2 * RAW_BETA_STAGE_SIZE,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )
    # W image staging (warps 1/3 stmatrix -> per-segment TMA bulk store).
    w_out_smem = cutlass.Array(
        input_dtype,
        TILE_ELEMS,
        space=cutlass.AddressSpace.smem,
        alignment=RAW_F16_TMA_SWIZZLE_ALIGNMENT_BYTES,
    )
    tma_tx_bytes = cutlass.const_expr(
        DK * BT * q.element_type.width // 8
        + DK * BT * k.element_type.width // 8
        + DK * BT * raw_gate.element_type.width // 8
    )

    if warp_idx == 0:
        if prims.elect_sync():
            prims.mbarrier_init(tma_mbar.subview(0), 1)
            prims.mbarrier_init(tma_mbar.subview(1), 1)
            prims.mbarrier_init(cg0_k_ready_mbar.subview(0), K1_WARPS)
            prims.mbarrier_init(cg0_k_half_ready_mbar.subview(0), K1_WARPS)
            prims.mbarrier_init(q_k_restore_ready_mbar.subview(0), K1_WARPS)
            prims.mbarrier_init(pairwise_ready_mbar.subview(0), 1)
    prims.fence_mbarrier_init()
    prep_cta_sync()

    # Per-CTA gate constants: direct global loads (one element per
    # (warp, lane) prefix column; no SMEM staging hop needed at this CTA
    # size).
    # Gate constants for BOTH the safe sigmoid and the FLA non-safe softplus
    # activation.  Loaded unconditionally; SAFE_GATE=True traces identically
    # (init-then-overwrite, byte-identical generated code) and the non-safe path reuses
    # the same constants.
    a_log_exp = cutlass.Float32(1.0)
    dt_bias_value = cutlass.Float32(0.0)
    prefix_dim = warp_idx * THREADS_PER_WARP + lane
    a_log_exp = cute.math.exp2(
        a_log[bidy].to(cutlass.Float32) * LOG2_E,
        fastmath=True,
    )
    dt_bias_value = dt_bias[bidy, prefix_dim].to(cutlass.Float32)

    for local_chunk in cutlass.range(my_chunks, unroll=1):
        gchunk = chunk_lo + local_chunk * gchunk_stride
        sequence_start, seqlen, chunk_start_tok = chunk_coords(
            cu_seqlens,
            cu_chunks,
            gchunk,
        )
        phase = local_chunk % 2
        beta_stage = local_chunk % 2
        tma_slot = local_chunk % 2
        tma_phase = (local_chunk // 2) % 2
        ws_row_start = gchunk * cutlass.Int32(BT)

        if warp_idx == 0:
            if local_chunk == 0:
                # First chunk load; chunks c+1 are PREFETCHED mid-chunk by
                # warp 2 (below), overlapping the DRAM round trip with the
                # A_inv/W/QK' phase of the chunk in flight.
                tma_prep_stage_load_inputs(
                    tma_desc_q,
                    tma_desc_k,
                    tma_desc_gate,
                    beta,
                    raw_q_smem,
                    raw_k_smem,
                    raw_gate_smem,
                    raw_beta_smem.subview(beta_stage * RAW_BETA_STAGE_SIZE),
                    sequence_start,
                    bidy,
                    lane,
                    chunk_start_tok,
                    seqlen,
                    tma_mbar.subview(tma_slot),
                    tma_tx_bytes,
                    gate_dtype,
                )
        tma_transfer_wait(tma_mbar.subview(tma_slot), tma_phase)

        if chunk_start_tok + cutlass.Int32(BT) > seqlen:
            if warp_idx == 0:
                # No raw V in kernel 1: pass raw_q twice (idempotent
                # double zero of the q tail rows).
                cg0_zero_tail_raw_operands(
                    raw_q_smem,
                    raw_k_smem,
                    raw_q_smem,
                    raw_gate_smem,
                    lane,
                    chunk_start_tok,
                    seqlen,
                    q.element_type,
                    gate_dtype,
                )
            prep_cta_sync()

        cg0_materialize_prep_operands(
            raw_q_smem,
            raw_k_smem,
            raw_gate_smem,
            gate_exchange_smem,
            a_log_exp,
            dt_bias_value,
            k_inv_smem,
            tcgen05_k_decay_smem,
            tcgen05_q_decay_smem,
            tcgen05_k_restore_smem,
            cg0_k_ready_mbar.subview(0),
            cg0_k_half_ready_mbar.subview(0),
            ws_diag,
            bidy,
            gchunk,
            chunk_start_tok,
            seqlen,
            input_dtype,
            gate_dtype,
            SAFE_GATE,
            GATE_SCALE_LOG2,
            warp_idx,
            lane,
        )
        q_k_restore_ready_arrive(q_k_restore_ready_mbar.subview(0))

        if warp_idx == 0:
            # KK -> L -> blockwise A_inv (starts on the half-DK arrival).
            cg0_k_ready_wait(cg0_k_half_ready_mbar.subview(0), phase)
            super_mma_stage_pairwise_pipeline(
                tcgen05_k_decay_smem,
                k_inv_smem,
                pairwise_smem,
                raw_beta_smem.subview(beta_stage * RAW_BETA_STAGE_SIZE),
                cg0_k_ready_mbar.subview(0),
                phase,
                lane,
                input_dtype,
            )
            pairwise_ready_arrive(pairwise_ready_mbar.subview(0))
        elif warp_idx == 1:
            q_k_restore_ready_wait(q_k_restore_ready_mbar.subview(0), phase)
            if cutlass.const_expr(K1_PREFETCH_WARP == 1):
                prefetch_next_chunk(
                    tma_desc_q,
                    tma_desc_k,
                    tma_desc_gate,
                    beta,
                    raw_q_smem,
                    raw_k_smem,
                    raw_gate_smem,
                    raw_beta_smem,
                    cu_seqlens,
                    cu_chunks,
                    local_chunk,
                    my_chunks,
                    gchunk,
                    gchunk_stride,
                    bidy,
                    lane,
                    tma_mbar,
                    tma_tx_bytes,
                    gate_dtype,
                )
            pairwise_ready_wait(pairwise_ready_mbar.subview(0), phase)
            super_mma_stage_w_half(
                tcgen05_k_restore_smem,
                pairwise_smem,
                raw_beta_smem.subview(beta_stage * RAW_BETA_STAGE_SIZE),
                w_out_smem,
                lane,
                input_dtype,
                0,
                K1_W_N_SPLIT,
            )
            prims.fence_proxy(
                prims.Proxy.ASYNC_SHARED,
                space=prims.SharedSpace.shared_cta,
            )
            prims.bar_warp_sync(cute.arch.FULL_MASK)
            if prims.elect_sync():
                w_coord = (
                    cutlass.Int32(0),
                    ws_row_start,
                    bidy,
                    cutlass.Int32(0),
                )
                prims.cp_async_bulk_tensor_global_shared_cta(
                    tma_desc_ws_w.get_ptr(),
                    w_out_smem.subview(0),
                    w_coord,
                )
                prims.cp_async_bulk_commit_group()
        elif warp_idx == 2:
            q_k_restore_ready_wait(q_k_restore_ready_mbar.subview(0), phase)
            if cutlass.const_expr(K1_PREFETCH_WARP == 2):
                prefetch_next_chunk(
                    tma_desc_q,
                    tma_desc_k,
                    tma_desc_gate,
                    beta,
                    raw_q_smem,
                    raw_k_smem,
                    raw_gate_smem,
                    raw_beta_smem,
                    cu_seqlens,
                    cu_chunks,
                    local_chunk,
                    my_chunks,
                    gchunk,
                    gchunk_stride,
                    bidy,
                    lane,
                    tma_mbar,
                    tma_tx_bytes,
                    gate_dtype,
                )
            pairwise_ready_wait(pairwise_ready_mbar.subview(0), phase)
            super_mma_stage_qk_prime(
                tcgen05_q_decay_smem,
                k_inv_smem,
                pairwise_smem,
                raw_beta_smem.subview(beta_stage * RAW_BETA_STAGE_SIZE),
                ws_qk,
                bidy,
                gchunk,
                lane,
                input_dtype,
            )
        else:
            # kd/qd workspace TMA stores as soon as both tiles are staged.
            cg0_k_ready_wait(cg0_k_ready_mbar.subview(0), phase)
            q_k_restore_ready_wait(q_k_restore_ready_mbar.subview(0), phase)
            if prims.elect_sync():
                for segment in cutlass.range_constexpr(RAW_F16_TMA_SEGMENTS):
                    ws_coord = (
                        cutlass.Int32(segment * RAW_F16_TMA_SWIZZLE_ELEMS),
                        ws_row_start,
                        bidy,
                        cutlass.Int32(0),
                    )
                    prims.cp_async_bulk_tensor_global_shared_cta(
                        tma_desc_ws_kd.get_ptr(),
                        tcgen05_k_decay_smem.subview(
                            segment * RAW_F16_TMA_SEGMENT_ELEMS
                        ),
                        ws_coord,
                    )
                    prims.cp_async_bulk_tensor_global_shared_cta(
                        tma_desc_ws_qd.get_ptr(),
                        tcgen05_q_decay_smem.subview(
                            segment * RAW_F16_TMA_SEGMENT_ELEMS
                        ),
                        ws_coord,
                    )
                prims.cp_async_bulk_commit_group()
            pairwise_ready_wait(pairwise_ready_mbar.subview(0), phase)
            super_mma_stage_w_half(
                tcgen05_k_restore_smem,
                pairwise_smem,
                raw_beta_smem.subview(beta_stage * RAW_BETA_STAGE_SIZE),
                w_out_smem,
                lane,
                input_dtype,
                K1_W_N_SPLIT,
                K1_W_N_GROUPS,
            )
            prims.fence_proxy(
                prims.Proxy.ASYNC_SHARED,
                space=prims.SharedSpace.shared_cta,
            )
            prims.bar_warp_sync(cute.arch.FULL_MASK)
            if prims.elect_sync():
                w_coord = (
                    cutlass.Int32(K1_W_N_SPLIT * 16),
                    ws_row_start,
                    bidy,
                    cutlass.Int32(0),
                )
                prims.cp_async_bulk_tensor_global_shared_cta(
                    tma_desc_ws_w.get_ptr(),
                    w_out_smem.subview(RAW_F16_TMA_SEGMENT_ELEMS),
                    w_coord,
                )
                prims.cp_async_bulk_commit_group()

        if (warp_idx == 1) | (warp_idx == 3):
            if prims.elect_sync():
                prims.cp_async_bulk_wait_group(0, read=True)
        prep_cta_sync()


# ---------------------------------------------------------------------------
# kernel-2 device helpers (value-correct forms taken from the engine path).
# ---------------------------------------------------------------------------


@cute.jit
def tma_chain_load_tile(
    tma_desc: cutlass.GridConstant[cuda.TensorMap],
    tile_smem,
    row_start,
    head_idx,
    tma_mbar,
) -> None:
    """Issue the two s128 segments of one [16, DK] bf16 workspace tile."""

    for segment in cutlass.range_constexpr(RAW_F16_TMA_SEGMENTS):
        tma_coord = (
            cutlass.Int32(segment * RAW_F16_TMA_SWIZZLE_ELEMS),
            row_start,
            head_idx,
            cutlass.Int32(0),
        )
        prims.cp_async_bulk_tensor_shared_cta_global(
            tile_smem.subview(segment * RAW_F16_TMA_SEGMENT_ELEMS),
            tma_desc.get_ptr(),
            tma_coord,
            tma_mbar,
        )


@cute.jit
def tcgen05_commit(mbar) -> None:
    """Commit the tensor pipe's progress to an mbarrier (single elected lane)."""

    if prims.elect_sync():
        prims.tcgen05_commit(mbar, group=prims.CTAGroup.CTA_1)


# --- chain state-input helpers --------------------------------------------


# =============================================================================
# kernel-2 DV split (M = 64, PTX Layout F).
#
# Everything column-indexed (TMEM layout, K blocks, B-operand SMEM
# descriptors) is IDENTICAL to kernel 2; only M changes.  Layout F places
# an M=64 tile's rows 16q..16q+15 at lanes 32q..32q+15 (q = warp quadrant,
# lane alignment 0), so:
#   - the state pack stays the shared `tcgen05_stage_state_input_tmem`
#     verbatim: it is lane-local (each lane converts its own row), valid
#     lanes map to valid lanes and the alignment-16 junk lanes feed junk
#     lanes no M=64 MMA reads;
#   - per-warp ROW-OWNING helpers (V ldmatrix stage, -X repack, qstate
#     drain, state ABI) drop from 32 to 16 rows per warp (one 16-lane
#     tcgen05 access instead of two);
#   - all mbar arrival counts are unchanged (all four warps of every
#     group still participate).
# =============================================================================


@cute.jit
def tcgen05_chain_issue_delta_half_mma(
    b_smem,
    tmem_raw_addr,
    a_input_col,
    input_dtype: cutlass.Constexpr,
    HALF: cutlass.Constexpr,
) -> None:
    """M=64 delta-half MMA issue."""

    half_n = DK // 2
    tmem_ptr = cutlass.inttoptr(
        tmem_raw_addr + KDA_TMEM_STATE_COL_OFFSET + HALF * half_n,
        6,
        cutlass.Float32,
    )
    desc_b = prims.Tcgen05SmemDesc.build(
        b_smem.subview(0),
        leading_byte_offset=TCGEN05_FINAL_STATE_B_LEADING_BYTES,
        stride_byte_offset=TCGEN05_FINAL_STATE_B_STRIDE_BYTES,
        layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
    )
    idesc = prims.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=input_dtype,
        b_dtype=input_dtype,
        n_dim=half_n,
        m_dim=DV_HALF,
        b_major=1,
    )
    half_byte_offset = HALF * TCGEN05_FINAL_STATE_B_LEADING_BYTES
    a_tmem = prims.make_tmem_ptr(tmem_raw_addr, cutlass.Int8).subview(a_input_col)
    if prims.elect_sync():
        prims.tcgen05_mma(
            prims.Tcgen05MMAKind.F16,
            prims.CTAGroup.CTA_1,
            tmem_ptr,
            a_tmem,
            desc_b.advance_start_address(half_byte_offset),
            idesc,
            True,
        )


@cute.jit
def tcgen05_chain_issue_qkv_mma(
    qk_stage_smem,
    tmem_raw_addr,
    a_input_col,
    qstate_acc_stage,
    acc_ready_mbar,
    input_dtype: cutlass.Constexpr,
    COMMIT: cutlass.Constexpr,
) -> None:
    """M=64 qkv MMA issue."""

    tmem_ptr = cutlass.inttoptr(
        tmem_raw_addr + tcgen05_qstate_acc_tmem_col_offset(qstate_acc_stage),
        6,
        cutlass.Float32,
    )
    idesc = prims.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=input_dtype,
        b_dtype=input_dtype,
        n_dim=BT,
        m_dim=DV_HALF,
        b_major=0,
    )
    lhs_tmem = prims.make_tmem_ptr(tmem_raw_addr, cutlass.Int8).subview(a_input_col)
    desc_pairwise = prims.Tcgen05SmemDesc.build(
        qk_stage_smem.subview(0),
        leading_byte_offset=TCGEN05_VALUE_PAIRWISE_B_LEADING_BYTES,
        stride_byte_offset=TCGEN05_VALUE_PAIRWISE_B_STRIDE_BYTES,
        layout=prims.Tcgen05SmemSwizzle.SWIZZLE_32B,
    )
    if prims.elect_sync():
        prims.tcgen05_mma(
            prims.Tcgen05MMAKind.F16,
            prims.CTAGroup.CTA_1,
            tmem_ptr,
            lhs_tmem,
            desc_pairwise,
            idesc,
            True,
        )
        if cutlass.const_expr(COMMIT):
            prims.tcgen05_commit(acc_ready_mbar, group=prims.CTAGroup.CTA_1)


@cute.jit
def tcgen05_chain_stage_vmx_input_tmem(
    tmem_raw_addr,
    raw_v_smem,
    warp_idx,
    lane,
    shared_acc_stage,
    shared_input_stage,
    input_dtype: cutlass.Constexpr,
) -> None:
    """M=64 (v - X) fused repack staging.

    ONE v ldmatrix ([16,64] half tile) + ONE 16x256b X
    fragment read + sub.b16x2 + ONE 16x128b store.
    """

    base_col_id = tmem_raw_addr & 0xFFFF
    base_row_id = tmem_raw_addr >> 16
    tmem_sp = warp_idx % TCGEN05_STATE_K_TMEM_ROW_BLOCKS

    projection_col_id = base_col_id + tcgen05_shared_acc_tmem_col_offset(
        shared_acc_stage
    )
    input_col_id = base_col_id + tcgen05_shared_input_tmem_col_offset(
        shared_input_stage
    )
    value_dim_base = tmem_sp * ROWS_PER_WARP

    row_id0 = base_row_id + tmem_sp * THREADS_PER_WARP
    block_ptr0 = cutlass.inttoptr(
        (row_id0 << 16) | projection_col_id, 6, cutlass.Float32
    )
    state_k0 = prims.tcgen05_ld("16x256b", block_ptr0, num=2)

    raw_v_regs0 = prims.ldmatrix(
        raw_v_ldmatrix_trans_ptr(raw_v_smem, value_dim_base, lane),
        4,
        prims.MMALayout.COL,
    )
    prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)

    packed0 = cutlass.Array(cutlass.Int32, 4, space=cutlass.AddressSpace.rmem)
    for reg_idx in cutlass.range_constexpr(4):
        raw_matrix: cutlass.Constexpr[int] = (1 - (reg_idx // 2)) * 2 + (reg_idx & 1)
        val0, val1 = tcgen05_rhs_token_pair_from_16x256b_fragment(
            state_k0,
            reg_idx,
        )
        packed0[reg_idx] = sub_b16x2_input_dtype(
            raw_v_regs0[raw_matrix],
            pack_input_b16x2_to_i32(val0, val1, input_dtype),
            input_dtype,
        )

    input_block_addr0 = (base_row_id << 16) | input_col_id
    input_block_ptr0 = prims.make_tmem_ptr(input_block_addr0, cutlass.Int8)
    prims.tcgen05_st("16x128b", input_block_ptr0, packed0[0:4])
    prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)


@cute.jit
def tcgen05_chain_load_qstate_output_tmem(
    tmem_raw_addr,
    o_smem,
    warp_idx,
    lane,
    o_stage_base,
    qstate_acc_stage,
    scale: cutlass.Float32,
    output_dtype: cutlass.Constexpr,
) -> None:
    """M=64 variant of tcgen05_load_qstate_output_tmem (16 rows per warp)."""

    base_col_id = tmem_raw_addr & 0xFFFF
    base_row_id = tmem_raw_addr >> 16
    tmem_sp = warp_idx % TCGEN05_STATE_K_TMEM_ROW_BLOCKS

    projection_col_id = base_col_id + tcgen05_qstate_acc_tmem_col_offset(
        qstate_acc_stage
    )
    value_dim_base = tmem_sp * ROWS_PER_WARP

    row_id0 = base_row_id + tmem_sp * THREADS_PER_WARP
    block_addr0 = (row_id0 << 16) | projection_col_id
    block_ptr0 = cutlass.inttoptr(block_addr0, 6, cutlass.Float32)
    loaded0 = prims.tcgen05_ld(
        "16x256b",
        block_ptr0,
        num=2,
    )
    prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)

    stsm_regs0 = cutlass.Array(
        cutlass.Int32,
        4,
        space=cutlass.AddressSpace.rmem,
    )
    for reg_idx in cutlass.range_constexpr(4):
        scaled0_0, scaled0_1 = fmul2(
            (loaded0[2 * reg_idx], loaded0[2 * reg_idx + 1]),
            (scale, scale),
        )
        stsm_regs0[reg_idx] = pack_output_b16x2_to_i32(
            scaled0_0,
            scaled0_1,
            output_dtype,
        )

    smem_dst0 = o_smem_stmatrix_128b_ptr(
        o_smem,
        o_stage_base,
        value_dim_base,
        lane,
    )
    prims.stmatrix(
        smem_dst0,
        stsm_regs0.data_ptr().load(count=4, alignment=4),
        prims.MMALayout.COL,
        shape=prims.StoreShape.M8N8,
    )
    cute.arch.fence_view_async_shared()


@cute.jit
def tma_chain_stage_load_inputs(
    tma_desc_kd: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_w: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_qd: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_v: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_diag: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_qk: cutlass.GridConstant[cuda.TensorMap],
    kd_stage,
    w_stage,
    qd_stage,
    v_stage,
    diag_stage,
    qk_stage,
    head_idx,
    dv_half,
    ws_row_start,
    ws_chunk,
    v_row_start,
    tma_mbar,
    tma_tx_bytes: cutlass.Constexpr,
) -> None:
    """DV2 chunk operand transaction: full kd/w/qd, HALF v (one segment)."""

    if prims.elect_sync():
        prims.mbarrier_arrive_expect_tx(tma_mbar, tma_tx_bytes)
    if prims.elect_sync():
        tma_chain_load_tile(tma_desc_kd, kd_stage, ws_row_start, head_idx, tma_mbar)
        tma_chain_load_tile(tma_desc_w, w_stage, ws_row_start, head_idx, tma_mbar)
        tma_chain_load_tile(tma_desc_qd, qd_stage, ws_row_start, head_idx, tma_mbar)
        v_coord = (
            dv_half * cutlass.Int32(DV_HALF),
            v_row_start,
            head_idx,
            cutlass.Int32(0),
        )
        prims.cp_async_bulk_tensor_shared_cta_global(
            v_stage.subview(0),
            tma_desc_v.get_ptr(),
            v_coord,
            tma_mbar,
        )
        diag_coord = (
            cutlass.Int32(0),
            ws_chunk,
            head_idx,
            cutlass.Int32(0),
        )
        prims.cp_async_bulk_tensor_shared_cta_global(
            diag_stage.subview(0),
            tma_desc_diag.get_ptr(),
            diag_coord,
            tma_mbar,
        )
        prims.cp_async_bulk_tensor_shared_cta_global(
            qk_stage.subview(0),
            tma_desc_qk.get_ptr(),
            diag_coord,
            tma_mbar,
        )


@cute.jit
def epilogue_chain_stage_store(
    tma_desc_o: cutlass.GridConstant[cuda.TensorMap],
    o_smem,
    sequence_start,
    head_idx,
    dv_half,
    chunk_start,
    o_stage_base,
) -> None:
    """Store the staged `[BT, 64]` half-output tile (one s128 segment)."""

    global_chunk_start = sequence_start + chunk_start
    if prims.elect_sync():
        o_coord = (
            dv_half * cutlass.Int32(DV_HALF),
            global_chunk_start,
            head_idx,
            cutlass.Int32(0),
        )
        prims.cp_async_bulk_tensor_global_shared_cta(
            tma_desc_o.get_ptr(),
            o_smem.subview(o_stage_base + O_OUT_OFFSET),
            o_coord,
        )
        prims.cp_async_bulk_commit_group()
        prims.cp_async_bulk_wait_group(0, read=True)
    prims.bar_warp_sync(cute.arch.FULL_MASK)


@cute.jit
def epilogue_chain_tail_store(
    out,
    o_smem,
    sequence_start,
    head_idx,
    dv_half,
    chunk_start,
    seqlen,
    o_stage_base,
    lane,
) -> None:
    """Store a partial packed-sequence half-tail without crossing its boundary."""

    valid_tokens = seqlen - chunk_start
    value_base = dv_half * cutlass.Int32(DV_HALF)
    for elem_iter in cutlass.range_constexpr((BT * DV_HALF) // THREADS_PER_WARP):
        linear_idx = elem_iter * THREADS_PER_WARP + lane
        token_coord = linear_idx // DV_HALF
        value_dim = linear_idx - token_coord * DV_HALF
        if token_coord < valid_tokens:
            smem_idx = o_smem_swizzle_128b_elem_index(
                o_stage_base,
                value_dim,
                token_coord,
            )
            out[
                0,
                sequence_start + chunk_start + token_coord,
                head_idx,
                value_base + value_dim,
            ] = o_smem[smem_idx]
    prims.bar_warp_sync(cute.arch.FULL_MASK)


@cute.jit
def epilogue_chain_wait_and_store_full_output(
    tma_desc_o: cutlass.GridConstant[cuda.TensorMap],
    o_smem,
    output_ready_mbar,
    output_consumed_mbar,
    sequence_start,
    head_idx,
    dv_half,
    output_chunk,
    O_STAGES: cutlass.Constexpr,
):
    """Drain one full staged half-output chunk from SMEM with TMA."""

    output_chunk_start = output_chunk * BT
    o_stage = output_chunk % O_STAGES
    o_stage_base = o_stage * K2_O_SMEM_STAGE_SIZE
    output_ready_wait(
        output_ready_mbar.subview(o_stage),
        (output_chunk // O_STAGES) % 2,
    )
    epilogue_chain_stage_store(
        tma_desc_o,
        o_smem,
        sequence_start,
        head_idx,
        dv_half,
        output_chunk_start,
        o_stage_base,
    )
    output_consumed_arrive(output_consumed_mbar.subview(o_stage))


@cute.jit
def epilogue_chain_wait_and_store_final_output(
    tma_desc_o: cutlass.GridConstant[cuda.TensorMap],
    out,
    o_smem,
    output_ready_mbar,
    output_consumed_mbar,
    sequence_start,
    head_idx,
    dv_half,
    seqlen,
    output_chunk,
    lane,
    O_STAGES: cutlass.Constexpr,
):
    """Drain the final half-output chunk, guarding a partial packed tail."""

    if seqlen % BT == 0:
        epilogue_chain_wait_and_store_full_output(
            tma_desc_o,
            o_smem,
            output_ready_mbar,
            output_consumed_mbar,
            sequence_start,
            head_idx,
            dv_half,
            output_chunk,
            O_STAGES,
        )
    else:
        output_chunk_start = output_chunk * BT
        o_stage = output_chunk % O_STAGES
        o_stage_base = o_stage * K2_O_SMEM_STAGE_SIZE
        output_ready_wait(
            output_ready_mbar.subview(o_stage),
            (output_chunk // O_STAGES) % 2,
        )
        epilogue_chain_tail_store(
            out,
            o_smem,
            sequence_start,
            head_idx,
            dv_half,
            output_chunk_start,
            seqlen,
            o_stage_base,
            lane,
        )
        output_consumed_arrive(output_consumed_mbar.subview(o_stage))


@cute.kernel
def kernel_chain_dv2(
    tma_desc_kd: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_w: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_qd: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_v: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_diag: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_qk: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_o: cutlass.GridConstant[cuda.TensorMap],
    v: cute.Tensor,
    cu_seqlens: cute.Tensor,
    cu_chunks: cute.Tensor,
    state_indices: cute.Tensor | None,
    initial_state: cute.Tensor | None,
    out: cute.Tensor,
    final_state: cute.Tensor | None,
    head_base: cutlass.Int32,
    SCALE: cutlass.Float32,
    state_ckpt: cute.Tensor | None,
    cu_ckpts: cute.Tensor | None,
    checkpoint_state_indices: cute.Tensor | None,
    checkpoint_stride_chunks: cutlass.Int32,
) -> None:
    """kernel 2, DV-split: each CTA owns half the hidden dimension.

    Grid `(num_sequences, launch_heads * 2, 1)`; bidy = head * 2 + half.
    Identical schedule and mbar topology to the base chain with M=64
    MMAs (PTX Layout F: 16 rows per warp quadrant, lane alignment 0);
    kd/W/QK'/diag are read identically by both halves, v/o/state are
    DV-split (one 64-elem s128 TMA segment at value offset half*64).
    """

    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    dv_half = bidy % 2
    bidy = head_base + bidy // 2
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = tidx % THREADS_PER_WARP

    sequence_start = cutlass.Int32(cu_seqlens[bidx])
    sequence_end = cutlass.Int32(cu_seqlens[bidx + 1])
    seqlen = sequence_end - sequence_start
    num_chunks = cute.ceil_div(seqlen, BT)
    chunk_base = cutlass.Int32(cu_chunks[bidx])
    input_dtype = v.element_type

    # --- mbarriers (identical topology to the base chain) ----------------
    tma_mbar = cutlass.Array(
        cutlass.Int64,
        K2_TMA_MBAR_STAGE_COUNT,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    raw_ready_mbar = cutlass.Array(
        cutlass.Int64,
        K2_RAW_STAGE_COUNT,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    raw_consumed_mbar = cutlass.Array(
        cutlass.Int64,
        K2_RAW_STAGE_COUNT,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    state_input_ready_l_mbar = cutlass.Array(
        cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8
    )
    state_input_ready_mbar = cutlass.Array(
        cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8
    )
    u_input_ready_mbar = cutlass.Array(
        cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8
    )
    update_ready_mbar = cutlass.Array(
        cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8
    )
    shared_acc_ready_mbar = cutlass.Array(
        cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8
    )
    k_restore_consumed_l_mbar = cutlass.Array(
        cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8
    )
    k_restore_consumed_mbar = cutlass.Array(
        cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8
    )
    qstate_acc_ready_mbar = cutlass.Array(
        cutlass.Int64,
        K2_QSTATE_STAGE_COUNT,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    stateq_done_mbar = cutlass.Array(
        cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8
    )
    output_ready_mbar = cutlass.Array(
        cutlass.Int64,
        K2_QSTATE_STAGE_COUNT,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    output_consumed_mbar = cutlass.Array(
        cutlass.Int64,
        K2_QSTATE_STAGE_COUNT,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    final_state_stored_mbar = cutlass.Array(
        cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8
    )
    checkpoint_read_done_mbar = cutlass.Array(
        cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8
    )
    tmem_ptr_i32 = cutlass.Array(
        cutlass.Int32, 1, space=cutlass.AddressSpace.smem, alignment=4
    )

    # --- SMEM rings (v halved to one segment; o halved) --------------------
    kd_smem = cutlass.Array(
        input_dtype,
        K2_RAW_STAGE_COUNT * TILE_ELEMS,
        space=cutlass.AddressSpace.smem,
        alignment=RAW_F16_TMA_SWIZZLE_ALIGNMENT_BYTES,
    )
    w_smem = cutlass.Array(
        input_dtype,
        K2_RAW_STAGE_COUNT * TILE_ELEMS,
        space=cutlass.AddressSpace.smem,
        alignment=RAW_F16_TMA_SWIZZLE_ALIGNMENT_BYTES,
    )
    qd_smem = cutlass.Array(
        input_dtype,
        K2_RAW_STAGE_COUNT * TILE_ELEMS,
        space=cutlass.AddressSpace.smem,
        alignment=RAW_F16_TMA_SWIZZLE_ALIGNMENT_BYTES,
    )
    v_raw_smem = cutlass.Array(
        input_dtype,
        K2_RAW_STAGE_COUNT * V_TILE_ELEMS,
        space=cutlass.AddressSpace.smem,
        alignment=RAW_F16_TMA_SWIZZLE_ALIGNMENT_BYTES,
    )
    diag_raw_smem = cutlass.Array(
        cutlass.Float32,
        K2_RAW_STAGE_COUNT * DIAG_REC_ELEMS,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )
    qk_smem = cutlass.Array(
        input_dtype,
        K2_RAW_STAGE_COUNT * QK_REC_ELEMS,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )
    o_smem = cutlass.Array(
        out.element_type,
        K2_O_SMEM_TILE_SIZE,
        space=cutlass.AddressSpace.smem,
        alignment=O_TMA_SWIZZLE_ALIGNMENT_BYTES,
    )

    # --- init (identical to the base chain) ------------------------------
    if warp_idx == ROLES.tma_load:
        if prims.elect_sync():
            for stage in cutlass.range_constexpr(K2_TMA_MBAR_STAGE_COUNT):
                prims.mbarrier_init(tma_mbar.subview(stage), 1)
            for stage in cutlass.range_constexpr(K2_RAW_STAGE_COUNT):
                prims.mbarrier_init(raw_ready_mbar.subview(stage), 1)
                prims.mbarrier_init(
                    raw_consumed_mbar.subview(stage),
                    5,
                )
    elif warp_idx == ROLES.tcgen05_mma:
        if prims.elect_sync():
            prims.mbarrier_init(state_input_ready_l_mbar, 4)
            prims.mbarrier_init(state_input_ready_mbar, 4)
            for stage in cutlass.range_constexpr(2):
                prims.mbarrier_init(u_input_ready_mbar.subview(stage), 4)
                prims.mbarrier_init(shared_acc_ready_mbar.subview(stage), 1)
                prims.mbarrier_init(k_restore_consumed_l_mbar.subview(stage), 1)
                prims.mbarrier_init(k_restore_consumed_mbar.subview(stage), 1)
                prims.mbarrier_init(stateq_done_mbar.subview(stage), 1)
            # CG1 fuses the right-half pack/scale while idle CG0 warps 4-7
            # fuse the left half. Both groups must finish before delta.
            prims.mbarrier_init(update_ready_mbar, 8)
            for stage in cutlass.range_constexpr(K2_QSTATE_STAGE_COUNT):
                prims.mbarrier_init(qstate_acc_ready_mbar.subview(stage), 1)
            prims.mbarrier_init(final_state_stored_mbar, 8)
            if cutlass.const_expr(state_ckpt is not None):
                prims.mbarrier_init(checkpoint_read_done_mbar, 4)
    elif warp_idx == ROLES.epilogue:
        if prims.elect_sync():
            for stage in cutlass.range_constexpr(K2_QSTATE_STAGE_COUNT):
                prims.mbarrier_init(output_ready_mbar.subview(stage), 4)
                prims.mbarrier_init(output_consumed_mbar.subview(stage), 1)
    prims.fence_mbarrier_init()
    cta_sync()

    if is_tmem_user_warp(warp_idx):
        if warp_idx == ROLES.tcgen05_mma:
            prims.tcgen05_alloc(tmem_ptr_i32, TMEM_ALLOC_COLS, group="cta_1")
        tmem_user_sync()
        if warp_idx == ROLES.tcgen05_mma:
            prims.tcgen05_relinquish_alloc_permit(group="cta_1")
        tmem_user_sync()
    cta_sync()

    if is_service_warpgroup(warp_idx):
        prims.setmaxregister(KDA_SERVICE_REGS, prims.SetMaxRegisterAction.DECREASE)

    # ======================= warp 14: TMA ring =========================
    if warp_idx == ROLES.tma_load:
        raw_stage = cutlass.Int32(0)
        raw_consumed_phase = cutlass.Int32(1)
        issue_mbar_slot = cutlass.Int32(0)
        wait_mbar_slot = cutlass.Int32(0)
        ready_stage = cutlass.Int32(0)
        tma_phase = cutlass.Int32(0)
        for chunk in cutlass.range(num_chunks, unroll=1):
            ws_chunk = chunk_base + chunk
            ws_row_start = ws_chunk * cutlass.Int32(BT)
            v_row_start = sequence_start + chunk * cutlass.Int32(BT)
            raw_consumed_wait(
                raw_consumed_mbar.subview(raw_stage),
                raw_consumed_phase,
            )
            tma_chain_stage_load_inputs(
                tma_desc_kd,
                tma_desc_w,
                tma_desc_qd,
                tma_desc_v,
                tma_desc_diag,
                tma_desc_qk,
                kd_smem.subview(raw_stage * TILE_ELEMS),
                w_smem.subview(raw_stage * TILE_ELEMS),
                qd_smem.subview(raw_stage * TILE_ELEMS),
                v_raw_smem.subview(raw_stage * V_TILE_ELEMS),
                diag_raw_smem.subview(raw_stage * DIAG_REC_ELEMS),
                qk_smem.subview(raw_stage * QK_REC_ELEMS),
                bidy,
                dv_half,
                ws_row_start,
                ws_chunk,
                v_row_start,
                tma_mbar.subview(issue_mbar_slot),
                K2_TX_BYTES,
            )
            issue_mbar_slot, _ = advance_ring_stage(
                issue_mbar_slot, 1, K2_TMA_MBAR_STAGE_COUNT
            )
            if chunk >= cutlass.Int32(K2_TMA_MBAR_STAGE_COUNT - 1):
                tma_transfer_wait(tma_mbar.subview(wait_mbar_slot), tma_phase)
                wait_mbar_slot, wait_wrapped = advance_ring_stage(
                    wait_mbar_slot, 1, K2_TMA_MBAR_STAGE_COUNT
                )
                tma_phase = tma_phase ^ wait_wrapped
                raw_ready_arrive(raw_ready_mbar.subview(ready_stage))
                ready_stage, _ = advance_ring_stage(ready_stage, 1, K2_RAW_STAGE_COUNT)
            raw_stage, raw_wrapped = advance_ring_stage(
                raw_stage, 1, K2_RAW_STAGE_COUNT
            )
            raw_consumed_phase = raw_consumed_phase ^ raw_wrapped
        tma_full = cutlass.Int32(
            num_chunks >= cutlass.Int32(K2_TMA_MBAR_STAGE_COUNT - 1)
        )
        tma_tail = (
            tma_full * cutlass.Int32(K2_TMA_MBAR_STAGE_COUNT - 1)
            + (cutlass.Int32(1) - tma_full) * num_chunks
        )
        for _tail in cutlass.range(tma_tail, unroll=1):
            tma_transfer_wait(tma_mbar.subview(wait_mbar_slot), tma_phase)
            wait_mbar_slot, wait_wrapped = advance_ring_stage(
                wait_mbar_slot, 1, K2_TMA_MBAR_STAGE_COUNT
            )
            tma_phase = tma_phase ^ wait_wrapped
            raw_ready_arrive(raw_ready_mbar.subview(ready_stage))
            ready_stage, _ = advance_ring_stage(ready_stage, 1, K2_RAW_STAGE_COUNT)

    # ====== warp 12: output-group issuer (OUT_ISSUER12) / flag relay ======
    elif warp_idx == ROLES.super_mma:
        tmem_raw_addr = tmem_ptr_i32.load()
        si_phase12 = cutlass.Int32(0)
        upd_phase12 = cutlass.Int32(0)
        for chunk in cutlass.range(num_chunks, unroll=1):
            raw_stage = chunk % K2_RAW_STAGE_COUNT
            qstate_stage = chunk % K2_QSTATE_STAGE_COUNT
            kr_stage = chunk % 2
            acc_stage = chunk % 2
            if chunk > 0:
                prev12 = chunk - cutlass.Int32(1)
                tcgen05_wait_acc_buffer_ready(
                    qstate_acc_ready_mbar.subview(prev12 % K2_QSTATE_STAGE_COUNT),
                    (prev12 // K2_QSTATE_STAGE_COUNT) % 2,
                )
                raw_consumed_arrive(
                    raw_consumed_mbar.subview(prev12 % K2_RAW_STAGE_COUNT)
                )
            raw_ready_wait(
                raw_ready_mbar.subview(raw_stage),
                (chunk // K2_RAW_STAGE_COUNT) % 2,
            )
            si_phase12 = state_input_ready_wait(state_input_ready_mbar, si_phase12)
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            output_ready_wait(
                output_ready_mbar.subview(qstate_stage),
                (chunk // K2_QSTATE_STAGE_COUNT + cutlass.Int32(1)) % 2,
            )
            tcgen05_issue_state_projection_mma(
                qd_smem.subview(raw_stage * TILE_ELEMS),
                tmem_raw_addr,
                stateq_done_mbar.subview(kr_stage),
                tcgen05_qstate_acc_tmem_col_offset(qstate_stage),
                input_dtype,
                0,
                DK // TCGEN05_F16_K_ATOM,
                False,
                True,
                DV_HALF,
            )
            upd_phase12 = update_ready_wait(update_ready_mbar, upd_phase12)
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            xpack_col12 = tcgen05_shared_input_tmem_col_offset(acc_stage)
            tcgen05_chain_issue_qkv_mma(
                qk_smem.subview(raw_stage * QK_REC_ELEMS),
                tmem_raw_addr,
                xpack_col12,
                qstate_stage,
                qstate_acc_ready_mbar.subview(qstate_stage),
                input_dtype,
                True,
            )

    # =============== warp 13: tcgen05 issuer (the chain owner) =============
    elif warp_idx == ROLES.tcgen05_mma:
        tmem_raw_addr = tmem_ptr_i32.load()
        si_l_phase = cutlass.Int32(0)
        si_phase = cutlass.Int32(0)
        upd_phase = cutlass.Int32(0)
        for chunk in cutlass.range(num_chunks, unroll=1):
            raw_stage = chunk % K2_RAW_STAGE_COUNT
            raw_phase = (chunk // K2_RAW_STAGE_COUNT) % 2
            acc_stage = chunk % 2
            qstate_stage = chunk % K2_QSTATE_STAGE_COUNT
            kr_stage = chunk % 2
            kd_stage_smem = kd_smem.subview(raw_stage * TILE_ELEMS)
            w_stage_smem = w_smem.subview(raw_stage * TILE_ELEMS)

            raw_ready_wait(raw_ready_mbar.subview(raw_stage), raw_phase)

            si_l_phase = state_input_ready_wait(state_input_ready_l_mbar, si_l_phase)
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            tcgen05_issue_state_k_mma(
                kd_stage_smem,
                tmem_raw_addr,
                shared_acc_ready_mbar.subview(acc_stage),
                acc_stage,
                input_dtype,
                0,
                (DK // TCGEN05_F16_K_ATOM) // 2,
                False,
                False,
                DV_HALF,
            )
            si_phase = state_input_ready_wait(state_input_ready_mbar, si_phase)
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            tcgen05_issue_state_k_mma(
                kd_stage_smem,
                tmem_raw_addr,
                shared_acc_ready_mbar.subview(acc_stage),
                acc_stage,
                input_dtype,
                (DK // TCGEN05_F16_K_ATOM) // 2,
                DK // TCGEN05_F16_K_ATOM,
                True,
                True,
                DV_HALF,
            )

            xpack_col = tcgen05_shared_input_tmem_col_offset(acc_stage)
            upd_phase = update_ready_wait(update_ready_mbar, upd_phase)
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            tcgen05_chain_issue_delta_half_mma(
                w_stage_smem, tmem_raw_addr, xpack_col, input_dtype, 0
            )
            tcgen05_commit(k_restore_consumed_l_mbar.subview(kr_stage))
            tcgen05_chain_issue_delta_half_mma(
                w_stage_smem, tmem_raw_addr, xpack_col, input_dtype, 1
            )
            tcgen05_commit(k_restore_consumed_mbar.subview(kr_stage))

        final_state_stored_wait(final_state_stored_mbar, cutlass.Int32(0))
        tmem_ptr = cutlass.inttoptr(tmem_raw_addr, 6, cutlass.Float32)
        prims.tcgen05_dealloc(tmem_ptr, TMEM_ALLOC_COLS, group="cta_1")

    # ======================= warp 15: output store =========================
    elif warp_idx == ROLES.epilogue:
        for chunk in cutlass.range(num_chunks, unroll=1):
            if chunk > 0:
                epilogue_chain_wait_and_store_full_output(
                    tma_desc_o,
                    o_smem,
                    output_ready_mbar,
                    output_consumed_mbar,
                    sequence_start,
                    bidy,
                    dv_half,
                    chunk - cutlass.Int32(1),
                    K2_QSTATE_STAGE_COUNT,
                )
        if num_chunks > 0:
            epilogue_chain_wait_and_store_final_output(
                tma_desc_o,
                out,
                o_smem,
                output_ready_mbar,
                output_consumed_mbar,
                sequence_start,
                bidy,
                dv_half,
                seqlen,
                num_chunks - cutlass.Int32(1),
                lane,
                K2_QSTATE_STAGE_COUNT,
            )

    # ================= CG1 (warps 8-11): TMEM epilogues ====================
    elif is_compute_group1_warp(warp_idx):
        prims.setmaxregister(KDA_CG1_REGS, prims.SetMaxRegisterAction.INCREASE)
        tmem_raw_addr = tmem_ptr_i32.load()
        # Only CG1 touches recurrent state. Keep the pool lookup outside the
        # chunk loop and out of producer/MMAs warps.
        state_slot = bidx
        if cutlass.const_expr(state_indices is not None):
            state_slot = cutlass.Int32(state_indices[bidx])
        ckpt_slot = cutlass.Int32(0)
        if cutlass.const_expr(state_ckpt is not None):
            # cu_ckpts supplies the per-sequence base slot offsets.  A
            # loop-carried counter replaces a per-chunk div/mod.
            ckpt_stride = checkpoint_stride_chunks
            ckpt_next = ckpt_stride
            ckpt_slot = cutlass.Int32(cu_ckpts[bidx])
        if cutlass.const_expr(checkpoint_state_indices is not None):
            tcgen05_store_initial_state_tmem(
                tmem_raw_addr,
                initial_state,
                None,
                cutlass.Int32(0),
                state_slot,
                bidy,
                dv_half,
                warp_idx,
                lane,
                HALF=True,
            )
        else:
            tcgen05_store_initial_state_tmem(
                tmem_raw_addr,
                initial_state,
                state_ckpt,
                ckpt_slot,
                state_slot,
                bidy,
                dv_half,
                warp_idx,
                lane,
                HALF=True,
            )
        if cutlass.const_expr(
            state_ckpt is not None and checkpoint_state_indices is None
        ):
            ckpt_slot += cutlass.Int32(1)
        if num_chunks > 0:
            diag_raw_stage = diag_raw_smem.subview(0)
            state_left = tcgen05_stage_state_input_dv2_half_tmem(
                tmem_raw_addr,
                warp_idx,
                input_dtype,
                0,
            )
            raw_ready_wait(raw_ready_mbar.subview(0), 0)
            tcgen05_rescale_state_dv2_half_regs(
                tmem_raw_addr,
                diag_raw_stage,
                warp_idx,
                state_left,
                state_input_ready_l_mbar,
                0,
            )
            state_right = tcgen05_stage_state_input_dv2_half_tmem(
                tmem_raw_addr,
                warp_idx,
                input_dtype,
                1,
            )
            tcgen05_rescale_state_dv2_half_regs(
                tmem_raw_addr,
                diag_raw_stage,
                warp_idx,
                state_right,
                state_input_ready_mbar,
                1,
            )
            tcgen05_wait_acc_buffer_ready(shared_acc_ready_mbar.subview(0), 0)
            vmx_lane = cute.arch.lane_idx()
            tcgen05_chain_stage_vmx_input_tmem(
                tmem_raw_addr,
                v_raw_smem.subview(0),
                warp_idx,
                vmx_lane,
                0,
                0,
                input_dtype,
            )
            post_scale_lane = cute.arch.lane_idx()
            update_ready_arrive(update_ready_mbar)
            if cutlass.const_expr(state_ckpt is not None):
                # Peeled chunk 0's checkpoint (stride 1 only).
                checkpoint_due = ckpt_next == cutlass.Int32(1)
                if cutlass.const_expr(checkpoint_state_indices is not None):
                    checkpoint_due = checkpoint_due & (cutlass.Int32(BT) <= seqlen)
                else:
                    checkpoint_due = checkpoint_due & (cutlass.Int32(1) < num_chunks)
                if checkpoint_due:
                    tcgen05_wait_acc_buffer_ready(
                        k_restore_consumed_l_mbar.subview(0), 0
                    )
                    tcgen05_wait_acc_buffer_ready(k_restore_consumed_mbar.subview(0), 0)
                    checkpoint_output_slot = ckpt_slot
                    if cutlass.const_expr(checkpoint_state_indices is not None):
                        checkpoint_output_slot = cutlass.Int32(
                            checkpoint_state_indices[ckpt_slot]
                        )
                    if (checkpoint_output_slot >= cutlass.Int32(0)) & (
                        checkpoint_output_slot < cutlass.Int32(state_ckpt.shape[0])
                    ):
                        tcgen05_store_final_state_tmem(
                            tmem_raw_addr,
                            KDA_TMEM_STATE_COL_OFFSET,
                            state_ckpt,
                            checkpoint_output_slot,
                            bidy,
                            dv_half,
                            warp_idx,
                            post_scale_lane,
                            HALF=True,
                        )
                    checkpoint_read_done_arrive(checkpoint_read_done_mbar)
                    ckpt_slot += cutlass.Int32(1)
                    ckpt_next += ckpt_stride
        for chunk in cutlass.range(1, num_chunks, 1, unroll=1):
            prev = chunk - cutlass.Int32(1)
            raw_stage = chunk % K2_RAW_STAGE_COUNT
            diag_raw_stage = diag_raw_smem.subview(raw_stage * DIAG_REC_ELEMS)
            acc_stage = chunk % 2
            prev_kr_phase = (prev // 2) % 2
            tcgen05_wait_acc_buffer_ready(
                stateq_done_mbar.subview(prev % 2), prev_kr_phase
            )
            tcgen05_wait_acc_buffer_ready(
                k_restore_consumed_mbar.subview(prev % 2), prev_kr_phase
            )
            state_right = tcgen05_stage_state_input_dv2_half_tmem(
                tmem_raw_addr,
                warp_idx,
                input_dtype,
                1,
            )
            raw_consumed_arrive(raw_consumed_mbar.subview(prev % K2_RAW_STAGE_COUNT))
            raw_ready_wait(
                raw_ready_mbar.subview(raw_stage),
                (chunk // K2_RAW_STAGE_COUNT) % 2,
            )
            tcgen05_rescale_state_dv2_half_regs(
                tmem_raw_addr,
                diag_raw_stage,
                warp_idx,
                state_right,
                state_input_ready_mbar,
                1,
            )
            tcgen05_wait_acc_buffer_ready(
                shared_acc_ready_mbar.subview(acc_stage), (chunk // 2) % 2
            )
            vmx_lane = cute.arch.lane_idx()
            tcgen05_chain_stage_vmx_input_tmem(
                tmem_raw_addr,
                v_raw_smem.subview(raw_stage * V_TILE_ELEMS),
                warp_idx,
                vmx_lane,
                acc_stage,
                acc_stage,
                input_dtype,
            )
            post_scale_lane = cute.arch.lane_idx()
            update_ready_arrive(update_ready_mbar)
            if cutlass.const_expr(state_ckpt is not None):
                # Same contract as the engine-side checkpoint; both M=64 TMEM
                # halves wait their own k_restore parity slot (the same wait
                # the post-loop final store performs) before draining.
                checkpoint_due = chunk + cutlass.Int32(1) == ckpt_next
                if cutlass.const_expr(checkpoint_state_indices is not None):
                    checkpoint_due = checkpoint_due & (
                        (chunk + cutlass.Int32(1)) * cutlass.Int32(BT) <= seqlen
                    )
                else:
                    checkpoint_due = checkpoint_due & (
                        chunk + cutlass.Int32(1) < num_chunks
                    )
                if checkpoint_due:
                    tcgen05_wait_acc_buffer_ready(
                        k_restore_consumed_l_mbar.subview(chunk % 2),
                        (chunk // 2) % 2,
                    )
                    tcgen05_wait_acc_buffer_ready(
                        k_restore_consumed_mbar.subview(chunk % 2),
                        (chunk // 2) % 2,
                    )
                    checkpoint_output_slot = ckpt_slot
                    if cutlass.const_expr(checkpoint_state_indices is not None):
                        checkpoint_output_slot = cutlass.Int32(
                            checkpoint_state_indices[ckpt_slot]
                        )
                    if (checkpoint_output_slot >= cutlass.Int32(0)) & (
                        checkpoint_output_slot < cutlass.Int32(state_ckpt.shape[0])
                    ):
                        tcgen05_store_final_state_tmem(
                            tmem_raw_addr,
                            KDA_TMEM_STATE_COL_OFFSET,
                            state_ckpt,
                            checkpoint_output_slot,
                            bidy,
                            dv_half,
                            warp_idx,
                            post_scale_lane,
                            HALF=True,
                        )
                    checkpoint_read_done_arrive(checkpoint_read_done_mbar)
                    ckpt_slot += cutlass.Int32(1)
                    ckpt_next += ckpt_stride
        if num_chunks > 0:
            last = num_chunks - cutlass.Int32(1)
            last_kr_phase = (last // 2) % 2
            tcgen05_wait_acc_buffer_ready(
                k_restore_consumed_l_mbar.subview(last % 2), last_kr_phase
            )
            tcgen05_wait_acc_buffer_ready(
                k_restore_consumed_mbar.subview(last % 2), last_kr_phase
            )
        if cutlass.const_expr(final_state is not None):
            final_lane = cute.arch.lane_idx()
            tcgen05_store_final_state_tmem(
                tmem_raw_addr,
                KDA_TMEM_STATE_COL_OFFSET,
                final_state,
                state_slot,
                bidy,
                dv_half,
                warp_idx,
                final_lane,
                HALF=True,
            )
        final_state_stored_arrive(final_state_stored_mbar)
    elif is_compute_group0_warp(warp_idx):
        if warp_idx < 4:
            prims.setmaxregister(KDA_CG1_REGS, prims.SetMaxRegisterAction.INCREASE)
            tmem_raw_addr = tmem_ptr_i32.load()
            for chunk in cutlass.range(num_chunks, unroll=1):
                o_stage = chunk % K2_QSTATE_STAGE_COUNT
                o_phase = (chunk // K2_QSTATE_STAGE_COUNT) % 2
                output_consumed_wait(
                    output_consumed_mbar.subview(o_stage),
                    (o_phase + 1) % 2,
                )
                tcgen05_wait_acc_buffer_ready(
                    qstate_acc_ready_mbar.subview(o_stage), o_phase
                )
                tcgen05_chain_load_qstate_output_tmem(
                    tmem_raw_addr,
                    o_smem,
                    warp_idx,
                    lane,
                    o_stage * K2_O_SMEM_STAGE_SIZE,
                    o_stage,
                    SCALE,
                    out.element_type,
                )
                output_ready_arrive(output_ready_mbar.subview(o_stage))
            final_state_stored_arrive(final_state_stored_mbar)
        else:
            # Chunk 0 is initialized by CG1. Afterwards, warps 4-7 own the
            # fused left-half state pack and FP32 decay while CG1 owns right.
            prims.setmaxregister(KDA_CG1_REGS, prims.SetMaxRegisterAction.INCREASE)
            tmem_raw_addr = tmem_ptr_i32.load()
            if cutlass.const_expr(state_ckpt is not None):
                cg0_ckpt_stride = checkpoint_stride_chunks
            if num_chunks > 0:
                state_input_ready_wait(
                    state_input_ready_l_mbar,
                    cutlass.Int32(0),
                )
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                update_ready_arrive(update_ready_mbar)
            for chunk in cutlass.range(1, num_chunks, 1, unroll=1):
                prev = chunk - cutlass.Int32(1)
                raw_stage = chunk % K2_RAW_STAGE_COUNT
                prev_kr_phase = (prev // 2) % 2
                tcgen05_wait_acc_buffer_ready(
                    stateq_done_mbar.subview(prev % 2),
                    prev_kr_phase,
                )
                tcgen05_wait_acc_buffer_ready(
                    k_restore_consumed_l_mbar.subview(prev % 2),
                    prev_kr_phase,
                )
                if cutlass.const_expr(state_ckpt is not None):
                    if chunk % cg0_ckpt_stride == 0:
                        checkpoint_read_done_wait(
                            checkpoint_read_done_mbar,
                            ((chunk // cg0_ckpt_stride) + cutlass.Int32(1)) % 2,
                        )
                        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                state_left = tcgen05_stage_state_input_dv2_half_tmem(
                    tmem_raw_addr,
                    warp_idx,
                    input_dtype,
                    0,
                )
                raw_ready_wait(
                    raw_ready_mbar.subview(raw_stage),
                    (chunk // K2_RAW_STAGE_COUNT) % 2,
                )
                diag_raw_stage = diag_raw_smem.subview(raw_stage * DIAG_REC_ELEMS)
                tcgen05_rescale_state_dv2_half_regs(
                    tmem_raw_addr,
                    diag_raw_stage,
                    warp_idx,
                    state_left,
                    state_input_ready_l_mbar,
                    0,
                )
                update_ready_arrive(update_ready_mbar)


@cute.jit
def host_chain_dv2(
    v: cute.Tensor,
    cu_seqlens: cute.Tensor,
    cu_chunks: cute.Tensor,
    ws_kd: cute.Tensor,
    ws_qd: cute.Tensor,
    ws_w: cute.Tensor,
    ws_qk: cute.Tensor,
    ws_diag: cute.Tensor,
    state_indices: cute.Tensor | None,
    initial_state: cute.Tensor | None,
    out: cute.Tensor,
    final_state: cute.Tensor | None,
    stream,
    head_base: cutlass.Int32,
    launch_heads: cutlass.Int32,
    SCALE: cutlass.Float32,
    state_ckpt: cute.Tensor | None,
    cu_ckpts: cute.Tensor | None,
    checkpoint_state_indices: cute.Tensor | None,
    checkpoint_stride_chunks: cutlass.Int32,
    THREADS: cutlass.Constexpr,
) -> None:
    """DV2 host: identical tensor maps to the base chain host, doubled grid-y."""

    num_sequences = cu_seqlens.shape[0] - 1
    seqlen = v.shape[1]
    heads = v.shape[2]
    ws_rows = ws_kd.shape[2]
    num_chunks_total = ws_qk.shape[2]
    # Keep singleton TensorMap modes canonical for the same reason as K1.
    tile_head_stride = DK
    diag_head_stride = DIAG_REC_ELEMS
    qk_head_stride = QK_REC_ELEMS
    if heads != cutlass.Int32(1):
        tile_head_stride = DK * ws_rows
        diag_head_stride = DIAG_REC_ELEMS * num_chunks_total
        qk_head_stride = QK_REC_ELEMS * num_chunks_total
    tile_layout = cute.make_layout(
        (DK, ws_rows, heads, 1),
        stride=(1, DK, tile_head_stride, DK),
    )
    v_layout = cute.make_layout(
        (DV, seqlen, heads, 1),
        stride=(1, DV * heads, DV, DV * seqlen * heads),
    )
    diag_layout = cute.make_layout(
        (DIAG_REC_ELEMS, num_chunks_total, heads, 1),
        stride=(
            1,
            DIAG_REC_ELEMS,
            diag_head_stride,
            DIAG_REC_ELEMS,
        ),
    )
    qk_layout = cute.make_layout(
        (QK_REC_ELEMS, num_chunks_total, heads, 1),
        stride=(
            1,
            QK_REC_ELEMS,
            qk_head_stride,
            QK_REC_ELEMS,
        ),
    )
    f16_box = (RAW_F16_TMA_SWIZZLE_ELEMS, BT, 1, 1)
    tma_desc_kd = cuda.create_tensor_map_tiled_from_view(
        cute.make_tensor(ws_kd.iterator, tile_layout),
        box_dims=f16_box,
        stride_order=(0, 1, 2, 3),
        swizzle=cuda.TensorMapSwizzle.s128b,
    )
    tma_desc_w = cuda.create_tensor_map_tiled_from_view(
        cute.make_tensor(ws_w.iterator, tile_layout),
        box_dims=f16_box,
        stride_order=(0, 1, 2, 3),
        swizzle=cuda.TensorMapSwizzle.s128b,
    )
    tma_desc_qd = cuda.create_tensor_map_tiled_from_view(
        cute.make_tensor(ws_qd.iterator, tile_layout),
        box_dims=f16_box,
        stride_order=(0, 1, 2, 3),
        swizzle=cuda.TensorMapSwizzle.s128b,
    )
    tma_desc_v = cuda.create_tensor_map_tiled_from_view(
        cute.make_tensor(v.iterator, v_layout),
        box_dims=f16_box,
        stride_order=(0, 1, 2, 3),
        swizzle=cuda.TensorMapSwizzle.s128b,
    )
    tma_desc_diag = cuda.create_tensor_map_tiled_from_view(
        cute.make_tensor(ws_diag.iterator, diag_layout),
        box_dims=(DIAG_REC_ELEMS, 1, 1, 1),
        stride_order=(0, 1, 2, 3),
        swizzle=cuda.TensorMapSwizzle.none,
    )
    tma_desc_qk = cuda.create_tensor_map_tiled_from_view(
        cute.make_tensor(ws_qk.iterator, qk_layout),
        box_dims=(QK_REC_ELEMS, 1, 1, 1),
        stride_order=(0, 1, 2, 3),
        swizzle=cuda.TensorMapSwizzle.none,
    )
    tma_desc_o = cuda.create_tensor_map_tiled_from_view(
        cute.make_tensor(out.iterator, v_layout),
        box_dims=(O_TMA_SWIZZLE_ELEMS, BT, 1, 1),
        stride_order=(0, 1, 2, 3),
        swizzle=cuda.TensorMapSwizzle.s128b,
    )
    kernel_chain_dv2(
        tma_desc_kd,
        tma_desc_w,
        tma_desc_qd,
        tma_desc_v,
        tma_desc_diag,
        tma_desc_qk,
        tma_desc_o,
        v,
        cu_seqlens,
        cu_chunks,
        state_indices,
        initial_state,
        out,
        final_state,
        head_base,
        SCALE,
        state_ckpt,
        cu_ckpts,
        checkpoint_state_indices,
        checkpoint_stride_chunks,
    ).launch(
        grid=(num_sequences, launch_heads * 2, 1),
        block=(THREADS, 1, 1),
        stream=stream,
        min_blocks_per_mp=1,
        preferred_smem_carveout=100,
    )


# =============================================================================
# Unified single-`cute.compile`-target host.
#
# ONE @cute.jit that runtime-branches the plain engine vs the two-kernel
# decomposition (kernel-1 factor prep + kernel-2 DV2 chain), calling `host` /
# `host_prep` / `host_chain_dv2` as NESTED jit sub-calls.  The occupancy rule
# lives inside the host (`n_seq * heads * 2 <= sm_count`, sm_count passed as a
# runtime arg).  Decomp always runs prep-first on one stream: k1 completes its
# full grid, then k2 reads a workspace k1 has fully written -- the kernel
# boundary is the only handoff, so neither kernel carries any cross-kernel
# synchronization.
#
# `cute.compile(host_unified, ...)` is the single target; `export_to_c` emits it
# to one `.o` with every device cubin embedded under one host entry.
# =============================================================================
@cute.jit
def host_unified(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    raw_gate: cute.Tensor,
    a_log: cute.Tensor,
    dt_bias: cute.Tensor,
    beta: cute.Tensor,
    cu_seqlens: cute.Tensor,
    seq_order: cute.Tensor,
    generate_seq_order: cutlass.Int32,
    state_indices: cute.Tensor | None,
    cu_chunks: cute.Tensor,
    generate_cu_chunks: cutlass.Int32,
    ws_kd: cute.Tensor,
    ws_qd: cute.Tensor,
    ws_w: cute.Tensor,
    ws_qk: cute.Tensor,
    ws_diag: cute.Tensor,
    initial_state: cute.Tensor | None,
    out: cute.Tensor,
    final_state: cute.Tensor | None,
    stream_a: cuda_driver.CUstream,
    seq_route: cutlass.Int32,
    sm_count: cutlass.Int32,
    checkpoint_stride_chunks: cutlass.Int32,
    prep_num_ctas: cutlass.Int32,
    prep_cpc: cutlass.Int32,
    scale: cutlass.Float32,
    state_ckpt: cute.Tensor | None,
    cu_ckpts: cute.Tensor | None,
    checkpoint_state_indices: cute.Tensor | None,
    SAFE_GATE: cutlass.Constexpr,
    GATE_SCALE_LOG2: cutlass.Constexpr,
    THREADS: cutlass.Constexpr,
    MODE: cutlass.Constexpr,
    gate_dtype: cutlass.Constexpr,
) -> None:
    n_seq = cu_seqlens.shape[0] - 1
    heads = q.shape[2]
    h32 = cutlass.Int32(heads)
    z = cutlass.Int32(0)
    if (generate_seq_order != z) | (generate_cu_chunks != z):
        # A separate launch supplies the grid-wide publication boundary needed
        # before either the persistent engine or decomposed kernels consume the
        # generated metadata.  The launch is captured as part of a CUDA Graph.
        packed_metadata_kernel(
            cu_seqlens,
            seq_order,
            cu_chunks,
            generate_seq_order,
            generate_cu_chunks,
        ).launch(
            grid=(1, 1, 1),
            block=(PACKED_METADATA_THREADS, 1, 1),
            stream=stream_a,
        )
    # Route selection.  MODE is a COMPILE-TIME constexpr:
    #   MODE is None    -> RUNTIME routing (both routes emitted, one .o handles
    #                      all shapes) — the default build.
    #   MODE == "engine"-> engine only (decomp branch const-folds away).
    #   MODE == "decomp"-> decomp only (engine branch const-folds away).
    # For MODE None the route is the single occupancy test: run the decomposition
    # iff its doubled k2 grid fits one wave (n_seq*heads*2 <= sm_count).  The
    # wrapper re-derives the SAME decision host-side.
    if cutlass.const_expr(MODE == "engine"):
        route_decomp = False
    elif cutlass.const_expr(MODE == "decomp"):
        route_decomp = True
    else:
        free_sms = sm_count - n_seq * heads * 2
        route_decomp = free_sms >= cutlass.Int32(0)
    if route_decomp:
        # ------------------------------ DECOMP ------------------------------
        # PREP-FIRST on a single stream, always.  k1 runs its full grid and
        # completes; k2 then reads a workspace that is already whole, so no
        # flag-ring, no per-chunk fences and no cross-stream ordering are needed
        # -- the kernel boundary supplies the visibility the flags used to.  The
        # co-resident (chain-first, two-stream) variant was removed: it required
        # the flag machinery on both kernels, and that cost more than the
        # concurrency returned.
        host_prep(
            q,
            k,
            raw_gate,
            a_log,
            dt_bias,
            beta,
            cu_seqlens,
            cu_chunks,
            ws_kd,
            ws_qd,
            ws_w,
            ws_qk,
            ws_diag,
            stream_a,
            prep_num_ctas,
            prep_cpc,
            z,
            h32,
            SAFE_GATE,
            GATE_SCALE_LOG2,
            THREADS,
            gate_dtype,
        )
        host_chain_dv2(
            v,
            cu_seqlens,
            cu_chunks,
            ws_kd,
            ws_qd,
            ws_w,
            ws_qk,
            ws_diag,
            state_indices,
            initial_state,
            out,
            final_state,
            stream_a,
            z,
            h32,
            scale,
            state_ckpt,
            cu_ckpts,
            checkpoint_state_indices,
            checkpoint_stride_chunks,
            THREADS,
        )
    else:
        # ------------------------------ ENGINE ------------------------------
        host(
            q,
            k,
            v,
            raw_gate,
            a_log,
            dt_bias,
            beta,
            cu_seqlens,
            seq_order,
            state_indices,
            initial_state,
            out,
            final_state,
            stream_a,
            scale,
            state_ckpt,
            cu_ckpts,
            checkpoint_state_indices,
            checkpoint_stride_chunks,
            SAFE_GATE,
            GATE_SCALE_LOG2,
            THREADS,
            gate_dtype,
        )


def _unified_fakes(
    dtype: type,
    state_dtype: type,
    has_state_in: bool,
    has_state_out: bool,
    has_state_ckpt: bool,
    has_state_indices: bool,
    has_checkpoint_state_indices: bool,
    gate_dtype: type = cutlass.BFloat16,
) -> tuple:
    """Union fake-tensor set for the single `host_unified` compile.

    Independent symbols where lengths are unrelated: cu_chunks (`scu2`) is NOT
    tied to cu_seqlens (`scu`), so the engine route can pass a length-1 dummy;
    ws_qk / ws_diag share `sc` (chunk count), while ws_kd/qd/w use the
    16-divisible `sc16` (chunk*BT rows).
    """
    F = make_fake_compact_tensor
    scu = cute.sym_int64(divisibility=1)
    scu2 = cute.sym_int64(divisibility=1)
    sn = cute.sym_int64(divisibility=1)
    sp = cute.sym_int64(divisibility=1)
    sh = cute.sym_int64(divisibility=1)
    ss = cute.sym_int64(divisibility=1)
    sc = cute.sym_int64(divisibility=1)
    sc16 = cute.sym_int64(divisibility=16)
    fq = F(dtype, (1, ss, sh, DK), stride_order=(3, 2, 1, 0), assumed_align=16)
    fk = F(dtype, (1, ss, sh, DK), stride_order=(3, 2, 1, 0), assumed_align=16)
    fv = F(dtype, (1, ss, sh, DV), stride_order=(3, 2, 1, 0), assumed_align=16)
    fgate = F(gate_dtype, (1, ss, sh, DK), stride_order=(3, 2, 1, 0), assumed_align=16)
    fa_log = F(cutlass.Float32, (sh,), stride_order=(0,), assumed_align=16)
    fdt = F(cutlass.Float32, (sh, DK), stride_order=(1, 0), assumed_align=16)
    fbeta = F(cutlass.BFloat16, (1, ss, sh), stride_order=(2, 1, 0), assumed_align=16)
    fcu = F(cutlass.Int64, (scu,), stride_order=(0,), assumed_align=8)
    forder = F(cutlass.Int32, (sn,), stride_order=(0,), assumed_align=8)
    fstate_indices = (
        F(cutlass.Int32, (sn,), stride_order=(0,), assumed_align=4)
        if has_state_indices
        else None
    )
    fcuc = F(cutlass.Int32, (scu2,), stride_order=(0,), assumed_align=8)

    def ws_tile():
        return F(dtype, (1, sh, sc16, DK), stride_order=(3, 2, 1, 0), assumed_align=16)

    fkd, fqd, fw = ws_tile(), ws_tile(), ws_tile()
    fqk = F(
        dtype, (1, sh, sc, QK_REC_ELEMS), stride_order=(3, 2, 1, 0), assumed_align=16
    )
    fdiag = F(
        cutlass.Float32,
        (1, sh, sc, DIAG_REC_ELEMS),
        stride_order=(3, 2, 1, 0),
        assumed_align=16,
    )

    def state_pool_fake():
        return cute.runtime.make_fake_tensor(
            state_dtype,
            shape=(sp, sh, DV, DK),
            stride=(cute.sym_int64(divisibility=8), DV * DK, DK, 1),
            assumed_align=16,
        )

    fstate = state_pool_fake() if has_state_in else None
    fout = F(dtype, (1, ss, sh, DV), stride_order=(3, 2, 1, 0), assumed_align=16)
    ffinal = state_pool_fake() if has_state_out else None
    # State checkpoints: a flat [total_ckpts, H, DV, DK] tensor (same VK layout
    # as final_state) plus per-sequence base offsets; own symbols so capacity is
    # unrelated to n_seq.
    # The symbols are only minted for checkpoint builds: sym_int64() registers
    # into the shared symbol table, and a dangling entry shifts every kernel's
    # constant-bank layout of the non-checkpoint build.
    if has_state_ckpt:
        sk = cute.sym_int64(divisibility=1)
        scuk = cute.sym_int64(divisibility=1)
        fckpt = F(
            state_dtype, (sk, sh, DV, DK), stride_order=(3, 2, 1, 0), assumed_align=16
        )
        fcuk = F(cutlass.Int64, (scuk,), stride_order=(0,), assumed_align=8)
        fcheckpoint_state_indices = (
            F(
                cutlass.Int32,
                (cute.sym_int64(divisibility=1),),
                stride_order=(0,),
                assumed_align=4,
            )
            if has_checkpoint_state_indices
            else None
        )
    else:
        fckpt = None
        fcuk = None
        fcheckpoint_state_indices = None
    return (
        (
            fq,
            fk,
            fv,
            fgate,
            fa_log,
            fdt,
            fbeta,
            fcu,
            forder,
            0,  # generate_seq_order
            fstate_indices,
            fcuc,
            0,  # generate_cu_chunks
            fkd,
            fqd,
            fw,
            fqk,
            fdiag,
            fstate,
            fout,
            ffinal,
        ),
        fckpt,
        fcuk,
        fcheckpoint_state_indices,
    )


_ENGINE_DUMMIES: dict = {}


def _engine_dummies(heads: int, device, dtype: type):
    """Minimal length-1 workspace views for the engine route.

    The unified ABI takes the workspace tiles unconditionally; the engine branch
    never reads them, so a cached length-1 set (satisfying the compiled symbolic
    shape constraints) is passed instead of allocating a real workspace.
    """
    torch_dtype = DTYPE_MAP[dtype]
    key = (heads, str(device), torch_dtype)
    d = _ENGINE_DUMMIES.get(key)
    if d is None:
        kd = torch.zeros(1, heads, 16, DK, dtype=torch_dtype, device=device)
        qk = torch.zeros(1, heads, 1, QK_REC_ELEMS, dtype=torch_dtype, device=device)
        diag = torch.zeros(
            1, heads, 1, DIAG_REC_ELEMS, dtype=torch.float32, device=device
        )
        cuc = torch.zeros(1, dtype=torch.int32, device=device)
        d = (kd, kd, kd, qk, diag, cuc)
        _ENGINE_DUMMIES[key] = d
    return d


_CKPT_OFFSETS: dict = {}


_UNIFORM_CU_CACHE: dict = {}


_IDENTITY_SEQUENCE_ORDER_CACHE: dict = {}
_DEVICE_SEQUENCE_ORDER_CACHE: dict = {}


def _stream_pointer(stream) -> int:
    value = getattr(stream, "value", stream)
    return int(value or 0)


def _device_sequence_order_buffer(num_sequences: int, device, stream) -> torch.Tensor:
    """Reusable per-stream output for device-side packed sequence sorting."""

    key = (str(device), _stream_pointer(stream), int(num_sequences))
    order = _DEVICE_SEQUENCE_ORDER_CACHE.get(key)
    if order is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "KDA device sequence order must be warmed before CUDA graph capture"
            )
        order = torch.empty(num_sequences, dtype=torch.int32, device=device)
        _DEVICE_SEQUENCE_ORDER_CACHE[key] = order
    return order


def _identity_sequence_order(num_sequences: int, device) -> torch.Tensor:
    """Cached original sequence order for the low-level callable."""

    key = (str(device), int(num_sequences))
    order = _IDENTITY_SEQUENCE_ORDER_CACHE.get(key)
    if order is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "KDA identity sequence order must be warmed before CUDA graph capture"
            )
        order = torch.arange(num_sequences, dtype=torch.int32, device=device)
        _IDENTITY_SEQUENCE_ORDER_CACHE[key] = order
    return order


def _uniform_cu_seqlens(batch: int, seqlen: int, device) -> torch.Tensor:
    """Cached uniform cu_seqlens for the batched [B, T, H, D] entry."""

    key = (batch, seqlen, str(device))
    cu = _UNIFORM_CU_CACHE.get(key)
    if cu is None:
        cu = torch.arange(
            0, (batch + 1) * seqlen, seqlen, dtype=torch.int64, device=device
        )
        _UNIFORM_CU_CACHE[key] = cu
    return cu


def _make_call(unified: Callable, spec: dict) -> CompiledKDA:
    """Build the single user host callable over the one compiled `host_unified`.

    Runtime ABI: (q, k, v, raw_gate, a_log, dt_bias, beta, cu_seqlens,
    initial_state, out, final_state, workspace, stream_a, scale, ..., seq_order)
    — packed engine calls with ``seq_order=None`` generate an LPT order on the
    device; fixed and decomp calls retain the original sequence order unless a
    wrapper requests device-generated metadata. An explicit packed CUDA int32
    permutation overrides the implicit engine order. The remaining
    positional ABI is
    the reference host ABI (stream_a == the reference `stream`, scale in the same
    position) plus the `workspace` operand.  Decomp always runs prep-first on
    overlap.  `.workspace_size(cu_seqlens, heads)` is attached.  There is NO
    runtime `mode`: the route (engine/decomp) is fixed by the compile-time `mode`
    constexpr — None (default) keeps the runtime occupancy rule, "engine"/"decomp"
    specialize the build to one route.

    Both routes launch on the caller's single stream.
    """

    decisions: dict = {}
    compile_mode = spec["mode"]
    dtype = spec["dtype"]
    built_states = (
        spec["has_state_in"],
        spec["has_state_out"],
        spec["has_state_ckpt"],
        spec["has_state_indices"],
        spec.get("has_checkpoint_state_indices", False),
    )

    def call(
        q,
        k,
        v,
        raw_gate,
        a_log,
        dt_bias,
        beta,
        cu_seqlens,
        initial_state,
        out,
        final_state,
        workspace,
        stream_a,
        scale=DEFAULT_SCALE,
        state_ckpt=None,
        checkpoint_cu_starts=None,
        ckpt_interval=0,
        seq_order=None,
        planned_cu_chunks=None,
        planned_total_chunks=None,
        generate_planned_metadata=False,
        state_indices=None,
        checkpoint_state_indices=None,
    ):
        if checkpoint_state_indices is not None and state_ckpt is None:
            raise ValueError("checkpoint_state_indices requires state_ckpt")
        # The state specialization is DERIVED from which state tensors the call
        # actually passes (initial/final/checkpoint present or None); if it
        # differs from this build, transparently delegate to the matching one
        # (compile() is cached -- the first such call pays one compilation).
        want = (
            initial_state is not None,
            final_state is not None,
            state_ckpt is not None,
            state_indices is not None,
            checkpoint_state_indices is not None,
        )
        if want != built_states:
            sibling = compile(
                dtype=spec["dtype"],
                state_dtype=spec["state_dtype"],
                gate_dtype=spec["gate_dtype"],
                safe_gate=spec["safe_gate"],
                gate_lower_bound=spec["gate_lower_bound"],
                has_state_in=want[0],
                has_state_out=want[1],
                has_state_ckpt=want[2],
                has_state_indices=want[3],
                has_checkpoint_state_indices=want[4],
                mode=spec["mode"],
            )
            return sibling(
                q,
                k,
                v,
                raw_gate,
                a_log,
                dt_bias,
                beta,
                cu_seqlens,
                initial_state,
                out,
                final_state,
                workspace,
                stream_a,
                scale,
                state_indices=state_indices,
                state_ckpt=state_ckpt,
                checkpoint_cu_starts=checkpoint_cu_starts,
                checkpoint_state_indices=checkpoint_state_indices,
                ckpt_interval=ckpt_interval,
                seq_order=seq_order,
                planned_cu_chunks=planned_cu_chunks,
                planned_total_chunks=planned_total_chunks,
                generate_planned_metadata=generate_planned_metadata,
            )
        # One stream.  There used to be an optional `stream_b` that opted into a
        # co-resident k1/k2 overlap; it was removed because the flag-ring it
        # required cost more than the concurrency it bought (see host_unified).
        # Decomp is now always prep-first on `stream_a`.
        # Token-major user convention: cu_seqlens given -> packed
        # [1, T_total, H, D] activations (beta [1, T_total, H]); cu_seqlens
        # None -> batched [B, T, H, D] (beta [B, T, H]), flattened zero-copy
        # into a packed call with synthesized uniform boundaries.  The tensors
        # feed the kernels directly in token-major memory order: the hosts
        # build every TMA descriptor from explicit (shape, stride) tuples, so
        # no logical permute is needed anywhere.
        packed_layout = cu_seqlens is not None
        if not packed_layout:
            bsz = int(q.shape[0])
            tlen = int(q.shape[1])
            cu_seqlens = _uniform_cu_seqlens(bsz, tlen, q.device)
            q = q.reshape(1, bsz * tlen, q.shape[2], q.shape[3])
            k = k.reshape(1, bsz * tlen, k.shape[2], k.shape[3])
            v = v.reshape(1, bsz * tlen, v.shape[2], v.shape[3])
            raw_gate = raw_gate.reshape(
                1, bsz * tlen, raw_gate.shape[2], raw_gate.shape[3]
            )
            beta = beta.reshape(1, bsz * tlen, beta.shape[2])
            out = out.reshape(1, bsz * tlen, out.shape[2], out.shape[3])
        heads = int(q.shape[2])
        device = q.device
        sm_count = _device_sm_count(device)
        n_seq = cu_seqlens.numel() - 1
        if state_indices is not None and (
            initial_state is None
            or state_indices.device != device
            or state_indices.dtype != torch.int32
            or state_indices.ndim != 1
            or not state_indices.is_contiguous()
            or state_indices.numel() != n_seq
        ):
            raise ValueError(
                "state_indices requires initial_state and must be a contiguous "
                "CUDA int32 tensor with one entry per sequence"
            )
        if checkpoint_state_indices is not None and (
            state_ckpt is None
            or checkpoint_state_indices.device != device
            or checkpoint_state_indices.dtype != torch.int32
            or checkpoint_state_indices.ndim != 1
            or not checkpoint_state_indices.is_contiguous()
        ):
            raise ValueError(
                "checkpoint_state_indices requires state_ckpt and must be a "
                "contiguous CUDA int32 tensor"
            )
        key = (n_seq, str(device), heads, compile_mode)
        kind = decisions.get(key)
        if kind is None:
            kind = _route_for_workspace(n_seq, heads, device, compile_mode or "auto")
            decisions[key] = kind
        generate_seq_order = bool(generate_planned_metadata)
        if seq_order is None:
            if packed_layout and kind == "engine":
                generate_seq_order = True
                seq_order = _device_sequence_order_buffer(n_seq, device, stream_a)
            else:
                seq_order = _identity_sequence_order(n_seq, device)
        elif (
            seq_order.device != device
            or seq_order.dtype != torch.int32
            or seq_order.ndim != 1
            or not seq_order.is_contiguous()
            or seq_order.numel() != n_seq
        ):
            raise ValueError(
                "seq_order must be a contiguous CUDA int32 tensor with one "
                "entry per sequence"
            )
        has_planned_chunks = (
            planned_cu_chunks is not None and planned_total_chunks is not None
        )
        if (planned_cu_chunks is None) != (planned_total_chunks is None):
            raise ValueError(
                "planned_cu_chunks and planned_total_chunks must be provided together"
            )
        if generate_planned_metadata and not has_planned_chunks:
            raise ValueError(
                "device-generated chunk metadata requires planned_cu_chunks and "
                "planned_total_chunks capacity"
            )
        cu_list = None
        if (kind == "decomp" and not has_planned_chunks) or (
            state_ckpt is not None and checkpoint_cu_starts is None
        ):
            cu_list = _cu_seqlens_contents(cu_seqlens)
        checkpoint_stride_chunks = 0
        if state_ckpt is not None:
            if ckpt_interval <= 0 or ckpt_interval % BT:
                raise ValueError(
                    f"ckpt_interval must be a positive multiple of {BT}, "
                    f"got {ckpt_interval}"
                )
            if checkpoint_cu_starts is not None and (
                checkpoint_cu_starts.device != device
                or checkpoint_cu_starts.dtype != torch.int64
                or checkpoint_cu_starts.ndim != 1
                or not checkpoint_cu_starts.is_contiguous()
                or checkpoint_cu_starts.numel() != n_seq + 1
            ):
                raise ValueError(
                    "checkpoint_cu_starts must be a contiguous CUDA int64 "
                    "tensor with one entry per sequence plus one"
                )
            checkpoint_stride_chunks = ckpt_interval // BT
            if checkpoint_cu_starts is None:
                assert cu_list is not None
                offsets, total = [0], 0
                for i in range(n_seq):
                    seq_len = cu_list[i + 1] - cu_list[i]
                    if checkpoint_state_indices is None:
                        total += (seq_len + ckpt_interval - 1) // ckpt_interval
                    else:
                        total += seq_len // ckpt_interval
                    offsets.append(total)
                cu_ckpts_arg = torch.tensor(offsets, dtype=torch.int64, device=device)
            else:
                cu_ckpts_arg = checkpoint_cu_starts
        else:
            cu_ckpts_arg = None
        seq_route = 1  # single stream; kept as an arg so host_unified re-derives
        # The route rule lives in exactly two places: `_route_for_workspace`
        # (which `workspace_size` uses, so the caller allocates the right
        # buffer) and `host_unified` (which re-derives it to pick the branch to
        # execute).  This wrapper does NOT own a third copy -- it only asks the
        # workspace helper which operands to marshal.
        if kind == "engine":
            # beta rides a TMA descriptor built over the compact [T, H] layout:
            # the base must be 16B-aligned (cuTensorMapEncodeTiled hard rule)
            # and the memory contiguous.  Fresh allocator tensors always pass;
            # an odd-token-offset VIEW at heads % 8 != 0 is the one real way
            # to violate it -- fail loudly with the one-line fix.
            if (not beta.is_contiguous()) or beta.data_ptr() % 16:
                raise ValueError(
                    "beta must be contiguous [1, T, H] with a 16-byte-aligned "
                    "base for the TMA beta transport; pass the unsliced tensor "
                    "or call .contiguous()"
                )
            if (
                heads % 8
                and heads
                * (BT + 8 // (heads & -heads if heads & -heads < 8 else 8))
                * 2
                > 2 * BETA_TILE_STAGE_ELEMS
            ):
                raise ValueError(
                    f"heads={heads}: the pair-packed beta tile exceeds the SMEM "
                    "stage; supported are heads % 8 == 0 or heads <= 14"
                )
            kd, qd, wt, qk, diag, cuc = _engine_dummies(heads, device, dtype)
            unified(
                q,
                k,
                v,
                raw_gate,
                a_log,
                dt_bias,
                beta,
                cu_seqlens,
                seq_order,
                int(generate_seq_order),
                state_indices,
                cuc,
                0,  # engine does not consume a chunk prefix
                kd,
                qd,
                wt,
                qk,
                diag,
                initial_state,
                out,
                final_state,
                stream_a,
                seq_route,
                sm_count,
                checkpoint_stride_chunks,
                1,
                1,
                scale,
                state_ckpt,
                cu_ckpts_arg,
                checkpoint_state_indices,
            )
            return

        # ---- decomp route ----
        if workspace is None:
            raise ValueError(
                "workspace required for this shape; call workspace_size() and allocate"
            )
        if has_planned_chunks:
            if (
                planned_cu_chunks.device != device
                or planned_cu_chunks.dtype != torch.int32
                or planned_cu_chunks.ndim != 1
                or not planned_cu_chunks.is_contiguous()
                or planned_cu_chunks.numel() != n_seq + 1
                or not isinstance(planned_total_chunks, int)
                or planned_total_chunks <= 0
            ):
                raise ValueError(
                    "planned chunk metadata must be a contiguous CUDA int32 "
                    "cu_chunks[N+1] tensor and a positive host total_chunks"
                )
            plan = {
                "total_chunks": planned_total_chunks,
                "cu_chunks": planned_cu_chunks,
            }
        else:
            plan = _plan(cu_seqlens)
        total_chunks = plan["total_chunks"]
        ws = _partition_workspace(workspace, heads, total_chunks)
        # Decomp is prep-first on one stream: k1 runs its full grid, then k2
        # consumes what k1 left behind.  The co-resident overlap variant that
        # used to live here was removed because it never paid: enabling it forced
        # the per-chunk flag publish/wait onto BOTH kernels, and that tax exceeds
        # the concurrency it buys.  Measured on the historical dual-variant
        # build (flag-free and flag-carrying kernels side by side), dropping
        # the flags makes prep 30-44% cheaper (H32-T8192 130.7 -> 91.2 us,
        # H12-2048x4 73.6 -> 41.5) and 4 of 5 decomp shapes tie or improve
        # end-to-end (H12-2048x4 -22.6%, H32-T8192 -13.0%) against a single 1.3%
        # loss on H12-varlen -- with verify() bit-identical between the two.
        num_ctas, cpc = _k1_grid(total_chunks, heads)
        unified(
            q,
            k,
            v,
            raw_gate,
            a_log,
            dt_bias,
            beta,
            cu_seqlens,
            seq_order,
            int(generate_seq_order),
            state_indices,
            plan["cu_chunks"],
            int(generate_planned_metadata),
            ws["kd"],
            ws["qd"],
            ws["w"],
            ws["qk"],
            ws["diag"],
            initial_state,
            out,
            final_state,
            stream_a,
            seq_route,
            sm_count,
            checkpoint_stride_chunks,
            num_ctas,
            cpc,
            scale,
            state_ckpt,
            cu_ckpts_arg,
            checkpoint_state_indices,
        )

    def _ws_size(cu_seqlens, heads, mode=None, *, batch=None, seqlen=None):
        if cu_seqlens is None:
            if batch is None or seqlen is None:
                raise ValueError(
                    "workspace_size(None, heads) needs batch= and seqlen= "
                    "for the non-varlen [B, T, H, D] entry"
                )
            cu_seqlens = tuple(int(seqlen) * i for i in range(int(batch) + 1))
        return workspace_size(cu_seqlens, heads, mode or (compile_mode or "auto"))

    def _ws_size_from_total_chunks(num_sequences, heads, total_chunks, device):
        route_mode = compile_mode or "auto"
        if _route_for_workspace(num_sequences, heads, device, route_mode) == "engine":
            return 0
        return _decomp_ws_bytes(int(heads), int(total_chunks))

    compiled_call = cast(CompiledKDA, call)
    compiled_call.workspace_size = _ws_size
    compiled_call.workspace_size_from_total_chunks = _ws_size_from_total_chunks
    return compiled_call


@lru_cache(maxsize=None)
def compile(  # noqa: A001
    dtype: type = cutlass.BFloat16,
    state_dtype: type = cutlass.Float32,
    gate_dtype: type = cutlass.BFloat16,
    safe_gate: bool = True,
    gate_lower_bound: float = DEFAULT_GATE_LOWER_BOUND,
    has_state_in: bool = True,
    has_state_out: bool = True,
    has_state_ckpt: bool = False,
    has_state_indices: bool = False,
    has_checkpoint_state_indices: bool = False,
    mode=None,
) -> CompiledKDA:
    """Compile KDA; the returned callable is the single host entry.

    The signature matches the reference `kda_chunked_bt16.compile`; the only
    user-visible additions are the decomp `workspace` (a single opaque uint8
    buffer sized by the attached `.workspace_size(cu_seqlens, heads)` and passed
    before the stream).

    ONE `cute.compile(host_unified, ...)` target.  `mode` is a COMPILE-TIME
    constexpr fixing the route:
      * None (default) -> the returned callable routes engine vs the two-kernel
        decomposition at RUNTIME by the occupancy rule (decomp-DV2 iff
        `n_seq*heads*2 <= sm_count`, else engine); both paths are in the one .o.
      * "engine" -> engine-only build (decomp branch const-folds away).
      * "decomp" -> decomp-only build (engine branch const-folds away).
    ("auto" is accepted as an alias for None.)
    """

    if dtype not in CLI_DTYPES.values():
        raise ValueError(f"Unsupported dtype: {dtype}")
    if state_dtype not in (cutlass.BFloat16, cutlass.Float32):
        raise ValueError(f"Unsupported state dtype: {state_dtype}")
    if has_checkpoint_state_indices and not has_state_ckpt:
        raise ValueError("checkpoint state indices require checkpoint state output")
    validate_gate_dtype(gate_dtype)
    if mode == "auto":
        mode = None
    if mode not in (None, "engine", "decomp"):
        raise ValueError(f"Unknown mode: {mode}")
    gate_scale_log2 = gate_lower_bound * LOG2_E

    fakes, fckpt, fcuk, fcheckpoint_state_indices = _unified_fakes(
        dtype,
        state_dtype,
        has_state_in,
        has_state_out,
        has_state_ckpt,
        has_state_indices,
        has_checkpoint_state_indices,
        gate_dtype,
    )
    unified = cute.compile(
        host_unified,
        *fakes,
        make_fake_stream(),
        0,  # seq_route
        0,  # sm_count
        0,  # checkpoint_stride_chunks
        0,  # prep_num_ctas
        0,  # prep_cpc
        DEFAULT_SCALE,
        fckpt,
        fcuk,
        fcheckpoint_state_indices,
        safe_gate,
        gate_scale_log2,
        THREADS_PER_CTA,
        mode,  # MODE constexpr (route specialization)
        gate_dtype,  # gate ABI dtype constexpr (32-bit vs 16-bit gate path)
        options="--enable-tvm-ffi",
    )
    fn = _make_call(
        unified,
        dict(
            dtype=dtype,
            state_dtype=state_dtype,
            gate_dtype=gate_dtype,
            safe_gate=safe_gate,
            gate_lower_bound=gate_lower_bound,
            has_state_in=has_state_in,
            has_state_out=has_state_out,
            has_state_ckpt=has_state_ckpt,
            has_state_indices=has_state_indices,
            has_checkpoint_state_indices=has_checkpoint_state_indices,
            mode=mode,
        ),
    )
    # Eager device-module load: keep the module resident from build time with one
    # tiny decomp launch, so no runtime call pays a lazy cuModuleLoad.  Guarded so
    # a login-node dry-run (no GPU) still returns the compiled callable.
    if torch.cuda.is_available():
        _eager_module_load(
            fn,
            dtype,
            state_dtype,
            has_state_in,
            has_state_out,
            has_state_ckpt,
            has_state_indices,
            has_checkpoint_state_indices,
            gate_dtype,
        )
    return fn


def _eager_module_load(
    fn: CompiledKDA,
    dtype: type,
    state_dtype: type,
    has_state_in: bool,
    has_state_out: bool,
    has_state_ckpt: bool = False,
    has_state_indices: bool = False,
    has_checkpoint_state_indices: bool = False,
    gate_dtype: type = cutlass.BFloat16,
) -> None:
    """One launch to make the device module resident so no real call lazy-loads.

    heads=1 keeps free_sms >= 0, so a MODE=None build routes this to DECOMP and
    loads the prep+chain cubins (an engine-routed preload would not).
    MODE="engine" routes it to engine (the only cubin there); MODE="decomp" to
    decomp — either way the single module's cubins become resident."""
    try:
        device = torch.cuda.current_device()
        heads = 1
        seqlen = 64 * BT  # a few chunks: enough to exercise prep + chain
        q, k, v, raw_gate, a_log, dt_bias, beta, cu_seqlens, state = _make_inputs(
            1,
            heads,
            seqlen,
            dtype,
            state_dtype,
            device,
            random_state=False,
            gate_dtype=gate_dtype,
        )
        out = torch.zeros_like(v)
        initial_state = state if has_state_in else None
        final_state = torch.zeros_like(state) if has_state_out else None
        ws_bytes = fn.workspace_size(cu_seqlens, heads)
        workspace = (
            torch.empty(ws_bytes, dtype=torch.uint8, device=cu_seqlens.device)
            if ws_bytes > 0
            else None
        )
        s = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
        # Prep-first decomp launch: loads the single module's cubins eagerly.
        fn(
            q,
            k,
            v,
            raw_gate,
            a_log,
            dt_bias,
            beta,
            cu_seqlens,
            initial_state,
            out,
            final_state,
            workspace,
            s,
            DEFAULT_SCALE,
            state_indices=(
                torch.zeros(1, dtype=torch.int32, device="cuda")
                if has_state_indices
                else None
            ),
            state_ckpt=(
                torch.zeros(
                    64,
                    heads,
                    DV,
                    DK,
                    dtype=DTYPE_MAP[state_dtype],
                    device="cuda",
                )
                if has_state_ckpt
                else None
            ),
            ckpt_interval=(BT if has_state_ckpt else 0),
            checkpoint_state_indices=(
                torch.arange(64, dtype=torch.int32, device="cuda")
                if has_checkpoint_state_indices
                else None
            ),
        )
        torch.cuda.synchronize()
    except Exception:  # noqa: BLE001 — best-effort module preload; never fatal
        pass


def _make_inputs(
    batch: int,
    heads: int,
    seqlen: int,
    dtype: type,
    state_dtype: type,
    device: str,
    random_state: bool = False,
    safe_gate: bool = True,
    seq_lens: tuple[int, ...] | None = None,
    gate_dtype: type = cutlass.BFloat16,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    if seq_lens is None:
        seq_lens = (seqlen,) * batch
    if len(seq_lens) != batch or any(length < 0 for length in seq_lens):
        raise ValueError("seq_lens must contain one non-negative length per sequence")
    total_seqlen = sum(seq_lens)
    torch_dtype = DTYPE_MAP[dtype]
    torch_state_dtype = DTYPE_MAP[state_dtype]
    # The gate is materialized in FP32 (RNG parity with the historical harness,
    # byte-for-byte identical draws) and then ROUNDED to the ABI dtype.  The
    # torch reference consumes THIS tensor, so it models the input rounding
    # exactly while keeping every internal gate op in FP32.
    torch_gate_dtype = DTYPE_MAP[gate_dtype]
    q = 0.25 * torch.randn(1, total_seqlen, heads, DK, dtype=torch_dtype, device=device)
    k = 0.25 * torch.randn(1, total_seqlen, heads, DK, dtype=torch_dtype, device=device)
    v = torch.randn(1, total_seqlen, heads, DV, dtype=torch_dtype, device=device)
    if safe_gate:
        raw_gate = (
            0.25
            * torch.randn(
                1,
                total_seqlen,
                heads,
                DK,
                dtype=torch.float32,
                device=device,
            )
        ).to(torch_gate_dtype)
        # Safe path draws are unchanged (zeros consume no RNG): reproducible
        # byte-for-byte vs the original harness.
        a_log = torch.zeros(heads, dtype=torch.float32, device=device)
        dt_bias = torch.zeros(heads, DK, dtype=torch.float32, device=device)
    else:
        # FLA non-safe gate: RAW standard-normal gate logits plus non-trivial
        # A_log and dt_bias so the in-kernel softplus activation is exercised.
        raw_gate = torch.randn(
            1,
            total_seqlen,
            heads,
            DK,
            dtype=torch.float32,
            device=device,
        ).to(torch_gate_dtype)
        a_log = torch.rand(heads, dtype=torch.float32, device=device)
        dt_bias = torch.rand(heads, DK, dtype=torch.float32, device=device)
    beta = torch.randn(
        1,
        total_seqlen,
        heads,
        dtype=torch.bfloat16,
        device=device,
    )
    cu_seqlens = torch.zeros(batch + 1, dtype=torch.int64, device=device)
    if batch > 0:
        cu_seqlens[1:] = torch.tensor(
            seq_lens, dtype=torch.int64, device=device
        ).cumsum(0)
    if random_state:
        state = 0.05 * torch.randn(
            batch, heads, DV, DK, dtype=torch_state_dtype, device=device
        )
    else:
        state = torch.zeros(
            batch,
            heads,
            DV,
            DK,
            dtype=torch_state_dtype,
            device=device,
        )
    return q, k, v, raw_gate, a_log, dt_bias, beta, cu_seqlens, state


def run(
    compiled_fn: CompiledKDA,
    batch: int = DEFAULT_BATCH,
    heads: int = DEFAULT_HEADS,
    seqlen: int = DEFAULT_SEQLEN,
    dtype: type = cutlass.BFloat16,
    state_dtype: type = cutlass.Float32,
    gate_dtype: type = cutlass.BFloat16,
    has_state_in: bool = True,
    has_state_out: bool = True,
    seq_lens: tuple[int, ...] | None = None,
    scale: float = DEFAULT_SCALE,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run packed KDA with runtime scale and optionally return the VK state.

    This public runtime helper is also re-exported by the Rubin specialization.
    ``compiled_fn`` must use the same state-presence specialization selected by
    ``has_state_in`` and ``has_state_out``.
    """

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    q, k, v, raw_gate, a_log, dt_bias, beta, cu_seqlens, state = _make_inputs(
        batch,
        heads,
        seqlen,
        dtype,
        state_dtype,
        device="cuda",
        seq_lens=seq_lens,
        gate_dtype=gate_dtype,
    )
    out = torch.zeros_like(v)
    initial_state = state if has_state_in else None
    final_state = torch.zeros_like(state) if has_state_out else None

    # User-allocated opaque workspace: query bytes for the route this host will
    # take, then allocate a raw (uninitialized) buffer. The host initializes it.
    ws_bytes = compiled_fn.workspace_size(cu_seqlens, heads)
    workspace = (
        torch.empty(ws_bytes, dtype=torch.uint8, device=cu_seqlens.device)
        if ws_bytes > 0
        else None
    )

    stream = torch.cuda.current_stream(cu_seqlens.device).cuda_stream
    compiled_fn(
        q,
        k,
        v,
        raw_gate,
        a_log,
        dt_bias,
        beta,
        cu_seqlens,
        initial_state,
        out,
        final_state,
        workspace,
        stream,
        scale,
    )
    torch.cuda.synchronize()
    return out, final_state


def verify(
    batch: int = DEFAULT_BATCH,
    heads: int = DEFAULT_HEADS,
    seqlen: int = BT,
    dtype: type = cutlass.BFloat16,
    state_dtype: type = cutlass.Float32,
    gate_dtype: type = cutlass.BFloat16,
    compile_only: bool = True,
    safe_gate: bool = True,
    has_state_in: bool = True,
    has_state_out: bool = True,
    seq_lens: tuple[int, ...] | None = None,
    scale: float = DEFAULT_SCALE,
    mode: str = "engine",
) -> None:
    """Compile the kernel and optionally compare a device run to the MMA reference."""

    compiled_fn = compile(
        dtype=dtype,
        state_dtype=state_dtype,
        gate_dtype=gate_dtype,
        safe_gate=safe_gate,
        has_state_in=has_state_in,
        has_state_out=has_state_out,
        mode=mode,
    )
    print(
        f"Compile KDA kernel (BT={BT}, runtime seqlen, dtype={dtype}, "
        f"gate_dtype={gate_dtype}) OK"
    )

    if compile_only:
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    q, k, v, raw_gate, a_log, dt_bias, beta, cu_seqlens, state = _make_inputs(
        batch,
        heads,
        seqlen,
        dtype,
        state_dtype,
        device="cuda",
        random_state=True,
        safe_gate=safe_gate,
        seq_lens=seq_lens,
        gate_dtype=gate_dtype,
    )
    out = torch.zeros_like(v)
    initial_state = state if has_state_in else None
    final_state = torch.zeros_like(state) if has_state_out else None

    ws_bytes = compiled_fn.workspace_size(cu_seqlens, heads)
    workspace = (
        torch.empty(ws_bytes, dtype=torch.uint8, device=cu_seqlens.device)
        if ws_bytes > 0
        else None
    )

    stream_a = torch.cuda.current_stream(cu_seqlens.device).cuda_stream
    compiled_fn(
        q,
        k,
        v,
        raw_gate,
        a_log,
        dt_bias,
        beta,
        cu_seqlens,
        initial_state,
        out,
        final_state,
        workspace,
        stream_a,
        scale,
    )
    torch.cuda.synchronize()

    state_kv = _state_vk_to_kv(state) if has_state_in else None
    # Reference keeps the head-major logical convention; permute the
    # token-major user tensors to zero-copy [1, H, T, D] / [1, H, T] views.
    out_ref, state_ref_kv = kda_chunked_mma_reference(
        q.permute(0, 2, 1, 3),
        k.permute(0, 2, 1, 3),
        v.permute(0, 2, 1, 3),
        raw_gate.permute(0, 2, 1, 3),
        beta.permute(0, 2, 1),
        state_kv,
        mma_dtype=DTYPE_MAP[dtype],
        a_log=a_log,
        dt_bias=dt_bias,
        safe_gate=safe_gate,
        cu_seqlens=cu_seqlens,
        scale=scale,
    )
    state_ref_vk = _state_kv_to_vk(state_ref_kv)
    verify_atol = 1.0e-2 if dtype is cutlass.BFloat16 else VERIFY_ATOL
    out_hm = out.permute(0, 2, 1, 3)
    torch.testing.assert_close(
        out_hm.float(), out_ref.float(), rtol=VERIFY_RTOL, atol=verify_atol
    )
    if has_state_out:
        torch.testing.assert_close(
            final_state,
            state_ref_vk.to(final_state.dtype),
            rtol=VERIFY_RTOL,
            atol=verify_atol,
        )
    out_diff = (out_hm.float() - out_ref.float()).abs()
    state_diff_text = "disabled"
    if has_state_out:
        state_diff = (final_state.float() - state_ref_vk.float()).abs()
        state_diff_text = f"{state_diff.max().item():.6g}"
    print(
        "Run KDA staged-SMEM/TMA dtype-rounded MMA reference check: PASS "
        f"max_out_abs={out_diff.max().item():.6g}, "
        f"max_state_abs={state_diff_text}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BT=16 KDA CUTLASS primitives kernel",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--heads", type=int, default=DEFAULT_HEADS)
    parser.add_argument("--seqlen", type=int, default=BT)
    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE)
    parser.add_argument("--dtype", choices=sorted(CLI_DTYPES), default="bf16")
    parser.add_argument(
        "--state-dtype",
        choices=sorted(CLI_STATE_DTYPES),
        default="fp32",
    )
    parser.add_argument(
        "--gate-dtype",
        choices=sorted(CLI_GATE_DTYPES),
        default="bf16",
        help="gate input ABI dtype (bf16 = FlashKDA-aligned default)",
    )
    parser.add_argument(
        "--no-initial-state",
        action="store_true",
        help="initialize the recurrent state to zero inside the kernel",
    )
    parser.add_argument(
        "--no-final-state",
        action="store_true",
        help="skip the final recurrent-state global store",
    )
    parser.add_argument(
        "--disable-safe-gate",
        action="store_true",
        help="interpret raw gate input as a precomputed log2-domain increment",
    )
    parser.add_argument(
        "--run-kernel",
        action="store_true",
        help="run the current staged-SMEM/TMA device path after compile",
    )
    parser.add_argument(
        "--mode",
        choices=("engine", "decomp", "auto"),
        default="engine",
        help="engine = plain kernel; decomp = two-kernel decomposition; "
        "auto = two-way shape router (plain engine / decomp-DV2)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    verify(
        batch=args.batch,
        heads=args.heads,
        seqlen=args.seqlen,
        dtype=CLI_DTYPES[args.dtype],
        state_dtype=CLI_STATE_DTYPES[args.state_dtype],
        gate_dtype=CLI_GATE_DTYPES[args.gate_dtype],
        compile_only=not args.run_kernel,
        safe_gate=not args.disable_safe_gate,
        has_state_in=not args.no_initial_state,
        has_state_out=not args.no_final_state,
        scale=args.scale,
        mode=args.mode,
    )
