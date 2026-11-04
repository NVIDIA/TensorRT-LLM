# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM CuTe DSL projection: self_attn.indexer.pre_indexer_proj for GLM-5.2 decode.

Contract: C[M, 160] = A[M, 6144] @ B[160, 6144].T, no bias, FP32 accumulate; M in
{1, 2, 4, 8, 16, 32, 48, 64} (rows past M compute on zeros, nothing is written).
Split-output entry: columns 0-127 stored as BF16 ``indexer_k``, 128-159 as FP32
``indexer_weight`` (caller-owned, non-aliasing, 16 B aligned). Kernel: one CTA_1
tcgen05 MMA per (M tile, N tile), N tile 16, K split over an 8-CTA cluster with a
DSMEM FP32 reduction, 10-stage TMA pipeline, L2 prefetch of the rank's weight
slice before barrier init and the PDL wait (A is the predecessor's output and
stays behind ``griddepcontrol.wait``). Accuracy contract for both modes: within
atol 0.05 / rtol 0.02 of FP64 truth with no more violations than native TF32.

tf32 vs bf16 mode (``operand_dtype``, default ``"tf32"``): same kernel and host
path; the option selects the A/B element type, the MMA kind (``kind::tf32`` vs
``kind::f16``), ``K_TILE`` (32 vs 64 elements = one 128 B row either way, same
stage bytes) and the prefetch bytes (48 KB vs 24 KB per CTA). Every product is
exact in FP32 in both modes; only the accumulation grouping differs (8 vs 16 K
elements per MMA), so outputs are not bitwise-equal across modes but both meet
the same FP64-truth contract. bf16 mode requires the weight to be exactly
BF16-representable (checked at runtime, ``weight.bfloat16().float() == weight``;
falls back to tf32 otherwise) and the BF16 hidden state as A, so it removes the
FP32 cast of the hidden state from the route.
"""

from collections.abc import Callable
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda_exp
import torch
from cuda.bindings import driver as cuda_driver
from cutlass.cute.runtime import make_fake_compact_tensor
from cutlass.experimental import primitives as prims
from cutlass.experimental.primitives import Tcgen05InstrDesc

from ..common import compat48 as _compat48

MODULE_NAME = "self_attn.indexer.pre_indexer_proj"

# ----------------------------------------------------------------- tile config
N = 160
K = 6144
INDEXER_K_DIM = 128
INDEXER_WEIGHT_DIM = N - INDEXER_K_DIM
MMA_M = 64
MMA_N = 16
K_TILE = 32
PIPE_STAGES = 10
SPLITK_K_PARTS = 8
SPLITK_OUTPUT_STORE_NA = True
# L2 weight prefetch depth in K tiles, issued ahead of barrier init and the PDL
# wait (module docstring): this rank's whole K slice.
PREFETCH_W_K_STEPS = K // (K_TILE * SPLITK_K_PARTS)

THREADS = 128
TMEM_LD_WARPS = 4
TF32_K_GRANULE = 8
# Opt-in BF16 operand path (see the module docstring).  K_TILE_BF16 is twice
# the TF32 K tile so a row stays 128 B (s128b swizzle) and one pipeline stage
# keeps the same byte size; kind::f16 consumes 16 K elements per tcgen05.mma.
TF32_ELEM_BYTES = 4
BF16_ELEM_BYTES = 2
K_TILE_BF16 = 64
BF16_K_GRANULE = 16
PREFETCH_W_K_STEPS_BF16 = K // (K_TILE_BF16 * SPLITK_K_PARTS)
# The operand-dtype option. tf32 (default): FP32 A/B via TFloat32 descriptors,
# kind::tf32, K_TILE 32. bf16: BF16 A/B, kind::f16, K_TILE 64 (same 128 B row,
# same stage bytes, half the prefetch bytes). Products are exact in FP32 in
# both modes; the 8- vs 16-element K grouping per MMA changes the accumulation
# order, so the modes are not bitwise-equal but share one accuracy contract.
# bf16 needs an exactly BF16-representable weight (runtime-checked) and the
# BF16 hidden state, and removes the FP32 cast kernel from the route.
OPERAND_DTYPES = ("tf32", "bf16")
DEFAULT_OPERAND_DTYPE = "tf32"
SPLITK_M_PARTS = 1
SPLITK_B_MCAST = False
SPLITK_CLUSTER_SIZE = SPLITK_M_PARTS * SPLITK_K_PARTS

SMEM_LIMIT_BYTES = 227 * 1024
TMEM_LIMIT_COLS = 512


def _lowest_set_bit(value: int) -> int:
    """Largest power of two dividing ``value``."""
    return value & -value


def _next_pow2(value: int) -> int:
    out = 1
    while out < value:
        out *= 2
    return out


# --------------------------------------------------------- derived constants
# The epilogue tcgen05.ld width. tcgen05_ld takes a power-of-two register count, and
# a chunk that divides MMA_N means no ragged chunk inside an N tile.
SPLITK_CHUNK_COLS = min(32, _lowest_set_bit(MMA_N))
SPLITK_NCHUNK = MMA_N // SPLITK_CHUNK_COLS

# Rank r owns chunks {r, r+P, r+2P, ...} within [0, NCHUNK), so it owns
# BASE + (1 if r < REM else 0) chunks and expects that many x (P-1) peer
# payloads. Both counts are compile-time, so the expected byte count is one of
# exactly two immediates -- no runtime divide, no runtime multiply.
SPLITK_OWNED_BASE, SPLITK_OWNED_REM = divmod(SPLITK_NCHUNK, SPLITK_K_PARTS)
SPLITK_MAX_OWNED = -(-SPLITK_NCHUNK // SPLITK_K_PARTS)
SPLITK_CHUNK_ELEMS = TMEM_LD_WARPS * 16 * SPLITK_CHUNK_COLS
SPLITK_RECV_ELEMS = SPLITK_MAX_OWNED * (SPLITK_K_PARTS - 1) * SPLITK_CHUNK_ELEMS
SPLITK_CHUNK_BYTES = SPLITK_CHUNK_ELEMS * 4
SPLITK_RECV_BYTES = SPLITK_RECV_ELEMS * 4
SPLITK_EXPECT_BYTES_LO = SPLITK_OWNED_BASE * (SPLITK_K_PARTS - 1) * SPLITK_CHUNK_BYTES
SPLITK_EXPECT_BYTES_HI = (SPLITK_OWNED_BASE + 1) * (SPLITK_K_PARTS - 1) * SPLITK_CHUNK_BYTES
SPLITK_TMEM_COLS = _next_pow2(max(32, MMA_N))

SMEM_AB_BYTES = PIPE_STAGES * (MMA_M + MMA_N) * K_TILE * 4
SMEM_MBAR_BYTES = (2 * PIPE_STAGES + 1) * 8 + 8 + 4
SMEM_TOTAL_BYTES = SMEM_AB_BYTES + SMEM_MBAR_BYTES + SPLITK_RECV_BYTES

# --------------------------------------------------------------- legality
# These held only at runtime before; they are cheap and they fail at import.
_OWNED_BY_RANK = tuple(
    SPLITK_OWNED_BASE + (1 if r < SPLITK_OWNED_REM else 0) for r in range(SPLITK_K_PARTS)
)
assert 8 <= MMA_N <= 256 and MMA_N % 8 == 0, "tcgen05 N is a multiple of 8"
assert MMA_M in (64, 128), "this kernel tiles M by 64 or 128 rows"
assert K % K_TILE == 0, "K must be a whole number of K tiles"
assert 1 <= SPLITK_K_PARTS <= K // K_TILE, "split-K cannot exceed the K tiles"
assert SPLITK_CLUSTER_SIZE <= 16, "cluster is capped at 16 CTAs"
assert SPLITK_CHUNK_COLS % 4 == 0, "store_async ships 4 elements at a time"
assert INDEXER_K_DIM % SPLITK_CHUNK_COLS == 0, (
    "the BF16/FP32 split must fall on an epilogue chunk boundary"
)
assert K_TILE % TF32_K_GRANULE == 0, "TF32 MMA needs whole K granules"
assert K_TILE_BF16 % BF16_K_GRANULE == 0, "BF16 MMA needs whole K granules"
assert K_TILE * TF32_ELEM_BYTES == 128 == K_TILE_BF16 * BF16_ELEM_BYTES, (
    "both operand paths need 128-byte rows (s128b swizzle, equal stage bytes)"
)
assert K_TILE // TF32_K_GRANULE == K_TILE_BF16 // BF16_K_GRANULE, (
    "both operand paths issue the same number of MMAs per K tile"
)
assert (K // K_TILE_BF16) % SPLITK_K_PARTS == 0, (
    "the BF16 prefetch assumes every rank owns exactly K/(K_TILE_BF16*P) tiles"
)
assert sum(_OWNED_BY_RANK) == SPLITK_NCHUNK, "ownership must partition the N tile"
assert max(_OWNED_BY_RANK) == SPLITK_MAX_OWNED, "receive buffer is undersized"
assert SPLITK_TMEM_COLS <= TMEM_LIMIT_COLS, f"MMA_N={MMA_N} needs too much TMEM"
assert (K // K_TILE) % SPLITK_K_PARTS == 0, (
    "the prefetch assumes every rank owns exactly K/(K_TILE*P) tiles"
)
assert 0 <= PREFETCH_W_K_STEPS <= K // (K_TILE * SPLITK_K_PARTS), (
    "prefetch depth must be within this rank's K slice"
)
assert SMEM_TOTAL_BYTES <= SMEM_LIMIT_BYTES, (
    f"SMEM overflow: {SMEM_TOTAL_BYTES} B > {SMEM_LIMIT_BYTES} B "
    f"(A+B pipe {SMEM_AB_BYTES} B at PIPE_STAGES={PIPE_STAGES}, "
    f"MMA_M={MMA_M}, MMA_N={MMA_N}; split-K recv {SPLITK_RECV_BYTES} B). "
    f"Max PIPE_STAGES for this tile is "
    f"{(SMEM_LIMIT_BYTES - SMEM_MBAR_BYTES - SPLITK_RECV_BYTES) // ((MMA_M + MMA_N) * K_TILE * 4 + 16)}."
)


def _current_stream(device_index: int) -> cuda_driver.CUstream:
    """Return the active torch stream wrapped for the CuTe runtime."""
    return cuda_driver.CUstream(torch.cuda.current_stream(device_index).cuda_stream)


@dataclass(frozen=True)
class OperandPath:
    """Host-side description of one operand dtype path of the kernel.

    ``bf16_operands`` is the single kernel constexpr that selects the path;
    the remaining fields are what the host needs to build descriptors, fake
    tensors and validators for it (see the module docstring).
    """

    name: str
    bf16_operands: bool
    elem_bytes: int
    k_tile: int
    prefetch_w_k_steps: int
    torch_dtype: torch.dtype


_OPERAND_PATHS = {
    "tf32": OperandPath(
        name="tf32",
        bf16_operands=False,
        elem_bytes=TF32_ELEM_BYTES,
        k_tile=K_TILE,
        prefetch_w_k_steps=PREFETCH_W_K_STEPS,
        torch_dtype=torch.float32,
    ),
    "bf16": OperandPath(
        name="bf16",
        bf16_operands=True,
        elem_bytes=BF16_ELEM_BYTES,
        k_tile=K_TILE_BF16,
        prefetch_w_k_steps=PREFETCH_W_K_STEPS_BF16,
        torch_dtype=torch.bfloat16,
    ),
}


def operand_path(operand_dtype: str) -> OperandPath:
    """Resolve ``operand_dtype`` (one of ``OPERAND_DTYPES``) or raise."""
    if operand_dtype not in _OPERAND_PATHS:
        raise ValueError(f"operand_dtype must be one of {OPERAND_DTYPES}, got {operand_dtype!r}")
    return _OPERAND_PATHS[operand_dtype]


@cute.kernel
def _indexer_splitk_kernel(
    tma_a_desc: cutlass.GridConstant[cuda_exp.TensorMap],
    tma_b_desc: cutlass.GridConstant[cuda_exp.TensorMap],
    c_tensor: cute.Tensor,
    indexer_k_tensor: cute.Tensor,
    indexer_weight_tensor: cute.Tensor,
    split_outputs: cutlass.Constexpr[bool],
    num_k_tiles_total: cutlass.Int64,
    num_stages: cutlass.Constexpr[int],
    prefetch_w_k_steps: cutlass.Constexpr[int],
    bf16_operands: cutlass.Constexpr[bool],
) -> None:
    """CTA_1 GEMM (TF32 or BF16 operands, FP32 accumulate) with cluster split-K
    and DSMEM FP32 reduction.

    K tiles are dealt round-robin (rank r takes tiles r, r+P, r+2P, ...), so
    each rank computes its own tile count and SPLITK_K_PARTS need not divide
    the total.  ``bf16_operands`` selects the operand path (module docstring):
    it changes the SMEM element type, the K tile in elements, the instruction
    descriptor and the MMA kind; every byte quantity is the same on both paths.
    """
    warp_idx = cute.arch.warp_idx()
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    lane = tidx % cutlass.Int32(32)
    cta_rank = cute.arch.block_idx_in_cluster()
    is_tmem_ld_warp = warp_idx < cutlass.Int32(TMEM_LD_WARPS)

    cluster_tile = bidx // cutlass.Int32(SPLITK_CLUSTER_SIZE)
    k_rank = cta_rank % cutlass.Int32(SPLITK_K_PARTS)
    m_rank = cta_rank // cutlass.Int32(SPLITK_K_PARTS)
    tile_m = cluster_tile * cutlass.Int32(SPLITK_M_PARTS) + m_rank
    num_split_k_tiles = (
        num_k_tiles_total - k_rank.to(cutlass.Int64) + cutlass.Int64(SPLITK_K_PARTS - 1)
    ) // cutlass.Int64(SPLITK_K_PARTS)
    coord_m = tile_m * cutlass.Int32(MMA_M)
    coord_n = bidy * cutlass.Int32(MMA_N)

    # Operand path: TF32 rows are 32 x 4 B, BF16 rows 64 x 2 B -- both 128 B,
    # so stage bytes, descriptor offsets and the k-block byte advance agree.
    ab_dtype = cutlass.BFloat16 if cutlass.const_expr(bf16_operands) else cutlass.Float32
    ab_elem_bytes: cutlass.Constexpr[int] = (
        BF16_ELEM_BYTES if cutlass.const_expr(bf16_operands) else TF32_ELEM_BYTES
    )
    k_tile: cutlass.Constexpr[int] = K_TILE_BF16 if cutlass.const_expr(bf16_operands) else K_TILE
    mma_k_granule: cutlass.Constexpr[int] = (
        BF16_K_GRANULE if cutlass.const_expr(bf16_operands) else TF32_K_GRANULE
    )
    stage_a_elems: cutlass.Constexpr[int] = MMA_M * k_tile
    stage_b_elems: cutlass.Constexpr[int] = MMA_N * k_tile
    stage_a_bytes: cutlass.Constexpr[int] = stage_a_elems * ab_elem_bytes
    stage_b_bytes: cutlass.Constexpr[int] = stage_b_elems * ab_elem_bytes
    k_block_offset_bytes: cutlass.Constexpr[int] = mma_k_granule * ab_elem_bytes

    smem_a = cutlass.Array(
        ab_dtype,
        PIPE_STAGES * MMA_M * k_tile,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )
    smem_b = cutlass.Array(
        ab_dtype,
        PIPE_STAGES * MMA_N * k_tile,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )
    smem_a_ptr = smem_a.data_ptr()
    smem_b_ptr = smem_b.data_ptr()
    mbar = cutlass.Array(
        cutlass.Int64,
        2 * PIPE_STAGES + 1,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    full_mbar = mbar
    empty_mbar = mbar.subview(PIPE_STAGES)
    acc_mbar = mbar.subview(2 * PIPE_STAGES)
    recv_accum = cutlass.Array(
        cutlass.Int32,
        SPLITK_RECV_ELEMS,
        space=cutlass.AddressSpace.smem,
        alignment=16,
    )
    recv_mbar = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    tmem_ptr_i32 = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem, alignment=4)

    # ---- L2 weight prefetch: before barrier init, before the PDL wait ------
    # Only the weight (B) descriptor. The A operand is produced by the
    # predecessor grid and must stay behind griddepcontrol.wait below.
    # Same round-robin deal as the k-loop: this rank owns tiles
    # k_rank, k_rank + P, k_rank + 2P, ... The prefetch takes no mbarrier and
    # no completion, so it needs none of the init that follows; everything
    # between here and the first TMA load hides the DRAM round trip.
    if cutlass.const_expr(prefetch_w_k_steps > 0):
        if warp_idx == 0:
            if prims.elect_sync():
                for prefetch_step in cutlass.range_constexpr(prefetch_w_k_steps):
                    prefetch_coord_k = (
                        cutlass.Int64(prefetch_step * SPLITK_K_PARTS) + k_rank.to(cutlass.Int64)
                    ) * cutlass.Int64(k_tile)
                    prims.cp_async_bulk_tensor_prefetch(
                        tma_b_desc.get_ptr(),
                        (prefetch_coord_k, coord_n),
                        [],
                    )

    if warp_idx == 0:
        if prims.elect_sync():
            for stage in cutlass.range_constexpr(PIPE_STAGES):
                prims.mbarrier_init(full_mbar.subview(stage), 1)
                empty_count: cutlass.Constexpr[int] = SPLITK_M_PARTS if SPLITK_B_MCAST else 1
                prims.mbarrier_init(empty_mbar.subview(stage), empty_count)
            prims.mbarrier_init(acc_mbar, 1)
    if tidx == 0:
        prims.mbarrier_init(recv_mbar, 1)
    prims.fence_mbarrier_init()
    prims.barrier_cluster_arrive_relaxed()
    prims.barrier_cluster_wait()
    prims.barrier_cta_sync(0)

    num_tmem_cols: cutlass.Constexpr[int] = SPLITK_TMEM_COLS
    if warp_idx == 0:
        prims.tcgen05_alloc(tmem_ptr_i32, num_tmem_cols, group="cta_1")
    prims.barrier_cta_sync(0)

    tmem_base_raw = tmem_ptr_i32.load()
    tmem_ptr = cutlass.inttoptr(tmem_base_raw, 6, cutlass.Float32)
    if warp_idx == 0:
        prims.tcgen05_relinquish_alloc_permit(group="cta_1")

    idesc = Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=ab_dtype,
        b_dtype=ab_dtype,
        n_dim=MMA_N,
        m_dim=MMA_M,
    )
    smem_swizzle: cutlass.Constexpr = tma_a_desc.swizzle.to(prims.Tcgen05SmemSwizzle)
    # 128 B rows (k_tile x ab_elem_bytes), 8-row core-matrix stride 1024 B on
    # both operand paths.
    desc_a_root = prims.Tcgen05SmemDesc.build(
        start_address=smem_a,
        leading_byte_offset=k_tile * ab_elem_bytes,
        stride_byte_offset=8 * k_tile * ab_elem_bytes,
        layout=smem_swizzle,
    )
    desc_b_root = prims.Tcgen05SmemDesc.build(
        start_address=smem_b,
        leading_byte_offset=k_tile * ab_elem_bytes,
        stride_byte_offset=8 * k_tile * ab_elem_bytes,
        layout=smem_swizzle,
    )
    tma_load_bytes = tma_a_desc.global_tx_bytes() + tma_b_desc.global_tx_bytes()
    b_mcast_mask = cutlass.Int32(1)
    if cutlass.const_expr(SPLITK_B_MCAST):
        b_mcast_mask = (cutlass.Int32(1) << k_rank) + (
            cutlass.Int32(1) << (k_rank + cutlass.Int32(SPLITK_K_PARTS))
        )

    if warp_idx == 0:
        # Only this warp dereferences the input tensor maps. Delay the PDL wait
        # until after CTA/cluster setup, but keep it before the first TMA load.
        if prims.elect_sync():
            prims.griddepcontrol(prims.GridDepAction.WAIT)
        for local_k_tile_idx in cutlass.range(num_split_k_tiles):
            stage = local_k_tile_idx % cutlass.Int64(PIPE_STAGES)
            prod_phase = cutlass.Int32(1) ^ (
                (local_k_tile_idx // cutlass.Int64(PIPE_STAGES)) & cutlass.Int64(1)
            ).to(cutlass.Int32)
            while not prims.mbarrier_try_wait_parity(
                empty_mbar.subview(stage), prod_phase, time_limit=10_000_000
            ):
                pass
            if prims.elect_sync():
                prims.mbarrier_arrive_expect_tx(full_mbar.subview(stage), tma_load_bytes)
                global_k_tile_idx = local_k_tile_idx * cutlass.Int64(
                    SPLITK_K_PARTS
                ) + cutlass.Int64(k_rank)
                coord_k = global_k_tile_idx * cutlass.Int64(k_tile)
                prims.cp_async_bulk_tensor_shared_cta_global(
                    smem_a_ptr + stage * stage_a_elems,
                    tma_a_desc.get_ptr(),
                    (coord_k, coord_m),
                    full_mbar.subview(stage),
                )
                if cutlass.const_expr(SPLITK_B_MCAST):
                    if m_rank == cutlass.Int32(0):
                        prims.cp_async_bulk_tensor_shared_cluster_global(
                            smem_b_ptr + stage * stage_b_elems,
                            tma_b_desc.get_ptr(),
                            (coord_k, coord_n),
                            full_mbar.subview(stage),
                            [],
                            multicast_mask=b_mcast_mask,
                            group=prims.CTAGroup.CTA_1,
                        )
                else:
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        smem_b_ptr + stage * stage_b_elems,
                        tma_b_desc.get_ptr(),
                        (coord_k, coord_n),
                        full_mbar.subview(stage),
                    )

    if warp_idx == 1:
        scale_d = cutlass.Boolean(False)
        for local_k_tile_idx in cutlass.range(num_split_k_tiles):
            stage = local_k_tile_idx % cutlass.Int64(PIPE_STAGES)
            cons_phase = ((local_k_tile_idx // cutlass.Int64(PIPE_STAGES)) & cutlass.Int64(1)).to(
                cutlass.Int32
            )
            while not prims.mbarrier_try_wait_parity(
                full_mbar.subview(stage), cons_phase, time_limit=10_000_000
            ):
                pass

            desc_a_stage = desc_a_root.advance_start_address(cutlass.Int64(stage * stage_a_bytes))
            desc_b_stage = desc_b_root.advance_start_address(cutlass.Int64(stage * stage_b_bytes))
            for k_block_idx in cutlass.range_constexpr(k_tile // mma_k_granule):
                offset_bytes: cutlass.Constexpr[int] = k_block_idx * k_block_offset_bytes
                if prims.elect_sync():
                    prims.tcgen05_mma(
                        prims.Tcgen05MMAKind.F16
                        if cutlass.const_expr(bf16_operands)
                        else prims.Tcgen05MMAKind.TF32,
                        prims.CTAGroup.CTA_1,
                        tmem_ptr,
                        desc_a_stage.advance_start_address(offset_bytes),
                        desc_b_stage.advance_start_address(offset_bytes),
                        idesc,
                        scale_d,
                    )
                scale_d = cutlass.Boolean(True)

            if prims.elect_sync():
                if cutlass.const_expr(SPLITK_B_MCAST):
                    prims.tcgen05_commit(
                        empty_mbar.subview(stage),
                        multicast_mask=b_mcast_mask,
                        group=prims.CTAGroup.CTA_1,
                    )
                else:
                    prims.tcgen05_commit(empty_mbar.subview(stage), group=prims.CTAGroup.CTA_1)

        if prims.elect_sync():
            prims.tcgen05_commit(acc_mbar, group=prims.CTAGroup.CTA_1)

    prims.barrier_cta_sync(0)
    while not prims.mbarrier_try_wait_parity(acc_mbar, 0, time_limit=10_000_000):
        pass

    # ---------------------------------------------------------------- epilogue
    # Generic split-K reduction over an arbitrary MMA_N and an arbitrary number
    # of K partitions.
    #
    # The N tile is cut into SPLITK_NCHUNK chunks of SPLITK_CHUNK_COLS columns.
    # SPLITK_CHUNK_COLS is the tcgen05.ld width, chosen from MMA_N (see the module
    # header) so that it always divides MMA_N -- no ragged chunk inside a tile.
    # Chunk c is owned by cluster K-rank ``c % SPLITK_K_PARTS``; every other
    # K-rank ships its partial sum for c to that owner over DSMEM.
    #
    # Two passes, because an owner must not wait until every sender has sent:
    #   pass 1  send every chunk I do not own
    #   pass 2  wait once, then reduce + store every chunk I do own
    # Only one 32-bit vector is live at a time, so register pressure does not
    # grow with SPLITK_NCHUNK.
    #
    # Rank r owns SPLITK_OWNED_BASE chunks, plus one more if r < OWNED_REM.
    # Both byte counts are compile-time immediates; the only runtime work left
    # is a single compare against a constant. A rank owns nothing iff BASE == 0
    # and r >= REM, and must then neither arrive on nor wait for recv_mbar.
    owns_extra = k_rank < cutlass.Int32(SPLITK_OWNED_REM)

    # Thread 0 of an owning rank posts the expected byte count and arrives
    # once on recv_mbar.
    if tidx == 0:
        if cutlass.const_expr(SPLITK_OWNED_REM == 0):
            if cutlass.const_expr(SPLITK_EXPECT_BYTES_LO > 0):
                prims.mbarrier_expect_tx(recv_mbar, SPLITK_EXPECT_BYTES_LO)
                prims.mbarrier_arrive(recv_mbar)
        else:
            if owns_extra:
                prims.mbarrier_expect_tx(recv_mbar, SPLITK_EXPECT_BYTES_HI)
                prims.mbarrier_arrive(recv_mbar)
            else:
                if cutlass.const_expr(SPLITK_EXPECT_BYTES_LO > 0):
                    prims.mbarrier_expect_tx(recv_mbar, SPLITK_EXPECT_BYTES_LO)
                    prims.mbarrier_arrive(recv_mbar)
    prims.barrier_cta_sync(0)

    if is_tmem_ld_warp:
        tmem_base = prims.TmemAddr(tmem_base_raw)
        base_col = tmem_base.col_id
        tmem_row = tmem_base.row_id
        row_in_tile = warp_idx * cutlass.Int32(16) + lane
        should_store = lane < cutlass.Int32(16)
        g_c = c_tensor.iterator.raw_ptr()
        g_indexer_k = indexer_k_tensor.iterator.raw_ptr()
        g_indexer_weight = indexer_weight_tensor.iterator.raw_ptr()
        n_stride = c_tensor.shape[1]
        indexer_k_stride = indexer_k_tensor.shape[1]
        indexer_weight_stride = indexer_weight_tensor.shape[1]
        recv_ptr = recv_accum.data_ptr()
        row_payload_offset = row_in_tile * cutlass.Int32(SPLITK_CHUNK_COLS)
        row_out = coord_m + row_in_tile

        # ---- pass 1: ship every chunk I do not own to its owner -----------
        for chunk in cutlass.range_constexpr(SPLITK_NCHUNK):
            owner: cutlass.Constexpr[int] = chunk % SPLITK_K_PARTS
            owned_slot: cutlass.Constexpr[int] = chunk // SPLITK_K_PARTS
            if k_rank != cutlass.Int32(owner):
                tmem_sub = prims.TmemAddr.from_row_col(
                    tmem_row, base_col + chunk * SPLITK_CHUNK_COLS
                ).as_ptr(cutlass.Float32)
                vec = prims.tcgen05_ld("32x32b", tmem_sub, num=SPLITK_CHUNK_COLS)
                prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                if should_store:
                    # Senders for one chunk take slots 0..K_PARTS-2 in the
                    # owner's receive buffer, ordered by distance from the
                    # owner so the mapping is a bijection.
                    slot = (k_rank + cutlass.Int32(SPLITK_K_PARTS - owner - 1)) % cutlass.Int32(
                        SPLITK_K_PARTS
                    )
                    dst = (
                        cutlass.Int32(owned_slot * (SPLITK_K_PARTS - 1) * SPLITK_CHUNK_ELEMS)
                        + slot * cutlass.Int32(SPLITK_CHUNK_ELEMS)
                        + row_payload_offset
                    )
                    owner_cta = m_rank * cutlass.Int32(SPLITK_K_PARTS) + cutlass.Int32(owner)
                    remote_recv = prims.mapa(recv_ptr + dst, owner_cta)
                    remote_mbar = prims.mapa(recv_mbar, owner_cta)
                    for elem_base in cutlass.range_constexpr(0, SPLITK_CHUNK_COLS, 4):
                        _compat48.store_async(
                            remote_recv + cutlass.Int32(elem_base),
                            vec[elem_base : elem_base + 4].bitcast(cutlass.Int32),
                            remote_mbar,
                        )

        # ---- pass 2: reduce + store every chunk I do own ------------------
        # Owns something iff BASE > 0 (constexpr: every rank owns) or
        # k_rank < REM. When BASE > 0 this folds away entirely.
        owns_any = (
            cutlass.Boolean(True) if cutlass.const_expr(SPLITK_OWNED_BASE > 0) else owns_extra
        )
        if owns_any:
            # THE WAIT MUST NOT BE PREDICATED ON should_store.
            # should_store is `lane < 16`, so guarding the wait with it parks
            # lanes 0-15 in the spin loop while lanes 16-31 run ahead into the
            # tcgen05_ld below -- and tcgen05_ld is a warp-collective. Executing
            # it under intra-warp divergence corrupts the TMEM state, which
            # surfaces later as a TMEM dealloc error.
            # owned_count is CTA-uniform, so every lane of every warp waits
            # together here and the warp re-converges before the collective.
            while not prims.mbarrier_try_wait_parity(recv_mbar, 0, time_limit=10_000_000):
                pass
            for chunk in cutlass.range_constexpr(SPLITK_NCHUNK):
                owner: cutlass.Constexpr[int] = chunk % SPLITK_K_PARTS
                owned_slot: cutlass.Constexpr[int] = chunk // SPLITK_K_PARTS
                if k_rank == cutlass.Int32(owner):
                    tmem_sub = prims.TmemAddr.from_row_col(
                        tmem_row, base_col + chunk * SPLITK_CHUNK_COLS
                    ).as_ptr(cutlass.Float32)
                    acc = prims.tcgen05_ld("32x32b", tmem_sub, num=SPLITK_CHUNK_COLS)
                    prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                    if should_store:
                        for peer_slot in cutlass.range_constexpr(SPLITK_K_PARTS - 1):
                            peer_base = (
                                cutlass.Int32(
                                    (owned_slot * (SPLITK_K_PARTS - 1) + peer_slot)
                                    * SPLITK_CHUNK_ELEMS
                                )
                                + row_payload_offset
                            )
                            peer_bits = (recv_ptr + peer_base).load(
                                count=SPLITK_CHUNK_COLS, alignment=16
                            )
                            acc = acc + peer_bits.bitcast(cutlass.Float32)
                        col = coord_n + cutlass.Int32(chunk * SPLITK_CHUNK_COLS)
                        output_rows = (
                            indexer_k_tensor.shape[0]
                            if cutlass.const_expr(split_outputs)
                            else c_tensor.shape[0]
                        )
                        if row_out < output_rows:
                            if cutlass.const_expr(N % SPLITK_CHUNK_COLS == 0):
                                # Whole-vector fast path: N is a multiple of the
                                # chunk width, so a chunk is either entirely in
                                # range or entirely out.
                                if (col + cutlass.Int32(SPLITK_CHUNK_COLS)) <= cutlass.Int32(N):
                                    if cutlass.const_expr(split_outputs):
                                        if col < cutlass.Int32(INDEXER_K_DIM):
                                            (g_indexer_k + row_out * indexer_k_stride + col).store(
                                                acc.to(cutlass.BFloat16),
                                                alignment=16,
                                            )
                                        else:
                                            weight_col = col - cutlass.Int32(INDEXER_K_DIM)
                                            (
                                                g_indexer_weight
                                                + row_out * indexer_weight_stride
                                                + weight_col
                                            ).store(acc, alignment=16)
                                    else:
                                        if cutlass.const_expr(SPLITK_OUTPUT_STORE_NA):
                                            for elem_base in cutlass.range_constexpr(
                                                0, SPLITK_CHUNK_COLS, 8
                                            ):
                                                (
                                                    g_c
                                                    + row_out * n_stride
                                                    + col
                                                    + cutlass.Int32(elem_base)
                                                ).nvvm_store_ext(
                                                    acc[elem_base : elem_base + 8],
                                                    evict="noallocate",
                                                )
                                        else:
                                            (g_c + row_out * n_stride + col).store(
                                                acc, alignment=16
                                            )
                            else:
                                # Predicated tail: MMA_N need not divide N, so
                                # the final chunk can be partially in range.
                                for elem in cutlass.range_constexpr(SPLITK_CHUNK_COLS):
                                    out_col = col + cutlass.Int32(elem)
                                    if out_col < cutlass.Int32(N):
                                        (g_c + row_out * n_stride + out_col).store(
                                            acc[elem], alignment=4
                                        )

    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    prims.barrier_cta_sync(0)
    # Every CTA must publish completion, including split-K ranks that do not
    # write C. The dependent kernel's WAIT provides global-memory visibility.
    if tidx == 0:
        prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)
    # No cluster barrier is needed here.
    if warp_idx == 0:
        prims.tcgen05_dealloc(tmem_ptr, num_tmem_cols, group="cta_1")


@cute.jit
def _launch_splitk(
    a_tensor: cute.Tensor,
    b_tensor: cute.Tensor,
    c_tensor: cute.Tensor,
    indexer_k_tensor: cute.Tensor,
    indexer_weight_tensor: cute.Tensor,
    split_outputs: cutlass.Constexpr[bool],
    stream: cuda_driver.CUstream,
    prefetch_w_k_steps: cutlass.Constexpr[int],
    bf16_operands: cutlass.Constexpr[bool],
) -> None:
    """Build TMA descriptors and launch the cluster split-K CTA_1 kernel.

    ``bf16_operands`` selects the operand path: BF16 descriptors over 64-element
    K tiles, or ``TFloat32`` descriptors (round-to-nearest-even operand
    rounding, like cuBLAS) over 32-element K tiles.  Both are 128-byte rows.
    """
    tma_dtype = cutlass.BFloat16 if cutlass.const_expr(bf16_operands) else cutlass.TFloat32
    ab_elem_bytes: cutlass.Constexpr[int] = (
        BF16_ELEM_BYTES if cutlass.const_expr(bf16_operands) else TF32_ELEM_BYTES
    )
    k_tile: cutlass.Constexpr[int] = K_TILE_BF16 if cutlass.const_expr(bf16_operands) else K_TILE
    tma_a_desc = cuda_exp.create_tensor_map_tiled(
        global_address=a_tensor.iterator.toint(),
        dtype=tma_dtype,
        global_dims=[a_tensor.shape[1], a_tensor.shape[0]],
        global_strides=[a_tensor.shape[1] * ab_elem_bytes // 16],
        box_dims=[k_tile, MMA_M],
        swizzle=cuda_exp.TensorMapSwizzle.s128b,
    )
    tma_b_desc = cuda_exp.create_tensor_map_tiled(
        global_address=b_tensor.iterator.toint(),
        dtype=tma_dtype,
        global_dims=[b_tensor.shape[1], b_tensor.shape[0]],
        global_strides=[b_tensor.shape[1] * ab_elem_bytes // 16],
        box_dims=[k_tile, MMA_N],
        swizzle=cuda_exp.TensorMapSwizzle.s128b,
    )
    grid_m: cutlass.Constexpr[int] = cute.ceil_div(a_tensor.shape[0], MMA_M * SPLITK_M_PARTS)
    grid_n: cutlass.Constexpr[int] = cute.ceil_div(b_tensor.shape[0], MMA_N)
    num_k_tiles_total = a_tensor.shape[1] // k_tile
    _indexer_splitk_kernel(
        tma_a_desc,
        tma_b_desc,
        c_tensor,
        indexer_k_tensor,
        indexer_weight_tensor,
        split_outputs,
        num_k_tiles_total,
        PIPE_STAGES,
        prefetch_w_k_steps,
        bf16_operands,
    ).launch(
        grid=(grid_m * SPLITK_CLUSTER_SIZE, grid_n, 1),
        block=(THREADS, 1, 1),
        cluster=(SPLITK_CLUSTER_SIZE, 1, 1),
        stream=stream,
        use_pdl=True,
    )


# ---------------------------------------------------------------- public API

CompiledKernel = Callable[
    [
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        cuda_driver.CUstream,
    ],
    object,
]
LinearLauncher = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], object]
# Compile caches are keyed by (active M, operand dtype).
_COMPILED: dict[tuple[int, str], CompiledKernel] = {}


@dataclass(frozen=True)
class CompiledSplitOutputs:
    """A split-output callable bound to its output ABI, active M and operand path."""

    _raw: CompiledKernel
    active_m: int
    split_outputs: bool = True
    operand_dtype: str = DEFAULT_OPERAND_DTYPE


_COMPILED_SPLIT_OUTPUTS: dict[tuple[int, str], CompiledSplitOutputs] = {}


def _ab_fake(shape: tuple[object, ...], path: OperandPath):
    """Fake A / B operand tensor of one operand path (row-major, 16 B aligned)."""
    dtype = cutlass.BFloat16 if path.bf16_operands else cutlass.Float32
    return make_fake_compact_tensor(dtype, shape, stride_order=(1, 0), assumed_align=16)


def _compile_for_m(active_m: int, operand_dtype: str = DEFAULT_OPERAND_DTYPE) -> CompiledKernel:
    """Compile the legacy-output kernel for one M and operand path, memoized.

    ``active_m`` need not be a multiple of ``MMA_M``; see the module docstring.
    """
    if active_m <= 0:
        raise ValueError(f"M must be positive; got {active_m}")
    path = operand_path(operand_dtype)
    cached = _COMPILED.get((active_m, path.name))
    if cached is not None:
        return cached
    compiled = cute.compile(
        _launch_splitk,
        _ab_fake((active_m, K), path),
        _ab_fake((N, K), path),
        make_fake_compact_tensor(
            cutlass.Float32, (active_m, N), stride_order=(1, 0), assumed_align=16
        ),
        make_fake_compact_tensor(
            cutlass.Float32, (active_m, N), stride_order=(1, 0), assumed_align=16
        ),
        make_fake_compact_tensor(
            cutlass.Float32, (active_m, N), stride_order=(1, 0), assumed_align=16
        ),
        False,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        path.prefetch_w_k_steps,
        path.bf16_operands,
        options="--enable-tvm-ffi",
    )
    _COMPILED[(active_m, path.name)] = compiled
    return compiled


def _validate(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    operand_dtype: str = DEFAULT_OPERAND_DTYPE,
) -> None:
    """Check the legacy ``[M, 160]`` launch contract (dtype, CUDA placement, one
    device, row-major contiguity, 16-byte alignment, shapes); called on every
    launch like :func:`_validate_split_outputs`."""
    ab_dtype = operand_path(operand_dtype).torch_dtype
    for name, tensor, dtype in (
        ("input", input_tensor, ab_dtype),
        ("weight", weight, ab_dtype),
        ("output", output, torch.float32),
    ):
        if tensor.dtype != dtype:
            raise TypeError(f"{name} must be {dtype}, got {tensor.dtype}")
        if not tensor.is_cuda:
            raise TypeError(f"{name} must be a CUDA tensor")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be row-major contiguous")
        if tensor.data_ptr() % 16:
            raise ValueError(f"{name} must be 16-byte aligned")
    if not (input_tensor.device == weight.device == output.device):
        raise ValueError("input, weight and output must share one CUDA device")
    m = int(input_tensor.shape[0])
    if tuple(input_tensor.shape) != (m, K):
        raise ValueError(f"expected input[M, {K}], got {tuple(input_tensor.shape)}")
    if tuple(weight.shape) != (N, K):
        raise ValueError(f"expected weight[{N}, {K}], got {tuple(weight.shape)}")
    if tuple(output.shape) != (m, N):
        raise ValueError(f"expected output[{m}, {N}], got {tuple(output.shape)}")


def _device_index(tensor: torch.Tensor) -> int:
    index = tensor.device.index
    return torch.cuda.current_device() if index is None else index


def _compile_split_outputs_for_m(
    active_m: int, operand_dtype: str = DEFAULT_OPERAND_DTYPE
) -> CompiledSplitOutputs:
    """Compile the direct mixed-output specialization for one M and operand path."""
    if active_m <= 0:
        raise ValueError(f"M must be positive; got {active_m}")
    path = operand_path(operand_dtype)
    cached = _COMPILED_SPLIT_OUTPUTS.get((active_m, path.name))
    if cached is not None:
        return cached
    indexer_weight_fake = make_fake_compact_tensor(
        cutlass.Float32,
        (active_m, INDEXER_WEIGHT_DIM),
        stride_order=(1, 0),
        assumed_align=16,
    )
    compiled = cute.compile(
        _launch_splitk,
        _ab_fake((active_m, K), path),
        _ab_fake((N, K), path),
        indexer_weight_fake,
        make_fake_compact_tensor(
            cutlass.BFloat16,
            (active_m, INDEXER_K_DIM),
            stride_order=(1, 0),
            assumed_align=16,
        ),
        indexer_weight_fake,
        True,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        path.prefetch_w_k_steps,
        path.bf16_operands,
        options="--enable-tvm-ffi",
    )
    launcher = CompiledSplitOutputs(compiled, active_m, operand_dtype=path.name)
    _COMPILED_SPLIT_OUTPUTS[(active_m, path.name)] = launcher
    return launcher


def compile_split_outputs(
    active_m: int = MMA_M, operand_dtype: str = DEFAULT_OPERAND_DTYPE
) -> CompiledSplitOutputs:
    """Compile the direct mixed-output specialization of one decode bucket.

    ``active_m`` is the bucket's row count (default: the 64-row bucket); the
    handle is bound to it and :func:`run_split_outputs` enforces the match.
    """
    return _compile_split_outputs_for_m(active_m, operand_dtype)


def _validate_split_outputs(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    indexer_k: torch.Tensor,
    indexer_weight: torch.Tensor,
    operand_dtype: str = DEFAULT_OPERAND_DTYPE,
) -> None:
    """Validate the mixed-dtype split-output tensor contract of one operand path."""
    ab_dtype = operand_path(operand_dtype).torch_dtype
    if input_tensor.dtype != ab_dtype:
        raise TypeError(f"input must be {ab_dtype}, got {input_tensor.dtype}")
    if weight.dtype != ab_dtype:
        raise TypeError(f"weight must be {ab_dtype}, got {weight.dtype}")
    if indexer_k.dtype != torch.bfloat16:
        raise TypeError(f"indexer_k must be bfloat16, got {indexer_k.dtype}")
    if indexer_weight.dtype != torch.float32:
        raise TypeError(f"indexer_weight must be float32, got {indexer_weight.dtype}")
    for name, tensor in (
        ("input", input_tensor),
        ("weight", weight),
        ("indexer_k", indexer_k),
        ("indexer_weight", indexer_weight),
    ):
        if not tensor.is_cuda:
            raise TypeError(f"{name} must be a CUDA tensor")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be row-major contiguous")
        if tensor.data_ptr() % 16:
            raise ValueError(f"{name} must be 16-byte aligned")
    m = int(input_tensor.shape[0])
    expected_shapes = {
        "input": (m, K),
        "weight": (N, K),
        "indexer_k": (m, INDEXER_K_DIM),
        "indexer_weight": (m, INDEXER_WEIGHT_DIM),
    }
    for name, tensor in (
        ("input", input_tensor),
        ("weight", weight),
        ("indexer_k", indexer_k),
        ("indexer_weight", indexer_weight),
    ):
        if tuple(tensor.shape) != expected_shapes[name]:
            raise ValueError(f"expected {name}{expected_shapes[name]}, got {tuple(tensor.shape)}")
    if not (
        input_tensor.device == weight.device
        and input_tensor.device == indexer_k.device
        and input_tensor.device == indexer_weight.device
    ):
        raise ValueError("input, weight, indexer_k, and indexer_weight must share one device")
    tensors = (input_tensor, weight, indexer_k, indexer_weight)
    for left_index, left in enumerate(tensors):
        for right in tensors[left_index + 1 :]:
            if _overlaps(left, right):
                raise ValueError("split-output inputs and outputs must not overlap")


def _byte_interval(tensor: torch.Tensor) -> tuple[int, int, int]:
    """Return storage identity and active byte interval for a compact tensor."""
    storage = tensor.untyped_storage()
    begin = tensor.storage_offset() * tensor.element_size()
    end = begin + tensor.numel() * tensor.element_size()
    return storage.data_ptr(), begin, end


def _overlaps(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Return whether two compact tensors alias any active byte."""
    left_storage, left_begin, left_end = _byte_interval(left)
    right_storage, right_begin, right_end = _byte_interval(right)
    return left_storage == right_storage and max(left_begin, right_begin) < min(left_end, right_end)


def run_split_outputs(
    compiled_fn: CompiledSplitOutputs,
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    indexer_k: torch.Tensor,
    indexer_weight: torch.Tensor,
) -> None:
    """Run the precompiled split-output specialization of the handle's bucket.

    The operand path is the handle's: the validator requires A / B in the dtype
    the handle was compiled for (FP32 for ``"tf32"``, BF16 for ``"bf16"``).
    The row count must equal the handle's ``active_m``: the caller selects the
    handle from the route's bucket, so a mismatch is a routing fault, not a
    shape to adapt to.
    """
    if not isinstance(compiled_fn, CompiledSplitOutputs):
        raise TypeError("compiled_fn must be a CompiledSplitOutputs handle")
    if not compiled_fn.split_outputs:
        raise ValueError(
            "compiled split-output specialization mismatch: "
            f"split_outputs={compiled_fn.split_outputs}, "
            f"active_m={compiled_fn.active_m}"
        )
    if input_tensor.shape[0] != compiled_fn.active_m:
        raise ValueError(
            "run_split_outputs handle is compiled for "
            f"M={compiled_fn.active_m}, got M={int(input_tensor.shape[0])}"
        )
    _validate_split_outputs(
        input_tensor, weight, indexer_k, indexer_weight, compiled_fn.operand_dtype
    )
    compiled_fn._raw(
        input_tensor,
        weight,
        indexer_weight,
        indexer_k,
        indexer_weight,
        _current_stream(_device_index(input_tensor)),
    )


def build(operand_dtype: str = DEFAULT_OPERAND_DTYPE) -> LinearLauncher:
    """Return a launcher pinned to the first M it sees.

    The launch contract (:func:`_validate`) is checked on every call; the M pin
    is checked against the compiled cell.  ``operand_dtype`` selects the operand
    path once, at build time.
    """
    operand_path(operand_dtype)
    compiled_fn: CompiledKernel | None = None
    pinned_m = -1

    def run_linear(
        input_tensor: torch.Tensor,
        weight: torch.Tensor,
        output: torch.Tensor,
    ) -> object:
        nonlocal compiled_fn, pinned_m
        _validate(input_tensor, weight, output, operand_dtype)
        if compiled_fn is None:
            active_m = int(input_tensor.shape[0])
            cutlass.cuda.initialize_cuda_context(_device_index(input_tensor))
            compiled_fn = _compile_for_m(active_m, operand_dtype)
            pinned_m = active_m
        elif input_tensor.shape[0] != pinned_m:
            raise ValueError(
                f"{MODULE_NAME}: this closure is pinned to M={pinned_m}; got "
                f"M={int(input_tensor.shape[0])}. Build one closure per M bucket."
            )
        return compiled_fn(
            input_tensor,
            weight,
            output,
            output,
            output,
            _current_stream(_device_index(input_tensor)),
        )

    return run_linear
