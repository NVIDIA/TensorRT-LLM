# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Primitive-level tcgen05 MXF8 projection GEMM for GLM-5.2 decode rows (M <= 64; with
``rows_max`` 128 / 192, the MTP3 / MTP5 decode rows).

``C[M, N] = (A_q * SFA) @ (W_q * SFW)^T`` with e4m3 operands and e8m0 scales per 32
elements of K (tcgen05 ``kind::mxf8f6f4`` block scaling), FP32 accumulation, BF16 out.
A is the decode rows' activation quantized by ``self_attn_prefill_act_quant_mxf8``
(blocked scale layout, rows padded to 128); W is the projection weight in the form the
context path already keeps resident (``self_attn_prefill_projections_mxf8.QuantizedWeight``:
e4m3 ``[N, K]`` and the same blocked scale layout over N).  At decode sizes the BF16
projections are weight-read bound, so reading half the bytes is the point.

Wide decode rows (``Mxf8ProjectionConfig.rows_max`` 128 or 192, split-K 1): the activation
box carries all ``rows_max`` rows; each 128-row block of them is its own ``M=128`` MMA
into its own TMEM accumulator (``m_blocks`` = 1 or 2) with its own scale atom, issued
in the same K order, and all four epilogue warps drain every block.  At ``rows_max`` 64
(the default) the kernel is the M <= 64 one, unchanged.

Schedule (one CTA per ``mma_n`` output columns and K slice; split-K ranks form a
cluster along z and reduce through DSMEM, as ``_projection_tcgen05_template``):

* warp 0 (TMA): before ``griddepcontrol.wait`` it issues every pipeline stage's weight
  and weight-scale boxes (they do not depend on the predecessor grid) and prefetches
  the rest of its weight slice into L2; after the wait, the activation and its scales.
* warp 1 (MMA): per K tile, S2T copies of the two scale atoms into TMEM, then four
  ``M=128, N=mma_n, K=32`` block-scaled MMAs.
* warps 2-5 (epilogue): the activation occupies MMA rows 0..63 (the TMA box is 128
  rows; rows past the scratch buffer's 64 are zero-filled), i.e. TMEM lanes 0..63 --
  subpartitions 0 and 1, read by the two warps whose ``warp_idx % 4`` is 0 or 1.

The row count is a runtime argument (the activation scratch always holds 64 rows; rows
at or past ``rows`` are computed from stale bytes and never stored), so one compiled
instance serves every decode bucket.  Pointer arguments only: CUDA-graph safe.
"""

from collections.abc import Callable
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda_exp
from cuda.bindings import driver as cuda
from cutlass.base_dsl.array import L1EvictKind
from cutlass.experimental import primitives as prims

from ..common import compat48 as _compat48

MMA_M = 128  # block-scaled tcgen05 cta_group::1 M
TILE_K = 128  # e4m3 elements (= bytes) of K per pipeline stage: 128 B rows -> s128b swizzle
INST_K = 32  # K of one MXF8 MMA instruction
KBLOCKS = TILE_K // INST_K  # 4
SF_VEC = 32
SF_ATOM_BYTES = 512  # one blocked e8m0 atom: 128 rows x 4 scale columns (128 of K)
SF_ATOM_ROWS = 128
SF_TMEM_COLS = 4  # TMEM columns of one S2T-copied scale atom
ROWS_MAX = 64  # decode rows: TMEM lanes 0..63
SWIZZLED_SMEM_ALIGNMENT = 1024

WARP_THREADS = 32
TMA_WARP = 0
MMA_WARP = 1
EPI_WARP0 = 2
EPI_WARPS = 4
WARPS_PER_CTA = EPI_WARP0 + EPI_WARPS  # 6
THREADS = WARPS_PER_CTA * WARP_THREADS
OWNER_TOTAL = ROWS_MAX  # one epilogue thread per activation row
SMEM_LIMIT_BYTES = 227 * 1024

StorePlan = tuple[tuple[int, int], ...]


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _store_plan(chunk: int) -> StorePlan:
    # One STG carries at most 32 B: 16 BF16.
    plan: list[tuple[int, int]] = []
    pos = 0
    while pos < chunk:
        rem = chunk - pos
        width = next((w for w in (16, 8, 4) if rem >= w), 4)
        plan.append((pos, width))
        pos += width
    return tuple(plan)


@dataclass(frozen=True)
class Mxf8ProjectionConfig:
    module_name: str
    n: int
    k: int
    mma_n: int
    stages: int
    split_k: int
    # Weight K tiles prefetched into L2 at kernel entry (before barrier init and the PDL
    # wait): "rest" = those past the SMEM stages, "all" = every tile of the CTA's slice.
    prefetch_w: str = "all"
    # TMEM scale columns per stage (True) or one set reused by every K tile (False).
    sf_per_stage: bool = False
    # Activation rows the TMA box carries: 128 (own 16 KB per stage) or 64 (stage packed as
    # [A 64 rows | W]: the MMA's rows 64..127 read the stage's weight bytes, which only feed
    # accumulator rows that are never stored), freeing SMEM for more stages.
    a_box_rows: int = 64
    # 128-K boxes per pipeline stage (1 or 2): two halve the per-stage waits, commits and
    # TMA issues per byte.
    k_sub: int = 2
    kernel_name_prefix: str = ""
    # Activation rows the operand scratch holds (the TMA's global row count): 64 for the
    # M <= 64 decode rows, 128 / 192 for the MTP3 / MTP5 decode rows (``a_box_rows`` then
    # equals it and ``split_k`` is 1).
    rows_max: int = ROWS_MAX

    def __post_init__(self) -> None:
        if self.rows_max not in (ROWS_MAX, 128, 192):
            raise ValueError(
                f"{self.module_name}: rows_max must be {ROWS_MAX}, 128 or 192, got {self.rows_max}"
            )
        if self.wide and (self.a_box_rows != self.rows_max or self.split_k != 1):
            raise ValueError(
                f"{self.module_name}: rows_max={self.rows_max} needs a_box_rows={self.rows_max} "
                f"and split_k=1, got {self.a_box_rows} / {self.split_k}"
            )
        if not self.wide and self.a_box_rows not in (64, 128):
            raise ValueError(
                f"{self.module_name}: a_box_rows must be 64 or 128, got {self.a_box_rows}"
            )
        if self.prefetch_w not in ("none", "rest", "all"):
            raise ValueError(
                f"{self.module_name}: prefetch_w must be none/rest/all, got {self.prefetch_w!r}"
            )
        if self.k_sub not in (1, 2):
            raise ValueError(f"{self.module_name}: k_sub must be 1 or 2, got {self.k_sub}")
        if self.local_k_steps % self.k_sub:
            raise ValueError(
                f"{self.module_name}: per-rank K tiles {self.local_k_steps} not a multiple of k_sub={self.k_sub}"
            )
        if self.mma_n not in (64, 128):
            raise ValueError(f"{self.module_name}: mma_n must be 64 or 128, got {self.mma_n}")
        if self.n % self.mma_n:
            raise ValueError(
                f"{self.module_name}: N={self.n} must be a multiple of mma_n={self.mma_n}"
            )
        if self.k % (TILE_K * self.split_k):
            raise ValueError(
                f"{self.module_name}: K={self.k} must be a multiple of {TILE_K} x split_k={self.split_k}"
            )
        if not 1 <= self.split_k <= 8:
            raise ValueError(f"{self.module_name}: split_k={self.split_k} must be in [1, 8]")
        if not 1 <= self.stages <= self.local_k_steps // self.k_sub:
            raise ValueError(
                f"{self.module_name}: stages={self.stages} must be in [1, {self.local_k_steps // self.k_sub}]"
            )
        if self.mma_n % self.reducers or self.chunk % 4:
            raise ValueError(
                f"{self.module_name}: mma_n={self.mma_n} does not split over {self.reducers} reducers"
            )
        if self.tmem_cols_alloc > 512:
            raise ValueError(f"{self.module_name}: {self.tmem_cols_alloc} TMEM columns > 512")
        if self.smem_total_bytes > SMEM_LIMIT_BYTES:
            raise ValueError(
                f"{self.module_name}: stages={self.stages} split_k={self.split_k} needs "
                f"{self.smem_total_bytes / 1024:.1f} KB of SMEM (limit {SMEM_LIMIT_BYTES // 1024})"
            )

    @property
    def prefix(self) -> str:
        return self.kernel_name_prefix or f"glm52_{self.module_name.replace('.', '_')}_mxf8"

    @property
    def n_ctas(self) -> int:
        return self.n // self.mma_n

    @property
    def k_tiles(self) -> int:
        return self.k // TILE_K

    @property
    def local_k_steps(self) -> int:
        return self.k_tiles // self.split_k

    @property
    def n_sf_atoms(self) -> int:
        return _ceil_div(self.n, SF_ATOM_ROWS)

    @property
    def tmem_subs(self) -> int:
        return self.mma_n // 32

    @property
    def sf_bufs(self) -> int:
        return self.stages if self.sf_per_stage else 1

    @property
    def wide(self) -> bool:
        """MTP3 / MTP5 decode rows (``rows_max`` above 64)."""
        return self.rows_max > ROWS_MAX

    @property
    def m_blocks(self) -> int:
        """128-row MMA blocks of the activation (one TMEM accumulator and scale atom each)."""
        return _ceil_div(self.rows_max, MMA_M)

    @property
    def owner_warps(self) -> int:
        """Epilogue warps holding activation rows: TMEM subpartitions 0..owner_warps-1."""
        return min(self.rows_max, MMA_M) // WARP_THREADS

    @property
    def tmem_cols_alloc(self) -> int:
        # m_blocks accumulators, then per scale buffer m_blocks activation and one weight
        # scale atom per 128-K box (m_blocks = 1: the M <= 64 kernel's columns).
        cols = (
            self.m_blocks * self.mma_n
            + (self.m_blocks + 1) * SF_TMEM_COLS * self.k_sub * self.sf_bufs
        )
        return 1 << (cols - 1).bit_length()

    @property
    def stage_ab_bytes(self) -> int:
        return (self.a_box_rows + self.mma_n) * TILE_K * self.k_sub

    @property
    def local_stage_steps(self) -> int:
        return self.local_k_steps // self.k_sub

    @property
    def prefetch_lo(self) -> int:
        return {"none": self.local_k_steps, "rest": self.stages * self.k_sub, "all": 0}[
            self.prefetch_w
        ]

    @property
    def reduce_peers(self) -> int:
        return self.split_k - 1

    @property
    def reducers(self) -> int:
        return 1 << (max(self.split_k, 1).bit_length() - 1)

    @property
    def chunk(self) -> int:
        return self.mma_n // self.reducers

    @property
    def mailbox_stride(self) -> int:
        return self.chunk + 4

    @property
    def mailbox_values(self) -> int:
        return (
            1 if self.reduce_peers == 0 else self.reduce_peers * OWNER_TOTAL * self.mailbox_stride
        )

    @property
    def reduce_payload_bytes(self) -> int:
        return self.reduce_peers * OWNER_TOTAL * self.chunk * 4

    @property
    def smem_total_bytes(self) -> int:
        stage = self.stage_ab_bytes + (self.m_blocks + 1) * SF_ATOM_BYTES * self.k_sub
        return self.stages * stage + self.mailbox_values * 4 + 8 * (2 * self.stages + 2) + 4 + 1024

    @property
    def weight_bytes(self) -> int:
        return self.n * self.k + self.n_sf_atoms * SF_ATOM_ROWS * (self.k // SF_VEC)


@cute.kernel
def kernel(
    tma_a: cutlass.GridConstant[cuda_exp.TensorMap],
    tma_w: cutlass.GridConstant[cuda_exp.TensorMap],
    tma_sfa: cutlass.GridConstant[cuda_exp.TensorMap],
    tma_sfw: cutlass.GridConstant[cuda_exp.TensorMap],
    c_ptr: cute.Tensor,
    rows: cutlass.Int32,
    stages: cutlass.Constexpr[int],
    local_k_steps: cutlass.Constexpr[int],
    k_tiles: cutlass.Constexpr[int],
    output_n: cutlass.Constexpr[int],
    mma_n: cutlass.Constexpr[int],
    tmem_subs: cutlass.Constexpr[int],
    tmem_cols_alloc: cutlass.Constexpr[int],
    reduce_peers: cutlass.Constexpr[int],
    reducers: cutlass.Constexpr[int],
    chunk_cols: cutlass.Constexpr[int],
    mailbox_stride: cutlass.Constexpr[int],
    mailbox_values: cutlass.Constexpr[int],
    reduce_payload_bytes: cutlass.Constexpr[int],
    store_plan: cutlass.Constexpr[tuple],
    prefetch_lo: cutlass.Constexpr[int],
    sf_bufs: cutlass.Constexpr[int],
    a_box_rows: cutlass.Constexpr[int],
    k_sub: cutlass.Constexpr[int],
    m_blocks: cutlass.Constexpr[int],
    owner_warps: cutlass.Constexpr[int],
    wide: cutlass.Constexpr[bool],
) -> None:
    # Stage pool: [A_0 .. A_{k_sub-1} | W_0 .. W_{k_sub-1}], one box per 128 of K. With a
    # 64-row A box the MMA's 128-row A operand of chunk j runs on into the next box (rows
    # 64..127 only feed accumulator rows that are never stored); likewise the second
    # 128-row block of a 192-row box (rows 192..255).
    A_BOX: cutlass.Constexpr[int] = a_box_rows * TILE_K
    W_BOX: cutlass.Constexpr[int] = mma_n * TILE_K
    STAGE_AB: cutlass.Constexpr[int] = (A_BOX + W_BOX) * k_sub
    W_OFF: cutlass.Constexpr[int] = A_BOX * k_sub
    # Scale atoms per stage: [block 0: k_sub atoms | block 1: k_sub atoms] of the activation,
    # k_sub atoms of the weight.
    SF_STAGE: cutlass.Constexpr[int] = SF_ATOM_BYTES * k_sub
    SFA_STAGE: cutlass.Constexpr[int] = SF_STAGE * m_blocks
    # A descriptor step between 128-row MMA blocks: 128 rows of 128 B (16-byte units).
    A_MBLOCK_STEP: cutlass.Constexpr[int] = (MMA_M * TILE_K) >> 4
    stage_steps: cutlass.Constexpr[int] = local_k_steps // k_sub
    smem_ab = cutlass.Array(
        cutlass.Int8,
        stages * STAGE_AB,
        space=cutlass.AddressSpace.smem,
        alignment=SWIZZLED_SMEM_ALIGNMENT,
    )
    smem_sfa = cutlass.Array(
        cutlass.Int8, stages * SFA_STAGE, space=cutlass.AddressSpace.smem, alignment=128
    )
    smem_sfw = cutlass.Array(
        cutlass.Int8, stages * SF_STAGE, space=cutlass.AddressSpace.smem, alignment=128
    )
    full_bar = cutlass.Array(cutlass.Int64, stages, space=cutlass.AddressSpace.smem, alignment=8)
    empty_bar = cutlass.Array(cutlass.Int64, stages, space=cutlass.AddressSpace.smem, alignment=8)
    acc_mbar = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    reduce_full = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    # Own storage (not aliased onto the stages): a peer's DSMEM partials may land while
    # this rank's TMA is still filling its stages (see _projection_tcgen05_template).
    reduce_mailbox = cutlass.Array(
        cutlass.Float32, mailbox_values, space=cutlass.AddressSpace.smem, alignment=128
    )
    tmem_ptr_i32 = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem, alignment=4)

    warp_idx = cute.arch.warp_idx()
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    lane = tidx % cutlass.Int32(WARP_THREADS)
    cta_rank = cute.arch.block_idx_in_cluster()

    coord_n = bidx * cutlass.Int32(mma_n)
    # Scale atoms cover 128 rows; a 64-column tile reads half of its atom's TMEM columns.
    n_atom = coord_n // cutlass.Int32(SF_ATOM_ROWS)
    sfw_col_offset = cutlass.Int32(0)
    if cutlass.const_expr(mma_n < SF_ATOM_ROWS):
        per_atom: cutlass.Constexpr[int] = SF_ATOM_ROWS // mma_n
        sfw_col_offset = (bidx % cutlass.Int32(per_atom)) * cutlass.Int32(SF_TMEM_COLS // per_atom)
    kt_base = cta_rank * cutlass.Int32(local_k_steps)
    sfw_atom_base = n_atom * cutlass.Int32(k_tiles) + kt_base

    # The weight slice (from 128-K tile ``prefetch_lo``) into L2 before anything else: no
    # mbarrier, no predecessor dependency.
    if cutlass.const_expr(prefetch_lo < local_k_steps):
        if warp_idx == TMA_WARP:
            if prims.elect_sync():
                for kk in cutlass.range_constexpr(local_k_steps - prefetch_lo):
                    prims.cp_async_bulk_tensor_prefetch(
                        tma_w.get_ptr(),
                        (
                            (kt_base + cutlass.Int32(prefetch_lo + kk)) * cutlass.Int32(TILE_K),
                            coord_n,
                        ),
                        [],
                    )

    if warp_idx == TMA_WARP:
        if prims.elect_sync():
            for s in cutlass.range_constexpr(stages):
                prims.mbarrier_init(full_bar.subview(s), 1)
                prims.mbarrier_init(empty_bar.subview(s), 1)
            prims.mbarrier_init(acc_mbar, 1)
            prims.mbarrier_init(reduce_full, 1)
    prims.fence_mbarrier_init()
    if cutlass.const_expr(reduce_peers > 0):
        if warp_idx == TMA_WARP:
            if prims.elect_sync():
                prims.mbarrier_arrive_expect_tx(reduce_full, reduce_payload_bytes)
        prims.barrier_cluster_arrive_relaxed()
    prims.barrier_cta_sync(0)

    if warp_idx == TMA_WARP:
        prims.tcgen05_alloc(tmem_ptr_i32, tmem_cols_alloc, group="cta_1")
    prims.barrier_cta_sync(0)
    tmem_base_raw = tmem_ptr_i32.load()
    tmem_ptr = cutlass.inttoptr(tmem_base_raw, 6, cutlass.Float32)
    prims.tcgen05_relinquish_alloc_permit(group="cta_1")

    tx_bytes = (
        (tma_a.global_tx_bytes() + tma_w.global_tx_bytes()) * k_sub
        + tma_sfa.global_tx_bytes() * m_blocks
        + tma_sfw.global_tx_bytes()
    )

    # ---- producer warp ------------------------------------------------------
    if warp_idx == TMA_WARP:
        a_desc = tma_a.get_ptr()
        w_desc = tma_w.get_ptr()
        sfa_desc = tma_sfa.get_ptr()
        sfw_desc = tma_sfw.get_ptr()
        # Weights first: every stage's weight and weight-scale boxes cross the PDL wait.
        for s in cutlass.range_constexpr(stages):
            if prims.elect_sync():
                kt = kt_base + cutlass.Int32(s * k_sub)
                prims.mbarrier_arrive_expect_tx(full_bar.subview(s), tx_bytes)
                for j in cutlass.range_constexpr(k_sub):
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        smem_ab.subview(s * STAGE_AB + W_OFF + j * W_BOX),
                        w_desc,
                        ((kt + cutlass.Int32(j)) * cutlass.Int32(TILE_K), coord_n),
                        full_bar.subview(s),
                    )
                prims.cp_async_bulk_tensor_shared_cta_global(
                    smem_sfw.subview(s * SF_STAGE),
                    sfw_desc,
                    (cutlass.Int32(0), sfw_atom_base + cutlass.Int32(s * k_sub)),
                    full_bar.subview(s),
                )
        if prims.elect_sync():
            cute.arch.griddepcontrol_wait()
            for s in cutlass.range_constexpr(stages):
                kt = kt_base + cutlass.Int32(s * k_sub)
                for j in cutlass.range_constexpr(k_sub):
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        smem_ab.subview(s * STAGE_AB + j * A_BOX),
                        a_desc,
                        (
                            (kt + cutlass.Int32(j)) * cutlass.Int32(TILE_K),
                            cutlass.Int32(0),
                        ),
                        full_bar.subview(s),
                    )
                for mb in cutlass.range_constexpr(m_blocks):
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        smem_sfa.subview(s * SFA_STAGE + mb * SF_STAGE),
                        sfa_desc,
                        (cutlass.Int32(0), kt if mb == 0 else kt + cutlass.Int32(mb * k_tiles)),
                        full_bar.subview(s),
                    )
        if cutlass.const_expr(stage_steps > stages):
            # Stage / phase counters, not ``k % stages``: a non-power-of-two divide per
            # iteration is a long integer sequence on the issue-bound producer.
            stage = cutlass.Int32(0)
            phase = cutlass.Int32(0)
            for k in cutlass.range(stages, stage_steps, 1):
                while not prims.mbarrier_try_wait_parity(
                    empty_bar.subview(stage), phase, time_limit=10_000_000
                ):
                    pass
                if prims.elect_sync():
                    kt = kt_base + k * cutlass.Int32(k_sub)
                    prims.mbarrier_arrive_expect_tx(full_bar.subview(stage), tx_bytes)
                    for j in cutlass.range_constexpr(k_sub):
                        prims.cp_async_bulk_tensor_shared_cta_global(
                            smem_ab.subview(stage * STAGE_AB + W_OFF + j * W_BOX),
                            w_desc,
                            ((kt + cutlass.Int32(j)) * cutlass.Int32(TILE_K), coord_n),
                            full_bar.subview(stage),
                        )
                        prims.cp_async_bulk_tensor_shared_cta_global(
                            smem_ab.subview(stage * STAGE_AB + j * A_BOX),
                            a_desc,
                            (
                                (kt + cutlass.Int32(j)) * cutlass.Int32(TILE_K),
                                cutlass.Int32(0),
                            ),
                            full_bar.subview(stage),
                        )
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        smem_sfw.subview(stage * SF_STAGE),
                        sfw_desc,
                        (cutlass.Int32(0), sfw_atom_base + k * cutlass.Int32(k_sub)),
                        full_bar.subview(stage),
                    )
                    for mb in cutlass.range_constexpr(m_blocks):
                        prims.cp_async_bulk_tensor_shared_cta_global(
                            smem_sfa.subview(stage * SFA_STAGE + mb * SF_STAGE),
                            sfa_desc,
                            (
                                cutlass.Int32(0),
                                kt if mb == 0 else kt + cutlass.Int32(mb * k_tiles),
                            ),
                            full_bar.subview(stage),
                        )
                stage = stage + cutlass.Int32(1)
                if stage == cutlass.Int32(stages):
                    stage = cutlass.Int32(0)
                    phase = phase ^ cutlass.Int32(1)
        # Producer tail: a stage's empty barrier is waited on only to refill it (never when
        # `stage_steps <= stages`), so the MMA warp's last `stages` commit-arrives are
        # awaited here. Nothing orders them against the `acc_mbar` commit the CTA does wait
        # for, and an arrive still in flight at exit lands in the shared memory of the next
        # CTA on this SM.
        for j in cutlass.range_constexpr(max(0, stage_steps - stages), stage_steps):
            while not prims.mbarrier_try_wait_parity(
                empty_bar.subview(j % stages), (j // stages) % 2, time_limit=10_000_000
            ):
                pass

    # ---- MMA warp -------------------------------------------------------------
    if warp_idx == MMA_WARP:
        idesc = prims.Tcgen05MxInstrDesc.build(
            a_dtype=cutlass.Float8E4M3FN,
            b_dtype=cutlass.Float8E4M3FN,
            scale_format=1,  # E8M0
            n_dim=mma_n,
            m_dim=MMA_M,
        )
        idesc_k = [idesc.set_sf_ids(a_sf_id=kb, b_sf_id=kb) for kb in range(KBLOCKS)]
        s2t_shape, s2t_multicast = prims.S2TCopyMode.S2T_32x128b_WARPX4
        base_col = tmem_base_raw & 0xFFFF
        base_row = tmem_base_raw >> 16
        # Scale columns after the m_blocks accumulators: per buffer and 128-K box j, the
        # activation atom of block mb at (mb * k_sub + j), the weight atom at
        # (m_blocks * k_sub + j), 4 columns each.
        sf0 = (base_row << 16) | (base_col + cutlass.Int32(mma_n * m_blocks))
        # Stage-0 descriptors, built once; a stage / chunk / K-block is an offset in 16-byte
        # descriptor units.
        desc_ab0 = _compat48.tcgen05_smem_desc(
            smem_ab,
            leading_byte_offset=16,
            stride_byte_offset=1024,
            version=0,
            base_offset=0,
            layout=2,
        )
        desc_sfa0 = _compat48.tcgen05_smem_desc(
            smem_sfa,
            leading_byte_offset=16,
            stride_byte_offset=128,
            version=0,
            base_offset=0,
            layout=0,
        )
        desc_sfw0 = _compat48.tcgen05_smem_desc(
            smem_sfw,
            leading_byte_offset=16,
            stride_byte_offset=128,
            version=0,
            base_offset=0,
            layout=0,
        )
        stage = cutlass.Int32(0)
        phase = cutlass.Int32(0)
        for k in cutlass.range(stage_steps):
            while not prims.mbarrier_try_wait_parity(
                full_bar.subview(stage), phase, time_limit=10_000_000
            ):
                pass
            if prims.elect_sync():
                off_ab = stage * cutlass.Int32(STAGE_AB >> 4)
                off_sf = stage * cutlass.Int32(SF_STAGE >> 4)
                off_sfa = off_sf
                if cutlass.const_expr(m_blocks > 1):
                    off_sfa = stage * cutlass.Int32(SFA_STAGE >> 4)
                sf_col = sf0
                if cutlass.const_expr(sf_bufs > 1):
                    sf_col = sf0 + stage * cutlass.Int32((m_blocks + 1) * SF_TMEM_COLS * k_sub)
                for j in cutlass.range_constexpr(k_sub):
                    for mb in cutlass.range_constexpr(m_blocks):
                        prims.tcgen05_cp(
                            s2t_shape,
                            cutlass.inttoptr(
                                sf_col + cutlass.Int32((mb * k_sub + j) * SF_TMEM_COLS),
                                6,
                                cutlass.Int32,
                            ),
                            desc_sfa0 + off_sfa + ((mb * k_sub + j) * SF_ATOM_BYTES >> 4),
                            multicast=s2t_multicast,
                        )
                    prims.tcgen05_cp(
                        s2t_shape,
                        cutlass.inttoptr(
                            sf_col + cutlass.Int32((m_blocks * k_sub + j) * SF_TMEM_COLS),
                            6,
                            cutlass.Int32,
                        ),
                        desc_sfw0 + off_sf + (j * SF_ATOM_BYTES >> 4),
                        multicast=s2t_multicast,
                    )
                for j in cutlass.range_constexpr(k_sub):
                    sfw_ptr = cutlass.inttoptr(
                        sf_col
                        + cutlass.Int32((m_blocks * k_sub + j) * SF_TMEM_COLS)
                        + sfw_col_offset,
                        6,
                        cutlass.Int32,
                    )
                    for kb in cutlass.range_constexpr(KBLOCKS):
                        a_step: cutlass.Constexpr[int] = (j * A_BOX >> 4) + (INST_K >> 4) * kb
                        w_step: cutlass.Constexpr[int] = ((W_OFF + j * W_BOX) >> 4) + (
                            INST_K >> 4
                        ) * kb
                        for mb in cutlass.range_constexpr(m_blocks):
                            # Block mb: its rows of the A box, its accumulator, its scales.
                            sfa_ptr = cutlass.inttoptr(
                                sf_col + cutlass.Int32((mb * k_sub + j) * SF_TMEM_COLS),
                                6,
                                cutlass.Int32,
                            )
                            acc_ptr = tmem_ptr
                            if cutlass.const_expr(mb > 0):
                                acc_ptr = cutlass.inttoptr(
                                    tmem_base_raw + cutlass.Int32(mb * mma_n), 6, cutlass.Float32
                                )
                            prims.tcgen05_mma_block_scale(
                                prims.MMABlockScaleKind.MXF8F6F4,
                                prims.CTAGroup.CTA_1,
                                acc_ptr,
                                desc_ab0 + off_ab + (a_step + mb * A_MBLOCK_STEP),
                                desc_ab0 + off_ab + w_step,
                                idesc_k[kb],
                                (k > 0) if (j == 0 and kb == 0) else True,
                                sfa_ptr,
                                sfw_ptr,
                            )
                prims.tcgen05_commit(empty_bar.subview(stage), group=prims.CTAGroup.CTA_1)
            stage = stage + cutlass.Int32(1)
            if stage == cutlass.Int32(stages):
                stage = cutlass.Int32(0)
                phase = phase ^ cutlass.Int32(1)
        if prims.elect_sync():
            prims.tcgen05_commit(acc_mbar, group=prims.CTAGroup.CTA_1)

    prims.barrier_cta_sync(0)
    while not prims.mbarrier_try_wait_parity(acc_mbar, 0, time_limit=10_000_000):
        pass

    sp = warp_idx % cutlass.Int32(4)
    owner_id = sp * cutlass.Int32(WARP_THREADS) + lane
    tmem_base = prims.TmemAddr(tmem_base_raw)
    acc = cutlass.Array(cutlass.Float32, mma_n, alignment=32)
    if cutlass.const_expr(wide):
        # ---- wide rows: every epilogue warp drains its 32 lanes of each 128-row block
        # (row mb * 128 + lane), converts and stores them, then the TMEM dealloc.
        is_owner = (warp_idx >= EPI_WARP0) & (sp < cutlass.Int32(owner_warps))
        if is_owner:
            gC = c_ptr.iterator.raw_ptr()
            for mb in cutlass.range_constexpr(m_blocks):
                for sub in cutlass.range_constexpr(tmem_subs):
                    tsub = prims.TmemAddr.from_row_col(
                        tmem_base.row_id + sp * cutlass.Int32(WARP_THREADS),
                        tmem_base.col_id + (mb * mma_n + sub * 32),
                    ).as_ptr(cutlass.Float32)
                    acc.store(prims.tcgen05_ld("32x32b", tsub, num=32), sub * 32)
                prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                row = owner_id + cutlass.Int32(mb * MMA_M)
                in_range = row < rows
                for off, width in store_plan:
                    col = coord_n + cutlass.Int32(off)
                    out = (
                        acc.load(off, vector_size=width, alignment=16)
                        .to(cutlass.BFloat16)
                        .bitcast(cutlass.Int16)
                    )
                    if in_range:
                        prims.store_ext(
                            out,
                            gC + row * cutlass.Int32(output_n) + col,
                            evict=L1EvictKind.NOALLOCATE,
                        )
        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
        prims.barrier_cta_sync(0)
        if warp_idx == TMA_WARP:
            prims.tcgen05_dealloc(tmem_ptr, tmem_cols_alloc, group="cta_1")
    else:
        # ---- drain TMEM: rows 0..63 = lanes of subpartitions 0 and 1 ------------
        is_owner = (warp_idx >= EPI_WARP0) & (sp < cutlass.Int32(2))
        if is_owner:
            for sub in cutlass.range_constexpr(tmem_subs):
                tsub = prims.TmemAddr.from_row_col(
                    tmem_base.row_id + sp * cutlass.Int32(WARP_THREADS),
                    tmem_base.col_id + sub * 32,
                ).as_ptr(cutlass.Float32)
                acc.store(prims.tcgen05_ld("32x32b", tsub, num=32), sub * 32)
            prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)

        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
        prims.barrier_cta_sync(0)
        if warp_idx == TMA_WARP:
            prims.tcgen05_dealloc(tmem_ptr, tmem_cols_alloc, group="cta_1")

        # ---- DSMEM split-K reduction (as _projection_tcgen05_template) -----------
        if cutlass.const_expr(reduce_peers > 0):
            prims.barrier_cluster_wait()
            if is_owner:
                for tgt in cutlass.range_constexpr(reducers):
                    if cta_rank != cutlass.Int32(tgt):
                        slot = (
                            cta_rank - cutlass.Int32(1)
                            if cta_rank > cutlass.Int32(tgt)
                            else cta_rank
                        )
                        base = slot * cutlass.Int32(
                            OWNER_TOTAL * mailbox_stride
                        ) + owner_id * cutlass.Int32(mailbox_stride)
                        peer_full = prims.mapa(reduce_full, cutlass.Int32(tgt))
                        for ch in cutlass.range_constexpr(chunk_cols // 4):
                            src = tgt * chunk_cols + ch * 4
                            payload = acc.load(src, vector_size=4, alignment=16).bitcast(
                                cutlass.Int32
                            )
                            _compat48.store_async(
                                prims.mapa(
                                    reduce_mailbox.data_ptr() + base + cutlass.Int32(ch * 4),
                                    cutlass.Int32(tgt),
                                ),
                                payload,
                                peer_full,
                            )
            if cta_rank < cutlass.Int32(reducers):
                while not prims.mbarrier_try_wait_parity(reduce_full, 0):
                    pass
                cute.arch.fence_acq_rel_cta()
                if is_owner:
                    for r in cutlass.range_constexpr(reducers):
                        if cta_rank == cutlass.Int32(r):
                            for peer in cutlass.range_constexpr(reduce_peers):
                                base = cutlass.Int32(
                                    peer * OWNER_TOTAL * mailbox_stride
                                ) + owner_id * cutlass.Int32(mailbox_stride)
                                for ch in cutlass.range_constexpr(chunk_cols // 4):
                                    got = reduce_mailbox.subview(base + cutlass.Int32(ch * 4)).load(
                                        vector_size=4, alignment=16
                                    )
                                    for i in cutlass.range_constexpr(4):
                                        dst = r * chunk_cols + ch * 4 + i
                                        acc[dst] = acc[dst] + got[i]

        # ---- each reducer stores its column slice of the valid rows -------------
        if cta_rank < cutlass.Int32(reducers):
            if is_owner:
                gC = c_ptr.iterator.raw_ptr()
                in_range = owner_id < rows
                for r in cutlass.range_constexpr(reducers):
                    if cta_rank == cutlass.Int32(r):
                        for off, width in store_plan:
                            col = coord_n + cutlass.Int32(r * chunk_cols + off)
                            out = (
                                acc.load(r * chunk_cols + off, vector_size=width, alignment=16)
                                .to(cutlass.BFloat16)
                                .bitcast(cutlass.Int16)
                            )
                            if in_range:
                                prims.store_ext(
                                    out,
                                    gC + owner_id * cutlass.Int32(output_n) + col,
                                    evict=L1EvictKind.NOALLOCATE,
                                )

    prims.barrier_cta_sync(0)
    if tidx == 0:
        cute.arch.griddepcontrol_launch_dependents()


@cute.jit
def launch(
    a: cute.Tensor,
    w: cute.Tensor,
    sfa: cute.Tensor,
    sfw: cute.Tensor,
    c: cute.Tensor,
    rows: cutlass.Int32,
    stream: cuda.CUstream,
    k_total: cutlass.Constexpr[int],
    n_total: cutlass.Constexpr[int],
    n_sf_atoms: cutlass.Constexpr[int],
    stages: cutlass.Constexpr[int],
    split_k: cutlass.Constexpr[int],
    local_k_steps: cutlass.Constexpr[int],
    mma_n: cutlass.Constexpr[int],
    n_ctas: cutlass.Constexpr[int],
    tmem_subs: cutlass.Constexpr[int],
    tmem_cols_alloc: cutlass.Constexpr[int],
    reduce_peers: cutlass.Constexpr[int],
    reducers: cutlass.Constexpr[int],
    chunk_cols: cutlass.Constexpr[int],
    mailbox_stride: cutlass.Constexpr[int],
    mailbox_values: cutlass.Constexpr[int],
    reduce_payload_bytes: cutlass.Constexpr[int],
    store_plan: cutlass.Constexpr[tuple],
    prefetch_lo: cutlass.Constexpr[int],
    sf_bufs: cutlass.Constexpr[int],
    a_box_rows: cutlass.Constexpr[int],
    k_sub: cutlass.Constexpr[int],
    rows_max: cutlass.Constexpr[int],
    m_blocks: cutlass.Constexpr[int],
    owner_warps: cutlass.Constexpr[int],
    wide: cutlass.Constexpr[bool],
) -> None:
    k_tiles: cutlass.Constexpr[int] = k_total // TILE_K
    # The activation scratch holds rows_max rows; a 128-row box over 64 zero-fills the rest.
    tma_a = cuda_exp.create_tensor_map_tiled(
        global_address=a.iterator.toint(),
        dtype=cutlass.Float8E4M3FN,
        global_dims=[k_total, rows_max],
        global_strides=[k_total // 16],
        box_dims=[TILE_K, a_box_rows],
        swizzle=cuda_exp.TensorMapSwizzle.s128b,
    )
    tma_w = cuda_exp.create_tensor_map_tiled(
        global_address=w.iterator.toint(),
        dtype=cutlass.Float8E4M3FN,
        global_dims=[k_total, n_total],
        global_strides=[k_total // 16],
        box_dims=[TILE_K, mma_n],
        swizzle=cuda_exp.TensorMapSwizzle.s128b,
    )
    # Blocked scales: atom (row_block, k_tile) is 512 contiguous bytes at
    # (row_block * k_tiles + k_tile) * 512; read as 256 u16 per atom (TMA box <= 256).
    tma_sfa = cuda_exp.create_tensor_map_tiled(
        global_address=sfa.iterator.toint(),
        dtype=cutlass.Uint16,
        global_dims=[SF_ATOM_BYTES // 2, m_blocks * k_tiles],
        global_strides=[SF_ATOM_BYTES // 16],
        box_dims=[SF_ATOM_BYTES // 2, k_sub],
        swizzle=cuda_exp.TensorMapSwizzle.none,
    )
    tma_sfw = cuda_exp.create_tensor_map_tiled(
        global_address=sfw.iterator.toint(),
        dtype=cutlass.Uint16,
        global_dims=[SF_ATOM_BYTES // 2, n_sf_atoms * k_tiles],
        global_strides=[SF_ATOM_BYTES // 16],
        box_dims=[SF_ATOM_BYTES // 2, k_sub],
        swizzle=cuda_exp.TensorMapSwizzle.none,
    )
    kernel(
        tma_a,
        tma_w,
        tma_sfa,
        tma_sfw,
        c,
        rows,
        stages,
        local_k_steps,
        k_tiles,
        n_total,
        mma_n,
        tmem_subs,
        tmem_cols_alloc,
        reduce_peers,
        reducers,
        chunk_cols,
        mailbox_stride,
        mailbox_values,
        reduce_payload_bytes,
        store_plan,
        prefetch_lo,
        sf_bufs,
        a_box_rows,
        k_sub,
        m_blocks,
        owner_warps,
        wide,
    ).launch(
        grid=(n_ctas, 1, split_k),
        cluster=(1, 1, split_k),
        block=(THREADS, 1, 1),
        stream=stream,
        use_pdl=True,
    )


CompiledKernel = Callable[..., object]


def compile_projection(config: Mxf8ProjectionConfig) -> CompiledKernel:
    """``fn(a_q [rows_max, K] u8, w_q [N, K] u8, sfa u8, sfw u8, c [rows, N] bf16, rows)`` on
    the TVM-FFI environment stream.  Build only -- no GPU tensors here."""
    fake = cute.runtime.make_fake_compact_tensor
    kernel.set_name_prefix(config.prefix, remove_cutlass_symbol=True)
    raw = cute.compile(
        launch,
        fake(cutlass.Uint8, (config.rows_max, config.k), stride_order=(1, 0), assumed_align=128),
        fake(cutlass.Uint8, (config.n, config.k), stride_order=(1, 0), assumed_align=128),
        fake(cutlass.Uint8, (cute.sym_int32(),), assumed_align=128),
        fake(cutlass.Uint8, (cute.sym_int32(),), assumed_align=128),
        cute.runtime.make_fake_tensor(
            cutlass.BFloat16,
            (cute.sym_int32(), config.n),
            stride=(config.n, 1),
            assumed_align=16,
        ),
        cutlass.Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        k_total=config.k,
        n_total=config.n,
        n_sf_atoms=config.n_sf_atoms,
        stages=config.stages,
        split_k=config.split_k,
        local_k_steps=config.local_k_steps,
        mma_n=config.mma_n,
        n_ctas=config.n_ctas,
        tmem_subs=config.tmem_subs,
        tmem_cols_alloc=config.tmem_cols_alloc,
        reduce_peers=config.reduce_peers,
        reducers=config.reducers,
        chunk_cols=config.chunk,
        mailbox_stride=config.mailbox_stride,
        mailbox_values=config.mailbox_values,
        reduce_payload_bytes=config.reduce_payload_bytes,
        store_plan=_store_plan(config.chunk),
        prefetch_lo=config.prefetch_lo,
        sf_bufs=config.sf_bufs,
        a_box_rows=config.a_box_rows,
        k_sub=config.k_sub,
        rows_max=config.rows_max,
        m_blocks=config.m_blocks,
        owner_warps=config.owner_warps,
        wide=config.wide,
        options="--enable-tvm-ffi",
    )
    return raw
