# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Derived from low_m_bf16_splitk.py in this directory (adapted from FlashInfer
# flashinfer/gemm/kernels/dense_bf16_gemm_sm100_splitk.py, Apache-2.0): the warp roles, the
# cluster split-K and the DSMEM reduction are the same. The block-scaled MMA, the scale-factor
# shared/tensor-memory layouts and the S2T copies follow dense_blockscaled_gemm_persistent.py.
"""SM10x low-M FP8 block-scaled GEMM on DeepGEMM 1d1d operands, with the weights loaded before
the PDL wait.

``out[M, N] = (A * sa)[M, K] @ ((W * sw)[N, K])^T`` in BF16 for decode row counts (M <= 32):

- ``A``: E4M3 ``[M, K]`` with one UE8M0 scale per row and 128 K values, packed four per int32 in
  DeepGEMM's MN-major layout (``[M, ceil(K / 512)]``, strides ``(1, ceil4(M))``), as written by
  ``fp8_quantize_1x128_packed_ue8m0``, the fused MNNVL all-reduce quant and the fused
  attn_res / gate quantizers;
- ``W``: E4M3 ``[N, K]`` with its 128x128 block scales in DeepGEMM's per-row packed int32
  MN-major layout (``transform_sf_into_required_layout(..., recipe=(1, 128, 128),
  is_sfa=False)``).

The MMA is tcgen05 ``kind::mxf8f6f4.block_scale`` (one UE8M0 scale per 32 K values). Each
128-K block's scale is written into all four of its 32-K slots. DeepGEMM 1d1d applies that same
scale to each of its four 32-K MMAs (it selects the block's byte with ``sf_id``). So with
``split_k == 1`` the MMA sequence, the FP32 accumulation order and the BF16 rounding are
DeepGEMM's. With ``split_k > 1`` the K slices are summed in a fixed order by cluster rank 0:
deterministic, and equal to DeepGEMM up to FP32 summation order.

One CTA owns a 128-row weight tile and one K slice. ``split_k`` CTAs of a cluster along grid-y
cover the K range (an uneven split gives the first ``K_tiles % split_k`` ranks one extra tile)
and reduce into rank 0 through DSMEM. Warp roles:

- warp 0: weight TMA loads, issued WITHOUT ``griddepcontrol.wait``. Under PDL the weight stream
  overlaps the predecessor's tail, up to ``ab_stages`` 16 KB tiles per CTA;
- warp 1: activation TMA loads, after ``griddepcontrol.wait``;
- warp 2: TMEM allocation, the scale-factor S2T copies and the block-scaled MMAs;
- warp 3: scale factors, once per CTA rather than per pipeline stage. Before the wait it writes
  the MXF8 scale tiles of the weights for the CTA's whole K slice; after the wait, those of the
  activations. Warp 2 copies each set into TMEM in one go (one 4x32dp128bit copy per K tile), so
  the K loop issues only MMAs (writing the scales per pipeline stage would put the scale-warp work
  on the critical path of every 128-K tile);
- warps 4-7: epilogue (TMEM -> registers -> cluster reduction -> BF16 store).

TMEM: the accumulator in columns [0, 32), then four columns per K tile for the weight scales, then
four per K tile for the activation scales. A CTA's K slice is therefore at most 56 tiles
(32 + 8 * 56 <= 512 columns); the kernel is compiled per (tactic, tiles per CTA).
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Dict, Optional, Tuple

import cuda.bindings.driver as _cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import torch as _torch
from cutlass import Int32
from cutlass.cute import experimental as cute_ext
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.runtime import make_ptr

from ..rubin.k3_lamport import mailbox_store_v4, pack_bf16x2, sanitize_bf16x2
from .low_m_bf16_splitk import OWNER_RANK, _store_shared_remote_v4

#: Per-CTA shared-memory capacity reported by CuTe DSL on SM100/SM103.
_SMEM_CAPACITY_BYTES = utils.get_smem_capacity_in_bytes("sm_100")

#: Shared memory this kernel leaves free so that a CTA of the predecessor (the MNNVL all-reduce
#: uses ~160 B of static shared memory plus the 1 KB per-block reservation) can stay resident
#: on the same SM while the weights stream in.
_SMEM_HEADROOM_BYTES = 4 * 1024

#: Weight rows per CTA = the block-scaled MMA M (cta_group 1 supports only M = 128).
_CTA_M = 128

#: K extent of one pipeline stage = one 128x128 weight-scale block.
_CTA_K = 128

#: Scale-factor vector size of the MXF8 MMA.
_SF_VEC = 32

#: Kernel-N carries public M.
_SUPPORTED_MMA_N = (8, 16, 32)

#: Largest public M this low-M path serves (DeepGEMM above it).
MAX_M = 32

#: Cluster CTAs along K (portable cluster size limit).
_MAX_SPLIT_K = 8

#: K tiles of one CTA's K slice (their scales all stay in TMEM: 32 + 8 * 56 <= 512 columns).
_MAX_K_TILES = 56

_MIN_AB_STAGES = 2
_MAX_AB_STAGES = 16

_FP32_BYTES = 4
_MBARRIER_BYTES = 8
_TMEM_POINTER_BYTES = 4
_AB_BUFFER_ALIGN_BYTES = 1024
_SF_BUFFER_ALIGN_BYTES = 128
_MAILBOX_ALIGN_BYTES = 128

#: MXF8 scale factors of one 128-row operand tile and one 128-K tile: 128 rows x four 32-K slots.
_SF_TILE_BYTES = 512

#: TMEM columns of the FP32 accumulator (MMA N <= 32); the scale factors follow it.
_ACC_TMEM_COLS = 32

#: TMEM columns per 128-K tile of one operand's scale factors (4x32dp128bit copy).
_SF_TMEM_COLS_PER_TILE = 4

#: UE8M0 1.0 in every byte: the scale of rows past M or N (their operand data is zero).
_SF_PAD_WORD = 0x7F7F7F7F

#: Barrier arrivals per full stage: the weight TMA and the activation TMA.
_FULL_ARRIVALS = 2


@dataclasses.dataclass(frozen=True)
class Fp8SplitKTactic:
    """One specialization: ``mma_n`` carries public M; the weight tile is always 128 rows."""

    mma_n: int
    split_k: int
    ab_stages: int


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _tmem_cols(k_tiles: int) -> int:
    """TMEM columns to allocate (a power of two >= 32): accumulator + both operands' scales."""
    need = _ACC_TMEM_COLS + 2 * _SF_TMEM_COLS_PER_TILE * k_tiles
    return max(32, 1 << (need - 1).bit_length())


def _sf_words(k_tiles: int) -> int:
    """Packed scale words per row a K slice of ``k_tiles`` tiles can straddle (any start offset)."""
    return (k_tiles - 1) // 4 + 2


def _smem_bytes(tactic: Fp8SplitKTactic, ab_stages: int, k_tiles: int) -> int:
    """Mirror the order and alignment of the kernel's shared-memory allocations."""
    cursor = _align_up(_CTA_M * _CTA_K * ab_stages, _AB_BUFFER_ALIGN_BYTES)
    cursor = _align_up(cursor, _AB_BUFFER_ALIGN_BYTES) + tactic.mma_n * _CTA_K * ab_stages
    cursor = _align_up(cursor, _SF_BUFFER_ALIGN_BYTES) + _SF_TILE_BYTES * k_tiles
    cursor = _align_up(cursor, _SF_BUFFER_ALIGN_BYTES) + _SF_TILE_BYTES * k_tiles
    cursor = _align_up(cursor, _MBARRIER_BYTES) + 2 * ab_stages * _MBARRIER_BYTES
    cursor += 4 * _MBARRIER_BYTES
    cursor = _align_up(cursor, _TMEM_POINTER_BYTES) + _TMEM_POINTER_BYTES
    if tactic.split_k > 1:
        cursor = _align_up(cursor, _MAILBOX_ALIGN_BYTES)
        cursor += (tactic.split_k - 1) * _CTA_M * tactic.mma_n * _FP32_BYTES
        cursor = _align_up(cursor, _MBARRIER_BYTES) + _MBARRIER_BYTES
    return cursor


def _smem_budget() -> int:
    return _SMEM_CAPACITY_BYTES - _SMEM_HEADROOM_BYTES


def _k_tiles_per_rank(k: int, split_k: int) -> Tuple[int, int]:
    """(min, max) K tiles over the cluster ranks of an uneven split."""
    tiles = k // _CTA_K
    return tiles // split_k, -(-tiles // split_k)


def max_ab_stages(mma_n: int, split_k: int, k: int) -> int:
    """Deepest pipeline that fits the shared-memory budget next to the K slice's scales."""
    base = Fp8SplitKTactic(mma_n, split_k, _MIN_AB_STAGES)
    k_tiles = _k_tiles_per_rank(k, split_k)[1]
    return next(
        (
            s
            for s in range(_MAX_AB_STAGES, _MIN_AB_STAGES - 1, -1)
            if _smem_bytes(base, s, k_tiles) <= _smem_budget()
        ),
        0,
    )


def validate_tactic(tactic: Fp8SplitKTactic, m: int, n: int, k: int) -> None:
    """Reject a tactic that cannot serve ``(m, n, k)``."""
    if tactic.mma_n not in _SUPPORTED_MMA_N:
        raise ValueError(f"unsupported mma_n={tactic.mma_n}")
    if not 1 <= tactic.split_k <= _MAX_SPLIT_K:
        raise ValueError(f"split_k must be in [1, {_MAX_SPLIT_K}], got {tactic.split_k}")
    if not _MIN_AB_STAGES <= tactic.ab_stages <= _MAX_AB_STAGES:
        raise ValueError(
            f"ab_stages must be in [{_MIN_AB_STAGES}, {_MAX_AB_STAGES}], got {tactic.ab_stages}"
        )
    if not 1 <= m <= MAX_M:
        raise ValueError(f"the low-M FP8 path requires 1 <= M <= {MAX_M}, got {m}")
    if n <= 0:
        raise ValueError(f"N must be positive, got {n}")
    if k <= 0 or k % _CTA_K:
        raise ValueError(f"K={k} must be a positive multiple of {_CTA_K}")
    lo, hi = _k_tiles_per_rank(k, tactic.split_k)
    if lo < 1:
        raise ValueError(f"K={k} has fewer than split_k={tactic.split_k} tiles of {_CTA_K}")
    if hi > _MAX_K_TILES:
        raise ValueError(
            f"a K slice of {hi} tiles keeps more than {_MAX_K_TILES} tiles of scales in TMEM; "
            "raise split_k"
        )
    smem = _smem_bytes(tactic, tactic.ab_stages, hi)
    if smem > _smem_budget():
        raise ValueError(
            f"tactic {tactic} needs {smem} B of shared memory (budget {_smem_budget()} B); "
            f"max ab_stages is {max_ab_stages(tactic.mma_n, tactic.split_k, k)}"
        )


def mma_n_for(m: int) -> int:
    """MMA N (the kernel's row tile) of the bucket that holds ``m`` rows."""
    for mma_n in _SUPPORTED_MMA_N:
        if m <= mma_n:
            return mma_n
    raise ValueError(f"the low-M FP8 path requires 1 <= M <= {MAX_M}, got {m}")


def candidate_tactics(m: int, n: int, k: int, num_sms: int = 152):
    """Tactics worth timing for ``(m, n, k)``: the M bucket's MMA N, every split that keeps the
    grid within about one wave, at the deepest pipeline."""
    mma_n = mma_n_for(m)
    n_tiles = -(-n // _CTA_M) * -(-m // mma_n)
    out = []
    for split_k in range(1, _MAX_SPLIT_K + 1):
        if n_tiles * split_k > num_sms + num_sms // 8:
            break
        tactic = Fp8SplitKTactic(mma_n, split_k, max_ab_stages(mma_n, split_k, k))
        try:
            validate_tactic(tactic, m, n, k)
        except ValueError:
            continue
        out.append(tactic)
    return out


#: ``(split_k, ab_stages)`` per (N, K) and MMA N bucket (8 / 16 / 32 rows), tuned for a launch in a
#: CUDA graph behind a PDL predecessor. The
#: 8-row bucket was picked with the weights L2-resident (Kimi K3 decode prefetches them at 8 rows),
#: the 16 / 32-row buckets with cold weights. Shapes are TP8 shards of the Kimi K3 decode FP8
#: projections: latent up / down, KDA qkvg, KDA / MLA o_proj, MLA g|q_a|kv_a, MLA q_b.
_TUNED: Dict[Tuple[int, int], Dict[int, Tuple[int, int]]] = {
    (7168, 3584): {8: (2, 8), 16: (2, 11), 32: (2, 9)},
    (3584, 7168): {8: (4, 7), 16: (4, 10), 32: (3, 8)},
    (6144, 7168): {8: (2, 7), 32: (2, 8)},
    (7168, 1536): {8: (2, 12), 32: (1, 10)},
    (3648, 7168): {8: (4, 7), 32: (3, 8)},
    (2304, 1536): {8: (3, 8), 32: (1, 10)},
}


def default_tactic(m: int, n: int, k: int, num_sms: int = 152) -> Fp8SplitKTactic:
    """The tuned tactic for ``(n, k)`` and M's bucket, else the split that best fills one wave
    (at least two K tiles per CTA), at the deepest pipeline."""
    mma_n = mma_n_for(m)
    tuned = _TUNED.get((n, k), {}).get(mma_n)
    if tuned is not None:
        split_k, stages = tuned
        tactic = Fp8SplitKTactic(mma_n, split_k, min(stages, max_ab_stages(mma_n, split_k, k)))
        validate_tactic(tactic, m, n, k)
        return tactic
    n_tiles = -(-n // _CTA_M) * -(-m // mma_n)
    tiles = k // _CTA_K
    best = None
    for split_k in range(1, _MAX_SPLIT_K + 1):
        if split_k > 1 and (n_tiles * split_k > num_sms or tiles // split_k < 2):
            break
        tactic = Fp8SplitKTactic(mma_n, split_k, max_ab_stages(mma_n, split_k, k))
        try:
            validate_tactic(tactic, m, n, k)
        except ValueError:
            continue
        best = tactic
    if best is None:
        raise ValueError(f"no low-M FP8 tactic for M={m}, N={n}, K={k}")
    return best


class Fp8BlockScaledSplitKGemmKernel:
    """Low-M FP8 block-scaled GEMM with a cluster-local split-K reduction."""

    def __init__(
        self, *, tactic: Fp8SplitKTactic, k_tiles: int, use_pdl: bool, lamport: bool = False
    ) -> None:
        # ``lamport``: the epilogue publishes the BF16 output into a Lamport mailbox slot
        # (runtime ``slot_addr`` / ``slot_row_bytes``) instead of storing it to ``c``.
        self.lamport = bool(lamport)
        self.acc_dtype = cutlass.Float32
        self.sf_dtype = cutlass.Float8E8M0FNU
        self.cta_m = _CTA_M
        self.cta_n = tactic.mma_n
        self.cta_k = _CTA_K
        self.num_ab_stage = tactic.ab_stages
        self.split_k = tactic.split_k
        # Largest K slice of a cluster rank, in 128-K tiles: the scale buffers hold all of it.
        self.k_tiles = k_tiles
        self.tmem_cols = _tmem_cols(k_tiles)
        self.use_pdl = use_pdl
        # Dynamic shared memory: the allocations plus one 1 KB alignment slack, not the whole SM,
        # so a predecessor CTA can stay co-resident while the weights stream in.
        self.smem_bytes = min(
            _SMEM_CAPACITY_BYTES, _smem_bytes(tactic, tactic.ab_stages, k_tiles) + 1024
        )

        self.threads_per_cta = 256
        self.epilog_threads = 128
        self.mma_tiler_mn = (_CTA_M, tactic.mma_n)
        self.cta_group = tcgen05.CtaGroup.ONE
        self.tma_op = cute_ext.OperationTypeEnum.SM90_TMA_LOAD
        self.cluster_shape = (1, tactic.split_k, 1)

        values_per_thread = (_CTA_M * tactic.mma_n) // self.epilog_threads
        if values_per_thread % 4:
            raise ValueError(
                f"tile (128, {tactic.mma_n}) gives {values_per_thread} values per "
                "epilogue thread; the remote stores need a multiple of 4"
            )
        self.mailbox_elements = (tactic.split_k - 1) * self.epilog_threads * values_per_thread
        self.expected_transaction_bytes = self.mailbox_elements * _FP32_BYTES

    @cute.experimental.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        sfa: cute.Tensor,
        sfb: cute.Tensor,
        slot_addr: cutlass.Int64,
        slot_row_bytes: cutlass.Int64,
        stream: _cuda.CUstream,
    ):
        # Grid-y packs the output-M tile and the cluster rank.
        self.kernel(a, b, c, sfa, sfb, slot_addr, slot_row_bytes).launch(
            grid=(
                cute.ceil_div(c.layout.shape[0], self.cta_m),
                cute.ceil_div(c.layout.shape[1], self.cta_n) * self.split_k,
                1,
            ),
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape,
            smem=cute.Int64(self.smem_bytes),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.experimental.kernel
    def kernel(
        self,
        mA: cute.Tensor,  # weights (N, K, 1), K-major
        mB: cute.Tensor,  # activations (M, K, 1), K-major
        mC: cute.Tensor,  # output (N, M, 1), N-major (row-major [M, N])
        mSFA: cute.Tensor,  # weight scales, int32 (N, ceil(K / 512)), MN-major
        mSFB: cute.Tensor,  # activation scales, int32 (M, ceil(K / 512)), MN-major
        slot_addr: cutlass.Int64,
        slot_row_bytes: cutlass.Int64,
    ):
        """Allocate storage and dispatch the specialized warps."""
        stages = self.num_ab_stage
        ab_dtype = mA.element_type
        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            ab_dtype,
            ab_dtype,
            utils.LayoutEnum.from_tensor(mA).mma_major_mode(),
            utils.LayoutEnum.from_tensor(mB).mma_major_mode(),
            self.sf_dtype,
            _SF_VEC,
            self.cta_group,
            self.mma_tiler_mn,
        )
        mnk_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], self.cta_k)
        # The scale factors of the whole K slice, as one tile of k_tiles * 128 K.
        sf_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], self.cta_k * self.k_tiles)

        block_idx = cute.arch.block_idx()
        bidx = block_idx[0]
        split_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        n_idx = block_idx[1] // self.split_k
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        sA = cute_ext.allocate(
            ab_dtype,
            cute.AddressSpace.smem,
            sm100_utils.make_smem_layout_a(tiled_mma, mnk_tiler, ab_dtype, stages),
            alignment=_AB_BUFFER_ALIGN_BYTES,
        )
        sB = cute_ext.allocate(
            ab_dtype,
            cute.AddressSpace.smem,
            sm100_utils.make_smem_layout_b(tiled_mma, mnk_tiler, ab_dtype, stages),
            alignment=_AB_BUFFER_ALIGN_BYTES,
        )
        sfa_layout = blockscaled_utils.make_smem_layout_sfa(tiled_mma, sf_tiler, _SF_VEC, 1)
        sfb_layout = blockscaled_utils.make_smem_layout_sfb(tiled_mma, sf_tiler, _SF_VEC, 1)
        sSFA = cute_ext.allocate(
            self.sf_dtype, cute.AddressSpace.smem, sfa_layout, alignment=_SF_BUFFER_ALIGN_BYTES
        )
        sSFB = cute_ext.allocate(
            self.sf_dtype, cute.AddressSpace.smem, sfb_layout, alignment=_SF_BUFFER_ALIGN_BYTES
        )

        acc_layout = cute_ext.make_tmem_layout_acc(tiled_mma, self.mma_tiler_mn, acc_stage=1)
        c_tiler_mn = (self.cta_m, self.cta_n)

        bar_full = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(stages),
            alignment=_MBARRIER_BYTES,
        ).iterator
        bar_empty = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(stages),
            alignment=_MBARRIER_BYTES,
        ).iterator
        bar_mma_epilog = cute_ext.allocate(
            cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(1), alignment=_MBARRIER_BYTES
        ).iterator
        bar_tmem_alloc = cute_ext.allocate(
            cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(1), alignment=_MBARRIER_BYTES
        ).iterator
        bar_sfa = cute_ext.allocate(
            cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(1), alignment=_MBARRIER_BYTES
        ).iterator
        bar_sfb = cute_ext.allocate(
            cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(1), alignment=_MBARRIER_BYTES
        ).iterator
        tmem_base_ptr = cute_ext.allocate(
            cutlass.Int32,
            cute.AddressSpace.smem,
            cute.make_layout(1),
            alignment=_TMEM_POINTER_BYTES,
        ).iterator

        if cutlass.const_expr(self.split_k > 1):
            mailbox = cute_ext.allocate(
                cutlass.Float32,
                cute.AddressSpace.smem,
                cute.make_layout(self.mailbox_elements),
                alignment=_MAILBOX_ALIGN_BYTES,
            )
            bar_reduce = cute_ext.allocate(
                cutlass.Int64,
                cute.AddressSpace.smem,
                cute.make_layout(1),
                alignment=_MBARRIER_BYTES,
            ).iterator
        else:
            # Dummy operands for the compile-time-elided reduction.
            mailbox = sA
            bar_reduce = bar_mma_epilog

        if warp_idx == 0:
            with cute.arch.elect_one():
                for i in range(stages):
                    cute.arch.mbarrier_init(bar_full + i, _FULL_ARRIVALS)
                    cute.arch.mbarrier_init(bar_empty + i, 1)
                cute.arch.mbarrier_init(bar_mma_epilog, 1)
                cute.arch.mbarrier_init(bar_tmem_alloc, 160)
                cute.arch.mbarrier_init(bar_sfa, 1)
                cute.arch.mbarrier_init(bar_sfb, 1)
                if cutlass.const_expr(self.split_k > 1):
                    # Owner arrival plus peer transaction-byte credits.
                    cute.arch.mbarrier_init(bar_reduce, 1)

        cute.arch.mbarrier_init_fence()
        if cutlass.const_expr(self.split_k > 1):
            # Publish peer barriers before cross-CTA stores.
            cute.arch.cluster_arrive_relaxed()
        else:
            cute.arch.barrier()

        # Uneven K partition: the first (tiles % split_k) ranks take one extra tile.
        num_k_tiles = cute.size(mA, mode=[1]) // self.cta_k
        base_tiles = num_k_tiles // self.split_k
        extra_tiles = num_k_tiles - base_tiles * self.split_k
        has_extra = Int32(split_rank < extra_tiles)
        k_tile_count = base_tiles + has_extra
        k_tile_start = (
            split_rank * base_tiles + extra_tiles + (split_rank - extra_tiles) * has_extra
        )

        if cutlass.const_expr(self.split_k > 1):
            cute.arch.cluster_wait()

        if warp_idx == 0:
            self.dma_warp(
                bar_full,
                bar_empty,
                cute.local_tile(mA, (self.cta_m, self.cta_k), (bidx, None, 0)),
                sA,
                cute_ext.get_cta_v_map_ab(mA, mnk_tiler, tiled_mma, "A"),
                k_tile_start,
                k_tile_count,
                True,
            )
        elif warp_idx == 1:
            self.dma_warp(
                bar_full,
                bar_empty,
                cute.local_tile(mB, (self.cta_n, self.cta_k), (n_idx, None, 0)),
                sB,
                cute_ext.get_cta_v_map_ab(mB, mnk_tiler, tiled_mma, "B"),
                k_tile_start,
                k_tile_count,
                False,
            )
        elif warp_idx == 2:
            self.mma_warp(
                bar_full,
                bar_empty,
                bar_mma_epilog,
                bar_tmem_alloc,
                bar_sfa,
                bar_sfb,
                tiled_mma,
                sf_tiler,
                sA,
                sB,
                sSFA,
                sSFB,
                sfa_layout,
                sfb_layout,
                tmem_base_ptr,
                acc_layout,
                k_tile_count,
            )
        elif warp_idx == 3:
            self.sf_warp(
                bar_sfa,
                bar_sfb,
                mSFA,
                mSFB,
                sSFA,
                sSFB,
                bidx * self.cta_m,
                n_idx * self.cta_n,
                k_tile_start,
            )
        elif warp_idx >= 4:
            self.epilog_warp(
                bar_mma_epilog,
                bar_tmem_alloc,
                tmem_base_ptr,
                acc_layout,
                cute.local_tile(mC, c_tiler_mn, (bidx, n_idx, 0)),
                cute.arch.thread_idx()[0] - 128,
                mC.element_type,
                utils.LayoutEnum.from_tensor(mC),
                mailbox,
                bar_reduce,
                split_rank,
                slot_addr,
                slot_row_bytes,
                bidx,
                n_idx,
                cute.size(mC, mode=[1]),
                sA.iterator,
            )

    @cute.experimental.jit
    def dma_warp(
        self,
        bar_full,
        bar_empty,
        g_tile: cute.Tensor,
        s_tile: cute.Tensor,
        cta_v_map: cute.Layout,
        k_tile_start: cutlass.Int32,
        k_tile_count: cutlass.Int32,
        is_weight: cutlass.Constexpr,
    ):
        stages = self.num_ab_stage
        if cutlass.const_expr(not is_weight and self.use_pdl):
            # Only the activation depends on the previous kernel.
            cute.arch.griddepcontrol_wait()

        empty_phase = cutlass.Int32(1)
        for k_tile in cutlass.range(k_tile_count, unroll=1):
            stage = k_tile % stages
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    bar_full + stage,
                    cute.size_in_bytes(
                        s_tile.element_type,
                        cute.slice_(s_tile.layout, (None, None, None, 0)),
                    ),
                )
            cute_ext.tma_load(
                g_tile[None, None, k_tile_start + k_tile],
                s_tile[None, None, None, stage],
                (bar_full + stage).value,
                cta_v_map=cta_v_map,
                tma_operation_type=self.tma_op,
                update_expect_tx=False,
            )
            if stage == stages - 1:
                empty_phase = empty_phase ^ 1

        if cutlass.const_expr(is_weight and self.use_pdl):
            cute.arch.griddepcontrol_launch_dependents()
        self._drain_producer(bar_empty, empty_phase, k_tile_count)

    @cute.experimental.jit
    def _drain_producer(self, bar_empty, empty_phase: cutlass.Int32, k_tile_count: cutlass.Int32):
        stages = self.num_ab_stage
        for tail in cutlass.range(stages, unroll=1):
            stage = (tail + k_tile_count) % stages
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
            if stage == stages - 1:
                empty_phase = empty_phase ^ 1

    @cute.experimental.jit
    def sf_warp(
        self,
        bar_sfa,
        bar_sfb,
        mSFA: cute.Tensor,
        mSFB: cute.Tensor,
        sSFA: cute.Tensor,
        sSFB: cute.Tensor,
        a_row0: cutlass.Int32,
        b_row0: cutlass.Int32,
        k_tile_start: cutlass.Int32,
    ):
        """Write the MXF8 scale tiles of the CTA's whole K slice from the DeepGEMM packed words.

        Tile layout (BlockScaledBasicChunk, K-major; 512 B per 128-K tile, tiles back to back):
        row ``r``'s four 32-K scales of tile ``t`` are bytes ``512 * t + 16 * (r % 32) + 4 * (r // 32)
        + (0..3)``, so lane ``l`` owns words ``128 * t + 4 * l + i`` = rows ``l + 32 * i``. Each
        128-K block's UE8M0 byte goes into all four 32-K slots. The weight scales are written
        before the PDL wait, the activation scales (MMA N <= 32 rows: lane ``l`` = row ``l``)
        after it; one barrier each. Rows past N / M get the pad scale (their data is zero).
        """
        k_tiles = self.k_tiles
        n_words = _sf_words(k_tiles)
        lane = cute.arch.lane_idx()
        n_rows = cute.size(mSFA, mode=[0])
        m_rows = cute.size(mSFB, mode=[0])
        a_words = cute.size(mSFA, mode=[1])
        b_words = cute.size(mSFB, mode=[1])
        word0 = k_tile_start // 4
        # Tile t of the slice is 128-K block k_tile_start + t: byte (sub + t % 4) % 4 of word
        # word0 + t // 4 + ((sub + t % 4) >= 4).
        sub = k_tile_start - word0 * 4
        sfa_words = cute.make_tensor(
            cute.recast_ptr(sSFA.iterator, dtype=cutlass.Int32), cute.make_layout(k_tiles * 128)
        )
        sfb_words = cute.make_tensor(
            cute.recast_ptr(sSFB.iterator, dtype=cutlass.Int32), cute.make_layout(k_tiles * 128)
        )
        pad = Int32(_SF_PAD_WORD)
        rep = Int32(0x01010101)

        # Weight scales: static, so before the wait. Loads are clamped in bounds.
        for i in cutlass.range_constexpr(4):
            row = a_row0 + lane + 32 * i
            row_c = cutlass.min(row, n_rows - 1)
            vals = [mSFA[row_c, cutlass.min(word0 + w, a_words - 1)] for w in range(n_words)]
            valid = row < n_rows
            for t in cutlass.range_constexpr(k_tiles):
                pos = sub + (t % 4)
                word = cutlass.select_(pos >= 4, vals[t // 4 + 1], vals[t // 4])
                scale = (word >> ((pos & 3) * 8)) & Int32(0xFF)
                sfa_words[t * 128 + lane * 4 + i] = cutlass.select_(valid, scale * rep, pad)
        # Activation rows 32..127 of the padded tile: static pad.
        for t in cutlass.range_constexpr(k_tiles):
            for i in cutlass.range_constexpr(1, 4):
                sfb_words[t * 128 + lane * 4 + i] = pad
        # Generic-proxy stores -> async-proxy (tcgen05.cp) reads after the barrier: every lane
        # fences its own stores, the warp syncs, one lane arrives.
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.sync_warp()
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive(bar_sfa)

        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()
        b_row = b_row0 + lane
        b_row_c = cutlass.min(b_row, m_rows - 1)
        b_vals = [mSFB[b_row_c, cutlass.min(word0 + w, b_words - 1)] for w in range(n_words)]
        b_valid = (lane < self.cta_n) & (b_row < m_rows)
        for t in cutlass.range_constexpr(k_tiles):
            pos = sub + (t % 4)
            word = cutlass.select_(pos >= 4, b_vals[t // 4 + 1], b_vals[t // 4])
            scale = (word >> ((pos & 3) * 8)) & Int32(0xFF)
            sfb_words[t * 128 + lane * 4] = cutlass.select_(b_valid, scale * rep, pad)
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.sync_warp()
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive(bar_sfb)

    @cute.experimental.jit
    def mma_warp(
        self,
        bar_full,
        bar_empty,
        bar_mma_epilog,
        bar_tmem_alloc,
        bar_sfa,
        bar_sfb,
        tiled_mma: cute.TiledMma,
        sf_tiler: cutlass.Constexpr,
        sA: cute.Tensor,
        sB: cute.Tensor,
        sSFA: cute.Tensor,
        sSFB: cute.Tensor,
        sfa_layout: cutlass.Constexpr,
        sfb_layout: cutlass.Constexpr,
        tmem_base_ptr,
        acc_layout: cutlass.Constexpr,
        k_tile_count: cutlass.Int32,
    ):
        cute.arch.alloc_tmem(self.tmem_cols, tmem_base_ptr, is_two_cta=False)
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.relinquish_tmem_alloc_permit(is_two_cta=False)

        tmem_ptr = cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr)
        accumulator = cute.make_tensor(tmem_ptr, acc_layout)[None, None, None, 0]

        # Scales of the whole K slice: SFA from column 32, SFB after it (4 columns per K tile).
        sfb_col = _ACC_TMEM_COLS + _SF_TMEM_COLS_PER_TILE * self.k_tiles
        tCtSFA = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + _ACC_TMEM_COLS, dtype=self.sf_dtype),
            blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma, sf_tiler, _SF_VEC, cute.slice_(sfa_layout, (None, None, None, 0))
            ),
        )
        tCtSFB = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + sfb_col, dtype=self.sf_dtype),
            blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma, sf_tiler, _SF_VEC, cute.slice_(sfb_layout, (None, None, None, 0))
            ),
        )
        s2t_sfa, sSFA_s2t, tSFA_s2t = self._s2t_partition(sSFA, tCtSFA)
        s2t_sfb, sSFB_s2t, tSFB_s2t = self._s2t_partition(sSFB, tCtSFB)

        # One S2T copy per operand for the whole slice; the K loop then issues only MMAs (the
        # tcgen05 pipeline keeps copies and MMAs of this thread in order).
        cute.arch.mbarrier_wait(bar_sfa, 0)
        cute.copy(s2t_sfa, sSFA_s2t[(None, None, None, None, 0)], tSFA_s2t)
        cute.arch.mbarrier_wait(bar_sfb, 0)
        cute.copy(s2t_sfb, sSFB_s2t[(None, None, None, None, 0)], tSFB_s2t)

        mma_atom = cute.make_mma_atom(tiled_mma.op)
        mma_inst_tile_k = self.cta_k // cute.size(tiled_mma.shape_mnk, mode=[2])
        full_phase = cutlass.Int32(0)
        for k_tile in cutlass.range(k_tile_count, unroll=1):
            stage = k_tile % self.num_ab_stage
            cute.arch.mbarrier_wait(bar_full + stage, full_phase)
            for k_block in range(mma_inst_tile_k):
                if k_block == 0:
                    mma_atom.set(tcgen05.Field.ACCUMULATE, k_tile != 0)
                else:
                    mma_atom.set(tcgen05.Field.ACCUMULATE, True)
                sf_block = k_tile * mma_inst_tile_k + k_block
                mma_atom.set(tcgen05.Field.SFA, tCtSFA[(None, None, sf_block)].iterator)
                mma_atom.set(tcgen05.Field.SFB, tCtSFB[(None, None, sf_block)].iterator)
                cute_ext.dot(
                    mma_atom,
                    cute.append_ones(sA[None, None, k_block, stage], up_to_rank=3),
                    cute.append_ones(sB[None, None, k_block, stage], up_to_rank=3),
                    accumulator,
                )
            with cute.arch.elect_one():
                tcgen05.commit(bar_empty + stage, None, self.cta_group)
            if stage == self.num_ab_stage - 1:
                full_phase = full_phase ^ 1

        with cute.arch.elect_one():
            tcgen05.commit(bar_mma_epilog, None, self.cta_group)
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 1)
        cute.arch.dealloc_tmem(tmem_ptr, self.tmem_cols, is_two_cta=False)

    def _s2t_partition(self, sSF: cute.Tensor, tSF: cute.Tensor):
        """S2T tiled copy for one scale-factor operand (as dense_blockscaled_gemm_persistent)."""
        tCsSF_compact = cute.filter_zeros(sSF)
        tCtSF_compact = cute.filter_zeros(tSF)
        copy_atom = cute.make_copy_atom(tcgen05.Cp4x32x128bOp(self.cta_group), self.sf_dtype)
        tiled_copy = tcgen05.make_s2t_copy(copy_atom, tCtSF_compact)
        thr_copy = tiled_copy.get_slice(0)
        src = tcgen05.get_s2t_smem_desc_tensor(tiled_copy, thr_copy.partition_S(tCsSF_compact))
        dst = thr_copy.partition_D(tCtSF_compact)
        return tiled_copy, src, dst

    @cute.experimental.jit
    def epilog_warp(
        self,
        bar_mma_epilog,
        bar_tmem_alloc,
        tmem_base_ptr,
        acc_layout: cutlass.Constexpr,
        gD_tile: cute.Tensor,
        epi_tid: cutlass.Int32,
        c_dtype: cutlass.Constexpr,
        d_layout: cutlass.Constexpr,
        mailbox,
        bar_reduce,
        split_rank: cutlass.Int32,
        slot_addr: cutlass.Int64,
        slot_row_bytes: cutlass.Int64,
        w_tile: cutlass.Int32,
        n_tile: cutlass.Int32,
        m_rows: cutlass.Int32,
        stage_ptr,
    ):
        # Wait until MMA publishes the TMEM base pointer.
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 0)

        acc_view = cute.make_tensor(
            cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr),
            acc_layout,
        )[((None, None), 0, 0, 0)]

        epi_tile = (self.cta_m, self.cta_n)
        tiled_copy_t2r = cute.nvgpu.tcgen05.make_tmem_copy(
            sm100_utils.get_tmem_load_op(
                (self.cta_m, self.cta_n, self.cta_k),
                d_layout,
                c_dtype,
                self.acc_dtype,
                epi_tile,
                False,
            ),
            acc_view,
        )
        gD_epi = cute.flat_divide(gD_tile, epi_tile)

        rmem_layout = cute_ext.make_t2r_rmem_layout(tiled_copy_t2r, gD_epi, epi_tid)
        rAcc = cute_ext.allocate(self.acc_dtype, cute.AddressSpace.rmem, rmem_layout, alignment=32)
        rD = cute_ext.allocate(c_dtype, cute.AddressSpace.rmem, rmem_layout, alignment=32)
        thr_t2r = tiled_copy_t2r.get_slice(epi_tid)

        cute.arch.mbarrier_wait(bar_mma_epilog, 0)
        cute_ext.partition_and_copy(thr_t2r, acc_view, rAcc)
        # Make tcgen05.ld visible before TMEM release and RMEM use.
        cute.arch.fence_view_async_tmem_load()
        cute.arch.mbarrier_arrive(bar_tmem_alloc)

        # Peers publish FP32 partials; only rank 0 reduces (in rank order) and stores.
        if cutlass.const_expr(self.split_k > 1):
            values_per_thread = cutlass.const_expr(cute.size(rmem_layout))
            values_per_peer = cutlass.const_expr(self.epilog_threads * values_per_thread)
            if split_rank != OWNER_RANK:
                for value_idx in cutlass.range_constexpr(0, values_per_thread, 4):
                    _store_shared_remote_v4(
                        rAcc[value_idx],
                        rAcc[value_idx + 1],
                        rAcc[value_idx + 2],
                        rAcc[value_idx + 3],
                        mailbox.iterator
                        + (split_rank - Int32(1)) * values_per_peer
                        + epi_tid * values_per_thread
                        + value_idx,
                        bar_reduce,
                        Int32(OWNER_RANK),
                    )
            else:
                if epi_tid == 0:
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        bar_reduce, self.expected_transaction_bytes
                    )
                cute.arch.mbarrier_wait(bar_reduce, 0)
                for peer in cutlass.range_constexpr(self.split_k - 1):
                    for value_idx in cutlass.range_constexpr(values_per_thread):
                        rAcc[value_idx] = (
                            rAcc[value_idx]
                            + mailbox[
                                peer * values_per_peer + epi_tid * values_per_thread + value_idx
                            ]
                        )

        if split_rank == OWNER_RANK:
            if cutlass.const_expr(self.lamport):
                # Lamport publish (the slot contract of ``kimi_k3_oproj_lamport``). The tile goes to shared memory
                # (the operand buffers are free: every MMA has completed) in output order, so the
                # accumulator's register layout never matters; then each thread publishes whole
                # 16-byte fragments (8 consecutive columns of one token), -0 turned into +0.
                rD.store(rAcc.load().to(c_dtype))
                sD = cute.make_tensor(
                    cute.recast_ptr(stage_ptr, dtype=c_dtype),
                    cute.make_layout((self.cta_m, self.cta_n), stride=(1, self.cta_m)),
                )
                cute_ext.partition_and_copy(thr_t2r, rD, sD)
                cute.arch.barrier(barrier_id=1, number_of_threads=self.epilog_threads)
                for f in cutlass.range_constexpr(self.cta_n // 8):
                    idx = epi_tid + Int32(self.epilog_threads * f)
                    tok_local = idx // 16
                    frag = idx % 16
                    token = n_tile * self.cta_n + tok_local
                    if token < m_rows:
                        vals = [
                            sD[(frag * 8 + e, tok_local)].to(cutlass.Float32) for e in range(8)
                        ]
                        words = [
                            sanitize_bf16x2(pack_bf16x2(vals[2 * j], vals[2 * j + 1]))
                            for j in range(4)
                        ]
                        mailbox_store_v4(
                            slot_addr
                            + cutlass.Int64(token) * slot_row_bytes
                            + cutlass.Int64(w_tile * self.cta_m + frag * 8) * 2,
                            words,
                        )
            else:
                rD.store(rAcc.load().to(c_dtype))
                # Preserve TMEM coordinates; the copy predicates output tails.
                cute_ext.partition_and_copy(thr_t2r, rD, gD_epi[None, None, 0, 0])

        # The reduction mbarrier covers the remote stores; no cluster barrier needed.


@cute.experimental.jit
def _launch(
    gemm_op: cutlass.Constexpr,
    a_ptr: cute.Pointer,
    b_ptr: cute.Pointer,
    c_ptr: cute.Pointer,
    sfa_ptr: cute.Pointer,
    sfb_ptr: cute.Pointer,
    n: cutlass.Int32,
    m: cutlass.Int32,
    k: cutlass.Int32,
    ld_b: cutlass.Int32,
    ld_c: cutlass.Int32,
    ld_sfa: cutlass.Int32,
    ld_sfb: cutlass.Int32,
    slot_addr: cutlass.Int64,
    slot_row_bytes: cutlass.Int64,
    stream: _cuda.CUstream,
):
    k_words = (k + 511) // 512
    gemm_op(
        cute.make_tensor(a_ptr, cute.make_layout((n, k, 1), stride=(k, 1, n * k))),
        cute.make_tensor(b_ptr, cute.make_layout((m, k, 1), stride=(ld_b, 1, m * ld_b))),
        cute.make_tensor(c_ptr, cute.make_layout((n, m, 1), stride=(1, ld_c, m * ld_c))),
        cute.make_tensor(sfa_ptr, cute.make_layout((n, k_words), stride=(1, ld_sfa))),
        cute.make_tensor(sfb_ptr, cute.make_layout((m, k_words), stride=(1, ld_sfb))),
        slot_addr,
        slot_row_bytes,
        stream,
    )


def _pointers(a, b, c, sfa, sfb):
    return (
        make_ptr(cutlass.Float8E4M3FN, a.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float8E4M3FN, b.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.BFloat16, c.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Int32, sfa.data_ptr(), cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.Int32, sfb.data_ptr(), cute.AddressSpace.gmem, assumed_align=4),
    )


@functools.cache
def _get_compiled(tactic: Fp8SplitKTactic, k_tiles: int, use_pdl: bool, lamport: bool = False):
    kernel = Fp8BlockScaledSplitKGemmKernel(
        tactic=tactic, k_tiles=k_tiles, use_pdl=use_pdl, lamport=lamport
    )
    # Representative operands only fix types and alignments; every extent and stride is dynamic
    # (K up to split_k * k_tiles tiles of 128).
    n, m, k = 256, tactic.mma_n, 4 * _CTA_K
    a = _torch.empty((n, k), dtype=_torch.float8_e4m3fn, device="cuda")
    b = _torch.empty((m, k), dtype=_torch.float8_e4m3fn, device="cuda")
    c = _torch.empty((m, n), dtype=_torch.bfloat16, device="cuda")
    sfa = _torch.empty((1, n), dtype=_torch.int32, device="cuda")
    sfb = _torch.empty((1, m), dtype=_torch.int32, device="cuda")
    stream = _cuda.CUstream(_torch.cuda.current_stream().cuda_stream)
    return cute_ext.compile(
        _launch,
        kernel,
        *_pointers(a, b, c, sfa, sfb),
        n,
        m,
        k,
        k,
        n,
        n,
        m,
        cutlass.Int64(0),
        cutlass.Int64(0),
        stream,
    )


def _operand_error(act, act_scale, weight, weight_scale, out) -> Optional[Tuple[type, str]]:
    """``(exception type, message)`` for operands the kernel does not take, else ``None``.

    ``out=None`` stands for the contiguous BF16 ``[M, N]`` output a caller would allocate.
    """
    if act.dtype != _torch.float8_e4m3fn or weight.dtype != _torch.float8_e4m3fn:
        return TypeError, "activation and weight must be float8_e4m3fn"
    if act_scale.dtype != _torch.int32 or weight_scale.dtype != _torch.int32:
        return TypeError, "scales must be DeepGEMM-packed int32 (UE8M0 x4)"
    if out is not None and out.dtype != _torch.bfloat16:
        return TypeError, "output must be bfloat16"
    if act.dim() != 2 or weight.dim() != 2 or (out is not None and out.dim() != 2):
        return ValueError, "activation, weight and output must be 2D"
    m, k = act.shape
    n = weight.shape[0]
    if weight.shape[1] != k or (out is not None and tuple(out.shape) != (m, n)):
        return ValueError, (
            f"shape mismatch: act {tuple(act.shape)}, weight {tuple(weight.shape)}, "
            f"out {None if out is None else tuple(out.shape)}"
        )
    if act.stride(1) != 1 or weight.stride(1) != 1 or (out is not None and out.stride(1) != 1):
        return ValueError, "activation, weight and output rows must be contiguous"
    if weight.stride(0) != k:
        return ValueError, "weight must be a dense [N, K] matrix"
    if act.stride(0) % 16 or act.data_ptr() % 16 or weight.data_ptr() % 16:
        return ValueError, "FP8 operands need 16-byte aligned rows (TMA)"
    out_ld, out_ptr = (n, 0) if out is None else (out.stride(0), out.data_ptr())
    if out_ld % 8 or out_ptr % 16:
        return ValueError, "output rows must be 16-byte aligned"
    k_words = -(-k // 512)
    for name, scale, rows in (("activation", act_scale, m), ("weight", weight_scale, n)):
        if scale.dim() != 2 or tuple(scale.shape) != (rows, k_words) or scale.stride(0) != 1:
            return ValueError, (
                f"{name} scale must be MN-major int32 [{rows}, {k_words}], got "
                f"shape {tuple(scale.shape)} stride {scale.stride()}"
            )
        if k_words > 1 and scale.stride(1) < rows:
            return ValueError, f"{name} scale column stride {scale.stride(1)} < rows {rows}"
    return None


def _check_operands(act, act_scale, weight, weight_scale, out) -> Tuple[int, int, int, int, int]:
    """Validate shapes, dtypes, strides and alignment; return (m, n, k, ld_sfa, ld_sfb)."""
    error = _operand_error(act, act_scale, weight, weight_scale, out)
    if error is not None:
        raise error[0](error[1])
    m, k = act.shape
    n = weight.shape[0]
    k_words = -(-k // 512)
    ld_sfa = weight_scale.stride(1) if k_words > 1 else n
    ld_sfb = act_scale.stride(1) if k_words > 1 else m
    return m, n, k, ld_sfa, ld_sfb


def unsupported_reason(act, act_scale, weight, weight_scale, out=None) -> Optional[str]:
    """Why ``run_fp8_blockscaled_splitk`` would refuse these operands, or ``None``; never raises.

    A caller checks here and keeps its DeepGEMM call for refused operands instead of raising.
    ``out=None`` stands for the contiguous BF16 ``[M, N]`` output the caller would allocate.
    """
    error = _operand_error(act, act_scale, weight, weight_scale, out)
    if error is not None:
        return error[1]
    if not 0 < act.shape[0] <= MAX_M:
        return f"{act.shape[0]} rows (serves 1 <= M <= {MAX_M})"
    return None


def supports(act, act_scale, weight, weight_scale, out=None) -> bool:
    """Whether ``run_fp8_blockscaled_splitk`` takes these operands (``unsupported_reason`` is ``None``)."""
    return unsupported_reason(act, act_scale, weight, weight_scale, out) is None


def run_fp8_blockscaled_splitk(
    act: _torch.Tensor,
    act_scale: _torch.Tensor,
    weight: _torch.Tensor,
    weight_scale: _torch.Tensor,
    out: _torch.Tensor,
    pdl: bool,
    tactic: Optional[Fp8SplitKTactic] = None,
    lamport: Optional[Tuple[int, int]] = None,
) -> _torch.Tensor:
    """``out[M, N] = act[M, K] @ weight[N, K]^T`` with DeepGEMM 1d1d FP8 operands (M <= 32).

    Args:
        act: E4M3 ``[M, K]`` (row stride a multiple of 16 bytes).
        act_scale: int32 ``[M, ceil(K / 512)]``, strides ``(1, ld)``: packed UE8M0 1x128 scales.
        weight: E4M3 ``[N, K]``, contiguous.
        weight_scale: int32 ``[N, ceil(K / 512)]``, strides ``(1, ld)``: per-row packed UE8M0
            128x128 block scales (DeepGEMM's SFB layout).
        out: BF16 ``[M, N]``.
        pdl: launch with programmatic dependent launch (the weights then load before the wait).
        tactic: tile/split/pipeline; ``default_tactic(M, N, K)`` when ``None``.
        lamport: ``(slot_address, row_bytes)``: publish the BF16 output (``-0`` as ``+0``) into a
            Lamport mailbox slot instead of storing it (N must be a multiple of 128; ``out`` is
            then only a shape / pointer carrier and is not written). Token i, column n goes to byte
            ``slot_address + i * row_bytes + 2 * n``, rows past M are not written.
    """
    m, n, k, ld_sfa, ld_sfb = _check_operands(act, act_scale, weight, weight_scale, out)
    if tactic is None:
        tactic = default_tactic(m, n, k)
    else:
        validate_tactic(tactic, m, n, k)
    if lamport is not None and (n % _CTA_M or lamport[0] % 16 or lamport[1] % 16):
        raise ValueError(
            f"lamport publish: N={n} must be a multiple of {_CTA_M}, slot and row stride 16-byte aligned"
        )
    compiled = _get_compiled(
        tactic, _k_tiles_per_rank(k, tactic.split_k)[1], bool(pdl), lamport is not None
    )
    stream = _cuda.CUstream(_torch.cuda.current_stream(act.device).cuda_stream)
    compiled(
        *_pointers(weight, act, out, weight_scale, act_scale),
        n,
        m,
        k,
        act.stride(0),
        out.stride(0),
        ld_sfa,
        ld_sfb,
        cutlass.Int64(lamport[0] if lamport is not None else 0),
        cutlass.Int64(lamport[1] if lamport is not None else 0),
        stream,
    )
    return out


__all__ = [
    "MAX_M",
    "Fp8SplitKTactic",
    "candidate_tactics",
    "default_tactic",
    "max_ab_stages",
    "run_fp8_blockscaled_splitk",
    "supports",
    "unsupported_reason",
    "validate_tactic",
]
