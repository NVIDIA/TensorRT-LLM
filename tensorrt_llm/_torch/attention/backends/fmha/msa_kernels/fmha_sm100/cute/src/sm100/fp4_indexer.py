# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""SM100 FP4 sparse-attention indexer kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import torch
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync, tcgen05

from src.common import pipeline as common_pipeline


FP4_FORMAT = Literal["mxfp4", "nvfp4"]
_FP4_PACKED_D_BYTES = 64
_HEAD_DIM = 128
_BLOCK_K = 128
_PAGE_SIZE = 128
_MMA_TILER_MN = (128, 128)
_MMA_INST_SHAPE_K = 64
_NON_CAUSAL_K_TILES_PER_CTA = 16
_CAUSAL_K_TILES_PER_CTA = 16
_DECODE_PACK_Q_LEN = 8
_DECODE_QHEAD_PER_KV = 16
_DECODE_K_TILES_PER_CTA = 16
_AB_DTYPE = cutlass.Float4E2M1FN


@dataclass(frozen=True)
class Fp4FormatSpec:
    name: FP4_FORMAT
    sf_vec_size: int
    scale_groups: int
    torch_scale_dtype: torch.dtype
    cutlass_scale_dtype: type


_FORMAT_SPECS: dict[str, Fp4FormatSpec] = {
    "mxfp4": Fp4FormatSpec(
        name="mxfp4",
        sf_vec_size=32,
        scale_groups=4,
        torch_scale_dtype=torch.float8_e8m0fnu,
        cutlass_scale_dtype=cutlass.Float8E8M0FNU,
    ),
    "nvfp4": Fp4FormatSpec(
        name="nvfp4",
        sf_vec_size=16,
        scale_groups=8,
        torch_scale_dtype=torch.float8_e4m3fn,
        cutlass_scale_dtype=cutlass.Float8E4M3FN,
    ),
}


def normalize_fp4_format(fmt: str) -> Fp4FormatSpec:
    key = str(fmt).lower()
    try:
        return _FORMAT_SPECS[key]
    except KeyError as exc:
        raise ValueError(f"format must be one of {sorted(_FORMAT_SPECS)}, got {fmt!r}") from exc


def ceil_div(x: int, y: int) -> int:
    return (int(x) + int(y) - 1) // int(y)


def k_tiles_per_cta_for(causal: bool) -> int:
    return _CAUSAL_K_TILES_PER_CTA if bool(causal) else _NON_CAUSAL_K_TILES_PER_CTA


class Fp4IndexerScaleReorderSm100:
    """Reorder public FP4 indexer scales to the 1CTA blockscaled MMA layout."""

    def __init__(self, *, fmt: str):
        spec = normalize_fp4_format(fmt)
        self.fmt = spec.name
        self.sf_dtype = spec.cutlass_scale_dtype
        self.scale_groups = spec.scale_groups
        self.threads_per_cta = 256

    @cute.jit
    def __call__(
        self,
        q_scale_ptr: cute.Pointer,
        k_scale_ptr: cute.Pointer,
        q_scale_mma_ptr: cute.Pointer,
        k_scale_mma_ptr: cute.Pointer,
        problem_size: tuple,
        stream: cuda.CUstream,
    ):
        total_q, heads_q, page_count, heads_k = problem_size
        rest_q_m = cute.ceil_div(total_q, 128)
        rest_g = cute.ceil_div(self.scale_groups, 4)
        k_l = page_count * heads_k

        q_scale = cute.make_tensor(
            q_scale_ptr,
            cute.make_layout(
                (total_q, heads_q, self.scale_groups),
                stride=(heads_q * self.scale_groups, self.scale_groups, 1),
            ),
        )
        k_scale = cute.make_tensor(
            k_scale_ptr,
            cute.make_layout(
                (page_count, heads_k, _PAGE_SIZE, self.scale_groups),
                stride=(
                    heads_k * _PAGE_SIZE * self.scale_groups,
                    _PAGE_SIZE * self.scale_groups,
                    self.scale_groups,
                    1,
                ),
            ),
        )

        q_mma_layout = cute.make_ordered_layout(
            (32, 4, rest_q_m, 4, rest_g, heads_q),
            order=(2, 1, 4, 0, 3, 5),
        )
        k_mma_layout = cute.make_ordered_layout(
            (32, 4, 1, 4, rest_g, k_l),
            order=(2, 1, 4, 0, 3, 5),
        )
        q_scale_mma = cute.make_tensor(q_scale_mma_ptr, q_mma_layout)
        k_scale_mma = cute.make_tensor(k_scale_mma_ptr, k_mma_layout)
        q_scale_mma = cute.group_modes(q_scale_mma, 0, 3)
        q_scale_mma = cute.group_modes(q_scale_mma, 1, 3)
        k_scale_mma = cute.group_modes(k_scale_mma, 0, 3)
        k_scale_mma = cute.group_modes(k_scale_mma, 1, 3)

        q_scale_count = total_q * heads_q * Int32(self.scale_groups)
        k_scale_count = page_count * heads_k * Int32(_PAGE_SIZE * self.scale_groups)
        total_scale_count = q_scale_count + k_scale_count
        grid_ctas = cute.ceil_div(total_scale_count, self.threads_per_cta)
        self.kernel(
            q_scale,
            k_scale,
            q_scale_mma,
            k_scale_mma,
            heads_q,
            heads_k,
            q_scale_count,
            total_scale_count,
        ).launch(
            grid=(grid_ctas, 1, 1),
            block=[self.threads_per_cta, 1, 1],
            cluster=(1, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        q_scale: cute.Tensor,
        k_scale: cute.Tensor,
        q_scale_mma: cute.Tensor,
        k_scale_mma: cute.Tensor,
        heads_q: Int32,
        heads_k: Int32,
        q_scale_count: Int32,
        total_scale_count: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        grid_dim, _, _ = cute.arch.grid_dim()
        linear = block_idx * Int32(self.threads_per_cta) + tidx
        stride = grid_dim * Int32(self.threads_per_cta)

        while linear < total_scale_count:
            if linear < q_scale_count:
                group = linear % Int32(self.scale_groups)
                tmp = linear // Int32(self.scale_groups)
                head = tmp % heads_q
                row = tmp // heads_q
                q_scale_mma[row, group, head] = q_scale[row, head, group]
            else:
                k_linear = linear - q_scale_count
                group = k_linear % Int32(self.scale_groups)
                tmp = k_linear // Int32(self.scale_groups)
                row = tmp % Int32(_PAGE_SIZE)
                tmp = tmp // Int32(_PAGE_SIZE)
                head = tmp % heads_k
                page = tmp // heads_k
                scale_l = page * heads_k + head
                k_scale_mma[row, group, scale_l] = k_scale[page, head, row, group]
            linear += stride


class Fp4IndexerStagedMmaSm100:
    """Single-kernel FP4 indexer for preordered MMA scale storage."""

    def __init__(
        self,
        *,
        fmt: str,
        causal: bool,
        preordered_q_scale_tma: bool = False,
        compact_schedule: bool = False,
        use_tmem_load_red: bool = False,
    ):
        spec = normalize_fp4_format(fmt)
        self.fmt = spec.name
        self.is_causal = bool(causal)
        self.preordered_q_scale_tma = bool(preordered_q_scale_tma)
        self.compact_schedule = bool(compact_schedule)
        self.use_tmem_load_red = bool(use_tmem_load_red)
        self.sf_vec_size = spec.sf_vec_size
        self.sf_dtype = spec.cutlass_scale_dtype
        self.scale_groups = spec.scale_groups
        self.use_nvfp4 = spec.name == "nvfp4"
        self.epi_threads_per_cta = 128
        self.epi_warps_per_group = 4
        self.num_epi_warpgroups = 2
        self.mma_warp_id = self.epi_warps_per_group * self.num_epi_warpgroups
        self.load_warp_id = self.mma_warp_id + 1
        self.threads_per_cta = 384
        self.num_tmem_alloc_cols = 512
        self.num_q_stage = 1
        self.num_acc_stage = 3
        self.num_ab_stage = 3
        self.k_tiles_per_cta = k_tiles_per_cta_for(self.is_causal)

    @cute.jit
    def __call__(
        self,
        q_ptr: cute.Pointer,
        k_ptr: cute.Pointer,
        q_scale_ptr: cute.Pointer,
        k_scale_ptr: cute.Pointer,
        scores_ptr: cute.Pointer,
        kv_indices_ptr: cute.Pointer,
        cu_seqlens_q_ptr: cute.Pointer,
        cu_seqlens_k_ptr: cute.Pointer,
        cu_page_offsets_ptr: cute.Pointer,
        qo_offset_ptr: cute.Pointer,
        problem_size: tuple,
        stream: cuda.CUstream,
    ):
        (
            m,
            _,
            k,
            _,
            lk,
            heads_q,
            heads_k,
            batch,
            max_k_tiles,
            total_q,
            has_qo_offset,
            compact_task_count,
        ) = problem_size
        page_count = lk // heads_k
        self.mma_tiler = (_MMA_TILER_MN[0], _MMA_TILER_MN[1], _MMA_INST_SHAPE_K * 2)
        self.cta_tile_shape_mnk = self.mma_tiler

        q_tma_tensor = cute.make_tensor(
            cute.recast_ptr(q_ptr, dtype=_AB_DTYPE),
            cute.make_layout(
                (total_q, _HEAD_DIM, heads_q),
                stride=(heads_q * _HEAD_DIM, 1, _HEAD_DIM),
            ),
        )
        k_tma_tensor = cute.make_tensor(
            cute.recast_ptr(k_ptr, dtype=_AB_DTYPE),
            cute.make_layout(
                (_PAGE_SIZE, _HEAD_DIM, heads_k, page_count),
                stride=(_HEAD_DIM, 1, _PAGE_SIZE * _HEAD_DIM, heads_k * _PAGE_SIZE * _HEAD_DIM),
            ),
        )
        q_scale_tensor = cute.make_tensor(
            q_scale_ptr,
            blockscaled_utils.tile_atom_to_shape_SF(
                (total_q, _HEAD_DIM, heads_q),
                self.sf_vec_size,
            ),
        )
        k_scale_tensor = cute.make_tensor(
            k_scale_ptr,
            blockscaled_utils.tile_atom_to_shape_SF(
                (_PAGE_SIZE, _HEAD_DIM, page_count * heads_k),
                self.sf_vec_size,
            ),
        )
        scores_tensor = cute.make_tensor(
            scores_ptr,
            cute.make_layout((heads_q, max_k_tiles, total_q), stride=(max_k_tiles * total_q, total_q, 1)),
        )
        kv_indices_tensor = cute.make_tensor(
            kv_indices_ptr,
            cute.make_layout((page_count,), stride=(1,)),
        )
        cu_layout = cute.make_layout((batch + 1,), stride=(1,))
        cu_q_tensor = cute.make_tensor(cu_seqlens_q_ptr, cu_layout)
        cu_k_tensor = cute.make_tensor(cu_seqlens_k_ptr, cu_layout)
        cu_page_offsets_tensor = cute.make_tensor(cu_page_offsets_ptr, cu_layout)
        qo_offset_tensor = cute.make_tensor(qo_offset_ptr, cute.make_layout((batch,), stride=(1,)))

        if const_expr(self.use_nvfp4):
            mma_op = tcgen05.MmaMXF4NVF4Op(
                self.sf_dtype,
                (*_MMA_TILER_MN, _MMA_INST_SHAPE_K),
                tcgen05.CtaGroup.ONE,
                tcgen05.OperandSource.SMEM,
            )
        else:
            mma_op = tcgen05.MmaMXF4Op(
                (*_MMA_TILER_MN, _MMA_INST_SHAPE_K),
                tcgen05.CtaGroup.ONE,
                tcgen05.OperandSource.SMEM,
            )
        tiled_mma = cute.make_tiled_mma(mma_op)
        q_smem_layout = sm100_utils.make_smem_layout_a(tiled_mma, self.mma_tiler, _AB_DTYPE, self.num_q_stage)
        k_smem_layout = sm100_utils.make_smem_layout_b(tiled_mma, self.mma_tiler, _AB_DTYPE, self.num_ab_stage)
        q_scale_smem_layout = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_q_stage,
        )
        k_scale_smem_layout = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_ab_stage,
        )
        cluster_layout_vmnk = cute.make_layout((1, 1, 1, 1))
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        q_smem_layout_stage = cute.slice_(q_smem_layout, (None, None, None, 0))
        k_smem_layout_stage = cute.slice_(k_smem_layout, (None, None, None, 0))
        tma_q = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            q_tma_tensor,
            q_smem_layout_stage,
            self.mma_tiler,
            tiled_mma,
            cluster_layout_vmnk.shape,
        )
        tma_k = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            k_tma_tensor,
            k_smem_layout_stage,
            self.mma_tiler,
            tiled_mma,
            cluster_layout_vmnk.shape,
        )
        if const_expr(self.preordered_q_scale_tma):
            tma_qs = cute.nvgpu.make_tiled_tma_atom_A(
                tma_load_op,
                q_scale_tensor,
                q_scale_smem_layout,
                self.mma_tiler,
                tiled_mma,
                cluster_layout_vmnk.shape,
                internal_type=cutlass.Int16,
            )
        else:
            tma_qs = tma_q
        tma_ks = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            k_scale_tensor,
            k_scale_smem_layout,
            self.mma_tiler,
            tiled_mma,
            cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        grid_q_tiles = cute.ceil_div(m, self.cta_tile_shape_mnk[0])
        grid_k_groups = cute.ceil_div(max_k_tiles, self.k_tiles_per_cta)
        if const_expr(self.compact_schedule):
            grid_x = compact_task_count
        else:
            grid_x = grid_q_tiles * grid_k_groups
        self.kernel(
            tiled_mma,
            tma_q,
            tma_qs,
            tma_k,
            tma_ks,
            q_scale_tensor,
            k_scale_tensor,
            scores_tensor,
            kv_indices_tensor,
            cu_q_tensor,
            cu_k_tensor,
            cu_page_offsets_tensor,
            qo_offset_tensor,
            q_smem_layout,
            k_smem_layout,
            q_scale_smem_layout,
            k_scale_smem_layout,
            heads_q,
            heads_k,
            has_qo_offset,
            max_k_tiles,
            grid_k_groups,
        ).launch(
            grid=(grid_x, batch * heads_q, 1),
            block=[self.threads_per_cta, 1, 1],
            cluster=(1, 1, 1),
            stream=stream,
        )

    @cute.jit
    def _group_has_visible(
        self,
        q_tile_start: Int32,
        q_tile_last: Int32,
        q_len: Int32,
        group_first_ktile: Int32,
        batch_k_tiles: Int32,
        causal_offset: Int32,
    ):
        visible = q_tile_start < q_len and group_first_ktile < batch_k_tiles
        if const_expr(self.is_causal):
            visible = visible and group_first_ktile * Int32(_BLOCK_K) <= q_tile_last + causal_offset
        return visible

    @cute.jit
    def _tile_has_visible(
        self,
        q_tile_start: Int32,
        q_tile_last: Int32,
        q_len: Int32,
        ktile: Int32,
        batch_k_tiles: Int32,
        causal_offset: Int32,
    ):
        visible = q_tile_start < q_len and ktile < batch_k_tiles
        if const_expr(self.is_causal):
            visible = visible and ktile * Int32(_BLOCK_K) <= q_tile_last + causal_offset
        return visible

    @cute.jit
    def _tile_mask_free(self, q_tile_start: Int32, ktile: Int32, causal_offset: Int32):
        if const_expr(self.is_causal):
            return ktile * Int32(_BLOCK_K) + Int32(_BLOCK_K - 1) <= q_tile_start + causal_offset
        return True

    @cute.jit
    def _full_tile_coord_visible(
        self,
        coord_m: Int32,
        target_m: Int32,
        q_local: Int32,
        k_local: Int32,
        causal_offset: Int32,
    ):
        visible = coord_m == target_m
        if const_expr(self.is_causal):
            visible = visible and k_local <= q_local + causal_offset
        return visible

    @cute.jit
    def _partial_tile_coord_visible(
        self,
        coord_m: Int32,
        target_m: Int32,
        q_local: Int32,
        k_local: Int32,
        q_len: Int32,
        k_len: Int32,
        causal_offset: Int32,
    ):
        visible = coord_m == target_m and q_local < q_len and k_local < k_len
        if const_expr(self.is_causal):
            visible = visible and k_local <= q_local + causal_offset
        return visible

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_q: cpasync.TmaInfo,
        tma_qs: cpasync.TmaInfo,
        tma_k: cpasync.TmaInfo,
        tma_ks: cpasync.TmaInfo,
        mQS: cute.Tensor,
        mKS: cute.Tensor,
        mScores: cute.Tensor,
        mKvIndices: cute.Tensor,
        mCuQ: cute.Tensor,
        mCuK: cute.Tensor,
        mCuPages: cute.Tensor,
        mQoOffset: cute.Tensor,
        q_smem_layout: cute.ComposedLayout,
        k_smem_layout: cute.ComposedLayout,
        q_scale_smem_layout: cute.Layout,
        k_scale_smem_layout: cute.Layout,
        heads_q: Int32,
        heads_k: Int32,
        has_qo_offset: Int32,
        max_k_tiles: Int32,
        k_group_count: Int32,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        lane_idx = cute.arch.lane_idx()
        epi_tidx = tidx % Int32(self.epi_threads_per_cta)
        epi_warpgroup_idx = warp_idx // Int32(self.epi_warps_per_group)
        task_idx, q_l, _ = cute.arch.block_idx()
        batch_idx = q_l // heads_q
        hq = q_l - batch_idx * heads_q
        hk = hq // (heads_q // heads_k)
        q_begin = mCuQ[batch_idx]
        q_end = mCuQ[batch_idx + 1]
        k_begin = mCuK[batch_idx]
        k_end = mCuK[batch_idx + 1]
        q_len = q_end - q_begin
        k_len = k_end - k_begin
        page_begin = mCuPages[batch_idx]
        batch_k_tiles = (k_len + Int32(_PAGE_SIZE - 1)) // Int32(_PAGE_SIZE)
        causal_offset = Int32(0)
        if const_expr(self.is_causal):
            causal_offset = k_len - q_len
            if has_qo_offset != 0:
                causal_offset = mQoOffset[batch_idx]
        task_valid = True
        q_tile_idx = Int32(0)
        ktile_group = Int32(0)
        if const_expr(self.compact_schedule):
            remaining = task_idx
            q_tile_count = (q_len + Int32(self.cta_tile_shape_mnk[0] - 1)) // Int32(self.cta_tile_shape_mnk[0])
            batch_k_group_count = (batch_k_tiles + Int32(self.k_tiles_per_cta - 1)) // Int32(self.k_tiles_per_cta)
            q_scan = Int32(0)
            task_valid = False
            while q_scan < q_tile_count and not task_valid:
                q_scan_start = q_scan * Int32(self.cta_tile_shape_mnk[0])
                q_scan_last = q_scan_start + Int32(self.cta_tile_shape_mnk[0] - 1)
                if q_scan_last >= q_len:
                    q_scan_last = q_len - Int32(1)
                visible_limit = q_scan_last + causal_offset
                visible_group_count = Int32(0)
                if visible_limit >= Int32(0):
                    visible_group_count = visible_limit // Int32(self.k_tiles_per_cta * _BLOCK_K) + Int32(1)
                    if visible_group_count > batch_k_group_count:
                        visible_group_count = batch_k_group_count
                task_valid = remaining < visible_group_count
                if not task_valid:
                    remaining -= visible_group_count
                    q_scan += Int32(1)
            if task_valid:
                q_tile_idx = q_scan
                ktile_group = remaining
            else:
                q_len = Int32(0)
                k_len = Int32(0)
        else:
            q_tile_idx = task_idx // k_group_count
            ktile_group = task_idx - q_tile_idx * k_group_count
        q_tile_start = q_tile_idx * Int32(self.cta_tile_shape_mnk[0])
        q_tile_last = q_tile_start + Int32(self.cta_tile_shape_mnk[0] - 1)
        if q_tile_last >= q_len:
            q_tile_last = q_len - Int32(1)
        q_tile_full = q_tile_start + Int32(self.cta_tile_shape_mnk[0] - 1) < q_len
        q_tile_global_start = q_begin + q_tile_start
        q_scale_tma_safe = q_tile_global_start == (q_tile_global_start // Int32(128)) * Int32(128)
        group_first_ktile = ktile_group * Int32(self.k_tiles_per_cta)
        group_has_visible = self._group_has_visible(
            q_tile_start,
            q_tile_last,
            q_len,
            group_first_ktile,
            batch_k_tiles,
            causal_offset,
        )

        @cute.struct
        class SharedStorage:
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            qs_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            k_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sQ_public = smem.allocate_tensor(_AB_DTYPE, q_smem_layout.outer, 128, swizzle=q_smem_layout.inner)
        sK_public = smem.allocate_tensor(_AB_DTYPE, k_smem_layout.outer, 128, swizzle=k_smem_layout.inner)
        sQS_public = smem.allocate_tensor(self.sf_dtype, q_scale_smem_layout, 128)
        sKS_public = smem.allocate_tensor(self.sf_dtype, k_scale_smem_layout, 128)
        mQ_tma = tma_q.tma_tensor
        mQS_tma = tma_qs.tma_tensor
        mK_tma = tma_k.tma_tensor
        mKS_tma = tma_ks.tma_tensor
        thr_mma = tiled_mma.get_slice(0)
        tCsQ = thr_mma.partition_A(sQ_public)
        tCsK = thr_mma.partition_B(sK_public)
        mQ_tma_cur = cute.domain_offset((q_begin, 0, 0), mQ_tma)
        gQ_tma = cute.local_tile(
            mQ_tma_cur,
            cute.slice_(self.mma_tiler, (None, 0, None)),
            (None, None, None),
        )
        tCgQ_tma = thr_mma.partition_A(gQ_tma)
        tQsQ_tma, tQgQ_tma = cpasync.tma_partition(
            tma_q.atom,
            0,
            cute.make_layout(1),
            cute.group_modes(sQ_public, 0, 3),
            cute.group_modes(tCgQ_tma, 0, 3),
        )
        if const_expr(self.preordered_q_scale_tma):
            mQS_tma_cur = cute.domain_offset((q_begin, 0, 0), mQS_tma)
            gQS_tma = cute.local_tile(
                mQS_tma_cur,
                cute.slice_(self.mma_tiler, (None, 0, None)),
                (None, None, None),
            )
            tCgQS_tma = thr_mma.partition_A(gQS_tma)
            tQsQS_tma, tQgQS_tma = cpasync.tma_partition(
                tma_qs.atom,
                0,
                cute.make_layout(1),
                cute.group_modes(sQS_public, 0, 3),
                cute.group_modes(tCgQS_tma, 0, 3),
            )
            tQsQS_tma = cute.filter_zeros(tQsQS_tma)
            tQgQS_tma = cute.filter_zeros(tQgQS_tma)
        gK_tma = cute.local_tile(
            mK_tma,
            cute.slice_(self.mma_tiler, (0, None, None)),
            (None, None, None, None),
        )
        tCgK_tma = thr_mma.partition_B(gK_tma)
        tKsK_tma, tKgK_tma = cpasync.tma_partition(
            tma_k.atom,
            0,
            cute.make_layout(1),
            cute.group_modes(sK_public, 0, 3),
            cute.group_modes(tCgK_tma, 0, 3),
        )
        gKS_tma = cute.local_tile(
            mKS_tma,
            cute.slice_(self.mma_tiler, (0, None, None)),
            (None, None, None),
        )
        tCgKS_tma = thr_mma.partition_B(gKS_tma)
        tKsKS_tma, tKgKS_tma = cpasync.tma_partition(
            tma_ks.atom,
            0,
            cute.make_layout(1),
            cute.group_modes(sKS_public, 0, 3),
            cute.group_modes(tCgKS_tma, 0, 3),
        )
        tKsKS_tma = cute.filter_zeros(tKsKS_tma)
        tKgKS_tma = cute.filter_zeros(tKgKS_tma)
        sQS = sQS_public
        sKS = sKS_public

        tCrQ = tiled_mma.make_fragment_A(sQ_public)
        tCrK = tiled_mma.make_fragment_B(sK_public)
        tCcC = thr_mma.partition_C(cute.make_identity_tensor(self.mma_tiler[:2]))
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))

        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=pipeline.NamedBarrier(
                barrier_id=1,
                num_threads=32 * (self.mma_warp_id + 1),
            ),
        )

        acc_pipeline = common_pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.epi_threads_per_cta),
            defer_sync=True,
        )
        acc_producer, _ = acc_pipeline.make_participants()
        q_tma_copy_bytes = cute.size_in_bytes(_AB_DTYPE, tma_q.smem_layout)
        k_tma_copy_bytes = cute.size_in_bytes(_AB_DTYPE, tma_k.smem_layout)
        if const_expr(self.preordered_q_scale_tma):
            qs_tma_copy_bytes = cute.size_in_bytes(
                self.sf_dtype,
                cute.select(tma_qs.smem_layout, mode=[0, 1, 2]),
            )
        ks_tma_copy_bytes = cute.size_in_bytes(
            self.sf_dtype,
            cute.select(tma_ks.smem_layout, mode=[0, 1, 2]),
        )
        k_pair_tma_copy_bytes = k_tma_copy_bytes + ks_tma_copy_bytes
        q_producer, q_consumer = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.q_mbar_ptr.data_ptr(),
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            tx_count=q_tma_copy_bytes,
            defer_sync=True,
        ).make_participants()
        if const_expr(self.preordered_q_scale_tma):
            qs_producer, qs_consumer = pipeline.PipelineTmaAsync.create(
                barrier_storage=storage.qs_mbar_ptr.data_ptr(),
                num_stages=1,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                tx_count=qs_tma_copy_bytes,
                defer_sync=True,
            ).make_participants()
        k_producer, k_consumer = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.k_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            tx_count=k_pair_tma_copy_bytes,
            defer_sync=True,
        ).make_participants()
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()
        if warp_idx == self.load_warp_id:
            if group_has_visible:
                q_empty = q_producer.acquire_and_advance()
                if const_expr(self.preordered_q_scale_tma):
                    if q_scale_tma_safe:
                        qs_empty = qs_producer.acquire_and_advance()
                        cute.copy(
                            tma_qs.atom,
                            tQgQS_tma[(None, q_tile_idx, 0, hq)],
                            tQsQS_tma[(None, qs_empty.index)],
                            tma_bar_ptr=qs_empty.barrier,
                        )
                        qs_empty.commit()
                    else:
                        for row_base in cutlass.range(0, Int32(self.cta_tile_shape_mnk[0]), 32):
                            row = row_base + lane_idx
                            q_local = q_tile_start + row
                            row_major = row // Int32(32)
                            row_atom = row - row_major * Int32(32)
                            for group in cutlass.range_constexpr(self.scale_groups):
                                group_i = Int32(group)
                                mma_k = group_i // Int32(_MMA_INST_SHAPE_K // self.sf_vec_size)
                                group_in_mma_k = group_i - mma_k * Int32(_MMA_INST_SHAPE_K // self.sf_vec_size)
                                sf_coord = ((((row_atom, row_major), Int32(0)), (Int32(0), group_in_mma_k)), Int32(0), mma_k, Int32(0))
                                q_scale_row = q_begin + q_local
                                if q_local >= q_len:
                                    q_scale_row = q_begin
                                sQS[sf_coord] = mQS[q_scale_row, group_i * Int32(self.sf_vec_size), hq]
                else:
                    for row_base in cutlass.range(0, Int32(self.cta_tile_shape_mnk[0]), 32):
                        row = row_base + lane_idx
                        q_local = q_tile_start + row
                        row_major = row // Int32(32)
                        row_atom = row - row_major * Int32(32)
                        for group in cutlass.range_constexpr(self.scale_groups):
                            group_i = Int32(group)
                            mma_k = group_i // Int32(_MMA_INST_SHAPE_K // self.sf_vec_size)
                            group_in_mma_k = group_i - mma_k * Int32(_MMA_INST_SHAPE_K // self.sf_vec_size)
                            sf_coord = ((((row_atom, row_major), Int32(0)), (Int32(0), group_in_mma_k)), Int32(0), mma_k, Int32(0))
                            q_scale_row = q_begin + q_local
                            if q_local >= q_len:
                                q_scale_row = q_begin
                            sQS[sf_coord] = mQS[q_scale_row, group_i * Int32(self.sf_vec_size), hq]
                cute.copy(
                    tma_q.atom,
                    tQgQ_tma[(None, q_tile_idx, 0, hq)],
                    tQsQ_tma[(None, q_empty.index)],
                    tma_bar_ptr=q_empty.barrier,
                )
                q_empty.commit()

        if warp_idx == self.mma_warp_id:
            tmem_pool = tmem.reserve(self.num_tmem_alloc_cols)
            tCtAcc = tmem_pool.allocate_tensor(tCtAcc_fake.layout, Float32)
            # Move block scales into TMEM and issue one FP4 GEMM per visible K tile.
            tCtQS_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(q_scale_smem_layout, (None, None, None, 0)),
            )
            tCtKS_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(k_scale_smem_layout, (None, None, None, 0)),
            )
            tCtQS = tmem_pool.allocate_tensor(tCtQS_layout, self.sf_dtype)
            tCtKS = tmem_pool.allocate_tensor(tCtKS_layout, self.sf_dtype)
            copy_atom_s2t = cute.make_copy_atom(tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE), self.sf_dtype)
            tCsQS_compact = cute.filter_zeros(sQS)
            tCtQS_compact = cute.filter_zeros(tCtQS)
            tiled_copy_s2t_qs = tcgen05.make_s2t_copy(copy_atom_s2t, tCtQS_compact)
            thr_copy_s2t_qs = tiled_copy_s2t_qs.get_slice(0)
            tCsQS_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
                tiled_copy_s2t_qs,
                thr_copy_s2t_qs.partition_S(tCsQS_compact),
            )
            tCtQS_compact_s2t = thr_copy_s2t_qs.partition_D(tCtQS_compact)
            tCsKS_compact = cute.filter_zeros(sKS)
            tCtKS_compact = cute.filter_zeros(tCtKS)
            tiled_copy_s2t_ks = tcgen05.make_s2t_copy(copy_atom_s2t, tCtKS_compact)
            thr_copy_s2t_ks = tiled_copy_s2t_ks.get_slice(0)
            tCsKS_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
                tiled_copy_s2t_ks,
                thr_copy_s2t_ks.partition_S(tCsKS_compact),
            )
            tCtKS_compact_s2t = thr_copy_s2t_ks.partition_D(tCtKS_compact)
            if group_has_visible:
                q_full = q_consumer.wait_and_advance()
                if const_expr(self.preordered_q_scale_tma):
                    if q_scale_tma_safe:
                        qs_full = qs_consumer.wait_and_advance()
                        qs_full.release()
                q_full.release()
                cute.copy(tiled_copy_s2t_qs, tCsQS_compact_s2t[(None, None, None, None, 0)], tCtQS_compact_s2t)
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                q_tile_crd = (None, None, None, 0)
                if const_expr(self.is_causal):
                    causal_group_full = group_first_ktile + Int32(self.k_tiles_per_cta) <= batch_k_tiles
                    causal_group_last_ktile = group_first_ktile + Int32(self.k_tiles_per_cta - 1)
                    causal_group_full = causal_group_full and causal_group_last_ktile * Int32(_BLOCK_K) <= q_tile_last + causal_offset
                    ktile = Int32(0)
                    if causal_group_full:
                        for ktile_inner in cutlass.range_constexpr(self.k_tiles_per_cta):
                            k_pair_full = k_consumer.wait_and_advance()
                            acc_empty = acc_producer.acquire_and_advance()
                            cute.copy(tiled_copy_s2t_ks, tCsKS_compact_s2t[(None, None, None, None, k_pair_full.index)], tCtKS_compact_s2t)
                            k_tile_crd = (None, None, None, k_pair_full.index)
                            tCtAcc_stage = tCtAcc[(None, None, None, acc_empty.index)]
                            cute.gemm(tiled_mma, tCtAcc_stage, [tCrQ[q_tile_crd], tCtQS], [tCrK[k_tile_crd], tCtKS], tCtAcc_stage)
                            acc_empty.commit()
                            k_pair_full.release()
                    else:
                        for ktile_inner in cutlass.range_constexpr(self.k_tiles_per_cta):
                            ktile = group_first_ktile + Int32(ktile_inner)
                            if ktile < max_k_tiles:
                                tile_has_visible = self._tile_has_visible(
                                    q_tile_start,
                                    q_tile_last,
                                    q_len,
                                    ktile,
                                    batch_k_tiles,
                                    causal_offset,
                                )
                                if tile_has_visible:
                                    k_pair_full = k_consumer.wait_and_advance()
                                    acc_empty = acc_producer.acquire_and_advance()
                                    cute.copy(tiled_copy_s2t_ks, tCsKS_compact_s2t[(None, None, None, None, k_pair_full.index)], tCtKS_compact_s2t)
                                    k_tile_crd = (None, None, None, k_pair_full.index)
                                    tCtAcc_stage = tCtAcc[(None, None, None, acc_empty.index)]
                                    cute.gemm(tiled_mma, tCtAcc_stage, [tCrQ[q_tile_crd], tCtQS], [tCrK[k_tile_crd], tCtKS], tCtAcc_stage)
                                    acc_empty.commit()
                                    k_pair_full.release()
                else:
                    k_group_full = group_first_ktile + Int32(self.k_tiles_per_cta) <= batch_k_tiles
                    ktile = Int32(0)
                    if k_group_full:
                        for ktile_inner in cutlass.range_constexpr(self.k_tiles_per_cta):
                            k_pair_full = k_consumer.wait_and_advance()
                            acc_empty = acc_producer.acquire_and_advance()
                            cute.copy(tiled_copy_s2t_ks, tCsKS_compact_s2t[(None, None, None, None, k_pair_full.index)], tCtKS_compact_s2t)
                            k_tile_crd = (None, None, None, k_pair_full.index)
                            tCtAcc_stage = tCtAcc[(None, None, None, acc_empty.index)]
                            cute.gemm(tiled_mma, tCtAcc_stage, [tCrQ[q_tile_crd], tCtQS], [tCrK[k_tile_crd], tCtKS], tCtAcc_stage)
                            acc_empty.commit()
                            k_pair_full.release()
                    else:
                        for ktile_inner in cutlass.range_constexpr(self.k_tiles_per_cta):
                            ktile = group_first_ktile + Int32(ktile_inner)
                            if ktile < batch_k_tiles:
                                k_pair_full = k_consumer.wait_and_advance()
                                acc_empty = acc_producer.acquire_and_advance()
                                cute.copy(tiled_copy_s2t_ks, tCsKS_compact_s2t[(None, None, None, None, k_pair_full.index)], tCtKS_compact_s2t)
                                k_tile_crd = (None, None, None, k_pair_full.index)
                                tCtAcc_stage = tCtAcc[(None, None, None, acc_empty.index)]
                                cute.gemm(tiled_mma, tCtAcc_stage, [tCrQ[q_tile_crd], tCtQS], [tCrK[k_tile_crd], tCtKS], tCtAcc_stage)
                                acc_empty.commit()
                                k_pair_full.release()
                acc_producer.tail()

        if warp_idx == self.load_warp_id:
            if group_has_visible:
                load_group_full = group_first_ktile + Int32(self.k_tiles_per_cta) <= batch_k_tiles
                if const_expr(self.is_causal):
                    load_group_last_ktile = group_first_ktile + Int32(self.k_tiles_per_cta - 1)
                    load_group_full = load_group_full and load_group_last_ktile * Int32(_BLOCK_K) <= q_tile_last + causal_offset
                ktile = Int32(0)
                if load_group_full:
                    for ktile_inner in cutlass.range_constexpr(self.k_tiles_per_cta):
                        ktile = group_first_ktile + Int32(ktile_inner)
                        k_pair_empty = k_producer.acquire_and_advance()
                        physical_page = mKvIndices[page_begin + ktile]
                        cute.copy(
                            tma_k.atom,
                            tKgK_tma[(None, 0, 0, hk, physical_page)],
                            tKsK_tma[(None, k_pair_empty.index)],
                            tma_bar_ptr=k_pair_empty.barrier,
                        )
                        scale_l = physical_page * heads_k + hk
                        cute.copy(
                            tma_ks.atom,
                            tKgKS_tma[(None, 0, 0, scale_l)],
                            tKsKS_tma[(None, k_pair_empty.index)],
                            tma_bar_ptr=k_pair_empty.barrier,
                        )
                        k_pair_empty.commit()
                else:
                    for ktile_inner in cutlass.range_constexpr(self.k_tiles_per_cta):
                        ktile = group_first_ktile + Int32(ktile_inner)
                        if ktile < max_k_tiles:
                            tile_has_visible = self._tile_has_visible(
                                q_tile_start,
                                q_tile_last,
                                q_len,
                                ktile,
                                batch_k_tiles,
                                causal_offset,
                            )
                            if tile_has_visible:
                                k_pair_empty = k_producer.acquire_and_advance()
                                physical_page = mKvIndices[page_begin + ktile]
                                cute.copy(
                                    tma_k.atom,
                                    tKgK_tma[(None, 0, 0, hk, physical_page)],
                                    tKsK_tma[(None, k_pair_empty.index)],
                                    tma_bar_ptr=k_pair_empty.barrier,
                                )
                                scale_l = physical_page * heads_k + hk
                                cute.copy(
                                    tma_ks.atom,
                                    tKgKS_tma[(None, 0, 0, scale_l)],
                                    tKsKS_tma[(None, k_pair_empty.index)],
                                    tma_bar_ptr=k_pair_empty.barrier,
                                )
                                k_pair_empty.commit()
                k_producer.tail()
                q_producer.tail()
                if const_expr(self.preordered_q_scale_tma):
                    if q_scale_tma_safe:
                        qs_producer.tail()

        if warp_idx < self.mma_warp_id:
            tmem_pool = tmem.reserve(self.num_tmem_alloc_cols)
            tCtAcc = tmem_pool.allocate_tensor(tCtAcc_fake.layout, Float32)
            # Load accumulators from TMEM, reduce per-row max, and store scores.
            if const_expr(self.use_tmem_load_red):
                copy_atom_t2r = cute.make_copy_atom(
                    tcgen05.LdRed32x32bOp(
                        tcgen05.Repetition.x128,
                        tcgen05.Pack.NONE,
                        tcgen05.TmemLoadRedOp.MAX,
                    ),
                    Float32,
                )
            else:
                copy_atom_t2r = cute.make_copy_atom(
                    tcgen05.Ld32x32bOp(tcgen05.Repetition.x128, tcgen05.Pack.NONE),
                    Float32,
                )
            tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc[(None, None, None, 0)])
            thr_copy_t2r = tiled_copy_t2r.get_slice(epi_tidx)
            tTR_tAcc = thr_copy_t2r.partition_S(tCtAcc)
            tTR_cC = thr_copy_t2r.partition_D(tCcC)
            tTR_rAcc = cute.make_rmem_tensor(tTR_cC.shape, Float32)
            if const_expr(self.use_tmem_load_red):
                tTR_rRed = cute.make_rmem_tensor((1,), Float32)
            q_local_store0 = q_tile_start + epi_tidx
            q_global_store0 = q_begin + q_local_store0
            if const_expr(self.cta_tile_shape_mnk[0] > self.epi_threads_per_cta):
                q_local_store1 = q_tile_start + epi_tidx + Int32(self.epi_threads_per_cta)
                q_global_store1 = q_begin + q_local_store1
            if group_has_visible:
                visible_tile_count = Int32(0)
                for ktile_inner in cutlass.range_constexpr(self.k_tiles_per_cta):
                    ktile = group_first_ktile + Int32(ktile_inner)
                    if ktile < max_k_tiles:
                        tile_has_visible = self._tile_has_visible(
                            q_tile_start,
                            q_tile_last,
                            q_len,
                            ktile,
                            batch_k_tiles,
                            causal_offset,
                        )
                        if tile_has_visible:
                            epilogue_owns_tile = epi_warpgroup_idx == Int32(
                                ktile_inner % self.num_epi_warpgroups
                            )
                            if epilogue_owns_tile:
                                acc_stage_index = visible_tile_count % Int32(self.num_acc_stage)
                                acc_stage_phase = (visible_tile_count // Int32(self.num_acc_stage)) % Int32(2)
                                tile_mask_free = self._tile_mask_free(q_tile_start, ktile, causal_offset)
                                k_tile_full = ktile * Int32(_BLOCK_K) + Int32(_BLOCK_K - 1) < k_len
                                tile_full = q_tile_full and k_tile_full
                                acc_pipeline.consumer_wait_w_index_phase(acc_stage_index, acc_stage_phase)
                                tTR_tAcc_stage = tTR_tAcc[(None, None, None, None, acc_stage_index)]
                                if const_expr(self.use_tmem_load_red):
                                    cute.copy(tiled_copy_t2r, tTR_tAcc_stage, [tTR_rAcc, tTR_rRed])
                                else:
                                    cute.copy(tiled_copy_t2r, tTR_tAcc_stage, tTR_rAcc)
                                row_max0 = -Float32.inf
                                row_max1 = -Float32.inf
                                if tile_mask_free:
                                    if tile_full:
                                        if const_expr(not self.use_tmem_load_red or self.cta_tile_shape_mnk[0] > self.epi_threads_per_cta):
                                            for i in cutlass.range(cute.size(tTR_rAcc), unroll_full=True):
                                                coord_m, _ = tTR_cC[i]
                                                if coord_m == epi_tidx:
                                                    row_max0 = cute.arch.fmax(row_max0, tTR_rAcc[i])
                                                if const_expr(self.cta_tile_shape_mnk[0] > self.epi_threads_per_cta):
                                                    if coord_m == epi_tidx + Int32(self.epi_threads_per_cta):
                                                        row_max1 = cute.arch.fmax(row_max1, tTR_rAcc[i])
                                        else:
                                            row_max0 = tTR_rRed[0]
                                    else:
                                        for i in cutlass.range(cute.size(tTR_rAcc), unroll_full=True):
                                            coord_m, coord_n = tTR_cC[i]
                                            q_local = q_tile_start + coord_m
                                            k_local = ktile * Int32(_BLOCK_K) + coord_n
                                            if coord_m == epi_tidx and q_local < q_len and k_local < k_len:
                                                row_max0 = cute.arch.fmax(row_max0, tTR_rAcc[i])
                                            if const_expr(self.cta_tile_shape_mnk[0] > self.epi_threads_per_cta):
                                                if coord_m == epi_tidx + Int32(self.epi_threads_per_cta) and q_local < q_len and k_local < k_len:
                                                    row_max1 = cute.arch.fmax(row_max1, tTR_rAcc[i])
                                else:
                                    if tile_full:
                                        for i in cutlass.range(cute.size(tTR_rAcc), unroll_full=True):
                                            coord_m, coord_n = tTR_cC[i]
                                            q_local = q_tile_start + coord_m
                                            k_local = ktile * Int32(_BLOCK_K) + coord_n
                                            if self._full_tile_coord_visible(
                                                coord_m,
                                                epi_tidx,
                                                q_local,
                                                k_local,
                                                causal_offset,
                                            ):
                                                row_max0 = cute.arch.fmax(row_max0, tTR_rAcc[i])
                                            if const_expr(self.cta_tile_shape_mnk[0] > self.epi_threads_per_cta):
                                                if self._full_tile_coord_visible(
                                                    coord_m,
                                                    epi_tidx + Int32(self.epi_threads_per_cta),
                                                    q_local,
                                                    k_local,
                                                    causal_offset,
                                                ):
                                                    row_max1 = cute.arch.fmax(row_max1, tTR_rAcc[i])
                                    else:
                                        for i in cutlass.range(cute.size(tTR_rAcc), unroll_full=True):
                                            coord_m, coord_n = tTR_cC[i]
                                            q_local = q_tile_start + coord_m
                                            k_local = ktile * Int32(_BLOCK_K) + coord_n
                                            if self._partial_tile_coord_visible(
                                                coord_m,
                                                epi_tidx,
                                                q_local,
                                                k_local,
                                                q_len,
                                                k_len,
                                                causal_offset,
                                            ):
                                                row_max0 = cute.arch.fmax(row_max0, tTR_rAcc[i])
                                            if const_expr(self.cta_tile_shape_mnk[0] > self.epi_threads_per_cta):
                                                if self._partial_tile_coord_visible(
                                                    coord_m,
                                                    epi_tidx + Int32(self.epi_threads_per_cta),
                                                    q_local,
                                                    k_local,
                                                    q_len,
                                                    k_len,
                                                    causal_offset,
                                                ):
                                                    row_max1 = cute.arch.fmax(row_max1, tTR_rAcc[i])
                                if q_tile_full:
                                    mScores[hq, ktile, q_global_store0] = row_max0
                                elif q_local_store0 < q_len:
                                    mScores[hq, ktile, q_global_store0] = row_max0
                                if const_expr(self.cta_tile_shape_mnk[0] > self.epi_threads_per_cta):
                                    if q_tile_full:
                                        mScores[hq, ktile, q_global_store1] = row_max1
                                    elif q_local_store1 < q_len:
                                        mScores[hq, ktile, q_global_store1] = row_max1
                                cute.arch.fence_view_async_tmem_load()
                                acc_pipeline.consumer_release_w_index(acc_stage_index)
                            visible_tile_count += Int32(1)
                        else:
                            if const_expr(not self.compact_schedule):
                                if epi_warpgroup_idx == Int32(0):
                                    if q_tile_full:
                                        mScores[hq, ktile, q_global_store0] = -Float32.inf
                                    elif q_local_store0 < q_len:
                                        mScores[hq, ktile, q_global_store0] = -Float32.inf
            else:
                if const_expr(not self.compact_schedule):
                    if epi_warpgroup_idx == Int32(0):
                        for ktile_inner in cutlass.range_constexpr(self.k_tiles_per_cta):
                            ktile = ktile_group * Int32(self.k_tiles_per_cta) + Int32(ktile_inner)
                            if ktile < max_k_tiles:
                                if q_tile_full:
                                    mScores[hq, ktile, q_global_store0] = -Float32.inf
                                elif q_local_store0 < q_len:
                                    mScores[hq, ktile, q_global_store0] = -Float32.inf
            cute.arch.barrier()
            tmem.free(tmem_pool.base_ptr)

class Fp4IndexerDecodeQPackSm100:
    """Pack decode Q rows as ``[B * Hk, 128, 64]`` and pack Q scales to MMA storage."""

    def __init__(self, *, fmt: str):
        spec = normalize_fp4_format(fmt)
        self.fmt = spec.name
        self.sf_dtype = spec.cutlass_scale_dtype
        self.scale_groups = spec.scale_groups
        self.threads_per_cta = 256

    @cute.jit
    def __call__(
        self,
        q_ptr: cute.Pointer,
        q_scale_ptr: cute.Pointer,
        q_pack_ptr: cute.Pointer,
        q_scale_pack_ptr: cute.Pointer,
        cu_seqlens_q_ptr: cute.Pointer,
        problem_size: tuple,
        stream: cuda.CUstream,
    ):
        total_q, heads_q, heads_k, batch = problem_size
        rest_q_m = cute.ceil_div(total_q, 128)
        rest_g = ceil_div(self.scale_groups, 4)
        q = cute.make_tensor(
            q_ptr,
            cute.make_layout(
                (total_q, heads_q, _FP4_PACKED_D_BYTES),
                stride=(heads_q * _FP4_PACKED_D_BYTES, _FP4_PACKED_D_BYTES, 1),
            ),
        )
        q_scale = cute.make_tensor(
            q_scale_ptr,
            cute.make_layout(
                (heads_q, rest_q_m, rest_g, 32, 4, 4),
                stride=(512 * rest_q_m * rest_g, 512 * rest_g, 512, 16, 4, 1),
            ),
        )
        q_pack_l = batch * heads_k
        q_pack = cute.make_tensor(
            q_pack_ptr,
            cute.make_layout(
                (q_pack_l, _PAGE_SIZE, _FP4_PACKED_D_BYTES),
                stride=(_PAGE_SIZE * _FP4_PACKED_D_BYTES, _FP4_PACKED_D_BYTES, 1),
            ),
        )
        q_scale_pack = cute.make_tensor(
            q_scale_pack_ptr,
            cute.make_layout(
                (q_pack_l, 1, rest_g, 32, 4, 4),
                stride=(512 * rest_g, 512 * rest_g, 512, 16, 4, 1),
            ),
        )
        cu_q = cute.make_tensor(cu_seqlens_q_ptr, cute.make_layout((batch + 1,), stride=(1,)))
        self.kernel(q, q_scale, q_pack, q_scale_pack, cu_q, heads_q, heads_k).launch(
            grid=(q_pack_l, 1, 1),
            block=[self.threads_per_cta, 1, 1],
            cluster=(1, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mQS: cute.Tensor,
        mQPack: cute.Tensor,
        mQSPack: cute.Tensor,
        mCuQ: cute.Tensor,
        heads_q: Int32,
        heads_k: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        q_pack_l, _, _ = cute.arch.block_idx()
        batch_idx = q_pack_l // heads_k
        hk = q_pack_l - batch_idx * heads_k
        q_begin = mCuQ[batch_idx]
        q_end = mCuQ[batch_idx + 1]
        q_len = q_end - q_begin
        qhead_per_kv = heads_q // heads_k

        linear = tidx
        while linear < Int32(_PAGE_SIZE * _FP4_PACKED_D_BYTES):
            row = linear // Int32(_FP4_PACKED_D_BYTES)
            byte = linear - row * Int32(_FP4_PACKED_D_BYTES)
            h_in_group = row // Int32(_DECODE_PACK_Q_LEN)
            q_local = row - h_in_group * Int32(_DECODE_PACK_Q_LEN)
            hq = hk * qhead_per_kv + h_in_group
            if q_local < q_len and h_in_group < qhead_per_kv:
                mQPack[q_pack_l, row, byte] = mQ[q_begin + q_local, hq, byte]
            else:
                mQPack[q_pack_l, row, byte] = cutlass.Uint8(0)
            linear += Int32(self.threads_per_cta)

        scale_linear = tidx
        while scale_linear < Int32(_PAGE_SIZE * self.scale_groups):
            row = scale_linear // Int32(self.scale_groups)
            group = scale_linear - row * Int32(self.scale_groups)
            h_in_group = row // Int32(_DECODE_PACK_Q_LEN)
            q_local = row - h_in_group * Int32(_DECODE_PACK_Q_LEN)
            hq = hk * qhead_per_kv + h_in_group
            q_abs = q_begin + q_local
            if q_local >= q_len or h_in_group >= qhead_per_kv:
                q_abs = q_begin
                hq = hk * qhead_per_kv
            src_rest_m = q_abs // Int32(128)
            src_row = q_abs - src_rest_m * Int32(128)
            src_row_atom = src_row % Int32(32)
            src_row_major = src_row // Int32(32)
            dst_row_atom = row % Int32(32)
            dst_row_major = row // Int32(32)
            rest_g = group // Int32(4)
            group_in_rest = group - rest_g * Int32(4)
            mQSPack[q_pack_l, Int32(0), rest_g, dst_row_atom, dst_row_major, group_in_rest] = mQS[
                hq, src_rest_m, rest_g, src_row_atom, src_row_major, group_in_rest
            ]
            scale_linear += Int32(self.threads_per_cta)


class Fp4IndexerDecodePackedQSm100:
    """Decode score kernel with M packed as ``qhead_per_kv * q_len == 128``."""

    def __init__(self, *, fmt: str, causal: bool, compact_schedule: bool, use_tmem_load_red: bool = False):
        spec = normalize_fp4_format(fmt)
        self.fmt = spec.name
        self.is_causal = bool(causal)
        self.compact_schedule = bool(compact_schedule)
        self.use_tmem_load_red = bool(use_tmem_load_red)
        self.sf_vec_size = spec.sf_vec_size
        self.sf_dtype = spec.cutlass_scale_dtype
        self.use_nvfp4 = spec.name == "nvfp4"
        self.epi_threads_per_cta = 128
        self.epi_warps_per_group = 4
        self.num_epi_warpgroups = 2
        self.mma_warp_id = self.epi_warps_per_group * self.num_epi_warpgroups
        self.load_warp_id = self.mma_warp_id + 1
        self.threads_per_cta = 384
        self.num_tmem_alloc_cols = 512
        self.num_q_stage = 1
        self.num_acc_stage = 3
        self.num_ab_stage = 3
        self.k_tiles_per_cta = _DECODE_K_TILES_PER_CTA
        self.mma_tiler = (_MMA_TILER_MN[0], _MMA_TILER_MN[1], _MMA_INST_SHAPE_K * 2)
        self.cta_tile_shape_mnk = self.mma_tiler

    @cute.jit
    def __call__(
        self,
        q_pack_ptr: cute.Pointer,
        k_ptr: cute.Pointer,
        q_scale_pack_ptr: cute.Pointer,
        k_scale_ptr: cute.Pointer,
        scores_ptr: cute.Pointer,
        kv_indices_ptr: cute.Pointer,
        cu_seqlens_q_ptr: cute.Pointer,
        cu_seqlens_k_ptr: cute.Pointer,
        cu_page_offsets_ptr: cute.Pointer,
        qo_offset_ptr: cute.Pointer,
        problem_size: tuple,
        stream: cuda.CUstream,
    ):
        (
            _,
            _,
            _,
            _,
            lk,
            heads_q,
            heads_k,
            batch,
            max_k_tiles,
            total_q,
            has_qo_offset,
        ) = problem_size
        page_count = lk // heads_k
        q_pack_l = batch * heads_k
        q_tma_tensor = cute.make_tensor(
            cute.recast_ptr(q_pack_ptr, dtype=_AB_DTYPE),
            cute.make_layout(
                (_PAGE_SIZE, _HEAD_DIM, q_pack_l),
                stride=(_HEAD_DIM, 1, _PAGE_SIZE * _HEAD_DIM),
            ),
        )
        k_tma_tensor = cute.make_tensor(
            cute.recast_ptr(k_ptr, dtype=_AB_DTYPE),
            cute.make_layout(
                (_PAGE_SIZE, _HEAD_DIM, heads_k, page_count),
                stride=(_HEAD_DIM, 1, _PAGE_SIZE * _HEAD_DIM, heads_k * _PAGE_SIZE * _HEAD_DIM),
            ),
        )
        q_scale_tensor = cute.make_tensor(
            q_scale_pack_ptr,
            blockscaled_utils.tile_atom_to_shape_SF(
                (_PAGE_SIZE, _HEAD_DIM, q_pack_l),
                self.sf_vec_size,
            ),
        )
        k_scale_tensor = cute.make_tensor(
            k_scale_ptr,
            blockscaled_utils.tile_atom_to_shape_SF(
                (_PAGE_SIZE, _HEAD_DIM, page_count * heads_k),
                self.sf_vec_size,
            ),
        )
        scores_tensor = cute.make_tensor(
            scores_ptr,
            cute.make_layout((heads_q, max_k_tiles, total_q), stride=(max_k_tiles * total_q, total_q, 1)),
        )
        kv_indices_tensor = cute.make_tensor(kv_indices_ptr, cute.make_layout((page_count,), stride=(1,)))
        cu_layout = cute.make_layout((batch + 1,), stride=(1,))
        cu_q_tensor = cute.make_tensor(cu_seqlens_q_ptr, cu_layout)
        cu_k_tensor = cute.make_tensor(cu_seqlens_k_ptr, cu_layout)
        cu_page_offsets_tensor = cute.make_tensor(cu_page_offsets_ptr, cu_layout)
        qo_offset_tensor = cute.make_tensor(qo_offset_ptr, cute.make_layout((batch,), stride=(1,)))

        if const_expr(self.use_nvfp4):
            mma_op = tcgen05.MmaMXF4NVF4Op(
                self.sf_dtype,
                (*_MMA_TILER_MN, _MMA_INST_SHAPE_K),
                tcgen05.CtaGroup.ONE,
                tcgen05.OperandSource.SMEM,
            )
        else:
            mma_op = tcgen05.MmaMXF4Op(
                (*_MMA_TILER_MN, _MMA_INST_SHAPE_K),
                tcgen05.CtaGroup.ONE,
                tcgen05.OperandSource.SMEM,
            )
        tiled_mma = cute.make_tiled_mma(mma_op)
        q_smem_layout = sm100_utils.make_smem_layout_a(tiled_mma, self.mma_tiler, _AB_DTYPE, self.num_q_stage)
        k_smem_layout = sm100_utils.make_smem_layout_b(tiled_mma, self.mma_tiler, _AB_DTYPE, self.num_ab_stage)
        q_scale_smem_layout = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_q_stage,
        )
        k_scale_smem_layout = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_ab_stage,
        )
        cluster_layout_vmnk = cute.make_layout((1, 1, 1, 1))
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        q_smem_layout_stage = cute.slice_(q_smem_layout, (None, None, None, 0))
        k_smem_layout_stage = cute.slice_(k_smem_layout, (None, None, None, 0))
        tma_q = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            q_tma_tensor,
            q_smem_layout_stage,
            self.mma_tiler,
            tiled_mma,
            cluster_layout_vmnk.shape,
        )
        tma_k = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            k_tma_tensor,
            k_smem_layout_stage,
            self.mma_tiler,
            tiled_mma,
            cluster_layout_vmnk.shape,
        )
        tma_qs = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            q_scale_tensor,
            q_scale_smem_layout,
            self.mma_tiler,
            tiled_mma,
            cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        tma_ks = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            k_scale_tensor,
            k_scale_smem_layout,
            self.mma_tiler,
            tiled_mma,
            cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        grid_k_groups = cute.ceil_div(max_k_tiles, self.k_tiles_per_cta)
        compact_k_groups = cute.ceil_div(page_count + batch * (self.k_tiles_per_cta - 1), self.k_tiles_per_cta)
        if const_expr(self.compact_schedule):
            grid = (compact_k_groups, heads_k, 1)
        else:
            grid = (grid_k_groups, batch * heads_k, 1)
        self.kernel(
            tiled_mma,
            tma_q,
            tma_qs,
            tma_k,
            tma_ks,
            scores_tensor,
            kv_indices_tensor,
            cu_q_tensor,
            cu_k_tensor,
            cu_page_offsets_tensor,
            qo_offset_tensor,
            q_smem_layout,
            k_smem_layout,
            q_scale_smem_layout,
            k_scale_smem_layout,
            heads_q,
            heads_k,
            batch,
            has_qo_offset,
            max_k_tiles,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(1, 1, 1),
            stream=stream,
        )

    @cute.jit
    def _group_has_visible(
        self,
        q_len: Int32,
        group_first_ktile: Int32,
        batch_k_tiles: Int32,
        causal_offset: Int32,
    ):
        visible = q_len > Int32(0) and group_first_ktile < batch_k_tiles
        if const_expr(self.is_causal):
            visible = visible and group_first_ktile * Int32(_BLOCK_K) <= (q_len - Int32(1)) + causal_offset
        return visible

    @cute.jit
    def _tile_has_visible(
        self,
        q_len: Int32,
        ktile: Int32,
        batch_k_tiles: Int32,
        causal_offset: Int32,
    ):
        visible = ktile < batch_k_tiles
        if const_expr(self.is_causal):
            visible = visible and ktile * Int32(_BLOCK_K) <= (q_len - Int32(1)) + causal_offset
        return visible

    @cute.jit
    def _tile_mask_free(self, ktile: Int32, causal_offset: Int32):
        if const_expr(self.is_causal):
            return ktile * Int32(_BLOCK_K) + Int32(_BLOCK_K - 1) <= causal_offset
        return True

    @cute.jit
    def _packed_coord_visible(
        self,
        coord_m: Int32,
        target_m: Int32,
        h_in_group: Int32,
        qhead_per_kv: Int32,
        q_local: Int32,
        q_len: Int32,
        k_local: Int32,
        k_len: Int32,
        causal_offset: Int32,
    ):
        visible = coord_m == target_m and h_in_group < qhead_per_kv and q_local < q_len and k_local < k_len
        if const_expr(self.is_causal):
            visible = visible and k_local <= q_local + causal_offset
        return visible

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_q: cpasync.TmaInfo,
        tma_qs: cpasync.TmaInfo,
        tma_k: cpasync.TmaInfo,
        tma_ks: cpasync.TmaInfo,
        mScores: cute.Tensor,
        mKvIndices: cute.Tensor,
        mCuQ: cute.Tensor,
        mCuK: cute.Tensor,
        mCuPages: cute.Tensor,
        mQoOffset: cute.Tensor,
        q_smem_layout: cute.ComposedLayout,
        k_smem_layout: cute.ComposedLayout,
        q_scale_smem_layout: cute.Layout,
        k_scale_smem_layout: cute.Layout,
        heads_q: Int32,
        heads_k: Int32,
        batch: Int32,
        has_qo_offset: Int32,
        max_k_tiles: Int32,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        epi_tidx = tidx % Int32(self.epi_threads_per_cta)
        epi_warpgroup_idx = warp_idx // Int32(self.epi_warps_per_group)
        task_x, task_y, _ = cute.arch.block_idx()
        task_valid = True
        batch_idx = Int32(0)
        hk = Int32(0)
        ktile_group = Int32(0)
        q_l = Int32(0)
        if const_expr(self.compact_schedule):
            hk = task_y
            group_base = Int32(0)
            scan_batch = Int32(0)
            task_valid = False
            while scan_batch < batch and not task_valid:
                batch_pages = mCuPages[scan_batch + Int32(1)] - mCuPages[scan_batch]
                batch_groups = (batch_pages + Int32(self.k_tiles_per_cta - 1)) // Int32(self.k_tiles_per_cta)
                task_valid = task_x < group_base + batch_groups
                if not task_valid:
                    group_base += batch_groups
                    scan_batch += Int32(1)
            if task_valid:
                batch_idx = scan_batch
                ktile_group = task_x - group_base
            q_l = batch_idx * heads_k + hk
        else:
            ktile_group = task_x
            q_l = task_y
            batch_idx = q_l // heads_k
            hk = q_l - batch_idx * heads_k
        qhead_per_kv = heads_q // heads_k
        q_begin = mCuQ[batch_idx]
        q_end = mCuQ[batch_idx + 1]
        k_begin = mCuK[batch_idx]
        k_end = mCuK[batch_idx + 1]
        q_len = q_end - q_begin
        k_len = k_end - k_begin
        if const_expr(self.compact_schedule):
            if not task_valid:
                q_len = Int32(0)
                k_len = Int32(0)
        page_begin = mCuPages[batch_idx]
        batch_k_tiles = (k_len + Int32(_PAGE_SIZE - 1)) // Int32(_PAGE_SIZE)
        causal_offset = Int32(0)
        if const_expr(self.is_causal):
            causal_offset = k_len - q_len
            if has_qo_offset != 0:
                causal_offset = mQoOffset[batch_idx]
        group_first_ktile = ktile_group * Int32(self.k_tiles_per_cta)
        group_has_visible = self._group_has_visible(
            q_len,
            group_first_ktile,
            batch_k_tiles,
            causal_offset,
        )

        @cute.struct
        class SharedStorage:
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            k_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sQ_public = smem.allocate_tensor(_AB_DTYPE, q_smem_layout.outer, 128, swizzle=q_smem_layout.inner)
        sK_public = smem.allocate_tensor(_AB_DTYPE, k_smem_layout.outer, 128, swizzle=k_smem_layout.inner)
        sQS_public = smem.allocate_tensor(self.sf_dtype, q_scale_smem_layout, 128)
        sKS_public = smem.allocate_tensor(self.sf_dtype, k_scale_smem_layout, 128)
        mQ_tma = tma_q.tma_tensor
        mQS_tma = tma_qs.tma_tensor
        mK_tma = tma_k.tma_tensor
        mKS_tma = tma_ks.tma_tensor
        thr_mma = tiled_mma.get_slice(0)
        tCrQ = tiled_mma.make_fragment_A(sQ_public)
        tCrK = tiled_mma.make_fragment_B(sK_public)
        tCcC = thr_mma.partition_C(cute.make_identity_tensor(self.mma_tiler[:2]))
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))

        gQ_tma = cute.local_tile(
            mQ_tma,
            cute.slice_(self.mma_tiler, (None, 0, None)),
            (None, None, None),
        )
        tCgQ_tma = thr_mma.partition_A(gQ_tma)
        tQsQ_tma, tQgQ_tma = cpasync.tma_partition(
            tma_q.atom,
            0,
            cute.make_layout(1),
            cute.group_modes(sQ_public, 0, 3),
            cute.group_modes(tCgQ_tma, 0, 3),
        )
        gQS_tma = cute.local_tile(
            mQS_tma,
            cute.slice_(self.mma_tiler, (None, 0, None)),
            (None, None, None),
        )
        tCgQS_tma = thr_mma.partition_A(gQS_tma)
        tQsQS_tma, tQgQS_tma = cpasync.tma_partition(
            tma_qs.atom,
            0,
            cute.make_layout(1),
            cute.group_modes(sQS_public, 0, 3),
            cute.group_modes(tCgQS_tma, 0, 3),
        )
        tQsQS_tma = cute.filter_zeros(tQsQS_tma)
        tQgQS_tma = cute.filter_zeros(tQgQS_tma)
        gK_tma = cute.local_tile(
            mK_tma,
            cute.slice_(self.mma_tiler, (0, None, None)),
            (None, None, None, None),
        )
        tCgK_tma = thr_mma.partition_B(gK_tma)
        tKsK_tma, tKgK_tma = cpasync.tma_partition(
            tma_k.atom,
            0,
            cute.make_layout(1),
            cute.group_modes(sK_public, 0, 3),
            cute.group_modes(tCgK_tma, 0, 3),
        )
        gKS_tma = cute.local_tile(
            mKS_tma,
            cute.slice_(self.mma_tiler, (0, None, None)),
            (None, None, None),
        )
        tCgKS_tma = thr_mma.partition_B(gKS_tma)
        tKsKS_tma, tKgKS_tma = cpasync.tma_partition(
            tma_ks.atom,
            0,
            cute.make_layout(1),
            cute.group_modes(sKS_public, 0, 3),
            cute.group_modes(tCgKS_tma, 0, 3),
        )
        tKsKS_tma = cute.filter_zeros(tKsKS_tma)
        tKgKS_tma = cute.filter_zeros(tKgKS_tma)

        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=pipeline.NamedBarrier(
                barrier_id=1,
                num_threads=32 * (self.mma_warp_id + 1),
            ),
        )
        acc_pipeline = common_pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.epi_threads_per_cta),
            defer_sync=True,
        )
        acc_producer, _ = acc_pipeline.make_participants()
        q_tma_copy_bytes = cute.size_in_bytes(_AB_DTYPE, tma_q.smem_layout)
        qs_tma_copy_bytes = cute.size_in_bytes(
            self.sf_dtype,
            cute.select(tma_qs.smem_layout, mode=[0, 1, 2]),
        )
        k_tma_copy_bytes = cute.size_in_bytes(_AB_DTYPE, tma_k.smem_layout)
        ks_tma_copy_bytes = cute.size_in_bytes(
            self.sf_dtype,
            cute.select(tma_ks.smem_layout, mode=[0, 1, 2]),
        )
        q_pair_tma_copy_bytes = q_tma_copy_bytes + qs_tma_copy_bytes
        k_pair_tma_copy_bytes = k_tma_copy_bytes + ks_tma_copy_bytes
        q_producer, q_consumer = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.q_mbar_ptr.data_ptr(),
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            tx_count=q_pair_tma_copy_bytes,
            defer_sync=True,
        ).make_participants()
        k_producer, k_consumer = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.k_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            tx_count=k_pair_tma_copy_bytes,
            defer_sync=True,
        ).make_participants()
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        if warp_idx == self.load_warp_id:
            if group_has_visible:
                q_pair_empty = q_producer.acquire_and_advance()
                cute.copy(
                    tma_q.atom,
                    tQgQ_tma[(None, 0, 0, q_l)],
                    tQsQ_tma[(None, q_pair_empty.index)],
                    tma_bar_ptr=q_pair_empty.barrier,
                )
                cute.copy(
                    tma_qs.atom,
                    tQgQS_tma[(None, 0, 0, q_l)],
                    tQsQS_tma[(None, q_pair_empty.index)],
                    tma_bar_ptr=q_pair_empty.barrier,
                )
                q_pair_empty.commit()
                load_group_full = group_first_ktile + Int32(self.k_tiles_per_cta) <= batch_k_tiles
                if const_expr(self.is_causal):
                    load_group_last_ktile = group_first_ktile + Int32(self.k_tiles_per_cta - 1)
                    load_group_full = load_group_full and load_group_last_ktile * Int32(_BLOCK_K) <= (q_len - Int32(1)) + causal_offset
                if load_group_full:
                    for ktile_inner in cutlass.range_constexpr(self.k_tiles_per_cta):
                        ktile = group_first_ktile + Int32(ktile_inner)
                        k_pair_empty = k_producer.acquire_and_advance()
                        physical_page = mKvIndices[page_begin + ktile]
                        cute.copy(
                            tma_k.atom,
                            tKgK_tma[(None, 0, 0, hk, physical_page)],
                            tKsK_tma[(None, k_pair_empty.index)],
                            tma_bar_ptr=k_pair_empty.barrier,
                        )
                        scale_l = physical_page * heads_k + hk
                        cute.copy(
                            tma_ks.atom,
                            tKgKS_tma[(None, 0, 0, scale_l)],
                            tKsKS_tma[(None, k_pair_empty.index)],
                            tma_bar_ptr=k_pair_empty.barrier,
                        )
                        k_pair_empty.commit()
                else:
                    for ktile_inner in cutlass.range_constexpr(self.k_tiles_per_cta):
                        ktile = group_first_ktile + Int32(ktile_inner)
                        if ktile < max_k_tiles:
                            tile_has_visible = self._tile_has_visible(
                                q_len,
                                ktile,
                                batch_k_tiles,
                                causal_offset,
                            )
                            if tile_has_visible:
                                k_pair_empty = k_producer.acquire_and_advance()
                                physical_page = mKvIndices[page_begin + ktile]
                                cute.copy(
                                    tma_k.atom,
                                    tKgK_tma[(None, 0, 0, hk, physical_page)],
                                    tKsK_tma[(None, k_pair_empty.index)],
                                    tma_bar_ptr=k_pair_empty.barrier,
                                )
                                scale_l = physical_page * heads_k + hk
                                cute.copy(
                                    tma_ks.atom,
                                    tKgKS_tma[(None, 0, 0, scale_l)],
                                    tKsKS_tma[(None, k_pair_empty.index)],
                                    tma_bar_ptr=k_pair_empty.barrier,
                                )
                                k_pair_empty.commit()
                k_producer.tail()
                q_producer.tail()

        if warp_idx == self.mma_warp_id:
            tmem_pool = tmem.reserve(self.num_tmem_alloc_cols)
            tCtAcc = tmem_pool.allocate_tensor(tCtAcc_fake.layout, Float32)
            tCtQS_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(q_scale_smem_layout, (None, None, None, 0)),
            )
            tCtKS_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(k_scale_smem_layout, (None, None, None, 0)),
            )
            tCtQS = tmem_pool.allocate_tensor(tCtQS_layout, self.sf_dtype)
            tCtKS = tmem_pool.allocate_tensor(tCtKS_layout, self.sf_dtype)
            copy_atom_s2t = cute.make_copy_atom(tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE), self.sf_dtype)
            tCsQS_compact = cute.filter_zeros(sQS_public)
            tCtQS_compact = cute.filter_zeros(tCtQS)
            tiled_copy_s2t_qs = tcgen05.make_s2t_copy(copy_atom_s2t, tCtQS_compact)
            thr_copy_s2t_qs = tiled_copy_s2t_qs.get_slice(0)
            tCsQS_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
                tiled_copy_s2t_qs,
                thr_copy_s2t_qs.partition_S(tCsQS_compact),
            )
            tCtQS_compact_s2t = thr_copy_s2t_qs.partition_D(tCtQS_compact)
            tCsKS_compact = cute.filter_zeros(sKS_public)
            tCtKS_compact = cute.filter_zeros(tCtKS)
            tiled_copy_s2t_ks = tcgen05.make_s2t_copy(copy_atom_s2t, tCtKS_compact)
            thr_copy_s2t_ks = tiled_copy_s2t_ks.get_slice(0)
            tCsKS_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
                tiled_copy_s2t_ks,
                thr_copy_s2t_ks.partition_S(tCsKS_compact),
            )
            tCtKS_compact_s2t = thr_copy_s2t_ks.partition_D(tCtKS_compact)
            if group_has_visible:
                q_pair_full = q_consumer.wait_and_advance()
                q_pair_full.release()
                cute.copy(tiled_copy_s2t_qs, tCsQS_compact_s2t[(None, None, None, None, 0)], tCtQS_compact_s2t)
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                q_tile_crd = (None, None, None, 0)
                if const_expr(self.is_causal):
                    causal_group_full = group_first_ktile + Int32(self.k_tiles_per_cta) <= batch_k_tiles
                    causal_group_last_ktile = group_first_ktile + Int32(self.k_tiles_per_cta - 1)
                    causal_group_full = causal_group_full and causal_group_last_ktile * Int32(_BLOCK_K) <= (q_len - Int32(1)) + causal_offset
                    if causal_group_full:
                        for ktile_inner in cutlass.range_constexpr(self.k_tiles_per_cta):
                            k_pair_full = k_consumer.wait_and_advance()
                            acc_empty = acc_producer.acquire_and_advance()
                            cute.copy(tiled_copy_s2t_ks, tCsKS_compact_s2t[(None, None, None, None, k_pair_full.index)], tCtKS_compact_s2t)
                            k_tile_crd = (None, None, None, k_pair_full.index)
                            tCtAcc_stage = tCtAcc[(None, None, None, acc_empty.index)]
                            cute.gemm(tiled_mma, tCtAcc_stage, [tCrQ[q_tile_crd], tCtQS], [tCrK[k_tile_crd], tCtKS], tCtAcc_stage)
                            acc_empty.commit()
                            k_pair_full.release()
                    else:
                        for ktile_inner in cutlass.range_constexpr(self.k_tiles_per_cta):
                            ktile = group_first_ktile + Int32(ktile_inner)
                            if ktile < max_k_tiles:
                                tile_has_visible = self._tile_has_visible(
                                    q_len,
                                    ktile,
                                    batch_k_tiles,
                                    causal_offset,
                                )
                                if tile_has_visible:
                                    k_pair_full = k_consumer.wait_and_advance()
                                    acc_empty = acc_producer.acquire_and_advance()
                                    cute.copy(tiled_copy_s2t_ks, tCsKS_compact_s2t[(None, None, None, None, k_pair_full.index)], tCtKS_compact_s2t)
                                    k_tile_crd = (None, None, None, k_pair_full.index)
                                    tCtAcc_stage = tCtAcc[(None, None, None, acc_empty.index)]
                                    cute.gemm(tiled_mma, tCtAcc_stage, [tCrQ[q_tile_crd], tCtQS], [tCrK[k_tile_crd], tCtKS], tCtAcc_stage)
                                    acc_empty.commit()
                                    k_pair_full.release()
                else:
                    k_group_full = group_first_ktile + Int32(self.k_tiles_per_cta) <= batch_k_tiles
                    if k_group_full:
                        for ktile_inner in cutlass.range_constexpr(self.k_tiles_per_cta):
                            k_pair_full = k_consumer.wait_and_advance()
                            acc_empty = acc_producer.acquire_and_advance()
                            cute.copy(tiled_copy_s2t_ks, tCsKS_compact_s2t[(None, None, None, None, k_pair_full.index)], tCtKS_compact_s2t)
                            k_tile_crd = (None, None, None, k_pair_full.index)
                            tCtAcc_stage = tCtAcc[(None, None, None, acc_empty.index)]
                            cute.gemm(tiled_mma, tCtAcc_stage, [tCrQ[q_tile_crd], tCtQS], [tCrK[k_tile_crd], tCtKS], tCtAcc_stage)
                            acc_empty.commit()
                            k_pair_full.release()
                    else:
                        for ktile_inner in cutlass.range_constexpr(self.k_tiles_per_cta):
                            ktile = group_first_ktile + Int32(ktile_inner)
                            if ktile < batch_k_tiles:
                                k_pair_full = k_consumer.wait_and_advance()
                                acc_empty = acc_producer.acquire_and_advance()
                                cute.copy(tiled_copy_s2t_ks, tCsKS_compact_s2t[(None, None, None, None, k_pair_full.index)], tCtKS_compact_s2t)
                                k_tile_crd = (None, None, None, k_pair_full.index)
                                tCtAcc_stage = tCtAcc[(None, None, None, acc_empty.index)]
                                cute.gemm(tiled_mma, tCtAcc_stage, [tCrQ[q_tile_crd], tCtQS], [tCrK[k_tile_crd], tCtKS], tCtAcc_stage)
                                acc_empty.commit()
                                k_pair_full.release()
                acc_producer.tail()

        if warp_idx < self.mma_warp_id:
            tmem_pool = tmem.reserve(self.num_tmem_alloc_cols)
            tCtAcc = tmem_pool.allocate_tensor(tCtAcc_fake.layout, Float32)
            if const_expr(self.use_tmem_load_red):
                copy_atom_t2r = cute.make_copy_atom(
                    tcgen05.LdRed32x32bOp(
                        tcgen05.Repetition.x128,
                        tcgen05.Pack.NONE,
                        tcgen05.TmemLoadRedOp.MAX,
                    ),
                    Float32,
                )
            else:
                copy_atom_t2r = cute.make_copy_atom(
                    tcgen05.Ld32x32bOp(tcgen05.Repetition.x128, tcgen05.Pack.NONE),
                    Float32,
                )
            tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc[(None, None, None, 0)])
            thr_copy_t2r = tiled_copy_t2r.get_slice(epi_tidx)
            tTR_tAcc = thr_copy_t2r.partition_S(tCtAcc)
            tTR_cC = thr_copy_t2r.partition_D(tCcC)
            tTR_rAcc = cute.make_rmem_tensor(tTR_cC.shape, Float32)
            if const_expr(self.use_tmem_load_red):
                tTR_rRed = cute.make_rmem_tensor((1,), Float32)
            h_store = epi_tidx // Int32(_DECODE_PACK_Q_LEN)
            q_local_store = epi_tidx - h_store * Int32(_DECODE_PACK_Q_LEN)
            h_global_store = hk * qhead_per_kv + h_store
            q_global_store = q_begin + q_local_store
            if group_has_visible:
                visible_tile_count = Int32(0)
                for ktile_inner in cutlass.range_constexpr(self.k_tiles_per_cta):
                    ktile = group_first_ktile + Int32(ktile_inner)
                    if ktile < max_k_tiles:
                        tile_has_visible = self._tile_has_visible(
                            q_len,
                            ktile,
                            batch_k_tiles,
                            causal_offset,
                        )
                        if tile_has_visible:
                            epilogue_owns_tile = epi_warpgroup_idx == Int32(
                                ktile_inner % self.num_epi_warpgroups
                            )
                            if epilogue_owns_tile:
                                acc_stage_index = visible_tile_count % Int32(self.num_acc_stage)
                                acc_stage_phase = (visible_tile_count // Int32(self.num_acc_stage)) % Int32(2)
                                tile_mask_free = self._tile_mask_free(ktile, causal_offset)
                                k_tile_full = ktile * Int32(_BLOCK_K) + Int32(_BLOCK_K - 1) < k_len
                                q_pack_full = q_len == Int32(_DECODE_PACK_Q_LEN)
                                tile_full = q_pack_full and k_tile_full
                                acc_pipeline.consumer_wait_w_index_phase(acc_stage_index, acc_stage_phase)
                                tTR_tAcc_stage = tTR_tAcc[(None, None, None, None, acc_stage_index)]
                                if const_expr(self.use_tmem_load_red):
                                    cute.copy(tiled_copy_t2r, tTR_tAcc_stage, [tTR_rAcc, tTR_rRed])
                                else:
                                    cute.copy(tiled_copy_t2r, tTR_tAcc_stage, tTR_rAcc)
                                row_max0 = -Float32.inf
                                if tile_mask_free and tile_full:
                                    if const_expr(self.use_tmem_load_red):
                                        row_max0 = tTR_rRed[0]
                                    else:
                                        for i in cutlass.range(cute.size(tTR_rAcc), unroll_full=True):
                                            coord_m, _ = tTR_cC[i]
                                            if coord_m == epi_tidx:
                                                row_max0 = cute.arch.fmax(row_max0, tTR_rAcc[i])
                                else:
                                    for i in cutlass.range(cute.size(tTR_rAcc), unroll_full=True):
                                        coord_m, coord_n = tTR_cC[i]
                                        h_in_group = coord_m // Int32(_DECODE_PACK_Q_LEN)
                                        q_local = coord_m - h_in_group * Int32(_DECODE_PACK_Q_LEN)
                                        k_local = ktile * Int32(_BLOCK_K) + coord_n
                                        valid = self._packed_coord_visible(
                                            coord_m,
                                            epi_tidx,
                                            h_in_group,
                                            qhead_per_kv,
                                            q_local,
                                            q_len,
                                            k_local,
                                            k_len,
                                            causal_offset,
                                        )
                                        if valid:
                                            row_max0 = cute.arch.fmax(row_max0, tTR_rAcc[i])
                                if h_store < qhead_per_kv and q_local_store < q_len:
                                    mScores[h_global_store, ktile, q_global_store] = row_max0
                                cute.arch.fence_view_async_tmem_load()
                                acc_pipeline.consumer_release_w_index(acc_stage_index)
                            visible_tile_count += Int32(1)
                        else:
                            if const_expr(not self.compact_schedule):
                                if epi_warpgroup_idx == Int32(0):
                                    if h_store < qhead_per_kv and q_local_store < q_len:
                                        mScores[h_global_store, ktile, q_global_store] = -Float32.inf
            else:
                if const_expr(not self.compact_schedule):
                    if epi_warpgroup_idx == Int32(0):
                        for ktile_inner in cutlass.range_constexpr(self.k_tiles_per_cta):
                            ktile = group_first_ktile + Int32(ktile_inner)
                            if ktile < max_k_tiles:
                                if h_store < qhead_per_kv and q_local_store < q_len:
                                    mScores[h_global_store, ktile, q_global_store] = -Float32.inf
            cute.arch.barrier()
            tmem.free(tmem_pool.base_ptr)
