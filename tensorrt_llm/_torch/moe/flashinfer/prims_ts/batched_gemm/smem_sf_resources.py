# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SMEM resources for scale factors A and B (TMA, gather4, LDGSTS)."""

from dataclasses import dataclass
from typing import Any, Optional

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass import Int32, Int64

from cutlass.experimental.task_scheduling.memory import SmemAllocation
from cutlass.experimental.task_scheduling.enums import (
    PipelineType,
    WorkAttr,
)
from cutlass.experimental.task_scheduling.resources import (
    MemoryResource,
    StageInfo,
    TaskLocalVariable,
    consumer_work,
    producer_work,
)

from .batched_gemm_config import (
    MAX_PRODUCER_COMMIT_PREFETCH_DEPTH,
    BatchedGemmConfig,
    SfLayout,
    SfSmemToTmemCopy,
    TMEM_SF_PACK_SIZE_BYTES,
)
from .gmem_ab_resources import nonnegative_div, nonnegative_mod
from cutlass.experimental import primitives as prims

Constexpr = cutlass.Constexpr


@dataclass(kw_only=True)
class SmemSfAResource(MemoryResource):
    """SMEM staging for scale-factor A (TMA producer)."""

    cfg: Constexpr[BatchedGemmConfig]
    tma_sfa_desc: Any = None
    smem_buf: Any = None
    _alloc_sfa: Constexpr[Optional[SmemAllocation]] = None
    desc_a_s2t_base: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    smem_sfa_stage_ptr: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()

    def __post_init__(self):
        if self._alloc_sfa is None:
            self._alloc_sfa = SmemAllocation(
                f"{self.name}_sfa",
                size_bytes=self.cfg.num_bytes_sfa_per_stage
                * self.cfg.num_stages_smem_sfa,
                alignment=1024,
            )
        self.desc_a_s2t_base = TaskLocalVariable(
            dtype=Int64, default=Int64(0), docs="S2T descriptor for SFA."
        )
        self.smem_sfa_stage_ptr = TaskLocalVariable(
            dtype=Int64, default=Int64(0), docs="SMEM stage pointer for SFA."
        )

    def get_smem_requirements(self):
        return [self._alloc_sfa]

    @cute.jit
    def _init_smem_sfa(self, context) -> None:
        self.smem_buf = cutlass.Array(
            context.smem_base.data_ptr() + self._alloc_sfa.offset,
            dtype=cutlass.Uint8,
            shape=(self._alloc_sfa.size_bytes,),
            addrspace=3,
        )

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_load_state(self, stage_info: StageInfo) -> None:
        self._init_smem_sfa(stage_info.context)

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_s2t_state(self, stage_info: StageInfo) -> None:
        self._init_smem_sfa(stage_info.context)

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def prepare_sfa_tile(self, stage_info: StageInfo) -> None:
        pass

    @producer_work
    @cute.jit
    def load_sfa_tile(
        self,
        stage_info: StageInfo,
        *,
        coord_sfa_k: Int32,
        coord_sfa_mn: cutlass.Int32,
    ) -> None:
        """TMA load SFA into SMEM using the descriptor's coordinate rank."""
        # Split R128c4 coords: (0, 0, sfk_block, outer_tile). The generated
        # descriptor uses a split (256, 2) leading tile to satisfy TMA.
        sf_vec_size = self.cfg.sf_vec_size
        sfk_block = coord_sfa_k * Int32(
            self.cfg.tile_k // sf_vec_size // TMEM_SF_PACK_SIZE_BYTES
        )
        stage_base = self.smem_buf.subview(
            self.cfg.num_bytes_sfa_per_stage * stage_info.stage_idx
        )
        if prims.elect_sync():
            if cutlass.const_expr(
                self.cfg.use_tile256_tmem_overlap
                or self.cfg.is_mx_mma
                or self.cfg.has_cast_a
                or self.cfg.is_nvfp4_mma
            ):
                # Generated R128c4 SF TMA uses [256, 2, K/(sfBlock*4), outer/128].
                self._tma_load(
                    stage_base,
                    self.tma_sfa_desc,
                    (Int32(0), Int32(0), sfk_block, coord_sfa_mn),
                    stage_info.barrier,
                )
            else:
                self._tma_load(
                    stage_base,
                    self.tma_sfa_desc,
                    (Int32(0), sfk_block, coord_sfa_mn),
                    stage_info.barrier,
                )

    @cute.jit
    def _tma_load(self, smem_dst, tma_desc, coords, barrier):
        """Single-CTA or per-CTA cluster TMA load for SFA."""
        if cutlass.const_expr(self.cfg.has_cast_a):
            prims.cp_async_bulk_tensor_shared_cluster_global(
                smem_dst,
                tma_desc,
                coords,
                barrier,
                [],
            )
        elif cutlass.const_expr(self.cfg.has_cluster):
            if cutlass.const_expr(self.cfg.is_mx_mma):
                cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
                mcast_mask = Int32(1) << cta_rank
                lead_cta_rank = nonnegative_div(cta_rank, self.cfg.cluster_m) * Int32(
                    self.cfg.cluster_m
                )
                barrier = prims.mapa(barrier, lead_cta_rank)
                barrier = prims.cvta_to(barrier, prims.CvtaSpace.SHARED)
                prims.cp_async_bulk_tensor_shared_cluster_global(
                    smem_dst,
                    tma_desc,
                    coords,
                    barrier,
                    [],
                    multicast_mask=mcast_mask,
                    group=self.cfg.cta_group,
                )
                return
            cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
            mcast_mask = Int32(1) << cta_rank
            lead_cta_rank = nonnegative_div(cta_rank, self.cfg.cluster_m) * Int32(
                self.cfg.cluster_m
            )
            barrier = prims.mapa(barrier, lead_cta_rank)
            barrier = prims.cvta_to(barrier, prims.CvtaSpace.SHARED)
            prims.cp_async_bulk_tensor_shared_cluster_global(
                smem_dst,
                tma_desc,
                coords,
                barrier,
                [],
                multicast_mask=mcast_mask,
                group=self.cfg.cta_group,
            )
        else:
            prims.cp_async_bulk_tensor_shared_cta_global(
                smem_dst,
                tma_desc,
                coords,
                barrier,
            )

    @consumer_work(returns=(desc_a_s2t_base, smem_sfa_stage_ptr))
    @cute.jit
    def build_sfa_s2t_desc(self, stage_info: StageInfo) -> tuple[Int64, Int64]:
        """Build SMEM descriptor for S2T copy of SFA."""
        stage_base = self.smem_buf.subview(
            self.cfg.num_bytes_sfa_per_stage * stage_info.stage_idx
        )
        desc_s2t = prims.Tcgen05SmemDesc.build(
            stage_base,
            leading_byte_offset=16,
            stride_byte_offset=128,
            layout=0,  # no swizzle
        )
        return desc_s2t, Int64(stage_base.data_ptr().toint())


@dataclass(kw_only=True)
class SmemSfBResource(MemoryResource):
    """SMEM staging for scale-factor B (TMA producer)."""

    cfg: Constexpr[BatchedGemmConfig]
    tma_sfb_desc: Any = None
    smem_buf: Any = None
    _alloc_sfb: Constexpr[Optional[SmemAllocation]] = None
    desc_b_s2t_base: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()

    def __post_init__(self):
        if self._alloc_sfb is None:
            self._alloc_sfb = SmemAllocation(
                f"{self.name}_sfb",
                size_bytes=self.cfg.num_bytes_sfb_per_stage
                * self.cfg.num_stages_smem_sfb,
                alignment=1024,
            )
        self.desc_b_s2t_base = TaskLocalVariable(
            dtype=Int64, default=Int64(0), docs="S2T descriptor for SFB."
        )

    def get_smem_requirements(self):
        return [self._alloc_sfb]

    @cute.jit
    def _init_smem_sfb(self, context) -> None:
        self.smem_buf = cutlass.Array(
            context.smem_base.data_ptr() + self._alloc_sfb.offset,
            dtype=cutlass.Uint8,
            shape=(self._alloc_sfb.size_bytes,),
            addrspace=3,
        )

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_load_state(self, stage_info: StageInfo) -> None:
        self._init_smem_sfb(stage_info.context)

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_s2t_state(self, stage_info: StageInfo) -> None:
        self._init_smem_sfb(stage_info.context)

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def prepare_sfb_tile(self, stage_info: StageInfo) -> None:
        pass

    @producer_work
    @cute.jit
    def load_sfb_tile(
        self,
        stage_info: StageInfo,
        *,
        coord_sfb_k: Int32,
        coord_sfb_mn: cutlass.Int32,
    ) -> None:
        """TMA load SFB into SMEM using the descriptor's coordinate rank."""
        # MX uses 32 elements per scale, NVFP4 uses 16.
        sf_vec_size = self.cfg.sf_vec_size
        sfk_block = coord_sfb_k * Int32(
            self.cfg.tile_k // sf_vec_size // TMEM_SF_PACK_SIZE_BYTES
        )
        stage_base = self.smem_buf.subview(
            self.cfg.num_bytes_sfb_per_stage * stage_info.stage_idx
        )
        if prims.elect_sync():
            if cutlass.const_expr(self.cfg.use_tile256_tmem_overlap):
                outer_tile = coord_sfb_mn * Int32((self.cfg.tile_n + 127) // 128)
                self._tma_load(
                    stage_base,
                    self.tma_sfb_desc,
                    (Int32(0), Int32(0), sfk_block, outer_tile),
                    stage_info.barrier,
                )
            elif cutlass.const_expr(self.cfg.uses_sfb_8x4_load):
                outer_tile = coord_sfb_mn * Int32((self.cfg.tile_n + 7) // 8)
                self._tma_load(
                    stage_base,
                    self.tma_sfb_desc,
                    (Int32(0), sfk_block, outer_tile),
                    stage_info.barrier,
                )
            elif cutlass.const_expr(self.cfg.is_mx_mma):
                outer_tile = coord_sfb_mn * Int32((self.cfg.tile_n + 127) // 128)
                self._tma_load(
                    stage_base,
                    self.tma_sfb_desc,
                    (Int32(0), Int32(0), sfk_block, outer_tile),
                    stage_info.barrier,
                )
            else:
                self._tma_load(
                    stage_base,
                    self.tma_sfb_desc,
                    (Int32(0), sfk_block, coord_sfb_mn),
                    stage_info.barrier,
                )

    @cute.jit
    def _tma_load(self, smem_dst, tma_desc, coords, barrier):
        """Single-CTA or per-CTA cluster TMA load for SFB."""
        if cutlass.const_expr(self.cfg.has_cluster):
            if cutlass.const_expr(
                self.cfg.sfb_smem_to_tmem_copy == int(SfSmemToTmemCopy.LDS_STTM)
                and self.cfg.smem_sfb_layout == int(SfLayout.R8c4)
            ):
                # Generated compact-SFB kernels use per-CTA cluster-space TMA
                # without CTA-group multicast, so each CTA's local mbarrier is
                # completed independently.
                prims.cp_async_bulk_tensor_shared_cluster_global(
                    smem_dst,
                    tma_desc,
                    coords,
                    barrier,
                    [],
                )
                return
            if cutlass.const_expr(self.cfg.is_mx_mma):
                cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
                mcast_mask = Int32(1) << cta_rank
                lead_cta_rank = nonnegative_div(cta_rank, self.cfg.cluster_m) * Int32(
                    self.cfg.cluster_m
                )
                barrier = prims.mapa(barrier, lead_cta_rank)
                barrier = prims.cvta_to(barrier, prims.CvtaSpace.SHARED)
                prims.cp_async_bulk_tensor_shared_cluster_global(
                    smem_dst,
                    tma_desc,
                    coords,
                    barrier,
                    [],
                    multicast_mask=mcast_mask,
                    group=self.cfg.cta_group,
                )
                return
            cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
            mcast_mask = Int32(1) << cta_rank
            lead_cta_rank = nonnegative_div(cta_rank, self.cfg.cluster_m) * Int32(
                self.cfg.cluster_m
            )
            barrier = prims.mapa(barrier, lead_cta_rank)
            barrier = prims.cvta_to(barrier, prims.CvtaSpace.SHARED)
            prims.cp_async_bulk_tensor_shared_cluster_global(
                smem_dst,
                tma_desc,
                coords,
                barrier,
                [],
                multicast_mask=mcast_mask,
                group=self.cfg.cta_group,
            )
        else:
            prims.cp_async_bulk_tensor_shared_cta_global(
                smem_dst,
                tma_desc,
                coords,
                barrier,
            )

    @consumer_work(returns=desc_b_s2t_base)
    @cute.jit
    def build_sfb_s2t_desc(self, stage_info: StageInfo) -> Int64:
        """Build SMEM descriptor for S2T copy of SFB."""
        stage_base = self.smem_buf.subview(
            self.cfg.num_bytes_sfb_per_stage * stage_info.stage_idx
        )
        desc_s2t = prims.Tcgen05SmemDesc.build(
            stage_base,
            leading_byte_offset=16,
            stride_byte_offset=128,
            layout=0,
        )
        return desc_s2t


# Module-level TMA gather4 helpers (shared with smem_ab_resources)
@cute.jit
def _tma_gather4_cta(smem_dst, tma_desc, k_coord, row0, row1, row2, row3, barrier):
    """Emit cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4."""
    smem_ptr = smem_dst.data_ptr()
    tma_ptr = tma_desc.data_ptr() if hasattr(tma_desc, "data_ptr") else tma_desc
    bar_ptr = barrier.data_ptr()
    # {$rN} = operand ref.  {{ = literal brace for LLVM asm.
    # We need PTX: [{smem}], [{tma}, {c0, c1, c2, c3, c4}], [{mbar}]
    # Template: [{$r0}], [{$r1}, {$r2, $r3, $r4, $r5, $r6}], [{$r7}]
    #   → $r2 wraps in {}: need {{$r2, {$r3}, {$r4}, {$r5}, {$r6}}}
    #   BUT inline_ptx_hl treats every { as opening an operand ref.
    # Escape nested braces explicitly in the inline PTX template.
    prims.inline_ptx_hl(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
        ".mbarrier::complete_tx::bytes"
        " [{$r0}], [{$r1}, {{$r2}, {$r3}, {$r4}, {$r5}, {$r6}}], [{$r7}];",
        read_only_args=[smem_ptr, tma_ptr, k_coord, row0, row1, row2, row3, bar_ptr],
    )


@dataclass(kw_only=True)
class SmemSfGatherResource(MemoryResource):
    """SMEM staging for routed-operand SF loaded via TMA gather4.

    The SF tensor has linear layout: 2D (K/sf_block_size, total_tokens) in E4M3.
    Each gather4 call loads 4 rows of SF data (one per routed token).
    The TMA descriptor is 2D with box=(tile_k/sf_block_size, 1).

    Replaces SmemSfA (non-swapAB) or SmemSfB (swapAB) for the routed operand
    when route_sfs_act == TMA.
    """

    cfg: Constexpr[BatchedGemmConfig]
    tma_sf_desc: Any = None  # 2D TMA descriptor for routed SF
    route_map: Any = None  # make_array_view of route map tensor
    mn_limit: Any = None  # make_array_view of TRT absolute token end-row limits
    smem_buf: Any = None
    routed_rows: Any = None
    _alloc_sf: Constexpr[Optional[SmemAllocation]] = None
    _operand: Constexpr[str] = "a"  # "a" or "b" — which SF operand this is
    desc_a_s2t_base: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    smem_sfa_stage_ptr: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    desc_b_s2t_base: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()

    def __post_init__(self):
        if self._alloc_sf is None:
            is_b = self._operand == "b"
            nbytes = (
                self.cfg.num_bytes_sfb_per_stage
                if is_b
                else self.cfg.num_bytes_sfa_per_stage
            )
            self._alloc_sf = SmemAllocation(
                f"{self.name}_sf_g4",
                size_bytes=nbytes
                * (
                    self.cfg.num_stages_smem_sfb
                    if is_b
                    else self.cfg.num_stages_smem_sfa
                ),
                alignment=1024,
            )
        self.desc_a_s2t_base = TaskLocalVariable(
            dtype=Int64, default=Int64(0), docs="S2T descriptor for gathered SFA."
        )
        self.smem_sfa_stage_ptr = TaskLocalVariable(
            dtype=Int64, default=Int64(0), docs="SMEM stage pointer for gathered SFA."
        )
        self.desc_b_s2t_base = TaskLocalVariable(
            dtype=Int64, default=Int64(0), docs="S2T descriptor for gathered SFB."
        )

    def get_smem_requirements(self):
        return [self._alloc_sf]

    @cute.jit
    def _setup_common(self, context=None) -> None:
        self.smem_buf = cutlass.Array(
            context.smem_base.data_ptr() + self._alloc_sf.offset,
            dtype=cutlass.Uint8,
            shape=(self._alloc_sf.size_bytes,),
            addrspace=3,
        )
        is_b = cutlass.const_expr(self._operand == "b")
        tile_rows = self.cfg.tile_n if is_b else self.cfg.tile_m
        num_gather4 = tile_rows // 4
        num_load_warps = (
            self.cfg.num_load_sfb_warps if is_b else self.cfg.num_load_sfa_warps
        )
        route_cache_groups = max(
            1, (num_gather4 + num_load_warps - 1) // num_load_warps
        )
        self.routed_rows = cutlass.Array(
            cutlass.Int32,
            route_cache_groups * 4,
            space=cutlass.AddressSpace.rmem,
        )

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_load_state(self, stage_info: StageInfo) -> None:
        self._setup_common(stage_info.context)

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_s2t_state(self, stage_info: StageInfo) -> None:
        self._setup_common(stage_info.context)

    @cute.jit
    def _local_tile_limit(self, raw_limit, token_tile, tile_rows):
        """Convert TRT-LLM Gen absolute end-row limit to a local row count."""
        local_limit = raw_limit - token_tile * Int32(tile_rows)
        if local_limit < Int32(0):
            local_limit = Int32(0)
        if local_limit > Int32(tile_rows):
            local_limit = Int32(tile_rows)
        return local_limit

    @cute.jit
    def _load_routed_row_or_zero(self, route_idx, row_in_tile, tile_limit):
        routed_row = Int32(0)
        if row_in_tile < tile_limit:
            routed_row = self.route_map.load(idx=route_idx, vector_size=1)[0]
        return cute.arch.make_warp_uniform(routed_row)

    @cute.jit
    def _prepare_gather_tile_impl(self, stage_info: StageInfo) -> None:
        """Prefetch per-warp SF gather4 route rows once per output tile."""
        tile_coord_m, tile_coord_n, _ = stage_info.work_tile.tile_idx

        is_b = cutlass.const_expr(self._operand == "b")
        tile_rows = self.cfg.tile_n if is_b else self.cfg.tile_m
        if cutlass.const_expr(is_b):
            if cutlass.const_expr(self.cfg.has_routed_sfs and self.cfg.is_swap_ab):
                coord_mn = tile_coord_n * Int32(self.cfg.tile_n)
            else:
                coord_mn = tile_coord_n
            num_load_warps = self.cfg.num_load_sfb_warps
            load_warp_idx = self.cfg.load_sfb_warp_idx
        else:
            coord_mn = tile_coord_m * Int32(self.cfg.tile_m)
            num_load_warps = self.cfg.num_load_sfa_warps
            load_warp_idx = self.cfg.load_sfa_warp_idx

        num_gather4 = tile_rows // 4
        route_cache_groups = max(
            1, (num_gather4 + num_load_warps - 1) // num_load_warps
        )
        token_tile = coord_mn // Int32(tile_rows)
        tile_limit = Int32(tile_rows)
        if cutlass.const_expr(self.mn_limit is not None):
            tile_limit = self._local_tile_limit(
                self.mn_limit.load(idx=token_tile, vector_size=1)[0],
                token_tile,
                tile_rows,
            )

        warp_idx = cute.arch.warp_idx()
        warp_in_task = warp_idx - Int32(load_warp_idx)
        warp_in_task = cute.arch.make_warp_uniform(warp_in_task)

        for wi in cutlass.range_constexpr(route_cache_groups):
            gi = warp_in_task + Int32(wi * num_load_warps)
            is_valid_gather = gi * Int32(4) < Int32(tile_rows)
            if is_valid_gather:
                base_row = coord_mn + gi * Int32(4)
                row_base = gi * Int32(4)
                self.routed_rows[wi * 4] = self._load_routed_row_or_zero(
                    base_row, row_base, tile_limit
                )
                self.routed_rows[wi * 4 + 1] = self._load_routed_row_or_zero(
                    base_row + Int32(1), row_base + Int32(1), tile_limit
                )
                self.routed_rows[wi * 4 + 2] = self._load_routed_row_or_zero(
                    base_row + Int32(2), row_base + Int32(2), tile_limit
                )
                self.routed_rows[wi * 4 + 3] = self._load_routed_row_or_zero(
                    base_row + Int32(3), row_base + Int32(3), tile_limit
                )

    @cute.jit
    def _producer_work_impl(self, stage_info: StageInfo, coord_k, coord_mn) -> None:
        """TMA gather4 for SF: load routed SF rows."""
        is_b = cutlass.const_expr(self._operand == "b")
        tile_rows = self.cfg.tile_n if is_b else self.cfg.tile_m
        bytes_per_stage = (
            self.cfg.num_bytes_sfb_per_stage
            if is_b
            else self.cfg.num_bytes_sfa_per_stage
        )

        stage_base = self.smem_buf.subview(bytes_per_stage * stage_info.stage_idx)

        # SF elements per row = tile_k / sf_block_size.
        sf_block_size = (
            self.cfg.input_sf_block_size_b if is_b else self.cfg.input_sf_block_size_a
        )
        sf_k = self.cfg.tile_k // sf_block_size
        # Each gather4 loads 4 rows × sf_k SF elements (E4M3 = 1 byte each)
        num_gather4 = tile_rows // 4
        num_load_warps = (
            self.cfg.num_load_sfb_warps if is_b else self.cfg.num_load_sfa_warps
        )
        load_warp_idx = (
            self.cfg.load_sfb_warp_idx if is_b else self.cfg.load_sfa_warp_idx
        )
        warp_batches = (num_gather4 + num_load_warps - 1) // num_load_warps

        warp_idx = cute.arch.warp_idx()
        warp_in_task = warp_idx - Int32(load_warp_idx)
        warp_in_task = cute.arch.make_warp_uniform(warp_in_task)

        if prims.elect_sync():
            # coord_k is a K-tile index from GmemSfA/BResource. TMA gather4
            # wants the linear SF element offset for this tile.
            k_sf = coord_k * Int32(self.cfg.tile_k // sf_block_size)
            for wi in cutlass.range_constexpr(warp_batches):
                gi = warp_in_task + Int32(wi * num_load_warps)
                is_valid_gather = gi * Int32(4) < Int32(tile_rows)
                if is_valid_gather:
                    route_cache_base = wi * 4
                    r0 = self.routed_rows[route_cache_base]
                    r1 = self.routed_rows[route_cache_base + 1]
                    r2 = self.routed_rows[route_cache_base + 2]
                    r3 = self.routed_rows[route_cache_base + 3]

                    # SMEM offset: gi * 4 rows × sf_k bytes (E4M3 = 1 byte)
                    smem_offset = gi * Int32(4 * sf_k)
                    smem_dst = stage_base.subview(smem_offset)

                    _tma_gather4_cta(
                        smem_dst,
                        self.tma_sf_desc,
                        k_sf,
                        r0,
                        r1,
                        r2,
                        r3,
                        stage_info.barrier,
                    )

    @cute.jit
    def _build_desc_impl(self, stage_info: StageInfo):
        """Build SMEM descriptor for S2T copy of gathered SF."""
        is_b = cutlass.const_expr(self._operand == "b")
        bytes_per_stage = (
            self.cfg.num_bytes_sfb_per_stage
            if is_b
            else self.cfg.num_bytes_sfa_per_stage
        )
        stage_base = self.smem_buf.subview(bytes_per_stage * stage_info.stage_idx)
        desc_s2t = prims.Tcgen05SmemDesc.build(
            stage_base,
            leading_byte_offset=16,
            stride_byte_offset=128,
            layout=0,  # no swizzle for linear SF layout
        )
        return desc_s2t


@dataclass(kw_only=True)
class SmemSfGatherAResource(SmemSfGatherResource):
    """SmemSfGather for operand A with static var names."""

    _operand: Constexpr[str] = "a"

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def prepare_sfa_tile(self, stage_info: StageInfo) -> None:
        self._prepare_gather_tile_impl(stage_info)

    @producer_work
    @cute.jit
    def load_sfa_tile(
        self,
        stage_info: StageInfo,
        *,
        coord_sfa_k: Int32,
        coord_sfa_mn: cutlass.Int32,
    ) -> None:
        self._producer_work_impl(stage_info, coord_sfa_k, coord_sfa_mn)

    @consumer_work(
        returns=(
            SmemSfGatherResource.desc_a_s2t_base,
            SmemSfGatherResource.smem_sfa_stage_ptr,
        )
    )
    @cute.jit
    def build_sfa_s2t_desc(self, stage_info: StageInfo) -> tuple[Int64, Int64]:
        desc = self._build_desc_impl(stage_info)
        is_b = cutlass.const_expr(self._operand == "b")
        nbytes = (
            self.cfg.num_bytes_sfb_per_stage
            if is_b
            else self.cfg.num_bytes_sfa_per_stage
        )
        stage_base = self.smem_buf.subview(nbytes * stage_info.stage_idx)
        return desc, Int64(stage_base.data_ptr().toint())


@dataclass(kw_only=True)
class SmemSfGatherBResource(SmemSfGatherResource):
    """SmemSfGather for operand B with static var names."""

    _operand: Constexpr[str] = "b"

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def prepare_sfb_tile(self, stage_info: StageInfo) -> None:
        self._prepare_gather_tile_impl(stage_info)

    @producer_work
    @cute.jit
    def load_sfb_tile(
        self,
        stage_info: StageInfo,
        *,
        coord_sfb_k: Int32,
        coord_sfb_mn: cutlass.Int32,
    ) -> None:
        self._producer_work_impl(stage_info, coord_sfb_k, coord_sfb_mn)

    @consumer_work(returns=SmemSfGatherResource.desc_b_s2t_base)
    @cute.jit
    def build_sfb_s2t_desc(self, stage_info: StageInfo) -> Int64:
        return self._build_desc_impl(stage_info)


# ---------------------------------------------------------------------------
@dataclass(kw_only=True)
class SmemSfLdgstsResource(MemoryResource):
    """SMEM staging for routed-operand SF loaded via LDGSTS (cp.async).

    Each thread loads 4 E4M3 elements from GMEM at a routed address and stores
    them to SMEM using a block-shuffled layout compatible with the STTM copy.

    SF SMEM layout pattern:
      - threadBaseOffset = warpGrpThreadIdx * 4
      - GMEM: routedRowIdx * (K/sf_vec_size) + tile SF offset
        + (threadBaseOffset % sf_k)
      - SMEM: dataBlkIdx * 32 + idxInDataBlk (block-shuffled)
      - cp.async: 4 bytes per thread
    """

    cfg: Constexpr[BatchedGemmConfig]
    sf_gmem_ptr: Any = None  # make_array_view of SF GMEM tensor (E4M3)
    sf_gmem_stride: Any = None  # GMEM row stride in SF elements
    route_map: Any = None  # make_array_view of route map tensor
    mn_limit: Any = None  # make_array_view of TRT absolute token end-row limits
    smem_buf: Any = None
    routed_rows: Any = None
    tile_limit: Any = None
    _alloc_sf: Constexpr[Optional[SmemAllocation]] = None
    _operand: Constexpr[str] = "a"
    producer_commit_prefetch_depth: Constexpr[int] = 0
    desc_a_s2t_base: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    smem_sfa_stage_ptr: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    desc_b_s2t_base: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()

    @property
    def uses_async_load_pipeline(self) -> bool:
        if self.pipeline_config is None:
            return False
        if self.pipeline_config.pipeline_type == PipelineType.AsyncAsync:
            return (
                self.pipeline_config.async_producer_op == pipeline.PipelineOp.AsyncLoad
            )
        if self.pipeline_config.pipeline_type == PipelineType.AsyncUmma:
            return (
                self.pipeline_config.umma_consumer_producer_op
                == pipeline.PipelineOp.AsyncLoad
            )
        return False

    @property
    def cp_async_wait_group_depth(self) -> int:
        """LDGSTS cp.async wait-group depth."""
        return max(0, self.producer_commit_prefetch_depth - 1)

    def _tail_cp_async_wait_group_depth(
        self, stage_info: StageInfo, prefetch_idx: int
    ) -> int:
        """Return the tail drain depth for the groups actually prefetched.

        ``prefetch_idx`` is the per-section ordinal of the enclosing ``drain_tail``
        work call, threaded in by the work method from its constexpr work arg.
        """
        if isinstance(stage_info.loop_end, int):
            active_prefetch = min(
                stage_info.loop_end,
                self.producer_commit_prefetch_depth,
            )
            return max(0, active_prefetch - 1 - prefetch_idx)
        return max(0, self.cp_async_wait_group_depth - prefetch_idx)

    @cute.jit
    def _tail_cp_async_wait_group(
        self, stage_info: StageInfo, prefetch_idx: int
    ) -> None:
        """Drain only the cp.async groups that exist for this runtime K depth.

        ``prefetch_idx`` is the per-section ordinal of the enclosing ``drain_tail``
        work call, threaded in by the work method from its constexpr work arg.
        """
        if isinstance(stage_info.loop_end, int):
            depth = self._tail_cp_async_wait_group_depth(stage_info, prefetch_idx)
            if cutlass.const_expr(depth == 0):
                cute.arch.cp_async_wait_group(0)
            elif cutlass.const_expr(depth == 1):
                cute.arch.cp_async_wait_group(1)
            else:
                cute.arch.cp_async_wait_group(2)
        else:
            # cp.async.wait_group takes an immediate.  For runtime K depths
            # shorter than the prefetch window, branch to the matching static
            # wait depth instead of using the maximum-depth tail sequence.
            # `__post_init__` rejects producer_commit_prefetch_depth >
            # MAX_PRODUCER_COMMIT_PREFETCH_DEPTH (== 3), so
            # cp_async_wait_group_depth is always in {0, 1, 2} below.
            if cutlass.const_expr(self.cp_async_wait_group_depth == 0):
                cute.arch.cp_async_wait_group(0)
            elif cutlass.const_expr(self.cp_async_wait_group_depth == 1):
                if cutlass.const_expr(prefetch_idx > 0):
                    cute.arch.cp_async_wait_group(0)
                else:
                    if stage_info.loop_end <= cutlass.Int32(1):
                        cute.arch.cp_async_wait_group(0)
                    else:
                        cute.arch.cp_async_wait_group(1)
            else:
                # cp_async_wait_group_depth == 2
                if cutlass.const_expr(prefetch_idx == 0):
                    if stage_info.loop_end <= cutlass.Int32(1):
                        cute.arch.cp_async_wait_group(0)
                    elif stage_info.loop_end <= cutlass.Int32(2):
                        cute.arch.cp_async_wait_group(1)
                    else:
                        cute.arch.cp_async_wait_group(2)
                elif cutlass.const_expr(prefetch_idx == 1):
                    if stage_info.loop_end <= cutlass.Int32(2):
                        cute.arch.cp_async_wait_group(0)
                    else:
                        cute.arch.cp_async_wait_group(1)
                else:
                    cute.arch.cp_async_wait_group(0)

    def __post_init__(self):
        if self.producer_commit_prefetch_depth < 0:
            raise ValueError(
                "producer_commit_prefetch_depth must be non-negative, "
                f"got {self.producer_commit_prefetch_depth}."
            )
        # The dynamic-`loop_end` path in `_tail_cp_async_wait_group` only
        # hand-rolls cp_async_wait_group_depth in {0, 1, 2}; keep this in
        # lockstep with MAX_PRODUCER_COMMIT_PREFETCH_DEPTH so the chain
        # stays exhaustive.
        if self.producer_commit_prefetch_depth > MAX_PRODUCER_COMMIT_PREFETCH_DEPTH:
            raise ValueError(
                "producer_commit_prefetch_depth must be at most "
                f"{MAX_PRODUCER_COMMIT_PREFETCH_DEPTH}, got "
                f"{self.producer_commit_prefetch_depth}."
            )
        if self.producer_commit_prefetch_depth > 0:
            if (
                self.pipeline_config is None
                or not self.pipeline_config.advance_on_acquire
            ):
                raise ValueError(
                    "producer_commit_prefetch_depth requires advance_on_acquire=True."
                )
            if self.producer_commit_prefetch_depth >= self.pipeline_config.num_stages:
                raise ValueError(
                    "producer_commit_prefetch_depth must be smaller than "
                    f"num_stages={self.pipeline_config.num_stages}, got "
                    f"{self.producer_commit_prefetch_depth}."
                )
        if self._alloc_sf is None:
            is_b = self._operand == "b"
            nbytes = (
                self.cfg.num_bytes_sfb_per_stage
                if is_b
                else self.cfg.num_bytes_sfa_per_stage
            )
            self._alloc_sf = SmemAllocation(
                f"{self.name}_sf_ldg",
                size_bytes=nbytes
                * (
                    self.cfg.num_stages_smem_sfb
                    if is_b
                    else self.cfg.num_stages_smem_sfa
                ),
                alignment=1024,
            )
        self.desc_a_s2t_base = TaskLocalVariable(
            dtype=Int64, default=Int64(0), docs="S2T descriptor for LDGSTS SFA."
        )
        self.smem_sfa_stage_ptr = TaskLocalVariable(
            dtype=Int64, default=Int64(0), docs="SMEM stage pointer for LDGSTS SFA."
        )
        self.desc_b_s2t_base = TaskLocalVariable(
            dtype=Int64, default=Int64(0), docs="S2T descriptor for LDGSTS SFB."
        )

    def get_smem_requirements(self):
        return [self._alloc_sf]

    @cute.jit
    def _setup_common(self, context=None) -> None:
        self.smem_buf = cutlass.Array(
            context.smem_base.data_ptr() + self._alloc_sf.offset,
            dtype=cutlass.Uint8,
            shape=(self._alloc_sf.size_bytes,),
            addrspace=3,
        )
        is_b = cutlass.const_expr(self._operand == "b")
        tile_rows = self.cfg.tile_n if is_b else self.cfg.tile_m
        sf_block_size = (
            self.cfg.input_sf_block_size_b if is_b else self.cfg.input_sf_block_size_a
        )
        sf_k = self.cfg.tile_k // sf_block_size
        num_load_warps = (
            self.cfg.num_load_sfb_warps if is_b else self.cfg.num_load_sfa_warps
        )
        num_threads = max(1, num_load_warps) * 32
        elts_per_load = 4
        total_elts = tile_rows * sf_k
        loads_per_thread = (total_elts + num_threads * elts_per_load - 1) // (
            num_threads * elts_per_load
        )
        self.routed_rows = cutlass.Array(
            cutlass.Int32,
            max(1, loads_per_thread),
            space=cutlass.AddressSpace.rmem,
        )
        self.tile_limit = Int32(0)

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_load_state(self, stage_info: StageInfo) -> None:
        self._setup_common(stage_info.context)

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_s2t_state(self, stage_info: StageInfo) -> None:
        self._setup_common(stage_info.context)

    @cute.jit
    def _local_tile_limit(self, raw_limit, token_tile, tile_rows):
        """Convert TRT-LLM Gen absolute end-row limit to a local row count."""
        local_limit = raw_limit - token_tile * Int32(tile_rows)
        if local_limit < Int32(0):
            local_limit = Int32(0)
        if local_limit > Int32(tile_rows):
            local_limit = Int32(tile_rows)
        return local_limit

    @cute.jit
    def _prepare_ldgsts_tile_impl(self, stage_info: StageInfo) -> None:
        """Prefetch routed SF rows once per output tile."""
        tile_coord_m, tile_coord_n, _ = stage_info.work_tile.tile_idx
        is_b = cutlass.const_expr(self._operand == "b")
        tile_rows = self.cfg.tile_n if is_b else self.cfg.tile_m
        sf_block_size = (
            self.cfg.input_sf_block_size_b if is_b else self.cfg.input_sf_block_size_a
        )
        sf_k = self.cfg.tile_k // sf_block_size
        elts_per_load = 4
        total_elts = tile_rows * sf_k

        if cutlass.const_expr(is_b):
            if cutlass.const_expr(self.cfg.has_routed_sfs and self.cfg.is_swap_ab):
                coord_mn = tile_coord_n * Int32(self.cfg.tile_n)
            else:
                coord_mn = tile_coord_n
            load_warp_idx = self.cfg.load_sfb_warp_idx
            num_load_warps = self.cfg.num_load_sfb_warps
        else:
            coord_mn = tile_coord_m
            load_warp_idx = self.cfg.load_sfa_warp_idx
            num_load_warps = self.cfg.num_load_sfa_warps

        num_threads = max(1, num_load_warps) * 32
        loads_per_thread = (total_elts + num_threads * elts_per_load - 1) // (
            num_threads * elts_per_load
        )
        tidx, _, _ = cute.arch.thread_idx()
        local_tid = tidx - Int32(load_warp_idx * 32)
        if cutlass.const_expr(is_b and self.cfg.has_routed_sfs and self.cfg.is_swap_ab):
            tile_limit_idx = nonnegative_div(coord_mn, tile_rows)
        else:
            tile_limit_idx = coord_mn
        self.tile_limit = self._local_tile_limit(
            self.mn_limit.load(idx=tile_limit_idx, vector_size=1)[0],
            tile_limit_idx,
            tile_rows,
        )

        for li in cutlass.range_constexpr(loads_per_thread):
            thread_base = local_tid * Int32(elts_per_load) + Int32(
                li * num_threads * elts_per_load
            )
            row_in_tile = nonnegative_div(thread_base, sf_k)
            if cutlass.const_expr(
                is_b and self.cfg.has_routed_sfs and self.cfg.is_swap_ab
            ):
                route_idx = coord_mn + row_in_tile
            else:
                route_idx = coord_mn * Int32(tile_rows) + row_in_tile

            is_valid = (
                (local_tid >= Int32(0))
                & (local_tid < Int32(num_threads))
                & (thread_base < Int32(total_elts))
                & (row_in_tile < self.tile_limit)
            )
            if is_valid:
                self.routed_rows[li] = self.route_map.load(
                    idx=route_idx, vector_size=1
                )[0]

    @cute.jit
    def _producer_work_impl(self, stage_info: StageInfo, coord_k, coord_mn) -> None:
        """LDGSTS: load SF from GMEM at routed addresses into block-shuffled SMEM.

        Each thread issues as many 4-byte cp.async operations as needed to
        cover the routed SF tile. The SMEM layout uses the R128c4 shuffle so
        the following tcgen05_cp path can consume it.
        """
        is_b = cutlass.const_expr(self._operand == "b")
        bytes_per_stage = (
            self.cfg.num_bytes_sfb_per_stage
            if is_b
            else self.cfg.num_bytes_sfa_per_stage
        )
        tile_rows = self.cfg.tile_n if is_b else self.cfg.tile_m
        sf_block_size = (
            self.cfg.input_sf_block_size_b if is_b else self.cfg.input_sf_block_size_a
        )
        sf_k = self.cfg.tile_k // sf_block_size
        # Low-N routed SFB uses compact R8c4 staging. Tile N >= 128 remains
        # physically R128c4; the N=192 LDS+STTM path compacts its padded
        # 8-column representation to the six columns consumed by MMA.
        use_r128c4 = cutlass.const_expr((not is_b) or self.cfg.tile_n >= 128)

        stage_idx = stage_info.stage_idx
        stage_base = self.smem_buf.subview(bytes_per_stage * stage_idx)

        tidx, _, _ = cute.arch.thread_idx()
        load_warp_idx = (
            self.cfg.load_sfb_warp_idx if is_b else self.cfg.load_sfa_warp_idx
        )
        num_load_warps = (
            self.cfg.num_load_sfb_warps if is_b else self.cfg.num_load_sfa_warps
        )
        local_tid = tidx - Int32(load_warp_idx * 32)

        k_sf_offset = coord_k * Int32(sf_k)
        num_col_blocks = sf_k // TMEM_SF_PACK_SIZE_BYTES
        total_elts = tile_rows * sf_k
        elts_per_load = 4
        num_threads = max(1, num_load_warps) * 32
        loads_per_thread = (total_elts + num_threads * elts_per_load - 1) // (
            num_threads * elts_per_load
        )
        tile_limit = self.tile_limit

        for li in cutlass.range_constexpr(loads_per_thread):
            thread_base = local_tid * Int32(elts_per_load) + Int32(
                li * num_threads * elts_per_load
            )
            row_in_tile = nonnegative_div(thread_base, sf_k)
            col_in_tile = nonnegative_mod(thread_base, sf_k)

            is_valid = (
                (local_tid >= Int32(0))
                & (local_tid < Int32(num_threads))
                & (thread_base < Int32(total_elts))
                & (row_in_tile < tile_limit)
            )
            routed_row = Int32(0)
            if is_valid:
                routed_row = self.routed_rows[li]
            gmem_offset = routed_row * self.sf_gmem_stride + k_sf_offset + col_in_tile
            gmem_src = self.sf_gmem_ptr.data_ptr() + gmem_offset

            if cutlass.const_expr(use_r128c4):
                data_blk_row = nonnegative_div(row_in_tile, 128)
                data_blk_col = nonnegative_div(col_in_tile, 4)
                row_in_blk = nonnegative_mod(row_in_tile, 128)
                col_in_blk = nonnegative_mod(col_in_tile, 4)
                row_in_blk0 = nonnegative_mod(row_in_blk, 32)
                row_in_blk1 = nonnegative_div(row_in_blk, 32)
                row_in_blk_shuffled = row_in_blk0 * Int32(4) + row_in_blk1
                data_blk_idx = data_blk_row * Int32(num_col_blocks) + data_blk_col
                idx_in_blk = row_in_blk_shuffled * Int32(4) + col_in_blk
                smem_offset = data_blk_idx * Int32(512) + idx_in_blk
            else:
                data_blk_row = nonnegative_div(row_in_tile, 8)
                data_blk_col = nonnegative_div(col_in_tile, 4)
                row_in_blk = nonnegative_mod(row_in_tile, 8)
                col_in_blk = nonnegative_mod(col_in_tile, 4)
                data_blk_idx = data_blk_row * Int32(num_col_blocks) + data_blk_col
                idx_in_blk = row_in_blk * Int32(4) + col_in_blk
                smem_offset = data_blk_idx * Int32(32) + idx_in_blk
            smem_dst = stage_base.subview(smem_offset)

            # BS=1 leaves most routed rows padded. Predicate invalid rows so
            # they do not issue zero-byte cp.async instructions.
            prims.inline_ptx_hl(
                "cp.async.ca.shared.global.L2::128B [{$r0}], [{$r1}], 4;",
                read_only_args=[smem_dst.data_ptr(), gmem_src],
                pred=is_valid,
            )

        # Non-cluster route-SF LDGSTS uses an AsyncLoad full barrier, so the
        # following producer_commit lowers to cp.async.mbarrier.arrive.noinc.
        # Clustered AsyncThread commits the cp.async group here, on the issue
        # side.  The named producer work "drain_loop"/"drain_tail" throttles/
        # drains before publishing the lagging stage.
        if cutlass.const_expr(self.cfg.has_cluster):
            if cutlass.const_expr(self.uses_async_load_pipeline):
                pass
            else:
                cute.arch.cp_async_commit_group()

    @producer_work
    @cute.jit
    def drain_loop(self, stage_info: StageInfo) -> None:
        """Drain producer cp.async work before producer_commit (loop phase)."""
        del stage_info
        if cutlass.const_expr(self.cfg.has_cluster):
            if cutlass.const_expr(self.uses_async_load_pipeline):
                pass
            else:
                if cutlass.const_expr(self.cp_async_wait_group_depth == 0):
                    cute.arch.cp_async_wait_group(0)
                elif cutlass.const_expr(self.cp_async_wait_group_depth == 1):
                    cute.arch.cp_async_wait_group(1)
                else:
                    cute.arch.cp_async_wait_group(2)
        else:
            cute.arch.fence_view_async_shared()

    @producer_work
    @cute.jit
    def drain_tail(
        self, stage_info: StageInfo, *, prefetch_idx: cutlass.Constexpr[int]
    ) -> None:
        """Drain producer cp.async work before producer_commit (tail phase)."""
        if cutlass.const_expr(self.cfg.has_cluster):
            if cutlass.const_expr(self.uses_async_load_pipeline):
                pass
            else:
                self._tail_cp_async_wait_group(stage_info, prefetch_idx)
        else:
            cute.arch.fence_view_async_shared()

    @cute.jit
    def _build_desc_impl(self, stage_info: StageInfo):
        """Build SMEM descriptor for S2T copy of gathered SF (block-shuffled layout)."""
        is_b = cutlass.const_expr(self._operand == "b")
        bytes_per_stage = (
            self.cfg.num_bytes_sfb_per_stage
            if is_b
            else self.cfg.num_bytes_sfa_per_stage
        )
        stage_base = self.smem_buf.subview(bytes_per_stage * stage_info.stage_idx)
        use_r128c4 = cutlass.const_expr((not is_b) or self.cfg.tile_n >= 128)
        if cutlass.const_expr(use_r128c4):
            desc_s2t = prims.Tcgen05SmemDesc.build(
                stage_base,
                leading_byte_offset=16,
                stride_byte_offset=128,
                layout=0,
            )
        else:
            desc_s2t = prims.Tcgen05SmemDesc.build(
                stage_base,
                leading_byte_offset=16,
                stride_byte_offset=128,
                layout=0,
            )
        return desc_s2t


@dataclass(kw_only=True)
class SmemSfLdgstsAResource(SmemSfLdgstsResource):
    """SmemSfLdgsts for operand A with static var names."""

    _operand: Constexpr[str] = "a"

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def prepare_sfa_tile(self, stage_info: StageInfo) -> None:
        self._prepare_ldgsts_tile_impl(stage_info)

    @producer_work
    @cute.jit
    def load_sfa_tile(
        self,
        stage_info: StageInfo,
        *,
        coord_sfa_k: Int32,
        coord_sfa_mn: cutlass.Int32,
    ) -> None:
        self._producer_work_impl(stage_info, coord_sfa_k, coord_sfa_mn)

    @consumer_work(
        returns=(
            SmemSfLdgstsResource.desc_a_s2t_base,
            SmemSfLdgstsResource.smem_sfa_stage_ptr,
        )
    )
    @cute.jit
    def build_sfa_s2t_desc(self, stage_info: StageInfo) -> tuple[Int64, Int64]:
        desc = self._build_desc_impl(stage_info)
        is_b = cutlass.const_expr(self._operand == "b")
        nbytes = (
            self.cfg.num_bytes_sfb_per_stage
            if is_b
            else self.cfg.num_bytes_sfa_per_stage
        )
        stage_base = self.smem_buf.subview(nbytes * stage_info.stage_idx)
        return desc, Int64(stage_base.data_ptr().toint())


@dataclass(kw_only=True)
class SmemSfLdgstsBResource(SmemSfLdgstsResource):
    """SmemSfLdgsts for operand B with static var names."""

    _operand: Constexpr[str] = "b"

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def prepare_sfb_tile(self, stage_info: StageInfo) -> None:
        self._prepare_ldgsts_tile_impl(stage_info)

    @producer_work
    @cute.jit
    def load_sfb_tile(
        self,
        stage_info: StageInfo,
        *,
        coord_sfb_k: Int32,
        coord_sfb_mn: cutlass.Int32,
    ) -> None:
        self._producer_work_impl(stage_info, coord_sfb_k, coord_sfb_mn)

    @consumer_work(returns=SmemSfLdgstsResource.desc_b_s2t_base)
    @cute.jit
    def build_sfb_s2t_desc(self, stage_info: StageInfo) -> Int64:
        return self._build_desc_impl(stage_info)
