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

"""TMEM resources for scale factors and CastA."""

from dataclasses import dataclass
from typing import Any, Optional

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64

from cutlass.experimental.task_scheduling.memory import TmemAllocation
from cutlass.experimental.task_scheduling.resources import (
    WorkAttr,
    MemoryResource,
    StageInfo,
    TaskLocalVariable,
    consumer_work,
    producer_work,
)

from .batched_gemm_config import (
    BatchedGemmConfig,
    SfLayout,
    SfSmemToTmemCopy,
    TMEM_SF_UTCCP_COLS_PER_COPY,
    TMEM_SF_PACK_SIZE_BYTES,
)
from cutlass.experimental import primitives as prims

Constexpr = cutlass.Constexpr


@dataclass(kw_only=True)
class TmemSfAResource(MemoryResource):
    """TMEM for scale-factor A (LDS+STTM from CopySfA task)."""

    cfg: Constexpr[BatchedGemmConfig]
    sfa_tmem_addr_base: Any = None
    cta_rank: Any = None
    _alloc_sfa: Constexpr[Optional[TmemAllocation]] = None
    sfa_stage_col_offset: Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )

    def __post_init__(self):
        if self._alloc_sfa is None:
            stage_mult = (
                self.cfg.num_stages_tmem_sfa
                if self.cfg.uses_unfused_tmem_sf_copy
                else 1
            )
            self._alloc_sfa = TmemAllocation(
                f"{self.name}_tmem_sfa",
                num_columns=self.cfg.tmem_sfa_cols * stage_mult,
            )
        self.sfa_stage_col_offset = TaskLocalVariable(
            dtype=Int32, default=Int32(0), docs="TMEM SFA stage column offset."
        )

    def get_tmem_requirements(self):
        return [self._alloc_sfa]

    @cute.jit
    def _init_tmem_state(self, stage_info: StageInfo) -> None:
        context = stage_info.context
        tmem_raw = context.tmem_ptr_i32.load()
        base_col = tmem_raw & 0xFFFF
        base_row = tmem_raw >> 16
        sfa_col = base_col + self._alloc_sfa.offset
        self.sfa_tmem_addr_base = (base_row << 16) | sfa_col
        self.cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_copy_state(self, stage_info: StageInfo) -> None:
        self._init_tmem_state(stage_info)

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_mma_state(self, stage_info: StageInfo) -> None:
        self._init_tmem_state(stage_info)

    @producer_work
    @cute.jit
    def copy_sfa(
        self,
        stage_info: StageInfo,
        *,
        desc_a_s2t_base: prims.Tcgen05SmemDesc,
        smem_sfa_stage_ptr: Int64,
    ) -> None:
        """S2T copy SFA from SMEM to TMEM via tcgen05_cp."""
        desc_s2t_base = desc_a_s2t_base
        sf_stage_col_offset = stage_info.stage_idx * self.cfg.tmem_sfa_cols
        s2t_shape, s2t_multicast = prims.S2TCopyMode.S2T_32x128b_WARPX4
        num_s2t_iters = self.cfg.tmem_sfa_cols // TMEM_SF_UTCCP_COLS_PER_COPY

        do_copy = cutlass.Boolean(True)
        if cutlass.const_expr(self.cfg.has_routed_sfs and self.cfg.has_cluster):
            do_copy = self.cta_rank == Int32(0)
        if do_copy:
            for s2t_idx in cutlass.range_constexpr(num_s2t_iters):
                tmem_addr = (
                    self.sfa_tmem_addr_base
                    + sf_stage_col_offset
                    + s2t_idx * TMEM_SF_UTCCP_COLS_PER_COPY
                )
                tmem_ptr = prims.make_tmem_ptr(tmem_addr, cutlass.Int32)
                desc_s2t = desc_s2t_base + 32 * s2t_idx
                if prims.elect_sync():
                    prims.tcgen05_cp(
                        s2t_shape,
                        tmem_ptr,
                        desc_s2t,
                        group=self.cfg.cta_group,
                        multicast=s2t_multicast,
                    )

    @consumer_work(returns=sfa_stage_col_offset)
    @cute.jit
    def publish_sfa_offset(self, stage_info: StageInfo) -> cutlass.Int32:
        """Publish TMEM SF stage offset for MMA."""
        sf_col_offset = stage_info.stage_idx * self.cfg.tmem_sfa_cols
        return sf_col_offset


@dataclass(kw_only=True)
class TmemSfBResource(MemoryResource):
    """TMEM for scale-factor B (LDS+STTM from CopySfB task)."""

    cfg: Constexpr[BatchedGemmConfig]
    sfb_tmem_addr_base: Any = None
    _alloc_sfb: Constexpr[Optional[TmemAllocation]] = None
    sfb_stage_col_offset: Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )

    def __post_init__(self):
        if self._alloc_sfb is None:
            stage_mult = (
                self.cfg.num_stages_tmem_sfb
                if self.cfg.uses_unfused_tmem_sf_copy
                else 1
            )
            self._alloc_sfb = TmemAllocation(
                f"{self.name}_tmem_sfb",
                num_columns=self.cfg.tmem_sfb_cols * stage_mult,
            )
        self.sfb_stage_col_offset = TaskLocalVariable(
            dtype=Int32, default=Int32(0), docs="TMEM SFB stage column offset."
        )

    def get_tmem_requirements(self):
        return [self._alloc_sfb]

    @cute.jit
    def _init_tmem_state(self, stage_info: StageInfo) -> None:
        context = stage_info.context
        tmem_raw = context.tmem_ptr_i32.load()
        base_col = tmem_raw & 0xFFFF
        base_row = tmem_raw >> 16
        sfb_col = base_col + self._alloc_sfb.offset
        self.sfb_tmem_addr_base = (base_row << 16) | sfb_col

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_copy_state(self, stage_info: StageInfo) -> None:
        self._init_tmem_state(stage_info)

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_mma_state(self, stage_info: StageInfo) -> None:
        self._init_tmem_state(stage_info)

    @producer_work
    @cute.jit
    def copy_sfb(
        self,
        stage_info: StageInfo,
        *,
        desc_b_s2t_base: prims.Tcgen05SmemDesc,
    ) -> None:
        """S2T copy SFB from SMEM to TMEM via tcgen05_cp."""
        desc_s2t_base = desc_b_s2t_base
        sf_stage_col_offset = stage_info.stage_idx * self.cfg.tmem_sfb_cols
        s2t_shape, s2t_multicast = prims.S2TCopyMode.S2T_32x128b_WARPX4
        num_s2t_iters = self.cfg.tmem_sfb_cols // TMEM_SF_UTCCP_COLS_PER_COPY

        do_copy = cutlass.Boolean(True)
        if cutlass.const_expr(self.cfg.has_routed_sfs and self.cfg.has_cluster):
            cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
            do_copy = cta_rank == Int32(0)
        if do_copy:
            for s2t_idx in cutlass.range_constexpr(num_s2t_iters):
                tmem_addr = (
                    self.sfb_tmem_addr_base
                    + sf_stage_col_offset
                    + s2t_idx * TMEM_SF_UTCCP_COLS_PER_COPY
                )
                tmem_ptr = prims.make_tmem_ptr(tmem_addr, cutlass.Int32)
                # SFB increment pattern differs from SFA (interleaved)
                increment = 32 * (s2t_idx // 2) + 128 * (s2t_idx % 2)
                desc_s2t = desc_s2t_base + increment
                if prims.elect_sync():
                    prims.tcgen05_cp(
                        s2t_shape,
                        tmem_ptr,
                        desc_s2t,
                        group=self.cfg.cta_group,
                        multicast=s2t_multicast,
                    )

    @consumer_work(returns=sfb_stage_col_offset)
    @cute.jit
    def publish_sfb_offset(self, stage_info: StageInfo) -> cutlass.Int32:
        """Publish TMEM SF stage offset for MMA."""
        sf_col_offset = stage_info.stage_idx * self.cfg.tmem_sfb_cols
        return sf_col_offset


@dataclass(kw_only=True)
class TmemSfABResource(MemoryResource):
    """Combined TMEM for generated HT SFA+SFB CopySfAb task."""

    cfg: Constexpr[BatchedGemmConfig]
    sfab_tmem_addr_base: Any = None
    _alloc_sfab: Constexpr[Optional[TmemAllocation]] = None
    sfa_stage_col_offset: Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    sfb_stage_col_offset: Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )

    def __post_init__(self):
        if self._alloc_sfab is None:
            num_columns = (
                self.cfg.tmem_sfa_cols * self.cfg.num_stages_tmem_sfa
                + self.cfg.tmem_sfb_cols * self.cfg.num_stages_tmem_sfb
            )
            self._alloc_sfab = TmemAllocation(
                f"{self.name}_tmem_sfab",
                num_columns=num_columns,
            )
        self.sfa_stage_col_offset = TaskLocalVariable(
            dtype=Int32, default=Int32(0), docs="Combined TMEM SFA offset."
        )
        self.sfb_stage_col_offset = TaskLocalVariable(
            dtype=Int32, default=Int32(0), docs="Combined TMEM SFB offset."
        )

    def get_tmem_requirements(self):
        return [self._alloc_sfab]

    @cute.jit
    def _init_tmem_state(self, stage_info: StageInfo) -> None:
        context = stage_info.context
        tmem_raw = context.tmem_ptr_i32.load()
        base_col = tmem_raw & 0xFFFF
        base_row = tmem_raw >> 16
        sfab_col = base_col + self._alloc_sfab.offset
        self.sfab_tmem_addr_base = (base_row << 16) | sfab_col

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_copy_state(self, stage_info: StageInfo) -> None:
        self._init_tmem_state(stage_info)

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_mma_state(self, stage_info: StageInfo) -> None:
        self._init_tmem_state(stage_info)

    @producer_work
    @cute.jit
    def copy_sfab(
        self,
        stage_info: StageInfo,
        *,
        desc_a_s2t_base: prims.Tcgen05SmemDesc,
        smem_sfa_stage_ptr: Int64,
        desc_b_s2t_base: prims.Tcgen05SmemDesc,
    ) -> None:
        """Copy both SFA and SFB stages through one generated-style pipeline."""
        sfa_stage_col_offset = stage_info.stage_idx * self.cfg.tmem_sfa_cols
        sfb_stage_col_offset = stage_info.stage_idx * self.cfg.tmem_sfb_cols
        sfb_base_col = self.cfg.tmem_sfa_cols * self.cfg.num_stages_tmem_sfa
        s2t_shape, s2t_multicast = prims.S2TCopyMode.S2T_32x128b_WARPX4

        cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        if cta_rank == Int32(0):
            for s2t_idx in cutlass.range_constexpr(
                self.cfg.tmem_sfa_cols // TMEM_SF_UTCCP_COLS_PER_COPY
            ):
                tmem_addr = (
                    self.sfab_tmem_addr_base
                    + sfa_stage_col_offset
                    + s2t_idx * TMEM_SF_UTCCP_COLS_PER_COPY
                )
                tmem_ptr = prims.make_tmem_ptr(tmem_addr, cutlass.Int32)
                desc_s2t = desc_a_s2t_base + 32 * s2t_idx
                if prims.elect_sync():
                    prims.tcgen05_cp(
                        s2t_shape,
                        tmem_ptr,
                        desc_s2t,
                        group=self.cfg.cta_group,
                        multicast=s2t_multicast,
                    )

            for s2t_idx in cutlass.range_constexpr(
                self.cfg.tmem_sfb_cols // TMEM_SF_UTCCP_COLS_PER_COPY
            ):
                tmem_addr = (
                    self.sfab_tmem_addr_base
                    + sfb_base_col
                    + sfb_stage_col_offset
                    + s2t_idx * TMEM_SF_UTCCP_COLS_PER_COPY
                )
                tmem_ptr = prims.make_tmem_ptr(tmem_addr, cutlass.Int32)
                desc_s2t = desc_b_s2t_base + 32 * s2t_idx
                if prims.elect_sync():
                    prims.tcgen05_cp(
                        s2t_shape,
                        tmem_ptr,
                        desc_s2t,
                        group=self.cfg.cta_group,
                        multicast=s2t_multicast,
                    )

    @consumer_work(returns=(sfa_stage_col_offset, sfb_stage_col_offset))
    @cute.jit
    def publish_sfab_offset(
        self, stage_info: StageInfo
    ) -> tuple[cutlass.Int32, cutlass.Int32]:
        return (
            stage_info.stage_idx * self.cfg.tmem_sfa_cols,
            stage_info.stage_idx * self.cfg.tmem_sfb_cols,
        )


@dataclass(kw_only=True)
class TmemSfRouteResource(MemoryResource):
    """TMEM for SF copied through the generated low-N/routed layout.

    Low-N SFB uses the generated compact LDS+STTM path even when the scale
    factors are not routed. Routed high-N operands use the R128c4 SMEM layout
    and bulk tcgen05_cp path.

    The CopySfRoute task is the producer; MMA is the consumer.
    """

    cfg: Constexpr[BatchedGemmConfig]
    smem_sf_resource: Any = (
        None  # reference to SmemSf resource (for SMEM buffer access)
    )
    sf_tmem_addr_base: Any = None
    _alloc_sf: Constexpr[Optional[TmemAllocation]] = None
    _operand: Constexpr[str] = "a"  # which SF operand: "a" or "b"
    sfa_stage_col_offset: Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    sfb_stage_col_offset: Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )

    def __post_init__(self):
        if self._alloc_sf is None:
            is_b = self._operand == "b"
            cols = self.cfg.tmem_sfb_cols if is_b else self.cfg.tmem_sfa_cols
            self._alloc_sf = TmemAllocation(
                f"{self.name}_tmem_sf_route",
                num_columns=cols
                * (
                    self.cfg.num_stages_tmem_sfb
                    if is_b
                    else self.cfg.num_stages_tmem_sfa
                ),
            )
        self.sfa_stage_col_offset = TaskLocalVariable(
            dtype=Int32, default=Int32(0), docs="Routed TMEM SFA stage offset."
        )
        self.sfb_stage_col_offset = TaskLocalVariable(
            dtype=Int32, default=Int32(0), docs="Routed TMEM SFB stage offset."
        )

    def get_tmem_requirements(self):
        return [self._alloc_sf]

    @cute.jit
    def _setup_common(self, context=None) -> None:
        tmem_raw = context.tmem_ptr_i32.load()
        base_col = tmem_raw & 0xFFFF
        base_row = tmem_raw >> 16
        sf_col = base_col + self._alloc_sf.offset
        self.sf_tmem_addr_base = (base_row << 16) | sf_col

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_copy_state(self, stage_info: StageInfo) -> None:
        self._setup_common(stage_info.context)

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_mma_state(self, stage_info: StageInfo) -> None:
        self._setup_common(stage_info.context)

    @cute.jit
    def _producer_work_impl(self, stage_info: StageInfo, desc_s2t_base) -> None:
        """Copy SF from SMEM to TMEM with the generated layout."""
        is_b = cutlass.const_expr(self._operand == "b")
        smem_layout = self.cfg.smem_sfb_layout if is_b else self.cfg.smem_sfa_layout
        copy_mode = (
            self.cfg.sfb_smem_to_tmem_copy if is_b else self.cfg.sfa_smem_to_tmem_copy
        )
        use_sttm = cutlass.const_expr(copy_mode == int(SfSmemToTmemCopy.LDS_STTM))
        cols_per_stage = self.cfg.tmem_sfb_cols if is_b else self.cfg.tmem_sfa_cols
        sf_stage_col_offset = stage_info.stage_idx * cols_per_stage

        if cutlass.const_expr(use_sttm):
            # Uses LDS+STTM for non-R128c4 SF SMEM layouts.
            # LINEAR uses the uint128 gather4 layout; R8c4 uses the compact
            # uint32 8-row-block layout.
            bytes_per_stage = (
                self.cfg.num_bytes_sfb_per_stage
                if is_b
                else self.cfg.num_bytes_sfa_per_stage
            )
            stage_base = self.smem_sf_resource.smem_buf.subview(
                bytes_per_stage * stage_info.stage_idx
            )
            sf_block_size = (
                self.cfg.input_sf_block_size_b
                if is_b
                else self.cfg.input_sf_block_size_a
            )
            sf_k = self.cfg.tile_k // sf_block_size
            tile_rows = self.cfg.tile_n if is_b else self.cfg.tile_m
            source_slices = (tile_rows + 31) // 32
            sf_operand = "b" if is_b else "a"
            col_stride = self.cfg.tmem_sf_col_stride(sf_operand)
            tidx, _, _ = cute.arch.thread_idx()
            lane_id = tidx % Int32(32)
            if cutlass.const_expr(smem_layout == int(SfLayout.R8c4)):
                num_sttm_iters = sf_k // TMEM_SF_PACK_SIZE_BYTES
                for sttm_idx in cutlass.range_constexpr(num_sttm_iters):
                    src_vals = []
                    for source_slice in cutlass.range_constexpr(source_slices):
                        source_lane = lane_id + Int32(32 * source_slice)
                        sf_tile_idx = (source_lane // Int32(8)) * Int32(
                            num_sttm_iters
                        ) + Int32(sttm_idx)
                        local_vec_idx = source_lane % Int32(8)
                        smem_word_offset = (
                            sf_tile_idx * Int32(8) + local_vec_idx
                        ) * Int32(TMEM_SF_PACK_SIZE_BYTES)
                        smem_word = cutlass.Array(
                            stage_base.data_ptr() + smem_word_offset,
                            dtype=cutlass.Int32,
                            shape=(1,),
                            addrspace=3,
                        )
                        sf_val = Int32(0)
                        if source_lane < Int32(tile_rows):
                            sf_val = smem_word.load()
                        src_vals.append(sf_val)
                    tmem_addr = (
                        self.sf_tmem_addr_base
                        + sf_stage_col_offset
                        + Int32(sttm_idx * col_stride)
                    )
                    tmem_ptr = prims.make_tmem_ptr(tmem_addr, cutlass.Int32)
                    if cutlass.const_expr(source_slices == 6):
                        # tcgen05.st only accepts power-of-two vector lengths.
                        # A 192-row R8c4 tile has six 32-row source slices, so
                        # publish it as the hardware-supported 4+2 sequence.
                        prims.tcgen05_st(
                            "32x32b",
                            tmem_ptr,
                            cutlass.Vector.from_elements(
                                tuple(src_vals[:4]), dtype=cutlass.Int32
                            ),
                        )
                        tail_tmem_ptr = prims.make_tmem_ptr(
                            tmem_addr + Int32(4), cutlass.Int32
                        )
                        prims.tcgen05_st(
                            "32x32b",
                            tail_tmem_ptr,
                            cutlass.Vector.from_elements(
                                tuple(src_vals[4:]), dtype=cutlass.Int32
                            ),
                        )
                    elif cutlass.const_expr(source_slices == 1):
                        src = src_vals[0]
                        prims.tcgen05_st(
                            "32x32b",
                            tmem_ptr,
                            src,
                        )
                    else:
                        src = cutlass.Vector.from_elements(
                            tuple(src_vals), dtype=cutlass.Int32
                        )
                        prims.tcgen05_st(
                            "32x32b",
                            tmem_ptr,
                            src,
                        )
            elif cutlass.const_expr(smem_layout == int(SfLayout.R128c4)):
                # Routed N=192 SFB is staged in two physical R128 blocks but
                # MMA expects six contiguous 32-row TMEM columns. Load the six
                # logical slices from their shuffled R128 positions and emit
                # the same legal x4+x2 STTM sequence used by compact R8c4.
                num_sttm_iters = sf_k // TMEM_SF_PACK_SIZE_BYTES
                for sttm_idx in cutlass.range_constexpr(num_sttm_iters):
                    src_vals = []
                    for source_slice in cutlass.range_constexpr(source_slices):
                        block_idx = source_slice // 4
                        quadrant = source_slice % 4
                        smem_word_offset = (
                            (block_idx * num_sttm_iters + sttm_idx) * 512
                            + lane_id * Int32(16)
                            + Int32(quadrant * 4)
                        )
                        smem_word = cutlass.Array(
                            stage_base.data_ptr() + smem_word_offset,
                            dtype=cutlass.Int32,
                            shape=(1,),
                            addrspace=3,
                        )
                        src_vals.append(smem_word.load())
                    tmem_addr = (
                        self.sf_tmem_addr_base
                        + sf_stage_col_offset
                        + Int32(sttm_idx * col_stride)
                    )
                    tmem_ptr = prims.make_tmem_ptr(tmem_addr, cutlass.Int32)
                    prims.tcgen05_st(
                        "32x32b",
                        tmem_ptr,
                        cutlass.Vector.from_elements(
                            tuple(src_vals[:4]), dtype=cutlass.Int32
                        ),
                    )
                    tail_tmem_ptr = prims.make_tmem_ptr(
                        tmem_addr + Int32(4), cutlass.Int32
                    )
                    prims.tcgen05_st(
                        "32x32b",
                        tail_tmem_ptr,
                        cutlass.Vector.from_elements(
                            tuple(src_vals[4:]), dtype=cutlass.Int32
                        ),
                    )
            elif cutlass.const_expr(smem_layout == int(SfLayout.LINEAR)):
                num_vec4 = sf_k // TMEM_SF_PACK_SIZE_BYTES
                num_groups_per_16sf = 16 // TMEM_SF_PACK_SIZE_BYTES
                vec16_per_row = num_vec4 // num_groups_per_16sf
                for vec16_idx in cutlass.range_constexpr(vec16_per_row):
                    for group_in_vec16 in cutlass.range_constexpr(num_groups_per_16sf):
                        word_idx = group_in_vec16
                        src_vals = []
                        for source_slice in cutlass.range_constexpr(source_slices):
                            source_lane = lane_id + Int32(32 * source_slice)
                            raw_vec_idx = source_lane * Int32(vec16_per_row) + Int32(
                                vec16_idx
                            )
                            if cutlass.const_expr(vec16_per_row > 1):
                                rows_per_swizzle = 8 // vec16_per_row
                                smem_row_idx = source_lane // Int32(rows_per_swizzle)
                                xor_mask = smem_row_idx % Int32(vec16_per_row)
                                vec_idx = raw_vec_idx ^ xor_mask
                            else:
                                vec_idx = source_lane // Int32(
                                    TMEM_SF_PACK_SIZE_BYTES
                                ) * Int32(8) + source_lane % Int32(
                                    TMEM_SF_PACK_SIZE_BYTES
                                )
                            smem_word_offset = vec_idx * Int32(16) + Int32(
                                word_idx * TMEM_SF_PACK_SIZE_BYTES
                            )
                            smem_word = cutlass.Array(
                                stage_base.data_ptr() + smem_word_offset,
                                dtype=cutlass.Int32,
                                shape=(1,),
                                addrspace=3,
                            )
                            sf_val = Int32(0)
                            if source_lane < Int32(tile_rows):
                                sf_val = smem_word.load()
                            src_vals.append(sf_val)
                        group_idx = vec16_idx * num_groups_per_16sf + group_in_vec16
                        tmem_addr = (
                            self.sf_tmem_addr_base
                            + sf_stage_col_offset
                            + Int32(col_stride * group_idx)
                        )
                        tmem_ptr = prims.make_tmem_ptr(tmem_addr, cutlass.Int32)
                        if cutlass.const_expr(source_slices == 1):
                            src = src_vals[0]
                        else:
                            src = cutlass.Vector.from_elements(
                                tuple(src_vals), dtype=cutlass.Int32
                            )
                        prims.tcgen05_st(
                            "32x32b",
                            tmem_ptr,
                            src,
                        )
            else:
                raise AssertionError(
                    f"Unsupported LDS+STTM SMEM SF layout: {smem_layout}"
                )
            # Lowers to tcgen05.wait::st.sync.aligned, ordering the STTM
            # writes above before the TS full-barrier commit consumed by MMA.
            cute.arch.fence_view_async_tmem_store()
        else:
            s2t_shape, s2t_multicast = prims.S2TCopyMode.S2T_32x128b_WARPX4
            num_s2t_iters = cols_per_stage // TMEM_SF_UTCCP_COLS_PER_COPY
            cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
            if cta_rank == Int32(0):
                for s2t_idx in cutlass.range_constexpr(num_s2t_iters):
                    tmem_addr = (
                        self.sf_tmem_addr_base
                        + sf_stage_col_offset
                        + s2t_idx * TMEM_SF_UTCCP_COLS_PER_COPY
                    )
                    tmem_ptr = prims.make_tmem_ptr(tmem_addr, cutlass.Int32)
                    desc_s2t = desc_s2t_base + s2t_idx * 32
                    if prims.elect_sync():
                        prims.tcgen05_cp(
                            s2t_shape,
                            tmem_ptr,
                            desc_s2t,
                            group=self.cfg.cta_group,
                            multicast=s2t_multicast,
                        )
                prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
                cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def _consumer_work_impl(self, stage_info: StageInfo):
        """Publish TMEM SF stage offset for MMA."""
        is_b = cutlass.const_expr(self._operand == "b")
        cols_per_stage = self.cfg.tmem_sfb_cols if is_b else self.cfg.tmem_sfa_cols
        sf_col_offset = stage_info.stage_idx * cols_per_stage
        return sf_col_offset


@dataclass(kw_only=True)
class TmemSfRouteAResource(TmemSfRouteResource):
    """TmemSfRoute for operand A."""

    _operand: Constexpr[str] = "a"

    @producer_work
    @cute.jit
    def copy_sfa(
        self,
        stage_info: StageInfo,
        *,
        desc_a_s2t_base: prims.Tcgen05SmemDesc,
        smem_sfa_stage_ptr: Int64,
    ) -> None:
        self._producer_work_impl(stage_info, desc_a_s2t_base)

    @consumer_work(returns=TmemSfRouteResource.sfa_stage_col_offset)
    @cute.jit
    def publish_sfa_offset(self, stage_info: StageInfo) -> cutlass.Int32:
        return self._consumer_work_impl(stage_info)


@dataclass(kw_only=True)
class TmemSfRouteBResource(TmemSfRouteResource):
    """TmemSfRoute for operand B."""

    _operand: Constexpr[str] = "b"

    @producer_work
    @cute.jit
    def copy_sfb(
        self,
        stage_info: StageInfo,
        *,
        desc_b_s2t_base: prims.Tcgen05SmemDesc,
    ) -> None:
        self._producer_work_impl(stage_info, desc_b_s2t_base)

    @consumer_work(returns=TmemSfRouteResource.sfb_stage_col_offset)
    @cute.jit
    def publish_sfb_offset(self, stage_info: StageInfo) -> cutlass.Int32:
        return self._consumer_work_impl(stage_info)


@dataclass(kw_only=True)
class TmemCastAResource(MemoryResource):
    """TMEM for the generated CastA path.

    The producer consumes packed MXE2M1 A and its UE8M0 scale-factor stage,
    casts one 128x256 tile to BF16, and writes it to TMEM.  MMA then consumes
    A from TMEM with the normal BF16 `tcgen05_mma` instruction.
    """

    cfg: Constexpr[BatchedGemmConfig]
    cast_a_tmem_addr_base: Any = None
    _alloc_cast_a: Constexpr[Optional[TmemAllocation]] = None
    tmem_cast_a_addr: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()

    def __post_init__(self):
        if self._alloc_cast_a is None:
            self._alloc_cast_a = TmemAllocation(
                f"{self.name}_tmem_cast_a",
                num_columns=self.cfg.tmem_cast_a_cols * self.cfg.num_stages_cast_a,
            )
        self.tmem_cast_a_addr = TaskLocalVariable(
            dtype=Int32, default=Int32(0), docs="TMEM address for CastA output."
        )

    def get_tmem_requirements(self):
        return [self._alloc_cast_a]

    @cute.jit
    def _init_tmem_state(self, stage_info: StageInfo) -> None:
        context = stage_info.context
        tmem_raw = context.tmem_ptr_i32.load()
        base_col = tmem_raw & 0xFFFF
        base_row = tmem_raw >> 16
        cast_a_col = base_col + self._alloc_cast_a.offset
        self.cast_a_tmem_addr_base = (base_row << 16) | cast_a_col

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_copy_state(self, stage_info: StageInfo) -> None:
        self._init_tmem_state(stage_info)

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_mma_state(self, stage_info: StageInfo) -> None:
        self._init_tmem_state(stage_info)

    @cute.jit
    def _e2m1_bf16_bits(self, nibble: Int32) -> Int32:
        mag = nibble & Int32(7)
        sign = (nibble & Int32(8)) << Int32(12)
        exponent_bits = (Int32(126) + (mag >> Int32(1))) << Int32(7)
        mantissa_bits = (mag & Int32(1)) << Int32(6)
        bits = exponent_bits | mantissa_bits
        if mag == Int32(1):
            bits = exponent_bits
        if mag == Int32(0):
            bits = Int32(0)
        return bits | sign

    @cute.jit
    def _decode_cast_pair(self, packed_byte: Int32) -> Int32:
        lo = packed_byte & Int32(0x0F)
        hi = (packed_byte >> Int32(4)) & Int32(0x0F)
        lo_bits = self._e2m1_bf16_bits(lo)
        hi_bits = self._e2m1_bf16_bits(hi)
        return lo_bits | (hi_bits << Int32(16))

    @cute.jit
    def _cvt_e2m1x2_to_bf16x2_scaled(
        self,
        packed_word: Int32,
        packed_scale_word: Int32,
        byte_lane: Constexpr[int],
        scale_lane: Constexpr[int],
    ) -> Int32:
        # The fused PTX form is bit-exact with this expansion in isolation on
        # CUDA 13.2, but its lowering inside the CastA tile16/cluster2 kernel
        # produces incorrect results. Keep the explicit sequence on every
        # toolchain until that context-sensitive compiler issue is resolved.
        packed_byte = (packed_word >> Int32(byte_lane * 8)) & Int32(0xFF)
        scale_byte = (packed_scale_word >> Int32(scale_lane * 8)) & Int32(0xFF)
        value = self._decode_cast_pair(packed_byte)
        scale_pair = scale_byte | (scale_byte << Int32(8))
        scale = self._cvt_ue8m0x2_to_bf16x2(scale_pair)
        return self._mul_bf16x2(value, scale)

    @cute.jit
    def _cvt_ue8m0x2_to_bf16x2(self, packed_scale: Int32) -> Int32:
        return prims.inline_ptx_hl(
            "{ .reg .b16 scale; "
            "mov.b32 {scale, _}, {$r0}; "
            "cvt.rn.bf16x2.ue8m0x2 {$w0}, scale; }",
            write_only_types=[Int32],
            read_only_args=[packed_scale & Int32(0xFFFF)],
        )

    @cute.jit
    def _mul_bf16x2(self, a: Int32, b: Int32) -> Int32:
        return prims.mul_bf16x2(a, b)

    @cute.jit
    def _load_cast_chunk_word(
        self,
        smem_a_ptr: Int64,
        row: Int32,
        chunk_idx: int,
        word_idx: int,
    ) -> Int32:
        byte_offset = row * Int32(self.cfg.tile_k // 2) + Int32(chunk_idx * 16)
        swizzle_mask = (row % Int32(8)) * Int32(16)
        word_byte_offset = (byte_offset ^ swizzle_mask) + Int32(word_idx * 4)
        word_ptr = cutlass.inttoptr(
            smem_a_ptr + Int64(word_byte_offset),
            3,
            cutlass.Int32,
        )
        return word_ptr.load()

    @cute.jit
    def _load_cast_chunk_words(
        self,
        smem_a_ptr: Int64,
        row: Int32,
        chunk_idx: int,
    ):
        byte_offset = row * Int32(self.cfg.tile_k // 2) + Int32(chunk_idx * 16)
        swizzle_mask = (row % Int32(8)) * Int32(16)
        chunk_byte_offset = byte_offset ^ swizzle_mask
        word_ptr = cutlass.inttoptr(
            smem_a_ptr + Int64(chunk_byte_offset),
            3,
            cutlass.Int32,
        )
        return word_ptr.load(count=4, alignment=16)

    @cute.jit
    def _load_cast_sfa_word(
        self,
        smem_sfa_ptr: Int64,
        row: Int32,
        sf_word_idx: int,
    ) -> Int32:
        sf_word_offset = Int32(sf_word_idx * 128)
        sf_lane_offset = (row % Int32(32)) * Int32(4)
        sf_quad_offset = (row % Int32(128)) // Int32(32)
        sf_word_ptr = cutlass.inttoptr(
            smem_sfa_ptr
            + Int64((sf_word_offset + sf_lane_offset + sf_quad_offset) * Int32(4)),
            3,
            cutlass.Int32,
        )
        return sf_word_ptr.load()

    @producer_work
    @cute.jit
    def cast_a(
        self,
        stage_info: StageInfo,
        *,
        smem_a_stage_ptr: Int64,
        smem_sfa_stage_ptr: Int64,
    ) -> None:
        smem_a_ptr = smem_a_stage_ptr
        smem_sfa_ptr = smem_sfa_stage_ptr
        tidx, _, _ = cute.arch.thread_idx()
        local_thread = tidx % Int32(self.cfg.num_cast_a_warps * 32)
        stage_col_offset = stage_info.stage_idx * self.cfg.tmem_cast_a_cols
        sf_word0 = self._load_cast_sfa_word(smem_sfa_ptr, local_thread, 0)
        sf_word1 = self._load_cast_sfa_word(smem_sfa_ptr, local_thread, 1)

        for slice_idx in cutlass.range_constexpr(4):
            slice_vals = []
            for chunk_in_slice in cutlass.range_constexpr(2):
                chunk_idx = slice_idx * 2 + chunk_in_slice
                sf_word = sf_word0
                if chunk_idx >= 4:
                    sf_word = sf_word1
                words = self._load_cast_chunk_words(smem_a_ptr, local_thread, chunk_idx)
                for word_idx in cutlass.range_constexpr(4):
                    word = words[word_idx]
                    for byte_lane in cutlass.range_constexpr(4):
                        value = self._cvt_e2m1x2_to_bf16x2_scaled(
                            word, sf_word, byte_lane, chunk_idx % 4
                        )
                        slice_vals.append(value)
            src = cutlass.Vector.from_elements(tuple(slice_vals), dtype=cutlass.Int32)
            tmem_addr = (
                self.cast_a_tmem_addr_base + stage_col_offset + Int32(slice_idx * 32)
            )
            tmem_ptr = prims.make_tmem_ptr(tmem_addr, cutlass.Int32)
            prims.tcgen05_st(
                "32x32b",
                tmem_ptr,
                src,
            )
        # Publish the async TMEM-store view before
        # committing the TS full barrier consumed by MMA.
        cute.arch.fence_view_async_tmem_store()

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def sync_cast_a_warps(self, stage_info: StageInfo) -> None:
        prims.barrier_cta_sync(barrier_id=1, thread_count=128)

    @consumer_work(returns=tmem_cast_a_addr)
    @cute.jit
    def publish_cast_a_addr(self, stage_info: StageInfo) -> Int32:
        tmem_cast_a_addr = (
            self.cast_a_tmem_addr_base
            + stage_info.stage_idx * self.cfg.tmem_cast_a_cols
        )
        return tmem_cast_a_addr
