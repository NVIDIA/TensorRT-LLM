# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""TMEM output-accumulator resource for PV MMA."""

from dataclasses import dataclass
from typing import ClassVar

from cutlass.experimental import primitives as prims

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Float32, Int32
from cutlass.experimental import primitives as cprims
from cutlass.experimental.task_scheduling.enums import WorkAttr
from cutlass.experimental.task_scheduling.memory import TmemAllocation
from cutlass.experimental.task_scheduling.resources import (
    StageInfo,
    TaskLocalVariable,
    consumer_work,
    producer_work,
)

from ...helpers.constants import (
    WARP_LANES,
)

from ...helpers.layout import (
    _TASK_CACHE_TMEM_BASE_OFFSET,
    SMEM_ROW_BYTES,
    decode_gen_task_cache,
    o_stage_tmem_col_offset,
    q_p_desc_k_block_wrap_bytes,
)
from ...helpers.math import (
    mma_k_step_for_qkv,
    mma_kind_for_qkv,
    qkv_dtype,
    qkv_smem_swizzle,
)
from ...helpers.ops import (
    freeze_smem_descriptor,
    tcgen05_panel_addr,
)
from ...helpers.tile import (
    batch_idx_for_stage_cfg,
    cta_idx_q_for_stage,
    runtime_local_kv_tiles,
    runtime_seq_len_kv_from_task_cache,
)

from .common import (
    TCGEN05_BF16_SWIZZLE_STRIDE_BYTES,
    MlaResource,
)

# =====================================================================
# TmemOResource — O accumulator in TMEM, UmmaProducerAsync pipeline
# =====================================================================


@dataclass(kw_only=True)
class TmemOResource(MlaResource):
    """TMEM O accumulator resource that corrects and stores final outputs."""

    _task_local_specs: ClassVar[tuple[tuple, ...]] = (
        ("o_stage_idx", Int32, Int32(0), "Current O TMEM stage."),
        ("tail_o_stage_idx_0", Int32, Int32(0), "Final O stage for instance 0."),
        ("tail_o_stage_idx_1", Int32, Int32(1), "Final O stage for instance 1."),
    )
    p0_ref: object = None
    p1_ref: object = None
    p_tmem_ref: object = None
    cache_seqs: object = None
    batch_idx: object = None
    cta_idx_q: object = None
    o_stage_idx: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    tail_o_stage_idx_0: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    tail_o_stage_idx_1: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )

    def get_tmem_requirements(self):
        if self._tmem_alloc is None:
            o_buffer_cols = self.cfg.tmem_o_buffer_cols * self.cfg.v_head_dim_stages
            if (
                self.cfg.kernel_variant == "keeps_mma_ab"
                and self.cfg.head_dim_per_cta_v > 256
            ):
                o_buffer_cols = 2 * self.cfg.tmem_o_buffer_cols
            self._tmem_alloc = TmemAllocation(
                name=f"{self.name}",
                num_columns=self.cfg.o_stages * o_buffer_cols,
            )
        return [self._tmem_alloc]

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_mma_state(self, stage_info: StageInfo) -> None:
        """Initialize O TMEM state needed by the MMA producer path."""
        # Producer aux work initializes the O accumulator TMEM view before PV
        # MMA starts writing into it.
        self._init_tmem_state(stage_info)

    @consumer_work(
        work_attrs=WorkAttr.AUXILIARY,
        returns=(o_stage_idx, tail_o_stage_idx_0, tail_o_stage_idx_1),
    )
    @cute.jit
    def init_stage_state(self, stage_info: StageInfo):
        """Initialize O-stage index variables for the first work tile."""
        # Consumer aux work creates the stage-tracking variables used by the
        # correction task after it waits for O.
        self._init_tmem_state(stage_info)
        return Int32(0), Int32(0), Int32(1)

    @consumer_work(
        work_attrs=WorkAttr.AUXILIARY,
        returns=(o_stage_idx, tail_o_stage_idx_0, tail_o_stage_idx_1),
    )
    @cute.jit
    def init_stage_work_tile_state(self, stage_info: StageInfo):
        """Initialize O-stage index variables for a persistent work tile."""
        # Reset O stage bookkeeping for each persistent work tile.
        del stage_info
        return Int32(0), Int32(0), Int32(1)

    @cute.jit
    def _tail_has_prior_o(self, stage_info: StageInfo):
        """Return whether tail PV should accumulate into existing O state."""
        batch_idx = batch_idx_for_stage_cfg(self.batch_idx, self.cfg, stage_info)
        cta_idx_q = cta_idx_q_for_stage(self.cta_idx_q, stage_info)
        seq_len_kv = runtime_seq_len_kv_from_task_cache(
            self.cfg,
            decode_gen_task_cache(stage_info),
            cta_idx_q,
            self.cu_seqlens_q,
            batch_idx,
        )
        return runtime_local_kv_tiles(self.cfg, seq_len_kv) > Int32(
            self.cfg.num_insts_kv
        )

    @cute.jit
    def _p_desc_for_inst(self, p_inst: int):
        """Build the P SMEM descriptor for one softmax/PV pipe instance."""
        if cutlass.const_expr(p_inst == 0):
            return cprims.Tcgen05SmemDesc.build(
                self.p0_ref._smem_p,
                leading_byte_offset=Int32(self.cfg.tile_size_q * SMEM_ROW_BYTES),
                stride_byte_offset=TCGEN05_BF16_SWIZZLE_STRIDE_BYTES,
                layout=qkv_smem_swizzle(self.cfg),
            )
        return cprims.Tcgen05SmemDesc.build(
            self.p1_ref._smem_p,
            leading_byte_offset=Int32(self.cfg.tile_size_q * SMEM_ROW_BYTES),
            stride_byte_offset=TCGEN05_BF16_SWIZZLE_STRIDE_BYTES,
            layout=qkv_smem_swizzle(self.cfg),
        )

    @cute.jit
    def _pv_o_base_addr(
        self, stage_info: StageInfo, *, v_subtile_idx: cutlass.Constexpr[int]
    ):
        """Return the TMEM O base address for this PV stage."""
        cfg = self.cfg
        v_stage_idx = v_subtile_idx % cfg.v_head_dim_stages
        task_cache = decode_gen_task_cache(stage_info)
        return (
            task_cache[_TASK_CACHE_TMEM_BASE_OFFSET]
            + Int32(self._tmem_alloc.offset)
            + o_stage_tmem_col_offset(cfg, stage_info.stage_idx, v_stage_idx)
        )

    @cute.jit
    def _issue_smem_p_pv_mma(
        self,
        stage_info: StageInfo,
        p_desc,
        v_desc,
        scale_d,
        *,
        v_subtile_idx: cutlass.Constexpr[int],
    ):
        """Issue PV MMA with P sourced from SMEM and O accumulated in TMEM."""
        cfg = self.cfg
        idesc = cprims.Tcgen05InstrDesc.build(
            c_dtype=Float32,
            a_dtype=qkv_dtype(cfg),
            b_dtype=qkv_dtype(cfg),
            a_major=1,
            b_major=0,
            n_dim=cfg.tile_size_q,
            m_dim=cfg.head_dim_per_stage_v,
        )
        if prims.elect_sync():
            tmem_col = prims.make_tmem_ptr(
                self._pv_o_base_addr(stage_info, v_subtile_idx=v_subtile_idx),
                Float32,
            )
            for k_block in cutlass.range_constexpr(
                cfg.tile_size_kv // mma_k_step_for_qkv(cfg)
            ):
                prims.tcgen05_mma(
                    mma_kind_for_qkv(cfg),
                    prims.CTAGroup.CTA_1,
                    tmem_col,
                    v_desc,
                    p_desc,
                    idesc,
                    scale_d,
                )
                scale_d = Boolean(True)
                if cutlass.const_expr(
                    k_block + 1 < cfg.tile_size_kv // mma_k_step_for_qkv(cfg)
                ):
                    v_desc = v_desc + Int32(256 if cfg.is_fp8_qkv() else 128)
                    if cutlass.const_expr(k_block == 3):
                        p_desc = p_desc.advance_start_address(
                            Int32(q_p_desc_k_block_wrap_bytes(cfg))
                        )
                    else:
                        p_desc = p_desc.advance_start_address(Int32(16 * 2))

    @producer_work
    @cute.jit
    def pv_mma_loop_0(
        self,
        stage_info: StageInfo,
        *,
        v_desc_0,
        v_subtile_idx: cutlass.Constexpr[int],
        is_tail: cutlass.Constexpr[bool] = False,
    ):
        """Issue loop PV MMA instance 0 from P0/V0."""
        del is_tail
        self._issue_smem_p_pv_mma(
            stage_info,
            self._p_desc_for_inst(0),
            freeze_smem_descriptor(v_desc_0),
            stage_info.loop_offset != Int32(0),
            v_subtile_idx=v_subtile_idx,
        )

    @producer_work
    @cute.jit
    def pv_mma_tail_0(
        self,
        stage_info: StageInfo,
        *,
        v_desc_0,
        v_subtile_idx: cutlass.Constexpr[int],
        is_tail: cutlass.Constexpr[bool] = True,
    ):
        """Issue tail PV MMA instance 0 from P0/V0."""
        del is_tail
        self._issue_smem_p_pv_mma(
            stage_info,
            self._p_desc_for_inst(0),
            freeze_smem_descriptor(v_desc_0),
            self._tail_has_prior_o(stage_info),
            v_subtile_idx=v_subtile_idx,
        )

    @producer_work
    @cute.jit
    def pv_mma_loop_1(
        self,
        stage_info: StageInfo,
        *,
        v_desc_1,
        v_subtile_idx: cutlass.Constexpr[int],
        is_tail: cutlass.Constexpr[bool] = False,
    ):
        """Issue loop PV MMA instance 1 from P1/V1."""
        del is_tail
        self._issue_smem_p_pv_mma(
            stage_info,
            self._p_desc_for_inst(1),
            freeze_smem_descriptor(v_desc_1),
            stage_info.loop_offset != Int32(0),
            v_subtile_idx=v_subtile_idx,
        )

    @producer_work
    @cute.jit
    def pv_mma_tail_1(
        self,
        stage_info: StageInfo,
        *,
        v_desc_1,
        v_subtile_idx: cutlass.Constexpr[int],
        is_tail: cutlass.Constexpr[bool] = True,
    ):
        """Issue tail PV MMA instance 1 from P1/V1."""
        del is_tail
        self._issue_smem_p_pv_mma(
            stage_info,
            self._p_desc_for_inst(1),
            freeze_smem_descriptor(v_desc_1),
            self._tail_has_prior_o(stage_info),
            v_subtile_idx=v_subtile_idx,
        )

    @producer_work
    @cute.jit
    def pv_mma_loop_tmem_p(
        self,
        stage_info: StageInfo,
        *,
        p_stage_idx,
        v_desc_0,
        v_subtile_idx: cutlass.Constexpr[int],
        is_tail: cutlass.Constexpr[bool] = False,
    ):
        """Issue loop PV MMA with P read from TMEM instead of SMEM."""
        del is_tail
        self._issue_tmem_p_pv_mma(
            stage_info,
            p_stage_idx=p_stage_idx,
            v_desc=freeze_smem_descriptor(v_desc_0),
            scale_d=stage_info.loop_offset != Int32(0),
            v_subtile_idx=v_subtile_idx,
        )

    @producer_work
    @cute.jit
    def pv_mma_tail_tmem_p(
        self,
        stage_info: StageInfo,
        *,
        p_stage_idx,
        v_desc_0,
        v_subtile_idx: cutlass.Constexpr[int],
        is_tail: cutlass.Constexpr[bool] = True,
    ):
        """Issue tail PV MMA with P read from TMEM instead of SMEM."""
        del is_tail
        self._issue_tmem_p_pv_mma(
            stage_info,
            p_stage_idx=p_stage_idx,
            v_desc=freeze_smem_descriptor(v_desc_0),
            scale_d=self._tail_has_prior_o(stage_info),
            v_subtile_idx=v_subtile_idx,
        )

    @cute.jit
    def _issue_tmem_p_pv_mma(
        self,
        stage_info: StageInfo,
        *,
        p_stage_idx,
        v_desc,
        scale_d,
        v_subtile_idx: cutlass.Constexpr[int],
    ):
        """Issue PV MMA for keeps-MMA-AB with P sourced from TMEM."""
        cfg = self.cfg
        v_stage_idx = v_subtile_idx % cfg.v_head_dim_stages
        task_cache = decode_gen_task_cache(stage_info)
        p_addr = (
            task_cache[_TASK_CACHE_TMEM_BASE_OFFSET]
            + Int32(self.p_tmem_ref.tmem_alias_ref._tmem_alloc.offset)
            + Int32(WARP_LANES)
            + Int32(p_stage_idx) * Int32(cfg.tmem_s_cols)
        )
        if cutlass.const_expr(cfg.head_dim_per_cta_v > 256):
            p_addr = tcgen05_panel_addr(p_addr, v_stage_idx // 2)
        idesc = cprims.Tcgen05InstrDesc.build(
            c_dtype=Float32,
            a_dtype=qkv_dtype(cfg),
            b_dtype=qkv_dtype(cfg),
            a_major=0,
            b_major=1,
            n_dim=cfg.head_dim_per_stage_v,
            m_dim=cfg.tile_size_q,
        )
        if prims.elect_sync():
            base_addr = self._pv_o_base_addr(stage_info, v_subtile_idx=v_subtile_idx)
            for k_block in cutlass.range_constexpr(
                cfg.tile_size_kv // mma_k_step_for_qkv(cfg)
            ):
                prims.tcgen05_mma(
                    mma_kind_for_qkv(cfg),
                    prims.CTAGroup.CTA_1,
                    prims.make_tmem_ptr(base_addr, Float32),
                    prims.make_tmem_ptr(p_addr, qkv_dtype(cfg)),
                    v_desc,
                    idesc,
                    scale_d,
                )
                scale_d = Boolean(True)
                if cutlass.const_expr(
                    k_block + 1 < cfg.tile_size_kv // mma_k_step_for_qkv(cfg)
                ):
                    p_addr = p_addr + Int32(8)
                    v_desc = v_desc + Int32(256 if cfg.is_fp8_qkv() else 128)

    @consumer_work(
        returns=(o_stage_idx, tail_o_stage_idx_0, tail_o_stage_idx_1),
    )
    @cute.jit
    def o_stage(
        self,
        stage_info: StageInfo,
        *,
        tail_o_stage_idx_0,
        tail_o_stage_idx_1,
        inst_idx: cutlass.Constexpr[int],
        is_tail: cutlass.Constexpr[bool] = False,
    ):
        """Publish the TMEM O stage index for correction/output."""
        # Consumer work for O: correction has waited for PV MMA to finish.  It
        # receives the current O stage, plus tail-stage bookkeeping so it can
        # combine the two interleaved softmax/MMA instances.
        o_stage_idx = stage_info.stage_idx
        # The tail stage is the only point where the correction task needs to
        # remember both final O stages for cross-instance normalization.
        if cutlass.const_expr(is_tail):
            if cutlass.const_expr(self.cfg.kernel_variant == "keeps_mma_ab"):
                tail_o_stage_idx_0 = stage_info.stage_idx
                tail_o_stage_idx_1 = stage_info.stage_idx
            elif cutlass.const_expr(inst_idx == 0):
                tail_o_stage_idx_0 = stage_info.stage_idx
            else:
                tail_o_stage_idx_1 = stage_info.stage_idx
        return o_stage_idx, tail_o_stage_idx_0, tail_o_stage_idx_1
