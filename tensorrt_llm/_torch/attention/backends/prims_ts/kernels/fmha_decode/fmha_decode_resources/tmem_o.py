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

"""``TmemOResource`` — TMEM O accumulator for BMM2.

Producer (MmaTask): P × V MMA → O. Consumer (Correction): tracks which
O stage is ready (``o_stage_idx`` plus tail stage indices) so the in-place
rescale path can find the correct columns.
"""

from dataclasses import dataclass
from typing import ClassVar

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32
from cutlass.experimental import primitives as prims

from cutlass.experimental.task_scheduling.enums import WorkAttr
from cutlass.experimental.task_scheduling.memory import SmemAllocation, TmemAllocation
from cutlass.experimental.task_scheduling.resources import (
    StageInfo,
    TaskLocalVariable,
    consumer_work,
    producer_work,
)

from ..fmha_decode_constants import KV_INST0
from ..fmha_decode_config import FmhaDecodeConfig
from ...tcgen05_compat import tcgen05_mma_ws
from .helpers_common import (
    Constexpr,
    DecodeGenResourceBase,
    DescriptorValue,
    _TASK_CACHE_TMEM_BASE_OFFSET,
    _decode_gen_task_cache,
    _freeze_smem_descriptor,
    _mma_k_step,
    _mma_kind_for_qkv,
)


def _pv_mma_operand_contract_for_config(
    cfg: FmhaDecodeConfig,
) -> tuple[bool, int, int, int, int]:
    """Return ``(P-is-A, M, N, A-major, B-major)`` for BMM2."""
    active_head_dim = (
        cfg.headdim if cfg.head_dim_per_stage_kv == 0 else cfg.head_dim_kv_stage
    )
    if cfg.use_keeps_mma_ab:
        if cfg.tile_size_kv == 256:
            # The WS 2x2 PV instruction exposes two spatial D128 partials as
            # one physical KV256 operation. Correction merges those spatial
            # halves after the two temporal decode streams are complete.
            return True, cfg.tile_size_q, cfg.tile_size_kv, 0, 1
        return True, cfg.tile_size_q, active_head_dim, 0, 1
    return False, active_head_dim, cfg.tile_size_q, 1, 0


@dataclass(kw_only=True)
class TmemOResource(DecodeGenResourceBase):
    """TMEM O accumulator for BMM2.

    Producers run logical P x V MMA into staged TMEM O columns. Correction
    consumers track which O stage is ready and rescale or output it.
    """

    _task_local_specs: ClassVar[tuple[tuple, ...]] = (
        ("o_stage_idx", Int32, Int32(0), "Current O TMEM stage index."),
        (
            "tail_o_stage_idx_0",
            Int32,
            Int32(0),
            "O TMEM stage index for the first tail VP wave.",
        ),
        (
            "tail_o_stage_idx_1",
            Int32,
            Int32(1),
            "O TMEM stage index for the second tail VP wave.",
        ),
    )
    cfg: Constexpr[FmhaDecodeConfig] = None
    scale_softmax_log2: Float32 = None
    _alloc: Constexpr[TmemAllocation | None] = None
    o_stage_idx: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    tail_o_stage_idx_0: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    tail_o_stage_idx_1: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()

    def get_smem_requirements(self) -> list[SmemAllocation]:
        """O accumulation lives in TMEM and needs no SMEM allocation."""
        return []

    def get_tmem_requirements(self) -> list[TmemAllocation]:
        """Allocate staged TMEM columns for logical O accumulators."""
        cfg = self.cfg
        if self._alloc is None:
            self._alloc = TmemAllocation(
                name=f"{self.name}",
                num_columns=cfg.tmem_o_stage_cols * cfg.o_stages,
            )
        return [self._alloc]

    @consumer_work(
        work_attrs=WorkAttr.AUXILIARY,
        returns=(o_stage_idx, tail_o_stage_idx_0, tail_o_stage_idx_1),
    )
    @cute.jit
    def init_stage_state(self, stage_info: StageInfo) -> tuple[Int32, Int32, Int32]:
        """Initialize live and tail O-stage indices for correction."""
        # ConsAuxWork: seed correction's O-stage tracker before any PV MMA
        # stages have been committed.
        del stage_info
        return Int32(0), Int32(0), Int32(1)

    @producer_work
    @cute.jit
    def vp_mma_loop(
        self,
        stage_info: StageInfo,
        *,
        v_desc_0: DescriptorValue,
        v_desc_1: DescriptorValue,
        p_desc_0: DescriptorValue,
        p_desc_1: DescriptorValue,
        p_tmem_addr_0: Int32,
        p_tmem_addr_1: Int32,
        inst_idx: Constexpr[int],
        head_dim_stage_idx: Constexpr[int],
    ) -> None:
        """Issue loop PV MMA with overwrite on the first K/V tile only."""
        # ProdWork: loop PV accumulates into the live O stage; later correction
        # work consumes this stage before another PV wave reuses it.
        self._vp_mma(
            stage_info,
            v_desc_0=v_desc_0,
            v_desc_1=v_desc_1,
            p_desc_0=p_desc_0,
            p_desc_1=p_desc_1,
            p_tmem_addr_0=p_tmem_addr_0,
            p_tmem_addr_1=p_tmem_addr_1,
            initial_scale_d=stage_info.loop_offset != Int32(0),
            inst_idx=inst_idx,
            head_dim_stage_idx=head_dim_stage_idx,
        )

    @producer_work
    @cute.jit
    def vp_mma_tail(
        self,
        stage_info: StageInfo,
        *,
        v_desc_0: DescriptorValue,
        v_desc_1: DescriptorValue,
        p_desc_0: DescriptorValue,
        p_desc_1: DescriptorValue,
        p_tmem_addr_0: Int32,
        p_tmem_addr_1: Int32,
        inst_idx: Constexpr[int],
        head_dim_stage_idx: Constexpr[int],
    ) -> None:
        """Issue tail PV MMA after all loop K tiles have been launched."""
        # ProdWork: tail PV publishes one of the final O stages consumed by the
        # tail correction path.
        self._vp_mma(
            stage_info,
            v_desc_0=v_desc_0,
            v_desc_1=v_desc_1,
            p_desc_0=p_desc_0,
            p_desc_1=p_desc_1,
            p_tmem_addr_0=p_tmem_addr_0,
            p_tmem_addr_1=p_tmem_addr_1,
            initial_scale_d=stage_info.loop_end != Int32(0),
            inst_idx=inst_idx,
            head_dim_stage_idx=head_dim_stage_idx,
        )

    @producer_work
    @cute.jit
    def vp_mma_loop_fragment(
        self,
        stage_info: StageInfo,
        *,
        v_desc: DescriptorValue,
        p_tmem_addr: Int32,
        fragment_idx: Constexpr[int],
    ) -> None:
        """Issue one K32 fragment of a KV256 loop PV tile."""
        self._vp_mma_fragment(
            stage_info,
            v_desc=v_desc,
            p_tmem_addr=p_tmem_addr,
            fragment_idx=fragment_idx,
            initial_scale_d=stage_info.loop_offset != Int32(0),
        )

    @producer_work
    @cute.jit
    def vp_mma_tail_fragment(
        self,
        stage_info: StageInfo,
        *,
        v_desc: DescriptorValue,
        p_tmem_addr: Int32,
        fragment_idx: Constexpr[int],
    ) -> None:
        """Issue one K32 fragment of the final KV256 PV tile."""
        self._vp_mma_fragment(
            stage_info,
            v_desc=v_desc,
            p_tmem_addr=p_tmem_addr,
            fragment_idx=fragment_idx,
            initial_scale_d=stage_info.loop_end != Int32(0),
        )

    @cute.jit
    def _vp_mma_fragment(
        self,
        stage_info: StageInfo,
        *,
        v_desc: DescriptorValue,
        p_tmem_addr: Int32,
        fragment_idx: Constexpr[int],
        initial_scale_d,
    ) -> None:
        """Issue the two WS MMA steps covered by one KV256 P fragment.

        ``p_tmem_addr`` is already the base of the fragment selected by
        ``wait_p_fragment``. Only the two local K-step offsets are added here;
        ``fragment_idx`` must not be applied to the TMEM address again.
        """
        cfg = self.cfg
        assert cfg.tile_size_kv == 256 and cfg.uses_two_inst_tmem_p
        v_desc = _freeze_smem_descriptor(v_desc)

        task_cache = _decode_gen_task_cache(stage_info)
        tmem_col = prims.make_tmem_ptr(
            task_cache[_TASK_CACHE_TMEM_BASE_OFFSET]
            + Int32(self._alloc.offset)
            + stage_info.stage_idx * cfg.tmem_o_cols,
            Float32,
        )
        _, mma_m, mma_n, a_major, b_major = _pv_mma_operand_contract_for_config(cfg)
        idesc = prims.Tcgen05InstrDesc.build(
            c_dtype=Float32,
            a_dtype=cfg.q_dtype,
            b_dtype=cfg.q_dtype,
            a_major=a_major,
            b_major=b_major,
            n_dim=mma_n,
            m_dim=mma_m,
        )
        first_k_step = fragment_idx * 2
        if prims.elect_sync():
            for local_k_step in cutlass.range_constexpr(2):
                k_step = first_k_step + local_k_step
                p_operand = prims.make_tmem_ptr(
                    p_tmem_addr + Int32(local_k_step * 8), Int32
                )
                iter_v_desc = v_desc + Int32(
                    (k_step // 4) * cfg.headdim * 16 + (k_step % 4) * 128
                )
                tcgen05_mma_ws(
                    _mma_kind_for_qkv(cfg),
                    tmem_col,
                    p_operand,
                    iter_v_desc,
                    idesc,
                    initial_scale_d or fragment_idx != 0 or local_k_step != 0,
                )

    @cute.jit
    def _vp_mma(
        self,
        stage_info: StageInfo,
        *,
        v_desc_0: DescriptorValue,
        v_desc_1: DescriptorValue,
        p_desc_0: DescriptorValue,
        p_desc_1: DescriptorValue,
        p_tmem_addr_0: Int32,
        p_tmem_addr_1: Int32,
        initial_scale_d,
        inst_idx: Constexpr[int],
        head_dim_stage_idx: Constexpr[int],
    ) -> None:
        """Issue one non-fragmented PV MMA wave for loop or tail work."""
        cfg = self.cfg
        # KV256 is always streamed through _vp_mma_fragment so no full 128-P
        # row is kept live. Keep this routine as the sole generic PV path.
        assert cfg.tile_size_kv != 256
        # Select the descriptor pair for this BMM2 call. With staged head
        # dimensions, consecutive calls belong to the same KV instance and
        # different head-dim slices.
        if cutlass.const_expr(inst_idx == KV_INST0):
            v_desc = v_desc_0
            p_desc = p_desc_0
            p_tmem_addr = p_tmem_addr_0
        else:
            v_desc = v_desc_1
            p_desc = p_desc_1
            p_tmem_addr = p_tmem_addr_1
        v_desc = _freeze_smem_descriptor(v_desc)
        if cutlass.const_expr(not cfg.uses_tmem_p):
            p_desc = _freeze_smem_descriptor(p_desc)
        task_cache = _decode_gen_task_cache(stage_info)
        p_is_a, mma_m, mma_n, a_major, b_major = _pv_mma_operand_contract_for_config(
            cfg
        )

        if cutlass.const_expr(cfg.head_dim_per_stage_kv == 0):
            # Full-headDim path: all V columns for this O stage live in one
            # contiguous TMEM stage selected by the TS pipeline stage index.
            base_addr = task_cache[_TASK_CACHE_TMEM_BASE_OFFSET] + Int32(
                self._alloc.offset + stage_info.stage_idx * cfg.tmem_o_cols
            )
            tmem_col = cutlass.inttoptr(
                base_addr,
                6,
                Float32,
            )
            idesc = prims.Tcgen05InstrDesc.build(
                c_dtype=Float32,
                a_dtype=cfg.q_dtype,
                b_dtype=cfg.q_dtype,
                a_major=a_major,
                b_major=b_major,
                n_dim=mma_n,
                m_dim=mma_m,
            )

            if prims.elect_sync():
                # Accumulate into O after the first K/V tile for this output
                # stage; the first wave overwrites the TMEM O stage.
                scale_d = initial_scale_d
                pv_k_steps = cfg.tile_size_kv // _mma_k_step(cfg)
                for ki in cutlass.range_constexpr(pv_k_steps):
                    # Keeps computes P x V (A=P, B=V); Swaps computes the
                    # transposed V^T x P^T tile (A=V, B=P).
                    if cutlass.const_expr(cfg.uses_tmem_p):
                        # TMEM P stores two 16-bit values per column (or four
                        # FP8 values), so each 16-wide MMA-K step advances by
                        # the corresponding packed-column count.
                        p_cols_per_k_step = _mma_k_step(cfg) * cfg.q_dtype_bytes // 4
                        p_operand = prims.make_tmem_ptr(
                            p_tmem_addr + Int32(ki * p_cols_per_k_step),
                            Int32,
                        )
                    else:
                        p_operand = p_desc
                    if cutlass.const_expr(p_is_a):
                        a_desc, b_desc = p_operand, v_desc
                    else:
                        a_desc, b_desc = v_desc, p_operand
                    prims.tcgen05_mma(
                        _mma_kind_for_qkv(cfg),
                        prims.CTAGroup.CTA_1,
                        tmem_col,
                        a_desc,
                        b_desc,
                        idesc,
                        scale_d,
                    )
                    scale_d = True
                    if cutlass.const_expr(ki + 1 < pv_k_steps):
                        # Advance V and P descriptors to the next MMA-K
                        # slice, including the 16-bit 128-token jump across
                        # split SMEM rows.
                        v_desc = v_desc + Int32(
                            (cfg.headdim * 2) if cfg.use_fp8_qkv else 128
                        )
                        if cutlass.const_expr(not cfg.uses_tmem_p):
                            if cutlass.const_expr(
                                not cfg.use_fp8_qkv
                                and cfg.tile_size_kv == 128
                                and ki == 3
                            ):
                                if cutlass.const_expr(cfg.tile_size_q >= 16):
                                    p_desc = p_desc + Int32(8 * cfg.tile_size_q - 6)
                                else:
                                    p_desc = p_desc + Int32(58)
                            else:
                                p_desc = p_desc + Int32(2)
        else:
            # Staged-headDim path: each call owns one head-dim slice of the same
            # logical O stage. The TMEM offset selects that slice before MMA.
            head_dim_stage_tmem_offset = cfg.pv_head_dim_stage_tmem_offset(
                head_dim_stage_idx
            )
            base_addr = (
                task_cache[_TASK_CACHE_TMEM_BASE_OFFSET]
                + Int32(
                    self._alloc.offset + stage_info.stage_idx * cfg.tmem_o_stage_cols
                )
                + Int32(head_dim_stage_tmem_offset)
            )
            tmem_col = cutlass.inttoptr(
                base_addr,
                6,
                Float32,
            )
            idesc = prims.Tcgen05InstrDesc.build(
                c_dtype=Float32,
                a_dtype=cfg.q_dtype,
                b_dtype=cfg.q_dtype,
                a_major=a_major,
                b_major=b_major,
                n_dim=mma_n,
                m_dim=mma_m,
            )

            if prims.elect_sync():
                # Issue one MMA per K step. The first wave may overwrite the
                # slice, while later loop/tail waves accumulate into it.
                scale_d = initial_scale_d
                for ki in cutlass.range_constexpr(cfg.tile_size_kv // _mma_k_step(cfg)):
                    if cutlass.const_expr(cfg.uses_tmem_p):
                        p_cols_per_k_step = _mma_k_step(cfg) * cfg.q_dtype_bytes // 4
                        p_operand = prims.make_tmem_ptr(
                            p_tmem_addr + Int32(ki * p_cols_per_k_step),
                            Int32,
                        )
                    else:
                        p_operand = p_desc
                    if cutlass.const_expr(p_is_a):
                        a_desc, b_desc = p_operand, v_desc
                    else:
                        a_desc, b_desc = v_desc, p_operand
                    prims.tcgen05_mma(
                        _mma_kind_for_qkv(cfg),
                        prims.CTAGroup.CTA_1,
                        tmem_col,
                        a_desc,
                        b_desc,
                        idesc,
                        scale_d,
                    )
                    scale_d = True
                    if cutlass.const_expr(
                        ki + 1 < cfg.tile_size_kv // _mma_k_step(cfg)
                    ):
                        # Advance V and P to the next MMA-K slice inside the
                        # staged head-dim tile.
                        v_desc = v_desc + Int32(
                            (cfg.head_dim_kv_stage * 2) if cfg.use_fp8_qkv else 128
                        )
                        if cutlass.const_expr(not cfg.uses_tmem_p):
                            if cutlass.const_expr(
                                not cfg.use_fp8_qkv
                                and cfg.tile_size_kv == 128
                                and ki == 3
                            ):
                                if cutlass.const_expr(cfg.tile_size_q >= 16):
                                    p_desc = p_desc + Int32(8 * cfg.tile_size_q - 6)
                                else:
                                    p_desc = p_desc + Int32(58)
                            else:
                                p_desc = p_desc + Int32(2)

    @cute.jit
    def _return_o_stage_state(
        self,
        o_stage_idx: Int32,
        tail_o_stage_idx_0: Int32,
        tail_o_stage_idx_1: Int32,
    ) -> tuple[object, object, object]:
        """Return the O-stage task-local tuple in scheduler order."""
        return o_stage_idx, tail_o_stage_idx_0, tail_o_stage_idx_1

    @consumer_work(returns=(o_stage_idx, tail_o_stage_idx_0, tail_o_stage_idx_1))
    @cute.jit
    def update_o_stage_loop(
        self,
        stage_info: StageInfo,
        *,
        tail_o_stage_idx_0: Int32,
        tail_o_stage_idx_1: Int32,
    ) -> tuple[object, object, object]:
        """Record the loop O stage that correction will rescale in place."""
        # ConsWork: loop correction records which O stage became available for
        # in-place rescale.
        o_stage_idx = stage_info.stage_idx
        return self._return_o_stage_state(
            o_stage_idx, tail_o_stage_idx_0, tail_o_stage_idx_1
        )

    @consumer_work(returns=(o_stage_idx, tail_o_stage_idx_0, tail_o_stage_idx_1))
    @cute.jit
    def update_o_stage_tail(
        self,
        stage_info: StageInfo,
        *,
        tail_o_stage_idx_0: Int32,
        tail_o_stage_idx_1: Int32,
        inst_idx: Constexpr[int],
    ) -> tuple[object, object, object]:
        """Record the two final O stages consumed by tail correction."""
        # ConsWork: tail correction consumes one completed O stage per K/V
        # instance and records which TMEM columns hold inst0/inst1.
        o_stage_idx = stage_info.stage_idx
        if cutlass.const_expr(inst_idx == KV_INST0):
            tail_o_stage_idx_0 = stage_info.stage_idx
        else:
            tail_o_stage_idx_1 = stage_info.stage_idx
        return self._return_o_stage_state(
            o_stage_idx, tail_o_stage_idx_0, tail_o_stage_idx_1
        )
