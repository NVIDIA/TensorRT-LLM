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

"""SMEM resource for DeepSeek FP8 dequant scale factors (Act + Weights).

Uses ``SmemDeepSeekSfAbSmem`` struct:

    float mDqSfsAct[num_stages_sf][max(tile_token_dim, 32)]
    float mDqSfsWeights[num_stages_sf][1] # per-expert weight scales

A single cp.async producer warp (LoadSfAbTask) fills these arrays per K-chunk.
The epilogue consumes them in
FP32 (``ffma2(act_lane * weight, partial_acc, acc)``) BEFORE casting the
accumulator to the output dtype. The pipeline backing factory is
``CutlassCpAsyncPipeline<numStagesSmemSfA>`` — in TS, that is
``num_stages_smem_sfa`` passed to
``PipelineConfig.create_async_async_pipeline_cfg(...)``. This module owns the
SMEM allocation, cp.async producer, and epilogue-side dequant helpers needed
to feed per-K-chunk scales into the DeepSeek FP8 epilogue.
"""

from dataclasses import dataclass
from typing import Any, Optional

import cutlass
import cutlass.cute as cute

from cutlass import Float32, Int32, Int64
from cutlass.experimental import primitives as prims

from cutlass.experimental.task_scheduling.enums import WorkAttr
from cutlass.experimental.task_scheduling.memory import SmemAllocation
from cutlass.experimental.task_scheduling.resources import (
    MemoryResource,
    StageInfo,
    TaskLocalVariable,
    consumer_work,
    producer_work,
)

from .batched_gemm_config import BatchedGemmConfig

Constexpr = cutlass.Constexpr


@dataclass(kw_only=True)
class SmemDeepSeekSfAbResource(MemoryResource):
    """SMEM staging for DeepSeek FP8 per-K-chunk FP32 dequant SFs.

    Producer (LoadSfAbTask, single warp): cp.async loads one tile of activation
    floats from ``ptrSfB[token, k_chunk]`` and ``1`` weight float from
    ``ptrSfA[expert, k_chunk]`` per K-tile into the current stage.

    Consumer (epilogue, all 4 warps): waits per K-tile, reads
    ``act_lane * weight`` from the stage, FMA-applies to the partial
    accumulator, then releases.
    """

    cfg: Constexpr[BatchedGemmConfig]
    # GMEM source pointers. Validation builds may
    # leave these unset because schedule construction does not dereference them.
    sfa_gmem_base: Any = None  # ptrSfA — per-expert weight scales
    sfb_gmem_base: Any = None  # ptrSfB — per-token activation scales
    tile_idx_view: Any = None
    mn_limit_view: Any = None
    route_map_view: Any = None
    problem_m: Any = None
    problem_n: Any = None
    problem_k: Any = None
    num_tokens: Any = None
    total_num_padded_tokens_tensor: Any = None
    gTotalNumPaddedTokens: Any = None
    smem_buf: Any = None
    t2r_rmem: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    t2r_rmem_1: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    t2r_output_call_idx: Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    t2r_dequant_scale: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    _alloc_dsfp8_sf: Constexpr[Optional[SmemAllocation]] = None

    def __post_init__(self):
        if self._alloc_dsfp8_sf is None:
            self._alloc_dsfp8_sf = SmemAllocation(
                f"{self.name}_dsfp8_sfab",
                size_bytes=self.cfg.num_bytes_dsfp8_sfab_per_stage
                * self.cfg.num_stages_smem_sfa,
                alignment=16,
            )
        self.t2r_rmem = TaskLocalVariable(
            dtype=cutlass.Float32,
            default_factory=self._t2r_rmem_default,
            docs="Primary dequantized TMEM-to-register fragment.",
        )
        self.t2r_rmem_1 = TaskLocalVariable(
            dtype=cutlass.Float32,
            default_factory=self._t2r_rmem_1_default,
            docs="Secondary dequantized TMEM-to-register fragment.",
        )
        self.t2r_output_call_idx = TaskLocalVariable(
            dtype=Int32,
            default=Int32(0),
            docs="Logical epilogue output subtile index.",
        )
        self.t2r_dequant_scale = TaskLocalVariable(
            dtype=cutlass.Float32,
            default_factory=self._t2r_rmem_default,
            docs="Per-register DeepSeek FP8 dequantization scale.",
        )
        if self.total_num_padded_tokens_tensor is not None:
            self.gTotalNumPaddedTokens = cutlass.make_array_view(
                self.total_num_padded_tokens_tensor
            )

    @cute.jit
    def _t2r_default_values(self):
        if cutlass.const_expr(self.cfg.is_swap_ab):
            swap_t2r_repx = max(1, self.cfg.epi_tile_n // 8)
            t2r_init = cutlass.vector.full([swap_t2r_repx * 4], 0.0, cutlass.Float32)
            t2r_1_init = cutlass.vector.full([swap_t2r_repx * 4], 0.0, cutlass.Float32)
        else:
            epi_t2r_repx = max(1, self.cfg.epi_tile_n // 4)
            t2r_init = cutlass.vector.full([epi_t2r_repx], 0.0, cutlass.Float32)
            t2r_1_init = cutlass.vector.full([1], 0.0, cutlass.Float32)
        return t2r_init, t2r_1_init

    @cute.jit
    def _t2r_rmem_default(self):
        t2r_init, _ = self._t2r_default_values()
        return t2r_init

    @cute.jit
    def _t2r_rmem_1_default(self):
        _, t2r_1_init = self._t2r_default_values()
        return t2r_1_init

    def get_smem_requirements(self):
        return [self._alloc_dsfp8_sf]

    @cute.jit
    def _init_smem(self, context) -> None:
        self.smem_buf = cutlass.Array(
            context.smem_base.data_ptr() + self._alloc_dsfp8_sf.offset,
            dtype=cutlass.Uint8,
            shape=(self._alloc_dsfp8_sf.size_bytes,),
            addrspace=3,
        )

    @cute.jit
    def create_function_variables(self, context=None) -> dict:
        self._init_smem(context)
        self.dsfp8_expert_idx = Int32(0)
        self.dsfp8_token_base = Int32(0)
        self.dsfp8_token_limit = Int32(0)
        self.dsfp8_weight_tile_idx = Int32(0)
        self.dsfp8_num_weight_tiles = Int32(1)
        self.dsfp8_num_k_blocks = Int32(1)
        self.dsfp8_act_token_stride = Int32(1)
        self.dsfp8_last_act_stage_ptr = Int64(0)
        self.dsfp8_last_wt_stage_ptr = Int64(0)
        if cutlass.const_expr(self.cfg.is_swap_ab):
            swap_t2r_repx = max(1, self.cfg.epi_tile_n // 8)
            t2r_rmem = cutlass.vector.full([swap_t2r_repx * 4], 0.0, cutlass.Float32)
            t2r_rmem_1 = cutlass.vector.full([swap_t2r_repx * 4], 0.0, cutlass.Float32)
            dsfp8_acc_rmem = cutlass.vector.full(
                [swap_t2r_repx * 4], 0.0, cutlass.Float32
            )
            dsfp8_acc_rmem_1 = cutlass.vector.full(
                [swap_t2r_repx * 4], 0.0, cutlass.Float32
            )
        else:
            epi_t2r_repx = max(1, self.cfg.epi_tile_n // 4)
            t2r_rmem = cutlass.vector.full([epi_t2r_repx], 0.0, cutlass.Float32)
            t2r_rmem_1 = cutlass.vector.full([1], 0.0, cutlass.Float32)
            dsfp8_acc_rmem = cutlass.vector.full([epi_t2r_repx], 0.0, cutlass.Float32)
            dsfp8_acc_rmem_1 = cutlass.vector.full([1], 0.0, cutlass.Float32)
        self.dsfp8_acc_rmem_state = dsfp8_acc_rmem
        self.dsfp8_acc_rmem_1_state = dsfp8_acc_rmem_1
        return {
            "dsfp8_act_stage_ptr": Int64(0),
            "dsfp8_wt_stage_ptr": Int64(0),
            "t2r_rmem": t2r_rmem,
            "t2r_rmem_1": t2r_rmem_1,
            "dsfp8_acc_rmem": dsfp8_acc_rmem,
            "dsfp8_acc_rmem_1": dsfp8_acc_rmem_1,
            "t2r_output_call_idx": Int32(0),
        }

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_load_state(self, stage_info: StageInfo) -> None:
        self.create_function_variables(stage_info.context)

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_epilogue_state(self, stage_info: StageInfo) -> None:
        self.create_function_variables(stage_info.context)

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def reset_dequant_accumulator(self, stage_info: StageInfo) -> None:
        """Reset the epilogue-local DeepSeek FP8 accumulator for this C tile."""
        if cutlass.const_expr(self.cfg.is_swap_ab):
            swap_t2r_repx = max(1, self.cfg.epi_tile_n // 8)
            self.dsfp8_acc_rmem_state = cutlass.vector.full(
                [swap_t2r_repx * 4], 0.0, cutlass.Float32
            )
            self.dsfp8_acc_rmem_1_state = cutlass.vector.full(
                [swap_t2r_repx * 4], 0.0, cutlass.Float32
            )
        else:
            epi_t2r_repx = max(1, self.cfg.epi_tile_n // 4)
            self.dsfp8_acc_rmem_state = cutlass.vector.full(
                [epi_t2r_repx], 0.0, cutlass.Float32
            )
            self.dsfp8_acc_rmem_1_state = cutlass.vector.full([1], 0.0, cutlass.Float32)

    @cute.jit
    def _local_tile_limit(self, raw_limit, token_tile, tile_rows):
        """Convert TRT-LLM Gen absolute end-row limit to a local row count."""
        local_limit = raw_limit - token_tile * Int32(tile_rows)
        if local_limit < Int32(0):
            local_limit = Int32(0)
        if local_limit > Int32(tile_rows):
            local_limit = Int32(tile_rows)
        return local_limit

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def prepare_sfab_tile(self, stage_info: StageInfo) -> None:
        """Cache per-output-tile metadata used by every K-block load."""
        tile_coord_m, tile_coord_n, _ = stage_info.work_tile.tile_idx
        token_tile = (
            tile_coord_n if cutlass.const_expr(self.cfg.is_swap_ab) else tile_coord_m
        )

        expert_idx = Int32(0)
        if cutlass.const_expr(self.tile_idx_view is not None):
            expert_idx = self.tile_idx_view.load(idx=token_tile, vector_size=1)[0]
        tile_rows = self.cfg.tile_n if self.cfg.is_swap_ab else self.cfg.tile_m
        token_limit = Int32(tile_rows)
        if cutlass.const_expr(self.mn_limit_view is not None):
            token_limit = self._local_tile_limit(
                self.mn_limit_view.load(idx=token_tile, vector_size=1)[0],
                token_tile,
                tile_rows,
            )

        if cutlass.const_expr(self.cfg.is_swap_ab):
            self.dsfp8_token_base = tile_coord_n * Int32(self.cfg.tile_n)
            self.dsfp8_weight_tile_idx = tile_coord_m
            self.dsfp8_num_weight_tiles = (
                self.problem_m + Int32(self.cfg.tile_m - 1)
            ) // Int32(self.cfg.tile_m)
        else:
            self.dsfp8_token_base = tile_coord_m * Int32(self.cfg.tile_m)
            self.dsfp8_weight_tile_idx = tile_coord_n
            self.dsfp8_num_weight_tiles = (
                self.problem_n + Int32(self.cfg.tile_n - 1)
            ) // Int32(self.cfg.tile_n)

        self.dsfp8_expert_idx = expert_idx
        self.dsfp8_token_limit = token_limit
        self.dsfp8_num_k_blocks = (self.problem_k + Int32(127)) // Int32(128)
        if cutlass.const_expr(self.cfg.has_routed_act):
            # Routes expanded rows through ptrRouteMap,
            # then strides activation DQ SFs by params.numTokens.
            self.dsfp8_act_token_stride = self.num_tokens
        elif cutlass.const_expr(self.cfg.is_swap_ab):
            self.dsfp8_act_token_stride = self.problem_n
            if cutlass.const_expr(self.gTotalNumPaddedTokens is not None):
                self.dsfp8_act_token_stride = self.gTotalNumPaddedTokens.load(
                    idx=Int32(0), vector_size=1
                )[0]
        else:
            self.dsfp8_act_token_stride = self.problem_m
            if cutlass.const_expr(self.gTotalNumPaddedTokens is not None):
                self.dsfp8_act_token_stride = self.gTotalNumPaddedTokens.load(
                    idx=Int32(0), vector_size=1
                )[0]

    @producer_work
    @cute.jit
    def load_sfab_tile(self, stage_info: StageInfo) -> None:
        """LDGSTS copy activation SFs and one weight SF into this stage."""
        stage_base = self.smem_buf.data_ptr(
            self.cfg.num_bytes_dsfp8_sfab_per_stage * stage_info.stage_idx
        )

        tidx, _, _ = cute.arch.thread_idx()
        local_tid = tidx - Int32(self.cfg.load_sfab_warp_idx * 32)
        k_block = stage_info.loop_offset

        num_act_load_iters = (self.cfg.num_dsfp8_act_sfs_per_stage + 31) // 32
        for load_iter in cutlass.range_constexpr(num_act_load_iters):
            local_act_col = local_tid + Int32(load_iter * 32)
            act_row = self.dsfp8_token_base + local_act_col
            is_valid_act = (
                (local_tid >= Int32(0))
                & (local_act_col < Int32(self.cfg.num_dsfp8_act_sfs_per_stage))
                & (local_act_col < self.dsfp8_token_limit)
            )
            if is_valid_act:
                if cutlass.const_expr(
                    self.cfg.has_routed_act and self.route_map_view is not None
                ):
                    act_row = self.route_map_view.load(idx=act_row, vector_size=1)[0]
            act_gmem_offset = k_block * self.dsfp8_act_token_stride + act_row
            act_gmem_src = self.sfb_gmem_base.data_ptr() + act_gmem_offset
            act_smem_dst = stage_base + local_act_col * Int32(4)
            # Predicated LDGSTS has no NVVM op form — emit @p cp.async inline.
            prims.inline_ptx_hl(
                "cp.async.ca.shared.global [{$r0}], [{$r1}], 4;",
                read_only_args=[act_smem_dst, act_gmem_src],
                pred=is_valid_act,
            )

        wt_gmem_offset = (
            (self.dsfp8_expert_idx * self.dsfp8_num_weight_tiles)
            + self.dsfp8_weight_tile_idx
        ) * self.dsfp8_num_k_blocks + k_block
        wt_gmem_src = self.sfa_gmem_base.data_ptr() + wt_gmem_offset
        wt_smem_dst = stage_base + Int32(self.cfg.num_dsfp8_act_sfs_per_stage * 4)
        if local_tid == Int32(0):
            prims.cp_async_shared_global(
                wt_smem_dst,
                wt_gmem_src,
                size=4,
                modifier=prims.LoadCacheModifier.CA,
            )

        cute.arch.fence_view_async_shared()

    # ── Consumer side (epilogue) ──────────────────────────────────────────
    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def consume_sfab_tile(self, stage_info: StageInfo) -> None:
        """Cache per-stage SMEM pointers for the dequant FMA in the epilogue.

        The epilogue reads ``dqSfW = *wt_ptr``, forms
        ``dqSfAb = act_ptr[lane] * dqSfW``, and applies it with ffma2.
        """
        stage_base_ptr = self.smem_buf.data_ptr(
            self.cfg.num_bytes_dsfp8_sfab_per_stage * stage_info.stage_idx
        )
        act_ptr = Int64(stage_base_ptr.toint())
        # Weight scalar sits immediately after the activation SF array.
        wt_ptr = act_ptr + Int64(self.cfg.num_dsfp8_act_sfs_per_stage * 4)
        self.dsfp8_last_act_stage_ptr = act_ptr
        self.dsfp8_last_wt_stage_ptr = wt_ptr

    @consumer_work(
        returns=(t2r_rmem, t2r_rmem_1, t2r_output_call_idx, t2r_dequant_scale)
    )
    @cute.jit
    def apply_dequant_to_t2r(
        self,
        stage_info: StageInfo,
        *,
        t2r_rmem,
        t2r_rmem_1,
        t2r_output_call_idx,
    ):
        """Attach the cached DS-FP8 scales to a T2R fragment.

        The separate accumulation work hook consumes the returned fragment and
        scale vector. Keeping the loop-carried accumulator update in that hook
        preserves the task scheduler's SSA boundary while still lowering the
        arithmetic to packed FFMA2 instructions.
        """
        act_smem = cutlass.Array(
            self.dsfp8_last_act_stage_ptr,
            dtype=cutlass.Float32,
            shape=(self.cfg.num_dsfp8_act_sfs_per_stage,),
            addrspace=3,
        )
        wt_smem = cutlass.Array(
            self.dsfp8_last_wt_stage_ptr,
            dtype=cutlass.Float32,
            shape=(self.cfg.num_dsfp8_weight_sfs_per_stage,),
            addrspace=3,
        )
        dq_w = wt_smem.load(idx=0, vector_size=1)[0]

        if cutlass.const_expr(self.cfg.is_swap_ab):
            tidx, _, _ = cute.arch.thread_idx()
            lane_id = tidx % Int32(32)
            warp_idx = cute.arch.warp_idx()
            warp_in_epi = warp_idx - Int32(self.cfg.epilogue_warp_idx)
            warp_in_epi = cute.arch.make_warp_uniform(warp_in_epi)
            warpgroup_idx = warp_in_epi // Int32(4)
            warpgroup_idx = cute.arch.make_warp_uniform(warpgroup_idx)
            warpgroup_count = max(1, self.cfg.num_epilogue_warps // 4)
            n_subtile_offset = t2r_output_call_idx * Int32(
                self.cfg.epi_tile_n * warpgroup_count
            ) + warpgroup_idx * Int32(self.cfg.epi_tile_n)
            if cutlass.const_expr(self.cfg.has_deepseek_fp8_two_epilogue):
                n_subtile_offset = t2r_output_call_idx * Int32(self.cfg.epi_tile_n)
            base_tmem_col = (lane_id % Int32(4)) * Int32(2)

            dequant_scales = [Float32(0.0)] * (max(1, self.cfg.epi_tile_n // 8) * 4)
            for group in cutlass.range_constexpr(max(1, self.cfg.epi_tile_n // 8)):
                reg_base = group * 4
                col_off = Int32(group * 8)
                sf_col = n_subtile_offset + base_tmem_col + col_off
                dq_ab0 = act_smem.load(idx=sf_col, vector_size=1)[0] * dq_w
                dq_ab1 = act_smem.load(idx=sf_col + Int32(1), vector_size=1)[0] * dq_w
                dequant_scales[reg_base] = dq_ab0
                dequant_scales[reg_base + 1] = dq_ab1
                dequant_scales[reg_base + 2] = dq_ab0
                dequant_scales[reg_base + 3] = dq_ab1
            return (
                t2r_rmem,
                t2r_rmem_1,
                t2r_output_call_idx,
                cutlass.Vector.from_elements(
                    tuple(dequant_scales), dtype=cutlass.Float32
                ),
            )

        warp_idx = cute.arch.warp_idx()
        warp_in_epi = warp_idx - Int32(self.cfg.epilogue_warp_idx)
        warp_in_epi = cute.arch.make_warp_uniform(warp_in_epi)
        lane_id = cute.arch.lane_idx()
        row_in_tile = warp_in_epi * Int32(32) + lane_id
        dq_ab = act_smem.load(idx=row_in_tile, vector_size=1)[0] * dq_w
        return (
            t2r_rmem,
            t2r_rmem_1,
            t2r_output_call_idx,
            cutlass.vector.full_like(t2r_rmem, dq_ab.ir_value()),
        )

    @consumer_work(returns=(t2r_rmem, t2r_rmem_1, t2r_output_call_idx))
    @cute.jit
    def accumulate_scaled_t2r(
        self,
        stage_info: StageInfo,
        *,
        t2r_rmem,
        t2r_rmem_1,
        t2r_output_call_idx,
        t2r_dequant_scale,
    ):
        """FFMA2 one DS-FP8 T2R partial into the tile accumulator."""
        if cutlass.const_expr(self.cfg.is_swap_ab):
            t2r_repx = max(1, self.cfg.epi_tile_n // 8) * 4
        else:
            t2r_repx = max(1, self.cfg.epi_tile_n // 4)
        acc0 = self.dsfp8_acc_rmem_state
        vals0 = [Float32(0.0)] * t2r_repx
        for pair in cutlass.range_constexpr(t2r_repx // 2):
            reg_base = pair * 2
            vals0[reg_base], vals0[reg_base + 1] = prims.fma_packed_f32x2(
                (t2r_rmem[reg_base], t2r_rmem[reg_base + 1]),
                (t2r_dequant_scale[reg_base], t2r_dequant_scale[reg_base + 1]),
                (acc0[reg_base], acc0[reg_base + 1]),
                ftz=True,
                rnd="rn",
            )
        if cutlass.const_expr(t2r_repx % 2 != 0):
            vals0[-1] = t2r_rmem[-1] * t2r_dequant_scale[-1] + acc0[-1]
        self.dsfp8_acc_rmem_state = cutlass.Vector.from_elements(
            tuple(vals0), dtype=cutlass.Float32
        )

        if cutlass.const_expr(self.cfg.is_swap_ab):
            acc1 = self.dsfp8_acc_rmem_1_state
            vals1 = [Float32(0.0)] * t2r_repx
            for pair in cutlass.range_constexpr(t2r_repx // 2):
                reg_base = pair * 2
                vals1[reg_base], vals1[reg_base + 1] = prims.fma_packed_f32x2(
                    (t2r_rmem_1[reg_base], t2r_rmem_1[reg_base + 1]),
                    (
                        t2r_dequant_scale[reg_base],
                        t2r_dequant_scale[reg_base + 1],
                    ),
                    (acc1[reg_base], acc1[reg_base + 1]),
                    ftz=True,
                    rnd="rn",
                )
            if cutlass.const_expr(t2r_repx % 2 != 0):
                vals1[-1] = t2r_rmem_1[-1] * t2r_dequant_scale[-1] + acc1[-1]
            self.dsfp8_acc_rmem_1_state = cutlass.Vector.from_elements(
                tuple(vals1), dtype=cutlass.Float32
            )
        else:
            self.dsfp8_acc_rmem_1_state = self.dsfp8_acc_rmem_1_state + t2r_rmem_1
        return (
            self.dsfp8_acc_rmem_state,
            self.dsfp8_acc_rmem_1_state,
            t2r_output_call_idx,
        )
