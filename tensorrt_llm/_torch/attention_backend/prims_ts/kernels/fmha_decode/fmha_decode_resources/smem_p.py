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

"""``SmemPResource`` — P operand staging for BMM2.

Producer (Softmax): converts S in registers to P and publishes per-lane local
sums back through ``TmemSResource``. Keeps Q64/Q128 overlays P on consumed S
columns in TMEM; Swaps retains the SMEM operand layout. Consumer (MmaTask)
publishes the corresponding TMEM address or SMEM descriptor.
"""

from dataclasses import dataclass
from typing import ClassVar

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32
from cutlass.experimental import primitives as prims

from cutlass.experimental.task_scheduling.memory import (
    ResourceContext,
    SmemAllocation,
    TmemAllocation,
)
from cutlass.experimental.task_scheduling.resources import (
    WorkAttr,
    StageInfo,
    TaskLocalVariable,
    consumer_work,
    producer_work,
)

from ..fmha_decode_config import FmhaDecodeConfig
from ...placeholder_helpers import _placeholder_smem_array
from .helpers_common import (
    Constexpr,
    DecodeGenResourceBase,
    ResourceVars,
    fadd2,
    ffma2,
    fmul2,
    _TASK_CACHE_LANE_IDX,
    _TASK_CACHE_WARP_GRP_THREAD_IDX,
    _TASK_CACHE_WARP_IDX,
    _decode_gen_task_cache,
    _fp8_log2_quant_scale,
    _keeps_col_base,
    _keeps_row_idx,
    _keeps_tcgen05_st,
    _named_barrier_arrive,
    _neg_max_f32,
    _pack_float2_to_bf16,
    _pack_float2_to_fp16,
)
from .helpers_output import (
    _keeps_p_smem_block_offset_bytes,
    _p_stsm_smem_offset_bytes,
    _store_transposed_smem8b,
    _store_transposed_smem8b_x2,
    _store_transposed_smem8b_x4,
)
from .helpers_softmax import (
    _compute_fp8_p_regs_and_local_sums,
    _compute_fp8_p_regs_and_local_sums_dense,
    _compute_p_values_and_local_sums_dense,
    _pack_float4_to_fp8_e4m3,
    _pack_float4_to_fp8_e4m3_inline,
)
from .tmem_s import TmemSResource


@dataclass(kw_only=True)
class SmemPResource(DecodeGenResourceBase):
    """P operand resource consumed by BMM2.

    Softmax producers convert S to P, store it in the profile's TMEM or SMEM
    layout, and publish local sums back to TmemS.
    """

    _task_local_specs: ClassVar[tuple[tuple, ...]] = (
        (
            "p_desc_0_slot",
            prims.Tcgen05SmemDesc,
            prims.Tcgen05SmemDesc(0),
            "P descriptor for VP MMA call 0.",
        ),
        (
            "p_desc_1_slot",
            prims.Tcgen05SmemDesc,
            prims.Tcgen05SmemDesc(0),
            "P descriptor for VP MMA call 1.",
        ),
        (
            "p_tmem_addr_0_slot",
            Int32,
            Int32(0),
            "TMEM P address for VP MMA call 0.",
        ),
        (
            "p_tmem_addr_1_slot",
            Int32,
            Int32(0),
            "TMEM P address for VP MMA call 1.",
        ),
    )
    inst_id: Constexpr[int] = 0
    cfg: Constexpr[FmhaDecodeConfig] = None
    scale_softmax_log2: Float32 = None
    use_variable_seqlens_kv: Constexpr[bool] = False
    tmem_s_ref: Constexpr[TmemSResource] = None
    _alloc: Constexpr[SmemAllocation | None] = None
    _tmem_alloc: Constexpr[TmemAllocation | None] = None
    _tmem_base_addr: Int32 = None
    _smem_base_p: cutlass.Array = None
    _smem_base_p_i32: cutlass.Array = None
    p_desc_0_slot: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    p_desc_1_slot: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    p_tmem_addr_0_slot: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    p_tmem_addr_1_slot: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()

    def _init_placeholder_state(self) -> None:
        """Create placeholder P storage state."""
        self._tmem_base_addr = Int32(0)
        self._smem_base_p = _placeholder_smem_array(
            self.cfg.q_dtype,
            self.cfg.smem_p_tile_bytes // self.cfg.q_dtype_bytes,
        )
        self._smem_base_p_i32 = _placeholder_smem_array(
            Int32, self.cfg.smem_p_tile_bytes // 4
        )

    def get_smem_requirements(self) -> list[SmemAllocation]:
        """Allocate the SMEM P tile used by non-TMEM profiles."""
        if self.cfg.uses_tmem_p:
            return []
        if self._alloc is None:
            self._alloc = SmemAllocation(
                name=f"{self.name}",
                size_bytes=self.cfg.smem_p_tile_bytes,
                alignment=self.cfg.stensor_align,
            )
        return [self._alloc]

    def get_tmem_requirements(self) -> list[TmemAllocation]:
        """Allocate the TMEM columns occupied by the Keeps P operand."""
        if not self.cfg.uses_tmem_p:
            return []
        if self._tmem_alloc is None:
            num_columns = self.cfg.tmem_p_cols
            if self.cfg.uses_two_inst_tmem_p:
                num_columns = self.cfg.tile_size_kv * self.cfg.q_dtype_bytes // 4
            self._tmem_alloc = TmemAllocation(
                name=f"{self.name}_tmem",
                num_columns=num_columns,
            )
        return [self._tmem_alloc]

    @cute.jit
    def _create_initial_task_locals(
        self, context: ResourceContext | None = None
    ) -> ResourceVars:
        """Bind P storage and initialize operand task locals."""
        if cutlass.const_expr(
            not self.cfg.uses_tmem_p
            and context is not None
            and context.smem_base is not None
        ):
            # P is materialized in SMEM because BMM2 consumes it as a tcgen05
            # SMEM operand.
            smem_base_ptr = context.smem_base.data_ptr() + self._alloc.offset
            self._smem_base_p = cutlass.Array(
                smem_base_ptr,
                dtype=self.cfg.q_dtype,
                shape=(self.cfg.smem_p_tile_bytes // self.cfg.q_dtype_bytes,),
                addrspace=3,
            )
            self._smem_base_p_i32 = cutlass.Array(
                smem_base_ptr,
                dtype=Int32,
                shape=(self.cfg.smem_p_tile_bytes // 4,),
                addrspace=3,
            )
        if cutlass.const_expr(
            self.cfg.uses_tmem_p
            and context is not None
            and context.tmem_ptr_i32 is not None
        ):
            self._tmem_base_addr = context.tmem_ptr_i32.load()
        return {
            "p_desc_0": cutlass.Int64(0),
            "p_desc_1": cutlass.Int64(0),
            "p_tmem_addr_0": Int32(0),
            "p_tmem_addr_1": Int32(0),
        }

    @cute.jit
    def _create_work_tile_task_locals(
        self, context: ResourceContext | None = None
    ) -> ResourceVars:
        """Provide P operand slots for one work tile."""
        _ = context
        return {
            "p_desc_0": cutlass.Int64(0),
            "p_desc_1": cutlass.Int64(0),
            "p_tmem_addr_0": Int32(0),
            "p_tmem_addr_1": Int32(0),
        }

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_compute_state(self, stage_info: StageInfo) -> None:
        """Initialize producer-side P registers and local sums."""
        # ProdAuxWork: bind the SMEM P tile and reset producer-local P/sum
        # state before the softmax producer starts writing this work tile.
        self._create_initial_task_locals(stage_info.context)

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_descriptor_state(self, stage_info: StageInfo) -> None:
        """Initialize consumer-side P operand placeholders."""
        # ConsAuxWork: mirror the P storage binding on the BMM2 side so operand
        # work can publish a valid descriptor or TMEM address for this tile.
        self._create_initial_task_locals(stage_info.context)

    @cute.jit
    def _compute_keeps_p(
        self,
        stage_info: StageInfo,
        *,
        new_max_arr: cutlass.Array,
        s_arr: cutlass.Array,
    ) -> None:
        """Materialize one row-major Keeps probability fragment.

        TQ128 gives each warp-group thread a complete 128-column row. TQ64
        gives paired lanes the low/high 64-column halves of one row. Each lane
        writes disjoint packed blocks into the TMEM or SMEM layout consumed by
        BMM2.
        """
        cfg = self.cfg
        task_cache = _decode_gen_task_cache(stage_info)
        warp_grp_thread_idx = task_cache[_TASK_CACHE_WARP_GRP_THREAD_IDX]
        lane_idx = task_cache[_TASK_CACHE_LANE_IDX]
        num_s_regs = cfg.num_s_regs_per_thread
        vector_elements = cfg.keeps_p_smem_vector_elements
        num_vector_blocks = num_s_regs // vector_elements
        row_idx = _keeps_row_idx(cfg, warp_grp_thread_idx)
        col_base = _keeps_col_base(cfg, lane_idx, num_s_regs)
        p_tmem_stage_base = Int32(0)
        if cutlass.const_expr(cfg.uses_tmem_p):
            p_tmem_stage_base = (
                self._tmem_base_addr
                + Int32(self._tmem_alloc.offset)
                + stage_info.stage_idx * cfg.tmem_s_cols
            )

        new_max = new_max_arr[0]
        safe_new_max = new_max
        if safe_new_max == _neg_max_f32():
            safe_new_max = Float32(0.0)
        neg_scaled_max = -self.scale_softmax_log2 * safe_new_max
        if cutlass.const_expr(cfg.use_fp8_qkv):
            neg_scaled_max += _fp8_log2_quant_scale()

        # Preserve four independent modulo-4 sum chains as two packed pairs,
        # without keeping a second 16-value P array live beside the S row.
        local_sum_pair_01 = (Float32(0.0), Float32(0.0))
        local_sum_pair_23 = (Float32(0.0), Float32(0.0))

        # Each vector block is exactly 16 bytes after conversion. Compute and
        # pack adjacent pairs directly into their final register payload.
        packed_p_regs = cfg.num_packed_p_regs if cfg.uses_two_inst_tmem_p else 4
        packed_p = cutlass.Array(Int32, packed_p_regs, space=cutlass.AddressSpace.rmem)
        for block_idx in cutlass.range_constexpr(num_vector_blocks):
            s_base = block_idx * vector_elements
            packed_base = block_idx * 4 if cfg.uses_two_inst_tmem_p else 0
            if cutlass.const_expr(cfg.use_fp8_qkv):
                for packed_idx in cutlass.range_constexpr(4):
                    val_base = packed_idx * 4
                    scaled_pair_01 = ffma2(
                        (
                            s_arr[s_base + val_base],
                            s_arr[s_base + val_base + 1],
                        ),
                        (self.scale_softmax_log2, self.scale_softmax_log2),
                        (neg_scaled_max, neg_scaled_max),
                    )
                    p_pair_01 = (
                        cute.math.exp2(scaled_pair_01[0], fastmath=True),
                        cute.math.exp2(scaled_pair_01[1], fastmath=True),
                    )
                    local_sum_pair_01 = fadd2(local_sum_pair_01, p_pair_01)
                    scaled_pair_23 = ffma2(
                        (
                            s_arr[s_base + val_base + 2],
                            s_arr[s_base + val_base + 3],
                        ),
                        (self.scale_softmax_log2, self.scale_softmax_log2),
                        (neg_scaled_max, neg_scaled_max),
                    )
                    p_pair_23 = (
                        cute.math.exp2(scaled_pair_23[0], fastmath=True),
                        cute.math.exp2(scaled_pair_23[1], fastmath=True),
                    )
                    local_sum_pair_23 = fadd2(local_sum_pair_23, p_pair_23)
                    packed_p[packed_base + packed_idx] = (
                        _pack_float4_to_fp8_e4m3_inline(
                            p_pair_01[0],
                            p_pair_01[1],
                            p_pair_23[0],
                            p_pair_23[1],
                        )
                    )
            else:
                for packed_idx in cutlass.range_constexpr(4):
                    val_base = packed_idx * 2
                    scaled_pair = ffma2(
                        (
                            s_arr[s_base + val_base],
                            s_arr[s_base + val_base + 1],
                        ),
                        (self.scale_softmax_log2, self.scale_softmax_log2),
                        (neg_scaled_max, neg_scaled_max),
                    )
                    p_pair = (
                        cute.math.exp2(scaled_pair[0], fastmath=True),
                        cute.math.exp2(scaled_pair[1], fastmath=True),
                    )
                    if cutlass.const_expr(packed_idx % 2 == 0):
                        local_sum_pair_01 = fadd2(local_sum_pair_01, p_pair)
                    else:
                        local_sum_pair_23 = fadd2(local_sum_pair_23, p_pair)
                    if cutlass.const_expr(cfg.use_bf16_qkv):
                        packed_p[packed_base + packed_idx] = _pack_float2_to_bf16(
                            p_pair[0], p_pair[1]
                        )
                    else:
                        packed_p[packed_base + packed_idx] = _pack_float2_to_fp16(
                            p_pair[0], p_pair[1]
                        )

            if cutlass.const_expr(cfg.uses_two_inst_tmem_p):
                # Retain the complete packed row and publish it once below.
                pass
            elif cutlass.const_expr(cfg.uses_tmem_p):
                # Each register packs two 16-bit P values. The q64 TMEM store shape
                # maps paired half-warps onto the low/high 32-column halves of
                # the 64-column UInt32 P tile.
                p_tmem_addr = p_tmem_stage_base + Int32(block_idx * 4)
                _keeps_tcgen05_st(
                    cfg,
                    prims.make_tmem_ptr(p_tmem_addr, Int32),
                    packed_p.data_ptr().load(count=4, alignment=4),
                    offset=cfg.num_packed_p_regs,
                )
            else:
                logical_col = col_base + Int32(s_base)
                smem_offset_bytes = _keeps_p_smem_block_offset_bytes(
                    cfg, row_idx, logical_col
                )
                smem_dst = self._smem_base_p_i32.subview(
                    smem_offset_bytes >> Int32(2)
                ).data_ptr()
                smem_dst.store(
                    packed_p.data_ptr().load(count=4, alignment=4), alignment=16
                )
        if cutlass.const_expr(cfg.uses_two_inst_tmem_p):
            # FP8 publishes Q64/Q128 with one x16/x32 TMEM store. FP16/BF16 uses x16
            # slices to limit Softmax register pressure, producing two/four
            # stores. Every slice targets this instance's consumed S region.
            assert cfg.num_packed_p_regs in (16, 32, 64)
            regs_per_store = cfg.num_packed_p_regs if cfg.use_fp8_qkv else 16
            assert cfg.num_packed_p_regs % regs_per_store == 0
            for store_idx in cutlass.range_constexpr(
                cfg.num_packed_p_regs // regs_per_store
            ):
                packed_offset = store_idx * regs_per_store
                _keeps_tcgen05_st(
                    cfg,
                    prims.make_tmem_ptr(
                        p_tmem_stage_base + Int32(packed_offset), Int32
                    ),
                    (packed_p.data_ptr() + packed_offset).load(
                        count=regs_per_store, alignment=4
                    ),
                    # Q64's 16x32bx2 mapping uses this as the low/high-half
                    # split: 16 columns for FP8 and 32 for FP16/BF16.
                    offset=cfg.num_packed_p_regs,
                )
            if cutlass.const_expr(cfg.ordered_softmax_early_release):
                # Hand the baton over as soon as this group's TMEM store has
                # issued: the partner's exp2/pack/TMEM store touch only its own
                # registers and TMEM region, so it need not wait for this
                # store to drain, the async fence, or the pipeline commit.
                # The exp2 phases stay serialized (shared MUFU), but the
                # store-drain + commit tail overlaps the partner's wakeup.
                _named_barrier_arrive(
                    cfg.resolved_softmax_order_barrier_threads,
                    barrier_id=cfg.softmax_order_barrier_id + 1 - self.inst_id,
                )
        local_sum_pair0 = fadd2(local_sum_pair_01, local_sum_pair_23)
        local_sum = local_sum_pair0[0] + local_sum_pair0[1]
        self.tmem_s_ref.store_p_local_sum(0, local_sum)

        # Publish the selected memory view before the task-level P pipeline
        # exposes this stage to BMM2.
        if cutlass.const_expr(cfg.uses_tmem_p):
            prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
            cute.arch.fence_view_async_tmem_store()
            if cutlass.const_expr(cfg.uses_staged_one_inst_tmem_p):
                # Synchronize the D256 producer warp group after its TMEM stores
                # are visible.
                prims.barrier_cta_sync(4 + self.inst_id, thread_count=128)
        else:
            # Each producer thread orders its own SMEM P stores with an
            # async-proxy fence before its own AsyncUmma producer-commit
            # mbarrier arrive. The full barrier counts all 128 softmax
            # threads, so the commit itself is the warp-group visibility
            # point and no extra named barrier is needed here.
            cute.arch.fence_view_async_shared()

    @producer_work
    @cute.jit
    def compute_p(
        self,
        stage_info: StageInfo,
        *,
        new_max_arr: cutlass.Array,
        s_arr: cutlass.Array,
    ) -> None:
        """Compute P from S, stage its BMM2 operand, and publish local sums."""
        cfg = self.cfg
        if cutlass.const_expr(cfg.use_keeps_mma_ab):
            self._compute_keeps_p(
                stage_info,
                new_max_arr=new_max_arr,
                s_arr=s_arr,
            )
            return
        # ProdWork: transform the softmax S registers into the P operand layout
        # expected by BMM2, while recording the per-scale local sums consumed by
        # the denominator update.
        # Decode the scheduler cache once so every store path uses the same
        # warp/lane ownership for SMEM offsets and STSM swizzles.
        task_cache = _decode_gen_task_cache(stage_info)
        warp_grp_thread_idx = task_cache[_TASK_CACHE_WARP_GRP_THREAD_IDX]
        if cutlass.const_expr(cfg.tile_size_q == 32 and cfg.use_fp8_qkv):
            # Tile-Q=32 FP8 fast path: compute E4M3 P registers in the
            # same order consumed by the STSM helper, while also capturing
            # one local denominator sum per softmax scale group.
            packed_p = cutlass.Array(
                Int32, cfg.num_packed_p_regs, space=cutlass.AddressSpace.rmem
            )
            local_sum = cutlass.Array(
                Float32,
                cfg.num_softmax_scale_groups,
                space=cutlass.AddressSpace.rmem,
            )
            if cutlass.const_expr(
                not self.use_variable_seqlens_kv
                and cfg.total_kv_tiles > 0
                and (cfg.total_kv_tiles % cfg.num_insts_kv) == 0
                and cfg.q_tiles_are_full
            ):
                # Dense full tiles have no tail/window masking. Use the
                # straight-line helper to keep this path compact.
                for scale_pair_idx in cutlass.range_constexpr(4):
                    scale_base = scale_pair_idx * 2
                    q32_s_base = scale_pair_idx * 4
                    p_lo, p_hi, sum_lo, sum_hi = (
                        _compute_fp8_p_regs_and_local_sums_dense(
                            self.scale_softmax_log2,
                            new_max_arr[scale_base],
                            new_max_arr[scale_base + 1],
                            s_arr[q32_s_base],
                            s_arr[q32_s_base + 1],
                            s_arr[q32_s_base + 2],
                            s_arr[q32_s_base + 3],
                            s_arr[q32_s_base + 16],
                            s_arr[q32_s_base + 17],
                            s_arr[q32_s_base + 18],
                            s_arr[q32_s_base + 19],
                        )
                    )
                    packed_p[scale_pair_idx] = p_lo
                    packed_p[scale_pair_idx + 4] = p_hi
                    local_sum[scale_base] = sum_lo
                    local_sum[scale_base + 1] = sum_hi
            else:
                # General path handles masked S values from variable
                # seqlens, odd tail waves, sliding window, or split-KV.
                for scale_pair_idx in cutlass.range_constexpr(4):
                    scale_base = scale_pair_idx * 2
                    q32_s_base = scale_pair_idx * 4
                    p_lo, p_hi, sum_lo, sum_hi = _compute_fp8_p_regs_and_local_sums(
                        self.scale_softmax_log2,
                        new_max_arr[scale_base],
                        new_max_arr[scale_base + 1],
                        s_arr[q32_s_base],
                        s_arr[q32_s_base + 1],
                        s_arr[q32_s_base + 2],
                        s_arr[q32_s_base + 3],
                        s_arr[q32_s_base + 16],
                        s_arr[q32_s_base + 17],
                        s_arr[q32_s_base + 18],
                        s_arr[q32_s_base + 19],
                    )
                    packed_p[scale_pair_idx] = p_lo
                    packed_p[scale_pair_idx + 4] = p_hi
                    local_sum[scale_base] = sum_lo
                    local_sum[scale_base + 1] = sum_hi
            # Publish the local denominator contribution before committing P so
            # TmemS can update the online-softmax sum after P materialization.
            for scale_idx in cutlass.range_constexpr(cfg.num_softmax_scale_groups):
                self.tmem_s_ref.store_p_local_sum(scale_idx, local_sum[scale_idx])
            # Store the low and high K halves separately. The x4 helper writes
            # transposed E4M3 bytes into the SMEM layout expected by BMM2.
            _store_transposed_smem8b_x4(
                self._smem_base_p_i32,
                packed_p[0],
                packed_p[1],
                packed_p[2],
                packed_p[3],
                warp_grp_thread_idx,
                cfg.tile_size_q,
                cfg.tile_size_kv,
            )
            _store_transposed_smem8b_x4(
                self._smem_base_p_i32,
                packed_p[4],
                packed_p[5],
                packed_p[6],
                packed_p[7],
                warp_grp_thread_idx,
                cfg.tile_size_q,
                cfg.tile_size_kv,
                1,
            )
            # The P producer writes SMEM directly with STSM helpers. Fence
            # and synchronize the warpgroup before the TS pipeline commits
            # the stage to the BMM2 consumer.
            cute.arch.fence_view_async_shared()
            prims.barrier_cta_sync(4 + self.inst_id, thread_count=128)
            return

        if cutlass.const_expr(cfg.tile_size_q == 16 and cfg.use_fp8_qkv):
            # Tile-Q=16 FP8 fast path: each helper call handles two softmax
            # scale groups and returns the low/high K halves already packed for
            # STSM. This avoids keeping all 16 FP32 P values live and packing
            # them in a separate pass.
            packed_p = cutlass.Array(
                Int32, cfg.num_packed_p_regs, space=cutlass.AddressSpace.rmem
            )
            local_sum = cutlass.Array(
                Float32,
                cfg.num_softmax_scale_groups,
                space=cutlass.AddressSpace.rmem,
            )
            for scale_pair_idx in cutlass.range_constexpr(2):
                scale_base = scale_pair_idx * 2
                s_base = scale_pair_idx * 4
                if cutlass.const_expr(
                    not self.use_variable_seqlens_kv
                    and cfg.total_kv_tiles > 0
                    and (cfg.total_kv_tiles % cfg.num_insts_kv) == 0
                    and cfg.q_tiles_are_full
                ):
                    p_lo, p_hi, sum_lo, sum_hi = (
                        _compute_fp8_p_regs_and_local_sums_dense(
                            self.scale_softmax_log2,
                            new_max_arr[scale_base],
                            new_max_arr[scale_base + 1],
                            s_arr[s_base],
                            s_arr[s_base + 1],
                            s_arr[s_base + 2],
                            s_arr[s_base + 3],
                            s_arr[s_base + 8],
                            s_arr[s_base + 9],
                            s_arr[s_base + 10],
                            s_arr[s_base + 11],
                        )
                    )
                else:
                    p_lo, p_hi, sum_lo, sum_hi = _compute_fp8_p_regs_and_local_sums(
                        self.scale_softmax_log2,
                        new_max_arr[scale_base],
                        new_max_arr[scale_base + 1],
                        s_arr[s_base],
                        s_arr[s_base + 1],
                        s_arr[s_base + 2],
                        s_arr[s_base + 3],
                        s_arr[s_base + 8],
                        s_arr[s_base + 9],
                        s_arr[s_base + 10],
                        s_arr[s_base + 11],
                    )
                packed_p[scale_pair_idx] = p_lo
                packed_p[scale_pair_idx + 2] = p_hi
                local_sum[scale_base] = sum_lo
                local_sum[scale_base + 1] = sum_hi

            for scale_idx in cutlass.range_constexpr(cfg.num_softmax_scale_groups):
                self.tmem_s_ref.store_p_local_sum(scale_idx, local_sum[scale_idx])
            _store_transposed_smem8b(
                self._smem_base_p_i32,
                packed_p.data_ptr().load(count=cfg.num_packed_p_regs, alignment=4),
                warp_grp_thread_idx,
                cfg.tile_size_q,
                cfg.tile_size_kv,
                cfg.num_packed_p_regs,
            )
            cute.arch.fence_view_async_shared()
            prims.barrier_cta_sync(4 + self.inst_id, thread_count=128)
            return

        if cutlass.const_expr(cfg.tile_size_q in (16, 32)):
            # Generic tile-Q 16/32 path: compute P scalars for each
            # softmax scale group, accumulate local sums, then pack/store
            # to the BMM2 SMEM layout.
            q_repeats = max(cfg.tile_size_q // 8, 1)
            num_s_regs = cfg.num_s_regs_per_thread
            num_scale_groups = cfg.num_softmax_scale_groups
            p_vals = cutlass.Array(Float32, num_s_regs, space=cutlass.AddressSpace.rmem)
            local_sums = cutlass.Array(
                Float32, num_scale_groups, space=cutlass.AddressSpace.rmem
            )
            for idx in cutlass.range_constexpr(num_s_regs):
                p_vals[idx] = Float32(0.0)
            for idx in cutlass.range_constexpr(num_scale_groups):
                local_sums[idx] = Float32(0.0)

            for scale_idx in cutlass.range_constexpr(num_scale_groups):
                # Convert each softmax scale group from S to P. Masked rows have
                # new_max == -inf and keep their initialized zero P/local_sum.
                new_max = new_max_arr[scale_idx]
                safe_new_max = new_max
                if safe_new_max == _neg_max_f32():
                    safe_new_max = Float32(0.0)
                neg_scaled_max = -self.scale_softmax_log2 * safe_new_max
                if cutlass.const_expr(cfg.use_fp8_qkv):
                    neg_scaled_max += _fp8_log2_quant_scale()
                if new_max != _neg_max_f32():
                    repeat_idx = scale_idx // 2
                    pair_idx = scale_idx % 2
                    generic_s_base = repeat_idx * 4 + pair_idx
                    for k_pair_idx in cutlass.range_constexpr(4):
                        if cutlass.const_expr(k_pair_idx < 2):
                            s_idx = generic_s_base + k_pair_idx * 2
                        else:
                            s_idx = (
                                generic_s_base + q_repeats * 4 + (k_pair_idx - 2) * 2
                            )
                        p_val = cute.math.exp2(
                            s_arr[s_idx] * self.scale_softmax_log2 + neg_scaled_max,
                            fastmath=True,
                        )
                        p_vals[s_idx] = p_val
                        local_sums[scale_idx] += p_val
            # Hand off denominator contributions through TmemS. P remains a pure
            # MMA operand in SMEM; sums are not reloaded from the P tile.
            for scale_idx in cutlass.range_constexpr(num_scale_groups):
                self.tmem_s_ref.store_p_local_sum(scale_idx, local_sums[scale_idx])

            if cutlass.const_expr(cfg.use_fp8_qkv):
                # FP8 P is packed four values per register and stored with
                # transposed 8-bit helpers so BMM2 sees the tcgen05 layout.
                packed_p = cutlass.Array(
                    Int32, cfg.num_packed_p_regs, space=cutlass.AddressSpace.rmem
                )
                for packed_idx in cutlass.range_constexpr(cfg.num_packed_p_regs):
                    val_base = packed_idx * 4
                    packed_p[packed_idx] = _pack_float4_to_fp8_e4m3(
                        p_vals[val_base],
                        p_vals[val_base + 1],
                        p_vals[val_base + 2],
                        p_vals[val_base + 3],
                    )
                _store_transposed_smem8b(
                    self._smem_base_p_i32,
                    packed_p.data_ptr().load(count=cfg.num_packed_p_regs, alignment=4),
                    warp_grp_thread_idx,
                    cfg.tile_size_q,
                    cfg.tile_size_kv,
                    cfg.num_packed_p_regs,
                )
                # Inline byte stores need an explicit CTA barrier before the
                # pipeline stage can be observed by the BMM2 consumer.
                cute.arch.fence_view_async_shared()
                prims.barrier_cta_sync(4 + self.inst_id, thread_count=128)
                return

            # FP16/BF16 P uses stmatrix stores. Each stmatrix group writes
            # one 8x8 fragment to the swizzled SMEM tile consumed by UMMA.
            regs_p = cutlass.Array(
                Int32,
                cfg.num_packed_p_regs,
                space=cutlass.AddressSpace.rmem,
            )
            for pair_idx in cutlass.range_constexpr(cfg.num_packed_p_regs):
                val_base = pair_idx * 2
                if cutlass.const_expr(cfg.use_bf16_qkv):
                    regs_p[pair_idx] = _pack_float2_to_bf16(
                        p_vals[val_base], p_vals[val_base + 1]
                    )
                else:
                    regs_p[pair_idx] = _pack_float2_to_fp16(
                        p_vals[val_base], p_vals[val_base + 1]
                    )
            warp_idx = task_cache[_TASK_CACHE_WARP_IDX]
            lane_idx = task_cache[_TASK_CACHE_LANE_IDX]
            for stsm_group_idx in cutlass.range_constexpr(cfg.num_packed_p_regs // 4):
                smem_offset_bytes = _p_stsm_smem_offset_bytes(
                    warp_idx, lane_idx, stsm_group_idx, cfg.tile_size_q
                )
                smem_dst = (
                    self._smem_base_p_i32.subview((smem_offset_bytes >> 2))
                ).data_ptr()
                prims.stmatrix(
                    smem_dst,
                    (regs_p.data_ptr() + stsm_group_idx * 4).load(count=4, alignment=4),
                    prims.MMALayout.COL,
                    shape=prims.StoreShape.M8N8,
                )
            cute.arch.fence_view_async_shared()
            return

        if cutlass.const_expr(cfg.use_fp8_qkv):
            # Tile-Q=8 FP8 path: compute packed P and local sums directly
            # from the eight S registers owned by this lane.
            packed_p = cutlass.Array(
                Int32, cfg.num_packed_p_regs, space=cutlass.AddressSpace.rmem
            )
            local_sum = cutlass.Array(
                Float32,
                cfg.num_softmax_scale_groups,
                space=cutlass.AddressSpace.rmem,
            )
            if cutlass.const_expr(
                not self.use_variable_seqlens_kv
                and cfg.total_kv_tiles > 0
                and (cfg.total_kv_tiles % cfg.num_insts_kv) == 0
            ):
                # Dense full-tile FP8 can produce packed P and sums in one
                # straight-line helper because no S entry is masked.
                packed_p[0], packed_p[1], local_sum[0], local_sum[1] = (
                    _compute_fp8_p_regs_and_local_sums_dense(
                        self.scale_softmax_log2,
                        new_max_arr[0],
                        new_max_arr[1],
                        s_arr[0],
                        s_arr[1],
                        s_arr[2],
                        s_arr[3],
                        s_arr[4],
                        s_arr[5],
                        s_arr[6],
                        s_arr[7],
                    )
                )
            else:
                # General FP8 path preserves masks from softmax by letting the
                # helper suppress entries whose S value is -inf.
                packed_p[0], packed_p[1], local_sum[0], local_sum[1] = (
                    _compute_fp8_p_regs_and_local_sums(
                        self.scale_softmax_log2,
                        new_max_arr[0],
                        new_max_arr[1],
                        s_arr[0],
                        s_arr[1],
                        s_arr[2],
                        s_arr[3],
                        s_arr[4],
                        s_arr[5],
                        s_arr[6],
                        s_arr[7],
                    )
                )
            # Publish sums through TmemS; packed E4M3 P bytes use the
            # transposed SMEM tile consumed by BMM2.
            for scale_idx in cutlass.range_constexpr(cfg.num_softmax_scale_groups):
                self.tmem_s_ref.store_p_local_sum(scale_idx, local_sum[scale_idx])
            _store_transposed_smem8b_x2(
                self._smem_base_p_i32,
                packed_p[0],
                packed_p[1],
                warp_grp_thread_idx,
                cfg.tile_size_q,
                cfg.tile_size_kv,
            )
        else:
            # Tile-Q=8 16-bit path: compute P scalars, accumulate local
            # sums, pack to 16-bit, and store a matrix tile into SMEM.
            local_sum = cutlass.Array(
                Float32,
                cfg.num_softmax_scale_groups,
                space=cutlass.AddressSpace.rmem,
            )
            p_vals = cutlass.Array(Float32, 8, space=cutlass.AddressSpace.rmem)
            for scale_idx in cutlass.range_constexpr(cfg.num_softmax_scale_groups):
                local_sum[scale_idx] = Float32(0.0)
            for p_idx in cutlass.range_constexpr(8):
                p_vals[p_idx] = Float32(0.0)
            if cutlass.const_expr(
                not self.use_variable_seqlens_kv
                and not cfg.use_split_kv
                and cfg.total_kv_tiles > 0
                and (cfg.total_kv_tiles % cfg.num_insts_kv) == 0
            ):
                # Dense full-tile cases can compute P and both local sums with
                # the straight-line helper because no mask can suppress entries.
                p_result = _compute_p_values_and_local_sums_dense(
                    self.scale_softmax_log2,
                    new_max_arr[0],
                    new_max_arr[1],
                    s_arr[0],
                    s_arr[1],
                    s_arr[2],
                    s_arr[3],
                    s_arr[4],
                    s_arr[5],
                    s_arr[6],
                    s_arr[7],
                )
                for p_idx in cutlass.range_constexpr(8):
                    p_vals[p_idx] = p_result[p_idx]
                for scale_idx in cutlass.range_constexpr(cfg.num_softmax_scale_groups):
                    local_sum[scale_idx] = p_result[8 + scale_idx]
            else:
                # General tile-Q=8 path preserves masked S entries as zero P
                # contribution by skipping groups whose new max stayed -inf.
                for scale_idx in cutlass.range_constexpr(cfg.num_softmax_scale_groups):
                    new_max = new_max_arr[scale_idx]
                    if new_max != _neg_max_f32():
                        for pair_idx in cutlass.range_constexpr(2):
                            p_base = scale_idx + pair_idx * 2
                            scaled_pair = fmul2(
                                (
                                    self.scale_softmax_log2,
                                    self.scale_softmax_log2,
                                ),
                                fadd2(
                                    (s_arr[p_base], s_arr[p_base + 4]),
                                    (-new_max, -new_max),
                                ),
                            )
                            p_pair = (
                                cute.math.exp2(scaled_pair[0], fastmath=True),
                                cute.math.exp2(scaled_pair[1], fastmath=True),
                            )
                            p_vals[p_base] = p_pair[0]
                            p_vals[p_base + 4] = p_pair[1]
                            local_sum[scale_idx] += p_pair[0]
                            local_sum[scale_idx] += p_pair[1]
            # Pack the P scalars to match the dtype consumed by BMM2.
            regs_p = cutlass.Array(
                Int32, cfg.num_packed_p_regs, space=cutlass.AddressSpace.rmem
            )
            if cutlass.const_expr(cfg.use_bf16_qkv):
                for reg_idx in cutlass.range_constexpr(cfg.num_packed_p_regs):
                    val_base = reg_idx * 2
                    regs_p[reg_idx] = _pack_float2_to_bf16(
                        p_vals[val_base], p_vals[val_base + 1]
                    )
            else:
                for reg_idx in cutlass.range_constexpr(cfg.num_packed_p_regs):
                    val_base = reg_idx * 2
                    regs_p[reg_idx] = _pack_float2_to_fp16(
                        p_vals[val_base], p_vals[val_base + 1]
                    )
            # Publish the denominator contribution after P has been computed,
            # before the SMEM fence exposes the P tile to the downstream MMA.
            for scale_idx in cutlass.range_constexpr(cfg.num_softmax_scale_groups):
                self.tmem_s_ref.store_p_local_sum(scale_idx, local_sum[scale_idx])
            warp_idx = task_cache[_TASK_CACHE_WARP_IDX]
            lane_idx = task_cache[_TASK_CACHE_LANE_IDX]
            # Compute the stmatrix destination matching the P descriptor
            # swizzle and store the register fragment.
            slice_idx = warp_idx // Int32(2)
            warp_idx_in_slice = warp_idx % Int32(2)
            mtx_idx = lane_idx // Int32(8)
            thr_row_idx = lane_idx % Int32(8)
            mtx_col_idx = warp_idx_in_slice * Int32(4) + (mtx_idx % Int32(4))
            smem_offset_bytes = (
                slice_idx * Int32(8 * 128)
                + thr_row_idx * Int32(128)
                + ((mtx_col_idx ^ thr_row_idx) * Int32(16))
            )
            smem_dst = (
                self._smem_base_p_i32.subview((smem_offset_bytes >> 2))
            ).data_ptr()
            prims.stmatrix(
                smem_dst,
                regs_p.data_ptr().load(count=4, alignment=4),
                prims.MMALayout.COL,
                shape=prims.StoreShape.M8N8,
            )
        cute.arch.fence_view_async_shared()
        if cutlass.const_expr(cfg.use_fp8_qkv):
            # FP8 P uses inline STSM stores. Synchronize the producer
            # warpgroup before the UMMA-consumer pipeline is committed so
            # BMM2 cannot observe a partially written P tile.
            prims.barrier_cta_sync(4 + self.inst_id, thread_count=128)

    @consumer_work(
        returns=(
            p_desc_0_slot,
            p_desc_1_slot,
            p_tmem_addr_0_slot,
            p_tmem_addr_1_slot,
        )
    )
    @cute.jit
    def p_operands(
        self, stage_info: StageInfo
    ) -> tuple[
        prims.Tcgen05SmemDesc,
        prims.Tcgen05SmemDesc,
        Int32,
        Int32,
    ]:
        """Publish the stage-specific P operand consumed by BMM2."""
        cfg = self.cfg
        p_desc_0 = prims.Tcgen05SmemDesc(0)
        p_desc_1 = prims.Tcgen05SmemDesc(0)
        p_tmem_addr_0 = Int32(0)
        p_tmem_addr_1 = Int32(0)
        if cutlass.const_expr(cfg.uses_tmem_p):
            # ConsWork: select the physical TMEM stage paired with the P
            # pipeline token that was just waited. The allocation aliases the
            # stats-free columns of the corresponding S stage.
            p_tmem_addr = (
                self._tmem_base_addr
                + Int32(self._tmem_alloc.offset)
                + stage_info.stage_idx * cfg.tmem_s_cols
            )
            if cutlass.const_expr(self.inst_id == 0):
                p_tmem_addr_0 = p_tmem_addr
            else:
                p_tmem_addr_1 = p_tmem_addr
        else:
            # ConsWork: build the SMEM descriptor for P. Only the descriptor
            # slot corresponding to this resource instance is populated; the
            # MmaTask receives both slots and selects the active one for BMM2.
            p_desc = prims.Tcgen05SmemDesc.build(
                self._smem_base_p,
                leading_byte_offset=Int32(cfg.tile_size_q * 128),
                stride_byte_offset=1024,
                layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
            )
            if cutlass.const_expr(self.inst_id == 0):
                p_desc_0 = p_desc
            else:
                p_desc_1 = p_desc
        return p_desc_0, p_desc_1, p_tmem_addr_0, p_tmem_addr_1
