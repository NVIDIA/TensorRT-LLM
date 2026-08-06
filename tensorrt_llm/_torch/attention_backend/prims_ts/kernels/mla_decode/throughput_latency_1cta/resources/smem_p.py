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

"""SMEM probability resource for the swaps-MMA-AB PV operand."""

from dataclasses import dataclass
from typing import Optional

from cutlass.experimental import primitives as prims

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64
from cutlass.experimental.task_scheduling.enums import WorkAttr
from cutlass.experimental.task_scheduling.memory import SmemAllocation
from cutlass.experimental.task_scheduling.resources import (
    MemoryResource,
    StageInfo,
    consumer_work,
    producer_work,
)

from ...helpers.constants import (
    SMEM_WORD_BYTE_SHIFT,
    SMEM_WORD_BYTES,
    TCGEN05_16X256B_REGS_PER_LOAD,
    TMEM_LIFECYCLE_BARRIER_ID,
)


from ...helpers.layout import (
    _TASK_CACHE_LANE_IDX,
    _TASK_CACHE_WARP_GRP_THREAD_IDX,
    _TASK_CACHE_WARP_IDX,
    decode_gen_task_cache,
    num_o_stsm_row_blocks,
    num_packed_p_regs,
    num_softmax_scale_groups,
    p_stsm_smem_offset_bytes,
    smem_array,
)
from ...helpers.math import (
    neg_max_f32,
    pack_float2_to_bf16,
    qkv_dtype,
)
from ...helpers.ops import (
    fp8_log2_quant_scale,
    pack_float4_to_fp8_e4m3,
    store_transposed_smem8b_x2,
    store_transposed_smem8b_x4,
)

from .common import (
    MlaResource,
)

# =====================================================================
# SmemPResource — P in SMEM, AsyncUmma pipeline
# =====================================================================


@dataclass(kw_only=True)
class SmemPResource(MlaResource):
    """SMEM probability tile exchanged from softmax to PV MMA.

    AsyncUmma synchronizes the softmax producer with the UMMA PV consumer.
    Softmax writes P with stmatrix and fences the async-shared view before
    commit; MmaTask waits before PV and releases after the tensor-core read.
    """

    inst_id: cutlass.Constexpr[int] = 0
    scale_softmax_log2: Float32 = None
    tmem_s_ref: Optional[MemoryResource] = None
    order_p01_alloc: cutlass.Constexpr[Optional[SmemAllocation]] = None
    owns_order_p01_alloc: cutlass.Constexpr[bool] = False
    _smem_p: object = None
    _smem_p_i32: object = None
    _order_p01_barrier_ptr: object = None
    _order_p01_phase: object = None

    def get_smem_requirements(self):
        """Return P SMEM plus optional ordering-barrier allocation."""
        if self._alloc is None:
            self._alloc = SmemAllocation(
                name=f"{self.name}",
                size_bytes=self.cfg.p_smem_tile_bytes,
                alignment=self.cfg.stensor_align,
            )
        allocs = [self._alloc]
        if self.owns_order_p01_alloc and self.order_p01_alloc is not None:
            allocs.append(self.order_p01_alloc)
        return allocs

    @cute.jit
    def _init_smem_state_from_context(self, context) -> None:
        """Create P SMEM views and optional producer-order barrier pointer."""
        self._smem_p = smem_array(
            context,
            self._alloc,
            qkv_dtype(self.cfg),
            self.cfg.p_smem_tile_bytes // self.cfg.qkv_dtype_bytes,
        )
        self._smem_p_i32 = smem_array(
            context,
            self._alloc,
            Int32,
            self.cfg.p_smem_tile_bytes // 4,
        )
        if cutlass.const_expr(
            context is not None
            and context.smem_base is not None
            and self.order_p01_alloc is not None
        ):
            self._order_p01_barrier_ptr = cute.make_ptr(
                Int64,
                context.smem_base.data_ptr() + self.order_p01_alloc.offset,
                mem_space=cute.AddressSpace.smem,
            )
        self._order_p01_phase = Int32(1 if self.inst_id == 0 else 0)

    @cute.jit
    def initialize_runtime_state_internal(
        self,
        context=None,
        captured_schedule: cutlass.Constexpr[bool] = False,
    ) -> None:
        """Initialize pipeline state and CTA-wide P-order barriers."""
        super().initialize_runtime_state_internal(context, captured_schedule)
        self._init_smem_state_from_context(context)
        if cutlass.const_expr(
            self.cfg.use_clc_dynamic_persistent_scheduler == 1
            and self._order_p01_barrier_ptr is not None
        ):
            self._init_order_p01_barriers()

    @cute.jit
    def _init_smem_state(self, stage_info: StageInfo) -> None:
        """Create P SMEM views for captured schedule aux work."""
        self._init_smem_state_from_context(stage_info.context)

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_materialize_state(self, stage_info: StageInfo) -> None:
        """Initialize P SMEM state before softmax materializes probabilities."""

        # Producer aux work prepares P SMEM for the softmax task before it
        # materializes probabilities.
        self._init_smem_state(stage_info)

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_descriptor_state(self, stage_info: StageInfo) -> None:
        """Initialize P SMEM state before PV MMA consumes descriptors."""

        # Consumer aux work prepares the same P SMEM view for the MMA task.
        self._init_smem_state(stage_info)

    @cute.jit
    def _init_order_p01_barriers(self):
        """Initialize the ordered P0/P1 barriers used by CLC scheduling."""

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bar_init_warp = Int32(2 if self.inst_id == 0 else 6)
        if warp_idx == bar_init_warp:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(
                    self._order_p01_barrier_ptr,
                    Int32(128),
                )
                cute.arch.mbarrier_init(
                    self._order_p01_barrier_ptr + 1,
                    Int32(128),
                )
        cute.arch.mbarrier_init_fence()
        prims.barrier_cta_sync(
            barrier_id=TMEM_LIFECYCLE_BARRIER_ID,
            thread_count=self.cfg.threads_per_cta,
        )

    @cute.jit
    def _ordered_sequence_wait(self):
        """Wait for this P instance's ordered materialization turn."""

        if cutlass.const_expr(self.cfg.use_clc_dynamic_persistent_scheduler == 1):
            cute.arch.mbarrier_wait(
                self._order_p01_barrier_ptr + self.inst_id,
                self._order_p01_phase,
            )

    @cute.jit
    def _ordered_sequence_arrive(self):
        """Signal the peer P instance after materialization completes."""

        if cutlass.const_expr(self.cfg.use_clc_dynamic_persistent_scheduler == 1):
            signaling_id = 1 if self.inst_id == 0 else 0
            cute.arch.mbarrier_arrive(
                self._order_p01_barrier_ptr + signaling_id,
            )
            self._order_p01_phase = self._order_p01_phase ^ Int32(1)

    @producer_work
    @cute.jit
    def materialize_p(
        self,
        stage_info: StageInfo,
        *,
        new_max_arr,
        s_arr,
        local_sum_arr,
    ):
        """Materialize grouped-head softmax probabilities in P SMEM for BMM2."""
        # Producer work for SmemPResource: softmax writes P into the acquired
        # SMEM stage.  The AsyncUmma commit protects the payload until PV MMA
        # waits on p_desc() and releases it.
        cfg = self.cfg
        task_cache = decode_gen_task_cache(stage_info)

        num_scale_groups = num_softmax_scale_groups(cfg)
        neg_scaled_max = cutlass.Array(
            Float32, num_scale_groups, space=cutlass.AddressSpace.rmem
        )
        local_sums = cutlass.Array(
            Float32, num_scale_groups, space=cutlass.AddressSpace.rmem
        )
        for idx in cutlass.range_constexpr(num_scale_groups):
            new_max = new_max_arr[idx]
            safe_new_max = new_max
            if safe_new_max == neg_max_f32():
                safe_new_max = Float32(0.0)
            neg_scaled_max[idx] = -self.scale_softmax_log2 * safe_new_max
            if cutlass.const_expr(cfg.is_fp8_qkv()):
                neg_scaled_max[idx] += fp8_log2_quant_scale()
            local_sums[idx] = Float32(0.0)

        # Convert S registers to P and accumulate local softmax sums. FP8 packs
        # four probabilities per register; BF16 packs two.
        packed_p_reg_count = num_packed_p_regs(cfg)
        regs_p = cutlass.Array(
            Int32, packed_p_reg_count, space=cutlass.AddressSpace.rmem
        )
        self._ordered_sequence_wait()
        if cutlass.const_expr(cfg.is_fp8_qkv()):
            for packed_idx in cutlass.range_constexpr(packed_p_reg_count):
                p_vals = cutlass.Array(Float32, 4, space=cutlass.AddressSpace.rmem)
                for elem_idx in cutlass.range_constexpr(4):
                    s_idx = packed_idx * 4 + elem_idx
                    pair_idx = s_idx // 2
                    scale_base = (
                        (pair_idx % (2 * max(cfg.tile_size_q // 8, 1))) // 2
                    ) * 2
                    scale_idx = scale_base + (s_idx % 2)
                    p_val = cute.math.exp2(
                        s_arr[s_idx] * self.scale_softmax_log2
                        + neg_scaled_max[scale_idx],
                        fastmath=True,
                    )
                    p_vals[elem_idx] = p_val
                    local_sums[scale_idx] += p_val
                regs_p[packed_idx] = pack_float4_to_fp8_e4m3(
                    p_vals[0], p_vals[1], p_vals[2], p_vals[3]
                )
                if cutlass.const_expr(packed_idx == packed_p_reg_count // 2):
                    self._ordered_sequence_arrive()
        else:
            for pair_idx in cutlass.range_constexpr(packed_p_reg_count):
                scale0 = ((pair_idx % (2 * max(cfg.tile_size_q // 8, 1))) // 2) * 2
                scale1 = scale0 + 1
                s0 = pair_idx * 2
                s1 = s0 + 1
                p0 = Float32(0.0)
                p1 = Float32(0.0)
                new_max0 = new_max_arr[scale0]
                new_max1 = new_max_arr[scale1]
                if new_max0 != neg_max_f32():
                    p0 = cute.math.exp2(
                        s_arr[s0] * self.scale_softmax_log2 + neg_scaled_max[scale0],
                        fastmath=True,
                    )
                if new_max1 != neg_max_f32():
                    p1 = cute.math.exp2(
                        s_arr[s1] * self.scale_softmax_log2 + neg_scaled_max[scale1],
                        fastmath=True,
                    )
                local_sums[scale0] += p0
                local_sums[scale1] += p1
                regs_p[pair_idx] = pack_float2_to_bf16(p0, p1)
                if cutlass.const_expr(pair_idx == packed_p_reg_count // 2):
                    self._ordered_sequence_arrive()

        # Publish local sums before committing P so the softmax sum update can
        # consume them while PV MMA waits on the AsyncUmma P stage.
        for scale_idx in cutlass.range_constexpr(num_scale_groups):
            self.tmem_s_ref._p_local_sum_arr[scale_idx] = local_sums[scale_idx]
            local_sum_arr[scale_idx] = local_sums[scale_idx]

        warp_idx = task_cache[_TASK_CACHE_WARP_IDX]
        lane_idx = task_cache[_TASK_CACHE_LANE_IDX]
        warp_grp_thread_idx = task_cache[_TASK_CACHE_WARP_GRP_THREAD_IDX]
        if cutlass.const_expr(cfg.is_fp8_qkv()):
            # FP8 P uses byte-transposed STSM stores so PV MMA can consume the
            # same logical P tile layout as the BF16 path.
            if cutlass.const_expr(packed_p_reg_count == 2):
                store_transposed_smem8b_x2(
                    self._smem_p_i32,
                    regs_p[0],
                    regs_p[1],
                    warp_grp_thread_idx,
                    cfg.tile_size_q,
                    cfg.tile_size_kv,
                )
            else:
                for stsm_chunk_idx in cutlass.range_constexpr(packed_p_reg_count // 4):
                    reg_base = stsm_chunk_idx * TCGEN05_16X256B_REGS_PER_LOAD
                    store_transposed_smem8b_x4(
                        self._smem_p_i32,
                        regs_p[reg_base],
                        regs_p[reg_base + 1],
                        regs_p[reg_base + 2],
                        regs_p[reg_base + 3],
                        warp_grp_thread_idx,
                        cfg.tile_size_q,
                        cfg.tile_size_kv,
                        stsm_idx=stsm_chunk_idx,
                    )
            # Every softmax producer thread reaches the AsyncUmma commit after
            # this proxy fence.  Its 128-thread full-mbarrier arrival count is
            # already the cross-warp rendezvous consumed by the MMA wait, so a
            # second named CTA barrier would only serialize the producer.
            cute.arch.fence_view_async_shared()
            return

        for stsm_chunk_idx in cutlass.range_constexpr(packed_p_reg_count // 4):
            stsm_group_idx = stsm_chunk_idx // num_o_stsm_row_blocks(cfg)
            stsm_row_block_idx = stsm_chunk_idx % num_o_stsm_row_blocks(cfg)
            smem_offset_bytes = p_stsm_smem_offset_bytes(
                warp_idx,
                lane_idx,
                stsm_group_idx,
                stsm_row_block_idx,
                cfg.tile_size_q,
            )
            smem_dst = self._smem_p_i32.data_ptr(
                smem_offset_bytes >> SMEM_WORD_BYTE_SHIFT
            )
            prims.stmatrix(
                smem_dst,
                (
                    regs_p.data_ptr() + stsm_chunk_idx * TCGEN05_16X256B_REGS_PER_LOAD
                ).load(
                    count=TCGEN05_16X256B_REGS_PER_LOAD,
                    alignment=SMEM_WORD_BYTES,
                ),
                prims.MMALayout.COL,
                shape=prims.StoreShape.M8N8,
            )
        cute.arch.fence_view_async_shared()

    @consumer_work
    @cute.jit
    def p_desc(self, stage_info: StageInfo):
        """Publish the P SMEM descriptor consumed by PV MMA."""
        # The descriptor itself is reconstructed by TmemOResource from this
        # resource's SMEM pointer.  This consumer work is still needed as the TS
        # wait/release edge that keeps P live until PV MMA is done.
        del stage_info
