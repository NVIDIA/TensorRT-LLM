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

"""TMEM probability resource for the keeps-MMA-AB PV operand."""

from dataclasses import dataclass
from typing import ClassVar, Optional

from cutlass.experimental import primitives as prims

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32
from cutlass.experimental.task_scheduling.enums import WorkAttr
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
    decode_gen_task_cache,
    num_packed_p_regs,
    num_softmax_scale_groups,
)
from ...helpers.math import (
    fadd2,
    fmul2,
    neg_max_f32,
    pack_float2_to_bf16,
)
from ...helpers.ops import (
    float_to_u32_bits,
    fp8_log2_quant_scale,
    pack_float4_to_fp8_e4m3,
    softmax_sum_state_ptr,
    tcgen05_second_panel_addr,
    tcgen05_store_p_16x32bx2_x16,
    tcgen05_store_p_fp8_16x32bx2_x16,
    u32_bits_to_float,
)

from .common import (
    MlaResource,
)

# =====================================================================
# TmemPResource — P in TMEM for keeps-MMA-AB PV MMA
# =====================================================================


@dataclass(kw_only=True)
class TmemPResource(MlaResource):
    """TMEM probability tile exchanged from softmax to PV MMA.

    The keeps-MMA-AB schedule uses one P pipe: softmax writes BF16 P directly
    into the score TMEM stage, and MmaTask consumes that TMEM stage with
    ``tcgen05.mma`` A-from-TMEM.
    """

    _task_local_specs: ClassVar[tuple[tuple, ...]] = (
        ("p_stage_idx", Int32, Int32(0), "Current TMEM P pipeline stage."),
    )
    inst_id: cutlass.Constexpr[int] = 0
    scale_softmax_log2: Float32 = None
    tmem_alias_ref: Optional[MlaResource] = None
    p_stage_idx: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )

    @consumer_work(work_attrs=WorkAttr.AUXILIARY, returns=p_stage_idx)
    @cute.jit
    def init_stage_state(self, stage_info: StageInfo):
        """Initialize the TMEM P stage index for the first work tile."""
        # Keeps-MMA-AB uses TMEM P instead of SMEM P.  Consumer aux state tracks
        # which score/TMEM stage the PV MMA consumer should read.
        self._init_tmem_state(stage_info)
        return Int32(0)

    @consumer_work(work_attrs=WorkAttr.AUXILIARY, returns=p_stage_idx)
    @cute.jit
    def init_stage_work_tile_state(self, stage_info: StageInfo):
        """Initialize the TMEM P stage index for a persistent work tile."""
        # Reset the TMEM P stage index when a persistent CTA advances to a new
        # work tile.
        del stage_info
        return Int32(0)

    @cute.jit
    def _p_base_addr(self, stage_info: StageInfo, task_cache):
        p_base = (
            task_cache[_TASK_CACHE_TMEM_BASE_OFFSET]
            + Int32(self.tmem_alias_ref._tmem_alloc.offset)
            + stage_info.stage_idx * Int32(self.cfg.tmem_s_cols)
        )
        if cutlass.const_expr(self.cfg.kernel_variant == "keeps_mma_ab"):
            p_base = p_base + Int32(WARP_LANES)
        return p_base

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
        """Materialize grouped-head softmax probabilities into TMEM P."""
        # Producer work for TmemPResource: softmax converts S to P and stores it
        # directly into the TMEM score/P stage.  PV MMA consumes the returned
        # p_stage_idx through the loop/tail TMEM-P producer labels.
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
            local_sums[idx] = Float32(0.0)

        packed_p_reg_count = num_packed_p_regs(cfg)
        regs_p = cutlass.Array(
            Int32, packed_p_reg_count, space=cutlass.AddressSpace.rmem
        )
        if cutlass.const_expr(
            cfg.is_fp8_qkv() and cfg.kernel_variant == "keeps_mma_ab"
        ):
            log2_scale_pair = (self.scale_softmax_log2, self.scale_softmax_log2)
            neg_scaled_pair = (
                neg_scaled_max[0] + fp8_log2_quant_scale(),
                neg_scaled_max[0] + fp8_log2_quant_scale(),
            )
            has_finite_max = new_max_arr[0] != neg_max_f32()
            for packed_idx in cutlass.range_constexpr(packed_p_reg_count):
                s_base = packed_idx * 4
                p0 = Float32(0.0)
                p1 = Float32(0.0)
                p2 = Float32(0.0)
                p3 = Float32(0.0)
                if has_finite_max:
                    scaled01 = fadd2(
                        fmul2(
                            (
                                s_arr[s_base + 0],
                                s_arr[s_base + 1],
                            ),
                            log2_scale_pair,
                        ),
                        neg_scaled_pair,
                    )
                    scaled23 = fadd2(
                        fmul2(
                            (
                                s_arr[s_base + 2],
                                s_arr[s_base + 3],
                            ),
                            log2_scale_pair,
                        ),
                        neg_scaled_pair,
                    )
                    p0 = cute.math.exp2(scaled01[0], fastmath=True)
                    p1 = cute.math.exp2(scaled01[1], fastmath=True)
                    p2 = cute.math.exp2(scaled23[0], fastmath=True)
                    p3 = cute.math.exp2(scaled23[1], fastmath=True)
                local_sums[0] += p0
                local_sums[0] += p1
                local_sums[0] += p2
                local_sums[0] += p3
                regs_p[packed_idx] = pack_float4_to_fp8_e4m3(p0, p1, p2, p3)
        elif cutlass.const_expr(cfg.is_fp8_qkv()):
            for packed_idx in cutlass.range_constexpr(packed_p_reg_count):
                p_vals = cutlass.Array(Float32, 4, space=cutlass.AddressSpace.rmem)
                for elem_idx in cutlass.range_constexpr(4):
                    s_idx = packed_idx * 4 + elem_idx
                    pair_idx = s_idx // 2
                    scale_base = (
                        (pair_idx % (2 * max(cfg.tile_size_q // 8, 1))) // 2
                    ) * 2
                    scale_idx = scale_base + (s_idx % 2)
                    p_val = Float32(0.0)
                    if new_max_arr[scale_idx] != neg_max_f32():
                        p_val = cute.math.exp2(
                            s_arr[s_idx] * self.scale_softmax_log2
                            + neg_scaled_max[scale_idx]
                            + fp8_log2_quant_scale(),
                            fastmath=True,
                        )
                    p_vals[elem_idx] = p_val
                    local_sums[scale_idx] += p_val
                regs_p[packed_idx] = pack_float4_to_fp8_e4m3(
                    p_vals[0], p_vals[1], p_vals[2], p_vals[3]
                )
        else:
            for pair_idx in cutlass.range_constexpr(packed_p_reg_count):
                s0 = pair_idx * 2
                s1 = s0 + 1
                p0 = Float32(0.0)
                p1 = Float32(0.0)
                if cutlass.const_expr(cfg.kernel_variant == "keeps_mma_ab"):
                    new_max = new_max_arr[0]
                    if new_max != neg_max_f32():
                        p0 = cute.math.exp2(
                            s_arr[s0] * self.scale_softmax_log2 + neg_scaled_max[0],
                            fastmath=True,
                        )
                        p1 = cute.math.exp2(
                            s_arr[s1] * self.scale_softmax_log2 + neg_scaled_max[0],
                            fastmath=True,
                        )
                    local_sums[0] += p0
                    local_sums[0] += p1
                else:
                    scale0 = ((pair_idx % (2 * max(cfg.tile_size_q // 8, 1))) // 2) * 2
                    scale1 = scale0 + 1
                    new_max0 = new_max_arr[scale0]
                    new_max1 = new_max_arr[scale1]
                    if new_max0 != neg_max_f32():
                        p0 = cute.math.exp2(
                            s_arr[s0] * self.scale_softmax_log2
                            + neg_scaled_max[scale0],
                            fastmath=True,
                        )
                    if new_max1 != neg_max_f32():
                        p1 = cute.math.exp2(
                            s_arr[s1] * self.scale_softmax_log2
                            + neg_scaled_max[scale1],
                            fastmath=True,
                        )
                    local_sums[scale0] += p0
                    local_sums[scale1] += p1
                regs_p[pair_idx] = pack_float2_to_bf16(p0, p1)

        for scale_idx in cutlass.range_constexpr(num_scale_groups):
            self.tmem_alias_ref._p_local_sum_arr[scale_idx] = local_sums[scale_idx]
            local_sum_arr[scale_idx] = local_sums[scale_idx]

        if cutlass.const_expr(cfg.kernel_variant == "keeps_mma_ab"):
            state_idx = cute.arch.thread_idx()[0]
            state_ptr = self.tmem_alias_ref._softmax_scratch.data_ptr(state_idx)
            old_max = u32_bits_to_float(state_ptr.load(is_volatile=True, alignment=4))
            old_sum = u32_bits_to_float(
                softmax_sum_state_ptr(state_ptr).load(is_volatile=True, alignment=4)
            )
            new_max = new_max_arr[0]
            exp_scale = cute.math.exp2(
                self.scale_softmax_log2 * (old_max - new_max),
                fastmath=True,
            )
            updated_sum = exp_scale * old_sum + local_sums[0]
            state_ptr.store(
                float_to_u32_bits(new_max),
                is_volatile=True,
                alignment=4,
            )
            softmax_sum_state_ptr(state_ptr).store(
                float_to_u32_bits(updated_sum),
                is_volatile=True,
                alignment=4,
            )

        p_base = self._p_base_addr(stage_info, task_cache)
        if cutlass.const_expr(cfg.is_fp8_qkv()):
            tcgen05_store_p_fp8_16x32bx2_x16(p_base, regs_p)
            if cutlass.const_expr(cfg.head_dim_per_cta_v > 256):
                tcgen05_store_p_fp8_16x32bx2_x16(
                    tcgen05_second_panel_addr(p_base), regs_p
                )
        else:
            tcgen05_store_p_16x32bx2_x16(p_base, regs_p, 0)
            tcgen05_store_p_16x32bx2_x16(p_base + Int32(16), regs_p, 16)
            if cutlass.const_expr(cfg.head_dim_per_cta_v > 256):
                tcgen05_store_p_16x32bx2_x16(
                    tcgen05_second_panel_addr(p_base), regs_p, 0
                )
                tcgen05_store_p_16x32bx2_x16(
                    tcgen05_second_panel_addr(p_base) + Int32(16), regs_p, 16
                )
        prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
        cute.arch.fence_view_async_tmem_store()

    @consumer_work(returns=p_stage_idx)
    @cute.jit
    def p_stage(self, stage_info: StageInfo):
        """Publish the TMEM P pipeline stage consumed by PV MMA."""
        # Consumer work returns only the live TMEM stage index; the payload is
        # the P data already stored in the aliased TMEM allocation.
        return stage_info.stage_idx
