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

"""VC-Attention-QK16 operands and helpers of the context kernel.

VC-Attention-QK16 stores E4M3 V as residuals around the bf16 mean of each
128-token K/V tile and restores the mean inside the online softmax with one
bf16 K=16 UMMA step per tile, ``O += rowsum(P_tile) * mean_tile``.
:class:`SmemMuResource` is the TMA-fed ring of the ``mean_tile`` operands. The
ExpCast helpers code the probabilities directly as E4M3 bytes and accumulate
the quantized row sums that become the step's A operand.
The resources in ``fmha_resources`` own the row-sum operand (behind the
SMEM P tile), the exp2 and store paths and the mean UMMA issue.
"""

from ...vc_attention import VC_MEAN_GROUP_TILES, VC_MEAN_OPERANDS
from dataclasses import dataclass, field
from typing import Any, Optional

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int16, Int32
from cutlass.cutlass_dsl import Constexpr
from cutlass.experimental import primitives as prims
from cutlass.experimental.task_scheduling.enums import WorkAttr
from cutlass.experimental.task_scheduling.memory import SmemAllocation
from cutlass.experimental.task_scheduling.resources import (
    MemoryResource,
    PipelineConfig,
    StageInfo,
    TaskLocalVariable,
    consumer_work,
    producer_work,
)

from ..placeholder_helpers import _placeholder_smem_array

# VC-Attention-QK16 mean-restore operands of the bf16 K=16 UMMA step that adds
# ``rowsum(P_tile) * mean_tile`` into O. Unswizzled K-major 32-byte rows with
# LBO 128 between the two K core matrices and SBO 256 between 8-row groups.
VC_MEAN_TILE_LBO = 128
VC_MEAN_TILE_SBO = 256

# ExpCast-FP8 (VC-Attention-QK16). An E4M3 byte read as an integer equals
# ``8*log2(v) + 56 + eps``. With ``u`` the log2-domain probability and the
# paper's 2^8 P scaling, ``code = round(8*u + VC_EXPCAST_CODE_BIAS)`` replaces
# exp2 and the E4M3 cast by one FMA. beta is the paper's centering of
# Mitchell's error.
_E4M3_EXP_BIAS = 7
E4M3_CODES_PER_OCTAVE = 8
_VC_EXPCAST_BETA = -0.35
_VC_EXPCAST_P_SCALE_LOG2 = 8
VC_EXPCAST_CODE_BIAS = (
    E4M3_CODES_PER_OCTAVE * (_E4M3_EXP_BIAS + _VC_EXPCAST_P_SCALE_LOG2)
    + _VC_EXPCAST_BETA
)


@cute.jit
def _split_bf16_hi_lo_word(value: Float32) -> Int32:
    """Pack ``value`` as two bf16 halves: low 16 bits = truncated high part,
    high 16 bits = truncated remainder ``value - hi``."""
    return cute.arch.inline_ptx(
        """{
            .reg .b32 bits, hb, hf, lobits, lb;
            .reg .f32 hif, lof;
            mov.b32 bits, {$r0};
            shr.u32 hb, bits, 16;
            shl.b32 hf, hb, 16;
            mov.b32 hif, hf;
            sub.f32 lof, {$r0}, hif;
            mov.b32 lobits, lof;
            and.b32 lb, lobits, 0xffff0000;
            or.b32 {$w0}, hb, lb;
        }""",
        write_only_types=[Int32],
        read_only_args=[value],
    )


@cute.jit
def _expcast_e4m3_quad_relu(
    c0: Float32,
    c1: Float32,
    c2: Float32,
    c3: Float32,
    acc01: Int32,
    acc23: Int32,
) -> tuple[Int32, Int32, Int32]:
    """ExpCast four unclamped codes into one packed E4M3 word and accumulate
    their values into two f16x2 row-sum registers.

    ``c = 8 * (s * scale - m) + VC_EXPCAST_CODE_BIAS``

    ``cvt.rn.relu.f16x2.f32`` packs a pair and clamps underflow to code 0, an
    f16 FMA adds 1024 so the low byte of each half is the rounded code, one
    ``prmt`` gathers the four bytes, and the E4M3-pair to f16x2 conversion
    with one f16 FMA per pair accumulates the quantized values.
    """
    return cute.arch.inline_ptx(
        """
        {
            .reg .b32 h01, h23, y01, y23, v01, v23, a01, a23, r01, r23, one2, k1024;
            .reg .b16 c01, c23;
            mov.b32 one2, 0x3C003C00;
            mov.b32 k1024, 0x64006400;
            mov.b32 a01, {$r4};
            mov.b32 a23, {$r5};
            cvt.rn.relu.f16x2.f32 h01, {$r1}, {$r0};
            cvt.rn.relu.f16x2.f32 h23, {$r3}, {$r2};
            fma.rn.f16x2 y01, h01, one2, k1024;
            fma.rn.f16x2 y23, h23, one2, k1024;
            prmt.b32 {$w0}, y01, y23, 0x6420;
            mov.b32 {c01, c23}, {$w0};
            cvt.rn.f16x2.e4m3x2 v01, c01;
            cvt.rn.f16x2.e4m3x2 v23, c23;
            fma.rn.f16x2 r01, v01, one2, a01;
            fma.rn.f16x2 r23, v23, one2, a23;
            mov.b32 {$w1}, r01;
            mov.b32 {$w2}, r23;
        }""",
        write_only_types=[Int32, Int32, Int32],
        read_only_args=[c0, c1, c2, c3, acc01, acc23],
    )


@cute.jit
def _expcast_e4m3_quad_relu_init(
    c0: Float32, c1: Float32, c2: Float32, c3: Float32
) -> tuple[Int32, Int32, Int32]:
    """First quad of a chain for :func:`_expcast_e4m3_quad_relu` (zero accumulators)."""
    return cute.arch.inline_ptx(
        """
        {
            .reg .b32 h01, h23, y01, y23, one2, k1024;
            .reg .b16 c01, c23;
            mov.b32 one2, 0x3C003C00;
            mov.b32 k1024, 0x64006400;
            cvt.rn.relu.f16x2.f32 h01, {$r1}, {$r0};
            cvt.rn.relu.f16x2.f32 h23, {$r3}, {$r2};
            fma.rn.f16x2 y01, h01, one2, k1024;
            fma.rn.f16x2 y23, h23, one2, k1024;
            prmt.b32 {$w0}, y01, y23, 0x6420;
            mov.b32 {c01, c23}, {$w0};
            cvt.rn.f16x2.e4m3x2 {$w1}, c01;
            cvt.rn.f16x2.e4m3x2 {$w2}, c23;
        }""",
        write_only_types=[Int32, Int32, Int32],
        read_only_args=[c0, c1, c2, c3],
    )


@cute.jit
def _f16x2_sum4(a0: Int32, a1: Int32, a2: Int32, a3: Int32) -> Float32:
    """fp32 sum of the eight fp16 halves of four f16x2 registers."""
    return cute.arch.inline_ptx(
        """
        {
            .reg .b16 h<8>;
            .reg .f32 f<8>;
            mov.b32 {h0, h1}, {$r0};
            mov.b32 {h2, h3}, {$r1};
            mov.b32 {h4, h5}, {$r2};
            mov.b32 {h6, h7}, {$r3};
            cvt.f32.f16 f0, h0;
            cvt.f32.f16 f1, h1;
            cvt.f32.f16 f2, h2;
            cvt.f32.f16 f3, h3;
            cvt.f32.f16 f4, h4;
            cvt.f32.f16 f5, h5;
            cvt.f32.f16 f6, h6;
            cvt.f32.f16 f7, h7;
            add.f32 f0, f0, f1;
            add.f32 f2, f2, f3;
            add.f32 f4, f4, f5;
            add.f32 f6, f6, f7;
            add.f32 f0, f0, f2;
            add.f32 f4, f4, f6;
            add.f32 {$w0}, f0, f4;
        }""",
        write_only_types=[Float32],
        read_only_args=[a0, a1, a2, a3],
    )


# ---------------------------------------------------------------------------
# SmemMuResource -- VC-Attention-QK16 bf16 tile-mean operand ring
# ---------------------------------------------------------------------------
@dataclass(kw_only=True)
class SmemMuResource(MemoryResource):
    """The bf16 [D x 16] K-major mean operands of one group of 16 K/V tiles.

    Producer: the load warp TMA-loads the operands of the group holding K/V tile
    ``i-1`` beside V tile ``i``. Consumer: the UMMA warp issues two
    ``kind::f16`` K=16 steps ``O += sum_j rowsum(P_j) * mean_j`` over the group's
    tiles after the PV steps of the tile that follows the group, so the means
    restored into O are rescaled by the same correction as the residual
    product. Rows ``k=2i`` and ``k=2i+1`` of operand ``o`` hold the mean of tile
    ``8o+i`` of the group (divided by the V dequant scale).
    """

    cfg: Constexpr[Any] = field(init=False, default=None)
    tma_mu_desc: cutlass.Pointer | None = field(init=False, default=None)
    sMu_array: cutlass.Array = field(init=False, default=None)
    _alloc: Constexpr[Optional[SmemAllocation]] = field(init=False, default=None)
    desc_mu_base: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    # Second live handle for the tail, where the last tile's operand is waited
    # while the previous tile's is still in use.
    desc_mu_last: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()

    def __init__(
        self,
        tma_mu_desc: cutlass.Pointer | None,
        pipeline_config: PipelineConfig,
        cfg: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(pipeline_config=pipeline_config, **kwargs)
        self.cfg = cfg
        self.tma_mu_desc = tma_mu_desc
        self._alloc = SmemAllocation(
            "smem_mu",
            pipeline_config.num_stages * cfg.vc_mean_tile_bytes,
            alignment=cfg.buffer_align_bytes,
        )
        self.sMu_array = _placeholder_smem_array(cutlass.BFloat16)
        self.desc_mu_base = TaskLocalVariable(
            dtype=cutlass.Int64,
            default=cutlass.Int64(0),
            docs="SMEM descriptor base for the current tile-mean operand.",
        )
        self.desc_mu_last = TaskLocalVariable(
            dtype=cutlass.Int64,
            default=cutlass.Int64(0),
            docs="SMEM descriptor base for the last tile's mean operand.",
        )

    def get_smem_requirements(self) -> list[SmemAllocation]:
        return [self._alloc]

    @property
    def stage_elements(self) -> int:
        return self.cfg.vc_mean_tile_bytes // 2

    @cute.jit
    def _init_smem_state(self, stage_info: StageInfo) -> None:
        smem_base = stage_info.context.smem_base
        self.sMu_array = cutlass.Array(
            smem_base.data_ptr() + self._alloc.offset,
            dtype=cutlass.BFloat16,
            shape=(self.pipeline_config.num_stages * self.stage_elements,),
            addrspace=3,
        )

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_load_state(self, stage_info: StageInfo) -> None:
        self._init_smem_state(stage_info)

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_descriptor_state(self, stage_info: StageInfo) -> None:
        self._init_smem_state(stage_info)

    @property
    def loop_offset_sensitive(self) -> bool:
        return True

    @cute.jit
    def _tma_load_tile(
        self,
        stage_info: StageInfo,
        tile_idx: Int32,
        kv_head_coord: Int32,
        batch_coord: Int32,
    ) -> None:
        if prims.elect_sync():
            operand_rows = self.cfg.vc_mean_operand_bytes // 512
            operand_elems = (
                self.cfg.vc_mean_operand_bytes // self.cfg.cta_group_size // 2
            )
            for operand in cutlass.range_constexpr(VC_MEAN_OPERANDS):
                sMu_curr = self.sMu_array.subview(
                    stage_info.stage_idx * self.stage_elements + operand * operand_elems
                )
                row = Int32(operand * operand_rows)
                if cutlass.const_expr(self.cfg.two_cta_umma):
                    # Each CTA stages the 4 packed rows (64 channels) its half of
                    # the M=256 mean step reads, like K/V.
                    cta_rank = cute.arch.make_warp_uniform(
                        cute.arch.block_idx_in_cluster()
                    )
                    prims.cp_async_bulk_tensor_shared_cluster_global(
                        sMu_curr,
                        self.tma_mu_desc,
                        (
                            Int32(0),
                            row + cta_rank * Int32(operand_rows // 2),
                            tile_idx,
                            kv_head_coord,
                            batch_coord,
                        ),
                        cutlass.Array(
                            stage_info.barrier.data_ptr(), dtype=cutlass.Int64
                        ),
                        [],
                        multicast_mask=Int16(Int32(1) << cta_rank),
                        group=prims.CTAGroup.CTA_2,
                    )
                else:
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        sMu_curr,
                        self.tma_mu_desc,
                        (Int32(0), row, tile_idx, kv_head_coord, batch_coord),
                        stage_info.barrier,
                    )

    @producer_work
    @cute.jit
    def mu_load(
        self,
        stage_info: StageInfo,
        *,
        kv_head_coord: Int32,
        batch_coord: Int32,
        kv_tile_start: Int32,
    ) -> None:
        """TMA-load the mean operand of the group holding K/V tile ``loop_offset - 1``.

        A group's mean step is issued one tile late, beside PV of the tile after
        the group, so the ring runs one tile behind V.
        """
        tile_idx = cute.math.max(
            kv_tile_start + stage_info.loop_offset - Int32(1), Int32(0)
        )
        self._tma_load_tile(
            stage_info,
            tile_idx // Int32(VC_MEAN_GROUP_TILES),
            kv_head_coord,
            batch_coord,
        )

    @producer_work
    @cute.jit
    def mu_load_last(
        self,
        stage_info: StageInfo,
        *,
        kv_head_coord: Int32,
        batch_coord: Int32,
        kv_tile_start: Int32,
    ) -> None:
        """TMA-load the mean operand of the last group for the tail mean step."""
        tile_idx = kv_tile_start + Int32(stage_info.loop_end) - Int32(1)
        self._tma_load_tile(
            stage_info,
            tile_idx // Int32(VC_MEAN_GROUP_TILES),
            kv_head_coord,
            batch_coord,
        )

    @cute.jit
    def _build_desc(self, stage_info: StageInfo) -> prims.Tcgen05SmemDesc:
        sMu_curr = self.sMu_array.subview(stage_info.stage_idx * self.stage_elements)
        return prims.Tcgen05SmemDesc.build(
            sMu_curr,
            leading_byte_offset=VC_MEAN_TILE_LBO,
            stride_byte_offset=VC_MEAN_TILE_SBO,
            layout=prims.Tcgen05SmemSwizzle.NONE,
        )

    @consumer_work(returns=desc_mu_base)
    @cute.jit
    def mu_desc(self, stage_info: StageInfo) -> prims.Tcgen05SmemDesc:
        """Build the unswizzled K-major descriptor of the current mean operand."""
        return self._build_desc(stage_info)

    @consumer_work(returns=desc_mu_last)
    @cute.jit
    def mu_desc_last(self, stage_info: StageInfo) -> prims.Tcgen05SmemDesc:
        """Descriptor of the last tile's mean operand, held beside the previous one."""
        return self._build_desc(stage_info)
