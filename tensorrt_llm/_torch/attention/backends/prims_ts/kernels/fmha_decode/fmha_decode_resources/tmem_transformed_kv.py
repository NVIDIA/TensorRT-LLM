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

"""Transformed NVFP4 K/V resource backed by tensor memory."""

from dataclasses import dataclass
from typing import ClassVar

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint8
from cutlass.experimental import primitives as prims
from cutlass.experimental.task_scheduling.enums import WorkAttr
from cutlass.experimental.task_scheduling.memory import (
    ResourceContext,
    SmemAllocation,
    TmemAllocation,
)
from cutlass.experimental.task_scheduling.resources import (
    StageInfo,
    TaskLocalVariable,
    consumer_work,
    producer_work,
)

from ..fmha_decode_config import FmhaDecodeConfig
from .helpers_common import (
    _TASK_CACHE_TMEM_BASE_OFFSET,
    _TASK_CACHE_WARP_GRP_THREAD_IDX,
    Constexpr,
    DecodeGenResourceBase,
    ResourceVars,
    _decode_gen_task_cache,
)
from .smem_resources import SmemKvResource


@cute.jit
def _mul_e2m1x4_e4m3x4(packed_fp4: Int32, packed_sf: Int32) -> Int32:
    """Apply four E4M3 scale factors to four unpacked E2M1 values.

    PTX 9.4 (CUDA 13.4) supports the byte-padded E2M1 operand directly.
    Each FP4 value occupies the low nibble of one byte in the 32-bit input.
    """
    if cutlass.const_expr(cutlass.target_version(min_version="13.4")):
        return cute.arch.inline_ptx(
            "mul.e4m3x4.e2m1p4x4.e4m3x4.satfinite {$w0}, {$r0}, {$r1};",
            write_only_types=[Int32],
            read_only_args=[packed_fp4, packed_sf],
        )
    else:
        return cute.arch.inline_ptx(
            """
            {
                .reg .b8 fp4_01, fp4_23, unused0, unused1, unused2;
                .reg .b16 sf_01, sf_23, e4m3_01, e4m3_23;
                .reg .b32 tmp, h_01, h_23, sf_h_01, sf_h_23;
                and.b32 tmp, {$r0}, 0x0000000f;
                bfe.u32 h_01, {$r0}, 8, 4;
                shl.b32 h_01, h_01, 4;
                or.b32 h_01, h_01, tmp;
                mov.b32 {fp4_01, unused0, unused1, unused2}, h_01;
                bfe.u32 h_23, {$r0}, 16, 4;
                bfe.u32 tmp, {$r0}, 24, 4;
                shl.b32 tmp, tmp, 4;
                or.b32 h_23, h_23, tmp;
                mov.b32 {fp4_23, unused0, unused1, unused2}, h_23;
                mov.b32 {sf_01, sf_23}, {$r1};
                cvt.rn.f16x2.e2m1x2 h_01, fp4_01;
                cvt.rn.f16x2.e2m1x2 h_23, fp4_23;
                cvt.rn.f16x2.e4m3x2 sf_h_01, sf_01;
                cvt.rn.f16x2.e4m3x2 sf_h_23, sf_23;
                mul.rn.f16x2 h_01, h_01, sf_h_01;
                mul.rn.f16x2 h_23, h_23, sf_h_23;
                cvt.rn.satfinite.e4m3x2.f16x2 e4m3_01, h_01;
                cvt.rn.satfinite.e4m3x2.f16x2 e4m3_23, h_23;
                mov.b32 {$w0}, {e4m3_01, e4m3_23};
            }
            """,
            write_only_types=[Int32],
            read_only_args=[packed_fp4, packed_sf],
        )


@cute.jit
def _mul_e2m1x4_e4m3x4_bx(
    packed_fp4: Int32, packed_sf: Int32, sf_byte_idx: Constexpr[int]
) -> Int32:
    """Apply one selected E4M3 scale byte to four unpacked E2M1 values."""
    imm = "0x0000"
    if cutlass.const_expr(sf_byte_idx == 1):
        imm = "0x1111"
    elif cutlass.const_expr(sf_byte_idx == 2):
        imm = "0x2222"
    elif cutlass.const_expr(sf_byte_idx == 3):
        imm = "0x3333"
    broadcast_sf = cute.arch.inline_ptx(
        f"prmt.b32 {{$w0}}, {{$r0}}, {{$r0}}, {imm};",
        write_only_types=[Int32],
        read_only_args=[packed_sf],
    )
    return _mul_e2m1x4_e4m3x4(packed_fp4, broadcast_sf)


@dataclass(kw_only=True)
class TmemTransformedKvResource(DecodeGenResourceBase):
    """Pipeline transformed NVFP4 K/V through TMEM.

    The transform warp group dequantizes an unpacking-TMA NVFP4 stage from
    :class:`SmemKvResource` into E4M3 registers. K is stored token-major and V
    is transposed, so both BMMs consume the transformed tile as operand A.
    """

    _task_local_specs: ClassVar[tuple[tuple, ...]] = (
        (
            "kv_desc_slot",
            Int32,
            Int32(0),
            "TMEM address for transformed K consumed by QK MMA.",
        ),
        (
            "v_desc_0_slot",
            Int32,
            Int32(0),
            "TMEM address for transformed V consumed by VP MMA instance 0.",
        ),
        (
            "v_desc_1_slot",
            Int32,
            Int32(0),
            "TMEM address for transformed V consumed by VP MMA instance 1.",
        ),
    )
    cfg: Constexpr[FmhaDecodeConfig] = None
    src_smem_kv: Constexpr[SmemKvResource] = None
    _alloc: Constexpr[TmemAllocation | None] = None
    kv_desc_slot: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    v_desc_0_slot: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    v_desc_1_slot: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()

    def get_smem_requirements(self) -> list[SmemAllocation]:
        """Return no data allocation; the pipeline barriers still use SMEM."""
        return []

    def get_tmem_requirements(self) -> list[TmemAllocation]:
        """Allocate the staged transformed K/V TMEM ring."""
        if self._alloc is None:
            self._alloc = TmemAllocation(
                name=f"{self.name}",
                num_columns=(
                    self.cfg.tmem_transformed_kv_stage_cols
                    * self.pipeline_config.num_stages
                ),
            )
        return [self._alloc]

    @cute.jit
    def _create_initial_task_locals(
        self, context: ResourceContext | None = None
    ) -> ResourceVars:
        """Initialize the stage-address task locals."""
        _ = context
        return {
            "kv_desc": Int32(0),
            "v_desc_0": Int32(0),
            "v_desc_1": Int32(0),
        }

    @cute.jit
    def _create_work_tile_task_locals(
        self, context: ResourceContext | None = None
    ) -> ResourceVars:
        """Initialize one work tile's stage-address task locals."""
        _ = context
        return {
            "kv_desc": Int32(0),
            "v_desc_0": Int32(0),
            "v_desc_1": Int32(0),
        }

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_load_state(self, stage_info: StageInfo) -> None:
        """Initialize producer-side TMEM stage state."""
        self._create_initial_task_locals(stage_info.context)

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_descriptor_state(self, stage_info: StageInfo) -> None:
        """Initialize consumer-side TMEM stage state."""
        self._create_initial_task_locals(stage_info.context)

    @cute.jit
    def _source_stage(self) -> tuple[cutlass.Array, cutlass.Array]:
        """Return unpacking-TMA NVFP4 and scale arrays for the raw stage."""
        cfg = self.cfg
        src_stage_idx = self.src_smem_kv.state_src.consumer_work_stage
        src_byte_base = self.src_smem_kv._smem_base_kv.subview(
            src_stage_idx * cfg.smem_kv_tile_elements
        )
        src_bytes = cutlass.Array(
            src_byte_base.data_ptr(),
            dtype=Uint8,
            shape=(cfg.smem_kv_tile_elements,),
            addrspace=3,
        )
        return src_bytes, self.src_smem_kv._sf_stage_base(src_stage_idx)

    @cute.jit
    def _destination_stage_addr(self, stage_info: StageInfo) -> Int32:
        """Return the raw TMEM address of one transformed pipeline stage."""
        task_cache = _decode_gen_task_cache(stage_info)
        return (
            task_cache[_TASK_CACHE_TMEM_BASE_OFFSET]
            + Int32(self._alloc.offset)
            + stage_info.stage_idx * Int32(self.cfg.tmem_transformed_kv_stage_cols)
        )

    @cute.jit
    def _store_k(
        self, stage_info: StageInfo, head_dim_stage_idx: Constexpr[int]
    ) -> None:
        """Dequantize K with unpacking LDSM and store token-major TMEM."""
        cfg = self.cfg
        task_cache = _decode_gen_task_cache(stage_info)
        transform_thread_idx = task_cache[_TASK_CACHE_WARP_GRP_THREAD_IDX] - Int32(
            (cfg.transform_kv_warp_idx % 4) * 32
        )
        warp_idx = transform_thread_idx >> Int32(5)
        lane_idx = transform_thread_idx & Int32(31)
        src_bytes, src_sf = self._source_stage()
        src_sf_qwords = cutlass.Array(
            src_sf.data_ptr(),
            dtype=Int64,
            shape=(cfg.smem_kv_sf_tile_bytes // 8,),
            addrspace=3,
        )

        # Each lane owns four K rows. H256 stages the complete 16-byte SF row,
        # so select the qword matching the active 128-column head slice.
        sf_rows = []
        sf_row_base = warp_idx * Int32(32) + lane_idx // Int32(4)
        sf_qwords_per_row = Int32(cfg.smem_kv_sf_bytes_per_token // 8)
        for row_group in cutlass.range_constexpr(4):
            sf_row = sf_row_base + Int32(row_group * 8)
            sf_qword = src_sf_qwords[
                sf_row * sf_qwords_per_row + Int32(head_dim_stage_idx)
            ]
            sf_rows.append((Int32(sf_qword), Int32(sf_qword >> Int64(32))))

        thread_row = warp_idx * Int32(32) + lane_idx
        swizzle_mask = (thread_row & Int32(7)) * Int32(16)
        stage_addr = self._destination_stage_addr(stage_info)
        for col_stage in cutlass.range_constexpr(8):
            dst_regs = []
            for row_block in cutlass.range_constexpr(2):
                local_offset = Int32(
                    row_block * 16 * cfg.head_dim_kv_stage + col_stage * 16
                )
                src_ptr = src_bytes.data_ptr() + (
                    thread_row * Int32(cfg.head_dim_kv_stage)
                    + (local_offset ^ swizzle_mask)
                )
                unpacked = prims.ldmatrix(
                    src_ptr,
                    2,
                    prims.MMALayout.ROW,
                    shape=prims.LoadShape.M8N16,
                    src_format=prims.LoadSrcFormat.B4X16_P64,
                )
                sf_word_idx = col_stage // 4
                sf_byte_idx = col_stage % 4
                dst_regs.append(
                    _mul_e2m1x4_e4m3x4_bx(
                        unpacked[0],
                        sf_rows[row_block * 2][sf_word_idx],
                        sf_byte_idx,
                    )
                )
                dst_regs.append(
                    _mul_e2m1x4_e4m3x4_bx(
                        unpacked[1],
                        sf_rows[row_block * 2 + 1][sf_word_idx],
                        sf_byte_idx,
                    )
                )

            tmem_addr = stage_addr + Int32(col_stage * 4)
            prims.tcgen05_st(
                "16x128b",
                prims.make_tmem_ptr(tmem_addr, Int32),
                cutlass.Vector.from_elements(tuple(dst_regs[:2]), dtype=Int32),
            )
            prims.tcgen05_st(
                "16x128b",
                prims.make_tmem_ptr(tmem_addr + (Int32(16) << Int32(16)), Int32),
                cutlass.Vector.from_elements(tuple(dst_regs[2:]), dtype=Int32),
            )
        cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def _store_v(
        self, stage_info: StageInfo, head_dim_stage_idx: Constexpr[int]
    ) -> None:
        """Dequantize and transpose V with LDSM into TMEM operand A."""
        cfg = self.cfg
        task_cache = _decode_gen_task_cache(stage_info)
        transform_thread_idx = task_cache[_TASK_CACHE_WARP_GRP_THREAD_IDX] - Int32(
            (cfg.transform_kv_warp_idx % 4) * 32
        )
        warp_idx = transform_thread_idx >> Int32(5)
        lane_idx = transform_thread_idx & Int32(31)
        src_bytes, src_sf = self._source_stage()
        src_sf_words = cutlass.Array(
            src_sf.data_ptr(),
            dtype=Int32,
            shape=(cfg.smem_kv_sf_tile_bytes // 4,),
            addrspace=3,
        )

        swizzle_mask = (lane_idx & Int32(7)) * Int32(16)
        stage_addr = self._destination_stage_addr(stage_info)
        sf_bytes_per_token = cfg.smem_kv_sf_bytes_per_token
        sf_head_stage_offset = head_dim_stage_idx * (cfg.head_dim_kv_stage // 16)
        for token_stage in cutlass.range_constexpr(8):
            # V scale factors use TRT-LLM's 4-token interleaved layout, so the
            # two packed QMUL4 vectors are adjacent 32-bit words.
            sf_byte_offset = (
                Int32(token_stage * 16 * sf_bytes_per_token)
                + (lane_idx % Int32(4)) * Int32(4 * sf_bytes_per_token)
                + Int32(sf_head_stage_offset * 4)
                + warp_idx * Int32(8)
            )
            sf_word_idx = sf_byte_offset // Int32(4)
            sf_packed0 = src_sf_words[sf_word_idx]
            sf_packed1 = src_sf_words[sf_word_idx + Int32(1)]

            dst_regs = []
            for col_block in cutlass.range_constexpr(2):
                local_offset = Int32(
                    token_stage * 16 * cfg.head_dim_kv_stage + col_block * 16
                ) + warp_idx * Int32(32)
                src_ptr = src_bytes.data_ptr() + (
                    lane_idx * Int32(cfg.head_dim_kv_stage)
                    + (local_offset ^ swizzle_mask)
                )
                unpacked = prims.ldmatrix(
                    src_ptr,
                    2,
                    prims.MMALayout.COL,
                    shape=prims.LoadShape.M16N16,
                    src_format=prims.LoadSrcFormat.B4X16_P64,
                )
                packed_sf = sf_packed0 if col_block == 0 else sf_packed1
                dst_regs.append(_mul_e2m1x4_e4m3x4(unpacked[0], packed_sf))
                dst_regs.append(_mul_e2m1x4_e4m3x4(unpacked[1], packed_sf))

            tmem_addr = stage_addr + Int32(token_stage * 4)
            prims.tcgen05_st(
                "16x128b",
                prims.make_tmem_ptr(tmem_addr, Int32),
                cutlass.Vector.from_elements(tuple(dst_regs[:2]), dtype=Int32),
            )
            prims.tcgen05_st(
                "16x128b",
                prims.make_tmem_ptr(tmem_addr + (Int32(16) << Int32(16)), Int32),
                cutlass.Vector.from_elements(tuple(dst_regs[2:]), dtype=Int32),
            )
        cute.arch.fence_view_async_tmem_store()

    @producer_work
    @cute.jit
    def transform_k0(
        self, stage_info: StageInfo, *, head_dim_stage_idx: Constexpr[int]
    ) -> None:
        """Transform K instance 0 into the acquired TMEM stage."""
        self._store_k(stage_info, head_dim_stage_idx)

    @producer_work
    @cute.jit
    def transform_k1(
        self, stage_info: StageInfo, *, head_dim_stage_idx: Constexpr[int]
    ) -> None:
        """Transform K instance 1 into the acquired TMEM stage."""
        self._store_k(stage_info, head_dim_stage_idx)

    @producer_work
    @cute.jit
    def transform_v0(
        self, stage_info: StageInfo, *, head_dim_stage_idx: Constexpr[int]
    ) -> None:
        """Transform V instance 0 into the acquired TMEM stage."""
        self._store_v(stage_info, head_dim_stage_idx)

    @producer_work
    @cute.jit
    def transform_v1(
        self, stage_info: StageInfo, *, head_dim_stage_idx: Constexpr[int]
    ) -> None:
        """Transform V instance 1 into the acquired TMEM stage."""
        self._store_v(stage_info, head_dim_stage_idx)

    @consumer_work(returns=kv_desc_slot)
    @cute.jit
    def k_desc_0(self, stage_info: StageInfo) -> Int32:
        """Publish the transformed K TMEM address for QK instance 0."""
        return self._destination_stage_addr(stage_info)

    @consumer_work(returns=kv_desc_slot)
    @cute.jit
    def k_desc_1(self, stage_info: StageInfo) -> Int32:
        """Publish the transformed K TMEM address for QK instance 1."""
        return self._destination_stage_addr(stage_info)

    @consumer_work(returns=v_desc_0_slot)
    @cute.jit
    def v_desc_0(self, stage_info: StageInfo) -> Int32:
        """Publish the transformed V TMEM address for VP MMA instance 0."""
        return self._destination_stage_addr(stage_info)

    @consumer_work(returns=v_desc_1_slot)
    @cute.jit
    def v_desc_1(self, stage_info: StageInfo) -> Int32:
        """Publish the transformed V TMEM address for VP MMA instance 1."""
        return self._destination_stage_addr(stage_info)
