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

"""TMEM accumulator resource for BatchedGemm C tiles.

TmemCResource owns the accumulator TMEM columns used by the MMA task and
publishes TMEM-to-register (T2R) fragments to the epilogue task. The dataflow is
MMA writes -> UmmaAsync commit -> epilogue wait -> T2R reads -> GMEM
store. Scale-factor S2T copies may be fused into the MMA producer path or
provided by separate TmemSf resources; in both cases TmemC only owns the C
allocation and derives SF addresses from the allocator layout when needed.

The allocation lifetime is one task resource lifetime. The TS TMEM allocator
assigns the base column, TmemCResource requests one or more staged C windows,
and the task schedule releases the resource after the epilogue has consumed the
fragment. Tile256 max-overlap kernels use two local logical C windows that
share a physical chunk; producer and consumer post hooks ping-pong those local
indices after the matching pipeline release/commit.

Public hooks:
  - producer_work: mma, mma_fused_sf, mma_separate_sf, mma_cast_a.
  - consumer_work: consumer_work.
  - ts_*_work(work_attrs=WorkAttr.AUXILIARY): overlap window setup and preloaded T2R fragments.
"""

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
    consumer_work as consumer_work_decorator,
    producer_work,
)

from .batched_gemm_config import (
    BatchedGemmConfig,
    MX_MMA_FORMAT_E2M1,
    MX_MMA_FORMAT_E4M3,
    SfLayout,
    TMEM_SF_UTCCP_COLS_PER_COPY,
    TMEM_SF_PACK_SIZE_BYTES,
)
from .gmem_ab_resources import nonnegative_div, nonnegative_mod
from cutlass.experimental import primitives as prims

Constexpr = cutlass.Constexpr


def _sfb_s2t_desc_increment(smem_layout: int, copy_idx: int) -> int:
    """Return the generated SFB SMEM-descriptor increment for one S2T copy."""
    if smem_layout == int(SfLayout.R128c4):
        return 32 * copy_idx
    return 32 * (copy_idx // 2) + 128 * (copy_idx % 2)


def _mx_dtype_from_format(format_id: int) -> type:
    """Translate a generated MX MMA format id to the cutlass operand dtype.

    Accepted format ids are MX_MMA_FORMAT_E4M3 (0), which maps to
    cutlass.Float8E4M3FN, and MX_MMA_FORMAT_E2M1 (5), which maps to
    cutlass.Float4E2M1FN. Unsupported ids raise ValueError during configuration so
    callers do not silently build an MMA descriptor for the wrong input dtype.
    """
    if format_id == MX_MMA_FORMAT_E4M3:
        return cutlass.Float8E4M3FN
    if format_id == MX_MMA_FORMAT_E2M1:
        return cutlass.Float4E2M1FN
    raise ValueError(f"Unsupported MX MMA dtype format: {format_id}")


@dataclass(kw_only=True)
class TmemCResource(MemoryResource):
    """TMEM accumulator written by MMA and read by epilogue T2R.

    Args:
        cfg: Compile-time BatchedGemmConfig controlling dtype, tile shape,
            staging count, overlap mode, and SF ownership.
        _alloc_c: Optional prebuilt TmemAllocation. When omitted, the resource
            allocates enough columns for all C accumulator stages.

    The resource is a MemoryResource with a TmemAllocation requirement. Its
    producer hooks issue tcgen05 MMA instructions for BF16, FP8, NVFP4, MXFP4,
    and CastA-backed modes. Its consumer hooks read the staged accumulator into
    Float32 register tensors for the epilogue. Edge cases are handled at
    configuration time: unsupported MX format ids raise ValueError, tile256
    overlap requires the generated 512-column layout, and dead config branches
    still declare type-stable TS variables.
    """

    cfg: Constexpr[BatchedGemmConfig]
    acc_tmem_ptr: Any = None
    sfa_tmem_addr_base: Any = None
    sfb_tmem_addr_base: Any = None
    tmem_raw_addr: Any = None
    idesc: Any = None
    scale_d: Any = None
    _alloc_c: Constexpr[Optional[TmemAllocation]] = None
    t2r_rmem: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    t2r_rmem_1: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    t2r_output_call_idx: Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )

    def __post_init__(self):
        if self._alloc_c is None:
            if self.cfg.use_tile256_tmem_overlap and self.cfg.num_epilogue_warps == 4:
                cols_per_stage = 2 * self.cfg.tile_n - self.cfg.epi_tile_n
            else:
                cols_per_stage = self.cfg.tmem_c_cols_per_stage
            total_cols = cols_per_stage * self.cfg.num_stages_tmem_acc
            if (
                self.cfg.has_scale_factors
                and not self.cfg.uses_unfused_tmem_sf_copy
                and not self.cfg.use_tile256_tmem_overlap
            ):
                # Fused S2T writes scale factors immediately after the
                # accumulator. Reserve those columns as part of this resource;
                # no standalone TmemSf resources participate in the task graph
                # (and therefore the allocator) on this path.
                total_cols += self.cfg.tmem_sfa_cols + self.cfg.tmem_sfb_cols
            self._alloc_c = TmemAllocation(
                f"{self.name}_tmem_c",
                num_columns=total_cols,
            )
        self.t2r_rmem = TaskLocalVariable(
            dtype=cutlass.Float32,
            default_factory=self._t2r_rmem_default,
            docs="Primary TMEM-to-register fragment for epilogue stores.",
        )
        self.t2r_rmem_1 = TaskLocalVariable(
            dtype=cutlass.Float32,
            default_factory=self._t2r_rmem_1_default,
            docs="Secondary TMEM-to-register fragment for swapAB stores.",
        )
        self.t2r_output_call_idx = TaskLocalVariable(
            dtype=Int32,
            default=Int32(0),
            docs="Logical epilogue output subtile index.",
        )

    @cute.jit
    def _t2r_rmem_default(self):
        t2r_init, _ = self._t2r_default_values()
        return t2r_init

    @cute.jit
    def _t2r_rmem_1_default(self):
        _, t2r_1_init = self._t2r_default_values()
        return t2r_1_init

    def get_tmem_requirements(self):
        return [self._alloc_c]

    @cute.jit
    def _t2r_default_values(self):
        if cutlass.const_expr(self.cfg.is_swap_ab):
            swap_t2r_repx = max(1, self.cfg.epi_tile_n // 8)
            t2r_init = cutlass.vector.full([swap_t2r_repx * 4], 0.0, cutlass.Float32)
            t2r_1_init = cutlass.vector.full([swap_t2r_repx * 4], 0.0, cutlass.Float32)
        else:
            epi_t2r_repx = self.cfg.epi_tile_n // 4
            t2r_init = cutlass.vector.full([max(1, epi_t2r_repx)], 0.0, cutlass.Float32)
            t2r_1_init = cutlass.vector.full([1], 0.0, cutlass.Float32)
        return t2r_init, t2r_1_init

    @cute.jit
    def _init_tmem_state(self, stage_info: StageInfo) -> None:
        context = stage_info.context
        self.tmem_raw_addr = context.tmem_ptr_i32.load()
        self.acc_tmem_ptr = prims.make_tmem_ptr(
            self.tmem_raw_addr + self._alloc_c.offset, cutlass.Float32
        )
        self.scale_d = cutlass.Boolean(False)
        # Initialise mutable per-iteration state as instance attributes
        if cutlass.const_expr(
            self.cfg.use_tile256_tmem_overlap and self.cfg.num_epilogue_warps == 4
        ):
            self._mma_local_idx = Int32(0)
            self._epi_local_idx = Int32(0)
        # Compute SFA/SFB TMEM base addresses for fused S2T+MMA
        if cutlass.const_expr(self.cfg.has_scale_factors):
            base_col = self.tmem_raw_addr & 0xFFFF
            base_row = self.tmem_raw_addr >> 16
            # TMEM allocator order: C first (at offset 0), then SFA, SFB
            # Matches nvfp4_gemm reference: acc at base, SFA after acc
            # The fused SF reservation is included in _alloc_c, but SFA starts
            # after the accumulator region rather than after the whole
            # allocation.
            c_total_cols = self._tmem_c_cols_per_stage() * self.cfg.num_stages_tmem_acc
            sf_stage_mult = (
                self.cfg.num_stages_tmem_sfa
                if self.cfg.uses_unfused_tmem_sf_copy
                else 1
            )
            if cutlass.const_expr(self.cfg.use_tile256_tmem_overlap):
                # Generated Tile256 fused-UTCCP kernels reserve the final
                # 64-column TMEM window for scale factors: SFA at 448 and
                # SFB at 464 when the allocation is 512 columns.  MXFP8/MXFP4
                # tile256 only needs 4 SFA columns, but SFB still has to keep
                # the generated 16-column boundary; placing it at 452 feeds
                # malformed B scales to tcgen05_mma_block_scale.
                sfa_col = base_col + self.cfg.tmem_total_cols - 64
            else:
                sfa_col = base_col + c_total_cols
            sfb_col_stride = self.cfg.tmem_sfa_cols * sf_stage_mult
            if cutlass.const_expr(self.cfg.use_tile256_tmem_overlap):
                sfb_col_stride = max(16, sfb_col_stride)
            sfb_col = sfa_col + sfb_col_stride
            self.sfa_tmem_addr_base = (base_row << 16) | sfa_col
            self.sfb_tmem_addr_base = (base_row << 16) | sfb_col
        # Build MMA instruction descriptor
        if cutlass.const_expr(self.cfg.uses_f16_mma):
            # BF16: standard MMA descriptor (tcgen05_mma F16)
            self.idesc = prims.Tcgen05InstrDesc.build(
                a_dtype=cutlass.BFloat16,
                b_dtype=cutlass.BFloat16,
                c_dtype=cutlass.Float32,
                n_dim=self.cfg.mma_n,
                m_dim=self.cfg.mma_m,
            )
        elif cutlass.const_expr(self.cfg.is_fp8_mma):
            # Plain FP8 per-tensor GEMM: tcgen05_mma F8F6F4 with E4M3 inputs
            # and FP32 accumulators.
            self.idesc = prims.Tcgen05InstrDesc.build(
                a_dtype=cutlass.Float8E4M3FN,
                b_dtype=cutlass.Float8E4M3FN,
                c_dtype=cutlass.Float32,
                n_dim=self.cfg.mma_n,
                m_dim=self.cfg.mma_m,
            )
        elif cutlass.const_expr(self.cfg.is_mx_mma):
            self.idesc = prims.Tcgen05MxInstrDesc.build(
                a_dtype=_mx_dtype_from_format(self.cfg.mx_a_format),
                b_dtype=_mx_dtype_from_format(self.cfg.mx_b_format),
                scale_format=1,  # UE8M0
                n_dim=self.cfg.mma_n,
                m_dim=self.cfg.mma_m,
            )
        else:
            # FP4 NVF4: block-scaled MMA descriptor (tcgen05_mma_block_scale
            # MXF4NVF4). NVF4 uses E4M3 scale factors (scale_format=0, UE4M3);
            # MXF4 (OCP MX) would use E8M0 (scale_format=1).
            self.idesc = prims.Tcgen05MxInstrDesc.build(
                a_dtype=cutlass.Float4E2M1FN,
                b_dtype=cutlass.Float4E2M1FN,
                scale_format=0,  # UE4M3
                n_dim=self.cfg.mma_n,
                m_dim=self.cfg.mma_m,
            )

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_accumulator_state(self, stage_info: StageInfo) -> None:
        self._init_tmem_state(stage_info)

    @consumer_work_decorator(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_epilogue_state(self, stage_info: StageInfo) -> None:
        self._init_tmem_state(stage_info)

    def _tmem_c_cols_per_stage(self) -> int:
        if self.cfg.use_tile256_tmem_overlap and self.cfg.num_epilogue_warps == 4:
            return 2 * self.cfg.tile_n - self.cfg.epi_tile_n
        return self.cfg.tmem_c_cols_per_stage

    @cute.jit
    def _deepseek_second_mma_desc_a(self, smem_a_stage_ptr):
        offset_bytes = Int64(
            (self.cfg.tile_m // 2) * self.cfg.tile_k * self.cfg.dtype_a_smem_bits // 8
        )
        stage_base = cutlass.Array(
            smem_a_stage_ptr + offset_bytes,
            dtype=cutlass.Uint8,
            shape=(self.cfg.num_bytes_a_per_stage,),
            addrspace=3,
        )
        return prims.Tcgen05SmemDesc.build(
            stage_base,
            leading_byte_offset=16384,
            stride_byte_offset=1024,
            layout=2,
        )

    @producer_work
    @cute.jit
    def mma(
        self,
        stage_info: StageInfo,
        *,
        desc_a_mma_base: prims.Tcgen05SmemDesc,
        smem_a_stage_ptr: Int64,
        desc_b_mma_base: prims.Tcgen05SmemDesc,
        smem_b_stage_ptr: Int64,
    ) -> None:
        self._mma_impl(
            stage_info,
            desc_a_mma_base,
            desc_b_mma_base,
            smem_a_stage_ptr=smem_a_stage_ptr,
        )

    @producer_work
    @cute.jit
    def mma_fused_sf(
        self,
        stage_info: StageInfo,
        *,
        desc_a_mma_base: prims.Tcgen05SmemDesc,
        smem_a_stage_ptr: Int64,
        desc_b_mma_base: prims.Tcgen05SmemDesc,
        smem_b_stage_ptr: Int64,
        desc_a_s2t_base: prims.Tcgen05SmemDesc,
        smem_sfa_stage_ptr: Int64,
        desc_b_s2t_base: prims.Tcgen05SmemDesc,
    ) -> None:
        self._mma_impl(
            stage_info,
            desc_a_mma_base,
            desc_b_mma_base,
            smem_a_stage_ptr=smem_a_stage_ptr,
            desc_a_s2t_base=desc_a_s2t_base,
            desc_b_s2t_base=desc_b_s2t_base,
        )

    @producer_work
    @cute.jit
    def mma_separate_sf(
        self,
        stage_info: StageInfo,
        *,
        desc_a_mma_base: prims.Tcgen05SmemDesc,
        smem_a_stage_ptr: Int64,
        desc_b_mma_base: prims.Tcgen05SmemDesc,
        smem_b_stage_ptr: Int64,
        sfa_stage_col_offset: cutlass.Int32,
        sfb_stage_col_offset: cutlass.Int32,
    ) -> None:
        self._mma_impl(
            stage_info,
            desc_a_mma_base,
            desc_b_mma_base,
            smem_a_stage_ptr=smem_a_stage_ptr,
            sfa_stage_col_offset=sfa_stage_col_offset,
            sfb_stage_col_offset=sfb_stage_col_offset,
        )

    @producer_work
    @cute.jit
    def mma_cast_a(
        self,
        stage_info: StageInfo,
        *,
        desc_b_mma_base: prims.Tcgen05SmemDesc,
        smem_b_stage_ptr: Int64,
        tmem_cast_a_addr: Int32,
    ) -> None:
        self._mma_impl(
            stage_info, Int64(0), desc_b_mma_base, tmem_cast_a_addr=tmem_cast_a_addr
        )

    @cute.jit
    def _mma_impl(
        self,
        stage_info: StageInfo,
        desc_a_mma_base,
        desc_b_mma_base,
        *,
        smem_a_stage_ptr=None,
        desc_a_s2t_base=None,
        desc_b_s2t_base=None,
        sfa_stage_col_offset=None,
        sfb_stage_col_offset=None,
        tmem_cast_a_addr=None,
    ) -> None:
        """Execute MMA (accumulates across K-tiles). Branches on BF16 vs FP4."""
        desc_a_mma = desc_a_mma_base
        desc_b_mma = desc_b_mma_base
        stage_col_offset = stage_info.stage_idx * self._tmem_c_cols_per_stage()
        if cutlass.const_expr(
            self.cfg.use_tile256_tmem_overlap and self.cfg.num_epilogue_warps == 4
        ):
            stage_col_offset += self._mma_local_idx * Int32(
                self.cfg.tile_n - self.cfg.epi_tile_n
            )
        acc_tmem_ptr = prims.make_tmem_ptr(
            self.tmem_raw_addr + self._alloc_c.offset + stage_col_offset,
            cutlass.Float32,
        )

        num_kblocks = self.cfg.tile_k // self.cfg.mma_k
        is_first_ktile = stage_info.loop_offset == Int32(0)

        if cutlass.const_expr(self.cfg.uses_plain_mma):
            # BF16 and plain FP8 paths: standard tcgen05_mma, no block scales.
            if cutlass.const_expr(self.cfg.is_fp8_mma):
                mma_kind = prims.Tcgen05MMAKind.F8F6F4
            else:
                mma_kind = prims.Tcgen05MMAKind.F16
            for kblock_idx in cutlass.range_constexpr(num_kblocks):
                if cutlass.const_expr(
                    (self.cfg.is_fp8_mma and self.cfg.tile_k > 128)
                    or (not self.cfg.is_fp8_mma and self.cfg.tile_k > 64)
                ):
                    # SM100 swizzled MMA descriptors are linear inside one
                    # generated K-box group: BF16 uses 64-wide TMA boxes and
                    # FP8 uses 128-wide TMA boxes. The next group starts at
                    # the next swizzled TMA box in SMEM.
                    k_minor = kblock_idx % 4
                    k_major = kblock_idx // 4
                    desc_a_increment = k_major * 1024 + k_minor * 2
                    if cutlass.const_expr(self.cfg.is_fp8_mma):
                        b_kbox = 128
                    else:
                        b_kbox = 64
                    b_group_stride = max(
                        64,
                        (
                            b_kbox
                            * self.cfg.num_bytes_b_smem_per_stage
                            // self.cfg.tile_k
                        )
                        >> 4,
                    )
                    desc_b_increment = k_major * b_group_stride + k_minor * 2
                else:
                    desc_a_increment = 2 * kblock_idx
                    desc_b_increment = 2 * kblock_idx
                desc_b = desc_b_mma + desc_b_increment
                if cutlass.const_expr(self.cfg.has_cast_a):
                    desc_a = prims.make_tmem_ptr(
                        tmem_cast_a_addr + Int32(kblock_idx * 8), cutlass.Int32
                    )
                else:
                    desc_a = desc_a_mma + desc_a_increment

                if cutlass.const_expr(kblock_idx == 0):
                    if cutlass.const_expr(self.cfg.has_deepseek_fp8):
                        accumulate = cutlass.Boolean(False)
                    else:
                        accumulate = ~is_first_ktile
                else:
                    accumulate = cutlass.Boolean(True)

                if prims.elect_sync():
                    prims.tcgen05_mma(
                        mma_kind,
                        self.cfg.cta_group,
                        acc_tmem_ptr,
                        desc_a,
                        desc_b,
                        self.idesc,
                        accumulate,
                    )

            if cutlass.const_expr(self.cfg.has_deepseek_fp8_two_epilogue):
                desc_a_mma_1 = self._deepseek_second_mma_desc_a(smem_a_stage_ptr)
                acc_tmem_ptr_1 = prims.make_tmem_ptr(
                    self.tmem_raw_addr
                    + self._alloc_c.offset
                    + stage_col_offset
                    + Int32(0x100000),
                    cutlass.Float32,
                )
                for kblock_idx in cutlass.range_constexpr(num_kblocks):
                    if cutlass.const_expr(self.cfg.tile_k > 128):
                        k_minor = kblock_idx % 4
                        k_major = kblock_idx // 4
                        desc_a_increment = k_major * 1024 + k_minor * 2
                        b_group_stride = max(
                            64,
                            (
                                128
                                * self.cfg.num_bytes_b_smem_per_stage
                                // self.cfg.tile_k
                            )
                            >> 4,
                        )
                        desc_b_increment = k_major * b_group_stride + k_minor * 2
                    else:
                        desc_a_increment = 2 * kblock_idx
                        desc_b_increment = 2 * kblock_idx
                    desc_a = desc_a_mma_1 + desc_a_increment
                    desc_b = desc_b_mma + desc_b_increment

                    if cutlass.const_expr(kblock_idx == 0):
                        accumulate = cutlass.Boolean(False)
                    else:
                        accumulate = cutlass.Boolean(True)

                    if prims.elect_sync():
                        prims.tcgen05_mma(
                            mma_kind,
                            self.cfg.cta_group,
                            acc_tmem_ptr_1,
                            desc_a,
                            desc_b,
                            self.idesc,
                            accumulate,
                        )

        else:
            # FP4 path: block-scaled MMA
            if cutlass.const_expr(not self.cfg.uses_unfused_tmem_sf_copy):
                # Fused S2T+MMA: do S2T copy here, then MMA.
                s2t_shape, s2t_multicast = prims.S2TCopyMode.S2T_32x128b_WARPX4
                num_sfa_iters = self.cfg.tmem_sfa_cols // TMEM_SF_UTCCP_COLS_PER_COPY
                num_sfb_iters = self.cfg.tmem_sfb_cols // TMEM_SF_UTCCP_COLS_PER_COPY

                for s2t_idx in cutlass.range_constexpr(num_sfa_iters):
                    sfa_tmem_addr = (
                        self.sfa_tmem_addr_base + s2t_idx * TMEM_SF_UTCCP_COLS_PER_COPY
                    )
                    sfa_tmem_ptr = prims.make_tmem_ptr(sfa_tmem_addr, cutlass.Int32)
                    desc_s2t = desc_a_s2t_base + 32 * s2t_idx
                    if prims.elect_sync():
                        prims.tcgen05_cp(
                            s2t_shape,
                            sfa_tmem_ptr,
                            desc_s2t,
                            group=self.cfg.cta_group,
                            multicast=s2t_multicast,
                        )

                for s2t_idx in cutlass.range_constexpr(num_sfb_iters):
                    if cutlass.const_expr(self.cfg.use_tile256_tmem_overlap):
                        sfb_col_stride = self.cfg.tmem_sf_col_stride("b")
                        sfb_k_groups = self.cfg.tmem_sfb_cols // sfb_col_stride
                        # Tile256 SFB descriptors are chunk-major; TMEM is
                        # grouped by four SF K-vectors.
                        sfb_tmem_delta = (s2t_idx % sfb_k_groups) * sfb_col_stride + (
                            s2t_idx // sfb_k_groups
                        ) * TMEM_SF_UTCCP_COLS_PER_COPY
                        increment_s2t = 32 * s2t_idx
                    else:
                        sfb_tmem_delta = s2t_idx * TMEM_SF_UTCCP_COLS_PER_COPY
                        increment_s2t = _sfb_s2t_desc_increment(
                            self.cfg.smem_sfb_layout, s2t_idx
                        )
                    sfb_tmem_addr = self.sfb_tmem_addr_base + sfb_tmem_delta
                    sfb_tmem_ptr = prims.make_tmem_ptr(sfb_tmem_addr, cutlass.Int32)
                    desc_s2t = desc_b_s2t_base + increment_s2t
                    if prims.elect_sync():
                        prims.tcgen05_cp(
                            s2t_shape,
                            sfb_tmem_ptr,
                            desc_s2t,
                            group=self.cfg.cta_group,
                            multicast=s2t_multicast,
                        )
                prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
                cute.arch.fence_view_async_tmem_store()
            else:
                # Separate CopySf: SF already in TMEM from CopySf tasks.
                # Read stage offsets from TmemSfRoute consumer variables.
                pass

            # Block-scaled MMA with scale factors in TMEM
            if cutlass.const_expr(self.cfg.uses_unfused_tmem_sf_copy):
                # SF stage offsets from separate CopySf pipeline
                sfa_offset = sfa_stage_col_offset
                sfb_offset = sfb_stage_col_offset
            else:
                sfa_offset = Int32(0)
                sfb_offset = Int32(0)

            for kblock_idx in cutlass.range_constexpr(num_kblocks):
                sfa_sf_vec_idx = (
                    kblock_idx * self.cfg.mma_k
                ) // self.cfg.input_sf_block_size_a
                sfb_sf_vec_idx = (
                    kblock_idx * self.cfg.mma_k
                ) // self.cfg.input_sf_block_size_b
                sfa_sf_group_idx = sfa_sf_vec_idx // TMEM_SF_PACK_SIZE_BYTES
                sfb_sf_group_idx = sfb_sf_vec_idx // TMEM_SF_PACK_SIZE_BYTES
                sfa_sf_id = sfa_sf_vec_idx % TMEM_SF_PACK_SIZE_BYTES
                sfb_sf_id = sfb_sf_vec_idx % TMEM_SF_PACK_SIZE_BYTES
                sfa_tmem_addr = (
                    self.sfa_tmem_addr_base
                    + sfa_offset
                    + sfa_sf_group_idx * self.cfg.tmem_sf_col_stride("a")
                )
                sfb_tmem_addr = (
                    self.sfb_tmem_addr_base
                    + sfb_offset
                    + sfb_sf_group_idx * self.cfg.tmem_sf_col_stride("b")
                )
                sfa_tmem_ptr = prims.make_tmem_ptr(sfa_tmem_addr, cutlass.Int32)
                sfb_tmem_ptr = prims.make_tmem_ptr(sfb_tmem_addr, cutlass.Int32)

                if cutlass.const_expr(self.cfg.is_mx_mma and self.cfg.tile_k > 128):
                    # MX descriptors are linear inside one 128-wide K slice:
                    # four mmaK=32 instructions share one set of SF IDs.
                    k_minor = kblock_idx % 4
                    k_major = kblock_idx // 4
                    desc_a_increment = k_major * 1024 + k_minor * 2
                    b_stage_bytes = self.cfg.num_bytes_b_smem_per_stage
                    if cutlass.const_expr(
                        self.cfg.has_cluster
                        and self.cfg.is_swap_ab
                        and self.cfg.has_tma_route
                        and not self.cfg.split_b_across_ctas
                    ):
                        b_stage_bytes //= self.cfg.cluster_m
                    desc_b_group_stride = max(
                        64,
                        (128 * b_stage_bytes // self.cfg.tile_k) >> 4,
                    )
                    desc_b_increment = k_major * desc_b_group_stride + k_minor * 2
                elif cutlass.const_expr(self.cfg.tile_k > 256):
                    # SM100 swizzled FP4 MMA descriptors are linear only
                    # within a 256-wide K group. At kblock 4, generated
                    # kernels jump to the next swizzle group instead of using
                    # desc += 8.
                    k_minor = kblock_idx % 4
                    k_major = kblock_idx // 4
                    desc_a_increment = k_major * 1024 + k_minor * 2
                    b_stage_bytes = self.cfg.num_bytes_b_smem_per_stage
                    if cutlass.const_expr(
                        self.cfg.has_cluster
                        and self.cfg.is_swap_ab
                        and self.cfg.has_tma_route
                        and not self.cfg.split_b_across_ctas
                    ):
                        b_stage_bytes //= self.cfg.cluster_m
                    desc_b_group_stride = max(
                        64,
                        (256 * b_stage_bytes // self.cfg.tile_k) >> 4,
                    )
                    desc_b_increment = k_major * desc_b_group_stride + k_minor * 2
                else:
                    desc_a_increment = 2 * kblock_idx
                    desc_b_increment = 2 * kblock_idx
                desc_a = desc_a_mma + desc_a_increment
                desc_b = desc_b_mma + desc_b_increment

                if cutlass.const_expr(kblock_idx == 0):
                    accumulate = ~is_first_ktile
                else:
                    accumulate = cutlass.Boolean(True)

                if prims.elect_sync():
                    if cutlass.const_expr(self.cfg.is_mx_mma):
                        prims.tcgen05_mma_block_scale(
                            prims.MMABlockScaleKind.MXF8F6F4,
                            self.cfg.cta_group,
                            acc_tmem_ptr,
                            desc_a,
                            desc_b,
                            self.idesc.set_sf_ids(
                                a_sf_id=sfa_sf_id,
                                b_sf_id=sfb_sf_id,
                            ),
                            enable_input_d=accumulate,
                            scale_a=sfa_tmem_ptr,
                            scale_b=sfb_tmem_ptr,
                            scale_vec_size=prims.Tcgen05MMABlockScale.BLOCK32,
                        )
                    else:
                        prims.tcgen05_mma_block_scale(
                            prims.MMABlockScaleKind.MXF4NVF4,
                            self.cfg.cta_group,
                            acc_tmem_ptr,
                            desc_a,
                            desc_b,
                            self.idesc,
                            enable_input_d=accumulate,
                            scale_a=sfa_tmem_ptr,
                            scale_b=sfb_tmem_ptr,
                            scale_vec_size=prims.Tcgen05MMABlockScale.BLOCK16,
                        )

        # Do not drain MMA stores here. The following UmmaAsync
        # producer_commit emits the tcgen05 mbarrier commit that makes the
        # accumulator visible to the epilogue.

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def advance_mma_overlap_window(self, stage_info: StageInfo) -> None:
        """Flip the local D-window index after publishing C."""
        if cutlass.const_expr(
            self.cfg.use_tile256_tmem_overlap and self.cfg.num_epilogue_warps == 4
        ):
            self._mma_local_idx = self._mma_local_idx ^ Int32(1)

    @consumer_work_decorator(returns=(t2r_rmem, t2r_rmem_1, t2r_output_call_idx))
    @cute.jit
    def consumer_work(
        self,
        stage_info: StageInfo,
        *,
        subtile_idx: cutlass.Constexpr[int],
    ) -> tuple[cutlass.Float32, cutlass.Float32, Int32]:
        t2r_rmem, t2r_rmem_1, t2r_output_call_idx = self._consumer_work_impl(
            stage_info, subtile_idx
        )
        return t2r_rmem, t2r_rmem_1, t2r_output_call_idx

    @consumer_work_decorator(
        returns=(t2r_rmem, t2r_rmem_1, t2r_output_call_idx),
        work_attrs=WorkAttr.AUXILIARY,
    )
    @cute.jit
    def load_overlap_subtile(
        self,
        stage_info: StageInfo,
        *,
        subtile_idx: cutlass.Constexpr[int],
    ) -> tuple[cutlass.Float32, cutlass.Float32, Int32]:
        if cutlass.const_expr(
            self.cfg.use_tile256_tmem_overlap and self.cfg.num_epilogue_warps == 4
        ):
            t2r_rmem, t2r_rmem_1, t2r_output_call_idx = self._consumer_work_impl(
                stage_info, subtile_idx + Int32(1)
            )
            if cutlass.const_expr(
                subtile_idx == self.cfg.tile_n // self.cfg.epi_tile_n - 2
            ):
                self._epi_local_idx = self._epi_local_idx ^ Int32(1)
            return t2r_rmem, t2r_rmem_1, t2r_output_call_idx
        t2r_rmem, t2r_rmem_1 = self._t2r_default_values()
        return t2r_rmem, t2r_rmem_1, Int32(0)

    @cute.jit
    def _consumer_work_impl(self, stage_info: StageInfo, logical_call_idx):
        """T2R: load TMEM accumulator sub-tile into registers.

        non-swapAB: 32x32b, 1 reg per TMEM column, epi_t2r_repx cols per call.
        swapAB: 16x256b, 4 regs per 8 columns, two 16-row loads.
        """
        base_col = self.tmem_raw_addr & 0xFFFF
        base_row = self.tmem_raw_addr >> 16
        stage_col_offset = stage_info.stage_idx * self._tmem_c_cols_per_stage()

        warp_idx = cute.arch.warp_idx()
        warp_in_group = warp_idx % self.cfg.num_epilogue_warps
        warp_in_group = cute.arch.make_warp_uniform(warp_in_group)

        if cutlass.const_expr(self.cfg.is_swap_ab):
            # 16x256b: each 4-warp epilogue subgroup handles one epilogueTileN
            # column chunk.  Tile256 generated kernels use two 4-warp groups,
            # so one schedule call covers 128 columns without over-unrolling.
            warpgroup_count = max(1, self.cfg.num_epilogue_warps // 4)
            warpgroup_idx = nonnegative_div(warp_in_group, 4)
            cols_per_call = self.cfg.epi_tile_n * warpgroup_count
            if cutlass.const_expr(self.cfg.has_deepseek_fp8_two_epilogue):
                cols_per_call = self.cfg.epi_tile_n
            call_idx_for_tmem = Int32(logical_call_idx)
            call_idx_for_output = Int32(logical_call_idx)
            if cutlass.const_expr(
                self.cfg.use_tile256_tmem_overlap and self.cfg.num_epilogue_warps == 4
            ):
                num_epilogue_tiles_n = self.cfg.tile_n // self.cfg.epi_tile_n
                actual_idx_n = logical_call_idx + warpgroup_idx
                # The tile256 max-overlap path ping-pongs two logical C tiles
                # through a 448-column physical window:
                #   local set 0 -> physical chunks 0,1,2,3
                #   local set 1 -> physical chunks 3,4,5,6
                # Chunk 3 is shared.  We read that shared chunk before release,
                # then read the remaining chunks after release while MMA can
                # write the next set.  Wrapping local set 1 back to chunks
                # 0,1,2,3 corrupts multi-N FC1 shapes because epilogue stores
                # stale C data for output tiles 1..3.
                offset_idx_n = actual_idx_n + Int32(num_epilogue_tiles_n - 1)
                wrapped_idx_n = nonnegative_mod(offset_idx_n, num_epilogue_tiles_n)
                local_set_is_zero = self._epi_local_idx == Int32(0)
                # Do not assign Python locals inside staged if/else here.
                # PyIR keeps the original local value live after the staged
                # region, so the generated PTX silently falls back to the
                # non-overlap 0,1,2,3 T2R order.  Use SSA selects instead.
                # selp.s32 (inline PTX): pick a if pred (.pred reg) else b.
                call_idx_for_tmem = prims.inline_ptx_hl(
                    "selp.s32 {$w0}, {$r0}, {$r1}, {$r2};",
                    write_only_types=[Int32],
                    read_only_args=[wrapped_idx_n, offset_idx_n, local_set_is_zero],
                )
                call_idx_for_output = prims.inline_ptx_hl(
                    "selp.s32 {$w0}, {$r0}, {$r1}, {$r2};",
                    write_only_types=[Int32],
                    read_only_args=[wrapped_idx_n, actual_idx_n, local_set_is_zero],
                )

            col_offset = call_idx_for_tmem * cols_per_call
            if cutlass.const_expr(not self.cfg.has_deepseek_fp8_two_epilogue):
                col_offset += warpgroup_idx * Int32(self.cfg.epi_tile_n)
            col_id = base_col + self._alloc_c.offset + stage_col_offset + col_offset
            addr = (base_row << 16) | col_id

            shape = "16x256b"
            if cutlass.const_expr(self.cfg.has_deepseek_fp8_two_epilogue):
                addr = addr + warpgroup_idx * Int32(0x100000)
                tmem = prims.make_tmem_ptr(addr, cutlass.Float32)
                swap_t2r_repx = max(1, self.cfg.epi_tile_n // 8)
                slice0 = prims.tcgen05_ld(shape, tmem, num=swap_t2r_repx)
                slice1 = cutlass.vector.full([swap_t2r_repx * 4], 0.0, cutlass.Float32)
                prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
                cute.arch.fence_view_async_tmem_load()
                return (slice0, slice1, call_idx_for_output)
            tmem0 = prims.make_tmem_ptr(addr, cutlass.Float32)
            tmem1 = prims.make_tmem_ptr(addr + 0x100000, cutlass.Float32)
            swap_t2r_repx = max(1, self.cfg.epi_tile_n // 8)
            slice0 = prims.tcgen05_ld(shape, tmem0, num=swap_t2r_repx)  # 4 * repx FP32
            slice1 = prims.tcgen05_ld(shape, tmem1, num=swap_t2r_repx)  # 4 * repx FP32
            prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
            cute.arch.fence_view_async_tmem_load()
            return (slice0, slice1, call_idx_for_output)
        else:
            # 32x32b: 1 reg per column
            epi_t2r_repx = self.cfg.epi_tile_n // 4
            row_offset = warp_in_group * 32
            col_offset = logical_call_idx * epi_t2r_repx
            col_id = base_col + self._alloc_c.offset + stage_col_offset + col_offset
            current_addr = ((base_row + row_offset) << 16) | col_id

            shape = "32x32b"
            tmem = prims.make_tmem_ptr(current_addr, cutlass.Float32)
            c_rmem = prims.tcgen05_ld(shape, tmem, num=max(1, epi_t2r_repx))
            prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
            cute.arch.fence_view_async_tmem_load()
            return (c_rmem, cutlass.vector.full([1], 0.0, cutlass.Float32), Int32(0))
