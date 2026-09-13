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

"""SMEM resources for operands A and B (including TMA gather)."""

from dataclasses import dataclass
from typing import Any, Optional

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint16

from cutlass.experimental.task_scheduling.memory import SmemAllocation
from cutlass.experimental.task_scheduling.resources import (
    WorkAttr,
    MemoryResource,
    StageInfo,
    TaskLocalVariable,
    consumer_work,
    producer_work,
)

from .batched_gemm_config import BatchedGemmConfig, DType
from .gmem_ab_resources import nonnegative_div, nonnegative_mod
from cutlass.experimental import primitives as prims

Constexpr = cutlass.Constexpr


@dataclass(kw_only=True)
class SmemAResource(MemoryResource):
    """SMEM staging for operand A (TMA producer, UMMA consumer)."""

    cfg: Constexpr[BatchedGemmConfig]
    tma_a_desc: Any = None
    smem_buf: Any = None
    _alloc_a: Constexpr[Optional[SmemAllocation]] = None
    desc_a_mma_base: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    smem_a_stage_ptr: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()

    def __post_init__(self):
        if self._alloc_a is None:
            self._alloc_a = SmemAllocation(
                f"{self.name}_a",
                size_bytes=self.cfg.num_bytes_a_per_stage * self.cfg.num_stages_a,
                alignment=1024,
            )
        self.desc_a_mma_base = TaskLocalVariable(
            dtype=Int64, default=Int64(0), docs="MMA descriptor for operand A."
        )
        self.smem_a_stage_ptr = TaskLocalVariable(
            dtype=Int64, default=Int64(0), docs="SMEM stage pointer for operand A."
        )

    def get_smem_requirements(self):
        return [self._alloc_a]

    @cute.jit
    def _init_smem_state(self, stage_info: StageInfo) -> None:
        context = stage_info.context
        self.smem_buf = cutlass.Array(
            context.smem_base.data_ptr() + self._alloc_a.offset,
            dtype=cutlass.Uint8,
            shape=(self._alloc_a.size_bytes,),
            addrspace=3,
        )

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_load_state(self, stage_info: StageInfo) -> None:
        self._init_smem_state(stage_info)

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_mma_state(self, stage_info: StageInfo) -> None:
        self._init_smem_state(stage_info)

    @producer_work
    @cute.jit
    def load_a_tile(
        self,
        stage_info: StageInfo,
        *,
        coord_a_k: cutlass.Int32,
        coord_a_mn: cutlass.Int32,
        coord_a_l: cutlass.Int32,
        expert_idx: Int32,
        mn_limit: Int32,
    ) -> None:
        """TMA load A into SMEM.

        Large K tiles are split into TMA boxes: BF16 uses 64-wide
        boxes, and FP4 uses 256-wide boxes. A single 512-wide FP4 TMA leaves
        the completion barrier unsatisfied on device.
        """
        stage_base = self.smem_buf.subview(
            self.cfg.num_bytes_a_per_stage * stage_info.stage_idx
        )
        if prims.elect_sync():
            if cutlass.const_expr(self.cfg.uses_block_major_k_weight_a):
                a_box_k = self.cfg.tile_k
            elif cutlass.const_expr(self.cfg.dtype_a_kind == int(DType.BF16)):
                if cutlass.const_expr(self.cfg.use_bf16_kbox_tma_a):
                    a_box_k = self.cfg.tile_k
                else:
                    a_box_k = 64
            elif cutlass.const_expr(self.cfg.has_cast_a):
                a_box_k = self.cfg.tile_k
            elif cutlass.const_expr(
                self.cfg.dtype_a_kind
                in (int(DType.MXE2M1), int(DType.MXE4M3), int(DType.E4M3))
            ):
                a_box_k = 128
            else:
                a_box_k = 256

            if cutlass.const_expr(self.cfg.tile_k > a_box_k):
                box_bytes = a_box_k * self.cfg.tile_m * self.cfg.dtype_a_smem_bits // 8
                for bi in cutlass.range_constexpr(self.cfg.tile_k // a_box_k):
                    self._tma_load(
                        stage_base.subview(bi * box_bytes),
                        self.tma_a_desc,
                        self._coords_for_load(
                            coord_a_k + Int32(bi * a_box_k),
                            coord_a_mn,
                            coord_a_l,
                            mn_limit,
                        ),
                        stage_info.barrier,
                    )
            else:
                self._tma_load(
                    stage_base,
                    self.tma_a_desc,
                    self._coords_for_load(coord_a_k, coord_a_mn, coord_a_l, mn_limit),
                    stage_info.barrier,
                )

    @cute.jit
    def _coords_for_load(self, coord_k, coord_mn, coord_l, mn_limit):
        if cutlass.const_expr(self.cfg.uses_block_major_k_weight_a):
            block_k = Int32(self.cfg.block_major_k_elems)
            return (coord_k % block_k, coord_mn, coord_k // block_k, coord_l)
        if cutlass.const_expr(self.cfg.use_bf16_kbox_tma_a):
            return (Int32(0), coord_mn, coord_k // Int32(64), coord_l)
        if cutlass.const_expr(self.cfg.use_tma_oob_opt_a):
            return self._tma_oob_coords(
                coord_k, coord_mn, mn_limit, Int32(0), self.cfg.tile_m
            )
        return (coord_k, coord_mn, coord_l)

    @cute.jit
    def _tma_oob_coords(self, coord_k, coord_mn, mn_limit, cta_row_offset, tile_mn):
        large_n = Int32(0x40000000)
        tile_mn_i32 = Int32(tile_mn)
        limit_mod = mn_limit % tile_mn_i32
        dist = (tile_mn_i32 - limit_mod) % tile_mn_i32
        return (
            coord_k,
            cta_row_offset + dist,
            large_n,
            coord_mn - dist + large_n,
        )

    @cute.jit
    def _tma_load(self, smem_dst, tma_desc, coords, barrier):
        """Single-CTA or cluster TMA load.

        For 2-CTA cluster (2×1): each CTA loads its own data.
        multicast_mask = (1 << cta_rank) targets only the local CTA.
        Both CTAs issue TMA independently with cta_group_2.
        """
        if cutlass.const_expr(self.cfg.has_cast_a):
            smem_fp4_dst = cutlass.inttoptr(
                smem_dst.data_ptr().toint(),
                3,
                cutlass.Float4E2M1FN,
            )
            self._emit_tma_load(smem_fp4_dst, tma_desc, coords, barrier)
        else:
            self._emit_tma_load(smem_dst, tma_desc, coords, barrier)

    @cute.jit
    def _emit_tma_load(self, smem_dst, tma_desc, coords, barrier):
        if cutlass.const_expr(self.cfg.has_cast_a):
            prims.cp_async_bulk_tensor_shared_cluster_global(
                smem_dst,
                tma_desc,
                coords,
                barrier,
                [],
            )
        elif cutlass.const_expr(self.cfg.has_cluster):
            cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
            mcast_mask = Int32(1) << cta_rank
            lead_cta_rank = nonnegative_div(cta_rank, self.cfg.cluster_m) * Int32(
                self.cfg.cluster_m
            )
            barrier = prims.mapa(barrier, lead_cta_rank)
            # TODO-NVVM: remove cvta_to once mbarrier intrinsics accept AS7.
            barrier = prims.cvta_to(barrier, prims.CvtaSpace.SHARED)
            prims.cp_async_bulk_tensor_shared_cluster_global(
                smem_dst,
                tma_desc,
                coords,
                barrier,
                [],
                multicast_mask=mcast_mask,
                group=self.cfg.cta_group,
            )
        else:
            prims.cp_async_bulk_tensor_shared_cta_global(
                smem_dst,
                tma_desc,
                coords,
                barrier,
            )

    @consumer_work(returns=(desc_a_mma_base, smem_a_stage_ptr))
    @cute.jit
    def build_mma_desc_a(self, stage_info: StageInfo) -> tuple[Int64, Int64]:
        """Build SMEM descriptor for MMA A operand (s128b swizzle)."""
        return self._build_mma_desc_a_impl(stage_info.stage_idx)

    @consumer_work(returns=(desc_a_mma_base, smem_a_stage_ptr))
    @cute.jit
    def build_mma_desc_a_at_stage(
        self, stage_info: StageInfo, *, pipeline_stage_idx
    ) -> tuple[Int64, Int64]:
        """Build the A descriptor at the stage reported ready by the proxy."""
        return self._build_mma_desc_a_impl(pipeline_stage_idx)

    @cute.jit
    def _build_mma_desc_a_impl(self, stage_idx) -> tuple[Int64, Int64]:
        stage_base = self.smem_buf.subview(self.cfg.num_bytes_a_per_stage * stage_idx)
        if cutlass.const_expr(self.cfg.is_fp8_mma):
            desc_a_mma_base = prims.Tcgen05SmemDesc.build(
                stage_base,
                leading_byte_offset=16384,
                stride_byte_offset=1024,
                layout=2,
            )
        elif cutlass.const_expr(self.cfg.uses_f16_mma):
            desc_a_mma_base = prims.Tcgen05SmemDesc.build(
                stage_base,
                leading_byte_offset=self.cfg.num_bytes_a_per_stage,
                stride_byte_offset=1024,
                layout=2,
            )
        else:
            desc_a_mma_base = prims.Tcgen05SmemDesc.build(
                stage_base,
                leading_byte_offset=16,
                stride_byte_offset=1024,
                layout=2,
            )
        return desc_a_mma_base, Int64(stage_base.data_ptr().toint())


@dataclass(kw_only=True)
class SmemBResource(MemoryResource):
    """SMEM staging for operand B (TMA producer, UMMA consumer)."""

    cfg: Constexpr[BatchedGemmConfig]
    tma_b_desc: Any = None
    smem_buf: Any = None
    _alloc_b: Constexpr[Optional[SmemAllocation]] = None
    desc_b_mma_base: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    smem_b_stage_ptr: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()

    def __post_init__(self):
        if self._alloc_b is None:
            self._alloc_b = SmemAllocation(
                f"{self.name}_b",
                size_bytes=self.cfg.num_bytes_b_smem_per_stage * self.cfg.num_stages_b,
                alignment=1024,
            )
        self.desc_b_mma_base = TaskLocalVariable(
            dtype=Int64, default=Int64(0), docs="MMA descriptor for operand B."
        )
        self.smem_b_stage_ptr = TaskLocalVariable(
            dtype=Int64, default=Int64(0), docs="SMEM stage pointer for operand B."
        )

    def get_smem_requirements(self):
        return [self._alloc_b]

    @cute.jit
    def _init_smem_state(self, stage_info: StageInfo) -> None:
        context = stage_info.context
        self.smem_buf = cutlass.Array(
            context.smem_base.data_ptr() + self._alloc_b.offset,
            dtype=cutlass.Uint8,
            shape=(self._alloc_b.size_bytes,),
            addrspace=3,
        )

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_load_state(self, stage_info: StageInfo) -> None:
        self._init_smem_state(stage_info)

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_mma_state(self, stage_info: StageInfo) -> None:
        self._init_smem_state(stage_info)

    @producer_work
    @cute.jit
    def load_b_tile(
        self,
        stage_info: StageInfo,
        *,
        coord_b_k: cutlass.Int32,
        coord_b_mn: cutlass.Int32,
        coord_b_l: cutlass.Int32,
        mn_limit: Int32,
    ) -> None:
        """TMA load B into SMEM using the generated K-box split."""
        split_b_across_ctas = cutlass.const_expr(self.cfg.split_b_across_ctas)
        if cutlass.const_expr(split_b_across_ctas and not self.cfg.use_tma_oob_opt_b):
            cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
            coord_b_mn = coord_b_mn + cta_rank * Int32(
                self.cfg.tile_n // self.cfg.cluster_m
            )
        stage_base = self.smem_buf.subview(
            self.cfg.num_bytes_b_smem_per_stage * stage_info.stage_idx
        )
        if prims.elect_sync():
            if cutlass.const_expr(self.cfg.uses_block_major_k_weight_b):
                b_box_k = self.cfg.tile_k
            elif cutlass.const_expr(self.cfg.dtype_b_kind == int(DType.BF16)):
                b_box_k = 64
            elif cutlass.const_expr(
                self.cfg.dtype_b_kind
                in (int(DType.MXE2M1), int(DType.MXE4M3), int(DType.E4M3))
            ):
                b_box_k = 128
            else:
                b_box_k = 256

            if cutlass.const_expr(self.cfg.tile_k > b_box_k):
                box_bytes = (
                    b_box_k * self.cfg.num_bytes_b_smem_per_stage // self.cfg.tile_k
                )
                for bi in cutlass.range_constexpr(self.cfg.tile_k // b_box_k):
                    self._tma_load(
                        stage_base.subview(bi * box_bytes),
                        self.tma_b_desc,
                        self._coords_for_load(
                            coord_b_k + Int32(bi * b_box_k),
                            coord_b_mn,
                            coord_b_l,
                            mn_limit,
                            split_b_across_ctas,
                        ),
                        stage_info.barrier,
                    )
            else:
                self._tma_load(
                    stage_base,
                    self.tma_b_desc,
                    self._coords_for_load(
                        coord_b_k,
                        coord_b_mn,
                        coord_b_l,
                        mn_limit,
                        split_b_across_ctas,
                    ),
                    stage_info.barrier,
                )

    @cute.jit
    def _coords_for_load(
        self, coord_k, coord_mn, coord_l, mn_limit, split_b_across_ctas
    ):
        if cutlass.const_expr(self.cfg.uses_block_major_k_weight_b):
            block_k = Int32(self.cfg.block_major_k_elems)
            return (coord_k % block_k, coord_mn, coord_k // block_k, coord_l)
        if cutlass.const_expr(self.cfg.use_tma_oob_opt_b):
            cta_row_offset = Int32(0)
            if cutlass.const_expr(split_b_across_ctas):
                cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
                cta_row_offset = cta_rank * Int32(self.cfg.tile_n // self.cfg.cluster_m)
            return self._tma_oob_coords(
                coord_k, coord_mn, mn_limit, cta_row_offset, self.cfg.tile_n
            )
        return (coord_k, coord_mn, coord_l)

    @cute.jit
    def _tma_oob_coords(self, coord_k, coord_mn, mn_limit, cta_row_offset, tile_mn):
        large_n = Int32(0x40000000)
        tile_mn_i32 = Int32(tile_mn)
        limit_mod = mn_limit % tile_mn_i32
        dist = (tile_mn_i32 - limit_mod) % tile_mn_i32
        return (
            coord_k,
            cta_row_offset + dist,
            large_n,
            coord_mn - dist + large_n,
        )

    @cute.jit
    def _tma_load(self, smem_dst, tma_desc, coords, barrier):
        """Single-CTA or cluster TMA load. Per-CTA mask for 2×1 cluster."""
        if cutlass.const_expr(self.cfg.has_cluster):
            cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
            mcast_mask = Int32(1) << cta_rank
            if cutlass.const_expr(self.cfg.split_b_across_ctas):
                lead_cta_rank = nonnegative_div(cta_rank, 2) * Int32(2)
                barrier = prims.mapa(barrier, lead_cta_rank)
                # TODO-NVVM: remove cvta_to once mbarrier intrinsics accept AS7.
                barrier = prims.cvta_to(barrier, prims.CvtaSpace.SHARED)
            prims.cp_async_bulk_tensor_shared_cluster_global(
                smem_dst,
                tma_desc,
                coords,
                barrier,
                [],
                multicast_mask=mcast_mask,
                group=self.cfg.cta_group,
            )
        else:
            prims.cp_async_bulk_tensor_shared_cta_global(
                smem_dst,
                tma_desc,
                coords,
                barrier,
            )

    @consumer_work(returns=(desc_b_mma_base, smem_b_stage_ptr))
    @cute.jit
    def build_mma_desc_b(self, stage_info: StageInfo) -> tuple[Int64, Int64]:
        """Build SMEM descriptor for MMA B operand (s128b swizzle)."""
        return self._build_mma_desc_b_impl(stage_info.stage_idx)

    @consumer_work(returns=(desc_b_mma_base, smem_b_stage_ptr))
    @cute.jit
    def build_mma_desc_b_at_stage(
        self, stage_info: StageInfo, *, pipeline_stage_idx
    ) -> tuple[Int64, Int64]:
        """Build the B descriptor at the stage reported ready by the proxy."""
        return self._build_mma_desc_b_impl(pipeline_stage_idx)

    @cute.jit
    def _build_mma_desc_b_impl(self, stage_idx) -> tuple[Int64, Int64]:
        stage_base = self.smem_buf.subview(
            self.cfg.num_bytes_b_smem_per_stage * stage_idx
        )
        if cutlass.const_expr(self.cfg.has_cast_a):
            # The CastA-generated BF16 MMA path uses the B
            # descriptor packing: tileN=8 -> 1024, tileN=16 -> 2048, etc.
            desc_b_mma_base = prims.Tcgen05SmemDesc.build(
                stage_base,
                leading_byte_offset=max(
                    1024, self.cfg.tile_n * (16384 // self.cfg.mma_m)
                ),
                stride_byte_offset=1024,
                layout=2,
            )
        elif cutlass.const_expr(self.cfg.is_fp8_mma):
            desc_b_mma_base = prims.Tcgen05SmemDesc.build(
                stage_base,
                leading_byte_offset=max(1024, self.cfg.tile_n * 64),
                stride_byte_offset=1024,
                layout=2,
            )
        elif cutlass.const_expr(self.cfg.uses_f16_mma):
            desc_b_mma_base = prims.Tcgen05SmemDesc.build(
                stage_base,
                leading_byte_offset=self.cfg.num_bytes_b_smem_per_stage,
                stride_byte_offset=1024,
                layout=2,
            )
        else:
            desc_b_mma_base = prims.Tcgen05SmemDesc.build(
                stage_base,
                leading_byte_offset=16,
                stride_byte_offset=1024,
                layout=2,
            )
        return desc_b_mma_base, Int64(stage_base.data_ptr().toint())


@dataclass(kw_only=True)
class SmemGatherResource(MemoryResource):
    """SMEM staging for activations loaded via LDGSTS gather.

    Replaces SmemA (non-swapAB) or SmemB (swapAB) when activations are
    scattered across experts and must be gathered via route map + cp.async.

    In swapAB mode: this is the B operand (activations, tokens × in_hidden).
    In non-swapAB mode: this is the A operand (activations, tokens × in_hidden).

    The route map (`ptrRouteMap`) maps logical token indices to physical GMEM
    positions. Each cp.async loads 16 bytes with s128b swizzle. Packed FP4 rows
    are addressed in bytes, so one 128-byte K box contains 256 logical elements.
    """

    cfg: Constexpr[BatchedGemmConfig]
    act_gmem_ptr: Any = None  # raw GMEM pointer to activations
    act_stride_bytes: Any = None  # stride in bytes between rows in GMEM
    route_map: Any = None  # make_array_view of route map tensor
    mn_limit: Any = None  # make_array_view of TRT absolute token end-row limits
    smem_buf: Any = None
    routed_rows: Any = None
    tile_limit: Any = None
    _alloc: Constexpr[Optional[SmemAllocation]] = None
    # Which MMA operand this maps to: "a" or "b"
    _operand: Constexpr[str] = "b"
    desc_a_mma_base: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    desc_b_mma_base: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    smem_a_stage_ptr: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    smem_b_stage_ptr: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()

    def __post_init__(self):
        if self._alloc is None:
            # Same size as the corresponding SmemA or SmemB
            if self._operand == "b":
                bytes_per_stage = self.cfg.num_bytes_b_smem_per_stage
            else:
                bytes_per_stage = self.cfg.num_bytes_a_per_stage
            self._alloc = SmemAllocation(
                f"{self.name}_gather",
                size_bytes=bytes_per_stage
                * (
                    self.cfg.num_stages_b
                    if self._operand == "b"
                    else self.cfg.num_stages_a
                ),
                alignment=1024,
            )
        self.desc_a_mma_base = TaskLocalVariable(
            dtype=Int64, default=Int64(0), docs="MMA descriptor for gathered A."
        )
        self.desc_b_mma_base = TaskLocalVariable(
            dtype=Int64, default=Int64(0), docs="MMA descriptor for gathered B."
        )
        self.smem_a_stage_ptr = TaskLocalVariable(
            dtype=Int64, default=Int64(0), docs="SMEM stage pointer for gathered A."
        )
        self.smem_b_stage_ptr = TaskLocalVariable(
            dtype=Int64, default=Int64(0), docs="SMEM stage pointer for gathered B."
        )

    def get_smem_requirements(self):
        return [self._alloc]

    @cute.jit
    def _init_smem_state(self, stage_info: StageInfo) -> None:
        context = stage_info.context
        self.smem_buf = cutlass.Array(
            context.smem_base.data_ptr() + self._alloc.offset,
            dtype=cutlass.Uint8,
            shape=(self._alloc.size_bytes,),
            addrspace=3,
        )
        is_operand_b = cutlass.const_expr(self._operand == "b")
        tile_rows = self.cfg.tile_n if is_operand_b else self.cfg.tile_m
        tile_rows_per_cta = (
            tile_rows // self.cfg.cluster_m
            if is_operand_b and self.cfg.has_cluster
            else tile_rows
        )
        copy_bytes = 16
        dtype_bits = (
            self.cfg.dtype_b_smem_bits if is_operand_b else self.cfg.dtype_a_smem_bits
        )
        box_k = 64 if dtype_bits == 16 else (256 if dtype_bits == 4 else 128)
        copies_per_row_box = box_k * dtype_bits // 8 // copy_bytes
        num_k_boxes = self.cfg.tile_k // box_k
        total_copies = tile_rows_per_cta * copies_per_row_box * num_k_boxes
        num_threads = self.cfg.num_gather_warps * 32
        copies_per_thread = (total_copies + num_threads - 1) // num_threads
        reuse_route_across_k_boxes = (
            num_threads == tile_rows_per_cta * copies_per_row_box
        )
        self.routed_rows = cutlass.Array(
            cutlass.Int32,
            1 if reuse_route_across_k_boxes else max(1, copies_per_thread),
            space=cutlass.AddressSpace.rmem,
        )
        self.tile_limit = Int32(0)

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_load_state(self, stage_info: StageInfo) -> None:
        self._init_smem_state(stage_info)

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_mma_state(self, stage_info: StageInfo) -> None:
        self._init_smem_state(stage_info)

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def prepare_gather_tile(self, stage_info: StageInfo) -> None:
        tile_coord_m, tile_coord_n, _ = stage_info.work_tile.tile_idx
        is_operand_b = cutlass.const_expr(self._operand == "b")
        tile_rows = self.cfg.tile_n if is_operand_b else self.cfg.tile_m
        if cutlass.const_expr(is_operand_b and self.cfg.has_cluster):
            tile_rows_per_cta = tile_rows // self.cfg.cluster_m
            cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
            cta_row_offset = cta_rank * Int32(tile_rows_per_cta)
        else:
            tile_rows_per_cta = tile_rows
            cta_row_offset = Int32(0)

        if cutlass.const_expr(is_operand_b):
            coord_tile = tile_coord_n * Int32(tile_rows)
            token_tile = tile_coord_n
        else:
            coord_tile = tile_coord_m * Int32(tile_rows)
            token_tile = tile_coord_m

        self.tile_limit = self._local_tile_limit(
            self.mn_limit.load(idx=token_tile, vector_size=1)[0],
            token_tile,
            tile_rows,
        )

        tidx, _, _ = cute.arch.thread_idx()
        local_tid = tidx - Int32(self.cfg.gather_warp_idx * 32)
        copy_bytes = 16
        dtype_bits = (
            self.cfg.dtype_b_smem_bits if is_operand_b else self.cfg.dtype_a_smem_bits
        )
        box_k = 64 if dtype_bits == 16 else (256 if dtype_bits == 4 else 128)
        copies_per_row_box = box_k * dtype_bits // 8 // copy_bytes
        num_k_boxes = self.cfg.tile_k // box_k
        total_copies = tile_rows_per_cta * copies_per_row_box * num_k_boxes
        num_threads = self.cfg.num_gather_warps * 32
        copies_per_thread = (total_copies + num_threads - 1) // num_threads
        reuse_route_across_k_boxes = cutlass.const_expr(
            num_threads == tile_rows_per_cta * copies_per_row_box
        )
        routed_row_count = 1 if reuse_route_across_k_boxes else copies_per_thread
        is_task_thread = (local_tid >= Int32(0)) & (local_tid < Int32(num_threads))
        for ci in cutlass.range_constexpr(routed_row_count):
            my_copy = local_tid + Int32(ci * num_threads)
            copy_in_box = nonnegative_mod(
                my_copy, tile_rows_per_cta * copies_per_row_box
            )
            row_in_tile = nonnegative_div(copy_in_box, copies_per_row_box)
            logical_row = coord_tile + cta_row_offset + row_in_tile
            is_valid = (
                is_task_thread
                & (my_copy < Int32(total_copies))
                & (cta_row_offset + row_in_tile < self.tile_limit)
            )
            if is_valid:
                self.routed_rows[ci] = self.route_map.load(
                    idx=logical_row, vector_size=1
                )[0]

    @cute.jit
    def _local_tile_limit(self, raw_limit, token_tile, tile_rows):
        """Convert TRT-LLM Gen absolute end-row limit to a local row count."""
        local_limit = raw_limit - token_tile * Int32(tile_rows)
        if local_limit < Int32(0):
            local_limit = Int32(0)
        if local_limit > Int32(tile_rows):
            local_limit = Int32(tile_rows)
        return local_limit

    @producer_work
    @cute.jit
    def load_a_tile(
        self,
        stage_info: StageInfo,
        *,
        coord_a_k: cutlass.Int32,
        coord_a_mn: cutlass.Int32,
        coord_a_l: cutlass.Int32,
        expert_idx: Int32,
        mn_limit: Int32,
    ) -> None:
        self._load_gather_tile_impl(stage_info, coord_a_k, coord_a_mn)

    @producer_work
    @cute.jit
    def load_b_tile(
        self,
        stage_info: StageInfo,
        *,
        coord_b_k: cutlass.Int32,
        coord_b_mn: cutlass.Int32,
        coord_b_l: cutlass.Int32,
        mn_limit: Int32,
    ) -> None:
        self._load_gather_tile_impl(stage_info, coord_b_k, coord_b_mn)

    @cute.jit
    def _load_gather_tile_impl(
        self, stage_info: StageInfo, coord_k, coord_tile_n
    ) -> None:
        """LDGSTS: gather activations from GMEM to SMEM via cp.async 16B.

        Uses cp.async.cg.shared.global for true LDGSTS (async copy from
        global to shared memory). Each thread copies 16 bytes. Packed inputs
        use bit-width-based byte offsets.
        Route map provides the physical GMEM row for each logical token.
        s128b swizzle applied to SMEM destination.
        """
        is_operand_b = cutlass.const_expr(self._operand == "b")
        bytes_per_stage = (
            self.cfg.num_bytes_b_smem_per_stage
            if is_operand_b
            else self.cfg.num_bytes_a_per_stage
        )
        tile_rows = self.cfg.tile_n if is_operand_b else self.cfg.tile_m
        if cutlass.const_expr(is_operand_b and self.cfg.has_cluster):
            tile_rows_per_cta = tile_rows // self.cfg.cluster_m
            cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
            cta_row_offset = cta_rank * Int32(tile_rows_per_cta)
        else:
            tile_rows_per_cta = tile_rows
            cta_row_offset = Int32(0)

        stage_base = self.smem_buf.subview(bytes_per_stage * stage_info.stage_idx)

        tidx, _, _ = cute.arch.thread_idx()
        local_tid = tidx - Int32(self.cfg.gather_warp_idx * 32)

        copy_bytes = 16
        dtype_bits = (
            self.cfg.dtype_b_smem_bits if is_operand_b else self.cfg.dtype_a_smem_bits
        )
        box_k = 64 if dtype_bits == 16 else (256 if dtype_bits == 4 else 128)
        row_box_bytes = box_k * dtype_bits // 8
        copies_per_row_box = row_box_bytes // copy_bytes
        num_k_boxes = self.cfg.tile_k // box_k
        total_copies = tile_rows_per_cta * copies_per_row_box * num_k_boxes
        num_threads = self.cfg.num_gather_warps * 32
        copies_per_thread = (total_copies + num_threads - 1) // num_threads
        reuse_route_across_k_boxes = cutlass.const_expr(
            num_threads == tile_rows_per_cta * copies_per_row_box
        )
        tile_limit = self.tile_limit
        is_task_thread = (local_tid >= Int32(0)) & (local_tid < Int32(num_threads))

        for ci in cutlass.range_constexpr(copies_per_thread):
            my_copy = local_tid + Int32(ci * num_threads)

            copies_per_box = tile_rows_per_cta * copies_per_row_box
            k_box = nonnegative_div(my_copy, copies_per_box)
            copy_in_box = nonnegative_mod(my_copy, copies_per_box)
            row_in_tile = nonnegative_div(copy_in_box, copies_per_row_box)
            col_chunk = nonnegative_mod(copy_in_box, copies_per_row_box)

            # SMEM destination with s128b swizzle
            smem_kbox_bytes = k_box * Int32(tile_rows_per_cta * row_box_bytes)
            smem_row_bytes = row_in_tile * Int32(row_box_bytes)
            smem_col_bytes = col_chunk * Int32(copy_bytes)
            smem_off_bytes = smem_kbox_bytes + smem_row_bytes + smem_col_bytes
            swizzle_mask = nonnegative_mod(row_in_tile, 8) * Int32(copy_bytes)
            swizzled_off = smem_off_bytes ^ swizzle_mask
            smem_ptr = stage_base.subview(swizzled_off)

            is_valid = is_task_thread & (my_copy < Int32(total_copies))
            if is_valid:
                in_bounds = cta_row_offset + row_in_tile < tile_limit
                if in_bounds:
                    phys_row = self.routed_rows[0 if reuse_route_across_k_boxes else ci]
                    # GMEM source: byte offset from activation base
                    gmem_col_bits = (coord_k + k_box * Int32(box_k)) * Int32(dtype_bits)
                    gmem_col_bytes = nonnegative_div(
                        gmem_col_bits, 8
                    ) + col_chunk * Int32(copy_bytes)
                    gmem_byte_off = phys_row * self.act_stride_bytes + gmem_col_bytes
                    gmem_ptr = self.act_gmem_ptr + gmem_byte_off
                    prims.inline_ptx_hl(
                        "cp.async.cg.shared.global.L2::128B [{$r0}], [{$r1}], 16;",
                        read_only_args=[smem_ptr.data_ptr(), gmem_ptr],
                    )

        # The following producer_commit uses cp.async.mbarrier.arrive for all
        # LDGSTS activation gathers.

    @consumer_work(returns=(desc_a_mma_base, smem_a_stage_ptr))
    @cute.jit
    def build_mma_desc_a(self, stage_info: StageInfo) -> tuple[Int64, Int64]:
        desc, stage_ptr = self._build_mma_desc_impl(stage_info.stage_idx)
        return desc, stage_ptr

    @consumer_work(returns=(desc_a_mma_base, smem_a_stage_ptr))
    @cute.jit
    def build_mma_desc_a_at_stage(
        self, stage_info: StageInfo, *, pipeline_stage_idx
    ) -> tuple[Int64, Int64]:
        desc, stage_ptr = self._build_mma_desc_impl(pipeline_stage_idx)
        return desc, stage_ptr

    @consumer_work(returns=(desc_b_mma_base, smem_b_stage_ptr))
    @cute.jit
    def build_mma_desc_b(self, stage_info: StageInfo) -> tuple[Int64, Int64]:
        desc, stage_ptr = self._build_mma_desc_impl(stage_info.stage_idx)
        return desc, stage_ptr

    @consumer_work(returns=(desc_b_mma_base, smem_b_stage_ptr))
    @cute.jit
    def build_mma_desc_b_at_stage(
        self, stage_info: StageInfo, *, pipeline_stage_idx
    ) -> tuple[Int64, Int64]:
        desc, stage_ptr = self._build_mma_desc_impl(pipeline_stage_idx)
        return desc, stage_ptr

    @cute.jit
    def _build_mma_desc_impl(self, stage_idx):
        """Build SMEM descriptor for MMA operand (s128b swizzle)."""
        is_operand_b = cutlass.const_expr(self._operand == "b")
        bytes_per_stage = (
            self.cfg.num_bytes_b_smem_per_stage
            if is_operand_b
            else self.cfg.num_bytes_a_per_stage
        )

        stage_base = self.smem_buf.subview(bytes_per_stage * stage_idx)
        if cutlass.const_expr(self.cfg.is_fp8_mma):
            if cutlass.const_expr(is_operand_b):
                leading_byte_offset = max(1024, self.cfg.tile_n * 64)
            else:
                leading_byte_offset = 16384
            desc = prims.Tcgen05SmemDesc.build(
                stage_base,
                leading_byte_offset=leading_byte_offset,
                stride_byte_offset=1024,
                layout=2,
            )
        elif cutlass.const_expr(self.cfg.uses_f16_mma):
            desc = prims.Tcgen05SmemDesc.build(
                stage_base,
                leading_byte_offset=bytes_per_stage,
                stride_byte_offset=1024,
                layout=2,
            )
        else:
            desc = prims.Tcgen05SmemDesc.build(
                stage_base,
                leading_byte_offset=16,
                stride_byte_offset=1024,
                layout=2,
            )
        return desc, Int64(stage_base.data_ptr().toint())


# Module-level TMA gather4 helpers used by SmemTmaGatherResource
@cute.jit
def _tma_gather4_cta(smem_dst, tma_desc, k_coord, row0, row1, row2, row3, barrier):
    """Emit cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4."""
    smem_ptr = smem_dst.data_ptr()
    tma_ptr = tma_desc.data_ptr() if hasattr(tma_desc, "data_ptr") else tma_desc
    bar_ptr = barrier.data_ptr()
    # {$rN} = operand ref.  {{ = literal brace for LLVM asm.
    # We need PTX: [{smem}], [{tma}, {c0, c1, c2, c3, c4}], [{mbar}]
    # Template: [{$r0}], [{$r1}, {$r2, $r3, $r4, $r5, $r6}], [{$r7}]
    #   → $r2 wraps in {}: need {{$r2, {$r3}, {$r4}, {$r5}, {$r6}}}
    #   BUT inline_ptx_hl treats every { as opening an operand ref.
    # Escape nested braces explicitly in the inline PTX template.
    prims.inline_ptx_hl(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
        ".mbarrier::complete_tx::bytes"
        " [{$r0}], [{$r1}, {{$r2}, {$r3}, {$r4}, {$r5}, {$r6}}], [{$r7}];",
        read_only_args=[smem_ptr, tma_ptr, k_coord, row0, row1, row2, row3, bar_ptr],
    )


@cute.jit
def _tma_gather4_cluster(
    smem_dst, tma_desc, k_coord, row0, row1, row2, row3, barrier, multicast_mask
):
    """Emit cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4."""
    smem_ptr = smem_dst.data_ptr()
    tma_ptr = tma_desc.data_ptr() if hasattr(tma_desc, "data_ptr") else tma_desc
    # Route completion to the lead CTA's mbarrier so the cluster converges on
    # mapa(leadCtaRank) before cluster gather4.
    lead_barrier = prims.cvta_to(prims.mapa(barrier, Int32(0)), prims.CvtaSpace.SHARED)
    bar_ptr = lead_barrier.data_ptr()
    mcast_mask_u16 = Uint16(multicast_mask)
    prims.inline_ptx_hl(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
        ".mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
        " [{$r0}], [{$r1}, {{$r2}, {$r3}, {$r4}, {$r5}, {$r6}}], [{$r7}], {$r8};",
        read_only_args=[
            smem_ptr,
            tma_ptr,
            k_coord,
            row0,
            row1,
            row2,
            row3,
            bar_ptr,
            mcast_mask_u16,
        ],
    )


@cute.jit
def _tma_gather4_cluster_x4(
    smem_dst0,
    smem_dst1,
    smem_dst2,
    smem_dst3,
    tma_desc,
    k_coord,
    row00,
    row01,
    row02,
    row03,
    row10,
    row11,
    row12,
    row13,
    row20,
    row21,
    row22,
    row23,
    row30,
    row31,
    row32,
    row33,
    barrier,
    multicast_mask,
):
    """Emit four cluster gather4 operations from one inline PTX block.

    Keeping the related gathers in one asm block lets ptxas materialize the
    uniform operands once and reuse them across the
    schedule for full TMA gather tiles.
    """
    smem_ptr0 = smem_dst0.data_ptr()
    smem_ptr1 = smem_dst1.data_ptr()
    smem_ptr2 = smem_dst2.data_ptr()
    smem_ptr3 = smem_dst3.data_ptr()
    tma_ptr = tma_desc.data_ptr() if hasattr(tma_desc, "data_ptr") else tma_desc
    lead_barrier = prims.cvta_to(prims.mapa(barrier, Int32(0)), prims.CvtaSpace.SHARED)
    bar_ptr = lead_barrier.data_ptr()
    mcast_mask_u16 = Uint16(multicast_mask)
    suffix = (
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
        ".mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
    )
    prims.inline_ptx_hl(
        f"{suffix} [{{$r0}}], [{{$r4}}, "
        f"{{{{$r5}}, {{$r6}}, {{$r7}}, {{$r8}}, {{$r9}}}}], [{{$r22}}], {{$r23}};\n"
        f"{suffix} [{{$r1}}], [{{$r4}}, "
        f"{{{{$r5}}, {{$r10}}, {{$r11}}, {{$r12}}, {{$r13}}}}], [{{$r22}}], {{$r23}};\n"
        f"{suffix} [{{$r2}}], [{{$r4}}, "
        f"{{{{$r5}}, {{$r14}}, {{$r15}}, {{$r16}}, {{$r17}}}}], [{{$r22}}], {{$r23}};\n"
        f"{suffix} [{{$r3}}], [{{$r4}}, "
        f"{{{{$r5}}, {{$r18}}, {{$r19}}, {{$r20}}, {{$r21}}}}], [{{$r22}}], {{$r23}};",
        read_only_args=[
            smem_ptr0,
            smem_ptr1,
            smem_ptr2,
            smem_ptr3,
            tma_ptr,
            k_coord,
            row00,
            row01,
            row02,
            row03,
            row10,
            row11,
            row12,
            row13,
            row20,
            row21,
            row22,
            row23,
            row30,
            row31,
            row32,
            row33,
            bar_ptr,
            mcast_mask_u16,
        ],
    )


@dataclass(kw_only=True)
class SmemTmaGatherResource(MemoryResource):
    """SMEM staging for activations loaded via TMA gather4.

    Uses ``cp.async.bulk.tensor.2d.tile::gather4`` to load 4 non-contiguous
    activation rows per instruction, addressed via the route map.

    The TMA descriptor is 2D ``(K, total_tokens)`` with ``box=(box_k, 1)``.
    Each gather4 call loads 4 rows × box_k elements.  For tile_n=N we need
    ``N / 4`` gather4 calls per K-box, and ``tile_k / box_k`` K-boxes.

    Replaces SmemA (non-swapAB) or SmemB (swapAB) for the TMA-routed path.
    """

    cfg: Constexpr[BatchedGemmConfig]
    tma_desc: Any = None  # 2D TMA descriptor: (K, total_tokens)
    route_map: Any = None  # make_array_view of Int32 route map
    mn_limit: Any = None  # make_array_view of TRT absolute token end-row limits
    smem_buf: Any = None
    routed_rows: Any = None
    _alloc: Constexpr[Optional[SmemAllocation]] = None
    _operand: Constexpr[str] = "b"  # "a" or "b" — which MMA operand this maps to
    desc_a_mma_base: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    desc_b_mma_base: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    smem_a_stage_ptr: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    smem_b_stage_ptr: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()

    def __post_init__(self):
        if self._alloc is None:
            is_b = self._operand == "b"
            tile_rows = self.cfg.tile_n if is_b else self.cfg.tile_m
            dtype_bits = (
                self.cfg.dtype_b_smem_bits if is_b else self.cfg.dtype_a_smem_bits
            )
            if self.cfg.has_cluster and is_b:
                tile_rows //= self.cfg.cluster_m
            nbytes = tile_rows * self.cfg.tile_k * dtype_bits // 8
            self._alloc = SmemAllocation(
                f"{self.name}_g4",
                size_bytes=nbytes
                * (self.cfg.num_stages_b if is_b else self.cfg.num_stages_a),
                alignment=1024,
            )
        self.desc_a_mma_base = TaskLocalVariable(
            dtype=Int64, default=Int64(0), docs="MMA descriptor for gathered A."
        )
        self.desc_b_mma_base = TaskLocalVariable(
            dtype=Int64, default=Int64(0), docs="MMA descriptor for gathered B."
        )
        self.smem_a_stage_ptr = TaskLocalVariable(
            dtype=Int64, default=Int64(0), docs="SMEM stage pointer for gathered A."
        )
        self.smem_b_stage_ptr = TaskLocalVariable(
            dtype=Int64, default=Int64(0), docs="SMEM stage pointer for gathered B."
        )

    def get_smem_requirements(self):
        return [self._alloc]

    @cute.jit
    def _init_smem_state(self, stage_info: StageInfo) -> None:
        context = stage_info.context
        self.smem_buf = cutlass.Array(
            context.smem_base.data_ptr() + self._alloc.offset,
            dtype=cutlass.Uint8,
            shape=(self._alloc.size_bytes,),
            addrspace=3,
        )
        is_b = self._operand == "b"
        tile_rows = (self.cfg.tile_n if is_b else self.cfg.tile_m) // (
            self.cfg.cluster_m if self.cfg.has_cluster and is_b else 1
        )
        num_gather4_per_kbox = tile_rows // 4
        num_load_warps = (
            self.cfg.num_load_b_warps if is_b else self.cfg.num_load_a_warps
        )
        route_cache_groups = max(
            1, (num_gather4_per_kbox + num_load_warps - 1) // num_load_warps
        )
        self.routed_rows = cutlass.Array(
            cutlass.Int32,
            route_cache_groups * 4,
            space=cutlass.AddressSpace.rmem,
        )

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_load_state(self, stage_info: StageInfo) -> None:
        self._init_smem_state(stage_info)

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_mma_state(self, stage_info: StageInfo) -> None:
        self._init_smem_state(stage_info)

    @cute.jit
    def _local_tile_limit(self, raw_limit, token_tile, tile_rows):
        """Convert TRT-LLM Gen absolute end-row limit to a local row count."""
        local_limit = raw_limit - token_tile * Int32(tile_rows)
        if local_limit < Int32(0):
            local_limit = Int32(0)
        if local_limit > Int32(tile_rows):
            local_limit = Int32(tile_rows)
        return local_limit

    @cute.jit
    def _load_routed_row_or_zero(self, route_idx, row_in_tile, tile_limit):
        routed_row = Int32(0)
        if row_in_tile < tile_limit:
            routed_row = self.route_map.load(idx=route_idx, vector_size=1)[0]
        return cute.arch.make_warp_uniform(routed_row)

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def prepare_gather_tile(self, stage_info: StageInfo) -> None:
        """Prefetch per-warp gather4 route rows once per output tile."""
        tile_coord_m, tile_coord_n, _ = stage_info.work_tile.tile_idx

        is_b = cutlass.const_expr(self._operand == "b")
        tile_rows = self.cfg.tile_n if is_b else self.cfg.tile_m
        if cutlass.const_expr(self.cfg.has_cluster and is_b):
            tile_rows_per_cta = tile_rows // self.cfg.cluster_m
            cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
            cta_row_offset = cta_rank * Int32(tile_rows_per_cta)
        else:
            tile_rows_per_cta = tile_rows
            cta_row_offset = Int32(0)

        if cutlass.const_expr(is_b):
            coord_mn = tile_coord_n * Int32(self.cfg.tile_n)
            token_tile = tile_coord_n
            num_load_warps = self.cfg.num_load_b_warps
            load_warp_idx = self.cfg.load_b_warp_idx
        else:
            coord_mn = tile_coord_m * Int32(self.cfg.tile_m)
            token_tile = tile_coord_m
            num_load_warps = self.cfg.num_load_a_warps
            load_warp_idx = self.cfg.load_a_warp_idx

        tile_limit = Int32(tile_rows)
        if cutlass.const_expr(self.mn_limit is not None):
            tile_limit = self._local_tile_limit(
                self.mn_limit.load(idx=token_tile, vector_size=1)[0],
                token_tile,
                tile_rows,
            )

        num_gather4_per_kbox = tile_rows_per_cta // 4
        route_cache_groups = max(
            1, (num_gather4_per_kbox + num_load_warps - 1) // num_load_warps
        )
        all_gathers_valid = num_gather4_per_kbox % num_load_warps == 0

        warp_idx = cute.arch.warp_idx()
        warp_in_task = warp_idx - Int32(load_warp_idx)
        warp_in_task = cute.arch.make_warp_uniform(warp_in_task)

        for wi in cutlass.range_constexpr(route_cache_groups):
            gi = warp_in_task + Int32(wi * num_load_warps)
            # The task dispatcher has already restricted execution to the
            # LoadB/LoadA warpgroup.  When the gather groups divide evenly over
            # the load warps, every participating warp owns a valid gather
            # group and the generated kernel does not need a per-issue
            # bounds guard.
            if cutlass.const_expr(all_gathers_valid):
                base_row = coord_mn + cta_row_offset + gi * Int32(4)
                row_base = cta_row_offset + gi * Int32(4)
                self.routed_rows[wi * 4] = self._load_routed_row_or_zero(
                    base_row, row_base, tile_limit
                )
                self.routed_rows[wi * 4 + 1] = self._load_routed_row_or_zero(
                    base_row + Int32(1), row_base + Int32(1), tile_limit
                )
                self.routed_rows[wi * 4 + 2] = self._load_routed_row_or_zero(
                    base_row + Int32(2), row_base + Int32(2), tile_limit
                )
                self.routed_rows[wi * 4 + 3] = self._load_routed_row_or_zero(
                    base_row + Int32(3), row_base + Int32(3), tile_limit
                )
            else:
                is_valid_gather = gi * Int32(4) < Int32(tile_rows_per_cta)
                if is_valid_gather:
                    base_row = coord_mn + cta_row_offset + gi * Int32(4)
                    row_base = cta_row_offset + gi * Int32(4)
                    self.routed_rows[wi * 4] = self._load_routed_row_or_zero(
                        base_row, row_base, tile_limit
                    )
                    self.routed_rows[wi * 4 + 1] = self._load_routed_row_or_zero(
                        base_row + Int32(1), row_base + Int32(1), tile_limit
                    )
                    self.routed_rows[wi * 4 + 2] = self._load_routed_row_or_zero(
                        base_row + Int32(2), row_base + Int32(2), tile_limit
                    )
                    self.routed_rows[wi * 4 + 3] = self._load_routed_row_or_zero(
                        base_row + Int32(3), row_base + Int32(3), tile_limit
                    )

    @producer_work
    @cute.jit
    def load_a_tile(
        self,
        stage_info: StageInfo,
        *,
        coord_a_k: cutlass.Int32,
        coord_a_mn: cutlass.Int32,
        coord_a_l: cutlass.Int32,
        expert_idx: Int32,
        mn_limit: Int32,
    ) -> None:
        self._load_tma_gather_tile_impl(stage_info, coord_a_k, coord_a_mn)

    @producer_work
    @cute.jit
    def load_b_tile(
        self,
        stage_info: StageInfo,
        *,
        coord_b_k: cutlass.Int32,
        coord_b_mn: cutlass.Int32,
        coord_b_l: cutlass.Int32,
        mn_limit: Int32,
    ) -> None:
        self._load_tma_gather_tile_impl(stage_info, coord_b_k, coord_b_mn)

    @cute.jit
    def _load_tma_gather_tile_impl(
        self, stage_info: StageInfo, coord_k, coord_mn
    ) -> None:
        """TMA gather4 load: fetch non-contiguous rows from GMEM.

        For non-swapAB the gather replaces SmemA (the activation operand)
        which has tile_m rows.  For swapAB it replaces SmemB with tile_n rows.
        """
        is_b = cutlass.const_expr(self._operand == "b")
        # Rows to gather = tile dimension of the operand this resource replaces
        tile_rows = self.cfg.tile_n if is_b else self.cfg.tile_m
        dtype_bits = self.cfg.dtype_b_smem_bits if is_b else self.cfg.dtype_a_smem_bits
        if cutlass.const_expr(self.cfg.has_cluster):
            cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        if cutlass.const_expr(self.cfg.has_cluster and is_b):
            tile_rows_per_cta = tile_rows // self.cfg.cluster_m
        else:
            tile_rows_per_cta = tile_rows
        bytes_per_stage = tile_rows_per_cta * self.cfg.tile_k * dtype_bits // 8

        stage_base = self.smem_buf.subview(bytes_per_stage * stage_info.stage_idx)

        # Each gather4 call loads 4 rows × box_k elements.
        # BF16: box_k = 64 (128 bytes/row for s128b).
        # FP8/MX: box_k = 128 (128 bytes/row for s128b).
        # FP4:  box_k = 256 (128 bytes/row for s128b).
        if cutlass.const_expr(self.cfg.is_bf16_mma):
            box_k = 64
        elif cutlass.const_expr(self.cfg.is_mx_mma or self.cfg.is_fp8_mma):
            box_k = 128
        else:
            box_k = 256

        num_k_boxes = self.cfg.tile_k // box_k
        num_gather4_per_kbox = tile_rows_per_cta // 4
        num_load_warps = (
            self.cfg.num_load_b_warps if is_b else self.cfg.num_load_a_warps
        )
        load_warp_idx = self.cfg.load_b_warp_idx if is_b else self.cfg.load_a_warp_idx
        warp_batches = (num_gather4_per_kbox + num_load_warps - 1) // num_load_warps
        all_gathers_valid = num_gather4_per_kbox % num_load_warps == 0

        warp_idx = cute.arch.warp_idx()
        warp_in_task = warp_idx - Int32(load_warp_idx)
        warp_in_task = cute.arch.make_warp_uniform(warp_in_task)

        if prims.elect_sync():
            for ki in cutlass.range_constexpr(num_k_boxes):
                k_offset = coord_k + Int32(ki * box_k)
                if cutlass.const_expr(
                    self.cfg.has_cluster and all_gathers_valid and warp_batches == 4
                ):
                    smem_offset0 = Int32(
                        ki * tile_rows_per_cta * 128
                    ) + warp_in_task * Int32(4 * 128)
                    smem_delta = Int32(num_load_warps * 4 * 128)
                    smem_dst0 = stage_base.subview(smem_offset0)
                    smem_dst1 = stage_base.subview(smem_offset0 + smem_delta)
                    smem_dst2 = stage_base.subview(smem_offset0 + smem_delta * Int32(2))
                    smem_dst3 = stage_base.subview(smem_offset0 + smem_delta * Int32(3))
                    mcast_mask = Int32(1) << cta_rank
                    _tma_gather4_cluster_x4(
                        smem_dst0,
                        smem_dst1,
                        smem_dst2,
                        smem_dst3,
                        self.tma_desc,
                        k_offset,
                        self.routed_rows[0],
                        self.routed_rows[1],
                        self.routed_rows[2],
                        self.routed_rows[3],
                        self.routed_rows[4],
                        self.routed_rows[5],
                        self.routed_rows[6],
                        self.routed_rows[7],
                        self.routed_rows[8],
                        self.routed_rows[9],
                        self.routed_rows[10],
                        self.routed_rows[11],
                        self.routed_rows[12],
                        self.routed_rows[13],
                        self.routed_rows[14],
                        self.routed_rows[15],
                        stage_info.barrier,
                        mcast_mask,
                    )
                else:
                    for wi in cutlass.range_constexpr(warp_batches):
                        gi = warp_in_task + Int32(wi * num_load_warps)
                        if cutlass.const_expr(all_gathers_valid):
                            route_cache_base = wi * 4
                            r0 = self.routed_rows[route_cache_base]
                            r1 = self.routed_rows[route_cache_base + 1]
                            r2 = self.routed_rows[route_cache_base + 2]
                            r3 = self.routed_rows[route_cache_base + 3]

                            # SMEM destination: each s128b row = 128 bytes.
                            # Clustered routes store only this CTA's row slice.
                            smem_offset = Int32(
                                ki * tile_rows_per_cta * 128
                            ) + gi * Int32(4 * 128)
                            smem_dst = stage_base.subview(smem_offset)

                            if cutlass.const_expr(self.cfg.has_cluster):
                                mcast_mask = Int32(1) << cta_rank
                                _tma_gather4_cluster(
                                    smem_dst,
                                    self.tma_desc,
                                    k_offset,
                                    r0,
                                    r1,
                                    r2,
                                    r3,
                                    stage_info.barrier,
                                    mcast_mask,
                                )
                            else:
                                _tma_gather4_cta(
                                    smem_dst,
                                    self.tma_desc,
                                    k_offset,
                                    r0,
                                    r1,
                                    r2,
                                    r3,
                                    stage_info.barrier,
                                )
                        else:
                            is_valid_gather = gi * Int32(4) < Int32(tile_rows_per_cta)
                            if is_valid_gather:
                                route_cache_base = wi * 4
                                r0 = self.routed_rows[route_cache_base]
                                r1 = self.routed_rows[route_cache_base + 1]
                                r2 = self.routed_rows[route_cache_base + 2]
                                r3 = self.routed_rows[route_cache_base + 3]

                                # SMEM destination: each s128b row = 128 bytes.
                                # Clustered routes store only this CTA's row slice.
                                smem_offset = Int32(
                                    ki * tile_rows_per_cta * 128
                                ) + gi * Int32(4 * 128)
                                smem_dst = stage_base.subview(smem_offset)

                                if cutlass.const_expr(self.cfg.has_cluster):
                                    mcast_mask = Int32(1) << cta_rank
                                    _tma_gather4_cluster(
                                        smem_dst,
                                        self.tma_desc,
                                        k_offset,
                                        r0,
                                        r1,
                                        r2,
                                        r3,
                                        stage_info.barrier,
                                        mcast_mask,
                                    )
                                else:
                                    _tma_gather4_cta(
                                        smem_dst,
                                        self.tma_desc,
                                        k_offset,
                                        r0,
                                        r1,
                                        r2,
                                        r3,
                                        stage_info.barrier,
                                    )

    @consumer_work(returns=(desc_a_mma_base, smem_a_stage_ptr))
    @cute.jit
    def build_mma_desc_a(self, stage_info: StageInfo) -> tuple[Int64, Int64]:
        desc, stage_ptr = self._build_mma_desc_impl(stage_info)
        return desc, stage_ptr

    @consumer_work(returns=(desc_b_mma_base, smem_b_stage_ptr))
    @cute.jit
    def build_mma_desc_b(self, stage_info: StageInfo) -> tuple[Int64, Int64]:
        desc, stage_ptr = self._build_mma_desc_impl(stage_info)
        return desc, stage_ptr

    @cute.jit
    def _build_mma_desc_impl(self, stage_info: StageInfo):
        """Build SMEM descriptor for MMA operand (s128b swizzle)."""
        is_b = self._operand == "b"
        tile_rows = (self.cfg.tile_n if is_b else self.cfg.tile_m) // (
            self.cfg.cluster_m if self.cfg.has_cluster and is_b else 1
        )
        dtype_bits = self.cfg.dtype_b_smem_bits if is_b else self.cfg.dtype_a_smem_bits
        bytes_per_stage = tile_rows * self.cfg.tile_k * dtype_bits // 8
        stage_base = self.smem_buf.subview(bytes_per_stage * stage_info.stage_idx)
        if cutlass.const_expr(self.cfg.is_fp8_mma):
            if cutlass.const_expr(is_b):
                leading_byte_offset = max(1024, self.cfg.tile_n * 64)
            else:
                leading_byte_offset = 16384
            desc = prims.Tcgen05SmemDesc.build(
                stage_base,
                leading_byte_offset=leading_byte_offset,
                stride_byte_offset=1024,
                layout=2,
            )
        elif cutlass.const_expr(self.cfg.uses_f16_mma):
            desc = prims.Tcgen05SmemDesc.build(
                stage_base,
                leading_byte_offset=bytes_per_stage,
                stride_byte_offset=1024,
                layout=2,
            )
        else:
            desc = prims.Tcgen05SmemDesc.build(
                stage_base,
                leading_byte_offset=16,
                stride_byte_offset=1024,
                layout=2,
            )
        return desc, Int64(stage_base.data_ptr().toint())
