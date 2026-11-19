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
from .batched_gemm_cga import (
    CgaPolicy,
    cga_tma_masks,
    has_multiple_umma_atoms,
    needs_extended_tma_multicast,
)
from .mixed_cga_ts import (
    override_tma_umma_pipeline,
    override_umma_atom_pipeline,
)
from .cache_policy import make_weight_l2_cache_hint
from .gmem_ab_resources import nonnegative_div, nonnegative_mod
from .smem_descriptor import build_tcgen05_smem_desc
from cutlass.experimental import primitives as prims
from cutlass.pipeline.sm100 import PipelineTmaUmma

Constexpr = cutlass.Constexpr


def _smem_k_box_elems(tile_k: int, smem_bits: int) -> int:
    """Return one at-most-128-byte swizzled K box."""
    max_box_k = 64 if smem_bits == 16 else (256 if smem_bits == 4 else 128)
    return min(tile_k, max_box_k)


def _native_mx_fp4_smem_desc_params(
    tile_rows: int, tile_k: int
) -> tuple[int, int, int]:
    """Return Rubin native-MXFP4 descriptor leading, stride, and layout.

    Native 4-bit rows shorter than 256 K elements share one 128-byte SMEM
    row.  The MMA descriptor must encode the resulting narrower swizzle;
    treating a 64-byte row as s128b makes the MMA address beyond its stage.
    """
    row_bytes = _smem_k_box_elems(tile_k, 4) // 2
    if row_bytes == 128:
        rows_per_smem_row, rows_per_swizzle, layout = 1, 8, 2
    elif row_bytes == 64:
        rows_per_smem_row, rows_per_swizzle, layout = 2, 4, 4
    elif row_bytes == 32:
        rows_per_smem_row, rows_per_swizzle, layout = 4, 2, 6
    else:
        raise ValueError(
            "native MXFP4 MMA requires a 32-, 64-, or 128-byte SMEM row, "
            f"got tile_k={tile_k} ({row_bytes} bytes)"
        )

    rows_per_slice = (
        tile_rows if tile_k > 256 else max(1, tile_rows // rows_per_smem_row)
    )
    return 128 * rows_per_slice, 128 * rows_per_swizzle, layout


def _uses_native_mx_fp4_operand(cfg: BatchedGemmConfig, dtype_kind: int) -> bool:
    return cfg.uses_native_mx_fp4_operands and dtype_kind == int(DType.MXE2M1)


@cute.jit
def _cta_rank_in_umma_atom(cfg: cutlass.Constexpr[BatchedGemmConfig]) -> Int32:
    """Return the CTA's 0/1 position inside its physical 2-CTA UMMA atom."""
    cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
    if cutlass.const_expr(has_multiple_umma_atoms(cfg)):
        return cta_rank & Int32(1)
    return cta_rank


@dataclass(kw_only=True)
class SmemAResource(MemoryResource):
    """SMEM staging for operand A (TMA producer, UMMA consumer)."""

    cfg: Constexpr[BatchedGemmConfig]
    tma_a_desc: Any = None
    weight_tile_idx_view: Any = None
    weight_num_token_tiles_view: Any = None
    weight_l2_cache_hint: Any = None
    smem_buf: Any = None
    _alloc_a: Constexpr[Optional[SmemAllocation]] = None
    desc_a_mma_base: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    smem_a_stage_ptr: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    mixed_cga_policy: Constexpr[int] = int(CgaPolicy.MCAST_A)
    mixed_cga_kind: Constexpr[str] = "a"

    def create_pipeline(self, pipeline_config):
        result = super().create_pipeline(pipeline_config)
        # Every physical 2-CTA UMMA atom needs rank-relative full/empty
        # barrier routing, even when A's TMA payload remains CTA-local.  The
        # operand-specific multicast decision is applied separately in load().
        if has_multiple_umma_atoms(self.cfg) and isinstance(result, PipelineTmaUmma):
            return override_tma_umma_pipeline(self.cfg, result, "a")
        return result

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
        if cutlass.const_expr(self.cfg.weight_l2_evict_first):
            # Seed loop-carried state before a persistent tile can early-exit.
            # prepare_weight_tile replaces it before any weight load.
            self.weight_l2_cache_hint = Int64(0)

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def prepare_weight_tile(self, stage_info: StageInfo) -> None:
        self.weight_l2_cache_hint = make_weight_l2_cache_hint(
            self.cfg.weight_l2_evict_first,
            self.weight_tile_idx_view,
            self.weight_num_token_tiles_view,
            stage_info.work_tile.tile_idx[1],
        )

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
            elif cutlass.const_expr(self.cfg.dtype_a == int(DType.BF16)):
                if cutlass.const_expr(self.cfg.use_bf16_kbox_tma_a):
                    a_box_k = self.cfg.tile_k
                else:
                    a_box_k = _smem_k_box_elems(
                        self.cfg.tile_k, self.cfg.dtype_a_smem_bits
                    )
            elif cutlass.const_expr(self.cfg.has_cast_a):
                a_box_k = self.cfg.tile_k
            else:
                a_box_k = _smem_k_box_elems(self.cfg.tile_k, self.cfg.dtype_a_smem_bits)

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
        elif cutlass.const_expr(self.cfg.uses_two_cta_mma):
            cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
            if cutlass.const_expr(needs_extended_tma_multicast(self.cfg, "a")):
                mcast_mask, _, _, _ = cga_tma_masks(self.cfg, "a")
                lead_cta_rank = (cta_rank // Int32(2)) * Int32(2)
                barrier = prims.mapa(barrier, lead_cta_rank)
                barrier = prims.cvta_to(barrier, prims.CvtaSpace.SHARED)
            else:
                mcast_mask = Int32(1) << cta_rank
                lead_cta_rank = nonnegative_div(
                    cta_rank, self.cfg.umma_cta_count
                ) * Int32(self.cfg.umma_cta_count)
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
                l2_cache_hint=self.weight_l2_cache_hint,
            )
        else:
            prims.cp_async_bulk_tensor_shared_cta_global(
                smem_dst,
                tma_desc,
                coords,
                barrier,
                l2_cache_hint=self.weight_l2_cache_hint,
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
        if cutlass.const_expr(_uses_native_mx_fp4_operand(self.cfg, self.cfg.dtype_a)):
            leading, stride, layout = _native_mx_fp4_smem_desc_params(
                self.cfg.tile_m, self.cfg.tile_k
            )
            desc_a_mma_base = build_tcgen05_smem_desc(
                self.cfg,
                stage_base,
                leading_byte_offset=leading,
                stride_byte_offset=stride,
                layout=layout,
            )
        elif cutlass.const_expr(self.cfg.is_fp8_mma):
            desc_a_mma_base = build_tcgen05_smem_desc(
                self.cfg,
                stage_base,
                leading_byte_offset=16384,
                stride_byte_offset=1024,
                layout=2,
            )
        elif cutlass.const_expr(self.cfg.uses_f16_mma):
            desc_a_mma_base = build_tcgen05_smem_desc(
                self.cfg,
                stage_base,
                leading_byte_offset=self.cfg.num_bytes_a_per_stage,
                stride_byte_offset=1024,
                layout=2,
            )
        else:
            desc_a_mma_base = build_tcgen05_smem_desc(
                self.cfg,
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
    mixed_cga_policy: Constexpr[int] = int(CgaPolicy.MCAST_B)
    mixed_cga_kind: Constexpr[str] = "b"

    def create_pipeline(self, pipeline_config):
        result = super().create_pipeline(pipeline_config)
        # Barrier routing follows physical UMMA atoms independently of whether
        # B uses extended TMA payload multicast.
        if has_multiple_umma_atoms(self.cfg) and isinstance(result, PipelineTmaUmma):
            return override_tma_umma_pipeline(self.cfg, result, "b")
        return result

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
            cta_rank_in_atom = _cta_rank_in_umma_atom(self.cfg)
            coord_b_mn = coord_b_mn + cta_rank_in_atom * Int32(
                self.cfg.tile_n // self.cfg.umma_cta_count
            )
        stage_base = self.smem_buf.subview(
            self.cfg.num_bytes_b_smem_per_stage * stage_info.stage_idx
        )
        if prims.elect_sync():
            if cutlass.const_expr(self.cfg.uses_block_major_k_weight_b):
                b_box_k = self.cfg.tile_k
            else:
                b_box_k = _smem_k_box_elems(self.cfg.tile_k, self.cfg.dtype_b_smem_bits)

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
                cta_rank_in_atom = _cta_rank_in_umma_atom(self.cfg)
                cta_row_offset = cta_rank_in_atom * Int32(
                    self.cfg.tile_n // self.cfg.umma_cta_count
                )
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
        if cutlass.const_expr(self.cfg.uses_two_cta_mma):
            cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
            if cutlass.const_expr(needs_extended_tma_multicast(self.cfg, "b")):
                mcast_mask, _, _, _ = cga_tma_masks(self.cfg, "b")
            else:
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
        if cutlass.const_expr(_uses_native_mx_fp4_operand(self.cfg, self.cfg.dtype_b)):
            tile_rows = self.cfg.tile_n // (
                self.cfg.umma_cta_count if self.cfg.uses_two_cta_mma else 1
            )
            leading, stride, layout = _native_mx_fp4_smem_desc_params(
                tile_rows, self.cfg.tile_k
            )
            desc_b_mma_base = build_tcgen05_smem_desc(
                self.cfg,
                stage_base,
                leading_byte_offset=leading,
                stride_byte_offset=stride,
                layout=layout,
            )
        elif cutlass.const_expr(self.cfg.has_cast_a):
            # The CastA-generated BF16 MMA path uses the B
            # descriptor packing: tileN=8 -> 1024, tileN=16 -> 2048, etc.
            desc_b_mma_base = build_tcgen05_smem_desc(
                self.cfg,
                stage_base,
                leading_byte_offset=max(
                    1024, self.cfg.tile_n * (16384 // self.cfg.mma_m)
                ),
                stride_byte_offset=1024,
                layout=2,
            )
        elif cutlass.const_expr(self.cfg.is_fp8_mma):
            desc_b_mma_base = build_tcgen05_smem_desc(
                self.cfg,
                stage_base,
                leading_byte_offset=max(1024, self.cfg.tile_n * 64),
                stride_byte_offset=1024,
                layout=2,
            )
        elif cutlass.const_expr(self.cfg.uses_f16_mma):
            desc_b_mma_base = build_tcgen05_smem_desc(
                self.cfg,
                stage_base,
                leading_byte_offset=self.cfg.num_bytes_b_smem_per_stage,
                stride_byte_offset=1024,
                layout=2,
            )
        else:
            desc_b_mma_base = build_tcgen05_smem_desc(
                self.cfg,
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

    def create_pipeline(self, pipeline_config):
        result = super().create_pipeline(pipeline_config)
        if has_multiple_umma_atoms(self.cfg):
            return override_umma_atom_pipeline(result)
        return result

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
            tile_rows // self.cfg.umma_cta_count
            if is_operand_b and self.cfg.uses_two_cta_mma
            else tile_rows
        )
        copy_bytes = 16
        dtype_bits = (
            self.cfg.dtype_b_smem_bits if is_operand_b else self.cfg.dtype_a_smem_bits
        )
        box_k = _smem_k_box_elems(self.cfg.tile_k, dtype_bits)
        copies_per_row_box = box_k * dtype_bits // 8 // copy_bytes
        num_k_boxes = self.cfg.tile_k // box_k
        total_copies = tile_rows_per_cta * copies_per_row_box * num_k_boxes
        num_threads = (
            self.cfg.warp_layout.num_load_b_warps
            if self.cfg.is_swap_ab
            else self.cfg.warp_layout.num_load_a_warps
        ) * 32
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
        if cutlass.const_expr(is_operand_b and self.cfg.uses_two_cta_mma):
            tile_rows_per_cta = tile_rows // self.cfg.umma_cta_count
            cta_rank_in_atom = _cta_rank_in_umma_atom(self.cfg)
            cta_row_offset = cta_rank_in_atom * Int32(tile_rows_per_cta)
        else:
            tile_rows_per_cta = tile_rows
            cta_row_offset = Int32(0)

        if cutlass.const_expr(is_operand_b):
            coord_tile = tile_coord_n * Int32(tile_rows)
            token_tile = tile_coord_n
        else:
            coord_tile = tile_coord_m * Int32(tile_rows)
            token_tile = tile_coord_m

        token_cga = nonnegative_div(token_tile, self.cfg.token_cluster_dim)
        self.tile_limit = self._local_tile_limit(
            self.mn_limit.load(idx=token_cga, vector_size=1)[0],
            token_tile,
            tile_rows,
        )

        tidx, _, _ = cute.arch.thread_idx()
        local_tid = tidx - Int32(
            (
                self.cfg.warp_layout.load_b_warp_idx
                if self.cfg.is_swap_ab
                else self.cfg.warp_layout.load_a_warp_idx
            )
            * 32
        )
        copy_bytes = 16
        dtype_bits = (
            self.cfg.dtype_b_smem_bits if is_operand_b else self.cfg.dtype_a_smem_bits
        )
        box_k = _smem_k_box_elems(self.cfg.tile_k, dtype_bits)
        copies_per_row_box = box_k * dtype_bits // 8 // copy_bytes
        num_k_boxes = self.cfg.tile_k // box_k
        total_copies = tile_rows_per_cta * copies_per_row_box * num_k_boxes
        num_threads = (
            self.cfg.warp_layout.num_load_b_warps
            if self.cfg.is_swap_ab
            else self.cfg.warp_layout.num_load_a_warps
        ) * 32
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
        The SMEM destination uses the swizzle selected by the physical row
        width (s128b normally, s64b for native MXFP4 with tile_k=128).
        """
        is_operand_b = cutlass.const_expr(self._operand == "b")
        bytes_per_stage = (
            self.cfg.num_bytes_b_smem_per_stage
            if is_operand_b
            else self.cfg.num_bytes_a_per_stage
        )
        tile_rows = self.cfg.tile_n if is_operand_b else self.cfg.tile_m
        if cutlass.const_expr(is_operand_b and self.cfg.uses_two_cta_mma):
            tile_rows_per_cta = tile_rows // self.cfg.umma_cta_count
            cta_rank_in_atom = _cta_rank_in_umma_atom(self.cfg)
            cta_row_offset = cta_rank_in_atom * Int32(tile_rows_per_cta)
        else:
            tile_rows_per_cta = tile_rows
            cta_row_offset = Int32(0)

        stage_base = self.smem_buf.subview(bytes_per_stage * stage_info.stage_idx)

        tidx, _, _ = cute.arch.thread_idx()
        local_tid = tidx - Int32(
            (
                self.cfg.warp_layout.load_b_warp_idx
                if self.cfg.is_swap_ab
                else self.cfg.warp_layout.load_a_warp_idx
            )
            * 32
        )

        copy_bytes = 16
        dtype_bits = (
            self.cfg.dtype_b_smem_bits if is_operand_b else self.cfg.dtype_a_smem_bits
        )
        box_k = _smem_k_box_elems(self.cfg.tile_k, dtype_bits)
        row_box_bytes = box_k * dtype_bits // 8
        rows_per_smem_row = 128 // row_box_bytes
        rows_per_swizzle = 8 // rows_per_smem_row
        copies_per_row_box = row_box_bytes // copy_bytes
        num_k_boxes = self.cfg.tile_k // box_k
        total_copies = tile_rows_per_cta * copies_per_row_box * num_k_boxes
        num_threads = (
            self.cfg.warp_layout.num_load_b_warps
            if self.cfg.is_swap_ab
            else self.cfg.warp_layout.num_load_a_warps
        ) * 32
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

            # Pack short logical rows into 128-byte physical rows before
            # applying the physical row's s128b/s64b/s32b XOR pattern.
            smem_kbox_bytes = k_box * Int32(tile_rows_per_cta * row_box_bytes)
            smem_row_bytes = row_in_tile * Int32(row_box_bytes)
            smem_col_bytes = col_chunk * Int32(copy_bytes)
            smem_off_bytes = smem_kbox_bytes + smem_row_bytes + smem_col_bytes
            physical_row = nonnegative_div(row_in_tile, rows_per_smem_row)
            swizzle_mask = nonnegative_mod(physical_row, rows_per_swizzle) * Int32(
                copy_bytes
            )
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
        """Build the SMEM descriptor for the gathered MMA operand."""
        is_operand_b = cutlass.const_expr(self._operand == "b")
        bytes_per_stage = (
            self.cfg.num_bytes_b_smem_per_stage
            if is_operand_b
            else self.cfg.num_bytes_a_per_stage
        )

        stage_base = self.smem_buf.subview(bytes_per_stage * stage_idx)
        dtype_kind = self.cfg.dtype_b if is_operand_b else self.cfg.dtype_a
        if cutlass.const_expr(_uses_native_mx_fp4_operand(self.cfg, dtype_kind)):
            tile_rows = self.cfg.tile_n if is_operand_b else self.cfg.tile_m
            if cutlass.const_expr(is_operand_b and self.cfg.uses_two_cta_mma):
                tile_rows //= self.cfg.umma_cta_count
            leading, stride, layout = _native_mx_fp4_smem_desc_params(
                tile_rows, self.cfg.tile_k
            )
            desc = build_tcgen05_smem_desc(
                self.cfg,
                stage_base,
                leading_byte_offset=leading,
                stride_byte_offset=stride,
                layout=layout,
            )
        elif cutlass.const_expr(self.cfg.is_fp8_mma):
            if cutlass.const_expr(is_operand_b):
                leading_byte_offset = max(1024, self.cfg.tile_n * 64)
            else:
                leading_byte_offset = 16384
            desc = build_tcgen05_smem_desc(
                self.cfg,
                stage_base,
                leading_byte_offset=leading_byte_offset,
                stride_byte_offset=1024,
                layout=2,
            )
        elif cutlass.const_expr(self.cfg.uses_f16_mma):
            desc = build_tcgen05_smem_desc(
                self.cfg,
                stage_base,
                leading_byte_offset=bytes_per_stage,
                stride_byte_offset=1024,
                layout=2,
            )
        else:
            desc = build_tcgen05_smem_desc(
                self.cfg,
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
    # Route completion to this 2-CTA UMMA atom's leader.  Rank zero is correct
    # only for the first atom of a wider physical CGA.
    cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
    atom_leader = cta_rank & Int32(~1)
    lead_barrier = prims.cvta_to(
        prims.mapa(barrier, atom_leader), prims.CvtaSpace.SHARED
    )
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
    cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
    atom_leader = cta_rank & Int32(~1)
    lead_barrier = prims.cvta_to(
        prims.mapa(barrier, atom_leader), prims.CvtaSpace.SHARED
    )
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
    mixed_cga_policy: Constexpr[int] = int(CgaPolicy.ALL)
    mixed_cga_kind: Constexpr[str] = ""

    def create_pipeline(self, pipeline_config):
        result = super().create_pipeline(pipeline_config)
        # A TMA-gather pipeline still synchronizes with the current physical
        # UMMA atom when its payload mask is local.
        if has_multiple_umma_atoms(self.cfg) and isinstance(result, PipelineTmaUmma):
            return override_tma_umma_pipeline(self.cfg, result, self._operand)
        return result

    def __post_init__(self):
        self.mixed_cga_kind = self._operand
        self.mixed_cga_policy = int(
            CgaPolicy.MCAST_B if self._operand == "b" else CgaPolicy.MCAST_A
        )
        if self._alloc is None:
            is_b = self._operand == "b"
            tile_rows = self.cfg.tile_n if is_b else self.cfg.tile_m
            dtype_bits = (
                self.cfg.dtype_b_smem_bits if is_b else self.cfg.dtype_a_smem_bits
            )
            if self.cfg.uses_two_cta_mma and is_b:
                tile_rows //= self.cfg.umma_cta_count
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
            self.cfg.umma_cta_count if self.cfg.uses_two_cta_mma and is_b else 1
        )
        num_gather4_per_kbox = tile_rows // 4
        num_load_warps = (
            self.cfg.warp_layout.num_load_b_warps
            if is_b
            else self.cfg.warp_layout.num_load_a_warps
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
    def _load_routed_row_or_oob(self, route_idx, row_in_tile, tile_limit):
        # An out-of-bounds coordinate requests TMA zero-fill for padding.
        # Reusing token 0 creates concentrated reads that can slow down FC1.
        routed_row = Int32(-1)
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
        if cutlass.const_expr(self.cfg.uses_two_cta_mma and is_b):
            tile_rows_per_cta = tile_rows // self.cfg.umma_cta_count
            cta_rank_in_atom = _cta_rank_in_umma_atom(self.cfg)
            cta_row_offset = cta_rank_in_atom * Int32(tile_rows_per_cta)
        else:
            tile_rows_per_cta = tile_rows
            cta_row_offset = Int32(0)

        if cutlass.const_expr(is_b):
            coord_mn = tile_coord_n * Int32(self.cfg.tile_n)
            token_tile = tile_coord_n
            num_load_warps = self.cfg.warp_layout.num_load_b_warps
            load_warp_idx = self.cfg.warp_layout.load_b_warp_idx
        else:
            coord_mn = tile_coord_m * Int32(self.cfg.tile_m)
            token_tile = tile_coord_m
            num_load_warps = self.cfg.warp_layout.num_load_a_warps
            load_warp_idx = self.cfg.warp_layout.load_a_warp_idx

        tile_limit = Int32(tile_rows)
        if cutlass.const_expr(self.mn_limit is not None):
            token_cga = token_tile // Int32(self.cfg.token_cluster_dim)
            tile_limit = self._local_tile_limit(
                self.mn_limit.load(idx=token_cga, vector_size=1)[0],
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
                self.routed_rows[wi * 4] = self._load_routed_row_or_oob(
                    base_row, row_base, tile_limit
                )
                self.routed_rows[wi * 4 + 1] = self._load_routed_row_or_oob(
                    base_row + Int32(1), row_base + Int32(1), tile_limit
                )
                self.routed_rows[wi * 4 + 2] = self._load_routed_row_or_oob(
                    base_row + Int32(2), row_base + Int32(2), tile_limit
                )
                self.routed_rows[wi * 4 + 3] = self._load_routed_row_or_oob(
                    base_row + Int32(3), row_base + Int32(3), tile_limit
                )
            else:
                is_valid_gather = gi * Int32(4) < Int32(tile_rows_per_cta)
                if is_valid_gather:
                    base_row = coord_mn + cta_row_offset + gi * Int32(4)
                    row_base = cta_row_offset + gi * Int32(4)
                    self.routed_rows[wi * 4] = self._load_routed_row_or_oob(
                        base_row, row_base, tile_limit
                    )
                    self.routed_rows[wi * 4 + 1] = self._load_routed_row_or_oob(
                        base_row + Int32(1), row_base + Int32(1), tile_limit
                    )
                    self.routed_rows[wi * 4 + 2] = self._load_routed_row_or_oob(
                        base_row + Int32(2), row_base + Int32(2), tile_limit
                    )
                    self.routed_rows[wi * 4 + 3] = self._load_routed_row_or_oob(
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
        if cutlass.const_expr(self.cfg.uses_two_cta_mma):
            cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        if cutlass.const_expr(self.cfg.uses_two_cta_mma and is_b):
            tile_rows_per_cta = tile_rows // self.cfg.umma_cta_count
        else:
            tile_rows_per_cta = tile_rows
        bytes_per_stage = tile_rows_per_cta * self.cfg.tile_k * dtype_bits // 8

        stage_base = self.smem_buf.subview(bytes_per_stage * stage_info.stage_idx)

        # Each gather4 call loads four rows from an at-most-128-byte K box.
        box_k = _smem_k_box_elems(self.cfg.tile_k, dtype_bits)
        row_box_bytes = box_k * dtype_bits // 8
        num_k_boxes = self.cfg.tile_k // box_k
        num_gather4_per_kbox = tile_rows_per_cta // 4
        num_load_warps = (
            self.cfg.warp_layout.num_load_b_warps
            if is_b
            else self.cfg.warp_layout.num_load_a_warps
        )
        load_warp_idx = (
            self.cfg.warp_layout.load_b_warp_idx
            if is_b
            else self.cfg.warp_layout.load_a_warp_idx
        )
        warp_batches = (num_gather4_per_kbox + num_load_warps - 1) // num_load_warps
        all_gathers_valid = num_gather4_per_kbox % num_load_warps == 0

        warp_idx = cute.arch.warp_idx()
        warp_in_task = warp_idx - Int32(load_warp_idx)
        warp_in_task = cute.arch.make_warp_uniform(warp_in_task)

        if prims.elect_sync():
            for ki in cutlass.range_constexpr(num_k_boxes):
                k_offset = coord_k + Int32(ki * box_k)
                if cutlass.const_expr(
                    self.cfg.uses_two_cta_mma
                    and all_gathers_valid
                    and warp_batches == 4
                ):
                    smem_offset0 = Int32(
                        ki * tile_rows_per_cta * row_box_bytes
                    ) + warp_in_task * Int32(4 * row_box_bytes)
                    smem_delta = Int32(num_load_warps * 4 * row_box_bytes)
                    smem_dst0 = stage_base.subview(smem_offset0)
                    smem_dst1 = stage_base.subview(smem_offset0 + smem_delta)
                    smem_dst2 = stage_base.subview(smem_offset0 + smem_delta * Int32(2))
                    smem_dst3 = stage_base.subview(smem_offset0 + smem_delta * Int32(3))
                    if cutlass.const_expr(
                        needs_extended_tma_multicast(self.cfg, self._operand)
                    ):
                        mcast_mask, _, _, _ = cga_tma_masks(self.cfg, self._operand)
                    else:
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

                            # SMEM destination: rows are contiguous in this K box.
                            # Clustered routes store only this CTA's row slice.
                            smem_offset = Int32(
                                ki * tile_rows_per_cta * row_box_bytes
                            ) + gi * Int32(4 * row_box_bytes)
                            smem_dst = stage_base.subview(smem_offset)

                            if cutlass.const_expr(self.cfg.uses_two_cta_mma):
                                if cutlass.const_expr(
                                    needs_extended_tma_multicast(
                                        self.cfg, self._operand
                                    )
                                ):
                                    mcast_mask, _, _, _ = cga_tma_masks(
                                        self.cfg, self._operand
                                    )
                                else:
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

                                # SMEM destination: rows are contiguous in this K box.
                                # Clustered routes store only this CTA's row slice.
                                smem_offset = Int32(
                                    ki * tile_rows_per_cta * row_box_bytes
                                ) + gi * Int32(4 * row_box_bytes)
                                smem_dst = stage_base.subview(smem_offset)

                                if cutlass.const_expr(self.cfg.uses_two_cta_mma):
                                    if cutlass.const_expr(
                                        needs_extended_tma_multicast(
                                            self.cfg, self._operand
                                        )
                                    ):
                                        mcast_mask, _, _, _ = cga_tma_masks(
                                            self.cfg, self._operand
                                        )
                                    else:
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
        """Build the SMEM descriptor for the TMA-gathered MMA operand."""
        is_b = self._operand == "b"
        tile_rows = (self.cfg.tile_n if is_b else self.cfg.tile_m) // (
            self.cfg.umma_cta_count if self.cfg.uses_two_cta_mma and is_b else 1
        )
        dtype_bits = self.cfg.dtype_b_smem_bits if is_b else self.cfg.dtype_a_smem_bits
        bytes_per_stage = tile_rows * self.cfg.tile_k * dtype_bits // 8
        stage_base = self.smem_buf.subview(bytes_per_stage * stage_info.stage_idx)
        dtype_kind = self.cfg.dtype_b if is_b else self.cfg.dtype_a
        if cutlass.const_expr(_uses_native_mx_fp4_operand(self.cfg, dtype_kind)):
            leading, stride, layout = _native_mx_fp4_smem_desc_params(
                tile_rows, self.cfg.tile_k
            )
            desc = build_tcgen05_smem_desc(
                self.cfg,
                stage_base,
                leading_byte_offset=leading,
                stride_byte_offset=stride,
                layout=layout,
            )
        elif cutlass.const_expr(self.cfg.is_fp8_mma):
            if cutlass.const_expr(is_b):
                leading_byte_offset = max(1024, self.cfg.tile_n * 64)
            else:
                leading_byte_offset = 16384
            desc = build_tcgen05_smem_desc(
                self.cfg,
                stage_base,
                leading_byte_offset=leading_byte_offset,
                stride_byte_offset=1024,
                layout=2,
            )
        elif cutlass.const_expr(self.cfg.uses_f16_mma):
            desc = build_tcgen05_smem_desc(
                self.cfg,
                stage_base,
                leading_byte_offset=bytes_per_stage,
                stride_byte_offset=1024,
                layout=2,
            )
        else:
            desc = build_tcgen05_smem_desc(
                self.cfg,
                stage_base,
                leading_byte_offset=16,
                stride_byte_offset=1024,
                layout=2,
            )
        return desc, Int64(stage_base.data_ptr().toint())
