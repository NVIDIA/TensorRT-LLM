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
  - auxiliary producer_work: copy_fused_sf.
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
    sfb_r128c4_utccp_offsets,
    TMEM_SF_UTCCP_COLS_PER_COPY,
    TMEM_SF_PACK_SIZE_BYTES,
    WARPS_PER_WARPGROUP,
    WARP_SIZE,
)
from .gmem_ab_resources import nonnegative_div, nonnegative_mod
from cutlass.experimental import primitives as prims

from .batched_gemm_cga import has_multiple_umma_atoms
from .mixed_cga_ts import override_umma_atom_pipeline
from .smem_descriptor import build_tcgen05_smem_desc

Constexpr = cutlass.Constexpr


def _mma_smem_desc_increment(
    kblock_idx: int,
    *,
    mma_k: int,
    smem_bits: int,
    tile_k: int,
    stage_bytes: Optional[int] = None,
) -> int:
    """Return one operand's swizzled MMA-descriptor K increment.

    Descriptor increments are measured in 16-byte units. A 128-byte swizzle
    row contains ``1024 // smem_bits`` K elements; within that row, one MMA
    advances by ``mma_k * smem_bits // 128`` descriptor units. Operand A uses
    the fixed 1024-unit next-row stride. Operand B supplies its per-CTA stage
    size so the next-row stride also reflects N tiling and CTA splitting.
    """
    bits_per_mma = mma_k * smem_bits
    if bits_per_mma % 128 != 0:
        raise ValueError(
            "MMA K slice must occupy whole 16-byte descriptor units, got "
            f"mma_k={mma_k}, smem_bits={smem_bits}"
        )
    if 1024 % bits_per_mma != 0:
        raise ValueError(
            "MMA K slice must divide one 128-byte swizzle row, got "
            f"mma_k={mma_k}, smem_bits={smem_bits}"
        )
    if tile_k <= 0:
        raise ValueError(f"tile_k must be positive, got {tile_k}")

    kblocks_per_group = 1024 // bits_per_mma
    desc_units_per_mma = bits_per_mma // 128
    k_minor = kblock_idx % kblocks_per_group
    k_major = kblock_idx // kblocks_per_group
    group_stride = 1024
    if stage_bytes is not None:
        if stage_bytes <= 0:
            raise ValueError(f"stage_bytes must be positive, got {stage_bytes}")
        k_group_elems = 1024 // smem_bits
        group_stride = max(64, (k_group_elems * stage_bytes // tile_k) >> 4)
    return k_major * group_stride + k_minor * desc_units_per_mma


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


def _build_fp8_instr_desc(*, n_dim: int, m_dim: int, mma_k: int):
    """Build a plain FP8 descriptor, including Rubin's K64 mode bit."""
    if mma_k not in (32, 64):
        raise ValueError(f"FP8 instruction descriptor requires K32/K64, got {mma_k}")
    return prims.Tcgen05InstrDesc.build(
        a_dtype=cutlass.Float8E4M3FN,
        b_dtype=cutlass.Float8E4M3FN,
        c_dtype=cutlass.Float32,
        n_dim=n_dim,
        m_dim=m_dim,
        k_dim=1 if mma_k == 64 else 0,
    )


def _build_mx_instr_desc(
    *,
    a_format: int,
    b_format: int,
    n_dim: int,
    m_dim: int,
    mma_k: int,
):
    """Build an MX descriptor, including Rubin's K64 mode bit."""
    if mma_k not in (32, 64):
        raise ValueError(f"MX instruction descriptor requires K32/K64, got {mma_k}")
    return prims.Tcgen05MxInstrDesc.build(
        a_dtype=_mx_dtype_from_format(a_format),
        b_dtype=_mx_dtype_from_format(b_format),
        scale_format=1,  # UE8M0
        n_dim=n_dim,
        m_dim=m_dim,
        k_dim=1 if mma_k == 64 else 0,
    )


def _build_nvfp4_instr_desc(
    *, n_dim: int, m_dim: int, mma_k: int = 64, is_rubin: bool = False
):
    """Build the architecture-specific ``kind::mxf4nvf4`` descriptor.

    Rubin's OMMA uses E2M1 format encoding 1 and a split K mode of 0 for K64
    or 2 for K128. Blackwell uses the legacy MX descriptor's E2M1 encoding 5.
    """
    if mma_k not in (64, 128):
        raise ValueError(f"NVFP4 instruction descriptor requires K64/K128, got {mma_k}")
    if not is_rubin:
        if mma_k != 64:
            raise ValueError("NVFP4 mma_k=128 requires a Rubin OMMA descriptor")
        return prims.Tcgen05MxInstrDesc.build(
            a_dtype=cutlass.Float4E2M1FN,
            b_dtype=cutlass.Float4E2M1FN,
            scale_format=0,
            n_dim=n_dim,
            m_dim=m_dim,
        )
    return prims.Tcgen05MxOmmaInstrDesc.build(
        a_dtype=cutlass.Float4E2M1FN,
        b_dtype=cutlass.Float4E2M1FN,
        scale_format=0,  # UE4M3
        n_dim=n_dim,
        m_dim=m_dim,
        k_dim=0 if mma_k == 64 else 2,
        sparsity_version=0,
    )


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

    def create_pipeline(self, pipeline_config):
        result = super().create_pipeline(pipeline_config)
        if has_multiple_umma_atoms(self.cfg):
            return override_umma_atom_pipeline(result)
        return result

    def __post_init__(self):
        if self._alloc_c is None:
            cols_per_stage = self.cfg.tmem_overlap_window_cols
            total_cols = cols_per_stage * self.cfg.num_stages_tmem_acc
            if self.cfg.has_input_block_scaling and self.cfg.fuse_sf_copy_to_mma:
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
            swap_t2r_regs = max(1, self.cfg.epi_tile_n // 2)
            t2r_init = cutlass.vector.full([swap_t2r_regs], 0.0, cutlass.Float32)
            t2r_1_init = cutlass.vector.full([swap_t2r_regs], 0.0, cutlass.Float32)
        else:
            epi_t2r_repx = self.cfg.non_swap_tmem_load_num_regs
            t2r_init = cutlass.vector.full([max(1, epi_t2r_repx)], 0.0, cutlass.Float32)
            needs_secondary_fragment = (
                self.cfg.has_gated_epilogue
                and self.cfg.non_swap_epilogue_fragments_per_group == 2
            ) or (
                self.cfg.uses_tmem_overlap_window
                and self.cfg.non_swap_tmem_overlap_loads == 2
            )
            secondary_repx = epi_t2r_repx if needs_secondary_fragment else 1
            t2r_1_init = cutlass.vector.full([secondary_repx], 0.0, cutlass.Float32)
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
        if cutlass.const_expr(self.cfg.uses_tmem_overlap_window):
            self._mma_local_idx = Int32(0)
            self._epi_local_idx = Int32(0)
        # Compute SFA/SFB TMEM base addresses for fused S2T+MMA
        if cutlass.const_expr(self.cfg.has_input_block_scaling):
            base_col = self.tmem_raw_addr & 0xFFFF
            base_row = self.tmem_raw_addr >> 16
            # TMEM allocator order: C first (at offset 0), then SFA, SFB
            # Matches nvfp4_gemm reference: acc at base, SFA after acc
            # The fused SF reservation is included in _alloc_c, but SFA starts
            # after the accumulator region rather than after the whole
            # allocation.
            c_total_cols = (
                self.cfg.tmem_overlap_window_cols * self.cfg.num_stages_tmem_acc
            )
            sf_stage_mult = (
                self.cfg.num_stages_tmem_sfa
                if (
                    self.cfg.has_input_block_scaling
                    and not self.cfg.fuse_sf_copy_to_mma
                )
                else 1
            )
            sfa_col = base_col + c_total_cols
            sfb_col = sfa_col + self.cfg.tmem_sfa_cols * sf_stage_mult
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
            self.idesc = _build_fp8_instr_desc(
                n_dim=self.cfg.mma_n,
                m_dim=self.cfg.mma_m,
                mma_k=self.cfg.mma_k,
            )
        elif cutlass.const_expr(self.cfg.is_mx_mma):
            self.idesc = _build_mx_instr_desc(
                a_format=self.cfg.mx_a_format,
                b_format=self.cfg.mx_b_format,
                n_dim=self.cfg.mma_n,
                m_dim=self.cfg.mma_m,
                mma_k=self.cfg.mma_k,
            )
        else:
            # FP4 NVF4: block-scaled MMA descriptor (tcgen05_mma_block_scale
            # MXF4NVF4). NVF4 uses E4M3 scale factors (scale_format=0, UE4M3);
            # MXF4 (OCP MX) would use E8M0 (scale_format=1).
            self.idesc = _build_nvfp4_instr_desc(
                n_dim=self.cfg.mma_n,
                m_dim=self.cfg.mma_m,
                mma_k=self.cfg.mma_k,
                is_rubin=self.cfg.is_rubin,
            )

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_accumulator_state(self, stage_info: StageInfo) -> None:
        self._init_tmem_state(stage_info)

    @consumer_work_decorator(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_epilogue_state(self, stage_info: StageInfo) -> None:
        self._init_tmem_state(stage_info)

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
        return build_tcgen05_smem_desc(
            self.cfg,
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
        first_k_tile: Constexpr[Optional[bool]] = None,
        k_subblock: Constexpr[Optional[int]] = None,
    ) -> None:
        self._mma_impl(
            stage_info,
            desc_a_mma_base,
            desc_b_mma_base,
            smem_a_stage_ptr=smem_a_stage_ptr,
            first_k_tile=first_k_tile,
            k_subblock=k_subblock,
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
        first_k_tile: Constexpr[Optional[bool]] = None,
    ) -> None:
        self._mma_impl(
            stage_info,
            desc_a_mma_base,
            desc_b_mma_base,
            smem_a_stage_ptr=smem_a_stage_ptr,
            first_k_tile=first_k_tile,
        )

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def copy_fused_sf(
        self,
        stage_info: StageInfo,
        *,
        desc_a_s2t_base: prims.Tcgen05SmemDesc,
        smem_sfa_stage_ptr: Int64,
        desc_b_s2t_base: prims.Tcgen05SmemDesc,
        smem_sfb_stage_ptr: Int64,
    ) -> None:
        """Copy fused scale factors to TMEM before acquiring the C stage."""
        # The stage pointers are routed here to preserve scheduler dependencies;
        # the descriptors already encode the addresses used by UTCCP.
        del stage_info, smem_sfa_stage_ptr, smem_sfb_stage_ptr
        s2t_shape, s2t_multicast = prims.S2TCopyMode.S2T_32x128b_WARPX4
        num_sfa_iters = self.cfg.tmem_sfa_cols // TMEM_SF_UTCCP_COLS_PER_COPY
        num_sfb_iters = self.cfg.tmem_sfb_cols // TMEM_SF_UTCCP_COLS_PER_COPY

        # PTX same-address ordering associates each MMA scale-factor read with
        # its base TMEM address. Issue higher-address copies first so the copy
        # at the first address consumed by MMA is the final ordering point.
        for copy_idx in cutlass.range_constexpr(num_sfa_iters):
            s2t_idx = num_sfa_iters - 1 - copy_idx
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

        for copy_idx in cutlass.range_constexpr(num_sfb_iters):
            s2t_idx = num_sfb_iters - 1 - copy_idx
            if cutlass.const_expr(self.cfg.uses_n_major_r128c4_sfb_utccp):
                num_utccp_k = (
                    self.cfg.tile_k
                    // self.cfg.input_sf_block_size_b
                    // TMEM_SF_PACK_SIZE_BYTES
                )
                assert num_sfb_iters == (self.cfg.tile_n // 128) * num_utccp_k, (
                    "R128c4 SFB UTCCP copies must cover every (N block, K group)"
                )
                increment_s2t, sfb_tmem_delta = sfb_r128c4_utccp_offsets(
                    self.cfg, s2t_idx
                )
            else:
                sfb_tmem_delta = s2t_idx * TMEM_SF_UTCCP_COLS_PER_COPY
                increment_s2t = 32 * (s2t_idx // 2) + 128 * (s2t_idx % 2)
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
        first_k_tile: Constexpr[Optional[bool]] = None,
    ) -> None:
        self._mma_impl(
            stage_info,
            desc_a_mma_base,
            desc_b_mma_base,
            smem_a_stage_ptr=smem_a_stage_ptr,
            sfa_stage_col_offset=sfa_stage_col_offset,
            sfb_stage_col_offset=sfb_stage_col_offset,
            first_k_tile=first_k_tile,
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
        first_k_tile: Constexpr[Optional[bool]] = None,
    ) -> None:
        self._mma_impl(
            stage_info,
            Int64(0),
            desc_b_mma_base,
            tmem_cast_a_addr=tmem_cast_a_addr,
            first_k_tile=first_k_tile,
        )

    @cute.jit
    def _mma_impl(
        self,
        stage_info: StageInfo,
        desc_a_mma_base,
        desc_b_mma_base,
        *,
        smem_a_stage_ptr=None,
        sfa_stage_col_offset=None,
        sfb_stage_col_offset=None,
        tmem_cast_a_addr=None,
        first_k_tile: Constexpr[Optional[bool]] = None,
        k_subblock: Constexpr[Optional[int]] = None,
    ) -> None:
        """Execute MMA (accumulates across K-tiles). Branches on BF16 vs FP4.

        ``k_subblock`` restricts the issue to one DeepSeek scale block of the K
        tile (``deepseek_fp8_block_size // mma_k`` MMAs starting from a fresh
        accumulator). The caller acquires one accumulator stage per sub-block.
        """
        desc_a_mma = desc_a_mma_base
        desc_b_mma = desc_b_mma_base
        stage_col_offset = stage_info.stage_idx * self.cfg.tmem_overlap_window_cols
        if cutlass.const_expr(self.cfg.uses_tmem_overlap_window):
            stage_col_offset += self._mma_local_idx * Int32(
                self.cfg.tile_n - self.cfg.epi_tile_n
            )
        acc_tmem_ptr = prims.make_tmem_ptr(
            self.tmem_raw_addr + self._alloc_c.offset + stage_col_offset,
            cutlass.Float32,
        )

        num_kblocks = self.cfg.tile_k // self.cfg.mma_k
        kblock_begin = 0
        kblock_end = num_kblocks
        if cutlass.const_expr(k_subblock is not None):
            kblocks_per_sub = self.cfg.deepseek_fp8_block_size // self.cfg.mma_k
            kblock_begin = k_subblock * kblocks_per_sub
            kblock_end = kblock_begin + kblocks_per_sub
        is_first_ktile = stage_info.loop_offset == Int32(0)
        if cutlass.const_expr(first_k_tile is not None):
            is_first_ktile = cutlass.Boolean(first_k_tile)

        if cutlass.const_expr(self.cfg.uses_plain_mma):
            # BF16 and plain FP8 paths: standard tcgen05_mma, no block scales.
            if cutlass.const_expr(self.cfg.is_fp8_mma):
                mma_kind = prims.Tcgen05MMAKind.F8F6F4
            else:
                mma_kind = prims.Tcgen05MMAKind.F16
            for kblock_idx in cutlass.range_constexpr(kblock_begin, kblock_end):
                if cutlass.const_expr(self.cfg.has_cast_a):
                    desc_a_increment = 0
                else:
                    desc_a_increment = _mma_smem_desc_increment(
                        kblock_idx,
                        mma_k=self.cfg.mma_k,
                        smem_bits=self.cfg.dtype_a_smem_bits,
                        tile_k=self.cfg.tile_k,
                    )
                desc_b_increment = _mma_smem_desc_increment(
                    kblock_idx,
                    mma_k=self.cfg.mma_k,
                    smem_bits=self.cfg.dtype_b_smem_bits,
                    tile_k=self.cfg.tile_k,
                    stage_bytes=self.cfg.num_bytes_b_smem_per_stage,
                )
                desc_b = desc_b_mma + desc_b_increment
                if cutlass.const_expr(self.cfg.has_cast_a):
                    desc_a = prims.make_tmem_ptr(
                        tmem_cast_a_addr + Int32(kblock_idx * 8), cutlass.Int32
                    )
                else:
                    desc_a = desc_a_mma + desc_a_increment

                if cutlass.const_expr(kblock_idx == kblock_begin):
                    if cutlass.const_expr(self.cfg.use_deepseek_fp8):
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
                for kblock_idx in cutlass.range_constexpr(kblock_begin, kblock_end):
                    desc_a_increment = _mma_smem_desc_increment(
                        kblock_idx,
                        mma_k=self.cfg.mma_k,
                        smem_bits=self.cfg.dtype_a_smem_bits,
                        tile_k=self.cfg.tile_k,
                    )
                    desc_b_increment = _mma_smem_desc_increment(
                        kblock_idx,
                        mma_k=self.cfg.mma_k,
                        smem_bits=self.cfg.dtype_b_smem_bits,
                        tile_k=self.cfg.tile_k,
                        stage_bytes=self.cfg.num_bytes_b_smem_per_stage,
                    )
                    desc_a = desc_a_mma_1 + desc_a_increment
                    desc_b = desc_b_mma + desc_b_increment

                    if cutlass.const_expr(kblock_idx == kblock_begin):
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
            # Block-scaled MMA with scale factors in TMEM
            if cutlass.const_expr(
                (self.cfg.has_input_block_scaling and not self.cfg.fuse_sf_copy_to_mma)
            ):
                # SF stage offsets from separate CopySf pipeline
                sfa_offset = sfa_stage_col_offset
                sfb_offset = sfb_stage_col_offset
            else:
                sfa_offset = Int32(0)
                sfb_offset = Int32(0)

            b_stage_bytes = self.cfg.num_bytes_b_smem_per_stage

            for kblock_idx in cutlass.range_constexpr(num_kblocks):
                sfa_sf_vec_idx = (
                    kblock_idx * self.cfg.mma_k
                ) // self.cfg.input_sf_block_size_a
                sfb_sf_vec_idx = (
                    kblock_idx * self.cfg.mma_k
                ) // self.cfg.input_sf_block_size_b
                sfa_sf_group_size = (
                    TMEM_SF_PACK_SIZE_BYTES * self.cfg.tmem_sf_vec4_groups_per_mma("a")
                )
                sfb_sf_group_size = (
                    TMEM_SF_PACK_SIZE_BYTES * self.cfg.tmem_sf_vec4_groups_per_mma("b")
                )
                sfa_sf_group_idx = sfa_sf_vec_idx // sfa_sf_group_size
                sfb_sf_group_idx = sfb_sf_vec_idx // sfb_sf_group_size
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

                desc_a_increment = _mma_smem_desc_increment(
                    kblock_idx,
                    mma_k=self.cfg.mma_k,
                    smem_bits=self.cfg.dtype_a_smem_bits,
                    tile_k=self.cfg.tile_k,
                )
                desc_b_increment = _mma_smem_desc_increment(
                    kblock_idx,
                    mma_k=self.cfg.mma_k,
                    smem_bits=self.cfg.dtype_b_smem_bits,
                    tile_k=self.cfg.tile_k,
                    stage_bytes=b_stage_bytes,
                )
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
                            self.idesc.set_sf_ids(
                                a_sf_id=sfa_sf_id,
                                b_sf_id=sfb_sf_id,
                            ),
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
        if cutlass.const_expr(self.cfg.uses_tmem_overlap_window):
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
    def preload_non_swap_overlap(
        self,
        stage_info: StageInfo,
    ) -> tuple[cutlass.Float32, cutlass.Float32, Int32]:
        """Copy the complete shared TMEM region before releasing its stage.

        Adjacent accumulator windows are separated by ``tile_n - epi_tile_n``;
        the shared region is therefore exactly one ``epi_tile_n`` chunk.
        """
        if cutlass.const_expr(not self.cfg.has_gated_epilogue):
            epi_t2r_repx = self.cfg.non_swap_tmem_load_num_regs
            num_calls = self.cfg.tile_n // epi_t2r_repx
            overlap_loads = self.cfg.non_swap_tmem_overlap_loads
            first_output_call = (Int32(1) - self._epi_local_idx) * Int32(
                num_calls - overlap_loads
            )
            t2r_rmem, _, t2r_output_call_idx = self._consumer_work_impl(
                stage_info, first_output_call
            )
            if cutlass.const_expr(overlap_loads == 2):
                t2r_rmem_1, _, _ = self._consumer_work_impl(
                    stage_info, first_output_call + Int32(1)
                )
            else:
                _, t2r_rmem_1 = self._t2r_default_values()
            return t2r_rmem, t2r_rmem_1, t2r_output_call_idx

        # TRTLLM-gen rotates pass zero onto the shared tail chunk while local
        # set zero is active. Each FC1 warpgroup loads the complete epi_tile_n.
        t2r_rmem, t2r_rmem_1, t2r_output_call_idx = self._consumer_work_impl(
            stage_info, Int32(0)
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
        if cutlass.const_expr(self.cfg.uses_tmem_overlap_window):
            if cutlass.const_expr(not self.cfg.is_swap_ab):
                if cutlass.const_expr(not self.cfg.has_gated_epilogue):
                    epi_t2r_repx = self.cfg.non_swap_tmem_load_num_regs
                    num_calls = self.cfg.tile_n // epi_t2r_repx
                    overlap_loads = self.cfg.non_swap_tmem_overlap_loads
                    logical_call_idx = Int32(subtile_idx) + self._epi_local_idx * Int32(
                        overlap_loads
                    )
                    t2r_rmem, t2r_rmem_1, t2r_output_call_idx = (
                        self._consumer_work_impl(stage_info, logical_call_idx)
                    )
                    if cutlass.const_expr(subtile_idx == num_calls - overlap_loads - 1):
                        self._epi_local_idx = self._epi_local_idx ^ Int32(1)
                    return t2r_rmem, t2r_rmem_1, t2r_output_call_idx

                num_calls = self.cfg.non_swap_epilogue_subtile_cnt
                logical_call_idx = Int32(subtile_idx) + Int32(1)
                t2r_rmem, t2r_rmem_1, t2r_output_call_idx = self._consumer_work_impl(
                    stage_info, logical_call_idx
                )
                if cutlass.const_expr(subtile_idx == num_calls - 2):
                    self._epi_local_idx = self._epi_local_idx ^ Int32(1)
                return t2r_rmem, t2r_rmem_1, t2r_output_call_idx
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

        non-swapAB: 32x32b, 1 reg per TMEM column, capped by
        tmem_ldst_max_num_regs.
        swapAB: 16x256b, 4 regs per 8 columns, two 16-row loads.
        """
        base_col = self.tmem_raw_addr & 0xFFFF
        base_row = self.tmem_raw_addr >> 16
        stage_col_offset = stage_info.stage_idx * self.cfg.tmem_overlap_window_cols

        warp_idx = cute.arch.warp_idx()
        warp_in_group = warp_idx % self.cfg.warp_layout.num_epilogue_warps
        warp_in_group = cute.arch.make_warp_uniform(warp_in_group)

        if cutlass.const_expr(self.cfg.is_swap_ab):
            # 16x256b: each 4-warp epilogue subgroup handles one epilogueTileN
            # column chunk.  Tile256 generated kernels use two 4-warp groups,
            # so one schedule call covers 128 columns without over-unrolling.
            warpgroup_count = max(1, self.cfg.warp_layout.num_epilogue_warps // 4)
            warpgroup_idx = nonnegative_div(warp_in_group, 4)
            cols_per_call = self.cfg.epi_tile_n * warpgroup_count
            if cutlass.const_expr(self.cfg.has_deepseek_fp8_two_epilogue):
                cols_per_call = self.cfg.epi_tile_n
            call_idx_for_tmem = Int32(logical_call_idx)
            call_idx_for_output = Int32(logical_call_idx)
            if cutlass.const_expr(self.cfg.uses_tmem_overlap_window):
                num_epilogue_tiles_n = self.cfg.tile_n // self.cfg.epi_tile_n
                actual_idx_n = logical_call_idx + warpgroup_idx
                # The tile256 max-overlap path ping-pongs two logical C tiles
                # through a shared physical window. For n epilogue chunks:
                #   local set 0 -> physical chunks 0..n-1
                #   local set 1 -> physical chunks n-1..2n-2
                # Chunk n-1 is shared. We read it before release,
                # then read the remaining chunks after release while MMA can
                # write the next set.  Wrapping local set 1 back to chunks
                # 0..n-1 corrupts multi-N FC1 shapes because epilogue stores
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
            swap_t2r_repx = self.cfg.swap_tmem_load_reps
            swap_t2r_regs = self.cfg.swap_tmem_load_num_regs
            swap_t2r_loads = self.cfg.swap_tmem_overlap_loads
            slice0_vals = []
            slice1_vals = []
            if cutlass.const_expr(self.cfg.has_deepseek_fp8_two_epilogue):
                addr = addr + warpgroup_idx * Int32(0x100000)
                for load_idx in cutlass.range_constexpr(swap_t2r_loads):
                    load_addr = addr + Int32(load_idx * self.cfg.swap_tmem_load_cols)
                    tmem = prims.make_tmem_ptr(load_addr, cutlass.Float32)
                    load_slice0 = prims.tcgen05_ld(shape, tmem, num=swap_t2r_repx)
                    for reg_idx in cutlass.range_constexpr(swap_t2r_regs):
                        slice0_vals.append(load_slice0[reg_idx])
                prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
                cute.arch.fence_view_async_tmem_load()
                slice0 = cutlass.Vector.from_elements(
                    tuple(slice0_vals), dtype=cutlass.Float32
                )
                slice1 = cutlass.vector.full(
                    [self.cfg.epi_tile_n // 2], 0.0, cutlass.Float32
                )
                return (slice0, slice1, call_idx_for_output)
            for load_idx in cutlass.range_constexpr(swap_t2r_loads):
                load_addr = addr + Int32(load_idx * self.cfg.swap_tmem_load_cols)
                tmem0 = prims.make_tmem_ptr(load_addr, cutlass.Float32)
                tmem1 = prims.make_tmem_ptr(load_addr + 0x100000, cutlass.Float32)
                load_slice0 = prims.tcgen05_ld(shape, tmem0, num=swap_t2r_repx)
                load_slice1 = prims.tcgen05_ld(shape, tmem1, num=swap_t2r_repx)
                for reg_idx in cutlass.range_constexpr(swap_t2r_regs):
                    slice0_vals.append(load_slice0[reg_idx])
                    slice1_vals.append(load_slice1[reg_idx])
            prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
            cute.arch.fence_view_async_tmem_load()
            slice0 = cutlass.Vector.from_elements(
                tuple(slice0_vals), dtype=cutlass.Float32
            )
            slice1 = cutlass.Vector.from_elements(
                tuple(slice1_vals), dtype=cutlass.Float32
            )
            return (slice0, slice1, call_idx_for_output)
        else:
            # 32x32b: 1 reg per column
            epi_t2r_repx = self.cfg.non_swap_tmem_load_num_regs
            warpgroup_idx = warp_in_group // Int32(WARPS_PER_WARPGROUP)
            warp_in_epi4 = warp_in_group % Int32(WARPS_PER_WARPGROUP)
            warpgroup_idx = cute.arch.make_warp_uniform(warpgroup_idx)
            warp_in_epi4 = cute.arch.make_warp_uniform(warp_in_epi4)
            row_offset = warp_in_epi4 * WARP_SIZE
            if cutlass.const_expr(self.cfg.has_gated_epilogue):
                steps_per_epi_tile = self.cfg.non_swap_gated_steps_per_epi_tile
                step_idx = logical_call_idx % Int32(steps_per_epi_tile)
                epi_tile_call_idx = logical_call_idx // Int32(steps_per_epi_tile)
                output_group_idx = (
                    epi_tile_call_idx
                    * Int32(self.cfg.non_swap_epilogue_warpgroup_count)
                    + warpgroup_idx
                )
                tmem_group_idx = output_group_idx
            else:
                output_group_idx = Int32(logical_call_idx)
                col_offset = logical_call_idx * Int32(epi_t2r_repx)

            if cutlass.const_expr(
                self.cfg.uses_tmem_overlap_window and self.cfg.has_gated_epilogue
            ):
                # Match TRTLLM-gen's four-chunk rotation exactly.  The two MMA
                # windows occupy chunks [0, 3] and [3, 6], sharing chunk 3.
                # Pass zero reads the shared chunk before release:
                #   local 0: physical/output (wg + 3) % 4
                #   local 1: physical wg + 3, output wg
                # Pass one applies the same mapping to logical chunks 2 and 3.
                num_output_groups = self.cfg.tile_n // self.cfg.epi_tile_n
                offset_group_idx = output_group_idx + Int32(num_output_groups - 1)
                wrapped_group_idx = offset_group_idx % Int32(num_output_groups)
                local_set_is_zero = self._epi_local_idx == Int32(0)
                tmem_group_idx = prims.inline_ptx_hl(
                    "selp.s32 {$w0}, {$r0}, {$r1}, {$r2};",
                    write_only_types=[Int32],
                    read_only_args=[
                        wrapped_group_idx,
                        offset_group_idx,
                        local_set_is_zero,
                    ],
                )
                output_group_idx = prims.inline_ptx_hl(
                    "selp.s32 {$w0}, {$r0}, {$r1}, {$r2};",
                    write_only_types=[Int32],
                    read_only_args=[
                        wrapped_group_idx,
                        output_group_idx,
                        local_set_is_zero,
                    ],
                )
            if cutlass.const_expr(self.cfg.has_gated_epilogue):
                col_offset = tmem_group_idx * Int32(
                    self.cfg.epi_tile_n
                ) + step_idx * Int32(self.cfg.non_swap_gated_accumulator_cols_per_step)
                # Keep the output index group-major; GmemC converts each step
                # to its first TMEM fragment index.
                output_group_idx = (
                    output_group_idx * Int32(steps_per_epi_tile) + step_idx
                )
            elif cutlass.const_expr(self.cfg.uses_tmem_overlap_window):
                col_offset += self._epi_local_idx * Int32(
                    self.cfg.tile_n - self.cfg.epi_tile_n
                )
            col_id = base_col + self._alloc_c.offset + stage_col_offset + col_offset
            current_addr = ((base_row + row_offset) << 16) | col_id

            shape = "32x32b"
            tmem = prims.make_tmem_ptr(current_addr, cutlass.Float32)
            c_rmem = prims.tcgen05_ld(shape, tmem, num=max(1, epi_t2r_repx))
            if cutlass.const_expr(self.cfg.non_swap_epilogue_fragments_per_group == 2):
                tmem_1 = prims.make_tmem_ptr(
                    current_addr + Int32(epi_t2r_repx), cutlass.Float32
                )
                t2r_rmem_1 = prims.tcgen05_ld(shape, tmem_1, num=max(1, epi_t2r_repx))
            else:
                _, t2r_rmem_1 = self._t2r_default_values()
            prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
            cute.arch.fence_view_async_tmem_load()
            return (
                c_rmem,
                t2r_rmem_1,
                output_group_idx,
            )
