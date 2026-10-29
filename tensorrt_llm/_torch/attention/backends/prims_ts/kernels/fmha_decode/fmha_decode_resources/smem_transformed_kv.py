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

"""Transformed K/V SMEM resource for mixed-dtype FMHA decode."""

from dataclasses import dataclass
from typing import ClassVar

import cutlass
import cutlass.cute as cute
from cutlass import Int32
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

from ...placeholder_helpers import _placeholder_smem_array
from ..fmha_decode_config import FmhaDecodeConfig
from ..fmha_decode_constants import KV_INST0, KV_KIND_K, KV_KIND_V
from .helpers_common import (
    _TASK_CACHE_WARP_GRP_THREAD_IDX,
    Constexpr,
    DecodeGenResourceBase,
    ResourceVars,
    _decode_gen_task_cache,
    _major_k_stride_bytes,
    _qkv_smem_swizzle,
)
from .smem_resources import SmemKvResource, SmemKvTileResource


@cute.jit
def _cvt_e4m3x2_word_to_bf16x2(packed_word: Int32, shift: Constexpr[int]) -> Int32:
    """Extract an E4M3x2 half-word and convert it to BF16x2."""
    packed_fp8 = (packed_word >> Int32(shift)) & Int32(0xFFFF)
    return cute.arch.inline_ptx(
        "{ .reg .b8 f0, f1, u0, u1; .reg .b16 fp8, h0, h1, b0, b1; "
        ".reg .b32 half2; "
        "mov.b32 {f0, f1, u0, u1}, {$r0}; "
        "mov.b16 fp8, {f0, f1}; cvt.rn.f16x2.e4m3x2 half2, fp8; "
        "mov.b32 {h0, h1}, half2; "
        "cvt.rn.bf16.f16 b0, h0; cvt.rn.bf16.f16 b1, h1; "
        "mov.b32 {$w0}, {b0, b1}; }",
        write_only_types=[Int32],
        read_only_args=[packed_fp8],
    )


@cute.jit
def _cvt_e4m3x2_byte_to_bf16x2(packed_byte: Int32) -> Int32:
    """Broadcast one E4M3 byte and convert the pair to BF16x2."""
    sf_pair = (packed_byte & Int32(0xFF)) * Int32(0x101)
    return cute.arch.inline_ptx(
        "{ .reg .b8 f0, f1, u0, u1; .reg .b16 fp8, h0, h1, b0, b1; "
        ".reg .b32 half2; "
        "mov.b32 {f0, f1, u0, u1}, {$r0}; "
        "mov.b16 fp8, {f0, f1}; cvt.rn.f16x2.e4m3x2 half2, fp8; "
        "mov.b32 {h0, h1}, half2; "
        "cvt.rn.bf16.f16 b0, h0; cvt.rn.bf16.f16 b1, h1; "
        "mov.b32 {$w0}, {b0, b1}; }",
        write_only_types=[Int32],
        read_only_args=[sf_pair],
    )


@cute.jit
def _cvt_e2m1_word_to_e4m3_words(
    packed_word: Int32, sf_byte: Int32
) -> tuple[Int32, Int32]:
    """Scale eight dense-packed E2M1 values and return two E4M3 words."""
    sf_pair = (sf_byte & Int32(0xFF)) * Int32(0x101)
    return cute.arch.inline_ptx(
        """
        {
            .reg .b16 sf;
            .reg .b32 sfFp16x2;
            .reg .b8 b0, b1, b2, b3;
            .reg .b32 h0, h1, h2, h3;
            .reg .b16 e0, e1, e2, e3;
            mov.b32 {sf, _}, {$r1};
            cvt.rn.f16x2.e4m3x2 sfFp16x2, sf;
            mov.b32 {b0, b1, b2, b3}, {$r0};
            cvt.rn.f16x2.e2m1x2 h0, b0;
            cvt.rn.f16x2.e2m1x2 h1, b1;
            cvt.rn.f16x2.e2m1x2 h2, b2;
            cvt.rn.f16x2.e2m1x2 h3, b3;
            mul.rn.f16x2 h0, h0, sfFp16x2;
            mul.rn.f16x2 h1, h1, sfFp16x2;
            mul.rn.f16x2 h2, h2, sfFp16x2;
            mul.rn.f16x2 h3, h3, sfFp16x2;
            cvt.rn.satfinite.e4m3x2.f16x2 e0, h0;
            cvt.rn.satfinite.e4m3x2.f16x2 e1, h1;
            cvt.rn.satfinite.e4m3x2.f16x2 e2, h2;
            cvt.rn.satfinite.e4m3x2.f16x2 e3, h3;
            mov.b32 {$w0}, {e0, e1};
            mov.b32 {$w1}, {e2, e3};
        }
        """,
        write_only_types=[Int32, Int32],
        read_only_args=[packed_word, sf_pair],
    )


@cute.jit
def _cvt_e2m1x2_byte_to_bf16x2(packed_byte: Int32) -> Int32:
    """Convert a packed E2M1x2 byte to BF16x2."""
    return cute.arch.inline_ptx(
        "{ .reg .b8 byte, u0, u1, u2; .reg .b16 h0, h1, b0, b1; "
        ".reg .b32 h; mov.b32 {byte, u0, u1, u2}, {$r0}; "
        "cvt.rn.f16x2.e2m1x2 h, byte; mov.b32 {h0, h1}, h; "
        "cvt.rn.bf16.f16 b0, h0; cvt.rn.bf16.f16 b1, h1; "
        "mov.b32 {$w0}, {b0, b1}; }",
        write_only_types=[Int32],
        read_only_args=[packed_byte & Int32(0xFF)],
    )


@dataclass(kw_only=True)
class SmemTransformedKvResource(DecodeGenResourceBase):
    """Shared transformed K/V staging resource for mixed Q/KV decode.

    The upstream ``SmemKvResource`` or ``SmemKvTileResource`` hold raw K/V
    in their GMEM dtype. Mixed precision KV introduces this async producer
    between raw K/V TMA and MMA so BMM1/BMM2 consume the Q-side MMA input dtype.
    """

    _task_local_specs: ClassVar[tuple[tuple, ...]] = (
        (
            "kv_desc_slot",
            prims.Tcgen05SmemDesc,
            prims.Tcgen05SmemDesc(0),
            "SMEM descriptor for transformed K loads consumed by QK MMA.",
        ),
        (
            "v_desc_0_slot",
            prims.Tcgen05SmemDesc,
            prims.Tcgen05SmemDesc(0),
            "First transformed V descriptor consumed by VP MMA.",
        ),
        (
            "v_desc_1_slot",
            prims.Tcgen05SmemDesc,
            prims.Tcgen05SmemDesc(0),
            "Second transformed V descriptor consumed by VP MMA.",
        ),
    )
    cfg: Constexpr[FmhaDecodeConfig] = None
    src_smem_kv: Constexpr[SmemKvResource | None] = None
    src_smem_k0: Constexpr[SmemKvTileResource | None] = None
    src_smem_k1: Constexpr[SmemKvTileResource | None] = None
    src_smem_v0: Constexpr[SmemKvTileResource | None] = None
    src_smem_v1: Constexpr[SmemKvTileResource | None] = None
    page_idx_kv: cute.Pointer | None = None
    num_heads_kv: Int32 | None = None
    _alloc: Constexpr[SmemAllocation | None] = None
    _smem_base_kv: cutlass.Array = None
    _k_desc_base: prims.Tcgen05SmemDesc = None
    _v_desc_base: prims.Tcgen05SmemDesc = None
    kv_desc_slot: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    v_desc_0_slot: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    v_desc_1_slot: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()

    def _init_placeholder_state(self) -> None:
        """Create placeholder state for the shared K/V SMEM ring."""
        num_stages = self.pipeline_config.num_stages
        self._smem_base_kv = _placeholder_smem_array(
            self.cfg.qk_dtype,
            (self.cfg.smem_transformed_kv_tile_bytes // self.cfg.qk_dtype_bytes)
            * num_stages,
        )
        self._k_desc_base = prims.Tcgen05SmemDesc(0)
        self._v_desc_base = prims.Tcgen05SmemDesc(0)

    def get_smem_requirements(self) -> list[SmemAllocation]:
        """Allocate the shared K/V staged SMEM ring."""
        num_stages = (
            self.pipeline_config.num_stages
            if self.pipeline_config is not None
            else self.cfg.transformed_kv_stages
        )
        if self._alloc is None:
            self._alloc = SmemAllocation(
                name=f"{self.name}",
                size_bytes=self.cfg.smem_transformed_kv_tile_bytes * num_stages,
                alignment=self.cfg.stensor_align,
            )
        return [self._alloc]

    def get_tmem_requirements(self) -> list[TmemAllocation]:
        """Return no TMEM requirements because this resource uses only SMEM."""
        return []

    @cute.jit
    def _create_initial_task_locals(
        self, context: ResourceContext | None = None
    ) -> ResourceVars:
        """Bind the shared K/V ring and build K/V base descriptors."""
        if cutlass.const_expr(context is not None and context.smem_base is not None):
            num_stages = self.pipeline_config.num_stages
            stage_elems = (
                self.cfg.smem_transformed_kv_tile_bytes // self.cfg.qk_dtype_bytes
            )
            self._smem_base_kv = cutlass.Array(
                context.smem_base.data_ptr() + self._alloc.offset,
                dtype=self.cfg.qk_dtype,
                shape=(stage_elems * num_stages,),
                addrspace=3,
            )
            kv_tile_bytes = Int32(self.cfg.smem_transformed_kv_tile_bytes)
            k_leading_byte_offset = Int32(16384)
            stride_byte_offset = Int32(1024)
            if cutlass.const_expr(self.cfg.use_fp8_qk):
                k_leading_byte_offset = kv_tile_bytes
                stride_byte_offset = Int32(
                    _major_k_stride_bytes(
                        self.cfg.qk_dtype_bytes, self.cfg.head_dim_kv_stage
                    )
                )
            v_leading_byte_offset = k_leading_byte_offset
            if cutlass.const_expr(
                self.cfg.use_fp8_qk or self.cfg.head_dim_kv_stage == 64
            ):
                v_leading_byte_offset = Int32(0)
            self._k_desc_base = prims.Tcgen05SmemDesc.build(
                self._smem_base_kv,
                leading_byte_offset=k_leading_byte_offset,
                stride_byte_offset=stride_byte_offset,
                layout=_qkv_smem_swizzle(self.cfg),
            )
            self._v_desc_base = prims.Tcgen05SmemDesc.build(
                self._smem_base_kv,
                leading_byte_offset=v_leading_byte_offset,
                stride_byte_offset=stride_byte_offset,
                layout=_qkv_smem_swizzle(self.cfg),
            )
        return {
            "kv_desc": cutlass.Int64(0),
            "v_desc_0": cutlass.Int64(0),
            "v_desc_1": cutlass.Int64(0),
        }

    @cute.jit
    def _create_work_tile_task_locals(
        self, context: ResourceContext | None = None
    ) -> ResourceVars:
        """Provide shared K/V descriptor slots for one work tile."""
        _ = context
        return {
            "kv_desc": cutlass.Int64(0),
            "v_desc_0": cutlass.Int64(0),
            "v_desc_1": cutlass.Int64(0),
        }

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_load_state(self, stage_info: StageInfo) -> None:
        """Initialize producer-side shared K/V SMEM state."""
        # ProdAuxWork: bind the shared K/V ring and descriptor bases before
        # the transform task alternates K and V stages through it.
        self._create_initial_task_locals(stage_info.context)

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_descriptor_state(self, stage_info: StageInfo) -> None:
        """Initialize consumer-side shared K/V descriptor state."""
        # ConsAuxWork: initialize descriptor slots for both K and V consumers
        # of the shared K/V ring.
        self._create_initial_task_locals(stage_info.context)

    @cute.jit
    def _dst_q_smem_byte_offset(self, elem_idx: Int32) -> Int32:
        """Map a logical transformed-K/V element to its swizzled SMEM byte offset."""
        cfg = self.cfg
        tile_row_idx = elem_idx // Int32(cfg.head_dim_kv_stage)
        tile_col_idx = elem_idx % Int32(cfg.head_dim_kv_stage)
        if cutlass.const_expr(cfg.use_fp8_qk and cfg.head_dim_kv_stage == 64):
            swizzle_bytes = 64
        else:
            swizzle_bytes = 128
        q_elems_per_smem_row = swizzle_bytes // cfg.qk_dtype_bytes
        smem_slice_idx = tile_col_idx // Int32(q_elems_per_smem_row)
        smem_col_idx = tile_col_idx % Int32(q_elems_per_smem_row)
        smem_row_idx = smem_slice_idx * Int32(cfg.tile_size_kv) + tile_row_idx
        dst_byte_offset = smem_row_idx * Int32(swizzle_bytes) + smem_col_idx * Int32(
            cfg.qk_dtype_bytes
        )
        if cutlass.const_expr(cfg.use_fp8_qk and cfg.head_dim_kv_stage == 64):
            return dst_byte_offset ^ (
                ((dst_byte_offset >> Int32(7)) & Int32(3)) << Int32(4)
            )
        return dst_byte_offset ^ ((smem_row_idx & Int32(7)) << Int32(4))

    @cute.jit
    def _transform_load_fp8_chunk(
        self,
        chunk_idx: Int32,
        elem_idx: Int32,
        src_word_base: cutlass.Array,
        dst_pair_base: cutlass.Array,
    ) -> None:
        """Convert one eight-element FP8 chunk to BF16 in transformed SMEM."""
        cfg = self.cfg
        src_word_idx = chunk_idx * Int32(2)
        if cutlass.const_expr(cfg.head_dim_kv_stage == 64):
            # H64 transform chunks always cover eight consecutive FP8
            # values, so derive the TMA-swizzled source word directly from
            # the chunk index instead of expanding to elem row/column first.
            src_row_idx = chunk_idx >> Int32(3)
            src_sector_idx = (src_row_idx << Int32(2)) + (
                (chunk_idx & Int32(7)) >> Int32(1)
            )
            src_swizzled_sector_idx = src_sector_idx ^ (
                (src_row_idx >> Int32(1)) & Int32(3)
            )
            src_word_idx = src_swizzled_sector_idx * Int32(4) + (
                chunk_idx & Int32(1)
            ) * Int32(2)
        else:
            # The 128-byte TMA swizzle XORs bits [6:4] with the logical row
            # modulo eight. Convert that byte-address mapping to an Int32 word
            # index before loading the raw FP8 chunk.
            src_row_idx = elem_idx // Int32(cfg.head_dim_kv_stage)
            src_word_idx = src_word_idx ^ ((src_row_idx & Int32(7)) << Int32(2))
        packed_fp8_word0 = src_word_base[src_word_idx]
        packed_fp8_word1 = src_word_base[src_word_idx + Int32(1)]
        dst_pair_idx = self._dst_q_smem_byte_offset(elem_idx) >> Int32(2)
        bf16_pair0 = _cvt_e4m3x2_word_to_bf16x2(packed_fp8_word0, 0)
        bf16_pair1 = _cvt_e4m3x2_word_to_bf16x2(packed_fp8_word0, 16)
        bf16_pair2 = _cvt_e4m3x2_word_to_bf16x2(packed_fp8_word1, 0)
        bf16_pair3 = _cvt_e4m3x2_word_to_bf16x2(packed_fp8_word1, 16)
        dst_pair_base[dst_pair_idx] = bf16_pair0
        dst_pair_base[dst_pair_idx + Int32(1)] = bf16_pair1
        dst_pair_base[dst_pair_idx + Int32(2)] = bf16_pair2
        dst_pair_base[dst_pair_idx + Int32(3)] = bf16_pair3

    @cute.jit
    def _load_nvfp4_scale_byte(
        self,
        src_sf_base: cutlass.Array,
        head_dim_stage_idx: Constexpr[int],
        is_v: int,
        elem_idx: Int32,
    ) -> Int32:
        """Read one E4M3 scale-factor byte from the SMEM-staged SF tile.

        The SF for the raw K/V tile is TMA'd into ``SmemKvResource``'s SF ring
        (``src_sf_base`` is the matching stage's Uint8 slice). K uses
        token-major, SF-minor order; V uses TRT-LLM's 4-token interleaved
        layout. ``sf_idx`` includes the logical staged head-slice offset.
        """
        cfg = self.cfg
        sf_bytes_per_token = Int32(cfg.smem_kv_sf_bytes_per_token)
        row_idx = elem_idx // Int32(cfg.head_dim_kv_stage)
        sf_idx = Int32(head_dim_stage_idx * (cfg.head_dim_kv_stage // 16)) + (
            elem_idx % Int32(cfg.head_dim_kv_stage)
        ) // Int32(16)
        sf_offset = row_idx * sf_bytes_per_token + sf_idx
        if cutlass.const_expr(is_v):
            sf_offset = (
                (row_idx // Int32(4)) * Int32(4) * sf_bytes_per_token
                + sf_idx * Int32(4)
                + row_idx % Int32(4)
            )
        return Int32(src_sf_base[sf_offset])

    @cute.jit
    def _transform_load_nvfp4_chunk(
        self,
        stage_info: StageInfo,
        head_dim_stage_idx: Constexpr[int],
        inst_id: int,
        is_v: int,
        chunk_idx: Int32,
        elem_idx: Int32,
        src_word_base: cutlass.Array,
        src_sf_base: cutlass.Array,
        dst_word_base: cutlass.Array,
        dst_pair_base: cutlass.Array,
    ) -> None:
        """Dequantize one eight-element NVFP4 chunk to the effective MMA dtype."""
        cfg = self.cfg
        packed_word = src_word_base[chunk_idx]
        sf_byte = self._load_nvfp4_scale_byte(
            src_sf_base, head_dim_stage_idx, is_v, elem_idx
        )
        dst_word_idx = self._dst_q_smem_byte_offset(elem_idx) >> Int32(2)
        if cutlass.const_expr(cfg.use_fp8_qk):
            fp8_word0, fp8_word1 = _cvt_e2m1_word_to_e4m3_words(packed_word, sf_byte)
            dst_word_base[dst_word_idx] = fp8_word0
            dst_word_base[dst_word_idx + Int32(1)] = fp8_word1
        else:
            sf_bf16x2 = _cvt_e4m3x2_byte_to_bf16x2(sf_byte)
            for pair_idx in cutlass.range_constexpr(4):
                packed_byte = (packed_word >> Int32(pair_idx * 8)) & Int32(0xFF)
                bf16_pair = _cvt_e2m1x2_byte_to_bf16x2(packed_byte)
                dst_pair_base[dst_word_idx + Int32(pair_idx)] = prims.mul_bf16x2(
                    bf16_pair, sf_bf16x2
                )

    @cute.jit
    def _transform_load(
        self,
        stage_info: StageInfo,
        inst_id: int,
        is_v: int,
        *,
        head_dim_stage_idx: Constexpr[int],
    ) -> None:
        """Transform one raw K or V pipeline stage into MMA-ready SMEM."""
        cfg = self.cfg
        task_cache = _decode_gen_task_cache(stage_info)
        warp_grp_thread_idx = task_cache[_TASK_CACHE_WARP_GRP_THREAD_IDX]
        transform_thread_idx = warp_grp_thread_idx - Int32(
            (cfg.transform_kv_warp_idx % 4) * 32
        )

        dst_stage_elems = cfg.smem_transformed_kv_tile_bytes // cfg.qk_dtype_bytes
        dst_base = self._smem_base_kv.subview(stage_info.stage_idx * dst_stage_elems)
        dst_word_base = cutlass.Array(
            dst_base.data_ptr(),
            dtype=Int32,
            shape=(cfg.smem_transformed_kv_tile_bytes // 4,),
            addrspace=3,
        )
        dst_pair_base = cutlass.Array(
            dst_base.data_ptr(),
            dtype=Int32,
            shape=(cfg.smem_transformed_kv_tile_bytes // 4,),
            addrspace=3,
        )

        src = self.src_smem_kv
        if cutlass.const_expr(self.src_smem_kv is None):
            if cutlass.const_expr(is_v):
                src = self.src_smem_v0 if inst_id == KV_INST0 else self.src_smem_v1
            else:
                src = self.src_smem_k0 if inst_id == KV_INST0 else self.src_smem_k1
        assert src is not None
        src_stage_idx = src.state_src.consumer_work_stage
        src_stage_elems = cfg.smem_kv_tile_elements
        src_base = src._smem_base_kv.subview(src_stage_idx * src_stage_elems)
        src_word_base = cutlass.Array(
            src_base.data_ptr(),
            dtype=Int32,
            shape=(cfg.smem_kv_storage_tile_bytes // 4,),
            addrspace=3,
        )
        # SF for this raw-KV stage, staged into SmemKv's SF ring by the loader.
        src_sf_base = None
        if cutlass.const_expr(cfg.use_nvfp4_kv):
            src_sf_base = src._sf_stage_base(src_stage_idx)
        total_elems = cfg.tile_size_kv * cfg.head_dim_kv_stage
        num_threads = cfg.transform_kv_num_warps * 32
        chunks_per_thread = total_elems // (num_threads * 8)
        for e in cutlass.range(chunks_per_thread, unroll=8):
            chunk_idx = Int32(e * num_threads) + transform_thread_idx
            elem_idx = chunk_idx * Int32(8)
            if cutlass.const_expr(cfg.use_nvfp4_kv):
                self._transform_load_nvfp4_chunk(
                    stage_info,
                    head_dim_stage_idx,
                    inst_id,
                    is_v,
                    chunk_idx,
                    elem_idx,
                    src_word_base,
                    src_sf_base,
                    dst_word_base,
                    dst_pair_base,
                )
            else:
                self._transform_load_fp8_chunk(
                    chunk_idx,
                    elem_idx,
                    src_word_base,
                    dst_pair_base,
                )

        cute.arch.fence_view_async_shared()

    @producer_work
    @cute.jit
    def transform_k0(
        self, stage_info: StageInfo, *, head_dim_stage_idx: Constexpr[int]
    ) -> None:
        """Transform K instance 0 into the acquired destination stage."""
        self._transform_load(stage_info, 0, 0, head_dim_stage_idx=head_dim_stage_idx)

    @producer_work
    @cute.jit
    def transform_k1(
        self, stage_info: StageInfo, *, head_dim_stage_idx: Constexpr[int]
    ) -> None:
        """Transform K instance 1 into the acquired destination stage."""
        self._transform_load(stage_info, 1, 0, head_dim_stage_idx=head_dim_stage_idx)

    @producer_work
    @cute.jit
    def transform_v0(
        self, stage_info: StageInfo, *, head_dim_stage_idx: Constexpr[int]
    ) -> None:
        """Transform V instance 0 into the acquired destination stage."""
        self._transform_load(stage_info, 0, 1, head_dim_stage_idx=head_dim_stage_idx)

    @producer_work
    @cute.jit
    def transform_v1(
        self, stage_info: StageInfo, *, head_dim_stage_idx: Constexpr[int]
    ) -> None:
        """Transform V instance 1 into the acquired destination stage."""
        self._transform_load(stage_info, 1, 1, head_dim_stage_idx=head_dim_stage_idx)

    @cute.jit
    def _build_kv_desc(self, stage_info: StageInfo, is_v: int) -> prims.Tcgen05SmemDesc:
        """Advance the shared K or V descriptor to the committed stage."""
        # Consumers see the same descriptor layout for every stage; only the
        # base address advances by the committed SMEM stage index.
        stage_offset_bytes = stage_info.stage_idx * Int32(
            self.cfg.smem_transformed_kv_tile_bytes
        )
        return (
            self._v_desc_base.advance_start_address(stage_offset_bytes)
            if cutlass.const_expr(is_v)
            else self._k_desc_base.advance_start_address(stage_offset_bytes)
        )

    @consumer_work(returns=kv_desc_slot)
    @cute.jit
    def k_desc_0(self, stage_info: StageInfo) -> prims.Tcgen05SmemDesc:
        """Publish the first K descriptor consumed by QK MMA."""
        # ConsWork: expose the committed shared-ring K stage for QK instance 0.
        return self._build_kv_desc(stage_info, KV_KIND_K)

    @consumer_work(returns=kv_desc_slot)
    @cute.jit
    def k_desc_1(self, stage_info: StageInfo) -> prims.Tcgen05SmemDesc:
        """Publish the second K descriptor consumed by QK MMA."""
        # ConsWork: expose the committed shared-ring K stage for QK instance 1.
        return self._build_kv_desc(stage_info, KV_KIND_K)

    @consumer_work(returns=v_desc_0_slot)
    @cute.jit
    def v_desc_0(self, stage_info: StageInfo) -> prims.Tcgen05SmemDesc:
        """Publish the first V descriptor consumed by PV MMA."""
        # ConsWork: expose the committed shared-ring V stage for PV instance 0.
        return self._build_kv_desc(stage_info, KV_KIND_V)

    @consumer_work(returns=v_desc_1_slot)
    @cute.jit
    def v_desc_1(self, stage_info: StageInfo) -> prims.Tcgen05SmemDesc:
        """Publish the second V descriptor consumed by PV MMA."""
        # ConsWork: expose the committed shared-ring V stage for PV instance 1.
        return self._build_kv_desc(stage_info, KV_KIND_V)
