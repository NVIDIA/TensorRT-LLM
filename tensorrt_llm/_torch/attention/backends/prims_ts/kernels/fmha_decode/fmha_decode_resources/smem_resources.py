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

"""SMEM-side resources for FMHA decode TS kernel.

Holds ``SmemQResource``, ``SmemPageOffsetsKvResource``, and ``SmemKvResource``
— the Q tile, paged-KV page-table cache, and shared K/V SMEM ring,
respectively. All three are TMA producers and tcgen05-descriptor consumers.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

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

from ...._block_sparse.common import (
    _block_sparse_kv_atom_size,
    _prepared_kv_routes_are_block_aligned,
)
from ..fmha_decode_config import FmhaDecodeConfig
from ..fmha_decode_constants import (
    KV_INST0,
    KV_INST1,
    KV_KIND_K,
    KV_KIND_V,
    KV_TILE_256_K_SLOT_FOR_SEMANTIC_ATOM,
)
from ...stage import FmhaStage
from ...tensor_map import transform_ragged_coords
from ...placeholder_helpers import (
    _placeholder_local_array,
    _placeholder_smem_array,
)
from .helpers_common import (
    _TASK_CACHE_KV_PAGE_IDX_UB,
    _TASK_CACHE_KV_RAW_TILE_BASE,
    Constexpr,
    DecodeGenResourceBase,
    ResourceVars,
    _clamp_valid_tile_idx,
    _decode_gen_task_cache,
    _logical_head_batch,
    _logical_q_group_idx,
    _major_k_stride_bytes,
    _q_group_token_base,
    _qkv_smem_swizzle,
)
from .helpers_kv_tile_idx import (
    _load_runtime_seq_len_kv,
    _num_skipped_kv_tiles,
    _runtime_clamp_valid_tile_idx,
    _runtime_last_valid_page_idx,
    _runtime_split_kv_global_tile_idx,
    _static_split_kv_global_tile_idx,
)

if TYPE_CHECKING:
    from .smem_block_sparse_metadata import SmemBlockSparseKvMetadataResource


def _paged_sparse_kv_tma_transaction_geometry(
    *,
    tile_size_kv: int,
    kv_atom_size: int,
    head_dim_stage: int,
    kv_dtype_bytes: int,
) -> tuple[int, int, int]:
    """Return fixed copy count, bytes per copy, and bytes per paged K/V load.

    Every logical atom participates, including atoms mapped to an OOB
    coordinate. This keeps the TMA pipeline's expected transaction bytes
    independent of route validity.
    """

    chunk_head_dim = min(head_dim_stage, 64)
    assert tile_size_kv % kv_atom_size == 0
    assert head_dim_stage % chunk_head_dim == 0
    transactions_per_load = (tile_size_kv // kv_atom_size) * (
        head_dim_stage // chunk_head_dim
    )
    transaction_bytes = kv_atom_size * chunk_head_dim * kv_dtype_bytes
    return (
        transactions_per_load,
        transaction_bytes,
        transactions_per_load * transaction_bytes,
    )


@cute.jit
def _cp_async_bulk_tensor_4d_shared_cta_global_predicated(
    dst_mem: cutlass.Array,
    tma_desc: cutlass.Pointer,
    coordinates: tuple[Int32, Int32, Int32, Int32],
    mbar: cutlass.Array,
) -> None:
    """Issue one 4-D TMA load without branching the producer warp."""
    c0, c1, c2, c3 = coordinates
    cute.arch.inline_ptx(
        "cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
        "[{$r0}], [{$r1}, {{$r3}, {$r4}, {$r5}, {$r6}}], [{$r2}];",
        read_only_args=[
            dst_mem.data_ptr().toint(cutlass.Int32),
            tma_desc.toint(cutlass.Int64),
            mbar.data_ptr().toint(cutlass.Int32),
            c0,
            c1,
            c2,
            c3,
        ],
        predicate=prims.elect_sync(),
    )


@cute.jit
def _local_kv_tile_idx_for_section(
    cfg: Constexpr[FmhaDecodeConfig],
    stage_info: StageInfo,
    inst_id: Constexpr[int],
    kv_kind: Constexpr[int],
    section: Constexpr[FmhaStage],
) -> Int32:
    """Return the local K/V tile index implied by the TS HEAD/LOOP/TAIL cadence.

    The schedule names whether a producer is K0/K1/V0/V1, while the phase
    defines where that logical tile sits in the staggered decode pipeline:

    - HEAD produces only initial K tiles.
    - LOOP produces V for the current MMA iteration and K for the next one.
    - TAIL drains the final V tiles after the last loop iteration.
    """
    num_insts_kv = Int32(cfg.num_insts_kv)
    if cutlass.const_expr(section == FmhaStage.Head):
        return Int32(inst_id)
    if cutlass.const_expr(section == FmhaStage.Loop):
        base = stage_info.loop_offset * num_insts_kv + Int32(inst_id)
        if cutlass.const_expr(kv_kind == KV_KIND_V):
            return base
        return base + num_insts_kv
    return stage_info.loop_end * num_insts_kv + Int32(inst_id)


@dataclass(kw_only=True)
class SmemQResource(DecodeGenResourceBase):
    """Q tile in SMEM. Loaded once in HEAD."""

    _task_local_specs: ClassVar[tuple[tuple, ...]] = (
        (
            "q_desc_slot",
            prims.Tcgen05SmemDesc,
            prims.Tcgen05SmemDesc(0),
            "SMEM descriptor for the staged Q tile.",
        ),
    )
    cfg: Constexpr[FmhaDecodeConfig] = None
    tma_desc_q: cutlass.Pointer | None = None
    h_k_idx: Int32 = None
    b_idx: Int32 = None
    q_group_idx: Int32 = None
    q_token_offset: Int32 = None
    seq_len_q: Int32 = None
    _alloc: Constexpr[SmemAllocation | None] = None
    _smem_base_q: cutlass.Array = None
    _q_descs: cutlass.Array = None
    q_desc_slot: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()

    def _init_placeholder_state(self) -> None:
        """Create placeholder Q SMEM and descriptor state for static analysis."""
        self._smem_base_q = _placeholder_smem_array(
            self.cfg.q_dtype,
            self.cfg.smem_q_tile_elements * self.cfg.q_stages,
        )
        self._q_descs = _placeholder_local_array(
            cutlass.Int64, self.cfg.q_stages, alignment=8
        )

    def get_smem_requirements(self) -> list[SmemAllocation]:
        """Allocate one staged Q tile buffer per Q pipeline stage."""
        if self._alloc is None:
            self._alloc = SmemAllocation(
                name=f"{self.name}",
                size_bytes=self.cfg.smem_q_tile_bytes * self.cfg.q_stages,
                alignment=self.cfg.stensor_align,
            )
        return [self._alloc]

    def get_tmem_requirements(self) -> list[TmemAllocation]:
        """Q staging uses SMEM only."""
        return []

    @cute.jit
    def _create_initial_task_locals(
        self, context: ResourceContext | None = None
    ) -> ResourceVars:
        """Bind Q SMEM and build per-stage descriptors for producer/consumer use."""
        if cutlass.const_expr(context is not None and context.smem_base is not None):
            # Bind the Q SMEM allocation and create one tcgen05 descriptor per
            # pipeline stage. Q is loaded in HEAD and reused by all BMM1 calls.
            self._smem_base_q = cutlass.Array(
                context.smem_base.data_ptr() + self._alloc.offset,
                dtype=self.cfg.q_dtype,
                shape=(self.cfg.smem_q_tile_elements * self.cfg.q_stages,),
                addrspace=3,
            )
            self._q_descs = cutlass.Array(
                cutlass.Int64,
                self.cfg.q_stages,
                space=cutlass.AddressSpace.rmem,
                alignment=8,
            )
            stage_elems = self.cfg.smem_q_tile_elements
            q_head_dim_partition = min(self.cfg.headdim, 64)
            # 16-bit Q uses two 64-column TMA loads for headDim=128. FP8 Q
            # uses a descriptor stride based on the actual MMA K-major swizzle.
            q_leading_bytes = Int32(
                self.cfg.tile_size_q * q_head_dim_partition * self.cfg.q_dtype_bytes
            )
            leading_byte_offset = q_leading_bytes
            stride_byte_offset = Int32(1024)
            if cutlass.const_expr(self.cfg.use_fp8_qkv):
                q_head_dim_stage = self.cfg.head_dim_kv_stage
                q_tile_bytes = Int32(
                    self.cfg.tile_size_q * q_head_dim_stage * self.cfg.q_dtype_bytes
                )
                leading_byte_offset = q_tile_bytes
                stride_byte_offset = Int32(
                    _major_k_stride_bytes(self.cfg.q_dtype_bytes, self.cfg.headdim)
                )
            for stage_idx in cutlass.range_constexpr(self.cfg.q_stages):
                # Advance the base address for each stage while preserving the
                # same swizzled layout parameters.
                smem_ptr = self._smem_base_q.subview(stage_idx * stage_elems)
                self._q_descs[stage_idx] = prims.Tcgen05SmemDesc.build(
                    smem_ptr,
                    leading_byte_offset=leading_byte_offset,
                    stride_byte_offset=stride_byte_offset,
                    layout=_qkv_smem_swizzle(self.cfg),
                )
        return {"q_desc": cutlass.Int64(0)}

    @cute.jit
    def _create_work_tile_task_locals(
        self, context: ResourceContext | None = None
    ) -> ResourceVars:
        """Provide the per-work-tile Q descriptor slot."""
        _ = context
        return {"q_desc": cutlass.Int64(0)}

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_load_state(self, stage_info: StageInfo) -> None:
        """Initialize Q producer-side SMEM pointers and descriptors."""
        # ProdAuxWork: bind the Q SMEM allocation and precompute descriptor
        # bases before the load task starts issuing TMA copies.
        self._create_initial_task_locals(stage_info.context)

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_descriptor_state(self, stage_info: StageInfo) -> None:
        """Initialize Q consumer-side descriptor state."""
        # ConsAuxWork: create the same descriptor slots on the MMA side so
        # q_desc() can return the stage committed by LoadTask.
        self._create_initial_task_locals(stage_info.context)

    @cute.jit
    def _tma_coords(
        self,
        logical_q_group_idx: Int32,
        logical_h_k_idx: Int32,
        logical_b_idx: Int32,
        head_dim_offset: Int32,
    ) -> tuple[Int32, Int32, Int32, Int32, Int32]:
        """Map a logical Q CTA to token-major tensor-map coordinates."""
        cfg = self.cfg
        if cutlass.const_expr(cfg.use_variable_seqlens_q):
            if cutlass.const_expr(cfg.groups_tokens_heads_q):
                token_idx = logical_q_group_idx * Int32(cfg.q_tokens_per_cta)
                global_head_idx = logical_h_k_idx * Int32(cfg.heads_q_per_kv)
                tokens_per_box = cfg.q_tokens_per_cta
            else:
                head_ctas_per_token = Int32(
                    (cfg.heads_q_per_kv + cfg.tile_size_q - 1) // cfg.tile_size_q
                )
                token_idx = logical_q_group_idx // head_ctas_per_token
                head_cta_idx = logical_q_group_idx - token_idx * head_ctas_per_token
                global_head_idx = logical_h_k_idx * Int32(
                    cfg.heads_q_per_kv
                ) + head_cta_idx * Int32(cfg.tile_size_q)
                tokens_per_box = 1
            packed_coords = transform_ragged_coords(
                (
                    head_dim_offset,
                    global_head_idx,
                    self.q_token_offset + token_idx,
                ),
                ragged_dim_idx=2,
                ragged_box_size=tokens_per_box,
                ragged_extent=self.seq_len_q - token_idx,
            )
            return (
                packed_coords[0],
                packed_coords[1],
                packed_coords[2],
                packed_coords[3],
                packed_coords[4],
            )

        if cutlass.const_expr(cfg.uses_nontrivial_grouped_q_layout):
            token_idx = logical_q_group_idx * Int32(cfg.q_tokens_per_cta)
            return (
                head_dim_offset,
                Int32(0),
                logical_h_k_idx,
                token_idx,
                logical_b_idx,
            )

        if cutlass.const_expr(cfg.max_seq_len_q == 1):
            return (
                head_dim_offset,
                logical_q_group_idx * Int32(cfg.tile_size_q),
                logical_h_k_idx,
                Int32(0),
                logical_b_idx,
            )

        head_ctas_per_token = Int32(
            (cfg.heads_q_per_kv + cfg.tile_size_q - 1) // cfg.tile_size_q
        )
        token_idx = logical_q_group_idx // head_ctas_per_token
        head_cta_idx = logical_q_group_idx - token_idx * head_ctas_per_token
        return (
            head_dim_offset,
            head_cta_idx * Int32(cfg.tile_size_q),
            logical_h_k_idx,
            token_idx,
            logical_b_idx,
        )

    @cute.jit
    def _complete_grouped_q_padding(self, stage_info: StageInfo) -> None:
        """Account structural grouped rows that have no corresponding TMA."""
        cfg = self.cfg
        if cutlass.const_expr(
            cfg.groups_tokens_heads_q and cfg.q_manual_padding_rows > 0
        ):
            padding_bytes = cfg.q_manual_padding_rows * cfg.headdim * cfg.q_dtype_bytes
            prims.mbarrier_complete_tx(stage_info.barrier, Int32(padding_bytes))

    @producer_work
    @cute.jit
    def tma_load(self, stage_info: StageInfo) -> None:
        """TMA load the current staged Q tile from GMEM to SMEM."""
        cfg = self.cfg
        # ProdWork: issue the Q TMA copies for the current pipeline stage. The
        # pipeline barrier in stage_info is the handoff to the MMA consumer.
        # Resolve logical coordinates from the work tile when persistent
        # scheduling is active; otherwise use the static launch values.
        logical_h_k_idx, logical_b_idx = _logical_head_batch(
            stage_info, self.h_k_idx, self.b_idx
        )
        logical_q_group_idx = _logical_q_group_idx(cfg, stage_info, self.q_group_idx)
        # Select the SMEM stage owned by this producer fire. The TS pipeline
        # barrier attached to stage_info protects this stage until QK consumes it.
        stage_elems = cfg.smem_q_tile_elements
        stage_base = self._smem_base_q.subview(stage_info.stage_idx * stage_elems)
        if cutlass.const_expr(cfg.use_fp8_qkv):
            if prims.elect_sync():
                if cutlass.const_expr(cfg.num_head_dim_stages_kv == 1):
                    # FP8 with one head-dim stage is one tensor copy into the
                    # complete staged Q tile.
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        stage_base,
                        self.tma_desc_q,
                        self._tma_coords(
                            logical_q_group_idx,
                            logical_h_k_idx,
                            logical_b_idx,
                            Int32(0),
                        ),
                        stage_info.barrier,
                    )
                else:
                    # H256 FP8 stages Q by head-dim slices so each QK head-dim
                    # stage sees a contiguous SMEM tile.
                    q_chunk_dim = cfg.head_dim_kv_stage
                    for q_chunk_idx in cutlass.range_constexpr(
                        cfg.num_head_dim_stages_kv
                    ):
                        head_dim_offset = q_chunk_idx * q_chunk_dim
                        smem_offset = head_dim_offset * cfg.tile_size_q
                        prims.cp_async_bulk_tensor_shared_cta_global(
                            stage_base.subview(smem_offset),
                            self.tma_desc_q,
                            self._tma_coords(
                                logical_q_group_idx,
                                logical_h_k_idx,
                                logical_b_idx,
                                Int32(head_dim_offset),
                            ),
                            stage_info.barrier,
                        )
                self._complete_grouped_q_padding(stage_info)
        else:
            # 16-bit Q is staged in 64-column chunks so the SMEM layout
            # matches the tcgen05 descriptor swizzle for H64/H128/H256.
            chunk_hd = min(cfg.headdim, 64)
            num_chunks = cfg.headdim // chunk_hd
            chunk_elems = chunk_hd * cfg.tile_size_q
            if prims.elect_sync():
                # One elected lane issues all TMA chunks for this Q tile; the
                # async pipeline tracks completion through stage_info.barrier.
                for chunk_idx in cutlass.range_constexpr(num_chunks):
                    head_dim_offset = chunk_idx * chunk_hd
                    smem_offset = chunk_idx * chunk_elems
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        stage_base.subview(smem_offset),
                        self.tma_desc_q,
                        self._tma_coords(
                            logical_q_group_idx,
                            logical_h_k_idx,
                            logical_b_idx,
                            Int32(head_dim_offset),
                        ),
                        stage_info.barrier,
                    )
                self._complete_grouped_q_padding(stage_info)

    @consumer_work(returns=q_desc_slot)
    @cute.jit
    def q_desc(self, stage_info: StageInfo) -> prims.Tcgen05SmemDesc:
        """Build Q SMEM descriptor for tcgen05 MMA B operand."""
        # ConsWork: return the descriptor for the stage LoadTask committed.
        q_desc = prims.Tcgen05SmemDesc(self._q_descs[Int32(stage_info.stage_idx)])
        return q_desc

    @cute.jit
    def current_consumer_q_desc(self) -> prims.Tcgen05SmemDesc:
        """Return Q's just-waited stage without creating a routed task local.

        Packed persistent skipping guards the complete data path. Routing a Q
        descriptor from guarded HEAD into LOOP would make that descriptor a
        cross-region task local. Q's advance-on-wait pipeline already records
        the selected stage in ``consumer_work_stage``, so QK work can derive
        the same descriptor directly from the shared resource state.
        """
        stage_idx = Int32(self.state_src.consumer_work_stage)
        return prims.Tcgen05SmemDesc(self._q_descs[stage_idx])


@dataclass(kw_only=True)
class SmemKvTileResource(DecodeGenResourceBase):
    """Single K or V tile in SMEM for the split K/V decode schedule."""

    _task_local_specs: ClassVar[tuple[tuple, ...]] = (
        (
            "kv_desc_slot",
            prims.Tcgen05SmemDesc,
            prims.Tcgen05SmemDesc(0),
            "SMEM descriptor for K loads consumed by QK MMA.",
        ),
        (
            "v_desc_slot",
            prims.Tcgen05SmemDesc,
            prims.Tcgen05SmemDesc(0),
            "SMEM descriptor for V loads consumed by VP MMA.",
        ),
    )
    cfg: Constexpr[FmhaDecodeConfig] = None
    tma_desc_k: cutlass.Pointer | None = None
    tma_desc_v: cutlass.Pointer | None = None
    tma_desc_k_atom: cutlass.Pointer | None = None
    tma_desc_v_atom: cutlass.Pointer | None = None
    sparse_kv_metadata: "SmemBlockSparseKvMetadataResource | None" = None
    page_offsets_kv: "SmemPageOffsetsKvResource | None" = None
    seqlens_kv: cute.Pointer | None = None
    max_seq_len_kv: Int32 = None
    h_k_idx: Int32 = None
    b_idx: Int32 = None
    q_group_idx: Int32 = None
    seq_len_q: Int32 = None
    inst_id: Constexpr[int] = 0
    kv_kind: Constexpr[int] = KV_KIND_K
    _alloc: Constexpr[SmemAllocation | None] = None
    _smem_base_kv: cutlass.Array = None
    _desc_base: prims.Tcgen05SmemDesc = None
    kv_desc_slot: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    v_desc_slot: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()

    def _init_placeholder_state(self) -> None:
        """Create placeholder split K/V SMEM and descriptor state."""
        num_stages = (
            self.pipeline_config.num_stages if self.pipeline_config is not None else 1
        )
        self._smem_base_kv = _placeholder_smem_array(
            self.cfg.kv_dtype,
            self.cfg.smem_kv_tile_elements * num_stages,
        )
        self._desc_base = prims.Tcgen05SmemDesc(0)

    def get_smem_requirements(self) -> list[SmemAllocation]:
        """Allocate this split K or V resource's staged tile ring."""
        num_stages = (
            self.pipeline_config.num_stages if self.pipeline_config is not None else 1
        )
        if self._alloc is None:
            self._alloc = SmemAllocation(
                name=f"{self.name}",
                size_bytes=self.cfg.smem_kv_tile_bytes * num_stages,
                alignment=self.cfg.stensor_align,
            )
        return [self._alloc]

    def get_tmem_requirements(self) -> list[TmemAllocation]:
        """Split K/V staging uses SMEM only."""
        return []

    @cute.jit
    def _create_initial_task_locals(
        self, context: ResourceContext | None = None
    ) -> ResourceVars:
        """Bind the split K/V SMEM ring and build its base descriptor."""
        if cutlass.const_expr(context is not None and context.smem_base is not None):
            num_stages = (
                self.pipeline_config.num_stages
                if self.pipeline_config is not None
                else 1
            )
            self._smem_base_kv = cutlass.Array(
                context.smem_base.data_ptr() + self._alloc.offset,
                dtype=self.cfg.kv_dtype,
                shape=(self.cfg.smem_kv_tile_elements * num_stages,),
                addrspace=3,
            )
            kv_tile_bytes = Int32(
                self.cfg.tile_size_kv
                * self.cfg.head_dim_kv_stage
                * self.cfg.kv_dtype_bytes
            )
            leading_byte_offset = Int32(
                self.cfg.tile_size_kv
                * min(self.cfg.head_dim_kv_stage, 64)
                * self.cfg.kv_dtype_bytes
            )
            stride_byte_offset = Int32(1024)
            if cutlass.const_expr(self.cfg.use_fp8_qkv):
                leading_byte_offset = kv_tile_bytes
                stride_byte_offset = Int32(
                    _major_k_stride_bytes(
                        self.cfg.kv_dtype_bytes, self.cfg.head_dim_kv_stage
                    )
                )
            if cutlass.const_expr(
                self.kv_kind == KV_KIND_V
                and (self.cfg.use_fp8_qkv or self.cfg.headdim == 64)
            ):
                leading_byte_offset = Int32(0)
            self._desc_base = prims.Tcgen05SmemDesc.build(
                self._smem_base_kv,
                leading_byte_offset=leading_byte_offset,
                stride_byte_offset=stride_byte_offset,
                layout=_qkv_smem_swizzle(self.cfg),
            )
        return {"kv_desc": cutlass.Int64(0), "v_desc": cutlass.Int64(0)}

    @cute.jit
    def _create_work_tile_task_locals(
        self, context: ResourceContext | None = None
    ) -> ResourceVars:
        """Provide descriptor slots for this split K/V work tile."""
        _ = context
        return {"kv_desc": cutlass.Int64(0), "v_desc": cutlass.Int64(0)}

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_load_state(self, stage_info: StageInfo) -> None:
        """Initialize split K/V producer-side SMEM state."""
        # ProdAuxWork: bind this split K or V SMEM ring and descriptor base
        # before the load task issues any staged K/V TMA copies.
        self._create_initial_task_locals(stage_info.context)

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_descriptor_state(self, stage_info: StageInfo) -> None:
        """Initialize split K/V consumer-side descriptor state."""
        # ConsAuxWork: initialize descriptor slots for the downstream QK/PV MMA
        # task that consumes this split K or V resource.
        self._create_initial_task_locals(stage_info.context)

    @cute.jit
    def _stage_base(self, stage_info: StageInfo) -> cutlass.Array:
        """Return the SMEM base for the current split K/V pipeline stage."""
        stage_elems = self.cfg.smem_kv_tile_bytes // self.cfg.kv_dtype_bytes
        return self._smem_base_kv.subview(stage_info.stage_idx * stage_elems)

    @cute.jit
    def _local_tile_idx(
        self, stage_info: StageInfo, section: Constexpr[FmhaStage]
    ) -> Int32:
        """Map the schedule phase to the local K or V tile index."""
        return _local_kv_tile_idx_for_section(
            self.cfg, stage_info, self.inst_id, self.kv_kind, section
        )

    @cute.jit
    def _maybe_runtime_tile_idx(self, stage_info: StageInfo, tile_idx: Int32) -> Int32:
        """Apply runtime sequence and split-KV transforms to a local tile index."""
        if cutlass.const_expr(self.cfg.use_paged_kv):
            return (
                Int32(_decode_gen_task_cache(stage_info)[_TASK_CACHE_KV_RAW_TILE_BASE])
                + tile_idx
            )
        if cutlass.const_expr(
            self.seqlens_kv is None and not self.cfg.uses_runtime_q_kv_union
        ):
            if cutlass.const_expr(self.cfg.use_split_kv):
                tile_idx = _static_split_kv_global_tile_idx(
                    self.cfg, stage_info, tile_idx
                )
            tile_idx = _clamp_valid_tile_idx(self.cfg, tile_idx)
            return tile_idx + Int32(self.cfg.static_num_skipped_kv_tiles)
        seq_len_kv = _load_runtime_seq_len_kv(
            self.seqlens_kv,
            self.max_seq_len_kv,
            stage_info,
            self.h_k_idx,
            self.b_idx,
        )
        logical_q_group_idx = _logical_q_group_idx(
            self.cfg, stage_info, self.q_group_idx
        )
        q_token_base = _q_group_token_base(self.cfg, logical_q_group_idx)
        tile_idx = _runtime_split_kv_global_tile_idx(
            self.cfg,
            stage_info,
            tile_idx,
            seq_len_kv,
            self.seq_len_q,
            q_token_base,
        )
        tile_idx = _runtime_clamp_valid_tile_idx(
            self.cfg,
            tile_idx,
            seq_len_kv,
            self.seq_len_q,
            q_token_base,
        )
        return tile_idx + _num_skipped_kv_tiles(
            self.cfg,
            seq_len_kv,
            self.seq_len_q,
            q_token_base,
        )

    @cute.jit
    def _producer_load(
        self,
        stage_info: StageInfo,
        section: Constexpr[FmhaStage],
        head_dim_stage_idx: Constexpr[int],
    ) -> None:
        """Issue the staged TMA load for this split K or V tile."""
        cfg = self.cfg
        # Resolve the schedule-local K/V tile, logical head/batch coordinates,
        # and descriptor kind before selecting paged or dense addressing.
        local_tile_idx = self._local_tile_idx(stage_info, section)
        logical_h_k_idx, logical_b_idx = _logical_head_batch(
            stage_info, self.h_k_idx, self.b_idx
        )
        tma_desc = (
            self.tma_desc_v
            if cutlass.const_expr(self.kv_kind == KV_KIND_V)
            else self.tma_desc_k
        )

        if cutlass.const_expr(cfg.use_block_sparse):
            assert self.sparse_kv_metadata is not None
            assert self.tma_desc_k_atom is not None
            assert self.tma_desc_v_atom is not None
            # The positional TensorMaps keep the decode ABI stable. The
            # primary K/V descriptors are KV128 for coarse routes and one atom
            # for fine routes. The auxiliary slots always expose the atom
            # descriptor and alias the primary descriptor for fine routes.
            tma_desc_atom = (
                self.tma_desc_v_atom
                if cutlass.const_expr(self.kv_kind == KV_KIND_V)
                else self.tma_desc_k_atom
            )
            kv_atom_size = _block_sparse_kv_atom_size(cfg.kv_block_size)
            head_dim_stage = cfg.head_dim_kv_stage
            head_dim_stage_offset = head_dim_stage_idx * head_dim_stage
            chunk_hd = min(head_dim_stage, 64)
            num_chunks = head_dim_stage // chunk_hd
            tile_chunk_elems = chunk_hd * cfg.tile_size_kv
            if cutlass.const_expr(cfg.use_paged_kv):
                # Paged sparse routes retain an independent physical page ID
                # for every logical atom.  Never infer adjacency: issue the
                # same atom-sized TMA sequence for valid and invalid atoms,
                # with invalid coordinates mapped just beyond page zero so
                # the fixed mbarrier transaction count is preserved.
                atoms_per_route = cfg.tile_size_kv // kv_atom_size
                (
                    transactions_per_load,
                    transaction_bytes,
                    total_transaction_bytes,
                ) = _paged_sparse_kv_tma_transaction_geometry(
                    tile_size_kv=cfg.tile_size_kv,
                    kv_atom_size=kv_atom_size,
                    head_dim_stage=head_dim_stage,
                    kv_dtype_bytes=cfg.kv_dtype_bytes,
                )
                assert total_transaction_bytes == cfg.smem_kv_tile_bytes
                num_chunks = transactions_per_load // atoms_per_route
                chunk_hd = transaction_bytes // (kv_atom_size * cfg.kv_dtype_bytes)
                atom_chunk_elems = transaction_bytes // cfg.kv_dtype_bytes
                tile_chunk_elems = chunk_hd * cfg.tile_size_kv
                if prims.elect_sync():
                    stage_base = self._stage_base(stage_info)
                    for atom_idx in cutlass.range_constexpr(atoms_per_route):
                        token_coord, storage_coord = (
                            self.sparse_kv_metadata.route_tma_coordinate(
                                Int32(atom_idx),
                                logical_b_idx,
                            )
                        )
                        for chunk_idx in cutlass.range_constexpr(num_chunks):
                            local_head_dim_offset = chunk_idx * chunk_hd
                            global_head_dim_offset = (
                                head_dim_stage_offset + local_head_dim_offset
                            )
                            local_tile_offset = chunk_idx * tile_chunk_elems
                            prims.cp_async_bulk_tensor_shared_cta_global(
                                stage_base.subview(
                                    local_tile_offset + atom_idx * atom_chunk_elems
                                ),
                                tma_desc_atom,
                                (
                                    Int32(global_head_dim_offset),
                                    token_coord,
                                    logical_h_k_idx,
                                    storage_coord,
                                ),
                                stage_info.barrier,
                            )
            elif cutlass.const_expr(kv_atom_size == 64):
                # A B128-aligned semantic block keeps every route inside one
                # BSR entry, so one KV128 TMA is always legal; TMA OOB fill
                # handles a partial physical tail. Other coarse blocks may
                # join unrelated entries and must prove physical adjacency.
                fragment_chunk_elems = chunk_hd * 64
                if prims.elect_sync():
                    origin0, _ = self.sparse_kv_metadata.route_tma_coordinate(
                        Int32(0),
                        logical_b_idx,
                    )
                    origin0 = Int32(origin0)
                    atom_valid_mask = Int32(
                        self.sparse_kv_metadata.route_atom_valid_mask()
                    )
                    valid0 = cutlass.Boolean((atom_valid_mask & Int32(1)) != Int32(0))
                    if not valid0:
                        origin0 = Int32(self.max_seq_len_kv)
                    stage_base = self._stage_base(stage_info)
                    if cutlass.const_expr(
                        _prepared_kv_routes_are_block_aligned(
                            cfg.kv_block_size, cfg.tile_size_kv
                        )
                    ):
                        for chunk_idx in cutlass.range_constexpr(num_chunks):
                            local_head_dim_offset = chunk_idx * chunk_hd
                            global_head_dim_offset = (
                                head_dim_stage_offset + local_head_dim_offset
                            )
                            local_tile_offset = chunk_idx * tile_chunk_elems
                            prims.cp_async_bulk_tensor_shared_cta_global(
                                stage_base.subview(local_tile_offset),
                                tma_desc,
                                (
                                    Int32(global_head_dim_offset),
                                    origin0,
                                    logical_h_k_idx,
                                    logical_b_idx,
                                ),
                                stage_info.barrier,
                            )
                    else:
                        origin1, _ = self.sparse_kv_metadata.route_tma_coordinate(
                            Int32(1),
                            logical_b_idx,
                        )
                        origin1 = Int32(origin1)
                        valid1 = cutlass.Boolean(
                            (atom_valid_mask & Int32(2)) != Int32(0)
                        )
                        adjacent = (valid0 & valid1) & cutlass.Boolean(
                            origin1 == origin0 + Int32(64)
                        )
                        if not valid1:
                            origin1 = Int32(self.max_seq_len_kv)
                        for chunk_idx in cutlass.range_constexpr(num_chunks):
                            local_head_dim_offset = chunk_idx * chunk_hd
                            global_head_dim_offset = (
                                head_dim_stage_offset + local_head_dim_offset
                            )
                            local_tile_offset = chunk_idx * tile_chunk_elems
                            if adjacent:
                                prims.cp_async_bulk_tensor_shared_cta_global(
                                    stage_base.subview(local_tile_offset),
                                    tma_desc,
                                    (
                                        Int32(global_head_dim_offset),
                                        origin0,
                                        logical_h_k_idx,
                                        logical_b_idx,
                                    ),
                                    stage_info.barrier,
                                )
                            else:
                                prims.cp_async_bulk_tensor_shared_cta_global(
                                    stage_base.subview(local_tile_offset),
                                    tma_desc_atom,
                                    (
                                        Int32(global_head_dim_offset),
                                        origin0,
                                        logical_h_k_idx,
                                        logical_b_idx,
                                    ),
                                    stage_info.barrier,
                                )
                                prims.cp_async_bulk_tensor_shared_cta_global(
                                    stage_base.subview(
                                        local_tile_offset + fragment_chunk_elems
                                    ),
                                    tma_desc_atom,
                                    (
                                        Int32(global_head_dim_offset),
                                        origin1,
                                        logical_h_k_idx,
                                        logical_b_idx,
                                    ),
                                    stage_info.barrier,
                                )
            else:
                # Fine routes stay fully general: issue one TMA per route
                # atom. Retained metadata has already mapped empty slots to
                # an OOB origin, avoiding a repeated predicate here. Keeping
                # the load policy independent of adjacency is faster for
                # irregular top-k rows.
                atom_chunk_elems = chunk_hd * kv_atom_size
                atoms_per_route = cfg.tile_size_kv // kv_atom_size
                if prims.elect_sync():
                    stage_base = self._stage_base(stage_info)
                    # Reuse each retained origin across all head-dimension
                    # chunks. The copies still target disjoint SMEM regions
                    # and share one completion barrier, so only issue order
                    # changes.
                    origin, _ = self.sparse_kv_metadata.route_tma_coordinate(
                        Int32(0),
                        logical_b_idx,
                    )
                    next_origin, _ = self.sparse_kv_metadata.route_tma_coordinate(
                        Int32(1),
                        logical_b_idx,
                    )
                    origin = Int32(origin)
                    next_origin = Int32(next_origin)
                    for atom_idx in cutlass.range_constexpr(atoms_per_route):
                        # Keep two origins ahead of TMA issue so the scalar
                        # LDS -> uniform-register handoff can overlap a full
                        # atom's asynchronous copies.
                        future_origin = next_origin
                        if cutlass.const_expr(atom_idx + 2 < atoms_per_route):
                            future_origin, _ = (
                                self.sparse_kv_metadata.route_tma_coordinate(
                                    Int32(atom_idx + 2),
                                    logical_b_idx,
                                )
                            )
                            future_origin = Int32(future_origin)
                        for chunk_idx in cutlass.range_constexpr(num_chunks):
                            local_head_dim_offset = chunk_idx * chunk_hd
                            global_head_dim_offset = (
                                head_dim_stage_offset + local_head_dim_offset
                            )
                            local_tile_offset = chunk_idx * tile_chunk_elems
                            prims.cp_async_bulk_tensor_shared_cta_global(
                                stage_base.subview(
                                    local_tile_offset + atom_idx * atom_chunk_elems
                                ),
                                tma_desc_atom,
                                (
                                    Int32(global_head_dim_offset),
                                    origin,
                                    logical_h_k_idx,
                                    logical_b_idx,
                                ),
                                stage_info.barrier,
                            )
                        origin = next_origin
                        next_origin = future_origin

        elif cutlass.const_expr(cfg.use_paged_kv):
            # Paged-KV path: the page-offset resource has already staged the
            # page IDs for this tile window into SMEM. This producer slices the
            # page IDs for one tile and emits one TMA per page fragment.
            head_dim_stage = cfg.head_dim_kv_stage
            head_dim_stage_offset = head_dim_stage_idx * head_dim_stage
            page_fragments = cfg.tile_size_kv // cfg.num_tokens_per_page
            tile_idx = self._maybe_runtime_tile_idx(stage_info, local_tile_idx)
            if cutlass.const_expr(cfg.use_fp8_qkv):
                if prims.elect_sync():
                    # FP8 pages are copied as one contiguous head-dim stage per
                    # page fragment.
                    stage_base = self._stage_base(stage_info)
                    page_ids = self.page_offsets_kv.page_ids(tile_idx)
                    for page_frag in cutlass.range_constexpr(page_fragments):
                        page_id = Int32(page_ids[page_frag])
                        smem_page_offset = Int32(
                            page_frag * cfg.num_tokens_per_page * head_dim_stage
                        )
                        prims.cp_async_bulk_tensor_shared_cta_global(
                            stage_base.subview(smem_page_offset),
                            tma_desc,
                            (
                                Int32(head_dim_stage_offset),
                                Int32(0),
                                logical_h_k_idx,
                                page_id,
                            ),
                            stage_info.barrier,
                        )
            else:
                # 16-bit pages are copied in 64-column chunks so the SMEM
                # layout matches the K/V tcgen05 descriptor swizzle.
                chunk_hd = min(head_dim_stage, 64)
                num_chunks = head_dim_stage // chunk_hd
                tile_chunk_elems = chunk_hd * cfg.tile_size_kv
                page_chunk_elems = chunk_hd * cfg.num_tokens_per_page
                if prims.elect_sync():
                    stage_base = self._stage_base(stage_info)
                    page_ids = self.page_offsets_kv.page_ids(tile_idx)
                    for chunk_idx in cutlass.range_constexpr(num_chunks):
                        local_head_dim_offset = chunk_idx * chunk_hd
                        global_head_dim_offset = (
                            head_dim_stage_offset + local_head_dim_offset
                        )
                        local_tile_offset = chunk_idx * tile_chunk_elems
                        for page_frag in cutlass.range_constexpr(page_fragments):
                            page_id = Int32(page_ids[page_frag])
                            smem_page_offset = Int32(
                                local_tile_offset + page_frag * page_chunk_elems
                            )
                            prims.cp_async_bulk_tensor_shared_cta_global(
                                stage_base.subview(smem_page_offset),
                                tma_desc,
                                (
                                    Int32(global_head_dim_offset),
                                    Int32(0),
                                    logical_h_k_idx,
                                    page_id,
                                ),
                                stage_info.barrier,
                            )
        else:
            # Dense-KV path: map the runtime-resolved tile directly to the
            # tensor's sequence dimension and copy one contiguous tile.
            tile_idx = self._maybe_runtime_tile_idx(stage_info, local_tile_idx)
            tile_offset = tile_idx * Int32(cfg.tile_size_kv)
            head_dim_stage = cfg.head_dim_kv_stage
            head_dim_stage_offset = head_dim_stage_idx * head_dim_stage
            if cutlass.const_expr(cfg.use_fp8_qkv):
                if prims.elect_sync():
                    # FP8 dense K/V needs one tensor copy for the active
                    # head-dim stage.
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        self._stage_base(stage_info),
                        tma_desc,
                        (
                            Int32(head_dim_stage_offset),
                            tile_offset,
                            logical_h_k_idx,
                            logical_b_idx,
                        ),
                        stage_info.barrier,
                    )
            else:
                # 16-bit dense K/V uses 64-column chunks for the staged
                # head-dim slice.
                chunk_hd = min(head_dim_stage, 64)
                num_chunks = head_dim_stage // chunk_hd
                tile_chunk_elems = chunk_hd * cfg.tile_size_kv
                if prims.elect_sync():
                    stage_base = self._stage_base(stage_info)
                    for chunk_idx in cutlass.range_constexpr(num_chunks):
                        local_head_dim_offset = chunk_idx * chunk_hd
                        global_head_dim_offset = (
                            head_dim_stage_offset + local_head_dim_offset
                        )
                        smem_offset = chunk_idx * tile_chunk_elems
                        prims.cp_async_bulk_tensor_shared_cta_global(
                            stage_base.subview(smem_offset),
                            tma_desc,
                            (
                                Int32(global_head_dim_offset),
                                tile_offset,
                                logical_h_k_idx,
                                logical_b_idx,
                            ),
                            stage_info.barrier,
                        )

    @producer_work
    @cute.jit
    def load_k0(
        self,
        stage_info: StageInfo,
        *,
        section: Constexpr[FmhaStage],
        head_dim_stage_idx: Constexpr[int],
    ) -> None:
        """Produce the first split K tile for this schedule phase."""
        # ProdWork: K0 uses inst slot 0; the section selects HEAD/LOOP/TAIL
        # tile numbering and head_dim_stage_idx selects the H256 slice.
        self._producer_load(stage_info, section, head_dim_stage_idx)

    @producer_work
    @cute.jit
    def load_k1(
        self,
        stage_info: StageInfo,
        *,
        section: Constexpr[FmhaStage],
        head_dim_stage_idx: Constexpr[int],
    ) -> None:
        """Produce the second split K tile for this schedule phase."""
        # ProdWork: K1 uses inst slot 1 but otherwise shares the same staged
        # K/V TMA path as K0.
        self._producer_load(stage_info, section, head_dim_stage_idx)

    @producer_work
    @cute.jit
    def load_v0(
        self,
        stage_info: StageInfo,
        *,
        section: Constexpr[FmhaStage],
        head_dim_stage_idx: Constexpr[int],
    ) -> None:
        """Produce the first split V tile for this schedule phase."""
        # ProdWork: V0 publishes the first V descriptor stream consumed by the
        # corresponding PV MMA call.
        self._producer_load(stage_info, section, head_dim_stage_idx)

    @producer_work
    @cute.jit
    def load_v1(
        self,
        stage_info: StageInfo,
        *,
        section: Constexpr[FmhaStage],
        head_dim_stage_idx: Constexpr[int],
    ) -> None:
        """Produce the second split V tile for this schedule phase."""
        # ProdWork: V1 publishes the second V descriptor stream consumed by the
        # corresponding PV MMA call.
        self._producer_load(stage_info, section, head_dim_stage_idx)

    @cute.jit
    def _build_desc(self, stage_info: StageInfo) -> prims.Tcgen05SmemDesc:
        """Advance the split K/V base descriptor to the committed stage."""
        stage_offset_bytes = stage_info.stage_idx * Int32(self.cfg.smem_kv_tile_bytes)
        return self._desc_base.advance_start_address(stage_offset_bytes)

    @consumer_work(returns=kv_desc_slot)
    @cute.jit
    def kv_desc(self, stage_info: StageInfo) -> prims.Tcgen05SmemDesc:
        """Publish the K descriptor consumed by QK MMA."""
        # ConsWork: advance the K descriptor to the stage committed by the
        # producer and route it through the kv_desc task-local slot.
        return self._build_desc(stage_info)

    @consumer_work(returns=v_desc_slot)
    @cute.jit
    def v_desc(self, stage_info: StageInfo) -> prims.Tcgen05SmemDesc:
        """Publish the V descriptor consumed by PV MMA."""
        # ConsWork: advance the V descriptor to the stage committed by the
        # producer and route it through the v_desc task-local slot.
        return self._build_desc(stage_info)


@dataclass(kw_only=True)
class SmemPageOffsetsKvResource(DecodeGenResourceBase):
    """Paged-KV logical-to-physical page IDs staged in SMEM.

    A dedicated warp prefetches the page table entries for the next K/V tile.
    The TMA load warp then reads these SMEM-cached offsets when issuing the
    page-sized TMA copies, matching the split producer layout.

    Paired K0/K1/V0/V1 schedules publish one stage per logical tile and store
    exactly that tile's page IDs. Shared-offset schedules retain a warp-aligned
    32-ID window so one coalesced load can serve adjacent logical tiles.
    """

    cfg: Constexpr[FmhaDecodeConfig] = None
    stage_page_ids_per_tile: Constexpr[bool] = False
    page_idx_kv: cute.Pointer | None = None
    seqlens_kv: cute.Pointer | None = None
    use_native_paged_kv: Constexpr[bool] = False
    block_tables: cute.Pointer | None = None
    block_table_row_stride: cutlass.Int64 = None
    max_seq_len_kv: Int32 = None
    h_k_idx: Int32 = None
    b_idx: Int32 = None
    q_group_idx: Int32 = None
    seq_len_q: Int32 = None
    _alloc: Constexpr[SmemAllocation | None] = None
    _smem_page_offsets: cutlass.Array = None
    cached_page_ids: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()

    def __post_init__(self) -> None:
        """Create a shape-stable register slot before task dispatch branches."""
        object.__setattr__(
            self,
            "cached_page_ids",
            TaskLocalVariable(
                dtype=cutlass.Array,
                default_factory=lambda: cutlass.Array(
                    Int32,
                    self.cfg.tile_size_kv // self.cfg.num_tokens_per_page,
                    space=cutlass.AddressSpace.rmem,
                ),
                docs="Page IDs reused by every head-dimension stage of one K/V tile.",
            ),
        )
        self._init_placeholder_state()

    def _init_placeholder_state(self) -> None:
        """Create placeholder storage for per-stage page-offset windows."""
        num_stages = (
            self.pipeline_config.num_stages if self.pipeline_config is not None else 1
        )
        pages_per_tile = self.cfg.tile_size_kv // self.cfg.num_tokens_per_page
        page_ids_per_stage = pages_per_tile if self.stage_page_ids_per_tile else 32
        self._smem_page_offsets = _placeholder_smem_array(
            Int32, num_stages * page_ids_per_stage
        )

    def get_smem_requirements(self) -> list[SmemAllocation]:
        """Allocate one tile or one held window per page-offset stage."""
        num_stages = (
            self.pipeline_config.num_stages if self.pipeline_config is not None else 1
        )
        pages_per_tile = self.cfg.tile_size_kv // self.cfg.num_tokens_per_page
        page_ids_per_stage = pages_per_tile if self.stage_page_ids_per_tile else 32
        if self._alloc is None:
            self._alloc = SmemAllocation(
                name=f"{self.name}",
                size_bytes=num_stages * page_ids_per_stage * 4,
                alignment=16,
            )
        return [self._alloc]

    def get_tmem_requirements(self) -> list[TmemAllocation]:
        """Page-offset staging uses SMEM only."""
        return []

    @cute.jit
    def _create_initial_task_locals(
        self, context: ResourceContext | None = None
    ) -> ResourceVars:
        """Bind the page-offset SMEM cache for producer and consumer tasks."""
        if cutlass.const_expr(context is not None and context.smem_base is not None):
            num_stages = (
                self.pipeline_config.num_stages
                if self.pipeline_config is not None
                else 1
            )
            pages_per_tile = self.cfg.tile_size_kv // self.cfg.num_tokens_per_page
            page_ids_per_stage = pages_per_tile if self.stage_page_ids_per_tile else 32
            self._smem_page_offsets = cutlass.Array(
                context.smem_base.data_ptr() + self._alloc.offset,
                dtype=cutlass.Int32,
                shape=(num_stages * page_ids_per_stage,),
                addrspace=3,
            )
        return {}

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_load_state(self, stage_info: StageInfo) -> None:
        """Initialize producer-side page-offset cache state."""
        # ProdAuxWork: bind the page-offset SMEM cache before the prefetch warp
        # starts publishing page-table windows.
        self._create_initial_task_locals(stage_info.context)

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_read_state(self, stage_info: StageInfo) -> None:
        """Initialize consumer-side page-offset cache state."""
        # ConsAuxWork: bind the same cache on the load-warp side so K/V TMA
        # work can slice out the page IDs for each tile.
        self._create_initial_task_locals(stage_info.context)

    @consumer_work(work_attrs=WorkAttr.AUXILIARY, returns=cached_page_ids)
    @cute.jit
    def init_cached_read_state(self, stage_info: StageInfo) -> cutlass.Array:
        """Initialize the per-tile page-ID register cache."""
        self._create_initial_task_locals(stage_info.context)
        return cutlass.Array(
            Int32,
            self.cfg.tile_size_kv // self.cfg.num_tokens_per_page,
            space=cutlass.AddressSpace.rmem,
        )

    @cute.jit
    def page_ids(self, tile_idx: Int32) -> cutlass.Array:
        """Load the tile's page IDs from its staged cache entry.

        Single-tile stages begin at offset zero. Multi-tile stages use the
        runtime-resolved tile index to select from their aligned 32-ID window.
        """
        cfg = self.cfg
        pages_per_tile = cfg.tile_size_kv // cfg.num_tokens_per_page
        if cutlass.const_expr(self.stage_page_ids_per_tile):
            offset = self.consumer_work_stage * Int32(pages_per_tile)
        else:
            group_page_idx = (tile_idx * Int32(pages_per_tile)) & Int32(31)
            offset = self.consumer_work_stage * Int32(32) + group_page_idx
        if cutlass.const_expr(pages_per_tile in (8, 16)):
            # Native shared-memory vector loads top out at four Int32 values.
            # Wide page-16 tiles therefore consume their IDs as independently
            # aligned 16-byte loads from the same 32-ID cache window.
            page_ids = cutlass.Array(
                Int32, pages_per_tile, space=cutlass.AddressSpace.rmem
            )
            for vector_idx in cutlass.range_constexpr(pages_per_tile // 4):
                vector = self._smem_page_offsets.load(
                    offset + Int32(vector_idx * 4),
                    vector_size=4,
                    alignment=16,
                )
                for elem_idx in cutlass.range_constexpr(4):
                    page_ids[vector_idx * 4 + elem_idx] = vector[elem_idx]
            return page_ids
        if cutlass.const_expr(pages_per_tile == 4):
            return self._smem_page_offsets.load(offset, vector_size=4, alignment=16)
        if cutlass.const_expr(pages_per_tile == 2):
            return self._smem_page_offsets.load(offset, vector_size=2, alignment=8)
        return self._smem_page_offsets.load(offset, vector_size=1, alignment=4)

    @consumer_work(returns=cached_page_ids)
    @cute.jit
    def cache_page_ids(
        self,
        stage_info: StageInfo,
        *,
        cached_page_ids: cutlass.Array,
        inst_id: Constexpr[int],
        kv_kind: Constexpr[int],
        section: Constexpr[FmhaStage],
    ) -> cutlass.Array:
        """Load one tile's page IDs once for all of its head-dim stages."""
        cfg = self.cfg
        local_tile_idx = _local_kv_tile_idx_for_section(
            cfg, stage_info, inst_id, kv_kind, section
        )
        tile_idx = (
            Int32(_decode_gen_task_cache(stage_info)[_TASK_CACHE_KV_RAW_TILE_BASE])
            + local_tile_idx
        )
        pages_per_tile = cfg.tile_size_kv // cfg.num_tokens_per_page

        # BF16 TMA is issued only by the elected lane, so only that lane needs
        # the register cache. FP8's predicated helper builds coordinates in
        # every lane and therefore keeps the existing all-lane semantics.
        if cutlass.const_expr(cfg.use_fp8_qkv):
            fp8_page_ids = self.page_ids(tile_idx)
            for page_frag in cutlass.range_constexpr(pages_per_tile):
                cached_page_ids[page_frag] = Int32(fp8_page_ids[page_frag])
        elif prims.elect_sync():
            bf16_page_ids = self.page_ids(tile_idx)
            for page_frag in cutlass.range_constexpr(pages_per_tile):
                cached_page_ids[page_frag] = Int32(bf16_page_ids[page_frag])
        return cached_page_ids

    @cute.jit
    def _producer_load_page_offsets(
        self,
        stage_info: StageInfo,
        inst_id: int,
        kv_kind: int,
        section: Constexpr[FmhaStage],
    ) -> None:
        """Prefetch one K/V tile or aligned multi-tile window."""
        cfg = self.cfg
        local_tile_idx = _local_kv_tile_idx_for_section(
            cfg, stage_info, inst_id, kv_kind, section
        )

        # Resolve the logical tile after split-KV and sliding-window
        # transforms so the staged page IDs match the K/V TMA descriptor.
        tile_idx = (
            Int32(_decode_gen_task_cache(stage_info)[_TASK_CACHE_KV_RAW_TILE_BASE])
            + local_tile_idx
        )
        _, logical_b_idx = _logical_head_batch(stage_info, self.h_k_idx, self.b_idx)
        pages_per_tile = Int32(cfg.tile_size_kv // cfg.num_tokens_per_page)
        if cutlass.const_expr(self.use_native_paged_kv):
            task_cache = _decode_gen_task_cache(stage_info)
            page_idx_ub = Int32(task_cache[_TASK_CACHE_KV_PAGE_IDX_UB])
            page_table_offset = (
                cutlass.Int64(logical_b_idx) * self.block_table_row_stride
            )
            page_idx_kv = self.block_tables
        else:
            if cutlass.const_expr(self.seqlens_kv is None):
                page_idx_ub = Int32(cfg.max_num_pages_per_seq_kv - 1)
            else:
                # Clamp page prefetches to the last valid page; softmax masking
                # removes invalid tokens in the final partial tile.
                seq_len_kv = _load_runtime_seq_len_kv(
                    self.seqlens_kv,
                    self.max_seq_len_kv,
                    stage_info,
                    self.h_k_idx,
                    self.b_idx,
                )
                page_idx_ub = _runtime_last_valid_page_idx(cfg, seq_len_kv)

            page_table_offset = logical_b_idx * Int32(2 * cfg.max_num_pages_per_seq_kv)
            if cutlass.const_expr(kv_kind == KV_KIND_V):
                # K and V page tables are stored as two consecutive per-batch ranges.
                page_table_offset += Int32(cfg.max_num_pages_per_seq_kv)
            page_idx_kv = self.page_idx_kv
        smem_page_offsets = self._smem_page_offsets
        lane_idx = cute.arch.thread_idx()[0] & Int32(0x1F)
        if cutlass.const_expr(self.stage_page_ids_per_tile):
            if lane_idx < pages_per_tile:
                logical_page_idx = cute.math.min(
                    tile_idx * pages_per_tile + lane_idx, page_idx_ub
                )
                smem_offset = stage_info.stage_idx * pages_per_tile + lane_idx
                smem_page_offsets[smem_offset] = Int32(
                    page_idx_kv[page_table_offset + logical_page_idx]
                )
        else:
            # Shared-offset schedules use one coalesced warp load for an
            # aligned 32-ID window; consumers select their tile within it.
            grouped_base_page_idx = ((tile_idx * pages_per_tile) >> Int32(5)) << Int32(
                5
            )
            grouped_logical_page_idx = cute.math.min(
                grouped_base_page_idx + lane_idx, page_idx_ub
            )
            grouped_smem_offset = stage_info.stage_idx * Int32(32) + lane_idx
            smem_page_offsets[grouped_smem_offset] = Int32(
                page_idx_kv[page_table_offset + grouped_logical_page_idx]
            )

    @producer_work
    @cute.jit
    def load_k0(self, stage_info: StageInfo, *, section: Constexpr[FmhaStage]) -> None:
        """Produce the first K page-offset stage."""
        # ProdWork: prefetch the page IDs that cover K0's tile.
        self._producer_load_page_offsets(stage_info, KV_INST0, KV_KIND_K, section)

    @producer_work
    @cute.jit
    def load_k1(self, stage_info: StageInfo, *, section: Constexpr[FmhaStage]) -> None:
        """Produce the second K page-offset stage."""
        # ProdWork: prefetch the page IDs that cover K1's tile.
        self._producer_load_page_offsets(stage_info, KV_INST1, KV_KIND_K, section)

    @producer_work
    @cute.jit
    def load_v0(self, stage_info: StageInfo, *, section: Constexpr[FmhaStage]) -> None:
        """Produce the first V page-offset stage."""
        # ProdWork: prefetch the page IDs that cover V0's tile.
        self._producer_load_page_offsets(stage_info, KV_INST0, KV_KIND_V, section)

    @producer_work
    @cute.jit
    def load_v1(self, stage_info: StageInfo, *, section: Constexpr[FmhaStage]) -> None:
        """Produce the second V page-offset stage."""
        # ProdWork: prefetch the page IDs that cover V1's tile.
        self._producer_load_page_offsets(stage_info, KV_INST1, KV_KIND_V, section)

    @consumer_work
    @cute.jit
    def read_offsets(self, stage_info: StageInfo) -> None:
        """Consume a generic page-offset window for shared-offset paths."""
        # ConsWork: the page IDs are read directly from SMEM by the K/V load
        # resource; this method records the schedule edge only.
        _ = stage_info
        return

    @consumer_work
    @cute.jit
    def read_offsets_k0(self, stage_info: StageInfo) -> None:
        """Consume the first K page-offset window."""
        # ConsWork: route the K0 offset token to the matching K0 TMA load.
        _ = stage_info
        return

    @consumer_work
    @cute.jit
    def read_offsets_k1(self, stage_info: StageInfo) -> None:
        """Consume the second K page-offset window."""
        # ConsWork: route the K1 offset token to the matching K1 TMA load.
        _ = stage_info
        return

    @consumer_work
    @cute.jit
    def read_offsets_v0(self, stage_info: StageInfo) -> None:
        """Consume the first V page-offset window."""
        # ConsWork: route the V0 offset token to the matching V0 TMA load.
        _ = stage_info
        return

    @consumer_work
    @cute.jit
    def read_offsets_v1(self, stage_info: StageInfo) -> None:
        """Consume the second V page-offset window."""
        # ConsWork: route the V1 offset token to the matching V1 TMA load.
        _ = stage_info
        return


@dataclass(kw_only=True)
class SmemKvResource(DecodeGenResourceBase):
    """Shared KV staging resource for the decode kernel.

    K and V loads share one SMEM allocation and one async pipeline/state.
    Loads alternate K and V into one ring buffer; the consumer descriptors
    target the same allocation.
    """

    _task_local_specs: ClassVar[tuple[tuple, ...]] = (
        (
            "kv_desc_slot",
            prims.Tcgen05SmemDesc,
            prims.Tcgen05SmemDesc(0),
            "SMEM descriptor for K loads consumed by QK MMA.",
        ),
        (
            "v_desc_0_slot",
            prims.Tcgen05SmemDesc,
            prims.Tcgen05SmemDesc(0),
            "First V descriptor consumed by VP MMA.",
        ),
        (
            "v_desc_1_slot",
            prims.Tcgen05SmemDesc,
            prims.Tcgen05SmemDesc(0),
            "Second V descriptor consumed by VP MMA.",
        ),
    )
    cfg: Constexpr[FmhaDecodeConfig] = None
    tma_desc_k: cutlass.Pointer | None = None
    tma_desc_v: cutlass.Pointer | None = None
    tma_desc_k_atom: cutlass.Pointer | None = None
    tma_desc_v_atom: cutlass.Pointer | None = None
    sparse_kv_metadata0: "SmemBlockSparseKvMetadataResource | None" = None
    sparse_kv_metadata1: "SmemBlockSparseKvMetadataResource | None" = None
    page_offsets_kv: SmemPageOffsetsKvResource | None = None
    seqlens_kv: cute.Pointer | None = None
    max_seq_len_kv: Int32 = None
    h_k_idx: Int32 = None
    b_idx: Int32 = None
    q_group_idx: Int32 = None
    seq_len_q: Int32 = None
    _alloc: Constexpr[SmemAllocation | None] = None
    _smem_base_kv: cutlass.Array = None
    _k_desc_base: prims.Tcgen05SmemDesc = None
    _v_desc_base: prims.Tcgen05SmemDesc = None
    kv_desc_slot: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    v_desc_0_slot: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    v_desc_1_slot: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()

    def _init_placeholder_state(self) -> None:
        """Create placeholder state for the shared K/V SMEM ring."""
        num_stages = (
            self.pipeline_config.num_stages
            if self.pipeline_config is not None
            else self.cfg.kv_stages
        )
        self._smem_base_kv = _placeholder_smem_array(
            self.cfg.kv_dtype,
            self.cfg.smem_kv_tile_elements * num_stages,
        )
        self._k_desc_base = prims.Tcgen05SmemDesc(0)
        self._v_desc_base = prims.Tcgen05SmemDesc(0)

    def get_smem_requirements(self) -> list[SmemAllocation]:
        """Allocate the shared K/V staged SMEM ring."""
        num_stages = (
            self.pipeline_config.num_stages
            if self.pipeline_config is not None
            else self.cfg.kv_stages
        )
        if self._alloc is None:
            self._alloc = SmemAllocation(
                name=f"{self.name}",
                size_bytes=self.cfg.smem_kv_tile_bytes * num_stages,
                alignment=self.cfg.stensor_align,
            )
        return [self._alloc]

    def get_tmem_requirements(self) -> list[TmemAllocation]:
        """Shared K/V staging uses SMEM only."""
        return []

    @cute.jit
    def _create_initial_task_locals(
        self, context: ResourceContext | None = None
    ) -> ResourceVars:
        """Bind the shared K/V ring and build K/V base descriptors."""
        if cutlass.const_expr(context is not None and context.smem_base is not None):
            # Bind the shared K/V SMEM ring. K and V descriptors use the same
            # allocation but may differ in leading-byte offset.
            num_stages = (
                self.pipeline_config.num_stages
                if self.pipeline_config is not None
                else self.cfg.kv_stages
            )
            self._smem_base_kv = cutlass.Array(
                context.smem_base.data_ptr() + self._alloc.offset,
                dtype=self.cfg.kv_dtype,
                shape=(self.cfg.smem_kv_tile_elements * num_stages,),
                addrspace=3,
            )
            kv_tile_bytes = Int32(
                self.cfg.tile_size_kv
                * self.cfg.head_dim_kv_stage
                * self.cfg.kv_dtype_bytes
            )
            # 16-bit K/V stages are made of 64-column chunks. FP8 uses a
            # stride derived from the staged head dimension.
            k_leading_byte_offset = Int32(
                self.cfg.tile_size_kv
                * min(self.cfg.head_dim_kv_stage, 64)
                * self.cfg.kv_dtype_bytes
            )
            stride_byte_offset = Int32(1024)
            if cutlass.const_expr(self.cfg.use_fp8_qkv):
                k_leading_byte_offset = kv_tile_bytes
                stride_byte_offset = Int32(
                    _major_k_stride_bytes(
                        self.cfg.kv_dtype_bytes, self.cfg.head_dim_kv_stage
                    )
                )
            v_leading_byte_offset = k_leading_byte_offset
            if cutlass.const_expr(self.cfg.tile_size_kv == 256):
                # K spans the complete KV256 row between D64 halves. V is
                # staged as four semantic KV64 blocks, each with D/64 adjacent
                # D64 halves, so its MMA-K leading step is one KV64 block.
                v_leading_byte_offset = Int32(64 * 64 * self.cfg.kv_dtype_bytes)
            if cutlass.const_expr(self.cfg.use_fp8_qkv or self.cfg.headdim == 64):
                v_leading_byte_offset = Int32(0)
            # Descriptor bases are advanced per stage at consumption time; the
            # swizzle parameters are invariant for the resource.
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
        # the load task alternates K and V stages through it.
        self._create_initial_task_locals(stage_info.context)

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_descriptor_state(self, stage_info: StageInfo) -> None:
        """Initialize consumer-side shared K/V descriptor state."""
        # ConsAuxWork: initialize descriptor slots for both K and V consumers
        # of the shared K/V ring.
        self._create_initial_task_locals(stage_info.context)

    @cute.jit
    def _stage_base(self, stage_info: StageInfo) -> cutlass.Array:
        """Return the SMEM base for the current shared K/V pipeline stage."""
        # Return the base pointer for the producer stage selected by TS.
        stage_elems = self.cfg.smem_kv_tile_bytes // self.cfg.kv_dtype_bytes
        return self._smem_base_kv.subview(stage_info.stage_idx * stage_elems)

    @cute.jit
    def _logical_coords(
        self, stage_info: StageInfo, tile_idx: Int32
    ) -> tuple[Int32, Int32, Int32]:
        """Return logical head, batch, and token offset for a KV tile."""
        # Translate logical head/batch and tile index into tensor coordinates.
        logical_h_k_idx, logical_b_idx = _logical_head_batch(
            stage_info, self.h_k_idx, self.b_idx
        )
        tile_offset = tile_idx * Int32(self.cfg.tile_size_kv)
        return logical_h_k_idx, logical_b_idx, tile_offset

    @cute.jit
    def _maybe_runtime_tile_idx(self, stage_info: StageInfo, tile_idx: Int32) -> Int32:
        """Apply runtime sequence and split-KV transforms to a shared K/V tile."""
        if cutlass.const_expr(self.cfg.use_paged_kv):
            return (
                Int32(_decode_gen_task_cache(stage_info)[_TASK_CACHE_KV_RAW_TILE_BASE])
                + tile_idx
            )
        # Apply runtime sequence-length, split-KV, and sliding-window
        # transforms to a local prefetch tile.
        if cutlass.const_expr(
            self.seqlens_kv is None and not self.cfg.uses_runtime_q_kv_union
        ):
            if cutlass.const_expr(self.cfg.use_split_kv):
                tile_idx = _static_split_kv_global_tile_idx(
                    self.cfg, stage_info, tile_idx
                )
            # Clamp out-of-range tile indices so the head/tail cadence stays
            # safe for short KV sequences. The softmax mask suppresses the
            # duplicated rows downstream.
            tile_idx = _clamp_valid_tile_idx(self.cfg, tile_idx)
            return tile_idx + Int32(self.cfg.static_num_skipped_kv_tiles)
        seq_len_kv = _load_runtime_seq_len_kv(
            self.seqlens_kv,
            self.max_seq_len_kv,
            stage_info,
            self.h_k_idx,
            self.b_idx,
        )
        logical_q_group_idx = _logical_q_group_idx(
            self.cfg, stage_info, self.q_group_idx
        )
        q_token_base = _q_group_token_base(self.cfg, logical_q_group_idx)
        tile_idx = _runtime_split_kv_global_tile_idx(
            self.cfg,
            stage_info,
            tile_idx,
            seq_len_kv,
            self.seq_len_q,
            q_token_base,
        )
        tile_idx = _runtime_clamp_valid_tile_idx(
            self.cfg,
            tile_idx,
            seq_len_kv,
            self.seq_len_q,
            q_token_base,
        )
        return tile_idx + _num_skipped_kv_tiles(
            self.cfg,
            seq_len_kv,
            self.seq_len_q,
            q_token_base,
        )

    @cute.jit
    def _local_tile_idx(
        self,
        stage_info: StageInfo,
        inst_id: int,
        kv_kind: int,
        section: Constexpr[FmhaStage],
    ) -> Int32:
        """Return the local K/V tile index for one shared-ring producer call."""
        return _local_kv_tile_idx_for_section(
            self.cfg, stage_info, inst_id, kv_kind, section
        )

    @cute.jit
    def _producer_load_kv(
        self,
        stage_info: StageInfo,
        inst_id: int,
        kv_kind: int,
        section: Constexpr[FmhaStage],
        head_dim_stage_idx: Constexpr[int],
        cached_page_ids: cutlass.Array | None = None,
    ) -> None:
        """Issue one shared-ring K or V TMA load for the schedule phase."""
        cfg = self.cfg
        tma_desc = (
            self.tma_desc_v
            if cutlass.const_expr(kv_kind == KV_KIND_V)
            else self.tma_desc_k
        )
        local_tile_idx = self._local_tile_idx(stage_info, inst_id, kv_kind, section)
        logical_h_k_idx, logical_b_idx = _logical_head_batch(
            stage_info, self.h_k_idx, self.b_idx
        )
        head_dim_stage = cfg.head_dim_kv_stage
        head_dim_stage_offset = head_dim_stage_idx * head_dim_stage

        if cutlass.const_expr(cfg.tile_size_kv == 256):
            sparse_kv_metadata = (
                self.sparse_kv_metadata0
                if cutlass.const_expr(inst_id == KV_INST0)
                else self.sparse_kv_metadata1
            )
            self._producer_load_kv_tile_256(
                stage_info,
                tma_desc,
                local_tile_idx,
                logical_h_k_idx,
                logical_b_idx,
                kv_kind,
                cached_page_ids,
                sparse_kv_metadata,
            )
        elif cutlass.const_expr(cfg.use_paged_kv):
            # Paged-KV path: LoadTask consumes pre-staged page IDs and issues
            # one TMA per page fragment into the current SMEM stage. Grouped
            # cache stages hold 32 page IDs per side; recover the
            # runtime-resolved tile_idx so the right per-tile slice is
            # selected from the shared window.
            page_fragments = cfg.tile_size_kv // cfg.num_tokens_per_page
            if cutlass.const_expr(cfg.use_fp8_qkv):
                # FP8 pages are contiguous across the staged head dimension.
                fp8_stage_base = self._stage_base(stage_info)
                if cutlass.const_expr(cached_page_ids is None):
                    grouped_tile_idx = self._maybe_runtime_tile_idx(
                        stage_info, local_tile_idx
                    )
                    fp8_page_ids = self.page_offsets_kv.page_ids(grouped_tile_idx)
                else:
                    fp8_page_ids = cached_page_ids
                for fp8_page_frag in cutlass.range_constexpr(page_fragments):
                    fp8_page_id = Int32(fp8_page_ids[fp8_page_frag])
                    fp8_smem_page_offset = Int32(
                        fp8_page_frag * cfg.num_tokens_per_page * head_dim_stage
                    )
                    _cp_async_bulk_tensor_4d_shared_cta_global_predicated(
                        fp8_stage_base.subview(fp8_smem_page_offset),
                        tma_desc,
                        (
                            Int32(head_dim_stage_offset),
                            Int32(0),
                            logical_h_k_idx,
                            fp8_page_id,
                        ),
                        stage_info.barrier,
                    )
            else:
                # 16-bit pages are split into 64-column chunks inside the
                # staged head-dim slice.
                chunk_hd = min(head_dim_stage, 64)
                num_chunks = head_dim_stage // chunk_hd
                tile_chunk_elems = chunk_hd * cfg.tile_size_kv
                page_chunk_elems = chunk_hd * cfg.num_tokens_per_page
                if cutlass.const_expr(cached_page_ids is None):
                    # Resolve the tile on every lane before the elected-lane
                    # branch. The release compiler rejects a local that is
                    # materialized only on the elected dynamic path.
                    grouped_tile_idx = self._maybe_runtime_tile_idx(
                        stage_info, local_tile_idx
                    )
                if prims.elect_sync():
                    stage_base = self._stage_base(stage_info)
                    if cutlass.const_expr(cached_page_ids is None):
                        page_ids = self.page_offsets_kv.page_ids(grouped_tile_idx)
                    else:
                        page_ids = cached_page_ids
                    # Consume each cached page ID across every head-dimension
                    # chunk before advancing. The copies are independent, and
                    # this order bounds coordinate live ranges in the unrolled
                    # TMA sequence for every supported page size.
                    for page_frag in cutlass.range_constexpr(page_fragments):
                        page_id = Int32(page_ids[page_frag])
                        for chunk_idx in cutlass.range_constexpr(num_chunks):
                            local_head_dim_offset = chunk_idx * chunk_hd
                            global_head_dim_offset = (
                                head_dim_stage_offset + local_head_dim_offset
                            )
                            local_tile_offset = chunk_idx * tile_chunk_elems
                            smem_page_offset = Int32(
                                local_tile_offset + page_frag * page_chunk_elems
                            )
                            prims.cp_async_bulk_tensor_shared_cta_global(
                                stage_base.subview(smem_page_offset),
                                tma_desc,
                                (
                                    Int32(global_head_dim_offset),
                                    Int32(0),
                                    logical_h_k_idx,
                                    page_id,
                                ),
                                stage_info.barrier,
                            )
        elif cutlass.const_expr(cfg.use_fp8_qkv):
            # Dense FP8 path: one tensor TMA loads the staged K or V tile.
            tile_idx = self._maybe_runtime_tile_idx(stage_info, local_tile_idx)
            tile_offset = tile_idx * Int32(cfg.tile_size_kv)
            if prims.elect_sync():
                stage_base = self._stage_base(stage_info)
                prims.cp_async_bulk_tensor_shared_cta_global(
                    stage_base,
                    tma_desc,
                    (
                        Int32(head_dim_stage_offset),
                        tile_offset,
                        logical_h_k_idx,
                        logical_b_idx,
                    ),
                    stage_info.barrier,
                )
        else:
            # Dense 16-bit path: issue 64-column TMA chunks for this staged
            # head-dim slice.
            chunk_hd = min(head_dim_stage, 64)
            num_chunks = head_dim_stage // chunk_hd
            tile_chunk_elems = chunk_hd * cfg.tile_size_kv
            tile_idx = self._maybe_runtime_tile_idx(stage_info, local_tile_idx)
            tile_offset = tile_idx * Int32(cfg.tile_size_kv)
            if prims.elect_sync():
                stage_base = self._stage_base(stage_info)
                for chunk_idx in cutlass.range_constexpr(num_chunks):
                    local_head_dim_offset = chunk_idx * chunk_hd
                    global_head_dim_offset = (
                        head_dim_stage_offset + local_head_dim_offset
                    )
                    smem_offset = chunk_idx * tile_chunk_elems
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        stage_base.subview(smem_offset),
                        tma_desc,
                        (
                            Int32(global_head_dim_offset),
                            tile_offset,
                            logical_h_k_idx,
                            logical_b_idx,
                        ),
                        stage_info.barrier,
                    )

    @cute.jit
    def _producer_load_kv_tile_256(
        self,
        stage_info: StageInfo,
        tma_desc: cutlass.Pointer,
        local_tile_idx: Int32,
        logical_h_k_idx: Int32,
        logical_b_idx: Int32,
        kv_kind: Constexpr[int],
        cached_page_ids: cutlass.Array | None,
        sparse_kv_metadata: "SmemBlockSparseKvMetadataResource | None",
    ) -> None:
        """Stage one KV256 tile in the physical 2x2-datapath layout.

        The public decode TensorMaps expose KV64 (or one smaller page)
        fragments. K places semantic KV64 blocks in physical order
        ``(0, 2, 1, 3)`` while V keeps semantic block order with adjacent D64
        halves. Dense and paged profiles derive those fragments from one
        contiguous tile; block-sparse profiles consume four prepared KV64
        origins retained by the instruction-local metadata resource.
        """
        cfg = self.cfg
        grouped_tile_idx = Int32(0)
        if cutlass.const_expr(not cfg.use_block_sparse):
            grouped_tile_idx = self._maybe_runtime_tile_idx(stage_info, local_tile_idx)
        stage_base = self._stage_base(stage_info)

        if prims.elect_sync():
            # Only the elected TMA issuer needs page IDs. In particular,
            # page16/KV256 otherwise makes all 32 load-warp lanes repeat four
            # vector loads for the same 16-entry page fragment.
            dense_page_ids = cached_page_ids
            if cutlass.const_expr(
                cfg.use_paged_kv
                and not cfg.use_block_sparse
                and cached_page_ids is None
            ):
                assert self.page_offsets_kv is not None
                dense_page_ids = self.page_offsets_kv.page_ids(grouped_tile_idx)
            for semantic_block in cutlass.range_constexpr(4):
                token_coord = Int32(0)
                storage_coord = logical_b_idx
                if cutlass.const_expr(cfg.use_block_sparse):
                    assert sparse_kv_metadata is not None
                    (
                        token_coord,
                        storage_coord,
                    ) = sparse_kv_metadata.route_tma_coordinate(
                        Int32(semantic_block),
                        logical_b_idx,
                    )
                physical_block = semantic_block
                if cutlass.const_expr(kv_kind == KV_KIND_K):
                    physical_block = KV_TILE_256_K_SLOT_FOR_SEMANTIC_ATOM[
                        semantic_block
                    ]
                for dim_half in cutlass.range_constexpr(2):
                    if cutlass.const_expr(kv_kind == KV_KIND_K):
                        block_base = (
                            dim_half * cfg.tile_size_kv * 64 + physical_block * 64 * 64
                        )
                    else:
                        block_base = (
                            semantic_block * cfg.headdim * 64 + dim_half * 64 * 64
                        )

                    if cutlass.const_expr(cfg.use_block_sparse):
                        sparse_tma_desc = (
                            self.tma_desc_v_atom
                            if cutlass.const_expr(kv_kind == KV_KIND_V)
                            else self.tma_desc_k_atom
                        )
                        assert sparse_tma_desc is not None
                        prims.cp_async_bulk_tensor_shared_cta_global(
                            stage_base.subview(block_base),
                            sparse_tma_desc,
                            (
                                Int32(dim_half * 64),
                                token_coord,
                                logical_h_k_idx,
                                storage_coord,
                            ),
                            stage_info.barrier,
                        )
                    elif cutlass.const_expr(cfg.use_paged_kv):
                        fragment_tokens = min(cfg.num_tokens_per_page, 64)
                        fragments_per_block = 64 // fragment_tokens
                        for fragment in cutlass.range_constexpr(fragments_per_block):
                            token_in_tile = (
                                semantic_block * 64 + fragment * fragment_tokens
                            )
                            logical_page = token_in_tile // cfg.num_tokens_per_page
                            token_in_page = token_in_tile % cfg.num_tokens_per_page
                            page_id = Int32(dense_page_ids[logical_page])
                            smem_offset = block_base + fragment * fragment_tokens * 64
                            prims.cp_async_bulk_tensor_shared_cta_global(
                                stage_base.subview(smem_offset),
                                tma_desc,
                                (
                                    Int32(dim_half * 64),
                                    Int32(token_in_page),
                                    logical_h_k_idx,
                                    page_id,
                                ),
                                stage_info.barrier,
                            )
                    else:
                        tile_offset = grouped_tile_idx * Int32(cfg.tile_size_kv)
                        prims.cp_async_bulk_tensor_shared_cta_global(
                            stage_base.subview(block_base),
                            tma_desc,
                            (
                                Int32(dim_half * 64),
                                tile_offset + Int32(semantic_block * 64),
                                logical_h_k_idx,
                                logical_b_idx,
                            ),
                            stage_info.barrier,
                        )

    @producer_work
    @cute.jit
    def load_k0(
        self,
        stage_info: StageInfo,
        *,
        section: Constexpr[FmhaStage],
        head_dim_stage_idx: Constexpr[int],
    ) -> None:
        """Produce the first shared-ring K tile for this schedule phase."""
        # ProdWork: K0 occupies the first K instruction slot in the shared ring.
        self._producer_load_kv(
            stage_info, KV_INST0, KV_KIND_K, section, head_dim_stage_idx
        )

    @producer_work
    @cute.jit
    def load_k1(
        self,
        stage_info: StageInfo,
        *,
        section: Constexpr[FmhaStage],
        head_dim_stage_idx: Constexpr[int],
    ) -> None:
        """Produce the second shared-ring K tile for this schedule phase."""
        # ProdWork: K1 occupies the second K instruction slot in the shared ring.
        self._producer_load_kv(
            stage_info, KV_INST1, KV_KIND_K, section, head_dim_stage_idx
        )

    @producer_work
    @cute.jit
    def load_v0(
        self,
        stage_info: StageInfo,
        *,
        section: Constexpr[FmhaStage],
        head_dim_stage_idx: Constexpr[int],
    ) -> None:
        """Produce the first shared-ring V tile for this schedule phase."""
        # ProdWork: V0 occupies the first V instruction slot in the shared ring.
        self._producer_load_kv(
            stage_info, KV_INST0, KV_KIND_V, section, head_dim_stage_idx
        )

    @producer_work
    @cute.jit
    def load_v1(
        self,
        stage_info: StageInfo,
        *,
        section: Constexpr[FmhaStage],
        head_dim_stage_idx: Constexpr[int],
    ) -> None:
        """Produce the second shared-ring V tile for this schedule phase."""
        # ProdWork: V1 occupies the second V instruction slot in the shared ring.
        self._producer_load_kv(
            stage_info, KV_INST1, KV_KIND_V, section, head_dim_stage_idx
        )

    @producer_work
    @cute.jit
    def load_k0_cached(
        self,
        stage_info: StageInfo,
        *,
        cached_page_ids: cutlass.Array,
        section: Constexpr[FmhaStage],
        head_dim_stage_idx: Constexpr[int],
    ) -> None:
        """Produce K0 while reusing page IDs across head-dim stages."""
        self._producer_load_kv(
            stage_info,
            KV_INST0,
            KV_KIND_K,
            section,
            head_dim_stage_idx,
            cached_page_ids,
        )

    @producer_work
    @cute.jit
    def load_k1_cached(
        self,
        stage_info: StageInfo,
        *,
        cached_page_ids: cutlass.Array,
        section: Constexpr[FmhaStage],
        head_dim_stage_idx: Constexpr[int],
    ) -> None:
        """Produce K1 while reusing page IDs across head-dim stages."""
        self._producer_load_kv(
            stage_info,
            KV_INST1,
            KV_KIND_K,
            section,
            head_dim_stage_idx,
            cached_page_ids,
        )

    @producer_work
    @cute.jit
    def load_v0_cached(
        self,
        stage_info: StageInfo,
        *,
        cached_page_ids: cutlass.Array,
        section: Constexpr[FmhaStage],
        head_dim_stage_idx: Constexpr[int],
    ) -> None:
        """Produce V0 while reusing page IDs across head-dim stages."""
        self._producer_load_kv(
            stage_info,
            KV_INST0,
            KV_KIND_V,
            section,
            head_dim_stage_idx,
            cached_page_ids,
        )

    @producer_work
    @cute.jit
    def load_v1_cached(
        self,
        stage_info: StageInfo,
        *,
        cached_page_ids: cutlass.Array,
        section: Constexpr[FmhaStage],
        head_dim_stage_idx: Constexpr[int],
    ) -> None:
        """Produce V1 while reusing page IDs across head-dim stages."""
        self._producer_load_kv(
            stage_info,
            KV_INST1,
            KV_KIND_V,
            section,
            head_dim_stage_idx,
            cached_page_ids,
        )

    @cute.jit
    def _build_kv_desc(
        self, stage_info: StageInfo, kv_kind: int
    ) -> prims.Tcgen05SmemDesc:
        """Advance the shared K or V descriptor to the committed stage."""
        # Consumers see the same descriptor layout for every stage; only the
        # base address advances by the committed SMEM stage index.
        stage_offset_bytes = stage_info.stage_idx * Int32(self.cfg.smem_kv_tile_bytes)
        return (
            self._v_desc_base.advance_start_address(stage_offset_bytes)
            if cutlass.const_expr(kv_kind == KV_KIND_V)
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
