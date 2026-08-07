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

"""SMEM staging resources for Q, page offsets, and K/V tiles."""

from dataclasses import dataclass
from typing import Any, ClassVar, Optional

from cutlass.experimental import primitives as prims
from ....tensor_map import transform_ragged_coords

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64
from cutlass.cutlass_dsl import Boolean, dsl_user_op, if_generate
from cutlass.pipeline import PipelineAsync, PipelineState
from cutlass.experimental import primitives as cprims
from cutlass.experimental.task_scheduling.enums import WorkAttr
from cutlass.experimental.task_scheduling.memory import SmemAllocation
from cutlass.experimental.task_scheduling.resources import (
    StageInfo,
    TaskLocalVariable,
    consumer_work,
    producer_work,
)

from ...helpers.constants import (
    CP_ASYNC_CACHE_CA,
    PAGE_OFFSET_BYTES,
    WARP_LANES,
)

from ...helpers.layout import (
    decode_gen_task_cache,
    head_dim_cta_offset_v,
    q_stage_smem_element_offset,
    smem_array,
    tma_inner_dim_elems,
)
from ...helpers.math import (
    qkv_dtype,
    qkv_major_k_stride_bytes_for,
    qkv_smem_swizzle,
    qkv_smem_swizzle_for_head_dim,
)
from ...helpers.ops import (
    lane_idx_from_thread,
)
from ...helpers.query import query_batch_bounds
from ...helpers.stage import MlaStage
from ...helpers.tile import (
    batch_idx_for_stage_cfg,
    cta_idx_head_dim_v_for_stage,
    cta_idx_kv_for_stage,
    cta_idx_q_for_stage,
    global_kv_tile_idx,
    head_idx_for_stage,
    local_kv_tile_idx,
    runtime_seq_len_kv_from_task_cache,
    staged_kv_head_dim_call_idx,
)

from .common import (
    TCGEN05_BF16_K_BLOCK_WIDTH,
    TCGEN05_BF16_SECOND_K_BLOCK_OFFSET_BYTES,
    TCGEN05_BF16_SWIZZLE_STRIDE_BYTES,
    MlaResource,
)

# =====================================================================
# SmemQResource — Q SMEM buffer with TmaUmmaAsync pipeline
# =====================================================================


@dataclass(kw_only=True)
class SmemQResource(MlaResource):
    """Stage Q latent/RoPE tiles in SMEM and publish Q MMA descriptors."""

    _task_local_specs: ClassVar[tuple[tuple, ...]] = (
        (
            "q_desc_var",
            Int64,
            Int64(0),
            "SMEM descriptor for the staged Q latent tile.",
            "q_desc",
        ),
        (
            "q_desc_rope_var",
            Int64,
            Int64(0),
            "SMEM descriptor for the staged Q rope tile.",
            "q_desc_rope",
        ),
    )
    tma_desc_q_latent: object = None
    tma_desc_q_rope: object = None
    head_idx: object = None
    batch_idx: object = None
    cta_idx_q: object = None
    _smem_q: object = None
    _q_desc_base: object = None
    q_desc_var: cutlass.Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    q_desc_rope_var: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )

    def get_smem_requirements(self):
        """Return the SMEM allocation used for staged Q tiles."""
        if self._alloc is None:
            self._alloc = SmemAllocation(
                name=f"{self.name}",
                size_bytes=self.cfg.qk_smem_tile_bytes * self.cfg.q_stages,
                alignment=self.cfg.stensor_align,
            )
        return [self._alloc]

    @cute.jit
    def _init_smem_state(self, stage_info: StageInfo) -> None:
        """Create the Q SMEM view and base descriptor state."""
        context = stage_info.context
        elem_count = (
            self.cfg.qk_smem_tile_bytes * self.cfg.q_stages
        ) // self.cfg.qkv_dtype_bytes
        self._smem_q = smem_array(context, self._alloc, qkv_dtype(self.cfg), elem_count)
        if self._smem_q is not None:
            # The BF16 leading offset advances by at most one 64-wide K block.
            # FP8 uses a dtype-specific major-K stride and therefore rebuilds
            # both descriptor offsets below.
            q_leading_byte_offset = Int32(
                self.cfg.tile_size_q
                * min(self.cfg.head_dim_per_stage_kv, TCGEN05_BF16_K_BLOCK_WIDTH)
                * self.cfg.qkv_dtype_bytes
            )
            q_stride_byte_offset = Int32(TCGEN05_BF16_SWIZZLE_STRIDE_BYTES)
            if cutlass.const_expr(self.cfg.is_fp8_qkv()):
                q_leading_byte_offset = Int32(
                    self.cfg.tile_size_q
                    * self.cfg.head_dim_per_stage_kv
                    * self.cfg.qkv_dtype_bytes
                )
                q_stride_byte_offset = Int32(
                    qkv_major_k_stride_bytes_for(
                        self.cfg, self.cfg.head_dim_per_stage_kv
                    )
                )
            self._q_desc_base = cprims.Tcgen05SmemDesc.build(
                self._smem_q,
                leading_byte_offset=q_leading_byte_offset,
                stride_byte_offset=q_stride_byte_offset,
                layout=qkv_smem_swizzle(self.cfg),
            )

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_load_state(self, stage_info: StageInfo) -> None:
        """Initialize the Q SMEM view for producer-side TMA loads."""

        # Producer aux work runs before the load task starts issuing Q TMA
        # copies.  It creates the SMEM view used by producer_work(load_q).
        self._init_smem_state(stage_info)

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_descriptor_state(self, stage_info: StageInfo) -> None:
        """Initialize the Q SMEM view for consumer-side MMA descriptors."""

        # Consumer aux work creates the same SMEM view in the MMA task so the
        # consumer can build descriptors after it waits on the Q stage.
        self._init_smem_state(stage_info)

    @cute.jit
    def _query_tma_coords(self, dim_offset, local_flat_query_row, batch_idx):
        """Return fixed or ragged TMA coordinates for one Q slice."""

        if cutlass.const_expr(self.cu_seqlens_q is None):
            return (dim_offset, local_flat_query_row, batch_idx)

        query_start, query_length = query_batch_bounds(
            self.cu_seqlens_q,
            batch_idx,
            self.cfg.logical_seq_len_q,
        )
        storage_flat_query_row = (
            query_start * Int32(self.cfg.logical_num_heads_q) + local_flat_query_row
        )
        ragged_extent = (
            query_length * Int32(self.cfg.logical_num_heads_q) - local_flat_query_row
        )
        return transform_ragged_coords(
            (dim_offset, storage_flat_query_row),
            ragged_dim_idx=1,
            ragged_box_size=self.cfg.tile_size_q,
            ragged_extent=ragged_extent,
        )

    @cute.jit
    def _load_q_stage(
        self,
        stage_info: StageInfo,
        stage_base,
        qk_stage_idx: int,
        dim_offset: Int32,
        tma_desc,
    ):
        """Issue one staged TMA load for a Q head-dimension slice."""
        if cutlass.const_expr(tma_desc is None):
            return
        active_width = self.cfg.qk_head_stage_width(qk_stage_idx)
        inner_width = min(tma_inner_dim_elems(self.cfg), active_width)
        head_idx = head_idx_for_stage(self.head_idx, self.cfg, stage_info)
        cta_idx_q = cta_idx_q_for_stage(self.cta_idx_q, stage_info)
        batch_idx = batch_idx_for_stage_cfg(self.batch_idx, self.cfg, stage_info)
        # Ragged storage adds the cumulative batch row offset while preserving
        # the unclamped logical row so a fully padded CTA becomes an OOB
        # zero-fill instead of reloading the final real query row.
        local_flat_query_row = cta_idx_q * Int32(self.cfg.num_heads_q) + head_idx
        query_coords = self._query_tma_coords(
            dim_offset,
            local_flat_query_row,
            batch_idx,
        )
        if prims.elect_sync():
            smem_offset = Int32(q_stage_smem_element_offset(self.cfg, qk_stage_idx))
            prims.cp_async_bulk_tensor_shared_cta_global(
                stage_base.data_ptr(smem_offset),
                tma_desc,
                query_coords,
                stage_info.barrier,
            )
            if cutlass.const_expr(active_width > inner_width):
                second_query_coords = self._query_tma_coords(
                    dim_offset + Int32(inner_width),
                    local_flat_query_row,
                    batch_idx,
                )
                prims.cp_async_bulk_tensor_shared_cta_global(
                    stage_base.data_ptr(
                        smem_offset + Int32(inner_width * self.cfg.tile_size_q)
                    ),
                    tma_desc,
                    second_query_coords,
                    stage_info.barrier,
                )

    @producer_work
    @cute.jit
    def load_q(self, stage_info: StageInfo):
        """Load the grouped-head Q tile as 8 latent chunks plus 1 rope chunk."""
        # This is the actual Q payload producer: the load task fills the
        # acquired Q SMEM stage and the pipeline commit makes it visible to MMA.
        if cutlass.const_expr(
            self.tma_desc_q_latent is None or self.tma_desc_q_rope is None
        ):
            return
        cfg = self.cfg
        stage_elems = cfg.q_smem_tile_elements
        stage_base = self._smem_q.subview(stage_info.stage_idx * Int32(stage_elems))
        latent_stages = cfg.latent_dim // cfg.head_dim_per_stage_kv
        for qk_stage_idx in cutlass.range_constexpr(latent_stages):
            dim_offset = Int32(qk_stage_idx * cfg.head_dim_per_stage_kv)
            self._load_q_stage(
                stage_info,
                stage_base,
                qk_stage_idx,
                dim_offset,
                self.tma_desc_q_latent,
            )
        self._load_q_stage(
            stage_info,
            stage_base,
            latent_stages,
            Int32(0),
            self.tma_desc_q_rope,
        )

    @consumer_work(returns=("q_desc", "q_desc_rope"))
    @cute.jit
    def q_desc(self, stage_info: StageInfo):
        """Publish the Q SMEM descriptor for staged QK MMA."""
        # The MMA task has waited for the Q stage.  It converts the live SMEM
        # stage into descriptors and returns them as task-local values for the
        # downstream TmemSResource producer_work(qk_mma).
        stage_base = self._smem_q.subview(
            stage_info.stage_idx * Int32(self.cfg.q_smem_tile_elements)
        )
        # Build the same descriptor as producer init, but relative to the live
        # pipeline stage that the MMA task has already waited on.
        q_leading_byte_offset = Int32(
            self.cfg.tile_size_q
            * min(self.cfg.head_dim_per_stage_kv, TCGEN05_BF16_K_BLOCK_WIDTH)
            * self.cfg.qkv_dtype_bytes
        )
        q_stride_byte_offset = Int32(TCGEN05_BF16_SWIZZLE_STRIDE_BYTES)
        if cutlass.const_expr(self.cfg.is_fp8_qkv()):
            q_leading_byte_offset = Int32(
                self.cfg.tile_size_q
                * self.cfg.head_dim_per_stage_kv
                * self.cfg.qkv_dtype_bytes
            )
            q_stride_byte_offset = Int32(
                qkv_major_k_stride_bytes_for(self.cfg, self.cfg.head_dim_per_stage_kv)
            )
        desc_q = cprims.Tcgen05SmemDesc.build(
            stage_base,
            leading_byte_offset=q_leading_byte_offset,
            stride_byte_offset=q_stride_byte_offset,
            layout=qkv_smem_swizzle(self.cfg),
        )
        desc_q_rope = desc_q
        if cutlass.const_expr(self.cfg.is_fp8_qkv() and self.cfg.rope_dim == 64):
            rope_stage_idx = self.cfg.latent_dim // self.cfg.head_dim_per_stage_kv
            rope_stage_offset = Int32(
                q_stage_smem_element_offset(self.cfg, rope_stage_idx)
            )
            desc_q_rope = cprims.Tcgen05SmemDesc.build(
                stage_base.subview(rope_stage_offset),
                leading_byte_offset=Int32(
                    self.cfg.tile_size_q * self.cfg.rope_dim * self.cfg.qkv_dtype_bytes
                ),
                stride_byte_offset=Int32(
                    qkv_major_k_stride_bytes_for(self.cfg, self.cfg.rope_dim)
                ),
                layout=qkv_smem_swizzle_for_head_dim(self.cfg, self.cfg.rope_dim),
            )
        return desc_q, desc_q_rope


# =====================================================================
# SmemPageOffsetsResource — Page-offset SMEM buffer with Async pipeline
# =====================================================================


@dataclass(frozen=True)
class _StructuredWaitPipelineAsync(PipelineAsync):
    """Page-offset pipeline with a public structured mbarrier retry loop."""

    @cute.jit
    def _retry_wait(
        self,
        sync_object: object,
        state: PipelineState,
        *,
        loc: Any = None,
        ip: Any = None,
    ) -> None:
        while not sync_object.try_wait(
            state.index,
            state.phase,
            loc=loc,
            ip=ip,
        ):
            pass

    @dsl_user_op
    def producer_acquire(
        self,
        state: PipelineState,
        try_acquire_token: Optional[Boolean] = None,
        *,
        loc: Any = None,
        ip: Any = None,
    ) -> None:
        if_generate(
            try_acquire_token is None or try_acquire_token == 0,
            lambda: self._retry_wait(self.sync_object_empty, state, loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def consumer_wait(
        self,
        state: PipelineState,
        try_wait_token: Optional[Boolean] = None,
        *,
        loc: Any = None,
        ip: Any = None,
    ) -> None:
        if_generate(
            try_wait_token is None or try_wait_token == 0,
            lambda: self._retry_wait(self.sync_object_full, state, loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )


@dataclass(kw_only=True)
class SmemPageOffsetsResource(MlaResource):
    """Prefetch page ids for the K/V tile currently owned by the load task."""

    _task_local_specs: ClassVar[tuple[tuple, ...]] = (
        (
            "cached_page_ids",
            cutlass.Array,
            None,
            "Cached logical page ids for the staged K/V tile.",
        ),
    )
    page_offsets: object = None
    cache_seqs: object = None
    batch_idx: object = None
    cta_idx_q: object = None
    cta_idx_kv: object = None
    _smem_page_offsets: object = None
    cached_page_ids: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )

    def create_pipeline(self, pipeline_config):
        """Preserve stock barrier objects and specialize only their wait path."""
        base = super().create_pipeline(pipeline_config)
        if not isinstance(base, PipelineAsync):
            raise TypeError(
                "page-offset staging requires cutlass.pipeline.PipelineAsync, "
                f"got {type(base).__name__}"
            )
        return _StructuredWaitPipelineAsync(
            base.sync_object_full,
            base.sync_object_empty,
            base.num_stages,
            base.producer_mask,
            base.consumer_mask,
        )

    def get_smem_requirements(self):
        """Return the SMEM allocation for cached page offsets."""
        if self._alloc is None:
            self._alloc = SmemAllocation(
                name=f"{self.name}",
                size_bytes=(
                    self.cfg.page_offsets_stages
                    * self.cfg.page_offsets_entries_per_stage
                    * PAGE_OFFSET_BYTES
                ),
                alignment=128,
            )
        return [self._alloc]

    @cute.jit
    def _init_smem_state(self, stage_info: StageInfo) -> None:
        """Create the SMEM page-offset array view."""
        context = stage_info.context
        count = self.cfg.page_offsets_stages * self.cfg.page_offsets_entries_per_stage
        self._smem_page_offsets = smem_array(
            context,
            self._alloc,
            Int32,
            count,
        )

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_load_state(self, stage_info: StageInfo) -> None:
        """Initialize page-offset SMEM before producer copies begin."""

        # Producer aux work creates the page-offset SMEM view for the page load
        # warp before it issues cp.async copies.
        self._init_smem_state(stage_info)

    @consumer_work(work_attrs=WorkAttr.AUXILIARY, returns=cached_page_ids)
    @cute.jit
    def init_read_state(self, stage_info: StageInfo):
        """Create register caches for the active K/V page-ID streams."""

        # Consumer aux work allocates the register cache that read_offsets()
        # returns after the page-offset stage is ready.
        self._init_smem_state(stage_info)
        cache_slots = 2 if self.cfg.kernel_variant != "keeps_mma_ab" else 1
        return cutlass.Array(
            Int32,
            max(1, cache_slots * self.cfg.pages_per_kv_tile),
            space=cutlass.AddressSpace.rmem,
        )

    @cute.jit
    def page_id(self, page_fragment_idx: int):
        """Return one cached logical page id from the consumer stage."""
        offset = self.consumer_work_stage * Int32(
            self.cfg.page_offsets_entries_per_stage
        ) + Int32(page_fragment_idx)
        return Int32(self._smem_page_offsets[offset])

    @cute.jit
    def _producer_load_page_offsets(
        self,
        stage_info: StageInfo,
        inst_id: int,
        is_v: int,
        *,
        section: cutlass.Constexpr[MlaStage],
    ):
        """Load page ids for one K or V producer stage into SMEM."""
        if cutlass.const_expr(self.page_offsets is None):
            return
        cfg = self.cfg
        lane_idx = lane_idx_from_thread(cute.arch.thread_idx()[0])
        pages_per_tile = Int32(cfg.pages_per_kv_tile)
        if lane_idx < pages_per_tile:
            batch_idx = batch_idx_for_stage_cfg(self.batch_idx, cfg, stage_info)
            cta_idx_q = cta_idx_q_for_stage(self.cta_idx_q, stage_info)
            cta_idx_kv = cta_idx_kv_for_stage(self.cta_idx_kv, stage_info)
            seq_len_kv = runtime_seq_len_kv_from_task_cache(
                cfg,
                decode_gen_task_cache(stage_info),
                cta_idx_q,
                self.cu_seqlens_q,
                batch_idx,
            )
            local_tile_idx = local_kv_tile_idx(
                cfg, stage_info, inst_id, is_v, section=section
            )
            tile_idx = global_kv_tile_idx(cfg, local_tile_idx, seq_len_kv, cta_idx_kv)
            last_valid_page = (
                seq_len_kv + Int32(cfg.num_tokens_per_page - 1)
            ) // Int32(cfg.num_tokens_per_page) - Int32(1)
            logical_page_idx = cute.math.min(
                tile_idx * pages_per_tile + lane_idx,
                last_valid_page,
            )
            smem_offset = (
                stage_info.stage_idx * Int32(cfg.page_offsets_entries_per_stage)
                + lane_idx
            )
            page_offsets_batch = self.page_offsets[None, batch_idx]
            page_offsets_flat = cute.flat_divide(page_offsets_batch, (1,))
            gmem_ptr = page_offsets_flat[None, logical_page_idx].iterator.llvm_ptr
            smem_ptr = cutlass.inttoptr(
                self._smem_page_offsets.data_ptr(smem_offset).toint(cutlass.Int32),
                3,
                cutlass.Int32,
            )
            prims.cp_async_shared_global(
                smem_ptr, gmem_ptr, PAGE_OFFSET_BYTES, CP_ASYNC_CACHE_CA
            )

    @producer_work
    @cute.jit
    def load_k0(self, stage_info: StageInfo, *, section: cutlass.Constexpr[MlaStage]):
        """Prefetch page offsets for K instance 0."""

        # Producer work for the page-offset stage that feeds K instance 0.
        self._producer_load_page_offsets(stage_info, 0, 0, section=section)

    @producer_work
    @cute.jit
    def load_k1(self, stage_info: StageInfo, *, section: cutlass.Constexpr[MlaStage]):
        """Prefetch page offsets for K instance 1."""

        # Producer work for the page-offset stage that feeds K instance 1.
        self._producer_load_page_offsets(stage_info, 1, 0, section=section)

    @producer_work
    @cute.jit
    def load_v0(self, stage_info: StageInfo, *, section: cutlass.Constexpr[MlaStage]):
        """Prefetch deferred page offsets for V instance 0."""

        # Producer work for the deferred page-offset stage that feeds V instance 0.
        self._producer_load_page_offsets(stage_info, 0, 1, section=section)

    @producer_work
    @cute.jit
    def load_v1(self, stage_info: StageInfo, *, section: cutlass.Constexpr[MlaStage]):
        """Prefetch deferred page offsets for V instance 1."""

        # Producer work for the deferred page-offset stage that feeds V instance 1.
        self._producer_load_page_offsets(stage_info, 1, 1, section=section)

    @consumer_work(returns=cached_page_ids)
    @cute.jit
    def read_offsets(
        self,
        stage_info: StageInfo,
        *,
        cached_page_ids,
        cache_slot: cutlass.Constexpr[int] = 0,
    ):
        """Cache staged page IDs in the selected K stream's register slot."""
        # The load task waits on the page-offset stage, snapshots the page ids
        # into registers, and keeps swaps K0/K1 live until their delayed V use.
        del stage_info
        cache_base = cache_slot * self.cfg.pages_per_kv_tile
        for page_frag in cutlass.range_constexpr(self.cfg.pages_per_kv_tile):
            cached_page_ids[cache_base + page_frag] = self.page_id(page_frag)
        return cached_page_ids


# =====================================================================
# SmemKvResource — K/V SMEM buffer with TmaUmmaAsync pipeline
# =====================================================================


@dataclass(kw_only=True)
class SmemKvResource(MlaResource):
    """Shared K/V staging resource for latent and rope tiles."""

    _task_local_specs: ClassVar[tuple[tuple, ...]] = (
        (
            "kv_desc_var",
            Int64,
            Int64(0),
            "SMEM descriptor for staged K tiles.",
            "kv_desc",
        ),
        (
            "v_desc_0_var",
            Int64,
            Int64(0),
            "First V descriptor consumed by PV MMA.",
            "v_desc_0",
        ),
        (
            "v_desc_1_var",
            Int64,
            Int64(0),
            "Second V descriptor consumed by PV MMA.",
            "v_desc_1",
        ),
    )
    tma_desc_c_latent: object = None
    tma_desc_c_rope: object = None
    tma_desc_v: object = None
    c_rope_tensor: object = None
    page_offsets_kv: object = None
    page_offsets: object = None
    cache_seqs: object = None
    head_idx: object = None
    batch_idx: object = None
    cta_idx_q: object = None
    cta_idx_kv: object = None
    cta_idx_head_dim_v: object = None
    _smem_kv: object = None
    _k_desc_base: object = None
    _v_desc_base: object = None
    kv_desc_var: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    v_desc_0_var: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    v_desc_1_var: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )

    def get_smem_requirements(self):
        """Return the SMEM allocation used for staged K/V tiles."""
        num_stages = (
            self.pipeline_config.num_stages
            if self.pipeline_config is not None
            else self.cfg.kv_stages
        )
        if self._alloc is None:
            self._alloc = SmemAllocation(
                name=f"{self.name}",
                size_bytes=self.cfg.kv_smem_tile_bytes * num_stages,
                alignment=self.cfg.stensor_align,
            )
        return [self._alloc]

    @cute.jit
    def _init_smem_state(self, stage_info: StageInfo) -> None:
        """Create K/V SMEM views and base descriptors."""
        context = stage_info.context
        num_stages = (
            self.pipeline_config.num_stages
            if self.pipeline_config is not None
            else self.cfg.kv_stages
        )
        elem_count = (
            self.cfg.kv_smem_tile_bytes * num_stages
        ) // self.cfg.qkv_dtype_bytes
        self._smem_kv = smem_array(
            context, self._alloc, qkv_dtype(self.cfg), elem_count
        )
        if self._smem_kv is not None:
            # FP8 descriptors use dtype-specific major-K strides rather than
            # the BF16 128B-row swizzle group.
            k_leading_byte_offset = Int32(TCGEN05_BF16_SECOND_K_BLOCK_OFFSET_BYTES)
            v_leading_byte_offset = (
                Int32(0)
                if self.cfg.head_dim_per_stage_v == TCGEN05_BF16_K_BLOCK_WIDTH
                else Int32(TCGEN05_BF16_SECOND_K_BLOCK_OFFSET_BYTES)
            )
            stride_byte_offset = Int32(TCGEN05_BF16_SWIZZLE_STRIDE_BYTES)
            if cutlass.const_expr(self.cfg.is_fp8_qkv()):
                k_leading_byte_offset = Int32(self.cfg.kv_smem_tile_bytes)
                v_leading_byte_offset = Int32(0)
                stride_byte_offset = Int32(
                    qkv_major_k_stride_bytes_for(
                        self.cfg, self.cfg.head_dim_per_stage_kv
                    )
                )
            self._k_desc_base = cprims.Tcgen05SmemDesc.build(
                self._smem_kv,
                leading_byte_offset=k_leading_byte_offset,
                stride_byte_offset=stride_byte_offset,
                layout=qkv_smem_swizzle(self.cfg),
            )
            self._v_desc_base = cprims.Tcgen05SmemDesc.build(
                self._smem_kv,
                leading_byte_offset=v_leading_byte_offset,
                stride_byte_offset=stride_byte_offset,
                layout=qkv_smem_swizzle(self.cfg),
            )

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_load_state(self, stage_info: StageInfo) -> None:
        """Initialize K/V SMEM state before producer TMA loads."""

        # Producer aux work initializes K/V SMEM views for the load task.
        self._init_smem_state(stage_info)

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_descriptor_state(self, stage_info: StageInfo) -> None:
        """Initialize K/V SMEM state before descriptor consumption."""

        # Consumer aux work initializes K/V SMEM views for the MMA task.
        self._init_smem_state(stage_info)

    @cute.jit
    def _stage_base(self, stage_info: StageInfo):
        """Return the base of the current K/V SMEM pipeline stage."""

        stage_elems = self.cfg.kv_smem_stage_elements
        return self._smem_kv.subview(stage_info.stage_idx * Int32(stage_elems))

    @cute.jit
    def _producer_copy_rope_stage(
        self,
        stage_info: StageInfo,
        page_ids,
        stage_base,
    ):
        """Copy paged rope values into the staged K/V SMEM tile."""
        cfg = self.cfg
        lane_idx = lane_idx_from_thread(cute.arch.thread_idx()[0])
        total_elems = Int32(cfg.tile_size_kv * cfg.head_dim_per_stage_kv)
        for elem_idx in cutlass.range(
            lane_idx, total_elems, Int32(WARP_LANES), unroll=1
        ):
            token_idx = elem_idx // Int32(cfg.head_dim_per_stage_kv)
            dim_idx = elem_idx - token_idx * Int32(cfg.head_dim_per_stage_kv)
            src_dim_idx = dim_idx
            if dim_idx >= Int32(cfg.rope_dim):
                src_dim_idx = dim_idx - Int32(cfg.rope_dim)
            page_frag = token_idx // Int32(cfg.num_tokens_per_page)
            token_in_page = token_idx - page_frag * Int32(cfg.num_tokens_per_page)
            page_id = Int32(page_ids[page_frag])
            stage_base[elem_idx] = self.c_rope_tensor[
                token_in_page, src_dim_idx, page_id
            ]
        cute.arch.fence_view_async_shared()
        if prims.elect_sync():
            prims.mbarrier_complete_tx(
                stage_info.barrier, Int32(cfg.kv_smem_tile_bytes)
            )

    @cute.jit
    def _producer_load_kv_stage(
        self,
        stage_info: StageInfo,
        inst_id: int,
        is_v: int,
        *,
        stage_idx: cutlass.Constexpr[int],
        section: cutlass.Constexpr[MlaStage],
        cached_page_ids,
        page_id_slot: cutlass.Constexpr[int] = 0,
    ):
        """Load one staged K or V tile using the consumed page-id payload."""
        cfg = self.cfg
        qk_stage_idx = staged_kv_head_dim_call_idx(
            cfg,
            stage_info,
            inst_id,
            is_v,
            stage_idx=stage_idx,
            section=section,
        )
        if cutlass.const_expr(is_v):
            active_width = cfg.v_head_stage_width(qk_stage_idx)
            cta_idx_head_dim_v = cta_idx_head_dim_v_for_stage(
                self.cta_idx_head_dim_v, stage_info
            )
            dim_offset = head_dim_cta_offset_v(cfg, cta_idx_head_dim_v) + Int32(
                qk_stage_idx * cfg.head_dim_per_stage_v
            )
            tma_desc = self.tma_desc_v
        else:
            active_width = cfg.qk_head_stage_width(qk_stage_idx)
            if cutlass.const_expr(
                qk_stage_idx < cfg.latent_dim // cfg.head_dim_per_stage_kv
            ):
                dim_offset = Int32(qk_stage_idx * cfg.head_dim_per_stage_kv)
                tma_desc = self.tma_desc_c_latent
            else:
                dim_offset = Int32(0)
                tma_desc = self.tma_desc_c_rope

        if cutlass.const_expr(tma_desc is None):
            return
        local_tile_idx = local_kv_tile_idx(
            cfg, stage_info, inst_id, is_v, section=section
        )
        batch_idx = batch_idx_for_stage_cfg(self.batch_idx, cfg, stage_info)
        cta_idx_q = cta_idx_q_for_stage(self.cta_idx_q, stage_info)
        cta_idx_kv = cta_idx_kv_for_stage(self.cta_idx_kv, stage_info)
        seq_len_kv = runtime_seq_len_kv_from_task_cache(
            cfg,
            decode_gen_task_cache(stage_info),
            cta_idx_q,
            self.cu_seqlens_q,
            batch_idx,
        )
        tile_idx = global_kv_tile_idx(cfg, local_tile_idx, seq_len_kv, cta_idx_kv)
        stage_base = self._stage_base(stage_info)
        inner_width = min(tma_inner_dim_elems(cfg), active_width)
        uses_compact_fp8_rope_stage = (
            cfg.load_num_warps == 1
            and cfg.is_fp8_qkv()
            and not is_v
            and cfg.rope_dim == 64
            and active_width == cfg.rope_dim
            and cfg.head_dim_per_stage_kv == 128
        )

        if cutlass.const_expr(cfg.use_paged_kv == 1):
            if cutlass.const_expr(
                self.page_offsets_kv is None and self.page_offsets is None
            ):
                return
            page_ids = None
            if cutlass.const_expr(self.page_offsets_kv is not None):
                page_ids = cached_page_ids
            pages_per_tile = cfg.pages_per_kv_tile
            first_page_elems = inner_width * cfg.num_tokens_per_page
            first_tile_elems = inner_width * cfg.tile_size_kv
            last_valid_page = (
                seq_len_kv + Int32(cfg.num_tokens_per_page - 1)
            ) // Int32(cfg.num_tokens_per_page) - Int32(1)
            # Keep the small set of independent page TMA issues straight-line.
            for page_frag in cutlass.range(pages_per_tile, unroll_full=True):
                if cutlass.const_expr(self.page_offsets_kv is not None):
                    page_id = Int32(
                        page_ids[page_id_slot * cfg.pages_per_kv_tile + page_frag]
                    )
                else:
                    logical_page_idx = cute.math.min(
                        tile_idx * Int32(pages_per_tile) + Int32(page_frag),
                        last_valid_page,
                    )
                    page_id = Int32(self.page_offsets[logical_page_idx, batch_idx])
                page_base = Int32(page_frag * first_page_elems)
                smem_page_offset = page_base
                if prims.elect_sync():
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        stage_base.data_ptr(smem_page_offset),
                        tma_desc,
                        (dim_offset, Int32(0), page_id),
                        stage_info.barrier,
                    )
                if cutlass.const_expr(active_width > inner_width):
                    second_half_offset = Int32(first_tile_elems) + smem_page_offset
                    if prims.elect_sync():
                        prims.cp_async_bulk_tensor_shared_cta_global(
                            stage_base.data_ptr(second_half_offset),
                            tma_desc,
                            (dim_offset + Int32(inner_width), Int32(0), page_id),
                            stage_info.barrier,
                        )
                if cutlass.const_expr(
                    active_width < cfg.head_dim_per_stage_kv
                    and not uses_compact_fp8_rope_stage
                ):
                    if prims.elect_sync():
                        # The shared K/V TMA pipeline expects one full QK-sized
                        # stage. Duplicate short K/V slices into the unused half
                        # so the expected transaction byte count is satisfied.
                        prims.cp_async_bulk_tensor_shared_cta_global(
                            stage_base.data_ptr(
                                Int32(first_tile_elems) + smem_page_offset
                            ),
                            tma_desc,
                            (dim_offset, Int32(0), page_id),
                            stage_info.barrier,
                        )
            if cutlass.const_expr(uses_compact_fp8_rope_stage):
                if prims.elect_sync():
                    # The FP8 RoPE MMA consumes only the 64-wide s64b payload.
                    # Retire the intentionally unwritten half of the fixed
                    # 128-wide pipeline stage instead of rereading every page.
                    missing_stage_bytes = (
                        cfg.tile_size_kv
                        * (cfg.head_dim_per_stage_kv - active_width)
                        * cfg.qkv_dtype_bytes
                    )
                    prims.mbarrier_complete_tx(
                        stage_info.barrier, Int32(missing_stage_bytes)
                    )
        else:
            tile_offset = tile_idx * Int32(cfg.tile_size_kv)
            if prims.elect_sync():
                prims.cp_async_bulk_tensor_shared_cta_global(
                    stage_base,
                    tma_desc,
                    (
                        dim_offset,
                        tile_offset,
                        head_idx_for_stage(self.head_idx, cfg, stage_info),
                        batch_idx,
                    ),
                    stage_info.barrier,
                )
                if cutlass.const_expr(active_width > inner_width):
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        stage_base.data_ptr(Int32(inner_width * cfg.tile_size_kv)),
                        tma_desc,
                        (
                            dim_offset + Int32(inner_width),
                            tile_offset,
                            head_idx_for_stage(self.head_idx, cfg, stage_info),
                            batch_idx,
                        ),
                        stage_info.barrier,
                    )
                if cutlass.const_expr(active_width < cfg.head_dim_per_stage_kv):
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        stage_base.data_ptr(Int32(active_width * cfg.tile_size_kv)),
                        tma_desc,
                        (
                            dim_offset,
                            tile_offset,
                            head_idx_for_stage(self.head_idx, cfg, stage_info),
                            batch_idx,
                        ),
                        stage_info.barrier,
                    )

    @producer_work
    @cute.jit
    def load_k0(
        self,
        stage_info: StageInfo,
        *,
        stage_idx: cutlass.Constexpr[int],
        section: cutlass.Constexpr[MlaStage],
        cached_page_ids,
        page_id_slot: cutlass.Constexpr[int] = 0,
    ):
        """Load K instance 0 into the acquired K/V SMEM stage."""

        # ProdWork: fill K instance 0 from the page IDs consumed by the load task.
        self._producer_load_kv_stage(
            stage_info,
            0,
            0,
            stage_idx=stage_idx,
            section=section,
            cached_page_ids=cached_page_ids,
            page_id_slot=page_id_slot,
        )

    @producer_work
    @cute.jit
    def load_k1(
        self,
        stage_info: StageInfo,
        *,
        stage_idx: cutlass.Constexpr[int],
        section: cutlass.Constexpr[MlaStage],
        cached_page_ids,
        page_id_slot: cutlass.Constexpr[int] = 0,
    ):
        """Load K instance 1 into the acquired K/V SMEM stage."""

        # ProdWork: fill K instance 1 from the page IDs consumed by the load task.
        self._producer_load_kv_stage(
            stage_info,
            1,
            0,
            stage_idx=stage_idx,
            section=section,
            cached_page_ids=cached_page_ids,
            page_id_slot=page_id_slot,
        )

    @producer_work
    @cute.jit
    def load_v0(
        self,
        stage_info: StageInfo,
        *,
        stage_idx: cutlass.Constexpr[int],
        section: cutlass.Constexpr[MlaStage],
        cached_page_ids,
        page_id_slot: cutlass.Constexpr[int] = 0,
    ):
        """Load deferred V instance 0 into the acquired K/V SMEM stage."""

        # ProdWork: fill V instance 0 from the page IDs consumed by the load task.
        self._producer_load_kv_stage(
            stage_info,
            0,
            1,
            stage_idx=stage_idx,
            section=section,
            cached_page_ids=cached_page_ids,
            page_id_slot=page_id_slot,
        )

    @producer_work
    @cute.jit
    def load_v1(
        self,
        stage_info: StageInfo,
        *,
        stage_idx: cutlass.Constexpr[int],
        section: cutlass.Constexpr[MlaStage],
        cached_page_ids,
        page_id_slot: cutlass.Constexpr[int] = 0,
    ):
        """Load deferred V instance 1 into the acquired K/V SMEM stage."""

        # ProdWork: fill V instance 1 from the page IDs consumed by the load task.
        self._producer_load_kv_stage(
            stage_info,
            1,
            1,
            stage_idx=stage_idx,
            section=section,
            cached_page_ids=cached_page_ids,
            page_id_slot=page_id_slot,
        )

    @cute.jit
    def _set_desc(
        self,
        stage_info: StageInfo,
        is_v: int,
        inst_id: int = 0,
        *,
        k_subtile_idx: cutlass.Constexpr[int] = 0,
    ):
        """Build a K or V SMEM descriptor for the current producer stage."""
        # Descriptor offsets mirror _init_smem_state, but are rebuilt against the
        # current pipeline stage because K and delayed-V can be consumed from
        # different stages.
        stride_byte_offset = Int32(TCGEN05_BF16_SWIZZLE_STRIDE_BYTES)
        if cutlass.const_expr(self.cfg.is_fp8_qkv()):
            stride_byte_offset = Int32(
                qkv_major_k_stride_bytes_for(self.cfg, self.cfg.head_dim_per_stage_kv)
            )
        if cutlass.const_expr(is_v):
            v_leading_byte_offset = (
                Int32(0)
                if self.cfg.head_dim_per_stage_v == TCGEN05_BF16_K_BLOCK_WIDTH
                else Int32(TCGEN05_BF16_SECOND_K_BLOCK_OFFSET_BYTES)
            )
            if cutlass.const_expr(self.cfg.is_fp8_qkv()):
                v_leading_byte_offset = Int32(0)
            desc = cprims.Tcgen05SmemDesc.build(
                self._stage_base(stage_info),
                leading_byte_offset=v_leading_byte_offset,
                stride_byte_offset=stride_byte_offset,
                layout=qkv_smem_swizzle(self.cfg),
            )
            return desc
        else:
            k_leading_byte_offset = Int32(TCGEN05_BF16_SECOND_K_BLOCK_OFFSET_BYTES)
            if cutlass.const_expr(self.cfg.is_fp8_qkv()):
                k_leading_byte_offset = Int32(self.cfg.kv_smem_tile_bytes)
                qk_stage_idx = k_subtile_idx
                if cutlass.const_expr(
                    self.cfg.rope_dim == 64
                    and qk_stage_idx
                    == self.cfg.latent_dim // self.cfg.head_dim_per_stage_kv
                ):
                    k_leading_byte_offset = Int32(
                        self.cfg.tile_size_kv
                        * self.cfg.rope_dim
                        * self.cfg.qkv_dtype_bytes
                    )
                    return cprims.Tcgen05SmemDesc.build(
                        self._stage_base(stage_info),
                        leading_byte_offset=k_leading_byte_offset,
                        stride_byte_offset=Int32(
                            qkv_major_k_stride_bytes_for(self.cfg, self.cfg.rope_dim)
                        ),
                        layout=qkv_smem_swizzle_for_head_dim(
                            self.cfg, self.cfg.rope_dim
                        ),
                    )
            desc = cprims.Tcgen05SmemDesc.build(
                self._stage_base(stage_info),
                leading_byte_offset=k_leading_byte_offset,
                stride_byte_offset=stride_byte_offset,
                layout=qkv_smem_swizzle(self.cfg),
            )
            return desc

    @consumer_work(returns=("kv_desc",))
    @cute.jit
    def k_desc_0(self, stage_info: StageInfo, *, k_subtile_idx: cutlass.Constexpr[int]):
        """Return the K descriptor for consumer instance 0."""
        # MMA has waited on the K stage; return a descriptor for qk_mma().
        return self._set_desc(stage_info, 0, 0, k_subtile_idx=k_subtile_idx)

    @consumer_work(returns=("kv_desc",))
    @cute.jit
    def k_desc_1(self, stage_info: StageInfo, *, k_subtile_idx: cutlass.Constexpr[int]):
        """Return the K descriptor for consumer instance 1."""
        # Same K SMEM payload, second interleaved QK instance.
        return self._set_desc(stage_info, 0, 1, k_subtile_idx=k_subtile_idx)

    @consumer_work(returns=("v_desc_0",))
    @cute.jit
    def v_desc_0(self, stage_info: StageInfo, *, v_subtile_idx: cutlass.Constexpr[int]):
        """Return the V descriptor for consumer instance 0."""
        # PV MMA consumes this descriptor after V instance 0 is ready.
        del v_subtile_idx
        return self._set_desc(stage_info, 1, 0)

    @consumer_work(returns=("v_desc_1",))
    @cute.jit
    def v_desc_1(self, stage_info: StageInfo, *, v_subtile_idx: cutlass.Constexpr[int]):
        """Return the V descriptor for consumer instance 1."""
        # PV MMA consumes this descriptor after V instance 1 is ready.
        del v_subtile_idx
        return self._set_desc(stage_info, 1, 1)


# =====================================================================
# SmemKResource — K-only SMEM view helper
# =====================================================================


@dataclass(kw_only=True)
class SmemKResource(MlaResource):
    """K/V SMEM staging buffer and descriptor producer for QK/PV MMA."""

    inst_id: cutlass.Constexpr[int] = 0
    is_v: cutlass.Constexpr[int] = 0


# =====================================================================
# SmemVResource — V-only SMEM view helper
# =====================================================================


@dataclass(kw_only=True)
class SmemVResource(MlaResource):
    """V-only view used by validation and resource graph naming."""

    inst_id: cutlass.Constexpr[int] = 0
    is_v: cutlass.Constexpr[int] = 1
