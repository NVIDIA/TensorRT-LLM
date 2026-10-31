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
from cutlass.experimental.task_scheduling.memory import SmemAllocation, TmemAllocation
from cutlass.experimental.task_scheduling.resources import (
    StageInfo,
    TaskLocalVariable,
    consumer_work,
    producer_work,
)

from ...helpers.sparse_indices import cache_sparse_source_lengths
from ...helpers.nvfp4_kv import (
    publish_nvfp4_kv,
    source_is_fp8,
    load_nvfp4_panel,
    prefetch_nvfp4_kv_row,
    copy_fp8_record,
    load_nvfp4_panels,
    issue_whole_record,
    convert_nvfp4_record,
)
from ...helpers.gather import (
    gather4,
    gather4_cached_warp,
    select_gather_map,
)

from ...helpers.constants import (
    CP_ASYNC_CACHE_CA,
    PAGE_OFFSET_BYTES,
    WARP_LANES,
)

from ...helpers.layout import (
    decode_gen_task_cache,
    _TASK_CACHE_TMEM_BASE_OFFSET,
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
from ...helpers.query import query_batch_bounds, copy_q_fp8_to_tmem
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
    q_temp_alloc: object = None
    q_temp_state: object = None
    kv_alias_ref: object = None
    q_tensor: object = None
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

    def get_tmem_requirements(self):
        """Reserve NoPE query columns when WS MMA reads Q from TMEM."""
        if not self.cfg.q_nope_tmem:
            return []
        if self._tmem_alloc is None:
            self._tmem_alloc = TmemAllocation(
                name=f"{self.name}_nope_tmem", num_columns=128
            )
        return [self._tmem_alloc]

    def get_smem_requirements(self):
        """Return the SMEM allocation used for staged Q tiles."""
        if self._alloc is None:
            self._alloc = SmemAllocation(
                name=f"{self.name}",
                size_bytes=(
                    4096 if self.cfg.q_nope_tmem else self.cfg.qk_smem_tile_bytes
                )
                * self.cfg.q_stages,
                alignment=self.cfg.stensor_align,
            )
        allocations = [self._alloc]
        if self.cfg.q_tma_stage:
            if self.q_temp_alloc is None:
                self.q_temp_alloc = SmemAllocation(
                    name=f"{self.name}_q_temp_state", size_bytes=32, alignment=8
                )
            allocations.append(self.q_temp_alloc)
        return allocations

    @cute.jit
    def _init_smem_state(self, stage_info: StageInfo) -> None:
        """Create the Q SMEM view and base descriptor state."""
        context = stage_info.context
        if cutlass.const_expr(self.cfg.q_tma_stage):
            self.q_temp_state = smem_array(context, self.q_temp_alloc, Int64, 4)
        elem_count = (
            (4096 if self.cfg.q_nope_tmem else self.cfg.qk_smem_tile_bytes)
            * self.cfg.q_stages
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
        if cutlass.const_expr(self.cfg.q_tma_stage):
            relative = cute.arch.thread_idx()[0] - Int32(self.cfg.load_warp_idx * 32)
            if relative == Int32(0):
                prims.mbarrier_init(self.q_temp_state.data_ptr(), 1)
                prims.mbarrier_init(self.q_temp_state.data_ptr(2), 1)
                self.q_temp_state[3] = Int64(0)
            cute.arch.mbarrier_init_fence()
            prims.barrier_cta_sync(barrier_id=12, thread_count=256)

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
        if prims.elect_sync() and (
            self.cfg.load_num_warps == 1
            or cute.arch.thread_idx()[0] // 32 - self.cfg.load_warp_idx
            == qk_stage_idx % self.cfg.load_num_warps
        ):
            smem_offset = Int32(q_stage_smem_element_offset(self.cfg, qk_stage_idx))
            if cutlass.const_expr(self.cfg.q_nope_tmem):
                smem_offset = Int32(0)
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

    @cute.jit
    def copy_nope_to_tmem(self, stage_info):
        """Copy Q NoPE through registers from GMEM to TMEM without a TMA scratch stage.

        Four warps write repeated M64 rows for the two WS column halves. This
        is the direct-copy alternative to stage_q_via_tma, which uses eight
        load warps and temporarily borrows an empty KV slot for TMA staging."""
        cfg = self.cfg
        assert (
            cfg.tile_size_q == 64
            and cfg.latent_dim == 512
            and cfg.is_fp8_qkv()
            and cfg.q_stages == 1
        )
        relative = cute.arch.thread_idx()[0] - Int32(cfg.load_warp_idx * 32)
        # One warpgroup covers128 DP rows; repeat logical64 Q rows for the
        # two WS column halves. The other load warps issue the RoPE TMA.
        if relative < Int32(128):
            row = relative % Int32(64)
            batch = batch_idx_for_stage_cfg(self.batch_idx, cfg, stage_info)
            local_flat = (
                cta_idx_q_for_stage(self.cta_idx_q, stage_info) * Int32(cfg.num_heads_q)
                + head_idx_for_stage(self.head_idx, cfg, stage_info)
                + row
            )
            storage_flat = local_flat
            source_batch = batch
            extent = Int32(self.q_tensor.shape[0] * self.q_tensor.shape[2])
            if cutlass.const_expr(self.cu_seqlens_q is not None):
                start, length = query_batch_bounds(
                    self.cu_seqlens_q, batch, cfg.logical_seq_len_q
                )
                storage_flat = start * Int32(cfg.logical_num_heads_q) + local_flat
                extent = length * Int32(cfg.logical_num_heads_q)
                source_batch = Int32(0)
            address = (
                self.q_tensor.iterator.toint()
                + Int64(source_batch) * Int64(self.q_tensor.stride[3])
                + Int64(storage_flat) * Int64(self.q_tensor.stride[0])
            )
            cache = decode_gen_task_cache(stage_info)
            tbase = cache[_TASK_CACHE_TMEM_BASE_OFFSET] + Int32(self._tmem_alloc.offset)
            for part in cutlass.range_constexpr(4):
                words = cutlass.Array(Int32, 32, space=cutlass.AddressSpace.rmem)
                for j in cutlass.range_constexpr(32):
                    words[j] = Int32(0)
                if local_flat < extent:
                    ptr = cutlass.inttoptr(
                        address + Int64(part * 128), mem_space=1, dtype=Int32
                    )
                    loaded = ptr.load(count=32, alignment=16)
                    for j in cutlass.range_constexpr(32):
                        words[j] = loaded[j]
                prims.tcgen05_st(
                    prims.Tcgen05LdStShape.SHAPE_32X32B,
                    prims.make_tmem_ptr(tbase + Int32(part * 32), Int32),
                    words.load(0, 32),
                )
            prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
            prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
            # Keep the existing36KiB Q transaction accounting: four warp
            # completions cover32KiB logical NoPE, RoPE TMA covers4KiB.
            publish_nvfp4_kv(stage_info.barrier, 8192)

    @cute.jit
    def stage_q_via_tma(self, stage_info, rope_stage_base):
        """Stage Q NoPE with TMA in an empty KV slot, then copy it from SMEM to TMEM.

        Unlike copy_nope_to_tmem, this path avoids wide per-thread GMEM loads.
        It requires eight loaders and single-raw-prefetch geometry. Acquiring
        scratch does not advance the KV pipeline cursor; Q finishes before
        normal KV loading reuses the slot. D576 RoPE keeps its own Q stage."""
        cfg = self.cfg
        assert cfg.q_nope_tmem and cfg.load_num_warps == 8 and cfg.single_raw_prefetch
        relative = cute.arch.thread_idx()[0] - Int32(cfg.load_warp_idx * 32)
        kv = self.kv_alias_ref
        # Form a local scratch view. Reinitializing the KV resource here would
        # retain descriptors defined inside the runtime-active Q branch and
        # leak those staged values into the subsequent KV task loop.
        kv_scratch = smem_array(
            stage_info.context,
            kv._alloc,
            qkv_dtype(cfg),
            cfg.whole_kv_tile_bytes * cfg.kv_stages // cfg.qkv_dtype_bytes,
        )
        first = kv.producer_state.index
        phase = kv.producer_state.phase
        # This is only scratch reuse: no KV acquire/commit cursor is changed.
        # The subsequent normal KV producer still acquires its own stages.
        for i in cutlass.range_constexpr(1 if cfg.whole_kv_tile else 2):
            kv.pipeline.sync_object_empty.wait(first + Int32(i), phase)
        temp = kv_scratch.data_ptr(kv.kv_stage_offset(first))
        if relative == Int32(0):
            token = prims.mbarrier_arrive_expect_tx(self.q_temp_state.data_ptr(), 32768)
            self.q_temp_state[1] = token
        prims.barrier_cta_sync(barrier_id=12, thread_count=256)
        qpart = relative // Int32(32)
        if qpart < Int32(4) and prims.elect_sync():
            head = head_idx_for_stage(self.head_idx, cfg, stage_info)
            query = cta_idx_q_for_stage(self.cta_idx_q, stage_info)
            batch = batch_idx_for_stage_cfg(self.batch_idx, cfg, stage_info)
            coords = self._query_tma_coords(
                qpart * Int32(128), query * Int32(cfg.num_heads_q) + head, batch
            )
            prims.cp_async_bulk_tensor_shared_cta_global(
                temp + qpart * Int32(8192),
                self.tma_desc_q_latent,
                coords,
                self.q_temp_state.data_ptr(),
            )
        if cutlass.const_expr(cfg.rope_dim):
            self._load_q_stage(
                stage_info, rope_stage_base, 4, Int32(0), self.tma_desc_q_rope
            )
        token = self.q_temp_state[1]
        while not prims.mbarrier_try_wait(self.q_temp_state.data_ptr(), token):
            pass
        if relative < Int32(32):
            cache = decode_gen_task_cache(stage_info)
            tbase = cache[_TASK_CACHE_TMEM_BASE_OFFSET] + Int32(self._tmem_alloc.offset)
            copy_phase = Int32(self.q_temp_state[3])
            if prims.elect_sync():
                copy_q_fp8_to_tmem(temp, tbase, cta_group=1)
                prims.tcgen05_commit(
                    self.q_temp_state.data_ptr(2), group=prims.CTAGroup.CTA_1
                )
            while not prims.mbarrier_try_wait_parity(
                self.q_temp_state.data_ptr(2), copy_phase
            ):
                pass
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            cute.arch.sync_warp()
            if prims.elect_sync():
                self.q_temp_state[3] = Int64(copy_phase ^ Int32(1))
            prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
            # One warp now publishes the whole32KiB logical NoPE payload,
            # after its TCGEN copy commit has completed.
            publish_nvfp4_kv(stage_info.barrier, 32768)
        # Every Q staging read ends before any loader may expand KV over it.
        prims.barrier_cta_sync(barrier_id=12, thread_count=256)

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
        if cutlass.const_expr(cfg.q_tma_stage):
            self.stage_q_via_tma(stage_info, stage_base)
            return
        if cutlass.const_expr(cfg.q_nope_tmem):
            self.copy_nope_to_tmem(stage_info)
        else:
            for qk_stage_idx in cutlass.range_constexpr(latent_stages):
                dim_offset = Int32(qk_stage_idx * cfg.head_dim_per_stage_kv)
                self._load_q_stage(
                    stage_info,
                    stage_base,
                    qk_stage_idx,
                    dim_offset,
                    self.tma_desc_q_latent,
                )
        if cutlass.const_expr(cfg.rope_dim > 0):
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
            if cutlass.const_expr(self.cfg.q_nope_tmem):
                rope_stage_offset = Int32(0)
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
        if cutlass.const_expr(self.cfg.q_nope_tmem):
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            cache = decode_gen_task_cache(stage_info)
            desc_q = Int64(
                cache[_TASK_CACHE_TMEM_BASE_OFFSET] + Int32(self._tmem_alloc.offset)
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
    nv_target: cutlass.Constexpr = None
    request_lengths: object = None
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
            sync_object_full=base.sync_object_full,
            sync_object_empty=base.sync_object_empty,
            num_stages=base.num_stages,
            producer_mask=base.producer_mask,
            consumer_mask=base.consumer_mask,
        )

    def get_smem_requirements(self):
        """Return the SMEM allocation for cached page offsets."""
        if self.cfg.single_raw_prefetch:
            assert self.cfg.page_offsets_entries_per_stage % 4 == 0
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
        if cutlass.const_expr(self.cfg.sparse_direct):
            self.request_lengths = cutlass.Array(
                Int32, 2, space=cutlass.AddressSpace.rmem
            )
        if cutlass.const_expr(self.cfg.async_nvfp4_kv_gather):
            self.nv_target._init_smem_state(stage_info)
            prims.barrier_cta_sync(
                barrier_id=13,
                thread_count=32
                * (self.cfg.load_num_warps + self.cfg.page_offsets_num_warps),
            )

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def bind_sparse_request(self, stage_info: StageInfo):
        """Cache live source lengths for this query before its KV/softmax loop."""
        if cutlass.const_expr(self.cfg.sparse_direct):
            cache_sparse_source_lengths(
                self.page_offsets,
                batch_idx_for_stage_cfg(self.batch_idx, self.cfg, stage_info),
                self.request_lengths,
                include_primary_tiles=False,
            )

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
            max(
                1,
                cache_slots
                * (
                    self.cfg.sparse_cache_words
                    if self.cfg.num_tokens_per_page == 1
                    else self.cfg.pages_per_kv_tile
                ),
            ),
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
        if cutlass.const_expr(cfg.sparse_direct):
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
                cfg,
                stage_info,
                inst_id,
                is_v,
                section=section,
                requested_splits=self.runtime_split_kv,
            )
            tile_idx = global_kv_tile_idx(
                cfg,
                local_tile_idx,
                seq_len_kv,
                cta_idx_kv,
                requested_splits=self.runtime_split_kv,
                task_cache=decode_gen_task_cache(stage_info),
            )
            tile_start = tile_idx * pages_per_tile
            rows = self.page_offsets.bind_query_row(
                batch_idx, self.request_lengths[0], self.request_lengths[1]
            )
            routes = rows.select_source_tile(tile_start)
            position = lane_idx * Int32(4)
            if cutlass.const_expr(cfg.async_nvfp4_kv_gather):
                issuer = cute.arch.thread_idx()[0] // 32 - cfg.page_offsets_warp_idx
                position = issuer * Int32(64) + (lane_idx % Int32(16)) * Int32(4)
            smem_offset = (
                stage_info.stage_idx * Int32(cfg.page_offsets_entries_per_stage)
                + position
            )
            raw = cutlass.Array(Int32, 4, space=cutlass.AddressSpace.rmem)
            for j in cutlass.range_constexpr(4):
                raw[j] = Int32(0x7FFFFFFF)
            if position < pages_per_tile and (
                not cfg.async_nvfp4_kv_gather or lane_idx < 16
            ):
                # Fetch four physical storage-row IDs for this lane's source.
                # The same quad fills the metadata ring and issues the raw TMA
                # gather, avoiding another global index read in KV loaders.
                # Full tiles can use unmasked reads; live-length tails and
                # holes preserve the source tag on synthesized invalid rows.
                if tile_start + pages_per_tile <= routes.origin + routes.length:
                    quad = routes.load_full_quad(tile_start + position)
                    for j in cutlass.range_constexpr(4):
                        raw[j] = quad[j]
                else:
                    quad = routes.load_quad(tile_start + position)
                    for j in cutlass.range_constexpr(4):
                        raw[j] = quad[j]
                self._smem_page_offsets.data_ptr(smem_offset).store(
                    raw.load(0, 4), alignment=16
                )
            cute.arch.fence_view_async_shared()
            if cutlass.const_expr(cfg.async_nvfp4_kv_gather):
                target = self.nv_target
                if lane_idx < Int32(16):
                    for j in cutlass.range_constexpr(4):
                        prefetch_nvfp4_kv_row(
                            target.c_latent_tensor,
                            target.c_rope_tensor,
                            Int32(raw[j]),
                            cfg,
                        )
                if cutlass.const_expr(cfg.single_raw_prefetch):
                    # PG has two stages; its index alternates once per tile,
                    # exactly matching the one raw-empty barrier's epochs.
                    while not prims.mbarrier_try_wait_parity(
                        target._nv_barriers.data_ptr(2), stage_info.stage_idx ^ Int32(1)
                    ):
                        pass
                    cute.arch.fence_view_async_shared()
                    destinations = (target._nv_raw.data_ptr(),)
                    if issuer == Int32(0) and prims.elect_sync():
                        header = stage_info.stage_idx * Int32(
                            cfg.page_offsets_entries_per_stage
                        ) + Int32(128)
                        self._smem_page_offsets[header] = stage_info.stage_idx
                        self._smem_page_offsets[header + Int32(1)] = (
                            self.producer_state.phase ^ Int32(1)
                        )
                else:
                    destinations = ()
                    for part in cutlass.range_constexpr(5):
                        slot = stage_info.stage_idx * Int32(5) + Int32(part)
                        target.pipeline.sync_object_empty.wait(
                            slot, self.producer_state.phase
                        )
                        destinations += (
                            target._smem_kv.data_ptr(target.kv_stage_offset(slot)),
                        )
                record_bytes = Int32(cfg.nvfp4_row_bytes)
                if source_is_fp8(cfg, raw[0] < Int32(0)):
                    record_bytes = Int32(512)
                issue_whole_record(
                    destinations[0],
                    target._nv_barriers.data_ptr(stage_info.stage_idx),
                    raw,
                    target.tma_desc_c_latent,
                    target.tma_desc_c_rope,
                    issuer=issuer,
                    row_bytes=record_bytes,
                )
            return
        for fragment in cutlass.range_constexpr((cfg.pages_per_kv_tile + 31) // 32):
            fragment_lane = lane_idx + Int32(fragment * 32)
            if fragment_lane < pages_per_tile:
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
                    cfg,
                    stage_info,
                    inst_id,
                    is_v,
                    section=section,
                    requested_splits=self.runtime_split_kv,
                )
                tile_idx = global_kv_tile_idx(
                    cfg,
                    local_tile_idx,
                    seq_len_kv,
                    cta_idx_kv,
                    requested_splits=self.runtime_split_kv,
                    task_cache=decode_gen_task_cache(stage_info),
                )
                last_valid_page = cute.math.max(
                    (seq_len_kv + Int32(cfg.num_tokens_per_page - 1))
                    // Int32(cfg.num_tokens_per_page)
                    - Int32(1),
                    Int32(0),
                )
                logical_page_idx = cute.math.min(
                    tile_idx * pages_per_tile + fragment_lane,
                    last_valid_page,
                )
                smem_offset = (
                    stage_info.stage_idx * Int32(cfg.page_offsets_entries_per_stage)
                    + fragment_lane
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
        if cutlass.const_expr(self.cfg.sparse_d576_reload):
            assert self.cfg.cache_uniform_sparse_quads
            # Snapshot this warp's eight quads once per K / V tile as
            # warp-uniform coordinates; every head-dimension stage of the
            # tile then issues them from one elected lane without the
            # per-lane R2UR waterfall.
            quads = self.cfg.tile_size_kv // (4 * self.cfg.load_num_warps)
            warp = cute.arch.make_warp_uniform(cute.arch.warp_idx()) - Int32(
                self.cfg.load_warp_idx
            )
            row_base = self.consumer_work_stage * Int32(
                self.cfg.page_offsets_entries_per_stage
            ) + warp * Int32(4 * quads)
            for quad in cutlass.range_constexpr(quads):
                rows = self._smem_page_offsets.data_ptr(
                    row_base + Int32(4 * quad)
                ).load(count=4, alignment=16)
                for j in cutlass.range_constexpr(4):
                    cached_page_ids[6 + quad * 4 + j] = cute.arch.make_warp_uniform(
                        Int32(rows[j])
                    )
            return cached_page_ids
        if cutlass.const_expr(self.cfg.num_tokens_per_page == 1):
            lane = lane_idx_from_thread(cute.arch.thread_idx()[0])
            warp = cute.arch.thread_idx()[0] // 32 - self.cfg.load_warp_idx
            groups_per_warp = self.cfg.tile_size_kv // (4 * self.cfg.load_num_warps)
            for j in cutlass.range_constexpr(4):
                cached_page_ids[cache_slot * self.cfg.sparse_cache_words + j] = Int32(
                    0x7FFFFFFF
                )
            if lane < groups_per_warp:
                quad = warp * groups_per_warp + lane
                offset = (
                    self.consumer_work_stage * self.cfg.page_offsets_entries_per_stage
                    + quad * 4
                )
                values = self._smem_page_offsets.data_ptr(offset).load(
                    count=4, alignment=16
                )
                for j in cutlass.range_constexpr(4):
                    cached_page_ids[cache_slot * self.cfg.sparse_cache_words + j] = (
                        values[j]
                    )
            base = cache_slot * self.cfg.sparse_cache_words
            # Actual physical adjacency within one source is required. A
            # full valid prefix can still contain random rows. Eight quads
            # (32 rows) are needed for bulk; smaller issuer shares use gather4.
            if cutlass.const_expr(groups_per_warp >= 8):
                # A chunk stays within one source. Invalid coordinates retain
                # its tag because the first coordinate selects the shared map.
                first = Int32(
                    prims.shfl_sync(
                        thread_mask=0xFFFFFFFF,
                        val=cached_page_ids[base],
                        offset=lane & Int32(-8),
                        mask_and_clamp=0x1F,
                        kind=prims.Shfl.IDX,
                    )
                )
                contiguous = (first & Int32(0x7FFFFFFF)) != Int32(0x7FFFFFFF)
                empty = cutlass.Boolean(True)
                for j in cutlass.range_constexpr(4):
                    raw = cached_page_ids[base + j]
                    contiguous = contiguous and raw == first + (lane % 8) * 4 + j
                    empty = empty and (raw & Int32(0x7FFFFFFF)) == Int32(0x7FFFFFFF)
                cached_page_ids[base + 4] = Int32(
                    cute.arch.vote_ballot_sync(contiguous)
                )
                cached_page_ids[base + 5] = Int32(cute.arch.vote_ballot_sync(empty))
            else:
                cached_page_ids[base + 4] = Int32(0)
                cached_page_ids[base + 5] = Int32(0)

        else:
            cache_base = cache_slot * self.cfg.pages_per_kv_tile
            for page_frag in cutlass.range_constexpr(self.cfg.pages_per_kv_tile):
                cached_page_ids[cache_base + page_frag] = self.page_id(page_frag)
        if cutlass.const_expr(self.cfg.cache_uniform_sparse_quads):
            # Stage once per token tile, then reuse these uniform coordinates
            # across all four head-dimension stages. Keep the raw lane cache
            # for the contiguous/empty bulk path and source selection.
            base = cache_slot * self.cfg.sparse_cache_words
            bulk = (cached_page_ids[base + 4] & Int32(255)) == Int32(255)
            empty = (cached_page_ids[base + 5] & Int32(255)) == Int32(255)
            if self.cfg.load_num_warps == 8 or not (bulk or empty):
                for owner in cutlass.range_constexpr(
                    self.cfg.tile_size_kv // (4 * self.cfg.load_num_warps)
                ):
                    for j in cutlass.range_constexpr(4):
                        raw = prims.shfl_sync(
                            thread_mask=0xFFFFFFFF,
                            val=cached_page_ids[base + j],
                            offset=owner,
                            mask_and_clamp=0x1F,
                            kind=prims.Shfl.IDX,
                        )
                        row = raw & Int32(0x7FFFFFFF)
                        row = row if row != Int32(0x7FFFFFFF) else Int32(-1)
                        cached_page_ids[base + 6 + owner * 4 + j] = (
                            cute.arch.make_warp_uniform(row)
                        )
            else:
                # No gather reads these coordinates on a bulk/empty tile.
                # Constants avoid carrying the previous tile's uniform cache.
                for j in cutlass.range_constexpr(self.cfg.sparse_cache_words - 6):
                    cached_page_ids[base + 6 + j] = Int32(0)
        if cutlass.const_expr(self.cfg.single_raw_prefetch):
            header = stage_info.stage_idx * Int32(
                self.cfg.page_offsets_entries_per_stage
            ) + Int32(128)
            cached_page_ids[cache_slot * self.cfg.sparse_cache_words + 4] = (
                self._smem_page_offsets[header]
            )
            cached_page_ids[cache_slot * self.cfg.sparse_cache_words + 5] = (
                self._smem_page_offsets[header + Int32(1)]
            )
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
    tma_desc_bulk_extra: object = None
    tma_desc_sparse_rope: object = None
    tma_desc_sparse_extra_rope: object = None
    c_rope_tensor: object = None
    c_latent_tensor: object = None
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

    _nv_raw_alloc: object = None
    _nv_raw: object = None
    _nv_scratch_alloc: object = None
    _nv_barrier_alloc: object = None
    _nv_scratch: object = None
    _nv_barriers: object = None

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
                size_bytes=(
                    self.cfg.whole_kv_tile_bytes * num_stages
                    if self.cfg.whole_kv_tile
                    else self.cfg.kv_smem_tile_bytes * num_stages
                    - (16384 if self.cfg.compact_nv_rope else 0)
                ),
                alignment=self.cfg.stensor_align,
            )
        allocations = [self._alloc]
        if self.cfg.has_nvfp4_kv:
            # Whole-record gather carries SF and RoPE inside the raw record;
            # it never reads the older separate scale scratch allocation.
            if not self.cfg.async_nvfp4_kv_gather:
                if self._nv_scratch_alloc is None:
                    self._nv_scratch_alloc = SmemAllocation(
                        name=f"{self.name}_nv_scratch",
                        size_bytes=self.cfg.tile_size_kv * 64,
                        alignment=1024,
                    )
                allocations.append(self._nv_scratch_alloc)
            if self._nv_barrier_alloc is None:
                self._nv_barrier_alloc = SmemAllocation(
                    name=f"{self.name}_nv_barriers",
                    size_bytes=self.cfg.load_num_warps * 8,
                    alignment=8,
                )
            allocations.append(self._nv_barrier_alloc)
            if self.cfg.single_raw_prefetch:
                if self._nv_raw_alloc is None:
                    self._nv_raw_alloc = SmemAllocation(
                        name=f"{self.name}_raw_record",
                        size_bytes=128 * self.cfg.raw_record_stride,
                        alignment=1024,
                    )
                allocations.append(self._nv_raw_alloc)
        return allocations

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
            self.cfg.whole_kv_tile_bytes * num_stages
            if self.cfg.whole_kv_tile
            else self.cfg.kv_smem_tile_bytes * num_stages
            - (16384 if self.cfg.compact_nv_rope else 0)
        ) // self.cfg.qkv_dtype_bytes
        self._smem_kv = smem_array(
            context, self._alloc, qkv_dtype(self.cfg), elem_count
        )
        if cutlass.const_expr(self.cfg.single_raw_prefetch):
            self._nv_raw = smem_array(
                context,
                self._nv_raw_alloc,
                cutlass.Uint8,
                128 * self.cfg.raw_record_stride,
            )
        if cutlass.const_expr(self.cfg.has_nvfp4_kv):
            self._nv_scratch = smem_array(
                context,
                self._nv_scratch_alloc,
                cutlass.Uint8,
                self.cfg.tile_size_kv * 64,
            )
            self._nv_barriers = smem_array(
                context, self._nv_barrier_alloc, Int64, self.cfg.load_num_warps
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
        if cutlass.const_expr(self.cfg.has_nvfp4_kv):
            warp = cute.arch.thread_idx()[0] // 32 - self.cfg.load_warp_idx
            if prims.elect_sync():
                count = Int32(
                    self.cfg.page_offsets_num_warps
                    if self.cfg.async_nvfp4_kv_gather
                    else 1
                )
                if cutlass.const_expr(self.cfg.single_raw_prefetch):
                    if warp == Int32(2):
                        count = Int32(self.cfg.load_num_warps)
                prims.mbarrier_init(self._nv_barriers.data_ptr(warp), count)
            cute.arch.mbarrier_init_fence()
            cute.arch.sync_warp()
            if cutlass.const_expr(self.cfg.async_nvfp4_kv_gather):
                # Raw barriers are shared by the gather and transform warps.
                # IDs 14/15 belong to TMEM; 13 is reserved for this load group.
                prims.barrier_cta_sync(
                    barrier_id=13,
                    thread_count=32
                    * (self.cfg.load_num_warps + self.cfg.page_offsets_num_warps),
                )

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_descriptor_state(self, stage_info: StageInfo) -> None:
        """Initialize K/V SMEM state before descriptor consumption."""

        # Consumer aux work initializes K/V SMEM views for the MMA task.
        self._init_smem_state(stage_info)

    @cute.jit
    def kv_stage_offset(self, stage):
        """Return the SMEM element offset for a payload slot, accounting for compact RoPE storage."""
        if cutlass.const_expr(self.cfg.whole_kv_tile):
            return stage * Int32(self.cfg.whole_kv_tile_bytes)
        offset = stage * Int32(self.cfg.kv_smem_stage_elements)
        if cutlass.const_expr(self.cfg.compact_nv_rope):
            assert (
                self.cfg.kv_stages == 10
                and self.cfg.tile_size_kv == 128
                and self.cfg.rope_dim == 64
            )
            # Each five-stage group has four16KiB NoPE panels and one8KiB
            # RoPE panel. The second group's base is therefore72KiB.
            offset -= Int32(stage >= Int32(5)) * Int32(8192)
        return offset

    @cute.jit
    def _stage_base(self, stage_info: StageInfo):
        """Return the base of the current K/V SMEM pipeline stage."""

        return self._smem_kv.subview(self.kv_stage_offset(stage_info.stage_idx))

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
    def load_nvfp4_stage_payload(
        self,
        stage_info,
        stage_base,
        cached_page_ids,
        dim_offset,
        warp,
        *,
        page_id_slot: cutlass.Constexpr[int],
        qk_stage_idx: cutlass.Constexpr[int],
        is_v: cutlass.Constexpr[int],
        active_width: cutlass.Constexpr[int],
    ):
        """Fill a KV stage using whole-record reuse or the shared panel loader.

        The caller owns the stage's full-barrier publication. Whole-record
        mode waits for metadata-issued TMA and expands/copies only the first
        feature stage; later stages reuse that payload through PV. Panel mode
        uses the same conversion helpers for separately staged K or V slices.
        """
        cfg = self.cfg
        rows_per_warp = cfg.tile_size_kv // cfg.load_num_warps
        if cutlass.const_expr(cfg.async_nvfp4_kv_gather):
            if cutlass.const_expr(qk_stage_idx == 0):
                raw_slot = stage_info.stage_idx // Int32(5)
                raw_phase = self.producer_state.phase ^ Int32(1)
                if cutlass.const_expr(cfg.single_raw_prefetch):
                    raw_slot = cached_page_ids[
                        page_id_slot * cfg.sparse_cache_words + 4
                    ]
                    raw_phase = cached_page_ids[
                        page_id_slot * cfg.sparse_cache_words + 5
                    ]
                    # Normal acquire already claimed the first stage.
                    # Retain all other expanded panels until old PV ends.
                    if cutlass.const_expr(not cfg.whole_kv_tile):
                        for part in cutlass.range_constexpr(1, 5):
                            self.pipeline.sync_object_empty.wait(
                                stage_info.stage_idx + Int32(part),
                                self.producer_state.phase,
                            )
                raw_barrier = self._nv_barriers.data_ptr(raw_slot)
                destinations = ()
                for part in cutlass.range_constexpr(5):
                    slot = stage_info.stage_idx + Int32(part)
                    destinations += (
                        self._smem_kv.data_ptr(
                            self.kv_stage_offset(stage_info.stage_idx)
                            + Int32(part * 16384)
                            if cfg.whole_kv_tile
                            else self.kv_stage_offset(slot)
                        ),
                    )
                # The metadata warp issues gathers independently, so
                # all eight original load warps can convert in parallel.
                while not prims.mbarrier_try_wait_parity(raw_barrier, raw_phase):
                    pass
                if source_is_fp8(
                    cfg,
                    cached_page_ids[page_id_slot * cfg.sparse_cache_words] < Int32(0),
                ):
                    copy_fp8_record(
                        destinations[0],
                        self._nv_raw.data_ptr(),
                        cute.arch.thread_idx()[0] - Int32(cfg.load_warp_idx * 32),
                        self._nv_barriers.data_ptr(2),
                    )
                elif cutlass.const_expr(cfg.single_raw_prefetch):
                    convert_nvfp4_record(
                        destinations[0],
                        cute.arch.thread_idx()[0] - Int32(cfg.load_warp_idx * 32),
                        stage_bytes=cfg.kv_smem_tile_bytes,
                        rope_dim=cfg.rope_dim,
                        raw=self._nv_raw.data_ptr(),
                        raw_empty=self._nv_barriers.data_ptr(2),
                    )
                else:
                    convert_nvfp4_record(
                        destinations[0],
                        cute.arch.thread_idx()[0] - Int32(cfg.load_warp_idx * 32),
                        stage_bytes=cfg.kv_smem_tile_bytes,
                        rope_dim=cfg.rope_dim,
                    )
            return
        if cutlass.const_expr(cfg.sparse_reuse_kv and not is_v):
            if cutlass.const_expr(qk_stage_idx == 0):
                # Retained-K schedules have room for two complete K
                # tiles. Claim each payload's existing empty barrier
                # before batching its gathers. The captured schedule
                # still arms/publishes each full barrier individually.
                destinations = ()
                for part in cutlass.range_constexpr(cfg.qk_head_dim_stages):
                    future = stage_info.stage_idx + part
                    slot = future - cfg.kv_stages if future >= cfg.kv_stages else future
                    if cutlass.const_expr(part > 0):
                        phase = self.producer_state.phase ^ Int32(
                            future >= cfg.kv_stages
                        )
                        self.pipeline.sync_object_empty.wait(slot, phase)
                    width = 64 if part == 4 else 128
                    destinations += (
                        self._smem_kv.data_ptr(
                            self.kv_stage_offset(slot) + warp * rows_per_warp * width
                        ),
                    )
                load_nvfp4_panels(
                    destinations,
                    (Int32(0), Int32(128), Int32(256), Int32(384)),
                    self._nv_scratch.data_ptr(warp * rows_per_warp * 64),
                    self._nv_barriers.data_ptr(warp),
                    cached_page_ids,
                    page_id_slot * cfg.sparse_cache_words,
                    self.tma_desc_c_latent,
                    self.tma_desc_c_rope,
                    self.tma_desc_v,
                    self.tma_desc_bulk_extra,
                    self.tma_desc_sparse_rope,
                    self.tma_desc_sparse_extra_rope,
                    rows=rows_per_warp,
                    rope_dim=cfg.rope_dim,
                    separate_rope=cfg.rope_dim > 0,
                    fp8=source_is_fp8(
                        cfg,
                        cached_page_ids[page_id_slot * cfg.sparse_cache_words]
                        < Int32(0),
                    ),
                )
            return
        load_nvfp4_panel(
            stage_base.data_ptr(warp * rows_per_warp * active_width),
            self._nv_scratch.data_ptr(warp * rows_per_warp * 64),
            self._nv_barriers.data_ptr(warp),
            cached_page_ids,
            page_id_slot * cfg.sparse_cache_words,
            dim_offset,
            self.tma_desc_c_latent,
            self.tma_desc_c_rope,
            self.tma_desc_v,
            self.tma_desc_bulk_extra,
            self.tma_desc_sparse_rope,
            self.tma_desc_sparse_extra_rope,
            rows=rows_per_warp,
            rope_dim=cfg.rope_dim,
            rope_only=not is_v and active_width == 64,
            load_scales=True,
            fp8=source_is_fp8(
                cfg,
                cached_page_ids[page_id_slot * cfg.sparse_cache_words] < Int32(0),
            ),
        )
        return

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
            tma_desc = (
                self.tma_desc_c_latent
                if cfg.num_tokens_per_page == 1
                else self.tma_desc_v
            )
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
            cfg,
            stage_info,
            inst_id,
            is_v,
            section=section,
            requested_splits=self.runtime_split_kv,
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
        tile_idx = global_kv_tile_idx(
            cfg,
            local_tile_idx,
            seq_len_kv,
            cta_idx_kv,
            requested_splits=self.runtime_split_kv,
            task_cache=decode_gen_task_cache(stage_info),
        )
        stage_base = self._stage_base(stage_info)
        inner_width = min(tma_inner_dim_elems(cfg), active_width)
        uses_compact_fp8_rope_stage = (
            (cfg.load_num_warps == 1 or cfg.sparse_d576_reload)
            and cfg.is_fp8_qkv()
            and not is_v
            and cfg.rope_dim == 64
            and active_width == cfg.rope_dim
            and cfg.head_dim_per_stage_kv == 128
        )

        if cutlass.const_expr(cfg.sparse_d576_reload):
            # D576 sparse K/V stage: each loader warp issues its quads (one
            # Gather4 of four token rows each) from the warp-uniform rows that
            # read_offsets cached for this tile.  The four rows land at
            # consecutive SMEM rows of the stage, so the TMA swizzle (SW128
            # latent / SW64 RoPE) is the one the UMMA descriptors expect.  Rows
            # addressed outside the pool are zero-filled by the TMA unit and
            # still count toward the stage's transaction bytes; only -1 slots
            # are also masked in softmax.
            assert cfg.cache_uniform_sparse_quads, (
                "D576 sparse loads issue from the cached uniform quads"
            )
            is_rope_stage = (not is_v) and (
                qk_stage_idx >= cfg.latent_dim // cfg.head_dim_per_stage_kv
            )
            num_load_warps = cfg.load_num_warps
            warp_in_task = Int32(0)
            if cutlass.const_expr(num_load_warps > 1):
                warp_in_task = cute.arch.make_warp_uniform(
                    cute.arch.warp_idx()
                ) - Int32(cfg.load_warp_idx)
            # Row maps: latent D128 slices (SW128) or the RoPE columns (SW64).
            descriptor = (
                self.tma_desc_c_rope if is_rope_stage else self.tma_desc_c_latent
            )
            quads_per_warp = cfg.tile_size_kv // (4 * num_load_warps)
            gather4_cached_warp(
                stage_base.data_ptr(
                    warp_in_task * Int32(quads_per_warp * 4 * active_width)
                ),
                descriptor,
                Int32(0) if is_rope_stage else dim_offset,
                cached_page_ids,
                stage_info.barrier,
                cache_start=6,
                num_quads=quads_per_warp,
                num_slices=1,
                quad_stride=4 * active_width,
                slice_stride=0,
                col_stride=0,
            )
            if cutlass.const_expr(uses_compact_fp8_rope_stage):
                if warp_in_task == Int32(0):
                    if prims.elect_sync():
                        missing_stage_bytes = (
                            cfg.tile_size_kv
                            * (cfg.head_dim_per_stage_kv - active_width)
                            * cfg.qkv_dtype_bytes
                        )
                        prims.mbarrier_complete_tx(
                            stage_info.barrier, Int32(missing_stage_bytes)
                        )
            return
        # Each lane owns four adjacent index-list entries. Bulk TMA requires
        # a verified 32-row physical run within one source; irregular rows
        # use gather4. The separate empty-window path only zero-fills padding.
        if cutlass.const_expr(cfg.num_tokens_per_page == 1):
            warp = cute.arch.thread_idx()[0] // 32 - cfg.load_warp_idx
            if cutlass.const_expr(cfg.has_nvfp4_kv):
                self.load_nvfp4_stage_payload(
                    stage_info,
                    stage_base,
                    cached_page_ids,
                    dim_offset,
                    warp,
                    page_id_slot=page_id_slot,
                    qk_stage_idx=qk_stage_idx,
                    is_v=is_v,
                    active_width=active_width,
                )
                publish_nvfp4_kv(
                    stage_info.barrier,
                    (
                        cfg.whole_kv_tile_bytes
                        if cfg.whole_kv_tile
                        else cfg.kv_smem_tile_bytes
                    )
                    // cfg.load_num_warps,
                )
                return
            if cutlass.const_expr(
                not is_v and active_width == 64 and cfg.rope_dim == 64
            ):
                # The QK-only tail uses its own 64-wide swizzle. Retire the
                # unused half of the fixed-width pipeline transaction once.
                groups_per_warp = cfg.tile_size_kv // (4 * cfg.load_num_warps)
                lane = cute.arch.thread_idx()[0] % 32
                base = page_id_slot * cfg.sparse_cache_words
                if lane < groups_per_warp:
                    raw = cached_page_ids[base]
                    desc = select_gather_map(
                        self.tma_desc_sparse_rope,
                        self.tma_desc_sparse_extra_rope,
                        raw < 0,
                    )
                    coords = cutlass.Array(Int32, 4, space=cutlass.AddressSpace.rmem)
                    for j in cutlass.range_constexpr(4):
                        row = cached_page_ids[base + j] & Int32(0x7FFFFFFF)
                        coords[j] = row if row != Int32(0x7FFFFFFF) else Int32(-1)
                    gather4(
                        stage_base.data_ptr((warp * groups_per_warp + lane) * 4 * 64),
                        desc,
                        Int32(0),
                        coords[0],
                        coords[1],
                        coords[2],
                        coords[3],
                        stage_info.barrier,
                    )
                if warp == 0 and prims.elect_sync():
                    prims.mbarrier_complete_tx(
                        stage_info.barrier,
                        Int32(cfg.tile_size_kv * 64 * cfg.qkv_dtype_bytes),
                    )
                return
            groups_per_warp = cfg.tile_size_kv // (4 * cfg.load_num_warps)
            base = page_id_slot * cfg.sparse_cache_words
            for chunk in cutlass.range_constexpr(
                max(1, cfg.tile_size_kv // (32 * cfg.load_num_warps))
            ):
                mask = Int32(255) << Int32(chunk * 8)
                contiguous = (cached_page_ids[base + 4] & mask) == mask
                empty = (cached_page_ids[base + 5] & mask) == mask
                first = Int32(
                    prims.shfl_sync(
                        thread_mask=0xFFFFFFFF,
                        val=cached_page_ids[base],
                        offset=chunk * 8,
                        mask_and_clamp=0x1F,
                        kind=prims.Shfl.IDX,
                    )
                )
                if (contiguous or empty) and groups_per_warp >= 8:
                    row = Int32(-32) if empty else first & Int32(0x7FFFFFFF)
                    desc = select_gather_map(
                        self.tma_desc_v, self.tma_desc_bulk_extra, first < 0
                    )
                    if prims.elect_sync():
                        for part in cutlass.range_constexpr(
                            cfg.head_dim_per_stage_kv // (128 // cfg.qkv_dtype_bytes)
                        ):
                            prims.cp_async_bulk_tensor_shared_cta_global(
                                stage_base.data_ptr(
                                    (warp * groups_per_warp + chunk * 8)
                                    * 4
                                    * inner_width
                                    + part * cfg.tile_size_kv * inner_width
                                ),
                                cute.make_ptr(cutlass.Int8, desc),
                                (dim_offset + part * inner_width, row),
                                stage_info.barrier,
                            )
                elif cutlass.const_expr(cfg.cache_uniform_sparse_quads):
                    desc = select_gather_map(tma_desc, self.tma_desc_c_rope, first < 0)
                    gather4_cached_warp(
                        stage_base.data_ptr(warp * groups_per_warp * 4 * inner_width),
                        desc,
                        dim_offset,
                        cached_page_ids,
                        stage_info.barrier,
                        cache_start=base + 6,
                        num_quads=min(8, groups_per_warp),
                        num_slices=cfg.head_dim_per_stage_kv
                        // (128 // cfg.qkv_dtype_bytes),
                        quad_stride=4 * inner_width * cfg.qkv_dtype_bytes,
                        slice_stride=cfg.tile_size_kv
                        * inner_width
                        * cfg.qkv_dtype_bytes,
                        col_stride=inner_width,
                    )
                else:
                    # Each active lane already owns a register-cached quad.
                    # Issue disjoint TMA transactions directly from those
                    # lanes, avoiding eight rounds of shuffles/election.
                    lane = cute.arch.thread_idx()[0] % 32
                    if lane >= chunk * 8 and lane < min(chunk * 8 + 8, groups_per_warp):
                        group = warp * groups_per_warp + lane
                        coords = cutlass.Array(
                            Int32, 4, space=cutlass.AddressSpace.rmem
                        )
                        for j in cutlass.range_constexpr(4):
                            raw = cached_page_ids[base + j]
                            row = raw & Int32(0x7FFFFFFF)
                            coords[j] = row if row != Int32(0x7FFFFFFF) else Int32(-1)
                        desc = select_gather_map(
                            tma_desc, self.tma_desc_c_rope, first < 0
                        )
                        for part in cutlass.range_constexpr(
                            cfg.head_dim_per_stage_kv // (128 // cfg.qkv_dtype_bytes)
                        ):
                            gather4(
                                stage_base.data_ptr(
                                    group * 4 * inner_width
                                    + part * cfg.tile_size_kv * inner_width
                                ),
                                desc,
                                dim_offset + part * inner_width,
                                coords[0],
                                coords[1],
                                coords[2],
                                coords[3],
                                stage_info.barrier,
                            )
            return

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
            active_tile_elems = active_width * cfg.tile_size_kv
            last_valid_page = cute.math.max(
                (seq_len_kv + Int32(cfg.num_tokens_per_page - 1))
                // Int32(cfg.num_tokens_per_page)
                - Int32(1),
                Int32(0),
            )
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
                                Int32(active_tile_elems) + smem_page_offset
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
        reuse_k: cutlass.Constexpr[bool] = False,
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
            stage_base = self._stage_base(stage_info)
            if cutlass.const_expr(reuse_k):
                # K waits advance independently of releases. Keep its payload
                # live until the corresponding delayed PV has consumed it.
                stage_base = self._smem_kv.subview(
                    self.kv_stage_offset(self.consumer_release_state.index)
                )
            if cutlass.const_expr(self.cfg.whole_kv_tile):
                stage_base = stage_base.subview(Int32(k_subtile_idx * 16384))
            v_leading_byte_offset = (
                Int32(0)
                if self.cfg.head_dim_per_stage_v == TCGEN05_BF16_K_BLOCK_WIDTH
                else Int32(self.cfg.tile_size_kv * 128)
            )
            if cutlass.const_expr(self.cfg.is_fp8_qkv()):
                v_leading_byte_offset = Int32(0)
            if cutlass.const_expr(self.cfg.ws_pv):
                # B is N-major: next128-byte output-feature atom is the
                # next retained128x128 panel, not the next KV row.
                v_leading_byte_offset = Int32(self.cfg.kv_smem_tile_bytes)
            desc = cprims.Tcgen05SmemDesc.build(
                stage_base,
                leading_byte_offset=v_leading_byte_offset,
                stride_byte_offset=stride_byte_offset,
                layout=qkv_smem_swizzle(self.cfg),
            )
            return desc
        else:
            k_leading_byte_offset = Int32(self.cfg.tile_size_kv * 128)
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
                        self._stage_base(stage_info).subview(
                            Int32(k_subtile_idx * 16384)
                            if self.cfg.whole_kv_tile
                            else Int32(0)
                        ),
                        leading_byte_offset=k_leading_byte_offset,
                        stride_byte_offset=Int32(
                            qkv_major_k_stride_bytes_for(self.cfg, self.cfg.rope_dim)
                        ),
                        layout=qkv_smem_swizzle_for_head_dim(
                            self.cfg, self.cfg.rope_dim
                        ),
                    )
            desc = cprims.Tcgen05SmemDesc.build(
                self._stage_base(stage_info).subview(
                    Int32(k_subtile_idx * 16384) if self.cfg.whole_kv_tile else Int32(0)
                ),
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

    @consumer_work(returns=("v_desc_0",))
    @cute.jit
    def v_desc_reused(
        self, stage_info: StageInfo, *, v_subtile_idx: cutlass.Constexpr[int]
    ):
        """Read delayed V from the unreleased K payload in the same SMEM slot."""
        return self._set_desc(
            stage_info, 1, 0, k_subtile_idx=v_subtile_idx, reuse_k=True
        )

    @consumer_work(returns=("v_desc_0",))
    @cute.jit
    def v_desc_reused_0(
        self, stage_info: StageInfo, *, v_subtile_idx: cutlass.Constexpr[int]
    ):
        """Bind V consumer zero to the selected feature slice of its retained K payload."""
        return self._set_desc(
            stage_info, 1, 0, k_subtile_idx=v_subtile_idx, reuse_k=True
        )

    @consumer_work(returns=("v_desc_1",))
    @cute.jit
    def v_desc_reused_1(
        self, stage_info: StageInfo, *, v_subtile_idx: cutlass.Constexpr[int]
    ):
        """Bind V consumer one to the selected feature slice of its retained K payload."""
        return self._set_desc(
            stage_info, 1, 1, k_subtile_idx=v_subtile_idx, reuse_k=True
        )

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
