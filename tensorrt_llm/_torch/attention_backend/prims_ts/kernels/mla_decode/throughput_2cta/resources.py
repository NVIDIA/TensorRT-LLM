# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Resource definitions for the throughput 2CTA MLA decode TS kernel.

Resource classes matching the throughput 2CTA pipeline structure:

GMEM/register resources (not pipelined)
----------------------------------------
- PageOffsetWindowResource: one warp-wide, 32-page-ID register window

SMEM resources (pipelined)
--------------------------
- SmemQResource     : TmaUmmaAsync(1 stage),   LoadTma -> Mma
- SmemKVResource    : TmaUmmaAsync(7 stages), LoadTma -> Mma

SMEM/TMEM resources (pipelined)
-------------------------------
- SmemPResource     : UmmaConsumerAsync(2 stages), SoftmaxTask -> MmaTask
- TmemSResource     : UmmaProducerAsync(2 stages), MmaTask -> SoftmaxTask
- TmemCorrResource  : Async(2 stages),              SoftmaxTask -> CorrectionTask
- TmemOResource     : UmmaProducerAsync(1 stage),   MmaTask -> CorrectionTask

GMEM (no pipeline)
------------------
- GmemOResource     : No pipeline, Correction -> GMEM
"""

from dataclasses import dataclass, field
from typing import Any, ClassVar

import cutlass
import cutlass.cute as cute
from cutlass.experimental import primitives as prims
from cutlass import Boolean, Float32, Int16, Int32, Int64
from cutlass.experimental import primitives as cprims

from cutlass.experimental.task_scheduling.resources import (
    MemoryResource,
    StageInfo,
    TaskLocalVariable,
    WorkQueue,
)
from cutlass.experimental.task_scheduling.resources import consumer_work, producer_work
from cutlass.experimental.task_scheduling.enums import WorkAttr
from ...mask import kv_tile_needs_right_mask
from ...tensor_map import transform_ragged_coords

from .config import (
    V_SMEM_K_BLOCK_TOKENS,
    V_TMA_LATENT_ELEMENTS,
    MlaDecodeConfig,
)
from ..helpers.constants import (
    BF16_OUTPUT_VECTOR_ELEMENTS,
    EPILOGUE_COLUMN_GROUP_SHIFT,
    EPILOGUE_ROW_MASK,
    EPILOGUE_THREAD_TILE_MASK,
    EPILOGUE_THREAD_TILE_THREADS,
    FP8_OUTPUT_VECTOR_ELEMENTS,
    PACKED_FP8_OUTPUT_REGS,
    TCGEN05_32B_REGS_PER_LOAD,
    TCGEN05_32B_SHAPE,
    WARP_LANE_SHIFT,
)
from ..helpers.tile_scheduler import (
    MLAStaticTileScheduler,
    MLAStaticTileSchedulerParams,
    create_mla_static_tile_scheduler,
    divmod_constexpr_power_of_two_or_fdd,
)
from ..helpers.math import (
    ceil_div,
    mma_k_step_for_qkv,
    mma_kind_for_qkv,
    add_packed_f32x2,
    fma_packed_f32x2,
    mul_packed_f32x2,
    output_dtype,
    p_desc_layout,
    p_desc_leading_byte_offset,
    p_desc_stride_byte_offset,
    qk_desc_layout,
    qk_desc_layout_for_head_dim,
    qk_desc_leading_byte_offset,
    qk_desc_leading_byte_offset_for_head_dim,
    qk_desc_stride_byte_offset,
    qk_desc_stride_byte_offset_for_head_dim,
    qkv_dtype,
    qkv_major_k_stride_bytes_for,
)
from ..helpers.ops import (
    fp8_log2_quant_scale,
    fp8_quant_scale_rcp,
    fmax_f32,
    pack_float4_to_fp8_e4m3,
)
from ..helpers.mask import MaskType, mask_visible_k_length
from ..helpers.query import (
    groups_tokens_heads_q_row_state,
    query_batch_bounds,
    runtime_query_group_has_rows,
)
from .work_partition import (
    runtime_split_kv_cap,
    runtime_split_tile_range,
)


def _install_task_local_specs(resource: object, specs: tuple[tuple, ...]) -> None:
    """Install TaskLocalVariable fields declared by resource classes."""
    for spec in specs:
        field_name, dtype, default, docs = spec[:4]
        runtime_slot_name = spec[4] if len(spec) > 4 else None
        object.__setattr__(
            resource,
            field_name,
            TaskLocalVariable(
                dtype=dtype,
                default=default,
                docs=docs,
                runtime_slot_name=runtime_slot_name,
            ),
        )


@dataclass(kw_only=True)
class HighThroughputMlaResource(MemoryResource):
    """Base class that binds captured-schedule task-local variables."""

    _task_local_specs: ClassVar[tuple[tuple, ...]] = ()

    def __post_init__(self) -> None:
        _install_task_local_specs(self, self._task_local_specs)


# =====================================================================
# WorkThrottleBarrierResource — Cluster-safe CLC scheduler pacing
# =====================================================================


@dataclass(kw_only=True)
class WorkThrottleBarrierResource(MemoryResource):
    """Pace CLC schedule token reuse against the leader CTA's active MMA task.

    The barrier has no payload.  Its producer task already runs only on CTA 0,
    so ordinary public pipeline ownership is sufficient for every stage and
    producer-tail operation.
    """

    is_barrier: cutlass.Constexpr[bool] = True


# =====================================================================
# MlaWorkQueue — Persistent tile scheduler for MLA decode
# =====================================================================


class MlaTsWorkTileInfo:
    """MLA work tile with scalar fields for TS persistent loop carry.

    Besides the scheduler coordinate, cache the per-tile K-domain values that
    hot resource paths need. Each persistent warp loop computes the K domain
    once and passes the derived indices to the page-offset, TMA, MMA, and
    softmax bodies.
    """

    @cute.jit
    def __init__(
        self,
        tile_idx,
        is_valid,
        k_len=0,
        k_tile_count=0,
        k_index_base=0,
    ):
        """Initialize the staged tile coordinate and cached K-domain metadata."""
        cluster_idx, seq_q_idx, batch_idx, split_kv_idx = tile_idx
        self._cluster_idx = cluster_idx
        self._seq_q_idx = seq_q_idx
        self._batch_idx = batch_idx
        self._split_kv_idx = split_kv_idx
        self._is_valid = Boolean(is_valid)
        self._k_len = Int32(k_len)
        self._k_tile_count = Int32(k_tile_count)
        self._k_index_base = Int32(k_index_base)

    @property
    @cute.jit
    def tile_idx(self):
        """Return the scheduler tile coordinate tuple."""
        return (
            self._cluster_idx,
            self._seq_q_idx,
            self._batch_idx,
            self._split_kv_idx,
        )

    @property
    @cute.jit
    def is_valid_tile(self):
        """Return whether this tile participates in the current launch."""
        return self._is_valid

    @property
    @cute.jit
    def k_len(self):
        """Return the runtime K length associated with this tile."""
        return self._k_len

    @property
    @cute.jit
    def k_tile_count(self):
        """Return the number of K tiles assigned to this work tile."""
        return self._k_tile_count

    @property
    @cute.jit
    def k_index_base(self):
        """Return the first K tile index owned by this work tile."""
        return self._k_index_base

    @cute.jit
    def update_from(self, other) -> None:
        """Replace this tile info with another tile info object."""
        cluster_idx, seq_q_idx, batch_idx, split_kv_idx = other.tile_idx
        self._cluster_idx = cluster_idx
        self._seq_q_idx = seq_q_idx
        self._batch_idx = batch_idx
        self._split_kv_idx = split_kv_idx
        self._is_valid = Boolean(other.is_valid_tile)
        self._k_len = Int32(other.k_len)
        self._k_tile_count = Int32(other.k_tile_count)
        self._k_index_base = Int32(other.k_index_base)

    def __extract_mlir_values__(self):
        """Extract scalar MLIR values for value-type lowering."""
        values = cutlass.extract_mlir_values(self._cluster_idx)
        values += cutlass.extract_mlir_values(self._seq_q_idx)
        values += cutlass.extract_mlir_values(self._batch_idx)
        values += cutlass.extract_mlir_values(self._split_kv_idx)
        values += cutlass.extract_mlir_values(self._is_valid)
        values += cutlass.extract_mlir_values(self._k_len)
        values += cutlass.extract_mlir_values(self._k_tile_count)
        values += cutlass.extract_mlir_values(self._k_index_base)
        return values

    def __new_from_mlir_values__(self, values):
        """Rebuild a tile info object from lowered scalar MLIR values."""
        return MlaTsWorkTileInfo(
            (
                cutlass.new_from_mlir_values(self._cluster_idx, [values[0]]),
                cutlass.new_from_mlir_values(self._seq_q_idx, [values[1]]),
                cutlass.new_from_mlir_values(self._batch_idx, [values[2]]),
                cutlass.new_from_mlir_values(self._split_kv_idx, [values[3]]),
            ),
            cutlass.new_from_mlir_values(self._is_valid, [values[4]]),
            cutlass.new_from_mlir_values(self._k_len, [values[5]]),
            cutlass.new_from_mlir_values(self._k_tile_count, [values[6]]),
            cutlass.new_from_mlir_values(self._k_index_base, [values[7]]),
        )


@dataclass(kw_only=True)
class MlaWorkQueue(WorkQueue):
    """WorkQueue that preserves MLA coordinates for static or CLC scheduling.

    Static scheduling uses ``MLAStaticTileScheduler`` directly.  BF16 CLC
    scheduling uses the public CUTLASS ``WorkQueue`` implementation, then maps
    its cluster response ``(cluster_rank, 0, linear_cluster)`` back to MLA's
    ``(cluster_rank, s_idx, b_idx, split_kv_idx)`` coordinate.  Both modes feed
    the same K-domain cache and task-local value type.

    ``tile_sched_params`` is the ``MLAStaticTileSchedulerParams`` object created
    by the kernel.
    """

    tile_sched_params: Any = None
    cache_seqs: Any = None  # cache_seqs tensor for k_tile_count
    split_kv: Any = None  # maximum split slots in the launch/workspace
    block_split_kvs: Any = None  # optional per-batch split caps
    is_var_split_kv: cutlass.Constexpr[bool] = False
    cfg: Any = None  # MlaDecodeConfig for tile sizes
    static_split_kv: cutlass.Constexpr = None
    static_seq_len_k: cutlass.Constexpr = None
    cu_seqlens_q: Any = None
    groups_tokens_heads_q_ratio: cutlass.Constexpr[int] = 1
    logical_num_heads_q: cutlass.Constexpr[int] = 128
    logical_seq_len_q: cutlass.Constexpr[int] = 1
    static_problem_shape_b: cutlass.Constexpr[int] = None
    static_problem_shape_s: cutlass.Constexpr[int] = None
    use_clc_dynamic: cutlass.Constexpr[bool] = False
    work_tile: cutlass.Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    skip_work_tile: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )

    def __init__(
        self,
        tile_sched_params,
        cache_seqs=None,
        split_kv=None,
        block_split_kvs=None,
        is_var_split_kv=False,
        cfg=None,
        static_split_kv=None,
        static_seq_len_k=None,
        cu_seqlens_q=None,
        groups_tokens_heads_q_ratio=1,
        logical_num_heads_q=128,
        logical_seq_len_q=1,
        static_problem_shape_b=None,
        static_problem_shape_s=None,
        use_clc_dynamic=False,
        tile_scheduler_config=None,
        **kwargs,
    ):
        self.use_clc_dynamic = use_clc_dynamic
        if use_clc_dynamic:
            WorkQueue.__init__(
                self,
                tile_scheduler_config=tile_scheduler_config,
                **kwargs,
            )
        else:
            # The custom static scheduler does not use TileSchedulerConfig.
            MemoryResource.__init__(self, **kwargs)
            self.tile_scheduler_config = None
        self.tile_sched_params = tile_sched_params
        self.cache_seqs = cache_seqs
        self.split_kv = split_kv
        self.block_split_kvs = block_split_kvs
        self.is_var_split_kv = is_var_split_kv
        self.cfg = cfg
        self.static_split_kv = static_split_kv
        self.static_seq_len_k = static_seq_len_k
        self.cu_seqlens_q = cu_seqlens_q
        self.groups_tokens_heads_q_ratio = groups_tokens_heads_q_ratio
        self.logical_num_heads_q = logical_num_heads_q
        self.logical_seq_len_q = logical_seq_len_q
        self.static_problem_shape_b = static_problem_shape_b
        self.static_problem_shape_s = static_problem_shape_s
        self.work_tile = TaskLocalVariable(
            dtype=MlaTsWorkTileInfo,
            default=MlaTsWorkTileInfo(
                (Int32(0), Int32(0), Int32(0), Int32(0)),
                Boolean(False),
                Int32(0),
                Int32(0),
                Int32(0),
            ),
            docs="Current MLA persistent-scheduler work tile.",
        )
        self.skip_work_tile = TaskLocalVariable(
            dtype=Boolean,
            default=Boolean(False),
            docs="Whether the current work tile should skip skippable work.",
        )

    def create_tile_scheduler(self):
        if cutlass.const_expr(self.use_clc_dynamic):
            return WorkQueue.create_tile_scheduler(self)
        return create_mla_static_tile_scheduler(
            self.tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )

    def _create_placeholder_tile_scheduler(self):
        """Create a dead structural scheduler for the shared prologue.

        Real persistent scheduling is created per task in MlaTask.  Keeping
        this placeholder independent of grid_dim avoids shared-prologue nctaid
        values being CSE'd into every warp-dispatch branch.
        """
        blk = cute.arch.block_idx()
        dummy_params = MLAStaticTileSchedulerParams(
            self.tile_sched_params.is_persistent,
            blk[0],
            blk[0],
            self.tile_sched_params.cluster_shape_mnk,
            blk[0],
        )
        return MLAStaticTileScheduler(dummy_params, blk[0], blk, blk)

    def create(self) -> None:
        """Create pipeline (if any) and the scheduler used by all TS tasks."""
        if cutlass.const_expr(self.use_clc_dynamic):
            WorkQueue.create(self)
            return
        # Call MemoryResource.create() (not WorkQueue.create) to avoid
        # the tile_scheduler_config check in the parent.
        MemoryResource.create(self)
        self.tile_scheduler = self._create_placeholder_tile_scheduler()

    @cute.jit
    def initial_work_tile_info(self):
        work_tile = self.tile_scheduler.initial_work_tile_info()
        if cutlass.const_expr(self.use_clc_dynamic):
            return self._work_tile_from_clc_tile(work_tile)
        return self.wrap_work_tile(work_tile)

    @cute.jit
    def wrap_work_tile(self, work_tile):
        return MlaTsWorkTileInfo(work_tile.tile_idx, work_tile.is_valid_tile)

    @cute.jit
    def _work_tile_from_clc_tile(self, work_tile):
        """Map one cluster-wide CLC response to MLA's four coordinates."""
        query_cluster_idx, _, split_batch_idx = work_tile.tile_idx
        params = self.tile_sched_params
        cluster_width = Int32(params.cluster_shape_mnk[0])
        s_idx = query_cluster_idx // cluster_width
        cluster_idx = query_cluster_idx % cluster_width
        split_kv_idx, b_idx = divmod_constexpr_power_of_two_or_fdd(
            split_batch_idx,
            self.static_problem_shape_b,
            params.problem_shape_b_fdd,
        )
        return self._make_work_tile_info(
            (cluster_idx, s_idx, b_idx, split_kv_idx),
            work_tile.is_valid_tile,
        )

    @cute.jit
    def _make_work_tile_info(
        self,
        tile_idx,
        is_valid,
    ):
        cfg = self.cfg
        params = self.tile_sched_params
        _, s_idx, b_idx, split_kv_idx = tile_idx
        safe_b_idx = cute.math.min(b_idx, params.problem_shape_b - Int32(1))
        if cutlass.const_expr(self.static_seq_len_k is not None):
            K = Int32(self.static_seq_len_k)
        else:
            K = Int32(self.cache_seqs[safe_b_idx])

        # The launch uses the largest grouped-Q count in the batch. Shorter
        # variable-Q requests can therefore receive a whole group containing
        # no real query rows. Give that group an empty K domain so every task
        # skips it before any Q/K/V or partial-output access.
        _, q_len = query_batch_bounds(
            self.cu_seqlens_q,
            safe_b_idx,
            self.logical_seq_len_q,
        )
        q_group_has_rows = runtime_query_group_has_rows(
            s_idx,
            self.groups_tokens_heads_q_ratio,
            self.logical_seq_len_q,
            self.cu_seqlens_q,
            safe_b_idx,
        )
        if cutlass.const_expr(
            cfg.mask_type == MaskType.CAUSAL.value and self.logical_seq_len_q > 1
        ):
            # Split partitioning owns the mask-visible CTA domain. The last
            # groups_tokens_heads_q row provides the group maximum; grouped
            # causal softmax narrows earlier rows without changing geometry.
            _, _, logical_q_idx, _, _ = groups_tokens_heads_q_row_state(
                Int32(self.logical_num_heads_q * self.groups_tokens_heads_q_ratio - 1),
                s_idx,
                self.groups_tokens_heads_q_ratio,
                self.logical_num_heads_q,
                self.logical_seq_len_q,
                self.cu_seqlens_q,
                safe_b_idx,
            )
            K = mask_visible_k_length(cfg.mask_type, K, logical_q_idx, q_len)
        K = K if q_group_has_rows else Int32(0)
        k_tile_total = (K + Int32(cfg.mma_qk_tiler[1] - 1)) // Int32(
            cfg.mma_qk_tiler[1]
        )
        if cutlass.const_expr(self.static_split_kv == 1):
            # Single-split fixed profiles process the whole K domain in one CTA.
            # Avoid the runtime split-KV ceil-div/min/max sequence in every
            # persistent task branch.
            k_index_base = Int32(0)
            k_tile_count = k_tile_total
        else:
            # Grid/workspace geometry stays at ``split_kv`` while each batch's
            # optional cap and valid K select the configured-span nonempty prefix.
            if cutlass.const_expr(
                self.static_split_kv is not None and not self.is_var_split_kv
            ):
                split_kv_cap = Int32(self.static_split_kv)
            else:
                split_kv_cap = runtime_split_kv_cap(
                    self.split_kv,
                    self.is_var_split_kv,
                    self.block_split_kvs,
                    safe_b_idx,
                )
            k_index_base, k_tile_count = runtime_split_tile_range(
                k_tile_total,
                split_kv_cap,
                split_kv_idx,
            )
        return MlaTsWorkTileInfo(tile_idx, is_valid, K, k_tile_count, k_index_base)

    @cute.jit
    def k_tile_count_for_tile(self, tile_idx):
        """Return the dynamic K-loop bound required by stock Task."""

        return self._make_work_tile_info(tile_idx, Boolean(True)).k_tile_count

    @cute.jit
    def skip_work_tile_if(self, work_tile):
        """Skip zero-K CLC tiles while retaining WorkQueue bookkeeping."""

        return work_tile.k_tile_count <= Int32(0)

    @cute.jit
    def _work_tile_from_linear_idx(self, current_work_linear_idx):
        params = self.tile_sched_params
        current_work_cluster_batch, cluster_idx = (
            current_work_linear_idx // params.cluster_shape_mnk[0],
            current_work_linear_idx % params.cluster_shape_mnk[0],
        )
        current_work_s_batch, s_idx = divmod_constexpr_power_of_two_or_fdd(
            current_work_cluster_batch,
            self.static_problem_shape_s,
            params.problem_shape_s_fdd,
        )
        current_work_b_batch, b_idx = divmod_constexpr_power_of_two_or_fdd(
            current_work_s_batch,
            self.static_problem_shape_b,
            params.problem_shape_b_fdd,
        )
        if cutlass.const_expr(self.static_split_kv == 1):
            split_kv_idx = Int32(0)
            num_blocks = (
                params.cluster_shape_mnk[0]
                * params.problem_shape_s
                * params.problem_shape_b
            )
        else:
            _, split_kv_idx = divmod(
                current_work_b_batch,
                params.split_kv_fdd,
            )
            num_blocks = (
                params.cluster_shape_mnk[0]
                * params.problem_shape_s
                * params.problem_shape_b
                * params.split_kv
            )
        return self._make_work_tile_info(
            (cluster_idx, s_idx, b_idx, split_kv_idx),
            current_work_linear_idx < num_blocks,
        )

    @cute.jit
    def _work_tile_from_block_idx(self, block_idx):
        params = self.tile_sched_params
        s_idx, b_idx = divmod_constexpr_power_of_two_or_fdd(
            block_idx[1],
            self.static_problem_shape_b,
            params.problem_shape_b_fdd,
        )
        return self._make_work_tile_info(
            (block_idx[0], s_idx, b_idx, block_idx[2]),
            Boolean(True),
        )

    @cute.jit
    def _linear_idx_from_tile(self, tile_idx):
        params = self.tile_sched_params
        cluster_idx, s_idx, b_idx, split_kv_idx = tile_idx
        return (
            ((split_kv_idx * params.problem_shape_b + b_idx) * params.problem_shape_s)
            + s_idx
        ) * params.cluster_shape_mnk[0] + cluster_idx

    @consumer_work(work_attrs=WorkAttr.AUXILIARY, returns=(work_tile, skip_work_tile))
    @cute.jit
    def init_work_tile(self, stage_info: StageInfo):
        """Seed captured schedules from the current custom MLA work tile."""
        return stage_info.work_tile, Boolean(False)

    @consumer_work(returns=work_tile)
    @cute.jit
    def advance_tile(self, stage_info: StageInfo):
        """Advance CLC through its response pipeline; static is task-owned."""
        if cutlass.const_expr(self.use_clc_dynamic):
            # Decode through the public scheduler/config surfaces so the
            # task-local value keeps its MLA type and cached runtime K-domain
            # fields without depending on WorkQueue's private response helpers.
            assert self.tile_scheduler_config is not None
            assert self.tile_scheduler_config.response_ptr is not None
            assert self.tile_scheduler is not None
            assert self.pipeline_config is not None
            stage_response_ptr = self.tile_scheduler_config.response_ptr
            if cutlass.const_expr(self.pipeline_config.num_stages > 1):
                stage_response_ptr = stage_response_ptr + stage_info.stage_idx
            work_tile = self.tile_scheduler.work_tile_info_from_clc_response(
                stage_response_ptr
            )
            return self._work_tile_from_clc_tile(work_tile)
        return stage_info.work_tile


# =====================================================================
# PageOffsetWindowResource — TMA-warp register page-table window
# =====================================================================


@dataclass(kw_only=True)
class PageOffsetWindowResource(HighThroughputMlaResource):
    """GMEM page table cached as one register per lane of the TMA warp."""

    page_offsets: Any = None  # GMEM page-offset tensor
    cfg: cutlass.Constexpr = field(default_factory=MlaDecodeConfig)
    cached_k_pages: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    cached_v_pages: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    cached_next_v_pages: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    cached_window_page: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    _task_local_specs: ClassVar[tuple[tuple, ...]] = (
        (
            "cached_k_pages",
            cutlass.Array,
            None,
            "Cached page ids for the current K tile.",
        ),
        (
            "cached_v_pages",
            cutlass.Array,
            None,
            "Cached page ids for the delayed V tile.",
        ),
        (
            "cached_next_v_pages",
            cutlass.Array,
            None,
            "Cached page ids for the next delayed V tile.",
        ),
        (
            "cached_window_page",
            Int32,
            Int32(0),
            "Per-lane page id retained across one 32-entry page-table window.",
        ),
    )

    @consumer_work(
        work_attrs=WorkAttr.AUXILIARY,
        returns=(
            cached_k_pages,
            cached_v_pages,
            cached_next_v_pages,
            cached_window_page,
        ),
    )
    @cute.jit
    def init_read_state(self, stage_info: StageInfo):
        """Create cached page-index arrays used by staged KV TMA loads."""
        del stage_info
        cfg = self.cfg
        return (
            cutlass.Array(
                Int32,
                cfg.pages_per_k_cta,
                space=cutlass.AddressSpace.rmem,
            ),
            cutlass.Array(
                Int32,
                cfg.pages_per_v_tile,
                space=cutlass.AddressSpace.rmem,
            ),
            cutlass.Array(
                Int32,
                cfg.pages_per_v_tile,
                space=cutlass.AddressSpace.rmem,
            ),
            Int32(0),
        )

    @consumer_work(
        returns=(
            cached_k_pages,
            cached_v_pages,
            cached_next_v_pages,
            cached_window_page,
        )
    )
    @cute.jit
    def read_page_offset_window(
        self,
        stage_info: StageInfo,
        *,
        cached_k_pages,
        cached_v_pages,
        cached_next_v_pages,
        cached_window_page,
        init_v_cache: cutlass.Constexpr[bool] = False,
    ):
        """Load and reuse one warp-wide 32-page-ID window.

        The TMA warp owns one page ID per lane. It refreshes the register
        window after ``32 / pages_per_k_tile`` logical K tiles, when the next
        group of 32 page-table entries is needed, then uses indexed warp
        shuffles to assemble the IDs for the current K/V tile. The refresh
        cadence is derived entirely from page and tile geometry.
        """
        cfg = self.cfg
        work_tile = stage_info.work_tile
        blk_coord = work_tile.tile_idx
        local_k_index = Int32(stage_info.loop_offset)
        global_k_index = work_tile.k_index_base + local_k_index
        pages_per_k_tile = cutlass.const_expr(cfg.pages_per_k_tile)
        page_window_tiles = cutlass.const_expr(32 // pages_per_k_tile)
        page_window_mask = cutlass.const_expr(page_window_tiles - 1)
        page_offsets_batch = self.page_offsets[None, blk_coord[2]]
        lane_idx = cute.arch.thread_idx()[0] & Int32(31)

        if (local_k_index & Int32(page_window_mask)) == Int32(0):
            logical_page_idx = global_k_index * Int32(pages_per_k_tile) + lane_idx
            bounded_page_idx = cute.math.min(
                logical_page_idx, Int32(page_offsets_batch.shape[0] - 1)
            )
            cached_window_page = Int32(page_offsets_batch[bounded_page_idx])

        page_lane_base = (local_k_index & Int32(page_window_mask)) * Int32(
            pages_per_k_tile
        )
        cta_v = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        cached_k = cached_k_pages
        cached_v = cached_v_pages
        cached_next_v = cached_next_v_pages

        if cutlass.const_expr(init_v_cache):
            for pk in cutlass.range_constexpr(cfg.pages_per_v_tile):
                cached_v[pk] = Int32(0)
        else:
            for pk in cutlass.range_constexpr(cfg.pages_per_v_tile):
                cached_v[pk] = cached_next_v[pk]

        for pk in cutlass.range_constexpr(cfg.pages_per_k_cta):
            source_lane = (
                page_lane_base + cta_v * Int32(cfg.pages_per_k_cta) + Int32(pk)
                if cfg.pages_per_k_tile > 1
                else page_lane_base
            )
            cached_k[pk] = Int32(
                cprims.shfl_sync(
                    thread_mask=0xFFFFFFFF,
                    val=cached_window_page,
                    offset=source_lane,
                    mask_and_clamp=0x1F,
                    kind=cprims.Shfl.IDX,
                )
            )

        for pk in cutlass.range_constexpr(cfg.pages_per_v_tile):
            source_lane = (
                page_lane_base
                if cfg.pages_per_v_tile == 1
                else page_lane_base + Int32(pk)
            )
            cached_next_v[pk] = Int32(
                cprims.shfl_sync(
                    thread_mask=0xFFFFFFFF,
                    val=cached_window_page,
                    offset=source_lane,
                    mask_and_clamp=0x1F,
                    kind=cprims.Shfl.IDX,
                )
            )
        return cached_k, cached_v, cached_next_v, cached_window_page

    @consumer_work(
        work_attrs=WorkAttr.AUXILIARY,
        returns=(cached_k_pages, cached_v_pages, cached_next_v_pages),
    )
    @cute.jit
    def forward_page_ids(
        self,
        stage_info: StageInfo,
        *,
        cached_k_pages,
        cached_v_pages,
        cached_next_v_pages,
    ):
        """Forward the latest delayed-V page IDs after the domain loop."""
        del stage_info
        return cached_k_pages, cached_v_pages, cached_next_v_pages


# =====================================================================
# SmemQResource — Q SMEM buffer with TmaUmmaAsync pipeline
# =====================================================================


@dataclass(kw_only=True)
class SmemQResource(HighThroughputMlaResource):
    """SMEM Q buffer (latent + rope).  Producer: LoadTma (TMA).  Consumer: Mma.

    Pipeline: TmaUmmaAsync, 1 stage.
    Q is loaded once (LoopFirstIter) and released once (LoopLastIter).
    """

    smem_q_latent: Any = None  # SMEM Q latent array
    smem_q_rope: Any = None  # SMEM Q rope array
    tma_desc_q_latent: Any = None
    tma_desc_q_rope: Any = None
    cu_seqlens_q: Any = None
    groups_tokens_heads_q_ratio: cutlass.Constexpr[int] = 1
    logical_num_heads_q: cutlass.Constexpr[int] = 128
    logical_seq_len_q: cutlass.Constexpr[int] = 1
    cfg: cutlass.Constexpr = field(default_factory=MlaDecodeConfig)
    cta_rank: Any = field(init=False, default=None)
    is_leader: Any = field(init=False, default=None)

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_load_state(self, stage_info: StageInfo) -> None:
        """Initialize CTA-local Q-load state for the high-throughput path."""
        del stage_info
        self.cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        self.is_leader = self.cta_rank == 0

    @cute.jit
    def _query_tma_coords(
        self,
        dim_coord,
        local_flat_query_row,
        batch_idx,
        storage_flat_query_row,
        query_row_extent,
    ):
        """Return dense or ragged TMA coordinates for one Q dimension slice."""
        if cutlass.const_expr(self.cu_seqlens_q is not None):
            return transform_ragged_coords(
                (dim_coord, storage_flat_query_row),
                ragged_dim_idx=1,
                ragged_box_size=self.cfg.mma_qk_tiler[0] // self.cfg.num_mma_ctas,
                ragged_extent=query_row_extent,
            )
        return dim_coord, local_flat_query_row, batch_idx

    @producer_work
    @cute.jit
    def tma_load(self, stage_info: StageInfo) -> None:
        """TMA load Q latent + Q rope into SMEM (all sub-tiles in one commit)."""
        cfg = self.cfg
        work_tile = stage_info.work_tile
        blk_coord = work_tile.tile_idx

        cta_v = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        coord_m = cta_v * Int32(cfg.mma_qk_tiler[0] // cfg.num_mma_ctas)
        effective_num_heads_q = Int32(
            self.logical_num_heads_q * self.groups_tokens_heads_q_ratio
        )
        flat_query_row = Int32(blk_coord[1]) * effective_num_heads_q + coord_m
        batch_idx = Int32(blk_coord[2])
        storage_flat_query_row = flat_query_row
        query_row_extent = Int32(cfg.mma_qk_tiler[0] // cfg.num_mma_ctas)
        if cutlass.const_expr(self.cu_seqlens_q is not None):
            q_start, q_len = query_batch_bounds(
                self.cu_seqlens_q,
                batch_idx,
                self.logical_seq_len_q,
            )
            storage_flat_query_row = (
                q_start * Int32(self.logical_num_heads_q) + flat_query_row
            )
            query_row_extent = q_len * Int32(self.logical_num_heads_q) - flat_query_row
        mask_q = Int16(Int32(1) << cta_v)
        q_mbar_arr = cutlass.Array(stage_info.barrier.data_ptr(), dtype=Int64)

        if prims.elect_sync():
            # Load Q latent sub-tiles
            q_latent_stage_elems = cutlass.const_expr(
                cfg.mma_qk_tiler[0] // cfg.num_mma_ctas * cfg.mma_qk_tiler_k
            )
            for i in cutlass.range(cfg.iterations_qk_latent):
                coord_kl = cutlass.Int32(i * cfg.mma_qk_tiler_k)
                q_latent_coords = self._query_tma_coords(
                    coord_kl,
                    flat_query_row,
                    batch_idx,
                    storage_flat_query_row,
                    query_row_extent,
                )
                q_smem_arr = cutlass.Array(
                    self.smem_q_latent.data_ptr(i * q_latent_stage_elems),
                    dtype=qkv_dtype(cfg),
                )
                prims.cp_async_bulk_tensor_shared_cluster_global(
                    q_smem_arr,
                    self.tma_desc_q_latent,
                    q_latent_coords,
                    q_mbar_arr,
                    [],
                    multicast_mask=mask_q,
                    group=prims.CTAGroup.CTA_2,
                )
            # Load Q rope sub-tiles
            for i in cutlass.range(cfg.iterations_qk_rope):
                coord_kr = cutlass.Int32(i * cfg.mma_qk_tiler_k)
                q_rope_coords = self._query_tma_coords(
                    coord_kr,
                    flat_query_row,
                    batch_idx,
                    storage_flat_query_row,
                    query_row_extent,
                )
                qr_smem_arr = cutlass.Array(
                    self.smem_q_rope.data_ptr(), dtype=qkv_dtype(cfg)
                )
                prims.cp_async_bulk_tensor_shared_cluster_global(
                    qr_smem_arr,
                    self.tma_desc_q_rope,
                    q_rope_coords,
                    q_mbar_arr,
                    [],
                    multicast_mask=mask_q,
                    group=prims.CTAGroup.CTA_2,
                )

    @consumer_work
    @cute.jit
    def q_desc(self, stage_info: StageInfo) -> None:
        """Schedule marker after the Q SMEM stage is waited."""
        del stage_info


# =====================================================================
# SmemKVResource — K/V SMEM buffer with TmaUmmaAsync pipeline
# =====================================================================


@dataclass(kw_only=True)
class SmemKVResource(HighThroughputMlaResource):
    """SMEM K/V buffer.  Producer: LoadTma (TMA).  Consumer: Mma.

    Pipeline: TmaUmmaAsync, 7 stages.
    Per logical k-tile: K sub-tiles are loaded before delayed V sub-tiles.
    """

    smem_kv: Any = None
    tma_desc_c_latent: Any = None
    tma_desc_c_rope: Any = None
    tma_desc_c_transpose: Any = None
    cfg: cutlass.Constexpr = field(default_factory=MlaDecodeConfig)
    cta_rank: Any = field(init=False, default=None)
    is_leader: Any = field(init=False, default=None)
    desc_k_base: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    desc_v_base: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    _task_local_specs: ClassVar[tuple[tuple, ...]] = (
        ("desc_k_base", Int64, Int64(0), "SMEM descriptor for staged K."),
        ("desc_v_base", Int64, Int64(0), "SMEM descriptor for staged V."),
    )

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_load_state(self, stage_info: StageInfo) -> None:
        """Initialize CTA-local KV-load descriptors and leader state."""
        del stage_info
        self.cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        self.is_leader = self.cta_rank == 0

    @producer_work
    @cute.jit
    def tma_load(
        self,
        stage_info: StageInfo,
        *,
        cached_k_pages,
        cached_v_pages,
        cached_next_v_pages,
        is_v: cutlass.Constexpr[bool],
        subtile_idx: cutlass.Constexpr[int],
        use_next_v_pages: cutlass.Constexpr[bool] = False,
    ) -> None:
        """TMA load one K or delayed-V sub-tile into the shared KV ring."""
        cfg = self.cfg
        stage_idx = stage_info.stage_idx

        kc_page_smem_elems = cfg.kc_page_tile_size * cfg.mma_qk_tiler_k
        kv_mbar_arr = cutlass.Array(stage_info.barrier.data_ptr(), dtype=Int64)
        kv_stage_stride_elems = cutlass.const_expr(cfg.smem_k_stage_elems)

        pages_per_k_cta = cfg.pages_per_k_cta

        cta_v = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        coord_n_k = (cta_v * Int32(cfg.mma_qk_tiler[1] // cfg.num_mma_ctas)) % Int32(
            cfg.page_size
        )
        mask_k = Int16(Int32(1) << cta_v)

        is_v_subtile = is_v
        k_call = subtile_idx

        if cutlass.const_expr(
            not is_v_subtile and k_call < cfg.iterations_qk_latent_stages
        ):
            cached_k = cached_k_pages
            k_subtile_smem_elems = cutlass.const_expr(
                cfg.mma_qk_tiler[1] // cfg.num_mma_ctas * cfg.mma_qk_tiler_k
            )
            for stage_subtile_idx in cutlass.range_constexpr(cfg.kv_subtiles_per_stage):
                logical_k_call = k_call * cfg.kv_subtiles_per_stage + stage_subtile_idx
                coord_kcl = cutlass.Int32(logical_k_call * cfg.mma_qk_tiler_k)
                for pk in cutlass.range_constexpr(pages_per_k_cta):
                    if prims.elect_sync():
                        kcl_smem = cutlass.Array(
                            self.smem_kv.data_ptr(
                                stage_idx * kv_stage_stride_elems
                                + stage_subtile_idx * k_subtile_smem_elems
                                + pk * kc_page_smem_elems
                            ),
                            dtype=qkv_dtype(cfg),
                        )
                        prims.cp_async_bulk_tensor_shared_cluster_global(
                            kcl_smem,
                            self.tma_desc_c_latent,
                            (coord_kcl, coord_n_k, cutlass.Int32(cached_k[pk])),
                            kv_mbar_arr,
                            [],
                            multicast_mask=mask_k,
                            group=prims.CTAGroup.CTA_2,
                        )

        elif cutlass.const_expr(not is_v_subtile and k_call < cfg.iterations_qk_stages):
            rope_idx = k_call - cfg.iterations_qk_latent_stages
            cached_k = cached_k_pages

            coord_kcr = cutlass.Int32(rope_idx * cfg.mma_qk_rope_tiler[2])
            rope_page_smem_elems = cfg.kc_page_tile_size * cfg.mma_qk_rope_tiler[2]
            rope_tile_smem_elems = (
                cfg.mma_qk_tiler[1] // cfg.num_mma_ctas * cfg.mma_qk_rope_tiler[2]
            )
            for pk in cutlass.range_constexpr(pages_per_k_cta):
                if prims.elect_sync():
                    kcr_smem = cutlass.Array(
                        self.smem_kv.data_ptr(
                            stage_idx * kv_stage_stride_elems
                            + pk * rope_page_smem_elems
                        ),
                        dtype=qkv_dtype(cfg),
                    )
                    prims.cp_async_bulk_tensor_shared_cluster_global(
                        kcr_smem,
                        self.tma_desc_c_rope,
                        (coord_kcr, coord_n_k, cutlass.Int32(cached_k[pk])),
                        kv_mbar_arr,
                        [],
                        multicast_mask=mask_k,
                        group=prims.CTAGroup.CTA_2,
                    )
                    if cutlass.const_expr(cfg.kv_subtiles_per_stage > 1):
                        # The BF16 stage contains two legal K64 transfers.
                        # Duplicate the final K64 RoPE slice into the unused
                        # half so its mbarrier transaction count matches the
                        # common K128 stage contract.
                        kcr_smem_dup = cutlass.Array(
                            self.smem_kv.data_ptr(
                                stage_idx * kv_stage_stride_elems
                                + rope_tile_smem_elems
                                + pk * rope_page_smem_elems
                            ),
                            dtype=qkv_dtype(cfg),
                        )
                        prims.cp_async_bulk_tensor_shared_cluster_global(
                            kcr_smem_dup,
                            self.tma_desc_c_rope,
                            (coord_kcr, coord_n_k, cutlass.Int32(cached_k[pk])),
                            kv_mbar_arr,
                            [],
                            multicast_mask=mask_k,
                            group=prims.CTAGroup.CTA_2,
                        )

        else:
            pv_n_per_cta = cutlass.const_expr(cfg.mma_pv_tiler[1] // cfg.num_mma_ctas)
            coord_n_v = cta_v * pv_n_per_cta
            mask_v = Int16(Int32(1) << cta_v)
            v_tma_copy_smem_elems = cfg.v_tma_token_count * V_TMA_LATENT_ELEMENTS
            v_subtile_smem_elems = cutlass.const_expr(
                cfg.mma_pv_tiler[1] // cfg.num_mma_ctas * cfg.mma_pv_tiler[2]
            )

            pages_per_v_subtile = cfg.pages_per_v_subtile
            cached_v = (
                cached_next_v_pages
                if cutlass.const_expr(use_next_v_pages)
                else cached_v_pages
            )

            # A physical V stage contains two adjacent K32 slices for one
            # D256 output panel. Across four stages this is the same
            # head-dimension-128/token-partition-2 decomposition used by the
            # 2CTA BF16 schedule: (D0,K0:64), (D0,K64:128), then D256.
            pv_j = subtile_idx // cfg.kv_subtiles_per_stage
            token_partition = subtile_idx % cfg.kv_subtiles_per_stage
            for stage_subtile_idx in cutlass.range_constexpr(cfg.kv_subtiles_per_stage):
                pv_i = token_partition * cfg.kv_subtiles_per_stage + stage_subtile_idx
                coord_k_v = cutlass.Int32((pv_i * cfg.mma_pv_tiler[2]) % cfg.page_size)
                coord_nj = coord_n_v + cutlass.Int32(pv_j * cfg.mma_pv_tiler[1])

                for pk in cutlass.range_constexpr(pages_per_v_subtile):
                    k_idx_i = cached_v[
                        pk + pv_i // cfg.v_subtiles_per_page * pages_per_v_subtile
                    ]
                    if prims.elect_sync():
                        # Keep both 64-wide V panels contiguous within each
                        # K32 slice; the next K32 slice follows the complete
                        # first slice in this physical stage.
                        stage_subtile_base = stage_subtile_idx * v_subtile_smem_elems
                        v_page_offset = pk * v_tma_copy_smem_elems
                        v_second_panel_offset = (
                            pages_per_v_subtile * v_tma_copy_smem_elems + v_page_offset
                        )
                        v_smem_0 = cutlass.Array(
                            self.smem_kv.data_ptr(
                                stage_idx * kv_stage_stride_elems
                                + stage_subtile_base
                                + v_page_offset
                            ),
                            dtype=qkv_dtype(cfg),
                        )
                        prims.cp_async_bulk_tensor_shared_cluster_global(
                            v_smem_0,
                            self.tma_desc_c_transpose,
                            (coord_nj, coord_k_v, k_idx_i),
                            kv_mbar_arr,
                            [],
                            multicast_mask=mask_v,
                            group=prims.CTAGroup.CTA_2,
                        )
                        v_smem_1 = cutlass.Array(
                            self.smem_kv.data_ptr(
                                stage_idx * kv_stage_stride_elems
                                + stage_subtile_base
                                + v_second_panel_offset
                            ),
                            dtype=qkv_dtype(cfg),
                        )
                        prims.cp_async_bulk_tensor_shared_cluster_global(
                            v_smem_1,
                            self.tma_desc_c_transpose,
                            (
                                coord_nj + V_TMA_LATENT_ELEMENTS,
                                coord_k_v,
                                k_idx_i,
                            ),
                            kv_mbar_arr,
                            [],
                            multicast_mask=mask_v,
                            group=prims.CTAGroup.CTA_2,
                        )

    @consumer_work(returns=desc_k_base)
    @cute.jit
    def k_desc(self, stage_info: StageInfo, *, k_subtile_idx: cutlass.Constexpr[int]):
        """Build the SMEM descriptor consumed by QK MMA."""
        cfg = self.cfg
        stage_idx = stage_info.stage_idx

        kc_copy_elems = cutlass.const_expr(cfg.smem_k_stage_elems)
        tile_rows = cutlass.const_expr(cfg.mma_qk_tiler[1] // cfg.num_mma_ctas)
        leading_byte_offset = cutlass.const_expr(qk_desc_leading_byte_offset(cfg))
        stride_byte_offset = cutlass.const_expr(qk_desc_stride_byte_offset(cfg))
        layout = cutlass.const_expr(qk_desc_layout(cfg))
        if cutlass.const_expr(
            cfg.is_fp8_qkv()
            and k_subtile_idx >= cfg.iterations_qk_latent
            and k_subtile_idx < cfg.iterations_qk
        ):
            leading_byte_offset = cutlass.const_expr(
                qk_desc_leading_byte_offset_for_head_dim(
                    cfg, tile_rows, cfg.mma_qk_rope_tiler[2]
                )
            )
            stride_byte_offset = cutlass.const_expr(
                qk_desc_stride_byte_offset_for_head_dim(cfg, cfg.mma_qk_rope_tiler[2])
            )
            layout = cutlass.const_expr(
                qk_desc_layout_for_head_dim(cfg, cfg.mma_qk_rope_tiler[2])
            )

        sk_ptr = self.smem_kv.data_ptr(stage_idx * kc_copy_elems)
        desc = Int64(
            cprims.Tcgen05SmemDesc.build(
                start_address=sk_ptr.toint(Int32),
                leading_byte_offset=leading_byte_offset,
                stride_byte_offset=stride_byte_offset,
                layout=layout,
            )
        )
        return desc

    @consumer_work(returns=desc_v_base)
    @cute.jit
    def v_desc(self, stage_info: StageInfo, *, v_subtile_idx: cutlass.Constexpr[int]):
        """Build the SMEM descriptor consumed by PV MMA."""
        del v_subtile_idx
        cfg = self.cfg
        stage_idx = stage_info.stage_idx
        svc_copy_elems = cutlass.const_expr(cfg.smem_k_stage_elems)
        leading_byte_offset = cutlass.const_expr(4096)
        stride_byte_offset = cutlass.const_expr(1024)
        layout = cutlass.const_expr(2)
        if cutlass.const_expr(cfg.is_fp8_qkv()):
            leading_byte_offset = cutlass.const_expr(
                V_SMEM_K_BLOCK_TOKENS * V_TMA_LATENT_ELEMENTS * cfg.qkv_dtype_bytes
            )
            stride_byte_offset = cutlass.const_expr(
                qkv_major_k_stride_bytes_for(cfg, cfg.mma_pv_tiler[2])
            )
            layout = cutlass.const_expr(
                qk_desc_layout_for_head_dim(cfg, cfg.mma_pv_tiler[2])
            )

        svc_ptr = self.smem_kv.data_ptr(stage_idx * svc_copy_elems)
        desc = Int64(
            cprims.Tcgen05SmemDesc.build(
                start_address=svc_ptr.toint(Int32),
                leading_byte_offset=leading_byte_offset,
                stride_byte_offset=stride_byte_offset,
                layout=layout,
            )
        )
        return desc


# =====================================================================
# SmemKResource — FP8 K SMEM buffer with one pipeline stage per K tile
# =====================================================================


@dataclass(kw_only=True)
class SmemKResource(HighThroughputMlaResource):
    """SMEM K buffer for the FP8 split-MMA path.

    Producer: LoadTma. Consumer: MmaQkTask. A single producer stage contains
    all latent K sub-tiles plus the RoPE K sub-tile for one logical K tile.
    """

    smem_k: Any = None
    page_offsets: Any = None
    tma_desc_c_latent: Any = None
    tma_desc_c_rope: Any = None
    logical_seq_len_q: cutlass.Constexpr[int] = 1
    cfg: cutlass.Constexpr = field(default_factory=MlaDecodeConfig)
    desc_k_base: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    _task_local_specs: ClassVar[tuple[tuple, ...]] = (
        ("desc_k_base", Int64, Int64(0), "SMEM descriptor for staged K."),
    )

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_load_state(self, stage_info: StageInfo) -> None:
        """Initialize per-warp state used by K TMA loads."""
        del stage_info

    @producer_work
    @cute.jit
    def tma_load_direct(self, stage_info: StageInfo) -> None:
        """TMA load one full K tile using page offsets read directly from GMEM."""
        cfg = self.cfg
        stage_idx = stage_info.stage_idx
        stage_base = stage_idx * cfg.smem_k_stage_elems
        work_tile = stage_info.work_tile
        blk_coord = work_tile.tile_idx
        k_index = work_tile.k_index_base + Int32(stage_info.loop_offset)

        page_row_idx = blk_coord[2]
        page_offsets_batch = self.page_offsets[None, page_row_idx]
        kv_mbar_arr = cutlass.Array(stage_info.barrier.data_ptr(), dtype=Int64)
        pages_per_k_cta = cfg.pages_per_k_cta
        kc_page_smem_elems = cfg.kc_page_tile_size * cfg.mma_qk_tiler_k
        cta_v = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        coord_n_k = (cta_v * Int32(cfg.mma_qk_tiler[1] // cfg.num_mma_ctas)) % Int32(
            cfg.page_size
        )
        mask_k = Int16(Int32(1) << cta_v)
        k_latent_subtile_elems = cutlass.const_expr(
            cfg.mma_qk_tiler[1] // cfg.num_mma_ctas * cfg.mma_qk_tiler_k
        )
        # Use page 0 for masked fragments beyond a compact page table. A direct
        # bounds predicate avoids carrying runtime K-length division or a full
        # page-ID register array through this producer warp.

        for k_call in cutlass.range_constexpr(cfg.iterations_qk_latent):
            coord_kcl = cutlass.Int32(k_call * cfg.mma_qk_tiler_k)
            subtile_base = stage_base + k_call * k_latent_subtile_elems
            for pk in cutlass.range_constexpr(pages_per_k_cta):
                logical_page_idx = (
                    k_index
                    if cfg.pages_per_k_tile == 1
                    else (k_index * Int32(cfg.num_mma_ctas) + cta_v)
                    * Int32(pages_per_k_cta)
                    + Int32(pk)
                )
                page_idx = Int32(0)
                if cute.elem_less(logical_page_idx, page_offsets_batch.shape[0]):
                    page_idx = page_offsets_batch[logical_page_idx]
                if prims.elect_sync():
                    kcl_smem = cutlass.Array(
                        self.smem_k.data_ptr(subtile_base + pk * kc_page_smem_elems),
                        dtype=qkv_dtype(cfg),
                    )
                    prims.cp_async_bulk_tensor_shared_cluster_global(
                        kcl_smem,
                        self.tma_desc_c_latent,
                        (coord_kcl, coord_n_k, cutlass.Int32(page_idx)),
                        kv_mbar_arr,
                        [],
                        multicast_mask=mask_k,
                        group=prims.CTAGroup.CTA_2,
                    )

        rope_stage_base = stage_base + k_latent_subtile_elems * cfg.iterations_qk_latent
        rope_page_smem_elems = cfg.kc_page_tile_size * cfg.mma_qk_rope_tiler[2]
        k_rope_subtile_elems = cutlass.const_expr(
            cfg.mma_qk_rope_tiler[1] // cfg.num_mma_ctas * cfg.mma_qk_rope_tiler[2]
        )
        for rope_idx in cutlass.range_constexpr(cfg.iterations_qk_rope):
            coord_kcr = cutlass.Int32(rope_idx * cfg.mma_qk_rope_tiler[2])
            subtile_base = rope_stage_base + rope_idx * k_rope_subtile_elems
            for pk in cutlass.range_constexpr(pages_per_k_cta):
                logical_page_idx = (
                    k_index
                    if cfg.pages_per_k_tile == 1
                    else (k_index * Int32(cfg.num_mma_ctas) + cta_v)
                    * Int32(pages_per_k_cta)
                    + Int32(pk)
                )
                page_idx = Int32(0)
                if cute.elem_less(logical_page_idx, page_offsets_batch.shape[0]):
                    page_idx = page_offsets_batch[logical_page_idx]
                if prims.elect_sync():
                    kcr_smem = cutlass.Array(
                        self.smem_k.data_ptr(subtile_base + pk * rope_page_smem_elems),
                        dtype=qkv_dtype(cfg),
                    )
                    prims.cp_async_bulk_tensor_shared_cluster_global(
                        kcr_smem,
                        self.tma_desc_c_rope,
                        (coord_kcr, coord_n_k, cutlass.Int32(page_idx)),
                        kv_mbar_arr,
                        [],
                        multicast_mask=mask_k,
                        group=prims.CTAGroup.CTA_2,
                    )

    @consumer_work(returns=desc_k_base)
    @cute.jit
    def k_desc(self, stage_info: StageInfo, *, k_subtile_idx: cutlass.Constexpr[int]):
        """Build the K SMEM descriptor for the current QK sub-MMA."""
        cfg = self.cfg
        stage_idx = stage_info.stage_idx

        k_latent_subtile_elems = cutlass.const_expr(
            cfg.mma_qk_tiler[1] // cfg.num_mma_ctas * cfg.mma_qk_tiler_k
        )
        k_rope_subtile_elems = cutlass.const_expr(
            cfg.mma_qk_rope_tiler[1] // cfg.num_mma_ctas * cfg.mma_qk_rope_tiler[2]
        )
        subtile_offset = stage_idx * cfg.smem_k_stage_elems
        tile_rows = cutlass.const_expr(cfg.mma_qk_tiler[1] // cfg.num_mma_ctas)
        leading_byte_offset = cutlass.const_expr(qk_desc_leading_byte_offset(cfg))
        stride_byte_offset = cutlass.const_expr(qk_desc_stride_byte_offset(cfg))
        layout = cutlass.const_expr(qk_desc_layout(cfg))

        if cutlass.const_expr(k_subtile_idx < cfg.iterations_qk_latent):
            subtile_offset += k_subtile_idx * k_latent_subtile_elems
        else:
            rope_idx = k_subtile_idx - cfg.iterations_qk_latent
            subtile_offset += (
                k_latent_subtile_elems * cfg.iterations_qk_latent
                + rope_idx * k_rope_subtile_elems
            )
            leading_byte_offset = cutlass.const_expr(
                qk_desc_leading_byte_offset_for_head_dim(
                    cfg, tile_rows, cfg.mma_qk_rope_tiler[2]
                )
            )
            stride_byte_offset = cutlass.const_expr(
                qk_desc_stride_byte_offset_for_head_dim(cfg, cfg.mma_qk_rope_tiler[2])
            )
            layout = cutlass.const_expr(
                qk_desc_layout_for_head_dim(cfg, cfg.mma_qk_rope_tiler[2])
            )

        sk_ptr = self.smem_k.data_ptr(subtile_offset)
        desc = Int64(
            cprims.Tcgen05SmemDesc.build(
                start_address=sk_ptr.toint(Int32),
                leading_byte_offset=leading_byte_offset,
                stride_byte_offset=stride_byte_offset,
                layout=layout,
            )
        )
        return desc


# =====================================================================
# SmemVResource — FP8 V SMEM buffer with one pipeline stage per V tile
# =====================================================================


@dataclass(kw_only=True)
class SmemVResource(HighThroughputMlaResource):
    """SMEM V buffer for the FP8 split-MMA path.

    Producer: LoadTma. Consumer: MmaPvTask. A single stage contains every V
    sub-tile needed by the PV MMA for one logical K tile.
    """

    smem_v: Any = None
    page_offsets: Any = None
    tma_desc_c_transpose: Any = None
    logical_seq_len_q: cutlass.Constexpr[int] = 1
    cfg: cutlass.Constexpr = field(default_factory=MlaDecodeConfig)
    desc_v_base: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    _task_local_specs: ClassVar[tuple[tuple, ...]] = (
        ("desc_v_base", Int64, Int64(0), "SMEM descriptor for staged V."),
    )

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_load_state(self, stage_info: StageInfo) -> None:
        """Initialize per-warp state used by V TMA loads."""
        del stage_info

    @producer_work
    @cute.jit
    def tma_load_direct(self, stage_info: StageInfo) -> None:
        """TMA load one full V tile using page offsets read directly from GMEM."""
        cfg = self.cfg
        stage_idx = stage_info.stage_idx
        stage_base = stage_idx * cfg.smem_v_stage_elems
        work_tile = stage_info.work_tile
        blk_coord = work_tile.tile_idx
        k_index = work_tile.k_index_base + Int32(stage_info.loop_offset)
        page_row_idx = blk_coord[2]
        page_offsets_batch = self.page_offsets[None, page_row_idx]

        v_mbar_arr = cutlass.Array(stage_info.barrier.data_ptr(), dtype=Int64)
        cta_v = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        pv_n_per_cta = cutlass.const_expr(cfg.mma_pv_tiler[1] // cfg.num_mma_ctas)
        coord_n_v = cta_v * pv_n_per_cta
        mask_v = Int16(Int32(1) << cta_v)
        pages_per_v_tile = cfg.pages_per_v_tile
        v_smem_panel_elems = V_SMEM_K_BLOCK_TOKENS * V_TMA_LATENT_ELEMENTS
        v_smem_k_block_elems = cfg.num_mma_ctas * v_smem_panel_elems
        svc_copy_elems = cutlass.const_expr(
            cfg.mma_pv_tiler[1] // cfg.num_mma_ctas * cfg.mma_pv_tiler[2]
        )
        # As for K, out-of-table fragments use page 0 and are masked by the
        # runtime sequence length. p128 still shares one page ID across CTAs.

        for v_call in cutlass.range_constexpr(
            cfg.iterations_pv_k * cfg.iterations_pv_n
        ):
            pv_i = v_call // cfg.iterations_pv_n
            pv_j = v_call % cfg.iterations_pv_n
            coord_nj = coord_n_v + cutlass.Int32(pv_j * cfg.mma_pv_tiler[1])
            subtile_base = stage_base + v_call * svc_copy_elems

            # Assemble the fixed K32 SMEM blocks consumed by tcgen05 from
            # page-bounded TMA copies.  Physical pages only select the GMEM
            # page/coordinate; they never change the SMEM descriptor layout.
            for copy_idx in cutlass.range_constexpr(cfg.v_tma_copies_per_subtile):
                local_token_offset = copy_idx * cfg.v_tma_token_count
                tile_token_offset = pv_i * cfg.mma_pv_tiler[2] + local_token_offset
                page_offset = tile_token_offset // cfg.page_size
                coord_k_v = Int32(tile_token_offset % cfg.page_size)
                logical_page_idx = (
                    k_index
                    if pages_per_v_tile == 1
                    else k_index * Int32(pages_per_v_tile) + Int32(page_offset)
                )
                page_idx = Int32(0)
                if cute.elem_less(logical_page_idx, page_offsets_batch.shape[0]):
                    page_idx = page_offsets_batch[logical_page_idx]
                if prims.elect_sync():
                    smem_k_block_idx = local_token_offset // V_SMEM_K_BLOCK_TOKENS
                    token_offset_in_k_block = local_token_offset % V_SMEM_K_BLOCK_TOKENS
                    v_copy_offset = (
                        smem_k_block_idx * v_smem_k_block_elems
                        + token_offset_in_k_block * V_TMA_LATENT_ELEMENTS
                    )
                    v_smem_0 = cutlass.Array(
                        self.smem_v.data_ptr(subtile_base + v_copy_offset),
                        dtype=qkv_dtype(cfg),
                    )
                    prims.cp_async_bulk_tensor_shared_cluster_global(
                        v_smem_0,
                        self.tma_desc_c_transpose,
                        (coord_nj, coord_k_v, page_idx),
                        v_mbar_arr,
                        [],
                        multicast_mask=mask_v,
                        group=prims.CTAGroup.CTA_2,
                    )
                    v_smem_1 = cutlass.Array(
                        self.smem_v.data_ptr(
                            subtile_base + v_copy_offset + v_smem_panel_elems
                        ),
                        dtype=qkv_dtype(cfg),
                    )
                    prims.cp_async_bulk_tensor_shared_cluster_global(
                        v_smem_1,
                        self.tma_desc_c_transpose,
                        (
                            coord_nj + V_TMA_LATENT_ELEMENTS,
                            coord_k_v,
                            page_idx,
                        ),
                        v_mbar_arr,
                        [],
                        multicast_mask=mask_v,
                        group=prims.CTAGroup.CTA_2,
                    )

    @consumer_work(returns=desc_v_base)
    @cute.jit
    def v_desc(self, stage_info: StageInfo, *, v_subtile_idx: cutlass.Constexpr[int]):
        """Build the V SMEM descriptor for the current PV sub-MMA."""
        cfg = self.cfg
        svc_copy_elems = cutlass.const_expr(
            cfg.mma_pv_tiler[1] // cfg.num_mma_ctas * cfg.mma_pv_tiler[2]
        )
        subtile_offset = (
            stage_info.stage_idx * cfg.smem_v_stage_elems
            + v_subtile_idx * svc_copy_elems
        )
        leading_byte_offset = cutlass.const_expr(
            V_SMEM_K_BLOCK_TOKENS * V_TMA_LATENT_ELEMENTS * cfg.qkv_dtype_bytes
        )
        stride_byte_offset = cutlass.const_expr(
            qkv_major_k_stride_bytes_for(cfg, cfg.mma_pv_tiler[2])
        )
        layout = cutlass.const_expr(
            qk_desc_layout_for_head_dim(cfg, cfg.mma_pv_tiler[2])
        )
        svc_ptr = self.smem_v.data_ptr(subtile_offset)
        desc = Int64(
            cprims.Tcgen05SmemDesc.build(
                start_address=svc_ptr.toint(Int32),
                leading_byte_offset=leading_byte_offset,
                stride_byte_offset=stride_byte_offset,
                layout=layout,
            )
        )
        return desc

    @consumer_work(returns=desc_v_base)
    @cute.jit
    def v_desc_n_major(
        self,
        stage_info: StageInfo,
        *,
        pv_n_idx: cutlass.Constexpr[int],
        pv_k_idx: cutlass.Constexpr[int],
    ):
        """Build the V descriptor when PV commits one output-N slice at a time."""
        cfg = self.cfg
        pv_j = cutlass.const_expr(pv_n_idx)
        pv_i = cutlass.const_expr(pv_k_idx)
        v_call = pv_i * cfg.iterations_pv_n + pv_j
        svc_copy_elems = cutlass.const_expr(
            cfg.mma_pv_tiler[1] // cfg.num_mma_ctas * cfg.mma_pv_tiler[2]
        )
        subtile_offset = stage_info.stage_idx * cfg.smem_v_stage_elems + (
            v_call * svc_copy_elems
        )
        leading_byte_offset = cutlass.const_expr(
            V_SMEM_K_BLOCK_TOKENS * V_TMA_LATENT_ELEMENTS * cfg.qkv_dtype_bytes
        )
        stride_byte_offset = cutlass.const_expr(
            qkv_major_k_stride_bytes_for(cfg, cfg.mma_pv_tiler[2])
        )
        layout = cutlass.const_expr(
            qk_desc_layout_for_head_dim(cfg, cfg.mma_pv_tiler[2])
        )
        svc_ptr = self.smem_v.data_ptr(subtile_offset)
        desc = Int64(
            cprims.Tcgen05SmemDesc.build(
                start_address=svc_ptr.toint(Int32),
                leading_byte_offset=leading_byte_offset,
                stride_byte_offset=stride_byte_offset,
                layout=layout,
            )
        )
        return desc


# =====================================================================
# TmemSResource — S scores in TMEM, UmmaProducerAsync pipeline
# =====================================================================


@dataclass(kw_only=True)
class TmemSResource(HighThroughputMlaResource):
    """TMEM S scores.  Producer: MmaTask (QK MMA).  Consumer: SoftmaxTask.

    Pipeline: UmmaProducerAsync, 2 stages.
    Producer call_idx 0..iterations_qk-1: issue QK MMA for each K sub-tile.
    Consumer: load S, apply mask, compute softmax, and stage P for PV MMA.
    """

    tmem_base_addr: Any = None  # TMEM base address (from alloc)
    smem_q_latent: Any = None  # SMEM Q pointers for descriptor building
    smem_q_rope: Any = None
    smem_p: Any = None  # SMEM P array
    smem_exchange: Any = None  # SMEM array for cross-warp max exchange
    softmax_scale_log2: Any = None  # softmax_scale * log2(e)
    cache_seqs: Any = None  # per-batch valid K length
    cu_seqlens_q: Any = None  # cumulative compact-Q offsets, or None for fixed Q
    split_kv: Any = None  # per-work-tile split count
    groups_tokens_heads_q_ratio: cutlass.Constexpr[int] = 1
    logical_num_heads_q: cutlass.Constexpr[int] = 128
    logical_seq_len_q: cutlass.Constexpr[int] = 1
    cfg: cutlass.Constexpr = field(default_factory=MlaDecodeConfig)
    tiled_mma_qk: Any = None
    cta_rank: Any = field(init=False, default=None)
    is_leader: Any = field(init=False, default=None)
    row_max_state: Any = field(init=False, default=None)
    row_sum_state: Any = field(init=False, default=None)
    qk_acc_regs: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    row_max: cutlass.Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    row_sum: cutlass.Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    row_sum_out: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    row_max_new: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    correction_factor_out: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    no_correction_out: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    qk_acc_regs_odd: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    row_max_odd: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    row_sum_odd: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    row_sum_out_odd: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    row_max_new_odd: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    correction_factor_out_odd: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    no_correction_out_odd: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    _task_local_specs: ClassVar[tuple[tuple, ...]] = (
        ("qk_acc_regs", cutlass.Array, None, "Registers holding softmax P values."),
        ("row_max", Float32, Float32(-Float32.inf), "Running row maximum."),
        ("row_sum", Float32, Float32(0), "Running row sum."),
        ("row_sum_out", Float32, Float32(0), "Row sum published to correction."),
        ("row_max_new", Float32, Float32(0), "Updated row maximum."),
        (
            "correction_factor_out",
            Float32,
            Float32(0),
            "Correction factor for the previous O tile.",
        ),
        (
            "no_correction_out",
            Int32,
            Int32(0),
            "Whether O correction may be skipped.",
        ),
        (
            "qk_acc_regs_odd",
            cutlass.Array,
            None,
            "Odd-lane registers holding softmax P values.",
        ),
        ("row_max_odd", Float32, Float32(-Float32.inf), "Odd-lane row maximum."),
        ("row_sum_odd", Float32, Float32(0), "Odd-lane row sum."),
        (
            "row_sum_out_odd",
            Float32,
            Float32(0),
            "Odd-lane row sum published to correction.",
        ),
        ("row_max_new_odd", Float32, Float32(0), "Odd-lane updated row maximum."),
        (
            "correction_factor_out_odd",
            Float32,
            Float32(0),
            "Odd-lane correction factor for the previous O tile.",
        ),
        (
            "no_correction_out_odd",
            Int32,
            Int32(0),
            "Whether odd-lane O correction may be skipped.",
        ),
    )

    @consumer_work(
        work_attrs=WorkAttr.AUXILIARY,
        returns=(
            qk_acc_regs,
            row_max,
            row_sum,
            row_sum_out,
            row_max_new,
            correction_factor_out,
            no_correction_out,
        ),
    )
    @cute.jit
    def init_softmax_state(self, stage_info: StageInfo):
        """Create softmax accumulator registers and row-stat state."""
        del stage_info
        self.cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        self.is_leader = self.cta_rank == 0
        self.row_max_state = Float32(-Float32.inf)
        self.row_sum_state = Float32(0)
        return (
            cutlass.Array(
                Float32,
                64,
                space=cutlass.AddressSpace.rmem,
            ),
            Float32(-Float32.inf),
            Float32(0),
            Float32(0),
            Float32(0),
            Float32(0),
            Int32(0),
        )

    @consumer_work(
        work_attrs=WorkAttr.AUXILIARY,
        returns=(
            qk_acc_regs_odd,
            row_max_odd,
            row_sum_odd,
            row_sum_out_odd,
            row_max_new_odd,
            correction_factor_out_odd,
            no_correction_out_odd,
        ),
    )
    @cute.jit
    def init_softmax_state_odd(self, stage_info: StageInfo):
        """Create odd-lane softmax accumulator registers and row-stat state."""
        del stage_info
        self.cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        self.is_leader = self.cta_rank == 0
        self.row_max_state = Float32(-Float32.inf)
        self.row_sum_state = Float32(0)
        return (
            cutlass.Array(
                Float32,
                64,
                space=cutlass.AddressSpace.rmem,
            ),
            Float32(-Float32.inf),
            Float32(0),
            Float32(0),
            Float32(0),
            Float32(0),
            Int32(0),
        )

    @producer_work
    @cute.jit
    def qk_mma(
        self,
        stage_info: StageInfo,
        *,
        desc_k_base,
        k_subtile_idx: cutlass.Constexpr[int],
    ) -> None:
        """Issue one QK MMA sub-tile (K latent or K rope).
        Only leader CTA issues MMA (2CTA UMMA principle).

        Descriptor computation is hoisted OUTSIDE the leader-CTA gate so
        ptxas keeps values in uniform registers.
        Only the actual tcgen05_mma is gated by elect_sync + leader check.
        """
        cfg = self.cfg
        call_idx = k_subtile_idx

        # Hoist descriptor computation outside leader-CTA gate to preserve
        # uniform register allocation (avoids R2UR demote/promote).
        tmem_s_addr = self.tmem_base_addr + 64 * stage_info.stage_idx
        tmem_s_ptr = prims.make_tmem_ptr(tmem_s_addr, Float32)

        idesc_qk = cprims.Tcgen05InstrDesc.build(
            c_dtype=Float32,
            a_dtype=qkv_dtype(cfg),
            b_dtype=qkv_dtype(cfg),
            n_dim=cfg.mma_qk_tiler[1],
            m_dim=cfg.mma_qk_tiler[0],
        )
        mma_kind = mma_kind_for_qkv(cfg)
        cta_group = prims.CTAGroup.CTA_2
        k_block_count = cutlass.const_expr(
            ceil_div(cfg.mma_qk_tiler_k, mma_k_step_for_qkv(cfg))
        )
        is_leader_cta = (
            cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster()) == 0
        )

        desc_k = desc_k_base

        if cutlass.const_expr(call_idx < cfg.iterations_qk_latent_stages):
            qc_copy_elems = cutlass.const_expr(
                cfg.mma_qk_tiler[0] // cfg.num_mma_ctas * cfg.mma_qk_tiler_k
            )
            kc_copy_elems = cutlass.const_expr(
                cfg.mma_qk_tiler[1] // cfg.num_mma_ctas * cfg.mma_qk_tiler_k
            )
            desc_k_delta = cutlass.const_expr(kc_copy_elems * cfg.qkv_dtype_bytes // 16)
            for stage_subtile_idx in cutlass.range_constexpr(cfg.kv_subtiles_per_stage):
                logical_call_idx = (
                    call_idx * cfg.kv_subtiles_per_stage + stage_subtile_idx
                )
                q_smem_ptr = self.smem_q_latent.data_ptr(
                    logical_call_idx * qc_copy_elems
                )
                desc_q = Int64(
                    cprims.Tcgen05SmemDesc.build(
                        start_address=q_smem_ptr.toint(Int32),
                        leading_byte_offset=qk_desc_leading_byte_offset(cfg),
                        stride_byte_offset=qk_desc_stride_byte_offset(cfg),
                        layout=qk_desc_layout(cfg),
                    )
                )
                desc_k_subtile = desc_k + stage_subtile_idx * desc_k_delta
                if is_leader_cta:
                    for k_block in cutlass.range_constexpr(k_block_count):
                        scale_d = cutlass.const_expr(
                            logical_call_idx > 0 or k_block > 0
                        )
                        if prims.elect_sync():
                            prims.tcgen05_mma(
                                mma_kind,
                                cta_group,
                                tmem_s_ptr,
                                desc_q + k_block * 2,
                                desc_k_subtile + k_block * 2,
                                idesc_qk,
                                Boolean(scale_d),
                            )

        elif cutlass.const_expr(call_idx < cfg.iterations_qk_stages):
            # K rope: build Q rope descriptor unconditionally
            rope_idx = call_idx - cfg.iterations_qk_latent_stages
            qc_copy_elems = cutlass.const_expr(
                cfg.mma_qk_tiler[0] // cfg.num_mma_ctas * cfg.mma_qk_rope_tiler[2]
            )
            q_rope_smem_ptr = self.smem_q_rope.data_ptr(rope_idx * qc_copy_elems)
            q_rope_rows = cutlass.const_expr(cfg.mma_qk_tiler[0] // cfg.num_mma_ctas)
            q_rope_dim = cutlass.const_expr(cfg.mma_qk_rope_tiler[2])
            desc_qr = Int64(
                cprims.Tcgen05SmemDesc.build(
                    start_address=q_rope_smem_ptr.toint(Int32),
                    leading_byte_offset=qk_desc_leading_byte_offset_for_head_dim(
                        cfg, q_rope_rows, q_rope_dim
                    ),
                    stride_byte_offset=qk_desc_stride_byte_offset_for_head_dim(
                        cfg, q_rope_dim
                    ),
                    layout=qk_desc_layout_for_head_dim(cfg, q_rope_dim),
                )
            )
            if is_leader_cta:
                for k_block in cutlass.range_constexpr(
                    cfg.rope_dim // mma_k_step_for_qkv(cfg)
                ):
                    if prims.elect_sync():
                        prims.tcgen05_mma(
                            mma_kind,
                            cta_group,
                            tmem_s_ptr,
                            desc_qr + k_block * 2,
                            desc_k + k_block * 2,
                            idesc_qk,
                            Boolean(True),
                        )

    @consumer_work(work_attrs=WorkAttr.AUXILIARY, returns=(row_sum, row_sum_out))
    @cute.jit
    def finish_row_sum(
        self,
        stage_info: StageInfo,
        *,
        qk_acc_regs,
        row_sum,
        correction_factor_out,
    ):
        """Finish row-sum reduction after P is published to SMEM."""
        del stage_info
        row_sum = row_sum * correction_factor_out
        row_sum_vec = (Float32(0), Float32(0))
        for i in cutlass.range_constexpr(0, 64, 2):
            row_sum_vec = add_packed_f32x2(
                row_sum_vec,
                (qk_acc_regs[i], qk_acc_regs[i + 1]),
            )
        row_sum = row_sum_vec[0] + row_sum_vec[1] + row_sum
        self.row_sum_state = row_sum
        return row_sum, row_sum

    @consumer_work(
        work_attrs=WorkAttr.AUXILIARY, returns=(row_sum_odd, row_sum_out_odd)
    )
    @cute.jit
    def finish_row_sum_odd(
        self,
        stage_info: StageInfo,
        *,
        qk_acc_regs_odd,
        row_sum_odd,
        correction_factor_out_odd,
    ):
        """Finish odd-lane row-sum reduction after P is published to SMEM."""
        del stage_info
        row_sum = row_sum_odd * correction_factor_out_odd
        row_sum_vec = (Float32(0), Float32(0))
        for i in cutlass.range_constexpr(0, 64, 2):
            row_sum_vec = add_packed_f32x2(
                row_sum_vec,
                (qk_acc_regs_odd[i], qk_acc_regs_odd[i + 1]),
            )
        row_sum = row_sum_vec[0] + row_sum_vec[1] + row_sum
        self.row_sum_state = row_sum
        return row_sum, row_sum

    @cute.jit
    def _load_s_impl(
        self,
        stage_info: StageInfo,
        *,
        qk_acc_regs,
        row_max,
        row_sum,
        row_sum_out,
        row_max_new,
        correction_factor_out,
        no_correction_out,
    ):
        """Load S from TMEM and compute the local row max."""
        del row_sum_out, row_max_new, correction_factor_out, no_correction_out
        cfg = self.cfg
        tidx = cute.arch.thread_idx()[0]
        num_compute_threads = cfg.num_compute_warps * cfg.threads_per_warp
        local_tidx = tidx % num_compute_threads
        stage_idx = stage_info.stage_idx

        work_tile = stage_info.work_tile
        K = Int32(work_tile.k_len)
        k_index = work_tile.k_index_base + Int32(stage_info.loop_offset)
        tile_offset_k = k_index * Int32(cfg.mma_qk_tiler[1])
        needs_row_causal_mask = cutlass.const_expr(
            cfg.mask_type == MaskType.CAUSAL.value
            and self.groups_tokens_heads_q_ratio > 1
        )
        min_visible_k_len = K
        if cutlass.const_expr(needs_row_causal_mask):
            min_visible_k_len = K - Int32(self.groups_tokens_heads_q_ratio - 1)
        group_needs_mask = kv_tile_needs_right_mask(
            tile_offset_k,
            Int32(cfg.mma_qk_tiler[1]),
            min_visible_k_len,
        )

        neg_inf = Float32(-Float32.inf)
        row_max_tile = row_max
        warp_id = local_tidx >> 5
        tmem_warp_row_id = self.tmem_base_addr + warp_id * TCGEN05_32B_REGS_PER_LOAD
        stage_offset = stage_idx * 64
        tmem_raw_addr = (tmem_warp_row_id << 16) | stage_offset
        t2r_shape = TCGEN05_32B_SHAPE
        for load_idx in cutlass.range_constexpr(2):
            curr_addr = tmem_raw_addr + load_idx * TCGEN05_32B_REGS_PER_LOAD
            tmem_ptr = prims.make_tmem_ptr(curr_addr, Float32)
            loaded = prims.tcgen05_ld(
                t2r_shape, tmem_ptr, num=TCGEN05_32B_REGS_PER_LOAD
            )
            qk_acc_regs.store(loaded, load_idx * TCGEN05_32B_REGS_PER_LOAD)

        if group_needs_mask:
            row_k_len = K
            if cutlass.const_expr(needs_row_causal_mask):
                # Recover this row's exact endpoint only on a group boundary.
                batch_idx = Int32(work_tile.tile_idx[2])
                _, logical_seq_len_q = query_batch_bounds(
                    self.cu_seqlens_q,
                    batch_idx,
                    self.logical_seq_len_q,
                )
                effective_head_idx = self.cta_rank * Int32(
                    cfg.mma_qk_tiler[0] // cfg.num_mma_ctas
                ) + (local_tidx & Int32(EPILOGUE_ROW_MASK))
                _, _, logical_q_idx, _, _ = groups_tokens_heads_q_row_state(
                    effective_head_idx,
                    work_tile.tile_idx[1],
                    self.groups_tokens_heads_q_ratio,
                    self.logical_num_heads_q,
                    self.logical_seq_len_q,
                    self.cu_seqlens_q,
                    batch_idx,
                )
                row_k_len = mask_visible_k_length(
                    cfg.mask_type,
                    self.cache_seqs[batch_idx],
                    logical_q_idx,
                    logical_seq_len_q,
                )
            tidx_col = (
                local_tidx >> EPILOGUE_COLUMN_GROUP_SHIFT
            ) << EPILOGUE_COLUMN_GROUP_SHIFT
            for i in cutlass.range_constexpr(64):
                token_idx = tile_offset_k + tidx_col + Int32(i)
                qk_acc_regs[i] = qk_acc_regs[i] if token_idx < row_k_len else neg_inf

        max0 = neg_inf
        max1 = neg_inf
        max2 = neg_inf
        max3 = neg_inf
        for i in cutlass.range_constexpr(16):
            max0 = fmax_f32(max0, qk_acc_regs[i])
            max1 = fmax_f32(max1, qk_acc_regs[i + 16])
            max2 = fmax_f32(max2, qk_acc_regs[i + 32])
            max3 = fmax_f32(max3, qk_acc_regs[i + 48])
        row_max_tile = fmax_f32(
            row_max_tile,
            fmax_f32(fmax_f32(max0, max1), fmax_f32(max2, max3)),
        )
        cute.arch.fence_view_async_tmem_load()
        return (
            qk_acc_regs,
            row_max,
            row_sum,
            row_sum,
            row_max_tile,
            Float32(1),
            Int32(1),
        )

    @consumer_work(
        returns=(
            qk_acc_regs,
            row_max,
            row_sum,
            row_sum_out,
            row_max_new,
            correction_factor_out,
            no_correction_out,
        ),
    )
    @cute.jit
    def load_s(
        self,
        stage_info: StageInfo,
        *,
        qk_acc_regs,
        row_max,
        row_sum,
        row_sum_out,
        row_max_new,
        correction_factor_out,
        no_correction_out,
    ):
        """Load S for the even softmax group."""
        return self._load_s_impl(
            stage_info,
            qk_acc_regs=qk_acc_regs,
            row_max=row_max,
            row_sum=row_sum,
            row_sum_out=row_sum_out,
            row_max_new=row_max_new,
            correction_factor_out=correction_factor_out,
            no_correction_out=no_correction_out,
        )

    @consumer_work(
        returns=(
            qk_acc_regs_odd,
            row_max_odd,
            row_sum_odd,
            row_sum_out_odd,
            row_max_new_odd,
            correction_factor_out_odd,
            no_correction_out_odd,
        ),
    )
    @cute.jit
    def load_s_odd(
        self,
        stage_info: StageInfo,
        *,
        qk_acc_regs_odd,
        row_max_odd,
        row_sum_odd,
        row_sum_out_odd,
        row_max_new_odd,
        correction_factor_out_odd,
        no_correction_out_odd,
    ):
        """Load S for the odd softmax group."""
        return self._load_s_impl(
            stage_info,
            qk_acc_regs=qk_acc_regs_odd,
            row_max=row_max_odd,
            row_sum=row_sum_odd,
            row_sum_out=row_sum_out_odd,
            row_max_new=row_max_new_odd,
            correction_factor_out=correction_factor_out_odd,
            no_correction_out=no_correction_out_odd,
        )

    @cute.jit
    def _finish_softmax_impl(
        self,
        stage_info: StageInfo,
        *,
        qk_acc_regs,
        row_max,
        row_sum,
        row_sum_out,
        row_max_new,
        correction_factor_out,
        no_correction_out,
        softmax_group_id: cutlass.Constexpr[int] = 0,
    ):
        """Finish row-max exchange, online correction, and P exponentiation."""
        del row_sum_out, correction_factor_out, no_correction_out
        cfg = self.cfg
        tidx = cute.arch.thread_idx()[0]
        num_compute_threads = cfg.num_compute_warps * cfg.threads_per_warp
        local_tidx = tidx % num_compute_threads
        neg_inf = Float32(-Float32.inf)
        row_max_prev = row_max
        row_sum_prev = row_sum
        row_max_tile = row_max_new

        group_exchange_base = Int32(softmax_group_id * num_compute_threads)
        self.smem_exchange[group_exchange_base + local_tidx] = row_max_tile
        prims.barrier_cta_sync(
            cfg.softmax_sync_bar_id + softmax_group_id,
            thread_count=cfg.softmax_sync_threads,
        )
        peer_idx = (local_tidx + 64) % num_compute_threads
        row_max_tile = fmax_f32(
            row_max_tile, self.smem_exchange[group_exchange_base + peer_idx]
        )
        # The exchange buffer is reused on the next KV iteration.  Keep every
        # peer read ahead of any warp's next write to the same slot.
        prims.barrier_cta_sync(
            cfg.softmax_sync_bar_id + softmax_group_id,
            thread_count=cfg.softmax_sync_threads,
        )

        if cutlass.const_expr(cfg.use_fp8_dual_softmax_schedule):
            stage_idx = stage_info.stage_idx

            def load_peer_state():
                """Load the peer softmax group's correction state from TMEM."""

                peer_stage_idx = (stage_idx + Int32(1)) % Int32(cfg.p_cor_stage)
                corr_col_offset = cfg.correction_factor_offset + peer_stage_idx * 4
                peer_warp_id = local_tidx >> 5
                peer_tmem_row = (
                    self.tmem_base_addr + peer_warp_id * TCGEN05_32B_REGS_PER_LOAD
                )
                peer_tmem_addr = (peer_tmem_row << 16) | corr_col_offset
                peer_tmem_ptr = prims.make_tmem_ptr(peer_tmem_addr, Float32)
                peer_corr = prims.tcgen05_ld(TCGEN05_32B_SHAPE, peer_tmem_ptr, num=2)
                return peer_corr[1], peer_corr[0]

            if cutlass.const_expr(softmax_group_id == 1):
                if stage_info.loop_offset != Int32(0):
                    prims.barrier_cta_sync(
                        cfg.softmax_order_bar_1_id,
                        thread_count=2 * num_compute_threads,
                    )
                    cute.arch.fence_acq_rel_cta()
                    row_max_prev, row_sum_prev = load_peer_state()
            else:
                if stage_info.loop_offset != Int32(0):
                    prims.barrier_cta_sync(
                        cfg.softmax_order_bar_0_id,
                        thread_count=2 * num_compute_threads,
                    )
                    cute.arch.fence_acq_rel_cta()
                    row_max_prev, row_sum_prev = load_peer_state()

        row_max_new = fmax_f32(row_max_prev, row_max_tile)
        row_has_values = row_max_new != neg_inf
        safe_row_max_prev = row_max_prev if row_has_values else Float32(0)
        safe_row_max_new = row_max_new if row_has_values else Float32(0)
        # Exact max equality makes the correction scale exactly one. Keep that
        # lane on the identity value and avoid issuing exp2 altogether.
        max_changed = safe_row_max_prev != safe_row_max_new
        correction_factor = Float32(1)
        if max_changed:
            correction_factor = cute.math.exp2(
                (safe_row_max_prev - safe_row_max_new) * self.softmax_scale_log2,
                fastmath=True,
            )
        no_correction = Int32(not max_changed)

        fma_b = self.softmax_scale_log2
        fma_c = Float32(0) - safe_row_max_new * self.softmax_scale_log2
        if cutlass.const_expr(cfg.is_fp8_qkv()):
            # Match the 448-scaled E4M3 P convention used by the reference
            # output and the 1CTA implementation.
            fma_c = fma_c + fp8_log2_quant_scale()
        for i in cutlass.range_constexpr(0, 64, 2):
            fma_result = fma_packed_f32x2(
                (qk_acc_regs[i], qk_acc_regs[i + 1]),
                (fma_b, fma_b),
                (fma_c, fma_c),
            )
            qk_acc_regs[i] = cute.math.exp2(fma_result[0], fastmath=True)
            qk_acc_regs[i + 1] = cute.math.exp2(fma_result[1], fastmath=True)

        self.row_max_state = row_max_new
        return (
            qk_acc_regs,
            row_max_new,
            row_sum_prev,
            row_sum_prev,
            row_max_new,
            correction_factor,
            no_correction,
        )

    @consumer_work(
        work_attrs=WorkAttr.AUXILIARY,
        returns=(
            qk_acc_regs,
            row_max,
            row_sum,
            row_sum_out,
            row_max_new,
            correction_factor_out,
            no_correction_out,
        ),
    )
    @cute.jit
    def finish_softmax(
        self,
        stage_info: StageInfo,
        *,
        qk_acc_regs,
        row_max,
        row_sum,
        row_sum_out,
        row_max_new,
        correction_factor_out,
        no_correction_out,
    ):
        """Finish softmax for the even group after S release."""
        return self._finish_softmax_impl(
            stage_info,
            qk_acc_regs=qk_acc_regs,
            row_max=row_max,
            row_sum=row_sum,
            row_sum_out=row_sum_out,
            row_max_new=row_max_new,
            correction_factor_out=correction_factor_out,
            no_correction_out=no_correction_out,
            softmax_group_id=0,
        )

    @consumer_work(
        work_attrs=WorkAttr.AUXILIARY,
        returns=(
            qk_acc_regs_odd,
            row_max_odd,
            row_sum_odd,
            row_sum_out_odd,
            row_max_new_odd,
            correction_factor_out_odd,
            no_correction_out_odd,
        ),
    )
    @cute.jit
    def finish_softmax_odd(
        self,
        stage_info: StageInfo,
        *,
        qk_acc_regs_odd,
        row_max_odd,
        row_sum_odd,
        row_sum_out_odd,
        row_max_new_odd,
        correction_factor_out_odd,
        no_correction_out_odd,
    ):
        """Finish softmax for the odd group after S release."""
        return self._finish_softmax_impl(
            stage_info,
            qk_acc_regs=qk_acc_regs_odd,
            row_max=row_max_odd,
            row_sum=row_sum_odd,
            row_sum_out=row_sum_out_odd,
            row_max_new=row_max_new_odd,
            correction_factor_out=correction_factor_out_odd,
            no_correction_out=no_correction_out_odd,
            softmax_group_id=1,
        )


# =====================================================================
# SmemPResource — P in SMEM, UmmaConsumerAsync pipeline
# =====================================================================


@dataclass(kw_only=True)
class SmemPResource(HighThroughputMlaResource):
    """SMEM P buffer.  Producer: SoftmaxTask.  Consumer: MmaTask (PV MMA).

    Pipeline: UmmaConsumerAsync, 2 stages.
    """

    smem_p: Any = None
    cfg: cutlass.Constexpr = field(default_factory=MlaDecodeConfig)
    cta_rank: Any = field(init=False, default=None)
    desc_p_base: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    _task_local_specs: ClassVar[tuple[tuple, ...]] = (
        ("desc_p_base", Int64, Int64(0), "SMEM descriptor for staged P."),
    )

    @cute.jit
    def _store_p_impl(self, stage_info: StageInfo, *, qk_acc_regs) -> None:
        """Store softmax P tile to SMEM for PV MMA.

        Converts P values to FP16 and stores to SMEM with correct swizzle layout.
        """
        cfg = self.cfg
        stage_idx = stage_info.stage_idx
        num_mma_ctas = cfg.num_mma_ctas

        tidx = cute.arch.thread_idx()[0]
        num_compute_threads = cfg.num_compute_warps * cfg.threads_per_warp
        local_tidx = tidx % num_compute_threads
        lane_idx = local_tidx & 31
        warp_idx = local_tidx >> 5

        # Convert F32 -> QKV dtype into a vectorized local buffer.
        qkv_element_dtype = qkv_dtype(cfg)
        s_regs = cutlass.Array(qkv_element_dtype, 64, space=cutlass.AddressSpace.rmem)
        s_regs.store(qk_acc_regs.load(0, 64).to(qkv_element_dtype), 0)

        # Compute SMEM P stage offset
        sp_stage_stride_elems = cutlass.const_expr(
            cfg.mma_pv_tiler[0]
            // num_mma_ctas
            * cfg.mma_pv_tiler[2]
            * cfg.iterations_pv_k
        )
        smem_p_base_bytes = (
            self.smem_p.data_ptr().toint(cutlass.Int32)
            + stage_idx * sp_stage_stride_elems * cfg.qkv_dtype_bytes
        )

        if cutlass.const_expr(cfg.is_fp8_qkv()):
            # FP8 P layout:
            # S<2,4,3> o ((64,32),1,2,(2,2)):((64,1),0,32,(4096,8192)).
            # Each universal SMEM copy is 128 bits, i.e. 16 E4M3 elements.
            # Lane pairs write opposite halves of the 128B swizzled row.  The
            # signed strides below walk the four 16B blocks inside that row.
            m = ((lane_idx >> 1) & 3) * 128
            base = cutlass.Int32(0)
            base = base + (lane_idx & 1) * 64
            base = base + (lane_idx >> 3) * 512
            base = base + (warp_idx & 1) * 2048
            base = base + (warp_idx >> 1) * 4096
            swizzle_xor = m ^ ((m & 384) >> 3)
            off_base = base + swizzle_xor

            stride_a = cutlass.Int32(16)
            if (m & 128) != 0:
                stride_a = cutlass.Int32(-16)
            stride_b = cutlass.Int32(32)
            if (m & 256) != 0:
                stride_b = cutlass.Int32(-32)

            dst_blk_offs = (
                cutlass.Int32(0),
                stride_a,
                stride_b,
                stride_a + stride_b,
            )
            for blk in cutlass.range_constexpr(4):
                src_blk_base = blk * 16
                dst_blk_addr = smem_p_base_bytes + off_base + dst_blk_offs[blk]
                vec = s_regs.load(src_blk_base, 16)
                smem_ptr = cutlass.inttoptr(dst_blk_addr, 3, qkv_element_dtype)
                smem_ptr.store(vec, alignment=16)
        else:
            # BF16 P layout:
            # S<2,4,3> o ((64,16),1,2,(4,2)):((32,1),0,16,(2048,8192)).
            # Each universal SMEM copy is 128 bits, i.e. 8 BF16 elements.
            # The two K slices live 2048 elements apart in the staged P tile.
            # Per-block strides mirror the FP8 swizzle at half the byte width.
            m = ((lane_idx >> 1) & 3) * 64
            base = cutlass.Int32(0)
            base = base + (lane_idx & 1) * 32
            base = base + (lane_idx >> 3) * 256
            base = base + (warp_idx & 1) * 1024
            base = base + (warp_idx >> 1) * 4096
            swizzle_xor = m ^ ((m & 192) >> 3)
            off_base = base + swizzle_xor

            stride_a = cutlass.Int32(8)
            if (m & 64) != 0:
                stride_a = cutlass.Int32(-8)
            stride_b = cutlass.Int32(16)
            if (m & 128) != 0:
                stride_b = cutlass.Int32(-16)

            dst_blk_offs = (
                cutlass.Int32(0),
                stride_a,
                stride_b,
                stride_a + stride_b,
            )
            for k in cutlass.range_constexpr(2):
                k_base = off_base + k * 2048
                src_k_base = k * 32
                for blk in cutlass.range_constexpr(4):
                    src_blk_base = src_k_base + blk * 8
                    dst_blk_addr = (
                        smem_p_base_bytes
                        + (k_base + dst_blk_offs[blk]) * cfg.qkv_dtype_bytes
                    )
                    vec = s_regs.load(src_blk_base, 8)
                    smem_ptr = cutlass.inttoptr(dst_blk_addr, 3, qkv_element_dtype)
                    smem_ptr.store(vec, alignment=16)

        # Fence between SMEM store and MMA read
        prims.fence_proxy(
            kind=prims.Proxy.ASYNC_SHARED,
            space=prims.SharedSpace.shared_cta,
        )

    @producer_work
    @cute.jit
    def store_p(self, stage_info: StageInfo, *, qk_acc_regs) -> None:
        """Store P from the even softmax group."""
        self._store_p_impl(stage_info, qk_acc_regs=qk_acc_regs)

    @producer_work
    @cute.jit
    def store_p_odd(self, stage_info: StageInfo, *, qk_acc_regs_odd) -> None:
        """Store P from the odd softmax group."""
        self._store_p_impl(stage_info, qk_acc_regs=qk_acc_regs_odd)

    @consumer_work(returns=desc_p_base)
    @cute.jit
    def p_desc(self, stage_info: StageInfo):
        """MMA warp builds P SMEM descriptor for PV MMA."""
        cfg = self.cfg
        sp_stage_stride_elems = cutlass.const_expr(
            cfg.mma_pv_tiler[0]
            // cfg.num_mma_ctas
            * cfg.mma_pv_tiler[2]
            * cfg.iterations_pv_k
        )
        sp_ptr = self.smem_p.data_ptr(stage_info.stage_idx * sp_stage_stride_elems)
        desc_p = Int64(
            cprims.Tcgen05SmemDesc.build(
                start_address=sp_ptr.toint(Int32),
                leading_byte_offset=p_desc_leading_byte_offset(cfg),
                stride_byte_offset=p_desc_stride_byte_offset(cfg),
                layout=p_desc_layout(cfg),
            )
        )
        return desc_p


# =====================================================================
# TmemCorrResource — Correction factors via TMEM, Async pipeline
# =====================================================================


@dataclass(kw_only=True)
class TmemCorrResource(HighThroughputMlaResource):
    """Correction factors in TMEM.  Producer: SoftmaxTask.  Consumer: CorrectionTask.

    Pipeline: Async, 2 stages.
    Carries (row_sum, row_max, correction_factor, no_correction) per thread.
    """

    tmem_base_addr: Any = None
    smem_exchange: Any = None
    cfg: cutlass.Constexpr = field(default_factory=MlaDecodeConfig)
    cta_rank: Any = field(init=False, default=None)
    final_row_stats: Any = field(init=False, default=None)
    row_sum: cutlass.Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    row_max: cutlass.Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    epilogue_row_sum: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    epilogue_row_max: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    correction_factor: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    no_correction: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    _task_local_specs: ClassVar[tuple[tuple, ...]] = (
        ("row_sum", Float32, Float32(0), "Final running row sum."),
        ("row_max", Float32, Float32(0), "Final running row max."),
        ("epilogue_row_sum", Float32, Float32(0), "Final exchanged row sum."),
        ("epilogue_row_max", Float32, Float32(0), "Final row max for LSE."),
        (
            "correction_factor",
            Float32,
            Float32(0),
            "Correction factor for the previous O tile.",
        ),
        ("no_correction", Int32, Int32(0), "Whether O correction may be skipped."),
    )

    @consumer_work(
        work_attrs=WorkAttr.AUXILIARY,
        returns=(row_sum, row_max, correction_factor, no_correction),
    )
    @cute.jit
    def init_load_state(self, stage_info: StageInfo):
        """Create row-stat variables consumed by correction and epilogue code."""
        del stage_info
        self.cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        self.final_row_stats = cutlass.Array(
            Float32,
            2,
            space=cutlass.AddressSpace.rmem,
        )
        return Float32(0), Float32(0), Float32(0), Int32(0)

    @cute.jit
    def _store_corr_impl(
        self,
        stage_info: StageInfo,
        *,
        row_sum_out,
        row_max_new,
        correction_factor_out,
        no_correction_out,
        softmax_group_id: cutlass.Constexpr[int] = 0,
        arrive_peer: cutlass.Constexpr[bool] = True,
    ) -> None:
        """Store correction factors [row_sum, row_max, correction, no_correction] to TMEM."""
        cfg = self.cfg
        stage_idx = stage_info.stage_idx

        tidx = cute.arch.thread_idx()[0]
        num_compute_threads = cfg.num_compute_warps * cfg.threads_per_warp
        local_tidx = tidx % num_compute_threads
        warp_id = local_tidx >> 5

        col_offset = cfg.correction_factor_offset + stage_idx * 4
        tmem_warp_row_id = self.tmem_base_addr + warp_id * TCGEN05_32B_REGS_PER_LOAD
        # tcgen05 addresses pack the TMEM row into the high 16 bits.  Each warp
        # owns one correction row and four adjacent columns for sum/max/scale.
        tmem_raw_addr = (tmem_warp_row_id << 16) | col_offset
        tmem_ptr_arr = prims.make_tmem_ptr(tmem_raw_addr, Float32)

        correction_regs = cutlass.Array(Float32, 4, space=cutlass.AddressSpace.rmem)
        correction_regs[0] = row_sum_out
        correction_regs[1] = row_max_new
        correction_regs[2] = correction_factor_out
        correction_regs[3] = cprims.mov_b32(no_correction_out, target_type=Float32)

        prims.tcgen05_st(
            TCGEN05_32B_SHAPE,
            tmem_ptr_arr,
            correction_regs[0:4],
        )
        cute.arch.fence_view_async_tmem_store()

        if cutlass.const_expr(cfg.use_fp8_dual_softmax_schedule and arrive_peer):
            # Dual-softmax pipes are ordered one loop tile apart: after one pipe
            # publishes correction state, it releases the peer pipe for the next
            # loop tile if that peer still has work.
            has_peer_next = stage_info.loop_offset + Int32(1) < stage_info.loop_end
            if has_peer_next:
                prims.barrier_cta_arrive(
                    (
                        cfg.softmax_order_bar_0_id
                        if cutlass.const_expr(softmax_group_id == 1)
                        else cfg.softmax_order_bar_1_id
                    ),
                    2 * num_compute_threads,
                )

    @producer_work
    @cute.jit
    def store_corr(
        self,
        stage_info: StageInfo,
        *,
        row_sum_out,
        row_max_new,
        correction_factor_out,
        no_correction_out,
    ) -> None:
        """Store correction metadata for the even softmax group."""
        self._store_corr_impl(
            stage_info,
            row_sum_out=row_sum_out,
            row_max_new=row_max_new,
            correction_factor_out=correction_factor_out,
            no_correction_out=no_correction_out,
            softmax_group_id=0,
            arrive_peer=True,
        )

    @producer_work
    @cute.jit
    def store_corr_odd(
        self,
        stage_info: StageInfo,
        *,
        row_sum_out_odd,
        row_max_new_odd,
        correction_factor_out_odd,
        no_correction_out_odd,
    ) -> None:
        """Store correction metadata for the odd softmax group."""
        self._store_corr_impl(
            stage_info,
            row_sum_out=row_sum_out_odd,
            row_max_new=row_max_new_odd,
            correction_factor_out=correction_factor_out_odd,
            no_correction_out=no_correction_out_odd,
            softmax_group_id=1,
            arrive_peer=True,
        )

    @consumer_work(returns=(row_sum, row_max, correction_factor, no_correction))
    @cute.jit
    def load_corr(self, stage_info: StageInfo):
        """Load correction factors from TMEM."""
        cfg = self.cfg
        stage_idx = stage_info.stage_idx

        tidx = cute.arch.thread_idx()[0]
        # Use local tidx within 4-warp correction group (matching bare-metal)
        local_tidx = tidx % (cfg.num_compute_warps * cfg.threads_per_warp)
        warp_id = local_tidx >> 5

        col_offset = cfg.correction_factor_offset + stage_idx * 4
        tmem_warp_row_id = self.tmem_base_addr + warp_id * TCGEN05_32B_REGS_PER_LOAD
        # Load from the same packed row/column address used by store_corr so the
        # correction consumer sees the row statistics for its current stage.
        tmem_raw_addr = (tmem_warp_row_id << 16) | col_offset
        tmem_ptr_arr = prims.make_tmem_ptr(tmem_raw_addr, Float32)

        prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
        loaded = prims.tcgen05_ld(
            TCGEN05_32B_SHAPE,
            tmem_ptr_arr,
            num=4,
        )

        self.final_row_stats[0] = loaded[0]
        self.final_row_stats[1] = loaded[1]
        return loaded[0], loaded[1], loaded[2], loaded[3].bitcast(Int32)

    @consumer_work(
        work_attrs=WorkAttr.AUXILIARY, returns=(epilogue_row_sum, epilogue_row_max)
    )
    @cute.jit
    def prepare_epilogue_slice_store(self, stage_info: StageInfo):
        """Exchange final row statistics once before per-slice O stores."""
        del stage_info
        cfg = self.cfg
        tidx = cute.arch.thread_idx()[0]
        row_sum = self.final_row_stats[0]
        row_max = self.final_row_stats[1]

        num_compute_threads = cfg.num_compute_warps * cfg.threads_per_warp
        local_tidx = tidx % num_compute_threads
        smem_ex_ptr = cutlass.inttoptr(
            self.smem_exchange + local_tidx * 4,
            3,
            Float32,
        )
        smem_ex_ptr.store(row_sum)
        prims.barrier_cta_sync(
            cfg.epilogue_sync_bar_id, thread_count=cfg.epilogue_sync_threads
        )
        # The two CTAs in the cluster own complementary halves of the 2CTA row.
        # Exchanging row sums through SMEM gives both epilogue slices the same
        # denominator while preserving each CTA's local row max for LSE.
        peer_idx = (local_tidx + 64) % num_compute_threads
        peer_ptr = cutlass.inttoptr(
            self.smem_exchange + peer_idx * 4,
            3,
            Float32,
        )
        return row_sum + peer_ptr.load(), row_max


# =====================================================================
# TmemOResource — O accumulator in TMEM, UmmaProducerAsync pipeline
# =====================================================================


@dataclass(kw_only=True)
class TmemOResource(HighThroughputMlaResource):
    """TMEM O accumulator.  Producer: Mma (PV MMA).  Consumer: Correction.

    Pipeline: UmmaProducerAsync, 1 stage.
    """

    tmem_base_addr: Any = None
    tmem_corr_ref: Any = None  # Reference to TmemCorrResource for correction data
    smem_p: Any = None  # SMEM P for PV MMA
    cfg: cutlass.Constexpr = field(default_factory=MlaDecodeConfig)
    cta_rank: Any = field(init=False, default=None)
    is_leader: Any = field(init=False, default=None)

    @producer_work
    @cute.jit
    def pv_mma(
        self,
        stage_info: StageInfo,
        *,
        desc_p_base,
        desc_v_base,
        v_subtile_idx: cutlass.Constexpr[int],
        is_tail: cutlass.Constexpr[bool] = False,
    ) -> None:
        """Issue one PV MMA sub-tile.
        Only leader CTA issues MMA (2CTA UMMA principle).

        Descriptor computation is hoisted OUTSIDE the leader-CTA gate so
        ptxas keeps values in uniform registers.
        """
        cfg = self.cfg

        idesc_pv = cprims.Tcgen05InstrDesc.build(
            c_dtype=Float32,
            a_dtype=qkv_dtype(cfg),
            b_dtype=qkv_dtype(cfg),
            n_dim=cfg.mma_pv_tiler[1],
            m_dim=cfg.mma_pv_tiler[0],
            b_major=1,
        )
        mma_kind = mma_kind_for_qkv(cfg)
        cta_group = prims.CTAGroup.CTA_2
        is_leader_cta = (
            cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster()) == 0
        )

        sp_copy_elems = cutlass.const_expr(
            cfg.mma_pv_tiler[0] // cfg.num_mma_ctas * cfg.mma_pv_tiler[2]
        )
        desc_p_delta = cutlass.const_expr(sp_copy_elems // 8)
        if cutlass.const_expr(cfg.is_fp8_qkv()):
            desc_p_delta = cutlass.const_expr(256)
        sv_copy_elems = cutlass.const_expr(
            cfg.mma_pv_tiler[1] // cfg.num_mma_ctas * cfg.mma_pv_tiler[2]
        )
        desc_v_delta = cutlass.const_expr(sv_copy_elems * cfg.qkv_dtype_bytes // 16)

        pv_k_block_count = cutlass.const_expr(
            cfg.mma_pv_tiler[2] // mma_k_step_for_qkv(cfg)
        )

        # Match the producer's physical V-stage decomposition. Each stage
        # carries two adjacent K32 slices for one D256 output panel.
        pv_j = v_subtile_idx // cfg.kv_subtiles_per_stage
        token_partition = v_subtile_idx % cfg.kv_subtiles_per_stage
        tmem_o_base_addr = self.tmem_base_addr + cfg.tmem_o_offset
        tmem_o_addr = tmem_o_base_addr + pv_j * 128

        for stage_subtile_idx in cutlass.range_constexpr(cfg.kv_subtiles_per_stage):
            pv_i = token_partition * cfg.kv_subtiles_per_stage + stage_subtile_idx
            desc_p = desc_p_base + pv_i * desc_p_delta
            desc_v = desc_v_base + stage_subtile_idx * desc_v_delta

            # Clear O for each output-N panel on its first PV K slice, then
            # accumulate the remaining slices and subsequent sequence tiles.
            scale_d_pv = Boolean(True)
            if cutlass.const_expr(pv_i == 0):
                if cutlass.const_expr(is_tail):
                    scale_d_pv = Boolean(
                        stage_info.loop_end > Int32(stage_info.loop_start)
                    )
                else:
                    scale_d_pv = Boolean(
                        stage_info.loop_offset != Int32(stage_info.loop_start)
                    )
            if is_leader_cta:
                for k_block in cutlass.range_constexpr(pv_k_block_count):
                    if prims.elect_sync():
                        prims.tcgen05_mma(
                            mma_kind,
                            cta_group,
                            prims.make_tmem_ptr(tmem_o_addr, Float32),
                            desc_p + k_block * 2,
                            desc_v + k_block * (256 if cfg.is_fp8_qkv() else 128),
                            idesc_pv,
                            Boolean(scale_d_pv),
                        )
                    scale_d_pv = Boolean(True)

    @producer_work
    @cute.jit
    def pv_mma_n_major(
        self,
        stage_info: StageInfo,
        *,
        desc_p_base,
        desc_v_base,
        pv_n_idx: cutlass.Constexpr[int],
        pv_k_idx: cutlass.Constexpr[int],
        is_tail: cutlass.Constexpr[bool] = False,
    ) -> None:
        """Issue PV MMA in output-N-major order for per-slice O tokens."""
        cfg = self.cfg

        pv_j = pv_n_idx
        pv_i = pv_k_idx
        tmem_o_base_addr = self.tmem_base_addr + cfg.tmem_o_offset
        tmem_o_addr = tmem_o_base_addr + pv_j * 128

        idesc_pv = cprims.Tcgen05InstrDesc.build(
            c_dtype=Float32,
            a_dtype=qkv_dtype(cfg),
            b_dtype=qkv_dtype(cfg),
            n_dim=cfg.mma_pv_tiler[1],
            m_dim=cfg.mma_pv_tiler[0],
            b_major=1,
        )
        mma_kind = mma_kind_for_qkv(cfg)
        cta_group = prims.CTAGroup.CTA_2
        is_leader_cta = (
            cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster()) == 0
        )

        sp_copy_elems = cutlass.const_expr(
            cfg.mma_pv_tiler[0] // cfg.num_mma_ctas * cfg.mma_pv_tiler[2]
        )
        desc_p_delta = cutlass.const_expr(sp_copy_elems // 8)
        if cutlass.const_expr(cfg.is_fp8_qkv()):
            desc_p_delta = cutlass.const_expr(256)
        desc_p = desc_p_base + pv_i * desc_p_delta

        pv_k_block_count = cutlass.const_expr(
            cfg.mma_pv_tiler[2] // mma_k_step_for_qkv(cfg)
        )
        scale_d_pv = Boolean(True)
        if cutlass.const_expr(pv_i == 0):
            if cutlass.const_expr(is_tail):
                scale_d_pv = Boolean(stage_info.loop_end > Int32(stage_info.loop_start))
            else:
                scale_d_pv = Boolean(
                    stage_info.loop_offset != Int32(stage_info.loop_start)
                )

        if is_leader_cta:
            for k_block in cutlass.range_constexpr(pv_k_block_count):
                if prims.elect_sync():
                    prims.tcgen05_mma(
                        mma_kind,
                        cta_group,
                        prims.make_tmem_ptr(tmem_o_addr, Float32),
                        desc_p + k_block * 2,
                        desc_v_base + k_block * (256 if cfg.is_fp8_qkv() else 128),
                        idesc_pv,
                        Boolean(scale_d_pv),
                    )
                scale_d_pv = Boolean(True)

    @consumer_work
    @cute.jit
    def rescale_o(
        self, stage_info: StageInfo, *, correction_factor, no_correction
    ) -> None:
        """Rescale O in-place in TMEM by correction_factor."""
        cfg = self.cfg
        tidx = cute.arch.thread_idx()[0]

        # Use local tidx within 4-warp correction group (matching bare-metal)
        local_tidx = tidx % (cfg.num_compute_warps * cfg.threads_per_warp)
        tmem_warp_row_id = (
            self.tmem_base_addr
            + (local_tidx >> WARP_LANE_SHIFT) * TCGEN05_32B_REGS_PER_LOAD
        )
        tmem_raw_addr = (tmem_warp_row_id << 16) | cfg.tmem_o_offset

        t2r_shape = TCGEN05_32B_SHAPE
        num_tmem_ops = 4  # 4 loads x 32 = 128 elements per iter_n

        skip_correction = prims.vote_sync(
            cute.arch.FULL_MASK,
            no_correction == 1,
            prims.VoteSync.ALL,
        )

        if not skip_correction:
            for iter_n in cutlass.range_constexpr(cfg.iterations_pv_n):
                tmem_addr_offset = iter_n * (cfg.mma_pv_tiler[1] // cfg.warps_in_n)
                for idx in cutlass.range_constexpr(num_tmem_ops):
                    curr_addr = (
                        tmem_raw_addr
                        + tmem_addr_offset
                        + idx * TCGEN05_32B_REGS_PER_LOAD
                    )
                    tmem_ptr = prims.make_tmem_ptr(curr_addr, Float32)
                    chunk = prims.tcgen05_ld(
                        t2r_shape, tmem_ptr, num=TCGEN05_32B_REGS_PER_LOAD
                    )
                    scaled = chunk * cutlass.full_like(chunk, correction_factor)
                    prims.tcgen05_st(t2r_shape, tmem_ptr, scaled)

        prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
        cute.arch.fence_view_async_tmem_store()

    @consumer_work
    @cute.jit
    def rescale_o_slice(
        self,
        stage_info: StageInfo,
        *,
        correction_factor,
        no_correction,
        iter_n: cutlass.Constexpr[int],
    ) -> None:
        """Rescale one PV output-N slice after its O token is ready."""
        cfg = self.cfg
        tidx = cute.arch.thread_idx()[0]
        local_tidx = tidx % (cfg.num_compute_warps * cfg.threads_per_warp)
        tmem_warp_row_id = (
            self.tmem_base_addr
            + (local_tidx >> WARP_LANE_SHIFT) * TCGEN05_32B_REGS_PER_LOAD
        )
        tmem_raw_addr = (tmem_warp_row_id << 16) | (
            cfg.tmem_o_offset + iter_n * (cfg.mma_pv_tiler[1] // cfg.warps_in_n)
        )

        t2r_shape = TCGEN05_32B_SHAPE
        num_tmem_ops = 4
        skip_correction = prims.vote_sync(
            cute.arch.FULL_MASK,
            no_correction == 1,
            prims.VoteSync.ALL,
        )
        if not skip_correction:
            for idx in cutlass.range_constexpr(num_tmem_ops):
                curr_addr = tmem_raw_addr + idx * TCGEN05_32B_REGS_PER_LOAD
                tmem_ptr = prims.make_tmem_ptr(curr_addr, Float32)
                chunk = prims.tcgen05_ld(
                    t2r_shape, tmem_ptr, num=TCGEN05_32B_REGS_PER_LOAD
                )
                scaled = chunk * cutlass.full_like(chunk, correction_factor)
                prims.tcgen05_st(t2r_shape, tmem_ptr, scaled)

        prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
        cute.arch.fence_view_async_tmem_store()


# =====================================================================
# GmemOResource — Output to GMEM (no pipeline)
# =====================================================================


@dataclass(kw_only=True)
class GmemOResource(HighThroughputMlaResource):
    """GMEM output.  Producer: Correction (epilogue store).  No pipeline."""

    output: Any = None
    partial_output: Any = None
    lse: Any = None
    partial_lse: Any = None
    tmem_o_ref: Any = None  # Reference to TmemOResource for tmem_base_addr
    tmem_corr_ref: Any = None  # Reference to TmemCorrResource for correction data
    output_scale: Any = None
    softmax_scale_log2: Any = None
    smem_exchange: Any = None  # SMEM for row_sum exchange (as Int32 base addr)
    split_kv: Any = None
    cu_seqlens_q: Any = None
    groups_tokens_heads_q_ratio: cutlass.Constexpr[int] = 1
    logical_num_heads_q: cutlass.Constexpr[int] = 128
    logical_seq_len_q: cutlass.Constexpr[int] = 1
    cfg: cutlass.Constexpr = field(default_factory=MlaDecodeConfig)

    @cute.jit
    def _query_row_state(self, effective_head_idx, effective_seq_group_idx, batch_idx):
        """Map one groups_tokens_heads_q epilogue row to public storage."""
        return groups_tokens_heads_q_row_state(
            effective_head_idx,
            effective_seq_group_idx,
            self.groups_tokens_heads_q_ratio,
            self.logical_num_heads_q,
            self.logical_seq_len_q,
            self.cu_seqlens_q,
            batch_idx,
        )

    @producer_work
    @cute.jit
    def epilogue_store(self, stage_info: StageInfo) -> None:
        """Full epilogue: load O from TMEM, normalize by row_sum, store to GMEM.

        Steps:
        1. Exchange row_sum across warp pairs via SMEM
        2. Load O from TMEM
        3. Normalize by output_scale / row_sum
        4. Store to GMEM
        5. Compute and store LSE
        """
        cfg = self.cfg
        tidx = cute.arch.thread_idx()[0]
        # Use work_tile's blk_coord (updated by persistent loop) instead of
        # self.blk_coord (captured at construction time) so that each work
        # tile writes to the correct output location.
        blk_coord = stage_info.work_tile.tile_idx

        row_sum = self.tmem_corr_ref.final_row_stats[0]
        row_max = self.tmem_corr_ref.final_row_stats[1]

        # Exchange row_sum between warp pairs (0,2) and (1,3) via SMEM
        # Use local thread index within the 4-warp correction group,
        # matching bare-metal: tidx % (num_compute_warps * threads_per_warp)
        num_compute_threads = cfg.num_compute_warps * cfg.threads_per_warp
        local_tidx = tidx % num_compute_threads
        smem_ex_ptr = cutlass.inttoptr(
            self.smem_exchange + local_tidx * 4,
            3,
            Float32,
        )
        smem_ex_ptr.store(row_sum)
        prims.barrier_cta_sync(
            cfg.epilogue_sync_bar_id, thread_count=cfg.epilogue_sync_threads
        )
        peer_idx = (local_tidx + 64) % num_compute_threads
        peer_ptr = cutlass.inttoptr(
            self.smem_exchange + peer_idx * 4,
            3,
            Float32,
        )
        row_sum = row_sum + peer_ptr.load()

        # TMEM address for O — use local tidx within 4-warp correction group
        tmem_base_addr = self.tmem_o_ref.tmem_base_addr
        tmem_warp_row_id = (
            tmem_base_addr + (local_tidx >> WARP_LANE_SHIFT) * TCGEN05_32B_REGS_PER_LOAD
        )
        tmem_raw_addr = (tmem_warp_row_id << 16) | cfg.tmem_o_offset

        t2r_shape = TCGEN05_32B_SHAPE
        num_tmem_loads = 4  # 4 x 32 = 128 elements per iter_n

        # Per-thread O indexing — use local tidx
        tile_h = cfg.mma_pv_tiler[0] // cfg.num_mma_ctas  # 64
        tile_d = cfg.mma_pv_tiler[1]  # 256
        tidx_g = local_tidx & EPILOGUE_THREAD_TILE_MASK
        g_i = tidx_g & EPILOGUE_ROW_MASK
        g_j = (tidx_g >> EPILOGUE_COLUMN_GROUP_SHIFT) * EPILOGUE_THREAD_TILE_THREADS

        # Public O/LSE remain in logical coordinates. Split-KV partials retain
        # the groups_tokens_heads_q scheduler/workspace coordinates until final reduction.
        logical_num_heads_q = Int32(self.logical_num_heads_q)
        effective_num_heads_q = Int32(
            self.logical_num_heads_q * self.groups_tokens_heads_q_ratio
        )
        D = cfg.latent_dim
        head_tile_idx = blk_coord[0]
        seq_q_idx = blk_coord[1]
        batch_idx = blk_coord[2]
        split_kv_idx = blk_coord[3]
        effective_head_idx = head_tile_idx * tile_h + g_i
        (
            storage_flat_query_row,
            logical_head_idx,
            logical_q_idx,
            _,
            query_is_valid,
        ) = self._query_row_state(effective_head_idx, seq_q_idx, batch_idx)

        # Fully masked split rows can occur when padded query rows have
        # different causal K lengths.  Store zero O and -inf LSE for those
        # rows so the split-KV reduction gives them zero weight.
        row_has_values = row_sum > Float32(0)
        safe_row_sum = row_sum if row_has_values else Float32(1)
        norm_scale = self.output_scale * cute.math.rcp(safe_row_sum, approx=True)

        for iter_n in cutlass.range_constexpr(cfg.iterations_pv_n):
            # Load O from TMEM
            qk_acc_regs = cutlass.Array(Float32, 128, space=cutlass.AddressSpace.rmem)
            tmem_raw_addr_n = tmem_raw_addr + (
                iter_n * (cfg.mma_pv_tiler[1] // cfg.warps_in_n)
            )
            for load_idx in cutlass.range_constexpr(num_tmem_loads):
                curr_addr = tmem_raw_addr_n + load_idx * TCGEN05_32B_REGS_PER_LOAD
                tmem_ptr = prims.make_tmem_ptr(curr_addr, Float32)
                loaded = prims.tcgen05_ld(
                    t2r_shape, tmem_ptr, num=TCGEN05_32B_REGS_PER_LOAD
                )
                qk_acc_regs.store(loaded, load_idx * TCGEN05_32B_REGS_PER_LOAD)

            # Normalize: O = O * output_scale / row_sum
            for i in cutlass.range_constexpr(0, 128, 2):
                scaled = mul_packed_f32x2(
                    (qk_acc_regs[i], qk_acc_regs[i + 1]),
                    (norm_scale, norm_scale),
                )
                qk_acc_regs[i] = scaled[0]
                qk_acc_regs[i + 1] = scaled[1]

            # Store O to GMEM
            if effective_head_idx < effective_num_heads_q and query_is_valid:
                if cutlass.const_expr(self.partial_output is not None):
                    # Split-KV partial O uses BF16 workspace storage.  LSE and
                    # the eventual cross-split accumulation remain FP32.
                    S_q = (
                        cutlass.Int32(self.partial_output.shape[3])
                        if self.partial_output is not None
                        else Int32(1)
                    )
                    o_base_ptr = (
                        self.partial_output.iterator.raw_ptr()
                        + Int64(effective_head_idx) * Int64(self.split_kv) * Int64(D)
                        + Int64(split_kv_idx) * Int64(D)
                        + Int64(seq_q_idx)
                        * Int64(self.split_kv)
                        * Int64(effective_num_heads_q)
                        * Int64(D)
                        + Int64(batch_idx)
                        * Int64(effective_num_heads_q)
                        * Int64(self.split_kv)
                        * Int64(S_q)
                        * Int64(D)
                    )
                    output_base = o_base_ptr + iter_n * tile_d + g_j
                    for load_idx in cutlass.range_constexpr(num_tmem_loads):
                        for j in cutlass.range_constexpr(4):
                            offset = (
                                load_idx * TCGEN05_32B_REGS_PER_LOAD
                                + j * BF16_OUTPUT_VECTOR_ELEMENTS
                            )
                            vec_f32 = qk_acc_regs.load(
                                offset, BF16_OUTPUT_VECTOR_ELEMENTS
                            )
                            vec_partial = vec_f32.to(cutlass.BFloat16)
                            (
                                output_base
                                + load_idx * TCGEN05_32B_REGS_PER_LOAD
                                + j * BF16_OUTPUT_VECTOR_ELEMENTS
                            ).nvvm_store_ext(
                                vec_partial,
                                evict="noallocate",
                            )
                else:
                    # 16-bit output (split_kv == 1, direct output)
                    if cutlass.const_expr(self.cu_seqlens_q is not None):
                        o_base_ptr = self.output.iterator.raw_ptr() + Int64(
                            storage_flat_query_row
                        ) * Int64(D)
                    else:
                        S_q = (
                            cutlass.Int32(self.output.shape[2])
                            if self.output is not None
                            else Int32(1)
                        )
                        o_base_ptr = (
                            self.output.iterator.raw_ptr()
                            + Int64(logical_head_idx) * Int64(D)
                            + Int64(logical_q_idx)
                            * Int64(logical_num_heads_q)
                            * Int64(D)
                            + Int64(batch_idx)
                            * Int64(logical_num_heads_q)
                            * Int64(D)
                            * Int64(S_q)
                        )
                    output_base = o_base_ptr + iter_n * tile_d + g_j
                    for load_idx in cutlass.range_constexpr(num_tmem_loads):
                        for j in cutlass.range_constexpr(2):
                            offset = (
                                load_idx * TCGEN05_32B_REGS_PER_LOAD
                                + j * FP8_OUTPUT_VECTOR_ELEMENTS
                            )
                            vec_f32 = qk_acc_regs.load(
                                offset, FP8_OUTPUT_VECTOR_ELEMENTS
                            )
                            if cutlass.const_expr(cfg.use_fp8_output == 1):
                                packed_o = cutlass.Array(
                                    Int32,
                                    PACKED_FP8_OUTPUT_REGS,
                                    space=cutlass.AddressSpace.rmem,
                                )
                                for pack_idx in cutlass.range_constexpr(
                                    PACKED_FP8_OUTPUT_REGS
                                ):
                                    pack_offset = pack_idx * PACKED_FP8_OUTPUT_REGS
                                    packed_o[pack_idx] = pack_float4_to_fp8_e4m3(
                                        vec_f32[pack_offset],
                                        vec_f32[pack_offset + 1],
                                        vec_f32[pack_offset + 2],
                                        vec_f32[pack_offset + 3],
                                    )
                                raw_ptr = cutlass.inttoptr(
                                    (
                                        output_base
                                        + load_idx * TCGEN05_32B_REGS_PER_LOAD
                                        + j * FP8_OUTPUT_VECTOR_ELEMENTS
                                    ).toint(Int64),
                                    mem_space=1,
                                    dtype=Int32,
                                )
                                raw_ptr.store(
                                    packed_o.load(0, PACKED_FP8_OUTPUT_REGS),
                                    alignment=16,
                                )
                            else:
                                vec_o = vec_f32.to(output_dtype(self.cfg))
                                (
                                    output_base
                                    + load_idx * TCGEN05_32B_REGS_PER_LOAD
                                    + j * FP8_OUTPUT_VECTOR_ELEMENTS
                                ).nvvm_store_ext(
                                    vec_o,
                                    evict="noallocate",
                                )

        # Compute and store LSE in the same row-sum domain used by P.
        lse_row_sum = row_sum
        if cutlass.const_expr(cfg.is_fp8_qkv()):
            lse_row_sum = lse_row_sum * fp8_quant_scale_rcp()
        lse = (
            cute.math.log2(lse_row_sum, fastmath=True)
            + self.softmax_scale_log2 * row_max
            if row_has_values
            else Float32(-Float32.inf)
        )

        # Use local_tidx (0..127 within correction warpgroup) for LSE
        # indexing, not global tidx (which is 128..255 for correction warps).
        lse_tidx = local_tidx
        if lse_tidx < tile_h:
            effective_lse_head_idx = head_tile_idx * tile_h + lse_tidx
            (
                storage_flat_lse_row,
                logical_lse_head_idx,
                logical_lse_q_idx,
                _,
                lse_query_is_valid,
            ) = self._query_row_state(effective_lse_head_idx, seq_q_idx, batch_idx)
            if effective_lse_head_idx < effective_num_heads_q and lse_query_is_valid:
                if cutlass.const_expr(self.partial_lse is not None):
                    S_q = (
                        cutlass.Int32(self.partial_lse.shape[2])
                        if self.partial_lse is not None
                        else Int32(1)
                    )
                    lse_base_ptr = (
                        self.partial_lse.iterator.raw_ptr()
                        + Int64(effective_lse_head_idx) * Int64(self.split_kv)
                        + Int64(split_kv_idx)
                        + Int64(seq_q_idx)
                        * Int64(effective_num_heads_q)
                        * Int64(self.split_kv)
                        + Int64(batch_idx)
                        * Int64(effective_num_heads_q)
                        * Int64(self.split_kv)
                        * Int64(S_q)
                    )
                    lse_base_ptr.store(lse)
                elif cutlass.const_expr(self.lse is not None):
                    if cutlass.const_expr(self.cu_seqlens_q is not None):
                        lse_base_ptr = (
                            self.lse.iterator.raw_ptr() + storage_flat_lse_row
                        )
                    else:
                        S_q = (
                            cutlass.Int32(self.lse.shape[1])
                            if self.lse is not None
                            else Int32(1)
                        )
                        lse_base_ptr = (
                            self.lse.iterator.raw_ptr()
                            + Int64(logical_lse_head_idx)
                            + Int64(logical_lse_q_idx) * Int64(logical_num_heads_q)
                            + Int64(batch_idx) * Int64(logical_num_heads_q) * Int64(S_q)
                        )
                    lse_base_ptr.store(lse)

        prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
        cute.arch.fence_view_async_tmem_load()

    @producer_work
    @cute.jit
    def epilogue_store_slice(
        self,
        stage_info: StageInfo,
        *,
        row_sum,
        row_max,
        iter_n: cutlass.Constexpr[int],
    ) -> None:
        """Store one output-N slice after its O pipeline token is ready."""
        cfg = self.cfg
        tidx = cute.arch.thread_idx()[0]
        blk_coord = stage_info.work_tile.tile_idx

        num_compute_threads = cfg.num_compute_warps * cfg.threads_per_warp
        local_tidx = tidx % num_compute_threads

        tmem_base_addr = self.tmem_o_ref.tmem_base_addr
        tmem_warp_row_id = (
            tmem_base_addr + (local_tidx >> WARP_LANE_SHIFT) * TCGEN05_32B_REGS_PER_LOAD
        )
        tmem_raw_addr = (tmem_warp_row_id << 16) | (
            cfg.tmem_o_offset + iter_n * (cfg.mma_pv_tiler[1] // cfg.warps_in_n)
        )

        t2r_shape = TCGEN05_32B_SHAPE
        num_tmem_loads = 4
        qk_acc_regs = cutlass.Array(Float32, 128, space=cutlass.AddressSpace.rmem)
        for load_idx in cutlass.range_constexpr(num_tmem_loads):
            curr_addr = tmem_raw_addr + load_idx * TCGEN05_32B_REGS_PER_LOAD
            tmem_ptr = prims.make_tmem_ptr(curr_addr, Float32)
            loaded = prims.tcgen05_ld(
                t2r_shape, tmem_ptr, num=TCGEN05_32B_REGS_PER_LOAD
            )
            qk_acc_regs.store(loaded, load_idx * TCGEN05_32B_REGS_PER_LOAD)

        row_has_values = row_sum > Float32(0)
        safe_row_sum = row_sum if row_has_values else Float32(1)
        norm_scale = self.output_scale * cute.math.rcp(safe_row_sum, approx=True)
        for i in cutlass.range_constexpr(0, 128, 2):
            scaled = mul_packed_f32x2(
                (qk_acc_regs[i], qk_acc_regs[i + 1]),
                (norm_scale, norm_scale),
            )
            qk_acc_regs[i] = scaled[0]
            qk_acc_regs[i + 1] = scaled[1]

        tile_h = cfg.mma_pv_tiler[0] // cfg.num_mma_ctas
        tile_d = cfg.mma_pv_tiler[1]
        tidx_g = local_tidx & EPILOGUE_THREAD_TILE_MASK
        g_i = tidx_g & EPILOGUE_ROW_MASK
        g_j = (tidx_g >> EPILOGUE_COLUMN_GROUP_SHIFT) * EPILOGUE_THREAD_TILE_THREADS

        logical_num_heads_q = Int32(self.logical_num_heads_q)
        effective_num_heads_q = Int32(
            self.logical_num_heads_q * self.groups_tokens_heads_q_ratio
        )
        D = cfg.latent_dim
        head_tile_idx = blk_coord[0]
        seq_q_idx = blk_coord[1]
        batch_idx = blk_coord[2]
        split_kv_idx = blk_coord[3]
        effective_head_idx = head_tile_idx * tile_h + g_i
        (
            storage_flat_query_row,
            logical_head_idx,
            logical_q_idx,
            _,
            query_is_valid,
        ) = self._query_row_state(effective_head_idx, seq_q_idx, batch_idx)

        if effective_head_idx < effective_num_heads_q and query_is_valid:
            if cutlass.const_expr(self.partial_output is not None):
                S_q = (
                    cutlass.Int32(self.partial_output.shape[3])
                    if self.partial_output is not None
                    else Int32(1)
                )
                o_base_ptr = (
                    self.partial_output.iterator.raw_ptr()
                    + Int64(effective_head_idx) * Int64(self.split_kv) * Int64(D)
                    + Int64(split_kv_idx) * Int64(D)
                    + Int64(seq_q_idx)
                    * Int64(self.split_kv)
                    * Int64(effective_num_heads_q)
                    * Int64(D)
                    + Int64(batch_idx)
                    * Int64(effective_num_heads_q)
                    * Int64(self.split_kv)
                    * Int64(S_q)
                    * Int64(D)
                )
                output_base = o_base_ptr + iter_n * tile_d + g_j
                for load_idx in cutlass.range_constexpr(num_tmem_loads):
                    for j in cutlass.range_constexpr(4):
                        offset = (
                            load_idx * TCGEN05_32B_REGS_PER_LOAD
                            + j * BF16_OUTPUT_VECTOR_ELEMENTS
                        )
                        vec_f32 = qk_acc_regs.load(offset, BF16_OUTPUT_VECTOR_ELEMENTS)
                        vec_partial = vec_f32.to(cutlass.BFloat16)
                        (
                            output_base
                            + load_idx * TCGEN05_32B_REGS_PER_LOAD
                            + j * BF16_OUTPUT_VECTOR_ELEMENTS
                        ).nvvm_store_ext(
                            vec_partial,
                            evict="noallocate",
                        )
            else:
                if cutlass.const_expr(self.cu_seqlens_q is not None):
                    o_base_ptr = self.output.iterator.raw_ptr() + Int64(
                        storage_flat_query_row
                    ) * Int64(D)
                else:
                    S_q = (
                        cutlass.Int32(self.output.shape[2])
                        if self.output is not None
                        else Int32(1)
                    )
                    o_base_ptr = (
                        self.output.iterator.raw_ptr()
                        + Int64(logical_head_idx) * Int64(D)
                        + Int64(logical_q_idx) * Int64(logical_num_heads_q) * Int64(D)
                        + Int64(batch_idx)
                        * Int64(logical_num_heads_q)
                        * Int64(D)
                        * Int64(S_q)
                    )
                output_base = o_base_ptr + iter_n * tile_d + g_j
                for load_idx in cutlass.range_constexpr(num_tmem_loads):
                    for j in cutlass.range_constexpr(2):
                        offset = (
                            load_idx * TCGEN05_32B_REGS_PER_LOAD
                            + j * FP8_OUTPUT_VECTOR_ELEMENTS
                        )
                        vec_f32 = qk_acc_regs.load(offset, FP8_OUTPUT_VECTOR_ELEMENTS)
                        if cutlass.const_expr(cfg.use_fp8_output == 1):
                            packed_o = cutlass.Array(
                                Int32,
                                PACKED_FP8_OUTPUT_REGS,
                                space=cutlass.AddressSpace.rmem,
                            )
                            for pack_idx in cutlass.range_constexpr(
                                PACKED_FP8_OUTPUT_REGS
                            ):
                                pack_offset = pack_idx * PACKED_FP8_OUTPUT_REGS
                                packed_o[pack_idx] = pack_float4_to_fp8_e4m3(
                                    vec_f32[pack_offset],
                                    vec_f32[pack_offset + 1],
                                    vec_f32[pack_offset + 2],
                                    vec_f32[pack_offset + 3],
                                )
                            raw_ptr = cutlass.inttoptr(
                                (
                                    output_base
                                    + load_idx * TCGEN05_32B_REGS_PER_LOAD
                                    + j * FP8_OUTPUT_VECTOR_ELEMENTS
                                ).toint(Int64),
                                mem_space=1,
                                dtype=Int32,
                            )
                            raw_ptr.store(
                                packed_o.load(0, PACKED_FP8_OUTPUT_REGS),
                                alignment=16,
                            )
                        else:
                            vec_o = vec_f32.to(output_dtype(self.cfg))
                            (
                                output_base
                                + load_idx * TCGEN05_32B_REGS_PER_LOAD
                                + j * FP8_OUTPUT_VECTOR_ELEMENTS
                            ).nvvm_store_ext(
                                vec_o,
                                evict="noallocate",
                            )

        if cutlass.const_expr(iter_n == 0):
            lse_row_sum = row_sum
            if cutlass.const_expr(cfg.is_fp8_qkv()):
                lse_row_sum = lse_row_sum * fp8_quant_scale_rcp()
            lse = (
                cute.math.log2(lse_row_sum, fastmath=True)
                + self.softmax_scale_log2 * row_max
                if row_has_values
                else Float32(-Float32.inf)
            )
            lse_tidx = local_tidx
            if lse_tidx < tile_h:
                effective_lse_head_idx = head_tile_idx * tile_h + lse_tidx
                (
                    storage_flat_lse_row,
                    logical_lse_head_idx,
                    logical_lse_q_idx,
                    _,
                    lse_query_is_valid,
                ) = self._query_row_state(effective_lse_head_idx, seq_q_idx, batch_idx)
                if (
                    effective_lse_head_idx < effective_num_heads_q
                    and lse_query_is_valid
                ):
                    if cutlass.const_expr(self.partial_lse is not None):
                        S_q = (
                            cutlass.Int32(self.partial_lse.shape[2])
                            if self.partial_lse is not None
                            else Int32(1)
                        )
                        lse_base_ptr = (
                            self.partial_lse.iterator.raw_ptr()
                            + Int64(effective_lse_head_idx) * Int64(self.split_kv)
                            + Int64(split_kv_idx)
                            + Int64(seq_q_idx)
                            * Int64(effective_num_heads_q)
                            * Int64(self.split_kv)
                            + Int64(batch_idx)
                            * Int64(effective_num_heads_q)
                            * Int64(self.split_kv)
                            * Int64(S_q)
                        )
                        lse_base_ptr.store(lse)
                    elif cutlass.const_expr(self.lse is not None):
                        if cutlass.const_expr(self.cu_seqlens_q is not None):
                            lse_base_ptr = (
                                self.lse.iterator.raw_ptr() + storage_flat_lse_row
                            )
                        else:
                            S_q = (
                                cutlass.Int32(self.lse.shape[1])
                                if self.lse is not None
                                else Int32(1)
                            )
                            lse_base_ptr = (
                                self.lse.iterator.raw_ptr()
                                + Int64(logical_lse_head_idx)
                                + Int64(logical_lse_q_idx) * Int64(logical_num_heads_q)
                                + Int64(batch_idx)
                                * Int64(logical_num_heads_q)
                                * Int64(S_q)
                            )
                        lse_base_ptr.store(lse)

        prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
        cute.arch.fence_view_async_tmem_load()
