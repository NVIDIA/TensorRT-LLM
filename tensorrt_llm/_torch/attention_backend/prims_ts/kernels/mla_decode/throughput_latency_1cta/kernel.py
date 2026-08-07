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

"""Throughput-latency 1CTA MLA TS schedule and kernel wrapper.

The throughput-latency 1CTA policy builds one captured task graph per concrete
``MlaConfig``. It supports the non-persistent baseline,
static-persistent, CLC-persistent, and GMEM split-KV reduction profiles selected
by ``kernel_policy.py``. Python-side launch validation catches unsupported
profile/split/workspace combinations before the JIT body constructs tensor
layouts that depend on those constants.
"""

from dataclasses import dataclass, replace as dataclass_replace

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass.cute.testing import assert_ as runtime_assert
from cutlass import Float32, Int32, Int64
from ...tensor_map import (
    create_tensor_map_ragged_from_tensor,
    create_tensor_map_tiled_from_view,
)
from cutlass.experimental import cuda
from cutlass.experimental import primitives as prims
from cutlass.utils.static_persistent_tile_scheduler import WorkTileInfo

from cutlass.experimental.task_scheduling.enums import PipelineType, SignalingThreads
from cutlass.experimental.task_scheduling.memory import (
    ResourceContext,
    SmemAllocation,
    SmemAllocator,
    TmemAllocator,
)
from cutlass.experimental.task_scheduling.resources import (
    PipelineConfig,
    TileSchedulerConfig,
    WorkQueue,
)
from cutlass.experimental.task_scheduling.task_manager import TaskManager

from .config import (
    MlaConfig,
    make_throughput_latency_mla_config,
    resolve_throughput_latency_groups_tokens_heads_q_shape,
)
from .resources import (
    SmemKvResource,
    SmemPageOffsetsResource,
    SmemQResource,
    TmemCorrResource,
    TmemOResource,
    TmemPResource,
    SmemPResource,
    TmemSKeepsResource,
    TmemSResource,
    TmemSoftmaxGlobalResource,
    TmemSoftmaxLocalResource,
    ScheduleTokenThrottleResource,
)
from .tasks import (
    MlaDecodeTask,
    create_throughput_latency_correction_task,
    create_keeps_mma_ab_correction_task,
    create_keeps_mma_ab_mma_task,
    create_keeps_mma_ab_softmax_task,
    create_load_page_offsets_task,
    create_padding_task,
    create_throughput_latency_scheduler_task,
    create_throughput_latency_load_task,
    create_throughput_latency_mma_task,
    create_throughput_latency_softmax0_task,
    create_throughput_latency_softmax1_task,
)
from .parallel_reduction import (
    PARALLEL_GMEM_REDUCTION_ELEMENTS_PER_SLICE,
    PARALLEL_GMEM_REDUCTION_SWAPS_ELEMENTS_PER_SLICE,
    parallel_gmem_reduction_base_clusters,
    parallel_gmem_reduction_launch_shape,
    parallel_gmem_reduction_threads,
    run_parallel_gmem_reduction_kernel,
    supports_parallel_gmem_reduction,
)
from .reduction import gmem_reduction_launch_shape, run_gmem_reduction_kernel
from ..helpers.constants import TMEM_LIFECYCLE_BARRIER_ID
from ..helpers.mask import MaskType, normalize_mask_type
from ..helpers.query import groups_tokens_heads_q_group_count
from ..helpers.tile import (
    runtime_query_tile_is_active,
    runtime_split_pruning_is_profitable,
    runtime_split_tile_is_active,
    runtime_work_tile_is_active,
)
from ..parallel_reduction_topology import (
    choose_q64_parallel_reducer_cluster_size,
    make_balanced_parallel_reduction_topology,
    validate_parallel_reduction_workspace,
)


# Softmax uses exp2, so natural-scale scores are multiplied by log2(e).
LOG2_E = 1.4426950408889634


@cute.jit
def _publish_neutral_standalone_partial(
    cfg,
    acc_o,
    acc_lse,
    batch_idx,
    cta_idx_q,
    cta_idx_kv,
    cta_idx_head_dim_v,
    head_base_idx,
):
    """Publish ``(O=0, LSE=-inf)`` for one pruned producer tile.

    Standalone reducers use the configured split geometry.  A runtime-pruned
    producer must therefore initialize its workspace slot before releasing the
    PDL-dependent reducer; otherwise a fixed-S reducer could consume stale
    partials left by an earlier graph replay.  Each thread clears aligned BF16
    vec8 fragments from this CTA's V slice.  The first V-slice CTA also
    publishes one neutral LSE per covered Q/head row.
    """

    thread_idx, _, _ = cute.arch.thread_idx()
    vectors_per_row = cfg.head_dim_per_cta_v // 8
    vectors_per_tile = cfg.tile_size_q * vectors_per_row
    vectors_per_thread = (
        vectors_per_tile + cfg.threads_per_cta - 1
    ) // cfg.threads_per_cta
    zero_vec = cutlass.Array(
        Int32,
        4,
        space=cutlass.AddressSpace.rmem,
    )
    for elem_idx in cutlass.range_constexpr(4):
        zero_vec[elem_idx] = Int32(0)

    for iter_idx in cutlass.range_constexpr(vectors_per_thread):
        vector_idx = thread_idx + Int32(iter_idx * cfg.threads_per_cta)
        if vector_idx < Int32(vectors_per_tile):
            local_row_idx = vector_idx // Int32(vectors_per_row)
            row_vector_idx = vector_idx - local_row_idx * Int32(vectors_per_row)
            head_idx = head_base_idx + local_row_idx
            if head_idx < Int32(cfg.num_heads_q):
                dim_idx = cta_idx_head_dim_v * Int32(
                    cfg.head_dim_per_cta_v
                ) + row_vector_idx * Int32(8)
                if dim_idx < Int32(cfg.head_dim_v):
                    elem_offset = (
                        Int64(batch_idx)
                        * Int64(
                            cfg.seq_len_q
                            * cfg.num_heads_q
                            * cfg.num_ctas_per_seq_kv
                            * cfg.head_dim_v
                        )
                        + Int64(cta_idx_q)
                        * Int64(
                            cfg.num_heads_q * cfg.num_ctas_per_seq_kv * cfg.head_dim_v
                        )
                        + Int64(head_idx)
                        * Int64(cfg.num_ctas_per_seq_kv * cfg.head_dim_v)
                        + Int64(cta_idx_kv) * Int64(cfg.head_dim_v)
                        + Int64(dim_idx)
                    )
                    dst_ptr = cutlass.inttoptr(
                        acc_o.iterator.raw_ptr().toint(Int64)
                        + elem_offset * Int64(cfg.partial_o_dtype_bytes),
                        mem_space=1,
                        dtype=Int32,
                    )
                    dst_ptr.store(
                        zero_vec.data_ptr().load(count=4, alignment=16),
                        alignment=16,
                    )

    if cta_idx_head_dim_v == Int32(0) and thread_idx < Int32(cfg.tile_size_q):
        head_idx = head_base_idx + thread_idx
        if head_idx < Int32(cfg.num_heads_q):
            lse_offset = (
                batch_idx
                * Int32(cfg.seq_len_q * cfg.num_heads_q * cfg.num_ctas_per_seq_kv)
                + cta_idx_q * Int32(cfg.num_heads_q * cfg.num_ctas_per_seq_kv)
                + head_idx * Int32(cfg.num_ctas_per_seq_kv)
                + cta_idx_kv
            )
            (acc_lse.iterator.raw_ptr() + lse_offset).store(Float32(-Float32.inf))


@cute.jit
def _persistent_work_tile_is_inactive(cfg, cache_seqs, cu_seqlens_q, work_tile):
    """Return whether a persistent Q tile is runtime padding."""

    cta_idx_q, _, batch_head_idx = work_tile.tile_idx
    batch_idx = Int32(batch_head_idx) // Int32(cfg.num_ctas_for_all_heads)
    return not runtime_work_tile_is_active(
        cfg,
        cache_seqs,
        cu_seqlens_q,
        batch_idx,
        cta_idx_q,
        Int32(0),
    )


@dataclass(kw_only=True)
class ThroughputLatencyMlaStaticWorkQueue(WorkQueue):
    """Static persistent scheduler for MLA tiles shaped as (cta_q, cta_head_dim, batch_head_tile)."""

    cfg: cutlass.Constexpr[MlaConfig] = None
    cache_seqs: object = None
    cu_seqlens_q: object = None
    enable_runtime_skip: cutlass.Constexpr[bool] = False

    def __init__(
        self,
        tile_scheduler_config: TileSchedulerConfig,
        cfg: cutlass.Constexpr[MlaConfig] = None,
        cache_seqs=None,
        cu_seqlens_q=None,
        **kwargs,
    ) -> None:
        WorkQueue.__init__(
            self,
            tile_scheduler_config=tile_scheduler_config,
            **kwargs,
        )
        self.cfg = cfg
        self.cache_seqs = cache_seqs
        self.cu_seqlens_q = cu_seqlens_q
        self.enable_runtime_skip = cu_seqlens_q is not None

    @cute.jit
    def skip_work_tile_if(self, work_tile: WorkTileInfo):
        """Skip packed-Q padding while retaining queue bookkeeping."""

        return _persistent_work_tile_is_inactive(
            self.cfg,
            self.cache_seqs,
            self.cu_seqlens_q,
            work_tile,
        )

    @cute.jit
    def _work_tile_from_linear(self, linear_idx: Int32) -> WorkTileInfo:
        """Decode a linear persistent index into a throughput-latency work tile."""
        ctas_q = Int32(self.cfg.num_ctas_per_seq_q)
        ctas_head_dim = Int32(self.cfg.num_ctas_per_head_dim)
        tiles_per_batch_head = ctas_q * ctas_head_dim
        batch_head_idx = linear_idx // tiles_per_batch_head
        in_batch_head_idx = linear_idx - batch_head_idx * tiles_per_batch_head
        cta_idx_head_dim = in_batch_head_idx // ctas_q
        cta_idx_q = in_batch_head_idx - cta_idx_head_dim * ctas_q
        total_tiles = (
            tiles_per_batch_head
            * Int32(self.cfg.batch_size)
            * Int32(self.cfg.num_ctas_for_all_heads)
        )
        return WorkTileInfo(
            (cta_idx_q, cta_idx_head_dim, batch_head_idx),
            linear_idx < total_tiles,
        )

    @cute.jit
    def _make_initial_work_tile(self) -> WorkTileInfo:
        """Return the initial work tile for the current CTA."""
        return self._work_tile_from_linear(Int32(cute.arch.block_idx()[2]))

    @cute.jit
    def initial_work_tile_info(self) -> WorkTileInfo:
        """Return the initial TS work-tile info."""
        return self._make_initial_work_tile()

    @cute.jit
    def _get_and_advance_work_tile_impl(
        self,
        stage_info,
    ) -> WorkTileInfo:
        """Advance the persistent work tile by one grid-stride step."""
        cta_idx_q, cta_idx_head_dim, batch_head_idx = stage_info.work_tile.tile_idx
        linear_idx = Int32(cta_idx_q) + Int32(self.cfg.num_ctas_per_seq_q) * (
            Int32(cta_idx_head_dim)
            + Int32(self.cfg.num_ctas_per_head_dim) * Int32(batch_head_idx)
        )
        next_linear_idx = linear_idx + Int32(cute.arch.grid_dim()[2])
        return self._work_tile_from_linear(next_linear_idx)


@dataclass(kw_only=True)
class ThroughputLatencyMlaClcWorkQueue(WorkQueue):
    """CLC work queue shim for throughput-latency 1CTA captured schedules."""

    cfg: cutlass.Constexpr[MlaConfig] = None
    cache_seqs: object = None
    cu_seqlens_q: object = None
    enable_runtime_skip: cutlass.Constexpr[bool] = False

    def __init__(
        self,
        tile_scheduler_config: TileSchedulerConfig,
        cfg: cutlass.Constexpr[MlaConfig] = None,
        cache_seqs=None,
        cu_seqlens_q=None,
        **kwargs,
    ) -> None:
        WorkQueue.__init__(
            self,
            tile_scheduler_config=tile_scheduler_config,
            **kwargs,
        )
        self.cfg = cfg
        self.cache_seqs = cache_seqs
        self.cu_seqlens_q = cu_seqlens_q
        self.enable_runtime_skip = cu_seqlens_q is not None

    @cute.jit
    def skip_work_tile_if(self, work_tile: WorkTileInfo):
        """Skip packed-Q padding while retaining queue bookkeeping."""

        return _persistent_work_tile_is_inactive(
            self.cfg,
            self.cache_seqs,
            self.cu_seqlens_q,
            work_tile,
        )


def _default_scales(scale_softmax_log2, output_scale):
    """Return default softmax and output scales for validation-only builds."""
    if scale_softmax_log2 is None:
        scale_softmax_log2 = Float32(1.0)
    if output_scale is None:
        output_scale = Float32(1.0)
    return scale_softmax_log2, output_scale


def _check_persistent_scheduler_modes(
    use_clc_dynamic_scheduler: bool,
    use_static_persistent_scheduler: bool,
) -> None:
    """Reject mutually exclusive persistent scheduler selections."""
    if use_clc_dynamic_scheduler and use_static_persistent_scheduler:
        raise ValueError(
            "throughput-latency 1CTA MLA cannot enable both CLC dynamic and static "
            "persistent schedulers"
        )


def _make_static_work_queue(
    cfg,
    tile_sched_params,
    cache_seqs,
    cu_seqlens_q,
    name: str,
):
    """Create the static persistent work queue shared by both 1CTA variants."""
    return ThroughputLatencyMlaStaticWorkQueue(
        tile_scheduler_config=TileSchedulerConfig.create_static_persistent_tile_scheduler_params(
            tile_scheduler_params=tile_sched_params,
        ),
        cfg=cfg,
        cache_seqs=cache_seqs,
        cu_seqlens_q=cu_seqlens_q,
        name=name,
    )


def _make_clc_work_queue_and_throttle(
    cfg,
    tile_sched_params,
    clc_response_ptr,
    cache_seqs,
    cu_seqlens_q,
    cta_layout_vmnk,
    load_group,
    scheduler_group,
    queue_name: str,
    throttle_name: str,
):
    """Create the CLC work queue and the load-to-scheduler throttle edge."""
    agent = pipeline.Agent
    schedule_token_throttle_pipeline_config = (
        PipelineConfig.create_async_async_pipeline_cfg(
            num_stages=2,
            producer_group=load_group,
            consumer_group=scheduler_group,
            cta_layout_vmnk=cta_layout_vmnk,
        )
    )
    schedule_token_pipeline_config = PipelineConfig.create_clc_fetch_async_pipeline_cfg(
        num_stages=2,
        num_bytes=16,
        producer_group=pipeline.CooperativeGroup(agent.Thread),
        consumer_group=pipeline.CooperativeGroup(
            agent.Thread,
            cfg.threads_per_cta,
        ),
        cta_layout_vmnk=cta_layout_vmnk,
    )
    return (
        ThroughputLatencyMlaClcWorkQueue(
            tile_scheduler_config=TileSchedulerConfig.create_clc_dynamic_persistent_tile_scheduler_params(
                tile_scheduler_params=tile_sched_params,
                response_ptr=clc_response_ptr,
            ),
            cfg=cfg,
            cache_seqs=cache_seqs,
            cu_seqlens_q=cu_seqlens_q,
            pipeline_config=schedule_token_pipeline_config,
            name=queue_name,
        ),
        ScheduleTokenThrottleResource(
            pipeline_config=schedule_token_throttle_pipeline_config,
            name=throttle_name,
        ),
    )


def build_throughput_latency_mla_task_manager(
    cfg: MlaConfig,
    *,
    total_kv_tiles: int = 4,
    use_page_offsets: bool = False,
    tma_desc_q_latent=None,
    tma_desc_q_rope=None,
    tma_desc_c_latent=None,
    tma_desc_c_rope=None,
    tma_desc_v=None,
    c_rope_tensor=None,
    page_offsets=None,
    cache_seqs=None,
    cu_seqlens_q=None,
    head_idx=None,
    batch_idx=None,
    cta_idx_q=None,
    cta_idx_kv=None,
    cta_idx_head_dim_v=None,
    scale_softmax_log2=None,
    output_scale=None,
    o_tensor=None,
    lse_tensor=None,
    acc_o_tensor=None,
    acc_lse_tensor=None,
    tile_sched_params=None,
    clc_response_ptr=None,
    use_clc_dynamic_scheduler: bool = False,
    use_static_persistent_scheduler: bool = False,
    verbose: bool = False,
    exhaustive_deadlock_race_check: bool = False,
) -> tuple[TaskManager, object]:
    """Build the throughput-latency 1CTA MLA TaskManager for validation or execution.

    The caller supplies either concrete tensors/descriptors for JIT execution or
    leaves them as ``None`` for schedule-only validation. Exactly one persistent
    scheduler mode may be enabled: CLC dynamic persistent or static persistent.
    GMEM split-KV profiles wire an extra correction/reduction path, while
    non-split profiles store O/LSE directly from the correction task.
    """

    scale_softmax_log2, output_scale = _default_scales(
        scale_softmax_log2,
        output_scale,
    )
    _check_persistent_scheduler_modes(
        use_clc_dynamic_scheduler,
        use_static_persistent_scheduler,
    )
    if cfg.kernel_variant == "keeps_mma_ab":
        return _make_keeps_mma_ab_task_graph(
            cfg,
            total_kv_tiles=total_kv_tiles,
            use_page_offsets=use_page_offsets,
            tma_desc_q_latent=tma_desc_q_latent,
            tma_desc_q_rope=tma_desc_q_rope,
            tma_desc_c_latent=tma_desc_c_latent,
            tma_desc_c_rope=tma_desc_c_rope,
            tma_desc_v=tma_desc_v,
            c_rope_tensor=c_rope_tensor,
            page_offsets=page_offsets,
            cache_seqs=cache_seqs,
            cu_seqlens_q=cu_seqlens_q,
            head_idx=head_idx,
            batch_idx=batch_idx,
            cta_idx_q=cta_idx_q,
            cta_idx_kv=cta_idx_kv,
            cta_idx_head_dim_v=cta_idx_head_dim_v,
            scale_softmax_log2=scale_softmax_log2,
            output_scale=output_scale,
            o_tensor=o_tensor,
            lse_tensor=lse_tensor,
            acc_o_tensor=acc_o_tensor,
            acc_lse_tensor=acc_lse_tensor,
            tile_sched_params=tile_sched_params,
            clc_response_ptr=clc_response_ptr,
            use_clc_dynamic_scheduler=use_clc_dynamic_scheduler,
            use_static_persistent_scheduler=use_static_persistent_scheduler,
            verbose=verbose,
            exhaustive_deadlock_race_check=exhaustive_deadlock_race_check,
        )

    agent = pipeline.Agent
    cta_layout_vmnk = (1, 1, 1, 1)
    tma_group = pipeline.CooperativeGroup(agent.Thread)
    umma_group = pipeline.CooperativeGroup(agent.Thread)
    load_group = pipeline.CooperativeGroup(agent.Thread, 32)
    page_group = pipeline.CooperativeGroup(agent.Thread, 32)
    scheduler_group = pipeline.CooperativeGroup(
        agent.Thread,
        cfg.scheduler_num_warps * 32,
    )
    softmax0_group = pipeline.CooperativeGroup(
        agent.Thread,
        cfg.softmax_num_warps * 32,
    )
    softmax1_group = pipeline.CooperativeGroup(
        agent.Thread,
        cfg.softmax_num_warps * 32,
    )
    corr_group = pipeline.CooperativeGroup(
        agent.Thread,
        cfg.correction_num_warps * 32,
    )

    # Page offsets feed the load warp; Q/KV SMEM feed the MMA warp; TMEM
    # score/output resources hand off to softmax and correction warps.
    page_offsets_cfg = PipelineConfig(
        num_stages=cfg.page_offsets_stages,
        num_bytes=0,
        producer_group=page_group,
        consumer_group=load_group,
        pipeline_type=PipelineType.AsyncAsync,
        cta_layout_vmnk=cta_layout_vmnk,
        async_producer_op=pipeline.PipelineOp.AsyncLoad,
        advance_on_wait=True,
    )
    smem_q_cfg = PipelineConfig.create_tma_umma_pipeline_cfg(
        num_stages=cfg.q_stages,
        num_bytes=cfg.qk_smem_tile_bytes,
        producer_group=tma_group,
        consumer_group=umma_group,
        cta_layout_vmnk=cta_layout_vmnk,
        consumer_signaling_threads=SignalingThreads.CtaLeader,
    )
    smem_kv_cfg = PipelineConfig.create_tma_umma_pipeline_cfg(
        num_stages=cfg.kv_stages,
        num_bytes=cfg.kv_smem_tile_bytes,
        producer_group=tma_group,
        consumer_group=umma_group,
        cta_layout_vmnk=cta_layout_vmnk,
        consumer_signaling_threads=SignalingThreads.CtaLeader,
    )
    tmem_s_cfg = PipelineConfig.create_umma_async_pipeline_cfg(
        num_stages=1,
        producer_group=umma_group,
        consumer_group=softmax0_group,
        cta_layout_vmnk=cta_layout_vmnk,
        producer_signaling_threads=SignalingThreads.CtaLeader,
    )
    tmem_s1_cfg = PipelineConfig.create_umma_async_pipeline_cfg(
        num_stages=1,
        producer_group=umma_group,
        consumer_group=softmax1_group,
        cta_layout_vmnk=cta_layout_vmnk,
        producer_signaling_threads=SignalingThreads.CtaLeader,
    )
    tmem_o_cfg = PipelineConfig.create_umma_async_pipeline_cfg(
        num_stages=cfg.o_stages,
        producer_group=umma_group,
        consumer_group=corr_group,
        cta_layout_vmnk=cta_layout_vmnk,
        producer_signaling_threads=SignalingThreads.CtaLeader,
    )
    tmem_o_cfg = dataclass_replace(tmem_o_cfg, advance_on_wait=True)
    local0_cfg = PipelineConfig.create_async_async_pipeline_cfg(
        num_stages=1,
        producer_group=softmax0_group,
        consumer_group=corr_group,
        cta_layout_vmnk=cta_layout_vmnk,
    )
    local1_cfg = PipelineConfig.create_async_async_pipeline_cfg(
        num_stages=1,
        producer_group=softmax1_group,
        consumer_group=corr_group,
        cta_layout_vmnk=cta_layout_vmnk,
    )
    smem_p0_cfg = PipelineConfig.create_async_umma_pipeline_cfg(
        num_stages=1,
        producer_group=softmax0_group,
        consumer_group=umma_group,
        cta_layout_vmnk=cta_layout_vmnk,
        consumer_signaling_threads=SignalingThreads.CtaLeader,
    )
    smem_p1_cfg = PipelineConfig.create_async_umma_pipeline_cfg(
        num_stages=1,
        producer_group=softmax1_group,
        consumer_group=umma_group,
        cta_layout_vmnk=cta_layout_vmnk,
        consumer_signaling_threads=SignalingThreads.CtaLeader,
    )

    work_queue = None
    schedule_token_throttle = None
    use_clc_dynamic = use_clc_dynamic_scheduler
    use_static_persistent = use_static_persistent_scheduler
    # Exactly one persistent scheduler flavor is selected by the profile.
    # Non-persistent profiles leave work_queue unset and use block indices.
    if use_clc_dynamic:
        work_queue, schedule_token_throttle = _make_clc_work_queue_and_throttle(
            cfg,
            tile_sched_params,
            clc_response_ptr,
            cache_seqs,
            cu_seqlens_q,
            cta_layout_vmnk,
            load_group,
            scheduler_group,
            "ll_mla_work_queue",
            "ll_mla_schedule_token_throttle",
        )
    if use_static_persistent:
        work_queue = _make_static_work_queue(
            cfg,
            tile_sched_params,
            cache_seqs,
            cu_seqlens_q,
            "ll_mla_work_queue",
        )

    smem_page_offsets = SmemPageOffsetsResource(
        cfg=cfg,
        pipeline_config=page_offsets_cfg,
        page_offsets=page_offsets,
        cache_seqs=cache_seqs,
        cu_seqlens_q=cu_seqlens_q,
        batch_idx=batch_idx,
        cta_idx_q=cta_idx_q,
        cta_idx_kv=cta_idx_kv,
        name="ll_mla_page_offsets",
    )
    smem_q = SmemQResource(
        cfg=cfg,
        pipeline_config=smem_q_cfg,
        cu_seqlens_q=cu_seqlens_q,
        name="ll_mla_smem_q",
    )
    smem_q.tma_desc_q_latent = tma_desc_q_latent
    smem_q.tma_desc_q_rope = tma_desc_q_rope
    smem_q.head_idx = head_idx
    smem_q.batch_idx = batch_idx
    smem_q.cta_idx_q = cta_idx_q
    smem_kv = SmemKvResource(
        cfg=cfg,
        pipeline_config=smem_kv_cfg,
        tma_desc_c_latent=tma_desc_c_latent,
        tma_desc_c_rope=tma_desc_c_rope,
        tma_desc_v=tma_desc_v,
        c_rope_tensor=c_rope_tensor,
        page_offsets_kv=smem_page_offsets if use_page_offsets else None,
        page_offsets=page_offsets,
        cache_seqs=cache_seqs,
        cu_seqlens_q=cu_seqlens_q,
        head_idx=head_idx,
        batch_idx=batch_idx,
        cta_idx_q=cta_idx_q,
        cta_idx_kv=cta_idx_kv,
        cta_idx_head_dim_v=cta_idx_head_dim_v,
        name="ll_mla_smem_kv",
    )
    tmem_s0 = TmemSResource(
        cfg=cfg,
        pipeline_config=tmem_s_cfg,
        inst_id=0,
        scale_softmax_log2=scale_softmax_log2,
        cache_seqs=cache_seqs,
        cu_seqlens_q=cu_seqlens_q,
        head_idx=head_idx,
        batch_idx=batch_idx,
        cta_idx_q=cta_idx_q,
        cta_idx_kv=cta_idx_kv,
        sync_barrier_id=0,
        name="ll_mla_tmem_s0",
    )
    tmem_s1 = TmemSResource(
        cfg=cfg,
        pipeline_config=tmem_s1_cfg,
        inst_id=1,
        scale_softmax_log2=scale_softmax_log2,
        cache_seqs=cache_seqs,
        cu_seqlens_q=cu_seqlens_q,
        head_idx=head_idx,
        batch_idx=batch_idx,
        cta_idx_q=cta_idx_q,
        cta_idx_kv=cta_idx_kv,
        sync_barrier_id=1,
        name="ll_mla_tmem_s1",
    )
    order_p01_alloc = SmemAllocation(
        name="ll_mla_order_p01",
        size_bytes=16,
        alignment=8,
    )
    smem_p0 = SmemPResource(
        cfg=cfg,
        pipeline_config=smem_p0_cfg,
        inst_id=0,
        scale_softmax_log2=scale_softmax_log2,
        order_p01_alloc=order_p01_alloc,
        owns_order_p01_alloc=True,
        name="ll_mla_smem_p0",
    )
    smem_p1 = SmemPResource(
        cfg=cfg,
        pipeline_config=smem_p1_cfg,
        inst_id=1,
        scale_softmax_log2=scale_softmax_log2,
        order_p01_alloc=order_p01_alloc,
        name="ll_mla_smem_p1",
    )
    tmem_s0.p_ref = smem_p0
    tmem_s1.p_ref = smem_p1
    smem_p0.tmem_s_ref = tmem_s0
    smem_p1.tmem_s_ref = tmem_s1
    tmem_o = TmemOResource(
        cfg=cfg,
        pipeline_config=tmem_o_cfg,
        p0_ref=smem_p0,
        p1_ref=smem_p1,
        cache_seqs=cache_seqs,
        cu_seqlens_q=cu_seqlens_q,
        batch_idx=batch_idx,
        cta_idx_q=cta_idx_q,
        name="ll_mla_tmem_o",
    )
    local0 = TmemSoftmaxLocalResource(
        cfg=cfg,
        pipeline_config=local0_cfg,
        inst_id=0,
        name="ll_mla_local0",
    )
    local1 = TmemSoftmaxLocalResource(
        cfg=cfg,
        pipeline_config=local1_cfg,
        inst_id=1,
        name="ll_mla_local1",
    )
    global0 = TmemSoftmaxGlobalResource(cfg=cfg, inst_id=0, name="ll_mla_global0")
    global1 = TmemSoftmaxGlobalResource(cfg=cfg, inst_id=1, name="ll_mla_global1")
    corr0 = TmemCorrResource(
        cfg=cfg,
        inst_id=0,
        scale_softmax_log2=scale_softmax_log2,
        output_scale=output_scale,
        o_tensor=o_tensor,
        lse_tensor=lse_tensor,
        acc_o_tensor=acc_o_tensor,
        acc_lse_tensor=acc_lse_tensor,
        cache_seqs=cache_seqs,
        cu_seqlens_q=cu_seqlens_q,
        head_idx=head_idx,
        batch_idx=batch_idx,
        cta_idx_q=cta_idx_q,
        cta_idx_kv=cta_idx_kv,
        cta_idx_head_dim_v=cta_idx_head_dim_v,
        name="ll_mla_corr0",
    )
    corr1 = TmemCorrResource(
        cfg=cfg,
        inst_id=1,
        scale_softmax_log2=scale_softmax_log2,
        output_scale=output_scale,
        o_tensor=o_tensor,
        lse_tensor=lse_tensor,
        acc_o_tensor=acc_o_tensor,
        acc_lse_tensor=acc_lse_tensor,
        cache_seqs=cache_seqs,
        cu_seqlens_q=cu_seqlens_q,
        head_idx=head_idx,
        batch_idx=batch_idx,
        cta_idx_q=cta_idx_q,
        cta_idx_kv=cta_idx_kv,
        cta_idx_head_dim_v=cta_idx_head_dim_v,
        name="ll_mla_corr1",
    )
    local_kv_tiles = cfg.local_kv_tiles(total_kv_tiles)
    loop_domain = cfg.loop_domain(local_kv_tiles)
    load_domain = loop_domain
    mma_domain = loop_domain
    softmax_domain = loop_domain + 1
    corr_domain = loop_domain
    task_domain_kwargs = {
        "seqlens_kv": cache_seqs,
        "cu_seqlens_q": cu_seqlens_q,
    }

    tasks = []
    if use_page_offsets:
        tasks.append(
            create_load_page_offsets_task(
                smem_page_offsets,
                work_queue,
                cfg,
                domain=load_domain,
                task_class=MlaDecodeTask,
                **task_domain_kwargs,
            )
        )
    tasks.extend(
        [
            create_throughput_latency_load_task(
                smem_q,
                smem_kv,
                work_queue,
                schedule_token_throttle,
                cfg,
                domain=load_domain,
                smem_page_offsets=smem_page_offsets if use_page_offsets else None,
                use_page_offsets=use_page_offsets,
                task_class=MlaDecodeTask,
                **task_domain_kwargs,
            ),
            create_throughput_latency_softmax0_task(
                tmem_s0,
                local0,
                smem_p0,
                global0,
                work_queue,
                cfg,
                domain=softmax_domain,
                task_class=MlaDecodeTask,
                domain_bias=1,
                **task_domain_kwargs,
            ),
            create_throughput_latency_softmax1_task(
                tmem_s1,
                local1,
                smem_p1,
                global1,
                work_queue,
                cfg,
                domain=softmax_domain,
                task_class=MlaDecodeTask,
                domain_bias=1,
                **task_domain_kwargs,
            ),
            create_throughput_latency_correction_task(
                local0,
                local1,
                tmem_o,
                corr0,
                corr1,
                work_queue,
                cfg,
                domain=corr_domain,
                task_class=MlaDecodeTask,
                **task_domain_kwargs,
            ),
            create_throughput_latency_mma_task(
                smem_q,
                smem_kv,
                tmem_s0,
                tmem_s1,
                smem_p0,
                smem_p1,
                tmem_o,
                work_queue,
                cfg,
                domain=mma_domain,
                task_class=MlaDecodeTask,
                **task_domain_kwargs,
            ),
        ]
    )
    if not (use_page_offsets and use_clc_dynamic):
        tasks.append(
            create_padding_task(
                cfg,
                work_queue,
                warp_idx=(
                    cfg.page_offsets_warp_idx + cfg.page_offsets_num_warps
                    if use_page_offsets
                    else (cfg.clc_padding_warp_idx if use_clc_dynamic else None)
                ),
                num_warps=(
                    1
                    if use_page_offsets
                    else (cfg.clc_padding_num_warps if use_clc_dynamic else None)
                ),
                task_class=MlaDecodeTask,
            )
        )
    if use_clc_dynamic:
        tasks.append(
            create_throughput_latency_scheduler_task(
                work_queue,
                schedule_token_throttle,
                cfg,
                task_class=MlaDecodeTask,
            )
        )

    deps = {
        smem_q: [],
        smem_kv: [],
        tmem_s0: [smem_q, smem_kv],
        tmem_s1: [smem_q, smem_kv],
        smem_p0: [tmem_s0],
        smem_p1: [tmem_s1],
        global0: [tmem_s0],
        global1: [tmem_s1],
        local0: [tmem_s0],
        local1: [tmem_s1],
        tmem_o: [smem_p0, smem_p1, smem_kv],
        corr0: [local0, tmem_o],
        corr1: [local0, local1, tmem_o],
    }
    if use_page_offsets:
        deps[smem_page_offsets] = []
        deps[smem_kv].append(smem_page_offsets)
    if work_queue is not None:
        work_queue_deps = [work_queue] if use_clc_dynamic else []
        deps = {
            smem_q: [work_queue],
            smem_kv: [work_queue],
            tmem_s0: [smem_q, smem_kv, work_queue],
            tmem_s1: [smem_q, smem_kv, work_queue],
            smem_p0: [tmem_s0, work_queue],
            smem_p1: [tmem_s1, work_queue],
            global0: [tmem_s0, work_queue],
            global1: [tmem_s1, work_queue],
            local0: [tmem_s0, work_queue],
            local1: [tmem_s1, work_queue],
            tmem_o: [smem_p0, smem_p1, smem_kv, work_queue],
            corr0: [local0, tmem_o, work_queue],
            corr1: [local0, local1, tmem_o, work_queue],
            work_queue: (
                work_queue_deps + [schedule_token_throttle]
                if schedule_token_throttle is not None
                else work_queue_deps
            ),
        }
        if use_page_offsets:
            deps[smem_page_offsets] = [work_queue]
            deps[smem_kv].append(smem_page_offsets)
        if schedule_token_throttle is not None:
            deps[schedule_token_throttle] = [work_queue]

    dma_release_labels = {
        (smem_kv, tmem_s0): {"k_desc_0"},
        (smem_kv, tmem_s1): {"k_desc_1"},
        (smem_kv, tmem_o): {"v_desc_0", "v_desc_1"},
    }

    smem_allocator = SmemAllocator()
    smem_allocator.add_resource(smem_q)
    if use_page_offsets:
        smem_allocator.add_resource(smem_page_offsets)
    smem_allocator.add_resource(smem_kv)
    smem_allocator.add_resource(smem_p0)
    smem_allocator.add_resource(smem_p1)
    smem_allocator.add_resource(tmem_s0)
    smem_allocator.add_resource(tmem_s1)
    smem_allocator.add_resource(tmem_o)
    smem_allocator.add_resource(local0)
    smem_allocator.add_resource(local1)
    smem_allocator.add_resource(global0)
    smem_allocator.add_resource(global1)
    smem_allocator.add_resource(corr0)
    smem_allocator.add_resource(corr1)
    smem_allocator.add_tmem_ptr(
        SmemAllocation("ll_mla_tmem_ptr_i32", dtype=cutlass.Int32, alignment=4)
    )
    smem_allocator.compute_layout()

    tmem_allocator = TmemAllocator()
    tmem_allocator.add_resource(tmem_s0)
    tmem_allocator.add_resource(tmem_s1)
    tmem_allocator.add_resource(local0)
    tmem_allocator.add_resource(local1)
    tmem_allocator.add_resource(tmem_o)
    tmem_allocator.compute_layout()

    task_manager = TaskManager(
        tasks=tasks,
        resource_dependency_graph=deps,
        dma_consumer_release_labels=dma_release_labels,
        smem_allocator=smem_allocator,
        tmem_allocator=tmem_allocator,
        verbose=verbose,
        exhaustive_deadlock_race_check=exhaustive_deadlock_race_check,
    )
    return task_manager, corr1


def _make_keeps_mma_ab_task_graph(
    cfg: MlaConfig,
    *,
    total_kv_tiles: int,
    use_page_offsets: bool = True,
    tma_desc_q_latent=None,
    tma_desc_q_rope=None,
    tma_desc_c_latent=None,
    tma_desc_c_rope=None,
    tma_desc_v=None,
    c_rope_tensor=None,
    page_offsets=None,
    cache_seqs=None,
    cu_seqlens_q=None,
    head_idx=None,
    batch_idx=None,
    cta_idx_q=None,
    cta_idx_kv=None,
    cta_idx_head_dim_v=None,
    scale_softmax_log2=None,
    output_scale=None,
    o_tensor=None,
    lse_tensor=None,
    acc_o_tensor=None,
    acc_lse_tensor=None,
    tile_sched_params=None,
    clc_response_ptr=None,
    use_clc_dynamic_scheduler: bool = False,
    use_static_persistent_scheduler: bool = False,
    verbose: bool = False,
    exhaustive_deadlock_race_check: bool = False,
) -> TaskManager:
    """Build the keeps-MMA-AB 1CTA task graph for the generic builder.

    This variant uses one score/P pipe with P stored in TMEM. The swaps path
    keeps its two SMEM-P pipes and is intentionally left untouched.
    """

    scale_softmax_log2, output_scale = _default_scales(
        scale_softmax_log2,
        output_scale,
    )
    _check_persistent_scheduler_modes(
        use_clc_dynamic_scheduler,
        use_static_persistent_scheduler,
    )
    agent = pipeline.Agent
    cta_layout_vmnk = (1, 1, 1, 1)
    tma_group = pipeline.CooperativeGroup(agent.Thread)
    umma_group = pipeline.CooperativeGroup(agent.Thread)
    load_group = pipeline.CooperativeGroup(agent.Thread, 32)
    page_group = pipeline.CooperativeGroup(agent.Thread, 32)
    softmax_group = pipeline.CooperativeGroup(
        agent.Thread,
        cfg.softmax_num_warps * 32,
    )
    corr_group = pipeline.CooperativeGroup(
        agent.Thread,
        cfg.correction_num_warps * 32,
    )

    page_offsets_cfg = PipelineConfig(
        num_stages=cfg.page_offsets_stages,
        num_bytes=0,
        producer_group=page_group,
        consumer_group=load_group,
        pipeline_type=PipelineType.AsyncAsync,
        cta_layout_vmnk=cta_layout_vmnk,
        async_producer_op=pipeline.PipelineOp.AsyncLoad,
        advance_on_wait=True,
    )
    smem_q_cfg = PipelineConfig.create_tma_umma_pipeline_cfg(
        num_stages=cfg.q_stages,
        num_bytes=cfg.qk_smem_tile_bytes,
        producer_group=tma_group,
        consumer_group=umma_group,
        cta_layout_vmnk=cta_layout_vmnk,
        consumer_signaling_threads=SignalingThreads.CtaLeader,
    )
    smem_kv_cfg = PipelineConfig.create_tma_umma_pipeline_cfg(
        num_stages=cfg.kv_stages,
        num_bytes=cfg.kv_smem_tile_bytes,
        producer_group=tma_group,
        consumer_group=umma_group,
        cta_layout_vmnk=cta_layout_vmnk,
        consumer_signaling_threads=SignalingThreads.CtaLeader,
    )
    tmem_s_cfg = PipelineConfig.create_umma_async_pipeline_cfg(
        num_stages=2,
        producer_group=umma_group,
        consumer_group=softmax_group,
        cta_layout_vmnk=cta_layout_vmnk,
        producer_signaling_threads=SignalingThreads.CtaLeader,
    )
    local_cfg = PipelineConfig.create_async_async_pipeline_cfg(
        num_stages=2,
        producer_group=softmax_group,
        consumer_group=corr_group,
        cta_layout_vmnk=cta_layout_vmnk,
    )
    tmem_p_cfg = PipelineConfig.create_async_umma_pipeline_cfg(
        num_stages=2,
        producer_group=softmax_group,
        consumer_group=umma_group,
        cta_layout_vmnk=cta_layout_vmnk,
        consumer_signaling_threads=SignalingThreads.CtaLeader,
    )
    tmem_o_cfg = PipelineConfig.create_umma_async_pipeline_cfg(
        num_stages=1,
        producer_group=umma_group,
        consumer_group=corr_group,
        cta_layout_vmnk=cta_layout_vmnk,
        producer_signaling_threads=SignalingThreads.CtaLeader,
    )

    work_queue = None
    schedule_token_throttle = None
    use_clc_dynamic = use_clc_dynamic_scheduler
    if use_static_persistent_scheduler:
        work_queue = _make_static_work_queue(
            cfg,
            tile_sched_params,
            cache_seqs,
            cu_seqlens_q,
            "ll_mla_q64_work_queue",
        )
    if use_clc_dynamic:
        scheduler_group = pipeline.CooperativeGroup(agent.Thread, 32)
        work_queue, schedule_token_throttle = _make_clc_work_queue_and_throttle(
            cfg,
            tile_sched_params,
            clc_response_ptr,
            cache_seqs,
            cu_seqlens_q,
            cta_layout_vmnk,
            load_group,
            scheduler_group,
            "ll_mla_q64_work_queue",
            "ll_mla_q64_schedule_token_throttle",
        )

    smem_page_offsets = SmemPageOffsetsResource(
        cfg=cfg,
        pipeline_config=page_offsets_cfg,
        page_offsets=page_offsets,
        cache_seqs=cache_seqs,
        cu_seqlens_q=cu_seqlens_q,
        batch_idx=batch_idx,
        cta_idx_q=cta_idx_q,
        cta_idx_kv=cta_idx_kv,
        name="ll_mla_q64_page_offsets",
    )
    smem_q = SmemQResource(
        cfg=cfg,
        pipeline_config=smem_q_cfg,
        tma_desc_q_latent=tma_desc_q_latent,
        tma_desc_q_rope=tma_desc_q_rope,
        cu_seqlens_q=cu_seqlens_q,
        head_idx=head_idx,
        batch_idx=batch_idx,
        cta_idx_q=cta_idx_q,
        name="ll_mla_q64_smem_q",
    )
    smem_kv = SmemKvResource(
        cfg=cfg,
        pipeline_config=smem_kv_cfg,
        tma_desc_c_latent=tma_desc_c_latent,
        tma_desc_c_rope=tma_desc_c_rope,
        tma_desc_v=tma_desc_v,
        c_rope_tensor=c_rope_tensor,
        page_offsets_kv=smem_page_offsets if use_page_offsets else None,
        page_offsets=page_offsets,
        cache_seqs=cache_seqs,
        cu_seqlens_q=cu_seqlens_q,
        head_idx=head_idx,
        batch_idx=batch_idx,
        cta_idx_q=cta_idx_q,
        cta_idx_kv=cta_idx_kv,
        cta_idx_head_dim_v=cta_idx_head_dim_v,
        name="ll_mla_q64_smem_kv",
    )
    tmem_s = TmemSKeepsResource(
        cfg=cfg,
        pipeline_config=tmem_s_cfg,
        inst_id=0,
        scale_softmax_log2=scale_softmax_log2,
        cache_seqs=cache_seqs,
        cu_seqlens_q=cu_seqlens_q,
        head_idx=head_idx,
        batch_idx=batch_idx,
        cta_idx_q=cta_idx_q,
        cta_idx_kv=cta_idx_kv,
        sync_barrier_id=0,
        name="ll_mla_q64_tmem_s",
    )
    tmem_p = TmemPResource(
        cfg=cfg,
        pipeline_config=tmem_p_cfg,
        inst_id=0,
        scale_softmax_log2=scale_softmax_log2,
        tmem_alias_ref=tmem_s,
        name="ll_mla_q64_tmem_p",
    )
    tmem_s.p_ref = tmem_p
    tmem_o = TmemOResource(
        cfg=cfg,
        pipeline_config=tmem_o_cfg,
        p_tmem_ref=tmem_p,
        cache_seqs=cache_seqs,
        cu_seqlens_q=cu_seqlens_q,
        batch_idx=batch_idx,
        cta_idx_q=cta_idx_q,
        name="ll_mla_q64_tmem_o",
    )
    local = TmemSoftmaxLocalResource(
        cfg=cfg,
        pipeline_config=local_cfg,
        inst_id=0,
        tmem_alias_ref=tmem_s,
        name="ll_mla_q64_local",
    )
    global_softmax = TmemSoftmaxGlobalResource(
        cfg=cfg, inst_id=0, name="ll_mla_q64_global"
    )
    corr = TmemCorrResource(
        cfg=cfg,
        inst_id=1,
        scale_softmax_log2=scale_softmax_log2,
        output_scale=output_scale,
        o_tensor=o_tensor,
        lse_tensor=lse_tensor,
        acc_o_tensor=acc_o_tensor,
        acc_lse_tensor=acc_lse_tensor,
        cache_seqs=cache_seqs,
        cu_seqlens_q=cu_seqlens_q,
        head_idx=head_idx,
        batch_idx=batch_idx,
        cta_idx_q=cta_idx_q,
        cta_idx_kv=cta_idx_kv,
        cta_idx_head_dim_v=cta_idx_head_dim_v,
        name="ll_mla_q64_corr",
    )

    local_kv_tiles = cfg.local_kv_tiles(total_kv_tiles)
    loop_domain = cfg.loop_domain(local_kv_tiles)
    task_domain_kwargs = {
        "seqlens_kv": cache_seqs,
        "cu_seqlens_q": cu_seqlens_q,
    }

    tasks = []
    if use_page_offsets:
        tasks.append(
            create_load_page_offsets_task(
                smem_page_offsets,
                work_queue,
                cfg,
                domain=loop_domain,
                task_class=MlaDecodeTask,
                **task_domain_kwargs,
            )
        )
    tasks.extend(
        [
            create_throughput_latency_load_task(
                smem_q,
                smem_kv,
                work_queue,
                schedule_token_throttle,
                cfg,
                domain=loop_domain,
                smem_page_offsets=smem_page_offsets if use_page_offsets else None,
                use_page_offsets=use_page_offsets,
                task_class=MlaDecodeTask,
                **task_domain_kwargs,
            ),
            create_keeps_mma_ab_softmax_task(
                tmem_s,
                local,
                tmem_p,
                global_softmax,
                work_queue,
                cfg,
                domain=loop_domain + 1,
                task_class=MlaDecodeTask,
                domain_bias=1,
                **task_domain_kwargs,
            ),
            create_keeps_mma_ab_correction_task(
                local,
                tmem_o,
                corr,
                work_queue,
                cfg,
                domain=loop_domain,
                task_class=MlaDecodeTask,
                **task_domain_kwargs,
            ),
            create_keeps_mma_ab_mma_task(
                smem_q,
                smem_kv,
                tmem_s,
                tmem_p,
                tmem_o,
                work_queue,
                cfg,
                domain=loop_domain,
                task_class=MlaDecodeTask,
                **task_domain_kwargs,
            ),
        ]
    )
    if not use_clc_dynamic:
        tasks.append(
            create_padding_task(
                cfg,
                work_queue,
                warp_idx=11,
                num_warps=1,
                task_class=MlaDecodeTask,
            )
        )
    elif not use_page_offsets:
        tasks.append(
            create_padding_task(
                cfg,
                work_queue,
                warp_idx=cfg.page_offsets_warp_idx,
                num_warps=cfg.page_offsets_num_warps,
                task_class=MlaDecodeTask,
            )
        )
    if use_clc_dynamic:
        tasks.append(
            create_throughput_latency_scheduler_task(
                work_queue,
                schedule_token_throttle,
                cfg,
                task_class=MlaDecodeTask,
            )
        )

    deps = {
        smem_q: [],
        smem_kv: [],
        tmem_s: [smem_q, smem_kv],
        local: [tmem_s],
        global_softmax: [tmem_s],
        tmem_p: [tmem_s],
        tmem_o: [tmem_p, smem_kv],
        corr: [local, tmem_o],
    }
    if use_page_offsets:
        deps[smem_page_offsets] = []
        deps[smem_kv].append(smem_page_offsets)
    if work_queue is not None:
        deps = {
            smem_q: [work_queue],
            smem_kv: [work_queue],
            tmem_s: [smem_q, smem_kv, work_queue],
            local: [tmem_s, work_queue],
            global_softmax: [tmem_s, work_queue],
            tmem_p: [tmem_s, work_queue],
            tmem_o: [tmem_p, smem_kv, work_queue],
            corr: [local, tmem_o, work_queue],
            work_queue: (
                [work_queue, schedule_token_throttle]
                if schedule_token_throttle is not None
                else []
            ),
        }
        if use_page_offsets:
            deps[smem_page_offsets] = [work_queue]
            deps[smem_kv].append(smem_page_offsets)
        if schedule_token_throttle is not None:
            deps[schedule_token_throttle] = [work_queue]

    dma_release_labels = {
        (smem_kv, tmem_s): {"k_desc_0"},
        (smem_kv, tmem_o): {"v_desc_0"},
    }

    smem_allocator = SmemAllocator()
    smem_allocator.add_resource(smem_q)
    if use_page_offsets:
        smem_allocator.add_resource(smem_page_offsets)
    for resource in (smem_kv, tmem_s, local, global_softmax, tmem_p, tmem_o, corr):
        smem_allocator.add_resource(resource)
    smem_allocator.add_tmem_ptr(
        SmemAllocation("ll_mla_q64_tmem_ptr_i32", dtype=Int32, alignment=4)
    )
    smem_allocator.compute_layout()

    tmem_allocator = TmemAllocator()
    tmem_allocator.add_resource(tmem_s)
    tmem_allocator.add_resource(local)
    tmem_allocator.add_resource(tmem_o)
    tmem_allocator.compute_layout()

    task_manager = TaskManager(
        tasks=tasks,
        resource_dependency_graph=deps,
        dma_consumer_release_labels=dma_release_labels,
        smem_allocator=smem_allocator,
        tmem_allocator=tmem_allocator,
        verbose=verbose,
        exhaustive_deadlock_race_check=exhaustive_deadlock_race_check,
    )
    return task_manager, corr


class ThroughputLatencyMlaDecodeTs:
    """Dense throughput-latency 1CTA MLA TS wrapper."""

    def __init__(
        self,
        *,
        batch_size: int,
        num_heads: int,
        seq_len_q: int,
        seq_len_k: int,
        latent_dim: int = 512,
        rope_dim: int = 64,
        page_size: int = 32,
        max_active_clusters: int,
        acc_dtype=None,
        lse_dtype=None,
        qkv_dtype: str = "bf16",
        out_dtype: str = "bf16",
        profile: str | None = None,
        persistent_wave_sm_count: int | None = None,
        reduction_mode: str | None = None,
        groups_tokens_heads: bool = False,
        groups_tokens_heads_ratio: int = 1,
        logical_num_heads: int | None = None,
        logical_seq_len_q: int | None = None,
        tile_size_q: int | None = None,
        explicit_split_kv: int | None = None,
        explicit_persistent: bool | None = None,
        groups_tokens_heads_q_ratio: int | None = None,
        mask_type: MaskType | str = MaskType.CAUSAL,
    ):
        """Initialize the wrapper, deriving groups_tokens_heads_q rows when unspecified."""
        import cutlass as _cutlass

        if acc_dtype is None:
            acc_dtype = _cutlass.Float32
        if lse_dtype is None:
            lse_dtype = _cutlass.Float32

        has_explicit_groups_tokens_heads_q_shape = (
            groups_tokens_heads_q_ratio is not None
            or groups_tokens_heads
            or groups_tokens_heads_ratio != 1
            or logical_num_heads is not None
            or logical_seq_len_q is not None
        )
        if not has_explicit_groups_tokens_heads_q_shape:
            groups_tokens_heads_q_shape = (
                resolve_throughput_latency_groups_tokens_heads_q_shape(
                    num_heads_q=num_heads,
                    seq_len_q=seq_len_q,
                    explicit_tile_size_q=tile_size_q,
                    profile=profile,
                )
            )
            num_heads = groups_tokens_heads_q_shape.num_heads_q
            seq_len_q = groups_tokens_heads_q_shape.seq_len_q
            logical_num_heads = groups_tokens_heads_q_shape.logical_num_heads_q
            logical_seq_len_q = groups_tokens_heads_q_shape.logical_seq_len_q
            tile_size_q = groups_tokens_heads_q_shape.tile_size_q
            groups_tokens_heads_q_ratio = groups_tokens_heads_q_shape.ratio

        self.batch_size = batch_size
        self.num_heads = num_heads
        self.seq_len_q = seq_len_q
        self.seq_len_k = seq_len_k
        self.latent_dim = latent_dim
        self.rope_dim = rope_dim
        self.page_size = page_size
        self.max_active_clusters = max_active_clusters
        self.qkv_dtype = qkv_dtype
        self.out_dtype = out_dtype
        self.profile = profile
        self.persistent_wave_sm_count = persistent_wave_sm_count
        self.reduction_mode = reduction_mode
        if groups_tokens_heads_q_ratio is None:
            group = groups_tokens_heads_ratio if groups_tokens_heads else 1
        else:
            if groups_tokens_heads or groups_tokens_heads_ratio != 1:
                raise ValueError(
                    "groups_tokens_heads_q_ratio cannot be combined with explicit "
                    "groups_tokens_heads arguments"
                )
            group = groups_tokens_heads_q_ratio
        if group <= 0:
            raise ValueError("groups_tokens_heads_q_ratio must be positive")
        self.groups_tokens_heads_ratio = group
        self.groups_tokens_heads = group > 1
        if logical_num_heads is None:
            logical_num_heads = num_heads // group
        if logical_seq_len_q is None:
            if group != 1:
                raise ValueError(
                    "logical_seq_len_q is required for groups_tokens_heads_q launches"
                )
            logical_seq_len_q = seq_len_q
        self.logical_num_heads = logical_num_heads
        self.logical_seq_len_q = logical_seq_len_q
        self.tile_size_q = tile_size_q
        self.explicit_split_kv = explicit_split_kv
        self.explicit_persistent = explicit_persistent
        self.mask_type = normalize_mask_type(mask_type)

        cfg = self._make_config()
        # The parallel standalone reducer is a production-derived choice, not
        # a user knob.  It requires a fixed split profile and a static producer
        # schedule so the reducer topology and workspace contract are compile-
        # time invariant.  Every other supported profile keeps the established
        # FlashInfer reducer as an automatic fallback.
        self.use_parallel_reduction = (
            supports_parallel_gmem_reduction(cfg)
            and lse_dtype == _cutlass.Float32
            and cfg.use_persistent_scheduler == 0
            and cfg.use_clc_dynamic_persistent_scheduler == 0
        )
        self.parallel_reduction_topology = None
        self.parallel_reduction_elements_per_slice = (
            PARALLEL_GMEM_REDUCTION_SWAPS_ELEMENTS_PER_SLICE
            if cfg.tile_size_q in (8, 16, 32)
            else PARALLEL_GMEM_REDUCTION_ELEMENTS_PER_SLICE
        )
        if cfg.use_multi_ctas_kv == 1 and cfg.use_cluster_reduction != 1:
            # Both the retained reducer and the parallel implementation use
            # the same normalized partial-O workspace. Qualify the configured
            # separate-GMEM launch regardless of which reducer is selected.
            validate_parallel_reduction_workspace(
                batch_size=cfg.batch_size,
                num_heads_q=cfg.num_heads_q,
                seq_len_q=cfg.seq_len_q,
                splits_kv=cfg.num_ctas_per_seq_kv,
                head_dim=cfg.head_dim_v,
            )
        if self.use_parallel_reduction:
            base_clusters = parallel_gmem_reduction_base_clusters(
                cfg,
                self.parallel_reduction_elements_per_slice,
            )
            cluster_size = (
                choose_q64_parallel_reducer_cluster_size(
                    cfg.num_ctas_per_seq_kv,
                    base_clusters=base_clusters,
                    sm_count=self.max_active_clusters,
                )
                if cfg.tile_size_q == 64
                else 1
            )
            self.parallel_reduction_topology = (
                make_balanced_parallel_reduction_topology(
                    cfg.num_ctas_per_seq_kv,
                    cluster_size=cluster_size,
                )
            )
        if num_heads > cfg.tile_size_q and num_heads % cfg.tile_size_q != 0:
            raise NotImplementedError(
                "Throughput-latency 1CTA TS MLA runtime requires multi-tile num_heads to be "
                f"divisible by tile_size_q: num_heads={num_heads}, "
                f"tile_size_q={cfg.tile_size_q}."
            )

        self.acc_dtype = acc_dtype
        self.lse_dtype = lse_dtype

    @property
    def groups_tokens_heads_q_ratio(self) -> int:
        """Return the effective groups_tokens_heads_q capacity."""

        return self.groups_tokens_heads_ratio

    def _make_config(self):
        """Create the static throughput-latency MLA config for this wrapper."""
        cfg = make_throughput_latency_mla_config(
            batch_size=self.batch_size,
            num_heads_q=self.num_heads,
            seq_len_q=self.seq_len_q,
            seq_len_kv=self.seq_len_k,
            latent_dim=self.latent_dim,
            rope_dim=self.rope_dim,
            num_tokens_per_page=self.page_size,
            qkv_dtype=self.qkv_dtype,
            o_dtype=self.out_dtype,
            profile=self.profile,
            persistent_wave_sm_count=self.persistent_wave_sm_count,
            max_active_clusters=self.max_active_clusters,
            reduction_mode=self.reduction_mode,
            logical_num_heads_q=self.logical_num_heads,
            logical_seq_len_q=self.logical_seq_len_q,
            groups_tokens_heads_q_ratio=self.groups_tokens_heads_q_ratio,
            tile_size_q=self.tile_size_q,
            explicit_split_kv=self.explicit_split_kv,
            explicit_persistent=self.explicit_persistent,
            mask_type=self.mask_type,
        )
        return cfg

    def validate_split_kv_launch(self, split_kv: int, workspace) -> None:
        """Validate host-side split-KV arguments before compiling the JIT body."""

        cfg = self._make_config()
        if (
            cfg.use_multi_ctas_kv == 1
            and cfg.use_cluster_reduction != 1
            and workspace is None
        ):
            raise ValueError(
                "multi-CTA-KV throughput-latency 1CTA MLA requires workspace"
            )
        if cfg.use_multi_ctas_kv == 1 and split_kv < cfg.num_ctas_per_seq_kv:
            raise ValueError(
                "split_kv is smaller than the configured multi-CTA-KV split count"
            )

    def validate_groups_tokens_heads_launch_shape(
        self,
        q_latent_shape,
        q_rope_shape,
        o_shape,
        lse_shape,
        logical_num_heads: int,
        logical_seq_len_q: int,
        cu_seqlens_q_shape=None,
    ) -> None:
        """Validate logical public tensors against the groups_tokens_heads_q launch shape."""

        is_ragged = len(q_latent_shape) == 3
        if is_ragged:
            if cu_seqlens_q_shape is None:
                raise ValueError("rank-3 compact Q/O tensors require cu_seqlens_q")
            if len(cu_seqlens_q_shape) != 1 or int(cu_seqlens_q_shape[0]) != (
                self.batch_size + 1
            ):
                raise ValueError(
                    "cu_seqlens_q must be rank-1 with batch_size + 1 offsets"
                )
            for name, shape in (
                ("q_latent", q_latent_shape),
                ("q_rope", q_rope_shape),
                ("o", o_shape),
            ):
                if len(shape) != 3:
                    raise ValueError(
                        f"{name} must be rank-3 for a compact ragged-query launch"
                    )
                if int(shape[0]) != logical_num_heads:
                    raise ValueError(
                        "compact ragged-query launch tensors must agree on "
                        f"logical num_heads for {name}"
                    )
            if len(lse_shape) != 2:
                raise ValueError("lse must be rank-2 for a compact ragged-query launch")
            if int(lse_shape[0]) != logical_num_heads:
                raise ValueError(
                    "compact ragged-query launch tensors must agree on "
                    "logical num_heads for lse"
                )
            total_query_rows = int(q_latent_shape[2])
            if (
                int(q_rope_shape[2]) != total_query_rows
                or int(o_shape[2]) != total_query_rows
                or int(lse_shape[1]) != total_query_rows
            ):
                raise ValueError(
                    "compact ragged-query launch tensors must agree on total Q rows"
                )
        else:
            if cu_seqlens_q_shape is not None:
                raise ValueError("cu_seqlens_q requires rank-3 compact Q/O tensors")
            for name, shape in (
                ("q_latent", q_latent_shape),
                ("q_rope", q_rope_shape),
                ("o", o_shape),
            ):
                if len(shape) != 4:
                    raise ValueError(
                        f"{name} must be rank-4 for groups_tokens_heads_q launch"
                    )
                if (
                    int(shape[0]) != logical_num_heads
                    or int(shape[2]) != logical_seq_len_q
                ):
                    raise ValueError(
                        "groups_tokens_heads_q launch tensors must agree on "
                        f"logical num_heads/seq_len_q for {name}"
                    )
            if len(lse_shape) != 3:
                raise ValueError("lse must be rank-3 for groups_tokens_heads_q launch")
            if (
                int(lse_shape[0]) != logical_num_heads
                or int(lse_shape[1]) != logical_seq_len_q
            ):
                raise ValueError(
                    "groups_tokens_heads_q launch tensors must agree on "
                    "logical num_heads/seq_len_q for lse"
                )

        group = self.groups_tokens_heads_q_ratio
        if group <= 1:
            return
        if logical_num_heads * group != self.num_heads:
            raise ValueError(
                "groups_tokens_heads_q effective heads must match kernel num_heads"
            )
        if (
            groups_tokens_heads_q_group_count(logical_seq_len_q, group)
            != self.seq_len_q
        ):
            raise ValueError(
                "groups_tokens_heads_q effective seq_len_q must be the ceil-divided "
                "logical query length"
            )

    def initialize_workspace(
        self,
        H: cutlass.Int32,
        D: cutlass.Int32,
        S: cutlass.Int32,
        B: cutlass.Int32,
        split_kv: cutlass.Int32,
        workspace: cute.Tensor,
    ):
        """Construct throughput-latency 1CTA split-KV GMEM reduction tensors."""
        acc_o, acc_lse = None, None
        if cutlass.const_expr(workspace is not None):
            align = 256 // cutlass.Float16.width
            acc_o_layout = cute.make_layout(
                (H, split_kv, D, S, B),
                stride=(
                    cute.assume(split_kv * D, align),
                    cute.assume(D, align),
                    1,
                    cute.assume(split_kv * H * D, align),
                    cute.assume(H * split_kv * S * D, align),
                ),
            )
            acc_o_iter = cute.recast_ptr(workspace.iterator, dtype=cutlass.BFloat16)
            acc_o = cute.make_tensor(acc_o_iter, acc_o_layout)
            acc_lse_layout = cute.make_layout(
                (H, split_kv, S, B),
                stride=(split_kv, 1, H * split_kv, H * split_kv * S),
            )
            acc_lse_iter = cute.recast_ptr(
                workspace.iterator
                + Int64(cute.cosize(acc_o_layout)) * Int64(cutlass.Float16.width // 8),
                dtype=self.lse_dtype,
            )
            acc_lse = cute.make_tensor(acc_lse_iter, acc_lse_layout)
        return acc_o, acc_lse

    @cute.jit
    def __call__(
        self,
        q_latent: cute.Tensor,
        q_rope: cute.Tensor,
        c_latent: cute.Tensor,
        c_rope: cute.Tensor,
        page_offsets: cute.Tensor,
        o: cute.Tensor,
        lse: cute.Tensor,
        workspace: cute.Tensor,
        split_kv: cutlass.Int32,
        cache_seqs: cute.Tensor,
        cu_seqlens_q: cute.Tensor | None,
        block_split_kvs: cute.Tensor,
        softmax_scale: cutlass.Float32,
        output_scale: cutlass.Float32,
        stream: object,
    ):
        """Execute throughput-latency 1CTA MLA with fixed or compact ragged Q."""
        cfg = self._make_config()

        # Public fixed tensors use [H,D,SQ,B]/[H,SQ,B]; compact ragged tensors
        # use [H,D,totalQ]/[H,totalQ]. Both flatten H x Q for TMA because the
        # scheduler operates on groups_tokens_heads_q rows. Resource-level
        # predicates own the padded final group and map outputs back to storage.
        tma_box0 = min(128 // cfg.qkv_dtype_bytes, cfg.head_dim_per_stage_kv)
        tma_page_tokens = cfg.num_tokens_per_page
        if cutlass.const_expr(q_latent.stride[1] != 1 or q_rope.stride[1] != 1):
            raise ValueError("q_latent and q_rope must have leading dimension 1")
        runtime_assert(
            q_latent.stride[2] == q_latent.shape[0] * q_latent.stride[0],
            "q_latent must be compact across the head and query dimensions",
        )
        runtime_assert(
            q_rope.stride[2] == q_rope.shape[0] * q_rope.stride[0],
            "q_rope must be compact across the head and query dimensions",
        )
        if cutlass.const_expr(c_latent.stride[1] != 1 or c_rope.stride[1] != 1):
            raise ValueError("c_latent and c_rope must have leading dimension 1")
        if cutlass.const_expr(o.stride[1] != 1):
            raise ValueError("o must have leading dimension 1")
        runtime_assert(
            o.stride[0] == o.shape[1] * o.stride[1],
            "o must be compact from the dimension axis into the head axis",
        )
        runtime_assert(
            o.stride[2] == o.shape[0] * o.stride[0],
            "o must be compact from the head axis into the query axis",
        )
        if cutlass.const_expr(cu_seqlens_q is None):
            runtime_assert(
                o.stride[3] == o.shape[2] * o.stride[2],
                "o must be compact from the query axis into the batch axis",
            )
        if cutlass.const_expr(lse.stride[0] != 1):
            raise ValueError("lse must have leading dimension 0")
        runtime_assert(
            lse.stride[1] == lse.shape[0] * lse.stride[0],
            "lse must be compact from the head axis into the query axis",
        )
        if cutlass.const_expr(cu_seqlens_q is None):
            runtime_assert(
                lse.stride[2] == lse.shape[1] * lse.stride[1],
                "lse must be compact from the query axis into the batch axis",
            )
        else:
            runtime_assert(
                cute.size(cu_seqlens_q) == Int32(cfg.batch_size + 1),
                "cu_seqlens_q must contain batch_size + 1 offsets",
            )
        if cutlass.const_expr(
            cfg.use_multi_ctas_kv == 1
            and cfg.use_cluster_reduction != 1
            and workspace is None
        ):
            raise ValueError(
                "multi-CTA-KV throughput-latency 1CTA MLA requires workspace"
            )
        workspace_split_kv = split_kv
        if cutlass.const_expr(cfg.use_multi_ctas_kv == 1):
            workspace_split_kv = cutlass.Int32(cfg.num_ctas_per_seq_kv)
            runtime_assert(
                split_kv >= workspace_split_kv,
                "split_kv is smaller than the configured multi-CTA-KV split count",
            )

        if cutlass.const_expr(cu_seqlens_q is not None):
            q_latent_tma = cute.make_tensor(
                q_latent.iterator,
                cute.make_layout(
                    (q_latent.shape[1], q_latent.shape[0] * q_latent.shape[2]),
                    stride=(q_latent.stride[1], q_latent.stride[0]),
                ),
            )
            tma_desc_q_latent = create_tensor_map_ragged_from_tensor(
                q_latent_tma,
                box_dims=(tma_box0, cfg.tile_size_q),
                ragged_dim=1,
                stride_order=(0, 1),
                swizzle=cuda.TensorMapSwizzle.s128b,
                l2_promotion=cuda.TensorMapL2Promotion.l2_128b,
            )
        else:
            q_latent_tma = cute.make_tensor(
                q_latent.iterator,
                cute.make_layout(
                    (
                        q_latent.shape[1],
                        q_latent.shape[0] * q_latent.shape[2],
                        q_latent.shape[3],
                    ),
                    stride=(
                        q_latent.stride[1],
                        q_latent.stride[0],
                        q_latent.stride[3],
                    ),
                ),
            )
            tma_desc_q_latent = create_tensor_map_tiled_from_view(
                q_latent_tma,
                box_dims=(tma_box0, cfg.tile_size_q, 1),
                stride_order=(0, 1, 2),
                swizzle=cuda.TensorMapSwizzle.s128b,
                l2_promotion=cuda.TensorMapL2Promotion.l2_128b,
            )

        q_rope_swizzle = cuda.TensorMapSwizzle.s128b
        if cutlass.const_expr(cfg.is_fp8_qkv() and cfg.rope_dim == 64):
            q_rope_swizzle = cuda.TensorMapSwizzle.s64b
        if cutlass.const_expr(cu_seqlens_q is not None):
            q_rope_tma = cute.make_tensor(
                q_rope.iterator,
                cute.make_layout(
                    (q_rope.shape[1], q_rope.shape[0] * q_rope.shape[2]),
                    stride=(q_rope.stride[1], q_rope.stride[0]),
                ),
            )
            tma_desc_q_rope = create_tensor_map_ragged_from_tensor(
                q_rope_tma,
                box_dims=(min(tma_box0, cfg.rope_dim), cfg.tile_size_q),
                ragged_dim=1,
                stride_order=(0, 1),
                swizzle=q_rope_swizzle,
                l2_promotion=cuda.TensorMapL2Promotion.l2_128b,
            )
        else:
            q_rope_tma = cute.make_tensor(
                q_rope.iterator,
                cute.make_layout(
                    (
                        q_rope.shape[1],
                        q_rope.shape[0] * q_rope.shape[2],
                        q_rope.shape[3],
                    ),
                    stride=(
                        q_rope.stride[1],
                        q_rope.stride[0],
                        q_rope.stride[3],
                    ),
                ),
            )
            tma_desc_q_rope = create_tensor_map_tiled_from_view(
                q_rope_tma,
                box_dims=(min(tma_box0, cfg.rope_dim), cfg.tile_size_q, 1),
                stride_order=(0, 1, 2),
                swizzle=q_rope_swizzle,
                l2_promotion=cuda.TensorMapL2Promotion.l2_128b,
            )

        c_latent_tma = cute.make_tensor(
            c_latent.iterator,
            cute.select(c_latent.layout, mode=[1, 0, 2]),
        )
        tma_desc_c_latent = create_tensor_map_tiled_from_view(
            c_latent_tma,
            box_dims=(tma_box0, tma_page_tokens, 1),
            stride_order=(0, 1, 2),
            swizzle=cuda.TensorMapSwizzle.s128b,
            l2_promotion=cuda.TensorMapL2Promotion.l2_128b,
        )

        c_rope_tma = cute.make_tensor(
            c_rope.iterator,
            cute.select(c_rope.layout, mode=[1, 0, 2]),
        )
        c_rope_swizzle = cuda.TensorMapSwizzle.s128b
        if cutlass.const_expr(cfg.is_fp8_qkv() and cfg.rope_dim == 64):
            c_rope_swizzle = cuda.TensorMapSwizzle.s64b
        tma_desc_c_rope = create_tensor_map_tiled_from_view(
            c_rope_tma,
            box_dims=(min(tma_box0, cfg.rope_dim), tma_page_tokens, 1),
            stride_order=(0, 1, 2),
            swizzle=c_rope_swizzle,
            l2_promotion=cuda.TensorMapL2Promotion.l2_128b,
        )

        softmax_scale_log2 = softmax_scale * LOG2_E
        use_gmem_reduction = cutlass.const_expr(
            cfg.use_multi_ctas_kv == 1 and cfg.use_cluster_reduction != 1
        )
        acc_o, acc_lse = self.initialize_workspace(
            cutlass.Int32(cfg.num_heads_q),
            cfg.head_dim_v,
            cutlass.Int32(cfg.seq_len_q),
            cutlass.Int32(cfg.batch_size),
            workspace_split_kv,
            workspace if use_gmem_reduction else None,
        )

        use_clc_dynamic = cutlass.const_expr(
            cfg.use_clc_dynamic_persistent_scheduler == 1
        )
        use_static_persistent = cutlass.const_expr(
            cfg.use_persistent_scheduler == 1
            and cfg.use_clc_dynamic_persistent_scheduler != 1
        )
        tile_sched_params = None
        if cutlass.const_expr(use_clc_dynamic):
            tile_sched_params = utils.ClcDynamicPersistentTileSchedulerParams(
                (
                    cfg.num_ctas_per_seq_q,
                    cfg.num_ctas_per_head_dim,
                    cfg.batch_size * cfg.num_ctas_for_all_heads,
                ),
                (1, 1, 1),
            )
            grid = tile_sched_params.get_grid_shape()
        elif cutlass.const_expr(use_static_persistent):
            tile_sched_params = utils.PersistentTileSchedulerParams(
                (
                    cfg.num_ctas_per_seq_q,
                    cfg.num_ctas_per_head_dim,
                    cfg.batch_size * cfg.num_ctas_for_all_heads,
                ),
                (1, 1, 1),
            )
            grid = utils.StaticPersistentTileScheduler.get_grid_shape(
                tile_sched_params,
                self.max_active_clusters,
            )
        else:
            grid = (
                cfg.num_ctas_for_all_heads
                * cfg.num_ctas_per_seq_q
                * cfg.num_ctas_per_seq_kv,
                cfg.num_ctas_per_head_dim,
                cfg.batch_size,
            )
        cluster_shape = None
        if cutlass.const_expr(cfg.use_cluster_reduction == 1):
            cluster_shape = (cfg.num_ctas_per_seq_kv, 1, 1)
        self.dense_kernel(
            tma_desc_q_latent,
            tma_desc_q_rope,
            tma_desc_c_latent,
            tma_desc_c_rope,
            tma_desc_c_latent,
            c_rope,
            page_offsets,
            o,
            lse,
            acc_o,
            acc_lse,
            cache_seqs,
            cu_seqlens_q,
            softmax_scale_log2,
            output_scale,
            tile_sched_params,
        ).launch(
            grid=grid,
            block=[cfg.threads_per_cta, 1, 1],
            cluster=cluster_shape,
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=acc_o is not None,
        )
        if cutlass.const_expr(acc_o is not None):
            if cutlass.const_expr(self.use_parallel_reduction):
                topology = self.parallel_reduction_topology
                reduction_grid, reduction_cluster = (
                    parallel_gmem_reduction_launch_shape(
                        cfg,
                        topology,
                        self.parallel_reduction_elements_per_slice,
                    )
                )
                reduction_threads = parallel_gmem_reduction_threads(
                    self.parallel_reduction_elements_per_slice
                )
                parallel_reducer = self.parallel_gmem_reduction_kernel(
                    o,
                    lse,
                    acc_o,
                    acc_lse,
                    cache_seqs,
                    cu_seqlens_q,
                )
                if cutlass.const_expr(topology.cluster_size == 1):
                    parallel_reducer.launch(
                        grid=reduction_grid,
                        block=[reduction_threads, 1, 1],
                        stream=stream,
                        min_blocks_per_mp=1,
                        use_pdl=True,
                    )
                else:
                    parallel_reducer.launch(
                        grid=reduction_grid,
                        block=[reduction_threads, 1, 1],
                        cluster=reduction_cluster,
                        stream=stream,
                        min_blocks_per_mp=1,
                        use_pdl=True,
                    )
                return
            (
                reduction_grid,
                reduction_smem,
                reduction_threads,
                reduction_ctas,
            ) = gmem_reduction_launch_shape(
                cfg,
                cfg.seq_len_q,
                cfg.batch_size,
                self.lse_dtype.width,
                self.max_active_clusters,
            )
            self.gmem_reduction_kernel(
                o,
                lse,
                acc_o,
                acc_lse,
                cache_seqs,
                cu_seqlens_q,
                cutlass.Int32(reduction_ctas),
            ).launch(
                grid=reduction_grid,
                block=[reduction_threads, 1, 1],
                smem=reduction_smem,
                stream=stream,
                min_blocks_per_mp=1,
                use_pdl=True,
            )

    @cute.kernel
    def dense_kernel(
        self,
        tma_desc_q_latent: cutlass.GridConstant[cuda.TensorMap],
        tma_desc_q_rope: cutlass.GridConstant[cuda.TensorMap],
        tma_desc_c_latent: cutlass.GridConstant[cuda.TensorMap],
        tma_desc_c_rope: cutlass.GridConstant[cuda.TensorMap],
        tma_desc_v: cutlass.GridConstant[cuda.TensorMap],
        c_rope: cute.Tensor,
        page_offsets: cute.Tensor,
        o: cute.Tensor,
        lse: cute.Tensor,
        acc_o: cute.Tensor,
        acc_lse: cute.Tensor,
        cache_seqs: cute.Tensor,
        cu_seqlens_q: cute.Tensor,
        softmax_scale_log2: cutlass.Float32,
        output_scale: cutlass.Float32,
        tile_sched_params: object,
    ):
        """Execute one groups_tokens_heads_q, batch, KV-split, and V head-dimension tile."""
        cfg = self._make_config()

        # The grid is expressed in effective grouped-Q coordinates. Decode it
        # once here; Q/O resources retain responsibility for logical row
        # mapping, compact-ragged offsets, and padded-tail publication.
        cta_idx_x, cta_idx_head_dim_v, batch_idx = cute.arch.block_idx()
        if cutlass.const_expr(cfg.use_cluster_reduction == 1):
            cta_idx_x = cta_idx_x // Int32(cfg.num_ctas_per_seq_kv)
            ctas_per_head_tile = Int32(cfg.num_ctas_per_seq_q)
            cta_idx_head_q = cta_idx_x // ctas_per_head_tile
            cta_idx_q = cta_idx_x - cta_idx_head_q * ctas_per_head_tile
            cta_idx_kv = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        else:
            ctas_per_head_tile = Int32(cfg.num_ctas_per_seq_q * cfg.num_ctas_per_seq_kv)
            cta_idx_head_q = cta_idx_x // ctas_per_head_tile
            cta_idx_x_in_head = cta_idx_x - cta_idx_head_q * ctas_per_head_tile
            cta_idx_q = cta_idx_x_in_head // Int32(cfg.num_ctas_per_seq_kv)
            cta_idx_kv = cta_idx_x_in_head - cta_idx_q * Int32(cfg.num_ctas_per_seq_kv)
        head_idx = cta_idx_head_q * Int32(cfg.tile_size_q)
        run_task_graph = cutlass.Boolean(True)
        query_tile_is_active = cutlass.Boolean(True)
        split_tile_is_active = cutlass.Boolean(True)
        enable_runtime_split_pruning = (
            cfg.use_multi_ctas_kv == 1
            and runtime_split_pruning_is_profitable(cfg.num_ctas_per_seq_kv)
        )
        has_runtime_activity_guard = tile_sched_params is None and (
            cu_seqlens_q is not None or enable_runtime_split_pruning
        )
        if cutlass.const_expr(has_runtime_activity_guard):
            # Decode launch coordinates against the configured maximum grid,
            # then independently drop compact-Q padding and profitable split
            # suffixes. S2/S3 retain configured split work: too few mainloop
            # CTAs can retire to amortize the activity branch plus mandatory
            # neutral publication/reduction. Q padding remains prunable.
            if cutlass.const_expr(cu_seqlens_q is not None):
                query_tile_is_active = runtime_query_tile_is_active(
                    cfg,
                    cu_seqlens_q,
                    batch_idx,
                    cta_idx_q,
                )
                query_tile_is_active = cute.arch.make_warp_uniform(query_tile_is_active)
            if cutlass.const_expr(enable_runtime_split_pruning):
                split_tile_is_active = runtime_split_tile_is_active(
                    cfg,
                    cache_seqs,
                    cu_seqlens_q,
                    batch_idx,
                    cta_idx_q,
                    cta_idx_kv,
                )
                split_tile_is_active = cute.arch.make_warp_uniform(split_tile_is_active)
            run_task_graph = query_tile_is_active and split_tile_is_active
        if cutlass.const_expr(tile_sched_params is not None):
            cta_idx_q = None
            batch_idx = None
            cta_idx_head_dim_v = None
            cta_idx_kv = Int32(0)
            head_idx = None
        use_clc_dynamic = cutlass.const_expr(
            cfg.use_clc_dynamic_persistent_scheduler == 1
        )
        use_static_persistent = cutlass.const_expr(
            cfg.use_persistent_scheduler == 1
            and cfg.use_clc_dynamic_persistent_scheduler != 1
        )
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        if warp_idx == Int32(cfg.load_warp_idx):
            prims.prefetch_tensormap(tma_desc_q_latent.get_ptr())
            prims.prefetch_tensormap(tma_desc_q_rope.get_ptr())
            prims.prefetch_tensormap(tma_desc_c_latent.get_ptr())
            prims.prefetch_tensormap(tma_desc_c_rope.get_ptr())
            prims.prefetch_tensormap(tma_desc_v.get_ptr())

        clc_response_ptr = None
        if cutlass.const_expr(tile_sched_params is not None):
            clc_response_ptr = cute.arch.alloc_smem(cutlass.Int128, 2)

        task_manager, cluster_corr_resource = build_throughput_latency_mla_task_manager(
            cfg,
            total_kv_tiles=cfg.total_kv_tiles,
            use_page_offsets=True,
            tma_desc_q_latent=tma_desc_q_latent.get_ptr(),
            tma_desc_q_rope=tma_desc_q_rope.get_ptr(),
            tma_desc_c_latent=tma_desc_c_latent.get_ptr(),
            tma_desc_c_rope=tma_desc_c_rope.get_ptr(),
            tma_desc_v=tma_desc_v.get_ptr(),
            c_rope_tensor=c_rope,
            page_offsets=page_offsets,
            cache_seqs=cache_seqs,
            cu_seqlens_q=cu_seqlens_q,
            head_idx=head_idx,
            batch_idx=batch_idx,
            cta_idx_q=cta_idx_q,
            cta_idx_kv=cta_idx_kv,
            cta_idx_head_dim_v=cta_idx_head_dim_v,
            scale_softmax_log2=softmax_scale_log2,
            output_scale=output_scale,
            o_tensor=o,
            lse_tensor=lse,
            acc_o_tensor=acc_o,
            acc_lse_tensor=acc_lse,
            tile_sched_params=tile_sched_params,
            clc_response_ptr=clc_response_ptr,
            use_clc_dynamic_scheduler=use_clc_dynamic,
            use_static_persistent_scheduler=use_static_persistent,
        )
        task_manager.setup_resources_and_tasks()
        smem_allocator = task_manager.smem_allocator
        assert smem_allocator is not None
        tmem_ptr_alloc = smem_allocator.tmem_ptr_alloc
        assert tmem_ptr_alloc is not None
        tmem_ptr_i32 = smem_allocator.get(tmem_ptr_alloc)

        if cutlass.const_expr(cfg.use_cluster_reduction == 1):
            resource_context = ResourceContext(
                smem_base=smem_allocator.smem_base,
                tmem_ptr_i32=tmem_ptr_i32,
            )
            cluster_corr_resource.create_cluster_function_variables(resource_context)

        prims.fence_mbarrier_init()
        if cutlass.const_expr(cfg.use_cluster_reduction == 1):
            prims.barrier_cluster_arrive_relaxed()
            prims.barrier_cluster_wait()
        prims.barrier_cta_sync(
            barrier_id=TMEM_LIFECYCLE_BARRIER_ID,
            thread_count=cfg.threads_per_cta,
        )

        if warp_idx == Int32(cfg.mma_warp_idx):
            prims.tcgen05_alloc(
                tmem_ptr_i32,
                cfg.tmem_alloc_cols,
                group=prims.CTAGroup.CTA_1,
            )
            prims.tcgen05_relinquish_alloc_permit(group=prims.CTAGroup.CTA_1)

        prims.barrier_cta_sync(
            barrier_id=TMEM_LIFECYCLE_BARRIER_ID,
            thread_count=cfg.threads_per_cta,
        )
        if cutlass.const_expr(has_runtime_activity_guard):
            if run_task_graph:
                task_manager.run()
            elif query_tile_is_active:
                if cutlass.const_expr(cfg.use_cluster_reduction == 1):
                    # Publish an online-softmax identity and retain this rank's
                    # configured cluster row-owner reduction duties.
                    cluster_corr_resource.publish_neutral_cluster_partial_and_reduce(
                        batch_idx,
                        head_idx,
                        cta_idx_q,
                        cta_idx_kv,
                        cta_idx_head_dim_v,
                    )
                elif cutlass.const_expr(acc_o is not None):
                    # Standalone reducers retain configured split loops, so a
                    # pruned producer must overwrite its workspace slot before
                    # releasing the PDL-dependent reducer.
                    _publish_neutral_standalone_partial(
                        cfg,
                        acc_o,
                        acc_lse,
                        batch_idx,
                        cta_idx_q,
                        cta_idx_kv,
                        cta_idx_head_dim_v,
                        head_idx,
                    )
        else:
            # In particular, SQ1/S2 without packed-Q metadata lowers to the
            # original straight-line task graph with no runtime activity test.
            task_manager.run()
        prims.barrier_cta_sync(
            barrier_id=TMEM_LIFECYCLE_BARRIER_ID,
            thread_count=cfg.threads_per_cta,
        )

        if warp_idx == Int32(cfg.mma_warp_idx):
            tmem_arr_for_dealloc = prims.make_tmem_ptr(
                tmem_ptr_i32.load(),
                Float32,
            )
            prims.tcgen05_dealloc(
                tmem_arr_for_dealloc,
                cfg.tmem_alloc_cols,
                group=prims.CTAGroup.CTA_1,
            )
        if cutlass.const_expr(acc_o is not None):
            # Every CTA in the producer grid must release the dependent
            # reducer, including runtime-padded Q/split CTAs that skipped the
            # task graph. One elected thread emits the convergent CTA signal.
            thread_idx, _, _ = cute.arch.thread_idx()
            if thread_idx == Int32(0):
                prims.griddepcontrol(kind=prims.GridDepAction.LAUNCH_DEPENDENTS)

    @cute.kernel
    def gmem_reduction_kernel(
        self,
        output: cute.Tensor,
        lse: cute.Tensor,
        acc_output: cute.Tensor,
        acc_lse: cute.Tensor,
        cache_seqs: cute.Tensor,
        cu_seqlens_q: cute.Tensor,
        num_reduction_ctas: cutlass.Int32,
    ):
        """Dispatch the throughput-latency split-KV reduction body."""
        cfg = self._make_config()
        run_gmem_reduction_kernel(
            self,
            output,
            lse,
            acc_output,
            acc_lse,
            cache_seqs,
            cu_seqlens_q,
            cfg,
            num_reduction_ctas,
        )

    @cute.kernel
    def parallel_gmem_reduction_kernel(
        self,
        output: cute.Tensor,
        lse: cute.Tensor,
        acc_output: cute.Tensor,
        acc_lse: cute.Tensor,
        cache_seqs: cute.Tensor,
        cu_seqlens_q: cute.Tensor,
    ):
        """Dispatch the automatically selected parallel standalone reducer."""

        cfg = self._make_config()
        topology = self.parallel_reduction_topology
        run_parallel_gmem_reduction_kernel(
            output,
            lse,
            acc_output,
            acc_lse,
            cache_seqs,
            cu_seqlens_q,
            cfg,
            topology.cluster_size,
            topology.slots_per_rank,
            topology.actual_splits,
            self.parallel_reduction_elements_per_slice,
        )
