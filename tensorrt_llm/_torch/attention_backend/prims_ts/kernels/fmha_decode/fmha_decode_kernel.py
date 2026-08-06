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

"""FMHA decode TS kernel assembly.

Assembles all resources, tasks, pipeline configs, and the dependency graph
into a TaskManager for the SwapsMmaAb decode kernel.

SwapsMmaAb, also shortened to swapsAb, names the MMA layout choice that maps
the logical attention operands onto the opposite MMA A/B roles from the
textbook QK form. BMM1 issues K as the MMA A operand and Q as the MMA B
operand, producing S = K * Q^T so KV tokens occupy the MMA M axis and the GQA
head group occupies the small N axis. BMM2 follows the same convention with V
as A and P as B.

Entry points:
  - build_decode_task_manager()  — pure Python, validation only (no GPU)
  - FmhaDecodeTs                — GPU kernel class with @cute.jit + @cute.kernel
"""

import math

import cutlass
import cutlass.experimental.cuda as cuda
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cuda.bindings import driver as cuda_drv
from cutlass import Float32, Int32, Int64
from cutlass.experimental import primitives as prims
from ..tensor_map import (
    create_tensor_map_ragged_from_tensor,
    create_tensor_map_tiled_from_view,
)

from cutlass.experimental.task_scheduling.enums import PipelineType
from cutlass.experimental.task_scheduling.memory import (
    ResourceContext,
    SmemAllocation,
    SmemAllocator,
    TmemAllocator,
)
from cutlass.experimental.task_scheduling.resources import (
    MemoryResource,
    PipelineConfig,
    TileSchedulerConfig,
    WorkQueue,
)
from cutlass.experimental.task_scheduling.task import Task
from cutlass.experimental.task_scheduling.task_manager import TaskManager

from .fmha_decode_config import FmhaDecodeConfig, validate_paged_kv_staging_config
from .fmha_decode_constants import KV_KIND_K, KV_KIND_V
from .fmha_decode_resources import (
    SmemKvTileResource,
    SmemKvResource,
    SmemPageOffsetsKvResource,
    SmemQResource,
    TmemCorrResource,
    TmemOResource,
    SmemPResource,
    TmemSResource,
    TmemStatsDoneResource,
    TmemSoftmaxGlobalResource,
    TmemSoftmaxLocalResource,
    TmemSoftmaxOrderResource,
)
from .fmha_decode_resources.helpers_common import (
    _q_group_token_base,
    _q_seq_bounds,
)
from .fmha_decode_resources.helpers_kv_tile_idx import _runtime_active_splits_kv
from .fmha_decode_tasks import (
    PackedDecodeWorkQueue,
    ScheduleTokenThrottleResource,
    create_correction_task,
    create_correction_task_one_inst_qkv,
    create_load_task,
    create_load_task_one_inst_qkv,
    create_load_task_split_kv,
    create_mma_task,
    create_mma_task_one_inst_qkv,
    create_mma_task_split_kv,
    create_page_offsets_task,
    create_page_offsets_task_one_inst_qkv,
    create_page_offsets_task_split_kv,
    create_padding_task,
    create_scheduler_task,
    create_softmax0_task,
    create_softmax1_task,
)

from .reduction import (  # noqa: F401
    decode_gen_separate_reduction_kernel,
    fmha_decode_separate_reduction_launch,
)

_PERSISTENT_SCHEDULE_TOKEN_STAGES = 2


def _stages_page_ids_per_tile(uses_paired_page_offset_resources: bool) -> bool:
    """Return whether each published page-offset stage owns exactly one tile."""

    # Shared resources retain an aligned 32-ID window so all lanes issue one
    # coalesced page-table transaction. Paired K0/K1 and V0/V1 resources use
    # exact per-tile stages so a pair may safely cross a 32-ID boundary.
    return uses_paired_page_offset_resources


def _compute_decode_gen_loop_domain(total_kv_tiles: int, num_insts_kv: int) -> int:
    """Number of post-head steady-state iterations.

    HEAD consumes the first `num_insts_kv` K tiles. The remaining tiles are
    processed in staggered groups of `num_insts_kv`, so odd tail groups still
    need one final loop iteration, matching the schedule's pull-down behavior.
    """
    remaining_kv_tiles = max(total_kv_tiles - num_insts_kv, 0)
    return (remaining_kv_tiles + num_insts_kv - 1) // num_insts_kv


def _compute_total_kv_tiles(seq_len_kv: int, tile_size_kv: int) -> int:
    """Number of KV tiles needed for a fixed-length launch."""
    return (seq_len_kv + tile_size_kv - 1) // tile_size_kv


def _compute_static_num_skipped_kv_tiles(cfg: FmhaDecodeConfig, seq_len_kv: int) -> int:
    """Return full leading KV tiles skipped by a static sliding window."""
    if not cfg.use_sliding_window_causal or cfg.max_seq_len_q > 1:
        return 0
    return max(seq_len_kv - cfg.attention_window_size, 0) // cfg.tile_size_kv


def _compute_static_window_start_idx(cfg: FmhaDecodeConfig, seq_len_kv: int) -> int:
    """Return the token index where a static sliding window begins."""
    if not cfg.use_sliding_window_causal or cfg.max_seq_len_q > 1:
        return 0
    return max(seq_len_kv - cfg.attention_window_size, 0)


def _configure_static_sliding_window(
    cfg: FmhaDecodeConfig, seq_len_kv: int, bias_kv_tma: bool = False
) -> int:
    """Populate fixed-length sliding metadata and return effective seqLenKv."""
    skipped_tiles = _compute_static_num_skipped_kv_tiles(cfg, seq_len_kv)
    skipped_tokens = skipped_tiles * cfg.tile_size_kv
    window_start_idx = _compute_static_window_start_idx(cfg, seq_len_kv)
    effective_seq_len_kv = seq_len_kv - skipped_tokens
    cfg.use_static_sliding_kv_tma_bias = bias_kv_tma and skipped_tiles > 0
    cfg.static_seq_len_kv = (
        effective_seq_len_kv if cfg.use_static_sliding_kv_tma_bias else seq_len_kv
    )
    cfg.static_num_skipped_kv_tiles = (
        0 if cfg.use_static_sliding_kv_tma_bias else skipped_tiles
    )
    cfg.static_window_start_idx = (
        window_start_idx - skipped_tokens
        if cfg.use_static_sliding_kv_tma_bias
        else window_start_idx
    )
    return effective_seq_len_kv


def _compute_local_kv_tiles(cfg: FmhaDecodeConfig, total_kv_tiles: int) -> int:
    """KV tiles covered by each CtaKv in split-KV mode."""
    if not cfg.use_split_kv:
        return total_kv_tiles
    tiles_per_cta_group = cfg.splits_kv * cfg.num_insts_kv
    num_groups = (total_kv_tiles + tiles_per_cta_group - 1) // tiles_per_cta_group
    return max(
        cfg.num_insts_kv,
        num_groups * cfg.num_insts_kv,
    )


def _build_decode_gen_schedule(
    cfg: FmhaDecodeConfig,
    total_kv_tiles: int | Int32,
    scale_softmax_log2: Float32 | None = None,
    o_ptr: cute.Pointer | None = None,
    output_scale: Float32 | None = None,
    partial_o_ptr: cute.Pointer | None = None,
    partial_stats_ptr: cute.Pointer | None = None,
    split_kv_counter_ptr: cute.Pointer | None = None,
    attention_sinks_ptr: cute.Pointer | None = None,
    seqlens_kv: cute.Pointer | None = None,
    cu_seqlens_q: cute.Pointer | None = None,
    max_seq_len_kv: int | Int32 = 0,
    corr_max_seq_len_kv: int | Int32 | None = None,
    num_heads_kv: Int32 | None = None,
    h_r: Int32 | None = None,
    tma_desc_q: cutlass.Pointer | None = None,
    tma_desc_k: cutlass.Pointer | None = None,
    tma_desc_v: cutlass.Pointer | None = None,
    page_idx_kv: cute.Pointer | None = None,
    h_k_idx: Int32 | None = None,
    b_idx: Int32 | None = None,
    q_group_idx: Int32 | None = None,
    q_token_offset: Int32 | None = None,
    seq_len_q: Int32 | None = None,
    active_splits_kv: Int32 | None = None,
    static_full_split_prefix: bool = False,
    tile_sched_params: utils.ClcDynamicPersistentTileSchedulerParams | None = None,
    clc_response_ptr: cute.Pointer | None = None,
    use_variable_seqlens_kv: bool = False,
    use_native_paged_kv: bool = False,
    use_static_native_seqlens_kv: bool = False,
    paged_kv_indptr: cute.Pointer | None = None,
    paged_kv_indices: cute.Pointer | None = None,
) -> tuple[
    list[Task],
    dict[MemoryResource, list[MemoryResource]],
    dict[tuple[MemoryResource, MemoryResource], set[str]],
    SmemAllocator,
    TmemAllocator,
    list[TmemCorrResource],
]:
    """Build all resources, tasks, and dep graph.

    Parameters
    ----------
    cfg : FmhaDecodeConfig
        Kernel configuration.
    total_kv_tiles : int or Int32
        Total number of KV tiles (seqLenKv / tileSizeKv).
    tma_desc_q/k/v : TMA descriptor pointers (None for validation-only mode).
    h_k_idx, b_idx, q_group_idx : Grid coordinates (None for validation-only mode).
    corr_max_seq_len_kv : Bound passed to TmemCorrResource; defaults to
        ``max_seq_len_kv``. The GPU kernel uses the constexpr ``seq_len_kv``
        here (full sequence) while other resources use the static/varlen
        runtime value.

    Returns
    -------
    tuple
        (task_list, dep_graph, dma labels, smem_allocator, tmem_allocator,
        correction_init_resources)
    """
    if cfg.use_keeps_mma_ab and cfg.num_insts_kv == 1 and not cfg.uses_tmem_p:
        raise ValueError(
            "one-instance KeepsMmaAb is enabled only for the staged headDim=256 "
            "profile with head_dim_per_stage_kv=128 and o_stages=1"
        )
    if use_native_paged_kv and not cfg.use_paged_kv:
        raise ValueError("native paged-KV ABI requires cfg.use_paged_kv=True")
    if cfg.use_paged_kv:
        validate_paged_kv_staging_config(
            tile_size_kv=cfg.tile_size_kv,
            num_tokens_per_page=cfg.num_tokens_per_page,
            page_offsets_num_warps=cfg.page_offsets_num_warps,
        )
    if corr_max_seq_len_kv is None:
        corr_max_seq_len_kv = max_seq_len_kv
    if h_k_idx is None:
        h_k_idx = Int32(0)
    if b_idx is None:
        b_idx = Int32(0)
    if q_group_idx is None:
        q_group_idx = Int32(0)
    if q_token_offset is None:
        q_token_offset = Int32(0)
    if seq_len_q is None:
        seq_len_q = Int32(cfg.max_seq_len_q)

    WARP_SIZE = 32
    Agent = pipeline.Agent
    cta_layout = (1, 1, 1, 1)

    # ------------------------------------------------------------------
    # Cooperative groups
    # ------------------------------------------------------------------
    tma_producer = pipeline.CooperativeGroup(Agent.Thread)
    page_offsets_grp = pipeline.CooperativeGroup(
        Agent.Thread, cfg.page_offsets_num_warps * WARP_SIZE
    )
    load_grp = pipeline.CooperativeGroup(Agent.Thread, cfg.load_num_warps * WARP_SIZE)
    umma_hw = pipeline.CooperativeGroup(Agent.Thread)
    # The staged one-instance S/P overlay uses this group for overwrite credit.
    mma_grp = pipeline.CooperativeGroup(Agent.Thread, cfg.mma_num_warps * WARP_SIZE)
    softmax0_grp = pipeline.CooperativeGroup(
        Agent.Thread, cfg.softmax0_num_warps * WARP_SIZE
    )
    softmax1_grp = pipeline.CooperativeGroup(
        Agent.Thread, cfg.softmax1_num_warps * WARP_SIZE
    )
    correction_grp = pipeline.CooperativeGroup(
        Agent.Thread, cfg.correction_num_warps * WARP_SIZE
    )
    scheduler_grp = pipeline.CooperativeGroup(
        Agent.Thread, cfg.scheduler_num_warps * WARP_SIZE
    )

    # ------------------------------------------------------------------
    # Pipeline configs
    # ------------------------------------------------------------------
    # Leave barrier_ptr unset so SmemAllocator packs every pipeline barrier
    # into the unified block. Separate barrier arrays create an alignment gap
    # before the 1024-byte-aligned data block and overflow near-capacity Q128.
    use_paged_kv = cfg.use_paged_kv
    use_one_inst_qkv = cfg.use_keeps_mma_ab and cfg.num_insts_kv == 1
    one_inst_tmem_stages = 2 if use_one_inst_qkv else 1
    one_inst_kv_stages = cfg.num_head_dim_stages_kv if use_one_inst_qkv else 1
    use_distributed_split_kv_stages = not use_one_inst_qkv
    if cfg.tile_size_q == 128 and use_distributed_split_kv_stages:
        # Q128's four instruction-local K0/K1/V0/V1 rings need equal depth.
        # Round the inferred aggregate budget down to a complete balanced set;
        # on the FP8 decode profile this is 2/2/2/2, matching the roughly
        # 165-KiB staged footprint of the corresponding reference profile.
        balanced_total_stages = max(
            (cfg.kv_stages // (2 * cfg.num_insts_kv)) * cfg.num_insts_kv,
            cfg.num_insts_kv,
        )
        split_total_k_stages = balanced_total_stages
        split_total_v_stages = balanced_total_stages
    else:
        split_total_k_stages = (
            max(cfg.kv_stages // 2, cfg.num_insts_kv)
            if use_distributed_split_kv_stages
            else cfg.num_insts_kv
        )
        split_total_v_stages = (
            max(cfg.kv_stages - split_total_k_stages, cfg.num_insts_kv)
            if use_distributed_split_kv_stages
            else cfg.num_insts_kv
        )
    split_k0_stages = (
        one_inst_kv_stages
        if use_one_inst_qkv
        else max((split_total_k_stages + cfg.num_insts_kv - 1) // cfg.num_insts_kv, 1)
    )
    split_k1_stages = (
        1
        if use_one_inst_qkv
        else max((split_total_k_stages + cfg.num_insts_kv - 2) // cfg.num_insts_kv, 1)
    )
    split_v0_stages = (
        one_inst_kv_stages
        if use_one_inst_qkv
        else max((split_total_v_stages + cfg.num_insts_kv - 1) // cfg.num_insts_kv, 1)
    )
    split_v1_stages = (
        1
        if use_one_inst_qkv
        else max((split_total_v_stages + cfg.num_insts_kv - 2) // cfg.num_insts_kv, 1)
    )
    use_ordered_softmax_barrier = (
        not use_one_inst_qkv and cfg.uses_ordered_softmax_barrier
    )
    # A two-inst Keeps profile can use the deeper shared K/V FIFO when stats
    # are standalone and P remains in SMEM.  Keep instruction-local FIFOs when
    # stats or TMEM-P alias S: their overwrite-credit cadence is tied to each
    # instruction.  Swaps always uses the shared FIFO, including staged H256.
    # With cfg.keeps_stats_via_smem the stats-alias justification no longer
    # applies, but the shared FIFO still causes a material Q128 regression, so
    # the instruction-local FIFO gate remains part of that kernel policy.
    use_split_head_dim_kv = cfg.use_keeps_mma_ab and (
        not cfg.keeps_separates_tmem_s_and_stats or cfg.uses_two_inst_tmem_p
    )
    split_head_dim_page_offsets_kv = (
        use_paged_kv and use_split_head_dim_kv and not use_one_inst_qkv
    )
    # Paired resources publish independent K0/K1 and V0/V1 stages. Shared
    # split-KV retains the aligned 32-ID representation for its optional
    # native held-window path.
    stage_page_ids_per_tile = _stages_page_ids_per_tile(
        split_head_dim_page_offsets_kv,
    )

    smem_q_cfg = PipelineConfig.create_tma_umma_pipeline_cfg(
        num_stages=cfg.q_stages,
        num_bytes=cfg.smem_q_tile_bytes,
        producer_group=tma_producer,
        consumer_group=umma_hw,
        cta_layout_vmnk=cta_layout,
        advance_on_wait=True,
    )
    smem_kv_cfg = PipelineConfig.create_tma_umma_pipeline_cfg(
        num_stages=cfg.kv_stages,
        num_bytes=cfg.smem_kv_tile_bytes,
        producer_group=tma_producer,
        consumer_group=umma_hw,
        cta_layout_vmnk=cta_layout,
        advance_on_wait=True,
    )
    smem_k0_cfg = PipelineConfig.create_tma_umma_pipeline_cfg(
        num_stages=split_k0_stages,
        num_bytes=cfg.smem_kv_tile_bytes,
        producer_group=tma_producer,
        consumer_group=umma_hw,
        cta_layout_vmnk=cta_layout,
        advance_on_wait=True,
    )
    smem_k1_cfg = PipelineConfig.create_tma_umma_pipeline_cfg(
        num_stages=split_k1_stages,
        num_bytes=cfg.smem_kv_tile_bytes,
        producer_group=tma_producer,
        consumer_group=umma_hw,
        cta_layout_vmnk=cta_layout,
        advance_on_wait=True,
    )
    smem_v0_cfg = PipelineConfig.create_tma_umma_pipeline_cfg(
        num_stages=split_v0_stages,
        num_bytes=cfg.smem_kv_tile_bytes,
        producer_group=tma_producer,
        consumer_group=umma_hw,
        cta_layout_vmnk=cta_layout,
        advance_on_wait=True,
    )
    smem_v1_cfg = PipelineConfig.create_tma_umma_pipeline_cfg(
        num_stages=split_v1_stages,
        num_bytes=cfg.smem_kv_tile_bytes,
        producer_group=tma_producer,
        consumer_group=umma_hw,
        cta_layout_vmnk=cta_layout,
        advance_on_wait=True,
    )

    def _make_page_offsets_cfg(num_stages: int | None = None) -> PipelineConfig:
        """Create the async page-offsets pipeline for the selected stage count."""
        if num_stages is None:
            num_stages = cfg.page_offsets_stages
        return PipelineConfig(
            num_stages=num_stages,
            num_bytes=0,
            producer_group=page_offsets_grp,
            consumer_group=load_grp,
            pipeline_type=PipelineType.AsyncAsync,
            cta_layout_vmnk=cta_layout,
            advance_on_wait=True,
        )

    smem_page_offsets_cfg = None
    smem_page_offsets_v_cfg = None
    if use_paged_kv:
        page_offsets_stages = (
            3 if split_head_dim_page_offsets_kv else cfg.page_offsets_stages
        )
        smem_page_offsets_cfg = _make_page_offsets_cfg(page_offsets_stages)
        if split_head_dim_page_offsets_kv:
            smem_page_offsets_v_cfg = _make_page_offsets_cfg(page_offsets_stages)
    # tmem_s0/s1, smem_p0/p1, tmem_o, softmax_local cfgs go through direct
    # PipelineConfig() so advance_on_wait=True can be set (factories don't
    # expose it).
    tmem_s0_cfg = PipelineConfig(
        num_stages=one_inst_tmem_stages,
        num_bytes=0,
        producer_group=umma_hw,
        consumer_group=softmax0_grp,
        pipeline_type=PipelineType.UmmaAsync,
        cta_layout_vmnk=cta_layout,
        advance_on_wait=True,
    )
    tmem_s1_cfg = PipelineConfig(
        num_stages=1,
        num_bytes=0,
        producer_group=umma_hw,
        consumer_group=softmax1_grp,
        pipeline_type=PipelineType.UmmaAsync,
        cta_layout_vmnk=cta_layout,
        advance_on_wait=True,
    )
    smem_p0_cfg = PipelineConfig(
        num_stages=one_inst_tmem_stages,
        num_bytes=0,
        producer_group=softmax0_grp,
        consumer_group=umma_hw,
        pipeline_type=PipelineType.AsyncUmma,
        cta_layout_vmnk=cta_layout,
        advance_on_wait=True,
    )
    smem_p1_cfg = PipelineConfig(
        num_stages=1,
        num_bytes=0,
        producer_group=softmax1_grp,
        consumer_group=umma_hw,
        pipeline_type=PipelineType.AsyncUmma,
        cta_layout_vmnk=cta_layout,
        advance_on_wait=True,
    )
    tmem_o_cfg = PipelineConfig(
        num_stages=cfg.o_stages,
        num_bytes=0,
        producer_group=umma_hw,
        consumer_group=correction_grp,
        pipeline_type=PipelineType.UmmaAsync,
        cta_layout_vmnk=cta_layout,
        advance_on_wait=True,
    )
    softmax_local0_cfg = PipelineConfig(
        num_stages=one_inst_tmem_stages,
        num_bytes=0,
        producer_group=softmax0_grp,
        consumer_group=correction_grp,
        pipeline_type=PipelineType.AsyncAsync,
        cta_layout_vmnk=cta_layout,
        advance_on_wait=True,
    )

    softmax_local1_cfg = PipelineConfig(
        num_stages=1,
        num_bytes=0,
        producer_group=softmax1_grp,
        consumer_group=correction_grp,
        pipeline_type=PipelineType.AsyncAsync,
        cta_layout_vmnk=cta_layout,
        advance_on_wait=True,
    )

    # Two-instance Keeps keeps stats outside S and orders each same-instance PV
    # before the next QK, so it needs no stats-done credit.
    # The staged one-instance path needs an overwrite-credit gate across its
    # double-buffered S/P overlay: correction returns the stage credit before
    # MMA can reissue QK into those columns.
    stats_done0_cfg = None
    stats_done1_cfg = None
    if use_one_inst_qkv:
        stats_done0_cfg = PipelineConfig.create_async_async_pipeline_cfg(
            num_stages=one_inst_tmem_stages,
            producer_group=mma_grp,
            consumer_group=correction_grp,
            cta_layout_vmnk=cta_layout,
        )

    # tmemSoftmaxGlobal, tmemCorr: no pipeline (pipeline_config=None)

    # ------------------------------------------------------------------
    # Create resources
    # ------------------------------------------------------------------
    work_queue = None
    schedule_token_throttle = None
    # CLC remains the single persistent policy for every supported topology.
    # The stock static WorkQueue advances and decodes coordinates separately
    # in every task, which regresses multi-wave decode workloads. CLC computes
    # each schedule token once on the scheduler warp and broadcasts it to the workers.
    use_clc_dynamic = cfg.use_persistent_scheduler
    if use_clc_dynamic:
        num_consumer_threads = 16 * WARP_SIZE
        wq_pipeline_config = PipelineConfig.create_clc_fetch_async_pipeline_cfg(
            num_stages=_PERSISTENT_SCHEDULE_TOKEN_STAGES,
            num_bytes=16,
            producer_group=pipeline.CooperativeGroup(Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                Agent.Thread, num_consumer_threads
            ),
            cta_layout_vmnk=cta_layout,
        )
        work_queue_kwargs = {
            "tile_scheduler_config": TileSchedulerConfig.create_clc_dynamic_persistent_tile_scheduler_params(
                tile_scheduler_params=tile_sched_params,
                response_ptr=clc_response_ptr,
            ),
            "pipeline_config": wq_pipeline_config,
            "name": "work_queue",
        }
        if cfg.use_variable_seqlens_q:
            work_queue = PackedDecodeWorkQueue(
                cfg=cfg,
                cu_seqlens_q=cu_seqlens_q,
                **work_queue_kwargs,
            )
        else:
            work_queue = WorkQueue(**work_queue_kwargs)
        schedule_token_throttle = ScheduleTokenThrottleResource(
            pipeline_config=PipelineConfig.create_async_async_pipeline_cfg(
                num_stages=_PERSISTENT_SCHEDULE_TOKEN_STAGES,
                producer_group=load_grp,
                consumer_group=scheduler_grp,
                cta_layout_vmnk=cta_layout,
            ),
            name="schedule_token_throttle",
        )
    smem_q = SmemQResource(
        pipeline_config=smem_q_cfg,
        cfg=cfg,
        tma_desc_q=tma_desc_q,
        h_k_idx=h_k_idx,
        b_idx=b_idx,
        q_group_idx=q_group_idx,
        q_token_offset=q_token_offset,
        seq_len_q=seq_len_q,
        name="smemQ",
    )
    # FlashInfer materializes one canonical sequence-length tensor from its
    # native CSR metadata, so native mode reuses the existing variable-length
    # domain, split, sliding-window, and masking paths.
    use_runtime_seqlens_kv = use_variable_seqlens_kv or (
        use_native_paged_kv and not use_static_native_seqlens_kv
    )
    kv_seqlens = seqlens_kv if use_runtime_seqlens_kv else None
    smem_page_offsets = None
    smem_page_offsets_v = None
    if use_paged_kv:
        smem_page_offsets = SmemPageOffsetsKvResource(
            pipeline_config=smem_page_offsets_cfg,
            cfg=cfg,
            stage_page_ids_per_tile=stage_page_ids_per_tile,
            page_idx_kv=page_idx_kv,
            seqlens_kv=kv_seqlens,
            use_native_paged_kv=use_native_paged_kv,
            paged_kv_indptr=paged_kv_indptr,
            paged_kv_indices=paged_kv_indices,
            max_seq_len_kv=max_seq_len_kv,
            h_k_idx=h_k_idx,
            b_idx=b_idx,
            q_group_idx=q_group_idx,
            seq_len_q=seq_len_q,
            name=(
                "smemPageOffsetsKvK"
                if split_head_dim_page_offsets_kv
                else "smemPageOffsetsKv"
            ),
        )
        if split_head_dim_page_offsets_kv:
            smem_page_offsets_v = SmemPageOffsetsKvResource(
                pipeline_config=smem_page_offsets_v_cfg,
                cfg=cfg,
                stage_page_ids_per_tile=stage_page_ids_per_tile,
                page_idx_kv=page_idx_kv,
                seqlens_kv=kv_seqlens,
                use_native_paged_kv=use_native_paged_kv,
                paged_kv_indptr=paged_kv_indptr,
                paged_kv_indices=paged_kv_indices,
                max_seq_len_kv=max_seq_len_kv,
                h_k_idx=h_k_idx,
                b_idx=b_idx,
                q_group_idx=q_group_idx,
                seq_len_q=seq_len_q,
                name="smemPageOffsetsKvV",
            )
    smem_kv = None
    smem_k0 = None
    smem_k1 = None
    smem_v0 = None
    smem_v1 = None
    if use_split_head_dim_kv:
        smem_k0 = SmemKvTileResource(
            pipeline_config=smem_k0_cfg,
            cfg=cfg,
            tma_desc_k=tma_desc_k,
            tma_desc_v=tma_desc_v,
            page_offsets_kv=smem_page_offsets,
            seqlens_kv=kv_seqlens,
            max_seq_len_kv=max_seq_len_kv,
            h_k_idx=h_k_idx,
            b_idx=b_idx,
            q_group_idx=q_group_idx,
            seq_len_q=seq_len_q,
            inst_id=0,
            kv_kind=KV_KIND_K,
            name="smemK0",
        )
        smem_k1 = SmemKvTileResource(
            pipeline_config=smem_k1_cfg,
            cfg=cfg,
            tma_desc_k=tma_desc_k,
            tma_desc_v=tma_desc_v,
            page_offsets_kv=smem_page_offsets,
            seqlens_kv=kv_seqlens,
            max_seq_len_kv=max_seq_len_kv,
            h_k_idx=h_k_idx,
            b_idx=b_idx,
            q_group_idx=q_group_idx,
            seq_len_q=seq_len_q,
            inst_id=1,
            kv_kind=KV_KIND_K,
            name="smemK1",
        )
        smem_v0 = SmemKvTileResource(
            pipeline_config=smem_v0_cfg,
            cfg=cfg,
            tma_desc_k=tma_desc_k,
            tma_desc_v=tma_desc_v,
            page_offsets_kv=smem_page_offsets_v or smem_page_offsets,
            seqlens_kv=kv_seqlens,
            max_seq_len_kv=max_seq_len_kv,
            h_k_idx=h_k_idx,
            b_idx=b_idx,
            q_group_idx=q_group_idx,
            seq_len_q=seq_len_q,
            inst_id=0,
            kv_kind=KV_KIND_V,
            name="smemV0",
        )
        smem_v1 = SmemKvTileResource(
            pipeline_config=smem_v1_cfg,
            cfg=cfg,
            tma_desc_k=tma_desc_k,
            tma_desc_v=tma_desc_v,
            page_offsets_kv=smem_page_offsets_v or smem_page_offsets,
            seqlens_kv=kv_seqlens,
            max_seq_len_kv=max_seq_len_kv,
            h_k_idx=h_k_idx,
            b_idx=b_idx,
            q_group_idx=q_group_idx,
            seq_len_q=seq_len_q,
            inst_id=1,
            kv_kind=KV_KIND_V,
            name="smemV1",
        )
    else:
        smem_kv = SmemKvResource(
            pipeline_config=smem_kv_cfg,
            cfg=cfg,
            tma_desc_k=tma_desc_k,
            tma_desc_v=tma_desc_v,
            page_offsets_kv=smem_page_offsets,
            seqlens_kv=kv_seqlens,
            max_seq_len_kv=max_seq_len_kv,
            h_k_idx=h_k_idx,
            b_idx=b_idx,
            q_group_idx=q_group_idx,
            seq_len_q=seq_len_q,
            name="smemKv",
        )

    tmem_s0 = TmemSResource(
        inst_id=0,
        pipeline_config=tmem_s0_cfg,
        cfg=cfg,
        scale_softmax_log2=scale_softmax_log2,
        seqlens_kv=kv_seqlens,
        max_seq_len_kv=max_seq_len_kv,
        h_r=h_r,
        q_group_idx=q_group_idx,
        seq_len_q=seq_len_q,
        sync_barrier_id=0,
        name="tmemS0",
    )
    tmem_s1 = TmemSResource(
        inst_id=1,
        pipeline_config=tmem_s1_cfg,
        cfg=cfg,
        scale_softmax_log2=scale_softmax_log2,
        seqlens_kv=kv_seqlens,
        max_seq_len_kv=max_seq_len_kv,
        h_r=h_r,
        q_group_idx=q_group_idx,
        seq_len_q=seq_len_q,
        sync_barrier_id=1,
        name="tmemS1",
    )
    # Packed persistent QK derives the descriptor from Q's just-waited
    # consumer stage, avoiding a routed HEAD-to-LOOP descriptor value across
    # the guarded work-tile region. Fixed/static schedules keep their existing
    # explicit descriptor route.
    tmem_s0.q_ref = smem_q
    tmem_s1.q_ref = smem_q

    smem_p0 = SmemPResource(
        inst_id=0,
        pipeline_config=smem_p0_cfg,
        cfg=cfg,
        scale_softmax_log2=scale_softmax_log2,
        use_variable_seqlens_kv=use_runtime_seqlens_kv,
        name="smemP0",
    )
    smem_p1 = SmemPResource(
        inst_id=1,
        pipeline_config=smem_p1_cfg,
        cfg=cfg,
        scale_softmax_log2=scale_softmax_log2,
        use_variable_seqlens_kv=use_runtime_seqlens_kv,
        name="smemP1",
    )

    tmem_o = TmemOResource(
        pipeline_config=tmem_o_cfg,
        cfg=cfg,
        scale_softmax_log2=scale_softmax_log2,
        name="tmemO",
    )

    tmem_softmax_local0 = TmemSoftmaxLocalResource(
        inst_id=0,
        pipeline_config=softmax_local0_cfg,
        cfg=cfg,
        name="tmemSoftmaxLocal0",
    )
    tmem_softmax_local1 = TmemSoftmaxLocalResource(
        inst_id=1,
        pipeline_config=softmax_local1_cfg,
        cfg=cfg,
        name="tmemSoftmaxLocal1",
    )
    tmem_stats_done0 = (
        TmemStatsDoneResource(
            pipeline_config=stats_done0_cfg,
            name="tmemStatsDone0",
        )
        if stats_done0_cfg is not None
        else None
    )
    tmem_stats_done1 = (
        TmemStatsDoneResource(
            pipeline_config=stats_done1_cfg,
            name="tmemStatsDone1",
        )
        if stats_done1_cfg is not None
        else None
    )
    tmem_softmax_global0 = TmemSoftmaxGlobalResource(
        inst_id=0,
        cfg=cfg,
        scale_softmax_log2=scale_softmax_log2,
        sum_barrier_id=2,
        name="tmemSoftmaxGlobal0",
    )
    tmem_softmax_global1 = TmemSoftmaxGlobalResource(
        inst_id=1,
        cfg=cfg,
        scale_softmax_log2=scale_softmax_log2,
        sum_barrier_id=3,
        name="tmemSoftmaxGlobal1",
    )
    tmem_softmax_order = (
        TmemSoftmaxOrderResource(cfg=cfg, name="tmemSoftmaxOrder")
        if use_ordered_softmax_barrier
        else None
    )
    smem_p0.tmem_s_ref = tmem_s0
    smem_p1.tmem_s_ref = tmem_s1
    tmem_o.tmem_p0_ref = smem_p0
    tmem_o.tmem_p1_ref = smem_p1
    tmem_softmax_global0.p_ref = smem_p0
    tmem_softmax_global1.p_ref = smem_p1
    tmem_softmax_global0.tmem_s_ref = tmem_s0
    tmem_softmax_global1.tmem_s_ref = tmem_s1

    tmem_corr0 = TmemCorrResource(
        inst_id=0,
        cfg=cfg,
        scale_softmax_log2=scale_softmax_log2,
        output_scale=output_scale,
        o_ptr=o_ptr,
        partial_o_ptr=partial_o_ptr,
        partial_stats_ptr=partial_stats_ptr,
        split_kv_counter_ptr=split_kv_counter_ptr,
        attention_sinks_ptr=attention_sinks_ptr,
        seqlens_kv=kv_seqlens,
        max_seq_len_kv=corr_max_seq_len_kv,
        num_heads_kv=num_heads_kv,
        h_r=h_r,
        h_k_idx=h_k_idx,
        b_idx=b_idx,
        q_group_idx=q_group_idx,
        q_token_offset=q_token_offset,
        seq_len_q=seq_len_q,
        active_splits_kv=active_splits_kv,
        static_full_split_prefix=static_full_split_prefix,
        name="tmemCorr0",
    )
    tmem_corr1 = TmemCorrResource(
        inst_id=1,
        cfg=cfg,
        scale_softmax_log2=scale_softmax_log2,
        output_scale=output_scale,
        o_ptr=o_ptr,
        partial_o_ptr=partial_o_ptr,
        partial_stats_ptr=partial_stats_ptr,
        split_kv_counter_ptr=split_kv_counter_ptr,
        attention_sinks_ptr=attention_sinks_ptr,
        seqlens_kv=kv_seqlens,
        max_seq_len_kv=corr_max_seq_len_kv,
        num_heads_kv=num_heads_kv,
        h_r=h_r,
        h_k_idx=h_k_idx,
        b_idx=b_idx,
        q_group_idx=q_group_idx,
        q_token_offset=q_token_offset,
        seq_len_q=seq_len_q,
        active_splits_kv=active_splits_kv,
        static_full_split_prefix=static_full_split_prefix,
        name="tmemCorr1",
    )
    tmem_corr1.smem_p0_ref = smem_p0
    tmem_corr1.smem_p1_ref = smem_p1
    tmem_corr0.tmem_o_ref = tmem_o
    tmem_corr1.tmem_o_ref = tmem_o
    tmem_corr0.softmax_local0_ref = tmem_softmax_local0
    tmem_corr1.softmax_local0_ref = tmem_softmax_local0
    if use_one_inst_qkv:
        tmem_corr0.softmax_local1_ref = None
        tmem_corr1.softmax_local1_ref = None
    else:
        tmem_corr0.softmax_local1_ref = tmem_softmax_local1
        tmem_corr1.softmax_local1_ref = tmem_softmax_local1

    # ------------------------------------------------------------------
    # Domain computation
    # ------------------------------------------------------------------
    # HEAD handles the first 2 K tiles, LOOP advances the staggered
    # Qk/Pv steady-state, and TAIL drains the final V wave. For odd tile
    # counts, the final wave still requires one additional loop iteration so
    # that inst0 can process the last K/V tile.
    local_kv_tiles = _compute_local_kv_tiles(cfg, total_kv_tiles)
    loop_domain = _compute_decode_gen_loop_domain(local_kv_tiles, cfg.num_insts_kv)

    load_domain = loop_domain
    mma_domain = loop_domain
    softmax_domain = loop_domain + 1
    corr_domain = loop_domain

    # ------------------------------------------------------------------
    # Create tasks
    # ------------------------------------------------------------------
    task_runtime_kwargs = {
        "seqlens_kv": kv_seqlens,
        "max_seq_len_kv": max_seq_len_kv,
        "seq_len_q": seq_len_q,
    }
    if use_one_inst_qkv:
        load_task = create_load_task_one_inst_qkv(
            smem_q,
            smem_k0,
            smem_v0,
            work_queue,
            schedule_token_throttle,
            cfg,
            domain=load_domain,
            smem_page_offsets=smem_page_offsets,
            domain_bias=0,
            warp_idx=cfg.clc_load_warp_idx if use_clc_dynamic else None,
            **task_runtime_kwargs,
        )
    elif use_split_head_dim_kv:
        load_task = create_load_task_split_kv(
            smem_q,
            smem_k0,
            smem_k1,
            smem_v0,
            smem_v1,
            work_queue,
            schedule_token_throttle,
            cfg,
            domain=load_domain,
            smem_page_offsets=smem_page_offsets,
            smem_page_offsets_v=smem_page_offsets_v,
            domain_bias=0,
            warp_idx=cfg.clc_load_warp_idx if use_clc_dynamic else None,
            **task_runtime_kwargs,
        )
    else:
        load_task = create_load_task(
            smem_q,
            smem_kv,
            work_queue,
            schedule_token_throttle,
            cfg,
            domain=load_domain,
            domain_bias=0,
            warp_idx=cfg.clc_load_warp_idx if use_clc_dynamic else None,
            smem_page_offsets=smem_page_offsets,
            **task_runtime_kwargs,
        )
    page_offsets_task = None
    if use_paged_kv:
        page_offsets_warp_idx = (
            cfg.clc_padding_warp_idx if use_clc_dynamic else cfg.page_offsets_warp_idx
        )
        if split_head_dim_page_offsets_kv:
            page_offsets_task = create_page_offsets_task_split_kv(
                smem_page_offsets,
                smem_page_offsets_v,
                work_queue,
                cfg,
                domain=load_domain,
                domain_bias=0,
                warp_idx=page_offsets_warp_idx,
                num_warps=cfg.page_offsets_num_warps,
                paged_kv_indptr=(paged_kv_indptr if use_native_paged_kv else None),
                **task_runtime_kwargs,
            )
        else:
            page_offsets_task_fn = (
                create_page_offsets_task_one_inst_qkv
                if use_one_inst_qkv
                else create_page_offsets_task
            )
            page_offsets_task = page_offsets_task_fn(
                smem_page_offsets,
                work_queue,
                cfg,
                domain=load_domain,
                domain_bias=0,
                warp_idx=page_offsets_warp_idx,
                num_warps=cfg.page_offsets_num_warps,
                paged_kv_indptr=(paged_kv_indptr if use_native_paged_kv else None),
                **task_runtime_kwargs,
            )
    if use_one_inst_qkv:
        mma_task = create_mma_task_one_inst_qkv(
            smem_q,
            smem_k0,
            smem_v0,
            tmem_s0,
            smem_p0,
            tmem_o,
            work_queue,
            cfg,
            tmem_stats_done=tmem_stats_done0,
            domain=mma_domain,
            domain_bias=0,
            **task_runtime_kwargs,
        )
    elif use_split_head_dim_kv:
        mma_task = create_mma_task_split_kv(
            smem_q,
            smem_k0,
            smem_k1,
            smem_v0,
            smem_v1,
            tmem_s0,
            tmem_s1,
            smem_p0,
            smem_p1,
            tmem_o,
            work_queue,
            cfg,
            domain=mma_domain,
            tmem_stats_done0=tmem_stats_done0,
            tmem_stats_done1=tmem_stats_done1,
            domain_bias=0,
            **task_runtime_kwargs,
        )
    else:
        mma_task = create_mma_task(
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
            domain_bias=0,
            **task_runtime_kwargs,
        )
    softmax0_task = create_softmax0_task(
        tmem_s0,
        tmem_softmax_local0,
        smem_p0,
        tmem_softmax_global0,
        tmem_softmax_order,
        work_queue,
        cfg,
        domain=softmax_domain,
        domain_bias=1,
        **task_runtime_kwargs,
    )
    softmax1_task = None
    if not use_one_inst_qkv:
        softmax1_task = create_softmax1_task(
            tmem_s1,
            tmem_softmax_local1,
            smem_p1,
            tmem_softmax_global1,
            tmem_softmax_order,
            work_queue,
            cfg,
            domain=softmax_domain,
            domain_bias=1,
            **task_runtime_kwargs,
        )
    if use_one_inst_qkv:
        correction_task = create_correction_task_one_inst_qkv(
            tmem_softmax_local0,
            tmem_o,
            tmem_corr0,
            work_queue,
            cfg,
            tmem_stats_done=tmem_stats_done0,
            domain=corr_domain,
            domain_bias=0,
            **task_runtime_kwargs,
        )
    else:
        correction_task = create_correction_task(
            tmem_softmax_local0,
            tmem_softmax_local1,
            tmem_o,
            tmem_corr0,
            tmem_corr1,
            work_queue,
            cfg,
            domain=corr_domain,
            tmem_stats_done0=tmem_stats_done0,
            tmem_stats_done1=tmem_stats_done1,
            domain_bias=0,
            **task_runtime_kwargs,
        )
    padding_task = None
    if not (use_paged_kv and use_clc_dynamic):
        padding_task = create_padding_task(
            cfg,
            work_queue,
            warp_idx=(
                cfg.page_offsets_warp_idx + cfg.page_offsets_num_warps
                if use_paged_kv
                else (cfg.clc_padding_warp_idx if use_clc_dynamic else None)
            ),
            num_warps=(
                1
                if use_paged_kv
                else (cfg.clc_padding_num_warps if use_clc_dynamic else None)
            ),
        )
    scheduler_task = None
    if use_clc_dynamic:
        scheduler_task = create_scheduler_task(work_queue, schedule_token_throttle, cfg)
    clc_tail_padding_task = None
    if use_clc_dynamic and cfg.clc_tail_padding_num_warps > 0:
        clc_tail_padding_task = create_padding_task(
            cfg,
            work_queue,
            warp_idx=cfg.clc_tail_padding_warp_idx,
            num_warps=cfg.clc_tail_padding_num_warps,
        )

    task_list = []
    if page_offsets_task is not None:
        task_list.append(page_offsets_task)
    if use_one_inst_qkv and not use_clc_dynamic:
        task_list.extend([load_task, correction_task, mma_task])
        if padding_task is not None:
            task_list.append(padding_task)
        task_list.append(softmax0_task)
    else:
        task_list.extend([load_task, softmax0_task])
        if softmax1_task is not None:
            task_list.append(softmax1_task)
        task_list.extend([correction_task, mma_task])
        if padding_task is not None:
            task_list.append(padding_task)
    if clc_tail_padding_task is not None:
        task_list.append(clc_tail_padding_task)
    if scheduler_task is not None:
        task_list.append(scheduler_task)
    # ------------------------------------------------------------------
    # Resource dependency graph
    # ------------------------------------------------------------------
    smem_kv_deps = []
    if smem_page_offsets is not None:
        smem_kv_deps.append(smem_page_offsets)
    smem_k_deps = list(smem_kv_deps)
    smem_v_deps = (
        [smem_page_offsets_v] if smem_page_offsets_v is not None else list(smem_kv_deps)
    )
    if use_one_inst_qkv:
        resource_dependency_graph = {
            smem_q: [],
            smem_k0: smem_kv_deps,
            smem_v0: smem_kv_deps,
            tmem_s0: [smem_k0, smem_q],
            smem_p0: [tmem_s0],
            tmem_softmax_local0: [tmem_s0],
            tmem_softmax_global0: [tmem_s0],
            tmem_o: [smem_p0, smem_v0],
            tmem_corr0: [tmem_softmax_local0, tmem_o],
        }
    elif use_split_head_dim_kv:
        resource_dependency_graph = {
            smem_q: [],
            smem_k0: smem_k_deps,
            smem_k1: smem_k_deps,
            smem_v0: smem_v_deps,
            smem_v1: smem_v_deps,
            tmem_s0: [smem_k0, smem_q],
            tmem_s1: [smem_k1, smem_q],
            smem_p0: [tmem_s0],
            smem_p1: [tmem_s1],
            tmem_softmax_local0: [tmem_s0],
            tmem_softmax_local1: [tmem_s1],
            tmem_softmax_global0: [tmem_s0],
            tmem_softmax_global1: [tmem_s1],
            tmem_o: [smem_p0, smem_p1, smem_v0, smem_v1],
            tmem_corr0: [tmem_softmax_local0, tmem_o],
            tmem_corr1: [tmem_softmax_local0, tmem_softmax_local1, tmem_o],
        }
    else:
        resource_dependency_graph = {
            smem_q: [],
            smem_kv: smem_kv_deps,
            tmem_s0: [smem_kv, smem_q],
            tmem_s1: [smem_kv, smem_q],
            smem_p0: [tmem_s0],
            smem_p1: [tmem_s1],
            tmem_softmax_local0: [tmem_s0],
            tmem_softmax_local1: [tmem_s1],
            tmem_softmax_global0: [tmem_s0],
            tmem_softmax_global1: [tmem_s1],
            tmem_o: [smem_p0, smem_p1, smem_kv],
            tmem_corr0: [tmem_softmax_local0, tmem_o],
            tmem_corr1: [tmem_softmax_local0, tmem_softmax_local1, tmem_o],
        }
    if tmem_stats_done0 is not None:
        resource_dependency_graph[tmem_s0].append(tmem_stats_done0)
        resource_dependency_graph[tmem_stats_done0] = [tmem_softmax_local0]
        if tmem_stats_done1 is not None:
            resource_dependency_graph[tmem_s1].append(tmem_stats_done1)
            resource_dependency_graph[tmem_stats_done1] = [tmem_softmax_local1]
    if cutlass.const_expr(use_ordered_softmax_barrier):
        resource_dependency_graph[tmem_softmax_order] = [tmem_s0]
        resource_dependency_graph[smem_p1] = [
            *resource_dependency_graph[smem_p1],
            tmem_softmax_order,
        ]
    if smem_page_offsets is not None:
        resource_dependency_graph[smem_page_offsets] = []
    if smem_page_offsets_v is not None:
        resource_dependency_graph[smem_page_offsets_v] = []
    if work_queue is not None:
        for deps in resource_dependency_graph.values():
            deps.append(work_queue)
        resource_dependency_graph[work_queue] = (
            [work_queue, schedule_token_throttle]
            if schedule_token_throttle is not None
            else ([work_queue] if use_clc_dynamic else [])
        )
    if schedule_token_throttle is not None:
        resource_dependency_graph[schedule_token_throttle] = [work_queue]
    dma_consumer_release_labels = {}
    if smem_page_offsets is not None:
        if use_one_inst_qkv:
            dma_consumer_release_labels.update(
                {
                    (smem_page_offsets, smem_k0): {"read_offsets_k0"},
                    (smem_page_offsets, smem_v0): {"read_offsets_v0"},
                }
            )
        elif use_split_head_dim_kv:
            if smem_page_offsets_v is not None:
                dma_consumer_release_labels.update(
                    {
                        (smem_page_offsets, smem_k0): {"read_offsets_k0"},
                        (smem_page_offsets, smem_k1): {"read_offsets_k1"},
                        (smem_page_offsets_v, smem_v0): {"read_offsets_v0"},
                        (smem_page_offsets_v, smem_v1): {"read_offsets_v1"},
                    }
                )
            else:
                dma_consumer_release_labels.update(
                    {
                        (smem_page_offsets, smem_k0): {"read_offsets_k0"},
                        (smem_page_offsets, smem_k1): {"read_offsets_k1"},
                        (smem_page_offsets, smem_v0): {"read_offsets_v0"},
                        (smem_page_offsets, smem_v1): {"read_offsets_v1"},
                    }
                )
        else:
            dma_consumer_release_labels[(smem_page_offsets, smem_kv)] = {
                "cache_page_ids" if cfg.num_head_dim_stages_kv > 1 else "read_offsets"
            }
    if smem_kv is not None:
        dma_consumer_release_labels.update(
            {
                (smem_kv, tmem_s0): {"k_desc_0"},
                (smem_kv, tmem_s1): {"k_desc_1"},
                (smem_kv, tmem_o): {"v_desc_0", "v_desc_1"},
            }
        )

    # ------------------------------------------------------------------
    # SMEM / TMEM allocators
    # ------------------------------------------------------------------
    smem_allocator = SmemAllocator()
    if work_queue is not None:
        smem_allocator.add_resource(work_queue)
    if schedule_token_throttle is not None:
        smem_allocator.add_resource(schedule_token_throttle)
    smem_allocator.add_resource(smem_q)
    if smem_page_offsets is not None:
        smem_allocator.add_resource(smem_page_offsets)
    if smem_page_offsets_v is not None:
        smem_allocator.add_resource(smem_page_offsets_v)
    if use_one_inst_qkv:
        smem_allocator.add_resource(smem_k0)
        smem_allocator.add_resource(smem_v0)
    elif use_split_head_dim_kv:
        smem_allocator.add_resource(smem_k0)
        smem_allocator.add_resource(smem_k1)
        smem_allocator.add_resource(smem_v0)
        smem_allocator.add_resource(smem_v1)
    else:
        smem_allocator.add_resource(smem_kv)
    smem_allocator.add_resource(smem_p0)
    if not use_one_inst_qkv:
        smem_allocator.add_resource(smem_p1)
    smem_allocator.add_resource(tmem_s0)
    if not use_one_inst_qkv:
        smem_allocator.add_resource(tmem_s1)
    smem_allocator.add_resource(tmem_o)
    smem_allocator.add_resource(tmem_softmax_local0)
    if not use_one_inst_qkv:
        smem_allocator.add_resource(tmem_softmax_local1)
    smem_allocator.add_resource(tmem_softmax_global0)
    if not use_one_inst_qkv:
        smem_allocator.add_resource(tmem_softmax_global1)
    smem_allocator.add_resource(tmem_corr0)
    if not use_one_inst_qkv:
        smem_allocator.add_resource(tmem_corr1)
    smem_allocator.add_tmem_ptr(
        SmemAllocation("fmha_tmem_ptr_i32", dtype=cutlass.Int32, alignment=4)
    )
    smem_allocator.compute_layout()

    tmem_allocator = TmemAllocator()
    if cfg.use_keeps_mma_ab:
        if use_one_inst_qkv:
            tmem_allocator.add_resource(tmem_s0)
        elif cfg.keeps_separates_tmem_s_and_stats:
            tmem_allocator.add_alias_group(
                [
                    tmem_s0.get_tmem_requirements(),
                    smem_p0.get_tmem_requirements(),
                ]
            )
            tmem_allocator.add_alias_group(
                [
                    tmem_s1.get_tmem_requirements(),
                    smem_p1.get_tmem_requirements(),
                ]
            )
            tmem_allocator.add_resource(tmem_softmax_local0)
            tmem_allocator.add_resource(tmem_softmax_local1)
        else:
            tmem_allocator.add_alias_group(
                [
                    tmem_s0.get_tmem_requirements(),
                    tmem_softmax_local0.get_tmem_requirements()
                    + smem_p0.get_tmem_requirements(),
                ]
            )
            tmem_allocator.add_alias_group(
                [
                    tmem_s1.get_tmem_requirements(),
                    tmem_softmax_local1.get_tmem_requirements()
                    + smem_p1.get_tmem_requirements(),
                ]
            )
    else:
        tmem_allocator.add_resource(tmem_s0)
        tmem_allocator.add_resource(tmem_s1)
        tmem_allocator.add_resource(tmem_softmax_local0)
        tmem_allocator.add_resource(tmem_softmax_local1)
    tmem_allocator.add_resource(tmem_o)
    tmem_allocator.compute_layout()
    if cfg.uses_two_inst_tmem_p:
        # Two independent S regions become stats+P regions after Softmax
        # consumes each QK result.  O starts after both 128-column regions.
        s0_alloc = tmem_s0.get_tmem_requirements()[0]
        s1_alloc = tmem_s1.get_tmem_requirements()[0]
        stats0_alloc = tmem_softmax_local0.get_tmem_requirements()[0]
        stats1_alloc = tmem_softmax_local1.get_tmem_requirements()[0]
        p0_alloc = smem_p0.get_tmem_requirements()[0]
        p1_alloc = smem_p1.get_tmem_requirements()[0]
        o_alloc = tmem_o.get_tmem_requirements()[0]
        # Re-state the intended phase offsets after layout so every resource
        # observes the same S/P alias. Stats remain standalone when the whole
        # allocation fits; otherwise they share the S/P phase.
        s0_alloc.offset = 0
        s1_alloc.offset = cfg.tmem_s_cols
        p0_alloc.offset = (
            0 if cfg.keeps_separates_tmem_s_and_stats else cfg.tmem_stats_cols
        )
        p1_alloc.offset = cfg.tmem_s_cols + (
            0 if cfg.keeps_separates_tmem_s_and_stats else cfg.tmem_stats_cols
        )
        if cfg.keeps_separates_tmem_s_and_stats:
            stats0_alloc.offset = 2 * cfg.tmem_s_cols
            stats1_alloc.offset = 2 * cfg.tmem_s_cols + cfg.tmem_stats_cols
            o_alloc.offset = 2 * (cfg.tmem_s_cols + cfg.tmem_stats_cols)
        else:
            stats0_alloc.offset = 0
            stats1_alloc.offset = cfg.tmem_s_cols
            o_alloc.offset = 2 * cfg.tmem_s_cols
        expected_p_cols = cfg.tile_size_kv * cfg.q_dtype_bytes // 4
        assert p0_alloc.num_columns == p1_alloc.num_columns == expected_p_cols
        assert s0_alloc.offset <= p0_alloc.offset
        assert p0_alloc.offset + expected_p_cols <= s0_alloc.offset + cfg.tmem_s_cols
        assert s1_alloc.offset <= p1_alloc.offset
        assert p1_alloc.offset + expected_p_cols <= s1_alloc.offset + cfg.tmem_s_cols
        assert (
            o_alloc.offset + cfg.tmem_o_stage_cols * cfg.o_stages == cfg.tmem_total_cols
        )
    if use_one_inst_qkv:
        tmem_s_alloc = tmem_s0.get_tmem_requirements()[0]
        tmem_softmax_local0.get_tmem_requirements()[0].offset = tmem_s_alloc.offset
        assert cfg.tmem_stats_cols + cfg.tmem_p_cols <= cfg.tmem_s_cols
        smem_p0.get_tmem_requirements()[0].offset = (
            tmem_s_alloc.offset + cfg.tmem_stats_cols
        )

    return (
        task_list,
        resource_dependency_graph,
        dma_consumer_release_labels,
        smem_allocator,
        tmem_allocator,
        [tmem_corr0] if use_one_inst_qkv else [tmem_corr0, tmem_corr1],
    )


def _round_up_tmem_columns(num_columns: int) -> int:
    """tcgen05_alloc requires a power-of-two column count in [32, 512]."""
    return max(32, 1 << (num_columns - 1).bit_length())


def build_decode_task_manager(
    cfg: FmhaDecodeConfig,
    seq_len_kv: int = 2048,
    batch_size: int = 8,
    num_heads_kv: int = 8,
    verbose: bool = True,
    skip_validation: bool = False,
    exhaustive_deadlock_race_check: bool = True,
) -> TaskManager:
    """Build and validate the decode TS TaskManager (pure Python, no GPU).

    Parameters
    ----------
    cfg : FmhaDecodeConfig
        Pre-built configuration. Build it externally with
        ``make_decode_config`` so callers can apply their own
        overrides without monkey-patching the kernel module.
    seq_len_kv : int
        KV sequence length.

    Returns
    -------
    TaskManager
        Fully validated task manager (deadlock-free, bracketing OK, etc.).
    """
    effective_seq_len_kv = _configure_static_sliding_window(cfg, seq_len_kv)
    total_kv_tiles = _compute_total_kv_tiles(effective_seq_len_kv, cfg.tile_size_kv)
    cfg.total_kv_tiles = total_kv_tiles

    (
        task_list,
        resource_dependency_graph,
        dma_consumer_release_labels,
        smem_allocator,
        tmem_allocator,
        _correction_init_resources,
    ) = _build_decode_gen_schedule(
        cfg,
        total_kv_tiles,
        tile_sched_params=None,
    )

    tm = TaskManager(
        tasks=task_list,
        resource_dependency_graph=resource_dependency_graph,
        dma_consumer_release_labels=dma_consumer_release_labels,
        smem_allocator=smem_allocator,
        tmem_allocator=tmem_allocator,
        verbose=verbose,
        skip_validation=skip_validation,
        exhaustive_deadlock_race_check=exhaustive_deadlock_race_check,
    )

    return tm


# =====================================================================
# GPU Kernel
# =====================================================================


@cute.jit
def _run_decode_gen_active(
    tma_desc_q: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_k: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_v: cutlass.GridConstant[cuda.TensorMap],
    o_iter: cute.Pointer,
    g_s_k: Int32,
    g_h_k: Int32,
    g_scale_s_log2_e: Float32,
    g_output_scale: Float32,
    g_seqlens_kv: cute.Pointer,
    g_cu_seqlens_q: cute.Pointer,
    g_page_idx_kv: cute.Pointer,
    g_partial_o: cute.Pointer,
    g_partial_stats: cute.Pointer,
    g_split_kv_counter: cute.Pointer,
    g_attention_sinks: cute.Pointer,
    g_h_r: Int32,
    q_group_idx: Int32,
    h_k_idx: Int32,
    b_idx: Int32,
    q_token_offset: Int32,
    seq_len_q: Int32,
    active_splits_kv: Int32,
    static_full_split_prefix: cutlass.Constexpr[bool],
    tile_sched_params: utils.ClcDynamicPersistentTileSchedulerParams | None,
    cfg: cutlass.Constexpr[FmhaDecodeConfig],
    seq_len_kv: cutlass.Constexpr[int] = 2048,
    use_variable_seqlens_kv: cutlass.Constexpr[bool] = False,
    use_native_paged_kv: cutlass.Constexpr[bool] = False,
    use_static_native_seqlens_kv: cutlass.Constexpr[bool] = False,
    g_paged_kv_indptr: cute.Pointer | None = None,
    g_paged_kv_indices: cute.Pointer | None = None,
) -> None:
    """Run the complete decode body for one runtime-valid Q tile.

    Builds resources/tasks via `_build_decode_gen_schedule` (shared with the
    validation path), then owns the matched TMA prefetch, SMEM/TMEM setup,
    TaskManager execution, and TMEM teardown lifecycle. Keeping that lifecycle
    in a void JIT helper lets the kernel wrapper omit it entirely for an
    overlaunched packed-Q tile without threading TaskManager state through a
    dynamic branch.
    """

    WARP_SIZE = 32

    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    bias_static_sliding_kv_tma = (
        cfg.use_sliding_window_causal
        and cfg.max_seq_len_q == 1
        and not cfg.use_paged_kv
        and not cfg.use_split_kv
        and not use_variable_seqlens_kv
    )
    effective_seq_len_kv = _configure_static_sliding_window(
        cfg, seq_len_kv, bias_static_sliding_kv_tma
    )
    total_kv_tiles = _compute_total_kv_tiles(effective_seq_len_kv, cfg.tile_size_kv)
    cfg.total_kv_tiles = total_kv_tiles
    use_runtime_seqlens_kv = use_variable_seqlens_kv or (
        use_native_paged_kv and not use_static_native_seqlens_kv
    )
    runtime_seqlens_kv = (
        g_seqlens_kv if cutlass.const_expr(use_runtime_seqlens_kv) else None
    )
    runtime_max_seq_len_kv = (
        g_s_k
        if cutlass.const_expr(use_runtime_seqlens_kv)
        else Int32(cfg.static_seq_len_kv)
    )
    use_clc_dynamic_scheduler = cfg.use_persistent_scheduler

    # Prefetch TMA
    init_warp = 1
    if warp_idx == init_warp:
        prims.prefetch_tensormap(tma_desc_q.get_ptr())
        prims.prefetch_tensormap(tma_desc_k.get_ptr())
        prims.prefetch_tensormap(tma_desc_v.get_ptr())
    init_warp += 1

    clc_response_ptr = None
    if cutlass.const_expr(use_clc_dynamic_scheduler):
        clc_response_ptr = cute.arch.alloc_smem(
            cutlass.Int128, _PERSISTENT_SCHEDULE_TOKEN_STAGES
        )

    q_output_rows = g_h_r
    if cutlass.const_expr(cfg.max_seq_len_q > 1):
        q_output_rows = g_h_r * Int32(cfg.max_seq_len_q)

    (
        task_list,
        dep_graph,
        dma_consumer_release_labels,
        smem_allocator,
        tmem_allocator,
        correction_init_resources,
    ) = _build_decode_gen_schedule(
        cfg,
        total_kv_tiles,
        scale_softmax_log2=g_scale_s_log2_e,
        o_ptr=o_iter,
        output_scale=g_output_scale,
        partial_o_ptr=g_partial_o,
        partial_stats_ptr=g_partial_stats,
        split_kv_counter_ptr=g_split_kv_counter,
        attention_sinks_ptr=g_attention_sinks,
        seqlens_kv=runtime_seqlens_kv,
        cu_seqlens_q=g_cu_seqlens_q,
        max_seq_len_kv=runtime_max_seq_len_kv,
        corr_max_seq_len_kv=seq_len_kv,
        num_heads_kv=g_h_k,
        h_r=q_output_rows,
        tma_desc_q=tma_desc_q.get_ptr(),
        tma_desc_k=tma_desc_k.get_ptr(),
        tma_desc_v=tma_desc_v.get_ptr(),
        page_idx_kv=g_page_idx_kv,
        h_k_idx=h_k_idx,
        b_idx=b_idx,
        q_group_idx=q_group_idx,
        q_token_offset=q_token_offset,
        seq_len_q=seq_len_q,
        active_splits_kv=active_splits_kv,
        static_full_split_prefix=static_full_split_prefix,
        tile_sched_params=tile_sched_params,
        clc_response_ptr=clc_response_ptr,
        use_variable_seqlens_kv=use_variable_seqlens_kv,
        use_native_paged_kv=use_native_paged_kv,
        use_static_native_seqlens_kv=use_static_native_seqlens_kv,
        paged_kv_indptr=g_paged_kv_indptr,
        paged_kv_indices=g_paged_kv_indices,
    )

    smem_allocator.allocate()

    tmem_cols = tmem_allocator.total_tmem_columns
    tmem_alloc_cols = _round_up_tmem_columns(tmem_cols)
    tmem_ptr_alloc = smem_allocator.tmem_ptr_alloc
    assert tmem_ptr_alloc is not None
    tmem_ptr_i32 = smem_allocator.get(tmem_ptr_alloc)
    if warp_idx == init_warp:
        prims.tcgen05_alloc(tmem_ptr_i32, Int32(tmem_alloc_cols))
        prims.tcgen05_relinquish_alloc_permit()
    init_warp += 1

    task_manager = TaskManager(
        tasks=task_list,
        resource_dependency_graph=dep_graph,
        dma_consumer_release_labels=dma_consumer_release_labels,
        skip_validation=True,
        verbose=False,
        # TODO: Model stage-selected allocation aliases in the exhaustive
        # checker. D256 uses two physical S/stats/P stages, but the checker
        # currently sees their aggregate allocation and reports false races.
        exhaustive_deadlock_race_check=not (
            cfg.use_keeps_mma_ab and cfg.num_insts_kv == 1
        ),
        smem_allocator=smem_allocator,
        tmem_allocator=tmem_allocator,
    )

    task_manager.setup_resources_and_tasks()
    resource_context = ResourceContext(
        smem_base=smem_allocator.smem_base,
        tmem_ptr_i32=tmem_ptr_i32,
    )
    for resource in correction_init_resources:
        # Materialize correction-side function variables before the TS tasks
        # start. In cluster transaction-barrier mode this initializes each owner
        # CTA's mbarrier and expected byte count before peer CTAs can issue
        # async distributed-SMEM partial stores against it.
        resource.create_function_variables(resource_context)
    # Ensure every CTA thread observes initialized mbarriers and SMEM resource
    # bases before any task body can use them.
    prims.fence_mbarrier_init()
    prims.barrier_cta_sync(0)
    if cutlass.const_expr(cfg.supports_cluster_smem_reduction):
        # Cluster-wide visibility point for the per-CTA transaction mbarriers.
        # After this, peers may safely signal an owner CTA's mbarrier via mapa.
        prims.barrier_cluster_arrive_relaxed()
        prims.barrier_cluster_wait()
    task_manager.run()

    if cutlass.const_expr(cfg.use_parallel_separate_reduction_pdl):
        # Publish the dependent reducer only after this producer CTA has made
        # its normalized partial O and log2-LSE stores visible.
        thread_idx, _, _ = cute.arch.thread_idx()
        if thread_idx == Int32(0):
            prims.griddepcontrol(kind=prims.GridDepAction.LAUNCH_DEPENDENTS)

    # Every peer async-stores into an owner CTA's distributed SMEM and charges
    # the bytes to that owner's transaction mbarrier. Producer-only CTAs are
    # never remote-SMEM targets, so they may retire after their store issues;
    # each owner remains resident until its mbarrier completes and its local
    # reduction consumes all delivered partials. A final cluster rendezvous
    # would therefore only make completed producers wait for the owners.

    # ── TMEM cleanup ──
    dealloc_barrier_id = 13
    if cutlass.const_expr(cfg.use_keeps_mma_ab):
        # Keeps aliases score/stat TMEM columns across task warp groups. Wait
        # for every task's producer tail before the correction owner warp
        # releases the allocation; a correction-only barrier can race those
        # tails and leave tcgen05.dealloc waiting on live TMEM users.
        prims.barrier_cta_sync(dealloc_barrier_id)
        if (
            warp_idx >= cfg.correction_warp_idx
            and warp_idx < cfg.correction_warp_idx + cfg.correction_num_warps
        ):
            tidx, _, _ = cute.arch.thread_idx()
            correction_thread_idx = tidx - cfg.correction_warp_idx * WARP_SIZE
            if correction_thread_idx < WARP_SIZE:
                tmem_ptr = prims.make_tmem_ptr(tmem_ptr_i32.load(), cutlass.Int8)
                prims.tcgen05_dealloc(tmem_ptr, Int32(tmem_alloc_cols))
    else:
        if (
            warp_idx >= cfg.correction_warp_idx
            and warp_idx < cfg.correction_warp_idx + cfg.correction_num_warps
        ):
            prims.barrier_cta_sync(
                dealloc_barrier_id,
                thread_count=cfg.correction_num_warps * WARP_SIZE,
            )
            tidx, _, _ = cute.arch.thread_idx()
            correction_thread_idx = tidx - cfg.correction_warp_idx * WARP_SIZE
            if correction_thread_idx < WARP_SIZE:
                tmem_ptr = prims.make_tmem_ptr(tmem_ptr_i32.load(), cutlass.Int8)
                prims.tcgen05_dealloc(tmem_ptr, Int32(tmem_alloc_cols))


@cute.jit
def _run_decode_gen_inactive_cluster_rank() -> None:
    """Join cluster initialization, then retire an inactive physical split rank."""
    # The physical cluster remains configured-max sized. Initialize a local
    # zero-traffic barrier and join the same cluster visibility point as active
    # ranks. Active peers never address this rank after runtime contraction.
    inactive_mbarrier = cutlass.Array(
        cutlass.Int64,
        1,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    thread_idx, _, _ = cute.arch.thread_idx()
    if thread_idx == Int32(0):
        prims.mbarrier_init(inactive_mbarrier.data_ptr(), 1)
    prims.fence_mbarrier_init()
    prims.barrier_cta_sync(0)
    prims.barrier_cluster_arrive_relaxed()
    prims.barrier_cluster_wait()


@cute.jit
def _signal_padded_pdl_producer(cfg: cutlass.Constexpr[FmhaDecodeConfig]) -> None:
    """Release the dependent reducer launch from a zero-work producer CTA."""
    if cutlass.const_expr(cfg.use_parallel_separate_reduction_pdl):
        thread_idx, _, _ = cute.arch.thread_idx()
        if thread_idx == Int32(0):
            prims.griddepcontrol(kind=prims.GridDepAction.LAUNCH_DEPENDENTS)


@cute.jit
def _run_decode_gen_runtime_prefix(
    tma_desc_q: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_k: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_v: cutlass.GridConstant[cuda.TensorMap],
    o_iter: cute.Pointer,
    g_s_k: Int32,
    g_h_k: Int32,
    g_scale_s_log2_e: Float32,
    g_output_scale: Float32,
    g_seqlens_kv: cute.Pointer,
    g_cu_seqlens_q: cute.Pointer,
    g_page_idx_kv: cute.Pointer,
    g_partial_o: cute.Pointer,
    g_partial_stats: cute.Pointer,
    g_split_kv_counter: cute.Pointer,
    g_attention_sinks: cute.Pointer,
    g_h_r: Int32,
    q_group_cta_idx: Int32,
    q_group_idx: Int32,
    h_k_idx: Int32,
    b_idx: Int32,
    q_token_offset: Int32,
    seq_len_q: Int32,
    q_token_base: Int32,
    tile_sched_params: utils.ClcDynamicPersistentTileSchedulerParams | None,
    cfg: cutlass.Constexpr[FmhaDecodeConfig],
    seq_len_kv: cutlass.Constexpr[int],
    use_variable_seqlens_kv: cutlass.Constexpr[bool],
    use_native_paged_kv: cutlass.Constexpr[bool],
    use_static_native_seqlens_kv: cutlass.Constexpr[bool],
    g_paged_kv_indptr: cute.Pointer | None,
    g_paged_kv_indices: cute.Pointer | None,
) -> None:
    """Run the general runtime split-prefix producer or retire its suffix."""

    split_is_active = cutlass.Boolean(True)
    active_splits_kv = Int32(1)
    if cutlass.const_expr(cfg.use_split_kv):
        runtime_seq_len_kv = g_s_k
        if cutlass.const_expr(
            use_variable_seqlens_kv
            or (use_native_paged_kv and not use_static_native_seqlens_kv)
        ):
            runtime_seq_len_kv = Int32(g_seqlens_kv[b_idx])
        active_splits_kv = _runtime_active_splits_kv(
            cfg,
            runtime_seq_len_kv,
            seq_len_q,
            q_token_base,
        )
        if cutlass.const_expr(not cfg.use_separate_reduction_kernel):
            # Preserve one neutral producer for fused empty-K semantics.
            active_splits_kv = cute.math.max(active_splits_kv, Int32(1))
        split_idx = q_group_cta_idx % Int32(cfg.splits_kv)
        split_is_active = split_idx < active_splits_kv

    if cutlass.const_expr(cfg.supports_cluster_smem_reduction):
        if split_is_active:
            _run_decode_gen_active(
                tma_desc_q,
                tma_desc_k,
                tma_desc_v,
                o_iter,
                g_s_k,
                g_h_k,
                g_scale_s_log2_e,
                g_output_scale,
                g_seqlens_kv,
                g_cu_seqlens_q,
                g_page_idx_kv,
                g_partial_o,
                g_partial_stats,
                g_split_kv_counter,
                g_attention_sinks,
                g_h_r,
                q_group_idx,
                h_k_idx,
                b_idx,
                q_token_offset,
                seq_len_q,
                active_splits_kv,
                False,
                tile_sched_params,
                cfg,
                seq_len_kv,
                use_variable_seqlens_kv,
                use_native_paged_kv,
                use_static_native_seqlens_kv,
                g_paged_kv_indptr,
                g_paged_kv_indices,
            )
        else:
            _run_decode_gen_inactive_cluster_rank()
    else:
        if split_is_active:
            _run_decode_gen_active(
                tma_desc_q,
                tma_desc_k,
                tma_desc_v,
                o_iter,
                g_s_k,
                g_h_k,
                g_scale_s_log2_e,
                g_output_scale,
                g_seqlens_kv,
                g_cu_seqlens_q,
                g_page_idx_kv,
                g_partial_o,
                g_partial_stats,
                g_split_kv_counter,
                g_attention_sinks,
                g_h_r,
                q_group_idx,
                h_k_idx,
                b_idx,
                q_token_offset,
                seq_len_q,
                active_splits_kv,
                False,
                tile_sched_params,
                cfg,
                seq_len_kv,
                use_variable_seqlens_kv,
                use_native_paged_kv,
                use_static_native_seqlens_kv,
                g_paged_kv_indptr,
                g_paged_kv_indices,
            )
        else:
            _signal_padded_pdl_producer(cfg)


@cute.kernel
def decode_gen_kernel(
    tma_desc_q: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_k: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_v: cutlass.GridConstant[cuda.TensorMap],
    o_iter: cute.Pointer,
    g_s_k: Int32,
    g_h_k: Int32,
    g_scale_s_log2_e: Float32,
    g_output_scale: Float32,
    g_seqlens_kv: cute.Pointer,
    g_cu_seqlens_q: cute.Pointer,
    g_page_idx_kv: cute.Pointer,
    g_partial_o: cute.Pointer,
    g_partial_stats: cute.Pointer,
    g_split_kv_counter: cute.Pointer,
    g_attention_sinks: cute.Pointer,
    g_h_r: Int32,
    tile_sched_params: utils.ClcDynamicPersistentTileSchedulerParams | None,
    cfg: cutlass.Constexpr[FmhaDecodeConfig],
    seq_len_kv: cutlass.Constexpr[int] = 2048,
    use_variable_seqlens_kv: cutlass.Constexpr[bool] = False,
    use_native_paged_kv: cutlass.Constexpr[bool] = False,
    use_static_native_seqlens_kv: cutlass.Constexpr[bool] = False,
    g_paged_kv_indptr: cute.Pointer | None = None,
    g_paged_kv_indices: cute.Pointer | None = None,
    static_full_split_prefix: cutlass.Constexpr[bool] = False,
) -> None:
    """Dispatch one static Q/split tile and drain padded launch slots safely."""
    q_group_cta_idx, h_k_idx, b_idx = cute.arch.block_idx()
    q_group_idx = q_group_cta_idx
    if cutlass.const_expr(cfg.use_split_kv):
        # Grid coordinates and scratch linearization retain configured fanout.
        q_group_idx = q_group_cta_idx // Int32(cfg.splits_kv)

    q_token_offset = Int32(0)
    seq_len_q = Int32(cfg.max_seq_len_q)
    q_token_base = Int32(0)
    q_tile_is_active = cutlass.Boolean(True)
    if cutlass.const_expr(not cfg.use_persistent_scheduler):
        if cutlass.const_expr(cfg.use_variable_seqlens_q):
            q_token_offset, seq_len_q = _q_seq_bounds(cfg, g_cu_seqlens_q, b_idx)
        q_token_base = _q_group_token_base(cfg, q_group_idx)
        q_tile_is_active = q_token_base < seq_len_q

    # Persistent block coordinates identify physical workers rather than
    # logical tiles; their WorkQueue owns the equivalent Q predicate. Split-KV
    # and persistent scheduling are mutually exclusive in supported profiles.
    if q_tile_is_active:
        if cutlass.const_expr(cfg.use_split_kv and static_full_split_prefix):
            _run_decode_gen_active(
                tma_desc_q,
                tma_desc_k,
                tma_desc_v,
                o_iter,
                g_s_k,
                g_h_k,
                g_scale_s_log2_e,
                g_output_scale,
                g_seqlens_kv,
                g_cu_seqlens_q,
                g_page_idx_kv,
                g_partial_o,
                g_partial_stats,
                g_split_kv_counter,
                g_attention_sinks,
                g_h_r,
                q_group_idx,
                h_k_idx,
                b_idx,
                q_token_offset,
                seq_len_q,
                Int32(cfg.splits_kv),
                True,
                tile_sched_params,
                cfg,
                seq_len_kv,
                use_variable_seqlens_kv,
                use_native_paged_kv,
                use_static_native_seqlens_kv,
                g_paged_kv_indptr,
                g_paged_kv_indices,
            )
        else:
            _run_decode_gen_runtime_prefix(
                tma_desc_q,
                tma_desc_k,
                tma_desc_v,
                o_iter,
                g_s_k,
                g_h_k,
                g_scale_s_log2_e,
                g_output_scale,
                g_seqlens_kv,
                g_cu_seqlens_q,
                g_page_idx_kv,
                g_partial_o,
                g_partial_stats,
                g_split_kv_counter,
                g_attention_sinks,
                g_h_r,
                q_group_cta_idx,
                q_group_idx,
                h_k_idx,
                b_idx,
                q_token_offset,
                seq_len_q,
                q_token_base,
                tile_sched_params,
                cfg,
                seq_len_kv,
                use_variable_seqlens_kv,
                use_native_paged_kv,
                use_static_native_seqlens_kv,
                g_paged_kv_indptr,
                g_paged_kv_indices,
            )
    else:
        # Packed-Q grids use a batch-wide maximum envelope. These Q CTAs own no
        # producer state, unlike a valid Q tile whose K domain is empty.
        _signal_padded_pdl_producer(cfg)


@cute.jit
def fmha_decode_launch(
    problem_shape: tuple[Int32, Int32, Int32, Int32, Int32],
    q_iter: cute.Pointer,
    k_iter: cute.Pointer,
    v_iter: cute.Pointer,
    o_iter: cute.Pointer,
    seqlens_kv_iter: cute.Pointer,
    cu_seqlens_q_iter: cute.Pointer,
    total_q_tokens: Int32,
    page_idx_kv_iter: cute.Pointer,
    partial_o_iter: cute.Pointer,
    partial_stats_iter: cute.Pointer,
    split_kv_counter_iter: cute.Pointer,
    attention_sinks_iter: cute.Pointer,
    scale_s: Float32,
    output_scale: Float32,
    kv_b_stride: Int32,
    max_active_clusters: Int32,
    stream: cuda_drv.CUstream,
    cfg: cutlass.Constexpr[FmhaDecodeConfig],
    seq_len_kv: cutlass.Constexpr[int] = 2048,
    use_variable_seqlens_kv: cutlass.Constexpr[bool] = False,
    use_native_paged_kv: cutlass.Constexpr[bool] = False,
    paged_kv_indptr_iter: cute.Pointer | None = None,
    paged_kv_indices_iter: cute.Pointer | None = None,
    num_physical_kv_pages: Int64 = 0,
    k_page_stride: Int64 = 0,
    v_page_stride: Int64 = 0,
    static_full_split_prefix: cutlass.Constexpr[bool] = False,
    use_static_native_seqlens_kv: cutlass.Constexpr[bool] = False,
) -> None:
    """Standalone JIT launcher for FMHA decode TS."""
    log2_e = math.log2(math.e)
    b, h_q, h_k, s_k, d = problem_shape
    h_r = h_q // h_k
    bias_static_sliding_kv_tma = (
        cfg.use_sliding_window_causal
        and cfg.max_seq_len_q == 1
        and not cfg.use_paged_kv
        and not cfg.use_split_kv
        and not use_variable_seqlens_kv
    )
    effective_seq_len_kv = _configure_static_sliding_window(
        cfg, seq_len_kv, bias_static_sliding_kv_tma
    )

    q_seq = Int32(cfg.max_seq_len_q)
    if cutlass.const_expr(cfg.use_paged_kv):
        if cutlass.const_expr(use_native_paged_kv):
            kv_shape = (
                d,
                Int32(cfg.num_tokens_per_page),
                h_k,
                num_physical_kv_pages,
            )
            k_layout = cute.make_layout(
                kv_shape,
                stride=(
                    1,
                    d,
                    d * Int32(cfg.num_tokens_per_page),
                    k_page_stride,
                ),
            )
            v_layout = cute.make_layout(
                kv_shape,
                stride=(
                    1,
                    d,
                    d * Int32(cfg.num_tokens_per_page),
                    v_page_stride,
                ),
            )
            k_tma = cute.make_tensor(k_iter, k_layout)
            v_tma = cute.make_tensor(v_iter, v_layout)
        else:
            total_pages = b * Int32(cfg.max_num_pages_per_seq_kv)
            kv_layout = cute.make_layout(
                (d, Int32(cfg.num_tokens_per_page), h_k, total_pages),
                stride=(
                    1,
                    d,
                    d * Int32(cfg.num_tokens_per_page),
                    d * Int32(cfg.num_tokens_per_page) * h_k,
                ),
            )
            k_tma_iter = k_iter
            v_tma_iter = v_iter
            k_tma = cute.make_tensor(k_tma_iter, kv_layout)
            v_tma = cute.make_tensor(v_tma_iter, kv_layout)
    else:
        kv_s_for_tma = s_k
        k_tma_iter = k_iter
        v_tma_iter = v_iter
        if cutlass.const_expr(cfg.use_static_sliding_kv_tma_bias):
            skipped_tokens = Int32(
                _compute_static_num_skipped_kv_tiles(cfg, seq_len_kv) * cfg.tile_size_kv
            )
            skipped_elems = skipped_tokens * d
            k_tma_iter = k_iter + skipped_elems
            v_tma_iter = v_iter + skipped_elems
            kv_s_for_tma = Int32(effective_seq_len_kv)
        kv_layout = cute.make_layout(
            (d, kv_s_for_tma, h_k, b), stride=(1, d, d * s_k, kv_b_stride)
        )
        k_tma = cute.make_tensor(k_tma_iter, kv_layout)
        v_tma = cute.make_tensor(v_tma_iter, kv_layout)

    # Keep the TMA inner box at 128B when possible, but never exceed headDim.
    # box_dim is expressed in elements of the source dtype, not bytes.
    tma_box0_q = min(128 // cfg.q_dtype_bytes, cfg.headdim)
    tma_box0_kv = min(128 // cfg.kv_dtype_bytes, cfg.headdim)
    tma_swizzle = cuda.TensorMapSwizzle.s128b
    if cutlass.const_expr(cfg.use_fp8_qkv and cfg.headdim == 64):
        tma_swizzle = cuda.TensorMapSwizzle.s64b
    tma_kv_tokens = (
        cfg.num_tokens_per_page
        if cutlass.const_expr(cfg.use_paged_kv)
        else cfg.tile_size_kv
    )
    q_box_dims: tuple[object, ...]
    if cutlass.const_expr(cfg.use_variable_seqlens_q):
        # Packed Q is physically [sum_q_tokens, num_heads_q, head_dim]. Flatten
        # the two head modes into Hq so the ragged token axis still fits in a
        # rank-5 tensor map after the helper inserts its two synthetic modes.
        q_tma = cute.make_tensor(
            q_iter,
            cute.make_layout(
                (d, h_q, total_q_tokens),
                stride=(1, d, h_q * d),
            ),
        )
        if cutlass.const_expr(cfg.groups_tokens_heads_q):
            q_box_dims = (
                tma_box0_q,
                cfg.heads_q_per_kv,
                cfg.q_tokens_per_cta,
            )
            q_groups = Int32(
                (cfg.max_seq_len_q + cfg.q_tokens_per_cta - 1) // cfg.q_tokens_per_cta
            )
        else:
            q_box_dims = (tma_box0_q, cfg.tile_size_q, 1)
            head_ctas_per_token = Int32(
                (cfg.heads_q_per_kv + cfg.tile_size_q - 1) // cfg.tile_size_q
            )
            q_groups = Int32(cfg.max_seq_len_q) * head_ctas_per_token
        tma_desc_q = create_tensor_map_ragged_from_tensor(
            q_tma,
            box_dims=q_box_dims,
            ragged_dim=2,
            stride_order=(0, 1, 2),
            swizzle=tma_swizzle,
        )
    else:
        q_tma = cute.make_tensor(
            q_iter,
            cute.make_layout(
                (d, h_r, h_k, q_seq, b),
                stride=(1, d, h_r * d, h_q * d, q_seq * h_q * d),
            ),
        )
        if cutlass.const_expr(cfg.uses_nontrivial_grouped_q_layout):
            q_box_dims = (
                tma_box0_q,
                cfg.heads_q_per_kv,
                1,
                cfg.q_tokens_per_cta,
                1,
            )
            q_groups = Int32(
                (cfg.max_seq_len_q + cfg.q_tokens_per_cta - 1) // cfg.q_tokens_per_cta
            )
        else:
            q_box_dims = (tma_box0_q, cfg.tile_size_q, 1, 1, 1)
            q_groups = (
                (h_r + Int32(cfg.tile_size_q - 1)) // Int32(cfg.tile_size_q)
            ) * q_seq
        tma_desc_q = create_tensor_map_tiled_from_view(
            q_tma,
            box_dims=q_box_dims,
            stride_order=(0, 1, 2, 3, 4),
            swizzle=tma_swizzle,
        )
    tma_desc_k = create_tensor_map_tiled_from_view(
        k_tma,
        box_dims=(tma_box0_kv, tma_kv_tokens, 1, 1),
        stride_order=(0, 1, 2, 3),
        swizzle=tma_swizzle,
    )
    tma_desc_v = create_tensor_map_tiled_from_view(
        v_tma,
        box_dims=(tma_box0_kv, tma_kv_tokens, 1, 1),
        stride_order=(0, 1, 2, 3),
        swizzle=tma_swizzle,
    )

    use_clc_dynamic = cutlass.const_expr(cfg.use_persistent_scheduler)
    grid_x = q_groups
    if cutlass.const_expr(cfg.use_split_kv):
        grid_x = q_groups * Int32(cfg.splits_kv)

    if cutlass.const_expr(use_clc_dynamic):
        tile_sched_params = utils.ClcDynamicPersistentTileSchedulerParams(
            (grid_x, h_k, b),
            (1, 1, 1),
        )
        grid = tile_sched_params.get_grid_shape()
    else:
        tile_sched_params = None
        grid = (grid_x, h_k, b)
    # cluster distributed-SMEM reduction groups the splits_kv split CTAs
    # of each sequence (contiguous in grid-X) into one cluster so they can write
    # partials into each other's shared memory via prims.mapa.
    cluster_shape = [1, 1, 1]
    if cutlass.const_expr(cfg.use_cluster_smem_reduction):
        cluster_shape = [cfg.splits_kv, 1, 1]
    decode_gen_kernel(
        tma_desc_q,
        tma_desc_k,
        tma_desc_v,
        o_iter,
        s_k,
        h_k,
        Float32(scale_s * log2_e),
        output_scale,
        seqlens_kv_iter,
        cu_seqlens_q_iter,
        page_idx_kv_iter,
        partial_o_iter,
        partial_stats_iter,
        split_kv_counter_iter,
        attention_sinks_iter,
        h_r,
        tile_sched_params,
        cfg,
        seq_len_kv,
        use_variable_seqlens_kv,
        use_native_paged_kv,
        use_static_native_seqlens_kv,
        paged_kv_indptr_iter,
        paged_kv_indices_iter,
        static_full_split_prefix,
    ).launch(
        grid=grid,
        block=[cfg.threads_per_cta, 1, 1],
        cluster=cluster_shape,
        stream=stream,
        min_blocks_per_mp=(
            1
            if cfg.tile_size_q == 8
            or (cfg.tile_size_q == 16 and cfg.q_dtype_bytes == 1)
            or (cfg.use_keeps_mma_ab and cfg.tile_size_q == 128)
            else 0
        ),
        use_pdl=cfg.use_parallel_separate_reduction_pdl,
    )
