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

from ..._block_sparse.common import (
    _block_sparse_contiguous_kv_copy_geometry,
    _block_sparse_proxy_summary_geometry,
)
from ..._block_sparse.prepared import _BlockSparseRouteLayout
from ..tensor_map import (
    create_tensor_map_ragged_from_tensor,
    create_tensor_map_tiled,
    create_tensor_map_tiled_from_view,
)
from .fmha_decode_config import FmhaDecodeConfig
from .fmha_decode_constants import (
    KV_KIND_K,
    KV_KIND_V,
    KV_TILE_256_REGISTER_REALLOCATION_MIN_TILES,
)
from .fmha_decode_resources import (
    SmemBlockSparseKvMetadataResource,
    SmemBlockSparseSoftmaxMetadataResource,
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
    SmemKvReuseCreditResource,
    create_block_sparse_load_tasks_per_inst,
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


def _block_sparse_bshd_tma_strides(
    *,
    q_seq: cutlass.Integer | int,
    h_q: cutlass.Integer | int,
    h_k: cutlass.Integer | int,
    s_k: cutlass.Integer | int,
    d: cutlass.Integer | int,
) -> tuple[
    tuple[cutlass.Integer | int, ...],
    tuple[cutlass.Integer | int, ...],
]:
    """Build BSHD TensorMap strides in 16-byte units using Int64 math.

    The raw TensorMap API omits the implicit contiguous stride and takes the
    remaining strides in 16-byte units. Block-sparse attention supports only
    16-bit Q/K/V with headDim=128, so one stride unit contains eight elements.
    Keep every returned value in Int64: the outer batch stride can exceed the
    signed Int32 range even though each public tensor dimension is Int32.
    """

    elements_per_stride_unit = 8
    d_units = Int64(d // elements_per_stride_unit)
    h_r = h_q // h_k
    return (
        (
            d_units,
            Int64(h_r) * d_units,
            Int64(h_q) * d_units,
            Int64(q_seq) * Int64(h_q) * d_units,
        ),
        (
            Int64(h_k) * d_units,
            d_units,
            Int64(s_k) * Int64(h_k) * d_units,
        ),
    )


def _resolve_block_sparse_per_inst_load_topology(
    cfg: FmhaDecodeConfig,
    *,
    use_clc_dynamic: bool,
) -> tuple[tuple[int, int], tuple[int, int] | None] | None:
    """Reuse idle WG3 padding warps for two independent sparse load streams.

    The return value contains the two load warp indices and an optional
    residual padding task ``(warp_idx, num_warps)``. ``None`` identifies a
    noncanonical override for which the caller keeps the common load task.
    """

    if cfg.load_num_warps != 1:
        return None
    padding_warps = tuple(
        range(
            cfg.wg3_padding_warp_idx,
            cfg.wg3_padding_warp_idx + cfg.wg3_padding_num_warps,
        )
    )
    if use_clc_dynamic:
        # Load1 consumes the only otherwise-idle warp in WG3.
        if cfg.scheduler_num_warps != 1 or len(padding_warps) != 1:
            return None
        role_warps = (
            cfg.mma_warp_idx,
            cfg.scheduler_warp_idx,
            cfg.clc_load_warp_idx,
            padding_warps[0],
        )
        if len(set(role_warps)) != len(role_warps):
            return None
        if len({warp_idx // 4 for warp_idx in role_warps}) != 1:
            return None
        return (cfg.clc_load_warp_idx, padding_warps[0]), None

    if len(padding_warps) != 2:
        return None
    role_warps = (cfg.mma_warp_idx, cfg.load_warp_idx, *padding_warps)
    if len(set(role_warps)) != len(role_warps):
        return None
    if len({warp_idx // 4 for warp_idx in role_warps}) != 1:
        return None
    return (cfg.load_warp_idx, padding_warps[0]), (padding_warps[1], 1)


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


def _decode_min_blocks_per_mp(cfg: FmhaDecodeConfig, seq_len_kv: int) -> int:
    """Return the launch bound needed for dynamic register reallocation.

    ``setmaxnreg`` needs a kernel-entry occupancy bound before ptxas can infer
    the initial per-thread register allocation. KV256 only pays that fixed
    hand-off cost for a long enough mainloop; the established profiles below
    retain their existing unconditional launch bounds.
    """
    kv256_reallocation = (
        cfg.tile_size_kv == 256
        and _compute_total_kv_tiles(seq_len_kv, cfg.tile_size_kv)
        >= KV_TILE_256_REGISTER_REALLOCATION_MIN_TILES
    )
    return int(
        kv256_reallocation
        or cfg.tile_size_q == 8
        or (cfg.tile_size_q == 16 and cfg.q_dtype_bytes == 1)
        or (cfg.use_keeps_mma_ab and cfg.tile_size_q == 128)
    )


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
    tma_desc_k_atom: cutlass.Pointer | None = None,
    tma_desc_v_atom: cutlass.Pointer | None = None,
    tma_desc_k_summary: cutlass.Pointer | None = None,
    tma_desc_v_summary: cutlass.Pointer | None = None,
    tma_desc_k_summary_atom: cutlass.Pointer | None = None,
    tma_desc_v_summary_atom: cutlass.Pointer | None = None,
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
    block_tables: cute.Pointer | None = None,
    block_table_capacity: Int32 | None = None,
    block_table_row_stride: Int64 | None = None,
    sparse_row_route_offsets: cute.Pointer | None = None,
    sparse_row_route_counts: cute.Pointer | None = None,
    sparse_route_metadata: cute.Pointer | None = None,
) -> tuple[
    list[Task],
    dict[MemoryResource, list[MemoryResource]],
    dict[tuple[MemoryResource, MemoryResource], set[str]],
    SmemAllocator,
    TmemAllocator,
    list[MemoryResource],
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
        eager_init_resources)
    """
    if cfg.use_keeps_mma_ab and cfg.num_insts_kv == 1 and not cfg.uses_tmem_p:
        raise ValueError(
            "one-instance KeepsMmaAb is enabled only for the staged headDim=256 "
            "profile with head_dim_per_stage_kv=128 and o_stages=1"
        )
    if use_native_paged_kv and not cfg.use_paged_kv:
        raise ValueError("native paged-KV ABI requires cfg.use_paged_kv=True")
    if use_native_paged_kv and (
        block_tables is None
        or block_table_capacity is None
        or block_table_row_stride is None
    ):
        raise ValueError(
            "native paged-KV ABI requires block tables, capacity, and row stride"
        )
    if cfg.use_paged_kv:
        cfg.validate_paged_kv_staging_config()
    if cfg.use_block_sparse:
        if tma_desc_q is not None:
            if sparse_row_route_offsets is None:
                raise ValueError(
                    "sparse_row_route_offsets is required for block-sparse kernel "
                    "construction"
                )
            if sparse_row_route_counts is None:
                raise ValueError(
                    "sparse_row_route_counts is required for block-sparse kernel "
                    "construction"
                )
            if sparse_route_metadata is None:
                raise ValueError(
                    "sparse_route_metadata is required for block-sparse kernel "
                    "construction"
                )
            segment_tensormaps = {
                "tma_desc_k_atom": tma_desc_k_atom,
                "tma_desc_v_atom": tma_desc_v_atom,
            }
            if cfg.use_block_sparse_proxy_routes:
                segment_tensormaps.update(
                    {
                        "tma_desc_k_summary": tma_desc_k_summary,
                        "tma_desc_v_summary": tma_desc_v_summary,
                        "tma_desc_k_summary_atom": tma_desc_k_summary_atom,
                        "tma_desc_v_summary_atom": tma_desc_v_summary_atom,
                    }
                )
            for name, descriptor in segment_tensormaps.items():
                if descriptor is None:
                    raise ValueError(
                        f"{name} is required for block-sparse kernel construction"
                    )
            if num_heads_kv is None:
                raise ValueError(
                    "num_heads_kv is required for block-sparse kernel construction"
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
    use_dense_page_offsets = use_paged_kv and not cfg.use_block_sparse
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
    # instruction. Dense Swaps uses the shared FIFO, including staged H256.
    # Sparse KV128 keeps instruction-local rings in either MMA orientation;
    # sparse KV256 reuses its only feasible three-stage shared data ring while
    # retaining instruction-local route metadata. The load warp issues V(route
    # R) before replacing that metadata with route R+1, so its lifetime remains
    # independent of the K/V data-ring depth.
    # With cfg.keeps_stats_via_smem the stats-alias justification no longer
    # applies, but the shared FIFO still causes a material Q128 regression, so
    # the instruction-local FIFO gate remains part of that kernel policy.
    use_per_inst_kv_resources = (cfg.use_block_sparse and cfg.tile_size_kv != 256) or (
        cfg.use_keeps_mma_ab
        and cfg.tile_size_kv != 256
        and (not cfg.keeps_separates_tmem_s_and_stats or cfg.uses_two_inst_tmem_p)
    )
    # B8/B16 issue enough fine-grained TMA copies to benefit from reusing a
    # padding warp as a second issuer. The host policy applies one KV-side
    # crossover across all two-instance Swaps Q tiles.
    supports_per_inst_block_sparse_load_tasks = (
        cfg.use_block_sparse
        and cfg.use_parallel_sparse_kv_loads
        and not cfg.use_keeps_mma_ab
        and cfg.kv_block_size in (8, 16)
        and cfg.num_insts_kv == 2
    )
    use_separate_kv_page_offset_resources = (
        use_dense_page_offsets and use_per_inst_kv_resources and not use_one_inst_qkv
    )
    # Paired resources publish independent K0/K1 and V0/V1 stages. Shared
    # split-KV retains the aligned 32-ID representation for its optional
    # native held-window path.
    stage_page_ids_per_tile = _stages_page_ids_per_tile(
        use_separate_kv_page_offset_resources,
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
    if use_dense_page_offsets:
        page_offsets_stages = (
            3 if use_separate_kv_page_offset_resources else cfg.page_offsets_stages
        )
        smem_page_offsets_cfg = _make_page_offsets_cfg(page_offsets_stages)
        if use_separate_kv_page_offset_resources:
            smem_page_offsets_v_cfg = _make_page_offsets_cfg(page_offsets_stages)
    sparse_softmax_metadata0_cfg = None
    sparse_softmax_metadata1_cfg = None
    if cfg.use_block_sparse:
        # Two stages are sufficient for the split-ring cadence: one route can
        # await Softmax while Load publishes the next route for the same inst.
        sparse_softmax_metadata0_cfg = PipelineConfig(
            num_stages=2,
            num_bytes=0,
            producer_group=load_grp,
            consumer_group=softmax0_grp,
            pipeline_type=PipelineType.AsyncAsync,
            cta_layout_vmnk=cta_layout,
            advance_on_wait=True,
        )
        sparse_softmax_metadata1_cfg = PipelineConfig(
            num_stages=2,
            num_bytes=0,
            producer_group=load_grp,
            consumer_group=softmax1_grp,
            pipeline_type=PipelineType.AsyncAsync,
            cta_layout_vmnk=cta_layout,
            advance_on_wait=True,
        )
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
    smem_p0_cfg = None
    smem_p1_cfg = None
    if not cfg.streams_tmem_p_fragments:
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
            num_stages=one_inst_tmem_stages,
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
    resource_dependency_graph: dict[MemoryResource, list[MemoryResource]]
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
    smem_kv_reuse_credit = None
    # CLC remains the single persistent policy for every supported topology.
    # The stock static WorkQueue advances and decodes coordinates separately
    # in every task, which regresses multi-wave decode workloads. CLC computes
    # each schedule token once on the scheduler warp and broadcasts it to the workers.
    use_clc_dynamic = cfg.use_persistent_scheduler
    per_inst_block_sparse_load_topology = None
    if supports_per_inst_block_sparse_load_tasks:
        # Dual issuers are a performance choice. If a future recipe has no
        # compatible idle-warp placement, the common one-warp load task remains
        # correct and consumes the same disjoint K/V metadata pipelines.
        per_inst_block_sparse_load_topology = (
            _resolve_block_sparse_per_inst_load_topology(
                cfg,
                use_clc_dynamic=use_clc_dynamic,
            )
        )
    use_per_inst_block_sparse_load_tasks = (
        per_inst_block_sparse_load_topology is not None
    )
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
            "tile_scheduler_config": (
                TileSchedulerConfig.create_clc_dynamic_persistent_tile_scheduler_params(
                    tile_scheduler_params=tile_sched_params,
                    response_ptr=clc_response_ptr,
                )
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
        if cfg.uses_rotating_kv256_exchange:
            smem_kv_reuse_credit = SmemKvReuseCreditResource(
                cfg=cfg,
                pipeline_config=PipelineConfig.create_async_async_pipeline_cfg(
                    num_stages=1,
                    producer_group=load_grp,
                    consumer_group=correction_grp,
                    cta_layout_vmnk=cta_layout,
                ),
                name="smem_kv_reuse_credit",
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
    # Native callers provide one canonical sequence-length tensor alongside
    # their fixed page table, so native mode reuses the existing variable-length
    # domain, split, sliding-window, and masking paths.
    use_runtime_seqlens_kv = use_variable_seqlens_kv or (
        use_native_paged_kv and not use_static_native_seqlens_kv
    )
    kv_seqlens = seqlens_kv if use_runtime_seqlens_kv else None
    smem_page_offsets = None
    smem_page_offsets_v = None
    if use_dense_page_offsets:
        smem_page_offsets = SmemPageOffsetsKvResource(
            pipeline_config=smem_page_offsets_cfg,
            cfg=cfg,
            stage_page_ids_per_tile=stage_page_ids_per_tile,
            page_idx_kv=page_idx_kv,
            seqlens_kv=kv_seqlens,
            use_native_paged_kv=use_native_paged_kv,
            block_tables=block_tables,
            block_table_row_stride=block_table_row_stride,
            max_seq_len_kv=max_seq_len_kv,
            h_k_idx=h_k_idx,
            b_idx=b_idx,
            q_group_idx=q_group_idx,
            seq_len_q=seq_len_q,
            name=(
                "smemPageOffsetsKvK"
                if use_separate_kv_page_offset_resources
                else "smemPageOffsetsKv"
            ),
        )
        if use_separate_kv_page_offset_resources:
            smem_page_offsets_v = SmemPageOffsetsKvResource(
                pipeline_config=smem_page_offsets_v_cfg,
                cfg=cfg,
                stage_page_ids_per_tile=stage_page_ids_per_tile,
                page_idx_kv=page_idx_kv,
                seqlens_kv=kv_seqlens,
                use_native_paged_kv=use_native_paged_kv,
                block_tables=block_tables,
                block_table_row_stride=block_table_row_stride,
                max_seq_len_kv=max_seq_len_kv,
                h_k_idx=h_k_idx,
                b_idx=b_idx,
                q_group_idx=q_group_idx,
                seq_len_q=seq_len_q,
                name="smemPageOffsetsKvV",
            )
    sparse_kv_metadata0 = None
    sparse_kv_metadata1 = None
    sparse_softmax_metadata0 = None
    sparse_softmax_metadata1 = None
    if cfg.use_block_sparse:
        # This selects the prepared-record storage ABI. Causal consumers still
        # intersect these column-validity words with each Q row's causal mask.
        prepared_route_layout = _BlockSparseRouteLayout.create(
            kv_route_size=cfg.tile_size_kv,
            kv_block_size=cfg.kv_block_size,
            has_token_bits=cfg.uses_prepared_score_keep_words,
            route_metadata_capacity=0,
            num_rows=1,
            page_size=cfg.num_tokens_per_page if cfg.use_paged_kv else None,
        )
        sparse_kv_metadata0 = SmemBlockSparseKvMetadataResource(
            pipeline_config=None,
            cfg=cfg,
            inst_id=0,
            route_metadata=sparse_route_metadata,
            route_layout=prepared_route_layout,
            tma_oob_origin=max_seq_len_kv,
            name="smemBlockSparseKvMetadata0",
        )
        sparse_kv_metadata1 = SmemBlockSparseKvMetadataResource(
            pipeline_config=None,
            cfg=cfg,
            inst_id=1,
            route_metadata=sparse_route_metadata,
            route_layout=prepared_route_layout,
            tma_oob_origin=max_seq_len_kv,
            name="smemBlockSparseKvMetadata1",
        )
        sparse_softmax_metadata0 = SmemBlockSparseSoftmaxMetadataResource(
            pipeline_config=sparse_softmax_metadata0_cfg,
            cfg=cfg,
            inst_id=0,
            route_metadata=sparse_route_metadata,
            route_layout=prepared_route_layout,
            name="smemBlockSparseSoftmaxMetadata0",
        )
        sparse_softmax_metadata1 = SmemBlockSparseSoftmaxMetadataResource(
            pipeline_config=sparse_softmax_metadata1_cfg,
            cfg=cfg,
            inst_id=1,
            route_metadata=sparse_route_metadata,
            route_layout=prepared_route_layout,
            name="smemBlockSparseSoftmaxMetadata1",
        )
    smem_kv = None
    smem_k0 = None
    smem_k1 = None
    smem_v0 = None
    smem_v1 = None
    if use_per_inst_kv_resources:
        smem_k0 = SmemKvTileResource(
            pipeline_config=smem_k0_cfg,
            cfg=cfg,
            tma_desc_k=tma_desc_k,
            tma_desc_v=tma_desc_v,
            tma_desc_k_atom=tma_desc_k_atom,
            tma_desc_v_atom=tma_desc_v_atom,
            tma_desc_k_summary=tma_desc_k_summary,
            tma_desc_v_summary=tma_desc_v_summary,
            tma_desc_k_summary_atom=tma_desc_k_summary_atom,
            tma_desc_v_summary_atom=tma_desc_v_summary_atom,
            sparse_kv_metadata=sparse_kv_metadata0,
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
            tma_desc_k_atom=tma_desc_k_atom,
            tma_desc_v_atom=tma_desc_v_atom,
            tma_desc_k_summary=tma_desc_k_summary,
            tma_desc_v_summary=tma_desc_v_summary,
            tma_desc_k_summary_atom=tma_desc_k_summary_atom,
            tma_desc_v_summary_atom=tma_desc_v_summary_atom,
            sparse_kv_metadata=sparse_kv_metadata1,
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
            tma_desc_k_atom=tma_desc_k_atom,
            tma_desc_v_atom=tma_desc_v_atom,
            tma_desc_k_summary=tma_desc_k_summary,
            tma_desc_v_summary=tma_desc_v_summary,
            tma_desc_k_summary_atom=tma_desc_k_summary_atom,
            tma_desc_v_summary_atom=tma_desc_v_summary_atom,
            sparse_kv_metadata=sparse_kv_metadata0,
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
            tma_desc_k_atom=tma_desc_k_atom,
            tma_desc_v_atom=tma_desc_v_atom,
            tma_desc_k_summary=tma_desc_k_summary,
            tma_desc_v_summary=tma_desc_v_summary,
            tma_desc_k_summary_atom=tma_desc_k_summary_atom,
            tma_desc_v_summary_atom=tma_desc_v_summary_atom,
            sparse_kv_metadata=sparse_kv_metadata1,
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
            tma_desc_k_atom=tma_desc_k_atom,
            tma_desc_v_atom=tma_desc_v_atom,
            tma_desc_k_summary=tma_desc_k_summary,
            tma_desc_v_summary=tma_desc_v_summary,
            tma_desc_k_summary_atom=tma_desc_k_summary_atom,
            tma_desc_v_summary_atom=tma_desc_v_summary_atom,
            sparse_kv_metadata0=sparse_kv_metadata0,
            sparse_kv_metadata1=sparse_kv_metadata1,
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
    smem_p0.tmem_o_ref = tmem_o
    smem_p1.tmem_o_ref = tmem_o
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
        "sparse_row_route_offsets": sparse_row_route_offsets,
        "sparse_row_route_counts": sparse_row_route_counts,
        "num_heads_kv": num_heads_kv,
    }
    if use_one_inst_qkv:
        load_tasks = (
            create_load_task_one_inst_qkv(
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
            ),
        )
    elif use_per_inst_kv_resources:
        if use_per_inst_block_sparse_load_tasks:
            assert per_inst_block_sparse_load_topology is not None
            load_warp_indices, _ = per_inst_block_sparse_load_topology
            load_tasks = create_block_sparse_load_tasks_per_inst(
                smem_q,
                smem_k0,
                smem_k1,
                smem_v0,
                smem_v1,
                work_queue,
                schedule_token_throttle,
                cfg,
                domain=load_domain,
                sparse_kv_metadata0=sparse_kv_metadata0,
                sparse_kv_metadata1=sparse_kv_metadata1,
                sparse_softmax_metadata0=sparse_softmax_metadata0,
                sparse_softmax_metadata1=sparse_softmax_metadata1,
                domain_bias=0,
                warp_indices=load_warp_indices,
                **task_runtime_kwargs,
            )
        else:
            load_tasks = (
                create_load_task_split_kv(
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
                    sparse_kv_metadata0=sparse_kv_metadata0,
                    sparse_kv_metadata1=sparse_kv_metadata1,
                    sparse_softmax_metadata0=sparse_softmax_metadata0,
                    sparse_softmax_metadata1=sparse_softmax_metadata1,
                    domain_bias=0,
                    warp_idx=cfg.clc_load_warp_idx if use_clc_dynamic else None,
                    **task_runtime_kwargs,
                ),
            )
    else:
        load_tasks = (
            create_load_task(
                smem_q,
                smem_kv,
                work_queue,
                schedule_token_throttle,
                smem_kv_reuse_credit,
                cfg,
                domain=load_domain,
                domain_bias=0,
                warp_idx=cfg.clc_load_warp_idx if use_clc_dynamic else None,
                smem_page_offsets=smem_page_offsets,
                sparse_kv_metadata0=sparse_kv_metadata0,
                sparse_kv_metadata1=sparse_kv_metadata1,
                sparse_softmax_metadata0=sparse_softmax_metadata0,
                sparse_softmax_metadata1=sparse_softmax_metadata1,
                **task_runtime_kwargs,
            ),
        )
    page_offsets_task = None
    if use_dense_page_offsets:
        page_offsets_warp_idx = cfg.page_offsets_warp_idx
        if use_separate_kv_page_offset_resources:
            page_offsets_task = create_page_offsets_task_split_kv(
                smem_page_offsets,
                smem_page_offsets_v,
                work_queue,
                cfg,
                domain=load_domain,
                domain_bias=0,
                warp_idx=page_offsets_warp_idx,
                num_warps=cfg.page_offsets_num_warps,
                block_table_capacity=(
                    block_table_capacity if use_native_paged_kv else None
                ),
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
                block_table_capacity=(
                    block_table_capacity if use_native_paged_kv else None
                ),
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
    elif use_per_inst_kv_resources:
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
        sparse_softmax_metadata0,
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
            sparse_softmax_metadata1,
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
            smem_kv_reuse_credit,
            cfg,
            domain=corr_domain,
            tmem_stats_done0=tmem_stats_done0,
            tmem_stats_done1=tmem_stats_done1,
            domain_bias=0,
            **task_runtime_kwargs,
        )
    if use_per_inst_block_sparse_load_tasks:
        assert per_inst_block_sparse_load_topology is not None
        _, residual_padding = per_inst_block_sparse_load_topology
        padding_warp_ranges = (
            (residual_padding,) if residual_padding is not None else ()
        )
    else:
        padding_warp_ranges = (
            (cfg.wg0_padding_warp_idx, cfg.wg0_padding_num_warps),
            (cfg.wg1_padding_warp_idx, cfg.wg1_padding_num_warps),
            (cfg.wg2_padding_warp_idx, cfg.wg2_padding_num_warps),
            (cfg.wg3_padding_warp_idx, cfg.wg3_padding_num_warps),
        )
    padding_tasks = [
        create_padding_task(
            cfg,
            work_queue,
            warp_idx=warp_idx,
            num_warps=num_warps,
        )
        for warp_idx, num_warps in padding_warp_ranges
        if num_warps > 0
    ]
    scheduler_task = None
    if use_clc_dynamic:
        scheduler_task = create_scheduler_task(work_queue, schedule_token_throttle, cfg)

    task_list = []
    if page_offsets_task is not None:
        task_list.append(page_offsets_task)
    task_list.extend(load_tasks)
    if use_one_inst_qkv and not use_clc_dynamic:
        task_list.extend([correction_task, mma_task])
        task_list.append(softmax0_task)
    else:
        task_list.append(softmax0_task)
        if softmax1_task is not None:
            task_list.append(softmax1_task)
        task_list.extend([correction_task, mma_task])
    if scheduler_task is not None:
        task_list.append(scheduler_task)
    task_list.extend(padding_tasks)
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
    elif use_per_inst_kv_resources:
        resource_dependency_graph = {
            **(
                {
                    sparse_kv_metadata0: [],
                    sparse_kv_metadata1: [],
                }
                if sparse_kv_metadata0 is not None
                else {}
            ),
            **(
                {
                    sparse_softmax_metadata0: [sparse_kv_metadata0],
                    sparse_softmax_metadata1: [sparse_kv_metadata1],
                }
                if sparse_softmax_metadata0 is not None
                else {}
            ),
            smem_q: [],
            smem_k0: smem_k_deps
            + ([sparse_kv_metadata0] if sparse_kv_metadata0 is not None else []),
            smem_k1: smem_k_deps
            + ([sparse_kv_metadata1] if sparse_kv_metadata1 is not None else []),
            smem_v0: smem_v_deps
            + ([sparse_kv_metadata0] if sparse_kv_metadata0 is not None else []),
            smem_v1: smem_v_deps
            + ([sparse_kv_metadata1] if sparse_kv_metadata1 is not None else []),
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
            **(
                {
                    sparse_kv_metadata0: [],
                    sparse_kv_metadata1: [],
                    sparse_softmax_metadata0: [sparse_kv_metadata0],
                    sparse_softmax_metadata1: [sparse_kv_metadata1],
                }
                if sparse_kv_metadata0 is not None
                else {}
            ),
            smem_q: [],
            smem_kv: smem_kv_deps
            + (
                [sparse_kv_metadata0, sparse_kv_metadata1]
                if sparse_kv_metadata0 is not None
                else []
            ),
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
    if sparse_softmax_metadata0 is not None:
        resource_dependency_graph[smem_p0].append(sparse_softmax_metadata0)
        resource_dependency_graph[tmem_softmax_local0].append(sparse_softmax_metadata0)
        resource_dependency_graph[tmem_softmax_global0].append(sparse_softmax_metadata0)
        assert sparse_softmax_metadata1 is not None
        resource_dependency_graph[smem_p1].append(sparse_softmax_metadata1)
        resource_dependency_graph[tmem_softmax_local1].append(sparse_softmax_metadata1)
        resource_dependency_graph[tmem_softmax_global1].append(sparse_softmax_metadata1)
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
    if smem_kv_reuse_credit is not None:
        # A self-edge models the one-slot ownership token: Load produces it
        # for the current tile and Correction consumes it before the next Load.
        resource_dependency_graph[smem_kv_reuse_credit] = [smem_kv_reuse_credit]
    dma_consumer_release_labels: dict[
        tuple[MemoryResource, MemoryResource], set[str]
    ] = {}
    if smem_page_offsets is not None:
        if use_one_inst_qkv:
            dma_consumer_release_labels.update(
                {
                    (smem_page_offsets, smem_k0): {"read_offsets_k0"},
                    (smem_page_offsets, smem_v0): {"read_offsets_v0"},
                }
            )
        elif use_per_inst_kv_resources:
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
    if smem_kv_reuse_credit is not None:
        smem_allocator.add_resource(smem_kv_reuse_credit)
    smem_allocator.add_resource(smem_q)
    if smem_page_offsets is not None:
        smem_allocator.add_resource(smem_page_offsets)
    if smem_page_offsets_v is not None:
        smem_allocator.add_resource(smem_page_offsets_v)
    if sparse_kv_metadata0 is not None:
        smem_allocator.add_resource(sparse_kv_metadata0)
        smem_allocator.add_resource(sparse_kv_metadata1)
    if sparse_softmax_metadata0 is not None:
        smem_allocator.add_resource(sparse_softmax_metadata0)
        smem_allocator.add_resource(sparse_softmax_metadata1)
    if use_one_inst_qkv:
        smem_allocator.add_resource(smem_k0)
        smem_allocator.add_resource(smem_v0)
    elif use_per_inst_kv_resources:
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
    if cfg.tile_size_kv == 256:
        # KV256 direct-output correction rotates one compact 35,840-byte
        # payload through the shared 192-KiB K/V ring. Split-KV retains the
        # fixed full exchange. Neither path increases the CTA SMEM footprint.
        correction_exchange_requirements = tmem_corr1.get_smem_requirements()
        if correction_exchange_requirements:
            smem_allocator.add_alias_group(
                [
                    smem_kv.get_smem_requirements(),
                    correction_exchange_requirements,
                ]
            )
    smem_allocator.add_tmem_ptr(
        SmemAllocation("fmha_tmem_ptr_i32", dtype=cutlass.Int32, alignment=4)
    )
    smem_allocator.compute_layout()
    tmem_allocator = TmemAllocator()
    if cfg.use_keeps_mma_ab:
        if use_one_inst_qkv:
            tmem_allocator.add_resource(tmem_s0)
        else:
            # Build the two instruction-local phases from the resources that
            # actually use TMEM.  Depending on the profile, P can overlay S
            # or live in SMEM, and stats can be standalone TMEM or an SMEM
            # handoff.  Empty resources must not become scheduler aliases.
            for tmem_s, p in (
                (tmem_s0, smem_p0),
                (tmem_s1, smem_p1),
            ):
                p_requirements = p.get_tmem_requirements()
                if p_requirements:
                    tmem_allocator.add_alias_group(
                        [tmem_s.get_tmem_requirements(), p_requirements]
                    )
                else:
                    tmem_allocator.add_resource(tmem_s)
            # Register standalone stats only after both S/P phases, preserving
            # the established allocation order. SMEM-backed stats are a no-op.
            tmem_allocator.add_resource(tmem_softmax_local0)
            tmem_allocator.add_resource(tmem_softmax_local1)
    else:
        tmem_allocator.add_resource(tmem_s0)
        tmem_allocator.add_resource(tmem_s1)
        tmem_allocator.add_resource(tmem_softmax_local0)
        tmem_allocator.add_resource(tmem_softmax_local1)
    tmem_allocator.add_resource(tmem_o)
    tmem_allocator.compute_layout()
    if cfg.use_keeps_mma_ab and not use_one_inst_qkv and not cfg.uses_tmem_p:
        # Two-inst Keeps with SMEM P (currently Q64) must retain the historical
        # standalone-first S/O layout after unused stats aliases are removed.
        s0_alloc = tmem_s0.get_tmem_requirements()[0]
        s1_alloc = tmem_s1.get_tmem_requirements()[0]
        o_alloc = tmem_o.get_tmem_requirements()[0]
        stats0_requirements = tmem_softmax_local0.get_tmem_requirements()
        stats1_requirements = tmem_softmax_local1.get_tmem_requirements()
        if stats0_requirements:
            assert len(stats0_requirements) == len(stats1_requirements) == 1
            stats0_requirements[0].offset = 0
            stats1_requirements[0].offset = cfg.tmem_stats_cols
            o_alloc.offset = 2 * cfg.tmem_stats_cols
        else:
            assert not stats1_requirements
            o_alloc.offset = 0
        s0_alloc.offset = o_alloc.offset + o_alloc.num_columns
        s1_alloc.offset = s0_alloc.offset + cfg.tmem_s_cols
        assert s1_alloc.offset + cfg.tmem_s_cols == cfg.tmem_total_cols
    elif cfg.uses_two_inst_tmem_p:
        # P reuses each independent S region after Softmax consumes QK. Stats
        # are either standalone TMEM or an SMEM handoff.
        s0_alloc = tmem_s0.get_tmem_requirements()[0]
        s1_alloc = tmem_s1.get_tmem_requirements()[0]
        p0_alloc = smem_p0.get_tmem_requirements()[0]
        p1_alloc = smem_p1.get_tmem_requirements()[0]
        o_alloc = tmem_o.get_tmem_requirements()[0]
        if cfg.tile_size_kv == 256:
            # KV256 keeps O in the low 256 columns and overlays packed P on
            # each S region from its first column. Softmax streams K32
            # fragments in order, so every 16-column P store only overwrites
            # scores that have already been consumed. Starting P after the
            # nominal stats columns would instead clobber the next unread S
            # fragment; KV256 keeps its softmax stats in SMEM.
            o_alloc.offset = 0
            s0_alloc.offset = 2 * cfg.tmem_o_stage_cols
            s1_alloc.offset = s0_alloc.offset + cfg.tmem_s_cols
            p0_alloc.offset = s0_alloc.offset
            p1_alloc.offset = s1_alloc.offset
        else:
            # Re-state the intended phase offsets after layout so every
            # resource observes the same S/P alias. Stats remain standalone
            # when the whole allocation fits; otherwise they use SMEM. Keep
            # the historical gap before P so this modeling cleanup does not
            # change runtime addresses.
            s0_alloc.offset = 0
            s1_alloc.offset = cfg.tmem_s_cols
            p0_alloc.offset = (
                0 if cfg.keeps_separates_tmem_s_and_stats else cfg.tmem_stats_cols
            )
            p1_alloc.offset = cfg.tmem_s_cols + (
                0 if cfg.keeps_separates_tmem_s_and_stats else cfg.tmem_stats_cols
            )
            if cfg.keeps_separates_tmem_s_and_stats:
                stats0_alloc = tmem_softmax_local0.get_tmem_requirements()[0]
                stats1_alloc = tmem_softmax_local1.get_tmem_requirements()[0]
                stats0_alloc.offset = 2 * cfg.tmem_s_cols
                stats1_alloc.offset = 2 * cfg.tmem_s_cols + cfg.tmem_stats_cols
                o_alloc.offset = 2 * (cfg.tmem_s_cols + cfg.tmem_stats_cols)
            else:
                o_alloc.offset = 2 * cfg.tmem_s_cols
        expected_p_cols = cfg.tmem_p_cols_per_inst
        assert p0_alloc.num_columns == p1_alloc.num_columns == expected_p_cols
        assert s0_alloc.offset <= p0_alloc.offset
        assert p0_alloc.offset + expected_p_cols <= s0_alloc.offset + cfg.tmem_s_cols
        assert s1_alloc.offset <= p1_alloc.offset
        assert p1_alloc.offset + expected_p_cols <= s1_alloc.offset + cfg.tmem_s_cols
        assert (
            max(
                s0_alloc.offset + cfg.tmem_s_cols,
                s1_alloc.offset + cfg.tmem_s_cols,
                o_alloc.offset + cfg.tmem_o_stage_cols * cfg.o_stages,
            )
            == cfg.tmem_total_cols
        )
    if use_one_inst_qkv:
        tmem_s_alloc = tmem_s0.get_tmem_requirements()[0]
        # One-inst Keeps also transports stats through SMEM, so there is no
        # TMEM stats allocation to alias with S.
        assert cfg.keeps_stats_via_smem
        assert cfg.tmem_stats_cols + cfg.tmem_p_cols <= cfg.tmem_s_cols
        smem_p0.get_tmem_requirements()[0].offset = (
            tmem_s_alloc.offset + cfg.tmem_stats_cols
        )

    eager_init_resources = (
        [tmem_corr0] if use_one_inst_qkv else [tmem_corr0, tmem_corr1]
    )
    if smem_kv_reuse_credit is not None:
        # Initialize the persistent ring cursor under the same CTA-wide fence
        # and barrier used by other manually managed SMEM control state.
        eager_init_resources.append(smem_kv_reuse_credit)
    if cfg.streams_tmem_p_fragments:
        # KV256's TMEM P operands use one-way per-fragment ready barriers.
        # Initialize them beside correction's manually managed SMEM state.
        eager_init_resources.extend([smem_p0, smem_p1])

    return (
        task_list,
        resource_dependency_graph,
        dma_consumer_release_labels,
        smem_allocator,
        tmem_allocator,
        eager_init_resources,
    )


def _round_up_tmem_columns(num_columns: int) -> int:
    """tcgen05_alloc requires a power-of-two column count in [32, 512]."""
    return max(32, 1 << (num_columns - 1).bit_length())


def _has_unmodeled_tmem_p_alias_protocol(cfg: FmhaDecodeConfig) -> bool:
    """Whether exhaustive TS checking would report a known false P/S race.

    The staged D256 path selects one of two physical P/S stages at runtime.
    Static KV256 instead orders streamed P fragments with private mbarriers and
    reuses the matching TmemO-full barrier as the next-QK overwrite credit.
    Those intra-work protocols are below TaskManager's resource transitions,
    so its allocation-level checker cannot prove them. Persistent KV256 has
    enough task-level ordering for the checker and remains covered.
    """
    return cfg.uses_staged_one_inst_tmem_p or (
        cfg.streams_tmem_p_fragments and not cfg.use_persistent_scheduler
    )


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
    exhaustive_deadlock_race_check : bool
        Run exhaustive interleaving validation where TaskManager can model the
        complete synchronization protocol. Structural checks always run.

    Returns
    -------
    TaskManager
        Structurally validated task manager, exhaustively checked when the
        profile does not use a private TMEM-P alias protocol.
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
        _eager_init_resources,
    ) = _build_decode_gen_schedule(
        cfg,
        total_kv_tiles,
        tile_sched_params=None,
        num_heads_kv=Int32(num_heads_kv),
    )

    tm = TaskManager(
        tasks=task_list,
        resource_dependency_graph=resource_dependency_graph,
        dma_consumer_release_labels=dma_consumer_release_labels,
        smem_allocator=smem_allocator,
        tmem_allocator=tmem_allocator,
        verbose=verbose,
        skip_validation=skip_validation,
        exhaustive_deadlock_race_check=(
            exhaustive_deadlock_race_check
            and not _has_unmodeled_tmem_p_alias_protocol(cfg)
        ),
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
    tma_desc_k_atom: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_v_atom: cutlass.GridConstant[cuda.TensorMap],
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
    g_block_tables: cute.Pointer | None = None,
    g_block_table_capacity: Int32 | None = None,
    g_block_table_row_stride: Int64 | None = None,
    g_sparse_row_route_offsets: cute.Pointer | None = None,
    g_sparse_row_route_counts: cute.Pointer | None = None,
    g_sparse_route_metadata: cute.Pointer | None = None,
    tma_desc_k_summary: cutlass.GridConstant[cuda.TensorMap] | None = None,
    tma_desc_v_summary: cutlass.GridConstant[cuda.TensorMap] | None = None,
    tma_desc_k_summary_atom: cutlass.GridConstant[cuda.TensorMap] | None = None,
    tma_desc_v_summary_atom: cutlass.GridConstant[cuda.TensorMap] | None = None,
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
    tma_desc_k_summary_ptr = None
    tma_desc_v_summary_ptr = None
    tma_desc_k_summary_atom_ptr = None
    tma_desc_v_summary_atom_ptr = None
    if cutlass.const_expr(cfg.use_block_sparse_proxy_routes):
        assert tma_desc_k_summary is not None
        assert tma_desc_v_summary is not None
        assert tma_desc_k_summary_atom is not None
        assert tma_desc_v_summary_atom is not None
        tma_desc_k_summary_ptr = tma_desc_k_summary.get_ptr()
        tma_desc_v_summary_ptr = tma_desc_v_summary.get_ptr()
        tma_desc_k_summary_atom_ptr = tma_desc_k_summary_atom.get_ptr()
        tma_desc_v_summary_atom_ptr = tma_desc_v_summary_atom.get_ptr()

    # Prefetch TMA
    uses_atom_desc = False
    if cutlass.const_expr(cfg.use_block_sparse):
        _, _, uses_atom_desc = _block_sparse_contiguous_kv_copy_geometry(
            kv_block_size=cfg.kv_block_size,
            kv_route_size=cfg.tile_size_kv,
        )
    init_warp = 1
    if warp_idx == init_warp:
        prims.prefetch_tensormap(tma_desc_q.get_ptr())
        prims.prefetch_tensormap(tma_desc_k.get_ptr())
        prims.prefetch_tensormap(tma_desc_v.get_ptr())
        if cutlass.const_expr(cfg.use_block_sparse and uses_atom_desc):
            # KV256 and non-aligned coarse KV128 may select the exact atom maps.
            prims.prefetch_tensormap(tma_desc_k_atom.get_ptr())
            prims.prefetch_tensormap(tma_desc_v_atom.get_ptr())
        if cutlass.const_expr(cfg.use_block_sparse_proxy_routes):
            if cutlass.const_expr(cfg.tile_size_kv != 256):
                prims.prefetch_tensormap(tma_desc_k_summary_ptr)
                prims.prefetch_tensormap(tma_desc_v_summary_ptr)
            if cutlass.const_expr(uses_atom_desc):
                prims.prefetch_tensormap(tma_desc_k_summary_atom_ptr)
                prims.prefetch_tensormap(tma_desc_v_summary_atom_ptr)
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
        eager_init_resources,
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
        tma_desc_k_atom=tma_desc_k_atom.get_ptr(),
        tma_desc_v_atom=tma_desc_v_atom.get_ptr(),
        tma_desc_k_summary=tma_desc_k_summary_ptr,
        tma_desc_v_summary=tma_desc_v_summary_ptr,
        tma_desc_k_summary_atom=tma_desc_k_summary_atom_ptr,
        tma_desc_v_summary_atom=tma_desc_v_summary_atom_ptr,
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
        block_tables=g_block_tables,
        block_table_capacity=g_block_table_capacity,
        block_table_row_stride=g_block_table_row_stride,
        sparse_row_route_offsets=g_sparse_row_route_offsets,
        sparse_row_route_counts=g_sparse_row_route_counts,
        sparse_route_metadata=g_sparse_route_metadata,
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
        exhaustive_deadlock_race_check=not _has_unmodeled_tmem_p_alias_protocol(cfg),
        smem_allocator=smem_allocator,
        tmem_allocator=tmem_allocator,
    )

    task_manager.setup_resources_and_tasks()
    resource_context = ResourceContext(
        smem_base=smem_allocator.smem_base,
        tmem_ptr_i32=tmem_ptr_i32,
    )
    for resource in eager_init_resources:
        # Materialize manually managed resource state before TS tasks start.
        # This covers correction's cluster transaction barriers and KV256's
        # per-fragment P-ready barriers.
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
    tma_desc_k_atom: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_v_atom: cutlass.GridConstant[cuda.TensorMap],
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
    g_block_tables: cute.Pointer | None,
    g_block_table_capacity: Int32 | None,
    g_block_table_row_stride: Int64 | None,
    g_sparse_row_route_offsets: cute.Pointer | None,
    g_sparse_row_route_counts: cute.Pointer | None,
    g_sparse_route_metadata: cute.Pointer | None,
    tma_desc_k_summary: cutlass.GridConstant[cuda.TensorMap] | None = None,
    tma_desc_v_summary: cutlass.GridConstant[cuda.TensorMap] | None = None,
    tma_desc_k_summary_atom: cutlass.GridConstant[cuda.TensorMap] | None = None,
    tma_desc_v_summary_atom: cutlass.GridConstant[cuda.TensorMap] | None = None,
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
                tma_desc_k_atom,
                tma_desc_v_atom,
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
                g_block_tables,
                g_block_table_capacity,
                g_block_table_row_stride,
                g_sparse_row_route_offsets,
                g_sparse_row_route_counts,
                g_sparse_route_metadata,
                tma_desc_k_summary=tma_desc_k_summary,
                tma_desc_v_summary=tma_desc_v_summary,
                tma_desc_k_summary_atom=tma_desc_k_summary_atom,
                tma_desc_v_summary_atom=tma_desc_v_summary_atom,
            )
        else:
            _run_decode_gen_inactive_cluster_rank()
    else:
        if split_is_active:
            _run_decode_gen_active(
                tma_desc_q,
                tma_desc_k,
                tma_desc_v,
                tma_desc_k_atom,
                tma_desc_v_atom,
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
                g_block_tables,
                g_block_table_capacity,
                g_block_table_row_stride,
                g_sparse_row_route_offsets,
                g_sparse_row_route_counts,
                g_sparse_route_metadata,
                tma_desc_k_summary=tma_desc_k_summary,
                tma_desc_v_summary=tma_desc_v_summary,
                tma_desc_k_summary_atom=tma_desc_k_summary_atom,
                tma_desc_v_summary_atom=tma_desc_v_summary_atom,
            )
        else:
            _signal_padded_pdl_producer(cfg)


@cute.kernel
def decode_gen_kernel(
    tma_desc_q: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_k: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_v: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_k_atom: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_v_atom: cutlass.GridConstant[cuda.TensorMap],
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
    g_block_tables: cute.Pointer | None = None,
    g_block_table_capacity: Int32 | None = None,
    g_block_table_row_stride: Int64 | None = None,
    g_sparse_row_route_offsets: cute.Pointer | None = None,
    g_sparse_row_route_counts: cute.Pointer | None = None,
    g_sparse_route_metadata: cute.Pointer | None = None,
    static_full_split_prefix: cutlass.Constexpr[bool] = False,
    tma_desc_k_summary: cutlass.GridConstant[cuda.TensorMap] | None = None,
    tma_desc_v_summary: cutlass.GridConstant[cuda.TensorMap] | None = None,
    tma_desc_k_summary_atom: cutlass.GridConstant[cuda.TensorMap] | None = None,
    tma_desc_v_summary_atom: cutlass.GridConstant[cuda.TensorMap] | None = None,
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
                tma_desc_k_atom,
                tma_desc_v_atom,
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
                g_block_tables,
                g_block_table_capacity,
                g_block_table_row_stride,
                g_sparse_row_route_offsets,
                g_sparse_row_route_counts,
                g_sparse_route_metadata,
                tma_desc_k_summary=tma_desc_k_summary,
                tma_desc_v_summary=tma_desc_v_summary,
                tma_desc_k_summary_atom=tma_desc_k_summary_atom,
                tma_desc_v_summary_atom=tma_desc_v_summary_atom,
            )
        else:
            _run_decode_gen_runtime_prefix(
                tma_desc_q,
                tma_desc_k,
                tma_desc_v,
                tma_desc_k_atom,
                tma_desc_v_atom,
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
                g_block_tables,
                g_block_table_capacity,
                g_block_table_row_stride,
                g_sparse_row_route_offsets,
                g_sparse_row_route_counts,
                g_sparse_route_metadata,
                tma_desc_k_summary=tma_desc_k_summary,
                tma_desc_v_summary=tma_desc_v_summary,
                tma_desc_k_summary_atom=tma_desc_k_summary_atom,
                tma_desc_v_summary_atom=tma_desc_v_summary_atom,
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
    block_tables_iter: cute.Pointer | None = None,
    block_table_capacity: Int32 = 0,
    block_table_row_stride: Int64 = 0,
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
    if cutlass.const_expr(cfg.tile_size_kv == 256):
        # The 2x2 datapath consumes K in a (0, 2, 1, 3) KV64 permutation.
        # A KV64 TensorMap atom lets the shared load resource place each
        # semantic block directly in its physical slot for both paged and
        # contiguous layouts.
        tma_kv_tokens = min(
            cfg.num_tokens_per_page if cfg.use_paged_kv else cfg.tile_size_kv,
            64,
        )
    else:
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

    grid_x = q_groups
    if cutlass.const_expr(cfg.use_split_kv):
        grid_x = q_groups * Int32(cfg.splits_kv)

    if cutlass.const_expr(cfg.use_persistent_scheduler):
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
    null_sparse_route_ptr = cute.make_ptr(
        Int32,
        0,
        mem_space=cutlass.AddressSpace.gmem,
    )
    decode_gen_kernel(
        tma_desc_q,
        tma_desc_k,
        tma_desc_v,
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
        block_tables_iter,
        block_table_capacity,
        block_table_row_stride,
        null_sparse_route_ptr,
        null_sparse_route_ptr,
        null_sparse_route_ptr,
        static_full_split_prefix,
    ).launch(
        grid=grid,
        block=[cfg.threads_per_cta, 1, 1],
        cluster=cluster_shape,
        stream=stream,
        min_blocks_per_mp=_decode_min_blocks_per_mp(cfg, effective_seq_len_kv),
        use_pdl=cfg.use_parallel_separate_reduction_pdl,
    )


@cute.jit
def fmha_block_sparse_launch(
    problem_shape: tuple[Int32, Int32, Int32, Int32, Int32],
    q_iter: cute.Pointer,
    k_iter: cute.Pointer,
    v_iter: cute.Pointer,
    k_summary_iter: cute.Pointer,
    v_summary_iter: cute.Pointer,
    o_iter: cute.Pointer,
    row_route_offsets_iter: cute.Pointer,
    row_route_counts_iter: cute.Pointer,
    route_metadata_iter: cute.Pointer,
    scale_s: Float32,
    stream: cuda_drv.CUstream,
    cfg: cutlass.Constexpr[FmhaDecodeConfig],
    seq_len_kv: cutlass.Constexpr[int],
    g_seqlens_kv: cute.Pointer | None = None,
    use_variable_seqlens_kv: cutlass.Constexpr[bool] = False,
    num_physical_kv_pages: Int64 = 0,
    k_page_stride: Int64 = 0,
    v_page_stride: Int64 = 0,
) -> None:
    """Launch attention over exact and typed exact/proxy prepared KV routes.

    A preceding prepare kernel has already resolved each BSR row into compact
    logical atom origins, storage locators, validity flags, and optional token
    words. Exact routes address K/V; proxy routes address summary K/V. Both
    layouts execute the same ``decode_gen_kernel`` schedule and
    physical copy policy. Exact builds constexpr-elide summary TensorMaps.
    """
    if cutlass.const_expr(not cfg.use_block_sparse):
        raise ValueError("fmha_block_sparse_launch requires block-sparse config")
    if cutlass.const_expr(cfg.use_block_sparse_proxy_routes and cfg.use_paged_kv):
        raise ValueError("block-sparse proxy routes require contiguous K/V")

    log2_e = math.log2(math.e)
    b, h_q, h_k, s_k, d = problem_shape
    h_r = h_q // h_k
    q_seq = Int32(cfg.max_seq_len_q)
    q_strides, kv_strides = _block_sparse_bshd_tma_strides(
        q_seq=q_seq,
        h_q=h_q,
        h_k=h_k,
        s_k=s_k,
        d=d,
    )

    # FP16/BF16 H128 uses a 64-element (128-byte) inner box.  The sparse
    # profile validator rejects other element widths and head dimensions.
    tma_box0 = min(128 // cfg.kv_dtype_bytes, cfg.headdim)
    tma_swizzle = cuda.TensorMapSwizzle.s128b
    # Public tensors are contiguous BSHD. Q is factored into (Hr, Hkv) so the
    # unchanged grouped-Q resource can address one KV head's grouped-Q tile.
    q_desc = create_tensor_map_tiled(
        global_address=q_iter.toint(),
        dtype=cfg.q_dtype,
        global_dims=(d, h_r, h_k, q_seq, b),
        global_strides=q_strides,
        box_dims=(
            tma_box0,
            cfg.heads_q_per_kv,
            1,
            cfg.q_tokens_per_cta,
            1,
        ),
        swizzle=tma_swizzle,
    )

    (
        primary_kv_box_size,
        kv_atom_size,
        uses_atom_desc,
    ) = _block_sparse_contiguous_kv_copy_geometry(
        kv_block_size=cfg.kv_block_size,
        kv_route_size=cfg.tile_size_kv,
    )
    k_desc_summary_primary = None
    v_desc_summary_primary = None
    k_desc_summary_atom = None
    v_desc_summary_atom = None
    if cutlass.const_expr(cfg.use_paged_kv):
        # Paged HND storage is addressed as (D, token-in-page, Hkv, page).
        # Prepared routes already contain each atom's physical page ID, so no
        # dense page table is passed to or staged by the attention kernel.
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
        k_desc_atom = create_tensor_map_tiled_from_view(
            cute.make_tensor(k_iter, k_layout),
            box_dims=(tma_box0, kv_atom_size, 1, 1),
            stride_order=(0, 1, 2, 3),
            swizzle=tma_swizzle,
        )
        v_desc_atom = create_tensor_map_tiled_from_view(
            cute.make_tensor(v_iter, v_layout),
            box_dims=(tma_box0, kv_atom_size, 1, 1),
            stride_order=(0, 1, 2, 3),
            swizzle=tma_swizzle,
        )
        k_desc_primary = k_desc_atom
        v_desc_primary = v_desc_atom
    else:
        # Exact and summary tensors form one logical segmented KV coordinate
        # space. Each physical source owns the same primary/atom descriptor
        # pair; the prepared route kind selects the pair, while the loader
        # retains the existing KV128/fine/KV256 copy policy.
        kv_dims = (d, s_k, h_k, b)
        k_desc_primary = create_tensor_map_tiled(
            global_address=k_iter.toint(),
            dtype=cfg.kv_dtype,
            global_dims=kv_dims,
            global_strides=kv_strides,
            box_dims=(tma_box0, primary_kv_box_size, 1, 1),
            swizzle=tma_swizzle,
        )
        v_desc_primary = create_tensor_map_tiled(
            global_address=v_iter.toint(),
            dtype=cfg.kv_dtype,
            global_dims=kv_dims,
            global_strides=kv_strides,
            box_dims=(tma_box0, primary_kv_box_size, 1, 1),
            swizzle=tma_swizzle,
        )
        k_desc_atom = k_desc_primary
        v_desc_atom = v_desc_primary
        if cutlass.const_expr(uses_atom_desc):
            # KV256 always stages four semantic KV64 atoms. KV128 needs this
            # map only when a route may join unrelated BSR entries.
            k_desc_atom = create_tensor_map_tiled(
                global_address=k_iter.toint(),
                dtype=cfg.kv_dtype,
                global_dims=kv_dims,
                global_strides=kv_strides,
                box_dims=(tma_box0, kv_atom_size, 1, 1),
                swizzle=tma_swizzle,
            )
            v_desc_atom = create_tensor_map_tiled(
                global_address=v_iter.toint(),
                dtype=cfg.kv_dtype,
                global_dims=kv_dims,
                global_strides=kv_strides,
                box_dims=(tma_box0, kv_atom_size, 1, 1),
                swizzle=tma_swizzle,
            )

        if cutlass.const_expr(cfg.use_block_sparse_proxy_routes):
            num_kv_blocks, _ = _block_sparse_proxy_summary_geometry(
                seq_len_kv,
                cfg.kv_block_size,
            )
            _, summary_kv_strides = _block_sparse_bshd_tma_strides(
                q_seq=q_seq,
                h_q=h_q,
                h_k=h_k,
                s_k=num_kv_blocks,
                d=d,
            )
            summary_dims = (d, num_kv_blocks, h_k, b)
            k_desc_summary_primary = create_tensor_map_tiled(
                global_address=k_summary_iter.toint(),
                dtype=cfg.kv_dtype,
                global_dims=summary_dims,
                global_strides=summary_kv_strides,
                box_dims=(tma_box0, primary_kv_box_size, 1, 1),
                swizzle=tma_swizzle,
            )
            v_desc_summary_primary = create_tensor_map_tiled(
                global_address=v_summary_iter.toint(),
                dtype=cfg.kv_dtype,
                global_dims=summary_dims,
                global_strides=summary_kv_strides,
                box_dims=(tma_box0, primary_kv_box_size, 1, 1),
                swizzle=tma_swizzle,
            )
            k_desc_summary_atom = k_desc_summary_primary
            v_desc_summary_atom = v_desc_summary_primary
            if cutlass.const_expr(uses_atom_desc):
                k_desc_summary_atom = create_tensor_map_tiled(
                    global_address=k_summary_iter.toint(),
                    dtype=cfg.kv_dtype,
                    global_dims=summary_dims,
                    global_strides=summary_kv_strides,
                    box_dims=(tma_box0, kv_atom_size, 1, 1),
                    swizzle=tma_swizzle,
                )
                v_desc_summary_atom = create_tensor_map_tiled(
                    global_address=v_summary_iter.toint(),
                    dtype=cfg.kv_dtype,
                    global_dims=summary_dims,
                    global_strides=summary_kv_strides,
                    box_dims=(tma_box0, kv_atom_size, 1, 1),
                    swizzle=tma_swizzle,
                )

    q_groups = Int32(
        (cfg.max_seq_len_q + cfg.q_tokens_per_cta - 1) // cfg.q_tokens_per_cta
    )
    if cutlass.const_expr(cfg.use_persistent_scheduler):
        tile_sched_params = utils.ClcDynamicPersistentTileSchedulerParams(
            (q_groups, h_k, b),
            (1, 1, 1),
        )
        grid = tile_sched_params.get_grid_shape()
    else:
        tile_sched_params = None
        grid = (q_groups, h_k, b)

    null_i32_ptr = cute.make_ptr(
        Int32,
        0,
        mem_space=cutlass.AddressSpace.gmem,
    )
    null_f32_ptr = cute.make_ptr(
        Float32,
        0,
        mem_space=cutlass.AddressSpace.gmem,
    )
    seqlens_kv_iter = (
        g_seqlens_kv if cutlass.const_expr(use_variable_seqlens_kv) else null_i32_ptr
    )
    decode_gen_kernel(
        q_desc,
        k_desc_primary,
        v_desc_primary,
        k_desc_atom,
        v_desc_atom,
        o_iter,
        s_k,
        h_k,
        Float32(scale_s * log2_e),
        Float32(1.0),
        seqlens_kv_iter,
        null_i32_ptr,
        null_i32_ptr,
        o_iter,
        null_f32_ptr,
        null_i32_ptr,
        null_f32_ptr,
        h_r,
        tile_sched_params,
        cfg,
        seq_len_kv,
        use_variable_seqlens_kv,
        False,  # use_native_paged_kv
        False,  # use_static_native_seqlens_kv
        null_i32_ptr,  # g_block_tables
        Int32(0),  # g_block_table_capacity
        Int64(0),  # g_block_table_row_stride
        row_route_offsets_iter,
        row_route_counts_iter,
        route_metadata_iter,
        False,  # static_full_split_prefix
        tma_desc_k_summary=k_desc_summary_primary,
        tma_desc_v_summary=v_desc_summary_primary,
        tma_desc_k_summary_atom=k_desc_summary_atom,
        tma_desc_v_summary_atom=v_desc_summary_atom,
    ).launch(
        grid=grid,
        block=[cfg.threads_per_cta, 1, 1],
        cluster=[1, 1, 1],
        stream=stream,
        # Reuse dense decode's entry-occupancy contract, including the long
        # KV256 profiles that execute dynamic register reallocation.
        min_blocks_per_mp=_decode_min_blocks_per_mp(cfg, seq_len_kv),
        use_pdl=cfg.use_parallel_separate_reduction_pdl,
    )
