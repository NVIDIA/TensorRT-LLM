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
from .direct_sparse_metadata import DirectSparseMetadataView, HeadIndexedMetadataView
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cuda.bindings import driver as cuda_drv
from cutlass import Float32, Int32, Int64
from cutlass.experimental import primitives as prims
from cutlass.experimental.task_scheduling.enums import PipelineType, SignalingThreads
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
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T as mlir_T

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
    BYTES_PER_KIB,
    KV_KIND_K,
    KV_KIND_V,
    KV_TILE_256_REGISTER_REALLOCATION_MIN_TILES,
    MAX_KV_STAGE_SMEM_KIB,
    SMEM_CAPACITY_KIB,
)
from .fmha_decode_resources import (
    SmemBlockSparseKvMetadataResource,
    SmemBlockSparseSoftmaxMetadataResource,
    SmemKvTileResource,
    SmemKvResource,
    SmemPageOffsetsKvResource,
    SmemQResource,
    SmemTransformedKvResource,
    TmemCorrResource,
    TmemOResource,
    SmemPResource,
    TmemSResource,
    TmemStatsDoneResource,
    TmemSoftmaxGlobalResource,
    TmemSoftmaxLocalResource,
    TmemSoftmaxOrderResource,
    TmemTransformedKvResource,
)
from .fmha_decode_resources.helpers_common import (
    _q_group_token_base,
    _q_seq_bounds,
)
from .fmha_decode_resources.helpers_kv_tile_idx import _runtime_active_splits_kv
from .fmha_decode_resources.sage_scales import (
    SAGE_K_SCALES_RING_STAGES,
    SageKScalesResource,
    SageScaleTensors,
    SageVScalesResource,
)
from .fmha_decode_tasks import (
    PackedDecodeWorkQueue,
    ScheduleTokenThrottleResource,
    SparseMembershipLifetimeResource,
    _can_hold_native_page_window,
    _prefetch_prepared_sparse_row,
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
    create_transform_kv_task,
)
from .reduction import (  # noqa: F401
    decode_gen_separate_reduction_kernel,
    fmha_decode_separate_reduction_launch,
)

_PERSISTENT_SCHEDULE_TOKEN_STAGES = 2
# Response slots of the CLC work-queue fetch pipeline. Every role consumes
# the queue one tile behind the scheduler warp, so a second slot adds no
# lookahead, only a loop-carried stage index that spills to local memory.
_WORK_QUEUE_STAGES = 1


def _block_sparse_bshd_tma_strides(
    *,
    q_seq: cutlass.Integer | int,
    h_q: cutlass.Integer | int,
    h_k: cutlass.Integer | int,
    s_k: cutlass.Integer | int,
    d: cutlass.Integer | int,
    element_bytes: int,
) -> tuple[
    tuple[cutlass.Integer | int, ...],
    tuple[cutlass.Integer | int, ...],
]:
    """Build BSHD TensorMap strides in 16-byte units using Int64 math.

    The raw TensorMap API omits the implicit contiguous stride and takes the
    remaining strides in 16-byte units, so one unit holds ``16 /
    element_bytes`` elements of the 16-bit or 8-bit Q/K/V (headDim=128). Keep
    every returned value in Int64: the outer batch stride can exceed the
    signed Int32 range even though each public tensor dimension is Int32.
    """

    elements_per_stride_unit = 16 // element_bytes
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
        cfg.use_transform_kv
        or kv256_reallocation
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
    tma_desc_k_sf: cutlass.Pointer | None = None,
    tma_desc_v_sf: cutlass.Pointer | None = None,
    tma_desc_k_atom: cutlass.Pointer | None = None,
    tma_desc_v_atom: cutlass.Pointer | None = None,
    tma_desc_k_summary: cutlass.Pointer | None = None,
    tma_desc_v_summary: cutlass.Pointer | None = None,
    tma_desc_k_summary_atom: cutlass.Pointer | None = None,
    tma_desc_v_summary_atom: cutlass.Pointer | None = None,
    page_idx_kv: cute.Pointer | None = None,
    page_table_stride: Int64 | None = None,
    page_table_capacity: Int32 | None = None,
    q_token_kv_block_sparse_page_memberships: cute.Pointer | None = None,
    q_token_kv_block_sparse_page_membership_stride: Int32 | None = None,
    h_k_idx: Int32 | None = None,
    b_idx: Int32 | None = None,
    q_group_idx: Int32 | None = None,
    q_token_offset: Int32 | None = None,
    seq_len_q: Int32 | None = None,
    active_splits_kv: Int32 | None = None,
    static_full_split_prefix: bool = False,
    tile_sched_params: utils.ClcDynamicPersistentTileSchedulerParams | None = None,
    use_variable_seqlens_kv: bool = False,
    use_native_paged_kv: bool = False,
    use_static_native_seqlens_kv: bool = False,
    sparse_row_route_offsets: cute.Pointer | None = None,
    sparse_row_route_counts: cute.Pointer | None = None,
    sparse_route_metadata: cute.Pointer | None = None,
    sparse_row_route_begin: Int32 | None = None,
    sparse_route_count: Int32 | None = None,
    sage_q_scale_ptr: cute.Pointer | None = None,
    sage_k_scale_ptr: cute.Pointer | None = None,
    sage_k_summary_scale_ptr: cute.Pointer | None = None,
    sage_v_scale_ptr: cute.Pointer | None = None,
    sage_v_mean_ptr: cute.Pointer | None = None,
    sage_q_scale_head_stride: Int32 | None = None,
    sage_k_scale_head_stride: Int32 | None = None,
    sage_k_summary_scale_head_stride: Int32 | None = None,
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
    if cfg.use_block_sparse and cfg.num_insts_kv != 2:
        raise ValueError("block-sparse attention requires num_insts_kv == 2")
    if cfg.use_keeps_mma_ab and cfg.num_insts_kv == 1 and not cfg.uses_tmem_p:
        raise ValueError(
            "one-instance KeepsMmaAb is enabled only for the staged headDim=256 "
            "profile with head_dim_per_stage_kv=128 and o_stages=1"
        )
    if use_native_paged_kv and not cfg.use_paged_kv:
        raise ValueError("native paged-KV ABI requires cfg.use_paged_kv=True")
    if use_native_paged_kv and (
        page_idx_kv is None or page_table_capacity is None or page_table_stride is None
    ):
        raise ValueError(
            "native paged-KV ABI requires block tables, capacity, and row stride"
        )
    if cfg.uses_scattered_page_route and not use_native_paged_kv:
        raise ValueError("scattered page routes require the native paged-KV ABI")
    if (
        cfg.uses_q_token_kv_block_sparse_page_membership
        and tma_desc_q is not None
        and q_token_kv_block_sparse_page_memberships is None
    ):
        raise ValueError(
            "grouped QToken-KvBlock-Sparse-Attention kernel construction requires q_token_kv_block_sparse_page_memberships"
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
    if cfg.use_transform_kv:
        transform_kv_grp = pipeline.CooperativeGroup(
            Agent.Thread, cfg.transform_kv_num_warps * WARP_SIZE
        )
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
    tma_producer_signaling = (
        SignalingThreads.TaskWarpLeader
        if cfg.load_num_warps > 1
        else SignalingThreads.All
    )
    q_bytes_per_load_warp = (
        cfg.smem_q_tile_bytes // cfg.load_num_warps if cfg.load_num_warps > 1 else None
    )

    # ------------------------------------------------------------------
    # Pipeline configs
    # ------------------------------------------------------------------
    # Leave barrier_ptr unset so SmemAllocator packs every pipeline barrier
    # into the unified block. Separate barrier arrays create an alignment gap
    # before the 1024-byte-aligned data block and overflow near-capacity Q128.
    use_paged_kv = cfg.use_paged_kv
    use_one_inst_kv = cfg.num_insts_kv == 1
    use_dense_page_offsets = use_paged_kv and not cfg.use_block_sparse
    use_one_inst_qkv = cfg.use_keeps_mma_ab and cfg.num_insts_kv == 1
    # K and V of different byte widths take the split-resource paths with one
    # K ring and one V ring, each shared by both K/V instances; equal widths
    # (including Int8 K with E4M3 V) share one ring.
    use_shared_inst_kv_rings = (
        cfg.k_dtype_bytes != cfg.v_dtype_bytes and cfg.tile_size_kv != 256
    )
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
    if use_shared_inst_kv_rings and not use_one_inst_qkv:
        # One K and one V stage per unit of depth.
        shared_inst_kv_stages = max(
            (MAX_KV_STAGE_SMEM_KIB * BYTES_PER_KIB)
            // (cfg.smem_k_tile_bytes + cfg.smem_v_tile_bytes),
            1,
        )
        split_k0_stages = shared_inst_kv_stages
        split_v0_stages = shared_inst_kv_stages
    use_ordered_softmax_barrier = (
        not use_one_inst_kv and cfg.uses_ordered_softmax_barrier
    )
    # A two-inst Keeps profile can use the deeper shared K/V FIFO when stats
    # are standalone and P remains in SMEM.  Keep instruction-local FIFOs when
    # stats or TMEM-P alias S: their overwrite-credit cadence is tied to each
    # instruction. Dense Swaps uses the shared FIFO, including staged H256.
    # Sparse KV128 keeps instruction-local rings in either MMA orientation;
    # sparse KV256 reuses the shared data ring at its element-width-derived
    # depth while retaining instruction-local route metadata. The load warp
    # issues V(route R) before replacing that metadata with route R+1, so its
    # lifetime remains independent of the K/V data-ring depth.
    # With cfg.keeps_stats_via_smem the stats-alias justification no longer
    # applies, but the shared FIFO still causes a material Q128 regression, so
    # the instruction-local FIFO gate remains part of that kernel policy.
    use_per_inst_kv_resources = (
        use_one_inst_qkv
        or (cfg.use_block_sparse and cfg.tile_size_kv != 256)
        or use_shared_inst_kv_rings
        or (
            cfg.use_keeps_mma_ab
            and cfg.tile_size_kv != 256
            and (not cfg.keeps_separates_tmem_s_and_stats or cfg.uses_two_inst_tmem_p)
        )
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
    # Independent K0/K1/V0/V1 data rings share identical native sparse locators.
    # A held route is published once and retained through the final V issue;
    # only schedules that replace locators per tile need separate K/V state.
    use_separate_kv_page_offset_resources = (
        use_dense_page_offsets
        and use_per_inst_kv_resources
        and not use_one_inst_qkv
        and not (
            use_native_paged_kv
            and cfg.uses_q_token_kv_block_sparse_page_route
            and cfg.uses_held_encoded_locator_window
        )
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
        producer_signaling_threads=tma_producer_signaling,
        num_bytes_per_warp_per_cta=q_bytes_per_load_warp,
        advance_on_wait=True,
    )

    def _make_raw_kv_cfg(num_stages: int, tile_bytes: int) -> PipelineConfig:
        """Create a per-operand raw pipeline with its actual TMA byte count."""
        num_bytes = tile_bytes + cfg.smem_kv_sf_tile_bytes
        bytes_per_load_warp = (
            num_bytes // cfg.load_num_warps if cfg.load_num_warps > 1 else None
        )
        if cfg.use_transform_kv:
            # NVFP4 payload and scale factors complete the same raw barrier.
            return PipelineConfig(
                num_stages=num_stages,
                num_bytes=num_bytes,
                producer_group=tma_producer,
                consumer_group=transform_kv_grp,
                pipeline_type=PipelineType.TmaAsync,
                cta_layout_vmnk=cta_layout,
                producer_signaling_threads=tma_producer_signaling,
                num_bytes_per_warp_per_cta=bytes_per_load_warp,
                advance_on_wait=True,
            )
        return PipelineConfig.create_tma_umma_pipeline_cfg(
            num_stages=num_stages,
            num_bytes=num_bytes,
            producer_group=tma_producer,
            consumer_group=umma_hw,
            cta_layout_vmnk=cta_layout,
            producer_signaling_threads=tma_producer_signaling,
            num_bytes_per_warp_per_cta=bytes_per_load_warp,
            advance_on_wait=True,
        )

    smem_kv_cfg = None
    if not use_per_inst_kv_resources:
        smem_kv_cfg = _make_raw_kv_cfg(cfg.kv_stages, cfg.smem_kv_tile_bytes)
    smem_transformed_kv_cfg = None
    if cfg.use_transform_kv:
        smem_transformed_kv_cfg = PipelineConfig.create_async_umma_pipeline_cfg(
            num_stages=cfg.transformed_kv_stages,
            producer_group=transform_kv_grp,
            consumer_group=umma_hw,
            cta_layout_vmnk=cta_layout,
            advance_on_wait=True,
        )
    smem_k0_cfg = _make_raw_kv_cfg(split_k0_stages, cfg.smem_k_tile_bytes)
    smem_k1_cfg = _make_raw_kv_cfg(split_k1_stages, cfg.smem_k_tile_bytes)
    smem_v0_cfg = _make_raw_kv_cfg(split_v0_stages, cfg.smem_v_tile_bytes)
    smem_v1_cfg = _make_raw_kv_cfg(split_v1_stages, cfg.smem_v_tile_bytes)

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
        if use_native_paged_kv and cfg.uses_held_encoded_locator_window:
            # QToken-KvBlock-Sparse-Attention publishes its complete CTA-local locator span once per work
            # tile and holds it through the K/V tail. Only one stage is live
            # for direct, persistent, and split-KV routes alike.
            page_offsets_stages = 1
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
        num_consumer_threads = cfg.threads_per_cta
        wq_pipeline_config = PipelineConfig.create_clc_fetch_async_pipeline_cfg(
            num_stages=_WORK_QUEUE_STAGES,
            num_bytes=16,
            producer_group=pipeline.CooperativeGroup(Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                Agent.Thread, num_consumer_threads
            ),
            cta_layout_vmnk=cta_layout,
        )
        # The scheduler config needs the CLC response slots' address; it is
        # bound once the unified SMEM layout below is computed.
        work_queue_kwargs = {
            "tile_scheduler_config": None,
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
    membership_lifetime = None
    if use_clc_dynamic and cfg.uses_q_token_kv_block_sparse_page_membership:
        # Page IDs are consumed by Load, but membership bytes are consumed by
        # Softmax. Do not overwrite the next work item's membership row until
        # every score stream has finished reading the current one.
        membership_lifetime = SparseMembershipLifetimeResource(
            name="sparseMembershipLifetime",
            pipeline_config=PipelineConfig(
                num_stages=1,
                num_bytes=0,
                producer_group=page_offsets_grp,
                consumer_group=pipeline.CooperativeGroup(
                    Agent.Thread,
                    WARP_SIZE
                    * (
                        cfg.softmax0_num_warps
                        + (0 if use_one_inst_kv else cfg.softmax1_num_warps)
                    ),
                ),
                pipeline_type=PipelineType.AsyncAsync,
                cta_layout_vmnk=cta_layout,
                advance_on_wait=True,
            ),
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
    # Native dense page tables pair fixed-capacity rows with one canonical
    # sequence-length tensor, so native mode reuses the existing
    # variable-length domain, split, sliding-window, and masking paths.
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
            page_table_stride=page_table_stride,
            num_heads_kv=num_heads_kv,
            q_token_kv_block_sparse_page_memberships=q_token_kv_block_sparse_page_memberships,
            q_token_kv_block_sparse_page_membership_stride=q_token_kv_block_sparse_page_membership_stride,
            seqlens_kv=kv_seqlens,
            use_native_paged_kv=use_native_paged_kv,
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
                # Softmax consumes only the K-side membership view.
                cache_memberships_in_smem=False,
                page_idx_kv=page_idx_kv,
                page_table_stride=page_table_stride,
                num_heads_kv=num_heads_kv,
                q_token_kv_block_sparse_page_memberships=q_token_kv_block_sparse_page_memberships,
                q_token_kv_block_sparse_page_membership_stride=q_token_kv_block_sparse_page_membership_stride,
                seqlens_kv=kv_seqlens,
                use_native_paged_kv=use_native_paged_kv,
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
    sage_scale_tensors = None
    if cfg.use_sage_attention:
        sage_scale_tensors = SageScaleTensors(
            q_scale_ptr=sage_q_scale_ptr,
            q_scale_head_stride=sage_q_scale_head_stride,
            k_scale_ptr=sage_k_scale_ptr,
            k_scale_head_stride=sage_k_scale_head_stride,
            k_summary_scale_ptr=sage_k_summary_scale_ptr,
            k_summary_scale_head_stride=sage_k_summary_scale_head_stride,
            v_scale_ptr=sage_v_scale_ptr,
            v_mean_ptr=sage_v_mean_ptr,
        )
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
            h_k_idx=h_k_idx,
            b_idx=b_idx,
            scale_tensors=sage_scale_tensors,
        )
        sparse_softmax_metadata1 = SmemBlockSparseSoftmaxMetadataResource(
            pipeline_config=sparse_softmax_metadata1_cfg,
            cfg=cfg,
            inst_id=1,
            route_metadata=sparse_route_metadata,
            route_layout=prepared_route_layout,
            name="smemBlockSparseSoftmaxMetadata1",
            h_k_idx=h_k_idx,
            b_idx=b_idx,
            scale_tensors=sage_scale_tensors,
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
            tma_desc_k_sf=tma_desc_k_sf,
            tma_desc_v_sf=tma_desc_v_sf,
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
        if not use_shared_inst_kv_rings:
            smem_k1 = SmemKvTileResource(
                pipeline_config=smem_k1_cfg,
                cfg=cfg,
                tma_desc_k=tma_desc_k,
                tma_desc_v=tma_desc_v,
                tma_desc_k_sf=tma_desc_k_sf,
                tma_desc_v_sf=tma_desc_v_sf,
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
            tma_desc_k_sf=tma_desc_k_sf,
            tma_desc_v_sf=tma_desc_v_sf,
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
        if not use_shared_inst_kv_rings:
            smem_v1 = SmemKvTileResource(
                pipeline_config=smem_v1_cfg,
                cfg=cfg,
                tma_desc_k=tma_desc_k,
                tma_desc_v=tma_desc_v,
                tma_desc_k_sf=tma_desc_k_sf,
                tma_desc_v_sf=tma_desc_v_sf,
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
            tma_desc_k_sf=tma_desc_k_sf,
            tma_desc_v_sf=tma_desc_v_sf,
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
    transformed_kv = None
    mma_smem_kv = smem_kv
    if cfg.use_transform_kv:
        if cfg.store_transformed_kv_in_tmem:
            assert smem_kv is not None
            transformed_kv = TmemTransformedKvResource(
                pipeline_config=smem_transformed_kv_cfg,
                cfg=cfg,
                src_smem_kv=smem_kv,
                name="tmemTransformedKv",
            )
        else:
            transformed_kv = SmemTransformedKvResource(
                pipeline_config=smem_transformed_kv_cfg,
                cfg=cfg,
                src_smem_kv=smem_kv,
                src_smem_k0=smem_k0,
                src_smem_k1=smem_k1,
                src_smem_v0=smem_v0,
                src_smem_v1=smem_v1,
                page_idx_kv=page_idx_kv,
                num_heads_kv=num_heads_kv,
                name="smemTransformedKv",
            )
        mma_smem_kv = transformed_kv

    # The correction task stages and reads the work tile's V scales itself.
    sage_v_scales = None
    if cfg.use_sage_attention:
        sage_v_scales = SageVScalesResource(
            pipeline_config=PipelineConfig(
                num_stages=1,
                num_bytes=0,
                producer_group=correction_grp,
                consumer_group=correction_grp,
                pipeline_type=PipelineType.AsyncAsync,
                cta_layout_vmnk=cta_layout,
                advance_on_wait=True,
            ),
            cfg=cfg,
            scale_tensors=sage_scale_tensors,
            h_k_idx=h_k_idx,
            b_idx=b_idx,
            name="sageVScales",
        )
    # Each softmax instance fills and reads its own ``sfK`` words; only the
    # SMEM form carries a pipeline. A mixed-geometry plan adds a second
    # resource for the proxy summaries.
    sage_k_scales0 = None
    sage_k_scales1 = None
    sage_summary_k_scales0 = None
    sage_summary_k_scales1 = None
    if cfg.use_sage_attention:

        def _sage_k_scales_cfg(groups: int, softmax_grp) -> PipelineConfig | None:
            if not cfg.sage_k_scales_in_smem_for(groups):
                return None
            return PipelineConfig(
                num_stages=SAGE_K_SCALES_RING_STAGES,
                num_bytes=0,
                producer_group=softmax_grp,
                consumer_group=softmax_grp,
                pipeline_type=PipelineType.AsyncAsync,
                cta_layout_vmnk=cta_layout,
                advance_on_wait=True,
            )

        token_groups = cfg.sage_k_groups_per_fragment
        sage_k_scales0 = SageKScalesResource(
            pipeline_config=_sage_k_scales_cfg(token_groups, softmax0_grp),
            inst_id=0,
            route_metadata=sparse_softmax_metadata0,
            name="sageKScales0",
            cfg=cfg,
            seqlens_kv=kv_seqlens,
            max_seq_len_kv=max_seq_len_kv,
            q_group_idx=q_group_idx,
            seq_len_q=seq_len_q,
            h_k_idx=h_k_idx,
            b_idx=b_idx,
            scale_tensors=sage_scale_tensors,
        )
        sage_k_scales1 = SageKScalesResource(
            pipeline_config=_sage_k_scales_cfg(token_groups, softmax1_grp),
            inst_id=1,
            route_metadata=sparse_softmax_metadata1,
            name="sageKScales1",
            cfg=cfg,
            seqlens_kv=kv_seqlens,
            max_seq_len_kv=max_seq_len_kv,
            q_group_idx=q_group_idx,
            seq_len_q=seq_len_q,
            h_k_idx=h_k_idx,
            b_idx=b_idx,
            scale_tensors=sage_scale_tensors,
        )
        if cfg.sage_mixed_k_geometry:
            summary_groups = cfg.sage_summary_k_groups_per_fragment
            sage_summary_k_scales0 = SageKScalesResource(
                pipeline_config=_sage_k_scales_cfg(summary_groups, softmax0_grp),
                inst_id=0,
                summary=True,
                route_metadata=sparse_softmax_metadata0,
                name="sageSummaryKScales0",
                cfg=cfg,
                seqlens_kv=kv_seqlens,
                max_seq_len_kv=max_seq_len_kv,
                q_group_idx=q_group_idx,
                seq_len_q=seq_len_q,
                h_k_idx=h_k_idx,
                b_idx=b_idx,
                scale_tensors=sage_scale_tensors,
            )
            sage_summary_k_scales1 = SageKScalesResource(
                pipeline_config=_sage_k_scales_cfg(summary_groups, softmax1_grp),
                inst_id=1,
                summary=True,
                route_metadata=sparse_softmax_metadata1,
                name="sageSummaryKScales1",
                cfg=cfg,
                seqlens_kv=kv_seqlens,
                max_seq_len_kv=max_seq_len_kv,
                q_group_idx=q_group_idx,
                seq_len_q=seq_len_q,
                h_k_idx=h_k_idx,
                b_idx=b_idx,
                scale_tensors=sage_scale_tensors,
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
        h_k_idx=h_k_idx,
        b_idx=b_idx,
        sync_barrier_id=0,
        sage_k_scales=sage_k_scales0,
        sage_summary_k_scales=sage_summary_k_scales0 or sage_k_scales0,
        name="tmemS0",
        scale_tensors=sage_scale_tensors,
    )
    tmem_s1 = None
    if not use_one_inst_kv:
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
            h_k_idx=h_k_idx,
            b_idx=b_idx,
            sync_barrier_id=1,
            score_seed_owner=tmem_s0,
            sage_k_scales=sage_k_scales1,
            sage_summary_k_scales=sage_summary_k_scales1 or sage_k_scales1,
            name="tmemS1",
            scale_tensors=sage_scale_tensors,
        )
    # Packed persistent QK derives the descriptor from Q's just-waited
    # consumer stage, avoiding a routed HEAD-to-LOOP descriptor value across
    # the guarded work-tile region. Fixed/static schedules keep their existing
    # explicit descriptor route.
    tmem_s0.q_ref = smem_q
    if tmem_s1 is not None:
        tmem_s1.q_ref = smem_q
    if cfg.uses_q_token_kv_block_sparse_page_membership:
        assert smem_page_offsets is not None
        tmem_s0.page_offsets_ref = smem_page_offsets
        if tmem_s1 is not None:
            tmem_s1.page_offsets_ref = smem_page_offsets

    smem_p0 = SmemPResource(
        inst_id=0,
        pipeline_config=smem_p0_cfg,
        cfg=cfg,
        scale_softmax_log2=scale_softmax_log2,
        use_variable_seqlens_kv=use_runtime_seqlens_kv,
        sage_k_scales=sage_k_scales0,
        sage_summary_k_scales=sage_summary_k_scales0 or sage_k_scales0,
        name="smemP0",
    )
    smem_p1 = None
    if not use_one_inst_kv:
        smem_p1 = SmemPResource(
            inst_id=1,
            pipeline_config=smem_p1_cfg,
            cfg=cfg,
            scale_softmax_log2=scale_softmax_log2,
            use_variable_seqlens_kv=use_runtime_seqlens_kv,
            sage_k_scales=sage_k_scales1,
            sage_summary_k_scales=sage_summary_k_scales1 or sage_k_scales1,
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
    tmem_softmax_local1 = None
    if not use_one_inst_kv:
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
    tmem_softmax_global1 = None
    if not use_one_inst_kv:
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
    smem_p0.tmem_o_ref = tmem_o
    tmem_softmax_global0.p_ref = smem_p0
    tmem_softmax_global0.tmem_s_ref = tmem_s0
    if not use_one_inst_kv:
        smem_p1.tmem_s_ref = tmem_s1
        smem_p1.tmem_o_ref = tmem_o
        tmem_softmax_global1.p_ref = smem_p1
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
        sage_v_scales=sage_v_scales,
    )
    tmem_corr0.tmem_o_ref = tmem_o
    tmem_corr0.softmax_local0_ref = tmem_softmax_local0
    tmem_corr0.softmax_local1_ref = None if use_one_inst_kv else tmem_softmax_local1
    tmem_corr1 = None
    if not use_one_inst_kv:
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
            sage_v_scales=sage_v_scales,
        )
        tmem_corr1.smem_p0_ref = smem_p0
        tmem_corr1.smem_p1_ref = smem_p1
        tmem_corr1.tmem_o_ref = tmem_o
        tmem_corr1.softmax_local0_ref = tmem_softmax_local0
        tmem_corr0.softmax_local1_ref = tmem_softmax_local1
        tmem_corr1.softmax_local1_ref = tmem_softmax_local1

    # ------------------------------------------------------------------
    # Finalize SMEM and membership mode before tracing tasks.
    # ------------------------------------------------------------------
    smem_resources = []
    if work_queue is not None:
        smem_resources.append(work_queue)
    if schedule_token_throttle is not None:
        smem_resources.append(schedule_token_throttle)
    if membership_lifetime is not None:
        smem_resources.append(membership_lifetime)
    smem_resources.append(smem_q)
    if smem_page_offsets is not None:
        smem_resources.append(smem_page_offsets)
    if smem_page_offsets_v is not None:
        smem_resources.append(smem_page_offsets_v)
    if sparse_kv_metadata0 is not None:
        smem_resources.append(sparse_kv_metadata0)
        smem_resources.append(sparse_kv_metadata1)
    if sparse_softmax_metadata0 is not None:
        smem_resources.append(sparse_softmax_metadata0)
        smem_resources.append(sparse_softmax_metadata1)
    if use_one_inst_qkv:
        smem_resources.append(smem_k0)
        smem_resources.append(smem_v0)
    elif use_per_inst_kv_resources:
        for resource in (smem_k0, smem_k1, smem_v0, smem_v1):
            if resource is not None:
                smem_resources.append(resource)
    else:
        smem_resources.append(smem_kv)
    if transformed_kv is not None:
        smem_resources.append(transformed_kv)
    smem_resources.append(smem_p0)
    if not use_one_inst_kv:
        smem_resources.append(smem_p1)
    # Each instance's scale rings follow its S resource in the layout.
    smem_resources.append(tmem_s0)
    smem_resources.extend(
        r for r in (sage_k_scales0, sage_summary_k_scales0) if r is not None
    )
    if not use_one_inst_kv:
        smem_resources.append(tmem_s1)
        smem_resources.extend(
            r for r in (sage_k_scales1, sage_summary_k_scales1) if r is not None
        )
    smem_resources.append(tmem_o)
    smem_resources.append(tmem_softmax_local0)
    if not use_one_inst_kv:
        smem_resources.append(tmem_softmax_local1)
    smem_resources.append(tmem_softmax_global0)
    if not use_one_inst_kv:
        smem_resources.append(tmem_softmax_global1)
    smem_resources.append(tmem_corr0)
    if not use_one_inst_kv:
        smem_resources.append(tmem_corr1)
    if sage_v_scales is not None:
        smem_resources.append(sage_v_scales)

    def allocate_smem() -> tuple[SmemAllocator, SmemAllocation | None]:
        allocator = SmemAllocator()
        clc_response_alloc = None
        for resource in smem_resources:
            allocator.add_resource(resource)
            if resource is work_queue and tile_sched_params is not None:
                # A compile-time offset from the unified base spares every
                # role a separate pointer that ptxas spills to local memory.
                clc_response_alloc = allocator.add(
                    SmemAllocation(
                        "clc_response",
                        dtype=cutlass.Int128,
                        count=_WORK_QUEUE_STAGES,
                        alignment=16,
                    )
                )
        allocator.add_tmem_ptr(
            SmemAllocation("fmha_tmem_ptr_i32", dtype=cutlass.Int32, alignment=4)
        )
        allocator.compute_layout()
        return allocator, clc_response_alloc

    def smem_bytes(allocator: SmemAllocator) -> int:
        # Include barriers and conservatively round data to tensor alignment.
        return (
            allocator.total_smem_bytes + cfg.stensor_align - 1
        ) // cfg.stensor_align * cfg.stensor_align + allocator.barrier_smem_bytes

    # The static profile checks bound only the Q and K/V pipelines; check the
    # complete layout against the SM100-family SMEM capacity.
    smem_capacity_bytes = SMEM_CAPACITY_KIB * BYTES_PER_KIB
    smem_allocator, clc_response_alloc = allocate_smem()
    if (
        cfg.uses_q_token_kv_block_sparse_page_membership
        and smem_bytes(smem_allocator) > smem_capacity_bytes
    ):
        # A full union can be much larger than a KV tile. Keep small routes
        # cached, but let Softmax read immutable packed GMEM words on demand
        # when the complete resource layout cannot hold the membership row.
        # This also accounts for the fixed K/V rings of one-inst D256.
        assert smem_page_offsets is not None
        smem_page_offsets.cache_memberships_in_smem = False
        smem_page_offsets._init_placeholder_state()
        smem_allocator, clc_response_alloc = allocate_smem()
    launch_smem_bytes = smem_bytes(smem_allocator)
    if launch_smem_bytes > smem_capacity_bytes:
        raise ValueError(
            "decode resources exceed the SM shared-memory capacity: "
            f"q_stages={cfg.q_stages}, kv_stages={cfg.kv_stages} need "
            f"{launch_smem_bytes} bytes, capacity is {smem_capacity_bytes} bytes"
        )
    if clc_response_alloc is not None:
        # Bind the scheduler config before the task manager creates the queue.
        smem_allocator.allocate()
        work_queue.tile_scheduler_config = (
            TileSchedulerConfig.create_clc_dynamic_persistent_tile_scheduler_params(
                tile_scheduler_params=tile_sched_params,
                response_ptr=cute.make_ptr(
                    cutlass.Int128,
                    smem_allocator.get(clc_response_alloc).data_ptr(),
                    mem_space=cutlass.AddressSpace.smem,
                    assumed_align=16,
                ),
            )
        )

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
        "sparse_row_route_begin": sparse_row_route_begin,
        "sparse_route_count": sparse_route_count,
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
                membership_lifetime=membership_lifetime,
                block_table_capacity=page_table_capacity
                if use_native_paged_kv
                else None,
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
                membership_lifetime=membership_lifetime,
                block_table_capacity=page_table_capacity
                if use_native_paged_kv
                else None,
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
    elif use_per_inst_kv_resources and not cfg.use_transform_kv:
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
            mma_smem_kv,
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
    transform_kv_task = None
    if cfg.use_transform_kv:
        transform_kv_task = create_transform_kv_task(
            smem_kv,
            smem_k0,
            smem_k1,
            smem_v0,
            smem_v1,
            transformed_kv,
            work_queue,
            cfg,
            domain=load_domain,
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
        membership_lifetime=membership_lifetime,
        sage_k_scales=sage_k_scales0,
        sage_summary_k_scales=sage_summary_k_scales0,
        **task_runtime_kwargs,
    )
    softmax1_task = None
    if not use_one_inst_kv:
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
            membership_lifetime=membership_lifetime,
            sage_k_scales=sage_k_scales1,
            sage_summary_k_scales=sage_summary_k_scales1,
            **task_runtime_kwargs,
        )
    if use_one_inst_kv:
        correction_task = create_correction_task_one_inst_qkv(
            tmem_softmax_local0,
            tmem_o,
            tmem_corr0,
            work_queue,
            cfg,
            tmem_stats_done=tmem_stats_done0,
            sage_v_scales=sage_v_scales,
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
            sage_v_scales=sage_v_scales,
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
            (cfg.wg4_padding_warp_idx, cfg.wg4_padding_num_warps),
            (cfg.wg5_padding_warp_idx, cfg.wg5_padding_num_warps),
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
    if transform_kv_task is not None:
        task_list.append(transform_kv_task)
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
    elif cfg.use_transform_kv and use_per_inst_kv_resources:
        resource_dependency_graph = {
            smem_q: [],
            smem_k0: smem_k_deps,
            smem_k1: smem_k_deps,
            smem_v0: smem_v_deps,
            smem_v1: smem_v_deps,
            transformed_kv: [smem_k0, smem_k1, smem_v0, smem_v1],
            tmem_s0: [transformed_kv, smem_q],
            smem_p0: [tmem_s0],
            tmem_softmax_local0: [tmem_s0],
            tmem_softmax_global0: [tmem_s0],
            tmem_o: [smem_p0, transformed_kv],
            tmem_corr0: [tmem_softmax_local0, tmem_o],
        }
        if not use_one_inst_kv:
            resource_dependency_graph.update(
                {
                    tmem_s1: [transformed_kv, smem_q],
                    smem_p1: [tmem_s1],
                    tmem_softmax_local1: [tmem_s1],
                    tmem_softmax_global1: [tmem_s1],
                    tmem_o: [smem_p0, smem_p1, transformed_kv],
                    tmem_corr1: [
                        tmem_softmax_local0,
                        tmem_softmax_local1,
                        tmem_o,
                    ],
                }
            )
    elif use_one_inst_kv:
        resource_dependency_graph = {
            smem_q: [],
            smem_kv: smem_kv_deps,
            tmem_s0: [mma_smem_kv, smem_q],
            smem_p0: [tmem_s0],
            tmem_softmax_local0: [tmem_s0],
            tmem_softmax_global0: [tmem_s0],
            tmem_o: [smem_p0, mma_smem_kv],
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
            smem_v0: smem_v_deps
            + ([sparse_kv_metadata0] if sparse_kv_metadata0 is not None else []),
            **(
                {
                    smem_k1: smem_k_deps
                    + (
                        [sparse_kv_metadata1] if sparse_kv_metadata1 is not None else []
                    ),
                    smem_v1: smem_v_deps
                    + (
                        [sparse_kv_metadata1] if sparse_kv_metadata1 is not None else []
                    ),
                }
                if smem_k1 is not None
                else {}
            ),
            tmem_s0: [smem_k0, smem_q],
            tmem_s1: [smem_k0 if smem_k1 is None else smem_k1, smem_q],
            smem_p0: [tmem_s0],
            smem_p1: [tmem_s1],
            tmem_softmax_local0: [tmem_s0],
            tmem_softmax_local1: [tmem_s1],
            tmem_softmax_global0: [tmem_s0],
            tmem_softmax_global1: [tmem_s1],
            tmem_o: [smem_p0, smem_p1, smem_v0]
            if smem_v1 is None
            else [smem_p0, smem_p1, smem_v0, smem_v1],
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
            tmem_s0: [mma_smem_kv, smem_q],
            tmem_s1: [mma_smem_kv, smem_q],
            smem_p0: [tmem_s0],
            smem_p1: [tmem_s1],
            tmem_softmax_local0: [tmem_s0],
            tmem_softmax_local1: [tmem_s1],
            tmem_softmax_global0: [tmem_s0],
            tmem_softmax_global1: [tmem_s1],
            tmem_o: [smem_p0, smem_p1, mma_smem_kv],
            tmem_corr0: [tmem_softmax_local0, tmem_o],
            tmem_corr1: [tmem_softmax_local0, tmem_softmax_local1, tmem_o],
        }
    if transformed_kv is not None and not use_per_inst_kv_resources:
        resource_dependency_graph[transformed_kv] = [smem_kv]
    if sparse_softmax_metadata0 is not None:
        resource_dependency_graph[smem_p0].append(sparse_softmax_metadata0)
        resource_dependency_graph[tmem_softmax_local0].append(sparse_softmax_metadata0)
        resource_dependency_graph[tmem_softmax_global0].append(sparse_softmax_metadata0)
        assert sparse_softmax_metadata1 is not None
        resource_dependency_graph[smem_p1].append(sparse_softmax_metadata1)
        resource_dependency_graph[tmem_softmax_local1].append(sparse_softmax_metadata1)
        resource_dependency_graph[tmem_softmax_global1].append(sparse_softmax_metadata1)
    if membership_lifetime is not None:
        resource_dependency_graph[membership_lifetime] = []
        for resource in (smem_p0, tmem_softmax_local0, tmem_softmax_global0):
            resource_dependency_graph[resource].append(membership_lifetime)
        if not use_one_inst_kv:
            for resource in (smem_p1, tmem_softmax_local1, tmem_softmax_global1):
                resource_dependency_graph[resource].append(membership_lifetime)
    if tmem_stats_done0 is not None:
        resource_dependency_graph[tmem_s0].append(tmem_stats_done0)
        resource_dependency_graph[tmem_stats_done0] = [tmem_softmax_local0]
        if tmem_stats_done1 is not None:
            assert tmem_s1 is not None
            assert tmem_softmax_local1 is not None
            resource_dependency_graph[tmem_s1].append(tmem_stats_done1)
            resource_dependency_graph[tmem_stats_done1] = [tmem_softmax_local1]
    # A softmax instance's outputs read its K scale words; an SMEM-form
    # resource fills them from the route's staged metadata, while a
    # register-form one hands them straight to the outputs. The correction
    # epilogue reads the V channel scales.
    for scales_pair, route_metadata, outputs in (
        (
            (sage_k_scales0, sage_summary_k_scales0),
            sparse_softmax_metadata0,
            (smem_p0, tmem_softmax_local0, tmem_softmax_global0),
        ),
        (
            (sage_k_scales1, sage_summary_k_scales1),
            sparse_softmax_metadata1,
            (smem_p1, tmem_softmax_local1, tmem_softmax_global1),
        ),
    ):
        for scales in scales_pair:
            if scales is None:
                continue
            resource_dependency_graph[scales] = (
                [route_metadata]
                if route_metadata is not None and scales.in_smem
                else []
            )
            for resource in outputs:
                resource_dependency_graph[resource].append(scales)
    if sage_v_scales is not None:
        resource_dependency_graph[sage_v_scales] = []
        for resource in (tmem_corr0, tmem_corr1):
            resource_dependency_graph[resource].append(sage_v_scales)
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
    dma_consumer_release_labels: dict[
        tuple[MemoryResource, MemoryResource], set[str]
    ] = {}
    if transformed_kv is not None:
        dma_consumer_release_labels[(transformed_kv, tmem_s0)] = {"k_desc_0"}
        dma_consumer_release_labels[(transformed_kv, tmem_o)] = {"v_desc_0"}
        if not use_one_inst_kv:
            dma_consumer_release_labels[(transformed_kv, tmem_s1)] = {"k_desc_1"}
            dma_consumer_release_labels[(transformed_kv, tmem_o)].add("v_desc_1")
    if smem_page_offsets is not None:
        if use_one_inst_qkv:
            if _can_hold_native_page_window(cfg, smem_page_offsets):
                # One consumer stage remains live across the complete K/V
                # cadence, so both DMA edges share its final release label.
                dma_consumer_release_labels.update(
                    {
                        (smem_page_offsets, smem_k0): {"read_offsets"},
                        (smem_page_offsets, smem_v0): {"read_offsets"},
                    }
                )
            else:
                dma_consumer_release_labels.update(
                    {
                        (smem_page_offsets, smem_k0): {"read_offsets_k0"},
                        (smem_page_offsets, smem_v0): {"read_offsets_v0"},
                    }
                )
        elif use_per_inst_kv_resources:
            if (
                smem_page_offsets_v is None
                and smem_page_offsets.holds_encoded_locator_window
            ):
                # Independent K/V data FIFOs can still share a single
                # read-only locator window until the last V tile is issued.
                dma_consumer_release_labels.update(
                    {
                        (smem_page_offsets, resource): {"read_offsets"}
                        for resource in (smem_k0, smem_k1, smem_v0, smem_v1)
                    }
                )
            elif smem_k1 is None:
                # One ring per operand serves both instances, so the single key
                # carries the release labels of both.
                dma_consumer_release_labels.update(
                    {
                        (smem_page_offsets, smem_k0): {
                            "read_offsets_k0",
                            "read_offsets_k1",
                        },
                        (smem_page_offsets_v, smem_v0): {
                            "read_offsets_v0",
                            "read_offsets_v1",
                        },
                    }
                )
            elif smem_page_offsets_v is not None:
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
            hold_page_window = _can_hold_native_page_window(cfg, smem_page_offsets)
            if cfg.num_head_dim_stages_kv > 1 and not cfg.uses_scattered_page_route:
                page_offset_labels = {"cache_page_ids"}
            elif hold_page_window:
                page_offset_labels = {"read_offsets"}
            elif use_one_inst_kv:
                page_offset_labels = {"read_offsets_k0", "read_offsets_v0"}
            else:
                page_offset_labels = {
                    "read_offsets_k0",
                    "read_offsets_k1",
                    "read_offsets_v0",
                    "read_offsets_v1",
                }
            dma_consumer_release_labels[(smem_page_offsets, smem_kv)] = (
                page_offset_labels
            )
    if smem_kv is not None and transformed_kv is None:
        dma_consumer_release_labels[(smem_kv, tmem_s0)] = {"k_desc_0"}
        dma_consumer_release_labels[(smem_kv, tmem_o)] = {"v_desc_0"}
        if not use_one_inst_kv:
            dma_consumer_release_labels[(smem_kv, tmem_s1)] = {"k_desc_1"}
            dma_consumer_release_labels[(smem_kv, tmem_o)].add("v_desc_1")

    # ------------------------------------------------------------------
    # TMEM allocator
    # ------------------------------------------------------------------
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
        tmem_allocator.add_resource(tmem_softmax_local0)
        if not use_one_inst_kv:
            tmem_allocator.add_resource(tmem_s1)
            tmem_allocator.add_resource(tmem_softmax_local1)
    tmem_allocator.add_resource(tmem_o)
    if cfg.store_transformed_kv_in_tmem:
        assert transformed_kv is not None
        tmem_allocator.add_resource(transformed_kv)
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
        if cfg.streams_tmem_p_fragments:
            # Streamed profiles keep O in the low 256 columns and overlay
            # packed P on each S region from its first column. Softmax streams
            # K32 fragments in order, so every 16-column P store only
            # overwrites scores that have already been consumed. Starting P
            # after the nominal stats columns would instead clobber the next
            # unread S fragment; streamed profiles keep their softmax stats in
            # SMEM.
            assert cfg.keeps_stats_via_smem
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

    eager_init_resources = [tmem_corr0] if use_one_inst_kv else [tmem_corr0, tmem_corr1]
    if cfg.streams_tmem_p_fragments:
        # Streamed TMEM P operands use one-way per-fragment ready barriers.
        # Initialize them beside correction's manually managed SMEM state.
        eager_init_resources.extend([smem_p0, smem_p1])
    if cfg.uses_int32_scores:
        # The score-seed operand tile is written once by its owning instance.
        eager_init_resources.append(tmem_s0)

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
    Static streamed profiles instead order P fragments with private mbarriers and
    reuses the matching TmemO-full barrier as the next-QK overwrite credit.
    Those intra-work protocols are below TaskManager's resource transitions,
    so its allocation-level checker cannot prove them. Persistent streaming has
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
def _deallocate_decode_tmem(
    tmem_ptr_i32: cute.Pointer,
    tmem_alloc_cols: Int32,
    cfg: cutlass.Constexpr[FmhaDecodeConfig],
) -> None:
    """Retire one CTA's TMEM after all of its task users have stopped."""
    warp_size = 32
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    dealloc_barrier_id = 13
    if cutlass.const_expr(cfg.use_keeps_mma_ab):
        prims.barrier_cta_sync(dealloc_barrier_id)
        if (
            warp_idx >= cfg.correction_warp_idx
            and warp_idx < cfg.correction_warp_idx + cfg.correction_num_warps
        ):
            tidx, _, _ = cute.arch.thread_idx()
            correction_thread_idx = tidx - cfg.correction_warp_idx * warp_size
            if correction_thread_idx < warp_size:
                tmem_ptr = prims.make_tmem_ptr(tmem_ptr_i32.load(), cutlass.Int8)
                prims.tcgen05_dealloc(tmem_ptr, tmem_alloc_cols)
    else:
        if (
            warp_idx >= cfg.correction_warp_idx
            and warp_idx < cfg.correction_warp_idx + cfg.correction_num_warps
        ):
            prims.barrier_cta_sync(
                dealloc_barrier_id,
                thread_count=cfg.correction_num_warps * warp_size,
            )
            tidx, _, _ = cute.arch.thread_idx()
            correction_thread_idx = tidx - cfg.correction_warp_idx * warp_size
            if correction_thread_idx < warp_size:
                tmem_ptr = prims.make_tmem_ptr(tmem_ptr_i32.load(), cutlass.Int8)
                prims.tcgen05_dealloc(tmem_ptr, tmem_alloc_cols)


@cute.jit
def _release_decode_dependents(
    cfg: cutlass.Constexpr[FmhaDecodeConfig],
) -> None:
    """Uniformly release a following PDL grid after one CTA's true tail."""
    if cutlass.const_expr(cfg.use_parallel_separate_reduction_pdl):
        prims.barrier_cta_sync(14)
        prims.griddepcontrol(kind=prims.GridDepAction.LAUNCH_DEPENDENTS)


@cute.jit
def _exit_cta_if_inactive(is_active: cutlass.Boolean) -> None:
    """Retire all threads in a CTA when a uniform activity predicate is false."""
    active_i32 = Int32(is_active)
    llvm.inline_asm(
        mlir_T.i32(),
        [active_i32.ir_value()],
        "{ .reg .pred _pexit; setp.eq.s32 _pexit, $1, 0;"
        " @_pexit exit; mov.u32 $0, 0; }",
        "=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def _retire_inactive_pdl_split(
    is_active: cutlass.Boolean,
    tmem_ptr_i32: cute.Pointer,
    tmem_alloc_cols: Int32,
    cfg: cutlass.Constexpr[FmhaDecodeConfig],
) -> None:
    """Cleanly retire a post-acquire split without staging TaskManager."""
    if not is_active:
        _deallocate_decode_tmem(tmem_ptr_i32, tmem_alloc_cols, cfg)
        _release_decode_dependents(cfg)
    # A runtime branch around ``TaskManager.run`` would make the DSL carry the
    # Python TaskManager meta object through ``scf.if``. Retiring only the
    # inactive, CTA-uniform path here leaves the active task graph straight-line.
    _exit_cta_if_inactive(is_active)


@cute.jit
def _run_decode_gen_active(
    tma_desc_q: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_k: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_v: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_k_sf: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_v_sf: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_k_atom: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_v_atom: cutlass.GridConstant[cuda.TensorMap],
    o_iter: cute.Pointer,
    g_s_k: Int32,
    g_h_k: Int32,
    g_scale_s_log2_e: Float32,
    g_output_scale: Float32,
    g_seqlens_kv: cute.Pointer | DirectSparseMetadataView,
    g_cu_seqlens_q: cute.Pointer,
    g_page_idx_kv: cute.Pointer | DirectSparseMetadataView,
    g_k_sf: cute.Pointer,
    g_v_sf: cute.Pointer,
    g_page_table_stride: Int64,
    g_page_table_capacity: Int32,
    g_q_token_kv_block_sparse_page_memberships: cute.Pointer,
    g_q_token_kv_block_sparse_page_membership_stride: Int32,
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
    defer_runtime_split_pruning: cutlass.Constexpr[bool],
    tile_sched_params: utils.ClcDynamicPersistentTileSchedulerParams | None,
    cfg: cutlass.Constexpr[FmhaDecodeConfig],
    seq_len_kv: cutlass.Constexpr[int] = 2048,
    use_variable_seqlens_kv: cutlass.Constexpr[bool] = False,
    use_native_paged_kv: cutlass.Constexpr[bool] = False,
    use_static_native_seqlens_kv: cutlass.Constexpr[bool] = False,
    g_sparse_row_route_offsets: cute.Pointer | None = None,
    g_sparse_row_route_counts: cute.Pointer | None = None,
    g_sparse_route_metadata: cute.Pointer | None = None,
    tma_desc_k_summary: cutlass.GridConstant[cuda.TensorMap] | None = None,
    tma_desc_v_summary: cutlass.GridConstant[cuda.TensorMap] | None = None,
    tma_desc_k_summary_atom: cutlass.GridConstant[cuda.TensorMap] | None = None,
    tma_desc_v_summary_atom: cutlass.GridConstant[cuda.TensorMap] | None = None,
    g_sage_q_scale: cute.Pointer | None = None,
    g_sage_k_scale: cute.Pointer | None = None,
    g_sage_k_summary_scale: cute.Pointer | None = None,
    g_sage_v_scale: cute.Pointer | None = None,
    g_sage_v_mean: cute.Pointer | None = None,
    g_sage_q_scale_head_stride: Int32 | None = None,
    g_sage_k_scale_head_stride: Int32 | None = None,
    g_sage_k_summary_scale_head_stride: Int32 | None = None,
) -> None:
    """Run the complete decode body for one runtime-valid Q tile.

    Builds resources/tasks via `_build_decode_gen_schedule` (shared with the
    validation path), then owns the matched TMA prefetch, SMEM/TMEM setup,
    TaskManager execution, and TMEM teardown lifecycle. Keeping that lifecycle
    in a void JIT helper lets the kernel wrapper omit it entirely for an
    overlaunched packed-Q tile without threading TaskManager state through a
    dynamic branch.
    """

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
        if cutlass.const_expr(cfg.use_nvfp4_kv):
            prims.prefetch_tensormap(tma_desc_k_sf.get_ptr())
            prims.prefetch_tensormap(tma_desc_v_sf.get_ptr())
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

    q_output_rows = g_h_r
    if cutlass.const_expr(cfg.max_seq_len_q > 1):
        q_output_rows = g_h_r * Int32(cfg.max_seq_len_q)

    # Static block-sparse tiles read their prepared row header here, before
    # TMEM allocation and barrier setup, so that global-memory round trip is
    # hidden instead of stalling every task at its first schedule step.
    sparse_row_route_begin = None
    sparse_route_count = None
    if cutlass.const_expr(
        cfg.use_block_sparse
        and not cfg.use_persistent_scheduler
        and g_sparse_row_route_offsets is not None
        and g_sparse_row_route_counts is not None
    ):
        sparse_row_route_begin, sparse_route_count = _prefetch_prepared_sparse_row(
            cfg,
            g_sparse_row_route_offsets,
            g_sparse_row_route_counts,
            q_group_idx,
            h_k_idx,
            b_idx,
            g_h_k,
        )

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
        tma_desc_k_sf=tma_desc_k_sf.get_ptr(),
        tma_desc_v_sf=tma_desc_v_sf.get_ptr(),
        tma_desc_k_atom=tma_desc_k_atom.get_ptr(),
        tma_desc_v_atom=tma_desc_v_atom.get_ptr(),
        tma_desc_k_summary=tma_desc_k_summary_ptr,
        tma_desc_v_summary=tma_desc_v_summary_ptr,
        tma_desc_k_summary_atom=tma_desc_k_summary_atom_ptr,
        tma_desc_v_summary_atom=tma_desc_v_summary_atom_ptr,
        sage_q_scale_ptr=g_sage_q_scale,
        sage_k_scale_ptr=g_sage_k_scale,
        sage_k_summary_scale_ptr=g_sage_k_summary_scale,
        sage_v_scale_ptr=g_sage_v_scale,
        sage_v_mean_ptr=g_sage_v_mean,
        sage_q_scale_head_stride=g_sage_q_scale_head_stride,
        sage_k_scale_head_stride=g_sage_k_scale_head_stride,
        sage_k_summary_scale_head_stride=g_sage_k_summary_scale_head_stride,
        page_idx_kv=g_page_idx_kv,
        page_table_stride=g_page_table_stride,
        page_table_capacity=g_page_table_capacity,
        q_token_kv_block_sparse_page_memberships=g_q_token_kv_block_sparse_page_memberships,
        q_token_kv_block_sparse_page_membership_stride=g_q_token_kv_block_sparse_page_membership_stride,
        h_k_idx=h_k_idx,
        b_idx=b_idx,
        q_group_idx=q_group_idx,
        q_token_offset=q_token_offset,
        seq_len_q=seq_len_q,
        active_splits_kv=(None if defer_runtime_split_pruning else active_splits_kv),
        static_full_split_prefix=static_full_split_prefix,
        tile_sched_params=tile_sched_params,
        use_variable_seqlens_kv=use_variable_seqlens_kv,
        use_native_paged_kv=use_native_paged_kv,
        use_static_native_seqlens_kv=use_static_native_seqlens_kv,
        sparse_row_route_offsets=g_sparse_row_route_offsets,
        sparse_row_route_counts=g_sparse_row_route_counts,
        sparse_route_metadata=g_sparse_route_metadata,
        sparse_row_route_begin=sparse_row_route_begin,
        sparse_route_count=sparse_route_count,
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
        # Kernel construction is a latency-sensitive JIT path and may run once
        # per CUDA-graph bucket on every TP rank.  Keep the cheap structural
        # checks here, but leave the exhaustive interleaving proof to
        # build_decode_task_manager() and its offline tests.  `skip_validation`
        # only changes failures into warnings; it does not skip the expensive
        # state-space search.
        exhaustive_deadlock_race_check=False,
        assume_pdl_wait_completed=cfg.use_pdl,
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
    if cutlass.const_expr(cfg.use_pdl):
        # Delay the acquire until every CTA-local pipeline barrier and
        # SMEM/TMEM resource is initialized, but keep it before TaskManager
        # resolves any value published by the preceding producer grid.
        prims.griddepcontrol(kind=prims.GridDepAction.WAIT)
    if cutlass.const_expr(defer_runtime_split_pruning):
        # A split launch cannot read its producer-owned sequence length until
        # the dependency above is acquired. Every physical split CTA
        # initializes the same static task resources first; only the useful
        # runtime prefix enters TaskManager. Pruned CTAs take the equivalent
        # TMEM teardown and dependent-release helpers, then retire before the
        # straight-line task graph.
        runtime_seq_len_kv = g_s_k
        if cutlass.const_expr(
            use_variable_seqlens_kv
            or (use_native_paged_kv and not use_static_native_seqlens_kv)
        ):
            runtime_seq_len_kv = Int32(g_seqlens_kv[b_idx])
        deferred_active_splits_kv = _runtime_active_splits_kv(
            cfg,
            runtime_seq_len_kv,
            seq_len_q,
            _q_group_token_base(cfg, q_group_idx),
        )
        split_idx = cute.arch.block_idx()[0] % Int32(cfg.splits_kv)
        run_task_graph = cute.arch.make_warp_uniform(
            split_idx < deferred_active_splits_kv
        )
        _retire_inactive_pdl_split(
            run_task_graph,
            tmem_ptr_i32,
            Int32(tmem_alloc_cols),
            cfg,
        )
    task_manager.run()

    # Every peer async-stores into an owner CTA's distributed SMEM and charges
    # the bytes to that owner's transaction mbarrier. Producer-only CTAs are
    # never remote-SMEM targets, so they may retire after their store issues;
    # each owner remains resident until its mbarrier completes and its local
    # reduction consumes all delivered partials. A final cluster rendezvous
    # would therefore only make completed producers wait for the owners.

    # Keeps aliases score/stat TMEM columns across task warp groups, while the
    # swaps path needs only its correction-warp rendezvous. In both cases the
    # following PDL grid is released only after deallocation completes.
    _deallocate_decode_tmem(tmem_ptr_i32, Int32(tmem_alloc_cols), cfg)
    _release_decode_dependents(cfg)


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
    """Preserve PDL ordering through a zero-work producer CTA."""
    if cutlass.const_expr(cfg.use_pdl):
        # This CTA has no task resources or producer-owned work, but it must
        # acquire the incoming dependency before retiring.
        prims.griddepcontrol(kind=prims.GridDepAction.WAIT)
    if cutlass.const_expr(cfg.use_parallel_separate_reduction_pdl):
        prims.griddepcontrol(kind=prims.GridDepAction.LAUNCH_DEPENDENTS)


@cute.jit
def _run_decode_gen_runtime_prefix(
    tma_desc_q: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_k: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_v: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_k_sf: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_v_sf: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_k_atom: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_v_atom: cutlass.GridConstant[cuda.TensorMap],
    o_iter: cute.Pointer,
    g_s_k: Int32,
    g_h_k: Int32,
    g_scale_s_log2_e: Float32,
    g_output_scale: Float32,
    g_seqlens_kv: cute.Pointer | DirectSparseMetadataView,
    g_cu_seqlens_q: cute.Pointer,
    g_page_idx_kv: cute.Pointer | DirectSparseMetadataView,
    g_k_sf: cute.Pointer,
    g_v_sf: cute.Pointer,
    g_page_table_stride: Int64,
    g_page_table_capacity: Int32,
    g_q_token_kv_block_sparse_page_memberships: cute.Pointer,
    g_q_token_kv_block_sparse_page_membership_stride: Int32,
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
    g_sparse_row_route_offsets: cute.Pointer | None,
    g_sparse_row_route_counts: cute.Pointer | None,
    g_sparse_route_metadata: cute.Pointer | None,
    tma_desc_k_summary: cutlass.GridConstant[cuda.TensorMap] | None = None,
    tma_desc_v_summary: cutlass.GridConstant[cuda.TensorMap] | None = None,
    tma_desc_k_summary_atom: cutlass.GridConstant[cuda.TensorMap] | None = None,
    tma_desc_v_summary_atom: cutlass.GridConstant[cuda.TensorMap] | None = None,
    g_sage_q_scale: cute.Pointer | None = None,
    g_sage_k_scale: cute.Pointer | None = None,
    g_sage_k_summary_scale: cute.Pointer | None = None,
    g_sage_v_scale: cute.Pointer | None = None,
    g_sage_v_mean: cute.Pointer | None = None,
    g_sage_q_scale_head_stride: Int32 | None = None,
    g_sage_k_scale_head_stride: Int32 | None = None,
    g_sage_k_summary_scale_head_stride: Int32 | None = None,
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
            if cutlass.const_expr(
                cfg.uses_q_token_kv_block_sparse_page_route
                and not cfg.supports_cluster_smem_reduction
            ):
                # When only the final configured split is empty, retaining its
                # neutral producer lets the fused GMEM reducer use the fully
                # unrolled prefix instead of its dynamically guarded fallback.
                # This also avoids changing the grid or adding a reducer launch.
                if active_splits_kv == Int32(cfg.splits_kv - 1):
                    active_splits_kv = Int32(cfg.splits_kv)
        split_idx = q_group_cta_idx % Int32(cfg.splits_kv)
        split_is_active = split_idx < active_splits_kv

    if cutlass.const_expr(cfg.supports_cluster_smem_reduction):
        if split_is_active:
            _run_decode_gen_active(
                tma_desc_q,
                tma_desc_k,
                tma_desc_v,
                tma_desc_k_sf,
                tma_desc_v_sf,
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
                g_k_sf,
                g_v_sf,
                g_page_table_stride,
                g_page_table_capacity,
                g_q_token_kv_block_sparse_page_memberships,
                g_q_token_kv_block_sparse_page_membership_stride,
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
                False,
                tile_sched_params,
                cfg,
                seq_len_kv,
                use_variable_seqlens_kv,
                use_native_paged_kv,
                use_static_native_seqlens_kv,
                g_sparse_row_route_offsets,
                g_sparse_row_route_counts,
                g_sparse_route_metadata,
                tma_desc_k_summary=tma_desc_k_summary,
                tma_desc_v_summary=tma_desc_v_summary,
                tma_desc_k_summary_atom=tma_desc_k_summary_atom,
                tma_desc_v_summary_atom=tma_desc_v_summary_atom,
                g_sage_q_scale=g_sage_q_scale,
                g_sage_k_scale=g_sage_k_scale,
                g_sage_k_summary_scale=g_sage_k_summary_scale,
                g_sage_v_scale=g_sage_v_scale,
                g_sage_v_mean=g_sage_v_mean,
                g_sage_q_scale_head_stride=g_sage_q_scale_head_stride,
                g_sage_k_scale_head_stride=g_sage_k_scale_head_stride,
                g_sage_k_summary_scale_head_stride=g_sage_k_summary_scale_head_stride,
            )
        else:
            _run_decode_gen_inactive_cluster_rank()
    else:
        if split_is_active:
            _run_decode_gen_active(
                tma_desc_q,
                tma_desc_k,
                tma_desc_v,
                tma_desc_k_sf,
                tma_desc_v_sf,
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
                g_k_sf,
                g_v_sf,
                g_page_table_stride,
                g_page_table_capacity,
                g_q_token_kv_block_sparse_page_memberships,
                g_q_token_kv_block_sparse_page_membership_stride,
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
                False,
                tile_sched_params,
                cfg,
                seq_len_kv,
                use_variable_seqlens_kv,
                use_native_paged_kv,
                use_static_native_seqlens_kv,
                g_sparse_row_route_offsets,
                g_sparse_row_route_counts,
                g_sparse_route_metadata,
                tma_desc_k_summary=tma_desc_k_summary,
                tma_desc_v_summary=tma_desc_v_summary,
                tma_desc_k_summary_atom=tma_desc_k_summary_atom,
                tma_desc_v_summary_atom=tma_desc_v_summary_atom,
                g_sage_q_scale=g_sage_q_scale,
                g_sage_k_scale=g_sage_k_scale,
                g_sage_k_summary_scale=g_sage_k_summary_scale,
                g_sage_v_scale=g_sage_v_scale,
                g_sage_v_mean=g_sage_v_mean,
                g_sage_q_scale_head_stride=g_sage_q_scale_head_stride,
                g_sage_k_scale_head_stride=g_sage_k_scale_head_stride,
                g_sage_k_summary_scale_head_stride=g_sage_k_summary_scale_head_stride,
            )
        else:
            _signal_padded_pdl_producer(cfg)


@cute.kernel
def decode_gen_kernel(
    tma_desc_q: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_k: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_v: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_k_sf: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_v_sf: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_k_atom: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_v_atom: cutlass.GridConstant[cuda.TensorMap],
    o_iter: cute.Pointer,
    g_s_k: Int32,
    g_h_k: Int32,
    g_scale_s_log2_e: Float32,
    g_output_scale: Float32,
    g_seqlens_kv: cute.Pointer | DirectSparseMetadataView,
    g_cu_seqlens_q: cute.Pointer,
    g_page_idx_kv: cute.Pointer | DirectSparseMetadataView,
    g_k_sf: cute.Pointer,
    g_v_sf: cute.Pointer,
    g_page_table_stride: Int64,
    g_page_table_capacity: Int32,
    g_q_token_kv_block_sparse_page_memberships: cute.Pointer,
    g_q_token_kv_block_sparse_page_membership_stride: Int32,
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
    g_sparse_row_route_offsets: cute.Pointer | None = None,
    g_sparse_row_route_counts: cute.Pointer | None = None,
    g_sparse_route_metadata: cute.Pointer | None = None,
    static_full_split_prefix: cutlass.Constexpr[bool] = False,
    tma_desc_k_summary: cutlass.GridConstant[cuda.TensorMap] | None = None,
    tma_desc_v_summary: cutlass.GridConstant[cuda.TensorMap] | None = None,
    tma_desc_k_summary_atom: cutlass.GridConstant[cuda.TensorMap] | None = None,
    tma_desc_v_summary_atom: cutlass.GridConstant[cuda.TensorMap] | None = None,
    g_sage_q_scale: cute.Pointer | None = None,
    g_sage_k_scale: cute.Pointer | None = None,
    g_sage_k_summary_scale: cute.Pointer | None = None,
    g_sage_v_scale: cute.Pointer | None = None,
    g_sage_v_mean: cute.Pointer | None = None,
    g_sage_q_scale_head_stride: Int32 | None = None,
    g_sage_k_scale_head_stride: Int32 | None = None,
    g_sage_k_summary_scale_head_stride: Int32 | None = None,
) -> None:
    """Dispatch one static Q/split tile and drain padded launch slots safely."""
    q_group_cta_idx, h_k_idx, b_idx = cute.arch.block_idx()
    if cutlass.const_expr(
        cfg.use_q_token_kv_block_sparse_route
        and not cfg.shares_sparse_pattern
        and not cfg.use_persistent_scheduler
    ):
        if cutlass.const_expr(isinstance(g_page_idx_kv, DirectSparseMetadataView)):
            g_page_idx_kv = g_page_idx_kv.with_head(h_k_idx)
        else:
            g_seqlens_kv = HeadIndexedMetadataView(g_seqlens_kv, g_h_k, h_k_idx)
            g_page_idx_kv = g_page_idx_kv + Int64(h_k_idx) * g_page_table_stride
            g_page_table_stride = g_page_table_stride * Int64(g_h_k)
            g_q_token_kv_block_sparse_page_memberships = (
                g_q_token_kv_block_sparse_page_memberships
                + Int64(h_k_idx)
                * Int64(g_q_token_kv_block_sparse_page_membership_stride)
            )
            g_q_token_kv_block_sparse_page_membership_stride = (
                g_q_token_kv_block_sparse_page_membership_stride * g_h_k
            )
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
                tma_desc_k_sf,
                tma_desc_v_sf,
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
                g_k_sf,
                g_v_sf,
                g_page_table_stride,
                g_page_table_capacity,
                g_q_token_kv_block_sparse_page_memberships,
                g_q_token_kv_block_sparse_page_membership_stride,
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
                False,
                tile_sched_params,
                cfg,
                seq_len_kv,
                use_variable_seqlens_kv,
                use_native_paged_kv,
                use_static_native_seqlens_kv,
                g_sparse_row_route_offsets,
                g_sparse_row_route_counts,
                g_sparse_route_metadata,
            )
        elif cutlass.const_expr(cfg.use_split_kv and cfg.use_pdl):
            # The immediately preceding grid may produce seq_lens. Send every
            # configured split through resource initialization and defer
            # useful-prefix pruning until after the PDL acquire.
            _run_decode_gen_active(
                tma_desc_q,
                tma_desc_k,
                tma_desc_v,
                tma_desc_k_sf,
                tma_desc_v_sf,
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
                g_k_sf,
                g_v_sf,
                g_page_table_stride,
                g_page_table_capacity,
                g_q_token_kv_block_sparse_page_memberships,
                g_q_token_kv_block_sparse_page_membership_stride,
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
                False,
                True,
                tile_sched_params,
                cfg,
                seq_len_kv,
                use_variable_seqlens_kv,
                use_native_paged_kv,
                use_static_native_seqlens_kv,
                g_sparse_row_route_offsets,
                g_sparse_row_route_counts,
                g_sparse_route_metadata,
                tma_desc_k_summary=tma_desc_k_summary,
                tma_desc_v_summary=tma_desc_v_summary,
                tma_desc_k_summary_atom=tma_desc_k_summary_atom,
                tma_desc_v_summary_atom=tma_desc_v_summary_atom,
                g_sage_q_scale=g_sage_q_scale,
                g_sage_k_scale=g_sage_k_scale,
                g_sage_k_summary_scale=g_sage_k_summary_scale,
                g_sage_v_scale=g_sage_v_scale,
                g_sage_v_mean=g_sage_v_mean,
                g_sage_q_scale_head_stride=g_sage_q_scale_head_stride,
                g_sage_k_scale_head_stride=g_sage_k_scale_head_stride,
                g_sage_k_summary_scale_head_stride=g_sage_k_summary_scale_head_stride,
            )
        else:
            _run_decode_gen_runtime_prefix(
                tma_desc_q,
                tma_desc_k,
                tma_desc_v,
                tma_desc_k_sf,
                tma_desc_v_sf,
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
                g_k_sf,
                g_v_sf,
                g_page_table_stride,
                g_page_table_capacity,
                g_q_token_kv_block_sparse_page_memberships,
                g_q_token_kv_block_sparse_page_membership_stride,
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
                g_sparse_row_route_offsets,
                g_sparse_row_route_counts,
                g_sparse_route_metadata,
                tma_desc_k_summary=tma_desc_k_summary,
                tma_desc_v_summary=tma_desc_v_summary,
                tma_desc_k_summary_atom=tma_desc_k_summary_atom,
                tma_desc_v_summary_atom=tma_desc_v_summary_atom,
                g_sage_q_scale=g_sage_q_scale,
                g_sage_k_scale=g_sage_k_scale,
                g_sage_k_summary_scale=g_sage_k_summary_scale,
                g_sage_v_scale=g_sage_v_scale,
                g_sage_v_mean=g_sage_v_mean,
                g_sage_q_scale_head_stride=g_sage_q_scale_head_stride,
                g_sage_k_scale_head_stride=g_sage_k_scale_head_stride,
                g_sage_k_summary_scale_head_stride=g_sage_k_summary_scale_head_stride,
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
    k_sf_iter: cute.Pointer,
    v_sf_iter: cute.Pointer,
    o_iter: cute.Pointer,
    seqlens_kv_iter: cute.Pointer | DirectSparseMetadataView,
    cu_seqlens_q_iter: cute.Pointer,
    total_q_tokens: Int32,
    page_idx_kv_iter: cute.Pointer | DirectSparseMetadataView,
    q_token_kv_block_sparse_page_memberships_iter: cute.Pointer,
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
    page_table_stride: Int64 = 0,
    page_table_capacity: Int32 = 0,
    q_token_kv_block_sparse_page_membership_stride: Int32 = 0,
    num_physical_kv_pages: Int64 = 0,
    k_page_stride: Int64 = 0,
    k_head_stride: Int64 = 0,
    k_token_stride: Int64 = 0,
    v_page_stride: Int64 = 0,
    v_head_stride: Int64 = 0,
    v_token_stride: Int64 = 0,
    static_full_split_prefix: cutlass.Constexpr[bool] = False,
    use_static_native_seqlens_kv: cutlass.Constexpr[bool] = False,
    sage_q_scale_iter: cute.Pointer | None = None,
    sage_k_scale_iter: cute.Pointer | None = None,
    sage_v_scale_iter: cute.Pointer | None = None,
    sage_v_mean_iter: cute.Pointer | None = None,
    sage_q_scale_head_stride: Int32 | None = None,
    sage_k_scale_head_stride: Int32 | None = None,
) -> None:
    """Standalone JIT launcher for FMHA decode TS.

    The ``sage_*`` scale arguments are used only by a Sage attention config.
    """
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
        kv_tma_d = d
        if cutlass.const_expr(
            cfg.use_nvfp4_kv and not cfg.store_transformed_kv_in_tmem
        ):
            kv_tma_d = d // Int32(2)
        if cutlass.const_expr(use_native_paged_kv):
            total_pages = num_physical_kv_pages
            storage_tokens_per_page = Int32(cfg.effective_storage_tokens_per_page)
            kv_shape = (
                kv_tma_d,
                storage_tokens_per_page,
                h_k,
                total_pages,
            )
            k_tma_page_stride = k_page_stride
            k_tma_head_stride = k_head_stride
            k_tma_token_stride = k_token_stride
            v_tma_page_stride = v_page_stride
            v_tma_head_stride = v_head_stride
            v_tma_token_stride = v_token_stride
            if cutlass.const_expr(cfg.store_transformed_kv_in_tmem):
                # B4X16_P64 tensor-map strides are expressed in logical FP4
                # lanes while the native cache reports packed-byte strides.
                k_tma_page_stride = k_page_stride * Int64(2)
                k_tma_head_stride = k_head_stride * Int64(2)
                k_tma_token_stride = k_token_stride * Int64(2)
                v_tma_page_stride = v_page_stride * Int64(2)
                v_tma_head_stride = v_head_stride * Int64(2)
                v_tma_token_stride = v_token_stride * Int64(2)
            k_layout = cute.make_layout(
                kv_shape,
                stride=(
                    1,
                    k_tma_token_stride,
                    k_tma_head_stride,
                    k_tma_page_stride,
                ),
            )
            v_layout = cute.make_layout(
                kv_shape,
                stride=(
                    1,
                    v_tma_token_stride,
                    v_tma_head_stride,
                    v_tma_page_stride,
                ),
            )
            k_tma = cute.make_tensor(k_iter, k_layout)
            v_tma = cute.make_tensor(v_iter, v_layout)
        else:
            total_pages = b * Int32(cfg.max_num_pages_per_seq_kv)
            kv_layout = cute.make_layout(
                (kv_tma_d, Int32(cfg.num_tokens_per_page), h_k, total_pages),
                stride=(
                    1,
                    kv_tma_d,
                    kv_tma_d * Int32(cfg.num_tokens_per_page),
                    kv_tma_d * Int32(cfg.num_tokens_per_page) * h_k,
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
            skipped_elems = skipped_tokens * d * h_k
            k_tma_iter = k_iter + skipped_elems
            v_tma_iter = v_iter + skipped_elems
            kv_s_for_tma = Int32(effective_seq_len_kv)
        # Contiguous K/V are compact BSHD, matching the public PrimTS tensor
        # contract and the Q layout above.
        kv_layout = cute.make_layout(
            (d, kv_s_for_tma, h_k, b), stride=(1, d * h_k, d, kv_b_stride)
        )
        k_tma = cute.make_tensor(k_tma_iter, kv_layout)
        v_tma = cute.make_tensor(v_tma_iter, kv_layout)

    # Keep the TMA inner box at 128B when possible, but never exceed headDim.
    # box_dim is expressed in elements of the source dtype, not bytes.
    tma_box0_q = min(128 // cfg.q_dtype_bytes, cfg.headdim)
    tma_box0_k = min(128 // cfg.k_dtype_bytes, cfg.headdim)
    tma_box0_v = min(128 // cfg.v_dtype_bytes, cfg.headdim)
    tma_swizzle_q = cuda.TensorMapSwizzle.s128b
    if cutlass.const_expr(cfg.q_dtype_bytes == 1 and cfg.headdim == 64):
        tma_swizzle_q = cuda.TensorMapSwizzle.s64b
    tma_swizzle_k = cuda.TensorMapSwizzle.s128b
    if cutlass.const_expr(cfg.k_dtype_bytes == 1 and cfg.headdim == 64):
        tma_swizzle_k = cuda.TensorMapSwizzle.s64b
    tma_swizzle_v = cuda.TensorMapSwizzle.s128b
    if cutlass.const_expr(cfg.v_dtype_bytes == 1 and cfg.headdim == 64):
        tma_swizzle_v = cuda.TensorMapSwizzle.s64b
    if cutlass.const_expr(cfg.use_nvfp4_kv):
        if cutlass.const_expr(cfg.store_transformed_kv_in_tmem):
            # Unpacking TMA consumes 128 logical FP4 lanes and materializes
            # one s128b-swizzled byte per lane in the raw SMEM stage.
            tma_box0_k = cfg.head_dim_kv_stage
            tma_box0_v = cfg.head_dim_kv_stage
            tma_swizzle_k = cuda.TensorMapSwizzle.s128b
            tma_swizzle_v = cuda.TensorMapSwizzle.s128b
        else:
            tma_box0_k = cfg.head_dim_kv_stage // 2
            tma_box0_v = cfg.head_dim_kv_stage // 2
            tma_swizzle_k = cuda.TensorMapSwizzle.none
            tma_swizzle_v = cuda.TensorMapSwizzle.none
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
            swizzle=tma_swizzle_q,
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
            swizzle=tma_swizzle_q,
        )
    if cutlass.const_expr(cfg.store_transformed_kv_in_tmem):
        tma_desc_k = create_tensor_map_tiled_from_view(
            k_tma,
            box_dims=(tma_box0_k, tma_kv_tokens, 1, 1),
            stride_order=(0, 1, 2, 3),
            swizzle=tma_swizzle_k,
            dtype=cutlass.Float4E2M1FNx2,
            tma_format=cuda.TensorMapDataFormat.B4X16_P64,
        )
        tma_desc_v = create_tensor_map_tiled_from_view(
            v_tma,
            box_dims=(tma_box0_v, tma_kv_tokens, 1, 1),
            stride_order=(0, 1, 2, 3),
            swizzle=tma_swizzle_v,
            dtype=cutlass.Float4E2M1FNx2,
            tma_format=cuda.TensorMapDataFormat.B4X16_P64,
        )
    else:
        tma_desc_k = create_tensor_map_tiled_from_view(
            k_tma,
            box_dims=(tma_box0_k, tma_kv_tokens, 1, 1),
            stride_order=(0, 1, 2, 3),
            swizzle=tma_swizzle_k,
        )
        tma_desc_v = create_tensor_map_tiled_from_view(
            v_tma,
            box_dims=(tma_box0_v, tma_kv_tokens, 1, 1),
            stride_order=(0, 1, 2, 3),
            swizzle=tma_swizzle_v,
        )

    # NVFP4 scale factors are staged into SMEM by the same SmemKv TMA pipeline
    # as the packed K/V. The SF tensor is logically
    # (headdim // 16, storage_tokens_per_page, h_k, total_pages) E4M3. K uses
    # token-major layout and V uses TRT-LLM's 4-token interleaved layout. The
    # inner SF box (headdim // 16 = 8 B) is below TMA's
    # 16 B minimum, so fold `r` tokens into dim0 for up to a 128 B inner box,
    # without spanning semantic page fragments.
    # This reshape is a pure reinterpretation of the same contiguous bytes.
    tma_desc_k_sf = tma_desc_k
    tma_desc_v_sf = tma_desc_v
    if cutlass.const_expr(cfg.use_nvfp4_kv):
        sf_per_token = cfg.headdim // 16
        sf_r = cfg.sf_tma_reshape_factor
        sf_inner = sf_per_token * sf_r
        sf_storage_tokens_per_page = cfg.num_tokens_per_page
        if cutlass.const_expr(use_native_paged_kv):
            sf_storage_tokens_per_page = cfg.effective_storage_tokens_per_page
        sf_tokens_outer = sf_storage_tokens_per_page // sf_r
        # The descriptor spans a physical page, but each transaction copies
        # only the semantic fragment selected by the decoded page locator.
        sf_box_tokens_outer = cfg.num_tokens_per_page // sf_r
        sf_page_elems = Int32(sf_per_token * sf_storage_tokens_per_page)
        sf_layout = cute.make_layout(
            (Int32(sf_inner), Int32(sf_tokens_outer), h_k, total_pages),
            stride=(1, Int32(sf_inner), sf_page_elems, sf_page_elems * h_k),
        )
        k_sf_tma = cute.make_tensor(k_sf_iter, sf_layout)
        v_sf_tma = cute.make_tensor(v_sf_iter, sf_layout)
        tma_desc_k_sf = cuda.create_tensor_map_tiled_from_view(
            k_sf_tma,
            box_dims=(sf_inner, sf_box_tokens_outer, 1, 1),
            stride_order=(0, 1, 2, 3),
            swizzle=cuda.TensorMapSwizzle.none,
        )
        tma_desc_v_sf = cuda.create_tensor_map_tiled_from_view(
            v_sf_tma,
            box_dims=(sf_inner, sf_box_tokens_outer, 1, 1),
            stride_order=(0, 1, 2, 3),
            swizzle=cuda.TensorMapSwizzle.none,
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
        tma_desc_k_sf,
        tma_desc_v_sf,
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
        k_sf_iter,
        v_sf_iter,
        page_table_stride,
        page_table_capacity,
        q_token_kv_block_sparse_page_memberships_iter,
        q_token_kv_block_sparse_page_membership_stride,
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
        null_sparse_route_ptr,
        null_sparse_route_ptr,
        null_sparse_route_ptr,
        static_full_split_prefix,
        g_sage_q_scale=sage_q_scale_iter,
        g_sage_k_scale=sage_k_scale_iter,
        g_sage_v_scale=sage_v_scale_iter,
        g_sage_v_mean=sage_v_mean_iter,
        g_sage_q_scale_head_stride=sage_q_scale_head_stride,
        g_sage_k_scale_head_stride=sage_k_scale_head_stride,
    ).launch(
        grid=grid,
        block=[cfg.threads_per_cta, 1, 1],
        cluster=cluster_shape,
        stream=stream,
        min_blocks_per_mp=_decode_min_blocks_per_mp(cfg, effective_seq_len_kv),
        # PDL-capable attention initializes all local task resources before
        # acquiring its immediately preceding producer. Split attention also
        # defers runtime prefix pruning until after that acquire.
        use_pdl=cfg.use_pdl,
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
    sage_q_scale_iter: cute.Pointer | None = None,
    sage_k_scale_iter: cute.Pointer | None = None,
    sage_k_summary_scale_iter: cute.Pointer | None = None,
    sage_v_scale_iter: cute.Pointer | None = None,
    sage_v_mean_iter: cute.Pointer | None = None,
    sage_q_scale_head_stride: Int32 | None = None,
    sage_k_scale_head_stride: Int32 | None = None,
    sage_k_summary_scale_head_stride: Int32 | None = None,
) -> None:
    """Launch attention over exact and typed exact/proxy prepared KV routes.

    A preceding prepare kernel has already resolved each BSR row into compact
    logical atom origins, storage locators, validity flags, and optional token
    words. Exact routes address K/V; proxy routes address summary K/V. Both
    layouts execute the same ``decode_gen_kernel`` schedule and
    physical copy policy. Exact builds constexpr-elide summary TensorMaps.
    The ``sage_*`` scale arguments are used only by a Sage attention config,
    the summary K scales only by its proxy routes.
    """
    if cutlass.const_expr(not cfg.use_block_sparse):
        raise ValueError("fmha_block_sparse_launch requires block-sparse config")
    if cutlass.const_expr(cfg.use_block_sparse_proxy_routes and cfg.use_paged_kv):
        raise ValueError("block-sparse proxy routes require contiguous K/V")

    log2_e = math.log2(math.e)
    b, h_q, h_k, s_k, d = problem_shape
    h_r = h_q // h_k
    q_seq = Int32(cfg.max_seq_len_q)
    q_strides, _ = _block_sparse_bshd_tma_strides(
        q_seq=q_seq,
        h_q=h_q,
        h_k=h_k,
        s_k=s_k,
        d=d,
        element_bytes=cfg.q_dtype_bytes,
    )
    _, kv_strides = _block_sparse_bshd_tma_strides(
        q_seq=q_seq,
        h_q=h_q,
        h_k=h_k,
        s_k=s_k,
        d=d,
        element_bytes=cfg.kv_dtype_bytes,
    )

    # H128 uses a 128-byte inner box: 64 two-byte or 128 one-byte elements.
    # The sparse profile validator rejects other head dimensions.
    tma_box0_q = min(128 // cfg.q_dtype_bytes, cfg.headdim)
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
            tma_box0_q,
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
            dtype=cfg.k_dtype,
            global_dims=kv_dims,
            global_strides=kv_strides,
            box_dims=(tma_box0, primary_kv_box_size, 1, 1),
            swizzle=tma_swizzle,
        )
        v_desc_primary = create_tensor_map_tiled(
            global_address=v_iter.toint(),
            dtype=cfg.v_dtype,
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
                dtype=cfg.k_dtype,
                global_dims=kv_dims,
                global_strides=kv_strides,
                box_dims=(tma_box0, kv_atom_size, 1, 1),
                swizzle=tma_swizzle,
            )
            v_desc_atom = create_tensor_map_tiled(
                global_address=v_iter.toint(),
                dtype=cfg.v_dtype,
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
                element_bytes=cfg.kv_dtype_bytes,
            )
            summary_dims = (d, num_kv_blocks, h_k, b)
            k_desc_summary_primary = create_tensor_map_tiled(
                global_address=k_summary_iter.toint(),
                dtype=cfg.k_dtype,
                global_dims=summary_dims,
                global_strides=summary_kv_strides,
                box_dims=(tma_box0, primary_kv_box_size, 1, 1),
                swizzle=tma_swizzle,
            )
            v_desc_summary_primary = create_tensor_map_tiled(
                global_address=v_summary_iter.toint(),
                dtype=cfg.v_dtype,
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
                    dtype=cfg.k_dtype,
                    global_dims=summary_dims,
                    global_strides=summary_kv_strides,
                    box_dims=(tma_box0, kv_atom_size, 1, 1),
                    swizzle=tma_swizzle,
                )
                v_desc_summary_atom = create_tensor_map_tiled(
                    global_address=v_summary_iter.toint(),
                    dtype=cfg.v_dtype,
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
        null_f32_ptr,
        null_f32_ptr,
        Int64(0),  # g_page_table_stride
        Int32(0),  # g_page_table_capacity
        null_i32_ptr,
        Int32(0),  # g_q_token_kv_block_sparse_page_membership_stride
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
        row_route_offsets_iter,
        row_route_counts_iter,
        route_metadata_iter,
        False,  # static_full_split_prefix
        tma_desc_k_summary=k_desc_summary_primary,
        tma_desc_v_summary=v_desc_summary_primary,
        tma_desc_k_summary_atom=k_desc_summary_atom,
        tma_desc_v_summary_atom=v_desc_summary_atom,
        g_sage_q_scale=sage_q_scale_iter,
        g_sage_k_scale=sage_k_scale_iter,
        g_sage_k_summary_scale=sage_k_summary_scale_iter,
        g_sage_v_scale=sage_v_scale_iter,
        g_sage_v_mean=sage_v_mean_iter,
        g_sage_q_scale_head_stride=sage_q_scale_head_stride,
        g_sage_k_scale_head_stride=sage_k_scale_head_stride,
        g_sage_k_summary_scale_head_stride=sage_k_summary_scale_head_stride,
    ).launch(
        grid=grid,
        block=[cfg.threads_per_cta, 1, 1],
        cluster=[1, 1, 1],
        stream=stream,
        # Reuse dense decode's entry-occupancy contract, including the long
        # KV256 profiles that execute dynamic register reallocation.
        min_blocks_per_mp=_decode_min_blocks_per_mp(cfg, seq_len_kv),
        use_pdl=False,
    )
