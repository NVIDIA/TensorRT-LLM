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

"""Tile and stage index helper functions for MLA decode."""

import cutlass
import cutlass.cute as cute
from cutlass import Int32

from cutlass.experimental.task_scheduling.resources import StageInfo

from .layout import _TASK_CACHE_SEQ_LEN_KV
from .mask import MaskType, mask_visible_k_length
from .query import (
    flat_query_row_state,
    query_batch_bounds,
    runtime_flat_query_tile_has_rows,
)
from .stage import MlaStage
from ..throughput_latency_1cta.config import MlaConfig


@cute.jit
def local_kv_tile_idx(
    cfg: MlaConfig,
    stage_info: StageInfo,
    inst_id: int,
    is_v: int,
    *,
    section: cutlass.Constexpr[MlaStage],
):
    """Return the local K/V tile id for one staged pipeline instance."""
    if cutlass.const_expr(section == MlaStage.Head):
        return Int32(inst_id)
    if cutlass.const_expr(section == MlaStage.Loop):
        base = stage_info.loop_offset * Int32(cfg.num_insts_kv)
        if cutlass.const_expr(is_v):
            return base + Int32(inst_id)
        return base + Int32(cfg.num_insts_kv + inst_id)
    if cutlass.const_expr(cfg.total_kv_tiles <= cfg.num_insts_kv):
        return Int32(inst_id)
    return stage_info.loop_end * Int32(cfg.num_insts_kv) + Int32(inst_id)


@cute.jit
def softmax_kv_tile_idx(cfg: MlaConfig, stage_info: StageInfo, inst_id: int):
    """Return the K tile consumed by one softmax instance in the loop body."""
    return stage_info.loop_offset * Int32(cfg.num_insts_kv) + Int32(inst_id)


@cute.jit
def runtime_total_kv_tiles(cfg: MlaConfig, seq_len_kv):
    """Return ceil(seq_len_kv / tile_size_kv) at runtime."""
    return (seq_len_kv + Int32(cfg.tile_size_kv - 1)) // Int32(cfg.tile_size_kv)


def active_split_kv_count(
    seq_len_kv: int,
    tile_size_kv: int,
    num_insts_kv: int,
    configured_splits_kv: int,
) -> int:
    """Return the host form of the runtime active split-prefix rule."""

    if seq_len_kv < 0:
        raise ValueError("seq_len_kv must be non-negative")
    if tile_size_kv <= 0:
        raise ValueError("tile_size_kv must be positive")
    if num_insts_kv <= 0:
        raise ValueError("num_insts_kv must be positive")
    if configured_splits_kv <= 0:
        raise ValueError("configured_splits_kv must be positive")
    total_kv_tiles = (seq_len_kv + tile_size_kv - 1) // tile_size_kv
    groups_per_cta = (total_kv_tiles + configured_splits_kv * num_insts_kv - 1) // (
        configured_splits_kv * num_insts_kv
    )
    local_kv_tiles = max(groups_per_cta * num_insts_kv, num_insts_kv)
    return (total_kv_tiles + local_kv_tiles - 1) // local_kv_tiles


def runtime_split_pruning_is_profitable(configured_splits_kv: int) -> bool:
    """Return whether split pruning can retire enough 1CTA work to pay for itself.

    A configured S2/S3 launch can retire at most one or two mainloop CTAs per
    logical tile, while still requiring every cluster rank to publish a neutral
    partial and perform its static row-owner reduction.  Hardware validation showed
    that this does not amortize the runtime activity branch.  Starting at S4,
    contracted requests retire enough K/V task graphs to recover the control
    cost.  This is a topology rule, not a problem-shape list or user knob.
    """

    if configured_splits_kv <= 0:
        raise ValueError("configured_splits_kv must be positive")
    return configured_splits_kv >= 4


@cute.jit
def _runtime_configured_local_kv_tiles(cfg: MlaConfig, seq_len_kv):
    """Return the instruction-aligned local span for configured split capacity."""

    total_kv_tiles = runtime_total_kv_tiles(cfg, seq_len_kv)
    num_insts_kv = Int32(cfg.num_insts_kv)
    tiles_per_group = Int32(cfg.num_ctas_per_seq_kv) * num_insts_kv
    num_groups = (total_kv_tiles + tiles_per_group - Int32(1)) // tiles_per_group
    return cute.math.max(num_groups * num_insts_kv, num_insts_kv)


@cute.jit
def runtime_num_ctas_kv(cfg: MlaConfig, seq_len_kv):
    """Return the active prefix of configured split-KV CTAs.

    One CTA's minimum useful K unit is ``tile_size_kv * num_insts_kv``.
    Launch and workspace shapes retain the configured maximum split count;
    runtime-short sequences activate only the prefix that owns real work.
    """
    if cutlass.const_expr(cfg.use_multi_ctas_kv != 1):
        return Int32(1)
    total_kv_tiles = runtime_total_kv_tiles(cfg, seq_len_kv)
    local_kv_tiles = _runtime_configured_local_kv_tiles(cfg, seq_len_kv)
    return (total_kv_tiles + local_kv_tiles - Int32(1)) // local_kv_tiles


@cute.jit
def runtime_local_kv_tiles(cfg: MlaConfig, seq_len_kv):
    """Return the padded local KV tile count assigned to each KV CTA group."""
    if cutlass.const_expr(cfg.use_multi_ctas_kv != 1):
        return runtime_total_kv_tiles(cfg, seq_len_kv)
    return _runtime_configured_local_kv_tiles(cfg, seq_len_kv)


@cute.jit
def runtime_base_seq_len_kv(cfg: MlaConfig, cache_seqs, batch_idx):
    """Return the raw KV sequence length for a batch row."""
    if cutlass.const_expr(cache_seqs is None):
        return Int32(cfg.seq_len_kv)
    return Int32(cache_seqs[batch_idx])


@cute.jit
def runtime_seq_len_kv_for_logical_q(
    cfg: MlaConfig,
    cache_seqs,
    batch_idx,
    logical_q_idx,
    cu_seqlens_q=None,
):
    """Return the mask-visible KV length for one logical Q row.

    Dense decode exposes the full runtime cache. Bottom-right causal decode
    removes the speculative K positions following ``logical_q_idx``.
    """
    _, logical_seq_len_q = query_batch_bounds(
        cu_seqlens_q,
        batch_idx,
        cfg.logical_seq_len_q,
    )
    return mask_visible_k_length(
        cfg.mask_type,
        runtime_base_seq_len_kv(cfg, cache_seqs, batch_idx),
        logical_q_idx,
        logical_seq_len_q,
    )


@cute.jit
def runtime_seq_len_kv_for_q(
    cfg: MlaConfig,
    cache_seqs,
    batch_idx,
    cta_idx_q,
    cu_seqlens_q=None,
):
    """Return the KV domain shared by flat query rows in a CTA.

    The last physical row has the largest causal domain. Row-causal softmax
    applies the remaining row-specific mask; all other modes can use
    this CTA-visible length with the ordinary dense tail predicate.
    """
    if cutlass.const_expr(cfg.mask_type == MaskType.DENSE.value):
        return runtime_base_seq_len_kv(cfg, cache_seqs, batch_idx)
    _, _, logical_q_idx, _, _ = flat_query_row_state(
        Int32(cfg.tile_size_q - 1),
        cta_idx_q,
        cfg.tile_size_q,
        cfg.logical_num_heads_q,
        cfg.logical_seq_len_q,
        cu_seqlens_q,
        batch_idx,
    )
    return runtime_seq_len_kv_for_logical_q(
        cfg,
        cache_seqs,
        batch_idx,
        logical_q_idx,
        cu_seqlens_q,
    )


@cute.jit
def runtime_query_tile_is_active(
    cfg: MlaConfig,
    cu_seqlens_q,
    batch_idx,
    cta_idx_q,
):
    """Return whether a configured Q tile owns any runtime query rows."""

    query_is_active = cutlass.Boolean(True)
    if cutlass.const_expr(cu_seqlens_q is not None):
        query_is_active = runtime_flat_query_tile_has_rows(
            cta_idx_q,
            cfg.tile_size_q,
            cfg.logical_num_heads_q,
            cfg.logical_seq_len_q,
            cu_seqlens_q,
            batch_idx,
        )
    return query_is_active


@cute.jit
def runtime_split_tile_is_active(
    cfg: MlaConfig,
    cache_seqs,
    cu_seqlens_q,
    batch_idx,
    cta_idx_q,
    cta_idx_kv,
):
    """Return whether a configured split rank owns runtime KV work."""

    seq_len_kv = runtime_seq_len_kv_for_q(
        cfg,
        cache_seqs,
        batch_idx,
        cta_idx_q,
        cu_seqlens_q,
    )
    return Int32(cta_idx_kv) < runtime_num_ctas_kv(cfg, seq_len_kv)


@cute.jit
def runtime_work_tile_activity(
    cfg: MlaConfig,
    cache_seqs,
    cu_seqlens_q,
    batch_idx,
    cta_idx_q,
    cta_idx_kv,
):
    """Return the independent runtime Q and split-KV activity predicates."""

    query_is_active = runtime_query_tile_is_active(
        cfg,
        cu_seqlens_q,
        batch_idx,
        cta_idx_q,
    )
    split_is_active = runtime_split_tile_is_active(
        cfg,
        cache_seqs,
        cu_seqlens_q,
        batch_idx,
        cta_idx_q,
        cta_idx_kv,
    )
    return query_is_active, split_is_active


@cute.jit
def runtime_work_tile_is_active(
    cfg: MlaConfig,
    cache_seqs,
    cu_seqlens_q,
    batch_idx,
    cta_idx_q,
    cta_idx_kv,
):
    """Return whether a Q/split work tile owns any runtime data."""

    query_is_active, split_is_active = runtime_work_tile_activity(
        cfg,
        cache_seqs,
        cu_seqlens_q,
        batch_idx,
        cta_idx_q,
        cta_idx_kv,
    )
    return query_is_active and split_is_active


@cute.jit
def runtime_seq_len_kv_for_query_row(
    cfg: MlaConfig,
    cache_seqs,
    batch_idx,
    cta_idx_q,
    row_in_tile,
    cu_seqlens_q=None,
):
    """Return the KV length visible to one physical flat-query row."""
    _, _, logical_q_idx, _, _ = flat_query_row_state(
        row_in_tile,
        cta_idx_q,
        cfg.tile_size_q,
        cfg.logical_num_heads_q,
        cfg.logical_seq_len_q,
        cu_seqlens_q,
        batch_idx,
    )
    return runtime_seq_len_kv_for_logical_q(
        cfg,
        cache_seqs,
        batch_idx,
        logical_q_idx,
        cu_seqlens_q,
    )


@cute.jit
def runtime_seq_len_kv_from_task_cache(
    cfg: MlaConfig,
    task_cache,
    cta_idx_q,
    cu_seqlens_q=None,
    batch_idx=None,
):
    """Return the CTA-visible KV domain from the task-cached raw length."""
    seq_len_kv = Int32(task_cache[_TASK_CACHE_SEQ_LEN_KV])
    if cutlass.const_expr(cfg.mask_type == MaskType.DENSE.value):
        return seq_len_kv
    _, logical_seq_len_q = query_batch_bounds(
        cu_seqlens_q,
        batch_idx,
        cfg.logical_seq_len_q,
    )
    _, _, logical_q_idx, _, _ = flat_query_row_state(
        Int32(cfg.tile_size_q - 1),
        cta_idx_q,
        cfg.tile_size_q,
        cfg.logical_num_heads_q,
        cfg.logical_seq_len_q,
        cu_seqlens_q,
        batch_idx,
    )
    return mask_visible_k_length(
        cfg.mask_type,
        seq_len_kv,
        logical_q_idx,
        logical_seq_len_q,
    )


@cute.jit
def global_kv_tile_idx(
    cfg: MlaConfig,
    local_tile_idx,
    seq_len_kv,
    cta_idx_kv,
):
    """Map a local KV tile id to the global KV tile id for split-KV mode."""
    if cutlass.const_expr(cfg.use_multi_ctas_kv != 1):
        return local_tile_idx
    return Int32(cta_idx_kv) * runtime_local_kv_tiles(cfg, seq_len_kv) + local_tile_idx


@cute.jit
def attr_or_work_tile_idx(attr, stage_info: StageInfo, coord_idx: int):
    """Return an explicit attribute value or the matching work-tile coordinate."""
    if cutlass.const_expr(attr is None):
        return Int32(stage_info.work_tile.tile_idx[coord_idx])
    return Int32(attr)


@cute.jit
def batch_idx_for_stage(attr, stage_info: StageInfo):
    """Return the batch index for resources that do not pack heads into z."""
    return attr_or_work_tile_idx(attr, stage_info, 2)


@cute.jit
def batch_idx_for_stage_cfg(attr, cfg: MlaConfig, stage_info: StageInfo):
    """Return the batch index from a batch-major combined batch/head coordinate."""
    if cutlass.const_expr(attr is None):
        _cta_idx_q, _cta_idx_head_dim, batch_head_idx = stage_info.work_tile.tile_idx
        del _cta_idx_q, _cta_idx_head_dim
        batch_head_idx = Int32(batch_head_idx)
        return batch_head_idx // Int32(cfg.num_ctas_for_all_heads)
    return Int32(attr)


@cute.jit
def head_idx_for_stage(attr, cfg: MlaConfig, stage_info: StageInfo):
    """Return the base Q/head row from a batch-major combined batch/head tile."""
    if cutlass.const_expr(attr is None):
        _cta_idx_q, _cta_idx_head_dim, batch_head_idx = stage_info.work_tile.tile_idx
        del _cta_idx_q, _cta_idx_head_dim
        batch_head_idx = Int32(batch_head_idx)
        head_tile_idx = batch_head_idx % Int32(cfg.num_ctas_for_all_heads)
        return head_tile_idx * Int32(cfg.tile_size_q)
    return Int32(attr)


@cute.jit
def cta_idx_q_for_stage(attr, stage_info: StageInfo):
    """Return the Q CTA index for the current stage."""
    return attr_or_work_tile_idx(attr, stage_info, 0)


@cute.jit
def cta_idx_head_dim_v_for_stage(attr, stage_info: StageInfo):
    """Return the V head-dim CTA index for the current stage."""
    return attr_or_work_tile_idx(attr, stage_info, 1)


@cute.jit
def cta_idx_kv_for_stage(attr, stage_info: StageInfo):
    """Return the KV CTA index, defaulting to zero for non-split KV."""
    if cutlass.const_expr(attr is None):
        return Int32(0)
    return Int32(attr)


@cute.jit
def staged_kv_head_dim_call_idx(
    cfg: MlaConfig,
    stage_info: StageInfo,
    inst_id: int,
    is_v: int,
    *,
    stage_idx: cutlass.Constexpr[int],
    section: cutlass.Constexpr[MlaStage],
):
    """Return the head-dim stage index within a K or V staged load group."""
    del cfg, stage_info, inst_id, is_v, section
    return stage_idx
