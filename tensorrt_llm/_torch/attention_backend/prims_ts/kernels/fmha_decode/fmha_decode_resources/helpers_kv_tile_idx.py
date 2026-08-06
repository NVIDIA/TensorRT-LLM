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

"""KV tile-index resolution helpers for FMHA decode TS resources.

Handles seq-len lookup, sliding-window prefix skipping, and
split-KV / static-vs-runtime tile-index math used by ``SmemPageOffsetsKvResource``,
``SmemKvResource``, and the softmax / correction consumers.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Int32

from cutlass.experimental.task_scheduling.resources import StageInfo

from ...mask import kv_tile_is_fully_visible
from ..fmha_decode_config import CAUSAL, FmhaDecodeConfig
from .helpers_common import (
    _TASK_CACHE_SEQ_LEN_KV,
    _decode_gen_task_cache,
    _logical_head_batch,
)


@cute.jit
def _load_runtime_seq_len_kv(
    seq_lens_kv: cute.Pointer | None,
    max_seq_len_kv: Int32 | int,
    stage_info: StageInfo,
    fallback_h_k_idx: Int32,
    fallback_b_idx: Int32,
) -> Int32:
    """Load runtime KV length for the logical batch, or return static max."""
    # Persistent scheduling carries logical (head, batch) in the work tile;
    # non-persistent kernels use the launch-time fallback coordinates.
    _, logical_b_idx = _logical_head_batch(stage_info, fallback_h_k_idx, fallback_b_idx)
    if cutlass.const_expr(seq_lens_kv is None):
        return Int32(max_seq_len_kv)
    if cutlass.const_expr(stage_info.task_cache is not None):
        return Int32(_decode_gen_task_cache(stage_info)[_TASK_CACHE_SEQ_LEN_KV])
    return Int32(seq_lens_kv[logical_b_idx])


@cute.jit
def _runtime_total_kv_tiles(
    cfg: FmhaDecodeConfig,
    seq_len_kv: Int32,
    seq_len_q: Int32,
    q_token_base: Int32,
) -> Int32:
    """Return runtime KV tiles spanning this CTA's causal/window union."""
    seq_len_kv = _runtime_effective_seq_len_kv(cfg, seq_len_kv, seq_len_q, q_token_base)
    return cute.ceil_div(seq_len_kv, cfg.tile_size_kv)


@cute.jit
def _runtime_configured_local_kv_tiles(
    cfg: FmhaDecodeConfig,
    seq_len_kv: Int32,
    seq_len_q: Int32,
    q_token_base: Int32,
) -> Int32:
    """Return the instruction-aligned local span for configured split capacity."""
    total_kv_tiles = _runtime_total_kv_tiles(
        cfg,
        seq_len_kv,
        seq_len_q,
        q_token_base,
    )
    num_insts_kv = Int32(cfg.num_insts_kv)
    configured_splits = Int32(cfg.splits_kv if cfg.use_split_kv else 1)
    tiles_per_group = configured_splits * num_insts_kv
    num_groups = (total_kv_tiles + tiles_per_group - Int32(1)) // tiles_per_group
    return cute.math.max(num_groups * num_insts_kv, num_insts_kv)


@cute.jit
def _sliding_window_start_idx(
    cfg: FmhaDecodeConfig,
    seq_len_kv: Int32,
    seq_len_q: Int32,
    q_token_base: Int32,
) -> Int32:
    """Return the earliest sliding-window token needed by one logical Q CTA."""
    if cutlass.const_expr(not cfg.use_sliding_window_causal):
        return Int32(0)
    return cute.math.max(
        seq_len_kv
        - seq_len_q
        + q_token_base
        + Int32(1)
        - Int32(cfg.attention_window_size),
        Int32(0),
    )


@cute.jit
def _num_skipped_kv_tiles(
    cfg: FmhaDecodeConfig,
    seq_len_kv: Int32,
    seq_len_q: Int32,
    q_token_base: Int32,
) -> Int32:
    """Return full leading KV tiles skipped by the runtime sliding window."""
    return _sliding_window_start_idx(cfg, seq_len_kv, seq_len_q, q_token_base) // Int32(
        cfg.tile_size_kv
    )


@cute.jit
def _runtime_effective_seq_len_kv(
    cfg: FmhaDecodeConfig,
    seq_len_kv: Int32,
    seq_len_q: Int32,
    q_token_base: Int32,
) -> Int32:
    """Return the CTA-visible K span after causal/window trimming.

    The left endpoint is rounded down to a complete K tile so the partial
    boundary remains available for row-exact masking. The right endpoint is
    the last valid Q row owned by this CTA, which is the union bound needed by
    grouped multi-token Q tiles.
    """
    skipped_tokens = _num_skipped_kv_tiles(
        cfg, seq_len_kv, seq_len_q, q_token_base
    ) * Int32(cfg.tile_size_kv)
    visible_k_end = seq_len_kv
    if cutlass.const_expr(cfg.mask_type == CAUSAL):
        q_token_end = cute.math.min(
            q_token_base + Int32(cfg.q_tokens_per_cta),
            seq_len_q,
        )
        visible_k_end = cute.math.min(
            cute.math.max(
                seq_len_kv - seq_len_q + q_token_end,
                Int32(0),
            ),
            seq_len_kv,
        )
    return cute.math.max(visible_k_end - skipped_tokens, Int32(0))


@cute.jit
def _kv_tile_is_fully_unmasked_for_q_group(
    cfg: FmhaDecodeConfig,
    tile_offset_k: Int32,
    seq_len_kv: Int32,
    seq_len_q: Int32,
    q_token_base: Int32,
    tile_has_valid_scores: cutlass.Boolean,
) -> cutlass.Boolean:
    """Return whether a KV tile needs no score mask for any active Q row.

    Grouped causal rows expose an intersection bounded on the right by the
    earliest Q token.  A sliding window also bounds it on the left by the
    latest active Q token.  Tiles outside that intersection retain the exact
    per-row boundary path.
    """
    visible_k_begin = Int32(0)
    visible_k_end = seq_len_kv
    if cutlass.const_expr(cfg.mask_type == CAUSAL):
        visible_k_end = cute.math.min(
            cute.math.max(
                seq_len_kv - seq_len_q + q_token_base + Int32(1),
                Int32(0),
            ),
            seq_len_kv,
        )
        if cutlass.const_expr(cfg.use_sliding_window_causal):
            q_token_end = cute.math.min(
                q_token_base + Int32(cfg.q_tokens_per_cta),
                seq_len_q,
            )
            latest_causal_end = cute.math.min(
                cute.math.max(
                    seq_len_kv - seq_len_q + q_token_end,
                    Int32(0),
                ),
                seq_len_kv,
            )
            visible_k_begin = cute.math.max(
                latest_causal_end - Int32(cfg.attention_window_size),
                Int32(0),
            )
    return tile_has_valid_scores and kv_tile_is_fully_visible(
        tile_offset_k,
        Int32(cfg.tile_size_kv),
        visible_k_begin,
        visible_k_end,
    )


@cute.jit
def _runtime_active_splits_kv(
    cfg: FmhaDecodeConfig,
    seq_len_kv: Int32,
    seq_len_q: Int32,
    q_token_base: Int32,
) -> Int32:
    """Return the useful split prefix for one runtime Q/KV work item."""
    if cutlass.const_expr(not cfg.use_split_kv):
        return Int32(1)
    total_kv_tiles = _runtime_total_kv_tiles(
        cfg,
        seq_len_kv,
        seq_len_q,
        q_token_base,
    )
    local_kv_tiles = _runtime_configured_local_kv_tiles(
        cfg,
        seq_len_kv,
        seq_len_q,
        q_token_base,
    )
    # Split ranges are instruction-group aligned. Only ranges intersecting the
    # valid K domain participate; remaining configured grid slots are padding.
    return (total_kv_tiles + local_kv_tiles - Int32(1)) // local_kv_tiles


@cute.jit
def _runtime_execution_splits_kv(
    cfg: FmhaDecodeConfig,
    seq_len_kv: Int32,
    seq_len_q: Int32,
    q_token_base: Int32,
) -> Int32:
    """Return a nonzero producer/reduction fanout for a runtime-valid Q tile."""
    return cute.math.max(
        _runtime_active_splits_kv(cfg, seq_len_kv, seq_len_q, q_token_base),
        Int32(1),
    )


@cute.jit
def _runtime_last_valid_tile_idx(
    cfg: FmhaDecodeConfig,
    seq_len_kv: Int32,
    seq_len_q: Int32,
    q_token_base: Int32,
) -> Int32:
    """Return the last valid runtime KV tile index."""
    return cute.math.max(
        _runtime_total_kv_tiles(cfg, seq_len_kv, seq_len_q, q_token_base) - Int32(1),
        Int32(0),
    )


@cute.jit
def _runtime_clamp_valid_tile_idx(
    cfg: FmhaDecodeConfig,
    tile_idx: Int32,
    seq_len_kv: Int32,
    seq_len_q: Int32,
    q_token_base: Int32,
) -> Int32:
    """Clamp a runtime KV tile index to the last valid tile."""
    return cute.math.min(
        tile_idx,
        _runtime_last_valid_tile_idx(cfg, seq_len_kv, seq_len_q, q_token_base),
    )


@cute.jit
def _runtime_last_valid_page_idx(cfg: FmhaDecodeConfig, seq_len_kv: Int32) -> Int32:
    """Return the last valid page index for a runtime KV length."""
    num_pages = cute.ceil_div(seq_len_kv, cfg.num_tokens_per_page)
    return cute.math.max(num_pages - Int32(1), Int32(0))


@cute.jit
def _static_split_kv_global_tile_idx(
    cfg: FmhaDecodeConfig, stage_info: StageInfo, local_tile_idx: Int32
) -> Int32:
    """Map a static local tile index to a global split-KV tile index."""
    if cutlass.const_expr(not cfg.use_split_kv):
        return local_tile_idx
    return (
        _logical_cta_kv_idx(cfg, stage_info) * Int32(cfg.static_local_kv_tiles)
        + local_tile_idx
    )


@cute.jit
def _logical_cta_kv_idx(cfg: FmhaDecodeConfig, stage_info: StageInfo) -> Int32:
    """Resolve the split index from the Q-group-major launch coordinate."""
    if cutlass.const_expr(stage_info.work_tile is not None):
        q_group_cta_idx = Int32(stage_info.work_tile.tile_idx[0])
    else:
        q_group_cta_idx, _, _ = cute.arch.block_idx()
    if cutlass.const_expr(cfg.use_split_kv):
        return q_group_cta_idx % Int32(cfg.splits_kv)
    return q_group_cta_idx


@cute.jit
def _runtime_local_kv_tiles(
    cfg: FmhaDecodeConfig,
    seq_len_kv: Int32,
    seq_len_q: Int32,
    q_token_base: Int32,
) -> Int32:
    """Return local KV tiles assigned to one split CTA at runtime."""
    return _runtime_configured_local_kv_tiles(
        cfg,
        seq_len_kv,
        seq_len_q,
        q_token_base,
    )


@cute.jit
def _runtime_split_kv_global_tile_idx(
    cfg: FmhaDecodeConfig,
    stage_info: StageInfo,
    local_tile_idx: Int32,
    seq_len_kv: Int32,
    seq_len_q: Int32,
    q_token_base: Int32,
) -> Int32:
    """Map a runtime local tile index to a global split-KV tile index."""
    if cutlass.const_expr(not cfg.use_split_kv):
        return local_tile_idx
    return (
        _logical_cta_kv_idx(cfg, stage_info)
        * _runtime_local_kv_tiles(cfg, seq_len_kv, seq_len_q, q_token_base)
        + local_tile_idx
    )
