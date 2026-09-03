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

"""Paged-KV helpers for the task-scheduled FMHA *context* kernel.

Mirrors the decode-side helpers in
``../fmha_decode/fmha_decode_resources/helpers_kv_tile_idx.py`` but is stripped
down to what context needs: context has no multi-CTA-KV split, folds any
sliding-window prefix into ``kv_tile_start``, and uses one
``kv_tile_start + loop_offset`` expression for the current K/V tile index.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Int32

from cutlass.experimental.task_scheduling.resources import StageInfo

from .fmha_resources import FmhaConfig


@cute.jit
def _load_runtime_seq_len_kv(
    seq_lens_kv: cute.Pointer | None,
    max_seq_len_kv: Int32 | int,
    batch_coord: Int32,
) -> Int32:
    # Context resolves (head, batch, seq) directly from the work tile, so the
    # batch coord is always known by the caller — no persistent-vs-launch
    # fallback indirection required.
    if cutlass.const_expr(seq_lens_kv is None):
        return Int32(max_seq_len_kv)
    return Int32(seq_lens_kv[batch_coord])


@cute.jit
def _runtime_last_valid_page_idx(cfg: FmhaConfig, seq_len_kv: Int32) -> Int32:
    num_pages = cute.ceil_div(seq_len_kv, cfg.num_tokens_per_page)
    return cute.math.max(num_pages - Int32(1), Int32(0))


@cute.jit
def _load_paged_request_bounds(
    paged_kv_indptr: cute.Pointer,
    cfg: FmhaConfig,
    seq_len_kv: Int32,
    batch_coord: Int32,
) -> tuple[Int32, Int32]:
    """Load one CSR row base and its runtime-valid inclusive page bound."""
    request_begin = Int32(paged_kv_indptr[batch_coord])
    request_end = Int32(paged_kv_indptr[batch_coord + Int32(1)])
    row_page_idx_ub = cute.math.max(request_end - request_begin - Int32(1), Int32(0))
    page_idx_ub = cute.math.min(
        row_page_idx_ub,
        _runtime_last_valid_page_idx(cfg, seq_len_kv),
    )
    return request_begin, page_idx_ub


@cute.jit
def _resolve_kv_tile_idx_context(
    stage_info: StageInfo,
    kv_tile_start: Int32,
    tile_offset: cutlass.Constexpr[int] = 0,
) -> Int32:
    # Context's current K/V tile index is kv_tile_start + loop_offset, with an
    # optional compile-time shift for the staged D>128 schedule's previous V.
    # No multi-CTA-KV transform; sliding-window prefix skipping is already
    # folded into kv_tile_start by GmemQKVResource.
    return kv_tile_start + Int32(stage_info.loop_offset) + Int32(tile_offset)
