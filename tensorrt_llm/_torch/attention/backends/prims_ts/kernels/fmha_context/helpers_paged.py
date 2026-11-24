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

Context has no multi-CTA-KV split, folds any sliding-window prefix into
``kv_tile_start``, and uses one ``kv_tile_start + loop_offset`` expression for
the current K/V tile index. Page IDs come from fixed row-strided block tables.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Int32

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
def _load_block_table_row_bounds(
    block_table_row_stride: Int32,
    cfg: FmhaConfig,
    seq_len_kv: Int32,
    batch_coord: Int32,
) -> tuple[Int32, Int32]:
    """Return one fixed-table row base and runtime-valid inclusive page bound."""

    row_begin = batch_coord * block_table_row_stride
    return row_begin, _runtime_last_valid_page_idx(cfg, seq_len_kv)
