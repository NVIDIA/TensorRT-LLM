# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Triton kernels of the Mamba cache managers.

They live apart from ``mamba_cache_manager`` so that importing them needs only
Triton: the JIT prefetch helper (``jit_prefetch_helper.py``) compiles them in a
process that does not import the rest of ``tensorrt_llm``. The launchers stay
in ``mamba_cache_manager``.
"""

import triton
import triton.language as tl


@triton.jit
def _promote_mamba_state_kernel(
    src_ptr,
    dst_ptr,
    src_idx_ptr,
    accepted_position_source_ptr,
    num_accepted_tokens_ptr,
    blk_ptr,
    replay_pnat_ptr,
    replay_cache_buf_idx_ptr,
    dummy_request_mask_ptr,
    num_gens,
    count,
    src_s_layer,
    src_s_row,
    src_s_step,
    dst_s_layer,
    dst_s_block,
    POSITION_SOURCE_IS_TOKEN_COUNT: tl.constexpr,
    REPLAY_STEP_WIDTH: tl.constexpr,
    REPLAY_HISTORY_SIZE: tl.constexpr,
    UPDATE_REPLAY_BOOKKEEPING: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # One program copies a BLOCK-sized tile of the contiguous inner state for a
    # single (layer, gen) pair. grid = (num_layers * num_gens, ceil(count/BLOCK)).
    pair = tl.program_id(0)
    tile = tl.program_id(1)
    layer = (pair // num_gens).to(tl.int64)
    g = pair % num_gens
    row = tl.load(src_idx_ptr + g).to(tl.int64)
    accepted_position = tl.load(accepted_position_source_ptr + g)
    if POSITION_SOURCE_IS_TOKEN_COUNT:
        accepted_position -= 1
    acc = accepted_position.to(tl.int64)
    blk = tl.load(blk_ptr + g).to(tl.int64)

    # The replay metadata update is batch-scoped, so layer 0's first copy tile
    # owns it. This folds PNAT and active-buffer maintenance into the existing
    # accepted-convolution-state promotion launch.
    if UPDATE_REPLAY_BOOKKEEPING:
        if (layer == 0) & (tile == 0):
            num_accepted_tokens = tl.load(num_accepted_tokens_ptr + g)
            is_dummy_request = tl.load(dummy_request_mask_ptr + g)
            pnat = tl.load(replay_pnat_ptr + blk)
            cache_buf_idx = tl.load(replay_cache_buf_idx_ptr + blk)
            wrote_checkpoint = pnat + REPLAY_STEP_WIDTH > REPLAY_HISTORY_SIZE
            next_pnat = tl.where(wrote_checkpoint, num_accepted_tokens, pnat + num_accepted_tokens)
            next_cache_buf_idx = tl.where(wrote_checkpoint, 1 - cache_buf_idx, cache_buf_idx)
            tl.store(replay_pnat_ptr + blk, next_pnat, mask=~is_dummy_request)
            tl.store(replay_cache_buf_idx_ptr + blk, next_cache_buf_idx, mask=~is_dummy_request)

    # int64 throughout: per-layer strides are O(1e8), so layer*stride overflows
    # int32 and would corrupt addresses / fault.
    src_base = layer * src_s_layer + row * src_s_row + acc * src_s_step
    dst_base = layer * dst_s_layer + blk * dst_s_block
    offs = tile * BLOCK + tl.arange(0, BLOCK)
    mask = offs < count
    v = tl.load(src_ptr + src_base + offs, mask=mask)
    tl.store(dst_ptr + dst_base + offs, v, mask=mask)
