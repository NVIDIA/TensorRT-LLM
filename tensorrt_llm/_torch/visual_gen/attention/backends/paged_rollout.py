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
"""Attention over a ``PagedRolloutKVCache``, shared by the backends that read paged K/V."""

import torch

from ....attention.backends.interface import PredefinedAttentionMask
from ...cache import ForwardPlan, PagedRolloutKVCache
from .interface import AttentionBackend


def forward_over_paged_rollout(
    backend: AttentionBackend,
    layer_idx: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    plan: ForwardPlan,
    attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.FULL,
) -> torch.Tensor:
    """Stage ``k``/``v`` in the plan's cache, then attend with ``backend.forward_paged_kv``.

    ``q``, ``k``, ``v`` are ``[1, S, H, D]`` / ``[1, S, H_kv, D]``: one video. The first
    ``plan.num_tokens`` rows are real; rows past them are padding, neither written nor
    attended, and their output rows are zero. Each causal block of the plan attends to the
    cache's fixed region, its window and the new tokens up to its end. Returns
    ``[1, S, H, D]``.
    """
    plan.validate()
    cache = plan.cache
    if not isinstance(cache, PagedRolloutKVCache):
        raise NotImplementedError(
            f"{type(backend).__name__} attends over a PagedRolloutKVCache only; got "
            f"{type(cache).__name__}"
        )
    if attention_mask != PredefinedAttentionMask.FULL:
        raise NotImplementedError("K/V cache attention is full attention over the cache.")
    batch, num_rows, num_heads, head_dim = q.shape
    if batch != 1 or k.shape[0] != 1 or k.shape[1] != num_rows or v.shape != k.shape:
        raise ValueError("K/V cache attention takes one video: q, k, v of [1, S, heads, head_dim].")
    if k.shape[2] != cache.num_kv_heads:
        raise ValueError(
            f"k has {k.shape[2]} heads but the cache holds {cache.num_kv_heads} per rank."
        )
    num_tokens = plan.num_tokens
    if num_tokens > num_rows:
        raise ValueError(
            f"the plan's {num_tokens} tokens exceed the {num_rows} rows: it counts the real "
            "tokens; rows past it are padding."
        )
    if not backend.support_kv_cache_page_size(cache.tokens_per_page, q.dtype):
        raise NotImplementedError(
            f"{type(backend).__name__} has no paged attention kernel for "
            f"{cache.tokens_per_page}-token pages of {q.dtype} on this GPU."
        )
    cache.write_staged(layer_idx, k[0, :num_tokens], v[0, :num_tokens], plan)
    size, num_blocks = plan.causal_block_size, plan.num_causal_blocks
    out = backend.forward_paged_kv(
        q[0, :num_tokens].view(num_blocks, size, num_heads, head_dim),
        cache.kv_buffer(layer_idx),
        cache.tables(plan),
    ).reshape(1, num_tokens, num_heads, head_dim)
    if num_tokens == num_rows:
        return out
    full = out.new_empty(1, num_rows, num_heads, head_dim)
    full[:, :num_tokens] = out
    full[:, num_tokens:].zero_()
    return full
