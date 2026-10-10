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
"""Every read or write of manager and runtime internals the lender makes, one function each, so
moving the lender into the manager or changing a member touches this file alone. Imports of the
manager and the runtime stay inside the functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, NamedTuple, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    import torch

    from ...llm_request import LlmRequest
    from ..kv_cache_manager_v2 import KVCacheManagerV2


class CacheState(NamedTuple):
    """A request's runtime cache as the lender reads it: tokens committed, the history length its
    sliding windows keep, and whether it is active (not suspended)."""

    committed: int
    history: int
    active: bool


def require_v2(manager) -> None:
    """Raise ``TypeError`` unless ``manager`` is a ``KVCacheManagerV2`` (subclasses included)."""
    from ..kv_cache_manager_v2 import KVCacheManagerV2

    if not isinstance(manager, KVCacheManagerV2):
        raise TypeError(
            f"the KV cache lender needs a KVCacheManagerV2, got {type(manager).__name__}"
        )


def commits_blocks(manager: KVCacheManagerV2) -> bool:
    """Whether the manager commits blocks to its prefix-reuse tree: block reuse on and, for a draft
    manager, joint reuse with its target."""
    return bool(manager.enable_block_reuse) and bool(manager._can_publish_block_reuse)


def attached(manager: KVCacheManagerV2) -> Optional[object]:
    """The lender attached to ``manager``, or ``None``."""
    return getattr(manager, "_sharing", None)


def install(manager: KVCacheManagerV2, lender: object) -> None:
    """Attach ``lender`` for the manager's life: its free, in-place shrinks and shutdown then call
    the lender's ``_on_free``, ``_on_shrink`` and ``_on_shutdown``."""
    manager._sharing = lender


def index_buffer(manager: KVCacheManagerV2) -> torch.Tensor:
    """The host buffer the manager's caches write their page indices into, through a raw pointer
    each keeps until it is detached or closed, whether or not the manager still exists."""
    return manager.host_kv_cache_block_offsets


def virtual_layers(manager: KVCacheManagerV2) -> Optional[Tuple[Dict[int, Tuple[int, int]], int]]:
    """Virtual layers (DeepSeek-V4): internal layer id -> (model layer, attention type value), and
    the number of attention types the enum defines. ``None`` for a manager without them."""
    virtual = getattr(manager, "_layer_attn_to_layer_id", None)
    if not virtual:
        return None
    inverse: Dict[int, Tuple[int, int]] = {}
    for (model_layer, attn_type), layer_id in virtual.items():
        inverse[int(layer_id)] = (int(model_layer), int(attn_type.value))
    # Every member of the enum counts, so stages holding different attention types agree.
    attn_type_class = type(next(iter(virtual))[1])
    return inverse, max(int(member.value) for member in attn_type_class) + 1


def stream(manager: KVCacheManagerV2) -> torch.cuda.Stream:
    """The manager's execution stream: a copy queued on it runs after the forward passes that wrote
    the pages and before the pages' next writer."""
    return manager._stream


def kv_of(manager: KVCacheManagerV2, request_id: int) -> Optional[object]:
    """The request's runtime cache, or ``None`` when it has none (never had one, freed, shut down).
    Caches are compared by identity to see a free or a replaced cache."""
    return manager.kv_cache_map.get(request_id)


def cache_state(kv) -> CacheState:
    """Committed tokens, history length and activity of runtime cache ``kv``, as plain values."""
    return CacheState(int(kv.num_committed_tokens), int(kv.history_length), bool(kv.is_active))


def scratch_reuse(kv) -> bool:
    """Whether runtime cache ``kv`` has SWA scratch reuse on: its window blocks may sit in scratch
    slots, which the next chunk overwrites."""
    return bool(kv.enable_swa_scratch_reuse)


def pages(kv, layer_group: int) -> np.ndarray:
    """``int64`` page (pool-group slot) of each block ordinal of ``layer_group``, -1 where the block
    has no page."""
    return np.fromiter(
        kv.get_aggregated_page_indices(layer_group, valid_only=False), dtype=np.int64
    )


def locked_pages(kv, layer_group: int) -> np.ndarray:
    """``int64`` page of each block of an active cache from its locked base page indices, -1 where
    the block has none: a window block behind the history only holds its page, which may be on
    another tier."""
    return np.array(kv.get_base_page_indices(layer_group)[: kv.num_blocks], dtype=np.int64)


def num_blocks(kv) -> int:
    """Block ordinals of runtime cache ``kv``: a shrink drops those past its capacity, with their
    pages."""
    return int(kv.num_blocks)


def stale_blocks(manager: KVCacheManagerV2, layer_group: int, history: int) -> Tuple[int, int]:
    """Block ordinals ``[beg, end)`` behind a windowed layer group's window at ``history``, sinks
    excepted. Called only for layer groups with a window."""
    beg, end = manager._stale_block_range(layer_group, history)
    return int(beg), int(end)


def block_keys(manager: KVCacheManagerV2, request: LlmRequest, kv, num_blocks: int) -> List[bytes]:
    """The 32-byte reuse keys of the request's first ``num_blocks`` whole blocks, over the tokens
    and reuse scope the manager commits with. ``ValueError`` if the request has fewer."""
    if num_blocks <= 0:
        return []
    from tensorrt_llm.runtime.kv_cache_manager_v2 import sequence_to_blockchain_keys

    tpb = int(manager.tokens_per_block)
    source = manager._reuse_token_source(request)
    need = num_blocks * tpb
    if len(source) < need:
        raise ValueError(
            f"request {request.py_request_id} has {len(source) // tpb} whole blocks of tokens, "
            f"{num_blocks} asked"
        )
    # Multimodal digests take the place of placeholder tokens, as when the manager commits.
    tokens = manager._augment_tokens_for_block_reuse(source, request, 0, need)
    if isinstance(tokens, np.ndarray):
        tokens = tokens.tolist()
    keys: List[bytes] = []
    for i, (block_tokens, key) in enumerate(
        sequence_to_blockchain_keys(tpb, kv.reuse_scope, tokens)
    ):
        if i == 0:
            continue  # the root: the reuse scope's own key
        if len(block_tokens) < tpb:
            break
        keys.append(bytes(key))
        if len(keys) == num_blocks:
            break
    if len(keys) < num_blocks:
        raise ValueError(
            f"request {request.py_request_id} has {len(keys)} whole blocks of tokens, "
            f"{num_blocks} asked"
        )
    return keys


def grow(manager: KVCacheManagerV2, request: LlmRequest, kv, position: int, end: int) -> bool:
    """Cover ``[0, end)`` and move the history to ``position`` with the manager's own resize, then
    run its fresh-page fill (a diagnostic, off unless set), as every resize of the manager does.
    ``False``, the cache unchanged, when pages run out."""
    if not manager._resize_for_connector_prefix(request, kv, position, end):
        return False
    # The fill marks the new pages before a fetch lands in them; a later resize then sees them as
    # the request's own and leaves the fetched blocks alone.
    manager._fill_fresh_kv_pages(request.py_request_id)
    return True


def close_cache(kv) -> None:
    """Close runtime cache ``kv``: its pages go back to the pools."""
    kv.close()
