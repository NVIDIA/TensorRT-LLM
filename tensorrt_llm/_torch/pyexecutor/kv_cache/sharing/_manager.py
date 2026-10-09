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

from typing import TYPE_CHECKING, NamedTuple

import numpy as np

if TYPE_CHECKING:
    import torch

    from tensorrt_llm.runtime.kv_cache_manager_v2 import (
        AttentionLayerConfig,
        SsmLayerConfig,
        _KVCache,
    )

    from ...llm_request import LlmRequest
    from ..kv_cache_manager_v2 import KVCacheManagerV2

# The largest 32-bit count: the manager's window ranges compute in 32-bit token counts.
_MAX_TOKENS = 2**31 - 1


class CacheState(NamedTuple):
    """A request's runtime cache as the lender reads it: tokens committed, the history length its
    sliding windows keep, and whether it is active (not suspended)."""

    committed: int
    history: int
    active: bool


def require_v2(manager: object) -> None:
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


def kv_connector(manager: KVCacheManagerV2) -> object | None:
    """The manager's KV cache connector, or ``None`` without one."""
    return manager.kv_connector_manager


def context_moves_history(manager: KVCacheManagerV2) -> bool:
    """Whether the manager's context update moves a request's history to the end of each chunk,
    raising where that would move it back: under a block reuse policy other than all-reusable, or
    when the manager commits no blocks; otherwise the update only commits."""
    from ..kv_cache_manager_v2 import BlockReusePolicy

    if not commits_blocks(manager):
        return True
    return manager.block_reuse_policy != BlockReusePolicy.ALL_REUSABLE


def resize_ends_at_chunk(manager: KVCacheManagerV2) -> bool:
    """Whether the manager's context resize sets a cache's capacity from the chunk it runs, as a
    draft manager's does, so a chunk resumed below the history may leave the capacity below it and
    raise there."""
    return bool(manager.is_draft)


def keeps_passed_pages(manager: KVCacheManagerV2) -> bool:
    """Whether a block a window leaves behind before the manager commits it keeps its page, which
    the commit then stores whole under the block's key: the manager commits blocks without the
    minimal snapshot, as under the all-reusable policy."""
    return commits_blocks(manager) and not bool(manager.impl.commit_min_snapshot)


def prompt_lookahead(manager: KVCacheManagerV2) -> int:
    """Prompt tokens past a position the one-model draft the manager was built for reads (Eagle 1,
    vanilla MTP its draft length), wherever its layers live, also where reuse matches are not backed
    off by them; 0 for a draft reading none or whose read-ahead upstream has not established."""
    return max(int(manager._draft_prompt_lookahead), int(manager.reuse_match_backoff))


def attached(manager: KVCacheManagerV2) -> object | None:
    """The lender attached to ``manager``, or ``None``."""
    return getattr(manager, "_sharing", None)


def install(manager: KVCacheManagerV2, lender: object) -> None:
    """Attach ``lender`` for the manager's life: its free, in-place shrinks, reuse reset and
    shutdown then call the lender's ``_on_free``, ``_on_shrink``, ``_on_reset``, ``_on_shutdown``
    and, once the runtime's shutdown has returned, ``_on_caches_closed``."""
    manager._sharing = lender


def index_buffer(manager: KVCacheManagerV2) -> torch.Tensor:
    """The host buffer the manager's caches write their page indices into, through a raw pointer
    each keeps until it is detached or closed, whether or not the manager still exists."""
    return manager.host_kv_cache_block_offsets


def virtual_layers(manager: KVCacheManagerV2) -> tuple[dict[int, tuple[int, int]], int] | None:
    """Virtual layers (DeepSeek-V4): internal layer id -> (model layer, attention type value), and
    the number of attention types the enum defines. ``None`` for a manager without them."""
    virtual = getattr(manager, "_layer_attn_to_layer_id", None)
    if not virtual:
        return None
    inverse: dict[int, tuple[int, int]] = {}
    for (model_layer, attn_type), layer_id in virtual.items():
        inverse[int(layer_id)] = (int(model_layer), int(attn_type.value))
    # Every member of the enum counts, so stages holding different attention types agree.
    attn_type_class = type(next(iter(virtual))[1])
    return inverse, max(int(member.value) for member in attn_type_class) + 1


def stream(manager: KVCacheManagerV2) -> torch.cuda.Stream:
    """The manager's execution stream: a copy queued on it runs after the forward passes that wrote
    the pages and before later work on the stream, which a page's new owner waits for; a writer
    off the stream is not ordered after it."""
    return manager._stream


def kv_of(manager: KVCacheManagerV2, request_id: int) -> _KVCache | None:
    """The request's runtime cache, or ``None`` when it has none (never had one, freed, shut down).
    Caches are compared by identity to see a free or a replaced cache."""
    return manager.kv_cache_map.get(request_id)


def cache_state(kv: _KVCache) -> CacheState:
    """Committed tokens, history length and activity of runtime cache ``kv``, as plain values."""
    return CacheState(int(kv.num_committed_tokens), int(kv.history_length), bool(kv.is_active))


def scratch_reuse(kv: _KVCache) -> bool:
    """Whether runtime cache ``kv`` has SWA scratch reuse on: each capacity change then keeps its
    history within the scratch rewind of the old capacity, and its window blocks may sit in scratch
    slots, which the next chunk overwrites."""
    return bool(kv.enable_swa_scratch_reuse)


def pages(kv: _KVCache, layer_group: int) -> np.ndarray:
    """``int64`` page (pool-group slot) of each block ordinal of ``layer_group``, -1 where the block
    has no page."""
    return np.fromiter(
        kv.get_aggregated_page_indices(layer_group, valid_only=False), dtype=np.int64
    )


def locked_pages(kv: _KVCache, layer_group: int) -> np.ndarray:
    """``int64`` page of each block of an active cache from its locked base page indices, -1 where
    the block has none: a window block behind the history keeps at most a held page, which may be
    on another tier."""
    return np.array(kv.get_base_page_indices(layer_group)[: kv.num_blocks], dtype=np.int64)


def num_blocks(kv: _KVCache) -> int:
    """Block ordinals of runtime cache ``kv``: a shrink drops those past its capacity, with their
    pages."""
    return int(kv.num_blocks)


def holds_state(layer_config: AttentionLayerConfig | SsmLayerConfig) -> bool:
    """Whether a layer config of the runtime describes a recurrent-state layer."""
    from tensorrt_llm.runtime import kv_cache_manager_v2 as runtime

    return isinstance(layer_config, runtime.SsmLayerConfig)


def stale_blocks(manager: KVCacheManagerV2, layer_group: int, history: int) -> tuple[int, int]:
    """Block ordinals ``[beg, end)`` behind a windowed group's window at ``history``, sinks
    excepted. The range adds a block's tokens to the history in 32 bits, so a longer history is read
    as ``_MAX_TOKENS`` less one block, which lowers only ``end``."""
    # TODO: with a window near ``_MAX_TOKENS``, a history past that bound lowers ``end`` below
    # blocks the window has passed, which then count as inside it.
    longest = _MAX_TOKENS - int(manager.tokens_per_block)
    beg, end = manager._stale_block_range(layer_group, min(int(history), longest))
    return int(beg), int(end)


def keyed_by_placeholders(request: LlmRequest) -> bool:
    """Whether the request carries multimodal data that the manager keys by its placeholder tokens
    alone, as ``_augment_tokens_for_block_reuse`` does without a digest, position or length."""
    return request.py_multimodal_data is not None and (
        request.multimodal_hashes is None
        or request.multimodal_positions is None
        or request.multimodal_lengths is None
    )


def has_encoder_input(request: LlmRequest) -> bool:
    """Whether the request has encoder input, which its decoder's self-attention bytes depend on."""
    return request.try_get_encoder_output_len() is not None


def bidirectional_runs(request: LlmRequest) -> list[tuple[int, int]]:
    """The runs ``[b, e)`` of multimodal tokens the scheduler keeps within one context chunk: none
    unless the request's multimodal data sets ``mm_bidirectional_blocks`` and holds a
    ``multimodal_embed_mask_cumsum``, the gate the scheduler uses."""
    data = request.py_multimodal_data
    if not isinstance(data, dict) or not data.get("mm_bidirectional_blocks", False):
        return []
    cumsum = data.get("multimodal_embed_mask_cumsum")
    if cumsum is None:
        return []
    multimodal = np.diff(np.asarray(cumsum, dtype=np.int64), prepend=0) == 1
    edges = np.flatnonzero(np.diff(np.concatenate([[False], multimodal, [False]]).astype(np.int8)))
    return [(int(b), int(e)) for b, e in zip(edges[::2], edges[1::2])]


def returns_context_outputs(request: LlmRequest) -> bool:
    """Whether the request returns context logits, as prompt logprobs make it do, or asks for
    additional model outputs, which a model may give per context token."""
    return bool(request.py_return_context_logits) or bool(request.py_additional_outputs)


def prompt_length(request: LlmRequest) -> int:
    """The request's prompt length."""
    return int(request.prompt_len)


def block_keys(
    manager: KVCacheManagerV2, request: LlmRequest, kv: _KVCache, count: int
) -> list[bytes]:
    """The 32-byte reuse keys of the request's first ``count`` whole blocks, over the tokens
    and reuse scope the manager commits with. ``ValueError`` if the request has fewer."""
    # TODO: reuse keys miss a request's encoder input, multimodal embeddings without digests and
    # how the input processor treats a digested item, so requests that differ only there share keys.
    if count <= 0:
        return []
    from tensorrt_llm.runtime.kv_cache_manager_v2 import sequence_to_blockchain_keys

    tpb = int(manager.tokens_per_block)
    source = manager._reuse_token_source(request)
    need = count * tpb
    if len(source) < need:
        raise ValueError(
            f"request {request.py_request_id} has {len(source) // tpb} whole blocks of tokens, "
            f"{count} asked"
        )
    # Multimodal digests take the place of placeholder tokens, as when the manager commits.
    tokens = manager._augment_tokens_for_block_reuse(source, request, 0, need)
    if isinstance(tokens, np.ndarray):
        tokens = tokens.tolist()
    keys: list[bytes] = []
    for i, (block_tokens, key) in enumerate(
        sequence_to_blockchain_keys(tpb, kv.reuse_scope, tokens)
    ):
        if i == 0:
            continue  # the root: the reuse scope's own key
        if len(block_tokens) < tpb:
            break
        keys.append(bytes(key))
        if len(keys) == count:
            break
    if len(keys) < count:
        raise ValueError(
            f"request {request.py_request_id} has {len(keys)} whole blocks of tokens, {count} asked"
        )
    return keys


def grow(
    manager: KVCacheManagerV2, request: LlmRequest, kv: _KVCache, position: int, end: int
) -> bool:
    """Cover ``[0, end)`` and move the history to ``position`` with the manager's own resize, then
    run its fresh-page fill (a diagnostic, off unless set) on the pages the resize added alone.
    ``False``, the cache unchanged, when pages run out."""
    before = _pages_by_pool(manager, kv)
    if not manager._resize_for_connector_prefix(request, kv, position, end):
        return False
    if before is not None:
        # The fill takes as fresh the pages its record lacks, and a joint draft pool's own resizes
        # keep no record: the pages before the grow become it, so the fill leaves them alone.
        manager._fresh_pages_filled[int(request.py_request_id)] = before
        manager._fill_fresh_kv_pages(request.py_request_id)
    return True


def _pages_by_pool(manager: KVCacheManagerV2, kv: _KVCache) -> dict[int, np.ndarray] | None:
    """Runtime cache ``kv``'s base page of each block, per pool, as the fresh-page fill records
    them; ``None`` with the fill off."""
    if manager._fresh_page_fill is None:
        return None
    count = int(kv.num_blocks)
    return {
        pool: np.frombuffer(kv.get_base_page_indices(pool), dtype=np.int32, count=count).copy()
        for pool in range(int(manager.num_pools))
    }
