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
"""Facade for V2 operations, runtime metadata and lifecycle hooks; heavy imports stay local.
Layout also reads structural manager fields directly."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Sequence

import numpy as np

if TYPE_CHECKING:
    import torch

    from tensorrt_llm.mapping import Mapping
    from tensorrt_llm.runtime.kv_cache_manager_v2 import _KVCache

    from ...llm_request import LlmRequest
    from .._lender_hooks import _ManagerHooks
    from ..kv_cache_manager_v2 import KVCacheManagerV2

# The largest 32-bit count: the manager's window ranges compute in 32-bit token counts.
_MAX_TOKENS = 2**31 - 1
# ``BlockReusePolicy.ALL_REUSABLE``, imported at the first call that needs it.
_all_reusable = None


class CacheState(NamedTuple):
    """Committed tokens, retained history and activity (not suspended) of a runtime cache."""

    committed: int
    history: int
    active: bool


def mapping(manager: KVCacheManagerV2) -> Mapping:
    """The manager's mapping; ``TypeError`` for a manager without one."""
    value = getattr(manager, "mapping", None)
    if value is None:
        raise TypeError("the KV cache lender needs a manager with a mapping")
    return value


def has_lossy_cold_page_codec(manager: KVCacheManagerV2) -> bool:
    """Whether the configured cold-page codec changes the original KV bytes."""
    return bool(getattr(manager, "_has_lossy_cold_page_codec", False))


def require_v2(manager: object) -> None:
    """Raise ``TypeError`` unless ``manager`` is a ``KVCacheManagerV2`` (subclasses included)."""
    from tensorrt_llm.bindings.internal.batch_manager import CacheType

    from ..kv_cache_manager_v2 import KVCacheManagerV2

    if not isinstance(manager, KVCacheManagerV2):
        raise TypeError(
            f"the KV cache lender needs a KVCacheManagerV2, got {type(manager).__name__}"
        )
    if getattr(manager, "kv_cache_type", None) == CacheType.CROSS:
        raise ValueError(
            "the KV cache lender does not support CROSS caches or encoder-decoder models"
        )


def model_has_recurrent_state(manager: KVCacheManagerV2) -> bool:
    """Whether the manager serves a model with recurrent state, on every pipeline stage, whichever
    layers this stage holds (``CacheLayoutAdapter.model_has_recurrent_state``)."""
    from ..cache_layout.adapter import adapter_for

    return bool(adapter_for(manager).model_has_recurrent_state)


def model_has_attention_window(manager: KVCacheManagerV2) -> bool:
    """Whether any pipeline stage retains a configured or backend-defined window."""
    pattern = manager._model_attention_window_pattern
    return manager._model_has_attention_window or any(
        0 < window < manager.max_seq_len for window in pattern[: manager.num_layers]
    )


def model_has_sparse_buffers(manager: KVCacheManagerV2) -> bool:
    """Whether any pipeline stage holds sparse buffers."""
    return manager._model_has_sparse_buffers


def commits_blocks(manager: KVCacheManagerV2) -> bool:
    """Whether the manager commits blocks to its prefix-reuse tree: block reuse on and, for a draft
    manager, joint reuse with its target."""
    return bool(manager.enable_block_reuse) and bool(manager._can_publish_block_reuse)


def connector_loads(manager: KVCacheManagerV2) -> bool:
    """Whether a connector loads KV; capacity-only connectors transfer nothing."""
    connector = manager.kv_connector_manager
    return connector is not None and not connector.capacity_only


def context_moves_history(manager: KVCacheManagerV2) -> bool:
    """Whether the context update moves the history to each chunk's end, raising where that moves it
    back: when no blocks commit or the reuse policy is not all-reusable; else it only commits."""
    if not commits_blocks(manager):
        return True
    global _all_reusable
    if _all_reusable is None:
        from ..kv_cache_manager_v2 import BlockReusePolicy

        _all_reusable = BlockReusePolicy.ALL_REUSABLE
    return manager.block_reuse_policy != _all_reusable


def resize_ends_at_chunk(manager: KVCacheManagerV2) -> bool:
    """Whether the context resize sets capacity from the chunk it runs, as a draft manager's does,
    so a chunk resumed below the history may leave the capacity below it and raise."""
    return bool(manager.is_draft)


def keeps_stale_pages(manager: KVCacheManagerV2) -> bool:
    """Whether a block a window leaves behind before its commit keeps its page, which the commit
    then stores whole under the block's key: the manager commits blocks without the minimal
    snapshot, as under the all-reusable policy."""
    return commits_blocks(manager) and not bool(manager.impl.commit_min_snapshot)


def prompt_lookahead(manager: KVCacheManagerV2) -> int:
    """Prompt tokens past a position the manager's one-model draft reads (Eagle 1, vanilla MTP its
    draft length), wherever its layers live, also where reuse matches are not backed off by them; 0
    for a draft reading none or whose read-ahead upstream has not established."""
    return max(int(manager.draft_prompt_lookahead), int(manager.reuse_match_backoff))


def attached_lender(manager: KVCacheManagerV2) -> object | None:
    """The lender attached to ``manager``, or ``None``."""
    return getattr(manager, "_sharing", None)


def install_lender(manager: KVCacheManagerV2, lender: _ManagerHooks) -> None:
    """Install ``lender``'s hooks (``_lender_hooks._ManagerHooks``) for the manager's lifetime;
    ``TypeError`` unless it implements them all."""
    from .._lender_hooks import _ManagerHooks

    hooks = (
        "_on_free",
        "_on_shrink",
        "_on_resume",
        "_parking_row",
        "_parked_row",
        "_before_reset",
        "_on_reset",
        "_on_shutdown",
        "_on_caches_closed",
        "_is_locked",
        "_defer_close",
        "_keeps_pools",
        "_history_cap",
        "_page_lock_state",
    )
    if not isinstance(lender, _ManagerHooks) or any(
        not callable(getattr(lender, name, None)) for name in hooks
    ):
        raise TypeError("the KV cache lender must implement all manager hooks")
    manager._sharing = lender


def layer_group_pool_indices(manager: KVCacheManagerV2) -> list[int]:
    """The pool-group index of each runtime layer group, in layer-group order."""
    return [int(value) for value in manager.impl.get_life_cycle_pool_group_indices()]


def index_buffer(manager: KVCacheManagerV2) -> torch.Tensor:
    """The host buffer the manager's caches write their page indices into, through a raw pointer
    each keeps until it is detached or closed, whether or not the manager still exists."""
    return manager.host_kv_cache_block_offsets


def index_row(manager: KVCacheManagerV2, request_id: int) -> int | None:
    """The row of ``index_buffer``, ``[layer group, row, 0, block]``, into which the request's
    active cache writes its beam-0 page indices itself at each page change, on the thread making it;
    ``None`` where the manager released the row before the free, to the next request it admits."""
    if request_id in manager._early_freed_index_requests:
        return None
    return int(manager.index_mapper.get_index(request_id)) * int(manager.max_beam_width)


def runtime(manager: KVCacheManagerV2) -> object:
    """Runtime owning the device pools and freeing them on destruction."""
    return manager.impl


def context_computed(request: LlmRequest) -> int:
    """The prompt tokens the request's context steps computed: its context position, which the
    executor moves past each context step once it ran."""
    return int(request.context_current_position)


def connector_load_pending(manager: KVCacheManagerV2, request: LlmRequest) -> bool:
    """Whether the connector has unfinished off-stream prefix writes into ``request``."""
    connector = manager.kv_connector_manager
    return connector is not None and bool(connector.has_pending_load(request))


def connector_reserved(manager: KVCacheManagerV2, request: LlmRequest) -> bool:
    """Whether the connector reserved a prefix already in history but not yet loading."""
    connector = manager.kv_connector_manager
    return connector is not None and connector.get_prefix_reservation(request) is not None


def connector_state(manager: KVCacheManagerV2, request: LlmRequest) -> tuple[bool, bool]:
    """``(connector_reserved, connector_load_pending)`` of the request in one call."""
    connector = manager.kv_connector_manager
    if connector is None:
        return False, False
    return (
        connector.get_prefix_reservation(request) is not None,
        bool(connector.has_pending_load(request)),
    )


def served_prefix_end(request: LlmRequest) -> int:
    """End of the prefix served by a connector or a fetch the request resumed from."""
    return int(request.py_served_prefix_end)


def stream(manager: KVCacheManagerV2) -> torch.cuda.Stream:
    """Execution stream ordering copies after forward writes and before later work, which new page
    owners wait for; writers on other streams need explicit ordering. A cache still on the
    allocation side stream is ordered behind ``allocation_fence`` instead, until adopted."""
    return manager._stream


def cache_stream(manager: KVCacheManagerV2, request_id: int) -> torch.cuda.Stream | None:
    """The stream that orders operations on a request's cache during a named grow."""
    import torch

    cache_stream_for = getattr(manager, "_cache_stream", None)
    if cache_stream_for is not None:
        return torch.cuda.ExternalStream(
            cache_stream_for(request_id), device=manager._stream.device
        )
    return getattr(manager, "_stream", None)


def allocation_fence(manager: KVCacheManagerV2, request_id: int) -> torch.cuda.Event | None:
    """The manager's event after the allocation work queued for the request's cache while that
    cache is still on the allocation side stream; ``None`` once the execution stream adopted it."""
    # A cache off the side stream, the common case, costs one set lookup.
    if request_id not in manager._on_allocation_stream:
        return None
    return manager._allocation_fence(request_id)


def kv_of(manager: KVCacheManagerV2, request_id: int) -> _KVCache | None:
    """Request cache, or ``None`` when absent/freed/shut down; compare identity to detect replacement."""
    return manager.kv_cache_map.get(request_id)


_new_tuple = tuple.__new__


def cache_state(kv: _KVCache) -> CacheState:
    """Committed tokens, history length and activity of runtime cache ``kv``, as plain values."""
    # Built as the tuple it is, without the NamedTuple's Python-level constructor.
    return _new_tuple(
        CacheState, (int(kv.num_committed_tokens), int(kv.history_length), bool(kv.is_active))
    )


def cache_extent(kv: _KVCache) -> tuple[CacheState, int]:
    """``(cache_state(kv), num_blocks(kv))`` in one call."""
    return _new_tuple(
        CacheState, (int(kv.num_committed_tokens), int(kv.history_length), bool(kv.is_active))
    ), int(kv.num_blocks)


def has_scratch_reuse(kv: _KVCache) -> bool:
    """Whether ``kv`` has SWA scratch reuse on: capacity changes keep its history within the old
    capacity's scratch rewind, and its window blocks may sit in slots the next chunk overwrites."""
    return bool(kv.enable_swa_scratch_reuse)


def pages(kv: _KVCache, layer_group: int) -> np.ndarray:
    """Read-only view of the page (pool-group slot) each block of an active cache locks, -1 for
    none, read on the manager's thread, valid until the cache next changes: copy what must outlive
    it. A window block behind the history, its held page maybe on another tier, counts as none."""
    return np.asarray(kv.get_base_page_indices(layer_group))[: kv.num_blocks]


def page_list(kv: _KVCache, layer_group: int) -> list[int]:
    """The page each block of ``layer_group`` holds, -1 where none, as a list: for an active cache
    ``pages``'s, except at window blocks behind the history, whose held pages it gives though it
    does not lock them. For few blocks cheaper than that view."""
    return kv.get_aggregated_page_indices(layer_group)


def held_pages(kv: _KVCache, layer_group: int, *, recurrent: bool = False) -> int:
    """Distinct ordinary, active STATE or scratch slots held by this group, across beams."""
    slots = set()
    for beam in range(int(kv.beam_width)):
        if recurrent:
            slots.add(int(kv.get_ssm_block_base_index(layer_group, beam_id=beam)))
        else:
            slots.update(kv.get_aggregated_page_indices(layer_group, beam_id=beam, valid_only=True))
    scratch = kv.get_scratch_desc(layer_group)
    if scratch is not None:
        slots.update(scratch.slot_ids)
    slots.discard(-1)
    return len(slots)


def state_slot(kv: _KVCache, layer_group: int) -> int:
    """Active cache's in-place recurrent-state slot for this group, or -1."""
    return int(kv.get_ssm_block_base_index(layer_group))


def prepared_state_slots(manager: KVCacheManagerV2) -> tuple[int, dict[int, int]] | None:
    """The layer group of the manager's recurrent state and its record of each request's slot,
    written from the cache at each preparation of its resources and dropped at its free, left stale
    by a resume that moves the state until the next; ``None`` without one."""
    layer_group = getattr(manager, "ssm_layer_group_id", None)
    record = getattr(manager, "_request_id_to_state_index", None)
    if layer_group is None or not isinstance(record, dict):
        return None
    return int(layer_group), record


def locked_pages(kv: _KVCache, layer_group: int) -> np.ndarray:
    """The same read-only view as ``pages``."""
    return np.asarray(kv.get_base_page_indices(layer_group))[: kv.num_blocks]


def committed_by_group(kv: _KVCache, num_layer_groups: int) -> tuple[int, ...]:
    """Count a claimed prefix only in groups whose reuse status is complete; a group also counts
    tokens committed past its claim endpoint, OPTIONAL/PRIVATE groups included."""
    committed = int(kv.num_committed_tokens)
    if committed == 0:
        return (0,) * num_layer_groups
    statuses = kv.reuse_status  # one per layer group, indexed by its id
    return tuple(
        committed if statuses[lg].complete or committed > int(statuses[lg].endpoint) else 0
        for lg in range(num_layer_groups)
    )


def committed_blocks(
    kv: _KVCache, num_layer_groups: int, committed: Sequence[int] | None = None
) -> tuple[int, ...]:
    """Per layer group, its whole committed blocks (``committed_by_group``, rounded down);
    ``committed``: ``committed_by_group``'s tokens, if read."""
    tpb = int(kv.tokens_per_block)
    if committed is None:
        committed = committed_by_group(kv, num_layer_groups)
    return tuple(tokens // tpb for tokens in committed)


def close_cache(kv: _KVCache) -> None:
    """Close the cache, returning its pages to the manager's pools."""
    kv.close()


def commit_held_back(manager: KVCacheManagerV2, kv: _KVCache) -> None:
    """Retry a deferred commit; remaining overlapping locks defer it again."""
    manager._commit_held_back(kv)


def has_held_commit(manager: KVCacheManagerV2, kv: _KVCache) -> bool:
    """Whether the manager holds back a commit of ``kv`` for ``commit_held_back`` to retry; true
    for a manager that keeps no record of the commits it holds back."""
    held = getattr(manager, "_held_commits", None)
    return held is None or id(kv) in held


def drop_held_commit(manager: KVCacheManagerV2, kv: _KVCache, start: int) -> None:
    """Drop a commit held back for ``kv`` that would commit computed tokens from ``start`` on, which
    a fetch from ``start`` may overwrite; nothing where none is held back."""
    if getattr(manager, "_held_commits", None):
        manager._drop_held_commit(kv, start)


def is_closed(kv: _KVCache) -> bool:
    """Whether runtime cache ``kv`` has closed, after which it writes no page index anywhere."""
    return kv.status == kv.Status.CLOSED


def num_blocks(kv: _KVCache) -> int:
    """Block ordinals of ``kv``: a shrink drops those past its capacity, with their pages."""
    return int(kv.num_blocks)


def stale_range(manager: KVCacheManagerV2, layer_group: int, history: int) -> tuple[int, int]:
    """Block ordinals ``[beg, end)`` behind a windowed group's window at ``history``, sinks aside;
    in 32 bits a longer history reads as ``_MAX_TOKENS`` less a block, lowering only ``end``."""
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
    """Runs ``[b, e)`` of multimodal tokens the scheduler keeps in one context chunk, by its gate:
    none without ``mm_bidirectional_blocks`` set and a ``multimodal_embed_mask_cumsum``."""
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


def prompt_length(
    request: LlmRequest, manager: KVCacheManagerV2 | None = None, *, cp_size: int = 1
) -> int:
    """Prompt tokens, using whole-rank ``total_input_len_cp`` under Helix."""
    if cp_size > 1 or (manager is not None and helix(manager)[1] > 1):
        return int(request.total_input_len_cp)
    return int(request.prompt_len)


def context_end(request: LlmRequest, manager: KVCacheManagerV2 | None = None) -> int:
    """Where this worker's context ends, at most the prompt, or all of it under ``manager``'s Helix;
    a request leaving the prompt's tail to another worker reports the tail's start."""
    prompt = prompt_length(request, manager)
    if manager is not None and helix(manager)[1] > 1:
        return prompt
    end = int(request.context_current_position) + int(request.context_remaining_length)
    return min(prompt, end)


def helix(manager: KVCacheManagerV2) -> tuple[int, int]:
    """Helix ``(rank, size)``, or ``(0, 1)``: block ``b`` spans ``size`` pages of the manager's
    ``tokens_per_block`` tokens, rank ``j``'s from ``(b * size + j) * tokens_per_block``."""
    if not getattr(manager, "_has_cp_helix", False):
        return 0, 1
    return int(manager._helix_cp_rank), int(manager._helix_cp_size)


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
    run its fresh-page fill (a diagnostic, off unless set) on the pages the resize added alone;
    ``False``, the cache unchanged, when pages run out."""
    from .._lender_hooks import fill_fresh_pages_past, fresh_page_record

    before = fresh_page_record(manager, kv)
    if not manager._resize_for_served_prefix(request, kv, position, end):
        return False
    # A joint draft pool's own resizes keep no fill record: the pages before the grow stand for
    # it, so the fill leaves them alone.
    fill_fresh_pages_past(manager, request.py_request_id, before)
    return True
