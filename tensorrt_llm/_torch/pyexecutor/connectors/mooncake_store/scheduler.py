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
"""Leader side of the Mooncake store KV cache connector.

Runs on rank 0 under TP, or each attention-DP owner, deciding what to load
and save while workers move the pages. That needs per-request bookkeeping:

* a hash chain, since `KVCacheManagerV2` reports `RequestData.block_hashes`
  empty and a block has no content identity otherwise;
* the page slot index per block ordinal, accumulated across iterations. The
  manager reports only newly allocated indices each step, but a block is
  allocated before it is full and is savable only once it is full.
"""

from typing import Dict, List, Optional, Tuple

from tensorrt_llm.bindings.internal.batch_manager import LlmRequest
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime.kv_cache_manager_v2 import BAD_PAGE_INDEX

from ..kv_cache_connector import (
    AllocationFence,
    KvCacheConnectorScheduler,
    RequestData,
    SchedulerOutput,
)
from .config import MooncakeStoreConnectorConfig
from .keys import BlockHashChain, ReuseScope
from .metadata import MooncakeStoreMetadata, PageTransfer, RequestTransfers
from .validation import validate_llm_args, validate_speculative_config
from .worker import MooncakeStoreConnectorWorker, resolve_local_worker

__all__ = ["MooncakeStoreConnectorScheduler"]


def _multimodal_item_digests(request: LlmRequest) -> Optional[Tuple[bytes, ...]]:
    """Content digest per multimodal item of *request*, in prompt order.

    Returns:
        An empty tuple for a text-only request, one digest per item for a
        multimodal one, and None when the request carries media the digests do
        not describe. The placeholder tokens of such a request say nothing
        about its content, so no key names its pages without naming someone
        else's; the caller leaves it alone.
    """
    # The same three fields `KVCacheManagerV2` reads to decide whether a
    # request is multimodal, so local reuse and the store agree.
    hashes = request.multimodal_hashes
    if hashes:
        # 8 int32 chunks per item, most significant byte first, matching
        # `_hash_to_digest` in kv_cache_manager_v2 and C++ `getNthByte`.
        return tuple(
            b"".join(int(chunk).to_bytes(4, "big", signed=True) for chunk in item_hash)
            for item_hash in hashes
        )
    if request.multimodal_positions or request.multimodal_lengths:
        return None
    return ()


def _reuse_scope(request: LlmRequest) -> Optional[ReuseScope]:
    """The scope *request*'s block hashes belong in, or None to leave it alone.

    None for a request whose pages no key can name without naming someone
    else's. The caller gives such a request neither a lookup nor a save.
    """
    if request.lora_task_id is not None:
        logger.warning_once(
            "mooncake-store is bypassing requests that carry a LoRA adapter: an "
            "adapter id is assigned by whoever submitted the request, so the "
            "same id names different weights on two servers sharing a pool and "
            "a page keyed by it would be served against the wrong weights.",
            key="mooncake-store-lora",
        )
        return None
    digests = _multimodal_item_digests(request)
    if digests is None:
        logger.warning_once(
            "mooncake-store is bypassing requests whose multimodal content "
            "carries no hashes: their prompt tokens do not identify the media "
            "behind them, so a stored page could be served to a different "
            "request.",
            key="mooncake-store-unhashed-multimodal",
        )
        return None
    return ReuseScope(cache_salt=request.cache_salt, multimodal_digests=digests)


class _RequestState:
    """Per-request bookkeeping that has to outlive a single iteration."""

    __slots__ = (
        "chain",
        "tokens",
        "pages",
        "saved_upto",
        "load_first_block",
        "load_blocks",
        "emitted_saves",
        "load_async",
        "async_loaded_upto",
    )

    def __init__(self, chain: BlockHashChain):
        self.chain = chain
        #: The request's tokens, accumulated from the per-step deltas.
        self.tokens: List[int] = []
        #: Page slot index per block ordinal, per layer group.
        self.pages: Dict[int, List[int]] = {}
        #: First block ordinal not yet considered for saving.
        self.saved_upto = 0
        #: The offer made by `get_num_new_matched_tokens`, in block ordinals.
        self.load_first_block = 0
        self.load_blocks = 0
        self.emitted_saves = False
        self.load_async = False
        self.async_loaded_upto = 0


class MooncakeStoreConnectorScheduler(KvCacheConnectorScheduler):
    """Chooses which pages the Mooncake pool serves and which it receives."""

    supports_attention_dp = True
    supports_speculative_prefill = True

    def __init__(self, llm_args: TorchLlmArgs):
        super().__init__(llm_args)

        validate_llm_args(llm_args)
        self._config = MooncakeStoreConnectorConfig.resolve(llm_args)
        self._tokens_per_block = int(llm_args.kv_cache_config.tokens_per_block)
        self._prompt_lookahead = validate_speculative_config(llm_args, self._config)
        self._async_load = self._config.async_load and self._config.role.loads
        self._pending_async_loads: List[RequestTransfers] = []
        self.supports_scheduler_async_load = self._async_load and (
            getattr(llm_args, "enable_attention_dp", False) or llm_args.tensor_parallel_size == 1
        )
        self.scheduler_async_load_capacity = self._config.async_load_workers
        self._requests: Dict[int, _RequestState] = {}
        self._worker: Optional[MooncakeStoreConnectorWorker] = None

        logger.warning(
            f"mooncake-store leader ready (role={self._config.role.value}, "
            f"tokens_per_block={self._tokens_per_block})"
        )

    def wait_for_initialization(self):
        """Bind to the process-local worker, which owns the store handle."""
        self._worker = resolve_local_worker()

    # ---- lookup ----

    def get_num_new_matched_tokens(
        self, request: LlmRequest, num_computed_tokens: int
    ) -> Tuple[int, bool]:
        """Offer the longest stored prefix beyond what the device already has.

        Args:
            request: The request being scheduled.
            num_computed_tokens: Tokens already matched in the local KV cache.

        Returns:
            Tokens the store can supply, and whether loading is asynchronous.
        """
        if self._config.capacity_only:
            # Neither a lookup nor a save will read the hash chain this would
            # otherwise build, which costs a SHA-256 over the whole prompt of
            # every request.
            return 0, False

        scope = _reuse_scope(request)
        if scope is None:
            # Leaving it without a `_RequestState` suppresses its saves too,
            # since `build_connector_meta` skips requests it has no state for.
            return 0, False

        tokens = request.get_tokens(0)
        state = self._state_for(request, tokens, scope)
        state.load_first_block = 0
        state.load_blocks = 0
        state.load_async = False

        if not self._config.role.loads:
            return 0, False

        # A partial local match means the boundary block is half computed on
        # device. Overwriting it with a stored page would discard tokens the
        # runtime already counted, so only whole-block offers are made.
        if num_computed_tokens % self._tokens_per_block:
            return 0, False

        first_block = num_computed_tokens // self._tokens_per_block
        # Stop one token short of the prompt: the runtime still has to run a
        # forward pass for this request, and it cannot do that with nothing left
        # to compute.
        last_block = (len(tokens) - 1) // self._tokens_per_block
        candidates = state.chain.hashes[first_block:last_block]
        if not candidates:
            return 0, False

        hit_blocks = self._require_worker().count_prefix_hit(candidates)
        if hit_blocks == 0:
            return 0, False

        state.load_first_block = first_block
        state.load_blocks = hit_blocks
        state.load_async = self._async_load
        logger.debug(
            f"mooncake-store matched {hit_blocks} blocks "
            f"({hit_blocks * self._tokens_per_block} tokens) "
            f"for request {request.request_id}"
        )
        return hit_blocks * self._tokens_per_block, state.load_async

    def update_state_after_alloc(self, request: LlmRequest, block_ids: List[int]):
        self.update_state_after_alloc_by_layer_group(request, [block_ids])

    def update_state_after_alloc_by_layer_group(
        self, request: LlmRequest, block_ids_by_layer_group: List[List[int]]
    ) -> None:
        """Queue the honoured async prefix against its complete destination allocation."""
        state = self._requests.get(request.request_id)
        if state is None or not state.load_blocks:
            return
        honoured = int(request.py_num_connector_matched_tokens)
        blocks, remainder = divmod(honoured, self._tokens_per_block)
        if remainder or not 0 <= blocks <= state.load_blocks:
            raise RuntimeError(
                f"mooncake-store: request {request.request_id} honoured {honoured} "
                f"tokens outside its {state.load_blocks}-block offer"
            )
        state.load_blocks = blocks
        if not state.load_async:
            return
        transfers = RequestTransfers(request.request_id)
        limit = min(
            len(state.chain.hashes),
            min((len(indices) for indices in block_ids_by_layer_group), default=0),
        )
        load_end = state.load_first_block + blocks
        for block in range(state.load_first_block, load_end):
            if block >= limit:
                raise RuntimeError("mooncake-store: honoured prefix has missing destination pages")
            for group, indices in enumerate(block_ids_by_layer_group):
                index = int(indices[block])
                if index == BAD_PAGE_INDEX:
                    raise RuntimeError("mooncake-store: honoured prefix has an invalid destination")
                transfers.pages.append(PageTransfer(state.chain.hashes[block], group, index))
        state.async_loaded_upto = load_end
        state.load_blocks = 0
        # An empty honoured offer still releases the request parked by the runtime.
        self._pending_async_loads.append(transfers)

    def start_scheduler_async_loads(self, allocation_fence: AllocationFence) -> None:
        """Start owner-local loads admitted before a forward batch exists."""
        if not self.supports_scheduler_async_load:
            raise RuntimeError("mooncake-store scheduler async loading is not enabled")
        pending, self._pending_async_loads = self._pending_async_loads, []
        if pending:
            self._require_worker().start_async_loads(pending, allocation_fence)

    # ---- work lists ----

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> MooncakeStoreMetadata:
        """Turn this iteration's scheduled requests into load and save lists."""
        metadata = MooncakeStoreMetadata(async_loads=self._pending_async_loads)
        self._pending_async_loads = []
        for request_data in (*scheduler_output.new_requests, *scheduler_output.cached_requests):
            state = self._requests.get(request_data.request_id)
            if state is None:
                # Only requests that went through get_num_new_matched_tokens
                # have a hash chain.
                continue

            state.tokens.extend(request_data.new_tokens)
            state.chain.extend(state.tokens)
            self._record_pages(state, request_data)

            num_loaded_tokens = state.load_blocks * self._tokens_per_block
            loads = self._loads_for(state, request_data)
            if loads.pages:
                metadata.loads.append(loads)

            # Whatever the store just supplied, and whatever the local cache
            # matched, is not ours to write back: the store already has the
            # former, and the latter was never allocated during this run.
            state.saved_upto = max(
                state.saved_upto,
                state.load_first_block + state.load_blocks,
                state.async_loaded_upto,
            )
            # An offer is consumed once. The load is issued in exactly the
            # iteration the runtime allocated pages to hold it.
            state.load_blocks = 0

            if self._config.role.saves:
                saves = self._saves_for(state, request_data, num_loaded_tokens)
                if saves.pages:
                    state.emitted_saves = True
                    metadata.saves.append(saves)
        return metadata

    def request_finished(self, request: LlmRequest, cache_block_ids: List[int]) -> bool:
        """Report whether pages must stay pinned for in-flight saves.

        Whether a write has landed is known per rank and this runs only on the
        leader, so the answer is "any save was emitted" rather than "a save is
        in flight". A request whose writes already retired is reported done by
        the next `get_finished`, costing one iteration; the exact answer would
        cost a per-rank report back to the leader.

        Returns:
            True when this request handed any page to the background save
            thread, so freeing its pages now could let a later request
            overwrite bytes mid-transfer.
        """
        state = self._requests.pop(request.request_id, None)
        return bool(state is not None and state.emitted_saves)

    def request_reset(self, request: LlmRequest) -> None:
        """Forget a request whose pages were released for it to run again.

        `pages` and `saved_upto` describe the allocation that was just freed,
        and the lookup preceding the next attempt rebuilds the rest.
        """
        self._requests.pop(request.request_id, None)
        self._pending_async_loads = [
            entry for entry in self._pending_async_loads if entry.request_id != request.request_id
        ]

    # ---- internals ----

    def _require_worker(self) -> MooncakeStoreConnectorWorker:
        if self._worker is None:
            self._worker = resolve_local_worker()
        return self._worker

    def _state_for(
        self, request: LlmRequest, tokens: List[int], scope: ReuseScope
    ) -> _RequestState:
        state = self._requests.get(request.request_id)
        if state is None:
            state = _RequestState(
                BlockHashChain(
                    self._tokens_per_block, scope=scope, prompt_lookahead=self._prompt_lookahead
                )
            )
            self._requests[request.request_id] = state
        # The lookup runs before the request is scheduled, so the chain has to
        # be ready before any scheduler output exists.
        state.chain.extend(tokens)
        return state

    def _record_pages(self, state: _RequestState, request_data: RequestData) -> None:
        """Append this step's newly allocated page indices, by block ordinal."""
        by_group = request_data.new_block_ids_by_layer_group
        if not by_group:
            # The flat `new_block_ids` the manager also fills does not say
            # which layer group its indices belong to, so it is unusable here.
            return
        for layer_group_id, indices in enumerate(by_group):
            state.pages.setdefault(layer_group_id, []).extend(int(index) for index in indices)

    def _addressable_blocks(self, state: _RequestState) -> int:
        """Block ordinals that are both hashed and backed by a page everywhere."""
        if not state.pages:
            return 0
        return min(len(state.chain.hashes), min(len(indices) for indices in state.pages.values()))

    def _loads_for(self, state: _RequestState, request_data: RequestData) -> RequestTransfers:
        transfers = RequestTransfers(request_data.request_id)
        limit = self._addressable_blocks(state)
        for offset in range(state.load_blocks):
            block = state.load_first_block + offset
            if block >= limit:
                raise RuntimeError("mooncake-store: honoured prefix has missing destination pages")
            before = len(transfers.pages)
            self._append_pages(state, transfers, block)
            if len(transfers.pages) - before != len(state.pages):
                raise RuntimeError("mooncake-store: honoured prefix has an invalid destination")
        return transfers

    def _saves_for(
        self, state: _RequestState, request_data: RequestData, num_loaded_tokens: int
    ) -> RequestTransfers:
        transfers = RequestTransfers(request_data.request_id)
        # Allocation can run ahead of a trimmed context chunk or MTP lookahead.
        computed_end = (
            request_data.computed_position + num_loaded_tokens + request_data.num_scheduled_tokens
        )
        limit = min(self._addressable_blocks(state), computed_end // self._tokens_per_block)
        for block in range(state.saved_upto, limit):
            self._append_pages(state, transfers, block)
        state.saved_upto = max(state.saved_upto, limit)
        return transfers

    def _append_pages(self, state: _RequestState, transfers: RequestTransfers, block: int) -> None:
        """Add one block's page from every layer group, or none of them."""
        block_hash = state.chain.hashes[block]
        pages: List[PageTransfer] = []
        for layer_group_id, indices in state.pages.items():
            page_index = indices[block]
            if page_index == BAD_PAGE_INDEX:
                # A sliding window dropped the block in this group, and a
                # partial block is not a usable cache entry.
                return
            pages.append(PageTransfer(block_hash, layer_group_id, page_index))
        transfers.pages.extend(pages)
