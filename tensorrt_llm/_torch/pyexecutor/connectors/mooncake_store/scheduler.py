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

Runs only on rank 0. It decides what to load and what to save; the workers do
the moving. Two pieces of bookkeeping make that possible, and both exist because
``KVCacheManagerV2`` reports ``RequestData.block_hashes`` empty:

* a hash chain per request, so a block has a content identity at all;
* the page slot index per block ordinal, accumulated across iterations. The
  manager reports only *newly allocated* indices each step, but a block is
  allocated before it is full and is only savable once it is full, so the index
  has to be remembered from the step that reported it.
"""

from typing import Dict, List, Optional, Tuple

from tensorrt_llm.bindings.internal.batch_manager import LlmRequest
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime.kv_cache_manager_v2 import BAD_PAGE_INDEX

from ..kv_cache_connector import KvCacheConnectorScheduler, RequestData, SchedulerOutput
from .config import MooncakeStoreConnectorConfig
from .keys import BlockHashChain
from .metadata import MooncakeStoreMetadata, PageTransfer, RequestTransfers
from .validation import validate_llm_args
from .worker import MooncakeStoreConnectorWorker, resolve_local_worker

__all__ = ["MooncakeStoreConnectorScheduler"]


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
    )

    def __init__(self, chain: BlockHashChain):
        self.chain = chain
        #: The request's tokens, accumulated from the per-step deltas.
        self.tokens: List[int] = []
        #: Page slot index per block ordinal, per layer group.
        self.pages: Dict[int, List[int]] = {}
        #: First block ordinal not yet considered for saving.
        self.saved_upto = 0
        #: The offer made by ``get_num_new_matched_tokens``, in block ordinals.
        self.load_first_block = 0
        self.load_blocks = 0
        self.emitted_saves = False


class MooncakeStoreConnectorScheduler(KvCacheConnectorScheduler):
    """Chooses which pages the Mooncake pool serves and which it receives."""

    def __init__(self, llm_args: TorchLlmArgs):
        super().__init__(llm_args)

        validate_llm_args(llm_args)
        self._config = MooncakeStoreConnectorConfig.from_env()
        self._tokens_per_block = int(llm_args.kv_cache_config.tokens_per_block)
        self._requests: Dict[int, _RequestState] = {}
        self._worker: Optional[MooncakeStoreConnectorWorker] = None

        logger.info(
            "mooncake-store leader ready (role=%s, tokens_per_block=%d)",
            self._config.role.value,
            self._tokens_per_block,
        )

    def wait_for_initialization(self):
        """Bind to the process-local worker, which owns the store handle.

        Called after the executor has built both halves and registered the KV
        cache layout, which is what the worker needs before it can name a key.
        """
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
            Tokens the store can supply, and ``False`` for a synchronous load.
        """
        tokens = request.get_tokens(0)
        state = self._state_for(request, tokens)
        state.load_first_block = 0
        state.load_blocks = 0

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
        logger.debug(
            "mooncake-store matched %d blocks (%d tokens) for request %d",
            hit_blocks,
            hit_blocks * self._tokens_per_block,
            request.request_id,
        )
        return hit_blocks * self._tokens_per_block, False

    def cancel_load(self, request: LlmRequest, start: int, end: int):
        """Drop offered blocks whose tokens the runtime will not consume.

        Loads here are synchronous and nothing has been transferred yet, so this
        is exact: the offer is truncated before ``build_connector_meta`` turns it
        into work.
        """
        state = self._requests.get(request.request_id)
        if state is None or state.load_blocks == 0:
            return
        kept = 0
        for offset in range(state.load_blocks):
            block = state.load_first_block + offset
            block_start = block * self._tokens_per_block
            if block_start + self._tokens_per_block > start and block_start < end:
                break
            kept += 1
        state.load_blocks = kept

    def update_state_after_alloc(self, request: LlmRequest, block_ids: List[int]):
        """No-op: page indices are read from the scheduler output instead.

        The flat ``block_ids`` here are a single space, but a V2 page index is
        scoped to a layer group. ``RequestData.new_block_ids_by_layer_group`` is
        the form that stays correct for every model, so that is the only source
        this connector uses.
        """

    # ---- work lists ----

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> MooncakeStoreMetadata:
        """Turn this iteration's scheduled requests into load and save lists."""
        metadata = MooncakeStoreMetadata()
        for request_data in (*scheduler_output.new_requests, *scheduler_output.cached_requests):
            state = self._requests.get(request_data.request_id)
            if state is None:
                # Only requests that went through get_num_new_matched_tokens have
                # a hash chain. Generation-only requests never do, and the
                # connector manager refuses them outright.
                continue

            state.tokens.extend(request_data.new_tokens)
            state.chain.extend(state.tokens)
            self._record_pages(state, request_data)

            loads = self._loads_for(state, request_data)
            if loads.pages:
                metadata.loads.append(loads)

            # Whatever the store just supplied, and whatever the local cache
            # matched, is not ours to write back: the store already has the
            # former, and the latter was never allocated during this run.
            state.saved_upto = max(state.saved_upto, state.load_first_block + state.load_blocks)
            # An offer is consumed once. The load is issued in exactly the
            # iteration the runtime allocated pages to hold it.
            state.load_blocks = 0

            if self._config.role.saves:
                saves = self._saves_for(state, request_data)
                if saves.pages:
                    state.emitted_saves = True
                    metadata.saves.append(saves)
        return metadata

    def request_finished(self, request: LlmRequest, cache_block_ids: List[int]) -> bool:
        """Report whether pages must stay pinned for in-flight saves.

        Returns:
            True when this request handed any page to the background save
            thread. Its pages are the source of those RDMA reads, so freeing
            them now would let a later request overwrite bytes mid-transfer.
        """
        state = self._requests.pop(request.request_id, None)
        return bool(state is not None and state.emitted_saves)

    # ---- internals ----

    def _require_worker(self) -> MooncakeStoreConnectorWorker:
        if self._worker is None:
            self._worker = resolve_local_worker()
        return self._worker

    def _state_for(self, request: LlmRequest, tokens: List[int]) -> _RequestState:
        state = self._requests.get(request.request_id)
        if state is None:
            state = _RequestState(
                BlockHashChain(self._tokens_per_block, cache_salt=request.cache_salt)
            )
            self._requests[request.request_id] = state
        # Hashing the prompt here rather than waiting for the first scheduler
        # output is the whole point: the lookup happens before the request is
        # scheduled, so the chain has to be ready before any metadata exists.
        state.chain.extend(tokens)
        return state

    def _record_pages(self, state: _RequestState, request_data: RequestData) -> None:
        """Append this step's newly allocated page indices, by block ordinal."""
        by_group = request_data.new_block_ids_by_layer_group
        if not by_group:
            # Under a single layer group the manager also mirrors that group's
            # indices into the flat ``new_block_ids``, but it does not say which
            # group they belong to, so there is nothing safe to record from it.
            return
        for layer_group_id, indices in by_group.items():
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
                # The runtime allocated fewer pages than it accepted tokens for.
                # It reports the shortfall through cancel_load; until then the
                # unaddressable tail is simply not loaded.
                break
            self._append_pages(state, transfers, block)
        return transfers

    def _saves_for(self, state: _RequestState, request_data: RequestData) -> RequestTransfers:
        transfers = RequestTransfers(request_data.request_id)
        limit = self._addressable_blocks(state)
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
                # The block has no page in this group -- a sliding window has
                # already dropped it. A partial page is not a usable cache entry,
                # so the whole block is skipped.
                return
            pages.append(PageTransfer(block_hash, layer_group_id, page_index))
        transfers.pages.extend(pages)
