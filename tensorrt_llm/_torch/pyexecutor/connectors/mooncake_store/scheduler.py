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
"""Scheduler adapter side of the Mooncake store KV cache connector.

Runs on rank 0 under tensor parallelism and on every owner under attention data
parallelism, deciding what to load and what to save for its own requests while
the workers do the moving. Each adapter asks the worker in its own process to
look prefixes up. That needs two pieces of per-request bookkeeping:

* a hash chain, since `KVCacheManagerV2` reports `RequestData.block_hashes`
  empty and a block has no content identity otherwise;
* the page slot index per block ordinal, accumulated across iterations. The
  manager reports only newly allocated indices each step, but a block is
  allocated before it is full and is savable only once it is full.

Prompt hashes and pages both exist before the KV they name is computed, so a
block is published only by the pass that computes its last token, and it
counts as stored only once the worker's `wait_for_save` queued that pass.
"""

from typing import Dict, List, Optional, Tuple

from tensorrt_llm.bindings.internal.batch_manager import LlmRequest
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime.kv_cache_manager_v2 import BAD_PAGE_INDEX

from ..kv_cache_connector import KvCacheConnectorScheduler, RequestData, SchedulerOutput
from .config import MooncakeStoreConnectorConfig
from .keys import BlockHashChain, ReuseScope
from .metadata import MooncakeStoreMetadata, PageTransfer, RequestTransfers
from .validation import validate_llm_args
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
    )

    def __init__(self, chain: BlockHashChain):
        self.chain = chain
        #: The request's tokens, accumulated from the per-step deltas.
        self.tokens: List[int] = []
        #: Page slot index per block ordinal, per layer group.
        self.pages: Dict[int, List[int]] = {}
        #: First block ordinal not yet handed to the store. Blocks a pass bound
        #: for saving count only once the worker queued them; see
        #: `MooncakeStoreConnectorScheduler._confirm_submitted_saves`.
        self.saved_upto = 0
        #: The offer made by `get_num_new_matched_tokens`, in block ordinals.
        self.load_first_block = 0
        self.load_blocks = 0
        self.emitted_saves = False


class MooncakeStoreConnectorScheduler(KvCacheConnectorScheduler):
    """Chooses which pages the Mooncake pool serves and which it receives."""

    #: Request state is owner-local and lookups go to the worker in this
    #: process, whose attention-DP keys every owner shares.
    supports_attention_dp = True

    def __init__(self, llm_args: TorchLlmArgs):
        super().__init__(llm_args)

        validate_llm_args(llm_args)
        self._config = MooncakeStoreConnectorConfig.resolve(llm_args)
        self._tokens_per_block = int(llm_args.kv_cache_config.tokens_per_block)
        self._requests: Dict[int, _RequestState] = {}
        self._worker: Optional[MooncakeStoreConnectorWorker] = None
        # Saves bound in the last metadata, by request: the block ordinal they
        # end at. Settled by `_confirm_submitted_saves` at the next build.
        self._unconfirmed_saves: Dict[int, int] = {}

        logger.warning(
            f"mooncake-store scheduler adapter ready (role={self._config.role.value}, "
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
            Tokens the store can supply, and `False` for a synchronous load.
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
            f"mooncake-store matched {hit_blocks} blocks "
            f"({hit_blocks * self._tokens_per_block} tokens) "
            f"for request {request.request_id}"
        )
        return hit_blocks * self._tokens_per_block, False

    def update_state_after_alloc(self, request: LlmRequest, block_ids: List[int]):
        """Trim the outstanding offer to what the runtime honoured.

        Page indices still come from the scheduler output: the flat `block_ids`
        here are a single space, but a V2 page index is scoped to a layer
        group, so this connector reads only
        `RequestData.new_block_ids_by_layer_group`. `KVCacheManagerV2` reports
        through `update_state_after_alloc_by_layer_group`, whose base form
        folds the single layer group this connector accepts into this call.

        Args:
            request: The request that was allocated resources. Its
                `py_num_connector_matched_tokens` is the honoured part of the
                offer, which may be shorter than the offer when this
                iteration's pages cannot hold all of it.
            block_ids: Unused; see above.
        """
        self._settle_offer(request)

    # ---- work lists ----

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> MooncakeStoreMetadata:
        """Turn this iteration's scheduled requests into load and save lists."""
        self._confirm_submitted_saves()
        metadata = MooncakeStoreMetadata()
        for request_data in (*scheduler_output.new_requests, *scheduler_output.cached_requests):
            state = self._requests.get(request_data.request_id)
            if state is None:
                # Only requests that went through get_num_new_matched_tokens
                # have a hash chain.
                continue

            state.tokens.extend(request_data.new_tokens)
            state.chain.extend(state.tokens)
            self._record_pages(state, request_data)

            loads = self._loads_for(state, request_data)
            if loads.pages:
                metadata.loads.append(loads)
            # The honoured restore, which this iteration's `computed_position`
            # leaves out; see `_saves_for`.
            num_loaded_tokens = state.load_blocks * self._tokens_per_block

            # Whatever the store just supplied, and whatever the local cache
            # matched, is not ours to write back: the store already has the
            # former, and the latter was never allocated during this run.
            state.saved_upto = max(state.saved_upto, state.load_first_block + state.load_blocks)
            # An offer is consumed once. The load is issued in exactly the
            # iteration the runtime allocated pages to hold it.
            state.load_blocks = 0

            if self._config.role.saves:
                saves, saves_end = self._saves_for(state, request_data, num_loaded_tokens)
                if saves.pages:
                    state.emitted_saves = True
                    metadata.saves.append(saves)
                    self._unconfirmed_saves[request_data.request_id] = saves_end
                else:
                    # Nothing publishable in the range, so nothing to confirm.
                    state.saved_upto = max(state.saved_upto, saves_end)
        return metadata

    def request_finished(self, request: LlmRequest, cache_block_ids: List[int]) -> bool:
        """Report whether pages must stay pinned for in-flight saves.

        The answer is "any save was emitted" rather than "a save is in
        flight". Under tensor parallelism only rank 0's adapter answers while
        write completion is known per rank; under attention DP the adapter and
        its worker share the owner. Either way a request whose writes already
        retired, or whose emitted saves were never queued because their pass
        did not run, is reported done by the next `get_finished`, costing one
        iteration.

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
        and the lookup preceding the next attempt rebuilds the rest. Saves still
        awaiting confirmation belong to that allocation too, so a report about
        them cannot advance the next attempt's save mark.
        """
        self._requests.pop(request.request_id, None)
        self._unconfirmed_saves.pop(request.request_id, None)

    # ---- internals ----

    def _settle_offer(self, request: LlmRequest) -> None:
        """Trim the request's outstanding offer to the whole blocks the runtime honoured.

        Both the load and the computed-block bound of this pass's saves then
        use the honoured count rather than the offer. The runtime sets
        `py_num_connector_matched_tokens` when it commits the honoured part of
        the answer, before it reports the allocation, and it only ever honours
        a whole-block prefix of what was offered.

        Raises:
            RuntimeError: When the honoured count is not a whole-block part of
                the offer. Loading it anyway would leave the runtime treating
                tokens as computed that no page transfer fills.
        """
        state = self._requests.get(request.request_id)
        if state is None or state.load_blocks == 0:
            return
        honoured = int(request.py_num_connector_matched_tokens)
        blocks, remainder = divmod(honoured, self._tokens_per_block)
        if remainder or not 0 <= blocks <= state.load_blocks:
            raise RuntimeError(
                f"mooncake-store: the runtime honoured {honoured} tokens of request "
                f"{request.request_id}, which is not a whole-block part of the "
                f"{state.load_blocks}-block offer"
            )
        state.load_blocks = blocks

    def _confirm_submitted_saves(self) -> None:
        """Count the last bound saves as stored once the worker queued them.

        Bound metadata may never reach `wait_for_save`: the iteration can fail
        between `handle_metadata` and the forward pass, and once attention data
        parallelism or asynchronous loads are supported, a fleet vote, a
        same-iteration rebuild or a batch the connector emptied can supersede
        it. Such blocks were not computed by a pass that ran, so they are
        emitted again when the request is next scheduled instead of being lost.
        When the executor cannot queue a batch nothing is bound in the first
        place, because only `prepare_resources` publishes a scheduler output.

        The worker's report is taken on every build, even with nothing
        unconfirmed, so a report about a request reset since cannot confirm
        that request's next pass if that pass never runs.
        """
        if not self._config.role.saves:
            return
        submitted = self._require_worker().take_submitted_saves()
        for request_id, saves_end in self._unconfirmed_saves.items():
            state = self._requests.get(request_id)
            if state is not None and request_id in submitted:
                state.saved_upto = max(state.saved_upto, saves_end)
        self._unconfirmed_saves.clear()

    def _require_worker(self) -> MooncakeStoreConnectorWorker:
        if self._worker is None:
            self._worker = resolve_local_worker()
        return self._worker

    def _state_for(
        self, request: LlmRequest, tokens: List[int], scope: ReuseScope
    ) -> _RequestState:
        state = self._requests.get(request.request_id)
        if state is None:
            state = _RequestState(BlockHashChain(self._tokens_per_block, scope=scope))
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
        """The honoured part of the offer, one transfer per block.

        Raises:
            RuntimeError: When an honoured block has no destination page. The
                runtime counts honoured tokens as computed, so skipping the
                block would leave it to attend to KV nothing wrote.
        """
        transfers = RequestTransfers(request_data.request_id)
        limit = self._addressable_blocks(state)
        for offset in range(state.load_blocks):
            block = state.load_first_block + offset
            if block >= limit:
                raise RuntimeError(
                    f"mooncake-store: honoured prefix block {block} of request "
                    f"{request_data.request_id} has no destination page "
                    f"({limit} addressable)"
                )
            self._append_pages(state, transfers, block)
        return transfers

    def _saves_for(
        self, state: _RequestState, request_data: RequestData, num_loaded_tokens: int
    ) -> Tuple[RequestTransfers, int]:
        """This pass's newly computed full blocks, and the block ordinal they end at.

        Prompt hashes exist before prefill, and pages can run ahead of
        computation: the post-allocation token-budget trim shrinks a context
        chunk whose pages were already allocated, and a served prefix grows
        capacity past the chunk by any extra KV tokens a one-model draft needs.
        The worker saves after this pass, so only the full blocks it completes
        are publishable.

        Args:
            state: The request's bookkeeping.
            request_data: This iteration's scheduler output for the request.
                Its `computed_position` excludes this iteration's connector
                restore.
            num_loaded_tokens: The honoured restore of this iteration, added
                back to `computed_position`.

        Returns:
            The pages to save, and the block ordinal the save mark may advance
            to. The mark itself is left alone; the caller advances it once the
            saves are known to have been handed to the store.
        """
        transfers = RequestTransfers(request_data.request_id)
        computed_end = (
            request_data.computed_position + num_loaded_tokens + request_data.num_scheduled_tokens
        )
        limit = min(self._addressable_blocks(state), computed_end // self._tokens_per_block)
        for block in range(state.saved_upto, limit):
            self._append_pages(state, transfers, block)
        return transfers, max(state.saved_upto, limit)

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
