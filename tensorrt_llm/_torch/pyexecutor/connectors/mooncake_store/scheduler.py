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

Runs only on rank 0, deciding what to load and what to save while the workers
do the moving. That needs two pieces of per-request bookkeeping:

* a hash chain, since `KVCacheManagerV2` reports `RequestData.block_hashes`
  empty and a block has no content identity otherwise;
* the page slot index per block ordinal, accumulated across iterations. The
  manager reports only newly allocated indices each step, but a block is
  allocated before it is full and is savable only once it is full.

A sliding-window layer group keeps only the blocks its window still reads, so
its pages are saved once per prompt, in the prompt's last chunk: the blocks the
window still reads where the prompt ends. That is what the next turn of a
conversation resumes from, so a lookup resumes only at a prompt end whose
window is stored, searching back up to `swa_endpoint_rewind_tokens` from the
longest full-attention hit. Full-attention groups are saved block by block.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from tensorrt_llm.bindings.internal.batch_manager import LlmRequest
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime.kv_cache_manager_v2 import BAD_PAGE_INDEX

from ..kv_cache_connector import KvCacheConnectorScheduler, PrefixLoad, RequestData, SchedulerOutput
from .config import MooncakeStoreConnectorConfig
from .keys import BlockHashChain, ReuseScope
from .metadata import MooncakeStoreMetadata, PageTransfer, PrefixLoadTransfers, RequestTransfers
from .validation import validate_llm_args
from .worker import (
    RESAVE_MIN_TOKENS_ENV,
    SYNC_LOAD_ENV,
    MooncakeStoreConnectorWorker,
    resolve_local_worker,
)

__all__ = ["MooncakeStoreConnectorScheduler"]

#: Whether a prompt's window pages go to the save thread ahead of its full-attention blocks (on by default; 0 queues
#: the full-attention blocks first). A lookup that lands between the two then finds the previous prompt end intact
#: instead of full-attention blocks with no window.
WINDOW_FIRST_ENV = "TRTLLM_MOONCAKE_STORE_WINDOW_FIRST"


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


@dataclass
class _Reservation:
    """What `reserve_prefix` promised, in tokens, and the windows it verified."""

    start: int
    end: int
    #: Per sliding-window group, the first block from which the store held every
    #: page up to the promised end.
    window_present_from: Dict[int, int] = field(default_factory=dict)


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
        "request",
        "window_present_from",
        "saved_window",
        "prompt_blocks",
        "reservations",
        "resume_block",
        "resaved",
    )

    def __init__(self, chain: BlockHashChain, request: LlmRequest, prompt_blocks: int):
        self.chain = chain
        #: The request itself, read for where this iteration's chunk starts and
        #: whether it finishes the prompt.
        self.request = request
        #: Full blocks of the prompt, the only blocks saved under speculative
        #: decoding.
        self.prompt_blocks = prompt_blocks
        #: Per sliding-window group, the first block of the offer's window the
        #: store was found to hold.
        self.window_present_from: Dict[int, int] = {}
        #: Whether the window pages the prompt ends with were handed over.
        self.saved_window = False
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
        #: Open prefix reservations by id, until released.
        self.reservations: Dict[int, _Reservation] = {}
        #: The block the request's computation started at (the prefix the local cache
        #: and the store gave it), read at its first scheduled chunk.
        self.resume_block: Optional[int] = None
        #: Whether its whole prompt was listed again at the last chunk (long prefills only).
        self.resaved = False


class MooncakeStoreConnectorScheduler(KvCacheConnectorScheduler):
    """Chooses which pages the Mooncake pool serves and which it receives."""

    def __init__(self, llm_args: TorchLlmArgs):
        super().__init__(llm_args)

        validate_llm_args(llm_args)
        self._config = MooncakeStoreConnectorConfig.resolve(llm_args)
        self._tokens_per_block = int(llm_args.kv_cache_config.tokens_per_block)
        # How far back from the longest full-attention hit a lookup searches for
        # a prompt end whose window is stored.
        reuse_config = getattr(llm_args.kv_cache_config, "block_reuse_config", None)
        self._window_rewind_tokens = int(
            getattr(reuse_config, "swa_endpoint_rewind_tokens", 0) or 0
        )
        # Rejected draft tokens shrink a request's page list during generation,
        # and the freed slot goes to whichever request allocates next, while the
        # page indices here only ever grow. Prompt blocks are allocated and
        # filled before the first draft, so saving only those never names a
        # slot another request owns. A disaggregated context server, which never
        # generates, loses nothing by it.
        spec_config = getattr(llm_args, "speculative_config", None)
        self._prompt_blocks_only = bool(
            spec_config is not None and (getattr(spec_config, "max_draft_len", 0) or 0) > 0
        )
        self._async_loads = os.environ.get(SYNC_LOAD_ENV, "0") != "1"
        self._window_first = os.environ.get(WINDOW_FIRST_ENV, "1") != "0"
        self._resave_min_tokens = int(os.environ.get(RESAVE_MIN_TOKENS_ENV, "0"))
        self._requests: Dict[int, _RequestState] = {}
        self._worker: Optional[MooncakeStoreConnectorWorker] = None

        logger.warning(
            f"mooncake-store leader ready (role={self._config.role.value}, "
            f"tokens_per_block={self._tokens_per_block}, "
            f"reserved prefixes load {'asynchronously' if self._async_loads else 'synchronously'})"
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
        found = self._lookup(request, num_computed_tokens)
        if found is None:
            return 0, False
        state, first_block, hit_blocks, window_present_from = found
        state.load_first_block = first_block
        state.load_blocks = hit_blocks
        state.window_present_from = window_present_from
        return hit_blocks * self._tokens_per_block, False

    def reserve_prefix(
        self, request: LlmRequest, num_computed_tokens: int, reservation_id: int
    ) -> Tuple[int, bool]:
        """Promise the longest stored prefix beyond the local match, loaded later.

        The runtime confirms all or part of the promise when it admits the
        request, and the pages move only then, from `SchedulerOutput.prefix_loads`.
        Nothing is pinned in the pool: a page evicted in between fails its load,
        which is fatal under tensor parallelism, as on the synchronous path.

        Returns:
            Tokens promised from `num_computed_tokens`, and whether they load
            asynchronously, with the request parked outside the batch meanwhile.
        """
        found = self._lookup(request, num_computed_tokens)
        if found is None:
            return 0, False
        state, first_block, hit_blocks, window_present_from = found
        start = first_block * self._tokens_per_block
        end = start + hit_blocks * self._tokens_per_block
        state.reservations[reservation_id] = _Reservation(start, end, window_present_from)
        return end - start, self._async_loads

    def release_prefix_reservation(
        self, request: LlmRequest, reservation_id: int, start: int, end: int
    ) -> None:
        """Forget the released part of a reservation.

        The pool holds no pin for it, so there is nothing to hand back: the
        record only serves to check a confirmed load against the windows its
        lookup verified.
        """
        state = self._requests.get(request.request_id)
        reservation = None if state is None else state.reservations.get(reservation_id)
        if reservation is None:
            return
        if start <= reservation.start and end >= reservation.end:
            del state.reservations[reservation_id]
        elif end >= reservation.end:
            reservation.end = min(reservation.end, start)
        elif start <= reservation.start:
            reservation.start = max(reservation.start, end)

    def _lookup(
        self, request: LlmRequest, num_computed_tokens: int
    ) -> Optional[Tuple[_RequestState, int, int, Dict[int, int]]]:
        """The stored prefix after the local match: state, first block, blocks, windows.

        None when there is nothing to offer, including for a request whose pages
        no key can name; such a request gets no state and so no saves either.
        """
        if self._config.capacity_only:
            # Neither a lookup nor a save will read the hash chain this would
            # otherwise build, which costs a SHA-256 over the whole prompt of
            # every request.
            return None

        scope = _reuse_scope(request)
        if scope is None:
            # Leaving it without a `_RequestState` suppresses its saves too,
            # since `build_connector_meta` skips requests it has no state for.
            return None

        tokens = request.get_tokens(0)
        state = self._state_for(request, tokens, scope)
        state.load_first_block = 0
        state.load_blocks = 0
        state.window_present_from = {}

        if not self._config.role.loads:
            return None

        # A partial local match means the boundary block is half computed on
        # device. Overwriting it with a stored page would discard tokens the
        # runtime already counted, so only whole-block offers are made.
        if num_computed_tokens % self._tokens_per_block:
            return None

        first_block = num_computed_tokens // self._tokens_per_block
        # Stop one token short of the prompt: the runtime still has to run a
        # forward pass for this request, and it cannot do that with nothing left
        # to compute.
        last_block = (len(tokens) - 1) // self._tokens_per_block
        candidates = state.chain.hashes[first_block:last_block]
        if not candidates:
            return None

        window_present_from: Dict[int, int] = {}
        if self._windowed_groups():
            hit = self._require_worker().find_prefix_hit(
                candidates, first_block, self._window_rewind_tokens
            )
            hit_blocks = hit.blocks
            window_present_from = dict(hit.window_present_from)
        else:
            hit_blocks = self._require_worker().count_prefix_hit(candidates)
        if hit_blocks == 0:
            return None

        logger.debug(
            f"mooncake-store matched {hit_blocks} blocks "
            f"({hit_blocks * self._tokens_per_block} tokens) "
            f"for request {request.request_id}"
        )
        return state, first_block, hit_blocks, window_present_from

    def update_state_after_alloc(self, request: LlmRequest, block_ids: List[int]):
        """No-op: page indices are read from the scheduler output instead.

        The flat `block_ids` here are a single space, but a V2 page index is
        scoped to a layer group, so this connector reads only
        `RequestData.new_block_ids_by_layer_group`.
        """

    def update_state_after_alloc_by_layer_group(
        self, request: LlmRequest, block_ids_by_layer_group: List[List[int]]
    ):
        """No-op, as `update_state_after_alloc`: the scheduler output carries these."""

    # ---- work lists ----

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> MooncakeStoreMetadata:
        """Turn this iteration's scheduled requests into load and save lists."""
        metadata = MooncakeStoreMetadata()
        # Confirmed reservations, each listed once; an asynchronous one's
        # request is parked outside the batch until its pages are reported.
        for load in scheduler_output.prefix_loads:
            metadata.prefix_loads.append(self._prefix_load_transfers(load))
        for request_data in (*scheduler_output.new_requests, *scheduler_output.cached_requests):
            state = self._requests.get(request_data.request_id)
            if state is None:
                # Only requests that went through get_num_new_matched_tokens
                # have a hash chain.
                continue

            state.tokens.extend(request_data.new_tokens)
            state.chain.extend(state.tokens)
            self._record_pages(state, request_data)
            if state.resume_block is None:
                # The first chunk starts where the local cache and the store left off. Unlike
                # `computed_position`, this counts the store's tokens too.
                position = int(getattr(state.request, "context_current_position", 0) or 0)
                state.resume_block = position // self._tokens_per_block

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

    def request_finished_by_layer_group(
        self, request: LlmRequest, cache_block_ids_by_layer_group: List[List[int]]
    ) -> bool:
        """`request_finished` for a cache of several layer groups, a sliding window's.

        The answer does not depend on the page indices, so it is the same.
        """
        return self.request_finished(request, [])

    def request_reset(self, request: LlmRequest) -> None:
        """Forget a request whose pages were released for it to run again.

        `pages` and `saved_upto` describe the allocation that was just freed,
        and the lookup preceding the next attempt rebuilds the rest.
        """
        self._requests.pop(request.request_id, None)

    # ---- internals ----

    def _require_worker(self) -> MooncakeStoreConnectorWorker:
        if self._worker is None:
            self._worker = resolve_local_worker()
        return self._worker

    def _windowed_groups(self) -> Tuple[int, ...]:
        """Sliding-window layer groups of the registered layout."""
        return tuple(getattr(self._require_worker(), "windowed_group_ids", ()))

    def _prefix_load_transfers(self, load: PrefixLoad) -> PrefixLoadTransfers:
        """The pages a confirmed reservation fills, into the pages the runtime allocated.

        A sliding-window group has pages only for the blocks its window reads at
        the load's end, and each of those has to be one the lookup found
        stored, or the replay would read a page nothing wrote.
        """
        state = self._requests.get(load.request_id)
        reservation = None if state is None else state.reservations.get(load.reservation_id)
        if reservation is None:
            raise RuntimeError(
                f"mooncake-store: prefix load {load.reservation_id} of request "
                f"{load.request_id} has no reservation behind it"
            )
        tokens_per_block = self._tokens_per_block
        first, end = load.start // tokens_per_block, load.end // tokens_per_block
        windowed = self._windowed_groups()
        worker = self._require_worker()
        transfers = RequestTransfers(load.request_id)
        for layer_group_id, indices in enumerate(load.block_ids_by_layer_group):
            is_window = layer_group_id in windowed
            # Sinks sit below the window's tail and every lookup requires them.
            sinks = worker.sink_blocks(layer_group_id) if is_window else 0
            present_from = reservation.window_present_from.get(layer_group_id, end)
            for block in range(first, end):
                page_index = indices[block] if block < len(indices) else BAD_PAGE_INDEX
                if page_index == BAD_PAGE_INDEX and is_window:
                    # The window has passed this block, so nothing reads it.
                    continue
                if page_index == BAD_PAGE_INDEX or (is_window and sinks <= block < present_from):
                    raise RuntimeError(
                        f"mooncake-store: request {load.request_id} loads blocks "
                        f"[{first}, {end}) and layer group {layer_group_id} would read "
                        f"block {block}, which the store was not found to hold. Loading "
                        "the rest would leave that page unwritten."
                    )
                transfers.pages.append(
                    PageTransfer(state.chain.hashes[block], layer_group_id, page_index)
                )
        # The store already has what it supplies, so none of it is written back.
        state.saved_upto = max(state.saved_upto, end)
        return PrefixLoadTransfers(load.reservation_id, load.is_async, transfers)

    def _state_for(
        self, request: LlmRequest, tokens: List[int], scope: ReuseScope
    ) -> _RequestState:
        state = self._requests.get(request.request_id)
        if state is None:
            # Looked up before the request is scheduled, so `tokens` is its prompt.
            state = _RequestState(
                BlockHashChain(self._tokens_per_block, scope=scope),
                request,
                len(tokens) // self._tokens_per_block,
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
        if state.load_blocks:
            windowed = self._windowed_groups()
            if windowed:
                self._append_window_loads(state, transfers, windowed)
                return transfers
        limit = self._addressable_blocks(state)
        for offset in range(state.load_blocks):
            block = state.load_first_block + offset
            if block >= limit:
                # The runtime allocated fewer pages than it accepted tokens
                # for, so the tail of the offer has nowhere to land. Dropping
                # it here leaves the runtime to recompute it.
                break
            self._append_pages(state, transfers, block)
        return transfers

    def _saves_for(self, state: _RequestState, request_data: RequestData) -> RequestTransfers:
        transfers = RequestTransfers(request_data.request_id)
        windowed = self._windowed_groups()
        limit = self._addressable_blocks(state)
        if self._prompt_blocks_only:
            limit = min(limit, state.prompt_blocks)
        request = state.request
        chunk_size = getattr(request, "context_chunk_size", None)
        if getattr(request, "is_context_init_state", False) and chunk_size is not None:
            # The cache allocates a little past the chunk (a drafter's extra KV tokens), so a page can exist for a
            # block only the next chunk computes. Save just the blocks this forward completes.
            completed = int(request.context_current_position) + int(chunk_size)
            limit = min(limit, completed // self._tokens_per_block)
        save_window = bool(windowed) and not state.saved_window and self._finishes_prompt(state)
        if save_window and self._window_first:
            self._append_window_pages(state, transfers, windowed)
        first = state.saved_upto
        if self._resave_min_tokens and not state.resaved and self._finishes_prompt(state):
            computed = len(state.tokens) - (state.resume_block or 0) * self._tokens_per_block
            if computed >= self._resave_min_tokens:
                # A long prefill wrote its first blocks minutes ago and never touched them again, so the pool can
                # evict the head of the chain before the conversation's next turn arrives. Listing the whole prompt
                # once more makes `_put` refresh what the pool still holds and rewrite what it evicted, from pages
                # this request still holds.
                first = 0
                state.resaved = True
        for block in range(first, limit):
            self._append_pages(state, transfers, block, skip=windowed)
        state.saved_upto = max(state.saved_upto, limit)
        if save_window:
            if not self._window_first:
                self._append_window_pages(state, transfers, windowed)
            state.saved_window = True
        return transfers

    def _append_pages(
        self,
        state: _RequestState,
        transfers: RequestTransfers,
        block: int,
        skip: Tuple[int, ...] = (),
    ) -> None:
        """Add one block's page from every layer group but `skip`, or none of them."""
        block_hash = state.chain.hashes[block]
        pages: List[PageTransfer] = []
        for layer_group_id, indices in state.pages.items():
            if layer_group_id in skip:
                continue
            page_index = indices[block]
            if page_index == BAD_PAGE_INDEX:
                # A sliding window dropped the block in this group, and a
                # partial block is not a usable cache entry.
                return
            pages.append(PageTransfer(block_hash, layer_group_id, page_index))
        transfers.pages.extend(pages)

    @staticmethod
    def _finishes_prompt(state: _RequestState) -> bool:
        """Whether this iteration computes the last chunk of the request's prompt."""
        request = state.request
        return bool(request.is_context_init_state and request.is_last_context_chunk)

    def _append_window_pages(
        self, state: _RequestState, transfers: RequestTransfers, windowed: Tuple[int, ...]
    ) -> None:
        """Add the window pages a later turn resumes from: the ones this prompt ends with.

        Only the blocks the window still reads at the prompt's end. The commit
        that closes the last chunk releases every block below that window in
        the same iteration, before the save thread copies anything, so a page
        further back (the rewind margin) may already hold another request's KV.
        A lookup therefore resumes exactly at a stored prompt end, or through
        the fallback at an earlier one.
        """
        worker = self._require_worker()
        tokens_per_block = self._tokens_per_block
        chunk_start = int(state.request.context_current_position)
        prompt_end = min(len(state.tokens) // tokens_per_block, len(state.chain.hashes))
        earliest_resume = prompt_end * tokens_per_block
        for layer_group_id in windowed:
            indices = state.pages.get(layer_group_id, [])
            sinks = worker.sink_blocks(layer_group_id)
            _, needed_from = worker.stale_blocks(layer_group_id, earliest_resume)
            _, held_from = worker.stale_blocks(layer_group_id, chunk_start)
            first = max(needed_from, held_from, sinks)
            for block in (*range(min(sinks, prompt_end)), *range(first, prompt_end)):
                page_index = indices[block] if block < len(indices) else BAD_PAGE_INDEX
                if page_index == BAD_PAGE_INDEX:
                    continue
                transfers.pages.append(
                    PageTransfer(state.chain.hashes[block], layer_group_id, page_index)
                )

    def _append_window_loads(
        self, state: _RequestState, transfers: RequestTransfers, windowed: Tuple[int, ...]
    ) -> None:
        """Loads for a layout with sliding windows, up to where the runtime resumes.

        The runtime may honour less of the offer than was made. It resumes at
        this chunk's start and gives a window group pages only for the blocks
        it reads from there, so those are what is loaded, with every honoured
        full-attention block. A window block the lookup did not find stored
        means the replay would read a page nothing wrote, which is refused.
        """
        worker = self._require_worker()
        resume = int(state.request.context_current_position) // self._tokens_per_block
        end = min(state.load_first_block + state.load_blocks, resume)
        end = min(end, self._addressable_blocks(state))
        for layer_group_id, indices in state.pages.items():
            is_window = layer_group_id in windowed
            # Sinks sit below the window's tail and every lookup requires them.
            sinks = worker.sink_blocks(layer_group_id) if is_window else 0
            present_from = state.window_present_from.get(layer_group_id, end)
            for block in range(state.load_first_block, end):
                page_index = indices[block]
                if page_index == BAD_PAGE_INDEX and is_window:
                    # The window has passed this block, so nothing reads it.
                    continue
                if page_index == BAD_PAGE_INDEX or (is_window and sinks <= block < present_from):
                    raise RuntimeError(
                        f"mooncake-store: request {transfers.request_id} resumes at "
                        f"block {resume} of the {state.load_first_block + state.load_blocks} "
                        f"offered, and layer group {layer_group_id} would read block "
                        f"{block}, which the store was not found to hold. Loading the "
                        "rest would leave that page unwritten."
                    )
                transfers.pages.append(
                    PageTransfer(state.chain.hashes[block], layer_group_id, page_index)
                )
