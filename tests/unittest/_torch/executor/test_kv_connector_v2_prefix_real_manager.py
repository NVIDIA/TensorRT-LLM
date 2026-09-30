# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The KV connector prefix against a *real* cache manager.

``test_kv_connector_v2_prefix.py`` drives the same code against a stub cache.
That is what makes it fast and exhaustive, and it is the right place for the
arithmetic and the ask-once rules -- but a stub cannot show that a real
``_KVCache`` survives the sequence: that ``resize`` finds real pages for the
offered prefix, that ``history_length`` really moves, that the grow the chunked
path needs succeeds against real pools, and that the page slots handed to the
connector are distinct and real.

The engine-level suite cannot show the scheduling-order claims either, because
whether a request is dropped after being prepared depends on which pass it
reaches the scheduler in -- a race. Preparation and delivery are therefore
driven directly here: ``prepare_context`` plus ``resize_context`` is one
scheduling pass, and ``prepare_resources`` is the batch actually running.

These tests allocate device memory pools.
"""

import gc
from types import SimpleNamespace

import pytest
import torch

import tensorrt_llm
import tensorrt_llm.bindings
from tensorrt_llm._torch.pyexecutor.connectors.kv_cache_layout import valid_page_slots
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, SamplingConfig
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import BAD_PAGE_INDEX, AttnLifeCycle

DataType = tensorrt_llm.bindings.DataType
CacheType = tensorrt_llm.bindings.internal.batch_manager.CacheType

# These build a real manager, which allocates device pools. The directory is
# listed in the GPU-less l0_cpu stage, so the requirement is declared rather
# than left to fail at `torch.cuda.init()`.
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="allocates real KV cache pools"
)

TOKENS_PER_BLOCK = 32
PROMPT_LEN = 96
OFFER_TOKENS = 32


class FakeConnectorManager:
    """Records what the prefix path tells the connector, in order."""

    def __init__(self, num_matched=OFFER_TOKENS, load_async=False):
        self.prefix_reservations_enabled = False
        self.reservations = {}
        self.reservation_requests = {}
        self.next_reservation_id = 1
        self.releases = []
        self.accepted = []
        self.dispatched = set()
        self.num_matched = num_matched
        self.load_async = load_async
        self.queries = []
        self.commits = []
        self.allocs = []
        self.allocs_by_group = []
        self.forgotten = []

    def reserve_prefix(self, request: LlmRequest, local_end: int) -> SimpleNamespace | None:
        if request.request_id in self.reservations:
            return self.reservations[request.request_id]
        self.queries.append((request.request_id, local_end))
        if not self.num_matched:
            return None
        reservation = SimpleNamespace(
            reservation_id=self.next_reservation_id,
            request_id=request.request_id,
            start=local_end,
            end=local_end + self.num_matched,
            is_async=self.load_async,
        )
        self.next_reservation_id += 1
        self.reservations[request.request_id] = reservation
        self.reservation_requests[request.request_id] = request
        return reservation

    def get_prefix_reservation(self, request: LlmRequest) -> SimpleNamespace | None:
        return self.reservations.get(request.request_id)

    def trim_prefix_reservation(self, request: LlmRequest, start: int, end: int) -> SimpleNamespace:
        reservation = self.reservations[request.request_id]
        if reservation.start < start:
            self.releases.append((reservation.reservation_id, reservation.start, start))
        if end < reservation.end:
            self.releases.append((reservation.reservation_id, end, reservation.end))
        reservation.start, reservation.end = start, end
        return reservation

    def release_prefix_reservation(self, request: LlmRequest) -> None:
        reservation = self.reservations.pop(request.request_id, None)
        self.reservation_requests.pop(request.request_id, None)
        if reservation is not None:
            self.releases.append((reservation.reservation_id, reservation.start, reservation.end))

    def pending_prefix_requests(self) -> list[LlmRequest]:
        return list(self.reservation_requests.values())

    def accept_prefix_load(
        self, request: LlmRequest, start: int, end: int, block_ids_by_layer_group: list[list[int]]
    ) -> None:
        reservation = self.reservations.pop(request.request_id)
        self.reservation_requests.pop(request.request_id)
        self.accepted.append((reservation, block_ids_by_layer_group))
        request.py_num_connector_matched_tokens = end - start

    def release_unstarted_prefix_loads(self, request: LlmRequest) -> None:
        self.accepted = [
            entry
            for entry in self.accepted
            if entry[0].request_id != request.request_id or request.request_id in self.dispatched
        ]

    def has_pending_load(self, request: LlmRequest) -> bool:
        return any(entry[0].request_id == request.request_id for entry in self.accepted)

    def query_num_new_matched_tokens(self, request, num_computed_tokens):
        self.queries.append((request.py_request_id, num_computed_tokens))
        return self.num_matched, self.load_async

    def commit_new_matched_tokens(self, request, num_tokens, load_kv_async):
        self.commits.append((request.py_request_id, num_tokens, load_kv_async))
        request.py_num_connector_matched_tokens = num_tokens

    def should_add_sequence(self, request):
        return True

    def reset_request_state(self, request):
        self.forgotten.append(request.py_request_id)

    def update_state_after_alloc(self, request, block_ids, by_layer_group=None):
        self.allocs.append((request.py_request_id, list(block_ids)))
        self.allocs_by_group.append(
            (
                request.py_request_id,
                None if by_layer_group is None else [list(g) for g in by_layer_group],
            )
        )

    def build_scheduler_output(self, scheduled_batch, kv_cache_manager):
        pass


def make_manager(connector, **overrides):
    kwargs = dict(
        kv_cache_config=KvCacheConfig(max_tokens=2048, enable_block_reuse=True),
        kv_cache_type=CacheType.SELF,
        num_layers=2,
        num_kv_heads=4,
        head_dim=64,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=256,
        max_batch_size=4,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=DataType.HALF,
        vocab_size=32000,
        kv_connector_manager=connector,
    )
    kwargs.update(overrides)
    return KVCacheManagerV2(**kwargs)


def make_request(request_id=1, prompt_len=PROMPT_LEN):
    return LlmRequest(
        request_id=request_id,
        max_new_tokens=4,
        input_tokens=list(range(prompt_len)),
        sampling_config=SamplingConfig(1),
        is_streaming=False,
    )


def schedule(manager, request, num_tokens=None):
    """One scheduling pass: prepare the cache and size it for the chunk."""
    assert manager.prepare_context(request)
    if num_tokens is None:
        num_tokens = request.context_remaining_length
    return manager.resize_context(request, num_tokens)


def run(manager, *requests):
    """One ``prepare_resources``, i.e. the requests reached the final batch."""
    batch = ScheduledRequests()
    for request in requests:
        batch.append_context_request(request)
    manager.prepare_resources(batch)
    return batch


@pytest.fixture
def connector():
    return FakeConnectorManager()


@pytest.fixture
def manager(connector):
    torch.cuda.init()
    gc.collect()
    torch.cuda.empty_cache()
    mgr = make_manager(connector)
    yield mgr
    mgr.shutdown()
    del mgr
    gc.collect()
    torch.cuda.empty_cache()


def test_a_request_dropped_before_the_batch_is_never_asked(manager, connector):
    """The whole point of asking in ``prepare_resources``.

    A request can be prepared and sized and then lose the token budget, fail
    multimodal alignment, or be dropped when the batch cannot be queued. None
    of that can strand an offer, because the query comes after all of it.
    """
    request = make_request()

    assert schedule(manager, request)

    assert connector.queries == []
    assert connector.commits == []
    assert request.context_current_position == 0


def test_offer_is_backed_by_real_pages(manager, connector):
    """Read capacity, history and page slots back off the cache the forward
    pass would use -- the difference between "resize was called correctly" and
    "the offered prefix is resident"."""
    request = make_request()
    assert schedule(manager, request)

    run(manager, request)

    kv_cache = manager.kv_cache_map[request.py_request_id]
    assert request.context_current_position == OFFER_TOKENS
    assert kv_cache.history_length == OFFER_TOKENS
    assert kv_cache.capacity >= PROMPT_LEN
    assert kv_cache.is_active

    assert connector.commits == [(request.py_request_id, OFFER_TOKENS, False)]
    assert len(connector.allocs) == 1

    _, page_indices = connector.allocs[0]
    assert len(page_indices) >= PROMPT_LEN // TOKENS_PER_BLOCK
    assert all(index != BAD_PAGE_INDEX for index in page_indices)
    assert len(set(page_indices)) == len(page_indices)


def test_the_unchunked_path_allocates_nothing_for_the_prefix(manager, connector):
    """``resize_context`` already covered the whole prompt, so honouring the
    offer only moves the request's start."""
    request = make_request()
    assert schedule(manager, request)
    kv_cache = manager.kv_cache_map[request.py_request_id]
    before = kv_cache.capacity

    run(manager, request)

    assert kv_cache.capacity == before
    assert request.context_current_position + request.context_chunk_size == PROMPT_LEN


def test_a_chunked_offer_beyond_the_chunk_grows_and_shifts(manager, connector):
    """Chunked prefill keeps its per-chunk allocation, so an offer past the
    chunk has to grow the cache before it can be honoured."""
    connector.num_matched = 64
    request = make_request()
    request.context_chunk_size = TOKENS_PER_BLOCK
    assert schedule(manager, request, num_tokens=TOKENS_PER_BLOCK)
    kv_cache = manager.kv_cache_map[request.py_request_id]
    assert kv_cache.capacity < 64 + TOKENS_PER_BLOCK

    run(manager, request)

    assert request.context_current_position == 64
    assert request.context_chunk_size == TOKENS_PER_BLOCK
    assert kv_cache.capacity >= 64 + TOKENS_PER_BLOCK
    assert kv_cache.history_length == 64
    assert connector.commits == [(request.py_request_id, 64, False)]


def test_a_second_pass_reports_one_allocation(manager, connector):
    """An asynchronously loaded request re-enters on its first context chunk,
    with the same pages and nothing left to load."""
    request = make_request()
    assert schedule(manager, request)

    run(manager, request)
    run(manager, request)

    assert len(connector.queries) == 1
    assert len(connector.commits) == 1
    assert len(connector.allocs) == 1


def test_a_served_prefix_survives_the_scheduling_pass_that_brings_it_back(manager, connector):
    """Two ``run`` calls are not the re-entry.

    An asynchronous load parks the request, and the scheduler calls
    ``prepare_context`` again when it returns -- that is where the cursor is
    settled, from a commit depth that describes the local match alone.
    """
    connector.load_async = True
    request = make_request()
    assert schedule(manager, request)
    run(manager, request)
    served = request.context_current_position
    assert served > 0, "the serve must have moved the position"

    assert schedule(manager, request)

    assert request.context_current_position == served
    assert request.context_current_position + request.context_chunk_size == request.prompt_len

    kv_cache = manager.kv_cache_map[request.py_request_id]
    assert kv_cache.history_length == served
    assert kv_cache.num_committed_tokens < served, (
        "held but not committed -- which is why the cursor needs a floor"
    )


def test_freeing_the_allocation_makes_the_request_askable_again(manager, connector):
    """A destructive pause replays the sequence add and with it the query,
    because the pages the first answer described are gone."""
    request = make_request()
    assert schedule(manager, request)
    run(manager, request)
    manager.free_resources(request)

    # Everything keyed to the dead allocation goes with it: the ask memo, and
    # the scheduler-output deltas whose block ids describe pages that no longer
    # exist. Leaving the latter is D1, which reports the replay as a cached
    # request a `new_requests`-only connector never loads for.
    assert connector.forgotten == [request.py_request_id]

    request.reset_for_recompute(PROMPT_LEN)
    assert schedule(manager, request)
    run(manager, request)

    assert len(connector.queries) == 2


# ---------------------------------------------------------------------------
# Variable sliding-window attention.
#
# Two distinct windows over two layers give two layer groups, which is the only
# shape where the per-layer-group callbacks are reachable and the only shape
# where `_stale_block_range` has to pick a window rather than being handed the
# one there is. VSWA_PROMPT_LEN and VSWA_OFFER are sized so the offered prefix
# straddles the sliding window's edge: some block ordinals fall out of window
# and some stay live, in the same request.
# ---------------------------------------------------------------------------

VSWA_WINDOW = 64
VSWA_MAX_SEQ_LEN = 256
VSWA_PROMPT_LEN = 160
VSWA_OFFER = 128


def make_vswa_manager(connector, **overrides):
    return make_manager(
        connector,
        kv_cache_config=KvCacheConfig(
            max_tokens=2048,
            enable_block_reuse=True,
            # Layer 0 slides, layer 1 is full attention: an entry equal to
            # max_seq_len normalizes to None.
            max_attention_window=[VSWA_WINDOW, VSWA_MAX_SEQ_LEN],
        ),
        max_seq_len=VSWA_MAX_SEQ_LEN,
        **overrides,
    )


@pytest.fixture
def vswa_connector():
    return FakeConnectorManager(num_matched=VSWA_OFFER)


@pytest.fixture
def vswa_manager(vswa_connector):
    torch.cuda.init()
    gc.collect()
    torch.cuda.empty_cache()
    mgr = make_vswa_manager(vswa_connector)
    yield mgr
    mgr.shutdown()
    del mgr
    gc.collect()
    torch.cuda.empty_cache()


def _sliding_and_full(manager):
    """Layer group ids of the sliding and the full-attention group."""
    windows = [lc.window_size for lc in manager._life_cycle_by_layer_group()]
    assert len(windows) == 2, f"expected two layer groups, got {windows}"
    return windows.index(VSWA_WINDOW), windows.index(None)


def test_window_size_is_read_per_layer_group(vswa_manager):
    """Each group carries its own life cycle, and the pair is not interchangeable.

    Everything downstream -- the masking boundary, what a connector is offered
    to save -- is derived from this list by index, so an off-by-one here is
    silent and total.
    """
    life_cycles = vswa_manager._life_cycle_by_layer_group()
    windows = [lc.window_size for lc in life_cycles]

    assert len(windows) == 2
    assert sorted(windows, key=lambda w: (w is None, w)) == [VSWA_WINDOW, None]

    layers = vswa_manager.kv_cache_manager_py_config.layers
    for layer_group_id, local_layer_ids in enumerate(vswa_manager.impl.layer_grouping):
        for local_layer_id in local_layer_ids:
            assert windows[layer_group_id] == layers[int(local_layer_id)].window_size


def test_stale_block_range_uses_each_group_s_own_window(vswa_manager):
    """The full-attention group must never be masked, whatever the history.

    Computed here from the window rather than copied from the implementation:
    a test that reuses `_stale_block_range` to predict `_stale_block_range`
    cannot catch a wrong window being selected for the group.
    """
    sliding, full = _sliding_and_full(vswa_manager)
    tokens_per_block = vswa_manager.tokens_per_block

    for history_length in (0, 32, 64, 96, 128, 200):
        expected_end = max(0, (history_length + 1 - VSWA_WINDOW) // tokens_per_block)
        assert vswa_manager._stale_block_range(sliding, history_length) == (0, expected_end)
        assert vswa_manager._stale_block_range(full, history_length) == (0, 0), (
            "a full-attention group has no stale range; masking it would hide "
            "pages the connector is entitled to save"
        )


def test_attention_sinks_are_never_masked(vswa_manager: KVCacheManagerV2) -> None:
    """Sink blocks stay live below the window, so they must not be masked.

    The stale range starts at `num_sink_blocks`, not at 0. Attention reads the
    sink positions on every step however far they fall behind the window, so a
    connector handed `BAD_PAGE_INDEX` for them would neither save nor restore
    KV the model goes on to read.

    Both `AttentionLayerConfig` construction sites pass `num_sink_tokens=None`
    today, so the life cycle is substituted here rather than configured. This
    pins the masking loop's use of the range's lower bound, which is the part
    that would silently do the wrong thing once sinks are wired up.
    """
    sliding, full = _sliding_and_full(vswa_manager)
    tokens_per_block = vswa_manager.tokens_per_block
    # Keep one stale non-sink block so the masking assertion is exercised.
    num_sink_tokens = tokens_per_block

    life_cycles = list(vswa_manager._life_cycle_by_layer_group())
    life_cycles[sliding] = AttnLifeCycle.make(VSWA_WINDOW, num_sink_tokens, tokens_per_block)
    vswa_manager._connector_life_cycle_by_group = life_cycles

    request = make_request(prompt_len=VSWA_PROMPT_LEN)
    assert schedule(vswa_manager, request)
    run(vswa_manager, request)

    by_group = vswa_manager.get_page_indices_by_layer_group(request)
    num_sink_blocks = num_sink_tokens // tokens_per_block
    stale_end = max(0, (VSWA_OFFER + 1 - VSWA_WINDOW) // tokens_per_block)
    assert num_sink_blocks < stale_end, (
        f"test sizes put the sinks outside the stale range "
        f"(sinks={num_sink_blocks}, stale_end={stale_end}), so nothing is proven"
    )

    assert all(index != BAD_PAGE_INDEX for index in by_group[sliding][:num_sink_blocks]), (
        f"a sink block was masked: {by_group[sliding]}"
    )
    assert all(index == BAD_PAGE_INDEX for index in by_group[sliding][num_sink_blocks:stale_end]), (
        f"a block the window has passed was reported as a page: {by_group[sliding]}"
    )
    assert all(index != BAD_PAGE_INDEX for index in by_group[full]), (
        f"the full-attention group must keep every block: {by_group[full]}"
    )


def test_page_indices_mask_only_the_group_whose_window_passed(vswa_manager, vswa_connector):
    """The mask is per group, in place, and against real pages.

    The offered prefix straddles the sliding window's edge, so the same request
    has out-of-window ordinals in one group and live pages at those same
    ordinals in the other. That is the case a single flat block-id list cannot
    describe, and the case a `-1`-blind connector corrupts.
    """
    request = make_request(prompt_len=VSWA_PROMPT_LEN)
    assert schedule(vswa_manager, request)
    run(vswa_manager, request)

    kv_cache = vswa_manager.kv_cache_map[request.py_request_id]
    assert kv_cache.history_length == VSWA_OFFER, (
        "the prefix was not honoured in full, so the masking boundary below "
        "is not the one this test was sized for"
    )

    by_group = vswa_manager.get_page_indices_by_layer_group(request)
    assert len(by_group) == 2

    sliding, full = _sliding_and_full(vswa_manager)
    tokens_per_block = vswa_manager.tokens_per_block
    stale_end = max(0, (VSWA_OFFER + 1 - VSWA_WINDOW) // tokens_per_block)
    assert 0 < stale_end < len(by_group[sliding]), (
        f"test sizes no longer split the request across the window edge "
        f"(stale_end={stale_end}, blocks={len(by_group[sliding])})"
    )

    # Ordinals stay positionally aligned across groups, which is what makes an
    # append-delta over successive calls valid.
    assert len(by_group[sliding]) == len(by_group[full])

    assert all(index == BAD_PAGE_INDEX for index in by_group[sliding][:stale_end]), (
        f"a block the sliding window has passed was reported as a page: {by_group[sliding]}"
    )
    live = by_group[sliding][stale_end:]
    assert all(index != BAD_PAGE_INDEX for index in live), (
        f"an in-window block was reported with no page: {by_group[sliding]}"
    )
    assert len(set(live)) == len(live), f"page slots are not distinct: {live}"

    assert all(index != BAD_PAGE_INDEX for index in by_group[full]), (
        f"the full-attention group must keep every block: {by_group[full]}"
    )
    assert len(set(by_group[full])) == len(by_group[full])

    # The pair is not the same list read twice: at the masked ordinals one
    # group has pages and the other does not.
    assert by_group[sliding] != by_group[full]


def test_alloc_is_reported_per_layer_group_and_the_flat_list_is_empty(vswa_manager, vswa_connector):
    """A page index is scoped to a group, so the flat list must be withheld.

    Reporting group 0's indices as `block_ids` would look right to a connector
    that never checks, and address the wrong pool for every layer outside that
    group.
    """
    request = make_request(prompt_len=VSWA_PROMPT_LEN)
    assert schedule(vswa_manager, request)
    run(vswa_manager, request)

    assert len(vswa_connector.allocs) == 1
    _, flat = vswa_connector.allocs[0]
    assert flat == [], "the flat block-id list must be empty with several layer groups"

    assert len(vswa_connector.allocs_by_group) == 1
    _, by_group = vswa_connector.allocs_by_group[0]
    assert len(by_group) == 2
    assert by_group == vswa_manager.get_page_indices_by_layer_group(request)


def test_a_released_request_still_reports_one_list_per_layer_group(vswa_manager):
    """A released allocation reads back as empty lists, not as no groups at all.

    The connector callbacks are routed by this outer length, so collapsing a
    released request to `[]` sends it to the flat form, which a connector
    written for VSWA does not define.
    """
    request = make_request(prompt_len=VSWA_PROMPT_LEN)
    assert schedule(vswa_manager, request)
    run(vswa_manager, request)

    num_groups = len(vswa_manager.impl.layer_grouping)
    assert num_groups == 2, "the fixture must keep two windows for this to mean anything"
    assert len(vswa_manager.get_page_indices_by_layer_group(request)) == num_groups

    vswa_manager.free_resources(request)

    assert vswa_manager.get_page_indices_by_layer_group(request) == [[], []]


def test_a_single_window_still_reports_the_flat_list(connector):
    """Sibling check: the same code path with one group keeps the flat shape.

    Withholding the flat list is conditional on the group count, so the
    single-group arm has to be pinned here or a change to that condition
    breaks every existing connector without failing a VSWA test.
    """
    torch.cuda.init()
    mgr = make_manager(connector)
    try:
        request = make_request()
        assert schedule(mgr, request)
        run(mgr, request)

        assert len(connector.allocs) == 1
        _, flat = connector.allocs[0]
        assert flat, "a single-group cache must still report the flat block ids"

        _, by_group = connector.allocs_by_group[0]
        assert len(by_group) == 1
        assert by_group[0] == flat
    finally:
        mgr.shutdown()
        del mgr
        gc.collect()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# One sliding window across every layer: a single layer group that still
# reports the flat list, which is the arm `reject_flat_only_scheduler` lets
# through and the only one where that list can carry a sentinel.
# ---------------------------------------------------------------------------


def make_swa_manager(connector, **overrides):
    return make_manager(
        connector,
        kv_cache_config=KvCacheConfig(
            max_tokens=2048,
            enable_block_reuse=True,
            # One window for every layer: one life cycle, so one layer group.
            max_attention_window=[VSWA_WINDOW],
        ),
        max_seq_len=VSWA_MAX_SEQ_LEN,
        **overrides,
    )


def test_a_sliding_single_group_reports_sentinels_in_the_flat_list():
    """The flat list is still reported, and it carries sentinels.

    Blocks the window has passed are held in place as ``BAD_PAGE_INDEX`` so the
    ordinals stay aligned, which means a connector has to filter -- that is
    what `valid_page_slots` is for.
    """
    torch.cuda.init()
    connector = FakeConnectorManager(num_matched=VSWA_OFFER)
    mgr = make_swa_manager(connector)
    try:
        request = make_request(prompt_len=VSWA_PROMPT_LEN)
        assert schedule(mgr, request)
        run(mgr, request)

        assert len(mgr.impl.layer_grouping) == 1, "one window is one layer group"
        _, flat = connector.allocs[0]
        assert flat, "a single-group cache must still report the flat block ids"

        stale_beg, stale_end = mgr._stale_block_range(0, request.context_current_position)
        assert stale_end > stale_beg, "the window must have passed a block for this to bite"
        assert BAD_PAGE_INDEX in flat
        live = [ordinal for ordinal, _ in valid_page_slots(flat)]
        assert all(ordinal < stale_beg or ordinal >= stale_end for ordinal in live)
    finally:
        mgr.shutdown()
        del mgr
        gc.collect()
        torch.cuda.empty_cache()


def test_a_served_prefix_survives_re_entry_under_a_sliding_window():
    """Here the rewind is not only wasted work.

    ``_resize_for_connector_prefix`` sets the cache's ``history_length`` to the
    served end, and raising history unlocks the blocks the window has passed. A
    cursor settled below that end points the forward pass at ordinals whose
    pages are gone.
    """
    torch.cuda.init()
    connector = FakeConnectorManager(num_matched=VSWA_OFFER, load_async=True)
    mgr = make_swa_manager(connector)
    try:
        request = make_request(prompt_len=VSWA_PROMPT_LEN)
        assert schedule(mgr, request)
        run(mgr, request)
        served = request.context_current_position
        stale_beg, stale_end = mgr._stale_block_range(0, served)
        assert stale_end > stale_beg, "the window must have passed a block for this to bite"

        assert schedule(mgr, request)

        assert request.context_current_position == served
        indices = mgr.get_page_indices_by_layer_group(request)[0]
        first = request.context_current_position // TOKENS_PER_BLOCK
        last = (
            request.context_current_position + request.context_chunk_size - 1
        ) // TOKENS_PER_BLOCK
        assert all(indices[ordinal] != BAD_PAGE_INDEX for ordinal in range(first, last + 1)), (
            "the forward pass must not be pointed at a block the window has released"
        )
    finally:
        mgr.shutdown()
        del mgr
        gc.collect()
        torch.cuda.empty_cache()


@pytest.mark.parametrize("preallocated", [False, True])
def test_rejected_connector_candidate_releases_and_rewinds(
    manager: KVCacheManagerV2, connector: FakeConnectorManager, preallocated: bool
) -> None:
    request = make_request()
    if preallocated:
        assert schedule(manager, request)
    connector.prefix_reservations_enabled = True
    assert schedule(manager, request)
    cache = manager.kv_cache_map[request.request_id]
    assert cache.history_length == OFFER_TOKENS
    if preallocated:
        assert request.py_ctx_pre_resize_cap is None
    assert connector.accepted == []

    manager.release_unused_connector_reservations(set())

    assert request.request_id not in manager.kv_cache_map
    assert connector.releases == [(1, 0, OFFER_TOKENS)]
    assert request.context_current_position == 0
    assert request.prepopulated_prompt_len == 0
    assert request.context_chunk_size == PROMPT_LEN
    assert request.py_connector_served_position == 0
    assert schedule(manager, request)
    assert connector.get_prefix_reservation(request).reservation_id == 2
    assert connector.queries == [(request.request_id, 0), (request.request_id, 0)]


def test_token_budget_rejection_releases_real_cache(
    manager: KVCacheManagerV2, connector: FakeConnectorManager
) -> None:
    from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import KVCacheV2Scheduler
    from tensorrt_llm.llmapi.llm_args import CapacitySchedulerPolicy

    connector.prefix_reservations_enabled = True
    request = make_request()
    scheduler = KVCacheV2Scheduler(
        max_batch_size=4,
        max_num_tokens=32,
        kv_cache_manager=manager,
        scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
    )

    output = scheduler.schedule_request([request], set())

    assert output.context_requests == []
    assert request.request_id not in manager.kv_cache_map
    assert connector.releases == [(1, 0, OFFER_TOKENS)]
    assert request.context_current_position == 0
    assert connector.accepted == []


def test_reserved_prefix_tracks_range_and_allocation(
    manager: KVCacheManagerV2, connector: FakeConnectorManager
) -> None:
    from tensorrt_llm._torch.pyexecutor.llm_request import rewind_context_after_cache_drop

    connector.prefix_reservations_enabled = True
    connector.num_matched = PROMPT_LEN
    request = make_request()
    assert schedule(manager, request)
    batch = run(manager, request)
    assert connector.accepted == []
    manager.report_batch_to_connector(batch)
    first, first_groups = connector.accepted[0]
    assert (first.start, first.end) == (0, 64)
    assert connector.releases == [(first.reservation_id, 64, PROMPT_LEN)]
    assert all(slot >= 0 for _, slot in valid_page_slots(first_groups[0]))
    assert len(first_groups[0]) == 3

    manager.free_resources(request)
    rewind_context_after_cache_drop(request, TOKENS_PER_BLOCK)
    assert schedule(manager, request)
    manager.report_batch_to_connector(run(manager, request))

    replay, replay_groups = connector.accepted[0]
    assert replay.reservation_id != first.reservation_id
    assert (replay.start, replay.end) == (first.start, first.end)
    assert len(replay_groups[0]) == 3
    assert connector.queries == [(request.request_id, 0), (request.request_id, 0)]
    assert len(connector.allocs) == 2


def test_dispatched_load_retains_real_destination_pages(
    manager: KVCacheManagerV2, connector: FakeConnectorManager
) -> None:
    connector.prefix_reservations_enabled = True
    connector.load_async = True
    request = make_request()
    assert schedule(manager, request)
    manager.report_batch_to_connector(run(manager, request))
    connector.dispatched.add(request.request_id)
    pages_before = manager.get_page_indices_by_layer_group(request)

    with pytest.raises(RuntimeError, match="while request .* is loading"):
        manager.free_resources(request)

    other = make_request(request_id=2)
    assert schedule(manager, other)
    pages_after = manager.get_page_indices_by_layer_group(request)
    other_pages = manager.get_page_indices_by_layer_group(other)
    assert pages_after == pages_before
    for held, allocated in zip(pages_before, other_pages):
        assert {slot for _, slot in valid_page_slots(held)}.isdisjoint(
            slot for _, slot in valid_page_slots(allocated)
        )
    assert manager.kv_cache_map[request.request_id].is_active


def test_reserved_vswa_prefix_reports_live_group_ordinals(
    vswa_manager: KVCacheManagerV2, vswa_connector: FakeConnectorManager
) -> None:
    vswa_connector.prefix_reservations_enabled = True
    request = make_request(prompt_len=VSWA_PROMPT_LEN)
    assert schedule(vswa_manager, request)
    vswa_manager.report_batch_to_connector(run(vswa_manager, request))
    reservation, groups = vswa_connector.accepted[0]
    sliding, full = _sliding_and_full(vswa_manager)
    assert (reservation.start, reservation.end) == (0, VSWA_OFFER)
    assert len(groups[sliding]) == len(groups[full]) == VSWA_PROMPT_LEN // TOKENS_PER_BLOCK
    stale_end = (VSWA_OFFER + 1 - VSWA_WINDOW) // TOKENS_PER_BLOCK
    assert groups[sliding][:stale_end] == [BAD_PAGE_INDEX] * stale_end
    assert all(slot >= 0 for slot in groups[sliding][stale_end:])
    assert all(slot >= 0 for slot in groups[full])


def test_real_kv_pressure_rejects_reserved_prefix_without_transmission() -> None:
    connector = FakeConnectorManager(num_matched=64)
    connector.prefix_reservations_enabled = True
    manager = make_manager(
        connector,
        kv_cache_config=KvCacheConfig(max_tokens=64, enable_block_reuse=True),
    )
    blocker = None
    try:
        available_blocks = manager.get_num_free_blocks()
        assert available_blocks > 0
        request = make_request()
        assert manager.prepare_context(request)
        assert request.context_current_position == 64

        # Preparation can already hold pages. Fill the remaining pool until
        # the allocator refuses another block, preserving those real holds.
        blocker = manager.impl.create_kv_cache()
        assert blocker.resume(manager._stream.cuda_stream)
        for num_blocks in range(1, available_blocks + 2):
            if not blocker.resize(num_blocks * TOKENS_PER_BLOCK):
                break
        else:
            pytest.fail("The blocker did not exhaust the resolved KV pool")
        assert blocker.capacity > 0

        assert not manager.resize_context(request, 32)
        assert blocker.is_active
        assert connector.accepted == []
        assert connector.releases == [(1, 0, 64)]
        assert request.request_id not in manager.kv_cache_map
        assert request.context_current_position == 0
    finally:
        if blocker is not None:
            blocker.close()
        manager.shutdown()
