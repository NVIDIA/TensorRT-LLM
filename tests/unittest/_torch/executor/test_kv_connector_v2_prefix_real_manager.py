# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The KV connector prefix against a *real* ``KVCacheManagerV2``.

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

import pytest
import torch

import tensorrt_llm
import tensorrt_llm.bindings
from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, SamplingConfig
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import BAD_PAGE_INDEX

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
        self.num_matched = num_matched
        self.load_async = load_async
        self.queries = []
        self.commits = []
        self.allocs = []
        self.allocs_by_group = []
        self.forgotten = []

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
    multimodal alignment, or be dropped when the batch cannot be queued. On the
    V1 manager none of that can strand an offer, because V1 asks after all of
    it. Neither can it here.
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


def test_freeing_the_allocation_makes_the_request_askable_again(manager, connector):
    """V1-faithful: a destructive pause replays ``addSequence`` and with it the
    query, because the pages the first answer described are gone."""
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


def test_no_connector_leaves_prepare_resources_inert(connector):
    torch.cuda.init()
    mgr = make_manager(connector, kv_connector_manager=None)
    try:
        request = make_request()
        assert schedule(mgr, request)
        run(mgr, request)
        assert request.context_current_position == 0
        assert connector.queries == []
    finally:
        mgr.shutdown()
        del mgr
        gc.collect()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Variable sliding-window attention.
#
# Two distinct windows over two layers give two layer groups, which is the only
# shape where the per-layer-group callbacks are reachable and the only shape
# where `_stale_block_end` has to pick a window rather than being handed the
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
    windows = manager._window_size_by_layer_group()
    assert len(windows) == 2, f"expected two layer groups, got {windows}"
    return windows.index(VSWA_WINDOW), windows.index(None)


def test_window_size_is_read_per_layer_group(vswa_manager):
    """Each group carries its own window, and the pair is not interchangeable.

    Everything downstream -- the masking boundary, what a connector is offered
    to save -- is derived from this list by index, so an off-by-one here is
    silent and total.
    """
    windows = vswa_manager._window_size_by_layer_group()

    assert len(windows) == 2
    assert sorted(windows, key=lambda w: (w is None, w)) == [VSWA_WINDOW, None]

    layers = vswa_manager.kv_cache_manager_py_config.layers
    for layer_group_id, local_layer_ids in enumerate(vswa_manager.impl.layer_grouping):
        for local_layer_id in local_layer_ids:
            assert windows[layer_group_id] == layers[int(local_layer_id)].window_size


def test_stale_block_end_uses_each_group_s_own_window(vswa_manager):
    """The full-attention group must never be masked, whatever the history.

    Computed here from the window rather than copied from the implementation:
    a test that reuses `_stale_block_end` to predict `_stale_block_end` cannot
    catch a wrong window being selected for the group.
    """
    sliding, full = _sliding_and_full(vswa_manager)
    tokens_per_block = vswa_manager.tokens_per_block

    for history_length in (0, 32, 64, 96, 128, 200):
        expected = max(0, (history_length + 1 - VSWA_WINDOW) // tokens_per_block)
        assert vswa_manager._stale_block_end(sliding, history_length) == expected
        assert vswa_manager._stale_block_end(full, history_length) == 0, (
            "a full-attention group has no stale range; masking it would hide "
            "pages the connector is entitled to save"
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


def test_a_single_window_still_reports_the_flat_list(connector):
    """Sibling check: the same code path with one group keeps the V1 shape.

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
