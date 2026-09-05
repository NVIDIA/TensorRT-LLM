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
"""Unit tests for the KV connector prefix on KVCacheManagerV2.

The connector is asked in ``prepare_resources``, on the batch the forward pass
will run, because that is where the V1 manager asks -- from
``onboardAndAllocateBlocks`` inside ``KVCacheManager.prepare_resources``, which
is downstream of every stage that can drop a request. V1's
*asked => scheduled => eventually request_finished* invariant therefore holds
verbatim, an offer is never abandoned, and no ``cancel_load`` is needed.

Two things do not carry over from V1 and are what these tests pin.

* **V1 allocates the whole prompt at the first chunk**, so an offer can never
  outrun its pages and ``setPrepopulatedPromptLen`` can shift the chunk window
  freely. V2 allocates per chunk, deliberately -- that is what chunked prefill
  is for -- so an offer beyond the chunk needs a bounded grow, and the grow can
  fail.
* **V1's local match is floored to whole shared blocks**; V2's
  ``num_committed_tokens`` is token-granular, so arithmetic V1 could not make
  negative, V2 can.

``FakeRequest`` reproduces ``LlmRequest``'s chunk arithmetic including
``setContextChunkSize``'s non-negative check and ``setPrepopulatedPromptLen``'s
block-alignment assertion, so a version of this code that violates either fails
here rather than only on hardware.
"""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2

TOKENS_PER_BLOCK = 32
PROMPT_LEN = 256


class FakeKvCache:
    """The parts of ``_KVCache`` the connector path touches.

    ``resize`` reproduces the real invariants -- history never decreases,
    history never exceeds capacity -- and ``grow_ok`` models
    ``OutOfPagesError``, which the real ``resize`` reports by returning False
    after restoring the state it unlocked.
    """

    def __init__(self, committed=0, capacity=None, grow_ok=True):
        self.num_committed_tokens = committed
        self.capacity = committed if capacity is None else capacity
        self.history_length = committed
        self.enable_swa_scratch_reuse = True
        self.is_active = True
        self.grow_ok = grow_ok
        self.resize_calls = []

    def resume(self, cuda_stream):
        self.is_active = True
        return True

    def suspend(self):
        self.is_active = False

    def resize(self, capacity, history_length=None):
        assert self.is_active, "resize on a suspended cache"
        self.resize_calls.append((capacity, history_length))
        growing = capacity is not None and capacity > self.capacity
        if growing and not self.grow_ok:
            return False
        if history_length is not None:
            if history_length < self.history_length:
                raise ValueError("History length cannot be decreased")
            if capacity is not None and capacity < history_length:
                raise ValueError("History length cannot be greater than capacity")
            self.history_length = history_length
        if capacity is not None:
            self.capacity = capacity
        return True


class FakeRequest:
    """``LlmRequest``'s context-chunk arithmetic, reproduced faithfully.

    The properties mirror ``llmRequest.h``: ``mContextChunkSize`` defaults to
    ``mPromptLen``, ``setContextChunkSize`` rejects a negative size and clamps
    to the remaining length, ``isFirstContextChunk`` is
    ``contextCurrentPosition == prepopulatedPromptLen``, and
    ``setPrepopulatedPromptLen`` floors a non-final chunk's end to a block
    boundary and then asserts it.
    """

    def __init__(self, request_id=0, prompt_len=PROMPT_LEN):
        self.py_request_id = request_id
        self.request_id = request_id
        self.prompt_len = prompt_len
        self.context_current_position = 0
        self.prepopulated_prompt_len = 0
        self.is_dummy = False
        self.is_generation_only_request = False
        self.is_disagg_generation_init_state = False
        self.py_num_connector_matched_tokens = 0
        self.py_connector_allocation_reported = False
        self.py_ctx_pre_resize_cap = None
        self.py_draft_tokens = []
        self._context_chunk_size = prompt_len

    @property
    def context_remaining_length(self):
        return self.prompt_len - self.context_current_position

    @property
    def context_chunk_size(self):
        return self._context_chunk_size

    @context_chunk_size.setter
    def context_chunk_size(self, size):
        assert size >= 0, f"The chunk size of context ({size}) can't be negative."
        self._context_chunk_size = min(size, self.context_remaining_length)

    @property
    def is_first_context_chunk(self):
        return self.context_current_position == self.prepopulated_prompt_len

    @property
    def is_last_context_chunk(self):
        return self.context_current_position + self.context_chunk_size == self.prompt_len

    def set_prepopulated_prompt_len(self, prepopulated_prompt_len, tokens_per_block):
        assert prepopulated_prompt_len < self.prompt_len, (
            f"prepopulatedPromptLen ({prepopulated_prompt_len}) >= promptLen ({self.prompt_len})"
        )
        self.prepopulated_prompt_len = prepopulated_prompt_len
        if prepopulated_prompt_len > 0:
            chunk_size = self.context_chunk_size
            if prepopulated_prompt_len + chunk_size < self.prompt_len:
                floored = (
                    (prepopulated_prompt_len + chunk_size) // tokens_per_block * tokens_per_block
                )
                chunk_size = floored - prepopulated_prompt_len
            self.context_current_position = prepopulated_prompt_len
            self.context_chunk_size = chunk_size
            if not self.is_last_context_chunk:
                assert (
                    self.context_current_position + self.context_chunk_size
                ) % tokens_per_block == 0, (
                    "the context position after the current chunk must be block-aligned"
                )


class FakeConnectorManager:
    """Records the calls the prefix path makes, in order."""

    def __init__(self, num_matched=0, load_async=False, add_sequence=True):
        self.num_matched = num_matched
        self.load_async = load_async
        self.add_sequence = add_sequence
        self.queries = []
        self.commits = []
        self.allocs = []
        self.alloc_by_group = []
        self.forgotten = []

    def query_num_new_matched_tokens(self, request, num_computed_tokens):
        self.queries.append((request.request_id, num_computed_tokens))
        return self.num_matched, self.load_async

    def commit_new_matched_tokens(self, request, num_tokens, load_kv_async):
        self.commits.append((request.request_id, num_tokens, load_kv_async))
        request.py_num_connector_matched_tokens = num_tokens

    def should_add_sequence(self, request):
        return self.add_sequence

    def reset_request_state(self, request):
        self.forgotten.append(request.request_id)

    def update_state_after_alloc(self, request, page_indices, by_layer_group=None):
        self.allocs.append((request.request_id, tuple(page_indices)))
        self.alloc_by_group.append(by_layer_group)

    def build_scheduler_output(self, scheduled_batch, kv_cache_manager):
        pass


def make_manager(connector, num_extra_kv_tokens=0, is_draft=False):
    """A ``KVCacheManagerV2`` carrying only the fields the prefix path reads.

    Constructing a real one needs a GPU and a pool allocation; what is under
    test is request/cache bookkeeping, so bypass ``__init__`` rather than
    turning this into an integration test. ``test_kv_connector_v2_prefix_real_manager``
    covers the same path against real pools.
    """
    manager = object.__new__(KVCacheManagerV2)
    manager.kv_connector_manager = connector
    manager.is_draft = is_draft
    manager.tokens_per_block = TOKENS_PER_BLOCK
    manager.num_extra_kv_tokens = num_extra_kv_tokens
    manager.kv_cache_map = {}
    manager.enable_block_reuse = True
    manager.conversation_manager = None
    manager._stream = SimpleNamespace(cuda_stream=0)
    # One layer group, the shape every non-VSWA, non-hybrid model has. The real
    # accessor reads `impl.layer_grouping`, which only a pool allocation fills
    # in, so it is stubbed here rather than bypassed -- `_run_kv_connector_hooks`
    # derives the flat list from this and hands the connector both forms.
    manager.get_page_indices_by_layer_group = lambda request: [[]]
    return manager


def scheduled(*requests):
    batch = SimpleNamespace(context_requests=list(requests), reset_calls=0)

    def reset_context_requests():
        batch.reset_calls += 1

    batch.reset_context_requests = reset_context_requests
    return batch


def serve(manager, req, kv_cache):
    """Drive the real ``_apply_connector_matched_prefix`` for one scheduled request."""
    manager.kv_cache_map[req.py_request_id] = kv_cache
    req.context_current_position = kv_cache.num_committed_tokens
    req.set_prepopulated_prompt_len(kv_cache.num_committed_tokens, TOKENS_PER_BLOCK)
    return manager._apply_connector_matched_prefix(req)


class TestAskTiming:
    """The ask happens on the final batch, and only there.

    This is the whole of A0: V1 guarantees *asked => scheduled* because its
    query is downstream of every drop, and V2 regains the guarantee by asking
    in the same place.
    """

    def test_the_scheduling_pass_does_not_ask(self):
        connector = FakeConnectorManager(num_matched=64)
        manager = make_manager(connector)
        req = FakeRequest()
        kv_cache = FakeKvCache(committed=64, capacity=PROMPT_LEN)
        manager.kv_cache_map[req.py_request_id] = kv_cache

        assert manager._prepare_context_impl(req)

        assert connector.queries == []
        assert req.context_current_position == 64

    def test_the_connector_is_asked_once_per_allocation(self):
        """Serving leaves ``is_first_context_chunk`` true, by design -- it is
        ``context_current_position == prepopulated_prompt_len`` and
        ``set_prepopulated_prompt_len`` moves both. So the first-chunk test
        cannot be the ask-once guard, and a re-entry before the forward pass
        would otherwise take remote ownership twice for one allocation."""
        connector = FakeConnectorManager(num_matched=64)
        manager = make_manager(connector)
        req = FakeRequest()
        kv_cache = FakeKvCache(committed=0, capacity=PROMPT_LEN)
        manager.kv_cache_map[req.py_request_id] = kv_cache

        manager._run_kv_connector_hooks(scheduled(req))
        assert req.is_first_context_chunk
        manager._run_kv_connector_hooks(scheduled(req))

        assert len(connector.queries) == 1

    def test_a_second_pass_reports_one_allocation(self):
        """``should_add_sequence`` is not enough on its own.

        It only goes false once an asynchronous load has *completed*. A batch
        dropped between ``prepare_resources`` and the forward pass -- the second
        ``_can_queue``, which a parked request can flip -- brings every context
        request back on its first chunk with the predicate still true. The ask
        is idempotent by memo; a second ``update_state_after_alloc`` would
        report the same pages twice.
        """
        connector = FakeConnectorManager(num_matched=64)
        manager = make_manager(connector)
        req = FakeRequest()
        manager.kv_cache_map[req.py_request_id] = FakeKvCache(committed=0, capacity=PROMPT_LEN)

        manager._run_kv_connector_hooks(scheduled(req))
        manager._run_kv_connector_hooks(scheduled(req))

        assert len(connector.queries) == 1
        assert len(connector.allocs) == 1

    def test_an_excluded_request_is_still_reported_once(self):
        """V1 reports the allocation for every first-chunk sequence it adds,
        including the kinds the connector is never asked about."""
        connector = FakeConnectorManager(num_matched=64)
        manager = make_manager(connector)
        req = FakeRequest()
        req.is_disagg_generation_init_state = True
        manager.kv_cache_map[req.py_request_id] = FakeKvCache(committed=0, capacity=PROMPT_LEN)

        manager._run_kv_connector_hooks(scheduled(req))
        manager._run_kv_connector_hooks(scheduled(req))

        assert connector.queries == []
        assert len(connector.allocs) == 1

    def test_a_destroyed_allocation_is_asked_again(self):
        """V1-faithful: a destructive pause replays ``addSequence`` and with it
        ``getNumNewMatchedTokens``, because the pages the first answer described
        are gone."""
        connector = FakeConnectorManager(num_matched=64)
        manager = make_manager(connector)
        req = FakeRequest()
        manager.kv_cache_map[req.py_request_id] = FakeKvCache(committed=0, capacity=PROMPT_LEN)

        manager._run_kv_connector_hooks(scheduled(req))
        req.py_connector_allocation_reported = False  # what free_resources does
        req.context_current_position = 0
        req.prepopulated_prompt_len = 0
        manager._run_kv_connector_hooks(scheduled(req))

        assert len(connector.queries) == 2

    def test_a_completed_async_load_is_not_asked_again(self):
        """``should_add_sequence`` is V1's re-entry gate, used the same way.

        A request whose asynchronous load finished re-enters the batch still on
        its first context chunk, with the same pages and nothing left to load.
        Asking again would take ownership twice for one allocation.
        """
        connector = FakeConnectorManager(num_matched=64, add_sequence=False)
        manager = make_manager(connector)
        req = FakeRequest()
        manager.kv_cache_map[req.py_request_id] = FakeKvCache(committed=0, capacity=PROMPT_LEN)

        manager._run_kv_connector_hooks(scheduled(req))

        assert connector.queries == []
        assert connector.allocs == []


class TestOfferIsClamped:
    def test_the_last_prompt_token_stays_local(self):
        """The first generation step consumes its activations, so it must be computed."""
        connector = FakeConnectorManager(num_matched=PROMPT_LEN)
        manager = make_manager(connector)
        req = FakeRequest()

        assert serve(manager, req, FakeKvCache(committed=0, capacity=PROMPT_LEN))

        assert req.context_current_position == PROMPT_LEN - 1
        assert req.context_chunk_size == 1
        assert connector.commits == [(0, PROMPT_LEN - 1, False)]

    def test_the_query_is_anchored_at_the_local_match(self):
        """The connector is asked what it can serve *past* what the radix tree
        already holds, and the offer is an extent measured from there."""
        connector = FakeConnectorManager(num_matched=16)
        manager = make_manager(connector)
        req = FakeRequest()

        assert serve(manager, req, FakeKvCache(committed=128, capacity=PROMPT_LEN))

        assert connector.queries == [(0, 128)]
        assert req.context_current_position == 144
        assert connector.commits == [(0, 16, False)]

    def test_an_empty_offer_touches_nothing(self):
        connector = FakeConnectorManager(num_matched=0)
        manager = make_manager(connector)
        req = FakeRequest()
        kv_cache = FakeKvCache(committed=64, capacity=PROMPT_LEN)

        assert not serve(manager, req, kv_cache)

        assert req.context_current_position == 64
        assert kv_cache.resize_calls == []
        assert connector.commits == [(0, 0, False)]


class TestUnchunkedIsPureShrink:
    """With chunking off, ``resize_context`` already covered the whole prompt.

    The offer is inside that by construction, so the chunk's end stays where
    the scheduler put it, its start moves up, and no page is allocated.
    """

    def test_the_chunk_end_does_not_move(self):
        connector = FakeConnectorManager(num_matched=96)
        manager = make_manager(connector)
        req = FakeRequest()
        kv_cache = FakeKvCache(committed=32, capacity=PROMPT_LEN)

        assert serve(manager, req, kv_cache)

        assert req.context_current_position == 128
        assert req.context_current_position + req.context_chunk_size == PROMPT_LEN
        assert req.context_chunk_size == PROMPT_LEN - 128

    def test_no_capacity_is_allocated(self):
        connector = FakeConnectorManager(num_matched=96)
        manager = make_manager(connector)
        kv_cache = FakeKvCache(committed=32, capacity=PROMPT_LEN)

        serve(manager, FakeRequest(), kv_cache)

        assert kv_cache.capacity == PROMPT_LEN
        assert kv_cache.resize_calls == [(PROMPT_LEN, 128)]

    def test_history_is_raised_to_the_served_position(self):
        """The sole input to the stale-range computation, so a sliding-window
        layer group does not keep a page per served block."""
        connector = FakeConnectorManager(num_matched=96)
        manager = make_manager(connector)
        kv_cache = FakeKvCache(committed=32, capacity=PROMPT_LEN)

        serve(manager, FakeRequest(), kv_cache)

        assert kv_cache.history_length == 128

    def test_the_request_stays_on_its_first_chunk(self):
        connector = FakeConnectorManager(num_matched=96)
        manager = make_manager(connector)
        req = FakeRequest()

        serve(manager, req, FakeKvCache(committed=32, capacity=PROMPT_LEN))

        assert req.is_first_context_chunk


class TestChunkedInsideTheChunk:
    def test_the_scheduler_s_chunk_end_is_preserved(self):
        connector = FakeConnectorManager(num_matched=32)
        manager = make_manager(connector)
        req = FakeRequest()
        kv_cache = FakeKvCache(committed=0, capacity=64)
        req.context_chunk_size = 64  # the scheduler's choice this iteration

        assert serve(manager, req, kv_cache)

        assert req.context_current_position == 32
        assert req.context_current_position + req.context_chunk_size == 64
        assert kv_cache.capacity == 64


class TestChunkedBeyondTheChunk:
    """Only reachable with chunked prefill, and the only place a grow happens.

    The offer costs pages whatever the query timing -- the connector writes real
    KV into them -- so the chunk window shifts forward and the allocation grows
    by the offer plus the compute the scheduler budgeted. Bounded by one chunk,
    against V1 which allocates the whole prompt.
    """

    def test_the_window_shifts_and_the_allocation_grows(self):
        connector = FakeConnectorManager(num_matched=128)
        manager = make_manager(connector)
        req = FakeRequest()
        kv_cache = FakeKvCache(committed=0, capacity=64)
        req.context_chunk_size = 64

        assert serve(manager, req, kv_cache)

        assert req.context_current_position == 128
        assert req.context_chunk_size == 64
        assert kv_cache.capacity == 192

    def test_a_non_final_chunk_end_stays_block_aligned(self):
        """V1's rule, for V1's reason: otherwise the next chunk fragments the cache."""
        connector = FakeConnectorManager(num_matched=100)
        manager = make_manager(connector)
        req = FakeRequest()
        kv_cache = FakeKvCache(committed=0, capacity=48)
        req.context_chunk_size = 48

        assert serve(manager, req, kv_cache)

        assert req.context_current_position == 100
        end = req.context_current_position + req.context_chunk_size
        assert end % TOKENS_PER_BLOCK == 0
        assert end == 128

    def test_a_failed_grow_falls_back_to_the_pages_that_exist(self):
        connector = FakeConnectorManager(num_matched=128)
        manager = make_manager(connector)
        req = FakeRequest()
        kv_cache = FakeKvCache(committed=0, capacity=64, grow_ok=False)
        req.context_chunk_size = 64

        assert serve(manager, req, kv_cache)

        # The largest whole block inside the 64 tokens the pages already hold
        # that still leaves the forward pass a chunk to compute.
        assert req.context_current_position == 32
        assert req.context_chunk_size == 32
        assert kv_cache.capacity == 64

    def test_a_failed_grow_commits_only_what_was_honoured(self):
        """Over-reporting would point the connector at an offset it has no page for.

        The unconsumed tail of the offer is recomputed locally and released at
        ``request_finished``, which is what V1 does when block reuse is off and
        it applies none of an offer it asked for.
        """
        connector = FakeConnectorManager(num_matched=128)
        manager = make_manager(connector)
        req = FakeRequest()
        kv_cache = FakeKvCache(committed=0, capacity=64, grow_ok=False)
        req.context_chunk_size = 64

        serve(manager, req, kv_cache)

        assert connector.commits == [(0, 32, False)]

    def test_an_offer_that_reaches_the_chunk_end_exactly_shifts(self):
        """Honouring it in place would leave a zero-token context chunk."""
        connector = FakeConnectorManager(num_matched=64)
        manager = make_manager(connector)
        req = FakeRequest()
        kv_cache = FakeKvCache(committed=0, capacity=64)
        req.context_chunk_size = 64

        assert serve(manager, req, kv_cache)

        assert req.context_current_position == 64
        assert req.context_chunk_size == 64
        assert kv_cache.capacity == 128

    def test_a_fallback_that_cannot_advance_honours_nothing(self):
        connector = FakeConnectorManager(num_matched=64)
        manager = make_manager(connector)
        req = FakeRequest()
        # One block of pages: the offer needs a grow, the grow fails, and the
        # capped fallback lands back on the local match.
        kv_cache = FakeKvCache(committed=0, capacity=TOKENS_PER_BLOCK, grow_ok=False)
        req.context_chunk_size = TOKENS_PER_BLOCK

        assert not serve(manager, req, kv_cache)

        assert req.context_current_position == 0
        assert connector.commits == [(0, 0, False)]

    def test_a_grow_that_cannot_leave_a_token_to_compute_is_refused(self):
        connector = FakeConnectorManager(num_matched=PROMPT_LEN)
        manager = make_manager(connector)
        req = FakeRequest()
        # The whole prompt is already committed but for its last token, so the
        # clamped offer lands exactly on the chunk end and nothing is left.
        kv_cache = FakeKvCache(committed=PROMPT_LEN - 1, capacity=PROMPT_LEN, grow_ok=False)
        req.context_chunk_size = 1

        assert not serve(manager, req, kv_cache)

        assert req.context_current_position == PROMPT_LEN - 1
        assert connector.commits == [(0, 0, False)]


class TestExtraKvTokens:
    def test_the_grow_target_reserves_the_extra_tokens(self):
        connector = FakeConnectorManager(num_matched=128)
        manager = make_manager(connector, num_extra_kv_tokens=4)
        req = FakeRequest()
        kv_cache = FakeKvCache(committed=0, capacity=64)
        req.context_chunk_size = 64

        serve(manager, req, kv_cache)

        assert kv_cache.capacity == 192 + 4


class TestExclusions:
    @pytest.mark.parametrize(
        "attribute",
        ["is_dummy", "is_generation_only_request", "is_disagg_generation_init_state"],
    )
    def test_request_kinds_that_are_never_asked(self, attribute):
        connector = FakeConnectorManager(num_matched=64)
        manager = make_manager(connector)
        req = FakeRequest()
        setattr(req, attribute, True)

        assert not serve(manager, req, FakeKvCache(committed=0, capacity=PROMPT_LEN))

        assert connector.queries == []
        assert connector.commits == []

    def test_no_connector_is_a_no_op(self):
        manager = make_manager(None)
        req = FakeRequest()

        assert not serve(manager, req, FakeKvCache(committed=0, capacity=PROMPT_LEN))

        assert req.context_current_position == 0

    def test_the_draft_manager_never_asks(self):
        connector = FakeConnectorManager(num_matched=64)
        manager = make_manager(connector, is_draft=True)
        req = FakeRequest()

        assert not serve(manager, req, FakeKvCache(committed=0, capacity=PROMPT_LEN))

        assert connector.queries == []


class TestAsyncLoad:
    def test_the_async_flag_reaches_the_commit(self):
        connector = FakeConnectorManager(num_matched=64, load_async=True)
        manager = make_manager(connector)

        serve(manager, FakeRequest(), FakeKvCache(committed=0, capacity=PROMPT_LEN))

        assert connector.commits == [(0, 64, True)]

    def test_the_async_hold_survives_an_under_honoured_offer(self):
        """The connector has already started the transfer, so the request must
        still be parked even though the runtime took less than it offered."""
        connector = FakeConnectorManager(num_matched=128, load_async=True)
        manager = make_manager(connector)
        req = FakeRequest()
        req.context_chunk_size = 64

        serve(manager, req, FakeKvCache(committed=0, capacity=64, grow_ok=False))

        assert connector.commits == [(0, 32, True)]


class TestBatchBookkeeping:
    def test_a_served_prefix_rebuilds_the_chunking_split(self):
        """A shift can carry a request onto its last chunk, which moves it
        between ``context_requests_chunking`` and ``context_requests_last_chunk``
        -- the lists ``build_scheduler_output`` walks."""
        connector = FakeConnectorManager(num_matched=64)
        manager = make_manager(connector)
        req = FakeRequest()
        manager.kv_cache_map[req.py_request_id] = FakeKvCache(committed=0, capacity=PROMPT_LEN)
        batch = scheduled(req)

        manager._run_kv_connector_hooks(batch)

        assert batch.reset_calls == 1

    def test_an_unserved_batch_leaves_the_split_alone(self):
        connector = FakeConnectorManager(num_matched=0)
        manager = make_manager(connector)
        req = FakeRequest()
        manager.kv_cache_map[req.py_request_id] = FakeKvCache(committed=0, capacity=PROMPT_LEN)
        batch = scheduled(req)

        manager._run_kv_connector_hooks(batch)

        assert batch.reset_calls == 0

    def test_allocation_is_reported_after_the_prefix_is_served(self):
        """The pages the connector may write into include the ones the grow
        added, so the report has to follow the serve."""
        seen = []
        connector = FakeConnectorManager(num_matched=128)
        manager = make_manager(connector)
        req = FakeRequest()
        req.context_chunk_size = 64
        kv_cache = FakeKvCache(committed=0, capacity=64)
        manager.kv_cache_map[req.py_request_id] = kv_cache
        manager.get_page_indices_by_layer_group = lambda request: seen.append(
            kv_cache.capacity
        ) or [[]]

        manager._run_kv_connector_hooks(scheduled(req))

        assert seen == [192]
        # The per-layer-group form is what reaches the connector. Asserting it
        # arrives keeps the hook wired to the accessor the manager actually
        # implements: overriding only the flat one leaves the real accessor to
        # run against a manager with no pools, which raises.
        assert connector.alloc_by_group == [[[]]]


class TestReEntryAfterAServe:
    """A served request that comes back before it runs.

    The reachable case is an asynchronous load: the connector reports it will
    transfer in the background, the runtime parks the request out of the batch,
    and the scheduler runs ``prepare_context`` on it again when it returns.
    """

    def _serve_then_re_enter(self, load_async=True):
        connector = FakeConnectorManager(num_matched=64, load_async=load_async)
        manager = make_manager(connector)
        req = FakeRequest()
        manager.kv_cache_map[req.py_request_id] = FakeKvCache(committed=0, capacity=PROMPT_LEN)
        assert manager._prepare_context_impl(req)
        manager._run_kv_connector_hooks(scheduled(req))
        served_position = req.context_current_position
        assert manager._prepare_context_impl(req)
        return req, served_position

    def test_the_chunk_still_spans_to_the_end_of_the_prompt(self):
        """The position and the chunk have to be re-derived together.

        ``prepare_context`` re-derives the position from the radix tree, which
        undoes the serve. Leaving the chunk narrowed by the offer makes the pair
        describe two different ranges: ``position + chunk`` stops reaching
        ``prompt_len``, so a non-chunked request silently becomes a chunked one
        and the forward pass is handed inconsistent metadata.
        """
        req, served_position = self._serve_then_re_enter()

        assert served_position == 64, "the serve must have moved the position"
        assert req.context_current_position + req.context_chunk_size == req.prompt_len
        assert req.is_last_context_chunk

    def test_the_same_holds_for_a_synchronous_serve(self):
        req, _ = self._serve_then_re_enter(load_async=False)

        assert req.context_current_position + req.context_chunk_size == req.prompt_len


class TestSwaScratchReuse:
    def test_scratch_reuse_is_disabled_before_the_scheduler_can_take_slots(self):
        """The flag has to be cleared in ``prepare_context``: by the time the
        connector is asked, ``resize_context`` has already run and may have
        taken scratch slots for blocks the connector is about to write."""
        connector = FakeConnectorManager(num_matched=64)
        manager = make_manager(connector)
        req = FakeRequest()
        kv_cache = FakeKvCache(committed=0, capacity=PROMPT_LEN)
        manager.kv_cache_map[req.py_request_id] = kv_cache

        assert manager._prepare_context_impl(req)

        assert kv_cache.enable_swa_scratch_reuse is False

    def test_scratch_reuse_survives_without_a_connector(self):
        manager = make_manager(None)
        req = FakeRequest()
        kv_cache = FakeKvCache(committed=0, capacity=PROMPT_LEN)
        manager.kv_cache_map[req.py_request_id] = kv_cache

        assert manager._prepare_context_impl(req)

        assert kv_cache.enable_swa_scratch_reuse is True
