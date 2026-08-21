# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the KV connector prefix phases on KVCacheManagerV2.

KVCacheManagerV2's scheduling pass is speculative: a request can be prepared
and then dropped in the same iteration at the token budget, at
``resize_context``, at multimodal alignment or at cross attention, and retried
later. The connector ABC, meanwhile, promises exactly one
``get_num_new_matched_tokens`` per request and lets connectors take ownership
of remote blocks inside it.

Reconciling those two is the whole point of the three phases exercised here,
and none of it is observable from the end-to-end connector tests, which never
defer a request. So these drive the phases directly against a stub cache.

The stub cache models V2's own two-phase split, because that split is what
forces the connector phases apart: ``_create_kv_cache`` returns a **suspended**
cache, ``_KVCache.resize`` asserts the cache is ACTIVE, and
``_resume_and_restore`` is what activates it. So the offer has to be taken --
and the context position advanced over it -- *before* the resume, while the
capacity and history that back it can only be reserved *after*. ``FakeKvCache``
therefore starts suspended and refuses ``resize`` until resumed, so a version of
this code that folded the two phases into one would fail here rather than only
on hardware.
"""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2

TOKENS_PER_BLOCK = 32
PROMPT_LEN = 256


class FakeKvCache:
    """The parts of ``_KVCache`` the connector phases touch.

    Starts suspended, as ``_create_kv_cache`` leaves it. ``resize`` reproduces
    the real assertions -- ACTIVE status, history never decreasing, history
    never above capacity -- since those are exactly what dictate when each
    phase may run and how it must call ``resize``.
    """

    def __init__(self, committed=0, resize_ok=True, active=False, trace=None):
        self.num_committed_tokens = committed
        self.capacity = committed
        self.history_length = committed
        self.enable_swa_scratch_reuse = True
        self.resize_ok = resize_ok
        self.is_active = active
        self.resize_calls = []
        self.trace = trace if trace is not None else []

    def resume(self, cuda_stream):
        self.is_active = True
        self.trace.append("resume")
        return True

    def suspend(self):
        self.is_active = False
        self.trace.append("suspend")

    def resize(self, capacity, history_length=None):
        # `_KVCache.resize` asserts ACTIVE. A cache that has just been created,
        # or that was suspended when its request was deferred, is not.
        assert self.is_active, "resize on a suspended cache"
        self.resize_calls.append((capacity, history_length))
        self.trace.append(("resize", capacity, history_length))
        if not self.resize_ok:
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
    def __init__(self, request_id=0, prompt_len=PROMPT_LEN):
        self.py_request_id = request_id
        self.request_id = request_id
        self.prompt_len = prompt_len
        self.context_current_position = 0
        self.prepopulated_prompt_len = 0
        self.is_first_context_chunk = True
        self.is_dummy = False
        self.is_generation_only_request = False
        self.is_disagg_generation_init_state = False
        self.py_num_connector_matched_tokens = 0
        self.py_connector_prefix_start = None
        self.py_connector_prefix_end = None
        self.py_connector_load_async = False
        self.py_connector_delivered = False

    def set_prepopulated_prompt_len(self, prepopulated_prompt_len, tokens_per_block):
        assert prepopulated_prompt_len < self.prompt_len, (
            f"prepopulatedPromptLen ({prepopulated_prompt_len}) >= promptLen ({self.prompt_len})"
        )
        self.prepopulated_prompt_len = prepopulated_prompt_len
        if prepopulated_prompt_len > 0:
            self.context_current_position = prepopulated_prompt_len


class FakeConnectorManager:
    """Records the calls the phases make, in order."""

    def __init__(self, num_matched=0, load_async=False, trace=None):
        self.num_matched = num_matched
        self.load_async = load_async
        self.queries = []
        self.commits = []
        self.cancels = []
        self.trace = trace if trace is not None else []

    def query_num_new_matched_tokens(self, request, num_computed_tokens):
        self.queries.append((request.request_id, num_computed_tokens))
        self.trace.append(("query", num_computed_tokens))
        return self.num_matched, self.load_async

    def commit_new_matched_tokens(self, request, num_tokens, load_kv_async):
        self.commits.append((request.request_id, num_tokens, load_kv_async))
        self.trace.append(("commit", num_tokens))
        request.py_num_connector_matched_tokens = num_tokens

    def cancel_load(self, request, start, end):
        self.cancels.append((request.request_id, start, end))
        self.trace.append(("cancel", start, end))


def make_manager(connector):
    """A KVCacheManagerV2 with only the fields the connector phases read.

    Constructing a real one needs a GPU and a pool allocation; the phases under
    test are pure request/cache bookkeeping, so bypass __init__ rather than
    turning this into an integration test.
    """
    manager = object.__new__(KVCacheManagerV2)
    manager.kv_connector_manager = connector
    manager.is_draft = False
    manager.tokens_per_block = TOKENS_PER_BLOCK
    manager.kv_cache_map = {}
    # Read by `_prepare_context_impl` on the first-chunk path, so that the
    # ordering tests below can drive the real thing rather than a re-statement
    # of it.
    manager.enable_block_reuse = True
    manager.conversation_manager = None
    manager._stream = SimpleNamespace(cuda_stream=0)
    # Page-index buffer plumbing: needs the real IndexMapper and pool tensors,
    # and has no bearing on which phase runs when.
    manager._restore_page_index_bufs = lambda request_id, kv_cache: None
    return manager


def prepare(manager, req, kv_cache):
    """One scheduling attempt, driving the real ``_prepare_context_impl``.

    Seeding ``kv_cache_map`` skips the ``_create_kv_cache`` branch, which needs
    a GPU; everything after it -- the local-match anchor, phase 1, the resume,
    phase 2 -- is the production code, so the phase ordering is observed rather
    than restated.
    """
    manager.kv_cache_map[req.py_request_id] = kv_cache
    assert manager._prepare_context_impl(req)
    # A memoised re-read of the position the attempt settled on, not a second
    # ask -- `py_connector_prefix_end` is set by now, so the connector is not
    # consulted again. `TestAskOnce` pins that.
    return manager._connector_prefix_position(req, kv_cache)


class TestTwoPhaseSplit:
    """The connector phases are split because V2's own allocation is.

    V2 separates *match and take ownership*, which needs only the token
    sequence, from *become resident on GPU*, which needs slots and can fail.
    `_create_kv_cache` returns a suspended cache and `_KVCache.resize` asserts
    ACTIVE, so the two connector steps straddle `_resume_and_restore`. Every
    test in this class fails if they are folded into one.
    """

    def test_the_offer_is_taken_before_the_cache_is_resident(self):
        trace = []
        connector = FakeConnectorManager(num_matched=64, trace=trace)
        manager = make_manager(connector)
        req = FakeRequest()
        kv_cache = FakeKvCache(committed=32, trace=trace)

        prepare(manager, req, kv_cache)

        assert trace == [("query", 32), "resume", ("resize", 96, 96)], (
            "the connector is asked while the cache is still suspended, and the "
            "capacity backing its answer is reserved only once it is active"
        )

    def test_phase_one_touches_no_residency(self):
        """Asking needs the token sequence and nothing else.

        This is what makes it safe to ask during a speculative scheduling pass:
        no slot is claimed, so a request that is then dropped has cost nothing
        locally.
        """
        connector = FakeConnectorManager(num_matched=64)
        manager = make_manager(connector)
        req = FakeRequest()
        kv_cache = FakeKvCache(committed=32)

        position = manager._connector_prefix_position(req, kv_cache)

        assert position == 96
        assert kv_cache.is_active is False
        assert kv_cache.resize_calls == []
        assert kv_cache.capacity == 32

    def test_phase_two_refuses_a_suspended_cache(self):
        """The assertion that forces the split to exist.

        `_KVCache.resize` asserts ACTIVE, so reserving before the resume -- the
        shape a single merged step would have -- fails on the first request
        that has anything to reserve.
        """
        connector = FakeConnectorManager(num_matched=64)
        manager = make_manager(connector)
        req = FakeRequest()
        kv_cache = FakeKvCache(committed=32)

        position = manager._connector_prefix_position(req, kv_cache)

        with pytest.raises(AssertionError, match="suspended"):
            manager._reserve_connector_prefix(req, kv_cache, position)

    def test_a_deferred_request_is_resumed_again_before_reserving(self):
        """A deferred first chunk is suspended, so the split holds every attempt.

        `resize_context` suspends the cache when it cannot grow it, so the next
        attempt starts from the suspended state again -- and must still ask
        nothing, resume, and only then reserve.
        """
        trace = []
        connector = FakeConnectorManager(num_matched=64, trace=trace)
        manager = make_manager(connector)
        req = FakeRequest()
        kv_cache = FakeKvCache(committed=32, trace=trace)

        prepare(manager, req, kv_cache)
        kv_cache.suspend()
        trace.clear()

        prepare(manager, req, kv_cache)

        assert trace == ["resume", ("resize", 96, 96)]
        assert connector.queries == [(0, 32)]


class TestAskOnce:
    def test_deferred_request_is_not_asked_again(self):
        connector = FakeConnectorManager(num_matched=64)
        manager = make_manager(connector)
        req = FakeRequest()
        kv_cache = FakeKvCache(committed=0)

        # Iteration 1: prepared, then dropped before it reached the batch.
        prepare(manager, req, kv_cache)
        # Iteration 2: prepared again.
        prepare(manager, req, kv_cache)

        assert connector.queries == [(0, 0)], (
            "the connector takes ownership of remote blocks inside the query, "
            "so a second one double-pins them and breaks the ABC's "
            "at-most-once promise"
        )

    def test_delivering_records_once_for_the_iteration_that_ran(self):
        connector = FakeConnectorManager(num_matched=64)
        manager = make_manager(connector)
        req = FakeRequest()
        kv_cache = FakeKvCache(committed=0)

        prepare(manager, req, kv_cache)
        assert connector.commits == []

        manager._deliver_connector_prefix(req)
        assert connector.commits == [(0, 64, False)]

    def test_unasked_request_delivers_nothing(self):
        connector = FakeConnectorManager()
        manager = make_manager(connector)
        req = FakeRequest()

        manager._deliver_connector_prefix(req)

        assert connector.commits == []
        assert connector.cancels == []


class TestOfferEndIsAbsolute:
    def test_position_does_not_overshoot_when_the_local_match_grows(self):
        """The reason the offer end is memoised rather than the returned delta.

        A deferred request re-derives its local match from the radix tree, and
        another request's commit may have grown it in the meantime. Adding the
        delta to the new match would set the position past the union of what is
        locally computed and what the connector holds, leaving the tokens in
        between neither computed nor loaded -- silently garbage KV.
        """
        connector = FakeConnectorManager(num_matched=64)
        manager = make_manager(connector)
        req = FakeRequest()
        kv_cache = FakeKvCache(committed=0)

        prepare(manager, req, kv_cache)
        assert req.context_current_position == 64

        # Another request committed the first 32 tokens while this one waited.
        kv_cache.num_committed_tokens = 32
        position = prepare(manager, req, kv_cache)

        assert position == 64, "0 + 64 offered, so 64 -- not 32 + 64"
        assert req.context_current_position == 64

    def test_position_follows_the_local_match_when_it_overtakes_the_offer(self):
        connector = FakeConnectorManager(num_matched=64)
        manager = make_manager(connector)
        req = FakeRequest()
        kv_cache = FakeKvCache(committed=0)

        prepare(manager, req, kv_cache)

        kv_cache.num_committed_tokens = 128
        position = prepare(manager, req, kv_cache)

        assert position == 128
        assert req.context_current_position == 128
        assert connector.queries == [(0, 0)]

    def test_offer_end_is_clamped_below_the_prompt(self):
        """A connector holding the whole prompt is the steady state of a repeat.

        The last prompt position must be computed locally regardless, since the
        first generation step consumes its activations, so the offer is clamped
        rather than rejected -- and the *clamped* delta is what gets recorded,
        or the connector is pointed at the wrong offset.
        """
        connector = FakeConnectorManager(num_matched=PROMPT_LEN)
        manager = make_manager(connector)
        req = FakeRequest()
        kv_cache = FakeKvCache(committed=0)

        prepare(manager, req, kv_cache)
        manager._deliver_connector_prefix(req)

        assert req.py_connector_prefix_end == PROMPT_LEN - 1
        assert req.context_current_position == PROMPT_LEN - 1
        assert connector.commits == [(0, PROMPT_LEN - 1, False)]
        # The clamp is the one place an offer shrinks without passing through
        # phase 3 or the release path, both of which read the clamped end. So
        # it has to hand the remainder back itself or the connector keeps
        # ownership of it for the life of the process.
        assert connector.cancels == [(0, PROMPT_LEN - 1, PROMPT_LEN)]


class TestWriteRangeAtDelivery:
    def test_subsumed_head_is_handed_back_and_not_transferred(self):
        """``[start, committed)`` is locally owned by the time we deliver.

        Those are committed pages in the radix tree, potentially shared with
        other live requests, and V2's rule is that only the owning request
        writes into its own padding. So the connector is told to transfer only
        what is still privately owned, and the rest is cancelled.
        """
        connector = FakeConnectorManager(num_matched=64)
        manager = make_manager(connector)
        req = FakeRequest()
        kv_cache = FakeKvCache(committed=0)

        prepare(manager, req, kv_cache)
        kv_cache.num_committed_tokens = 32
        prepare(manager, req, kv_cache)
        manager._deliver_connector_prefix(req)

        assert connector.cancels == [(0, 0, 32)]
        assert connector.commits == [(0, 32, False)], "[32, 64), not [0, 64)"

    def test_fully_subsumed_offer_transfers_nothing(self):
        connector = FakeConnectorManager(num_matched=64)
        manager = make_manager(connector)
        req = FakeRequest()
        kv_cache = FakeKvCache(committed=0)

        prepare(manager, req, kv_cache)
        kv_cache.num_committed_tokens = 128
        prepare(manager, req, kv_cache)
        manager._deliver_connector_prefix(req)

        assert connector.cancels == [(0, 0, 64)]
        assert connector.commits == [(0, 0, False)]

    def test_recorded_delta_restores_the_locally_computed_position(self):
        """``computed_position`` is reported as ``end - recorded``.

        That is what a connector uses to find where its blocks start, so the
        recorded delta has to be the one anchored at the *current* commit
        boundary.
        """
        connector = FakeConnectorManager(num_matched=96)
        manager = make_manager(connector)
        req = FakeRequest()
        kv_cache = FakeKvCache(committed=32)

        prepare(manager, req, kv_cache)
        kv_cache.num_committed_tokens = 64
        prepare(manager, req, kv_cache)
        manager._deliver_connector_prefix(req)

        recorded = connector.commits[0][1]
        assert req.context_current_position - recorded == 64


class TestResidency:
    def test_capacity_and_history_move_together(self):
        """After a reuse match both equal the local match.

        Raising history alone would trip "History length cannot be greater than
        capacity"; raising capacity alone would leave a sliding-window group
        allocating a page for every block of the served prefix.
        """
        connector = FakeConnectorManager(num_matched=64)
        manager = make_manager(connector)
        req = FakeRequest()
        kv_cache = FakeKvCache(committed=32)

        prepare(manager, req, kv_cache)

        assert kv_cache.resize_calls == [(96, 96)]
        assert kv_cache.capacity == 96
        assert kv_cache.history_length == 96

    def test_existing_capacity_is_not_shrunk(self):
        connector = FakeConnectorManager(num_matched=64)
        manager = make_manager(connector)
        req = FakeRequest()
        kv_cache = FakeKvCache(committed=0)
        kv_cache.capacity = 256

        prepare(manager, req, kv_cache)

        assert kv_cache.resize_calls == [(256, 64)]

    def test_swa_scratch_reuse_is_disabled_for_a_served_prefix(self):
        connector = FakeConnectorManager(num_matched=64)
        manager = make_manager(connector)
        req = FakeRequest()
        kv_cache = FakeKvCache(committed=0)

        prepare(manager, req, kv_cache)

        assert kv_cache.enable_swa_scratch_reuse is False

    def test_empty_offer_touches_nothing(self):
        connector = FakeConnectorManager(num_matched=0)
        manager = make_manager(connector)
        req = FakeRequest()
        kv_cache = FakeKvCache(committed=32)

        prepare(manager, req, kv_cache)
        manager._deliver_connector_prefix(req)

        assert kv_cache.resize_calls == []
        assert kv_cache.enable_swa_scratch_reuse is True
        assert connector.cancels == []
        assert connector.commits == [(0, 0, False)]

    def test_offer_is_handed_back_when_pages_run_out(self):
        connector = FakeConnectorManager(num_matched=64)
        manager = make_manager(connector)
        req = FakeRequest()
        kv_cache = FakeKvCache(committed=32, resize_ok=False)

        prepare(manager, req, kv_cache)

        assert connector.cancels == [(0, 32, 96)]
        assert req.context_current_position == 32
        # The request runs local-only from here, and delivering must not then
        # record an external load for a range that was never covered.
        manager._deliver_connector_prefix(req)
        assert connector.commits == [(0, 0, False)]


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
        kv_cache = FakeKvCache(committed=0)

        assert prepare(manager, req, kv_cache) is None
        assert connector.queries == []
        assert req.context_current_position == 0

    def test_no_connector_is_a_no_op(self):
        manager = make_manager(None)
        req = FakeRequest()
        kv_cache = FakeKvCache(committed=32)

        assert manager._connector_prefix_position(req, kv_cache) is None

    def test_draft_manager_never_asks(self):
        """The draft manager's prepare_resources skips the connector hooks, so
        a query there would never be delivered."""
        connector = FakeConnectorManager(num_matched=64)
        manager = make_manager(connector)
        manager.is_draft = True
        req = FakeRequest()
        kv_cache = FakeKvCache(committed=0)

        assert manager._connector_prefix_position(req, kv_cache) is None
        assert connector.queries == []


class TestUndeliveredOfferIsReleased:
    def test_request_that_dies_before_delivery_hands_the_offer_back(self):
        connector = FakeConnectorManager(num_matched=64)
        manager = make_manager(connector)
        req = FakeRequest()
        kv_cache = FakeKvCache(committed=0)

        prepare(manager, req, kv_cache)
        # Cancelled, timed out, or failed before it ever reached a batch.
        manager._release_undelivered_connector_prefix(req)

        assert connector.cancels == [(0, 0, 64)]

    def test_delivered_request_keeps_its_offer_on_free(self):
        connector = FakeConnectorManager(num_matched=64)
        manager = make_manager(connector)
        req = FakeRequest()
        kv_cache = FakeKvCache(committed=0)

        prepare(manager, req, kv_cache)
        manager._deliver_connector_prefix(req)
        req.py_connector_delivered = True
        manager._release_undelivered_connector_prefix(req)

        assert connector.cancels == []

    def test_unasked_request_releases_nothing(self):
        connector = FakeConnectorManager()
        manager = make_manager(connector)

        manager._release_undelivered_connector_prefix(FakeRequest())

        assert connector.cancels == []


class TestAsyncLoad:
    def test_async_flag_is_carried_from_query_to_delivery(self):
        connector = FakeConnectorManager(num_matched=64, load_async=True)
        manager = make_manager(connector)
        req = FakeRequest()
        kv_cache = FakeKvCache(committed=0)

        prepare(manager, req, kv_cache)
        assert req.py_connector_load_async is True

        manager._deliver_connector_prefix(req)
        assert connector.commits == [(0, 64, True)]

    def test_async_hold_survives_the_clamp(self):
        """An async connector starts its transfer inside the query itself.

        So even when the clamp leaves nothing to load, the request still has to
        be held out of the batch until the connector reports the transfer done,
        or the forward races the writes.
        """
        connector = FakeConnectorManager(num_matched=1, load_async=True)
        manager = make_manager(connector)
        req = FakeRequest(prompt_len=PROMPT_LEN)
        kv_cache = FakeKvCache(committed=PROMPT_LEN - 1)

        prepare(manager, req, kv_cache)
        manager._deliver_connector_prefix(req)

        assert req.py_connector_prefix_end == PROMPT_LEN - 1
        assert connector.commits == [(0, 0, True)]
        # Lossy for an async load -- the transfer began inside the query -- but
        # still the only signal the connector gets that the tail is dead.
        assert connector.cancels == [(0, PROMPT_LEN - 1, PROMPT_LEN)]


class TestPhasesMustAgree:
    """Phase 3 refuses to report a load phase 2 never made room for.

    This is the single coupling in the design that nothing downstream checks:
    the runtime hands the connector ``context_current_position - recorded`` as
    the range it computed locally, the subtraction is unguarded
    (``kv_cache_connector.py:480``), and connectors divide the result into block
    ordinals rather than validating it. A phase 2 that silently no-ops -- by
    regression, or by a future caller reordering the phases -- therefore points
    the connector at a negative offset inside its own code rather than failing.
    """

    def test_recording_more_than_the_position_covers_is_an_error(self):
        connector = FakeConnectorManager(num_matched=64)
        manager = make_manager(connector)
        req = FakeRequest()
        kv_cache = FakeKvCache(committed=0)

        prepare(manager, req, kv_cache)
        # The shape a silently skipped phase 2 leaves behind: the offer is on
        # the request, but no capacity, history or position backs it.
        req.context_current_position = 0

        with pytest.raises(AssertionError, match="phase 2 did not reserve"):
            manager._deliver_connector_prefix(req)

        assert connector.commits == [], (
            "the load must not be recorded when the assertion fires, or the "
            "connector is left holding an offer the runtime denied"
        )

    def test_a_fully_reserved_offer_is_accepted(self):
        """Anti-vacuity: the assertion must not fire on the normal path."""
        connector = FakeConnectorManager(num_matched=64)
        manager = make_manager(connector)
        req = FakeRequest()
        kv_cache = FakeKvCache(committed=32)

        prepare(manager, req, kv_cache)
        manager._deliver_connector_prefix(req)

        assert connector.commits == [(0, 64, False)]
