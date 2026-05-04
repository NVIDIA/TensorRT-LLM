# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Integration tests for the budget guard inside ``KVCacheManager.prepare_resources``.

The guard re-probes the radix tree before admitting first-chunk context requests
and skips any whose post-reuse forward cost would exceed the remaining
``max_num_tokens`` budget. This file drives the real ``prepare_resources`` method
against a stand-in manager constructed via ``__new__`` (skips C++ initialization)
plus a real ``ScheduledRequests``.

Branch coverage: skip / no-skip / non-first-chunk pre-subtraction / draft tokens /
connector should-add-sequence false branch / connector callbacks / no-op when
reuse is disabled / no-op when this is the draft manager / reset_context_requests
filtering / analyze_prefix_reuse call shape / non-first-chunk fallthrough.

Regression: a deterministic pair simulating the eviction race that the guard
prevents — ``test_invariant_passes_with_guard`` admits to within budget;
``test_invariant_breaks_when_guard_disabled`` is the negative control that
patches the helper out and confirms the synthetic workload would otherwise
overshoot.
"""

from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import ScheduledRequests


# ---------------------------------------------------------------------------
# Mock factories — avoid Mock(spec=LlmRequest) because LlmRequest is a C++
# binding; we just set the attributes prepare_resources reads.
# ---------------------------------------------------------------------------
def _make_first_chunk_req(
    rid, prompt_len, chunk_size, est_reuse=0, num_draft_tokens=0, unique_tokens=None, beam_width=1
):
    req = Mock()
    req.request_id = rid
    req.py_request_id = rid
    req.prompt_len = prompt_len
    req.context_chunk_size = chunk_size
    req.context_current_position = 0
    req.is_first_context_chunk = True
    req.is_last_context_chunk = chunk_size >= prompt_len
    req.estimated_reusable_tokens = est_reuse
    req.num_draft_tokens = num_draft_tokens
    req.py_draft_tokens = list(range(num_draft_tokens))
    req.get_unique_tokens.return_value = unique_tokens or [0] * prompt_len
    # The regression-net helper reads len(req.get_tokens(0)) to mirror the
    # actual model-engine slicing; provide a concrete list of that length.
    req.get_tokens.return_value = list(range(prompt_len))
    req.sampling_config = Mock()
    req.sampling_config.beam_width = beam_width
    return req


def _make_non_first_chunk_req(rid, chunk_size, prompt_len=None, current_pos=128):
    req = Mock()
    req.request_id = rid
    req.py_request_id = rid
    pl = prompt_len if prompt_len is not None else (chunk_size + current_pos + 256)
    req.prompt_len = pl
    req.context_chunk_size = chunk_size
    req.context_current_position = current_pos
    req.is_first_context_chunk = False
    req.is_last_context_chunk = False
    req.num_draft_tokens = 0
    req.py_draft_tokens = []
    req.get_tokens.return_value = list(range(pl))
    req.sampling_config = Mock()
    req.sampling_config.beam_width = 1
    return req


def _summary(reusable_blocks):
    """Stand-in for ``analyze_prefix_reuse``'s ``PrefixReuseSummary`` —
    only ``reusable_blocks_all`` is read by the budget guard."""
    return Mock(reusable_blocks_all=reusable_blocks)


def _make_gen_req(rid, num_draft_tokens=0, beam_width=1):
    req = Mock()
    req.request_id = rid
    req.py_request_id = rid
    req.num_draft_tokens = num_draft_tokens
    req.py_draft_tokens = list(range(num_draft_tokens))
    req.get_beam_width_by_iter.return_value = beam_width
    req.sampling_config = Mock()
    req.sampling_config.beam_width = beam_width
    return req


@pytest.fixture
def fake_kv_manager():
    """``KVCacheManager``-shaped object holding every attribute and stubbed
    method that ``prepare_resources`` touches before the budget code runs.
    Skips ``__init__`` to avoid pulling in C++ resources. Each test gets a
    fresh fixture — never share across tests because ``prepare_resources``
    mutates ``scheduled_batch`` and reads request fields."""
    mgr = KVCacheManager.__new__(KVCacheManager)
    mgr.is_draft = False
    mgr.enable_block_reuse = True
    mgr.max_num_tokens = 512
    mgr.tokens_per_block = 64
    mgr.num_extra_kv_tokens = 0
    mgr.mapping = Mock()
    mgr.mapping.cp_config = {}
    mgr.mapping.has_cp_helix.return_value = False
    mgr.kv_connector_manager = None  # default; tests override for connector cases
    mgr.impl = Mock()
    mgr.impl.analyze_prefix_reuse.return_value = _summary(0)
    mgr.impl.add_sequence_batch = Mock()
    mgr.impl.add_token = Mock()
    mgr.impl.sync_transfer_manager_with_buffer_manager = Mock()
    mgr.impl.refresh_blocks = Mock()
    # Stubbed for connector tests; harmless when kv_connector_manager is None.
    mgr.get_cache_indices = Mock(return_value=[])
    return mgr


@pytest.fixture
def make_batch():
    """Build a fresh ``ScheduledRequests`` via the public append API.
    NEVER assign ``batch.context_requests`` directly — it's a property
    derived from ``context_requests_chunking + context_requests_last_chunk``."""

    def _factory(ctx_reqs, gen_reqs):
        batch = ScheduledRequests()
        for r in ctx_reqs:
            batch.append_context_request(r)
        for r in gen_reqs:
            batch.append_generation_request(r)
        return batch

    return _factory


# ===========================================================================
# Branch coverage
# ===========================================================================
class TestSkipPath:
    """The guard skips first-chunk reqs whose actual reuse is less than
    the scheduler estimated, so admitting them would overshoot."""

    def test_skip_when_actual_reuse_less_than_estimated(self, fake_kv_manager, make_batch):
        # 3 ctx reqs; scheduler thought each had ~768-token reuse, actual is 0.
        # remaining_budget starts at 512. Each req: helper(0, 256, 1024) = 256.
        #   req0: 256 ≤ 512 → admit, remaining 256
        #   req1: 256 ≤ 256 → admit, remaining 0
        #   req2: 256 > 0 → SKIP via continue
        fake_kv_manager.impl.analyze_prefix_reuse.return_value = _summary(0)
        ctx = [
            _make_first_chunk_req(rid=i, prompt_len=1024, chunk_size=256, est_reuse=768)
            for i in range(3)
        ]
        batch = make_batch(ctx, [])

        fake_kv_manager.prepare_resources(batch)

        surviving = [r.py_request_id for r in batch.context_requests]
        assert surviving == [0, 1], f"req2 should be skipped via continue; got {surviving}"

        # The two admitted reqs went into add_sequence_batch.
        fake_kv_manager.impl.add_sequence_batch.assert_called_once()
        infos = fake_kv_manager.impl.add_sequence_batch.call_args[0][0]
        assert {info[0] for info in infos} == {0, 1}


class TestNoSkipPath:
    """When actual reuse matches the scheduler estimate, no requests are skipped."""

    def test_no_skip_when_estimate_matches_actual(self, fake_kv_manager, make_batch):
        # 3 ctx reqs; each has est_reuse=896 (14 blocks at block=64), and
        # analyze_prefix_reuse returns reusable_blocks_all=14 → actual_reuse=896.
        # helper(896, 128, 1024): P+chunk=1024 == prompt_len → last-chunk
        #     → max(1, 1024-896)=128
        # 3 * 128 = 384 ≤ 512 → all admitted
        fake_kv_manager.impl.analyze_prefix_reuse.return_value = _summary(14)
        ctx = [
            _make_first_chunk_req(rid=i, prompt_len=1024, chunk_size=128, est_reuse=896)
            for i in range(3)
        ]
        batch = make_batch(ctx, [])

        fake_kv_manager.prepare_resources(batch)

        assert {r.py_request_id for r in batch.context_requests} == {0, 1, 2}
        fake_kv_manager.impl.add_sequence_batch.assert_called_once()


class TestNonFirstChunkPreSubtract:
    """Non-first-chunk reqs reduce the budget for first-chunk reqs."""

    def test_non_first_chunk_pre_subtracted(self, fake_kv_manager, make_batch):
        # remaining_budget = 512 - 0(gen) - 300(non-first chunk_size) = 212
        # First-chunk req: helper(0, 256, 1024)=256 > 212 → SKIP
        fake_kv_manager.impl.analyze_prefix_reuse.return_value = _summary(0)
        nf = _make_non_first_chunk_req(rid=10, chunk_size=300, current_pos=256, prompt_len=1024)
        fc = _make_first_chunk_req(rid=20, prompt_len=1024, chunk_size=256, est_reuse=0)
        batch = make_batch([nf, fc], [])

        fake_kv_manager.prepare_resources(batch)

        surviving = [r.py_request_id for r in batch.context_requests]
        # nf falls through both if/elif (not first_context_chunk) → kept.
        # fc hits the if branch and is skipped via `continue`.
        assert surviving == [10]
        # No first-chunk admitted, so add_sequence_batch is never called.
        fake_kv_manager.impl.add_sequence_batch.assert_not_called()


class TestGenTokensConsumeBudget:
    """Generation-request tokens (including draft) reduce the remaining budget."""

    def test_gen_with_draft_consumes_budget(self, fake_kv_manager, make_batch):
        # remaining_budget = 512 - (1+5)(gen with 5 draft) = 506
        # First-chunk req: helper(0, 600, 1024)=600 > 506 → SKIP
        fake_kv_manager.impl.analyze_prefix_reuse.return_value = _summary(0)
        gen = _make_gen_req(rid=99, num_draft_tokens=5)
        fc = _make_first_chunk_req(rid=20, prompt_len=1024, chunk_size=600, est_reuse=0)
        batch = make_batch([fc], [gen])

        fake_kv_manager.prepare_resources(batch)

        # fc skipped via `continue`; gen requests are not in context_requests anyway.
        assert [r.py_request_id for r in batch.context_requests] == []

    def test_gen_without_draft_admits(self, fake_kv_manager, make_batch):
        # remaining_budget = 512 - 1(gen, no draft) = 511
        # First-chunk req: helper(0, 256, 1024)=256 ≤ 511 → admit
        fake_kv_manager.impl.analyze_prefix_reuse.return_value = _summary(0)
        gen = _make_gen_req(rid=99, num_draft_tokens=0)
        fc = _make_first_chunk_req(rid=20, prompt_len=1024, chunk_size=256, est_reuse=0)
        batch = make_batch([fc], [gen])

        fake_kv_manager.prepare_resources(batch)

        assert [r.py_request_id for r in batch.context_requests] == [20]


class TestNoOpPaths:
    """The guard is a no-op when reuse is disabled or this is the draft manager."""

    def test_no_op_when_reuse_disabled(self, fake_kv_manager, make_batch):
        fake_kv_manager.enable_block_reuse = False
        ctx = [
            _make_first_chunk_req(rid=i, prompt_len=1024, chunk_size=256, est_reuse=768)
            for i in range(4)
        ]
        batch = make_batch(ctx, [])

        fake_kv_manager.prepare_resources(batch)

        # analyze_prefix_reuse must NEVER be called when the guard is off.
        fake_kv_manager.impl.analyze_prefix_reuse.assert_not_called()
        # All 4 admitted; no skip path engaged.
        assert {r.py_request_id for r in batch.context_requests} == {0, 1, 2, 3}

    def test_no_op_when_is_draft(self, fake_kv_manager, make_batch):
        fake_kv_manager.is_draft = True
        ctx = [
            _make_first_chunk_req(rid=i, prompt_len=1024, chunk_size=256, est_reuse=768)
            for i in range(4)
        ]
        batch = make_batch(ctx, [])

        fake_kv_manager.prepare_resources(batch)

        fake_kv_manager.impl.analyze_prefix_reuse.assert_not_called()
        assert {r.py_request_id for r in batch.context_requests} == {0, 1, 2, 3}


class TestResetContextRequestsFiltered:
    """``reset_context_requests`` reclassifies the post-skip request list."""

    def test_reset_with_chunk_type_mutation(self, fake_kv_manager, make_batch):
        """A request may flip ``is_last_context_chunk`` inside ``add_sequence_batch``
        (when reuse covers most of the prompt, leaving only the last chunk).
        After ``prepare_resources``, the request must land in
        ``context_requests_last_chunk`` via the ``reset_context_requests`` call."""
        fake_kv_manager.impl.analyze_prefix_reuse.return_value = _summary(0)
        # Start with a non-last chunk that gets flipped during add_sequence_batch.
        fc = _make_first_chunk_req(rid=0, prompt_len=1024, chunk_size=256, est_reuse=0)
        fc.is_last_context_chunk = False  # initial classification

        def _flip_to_last(infos, reqs):
            for r in reqs:
                r.is_last_context_chunk = True

        fake_kv_manager.impl.add_sequence_batch.side_effect = _flip_to_last

        batch = make_batch([fc], [])
        # initial bucketing: non-last
        assert len(batch.context_requests_chunking) == 1
        assert len(batch.context_requests_last_chunk) == 0

        fake_kv_manager.prepare_resources(batch)

        # After prepare_resources + reset, the flipped req moves to last bucket.
        assert len(batch.context_requests_chunking) == 0
        assert len(batch.context_requests_last_chunk) == 1
        assert batch.context_requests_last_chunk[0].py_request_id == 0


class TestKvConnectorElifBranch:
    """When the connector rejects a first-chunk req, the elif branch charges
    the budget using the scheduler's estimate (without re-probing) and the
    request is not handed to ``add_sequence_batch``."""

    def test_connector_should_not_add_charges_estimated_reuse(self, fake_kv_manager, make_batch):
        connector = Mock()
        # req0: rejected by connector → elif branch
        # req1, req2: accepted by connector → if branch
        connector.should_add_sequence.side_effect = [False, True, True]
        fake_kv_manager.kv_connector_manager = connector
        fake_kv_manager.impl.analyze_prefix_reuse.return_value = _summary(0)

        ctx = [
            _make_first_chunk_req(rid=i, prompt_len=512, chunk_size=256, est_reuse=448)
            for i in range(3)
        ]
        batch = make_batch(ctx, [])

        fake_kv_manager.prepare_resources(batch)

        # Arithmetic walk:
        #   Initial budget = 512 - 0(gen) - 0(non-first) = 512
        #   req0 elif: helper(448, 256, 512): P+chunk=704 >= 512 → max(1, 64) = 64
        #              budget -= 64 → remaining 448
        #   req1 if: actual_reuse=0 → helper(0, 256, 512)=256 ≤ 448 → admit, budget 192
        #   req2 if: helper=256 > 192 → SKIP
        surviving = [r.py_request_id for r in batch.context_requests]
        assert surviving == [0, 1], f"expected [0, 1], got {surviving}"
        # Only req1 went into add_sequence_batch (req0 was rejected by connector).
        infos = fake_kv_manager.impl.add_sequence_batch.call_args[0][0]
        assert [info[0] for info in infos] == [1]


class TestKvConnectorCallbacks:
    """Connector callbacks fire only for admitted requests, and the post-loop
    ``build_scheduler_output`` sees the filtered batch."""

    def test_update_state_after_alloc_only_for_admitted(self, fake_kv_manager, make_batch):
        connector = Mock()
        connector.should_add_sequence.return_value = True
        fake_kv_manager.kv_connector_manager = connector
        fake_kv_manager.get_cache_indices = Mock(return_value=[10, 20])

        # 3 ctx reqs: budget allows 2 to be admitted, 1 skipped.
        fake_kv_manager.impl.analyze_prefix_reuse.return_value = _summary(0)
        ctx = [
            _make_first_chunk_req(rid=i, prompt_len=1024, chunk_size=256, est_reuse=768)
            for i in range(3)
        ]
        batch = make_batch(ctx, [])

        fake_kv_manager.prepare_resources(batch)

        # update_state_after_alloc should be called once per admitted req (2).
        # The skipped req should NOT trigger the callback.
        admitted_ids = [
            call.args[0].py_request_id for call in connector.update_state_after_alloc.call_args_list
        ]
        assert sorted(admitted_ids) == [0, 1]

    def test_build_scheduler_output_after_filter(self, fake_kv_manager, make_batch):
        connector = Mock()
        connector.should_add_sequence.return_value = True
        fake_kv_manager.kv_connector_manager = connector

        fake_kv_manager.impl.analyze_prefix_reuse.return_value = _summary(0)
        ctx = [
            _make_first_chunk_req(rid=i, prompt_len=1024, chunk_size=256, est_reuse=768)
            for i in range(3)
        ]
        batch = make_batch(ctx, [])

        fake_kv_manager.prepare_resources(batch)

        # build_scheduler_output should be called exactly once after the loop
        # with the (already-filtered) batch.
        connector.build_scheduler_output.assert_called_once()
        passed_batch = connector.build_scheduler_output.call_args[0][0]
        assert {r.py_request_id for r in passed_batch.context_requests} == {0, 1}


class TestContextDraftTokens:
    """A context request with draft tokens triggers extra ``add_token`` calls."""

    def test_context_with_draft_tokens(self, fake_kv_manager, make_batch):
        # num_extra_kv_tokens=0, num_draft_tokens=3 → add_token called 3 times.
        fake_kv_manager.impl.analyze_prefix_reuse.return_value = _summary(0)
        ctx = [
            _make_first_chunk_req(
                rid=0, prompt_len=512, chunk_size=256, est_reuse=0, num_draft_tokens=3
            )
        ]
        batch = make_batch(ctx, [])

        fake_kv_manager.prepare_resources(batch)

        assert fake_kv_manager.impl.add_token.call_count == 3


class TestAnalyzePrefixReuseCallShape:
    """The budget guard calls ``analyze_prefix_reuse`` with the expected
    argument shape — a signature drift would silently break re-probing."""

    def test_analyze_prefix_reuse_call_args(self, fake_kv_manager, make_batch):
        sentinel_unique = [101, 102, 103, 104]
        fake_kv_manager.impl.analyze_prefix_reuse.return_value = _summary(0)
        req = _make_first_chunk_req(
            rid=0, prompt_len=1024, chunk_size=256, est_reuse=0, unique_tokens=sentinel_unique
        )
        batch = make_batch([req], [])

        fake_kv_manager.prepare_resources(batch)

        fake_kv_manager.impl.analyze_prefix_reuse.assert_called_once_with(sentinel_unique, req)


class TestNonFirstChunkFallthrough:
    """Non-first-chunk reqs fall through both the if and elif branches and
    are kept by ``reset_context_requests``, but are NOT added to
    ``add_sequence_batch``."""

    def test_non_first_chunk_falls_through(self, fake_kv_manager, make_batch):
        nf = _make_non_first_chunk_req(rid=0, chunk_size=128, current_pos=128, prompt_len=1024)
        batch = make_batch([nf], [])

        fake_kv_manager.prepare_resources(batch)

        assert [r.py_request_id for r in batch.context_requests] == [0]
        fake_kv_manager.impl.add_sequence_batch.assert_not_called()


# ===========================================================================
# Regression-net pair
# ===========================================================================
def _model_engine_total_tokens(scheduled_batch):
    """Mirror of the actual ``_prepare_tp_inputs`` ``len(position_ids)``
    computation, INDEPENDENT of ``_estimate_post_reuse_compute``. The
    negative-control test patches the helper out; the verifier MUST not
    go through the same code path or it would self-invalidate.

    Per ctx req: ``len(prompt_tokens[begin:begin+chunk_size])`` =
        ``min(chunk_size, len(get_tokens(0)) - context_current_position)``.
    Per gen req:
      - draft path: ``1 + len(py_draft_tokens)`` (extend path, NOT per beam).
      - no-draft path: one position per beam.
    """
    total = 0
    for req in scheduled_batch.context_requests:
        begin = req.context_current_position
        all_tokens = req.get_tokens(0)
        total += max(0, min(req.context_chunk_size, len(all_tokens) - begin))
    for req in scheduled_batch.generation_requests:
        if len(req.py_draft_tokens) > 0:
            total += 1 + len(req.py_draft_tokens)
        else:
            total += req.sampling_config.beam_width
    return total


class TestEvictionRaceRegression:
    """Deterministic regression for the eviction-race overshoot the guard
    prevents.

    ``test_invariant_passes_with_guard`` — synthesize a scheduler that
    estimated full reuse and then full eviction (``analyze_prefix_reuse``
    returns 0). The guard must skip enough requests that the post-prepare
    forward token total fits within ``max_num_tokens``.

    ``test_invariant_breaks_when_guard_disabled`` — negative control. With
    the helper monkey-patched to ``0`` the skip predicate ``> remaining``
    is always false, so all four requests are admitted and the same
    workload overshoots. The verifier uses ``_model_engine_total_tokens``
    instead of the patched helper, so the comparison is meaningful.
    """

    def test_invariant_passes_with_guard(self, fake_kv_manager, make_batch):
        fake_kv_manager.impl.analyze_prefix_reuse.return_value = _summary(0)  # eviction

        ctx = [
            _make_first_chunk_req(rid=i, prompt_len=512, chunk_size=256, est_reuse=448)
            for i in range(4)
        ]
        batch = make_batch(ctx, [])

        # Simulate prepare_context (called inside add_sequence_batch on the
        # real path): for admitted first-chunk reqs, set
        # context_current_position to the actual reused offset (0 here).
        def _set_pos(infos, reqs):
            for r in reqs:
                r.context_current_position = 0

        fake_kv_manager.impl.add_sequence_batch.side_effect = _set_pos

        fake_kv_manager.prepare_resources(batch)

        total = _model_engine_total_tokens(batch)
        assert total <= fake_kv_manager.max_num_tokens, (
            f"post-prepare forward total {total} > max_num_tokens "
            f"{fake_kv_manager.max_num_tokens}; surviving: "
            f"{[r.py_request_id for r in batch.context_requests]}"
        )

    def test_invariant_breaks_when_guard_disabled(self, fake_kv_manager, make_batch, monkeypatch):
        monkeypatch.setattr(
            KVCacheManager,
            "_estimate_post_reuse_compute",
            lambda self, *a, **k: 0,
        )
        fake_kv_manager.impl.analyze_prefix_reuse.return_value = _summary(0)

        ctx = [
            _make_first_chunk_req(rid=i, prompt_len=512, chunk_size=256, est_reuse=448)
            for i in range(4)
        ]
        batch = make_batch(ctx, [])

        def _set_pos(infos, reqs):
            for r in reqs:
                r.context_current_position = 0

        fake_kv_manager.impl.add_sequence_batch.side_effect = _set_pos

        fake_kv_manager.prepare_resources(batch)

        total = _model_engine_total_tokens(batch)
        assert total > fake_kv_manager.max_num_tokens, (
            f"negative control: expected overshoot, got total={total} ≤ "
            f"max={fake_kv_manager.max_num_tokens}. The synthetic scenario "
            f"no longer triggers the bug condition with the guard disabled — "
            f"strengthen the workload (more requests / larger chunks)."
        )
