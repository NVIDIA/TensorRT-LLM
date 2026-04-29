# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Integration tests for KVCacheManager.prepare_resources() budget guard
(DYN-2868). Drives the real ``prepare_resources`` method against a mocked
manager (constructed via ``__new__`` to skip C++ deps) and a real
``ScheduledRequests``.

Tier 2: branch coverage of the budget guard (skip / no-skip / non-first
        pre-subtraction / draft tokens / connector branches / no-op when
        reuse off / no-op when is_draft / reset_context_requests filter /
        count_reusable_blocks call shape / non-first fallthrough).
Tier 3: deterministic DYN-2868 regression pair (Test M passes with guard,
        Test N — negative control — overshoots when guard disabled).
"""
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import ScheduledRequests


# ---------------------------------------------------------------------------
# Mock factories — avoid Mock(spec=LlmRequest) because LlmRequest is a C++
# binding; we just set the attributes prepare_resources reads.
# ---------------------------------------------------------------------------
def _make_first_chunk_req(rid, prompt_len, chunk_size, est_reuse=0,
                          num_draft_tokens=0, unique_tokens=None,
                          beam_width=1):
    req = Mock()
    req.request_id = rid
    req.py_request_id = rid
    req.prompt_len = prompt_len
    req.context_chunk_size = chunk_size
    req.context_current_position = 0
    req.is_first_context_chunk = True
    req.is_last_context_chunk = (chunk_size >= prompt_len)
    req.estimated_reusable_tokens = est_reuse
    req.num_draft_tokens = num_draft_tokens
    req.py_draft_tokens = list(range(num_draft_tokens))
    req.get_unique_tokens.return_value = unique_tokens or [0] * prompt_len
    # Tier 3 helper uses len(req.get_tokens(0)) per model_engine.py:2291.
    req.get_tokens.return_value = list(range(prompt_len))
    req.sampling_config = Mock()
    req.sampling_config.beam_width = beam_width
    return req


def _make_non_first_chunk_req(rid, chunk_size, prompt_len=None,
                              current_pos=128):
    req = Mock()
    req.request_id = rid
    req.py_request_id = rid
    pl = prompt_len if prompt_len is not None else (
        chunk_size + current_pos + 256)
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
    """KVCacheManager-shaped object with all attrs/methods that
    ``prepare_resources`` touches before our budget code runs. Skips
    ``__init__`` to avoid C++ deps. Each test gets a fresh fixture —
    never share across tests because ``prepare_resources`` mutates
    ``scheduled_batch`` and reads request fields."""
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
    mgr.impl.count_reusable_blocks.return_value = 0
    mgr.impl.add_sequence_batch = Mock()
    mgr.impl.add_token = Mock()
    mgr.impl.sync_transfer_manager_with_buffer_manager = Mock()
    mgr.impl.refresh_blocks = Mock()
    # For connector tests; harmless when kv_connector_manager is None.
    mgr.get_cache_indices = Mock(return_value=[])
    return mgr


@pytest.fixture
def make_batch():
    """Construct a fresh ScheduledRequests using the public append API.
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
# Tier 2: branch coverage
# ===========================================================================
class TestSkipPath:
    """Test A — skip when actual reuse < estimated reuse."""

    def test_a_skip_when_actual_reuse_less_than_estimated(
        self, fake_kv_manager, make_batch
    ):
        # 3 ctx reqs; scheduler thought each had ~768-token reuse, actual is 0.
        # remaining_budget starts at 512. Each req: helper(0, 256, 1024) = 256.
        #   req0: 256 ≤ 512 → admit, remaining 256
        #   req1: 256 ≤ 256 → admit, remaining 0
        #   req2: 256 > 0 → SKIP via continue
        fake_kv_manager.impl.count_reusable_blocks.return_value = 0
        ctx = [
            _make_first_chunk_req(rid=i, prompt_len=1024, chunk_size=256,
                                  est_reuse=768)
            for i in range(3)
        ]
        batch = make_batch(ctx, [])

        fake_kv_manager.prepare_resources(batch)

        surviving = [r.py_request_id for r in batch.context_requests]
        assert surviving == [0, 1], (
            f"req2 should be skipped via continue; got {surviving}")

        # The two admitted reqs went into add_sequence_batch.
        fake_kv_manager.impl.add_sequence_batch.assert_called_once()
        infos = fake_kv_manager.impl.add_sequence_batch.call_args[0][0]
        assert {info[0] for info in infos} == {0, 1}


class TestNoSkipPath:
    """Test B — no skip when estimate matches actual."""

    def test_b_no_skip_when_estimate_matches_actual(
        self, fake_kv_manager, make_batch
    ):
        # 3 ctx reqs; each has est_reuse=896 (14 blocks at block=64), and
        # actual count_reusable_blocks=14 → actual_reuse=896.
        # helper(896, 128, 1024): P+chunk=1024 == prompt_len → last-chunk
        #     → max(1, 1024-896)=128
        # 3 * 128 = 384 ≤ 512 → all admitted
        fake_kv_manager.impl.count_reusable_blocks.return_value = 14
        ctx = [
            _make_first_chunk_req(rid=i, prompt_len=1024, chunk_size=128,
                                  est_reuse=896)
            for i in range(3)
        ]
        batch = make_batch(ctx, [])

        fake_kv_manager.prepare_resources(batch)

        assert {r.py_request_id for r in batch.context_requests} == {0, 1, 2}
        fake_kv_manager.impl.add_sequence_batch.assert_called_once()


class TestNonFirstChunkPreSubtract:
    """Test C — non-first-chunk reqs reduce budget for first-chunk reqs."""

    def test_c_non_first_chunk_pre_subtracted(
        self, fake_kv_manager, make_batch
    ):
        # remaining_budget = 512 - 0(gen) - 300(non-first chunk_size) = 212
        # First-chunk req: helper(0, 256, 1024)=256 > 212 → SKIP
        fake_kv_manager.impl.count_reusable_blocks.return_value = 0
        nf = _make_non_first_chunk_req(rid=10, chunk_size=300,
                                       current_pos=256, prompt_len=1024)
        fc = _make_first_chunk_req(rid=20, prompt_len=1024, chunk_size=256,
                                   est_reuse=0)
        batch = make_batch([nf, fc], [])

        fake_kv_manager.prepare_resources(batch)

        surviving = [r.py_request_id for r in batch.context_requests]
        # nf falls through both if/elif (not first_context_chunk) → kept.
        # fc hits the if branch and is skipped via `continue`.
        assert surviving == [10]
        # No first-chunk admitted, so add_sequence_batch is never called.
        fake_kv_manager.impl.add_sequence_batch.assert_not_called()


class TestGenTokensConsumeBudget:
    """Test D — gen tokens (including draft) reduce remaining_budget."""

    def test_d_gen_with_draft_consumes_budget(
        self, fake_kv_manager, make_batch
    ):
        # remaining_budget = 512 - (1+5)(gen with 5 draft) = 506
        # First-chunk req: helper(0, 600, 1024)=600 > 506 → SKIP
        fake_kv_manager.impl.count_reusable_blocks.return_value = 0
        gen = _make_gen_req(rid=99, num_draft_tokens=5)
        fc = _make_first_chunk_req(rid=20, prompt_len=1024, chunk_size=600,
                                   est_reuse=0)
        batch = make_batch([fc], [gen])

        fake_kv_manager.prepare_resources(batch)

        # fc skipped via `continue`; only gen survives in context_requests
        # (gen requests are not in context_requests anyway).
        assert [r.py_request_id for r in batch.context_requests] == []

    def test_d_gen_without_draft_admits(
        self, fake_kv_manager, make_batch
    ):
        # remaining_budget = 512 - 1(gen, no draft) = 511
        # First-chunk req: helper(0, 256, 1024)=256 ≤ 511 → admit
        fake_kv_manager.impl.count_reusable_blocks.return_value = 0
        gen = _make_gen_req(rid=99, num_draft_tokens=0)
        fc = _make_first_chunk_req(rid=20, prompt_len=1024, chunk_size=256,
                                   est_reuse=0)
        batch = make_batch([fc], [gen])

        fake_kv_manager.prepare_resources(batch)

        assert [r.py_request_id for r in batch.context_requests] == [20]


class TestNoOpPaths:
    """Tests E and F — guard is a no-op when reuse is off / draft manager."""

    def test_e_no_op_when_reuse_disabled(self, fake_kv_manager, make_batch):
        fake_kv_manager.enable_block_reuse = False
        ctx = [
            _make_first_chunk_req(rid=i, prompt_len=1024, chunk_size=256,
                                  est_reuse=768)
            for i in range(4)
        ]
        batch = make_batch(ctx, [])

        fake_kv_manager.prepare_resources(batch)

        # count_reusable_blocks must NEVER be called when the guard is off.
        fake_kv_manager.impl.count_reusable_blocks.assert_not_called()
        # All 4 admitted; no skip path engaged.
        assert {r.py_request_id for r in batch.context_requests} == {0, 1, 2, 3}

    def test_f_no_op_when_is_draft(self, fake_kv_manager, make_batch):
        fake_kv_manager.is_draft = True
        ctx = [
            _make_first_chunk_req(rid=i, prompt_len=1024, chunk_size=256,
                                  est_reuse=768)
            for i in range(4)
        ]
        batch = make_batch(ctx, [])

        fake_kv_manager.prepare_resources(batch)

        fake_kv_manager.impl.count_reusable_blocks.assert_not_called()
        assert {r.py_request_id for r in batch.context_requests} == {0, 1, 2, 3}


class TestResetContextRequestsFiltered:
    """Test G — reset_context_requests reclassifies the FILTERED list."""

    def test_g_reset_with_chunk_type_mutation(
        self, fake_kv_manager, make_batch
    ):
        """A request may flip is_last_context_chunk inside add_sequence_batch
        (e.g., reuse covers most of the prompt, leaving only the last chunk).
        Verify that after prepare_resources, the request lands in
        context_requests_last_chunk via the reset_context_requests call."""
        fake_kv_manager.impl.count_reusable_blocks.return_value = 0
        # Start with a non-last chunk that gets flipped during add_sequence_batch.
        fc = _make_first_chunk_req(rid=0, prompt_len=1024, chunk_size=256,
                                   est_reuse=0)
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
    """Test H — kv_connector returns False routes to elif branch."""

    def test_h_connector_should_not_add_charges_estimated_reuse(
        self, fake_kv_manager, make_batch
    ):
        """When kv_connector_manager.should_add_sequence(req) returns False
        for a first-chunk req, resource_manager.py:741-744 charges the
        budget using req.estimated_reusable_tokens (NOT actual reuse, since
        we skip the count_reusable_blocks call for these). The req is also
        NOT added to add_sequence_batch."""
        connector = Mock()
        # req0: rejected by connector → elif branch
        # req1, req2: accepted by connector → if branch
        connector.should_add_sequence.side_effect = [False, True, True]
        fake_kv_manager.kv_connector_manager = connector
        fake_kv_manager.impl.count_reusable_blocks.return_value = 0

        ctx = [
            _make_first_chunk_req(rid=i, prompt_len=512, chunk_size=256,
                                  est_reuse=448)
            for i in range(3)
        ]
        batch = make_batch(ctx, [])

        fake_kv_manager.prepare_resources(batch)

        # Arithmetic walk:
        #   Initial budget = 512 - 0(gen) - 0(non-first) = 512
        #   req0 elif: helper(448, 256, 512): P+chunk=704 >= 512 → max(1, 64) = 64
        #              budget -= 64 → remaining 448
        #   req1 if: count_reusable=0 → helper(0, 256, 512)=256 ≤ 448 → admit, budget 192
        #   req2 if: helper=256 > 192 → SKIP
        surviving = [r.py_request_id for r in batch.context_requests]
        assert surviving == [0, 1], f"expected [0, 1], got {surviving}"
        # Only req1 went into add_sequence_batch (req0 was rejected by connector).
        infos = fake_kv_manager.impl.add_sequence_batch.call_args[0][0]
        assert [info[0] for info in infos] == [1]


class TestKvConnectorCallbacks:
    """Tests I, J — connector update_state_after_alloc and build_scheduler_output."""

    def test_i_update_state_after_alloc_only_for_admitted(
        self, fake_kv_manager, make_batch
    ):
        connector = Mock()
        connector.should_add_sequence.return_value = True
        fake_kv_manager.kv_connector_manager = connector
        fake_kv_manager.get_cache_indices = Mock(return_value=[10, 20])

        # 3 ctx reqs: budget allows 2 to be admitted, 1 skipped.
        fake_kv_manager.impl.count_reusable_blocks.return_value = 0
        ctx = [
            _make_first_chunk_req(rid=i, prompt_len=1024, chunk_size=256,
                                  est_reuse=768)
            for i in range(3)
        ]
        batch = make_batch(ctx, [])

        fake_kv_manager.prepare_resources(batch)

        # update_state_after_alloc should be called once per admitted req (2).
        # Skipped req2 should NOT trigger the callback.
        admitted_ids = [
            call.args[0].py_request_id
            for call in connector.update_state_after_alloc.call_args_list
        ]
        assert sorted(admitted_ids) == [0, 1]

    def test_j_build_scheduler_output_after_filter(
        self, fake_kv_manager, make_batch
    ):
        connector = Mock()
        connector.should_add_sequence.return_value = True
        fake_kv_manager.kv_connector_manager = connector

        fake_kv_manager.impl.count_reusable_blocks.return_value = 0
        ctx = [
            _make_first_chunk_req(rid=i, prompt_len=1024, chunk_size=256,
                                  est_reuse=768)
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
    """Test K — ctx request with draft tokens triggers extra add_token calls."""

    def test_k_context_with_draft_tokens(
        self, fake_kv_manager, make_batch
    ):
        # num_extra_kv_tokens=0, num_draft_tokens=3 → add_token called 3 times.
        fake_kv_manager.impl.count_reusable_blocks.return_value = 0
        ctx = [
            _make_first_chunk_req(rid=0, prompt_len=512, chunk_size=256,
                                  est_reuse=0, num_draft_tokens=3)
        ]
        batch = make_batch(ctx, [])

        fake_kv_manager.prepare_resources(batch)

        assert fake_kv_manager.impl.add_token.call_count == 3


class TestCountReusableBlocksCallShape:
    """Test L — the budget guard calls count_reusable_blocks with the
    expected argument shape. A signature drift would silently break
    re-probing."""

    def test_l_count_reusable_blocks_call_args(
        self, fake_kv_manager, make_batch
    ):
        sentinel_unique = [101, 102, 103, 104]
        fake_kv_manager.impl.count_reusable_blocks.return_value = 0
        req = _make_first_chunk_req(rid=0, prompt_len=1024, chunk_size=256,
                                    est_reuse=0,
                                    unique_tokens=sentinel_unique)
        batch = make_batch([req], [])

        fake_kv_manager.prepare_resources(batch)

        fake_kv_manager.impl.count_reusable_blocks.assert_called_once_with(
            sentinel_unique, req, False)


class TestNonFirstChunkFallthrough:
    """Test M (catalogue) — non-first-chunk reqs fall through both if/elif
    branches and are kept by reset_context_requests, but NOT added to
    add_sequence_batch."""

    def test_m_non_first_chunk_falls_through(
        self, fake_kv_manager, make_batch
    ):
        nf = _make_non_first_chunk_req(rid=0, chunk_size=128,
                                       current_pos=128, prompt_len=1024)
        batch = make_batch([nf], [])

        fake_kv_manager.prepare_resources(batch)

        assert [r.py_request_id for r in batch.context_requests] == [0]
        fake_kv_manager.impl.add_sequence_batch.assert_not_called()


# ===========================================================================
# Tier 3: deterministic DYN-2868 regression
# ===========================================================================
def _model_engine_total_tokens(scheduled_batch):
    """Reimplement model_engine.py:2291-2297 + 2615-2616 token counting,
    INDEPENDENT of _estimate_post_reuse_compute. The negative control
    (Test N) patches the helper to 0; the verifier MUST not go through
    the same code path or it would self-invalidate."""
    total = 0
    for req in scheduled_batch.context_requests:
        begin = req.context_current_position
        all_tokens = req.get_tokens(0)
        total += max(0, min(req.context_chunk_size, len(all_tokens) - begin))
    for req in scheduled_batch.generation_requests:
        # Draft path (model_engine.py:2399, 2459): 1 + len(py_draft_tokens),
        # NOT multiplied by beam.
        # No-draft path (model_engine.py:2615-2616): one position per beam.
        if len(req.py_draft_tokens) > 0:
            total += 1 + len(req.py_draft_tokens)
        else:
            total += req.sampling_config.beam_width
    return total


class TestDYN2868Regression:
    """Tests M, N — the regression-net pair."""

    def test_m_invariant_passes_with_guard(
        self, fake_kv_manager, make_batch
    ):
        """The PR's purpose: the guard prevents post-prepare_resources
        token count from exceeding max_num_tokens under run-14 conditions."""
        fake_kv_manager.impl.count_reusable_blocks.return_value = 0  # eviction

        ctx = [
            _make_first_chunk_req(rid=i, prompt_len=512, chunk_size=256,
                                  est_reuse=448)
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
            f"DYN-2868 regression: post-prepare total {total} > "
            f"max_num_tokens {fake_kv_manager.max_num_tokens}. Surviving: "
            f"{[r.py_request_id for r in batch.context_requests]}"
        )

    def test_n_invariant_breaks_when_guard_disabled(
        self, fake_kv_manager, make_batch, monkeypatch
    ):
        """Negative control: with the guard disabled (helper patched → 0),
        the same scenario admits ALL 4 reqs and overshoots.

        Trace with helper=0:
          - Initial budget = 512
          - req_compute = 0 for all reqs
          - line 725: `0 > 512` is False; no skips
          - line 734: budget -= 0; remaining stays 512
          - All 4 admitted → total = 4 * 256 = 1024 > 512.

        Verifier uses _model_engine_total_tokens (NOT the patched helper),
        so this is not self-invalidating."""
        monkeypatch.setattr(
            KVCacheManager, "_estimate_post_reuse_compute",
            lambda self, *a, **k: 0,
        )
        fake_kv_manager.impl.count_reusable_blocks.return_value = 0

        ctx = [
            _make_first_chunk_req(rid=i, prompt_len=512, chunk_size=256,
                                  est_reuse=448)
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
            f"Negative control: expected overshoot, got total={total} ≤ "
            f"max={fake_kv_manager.max_num_tokens}. The synthetic scenario "
            f"no longer triggers DYN-2868 with the guard disabled — "
            f"strengthen the workload (more requests / larger chunks)."
        )
