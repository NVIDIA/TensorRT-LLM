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
"""The spec-recompute tail: cap reuse so a prompt tail passes the target forward.

Hidden-state-conditioned drafters capture target hidden states only for tokens
that physically run through a target forward, so a prefix-cache hit starves the
drafter of context. ``_spec_recompute_target`` (V2 manager) and
``KVCacheManager._maybe_rewind_reused_context`` (V1 manager) rewind the context
start of a cache-hit request to a block-aligned position that leaves at least
``context_recompute_tail`` prompt tokens to recompute.

``_spec_recompute_target`` is deliberately a module-level function taking the
manager scalars explicitly: the V2 scheduler tests bind the real
``prepare_context`` onto a bare ``Mock`` manager, where a method-dispatched
helper would resolve to an auto-created Mock attribute and feed a Mock into the
cursor arithmetic. The request-only no-op guards run before ``tail`` is read,
so that binding stays safe.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    KVCacheManagerV2,
    _spec_recompute_target,
)
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager

pytestmark = pytest.mark.cpu_only


def _req(prompt_len, is_dummy=False):
    return SimpleNamespace(prompt_len=prompt_len, is_dummy=is_dummy)


class TestSpecRecomputeTarget:
    def test_nothing_committed_is_a_no_op(self):
        req = _req(prompt_len=16)
        assert _spec_recompute_target(req, 0, tail=4, tokens_per_block=4, is_draft=False) == 0

    def test_the_draft_pool_never_rewinds(self):
        req = _req(prompt_len=16)
        assert _spec_recompute_target(req, 8, tail=4, tokens_per_block=4, is_draft=True) == 8

    def test_a_dummy_request_never_rewinds(self):
        req = _req(prompt_len=16, is_dummy=True)
        assert _spec_recompute_target(req, 8, tail=4, tokens_per_block=4, is_draft=False) == 8

    def test_a_zero_tail_disables_the_recompute(self):
        req = _req(prompt_len=16)
        assert _spec_recompute_target(req, 8, tail=0, tokens_per_block=4, is_draft=False) == 8

    def test_a_negative_tail_forces_a_full_re_prefill(self):
        req = _req(prompt_len=10)
        assert _spec_recompute_target(req, 8, tail=-1, tokens_per_block=4, is_draft=False) == 0

    def test_the_rewind_target_is_block_aligned_and_covers_the_tail(self):
        req = _req(prompt_len=13)
        # 13 - 12 = 1 < 6: rewind to floor_block(13 - 6) = 4, leaving 9 >= 6
        # prompt tokens to recompute from a block boundary.
        assert _spec_recompute_target(req, 12, tail=6, tokens_per_block=4, is_draft=False) == 4

    def test_an_uncommitted_span_covering_the_tail_is_left_alone(self):
        req = _req(prompt_len=12)
        # 12 - 4 = 8 >= 4: the natural recompute already covers the tail.
        assert _spec_recompute_target(req, 4, tail=4, tokens_per_block=4, is_draft=False) == 4

    def test_request_only_guards_run_before_the_manager_scalars_are_read(self):
        # The V2 scheduler tests bind the real prepare_context onto a Mock
        # manager, so tail/is_draft arrive as auto-created Mock attributes.
        # A zero-reuse request must pass through before either is inspected.
        req = _req(prompt_len=16)
        assert (
            _spec_recompute_target(req, 0, tail=Mock(), tokens_per_block=Mock(), is_draft=Mock())
            == 0
        )


class _V1Req:
    """The slice of LlmRequest that _maybe_rewind_reused_context touches."""

    def __init__(self, prompt_len, prepopulated):
        self.prompt_len = prompt_len
        self.prepopulated_prompt_len = prepopulated
        self.context_current_position = prepopulated
        self.context_chunk_size = prompt_len - prepopulated
        self.setter_calls = []

    def set_prepopulated_prompt_len(self, value, tokens_per_block):
        self.setter_calls.append((value, tokens_per_block))
        self.prepopulated_prompt_len = value
        # Mirrors the C++ setter: only a nonzero value moves the position.
        if value:
            self.context_current_position = value


def _rewind(reqs, tail, tokens_per_block=4):
    mgr = SimpleNamespace(_spec_recompute_tail=tail, tokens_per_block=tokens_per_block)
    KVCacheManager._maybe_rewind_reused_context(mgr, list(reqs))


class TestV1RewindReusedContext:
    def test_a_request_without_reuse_is_untouched(self):
        req = _V1Req(prompt_len=16, prepopulated=0)
        _rewind([req], tail=4)
        assert req.setter_calls == []
        assert req.context_chunk_size == 16

    def test_a_full_hit_is_rewound_to_a_block_aligned_tail(self):
        req = _V1Req(prompt_len=16, prepopulated=16)
        _rewind([req], tail=4)
        assert req.setter_calls == [(12, 4)]
        assert req.context_current_position == 12
        assert req.context_chunk_size == 4

    def test_a_negative_tail_forces_a_full_re_prefill(self):
        req = _V1Req(prompt_len=16, prepopulated=16)
        _rewind([req], tail=-1)
        assert req.setter_calls == [(0, 4)]
        # The C++ setter ignores zero, so the rewind resets the position
        # explicitly.
        assert req.context_current_position == 0
        assert req.context_chunk_size == 16

    def test_a_hit_already_leaving_the_tail_uncached_is_untouched(self):
        req = _V1Req(prompt_len=16, prepopulated=4)
        _rewind([req], tail=4)
        assert req.setter_calls == []
        assert req.context_chunk_size == 12


class _ConnectorReq:
    """The slice of LlmRequest that _prepare_connector_prefix_reservation touches."""

    def __init__(self, prompt_len):
        self.py_request_id = 1
        self.prompt_len = prompt_len
        self.is_dummy = False
        self.is_first_context_chunk = True
        self.py_connector_allocation_reported = False
        self.py_connector_served_position = 0
        self.context_current_position = 0
        self.prepopulated_prompt_len = 0
        self.context_chunk_size = prompt_len

    @property
    def context_remaining_length(self):
        return self.prompt_len - self.context_current_position

    def set_prepopulated_prompt_len(self, value, tokens_per_block):
        self.prepopulated_prompt_len = value
        # Mirrors the C++ setter: only a nonzero value moves the position.
        if value:
            self.context_current_position = value


def _reserve(req, *, local_end, offered_end, tail, tokens_per_block=4):
    connector = Mock()
    connector.should_add_sequence.return_value = True
    connector.reserve_prefix.return_value = SimpleNamespace(end=offered_end)
    connector.trim_prefix_reservation.side_effect = lambda req, start, end: SimpleNamespace(end=end)
    mgr = Mock()
    mgr._connector_reservations_enabled.return_value = True
    mgr._connector_may_serve.return_value = True
    mgr.kv_connector_manager = connector
    mgr.kv_cache_map = {req.py_request_id: SimpleNamespace(num_committed_tokens=local_end)}
    mgr.tokens_per_block = tokens_per_block
    mgr._spec_recompute_tail = tail
    mgr.is_draft = False
    KVCacheManagerV2._prepare_connector_prefix_reservation(mgr, req)
    return connector


class TestConnectorReservationRecomputeCap:
    """A connector-served prefix starves the drafter exactly like local reuse:
    served tokens never pass a target forward, and once the load is accepted
    the cursor cannot rewind below ``py_connector_served_position``. The
    reservation end is therefore capped at the same recomputed-tail boundary
    as local reuse, so the tail stays local by construction."""

    def test_a_reservation_is_capped_so_the_tail_stays_local(self):
        req = _ConnectorReq(prompt_len=33)
        connector = _reserve(req, local_end=8, offered_end=32, tail=6)
        # floor_block(33 - 6) = 24: at least the last 6 prompt tokens run
        # through the local target forward.
        connector.trim_prefix_reservation.assert_called_once_with(req, 8, 24)
        assert req.context_current_position == 24
        assert req.context_chunk_size == 9

    def test_a_reservation_already_leaving_the_tail_local_is_untouched(self):
        req = _ConnectorReq(prompt_len=33)
        connector = _reserve(req, local_end=8, offered_end=24, tail=6)
        # 33 - 24 = 9 >= 6: the unserved span already recomputes the tail.
        connector.trim_prefix_reservation.assert_called_once_with(req, 8, 24)

    def test_a_full_re_prefill_tail_releases_the_reservation(self):
        req = _ConnectorReq(prompt_len=33)
        connector = _reserve(req, local_end=8, offered_end=32, tail=-1)
        connector.release_prefix_reservation.assert_called_once_with(req)
        connector.trim_prefix_reservation.assert_not_called()
        assert req.context_current_position == 0

    def test_a_cap_at_or_below_the_local_commit_releases_the_reservation(self):
        req = _ConnectorReq(prompt_len=33)
        # floor_block(33 - 20) = 12 <= local 16: nothing left worth serving.
        connector = _reserve(req, local_end=16, offered_end=32, tail=20)
        connector.release_prefix_reservation.assert_called_once_with(req)

    def test_a_zero_tail_leaves_the_reservation_alone(self):
        req = _ConnectorReq(prompt_len=33)
        connector = _reserve(req, local_end=8, offered_end=32, tail=0)
        connector.trim_prefix_reservation.assert_called_once_with(req, 8, 32)
        assert req.context_current_position == 32
