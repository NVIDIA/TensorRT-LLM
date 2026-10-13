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
drafter of context. ``_spec_recompute_claim_limit`` (V2 manager) caps the reuse
CLAIM of a cache-hit request at a block-aligned position that leaves at least
``context_recompute_tail`` prompt tokens to recompute; capping the claim (rather
than rewinding the cursor after it) keeps every page the recompute needs
materialized, which a rewind cannot guarantee for sliding-window layers.
``KVCacheManager._maybe_rewind_reused_context`` (V1 manager) rewinds the context
start instead, and is therefore gated to managers without windowed layers.

``_spec_recompute_claim_limit`` is deliberately a module-level function taking
the manager scalars explicitly: the V2 scheduler tests bind the real
``prepare_context`` onto a bare ``Mock`` manager (setting ``_spec_recompute_tail``
to 0), where a method-dispatched helper would resolve to an auto-created Mock
attribute and feed a Mock into the cursor arithmetic.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    KVCacheManagerV2,
    _spec_recompute_claim_limit,
)
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager

pytestmark = pytest.mark.cpu_only


def _req(prompt_len, is_dummy=False):
    return SimpleNamespace(prompt_len=prompt_len, is_dummy=is_dummy)


def _limit(req, *, tail, tokens_per_block=4, is_draft=False):
    return _spec_recompute_claim_limit(
        req, tail=tail, tokens_per_block=tokens_per_block, is_draft=is_draft
    )


class TestSpecRecomputeClaimLimit:
    def test_a_zero_tail_means_no_cap(self):
        assert _limit(_req(prompt_len=16), tail=0) is None

    def test_the_draft_pool_is_never_capped(self):
        assert _limit(_req(prompt_len=16), tail=4, is_draft=True) is None

    def test_a_dummy_request_is_never_capped(self):
        assert _limit(_req(prompt_len=16, is_dummy=True), tail=4) is None

    def test_a_negative_tail_forces_a_full_re_prefill(self):
        assert _limit(_req(prompt_len=10), tail=-1) == 0

    def test_the_cap_is_block_aligned_and_leaves_the_tail(self):
        # floor_block(13 - 6) = 4: a claim of at most 4 leaves 9 >= 6 prompt
        # tokens to recompute from a block boundary.
        assert _limit(_req(prompt_len=13), tail=6) == 4

    def test_a_tail_covering_the_whole_prompt_caps_the_claim_to_zero(self):
        assert _limit(_req(prompt_len=3), tail=4) == 0

    def test_the_tail_guard_runs_before_the_other_manager_scalars_are_read(self):
        # The V2 scheduler tests bind the real prepare_context onto a Mock
        # manager with _spec_recompute_tail = 0; the remaining scalars arrive
        # as auto-created Mock attributes and must not be touched.
        assert (
            _spec_recompute_claim_limit(
                _req(prompt_len=16), tail=0, tokens_per_block=Mock(), is_draft=Mock()
            )
            is None
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


def _v1_tail_gate(mgr, requests):
    """Run the prepare_resources gate slice that guards the V1 rewind."""
    KVCacheManager._maybe_apply_spec_recompute_tail(mgr, requests)


class TestV1WindowedLayerGate:
    """The V1 rewind reattaches no pages, so it is only sound when every
    matched block of every layer is attached: a sliding-window layer detaches
    out-of-window blocks, and a rewound cursor would read and write through
    their placeholder slots."""

    def _mgr(self, *, windows, tail=4, chunked=True):
        return SimpleNamespace(
            _spec_recompute_tail=tail,
            tokens_per_block=4,
            is_draft=False,
            enable_chunked_prefill=chunked,
            max_attention_window_vec=windows,
            max_seq_len=16,
            _maybe_rewind_reused_context=Mock(),
        )

    def test_full_attention_layers_keep_the_tail(self):
        mgr = self._mgr(windows=[16, 16])
        req = _V1Req(prompt_len=16, prepopulated=16)
        _v1_tail_gate(mgr, [req])
        assert mgr._spec_recompute_tail == 4
        mgr._maybe_rewind_reused_context.assert_called_once_with([req])

    def test_a_windowed_layer_disables_the_tail(self):
        mgr = self._mgr(windows=[16, 8])
        req = _V1Req(prompt_len=16, prepopulated=16)
        _v1_tail_gate(mgr, [req])
        assert mgr._spec_recompute_tail == 0
        mgr._maybe_rewind_reused_context.assert_not_called()

    def test_missing_chunked_prefill_disables_the_tail(self):
        mgr = self._mgr(windows=[16], chunked=False)
        _v1_tail_gate(mgr, [_V1Req(prompt_len=16, prepopulated=16)])
        assert mgr._spec_recompute_tail == 0
        mgr._maybe_rewind_reused_context.assert_not_called()

    def test_the_draft_pool_never_rewinds(self):
        mgr = self._mgr(windows=[16])
        mgr.is_draft = True
        _v1_tail_gate(mgr, [_V1Req(prompt_len=16, prepopulated=16)])
        mgr._maybe_rewind_reused_context.assert_not_called()


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
    reservation end is therefore capped at the same claim limit as local
    reuse, so the tail stays local by construction."""

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
