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
"""A new context request waits for a prefix another request is prefilling."""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache import kv_cache_manager_v2 as v2
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    KVCacheManagerV2,
    _InflightPrefixIndex,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import MambaHybridCacheManagerV2

TOKENS_PER_BLOCK = 4
UNIT_TOKENS = v2._INFLIGHT_PREFIX_UNIT_BLOCKS * TOKENS_PER_BLOCK
MIN_GAIN_TOKENS = v2._INFLIGHT_PREFIX_MIN_GAIN_BLOCKS * TOKENS_PER_BLOCK


def _request(request_id, tokens, position=0, first_chunk=True, salt=None):
    return SimpleNamespace(
        py_request_id=request_id,
        tokens=list(tokens),
        prompt_len=len(tokens),
        context_current_position=position,
        context_remaining_length=len(tokens) - position,
        is_first_context_chunk=first_chunk,
        is_dummy_request=False,
        is_disagg_generation_init_state=False,
        lora_task_id=None,
        cache_salt=salt,
        py_multimodal_data=None,
        multimodal_hashes=None,
    )


def _manager(reusable=0, snapshot_ok=True):
    manager = object.__new__(KVCacheManagerV2)
    manager.tokens_per_block = TOKENS_PER_BLOCK
    manager.kv_cache_map = {}
    manager._inflight_prefix = _InflightPrefixIndex(UNIT_TOKENS)
    manager._reuse_token_source = lambda req: req.tokens
    manager.probe_context_reuse = lambda req: reusable
    manager.requested = []

    def request_snapshot_point(owner, point):
        manager.requested.append((owner.py_request_id, point))
        return snapshot_ok

    manager._request_snapshot_point = request_snapshot_point
    return manager


def _start_prefill(manager, request):
    """What prepare_context_cache does for a fresh claim."""
    manager.kv_cache_map[request.py_request_id] = object()
    manager._inflight_prefix_fingerprints(request)
    manager._inflight_prefix.add(request)


SHARED = list(range(1, 2 * MIN_GAIN_TOKENS + 1))


def test_waits_for_a_prefilling_request_and_requests_its_snapshot():
    manager = _manager()
    owner = _request(1, SHARED + [7] * 40, position=UNIT_TOKENS)
    _start_prefill(manager, owner)
    waiter = _request(2, SHARED + [9] * 40)

    assert manager.plan_context_waits([owner, waiter]) == {2: None}
    assert manager.requested == [(1, len(SHARED))]
    assert manager._inflight_prefix.wait_passes == {2: 1}


def test_waits_for_a_new_request_ahead_in_the_same_pass():
    manager = _manager()
    first = _request(1, SHARED + [7] * 40)
    second = _request(2, SHARED + [9] * 40)
    third = _request(3, SHARED + [5] * 40)

    assert manager.plan_context_waits([first, second, third]) == {2: 1, 3: 1}
    assert {owner for owner, _ in manager.requested} == {1}


@pytest.mark.parametrize(
    "reusable,snapshot_ok,salt,waits",
    [
        (0, True, None, True),
        # The reuse tree already offers most of the shared prefix.
        (len(SHARED) - MIN_GAIN_TOKENS + 1, True, None, False),
        # The owner cannot make the shared depth reusable.
        (0, False, None, False),
        # A different reuse scope never shares a prefix.
        (0, True, "other", False),
    ],
)
def test_wait_conditions(reusable, snapshot_ok, salt, waits):
    manager = _manager(reusable=reusable, snapshot_ok=snapshot_ok)
    owner = _request(1, SHARED + [7] * 40, position=UNIT_TOKENS)
    _start_prefill(manager, owner)
    waiter = _request(2, SHARED + [9] * 40, salt=salt)

    assert (manager.plan_context_waits([waiter]) == {2: None}) is waits


def test_short_shared_prefix_does_not_wait():
    manager = _manager()
    short = list(range(1, MIN_GAIN_TOKENS))
    owner = _request(1, short + [7] * 400, position=UNIT_TOKENS)
    _start_prefill(manager, owner)

    assert manager.plan_context_waits([_request(2, short + [9] * 400)]) == {}


def test_wait_is_bounded_and_ends_with_the_owner_context():
    manager = _manager()
    owner = _request(1, SHARED + [7] * 40, position=UNIT_TOKENS)
    _start_prefill(manager, owner)
    waiter = _request(2, SHARED + [9] * 40)

    for _ in range(v2._INFLIGHT_PREFIX_MAX_WAIT_PASSES):
        assert manager.plan_context_waits([waiter]) == {2: None}
    assert manager.plan_context_waits([waiter]) == {}

    manager._inflight_prefix.wait_passes.clear()
    owner.context_remaining_length = 0
    assert manager.plan_context_waits([waiter]) == {}


def test_requests_with_a_claim_or_a_later_chunk_never_wait():
    manager = _manager()
    owner = _request(1, SHARED + [7] * 40, position=UNIT_TOKENS)
    _start_prefill(manager, owner)
    claimed = _request(2, SHARED + [9] * 40)
    manager.kv_cache_map[2] = object()
    continuing = _request(3, SHARED + [5] * 40, first_chunk=False)

    assert manager.plan_context_waits([claimed, continuing]) == {}


def test_freed_owner_leaves_the_index():
    manager = _manager()
    owner = _request(1, SHARED + [7] * 40, position=UNIT_TOKENS)
    _start_prefill(manager, owner)
    manager._inflight_prefix.remove(1)

    assert manager.plan_context_waits([_request(2, SHARED + [9] * 40)]) == {}


def test_disabled_without_block_reuse():
    manager = _manager()
    manager._inflight_prefix = None

    assert manager.plan_context_waits([_request(2, SHARED)]) == {}


def _recurrent_manager():
    manager = object.__new__(MambaHybridCacheManagerV2)
    manager.tokens_per_block = 32
    manager._requested_snapshot_points = {}
    return manager


def test_recurrent_manager_requests_a_block_aligned_snapshot():
    manager = _recurrent_manager()
    owner = SimpleNamespace(
        py_request_id=1,
        prompt_len=1000,
        context_current_position=0,
        expect_snapshot_points=[1000],
        is_context_init_state=True,
        is_dummy_request=False,
    )

    assert manager._request_snapshot_point(owner, 700)
    assert owner.expect_snapshot_points == [672, 1000]
    assert manager._request_snapshot_point(owner, 690)
    assert manager._requested_snapshot_points == {1: {672}}

    # The point merges into the snapshot points of later passes.
    owner.expect_snapshot_points = [1000]
    manager._apply_requested_snapshot_points(owner)
    assert owner.expect_snapshot_points == [672, 1000]


@pytest.mark.parametrize("position,point", [(700, 700), (0, 1024), (0, 10)])
def test_recurrent_manager_rejects_points_the_owner_cannot_take(position, point):
    manager = _recurrent_manager()
    owner = SimpleNamespace(
        py_request_id=1,
        prompt_len=1000,
        context_current_position=position,
        expect_snapshot_points=[1000],
    )

    assert not manager._request_snapshot_point(owner, point)
    assert owner.expect_snapshot_points == [1000]


@pytest.mark.parametrize(
    "record_branch_snapshots,position,point,accepted,points",
    [
        (False, 0, 700, True, [1000]),
        (True, 0, 700, True, [696, 1000]),
        (True, 700, 700, False, [1000]),
        (True, 0, 1024, False, [1000]),
    ],
    ids=["attention", "swa-chunk-end", "swa-passed", "swa-beyond-prompt"],
)
def test_attention_manager_ends_an_swa_chunk_at_the_requested_point(
    record_branch_snapshots, position, point, accepted, points
):
    """Where only chunk-end SWA windows reach the radix tree, the owner ends a
    chunk at the block-aligned point; other attention caches need nothing."""
    manager = object.__new__(KVCacheManagerV2)
    manager.tokens_per_block = 8
    manager._record_branch_snapshots = record_branch_snapshots
    owner = SimpleNamespace(
        py_request_id=1,
        prompt_len=1000,
        context_current_position=position,
        expect_snapshot_points=[1000],
    )

    assert manager._request_snapshot_point(owner, point) is accepted
    assert owner.expect_snapshot_points == points
