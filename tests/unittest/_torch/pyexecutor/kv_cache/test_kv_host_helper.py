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
"""KvHostHelper windows and the KVCacheManagerV2 deferred closes it runs, on mock KvCaches. No GPU
required."""

import threading
import time
from collections import deque
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache.host_helper import (
    CLAIM,
    PREFETCH,
    RELEASE,
    KvHostHelper,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2

pytestmark = pytest.mark.cpu_only


def _helper() -> KvHostHelper:
    """A helper driven by `run_one` on the test thread."""
    return KvHostHelper(None, start_thread=False)


# ---------------------------------------------------------------------------
# KvHostHelper
# ---------------------------------------------------------------------------


def test_nothing_runs_outside_a_window() -> None:
    helper = _helper()
    ran = []
    for kind in (RELEASE, CLAIM, PREFETCH):
        helper.submit(kind, lambda kind=kind: ran.append(kind))

    assert helper.run_one() is False
    helper.begin_wait()
    helper.end_wait()
    helper.begin_forward(may_capture=True)
    assert helper.run_one() is False
    helper.end_forward()
    assert ran == []


def test_the_wait_window_runs_releases_only() -> None:
    helper = _helper()
    ran = []
    helper.submit(CLAIM, lambda: ran.append("claim"))
    helper.submit(PREFETCH, lambda: ran.append("prefetch"))
    helper.submit(RELEASE, lambda: ran.append("release"))

    helper.begin_wait()
    while helper.run_one():
        pass

    assert ran == ["release"]
    assert helper.num_pending(CLAIM) == 1 and helper.num_pending(PREFETCH) == 1


def test_a_forward_without_capture_runs_every_kind_in_priority_order() -> None:
    helper = _helper()
    ran = []
    helper.submit(PREFETCH, lambda: ran.append("prefetch"))
    helper.submit(CLAIM, lambda: ran.append("claim"))
    helper.submit(RELEASE, lambda: ran.append("release"))

    helper.begin_forward(may_capture=False)
    while helper.run_one():
        pass

    assert ran == ["release", "claim", "prefetch"]
    assert helper.num_run == [1, 1, 1]


def test_claims_and_prefetches_start_only_early_in_a_forward() -> None:
    helper = KvHostHelper(None, start_thread=False, forward_start_fraction=0.5)
    helper.begin_forward(may_capture=False)
    helper.end_forward()
    helper._forward_ema = 0.01  # 10 ms forwards
    ran = []
    helper.submit(CLAIM, lambda: ran.append("claim"))
    helper.submit(RELEASE, lambda: ran.append("release"))

    helper.begin_forward(may_capture=False)
    time.sleep(0.02)  # past half of the running average
    while helper.run_one():
        pass

    assert ran == ["release"]
    helper.end_forward()
    helper.begin_forward(may_capture=False)
    assert helper.run_one() is True
    assert ran == ["release", "claim"]


def test_a_forward_that_may_capture_waits_for_the_running_item() -> None:
    helper = KvHostHelper(None)
    started, release = threading.Event(), threading.Event()
    done = []

    def item():
        started.set()
        release.wait(5)
        done.append(True)

    try:
        helper.submit(RELEASE, item)
        helper.begin_wait()
        assert started.wait(5)
        helper.end_wait()
        threading.Timer(0.05, release.set).start()

        helper.begin_forward(may_capture=True)

        assert done == [True]
        assert helper.num_forward_waits == 1
    finally:
        helper.stop()


def test_an_item_error_is_raised_on_the_executor_thread() -> None:
    helper = _helper()

    def failing():
        raise RuntimeError("close failed")

    helper.submit(RELEASE, failing)
    helper.begin_wait()
    assert helper.run_one() is True

    with pytest.raises(RuntimeError, match="close failed"):
        helper.begin_forward(may_capture=True)
    helper.begin_forward(may_capture=True)  # raised once


def test_drain_runs_inline_and_drops_the_queued_items() -> None:
    helper = _helper()
    ran = []
    helper.submit(RELEASE, lambda: ran.append("ticket"))

    helper.drain(RELEASE, lambda: ran.append("inline"))

    assert ran == ["inline"]
    assert helper.num_pending(RELEASE) == 0


def test_the_thread_runs_items_and_stops() -> None:
    helper = KvHostHelper(None)
    done = threading.Event()
    try:
        helper.submit(RELEASE, done.set)
        assert not done.wait(0.05)  # no window yet
        helper.begin_wait()
        assert done.wait(5)
    finally:
        helper.stop()
    with pytest.raises(RuntimeError, match="stopped"):
        helper.submit(RELEASE, done.set)


# ---------------------------------------------------------------------------
# Deferred closes on the helper
# ---------------------------------------------------------------------------


def _manager(request_ids=(), is_draft=False):
    """A manager whose release path runs for real on mock KvCaches."""
    manager = object.__new__(KVCacheManagerV2)
    caches = {rid: Mock(beam_width=1, locality_domain_id=0) for rid in request_ids}
    manager.kv_cache_map = dict(caches)
    manager.impl = Mock()
    manager.index_mapper = Mock()
    manager.index_mapper.num_free_slots.return_value = 8
    manager.num_pools = 1
    manager.sparse_metadata_batch = None
    manager.is_draft = is_draft
    manager.kv_connector_manager = None
    manager.conversation_manager = None
    manager.enable_block_reuse = True
    manager.tokens_per_block = 4
    manager._has_cp_helix = False
    manager._early_freed_index_requests = set()
    manager._mixed_precision_low_prefix_len = {}
    manager._deferred_closes = deque()
    manager._pending_preemption = {}
    manager._allocated_draft_lens = {}
    manager._inflight_draft_lens = {}
    manager._request_stats_enabled_ids = set()
    manager._fresh_pages_filled = {}
    manager._disagg_receive_ready = {}
    manager._on_allocation_stream = set()
    manager._allocation_ready = {}
    return manager, caches


def _finished(request_id, *, generation_only=False, context_only=False):
    return Mock(
        py_request_id=request_id,
        is_generation_only_request=generation_only,
        is_context_only_request=context_only,
        is_dummy=False,
    )


def test_deferred_closes_run_on_the_helper_in_a_wait() -> None:
    manager, caches = _manager([7])
    manager.enable_deferred_close(True)
    helper = manager.enable_host_helper(None, releases=True, start_thread=False)

    manager.free_resources(_finished(7, generation_only=True))

    caches[7].close.assert_not_called()
    assert helper.num_pending(RELEASE) == 1
    helper.begin_wait()
    assert helper.run_one() is True
    caches[7].close.assert_called_once_with()
    manager.impl.clear_stats_excluded.assert_called_once_with(7)
    # The flush reports the close the helper made, so a scheduler out of pages retries.
    assert manager.flush_deferred_closes() is True
    assert manager.flush_deferred_closes() is False


def test_a_flush_closes_what_the_helper_has_not_started() -> None:
    manager, caches = _manager([7, 8])
    manager.enable_deferred_close(True)
    helper = manager.enable_host_helper(None, releases=True, start_thread=False)
    manager.free_resources(_finished(7, generation_only=True))
    manager.free_resources(_finished(8, generation_only=True))

    assert manager.flush_deferred_closes() is True

    caches[7].close.assert_called_once_with()
    caches[8].close.assert_called_once_with()
    helper.begin_wait()
    assert helper.run_one() is False  # its tickets were dropped
    assert not manager._deferred_closes


def test_context_only_requests_are_deferred_only_when_asked() -> None:
    manager, caches = _manager([7, 8])
    manager.enable_deferred_close(True)
    manager.free_resources(_finished(7, context_only=True))
    caches[7].close.assert_called_once_with()

    manager.enable_deferred_close(True, context_only=True)
    manager.free_resources(_finished(8, context_only=True))
    caches[8].close.assert_not_called()
    assert [rid for rid, _ in manager._deferred_closes] == [8]


def test_a_deferred_release_forgets_the_request_at_once() -> None:
    manager, _ = _manager([7])
    manager.impl.num_locality_domains = 2  # read through the impl
    manager._locality_domain_load = [1, 0]
    manager._mixed_precision_low_prefix_len[7] = 16
    manager.enable_deferred_close(True)

    manager.free_resources(_finished(7, generation_only=True))

    assert manager._locality_domain_load == [0, 0]
    assert 7 not in manager._mixed_precision_low_prefix_len


def test_the_helper_does_not_release_for_a_cache_lender() -> None:
    manager, _ = _manager([7])
    manager._sharing = Mock()
    manager.enable_deferred_close(True)

    assert manager.enable_host_helper(None, releases=True, start_thread=False) is None
    assert manager._helper_releases is False


def test_stopping_the_helper_runs_the_pending_closes() -> None:
    manager, caches = _manager([7])
    manager.enable_deferred_close(True)
    manager.enable_host_helper(None, releases=True, start_thread=False)
    manager.free_resources(_finished(7, generation_only=True))

    manager.stop_host_helper()

    caches[7].close.assert_called_once_with()
    assert manager._host_helper is None
