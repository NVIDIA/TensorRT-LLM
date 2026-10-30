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
"""KVCacheManagerV2's opt-in deferred close (TRTLLM_KV_DEFER_CLOSE=1).

While armed, free_resources inside deferring_closes() releases a terminated
request's IndexMapper slot at once and leaves the cache's close() to
flush_deferred_closes(); every other release closes at once. The overlap
executor arms it for its loop, releases terminated requests inside the scope,
and flushes after launching each forward step, before waiting for requests with
none left, and before firing a control request.
"""

import types
from unittest.mock import MagicMock, Mock

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor


def _manager(events, *, enabled=True, num_pools=2, sparse=False):
    mgr = object.__new__(KVCacheManagerV2)
    mgr._defer_close_enabled = enabled
    mgr._pending_preemption = {}
    mgr.kv_connector_manager = None
    mgr.conversation_manager = None
    mgr._allocated_draft_lens = {}
    mgr._request_stats_enabled_ids = set()
    mgr._fresh_pages_filled = {}
    mgr._disagg_receive_ready = {}
    mgr.kv_cache_map = {}
    mgr._early_freed_index_requests = set()
    mgr.num_pools = num_pools
    mgr.is_draft = False
    mgr._stream = types.SimpleNamespace(cuda_stream=0)
    mgr.impl = Mock()
    mgr.impl.clear_stats_excluded.side_effect = lambda rid: events.append(("clear_stats", rid))
    mgr.index_mapper = Mock()
    mgr.index_mapper.remove_sequence.side_effect = lambda rid: events.append(("remove_seq", rid))
    if sparse:
        mgr.sparse_metadata_batch = Mock()
        mgr.sparse_metadata_batch.remove.side_effect = lambda kv: events.append(
            ("sparse_remove", kv.rid)
        )
        mgr.sparse_metadata_batch.close.side_effect = lambda: events.append(("sparse_close",))
    return mgr


def _cache(mgr, events, rid):
    kv = Mock()
    kv.rid = rid
    kv.beam_width = 1
    kv.close.side_effect = lambda: events.append(("close", rid))
    kv.set_base_page_index_buf.side_effect = lambda beam, pool, buf: events.append(
        ("unbind", rid, pool, buf)
    )
    mgr.kv_cache_map[rid] = kv
    return kv


def _request(rid, *, dummy=False):
    return types.SimpleNamespace(py_request_id=rid, is_dummy=dummy)


def _finish(mgr, request):
    """Release *request* as the executor releases a terminated one."""
    with mgr.deferring_closes():
        mgr.free_resources(request)


def test_off_unless_enabled_closes_at_once():
    events = []
    mgr = _manager(events, enabled=False)
    _cache(mgr, events, 7)
    assert mgr.arm_deferred_close(True) is False
    _finish(mgr, _request(7))
    assert events == [("close", 7), ("clear_stats", 7), ("remove_seq", 7)]
    assert mgr._deferred_closes == ()


def test_class_defaults_keep_managers_built_without_init_unchanged():
    events = []
    mgr = _manager(events)
    del mgr._defer_close_enabled
    _cache(mgr, events, 3)
    assert mgr.arm_deferred_close(True) is False
    _finish(mgr, _request(3))
    assert events == [("close", 3), ("clear_stats", 3), ("remove_seq", 3)]


def test_armed_releases_slot_now_and_closes_on_flush():
    events = []
    mgr = _manager(events, num_pools=2)
    kv = _cache(mgr, events, 7)
    assert mgr.arm_deferred_close(True) is True
    _finish(mgr, _request(7))
    # Slot released at once, after the cache is unbound from it; no close yet.
    assert events == [("unbind", 7, 0, None), ("unbind", 7, 1, None), ("remove_seq", 7)]
    kv.close.assert_not_called()
    assert 7 not in mgr.kv_cache_map
    events.clear()
    assert mgr.flush_deferred_closes() == 1
    assert events == [("close", 7), ("clear_stats", 7)]
    assert mgr.flush_deferred_closes() == 0


def test_flush_closes_in_finish_order():
    events = []
    mgr = _manager(events, num_pools=1)
    for rid in (5, 2, 9):
        _cache(mgr, events, rid)
    mgr.arm_deferred_close(True)
    for rid in (5, 2, 9):
        _finish(mgr, _request(rid))
    events.clear()
    assert mgr.flush_deferred_closes() == 3
    assert [e for e in events if e[0] == "close"] == [("close", 5), ("close", 2), ("close", 9)]
    assert [e for e in events if e[0] == "clear_stats"] == [
        ("clear_stats", 5),
        ("clear_stats", 2),
        ("clear_stats", 9),
    ]


def test_each_stats_exclusion_is_cleared_after_its_close():
    events = []
    mgr = _manager(events, num_pools=1)
    for rid in (1, 4):
        _cache(mgr, events, rid)
    mgr.arm_deferred_close(True)
    for rid in (1, 4):
        _finish(mgr, _request(rid))
    events.clear()
    mgr.flush_deferred_closes()
    assert events == [("close", 1), ("clear_stats", 1), ("close", 4), ("clear_stats", 4)]


def test_dummy_requests_close_at_once_while_armed():
    events = []
    mgr = _manager(events, num_pools=1)
    _cache(mgr, events, 11)
    mgr.arm_deferred_close(True)
    _finish(mgr, _request(11, dummy=True))
    assert events == [("close", 11), ("clear_stats", 11), ("remove_seq", 11)]
    assert mgr._deferred_closes == ()


def test_early_released_slot_is_not_released_twice():
    events = []
    mgr = _manager(events, num_pools=1)
    _cache(mgr, events, 6)
    mgr._early_freed_index_requests.add(6)
    mgr.arm_deferred_close(True)
    _finish(mgr, _request(6))
    assert events == []
    assert 6 not in mgr._early_freed_index_requests
    mgr.flush_deferred_closes()
    assert events == [("close", 6), ("clear_stats", 6)]


def test_sparse_batch_is_left_before_the_slot_is_released():
    events = []
    mgr = _manager(events, num_pools=1, sparse=True)
    _cache(mgr, events, 8)
    mgr.arm_deferred_close(True)
    _finish(mgr, _request(8))
    assert events == [("sparse_remove", 8), ("unbind", 8, 0, None), ("remove_seq", 8)]
    mgr.sparse_metadata_batch.record_read.assert_called_once_with(0)


def test_disarm_flushes_and_stops_deferring():
    events = []
    mgr = _manager(events, num_pools=1)
    _cache(mgr, events, 1)
    _cache(mgr, events, 2)
    mgr.arm_deferred_close(True)
    _finish(mgr, _request(1))
    events.clear()
    assert mgr.arm_deferred_close(False) is False
    assert events == [("close", 1), ("clear_stats", 1)]
    events.clear()
    _finish(mgr, _request(2))
    assert events == [("close", 2), ("clear_stats", 2), ("remove_seq", 2)]


def test_shutdown_closes_deferred_caches_before_the_sparse_batch():
    events = []
    mgr = _manager(events, num_pools=1, sparse=True)
    _cache(mgr, events, 3)
    mgr.arm_deferred_close(True)
    _finish(mgr, _request(3))
    events.clear()
    mgr.event_manager = None
    mgr.shutdown()
    assert events.index(("close", 3)) < events.index(("sparse_close",))
    assert mgr._defer_close_armed is False
    assert mgr._deferred_closes == ()


def _overlap_loop_stub(kv_cache_manager):
    executor = object.__new__(PyExecutor)
    executor.device_id = 0
    profiler = MagicMock()
    profiler.__enter__.return_value = Mock()
    executor._profiler = Mock(return_value=profiler)
    executor.hang_detector = MagicMock()
    executor.enable_iter_perf_stats = False
    executor._resource_governor_enabled = False
    executor._is_kv_manager_v2 = True
    executor.kv_cache_manager = kv_cache_manager
    executor._can_pause_for_rebalance = Mock(return_value=False)
    executor._wait_for_model_engine_input_copy = Mock()
    executor._prepare_and_schedule_batch = Mock(return_value=(None, None))
    executor._mm_encoder_item_scheduling_enabled = False
    executor.is_benchmark_disagg = False
    executor._flush_pending_transfer_responses = Mock()
    return executor


def _patch_loop_cuda(monkeypatch):
    for name in ("torch.cuda.set_device", "cudart.cudaSetDevice", "CUASSERT"):
        monkeypatch.setattr(f"tensorrt_llm._torch.pyexecutor.py_executor.{name}", Mock())


def test_overlap_loop_arms_for_its_run_and_disarms_on_exit(monkeypatch):
    mgr = Mock()
    mgr.arm_deferred_close = Mock(side_effect=lambda armed: armed)
    executor = _overlap_loop_stub(mgr)
    _patch_loop_cuda(monkeypatch)

    PyExecutor._executor_loop_overlap(executor)

    assert [c.args for c in mgr.arm_deferred_close.call_args_list] == [(True,), (False,)]
    assert executor._kv_defer_close_armed is False


def test_overlap_loop_leaves_the_manager_alone_when_not_enabled(monkeypatch):
    mgr = Mock()
    mgr.arm_deferred_close = Mock(return_value=False)
    executor = _overlap_loop_stub(mgr)
    _patch_loop_cuda(monkeypatch)

    PyExecutor._executor_loop_overlap(executor)

    mgr.arm_deferred_close.assert_called_once_with(True)
    mgr.flush_deferred_closes.assert_not_called()
    assert executor._kv_defer_close_armed is False


class _QueueWait(Exception):
    pass


def _fetch_stub(kv_cache_manager, events, *, armed=True):
    executor = object.__new__(PyExecutor)
    executor.kv_cache_manager = kv_cache_manager
    executor._kv_defer_close_armed = armed
    executor.kv_connector_manager = None
    executor.control_requests = []
    executor.is_shutdown = False
    executor._has_pending_connector_transfers = Mock(return_value=False)
    executor._disable_mpi = False
    executor.dist = types.SimpleNamespace(rank=0)
    executor.request_accumulated = []
    executor.hang_detector = MagicMock()
    executor.executor_request_queue = Mock()

    def wait(timeout):
        events.append(("queue_wait", timeout))
        raise _QueueWait

    executor.executor_request_queue.get_from_request_queue.side_effect = wait
    return executor


def _fetch(executor, total_num_live_requests, waiting=()):
    try:
        PyExecutor._fetch_and_enqueue_requests(executor, list(waiting), total_num_live_requests)
    except _QueueWait:
        pass


def test_last_request_closes_before_the_idle_queue_wait():
    events = []
    mgr = _manager(events, num_pools=1)
    _cache(mgr, events, 4)
    mgr.arm_deferred_close(True)
    _finish(mgr, _request(4))
    events.clear()
    executor = _fetch_stub(mgr, events)

    _fetch(executor, total_num_live_requests=0)

    # The blocking wait for the next request starts only after the close.
    assert events == [("close", 4), ("clear_stats", 4), ("queue_wait", None)]
    assert mgr._deferred_closes == ()


def test_busy_fetch_leaves_deferred_closes_to_the_loop():
    events = []
    mgr = _manager(events, num_pools=1)
    _cache(mgr, events, 4)
    mgr.arm_deferred_close(True)
    _finish(mgr, _request(4))
    events.clear()
    executor = _fetch_stub(mgr, events)

    _fetch(executor, total_num_live_requests=3)
    _fetch(executor, total_num_live_requests=0, waiting=[object()])

    assert [e[0] for e in events] == ["queue_wait", "queue_wait"]
    assert len(mgr._deferred_closes) == 1


def test_idle_fetch_does_not_flush_when_not_armed():
    events = []
    mgr = Mock()
    executor = _fetch_stub(mgr, events, armed=False)

    _fetch(executor, total_num_live_requests=0)

    mgr.flush_deferred_closes.assert_not_called()
    assert events == [("queue_wait", None)]


def test_rebalance_flushes_deferred_closes_before_adjusting(monkeypatch):
    order = []
    executor = object.__new__(PyExecutor)
    executor.kv_cache_manager = Mock()
    executor.kv_cache_manager.flush_deferred_closes.side_effect = lambda: order.append("flush")
    executor._agreed_need_adjustment = Mock(return_value=True)
    executor._consume_previous_batch_for_rebalance = Mock(
        side_effect=lambda: order.append("consume")
    )
    executor._rebalance_kv_pools_now = Mock(side_effect=lambda: order.append("adjust"))
    executor._kv_defer_close_armed = True
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.py_executor.torch.cuda.current_stream", Mock()
    )

    PyExecutor._maybe_rebalance_kv_pools(executor)

    assert order == ["consume", "flush", "adjust"]


def test_armed_release_outside_the_terminate_scope_closes_at_once():
    # Admission rollbacks and reservation drops call free_resources directly
    # and count on the pages being free when it returns.
    events = []
    mgr = _manager(events, num_pools=1)
    _cache(mgr, events, 7)
    mgr.arm_deferred_close(True)
    mgr.free_resources(_request(7))
    assert events == [("close", 7), ("clear_stats", 7), ("remove_seq", 7)]
    assert mgr._deferred_closes == ()


def test_preemption_releases_pages_at_once_while_armed():
    # preempt_request reports the pages released; the scheduler retries the
    # allocation right after each victim, so the close cannot wait for a flush.
    events = []
    mgr = _manager(events, num_pools=1)
    _cache(mgr, events, 9)
    mgr.arm_deferred_close(True)
    assert mgr.preempt_request(_request(9)) is True
    assert events == [("close", 9), ("clear_stats", 9), ("remove_seq", 9)]
    assert mgr._deferred_closes == ()


def test_preemption_inside_the_terminate_scope_still_closes_at_once():
    events = []
    mgr = _manager(events, num_pools=1)
    _cache(mgr, events, 2)
    _cache(mgr, events, 3)
    mgr.arm_deferred_close(True)
    with mgr.deferring_closes():
        mgr.free_resources(_request(2))
        with mgr.deferring_closes():
            pass
        assert mgr._defer_close_scope is True
    assert mgr._defer_close_scope is False
    mgr.preempt_request(_request(3))
    assert ("close", 3) in events and ("close", 2) not in events
    assert [rid for rid, _ in mgr._deferred_closes] == [2]


def _terminate_stub(kv_cache_manager, *, armed=True):
    executor = object.__new__(PyExecutor)
    executor.kv_cache_manager = kv_cache_manager
    executor._kv_defer_close_armed = armed
    executor.resource_manager = types.SimpleNamespace(
        free_resources=kv_cache_manager.free_resources
    )
    executor._prefetched_request_ids = set()
    executor._disagg_coordinator = Mock()
    executor.gather_all_responses = False
    executor.dist = types.SimpleNamespace(rank=1)
    executor.result_wait_queues = {}
    return executor


def test_terminated_request_defers_and_recompute_pause_does_not():
    events = []
    mgr = _manager(events, num_pools=1)
    _cache(mgr, events, 4)
    _cache(mgr, events, 5)
    mgr.arm_deferred_close(True)
    executor = _terminate_stub(mgr)

    PyExecutor._do_terminate_request(executor, _request(4))
    assert ("close", 4) not in events
    assert [rid for rid, _ in mgr._deferred_closes] == [4]

    # The recompute-pause teardown releases through _free_request_resources.
    PyExecutor._free_request_resources(executor, _request(5))
    assert ("close", 5) in events
    assert [rid for rid, _ in mgr._deferred_closes] == [4]


def test_terminate_closes_at_once_when_not_armed():
    events = []
    mgr = _manager(events, num_pools=1)
    _cache(mgr, events, 6)
    executor = _terminate_stub(mgr, armed=False)

    PyExecutor._do_terminate_request(executor, _request(6))

    assert events == [("close", 6), ("clear_stats", 6), ("remove_seq", 6)]


def _control_stub(kv_cache_manager, events, *, active=()):
    executor = object.__new__(PyExecutor)
    executor.kv_cache_manager = kv_cache_manager
    executor._kv_defer_close_armed = True
    executor.control_requests = [
        types.SimpleNamespace(control_requires_drain=True, control_id=None)
    ]
    executor.active_requests = list(active)
    executor.waiting_queue = []
    executor._has_pending_connector_transfers = Mock(return_value=False)
    executor.control_request_barrier = Mock()
    executor.control_request_barrier.set.side_effect = lambda: events.append(("fire",))
    executor.control_action_done = Mock()
    return executor


def test_drained_control_request_fires_after_the_deferred_closes(monkeypatch):
    events = []
    mgr = _manager(events, num_pools=1)
    _cache(mgr, events, 8)
    mgr.arm_deferred_close(True)
    _finish(mgr, _request(8))
    events.clear()
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.py_executor.torch.cuda.synchronize",
        Mock(side_effect=lambda: events.append(("sync",))),
    )
    executor = _control_stub(mgr, events)

    PyExecutor._handle_control_request(executor)

    assert events == [("close", 8), ("clear_stats", 8), ("sync",), ("fire",)]
    assert mgr._deferred_closes == ()


def test_parked_control_request_leaves_deferred_closes_alone():
    events = []
    mgr = _manager(events, num_pools=1)
    _cache(mgr, events, 8)
    mgr.arm_deferred_close(True)
    _finish(mgr, _request(8))
    events.clear()
    executor = _control_stub(mgr, events, active=[object()])

    PyExecutor._handle_control_request(executor)

    assert events == []
    assert len(mgr._deferred_closes) == 1
