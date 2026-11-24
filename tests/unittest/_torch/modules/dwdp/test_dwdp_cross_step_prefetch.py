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
"""Cross-step DWDP prefetch (TRTLLM_DWDP_CROSS_STEP_PREFETCH=1) on CPU.

The issue order is checked on ``CyclicPrefetchSchedule`` alone, the event protocol on a weight
manager built without its constructor (mocked streams and events), and the routing on the
``DwdpManager`` facade.
"""

import contextlib
import random
from unittest.mock import MagicMock, call, patch

import pytest

from tensorrt_llm._torch.modules.dwdp import weight_manager as wm_mod
from tensorrt_llm._torch.modules.dwdp.page_pool import compute_slot_sizes
from tensorrt_llm._torch.modules.dwdp.prefetch_schedule import CyclicPrefetchSchedule
from tensorrt_llm._torch.modules.dwdp.weight_buffer import WeightBuffer
from tensorrt_llm._torch.modules.dwdp.weight_manager import (
    DWDPWeightManager,
    cross_step_prefetch_enabled,
    prefetch_slot_count,
)
from tensorrt_llm._torch.pyexecutor.dwdp import DwdpManager

pytestmark = pytest.mark.cpu_only

SHAPES = [(92, 2), (92, 3), (5, 2), (4, 3), (1, 2), (2, 3), (3, 3), (7, 4)]


def _run_pass(schedule, log, call_start=True, orders=None):
    if call_start:
        issued, abandoned = schedule.start_forward()
        assert not abandoned
        log.append(("start", None, issued))
    for order in range(schedule.num_layers) if orders is None else orders:
        issued, abandoned = schedule.bind(order)
        assert not abandoned
        log.append(("bind", order, issued))
        log.append(("consume", order, schedule.consume(order)))


# ----------------------------------------------------------------------------------------------
# CyclicPrefetchSchedule
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("num_layers,num_slots", SHAPES)
def test_each_layer_issued_once_per_pass_in_order(num_layers, num_slots):
    schedule = CyclicPrefetchSchedule(num_layers, num_slots)
    log = []
    for _ in range(4):
        _run_pass(schedule, log)
    issued = [order for _, _, orders in log for order in orders]
    # One cyclic sequence: every layer once per pass, plus the next pass's first layers in flight.
    assert issued == [i % num_layers for i in range(len(issued))]
    in_flight = min(num_slots, num_layers)
    assert len(issued) == 4 * num_layers + len(schedule.pending())
    assert schedule.pending() == list(range(len(schedule.pending())))
    assert 1 <= len(schedule.pending()) <= in_flight
    # Only the very first pass issues anything at its start.
    starts = [orders for kind, _, orders in log if kind == "start"]
    assert starts[0] == list(range(in_flight)) and all(not s for s in starts[1:])


def test_double_buffer_even_layer_count_wraps_two_ahead():
    """92 MoE layers, 2 slots: MoE(L-2) issues L, across the pass boundary too."""
    schedule = CyclicPrefetchSchedule(92, 2)
    log = []
    _run_pass(schedule, log)
    consumes = {order: issued for kind, order, issued in log if kind == "consume"}
    assert consumes[0] == [2] and consumes[89] == [91]
    assert consumes[90] == [0] and consumes[91] == [1]
    assert all(not issued for kind, _, issued in log if kind == "bind")


def test_odd_layer_count_issues_next_pass_at_last_layer():
    """The next pass's order 0 shares a slot with the last layer, so both wait for it."""
    schedule = CyclicPrefetchSchedule(5, 2)
    log = []
    _run_pass(schedule, log)
    consumes = {order: issued for kind, order, issued in log if kind == "consume"}
    assert consumes[2] == [4] and consumes[3] == []
    assert consumes[4] == [0, 1]


def test_three_slots_wrap_with_uneven_cycle():
    schedule = CyclicPrefetchSchedule(92, 3)
    log = []
    _run_pass(schedule, log)
    consumes = {order: issued for kind, order, issued in log if kind == "consume"}
    assert consumes[0] == [3] and consumes[88] == [91]
    assert consumes[89] == []  # order 0 of the next pass is blocked by order 90 (slot 0)
    assert consumes[90] == [0] and consumes[91] == [1, 2]


def test_pass_without_start_forward_issues_on_bind():
    """A warmup forward that never called start_forward still prefetches before binding."""
    schedule = CyclicPrefetchSchedule(4, 2)
    issued, abandoned = schedule.bind(0)
    assert issued == [0, 1] and not abandoned
    assert schedule.consume(0) == [2]


def test_unfinished_pass_restarts_and_reports_abandoned_read():
    schedule = CyclicPrefetchSchedule(6, 2)
    schedule.start_forward()
    for order in range(3):
        schedule.bind(order)
        schedule.consume(order)
    schedule.bind(3)  # the MoE of order 3 raised before finish_layer
    issued, abandoned = schedule.start_forward()
    assert abandoned and issued == [0, 1]
    assert schedule.num_restarts == 1
    log = []
    _run_pass(schedule, log, call_start=False)
    assert [o for _, _, orders in log for o in orders] == [2, 3, 4, 5, 0, 1]


def test_pass_stopped_between_layers_restarts_without_abandoned_read():
    schedule = CyclicPrefetchSchedule(6, 2)
    schedule.start_forward()
    schedule.bind(0)
    schedule.consume(0)
    issued, abandoned = schedule.start_forward()
    assert not abandoned and issued == [0, 1]
    assert schedule.num_restarts == 1


def test_out_of_order_bind_restarts_at_that_layer():
    schedule = CyclicPrefetchSchedule(6, 2)
    schedule.start_forward()
    issued, abandoned = schedule.bind(3)
    assert issued == [3, 4] and not abandoned
    assert schedule.num_restarts == 1
    assert schedule.consume(3) == [5]


def test_consume_without_bind_raises():
    schedule = CyclicPrefetchSchedule(4, 2)
    schedule.start_forward()
    with pytest.raises(RuntimeError):
        schedule.consume(0)
    schedule.bind(0)
    with pytest.raises(RuntimeError):
        schedule.consume(1)


def test_invalidate_reissues_first_layers():
    schedule = CyclicPrefetchSchedule(4, 2)
    _run_pass(schedule, [])
    assert schedule.pending() == [0, 1]
    assert schedule.invalidate() is False
    assert schedule.pending() == []
    issued, abandoned = schedule.start_forward()
    assert issued == [0, 1] and not abandoned
    assert schedule.num_restarts == 0


@pytest.mark.parametrize("num_layers,num_slots", SHAPES)
@pytest.mark.parametrize("seed", range(4))
def test_random_passes_keep_slot_contents_consistent(num_layers, num_slots, seed):
    """Model each slot's last copy and live reader; check RAW at every bind and WAR at every issue.

    Passes may skip start_forward, stop early, stop after a bind (an exception inside the MoE),
    skip a layer, or be followed by an invalidation.
    """
    rng = random.Random(seed)
    schedule = CyclicPrefetchSchedule(num_layers, num_slots)
    content = [None] * num_slots  # MoE order of the last copy enqueued into the slot
    reader = [None] * num_slots  # order bound to the slot whose consume is not recorded yet

    def apply(issued, abandoned):
        if abandoned:  # the caller synchronizes the device: every earlier read has finished
            reader[:] = [None] * num_slots
        for order in issued:
            slot = schedule.slot(order)
            assert reader[slot] is None, f"WAR: order {order} overwrites a slot still being read"
            content[slot] = order

    for _ in range(60):
        if rng.random() < 0.8:
            apply(*schedule.start_forward())
        orders = list(range(num_layers))
        kind = rng.random()
        if kind < 0.1 and num_layers > 1:
            orders.pop(rng.randrange(num_layers))
        stop = rng.randrange(num_layers) if kind > 0.85 else None
        for idx, order in enumerate(orders):
            apply(*schedule.bind(order))
            slot = schedule.slot(order)
            assert content[slot] == order, f"RAW: order {order} reads a copy of {content[slot]}"
            reader[slot] = order
            if stop == idx and rng.random() < 0.5:
                break  # exception between bind and consume
            reader[slot] = None
            apply(schedule.consume(order), False)
            if stop == idx:
                break
        if rng.random() < 0.1:
            apply([], schedule.invalidate())
            assert schedule.pending() == []


# ----------------------------------------------------------------------------------------------
# DWDPWeightManager event protocol (mocked streams and events)
# ----------------------------------------------------------------------------------------------

MOE_LAYERS = [1, 2, 3, 4]


def _make_manager(num_slots=2, layers=MOE_LAYERS):
    trace = MagicMock(name="trace")
    wm = object.__new__(DWDPWeightManager)
    buf = MagicMock()
    buf.device_id = 0
    buf.buffer_index_for_layer.side_effect = lambda layer: layers.index(layer) % num_slots
    buf.get_full_tensor.side_effect = lambda layer, name: f"full{layer}"
    wm._weight_buffer = buf
    wm._moe_layer_indices = list(layers)
    wm._moe_layer_set = set(layers)
    wm._weight_names = ["w"]
    wm._num_slots = num_slots
    wm._layer_order = {layer: order for order, layer in enumerate(layers)}
    wm._schedule = CyclicPrefetchSchedule(len(layers), num_slots)
    wm._schedule_restarts = 0
    wm._use_gather = False
    wm._use_batched_prefetch = False
    wm._copy_stream = getattr(trace, "copy_stream")
    wm._prefetch_events = [getattr(trace, f"prefetch{i}") for i in range(num_slots)]
    wm._consume_events = [getattr(trace, f"consume{i}") for i in range(num_slots)]
    return wm, trace


@contextlib.contextmanager
def _patched(trace):
    with (
        patch.object(wm_mod.torch.cuda, "current_stream", return_value=trace.compute_stream),
        patch.object(wm_mod.torch.cuda, "stream", return_value=contextlib.nullcontext()),
        patch.object(wm_mod.torch.cuda, "synchronize", side_effect=trace.device_synchronize),
        patch.object(
            DWDPWeightManager,
            "_prefetch_layer_per_slice",
            autospec=True,
            side_effect=lambda self, layer: trace.copy(layer),
        ),
    ):
        yield


def _copies(trace):
    return [c.args[0] for c in trace.mock_calls if c[0] == "copy"]


def test_manager_prefetch_waits_consume_then_records_prefetch():
    wm, trace = _make_manager()
    with _patched(trace):
        wm.start_forward()
    assert trace.mock_calls == [
        call.copy_stream.wait_event(trace.consume0),
        call.copy(1),
        call.prefetch0.record(trace.copy_stream),
        call.copy_stream.wait_event(trace.consume1),
        call.copy(2),
        call.prefetch1.record(trace.copy_stream),
    ]


def test_manager_full_passes_gather_each_layer_once_and_prefetch_next_pass():
    wm, trace = _make_manager()
    backend = MagicMock()
    with _patched(trace):
        wm.start_forward()
        for _ in range(3):
            wm.start_forward()
            for layer in MOE_LAYERS:
                wm.wait_and_bind_scheduled(backend, layer)
                assert backend.w.data == f"full{layer}"
                wm.finish_layer(layer)
    assert _copies(trace) == [1, 2] + [3, 4, 1, 2] * 3

    trace.reset_mock()
    with _patched(trace):
        wm.wait_and_bind_scheduled(backend, 3)  # pretend the next pass skipped layers 1-2
    # Out of sequence: restart at layer 3 instead of reading whatever the slot holds.
    assert _copies(trace) == [3, 4]


def test_manager_finish_records_consume_before_issuing_next():
    wm, trace = _make_manager()
    backend = MagicMock()
    with _patched(trace):
        wm.start_forward()
        wm.wait_and_bind_scheduled(backend, 1)
        trace.reset_mock()
        wm.finish_layer(1)
    assert trace.mock_calls == [
        call.consume0.record(trace.compute_stream),
        call.copy_stream.wait_event(trace.consume0),
        call.copy(3),
        call.prefetch0.record(trace.copy_stream),
    ]


def test_manager_bind_waits_for_prefetch_of_its_slot():
    wm, trace = _make_manager()
    backend = MagicMock()
    with _patched(trace):
        wm.start_forward()
        trace.reset_mock()
        wm.wait_and_bind_scheduled(backend, 2)  # out of sequence on purpose: restarts at 2
        assert call.compute_stream.wait_event(trace.prefetch1) in trace.mock_calls
        trace.reset_mock()
        wm.finish_layer(2)
        wm.wait_and_bind_scheduled(backend, 3)
    assert trace.mock_calls[-1] == call.compute_stream.wait_event(trace.prefetch0)


def test_manager_last_layers_prefetch_next_pass_and_start_issues_nothing():
    wm, trace = _make_manager()
    backend = MagicMock()
    with _patched(trace):
        wm.start_forward()
        for layer in MOE_LAYERS:
            wm.wait_and_bind_scheduled(backend, layer)
            trace.reset_mock()
            wm.finish_layer(layer)
            if layer == 3:
                assert _copies(trace) == [1]
            if layer == 4:
                assert _copies(trace) == [2]
        trace.reset_mock()
        wm.start_forward()
    assert trace.mock_calls == []


def test_manager_unfinished_pass_synchronizes_before_reusing_slots():
    wm, trace = _make_manager()
    backend = MagicMock()
    with _patched(trace):
        wm.start_forward()
        wm.wait_and_bind_scheduled(backend, 1)
        trace.reset_mock()
        wm.start_forward()  # the previous pass raised inside MoE(1)
    names = [c[0] for c in trace.mock_calls]
    assert names[0] == "device_synchronize"
    assert _copies(trace) == [1, 2]


def test_manager_invalidate_drops_in_flight_prefetch():
    wm, trace = _make_manager()
    backend = MagicMock()
    with _patched(trace):
        wm.start_forward()
        for layer in MOE_LAYERS:
            wm.wait_and_bind_scheduled(backend, layer)
            wm.finish_layer(layer)
        trace.reset_mock()
        wm.invalidate_prefetch()
        assert trace.mock_calls == []
        wm.start_forward()
    assert _copies(trace) == [1, 2]


def test_manager_three_slots():
    wm, trace = _make_manager(num_slots=3, layers=[1, 2, 3, 4, 5])
    backend = MagicMock()
    with _patched(trace):
        wm.start_forward()
        assert _copies(trace) == [1, 2, 3]
        trace.reset_mock()
        wm.wait_and_bind_scheduled(backend, 1)
        wm.finish_layer(1)
    assert trace.mock_calls[-3:] == [
        call.copy_stream.wait_event(trace.consume0),
        call.copy(4),
        call.prefetch0.record(trace.copy_stream),
    ]


def test_manager_scheduled_entry_points_require_schedule():
    wm, _ = _make_manager()
    wm._schedule = None
    with pytest.raises(RuntimeError):
        wm.start_forward()
    with pytest.raises(RuntimeError):
        wm.finish_layer(1)
    wm.invalidate_prefetch()  # no-op


def test_manager_rejects_non_moe_layer():
    wm, trace = _make_manager()
    with _patched(trace), pytest.raises(KeyError):
        wm.wait_and_bind_scheduled(MagicMock(), 7)


# ----------------------------------------------------------------------------------------------
# Env switches, slot sizing
# ----------------------------------------------------------------------------------------------


def test_env_switches(monkeypatch):
    monkeypatch.delenv("TRTLLM_DWDP_CROSS_STEP_PREFETCH", raising=False)
    monkeypatch.delenv("TRTLLM_DWDP_PREFETCH_SLOTS", raising=False)
    assert not cross_step_prefetch_enabled()
    assert prefetch_slot_count(False) == 2 and prefetch_slot_count(True) == 2
    monkeypatch.setenv("TRTLLM_DWDP_CROSS_STEP_PREFETCH", "1")
    assert cross_step_prefetch_enabled()
    monkeypatch.setenv("TRTLLM_DWDP_PREFETCH_SLOTS", "3")
    assert prefetch_slot_count(True) == 3
    assert prefetch_slot_count(False) == 2  # the per-step protocol is double-buffered
    for bad in ("1", "x", "-2"):
        monkeypatch.setenv("TRTLLM_DWDP_PREFETCH_SLOTS", bad)
        assert prefetch_slot_count(True) == 2


def test_compute_slot_sizes_three_slots():
    def layout(pre, post):
        return MagicMock(pre_size=pre, post_size=post)

    layouts = {
        1: {"a": layout(1, 2), "b": layout(3, 0)},  # 6 bytes
        2: {"a": layout(10, 0)},
        3: {"a": layout(0, 7)},
        4: {"a": layout(4, 4)},
    }
    assignments = {layer: (layer - 1) % 3 for layer in layouts}
    assert compute_slot_sizes(layouts, assignments, num_slots=3) == [8, 10, 7]
    assert compute_slot_sizes(layouts, {k: (k - 1) % 2 for k in layouts}) == [7, 10]


def test_weight_buffer_slot_of_layer():
    specs = {layer: {} for layer in (3, 5, 6, 9, 11)}
    buffer = WeightBuffer(specs, MagicMock(), 0, 1, 2, 0, granularity=1 << 21, num_slots=3)
    assert buffer.num_slots == 3
    assert [buffer.buffer_index_for_layer(layer) for layer in (3, 5, 6, 9, 11)] == [0, 1, 2, 0, 1]
    with pytest.raises(ValueError):
        WeightBuffer(specs, MagicMock(), 0, 1, 2, 0, granularity=1 << 21, num_slots=1)


# ----------------------------------------------------------------------------------------------
# DwdpManager facade
# ----------------------------------------------------------------------------------------------


def _facade(cross_step: bool, num_slots: int = 2):
    mgr = object.__new__(DwdpManager)
    mgr._weight_manager = MagicMock()
    mgr._weight_manager.num_slots = num_slots
    mgr._cross_step_prefetch = cross_step
    return mgr


@pytest.fixture
def not_capturing():
    with patch(
        "tensorrt_llm._torch.pyexecutor.dwdp.torch.cuda.is_current_stream_capturing",
        return_value=False,
    ) as p:
        yield p


def test_facade_routes_to_schedule(not_capturing):
    mgr = _facade(cross_step=True)
    wm = mgr._weight_manager
    backend = MagicMock()
    mgr.prefetch_first_layers()
    mgr.wait_and_bind(backend, 5)
    mgr.record_compute_and_prefetch_next(5)
    mgr.invalidate_prefetch()
    assert wm.mock_calls == [
        call.start_forward(),
        call.wait_and_bind_scheduled(backend, 5),
        call.finish_layer(5),
        call.invalidate_prefetch(),
    ]


def test_facade_default_keeps_per_step_protocol():
    mgr = _facade(cross_step=False)
    wm = mgr._weight_manager
    wm.first_moe_layer.return_value = 1
    wm.next_moe_layer.side_effect = lambda layer: layer + 1
    backend = MagicMock()
    mgr.prefetch_first_layers()
    mgr.wait_and_bind(backend, 1)
    mgr.record_compute_and_prefetch_next(1)
    mgr.invalidate_prefetch()
    # Per-step protocol: the first two layers up front, then each MoE prefetches two ahead.
    assert wm.prefetch_layer.call_args_list == [call(1), call(2), call(3)]
    wm.wait_and_bind.assert_called_once_with(backend, 1)
    wm.record_compute.assert_called_once_with(1)
    for name in ("start_forward", "wait_and_bind_scheduled", "finish_layer", "invalidate_prefetch"):
        getattr(wm, name).assert_not_called()


def test_facade_class_default_is_per_step():
    mgr = object.__new__(DwdpManager)
    assert mgr._cross_step_prefetch is False


def test_facade_capture_falls_back_to_per_step():
    mgr = _facade(cross_step=True)
    wm = mgr._weight_manager
    backend = MagicMock()
    with patch(
        "tensorrt_llm._torch.pyexecutor.dwdp.torch.cuda.is_current_stream_capturing",
        return_value=True,
    ):
        mgr.wait_and_bind(backend, 1)
    assert mgr._cross_step_prefetch is False
    wm.wait_and_bind.assert_called_once_with(backend, 1)
    wm.wait_and_bind_scheduled.assert_not_called()
    wm.next_moe_layer.return_value = 2
    mgr.record_compute_and_prefetch_next(1)
    wm.record_compute.assert_called_once_with(1)
    wm.finish_layer.assert_not_called()


def test_facade_capture_with_three_slots_raises():
    mgr = _facade(cross_step=True, num_slots=3)
    with (
        patch(
            "tensorrt_llm._torch.pyexecutor.dwdp.torch.cuda.is_current_stream_capturing",
            return_value=True,
        ),
        pytest.raises(RuntimeError),
    ):
        mgr.wait_and_bind(MagicMock(), 1)


def test_facade_cleanup_releases_with_prefetch_in_flight():
    """The last pass may leave the next pass's prefetches in flight; ``release`` synchronizes the
    device before unmapping the slots they write."""
    mgr = _facade(cross_step=True)
    wm = mgr._weight_manager
    mgr._model_ref = None
    mgr.dwdp_group = MagicMock()
    mgr.cleanup()
    wm.release.assert_called_once_with()
    assert mgr._weight_manager is None


@pytest.mark.parametrize("env,expected", [(None, False), ("0", False), ("1", True)])
def test_facade_setup_reads_switch(monkeypatch, env, expected):
    if env is None:
        monkeypatch.delenv("TRTLLM_DWDP_CROSS_STEP_PREFETCH", raising=False)
    else:
        monkeypatch.setenv("TRTLLM_DWDP_CROSS_STEP_PREFETCH", env)
    mgr = object.__new__(DwdpManager)
    mgr._store_rendezvous = False
    mgr._registered_layers = [3, 1]
    mgr._mapping = MagicMock()
    mgr.dwdp_group = MagicMock()
    mgr.num_experts_per_worker = 4
    mgr.num_prefetch_experts = 4
    mgr.config = MagicMock(contention_opt=False)
    fake_wm = MagicMock(cross_step_prefetch=expected)
    with (
        patch("tensorrt_llm._torch.pyexecutor.dwdp.setup_dwdp", return_value=fake_wm) as setup,
        patch("tensorrt_llm._torch.pyexecutor.dwdp.torch.cuda.current_device", return_value=0),
    ):
        assert mgr.setup(MagicMock()) is fake_wm
    assert setup.call_args.kwargs["cross_step_prefetch"] is expected
    assert setup.call_args.kwargs["layer_indices"] == [1, 3]
    assert mgr._cross_step_prefetch is expected
