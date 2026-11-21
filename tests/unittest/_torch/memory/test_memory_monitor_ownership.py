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
"""Who owns the monitor's process-wide state: the allocator recording, the
KV pool readings, the registered KV caches and the startup plan.

CPU only: the allocator and the managers are faked.
"""

import pytest
import torch
from memory_monitor_test_utils import GIB, MB, FakeKvManager, capability, kv_stats

from tensorrt_llm._torch.memory.monitor import pools, providers, startup, tagging, triggers
from tensorrt_llm._torch.memory.monitor.capture import _run_checks
from tensorrt_llm._torch.memory.monitor.frame import DeviceReading, TorchReading, build_frame
from tensorrt_llm._torch.memory.monitor.providers.kv_cache import KvCacheProvider
from tensorrt_llm._torch.memory.monitor.report import render
from tensorrt_llm._torch.memory.monitor.schema import Basis, Category, Entry, Placement, Source

pytestmark = pytest.mark.cpu_only


@pytest.fixture(autouse=True)
def _full_mode(monkeypatch):
    monkeypatch.setenv("TLLM_MEM_MONITOR", "1")
    yield
    monkeypatch.setattr(tagging, "_recording", False)
    pools.clear_registered_pools()


def _fake_recorder(monkeypatch):
    """Fake allocator history: records what the monitor asks of it."""
    calls = []
    memory = torch.cuda.memory
    monkeypatch.setattr(tagging, "tagging_supported", lambda: True)
    monkeypatch.setattr(tagging, "history_recording_active", lambda: False)
    monkeypatch.setattr(
        memory, "_record_memory_history", lambda *a, **k: calls.append(("record", a, k))
    )
    monkeypatch.setattr(memory, "_get_memory_metadata", lambda: "")
    monkeypatch.setattr(memory, "_set_memory_metadata", lambda name: calls.append(("tag", name)))
    return calls


def test_tag_writes_only_inside_our_window(monkeypatch):
    calls = _fake_recorder(monkeypatch)

    # Someone else's recording, or none: the label would land on their trace.
    with tagging.tag("sampler"):
        pass
    assert not [c for c in calls if c[0] == "tag"], "tagged outside the window"
    assert not tagging._windows, "a window outside the recording"

    assert tagging.start_tagging_window()
    with tagging.tag("sampler"):
        pass
    assert ("tag", "sampler") in calls
    assert [w[2] for w in tagging._windows] == ["sampler"]

    tagging.stop_tagging_window()
    calls.clear()
    with tagging.tag("sampler"):
        pass
    assert not [c for c in calls if c[0] == "tag"], "tagged after the window closed"


def test_live_blocks_read_only_the_current_device(monkeypatch):
    def segment(device, address):
        block = {"address": address, "size": MB, "state": "active_allocated"}
        return {"device": device, "segment_pool_id": (0, 0), "blocks": [block]}

    snapshot = {"segments": [segment(0, 0x1000), segment(1, 0x2000)]}
    monkeypatch.setattr(torch.cuda.memory, "_snapshot", lambda *a, **k: snapshot)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    assert [b[0] for b in tagging._live_blocks()] == [0x2000], "another device's blocks"


def _startup(monkeypatch, *, raises: type[BaseException] | None):
    """Run a startup window with the recorder, stacks and report faked."""
    events = []
    monkeypatch.setattr(tagging, "start_tagging_window", lambda: _start(monkeypatch, events))
    monkeypatch.setattr(tagging, "stop_tagging_window", lambda: _stop(monkeypatch, events))
    monkeypatch.setattr(tagging, "freeze_tag_map", lambda: tagging.TagSnapshot(supported=True))
    monkeypatch.setattr(tagging, "stacks_enabled", lambda: True)
    monkeypatch.setattr(
        tagging, "start_stack_recording", lambda: events.append("start stacks") or True
    )
    monkeypatch.setattr(tagging, "stop_stack_recording", lambda: events.append("stop stacks"))
    monkeypatch.setattr(tagging, "_log_window_result", lambda *a, **k: None)
    with pytest.raises(raises) if raises else _nothing():
        with tagging.startup_window():
            if raises:
                raise raises("startup failed")
    return events


def _start(monkeypatch, events):
    monkeypatch.setattr(tagging, "_recording", True)
    events.append("start window")
    return True


def _stop(monkeypatch, events):
    monkeypatch.setattr(tagging, "_recording", False)
    events.append("stop window")


class _nothing:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def test_stack_recording_follows_only_a_completed_startup(monkeypatch):
    providers.clear_registered_providers()
    try:
        for failure in (ValueError, KeyboardInterrupt):
            providers.clear_registered_providers()
            providers.begin_executor()
            events = _startup(monkeypatch, raises=failure)
            # Nobody is left to stop a recording a failed startup started.
            assert "start stacks" not in events, failure
            assert events[-1] == "stop window", failure
            assert not tagging.tagging_window_active()

        providers.clear_registered_providers()
        token = providers.begin_executor()
        events = _startup(monkeypatch, raises=None)
        assert events == ["start window", "start stacks"]
        assert not tagging.tagging_window_active(), "the tagging window ended"

        # The executor that started stack recording stops it as it ends, and
        # another executor's shutdown does not.
        events.clear()
        tagging.forget_startup_of(object())
        assert events == []
        tagging.forget_startup_of(token)
        assert events == ["stop stacks"]
    finally:
        providers.clear_registered_providers()
        tagging.forget_startup_of(tagging._startup_executor)


def test_pool_check_runs_on_the_reading_it_came_from():
    manager = FakeKvManager(kv_stats(100, 30, 20))
    provider = KvCacheProvider(manager)
    rows = {}
    providers.collect_all([provider], pool_rows=rows)
    assert list(rows) == ["kv_cache.main"]

    # Another report reads the pool while it is inconsistent ...
    manager._stats = [kv_stats(100, 70, 50)]
    other = {}
    providers.collect_all([provider], pool_rows=other)
    assert not _run_checks({}, 0, other)[-1].passed

    # ... which must not change what the first report checks.
    assert _run_checks({}, 0, rows)[-1].passed
    assert not hasattr(provider, "last_pool_group_rows"), "no reading kept on the provider"


def test_released_kv_cache_stops_being_reported(monkeypatch):
    from tensorrt_llm._torch.pyexecutor._util import KvCacheCreator
    from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType

    class _V1:
        def get_kv_cache_stats(self):
            return type("Stats", (), {"allocated_bytes": GIB})()

        def shutdown(self):
            pass

    providers.clear_registered_providers()
    try:
        sizing, other = _V1(), _V1()
        providers.begin_executor()
        providers.register_provider(KvCacheProvider(sizing, "main"))
        providers.register_provider(KvCacheProvider(other, "draft"))
        resources = {
            ResourceManagerType.KV_CACHE_MANAGER: sizing,
            ResourceManagerType.DRAFT_KV_CACHE_MANAGER: None,
        }
        # The sizing manager outlives its release (``sizing`` is still held
        # here), and V1 keeps reporting the bytes it allocated.
        KvCacheCreator.teardown_managers(None, resources)
        left = [p for p in providers.default_providers() if isinstance(p, KvCacheProvider)]
        assert [p._manager for p in left] == [other], "only the released pool is dropped"
    finally:
        providers.clear_registered_providers()


def test_sleep_release_lasts_as_long_as_the_monitored_executor(monkeypatch):
    """``release_with_tag`` releases every blob under the tag in the process;
    the monitor keeps that state until the executor it follows ends."""
    providers.clear_registered_providers()
    try:
        manager = FakeKvManager(kv_stats(10, 10, 0))
        monitored = providers.begin_executor()
        providers.register_provider(KvCacheProvider(manager))
        pools.register_pool((7, 1), "sleep: kv_cache", sleep_tag="kv_cache")
        other = providers.begin_executor()

        pools.mark_released(["kv_cache"])
        providers.end_executor(other)
        (provider,) = [p for p in providers.default_providers() if isinstance(p, KvCacheProvider)]
        physical = [e for e in provider.collect() if e.basis is Basis.PHYSICAL_BACKING]
        assert physical and all(e.released for e in physical), "the monitored pool is released"
        providers.end_executor(monitored)
        assert not pools.is_tag_released("kv_cache"), "a new executor starts awake"
    finally:
        providers.clear_registered_providers()
        pools.clear_registered_pools()


def test_another_executors_sleep_does_not_release_the_monitored_kv_cache():
    """The monitored executor's KV cache is not a sleep-mode blob without
    ``sleep_config``, so ``release_with_tag("kv_cache")`` from another
    executor in the process leaves it resident."""
    providers.clear_registered_providers()
    pools.clear_registered_pools()
    try:
        manager = FakeKvManager(kv_stats(10, 10, 0))  # the provider holds it weakly
        monitored = providers.begin_executor()
        providers.register_provider(KvCacheProvider(manager))
        other = providers.begin_executor()
        # Only the other executor runs with sleep_config.
        pools.register_pool((7, 2), "sleep: kv_cache", sleep_tag="kv_cache")
        pools.mark_released(["kv_cache"])

        assert not pools.is_tag_released("kv_cache")
        (provider,) = [p for p in providers.default_providers() if isinstance(p, KvCacheProvider)]
        physical = [e for e in provider.collect() if e.basis is Basis.PHYSICAL_BACKING]
        assert physical and not any(e.released for e in physical), "the pool is resident"
        providers.end_executor(other)
        providers.end_executor(monitored)
    finally:
        providers.clear_registered_providers()
        pools.clear_registered_pools()


def test_plan_compares_against_the_kv_pools_only():
    entries = [
        Entry(
            Category.KV_CACHE_POOL,
            10 * GIB,
            Source.COMPONENT,
            Placement.NON_TORCH,
            Basis.PHYSICAL_BACKING,
            owner="kv_cache.main",
        ),
        # Buffers the KV cache step allocated in PyTorch: in the KV cache
        # group, but already part of the plan's non-KV peak.
        Entry(
            Category.KV_CACHE_AUX,
            GIB,
            Source.TAG,
            Placement.TORCH_DEFAULT,
            Basis.LOGICAL,
            owner="kv_cache",
        ),
    ]
    frame = build_frame(
        DeviceReading(total=80 * GIB, free=63 * GIB, process_used=17 * GIB),
        TorchReading(reserved=7 * GIB, live=7 * GIB),
        entries,
    )
    startup.clear()
    startup.record_memory_plan(non_kv_peak=7 * GIB, activation=None)
    try:
        text = render(frame, capability())
    finally:
        startup.clear()
    (line,) = [line for line in text.splitlines() if "Startup plan" in line]
    assert "KV cache 10.00 GiB = 17.00 GiB" in line, line
    assert "(+0.00)" in line, line


def test_a_failure_not_about_memory_gets_no_advice():
    from tensorrt_llm._torch.memory.monitor.capture import MonitorCapture, render_capture

    pool = Entry(
        Category.KV_CACHE_POOL,
        10 * GIB,
        Source.COMPONENT,
        Placement.NON_TORCH,
        Basis.PHYSICAL_BACKING,
        owner="kv_cache.main",
    )
    frame = build_frame(
        DeviceReading(total=80 * GIB, free=68 * GIB, process_used=12 * GIB),
        TorchReading(reserved=GIB, live=GIB),
        [pool],
    )
    startup.clear()
    startup.set_stage_advice({"kv_cache": "KV cache"}, {"kv_cache": "reduce the fraction"})
    try:
        texts = {
            advise: render_capture(
                MonitorCapture(
                    frame=frame,
                    capability=capability(),
                    tag="startup_failed",
                    pid=1,
                    rank=0,
                    device_index=0,
                    headline="STARTUP FAILED while creating: KV cache.",
                    advise=advise,
                )
            )
            for advise in (True, False)
        }
    finally:
        startup.clear()
    assert "Suggestion: reduce the fraction." in texts[True], "the largest group's advice"
    assert "Suggestion" not in texts[False], "a missing checkpoint key is not a memory problem"


def test_a_failed_startup_is_not_reported_while_a_crash_kill_is_pending(monkeypatch):
    """The kill's grace period is for delivering the error, not for building a report."""
    from tensorrt_llm._torch.pyexecutor import hang_detector

    class _Armed:
        cancelled = False

        def is_alive(self):
            return True

    monkeypatch.setattr(tagging, "tagging_supported", lambda: False)
    reports = []
    monkeypatch.setattr(triggers, "report_now", lambda **kw: reports.append(kw["tag"]))

    def fail_startup():
        with pytest.raises(torch.OutOfMemoryError):
            with tagging.startup_window():
                raise torch.OutOfMemoryError("warmup")

    armed = _Armed()
    monkeypatch.setattr(hang_detector, "_armed_watchdogs", {armed})
    fail_startup()
    assert reports == [], "reported while the watchdog's grace period was running"

    armed.cancelled = True
    fail_startup()
    assert reports == ["startup_failed"], "reported once the kill is no longer pending"


def test_a_block_with_frames_never_takes_a_frozen_startup_label():
    """Stack mode: same-size reuse of a freed startup block is named by its own code."""
    from tensorrt_llm._torch.memory.monitor.tagging import TagSnapshot, live_attribution

    address, size = 1 << 32, 4 * MB
    frozen = TagSnapshot(supported=True, owners={address: "SAMPLER"}, sizes={address: size})
    frame = {"filename": "/work/my_app/cache.py", "line": 7, "name": "put"}

    def segments(frames):
        block = {"address": address, "size": size, "requested_size": size}
        block.update(state="active_allocated", frames=frames)
        return [{"segment_pool_id": (0, 0), "blocks": [block]}]

    totals, _ = live_attribution(frozen, segments([]))
    assert totals.get("SAMPLER") == size, "a startup block keeps its label"

    totals, sites = live_attribution(frozen, segments([frame]))
    assert "SAMPLER" not in totals, "allocated after the freeze, so not the startup buffer"
    assert totals["untagged"] == size
    assert sum(sites.values()) == size
