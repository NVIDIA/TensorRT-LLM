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
"""Unit tests for AlltoAllWatchdog (WideEP fault tolerance, PR 1a.4)."""

import threading
import time
from collections.abc import Callable
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.alltoall_watchdog import (
    DEFAULT_ALLTOALL_WATCHDOG_POLL_INTERVAL_S,
    DEFAULT_ALLTOALL_WATCHDOG_TIMEOUT_S,
    UNKNOWN_COMPLETION_FLAG,
    AlltoAllWatchdog,
    AlltoAllWatchdogCoordinator,
    AlltoAllWatchdogTimeout,
    CompletionFlagReadTimeout,
)
from tensorrt_llm._torch.modules.fused_moe.ep_group_health import EPGroupHealth
from tensorrt_llm._torch.modules.fused_moe.wide_ep_ft import get_wide_ep_ft_options


class FakeCompletionFlagReader:
    """Thread-safe completion flag reader for pure-Python watchdog tests."""

    def __init__(self, ep_size: int) -> None:
        self._lock = threading.Lock()
        self._flags = {
            "dispatch": [0 for _ in range(ep_size)],
            "combine": [0 for _ in range(ep_size)],
        }

    def set_flags(self, phase: str, flags: list[int]) -> None:
        with self._lock:
            self._flags[phase] = list(flags)

    def read_completion_flags(self, phase: str) -> tuple[int, ...]:
        with self._lock:
            return tuple(self._flags[phase])


class TimeoutCompletionFlagReader:
    def read_completion_flags(self, phase: str) -> tuple[int, ...]:
        raise CompletionFlagReadTimeout("blocked")


class OneGoodReadThenTimeoutReader:
    def __init__(self, flags: tuple[int, ...]) -> None:
        self._flags = flags
        self._read_count = 0

    def read_completion_flags(self, phase: str) -> tuple[int, ...]:
        self._read_count += 1
        if self._read_count == 1:
            return self._flags
        raise CompletionFlagReadTimeout("blocked")


def _wait_for(predicate: Callable[[], bool], timeout_s: float = 1.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition was not reached before timeout")


def test_watchdog_completes_when_all_active_flags_arrive() -> None:
    health = EPGroupHealth(4)
    reader = FakeCompletionFlagReader(ep_size=4)
    events: list[AlltoAllWatchdogTimeout] = []

    with AlltoAllWatchdog(
        ep_size=4,
        ep_rank=0,
        completion_reader=reader,
        timeout_s=0.2,
        poll_interval_s=0.005,
        health=health,
        on_timeout=events.append,
    ) as watchdog:
        watchdog.watch(phase="dispatch", expected_flag=1)
        reader.set_flags("dispatch", [1, 1, 1, 1])
        assert watchdog.wait_until_idle(timeout_s=1.0)

    assert events == []
    assert health.all_active() is True


def test_watchdog_completes_when_flags_advance_past_expected_generation() -> None:
    health = EPGroupHealth(4)
    reader = FakeCompletionFlagReader(ep_size=4)
    reader.set_flags("dispatch", [2, 1, 5, 3])
    events: list[AlltoAllWatchdogTimeout] = []

    with AlltoAllWatchdog(
        ep_size=4,
        ep_rank=0,
        completion_reader=reader,
        timeout_s=0.02,
        poll_interval_s=0.005,
        health=health,
        on_timeout=events.append,
    ) as watchdog:
        watchdog.watch(phase="dispatch", expected_flag=1)
        assert watchdog.wait_until_idle(timeout_s=1.0)

    assert events == []
    assert health.all_active() is True


def test_watchdog_handles_signed_uint32_generation_boundaries() -> None:
    reader = FakeCompletionFlagReader(ep_size=2)
    events: list[AlltoAllWatchdogTimeout] = []

    with AlltoAllWatchdog(
        ep_size=2,
        ep_rank=0,
        completion_reader=reader,
        timeout_s=0.05,
        poll_interval_s=0.005,
        on_timeout=events.append,
    ) as watchdog:
        reader.set_flags("dispatch", [-(1 << 31), -(1 << 31)])
        watchdog.watch(phase="dispatch", expected_flag=1 << 31)
        assert watchdog.wait_until_idle(timeout_s=1.0)

        reader.set_flags("dispatch", [-1, -1])
        watchdog.watch(phase="dispatch", expected_flag=(1 << 32) - 1)
        assert watchdog.wait_until_idle(timeout_s=1.0)

        reader.set_flags("dispatch", [0, 0])
        watchdog.watch(phase="dispatch", expected_flag=0)
        assert watchdog.wait_until_idle(timeout_s=1.0)

    assert events == []


def test_watchdog_defaults_match_design_doc() -> None:
    reader = FakeCompletionFlagReader(ep_size=1)
    watchdog = AlltoAllWatchdog(ep_size=1, ep_rank=0, completion_reader=reader)

    assert watchdog._timeout_s == DEFAULT_ALLTOALL_WATCHDOG_TIMEOUT_S
    assert watchdog._poll_interval_s == DEFAULT_ALLTOALL_WATCHDOG_POLL_INTERVAL_S


def test_watchdog_stop_is_terminal() -> None:
    reader = FakeCompletionFlagReader(ep_size=1)
    watchdog = AlltoAllWatchdog(
        ep_size=1,
        ep_rank=0,
        completion_reader=reader,
        timeout_s=0.2,
        poll_interval_s=0.005,
    )
    watchdog.start()
    watchdog.stop(timeout_s=1.0)

    with pytest.raises(RuntimeError, match="stopped AlltoAllWatchdog"):
        watchdog.start()
    with pytest.raises(RuntimeError, match="stopped AlltoAllWatchdog"):
        watchdog.watch(phase="dispatch", expected_flag=1)


def test_wide_ep_ft_options_create_shared_health_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TLLM_FAULT_TOLERANCE_MODE", "1")
    model_config = SimpleNamespace(
        extra_attrs={},
        mapping=SimpleNamespace(moe_ep_size=4),
    )

    health, timeout_s, poll_interval_s = get_wide_ep_ft_options(model_config)
    health_again, timeout_again_s, poll_again_s = get_wide_ep_ft_options(model_config)

    assert isinstance(health, EPGroupHealth)
    assert health_again is health
    assert timeout_s == DEFAULT_ALLTOALL_WATCHDOG_TIMEOUT_S
    assert timeout_again_s == timeout_s
    assert poll_interval_s == DEFAULT_ALLTOALL_WATCHDOG_POLL_INTERVAL_S
    assert poll_again_s == poll_interval_s


def test_wide_ep_ft_options_ignore_legacy_enable_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TLLM_FAULT_TOLERANCE_MODE", raising=False)
    monkeypatch.setenv("TRTLLM_ENABLE_WIDE_EP_FT", "1")
    model_config = SimpleNamespace(
        extra_attrs={},
        mapping=SimpleNamespace(moe_ep_size=4),
    )

    health, timeout_s, _ = get_wide_ep_ft_options(model_config)

    assert health is None
    assert timeout_s is None


def test_watchdog_coordinator_reuses_committed_mask_when_generation_is_unchanged() -> None:
    health = EPGroupHealth(4)
    coordinator = AlltoAllWatchdogCoordinator(
        workspace_state={},
        workspace=torch.zeros((4, 1), dtype=torch.uint8),
        metainfo=torch.zeros((1,), dtype=torch.int64),
        metainfo_index={},
        ep_rank=0,
        health=health,
    )
    dispatch_snapshot = coordinator.capture_active_rank_mask(None)
    assert dispatch_snapshot.active_rank_mask is not None
    assert dispatch_snapshot.committed_generation == 0

    combine_mask = coordinator.active_rank_mask_for_combine(dispatch_snapshot, None)

    assert combine_mask is dispatch_snapshot.active_rank_mask
    assert combine_mask.tolist() == [0b1111, 0]


def test_watchdog_coordinator_fails_closed_on_committed_generation_change() -> None:
    health = EPGroupHealth(4)
    coordinator = AlltoAllWatchdogCoordinator(
        workspace_state={},
        workspace=torch.zeros((4, 1), dtype=torch.uint8),
        metainfo=torch.zeros((1,), dtype=torch.int64),
        metainfo_index={},
        ep_rank=0,
        health=health,
    )
    dispatch_snapshot = coordinator.capture_active_rank_mask(None)

    health.mark_failed(2)
    health.mark_active(2)
    assert health.get_mask() == 0b1111

    with pytest.raises(
        RuntimeError,
        match="committed EP membership changed between dispatch and combine",
    ):
        coordinator.active_rank_mask_for_combine(dispatch_snapshot, None)


def test_watchdog_coordinator_converts_atomic_snapshot_to_two_mask_words() -> None:
    health = EPGroupHealth(72)
    health.mark_failed(70)
    coordinator = AlltoAllWatchdogCoordinator(
        workspace_state={},
        workspace=torch.zeros((72, 1), dtype=torch.uint8),
        metainfo=torch.zeros((1,), dtype=torch.int64),
        metainfo_index={},
        ep_rank=0,
        health=health,
    )

    dispatch_snapshot = coordinator.capture_active_rank_mask(None)

    assert dispatch_snapshot.committed_generation == 1
    assert dispatch_snapshot.active_rank_mask is not None
    assert dispatch_snapshot.active_rank_mask.tolist() == [(1 << 64) - 1, 0xBF]


def test_watchdog_coordinator_explicit_mask_is_not_bound_to_health_generation() -> None:
    health = EPGroupHealth(4)
    coordinator = AlltoAllWatchdogCoordinator(
        workspace_state={},
        workspace=torch.zeros((4, 1), dtype=torch.uint8),
        metainfo=torch.zeros((1,), dtype=torch.int64),
        metainfo_index={},
        ep_rank=0,
        health=health,
    )
    explicit_mask = torch.tensor([0b1101, 0], dtype=torch.uint64)
    dispatch_snapshot = coordinator.capture_active_rank_mask(explicit_mask)
    assert dispatch_snapshot.committed_generation is None

    health.mark_failed(2)
    combine_mask = coordinator.active_rank_mask_for_combine(dispatch_snapshot, None)

    assert combine_mask is dispatch_snapshot.active_rank_mask
    assert combine_mask.tolist() == explicit_mask.tolist()
    with pytest.raises(ValueError, match="mask captured at dispatch"):
        coordinator.active_rank_mask_for_combine(
            dispatch_snapshot,
            torch.tensor(health.get_mask_words(), dtype=torch.uint64),
        )


def test_watchdog_no_detected_failure_publication_to_committed_health() -> None:
    health = EPGroupHealth(4)
    committed_before = health.snapshot()
    reader = FakeCompletionFlagReader(ep_size=4)
    reader.set_flags("dispatch", [1, 0, 1, 0])
    events: list[AlltoAllWatchdogTimeout] = []

    with AlltoAllWatchdog(
        ep_size=4,
        ep_rank=0,
        completion_reader=reader,
        timeout_s=0.02,
        poll_interval_s=0.005,
        health=health,
        on_timeout=events.append,
    ) as watchdog:
        watchdog.watch(phase="dispatch", expected_flag=1)
        _wait_for(lambda: len(events) == 1)
        assert watchdog.wait_until_idle(timeout_s=1.0)

    event = events[0]
    assert event.phase == "dispatch"
    assert event.expected_flag == 1
    assert event.observed_flags == (1, 0, 1, 0)
    assert event.missing_ranks == (1, 3)
    assert not hasattr(event, "marked_failed_ranks")
    assert health.snapshot() == committed_before


def test_watchdog_ignores_ranks_already_failed_in_health_mask() -> None:
    health = EPGroupHealth(4)
    assert health.mark_failed(2) is True
    reader = FakeCompletionFlagReader(ep_size=4)
    reader.set_flags("dispatch", [1, 1, 0, 1])
    events: list[AlltoAllWatchdogTimeout] = []

    with AlltoAllWatchdog(
        ep_size=4,
        ep_rank=0,
        completion_reader=reader,
        timeout_s=0.05,
        poll_interval_s=0.005,
        health=health,
        on_timeout=events.append,
    ) as watchdog:
        watchdog.watch(phase="dispatch", expected_flag=1)
        assert watchdog.wait_until_idle(timeout_s=1.0)

    assert events == []
    assert health.get_failed_ranks() == frozenset({2})


def test_watchdog_reports_local_missing_without_changing_committed_health() -> None:
    health = EPGroupHealth(4)
    reader = FakeCompletionFlagReader(ep_size=4)
    reader.set_flags("combine", [0, 2, 2, 2])
    events: list[AlltoAllWatchdogTimeout] = []

    with AlltoAllWatchdog(
        ep_size=4,
        ep_rank=0,
        completion_reader=reader,
        timeout_s=0.02,
        poll_interval_s=0.005,
        health=health,
        on_timeout=events.append,
    ) as watchdog:
        watchdog.watch(phase="combine", expected_flag=2)
        _wait_for(lambda: len(events) == 1)

    event = events[0]
    assert event.missing_ranks == (0,)
    assert health.get_failed_ranks() == frozenset()


def test_watchdog_poll_timeout_without_snapshot_fails_closed() -> None:
    health = EPGroupHealth(3)
    events: list[AlltoAllWatchdogTimeout] = []

    with AlltoAllWatchdog(
        ep_size=3,
        ep_rank=0,
        completion_reader=TimeoutCompletionFlagReader(),
        timeout_s=0.02,
        poll_interval_s=0.005,
        health=health,
        on_timeout=events.append,
    ) as watchdog:
        watchdog.watch(phase="dispatch", expected_flag=1)
        _wait_for(lambda: len(events) == 1)

    event = events[0]
    assert event.poll_timed_out is True
    assert event.observed_flags == (UNKNOWN_COMPLETION_FLAG,) * 3
    assert event.missing_ranks == (0, 1, 2)
    assert health.all_active() is True


def test_watchdog_poll_timeout_with_prior_snapshot_does_not_mark_failed_rank() -> None:
    health = EPGroupHealth(3)
    events: list[AlltoAllWatchdogTimeout] = []

    with AlltoAllWatchdog(
        ep_size=3,
        ep_rank=0,
        completion_reader=OneGoodReadThenTimeoutReader((1, 0, 1)),
        timeout_s=0.02,
        poll_interval_s=0.005,
        health=health,
        on_timeout=events.append,
    ) as watchdog:
        watchdog.watch(phase="dispatch", expected_flag=1)
        _wait_for(lambda: len(events) == 1)

    event = events[0]
    assert event.poll_timed_out is True
    assert event.observed_flags == (1, 0, 1)
    assert event.missing_ranks == (1,)
    assert health.all_active() is True


def test_watchdog_callback_error_stops_and_clears_queue() -> None:
    health = EPGroupHealth(2)
    reader = FakeCompletionFlagReader(ep_size=2)
    reader.set_flags("dispatch", [1, 0])

    def raise_from_callback(event: AlltoAllWatchdogTimeout) -> None:
        raise RuntimeError(f"callback failed for {event.phase}")

    with AlltoAllWatchdog(
        ep_size=2,
        ep_rank=0,
        completion_reader=reader,
        timeout_s=0.02,
        poll_interval_s=0.005,
        health=health,
        on_timeout=raise_from_callback,
    ) as watchdog:
        watchdog.watch(phase="dispatch", expected_flag=1)
        _wait_for(lambda: watchdog.last_error is not None)
        assert watchdog.wait_until_idle(timeout_s=1.0)
        assert isinstance(watchdog.last_error, RuntimeError)
        with pytest.raises(RuntimeError, match="stopped AlltoAllWatchdog"):
            watchdog.watch(phase="dispatch", expected_flag=2)


def test_watchdog_preserves_fifo_order_and_clears_followups_after_timeout() -> None:
    health = EPGroupHealth(3)
    reader = FakeCompletionFlagReader(ep_size=3)
    reader.set_flags("dispatch", [1, 0, 1])
    reader.set_flags("combine", [0, 0, 0])
    events: list[AlltoAllWatchdogTimeout] = []

    with AlltoAllWatchdog(
        ep_size=3,
        ep_rank=0,
        completion_reader=reader,
        timeout_s=0.02,
        poll_interval_s=0.005,
        health=health,
        on_timeout=events.append,
    ) as watchdog:
        watchdog.watch(phase="dispatch", expected_flag=1)
        watchdog.watch(phase="combine", expected_flag=2)
        _wait_for(lambda: len(events) == 1)
        assert watchdog.wait_until_idle(timeout_s=1.0)
        time.sleep(0.05)

    assert len(events) == 1
    assert events[0].phase == "dispatch"
    assert events[0].missing_ranks == (1,)
    assert health.get_failed_ranks() == frozenset()


def test_watchdog_from_workspace_reads_phase_specific_offsets() -> None:
    ep_size = 3
    ep_rank = 1
    workspace = torch.zeros((ep_size, 64), dtype=torch.uint8)
    metainfo = torch.zeros((10,), dtype=torch.int64)
    metainfo_index = {
        "DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX": 4,
        "COMBINE_COMPLETION_FLAGS_OFFSET_INDEX": 5,
    }
    metainfo[4] = 4
    metainfo[5] = 20
    workspace[ep_rank, 4:16].view(torch.int32).copy_(torch.tensor([7, 7, 7], dtype=torch.int32))
    workspace[ep_rank, 20:32].view(torch.int32).copy_(torch.tensor([0, 8, 8], dtype=torch.int32))
    health = EPGroupHealth(ep_size)
    events: list[AlltoAllWatchdogTimeout] = []

    with AlltoAllWatchdog.from_workspace(
        workspace=workspace,
        metainfo=metainfo,
        metainfo_index=metainfo_index,
        ep_rank=ep_rank,
        ep_size=ep_size,
        timeout_s=0.02,
        poll_interval_s=0.005,
        health=health,
        on_timeout=events.append,
    ) as watchdog:
        watchdog.watch(phase="dispatch", expected_flag=7)
        assert watchdog.wait_until_idle(timeout_s=1.0)

        watchdog.watch(phase="combine", expected_flag=8)
        _wait_for(lambda: len(events) == 1)

    assert events[0].phase == "combine"
    assert events[0].missing_ranks == (0,)
    assert health.get_failed_ranks() == frozenset()


def test_workspace_coordinators_share_fifo_watchdog() -> None:
    ep_size = 3
    ep_rank = 0
    workspace_state: dict[str, object] = {}
    workspace = torch.zeros((ep_size, 64), dtype=torch.uint8)
    metainfo = torch.tensor([0, 4, 16], dtype=torch.int64)
    metainfo_index = {
        "FLAG_VAL_OFFSET_INDEX": 0,
        "DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX": 1,
        "COMBINE_COMPLETION_FLAGS_OFFSET_INDEX": 2,
    }
    workspace[ep_rank, 4:16].view(torch.int32).copy_(torch.tensor([1, 0, 1]))
    health = EPGroupHealth(ep_size)
    events: list[AlltoAllWatchdogTimeout] = []
    on_timeout = events.append
    coordinators = [
        AlltoAllWatchdogCoordinator(
            workspace_state=workspace_state,
            workspace=workspace,
            metainfo=metainfo,
            metainfo_index=metainfo_index,
            ep_rank=ep_rank,
            health=health,
        )
        for _ in range(2)
    ]
    watchdogs = [
        coordinator.acquire_watchdog(
            ep_size=ep_size,
            timeout_s=0.02,
            poll_interval_s=0.005,
            on_timeout=on_timeout,
        )
        for coordinator in coordinators
    ]

    try:
        assert watchdogs[0] is watchdogs[1]
        coordinators[0].watch_collective(watchdogs[0], "dispatch", None)
        coordinators[1].watch_collective(watchdogs[1], "combine", None)
        _wait_for(lambda: len(events) == 1)
        assert watchdogs[0].wait_until_idle(timeout_s=1.0)
        time.sleep(0.05)

        assert len(events) == 1
        assert events[0].phase == "dispatch"
        assert events[0].missing_ranks == (1,)
        assert health.get_failed_ranks() == frozenset()
    finally:
        for coordinator, watchdog in zip(coordinators, watchdogs):
            coordinator.release_watchdog(watchdog)


def test_workspace_coordinator_wraps_shared_generation() -> None:
    workspace_state: dict[str, object] = {}
    workspace = torch.zeros((1, 32), dtype=torch.uint8)
    workspace[0, 0:4].view(torch.int32).fill_(-1)
    metainfo = torch.tensor([0, 4, 8], dtype=torch.int64)
    metainfo_index = {
        "FLAG_VAL_OFFSET_INDEX": 0,
        "DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX": 1,
        "COMBINE_COMPLETION_FLAGS_OFFSET_INDEX": 2,
    }
    coordinator = AlltoAllWatchdogCoordinator(
        workspace_state=workspace_state,
        workspace=workspace,
        metainfo=metainfo,
        metainfo_index=metainfo_index,
        ep_rank=0,
    )
    watchdog = coordinator.acquire_watchdog(
        ep_size=1,
        timeout_s=0.05,
        poll_interval_s=0.005,
    )

    try:
        coordinator.watch_collective(watchdog, "dispatch", None)
        assert watchdog.wait_until_idle(timeout_s=1.0)
    finally:
        coordinator.release_watchdog(watchdog)


def test_unmonitored_coordinator_advances_shared_generation() -> None:
    workspace_state: dict[str, object] = {}
    workspace = torch.zeros((2, 32), dtype=torch.uint8)
    metainfo = torch.tensor([0, 4, 12], dtype=torch.int64)
    metainfo_index = {
        "FLAG_VAL_OFFSET_INDEX": 0,
        "DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX": 1,
        "COMBINE_COMPLETION_FLAGS_OFFSET_INDEX": 2,
    }
    coordinators = [
        AlltoAllWatchdogCoordinator(
            workspace_state=workspace_state,
            workspace=workspace,
            metainfo=metainfo,
            metainfo_index=metainfo_index,
            ep_rank=0,
        )
        for _ in range(2)
    ]
    events: list[AlltoAllWatchdogTimeout] = []
    watchdog = coordinators[0].acquire_watchdog(
        ep_size=2,
        timeout_s=0.02,
        poll_interval_s=0.005,
        on_timeout=events.append,
    )

    try:
        coordinators[1].watch_collective(None, "dispatch", None)
        coordinators[0].watch_collective(watchdog, "dispatch", None)
        _wait_for(lambda: len(events) == 1)
        assert events[0].expected_flag == 2
    finally:
        coordinators[0].release_watchdog(watchdog)


def test_watchdog_rejects_active_mask_without_local_rank() -> None:
    reader = FakeCompletionFlagReader(ep_size=4)
    with AlltoAllWatchdog(
        ep_size=4,
        ep_rank=2,
        completion_reader=reader,
        timeout_s=0.1,
        poll_interval_s=0.005,
    ) as watchdog:
        with pytest.raises(ValueError, match="local ep_rank"):
            watchdog.watch(phase="dispatch", expected_flag=1, active_mask=0b1011)
