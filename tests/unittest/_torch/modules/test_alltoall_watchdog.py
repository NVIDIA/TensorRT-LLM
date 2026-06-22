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
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.alltoall_watchdog import (
    DEFAULT_ALLTOALL_WATCHDOG_POLL_INTERVAL_S,
    DEFAULT_ALLTOALL_WATCHDOG_TIMEOUT_S,
    UNKNOWN_COMPLETION_FLAG,
    AlltoAllWatchdog,
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


def _wait_for(predicate, timeout_s: float = 1.0) -> None:
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


def test_watchdog_defaults_match_design_doc() -> None:
    reader = FakeCompletionFlagReader(ep_size=1)
    watchdog = AlltoAllWatchdog(ep_size=1, ep_rank=0, completion_reader=reader)

    assert watchdog._timeout_s == DEFAULT_ALLTOALL_WATCHDOG_TIMEOUT_S
    assert watchdog._poll_interval_s == DEFAULT_ALLTOALL_WATCHDOG_POLL_INTERVAL_S


def test_wide_ep_ft_options_create_shared_health_when_enabled(monkeypatch) -> None:
    monkeypatch.setenv("TRTLLM_ENABLE_WIDE_EP_FT", "1")
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


def test_watchdog_timeout_reports_and_marks_missing_remote_ranks() -> None:
    health = EPGroupHealth(4)
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
    assert event.marked_failed_ranks == (1, 3)
    assert health.get_failed_ranks() == frozenset({1, 3})


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


def test_watchdog_reports_local_missing_but_does_not_mark_local_failed() -> None:
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
    assert event.marked_failed_ranks == ()
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
    assert event.marked_failed_ranks == ()
    assert health.all_active() is True


def test_watchdog_poll_timeout_with_prior_snapshot_marks_known_missing_rank() -> None:
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
    assert event.marked_failed_ranks == (1,)
    assert health.get_failed_ranks() == frozenset({1})


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
    assert health.get_failed_ranks() == frozenset({1})


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
    assert events[0].marked_failed_ranks == (0,)
    assert health.get_failed_ranks() == frozenset({0})


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
