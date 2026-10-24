# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""":class:`CheckAdmission` runs on a fake reading of the sessions and a clock the test moves."""

from __future__ import annotations

import threading
import time

from agent_flow.workflows.trtllm_modeling_v2.infra.environment import RunEnvironment
from agent_flow.workflows.trtllm_modeling_v2.infra.execution import CheckAdmission
from agent_flow.workflows.trtllm_modeling_v2.infra.execution.capacity import CHECK_SETTLE_S


class _Sessions:
    """A reading of the sessions the test sets."""

    def __init__(self, free: int) -> None:
        self.free = free

    def __call__(self, environment, gpus, wall_s):
        return self.free


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


_ENV = RunEnvironment(kind="session_spool", python="python3", spool="/s/gates-1")


class _Check:
    """A check thread that starts once admitted and runs until :meth:`finish`."""

    def __init__(self, admission: CheckAdmission) -> None:
        self.started, self._done = threading.Event(), threading.Event()
        self._thread = threading.Thread(target=self._run, args=(admission,), daemon=True)
        self._thread.start()

    def _run(self, admission: CheckAdmission) -> None:
        with admission.slot(_ENV):
            self.started.set()
            self._done.wait(10)

    def finish(self) -> None:
        self._done.set()
        self._thread.join(5)


def _settles(predicate, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def test_checks_start_while_devices_are_free_and_wait_once_they_are_not():
    sessions, clock = _Sessions(free=3), _Clock()
    admission = CheckAdmission(sessions, clock=clock)
    held = [admission.slot(_ENV) for _ in range(3)]
    for slot in held:
        slot.__enter__()
    assert admission.in_flight == 3
    sessions.free = 0
    # The running checks' jobs have had time to show in the spools.
    clock.now += CHECK_SETTLE_S
    check = _Check(admission)
    assert _settles(lambda: admission.waiting == 1)
    assert not check.started.wait(0.1)
    # A check ends; its device is free, and the sessions are read again at once.
    sessions.free = 1
    held.pop().__exit__(None, None, None)
    assert check.started.wait(5)
    check.finish()
    for slot in held:
        slot.__exit__(None, None, None)
    assert admission.in_flight == 0
