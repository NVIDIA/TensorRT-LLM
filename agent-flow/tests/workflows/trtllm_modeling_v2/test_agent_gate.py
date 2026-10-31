# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
import time

import pytest

from agent_flow.workflows.trtllm_modeling_v2.agents import slots
from agent_flow.workflows.trtllm_modeling_v2.agents.slots import AgentGate, Priority


@pytest.fixture(autouse=True)
def _quick_polls(monkeypatch):
    """Background and side turns look again every 0.1s here, not every 5s."""
    monkeypatch.setattr(slots, "_BACKGROUND_POLL_S", 0.1)


class _Turns:
    """Turns that hold their slot until released, started one thread each."""

    def __init__(self, gate: AgentGate) -> None:
        self.gate = gate
        self.release = threading.Event()
        self.admitted: list[str] = []
        self._lock = threading.Lock()

    def start(self, name: str, priority: Priority) -> None:
        def hold():
            with self.gate.slot(priority):
                with self._lock:
                    self.admitted.append(name)
                self.release.wait()

        threading.Thread(target=hold, daemon=True).start()

    def settle(self) -> None:
        time.sleep(0.2)


def test_the_planner_owns_the_reserved_slot_and_takes_no_other():
    turns = _Turns(AgentGate(8))
    for i in range(8):
        turns.start(f"task{i}", Priority.REQUIRED)
    turns.settle()
    assert len(turns.admitted) == 7  # the eighth task turn waits: the last slot is the planner's
    turns.start("critic", Priority.SIDE)
    turns.settle()
    assert "critic" not in turns.admitted
    turns.start("planner", Priority.PLANNER)
    turns.settle()
    assert "planner" in turns.admitted and len(turns.admitted) == 8
    turns.release.set()
    # Arriving first, the planner still leaves every task slot to the tasks.
    first = _Turns(AgentGate(4))
    first.start("planner", Priority.PLANNER)
    first.settle()
    for i in range(3):
        first.start(f"task{i}", Priority.REQUIRED)
    first.settle()
    assert len(first.admitted) == 4
    first.release.set()
