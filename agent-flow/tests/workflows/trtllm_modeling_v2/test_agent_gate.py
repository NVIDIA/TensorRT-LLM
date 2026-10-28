# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
import time

from agent_flow.workflows.trtllm_modeling_v2.agents.slots import AgentGate, Priority


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


def test_the_planner_owns_the_reserved_slot():
    gate = AgentGate(8)
    turns = _Turns(gate)
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


def test_a_freed_slot_goes_to_the_highest_priority_waiting():
    gate = AgentGate(2)  # one slot for turns other than the planner's
    admitted: list[str] = []
    done = {name: threading.Event() for name in ("first", "tuner", "primer", "core")}

    def turn(name, priority):
        with gate.slot(priority):
            admitted.append(name)
            done[name].wait()

    def start(name, priority):
        threading.Thread(target=turn, args=(name, priority), daemon=True).start()
        time.sleep(0.2)

    start("first", Priority.REQUIRED)
    start("tuner", Priority.BACKGROUND)
    start("primer", Priority.SIDE)
    start("core", Priority.CRITICAL)
    assert admitted == ["first"]
    for name in ("first", "core", "primer"):
        done[name].set()  # each freed slot goes to the highest priority still waiting
        time.sleep(0.2)
    assert admitted == ["first", "core", "primer", "tuner"]
    done["tuner"].set()


def test_background_turns_use_only_the_slots_the_foreground_leaves_idle():
    idle = {"slots": 1}
    gate = AgentGate(8, background_limit=lambda: idle["slots"])
    turns = _Turns(gate)
    for i in range(4):
        turns.start(f"tuner{i}", Priority.BACKGROUND)
    turns.settle()
    assert len(turns.admitted) == 1
    idle["slots"] = 3  # the foreground's work ended: more idle slots, seen on the next look
    time.sleep(5.5)
    assert len(turns.admitted) == 3
    turns.release.set()


def test_a_planner_turn_in_its_reserved_slot_takes_no_slot_from_the_tasks():
    gate = AgentGate(4)
    turns = _Turns(gate)
    turns.start("planner", Priority.PLANNER)
    turns.settle()
    for i in range(3):
        turns.start(f"task{i}", Priority.REQUIRED)
    turns.settle()
    assert len(turns.admitted) == 4
    turns.release.set()


def test_a_side_turn_waits_for_idle_slots_and_takes_one_once_it_has_waited_long_enough():
    idle = {"slots": 0}
    gate = AgentGate(8, side_limit=lambda: idle["slots"], side_max_wait_s=2.0)
    turns = _Turns(gate)
    turns.start("primer", Priority.SIDE)
    turns.settle()
    assert "primer" not in turns.admitted, "the ready tasks the loop starts come first"
    time.sleep(5.5)  # past its wait, seen on the next look
    assert "primer" in turns.admitted, "it is not starved for ever"
    turns.release.set()
