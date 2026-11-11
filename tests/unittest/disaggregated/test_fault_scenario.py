# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for deterministic asynchronous fault scenarios."""

import asyncio

import pytest
from test_common.disagg_faults import Fault, FaultScenario, InjectedFault

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize(
    "kwargs",
    [
        {"boundary": ""},
        {"key": ""},
        {"action": "unknown"},
        {"occurrence": 0},
        {"occurrence": -1},
        {"occurrence": True},
        {"occurrence": 1.5},
        {"action": "delay"},
        {"gate": "unused"},
    ],
)
def test_invalid_fault(kwargs):
    fields = {"boundary": "deliver", "key": "r1", "action": "drop"}
    fields.update(kwargs)
    with pytest.raises(ValueError):
        Fault(**fields)


def test_duplicate_targets_and_gates():
    with pytest.raises(ValueError, match="Multiple faults"):
        FaultScenario([Fault("deliver", "r1", "drop")] * 2)
    with pytest.raises(ValueError, match="Duplicate gate"):
        FaultScenario([Fault("deliver", key, "delay", gate="shared") for key in ("r1", "r2")])


def test_targeted_drop_and_retry():
    async def exercise():
        scenario = FaultScenario([Fault("deliver", "r1", "drop", occurrence=2)])
        calls = []

        async def deliver(key):
            calls.append(key)
            return key

        assert await scenario.run("deliver", "r1", lambda: deliver("first")) == "first"
        assert await scenario.run("other", "r1", lambda: deliver("other")) == "other"
        assert await scenario.run("deliver", "r2", lambda: deliver("r2")) == "r2"
        assert await scenario.run("deliver", "r1", lambda: deliver("dropped")) is None
        assert await scenario.run("deliver", "r1", lambda: deliver("retry")) == "retry"
        assert calls == ["first", "other", "r2", "retry"]
        scenario.assert_consumed()
        scenario.assert_idle()

    asyncio.run(exercise())


def test_delay_does_not_block_another_key():
    async def exercise():
        scenario = FaultScenario([Fault("deliver", "r1", "delay", gate="resume")])
        calls = []

        async def deliver(key):
            calls.append(key)

        task = asyncio.create_task(scenario.run("deliver", "r1", lambda: deliver("r1")))
        try:
            await asyncio.wait_for(scenario.wait_until_blocked("resume"), 1)
            with pytest.raises(AssertionError, match="Active boundary"):
                scenario.assert_idle()
            await asyncio.wait_for(scenario.run("deliver", "r2", lambda: deliver("r2")), 1)
            assert calls == ["r2"]
            assert not task.done()
            scenario.release("resume")
            await asyncio.wait_for(task, 1)
            assert calls == ["r2", "r1"]
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        scenario.assert_consumed()
        scenario.assert_idle()
        assert [(e.key, e.event) for e in scenario.trace] == [
            ("r1", "entered"),
            ("r1", "blocked"),
            ("r2", "entered"),
            ("r2", "completed"),
            ("r1", "released"),
            ("r1", "completed"),
        ]

    asyncio.run(exercise())


@pytest.mark.parametrize("during_operation", [False, True])
def test_cancellation_cleans_active_call(during_operation):
    async def exercise():
        scenario = FaultScenario(
            [] if during_operation else [Fault("deliver", "r1", "delay", gate="resume")]
        )
        started = asyncio.Event()
        cleaned = []

        async def operation():
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.append("r1")

        task = asyncio.create_task(scenario.run("deliver", "r1", operation))
        try:
            reached = started.wait() if during_operation else scenario.wait_until_blocked("resume")
            await asyncio.wait_for(reached, 1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 1)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        assert cleaned == (["r1"] if during_operation else [])
        assert scenario.trace[-1].event == "cancelled"
        scenario.assert_consumed()
        scenario.assert_idle()

    asyncio.run(exercise())


def test_injected_failure_does_not_start_operation():
    async def exercise():
        scenario = FaultScenario([Fault("deliver", "r1", "raise")])

        async def operation():
            pytest.fail("The injected failure must precede the operation.")

        with pytest.raises(InjectedFault, match="r1"):
            await scenario.run("deliver", "r1", operation)
        scenario.assert_consumed()
        scenario.assert_idle()

    asyncio.run(exercise())


def test_operation_error_is_preserved():
    async def exercise():
        scenario = FaultScenario()
        error = ValueError("original error")

        async def operation():
            raise error

        with pytest.raises(ValueError) as caught:
            await scenario.run("deliver", "r1", operation)
        assert caught.value is error
        assert scenario.trace[-1].event == "failed"
        scenario.assert_idle()

    asyncio.run(exercise())


def test_unconsumed_fault_is_not_a_silent_success():
    scenario = FaultScenario([Fault("deliver", "r1", "drop")])
    with pytest.raises(AssertionError, match="Unconsumed faults"):
        scenario.assert_consumed()


def test_unknown_gate_is_rejected():
    scenario = FaultScenario()
    with pytest.raises(KeyError):
        scenario.release("typo")

    async def exercise():
        with pytest.raises(KeyError):
            await scenario.wait_until_blocked("typo")

    asyncio.run(exercise())
