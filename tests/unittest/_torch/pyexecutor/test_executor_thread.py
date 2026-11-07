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
"""Executor loop thread placement (``executor_thread.py``).

* ``TLLM_NUMA_AWARE_WORKER_AFFINITY=pin_executor``: CPU choice and the thread affinities it sets.
* ``enable_min_latency``: the interpreter's thread switch interval while the loop runs.
"""

import os
import sys
import threading

import pytest

from tensorrt_llm._torch.pyexecutor import executor_thread
from tensorrt_llm._torch.pyexecutor.executor_thread import (
    MIN_LATENCY_SWITCH_INTERVAL_S,
    NUMA_AFFINITY_ENV,
    PIN_EXECUTOR_AFFINITY,
    ExecutorThreadPin,
    executor_pin_requested,
    executor_switch_interval,
    pin_executor_cpu,
)

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize(
    "available,gpu,expected",
    [
        ({0, 1, 2}, 0, 1),
        ({4, 5, 6}, 0, 4),
        ({0}, 0, 0),
        (set(), 0, None),
        # Two GPUs sharing one NUMA-wide set get different cores.
        (set(range(8)), 0, 1),
        (set(range(8)), 1, 2),
        # GPUs 2 / 3 on the second socket's set.
        (set(range(8, 16)), 2, 10),
        (set(range(8, 16)), 3, 11),
    ],
)
def test_pin_executor_cpu_choice(available, gpu, expected, monkeypatch):
    monkeypatch.setattr(executor_thread, "_smt_siblings", lambda cpu: {cpu})
    assert pin_executor_cpu(available, gpu) == expected


def test_pin_executor_cpu_skips_smt_siblings(monkeypatch):
    # CPUs 0-3 are cores, 4-7 their SMT siblings (0/4, 1/5, ...).
    monkeypatch.setattr(executor_thread, "_smt_siblings", lambda cpu: {cpu % 4, cpu % 4 + 4})
    assert [pin_executor_cpu(set(range(8)), g) for g in range(4)] == [1, 2, 3, 1]


@pytest.mark.parametrize(
    "value,requested",
    [(None, False), ("", False), ("0", False), ("1", False), ("pin_executor", True)],
)
def test_only_the_pin_executor_policy_pins(value, requested, monkeypatch):
    if value is None:
        monkeypatch.delenv(NUMA_AFFINITY_ENV, raising=False)
    else:
        monkeypatch.setenv(NUMA_AFFINITY_ENV, value)
    assert executor_pin_requested() is requested


def test_executor_thread_pin_off_touches_nothing(monkeypatch):
    monkeypatch.setenv(NUMA_AFFINITY_ENV, "1")

    def fail(*args):
        raise AssertionError("affinity queried without the pin_executor policy")

    monkeypatch.setattr(os, "sched_getaffinity", fail)
    monkeypatch.setattr(os, "sched_setaffinity", fail)
    assert ExecutorThreadPin.create() is None


@pytest.mark.parametrize("min_latency", [False, True])
def test_switch_interval_follows_enable_min_latency(min_latency):
    original = sys.getswitchinterval()
    with executor_switch_interval(min_latency):
        inside = sys.getswitchinterval()
    assert sys.getswitchinterval() == original
    if min_latency:
        assert inside == pytest.approx(MIN_LATENCY_SWITCH_INTERVAL_S, rel=0.05)
    else:
        assert inside == original


def test_switch_interval_is_restored_on_error():
    original = sys.getswitchinterval()
    with pytest.raises(RuntimeError):
        with executor_switch_interval(True):
            raise RuntimeError("loop failed")
    assert sys.getswitchinterval() == original


@pytest.mark.skipif(len(os.sched_getaffinity(0)) < 4, reason="needs at least 4 CPUs")
def test_executor_thread_pin_affinities(monkeypatch):
    monkeypatch.setenv(NUMA_AFFINITY_ENV, PIN_EXECUTOR_AFFINITY)
    process_cpus = os.sched_getaffinity(0)
    original = {}
    for tid in os.listdir("/proc/self/task"):
        try:
            original[int(tid)] = os.sched_getaffinity(int(tid))
        except OSError:
            pass
    stop = threading.Event()
    bystander_ready = threading.Event()
    bystander_tid = []

    def bystander():
        bystander_tid.append(threading.get_native_id())
        bystander_ready.set()
        stop.wait(30)

    late_tid = []
    result = {}

    def executor():
        pin = ExecutorThreadPin.create()
        result["pin"] = pin
        result["own"] = os.sched_getaffinity(0)
        # A thread started later inherits the executor's single CPU until a refresh.
        late_ready = threading.Event()

        def late():
            late_tid.append(threading.get_native_id())
            late_ready.set()
            stop.wait(30)

        result["late_thread"] = threading.Thread(target=late, daemon=True)
        result["late_thread"].start()
        late_ready.wait(10)
        result["late_before"] = os.sched_getaffinity(late_tid[0])
        result["moved"] = pin.refresh()
        result["late_after"] = os.sched_getaffinity(late_tid[0])

    by = threading.Thread(target=bystander, daemon=True)
    by.start()
    bystander_ready.wait(10)
    ex = threading.Thread(target=executor)
    try:
        ex.start()
        ex.join(30)
        pin = result["pin"]
        assert pin is not None
        reserved = executor_thread._smt_siblings(pin.cpu)
        assert result["own"] == {pin.cpu}
        assert pin.cpu in process_cpus and pin.cpu != 0 or process_cpus == {0}
        others = os.sched_getaffinity(bystander_tid[0])
        assert not others & reserved and others <= process_cpus
        assert result["late_before"] == {pin.cpu}
        assert result["moved"] >= 1
        assert not result["late_after"] & reserved
    finally:
        stop.set()
        by.join(10)
        if "late_thread" in result:
            result["late_thread"].join(10)
        for tid, cpus in original.items():
            try:
                os.sched_setaffinity(tid, cpus)
            except OSError:
                pass
