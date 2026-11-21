# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Executor host-stall options (host_stall.py): the sampler event pool and the executor thread pinning plan.
CPU only."""

import os
import threading
import time

import pytest

from tensorrt_llm._torch.pyexecutor import host_stall as hs

pytestmark = pytest.mark.cpu_only


def test_event_pool_reuses_only_unreferenced_objects():
    class Ev:
        pass

    pool = hs.EventPool(Ev, cap=3)
    a = pool.acquire()
    held = [a]
    b = pool.acquire()
    assert b is not a, "held event reused"
    del b  # pool-only now
    c = pool.acquire()
    assert c is not a
    held.append(c)
    d = pool.acquire()
    assert all(d is not x for x in held)
    held.append(d)
    pool.acquire()  # cap reached and every pooled event held: an unpooled one
    assert len(pool) == 3 and pool.overflow == 1
    held.clear()
    del c, d, a
    f = pool.acquire()
    assert any(f is x for x in pool._pool)
    assert pool.hits >= 2


def test_plan_pin_two_numa_nodes_with_smt():
    numa = {
        0: set(range(0, 88)) | set(range(176, 264)),
        1: set(range(88, 176)) | set(range(264, 352)),
    }
    sib = {}
    for c in range(176):
        sib[c] = sib[c + 176] = {c, c + 176}
    allowed = set(range(352))
    plan = hs.plan_pin(allowed, numa, sib, {0: 0, 1: 0, 2: 1, 3: 1}, 2, 8)
    assert plan == {
        0: {78, 79, 254, 255},
        1: {76, 77, 252, 253},
        2: {166, 167, 342, 343},
        3: {164, 165, 340, 341},
    }
    excl = set().union(*plan.values())
    reserved = (
        set(range(80, 88)) | set(range(168, 176)) | set(range(256, 264)) | set(range(344, 352))
    )
    assert not excl & reserved
    assert hs.remask_target(set(range(352)), allowed, excl) == allowed - excl
    assert hs.remask_target(plan[0], allowed, excl) == allowed - excl
    assert hs.parse_cpulist("0-3,8,10-11") == {0, 1, 2, 3, 8, 10, 11}
    assert hs.format_cpulist({0, 1, 2, 5, 7, 8}) == "0-2,5,7-8"


def test_a_bad_reserve_tail_is_read_only_when_pinning(monkeypatch):
    monkeypatch.setenv(hs.PIN_RESERVE_TAIL_ENV, "eight")
    with pytest.raises(ValueError):
        hs.pin_reserve_tail()
    monkeypatch.setenv(hs.PIN_RESERVE_TAIL_ENV, "-1")
    assert hs.pin_reserve_tail() == 0


def test_pinning_is_off_by_default(monkeypatch):
    monkeypatch.delenv(hs.PIN_CORES_ENV, raising=False)
    called = []
    monkeypatch.setattr(hs, "pin_executor_thread", lambda *a, **k: called.append(a) or "")
    hs.arm_executor_thread(object())
    assert called == []


def _executor_with_pin_hook(cores):
    from types import SimpleNamespace

    from tensorrt_llm._torch.pyexecutor.runtime_hooks import RuntimeHooks

    model = SimpleNamespace(runtime_hooks=RuntimeHooks(executor_thread_pin_cores=cores))
    return SimpleNamespace(model_engine=SimpleNamespace(model=model), dist=SimpleNamespace(rank=0))


@pytest.mark.parametrize(
    "env, hook, expected",
    [(None, 2, 2), (None, 0, None), ("0", 2, None), ("3", 0, 3), ("1", 2, 1)],
)
def test_pinning_follows_the_model_unless_the_variable_is_set(monkeypatch, env, hook, expected):
    if env is None:
        monkeypatch.delenv(hs.PIN_CORES_ENV, raising=False)
    else:
        monkeypatch.setenv(hs.PIN_CORES_ENV, env)
    called = []
    monkeypatch.setattr(hs, "pin_executor_thread", lambda n, *a, **k: called.append(n) or "")
    hs.arm_executor_thread(_executor_with_pin_hook(hook))
    assert called == ([] if expected is None else [expected])


@pytest.mark.parametrize("how", ["executor_exited", "stop_set"])
def test_remask_loop_stops_with_its_executor(monkeypatch, how):
    calls = []
    monkeypatch.setattr(os, "sched_setaffinity", lambda tid, cpus: calls.append(tid), raising=False)
    monkeypatch.setattr(os, "sched_getaffinity", lambda tid: {0, 1}, raising=False)
    stop = threading.Event()
    if how == "executor_exited":
        exec_thread = threading.Thread(target=lambda: None)
        exec_thread.start()
        exec_thread.join()
    else:
        exec_thread = threading.current_thread()
        stop.set()
    loop = threading.Thread(
        target=hs._remask_loop,
        args=(12345, {0}, {0}, {0, 1}, time.monotonic(), exec_thread, stop),
        daemon=True,
    )
    loop.start()
    loop.join(timeout=2.0)
    assert not loop.is_alive()
    # Only the loop's own mask at start; no thread is re-pinned after the stop.
    assert calls == [0]
