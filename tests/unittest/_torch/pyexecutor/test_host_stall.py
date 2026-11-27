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
    # Only the loop's own mask, at start and restored at the end; no thread is re-pinned after the stop.
    assert calls == [0, 0]


def test_event_pools_are_per_thread():
    """Two threads never share a pool, so no object is handed to both."""
    pools, handed = {}, {}
    barrier = threading.Barrier(2)

    def worker(name):
        barrier.wait()
        pool = hs._thread_pool("dev", object)
        pools[name] = pool
        held = [pool.acquire() for _ in range(8)]
        handed[name] = {id(obj) for obj in held}
        barrier.wait()  # both still hold their objects here

    threads = [threading.Thread(target=worker, args=(n,)) for n in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert pools["a"] is not pools["b"]
    assert not handed["a"] & handed["b"]
    # The same thread and key get the same pool.
    assert hs._thread_pool("dev", object) is hs._thread_pool("dev", object)
    assert hs._thread_pool("other", object) is not hs._thread_pool("dev", object)


def test_a_held_object_is_never_handed_out_again():
    pool = hs.EventPool(object)
    held = [pool.acquire() for _ in range(4)]
    assert len({id(obj) for obj in held + [pool.acquire()]}) == 5


@pytest.mark.parametrize(
    "affinity, force, pinned",
    [(None, None, True), ("1", None, True), ("0", None, False), ("0", "2", True)],
)
def test_model_pinning_honors_the_affinity_opt_out(monkeypatch, affinity, force, pinned):
    if affinity is None:
        monkeypatch.delenv("TLLM_NUMA_AWARE_WORKER_AFFINITY", raising=False)
    else:
        monkeypatch.setenv("TLLM_NUMA_AWARE_WORKER_AFFINITY", affinity)
    if force is None:
        monkeypatch.delenv(hs.PIN_CORES_ENV, raising=False)
    else:
        monkeypatch.setenv(hs.PIN_CORES_ENV, force)
    called = []
    monkeypatch.setattr(hs, "pin_executor_thread", lambda n, *a, **k: called.append(n) or "")
    hs.arm_executor_thread(_executor_with_pin_hook(2))
    assert called == ([2] if pinned else [])


def test_a_second_loop_entry_stops_the_earlier_remask_thread(monkeypatch):
    monkeypatch.delenv(hs.PIN_CORES_ENV, raising=False)
    monkeypatch.delenv("TLLM_NUMA_AWARE_WORKER_AFFINITY", raising=False)
    events = []

    def fake_pin(ncores, stop=None, started=None, **_):
        def loop():
            stop.wait(30)
            events.append("old_restored")

        thread = threading.Thread(target=loop, daemon=True)
        thread.start()
        events.append("started")
        started.append(thread)
        return ""

    monkeypatch.setattr(hs, "pin_executor_thread", fake_pin)
    executor = _executor_with_pin_hook(2)
    hs.arm_executor_thread(executor)
    first, first_thread = executor._pin_remask_stop, executor._pin_remask_thread
    hs.arm_executor_thread(executor)
    try:
        assert first.is_set() and not executor._pin_remask_stop.is_set()
        # The old loop restored its masks before the new one started.
        assert not first_thread.is_alive()
        assert events == ["started", "old_restored", "started"], events
    finally:
        executor._pin_remask_stop.set()
        executor._pin_remask_thread.join(5)


def _two_gpus_per_socket(monkeypatch, own_phys):
    """2 sockets x 88 SMT cores (siblings c and c + 176), 4 GPUs, 2 per socket; masks per socket."""
    import sys
    import types

    socket0 = set(range(0, 88)) | set(range(176, 264))
    numa = {0: socket0, 1: set(range(88, 176)) | set(range(264, 352))}
    sib = {c: {c % 176, c % 176 + 176} for c in range(352)}
    monkeypatch.setattr(hs, "_topology", lambda: (numa, sib))
    monkeypatch.setattr(hs, "_node_gpu_numa", lambda own: ({0: 0, 1: 0, 2: 1, 3: 1}, own_phys))
    monkeypatch.setattr(os, "sched_getaffinity", lambda tid: set(socket0), raising=False)
    torch_stub = types.SimpleNamespace(
        cuda=types.SimpleNamespace(current_device=lambda: 0, device_count=lambda: 1)
    )
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setattr(hs, "_PIN_STATE", {})
    return socket0


def test_ranks_on_one_socket_get_disjoint_cores_below_the_reserved_tail(monkeypatch):
    plans = []
    for own_phys in (0, 1):
        socket0 = _two_gpus_per_socket(monkeypatch, own_phys)
        dedicated, _, _, _ = hs._plan_executor_pin(2, 8)
        plans.append(dedicated)
    assert plans[0] and plans[1] and not plans[0] & plans[1]
    tail = set(range(80, 88)) | set(range(256, 264))
    assert not (plans[0] | plans[1]) & tail
    assert (plans[0] | plans[1]) <= socket0


def test_a_second_plan_reuses_the_first(monkeypatch):
    _two_gpus_per_socket(monkeypatch, 0)
    first = hs._plan_executor_pin(2, 8)
    # The pinned thread's own mask is now its two cores; the plan must not shrink to it.
    monkeypatch.setattr(os, "sched_getaffinity", lambda tid: set(first[0]), raising=False)
    assert hs._plan_executor_pin(2, 8) is first


def test_remask_loop_checks_handled_threads_only_on_full_passes(monkeypatch):
    """After the first pass, an unchanged thread is not read again until the periodic full pass."""
    reads = []
    monkeypatch.setattr(hs, "_REMASK_FULL_EVERY", 3)
    monkeypatch.setattr(os, "getpid", lambda: 100)
    monkeypatch.setattr(os, "listdir", lambda path: ["101", "102"])
    monkeypatch.setattr(hs, "_children_pids", lambda pid: [])
    monkeypatch.setattr(os, "sched_setaffinity", lambda tid, cpus: None, raising=False)

    def getaffinity(tid):
        reads.append(tid)
        return {5} if tid == 101 else {6}

    monkeypatch.setattr(os, "sched_getaffinity", getaffinity, raising=False)
    passes = []

    class _Stop:
        def is_set(self):
            return len(passes) >= 4

        def wait(self, timeout):
            passes.append(list(reads))
            reads.clear()

    hs._remask_loop(101, {5}, {5}, {5, 6, 7}, time.monotonic(), threading.current_thread(), _Stop())
    # Pass 0 (full): both; passes 1-2: only the executor thread; pass 3 (full): both.
    assert [sorted(set(p) - {0}) for p in passes] == [[101, 102], [101], [101], [101, 102]]
