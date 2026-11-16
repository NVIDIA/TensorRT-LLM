# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Executor-thread core pinning (TRTLLM_EXECUTOR_THREAD_PIN, M3.1 hoststall PIN port): the pure plan."""

import os

import pytest

from tensorrt_llm._torch.pyexecutor import executor_affinity as ea

pytestmark = pytest.mark.cpu_only


def _no_smt(cpus):
    return {c: {c} for c in cpus}


SOCKET0, SOCKET1 = frozenset(range(0, 72)), frozenset(range(72, 144))
GB300 = {0: SOCKET0, 1: SOCKET0, 2: SOCKET1, 3: SOCKET1}  # 2 GPUs per Grace socket


def test_parse_and_format_cpulists():
    assert ea.parse_cpulist("0-3,8,10-11\n") == {0, 1, 2, 3, 8, 10, 11}
    assert ea.format_cpus({0, 1, 2, 3, 8, 10, 11}) == "0-3,8,10-11"
    assert ea.format_cpus(set()) == ""


def test_gb300_under_the_numa_mask_gives_each_socket_rank_its_own_cores():
    sib = _no_smt(range(144))
    plans = {}
    for gpu, socket in GB300.items():
        plans[gpu] = ea.plan_pin(set(socket), GB300, sib, ncores=2, reserve_tail=8)
    # Each rank sees only its socket (TLLM_NUMA_AWARE_WORKER_AFFINITY=1): its own and its neighbour's slots.
    assert plans[0] == plans[1] == {0: {62, 63}, 1: {60, 61}}
    assert plans[2] == plans[3] == {2: {134, 135}, 3: {132, 133}}
    dedicated = [plans[g][g] for g in GB300]
    assert all(not (a & b) for i, a in enumerate(dedicated) for b in dedicated[i + 1 :])


def test_unconstrained_process_plans_every_gpu_on_its_own_socket():
    sib = _no_smt(range(144))
    plan = ea.plan_pin(set(range(144)), GB300, sib, ncores=2, reserve_tail=8)
    assert plan == {0: {62, 63}, 1: {60, 61}, 2: {134, 135}, 3: {132, 133}}


def test_smt_siblings_go_together_and_partial_cores_are_skipped():
    # 8 physical cores, siblings c and c + 8; cpu 15 (sibling of 7) is not allowed.
    sib = {c: {c % 8, c % 8 + 8} for c in range(16)}
    gpus = {0: frozenset(range(16)), 1: frozenset(range(16))}
    allowed = set(range(15))
    plan = ea.plan_pin(allowed, gpus, sib, ncores=1, reserve_tail=1)
    # Usable physical cores 0..6 (7 has a disallowed sibling); tail 1 skipped -> GPU 0 core 5, GPU 1 core 4.
    assert plan == {0: {5, 13}, 1: {4, 12}}


def test_not_enough_cores_leaves_the_gpu_out():
    sib = _no_smt(range(10))
    gpus = {0: frozenset(range(10)), 1: frozenset(range(10))}
    assert ea.plan_pin(set(range(10)), gpus, sib, ncores=2, reserve_tail=7) == {0: {1, 2}}


def test_remask_target_keeps_the_threads_own_mask_minus_the_dedicated_cores():
    assert ea.remask_target({1, 2, 3}, {0, 1, 2, 3, 4}, {3}) == {1, 2}
    assert ea.remask_target({3}, {0, 1, 2, 3, 4}, {3}) == {0, 1, 2, 4}


def test_off_by_default_changes_nothing(monkeypatch):
    monkeypatch.delenv(ea.EXECUTOR_PIN_ENV, raising=False)
    monkeypatch.setattr(ea, "_pin", lambda rank, device=None: pytest.fail("pinned while off"))
    assert ea.maybe_pin_executor_thread(0) is None


def test_on_pins_once_and_never_raises(monkeypatch, capfd):
    monkeypatch.setenv(ea.EXECUTOR_PIN_ENV, "1")
    monkeypatch.setattr(ea, "_pinned", None)
    calls = []

    def boom(rank, device=None):
        calls.append(rank)
        raise OSError("no affinity here")

    monkeypatch.setattr(ea, "_pin", boom)
    assert ea.maybe_pin_executor_thread(3).startswith("fail(OSError")
    assert ea.maybe_pin_executor_thread(3).startswith("fail(OSError")
    assert calls == [3]
    assert "M31_EXEC_PIN_ARMED rank=3 fail(" in capfd.readouterr().err


def test_pin_uses_the_executor_device_not_the_threads_current_device(monkeypatch, capfd):
    """The event-loop thread pins before its first cudaSetDevice; current_device() there is 0 on every rank."""
    monkeypatch.setenv(ea.EXECUTOR_PIN_ENV, "1")
    monkeypatch.setattr(ea, "_pinned", None)
    seen = []
    monkeypatch.setattr(ea, "_pin", lambda rank, device=None: seen.append((rank, device)) or f"cpus=x gpu={device}")
    assert ea.maybe_pin_executor_thread(2, 2) == "cpus=x gpu=2"
    assert seen == [(2, 2)]
    assert "M31_EXEC_PIN_ARMED rank=2 cpus=x gpu=2" in capfd.readouterr().err
    assert ea._resolve_device(3) == 3


def test_physical_gpu_follows_cuda_visible_devices(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    assert ea._physical_gpu(0) == 2 and ea._physical_gpu(1) == 3
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    assert ea._physical_gpu(1) == 1


@pytest.mark.skipif(not hasattr(os, "sched_setaffinity"), reason="Linux only")
def test_live_topology_plan_is_inside_the_allowed_cpus():
    """On the test host: NVML locality (when available) and sysfs siblings give a plan inside the process mask."""
    allowed = set(os.sched_getaffinity(0))
    try:
        gpu_cpus = ea._nvml_gpu_cpus()
    except Exception:  # noqa: BLE001 - no NVML on a CPU host
        pytest.skip("no NVML")
    plan = ea.plan_pin(allowed, gpu_cpus, ea._siblings(allowed))
    for cpus in plan.values():
        assert cpus <= allowed and len(cpus) >= 2
    print("[exec-pin] host plan", {g: ea.format_cpus(c) for g, c in plan.items()})
