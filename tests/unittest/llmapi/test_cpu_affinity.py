# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import tensorrt_llm.llmapi.utils as utils

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def affinity(monkeypatch):
    current = [0, 1, 2, 3]
    bindings = []
    process = SimpleNamespace(cpu_affinity=lambda: list(current))

    def bind(cpus):
        bindings.append(list(cpus))
        current[:] = cpus
        return 3, 3

    monkeypatch.setattr(utils, "_configured_cpu_affinity", None, raising=False)
    monkeypatch.setattr(
        utils, "psutil", SimpleNamespace(Process=lambda *args: process, cpu_count=lambda: 4)
    )
    monkeypatch.setattr(utils, "_set_affinity_all_threads", bind)
    monkeypatch.setattr(
        utils, "get_numa_aware_cpu_affinity", lambda device_id: [0, 1] if device_id == 0 else [2, 3]
    )
    monkeypatch.setattr(utils, "logger", Mock())
    monkeypatch.delenv("TLLM_NUMA_AWARE_WORKER_AFFINITY", raising=False)
    return current, bindings


def test_repeated_configuration_reapplies_numa_mask(affinity):
    current, bindings = affinity
    utils.configure_cpu_affinity(0)
    utils.configure_cpu_affinity(0)
    utils.configure_cpu_affinity(0)
    assert current == [0, 1]
    assert bindings == [[0, 1]] * 3
    utils.logger.warning.assert_not_called()


def test_configuration_can_retarget_an_owned_mask(affinity):
    current, bindings = affinity
    utils.configure_cpu_affinity(0)
    utils.configure_cpu_affinity(1)
    assert current == [2, 3]
    assert bindings == [[0, 1], [2, 3]]


def test_external_mask_change_keeps_existing_policy(affinity):
    current, bindings = affinity
    utils.configure_cpu_affinity(0)
    current[:] = [3]
    utils.configure_cpu_affinity(0)
    assert current == [0, 1, 2, 3]
    assert bindings == [[0, 1], [0, 1, 2, 3]]


@pytest.mark.parametrize("setting", ["0", "other"])
def test_opt_out_does_not_rebind_owned_mask(affinity, monkeypatch, setting):
    current, bindings = affinity
    utils.configure_cpu_affinity(0)
    monkeypatch.setenv("TLLM_NUMA_AWARE_WORKER_AFFINITY", setting)
    utils.configure_cpu_affinity(1)
    assert current == [0, 1]
    assert bindings == [[0, 1]]
    utils.logger.warning.assert_not_called()


@pytest.mark.parametrize("setting, expected", [(None, [0, 1, 2, 3]), ("0", [3]), ("1", [0, 1])])
def test_external_affinity_policy(affinity, monkeypatch, setting, expected):
    current, _ = affinity
    current[:] = [3]
    if setting is not None:
        monkeypatch.setenv("TLLM_NUMA_AWARE_WORKER_AFFINITY", setting)
    utils.configure_cpu_affinity(0)
    assert current == expected


def test_owned_affinity_does_not_survive_pid_change(affinity, monkeypatch):
    current, _ = affinity
    utils.configure_cpu_affinity(0)
    child_pid = os.getpid() + 1
    monkeypatch.setattr(os, "getpid", lambda: child_pid)
    utils.configure_cpu_affinity(0)
    assert current == [0, 1, 2, 3]


@pytest.mark.parametrize("bound", [0, 2])
def test_failed_leader_binding_does_not_claim_external_mask(affinity, monkeypatch, bound):
    current, _ = affinity
    current[:] = [3]
    monkeypatch.setenv("TLLM_NUMA_AWARE_WORKER_AFFINITY", "1")
    monkeypatch.setattr(utils, "_set_affinity_all_threads", lambda cpus: (bound, 3))
    utils.configure_cpu_affinity(0)
    assert utils._configured_cpu_affinity is None


def test_partial_binding_keeps_owned_mask_for_retry(affinity, monkeypatch):
    current, bindings = affinity

    def partial_bind(cpus):
        bindings.append(list(cpus))
        current[:] = cpus
        return 1, 3

    monkeypatch.setattr(utils, "_set_affinity_all_threads", partial_bind)
    utils.configure_cpu_affinity(0)
    utils.configure_cpu_affinity(0)
    assert current == [0, 1]
    assert bindings == [[0, 1], [0, 1]]


def test_proper_subset_readback_remains_owned_for_retry(affinity, monkeypatch):
    current, bindings = affinity

    def subset_bind(cpus):
        bindings.append(list(cpus))
        current[:] = [0]
        return 1, 3

    monkeypatch.setattr(utils, "_set_affinity_all_threads", subset_bind)
    utils.configure_cpu_affinity(0)
    utils.configure_cpu_affinity(0)
    assert current == [0]
    assert bindings == [[0, 1], [0, 1]]
    utils.logger.warning.assert_not_called()


_REAL_CONFIGURE_PROBE = """
import os
import threading

import psutil
import tensorrt_llm.llmapi.utils as utils

available = psutil.Process().cpu_affinity()
if len(available) < 2:
    print("SKIP: fewer than two available CPUs")
    raise SystemExit(0)
subset = sorted(available)[:len(available) // 2]
utils.get_numa_aware_cpu_affinity = lambda device_id: subset
release = threading.Event()
threads = []

def start_worker():
    ready = threading.Event()
    def worker():
        ready.set()
        release.wait(30)
    thread = threading.Thread(target=worker, daemon=True)
    threads.append(thread)
    thread.start()
    assert ready.wait(10)

try:
    start_worker()
    # Force the first pin even when the test starts in a restricted cpuset.
    os.environ["TLLM_NUMA_AWARE_WORKER_AFFINITY"] = "1"
    utils.configure_cpu_affinity(0)
    del os.environ["TLLM_NUMA_AWARE_WORKER_AFFINITY"]
    start_worker()
    utils.configure_cpu_affinity(0)
    assert psutil.Process().cpu_affinity() == subset
    for thread in threads:
        assert sorted(os.sched_getaffinity(thread.native_id)) == subset
    print("OK: original and later threads retain the NUMA mask")
finally:
    release.set()
    for thread in threads:
        thread.join(10)
"""


@pytest.mark.skipif(
    not hasattr(os, "sched_setaffinity") or not os.path.isdir("/proc/self/task"),
    reason="Linux procfs affinity only",
)
def test_repeated_configuration_preserves_real_thread_masks():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", _REAL_CONFIGURE_PROBE], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    if result.stdout.startswith("SKIP:"):
        pytest.skip(result.stdout.strip())
    assert "OK:" in result.stdout, result.stdout
