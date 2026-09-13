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
import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import tensorrt_llm.llmapi.utils as utils

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def affinity_leases(monkeypatch):
    pid = 101
    all_cpus = frozenset({0, 1, 2, 3})
    numa_cpus = frozenset({0, 1})
    current = {pid: all_cpus, 102: frozenset({2, 3})}
    bindings = []
    configure_calls = []

    def read_thread_affinities():
        return dict(current)

    def configure(device_id):
        configure_calls.append(device_id)
        for tid in current:
            current[tid] = numa_cpus

    def setaffinity(tid, cpus):
        mask = frozenset(cpus)
        bindings.append((tid, mask))
        current[tid] = mask

    monkeypatch.setattr(utils, "_cpu_affinity_lease_state", None)
    monkeypatch.setattr(utils, "_next_cpu_affinity_lease", 0)
    monkeypatch.setattr(utils.os, "getpid", lambda: pid)
    monkeypatch.setattr(utils.os, "sched_getaffinity", lambda tid: set(current[pid]))
    monkeypatch.setattr(utils.os, "sched_setaffinity", setaffinity)
    monkeypatch.setattr(utils, "_read_thread_affinities", read_thread_affinities)
    monkeypatch.setattr(utils, "configure_cpu_affinity", configure)
    monkeypatch.setattr(utils, "logger", Mock())
    return current, bindings, configure_calls, all_cpus, numa_cpus


def test_last_lease_restores_existing_and_later_threads(affinity_leases):
    current, bindings, _, all_cpus, numa_cpus = affinity_leases
    lease = utils.acquire_cpu_affinity(0)
    current[103] = numa_cpus

    utils.release_cpu_affinity(lease)

    assert current == {101: all_cpus, 102: frozenset({2, 3}), 103: all_cpus}
    assert bindings == [(101, all_cpus), (102, frozenset({2, 3})), (103, all_cpus)]
    assert utils._cpu_affinity_lease_state is None


def test_later_threads_restore_calling_thread_mask(affinity_leases, monkeypatch):
    current, _, _, all_cpus, numa_cpus = affinity_leases
    caller_mask = frozenset({3})
    monkeypatch.setattr(utils.os, "sched_getaffinity", lambda tid: set(caller_mask))
    lease = utils.acquire_cpu_affinity(0)
    current[103] = numa_cpus

    utils.release_cpu_affinity(lease)

    assert current[101] == all_cpus
    assert current[103] == caller_mask


def test_affinity_is_restored_only_after_last_lease(affinity_leases):
    current, bindings, configure_calls, all_cpus, numa_cpus = affinity_leases
    first = utils.acquire_cpu_affinity(0)
    second = utils.acquire_cpu_affinity(0)

    assert configure_calls == [0]
    utils.release_cpu_affinity(first)
    assert current == {101: numa_cpus, 102: numa_cpus}
    assert bindings == []

    utils.release_cpu_affinity(second)
    assert current == {101: all_cpus, 102: frozenset({2, 3})}


def test_release_preserves_externally_changed_thread_mask(affinity_leases):
    current, _, _, all_cpus, _ = affinity_leases
    lease = utils.acquire_cpu_affinity(0)
    current[102] = frozenset({3})

    utils.release_cpu_affinity(lease)

    assert current == {101: all_cpus, 102: frozenset({3})}
    utils.logger.info.assert_any_call("Preserved externally changed CPU affinity on 1 threads.")


def test_affinity_lease_release_is_idempotent(affinity_leases):
    current, bindings, _, all_cpus, _ = affinity_leases
    lease = utils.acquire_cpu_affinity(0)

    utils.release_cpu_affinity(lease)
    first_bindings = list(bindings)
    utils.release_cpu_affinity(lease)

    assert current[101] == all_cpus
    assert bindings == first_bindings


def test_affinity_lease_lock_is_reentrant():
    with utils._cpu_affinity_lease_lock:
        assert utils._cpu_affinity_lease_lock.acquire(blocking=False)
        utils._cpu_affinity_lease_lock.release()


def test_acquire_skips_pinning_when_initial_snapshot_fails(affinity_leases, monkeypatch):
    _, _, configure_calls, _, _ = affinity_leases
    monkeypatch.setattr(utils, "_read_thread_affinities", lambda: {})

    assert utils.acquire_cpu_affinity(0) is None

    assert configure_calls == []
    assert utils._cpu_affinity_lease_state is None


def test_acquire_restores_masks_when_post_snapshot_fails(affinity_leases, monkeypatch):
    current, bindings, _, all_cpus, _ = affinity_leases
    snapshots = [dict(current), dict(current), {}]
    monkeypatch.setattr(utils, "_read_thread_affinities", Mock(side_effect=snapshots))

    assert utils.acquire_cpu_affinity(0) is None

    assert current == {101: all_cpus, 102: frozenset({2, 3})}
    assert bindings == [(101, all_cpus), (102, frozenset({2, 3}))]
    assert utils._cpu_affinity_lease_state is None


def test_release_restores_saved_masks_when_snapshot_fails(affinity_leases, monkeypatch):
    current, bindings, _, all_cpus, _ = affinity_leases
    lease = utils.acquire_cpu_affinity(0)
    bindings.clear()
    monkeypatch.setattr(utils, "_read_thread_affinities", lambda: {})

    utils.release_cpu_affinity(lease)

    assert current == {101: all_cpus, 102: frozenset({2, 3})}
    assert bindings == [(101, all_cpus), (102, frozenset({2, 3}))]
    assert utils._cpu_affinity_lease_state is None


@pytest.mark.parametrize("external_change", [False, True])
def test_release_psutil_fallback(affinity_leases, monkeypatch, external_change):
    current, _, _, all_cpus, _ = affinity_leases
    lease = utils.acquire_cpu_affinity(0)
    if external_change:
        current[101] = frozenset({3})

    class Process:
        def cpu_affinity(self, cpus=None):
            if cpus is not None:
                current[101] = frozenset(cpus)
            return list(current[101])

    monkeypatch.delattr(utils.os, "sched_setaffinity")
    monkeypatch.setattr(
        utils,
        "psutil",
        SimpleNamespace(Process=lambda *args: Process(), Error=RuntimeError),
    )

    utils.release_cpu_affinity(lease)

    assert current[101] == (frozenset({3}) if external_change else all_cpus)
    assert utils._cpu_affinity_lease_state is None


_REAL_LEASE_PROBE = """
import os
import threading

import psutil
import tensorrt_llm.llmapi.utils as utils

original = sorted(psutil.Process().cpu_affinity())
if len(original) < 2:
    print("SKIP: fewer than two available CPUs")
    raise SystemExit(0)
split = max(1, len(original) // 2)
numa_mask = original[:split]
existing_mask = original[split:] or original[-1:]
release = threading.Event()
existing_ready = threading.Event()
later_ready = threading.Event()


def existing_worker():
    os.sched_setaffinity(0, existing_mask)
    existing_ready.set()
    release.wait(30)


def later_worker():
    later_ready.set()
    release.wait(30)


existing = threading.Thread(target=existing_worker, daemon=True)
existing.start()
assert existing_ready.wait(10)
utils.get_numa_aware_cpu_affinity = lambda device_id: numa_mask
os.environ["TLLM_NUMA_AWARE_WORKER_AFFINITY"] = "1"
lease = utils.acquire_cpu_affinity(0)
del os.environ["TLLM_NUMA_AWARE_WORKER_AFFINITY"]
assert lease is not None
assert sorted(os.sched_getaffinity(existing.native_id)) == numa_mask

later = threading.Thread(target=later_worker, daemon=True)
later.start()
assert later_ready.wait(10)
assert sorted(os.sched_getaffinity(later.native_id)) == numa_mask

utils.release_cpu_affinity(lease)
assert sorted(os.sched_getaffinity(0)) == original
assert sorted(os.sched_getaffinity(existing.native_id)) == existing_mask
assert sorted(os.sched_getaffinity(later.native_id)) == original
print("OK: restored original and inherited thread masks")
release.set()
existing.join(10)
later.join(10)
"""


@pytest.mark.skipif(
    not hasattr(os, "sched_setaffinity") or not os.path.isdir("/proc/self/task"),
    reason="Linux procfs affinity only",
)
def test_affinity_lease_restores_real_thread_masks():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", _REAL_LEASE_PROBE], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    if result.stdout.startswith("SKIP:"):
        pytest.skip(result.stdout.strip())
    assert "OK:" in result.stdout, result.stdout
