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
"""Unit tests for the store host donor (host memory lent to a Mooncake pool).

Runs without Mooncake, without a GPU and without NUMA hardware: the store
client, the thread memory binding and the NUMA meminfo are injected.
"""

import json
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor import store_host_donor as shd

pytestmark = pytest.mark.cpu_only

GIB = 1 << 30
MIB = 1 << 20


def _tempfile_with_gettempdir(gettempdir):
    """A stand-in for the donor module's ``tempfile`` whose ``gettempdir`` is
    ``gettempdir``. Patching the global ``tempfile.gettempdir`` instead also
    redirects pytest's own capture files into a missing directory."""
    return SimpleNamespace(NamedTemporaryFile=tempfile.NamedTemporaryFile, gettempdir=gettempdir)


class FakeStore:
    """Records what a donor does with a Mooncake client."""

    instances = []

    def __init__(self, *, can_mount=True, setup_status=0, mount_result=None):
        self.calls = []
        self.setup_args = None
        self.setup_thread = None
        self.setup_status = setup_status
        self.mount_result = mount_result
        self.closed = False
        if not can_mount:
            self.allocate_and_mount_segment = None
        FakeStore.instances.append(self)

    def setup(self, *args):
        self.setup_args = args
        self.setup_thread = threading.current_thread().name
        self.calls.append(("setup", args[2]))
        return self.setup_status

    def get_hostname(self):
        return f"{self.setup_args[0]}:{12000 + len(FakeStore.instances)}"

    def allocate_and_mount_segment(self, size, protocol, location):
        self.calls.append(("mount", size, protocol, location))
        if self.mount_result is not None:
            return self.mount_result
        return {"ret": 0, "allocated_size": size, "segment_ids": [f"seg-{location}-{size}"]}

    def unmount_and_free_segment(self, ids, grace):
        self.calls.append(("unmount", list(ids), grace))
        return 0

    def close(self):
        self.calls.append(("close",))
        self.closed = True
        return 0


@pytest.fixture(autouse=True)
def reset(monkeypatch):
    FakeStore.instances = []
    monkeypatch.setattr(shd, "_ACTIVE", None)
    # No production node lock: a live donor on the test host must not block a test.
    monkeypatch.setattr(shd, "NODE_LOCK_PATH", None)
    # Tests size shares far beyond the test host: no real cgroup bound by default.
    monkeypatch.setattr(
        shd, "_default_job_memory", lambda: {"limit": None, "nonfile": None, "path": None}
    )
    monkeypatch.delenv(shd.DISABLE_ENV, raising=False)
    monkeypatch.delenv(shd.MASTER_ENV, raising=False)
    monkeypatch.delenv(shd.CONFIG_PATH_ENV, raising=False)
    yield
    shd._ACTIVE = None


@pytest.fixture
def logs(monkeypatch):
    lines = []
    for level in ("info", "warning", "error"):
        monkeypatch.setattr(
            shd.logger, level, lambda msg, _level=level: lines.append((_level, msg))
        )
    return lines


def records(lines, tag):
    prefix = f"{shd.LOG_TAG}_{tag} "
    return [
        json.loads(msg[len(prefix) :].split("; ")[0]) for _, msg in lines if msg.startswith(prefix)
    ]


def plenty(_node):
    return {"MemFree": 4096 * GIB, "FilePages": 0, "SReclaimable": 0}


def make_donor(shares, *, factory=None, bound=None, meminfo=plenty, **kwargs):
    bound = [] if bound is None else bound

    def bind(node):
        bound.append((node, threading.current_thread().name))

    return shd.StoreHostDonor(
        master="10.0.0.1:8700",
        shares=shares,
        rank=kwargs.pop("rank", 0),
        local_rank=0,
        local_size=4,
        device_name="mlx5_0",
        local_hostname="10.0.0.2",
        store_factory=factory or FakeStore,
        bind_thread=bind,
        meminfo=meminfo,
        **kwargs,
    )


# -- plan -------------------------------------------------------------------


def test_plan_four_ranks_two_sockets():
    owners = {rank: shd.plan_shares(512 * GIB, 64 * GIB, [0, 1], rank, 4) for rank in range(4)}
    assert owners[0] == [shd.DonorShare(0, 256 * GIB, 32 * GIB)]
    assert owners[2] == [shd.DonorShare(1, 256 * GIB, 32 * GIB)]
    assert owners[1] == [] and owners[3] == []


def test_plan_one_rank_owns_every_socket():
    shares = shd.plan_shares(2 * GIB, 0, [0, 1], 0, 1)
    assert [s.numa_node for s in shares] == [0, 1]
    assert all(s.startup_bytes == GIB and s.late_bytes == 0 for s in shares)


def test_plan_eight_ranks_and_two_ranks():
    assert [r for r in range(8) if shd.plan_shares(2, 2, [0, 1], r, 8)] == [0, 4]
    assert [r for r in range(2) if shd.plan_shares(2, 2, [0, 1], r, 2)] == [0, 1]


@pytest.mark.parametrize(
    "args",
    [
        (3, 0, [0, 1], 0, 4),
        (2, 1, [0, 1], 0, 4),
        (2, 2, [0, 0], 0, 4),
        (2, 2, [], 0, 4),
        (-2, 2, [0, 1], 0, 4),
        (2, 2, [0, 1], 4, 4),
    ],
)
def test_plan_rejects(args):
    with pytest.raises(ValueError):
        shd.plan_shares(*args)


# -- topology -----------------------------------------------------------------


def test_cpu_numa_nodes_skip_memory_only_and_disallowed(tmp_path):
    root = tmp_path / "node"
    for node, cpus in ((0, "0-71"), (1, "72-143"), (2, ""), (3, "\n"), (4, "144-150")):
        (root / f"node{node}").mkdir(parents=True)
        (root / f"node{node}" / "cpulist").write_text(cpus)
    (root / "possible").write_text("0-4")
    status = tmp_path / "status"
    status.write_text("Name:\tpython\nMems_allowed_list:\t0-3\n")
    assert shd.cpu_numa_nodes(str(root), str(status)) == [0, 1]
    status.write_text("Name:\tpython\n")
    assert shd.cpu_numa_nodes(str(root), str(status)) == [0, 1, 4]


def test_node_meminfo(tmp_path):
    (tmp_path / "node1").mkdir()
    (tmp_path / "node1" / "meminfo").write_text(
        "Node 1 MemTotal:       1000 kB\nNode 1 MemFree:         400 kB\n"
        "Node 1 FilePages:       100 kB\nNode 1 SReclaimable:     50 kB\n"
        "Node 1 HugePages_Total:     0\n"
    )
    info = shd.node_meminfo(1, str(tmp_path))
    assert info["MemFree"] == 400 * 1024 and "HugePages_Total" not in info
    assert shd.node_obtainable_bytes(info) == 550 * 1024
    assert shd.node_meminfo(7, str(tmp_path)) == {}
    assert shd.node_obtainable_bytes({}) is None
    # tmpfs / shared memory sits in FilePages but reclaim cannot free it.
    assert shd.node_obtainable_bytes(dict(info, Shmem=60 * 1024)) == 490 * 1024
    # Neither can it free mlocked (unevictable) file pages.
    assert (
        shd.node_obtainable_bytes(dict(info, Shmem=20 * 1024, Unevictable=30 * 1024)) == 500 * 1024
    )


def test_node_numa_disagreement():
    peers = [("a", [0, 1]), ("a", [0, 1]), ("b", [0]), ("a", None)]
    assert shd.node_numa_disagreement(peers, "a") is None
    assert shd.node_numa_disagreement(peers + [("a", [1])], "a").startswith(
        "store_host_donor: ranks on a"
    )


def test_headroom_refuses_when_page_cache_is_shared_memory():
    def shm_heavy(_node):
        return {"MemFree": 2 * GIB, "FilePages": 10 * GIB, "Shmem": 9 * GIB, "SReclaimable": 0}

    donor = make_donor([shd.DonorShare(0, 4 * GIB, 0)], meminfo=shm_heavy)
    donor.start_startup()
    with pytest.raises(RuntimeError):
        donor.wait_startup()
    assert FakeStore.instances == []


# -- master ---------------------------------------------------------------------


def test_resolve_master_order(tmp_path):
    config = tmp_path / "mooncake.json"
    config.write_text(json.dumps({"master_server_address": "10.0.0.9:50051"}))
    env = {shd.MASTER_ENV: "10.0.0.8:8700", shd.CONFIG_PATH_ENV: str(config)}
    assert shd.resolve_master("10.0.0.7:1", env) == "10.0.0.7:1"
    assert shd.resolve_master(None, env) == "10.0.0.8:8700"
    assert shd.resolve_master(None, {shd.CONFIG_PATH_ENV: str(config)}) == "10.0.0.9:50051"
    with pytest.raises(ValueError, match="needs the Mooncake master"):
        shd.resolve_master(None, {})
    with pytest.raises(ValueError, match="not host:port"):
        shd.resolve_master("10.0.0.7", {})


# -- lifecycle ------------------------------------------------------------------


def test_startup_then_late_same_client_then_retire(logs):
    bound = []
    donor = make_donor([shd.DonorShare(1, 256 * GIB, 32 * GIB)], bound=bound)
    donor.start_startup()
    donor.register_late()
    (store,) = FakeStore.instances
    assert store.setup_args == (
        "10.0.0.2",
        "P2PHANDSHAKE",
        256 * GIB,
        GIB,
        "rdma",
        "mlx5_0",
        "10.0.0.1:8700",
    )
    assert store.calls[1] == ("mount", 32 * GIB, "rdma", "cpu:1")
    # Every registration ran on its own thread bound to the share's node.
    assert [node for node, _ in bound] == [1, 1]
    assert store.setup_thread == bound[0][1] != threading.current_thread().name
    registered = records(logs, "REGISTERED")
    assert [(r["phase"], r["numa_node"], r["bytes"], r["client"]) for r in registered] == [
        ("startup", 1, 256 * GIB, "new"),
        ("late", 1, 32 * GIB, "same"),
    ]
    assert registered[1]["segment_ids"] == [f"seg-cpu:1-{32 * GIB}"]
    assert donor.summary()["registered_bytes"] == 288 * GIB

    assert donor.close()
    # Late segment unmounted before the client carrying it closes.
    assert store.calls[2:] == [("unmount", [f"seg-cpu:1-{32 * GIB}"], 0), ("close",)]
    assert [(r["phase"], r["ok"]) for r in records(logs, "RETIRED")] == [
        ("late", True),
        ("startup", True),
    ]
    assert donor.close()  # idempotent
    assert len(store.calls) == 4


def test_failed_late_unmount_keeps_parent_client_until_retry(logs):
    donor = make_donor([shd.DonorShare(0, 8 * GIB, 2 * GIB)])
    donor.start_startup()
    donor.register_late()
    (store,) = FakeStore.instances
    results = iter([-1, 0])
    store.unmount_and_free_segment = lambda ids, grace: (
        store.calls.append(("unmount", list(ids))),
        next(results),
    )[1]
    assert not donor.close()
    # The late segment failed to unmount, so the client carrying it stays open.
    assert not store.closed
    assert donor.summary()["registered_bytes"] == 10 * GIB
    assert [r["phase"] for r in records(logs, "RETIRE_FAILED")] == ["late"]
    # A later close retries the segment, then closes the client.
    assert donor.close()
    assert store.closed and donor.summary()["registered_bytes"] == 0
    assert [c[0] for c in store.calls[-3:]] == ["unmount", "unmount", "close"]


def test_late_new_client(logs):
    donor = make_donor([shd.DonorShare(0, 256 * GIB, 32 * GIB)], late_mode="new_client")
    donor.start_startup()
    donor.register_late()
    first, second = FakeStore.instances
    assert first.setup_args[2] == 256 * GIB and second.setup_args[2] == 32 * GIB
    assert donor.close() and first.closed and second.closed


def test_auto_falls_back_to_new_client_without_mount_api():
    donor = make_donor(
        [shd.DonorShare(0, 8 * GIB, 2 * GIB)], factory=lambda: FakeStore(can_mount=False)
    )
    donor.start_startup()
    donor.register_late()
    assert [s.setup_args[2] for s in FakeStore.instances] == [8 * GIB, 2 * GIB]
    assert donor.close()


def test_same_client_with_zero_startup_opens_empty_client():
    donor = make_donor([shd.DonorShare(0, 0, 2 * GIB)], late_mode="same_client")
    donor.start_startup()
    donor.wait_startup()
    (store,) = FakeStore.instances
    assert store.setup_args[2] == 0
    donor.register_late()
    assert store.calls[-1] == ("mount", 2 * GIB, "rdma", "cpu:0")
    assert donor.close()


def test_auto_late_only_uses_new_client():
    donor = make_donor([shd.DonorShare(0, 0, 2 * GIB)])
    donor.start_startup()
    donor.wait_startup()
    assert FakeStore.instances == []
    donor.register_late()
    (store,) = FakeStore.instances
    assert store.setup_args[2] == 2 * GIB
    assert donor.close()


def test_rank_without_share_does_nothing():
    donor = make_donor([])
    donor.start_startup()
    donor.register_late()
    donor.schedule_late()
    assert FakeStore.instances == [] and donor.close()


def test_setup_failure_surfaces_at_wait(logs):
    donor = make_donor([shd.DonorShare(0, 8 * GIB, 0)], factory=lambda: FakeStore(setup_status=-1))
    donor.start_startup()
    with pytest.raises(RuntimeError, match="startup registration failed"):
        donor.wait_startup()
    assert records(logs, "REGISTERED") == []
    assert donor.close()


def test_bad_mount_receipt_is_retired_and_raised(logs):
    bad = {"ret": 0, "allocated_size": 1, "segment_ids": ["seg-x"]}
    donor = make_donor(
        [shd.DonorShare(0, 8 * GIB, 2 * GIB)], factory=lambda: FakeStore(mount_result=bad)
    )
    donor.start_startup()
    with pytest.raises(RuntimeError, match="allocate_and_mount_segment failed"):
        donor.register_late()
    (store,) = FakeStore.instances
    assert ("unmount", ["seg-x"], 0) in store.calls
    assert donor.close() and store.closed


def test_headroom_refuses_before_allocating():
    def tight(_node):
        return {"MemFree": 8 * GIB, "FilePages": GIB // 2, "SReclaimable": 0}

    donor = make_donor([shd.DonorShare(0, 8 * GIB, 0)], meminfo=tight)
    donor.start_startup()
    with pytest.raises(RuntimeError):
        donor.wait_startup()
    assert FakeStore.instances == []


def test_headroom_counts_the_client_transfer_buffer():
    def twelve(_node):
        return {"MemFree": 12 * GIB, "FilePages": 0, "SReclaimable": 0}

    # 8 GiB share + 16 GiB transfer buffer + 1 GiB headroom > 12 GiB on the node.
    donor = make_donor([shd.DonorShare(0, 8 * GIB, 0)], meminfo=twelve, local_buffer_size=16 * GIB)
    donor.start_startup()
    with pytest.raises(RuntimeError) as failure:
        donor.wait_startup()
    assert "transfer buffer" in str(failure.value.__cause__ or failure.value)
    assert FakeStore.instances == []


def test_cgroup_room_refuses_fixed_startup_before_allocating():
    job = {"limit": 10 * GIB, "nonfile": 3 * GIB, "path": "/job"}
    # 8 GiB share + 1 GiB local buffer > 7 GiB room, with ample NUMA memory.
    donor = make_donor([shd.DonorShare(0, 8 * GIB, 0)], job_memory=lambda: job)
    donor.start_startup()
    with pytest.raises(RuntimeError) as failure:
        donor.wait_startup()
    assert "memory cgroup" in str(failure.value.__cause__ or failure.value)
    assert FakeStore.instances == []
    job["nonfile"] = GIB // 2
    donor = make_donor([shd.DonorShare(0, 8 * GIB, 0)], job_memory=lambda: job)
    donor.start_startup()
    donor.wait_startup()
    assert len(FakeStore.instances) == 1 and donor.close()


def test_cgroup_limit_with_unknown_usage_fails_closed():
    job = {"limit": 1000 * GIB, "nonfile": None, "path": "/job"}
    donor = make_donor([shd.DonorShare(0, 8 * GIB, 0)], job_memory=lambda: job)
    donor.start_startup()
    with pytest.raises(RuntimeError) as failure:
        donor.wait_startup()
    assert "unknown held" in str(failure.value.__cause__ or failure.value)
    assert FakeStore.instances == []


def test_no_node_lock_refuses_before_allocating(tmp_path, monkeypatch):
    blocked = tmp_path / "missing" / "donor.lock"
    monkeypatch.setattr(
        shd, "tempfile", _tempfile_with_gettempdir(lambda: str(tmp_path / "also-missing"))
    )
    donor = make_donor([shd.DonorShare(0, 2 * GIB, 0)], node_lock=str(blocked))
    donor.start_startup()
    with pytest.raises(RuntimeError) as failure:
        donor.wait_startup()
    assert "node-local lock" in str(failure.value.__cause__ or failure.value)
    assert FakeStore.instances == []


def test_cgroup_room_refuses_fixed_late_mount(tmp_path):
    job = {"limit": 10 * GIB, "nonfile": 4 * GIB, "path": "/job"}
    lock = tmp_path / "donor.lock"
    donor = make_donor(
        [shd.DonorShare(0, 2 * GIB, 8 * GIB)], job_memory=lambda: job, node_lock=str(lock)
    )
    donor.start_startup()
    donor.wait_startup()
    (store,) = FakeStore.instances
    with pytest.raises(RuntimeError, match="memory cgroup"):
        donor.register_late()
    assert [c for c in store.calls if c[0] == "mount"] == []
    assert lock.exists()  # registrations ran under the node lock
    assert donor.close()


def test_bind_failure_never_reaches_mooncake():
    def broken_bind(_node):
        raise OSError(22, "Invalid argument")

    donor = shd.StoreHostDonor(
        master="h:1",
        shares=[shd.DonorShare(0, GIB, 0)],
        rank=0,
        local_rank=0,
        local_size=1,
        local_hostname="h",
        store_factory=FakeStore,
        bind_thread=broken_bind,
        meminfo=plenty,
    )
    donor.start_startup()
    with pytest.raises(RuntimeError):
        donor.wait_startup()
    assert FakeStore.instances == []


def test_delayed_late_registers_in_background():
    donor = make_donor([shd.DonorShare(0, GIB, GIB)], late_delay_seconds=0.05)
    donor.start_startup()
    donor.schedule_late()
    deadline = time.monotonic() + 5
    while donor.summary()["registered_bytes"] < 2 * GIB and time.monotonic() < deadline:
        time.sleep(0.01)
    assert donor.summary()["registered_bytes"] == 2 * GIB
    assert donor.close()


def test_close_during_delay_skips_late():
    donor = make_donor([shd.DonorShare(0, GIB, GIB)], late_delay_seconds=30)
    donor.start_startup()
    donor.schedule_late()
    started = time.monotonic()
    assert donor.close()
    assert time.monotonic() - started < 5
    (store,) = FakeStore.instances
    assert [c[0] for c in store.calls] == ["setup", "close"]


def test_registered_line_reports_obtainable_drop(logs):
    """Page cache reclaimed to back the segment shows in the obtainable drop, not in MemFree."""
    state = {"MemFree": 600 * GIB, "FilePages": 100 * GIB, "SReclaimable": 0}

    def meminfo(_node):
        return dict(state)

    class ReclaimingStore(FakeStore):
        def setup(self, *args):
            # 48 GiB allocated: 40 from free memory, 8 by reclaiming page cache.
            state["MemFree"] -= 40 * GIB
            state["FilePages"] -= 8 * GIB
            return super().setup(*args)

    donor = make_donor([shd.DonorShare(0, 48 * GIB, 0)], factory=ReclaimingStore, meminfo=meminfo)
    donor.start_startup()
    donor.wait_startup()
    (reg,) = records(logs, "REGISTERED")
    assert reg["node_memfree_drop_gib"] == 40.0 and reg["node_obtainable_drop_gib"] == 48.0
    assert donor.close()


# -- process-wide entry point -----------------------------------------------------


def config(**fields):
    base = dict(
        master=None,
        protocol="rdma",
        device_name="mlx5_0",
        startup_size_per_node="512GiB",
        late_size_per_node="64GiB",
        numa_nodes=None,
        local_buffer_size="1GiB",
        local_hostname="10.0.0.2",
        late_mode="auto",
        late_delay_seconds=0.0,
        late_auto=None,
    )
    base.update(fields)
    return SimpleNamespace(**base)


def test_start_entry_point(monkeypatch, logs):
    monkeypatch.setenv(shd.MASTER_ENV, "10.0.0.1:8700")
    donor = shd.start_store_host_donor(
        config(),
        rank=6,
        local_rank=2,
        local_size=4,
        numa_nodes=[0, 1],
        store_factory=FakeStore,
        bind_thread=lambda node: None,
        meminfo=plenty,
    )
    assert shd.active_donor() is donor
    assert donor.shares == [shd.DonorShare(1, 256 * GIB, 32 * GIB)]
    (plan,) = records(logs, "PLAN")
    assert plan["owned"][0]["numa_node"] == 1 and plan["master"] == "10.0.0.1:8700"
    assert plan["late_per_node"] == 64 * GIB and plan["late_auto"] is None
    donor.register_late()
    with pytest.raises(RuntimeError, match="already active"):
        shd.start_store_host_donor(config(), rank=6, local_rank=2, local_size=4, numa_nodes=[0, 1])
    assert shd.close_active_donor() and shd.active_donor() is None
    assert shd.close_active_donor()


def test_start_entry_point_off(monkeypatch):
    assert shd.start_store_host_donor(None, rank=0, local_rank=0, local_size=1) is None
    monkeypatch.setenv(shd.DISABLE_ENV, "1")
    assert (
        shd.start_store_host_donor(config(master="h:1"), rank=0, local_rank=0, local_size=1) is None
    )
    assert shd.active_donor() is None


def test_start_entry_point_explicit_numa_nodes():
    donor = shd.start_store_host_donor(
        config(master="h:1", numa_nodes=[1, 0]),
        rank=0,
        local_rank=0,
        local_size=4,
        numa_nodes=[0, 1],
        store_factory=FakeStore,
        bind_thread=lambda node: None,
        meminfo=plenty,
    )
    assert [s.numa_node for s in donor.shares] == [1]
    assert shd.close_active_donor()


def test_start_entry_point_auto_needs_late_auto():
    with pytest.raises(ValueError, match="needs late_auto"):
        shd.start_store_host_donor(
            config(master="h:1", late_size_per_node="auto"),
            rank=0,
            local_rank=0,
            local_size=1,
            numa_nodes=[0],
        )
    assert shd.active_donor() is None


def test_concurrent_starts_create_one_donor(monkeypatch):
    monkeypatch.setenv(shd.MASTER_ENV, "10.0.0.1:8700")
    entered, release = threading.Event(), threading.Event()
    real_plan = shd.plan_shares

    def slow_plan(*args):
        entered.set()
        release.wait(timeout=2)  # the first caller is still building its donor
        return real_plan(*args)

    monkeypatch.setattr(shd, "plan_shares", slow_plan)
    results = []

    def start():
        try:
            results.append(
                shd.start_store_host_donor(
                    config(),
                    rank=0,
                    local_rank=0,
                    local_size=1,
                    numa_nodes=[0],
                    store_factory=FakeStore,
                    bind_thread=lambda node: None,
                    meminfo=plenty,
                )
            )
        except RuntimeError as error:
            results.append(error)

    first = threading.Thread(target=start, daemon=True)
    second = threading.Thread(target=start, daemon=True)
    first.start()
    try:
        assert entered.wait(timeout=2)
        second.start()
        second.join(timeout=1.5)
    finally:
        release.set()
        first.join(timeout=10)
        if second.is_alive() or second.ident is not None:
            second.join(timeout=10)
    assert not first.is_alive() and not second.is_alive()
    donors = [r for r in results if isinstance(r, shd.StoreHostDonor)]
    assert len(donors) == 1 and shd.active_donor() is donors[0]
    assert any(isinstance(r, RuntimeError) and "already active" in str(r) for r in results)
    assert shd.close_active_donor() and shd.active_donor() is None


def test_failed_start_releases_the_slot():
    with pytest.raises(ValueError, match="needs the Mooncake master"):
        shd.start_store_host_donor(config(), rank=0, local_rank=0, local_size=1, numa_nodes=[0])
    assert shd.active_donor() is None and shd._ACTIVE is None


# -- late size from a planned host budget (late_size_per_node: auto) ---------------
#
# The budget below gives a GPU on a non-head node 256 - 192 - 24 - 2.5 =
# 37.5 GiB and a GPU on the head node 256 - 192 - 61 - 2.5 = 0.5 GiB.


def auto_budget(**fields):
    base = dict(
        host_budget_per_gpu=256 * GIB,
        store_per_gpu=192 * GIB,
        process_peak_per_gpu=24 * GIB,
        head_process_peak_per_gpu=61 * GIB,
        margin_per_gpu=2560 * MIB,
        granularity=GIB,
    )
    base.update(fields)
    return shd.LateAuto(**base)


AMPLE = {"limit": 2000 * GIB, "nonfile": 100 * GIB}


def auto_donor(
    head,
    job=None,
    local_rank=0,
    late_mode="auto",
    startup_per_node=0,
    meminfo=plenty,
    head_role=None,
    **auto_fields,
):
    nodes = [0, 1]
    shares = shd.plan_shares(0, 0, nodes, local_rank, 2)
    return shd.StoreHostDonor(
        master="10.0.0.1:8700",
        shares=shares,
        rank=local_rank,
        local_rank=local_rank,
        local_size=2,
        device_name="mlx5_0",
        local_hostname="10.0.0.2",
        late_auto=auto_budget(**auto_fields),
        numa_nodes=nodes,
        head_role=head_role or (lambda master, hosts: (head, "test")),
        job_memory=job if callable(job) else (lambda: AMPLE if job is None else job),
        late_mode=late_mode,
        startup_per_node=startup_per_node,
        store_factory=FakeStore,
        bind_thread=lambda node: None,
        meminfo=meminfo,
    )


def socket_one_has(gib):
    def meminfo(node):
        free = gib * GIB if node == 1 else 4096 * GIB
        return {"MemFree": free, "FilePages": 0, "SReclaimable": 0}

    return meminfo


def test_late_auto_per_gpu_formula():
    auto = auto_budget()
    assert shd.late_auto_per_gpu(auto, head=False) == 256 * GIB - 192 * GIB - 24 * GIB - 2560 * MIB
    assert shd.late_auto_per_gpu(auto, head=False) == int(37.5 * GIB)
    assert shd.late_auto_per_gpu(auto, head=True) == GIB // 2
    assert shd.late_auto_per_gpu(auto_budget(store_per_gpu=220 * GIB), head=True) == 0


def test_head_node_role():
    def route(_master):
        return "10.0.0.5"

    def resolve(host):
        return {
            "10.0.0.5": ["10.0.0.5"],
            "10.0.0.9": ["10.0.0.9"],
            "localhost": ["127.0.0.1"],
        }[host]

    assert shd.head_node_role("10.0.0.5:8700", route_to=route, resolve=resolve)[0] is True
    assert shd.head_node_role("10.0.0.9:8700", route_to=route, resolve=resolve)[0] is False
    assert shd.head_node_role("localhost:8700", route_to=route, resolve=resolve)[0] is True

    def broken(host):
        raise OSError("no route")

    head, basis = shd.head_node_role("10.0.0.9:8700", route_to=route, resolve=broken)
    assert head is True and "sized as head" in basis
    assert shd.head_node_role("10.0.0.9:8700", ["host-a"], hostname="host-a.example")[0] is True
    assert shd.head_node_role("10.0.0.9:8700", ["host-a.example"], hostname="host-b")[0] is False


def test_cgroup_memory_binding_limit(tmp_path):
    proc = tmp_path / "cgroup"
    proc.write_text("0::/jobs/job7/step0/task0\n")
    root = tmp_path / "fs"
    for rel, limit in (
        ("", None),
        ("jobs", "max"),
        ("jobs/job7", str(1000 * GIB)),
        ("jobs/job7/step0", str(2000 * GIB)),
        ("jobs/job7/step0/task0", "max"),
    ):
        d = root / rel
        d.mkdir(parents=True, exist_ok=True)
        if limit is not None:
            (d / "memory.max").write_text(limit + "\n")
    (root / "jobs/job7/memory.current").write_text(str(700 * GIB))
    (root / "jobs/job7/memory.stat").write_text(
        f"anon {500 * GIB}\nfile {200 * GIB}\nshmem {30 * GIB}\n"
    )
    (root / "jobs/job7/step0/memory.current").write_text(str(600 * GIB))
    (root / "jobs/job7/step0/memory.stat").write_text(f"file {200 * GIB}\nshmem 0\n")
    out = shd.cgroup_memory(str(proc), str(root))
    # shmem sits inside "file" but reclaim cannot free it: 700 - 200 + 30.
    assert out["limit"] == 1000 * GIB and out["nonfile"] == 530 * GIB
    assert out["path"].endswith("jobs/job7")
    assert shd.cgroup_memory(str(tmp_path / "missing"), str(root))["limit"] is None
    # A parent with a larger limit but less room left binds.
    (root / "jobs/memory.max").write_text(str(2000 * GIB))
    (root / "jobs/memory.current").write_text(str(1950 * GIB))
    (root / "jobs/memory.stat").write_text(f"file {100 * GIB}\nshmem {50 * GIB}\n")
    out = shd.cgroup_memory(str(proc), str(root))
    assert out["limit"] == 2000 * GIB and out["nonfile"] == 1900 * GIB
    assert out["path"].endswith("jobs")
    (root / "jobs/memory.max").write_text("max\n")
    # Usage unknown at any limited level: no room is assumed.
    (root / "jobs/job7/step0/memory.stat").write_text(f"file {200 * GIB}\n")
    out = shd.cgroup_memory(str(proc), str(root))
    assert out["limit"] == 1000 * GIB and out["nonfile"] is None
    (root / "jobs/job7/step0/memory.stat").write_text(f"file {200 * GIB}\nshmem 0\n")
    (root / "jobs/job7/memory.stat").write_text(f"anon {500 * GIB}\nfile {200 * GIB}\n")
    assert shd.cgroup_memory(str(proc), str(root))["nonfile"] is None
    (root / "jobs/job7/memory.max").write_text("max\n")
    (root / "jobs/job7/step0/memory.max").write_text("max\n")
    assert shd.cgroup_memory(str(proc), str(root))["limit"] is None
    # Unevictable (mlocked) file pages are not reclaimable either: 700 - (200 - 30 - 160).
    (root / "jobs/job7/memory.max").write_text(str(1000 * GIB))
    (root / "jobs/job7/memory.stat").write_text(
        f"anon {500 * GIB}\nfile {200 * GIB}\nshmem {30 * GIB}\nunevictable {160 * GIB}\n"
    )
    assert shd.cgroup_memory(str(proc), str(root))["nonfile"] == 690 * GIB


def test_cgroup_memory_reports_own_usage_without_a_limit(tmp_path):
    proc = tmp_path / "cgroup"
    proc.write_text("0::/jobs/job7\n")
    root = tmp_path / "fs"
    (root / "jobs/job7").mkdir(parents=True)
    for rel in ("jobs", "jobs/job7"):
        (root / rel / "memory.max").write_text("max\n")
    (root / "jobs/job7/memory.current").write_text(str(300 * GIB))
    (root / "jobs/job7/memory.stat").write_text(f"file {100 * GIB}\nshmem {10 * GIB}\n")
    out = shd.cgroup_memory(str(proc), str(root))
    assert out["limit"] is None and out["nonfile"] is None
    assert out["usage"] == 210 * GIB


def test_auto_late_other_node_registers_per_gpu_share(logs):
    donor = auto_donor(head=False)
    assert donor.has_late
    donor.start_startup()
    donor.register_late()
    # 37.5 GiB per GPU x 2 GPUs = 75 GiB, rounded down to 2 x 1 GiB -> 37 GiB per socket.
    (rec,) = records(logs, "LATE_AUTO")
    assert rec["head"] is False and rec["worker_late"] == 74 * GIB
    assert rec["owned"] == {"0": 37 * GIB} and rec["capped_by_job_limit"] is False
    # The share mounts at cpu:0 on an empty client, so it cannot spill to socket 1.
    (store,) = FakeStore.instances
    assert store.setup_args[2] == 0
    assert [c for c in store.calls if c[0] == "mount"] == [("mount", 37 * GIB, "rdma", "cpu:0")]
    late = [r for r in records(logs, "REGISTERED") if r["phase"] == "late"]
    assert [(r["numa_node"], r["bytes"], r["client"]) for r in late] == [(0, 37 * GIB, "same")]
    assert donor.close()
    assert len(records(logs, "RETIRED")) == 2


def test_auto_late_same_client_opens_startup_client_with_zero_startup(logs):
    donor = auto_donor(head=False, late_mode="same_client")
    donor.start_startup()
    donor.wait_startup()
    # The size is unknown until after warmup, so the empty client opens up front.
    (store,) = FakeStore.instances
    assert store.setup_args[2] == 0
    donor.register_late()
    assert store.calls[-1] == ("mount", 37 * GIB, "rdma", "cpu:0")
    (reg,) = [r for r in records(logs, "REGISTERED") if r["phase"] == "late"]
    assert reg["client"] == "same"
    assert donor.close()


def test_auto_late_leaves_room_for_the_startup_share(logs):
    # 37.5 GiB per GPU minus a 20 GiB node startup share over 2 GPUs = 27.5 GiB per GPU.
    donor = auto_donor(head=False, startup_per_node=20 * GIB)
    donor.register_late()
    (rec,) = records(logs, "LATE_AUTO")
    assert rec["startup_per_gpu"] == 10 * GIB
    assert rec["worker_late"] == 54 * GIB and rec["owned"] == {"0": 27 * GIB}
    assert donor.close()


def test_auto_late_head_node_rounds_to_nothing(logs):
    donor = auto_donor(head=True, local_rank=1)
    donor.start_startup()
    donor.register_late()
    (rec,) = records(logs, "LATE_AUTO")
    assert rec["head"] is True and rec["worker_late"] == 0 and rec["owned"] == {"1": 0}
    # 2 x 0.5 GiB rounds to nothing, so no empty client was opened for it.
    assert records(logs, "REGISTERED") == [] and FakeStore.instances == []
    assert not donor.has_late
    assert donor.close()


def test_auto_late_capped_by_job_limit(logs):
    # 1200 GiB limit, 1130 GiB held: room = 70 - 2 x (24 + 2.5) = 17 GiB -> 16 GiB.
    job = {"limit": 1200 * GIB, "nonfile": 1130 * GIB}
    donor = auto_donor(head=False, job=job)
    donor.register_late()
    (rec,) = records(logs, "LATE_AUTO")
    assert rec["capped_by_job_limit"] is True and rec["worker_late"] == 16 * GIB
    assert rec["owned"] == {"0": 8 * GIB}
    logs.clear()
    auto_donor(head=False, job=AMPLE).register_late()
    assert records(logs, "LATE_AUTO")[0]["worker_late"] == 74 * GIB


def test_auto_late_shrinks_to_a_short_numa_node(logs):
    # The even split gives socket 1 37 GiB, but it holds only 20 GiB: lend
    # 20 - 1 (headroom) - 1 x 2.5 (margin of its one rank) = 16.5 -> 16 GiB
    # instead of refusing.
    donor = auto_donor(head=False, local_rank=1, meminfo=socket_one_has(20))
    donor.start_startup()
    donor.register_late()
    (rec,) = records(logs, "LATE_AUTO")
    assert rec["owned"] == {"1": 37 * GIB}
    (clamp,) = records(logs, "LATE_CLAMPED")
    assert (clamp["numa_node"], clamp["planned"], clamp["bytes"]) == (1, 37 * GIB, 16 * GIB)
    late = [r for r in records(logs, "REGISTERED") if r["phase"] == "late"]
    assert [(r["numa_node"], r["bytes"], r["client"]) for r in late] == [(1, 16 * GIB, "same")]
    (store,) = FakeStore.instances
    assert [c for c in store.calls if c[0] == "mount"] == [("mount", 16 * GIB, "rdma", "cpu:1")]
    assert donor.close()


def test_auto_late_skips_a_numa_node_with_nothing_to_lend(logs):
    donor = auto_donor(head=False, local_rank=1, meminfo=socket_one_has(2.5))
    donor.start_startup()
    donor.register_late()
    (clamp,) = records(logs, "LATE_CLAMPED")
    assert clamp["bytes"] == 0
    assert [r for r in records(logs, "REGISTERED") if r["phase"] == "late"] == []
    (store,) = FakeStore.instances
    assert [c for c in store.calls if c[0] == "mount"] == []
    assert donor.close()


def test_auto_late_shrinks_to_cgroup_room_at_registration(logs):
    # Room is ample when the size is resolved; others fill the job before this
    # rank registers. The configured 1000 GiB limit binds below the observed
    # 2000 GiB, and the peak reserve 2 x (24 + 2.5) GiB stays held:
    # 1000 - 915 - 53 = 32 GiB (the mount adds no transfer buffer).
    holder = []

    def job():
        planned = holder and holder[0].late_auto_record is not None
        return {"limit": 2000 * GIB, "nonfile": 915 * GIB} if planned else AMPLE

    donor = auto_donor(head=False, job=job, job_memory_limit=1000 * GIB)
    holder.append(donor)
    donor.start_startup()
    donor.register_late()
    (clamp,) = records(logs, "LATE_CLAMPED")
    assert (clamp["planned"], clamp["bytes"]) == (37 * GIB, 32 * GIB)
    late = [r for r in records(logs, "REGISTERED") if r["phase"] == "late"]
    assert [r["bytes"] for r in late] == [32 * GIB]
    assert donor.close()


def test_auto_late_same_client_mount_shrinks_without_a_second_client(logs):
    # The late segment mounts on the startup client, so no transfer buffer is
    # added: 20 - 1 (headroom) - 2.5 (margin) = 16.5 -> 16 GiB of the planned 37 GiB.
    donor = auto_donor(
        head=False, local_rank=1, late_mode="same_client", meminfo=socket_one_has(20)
    )
    donor.start_startup()
    donor.register_late()
    (clamp,) = records(logs, "LATE_CLAMPED")
    assert (clamp["planned"], clamp["bytes"]) == (37 * GIB, 16 * GIB)
    (store,) = FakeStore.instances
    assert [c for c in store.calls if c[0] == "mount"] == [("mount", 16 * GIB, "rdma", "cpu:1")]
    late = [r for r in records(logs, "REGISTERED") if r["phase"] == "late"]
    assert [(r["bytes"], r["client"]) for r in late] == [(16 * GIB, "same")]
    assert donor.close()


def test_fixed_late_share_is_still_refused_on_a_short_numa_node():
    donor = make_donor([shd.DonorShare(1, 0, 35 * GIB)], meminfo=socket_one_has(20))
    donor.start_startup()
    with pytest.raises(RuntimeError, match="NUMA node 1"):
        donor.register_late()
    assert FakeStore.instances == []


@pytest.mark.parametrize("late_mode", ["auto", "same_client"])
def test_auto_late_head_role_is_decided_once(logs, late_mode):
    # A lookup that fails at startup sizes the node as head (nothing to lend);
    # if it recovers by the late phase, the late size must not change under the
    # startup decision: no new client, no missing-client error.
    answers = [(True, "lookup failed: sized as head"), (False, "master is remote")]
    calls = []

    def role(master, hosts):
        calls.append(None)
        return answers[min(len(calls), len(answers)) - 1]

    donor = auto_donor(head=None, local_rank=0, late_mode=late_mode, head_role=role)
    donor.start_startup()
    donor.register_late()
    (rec,) = records(logs, "LATE_AUTO")
    assert rec["head"] is True and rec["worker_late"] == 0
    assert len(calls) == 1
    assert FakeStore.instances == [] and records(logs, "REGISTERED") == []
    assert donor.close()


@pytest.mark.parametrize(
    "job",
    [
        {"limit": None, "nonfile": None},
        {"limit": 2000 * GIB, "nonfile": None},
        {"limit": None, "nonfile": 100 * GIB},
    ],
)
def test_auto_late_unknown_job_bound_lends_nothing(logs, job):
    donor = auto_donor(head=False, job=job)
    donor.register_late()
    (rec,) = records(logs, "LATE_AUTO")
    assert (
        rec["job_bound_known"] is False and rec["worker_late"] == 0 and rec["capped_by_job_limit"]
    )
    assert records(logs, "REGISTERED") == []


def test_auto_late_explicit_job_limit_only_tightens(logs):
    job = {"limit": 2000 * GIB, "nonfile": 1200 * GIB}
    auto_donor(head=False, job=job, job_memory_limit=1280 * GIB).register_late()
    assert records(logs, "LATE_AUTO")[0]["job_limit"] == 1280 * GIB
    logs.clear()
    # A configured limit above the observed cgroup cap never enlarges it.
    job = {"limit": 1280 * GIB, "nonfile": 1270 * GIB}
    auto_donor(head=False, job=job, job_memory_limit=2048 * GIB).register_late()
    rec = records(logs, "LATE_AUTO")[0]
    assert rec["job_limit"] == 1280 * GIB and rec["worker_late"] == 0


def test_auto_late_explicit_job_limit_without_a_cgroup_limit(logs):
    # No cgroup is limited: the configured limit is checked against the
    # process's own cgroup usage.
    job = {"limit": None, "nonfile": None, "usage": 100 * GIB}
    auto_donor(head=False, job=job, job_memory_limit=1280 * GIB).register_late()
    rec = records(logs, "LATE_AUTO")[0]
    assert rec["job_bound_known"] is True and rec["job_limit"] == 1280 * GIB
    assert rec["worker_late"] > 0
    logs.clear()
    # Without a configured limit there is still no bound: nothing is lent.
    auto_donor(head=False, job=job).register_late()
    rec = records(logs, "LATE_AUTO")[0]
    assert rec["job_bound_known"] is False and rec["worker_late"] == 0


def auto_config(**fields):
    base = dict(
        host_budget_per_gpu="256GiB",
        store_per_gpu="192GiB",
        process_peak_per_gpu="24GiB",
        head_process_peak_per_gpu="61GiB",
        margin_per_gpu="2560MiB",
        granularity="1GiB",
        head_hosts=None,
        job_memory_limit=None,
    )
    base.update(fields)
    return SimpleNamespace(**base)


def test_start_entry_point_auto(monkeypatch, tmp_path, logs):
    monkeypatch.setenv(shd.MASTER_ENV, "10.0.0.1:8700")
    cfg = config(startup_size_per_node=0, late_size_per_node="auto", late_auto=auto_config())
    kwargs = dict(
        rank=0,
        local_rank=0,
        local_size=2,
        numa_nodes=[0, 1],
        store_factory=FakeStore,
        bind_thread=lambda node: None,
        meminfo=plenty,
        head_role=lambda master, hosts: (False, "test"),
        job_memory=lambda: AMPLE,
    )
    path = tmp_path / "mooncake.json"
    path.write_text(
        json.dumps({"master_server_address": "10.0.0.1:8700", "global_segment_size": 200 * GIB})
    )
    monkeypatch.setenv(shd.CONFIG_PATH_ENV, str(path))
    with pytest.raises(ValueError, match="would not hold"):
        shd.start_store_host_donor(cfg, **kwargs)
    path.write_text(json.dumps({"global_segment_size": 192 * GIB}))
    donor = shd.start_store_host_donor(cfg, **kwargs)
    (plan,) = records(logs, "PLAN")
    assert plan["late_per_node"] == "auto" and plan["late_auto"]["store_per_gpu"] == 192 * GIB
    assert donor.shares == [shd.DonorShare(0, 0, 0)]
    donor.register_late()
    assert donor.shares == [shd.DonorShare(0, 0, 37 * GIB)]
    assert shd.close_active_donor()


# -- LLM args (need the full package) ------------------------------------------------

LATE_AUTO = dict(
    host_budget_per_gpu="256GiB",
    store_per_gpu="192GiB",
    process_peak_per_gpu="24GiB",
    head_process_peak_per_gpu="61GiB",
)


def test_llm_args_config():
    pytest.importorskip("torch")
    from tensorrt_llm.llmapi.llm_args import StoreHostDonorConfig

    parsed = StoreHostDonorConfig(
        startup_size_per_node="512GiB", late_size_per_node=64 * GIB, late_delay_seconds=30
    )
    assert parsed.protocol == "rdma" and parsed.late_mode == "auto"
    for bad in (
        dict(),
        dict(startup_size_per_node="512GB"),
        dict(startup_size_per_node="1GiB", numa_nodes=[0, 0]),
        dict(startup_size_per_node="1GiB", numa_nodes=[]),
        dict(startup_size_per_node="1GiB", local_buffer_size=0),
        dict(startup_size_per_node="1GiB", late_mode="other"),
        dict(startup_size_per_node="1GiB", late_delay_seconds=-1),
        dict(startup_size_per_node="1GiB", unknown=1),
    ):
        with pytest.raises(ValueError):
            StoreHostDonorConfig(**bad)


def test_llm_args_late_auto():
    pytest.importorskip("torch")
    from tensorrt_llm.llmapi.llm_args import StoreHostDonorConfig

    parsed = StoreHostDonorConfig(late_size_per_node="auto", late_auto=LATE_AUTO)
    assert parsed.late_auto.margin_per_gpu == "2560MiB"
    for bad in (
        dict(late_size_per_node="auto"),
        dict(late_size_per_node="64GiB", late_auto=LATE_AUTO),
        dict(late_size_per_node="auto", late_auto={**LATE_AUTO, "store_per_gpu": "-1GiB"}),
        dict(late_size_per_node="auto", late_auto={**LATE_AUTO, "host_budget_per_gpu": 0}),
        dict(late_size_per_node="auto", late_auto={**LATE_AUTO, "granularity": "1GB"}),
        dict(late_size_per_node="auto", late_auto={**LATE_AUTO, "head_hosts": []}),
        dict(late_size_per_node="auto", late_auto={**LATE_AUTO, "other": 1}),
        dict(
            late_size_per_node="auto",
            late_auto={k: v for k, v in LATE_AUTO.items() if k != "head_process_peak_per_gpu"},
        ),
    ):
        with pytest.raises(ValueError):
            StoreHostDonorConfig(**bad)


def test_torch_llm_args_field():
    pytest.importorskip("torch")
    from tensorrt_llm.llmapi.llm_args import StoreHostDonorConfig, TorchLlmArgs

    args = TorchLlmArgs(
        model="/tmp/dummy_model",
        store_host_donor={"startup_size_per_node": "512GiB", "late_size_per_node": "64GiB"},
    )
    assert isinstance(args.store_host_donor, StoreHostDonorConfig)
    assert TorchLlmArgs(model="/tmp/dummy_model").store_host_donor is None
    args = TorchLlmArgs(
        model="/tmp/dummy_model",
        store_host_donor={"late_size_per_node": "auto", "late_auto": LATE_AUTO},
    )
    assert args.store_host_donor.late_auto.store_per_gpu == "192GiB"


# -- executor creation and shutdown hooks (need the full package) --------------------


class FakeDist:
    def __init__(self, rank, others):
        self.rank = rank
        self.others = others

    def allgather(self, obj):
        return [obj, *self.others]


class ScriptedDist:
    """allgather answers from a script, one entry per call (this rank's slot is replaced)."""

    def __init__(self, answers):
        self.rank = 0
        self.answers = list(answers)

    def allgather(self, obj):
        return [obj, *self.answers.pop(0)[1:]]


def _creator():
    pytest.importorskip("torch")
    from tensorrt_llm._torch.pyexecutor import py_executor_creator

    return py_executor_creator


def test_fleet_vote_and_late_phase():
    creator = _creator()
    calls = []
    donor = SimpleNamespace(
        wait_startup=lambda: calls.append("wait"),
        register_late=lambda: calls.append("late"),
        late_delay_seconds=0.0,
    )
    executor = SimpleNamespace()
    creator._register_store_host_donor_late(executor, donor, FakeDist(0, [None]))
    assert calls == ["wait", "late"] and executor.store_host_donor is donor

    calls.clear()
    donor.late_delay_seconds = 5.0
    creator._register_store_host_donor_late(executor, donor, FakeDist(0, [None]))
    assert calls == ["wait"]

    with pytest.raises(RuntimeError, match="other ranks"):
        creator._fleet_vote(FakeDist(0, ["rank 3: boom"]), lambda: None)

    def fail():
        raise ValueError("mine")

    with pytest.raises(ValueError, match="mine"):
        creator._fleet_vote(FakeDist(0, [None]), fail)

    gathered = []

    class RecordingDist:
        rank = 0

        def allgather(self, obj):
            gathered.append(obj)
            return [obj]

    def interrupted():
        raise KeyboardInterrupt

    # An interrupted rank still takes part in the vote, then re-raises.
    with pytest.raises(KeyboardInterrupt):
        creator._fleet_vote(RecordingDist(), interrupted)
    assert len(gathered) == 1 and gathered[0].startswith("rank 0:")


def test_start_vote_fails_every_rank_before_model_load(monkeypatch):
    creator = _creator()
    started = []
    monkeypatch.setattr(
        creator, "start_store_host_donor", lambda config, **kw: started.append(kw) or "donor"
    )
    monkeypatch.setattr(creator, "cpu_numa_nodes", lambda: [0, 1])
    # The local rank helpers divide by the GPU count; CPU-only runners have none.
    monkeypatch.setattr(creator, "local_mpi_rank", lambda: 0)
    monkeypatch.setattr(creator, "local_mpi_size", lambda: 1)
    host = creator.platform.node()
    fixed = SimpleNamespace(numa_nodes=[0, 1])
    detect = SimpleNamespace(numa_nodes=None)
    assert creator._start_store_host_donor(None, FakeDist(0, ["unused"]), 0) is None
    assert started == []
    ok = ScriptedDist([[None, (host, None, False)], [None, None]])
    assert creator._start_store_host_donor(fixed, ok, 0) == "donor"
    # Another rank failed to start: this rank fails too instead of loading the model.
    other = ScriptedDist([[None, (host, None, False)], [None, "rank 3: no NUMA node"]])
    with pytest.raises(RuntimeError, match="other ranks"):
        creator._start_store_host_donor(fixed, other, 0)

    # Detected sockets: ranks on one host must agree before any rank plans.
    started.clear()
    agree = ScriptedDist([[None, (host, [0, 1], False), ("other", [0], False)], [None, None, None]])
    assert creator._start_store_host_donor(detect, agree, 0) == "donor"
    assert started[0]["numa_nodes"] == [0, 1]
    split = ScriptedDist([[None, (host, [1], False)], [None, None]])
    with pytest.raises(ValueError, match="different CPU NUMA nodes"):
        creator._start_store_host_donor(detect, split, 0)


def test_kill_switch_must_agree_across_ranks(monkeypatch):
    creator = _creator()
    monkeypatch.setattr(creator, "start_store_host_donor", lambda config, **kw: "donor")
    fixed = SimpleNamespace(numa_nodes=[0, 1])
    host = creator.platform.node()
    # This rank is enabled, another is disabled: every rank fails, none starts.
    mixed = ScriptedDist([[None, (host, None, True)], [None, None]])
    with pytest.raises(ValueError, match="differs across ranks"):
        creator._start_store_host_donor(fixed, mixed, 0)
    # Disabled everywhere: every rank skips the donor, with no vote.
    monkeypatch.setenv(shd.DISABLE_ENV, "1")
    off = ScriptedDist([[None, (host, None, True)]])
    assert creator._start_store_host_donor(fixed, off, 0) is None
    assert off.answers == []


def test_failed_construction_keeps_another_executors_donor(monkeypatch):
    creator = _creator()
    existing = SimpleNamespace(close=lambda: True)
    monkeypatch.setattr(shd, "_ACTIVE", existing)

    def rejected(**kwargs):
        raise RuntimeError("store_host_donor already active")

    monkeypatch.setattr(creator, "_create_py_executor_impl", rejected)
    with pytest.raises(RuntimeError, match="already active"):
        creator.create_py_executor(llm_args=None)
    assert shd.active_donor() is existing

    closed = []
    started = SimpleNamespace(close=lambda: closed.append(True) or True)

    def starts_then_fails(**kwargs):
        shd._ACTIVE = started
        creator._construction_donors.donors.append(started)
        raise RuntimeError("model load failed")

    monkeypatch.setattr(shd, "_ACTIVE", None)
    monkeypatch.setattr(creator, "_create_py_executor_impl", starts_then_fails)
    with pytest.raises(RuntimeError, match="model load failed"):
        creator.create_py_executor(llm_args=None)
    assert closed == [True] and shd.active_donor() is None


def test_concurrent_construction_keeps_the_winning_donor(monkeypatch):
    creator = _creator()
    closed = []
    winner = SimpleNamespace(close=lambda: closed.append("winner") or True)
    published, finish = threading.Event(), threading.Event()

    rejected = threading.Event()

    def impl(**kwargs):
        if threading.current_thread().name == "first":
            shd._ACTIVE = winner
            creator._construction_donors.donors.append(winner)
            published.set()
            assert finish.wait(timeout=30), "the rejected construction never finished"
            return "executor"
        assert published.wait(timeout=30), "the winning construction never published"
        rejected.set()
        raise RuntimeError("store_host_donor: a donor is already active in this process")

    monkeypatch.setattr(shd, "_ACTIVE", None)
    monkeypatch.setattr(creator, "_create_py_executor_impl", impl)
    results = {}

    def build(name):
        try:
            results[name] = creator.create_py_executor(llm_args=None)
        except RuntimeError as error:
            results[name] = error

    first = threading.Thread(target=build, args=("first",), name="first")
    second = threading.Thread(target=build, args=("second",), name="second")
    first.start()
    second.start()
    try:
        second.join(timeout=30)
        # The rejected construction finished while the winner was still building.
        assert rejected.is_set() and not second.is_alive()
    finally:
        finish.set()
        first.join(timeout=30)
        second.join(timeout=30)
    assert not first.is_alive()
    # The rejected construction never closes the donor the other one owns.
    assert results["first"] == "executor" and isinstance(results["second"], RuntimeError)
    assert closed == [] and shd.active_donor() is winner


@pytest.mark.parametrize("retired", [True, False])
def test_executor_shutdown_retires_donor_after_worker_join(monkeypatch, retired):
    pytest.importorskip("torch")
    from tensorrt_llm._torch.pyexecutor import py_executor as pe

    order = []
    donor = object()

    def event():
        return SimpleNamespace(set=lambda: None, wait=lambda: None)

    executor = SimpleNamespace(
        _openengine_node_server=None,
        executor_request_queue=SimpleNamespace(enqueue_shutdown_request=lambda: None),
        shutdown_event=event(),
        enable_attention_dp=False,
        kv_connector_manager=None,
        _kv_connector_failed=False,
        _profile_manager=SimpleNamespace(cleanup=lambda: None),
        hang_detector=SimpleNamespace(detected=lambda: False),
        _pp_mpi_progress_stop=event(),
        worker_thread=SimpleNamespace(join=lambda: order.append("join")),
        store_host_donor=donor,
        dist=SimpleNamespace(pp_size=1),
        _shutdown_sleep_wakeup_listeners=lambda: None,
        sampler=object(),
        worker_started=True,
        model_engine=None,
        draft_model_engine=None,
        kv_cache_transceiver=None,
        resource_manager=SimpleNamespace(resource_managers={}),
        _release_execution_pools=lambda: True,
    )
    monkeypatch.setattr(shd, "close_active_donor", lambda: order.append("donor") or retired)
    monkeypatch.setattr(pe.torch.cuda, "is_available", lambda: False)
    assert pe.PyExecutor._release_execution_resources(executor)
    assert order == ["join", "donor"]
    # A donor that could not be retired stays referenced, still registered.
    assert executor.store_host_donor is (None if retired else donor)


# -- native smoke (real Mooncake, real NUMA binding; skipped where unavailable) ------


def _node_resident_bytes(node):
    """Resident bytes of this process on ``node`` per /proc/self/numa_maps."""
    total = 0
    with open("/proc/self/numa_maps") as handle:
        for line in handle:
            fields = line.split()
            page = 4096
            for item in fields:
                if item.startswith("kernelpagesize_kB="):
                    page = int(item.split("=")[1]) * 1024
            for item in fields:
                if item.startswith(f"N{node}="):
                    total += int(item.split("=")[1]) * page
    return total


def _native_capable():
    try:
        import mooncake.store  # noqa: F401
    except ImportError:
        return "mooncake is not installed"
    if shutil.which("mooncake_master") is None:
        return "no mooncake_master binary"
    if not os.path.exists("/proc/self/numa_maps") or not shd.cpu_numa_nodes():
        return "no NUMA information"
    error = []

    def probe():
        try:
            shd._bind_current_thread(shd.cpu_numa_nodes()[0])
        except OSError as exc:
            error.append(exc)

    worker = threading.Thread(target=probe)
    worker.start()
    worker.join()
    return f"NUMA binding not permitted: {error[0]}" if error else None


def test_native_registration_smoke(tmp_path, logs):
    reason = _native_capable()
    if reason is not None:
        pytest.skip(reason)
    node = shd.cpu_numa_nodes()[0]
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    master = subprocess.Popen(
        [shutil.which("mooncake_master"), f"--rpc_port={port}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    donor = None
    try:
        deadline = time.monotonic() + 30
        while True:
            try:
                socket.create_connection(("127.0.0.1", port), timeout=1).close()
                break
            except OSError:
                assert time.monotonic() < deadline, "mooncake_master did not come up"
                time.sleep(0.2)
        startup, late = 64 << 20, 32 << 20
        donor = shd.StoreHostDonor(
            master=f"127.0.0.1:{port}",
            shares=[shd.DonorShare(node, startup, late)],
            rank=0,
            local_rank=0,
            local_size=1,
            protocol="tcp",
            device_name="",
            local_buffer_size=16 << 20,
            local_hostname="127.0.0.1",
            node_lock=str(tmp_path / "donor.lock"),
        )
        nodes = shd.cpu_numa_nodes()
        before = {n: _node_resident_bytes(n) for n in nodes}
        donor.start_startup()
        donor.register_late()
        registered = records(logs, "REGISTERED")
        assert sorted(r["phase"] for r in registered) == ["late", "startup"]
        assert all(r["numa_node"] == node for r in registered)
        assert donor.summary()["registered_bytes"] == startup + late
        # Placement: what the registration made resident landed on the bound node.
        grown = {n: _node_resident_bytes(n) - before[n] for n in nodes}
        touched = sum(max(0, g) for g in grown.values())
        # TCP may leave the segment untouched; then there is nothing to place.
        # Threads the donor does not bind (client buffers, allocator arenas)
        # add a few MiB anywhere, so a misplaced segment is what the other
        # nodes must stay clear of: half of the smallest share.
        if touched >= late:
            elsewhere = sum(max(0, g) for n, g in grown.items() if n != node)
            assert elsewhere < late // 2, grown
        assert donor.close()
        assert all(r["ok"] for r in records(logs, "RETIRED"))
    finally:
        try:
            if donor is not None:
                donor.close()  # safe to repeat; retires what a failed assertion left
        finally:
            master.terminate()
            try:
                master.wait(timeout=10)
            except subprocess.TimeoutExpired:
                master.kill()
                master.wait(timeout=10)
