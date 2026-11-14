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
"""Unit tests for the store over NVLink with EGM segments (TLLM_MOONCAKE_STORE_EGM).

Runs without Mooncake, a GPU or NUMA hardware: the client is a fake whose
dictionary setup charges a simulated job cgroup, as EGM does.
"""

import json
from types import SimpleNamespace
from unittest import mock

import pytest

from tensorrt_llm._torch.pyexecutor import store_host_donor as shd
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store import config as store_config

pytestmark = pytest.mark.cpu_only

GIB = 1 << 30
MIB = 1 << 20
MASTER = "10.0.0.1:8700"


class EgmStore:
    """A Mooncake client of a wheel with the EGM store pool."""

    instances = []

    def __init__(self, node=None, status=0, charge=1.0):
        self.node = node
        self.status = status
        self.charge = charge
        self.args = None
        self.calls = []
        self.closed = False
        EgmStore.instances.append(self)

    def setup(self, *args):
        """setup(config: dict) -> int; keys include enable_egm_store_pool and egm_numa_nodes."""
        self.args = args
        self.calls.append("setup")
        if self.status == 0 and self.node is not None and isinstance(args[0], dict):
            size = int(args[0]["global_segment_size"])
            # The pages leave the NUMA node in full; ``charge`` is the share
            # the job cgroup sees.
            self.node.grow(int(size * self.charge))
            self.node.numa_free -= size - int(size * self.charge)
        return self.status

    def get_hostname(self):
        return f"10.0.0.2:{13000 + len(EgmStore.instances)}"

    def allocate_and_mount_segment(self, size, protocol, location):
        self.calls.append(("mount", size, protocol, location))
        return {"ret": 0, "allocated_size": size, "segment_ids": ["s1"]}

    def unmount_and_free_segment(self, ids, grace):
        return 0

    def close(self):
        self.calls.append("close")
        self.closed = True
        return 0


def egm_factory(node=None, **kwargs):
    class Bound(EgmStore):
        def __init__(self):
            super().__init__(node, **kwargs)

    return Bound


class OldStore(EgmStore):
    def setup(self, *args):
        """setup(config: dict) -> int"""
        return 0


class Node:
    """A job cgroup and two NUMA nodes; EGM setup grows the cgroup."""

    def __init__(self, usage=100 * GIB, limit=1200 * GIB, numa_free=600 * GIB):
        self.usage = usage
        self.limit = limit
        self.numa_free = numa_free

    def grow(self, size):
        self.usage += size
        self.numa_free -= size

    def job_memory(self):
        return {"limit": self.limit, "nonfile": self.usage, "path": "/job"}

    def numa(self, _node):
        return {"MemFree": self.numa_free, "FilePages": 0, "SReclaimable": 0}


@pytest.fixture(autouse=True)
def reset(monkeypatch):
    EgmStore.instances = []
    monkeypatch.setattr(shd, "_ACTIVE", None)
    monkeypatch.setattr(shd, "NODE_LOCK_PATH", None)
    monkeypatch.setitem(shd._EGM_STATE, "clients", 0)
    monkeypatch.setattr(shd, "EGM_SETTLE_INTERVAL", 0.0)
    for name in (
        shd.STORE_EGM_ENV,
        shd.EGM_MAX_CLIENTS_ENV,
        shd.DISABLE_ENV,
        shd.MASTER_ENV,
        shd.CONFIG_PATH_ENV,
        *shd._EGM_ENV_SET,
        *shd._EGM_ENV_POP,
    ):
        # setenv first, so teardown restores keys the latch creates.
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
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
    prefix = f"{tag} "
    return [json.loads(msg[len(prefix) :]) for _, msg in lines if msg.startswith(prefix)]


def make_donor(shares, node, *, egm=True, local_rank=0, local_size=4, **kwargs):
    return shd.StoreHostDonor(
        master=MASTER,
        shares=shares,
        rank=local_rank,
        local_rank=local_rank,
        local_size=local_size,
        device_name="mlx5_0",
        local_hostname="10.0.0.2",
        numa_nodes=kwargs.pop("numa_nodes", [0, 1]),
        store_factory=kwargs.pop("factory", None) or egm_factory(node),
        bind_thread=lambda numa: None,
        meminfo=node.numa,
        job_memory=node.job_memory,
        node_lock=None,
        egm=egm,
        egm_usable=kwargs.pop("egm_usable", lambda numa: None),
        **kwargs,
    )


# -- the knob -------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected", [(None, False), ("", False), ("0", False), ("off", False), ("1", True)]
)
def test_knob_values(value, expected):
    environ = {} if value is None else {shd.STORE_EGM_ENV: value}
    assert shd.store_egm_enabled(environ) is expected


def test_knob_rejects_garbage():
    with pytest.raises(ValueError, match=shd.STORE_EGM_ENV):
        shd.store_egm_enabled({shd.STORE_EGM_ENV: "egm"})


def test_environment_latch_sets_pops_and_refuses_after_a_client():
    environ = {"MC_USE_TENT": "1", "MC_USE_NVLINK_IPC": "1", "MC_FORCE_MNNVL": "1", "KEEP": "x"}
    changed = shd.latch_egm_environment(environ)
    assert {k: environ[k] for k in shd._EGM_ENV_SET} == shd._EGM_ENV_SET
    assert not any(k in environ for k in shd._EGM_ENV_POP) and environ["KEEP"] == "x"
    assert changed["set"]["MC_USE_TENT"] == ["1", "0"]
    assert sorted(changed["popped"]) == ["MC_FORCE_MNNVL", "MC_USE_NVLINK_IPC"]
    assert shd.latch_egm_environment(environ) == {"set": {}, "popped": []}
    shd._note_egm_client()
    assert shd.latch_egm_environment(environ) == {"set": {}, "popped": []}
    environ["MC_MS_AUTO_DISC"] = "1"
    with pytest.raises(RuntimeError, match="before the first client"):
        shd.latch_egm_environment(environ)


def test_wheel_check(logs):
    shd.egm_wheel_check(EgmStore)
    with pytest.raises(RuntimeError, match="no RDMA fallback"):
        shd.egm_wheel_check(OldStore)
    assert any(msg.startswith(f"{shd.EGM_TAG}_UNSUPPORTED reason=wheel") for _, msg in logs)


def test_setup_config_is_exact():
    assert shd.egm_setup_config(
        hostname="10.0.0.2", master=MASTER, size=512 * GIB, numa_node=1
    ) == {
        "local_hostname": "10.0.0.2",
        "metadata_server": "P2PHANDSHAKE",
        "master_server_addr": MASTER,
        "protocol": "nvlink",
        "rdma_devices": "",
        "global_segment_size": str(512 * GIB),
        "local_buffer_size": "0",
        "enable_egm_store_pool": "true",
        "egm_numa_nodes": "1",
    }


# -- donor, knob off: unchanged -----------------------------------------------------


def test_knob_off_keeps_the_positional_setup_and_mounts(logs):
    node = Node()
    plain = []

    class Plain(EgmStore):
        def __init__(self):
            super().__init__(None)
            plain.append(self)

    donor = make_donor([shd.DonorShare(0, 8 * GIB, 2 * GIB)], node, egm=False, factory=Plain)
    donor.start_startup()
    donor.wait_startup()
    donor.register_late()
    assert plain[0].args == ("10.0.0.2", "P2PHANDSHAKE", 8 * GIB, GIB, "rdma", "mlx5_0", MASTER)
    assert plain[0].calls[1] == ("mount", 2 * GIB, "rdma", "cpu:0")
    assert not records(logs, f"{shd.EGM_TAG}_ENGAGED")
    registered = records(logs, f"{shd.LOG_TAG}_REGISTERED")
    assert registered and all("memory" not in r and "kind" not in r for r in registered)
    assert shd._EGM_STATE["clients"] == 0
    assert "MC_USE_TENT" not in shd.os.environ


# -- donor, knob on ------------------------------------------------------------------


def test_each_share_is_its_own_egm_client_rounded_to_a_gib(logs):
    node = Node(numa_free=4096 * GIB)
    shares = [shd.DonorShare(0, 512 * GIB + 5 * MIB, 3 * GIB + 7), shd.DonorShare(1, 512 * GIB, 0)]
    donor = make_donor(shares, node)
    assert donor.protocol == "nvlink" and donor.local_buffer_size == 0
    assert donor.late_mode == "new_client"
    donor.start_startup()
    donor.wait_startup()
    donor.register_late()
    configs = [store.args[0] for store in EgmStore.instances]
    assert [c["egm_numa_nodes"] for c in configs] == ["0", "1", "0"]
    assert [int(c["global_segment_size"]) for c in configs] == [512 * GIB, 512 * GIB, 3 * GIB]
    assert all(c["local_buffer_size"] == "0" and c["protocol"] == "nvlink" for c in configs)
    assert all(c["master_server_addr"] == MASTER for c in configs)
    # No segment was mounted on an existing client.
    assert all(call == "setup" for store in EgmStore.instances for call in store.calls)
    registered = records(logs, f"{shd.LOG_TAG}_REGISTERED")
    assert [r["kind"] for r in registered] == ["startup", "startup", "late"]
    assert [r["bytes"] for r in registered] == [512 * GIB, 512 * GIB, 3 * GIB]
    assert all(r["memory"] == "egm" and r["cgroup_delta"] == r["bytes"] for r in registered)
    assert all(r["numa_free_delta"] == r["bytes"] for r in registered)
    engaged = records(logs, f"{shd.EGM_TAG}_ENGAGED")
    assert len(engaged) == 1 and engaged[0]["memory"] == "egm" and engaged[0]["numa"] == [0, 1]
    assert shd.os.environ["MC_USE_TENT"] == "0"
    assert donor.close()
    assert all(store.closed for store in EgmStore.instances)


def test_mount_on_an_existing_client_is_refused():
    node = Node()
    donor = make_donor([shd.DonorShare(0, 2 * GIB, 0)], node)
    base = shd._Registration("startup", shd.DonorShare(0, 2 * GIB, 0), 2 * GIB, store=EgmStore())
    with pytest.raises(RuntimeError, match="own client"):
        donor._mount_on_locked(base, shd._Registration("late", base.share, GIB))


def test_no_transfer_buffer_in_the_headroom_check():
    size = 10 * GIB

    def exact(_node):
        return {"MemFree": size + shd.NODE_HEADROOM_BYTES, "FilePages": 0, "SReclaimable": 0}

    node = Node()
    node.numa = exact
    make_donor([], node)._check_headroom(0, size, client=True)
    with pytest.raises(RuntimeError, match="transfer buffer"):
        make_donor([], node, egm=False)._check_headroom(0, size, client=True)


def test_setup_failure_is_fatal_and_closes_the_client():
    donor = make_donor([shd.DonorShare(0, 4 * GIB, 0)], Node(), factory=egm_factory(status=-1))
    donor.start_startup()
    with pytest.raises(RuntimeError, match="startup registration failed") as error:
        donor.wait_startup()
    assert "no RDMA fallback" in str(error.value.__cause__)
    assert EgmStore.instances[0].closed


def ceiling_factory(node, ceiling):
    """EGM setup fails above ``ceiling`` bytes, as on a socket with too little free memory."""

    class Capped(EgmStore):
        def __init__(self):
            super().__init__(node)

        def setup(self, *args):
            """setup(config: dict) -> int; keys include enable_egm_store_pool and egm_numa_nodes."""
            if int(args[0]["global_segment_size"]) > ceiling:
                self.args = args
                self.calls.append("setup")
                return -1
            return super().setup(*args)

    return Capped


def test_planned_late_lend_retries_until_the_full_share_is_lent(logs):
    node = Node()
    shares = [shd.DonorShare(0, 8 * GIB, 84 * GIB)]
    donor = make_donor(shares, node, factory=ceiling_factory(node, 50 * GIB))
    donor.start_startup()
    donor.wait_startup()
    donor.register_late()
    sizes = [int(store.args[0]["global_segment_size"]) for store in EgmStore.instances]
    # Nothing is known about the node, so the failed size is halved, and the
    # rest of the share follows in a second segment.
    assert sizes == [8 * GIB, 84 * GIB, 42 * GIB, 42 * GIB]
    assert EgmStore.instances[1].closed and not EgmStore.instances[2].closed
    (retry,) = records(logs, f"{shd.EGM_TAG}_LEND_RETRY")
    assert retry["requested"] == 84 * GIB and retry["failure"] == 1
    assert [r["bytes"] for r in records(logs, f"{shd.LOG_TAG}_REGISTERED")] == [
        8 * GIB,
        42 * GIB,
        42 * GIB,
    ]
    (lend,) = records(logs, f"{shd.EGM_TAG}_LATE_LEND")
    assert lend["planned"] == lend["lent"] == 84 * GIB and lend["short"] == 0
    assert donor._egm_stop is None


def test_planned_late_lend_that_never_fits_holds_and_serving_continues(logs):
    node = Node()
    donor = make_donor(
        [shd.DonorShare(0, 0, 40 * GIB)], node, factory=ceiling_factory(node, 0), local_rank=1
    )
    try:
        donor.register_late()  # no raise: the planned store is up, only the extra lend is missing
        sizes = [int(s.args[0]["global_segment_size"]) for s in EgmStore.instances]
        # Each NUMA node: the share, then halved, until the per-node failure limit.
        per_node = [40 * GIB, 20 * GIB, 10 * GIB, 5 * GIB, 2 * GIB]
        assert sizes == per_node + per_node
        assert [s.args[0]["egm_numa_nodes"] for s in EgmStore.instances] == ["0"] * 5 + ["1"] * 5
        (lend,) = records(logs, f"{shd.EGM_TAG}_LATE_LEND")
        assert lend["lent"] == 0 and lend["short"] == 40 * GIB and lend["failures"] == 10
        # Not sticky: a later lend tries again.
        assert donor._egm_stop is None
    finally:
        donor.close()


def test_late_retry_halves_the_fitted_size(logs, monkeypatch):
    # late_auto fitting cuts an 80 GiB plan to 8 GiB before setup; the retries
    # must work from the 8 GiB that was attempted, not the 80 GiB plan.
    node = Node()
    donor = make_donor(
        [shd.DonorShare(0, 0, 80 * GIB)], node, factory=ceiling_factory(node, 4 * GIB)
    )

    def clamp(reg, client):
        reg.bytes = min(reg.bytes, 8 * GIB)
        return True

    monkeypatch.setattr(donor, "_fit_late_auto", clamp)
    reg = shd._Registration("late", donor.shares[0], 80 * GIB)
    assert donor._open_egm_late(reg, fit=True)
    sizes = [int(store.args[0]["global_segment_size"]) for store in EgmStore.instances]
    assert sizes == [8 * GIB, 4 * GIB, 4 * GIB]
    (lend,) = records(logs, f"{shd.EGM_TAG}_LATE_LEND")
    assert lend["planned"] == lend["lent"] == 8 * GIB


def test_startup_share_failure_stays_fatal_after_the_shrink_change():
    node = Node()
    donor = make_donor([shd.DonorShare(0, 64 * GIB, 0)], node, factory=ceiling_factory(node, GIB))
    donor.start_startup()
    with pytest.raises(RuntimeError, match="startup registration failed") as error:
        donor.wait_startup()
    assert isinstance(error.value.__cause__, shd.EgmSetupError)


def test_wheel_without_egm_fails_at_construction():
    with pytest.raises(RuntimeError, match="cannot allocate EGM"):
        make_donor([], Node(), factory=OldStore)


def test_uncharged_segment_stops_lending(logs):
    node = Node()
    donor = make_donor(
        [shd.DonorShare(0, 8 * GIB, 0)], node, factory=egm_factory(node, charge=0.25)
    )
    donor.start_startup()
    donor.wait_startup()
    assert donor._egm_stop == "egm_uncharged"
    (record,) = records(logs, f"{shd.EGM_TAG}_UNCHARGED")
    assert record["bytes"] == 8 * GIB and record["cgroup_delta"] == 2 * GIB
    # Nothing beyond the planned store is registered after that.
    donor.shares = [shd.DonorShare(0, 8 * GIB, 2 * GIB)]
    with pytest.raises(RuntimeError, match="not charged"):
        donor.register_late()
    assert len(EgmStore.instances) == 1


def test_usage_falling_elsewhere_in_the_job_is_not_uncharged(logs):
    # Another rank of the job frees memory while this segment registers: the
    # cgroup grows by less than the segment, but the segment is charged.
    node = Node()
    donor = make_donor([shd.DonorShare(0, 2 * GIB, 0)], node, factory=egm_factory(node, charge=0.9))
    donor.start_startup()
    donor.wait_startup()
    assert donor._egm_stop is None
    assert records(logs, f"{shd.EGM_TAG}_UNCHARGED") == []


def test_retired_egm_is_reported_held_until_exit(logs):
    node = Node()
    donor = make_donor([shd.DonorShare(0, 4 * GIB, 0)], node)
    donor.start_startup()
    donor.wait_startup()
    reg = donor._registrations[0]
    assert donor._retire(reg)
    (retired,) = records(logs, f"{shd.LOG_TAG}_RETIRED")
    assert retired["held_until_exit"] and retired["egm_retired_held"] == 4 * GIB


def test_client_cap_stops_late_lends(logs):
    node = Node(numa_free=2000 * GIB)
    donor = make_donor([], node, egm_max_clients=1)
    donor.adopt_store_segment(4 * GIB)
    summary = donor._lend_egm_locked("late", 0, 8 * GIB, GIB)
    assert summary["lent"] == 0 and donor._egm_stop == "egm_client_cap"
    assert len(EgmStore.instances) == 1


# -- late lends on a fragmented NUMA node ---------------------------------------------


class Driver:
    """HOST_NUMA allocation as the driver does it on a fragmented node.

    Each NUMA node has ``usable`` bytes in free blocks an EGM allocation can
    use (MemFree counts more). An allocation first takes the memory this
    process's earlier failures left kept, then usable blocks. One that does
    not fit fails after taking every usable block; those stay kept, charged
    to the job cgroup, for the next allocation on that node. ``refuse``
    makes that many allocations per node fail the same way even when the
    memory is there, including retries from the kept memory.
    """

    def __init__(self, node, usable, refuse=None, overhead=0):
        self.node = node
        self.usable = dict(usable)
        self.kept = {numa: 0 for numa in usable}
        self.refuse = dict(refuse or {})
        self.overhead = overhead  # process-side bytes each successful setup adds
        self.calls = []

    def allocate(self, numa, size):
        self.calls.append((numa, size))
        refused = self.refuse.get(numa, 0) > 0
        if refused:
            self.refuse[numa] -= 1
        if refused or self.kept[numa] + self.usable[numa] < size:
            # A failed allocation holds at most what it asked for.
            taken = min(self.usable[numa], max(0, size - self.kept[numa]))
            self.usable[numa] -= taken
            self.kept[numa] += taken
            self.node.grow(taken)
            return -1
        reuse = min(self.kept[numa], size)
        self.kept[numa] -= reuse
        self.usable[numa] -= size - reuse
        self.node.grow(size - reuse + self.overhead)
        return 0

    def factory(self):
        driver = self

        class Fragmented(EgmStore):
            def __init__(self):
                super().__init__(None)

            def setup(self, *args):
                """setup(config: dict) -> int; keys include enable_egm_store_pool and egm_numa_nodes."""
                self.args = args
                self.calls.append("setup")
                config = args[0]
                return driver.allocate(
                    int(config["egm_numa_nodes"]), int(config["global_segment_size"])
                )

        return Fragmented

    def usable_bytes(self, numa):
        return self.usable[numa]


def test_usable_bytes_count_only_blocks_an_egm_allocation_can_use(tmp_path):
    path = tmp_path / "buddyinfo"
    path.write_text(
        "Node 0, zone      DMA      2      1      1      1      2      1      0\n"
        "Node 0, zone   Normal   1180  16528  25762  20012  16262  13330   8795\n"
        "Node 1, zone   Normal  21330  17368  13329  10672   6767   6492   4565\n"
    )
    page = 64 << 10
    # 2 MiB blocks with 64 KiB pages are order 5 and up.
    expected = (1 * 32 + 13330 * 32 + 8795 * 64) * page
    assert shd.node_egm_usable_bytes(0, str(path), page_size=page) == expected
    assert shd.node_egm_usable_bytes(1, str(path), page_size=page) == (6492 * 32 + 4565 * 64) * page
    # 4 KiB pages: 2 MiB is order 9, beyond these columns.
    assert shd.node_egm_usable_bytes(0, str(path), page_size=4096) == 0
    assert shd.node_egm_usable_bytes(2, str(path), page_size=page) is None
    assert shd.node_egm_usable_bytes(0, str(tmp_path / "missing"), page_size=page) is None


def test_failed_lend_is_retried_from_the_kept_memory_and_lent_in_full(logs):
    # About 123 GiB asked of a socket whose free memory is mostly in blocks
    # too small for EGM: only 110 GiB is usable.
    node = Node(usage=1000 * GIB, limit=1280 * GIB, numa_free=2000 * GIB)
    driver = Driver(node, {0: 110 * GIB, 1: 300 * GIB})
    donor = make_donor(
        [shd.DonorShare(0, 0, 0)],
        node,
        factory=driver.factory(),
        egm_usable=driver.usable_bytes,
    )
    usage_before = node.usage
    summary = donor._lend_egm_locked("late1", 0, 123 * GIB, 2 * GIB)
    assert summary["lent"] == summary["planned"] == 123 * GIB and summary["short"] == 0
    # Failed at 123 GiB (110 GiB kept), retried at the kept 110 GiB minus
    # nothing usable left, then the remaining 13 GiB on the other socket.
    assert driver.calls == [(0, 123 * GIB), (0, 110 * GIB), (1, 13 * GIB)]
    assert [(s["numa_node"], s["bytes"]) for s in summary["segments"]] == [
        (0, 110 * GIB),
        (1, 13 * GIB),
    ]
    assert summary["segments"][0]["kept_reused"] == 110 * GIB
    # The job grew by exactly what was lent: nothing is left kept, and the
    # reused segment is not mistaken for uncharged memory.
    assert node.usage - usage_before == 123 * GIB == summary["cgroup_growth"]
    assert driver.kept == {0: 0, 1: 0} and donor._egm_kept == {0: 0, 1: 0}
    assert records(logs, f"{shd.EGM_TAG}_UNCHARGED") == [] and donor._egm_stop is None
    assert records(logs, f"{shd.EGM_TAG}_KEPT") == []
    (retry,) = records(logs, f"{shd.EGM_TAG}_LEND_RETRY")
    assert retry["kept"] == 110 * GIB and retry["kept_basis"] == "job_cgroup"


def test_retry_uses_usable_blocks_beyond_the_kept_memory(logs):
    node = Node(usage=100 * GIB, numa_free=2000 * GIB)
    driver = Driver(node, {0: 100 * GIB, 1: 300 * GIB})
    donor = make_donor([], node, factory=driver.factory(), egm_usable=driver.usable_bytes)
    # Usable blocks appear after the failure (page cache freed elsewhere).
    real = driver.allocate

    def allocate(numa, size):
        status = real(numa, size)
        if status != 0:
            driver.usable[numa] += 10 * GIB
        return status

    driver.allocate = allocate
    summary = donor._lend_egm_locked("late", 0, 120 * GIB, GIB)
    # Kept 100 GiB plus 10 GiB usable less the 2 GiB margin.
    assert driver.calls[:2] == [(0, 120 * GIB), (0, 108 * GIB)]
    assert summary["lent"] == 120 * GIB


def test_spill_off_keeps_the_lend_on_its_socket_and_the_next_lend_tops_up(logs, monkeypatch):
    monkeypatch.setenv(shd.EGM_LEND_SPILL_ENV, "0")
    assert shd.egm_lend_spill() is False
    node = Node(usage=100 * GIB, numa_free=2000 * GIB)
    driver = Driver(node, {0: 50 * GIB, 1: 300 * GIB})
    donor = make_donor(
        [], node, factory=driver.factory(), egm_usable=driver.usable_bytes, egm_spill=False
    )
    first = donor._lend_egm_locked("late1", 0, 60 * GIB, 2 * GIB)
    assert first["lent"] == 50 * GIB and first["short"] == 10 * GIB
    assert all(numa == 0 for numa, _ in driver.calls)
    driver.usable[0] += 20 * GIB  # compaction or freed memory later on
    second = donor._lend_egm_locked("late2", 0, first["short"], 2 * GIB)
    assert second["lent"] == 10 * GIB and second["failures"] == 0


def test_reusing_memory_the_driver_pooled_is_not_uncharged(logs):
    # Another process of the job released EGM on socket 1; the driver keeps it
    # pooled (taken from the node and charged), so a new segment there grows
    # neither the cgroup nor the node's usage. That is not uncharged memory.
    node = Node(usage=500 * GIB, numa_free=600 * GIB)

    class Pooled(EgmStore):
        def __init__(self):
            super().__init__(None)

        def setup(self, *args):
            """setup(config: dict) -> int; keys include enable_egm_store_pool and egm_numa_nodes."""
            self.args = args
            return 0

    donor = make_donor([], node, factory=Pooled)
    summary = donor._lend_egm_locked("late1", 1, 40 * GIB, GIB)
    assert summary["lent"] == 40 * GIB
    assert records(logs, f"{shd.EGM_TAG}_UNCHARGED") == [] and donor._egm_stop is None
    (registered,) = records(logs, f"{shd.LOG_TAG}_REGISTERED")
    assert registered["cgroup_delta"] == 0 and registered["node_taken"] == 0


def test_lend_never_grows_the_job_beyond_the_plan(logs):
    import random

    rng = random.Random(7)
    for _ in range(200):
        node = Node(usage=500 * GIB, limit=10**15, numa_free=10**15)
        usable = {0: rng.randrange(0, 200) * GIB, 1: rng.randrange(0, 200) * GIB}
        driver = Driver(node, usable)
        EgmStore.instances = []
        donor = make_donor(
            [],
            node,
            factory=driver.factory(),
            egm_usable=driver.usable_bytes if rng.random() < 0.7 else (lambda numa: None),
            egm_spill=rng.random() < 0.7,
        )
        planned = rng.randrange(1, 250) * GIB
        before = node.usage
        summary = donor._lend_egm_locked("late", 0, planned, GIB)
        growth = node.usage - before
        assert summary["lent"] <= planned
        assert growth <= planned, (usable, planned, driver.calls)
        if donor._egm_spill:
            assert summary["lent"] == min(planned, usable[0] + usable[1]) or (
                summary["failures"] > 0
            )


def test_lend_never_grows_the_job_beyond_the_plan_when_kept_retries_fail(logs):
    # As above, but setups also fail where the memory is there (the kept
    # memory cannot back the retry), so failed nodes strand kept memory.
    import random

    rng = random.Random(11)
    stranded = 0
    for _ in range(300):
        node = Node(usage=500 * GIB, limit=10**15, numa_free=10**15)
        usable = {0: rng.randrange(0, 200) * GIB, 1: rng.randrange(0, 200) * GIB}
        refuse = {0: rng.randrange(0, 7), 1: rng.randrange(0, 4)}
        driver = Driver(node, usable, refuse=refuse)
        EgmStore.instances = []
        donor = make_donor(
            [],
            node,
            factory=driver.factory(),
            egm_usable=driver.usable_bytes if rng.random() < 0.7 else (lambda numa: None),
            egm_spill=rng.random() < 0.7,
        )
        planned = rng.randrange(1, 250) * GIB
        min_segment = rng.choice([GIB, 2 * GIB, 8 * GIB])
        before = node.usage
        summary = donor._lend_egm_locked("late", 0, planned, min_segment)
        growth = node.usage - before
        assert summary["lent"] <= planned
        assert growth <= planned, (usable, refuse, planned, driver.calls)
        assert growth == summary["cgroup_growth"]
        assert growth - summary["lent"] == sum(driver.kept.values())
        stranded += growth > summary["lent"]
    assert stranded > 0  # the cases the bound is about were exercised


def test_home_that_keeps_failing_does_not_let_spill_exceed_the_plan(logs):
    # Plan 100 GiB; home takes 60 GiB on its first failure and fails every
    # retry. Spill may only add what keeps the job within the plan.
    node = Node(usage=100 * GIB, numa_free=2000 * GIB)
    driver = Driver(node, {0: 60 * GIB, 1: 300 * GIB}, refuse={0: 99})
    donor = make_donor([], node, factory=driver.factory(), egm_usable=driver.usable_bytes)
    summary = donor._lend_egm_locked("late1", 0, 100 * GIB, 2 * GIB)
    assert node.usage - 100 * GIB == 100 * GIB == summary["cgroup_growth"]
    assert summary["lent"] == 40 * GIB and summary["segments"] == [
        {"numa_node": 1, "bytes": 40 * GIB, "segment": mock.ANY, "kept_reused": 0}
    ]
    assert summary["kept"] == {"0": 60 * GIB}
    assert [numa for numa, _ in driver.calls].count(0) == shd.EGM_LEND_FAILS_PER_NODE
    (kept,) = records(logs, f"{shd.EGM_TAG}_KEPT")
    assert kept["lent"] == 40 * GIB
    # The next lend reuses the stranded memory on the home socket first.
    driver.refuse = {}
    again = donor._lend_egm_locked("late2", 0, 60 * GIB, 2 * GIB)
    assert again["lent"] == 60 * GIB and driver.kept == {0: 0, 1: 0}
    # Both lends together: 100 GiB lent for 100 GiB of growth.
    assert node.usage - 100 * GIB == 100 * GIB


def test_retry_rounds_a_partial_kept_unit_up(logs):
    # The failed setup kept 110.5 GiB; with a little usable memory the retry
    # takes 111 GiB (all the kept memory) instead of leaving half a GiB
    # stranded, and the spill fills the plan exactly.
    node = Node(usage=100 * GIB, numa_free=2000 * GIB)
    driver = Driver(node, {0: 110 * GIB + GIB // 2, 1: 300 * GIB})
    real = driver.allocate

    def allocate(numa, size):
        status = real(numa, size)
        if status != 0:
            driver.usable[numa] += GIB  # a little freed after the failure
        return status

    driver.allocate = allocate
    donor = make_donor([], node, factory=driver.factory(), egm_usable=driver.usable_bytes)
    summary = donor._lend_egm_locked("late1", 0, 123 * GIB, 2 * GIB)
    assert driver.calls == [(0, 123 * GIB), (0, 111 * GIB), (1, 12 * GIB)]
    assert summary["lent"] == 123 * GIB == summary["cgroup_growth"]


def test_rounded_retry_that_fails_falls_back_to_the_kept_memory(logs):
    # No memory beyond the kept 110.5 GiB: 111 GiB fails, 110 GiB (kept, whole
    # units) is lent, and the strict bound leaves the stranded half GiB unlent.
    node = Node(usage=100 * GIB, numa_free=2000 * GIB)
    driver = Driver(node, {0: 110 * GIB + GIB // 2, 1: 300 * GIB})
    donor = make_donor([], node, factory=driver.factory(), egm_usable=driver.usable_bytes)
    summary = donor._lend_egm_locked("late1", 0, 123 * GIB, 2 * GIB)
    assert driver.calls == [(0, 123 * GIB), (0, 111 * GIB), (0, 110 * GIB), (1, 12 * GIB)]
    assert summary["lent"] == 122 * GIB and summary["cgroup_growth"] <= 123 * GIB


def test_near_plan_retry_repeats_the_failed_size_once_kept_memory_covers_it(logs):
    # 124 GiB fails keeping 123.5 GiB; 1 GiB frees up afterwards. Asking 124
    # again now needs only half a GiB of new memory, so it is tried once.
    node = Node(usage=100 * GIB, numa_free=2000 * GIB)
    driver = Driver(node, {0: 123 * GIB + GIB // 2, 1: 300 * GIB})
    real = driver.allocate

    def allocate(numa, size):
        status = real(numa, size)
        if status != 0:
            driver.usable[numa] += GIB
        return status

    driver.allocate = allocate
    donor = make_donor([], node, factory=driver.factory(), egm_usable=driver.usable_bytes)
    summary = donor._lend_egm_locked("late1", 0, 124 * GIB, 2 * GIB)
    assert driver.calls == [(0, 124 * GIB), (0, 124 * GIB)]
    assert summary["lent"] == 124 * GIB == summary["cgroup_growth"]


def test_setup_overhead_does_not_cost_a_unit_of_the_plan(logs):
    # Each setup also grows the job by a few MiB of process memory; that is
    # outside the EGM bound and must not round the spill down a whole GiB.
    node = Node(usage=100 * GIB, numa_free=2000 * GIB)
    driver = Driver(node, {0: 107 * GIB + GIB // 4, 1: 300 * GIB}, overhead=16 << 20)
    donor = make_donor([], node, factory=driver.factory(), egm_usable=driver.usable_bytes)
    summary = donor._lend_egm_locked("late1", 0, 124 * GIB, 2 * GIB)
    assert summary["lent"] == 123 * GIB  # 0.25 GiB stays kept on the home socket
    driver2 = Driver(
        Node(usage=100 * GIB, numa_free=2000 * GIB),
        {0: 108 * GIB, 1: 300 * GIB},
        overhead=16 << 20,
    )
    donor2 = make_donor(
        [], driver2.node, factory=driver2.factory(), egm_usable=driver2.usable_bytes
    )
    summary2 = donor2._lend_egm_locked("late1", 0, 124 * GIB, 2 * GIB)
    assert summary2["lent"] == 124 * GIB
    assert [(s["numa_node"], s["bytes"]) for s in summary2["segments"]] == [
        (0, 108 * GIB),
        (1, 16 * GIB),
    ]


def test_a_final_segment_below_min_segment_completes_the_plan(logs):
    node = Node(usage=100 * GIB, numa_free=2000 * GIB)
    driver = Driver(node, {0: 123 * GIB, 1: 300 * GIB})
    donor = make_donor([], node, factory=driver.factory(), egm_usable=driver.usable_bytes)
    summary = donor._lend_egm_locked("late1", 0, 124 * GIB, 2 * GIB)
    assert summary["lent"] == 124 * GIB
    assert [(s["numa_node"], s["bytes"]) for s in summary["segments"]] == [
        (0, 123 * GIB),
        (1, GIB),
    ]


def test_static_late_share_left_short_is_retried_in_the_background(logs, monkeypatch):
    # A static late share is lent once. A home socket that fails five times leaves 100 GiB kept and nothing lent; the
    # background retry reuses the kept memory and completes the share,
    # growing the job by the plan only.
    monkeypatch.setenv(shd.EGM_LATE_RETRY_SECONDS_ENV, "0")
    node = Node(usage=100 * GIB, numa_free=2000 * GIB)
    driver = Driver(node, {0: 100 * GIB, 1: 0}, refuse={0: 5, 1: 5})
    donor = make_donor(
        [shd.DonorShare(0, 0, 100 * GIB)],
        node,
        factory=driver.factory(),
        egm_usable=driver.usable_bytes,
    )
    donor.register_late()
    assert donor._late_done
    donor._egm_late_retry_thread.join(10)
    assert not donor._egm_late_retry_thread.is_alive()
    lends = records(logs, f"{shd.EGM_TAG}_LATE_LEND")
    assert lends[0]["lent"] == 0 and lends[0]["kept"] == {"0": 100 * GIB}
    assert lends[1]["phase"] == "late_retry1" and lends[1]["lent"] == 100 * GIB
    assert node.usage - 100 * GIB == 100 * GIB
    (done,) = records(logs, f"{shd.EGM_TAG}_LATE_COMPLETE")
    assert done["short"] == 0
    assert donor._egm_late_short[0][2] == 0


def test_static_late_share_that_never_recovers_is_reported_incomplete(logs, monkeypatch):
    monkeypatch.setenv(shd.EGM_LATE_RETRY_SECONDS_ENV, "0")
    monkeypatch.setattr(shd, "EGM_LATE_RETRY_PASSES", 2)
    node = Node(usage=100 * GIB, numa_free=2000 * GIB)
    driver = Driver(node, {0: 60 * GIB, 1: 0}, refuse={0: 99})
    donor = make_donor(
        [shd.DonorShare(0, 0, 100 * GIB)],
        node,
        factory=driver.factory(),
        egm_usable=driver.usable_bytes,
    )
    donor.register_late()
    donor._egm_late_retry_thread.join(10)
    (incomplete,) = records(logs, f"{shd.EGM_TAG}_LATE_INCOMPLETE")
    assert incomplete["short"] == 100 * GIB and incomplete["kept"] == 60 * GIB
    # Never beyond the plan: only the first failure's 60 GiB was taken.
    assert node.usage - 100 * GIB == 60 * GIB


def test_close_waits_for_an_in_flight_static_retry_and_releases_it(logs, monkeypatch):
    # close() begins while a background retry setup is in flight. It must not
    # report success until that setup has returned, and what it lent must be
    # released, not left registered with an open client.
    import threading

    monkeypatch.setenv(shd.EGM_LATE_RETRY_SECONDS_ENV, "0")
    node = Node(usage=100 * GIB, numa_free=2000 * GIB)
    driver = Driver(node, {0: 100 * GIB, 1: 0}, refuse={0: 5, 1: 5})
    in_retry, release = threading.Event(), threading.Event()
    real = driver.allocate

    def allocate(numa, size):
        if len(driver.calls) >= 5:  # the first pass made five failed setups
            in_retry.set()
            assert release.wait(10)
        return real(numa, size)

    driver.allocate = allocate
    donor = make_donor(
        [shd.DonorShare(0, 0, 100 * GIB)],
        node,
        factory=driver.factory(),
        egm_usable=driver.usable_bytes,
    )
    donor.register_late()
    assert in_retry.wait(10)
    result = {}
    closer = threading.Thread(target=lambda: result.setdefault("ok", donor.close()))
    closer.start()
    closer.join(0.3)
    assert closer.is_alive() and "ok" not in result  # waits for the setup
    release.set()
    closer.join(10)
    assert result["ok"] is True and donor._closed
    assert not donor._egm_late_retry_thread.is_alive()
    assert donor._registrations == []
    assert all(store.closed for store in EgmStore.instances if store.calls)
    # Nothing is retried after close.
    calls = len(driver.calls)
    donor._egm_start_late_retry()
    assert len(driver.calls) == calls


def test_kept_below_min_segment_does_not_let_spill_exceed_the_plan(logs):
    node = Node(usage=100 * GIB, numa_free=2000 * GIB)
    driver = Driver(node, {0: GIB, 1: 300 * GIB})
    donor = make_donor([], node, factory=driver.factory(), egm_usable=driver.usable_bytes)
    summary = donor._lend_egm_locked("late1", 0, 10 * GIB, 2 * GIB)
    assert summary["cgroup_growth"] == 10 * GIB
    assert summary["lent"] == 9 * GIB and driver.calls == [(0, 10 * GIB), (1, 9 * GIB)]


def test_kept_memory_from_an_earlier_call_is_not_trusted(logs):
    # A failed call leaves 50 GiB kept on socket 0; another process of the
    # job then reuses that pooled memory. The next call must not size its
    # retries (or its headroom) from the stale 50 GiB.
    node = Node(usage=100 * GIB, numa_free=2000 * GIB)
    driver = Driver(node, {0: 50 * GIB, 1: 0})
    donor = make_donor(
        [], node, factory=driver.factory(), egm_usable=driver.usable_bytes, egm_spill=False
    )
    first = donor._lend_egm_locked("late1", 0, 60 * GIB, 2 * GIB)
    assert first["lent"] == 50 * GIB
    driver.calls.clear()
    donor._egm_kept[0] = 50 * GIB  # as an earlier failure would have left it
    second = donor._lend_egm_locked("late2", 0, 40 * GIB, 2 * GIB)
    # One try at the plan; nothing kept is known in this call and nothing is
    # usable, so no retry is sized from the stale figure.
    assert driver.calls == [(0, 40 * GIB)]
    assert second["lent"] == 0 and second["failures"] == 1 and second["kept"] == {}


def test_cgroup_headroom_does_not_count_kept_memory():
    node = Node(usage=1200 * GIB, limit=1280 * GIB, numa_free=2000 * GIB)
    donor = make_donor([], node)
    donor.local_buffer_size = 0
    donor._check_headroom(0, 70 * GIB, client=True, kept=0)
    with pytest.raises(RuntimeError, match="memory cgroup"):
        donor._check_headroom(0, 100 * GIB, client=True, kept=60 * GIB)


# -- the KV connector's segment -----------------------------------------------------


def test_adoption_lands_on_the_ranks_socket(logs):
    node = Node()
    donor = make_donor([], node, local_rank=1, local_size=2)
    assert donor.adopt_store_segment(210 * GIB + 3 * MIB) == 210 * GIB
    (config,) = [store.args[0] for store in EgmStore.instances]
    assert config["egm_numa_nodes"] == "1" and config["global_segment_size"] == str(210 * GIB)
    (registered,) = records(logs, f"{shd.LOG_TAG}_REGISTERED")
    assert registered["kind"] == "startup_adopted" and registered["phase"] == "startup_adopted"
    with pytest.raises(RuntimeError, match="rounds to nothing"):
        donor.adopt_store_segment(GIB - 1)


def test_adoption_needs_the_knob_and_a_donor(monkeypatch):
    with pytest.raises(RuntimeError, match="needs"):
        make_donor([], Node(), egm=False).adopt_store_segment(GIB)
    with pytest.raises(RuntimeError, match="runs none"):
        shd.adopt_store_segment(GIB)
    donor = make_donor([], Node(), local_rank=0, local_size=4)
    monkeypatch.setattr(shd, "_ACTIVE", donor)
    assert shd.adopt_store_segment(2 * GIB) == 2 * GIB
    assert EgmStore.instances[0].args[0]["egm_numa_nodes"] == "0"


def test_start_from_config_with_the_knob(monkeypatch, logs):
    config = SimpleNamespace(
        numa_nodes=[0, 1],
        startup_size_per_node="2GiB",
        late_size_per_node=0,
        late_auto=None,
        master=MASTER,
        device_name=None,
        protocol="rdma",
        local_buffer_size="1GiB",
        local_hostname="10.0.0.2",
        late_mode="auto",
        late_delay_seconds=0.0,
        hbm_lend_size_per_gpu=0,
    )
    node = Node()
    donor = shd.start_store_host_donor(
        config,
        rank=0,
        local_rank=0,
        local_size=2,
        environ={shd.STORE_EGM_ENV: "1", shd.EGM_MAX_CLIENTS_ENV: "3"},
        store_factory=egm_factory(node),
        bind_thread=lambda numa: None,
        meminfo=node.numa,
        job_memory=node.job_memory,
        node_lock=None,
    )
    donor.wait_startup()
    assert donor.egm and donor._egm_max_clients == 3 and donor.device_name == ""
    (plan,) = records(logs, f"{shd.LOG_TAG}_PLAN")
    assert plan["memory"] == "egm" and plan["late_mode"] == "new_client"
    assert EgmStore.instances[0].args[0]["global_segment_size"] == str(GIB)
    assert donor.close()


def test_late_auto_cross_check_still_uses_the_store_size(tmp_path):
    path = tmp_path / "mooncake.json"
    path.write_text(json.dumps({"global_segment_size": "210GiB"}))
    late_auto = SimpleNamespace(
        host_budget_per_gpu="384GiB",
        store_per_gpu="210GiB",
        process_peak_per_gpu="32GiB",
        head_process_peak_per_gpu="64GiB",
        margin_per_gpu="2GiB",
        granularity="1GiB",
        head_hosts=None,
        job_memory_limit=None,
    )
    environ = {shd.CONFIG_PATH_ENV: str(path), shd.STORE_EGM_ENV: "1"}
    assert shd._late_auto_from_config(late_auto, environ).store_per_gpu == 210 * GIB
    late_auto.store_per_gpu = "200GiB"
    with pytest.raises(ValueError, match="budget would not hold"):
        shd._late_auto_from_config(late_auto, environ)


# -- connector config ---------------------------------------------------------------


def connector_config(**fields):
    base = dict(
        master_server_address=MASTER,
        protocol="rdma",
        device_name="mlx5_0,mlx5_1",
        global_segment_size=210 * GIB,
        role=store_config.StoreRole.BOTH,
    )
    base.update(fields)
    return store_config.MooncakeStoreConnectorConfig(**base)


def test_connector_knob_off_is_unchanged():
    config = connector_config()
    same, store_bytes = config.with_store_transport(0, environ={})
    assert same is config and store_bytes == 210 * GIB
    with pytest.raises(ValueError, match="halfway|set both or neither"):
        connector_config(protocol="nvlink").with_store_transport(0, environ={})


def test_connector_knob_on_overrides(monkeypatch):
    warnings = []
    monkeypatch.setattr(shd.logger, "warning", warnings.append)
    environ = {shd.STORE_EGM_ENV: "1"}
    config, store_bytes = connector_config().with_store_transport(3, environ=environ)
    assert store_bytes == 210 * GIB
    assert (config.protocol, config.device_name, config.global_segment_size) == ("nvlink", "", 0)
    # Pages go straight between EGM and the KV pool; the staging size is left alone.
    assert not config.stage_through_host
    assert config.staging_buffer_bytes == store_config.DEFAULT_STAGING_BUFFER_BYTES
    assert config.local_buffer_size == GIB
    overridden = sorted(
        json.loads(w[len(f"{shd.EGM_TAG}_OVERRIDE ") :])["key"]
        for w in warnings
        if w.startswith(f"{shd.EGM_TAG}_OVERRIDE ")
    )
    assert overridden == ["device_name", "global_segment_size", "protocol"]
    (engaged,) = [
        json.loads(w[len(f"{shd.EGM_TAG}_ENGAGED ") :])
        for w in warnings
        if w.startswith(f"{shd.EGM_TAG}_ENGAGED ")
    ]
    assert engaged["role"] == "connector" and engaged["store_bytes"] == 210 * GIB
    assert engaged["direct"] and engaged["staging_bytes"] == 0


def test_connector_knob_on_with_direct_off_stages_through_host(monkeypatch):
    warnings = []
    monkeypatch.setattr(shd.logger, "warning", warnings.append)
    environ = {shd.STORE_EGM_ENV: "1", shd.EGM_DIRECT_DISABLE_ENV: "1"}
    config, _ = connector_config().with_store_transport(3, environ=environ)
    assert config.stage_through_host and config.staging_buffer_bytes == 2 * GIB
    overridden = sorted(
        json.loads(w[len(f"{shd.EGM_TAG}_OVERRIDE ") :])["key"]
        for w in warnings
        if w.startswith(f"{shd.EGM_TAG}_OVERRIDE ")
    )
    assert overridden == [
        "device_name",
        "global_segment_size",
        "protocol",
        "stage_through_host",
        "staging_buffer_bytes",
    ]
    (engaged,) = [
        json.loads(w[len(f"{shd.EGM_TAG}_ENGAGED ") :])
        for w in warnings
        if w.startswith(f"{shd.EGM_TAG}_ENGAGED ")
    ]
    assert not engaged["direct"] and engaged["staging_bytes"] == 2 * GIB
    # A staging size the config chose is kept.
    kept, _ = connector_config(staging_buffer_bytes=512 * MIB).with_store_transport(
        0, environ=environ
    )
    assert kept.staging_buffer_bytes == 512 * MIB


# -- executor creation ---------------------------------------------------------------


class ScriptedDist:
    """allgather answers from a script, one entry per call (this rank's slot is replaced)."""

    def __init__(self, answers):
        self.rank = 0
        self.answers = list(answers)

    def allgather(self, obj):
        return [obj, *self.answers.pop(0)[1:]]


def test_ranks_must_agree_on_the_transport(monkeypatch):
    from tensorrt_llm._torch.pyexecutor import py_executor_creator as creator

    monkeypatch.setattr(creator, "start_store_host_donor", lambda config, **kw: "donor")
    fixed = SimpleNamespace(numa_nodes=[0, 1])
    host = creator.platform.node()
    monkeypatch.setenv(shd.STORE_EGM_ENV, "1")
    agree = ScriptedDist([[None, (host, None, False, True)], [None, None]])
    assert creator._start_store_host_donor(fixed, agree, 0) == "donor"
    mixed = ScriptedDist([[None, (host, None, False, False)], [None, None]])
    with pytest.raises(ValueError, match=f"{shd.STORE_EGM_ENV} differs across ranks"):
        creator._start_store_host_donor(fixed, mixed, 0)
    # Every donor off while the store runs over EGM: no memory at all.
    monkeypatch.setenv(shd.DISABLE_ENV, "1")
    off = ScriptedDist([[None, (host, None, True, True)]])
    with pytest.raises(ValueError, match="without memory"):
        creator._start_store_host_donor(fixed, off, 0)
