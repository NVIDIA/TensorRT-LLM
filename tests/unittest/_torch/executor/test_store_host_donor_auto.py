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
"""Unit tests for the measured late donation of the store host donor.

The node is simulated with fake /proc/meminfo, cgroup v2 and driver files and
a fake clock; the Mooncake client is a fake whose mounts charge the fake
cgroup, as pinned segments do.
"""

import dataclasses
import json
import threading
import time
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor import store_host_donor as shd

pytestmark = pytest.mark.cpu_only

GIB = 1 << 30
#: Hecate: X = 1430 GiB per node, 4 GPUs, job cgroup 1278.8 GiB per node.
MEMTOTAL = 1430 * GIB
CGROUP = int(1278.8 * GIB)


class FakeNode:
    """One node: meminfo, job cgroup and GPU files under ``root``."""

    def __init__(self, root, *, memtotal=MEMTOTAL, limit=CGROUP, usage=0, gpus=4, job="77"):
        self.root = root
        self.memtotal = memtotal
        self.limit = limit
        self.usage = usage
        self.job = job
        root.mkdir(parents=True, exist_ok=True)
        (root / "gpus").mkdir()
        for index in range(gpus):
            (root / "gpus" / f"0000:0{index}:00.0").mkdir()
        (root / "dev").mkdir()
        (root / "proc_cgroup").write_text(f"0::/job_{job}/step_0\n")
        self.cg = root / "cg" / f"job_{job}"
        (self.cg / "step_0").mkdir(parents=True)
        self.write()

    def write(self):
        free = self.memtotal - self.usage - 10 * GIB
        (self.root / "meminfo").write_text(
            f"MemTotal: {self.memtotal // 1024} kB\n"
            f"MemFree: {free // 1024} kB\n"
            "Buffers: 0 kB\n"
            f"Cached: {10 * GIB // 1024} kB\n"
            f"Shmem: {GIB // 1024} kB\n"
        )
        if self.limit is not None:
            (self.cg / "memory.max").write_text(f"{self.limit}\n")
        # 10 GiB page cache, 1 GiB of it shared memory: nonfile == usage.
        (self.cg / "memory.current").write_text(f"{self.usage + 9 * GIB}\n")
        (self.cg / "memory.stat").write_text(f"anon {self.usage}\nfile {10 * GIB}\nshmem {GIB}\n")

    def grow(self, size):
        self.usage += size
        self.write()

    def job_memory(self):
        return shd.cgroup_memory(str(self.root / "proc_cgroup"), str(self.root / "cg"))

    def numa(self, _node):
        free = self.memtotal - self.usage
        return {"MemFree": free // 2, "FilePages": 0, "SReclaimable": 0}


class FakeClock:
    def __init__(self, now=1_000_000.0):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class ChargingStore:
    """A Mooncake client whose segments charge the node, like pinned memory."""

    def __init__(self, node):
        self.node = node
        self.mounts = []

    def setup(self, hostname, metadata, size, buffer, protocol, device, master):
        self.node.grow(size + buffer)
        return 0

    def get_hostname(self):
        return "10.0.0.2:12001"

    def allocate_and_mount_segment(self, size, protocol, location):
        self.node.grow(size)
        self.mounts.append((size, location))
        return {"ret": 0, "allocated_size": size, "segment_ids": [f"s{len(self.mounts)}"]}

    def unmount_and_free_segment(self, ids, grace):
        return 0

    def close(self):
        return 0


@pytest.fixture(autouse=True)
def reset(monkeypatch):
    monkeypatch.setattr(shd, "_ACTIVE", None)
    for name in (
        shd.DISABLE_ENV,
        shd.AUTO_ENV,
        shd.LATE_GIB_ENV,
        shd.BUFFER_ENV,
        shd.HEAD_BUFFER_ENV,
        shd.MAX_LENT_ENV,
        shd.TOPUP_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
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
    return [json.loads(msg[len(prefix) :]) for _, msg in lines if msg.startswith(prefix)]


def donor_on(
    node,
    tmp_path,
    clock,
    *,
    rank=0,
    local_rank=0,
    local_size=4,
    numa_nodes=(0, 1),
    head=False,
    measured=None,
    environ=None,
    lock=True,
    late_mode="auto",
    first_needs_slope=False,
):
    environ = {"SLURM_JOB_ID": node.job, **(environ or {})}
    shares = shd.plan_shares(0, 0, list(numa_nodes), local_rank, local_size)
    donor = shd.StoreHostDonor(
        master="10.0.0.1:8700",
        shares=shares,
        rank=rank,
        local_rank=local_rank,
        local_size=local_size,
        device_name="mlx5_0",
        local_hostname="10.0.0.2",
        numa_nodes=list(numa_nodes),
        late_mode=late_mode,
        measured=dataclasses.replace(
            measured or shd.MeasuredAuto(), first_lend_needs_slope=first_needs_slope
        ),
        probes=shd.MeasuredProbes(
            meminfo_path=str(node.root / "meminfo"),
            gpu_proc_dir=str(node.root / "gpus"),
            dev_dir=str(node.root / "dev"),
            registry_path=str(tmp_path / "registry.json"),
            environ=environ,
            clock=clock,
            scontrol=lambda job: None,
        ),
        head_role=lambda master, hosts: (head, "test"),
        job_memory=node.job_memory,
        node_lock=str(tmp_path / "node.lock") if lock else None,
        store_factory=lambda: ChargingStore(node),
        bind_thread=lambda numa: None,
        meminfo=node.numa,
    )
    donor.start_startup()
    donor.wait_startup()
    return donor


def gen_node(tmp_path, usage=1077 * GIB):
    return FakeNode(tmp_path / "node", usage=usage)


def gen_worker(node, tmp_path, clock, **kwargs):
    """One DEP8 GEN worker's 4 ranks on a node: ranks 0 and 2 own a socket each."""
    return [
        donor_on(node, tmp_path, clock, rank=r, local_rank=r, local_size=4, **kwargs)
        for r in (0, 2)
    ]


# -- pure rule ----------------------------------------------------------------


def test_measured_lend_rule_terms():
    amount, reason, terms = shd.measured_lend(
        node_budget=1000 * GIB,
        node_buffer=100 * GIB,
        usage=700 * GIB,
        lent_node=0,
        lent_unit=0,
        weight=2,
        active_gpus=4,
        max_lent_per_node=None,
        slope=None,
        seconds_left=3600,
        require_slope=False,
        granularity=GIB,
        min_segment=2 * GIB,
    )
    # Node room 200 GiB, this donor's half of it.
    assert (amount, reason) == (100 * GIB, "room")
    assert terms["node_room"] == 200 * GIB and terms["unit_room"] == 100 * GIB


def test_growth_slope_least_squares():
    samples = [(t, 100 * GIB + t * GIB // 60) for t in range(0, 601, 30)]
    slope = shd.growth_slope(samples, 600, 900, 240)
    assert slope == pytest.approx(GIB / 60, rel=1e-6)
    assert shd.growth_slope(samples[:5], 120, 900, 240) is None
    # Only the window counts.
    assert shd.growth_slope(samples, 600, 100, 240) is None


def test_job_end_sources():
    assert shd.job_end_epoch({"SLURM_JOB_END_TIME": "2000"}) == (2000.0, "SLURM_JOB_END_TIME")
    assert shd.job_end_epoch({"SLURM_JOB_ID": "5"}, scontrol=lambda job: 3000.0) == (
        3000.0,
        "scontrol",
    )
    assert shd.job_end_epoch({"SLURM_JOB_ID": "5"}, scontrol=lambda job: None) == (
        None,
        "unknown",
    )


def test_node_probes(tmp_path):
    node = FakeNode(tmp_path, usage=100 * GIB, gpus=4)
    assert shd.node_memtotal(str(tmp_path / "meminfo")) == MEMTOTAL
    assert shd.node_gpu_count(str(tmp_path / "gpus"), str(tmp_path / "dev")) == 4
    (tmp_path / "dev" / "nvidia0").touch()
    (tmp_path / "dev" / "nvidiactl").touch()
    assert shd.node_gpu_count(str(tmp_path / "missing"), str(tmp_path / "dev")) == 1
    assert shd.node_gpu_count(str(tmp_path / "missing"), str(tmp_path / "missing")) is None
    # meminfo nonfile: total - free - cache + shmem == usage + 1 GiB shmem.
    assert shd.node_used_nonfile(str(tmp_path / "meminfo")) == 101 * GIB
    assert node.job_memory()["nonfile"] == 100 * GIB


# -- budget, buffer -------------------------------------------------------------


def test_lend_never_exceeds_budget_and_buffer_stays_free(tmp_path, logs):
    clock = FakeClock()
    node = gen_node(tmp_path)
    donors = gen_worker(node, tmp_path, clock, measured=shd.MeasuredAuto(min_segment=GIB))
    budget = CGROUP  # min(4 x 1430/4, 1278.8): the cgroup binds
    buffer = 4 * 36 * GIB
    for _ in range(6):
        for donor in donors:
            donor._auto_pass()
            assert node.usage <= budget - buffer
        clock.advance(300)
        for donor in donors:
            donor._auto_sample()
    lends = records(logs, "AUTO_LEND")
    assert lends, "the free room above the buffer is lent"
    assert all(r["budget"] == budget and r["buffer"] == buffer for r in lends)
    # Room 1278.8 - 144 - 1079 (usage + two 1 GiB client buffers) = 55.8 GiB.
    assert budget - buffer - node.usage < 2 * GIB
    assert sum(d._auto_lent for d in donors) == 54 * GIB
    for record in lends:
        assert record["amount"] <= record["node_room"]
        assert record["per_gpu_after_gib"] <= (budget - buffer) / 4 / GIB


def test_x4_bounds_the_budget_when_the_cgroup_is_larger(tmp_path, logs):
    clock = FakeClock()
    node = FakeNode(tmp_path / "node", limit=1400 * GIB, usage=300 * GIB)
    # A CTX DEP2 worker: 2 GPUs in use on a 4-GPU node -> 2 x X/4 = 715 GiB.
    donors = [donor_on(node, tmp_path, clock, rank=r, local_rank=r, local_size=2) for r in (0, 1)]
    for donor in donors:
        donor._auto_pass()
    (first, second) = records(logs, "AUTO_LEND")
    assert first["budget"] == 715 * GIB and first["active_gpus"] == 2
    assert node.usage <= 715 * GIB - 72 * GIB


def test_budget_override_caps_per_gpu(tmp_path, logs):
    clock = FakeClock()
    node = FakeNode(tmp_path / "node", limit=None, usage=300 * GIB)
    measured = shd.measured_auto_from_env({shd.BUDGET_ENV: "300"})
    donor = donor_on(node, tmp_path, clock, local_size=1, measured=measured)
    record = donor._auto_pass()
    # No cgroup limit: usage comes from /proc/meminfo, the budget from the override.
    assert record["usage_source"] == "node_meminfo_nonfile"
    assert record["budget"] == 300 * GIB and record["reason"] == "no_room"


def test_head_node_buffer(tmp_path, logs):
    clock = FakeClock()
    node = FakeNode(tmp_path / "node", limit=1400 * GIB, usage=600 * GIB)
    donor = donor_on(node, tmp_path, clock, local_size=2, numa_nodes=(0,), head=True)
    record = donor._auto_pass()
    # 715 - 601 used - 2 x 64 head buffer < 0: nothing.
    assert record["head"] and record["buffer"] == 128 * GIB
    assert record["reason"] == "no_room" and donor._auto_lent == 0
    other = FakeNode(tmp_path / "other", limit=1400 * GIB, usage=600 * GIB)
    (tmp_path / "o").mkdir()
    plain = donor_on(other, tmp_path / "o", clock, local_size=2, numa_nodes=(0,))
    record = plain._auto_pass()
    assert record["buffer"] == 72 * GIB and record["amount"] == 42 * GIB


def test_missing_inputs_lend_nothing(tmp_path, logs):
    clock = FakeClock()
    node = gen_node(tmp_path)
    (donor,) = [donor_on(node, tmp_path, clock, local_size=1, numa_nodes=(0,))]
    (node.root / "meminfo").write_text("")
    record = donor._auto_pass()
    assert record["reason"].startswith("unknown_") and donor._auto_lent == 0


# -- growth guard ---------------------------------------------------------------


def _ctx_donor(tmp_path, clock, environ=None, usage=600 * GIB):
    node = FakeNode(tmp_path / "node", limit=1400 * GIB, usage=usage)
    measured = shd.MeasuredAuto(min_segment=GIB)
    donor = donor_on(
        node,
        tmp_path,
        clock,
        local_size=2,
        numa_nodes=(0,),
        measured=measured,
        environ=environ or {"SLURM_JOB_END_TIME": str(int(clock() + 3600))},
    )
    return node, donor


def _serve(node, donor, clock, gib_per_min, minutes):
    for _ in range(int(minutes * 2)):
        clock.advance(30)
        node.grow(int(gib_per_min * GIB / 2))
        donor._auto_sample()


def test_topup_stops_when_the_slope_would_use_up_the_buffer(tmp_path, logs):
    clock = FakeClock()
    node, donor = _ctx_donor(tmp_path, clock)
    first = donor._auto_pass()
    assert first["reason"] == "room"
    # 2.2 GiB/min for the hour left > the 72 GiB buffer: no top-up.
    _serve(node, donor, clock, 2.2, 5)
    held = donor._auto_pass()
    assert held["reason"] == "growth_guard" and held["amount"] == 0
    assert held["slope_gib_per_min"] == pytest.approx(2.2, rel=0.02)
    assert held["end_source"] == "SLURM_JOB_END_TIME"


def test_topup_continues_under_a_slow_slope(tmp_path, logs):
    clock = FakeClock()
    node, donor = _ctx_donor(tmp_path, clock, usage=500 * GIB)
    donor._auto_pass()
    lent = donor._auto_lent
    # 0.5 GiB/min x 60 min = 30 GiB < 72 GiB: the grown-into room is not lent,
    # but a freed one is.
    _serve(node, donor, clock, 0.5, 5)
    node.grow(-20 * GIB)
    again = donor._auto_pass()
    assert again["reason"] == "room" and donor._auto_lent > lent
    assert node.usage <= 715 * GIB - 72 * GIB


def test_topup_needs_a_slope(tmp_path, logs):
    clock = FakeClock()
    node, donor = _ctx_donor(tmp_path, clock, usage=500 * GIB)
    donor._auto_pass()
    node.grow(-20 * GIB)
    assert donor._auto_pass()["reason"] == "slope_unknown"


def test_unknown_job_end_uses_the_horizon(tmp_path, logs):
    clock = FakeClock()
    node, donor = _ctx_donor(tmp_path, clock, environ={"SLURM_JOB_ID": "77"})
    donor._auto_pass()
    _serve(node, donor, clock, 0.4, 5)
    record = donor._auto_pass()
    # 0.4 GiB/min x 120 min horizon = 48 GiB < 72: not tripped.
    assert record["end_source"] == "horizon" and record["seconds_left"] == 7200
    assert record["reason"] != "growth_guard"


# -- co-located workers -----------------------------------------------------------


def test_colocated_workers_split_the_node(tmp_path, logs):
    clock = FakeClock()
    node = FakeNode(tmp_path / "node", usage=700 * GIB)
    # Two TP2 GEN workers on one node, two ranks each, one socket per rank.
    workers = [
        [donor_on(node, tmp_path, clock, rank=r, local_rank=r, local_size=2) for r in (0, 1)]
        for _ in range(2)
    ]
    first, second = workers
    # The first worker lends before the second one does: it still takes only its half.
    for donor in first + second:
        donor._auto_pass()
    lent = [[d._auto_lent for d in worker] for worker in workers]
    room = CGROUP - 4 * 36 * GIB - (700 + 4) * GIB  # 4 client buffers
    assert all(abs(x - room / 4) <= 2 * GIB for row in lent for x in row)
    assert node.usage <= CGROUP - 4 * 36 * GIB
    registry = json.loads((tmp_path / "registry.json").read_text())["units"]
    assert len(registry) == 4 and sum(u["weight"] for u in registry.values()) == 4
    assert sum(u["lent"] for u in registry.values()) == sum(map(sum, lent))


def test_registry_drops_other_jobs(tmp_path):
    path = tmp_path / "registry.json"
    path.write_text(json.dumps({"units": {"old": {"job": "1", "weight": 4, "lent": GIB}}}))
    units = shd.NodeRegistry(str(path), "2").put("me", {"weight": 2, "lent": 0})
    assert set(units) == {"me"}


def test_concurrent_passes_stay_under_budget(tmp_path, logs):
    clock = FakeClock()
    node = FakeNode(tmp_path / "node", usage=700 * GIB)
    donors = [
        donor_on(node, tmp_path, clock, rank=r, local_rank=r % 2, local_size=2) for r in range(4)
    ]
    results, errors = [None] * 4, []

    def run(index):
        try:
            results[index] = donors[index]._auto_pass()
        except Exception as error:  # noqa: BLE001 - reported below
            errors.append(error)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    assert not any(t.is_alive() for t in threads) and not errors
    assert all(r is not None and r["amount"] > 0 for r in results)
    assert node.usage <= CGROUP - 4 * 36 * GIB


def test_new_client_lend_keeps_its_transfer_buffer_out_of_the_buffer(tmp_path, logs):
    clock = FakeClock()
    node = FakeNode(tmp_path / "node", limit=1400 * GIB, usage=600 * GIB)
    donor = donor_on(node, tmp_path, clock, local_size=2, numa_nodes=(0,), late_mode="new_client")
    assert node.usage == 600 * GIB  # no startup client in new_client mode
    record = donor._auto_pass()
    assert record["amount"] > 0
    # The segment and its client's 1 GiB transfer buffer both fit above the buffer.
    assert node.usage <= 715 * GIB - 72 * GIB


def test_registry_falls_back_to_the_temporary_directory(tmp_path, logs, monkeypatch):
    fallback = tmp_path / "fallback"
    fallback.mkdir()
    monkeypatch.setattr(shd.tempfile, "gettempdir", lambda: str(fallback))
    clock = FakeClock()
    node = FakeNode(tmp_path / "node", limit=1400 * GIB, usage=600 * GIB)
    donor = donor_on(node, tmp_path, clock, local_size=2, numa_nodes=(0,))
    # Make the primary location unusable: its directory does not exist.
    donor._probes.registry_path = str(tmp_path / "missing" / "registry.json")
    donor._auto_register_unit()
    assert donor._probes.registry_path == str(fallback / "registry.json")
    record = donor._auto_pass()
    assert record["reason"] == "room" and record["amount"] > 0
    units = json.loads((fallback / "registry.json").read_text())["units"]
    assert [u["lent"] for u in units.values()] == [donor._auto_lent]


def test_no_registry_lends_nothing(tmp_path, logs, monkeypatch):
    monkeypatch.setattr(shd.tempfile, "gettempdir", lambda: str(tmp_path / "also-missing"))
    clock = FakeClock()
    node = FakeNode(tmp_path / "node", limit=1400 * GIB, usage=600 * GIB)
    donor = donor_on(node, tmp_path, clock, local_size=2, numa_nodes=(0,))
    usage = node.usage
    donor._probes.registry_path = str(tmp_path / "missing" / "registry.json")
    donor._auto_register_unit()
    record = donor._auto_pass()
    assert record["reason"] == "registry_unavailable" and record["amount"] == 0
    assert node.usage == usage
    assert any(level == "error" and "AUTO_FAILED registry" in msg for level, msg in logs)


def test_close_leaves_the_registry(tmp_path, logs):
    clock = FakeClock()
    node = FakeNode(tmp_path / "node", limit=1400 * GIB, usage=600 * GIB)
    donor = donor_on(node, tmp_path, clock, local_size=2, numa_nodes=(0,))
    donor._auto_pass()
    assert donor.close()
    registry = json.loads((tmp_path / "registry.json").read_text())["units"]
    assert registry == {}
    again = donor_on(node, tmp_path, clock, local_size=2, numa_nodes=(0,))
    record = again._auto_pass()
    assert record["active_gpus"] == 2 and record["lent_node"] == 0


# -- overrides ---------------------------------------------------------------------


def _config(late="auto", late_auto=None):
    return SimpleNamespace(late_size_per_node=late, late_auto=late_auto)


def test_override_precedence():
    auto_cfg = SimpleNamespace(head_hosts=["h1"])
    sizing = shd._late_sizing
    # Fixed GiB from the environment wins over everything.
    assert sizing(_config(), {shd.LATE_GIB_ENV: "160", shd.AUTO_ENV: "1"})[2] == 160 * GIB
    # AUTO=1 selects the measured mode on any config, keeping its head hosts.
    late_auto, measured, late, source = sizing(_config("160GiB"), {shd.AUTO_ENV: "1"})
    assert measured is not None and late_auto is None and late == 0
    late_auto, measured, _, _ = sizing(_config("auto", auto_cfg), {shd.AUTO_ENV: "on"})
    assert measured.head_hosts == ("h1",)
    # auto without late_auto is the measured mode; AUTO=0 turns it off.
    assert sizing(_config(), {})[1] is not None
    assert sizing(_config(), {shd.AUTO_ENV: "0"})[1:3] == (None, 0)
    with pytest.raises(ValueError):
        sizing(_config(), {shd.AUTO_ENV: "maybe"})


def test_measured_env_overrides():
    m = shd.measured_auto_from_env(
        {
            shd.BUFFER_ENV: "8",
            shd.HEAD_BUFFER_ENV: "40",
            shd.MAX_LENT_ENV: "100",
            shd.TOPUP_ENV: "60",
            shd.HEAD_HOSTS_ENV: "hecate0027, hecate0028",
        }
    )
    assert m.buffer_per_gpu == 8 * GIB and m.head_buffer_per_gpu == 40 * GIB
    assert m.max_lent_per_node == 100 * GIB and m.topup_seconds == 60
    assert m.head_hosts == ("hecate0027", "hecate0028")
    assert shd.measured_auto_from_env({}) == shd.MeasuredAuto()
    for bad in (
        {shd.BUFFER_ENV: "-1"},
        {shd.TOPUP_ENV: "0"},
        {shd.BUFFER_ENV: "nan"},
        {shd.SAMPLE_ENV: "600", shd.SLOPE_WINDOW_ENV: "900"},
    ):
        with pytest.raises(ValueError):
            shd.measured_auto_from_env(bad)


def test_buffer_and_max_lent_overrides_apply(tmp_path, logs):
    clock = FakeClock()
    node = gen_node(tmp_path)
    measured = shd.measured_auto_from_env({shd.BUFFER_ENV: "8", shd.MAX_LENT_ENV: "100"})
    donors = gen_worker(node, tmp_path, clock, measured=measured)
    for donor in donors:
        donor._auto_pass()
    assert sum(d._auto_lent for d in donors) == 100 * GIB
    clock.advance(300)
    for donor in donors:
        donor._auto_sample()
    clock.advance(300)
    for donor in donors:
        donor._auto_sample()
        assert donor._auto_pass()["reason"] in ("max_lent", "slope_unknown")


# -- timing ------------------------------------------------------------------------


def test_context_server_waits_for_traffic(tmp_path, logs, monkeypatch):
    monkeypatch.setattr(shd, "TRAFFIC_POLL_S", 0.01)
    node = FakeNode(tmp_path / "node", limit=1400 * GIB, usage=600 * GIB)
    measured = shd.MeasuredAuto(topup_seconds=0.05, sample_seconds=0.01, min_segment=GIB)
    donor = donor_on(node, tmp_path, time.time, local_size=2, numa_nodes=(0,), measured=measured)
    polled, allow = threading.Event(), threading.Event()

    def probe():
        polled.set()
        return allow.is_set()

    donor.wait_traffic = True
    donor.traffic_probe = probe
    try:
        donor.register_late()
        # The lending thread is at the traffic gate: nothing is lent yet.
        assert polled.wait(5)
        assert donor._auto_lent == 0 and not records(logs, "AUTO_LEND")
        allow.set()
        deadline = time.time() + 5
        while donor._auto_lent == 0 and time.time() < deadline:
            time.sleep(0.01)
    finally:
        assert donor.close()
    assert donor._auto_lent > 0
    assert any("first request" in msg for _, msg in logs)


def test_traffic_wait_times_out_and_lends(tmp_path, logs, monkeypatch):
    monkeypatch.setattr(shd, "TRAFFIC_POLL_S", 0.001)
    clock = FakeClock()
    node = FakeNode(tmp_path / "node", limit=1400 * GIB, usage=600 * GIB)
    donor = donor_on(node, tmp_path, clock, local_size=2, numa_nodes=(0,))
    donor.wait_traffic = True

    def probe():
        clock.advance(1000)
        return False

    donor.traffic_probe = probe
    try:
        donor.register_late()
        deadline = time.time() + 5
        while donor._auto_lent == 0 and time.time() < deadline:
            time.sleep(0.01)
    finally:
        assert donor.close()
    assert donor._auto_lent > 0
    assert any("no request within 1800 s" in msg for _, msg in logs)


def test_traffic_wait_ends_on_close(tmp_path, logs):
    clock = FakeClock()
    node = FakeNode(tmp_path / "node", limit=1400 * GIB, usage=600 * GIB)
    donor = donor_on(node, tmp_path, clock, local_size=2, numa_nodes=(0,))
    usage = node.usage
    donor.wait_traffic = True
    donor.traffic_probe = lambda: False
    donor._stop.set()
    assert donor._auto_wait_traffic() is False
    donor._auto_loop()
    assert donor._auto_lent == 0 and node.usage == usage


def test_generation_server_lends_after_warmup_in_background(tmp_path, logs):
    node = gen_node(tmp_path)
    measured = shd.MeasuredAuto(topup_seconds=0.05, sample_seconds=0.01, min_segment=GIB)
    donors = gen_worker(node, tmp_path, time.time, measured=measured)
    for donor in donors:
        donor.register_late()  # returns at once; lending runs on a thread
    deadline = time.time() + 5
    while not all(d._auto_lent > 0 for d in donors) and time.time() < deadline:
        time.sleep(0.01)
    for donor in donors:
        assert donor.close()
    assert all(d._auto_lent > 0 for d in donors)
    assert records(logs, "AUTO_START")
    assert node.usage <= CGROUP - 4 * 36 * GIB


# -- backward compatibility ------------------------------------------------------


def test_fixed_and_planned_paths_are_unchanged(tmp_path):
    assert shd._late_sizing(_config("96GiB"), {})[0:3] == (None, None, 96 * GIB)
    planned = SimpleNamespace(
        host_budget_per_gpu="357.5GiB",
        store_per_gpu="290GiB",
        process_peak_per_gpu="36GiB",
        head_process_peak_per_gpu="64.1GiB",
        margin_per_gpu="2560MiB",
        granularity="1GiB",
        head_hosts=None,
        job_memory_limit=None,
    )
    late_auto, measured, late, source = shd._late_sizing(_config("auto", planned), {})
    assert measured is None and late == 0 and source == "late_auto"
    assert late_auto.process_peak_per_gpu == 36 * GIB
    # AUTO=0 is a kill switch for the measured mode only.
    assert shd._late_sizing(_config("auto", planned), {shd.AUTO_ENV: "0"})[0] is not None
    # A fixed donor never touches the node registry.
    node = gen_node(tmp_path)
    donor = shd.StoreHostDonor(
        master="10.0.0.1:8700",
        shares=shd.plan_shares(0, 2 * GIB, [0, 1], 0, 1),
        rank=0,
        local_rank=0,
        local_size=1,
        device_name="mlx5_0",
        local_hostname="10.0.0.2",
        numa_nodes=[0, 1],
        probes=shd.MeasuredProbes(registry_path=str(tmp_path / "registry.json")),
        job_memory=node.job_memory,
        node_lock=None,
        store_factory=lambda: ChargingStore(node),
        bind_thread=lambda numa: None,
        meminfo=node.numa,
    )
    donor.start_startup()
    donor.register_late()
    assert donor.summary()["registered_bytes"] == 2 * GIB
    assert not (tmp_path / "registry.json").exists()
    assert donor.close()


def test_executor_wires_the_traffic_probe():
    from tensorrt_llm._torch.pyexecutor import py_executor_creator as creator

    class Dist:
        rank = 0

        def allgather(self, obj):
            return [obj]

    calls = []
    for connector, wait in ((object(), True), (None, False)):
        donor = SimpleNamespace(
            measured=shd.MeasuredAuto(),
            wait_startup=lambda: calls.append("wait"),
            register_late=lambda: calls.append("late"),
            late_delay_seconds=0.0,
        )
        executor = SimpleNamespace(
            llm_args=SimpleNamespace(kv_connector_config=connector), num_fetch_requests=0
        )
        creator._register_store_host_donor_late(executor, donor, Dist())
        assert donor.wait_traffic is wait and donor.traffic_probe() is False
        executor.num_fetch_requests = 3
        assert donor.traffic_probe() is True
    assert calls == ["wait", "late", "wait", "late"]


def test_first_lend_waits_for_the_growth_slope(tmp_path, logs):
    clock = FakeClock()
    node = FakeNode(tmp_path / "node", limit=1400 * GIB, usage=600 * GIB)
    environ = {"SLURM_JOB_END_TIME": str(int(clock() + 3600))}
    donor = donor_on(
        node,
        tmp_path,
        clock,
        local_size=2,
        numa_nodes=(0,),
        first_needs_slope=True,
        environ=environ,
    )
    assert donor._auto_pass()["reason"] == "slope_unknown" and donor._auto_lent == 0
    # Slow growth: the second pass, with a slope, lends.
    _serve(node, donor, clock, 0.2, 5)
    assert donor._auto_pass()["reason"] == "room" and donor._auto_lent > 0


def test_first_lend_is_guarded_on_a_growing_head_node(tmp_path, logs):
    clock = FakeClock()
    node = FakeNode(tmp_path / "node", limit=1400 * GIB, usage=500 * GIB)
    environ = {"SLURM_JOB_END_TIME": str(int(clock() + 3600))}
    donor = donor_on(
        node,
        tmp_path,
        clock,
        local_size=2,
        numa_nodes=(0,),
        head=True,
        first_needs_slope=True,
        environ=environ,
    )
    donor._auto_pass()
    # 2.6 GiB/min for the 55 min left (143 GiB) > the 128 GiB head buffer: never lends.
    _serve(node, donor, clock, 2.6, 5)
    assert donor._auto_pass()["reason"] == "growth_guard" and donor._auto_lent == 0


def test_production_default_holds_the_first_lend(tmp_path, logs):
    assert shd.MeasuredAuto().first_lend_needs_slope is True
    clock = FakeClock()
    node = FakeNode(tmp_path / "node", limit=1400 * GIB, usage=600 * GIB)
    shares = shd.plan_shares(0, 0, [0], 0, 2)
    donor = shd.StoreHostDonor(
        master="10.0.0.1:8700",
        shares=shares,
        rank=0,
        local_rank=0,
        local_size=2,
        device_name="mlx5_0",
        local_hostname="10.0.0.2",
        numa_nodes=[0],
        measured=shd.MeasuredAuto(),
        probes=shd.MeasuredProbes(
            meminfo_path=str(node.root / "meminfo"),
            gpu_proc_dir=str(node.root / "gpus"),
            dev_dir=str(node.root / "dev"),
            registry_path=str(tmp_path / "registry.json"),
            environ={"SLURM_JOB_ID": node.job},
            clock=clock,
            scontrol=lambda job: None,
        ),
        head_role=lambda master, hosts: (False, "test"),
        job_memory=node.job_memory,
        node_lock=None,
        store_factory=lambda: ChargingStore(node),
        bind_thread=lambda numa: None,
        meminfo=node.numa,
    )
    donor.start_startup()
    donor.wait_startup()
    record = donor._auto_pass()
    assert record["reason"] == "slope_unknown" and donor._auto_lent == 0
