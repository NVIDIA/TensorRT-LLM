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
"""Unit tests for the GPU memory (HBM) lend of the store host donor."""

import functools
import json
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor import store_hbm_lend as lend_mod
from tensorrt_llm._torch.pyexecutor import store_host_donor as shd

pytestmark = pytest.mark.cpu_only

GIB = 1 << 30
MASTER = "10.0.0.1:8700"


class LendStore:
    """A Mooncake client that can mount caller memory; records every call."""

    instances = []

    def __init__(self, *, mount_ret=0, unmount_status=0, journal=None):
        self.calls = [] if journal is None else journal
        self.mount_ret = mount_ret
        self.unmount_status = unmount_status
        LendStore.instances.append(self)

    def setup(self, *args):
        self.calls.append(("setup", *args))
        return 0

    def get_hostname(self):
        return "10.0.0.2:12345"

    def mount_buffer(self, pointer, size, protocol="tcp", location="*"):
        self.calls.append(("mount_buffer", pointer, size, protocol, location))
        if self.mount_ret:
            return {"ret": self.mount_ret, "segment_ids": []}
        return {"ret": 0, "segment_ids": ["seg-a", "seg-b"]}

    def unmount_segment(self, ids, grace):
        self.calls.append(("unmount_segment", list(ids), grace))
        return self.unmount_status

    def close(self):
        self.calls.append(("close",))
        return 0


class OldStore:
    """A Mooncake client from a release without mount_buffer."""

    def __init__(self):
        self.calls = []

    def setup(self, *args):
        self.calls.append(("setup", *args))
        return 0


class Buffer:
    def __init__(self, size, device, protocol, journal):
        self.size, self.device, self.protocol = size, device, protocol
        self.pointer = 0x7F0000000000
        self.journal = journal
        journal.append(("allocate", size, device, protocol))

    def free(self):
        self.journal.append(("free", self.pointer))
        self.pointer = 0


@pytest.fixture(autouse=True)
def reset(monkeypatch):
    LendStore.instances = []
    monkeypatch.setattr(shd, "_ACTIVE", None)
    monkeypatch.setattr(shd, "NODE_LOCK_PATH", None)
    monkeypatch.setattr(
        shd, "_default_job_memory", lambda: {"limit": None, "nonfile": None, "path": None}
    )
    for name in (shd.DISABLE_ENV, shd.MASTER_ENV, shd.CONFIG_PATH_ENV, shd.STORE_EGM_ENV):
        monkeypatch.delenv(name, raising=False)
    yield
    shd._ACTIVE = None


@pytest.fixture
def logs(monkeypatch):
    lines = []
    for level in ("info", "warning", "error"):
        monkeypatch.setattr(
            lend_mod.logger, level, lambda msg, _level=level: lines.append((_level, msg))
        )
    return lines


def tagged(lines, tag):
    prefix = f"{lend_mod.LOG_TAG}_{tag} "
    decoder = json.JSONDecoder()
    return [decoder.raw_decode(msg[len(prefix) :])[0] for _, msg in lines if msg.startswith(prefix)]


def make_lend(journal, *, store=None, free=80 * GIB, size=16 * GIB, protocol="rdma"):
    frees = [free, free - size]
    return lend_mod.StoreHbmLend(
        size=size,
        master=MASTER,
        protocol=protocol,
        device_name="mlx5_0,mlx5_1",
        metadata_server="P2PHANDSHAKE",
        rank=3,
        store_factory=store or functools.partial(LendStore, journal=journal),
        allocate=lambda s, d, p: Buffer(s, d, p, journal),
        free_memory=lambda device: frees.pop(0) if len(frees) > 1 else frees[0],
        device=lambda: 1,
    )


def test_mount_then_retire_in_order(logs):
    journal = []
    lend = make_lend(journal)
    assert lend.mount("10.0.0.2")
    assert lend.mounted and lend.segment_ids == ["seg-a", "seg-b"]
    assert journal == [
        ("setup", "10.0.0.2", "P2PHANDSHAKE", 0, 0, "rdma", "mlx5_0,mlx5_1", MASTER),
        ("allocate", 16 * GIB, 1, "rdma"),
        ("mount_buffer", 0x7F0000000000, 16 * GIB, "rdma", "*"),
    ]
    (record,) = tagged(logs, "MOUNTED")
    assert record["bytes"] == 16 * GIB and record["device"] == 1
    assert record["segment"] == "10.0.0.2:12345" and record["free_after"] == 64 * GIB

    journal.clear()
    assert lend.close()
    # The segment goes before the memory behind it, the client last.
    assert journal == [
        ("unmount_segment", ["seg-a", "seg-b"], 0),
        ("free", 0x7F0000000000),
        ("close",),
    ]
    assert not lend.mounted and lend.buffer is None
    assert tagged(logs, "RETIRED")
    assert lend.close()
    with pytest.raises(RuntimeError, match="already lent"):
        lend.store = object()
        lend.mount("10.0.0.2")


def test_a_release_without_mount_buffer_lends_nothing(logs):
    journal = []
    old = OldStore()
    lend = make_lend(journal, store=lambda: old)
    assert lend.mount("10.0.0.2") is False
    # Nothing was allocated and no client was set up.
    assert journal == [] and old.calls == []
    (record,) = tagged(logs, "UNSUPPORTED")
    assert record["rank"] == 3 and "mount_buffer" in record["reason"]
    assert lend.close()


def test_too_little_free_memory_fails_before_anything_is_taken():
    journal = []
    lend = make_lend(journal, free=20 * GIB)
    with pytest.raises(RuntimeError, match="free_gpu_memory_fraction"):
        lend.mount("10.0.0.2")
    assert journal == [] and lend.store is None


def test_a_failed_mount_releases_the_buffer_and_the_client():
    journal = []
    lend = make_lend(journal, store=functools.partial(LendStore, mount_ret=-600, journal=journal))
    with pytest.raises(RuntimeError, match="status -600"):
        lend.mount("10.0.0.2")
    assert journal[-2:] == [("free", 0x7F0000000000), ("close",)]
    assert lend.store is None and lend.buffer is None and lend.close()


def test_a_failed_unmount_keeps_the_memory(logs):
    journal = []
    lend = make_lend(journal)
    assert lend.mount("10.0.0.2")
    store = lend.store
    store.unmount_status = -1
    journal.clear()
    assert not lend.close()
    assert journal == [("unmount_segment", ["seg-a", "seg-b"], 0)]
    assert lend.buffer is not None and lend.mounted
    assert tagged(logs, "RETIRE_FAILED")
    # A later close can still retire it.
    store.unmount_status = 0
    assert lend.close() and lend.buffer is None


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(size=0), "positive"),
        (dict(protocol="tcp"), "cannot reach GPU memory"),
    ],
)
def test_invalid_lends(kwargs, match):
    base = dict(
        size=GIB,
        master=MASTER,
        protocol="rdma",
        device_name="",
        metadata_server="P2PHANDSHAKE",
        rank=0,
        store_factory=LendStore,
    )
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        lend_mod.StoreHbmLend(**base)


# -- through the donor ----------------------------------------------------------------


def config(**fields):
    base = dict(
        master=None,
        protocol="rdma",
        device_name="mlx5_0",
        startup_size_per_node=0,
        late_size_per_node=0,
        numa_nodes=None,
        local_buffer_size="1GiB",
        local_hostname="10.0.0.2",
        late_mode="auto",
        late_delay_seconds=0.0,
        late_auto=None,
        hbm_lend_size_per_gpu="16GiB",
    )
    base.update(fields)
    return SimpleNamespace(**base)


@pytest.fixture
def donor_journal(monkeypatch):
    journal = []
    monkeypatch.setenv(shd.MASTER_ENV, MASTER)

    def lend(**kwargs):
        return lend_mod.StoreHbmLend(
            **kwargs,
            allocate=lambda s, d, p: Buffer(s, d, p, journal),
            free_memory=lambda device: 80 * GIB,
            device=lambda: 0,
        )

    monkeypatch.setattr(shd, "StoreHbmLend", lend)
    return journal


def start(cfg):
    return shd.start_store_host_donor(
        cfg,
        rank=0,
        local_rank=0,
        local_size=4,
        numa_nodes=[0, 1],
        store_factory=LendStore,
        bind_thread=lambda node: None,
    )


def test_the_donor_lends_and_retires_the_gpu_memory(donor_journal):
    donor = start(config())
    assert donor.hbm_lend is not None and not donor.hbm_lend.mounted
    donor.register_late()
    donor.lend_hbm()
    (store,) = LendStore.instances
    assert store.calls[0] == ("setup", "10.0.0.2", "P2PHANDSHAKE", 0, 0, "rdma", "mlx5_0", MASTER)
    assert ("allocate", 16 * GIB, 0, "rdma") in donor_journal
    assert shd.close_active_donor() and shd.active_donor() is None
    assert donor_journal[-1] == ("free", 0x7F0000000000)


def test_a_retained_gpu_lend_keeps_the_donor(donor_journal):
    donor = start(config())
    donor.lend_hbm()
    donor.hbm_lend.store.unmount_status = -1
    assert not shd.close_active_donor()
    assert shd.active_donor() is donor
    donor.hbm_lend.store.unmount_status = 0
    assert shd.close_active_donor()


def test_no_gpu_lend_by_default(donor_journal):
    donor = start(config(hbm_lend_size_per_gpu=0, startup_size_per_node="2GiB"))
    assert donor.hbm_lend is None
    donor.lend_hbm()
    assert donor_journal == []
    shd.close_active_donor()


def test_a_tcp_store_cannot_take_gpu_memory(donor_journal):
    with pytest.raises(ValueError, match="cannot reach GPU memory"):
        start(config(protocol="tcp"))
    assert shd.active_donor() is None


def test_llm_args_hbm_lend():
    pytest.importorskip("torch")
    from tensorrt_llm.llmapi.llm_args import StoreHostDonorConfig

    parsed = StoreHostDonorConfig(hbm_lend_size_per_gpu="64GiB")
    assert parsed.hbm_lend_size_per_gpu == "64GiB"
    assert StoreHostDonorConfig(startup_size_per_node="1GiB").hbm_lend_size_per_gpu == 0
    for bad in ("64GB", -1):
        with pytest.raises(ValueError):
            StoreHostDonorConfig(hbm_lend_size_per_gpu=bad)


def test_the_installed_mooncake_is_detected():
    store_module = pytest.importorskip("mooncake.store")
    store = store_module.MooncakeDistributedStore()
    supported = lend_mod.store_supports_mount_buffer(store)
    if supported:
        pytest.skip("the installed Mooncake can mount caller memory")
    lend = lend_mod.StoreHbmLend(
        size=GIB,
        master=MASTER,
        protocol="rdma",
        device_name="",
        metadata_server="P2PHANDSHAKE",
        rank=0,
        store_factory=lambda: store,
        allocate=lambda *args: pytest.fail("nothing may be allocated"),
        free_memory=lambda device: pytest.fail("no GPU query"),
        device=lambda: 0,
    )
    assert lend.mount("127.0.0.1") is False
