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
"""The blob backend between two real staging lenders.

One manager publishes a request's KV blocks through its lender's staging slots; another fetches
them into a request of its own. A test lease holder drives the lender as the runtime does: lend,
wait for the view, submit it to the backend as it stands, wait until the attempt quiesces, mark
what arrived, release. The store is the in-memory one, or with ``TRTLLM_MOONCAKE_REAL=1`` a real
Mooncake pool (``TRTLLM_MOONCAKE_PROTOCOL``, default rdma). Allocates device pools.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, replace

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.disaggregation.base.backend import MemoryKind
from tensorrt_llm._torch.disaggregation.blob.backend import BlobBackend, BlobConfig, RegionBinding
from tensorrt_llm._torch.disaggregation.blob.keys import encode_key
from tensorrt_llm._torch.disaggregation.blob.mooncake import MooncakeDriver
from tensorrt_llm._torch.disaggregation.blob.mooncake_setup import MooncakeConfig, open_sdk
from tensorrt_llm._torch.disaggregation.testing.memory_store import (
    DEFAULT_CAPABILITIES,
    Commit,
    MemoryDriver,
    MemoryStore,
)
from tensorrt_llm._torch.disaggregation.testing.object_trailer import Checksum, encode_trailer
from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import (
    LeaseView,
    ResumeBounds,
    StagingOptions,
    attach_lender,
)

# An exception that escapes a thread fails the test.
pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="allocates KV cache pools"),
    pytest.mark.filterwarnings("error::pytest.PytestUnhandledThreadExceptionWarning"),
]

TPB = 32
PROMPT = list(range(1000, 1097))  # three whole blocks and one token
END = 96
BLOCKS = END // TPB
SOURCE, TARGET = 1, 2
SCOPE = b"blob-backend-round-trip"
KEY_PREFIX = "round-trip"
WAIT_S = 60.0
PROTOCOL = os.environ.get("TRTLLM_MOONCAKE_PROTOCOL", "rdma")
DEVICES = os.environ.get("TRTLLM_MOONCAKE_DEVICES", "")
ONE_ROW_PER_CALL = replace(DEFAULT_CAPABILITIES, max_rows_per_op=1)


class Wake:
    """The ``WakeFn`` a test hands the backend, with bounded waits on what it signals."""

    def __init__(self) -> None:
        self._woken = threading.Condition()

    def __call__(self) -> None:
        with self._woken:
            self._woken.notify_all()

    def settled(self, attempt):
        """The attempt's outcome, once it also quiesced."""
        with self._woken:
            done = self._woken.wait_for(
                lambda: attempt.poll() is not None and attempt.quiesced(), WAIT_S
            )
        assert done, "the attempt did not settle"
        return attempt.poll()


class Stores:
    """Drivers over one store: the in-memory one, or with ``master`` a real Mooncake pool. A test
    closes each driver it opened as its owner does, last; ``close`` closes those a failed test left
    open, only once every backend built over them shut down cleanly. ``cancel_effective=False``
    makes the in-memory drivers ignore a cancel."""

    def __init__(self, master: str | None = None, cancel_effective: bool = True) -> None:
        self._store = MemoryStore() if master is None else None
        self._master = master
        self._cancel_effective = cancel_effective
        self._drivers = []
        self.backends = []

    def open_driver(self):
        """A driver that moves every row on its own, several at once: the in-memory one takes one
        row per operation, Mooncake cuts each operation into one-row parts for two workers."""
        if self._store is not None:
            driver = MemoryDriver(self._store, capabilities=ONE_ROW_PER_CALL, background=True)
            driver.set_cancel_effective(self._cancel_effective)
        else:
            sdk = open_sdk(MooncakeConfig(self._master, 0, PROTOCOL, DEVICES, local_buffer_size=0))
            driver = MooncakeDriver(sdk, workers=2, rows_per_call=1)
        self._drivers.append(driver)
        return driver

    def commit(self, object_key: np.ndarray, data: bytes) -> None:
        """Commits ``data`` straight into the in-memory store, with the driver's trailer."""
        key = encode_key(KEY_PREFIX, object_key.tobytes())
        trailer = encode_trailer(key.encode(), len(data), checksum=Checksum.CRC32, payload=data)
        assert self._store.commit(key, data, trailer) is Commit.STORED

    def close(self) -> None:
        if all(backend.shutdown(WAIT_S) == frozenset() for backend in self.backends):
            for driver in self._drivers:
                driver.close()


@pytest.fixture(params=["memory", "mooncake"])
def stores(request, tmp_path):
    if request.param == "memory":
        yield Stores()
        return
    if os.environ.get("TRTLLM_MOONCAKE_REAL") != "1":
        pytest.skip("needs a real Mooncake pool; set TRTLLM_MOONCAKE_REAL=1")
    from test_common.mooncake_utils import running_master_on_free_ports, running_segment_provider

    with running_master_on_free_ports(str(tmp_path), protocol=PROTOCOL) as master:
        with running_segment_provider(
            master.address,
            str(tmp_path),
            segment_size=256 << 20,
            protocol=PROTOCOL,
            device_name=DEVICES,
        ):
            opened = Stores(master=master.address)
            yield opened
            opened.close()


def attach(mgr):
    return attach_lender(mgr, scope=SCOPE, staging=StagingOptions(END))


def ready_view(lease, mgr, tries: int = 3):
    for _ in range(tries):
        mgr._stream.synchronize()
        view = lease.poll()
        if view is not None:
            return view
    raise AssertionError(f"the lease never became ready: {lease.failure}")


@dataclass(frozen=True)
class HostRegion:
    """A lender region as the contract's ``SlotRegion``. The lender's own regions do not name their
    memory kind, which ``SlotRegion`` requires; the lender stages in host memory."""

    layout_label: str
    address: int
    nbytes: int
    slot_bytes: int
    slot_count: int
    memory_kind: MemoryKind = MemoryKind.HOST


def build(stores: Stores, lender, wake: Wake):
    """A backend over ``lender``'s regions, built on this thread: a driver, the lender's hold, the
    regions registered under it, and the backend. Returns the binding, whose ``release_hold``
    gives the hold back once the backend shut down, before its driver is closed, and the backend."""
    regions = [
        HostRegion(r.layout_label, r.address, r.nbytes, r.slot_bytes, r.slot_count)
        for r in lender.regions
    ]
    binding = RegionBinding(regions, stores.open_driver(), lender.hold_regions())
    binding.register()
    backend = BlobBackend(binding, wake, BlobConfig(key_prefix=KEY_PREFIX))
    stores.backends.append(backend)
    return binding, backend


def publish(stores, kit, mgr, select=None):
    """Publishes the source request's blocks (those ``select`` keeps); returns the view lent."""
    source = kit.published(mgr, SOURCE, PROMPT)
    lender, wake = attach(mgr), Wake()
    binding, backend = build(stores, lender, wake)
    lease = lender.lend_read(source, 0, END)
    view = ready_view(lease, mgr)
    submitted = view if select is None else LeaseView(tuple(g.select(select) for g in view.groups))
    outcome = wake.settled(backend.publish(submitted, None))
    assert outcome.fault is None
    assert all(mask.all() for mask in outcome.delivered_masks)
    lease.release()
    assert backend.shutdown(WAIT_S) == frozenset()
    binding.release_hold()
    binding.driver.close()
    return source, view


def fetch(stores, kit, mgr):
    """Fetches every block of the target request; returns the target, lender, lease, outcome,
    backend and binding, the lease not yet marked."""
    target = kit.admitted(mgr, TARGET, PROMPT)
    lender, wake = attach(mgr), Wake()
    binding, backend = build(stores, lender, wake)
    lease = lender.lend_write(target, 0, END)
    view = lease.poll()
    kit.fill_sentinel(mgr, target)
    outcome = wake.settled(backend.fetch(view, None))
    return target, lender, lease, outcome, backend, binding


def test_publish_then_fetch_restores_pages(kit, real_manager, stores) -> None:
    with real_manager() as mgr_a, real_manager() as mgr_b:
        source, _ = publish(stores, kit, mgr_a)
        target, lender, lease, outcome, backend, binding = fetch(stores, kit, mgr_b)
        assert outcome.fault is None
        lease.mark_arrived([np.array(mask) for mask in outcome.delivered_masks])
        mgr_b._stream.synchronize()
        assert lender.resume_bounds(target) == ResumeBounds(END, 0)
        for ordinal in range(BLOCKS):
            got = kit.digest(kit.page(mgr_b, target, 0, ordinal))
            assert got == kit.digest(kit.page(mgr_a, source, 0, ordinal)), "a fetched page differs"
        lease.release()
        assert backend.shutdown(WAIT_S) == frozenset()
        binding.release_hold()
        binding.driver.close()


def test_a_partial_fetch_resumes_only_up_to_its_first_undelivered_block(kit, real_manager) -> None:
    # Each row's call runs on a thread of its own, in any order. A later row's fault stops the
    # fetch, and the cancel it forwards could end the first row before its thread took it; the
    # driver ignores it, so the first row always lands.
    stores = Stores(cancel_effective=False)
    with real_manager() as mgr_a, real_manager() as mgr_b:
        first_block_only = np.arange(BLOCKS) == 0
        source, view = publish(stores, kit, mgr_a, select=first_block_only)
        for object_key in view.groups[0].object_keys[1:]:  # the later blocks are the wrong size
            stores.commit(object_key, b"short")
        target, lender, lease, outcome, backend, binding = fetch(stores, kit, mgr_b)
        assert outcome.delivered_masks[0].tolist() == [True, False, False]
        assert "object is 5 bytes" in outcome.fault
        lease.mark_arrived([np.array(mask) for mask in outcome.delivered_masks])
        mgr_b._stream.synchronize()
        assert lender.resume_bounds(target) == ResumeBounds(TPB, 0)  # usable up to the first gap
        sentinel = kit.digest(bytes([kit.SENTINEL]) * kit.DevicePages(mgr_b).page_bytes(0))
        pages = [kit.digest(kit.page(mgr_b, target, 0, ordinal)) for ordinal in range(BLOCKS)]
        assert pages == [kit.digest(kit.page(mgr_a, source, 0, 0)), sentinel, sentinel]
        lease.release()
        assert backend.shutdown(WAIT_S) == frozenset()
        binding.release_hold()
        binding.driver.close()
