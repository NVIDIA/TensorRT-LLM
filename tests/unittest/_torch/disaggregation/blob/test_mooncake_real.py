# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The blob backend over a real Mooncake pool: a master, and a separate process lending memory.

The tested clients lend nothing (``global_segment_size`` 0), so every object sits in the other
process's segment and every transfer goes through the transfer engine; their ``local_buffer_size``
is 0, so a call that needed the SDK's copying path would fail. Runs only with
``TRTLLM_MOONCAKE_REAL=1``, on a host with ``mooncake_master`` and ``mooncake.store``. The
transport is ``TRTLLM_MOONCAKE_PROTOCOL`` (default rdma) over ``TRTLLM_MOONCAKE_DEVICES`` (default:
the host's fastest active InfiniBand devices). The shared ``driver_conformance`` checks run here
too, each over a fresh client and under a key prefix of its own.
"""

from __future__ import annotations

import contextlib
import importlib.util
import itertools
import os
import shutil
import threading

import driver_conformance
import numpy as np
import pytest
from cpu_fixtures import WAIT_S, Endpoint, store_key
from driver_conformance import DriverConformance
from test_mooncake_driver import SHORT_OBJECT_LANDS

from tensorrt_llm._torch.disaggregation.base.backend import Outcome
from tensorrt_llm._torch.disaggregation.blob.mooncake import MooncakeDriver
from tensorrt_llm._torch.disaggregation.blob.mooncake_setup import MooncakeConfig, open_sdk

pytestmark = pytest.mark.skipif(
    os.environ.get("TRTLLM_MOONCAKE_REAL") != "1",
    reason="needs a real Mooncake pool; set TRTLLM_MOONCAKE_REAL=1",
)

PROTOCOL = os.environ.get("TRTLLM_MOONCAKE_PROTOCOL", "rdma")
DEVICES = os.environ.get("TRTLLM_MOONCAKE_DEVICES", "")
POOL_BYTES = 1 << 30
SLOT_BYTES = 64 << 10


def _require_mooncake() -> None:
    if shutil.which("mooncake_master") is None:
        pytest.skip("mooncake_master is not on PATH")
    try:
        found = importlib.util.find_spec("mooncake.store") is not None
    except ImportError:
        found = False
    if not found:
        pytest.skip("mooncake.store does not import")


def _running_pool(run_dir: str, segment_size: int) -> tuple[contextlib.ExitStack, str]:
    """A master and a segment provider: the stack that stops them, and the master's address."""
    from test_common.mooncake_utils import running_master_on_free_ports, running_segment_provider

    stack = contextlib.ExitStack()
    # Ratio 0: the master evicts only above its high watermark, never just because a put failed.
    master = stack.enter_context(
        running_master_on_free_ports(run_dir, protocol=PROTOCOL, eviction_ratio=0.0)
    )
    stack.enter_context(
        running_segment_provider(
            master.address,
            run_dir,
            segment_size=segment_size,
            protocol=PROTOCOL,
            device_name=DEVICES,
        )
    )
    return stack, master.address


@pytest.fixture(scope="module")
def pool(tmp_path_factory):
    _require_mooncake()
    stack, address = _running_pool(str(tmp_path_factory.mktemp("pool")), POOL_BYTES)
    with stack:
        yield address


class Clients:
    """SDK clients of one master that lend nothing, and the endpoints built over them.

    ``close`` closes the clients only once every endpoint's shutdown returns no labels: a backend
    thread still inside the SDK would crash on a closed client, so a failed test leaves them open.
    """

    def __init__(self, master: str) -> None:
        self.master = master
        self.opened = []
        self.endpoints: list[Endpoint] = []

    def __call__(self):
        self.opened.append(open_sdk(client_config(self.master)))
        return self.opened[-1]

    def endpoint(self, names, *, workers: int = 2, **options) -> Endpoint:
        options = {"slot_bytes": SLOT_BYTES, **options}
        return self.add(Endpoint(MooncakeDriver(self()), names, workers=workers, **options))

    def add(self, endpoint: Endpoint) -> Endpoint:
        self.endpoints.append(endpoint)
        return endpoint

    def close(self) -> None:
        if all(endpoint.backend.shutdown(WAIT_S) == frozenset() for endpoint in self.endpoints):
            for store in self.opened:
                store.close()


def client_config(master: str) -> MooncakeConfig:
    """A client that lends nothing and has no buffer for the SDK's copying calls."""
    return MooncakeConfig(master, 0, PROTOCOL, DEVICES, local_buffer_size=0)


@pytest.fixture
def clients(pool):
    opened = Clients(pool)
    yield opened
    opened.close()


def random_rows(endpoint: Endpoint, seed: int) -> dict[str, bytes]:
    generator = np.random.default_rng(seed)
    rows = {name: generator.bytes(endpoint.region.slot_bytes) for name in endpoint.names}
    for name, data in rows.items():
        endpoint.write(name, data)
    return rows


def finish(endpoint: Endpoint, submit, *names: str) -> Outcome:
    attempt = submit(endpoint.view(*names), None)
    outcome = endpoint.wait_outcome(attempt)
    endpoint.wait_quiesced(attempt)
    return outcome


def masks(outcome: Outcome) -> list[list[bool]]:
    return [mask.tolist() for mask in outcome.delivered_masks]


def test_real_store_full_is_declined(tmp_path, request) -> None:
    _require_mooncake()
    # A pool of its own that one publish overfills: a 16 MiB segment (the smallest a TCP client
    # may lend, one allocator slab) holds at most 256 of the 512 rows. A PutStart that finds no room
    # answers at once. The master evicts once the pool passes its high watermark, so how many rows
    # land varies, but some are always declined and none fault.
    stack, master = _running_pool(str(tmp_path), segment_size=16 << 20)
    clients = Clients(master)
    with stack:
        names = [f"{request.node.name}-{i}" for i in range(512)]
        writer = clients.endpoint(names, workers=1)
        random_rows(writer, seed=4)
        outcome = finish(writer, writer.backend.publish, *names)
        delivered = int(outcome.delivered_masks[0].sum())
        declined = writer.driver.inner.stats()["rows_declined_pool_full"]
        assert outcome.fault is None
        assert delivered > 0 and declined > 0 and delivered + declined == len(names)
        writer.close()
        clients.close()  # while the master still answers


def test_real_object_being_written_is_not_ready_for_put_and_get(clients, request) -> None:
    # The other client's reservation holds both directions off only while the master keeps it:
    # after put_start_discard_timeout_sec (30 s) another writer may take it over.
    name = f"{request.node.name}-row"
    other = clients()
    assert other.batch_put_session_start([store_key(name)], [SLOT_BYTES]) == [0]
    writer = clients.endpoint([name])
    reader = clients.endpoint([name], fill=0x2D)
    random_rows(writer, seed=5)

    outcome = finish(writer, writer.backend.publish, name)
    assert outcome.fault is None and masks(outcome) == [[False]]
    outcome = finish(reader, reader.backend.fetch, name)
    assert outcome.fault is None and masks(outcome) == [[False]]
    assert reader.read(name) == b"-" * SLOT_BYTES
    assert writer.driver.inner.stats()["rows_not_ready_written_elsewhere"] == 1
    assert reader.driver.inner.stats()["rows_not_ready_being_written"] == 1

    assert other.batch_put_session_revoke([store_key(name)]) == [0]
    assert masks(finish(writer, writer.backend.publish, name)) == [[True]]
    writer.close()
    reader.close()


def test_real_get_lands_a_shorter_object_and_refuses_a_longer_one(clients, request) -> None:
    # batch_get_into reads an object shorter than the buffer into the buffer's head and answers
    # its size; it refuses a longer one before any transfer (INVALID_PARAMS).
    short, long = f"{request.node.name}-short", f"{request.node.name}-long"
    half = clients.endpoint([short], slot_bytes=SLOT_BYTES // 2)
    double = clients.endpoint([long], slot_bytes=2 * SLOT_BYTES)
    head = random_rows(half, seed=9)[short]
    random_rows(double, seed=10)
    assert masks(finish(half, half.backend.publish, short)) == [[True]]
    assert masks(finish(double, double.backend.publish, long)) == [[True]]
    reader = clients.endpoint([short, long], fill=0x2D, rows_per_batch=1)
    outcome = finish(reader, reader.backend.fetch, short)  # access ends: finish waits for it
    assert masks(outcome) == [[False]]
    assert f"object is {SLOT_BYTES // 2} bytes, expected {SLOT_BYTES}" in outcome.fault
    assert reader.read(short) == head + b"-" * (SLOT_BYTES // 2)
    outcome = finish(reader, reader.backend.fetch, long)
    assert masks(outcome) == [[False]] and "INVALID_PARAMS(-600)" in outcome.fault
    assert reader.read(long) == b"-" * SLOT_BYTES
    for each in (half, double, reader):
        each.close()


class TestRealMooncakeConformance(DriverConformance):
    """The shared driver checks over the real pool, one fresh client per check."""

    timeout = 30.0
    _checks = itertools.count()

    @pytest.fixture(autouse=True)
    def _own_keys(self, pool, monkeypatch):
        """The pool outlives each check, so each check names its objects apart."""
        self.master = pool
        monkeypatch.setattr(driver_conformance, "PREFIX", f"c{next(self._checks)}")

    def make_driver(self) -> MooncakeDriver:
        return MooncakeDriver(open_sdk(client_config(self.master)))

    @pytest.mark.xfail(strict=True, reason=SHORT_OBJECT_LANDS)
    def test_an_object_of_the_wrong_size_is_a_fault_not_a_miss(self) -> None:
        super().test_an_object_of_the_wrong_size_is_a_fault_not_a_miss()


def test_real_four_workers_round_trip_2000_rows_without_crossed_bytes(clients, request) -> None:
    names = [f"{request.node.name}-{i}" for i in range(2000)]
    writer = clients.endpoint(names, workers=4, rows_per_batch=64)
    reader = clients.endpoint(names, workers=4, rows_per_batch=64, fill=0x2D)
    rows = random_rows(writer, seed=6)
    outcome = finish(writer, writer.backend.publish, *names)
    assert outcome.fault is None and all(masks(outcome)[0])
    outcome = finish(reader, reader.backend.fetch, *names)
    assert outcome.fault is None and all(masks(outcome)[0])
    crossed = [name for name in names if reader.read(name) != rows[name]]
    assert crossed == []
    assert writer.driver.peak_in_flight >= 2 and reader.driver.peak_in_flight >= 2
    writer.close()
    reader.close()


class FirstScatterWaits:
    """An SDK client whose first put scatter waits, after its session started, for ``release``."""

    def __init__(self, store) -> None:
        self.store = store
        self.entered = threading.Event()
        self.release = threading.Event()
        self._first = threading.Lock()

    def __getattr__(self, name: str):
        return getattr(self.store, name)

    def batch_put_from_multi_buffer_ranges(self, *args):
        if self._first.acquire(blocking=False):
            self.entered.set()
            self.release.wait(WAIT_S)
        return self.store.batch_put_from_multi_buffer_ranges(*args)


def test_real_concurrent_publishes_of_the_same_keys_from_one_client(clients, request) -> None:
    # The first publish holds every key's put session while its scatter waits, so the second
    # publish's session start finds them taken by this client (INVALID_PARAMS) and declines its
    # rows; then the first publish delivers them all.
    names = [f"{request.node.name}-{i}" for i in range(64)]
    sdk = FirstScatterWaits(clients())
    two_publishing = Endpoint(MooncakeDriver(sdk), names, slot_bytes=SLOT_BYTES, workers=3)
    writer = clients.add(two_publishing)  # one worker stays for fetches
    rows = random_rows(writer, seed=8)
    first = writer.backend.publish(writer.view(*names), None)
    try:
        assert sdk.entered.wait(WAIT_S)
        second = writer.backend.publish(writer.view(*names), None)
        declined = writer.wait_outcome(second)
        assert declined.fault is None and masks(declined) == [[False] * 64]
    finally:
        sdk.release.set()
    assert masks(writer.wait_outcome(first)) == [[True] * 64]
    for attempt in (first, second):
        writer.wait_quiesced(attempt)
    assert writer.driver.inner.stats()["rows_not_ready_written_here"] == 64
    assert writer.driver.peak_in_flight == 2
    reader = clients.endpoint(names, fill=0x2D)
    assert all(masks(finish(reader, reader.backend.fetch, *names))[0])
    assert all(reader.read(name) == rows[name] for name in names)
    writer.close()
    reader.close()
