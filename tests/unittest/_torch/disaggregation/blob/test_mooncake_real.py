# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The blob backend over a real Mooncake pool: a master, and a separate process holding objects.

The tested clients give the pool no segment (``global_segment_size`` 0), so every object sits in
the other process's segment and every transfer goes through the transfer engine; their
``local_buffer_size`` is 0, so a call that needed the SDK's copying path would fail. Runs only with
``TRTLLM_MOONCAKE_REAL=1``, on a host with ``mooncake_master`` and ``mooncake.store``. The
transport is ``TRTLLM_MOONCAKE_PROTOCOL`` (default rdma) over ``TRTLLM_MOONCAKE_DEVICES`` (default:
the host's fastest active InfiniBand devices). The master must be shipped by one of the pinned SDK
distributions and report 0.3.13, and every installed pinned client build must be 0.3.13; both
versions are printed and recorded as test-suite properties. The shared
``driver_conformance`` checks run here too, each over a fresh client and under a key prefix of its
own.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import importlib.util
import os
import pathlib
import re
import secrets
import shlex
import shutil
import signal
import subprocess  # nosec B404
import threading
import time
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING

import numpy as np
import pytest
from cpu_fixtures import WAIT_S, Endpoint, masks, store_key, wait_until
from fake_mooncake import KeyGatedClient

from tensorrt_llm._torch.disaggregation.base.backend import Outcome
from tensorrt_llm._torch.disaggregation.blob.mooncake import MooncakeDriver
from tensorrt_llm._torch.disaggregation.blob.mooncake_setup import (
    PINNED_SDK_DISTRIBUTIONS,
    PINNED_SDK_VERSION,
    MooncakeConfig,
    open_sdk,
)
from tensorrt_llm._torch.disaggregation.testing.driver_conformance import SLOT as CHECK_SLOT
from tensorrt_llm._torch.disaggregation.testing.driver_conformance import DriverConformance

if TYPE_CHECKING:
    from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.master import LaunchedMaster

# An exception that escapes a thread fails the test.
pytestmark = [
    pytest.mark.skipif(
        os.environ.get("TRTLLM_MOONCAKE_REAL") != "1",
        reason="needs a real Mooncake pool; set TRTLLM_MOONCAKE_REAL=1",
    ),
    pytest.mark.filterwarnings("error::pytest.PytestUnhandledThreadExceptionWarning"),
]

PROTOCOL = os.environ.get("TRTLLM_MOONCAKE_PROTOCOL", "rdma")
DEVICES = os.environ.get("TRTLLM_MOONCAKE_DEVICES", "")
POOL_BYTES = 1 << 30
SLOT_BYTES = 64 << 10


def _require_mooncake() -> str:
    """Skips without a master or an SDK; fails unless both are the pinned build. Returns a line
    naming both versions, for the run's log."""
    master = shutil.which("mooncake_master")
    if master is None:
        pytest.skip("mooncake_master is not on PATH")
    try:
        found = importlib.util.find_spec("mooncake.store") is not None
    except ImportError:
        found = False
    if not found:
        pytest.skip("mooncake.store does not import")
    clients, owners = {}, []
    for name in PINNED_SDK_DISTRIBUTIONS:
        try:
            build = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        clients[name] = build.version
        shipped = (pathlib.Path(build.locate_file(file)).resolve() for file in build.files or ())
        if pathlib.Path(master).resolve() in shipped:
            owners.append(name)
    answer = subprocess.run(  # nosec B603
        [master, "--version"], capture_output=True, text=True, timeout=WAIT_S
    ).stdout
    reported = re.search(r"version (\S+)", answer)
    versions = (
        f"mooncake_master {master} from {owners or 'no SDK build'}, reporting "
        f"{reported.group(1) if reported else answer!r}; client builds {clients}"
    )
    pinned = (
        set(clients.values()) == {PINNED_SDK_VERSION}
        and reported
        and reported[1] == PINNED_SDK_VERSION
    )
    assert owners and pinned, f"not the pinned Mooncake {PINNED_SDK_VERSION}: {versions}"
    return versions


def _master_binary(run_dir: str, flags: Sequence[str]) -> str:
    """``mooncake_master``, or with ``flags`` a script in ``run_dir`` that starts it with them."""
    master = shutil.which("mooncake_master")
    if not flags:
        return master
    script = pathlib.Path(run_dir) / "mooncake_master_with_flags"
    script.write_text(f'#!/bin/sh\nexec {shlex.quote(master)} "$@" {shlex.join(flags)}\n')
    script.chmod(0o755)
    return str(script)


def _running_pool(
    run_dir: str, segment_size: int, master_flags: Sequence[str] = ()
) -> tuple[contextlib.ExitStack, LaunchedMaster]:
    """A master and a segment provider: the stack that stops them, and the master."""
    from test_common.mooncake_utils import running_master_on_free_ports, running_segment_provider

    with contextlib.ExitStack() as stack:
        # Ratio 0: the master evicts only above its high watermark, never just because a put failed.
        master = stack.enter_context(
            running_master_on_free_ports(
                run_dir,
                protocol=PROTOCOL,
                eviction_ratio=0.0,
                binary=_master_binary(run_dir, master_flags),
            )
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
        return stack.pop_all(), master


@pytest.fixture(scope="module")
def pool(tmp_path_factory, record_testsuite_property):
    versions = _require_mooncake()
    print(versions)
    record_testsuite_property("mooncake", versions)
    stack, master = _running_pool(str(tmp_path_factory.mktemp("pool")), POOL_BYTES)
    with stack:
        yield master.address


class Clients:
    """SDK clients of one master that give the pool no segment, and the endpoints built over them.

    ``close`` closes the clients only once every endpoint's shutdown returns no labels: a backend
    thread still inside the SDK would crash on a closed client, so a failed test leaves them open.
    A closed endpoint's driver has closed its client already, and closing a client twice does
    nothing.
    """

    def __init__(self, master: str) -> None:
        self.master = master
        self.opened = []
        self.endpoints: list[Endpoint] = []

    def open_client(self):
        self.opened.append(open_sdk(client_config(self.master)))
        return self.opened[-1]

    def endpoint(self, names, *, sdk=None, workers: int = 2, **options) -> Endpoint:
        """An endpoint over ``sdk``, a new client by default, with a driver of ``workers``."""
        options = {"slot_bytes": SLOT_BYTES, **options}
        driver = MooncakeDriver(sdk or self.open_client(), workers=workers)
        return self.add(Endpoint(driver, names, **options))

    def add(self, endpoint: Endpoint) -> Endpoint:
        self.endpoints.append(endpoint)
        return endpoint

    def close(self) -> None:
        if all(endpoint.backend.shutdown(WAIT_S) == frozenset() for endpoint in self.endpoints):
            for store in self.opened:
                store.close()


def client_config(master: str) -> MooncakeConfig:
    """A client with no segment of its own and no buffer for the SDK's copying calls."""
    return MooncakeConfig(master, 0, PROTOCOL, DEVICES, local_buffer_size=0)


@pytest.fixture
def clients(pool):
    opened = Clients(pool)
    yield opened
    opened.close()


def write_random_rows(endpoint: Endpoint, seed: int) -> dict[str, bytes]:
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


def test_real_store_full_is_declined(tmp_path, request) -> None:
    _require_mooncake()
    # A pool of its own that one publish overfills: a 16 MiB segment (the smallest a TCP client
    # may give, one allocator slab) holds at most 256 of the 512 rows. A PutStart that finds no room
    # answers at once. The master evicts once the pool passes its high watermark, so how many rows
    # land varies, but some are always declined and none fault.
    stack, master = _running_pool(str(tmp_path), segment_size=16 << 20)
    clients = Clients(master.address)
    with stack:
        names = [f"{request.node.name}-{i}" for i in range(512)]
        writer = clients.endpoint(names, workers=1)
        write_random_rows(writer, seed=4)
        outcome = finish(writer, writer.backend.publish, *names)
        delivered = int(outcome.delivered_masks[0].sum())
        declined = writer.driver.inner.stats()["rows_declined_pool_full"]
        assert outcome.fault is None
        assert delivered > 0 and declined > 0 and delivered + declined == len(names)
        writer.close()
        clients.close()  # while the master still answers


def test_real_get_faults_a_shorter_object_it_landed_and_refuses_a_longer_one(
    clients, request
) -> None:
    # batch_get_into reads an object shorter than the buffer into the buffer's head and answers
    # its size; it refuses a longer one before any transfer (INVALID_PARAMS).
    short, long = f"{request.node.name}-short", f"{request.node.name}-long"
    half = clients.endpoint([short], slot_bytes=SLOT_BYTES // 2)
    double = clients.endpoint([long], slot_bytes=2 * SLOT_BYTES)
    head = write_random_rows(half, seed=9)[short]
    write_random_rows(double, seed=10)
    assert masks(finish(half, half.backend.publish, short)) == [[True]]
    assert masks(finish(double, double.backend.publish, long)) == [[True]]
    reader = clients.endpoint([short, long], fill=0x2D)
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
    """The shared driver checks over the real pool, one fresh client per check, with the scenarios
    a real master can stage: a key another client is writing, and a get or a put held inside the
    SDK."""

    timeout = 30.0

    @pytest.fixture(autouse=True)
    def _master(self, pool):
        self.master = pool

    def make_driver(self) -> MooncakeDriver:
        return MooncakeDriver(open_sdk(client_config(self.master)))

    def make_held_key(self):
        key = f"held{secrets.token_hex(4)}/v1/H"  # the pool outlives each check
        writer = open_sdk(client_config(self.master))  # another client, still writing the key
        assert writer.batch_put_session_start([key], [CHECK_SLOT]) == [0]

        def release() -> None:
            try:
                assert writer.batch_put_session_revoke([key]) == [0]
            finally:
                writer.close()

        return self.make_driver(), key, release

    def make_blocked_get(self):
        key = f"blocked{secrets.token_hex(4)}/v1/S"
        sdk = KeyGatedClient(open_sdk(client_config(self.master)), "batch_get_into", key)
        return MooncakeDriver(sdk), key, sdk.entered, sdk.release.set

    def make_blocked_put(self):
        key = f"blocked{secrets.token_hex(4)}/v1/P"
        sdk = KeyGatedClient(
            open_sdk(client_config(self.master)), "batch_put_from_multi_buffer_ranges", key
        )
        return MooncakeDriver(sdk), key, sdk.entered, sdk.release.set

    def make_put_still_reading_the_source(self):
        """The blocked put's gate stands before the scatter, the call that reads the source."""
        return self.make_blocked_put()


class DataCallsInFlight:
    """An SDK client that counts its row-moving calls running at once; ``peak`` is the most."""

    def __init__(self, store) -> None:
        self.store = store
        self.peak = self._running = 0
        self._lock = threading.Lock()

    def __getattr__(self, name: str):
        return getattr(self.store, name)

    def batch_get_into(self, *args):
        return self._counted(self.store.batch_get_into, args)

    def batch_put_from_multi_buffer_ranges(self, *args):
        return self._counted(self.store.batch_put_from_multi_buffer_ranges, args)

    def _counted(self, call, args):
        with self._lock:
            self._running += 1
            self.peak = max(self.peak, self._running)
        try:
            return call(*args)
        finally:
            with self._lock:
                self._running -= 1


def test_real_four_workers_round_trip_2000_rows_without_crossed_bytes(clients, request) -> None:
    names = [f"{request.node.name}-{i}" for i in range(2000)]
    writer_sdk = DataCallsInFlight(clients.open_client())
    reader_sdk = DataCallsInFlight(clients.open_client())
    # Parts of 64 rows, so the one 2000-row operation runs on several workers at once.
    writer = clients.add(
        Endpoint(
            MooncakeDriver(writer_sdk, workers=4, rows_per_call=64), names, slot_bytes=SLOT_BYTES
        )
    )
    reader = clients.add(
        Endpoint(
            MooncakeDriver(reader_sdk, workers=4, rows_per_call=64),
            names,
            slot_bytes=SLOT_BYTES,
            fill=0x2D,
        )
    )
    rows = write_random_rows(writer, seed=6)
    outcome = finish(writer, writer.backend.publish, *names)
    assert outcome.fault is None and all(masks(outcome)[0])
    outcome = finish(reader, reader.backend.fetch, *names)
    assert outcome.fault is None and all(masks(outcome)[0])
    crossed = [name for name in names if reader.read(name) != rows[name]]
    assert crossed == []
    assert writer_sdk.peak >= 2 and reader_sdk.peak >= 2
    writer.close()
    reader.close()


class StartsRecorded:
    """An SDK client whose ``starts`` lists the keys of every put session start."""

    def __init__(self, store) -> None:
        self.store = store
        self.starts: list[list[str]] = []

    def __getattr__(self, name: str):
        return getattr(self.store, name)

    def batch_put_session_start(self, keys, *args):
        self.starts.append(list(keys))
        return self.store.batch_put_session_start(keys, *args)


class FirstScatterWaits(StartsRecorded):
    """An SDK client whose first put scatter waits, after its session started, for ``release``
    (at most ``hold_s``)."""

    def __init__(self, store, hold_s: float = WAIT_S) -> None:
        super().__init__(store)
        self.hold_s = hold_s
        self.entered = threading.Event()
        self.release = threading.Event()
        self._first = threading.Lock()

    def batch_put_from_multi_buffer_ranges(self, *args):
        if self._first.acquire(blocking=False):
            self.entered.set()
            self.release.wait(self.hold_s)
        return self.store.batch_put_from_multi_buffer_ranges(*args)


def test_real_second_publish_of_keys_still_being_put_is_skipped_without_asking_the_sdk(
    clients, request
) -> None:
    # The first publish holds every key's put session while its scatter waits, so the driver
    # skips the second publish's rows without asking the SDK; then the first publish delivers
    # them all.
    names = [f"{request.node.name}-{i}" for i in range(64)]
    sdk = FirstScatterWaits(clients.open_client())
    # Three workers: puts leave one to gets, so the two publishes run side by side.
    writer = clients.endpoint(names, sdk=sdk, workers=3)
    rows = write_random_rows(writer, seed=8)
    first = writer.backend.publish(writer.view(*names), None)
    try:
        assert sdk.entered.wait(WAIT_S)
        second = writer.backend.publish(writer.view(*names), None)
        skipped = writer.wait_outcome(second)
        assert skipped.fault is None and masks(skipped) == [[False] * 64]
    finally:
        sdk.release.set()
    assert masks(writer.wait_outcome(first)) == [[True] * 64]
    for attempt in (first, second):
        writer.wait_quiesced(attempt)
    assert writer.driver.inner.stats()["rows_not_ready_held_by_own_put"] == 64
    assert len(sdk.starts) == 1
    reader = clients.endpoint(names, fill=0x2D)
    assert all(masks(finish(reader, reader.backend.fetch, *names))[0])
    assert all(reader.read(name) == rows[name] for name in names)
    writer.close()
    reader.close()


DISCARD_S = 2  # the master's put_start_discard_timeout_sec in the test below; 30 s by default


def test_real_put_stuck_past_the_discard_timeout_keeps_its_keys(tmp_path, request) -> None:
    # Past put_start_discard_timeout_sec the master would let another client's start discard a
    # put still being written. This client's own second put of the keys is still skipped, and the
    # objects never read as existing until the first put ends.
    _require_mooncake()
    flags = [f"--put_start_discard_timeout_sec={DISCARD_S}"]
    stack, master = _running_pool(str(tmp_path), 64 << 20, master_flags=flags)
    clients = Clients(master.address)
    with stack:
        names = [f"{request.node.name}-{i}" for i in range(8)]
        keys = [store_key(name) for name in names]
        sdk = FirstScatterWaits(clients.open_client(), hold_s=DISCARD_S + WAIT_S)
        writer = clients.endpoint(names, sdk=sdk, workers=3)
        observer = clients.open_client()
        rows = write_random_rows(writer, seed=11)
        first = writer.backend.publish(writer.view(*names), None)
        try:
            assert sdk.entered.wait(WAIT_S)
            time.sleep(DISCARD_S + 1)
            second = writer.backend.publish(writer.view(*names), None)
            skipped = writer.wait_outcome(second)
            assert skipped.fault is None and masks(skipped) == [[False] * 8]
            assert observer.batch_is_exist(keys) == [0] * 8
        finally:
            sdk.release.set()
        assert masks(writer.wait_outcome(first)) == [[True] * 8]
        for attempt in (first, second):
            writer.wait_quiesced(attempt)
        assert len(sdk.starts) == 1  # the second put never asked the SDK
        assert observer.batch_is_exist(keys) == [1] * 8
        reader = clients.endpoint(names, fill=0x2D)
        assert all(masks(finish(reader, reader.backend.fetch, *names))[0])
        assert all(reader.read(name) == rows[name] for name in names)
        writer.close()
        reader.close()
        clients.close()  # while the master still answers


RPC_TIMEOUT_MS = 1000  # the writer's MC_RPC_TIMEOUT_MS in the test below; 30 s by default
# The master's put_start_discard_timeout_sec in the test below: long enough that the observer's
# start, after the master resumes and the fetches find the unwritten objects, still lands within it.
LATE_START_DISCARD_S = 5


@contextlib.contextmanager
def frozen(process: subprocess.Popen) -> Iterator[None]:
    """Stops ``process`` for the body's duration: it answers nothing, while the kernel still
    queues what is sent to it."""
    process.send_signal(signal.SIGSTOP)
    try:
        stat = pathlib.Path(f"/proc/{process.pid}/stat")  # "pid (comm) state ..."
        wait_until(lambda: stat.read_text().rpartition(") ")[2][0] == "T", "the process to stop")
        yield
    finally:
        process.send_signal(signal.SIGCONT)


def finds_unwritten_objects(reader: Endpoint, *names: str) -> bool:
    """Whether a fetch of ``names`` finds each object reserved but not complete; unlike a start,
    a get reserves nothing."""
    stats = reader.driver.inner.stats
    before = stats()["rows_not_ready_no_complete_replica"]
    finish(reader, reader.backend.fetch, *names)
    return stats()["rows_not_ready_no_complete_replica"] - before == len(names)


def test_real_keys_whose_put_start_timed_out_are_never_started_again(
    tmp_path, request, monkeypatch
) -> None:
    # A put start that times out while the master is frozen still runs once the master resumes,
    # and reserves the keys for the writer. The writer never starts a session of those keys
    # again: its later puts only ask whether the object exists, DECLINED until another client
    # wrote it whole and READY after.
    _require_mooncake()
    flags = [f"--put_start_discard_timeout_sec={LATE_START_DISCARD_S}"]
    stack, master = _running_pool(str(tmp_path), 64 << 20, master_flags=flags)
    clients = Clients(master.address)
    with stack:
        names = [f"{request.node.name}-{i}" for i in range(4)]
        keys = [store_key(name) for name in names]
        with monkeypatch.context() as patch:
            patch.setenv("MC_RPC_TIMEOUT_MS", str(RPC_TIMEOUT_MS))  # read as the client sets up
            sdk = StartsRecorded(clients.open_client())
        writer = clients.endpoint(names, sdk=sdk)
        observer = clients.open_client()
        reader = clients.endpoint(names, fill=0x2D)
        with frozen(master.process):  # about one RPC timeout, well under the master's client TTL
            timed_out = finish(writer, writer.backend.publish, *names)
        assert masks(timed_out) == [[False] * 4]
        assert "put session start: RPC_TIMEOUT(-901)" in timed_out.fault
        assert writer.driver.inner.stats()["put_keys_left_for_close"] == 4

        wait_until(lambda: finds_unwritten_objects(reader, *names), "the late start to land")
        # Within the discard timeout another client's start finds each key held.
        assert observer.batch_put_session_start(keys, [SLOT_BYTES] * 4) == [-705] * 4

        declined = finish(writer, writer.backend.publish, *names)
        assert declined.fault is None and masks(declined) == [[False] * 4]
        assert writer.driver.inner.stats()["rows_declined_left_open_by_own_put"] == 4

        # Then another client's start discards the unwritten objects.
        time.sleep(LATE_START_DISCARD_S + 1)
        other = clients.endpoint(names)
        rows = write_random_rows(other, seed=13)
        assert masks(finish(other, other.backend.publish, *names)) == [[True] * 4]
        assert masks(finish(writer, writer.backend.publish, *names)) == [[True] * 4]
        assert sdk.starts == [keys]  # READY on the existence check alone
        assert all(masks(finish(reader, reader.backend.fetch, *names))[0])
        assert all(reader.read(name) == rows[name] for name in names)
        for each in (writer, other, reader):
            each.close()
        clients.close()  # while the master still answers
