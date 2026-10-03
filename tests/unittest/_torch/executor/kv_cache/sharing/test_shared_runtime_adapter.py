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
"""Real staging loans and CUDA copies bound to independently controlled backend evidence.

The provider double controls the external transfer boundary, including physical
access after a logical result. The manager, lender, lease, registration lifetime
adapter, retirement owner and CUDA completion events are production objects.
These module tests do not qualify NIXL transport or a scheduler integration.
"""

from __future__ import annotations

import ctypes
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Literal

import pytest
import torch

from tensorrt_llm._torch.disaggregation.base.shared import (
    Attempt,
    CacheExtent,
    Cancelled,
    Delivered,
    Failed,
    Outcome,
    Registration,
    Route,
)
from tensorrt_llm._torch.disaggregation.lifecycle.retirement import (
    QuiescenceFatalEvent,
    RetirementWatchdog,
)
from tensorrt_llm._torch.disaggregation.resource.shared import (
    SHARED_CONTRACT_REVISION,
    SharedRuntimeProfile,
)
from tensorrt_llm._torch.disaggregation.resource.shared_lifetime import (
    SharedStagingAdapter,
    cuda_copy_completion,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import (
    Lease,
    Part,
    Readiness,
    RegionView,
    StagingLender,
    StagingOptions,
    attach_staging,
)
from tensorrt_llm.bindings import DataType

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="allocates KV cache pools")

ManagerFactory = Callable[..., AbstractContextManager[KVCacheManagerV2]]
TOKENS_PER_BLOCK = 32
FETCH_END = 3 * TOKENS_PER_BLOCK
PROMPT = list(range(1000, 1000 + FETCH_END + 1))
OTHER_PROMPT = list(range(5000, 5000 + FETCH_END + 1))
SCOPE = b"shared-runtime-adapter-module-test"


@dataclass(eq=False)
class _Attempt:
    """External delivery with independent logical outcome and access-end gates."""

    extent: CacheExtent
    outcome: Outcome | None = None
    access_ended: threading.Event = field(default_factory=threading.Event)

    def poll(self) -> Outcome | None:
        """Return the immutable logical outcome independently of access.

        Returns:
            The current outcome, or None before the provider answers.
        """
        return self.outcome


@dataclass
class _Registration:
    """Observe revocation without substituting for adapter lifetime ownership."""

    address: int
    size: int
    closed: bool = False

    def close(self) -> None:
        """Record idempotent revocation of this exact registered span."""
        self.closed = True


class _Backend:
    """Controllable external provider resolving real registered staging addresses."""

    def __init__(self, parts: tuple[Part, ...], view: RegionView) -> None:
        """Retain external address resolution without taking over lease ownership.

        Args:
            parts: Public host regions offered for registration.
            view: Ready loan establishing each group's physical part.
        """
        self.parts = parts
        self.part_by_group = {run.layer_group: parts[run.part] for run in view.runs}
        self.registrations: list[_Registration] = []
        self.attempts: list[_Attempt] = []

    def register_pool(self, address: int, size: int) -> Registration:
        """Register only an actual manager-owned staging region.

        Args:
            address: Beginning of the manager allocation.
            size: Registered byte length.

        Returns:
            Observable handle for this registration instance.
        """
        assert any(part.address == address and part.nbytes == size for part in self.parts)
        registration = _Registration(address, size)
        self.registrations.append(registration)
        return registration

    def fetch(self, extent: CacheExtent, *, route: Route | None = None) -> _Attempt:
        """Submit without completing any transfer or logical outcome.

        Args:
            extent: Real adapter extent containing destination coordinates.
            route: Unused because this test provider has one source.

        Returns:
            A delivery whose completion and quiescence are controlled separately.
        """
        assert route is None
        assert self.registrations and all(not item.closed for item in self.registrations)
        attempt = _Attempt(extent)
        self.attempts.append(attempt)
        return attempt

    def publish(self, extent: CacheExtent) -> _Attempt:
        """Submit a readable source through the same external delivery boundary.

        Args:
            extent: Real adapter extent containing source coordinates.

        Returns:
            The pending publication handle.
        """
        return self.fetch(extent)

    def quiesce(self, attempts: Iterable[Attempt]) -> bool:
        """Block independently until every listed handle has ended memory access.

        Args:
            attempts: Handles covered by this proof, never unrelated submissions.

        Returns:
            True after explicit no-future-access evidence for all listed handles.
        """
        for attempt in attempts:
            assert isinstance(attempt, _Attempt)
            assert attempt.access_ended.wait(30), "test did not release backend access"
        return True

    def settle(self, attempts: Iterable[Attempt]) -> None:
        """Reject blocking outcome waits in the runtime adapter.

        Args:
            attempts: Handles a caller tried to wait for.

        Raises:
            AssertionError: This adapter must use nonblocking outcome polling.
        """
        raise AssertionError("the adapter must not call blocking settle")

    def probe(self, name: bytes, units: Sequence[bytes]) -> frozenset[bytes] | None:
        """Decline the optional lookup optimization without performing a transfer.

        Args:
            name: Extent namespace.
            units: Content names under consideration.

        Returns:
            None, indicating no cheaper answer than a fetch.
        """
        return None

    def open_route(self, hint: Mapping[str, object]) -> Route:
        """Reject route preparation for this single-source test provider.

        Args:
            hint: Advisory routing metadata.

        Raises:
            NotImplementedError: This provider has exactly one source.
        """
        raise NotImplementedError("single-source test provider")

    def deliver(self, attempt: _Attempt, store: Mapping[bytes, bytes]) -> None:
        """Write only requested hits into real staging, then publish their names.

        Args:
            attempt: Previously submitted destination.
            store: Content bytes available from the external source.
        """
        assert attempt.outcome is None
        served = set()
        for unit in attempt.extent.units:
            if unit.name not in store:
                continue
            part = self.part_by_group[unit.local_group]
            address = part.address + unit.local * part.slot_bytes
            data = store[unit.name]
            assert len(data) == part.slot_bytes
            assert part.address <= address < address + len(data) <= part.address + part.nbytes
            ctypes.memmove(address, data, len(data))
            served.add(unit.name)
        attempt.outcome = Delivered(frozenset(served))

    def end_access(self) -> None:
        """Supply no-future-access evidence and release every proof waiter."""
        for attempt in self.attempts:
            attempt.access_ended.set()


def _profile() -> SharedRuntimeProfile:
    """Describe the supported assembly while identifying the boundary test double.

    Returns:
        BF16 MHA HND, single-rank, manager-host compatibility facts.
    """
    return SharedRuntimeProfile(
        contract_revision=SHARED_CONTRACT_REVISION,
        backend_revision="controlled-boundary-module-test",
        backend_kind="native_nixl",
        manager_kind="KVCacheManagerV2",
        cache_dtype="bfloat16",
        attention_backend="TRTLLM",
        attention_kind="mha",
        layout="HND",
        parallelism=(1, 1, 1, 1),
        staging="manager_host",
        committed_whole_blocks=True,
        extra_features=frozenset(),
    )


def _attach(manager: KVCacheManagerV2) -> StagingLender:
    """Attach the public lender with exactly one fetch of staging capacity.

    Args:
        manager: Real BF16 cache manager owning its execution stream.

    Returns:
        Lender whose contested slots make premature release observable.
    """
    return attach_staging(manager, scope=SCOPE, staging=StagingOptions(FETCH_END))


def _ready(lease: Lease, manager: KVCacheManagerV2, kit: SimpleNamespace) -> RegionView:
    """Progress a real lease and its CUDA copies until it exposes its view.

    Args:
        lease: Pending read or write loan.
        manager: Manager whose stream performs staging copies.
        kit: Existing lender test helpers.

    Returns:
        The public ready view.
    """
    views: list[RegionView] = []

    def poll() -> bool:
        """Progress the owner and save a ready view.

        Returns:
            Whether the lease has become ready.
        """
        manager._stream.synchronize()
        view = lease.poll()
        if view is not None:
            views.append(view)
        return view is not None

    assert kit.wait_until(poll), lease.failure
    return views[0]


def _progress_until(
    adapter: SharedStagingAdapter, predicate: Callable[[], bool], kit: SimpleNamespace
) -> None:
    """Drive only nonblocking owner progress until an observable condition holds.

    Args:
        adapter: Real resource binding under test.
        predicate: Condition expected after local progress.
        kit: Existing bounded polling helper.
    """

    def progress() -> bool:
        """Poll once without waiting on a CUDA stream or backend.

        Returns:
            The externally observed test condition after this poll.
        """
        adapter.progress()
        return predicate()

    assert kit.wait_until(progress), "adapter did not reach the expected state"


def _adapter(
    manager: KVCacheManagerV2, lender: StagingLender, backend: _Backend
) -> tuple[SharedStagingAdapter, RetirementWatchdog, list[Callable[[], bool]]]:
    """Bind actual manager resources and capture real CUDA completion queries.

    Args:
        manager: Cache manager that owns the exact copy stream.
        lender: Its one public staging lender.
        backend: Controllable provider at the external dependency boundary.

    Returns:
        Adapter, lifecycle arbiter, and queries recorded after local copies.
    """
    failures: list[QuiescenceFatalEvent] = []
    watchdog = RetirementWatchdog(failures.append)
    queries: list[Callable[[], bool]] = []
    record = cuda_copy_completion(manager._stream)

    def record_copy() -> Callable[[], bool]:
        """Record actual stream completion while exposing evidence to assertions.

        Returns:
            The production event query, not a synthetic completion signal.
        """
        query = record()
        queries.append(query)
        return query

    adapter = SharedStagingAdapter(
        manager,
        lender,
        _profile(),
        backend,
        watchdog.create_owner(0, "receive", 120),
        record_copy,
    )
    adapter.register()
    return adapter, watchdog, queries


@pytest.mark.parametrize(
    ("rows", "usable"),
    [((0, 1, 2), 96), ((0, 1), 64), ((0, 2), 32), ((), 0)],
    ids=["complete", "prefix", "gap", "miss"],
)
def test_fetch_served_units_and_cuda_readiness(
    real_manager: ManagerFactory, kit: SimpleNamespace, rows: tuple[int, ...], usable: int
) -> None:
    """Transfer only served whole rows and wait for contiguous GPU-ready coverage.

    Args:
        real_manager: Existing real manager context factory.
        kit: Existing byte, request and bounded-wait oracles.
        rows: Block ordinals available from the source.
        usable: Contiguous ready prefix expected after the local copy.
    """
    with (
        real_manager(dtype=DataType.BF16, host_cache_size=0) as source_manager,
        real_manager(dtype=DataType.BF16, host_cache_size=0) as manager,
    ):
        source = kit.published(source_manager, 1, PROMPT)
        source_lender = _attach(source_manager)
        source_lease = source_lender.lend_read(source, 0, FETCH_END)
        source_view = _ready(source_lease, source_manager, kit)
        store = {
            name.tobytes(): kit.host_bytes(address, source_lender.parts[run.part].slot_bytes)
            for run in source_view.runs
            for name, address, ordinal in zip(run.names, run.addresses.tolist(), run.ordinals)
            if ordinal in rows
        }
        target = kit.admitted(manager, 2, PROMPT)
        lender = _attach(manager)
        lease = lender.lend_write(target, 0, FETCH_END)
        view = _ready(lease, manager, kit)
        kit.fill_sentinel(manager, target)
        backend = _Backend(lender.parts, view)
        adapter, watchdog, queries = _adapter(manager, lender, backend)
        operation = adapter.submit_fetch(lease, backend, watchdog.create_owner(2, "receive", 120))
        try:
            backend.deliver(backend.attempts[0], store)
            adapter.progress()
            assert operation.outcome == Delivered(frozenset(store))
            assert not operation.released
            assert lender.readiness(target) is None
            assert queries == [], "logical delivery must not race continuing backend access"
            backend.end_access()
            _progress_until(adapter, lambda: operation.released, kit)
            assert queries and all(query() for query in queries)
            assert lender.readiness(target) == Readiness(usable, 0)
            sentinel = bytes([kit.SENTINEL]) * kit.DevicePages(manager).page_bytes(0)
            for ordinal in range(FETCH_END // TOKENS_PER_BLOCK):
                expected = (
                    kit.page(source_manager, source, 0, ordinal) if ordinal in rows else sentinel
                )
                assert kit.digest(kit.page(manager, target, 0, ordinal)) == kit.digest(expected)
            assert adapter.close()
            assert all(registration.closed for registration in backend.registrations)
            watchdog.require_retired()
        finally:
            backend.end_access()
            source_lease.release()
            adapter.close()


@pytest.mark.parametrize("terminal", ["failure", "cancel"], ids=["failed", "cancelled"])
def test_terminal_outcome_retains_staging_until_access_ends(
    real_manager: ManagerFactory, kit: SimpleNamespace, terminal: Literal["failure", "cancel"]
) -> None:
    """A logical error cannot give a still-exposed staging slot to another request.

    Args:
        real_manager: Existing real manager context factory.
        kit: Existing request and bounded-wait helpers.
        terminal: Independently observable terminal outcome.
    """
    with real_manager(dtype=DataType.BF16, host_cache_size=0) as manager:
        target = kit.admitted(manager, 1, PROMPT)
        other = kit.admitted(manager, 2, OTHER_PROMPT)
        lender = _attach(manager)
        lease = lender.lend_write(target, 0, FETCH_END)
        view = _ready(lease, manager, kit)
        backend = _Backend(lender.parts, view)
        adapter, watchdog, _ = _adapter(manager, lender, backend)
        operation = adapter.submit_fetch(lease, backend, watchdog.create_owner(1, "receive", 120))
        waiting = lender.lend_write(other, 0, FETCH_END)
        try:
            if terminal == "failure":
                backend.attempts[0].outcome = Failed("controlled transfer failure")
                adapter.progress()
                assert isinstance(operation.outcome, Failed)
            else:
                adapter.cancel(operation)
                assert operation.outcome == Cancelled(by_peer=False)
            assert not operation.released
            assert waiting.poll() is None
            assert not adapter.close()
            assert all(not registration.closed for registration in backend.registrations)
            backend.end_access()
            _progress_until(adapter, lambda: operation.released, kit)
            assert lender.readiness(target) == Readiness(0, 0)
            waiting_view = _ready(waiting, manager, kit)
            waiting.mark_arrived(waiting_view.row_masks())
            waiting.release()
            assert adapter.close()
            watchdog.require_retired()
        finally:
            backend.end_access()
            waiting.release()
            adapter.close()


def test_delayed_cuda_copy_retains_staging_and_registration(
    real_manager: ManagerFactory, kit: SimpleNamespace
) -> None:
    """Real CUDA event ordering protects the slot and registration during shutdown.

    Args:
        real_manager: Existing real manager context factory.
        kit: Existing real held-stream gate and byte oracles.
    """
    with real_manager(dtype=DataType.BF16, host_cache_size=0) as manager:
        target = kit.admitted(manager, 1, PROMPT)
        other = kit.admitted(manager, 2, OTHER_PROMPT)
        lender = _attach(manager)
        lease = lender.lend_write(target, 0, FETCH_END)
        view = _ready(lease, manager, kit)
        kit.fill_sentinel(manager, target)
        backend = _Backend(lender.parts, view)
        adapter, watchdog, queries = _adapter(manager, lender, backend)
        operation = adapter.submit_fetch(lease, backend, watchdog.create_owner(1, "receive", 120))
        waiting = lender.lend_write(other, 0, FETCH_END)
        store = {
            name.tobytes(): bytes([0x5A]) * lender.parts[run.part].slot_bytes
            for run in view.runs
            for name in run.names
        }
        try:
            backend.deliver(backend.attempts[0], store)
            with kit.held_stream(manager._stream) as gate:
                backend.end_access()
                _progress_until(adapter, lambda: bool(queries), kit)
                assert not queries[0](), "completion must be recorded after the real copy"
                assert not operation.released
                assert lender.readiness(target) is None
                assert waiting.poll() is None
                assert not adapter.close()
                assert all(not registration.closed for registration in backend.registrations)
                gate.open()
            _progress_until(adapter, lambda: operation.released, kit)
            assert lender.readiness(target) == Readiness(FETCH_END, 0)
            expected = bytes([0x5A]) * kit.DevicePages(manager).page_bytes(0)
            assert kit.page(manager, target, 0, 0) == expected
            waiting_view = _ready(waiting, manager, kit)
            waiting.mark_arrived(waiting_view.row_masks())
            waiting.release()
            assert adapter.close()
            assert all(registration.closed for registration in backend.registrations)
            watchdog.require_retired()
        finally:
            backend.end_access()
            waiting.release()
            adapter.close()


def test_request_exit_and_page_reuse_do_not_end_backend_access(
    real_manager: ManagerFactory, kit: SimpleNamespace
) -> None:
    """Late delivery to retained staging cannot overwrite a replacement request.

    Args:
        real_manager: Existing real manager context factory.
        kit: Existing pool-pressure and independent device-page oracles.
    """
    with real_manager(
        dtype=DataType.BF16, host_cache_size=0, max_tokens=kit.POOL_TOKENS
    ) as manager:
        target = kit.admitted(manager, 1, PROMPT)
        lender = _attach(manager)
        lease = lender.lend_write(target, 0, FETCH_END)
        view = _ready(lease, manager, kit)
        lent_pages = {kit.pages(kit.kv(manager, target), 0)[ordinal] for ordinal in range(3)}
        backend = _Backend(lender.parts, view)
        adapter, watchdog, _ = _adapter(manager, lender, backend)
        operation = adapter.submit_fetch(lease, backend, watchdog.create_owner(1, "receive", 120))
        others = kit.Requests(manager)
        try:
            manager.free_resources(target)
            del target
            assert others.allocate(kit.pool_pages(manager))
            assert lent_pages <= others.pages(), "the test must recycle the original GPU pages"
            store = {
                name.tobytes(): bytes([0x5A]) * lender.parts[run.part].slot_bytes
                for run in view.runs
                for name in run.names
            }
            backend.deliver(backend.attempts[0], store)
            adapter.progress()
            assert isinstance(operation.outcome, Delivered)
            assert not operation.released
            assert all(not registration.closed for registration in backend.registrations)
            backend.end_access()
            _progress_until(adapter, lambda: operation.released, kit)
            device = kit.DevicePages(manager)
            expected = bytes([kit.SENTINEL]) * device.page_bytes(0)
            for slot in lent_pages:
                assert device.read(0, slot) == expected
            assert adapter.close()
            watchdog.require_retired()
        finally:
            backend.end_access()
            others.free()
            adapter.close()


def test_publish_source_survives_request_exit_until_quiescence(
    real_manager: ManagerFactory, kit: SimpleNamespace
) -> None:
    """An actual device-to-host loan stays readable after source request retirement.

    Args:
        real_manager: Existing real manager context factory.
        kit: Existing pool-pressure, staging-byte and request helpers.
    """
    with real_manager(
        dtype=DataType.BF16, host_cache_size=0, max_tokens=kit.POOL_TOKENS
    ) as manager:
        source = kit.published(manager, 1, PROMPT)
        lender = _attach(manager)
        lease = lender.lend_read(source, 0, FETCH_END)
        view = _ready(lease, manager, kit)
        expected = {
            address: kit.host_bytes(address, lender.parts[run.part].slot_bytes)
            for run in view.runs
            for address in run.addresses.tolist()
        }
        backend = _Backend(lender.parts, view)
        adapter, watchdog, queries = _adapter(manager, lender, backend)
        operation = adapter.submit_publish(lease, backend, watchdog.create_owner(1, "send", 120))
        try:
            backend.attempts[0].outcome = Delivered(
                frozenset(unit.name for unit in operation.extent.units)
            )
            manager.free_resources(source)
            del source
            assert kit.overwrite_free_pages(manager) > 0
            adapter.progress()
            assert not operation.released
            assert queries == [], "a publication has no destination copy to fabricate"
            assert not adapter.close()
            for address, content in expected.items():
                assert kit.host_bytes(address, len(content)) == content
            backend.end_access()
            _progress_until(adapter, lambda: operation.released, kit)
            assert adapter.close()
            watchdog.require_retired()
        finally:
            backend.end_access()
            adapter.close()
