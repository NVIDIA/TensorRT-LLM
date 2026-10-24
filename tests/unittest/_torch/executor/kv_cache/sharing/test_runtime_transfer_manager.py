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

"""Real-manager bytes and copy ownership behind the Runtime operation ledger.

Only the external provider is doubled. Requests, lender, GPU copies, projection,
TransferManager and Lifecycle arbitration use their production implementations.
"""

import ctypes
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from threading import Event, get_ident
from time import monotonic
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.disaggregation._assembly import (
    OwnerCleanupFn,
    ProfileSpec,
    attach_domain,
    bind_write_return,
)
from tensorrt_llm._torch.disaggregation.base.backend import (
    CacheKind,
    Capabilities,
    Hint,
    MemoryKind,
    Outcome,
    SlotRegion,
    View,
    WakeFn,
)
from tensorrt_llm._torch.disaggregation.lifecycle.retirement import RetirementWatchdog
from tensorrt_llm._torch.disaggregation.transfer_manager import (
    Accepted,
    Direction,
    LifecycleBinding,
    ReturnDecision,
    ReturnStatus,
    TransferIntent,
    TransferManager,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import RegionHold, RowSelection, StagingOptions
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.weight_sharing.artifact_identity import ArtifactIdentity
from tensorrt_llm._torch.weight_sharing.source_identity import (
    SOURCE_IDENTITY_FORMAT_VERSION,
    SourceIdentity,
)
from tensorrt_llm.bindings import DataType

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="allocates KV cache pools")

_END = 96
_BLOCKS = 3
_NOW = 100.0
_FETCH_BYTE = 0x2A


@dataclass
class _Attempt:
    """Independently observable delivery and ended access at the provider boundary."""

    view: View
    outcome: Outcome | None = None
    ended: bool = False

    def poll(self) -> Outcome | None:
        """Return the stable Outcome once the test's provider publishes it."""
        return self.outcome

    def quiesced(self) -> bool:
        """Report access ending independently from the delivery observation."""
        return self.ended

    def cancel(self) -> None:
        """Leave physical evidence unchanged by advisory cancellation."""

    def complete(self, *, delivered: bool = True) -> None:
        """Publish final whole-row masks without changing the access-end flag."""
        self.outcome = Outcome.for_view(
            self.view,
            tuple(
                np.full(len(group.object_keys), delivered, dtype=bool) for group in self.view.groups
            ),
        )


class _ByteBackend:
    """Deterministic provider reading and writing the actual protected host slots."""

    def __init__(
        self,
        regions: tuple[SlotRegion, ...],
        hold: RegionHold,
        wake: WakeFn,
        kit: SimpleNamespace,
        post_owner_cleanup: OwnerCleanupFn,
    ) -> None:
        """Retain the backend-owned staging hold before using any slot address."""
        self.capabilities = Capabilities(fetch=True, publish=True, cancel=True)
        self.regions = regions
        self.hold = hold
        self.wake = wake
        self.kit = kit
        self.attempts: list[_Attempt] = []
        self.published: dict[tuple[int, int], bytes] = {}
        self.owner = get_ident()
        self.finalized = False
        self.post_owner_cleanup = post_owner_cleanup
        self.finalizer_posted = False

    def fetch(self, view: View, source: Hint) -> _Attempt:
        """Write distinct real bytes into each submitted row, withholding its Outcome."""
        attempt = _Attempt(view)
        self.attempts.append(attempt)
        for group in view.groups:
            length = self.regions[group.region_index].slot_bytes
            for ordinal, address in enumerate(group.addresses.tolist()):
                data = self.kit.staged_row(_FETCH_BYTE, group.layer_group_index, ordinal, length)
                ctypes.memmove(address, data, length)
        return attempt

    def publish(self, view: View, destination: Hint) -> _Attempt:
        """Accept publication but defer the host read until after producer exit."""
        attempt = _Attempt(view)
        self.attempts.append(attempt)
        return attempt

    def read_publication(self, attempt: _Attempt) -> None:
        """Read real host bytes while the operation still protects its staging slots."""
        for group in attempt.view.groups:
            length = self.regions[group.region_index].slot_bytes
            for ordinal, address in enumerate(group.addresses.tolist()):
                self.published[group.layer_group_index, ordinal] = self.kit.host_bytes(
                    address, length
                )
        attempt.complete()
        self.wake()

    def end_all(self) -> None:
        """End every deterministic provider access before test cleanup."""
        for attempt in self.attempts:
            if attempt.outcome is None:
                attempt.complete(delivered=False)
            attempt.ended = True
        self.wake()

    def shutdown(self, timeout: float) -> frozenset:
        """Report ended provider access without releasing the manager-thread hold."""
        self.end_all()
        if not self.finalizer_posted:
            self.finalizer_posted = self.post_owner_cleanup(self.finalize)
        return frozenset()

    def finalize(self) -> None:
        """Release the provider's hold on its manager thread after shutdown evidence."""
        assert get_ident() == self.owner
        assert not self.finalized
        self.hold.release()
        self.finalized = True


class _Harness:
    """Compose the actual manager, lender, holder and Lifecycle with a byte provider."""

    def __init__(self, manager: KVCacheManagerV2, kit: SimpleNamespace) -> None:
        """Use a fixed clock so CUDA initialization never consumes a safety deadline."""
        self.manager = manager
        self.wake = Event()
        self.watchdog = RetirementWatchdog(lambda event: None, clock=lambda: _NOW)
        self.profile = ProfileSpec(
            id="real-runtime-test",
            lifecycle_profile_id="executor-fatal",
            source_identity=SourceIdentity(
                format_version=SOURCE_IDENTITY_FORMAT_VERSION,
                artifact_identity=ArtifactIdentity(1, "checkpoint_manifest_sha256", "a" * 64),
                model_fingerprint="b" * 64,
                quant_fingerprint="c" * 64,
                backend_fingerprint="d" * 64,
                parallel_fingerprint="e" * 64,
                rank=0,
                shard_fingerprint="f" * 64,
            ),
            compatibility_id="test-kv-semantics-v1",
            model_layout="mha-trtllm-hnd",
            dtype="bfloat16",
            topology=(1, 1, 1, 1),
            state_kinds=(CacheKind.PAGED,),
            full_attention=True,
            sparse_attention=False,
            sink_token_length=0,
            lora_enabled=False,
            pool_count=1,
            speculative_mode="none",
            model_role="target",
            draft_pool=None,
            joint_reuse=False,
            staging=StagingOptions(_END),
        )
        self.domain = attach_domain(manager, self.profile)
        self.runtime = TransferManager(wake=self.wake.set, clock=lambda: _NOW)
        self.backend: _ByteBackend | None = None

        def factory(
            regions: tuple[SlotRegion, ...],
            wake: WakeFn,
            config: object,
            post_owner_cleanup: OwnerCleanupFn,
        ) -> _ByteBackend:
            """Root the provider hold and use its manager-thread cleanup dispatcher."""
            self.backend = _ByteBackend(
                regions, self.domain.lender.hold_regions(), wake, kit, post_owner_cleanup
            )
            return self.backend

        self.runtime.assemble(
            "store",
            self.domain.namespace,
            self.profile.id,
            factory,
            domain_id="runtime-test-lender",
            lifecycle=LifecycleBinding(self.watchdog, timeout_s=5),
            provider_cleanup_complete=lambda: self.backend.finalizer_posted,
        )

    def tick(self) -> tuple:
        """Advance manager work before the holder observes already acquired leases."""
        self.domain.lender.progress()
        return self.runtime.advance(_NOW)

    def fetch(
        self, request: LlmRequest, *, ready_event: torch.cuda.Event | None = None
    ) -> Accepted:
        """Fence page use in orchestration and bind return to the originating lender."""
        if ready_event is None:
            ready_event = torch.cuda.Event()
            ready_event.record(self.manager._stream)
        lease = self.domain.lender.lend_write(
            request,
            0,
            _END,
            memory_kind=MemoryKind.HOST,
            rows=RowSelection.NAMED,
            ready_event=ready_event,
        )
        result = self.runtime.submit(
            lease,
            TransferIntent("store", self.profile.id, Direction.FETCH, MemoryKind.HOST),
            write_return=bind_write_return(self.domain.lender),
        )
        assert isinstance(result, Accepted)
        self.tick()
        return result

    def publish(self, request: LlmRequest) -> Accepted:
        """Fence producer work and hand off its lease without retaining the request."""
        ready_event = torch.cuda.Event()
        ready_event.record(self.manager._stream)
        lease = self.domain.lender.lend_read(
            request,
            0,
            _END,
            memory_kind=MemoryKind.HOST,
            rows=RowSelection.NAMED,
            ready_event=ready_event,
        )
        result = self.runtime.submit(
            lease, TransferIntent("store", self.profile.id, Direction.PUBLISH, MemoryKind.HOST)
        )
        assert isinstance(result, Accepted)
        self.tick()
        return result

    def authorize(self, operation: int, *, reusable: bool = True) -> None:
        """Authorize this single-rank fixture after its fixed result and access end."""
        self.runtime.authorize_return(operation, ReturnDecision(reusable, True))

    def close(self) -> None:
        """End provider access and drain the holder before the manager fixture shuts down."""
        self.backend.end_all()
        self.runtime.begin_shutdown(_NOW + 5)
        stop = monotonic() + 5
        while not self.runtime.shutdown_status().complete and monotonic() < stop:
            self.tick()
            self.manager._stream.synchronize()
            self.wake.wait(0.001)
            self.wake.clear()
        assert self.runtime.shutdown_status().complete, "holder retained completed test operations"
        assert self.backend.finalized
        self.watchdog.stop()


@contextmanager
def _runtime(manager: KVCacheManagerV2, kit: SimpleNamespace) -> Iterator[_Harness]:
    """Clean up deterministic backend accesses even when an assertion fails."""
    harness = _Harness(manager, kit)
    try:
        yield harness
    finally:
        harness.close()


def _assert_fetch_bytes(
    manager: KVCacheManagerV2, request: LlmRequest, kit: SimpleNamespace
) -> None:
    """Compare device pages through the fixture's independent pool-address oracle."""
    device = kit.DevicePages(manager)
    cache = kit.kv(manager, request)
    for group in range(kit.num_layer_groups(manager)):
        slots = kit.pages(cache, group)
        for ordinal in range(_BLOCKS):
            expected = kit.staged_row(_FETCH_BYTE, group, ordinal, device.page_bytes(group))
            assert kit.digest(device.read(group, slots[ordinal])) == kit.digest(expected)


@pytest.mark.parametrize("outcome_first", [False, True])
def test_real_fetch_waits_for_evidence_and_authorization_before_consumption(
    real_manager: Callable[..., AbstractContextManager[KVCacheManagerV2]],
    kit: SimpleNamespace,
    outcome_first: bool,
) -> None:
    """Delivery and ended access retain a fetch until coordinated consuming return."""
    with real_manager(dtype=DataType.BF16) as manager, _runtime(manager, kit) as harness:
        request = kit.admitted(manager, 1, list(range(1000, 1097)))
        started = harness.fetch(request)
        record = harness.runtime.operations[started.operation_id]
        attempt = harness.backend.attempts[0]
        if outcome_first:
            attempt.complete()
            harness.tick()
            assert record.outcome is attempt.outcome and not record.access_ended
            attempt.ended = True
        else:
            attempt.ended = True
            harness.tick()
            assert record.access_ended and record.outcome is None
            attempt.complete()
        assert not record.released
        harness.tick()
        manager._stream.synchronize()
        harness.tick()
        assert not record.released and record.return_status is ReturnStatus.PENDING
        assert harness.domain.lender.poll_resume_bounds(request) is None

        harness.authorize(started.operation_id)
        assert record.released and record.return_status is ReturnStatus.RETURNED
        manager._stream.synchronize()
        harness.tick()
        bounds = harness.domain.lender.poll_resume_bounds(request)
        assert bounds is not None and bounds.max_position == _END
        _assert_fetch_bytes(manager, request, kit)
        assert not harness.runtime.operations
        manager.free_resources(request)


def test_consuming_return_keeps_copy_pending_until_ready_event(
    real_manager: Callable[..., AbstractContextManager[KVCacheManagerV2]], kit: SimpleNamespace
) -> None:
    """A consumed HOST lease does not make device bytes ready before the page-use fence."""
    with real_manager(dtype=DataType.BF16) as manager, _runtime(manager, kit) as harness:
        request = kit.admitted(manager, 1, list(range(2000, 2097)))
        elsewhere = torch.cuda.Stream()
        with kit.held_stream(elsewhere, strict=False):
            ready = torch.cuda.Event()
            ready.record(elsewhere)
            started = harness.fetch(request, ready_event=ready)
            record = harness.runtime.operations[started.operation_id]
            attempt = harness.backend.attempts[0]
            attempt.complete()
            attempt.ended = True
            harness.tick()
            harness.authorize(started.operation_id)
            assert record.released and record.return_status is ReturnStatus.RETURNED
            for _ in range(3):
                harness.tick()
                assert harness.domain.lender.poll_resume_bounds(request) is None
        manager._stream.synchronize()
        harness.tick()
        bounds = harness.domain.lender.poll_resume_bounds(request)
        assert bounds is not None and bounds.max_position == _END
        _assert_fetch_bytes(manager, request, kit)
        manager.free_resources(request)


def test_real_miss_settles_only_after_authorized_return(
    real_manager: Callable[..., AbstractContextManager[KVCacheManagerV2]], kit: SimpleNamespace
) -> None:
    """All-false delivery counts no KV while settling the real lender's pending fetch."""
    with real_manager(dtype=DataType.BF16) as manager, _runtime(manager, kit) as harness:
        request = kit.admitted(manager, 1, list(range(2500, 2597)))
        started = harness.fetch(request)
        attempt = harness.backend.attempts[0]
        attempt.complete(delivered=False)
        attempt.ended = True
        harness.tick()
        assert harness.domain.lender.poll_resume_bounds(request) is None
        harness.authorize(started.operation_id, reusable=False)
        manager._stream.synchronize()
        harness.tick()
        bounds = harness.domain.lender.poll_resume_bounds(request)
        assert bounds is not None and bounds.max_position == 0
        assert not harness.runtime.operations
        manager.free_resources(request)


@pytest.mark.parametrize("copy_pending_at_exit", [False, True], ids=["readable", "copy-pending"])
def test_publication_keeps_original_bytes_after_producer_exit(
    real_manager: Callable[..., AbstractContextManager[KVCacheManagerV2]],
    kit: SimpleNamespace,
    copy_pending_at_exit: bool,
) -> None:
    """Granted read protection survives producer exit, including an unfinished local copy."""
    with (
        real_manager(dtype=DataType.BF16, max_tokens=kit.POOL_TOKENS, host_cache_size=0) as manager,
        _runtime(manager, kit) as harness,
    ):
        # Warm the first cache closure before holding a CUDA stream.
        manager.free_resources(kit.published(manager, 900, list(range(40000, 40033))))
        producer = kit.published(manager, 1, list(range(3000, 3097)))
        device = kit.DevicePages(manager)
        cache = kit.kv(manager, producer)
        expected = {
            (group, ordinal): device.read(group, kit.pages(cache, group)[ordinal])
            for group in range(kit.num_layer_groups(manager))
            for ordinal in range(_BLOCKS)
        }
        if copy_pending_at_exit:
            with kit.held_stream(manager._stream, strict=False) as gate:
                result = harness.publish(producer)
                assert not harness.backend.attempts
                manager.free_resources(producer)
                gate.open()
        else:
            result = harness.publish(producer)
            manager._stream.synchronize()
            harness.tick()
            assert len(harness.backend.attempts) == 1
            manager.free_resources(producer)

        assert kit.kv(manager, producer) is None
        manager._stream.synchronize()
        harness.tick()
        assert len(harness.backend.attempts) == 1
        record = harness.runtime.operations[result.operation_id]
        assert not record.abandoned and not record.released
        attempt = harness.backend.attempts[0]
        harness.backend.read_publication(attempt)
        assert kit.digest(harness.backend.published) == kit.digest(expected)
        harness.tick()
        assert not record.released
        attempt.ended = True
        harness.tick()
        assert record.released and not harness.runtime.operations
