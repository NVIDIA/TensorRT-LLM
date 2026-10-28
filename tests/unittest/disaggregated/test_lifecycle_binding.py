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

"""CPU contracts for shared arbitration and its native ownership binding.

The native fixture is not a backend-neutral adapter or profile qualification.
Only manager holds and backend status are doubles; native ownership, logical
outcomes, and shared retirement use production code. Allocation release remains
the caller's responsibility after access ends.
"""

import gc
import weakref
from dataclasses import dataclass
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.disaggregation.base.transfer import SessionStatus
from tensorrt_llm._torch.disaggregation.lifecycle.retirement import (
    QuiescenceFatalEvent,
    RetirementDeadline,
    RetirementWatchdog,
)
from tensorrt_llm._torch.disaggregation.native import transfer as transfer_mod
from tensorrt_llm._torch.disaggregation.native.ownership import (
    PhysicalOperationState,
    ReceiveOperationOwner,
    SendOperationOwner,
    TransferNotSubmittedError,
)
from tensorrt_llm.disaggregated_params import DisaggregatedParams

pytestmark = pytest.mark.cpu_only


class _Root:
    """Opaque, weak-referenceable allocation hold or backend request."""


@dataclass
class _Status:
    """Backend-owned completion evidence; no lifecycle transitions live here."""

    result: object = False
    queries: int = 0

    def is_completed(self) -> object:
        """Return the injected backend evidence, or its query failure."""
        self.queries += 1
        if isinstance(self.result, Exception):
            raise RuntimeError(str(self.result))
        return self.result


class _NativeBinding:
    """Bind native operation ownership to shared deadline arbitration."""

    def __init__(self) -> None:
        """Create an unpublished source binding without starting a worker thread."""
        self.clock = Mock(return_value=10.0)
        self.events: list[QuiescenceFatalEvent] = []
        self.watchdog = RetirementWatchdog(self.events.append, clock=self.clock)
        self.retirement = self.watchdog.create_owner(4, "send", 5.0)
        self.sender = SendOperationOwner(self.retirement)
        self.manager_hold = _Root()


@pytest.fixture
def binding() -> _NativeBinding:
    """Provide native owners without starting transport worker threads."""
    return _NativeBinding()


def test_native_compatibility_names_preserve_class_identity() -> None:
    """Compatibility imports must not fork state or exception identities."""
    from tensorrt_llm._torch.disaggregation.native import retirement, transfer

    assert retirement.RetirementWatchdog is RetirementWatchdog
    assert retirement.RetirementDeadline is RetirementDeadline
    assert retirement.QuiescenceFatalEvent is QuiescenceFatalEvent
    assert transfer._ReceiveOperationOwner is ReceiveOperationOwner
    assert transfer._PhysicalOperationState is PhysicalOperationState
    assert transfer._TransferNotSubmittedError is TransferNotSubmittedError
    assert transfer.SendOperationOwner is SendOperationOwner
    assert {state.value for state in PhysicalOperationState} == {
        "ADMITTED",
        "SUBMITTING",
        "SUBMITTED",
        "NOT_SUBMITTED",
        "BACKEND_DONE",
        "IN_DOUBT",
    }


def test_shared_arbitration_retains_opaque_claims_until_each_settles() -> None:
    """The shared arbiter needs claim identities, not native ownership types."""
    watchdog = RetirementWatchdog(Mock(), clock=Mock(return_value=10.0))
    retirement = watchdog.create_owner(4, "receive", 5.0)
    source, destination = _Root(), _Root()
    source_ref, destination_ref = weakref.ref(source), weakref.ref(destination)
    assert retirement.expose(source, destination)
    retirement.complete_pieces()
    del source, destination
    gc.collect()
    assert source_ref() is not None and destination_ref() is not None
    assert retirement.settle(source_ref())
    assert source_ref() is None
    assert not retirement.is_complete
    assert retirement.settle(destination_ref())
    assert destination_ref() is None
    assert retirement.is_complete
    assert retirement.close() and retirement.close()
    watchdog.stop()


def test_multi_owner_exposure_waits_for_local_completion(binding: _NativeBinding) -> None:
    """KV, AUX, and manager holds survive partial and duplicate remote evidence."""
    binding.retirement.close()
    retirement = binding.watchdog.create_owner(5, "receive", 5.0)
    kv, aux = ReceiveOperationOwner(retirement), ReceiveOperationOwner(retirement)
    assert retirement.expose(kv, aux, binding.manager_hold)
    for owner, cohort in ((kv, {7, 8}), (aux, {7})):
        owner.begin_publication()
        owner.seal_writer_cohort(len(cohort), cohort)
        owner.finish_publication()
    assert kv.record_writer_result(7, True, wait_for_local_completion=True) == (True, False)
    assert kv.record_writer_result(7, True, wait_for_local_completion=True) == (False, False)
    assert kv.record_writer_result(8, True, wait_for_local_completion=True) == (True, True)
    assert not kv.resources_drained
    assert not retirement.can_retire()
    assert aux.record_writer_result(7, True, wait_for_local_completion=False) == (True, True)
    retirement.complete_pieces()
    assert not retirement.is_complete
    assert id(kv) in retirement._claims
    kv.finish_local_completion()
    kv.finish_local_completion()
    assert set(retirement._claims) == {id(binding.manager_hold)}
    # The caller releases its allocation hold only after both accessors ended.
    assert retirement.settle(binding.manager_hold)
    assert kv.resources_drained and aux.resources_drained
    assert retirement.is_complete
    assert retirement.close() and retirement.close()
    binding.watchdog.require_retired()
    binding.watchdog.stop()


@pytest.mark.parametrize("rejected", [False, True])
def test_never_submitted_operation_retires_idempotently(
    binding: _NativeBinding, rejected: bool
) -> None:
    """Skipping submission and closed admission both establish no-access proof."""
    sender = binding.sender
    assert sender.begin_physical_operation(7)
    assert not sender.begin_physical_operation(7)
    if rejected:
        binding.retirement.request_drain("cancelled before submission")
        with pytest.raises(TransferNotSubmittedError, match="admission is closed"):
            sender.begin_backend_submission(7, _Root())
    sender.retire_unsubmitted_physical_operation(7)
    sender.retire_unsubmitted_physical_operation(7)
    assert sender.has_started_physical_operation(7)
    assert sender.resources_drained
    assert not binding.retirement._claims
    assert binding.retirement.close() and binding.retirement.close()
    binding.watchdog.stop()


@pytest.mark.parametrize("evidence", [False, 1, RuntimeError("query failed")])
def test_late_exact_done_retires_without_changing_logical_outcome(
    binding: _NativeBinding, evidence: object
) -> None:
    """Only literal DONE on the retained status ends ambiguous physical access."""
    sender = transfer_mod.SendTaskBase(DisaggregatedParams(disagg_request_id=4))
    sender.bind_logical_outcomes(transfer_mod._LogicalOutcomes(binding.retirement))
    status, request = _Status(evidence), _Root()
    request_ref, status_ref = weakref.ref(request), weakref.ref(status)
    assert sender.begin_physical_operation(7)
    sender.begin_backend_submission(7, request)
    sender.record_backend_submission(7, status)
    failure = RuntimeError("original failure")
    sender.fail(failure)
    committed = sender.logical_outcome
    assert committed is not None
    assert committed.status is SessionStatus.ERROR
    assert committed.reason == "original failure"
    sender.mark_physical_operation_in_doubt(7)
    sender.mark_physical_operation_in_doubt(7)
    del request
    gc.collect()
    assert request_ref() is not None
    assert not sender.poll_in_doubt_physical_operation(7)
    assert not sender.resources_drained
    assert sender.logical_outcome is committed
    status.result = True
    binding.clock.return_value = 14.999
    assert sender.poll_in_doubt_physical_operation(7)
    assert sender.resources_drained
    assert sender.retire_backend_done_physical_operation(7)
    assert not sender.poll_in_doubt_physical_operation(7)
    assert status.queries == 2
    assert sender.logical_outcome is committed
    assert sender.status is transfer_mod.TaskStatus.ERROR
    assert sender._exception is failure
    del status
    gc.collect()
    assert request_ref() is None and status_ref() is None
    assert binding.retirement.close() and binding.retirement.close()
    binding.watchdog.stop()


@pytest.mark.parametrize("has_status", [False, True])
def test_exact_expiry_contains_and_retains_all_roots(
    binding: _NativeBinding, has_status: bool
) -> None:
    """Missing handles and exact-boundary DONE both retain roots after fatal expiry."""
    sender, request, status = binding.sender, _Root(), _Status()
    request_ref, hold_ref = weakref.ref(request), weakref.ref(binding.manager_hold)
    assert binding.retirement.expose(binding.manager_hold)
    assert sender.begin_physical_operation(7)
    sender.begin_backend_submission(7, request)
    if has_status:
        sender.record_backend_submission(7, status)
    sender.mark_physical_operation_in_doubt(7)
    binding.clock.return_value = 14.0
    sender.mark_physical_operation_in_doubt(7)
    assert not sender.poll_in_doubt_physical_operation(7)
    assert not sender.resources_drained
    binding.clock.return_value = 15.0
    status.result = True
    assert not sender.poll_in_doubt_physical_operation(7)
    binding.watchdog.progress()
    binding.watchdog.progress()
    event = binding.watchdog.fatal
    assert event is not None
    assert (event.started_at, event.deadline, event.expired_at) == (10.0, 15.0, 15.0)
    assert binding.events == [event]
    assert not binding.retirement.settle(binding.manager_hold)
    assert not binding.retirement.close()
    assert not sender.resources_drained
    with pytest.raises(RuntimeError, match="admission is closed"):
        binding.watchdog.create_owner(6, "receive", 5.0)
    with pytest.raises(RuntimeError, match="teardown refused"):
        binding.watchdog.require_retired()
    watchdog = binding.watchdog
    # Pytest retains its fixture; remove the caller's roots before collecting.
    del binding.manager_hold, binding.sender
    del request, sender, binding
    gc.collect()
    assert request_ref() is not None and hold_ref() is not None
    with pytest.raises(RuntimeError, match="retained owners"):
        watchdog.stop()


def test_receive_ambiguity_requires_explicit_settlement(binding: _NativeBinding) -> None:
    """An ordinary failed result cannot substitute for access-end evidence."""
    owner = ReceiveOperationOwner(binding.retirement)
    assert binding.retirement.expose(owner, binding.manager_hold)
    owner.begin_publication()
    owner.seal_writer_cohort(1, {7})
    owner.finish_publication()
    assert owner.record_writer_in_doubt(7)
    assert not owner.record_writer_in_doubt(7)
    assert owner.record_writer_result(7, False, wait_for_local_completion=False) == (False, False)
    assert not owner.resources_drained
    assert owner.record_writer_settlement(7)
    assert not owner.record_writer_settlement(7)
    assert set(binding.retirement._claims) == {id(binding.manager_hold)}
    assert binding.retirement.settle(binding.manager_hold)
    assert owner.resources_drained
    assert binding.retirement.close() and binding.retirement.close()
    binding.watchdog.stop()
