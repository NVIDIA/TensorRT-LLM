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
"""CPU-only tests for native receiver transfer ownership."""

from __future__ import annotations

import pytest

from tensorrt_llm._torch.disaggregation.native.receive_lifecycle import (
    BounceState,
    ExposureState,
    LifecycleAction,
    LogicalState,
    PhysicalState,
    RecvTransferRegistry,
    WriterMode,
    WriterResult,
)

KEY = (101, 0)
WRITERS = frozenset({3, 7})


def _actions(update) -> set[LifecycleAction]:
    return set(update.actions)


def _prepare(
    registry: RecvTransferRegistry,
    *,
    key=KEY,
    writers=WRITERS,
    bounce: bool = True,
):
    # The initial containment implementation retains Python request ownership.
    # Allocator-backed destination leases belong to a future phase.
    update = registry.prepare(
        key,
        writers,
        has_bounce_slot=bounce,
    )
    assert update.accepted
    return update


def _publish(registry: RecvTransferRegistry, rank: int, *, key=KEY) -> None:
    update = registry.begin_publication(key, rank)
    assert update.accepted and update.publication_allowed
    assert registry.mark_published(key, rank).accepted


def _settle_bounce(registry: RecvTransferRegistry, *, key=KEY, succeeded: bool = True):
    update = registry.finish_bounce_scatter(key, succeeded=succeeded)
    assert not update.conflict
    if succeeded:
        assert update.accepted
    return update


class TestAdmissionAndPublication:
    def test_prepare_requires_an_exact_nonempty_writer_set(self) -> None:
        registry = RecvTransferRegistry()

        with pytest.raises(ValueError, match="must not be empty"):
            registry.prepare(KEY, set(), has_bounce_slot=False)
        with pytest.raises(ValueError, match="non-negative integer"):
            registry.prepare(KEY, {-1}, has_bounce_slot=False)
        with pytest.raises(ValueError, match="non-negative integer"):
            registry.prepare(KEY, {True}, has_bounce_slot=False)

        _prepare(registry)
        snapshot = registry.context_snapshot(KEY)
        assert snapshot is not None
        assert snapshot.expected_writers == WRITERS
        assert [writer.rank for writer in snapshot.writers] == [3, 7]

    def test_identical_prepare_is_idempotent_but_conflicting_prepare_is_rejected(self) -> None:
        registry = RecvTransferRegistry()
        _prepare(registry)

        duplicate = registry.prepare(
            KEY,
            WRITERS,
            has_bounce_slot=True,
        )
        conflict = registry.prepare(
            KEY,
            {3},
            has_bounce_slot=True,
        )

        assert not duplicate.accepted and duplicate.duplicate and not duplicate.conflict
        assert not conflict.accepted and conflict.conflict and not conflict.duplicate
        assert registry.context_snapshot(KEY).expected_writers == WRITERS

    def test_cancel_before_publication_closes_gate_and_releases_once(self) -> None:
        registry = RecvTransferRegistry()
        _prepare(registry)

        (cancelled,) = registry.cancel_request(KEY[0])

        assert cancelled.logical_state is LogicalState.CANCELLED
        assert cancelled.physical_state is PhysicalState.DRAINING
        assert cancelled.writers_terminal
        assert _actions(cancelled) == {
            LifecycleAction.NOTIFY_CANCELLED,
            LifecycleAction.RELEASE_BOUNCE,
        }
        settled = _settle_bounce(registry)
        assert settled.physical_state is PhysicalState.DRAINED
        assert _actions(settled) == {LifecycleAction.CONTEXT_DRAINED}
        rejected = registry.begin_publication(KEY, 3)
        assert not rejected.accepted and not rejected.publication_allowed
        assert registry.cancel_request(KEY[0])[0].actions == ()

    def test_begin_publication_is_the_atomic_cancellation_boundary(self) -> None:
        registry = RecvTransferRegistry()
        _prepare(registry)

        authorized = registry.begin_publication(KEY, 3)
        (cancelled,) = registry.cancel_request(KEY[0])

        assert authorized.publication_allowed
        assert cancelled.logical_state is LogicalState.CANCELLED
        assert cancelled.physical_state is PhysicalState.IN_DOUBT
        assert not cancelled.writers_terminal
        assert LifecycleAction.RELEASE_BOUNCE not in cancelled.actions
        snapshot = registry.context_snapshot(KEY)
        writers = {writer.rank: writer for writer in snapshot.writers}
        assert writers[3].exposure is ExposureState.POSSIBLY_EXPOSED
        assert writers[7].exposure is ExposureState.NEVER_EXPOSED

        # The send may complete after cancellation because authorization won.
        assert registry.mark_published(KEY, 3).accepted
        terminal = registry.record_result(KEY, 3, WriterResult.FAILED, WriterMode.BOUNCE)
        assert terminal.physical_state is PhysicalState.DRAINING
        assert terminal.writers_terminal
        assert LifecycleAction.RELEASE_BOUNCE in terminal.actions
        assert _settle_bounce(registry).physical_state is PhysicalState.DRAINED

    def test_definitive_nonpublication_unwinds_partial_fanout(self) -> None:
        registry = RecvTransferRegistry()
        _prepare(registry)
        assert registry.begin_publication(KEY, 3).publication_allowed

        terminal = registry.mark_never_published(KEY, 3)

        assert terminal.logical_state is LogicalState.FAILED
        assert terminal.physical_state is PhysicalState.DRAINING
        assert terminal.writers_terminal
        snapshot = registry.context_snapshot(KEY)
        assert all(writer.exposure is ExposureState.NEVER_EXPOSED for writer in snapshot.writers)
        assert _settle_bounce(registry).physical_state is PhysicalState.DRAINED

    def test_partial_publication_failure_retains_only_possibly_exposed_writer(self) -> None:
        registry = RecvTransferRegistry()
        _prepare(registry)
        assert registry.begin_publication(KEY, 3).publication_allowed

        failed = registry.fail_context(KEY, "second ZMQ publication failed")

        assert failed.logical_state is LogicalState.FAILED
        assert failed.physical_state is PhysicalState.IN_DOUBT
        assert not failed.writers_terminal
        assert _actions(failed) == {LifecycleAction.NOTIFY_FAILURE}
        snapshot = registry.context_snapshot(KEY)
        writers = {writer.rank: writer for writer in snapshot.writers}
        assert writers[3].exposure is ExposureState.POSSIBLY_EXPOSED
        assert writers[7].exposure is ExposureState.NEVER_EXPOSED

        terminal = registry.record_result(KEY, 3, WriterResult.FAILED, WriterMode.BOUNCE)
        assert terminal.physical_state is PhysicalState.DRAINING
        assert terminal.writers_terminal
        assert LifecycleAction.RELEASE_BOUNCE in terminal.actions
        assert _settle_bounce(registry).physical_state is PhysicalState.DRAINED

    def test_each_writer_can_cross_publication_boundary_only_once(self) -> None:
        registry = RecvTransferRegistry()
        _prepare(registry)

        first = registry.begin_publication(KEY, 3)
        duplicate = registry.begin_publication(KEY, 3)

        assert first.publication_allowed
        assert not duplicate.publication_allowed
        assert not duplicate.accepted and duplicate.duplicate

    def test_unexpected_writer_cannot_be_published(self) -> None:
        registry = RecvTransferRegistry()
        _prepare(registry)

        update = registry.begin_publication(KEY, 9)

        assert not update.accepted and update.conflict
        assert update.logical_state is LogicalState.FAILED
        assert update.physical_state is PhysicalState.DRAINING
        assert registry.context_snapshot(KEY).expected_writers == WRITERS
        assert _settle_bounce(registry).physical_state is PhysicalState.DRAINED


class TestWriterResults:
    def test_first_failure_survives_consumer_detach_until_late_sibling_result(self) -> None:
        registry = RecvTransferRegistry()
        _prepare(registry)
        _publish(registry, 3)
        _publish(registry, 7)

        failed = registry.record_result(KEY, 3, WriterResult.FAILED, WriterMode.BOUNCE)

        assert failed.logical_state is LogicalState.FAILED
        assert failed.physical_state is PhysicalState.DRAINING
        assert not failed.writers_terminal
        assert _actions(failed) == {LifecycleAction.NOTIFY_FAILURE}
        registry.detach_consumer(KEY[0])
        assert registry.context_snapshot(KEY) is not None
        assert not registry.is_request_drained(KEY[0])

        late = registry.record_result(KEY, 7, WriterResult.SUCCESS, WriterMode.BOUNCE)

        assert late.logical_state is LogicalState.FAILED
        assert late.physical_state is PhysicalState.DRAINING
        assert late.writers_terminal and not late.bounce_pending
        assert LifecycleAction.START_BOUNCE_SCATTER not in late.actions
        assert _actions(late) == {LifecycleAction.RELEASE_BOUNCE}
        settled = _settle_bounce(registry)
        assert settled.physical_state is PhysicalState.DRAINED
        assert _actions(settled) == {LifecycleAction.CONTEXT_DRAINED}

    def test_unexpected_writer_does_not_satisfy_exact_writer_set(self) -> None:
        registry = RecvTransferRegistry()
        _prepare(registry, bounce=False)
        _publish(registry, 3)
        _publish(registry, 7)

        unexpected = registry.record_result(KEY, 9, WriterResult.SUCCESS, WriterMode.DIRECT)

        assert unexpected.conflict
        assert unexpected.logical_state is LogicalState.FAILED
        assert unexpected.physical_state is PhysicalState.IN_DOUBT
        assert not unexpected.writers_terminal
        snapshot = registry.context_snapshot(KEY)
        assert all(writer.result is None for writer in snapshot.writers)

        registry.record_result(KEY, 3, WriterResult.SUCCESS, WriterMode.DIRECT)
        expected = registry.record_result(KEY, 7, WriterResult.SUCCESS, WriterMode.DIRECT)
        assert expected.writers_terminal
        assert expected.physical_state is PhysicalState.IN_DOUBT

        (quiesced,) = registry.mark_backend_quiesced()
        assert quiesced.physical_state is PhysicalState.DRAINED

    def test_identical_result_is_idempotent(self) -> None:
        registry = RecvTransferRegistry()
        _prepare(registry, writers={3}, bounce=False)
        _publish(registry, 3)

        first = registry.record_result(KEY, 3, WriterResult.SUCCESS, WriterMode.DIRECT)
        duplicate = registry.record_result(KEY, 3, WriterResult.SUCCESS, WriterMode.DIRECT)

        assert first.logical_state is LogicalState.SUCCEEDED
        assert LifecycleAction.NOTIFY_SUCCESS in first.actions
        assert not duplicate.accepted and duplicate.duplicate
        assert not duplicate.conflict and duplicate.actions == ()

    def test_contradictory_result_fails_closed_until_backend_quiescence(self) -> None:
        registry = RecvTransferRegistry()
        _prepare(registry, writers={3, 7}, bounce=False)
        _publish(registry, 3)
        _publish(registry, 7)
        registry.record_result(KEY, 3, WriterResult.SUCCESS, WriterMode.DIRECT)

        conflict = registry.record_result(KEY, 3, WriterResult.FAILED, WriterMode.DIRECT)

        assert conflict.conflict
        assert conflict.logical_state is LogicalState.FAILED
        assert conflict.physical_state is PhysicalState.IN_DOUBT
        registry.record_result(KEY, 7, WriterResult.SUCCESS, WriterMode.DIRECT)
        assert not registry.is_request_drained(KEY[0])

        registry.mark_backend_quiesced()
        assert registry.is_request_drained(KEY[0])

    def test_contradiction_after_single_writer_success_corrects_logical_outcome(self) -> None:
        registry = RecvTransferRegistry()
        _prepare(registry, writers={3}, bounce=False)
        _publish(registry, 3)

        success = registry.record_result(KEY, 3, WriterResult.SUCCESS, WriterMode.DIRECT)
        conflict = registry.record_result(KEY, 3, WriterResult.FAILED, WriterMode.DIRECT)

        assert LifecycleAction.NOTIFY_SUCCESS in success.actions
        assert conflict.conflict
        assert conflict.logical_state is LogicalState.FAILED
        assert conflict.physical_state is PhysicalState.IN_DOUBT
        assert conflict.actions == (LifecycleAction.NOTIFY_FAILURE,)
        assert not registry.is_request_drained(KEY[0])

        registry.mark_backend_quiesced()
        assert registry.is_request_drained(KEY[0])

    @pytest.mark.parametrize("writer_rank", [3, 9], ids=["expected", "unexpected"])
    def test_nonterminal_protocol_conflict_before_publication_requires_backend_fence(
        self, writer_rank
    ) -> None:
        registry = RecvTransferRegistry()
        _prepare(registry, writers={3}, bounce=False)

        conflict = registry.record_protocol_conflict(KEY, writer_rank, "unknown result status")

        assert conflict.conflict
        assert conflict.logical_state is LogicalState.FAILED
        assert conflict.physical_state is PhysicalState.IN_DOUBT
        assert not registry.is_request_drained(KEY[0])

        (quiesced,) = registry.mark_backend_quiesced()
        assert quiesced.physical_state is PhysicalState.DRAINED

    def test_direct_target_rejects_bounce_result_mode(self) -> None:
        registry = RecvTransferRegistry()
        _prepare(registry, writers={3}, bounce=False)
        _publish(registry, 3)

        conflict = registry.record_result(KEY, 3, WriterResult.SUCCESS, WriterMode.BOUNCE)

        assert not conflict.accepted and conflict.conflict
        assert conflict.logical_state is LogicalState.FAILED
        assert conflict.physical_state is PhysicalState.IN_DOUBT
        snapshot = registry.context_snapshot(KEY)
        (writer,) = snapshot.writers
        assert writer.mode is WriterMode.BOUNCE
        assert writer.conflict

        (quiesced,) = registry.mark_backend_quiesced()
        assert quiesced.physical_state is PhysicalState.DRAINED

    def test_bounce_offer_accepts_mixed_direct_and_bounce_writers(self) -> None:
        registry = RecvTransferRegistry()
        _prepare(registry, bounce=True)
        _publish(registry, 3)
        _publish(registry, 7)

        direct = registry.record_result(KEY, 3, WriterResult.SUCCESS, WriterMode.DIRECT)
        ready = registry.record_result(KEY, 7, WriterResult.SUCCESS, WriterMode.BOUNCE)

        assert direct.accepted and not direct.conflict
        assert direct.logical_state is LogicalState.PENDING
        assert LifecycleAction.START_BOUNCE_SCATTER not in direct.actions
        assert ready.accepted and ready.actions == (LifecycleAction.START_BOUNCE_SCATTER,)
        assert ready.bounce_pending
        snapshot = registry.context_snapshot(KEY)
        assert snapshot is not None
        modes = {writer.rank: writer.mode for writer in snapshot.writers}
        assert modes == {3: WriterMode.DIRECT, 7: WriterMode.BOUNCE}
        settled = _settle_bounce(registry)
        assert settled.logical_state is LogicalState.SUCCEEDED
        assert settled.physical_state is PhysicalState.DRAINED

    def test_bounce_offer_with_only_direct_writers_skips_scatter(self) -> None:
        registry = RecvTransferRegistry()
        _prepare(registry, bounce=True)
        _publish(registry, 3)
        _publish(registry, 7)

        registry.record_result(KEY, 3, WriterResult.SUCCESS, WriterMode.DIRECT)
        terminal = registry.record_result(KEY, 7, WriterResult.SUCCESS, WriterMode.DIRECT)

        assert terminal.logical_state is LogicalState.SUCCEEDED
        assert terminal.physical_state is PhysicalState.DRAINING
        assert terminal.writers_terminal
        assert LifecycleAction.START_BOUNCE_SCATTER not in terminal.actions
        assert _actions(terminal) == {
            LifecycleAction.NOTIFY_SUCCESS,
            LifecycleAction.RELEASE_BOUNCE,
        }
        settled = _settle_bounce(registry)
        assert settled.logical_state is LogicalState.SUCCEEDED
        assert settled.physical_state is PhysicalState.DRAINED
        assert _actions(settled) == {LifecycleAction.CONTEXT_DRAINED}


class TestBounceSettlement:
    def test_bounce_success_waits_for_scatter(self) -> None:
        registry = RecvTransferRegistry()
        _prepare(registry)
        _publish(registry, 3)
        _publish(registry, 7)

        registry.record_result(KEY, 3, WriterResult.SUCCESS, WriterMode.BOUNCE)
        ready = registry.record_result(KEY, 7, WriterResult.SUCCESS, WriterMode.BOUNCE)

        assert ready.logical_state is LogicalState.PENDING
        assert ready.physical_state is PhysicalState.DRAINING
        assert ready.writers_terminal and ready.bounce_pending
        assert ready.actions == (LifecycleAction.START_BOUNCE_SCATTER,)
        assert registry.context_snapshot(KEY).bounce_state is BounceState.SCATTERING

        settled = _settle_bounce(registry)

        assert settled.logical_state is LogicalState.SUCCEEDED
        assert settled.physical_state is PhysicalState.DRAINED
        assert not settled.bounce_pending
        assert _actions(settled) == {
            LifecycleAction.NOTIFY_SUCCESS,
            LifecycleAction.CONTEXT_DRAINED,
        }

    def test_scatter_failure_is_logical_failure_but_physically_drained(self) -> None:
        registry = RecvTransferRegistry()
        _prepare(registry, writers={3}, bounce=True)
        _publish(registry, 3)
        ready = registry.record_result(KEY, 3, WriterResult.SUCCESS, WriterMode.BOUNCE)
        assert ready.bounce_pending

        failed = _settle_bounce(registry, succeeded=False)

        assert failed.logical_state is LogicalState.FAILED
        assert failed.physical_state is PhysicalState.DRAINED
        assert LifecycleAction.NOTIFY_FAILURE in failed.actions
        assert LifecycleAction.RELEASE_BOUNCE not in failed.actions

    def test_cancel_while_scatter_runs_waits_for_settlement(self) -> None:
        registry = RecvTransferRegistry()
        _prepare(registry, writers={3}, bounce=True)
        _publish(registry, 3)
        ready = registry.record_result(KEY, 3, WriterResult.SUCCESS, WriterMode.BOUNCE)
        assert ready.bounce_pending

        (cancelled,) = registry.cancel_request(KEY[0])
        assert cancelled.logical_state is LogicalState.CANCELLED
        assert cancelled.physical_state is PhysicalState.IN_DOUBT
        assert cancelled.bounce_pending
        assert LifecycleAction.RELEASE_BOUNCE not in cancelled.actions

        settled = _settle_bounce(registry)
        assert settled.logical_state is LogicalState.CANCELLED
        assert settled.physical_state is PhysicalState.DRAINED
        assert LifecycleAction.RELEASE_BOUNCE not in settled.actions


class TestTimeoutQuiescenceAndShutdown:
    def test_timeout_retains_exposed_resources_until_late_results(self) -> None:
        registry = RecvTransferRegistry()
        _prepare(registry)
        _publish(registry, 3)
        _publish(registry, 7)

        (timed_out,) = registry.timeout_request(KEY[0])

        assert timed_out.logical_state is LogicalState.FAILED
        assert timed_out.physical_state is PhysicalState.IN_DOUBT
        assert not timed_out.writers_terminal
        assert _actions(timed_out) == {LifecycleAction.NOTIFY_FAILURE}

        first = registry.record_result(KEY, 3, WriterResult.SUCCESS, WriterMode.BOUNCE)
        assert first.physical_state is PhysicalState.IN_DOUBT
        late = registry.record_result(KEY, 7, WriterResult.FAILED, WriterMode.BOUNCE)
        assert late.physical_state is PhysicalState.DRAINING
        assert late.writers_terminal
        assert LifecycleAction.START_BOUNCE_SCATTER not in late.actions
        assert LifecycleAction.RELEASE_BOUNCE in late.actions
        assert _settle_bounce(registry).physical_state is PhysicalState.DRAINED

    def test_backend_quiescence_recovers_lost_result_without_claiming_success(self) -> None:
        registry = RecvTransferRegistry()
        _prepare(registry)
        _publish(registry, 3)
        _publish(registry, 7)
        registry.record_result(KEY, 3, WriterResult.SUCCESS, WriterMode.BOUNCE)

        (quiesced,) = registry.mark_backend_quiesced()

        assert quiesced.logical_state is LogicalState.FAILED
        assert quiesced.physical_state is PhysicalState.DRAINING
        assert quiesced.writers_terminal
        assert LifecycleAction.START_BOUNCE_SCATTER not in quiesced.actions
        assert LifecycleAction.RELEASE_BOUNCE in quiesced.actions
        assert _settle_bounce(registry).physical_state is PhysicalState.DRAINED

    def test_consumer_detach_suppresses_notification_not_retirement(self) -> None:
        registry = RecvTransferRegistry()
        _prepare(registry, writers={3}, bounce=False)
        _publish(registry, 3)
        registry.detach_consumer(KEY[0])

        terminal = registry.record_result(KEY, 3, WriterResult.SUCCESS, WriterMode.DIRECT)

        assert terminal.logical_state is LogicalState.SUCCEEDED
        assert LifecycleAction.NOTIFY_SUCCESS not in terminal.actions
        assert LifecycleAction.CONTEXT_DRAINED in terminal.actions

    def test_request_drain_and_retirement_cover_every_slice(self) -> None:
        registry = RecvTransferRegistry()
        _prepare(registry, key=(101, 0), writers={3}, bounce=False)
        _prepare(registry, key=(101, 1), writers={3}, bounce=False)

        registry.cancel_request(101)
        assert registry.is_request_drained(101)
        assert registry.retire_request(101) == ((101, 0), (101, 1))
        assert registry.snapshot().contexts == ()

    def test_atomic_retirement_keeps_every_slice_when_one_is_not_drained(self) -> None:
        registry = RecvTransferRegistry()
        first_key = (101, 0)
        second_key = (101, 1)
        _prepare(registry, key=first_key, writers={3}, bounce=False)
        _prepare(registry, key=second_key, writers={3}, bounce=False)
        _publish(registry, 3, key=first_key)
        first = registry.record_result(first_key, 3, WriterResult.SUCCESS, WriterMode.DIRECT)
        assert first.physical_state is PhysicalState.DRAINED

        assert not registry.retire_request_if_drained(101)
        assert registry.context_snapshot(first_key) is not None
        assert registry.context_snapshot(second_key) is not None

        registry.cancel_request(101)
        assert registry.retire_request_if_drained(101)
        assert registry.context_snapshot(first_key) is None
        assert registry.context_snapshot(second_key) is None

    def test_target_mode_tracks_offer_and_is_removed_with_request_index(self) -> None:
        registry = RecvTransferRegistry()
        direct_key = (101, 0)
        bounce_key = (101, 1)
        other_request_key = (202, 0)
        _prepare(registry, key=direct_key, writers={3}, bounce=False)
        _prepare(registry, key=bounce_key, writers={3}, bounce=True)
        _prepare(registry, key=other_request_key, writers={7}, bounce=True)

        assert registry.target_mode(direct_key) is WriterMode.DIRECT
        assert registry.target_mode(bounce_key) is WriterMode.BOUNCE
        assert registry.target_mode((101, 99)) is None

        registry.cancel_request(101)
        _settle_bounce(registry, key=bounce_key)
        assert registry.retire_request_if_drained(101)
        assert registry.target_mode(direct_key) is None
        assert registry.target_mode(bounce_key) is None
        assert registry.target_mode(other_request_key) is WriterMode.BOUNCE

        # Reusing the retired request ID must not retain stale index entries.
        replacement_key = (101, 2)
        _prepare(registry, key=replacement_key, writers={11}, bounce=False)
        assert registry.target_mode(replacement_key) is WriterMode.DIRECT
        updates = registry.cancel_request(101)
        assert tuple(update.key for update in updates) == (replacement_key,)

    def test_shutdown_closes_admission_and_retains_possible_access(self) -> None:
        registry = RecvTransferRegistry()
        _prepare(registry)
        _publish(registry, 3)

        (shutdown,) = registry.begin_shutdown()

        assert not registry.snapshot().accepting
        assert shutdown.logical_state is LogicalState.CANCELLED
        assert shutdown.physical_state is PhysicalState.IN_DOUBT
        assert not registry.is_drained()
        rejected = registry.prepare((202, 0), {3}, has_bounce_slot=False)
        assert not rejected.accepted
        assert "closed" in rejected.reason

        (quiesced,) = registry.mark_backend_quiesced()
        assert quiesced.physical_state is PhysicalState.DRAINING
        assert not registry.is_drained()
        assert _settle_bounce(registry).physical_state is PhysicalState.DRAINED
        assert registry.is_drained()
        assert registry.begin_shutdown() == ()


class TestInitialContainmentOwnership:
    def test_terminal_slice_emits_only_initial_containment_actions(self) -> None:
        """The future allocator-backed phase is not part of this lifecycle API."""
        registry = RecvTransferRegistry()
        _prepare(registry, writers={3}, bounce=False)
        _publish(registry, 3)

        terminal = registry.record_result(KEY, 3, WriterResult.SUCCESS, WriterMode.DIRECT)

        assert terminal.physical_state is PhysicalState.DRAINED
        assert _actions(terminal) == {
            LifecycleAction.NOTIFY_SUCCESS,
            LifecycleAction.CONTEXT_DRAINED,
        }
