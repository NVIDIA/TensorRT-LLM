# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Executor-facing entry points for disaggregated KV transfer.

The executor loops call the disagg state machine only through a
``DisaggTransferCoordinator``. This module must not import ``PyExecutor`` or
hold a reference to it: everything it needs is injected as callables.
"""

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Callable, List, Tuple

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest

if TYPE_CHECKING:
    from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import ScheduledRequests


@dataclass(frozen=True)
class DisaggLoopDelegates:
    """Executor callables the coordinator forwards to.

    Transitional: each field is removed once the corresponding logic moves
    into the coordinator.
    """

    handle_errors_synced: Callable[[], None]
    prepare_context_schedulable: Callable[[List[LlmRequest]], None]
    poll_gen_transfers: Callable[[], None]
    check_transfer_timeouts: Callable[[], None]
    admit: Callable[[List[LlmRequest]], Tuple[List[LlmRequest], bool]]
    revert_deferred_gen_init: Callable[[List[LlmRequest], List[LlmRequest]], None]
    receive_gen_init: Callable[[List[LlmRequest]], None]
    poll_progress_when_idle: Callable[[], None]
    prepare_transmission_completed: Callable[["ScheduledRequests"], None]
    send_completed_context: Callable[[List[LlmRequest]], None]
    reap_context_sends: Callable[[int], None]
    pace_idle: Callable[[], None]


class DisaggTransferCoordinator:
    """Disagg transfer entry points used by every executor loop variant.

    Several methods perform rank-consensus collectives inside the transceiver
    or over ``dist``; every rank must call them the same number of times per
    iteration. The loops therefore call them unconditionally and rely on the
    coordinator (or ``NoopDisaggCoordinator``) to be a no-op when
    disaggregation is off.
    """

    def __init__(self, delegates: DisaggLoopDelegates) -> None:
        self._d = delegates

    # -- loop head -----------------------------------------------------------

    def handle_errors_synced(self) -> None:
        """Fail requests whose transfer errored; rank-synchronized."""
        self._d.handle_errors_synced()

    def prepare_context_schedulable(self, new_requests: List[LlmRequest]) -> None:
        """Let the transceiver gate generation-first context requests."""
        self._d.prepare_context_schedulable(new_requests)

    def poll_gen_transfers(self) -> None:
        """Poll receive-side transfers and their timeouts; rank-synchronized."""
        self._d.poll_gen_transfers()

    def check_transfer_timeouts(self) -> None:
        """Flag transfers that exceeded ``kv_transfer_timeout_ms``."""
        self._d.check_transfer_timeouts()

    # -- scheduling ----------------------------------------------------------

    def admit(self, fitting_gen_init: List[LlmRequest]) -> Tuple[List[LlmRequest], bool]:
        """Select the gen-init requests that may start receiving this iteration.

        Returns ``(admitted, blocked_by_active_transfers)``.
        """
        return self._d.admit(fitting_gen_init)

    def revert_deferred_gen_init(
        self, candidates: List[LlmRequest], admitted: List[LlmRequest]
    ) -> None:
        """Release KV allocated for candidates that were not admitted."""
        self._d.revert_deferred_gen_init(candidates, admitted)

    def receive_gen_init(self, admitted: List[LlmRequest]) -> None:
        """Prepare resources and start the KV receive for admitted requests."""
        self._d.receive_gen_init(admitted)

    def poll_progress_when_idle(self) -> None:
        """Reap completed context sends; rank-symmetric."""
        self._d.poll_progress_when_idle()

    # -- batch execution -----------------------------------------------------

    def prepare_transmission_completed(self, scheduled_batch: "ScheduledRequests") -> None:
        """Turn gen requests whose receive completed into running requests."""
        self._d.prepare_transmission_completed(scheduled_batch)

    def send_completed_context(self, requests: List[LlmRequest]) -> None:
        """Start async KV sends for finished context-only requests."""
        self._d.send_completed_context(requests)

    def reap_context_sends(self, at_least: int = 0) -> None:
        """Poll send-side transfers and release settled requests."""
        self._d.reap_context_sends(at_least)

    # -- loop tail -----------------------------------------------------------

    def pace_idle(self) -> None:
        """Sleep briefly when only a transfer completing can make progress."""
        self._d.pace_idle()


class NoopDisaggCoordinator(DisaggTransferCoordinator):
    """Coordinator used when the executor has no KV cache transceiver."""

    def __init__(self) -> None:
        super().__init__(
            DisaggLoopDelegates(**{f.name: _noop for f in fields(DisaggLoopDelegates)})
        )

    def admit(self, fitting_gen_init: List[LlmRequest]) -> Tuple[List[LlmRequest], bool]:
        return fitting_gen_init, False


def _noop(*_args, **_kwargs) -> None:
    return None
