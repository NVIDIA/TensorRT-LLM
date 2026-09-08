# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Executor-facing entry points for disaggregated KV transfer.

The executor loops call the disagg state machine only through a
``DisaggTransferCoordinator``. This module must not import ``PyExecutor`` or
hold a reference to it: services are injected, executor-owned behavior is
reached through the ``interfaces`` Protocols.
"""

import os
import time
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Callable, List, Set, Tuple

from tensorrt_llm._torch.disaggregation.kv_cache_transceiver import (
    is_disagg_inflight_cancel_enabled,
)
from tensorrt_llm._torch.distributed.communicator import ReduceOp
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestState
from tensorrt_llm._utils import nvtx_range
from tensorrt_llm.logger import logger

from .interfaces import ActiveRequestRegistry, ExecutorEffects

if TYPE_CHECKING:
    from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import ScheduledRequests


def is_gen_only_no_context_benchmark() -> bool:
    """Whether the ``gen_only_no_context`` benchmark skips KV transfer."""
    return os.getenv("TRTLLM_DISAGG_BENCHMARK_GEN_ONLY") == "1"


def uses_async_gen_transfer() -> bool:
    """Whether generation KV transfers can remain in flight across iterations."""
    return (
        not is_gen_only_no_context_benchmark()
        and os.getenv("TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP") != "1"
    )


def attach_ctx_usage(request: LlmRequest, response) -> None:
    """Copy gen-first context usage from the transfer aux data onto the response."""
    disagg_params = request.py_disaggregated_params
    if disagg_params is not None and disagg_params.ctx_usage is not None:
        response.result.ctx_usage = disagg_params.ctx_usage


@dataclass(frozen=True)
class DisaggLoopDelegates:
    """Executor callables the coordinator still forwards to.

    Transitional: each field is removed once the corresponding logic moves
    into the coordinator (CS-2: progress; CS-3: errors, admission, receive).
    """

    handle_errors_synced: Callable[[], None]
    prepare_context_schedulable: Callable[[List[LlmRequest]], None]
    admit: Callable[[List[LlmRequest]], Tuple[List[LlmRequest], bool]]
    revert_deferred_gen_init: Callable[[List[LlmRequest], List[LlmRequest]], None]
    receive_gen_init: Callable[[List[LlmRequest]], None]
    poll_progress_when_idle: Callable[[], None]
    prepare_transmission_completed: Callable[["ScheduledRequests"], None]
    # Rank-local transfer error handling; reached from the reaps. CS-3.
    check_transfer_errors: Callable[[str], None]
    requests_in_error_state: Callable[[], List[LlmRequest]]


class DisaggTransferCoordinator:
    """Disagg transfer entry points used by every executor loop variant.

    Several methods perform rank-consensus collectives inside the transceiver
    or over ``dist``; every rank must call them the same number of times per
    iteration. The loops therefore call them unconditionally and rely on the
    coordinator (or ``NoopDisaggCoordinator``) to be a no-op when
    disaggregation is off.
    """

    def __init__(
        self,
        *,
        transceiver,
        transfer_manager,
        kv_cache_manager,
        dist,
        effects: ExecutorEffects,
        registry: ActiveRequestRegistry,
        enable_attention_dp: bool,
        force_terminate_ctx_for_partial_reuse: bool,
        delegates: DisaggLoopDelegates,
        draft_kv_cache_manager=None,
    ) -> None:
        self._transceiver = transceiver
        self._transfers = transfer_manager
        self._kv_cache_manager = kv_cache_manager
        self._draft_kv_cache_manager = draft_kv_cache_manager
        self._dist = dist
        self._effects = effects
        self._registry = registry
        self._enable_attention_dp = enable_attention_dp
        self._force_terminate_ctx_for_partial_reuse = force_terminate_ctx_for_partial_reuse
        self._d = delegates
        # Context sends that failed after leaving the transfer manager; applied
        # at the next rank-synchronized error pass.
        self._pending_ctx_transfer_failures: Set[int] = set()
        # Timed-out generation requests whose error response waits for the
        # ADP-safe consensus point.
        self._pending_timed_out_requests: List[LlmRequest] = []
        # Requests whose timed-out transfer was already cancelled in flight.
        self._timed_out_ctx_cancelled_ids: Set[int] = set()
        self._timed_out_gen_cancelled_ids: Set[int] = set()
        self._inflight_cancel_unsupported_logged = False

    # -- loop head -----------------------------------------------------------

    def handle_errors_synced(self) -> None:
        """Fail requests whose transfer errored; rank-synchronized."""
        self._d.handle_errors_synced()

    def prepare_context_schedulable(self, new_requests: List[LlmRequest]) -> None:
        """Let the transceiver gate generation-first context requests."""
        self._d.prepare_context_schedulable(new_requests)

    @nvtx_range("poll_gen_transfers")
    def poll_gen_transfers(self) -> None:
        """Poll receive-side transfers and their timeouts; rank-synchronized."""
        if not uses_async_gen_transfer():
            return
        # Gen-transfer status performs cross-rank consensus internally. Enter
        # it symmetrically; ranks with no ready local future contribute an
        # empty ready set.
        self.reap_gen_receives(0)
        if self.inflight_cancel_active():
            self._cancel_timed_out_gen_transfers()
            self._check_gen_transfer_errors_consensus()

    @nvtx_range("check_transfer_timeouts")
    def check_transfer_timeouts(self, only_with_context_sends: bool = False) -> None:
        """Flag transfers that exceeded ``kv_transfer_timeout_ms``.

        ``only_with_context_sends`` keeps the post-batch call sites gated on an
        in-flight context send, as they were before the extraction.
        """
        if only_with_context_sends and not self._transfers.has_any_inflight_requests():
            return
        timeout_ms = self._transceiver.kv_transfer_timeout_ms
        if timeout_ms is None:
            return

        def flag_if_timed_out(req: LlmRequest, kind: str) -> None:
            if req.py_kv_transfer_start_time is None:
                return
            elapsed_ms = (time.monotonic() - req.py_kv_transfer_start_time) * 1000
            if elapsed_ms > timeout_ms and not req.py_kv_transfer_timed_out:
                verb = (
                    "Requesting cancellation for"
                    if self.inflight_cancel_active()
                    else "Observed timeout on"
                )
                logger.warning(
                    f"{verb} {kind} request {req.py_request_id} due to KV cache "
                    f"transfer timeout: elapsed {elapsed_ms:.0f}ms > "
                    f"kv_transfer_timeout_ms={timeout_ms}ms"
                )
                req.py_kv_transfer_timed_out = True

        # Context requests start their clock on the last chunk, which is also
        # when they enter the transfer manager, so this covers the whole
        # context side.
        for req in self._transfers.requests_in_transfer().values():
            flag_if_timed_out(req, "context")
        for req in self._registry.active_requests():
            if req.is_disagg_generation_transmission_in_progress:
                flag_if_timed_out(req, "generation")

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
        # Do not send more chunks after an in-flight cancellation.
        cancel_pending_ids = set(self._registry.canceled_request_ids())
        for req in requests:
            if not req.is_context_only_request or req.is_finished_due_to_cancellation:
                continue
            request_id = req.parent_request_id if req.is_child else req.py_request_id
            if request_id in cancel_pending_ids:
                continue
            if self._transceiver.has_retired_send_session(req):
                # The peer registration went away with the session, so no
                # further slice can land.
                continue
            if req.is_context_finished or req.is_finished_due_to_length:
                # Forward is done: release the IndexMapper slot on every KV
                # manager that has one so new requests can reuse it. KV blocks
                # stay allocated for the transfer.
                for manager in (self._kv_cache_manager, self._draft_kv_cache_manager):
                    if hasattr(manager, "release_index_slot"):
                        manager.release_index_slot(req.py_request_id)
                # start_transfer commits the request's blocks to the reuse tree
                # and pins them; it must run before respond_and_send_async
                # sends the final slice and (for the Python transceiver) moves
                # the request toward completion.
                self._transfers.start_transfer(req)
                self._transceiver.respond_and_send_async(req)
                if self._transceiver.kv_transfer_timeout_ms is not None:
                    req.py_kv_transfer_start_time = time.monotonic()
            elif (
                self._transceiver.pipeline_transfer_enabled
                and req.state != LlmRequestState.GENERATION_COMPLETE
            ):
                # Intermediate chunk of a pipelined transfer. GENERATION_COMPLETE
                # means an error path already failed and freed this request, so
                # its chunk bounds are unset.
                self._transceiver.respond_and_send_async(req)

    @nvtx_range("reap_context_sends")
    def reap_context_sends(self, at_least: int = 0) -> None:
        """Poll send-side transfers and release settled requests."""
        ctx_status = self._transceiver.check_context_transfer_status(at_least)
        failed_req_ids = set(ctx_status.error_request_ids)
        completed_req_ids = set(ctx_status.completed_request_ids) | failed_req_ids

        requests_in_transfer = self._transfers.requests_in_transfer()
        for request_id in completed_req_ids:
            if request_id not in requests_in_transfer:
                if request_id in failed_req_ids:
                    self._pending_ctx_transfer_failures.add(request_id)
                else:
                    logger.warning(f"Request {request_id} not found in transfer manager")
                continue
            request = requests_in_transfer[request_id]
            if request_id in failed_req_ids:
                # Past the context phase: writing the error state here is safe.
                request.state = LlmRequestState.DISAGG_TRANS_ERROR
            self.release_transfer(request)

        # Releases above may have changed the set of requests in transfer.
        requests_in_transfer = self._transfers.requests_in_transfer()
        for request_id in list(requests_in_transfer.keys()):
            request = requests_in_transfer[request_id]
            if (
                not request.py_kv_transfer_timed_out
                or request_id in completed_req_ids
                or request_id in self._timed_out_ctx_cancelled_ids
            ):
                continue
            if not self.request_cancellation(request):
                continue
            if self.inflight_cancel_active():
                self._timed_out_ctx_cancelled_ids.add(request_id)
                logger.warning(
                    f"Cancelled timed-out context KV transfer for request "
                    f"{request.py_request_id}; waiting for C++ transfer status "
                    "to report final cleanup"
                )
            else:
                # Legacy timeout behavior: a queued transfer that can be
                # cancelled is released from the async manager immediately.
                request.py_kv_transfer_start_time = None
                request.state = LlmRequestState.DISAGG_CONTEXT_COMPLETE
                self.release_transfer(request)

        self._d.check_transfer_errors("context requests")

    @nvtx_range("reap_gen_receives")
    def reap_gen_receives(self, at_least: int = 0) -> None:
        """Poll receive-side transfers; rank-synchronized inside the transceiver."""
        gen_status = self._transceiver.check_gen_transfer_status(at_least)
        if gen_status.cancelled_requests:
            user_canceled_ids = set(self._registry.canceled_request_ids())
            for req in gen_status.cancelled_requests:
                req_id = req.parent_request_id if req.is_child else req.py_request_id
                if req_id not in user_canceled_ids:
                    req.state = LlmRequestState.DISAGG_TRANS_ERROR
        if not self.inflight_cancel_active():
            self._d.check_transfer_errors("generation requests")

    def release_transfer(self, request: LlmRequest) -> None:
        """Release one transfer claim and terminate once the last owner releases.

        The transceiver and the KV connector can both hold a claim on the same
        request; ``AsyncTransferManager`` counts them. A request that is still
        active gets its response created here (the transfer completed before
        the response pass could run) and staged for the rank-synchronized
        flush; a failed transfer only releases its claim so the synchronized
        error path can respond once every owner is done.
        """
        transfer_failed = request.state == LlmRequestState.DISAGG_TRANS_ERROR
        if self._registry.contains(request):
            if transfer_failed:
                self._transfers.end_transfer(request)
                return
            # Create the response while the state is still TRANS_IN_PROGRESS
            # (required by C++ createResult).
            response = request.create_response(False, self._dist.rank)
            if response:
                response.result.cached_tokens = request.cached_tokens
                attach_ctx_usage(request, response)
            released = self._transfers.end_transfer(request)
            if released:
                self._registry.remove(request)
            if response:
                self._effects.stage_transfer_response(
                    request.py_request_id, response, request if released else None
                )
            elif released:
                self._effects.terminate_request(request)
            return
        if self._transfers.end_transfer(request):
            if transfer_failed:
                return
            # Skip if the PP=1 early path already terminated this request;
            # under PP>1 that path is off, so terminate here on completion.
            if not self._force_terminate_ctx_for_partial_reuse:
                self._effects.terminate_request(request)

    # -- timeouts and cancellation -------------------------------------------

    def inflight_cancel_active(self) -> bool:
        """Whether timed-out transfers are cancelled in flight."""
        if not is_disagg_inflight_cancel_enabled():
            return False
        supports = getattr(self._transceiver, "supports_inflight_request_cancellation", None)
        if callable(supports) and supports() is True:
            return True
        if not self._inflight_cancel_unsupported_logged:
            logger.warning(
                "TRTLLM_DISAGG_ENABLE_INFLIGHT_CANCEL=1 was requested, but "
                f"{type(self._transceiver).__name__} does not advertise in-flight "
                "request cancellation support. Cancellation and transfer-buffer "
                "quarantine are currently scoped to the C++ NIXL transceiver "
                "with the UCX plugin; using the existing timeout and "
                "cancellation behavior for this transceiver."
            )
            self._inflight_cancel_unsupported_logged = True
        return False

    def request_cancellation(self, request: LlmRequest) -> bool:
        """Best-effort cancellation that leaves ownership intact on errors."""
        try:
            return self._transceiver.cancel_request(request)
        except Exception as error:
            logger.error(
                f"KV transfer cancellation failed for request "
                f"{request.py_request_id}; will retry: {error}"
            )
            return False

    def fail_timed_out(self, requests: List[LlmRequest]) -> None:
        """Fail generation requests whose transfer timed out.

        Under multi-rank ADP the error response enters a collective, so it is
        deferred to ``handle_timeouts_synced``.
        """
        if self._enable_attention_dp and self._dist.world_size != 1:
            self._pending_timed_out_requests.extend(requests)
            return
        for req in requests:
            self._effects.fail_requests(
                f"Request {req.py_request_id} timed out", [req], charge_budget=False
            )

    def handle_timeouts_synced(self) -> None:
        """ADP-safe drain of the KV-transfer-timeout consensus collective.

        Reached the same number of times on every rank per iteration; non-ADP
        runs failed timeouts inline and the buffer is empty here.
        """
        if not (self._enable_attention_dp and self._dist.world_size != 1):
            return
        timed_out = self._pending_timed_out_requests
        self._pending_timed_out_requests = []
        any_timed_out = bool(self._dist.tp_allgather_int64([bool(timed_out)]).any())
        if any_timed_out:
            self._effects.fail_requests(
                "Request timed out (KV transfer)", timed_out, charge_budget=False
            )

    def take_pending_context_failures(self) -> Set[int]:
        """Drain context sends that failed after leaving the transfer manager."""
        pending = self._pending_ctx_transfer_failures
        self._pending_ctx_transfer_failures = set()
        return pending

    def forget_request(self, request_id: int) -> None:
        """Drop per-request cancellation bookkeeping once the request is freed."""
        self._timed_out_ctx_cancelled_ids.discard(request_id)
        self._timed_out_gen_cancelled_ids.discard(request_id)

    @nvtx_range("cancel_timed_out_gen_transfers")
    def _cancel_timed_out_gen_transfers(self) -> None:
        """Request cancellation for timed-out generation transfers.

        Rank-synchronized: under attention-DP each TP rank owns a different
        request subset, but the later error responses pass through a TP
        collective, so the decision is based on the TP-wide id union rather
        than a rank-local timeout observation.
        """
        timeout_ms = self._transceiver.kv_transfer_timeout_ms
        if timeout_ms is None:
            return

        requests_in_transfer = {
            req.py_request_id: req
            for req in self._registry.active_requests()
            if req.is_disagg_generation_transmission_in_progress
        }
        current_time = time.monotonic()
        for request in requests_in_transfer.values():
            if request.py_kv_transfer_start_time is None:
                continue
            elapsed_ms = (current_time - request.py_kv_transfer_start_time) * 1000
            if elapsed_ms > timeout_ms and not request.py_kv_transfer_timed_out:
                logger.warning(
                    f"Requesting cancellation for generation request "
                    f"{request.py_request_id} due to KV cache transfer timeout"
                )
                request.py_kv_transfer_timed_out = True

        user_canceled_ids = set(self._registry.canceled_request_ids())
        local_timed_out_ids = sorted(
            request_id
            for request_id, request in requests_in_transfer.items()
            if request.py_kv_transfer_timed_out
            and request_id not in user_canceled_ids
            and request_id not in self._timed_out_gen_cancelled_ids
        )

        if self._dist.tp_size > 1:
            any_timed_out = self._dist.tp_allreduce(int(bool(local_timed_out_ids)), op=ReduceOp.MAX)
        else:
            any_timed_out = int(bool(local_timed_out_ids))
        if not any_timed_out:
            return

        if self._dist.tp_size > 1:
            gathered = self._dist.tp_allgather(local_timed_out_ids)
            timed_out_ids = sorted(set().union(*gathered))
        else:
            timed_out_ids = local_timed_out_ids

        for request_id in timed_out_ids:
            request = requests_in_transfer.get(request_id)
            if request is None:
                continue
            # A peer rank may have crossed the timeout first. Mirror the
            # TP-wide decision locally so a failed cancel attempt keeps
            # retrying even if this rank's wall clock had not yet expired.
            request.py_kv_transfer_timed_out = True
            if request_id in self._timed_out_gen_cancelled_ids:
                continue
            if self.request_cancellation(request):
                self._timed_out_gen_cancelled_ids.add(request_id)
                logger.warning(
                    f"Cancelled timed-out generation KV transfer for request "
                    f"{request.py_request_id}; waiting for C++ transfer status "
                    "to report final cleanup"
                )

    @nvtx_range("check_gen_transfer_errors_consensus")
    def _check_gen_transfer_errors_consensus(self) -> None:
        """Flush generation transfer errors through a TP-uniform path."""
        error_requests = [
            req for req in self._d.requests_in_error_state() if req.is_generation_only_request
        ]
        local_needs_flush = bool(error_requests)
        if self._dist.tp_size > 1:
            any_needs_flush = self._dist.tp_allreduce(int(local_needs_flush), op=ReduceOp.MAX)
        else:
            any_needs_flush = int(local_needs_flush)
        if not any_needs_flush:
            return
        self._effects.fail_requests(
            "Error in kv cache transfer for generation requests",
            error_requests,
            charge_budget=False,
        )

    # -- loop tail -----------------------------------------------------------

    def pace_idle(self) -> None:
        """Sleep briefly when only a KV transfer completing can make progress.

        Call this at the end of an iteration that queued nothing, after the
        pass has drained its ready work, so the pending-transfer check sees the
        state that work left behind. The check is rank-local; the sleep only
        paces and never gates a collective.
        """
        # Context sends are tracked by the transfer manager; generation
        # receives live in the request state, so both directions are covered.
        waiting_on_transfer = self._transfers.has_any_inflight_requests() or any(
            req.is_disagg_generation_init_state or req.is_disagg_generation_transmission_in_progress
            for req in self._registry.active_requests()
        )
        if waiting_on_transfer:
            time.sleep(0.001)


class NoopDisaggCoordinator(DisaggTransferCoordinator):
    """Coordinator used when the executor has no KV cache transceiver.

    A null object: no state, no dependencies, no side effects. The loops call
    every entry point unconditionally, so each one is a no-op here. Transceiver
    enablement is rank-uniform within a collective group, so every rank skips
    the same collectives symmetrically.
    """

    def __init__(self) -> None:
        super().__init__(
            transceiver=None,
            transfer_manager=None,
            kv_cache_manager=None,
            dist=None,
            effects=None,
            registry=None,
            enable_attention_dp=False,
            force_terminate_ctx_for_partial_reuse=False,
            delegates=DisaggLoopDelegates(**{f.name: _noop for f in fields(DisaggLoopDelegates)}),
        )

    def admit(self, fitting_gen_init: List[LlmRequest]) -> Tuple[List[LlmRequest], bool]:
        return fitting_gen_init, False

    def poll_gen_transfers(self) -> None:
        return None

    def check_transfer_timeouts(self, only_with_context_sends: bool = False) -> None:
        return None

    def send_completed_context(self, requests: List[LlmRequest]) -> None:
        return None

    def reap_context_sends(self, at_least: int = 0) -> None:
        return None

    def reap_gen_receives(self, at_least: int = 0) -> None:
        return None

    def release_transfer(self, request: LlmRequest) -> None:
        raise RuntimeError(
            "release_transfer has no transceiver to serve; connector-only "
            "releases are handled by the executor"
        )

    def inflight_cancel_active(self) -> bool:
        return False

    def request_cancellation(self, request: LlmRequest) -> bool:
        return True

    def fail_timed_out(self, requests: List[LlmRequest]) -> None:
        # Without a transceiver no request can time out on a KV transfer.
        return None

    def handle_timeouts_synced(self) -> None:
        return None

    def take_pending_context_failures(self) -> Set[int]:
        return set()

    def forget_request(self, request_id: int) -> None:
        return None

    def pace_idle(self) -> None:
        return None


def _noop(*_args, **_kwargs) -> None:
    return None
