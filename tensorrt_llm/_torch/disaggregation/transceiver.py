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

import os
import threading
import time
import uuid
from collections import OrderedDict, defaultdict
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, cast

import numpy as np
import torch

from tensorrt_llm import logger
from tensorrt_llm._torch.disaggregation.async_consensus import (
    PROTOCOL_VERSION,
    AsyncConsensusCoordinator,
    ConsensusEvent,
    ConsensusEventKind,
    ConsensusOutcome,
    MpiConsensusTransport,
)
from tensorrt_llm._torch.disaggregation.base.transfer import (
    KVSlice,
    RxSessionBase,
    SessionStatus,
    TokenRange,
    TxSessionBase,
    WaitResult,
    get_unique_rid,
)
from tensorrt_llm._torch.disaggregation.native.bounce import (
    config_from_size as bounce_config_from_size,
)
from tensorrt_llm._torch.disaggregation.native.transfer import TransferWorker, TransferWorkerConfig
from tensorrt_llm._torch.disaggregation.resource.cache_reuse import (
    CacheReuseAdapter,
    create_cache_reuse_adapter,
)
from tensorrt_llm._torch.disaggregation.resource.page import MambaLayerGroup
from tensorrt_llm._torch.disaggregation.resource.utils import get_physical_pool
from tensorrt_llm._torch.distributed.communicator import Distributed, MPIDist
from tensorrt_llm._torch.pyexecutor.kv_cache_transceiver import KvCacheTransceiver
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.mamba_cache_manager import MambaHybridCacheManager
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._utils import nvtx_range
from tensorrt_llm.bindings import LlmRequestState
from tensorrt_llm.bindings.executor import ContextPhaseParams
from tensorrt_llm.disaggregated_params import DisaggScheduleStyle
from tensorrt_llm.llmapi.llm_args import CacheTransceiverConfig
from tensorrt_llm.mapping import Mapping

_ASYNC_TERMINAL_ENV = "TRTLLM_PYTHON_TRANSCEIVER_ASYNC_CTX_TERMINAL_CONSENSUS"
_ASYNC_PEER_READY_ENV = "TRTLLM_PYTHON_TRANSCEIVER_ASYNC_CTX_PEER_READY_CONSENSUS"
_ASYNC_STARTUP_TAG = "TRTLLM_PYTHON_TRANSCEIVER_ASYNC_CONSENSUS"
_ASYNC_READY_CANCELLED_EPOCH_ATTR = "_trtllm_async_ready_cancelled_epoch"
_DIAG_ENV = "TRTLLM_PYTHON_TRANSCEIVER_DIAG"
_DIAG_PREFIX = "PYTHON_CONSENSUS_DIAG"
_DIAG_MAX_SAMPLED_REQUESTS = 16
_DIAG_MAX_TRACKED_REQUESTS_PER_STATE = 1024
_DIAG_MAX_SLOW_COLLECTIVE_LOGS = 8
_DIAG_SLOW_COLLECTIVE_S = 0.1
_DIAG_SUMMARY_INTERVAL_S = 60.0
_MAX_RETIRED_CONSENSUS_REQUESTS = 65536
_STARTUP_ROLLBACK_TIMEOUT_S = 30.0
_CONSENSUS_STARTUP_CLOSE_TIMEOUT_S = 1.0


class _PythonTransceiverDiagnostics:
    """Bounded, opt-in observability for the Python transfer lifecycle."""

    def __init__(
        self,
        rank: int,
        *,
        clock: Callable[[], float] = time.monotonic,
        summary_interval_s: float = _DIAG_SUMMARY_INTERVAL_S,
    ) -> None:
        self._rank = rank
        self._clock = clock
        self._summary_interval_s = summary_interval_s
        self._started_at = clock()
        self._last_summary_at = self._started_at - summary_interval_s
        self._last_poll_at: Optional[float] = None
        self._max_poll_gap_s = 0.0
        self._poll_count = 0
        self._transition_counts: Dict[str, int] = defaultdict(int)
        self._sampled_request_ids: OrderedDict[int, None] = OrderedDict()
        self._seen_transitions: set[tuple[str, int, int]] = set()
        self._active_states: Dict[str, OrderedDict[int, float]] = defaultdict(OrderedDict)
        self._collective_stats: Dict[str, Dict[str, float | int]] = {}
        self._active_collectives: Dict[str, tuple[int, float]] = {}

    def _elapsed_s(self, now: Optional[float] = None) -> float:
        return (self._clock() if now is None else now) - self._started_at

    def _is_sampled(self, request_id: int) -> bool:
        if request_id in self._sampled_request_ids:
            return True
        if len(self._sampled_request_ids) >= _DIAG_MAX_SAMPLED_REQUESTS:
            return False
        self._sampled_request_ids[request_id] = None
        return True

    def record_transition(
        self,
        transition: str,
        request_id: int,
        epoch: int = 0,
        outcome: Optional[str] = None,
        duration_s: Optional[float] = None,
        *,
        once: bool = False,
    ) -> None:
        key = (transition, request_id, epoch)
        if once and key in self._seen_transitions:
            return
        if len(self._seen_transitions) < _MAX_RETIRED_CONSENSUS_REQUESTS:
            self._seen_transitions.add(key)
        self._transition_counts[transition] += 1
        if not self._is_sampled(request_id):
            return
        outcome_text = "" if outcome is None else f" outcome={outcome}"
        duration_text = "" if duration_s is None else f" duration_s={duration_s:.6f}"
        logger.info(
            f"{_DIAG_PREFIX} event=transition transition={transition} "
            f"rank={self._rank} request_id={request_id} epoch={epoch} "
            f"elapsed_s={self._elapsed_s():.6f}{outcome_text}{duration_text}"
        )

    def enter_state(self, state: str, request_id: int) -> None:
        active = self._active_states[state]
        if request_id in active:
            return
        if len(active) >= _DIAG_MAX_TRACKED_REQUESTS_PER_STATE:
            return
        active[request_id] = self._clock()

    def leave_state(self, state: str, request_id: int) -> None:
        self._active_states[state].pop(request_id, None)

    def record_poll(self) -> None:
        now = self._clock()
        if self._last_poll_at is not None:
            self._max_poll_gap_s = max(self._max_poll_gap_s, now - self._last_poll_at)
        self._last_poll_at = now
        self._poll_count += 1

    def snapshot_due(self, *, force: bool = False) -> bool:
        return force or self._clock() - self._last_summary_at >= self._summary_interval_s

    def collective_enter(self, label: str) -> int:
        now = self._clock()
        stats = self._collective_stats.setdefault(
            label,
            {"count": 0, "total_s": 0.0, "max_s": 0.0, "slow_count": 0},
        )
        sequence = int(stats["count"]) + 1
        stats["count"] = sequence
        if sequence <= _DIAG_MAX_SLOW_COLLECTIVE_LOGS or sequence % 512 == 0:
            logger.info(
                f"{_DIAG_PREFIX} event=collective_enter label={label} "
                f"rank={self._rank} sequence={sequence} elapsed_s={self._elapsed_s(now):.6f}"
            )
        self._active_collectives[label] = (sequence, self._clock())
        return sequence

    def collective_exit(self, label: str, sequence: int) -> None:
        active = self._active_collectives.pop(label, None)
        if active is None:
            return
        active_sequence, started_at = active
        if active_sequence != sequence:
            return
        now = self._clock()
        duration_s = now - started_at
        stats = self._collective_stats[label]
        stats["total_s"] = float(stats["total_s"]) + duration_s
        stats["max_s"] = max(float(stats["max_s"]), duration_s)
        should_log = sequence <= _DIAG_MAX_SLOW_COLLECTIVE_LOGS or sequence % 512 == 0
        if duration_s >= _DIAG_SLOW_COLLECTIVE_S:
            stats["slow_count"] = int(stats["slow_count"]) + 1
            slow_count = int(stats["slow_count"])
            should_log = should_log or slow_count <= _DIAG_MAX_SLOW_COLLECTIVE_LOGS
        if should_log:
            logger.info(
                f"{_DIAG_PREFIX} event=collective_exit label={label} "
                f"rank={self._rank} sequence={sequence} duration_s={duration_s:.6f} "
                f"slow={int(duration_s >= _DIAG_SLOW_COLLECTIVE_S)} "
                f"elapsed_s={self._elapsed_s(now):.6f}"
            )

    def maybe_log_snapshot(
        self,
        state: Dict[str, Any],
        coordinator: Optional[Dict[str, Any]],
        *,
        force: bool = False,
        teardown: bool = False,
    ) -> None:
        now = self._clock()
        if not force and now - self._last_summary_at < self._summary_interval_s:
            return
        self._last_summary_at = now
        active_states = {}
        for name, requests in sorted(self._active_states.items()):
            if not requests:
                continue
            oldest_started_at = next(iter(requests.values()))
            active_states[name] = {
                "count": len(requests),
                "oldest_s": round(now - oldest_started_at, 6),
                "sample_ids": list(requests)[:_DIAG_MAX_SAMPLED_REQUESTS],
            }
        collective_stats = {
            label: {
                "count": int(stats["count"]),
                "total_s": round(float(stats["total_s"]), 6),
                "max_s": round(float(stats["max_s"]), 6),
                "slow_count": int(stats["slow_count"]),
            }
            for label, stats in sorted(self._collective_stats.items())
        }
        active_collectives = {
            label: {
                "sequence": sequence,
                "elapsed_s": round(now - started_at, 6),
            }
            for label, (sequence, started_at) in sorted(self._active_collectives.items())
        }
        logger.info(
            f"{_DIAG_PREFIX} event=snapshot rank={self._rank} "
            f"elapsed_s={self._elapsed_s(now):.6f} teardown={int(teardown)} "
            f"poll_count={self._poll_count} max_poll_gap_s={self._max_poll_gap_s:.6f} "
            f"transitions={dict(sorted(self._transition_counts.items()))} "
            f"active_states={active_states} collectives={collective_stats} "
            f"active_collectives={active_collectives} state={state} "
            f"coordinator={coordinator}"
        )


def _find_consensus_request_ids(request_ids_all_ranks, sync_size):
    frequency_map = defaultdict(int)
    consensus = []
    for rid in chain.from_iterable(request_ids_all_ranks):
        frequency_map[rid] += 1
    for rid, freq in sorted(frequency_map.items(), key=lambda x: x[1], reverse=True):
        if freq == sync_size:
            consensus.append(rid)
        else:
            break
    return consensus


class KvCacheTransceiverV2(KvCacheTransceiver):
    def __init__(
        self,
        mapping: Mapping,
        dist: Distributed,
        kv_cache_manager: KVCacheManager,
        cache_transceiver_config: CacheTransceiverConfig,
    ):
        self._dist: Distributed = dist
        self._kv_cache_manager = kv_cache_manager
        self._mapping = mapping
        self.kv_transfer_timeout_ms = cache_transceiver_config.kv_transfer_timeout_ms
        self.kv_transfer_poll_interval_ms = cache_transceiver_config.kv_transfer_poll_interval_ms
        self._sender_future_timeout_ms = (
            cache_transceiver_config.kv_transfer_sender_future_timeout_ms
        )
        # Initialize optional consensus state without adding a collective to
        # the default-off startup path. Explicit opt-in is negotiated later by
        # piggybacking on the existing endpoint exchange.
        self._init_async_consensus(cache_transceiver_config)
        self._check_compatible()
        self._init_sync_policy()
        self._reuse_adapter: CacheReuseAdapter = create_cache_reuse_adapter(kv_cache_manager)

        # Initialize teardown-visible ownership before native workers start.
        # Explicit consensus startup happens after TransferWorker construction,
        # so constructor rollback must be able to use the normal shutdown path
        # even when communicator negotiation fails partway through __init__.
        self._send_sessions: Dict[int, TxSessionBase] = {}
        self._recv_sessions: Dict[int, RxSessionBase] = {}
        self._send_reqs = {}
        self._recv_reqs = {}
        self._wait_reqs = {}
        self._legacy_failed_sessions: set[int] = set()
        self._shutdown = False
        self._shutdown_complete = False
        self._shutdown_sessions_complete = False
        self._shutdown_consensus_complete = False
        self._shutdown_worker_complete = False
        self._shutdown_worker_event: Optional[threading.Event] = None
        self._shutdown_deferred_errors: list[tuple[str, Exception]] = []

        self._device_id = torch.cuda.current_device()
        logger.info(f"device_id: {self._device_id} in KvCacheTransceiverV2")
        self._instance_name = self._broadcast_instance_name()
        self._transfer_worker = TransferWorker(
            TransferWorkerConfig(
                kv_cache_manager=kv_cache_manager,
                device_id=self._device_id,
                instance_name=self._instance_name,
                # Context-only requests are released after KV transfer completes, so many batches
                # can be in-flight simultaneously. AuxBuffer holds only small CPU metadata, so a
                # large multiplier is cheap.
                max_concurrent_sessions=max(1, int(kv_cache_manager.max_batch_size)) * 20000,
                tx_timeout_s=self._sender_future_timeout_ms / 1000.0,
                rx_timeout_s=self.kv_transfer_timeout_ms / 1000.0,
                # Size 0 turns bounce off; the block-count gate is internal (tuned via env).
                bounce=bounce_config_from_size(cache_transceiver_config.kv_cache_bounce_size_mb),
            )
        )
        self._dp_rank = mapping.tp_rank if mapping.enable_attention_dp else 0
        try:
            self._context_info_endpoint = self._broadcast_context_endpoint()
            self._exchange_rank_info()
        except Exception:
            self._rollback_failed_startup()
            raise

        self._page_table = self._transfer_worker.page_table
        # _slice_num_bytes() is this rank's KV shard, so scale by tp_size to get the request total (kv_cache_size),
        # except under attention DP where the local count already is the total.
        self._kv_size_rank_factor = 1 if mapping.enable_attention_dp else max(1, mapping.tp_size)

        # Sticky role markers; flip True once any session opens, used to short-circuit
        # per-iter tp_allgather when this transceiver never sends/receives.
        self._ever_had_send_session: bool = False
        self._ever_had_recv_session: bool = False

    @staticmethod
    def _parse_binary_env(name: str, value: str) -> bool:
        if value not in ("0", "1"):
            raise ValueError(f"{name} must be 0 or 1, got {value!r}")
        return value == "1"

    def _rollback_failed_startup(self) -> None:
        """Release native and consensus resources after constructor failure."""
        try:
            worker_event = self.shutdown()
            if isinstance(worker_event, threading.Event):
                if not worker_event.wait(_STARTUP_ROLLBACK_TIMEOUT_S):
                    logger.error(
                        "Python transceiver startup rollback timed out waiting "
                        "for deferred native shutdown"
                    )
                    return
                self.shutdown()
        except Exception as error:
            # Preserve the startup exception. shutdown() already attempts every
            # independent teardown step before surfacing its first error.
            logger.error(f"Python transceiver startup rollback failed: {error}")

    def _init_async_consensus(self, cache_transceiver_config: CacheTransceiverConfig) -> None:
        terminal_value = os.getenv(_ASYNC_TERMINAL_ENV, "0")
        peer_ready_value = os.getenv(_ASYNC_PEER_READY_ENV, "0")
        diag_value = os.getenv(_DIAG_ENV, "0")
        diag_enabled = diag_value == "1"
        if diag_value not in ("0", "1"):
            logger.warning(
                f"Ignoring invalid {_DIAG_ENV}={diag_value!r}; diagnostics remain disabled"
            )
        self._async_terminal_flag_value = terminal_value
        self._async_peer_ready_flag_value = peer_ready_value
        self._async_consensus_config = cache_transceiver_config
        self._diagnostics: Optional[_PythonTransceiverDiagnostics] = None
        if diag_enabled:
            self._diagnostics = _PythonTransceiverDiagnostics(int(self._dist.rank))
            logger.info(
                f"{_DIAG_PREFIX} event=mode_active rank={self._dist.rank} "
                f"world_size={self._mapping.world_size} tp_size={self._mapping.tp_size} "
                f"pp_size={self._mapping.pp_size} cp_size={self._mapping.cp_size} "
                f"attention_dp={int(self._mapping.enable_attention_dp)}"
            )

        self._async_terminal_consensus_enabled = False
        self._async_peer_ready_consensus_enabled = False
        self._async_consensus: Optional[AsyncConsensusCoordinator] = None
        self._async_terminal_epoch: OrderedDict[int, int] = OrderedDict()
        self._async_terminal_published: Dict[int, int] = {}
        self._async_terminal_commits: Dict[int, ConsensusEvent] = {}
        self._async_terminal_cancelled: Dict[int, tuple[int, LlmRequest]] = {}
        self._context_cancelled_request_ids: list[int] = []
        self._async_ready_epoch: OrderedDict[int, int] = OrderedDict()
        self._async_ready_published: Dict[int, int] = {}
        self._async_ready_prepared: Dict[tuple[int, int], LlmRequest] = {}
        self._async_ready_released: set[tuple[int, int]] = set()
        self._async_ready_activated: Dict[tuple[int, int], LlmRequest] = {}
        self._async_ready_acknowledged: set[tuple[int, int]] = set()
        self._async_ready_withdrawn: set[tuple[int, int]] = set()
        # READY_ABORT is intentionally broadcast to every participant, even
        # when a rank has not materialized the request locally yet.  Keep a
        # request-less tombstone until the request arrives (or until a bounded
        # post-finalize replay window consumes it); acknowledging the abort
        # must not depend on local request timing.
        self._async_ready_aborted: Dict[tuple[int, int], Optional[LlmRequest]] = {}
        self._async_ready_finalized_without_request: OrderedDict[int, int] = OrderedDict()
        self._async_consensus_counters: Dict[str, int] = defaultdict(int)

        # A singleton has no cross-rank state to reconcile. Parse locally and
        # treat either opt-in as a no-op instead of constructing a communicator.
        if self._mapping.world_size == 1:
            terminal_requested = self._parse_binary_env(_ASYNC_TERMINAL_ENV, terminal_value)
            peer_ready_requested = self._parse_binary_env(_ASYNC_PEER_READY_ENV, peer_ready_value)
            if terminal_requested or peer_ready_requested:
                logger.info(
                    "PYTHON_ASYNC_CONSENSUS transition=singleton_noop "
                    f"terminal={int(terminal_requested)} "
                    f"peer_ready={int(peer_ready_requested)}"
                )
            return

        # Multi-rank parsing and qualification happen in _exchange_rank_info.
        # A malformed or mismatched same-version opt-in must reach that
        # universally-entered allgather before every new worker rejects it.

    def _record_diag_transition(
        self,
        transition: str,
        request_id: int,
        epoch: int = 0,
        outcome: Optional[str] = None,
        duration_s: Optional[float] = None,
        *,
        once: bool = False,
    ) -> None:
        diagnostics = getattr(self, "_diagnostics", None)
        if diagnostics is not None:
            diagnostics.record_transition(
                transition,
                request_id,
                epoch,
                outcome,
                duration_s,
                once=once,
            )

    def _enter_diag_state(self, state: str, request_id: int) -> None:
        diagnostics = getattr(self, "_diagnostics", None)
        if diagnostics is not None:
            diagnostics.enter_state(state, request_id)

    def _leave_diag_state(self, state: str, request_id: int) -> None:
        diagnostics = getattr(self, "_diagnostics", None)
        if diagnostics is not None:
            diagnostics.leave_state(state, request_id)

    @staticmethod
    def _session_status_summary(sessions: Dict[int, Any]) -> Dict[str, Dict[str, Any]]:
        summary: Dict[str, Dict[str, Any]] = {}
        for request_id, session in sessions.items():
            status = getattr(session, "status", None)
            name = getattr(status, "name", str(status))
            status_summary = summary.setdefault(name, {"count": 0, "sample_ids": []})
            status_summary["count"] += 1
            sample_ids = status_summary["sample_ids"]
            if len(sample_ids) < _DIAG_MAX_SAMPLED_REQUESTS:
                sample_ids.append(request_id)
        return dict(sorted(summary.items()))

    def _maybe_log_diagnostics(
        self,
        *,
        force: bool = False,
        teardown: bool = False,
    ) -> None:
        diagnostics = getattr(self, "_diagnostics", None)
        if diagnostics is None:
            return
        if not diagnostics.snapshot_due(force=force):
            return
        coordinator = getattr(self, "_async_consensus", None)
        coordinator_snapshot = None
        if coordinator is not None:
            diagnostic_snapshot = getattr(coordinator, "diagnostic_snapshot", None)
            if callable(diagnostic_snapshot):
                coordinator_snapshot = diagnostic_snapshot()
        state = {
            "wait": len(getattr(self, "_wait_reqs", {})),
            "ready_published": len(getattr(self, "_async_ready_published", {})),
            "ready_prepared": len(getattr(self, "_async_ready_prepared", {})),
            "ready_released": len(getattr(self, "_async_ready_released", set())),
            "ready_activated": len(getattr(self, "_async_ready_activated", {})),
            "terminal_published": len(getattr(self, "_async_terminal_published", {})),
            "terminal_commits": len(getattr(self, "_async_terminal_commits", {})),
            "send_sessions": len(getattr(self, "_send_sessions", {})),
            "send_statuses": self._session_status_summary(getattr(self, "_send_sessions", {})),
            "recv_sessions": len(getattr(self, "_recv_sessions", {})),
            "recv_statuses": self._session_status_summary(getattr(self, "_recv_sessions", {})),
            "gen_need_sync": int(getattr(self, "_gen_need_sync", False)),
            "ctx_need_tp_sync": int(getattr(self, "_ctx_need_tp_sync", False)),
            "ctx_need_pp_sync": int(getattr(self, "_ctx_need_pp_sync", False)),
        }
        diagnostics.maybe_log_snapshot(
            state,
            coordinator_snapshot,
            force=force,
            teardown=teardown,
        )

    def _diagnostic_allgather(
        self,
        label: str,
        allgather: Callable,
        value: Any,
    ) -> Any:
        diagnostics = getattr(self, "_diagnostics", None)
        if diagnostics is None:
            return allgather(value)
        self._maybe_log_diagnostics()
        sequence = diagnostics.collective_enter(label)
        try:
            return allgather(value)
        finally:
            diagnostics.collective_exit(label, sequence)

    def _diagnostic_wait_complete(
        self,
        side: str,
        request_id: int,
        session: TxSessionBase,
        *,
        blocking: bool,
    ) -> Optional[WaitResult]:
        diagnostics = getattr(self, "_diagnostics", None)
        if diagnostics is None:
            return session.wait_complete(blocking=blocking)
        state = f"{side}_future_wait_call"
        self._enter_diag_state(state, request_id)
        self._record_diag_transition(
            f"{side}_future_wait_call_enter",
            request_id,
            once=True,
        )
        started_at = time.monotonic()
        try:
            result = session.wait_complete(blocking=blocking)
        finally:
            self._leave_diag_state(state, request_id)
        duration_s = time.monotonic() - started_at
        status = session.status
        if result is not None or status in (SessionStatus.CANCELLED, SessionStatus.ERROR):
            result_name = "NONE" if result is None else result.name
            self._record_diag_transition(
                f"{side}_future_wait_call_exit",
                request_id,
                outcome=f"result:{result_name},status:{status.name}",
                duration_s=duration_s,
                once=True,
            )
        return result

    def _async_startup_descriptor(self) -> tuple:
        config = self._async_consensus_config
        topology = (
            int(self._mapping.world_size),
            int(self._mapping.tp_size),
            int(self._mapping.pp_size),
            int(self._mapping.cp_size),
            bool(self._mapping.enable_attention_dp),
        )
        distributed_runtime = (
            "MPI"
            if isinstance(self._dist, MPIDist)
            else f"{type(self._dist).__module__}.{type(self._dist).__qualname__}"
        )
        return (
            PROTOCOL_VERSION,
            str(config.backend),
            str(config.transceiver_runtime),
            distributed_runtime,
            topology,
            self._async_terminal_flag_value,
            self._async_peer_ready_flag_value,
        )

    def _complete_async_consensus_startup(self, gathered: list) -> list[str]:
        """Validate opt-in metadata piggybacked on endpoint exchange.

        With both flags off, every contribution remains the exact legacy
        endpoint string. Explicit opt-in intentionally requires a same-version
        worker group; it is not a rolling-upgrade compatibility mechanism.
        """

        def is_tagged(value: Any) -> bool:
            return isinstance(value, tuple) and len(value) == 3 and value[0] == _ASYNC_STARTUP_TAG

        tagged = [is_tagged(value) for value in gathered]
        if not any(tagged):
            return [cast(str, endpoint) for endpoint in gathered]
        if not all(tagged):
            raise RuntimeError(
                "asynchronous Python consensus explicit opt-in requires a "
                "same-version worker group; mixed tagged and legacy endpoint "
                f"contributions were gathered: {gathered}"
            )

        endpoints = [cast(str, value[1]) for value in gathered]
        descriptors = [tuple(value[2]) for value in gathered]
        descriptor = self._async_startup_descriptor()
        if len(descriptors) != self._mapping.world_size or any(
            peer_descriptor != descriptor for peer_descriptor in descriptors
        ):
            raise RuntimeError(
                "asynchronous Python consensus startup descriptor mismatch "
                f"across worker ranks: local={descriptor}, gathered={descriptors}"
            )

        terminal_requested = self._parse_binary_env(
            _ASYNC_TERMINAL_ENV, self._async_terminal_flag_value
        )
        peer_ready_requested = self._parse_binary_env(
            _ASYNC_PEER_READY_ENV, self._async_peer_ready_flag_value
        )
        requested = terminal_requested or peer_ready_requested
        if not requested:
            raise RuntimeError(
                "asynchronous Python consensus startup metadata was tagged without an enabled mode"
            )

        negotiation_domain = (
            isinstance(self._dist, MPIDist)
            and self._mapping.tp_size == 1
            and self._mapping.cp_size == 1
            and not self._mapping.enable_attention_dp
            and self._mapping.pp_size > 1
            and self._mapping.world_size == self._mapping.pp_size
        )
        if not negotiation_domain:
            raise RuntimeError(
                "asynchronous Python transceiver CTX consensus currently requires "
                "the MPI distributed runtime, TP1, CP1, non-ADP, PP>1, and a "
                "PP domain equal to the worker world"
            )

        config = self._async_consensus_config
        qualified = config.backend == "NIXL" and config.transceiver_runtime == "PYTHON"
        if not qualified:
            raise RuntimeError(
                "asynchronous Python transceiver CTX consensus currently requires "
                "backend='NIXL' and transceiver_runtime='PYTHON'"
            )

        self._async_terminal_consensus_enabled = terminal_requested
        self._async_peer_ready_consensus_enabled = peer_ready_requested

        participants = tuple(int(rank) for rank in self._mapping.pp_group)
        # Construction runs on the executor thread after the existing startup
        # collectives. TransferWorker's background threads progress NIXL, not
        # this Distributed MPI communicator; the transport duplicates the
        # communicator before its asynchronous point-to-point traffic starts.
        transport = MpiConsensusTransport(participants)
        try:
            coordinator = AsyncConsensusCoordinator(
                transport,
                scheduling_rank=participants[0],
            )
        except Exception:
            # MpiConsensusTransport owns a duplicated communicator as soon as
            # construction returns.  The coordinator is not yet published to
            # shutdown(), so roll that ownership back transactionally while
            # preserving the original constructor exception.
            try:
                transport.close(_CONSENSUS_STARTUP_CLOSE_TIMEOUT_S)
            except Exception as close_error:
                logger.error(
                    "Python asynchronous-consensus startup rollback failed "
                    f"to close its transport: {close_error}"
                )
            raise
        self._async_consensus = coordinator
        logger.info(
            "PYTHON_ASYNC_CONSENSUS transition=mode_active "
            f"version={PROTOCOL_VERSION} side=ctx rank={self._dist.rank} "
            f"terminal={int(terminal_requested)} "
            f"peer_ready={int(peer_ready_requested)} "
            f"participants={participants}"
        )
        return endpoints

    def _broadcast_instance_name(self) -> str:
        if self._dist.rank == 0:
            name = str(uuid.uuid4())
            self._dist.broadcast(name, 0)
            return name
        return cast(str, self._dist.broadcast(None, 0))

    def _broadcast_context_endpoint(self) -> str:
        if self._dist.rank == 0:
            endpoint = self._transfer_worker.rank_info_server_endpoint or ""
            self._dist.broadcast(endpoint, 0)
            return endpoint
        return cast(str, self._dist.broadcast(None, 0))

    def _init_sync_policy(self):
        m = self._mapping
        self._ctx_need_tp_sync = m.tp_size > 1 and not m.enable_attention_dp
        self._ctx_need_pp_sync = m.pp_size > 1
        self._gen_need_sync = not (m.world_size == 1 or (m.enable_attention_dp and m.pp_size == 1))
        pp_allgather: Callable = getattr(self._dist, "pp_allgather")
        self._gen_allgather: Callable = (
            pp_allgather if m.enable_attention_dp else self._dist.allgather
        )

    def _exchange_rank_info(self):
        endpoint = self._transfer_worker.sender_endpoint
        raw_opt_in = self._mapping.world_size > 1 and (
            self._async_terminal_flag_value != "0" or self._async_peer_ready_flag_value != "0"
        )
        contribution: Any = endpoint
        if raw_opt_in:
            contribution = (
                _ASYNC_STARTUP_TAG,
                endpoint,
                self._async_startup_descriptor(),
            )
        gathered = list(
            self._diagnostic_allgather(
                "startup_rank_info",
                self._dist.allgather,
                contribution,
            )
        )
        layer_num = len(self._kv_cache_manager.pp_layers)
        if isinstance(self._kv_cache_manager, MambaHybridCacheManager):
            layer_num += len(self._kv_cache_manager._impl.mamba_layer_offsets)
        layer_num_per_pp = cast(
            list,
            self._diagnostic_allgather(
                "startup_layer_count",
                getattr(self._dist, "pp_allgather"),
                layer_num,
            ),
        )
        # Validate only after every rank has entered both pre-existing startup
        # exchanges. This lets a mixed old/new worker group with an accidental
        # explicit opt-in fail on both sides instead of leaving the old worker
        # blocked in the layer-count exchange. Flags-off workers still send
        # the exact legacy endpoint value and add no collective.
        endpoints = self._complete_async_consensus_startup(gathered)
        self._transfer_worker.populate_instance_and_rank_info(
            endpoints=endpoints, layer_num_per_pp=layer_num_per_pp
        )
        logger.info(f"transfer worker ctx_server_endpoints: {endpoints}")
        logger.info(f"layer_num_per_pp: {layer_num_per_pp}")
        logger.info(f"self._context_info_endpoint: {self._context_info_endpoint}")

    def shutdown(self):
        if getattr(self, "_shutdown_complete", False):
            return
        worker_event = getattr(self, "_shutdown_worker_event", None)
        if worker_event is not None and not worker_event.is_set():
            return worker_event
        diagnostics = getattr(self, "_diagnostics", None)
        if diagnostics is not None:
            diagnostics.record_transition("teardown_start", 0, once=True)
            self._maybe_log_diagnostics(force=True, teardown=True)
        self._shutdown = True
        shutdown_errors: list[tuple[str, Exception]] = []

        def run_shutdown_step(name: str, callback: Callable[[], None]) -> None:
            try:
                callback()
            except Exception as error:
                logger.error(f"Python transceiver shutdown step {name!r} failed: {error}")
                shutdown_errors.append((name, error))

        if not getattr(self, "_shutdown_sessions_complete", False):
            for rid, session in list(self._send_sessions.items()):
                run_shutdown_step(f"close TxSession {rid}", session.close)
            for rid, session in list(self._recv_sessions.items()):
                run_shutdown_step(f"close RxSession {rid}", session.close)
            self._send_sessions.clear()
            self._send_reqs.clear()
            self._recv_sessions.clear()
            self._recv_reqs.clear()
            getattr(self, "_legacy_failed_sessions", set()).clear()
            # Session close is best effort.  Transfer-worker shutdown is the
            # final drain, so session close must not be retried after the
            # worker may already have stopped.
            self._shutdown_sessions_complete = True

        if not getattr(self, "_shutdown_consensus_complete", False):
            if self._async_consensus is None:
                self._shutdown_consensus_complete = True
            else:
                errors_before_consensus = len(shutdown_errors)

                def shutdown_consensus() -> None:
                    self._async_consensus.shutdown()
                    logger.info(
                        "PYTHON_ASYNC_CONSENSUS transition=shutdown_summary "
                        f"rank={self._dist.rank} "
                        f"counters={dict(self._async_consensus_counters)}"
                    )

                run_shutdown_step("asynchronous consensus", shutdown_consensus)
                if len(shutdown_errors) == errors_before_consensus:
                    self._shutdown_consensus_complete = True

        if not getattr(self, "_shutdown_worker_complete", False):
            try:
                worker_result = self._transfer_worker.shutdown()
            except Exception as error:
                # A non-progress-thread shutdown raises only after native
                # teardown is terminal, so registered memory can now be
                # released even though the error still has to be surfaced.
                logger.error(f"Python transceiver shutdown step 'transfer worker' failed: {error}")
                shutdown_errors.append(("transfer worker", error))
                self._shutdown_worker_complete = True
                self._shutdown_worker_event = None
            else:
                if isinstance(worker_result, threading.Event):
                    # Internal progress threads cannot join themselves.  The
                    # returned event is set when deferred native teardown is
                    # terminal; the owner must wait and call shutdown again
                    # to surface any recorded native error.
                    self._shutdown_worker_event = worker_result
                elif worker_result is None:
                    self._shutdown_worker_complete = True
                    self._shutdown_worker_event = None
                else:
                    raise TypeError(
                        "TransferWorker.shutdown() must return None or threading.Event, "
                        f"got {type(worker_result).__name__}"
                    )

        self._shutdown_complete = (
            getattr(self, "_shutdown_sessions_complete", False)
            and getattr(self, "_shutdown_consensus_complete", False)
            and getattr(self, "_shutdown_worker_complete", False)
        )
        worker_event = getattr(self, "_shutdown_worker_event", None)
        if worker_event is not None and not getattr(self, "_shutdown_worker_complete", False):
            deferred_errors = getattr(self, "_shutdown_deferred_errors", [])
            deferred_errors.extend(shutdown_errors)
            self._shutdown_deferred_errors = deferred_errors
            return worker_event
        shutdown_errors = [
            *getattr(self, "_shutdown_deferred_errors", []),
            *shutdown_errors,
        ]
        self._shutdown_deferred_errors = []
        if shutdown_errors:
            raise shutdown_errors[0][1]

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.shutdown()

    def _create_kv_slice(self, req: LlmRequest) -> KVSlice:
        adapter = self._reuse_adapter
        tpb = adapter.tokens_per_block
        assert self._page_table is not None
        layer_groups = self._page_table.layer_groups

        is_gen_only = req.is_generation_only_request()
        cached_per_lg = (
            adapter.get_cached_token_count_per_layer_group(req, layer_groups)
            if is_gen_only
            else [0] * len(layer_groups)
        )

        token_range = None
        if req.prompt_len > 0:
            # end must match the trimmed block list below (ceil(prompt_len / tpb)
            # blocks). num_extra_kv_tokens slots (speculative decoding) are not
            # transferred. In the previously added support for ctx disabling
            # speculative decoding while gen enables it, both sides currently
            # use prompt_len as the transfer range, so the ranges stay
            # consistent.
            # TODO: the accuracy impact of not transferring num_extra_kv_tokens
            # on MTP and other speculative decoding paths is currently unclear;
            # revisit whether these extra KV slots need to be transferred.
            token_range = TokenRange(start=0, end=req.prompt_len)

        groups = []
        for idx, lg in enumerate(layer_groups):
            if isinstance(lg, MambaLayerGroup):
                groups.append(np.array([], dtype=np.int64))
                continue
            block_ids = adapter.get_block_ids(req, idx, lg)
            # Limit to prompt_len blocks, matching C++ cacheFormatter behavior.
            total_blocks = (req.prompt_len + tpb - 1) // tpb
            if block_ids.size > total_blocks:
                block_ids = block_ids[:total_blocks]
            window_size = lg.sliding_window_size

            if window_size is not None:
                # Drop stale blocks the manager may still expose (V1 pre-eviction).
                stale_end = max(0, (req.prompt_len + 1 - window_size) // tpb)
                expected_valid = max(0, total_blocks - stale_end)
                # Stale prefix already pruned above; skip reuse-hit blocks that
                # land inside the window. Clamp to 0: ctx side has cached_per_lg
                # synthetically 0, and a reuse hit may fall entirely inside the
                # stale region (those blocks were already pruned, no extra skip).
                cache_skip = max(0, cached_per_lg[idx] // tpb - stale_end)
            else:
                total_blocks = (req.prompt_len + tpb - 1) // tpb
                expected_valid = total_blocks
                cache_skip = cached_per_lg[idx] // tpb

            block_ids = self._trim_packed_beam_block_ids(
                block_ids,
                beam_width=req.py_beam_width,
                total_blocks=total_blocks,
                expected_valid=expected_valid,
                cache_skip=cache_skip,
            )

            groups.append(block_ids)

        mamba_state_index = None
        if isinstance(self._kv_cache_manager, MambaHybridCacheManager):
            mamba_state_index = self._kv_cache_manager.mamba_cache_index[req.py_request_id]

        return KVSlice(
            is_last_slice=True,
            block_ids_per_layer_groups=groups,
            mamba_state_index=mamba_state_index,
            token_range=token_range,
        )

    def _slice_num_bytes(self, slice: KVSlice) -> int:
        """Local-rank KV bytes covered by a slice (sum of num_valid_blocks * pool.slot_bytes), enough to populate
        kv_cache_size and unblock the perf-metric timestamps that gate on it."""
        pt = self._page_table
        if pt is None:
            return 0
        total = 0
        for lg_id, block_ids in enumerate(slice.block_ids_per_layer_groups):
            if block_ids is None or block_ids.size == 0:
                continue
            lg = pt.layer_groups[lg_id]
            if isinstance(lg, MambaLayerGroup):
                continue
            n = int((block_ids >= 0).sum())
            if n == 0:
                continue
            for pv in lg.pool_views:
                pool = get_physical_pool(pt, lg_id, pv.pool_idx)
                total += n * pool.slot_bytes
        return total

    @staticmethod
    def _split_packed_beam_block_ids(
        block_ids: np.ndarray,
        beam_width: int,
        total_blocks: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Split 1-D block IDs into beam-0 prefix and appended beam-tail blocks."""
        if beam_width <= 1 or block_ids.size <= total_blocks:
            return block_ids, np.array([], dtype=np.int64)
        tail_count = min(beam_width - 1, block_ids.size - total_blocks)
        if tail_count <= 0:
            return block_ids, np.array([], dtype=np.int64)
        return block_ids[:-tail_count], block_ids[-tail_count:]

    @staticmethod
    def _trim_packed_beam_block_ids(
        block_ids: np.ndarray,
        beam_width: int,
        total_blocks: int,
        expected_valid: int,
        cache_skip: int,
    ) -> np.ndarray:
        """Trim/skip beam-0 blocks while preserving packed beam-tail blocks."""
        if expected_valid <= 0:
            return np.array([], dtype=np.int64)

        beam0_block_ids, tail_block_ids = KvCacheTransceiverV2._split_packed_beam_block_ids(
            block_ids, beam_width, total_blocks
        )

        if beam0_block_ids.size > expected_valid:
            beam0_block_ids = (
                beam0_block_ids[-expected_valid:]
                if expected_valid > 0
                else np.array([], dtype=np.int64)
            )
            if beam0_block_ids.size == 0:
                tail_block_ids = np.array([], dtype=np.int64)

        if cache_skip > 0:
            if cache_skip < beam0_block_ids.size:
                beam0_block_ids = beam0_block_ids[cache_skip:]
            else:
                beam0_block_ids = np.array([], dtype=np.int64)
                tail_block_ids = np.array([], dtype=np.int64)

        if tail_block_ids.size == 0:
            return beam0_block_ids
        return np.concatenate([beam0_block_ids, tail_block_ids])

    @staticmethod
    def _need_aux_transfer(req: LlmRequest) -> bool:
        params = req.py_disaggregated_params
        return params is not None and params.schedule_style == DisaggScheduleStyle.GENERATION_FIRST

    def _ctx_consensus(self, local_ids: list) -> list:
        # TP consensus: ensure all TP ranks have peer info
        sync_size = self._dist.tp_size if self._ctx_need_tp_sync else 1
        all_ranks = (
            self._diagnostic_allgather(
                "ctx_ready_tp",
                self._dist.tp_allgather,
                local_ids,
            )
            if self._ctx_need_tp_sync
            else [local_ids]
        )
        ready_ids = _find_consensus_request_ids(all_ranks, sync_size)

        # PP consensus: ensure all PP ranks have peer info before promoting.
        # In PP, the first PP rank schedules and propagates to others. If a
        # request is promoted on the first rank but peer info hasn't arrived
        # on other ranks, respond_and_send_async on those ranks would fail
        # to dispatch the KV transfer (gen-first skips listener dispatch).
        # TODO: This is a workaround for functionality: pp_allgather impacts
        # the pp loop performance. One possible solution is to let pp rank0
        # decide the ready request ids, the other pp ranks treat the unready
        # request as ctx-first requests.
        if self._ctx_need_pp_sync:
            pp_all_ranks = self._diagnostic_allgather(
                "ctx_ready_pp",
                getattr(self._dist, "pp_allgather"),
                ready_ids,
            )
            ready_ids = _find_consensus_request_ids(pp_all_ranks, self._mapping.pp_size)

        return ready_ids

    def _gen_consensus(self, local_ids: list) -> list:
        sync_size = (
            self._mapping.pp_size if self._mapping.enable_attention_dp else self._mapping.world_size
        )
        all_ranks = (
            self._diagnostic_allgather(
                "gen_ready_ids",
                self._gen_allgather,
                local_ids,
            )
            if self._gen_need_sync
            else [local_ids]
        )
        return _find_consensus_request_ids(all_ranks, sync_size)

    @staticmethod
    def _union(all_lists: List[List[int]]) -> set:
        merged: set = set()
        for ids in all_lists:
            merged.update(ids)
        return merged

    @staticmethod
    def _intersection(all_lists: List[List[int]], n_ranks: int) -> set:
        if n_ranks == 0:
            return set()
        cnt: Dict[int, int] = defaultdict(int)
        for ids in all_lists:
            for rid in set(ids):
                cnt[rid] += 1
        return {rid for rid, c in cnt.items() if c == n_ranks}

    def _consensus_outcome(
        self,
        to_process,
        known_ids,
        cancelled,
        cancel_quiescent,
        failed,
        failed_quiescent,
        completed,
        allgather: Callable,
        need_sync: bool,
        collective_label: str = "outcome",
    ):
        # A cancellation decision is global when ANY rank observes it, but its
        # resources are reclaimable only when ALL ranks report local
        # quiescence. Failure uses the same decision/acknowledgement split;
        # ERROR alone does not prove that sibling native operations drained.
        # COMPLETED is global only when every rank agrees. Keep all five lists
        # in one packed allgather so the acknowledgements add no rendezvous.
        if not need_sync:
            all_c = [list(cancelled)]
            all_cq = [list(cancel_quiescent)]
            all_f = [list(failed)]
            all_fq = [list(failed_quiescent)]
            all_done = [list(completed)]
        else:
            packed = list(
                self._diagnostic_allgather(
                    collective_label,
                    allgather,
                    [
                        list(cancelled),
                        list(cancel_quiescent),
                        list(failed),
                        list(failed_quiescent),
                        list(completed),
                    ],
                )
            )
            all_c = [p[0] for p in packed]
            all_cq = [p[1] for p in packed]
            all_f = [p[2] for p in packed]
            all_fq = [p[3] for p in packed]
            all_done = [p[4] for p in packed]
        n = len(all_c)
        global_cancelled = self._union(all_c)
        global_cancel_quiescent = self._intersection(all_cq, n)
        global_failed = self._union(all_f)
        global_failed_quiescent = self._intersection(all_fq, n)
        global_completed = self._intersection(all_done, n)
        new_cancelled = [rid for rid in known_ids if rid in global_cancelled]
        reclaimable_cancelled = [rid for rid in new_cancelled if rid in global_cancel_quiescent]
        cancel_set = set(new_cancelled)
        new_failed = [rid for rid in known_ids if rid in global_failed and rid not in cancel_set]
        reclaimable_failed = [rid for rid in new_failed if rid in global_failed_quiescent]
        terminal = cancel_set | set(new_failed)
        new_completed = [
            rid for rid in to_process if rid in global_completed and rid not in terminal
        ]
        return (
            new_cancelled,
            reclaimable_cancelled,
            new_failed,
            reclaimable_failed,
            new_completed,
        )

    def _gen_consensus_outcome(
        self,
        to_process,
        known_ids,
        cancelled,
        cancel_quiescent,
        failed,
        failed_quiescent,
        completed,
    ):
        return self._consensus_outcome(
            to_process,
            known_ids,
            cancelled,
            cancel_quiescent,
            failed,
            failed_quiescent,
            completed,
            self._gen_allgather,
            self._gen_need_sync,
            "gen_outcome",
        )

    def _ctx_consensus_outcome(
        self,
        to_process,
        known_ids,
        cancelled,
        cancel_quiescent,
        failed,
        failed_quiescent,
        completed,
        timed_out,
    ):
        # TP first, then PP.  timed_out is local-only (back-off signal).
        c, cq, f, fq, d = self._consensus_outcome(
            to_process,
            known_ids,
            cancelled,
            cancel_quiescent,
            failed,
            failed_quiescent,
            completed,
            self._dist.tp_allgather,
            self._ctx_need_tp_sync,
            "ctx_outcome_tp",
        )
        if self._ctx_need_pp_sync:
            pp_allgather: Callable = getattr(self._dist, "pp_allgather")
            c, cq, f, fq, d = self._consensus_outcome(
                to_process,
                known_ids,
                c,
                cq,
                f,
                fq,
                d,
                pp_allgather,
                True,
                "ctx_outcome_pp",
            )
        return c, cq, f, fq, d, timed_out

    def _record_async_transition(
        self,
        transition: str,
        request_id: int,
        epoch: int,
        outcome: Optional[ConsensusOutcome] = None,
    ) -> None:
        self._record_diag_transition(
            transition,
            request_id,
            epoch,
            None if outcome is None else outcome.name,
        )
        self._async_consensus_counters[transition] += 1
        count = self._async_consensus_counters[transition]
        if count != 1 and count % 512 != 0:
            return
        outcome_text = "" if outcome is None else f" outcome={outcome.name}"
        logger.info(
            "PYTHON_ASYNC_CONSENSUS "
            f"transition={transition} count={count} "
            f"rank={self._dist.rank} request_id={request_id} epoch={epoch}"
            f"{outcome_text}"
        )

    @staticmethod
    def _retire_async_epoch(
        epochs: OrderedDict[int, int], request_id: int, next_epoch: int
    ) -> None:
        """Retain a bounded replay window for recently retired request IDs."""
        epochs[request_id] = max(next_epoch, epochs.get(request_id, 0))
        epochs.move_to_end(request_id)
        while len(epochs) > _MAX_RETIRED_CONSENSUS_REQUESTS:
            epochs.popitem(last=False)

    def _progress_async_consensus(self) -> None:
        coordinator = self._async_consensus
        if coordinator is None:
            return
        diagnostics = getattr(self, "_diagnostics", None)
        if diagnostics is not None:
            diagnostics.record_poll()
        for event in coordinator.poll():
            if event.kind == ConsensusEventKind.TERMINAL_COMMIT:
                published_epoch = self._async_terminal_published.get(event.request_id)
                if published_epoch != event.epoch:
                    raise RuntimeError(
                        "received an unexpected asynchronous terminal commit: "
                        f"request_id={event.request_id}, epoch={event.epoch}, "
                        f"published_epoch={published_epoch}"
                    )
                self._async_terminal_commits[event.request_id] = event
                self._record_async_transition(
                    "terminal_commit",
                    event.request_id,
                    event.epoch,
                    event.outcome,
                )
            elif event.kind == ConsensusEventKind.READY_PREPARE:
                self._apply_async_ready_prepare(event)
            elif event.kind == ConsensusEventKind.READY_RELEASE:
                self._apply_async_ready_release(event)
            elif event.kind == ConsensusEventKind.READY_COMPLETE:
                self._apply_async_ready_complete(event)
            elif event.kind == ConsensusEventKind.READY_ABORT:
                self._apply_async_ready_abort(event)
            elif event.kind == ConsensusEventKind.READY_ABORT_FINALIZE:
                self._apply_async_ready_abort_finalize(event)
            else:
                raise RuntimeError(f"unhandled asynchronous consensus event: {event}")
        self._maybe_log_diagnostics()

    def _apply_async_ready_prepare(self, event: ConsensusEvent) -> None:
        coordinator = cast(AsyncConsensusCoordinator, self._async_consensus)
        key = (event.request_id, event.epoch)
        if self._async_ready_published.get(event.request_id) != event.epoch:
            raise RuntimeError(f"READY_PREPARE has no matching local publication: {event}")
        req = self._wait_reqs.pop(event.request_id, None)
        if req is None:
            raise RuntimeError(f"READY_PREPARE has no waiting request: {event}")
        # PREPARE is a hidden, abortable lease.  Do not expose the request to
        # an ordinary scheduler yet: another rank may still withdraw, and a
        # state-only rollback cannot undo KV/scheduler mutations.  Rank zero
        # becomes visible at READY_RELEASE; followers become visible only when
        # the exact rank-zero PP schedule contains this request.
        self._async_ready_prepared[key] = req
        coordinator.acknowledge_ready(event.request_id, event.epoch)
        self._async_ready_acknowledged.add(key)
        self._leave_diag_state("ctx_wait", event.request_id)
        self._enter_diag_state("ready_prepared", event.request_id)
        self._record_async_transition("ready_prepare", event.request_id, event.epoch)

    def _apply_async_ready_release(self, event: ConsensusEvent) -> None:
        if self._async_ready_published.get(event.request_id) != event.epoch:
            raise RuntimeError(f"READY_RELEASE has no matching local publication: {event}")
        key = (event.request_id, event.epoch)
        req = self._async_ready_prepared.get(key)
        if req is None:
            raise RuntimeError(f"READY_RELEASE has no prepared request: {event}")
        req.state = LlmRequestState.CONTEXT_INIT
        self._async_ready_released.add(key)
        self._leave_diag_state("ready_prepared", event.request_id)
        self._enter_diag_state("ready_released", event.request_id)
        self._record_async_transition("ready_release", event.request_id, event.epoch)

    def _apply_async_ready_complete(self, event: ConsensusEvent) -> None:
        key = (event.request_id, event.epoch)
        req = self._async_ready_activated.pop(key, None)
        if req is None:
            raise RuntimeError(f"READY_COMPLETE has no activated request: {event}")
        self._async_ready_released.discard(key)
        self._finish_async_ready_epoch(event.request_id, event.epoch)
        self._leave_diag_state("ready_activated", event.request_id)
        self._record_async_transition("ready_complete", event.request_id, event.epoch)

    def _apply_async_ready_abort(self, event: ConsensusEvent) -> None:
        coordinator = cast(AsyncConsensusCoordinator, self._async_consensus)
        key = (event.request_id, event.epoch)
        if key in self._async_ready_aborted:
            coordinator.acknowledge_ready_abort(event.request_id, event.epoch)
            return
        req = self._async_ready_prepared.pop(key, None)
        if req is None:
            req = self._wait_reqs.pop(event.request_id, None)
        if req is not None:
            req.state = LlmRequestState.DISAGG_CONTEXT_WAIT_SCHEDULER
            setattr(req, _ASYNC_READY_CANCELLED_EPOCH_ATTR, event.epoch)
        self._async_ready_aborted[key] = req
        coordinator.acknowledge_ready_abort(event.request_id, event.epoch)
        self._leave_diag_state("ctx_wait", event.request_id)
        self._leave_diag_state("ready_prepared", event.request_id)
        self._leave_diag_state("ready_released", event.request_id)
        self._enter_diag_state("ready_aborted", event.request_id)
        self._record_async_transition("ready_abort", event.request_id, event.epoch)

    def _apply_async_ready_abort_finalize(self, event: ConsensusEvent) -> None:
        key = (event.request_id, event.epoch)
        if key not in self._async_ready_aborted:
            raise RuntimeError(f"READY_ABORT_FINALIZE has no aborted request: {event}")
        req = self._async_ready_aborted.pop(key)
        if req is None:
            # The protocol round is complete, but a request already in flight
            # through the executor may reach this rank after the final event.
            # Retain a bounded integration tombstone so that first late
            # materialization is excluded rather than being published as the
            # next readiness epoch.
            self._async_ready_finalized_without_request[event.request_id] = event.epoch
            self._async_ready_finalized_without_request.move_to_end(event.request_id)
            while (
                len(self._async_ready_finalized_without_request) > _MAX_RETIRED_CONSENSUS_REQUESTS
            ):
                self._async_ready_finalized_without_request.popitem(last=False)
        self._finish_async_ready_epoch(event.request_id, event.epoch)
        self._leave_diag_state("ready_aborted", event.request_id)
        self._record_async_transition("ready_abort_finalize", event.request_id, event.epoch)

    def _bind_async_ready_abort(self, req: LlmRequest) -> bool:
        """Bind a request that materialized after its readiness abort.

        READY_ABORT acknowledgement is a protocol action and cannot wait for
        executor timing.  Conversely, a late local request must not be treated
        as a fresh readiness vote.  Bind it to an active request-less abort or
        consume the bounded post-finalize tombstone and leave it in the
        cancellation wait state.
        """
        request_id = get_unique_rid(req)
        active_key = next(
            (
                key
                for key, aborted_req in self._async_ready_aborted.items()
                if key[0] == request_id and aborted_req is None
            ),
            None,
        )
        if active_key is not None:
            epoch = active_key[1]
            self._async_ready_aborted[active_key] = req
        else:
            epoch = self._async_ready_finalized_without_request.pop(request_id, None)
            if epoch is None:
                return False
        req.state = LlmRequestState.DISAGG_CONTEXT_WAIT_SCHEDULER
        setattr(req, _ASYNC_READY_CANCELLED_EPOCH_ATTR, epoch)
        return True

    def _finish_async_ready_epoch(self, request_id: int, epoch: int) -> None:
        if self._async_ready_published.get(request_id) == epoch:
            del self._async_ready_published[request_id]
        key = (request_id, epoch)
        self._async_ready_acknowledged.discard(key)
        self._async_ready_withdrawn.discard(key)
        self._retire_async_epoch(self._async_ready_epoch, request_id, epoch + 1)

    @staticmethod
    def _terminal_outcome_from_status(
        status: SessionStatus,
    ) -> Optional[ConsensusOutcome]:
        if status == SessionStatus.CANCELLED:
            return ConsensusOutcome.CANCELLED
        if status == SessionStatus.ERROR:
            return ConsensusOutcome.FAILED
        if status in (
            SessionStatus.KV_TRANSFERRED,
            SessionStatus.FULLY_TRANSFERRED,
        ):
            return ConsensusOutcome.COMPLETED
        return None

    def _seal_and_snapshot_terminal(
        self,
        session: TxSessionBase,
        result: Optional[WaitResult],
    ) -> Optional[ConsensusOutcome]:
        """Seal ``session`` and return its immutable local terminal vote.

        New native sessions expose an atomic terminal snapshot. Keep the
        fallback until every transfer backend implements that API; it re-reads
        status only after the old seal boundary so cancellation cannot replace
        a previously observed success between observation and publication.
        """
        snapshot_terminal = getattr(session, "seal_and_snapshot_terminal", None)
        if snapshot_terminal is not None:
            status = snapshot_terminal()
            if status is None:
                return None
            return self._terminal_outcome_from_status(status)

        candidate = self._terminal_outcome_from_status(session.status)
        if candidate is None:
            if result == WaitResult.FAILED or session.has_failed():
                candidate = ConsensusOutcome.FAILED
            elif result == WaitResult.COMPLETED or session.is_completed():
                candidate = ConsensusOutcome.COMPLETED
        if candidate is None:
            return None
        if not session.seal_and_check_quiescent():
            return None
        outcome = self._terminal_outcome_from_status(session.status)
        if outcome is not None:
            return outcome
        return candidate

    def _publish_async_ctx_terminal_votes(
        self,
        block_all: bool,
        request_ids: Optional[set[int]] = None,
    ) -> list[int]:
        coordinator = cast(AsyncConsensusCoordinator, self._async_consensus)
        timed_out: list[int] = []
        for rid, session in list(self._send_sessions.items()):
            if request_ids is not None and rid not in request_ids:
                continue
            if rid in self._async_terminal_published:
                continue
            if session.has_transferring_tasks():
                continue

            result = self._diagnostic_wait_complete(
                "ctx",
                rid,
                session,
                blocking=block_all,
            )
            outcome = self._seal_and_snapshot_terminal(session, result)
            if outcome is None:
                if result == WaitResult.TIMEOUT:
                    timed_out.append(rid)
                    logger.warning(
                        f"TxSession rid={session.disagg_request_id} timed out after "
                        f"{self._sender_future_timeout_ms}ms"
                    )
                continue
            epoch = self._async_terminal_epoch.get(rid, 0)
            coordinator.publish_terminal(rid, outcome, epoch)
            self._async_terminal_published[rid] = epoch
            self._record_async_transition("terminal_vote", rid, epoch, outcome)
        return timed_out

    def _apply_async_ctx_terminal_commits(
        self,
        *,
        mark_complete: bool,
        only_request_id: Optional[int] = None,
        request_ids: Optional[set[int]] = None,
    ) -> tuple[list[int], list[int]]:
        completed: list[int] = []
        failed: list[int] = []
        for rid, event in list(self._async_terminal_commits.items()):
            if only_request_id is not None and rid != only_request_id:
                continue
            if request_ids is not None and rid not in request_ids:
                continue
            session = self._send_sessions.get(rid)
            req = self._send_reqs.get(rid)
            if session is None or req is None:
                raise RuntimeError(
                    f"terminal commit arrived after local resources were released: rid={rid}"
                )
            if session.has_transferring_tasks():
                raise RuntimeError(f"terminal commit arrived before local quiescence: rid={rid}")

            if event.outcome == ConsensusOutcome.COMPLETED:
                if mark_complete:
                    req.state = LlmRequestState.DISAGG_CONTEXT_COMPLETE
                completed.append(rid)
            elif event.outcome == ConsensusOutcome.FAILED:
                req.state = LlmRequestState.DISAGG_TRANS_ERROR
                failed.append(rid)
            elif event.outcome == ConsensusOutcome.CANCELLED:
                # The ordinary status poll can observe the authoritative
                # cancellation before the executor retries its cancellation
                # request. Preserve a request-scoped tombstone so the
                # AsyncTransferManager can acknowledge and retire exactly its
                # transceiver leg instead of losing ownership permanently.
                self._async_terminal_cancelled[rid] = (event.epoch, req)
                self._record_context_cancelled_request_id(rid)
            else:
                raise RuntimeError(f"invalid terminal commit outcome: {event}")

            self._leave_diag_state("ctx_send", rid)
            self._record_diag_transition(
                "ctx_global_terminal",
                rid,
                event.epoch,
                event.outcome.name,
                once=True,
            )
            session.close()
            self._record_diag_transition("ctx_session_closed", rid, event.epoch, once=True)
            del self._send_reqs[rid]
            del self._send_sessions[rid]
            del self._async_terminal_commits[rid]
            self._async_terminal_published.pop(rid, None)
            if event.outcome != ConsensusOutcome.CANCELLED:
                self._retire_async_epoch(self._async_terminal_epoch, rid, event.epoch + 1)
        return completed, failed

    def _record_context_cancelled_request_id(self, request_id: int) -> None:
        request_ids = getattr(self, "_context_cancelled_request_ids", None)
        if request_ids is None:
            request_ids = []
            self._context_cancelled_request_ids = request_ids
        request_ids.append(request_id)

    def take_context_cancelled_request_ids(self) -> List[int]:
        """Transfer newly quiescent CTX cancellation IDs to the executor."""
        pending = getattr(self, "_context_cancelled_request_ids", None)
        if pending is None:
            return []
        request_ids = list(pending)
        pending.clear()
        return request_ids

    def _acknowledge_async_terminal_cancellation(self, request_id: int, req: LlmRequest) -> bool:
        cancelled_by_request = getattr(self, "_async_terminal_cancelled", None)
        if cancelled_by_request is None:
            return False
        cancelled = cancelled_by_request.get(request_id)
        if cancelled is None:
            return False
        epoch, owned_req = cancelled
        if owned_req is not req:
            raise RuntimeError(
                "request ID was reused before an asynchronous terminal "
                f"cancellation was acknowledged: request_id={request_id}, epoch={epoch}"
            )
        del cancelled_by_request[request_id]
        # A local user-cancellation retry can consume the commit before the
        # executor's ordinary status poll drains this handoff queue.  Remove
        # that now-local acknowledgement so it cannot later be misclassified
        # as a peer cancellation after the request has already terminated.
        pending = getattr(self, "_context_cancelled_request_ids", None)
        if pending:
            pending[:] = [pending_id for pending_id in pending if pending_id != request_id]
        self._retire_async_epoch(self._async_terminal_epoch, request_id, epoch + 1)
        self._record_async_transition("terminal_cancel_ack", request_id, epoch)
        return True

    def _check_context_transfer_status_async(
        self, at_least_request_num: Optional[int], mark_complete: bool
    ) -> tuple[list[int], list[int]]:
        self._maybe_log_diagnostics()
        snapshot_ids = set(self._send_sessions)
        drain_timeout_ms: Optional[int] = None
        if at_least_request_num is None:
            target_commits = len(snapshot_ids)
            drain_timeout_ms = (
                self._sender_future_timeout_ms or getattr(self, "kv_transfer_timeout_ms", 0) or 0
            )
            timeout_s = max(0, drain_timeout_ms) / 1000.0
        else:
            target_commits = min(max(0, at_least_request_num), len(snapshot_ids))
            timeout_s = max(0, self.kv_transfer_poll_interval_ms or 0) / 1000.0
        deadline = time.monotonic() + timeout_s
        completed: list[int] = []
        failed: list[int] = []
        committed = 0

        while True:
            self._progress_async_consensus()
            # Poll non-blockingly and use a single deadline for the fixed
            # entry snapshot. This preserves the configured timeout without
            # multiplying it by the number of sessions.
            self._publish_async_ctx_terminal_votes(
                block_all=False,
                request_ids=snapshot_ids,
            )
            self._progress_async_consensus()
            committed_ids = snapshot_ids.intersection(self._async_terminal_commits)
            new_completed, new_failed = self._apply_async_ctx_terminal_commits(
                mark_complete=mark_complete,
                request_ids=snapshot_ids,
            )
            completed.extend(new_completed)
            failed.extend(new_failed)
            committed += len(committed_ids)

            if at_least_request_num == 0 or committed >= target_commits:
                break
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                break
            time.sleep(min(0.001, remaining_s))

        if at_least_request_num is None:
            remaining_ids = snapshot_ids.intersection(self._send_sessions)
            if remaining_ids:
                logger.warning(
                    "Timed out draining asynchronous context transfer snapshot "
                    f"after {drain_timeout_ms}ms: "
                    f"remaining_rids={sorted(remaining_ids)}"
                )
        self._transfer_worker.sweep_stale_req_infos()
        self._maybe_log_diagnostics()
        return completed, failed

    def _collect_done(self, sessions: dict, reqs: dict):
        """Scan sessions and return (completed_rids, failed_rids)."""
        completed, failed = [], []
        for rid, session in sessions.items():
            if session.is_completed():
                completed.append(rid)
            elif session.has_failed():
                failed.append(rid)
        return completed, failed

    def _build_to_process(
        self, sessions: dict, consensus: list, wait_num: int, block_all: bool
    ) -> list:
        if block_all:
            return list(sessions.keys())
        to_process = list(consensus)
        for rid in sessions:
            if len(to_process) >= wait_num:
                break
            if rid not in to_process:
                to_process.append(rid)
        return to_process

    def _close_failed_sessions(self, sessions: dict, reqs: dict, failed: list):
        for rid in failed:
            reqs[rid].state = LlmRequestState.DISAGG_TRANS_ERROR
            sessions[rid].close()
            del reqs[rid]
            del sessions[rid]

    def _apply_aux(self, session, req: LlmRequest):
        """Unpack aux tokens from session into request's context_phase_params."""
        session.unpack_aux(req)
        first_gen_tokens = req.py_first_gen_tokens  # type: ignore[attr-defined]
        draft_tokens = req.py_draft_tokens
        if req.context_phase_params is None:
            assert req.py_request_id is not None
            req.context_phase_params = ContextPhaseParams(
                first_gen_tokens=first_gen_tokens,
                req_id=req.py_request_id,
                opaque_state=b"",
                draft_tokens=draft_tokens,
                ctx_dp_rank=0,
                disagg_info_endpoint="",
            )
        else:
            req.context_phase_params.first_gen_tokens = first_gen_tokens
            req.context_phase_params.draft_tokens = draft_tokens

    def _get_or_create_send_session(self, req: LlmRequest) -> TxSessionBase:
        rid = get_unique_rid(req)
        assert rid is not None
        if rid in getattr(self, "_async_terminal_cancelled", {}):
            raise RuntimeError(
                "cannot reuse a request ID before its asynchronous terminal "
                f"cancellation is acknowledged: request_id={rid}"
            )
        if rid not in self._send_sessions:
            self._enter_diag_state("ctx_session_create", rid)
            self._record_diag_transition("ctx_send_session_create_enter", rid, once=True)
            try:
                self._send_sessions[rid] = self._transfer_worker.create_tx_session(req)
            finally:
                self._leave_diag_state("ctx_session_create", rid)
            self._enter_diag_state("ctx_send", rid)
            self._record_diag_transition("ctx_send_session_create", rid, once=True)
        # Publish request ownership before any native dispatch. A pre-cancelled
        # session, or a cancellation racing send/send_aux, must never leave a
        # session without its matching request bookkeeping.
        self._send_reqs[rid] = req
        return self._send_sessions[rid]

    def _finalize_send(self, req: LlmRequest, session: TxSessionBase):
        """Pack aux and set context phase params. Call after the last slice."""
        rid = get_unique_rid(req)
        assert rid is not None
        if self._need_aux_transfer(req):
            self._record_diag_transition("ctx_aux_pack_enter", rid, once=True)
            session.pack_aux(req)
            self._record_diag_transition("ctx_aux_packed", rid, once=True)
            self._record_diag_transition("ctx_aux_send_enter", rid, once=True)
            session.send_aux()
            self._record_diag_transition("ctx_aux_send_dispatched", rid, once=True)
        req.context_phase_params = ContextPhaseParams(
            first_gen_tokens=[],
            req_id=rid,
            opaque_state=None,
            draft_tokens=None,
            ctx_dp_rank=self._dp_rank,
            disagg_info_endpoint=self._context_info_endpoint,
        )

    @nvtx_range("KvCacheTransceiverV2.respond_and_send_async")
    def respond_and_send_async(self, req: LlmRequest):
        self._ever_had_send_session = True
        diag_rid = get_unique_rid(req) if getattr(self, "_diagnostics", None) is not None else None
        session = self._get_or_create_send_session(req)
        req.state = LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS
        if session.status == SessionStatus.CANCELLED:
            return
        kv_slice = self._create_kv_slice(req)
        try:
            session.send(kv_slice)
            if diag_rid is not None:
                self._record_diag_transition("ctx_kv_send_dispatched", diag_rid, once=True)
            if session.status == SessionStatus.CANCELLED:
                return
            self._finalize_send(req, session)
            if diag_rid is not None:
                self._record_diag_transition("ctx_send_dispatched", diag_rid, once=True)
        except RuntimeError:
            # Native pre-cancellation seals the session. Retain the paired
            # maps so the ordinary cancellation/status path owns cleanup.
            if session.status == SessionStatus.CANCELLED:
                return
            raise

    @nvtx_range("KvCacheTransceiverV2.request_and_receive_sync")
    def request_and_receive_sync(self, req: LlmRequest):
        rid = get_unique_rid(req)
        if rid in self._recv_sessions:
            logger.warning(
                f"request_and_receive_sync: rid={rid} already has a recv session, skipping"
            )
            return
        req.state = LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
        session = None
        try:
            session = self._transfer_worker.create_rx_session(req)
            self._recv_sessions[rid] = session
            self._recv_reqs[rid] = req
            kv_slice = self._create_kv_slice(req)
            session.receive(kv_slice)
            result = session.wait_complete(blocking=True)

            if result == WaitResult.COMPLETED:
                # KV-transfer timing setters deferred to #15871 (clock-source consistency); size only.
                req.set_kv_cache_size(self._slice_num_bytes(kv_slice) * self._kv_size_rank_factor)
                if self._need_aux_transfer(req):
                    self._apply_aux(session, req)
                self._assert_disagg_history_declared(req)
                req.state = LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE
            else:
                req.state = LlmRequestState.DISAGG_TRANS_ERROR
        except Exception:
            req.state = LlmRequestState.DISAGG_TRANS_ERROR
            raise
        finally:
            if session is not None:
                session.close()
            self._recv_sessions.pop(rid, None)
            self._recv_reqs.pop(rid, None)

    @nvtx_range("KvCacheTransceiverV2.request_and_receive_async")
    def request_and_receive_async(self, req: LlmRequest):
        self._ever_had_recv_session = True
        rid = get_unique_rid(req)
        if rid in self._recv_sessions:
            logger.warning(
                f"request_and_receive_async: rid={rid} already has a recv session, skipping"
            )
            return
        req.state = LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
        self._enter_diag_state("gen_session_create", rid)
        self._record_diag_transition("gen_receive_session_create_enter", rid, once=True)
        try:
            session = self._transfer_worker.create_rx_session(req)
        finally:
            self._leave_diag_state("gen_session_create", rid)
        self._recv_sessions[rid] = session
        self._recv_reqs[rid] = req
        self._enter_diag_state("gen_recv", rid)
        self._record_diag_transition("gen_receive_session_create", rid, once=True)
        if session.status == SessionStatus.CANCELLED:
            return
        kv_slice = self._create_kv_slice(req)
        req.py_kv_cache_xfer_bytes = self._slice_num_bytes(kv_slice) * self._kv_size_rank_factor
        self._enter_diag_state("gen_receive_dispatch", rid)
        self._record_diag_transition("gen_receive_dispatch_enter", rid, once=True)
        try:
            session.receive(kv_slice)
            self._record_diag_transition("gen_receive_dispatched", rid, once=True)
        except RuntimeError:
            if session.status == SessionStatus.CANCELLED:
                return
            raise
        finally:
            self._leave_diag_state("gen_receive_dispatch", rid)

    def check_context_transfer_status(
        self, at_least_request_num: Optional[int], mark_complete: bool = False
    ):
        self._maybe_log_diagnostics()
        if getattr(self, "_async_terminal_consensus_enabled", False):
            return self._check_context_transfer_status_async(at_least_request_num, mark_complete)
        if getattr(self, "_async_consensus", None) is not None:
            self._progress_async_consensus()
        # A worker that never sends KV has nothing to reconcile here, so skip the consensus. Safe
        # because the flag flips together on every rank and never resets, so they all skip in step;
        # gating on the live session dict instead would not be, since a cancel clears it per-rank.
        # Keep the original sweep (only when tp/pp sync is on) so nothing is leaked.
        if not self._ever_had_send_session:
            if self._ctx_need_tp_sync or self._ctx_need_pp_sync:
                self._transfer_worker.sweep_stale_req_infos()
            return [], []
        block_all = at_least_request_num is None
        wait_num = at_least_request_num if not block_all else 0

        local_completed, local_failed = self._collect_done(self._send_sessions, self._send_reqs)
        to_process = self._build_to_process(
            self._send_sessions,
            self._ctx_consensus(local_completed + local_failed),
            wait_num,
            block_all,
        )

        known_ids = list(self._send_sessions)
        completed, timed_out, failed, failed_quiescent = [], [], [], []
        cancelled, cancel_quiescent = [], []
        for rid, session in self._send_sessions.items():
            if session.status == SessionStatus.CANCELLED:
                cancelled.append(rid)
                if not session.has_transferring_tasks():
                    cancel_quiescent.append(rid)
            elif rid in self._legacy_failed_sessions or session.has_failed():
                failed.append(rid)
                if session.seal_and_check_quiescent():
                    failed_quiescent.append(rid)
        for rid in to_process:
            session = self._send_sessions[rid]
            result = self._diagnostic_wait_complete(
                "ctx",
                rid,
                session,
                blocking=block_all,
            )
            if session.status == SessionStatus.CANCELLED:
                if rid in failed:
                    failed.remove(rid)
                if rid in failed_quiescent:
                    failed_quiescent.remove(rid)
                if rid not in cancelled:
                    cancelled.append(rid)
                if rid not in cancel_quiescent and not session.has_transferring_tasks():
                    cancel_quiescent.append(rid)
            elif result == WaitResult.COMPLETED:
                completed.append(rid)
            elif result is None:
                continue
            elif result == WaitResult.TIMEOUT:
                logger.warning(
                    f"TxSession rid={session.disagg_request_id} timed out after {self._sender_future_timeout_ms}ms"
                )
                timed_out.append(rid)
            else:
                logger.warning(f"TxSession rid={session.disagg_request_id} failed")
                if rid not in failed:
                    failed.append(rid)
                if rid not in failed_quiescent and session.seal_and_check_quiescent():
                    failed_quiescent.append(rid)

        # All ranks must agree on per-rid outcome to avoid req.state divergence.
        cancelled, reclaimable_cancelled, failed, reclaimable_failed, completed, timed_out = (
            self._ctx_consensus_outcome(
                to_process,
                known_ids,
                cancelled,
                cancel_quiescent,
                failed,
                failed_quiescent,
                completed,
                timed_out,
            )
        )

        for rid in cancelled:
            self._legacy_failed_sessions.discard(rid)
            session = self._send_sessions[rid]
            # A peer may be the first rank to observe cancellation.  Apply
            # that decision locally, but retain the request and session while
            # a native write is still active.  CANCELLED is a decision, not a
            # reclamation acknowledgement.
            session.cancel()

        for rid in failed:
            self._legacy_failed_sessions.add(rid)
            self._send_sessions[rid].seal_and_check_quiescent()

        for rid in reclaimable_cancelled:
            self._legacy_failed_sessions.discard(rid)
            self._leave_diag_state("ctx_send", rid)
            self._record_diag_transition("ctx_global_cancelled", rid, once=True)
            self._send_sessions[rid].close()
            del self._send_reqs[rid]
            del self._send_sessions[rid]
            self._record_context_cancelled_request_id(rid)

        for rid in completed:
            if mark_complete:
                self._send_reqs[rid].state = LlmRequestState.DISAGG_CONTEXT_COMPLETE
            self._leave_diag_state("ctx_send", rid)
            self._record_diag_transition("ctx_global_complete", rid, once=True)
            self._send_sessions[rid].close()
            del self._send_reqs[rid]
            del self._send_sessions[rid]
        for rid in reclaimable_failed:
            self._legacy_failed_sessions.discard(rid)
            self._leave_diag_state("ctx_send", rid)
            self._record_diag_transition("ctx_global_failed", rid, once=True)
        self._close_failed_sessions(self._send_sessions, self._send_reqs, reclaimable_failed)

        # Sweep orphaned RecvReqInfo entries from ADP broadcast on non-assigned
        # DP ranks (entries that will never have a TxSession created for them).
        self._transfer_worker.sweep_stale_req_infos()

        self._maybe_log_diagnostics()
        return completed, reclaimable_failed

    def check_gen_transfer_status(self, at_least_request_num: Optional[int]):
        self._maybe_log_diagnostics()
        if getattr(self, "_async_consensus", None) is not None:
            self._progress_async_consensus()
        if not self._ever_had_recv_session and not self._gen_need_sync:
            return [], [], []
        block_all = at_least_request_num is None
        wait_num = at_least_request_num if not block_all else 0
        need_progress = wait_num > 0
        if need_progress:
            self._poll_gen_sessions_for_poll_interval(wait_num)

        local_completed, local_failed = self._collect_done(self._recv_sessions, self._recv_reqs)
        for rid in local_completed:
            self._record_diag_transition("gen_local_complete", rid, once=True)
        for rid in local_failed:
            self._record_diag_transition("gen_local_failed", rid, once=True)
        to_process = self._build_to_process(
            self._recv_sessions,
            self._gen_consensus(local_completed + local_failed),
            0 if need_progress else wait_num,
            block_all,
        )

        known_ids = list(self._recv_sessions)
        completed, failed, failed_quiescent = [], [], []
        cancelled, cancel_quiescent = [], []
        for rid, session in self._recv_sessions.items():
            if session.status == SessionStatus.CANCELLED:
                cancelled.append(rid)
                if not session.has_transferring_tasks():
                    cancel_quiescent.append(rid)
            elif rid in self._legacy_failed_sessions or session.has_failed():
                failed.append(rid)
                if session.seal_and_check_quiescent():
                    failed_quiescent.append(rid)
        for rid in to_process:
            session = self._recv_sessions[rid]
            result = self._diagnostic_wait_complete(
                "gen",
                rid,
                session,
                blocking=block_all,
            )
            if session.status == SessionStatus.CANCELLED:
                # Session cancelled — either by local cancel_request() (user
                # cancel) or by a remote CANCEL_SESSION message (e.g. CTX
                # server timeout).  Return the req objects so the caller can
                # distinguish the two cases and set the appropriate state.
                if rid in failed:
                    failed.remove(rid)
                if rid in failed_quiescent:
                    failed_quiescent.remove(rid)
                if rid not in cancelled:
                    cancelled.append(rid)
                if rid not in cancel_quiescent and not session.has_transferring_tasks():
                    cancel_quiescent.append(rid)
            elif result == WaitResult.COMPLETED:
                completed.append(rid)
            elif result == WaitResult.FAILED:
                if rid not in failed:
                    failed.append(rid)
                if rid not in failed_quiescent and session.seal_and_check_quiescent():
                    failed_quiescent.append(rid)
            # else: None — KV done but aux still in flight; re-poll next cycle

        # All ranks must agree on per-rid outcome to avoid req.state divergence.
        cancelled, reclaimable_cancelled, failed, reclaimable_failed, completed = (
            self._gen_consensus_outcome(
                to_process,
                known_ids,
                cancelled,
                cancel_quiescent,
                failed,
                failed_quiescent,
                completed,
            )
        )

        cancelled_reqs = []
        for rid in cancelled:
            self._legacy_failed_sessions.discard(rid)
            session = self._recv_sessions[rid]
            # Native receive cancellation is not reclaimable until both
            # active writes and sender acknowledgements have drained.
            session.cancel()
        for rid in failed:
            self._legacy_failed_sessions.add(rid)
            self._recv_sessions[rid].seal_and_check_quiescent()
        for rid in reclaimable_cancelled:
            self._legacy_failed_sessions.discard(rid)
            session = self._recv_sessions[rid]
            cancelled_reqs.append(self._recv_reqs[rid])
            self._leave_diag_state("gen_recv", rid)
            self._record_diag_transition("gen_global_cancelled", rid, once=True)
            session.close()
            del self._recv_reqs[rid]
            del self._recv_sessions[rid]

        for rid in completed:
            session = self._recv_sessions[rid]
            req = self._recv_reqs[rid]
            # transfer_end already stamped at completion detection above.
            req.set_kv_cache_size(getattr(req, "py_kv_cache_xfer_bytes", 0))
            if self._need_aux_transfer(req):
                self._enter_diag_state("gen_aux_apply", rid)
                self._record_diag_transition("gen_aux_apply_enter", rid, once=True)
                try:
                    self._apply_aux(session, req)
                finally:
                    self._leave_diag_state("gen_aux_apply", rid)
                self._record_diag_transition("gen_aux_apply_exit", rid, once=True)
            self._assert_disagg_history_declared(req)
            req.state = LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE
            self._leave_diag_state("gen_recv", rid)
            self._record_diag_transition("gen_global_complete", rid, once=True)
            session.close()
            del self._recv_reqs[rid]
            del self._recv_sessions[rid]
        if reclaimable_failed:
            logger.warning(
                f"Disagg gen transfer FAILED rank={self._dist.rank} "
                f"rids={reclaimable_failed} gen_need_sync={self._gen_need_sync}"
            )
        for rid in reclaimable_failed:
            self._legacy_failed_sessions.discard(rid)
            self._leave_diag_state("gen_recv", rid)
            self._record_diag_transition("gen_global_failed", rid, once=True)
        self._close_failed_sessions(self._recv_sessions, self._recv_reqs, reclaimable_failed)

        self._maybe_log_diagnostics()
        return completed, reclaimable_failed, cancelled_reqs

    def _poll_gen_sessions_for_poll_interval(self, wait_num: int) -> None:
        poll_interval_s = (self.kv_transfer_poll_interval_ms or 0) / 1000.0
        deadline = time.monotonic() + poll_interval_s
        while True:
            completed, failed = self._collect_done(self._recv_sessions, self._recv_reqs)
            if len(completed) + len(failed) >= wait_num:
                return
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                return
            for rid, session in self._recv_sessions.items():
                self._diagnostic_wait_complete(
                    "gen",
                    rid,
                    session,
                    blocking=False,
                )
            time.sleep(min(0.001, remaining_s))

    def check_gen_transfer_complete(self):
        return len(self._recv_sessions) == 0

    def _assert_disagg_history_declared(self, req: LlmRequest) -> None:
        """Verify the V2 scheduler pre-declared prompt_len as history.

        Call right before the TRANS_COMPLETE state transition.  The V2
        scheduler's ``_try_schedule_disagg_gen_init`` calls
        ``prepare_disagg_gen_init``, which sets ``kv_cache.history_length``
        to ``prompt_len`` at allocation time so SWA stale computation
        skips pre-window blocks. If that contract is violated, SWA /
        sparse-attn pools may fill with pre-window prompt blocks and the
        V2 scheduler can deadlock under high concurrency (e.g., benchmark
        fill-phase).

        No-op for V1 managers (which lack ``get_history_length``) and for
        V2 caches with only full-context life cycles (where the watermark
        has no allocation effect).
        """
        get_history = getattr(self._kv_cache_manager, "get_history_length", None)
        if get_history is None:
            return
        prompt_len = getattr(req, "prompt_len", None)
        if not prompt_len or prompt_len <= 0:
            return
        history = get_history(req)
        if history is None:
            # Cache was already released (e.g., cancelled mid-transfer); nothing to verify.
            return
        if history < prompt_len:
            raise RuntimeError(
                f"req {req.py_request_id}: kv_cache.history_length={history} "
                f"< prompt_len={prompt_len} at TRANS_COMPLETE boundary. "
                f"V2 scheduler must call prepare_disagg_gen_init() in "
                f"_try_schedule_disagg_gen_init."
            )

    def owns_request(self, req: LlmRequest) -> bool:
        """Return whether transfer or readiness state still owns ``req``."""
        rid = get_unique_rid(req)
        return (
            rid in self._wait_reqs
            or rid in self._send_sessions
            or rid in self._send_reqs
            or rid in self._recv_sessions
            or rid in self._recv_reqs
            or rid in getattr(self, "_async_terminal_cancelled", {})
            or rid in self._async_ready_published
            or any(key[0] == rid for key in self._async_ready_prepared)
            or any(key[0] == rid for key in self._async_ready_activated)
            or any(key[0] == rid for key in self._async_ready_aborted)
        )

    def cancel_request(self, req: LlmRequest) -> bool:
        """Cancel the transfer for the given request.

        Returns False if any task is mid-write (TRANSFERRING); caller must
        retry next iteration. Returns True when safe to free KV memory.
        """
        rid = get_unique_rid(req)
        if getattr(self, "_async_consensus", None) is not None:
            self._progress_async_consensus()

        if self._acknowledge_async_terminal_cancellation(rid, req):
            return True

        # A published terminal vote is immutable. In particular, a late local
        # cancellation must not mutate a session that already voted COMPLETE
        # or FAILED, nor send a contradictory remote cancellation. Retain all
        # resources until the coordinator's authoritative commit is applied.
        if (
            getattr(self, "_async_terminal_consensus_enabled", False)
            and rid in self._async_terminal_published
        ):
            self._apply_async_ctx_terminal_commits(
                mark_complete=False,
                only_request_id=rid,
            )
            if self._acknowledge_async_terminal_cancellation(rid, req):
                return True
            return not self.owns_request(req)

        # A generation-first readiness round owns the request until its
        # authoritative schedule activation completes or its abort finalizes.
        # Never remove that ownership directly from the cancellation path.
        waiting_req = self._wait_reqs.get(rid)
        prepared_key = next((key for key in self._async_ready_prepared if key[0] == rid), None)
        prepared_req = (
            self._async_ready_prepared.get(prepared_key) if prepared_key is not None else None
        )
        activated_key = next(
            (key for key in self._async_ready_activated if key[0] == rid),
            None,
        )
        activated_req = (
            self._async_ready_activated.get(activated_key) if activated_key is not None else None
        )
        published_epoch = self._async_ready_published.get(rid)
        readiness_owned = (
            waiting_req is not None
            or prepared_req is not None
            or activated_req is not None
            or published_epoch is not None
            or any(key[0] == rid for key in self._async_ready_aborted)
        )
        if getattr(self, "_async_peer_ready_consensus_enabled", False) and readiness_owned:
            cancelled_req = (
                waiting_req
                if waiting_req is not None
                else prepared_req
                if prepared_req is not None
                else activated_req
                if activated_req is not None
                else req
            )
            setattr(
                cancelled_req,
                _ASYNC_READY_CANCELLED_EPOCH_ATTR,
                published_epoch
                if published_epoch is not None
                else self._async_ready_epoch.get(rid, 0),
            )
            if published_epoch is None:
                # A peer may already have voted READY even though local peer
                # metadata has not arrived. Join the same epoch with a
                # withdrawal and retain the waiting request until every rank
                # applies READY_ABORT and the coordinator finalizes it.
                published_epoch = self._async_ready_epoch.get(rid, 0)
                self._async_ready_published[rid] = published_epoch
            key = (rid, published_epoch)
            if key in self._async_ready_aborted:
                return False
            if key in self._async_ready_acknowledged:
                # ACK is an irrevocable lease. Let the authoritative PP
                # schedule activate it and wait for READY_COMPLETE before a
                # cancellation can reclaim request resources.
                return False
            coordinator = cast(AsyncConsensusCoordinator, self._async_consensus)
            if key not in self._async_ready_withdrawn:
                if coordinator.withdraw_ready(rid, published_epoch):
                    self._async_ready_withdrawn.add(key)
                    self._record_async_transition(
                        "ready_withdraw",
                        rid,
                        published_epoch,
                        ConsensusOutcome.WITHDRAWN,
                    )
                else:
                    # The coordinator already considers the local lease
                    # irrevocable; retain ownership until its final event.
                    self._async_ready_acknowledged.add(key)
            self._progress_async_consensus()
            if rid in self._async_ready_published:
                return False
        else:
            if self._wait_reqs.pop(rid, None) is not None:
                self._leave_diag_state("ctx_wait", rid)
                self._record_diag_transition("ctx_wait_cancelled", rid, once=True)

        has_transferring = False

        if rid in self._send_sessions:
            self._send_sessions[rid].cancel()
            if self._send_sessions[rid].has_transferring_tasks():
                has_transferring = True
            elif getattr(self, "_async_terminal_consensus_enabled", False):
                self._publish_async_ctx_terminal_votes(block_all=False)
                self._progress_async_consensus()
                self._apply_async_ctx_terminal_commits(mark_complete=False, only_request_id=rid)
                if rid in self._send_sessions:
                    has_transferring = True
            else:
                self._leave_diag_state("ctx_send", rid)
                self._record_diag_transition("ctx_send_cancelled", rid, once=True)
                self._send_sessions[rid].close()
                del self._send_reqs[rid]
                del self._send_sessions[rid]

        if rid in self._recv_sessions:
            self._recv_sessions[rid].cancel()
            if self._recv_sessions[rid].has_transferring_tasks():
                has_transferring = True
            else:
                self._leave_diag_state("gen_recv", rid)
                self._record_diag_transition("gen_receive_cancelled", rid, once=True)
                self._recv_sessions[rid].close()
                del self._recv_reqs[rid]
                del self._recv_sessions[rid]

        if has_transferring:
            return False
        return True

    def get_disaggregated_params(self) -> Dict[str, Any]:
        # Keep this aligned with fields populated in respond_and_send_async().
        # These values are server-level metadata used to seed generation-first
        # requests before context-phase response data arrives.
        #
        # With ADP (enable_attention_dp), ctx_dp_rank is not known at
        # registration time because the context scheduler has not yet assigned
        # the request to a DP rank.  Return None so that the gen-side Receiver
        # broadcasts REQUEST_DATA to all ctx DP ranks.  The actual ctx_dp_rank
        # is stamped into ContextPhaseParams by respond_and_send_async() after
        # the prefill is scheduled.
        ctx_dp_rank = None if self._mapping.enable_attention_dp else self._dp_rank
        return {
            "ctx_dp_rank": ctx_dp_rank,
            "ctx_info_endpoint": [self._context_info_endpoint]
            if self._context_info_endpoint
            else None,
        }

    def exclude_context_requests_from_readiness(self, requests: List[LlmRequest]) -> None:
        """Withdraw known-cancelled requests before readiness progression."""
        for req in requests:
            rid = get_unique_rid(req)
            epoch = self._async_ready_published.get(rid, self._async_ready_epoch.get(rid, 0))
            setattr(req, _ASYNC_READY_CANCELLED_EPOCH_ATTR, epoch)

            if not getattr(self, "_async_peer_ready_consensus_enabled", False):
                # Terminal-only/default-off readiness still uses the legacy
                # wait map. Remove an existing waiter before the immediately
                # following prepare_context_requests([]) can promote it.
                if self._wait_reqs.pop(rid, None) is not None:
                    self._leave_diag_state("ctx_wait", rid)
                    self._record_diag_transition("ctx_wait_excluded", rid, epoch, once=True)
                req.state = LlmRequestState.DISAGG_CONTEXT_WAIT_SCHEDULER
                continue
            prepared_key = next((key for key in self._async_ready_prepared if key[0] == rid), None)
            activated_key = next(
                (key for key in self._async_ready_activated if key[0] == rid),
                None,
            )
            readiness_owned = (
                rid in self._wait_reqs
                or rid in self._async_ready_published
                or prepared_key is not None
                or activated_key is not None
                or any(key[0] == rid for key in self._async_ready_aborted)
            )
            if not readiness_owned:
                # A newly fetched request has not entered a readiness round on
                # any rank yet. The cancellation marker keeps it excluded when
                # prepare_context_requests() runs immediately afterwards.
                req.state = LlmRequestState.DISAGG_CONTEXT_WAIT_SCHEDULER
                continue

            if rid not in self._async_ready_published:
                self._async_ready_published[rid] = epoch
            key = (rid, epoch)
            if key in self._async_ready_acknowledged:
                # The prepared lease is already irrevocable. Let its release
                # finish without rolling local state back; cancellation will
                # retry against the next lifecycle phase.
                continue
            req.state = LlmRequestState.DISAGG_CONTEXT_WAIT_SCHEDULER
            if key in self._async_ready_aborted:
                continue
            if key in self._async_ready_withdrawn:
                continue
            coordinator = cast(AsyncConsensusCoordinator, self._async_consensus)
            if coordinator.withdraw_ready(rid, epoch):
                self._async_ready_withdrawn.add(key)
                self._record_async_transition(
                    "ready_withdraw",
                    rid,
                    epoch,
                    ConsensusOutcome.WITHDRAWN,
                )
            else:
                self._async_ready_acknowledged.add(key)

    def activate_context_requests_for_schedule(self, requests: List[LlmRequest]) -> None:
        """Activate readiness leases selected by the authoritative PP schedule.

        Rank zero receives ``READY_RELEASE`` and runs the only authoritative
        scheduling decision.  Followers call this hook after deserializing
        that exact schedule and immediately before their mirrored scheduler
        pass.  Consequently no follower can mutate scheduler or KV state for
        a request that rank zero did not select.
        """
        if not getattr(self, "_async_peer_ready_consensus_enabled", False):
            return
        coordinator = cast(AsyncConsensusCoordinator, self._async_consensus)
        for req in requests:
            rid = get_unique_rid(req)
            epoch = self._async_ready_published.get(rid)
            if epoch is None:
                continue
            key = (rid, epoch)
            prepared_req = self._async_ready_prepared.get(key)
            if prepared_req is None:
                # The hook receives all scheduled requests, most of which do
                # not belong to a readiness round.  A published-but-not-yet-
                # prepared request, however, must not appear in the schedule.
                if key in self._async_ready_activated:
                    continue
                raise RuntimeError(
                    "authoritative PP schedule selected readiness before "
                    f"PREPARE: request_id={rid}, epoch={epoch}"
                )
            if self._dist.rank == 0 and key not in self._async_ready_released:
                raise RuntimeError(
                    "scheduling rank selected readiness before READY_RELEASE: "
                    f"request_id={rid}, epoch={epoch}"
                )
            prepared_req.state = LlmRequestState.CONTEXT_INIT
            del self._async_ready_prepared[key]
            self._async_ready_activated[key] = prepared_req
            coordinator.acknowledge_ready_activation(rid, epoch)
            self._leave_diag_state("ready_prepared", rid)
            self._leave_diag_state("ready_released", rid)
            self._enter_diag_state("ready_activated", rid)
            self._record_async_transition("ready_activate", rid, epoch)

    def prepare_context_requests(self, requests: List[LlmRequest]):
        # Place new generation-first context requests into wait state, then
        # use allgather consensus to promote ready requests to CONTEXT_INIT.
        self._maybe_log_diagnostics()
        for req in requests:
            rid = get_unique_rid(req)
            if rid not in self._send_sessions:
                if getattr(self, "_async_peer_ready_consensus_enabled", False):
                    if self._bind_async_ready_abort(req):
                        self._wait_reqs.pop(rid, None)
                        self._leave_diag_state("ctx_wait", rid)
                        self._record_diag_transition("ctx_wait_abort_bound", rid, once=True)
                        continue
                if getattr(req, _ASYNC_READY_CANCELLED_EPOCH_ATTR, None) is not None:
                    self._wait_reqs.pop(rid, None)
                    self._leave_diag_state("ctx_wait", rid)
                    self._record_diag_transition("ctx_wait_pre_cancelled", rid, once=True)
                    continue
                self._wait_reqs[rid] = req
                req.state = LlmRequestState.DISAGG_CONTEXT_WAIT_SCHEDULER
                self._enter_diag_state("ctx_wait", rid)
                self._record_diag_transition("ctx_wait_enter", rid, once=True)

        # Materialize this iteration's requests before polling.  A READY_ABORT
        # can legally arrive before this rank has voted, and its handler must
        # be able to bind the request instead of fail-stopping or allowing a
        # contradictory READY publication below.
        if getattr(self, "_async_consensus", None) is not None:
            self._progress_async_consensus()

        if getattr(self, "_async_peer_ready_consensus_enabled", False):
            self._prepare_context_requests_async()
            self._maybe_log_diagnostics()
            return

        # Nothing waiting on any rank, so skip the consensus. The waiting set is the same on every
        # rank, so they all skip together.
        if not self._wait_reqs:
            return

        # Check which waiting requests have peer info locally, then allgather
        # consensus so all TP/PP ranks agree before promoting.
        # Without consensus, background peer info arriving at different times on
        # different ranks causes scheduling mismatches → hang.
        local_ready = [
            rid
            for rid in self._wait_reqs
            if self._transfer_worker.has_all_peer_req_infos_for_send(rid)
        ]
        for rid in self._ctx_consensus(local_ready):
            self._wait_reqs[rid].state = LlmRequestState.CONTEXT_INIT
            del self._wait_reqs[rid]
            self._leave_diag_state("ctx_wait", rid)
            self._record_diag_transition("ctx_legacy_ready", rid, once=True)
        self._maybe_log_diagnostics()

    def _prepare_context_requests_async(self) -> None:
        coordinator = cast(AsyncConsensusCoordinator, self._async_consensus)
        for rid in list(self._wait_reqs):
            if rid in self._async_ready_published:
                continue
            if not self._transfer_worker.has_all_peer_req_infos_for_send(rid):
                continue
            epoch = self._async_ready_epoch.get(rid, 0)
            self._record_diag_transition("ctx_peer_info_ready", rid, epoch, once=True)
            coordinator.publish_ready(rid, epoch)
            self._async_ready_published[rid] = epoch
            self._record_async_transition("ready_vote", rid, epoch)
        # Poll again so the PP-last coordinator can begin prepare immediately
        # after recording its local vote.
        self._progress_async_consensus()

    def _check_compatible(self):
        if self._mapping.cp_size != 1:
            raise ValueError(
                f"KvCacheTransceiverV2: _check_compatible: only support context parallelism is 1: "
                f"cp_size: {self._mapping.cp_size}"
            )

    def commit_blocks_for_reuse(self, req) -> None:
        self._reuse_adapter.commit_blocks_for_reuse(req)

    def get_context_state(self):
        raise NotImplementedError("get_context_state is not implemented")
