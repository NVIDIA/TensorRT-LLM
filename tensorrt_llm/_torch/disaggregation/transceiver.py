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
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, cast

import numpy as np
import torch

import tensorrt_llm.bindings
from tensorrt_llm import logger
from tensorrt_llm._torch.disaggregation.base.transfer import (
    KVSlice,
    RxSessionBase,
    SessionStatus,
    TokenRange,
    TxSessionBase,
    WaitResult,
    get_unique_rid,
)
from tensorrt_llm._torch.disaggregation.capability_negotiation import (
    make_transceiver_lifecycle_advertisement,
)
from tensorrt_llm._torch.disaggregation.handoff import HandoffEventState, HandoffLifecycleEvent
from tensorrt_llm._torch.disaggregation.lifecycle import (
    CancelResult,
    LifecycleCapability,
    LogicalDisposition,
    PhysicalDisposition,
    ShutdownResult,
    TransceiverCapabilities,
)
from tensorrt_llm._torch.disaggregation.native.bounce import (
    config_from_size as bounce_config_from_size,
)
from tensorrt_llm._torch.disaggregation.native.perf_logger import perf_log_manager
from tensorrt_llm._torch.disaggregation.native.transfer import TransferWorker, TransferWorkerConfig
from tensorrt_llm._torch.disaggregation.protocol import (
    PROTOCOL_V1_REQUIRED_CAPABILITIES,
    ProtocolVersion,
)
from tensorrt_llm._torch.disaggregation.resource.allocation_lease import AllocationLease
from tensorrt_llm._torch.disaggregation.resource.cache_reuse import (
    CacheReuseAdapter,
    create_cache_reuse_adapter,
)
from tensorrt_llm._torch.disaggregation.resource.page import MambaLayerGroup
from tensorrt_llm._torch.disaggregation.resource.utils import get_physical_pool
from tensorrt_llm._torch.distributed.communicator import Distributed
from tensorrt_llm._torch.pyexecutor.kv_cache_transceiver import KvCacheTransceiver
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.mamba_cache_manager import (
    MambaHybridCacheManager,
    MambaHybridCacheManagerV2,
    MixedMambaHybridCacheManager,
)
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._utils import nvtx_range
from tensorrt_llm.bindings import LlmRequestState
from tensorrt_llm.bindings.executor import ContextPhaseParams
from tensorrt_llm.disaggregated_params import DisaggScheduleStyle, TransceiverLifecycleAdvertisement
from tensorrt_llm.llmapi.llm_args import CacheTransceiverConfig
from tensorrt_llm.mapping import Mapping

# A non-drained transceiver is deliberately retained even if its caller drops
# the last reference. It owns the LlmRequest/session objects that keep receive
# targets from being reclaimed before terminal transport evidence arrives.
_NON_DRAINED_TRANSCEIVERS: set["KvCacheTransceiverV2"] = set()
_LIFECYCLE_PROTOCOL_VERSION_ENV = "TRTLLM_DISAGG_LIFECYCLE_PROTOCOL_VERSION"


@dataclass
class _ContextRetirementProgress:
    """One rank's durable preparation of a context-side terminal candidate."""

    rid: int
    session: TxSessionBase
    request: LlmRequest
    local_outcome: str
    mark_complete: bool
    outcome: Optional[str] = None
    report_result: bool = True
    session_closed: bool = False
    maps_validated: bool = False
    state_updated: bool = False
    request_map_retired: bool = False
    session_map_retired: bool = False


@dataclass
class _GenerationRetirementProgress:
    """One rank's durable preparation of a generation-side terminal candidate."""

    rid: int
    session: RxSessionBase
    request: LlmRequest
    local_outcome: str
    outcome: Optional[str] = None
    report_result: bool = True
    kv_size_set: bool = False
    aux_applied: bool = False
    history_checked: bool = False
    session_closed: bool = False
    maps_validated: bool = False
    state_updated: bool = False
    request_map_retired: bool = False
    session_map_retired: bool = False


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
    @staticmethod
    def _configured_lifecycle_protocol_version() -> ProtocolVersion:
        raw_version = os.getenv(_LIFECYCLE_PROTOCOL_VERSION_ENV, "0")
        try:
            version = int(raw_version)
        except ValueError as error:
            raise ValueError(f"{_LIFECYCLE_PROTOCOL_VERSION_ENV} must be 0 or 1") from error
        if version not in (
            ProtocolVersion.QUALIFIED_LEGACY,
            ProtocolVersion.GENERATION_SAFE,
        ):
            raise ValueError(f"{_LIFECYCLE_PROTOCOL_VERSION_ENV} must be 0 or 1")
        return ProtocolVersion(version)

    @staticmethod
    def _build_lifecycle_capabilities(
        mapping: Mapping,
        *,
        protocol_version: ProtocolVersion,
        bounce_transfer_enabled: bool,
        supports_generation_first: bool,
    ) -> TransceiverCapabilities:
        """Build the one immutable contract shared by server and rank planes."""
        if protocol_version == ProtocolVersion.QUALIFIED_LEGACY:
            supported = {
                LifecycleCapability.DIRECT_TRANSFER,
                LifecycleCapability.EXACT_WRITER_TRACKING,
                LifecycleCapability.GENERATION_FIRST,
                LifecycleCapability.PIPELINE_PARALLEL,
                LifecycleCapability.PUBLICATION_GATE,
                LifecycleCapability.TENSOR_PARALLEL,
            }
            if bounce_transfer_enabled and not mapping.enable_attention_dp:
                supported.add(LifecycleCapability.BOUNCE_TRANSFER)
            return TransceiverCapabilities(
                protocol_version=int(protocol_version),
                supported=frozenset(supported),
                qualified_legacy_mode=True,
            )
        if protocol_version != ProtocolVersion.GENERATION_SAFE:
            raise ValueError(f"unsupported lifecycle protocol version {int(protocol_version)}")

        supported = {
            LifecycleCapability(capability) for capability in PROTOCOL_V1_REQUIRED_CAPABILITIES
        }
        supported.update(
            {
                LifecycleCapability.DIRECT_TRANSFER,
                LifecycleCapability.PIPELINE_PARALLEL,
                LifecycleCapability.TENSOR_PARALLEL,
            }
        )
        if mapping.enable_attention_dp:
            supported.add(LifecycleCapability.ATTENTION_DATA_PARALLEL)
        if bounce_transfer_enabled and not mapping.enable_attention_dp:
            supported.add(LifecycleCapability.BOUNCE_TRANSFER)
        if supports_generation_first:
            supported.add(LifecycleCapability.GENERATION_FIRST)
        return TransceiverCapabilities(
            protocol_version=int(protocol_version),
            supported=frozenset(supported),
            qualified_legacy_mode=False,
        )

    @staticmethod
    def _validate_generation_safe_allocator(
        kv_cache_manager: KVCacheManager,
        lifecycle_contract: TransceiverCapabilities,
    ) -> None:
        if lifecycle_contract.protocol_version != ProtocolVersion.GENERATION_SAFE:
            return
        if isinstance(kv_cache_manager, MixedMambaHybridCacheManager):
            raise RuntimeError(
                "lifecycle protocol v1 does not yet support "
                "MixedMambaHybridCacheManager because its independent Mamba "
                "state slots lack allocation-generation leases"
            )
        if not callable(getattr(kv_cache_manager, "snapshot_and_lease", None)):
            raise RuntimeError(
                "lifecycle protocol v1 requires allocator snapshot_and_lease support"
            )

    @staticmethod
    def _make_lifecycle_advertisement(
        *,
        lifecycle_contract: TransceiverCapabilities,
        instance_id: str,
        mapping: Mapping,
    ) -> TransceiverLifecycleAdvertisement:
        return make_transceiver_lifecycle_advertisement(
            protocol_version=lifecycle_contract.protocol_version,
            capabilities=lifecycle_contract.supported,
            qualified_legacy_mode=lifecycle_contract.qualified_legacy_mode,
            backend="python",
            instance_id=instance_id,
            world_size=mapping.world_size,
            tp_size=mapping.tp_size,
            pp_size=mapping.pp_size,
            cp_size=mapping.cp_size,
            attention_dp=mapping.enable_attention_dp,
        )

    @property
    def requires_physical_drain_before_request_release(self) -> bool:
        return True

    def __init__(
        self,
        mapping: Mapping,
        dist: Distributed,
        kv_cache_manager: KVCacheManager,
        cache_transceiver_config: CacheTransceiverConfig,
    ):
        self._session_admission_lock = threading.RLock()
        # Serialize every collective-bearing status/preparation call. Shutdown
        # does not wait on this lock: it closes admission and asks active
        # sessions to cancel, then returns non-drained until the active call
        # leaves the collective stream.
        self._status_call_lock = threading.Lock()
        self._active_status_calls = 0
        self._retirement_fault: Optional[BaseException] = None
        self._shutdown_started = False
        self._shutdown = False
        self._dist: Distributed = dist
        self._kv_cache_manager = kv_cache_manager
        self._mapping = mapping
        self.kv_transfer_timeout_ms = cache_transceiver_config.kv_transfer_timeout_ms
        self.kv_transfer_poll_interval_ms = cache_transceiver_config.kv_transfer_poll_interval_ms
        self._bounce_transfer_enabled = cache_transceiver_config.kv_cache_bounce_size_mb > 0
        protocol_version = self._configured_lifecycle_protocol_version()
        self._lifecycle_contract = self._build_lifecycle_capabilities(
            mapping,
            protocol_version=protocol_version,
            bounce_transfer_enabled=self._bounce_transfer_enabled,
            supports_generation_first=True,
        )
        self._validate_generation_safe_allocator(
            kv_cache_manager,
            self._lifecycle_contract,
        )
        self._sender_future_timeout_ms = (
            cache_transceiver_config.kv_transfer_sender_future_timeout_ms
        )
        self._check_compatible()
        self._reuse_adapter: CacheReuseAdapter = create_cache_reuse_adapter(kv_cache_manager)

        self._device_id = torch.cuda.current_device()
        logger.info(f"device_id: {self._device_id} in KvCacheTransceiverV2")
        self._instance_name = self._broadcast_instance_name()
        self._lifecycle_advertisement = self._make_lifecycle_advertisement(
            lifecycle_contract=self._lifecycle_contract,
            instance_id=self._instance_name,
            mapping=mapping,
        )
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
                lifecycle_contract=self._lifecycle_contract,
            )
        )
        self._dp_rank = mapping.tp_rank if mapping.enable_attention_dp else 0
        self._context_info_endpoint = self._broadcast_context_endpoint()
        self._init_sync_policy()
        self._exchange_rank_info()

        self._send_sessions: Dict[int, TxSessionBase] = {}
        self._recv_sessions: Dict[int, RxSessionBase] = {}
        self._send_reqs = {}
        self._recv_reqs = {}
        self._wait_reqs = {}
        # Exact wait owners whose cancellation raced a collective-bearing
        # prepare call. Promotion must ignore them until cancel_request retries
        # after that call exits and retires the wait entry.
        self._cancelled_wait_reqs: Dict[int, LlmRequest] = {}
        # Terminal candidates are enrolled provisionally before the existing
        # outcome consensus. Keep their exact local owners and phase progress
        # until this rank commits the distributed decision, including across
        # retryable local preparation failures.
        self._context_retirements: Dict[int, _ContextRetirementProgress] = {}
        self._generation_retirements: Dict[int, _GenerationRetirementProgress] = {}
        # A session owns every exact allocator lease whose descriptors it may
        # publish. Physical session close is the only path that may settle
        # these leases with reusable evidence.
        self._allocation_leases_by_session: dict[object, list[AllocationLease]] = {}
        # Settlement failures and allocator IN_DOUBT results are permanent
        # local quarantine roots. They deliberately keep shutdown non-drained.
        self._retained_allocation_leases: list[AllocationLease] = []
        # Control-only notifications consumed by PyExecutor after distributed
        # receive retirement. Only rank 0 publishes them to the result queue.
        self._handoff_lifecycle_events: list[tuple[int, Optional[int], HandoffLifecycleEvent]] = []
        self._page_table = self._transfer_worker.page_table
        # _slice_num_bytes() is this rank's KV shard, so scale by tp_size to get the request total (kv_cache_size),
        # except under attention DP where the local count already is the total.
        self._kv_size_rank_factor = 1 if mapping.enable_attention_dp else max(1, mapping.tp_size)

        # Sticky role markers; flip True once any session opens, used to short-circuit
        # per-iter tp_allgather when this transceiver never sends/receives.
        self._ever_had_send_session: bool = False
        self._ever_had_recv_session: bool = False

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

    def _lifecycle_contract_fingerprint(
        self,
    ) -> tuple[str, int, tuple[str, ...], bool]:
        advertisement = self._lifecycle_advertisement
        return (
            advertisement.instance_id,
            advertisement.protocol_version,
            advertisement.capabilities,
            advertisement.qualified_legacy_mode,
        )

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
        local_lifecycle = self._lifecycle_contract_fingerprint()
        lifecycle_contracts = cast(list, self._dist.allgather(local_lifecycle))
        if any(contract != local_lifecycle for contract in lifecycle_contracts):
            raise RuntimeError("distributed transfer workers have inconsistent lifecycle contracts")
        endpoints = cast(list, self._dist.allgather(self._transfer_worker.sender_endpoint))
        endpoint_incarnations = cast(
            list,
            self._dist.allgather(str(self._transfer_worker.endpoint_incarnation)),
        )
        layer_num = len(self._kv_cache_manager.pp_layers)
        if isinstance(self._kv_cache_manager, MambaHybridCacheManager) and not isinstance(
            self._kv_cache_manager, MambaHybridCacheManagerV2
        ):
            layer_num += len(self._kv_cache_manager._impl.mamba_layer_offsets)
        layer_num_per_pp = cast(list, getattr(self._dist, "pp_allgather")(layer_num))
        self._transfer_worker.populate_instance_and_rank_info(
            endpoints=endpoints,
            layer_num_per_pp=layer_num_per_pp,
            endpoint_incarnations=endpoint_incarnations,
            lifecycle_contract=self._lifecycle_contract,
            lifecycle_instance_id=self._lifecycle_advertisement.instance_id,
        )
        logger.info(f"transfer worker ctx_server_endpoints: {endpoints}")
        logger.info(f"layer_num_per_pp: {layer_num_per_pp}")
        logger.info(f"self._context_info_endpoint: {self._context_info_endpoint}")

    def get_status_dump(self) -> str:
        """Return a one-line summary of transceiver state for debugging hangs."""

        def summarize(
            sessions: Dict[int, Any],
            include_receiver_ready: bool,
        ) -> str:
            sessions_snapshot = list(sessions.values())
            status_counts = Counter()
            receiver_ready = 0
            for session in sessions_snapshot:
                status = session.status
                if isinstance(status, SessionStatus):
                    status_counts[status] += 1
                else:
                    status_counts["unknown"] += 1

                if include_receiver_ready:
                    receiver_ready += int(bool(session.receiver_ready))

            fields = [
                f"sessions={len(sessions_snapshot)}",
                f"init={status_counts[SessionStatus.INIT]}",
                f"ready_to_transfer={status_counts[SessionStatus.READY]}",
                f"transferring={status_counts[SessionStatus.TRANSFERRING]}",
                f"kv_transferred={status_counts[SessionStatus.KV_TRANSFERRED]}",
                f"fully_transferred={status_counts[SessionStatus.FULLY_TRANSFERRED]}",
                f"error={status_counts[SessionStatus.ERROR]}",
                f"cancelled={status_counts[SessionStatus.CANCELLED]}",
                f"unknown={status_counts['unknown']}",
            ]
            if include_receiver_ready:
                fields.append(f"peer_ready={receiver_ready}/{len(sessions_snapshot)}")
            return ", ".join(fields)

        tx_status = summarize(self._send_sessions, include_receiver_ready=True)
        rx_status = summarize(self._recv_sessions, include_receiver_ready=False)
        return (
            f"KV cache transceiver | backend=NIXL | TX({tx_status}) | RX({rx_status}) | "
            f"waiting_for_peer_info={len(self._wait_reqs)}"
        )

    def _get_session_admission_lock(self):
        """Return the launch/shutdown gate, including partial-init cleanup."""
        admission_lock = getattr(self, "_session_admission_lock", None)
        if admission_lock is None:
            admission_lock = threading.RLock()
            self._session_admission_lock = admission_lock
        return admission_lock

    def _get_status_call_lock(self):
        """Return the collective-stream lock, including object.__new__ fixtures."""
        status_lock = getattr(self, "_status_call_lock", None)
        if status_lock is None:
            status_lock = threading.Lock()
            self._status_call_lock = status_lock
        return status_lock

    @contextmanager
    def _status_call(self):
        """Serialize one complete collective-bearing transceiver call.

        Once admitted, a call finishes its collective sequence even if a
        concurrent shutdown closes admission. Shutdown may initiate session
        cancellation, but it cannot drain retirement ledgers until the active
        call leaves this scope.
        """
        status_lock = self._get_status_call_lock()
        status_lock.acquire()
        admitted = False
        try:
            with self._get_session_admission_lock():
                fault = getattr(self, "_retirement_fault", None)
                if fault is not None:
                    raise RuntimeError(
                        "KV cache transceiver retirement invariant failed"
                    ) from fault
                if not getattr(self, "_shutdown_started", False):
                    self._active_status_calls = getattr(self, "_active_status_calls", 0) + 1
                    admitted = True
            yield admitted
        finally:
            if admitted:
                with self._get_session_admission_lock():
                    self._active_status_calls -= 1
            status_lock.release()

    def _latch_retirement_fault(self, error: BaseException) -> None:
        """Fail stop after an asserted-infallible retirement commit fails."""
        with self._get_session_admission_lock():
            if getattr(self, "_retirement_fault", None) is None:
                self._retirement_fault = error
            self._shutdown_started = True
            _NON_DRAINED_TRANSCEIVERS.add(self)

    @staticmethod
    def _retirement_sessions(retirements: dict) -> set[int]:
        return {id(progress.session) for progress in retirements.values()}

    def _cancel_sessions_for_shutdown(self) -> set[int]:
        """Initiate cancellation without retiring any request/session owner."""
        context_retirement_sessions = self._retirement_sessions(self._get_context_retirements())
        generation_retirement_sessions = self._retirement_sessions(
            self._get_generation_retirements()
        )
        cancel_failed_session_ids: set[int] = set()
        for direction, sessions in (
            ("send", self._send_sessions),
            ("receive", self._recv_sessions),
        ):
            retirement_sessions = (
                context_retirement_sessions
                if direction == "send"
                else generation_retirement_sessions
            )
            for session in list(sessions.values()):
                if id(session) in retirement_sessions:
                    continue
                try:
                    # Cancellation closes future publication but leaves every
                    # already-exposed resource owned until terminal evidence.
                    session.cancel()
                except Exception as error:
                    cancel_failed_session_ids.add(id(session))
                    logger.error(
                        f"KvCacheTransceiverV2 shutdown failed to cancel {direction} "
                        f"session {session.disagg_request_id}: {error}"
                    )
        return cancel_failed_session_ids

    def shutdown(self) -> bool:
        admission_lock = self._get_session_admission_lock()
        with admission_lock:
            if getattr(self, "_shutdown", False):
                return True
            # Retain the ownership root before any fallible cancellation or
            # cleanup. A failed shutdown must remain retryable even when the
            # caller releases its last reference after this attempt.
            _NON_DRAINED_TRANSCEIVERS.add(self)
            # Close admission before taking any session snapshot. The same lock
            # covers owner enrollment through publication/worker enqueue, so a
            # launch cannot resume against a worker that shutdown just destroyed.
            self._shutdown_started = True
            cancel_failed_session_ids = self._cancel_sessions_for_shutdown()
            # A status call may be blocked waiting for receive resources to
            # drain. Cancellation above lets it make progress; waiting here
            # would deadlock shutdown behind the very call it must unblock.
            # Keep every owner/ledger intact and let a later shutdown retry do
            # the local drain after the collective stream is idle.
            if getattr(self, "_active_status_calls", 0) > 0:
                return False
            # A prior status poll may already have crossed distributed
            # consensus, or may have prepared a local terminal candidate.
            # With no active status call, shutdown can locally finish those
            # exact owners without entering another distributed collective.
            self._drain_context_retirements()
            self._drain_generation_retirements()
            try:
                worker_drained = self._transfer_worker.shutdown()
            except Exception as e:
                logger.error(
                    "KvCacheTransceiverV2 worker shutdown did not drain; retaining "
                    f"sessions and request resources for retry: {e}"
                )
                return False
            if not worker_drained:
                return False

            # Worker drain can make a previously failing session close
            # retryable. Advance the post-consensus ledgers again, but retain
            # any entry whose local commit is still incomplete.
            self._drain_context_retirements()
            self._drain_generation_retirements()
            context_retirement_sessions = {
                id(progress.session) for progress in self._get_context_retirements().values()
            }
            generation_retirement_sessions = {
                id(progress.session) for progress in self._get_generation_retirements().values()
            }

            close_failed = False
            for direction, sessions, requests in (
                ("send", self._send_sessions, self._send_reqs),
                ("receive", self._recv_sessions, self._recv_reqs),
            ):
                for rid, session in list(sessions.items()):
                    if id(session) in cancel_failed_session_ids:
                        continue
                    retirement_sessions = (
                        context_retirement_sessions
                        if direction == "send"
                        else generation_retirement_sessions
                    )
                    if id(session) in retirement_sessions:
                        continue
                    try:
                        self._close_session_with_allocation_leases(
                            session,
                            PhysicalDisposition.QUIESCED_FAILURE,
                        )
                    except Exception as e:
                        close_failed = True
                        logger.error(
                            f"KvCacheTransceiverV2 shutdown failed to close {direction} "
                            f"session {session.disagg_request_id}: {e}"
                        )
                        continue
                    if sessions.get(rid) is session:
                        sessions.pop(rid, None)
                    requests.pop(rid, None)

            if (
                cancel_failed_session_ids
                or close_failed
                or self._get_context_retirements()
                or self._get_generation_retirements()
                or self._get_session_allocation_leases()
                or self._get_retained_allocation_leases()
            ):
                return False
            getattr(self, "_wait_reqs", {}).clear()
            self._get_cancelled_wait_reqs().clear()
            self._shutdown = True
            _NON_DRAINED_TRANSCEIVERS.discard(self)
            return True

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        try:
            drained = self.shutdown()
        except Exception as shutdown_error:
            if _exc_type is None:
                raise
            logger.error(
                "KvCacheTransceiverV2 shutdown also failed while propagating "
                f"{_exc_type.__name__}: {shutdown_error}"
            )
            return False
        if not drained:
            if _exc_type is None:
                raise RuntimeError("KV cache transceiver shutdown did not drain")
            logger.error(
                "KV cache transceiver shutdown did not drain while propagating "
                f"{_exc_type.__name__}"
            )
        return False

    def _create_kv_slice(self, req: LlmRequest) -> KVSlice:
        allocation_lease = AllocationLease.acquire(
            self._kv_cache_manager,
            req.py_request_id,
        )
        try:
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
            copied_allocation_groups = []
            mamba_group_indices = set()
            group_window_sizes = []
            for idx, lg in enumerate(layer_groups):
                if isinstance(lg, MambaLayerGroup):
                    mamba_group_indices.add(idx)
                    group_window_sizes.append(None)
                    empty_group = np.array([], dtype=np.int64)
                    copied_allocation_groups.append(empty_group)
                    groups.append(empty_group)
                    continue
                block_ids = np.array(adapter.get_block_ids(req, idx, lg), copy=True)
                copied_allocation_groups.append(block_ids)
                group_window_sizes.append(lg.sliding_window_size)
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
            if isinstance(self._kv_cache_manager, MambaHybridCacheManagerV2):
                if self._kv_cache_manager.local_num_mamba_layers > 0:
                    mamba_state_index = self._kv_cache_manager._request_id_to_state_index[
                        req.py_request_id
                    ]
            elif isinstance(self._kv_cache_manager, MambaHybridCacheManager):
                mamba_state_index = self._kv_cache_manager.mamba_cache_index[req.py_request_id]

            allocation_lease.validate_copied_descriptors(
                copied_allocation_groups,
                group_window_sizes,
                frozenset(mamba_group_indices),
                mamba_state_index,
            )
            return KVSlice(
                is_last_slice=True,
                block_ids_per_layer_groups=groups,
                mamba_state_index=mamba_state_index,
                token_range=token_range,
                allocation_lease=allocation_lease,
            )
        except BaseException:
            self._settle_allocation_lease(
                allocation_lease,
                PhysicalDisposition.NOT_EXPOSED,
            )
            raise

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

    def _validate_generation_safe_request_identity(
        self,
        req: LlmRequest,
        operation: str,
    ) -> None:
        """Require exact cross-side identity before protocol-v1 publication."""
        lifecycle_contract = getattr(self, "_lifecycle_contract", None)
        if (
            lifecycle_contract is None
            or lifecycle_contract.protocol_version != ProtocolVersion.GENERATION_SAFE
        ):
            return
        if getattr(req, "py_disagg_transfer_protocol_identity", None) is None:
            raise RuntimeError(f"protocol-v1 {operation} requires exact transfer identity")

    def _validate_receive_publication_contract(self, req: LlmRequest) -> None:
        """Reject unsupported writer selection before creating a receive session."""
        if self.capabilities().protocol_version != ProtocolVersion.GENERATION_SAFE:
            return
        params = req.py_disaggregated_params
        if (
            params is not None
            and params.schedule_style == DisaggScheduleStyle.GENERATION_FIRST
            and self._mapping.enable_attention_dp
            and params.ctx_dp_rank is None
        ):
            raise RuntimeError(
                "protocol-v1 generation-first attention-DP requires an exact "
                "context writer rank before destination publication"
            )

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
        all_ranks = self._dist.tp_allgather(local_ids) if self._ctx_need_tp_sync else [local_ids]
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
            pp_all_ranks = getattr(self._dist, "pp_allgather")(ready_ids)
            ready_ids = _find_consensus_request_ids(pp_all_ranks, self._mapping.pp_size)

        return ready_ids

    def _gen_consensus(self, local_ids: list) -> list:
        sync_size = (
            self._mapping.pp_size if self._mapping.enable_attention_dp else self._mapping.world_size
        )
        all_ranks = self._gen_allgather(local_ids) if self._gen_need_sync else [local_ids]
        return _find_consensus_request_ids(all_ranks, sync_size)

    @staticmethod
    def _allgather_or_passthrough(local_value: list, allgather: Callable, need_sync: bool) -> list:
        if not need_sync:
            return [list(local_value)]
        return list(allgather(list(local_value)))

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
        cancelled,
        failed,
        completed,
        cleanup_ready,
        allgather: Callable,
        need_sync: bool,
    ):
        """Agree on logical outcome without outrunning local physical drain.

        Cancellation/failure is a union because one rank's logical failure must
        fail the distributed request. Cleanup readiness is an intersection:
        every rank must independently prove its local source/target accessors
        terminal before any rank removes its request owner.
        """
        payload = [
            list(cancelled),
            list(failed),
            list(completed),
            list(cleanup_ready),
        ]
        gathered = self._allgather_or_passthrough(payload, allgather, need_sync)
        all_c = [rank_payload[0] for rank_payload in gathered]
        all_f = [rank_payload[1] for rank_payload in gathered]
        all_done = [rank_payload[2] for rank_payload in gathered]
        all_ready = [rank_payload[3] for rank_payload in gathered]
        n = len(gathered)
        global_cancelled = self._union(all_c)
        global_failed = self._union(all_f)
        global_completed = self._intersection(all_done, n)
        globally_ready = self._intersection(all_ready, n)
        new_cancelled = [
            rid for rid in to_process if rid in global_cancelled and rid in globally_ready
        ]
        cancel_set = set(new_cancelled)
        new_failed = [
            rid
            for rid in to_process
            if rid in global_failed and rid in globally_ready and rid not in cancel_set
        ]
        terminal = cancel_set | set(new_failed)
        new_completed = [
            rid
            for rid in to_process
            if rid in global_completed and rid in globally_ready and rid not in terminal
        ]
        return new_cancelled, new_failed, new_completed

    def _gen_consensus_outcome(self, to_process, cancelled, failed, completed, cleanup_ready):
        return self._consensus_outcome(
            to_process,
            cancelled,
            failed,
            completed,
            cleanup_ready,
            self._gen_allgather,
            self._gen_need_sync,
        )

    def _ctx_consensus_outcome(
        self, to_process, cancelled, failed, completed, timed_out, cleanup_ready
    ):
        # TP first, then PP.  timed_out is local-only (back-off signal).
        c, f, d = self._consensus_outcome(
            to_process,
            cancelled,
            failed,
            completed,
            cleanup_ready,
            self._dist.tp_allgather,
            self._ctx_need_tp_sync,
        )
        if self._ctx_need_pp_sync:
            pp_allgather: Callable = getattr(self._dist, "pp_allgather")
            # A TP group appears ready to the PP stage only after the first
            # consensus admitted one terminal outcome for that request.
            c, f, d = self._consensus_outcome(
                to_process,
                c,
                f,
                d,
                c + f + d,
                pp_allgather,
                True,
            )
        return c, f, d, timed_out

    def _sync_transfer_timing(self, reqs: list):
        """Allgather timing for a batch of completed requests in one collective.

        Matches C++ ``batchUpdateKVCacheTransferBW()`` in ``cacheTransceiver.cpp``.
        Only runs when ``TRTLLM_KVCACHE_TIME_OUTPUT_PATH`` is set (same gate
        as C++) and multi-rank sync is needed.  All ranks that participate in
        the allgather update their local request objects.
        """
        if not reqs:
            return
        if not os.getenv("TRTLLM_KVCACHE_TIME_OUTPUT_PATH"):
            return
        if not self._gen_need_sync:
            return

        # Pack local timing for all completed requests into one dict.
        local_data = {
            get_unique_rid(req): (
                req.get_kv_cache_transfer_start(),
                req.get_kv_cache_transfer_end(),
                req.kv_cache_size,
            )
            for req in reqs
        }

        # Single allgather for the whole batch.
        all_data = self._gen_allgather(local_data)

        # Merge: per-rid min(start), max(end), sum(size) across ranks.
        merged: dict = {}
        for rank_data in all_data:
            for rid, (start, end, size) in rank_data.items():
                if rid in merged:
                    prev = merged[rid]
                    merged[rid] = (
                        min(prev[0], start),
                        max(prev[1], end),
                        prev[2] + size,
                    )
                else:
                    merged[rid] = (start, end, size)

        # Every rank updates its own local requests.
        rid_to_req = {get_unique_rid(r): r for r in reqs}
        for rid, (min_start, max_end, total_size) in merged.items():
            req = rid_to_req.get(rid)
            if req is not None:
                req.set_kv_cache_transfer_start(min_start)
                req.set_kv_cache_transfer_end(max_end)
                req.set_kv_cache_size(total_size)

    def _record_generation_completion_metrics(self, completed_rids: list[int]) -> None:
        """Best-effort timing synchronization before retirement drops request owners.

        Every rank calls this with the post-consensus completed set. Metrics
        failures are deliberately isolated from ownership retirement.
        """
        if not completed_rids or not os.getenv("TRTLLM_KVCACHE_TIME_OUTPUT_PATH"):
            return
        with self._get_session_admission_lock():
            retirements = self._get_generation_retirements()
            requests = [
                progress.request
                for rid in completed_rids
                if (progress := retirements.get(rid)) is not None and progress.report_result
            ]

        try:
            self._sync_transfer_timing(requests)
        except Exception as error:
            logger.warning(
                "KV transfer timing synchronization failed; continuing physical "
                f"retirement: {error}"
            )

        for request in requests:
            rid = get_unique_rid(request)
            if rid is None:
                continue
            try:
                perf_log_manager.log_gen_transfer_summary(
                    unique_rid=rid,
                    instance_name=self._instance_name,
                    instance_rank=self._mapping.rank,
                    gen_side_transfer_time_ms=request.kv_cache_transfer_time_ms,
                    kv_cache_size=request.kv_cache_size,
                )
            except Exception as error:
                logger.warning(
                    f"KV transfer summary logging failed for request {rid}; "
                    f"continuing physical retirement: {error}"
                )

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
            return list(dict.fromkeys([*consensus, *sessions.keys()]))
        to_process = list(consensus)
        for rid in sessions:
            if len(to_process) >= wait_num:
                break
            if rid not in to_process:
                to_process.append(rid)
        return to_process

    def _get_context_retirements(self) -> Dict[int, _ContextRetirementProgress]:
        """Return the context retirement ledger, including object.__new__ fixtures."""

        retirements = getattr(self, "_context_retirements", None)
        if retirements is None:
            retirements = {}
            self._context_retirements = retirements
        return retirements

    def _get_generation_retirements(self) -> Dict[int, _GenerationRetirementProgress]:
        """Return the generation retirement ledger, including object.__new__ fixtures."""

        retirements = getattr(self, "_generation_retirements", None)
        if retirements is None:
            retirements = {}
            self._generation_retirements = retirements
        return retirements

    def _get_cancelled_wait_reqs(self) -> Dict[int, LlmRequest]:
        """Return wait-owner cancellation markers, including test fixtures."""
        cancelled = getattr(self, "_cancelled_wait_reqs", None)
        if cancelled is None:
            cancelled = {}
            self._cancelled_wait_reqs = cancelled
        return cancelled

    def _get_session_allocation_leases(self) -> dict[object, list[AllocationLease]]:
        """Return exact leases by session, including object.__new__ fixtures."""
        leases = getattr(self, "_allocation_leases_by_session", None)
        if leases is None:
            leases = {}
            self._allocation_leases_by_session = leases
        return leases

    def _get_retained_allocation_leases(self) -> list[AllocationLease]:
        """Return fail-closed allocator quarantines, including test fixtures."""
        retained = getattr(self, "_retained_allocation_leases", None)
        if retained is None:
            retained = []
            self._retained_allocation_leases = retained
        return retained

    def _retain_allocation_lease(self, lease: AllocationLease, reason: str) -> None:
        retained = self._get_retained_allocation_leases()
        if lease not in retained:
            retained.append(lease)
        _NON_DRAINED_TRANSCEIVERS.add(self)
        logger.error(f"Retaining allocation lease {lease.lease_id} as non-reusable: {reason}")

    def _settle_allocation_lease(
        self,
        lease: AllocationLease,
        disposition: PhysicalDisposition,
    ) -> bool:
        try:
            released = lease.settle(disposition)
        except Exception as error:
            retirement_error = RuntimeError(
                f"allocator settlement raised for {disposition.value}: {error}"
            )
            self._retain_allocation_lease(
                lease,
                str(retirement_error),
            )
            self._latch_retirement_fault(retirement_error)
            raise retirement_error from error
        if not released:
            retirement_error = RuntimeError(
                f"allocator rejected {disposition.value} proof with {lease.last_settlement}"
            )
            self._retain_allocation_lease(
                lease,
                str(retirement_error),
            )
            self._latch_retirement_fault(retirement_error)
            raise retirement_error
        return True

    def _bind_slice_allocation_lease(self, session: object, kv_slice: KVSlice) -> None:
        """Make the session the exact lease owner before dispatch can publish."""
        lease = kv_slice.allocation_lease
        if not isinstance(lease, AllocationLease):
            raise RuntimeError("KV slice does not carry an allocator-generation lease")
        leases = self._get_session_allocation_leases().setdefault(session, [])
        if lease in leases:
            raise RuntimeError(
                f"allocation lease {lease.lease_id} is already bound to this session"
            )
        leases.append(lease)

    def _close_session_with_allocation_leases(
        self,
        session: object,
        disposition: PhysicalDisposition,
    ) -> None:
        """Close physical access first, then settle every exact session lease."""
        if not disposition.is_reusable:
            raise ValueError("session close requires reusable physical disposition")
        session.close()
        leases_by_session = self._get_session_allocation_leases()
        leases = leases_by_session.get(session, [])
        for lease in tuple(leases):
            self._settle_allocation_lease(lease, disposition)
            leases.remove(lease)
        if not leases:
            leases_by_session.pop(session, None)

    @staticmethod
    def _validate_exact_map_entry(mapping: dict, rid: int, expected, label: str) -> None:
        """Validate the commit owner before advertising cleanup readiness."""
        if mapping.get(rid) is not expected:
            raise RuntimeError(f"{label} {rid} changed identity before retirement commit")

    @staticmethod
    def _retire_exact_map_entry(mapping: dict, rid: int, expected, label: str) -> None:
        """Idempotently remove one exact owner during local shutdown retry."""

        current = mapping.get(rid)
        if current is not None and current is not expected:
            raise RuntimeError(f"{label} {rid} changed identity during retirement")
        if current is expected:
            mapping.pop(rid, None)

    def _enroll_context_retirements(
        self,
        outcomes: tuple[tuple[str, list], ...],
        sessions: dict,
        reqs: dict,
        *,
        mark_complete: bool,
    ) -> None:
        """Persist exact owners before fallible context-side preparation."""

        retirements = self._get_context_retirements()
        with self._get_session_admission_lock():
            for outcome, rids in outcomes:
                for rid in rids:
                    session = sessions.get(rid)
                    req = reqs.get(rid)
                    if session is None or req is None:
                        continue
                    existing = retirements.get(rid)
                    if existing is not None:
                        if (
                            existing.session is not session
                            or existing.request is not req
                            or existing.local_outcome != outcome
                        ):
                            raise RuntimeError(
                                f"Conflicting local context terminal candidate for request {rid}"
                            )
                        existing.mark_complete = existing.mark_complete or mark_complete
                        continue
                    progress = _ContextRetirementProgress(
                        rid=rid,
                        session=session,
                        request=req,
                        local_outcome=outcome,
                        mark_complete=mark_complete,
                    )
                    # Cancellation may retire this snapshot while the status
                    # call is outside the admission lock. Keep a no-op
                    # provisional record for collective symmetry, but never
                    # mutate a replacement owner that reused the integer RID.
                    exact_owner = (
                        self._send_reqs.get(rid) is req and self._send_sessions.get(rid) is session
                    )
                    if not exact_owner:
                        progress.report_result = False
                        progress.session_closed = True
                        progress.maps_validated = True
                        progress.state_updated = True
                        progress.request_map_retired = True
                        progress.session_map_retired = True
                    retirements[rid] = progress

    def _prepare_context_retirement(self, progress: _ContextRetirementProgress) -> None:
        """Complete fallible local cleanup before the distributed commit."""

        if not progress.session_closed:
            disposition = (
                PhysicalDisposition.QUIESCED_SUCCESS
                if progress.local_outcome == "completed"
                else PhysicalDisposition.QUIESCED_FAILURE
            )
            self._close_session_with_allocation_leases(
                progress.session,
                disposition,
            )
            progress.session_closed = True
        if not progress.maps_validated:
            self._validate_exact_map_entry(
                self._send_reqs, progress.rid, progress.request, "send request"
            )
            self._validate_exact_map_entry(
                self._send_sessions, progress.rid, progress.session, "send session"
            )
            progress.maps_validated = True

    @staticmethod
    def _context_retirement_prepared(progress: _ContextRetirementProgress) -> bool:
        return progress.session_closed and progress.maps_validated

    def _prepare_context_retirements(self) -> None:
        """Retry every provisional context cleanup without adding collectives."""
        with self._get_session_admission_lock():
            for rid, progress in list(self._get_context_retirements().items()):
                try:
                    self._prepare_context_retirement(progress)
                except Exception as error:
                    logger.error(
                        "Context transfer cleanup failed locally for request "
                        f"{rid}; retaining exact phase progress for retry: {error}"
                    )

    def _context_candidate_payload(
        self, to_process: list[int]
    ) -> tuple[list[int], list[int], list[int], list[int]]:
        cancelled: list[int] = []
        failed: list[int] = []
        completed: list[int] = []
        cleanup_ready: list[int] = []
        with self._get_session_admission_lock():
            retirements = self._get_context_retirements()
            for rid in to_process:
                progress = retirements.get(rid)
                if progress is None:
                    continue
                if progress.local_outcome == "cancelled":
                    cancelled.append(rid)
                elif progress.local_outcome == "failed":
                    failed.append(rid)
                else:
                    completed.append(rid)
                if self._context_retirement_prepared(progress):
                    cleanup_ready.append(rid)
        return cancelled, failed, completed, cleanup_ready

    def _commit_context_retirement(self, progress: _ContextRetirementProgress) -> None:
        """Publish one consensus decision using asserted-infallible operations."""

        if not progress.state_updated:
            outcome = progress.outcome or progress.local_outcome
            target_state = None
            if outcome == "completed":
                if progress.mark_complete:
                    target_state = LlmRequestState.DISAGG_CONTEXT_COMPLETE
            else:
                target_state = LlmRequestState.DISAGG_TRANS_ERROR
            if target_state is not None and progress.request.state != target_state:
                progress.request.state = target_state
            progress.state_updated = True

        if not progress.request_map_retired:
            self._retire_exact_map_entry(
                self._send_reqs, progress.rid, progress.request, "send request"
            )
            progress.request_map_retired = True
        if not progress.session_map_retired:
            self._retire_exact_map_entry(
                self._send_sessions, progress.rid, progress.session, "send session"
            )
            progress.session_map_retired = True

    @staticmethod
    def _context_retirement_committed(progress: _ContextRetirementProgress) -> bool:
        return (
            progress.state_updated and progress.request_map_retired and progress.session_map_retired
        )

    def _commit_context_decisions(
        self,
        outcomes: tuple[tuple[str, list[int]], ...],
    ) -> tuple[list[int], list[int]]:
        """Commit the outcome-consensus result without another sync round."""
        completed: list[int] = []
        failed: list[int] = []
        try:
            with self._get_session_admission_lock():
                retirements = self._get_context_retirements()
                for outcome, rids in outcomes:
                    for rid in rids:
                        progress = retirements.get(rid)
                        if progress is None or not self._context_retirement_prepared(progress):
                            raise RuntimeError(
                                f"Context outcome consensus admitted an unprepared request {rid}"
                            )
                        if progress.outcome is not None and progress.outcome != outcome:
                            raise RuntimeError(f"Context request {rid} changed global outcome")
                        progress.outcome = outcome
                        self._commit_context_retirement(progress)
                        if not self._context_retirement_committed(progress):
                            raise RuntimeError(f"Context request {rid} commit was incomplete")
                        retirements.pop(rid)
                        if not progress.report_result:
                            continue
                        if outcome == "completed":
                            completed.append(rid)
                        else:
                            failed.append(rid)
        except Exception as error:
            self._latch_retirement_fault(error)
            raise
        return completed, failed

    def _drain_context_retirements(self) -> tuple[list[int], list[int]]:
        """Locally force decided context retirement during shutdown."""

        completed: list[int] = []
        failed: list[int] = []
        retirements = self._get_context_retirements()
        with self._get_session_admission_lock():
            for rid, progress in list(retirements.items()):
                if retirements.get(rid) is not progress:
                    continue
                try:
                    self._prepare_context_retirement(progress)
                    if progress.outcome is None:
                        progress.outcome = progress.local_outcome
                    self._commit_context_retirement(progress)
                except Exception as error:
                    logger.error(
                        "Context transfer retirement failed locally for request "
                        f"{rid}; retaining exact phase progress for retry: {error}"
                    )
                    continue
                if retirements.get(rid) is progress:
                    retirements.pop(rid, None)
                if not progress.report_result:
                    continue
                if (progress.outcome or progress.local_outcome) == "completed":
                    completed.append(rid)
                else:
                    failed.append(rid)
        return completed, failed

    def _enroll_generation_retirements(
        self,
        outcomes: tuple[tuple[str, list], ...],
        sessions: dict,
        reqs: dict,
    ) -> None:
        """Persist exact owners before fallible generation-side preparation."""

        retirements = self._get_generation_retirements()
        with self._get_session_admission_lock():
            for outcome, rids in outcomes:
                for rid in rids:
                    session = sessions.get(rid)
                    req = reqs.get(rid)
                    if session is None or req is None:
                        continue
                    existing = retirements.get(rid)
                    if existing is not None:
                        if (
                            existing.session is not session
                            or existing.request is not req
                            or existing.local_outcome != outcome
                        ):
                            raise RuntimeError(
                                f"Conflicting local generation terminal candidate for request {rid}"
                            )
                        continue
                    progress = _GenerationRetirementProgress(
                        rid=rid,
                        session=session,
                        request=req,
                        local_outcome=outcome,
                    )
                    exact_owner = (
                        self._recv_reqs.get(rid) is req and self._recv_sessions.get(rid) is session
                    )
                    if not exact_owner:
                        progress.report_result = False
                        progress.kv_size_set = True
                        progress.aux_applied = True
                        progress.history_checked = True
                        progress.session_closed = True
                        progress.maps_validated = True
                        progress.state_updated = True
                        progress.request_map_retired = True
                        progress.session_map_retired = True
                    retirements[rid] = progress

    def _prepare_generation_retirement(self, progress: _GenerationRetirementProgress) -> None:
        """Complete fallible local generation cleanup before commit."""

        if progress.local_outcome == "completed":
            if not progress.kv_size_set:
                transfer_start = getattr(progress.session, "transfer_start_time", None)
                transfer_end = getattr(progress.session, "transfer_end_time", None)
                if transfer_start is not None:
                    progress.request.set_kv_cache_transfer_start(transfer_start)
                if transfer_end is not None:
                    progress.request.set_kv_cache_transfer_end(transfer_end)
                measured_size = getattr(progress.session, "kv_cache_size_bytes", 0)
                progress.request.set_kv_cache_size(
                    measured_size
                    if measured_size > 0
                    else getattr(progress.request, "py_kv_cache_xfer_bytes", 0)
                )
                progress.kv_size_set = True
            if not progress.aux_applied:
                if self._need_aux_transfer(progress.request):
                    self._apply_aux(progress.session, progress.request)
                progress.aux_applied = True
            if not progress.history_checked:
                self._assert_disagg_history_declared(progress.request)
                progress.history_checked = True
        else:
            # These phases are not part of cancelled/failed retirement.
            progress.kv_size_set = True
            progress.aux_applied = True
            progress.history_checked = True

        if not progress.session_closed:
            disposition = (
                PhysicalDisposition.QUIESCED_SUCCESS
                if progress.local_outcome == "completed"
                else PhysicalDisposition.QUIESCED_FAILURE
            )
            self._close_session_with_allocation_leases(
                progress.session,
                disposition,
            )
            progress.session_closed = True
        if not progress.maps_validated:
            self._validate_exact_map_entry(
                self._recv_reqs, progress.rid, progress.request, "receive request"
            )
            self._validate_exact_map_entry(
                self._recv_sessions, progress.rid, progress.session, "receive session"
            )
            progress.maps_validated = True

    def _commit_generation_retirement(self, progress: _GenerationRetirementProgress) -> None:
        """Publish one consensus decision using asserted-infallible operations."""

        if not progress.state_updated:
            outcome = progress.outcome or progress.local_outcome
            target_state = None
            if outcome == "completed":
                target_state = LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE
            elif outcome == "failed":
                target_state = LlmRequestState.DISAGG_TRANS_ERROR
            if target_state is not None and progress.request.state != target_state:
                progress.request.state = target_state
            progress.state_updated = True

        if not progress.request_map_retired:
            self._retire_exact_map_entry(
                self._recv_reqs, progress.rid, progress.request, "receive request"
            )
            progress.request_map_retired = True
        if not progress.session_map_retired:
            self._retire_exact_map_entry(
                self._recv_sessions, progress.rid, progress.session, "receive session"
            )
            progress.session_map_retired = True

    @staticmethod
    def _generation_retirement_prepared(progress: _GenerationRetirementProgress) -> bool:
        return (
            progress.kv_size_set
            and progress.aux_applied
            and progress.history_checked
            and progress.session_closed
            and progress.maps_validated
        )

    @staticmethod
    def _generation_retirement_committed(progress: _GenerationRetirementProgress) -> bool:
        return (
            progress.state_updated and progress.request_map_retired and progress.session_map_retired
        )

    def _prepare_generation_retirements(self) -> None:
        """Retry every provisional generation cleanup without extra collectives."""
        with self._get_session_admission_lock():
            for rid, progress in list(self._get_generation_retirements().items()):
                try:
                    self._prepare_generation_retirement(progress)
                except Exception as error:
                    logger.error(
                        "Generation transfer cleanup failed locally for request "
                        f"{rid}; retaining exact phase progress for retry: {error}"
                    )

    def _generation_candidate_payload(
        self, to_process: list[int]
    ) -> tuple[list[int], list[int], list[int], list[int]]:
        cancelled: list[int] = []
        failed: list[int] = []
        completed: list[int] = []
        cleanup_ready: list[int] = []
        with self._get_session_admission_lock():
            retirements = self._get_generation_retirements()
            for rid in to_process:
                progress = retirements.get(rid)
                if progress is None:
                    continue
                if progress.local_outcome == "cancelled":
                    cancelled.append(rid)
                elif progress.local_outcome == "failed":
                    failed.append(rid)
                else:
                    completed.append(rid)
                if self._generation_retirement_prepared(progress):
                    cleanup_ready.append(rid)
        return cancelled, failed, completed, cleanup_ready

    def _commit_generation_decisions(
        self,
        outcomes: tuple[tuple[str, list[int]], ...],
    ) -> tuple[list[int], list[int], list[LlmRequest]]:
        """Commit the outcome-consensus result without another sync round."""
        completed: list[int] = []
        failed: list[int] = []
        cancelled: list[LlmRequest] = []
        pending_handoff_events: list[tuple[int, Optional[int], HandoffLifecycleEvent]] = []
        try:
            with self._get_session_admission_lock():
                retirements = self._get_generation_retirements()
                for outcome, rids in outcomes:
                    for rid in rids:
                        progress = retirements.get(rid)
                        if progress is None or not self._generation_retirement_prepared(progress):
                            raise RuntimeError(
                                f"Generation outcome consensus admitted an unprepared request {rid}"
                            )
                        if progress.outcome is not None and progress.outcome != outcome:
                            raise RuntimeError(f"Generation request {rid} changed global outcome")
                        progress.outcome = outcome
                        handoff_event = (
                            self._make_handoff_lifecycle_event(progress, outcome)
                            if progress.report_result
                            else None
                        )
                        self._commit_generation_retirement(progress)
                        if not self._generation_retirement_committed(progress):
                            raise RuntimeError(f"Generation request {rid} commit was incomplete")
                        if handoff_event is not None and self._dist.rank == 0:
                            request_id = (
                                progress.request.py_request_id
                                if not progress.request.is_child
                                else progress.request.parent_request_id
                            )
                            pending_handoff_events.append(
                                (
                                    request_id,
                                    progress.request.py_client_id,
                                    handoff_event,
                                )
                            )
                        retirements.pop(rid)
                        if not progress.report_result:
                            continue
                        if outcome == "completed":
                            completed.append(rid)
                        elif outcome == "failed":
                            failed.append(rid)
                        else:
                            cancelled.append(progress.request)
                self._handoff_lifecycle_events.extend(pending_handoff_events)
        except Exception as error:
            self._latch_retirement_fault(error)
            raise
        return completed, failed, cancelled

    @staticmethod
    def _make_handoff_lifecycle_event(
        progress: _GenerationRetirementProgress,
        outcome: str,
    ) -> Optional[HandoffLifecycleEvent]:
        """Build exact post-consensus handoff evidence before state mutation."""
        params = getattr(progress.request, "py_disaggregated_params", None)
        if (
            params is None
            or getattr(progress.request, "py_disagg_transfer_protocol_identity", None) is None
        ):
            return None
        if outcome == "completed":
            state = HandoffEventState.HANDOFF_COMMITTED
            reason = ""
        elif outcome == "failed":
            state = HandoffEventState.HANDOFF_FAILED
            reason = str(progress.session.exception or "KV handoff failed")
        elif outcome == "cancelled":
            state = HandoffEventState.HANDOFF_ABORTED
            reason = "generation request cancelled"
        else:
            raise ValueError(f"unknown generation handoff outcome {outcome!r}")
        return HandoffLifecycleEvent.from_params(
            params,
            state,
            reason=reason,
        )

    def take_handoff_lifecycle_events(
        self,
    ) -> list[tuple[int, Optional[int], HandoffLifecycleEvent]]:
        """Atomically consume executor-facing handoff notifications."""
        with self._get_session_admission_lock():
            events = self._handoff_lifecycle_events
            self._handoff_lifecycle_events = []
            return events

    def _drain_generation_retirements(
        self,
    ) -> tuple[list[int], list[int], list[LlmRequest]]:
        """Locally force decided generation retirement during shutdown."""

        completed: list[int] = []
        failed: list[int] = []
        cancelled: list[LlmRequest] = []
        retirements = self._get_generation_retirements()
        with self._get_session_admission_lock():
            for rid, progress in list(retirements.items()):
                if retirements.get(rid) is not progress:
                    continue
                try:
                    self._prepare_generation_retirement(progress)
                    if progress.outcome is None:
                        progress.outcome = progress.local_outcome
                    self._commit_generation_retirement(progress)
                except Exception as error:
                    logger.error(
                        "Generation transfer retirement failed locally for request "
                        f"{rid}; retaining exact phase progress for retry: {error}"
                    )
                    continue
                if retirements.get(rid) is progress:
                    retirements.pop(rid, None)
                if not progress.report_result:
                    continue
                outcome = progress.outcome or progress.local_outcome
                if outcome == "completed":
                    completed.append(rid)
                elif outcome == "failed":
                    failed.append(rid)
                else:
                    cancelled.append(progress.request)
        return completed, failed, cancelled

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
        if rid not in self._send_sessions:
            self._send_sessions[rid] = self._transfer_worker.create_tx_session(req)
        return self._send_sessions[rid]

    def _finalize_send(self, req: LlmRequest, session: TxSessionBase):
        """Pack aux and set context phase params. Call after the last slice."""
        rid = get_unique_rid(req)
        assert rid is not None
        if self._need_aux_transfer(req):
            session.pack_aux(req)
            session.send_aux()
        req.context_phase_params = ContextPhaseParams(
            first_gen_tokens=[],
            req_id=rid,
            opaque_state=None,
            draft_tokens=None,
            ctx_dp_rank=self._dp_rank,
            disagg_info_endpoint=self._context_info_endpoint,
        )
        self._send_reqs[rid] = req

    @nvtx_range("KvCacheTransceiverV2.respond_and_send_async")
    def respond_and_send_async(self, req: LlmRequest):
        with self._get_session_admission_lock():
            if getattr(self, "_shutdown_started", False):
                raise RuntimeError("KV cache transceiver is shutting down")
            self._validate_generation_safe_request_identity(
                req,
                "source send",
            )
            rid = get_unique_rid(req)
            assert rid is not None
            wait_owner = getattr(self, "_wait_reqs", {}).get(rid)
            if wait_owner is not None:
                if wait_owner is not req:
                    raise RuntimeError(
                        f"waiting request {rid} already has a different live source owner"
                    )
                raise RuntimeError(f"send request {rid} is still waiting for scheduler promotion")
            cancelled_wait_owner = self._get_cancelled_wait_reqs().get(rid)
            if cancelled_wait_owner is not None:
                raise RuntimeError(f"send request {rid} has a pending wait-owner cancellation")
            if rid in self._get_context_retirements():
                raise RuntimeError(f"send request {rid} has a pending terminal retirement")
            existing_owner = self._send_reqs.get(rid)
            if existing_owner is not None and existing_owner is not req:
                raise RuntimeError(f"send request {rid} already has a different live source owner")
            owner_newly_enrolled = existing_owner is None
            self._ever_had_send_session = True
            if getattr(req, "py_disagg_transfer_protocol_identity", None) is None:
                req.set_kv_cache_transfer_start(tensorrt_llm.bindings.global_steady_clock_now())
            # Enroll the source owner before a worker can launch gather/NIXL.
            self._send_reqs[rid] = req
            session = None
            try:
                session = self._get_or_create_send_session(req)
                req.state = LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS
                kv_slice = self._create_kv_slice(req)
                self._bind_slice_allocation_lease(session, kv_slice)
                session.send(kv_slice)
                self._finalize_send(req, session)
            except Exception:
                if session is None:
                    if owner_newly_enrolled:
                        self._send_reqs.pop(rid, None)
                else:
                    try:
                        session.cancel()
                        if not session.has_transferring_tasks():
                            self._close_session_with_allocation_leases(
                                session,
                                PhysicalDisposition.QUIESCED_FAILURE,
                            )
                            self._send_sessions.pop(rid, None)
                            self._send_reqs.pop(rid, None)
                    except Exception as cleanup_error:
                        logger.warning(
                            f"respond_and_send_async: retaining rid={rid} after "
                            f"exception-path cleanup could not prove drain: {cleanup_error}"
                        )
                raise

    @nvtx_range("KvCacheTransceiverV2.request_and_receive_sync")
    def request_and_receive_sync(self, req: LlmRequest):
        admission_lock = self._get_session_admission_lock()
        with admission_lock:
            if getattr(self, "_shutdown_started", False):
                raise RuntimeError("KV cache transceiver is shutting down")
            self._validate_generation_safe_request_identity(
                req,
                "synchronous receive",
            )
            self._validate_receive_publication_contract(req)
            self._ever_had_recv_session = True
            lifecycle_contract = getattr(self, "_lifecycle_contract", None)
            generation_safe = (
                lifecycle_contract is not None
                and lifecycle_contract.protocol_version == ProtocolVersion.GENERATION_SAFE
            )
            if getattr(req, "py_disagg_transfer_protocol_identity", None) is None:
                req.set_kv_cache_transfer_start(tensorrt_llm.bindings.global_steady_clock_now())
            admitted = self._request_and_receive_sync_admitted(req)

        if admitted is None:
            return
        session, kv_slice = admitted
        rid = get_unique_rid(req)
        assert rid is not None

        try:
            result = session.wait_complete(blocking=True)
        except Exception:
            with admission_lock:
                if self._recv_sessions.get(rid) is session and self._recv_reqs.get(rid) is req:
                    req.state = LlmRequestState.DISAGG_TRANS_ERROR
                    session.cancel()
                    if not session.has_transferring_tasks():
                        self._close_session_with_allocation_leases(
                            session,
                            PhysicalDisposition.QUIESCED_FAILURE,
                        )
                        self._recv_sessions.pop(rid, None)
                        self._recv_reqs.pop(rid, None)
            raise

        # Cancellation and shutdown use the same gate. They may have retired
        # this session while wait_complete() was blocked, so only the exact
        # enrolled owner may consume AUX data or remove the map entries.
        with admission_lock:
            owns_session = (
                self._recv_sessions.get(rid) is session and self._recv_reqs.get(rid) is req
            )
            if not owns_session:
                if result != WaitResult.COMPLETED or session.status == SessionStatus.CANCELLED:
                    req.state = LlmRequestState.DISAGG_TRANS_ERROR
                return
            if generation_safe:
                if session.status == SessionStatus.CANCELLED:
                    outcome = "cancelled"
                elif result == WaitResult.COMPLETED:
                    outcome = "completed"
                else:
                    outcome = "failed"
                self._enroll_generation_retirements(
                    ((outcome, [rid]),),
                    self._recv_sessions,
                    self._recv_reqs,
                )
                # Complete every local fallible phase now, but leave the exact
                # request/session owner in the retirement ledger. PyExecutor
                # drains the full synchronous batch before entering the
                # existing distributed cleanup-ready/outcome consensus.
                self._prepare_generation_retirements()
                return
            physical_disposition = PhysicalDisposition.QUIESCED_FAILURE
            try:
                if result == WaitResult.COMPLETED:
                    transfer_start = getattr(session, "transfer_start_time", None)
                    transfer_end = getattr(session, "transfer_end_time", None)
                    if transfer_start is not None:
                        req.set_kv_cache_transfer_start(transfer_start)
                    if transfer_end is not None:
                        req.set_kv_cache_transfer_end(transfer_end)
                    measured_size = getattr(session, "kv_cache_size_bytes", 0)
                    req.set_kv_cache_size(
                        measured_size
                        if measured_size > 0
                        else self._slice_num_bytes(kv_slice) * self._kv_size_rank_factor
                    )
                    if self._need_aux_transfer(req):
                        self._apply_aux(session, req)
                    self._assert_disagg_history_declared(req)
                    req.state = LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE
                    physical_disposition = PhysicalDisposition.QUIESCED_SUCCESS
                else:
                    req.state = LlmRequestState.DISAGG_TRANS_ERROR
            except Exception:
                req.state = LlmRequestState.DISAGG_TRANS_ERROR
                raise
            finally:
                self._close_session_with_allocation_leases(
                    session,
                    physical_disposition,
                )
                if self._recv_sessions.get(rid) is session and self._recv_reqs.get(rid) is req:
                    self._recv_sessions.pop(rid, None)
                    self._recv_reqs.pop(rid, None)

    def _request_and_receive_sync_admitted(self, req: LlmRequest):
        rid = get_unique_rid(req)
        assert rid is not None
        if rid in self._get_generation_retirements():
            raise RuntimeError(f"receive request {rid} has a pending terminal retirement")
        if rid in self._recv_sessions:
            if self._recv_reqs.get(rid) is not req:
                raise RuntimeError(
                    f"receive request {rid} already has a different live destination owner"
                )
            logger.warning(
                f"request_and_receive_sync: rid={rid} already has a recv session, skipping"
            )
            return
        req.state = LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
        self._recv_reqs[rid] = req
        session = None
        try:
            session = self._transfer_worker.create_rx_session(req)
            self._recv_sessions[rid] = session
            kv_slice = self._create_kv_slice(req)
            self._bind_slice_allocation_lease(session, kv_slice)
            session.receive(kv_slice)
            return session, kv_slice
        except Exception:
            req.state = LlmRequestState.DISAGG_TRANS_ERROR
            if session is None:
                if self._recv_reqs.get(rid) is req:
                    self._recv_reqs.pop(rid, None)
            else:
                session.cancel()
                if not session.has_transferring_tasks():
                    self._close_session_with_allocation_leases(
                        session,
                        PhysicalDisposition.QUIESCED_FAILURE,
                    )
                    if self._recv_sessions.get(rid) is session:
                        self._recv_sessions.pop(rid, None)
                    if self._recv_reqs.get(rid) is req:
                        self._recv_reqs.pop(rid, None)
            raise

    @nvtx_range("KvCacheTransceiverV2.request_and_receive_async")
    def request_and_receive_async(self, req: LlmRequest):
        with self._get_session_admission_lock():
            if getattr(self, "_shutdown_started", False):
                raise RuntimeError("KV cache transceiver is shutting down")
            self._validate_generation_safe_request_identity(
                req,
                "asynchronous receive",
            )
            self._validate_receive_publication_contract(req)
            self._ever_had_recv_session = True
            rid = get_unique_rid(req)
            assert rid is not None
            if rid in self._get_generation_retirements():
                raise RuntimeError(f"receive request {rid} has a pending terminal retirement")
            if rid in self._recv_sessions:
                if self._recv_reqs.get(rid) is not req:
                    raise RuntimeError(
                        f"receive request {rid} already has a different live destination owner"
                    )
                logger.warning(
                    f"request_and_receive_async: rid={rid} already has a recv session, skipping"
                )
                return
            if getattr(req, "py_disagg_transfer_protocol_identity", None) is None:
                req.set_kv_cache_transfer_start(tensorrt_llm.bindings.global_steady_clock_now())
            req.state = LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
            # Enroll the request owner before session construction can allocate an
            # AUX slot and before receive() can publish KV/AUX target addresses.
            # If either step raises, keep every owner enrolled so cleanup continues
            # through the normal physical-drain gate rather than dropping the only
            # Python-level owner during exception unwinding.
            self._recv_reqs[rid] = req
            session = None
            try:
                session = self._transfer_worker.create_rx_session(req)
                self._recv_sessions[rid] = session
                kv_slice = self._create_kv_slice(req)
                self._bind_slice_allocation_lease(session, kv_slice)
                req.py_kv_cache_xfer_bytes = (
                    self._slice_num_bytes(kv_slice) * self._kv_size_rank_factor
                )
                session.receive(kv_slice)
            except Exception:
                if session is None:
                    # Constructor rollback owns any pre-publication allocation.
                    self._recv_reqs.pop(rid, None)
                else:
                    try:
                        session.cancel()
                        drained = not session.has_transferring_tasks()
                        if drained:
                            self._close_session_with_allocation_leases(
                                session,
                                PhysicalDisposition.QUIESCED_FAILURE,
                            )
                            self._recv_sessions.pop(rid, None)
                            self._recv_reqs.pop(rid, None)
                    except Exception as cleanup_error:
                        logger.warning(
                            f"request_and_receive_async: retaining rid={rid} after "
                            f"exception-path cleanup could not prove drain: {cleanup_error}"
                        )
                raise

    def check_context_transfer_status(
        self,
        at_least_request_num: Optional[int],
        mark_complete: bool = False,
    ):
        with self._status_call() as admitted:
            if not admitted:
                return [], []
            return self._check_context_transfer_status_impl(at_least_request_num, mark_complete)

    def _check_context_transfer_status_impl(
        self,
        at_least_request_num: Optional[int],
        mark_complete: bool,
    ):
        # This is also the native sender's periodic control-result progress
        # hook. Run it before the idle fast path so cancel-before-session
        # failures are retried even when no TxSession is ever created.
        self._prepare_context_retirements()
        with self._get_session_admission_lock():
            self._transfer_worker.sweep_stale_req_infos()
            pending_retirements = self._get_context_retirements()
            if not self._ever_had_send_session and not pending_retirements:
                return [], []
            if mark_complete:
                for progress in pending_retirements.values():
                    progress.mark_complete = True
            pending_candidate_rids = list(pending_retirements)
            pending_rids = set(pending_retirements)
            send_sessions = {
                rid: session
                for rid, session in self._send_sessions.items()
                if rid not in pending_rids
            }
            send_reqs = {rid: self._send_reqs[rid] for rid in send_sessions}
        block_all = at_least_request_num is None
        wait_num = at_least_request_num if not block_all else 0
        # A durable local terminal candidate already counts as progress. Do not
        # block on an unrelated request while retrying its preparation.
        if pending_rids and not block_all:
            wait_num = 0
        need_progress = wait_num > 0
        if need_progress:
            self._poll_sessions_for_interval(
                send_sessions,
                send_reqs,
                wait_num,
                self._sender_future_timeout_ms,
            )

        local_completed, local_failed = self._collect_done(send_sessions, send_reqs)
        local_terminal_candidates = list(
            dict.fromkeys(
                [
                    *pending_candidate_rids,
                    *local_completed,
                    *local_failed,
                ]
            )
        )
        to_process = self._build_to_process(
            send_sessions,
            self._ctx_consensus(local_terminal_candidates),
            0 if need_progress else wait_num,
            block_all,
        )

        observed_completed: list[int] = []
        observed_failed: list[int] = []
        observed_cancelled: list[int] = []
        timed_out: list[int] = []
        for rid in to_process:
            if rid in pending_rids:
                continue
            session = send_sessions[rid]
            result = session.wait_complete(blocking=block_all)
            if result is None:
                # Logical cancellation can precede source gather/NIXL drain.
                # Keep the source owner until wait_complete reports a terminal
                # physical state.
                continue
            if session.status == SessionStatus.CANCELLED:
                observed_cancelled.append(rid)
            elif result == WaitResult.COMPLETED:
                observed_completed.append(rid)
            elif result == WaitResult.TIMEOUT:
                logger.warning(
                    f"TxSession rid={session.disagg_request_id} timed out after {self._sender_future_timeout_ms}ms"
                )
                timed_out.append(rid)
            else:
                logger.warning(f"TxSession rid={session.disagg_request_id} failed")
                observed_failed.append(rid)

        # Persist the exact owner before close/map validation can fail. A
        # prepared candidate remains in the ready advertisement on later polls
        # until the existing outcome consensus makes the distributed decision.
        self._enroll_context_retirements(
            (
                ("completed", observed_completed),
                ("failed", observed_failed),
                ("cancelled", observed_cancelled),
            ),
            send_sessions,
            send_reqs,
            mark_complete=mark_complete,
        )
        self._prepare_context_retirements()
        cancelled, failed, completed, cleanup_ready = self._context_candidate_payload(to_process)

        # All ranks must agree on per-rid outcome to avoid req.state divergence.
        cancelled, failed, completed, timed_out = self._ctx_consensus_outcome(
            to_process,
            cancelled,
            failed,
            completed,
            timed_out,
            cleanup_ready,
        )

        # The outcome consensus is also the prepare barrier: cleanup_ready is
        # intersected across every participating rank. Commit contains only the
        # real C++ state assignment and exact built-in dict removals.
        return self._commit_context_decisions(
            (
                ("completed", completed),
                ("failed", failed),
                ("cancelled", cancelled),
            )
        )

    def check_gen_transfer_status(self, at_least_request_num: Optional[int]):
        with self._status_call() as admitted:
            if not admitted:
                return [], [], []
            return self._check_gen_transfer_status_impl(at_least_request_num)

    def _check_gen_transfer_status_impl(self, at_least_request_num: Optional[int]):
        # Terminal ACK receipts are immutable control state, not allocation
        # ownership. Progress them on the GEN polling path even after the last
        # receive session has retired.
        retry_terminal_receipts = getattr(
            getattr(self, "_transfer_worker", None),
            "retry_terminal_receipt_acks",
            None,
        )
        if callable(retry_terminal_receipts):
            retry_terminal_receipts()
        self._prepare_generation_retirements()
        with self._get_session_admission_lock():
            pending_retirements = self._get_generation_retirements()
            if (
                not self._ever_had_recv_session
                and not self._gen_need_sync
                and not pending_retirements
            ):
                return [], [], []
            pending_candidate_rids = list(pending_retirements)
            pending_rids = set(pending_candidate_rids)
            recv_sessions = {
                rid: session
                for rid, session in self._recv_sessions.items()
                if rid not in pending_rids
            }
            recv_reqs = {rid: self._recv_reqs[rid] for rid in recv_sessions}
        block_all = at_least_request_num is None
        wait_num = at_least_request_num if not block_all else 0
        if pending_rids and not block_all:
            wait_num = 0
        need_progress = wait_num > 0
        if need_progress:
            self._poll_gen_sessions_for_poll_interval(wait_num, recv_sessions, recv_reqs)

        local_completed, local_failed = self._collect_done(recv_sessions, recv_reqs)
        local_terminal_candidates = list(
            dict.fromkeys(
                [
                    *pending_candidate_rids,
                    *local_completed,
                    *local_failed,
                ]
            )
        )
        to_process = self._build_to_process(
            recv_sessions,
            self._gen_consensus(local_terminal_candidates),
            0 if need_progress else wait_num,
            block_all,
        )

        observed_completed: list[int] = []
        observed_failed: list[int] = []
        observed_cancelled: list[int] = []
        for rid in to_process:
            if rid in pending_rids:
                continue
            session = recv_sessions[rid]
            result = session.wait_complete(blocking=block_all)
            if result not in (WaitResult.COMPLETED, WaitResult.FAILED):
                # Logical cancellation/failure may precede physical drain.
                # Neither a non-blocking miss nor a timeout proves that request
                # cleanup can no longer race a transfer accessor.
                continue
            if session.status == SessionStatus.CANCELLED:
                # Session cancelled — either by local cancel_request() (user
                # cancel) or by a remote CANCEL_SESSION message (e.g. CTX
                # server timeout).  Return the req objects so the caller can
                # distinguish the two cases and set the appropriate state.
                observed_cancelled.append(rid)
            elif result == WaitResult.COMPLETED:
                observed_completed.append(rid)
            elif result == WaitResult.FAILED:
                observed_failed.append(rid)

        self._enroll_generation_retirements(
            (
                ("completed", observed_completed),
                ("failed", observed_failed),
                ("cancelled", observed_cancelled),
            ),
            recv_sessions,
            recv_reqs,
        )
        self._prepare_generation_retirements()
        cancelled, failed, completed, cleanup_ready = self._generation_candidate_payload(to_process)

        # All ranks must agree on per-rid outcome to avoid req.state divergence.
        cancelled, failed, completed = self._gen_consensus_outcome(
            to_process, cancelled, failed, completed, cleanup_ready
        )

        self._record_generation_completion_metrics(completed)
        retired_completed, retired_failed, cancelled_reqs = self._commit_generation_decisions(
            (
                ("completed", completed),
                ("failed", failed),
                ("cancelled", cancelled),
            )
        )
        if retired_failed:
            logger.warning(
                f"Disagg gen transfer FAILED rank={self._dist.rank} "
                f"rids={retired_failed} gen_need_sync={self._gen_need_sync}"
            )

        return retired_completed, retired_failed, cancelled_reqs

    def _poll_gen_sessions_for_poll_interval(
        self, wait_num: int, sessions: dict, reqs: dict
    ) -> None:
        self._poll_sessions_for_interval(
            sessions,
            reqs,
            wait_num,
            self.kv_transfer_poll_interval_ms,
        )

    def _poll_sessions_for_interval(
        self,
        sessions: dict,
        reqs: dict,
        wait_num: int,
        poll_interval_ms: Optional[int],
    ) -> None:
        poll_interval_s = (poll_interval_ms or 0) / 1000.0
        deadline = time.monotonic() + poll_interval_s
        while True:
            completed, failed = self._collect_done(sessions, reqs)
            if len(completed) + len(failed) >= wait_num:
                return
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                return
            for session in sessions.values():
                session.wait_complete(blocking=False)
            time.sleep(min(0.001, remaining_s))

    def check_gen_transfer_complete(self):
        with self._get_session_admission_lock():
            return not self._recv_sessions and not self._get_generation_retirements()

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

    def _cancel_request_legacy(self, req: LlmRequest) -> bool:
        """Preserve the existing cancellation behavior for legacy callers."""
        rid = get_unique_rid(req)
        assert rid is not None
        with self._get_session_admission_lock():
            # Status/preparation calls take request/session snapshots outside
            # this lock while they enter collectives or poll workers. Removing
            # an owner in that interval would let this rank enroll a no-op
            # terminal candidate while peers retire the real request. Signal
            # cancellation, but retain every exact map entry until the active
            # collective-bearing call exits and cancellation is retried.
            status_call_active = getattr(self, "_active_status_calls", 0) > 0
            cleanup_pending = status_call_active
            cancelled_wait_reqs = self._get_cancelled_wait_reqs()
            context_retirement = self._get_context_retirements().get(rid)
            generation_retirement = self._get_generation_retirements().get(rid)
            # A stale cancellation must not remove a replacement request that
            # happens to reuse the same integer request ID.
            if self._wait_reqs.get(rid) is req:
                if status_call_active:
                    cancelled_wait_reqs[rid] = req
                else:
                    self._wait_reqs.pop(rid, None)
                    if cancelled_wait_reqs.get(rid) is req:
                        cancelled_wait_reqs.pop(rid, None)
            elif (
                status_call_active
                and self._wait_reqs.get(rid) is None
                and self._send_reqs.get(rid) is None
                and self._recv_reqs.get(rid) is None
                and context_retirement is None
                and generation_retirement is None
                and (cancelled_wait_reqs.get(rid) is None or cancelled_wait_reqs.get(rid) is req)
            ):
                # prepare_context_requests() marks its collective-bearing call
                # active before it can enroll the incoming owner. Preserve an
                # exact cancellation marker for that otherwise-unowned window
                # so the preparation cannot subsequently admit this request.
                cancelled_wait_reqs[rid] = req
            elif not status_call_active and cancelled_wait_reqs.get(rid) is req:
                cancelled_wait_reqs.pop(rid, None)

            cleanup_pending = (
                cleanup_pending
                or (context_retirement is not None and context_retirement.request is req)
                or (generation_retirement is not None and generation_retirement.request is req)
            )
            for direction, sessions, reqs in (
                ("send", self._send_sessions, self._send_reqs),
                ("receive", self._recv_sessions, self._recv_reqs),
            ):
                session = sessions.get(rid)
                if session is None or reqs.get(rid) is not req:
                    continue
                retirement = context_retirement if direction == "send" else generation_retirement
                if (
                    retirement is not None
                    and retirement.request is req
                    and retirement.session is session
                ):
                    # Local terminal arbitration already enrolled this exact
                    # owner. Its provisional ledger is now the sole path that
                    # may prepare teardown and publish the consensus decision.
                    continue
                try:
                    session.cancel()
                    if status_call_active:
                        # The active status call may still enroll this exact
                        # snapshot. It owns close/map retirement for this pass.
                        cleanup_pending = True
                        continue
                    if session.has_transferring_tasks():
                        cleanup_pending = True
                        continue
                    self._close_session_with_allocation_leases(
                        session,
                        PhysicalDisposition.QUIESCED_FAILURE,
                    )
                except Exception as error:
                    cleanup_pending = True
                    logger.error(
                        f"Failed to cancel {direction} KV transfer for request {rid}; "
                        f"retaining its owner for retry: {error}"
                    )
                    continue
                if sessions.get(rid) is session and reqs.get(rid) is req:
                    reqs.pop(rid, None)
                    sessions.pop(rid, None)

            return not cleanup_pending

    def cancel_request(self, req: LlmRequest) -> bool:
        """Retain the legacy boolean cancellation contract."""
        return self._cancel_request_legacy(req)

    @staticmethod
    def _session_physical_disposition(session: object) -> PhysicalDisposition:
        resources_drained = getattr(session, "resources_drained", None)
        if callable(resources_drained) and resources_drained():
            status = getattr(session, "status", None)
            if status in (SessionStatus.ERROR, SessionStatus.CANCELLED):
                return PhysicalDisposition.QUIESCED_FAILURE
            return PhysicalDisposition.QUIESCED_SUCCESS

        status = getattr(session, "status", None)
        if status is SessionStatus.ERROR:
            return PhysicalDisposition.IN_DOUBT
        if status is SessionStatus.CANCELLED:
            return PhysicalDisposition.QUIESCING
        return PhysicalDisposition.ACTIVE

    def poll_session(self, req: LlmRequest) -> PhysicalDisposition:
        """Report the exact retained owner's local physical disposition."""
        rid = get_unique_rid(req)
        if rid is None:
            return PhysicalDisposition.IN_DOUBT

        with self._get_session_admission_lock():
            if self._wait_reqs.get(rid) is req:
                return PhysicalDisposition.NOT_EXPOSED

            context_retirement = self._get_context_retirements().get(rid)
            generation_retirement = self._get_generation_retirements().get(rid)
            if (
                context_retirement is not None
                and context_retirement.request is req
                or generation_retirement is not None
                and generation_retirement.request is req
            ):
                return PhysicalDisposition.QUIESCING

            sessions = [
                session
                for session_map, request_map in (
                    (self._send_sessions, self._send_reqs),
                    (self._recv_sessions, self._recv_reqs),
                )
                if request_map.get(rid) is req
                if (session := session_map.get(rid)) is not None
            ]

        if not sessions:
            # Raw request IDs are not generation-safe, so absence from the
            # current maps cannot prove that an old peer has no accessor.
            return PhysicalDisposition.IN_DOUBT

        dispositions = [self._session_physical_disposition(session) for session in sessions]
        if PhysicalDisposition.IN_DOUBT in dispositions:
            return PhysicalDisposition.IN_DOUBT
        if PhysicalDisposition.ACTIVE in dispositions:
            return PhysicalDisposition.ACTIVE
        if PhysicalDisposition.QUIESCING in dispositions:
            return PhysicalDisposition.QUIESCING
        if PhysicalDisposition.QUIESCED_FAILURE in dispositions:
            return PhysicalDisposition.QUIESCED_FAILURE
        return PhysicalDisposition.QUIESCED_SUCCESS

    def cancel_session(self, req: LlmRequest, reason: str) -> CancelResult:
        """Cancel one exact retained owner while separating logical and physical state."""
        rid = get_unique_rid(req)
        if rid is None:
            return CancelResult(
                logical=LogicalDisposition.REJECTED,
                physical=PhysicalDisposition.IN_DOUBT,
                reason=f"request has no disaggregated identity: {reason}",
            )

        with self._get_session_admission_lock():
            waiting = self._wait_reqs.get(rid) is req
            owned = waiting or any(
                request_map.get(rid) is req for request_map in (self._send_reqs, self._recv_reqs)
            )
            owned = owned or (
                (retirement := self._get_context_retirements().get(rid)) is not None
                and retirement.request is req
            )
            owned = owned or (
                (retirement := self._get_generation_retirements().get(rid)) is not None
                and retirement.request is req
            )

        if not owned:
            return CancelResult(
                logical=LogicalDisposition.NOT_FOUND,
                physical=PhysicalDisposition.IN_DOUBT,
                retryable=True,
                reason=(
                    "no generation-safe owner matched this request; an old"
                    f" peer accessor cannot be excluded: {reason}"
                ),
            )

        drained = self._cancel_request_legacy(req)
        if drained:
            physical = (
                PhysicalDisposition.NOT_EXPOSED if waiting else PhysicalDisposition.QUIESCED_FAILURE
            )
        else:
            physical = self.poll_session(req)
            if physical is PhysicalDisposition.NOT_EXPOSED:
                physical = PhysicalDisposition.QUIESCING
        return CancelResult(
            logical=LogicalDisposition.ACCEPTED,
            physical=physical,
            retryable=not physical.is_reusable,
            reason=reason,
        )

    def fence_submission(self, req: LlmRequest) -> LogicalDisposition:
        """Close local publication gates for the exact request owner."""
        result = self.cancel_session(req, "submission fence requested")
        return result.logical

    def quiesce_session(self, req: LlmRequest) -> PhysicalDisposition:
        """Return current evidence; backend drain remains asynchronously owned."""
        return self.poll_session(req)

    def capabilities(self) -> TransceiverCapabilities:
        lifecycle_contract = getattr(self, "_lifecycle_contract", None)
        if lifecycle_contract is not None:
            return lifecycle_contract
        return self._build_lifecycle_capabilities(
            self._mapping,
            protocol_version=ProtocolVersion.QUALIFIED_LEGACY,
            bounce_transfer_enabled=self._bounce_transfer_enabled,
            supports_generation_first=True,
        )

    def shutdown_lifecycle(self, deadline_s: Optional[float]) -> ShutdownResult:
        """Attempt bounded drain and report only allocator-safe evidence."""
        deadline_at_s = None if deadline_s is None else time.monotonic() + max(0.0, deadline_s)
        while True:
            try:
                if self.shutdown():
                    return ShutdownResult(
                        physical=PhysicalDisposition.QUIESCED_SUCCESS,
                        in_doubt_context_count=0,
                    )
            except Exception as error:
                self._latch_retirement_fault(error)
                return ShutdownResult(
                    physical=PhysicalDisposition.IN_DOUBT,
                    in_doubt_context_count=None,
                    fatal=True,
                    reason=f"Python transceiver shutdown failed: {error}",
                )

            fault = getattr(self, "_retirement_fault", None)
            if deadline_at_s is None or time.monotonic() >= deadline_at_s:
                return ShutdownResult(
                    physical=PhysicalDisposition.IN_DOUBT,
                    in_doubt_context_count=None,
                    fatal=fault is not None,
                    reason=(
                        "Python transceiver did not prove physical drain before "
                        "the lifecycle shutdown deadline"
                    ),
                )
            time.sleep(min(0.01, max(0.0, deadline_at_s - time.monotonic())))

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
        params = {
            "ctx_dp_rank": ctx_dp_rank,
            "ctx_info_endpoint": [self._context_info_endpoint]
            if self._context_info_endpoint
            else None,
        }
        lifecycle_contract = self.capabilities()
        if lifecycle_contract.protocol_version == ProtocolVersion.GENERATION_SAFE:
            advertisement = getattr(self, "_lifecycle_advertisement", None)
            if advertisement is None:
                raise RuntimeError(
                    "protocol-v1 transceiver has no immutable lifecycle advertisement"
                )
            expected_capabilities = tuple(
                sorted(capability.value for capability in lifecycle_contract.supported)
            )
            if (
                advertisement.instance_id != self._instance_name
                or advertisement.protocol_version != lifecycle_contract.protocol_version
                or advertisement.capabilities != expected_capabilities
                or advertisement.qualified_legacy_mode != lifecycle_contract.qualified_legacy_mode
            ):
                raise RuntimeError(
                    "server lifecycle advertisement diverged from the runtime contract"
                )
            params["context_transceiver_lifecycle"] = advertisement.to_dict()
        return params

    def prepare_context_requests(self, requests: List[LlmRequest]):
        with self._status_call() as admitted:
            if not admitted:
                raise RuntimeError("KV cache transceiver is shutting down")
            self._prepare_context_requests_impl(requests)

    def _prepare_context_requests_impl(self, requests: List[LlmRequest]) -> None:
        # Place new generation-first context requests into wait state, then
        # use allgather consensus to promote ready requests to CONTEXT_INIT.
        with self._get_session_admission_lock():
            pending_admissions = {}
            planned_wait_owners = dict(self._wait_reqs)
            cancelled_wait_reqs = self._get_cancelled_wait_reqs()
            for req in requests:
                rid = get_unique_rid(req)
                assert rid is not None
                cancelled_wait_owner = cancelled_wait_reqs.get(rid)
                if cancelled_wait_owner is not None:
                    if cancelled_wait_owner is not req:
                        raise RuntimeError(
                            f"waiting request {rid} has a pending cancellation for a different request"
                        )
                    # Cancellation won after status-call admission but before
                    # wait enrollment. Still publish this exact owner to the
                    # local wait ledger so every rank enters readiness
                    # consensus; the marker below prevents local promotion
                    # until cancel_request() retries and retires both entries.
                if rid in self._get_context_retirements():
                    raise RuntimeError(f"send request {rid} has a pending terminal retirement")
                send_owner = self._send_reqs.get(rid)
                if rid in self._send_sessions:
                    if send_owner is not req:
                        raise RuntimeError(
                            f"send request {rid} already has a different live source owner"
                        )
                    continue
                if send_owner is not None and send_owner is not req:
                    raise RuntimeError(
                        f"send request {rid} already has a different live source owner"
                    )
                wait_owner = planned_wait_owners.get(rid)
                if wait_owner is not None and wait_owner is not req:
                    raise RuntimeError(
                        f"waiting request {rid} already has a different live source owner"
                    )
                planned_wait_owners[rid] = req
                pending_admissions[rid] = req

            # Validate the full batch before publishing any owner. A collision
            # must not leave an earlier request stranded in the wait ledger.
            for rid, req in pending_admissions.items():
                self._wait_reqs[rid] = req
                req.state = LlmRequestState.DISAGG_CONTEXT_WAIT_SCHEDULER

            if not self._wait_reqs:
                return

            # Check which waiting requests have peer info while the worker is
            # protected from concurrent shutdown. Promotion happens after the
            # collective and therefore revalidates each exact owner below.
            ready_owners = {
                rid: owner
                for rid, owner in self._wait_reqs.items()
                if self._get_cancelled_wait_reqs().get(rid) is not owner
                if self._transfer_worker.has_all_peer_req_infos_for_send(rid)
            }

        # Without consensus, background peer info arriving at different times on
        # different ranks causes scheduling mismatches → hang. Do not hold the
        # local admission lock across the distributed collective.
        ready_ids = self._ctx_consensus(list(ready_owners))
        with self._get_session_admission_lock():
            if getattr(self, "_shutdown_started", False):
                return
            for rid in ready_ids:
                owner = ready_owners.get(rid)
                if owner is not None and self._wait_reqs.get(rid) is owner:
                    # Every rank applies the decision from the same ready
                    # snapshot. A cancellation that arrived after that
                    # snapshot remains marked to block launch/reuse, but must
                    # not make this rank reverse a distributed promotion that
                    # its peers have already committed.
                    owner.state = LlmRequestState.CONTEXT_INIT
                    self._wait_reqs.pop(rid, None)

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
