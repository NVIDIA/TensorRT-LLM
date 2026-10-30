# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worker-owned notification transport for non-pipelined MPI TP/CP."""

from __future__ import annotations

import ipaddress
import json
import os
import socket
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

from .idle_notification import IdleRequestNotification

if TYPE_CHECKING:
    from mpi4py import MPI

    from .py_executor import PyExecutor


def _validate_speculative_progress(executor: PyExecutor) -> None:
    """Admit the audited request-bound speculative implementations."""
    config = executor.model_engine.spec_config
    drafter = executor.drafter
    if config is None and drafter is None and executor.draft_model_engine is None:
        return

    from tensorrt_llm.llmapi import NGramDecodingConfig, SADecodingConfig

    if type(config) is SADecodingConfig:
        from ..speculative.sa_worker import SASpecMetadata, SAWorker
        from ..speculative.suffix_automaton import SuffixAutomatonManager
        from .resource_manager import ResourceManagerType

        engine = executor.model_engine
        model = engine.model
        worker = engine._get_spec_worker()
        metadata = engine.spec_metadata
        manager = executor.resource_manager.get_resource_manager(
            ResourceManagerType.SPEC_RESOURCE_MANAGER
        )
        # SA runs inside the selected model forward. ModelLoader may deepcopy
        # its ModelConfig, so the worker must share the selected model's config
        # object, while that config must equal the engine's typed SA config.
        # Pending copies/masks are next-forward state, not completion jobs;
        # native SA operations use the existing current CUDA stream.
        if (
            drafter is not None
            or executor.draft_model_engine is not None
            or type(worker) is not SAWorker
            or model.spec_worker is not worker
            or type(model.spec_config) is not SADecodingConfig
            or worker.spec_config is not model.spec_config
            or model.spec_config != config
            or model.draft_model is not None
            or worker._auxiliary_state_handlers
            or type(metadata) is not SASpecMetadata
            or type(manager) is not SuffixAutomatonManager
            or metadata.sa_manager is not manager
            or metadata.spec_dec_mode != config.spec_dec_mode
        ):
            raise ValueError("Idle notification requires the initialized built-in SA worker/pool")
        return

    from ..speculative.ngram import NGramDrafter, NGramPoolManager

    # NGram's factory-owned drafter and pool execute inline over scheduled
    # generation requests. There is no background completion to wake an empty
    # executor. Exact types intentionally exclude user-provided implementations
    # whose progress contract is not implied by the Drafter base class.
    if (
        type(config) is not NGramDecodingConfig
        or type(drafter) is not NGramDrafter
        or type(drafter.spec_resource_manager) is not NGramPoolManager
        or drafter.spec_config is not config
        or executor.draft_model_engine is not None
    ):
        raise ValueError(
            "Idle notification currently supports only built-in NGram or SA speculation"
        )


def _validate_progress_modes(executor: PyExecutor) -> None:
    dist = executor.dist
    # With PP disabled, RequestBroadcaster sends both its count header and
    # payload through the same MPI world. CP partitions request contents only
    # after that broadcast; it does not require another idle progress source.
    if (
        dist.world_size < 2
        or dist.tp_size * dist.cp_size != dist.world_size
        or dist.pp_size != 1
        or executor.enable_attention_dp
        or executor._disable_mpi
    ):
        raise ValueError("Idle notification requires non-pipelined MPI TP/CP")
    # Rebalance is deliberately not a gate: each ordinary executor loop checks
    # it before fetching requests, and the original idle leader already blocks
    # at that fetch. Parking followers does not remove a progress callback.
    for name in ("_resource_governor_enabled", "is_benchmark_disagg"):
        if getattr(executor, name, None) is not False:
            raise ValueError(f"Idle notification does not support executor mode {name}")
    if (
        executor.kv_cache_transceiver is not None
        or executor.kv_connector_manager is not None
        or executor.llm_args.sleep_config is not None
        or executor._gen_admission.enabled
        or executor.request_broadcaster.sync_clock
        or executor._first_token_timer is not None
    ):
        raise ValueError("Idle notification requires no transfer, sleep or timed admission")
    _validate_speculative_progress(executor)
    # Synchronous item scheduling has no independent wakeup source. Preserve
    # enabled model features, but reject inconsistent scheduler ownership.
    engine = executor.model_engine
    enabled = executor._mm_encoder_item_scheduling_enabled
    scheduler = engine._mm_item_scheduler
    if (
        type(enabled) is not bool
        or engine.mm_encoder_item_scheduling_enabled is not enabled
        or (enabled and type(scheduler).__name__ != "MultimodalItemScheduler")
        or (not enabled and scheduler is not None)
    ):
        raise ValueError("Inconsistent multimodal item scheduling state")


class IdleNotificationRuntime:
    """Own the private communicator and each worker's authenticated queues."""

    def __init__(self, comm: MPI.Comm, address: str) -> None:
        self._comm = comm
        self._address = address

    @contextmanager
    def worker(self, executor: PyExecutor) -> Iterator[None]:
        import zmq

        from tensorrt_llm.executor.ipc import ZeroMqQueue

        owner = threading.get_ident()
        queues: list[ZeroMqQueue] = []
        senders: list[ZeroMqQueue] = []
        receiver: ZeroMqQueue | None = None

        def check_owner() -> None:
            if threading.get_ident() != owner:
                raise RuntimeError("Idle notification socket used from another thread")

        def send(notice: tuple[int, bool]) -> None:
            check_owner()
            for queue in senders:
                queue.put(notice)

        def receive() -> tuple[int, bool]:
            check_owner()
            if receiver is None:
                raise RuntimeError("The leader has no follower notification receiver")
            return receiver.get()

        try:
            if executor.dist.rank != 0:
                receiver = ZeroMqQueue(
                    address=(f"tcp://{self._address}:*", None),
                    is_server=True,
                    name="idle-request-notification",
                )
                queues.append(receiver)
            # Authentication material is exchanged only through the existing
            # MPI bootstrap, never through settings, diagnostics or logs.
            endpoints = self._comm.gather(
                receiver.address if receiver is not None else None, root=0
            )
            if executor.dist.rank == 0:
                for endpoint in endpoints[1:]:
                    queue = ZeroMqQueue(
                        address=endpoint, is_server=False, name="idle-request-notification"
                    )
                    queues.append(queue)
                    senders.append(queue)
            executor.request_broadcaster.idle_notification = IdleRequestNotification(send, receive)
            yield
        finally:
            executor.request_broadcaster.idle_notification = None
            # The resumed count collective fences each delivered notice.
            # A completed loop must retire its communicator even if a local
            # profiler or queue teardown raises. A genuinely failed loop must
            # not wait in collective cleanup before launcher containment.
            cleanup_error: zmq.ZMQError | None = None
            try:
                check_owner()
                for queue in queues:
                    try:
                        if queue.socket is not None:
                            queue.socket.setsockopt(zmq.LINGER, 0)
                    except zmq.ZMQError as error:
                        if cleanup_error is None:
                            cleanup_error = error
                    try:
                        queue.close()
                    except zmq.ZMQError as error:
                        if cleanup_error is None:
                            cleanup_error = error
            finally:
                if executor._event_loop_completed:
                    self._comm.Free()
            if cleanup_error is not None:
                raise cleanup_error


def prepare_idle_notification(executor: PyExecutor) -> IdleNotificationRuntime | None:
    """Validate all ranks and duplicate the communicator before workers start.

    The feature is default-off. Its interface map is explicit so an MPI world
    spanning hosts never silently connects followers through loopback.
    """
    from ..distributed import MPIDist

    selected = os.environ.get("TLLM_MPI_IDLE_NOTIFICATION", "0")
    if not isinstance(executor.dist, MPIDist):
        if selected != "0":
            raise ValueError("Idle notification requires the MPI backend")
        return None
    from tensorrt_llm._utils import mpi_comm

    comm = mpi_comm()
    selections = comm.allgather(selected)
    if any(value != selections[0] for value in selections) or selected not in ("0", "1"):
        raise ValueError("Idle notification selection must match on every MPI rank")
    if selected == "0":
        return None
    from mpi4py import MPI

    failure = None
    address = ""
    try:
        _validate_progress_modes(executor)
        if (
            comm.Get_size() != executor.dist.world_size
            or comm.Get_rank() != executor.dist.rank
            or MPI.Query_thread() < MPI.THREAD_MULTIPLE
        ):
            raise ValueError("Idle notification requires matching MPI ranks and THREAD_MULTIPLE")
        addresses = json.loads(os.environ["TLLM_MPI_IDLE_NOTIFICATION_ADDRESSES"])
        address = addresses[socket.gethostname().split(".")[0]]
        ip = ipaddress.IPv4Address(address)
        if ip.is_loopback or ip.is_unspecified or ip.is_multicast:
            raise ValueError("Idle notification requires an explicit local IPv4 interface")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as check:
            check.bind((address, 0))
    except (AttributeError, KeyError, TypeError, ValueError, OSError) as error:
        failure = str(error)
    failures = comm.allgather(failure)
    if any(failures):
        raise ValueError(f"Idle notification setup rejected: {failures}")
    return IdleNotificationRuntime(comm.Dup(), address)
