# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worker-owned notification transport for non-pipelined MPI executors."""

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

    from tensorrt_llm.llmapi import MTPDecodingConfig, NGramDecodingConfig, SADecodingConfig

    if type(config) is MTPDecodingConfig:
        from ..speculative.eagle3 import Eagle3OneModelSpecMetadata, MTPEagleWorker
        from ..speculative.mtp import MTPSpecMetadata, MTPWorker
        from ..speculative.mtp_dynamic_tree import MTPEagleDynamicTreeWorker

        worker = executor.model_engine._get_spec_worker()
        metadata = executor.model_engine.spec_metadata
        if config.spec_dec_mode.is_mtp_vanilla():
            worker_type = MTPWorker
            metadata_type = MTPSpecMetadata
        else:
            worker_type = MTPEagleDynamicTreeWorker if config.use_dynamic_tree else MTPEagleWorker
            metadata_type = Eagle3OneModelSpecMetadata
        # These built-in workers draft and commit state inside the selected
        # model forward. Retained hidden states/SA caches are not independent
        # completion jobs. Do not infer the same contract for custom workers.
        if (
            drafter is not None
            or executor.draft_model_engine is not None
            or type(worker) is not worker_type
            or type(worker.spec_config) is not MTPDecodingConfig
            or worker.spec_config != config
            or worker._auxiliary_state_handlers
            or type(metadata) is not metadata_type
            or metadata.spec_dec_mode != config.spec_dec_mode
        ):
            raise ValueError(
                "Idle notification requires the initialized built-in MTP worker/metadata"
            )
        return

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
            "Idle notification requires request-bound built-in NGram, SA or MTP speculation"
        )


def can_park_idle_requests(executor: PyExecutor) -> bool:
    """Agree on quiescence before the existing idle request fetch can block.

    Call only at a group-uniform idle boundary (global live-request count and
    replicated waiting queue both empty). A completed request can still own a
    send/save, and an attention-DP peer can own work absent on rank zero. Keep
    the ordinary executor loop progressing until every rank has drained it.
    """
    transceiver = executor.kv_cache_transceiver
    pending = (
        executor._has_pending_connector_transfers()
        or executor.async_transfer_manager.has_any_inflight_requests()
        or (transceiver is not None and not transceiver.check_gen_transfer_complete())
        or bool(executor._pending_response_terminations)
        or bool(executor.control_requests)
        or bool(executor.request_accumulated)
    )
    return not any(executor.dist.allgather(pending))


def _validate_progress_modes(executor: PyExecutor) -> None:
    dist = executor.dist
    # With PP disabled, RequestBroadcaster sends both its count header and
    # payload through the same MPI world. CP partitions request contents only
    # after that broadcast; it does not require another idle progress source.
    if (
        dist.world_size < 2
        or dist.tp_size * dist.cp_size != dist.world_size
        or dist.pp_size != 1
        or executor._disable_mpi
    ):
        raise ValueError("Idle notification requires non-pipelined MPI TP/CP")
    # Rebalance is deliberately not a gate: each ordinary executor loop checks
    # it before fetching requests, and the original idle leader already blocks
    # at that fetch. Parking followers does not remove a progress callback.
    if executor._resource_governor_enabled is not False:
        raise ValueError(
            "Idle notification does not support the independent resource-governor queue"
        )
    if (
        executor.llm_args.sleep_config is not None
        or executor._gen_admission.enabled
        or executor.request_broadcaster.sync_clock
        or executor._first_token_timer is not None
    ):
        raise ValueError("Idle notification requires no sleep or timed admission")
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
    """Automatically select notification when every rank meets its prerequisites.

    An explicit opt-out or unavailable capability keeps the existing wait.
    The interface map remains explicit: loopback is not a multi-host route.
    """
    from ..distributed import MPIDist

    selected = os.environ.get("TLLM_MPI_IDLE_NOTIFICATION", "auto")
    if not isinstance(executor.dist, MPIDist):
        if selected not in ("auto", "0", "1"):
            raise ValueError("Idle notification selection must be auto, 0 or 1")
        return None
    from tensorrt_llm._utils import mpi_comm
    from tensorrt_llm.logger import logger

    comm = mpi_comm()
    selections = comm.allgather(selected)
    if any(value not in ("auto", "0", "1") for value in selections):
        raise ValueError("Idle notification selection must be auto, 0 or 1 on every rank")
    # A local opt-out applies to the entire group. "1" is a compatibility
    # alias for auto, never an instruction to bypass missing prerequisites.
    if "0" in selections:
        return None
    from mpi4py import MPI

    failure = None
    address = ""
    try:
        _validate_progress_modes(executor)
    except (AttributeError, TypeError, ValueError):
        failure = "unsupported or incomplete executor progress contract"
    if failure is None:
        if (
            comm.Get_size() != executor.dist.world_size
            or comm.Get_rank() != executor.dist.rank
            or MPI.Query_thread() < MPI.THREAD_MULTIPLE
        ):
            failure = "MPI rank/world or thread capability unavailable"
    if failure is None:
        try:
            addresses = json.loads(os.environ["TLLM_MPI_IDLE_NOTIFICATION_ADDRESSES"])
            address = addresses[socket.gethostname().split(".")[0]]
            ip = ipaddress.IPv4Address(address)
            if ip.is_loopback or ip.is_unspecified or ip.is_multicast:
                raise ValueError("Peer-reachable local IPv4 required")
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as check:
                check.bind((address, 0))
        except (KeyError, TypeError, ValueError, OSError):
            failure = "peer-reachable local interface configuration unavailable"
    failures = comm.allgather(failure)
    if any(failures):
        if executor.dist.rank == 0:
            reason = next(value for value in failures if value is not None)
            logger.info(f"MPI idle notification inactive (auto): {reason}; keeping existing wait")
        return None
    if executor.dist.rank == 0:
        logger.info("MPI idle notification active (auto)")
    return IdleNotificationRuntime(comm.Dup(), address)
