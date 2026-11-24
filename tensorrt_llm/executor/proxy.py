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
import atexit
import collections
import concurrent.futures
import json
import os
import pickle  # nosec B403
import shutil
import sys
import tempfile
import threading
import time
import weakref
from queue import Empty, Queue
from typing import Callable, Dict, List, Optional, Union

import torch
import zmq
import zmq.asyncio

from tensorrt_llm._errors import RuntimeInitError
from tensorrt_llm._startup import _StartupTimer
from tensorrt_llm._startup_errors import (StartupErrorInfo,
                                          is_worker_startup_error,
                                          log_startup_error,
                                          wrap_worker_startup_error)
from tensorrt_llm.logger import logger

from .._torch.pyexecutor.llm_request import release_undelivered_response
from .._utils import customized_gc_thresholds, mpi_rank, nvtx_range_debug
from ..llmapi.llm_args import TorchLlmArgs
from ..llmapi.mpi_session import (MpiCommSession, MpiPoolSession, MpiSession,
                                  RemoteMpiCommSessionClient,
                                  validate_session_world_size)
from ..llmapi.tracer import enable_llm_tracer, get_tracer, global_tracer
from ..llmapi.utils import (AsyncQueue, ManagedThread, _SyncQueue,
                            enable_llm_debug, logger_debug, print_colored)
from .executor import GenerationExecutor
from .ipc import FusedIpcQueue, IpcQueue, RequestSendCancelled
from .postproc_worker import PostprocWorker, PostprocWorkerConfig
from .prompt_tail import PromptTailPolicy
from .request import (CancellingRequest, GenerationRequest, StartProfileRequest,
                      StopProfileRequest)
from .result import GenerationResult, IterationResult
from .rpc import RPCClient
from .rpc.rpc_common import RPCError, get_unique_ipc_addr
from .utils import (EngineDeadError, ErrorResponse, RequestError,
                    WorkerCommIpcAddrs, create_mpi_comm_session,
                    get_spawn_proxy_process_env, is_llm_response,
                    multi_frontend_request_addr, multi_frontend_result_addr,
                    namespace_client_id, print_alive_threads)
from .worker import GenerationExecutorWorker, worker_main
from .worker_process_monitor import WorkerProcessIdentity, WorkerProcessMonitor

__all__ = [
    "GenerationExecutorProxy",
    "GenerationExecutorFrontendProxy",
]

# Methods that are explicitly implemented for multi-rank MPI/IPC executor
# deployments and may be called via collective_rpc even when world_size > 1.
# sleep() and wakeup() coordinate across all ranks via a dedicated control
# communicator started in PyExecutor.start_worker(); rank-0 is the entry
# point, so routing through the rank-0 RPC shim is correct.
# get_memory_report() is answered by rank-0, which in the memory monitor's
# full mode gathers the other ranks' reports over its own communicator.
_MULTI_RANK_ALLOWED_METHODS: frozenset[str] = frozenset(
    {"get_memory_report", "get_memory_status", "sleep", "wakeup"})


def _check_collective_rpc_guard(
    model_world_size: int,
    unique_reply_rank: Optional[int],
    target_ranks: Optional[Union[int, List[int]]],
    method: str = "",
) -> None:
    """Validate collective_rpc preconditions shared by IPC and RPC proxies.

    Args:
        model_world_size: Total number of GPU worker processes.
        unique_reply_rank: Must be ``None`` for this shim.
        target_ranks: Must be ``None`` for this shim.
        method: Name of the method being invoked; used to allow selected
            multi-rank-capable methods past the world_size guard.

    Raises:
        NotImplementedError: If ``model_world_size > 1`` and ``method`` is not
            in :data:`_MULTI_RANK_ALLOWED_METHODS`, or if
            ``unique_reply_rank`` or ``target_ranks`` are provided.
    """
    if model_world_size > 1 and method not in _MULTI_RANK_ALLOWED_METHODS:
        raise NotImplementedError(
            f"MPI collective_rpc does not support model_world_size > 1 for "
            f"method '{method}'; use the Ray executor for general multi-rank "
            "deployments, or use sleep()/wakeup() which handle multi-rank "
            "coordination internally.")
    if unique_reply_rank is not None or target_ranks is not None:
        raise NotImplementedError(
            "unique_reply_rank and target_ranks are not supported; "
            "this shim only reaches rank-0.")


def _send_quit(request_queue: IpcQueue, should_stop: Callable[[],
                                                              bool]) -> None:
    """Best-effort quit from the socket's writer, even when the peer is gone."""
    try:
        if not should_stop():
            try:
                request_queue.put_interruptible(None, should_stop)
                return
            except RequestSendCancelled:
                pass
        request_queue.put_nowait(None)
    except (RuntimeError, zmq.ZMQError) as exc:
        logger.warning(f"Executor proxy quit could not be delivered: {exc}")


class _RequestSenderClosed(RuntimeError):
    """The batched request sender no longer accepts requests."""


class _RequestSender:
    """Batch a proxy's requests to the rank-0 worker on one background thread.

    Used only with ``TRTLLM_PROXY_BATCH_REQUESTS=1``. Callers pickle each
    request themselves and ``send`` the bytes, so a request is captured as it
    is when it is submitted, and one that cannot be pickled fails in its
    caller. The sender thread sends every payload queued since its previous
    send as one list in one HMAC-signed message, so a burst of requests costs
    the rank-0 worker one receive and one HMAC check instead of one per
    request; the worker then unpickles each payload. The thread starts on
    the first ``send``. From then on it is the only thread that writes to the
    request queue's socket, which is not thread-safe, and it also sends the
    worker's shutdown sentinel.

    Trade-off: the sender thread must take the GIL before it can send. While
    the submitting thread keeps running Python, for example a busy
    trtllm-serve event loop, a lone request can wait up to the interpreter's
    switch interval (5 ms by default) before it is sent.
    """

    MAX_BATCH_SIZE = 1024

    def __init__(self,
                 request_queue: IpcQueue,
                 error_queue: Queue,
                 should_stop: Optional[Callable[[], bool]] = None) -> None:
        self._request_queue = request_queue
        self._error_queue = error_queue
        self._should_stop = should_stop
        self._deadline: Optional[float] = None
        self._pending = collections.deque()
        self._cond = threading.Condition()
        self._closed = False
        self._send_sentinel = False
        # Set by the thread once it has drained the queue and will stop.
        self._stopping = False
        self._thread: Optional[threading.Thread] = None
        # Bound here because ``close`` can run from a finalizer during
        # interpreter teardown, when module globals may already be None.
        self._is_finalizing = sys.is_finalizing

    def send(self, payload: bytes) -> None:
        """Queue a pickled request for the worker.

        Payloads sent after ``close`` fail explicitly in the caller.
        """
        with self._cond:
            if self._closed:
                raise _RequestSenderClosed(
                    "Executor proxy request sender is closed.")
            if self._thread is None:
                self._thread = threading.Thread(target=self._run,
                                                daemon=True,
                                                name="proxy_request_sender")
                self._thread.start()
            self._pending.append(payload)
            self._cond.notify()

    def close(self,
              timeout: Optional[float] = None,
              *,
              send_sentinel: bool = False) -> bool:
        """Send everything queued, then stop the sender thread.

        With ``send_sentinel``, the thread also sends the worker's shutdown
        sentinel after the last payload, so the request queue keeps a single
        writer. Calls after the thread has stopped return at once. During
        interpreter finalization this returns without doing anything: the
        daemon thread can no longer run, the ZMQ context may be gone, and
        joining a thread there fails.

        Returns:
            Whether the caller still has to send the sentinel: true only if
            ``send_sentinel`` is set and the thread did not send it, because it
            never started or had already stopped.
        """
        if self._is_finalizing():
            return False
        with self._cond:
            self._closed = True
            if timeout is not None:
                deadline = time.monotonic() + timeout
                self._deadline = (deadline if self._deadline is None else min(
                    deadline, self._deadline))
            thread = self._thread
            caller_sends_sentinel = send_sentinel and (thread is None
                                                       or self._stopping)
            self._send_sentinel = self._send_sentinel or send_sentinel
            self._cond.notify()
        if thread is not None:
            if thread is threading.current_thread():
                raise RuntimeError("Request sender cannot join itself.")
            # Delivery has its own deadline; allow a separate bounded interval
            # for the interrupted owner to unwind and exit.
            thread.join(None if timeout is None else timeout + 1.0)
            if thread.is_alive():
                raise TimeoutError(
                    "Executor proxy request sender is still closing.")
            self._thread = None
        return caller_sends_sentinel

    def _send_stopped(self) -> bool:
        return ((self._deadline is not None
                 and time.monotonic() >= self._deadline)
                or (self._should_stop is not None and self._should_stop()))

    def _run(self) -> None:
        while True:
            with self._cond:
                while not self._pending and not self._closed:
                    self._cond.wait()
                if not self._pending:
                    self._stopping = True
                    send_sentinel = self._send_sentinel
                    break
                batch = [
                    self._pending.popleft()
                    for _ in range(min(len(self._pending), self.MAX_BATCH_SIZE))
                ]
            try:
                self._request_queue.put_interruptible(batch, self._send_stopped)
            except Exception as e:
                with self._cond:
                    unsent = len(batch) + len(self._pending)
                    self._closed = True
                    self._pending.clear()
                    self._stopping = True
                    send_sentinel = self._send_sentinel
                if isinstance(e, RequestSendCancelled):
                    logger.warning("Executor proxy request sender stopped with "
                                   f"{unsent} request(s) unsent: {e}")
                else:
                    # Pickling already completed in the caller. Report all
                    # other failures, even if shutdown began during the send.
                    self._error_queue.put_nowait(e)
                break
        if send_sentinel:
            # A bare None, not a batch: the worker stops at a top-level None.
            try:
                _send_quit(self._request_queue, self._send_stopped)
            except Exception as e:
                self._error_queue.put_nowait(e)


def _create_request_sender(
    request_queue: IpcQueue,
    error_queue: Queue,
    should_stop: Optional[Callable[[],
                                   bool]] = None) -> Optional[_RequestSender]:
    """Return a batching request sender if TRTLLM_PROXY_BATCH_REQUESTS=1."""
    if os.getenv("TRTLLM_PROXY_BATCH_REQUESTS", "0") != "1":
        return None
    return _RequestSender(request_queue, error_queue, should_stop)


class GenerationExecutorProxy(GenerationExecutor):
    READY_SIGNAL = b"READY"
    WORKER_PROCESS_IDENTITIES_SIGNAL = b"WORKER_PROCESS_IDENTITIES"
    # Set when TRTLLM_PROXY_BATCH_REQUESTS=1; requests are put on
    # ``request_queue`` directly otherwise.
    _request_sender: Optional[_RequestSender] = None
    # Total wait for the worker futures in an orderly shutdown. Generous
    # enough for the teardown of a large engine.
    _WORKER_SHUTDOWN_TIMEOUT_S = 180.0
    _RESULT_RECV_TIMEOUT_S = 0.1
    _DISPATCH_DRAIN_TIMEOUT_S = 5.0
    _DISPATCH_STOP_TIMEOUT_S = 5.0
    # Set from gen_prompt_tail_tokens; None sends full prompts.
    _prompt_tail: Optional[PromptTailPolicy] = None

    def _init_lifecycle(self) -> None:
        self._admission_lock = threading.RLock()
        self._request_send_lock = threading.RLock()
        self._pre_shutdown_lock = threading.RLock()
        self._is_finalizing = sys.is_finalizing
        self._shutdown_lock = threading.RLock()
        self._pre_shutdown_complete = False
        self._shutdown_complete = False
        self._request_shutdown_deadline: Optional[float] = None
        self._request_transport_closed = False
        self._profile_control_futures: set[concurrent.futures.Future] = set()
        self._profile_control_thread: Optional[threading.Thread] = None
        self._profile_control_cancelled = False
        self.workers_started = False
        self.doing_shutdown = False
        self._engine_dead = False
        self._results: Dict[int, GenerationResult] = {}
        self.dispatch_result_thread: Optional[ManagedThread] = None
        self.rpc_client: Optional[RPCClient] = None
        self._shutdown_event = threading.Event()
        self._worker_process_monitor = WorkerProcessMonitor()

    def __init__(
        self,
        worker_kwargs: dict,
        model_world_size: int = 1,
        mpi_session: Optional[MpiSession] = None,
        *,
        worker_cls: type = GenerationExecutorWorker,
        postproc_worker_config: Optional[PostprocWorkerConfig] = None,
        is_llm_executor: Optional[bool] = None,
    ) -> None:
        self._init_lifecycle()
        proxy_start = time.perf_counter()
        logger.info(f"[startup][pid={os.getpid()}] executor_proxy: start")
        postproc_worker_config = postproc_worker_config or PostprocWorkerConfig(
        )
        super().__init__(
            num_postprocess_workers=postproc_worker_config.
            num_postprocess_workers,
            postprocess_tokenizer_dir=postproc_worker_config.
            postprocess_tokenizer_dir,
            is_llm_executor=is_llm_executor,
        )

        self.worker_cls = worker_cls

        mpi_process_pre_spawned: bool = get_spawn_proxy_process_env()
        self._owns_mpi_session = mpi_session is None

        with _StartupTimer("mpi_session_setup"):
            if mpi_session is None:
                if mpi_process_pre_spawned:
                    logger_debug('create comm session ...\n', "yellow")
                    self.mpi_session = create_mpi_comm_session(model_world_size)
                else:
                    logger_debug('create pool session ...\n', "yellow")
                    self.mpi_session = MpiPoolSession(
                        n_workers=model_world_size)
            else:
                # submit() launches one worker task per pool worker, so an
                # external session must match the model's world size exactly;
                # fail loudly instead of starting the wrong number of executors.
                validate_session_world_size(mpi_session, model_world_size)
                logger_debug('using external mpi session ...\n', "yellow")
                self.mpi_session = mpi_session

        if isinstance(self.mpi_session,
                      (MpiCommSession, RemoteMpiCommSessionClient)):
            print_colored(
                f"rank {mpi_rank()} using MpiCommSession to bind to external MPI processes\n",
                "yellow")
        else:
            print_colored(
                f"rank {mpi_rank()} using MpiPoolSession to spawn MPI processes\n",
                "yellow")

        self.model_world_size = model_world_size

        _llm_args = worker_kwargs.get("llm_args", None)
        self.garbage_collection_gen0_threshold = _llm_args.garbage_collection_gen0_threshold if _llm_args is not None else None
        self._enable_resource_governor = bool(
            getattr(_llm_args, "enable_resource_governor", False))
        prompt_tail = (_llm_args.gen_prompt_tail_tokens if isinstance(
            _llm_args, TorchLlmArgs) else None)
        if prompt_tail:
            self._prompt_tail = PromptTailPolicy(prompt_tail)
            logger.info(f"prompt tail trimming on, tail={prompt_tail}")

        # Multi-frontend serving: this launcher proxy owns the shared ipc
        # dir + HMAC key for the per-frontend endpoints (_setup_queues);
        # trtllm-serve hands them to the attached frontends via
        # multi_frontend_attach_info().
        self._num_frontends = (_llm_args.num_serve_frontends
                               if _llm_args is not None else 1)
        self._multi_frontend_ipc_dir: Optional[str] = None
        self._multi_frontend_hmac: Optional[bytes] = None
        if self._num_frontends > 1:
            if self._enable_resource_governor:
                raise ValueError(
                    "Multi-frontend serving does not support "
                    "enable_resource_governor: the resource-governor signal "
                    "only reaches the launcher frontend.")
            self._multi_frontend_ipc_dir = tempfile.mkdtemp(
                prefix="trtllm_frontends_")
            self._multi_frontend_hmac = os.urandom(32)

        # Generate RPC address and key for stats RPC
        self.rpc_addr = get_unique_ipc_addr()
        self.hmac_key = os.urandom(32)

        worker_kwargs = dict(**worker_kwargs,
                             worker_queues=self._setup_queues(),
                             postproc_worker_config=postproc_worker_config,
                             is_llm_executor=False,
                             rpc_addr=self.rpc_addr,
                             hmac_key=self.hmac_key)

        if "log_level" not in worker_kwargs:
            worker_kwargs["log_level"] = logger.level

        logger.info(
            f"[startup][pid={os.getpid()}] executor_proxy/pre_worker_setup: "
            f"done in {time.perf_counter() - proxy_start:.3f}s (includes mpi_session_setup)"
        )
        self._start_executor_workers(worker_kwargs)
        logger.info(
            f"[startup][pid={os.getpid()}] executor_proxy/workers_ready: "
            f"elapsed={time.perf_counter() - proxy_start:.3f}s")

        # Create RPC client after workers are started (worker starts RPC server)
        self.rpc_client = RPCClient(self.rpc_addr, hmac_key=self.hmac_key)

        # Start a background thread that monitors for fatal errors (e.g. MPI
        # worker crash) and triggers shutdown even when no health checks or
        # generate() calls are in flight.
        self._error_monitor_thread = threading.Thread(
            target=self._error_monitor_loop,
            daemon=True,
            name="proxy_error_monitor")
        self._error_monitor_thread.start()

        # Single-thread executor that owns *all* profile-control IPC traffic
        # (``request_queue.put`` for StartProfile/StopProfile and
        # ``profile_ack_queue.get`` for the matching ack). Pinning these
        # ZMQ socket operations to one owning thread is required because:
        #
        #   * pyzmq sockets are not thread-safe; concurrent access from
        #     multiple threads — even one writer + one reader — is undefined
        #     behavior in libzmq.
        #   * The HTTP handler reaches us through ``asyncio.to_thread``,
        #     which does not pin to the same worker thread between calls,
        #     so back-to-back ``/start_profile`` and ``/stop_profile``
        #     could otherwise touch ``profile_ack_queue`` from two
        #     different threads.
        #
        # The HTTP handler still blocks on a Future returned by this
        # executor, so the synchronous "chrome trace is on disk by the
        # time /stop_profile returns 200" contract is preserved without
        # any ZMQ socket op leaking into the FastAPI thread pool.
        #
        # NOTE: ``request_queue`` is also written from ``submit()`` /
        # ``abort_request()`` on user-caller threads — that pre-existing
        # cross-thread access is out of scope for this change. The fix
        # here narrowly targets the new profile-control path.
        # With TRTLLM_PROXY_BATCH_REQUESTS=1, the ``_request_sender`` thread
        # writes the queue instead once it has started, shutdown sentinel
        # included.
        self._profile_control_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="proxy_profile_control")

        # MPI registers its joiner using threading._register_atexit if possible.
        # These functions run before atexit.register, so to avoid deadlock,
        # we have to notify workers to exit before MPI starts to wait them.
        try:
            threading._register_atexit(  # type: ignore[attr-defined]
                self.pre_shutdown)
        except AttributeError:
            atexit.register(self.pre_shutdown)

    def _check_mpi_futures(self) -> bool:
        """Check if any MPI worker future has completed (crashed/exited).

        If a dead worker is found, records the fatal error and calls
        ``pre_shutdown()``.

        Returns:
            True if a dead worker was detected.
        """
        if not hasattr(self, 'mpi_futures') or not self.mpi_futures:
            return False
        for f in self.mpi_futures:
            if f.done():
                exc = f.exception() if not f.cancelled() else None
                error = exc or RuntimeError("MPI worker exited unexpectedly")
                self._set_fatal_error(error)
                if not self.doing_shutdown:
                    self.pre_shutdown()
                return True
        return False

    def _check_mpi_workers(self) -> bool:
        """Check OS process handles and MPI futures for worker death."""
        dead_worker = self._worker_process_monitor.find_dead_worker()
        if dead_worker is not None:
            self._set_fatal_error(
                RuntimeError("MPI worker rank "
                             f"{dead_worker.rank} (pid {dead_worker.pid}) "
                             "exited unexpectedly"))
            if not self.doing_shutdown:
                self.pre_shutdown()
            return True
        return self._check_mpi_futures()

    def _drain_error_queue(self) -> bool:
        """Drain all queued errors, skipping per-request errors.

        Sets ``_fatal_error`` and calls ``pre_shutdown()`` on the first
        system-level error found.

        Returns:
            True if any items were drained from the queue.
        """
        drained = False
        while True:
            try:
                e = self._error_queue.get_nowait()
                self._error_queue.task_done()
                drained = True
                if isinstance(e, (str, RequestError)):
                    continue
                self._set_fatal_error(e)
                if not self.doing_shutdown:
                    self.pre_shutdown()
                break
            except Empty:
                break
        return drained

    def check_health(self) -> bool:
        """Check executor health including MPI worker liveness.

        Inlines the base ``check_health()`` logic instead of calling
        ``super().check_health()`` so that fatal queue items trigger
        ``pre_shutdown()`` (non-blocking) rather than ``shutdown()``
        (which blocks on ``f.result()`` for surviving MPI workers).

        Returns:
            True if the executor and all MPI workers are healthy.
        """
        if self.doing_shutdown or self._fatal_error is not None:
            return False

        if self._drain_error_queue():
            return self._fatal_error is None and not self.doing_shutdown

        if self._check_mpi_workers():
            return False

        return True

    def _set_fatal_error(self, error: BaseException) -> None:
        """Record the fatal error, then unblock every pending request.

        Extends the base behavior so that the instant a worker death is
        recorded (from the error monitor, a health check, or a crashed future),
        all pending GenerationResults fail fast with EngineDeadError rather than
        blocking forever on a response queue whose producer is gone.
        """
        super()._set_fatal_error(error)
        self._mark_engine_dead(error)

    def _mark_engine_dead(self, error: Optional[BaseException] = None) -> None:
        """Set the sticky flag and push EngineDeadError onto pending results."""
        with self._admission_lock:
            if self._engine_dead:
                return
            self._engine_dead = True
            # Own the notifications before shutdown can clear the registry.
            results = list(self._results.values())
        dead_error = EngineDeadError(error)
        # Deliver outside the admission lock so queue notification cannot block it.
        # Use put() (not put_nowait()) so the async _SyncQueue path also wakes
        # the awaiting event loop, not just the sync Queue path.
        for result in results:
            try:
                result.queue.put(dead_error)
            except Exception:  # noqa: BLE001 - a full/closed queue must not stop the sweep
                pass
        # Release the session's exit joins here, not at teardown: interpreter
        # exit joins non-daemon threads before any teardown code runs, so a
        # wedged pool manager thread must be deregistered while user code is
        # still alive. Non-destructive, hence safe for unowned sessions.
        release = getattr(getattr(self, 'mpi_session', None),
                          'release_exit_joins', None)
        if release is not None:
            try:
                release()
            except Exception as e:  # noqa: BLE001 - best-effort cleanup
                logger.debug(
                    f"MPI session exit-join release failed (ignored): {e!r}")

    def _handle_worker_death(self, error: BaseException) -> None:
        """Event-driven worker-death handler.

        Runs from the MPI future's done-callback thread the instant a worker
        process exits. Enqueues the error for the monitor loop to record (which
        drives pre_shutdown) and immediately broadcasts EngineDeadError to
        pending requests, so they fail fast without waiting for the next poll
        tick. Propagation is therefore no longer gated by the poll interval.
        """
        self._error_queue.put_nowait(error)
        self._mark_engine_dead(error)

    def _check_remote_worker_death(self) -> bool:
        """Poll a remote (pre-spawned) MPI session for forwarded worker deaths.

        Under ``TLLM_SPAWN_PROXY_PROCESS=1`` / ``trtllm-llmapi-launch`` the
        session is a ``RemoteMpiCommSessionClient`` whose ``submit()`` returns
        no futures, so the ``mpi_done_callback`` path never fires and
        ``_check_mpi_futures()`` has nothing to watch. The remote server
        forwards worker exceptions over its control socket instead
        (``RemoteWorkerDeath``); surface them here into the same fast-death
        path so this deployment mode gets identical EngineDeadError behavior.

        Returns:
            True if a dead worker was detected.
        """
        check = getattr(self.mpi_session, "check_worker_error", None)
        if check is None:
            return False
        try:
            error = check()
        except Exception as exc:  # noqa: BLE001 - monitor must not die
            logger.debug(f"check_worker_error failed (ignored): {exc!r}")
            return False
        if error is None:
            return False
        logger.error(f"Remote MPI worker death reported: {error!r}")
        self._handle_worker_death(error)
        self._set_fatal_error(error)
        if not self.doing_shutdown:
            self.pre_shutdown()
        return True

    def _error_monitor_loop(self) -> None:
        """Background thread that reaps a dead engine and drives pre_shutdown.

        Checks local MPI worker process handles and futures, remote-session
        worker-death notifications, and the error queue using the shared
        ``_check_mpi_workers()``, ``_check_remote_worker_death()`` and
        ``_drain_error_queue()`` helpers.

        Propagation to pending requests is event-driven via
        ``_handle_worker_death`` (the MPI future done-callback) where futures
        exist; for remote sessions it is bounded by this loop's poll interval.
        Uses ``_shutdown_event`` for clean wakeup instead of a sleep loop, so
        shutdown is immediate.
        """
        while not self.doing_shutdown and self._fatal_error is None:
            try:
                if self._check_mpi_workers():
                    logger.error("Error monitor: MPI worker crash detected, "
                                 "shutting down")
                    return

                if self._check_remote_worker_death():
                    logger.error(
                        "Error monitor: remote MPI worker death detected, "
                        "shutting down")
                    return

                self._drain_error_queue()
                if self._fatal_error is not None:
                    return
            except Exception as exc:
                logger.debug(f"Error monitor: unexpected exception (ignored): "
                             f"{exc!r}")

            # Backstop poll only; the latency-sensitive propagation path is
            # event-driven (see _handle_worker_death), so a coarse interval is
            # fine. Wakes immediately if _shutdown_event is set.
            self._shutdown_event.wait(timeout=5.0)

    def _setup_queues(self) -> WorkerCommIpcAddrs:
        frontend_result_addrs = None
        if self._num_frontends > 1:
            # The rank0 worker BINDS the request ingress (PULL) so every
            # frontend can PUSH-connect; each frontend (incl. this launcher,
            # frontend 0) binds its own result lane (PULL).
            ipc_dir = self._multi_frontend_ipc_dir
            hmac_key = self._multi_frontend_hmac
            request_addr = (multi_frontend_request_addr(ipc_dir), hmac_key)
            frontend_result_addrs = [(multi_frontend_result_addr(ipc_dir,
                                                                 i), hmac_key)
                                     for i in range(self._num_frontends)]
            self.request_queue = IpcQueue(request_addr,
                                          is_server=False,
                                          socket_type=zmq.PUSH,
                                          name="proxy_request_queue")
            self._request_sender = _create_request_sender(
                self.request_queue, self._error_queue,
                self._request_send_stopped)
            self.result_queue = FusedIpcQueue(
                frontend_result_addrs[0],
                is_server=True,
                fuse_message=False,
                socket_type=zmq.PULL,
                recv_timeout=self._RESULT_RECV_TIMEOUT_S,
                name="proxy_result_queue")
        else:
            request_addr = None
            self.request_queue = IpcQueue(is_server=True,
                                          name="proxy_request_queue")
            self._request_sender = _create_request_sender(
                self.request_queue, self._error_queue,
                self._request_send_stopped)
            # TODO[chunweiy]: Unify IpcQueue and FusedIpcQueue
            # Use PULL mode when enable_postprocess_parallel as there are
            # multiple senders from multiple processes.
            self.result_queue = FusedIpcQueue(
                is_server=True,
                fuse_message=False,
                socket_type=zmq.PULL
                if self.enable_postprocess_parallel else zmq.PAIR,
                recv_timeout=self._RESULT_RECV_TIMEOUT_S,
                name="proxy_result_queue")
        self.worker_init_status_queue = IpcQueue(
            is_server=True,
            socket_type=zmq.ROUTER,
            name="worker_init_status_queue")
        self._resource_governor_queue = IpcQueue(
            is_server=True, name="proxy_resource_governor_queue"
        ) if self._enable_resource_governor else None
        # Worker -> proxy ack channel for synchronous control requests
        # (start_profile / stop_profile). Worker pushes a tuple
        # ``(kind, error_msg)`` after it has finished processing the
        # corresponding request; ``error_msg`` is non-None when
        # ``PyExecutor`` rejected the call (e.g. RequestError "already
        # in progress") so the proxy can re-raise. PAIR socket is fine
        # because the worker leader is the sole producer.
        # Created on first use, not here. Binding it at startup measurably
        # perturbs this process: with the queue created eagerly the disagg
        # /steady_clock_offset handshake takes 30-200 ms instead of 2-4 ms,
        # which is enough to break the causality assertion in
        # test_disaggregated_perf_metrics. Profiling is off in every normal
        # deployment, so the socket is pure cost there anyway.
        self.profile_ack_queue = None
        # Stats and KV events are now fetched via RPC, not IPC queues.
        return WorkerCommIpcAddrs(
            # A connect-mode queue has no bound .address; use the preset one.
            request_queue_addr=request_addr
            if request_addr is not None else self.request_queue.address,
            worker_init_status_queue_addr=self.worker_init_status_queue.address,
            result_queue_addr=self.result_queue.address,
            resource_governor_queue_addr=self._resource_governor_queue.address
            if self._resource_governor_queue is not None else None,
            frontend_result_queue_addrs=frontend_result_addrs,
        )

    def multi_frontend_attach_info(self) -> Optional[dict]:
        """The attach payload consumed by attached serving frontends.

        See GenerationExecutorFrontendProxy and commands/serve.py. Returns
        None unless multi-frontend mode is active.
        """
        if self._num_frontends <= 1:
            return None
        ipc_dir = self._multi_frontend_ipc_dir
        hmac_key = self._multi_frontend_hmac
        return {
            "mode":
            "classic",
            "request_addr":
            multi_frontend_request_addr(ipc_dir),
            "result_addrs": [
                multi_frontend_result_addr(ipc_dir, i)
                for i in range(self._num_frontends)
            ],
            "hmac_key":
            hmac_key.hex(),
            # Attached frontends must apply the same collective_rpc guard
            # as the launcher (see _check_collective_rpc_guard).
            "model_world_size":
            self.model_world_size,
            # Stats / KV events / disagg params RPC endpoint on the rank0
            # worker (ROUTER socket, natively multi-client).
            "rpc_addr":
            self.rpc_addr,
            "rpc_hmac_key":
            self.hmac_key.hex(),
            "prompt_tail_tokens":
            self._prompt_tail.tail if self._prompt_tail is not None else 0,
        }

    @property
    def resource_governor_queue(self):
        return self._resource_governor_queue

    def abort_request(self, request_id: int) -> None:
        """Abort a request by sending a cancelling request to the request queue.

        Args:
            request_id (int): The id of the request to abort.
        """
        # NOTE, it just sends a cancelling request to the request queue, but it
        # may take a while for the request to be cancelled in the worker and
        # send back a finished result.
        if self._request_send_stopped():
            return
        try:
            self._send_request(CancellingRequest(request_id))
        except RequestSendCancelled:
            # Teardown or engine death supplies the terminal notification.
            return
        except _RequestSenderClosed:
            if not self.doing_shutdown:
                raise

    # Per-call timeouts for the synchronous worker round-trip. The stop
    # path can take up to 30s because PyExecutor.stop_profile itself
    # polls ``_profile_enabled`` for that long while the executor loop
    # flushes the chrome trace; we add a small safety margin.
    _START_PROFILE_ACK_TIMEOUT_S = 60.0
    _STOP_PROFILE_ACK_TIMEOUT_S = 35.0

    def _wait_profile_ack(self, expected_kind: str, timeout: float) -> None:
        """Block on the worker's ack queue for a profile control request.

        Waits for the worker to push an ack on ``profile_ack_queue``
        matching ``expected_kind`` ("start" or "stop"), or until the
        cumulative ``timeout`` elapses across any number of stale-ack
        consumptions.

        If a stale or out-of-order ack is read (e.g. the previous call
        timed out and its ack arrived after we returned), we discard it
        and keep waiting for the expected kind. Returning on a mismatch
        would break the synchronous contract: the next call could return
        before its own request has been processed, making the documented
        guarantee ("trace is on disk by the time /stop_profile returns
        200") unsound under back-to-back invocations.

        If the worker reports a rejection (non-empty error message), we
        re-raise as ``RuntimeError`` so the HTTP layer can map "already
        in progress" / "pending" to 409. On true timeout we log a warning
        and return — the chrome trace may still be flushing on the
        backend, but blocking the FastAPI event loop indefinitely is
        worse than a possibly-stale 200.
        """
        deadline = time.monotonic() + timeout
        while True:
            self._check_profile_control_open(expected_kind)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.warning(
                    f"{expected_kind}_profile: timed out waiting for "
                    f"worker ack after {timeout:.1f}s. The chrome trace "
                    "may not be on disk yet.")
                return
            try:
                ack = self.profile_ack_queue.get(timeout=min(remaining, 0.05))
            except Empty:
                continue
            kind, error_msg = ack
            if kind != expected_kind:
                # Out-of-order ack from a previous call; discard and keep
                # waiting for the ack that belongs to this request.
                logger.warning(
                    f"profile ack kind mismatch (expected "
                    f"{expected_kind!r}, got {kind!r}); discarding stale "
                    "ack and continuing to wait.")
                continue
            if error_msg:
                raise RuntimeError(error_msg)
            return

    def _ensure_profile_ack_queue(self):
        """Bind the ack channel, once, on first profile control request.

        Only ever called from ``_profile_control_executor``, which is a
        single thread, so no additional locking is needed.
        """
        if self.profile_ack_queue is None:
            self.profile_ack_queue = IpcQueue(is_server=True,
                                              name="proxy_profile_ack_queue")
        return self.profile_ack_queue.address

    def _profile_control_call(self, kind: str, request, timeout: float) -> None:
        """Worker-side body of ``start_profile`` / ``stop_profile``.

        Runs on ``_profile_control_executor`` so all profile-related
        ZMQ socket ops (``request_queue.put`` and ``profile_ack_queue.get``
        via ``_wait_profile_ack``) happen on a single owning thread. With
        TRTLLM_PROXY_BATCH_REQUESTS=1 the request is put by
        ``_request_sender`` instead.
        """
        self._profile_control_thread = threading.current_thread()
        self._check_profile_control_open(kind)
        request.ack_addr = self._ensure_profile_ack_queue()
        self._send_request(request, should_stop=self._profile_send_stopped)
        self._wait_profile_ack(kind, timeout)

    def _profile_send_stopped(self) -> bool:
        return self._profile_control_cancelled or self._request_send_stopped()

    def _check_profile_control_open(self, kind: str) -> None:
        if self._profile_send_stopped():
            raise RuntimeError(
                f"{kind}_profile: executor proxy is shutting down or worker is dead."
            )

    def _close_profile_control(self) -> None:
        """Drain admitted calls, then interrupt them at the shutdown deadline."""
        executor = getattr(self, "_profile_control_executor", None)
        if executor is None:
            return
        if self._profile_control_thread is threading.current_thread():
            raise RuntimeError("Profile control cannot join itself.")
        with self._admission_lock:
            futures = tuple(self._profile_control_futures)
        executor.shutdown(wait=False)
        if self._engine_dead:
            self._profile_control_cancelled = True
        pending = [future for future in futures if not future.done()]
        if pending and not self._profile_control_cancelled:
            concurrent.futures.wait(pending,
                                    timeout=self._PROFILE_SHUTDOWN_TIMEOUT_S)
        if any(not future.done() for future in futures):
            self._profile_control_cancelled = True
            executor.shutdown(wait=False, cancel_futures=True)
            pending = [future for future in futures if not future.done()]
            if pending:
                concurrent.futures.wait(pending, timeout=1.0)
        if any(not future.done() for future in futures):
            raise TimeoutError(
                "Executor proxy profile calls are still closing.")
        # These private futures have no user callbacks. No accepted call can
        # still be using a socket; reap the idle worker, if one was started.
        executor.shutdown(wait=True)
        self._profile_control_executor = None

    def _submit_profile_control(self, kind: str, request,
                                timeout: float) -> None:
        """Dispatch a profile-control request onto the owning thread.

        ``shutdown()`` drains and clears ``_profile_control_executor``, so a
        request arriving during or after teardown would otherwise fail with
        ``AttributeError`` on ``None``. Raise ``RuntimeError`` instead — the
        HTTP handlers already map it to a controlled response.

        ``pre_shutdown()`` flips ``doing_shutdown`` and sends the quit
        sentinel to the workers well before ``shutdown()`` clears the
        executor, so check that first: a request submitted in that window
        would otherwise block for the whole ack timeout waiting on workers
        that are already exiting.
        """
        with self._admission_lock:
            executor = self._profile_control_executor
            if self.doing_shutdown or executor is None:
                raise RuntimeError(
                    f"{kind}_profile: the executor proxy is shutting down; "
                    "profile control is unavailable.")
            future = executor.submit(self._profile_control_call, kind, request,
                                     timeout)
            self._profile_control_futures.add(future)
        try:
            future.result()
        except concurrent.futures.CancelledError as exc:
            raise RuntimeError(
                f"{kind}_profile: executor proxy is shutting down.") from exc
        finally:
            with self._admission_lock:
                self._profile_control_futures.discard(future)

    def start_profile(self,
                      output_dir=None,
                      num_steps=None,
                      start_step: int = 0,
                      activities=None) -> None:
        """Forward runtime profiling start to the IPC worker process.

        Blocks until the worker has processed the request, then
        raises ``RuntimeError`` if ``PyExecutor`` rejected the start
        (e.g. a profile window is already active or pending) so the
        HTTP /start_profile endpoint can return 409.

        The actual ZMQ traffic (``request_queue.put`` +
        ``profile_ack_queue.get``) is dispatched onto
        ``_profile_control_executor`` — a single-thread executor — so the
        FastAPI thread-pool worker that called us never touches the
        ZMQ sockets directly. ``Future.result()`` re-raises any
        ``RuntimeError`` from ``_wait_profile_ack`` so the HTTP layer
        still sees worker rejections.
        """
        request = StartProfileRequest(output_dir=output_dir,
                                      num_steps=num_steps,
                                      start_step=start_step,
                                      activities=activities)
        self._submit_profile_control("start", request,
                                     self._START_PROFILE_ACK_TIMEOUT_S)

    def stop_profile(self) -> None:
        """Forward runtime profiling stop to the IPC worker process.

        Blocks until the worker has finished flushing the chrome
        trace. This makes the IPC-proxy path match the in-process and
        RPC-proxy paths: by the time this method returns the trace
        file is on disk, so HTTP callers can reliably read it
        immediately after /stop_profile returns 200.

        Like ``start_profile``, the actual ZMQ traffic is dispatched
        onto ``_profile_control_executor`` so the calling thread never
        touches the ZMQ sockets.
        """
        self._submit_profile_control("stop", StopProfileRequest(),
                                     self._STOP_PROFILE_ACK_TIMEOUT_S)

    def dispatch_result_task(self) -> bool:
        # TODO[chunweiy]: convert the dispatch_result_task to async, that should
        # benefit from zmq.asyncio.Context
        try:
            res = self.result_queue.get()
        except Empty:
            return True
        if res is None:
            return False  # shutdown the thread

        async_queues = []
        event_loop = None

        def process_res(res):
            client_id = res.client_id
            nonlocal event_loop
            nonlocal async_queues

            # The result may already be finalized and popped below — e.g. a
            # `terminate` verdict sent a final Output while the
            # engine still has in-flight responses (abort is async). Drop such
            # late responses instead of crashing with a KeyError.
            result = self._results.get(client_id)
            if result is None:
                release_undelivered_response(res)
                return
            queue = result.queue
            if isinstance(queue, _SyncQueue):
                queue.put_nowait(res)
                async_queues.append(queue)
                # all the loops are identical
                event_loop = event_loop or queue.loop
            else:
                queue.put(res)

            # FIXME: Add type annotations and make 'res' type more homogeneous (e.g.
            #        include PostprocWorker.Output in is_llm_response and unify is_final APIs).
            if (is_llm_response(res) and res.result.is_final) or isinstance(
                    res,
                    ErrorResponse) or (isinstance(res, PostprocWorker.Output)
                                       and res.is_final):
                self._results.pop(client_id, None)

        res = res if isinstance(res, list) else [res]

        keep_running = True
        for i in res:
            global_tracer().log_instant("IPC.get")
            if i is None:
                keep_running = False
            elif keep_running:
                process_res(i)
            else:
                release_undelivered_response(i)

        if async_queues:
            try:
                _SyncQueue.notify_many(event_loop, async_queues)
            except AsyncQueue.EventLoopShutdownError:
                logger.warning(
                    "proxy.py: EventLoopShutdownError because event loop is not running"
                )

        return keep_running

    # NOTE: _iteration_result_task, dispatch_stats_task, and dispatch_kv_cache_events_task
    # have been removed as stats and kv_events are now fetched via RPC directly.

    def _start_dispatch_threads(self) -> None:
        with self._admission_lock:
            if self.doing_shutdown:
                raise RuntimeError("Executor is shutting down")
            if self.dispatch_result_thread is None:
                self.dispatch_result_thread = ManagedThread(
                    weakref.WeakMethod(self.dispatch_result_task),
                    error_queue=self._error_queue,
                    name="proxy_dispatch_result_thread",
                    context=customized_gc_thresholds(
                        self.garbage_collection_gen0_threshold),
                )
                self.dispatch_result_thread.start()

        self._handle_background_error()

    def _stop_dispatch_thread(self,
                              *,
                              drain: bool,
                              deadline: Optional[float] = None) -> None:
        """Join the result reader before its socket can be released.

        A healthy engine sends its final responses followed by a sentinel.
        Only a dead peer, local frontend detach, or an expired drain switches
        the reader to local stop; its socket receive timeout makes that stop
        interruptible without a poll in the per-response path.
        """
        thread = self.dispatch_result_thread
        if thread is None or not thread.is_alive():
            return
        if thread is threading.current_thread():
            thread.stop()
            raise RuntimeError("Result dispatcher cannot join itself")
        if deadline is None:
            deadline = (time.monotonic() + self._DISPATCH_DRAIN_TIMEOUT_S +
                        self._DISPATCH_STOP_TIMEOUT_S)
        if drain:
            # Reserve one receive interval for a local stop if no sentinel
            # arrives before the shared shutdown deadline.
            drain_timeout = max(
                0.0,
                deadline - time.monotonic() - 2 * self._RESULT_RECV_TIMEOUT_S)
            thread.join(
                timeout=min(self._DISPATCH_DRAIN_TIMEOUT_S, drain_timeout))
        if thread.is_alive():
            thread.stop()
            thread.join(timeout=min(self._DISPATCH_STOP_TIMEOUT_S,
                                    max(0.0, deadline - time.monotonic())))
        if thread.is_alive():
            raise TimeoutError("Result dispatcher is still shutting down")

    def _abort_owned_session(self, reason: BaseException) -> None:
        """Force the worker world down, but only if this proxy created it.

        An externally owned (shared) session must stay alive for its owner to
        tear down.
        """
        if self._owns_mpi_session:
            self.mpi_session.shutdown_abort(reason=reason)

    def _fail_initialization(self,
                             error: BaseException,
                             cause: Optional[BaseException] = None):
        """Ordered teardown for every initialization-failure path.

        Abort strictly before marking: shutdown_abort() escalates to
        MPI_Abort only if its own blocking shutdown() overruns the grace
        period, and _mark_engine_dead() -> release_exit_joins() marks the
        pool dead, which forces that shutdown() non-blocking. Marking first
        would defang the abort and leave wedged ranks holding their weights.
        Every initialization failure must come through here so the ordering
        cannot be violated from a side entrance. Marking runs even if the
        abort itself raises, so the engine-dead bookkeeping (and the original
        initialization error) cannot be lost to a teardown failure.
        """
        # The abort can end this process (MPI_Abort) before the error reaches
        # the LLM boundary, so the classified hint is logged first.
        error.__cause__ = cause
        log_startup_error(error, "LLM initialization failed")
        root_cause = cause if cause is not None else error
        try:
            self._abort_owned_session(root_cause)
        except Exception as abort_error:  # noqa: BLE001 - teardown must not mask the init failure
            logger.error(f"Session abort failed during initialization "
                         f"teardown (continuing): {abort_error!r}")
        finally:
            self._set_fatal_error(root_cause)
            if not self.doing_shutdown:
                self.pre_shutdown()
        raise error from cause

    def _detect_worker_death_during_init(self) -> Optional[BaseException]:
        """Detect a worker death without recording or tearing anything down.

        The runtime checks (_check_mpi_workers, _check_remote_worker_death)
        mark the engine dead as a side effect of detection, and marking
        before the abort would defang it (see _fail_initialization), so the
        init wait loop needs detection kept separate from handling.
        """
        dead_worker = self._worker_process_monitor.find_dead_worker()
        if dead_worker is not None:
            return RuntimeInitError(
                f"MPI worker rank {dead_worker.rank} (pid {dead_worker.pid}) "
                "exited unexpectedly",
                reason="worker_died")
        for fut in self.mpi_futures:
            if fut.done():
                # exception() raises CancelledError on a cancelled future,
                # which would escape the init wait loop without the ordered
                # teardown; a cancelled worker future still means this rank
                # can never come up, so report it as a death instead.
                exc = fut.exception() if not fut.cancelled() else None
                return exc or RuntimeInitError("MPI worker exited unexpectedly",
                                               reason="worker_died")
        check = getattr(self.mpi_session, "check_worker_error", None)
        if check is None:
            return None
        try:
            return check()
        except Exception as exc:  # noqa: BLE001 - detection must not die
            logger.debug(f"check_worker_error failed (ignored): {exc!r}")
            return None

    def _start_executor_workers(self, worker_kwargs):

        self_ref = weakref.ref(self)

        def mpi_done_callback(future: concurrent.futures.Future):
            # This is called when the MPI worker is done, so future.exception()
            # will not block.
            if future.exception() is not None:
                if self_ := self_ref():
                    self_._handle_worker_death(future.exception())

        tracer_init_kwargs = get_tracer().init_kwargs if enable_llm_tracer(
        ) else None
        # Only the architectures a worker cannot find for itself, and only as
        # module names. Built-ins come from the worker's own static index, and
        # naming a module rather than sending the class keeps the worker's zoo
        # lazy -- a class in this payload would be resolved while unpickling,
        # importing its module in every worker whether or not it is ever used.
        from tensorrt_llm._torch.models.modeling_utils import \
            export_external_model_modules
        torch.cuda.Stream()

        # Strip the tokenizer from worker_kwargs to avoid MPI pickle failures.
        # Tokenizers using trust_remote_code=True reference classes in the
        # dynamic 'transformers_modules' namespace that cannot be deserialized
        # on remote MPI worker nodes.  The tokenizer is not needed by workers
        # (it is only consumed by the proxy / rank-0 leader).
        mpi_worker_kwargs = {
            k: v
            for k, v in worker_kwargs.items() if k != 'tokenizer'
        }
        worker_process_identities_signal = (
            self.WORKER_PROCESS_IDENTITIES_SIGNAL
            if self._can_monitor_worker_processes() else None)

        with _StartupTimer("worker_submit_to_ready") as startup_timer:
            with startup_timer.phase("worker_submit"):
                self.mpi_futures = self.mpi_session.submit(
                    worker_main,
                    **mpi_worker_kwargs,
                    worker_cls=self.worker_cls,
                    tracer_init_kwargs=tracer_init_kwargs,
                    _torch_external_model_modules=export_external_model_modules(
                    ),
                    ready_signal=GenerationExecutorProxy.READY_SIGNAL,
                    worker_process_identities_signal=
                    worker_process_identities_signal,
                )

            self.workers_started = True

            with startup_timer.phase("worker_initialization_wait"):
                status = self._wait_for_executor_workers_ready()

            ready_signal, error_trace = status[:2]
            if ready_signal != GenerationExecutorProxy.READY_SIGNAL:
                logger.error(
                    f"Executor worker initialization error: {error_trace}")
                # The worker classified the error before pickling dropped its
                # cause chain; carry that onto the error raised here.
                info = status[2] if len(status) > 2 and isinstance(
                    status[2], StartupErrorInfo) else None
                self._fail_initialization(
                    wrap_worker_startup_error("Executor worker returned error",
                                              info), ready_signal)

        # Register the fast-death callback only after the world reported
        # ready: on an already-completed future, add_done_callback() runs the
        # callback synchronously right here, and _handle_worker_death ->
        # _mark_engine_dead() would mark the pool dead BEFORE
        # _fail_initialization's abort, defanging the MPI_Abort escalation.
        # Until this point _detect_worker_death_during_init() covers worker
        # deaths.
        for fut in self.mpi_futures:
            fut.add_done_callback(mpi_done_callback)

    def _wait_for_executor_workers_ready(self) -> tuple:
        """Wait for worker readiness while monitoring published processes."""
        worker_processes_registered = False

        while True:
            if self.worker_init_status_queue.poll(1):
                status = self.worker_init_status_queue.get()
                # Send ACK to the worker
                self.worker_init_status_queue.put("ACK")
                logger.info("get signal from executor worker")

                signal = status[0]
                if signal == self.WORKER_PROCESS_IDENTITIES_SIGNAL:
                    if len(status) != 3:
                        raise RuntimeError(
                            "Executor worker returned invalid process identities"
                        )
                    self._register_worker_processes(status)
                    worker_processes_registered = True
                    continue

                # Backward compatibility for workers that only publish their
                # identities together with READY.
                if (signal == self.READY_SIGNAL
                        and not worker_processes_registered):
                    self._register_worker_processes(status)
                return status

            death = self._detect_worker_death_during_init()
            if death is not None:
                # A non-leader rank that fails here has no status queue, so its
                # peers stay blocked in the init collective still holding their
                # share of the weights. Raising alone leaks them until job end
                # and makes the next blocking shutdown() hang instead of
                # reporting this failure.
                if is_worker_startup_error(death):
                    # The rank raised its classified error, which already names
                    # the rank, stage and root cause.
                    self._fail_initialization(death)
                message = f"Executor worker died during initialization: {death}"
                logger.error(message)
                self._fail_initialization(RuntimeError(message), death)
            self._handle_background_error()

    def _register_worker_processes(self, status: tuple) -> None:
        """Register identities returned by locally spawned MPI workers.

        Test session reuse replaces this module's ``MpiPoolSession`` class
        reference with a factory, so identify pool-backed sessions by excluding
        the external communication session types.
        """
        if self._can_monitor_worker_processes() and len(status) == 3:
            worker_process_identities: List[WorkerProcessIdentity] = status[2]
            self._worker_process_monitor.register(worker_process_identities)

    def _can_monitor_worker_processes(self) -> bool:
        """Return whether the session uses locally spawned MPI workers."""
        return not isinstance(
            self.mpi_session,
            (MpiCommSession, RemoteMpiCommSessionClient),
        )

    def _abort_all_requests(self) -> None:
        # Results may finish while the snapshot is cancelled.
        for result in list(self._results.values()):
            result.abort()

    _PROFILE_SHUTDOWN_TIMEOUT_S = 5.0
    _REQUEST_SHUTDOWN_TIMEOUT_S = 5.0

    def _drain_control_and_requests(self, *, cancel_requests: bool,
                                    send_sentinel: bool) -> None:
        try:
            self._close_profile_control()
        finally:
            # The worker notification has its own budget even if profiling
            # timed out. A retry must not send more traffic after the quit.
            if not self._request_transport_closed:
                if self._request_shutdown_deadline is None:
                    self._request_shutdown_deadline = time.monotonic(
                    ) + self._REQUEST_SHUTDOWN_TIMEOUT_S
                try:
                    if cancel_requests:
                        self._abort_all_requests()
                finally:
                    self._close_request_sender(send_sentinel=send_sentinel)

    def _pre_shutdown(self) -> None:
        with self._admission_lock:
            self.doing_shutdown = True
        logger_debug('Proxy.pre_shutdown...\n', "yellow")

        self._worker_process_monitor.close()

        # Wake the error monitor thread immediately so it exits cleanly
        self._shutdown_event.set()

        # A remote session has no local futures; for a local session notify
        # surviving workers even if some ranks have already exited.
        send_sentinel = (self.workers_started
                         and (not self.mpi_futures
                              or any(not f.done() for f in self.mpi_futures)))
        self._drain_control_and_requests(cancel_requests=self.workers_started
                                         and not self._engine_dead,
                                         send_sentinel=send_sentinel)

    def _send_request(self,
                      request: object,
                      *,
                      should_stop: Optional[Callable[[], bool]] = None) -> None:
        """Send a request to the rank-0 worker.

        With TRTLLM_PROXY_BATCH_REQUESTS=1 the request is pickled here, on the
        caller's thread, and ``_request_sender`` batches the bytes; otherwise
        it is put on ``request_queue`` directly. Either way the worker receives
        the request as it is now, and a request that cannot be pickled raises
        here.
        """
        if self._request_sender is None:
            # The active writer can be a profiling call. A cancellation send
            # during teardown must not wait indefinitely for that writer.
            deadline = self._request_shutdown_deadline
            timeout = -1 if deadline is None else max(
                0.0, deadline - time.monotonic())
            if not self._request_send_lock.acquire(timeout=timeout):
                raise RequestSendCancelled(
                    "Executor proxy request send cancelled during shutdown.")
            try:
                self.request_queue.put_interruptible(
                    request, should_stop or self._request_send_stopped)
            finally:
                self._request_send_lock.release()
        else:
            self._request_sender.send(pickle.dumps(request))

    def _send_generation_request(self, request: object) -> None:
        """Order generation admission before teardown's cancel/quit traffic."""
        sender = self._request_sender
        if sender is not None:
            payload = pickle.dumps(request)
            with self._admission_lock:
                if self.doing_shutdown:
                    raise RuntimeError("Executor proxy is shutting down.")
                sender.send(payload)
        else:
            with self._request_send_lock:
                if self.doing_shutdown:
                    raise RuntimeError("Executor proxy is shutting down.")
                self.request_queue.put_interruptible(request,
                                                     self._request_send_stopped)

    def _request_send_stopped(self) -> bool:
        return (self._engine_dead or self._request_transport_closed
                or (self._request_shutdown_deadline is not None
                    and time.monotonic() >= self._request_shutdown_deadline))

    def _close_request_sender(self, send_sentinel: bool = False) -> None:
        """Stop the request sender, if any, then send the shutdown sentinel if asked.

        The sentinel follows every queued request. It is sent by the thread
        that has been writing ``request_queue``: the sender thread once it has
        started, the caller otherwise.
        """
        deadline = self._request_shutdown_deadline
        if deadline is None:
            deadline = time.monotonic() + self._REQUEST_SHUTDOWN_TIMEOUT_S
            self._request_shutdown_deadline = deadline
        if self._request_sender is not None:
            send_sentinel = self._request_sender.close(
                timeout=max(0.0, deadline - time.monotonic()),
                send_sentinel=send_sentinel)
        # Direct sends may be running on caller/profile threads. Never close
        # their socket until the last user has released it.
        if not self._request_send_lock.acquire(
                timeout=max(0.0, deadline - time.monotonic()) + 1.0):
            raise TimeoutError(
                "Executor proxy request transport is still closing.")
        try:
            if send_sentinel and not self._request_transport_closed:
                _send_quit(self.request_queue, self._request_send_stopped)
            self._request_transport_closed = True
        finally:
            self._request_send_lock.release()

    def _get_next_client_id(self) -> int:
        client_id = super()._get_next_client_id()
        if self._num_frontends > 1:
            # Lane 0 follows the same namespace rule as attached frontends
            # so a long-lived counter can never bleed into the frontend-id
            # bits (a no-op re-encode until the counter wraps).
            client_id = namespace_client_id(0, client_id)
        return client_id

    def _cleanup_multi_frontend_ipc_dir(self):
        """Remove the launcher-owned multi-frontend ipc directory.

        Unlinking ipc socket paths does not disturb established zmq
        connections; it only prevents new connects.
        """
        ipc_dir = getattr(self, "_multi_frontend_ipc_dir", None)
        if ipc_dir is not None:
            shutil.rmtree(ipc_dir, ignore_errors=True)
            self._multi_frontend_ipc_dir = None

    def pre_shutdown(self) -> None:
        # The monitor may already be notifying workers. A concurrent closer
        # must not tear down IPC until the notification phase finishes.
        with self._pre_shutdown_lock:
            if not self._pre_shutdown_complete:
                self._pre_shutdown()
                self._pre_shutdown_complete = True

    def shutdown(self) -> None:
        """Reclaim this proxy's owned workers, threads and transports.

        Cleanup is serialized across callers. A cleanup failure permits a
        later shutdown attempt, not new generation requests. Once cleanup
        succeeds, subsequent shutdown calls are no-ops. Worker failures can
        still be reported after local cleanup completes.

        Raises:
            TimeoutError: An owned resource did not stop within its shutdown
                budget. Retry from an external thread to finish cleanup.
            RuntimeError: Shutdown would join its own resource-owning thread.
            EngineDeadError: A worker failure is reported after cleanup.
        """
        # Serializes finalizers only; worker callbacks never need this lock
        # to make progress. Mark closed only after all owned cleanup succeeds.
        with self._shutdown_lock:
            if self._shutdown_complete:
                return
            had_workers = self.workers_started
            self._shutdown()
            self._shutdown_complete = True
        if had_workers:
            self._handle_background_error()

    def _finish_pending_requests(self) -> None:
        # The reader has exited. Either the death sweep owns the notifications,
        # or shutdown takes ownership before a later death can snapshot them.
        with self._admission_lock:
            results = [] if self._engine_dead else list(self._results.items())
            self._results.clear()
        for client_id, result in results:
            try:
                result.queue.put(
                    ErrorResponse(client_id, 'Executor proxy shut down',
                                  client_id))
            except AsyncQueue.EventLoopShutdownError:
                # The result is queued, but its consumer loop can no longer be notified.
                continue

    def _close_ipc_queues(self) -> None:
        for name in ('request_queue', 'worker_init_status_queue',
                     'result_queue', '_resource_governor_queue',
                     'profile_ack_queue'):
            ipc_queue = getattr(self, name, None)
            if ipc_queue is not None:
                ipc_queue.close()
        self._cleanup_multi_frontend_ipc_dir()

    def _shutdown(self) -> None:
        self.pre_shutdown()
        if not self.workers_started:
            self._close_ipc_queues()
            return

        logger_debug('Proxy.shutdown...\n', "yellow")

        # An abruptly-killed worker world (MPI_Abort, SIGKILL, OOM) never
        # completes its mpi4py futures: give them one short collective grace
        # instead of blocking on each, and skip the ones still pending.
        if self._engine_dead:
            concurrent.futures.wait(self.mpi_futures, timeout=5.0)
        else:
            # A worker that dies *after* delivering its results (e.g. a segfault
            # in its own teardown) queues no error, so _engine_dead is still
            # False. Bound the wait with one deadline shared by all futures so
            # that cannot wedge shutdown; pre_shutdown has already asked the
            # workers to quit.
            _, pending = concurrent.futures.wait(
                self.mpi_futures, timeout=self._WORKER_SHUTDOWN_TIMEOUT_S)
            if pending:
                stuck = [
                    i for i, f in enumerate(self.mpi_futures) if f in pending
                ]
                logger.warning(
                    f"MPI worker future(s) {stuck} did not complete within "
                    f"{self._WORKER_SHUTDOWN_TIMEOUT_S}s of shutdown; treating "
                    "the worker world as dead")
                # Route the rest of teardown through the dead-engine path:
                # skip the stuck futures, bound the dispatcher join and
                # abandon (rather than join) the session.
                self._mark_engine_dead(
                    TimeoutError(
                        f"MPI worker future(s) {stuck} timed out on shutdown"))

        for f in self.mpi_futures:
            if self._engine_dead and not f.done():
                continue
            try:
                f.result()
            except:
                # The errors are already captured in mpi_done_callback, ignored
                # here
                pass

        monitor = getattr(self, '_error_monitor_thread', None)
        if monitor is not None and monitor.is_alive():
            if threading.current_thread() is monitor:
                raise RuntimeError('The error monitor cannot join itself')
            monitor.join(timeout=5)
            if monitor.is_alive():
                raise TimeoutError('Executor error monitor has not stopped')
        self._close_request_sender()

        self._stop_dispatch_thread(drain=not self._engine_dead)
        self._finish_pending_requests()

        self._close_profile_control()

        # step3: finish all remaining work

        # close the RPC client
        if self.rpc_client is not None:
            self.rpc_client.close()
            self.rpc_client = None

        self._close_ipc_queues()

        if self._owns_mpi_session:
            if self._engine_dead:
                # Anything joining a dead worker world blocks forever.
                self.mpi_session.abandon()
            else:
                self.mpi_session.shutdown()

        self.workers_started = False

        if enable_llm_debug():
            print_alive_threads()

    def submit(self, request: GenerationRequest) -> GenerationResult:
        """Low-level API to the executor.

        Returns a "future" GenerationResult which can be waited.
        Forwards the request to the workers through the request queue.
        """

        # Sticky fast-fail: don't accept new work once the engine is known dead.
        if self._engine_dead or self._fatal_error is not None:
            raise EngineDeadError(self._fatal_error)

        self._start_dispatch_threads()

        request.set_id(self._get_next_client_id())
        logprob_params = self._get_logprob_params(request)

        result = GenerationResult(
            request,
            background_error_handler=self._handle_background_error,
            executor=self,
            disaggregated_params=request.disaggregated_params,
            logprob_params=logprob_params)
        with self._admission_lock:
            if self._engine_dead or self._fatal_error is not None:
                raise EngineDeadError(self._fatal_error)
            if self.doing_shutdown:
                raise RuntimeError('The executor proxy is shutting down')
            self._results[request.id] = result

        # Close the submit-vs-death race: _mark_engine_dead() runs on the
        # error-monitor thread and its one-shot sweep snapshots _results, so a
        # result registered just after that snapshot would never be unblocked
        # and would hang forever on a dead worker. Re-check after registering
        # and fail fast if the engine died in that window.
        if self._engine_dead or self._fatal_error is not None:
            self._results.pop(request.id, None)
            raise EngineDeadError(self._fatal_error)

        # Requests that read their full prompt keep it (see prompt_tail.py).
        if self._prompt_tail is not None:
            tail = self._prompt_tail(request)
            if tail:
                request._prompt_tail_tokens = tail

        with nvtx_range_debug("request_queue.put"):
            try:
                self._send_generation_request(request)
            except RequestSendCancelled as error:
                self._results.pop(request.id, None)
                if self._engine_dead or self._fatal_error is not None:
                    raise EngineDeadError(self._fatal_error) from error
                raise
            except (RuntimeError, zmq.ZMQError):
                self._results.pop(request.id, None)
                raise

        self._handle_background_error()

        return result

    def collective_rpc(
        self,
        method: str,
        args: tuple = (),
        kwargs: Optional[dict] = None,
        non_block: bool = False,
        unique_reply_rank: Optional[int] = None,
        target_ranks: Optional[Union[int, List[int]]] = None,
    ) -> List:
        """Execute a method call on the rank-0 GPU worker via the RPC client.

        Rank-0 RPC shim used for uniform dispatch from
        :meth:`~tensorrt_llm.llmapi.llm.LLM._collective_rpc`.  Most methods
        require ``model_world_size == 1``; ``sleep`` and ``wakeup`` are
        explicitly allowed for multi-rank deployments because
        :meth:`~tensorrt_llm.executor.base_worker.BaseWorker.sleep` and
        :meth:`~tensorrt_llm.executor.base_worker.BaseWorker.wakeup` handle
        cross-rank coordination internally via the control communicator.

        Args:
            method: Name of the worker method to invoke.
            args: Positional arguments forwarded to the worker method.
            kwargs: Keyword arguments forwarded to the worker method.
            non_block: If ``True``, return a ``Future`` without waiting.
            unique_reply_rank: Must be ``None``.
            target_ranks: Must be ``None``.

        Returns:
            A list containing the single return value when ``non_block=False``,
            or a list containing the pending
            :class:`~concurrent.futures.Future` when ``non_block=True``.

        Raises:
            RuntimeError: If the RPC client has not been initialised yet.
            NotImplementedError: If ``model_world_size > 1`` and the method
                is not in the allowed-methods set (currently ``sleep`` and
                ``wakeup``), or if ``unique_reply_rank`` or ``target_ranks``
                are provided.
        """
        if self.rpc_client is None:
            raise RuntimeError(
                "RPC client is not initialised — collective_rpc() cannot be "
                "called before the executor workers have started.")
        _check_collective_rpc_guard(self.model_world_size,
                                    unique_reply_rank,
                                    target_ranks,
                                    method=method)
        kwargs = kwargs or {}
        remote_call = getattr(self.rpc_client, method)(*args, **kwargs)
        if non_block:
            return [remote_call.remote_future()]
        return [remote_call.remote()]

    def get_stats(self, timeout: float) -> List[dict]:
        """Get iteration statistics from the runtime via RPC.

        Args:
            timeout (float): Max wait time in seconds for the RPC call.

        Returns:
            List[dict]: A list of runtime stats as dict.
        """
        if self.rpc_client is None:
            logger.warning("RPC client not initialized, cannot get stats")
            return []

        # Allow transport overhead beyond the worker's queue wait, while
        # bounding the client even if the worker never replies.
        stats = self.rpc_client.fetch_stats_wait_async(timeout=timeout).remote(
            timeout=None if timeout is None else timeout + 1.0)
        return [json.loads(s) if isinstance(s, str) else s for s in stats]

    def get_kv_cache_capacity(self) -> dict:
        """Get static primary/GPU KV cache capacity from the runtime via RPC."""
        if self.rpc_client is None:
            logger.warning(
                "RPC client not initialized, cannot get kv cache capacity")
            return {}

        try:
            capacity = self.rpc_client.fetch_kv_cache_capacity_async().remote(
                timeout=1.0)
            if isinstance(capacity, str):
                capacity = json.loads(capacity)
            return capacity if isinstance(capacity, dict) else {}
        except (RPCError, json.JSONDecodeError) as e:
            logger.debug(f"Error fetching kv cache capacity via RPC: {e}")
            return {}

    def get_openengine_discovery(self) -> dict:
        """Get startup OpenEngine discovery through its dedicated worker RPC."""
        if self.rpc_client is None:
            return {}
        try:
            discovery = self.rpc_client.fetch_openengine_discovery_async(
            ).remote(timeout=1.0)
            if isinstance(discovery, str):
                discovery = json.loads(discovery)
            return discovery if isinstance(discovery, dict) else {}
        except (RPCError, json.JSONDecodeError) as error:
            logger.debug(
                f"Error fetching OpenEngine discovery via RPC: {error}")
            return {}

    def get_kv_cache_load(self) -> dict:
        """Get a primary/GPU KV cache load snapshot via RPC."""
        if self.rpc_client is None:
            logger.warning(
                "RPC client not initialized, cannot get kv cache load")
            return {}

        try:
            load = self.rpc_client.fetch_kv_cache_load_async().remote(
                timeout=1.0)
            if isinstance(load, str):
                load = json.loads(load)
            return load if isinstance(load, dict) else {}
        except (RPCError, json.JSONDecodeError) as e:
            logger.debug(f"Error fetching kv cache load via RPC: {e}")
            return {}

    def get_disaggregated_params(self) -> dict:
        """Get disaggregated params from worker runtime via RPC."""
        if self.rpc_client is None:
            logger.warning(
                "RPC client not initialized, cannot get disaggregated params")
            return {}
        try:
            params = self.rpc_client.get_disaggregated_params().remote()
            return params if isinstance(params, dict) else {}
        except RPCError as e:
            logger.warning(f"Error fetching disaggregated params via RPC: {e}")
            return {}

    def get_data_transceiver_state(self) -> bytes:
        """Get serialized DataTransceiverState from worker runtime via RPC."""
        if self.rpc_client is None:
            return b""
        try:
            return self.rpc_client.get_data_transceiver_state().remote()
        except RPCError as e:
            logger.error(f"Error fetching data transceiver state via RPC: {e}")
            raise

    def get_startup_metrics(self) -> dict | None:
        """Get rank-0 startup metrics, or ``None`` if the RPC is unavailable."""
        if self.rpc_client is None:
            logger.warning(
                "RPC client not initialized, cannot get startup metrics")
            return None
        try:
            metrics = self.rpc_client.get_startup_metrics().remote()
            return metrics if isinstance(metrics, dict) else None
        except RPCError as e:
            logger.warning(f"Error fetching startup metrics via RPC: {e}")
            return None

    def get_engine_config(self) -> dict:
        """Get rank-0 resolved engine config metadata via RPC."""
        if self.rpc_client is None:
            return {}
        try:
            config = self.rpc_client.get_engine_config().remote()
            return config if isinstance(config, dict) else {}
        except RPCError as e:
            logger.debug(f"Error fetching engine config via RPC: {e}")
            return {}

    def aget_stats(self, timeout: float) -> IterationResult:
        """Get iteration statistics from the runtime via RPC (async).

        Args:
            timeout (float): Max wait time in seconds for the RPC call.

        Returns:
            IterationResult: An async iterable object containing runtime stats.
        """
        # Initialize iteration result if needed
        self._maybe_initialize_iteration_results()

        if self._iter_stats_result is None:
            logger.warning("Iteration statistics are not available yet.")
            from .executor import empty_async_iterable
            return empty_async_iterable()

        # Fetch stats via RPC and populate the result
        try:
            stats = self.rpc_client.fetch_stats_wait_async(
                timeout=timeout).remote(
                    timeout=None if timeout is None else timeout + 1.0)
        except Exception as e:
            logger.debug(f"Error fetching stats via RPC: {e}")
            stats = []

        for stat in stats:
            self._iter_stats_result.queue.put(stat)

        self._iter_stats_result.set_timeout(timeout)
        return self._iter_stats_result

    def get_kv_events(self, timeout: float) -> List[dict]:
        """Get iteration KV events from the runtime via RPC.

        Args:
            timeout (float): Max wait time in seconds for the RPC call.

        Returns:
            List[dict]: A list of runtime events as dict.
        """
        if self.rpc_client is None:
            logger.warning("RPC client not initialized, cannot get kv events")
            return []

        try:
            events = self.rpc_client.fetch_kv_cache_events_wait_async(
                timeout=timeout).remote()
            return [json.loads(e) if isinstance(e, str) else e for e in events]
        except Exception as e:
            logger.error(f"Error fetching kv events via RPC: {e}")
            return []

    def aget_kv_events(self, timeout: float) -> IterationResult:
        """Get iteration KV events from the runtime via RPC (async).

        Args:
            timeout (float): Max wait time in seconds for the RPC call.

        Returns:
            IterationResult: An async iterable object containing runtime events.
        """
        # Initialize iteration result if needed
        self._maybe_initialize_iteration_results()

        if self._iter_kv_events_result is None:
            from .executor import empty_async_iterable
            return empty_async_iterable()

        # Fetch kv events via RPC and populate the result
        try:
            events = self.rpc_client.fetch_kv_cache_events_wait_async(
                timeout=timeout).remote()
        except Exception as e:
            logger.debug(f"Error fetching kv events via RPC: {e}")
            events = []

        for event in events:
            self._iter_kv_events_result.queue.put(event)

        self._iter_kv_events_result.set_timeout(timeout)
        return self._iter_kv_events_result

    def __del__(self):
        # Daemon threads can no longer service close requests during late
        # interpreter finalization. Explicit shutdown has a stronger contract.
        if hasattr(self, '_shutdown_lock') and not self._is_finalizing():
            self.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.shutdown()
        return False  # propagate the exception


class GenerationExecutorFrontendProxy(GenerationExecutorProxy):
    """An attached serving frontend for the classic IPC executor path.

    Used for multi-frontend serving (num_serve_frontends > 1).
    PUSH-connects to the request ingress bound by the rank0 worker and binds
    its own per-frontend result lane (PULL); the worker routes responses to
    this lane by the frontend id embedded in the top bits of client_id (see
    utils.namespace_client_id). It never owns the engine: no MPI
    session, no worker launch, and shutdown never emits the worker's None
    shutdown sentinel -- that right is the launcher frontend's alone.
    """

    def __init__(
        self,
        attach_info: dict,
        *,
        frontend_id: int,
        postproc_worker_config: Optional[PostprocWorkerConfig] = None,
        is_llm_executor: Optional[bool] = None,
    ) -> None:
        self._init_lifecycle()
        num_lanes = len(attach_info["result_addrs"])
        if not 0 < frontend_id < num_lanes:
            raise ValueError(
                f"frontend_id {frontend_id} out of range: attached frontends "
                f"use ids 1..{num_lanes - 1} (id 0 is the launcher)")
        postproc_worker_config = postproc_worker_config or PostprocWorkerConfig(
        )
        # Deliberately skip GenerationExecutorProxy.__init__: it creates an
        # MPI session, launches workers, and registers the pre_shutdown
        # atexit hook that emits the engine shutdown sentinel.
        GenerationExecutor.__init__(
            self,
            num_postprocess_workers=postproc_worker_config.
            num_postprocess_workers,
            postprocess_tokenizer_dir=postproc_worker_config.
            postprocess_tokenizer_dir,
            is_llm_executor=is_llm_executor)

        # State consumed by methods inherited from GenerationExecutorProxy
        # (submit / check_health / collective_rpc). The engine lives with
        # the launcher: there are no local MPI workers, so the monitor
        # stays empty and worker death reaches this frontend through its
        # result lane / error queue instead.
        self.model_world_size = attach_info.get("model_world_size", 1)

        self._frontend_id = frontend_id
        self._num_frontends = num_lanes
        # The launcher's validated prompt-tail setting.
        prompt_tail = attach_info.get("prompt_tail_tokens", 0)
        if prompt_tail:
            self._prompt_tail = PromptTailPolicy(prompt_tail)
            logger.info(f"prompt tail trimming on, tail={prompt_tail} "
                        f"(frontend {frontend_id})")
        self.garbage_collection_gen0_threshold = None
        # Must be None: OpenAIServer reads the resource_governor_queue
        # property at init; the governor lives with the launcher only.
        self._resource_governor_queue = None

        hmac_key = bytes.fromhex(attach_info["hmac_key"])
        self.request_queue = IpcQueue(
            (attach_info["request_addr"], hmac_key),
            is_server=False,
            socket_type=zmq.PUSH,
            name=f"frontend_{frontend_id}_request_queue")
        self._request_sender = _create_request_sender(
            self.request_queue, self._error_queue, self._request_send_stopped)
        self.result_queue = FusedIpcQueue(
            (attach_info["result_addrs"][frontend_id], hmac_key),
            is_server=True,
            fuse_message=False,
            socket_type=zmq.PULL,
            recv_timeout=self._RESULT_RECV_TIMEOUT_S,
            name=f"frontend_{frontend_id}_result_queue")

        # Stats / KV events / disagg params share the rank0 worker's stats
        # RPC server with the launcher (ROUTER socket, multi-client).
        self.rpc_client: Optional[RPCClient] = None
        if attach_info.get("rpc_addr"):
            self.rpc_client = RPCClient(attach_info["rpc_addr"],
                                        hmac_key=bytes.fromhex(
                                            attach_info["rpc_hmac_key"]))

    def _get_next_client_id(self) -> int:
        # Embed the frontend id in the top bits so the worker routes the
        # responses back to this frontend's result lane.
        return namespace_client_id(self._frontend_id,
                                   super()._get_next_client_id())

    def check_health(self) -> bool:
        """Health contract of an attached frontend.

        An attached frontend owns no workers, so there is no process or
        MPI-future liveness to poll: it is healthy while no fatal error
        has been recorded and shutdown has not begun. Engine death
        reaches it through the per-lane result socket / dispatch-thread
        error path, which records the fatal error checked here.
        """
        if self.doing_shutdown or self._fatal_error is not None:
            return False

        if self._drain_error_queue():
            return self._fatal_error is None and not self.doing_shutdown

        return True

    def _request_send_stopped(self) -> bool:
        # A local fatal error does not establish that the shared engine is dead.
        # Keep this lane's cancellation traffic bounded by its shutdown budget.
        return (self._request_transport_closed
                or (self._request_shutdown_deadline is not None
                    and time.monotonic() >= self._request_shutdown_deadline))

    def _pre_shutdown(self) -> None:
        with self._admission_lock:
            self.doing_shutdown = True
        # Cancel only this frontend's requests; the launcher owns engine quit.
        self._drain_control_and_requests(cancel_requests=True,
                                         send_sentinel=False)

    def _shutdown(self) -> None:
        self.pre_shutdown()
        # Send the cancels queued by pre_shutdown before the lane goes idle.
        self._close_request_sender()
        self._stop_dispatch_thread(drain=False)
        self._finish_pending_requests()
        if self.rpc_client is not None:
            self.rpc_client.close()
            self.rpc_client = None
        self._worker_process_monitor.close()
        self._close_ipc_queues()
