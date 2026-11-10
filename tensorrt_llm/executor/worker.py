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

import gc
import os
import pickle  # nosec B403
import sys
import threading
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Optional

import zmq

from tensorrt_llm.logger import logger

from .._utils import mpi_comm, mpi_rank, print_all_stacks
from ..llmapi._load_balance_env import (LAUNCH_QUEUES_ENV,
                                        configure_moe_launch_queues,
                                        uses_collective_shutdown)
from ..llmapi.llm_args import BaseLlmArgs
from ..llmapi.mpi_session import set_mpi_session_cpp
from ..llmapi.tokenizer import TokenizerBase
from ..llmapi.tracer import VizTracer, set_global_tracer
from ..llmapi.utils import ManagedThread, logger_debug, print_traceback_on_error
from .base_worker import BaseWorker, _init_hf_modules
from .ipc import FusedIpcQueue, IpcQueue
from .postproc_worker import (PostprocWorker, PostprocWorkerConfig,
                              postproc_worker_main)
from .request import (CancellingRequest, GenerationRequest, StartProfileRequest,
                      StopProfileRequest)
from .rpc_worker_mixin import RpcWorkerMixin
from .utils import (ErrorResponse, IntraProcessQueue, RequestError,
                    WorkerCommIpcAddrs)
from .worker_process_monitor import capture_worker_process_identity

__all__ = [
    "GenerationExecutorWorker",
]

# Home of the architecture registries, imported on demand: a worker that never
# touches the PyTorch model zoo should not pay for it.
_MODELING_UTILS_MODULE = "tensorrt_llm._torch.models.modeling_utils"

# Probed (not imported) in shutdown() so a non-MegaMoE run does not pull in
# the MoE stack just to release an empty cache. Must stay in sync with the
# real module path -- a mismatch silently degrades the release to a no-op.
_MEGA_MOE_DEEPGEMM_MODULE = (
    "tensorrt_llm._torch.moe.fused_moe.mega_moe.mega_moe_deepgemm")


class GenerationExecutorWorker(RpcWorkerMixin, BaseWorker):

    def __init__(
        self,
        postproc_worker_config: Optional[PostprocWorkerConfig] = None,
        is_llm_executor: Optional[bool] = None,
        hf_model_dir: Optional[Path] = None,
        tokenizer: Optional[TokenizerBase] = None,
        llm_args: Optional[BaseLlmArgs] = None,
        rpc_addr: Optional[str] = None,
        hmac_key: bytes = b"",
    ) -> None:
        super().__init__(
            postproc_worker_config=postproc_worker_config,
            is_llm_executor=is_llm_executor,
            hf_model_dir=hf_model_dir,
            tokenizer=tokenizer,
            llm_args=llm_args,
        )
        self.rpc_server = None

        if (self.llm_args is not None
                and getattr(self.llm_args, "enable_resource_governor", False)):
            self._resource_governor_queue = IntraProcessQueue()

        self.setup_engine()

        # Setup RPC server for stats (skip init_rpc_worker to keep IPC response queue)
        # Only set up if rpc_addr is provided (for stats RPC support)
        if rpc_addr is not None:
            assert hmac_key, "hmac_key is required when rpc_addr is set"
            self.rpc_addr = rpc_addr
            self.hmac_key = hmac_key
            self.start_rpc_server()  # Reuse from RpcWorkerMixin

        self.await_response_thread = ManagedThread(
            self.await_response_task,
            error_queue=self._error_queue,
            name="await_response_thread")

    def collective_rpc(
        self,
        method: str,
        args: tuple = (),
        kwargs: Optional[dict] = None,
        non_block: bool = False,
        unique_reply_rank: Optional[int] = None,
        target_ranks: int | list[int] | None = None,
    ) -> list:
        """Call a worker method directly and return its result in a list.

        Only blocking calls without rank selection are supported. Runtime-memory
        methods coordinate multi-rank execution internally; other methods require
        a single-rank deployment, as with the MPI proxy dispatchers.
        """
        from .proxy import _check_collective_rpc_guard

        _check_collective_rpc_guard(self.llm_args.parallel_config.world_size,
                                    unique_reply_rank,
                                    target_ranks,
                                    method=method)
        if non_block:
            raise NotImplementedError(
                "GenerationExecutorWorker collective_rpc only supports blocking calls."
            )
        return [getattr(self, method)(*args, **(kwargs or {}))]

    def start_thread(self, thread: ManagedThread):
        if not self.engine.can_enqueue_requests():
            return
        if thread.is_alive():
            return
        if thread.ident is not None:
            # Already exited: either stop() at shutdown (nothing to surface) or
            # an engine event-loop crash, where restarting masks it into a peer
            # MPI-collective hang. Wrap, since start() runs on every submit().
            err = getattr(self.engine, "_event_loop_error", None)
            if err is not None:
                raise RequestError(str(err)) from err
            return
        thread.start()

    def await_response_task(self) -> bool:
        return self._await_response_helper()

    def start(self):
        # Stats and KV events are now fetched on-demand via RPC,
        # so we only need to start the response thread
        self.start_thread(self.await_response_thread)

    def shutdown(self):

        if self.doing_shutdown:
            return
        if not uses_collective_shutdown(getattr(self, "llm_args", None)):
            self._shutdown_rank_local()
            return
        self.doing_shutdown = True
        shutdown_succeeded = False

        try:
            logger_debug(f'Worker {mpi_rank()} shutdown...\n', "yellow")

            shutdown_error = None
            # The server references this worker; a long-lived worker process would
            # otherwise keep one server, its threads and sockets per LLM it hosted.
            if self.rpc_server is not None:
                try:
                    self.rpc_server.shutdown()
                    self.rpc_server = None
                except BaseException as error:
                    shutdown_error = error
            if self.engine is not None:
                engine = self.engine
                local_safe = bool(
                    getattr(self, "_engine_shutdown_completed", False))
                if not local_safe:
                    try:
                        if engine.can_enqueue_requests():
                            if self.await_response_thread.is_alive():
                                self.await_response_thread.stop()
                                self.await_response_thread.join()

                        engine.shutdown()
                        self._engine_shutdown_completed = True
                        local_safe = True
                    except BaseException as error:
                        shutdown_error = error
                try:
                    terminal_cleanup = getattr(engine, "terminal_cleanup", None)
                    if terminal_cleanup is not None:
                        if getattr(engine, "shutdown_all_ranks", False):
                            terminal_cleanup(local_safe=local_safe)
                        elif local_safe:
                            terminal_cleanup()
                except BaseException as error:
                    if shutdown_error is None:
                        shutdown_error = error

                if (self.llm_args.backend == "pytorch"
                        and hasattr(self, "checkpoint_loader")
                        and self.checkpoint_loader is not None
                        and shutdown_error is None):
                    try:
                        self.checkpoint_loader.cleanup()
                    except BaseException as error:
                        shutdown_error = error
                    else:
                        self.checkpoint_loader = None

            # Keep every rank aligned before entering the next collective phase.
            # A rank-local engine or checkpoint failure must prevent peers from
            # advancing to symmetric-buffer release and process-group teardown.
            shutdown_failures = mpi_comm().allgather(
                None if shutdown_error is
                None else f"{type(shutdown_error).__name__}: {shutdown_error}")
            shutdown_failures = [
                f"rank {rank}: {failure}"
                for rank, failure in enumerate(shutdown_failures)
                if failure is not None
            ]
            if shutdown_failures:
                if shutdown_error is not None:
                    raise shutdown_error
                raise RuntimeError("Worker shutdown failed: " +
                                   "; ".join(shutdown_failures))

            # MegaMoE's NVLink symmetric-memory activation workspaces are
            # rendezvoused over the EP group, so they must go before the
            # destroy_process_group() below.
            mega_moe = sys.modules.get(_MEGA_MOE_DEEPGEMM_MODULE)
            symm_release_error = None
            if mega_moe is not None:
                try:
                    mega_moe.release_symm_buffer_cache()
                except BaseException as error:
                    symm_release_error = error
            # No rank may destroy its process group while a peer still needs that
            # group to retry symmetric-buffer destruction.
            symm_release_failures = mpi_comm().allgather(
                None if symm_release_error is None else
                f"{type(symm_release_error).__name__}: {symm_release_error}")
            symm_release_failures = [
                f"rank {rank}: {failure}"
                for rank, failure in enumerate(symm_release_failures)
                if failure is not None
            ]
            if symm_release_failures:
                if symm_release_error is not None:
                    raise symm_release_error
                raise RuntimeError("MegaMoE symm-buffer release failed: " +
                                   "; ".join(symm_release_failures))

            # Destroy torch distributed process groups so that NCCL communicators
            # are torn down cleanly before MPI session shutdown and process exit.
            # This is done here (not in PyExecutor.shutdown()) because the MPI
            # worker owns the process group. In the Ray path the process group
            # belongs to RayWorkerWrapper and must not be destroyed by the engine.
            import torch.distributed
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()

            # Return this rank's GPU memory to the driver. Under an external MPI
            # launch (mpirun/srun, e.g. CI), the worker process is long-lived and
            # shared across successive LLM instances: a new GenerationExecutorWorker
            # is built for each LLM, but the OS process -- and with it the CUDA
            # context and PyTorch caching allocator -- persists.
            gc.collect()
            torch.cuda.empty_cache()

            self.engine = None
            self._engine_shutdown_completed = False
            shutdown_succeeded = True
            logger_debug(f"Worker {mpi_rank()} shutdown done.\n", "yellow")

            # Raise background-thread errors only after the commit above, which
            # every rank has reached, so a retry cannot re-enter collectives.
            self._handle_background_error()
        finally:
            # A failed collective shutdown can be retried only explicitly.
            self._collective_shutdown_failed = not shutdown_succeeded
            if not shutdown_succeeded:
                self.doing_shutdown = False

    def _shutdown_rank_local(self):
        """Best-effort teardown for configurations without collective EP resources."""
        self.doing_shutdown = True

        logger_debug(f'Worker {mpi_rank()} shutdown...\n', "yellow")

        # The server references this worker; a long-lived worker process would
        # otherwise keep one server, its threads and sockets per LLM it hosted.
        if self.rpc_server is not None:
            self.rpc_server.shutdown()
            self.rpc_server = None

        if self.engine is not None:
            if self.engine.can_enqueue_requests():
                if self.await_response_thread.is_alive():
                    self.await_response_thread.stop()
                    self.await_response_thread.join()

            self.engine.shutdown()
            local_cleanup = getattr(self.engine, "local_cleanup", None)
            if local_cleanup is not None:
                local_cleanup()
            self.engine = None

            if (self.llm_args.backend == "pytorch"
                    and hasattr(self, "checkpoint_loader")
                    and self.checkpoint_loader is not None):
                self.checkpoint_loader.cleanup()
                self.checkpoint_loader = None

        # MegaMoE's NVLink symmetric-memory activation workspaces are
        # rendezvoused over the EP group, so they must go before the
        # destroy_process_group() below. Never let a failure here escape:
        # doing_shutdown is already set, so a retry would no-op, and skipping
        # the teardown below would leave NCCL communicators alive -- turning a
        # MegaMoE-only failure into a worker teardown hang.
        mega_moe = sys.modules.get(_MEGA_MOE_DEEPGEMM_MODULE)
        if mega_moe is not None:
            try:
                mega_moe.release_symm_buffer_cache()
            except Exception as e:
                logger.error(
                    f"Failed to release MegaMoE symm buffers on shutdown: {e}")

        # Destroy torch distributed process groups so that NCCL communicators
        # are torn down cleanly before MPI session shutdown and process exit.
        # This is done here (not in PyExecutor.shutdown()) because the MPI
        # worker owns the process group. In the Ray path the process group
        # belongs to RayWorkerWrapper and must not be destroyed by the engine.
        import torch.distributed
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

        # Return this rank's GPU memory to the driver. Under an external MPI
        # launch (mpirun/srun, e.g. CI), the worker process is long-lived and
        # shared across successive LLM instances, so the previous model's
        # reservation would otherwise carry into the next LLM in this process.
        gc.collect()
        torch.cuda.empty_cache()

        # Check if there are any errors from the threads before shutdown.
        self._handle_background_error()

        logger_debug(f"Worker {mpi_rank()} shutdown done.\n", "yellow")

    def block_subordinates(self):
        if self.rank != 0:
            from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
            if isinstance(self.engine, PyExecutor):
                self.engine.wait_shutdown()


def _incoming_requests(request_queue: IpcQueue):
    """Yield the proxy's requests until its shutdown sentinel arrives.

    With TRTLLM_PROXY_BATCH_REQUESTS=1 the proxy sends a burst of requests as
    one list of pickled requests. ``request_queue.get()`` has already
    authenticated that message, and with it every payload inside.
    """
    while (msg := request_queue.get()) is not None:
        if isinstance(msg, list):
            for payload in msg:
                yield pickle.loads(payload)  # nosec B301
        else:
            yield msg


@print_traceback_on_error
def worker_main(
    worker_queues: WorkerCommIpcAddrs,
    log_level: str,
    worker_cls: type = GenerationExecutorWorker,
    tracer_init_kwargs: Optional[dict] = None,
    _torch_external_model_modules: Optional[dict] = None,
    postproc_worker_config: Optional[PostprocWorkerConfig] = None,
    ready_signal: Optional[str] = None,
    worker_process_identities_signal: Optional[bytes] = None,
    is_llm_executor: Optional[
        bool] = True,  # whether it's the main executor instance
    hf_model_dir: Optional[Path] = None,
    tokenizer: Optional[TokenizerBase] = None,
    llm_args: Optional[BaseLlmArgs] = None,
    rpc_addr: Optional[str] = None,
    hmac_key: bytes = b"",
) -> None:

    launch_queues = os.environ.get(LAUNCH_QUEUES_ENV)
    if llm_args is not None and llm_args.env_overrides:
        # MPI may have cached the parent's environment before this worker was
        # spawned, so replay overrides before any rebalance state is inspected.
        os.environ.update(llm_args.env_overrides)

    # Configure per-layer dynamic EPLB before MPI or CUDA state is created.
    if llm_args is not None and llm_args.backend == "pytorch":
        queue_overrides = configure_moe_launch_queues(
            getattr(llm_args, "moe_config", None),
            llm_args.env_overrides,
            initial_setting=launch_queues)
        if queue_overrides is not llm_args.env_overrides:
            llm_args.env_overrides = queue_overrides

    def _print_stacks():
        counter = 0
        while True:
            time.sleep(print_stacks_period)
            counter += 1
            logger.error(f"Printing stacks {counter} times")
            print_all_stacks()

    print_stacks_period = int(
        os.getenv("TRTLLM_WORKER_PRINT_STACKS_PERIOD", "-1"))
    if print_stacks_period > 0:
        print_stacks_thread = threading.Thread(target=_print_stacks,
                                               daemon=True)
        print_stacks_thread.start()

    mpi_comm().barrier()

    if llm_args is not None and llm_args.trust_remote_code:
        _init_hf_modules()

    logger_debug(f"Worker {mpi_rank()} entering worker_main...\n", "green")

    result_queue: Optional[IpcQueue] = None
    result_queues: Optional[List[IpcQueue]] = None
    resource_governor_queue: Optional[IpcQueue] = None

    postproc_worker_config = postproc_worker_config or PostprocWorkerConfig()

    is_leader: bool = mpi_rank() == 0
    # Multi-frontend serving: the worker binds the request ingress (PULL)
    # and pushes responses to per-frontend result lanes.
    multi_frontend_addrs = worker_queues.frontend_result_queue_addrs
    frontend_result_queues: Optional[List[FusedIpcQueue]] = None
    if tracer_init_kwargs is not None and is_leader:
        tracer = VizTracer(**tracer_init_kwargs)
        tracer.register_exit()
        tracer.start()
        set_global_tracer(tracer)

    # Architectures the driver registered from outside the built-in zoo, as
    # module names. Declaring them leaves this process's zoo lazy: the one
    # module behind an architecture is imported when that architecture is
    # looked up, not because the driver happened to have imported it. An empty
    # declaration is still worth making once the registries exist, so a worker
    # reused by a driver without custom modules releases the previous driver's
    # providers; before that there is nothing registered to release.
    if _torch_external_model_modules or _MODELING_UTILS_MODULE in sys.modules:
        from tensorrt_llm._torch.models.modeling_utils import \
            register_external_model_modules
        register_external_model_modules(_torch_external_model_modules or {})

    set_mpi_session_cpp(mpi_comm())

    if is_leader:
        # Only set the log level for the leader process, the other processes will
        # inherit the log level from "TLLM_LOG_LEVEL" environment variable
        logger.set_level(log_level)
        request_queue = IpcQueue(worker_queues.request_queue_addr,
                                 is_server=multi_frontend_addrs is not None,
                                 socket_type=zmq.PULL if multi_frontend_addrs
                                 is not None else zmq.PAIR,
                                 name="worker_request_queue")
        worker_init_status_queue = IpcQueue(
            worker_queues.worker_init_status_queue_addr,
            is_server=False,
            socket_type=zmq.DEALER,
            name="worker_init_status_queue")
        resource_governor_queue = IpcQueue(
            worker_queues.resource_governor_queue_addr,
            is_server=False,
            name="worker_resource_governor_queue"
        ) if worker_queues.resource_governor_queue_addr else None
        # Synchronous ack channel back to the proxy for control requests
        # (start_profile / stop_profile). Connected on first use rather
        # than at startup: the proxy only binds its end when profiling is
        # actually requested, and it passes the address on the request,
        # so a deployment that never profiles never creates this socket.
        profile_ack_queue = None

        def send_profile_ack(kind: str, err, req) -> None:
            nonlocal profile_ack_queue
            addr = getattr(req, "ack_addr", None)
            if addr is None:
                return
            if profile_ack_queue is None:
                profile_ack_queue = IpcQueue(addr,
                                             is_server=False,
                                             name="worker_profile_ack_queue")
            profile_ack_queue.put((kind, err))

        if postproc_worker_config.enabled:
            # IPC queues for sending inputs to the postprocess parallel
            # processes, each one is a PAIR zmq socket
            result_queues = [
                FusedIpcQueue(is_server=True,
                              fuse_message=False,
                              name=f"postprocess_{i}_feedin_queue")
                for i in range(postproc_worker_config.num_postprocess_workers)
            ]
        elif multi_frontend_addrs is not None:
            # One PUSH lane per frontend (see base_worker._send_rsp).
            frontend_result_queues = [
                FusedIpcQueue(addr,
                              is_server=False,
                              fuse_message=False,
                              socket_type=zmq.PUSH,
                              name=f"worker_result_queue_{i}")
                for i, addr in enumerate(multi_frontend_addrs)
            ]
        else:
            # IPC queue for sending results back to the proxy, and let the
            # Proxy process to handle the postprocess
            result_queue = FusedIpcQueue(worker_queues.result_queue_addr,
                                         is_server=False,
                                         fuse_message=False,
                                         name="worker_result_queue")

    def notify_proxy_threads_to_quit():
        # Signal the dispatcher thread in every frontend proxy to quit
        if result_queue is not None:
            result_queue.put(None)
        elif frontend_result_queues is not None:
            for q in frontend_result_queues:
                q.put(None)
        else:
            assert result_queues is not None
            for q in result_queues:
                q.put(None)

    postprocess_worker_futures = []
    if is_leader and postproc_worker_config.enabled:
        logger_debug(f"initiate postprocess workers...", "yellow")

        # Each postproc worker pushes to every frontend result lane (a
        # single lane in single-frontend mode).
        proxy_result_addrs = (multi_frontend_addrs
                              if multi_frontend_addrs is not None else
                              [worker_queues.result_queue_addr])

        assert result_queues is not None
        # A reused worker process may still hold cyclic garbage from the LLM it
        # hosted before, including CUDA objects. Collect it here: a forked
        # postprocess worker that finalizes them aborts on CUDA.
        gc.collect()
        postproc_worker_pool = ProcessPoolExecutor(
            max_workers=postproc_worker_config.num_postprocess_workers)
        for i in range(postproc_worker_config.num_postprocess_workers):
            fut = postproc_worker_pool.submit(
                postproc_worker_main,
                result_queues[i].address,
                proxy_result_addrs,
                postproc_worker_config.postprocess_tokenizer_dir,
                PostprocWorker.default_record_creator,
                postproc_worker_config.post_processor_hook,
            )
            postprocess_worker_futures.append(fut)

    # Error handling in the Worker/MPI process
    #   1. During Executor initialization, the errors will be captured and
    #      send back via request_error_queue.
    #   2. During execution, the errors will be captured by ManagedThreads
    #      a) For per-request error, the error will be send back via
    #         result_queue, and eventually raised in handle_response() in
    #         the main thread.
    #      b) For system error, the error will be raised in the MPI process
    #         and handled by future.done_callback, that will propagate the
    #         error to the error_queue in the main thread.

    mpi_comm().barrier()
    worker_process_identities = mpi_comm().allgather(
        capture_worker_process_identity(mpi_rank()))

    # Publish the process identities before backend construction begins. Model
    # construction can load weights for several minutes, and an externally
    # killed worker may not complete its MPI future. Registering the workers at
    # this point lets the proxy observe such a death while it is still waiting
    # for the READY signal.
    if is_leader and worker_process_identities_signal is not None:
        identities_msg = (worker_process_identities_signal, None,
                          worker_process_identities)
        if not worker_init_status_queue.notify_with_retry(identities_msg):
            # The failed status queue cannot report its own failure. Let this
            # escape through the MPI future so the proxy can observe it.
            raise RuntimeError(
                "Failed to deliver worker process identities to proxy")

    logger_debug(f"Worker {mpi_rank()} ready to setup backend...\n", "green")

    try:
        worker: GenerationExecutorWorker = worker_cls(
            postproc_worker_config=postproc_worker_config,
            is_llm_executor=is_llm_executor,
            hf_model_dir=hf_model_dir,
            tokenizer=tokenizer,
            llm_args=llm_args,
            rpc_addr=rpc_addr,
            hmac_key=hmac_key)
    except Exception as e:
        logger.error(f"Failed to initialize executor on rank {mpi_rank()}: {e}")
        logger.error(traceback.format_exc())
        logger_debug(f"error: {traceback.format_exc()}", "red")
        if is_leader:
            # Send error message with confirmation
            error_msg = (e, traceback.format_exc())
            if not worker_init_status_queue.notify_with_retry(error_msg):
                logger.error("Failed to deliver error message to proxy")
        return

    # Optionally disable GC (default: not disabled)
    if os.getenv("TRTLLM_WORKER_DISABLE_GC", "0") == "1":
        gc.disable()
        # With automatic GC off, dynamo's post-compile gc.collect(1) walks every
        # object allocated since the previous compile (seconds per recompile).
        if "TORCH_DYNAMO_RUN_GC_AFTER_COMPILE" not in os.environ:
            import torch._dynamo.config
            torch._dynamo.config.run_gc_after_compile = False

    with worker:
        try:
            worker.block_subordinates()

            if is_leader:
                if postproc_worker_config.enabled:
                    worker.set_postproc_queues(result_queues)
                elif frontend_result_queues is not None:
                    worker.set_frontend_result_queues(frontend_result_queues)
                else:
                    worker.set_result_queue(result_queue)

                # Send ready signal with confirmation
                ready_msg = (ready_signal, None, worker_process_identities)
                if not worker_init_status_queue.notify_with_retry(ready_msg):
                    logger.warning(
                        "Failed to deliver ready signal to proxy, continuing anyway"
                    )
                if resource_governor_queue is not None:
                    # Swap rank 0 to the proxy IPC queue after construction.
                    # The resource-governor flag is already enabled on all
                    # ranks.
                    worker.engine.set_resource_governor_queue(
                        resource_governor_queue)

                for req in _incoming_requests(request_queue):
                    if isinstance(req, CancellingRequest):
                        worker.abort_request(req.id)
                    elif isinstance(req, StartProfileRequest):
                        # ``PyExecutor.start_profile`` raises
                        # ``RequestError`` on the double-start path.
                        # Capture it and propagate back to the proxy via
                        # ``profile_ack_queue`` so the HTTP /start_profile
                        # handler can return 409 in the IPC-proxy
                        # deployment too.
                        ack_err: Optional[str] = None
                        try:
                            worker.start_profile(output_dir=req.output_dir,
                                                 num_steps=req.num_steps,
                                                 start_step=req.start_step,
                                                 activities=req.activities)
                        except RequestError as e:
                            logger.warning(f"start_profile rejected: {e}")
                            ack_err = str(e)
                        send_profile_ack("start", ack_err, req)
                    elif isinstance(req, StopProfileRequest):
                        # ``worker.stop_profile`` blocks until
                        # ``PyExecutor.stop_profile`` has actually fired
                        # the in-loop stop and exported the chrome trace
                        # (or its 30s timeout elapsed). Acking only after
                        # that return is what gives the proxy its
                        # synchronous "trace is on disk" guarantee.
                        ack_err = None
                        try:
                            worker.stop_profile()
                        except RequestError as e:
                            logger.warning(f"stop_profile rejected: {e}")
                            ack_err = str(e)
                        send_profile_ack("stop", ack_err, req)
                    elif isinstance(req, GenerationRequest):
                        try:
                            worker.submit(req)
                        except RequestError as e:
                            logger.error(f"submit request failed: {e}")
                            logger.error(traceback.format_exc())
                            worker._await_response_helper.temp_error_responses.put(
                                ErrorResponse(req.id, e, req.id))
                    else:
                        raise ValueError(f"Unknown request type: {type(req)}")

                notify_proxy_threads_to_quit()

        except GenerationExecutorWorker.WorkerExit as e:
            # This will capture by the with-statement and exit normally.
            raise e

        except Exception as e:  # other critical errors
            if is_leader:
                notify_proxy_threads_to_quit()
            logger.error(traceback.format_exc())
            # This will be captured by mpi4py and handled by future.done_callback
            raise e
