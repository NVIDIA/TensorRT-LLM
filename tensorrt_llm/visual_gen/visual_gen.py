# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import asyncio
import atexit
import itertools
import os
import queue
import socket
import sys
import threading
import time
import traceback
import weakref
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch.multiprocessing as mp
import zmq

from tensorrt_llm._torch.visual_gen import DiffusionRequest, DiffusionResponse
from tensorrt_llm._torch.visual_gen.executor import run_diffusion_worker
from tensorrt_llm._torch.visual_gen.output import MediaOutput
from tensorrt_llm._torch.visual_gen.pipeline import ExtraParamSchema
from tensorrt_llm.visual_gen.args import VisualGenArgs
from tensorrt_llm.visual_gen.params import VisualGenParams

__all__ = [
    "VisualGen",
    "VisualGenParams",
    "ExtraParamSchema",
    "MediaOutput",
    "VisualGenError",
    "VisualGenParamsError",
    "VisualGenResult",
]
from tensorrt_llm.executor.ipc import ZeroMqQueue
from tensorrt_llm.llmapi.utils import set_api_status
from tensorrt_llm.logger import logger

# Timeouts (seconds)
POLL_TIMEOUT = 0.01
AWAIT_TIMEOUT = 0.05
THREAD_TIMEOUT = 5.0
WORKER_TIMEOUT = 2.0


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def get_ip_address() -> str:
    """Get local IP address."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def _detect_external_launch() -> Optional[Tuple[int, int, int, str, int]]:
    """Detect whether the process was launched by an external distributed launcher.

    Checks for torchrun (``RANK`` + ``WORLD_SIZE``) and then SLURM
    (``SLURM_PROCID`` + ``SLURM_NTASKS``).  Returns a
    ``(rank, local_rank, world_size, master_addr, master_port)`` tuple when a
    multi-process launcher is detected (world_size > 1), or ``None`` for
    single-process / single-node ``mp.Process`` mode.
    """
    # torchrun / torchelastic sets RANK and WORLD_SIZE
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        if world_size > 1:
            local_rank = int(os.environ.get("LOCAL_RANK", rank))
            master_addr = os.environ.get("MASTER_ADDR")
            if master_addr is None:
                raise RuntimeError(
                    "MASTER_ADDR must be set for multi-node torchrun runs. "
                    "Add --master-addr=<node0-ip> to your torchrun command, or set "
                    "MASTER_ADDR in the environment before launching."
                )
            master_port = int(os.environ.get("MASTER_PORT", 29500))
            return rank, local_rank, world_size, master_addr, master_port

    # SLURM: srun --ntasks-per-node=GPUS_PER_NODE sets SLURM_PROCID / SLURM_NTASKS
    if "SLURM_PROCID" in os.environ and "SLURM_NTASKS" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        if world_size > 1:
            local_rank = int(os.environ.get("SLURM_LOCALID", rank))
            master_addr = os.environ.get("MASTER_ADDR")
            if master_addr is None:
                raise RuntimeError(
                    "MASTER_ADDR must be set for multi-node SLURM runs. "
                    "Add to your sbatch script:\n"
                    "  MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -1)"
                )
            master_port = int(os.environ.get("MASTER_PORT", 29500))
            return rank, local_rank, world_size, master_addr, master_port

    return None


@set_api_status("prototype")
class VisualGenError(RuntimeError):
    """Base exception for all VisualGen operations."""


@set_api_status("prototype")
class VisualGenParamsError(ValueError):
    """Raised when request parameters fail validation.

    This covers unknown parameter keys, unsupported universal fields
    for the loaded pipeline, type mismatches, and out-of-range values.
    Caught by the executor so it returns an error response rather than
    crashing the server.
    """


class DiffusionRemoteClient:
    """Client proxy for remote DiffusionExecutor in worker processes.

    Supports two launch modes:

    **Single-node (default)**
        ``VisualGen`` is called from an ordinary Python script.
        ``DiffusionRemoteClient`` spawns all worker processes locally via
        ``mp.Process`` with ``master_addr=127.0.0.1``.

    **Multi-node (external launcher)**
        The script is launched by ``torchrun`` or ``srun --ntasks-per-node=GPUS``.
        Each rank runs the same script; ``RANK`` / ``WORLD_SIZE`` / ``MASTER_ADDR``
        / ``MASTER_PORT`` are already set in the environment.

        - Rank 0: becomes the request coordinator.  It creates the ZMQ server
          sockets and starts its own worker in a background thread, then returns
          to the caller so the user script can call ``generate()``.
        - Rank > 0: handled by ``VisualGen.__init__`` before this class is
          instantiated — they call ``run_diffusion_worker`` directly and exit
          via ``sys.exit(0)``.  These ranks never reach ``DiffusionRemoteClient``.
    """

    def __init__(
        self,
        args: VisualGenArgs,
    ):
        self.args = args
        self.n_workers = args.parallel.n_workers

        # --- Detect external launcher (torchrun / srun) ---
        ext = _detect_external_launch()

        if ext is None:
            # Single-node: coordinator spawns all workers locally
            # Setup distributed env
            self.master_addr = "127.0.0.1"
            self.master_port = find_free_port()

            # Setup IPC addresses
            self.host_ip = get_ip_address()
            req_port, resp_port = find_free_port(), find_free_port()

            self.request_queue_addr = f"tcp://0.0.0.0:{req_port}"
            self.response_queue_addr = f"tcp://0.0.0.0:{resp_port}"
            self.req_addr_connect = f"tcp://{self.host_ip}:{req_port}"
            self.resp_addr_connect = f"tcp://{self.host_ip}:{resp_port}"

        else:
            # rank == 0 guaranteed — ranks 1..N-1 exited in VisualGen.__init__
            rank, local_rank, world_size, master_addr, master_port = ext
            req_port = find_free_port()
            resp_port = find_free_port()
            self.master_addr = master_addr
            self.master_port = master_port
            self.request_queue_addr = f"tcp://0.0.0.0:{req_port}"
            self.response_queue_addr = f"tcp://0.0.0.0:{resp_port}"
            self.req_addr_connect = f"tcp://{master_addr}:{req_port}"
            self.resp_addr_connect = f"tcp://{master_addr}:{resp_port}"

        # Generate shared HMAC keys for IPC authentication
        self.req_hmac_key = os.urandom(32)
        self.resp_hmac_key = os.urandom(32)

        # IPC setup
        self.requests_ipc = None
        self.responses_ipc = None
        self.pending_requests = queue.Queue()
        self.completed_responses: Dict[int, DiffusionResponse] = {}

        # We'll create asyncio primitives in the background thread's event loop
        self._event_loop = None
        self.response_event = None
        self.lock = None
        self.shutdown_event = threading.Event()
        self.event_loop_ready = threading.Event()

        # Start background thread (it will create its own event loop)
        self.background_thread = threading.Thread(target=self._serve_forever_thread, daemon=True)
        self.background_thread.start()

        # Wait for the background thread to initialize the event loop
        self.event_loop_ready.wait()

        # Pipeline metadata — populated by _wait_ready from the READY signal.
        self.default_generation_params: Dict = {}
        self.extra_param_specs: Dict = {}

        # --- Launch workers ---
        self.worker_processes = []
        self._ext_worker_thread: Optional[threading.Thread] = None

        if ext is None:
            logger.info(f"DiffusionClient: Launching {self.n_workers} workers")
            ctx = mp.get_context("spawn")
            for rank in range(self.n_workers):
                p = ctx.Process(
                    target=run_diffusion_worker,
                    kwargs={
                        "rank": rank,
                        "world_size": self.n_workers,
                        "master_addr": self.master_addr,
                        "master_port": self.master_port,
                        "request_queue_addr": self.req_addr_connect,
                        "response_queue_addr": self.resp_addr_connect,
                        "diffusion_args": self.args,
                        "req_hmac_key": self.req_hmac_key,
                        "resp_hmac_key": self.resp_hmac_key,
                        "log_level": logger.level,
                        "local_rank": rank,
                    },
                )
                p.start()
                self.worker_processes.append(p)
        else:
            # External launch: rank 0 runs its own worker in a background thread.
            # Other nodes' workers are already running (they were launched by the
            # external launcher and will connect to our ZMQ server once it binds).
            self._ext_worker_thread = threading.Thread(
                target=run_diffusion_worker,
                kwargs={
                    "rank": rank,
                    "world_size": self.n_workers,
                    "master_addr": master_addr,
                    "master_port": master_port,
                    "request_queue_addr": self.req_addr_connect,
                    "response_queue_addr": self.resp_addr_connect,
                    "diffusion_args": self.args,
                    "req_hmac_key": self.req_hmac_key,
                    "resp_hmac_key": self.resp_hmac_key,
                    "log_level": logger.level,
                    "local_rank": local_rank,
                },
                daemon=True,
            )
            self._ext_worker_thread.start()

        self._wait_ready()

    @staticmethod
    def _close_socket(ipc_queue):
        if ipc_queue and ipc_queue.socket:
            ipc_queue.socket.setsockopt(zmq.LINGER, 0)
            ipc_queue.close()

    def enqueue_requests(self, requests: List[DiffusionRequest]) -> List[int]:
        """Enqueue requests and return their IDs."""
        req_ids = []
        for req in requests:
            self.pending_requests.put(req)
            req_ids.append(req.request_id)
        return req_ids

    async def await_responses(
        self, request_ids: Union[int, List[int]], timeout: Optional[float] = None
    ) -> Union[DiffusionResponse, List[DiffusionResponse]]:
        """Wait for responses by request IDs.

        Args:
            request_ids: Single request ID or list of request IDs to wait for
            timeout: Maximum total wait time in seconds (None = wait indefinitely)

        Returns:
            Single response or list of responses (None if request timed out)
        """
        is_single = isinstance(request_ids, int)
        ids = [request_ids] if is_single else request_ids

        start_time = time.time()
        results = {}

        while len(results) < len(ids):
            async with self.lock:
                for req_id in ids:
                    if req_id in self.completed_responses:
                        results[req_id] = self.completed_responses.pop(req_id)

            # All responses collected
            if len(results) == len(ids):
                break

            # Check if overall timeout exceeded
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    break
                # Wait for remaining time or AWAIT_TIMEOUT, whichever is shorter
                wait_time = min(timeout - elapsed, AWAIT_TIMEOUT)
            else:
                wait_time = AWAIT_TIMEOUT

            try:
                await asyncio.wait_for(self.response_event.wait(), timeout=wait_time)
            except asyncio.TimeoutError:
                pass
            self.response_event.clear()

        out = [results.get(rid) for rid in ids]
        return out[0] if is_single else out

    def await_responses_sync(
        self, request_ids: Union[int, List[int]], timeout: Optional[float] = None
    ) -> Union[DiffusionResponse, List[DiffusionResponse]]:
        """Sync wrapper to await responses from the main thread."""
        future = asyncio.run_coroutine_threadsafe(
            self.await_responses(request_ids, timeout), self._event_loop
        )
        return future.result(timeout=timeout if timeout else None)

    def _init_ipc(self) -> bool:
        """Initialize IPC queues."""
        try:
            logger.info("DiffusionClient: Initializing IPC")
            self.requests_ipc = ZeroMqQueue(
                (self.request_queue_addr, self.req_hmac_key),
                is_server=True,
                socket_type=zmq.PUSH,
                use_hmac_encryption=True,
            )
            self.responses_ipc = ZeroMqQueue(
                (self.response_queue_addr, self.resp_hmac_key),
                is_server=True,
                socket_type=zmq.PULL,
                use_hmac_encryption=True,
            )
            logger.info("DiffusionClient: IPC ready")
            return True
        except Exception as e:
            logger.error(f"DiffusionClient: IPC init failed: {e}")
            return False

    def _send_shutdown(self):
        """Send shutdown signal."""
        logger.info("DiffusionClient: Sending shutdown signal")
        if self.requests_ipc:
            self.requests_ipc.put(None)
            self._close_socket(self.requests_ipc)

    def _process_requests(self):
        """Process pending requests."""
        try:
            req = self.pending_requests.get(timeout=POLL_TIMEOUT)
            if req is None:
                self._send_shutdown()
                self.shutdown_event.set()
                return

            logger.info(f"DiffusionClient: Sending request {req.request_id}")
            self.requests_ipc.put(req)
        except queue.Empty:
            pass
        except Exception as e:
            logger.error(f"DiffusionClient: Error sending request: {e}")
            logger.error(traceback.format_exc())

    def _process_responses(self):
        """Poll and process responses."""
        try:
            if self.responses_ipc.poll(timeout=POLL_TIMEOUT):
                response = self.responses_ipc.get()
                if isinstance(response, DiffusionResponse):
                    if response.request_id == -1:
                        logger.info("DiffusionClient: Received READY signal")

                    # Schedule the lock acquisition and event setting in the event loop
                    asyncio.run_coroutine_threadsafe(
                        self._store_response(response), self._event_loop
                    )
        except Exception as e:
            logger.error(f"DiffusionClient: Error processing response: {e}")

    async def _store_response(self, response: DiffusionResponse):
        """Store response in the completed_responses dict (async helper)."""
        async with self.lock:
            self.completed_responses[response.request_id] = response
        self.response_event.set()

    def _cleanup_ipc(self):
        """Cleanup IPC."""
        logger.info("DiffusionClient: Cleaning up IPC")
        self._close_socket(self.requests_ipc)
        self._close_socket(self.responses_ipc)

    def _serve_forever_thread(self):
        """Background thread wrapper that creates and runs an event loop."""
        logger.info("DiffusionClient: Background thread started")

        # Create a new event loop for this thread
        self._event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._event_loop)

        # Create async primitives in this thread's event loop
        self.response_event = asyncio.Event()
        self.lock = asyncio.Lock()

        # Signal that the event loop is ready
        self.event_loop_ready.set()

        # Run the async serve_forever
        try:
            self._event_loop.run_until_complete(self._serve_forever())
        finally:
            self._event_loop.close()
            logger.info("DiffusionClient: Background thread stopped")

    async def _serve_forever(self):
        """Background thread main loop (async version)."""
        if not self._init_ipc():
            return

        while not self.shutdown_event.is_set():
            self._process_requests()
            self._process_responses()
            await asyncio.sleep(0.001)  # Yield control to allow other coroutines to run

        self._cleanup_ipc()

    def shutdown(self):
        """Shutdown client and workers."""
        logger.info("DiffusionClient: Shutting down")
        self.pending_requests.put(None)

        self.background_thread.join(timeout=THREAD_TIMEOUT)
        if self.background_thread.is_alive():
            logger.warning("DiffusionClient: Force stopping background thread")
            self.shutdown_event.set()
            self.background_thread.join(timeout=1.0)

        # Shutdown workers
        logger.info("DiffusionClient: Stopping workers")
        for p in self.worker_processes:
            p.join(timeout=WORKER_TIMEOUT)
            if p.is_alive():
                logger.warning(f"DiffusionClient: Terminating worker {p.pid} with SIGTERM")
                p.terminate()
                p.join(timeout=WORKER_TIMEOUT)
                if p.is_alive():
                    logger.warning(f"DiffusionClient: Force killing worker {p.pid} with SIGKILL")
                    p.kill()
                    p.join(timeout=WORKER_TIMEOUT)

        # External-launch mode: join rank-0 worker thread
        if self._ext_worker_thread is not None and self._ext_worker_thread.is_alive():
            self._ext_worker_thread.join(timeout=WORKER_TIMEOUT)

    def _wait_ready(self):
        """Wait for workers to be ready (sync wrapper for async operation)."""
        logger.info("DiffusionClient: Waiting for workers")

        future = asyncio.run_coroutine_threadsafe(self._wait_ready_async(), self._event_loop)
        try:
            future.result()
        except Exception:
            self.shutdown()
            raise

    async def _wait_ready_async(self):
        """Wait for workers to be ready (async version).

        Polls indefinitely for the ready signal. If any worker process dies
        during initialization, raises RuntimeError immediately (LLM-style).
        """
        start_time = time.time()
        last_log_time = start_time
        log_interval = 300

        while True:
            async with self.lock:
                if -1 in self.completed_responses:
                    ready_resp = self.completed_responses.pop(-1)
                    # Extract pipeline metadata from the READY payload.
                    payload = ready_resp.output
                    if isinstance(payload, dict):
                        self.default_generation_params = payload.get(
                            "default_generation_params", {}
                        )
                        self.extra_param_specs = payload.get("extra_param_specs", {})
                    elapsed = time.time() - start_time
                    logger.info(f"DiffusionClient: Workers ready ({elapsed:.1f}s)")
                    return

            worker_dead = any(not p.is_alive() for p in self.worker_processes)
            ext_dead = (
                self._ext_worker_thread is not None and not self._ext_worker_thread.is_alive()
            )
            if worker_dead or ext_dead:
                raise RuntimeError("DiffusionClient: Worker died during initialization")

            now = time.time()
            if now - last_log_time >= log_interval:
                elapsed = now - start_time
                logger.info(f"DiffusionClient: Still waiting for workers ({elapsed:.0f}s elapsed)")
                last_log_time = now

            try:
                await asyncio.wait_for(self.response_event.wait(), timeout=AWAIT_TIMEOUT)
            except asyncio.TimeoutError:
                pass
            self.response_event.clear()


@set_api_status("prototype")
class VisualGenResult:
    """Future-like object for async generation."""

    def __init__(self, request_id: int, executor: DiffusionRemoteClient):
        self.request_id = request_id
        self.executor = executor
        self._result = None
        self._finished = False
        self._error = None

    @property
    def done(self) -> bool:
        """True if the generation has completed (successfully or with error)."""
        return self._finished

    async def result(self, timeout: Optional[float] = None) -> Any:
        """Wait for and return result (async version).

        Can be awaited from any async context (e.g., FastAPI background tasks).
        """
        if self._finished:
            if self._error:
                raise VisualGenError(self._error)
            return self._result

        # Use run_coroutine_threadsafe to execute in the background thread's event loop
        future = asyncio.run_coroutine_threadsafe(
            self.executor.await_responses(self.request_id, timeout=timeout),
            self.executor._event_loop,
        )

        # Await the future in the current event loop
        response = await asyncio.wrap_future(future)

        if response is None:
            raise VisualGenError("Generation timed out")

        if response.error_msg:
            self._error = response.error_msg
            self._finished = True
            raise VisualGenError(f"Generation failed: {response.error_msg}")

        self._result = response.output
        self._finished = True
        return self._result

    def result_sync(self, timeout: Optional[float] = None) -> Any:
        """Blocking wrapper around result() for non-async callers."""
        future = asyncio.run_coroutine_threadsafe(
            self.result(timeout=timeout),
            self.executor._event_loop,
        )
        return future.result(timeout=timeout)

    def cancel(self):
        raise NotImplementedError("Cancel request (not yet implemented).")


class VisualGen:
    """High-level API for visual generation."""

    @set_api_status("prototype")
    def __init__(
        self,
        model: Union[str, Path],
        args: Optional[VisualGenArgs] = None,
    ):
        self.model = str(model)
        self.args = (args or VisualGenArgs()).model_copy(update={"checkpoint_path": self.model})

        # In external-launch mode (torchrun/srun), ranks 1..N-1 run as pure
        # workers and never return to user code.
        ext = _detect_external_launch()
        if ext is not None:
            rank, local_rank, world_size, master_addr, master_port = ext
            n_workers = self.args.parallel.n_workers
            if world_size != n_workers:
                raise ValueError(
                    f"Launcher world_size ({world_size}) does not match "
                    f"n_workers ({n_workers}). "
                    "Launch exactly n_workers tasks."
                )
            if rank != 0:
                logger.info(
                    f"VisualGen: rank {rank}/{world_size}, local_rank {local_rank} — "
                    "starting as worker (external launch mode)"
                )
                run_diffusion_worker(
                    rank=rank,
                    world_size=n_workers,
                    master_addr=master_addr,
                    master_port=master_port,
                    request_queue_addr=None,  # unused: non-zero ranks receive requests via dist.broadcast_object_list
                    response_queue_addr=None,  # unused: only rank 0 sends responses over ZMQ
                    diffusion_args=self.args,
                    req_hmac_key=None,
                    resp_hmac_key=None,
                    local_rank=local_rank,
                )
                sys.exit(0)
            logger.info(
                f"VisualGen: rank 0/{world_size} — coordinator + worker (external launch mode)"
            )

        self.executor = DiffusionRemoteClient(
            args=self.args,
        )
        self._req_counter = itertools.count()

        atexit.register(VisualGen._atexit_shutdown, weakref.ref(self))

    @property
    def extra_param_specs(self) -> Dict[str, "ExtraParamSchema"]:
        """Returns extra param specs for the loaded pipeline.

        Use this to discover types, ranges, and descriptions of
        model-specific parameters passed via ``extra_params``.
        """
        return self.executor.extra_param_specs

    @property
    def default_params(self) -> "VisualGenParams":
        """Returns a ``VisualGenParams`` with all defaults resolved for the loaded pipeline.

        Universal fields (height, width, etc.) are filled from the
        pipeline's defaults.  All declared ``extra_params`` keys are
        included with their defaults (``None`` for params without one).

        Use this to inspect what the model will use, then modify and
        pass to ``generate()``::

            params = visual_gen.default_params
            params.extra_params["stg_scale"] = 0.5
            params.height = 1024
            output = visual_gen.generate(inputs="a cat", params=params)
        """
        kwargs = dict(self.executor.default_generation_params)
        extra = {}

        for key, spec in self.executor.extra_param_specs.items():
            extra[key] = spec.default

        if extra:
            kwargs["extra_params"] = extra

        return VisualGenParams(**kwargs)

    @set_api_status("prototype")
    def generate(
        self,
        inputs: Union[str, List[str]],
        params: Optional[VisualGenParams] = None,
    ) -> MediaOutput:
        """Synchronous generation. Blocks until complete.

        Args:
            inputs: Text prompt string or list of prompt strings.
            params: Generation parameters (optional; uses model defaults when None).

        Returns:
            MediaOutput: Generated media with model-specific fields populated:
                - FLUX2: MediaOutput(image=torch.Tensor)
                - WAN: MediaOutput(video=torch.Tensor)
                - LTX2: MediaOutput(video=torch.Tensor, audio=torch.Tensor)
        """
        future = self.generate_async(
            inputs=inputs,
            params=params,
        )

        # Use the sync wrapper to get result
        response = self.executor.await_responses_sync(future.request_id, timeout=None)
        if response.error_msg:
            raise VisualGenError(f"Generation failed: {response.error_msg}")
        return response.output

    @set_api_status("prototype")
    def generate_async(
        self,
        inputs: Union[str, List[str]],
        params: Optional[VisualGenParams] = None,
    ) -> VisualGenResult:
        """Async generation. Returns immediately with future-like object.

        Args:
            inputs: Text prompt string or list of prompt strings.
            params: Generation parameters (optional; uses model defaults when None).

        Returns:
            VisualGenResult: Call result() to get output dict.
        """
        req_id = next(self._req_counter)

        # Normalize to List[str] for DiffusionRequest.prompt
        if isinstance(inputs, str):
            prompt = [inputs]
        elif isinstance(inputs, (list, tuple)):
            if not inputs:
                raise ValueError("Batch inputs must contain at least one item")
            if not all(isinstance(item, str) for item in inputs):
                raise ValueError("Batch inputs must contain only strings (prompt text)")
            prompt = list(inputs)
        else:
            raise ValueError(f"Invalid inputs type: {type(inputs)}")

        # Snapshot caller-provided params so later mutations don't affect
        # the queued request (the dispatcher thread serializes it lazily).
        request = DiffusionRequest(
            request_id=req_id,
            prompt=prompt,
            params=params.model_copy(deep=True) if params is not None else None,
        )

        self.executor.enqueue_requests([request])
        return VisualGenResult(req_id, self.executor)

    @staticmethod
    def _atexit_shutdown(self_ref):
        instance = self_ref()
        if instance is not None:
            instance.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> Literal[False]:
        del exc_value, traceback
        self.shutdown()
        return False

    def __del__(self):
        self.shutdown()

    @set_api_status("prototype")
    def shutdown(self):
        """Shutdown executor and cleanup."""
        if not hasattr(self, "executor") or self.executor is None:
            return
        logger.info("VisualGen: Shutting down")
        self.executor.shutdown()
        self.executor = None
