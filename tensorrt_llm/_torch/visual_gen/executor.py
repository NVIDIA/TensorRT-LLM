# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import atexit
import contextlib
import os
import pickle
import queue
import socket
import sys
import threading
import time
import traceback
import weakref
from collections import OrderedDict, deque
from concurrent.futures import Future, InvalidStateError
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Deque,
    Dict,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Union,
)

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import zmq

from tensorrt_llm._torch.distributed.initialization import init_torch_distributed
from tensorrt_llm._torch.shared_tensor import SharedTensorContainer
from tensorrt_llm._torch.visual_gen.output import PipelineOutput
from tensorrt_llm._torch.visual_gen.pipeline import AdvanceOutcome
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
from tensorrt_llm._torch.visual_gen.session_protocol import (
    AdmissionFailure,
    ControlType,
    ResultType,
    SessionAdmitted,
    SessionClosed,
    SessionCommand,
    SessionControl,
    SessionEnded,
    SessionError,
    SessionOpen,
    SessionOpened,
    SessionQueueFull,
    SessionReply,
    SessionResultMessage,
    SessionSubmit,
)
from tensorrt_llm._torch.visual_gen.session_state import SessionState
from tensorrt_llm._torch.visual_gen.slot_pool import DeliverySlot, SlotPool, map_slots
from tensorrt_llm.bindings.BuildInfo import check_cuda_architecture_supported
from tensorrt_llm.bindings.internal import start_coordinator_watchdog
from tensorrt_llm.executor.ipc import ZeroMqQueue
from tensorrt_llm.llmapi.utils import configure_cpu_affinity
from tensorrt_llm.logger import logger
from tensorrt_llm.visual_gen.args import VisualGenArgs

if TYPE_CHECKING:
    from tensorrt_llm.visual_gen.params import VisualGenParams


# Timeouts (seconds) for the client-side coordinator.
POLL_TIMEOUT = 0.01
AWAIT_TIMEOUT = 0.05
THREAD_TIMEOUT = 5.0
WORKER_TIMEOUT = 2.0
EXT_WORKER_TEARDOWN_TIMEOUT = 30.0
WORKER_SPAWN_SHUTDOWN_TIMEOUT = 5.0

# Module-local seams keep lifecycle tests from monkeypatching process-wide
# module objects used by unrelated threads and tests.
_Event = threading.Event
_Thread = threading.Thread
_get_mp_context = mp.get_context
_register_atexit = atexit.register
_get_process_id = os.getpid
_start_coordinator_watchdog = start_coordinator_watchdog


# Default cap on the size of the iteration-stats snapshot buffer used by the
# /metrics endpoint.  Mirrors the LLM ``iter_stats_max_iterations`` default.
_DEFAULT_ITER_STATS_MAX = 1000


class _SessionExit(Enum):
    """Why a session ended. After CLOSE and IDLE_TIMEOUT the worker serves the
    next request; after SHUTDOWN it stops, once the session is torn down."""

    CLOSE = "close"  # the client closed the session
    SHUTDOWN = "shutdown"  # the executor's shutdown signal arrived during the session
    IDLE_TIMEOUT = "idle_timeout"  # the client went quiet past the configured limit


class _RankTag(Enum):
    """What rank 0 tells the other ranks to do next, one collective per step."""

    OPEN_GO = "open_go"  # every rank runs the pipeline's open
    OPEN_ABORT = "open_abort"  # rank 0 refused the session before any pipeline work
    ADVANCE = "advance"  # every rank runs one advance; the payload is the inputs
    CLOSE = "close"  # every rank runs the pipeline's close


def _reap_worker_process(process: mp.Process) -> bool:
    worker_pid = process.pid
    if worker_pid is None:
        return False
    process.join(timeout=WORKER_TIMEOUT)
    if process.is_alive():
        logger.warning(f"DiffusionClient: Terminating worker {worker_pid} with SIGTERM")
        process.terminate()
        process.join(timeout=WORKER_TIMEOUT)
        if process.is_alive():
            logger.warning(f"DiffusionClient: Force killing worker {worker_pid} with SIGKILL")
            process.kill()
            process.join(timeout=WORKER_TIMEOUT)
    return True


class _WorkerProcessSpawner:
    """Spawn workers off the signal-handling thread and reap late starts."""

    def __init__(self, processes: List[mp.Process]):
        self._processes = processes
        self._spawn_cancelled = _Event()
        self._spawn_complete = _Event()
        self._thread_entered = _Event()
        self._reap_locks = {id(process): threading.Lock() for process in processes}
        self._reaped_process_ids: Set[int] = set()
        self._spawn_error: Optional[BaseException] = None
        self._thread = _Thread(
            target=self._run,
            name="visualgen-worker-process-spawner",
            daemon=True,
        )

    def start(self) -> None:
        try:
            self._thread.start()
        except BaseException as e:
            # A main-thread signal can interrupt Thread.start() after the OS
            # thread exists but before start() returns. Give that thread a
            # chance to publish its entry before declaring the batch inert.
            self._thread_entered.wait(timeout=THREAD_TIMEOUT)
            if not self._thread_entered.is_set():
                self._spawn_error = e
                self._spawn_complete.set()
            raise

    def cancel_spawn(self) -> None:
        self._spawn_cancelled.set()

    def wait_for_spawn(self, timeout: Optional[float] = None, *, raise_error: bool = True) -> bool:
        if not self._spawn_complete.wait(timeout=timeout):
            if raise_error:
                raise TimeoutError(
                    f"VisualGen worker process spawn did not complete within {timeout:.0f}s"
                )
            return False
        if raise_error and self._spawn_error is not None:
            raise self._spawn_error
        return True

    def reap_started_processes(self) -> None:
        for process in self._processes:
            process_id = id(process)
            with self._reap_locks[process_id]:
                if process_id in self._reaped_process_ids:
                    continue
                if _reap_worker_process(process):
                    self._reaped_process_ids.add(process_id)

    def _run(self) -> None:
        try:
            self._thread_entered.set()
            for process in self._processes:
                if self._spawn_cancelled.is_set():
                    break
                process.start()
        except BaseException as e:
            self._spawn_error = e
            logger.error(f"VisualGen worker process spawn failed: {e}")
        finally:
            self._thread_entered.set()
            self._spawn_complete.set()
            # Process.start() may publish its pid after shutdown's bounded
            # wait. Reap that late worker before this short-lived thread exits.
            if self._spawn_cancelled.is_set():
                self.reap_started_processes()


class _IterationStatsTracker:
    """Visual-gen analog of the LLM iteration-stats producer.

    Mirrors the LLM /metrics shape where it makes sense (``numActiveRequests``,
    ``numQueuedRequests``) so any downstream consumer that already parses the
    LLM /metrics shape can read VisualGen /metrics with minimal code changes.

    Snapshots are produced on lifecycle events (request enqueued, request sent
    to workers, response received) rather than on a fixed cadence, so a
    consumer always sees the transitions between idle / queued / active states
    even between rapid-fire calls to ``/metrics``.
    """

    def __init__(self, maxlen: int = _DEFAULT_ITER_STATS_MAX):
        self._iter = 0
        self._buffer: deque = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        # Set of request ids that have been pushed onto the worker queue but
        # not yet completed (received their final response).  Tracking by id
        # (instead of just a counter) makes ``record_request_completed``
        # idempotent under duplicate completion events and keeps
        # ``currentRequestId`` valid when responses arrive out of order with
        # respect to dispatch order.
        self._active_request_ids: Set[int] = set()
        # Insertion order of active ids; we use the most-recently-added id as
        # the "current" request when the previous current request completes
        # while others remain in flight.  ``deque`` lets us pop from either
        # end in O(1) and preserve ordering across out-of-order completions.
        self._active_order: deque = deque()
        # Most-recently-sent in-flight request id; ``None`` when idle.
        self._current_request_id: Optional[int] = None
        # Cumulative diffusion-step count since the current request started.
        # Reset to 0 when a new request becomes the current one and frozen
        # once that request completes (becomes a stable post-mortem value
        # until the next request begins).
        self._current_steps_processed = 0
        # Per-request step index for the in-flight request; ``None`` when
        # idle or when no step-progress signal is available from the
        # underlying pipeline.
        self._current_request_step_idx: Optional[int] = None

    def _snapshot_locked(self, num_queued_requests: int) -> Dict:
        """Build a snapshot dict (caller must hold ``self._lock``)."""
        self._iter += 1
        return {
            "iter": self._iter,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "numQueuedRequests": int(num_queued_requests),
            "numActiveRequests": len(self._active_request_ids),
            "currentStepsProcessed": int(self._current_steps_processed),
            "currentRequestId": self._current_request_id,
            "currentRequestStepIdx": self._current_request_step_idx,
        }

    def record_enqueue(self, num_queued_requests: int) -> None:
        """Append a snapshot reflecting an enqueue event."""
        with self._lock:
            self._buffer.append(self._snapshot_locked(num_queued_requests))

    def record_request_started(self, request_id: int, num_queued_requests: int) -> None:
        """Append a snapshot reflecting a request being dispatched to workers."""
        with self._lock:
            if request_id not in self._active_request_ids:
                self._active_request_ids.add(request_id)
                self._active_order.append(request_id)
            # The most-recently-dispatched request becomes the "current" one
            # for step-progress reporting; reset step state to match.
            self._current_request_id = request_id
            self._current_steps_processed = 0
            self._current_request_step_idx = None
            self._buffer.append(self._snapshot_locked(num_queued_requests))

    def record_request_completed(self, request_id: int, num_queued_requests: int) -> None:
        """Append a snapshot reflecting a request completion.

        Idempotent: a duplicate completion event for the same ``request_id``
        is a no-op for the active count and current-request state, so the
        active count cannot underflow and an unrelated in-flight request's
        state is never disturbed.
        """
        with self._lock:
            if request_id in self._active_request_ids:
                self._active_request_ids.discard(request_id)
                # Lazy removal from the ordering deque -- we filter stale
                # entries when picking a fallback "current" id below.
                if self._current_request_id == request_id:
                    # Drop the completed id; preserve currentStepsProcessed
                    # as a post-mortem read for the next snapshot poller.
                    self._current_request_id = None
                    self._current_request_step_idx = None
                    # Fall back to the most-recently-dispatched still-active
                    # request, if any, so out-of-order completions don't
                    # spuriously park ``currentRequestId`` at None while
                    # other requests are still in flight.
                    while self._active_order:
                        candidate = self._active_order[-1]
                        if candidate in self._active_request_ids:
                            self._current_request_id = candidate
                            break
                        self._active_order.pop()
            self._buffer.append(self._snapshot_locked(num_queued_requests))

    def record_step(self, request_id: int, step_idx: int, num_queued_requests: int) -> None:
        """Append a snapshot for a per-step diffusion progress event.

        Currently no pipeline emits step callbacks, but the hook is kept for
        forward-compatibility so a future pipeline integration can populate
        ``currentRequestStepIdx`` and ``currentStepsProcessed`` accurately
        without re-shaping the buffer protocol.
        """
        with self._lock:
            if self._current_request_id == request_id:
                self._current_request_step_idx = int(step_idx)
                self._current_steps_processed = int(step_idx) + 1
            self._buffer.append(self._snapshot_locked(num_queued_requests))

    def drain(self) -> List[Dict]:
        """Return all buffered snapshots and clear the buffer."""
        with self._lock:
            stats = list(self._buffer)
            self._buffer.clear()
            return stats

    def current_snapshot(self, num_queued_requests: int) -> Dict:
        """Return a single snapshot of the *current* state (no buffering)."""
        with self._lock:
            return self._snapshot_locked(num_queued_requests)


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


def _cuda_memory_logging_enabled() -> bool:
    """Whether per-request CUDA peak-memory logging is enabled.

    This is a development-only knob exposed as an environment variable
    (rather than a public ``VisualGenArgs`` field) to keep the engine
    config surface clean, mirroring the nsys trace knob
    ``TLLM_PROFILE_VISUAL_GEN_START_STOP``.
    """
    return os.environ.get("TLLM_VISUAL_GEN_LOG_CUDA_MEMORY", "0").strip().lower() in (
        "1",
        "true",
        "yes",
    )


@dataclass
class DiffusionRequest:
    """Request for diffusion inference.

    Generation parameters live in the optional ``params`` object
    (a :class:`~tensorrt_llm.visual_gen.params.VisualGenParams` instance).
    When ``params`` is ``None`` (the default), the executor creates a
    ``VisualGenParams()`` and fills it with pipeline-specific defaults
    before calling ``pipeline.run_inference()``.
    """

    request_id: int
    prompt: List[str]
    params: Optional["VisualGenParams"] = None
    prepared_inputs: Dict[str, Any] = field(default_factory=dict, repr=False)
    # Set only between the two ends of the coordinator -> rank0 hop; see
    # ``refs_to_shm``.
    ref_handles: Optional[List[Dict[str, Any]]] = field(default=None, repr=False)
    # Set only while the request is in flight on the rank0 -> N-rank hop; see
    # ``DiffusionExecutor._broadcast_request``.
    ref_sizes: Optional[List[int]] = field(default=None, repr=False)

    def refs_to_shm(self) -> None:
        """Move reference payloads into shared memory, in place (producer side).

        Only the coordinator -> rank0 hop travels as handles: rank0 restores the
        bytes before broadcasting, because a shared-tensor handle is consumed
        exactly once and minting one for N ranks would free the block N-1 times.
        """
        if self.params is None:
            return
        self.ref_handles = handles = []
        for slot in ("image_reference", "video_reference", "audio_reference"):
            for index, ref in enumerate(getattr(self.params, slot, None) or []):
                # A read-only view: the one copy happens inside from_tensor(),
                # which is what moves the payload into shared memory.
                buffer = torch.frombuffer(ref.content, dtype=torch.uint8)
                handles.append(
                    {
                        "slot": slot,
                        "index": index,
                        "handle": SharedTensorContainer.from_tensor(buffer).dump_to_dict(),
                    }
                )
                ref.content = b""
        if not handles:
            self.ref_handles = None

    def refs_from_shm(self) -> None:
        """Restore reference payloads from shared memory, in place (consumer side).

        Each handle is taken independently so one that cannot be rebuilt does
        not strand the blocks behind it.
        """
        failures = []
        for entry in self.ref_handles or []:
            try:
                ref = getattr(self.params, entry["slot"])[entry["index"]]
                container = SharedTensorContainer.from_dict(entry["handle"])
                ref.content = container.get_local_view().numpy().tobytes()
            except Exception as exc:
                failures.append(f"{entry['slot']}[{entry['index']}]: {exc}")
        self.ref_handles = None
        if failures:
            raise RuntimeError(
                "failed to restore reference payloads from shared memory: " + "; ".join(failures)
            )


@dataclass
class DiffusionResponse:
    """Response with model-specific output.

    Attributes:
        request_id: Unique identifier for the request.
        output: Generated media as :class:`PipelineOutput` with the
            model-specific fields populated. Set to ``None`` on the error
            path; on the READY signal it carries a ``dict`` instead.
        error_msg: Error message if generation failed.
        error_type: Failure class when ``error_msg`` is set: ``"client"``
            (unusable request content → 400 / ``ValueError``), ``"capacity"``
            (valid request does not fit the deployment → 503 /
            ``MemoryError``), or ``None`` for unclassified runtime failures
            (500 / ``RuntimeError``).
        generation: Wall-clock time the executor measured around request
            preparation and the engine's inference call (host
            ``time.perf_counter()``), in seconds. Default ``0.0`` so the
            dataclass round-trips through pickling across worker/client; the
            error path leaves it at ``0.0``.
    """

    request_id: int
    output: Optional[PipelineOutput] = None
    error_msg: Optional[str] = None
    error_type: Optional[str] = None
    generation: float = 0.0


class DiffusionExecutor:
    """Execution engine for diffusion models running in worker processes."""

    def __init__(
        self,
        request_queue_addr: str,
        response_queue_addr: str,
        device_id: int,
        visual_gen_args: "VisualGenArgs",
        req_hmac_key: Optional[bytes] = None,
        resp_hmac_key: Optional[bytes] = None,
        in_client_process: bool = False,
        control_queue_addr: Optional[str] = None,
        ctl_hmac_key: Optional[bytes] = None,
    ):
        self.request_queue_addr = request_queue_addr
        self.response_queue_addr = response_queue_addr
        self.device_id = device_id
        self.visual_gen_args = visual_gen_args
        self.resp_hmac_key = resp_hmac_key
        self.in_client_process = in_client_process

        self.pipeline = None  # initialized in _load_pipeline
        self.requests_ipc = None
        # Session control (slot returns, close, keep-alive) arrives on its own
        # socket, so it is serviced whatever the state of the request socket.
        self.controls_ipc = None
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.response_queue = queue.Queue()
        self.sender_thread = None
        self.session_config = getattr(visual_gen_args, "session_config", None)

        # Only rank 0 handles IPC
        if self.rank == 0:
            logger.info(f"Worker {device_id}: Connecting to request queue")
            self.requests_ipc = ZeroMqQueue(
                (request_queue_addr, req_hmac_key),
                is_server=False,
                socket_type=zmq.PULL,
                use_hmac_encryption=True,
            )
            if control_queue_addr is not None:
                self.controls_ipc = ZeroMqQueue(
                    (control_queue_addr, ctl_hmac_key),
                    is_server=False,
                    socket_type=zmq.PULL,
                    use_hmac_encryption=True,
                )
            self.sender_thread = threading.Thread(target=self._sender_loop, daemon=True)
            self.sender_thread.start()

        self._load_pipeline()

    def _sender_loop(self):
        """Background thread for sending responses."""
        logger.info(f"Worker {self.device_id}: Connecting to response queue")
        responses_ipc = ZeroMqQueue(
            (self.response_queue_addr, self.resp_hmac_key),
            is_server=False,
            socket_type=zmq.PUSH,
            use_hmac_encryption=True,
        )

        while True:
            try:
                resp = self.response_queue.get()
                if resp is None:
                    break
                responses_ipc.put(resp)
            except Exception as e:
                logger.error(f"Worker {self.device_id}: Sender error: {e}")

        if responses_ipc.socket:
            responses_ipc.socket.setsockopt(zmq.LINGER, 0)
        responses_ipc.close()

    def _load_pipeline(self):
        """
        Load pipeline using proper flow:
        VisualGenArgs → PipelineLoader → DiffusionModelConfig → AutoPipeline → BasePipeline
        """
        logger.info(f"Worker {self.device_id}: Loading pipeline")

        try:
            args = self.visual_gen_args
            loader = PipelineLoader(args, device=f"cuda:{self.device_id}")
            self.pipeline = loader.load(
                skip_warmup=args.compilation_config.skip_warmup,
            )

        except Exception as e:
            logger.error(f"Worker {self.device_id}: Failed to load pipeline: {e}")
            raise

        logger.info(f"Worker {self.device_id}: Pipeline ready")

        # Sync all workers
        dist.barrier()

        # Send READY signal with pipeline metadata for the client.
        if self.rank == 0:
            logger.info(f"Worker {self.device_id}: Sending READY")
            self.response_queue.put(
                DiffusionResponse(
                    request_id=-1,
                    output={
                        "status": "READY",
                        "default_generation_params": self.pipeline.default_generation_params,
                        "extra_param_specs": self.pipeline.extra_param_specs,
                        "supports_image_edit": self.pipeline.supports_image_edit,
                        "ref_slot_specs": self.pipeline.ref_slot_specs,
                    },
                )
            )

    def _broadcast_request(self, req: Optional[DiffusionRequest]) -> Optional[DiffusionRequest]:
        """Send one request from rank0 to every rank.

        Reference payloads ride as raw uint8 tensors alongside the request
        rather than inside it, because ``broadcast_object_list`` pickles the
        object into a tensor first and that copy dominates the collective.
        """
        payloads = []
        request = req.request if isinstance(req, SessionOpen) else req
        if self.rank == 0 and request is not None:
            # Take the payloads out before the object is pickled, and leave their
            # sizes behind so the peers can size their receive buffers.
            for slot in ("image_reference", "video_reference", "audio_reference"):
                for ref in getattr(request.params, slot, None) or []:
                    payloads.append(ref.content)
                    ref.content = b""
            request.ref_sizes = [len(p) for p in payloads]

        obj_list = [req]
        dist.broadcast_object_list(obj_list, src=0)
        req = obj_list[0]
        if req is None:
            return None
        request = req.request if isinstance(req, SessionOpen) else req

        if self.rank == 0:
            # Read-only views: the src rank only reads its buffer.
            buffers = [torch.frombuffer(p, dtype=torch.uint8) for p in payloads]
        else:
            buffers = [torch.empty(n, dtype=torch.uint8) for n in request.ref_sizes or []]
        for buffer in buffers:
            dist.broadcast(buffer, src=0)

        if self.rank != 0:
            payloads = [b.numpy().tobytes() for b in buffers]

        refs = [
            ref
            for slot in ("image_reference", "video_reference", "audio_reference")
            for ref in getattr(request.params, slot, None) or []
        ]
        if len(refs) != len(payloads):
            # zip() would silently leave the tail of either side behind, and
            # clearing ref_sizes below would erase the evidence.
            raise ValueError(f"expected {len(refs)} reference payloads, got {len(payloads)}.")
        for ref, payload in zip(refs, payloads):
            ref.content = payload
        request.ref_sizes = None
        return req

    def serve_forever(self):
        """Main execution loop: one request or one session at a time."""
        while True:
            req = None
            if self.rank == 0:
                req = self._next_message()
                if isinstance(req, DiffusionRequest):
                    req.refs_from_shm()
                elif isinstance(req, SessionOpen):
                    req.request.refs_from_shm()
                elif req is not None:
                    self._answer_stray_command(req)
                    continue
                logger.info(f"Worker {self.device_id}: Request available")

            # Skipped at world_size 1: with no peer, the object broadcast would
            # still serialize the whole request to a tensor before finding out.
            if self.world_size > 1:
                req = self._broadcast_request(req)

            if req is None:
                self._stop_serving()
                break

            if isinstance(req, SessionOpen):
                logger.info(f"Worker {self.device_id}: Opening session {req.request.request_id}")
                if self._serve_session(req.request) is _SessionExit.SHUTDOWN:
                    # Rank 0 read the signal inside the session; the peers are back
                    # at the broadcast and get it now.
                    if self.world_size > 1:
                        self._broadcast_request(None)
                    self._stop_serving()
                    break
                continue

            logger.info(f"Worker {self.device_id}: Processing request {req.request_id}")
            self.process_request(req)

    def _stop_serving(self) -> None:
        logger.info(f"Worker {self.device_id}: Shutdown signal received")
        if self.rank == 0 and self.sender_thread:
            self.response_queue.put(None)
            self.sender_thread.join()

    def _next_message(self) -> Any:
        """Rank 0: the next message from the request socket. Controls that arrive
        with no session open are strays (a close that crossed its SessionClosed):
        answered, never queued."""
        while True:
            for command in self._drain_controls():
                self._answer_stray_command(command)
            try:
                return self.requests_ipc.get(timeout=POLL_TIMEOUT)
            except queue.Empty:
                continue

    def _drain_controls(self) -> List[Any]:
        if self.controls_ipc is None:
            return []
        return self.controls_ipc.drain()

    def _answer_stray_command(self, command: Any) -> None:
        """A session command with no open session: a submit is refused so the
        client does not wait forever; the rest are stale and dropped."""
        if isinstance(command, SessionSubmit):
            self.response_queue.put(
                SessionAdmitted(
                    command.session_id,
                    error=f"session {command.session_id} is not open",
                    failure=AdmissionFailure.ENDED,
                )
            )
        elif isinstance(command, SessionControl) and command.type is ControlType.CLOSE:
            self.response_queue.put(SessionClosed(command.session_id))

    def _merge_defaults(self, req: DiffusionRequest):
        """Fill ``None`` fields in *req.params* with pipeline-specific defaults.

        Merges both universal defaults (from ``default_generation_params``)
        and extra_param defaults (from ``extra_param_specs``). ``req.params``
        is expected to be a concrete :class:`VisualGenParams`; defaults are
        materialized at the :class:`VisualGen.generate_async` enqueue site.
        """
        params = req.params
        # Universal field defaults
        for field_name, default_value in self.pipeline.default_generation_params.items():
            if hasattr(params, field_name) and getattr(params, field_name) is None:
                if (
                    params.image_reference
                    and getattr(self.pipeline, "derive_output_size_from_reference", False) is True
                    and field_name in ("height", "width")
                ):
                    continue
                setattr(params, field_name, default_value)
                # Marks it as a pipeline default rather than caller intent, so
                # request-dependent defaults stay re-resolvable; assigning the
                # field re-marks it.
                # Assumes model_fields_set is the live __pydantic_fields_set__, not a
                # copy; TestDefaultMarksThroughRealPath fails loudly if that changes.
                params.model_fields_set.discard(field_name)

        # Extra param defaults — fill all declared keys so infer() can use direct access
        specs = self.pipeline.extra_param_specs
        if specs:
            if params.extra_params is None:
                params.extra_params = {}
            for key, spec in specs.items():
                if key not in params.extra_params:
                    params.extra_params[key] = spec.default

    def process_request(self, req: DiffusionRequest):
        """Process a single request."""
        log_cuda_memory = _cuda_memory_logging_enabled()
        if log_cuda_memory:
            self._reset_cuda_peak_memory_stats()
        try:
            self._merge_defaults(req)
            # Include request preparation in executor-side generation latency.
            # Model-specific preparation runs before the warmup lookup so it
            # can resolve shape-dependent request fields such as output size.
            generation_start = time.perf_counter()
            self.pipeline.prepare_request(req)
            cache_key = self.pipeline.request_warmup_cache_key(req)
            cache_key_is_resolved = all(value is not None for value in cache_key)
            if (
                cache_key_is_resolved
                and self.pipeline._warmed_up_shapes
                and cache_key not in self.pipeline._warmed_up_shapes
            ):
                logger.warning(
                    f"Requested shape {cache_key} was not warmed up. "
                    f"First request with this shape will be slower due to "
                    f"torch.compile recompilation or CUDA graph capture. "
                    f"Warmed-up shapes: {self.pipeline._warmed_up_shapes}"
                )
            output = self.pipeline.run_inference(req)
            if log_cuda_memory:
                self._log_cuda_peak_memory(req.request_id)
            generation = time.perf_counter() - generation_start  # seconds

            if self.rank == 0:
                # CUDA IPC handles are invalid within the producing process, so
                # a same-process client takes the media via in-process handoff.
                output.to_handle(local=self.in_client_process)
                self.response_queue.put(
                    DiffusionResponse(
                        request_id=req.request_id,
                        output=output,
                        generation=generation,
                    )
                )
        except Exception as e:
            if log_cuda_memory:
                self._log_cuda_peak_memory(req.request_id)
            logger.error(f"Worker {self.device_id}: Error: {e}")
            logger.error(traceback.format_exc())
            if self.rank == 0:
                self.response_queue.put(
                    DiffusionResponse(
                        request_id=req.request_id,
                        error_msg=str(e),
                        error_type=self.pipeline.classify_request_failure(e),
                    )
                )

    def _reset_cuda_peak_memory_stats(self) -> None:
        """Reset CUDA peak memory stats for this worker if CUDA is available."""
        if not torch.cuda.is_available():
            return

        try:
            torch.cuda.reset_peak_memory_stats(self.device_id)
        except RuntimeError as e:
            logger.warning(
                f"Worker {self.device_id} rank {self.rank}: "
                f"Unable to reset CUDA peak memory stats: {e}"
            )

    def _log_cuda_peak_memory(self, request_id: int) -> None:
        """Log peak CUDA memory observed for one request."""
        if not torch.cuda.is_available():
            return

        try:
            peak_allocated = torch.cuda.max_memory_allocated(self.device_id)
            logger.info(
                f"Worker {self.device_id} rank {self.rank}: "
                f"Request {request_id} peak CUDA memory: {peak_allocated / 2**30:.2f} GiB"
            )
        except RuntimeError as e:
            logger.warning(
                f"Worker {self.device_id} rank {self.rank}: "
                f"Unable to log CUDA peak memory for request {request_id}: {e}"
            )

    # ------------------------------------------------------------------ sessions

    def _session_slot_count(self) -> int:
        cfg = self.session_config
        return int(getattr(cfg, "delivery_slots", 2)) if cfg is not None else 2

    def _session_idle_timeout(self) -> Optional[float]:
        cfg = self.session_config
        value = getattr(cfg, "idle_timeout", None) if cfg is not None else None
        return float(value) if value else None

    def _send_tag(self, tag: _RankTag, payload: Any = None) -> None:
        """Rank 0: the other ranks' next step. One collective that every rank
        enters, the others through ``_recv_tag``; nothing to do alone."""
        if self.world_size > 1:
            dist.broadcast_object_list([(tag, payload)], src=0)

    def _recv_tag(self) -> Tuple[_RankTag, Any]:
        """Ranks other than 0: wait for rank 0's next step and its payload."""
        obj = [None]
        dist.broadcast_object_list(obj, src=0)
        return obj[0]

    def _open_pipeline(
        self, req: DiffusionRequest, slot: Optional[DeliverySlot]
    ) -> Tuple[Any, AdvanceOutcome]:
        """The pipeline's part of opening, on every rank: its open call, then the
        completion barrier. Returns the pipeline state and the open's outcome.
        Raises what the pipeline
        raised, or ``RuntimeError`` when another rank failed; either way nothing
        stays open on this rank. With peers a failure here ends this rank
        instead (``_fail_rank``)."""
        what = f"session {req.request_id} open"
        pipeline_state, outcome, error = None, None, None
        try:
            with torch.inference_mode():
                pipeline_state, outcome = self.pipeline.open_session(req, slot)
        except Exception as e:
            self._fail_rank(what, e)
            logger.error(f"Worker {self.device_id}: {what} failed: {e}")
            logger.error(traceback.format_exc())
            error = e
        if self._finalize_advance(outcome):
            return pipeline_state, outcome
        if pipeline_state is not None:
            self._close_pipeline(pipeline_state, what)
        if error is not None:
            raise error
        raise RuntimeError(f"{what} failed on another rank")

    def _advance_pipeline(
        self,
        pipeline_state: Any,
        inputs: Dict[str, object],
        slot: Optional[DeliverySlot],
        what: str,
    ) -> AdvanceOutcome:
        """The pipeline's part of one step, on every rank: its advance call, then
        the completion barrier. Raises what the pipeline raised, or
        ``RuntimeError`` when another rank failed. With peers a failure here ends
        this rank instead (``_fail_rank``)."""
        outcome, error = None, None
        try:
            with torch.inference_mode():
                outcome = self.pipeline.advance(pipeline_state, inputs, slot)
        except Exception as e:
            self._fail_rank(what, e)
            logger.error(f"Worker {self.device_id}: {what} failed: {e}")
            logger.error(traceback.format_exc())
            error = e
        if self._finalize_advance(outcome):
            return outcome
        if error is not None:
            raise error
        raise RuntimeError(f"{what} failed on another rank")

    def _close_pipeline(self, pipeline_state: Any, what: str) -> None:
        """The pipeline's close on this rank; a failure is logged, nothing else."""
        try:
            with torch.inference_mode():
                self.pipeline.close_session(pipeline_state)
        except Exception as e:
            logger.error(f"Worker {self.device_id}: {what}: close failed: {e}")

    def _finalize_advance(self, outcome: Optional[AdvanceOutcome]) -> bool:
        """Wait for this rank's writes, then agree with every rank whether the
        advance succeeded; ``None`` is this rank's pipeline call having raised.
        Returns whether every rank succeeded. With several ranks a failed rank
        never gets here (see ``_fail_rank``), so the all-reduce is the
        completion barrier every rank passes together."""
        failed = outcome is None
        try:
            if outcome is not None and outcome.event is not None:
                outcome.event.synchronize()
            if self.world_size == 1:
                return not failed
            status = torch.tensor([1 if failed else 0], dtype=torch.int32)
            dist.all_reduce(status)
            return int(status.item()) == 0
        except Exception as e:
            # An asynchronous device error surfacing at the event, or the barrier
            # itself failing: peers may be inside it, so this is rank-local.
            self._fail_rank("completion barrier", e)
            raise

    def _fail_rank(self, what: str, exc: BaseException) -> None:
        """A pipeline call raised on this rank while peers may be inside a
        collective, so no further collective (status, CLOSE) can be trusted to
        complete. End this worker without touching the process group: a dedicated
        worker process exits with status 1, which the coordinator (or the
        external launcher) turns into an abort of the whole group; a worker
        thread inside the client's process raises ``FatalRankError`` instead, so
        the thread ends and the application survives. With a single rank this
        returns and does nothing: there is no peer to fall out of step with, and
        the caller reports the error at the boundary."""
        if self.world_size == 1:
            return
        logger.error(f"Worker {self.device_id}: {what}: {exc}; aborting this rank")
        logger.error(traceback.format_exc())
        if self.in_client_process:
            raise FatalRankError(f"{what}: {exc}") from exc
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)

    def _serve_session(self, req: DiffusionRequest) -> _SessionExit:
        """Serve one session to its end on this rank and say why it ended. A
        follower ends with the leader's CLOSE broadcast, whatever the reason."""
        if self.rank == 0:
            return self._session_leader(req)
        self._session_follower(req)
        return _SessionExit.CLOSE

    def _session_leader(self, req: DiffusionRequest) -> _SessionExit:
        """Rank 0: open the session, run it, tear it down; says why it ended. A
        refused open counts as closed."""
        opened = self._open_session(req)
        if opened is None:
            return _SessionExit.CLOSE
        session, pool, pipeline_state = opened
        fatal = False
        ending = _SessionExit.CLOSE
        try:
            ending = self._run_session(session, pool, pipeline_state)
        except FatalRankError:
            fatal = True
            raise
        except Exception as e:
            # Not from a pipeline call (a broadcast, a slot, the state); with
            # peers it still cannot be followed by a CLOSE broadcast.
            fatal = self.world_size > 1
            self._fail_rank(f"session {session.session_id}", e)
            raise
        finally:
            self._end_session(session, pipeline_state, fatal=fatal)
        return ending

    def _open_session(self, req: DiffusionRequest) -> Optional[Tuple[SessionState, SlotPool, Any]]:
        """The open handshake on rank 0. Answers ``SessionOpened``, with the
        capabilities and slot handles or with the error, publishes the initial
        result if the pipeline wrote one, and returns the session's state, pool
        and pipeline state; ``None`` when the session was refused or failed to
        open, with the reply already sent and nothing left to release."""
        session_id = req.request_id
        pipeline = self.pipeline
        pool = None
        try:
            self._merge_defaults(req)
            pipeline.prepare_request(req)
            if not pipeline.supports_sessions:
                raise SessionsUnsupported(f"{type(pipeline).__name__} does not support sessions")
            caps = pipeline.session_capabilities(req)
            session = SessionState(session_id, caps)
            device = (
                torch.device(f"cuda:{self.device_id}")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            pool = SlotPool(caps, self._session_slot_count(), device, local=self.in_client_process)
            logger.info(
                f"Worker {self.device_id}: session {session_id} slot pool: "
                f"{len(pool)} slots x {caps.max_result_bytes} bytes = {pool.nbytes} bytes"
            )
        except Exception as e:
            # Refused before any rank did pipeline work: the peers are told to
            # abort and the reply carries the failure class.
            logger.error(f"Worker {self.device_id}: session {session_id} refused: {e}")
            logger.error(traceback.format_exc())
            self._send_tag(_RankTag.OPEN_ABORT)
            # Asking an unsupported pipeline for a session is the caller's mistake;
            # what the pipeline itself raised is classified by the pipeline.
            error_type = (
                "client"
                if isinstance(e, SessionsUnsupported)
                else pipeline.classify_request_failure(e)
            )
            self.response_queue.put(SessionOpened(session_id, error=str(e), error_type=error_type))
            return None

        self._send_tag(_RankTag.OPEN_GO)
        initial_index, initial_slot = pool.acquire() if caps.initial_result else (None, None)
        try:
            pipeline_state, outcome = self._open_pipeline(req, initial_slot)
        except Exception as e:
            self.response_queue.put(
                SessionOpened(
                    session_id, error=str(e), error_type=pipeline.classify_request_failure(e)
                )
            )
            return None
        self.response_queue.put(
            SessionOpened(session_id, capabilities=caps.public(), slots=pool.export_handles())
        )
        if initial_slot is not None:
            if outcome.written:
                try:
                    shapes = self._written_shapes(initial_slot, outcome.written)
                    message = session.media_result(None, initial_index, shapes)
                except ValueError as e:
                    pool.release(initial_index)
                    self.response_queue.put(session.terminal_result(error=str(e)))
                    return session, pool, pipeline_state
                self.response_queue.put(message)
            else:
                pool.release(initial_index)
        if outcome.done or session.complete():  # done at open, or declared with length 0
            self.response_queue.put(session.terminal_result())
        return session, pool, pipeline_state

    def _run_session(
        self, session: SessionState, pool: SlotPool, pipeline_state: Any
    ) -> _SessionExit:
        """The session loop on rank 0: advance whenever the advance condition
        holds, otherwise wait for the client. The advance that completes a finite
        session publishes its terminal; the loop ends on the close command, on
        the shutdown signal, or on the idle timeout, and says which."""
        idle_timeout = self._session_idle_timeout()
        while True:
            # Commands already waiting go first, so a value submitted during the
            # last advance is seen by the next dispatch.
            ending = self._service_commands_now(session, pool, pipeline_state)
            if ending is not None:
                return ending
            if session.can_advance(pool.free_count):
                self._advance_session(session, pool, pipeline_state)
                continue
            # Only the client can make progress now; wait for it.
            ending = self._wait_for_commands(session, pool, pipeline_state, timeout=idle_timeout)
            if ending is _SessionExit.IDLE_TIMEOUT:
                logger.warning(
                    f"Worker {self.device_id}: session {session.session_id} idle for "
                    f"{idle_timeout:.0f}s; closing it"
                )
                if not session.ended:
                    self.response_queue.put(
                        session.terminal_result(
                            error=f"session idle timeout ({idle_timeout:.0f}s) expired"
                        )
                    )
                return _SessionExit.IDLE_TIMEOUT
            if ending is not None:
                return ending

    def _end_session(self, session: SessionState, pipeline_state: Any, *, fatal: bool) -> None:
        """Tear the session down. ``fatal``: this rank is dying with peers possibly
        stuck in a collective, so nothing shared is touched: no CLOSE broadcast,
        no pipeline close, no ``SessionClosed``; the coordinator reports the dead
        worker and tears the group down."""
        if fatal:
            return
        session.discard_inputs()
        self._send_tag(_RankTag.CLOSE)
        self._close_pipeline(pipeline_state, f"session {session.session_id}")
        self.response_queue.put(SessionClosed(session.session_id))
        logger.info(f"Worker {self.device_id}: session {session.session_id} closed")

    def _advance_session(self, session: SessionState, pool: SlotPool, pipeline_state: Any) -> None:
        """The session side of one step on rank 0: a slot and the next inputs,
        the ADVANCE tag to the peers, ``_advance_pipeline``, then the result or
        the terminal to the client."""
        slot_id, slot = pool.acquire()
        index, inputs = session.dispatch()
        self._send_tag(_RankTag.ADVANCE, inputs)
        try:
            outcome = self._advance_pipeline(
                pipeline_state, inputs, slot, f"session {session.session_id} advance {index}"
            )
        except Exception as e:
            pool.release(slot_id)
            self.response_queue.put(session.terminal_result(error=str(e)))
            return
        if outcome.written:
            try:
                shapes = self._written_shapes(slot, outcome.written)
                message = session.media_result(index, slot_id, shapes)
            except ValueError as e:
                pool.release(slot_id)
                self.response_queue.put(session.terminal_result(error=str(e)))
                return
            self.response_queue.put(message)
        else:
            pool.release(slot_id)
            if not outcome.done:
                self.response_queue.put(
                    session.terminal_result(
                        error=f"advance {index} produced no stream and is not done"
                    )
                )
                return
        if outcome.done or session.complete():
            self.response_queue.put(session.terminal_result())

    @staticmethod
    def _written_shapes(
        slot: DeliverySlot, written: Dict[str, torch.Tensor]
    ) -> Dict[str, Tuple[Tuple[int, ...], torch.dtype]]:
        """The shape and dtype of each view a pipeline reports as written, after
        checking that each one is a view into the slot it was given."""
        shapes = {}
        for name, tensor in written.items():
            if name not in slot:
                raise ValueError(f"stream {name!r} was not declared")
            if not slot[name].is_view_of(tensor):
                raise ValueError(f"stream {name!r}: the reported tensor is not a view of the slot")
            shapes[name] = (tuple(tensor.shape), tensor.dtype)
        return shapes

    def _service_commands_now(
        self, session: SessionState, pool: SlotPool, pipeline_state: Any
    ) -> Optional[_SessionExit]:
        """Handle every command already waiting on either socket, controls first,
        without waiting for more. Returns why the session is to end if a close or
        the shutdown signal was among them."""
        commands = self._drain_controls() + self.requests_ipc.drain()
        return self._handle_session_commands(commands, session, pool, pipeline_state)

    def _wait_for_commands(
        self,
        session: SessionState,
        pool: SlotPool,
        pipeline_state: Any,
        *,
        timeout: Optional[float],
    ) -> Optional[_SessionExit]:
        """Wait up to ``timeout`` seconds (``None``: no limit) for a command on
        either socket, then handle it and everything behind it, controls first.
        Returns ``None`` when the session goes on, the reason when a command ended
        it, and ``IDLE_TIMEOUT`` when nothing arrived in time. Both sockets are
        watched in slices of ``POLL_TIMEOUT``, so a slot return or a close is seen
        whatever sits unread on the request socket."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            commands = self._drain_controls() + self.requests_ipc.drain()
            if commands:
                return self._handle_session_commands(commands, session, pool, pipeline_state)
            if deadline is not None and time.monotonic() >= deadline:
                return _SessionExit.IDLE_TIMEOUT
            try:
                message = self.requests_ipc.get(timeout=POLL_TIMEOUT)
            except queue.Empty:
                continue
            commands = [message] + self._drain_controls() + self.requests_ipc.drain()
            return self._handle_session_commands(commands, session, pool, pipeline_state)

    def _handle_session_commands(
        self, commands: List[Any], session: SessionState, pool: SlotPool, pipeline_state: Any
    ) -> Optional[_SessionExit]:
        """Handle commands in order until one ends the session; the rest are
        answered as strays. Returns why the session ends, or ``None``."""
        for i, message in enumerate(commands):
            ending = self._handle_session_command(message, session, pool, pipeline_state)
            if ending is not None:
                for stray in commands[i + 1 :]:
                    self._answer_stray_command(stray)
                return ending
        return None

    def _handle_session_command(
        self, message: Any, session: SessionState, pool: SlotPool, pipeline_state: Any
    ) -> Optional[_SessionExit]:
        """Handle one command. Returns why the session ends when it does: the
        close command, or the executor's shutdown signal."""
        if message is None:
            return _SessionExit.SHUTDOWN
        if isinstance(message, SessionSubmit):
            if message.session_id != session.session_id:
                self._answer_stray_command(message)
                return None
            reply = SessionAdmitted(session.session_id)
            try:
                # The pipeline's schema check first; ``admit`` mutates nothing
                # until its own checks have passed.
                self.pipeline.validate_input(pipeline_state, message.name, message.value)
                reply.advance_index = session.admit(message.name, message.value)
            except SessionQueueFull as e:
                reply.error, reply.failure = str(e), AdmissionFailure.QUEUE_FULL
            except SessionEnded as e:
                reply.error, reply.failure = str(e), AdmissionFailure.ENDED
            except (ValueError, TypeError) as e:
                reply.error, reply.failure = str(e), AdmissionFailure.INVALID
            self.response_queue.put(reply)
            return None
        if isinstance(message, SessionControl):
            if message.type is ControlType.KEEP_ALIVE:
                return None  # its arrival already restarted the idle wait
            if message.session_id != session.session_id:
                self._answer_stray_command(message)
                return None
            if message.type is ControlType.CLOSE:
                return _SessionExit.CLOSE
            if message.type is ControlType.SLOT_RETURN:
                try:
                    pool.release(message.slot_id)
                except ValueError as e:
                    logger.warning(f"Worker {self.device_id}: {e}")
            return None
        if isinstance(message, SessionOpen):
            # The client refuses a second open itself; one that was already in
            # flight is refused here rather than queued behind an open-ended session.
            message.request.refs_from_shm()
            self.response_queue.put(
                SessionOpened(
                    message.request.request_id,
                    error=f"session {session.session_id} is open on this model instance",
                    error_type="capacity",
                )
            )
            return None
        if isinstance(message, DiffusionRequest):
            # The model instance accepts nothing else while a session is open: the
            # same capacity refusal a second open gets. The payload is taken so its
            # shared memory is released.
            message.refs_from_shm()
            self.response_queue.put(
                DiffusionResponse(
                    request_id=message.request_id,
                    error_msg=f"session {session.session_id} is open on this model instance",
                    error_type="capacity",
                )
            )
            return None
        logger.warning(
            f"Worker {self.device_id}: dropped {type(message).__name__} during a session"
        )
        return None

    def _session_follower(self, req: DiffusionRequest) -> None:
        """Ranks other than 0: run the same pipeline calls as rank 0, in the order
        rank 0 broadcasts them. They write no slot and answer no client."""
        self._merge_defaults(req)
        self.pipeline.prepare_request(req)  # as process_request does, on every rank
        tag, _ = self._recv_tag()
        if tag is not _RankTag.OPEN_GO:
            return
        try:
            pipeline_state, _ = self._open_pipeline(req, None)
        except Exception:
            return  # rank 0's open failed; this rank's own would have ended it
        while True:
            tag, inputs = self._recv_tag()
            if tag is _RankTag.CLOSE:
                break
            if tag is not _RankTag.ADVANCE:
                raise RuntimeError(f"unexpected session tag {tag!r}")
            try:
                self._advance_pipeline(pipeline_state, inputs, None, "session advance")
            except Exception:
                continue  # rank 0 reports it; keep following its tags
        self._close_pipeline(pipeline_state, "session")


def run_diffusion_worker(
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
    request_queue_addr: Optional[str],
    response_queue_addr: Optional[str],
    visual_gen_args: "VisualGenArgs",
    log_level: str = "info",
    req_hmac_key: Optional[bytes] = None,
    resp_hmac_key: Optional[bytes] = None,
    local_rank: Optional[int] = None,
    in_client_process: bool = False,
    parent_pid: Optional[int] = None,
    control_queue_addr: Optional[str] = None,
    ctl_hmac_key: Optional[bytes] = None,
):
    """Entry point for worker process.

    ``in_client_process``: True only when this worker runs inside the client
    process. Declared by the launch site — never derive it from the
    environment here, the env writes below make every worker look external.
    """
    # This native watchdog starts before CUDA, distributed, or model
    # initialization and does not depend on the Python GIL. It follows the
    # coordinator process with a pidfd where available and otherwise polls the
    # parent relationship, regardless of which coordinator thread spawned this
    # worker.
    if parent_pid is not None:
        try:
            watchdog_warning = _start_coordinator_watchdog(parent_pid)
            if watchdog_warning is not None:
                logger.warning(f"VisualGen worker coordinator watchdog: {watchdog_warning}")
        except Exception as e:
            logger.error(f"VisualGen worker could not supervise its coordinator: {e}")
            raise

    try:
        # Set log level before any other work so loading logs are visible
        logger.set_level(log_level)

        # Setup distributed env — use PyTorch distributed, not MPI
        os.environ["TLLM_DISABLE_MPI"] = "1"
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)

        # Determine local_rank: explicit arg > LOCAL_RANK env > global rank.
        # In multi-node runs (torchrun / srun --ntasks-per-node) SLURM/torchelastic
        # sets LOCAL_RANK; in single-node mp.Process mode it equals the global rank.
        _local_rank = (
            local_rank if local_rank is not None else int(os.environ.get("LOCAL_RANK", rank))
        )
        os.environ["LOCAL_RANK"] = str(_local_rank)

        # Use local_rank for device assignment so that each node's ranks map to
        # GPUs 0..gpus_per_node-1 rather than wrapping the global rank.
        device_id = _local_rank % torch.cuda.device_count() if torch.cuda.is_available() else 0
        if torch.cuda.is_available():
            torch.cuda.set_device(device_id)
            check_cuda_architecture_supported(device_id)
            try:
                configure_cpu_affinity(device_id)
            except Exception as e:
                logger.warning(
                    f"[rank {rank}] NUMA-aware CPU affinity setup failed: {e}. "
                    f"The worker will run without NUMA pinning, which may impact "
                    f"performance."
                )

        init_torch_distributed(
            backend="cuda:nccl,cpu:gloo" if torch.cuda.is_available() else "gloo",
            init_method="env://",
            world_size=world_size,
            rank=rank,
            device_id=torch.device(f"cuda:{device_id}") if torch.cuda.is_available() else None,
        )

        executor = DiffusionExecutor(
            request_queue_addr=request_queue_addr,
            response_queue_addr=response_queue_addr,
            device_id=device_id,
            visual_gen_args=visual_gen_args,
            req_hmac_key=req_hmac_key,
            resp_hmac_key=resp_hmac_key,
            in_client_process=in_client_process,
            control_queue_addr=control_queue_addr,
            ctl_hmac_key=ctl_hmac_key,
        )
        executor.serve_forever()
        if executor.pipeline is not None:
            executor.pipeline.cleanup()
        dist.destroy_process_group()

    except Exception as e:
        logger.error(f"Worker failed: {e}")
        traceback.print_exc()
        raise


def _consume_exception(fut: "asyncio.Future") -> None:
    if not fut.cancelled():
        fut.exception()


class SessionsUnsupported(ValueError):
    """A session was requested from a pipeline that does not implement the session
    hooks. Refused by the worker before any pipeline work, as the caller's mistake."""


class FatalRankError(BaseException):
    """A rank hit an exception while its peers may be inside a collective. Raised
    only when the rank shares the client's process; a dedicated worker process
    exits instead. Either way the coordinator sees a dead worker and aborts the
    group, so the client gets the failure raised, not a terminal result. Not an
    ``Exception``: like ``SystemExit`` it ends the rank's thread, so no handler
    on the way out mistakes it for a failure to report."""


class PendingSubmit(NamedTuple):
    """The one submit operation in progress: its input and the admission future
    the caller waits on (result: advance index; exception: rejection;
    cancelled: the caller gave up before the reply)."""

    name: str
    admission: Future


class SessionChannel:
    """Coordinator side of one session: the hand-off point between the main
    thread (the public ``VisualGenSession`` calls) and the client's background
    thread (which routes the worker's replies here). Nine members, one per
    fact, each described at its field below.
    """

    def __init__(self, session_id: int):
        self.session_id = session_id
        # The open reply: the capabilities on success, the open error otherwise. Set
        # once, never replaced, so it is read without the lock; its own state is atomic.
        self._opened: Future[Dict[str, Any]] = Future()
        # Guards every member below: which operations exist and that they agree.
        self._lock = threading.Lock()
        # The client's mapping of the delivery slots, slots[slot_id][stream], from a
        # successful open until the session ends; dropping it unmaps the storage.
        self._slots: Optional[List[Dict[str, torch.Tensor]]] = None
        # None until a close begins; then the close operation. Completes on the
        # worker's confirmation, or with the sticky failure if the worker group died.
        self._closed: Optional[Future] = None
        # The one submit in flight: input name plus the admission reply. Set until
        # its caller finishes, so a second submit is refused meanwhile.
        self._submit: Optional[PendingSubmit] = None
        # The one read in flight. A fresh future per wait: the router completes it
        # when the queue changes; the reader re-arms under the lock, never missing one.
        self._reader: Optional[Future] = None
        # Unread stream entries in order: MEDIA and TERMINAL from the worker, then
        # one client-local ENDED or CLOSED entry. Bounded by the slot count plus two.
        self._results: Deque[SessionResultMessage] = deque()
        # Lockstep only: media the worker owes the client and the client has not
        # taken, from the initial result and each admission. advance() refuses while > 0.
        self._lockstep_due = 0

    # ---- derived views

    @property
    def capabilities(self) -> Optional[Dict[str, Any]]:
        fut = self._opened
        if fut.done() and not fut.cancelled() and fut.exception() is None:
            return fut.result()
        return None

    @property
    def slots(self) -> List[Dict[str, torch.Tensor]]:
        """The client's mapping of the delivery slots, ``slots[slot_id][stream]``.
        Only while the session is open. Read from a result builder, which ``take``
        runs inside its critical section, so no lock is taken here."""
        slots = self._slots
        if slots is None:
            raise RuntimeError(f"session {self.session_id} is not open")
        return slots

    def wait_open(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Block until the worker answered the open; the capabilities, or the open
        error raised."""
        return self._opened.result(timeout)

    def wait_open_async(self) -> "asyncio.Future":
        """The open reply as a future of the running event loop."""
        return asyncio.wrap_future(self._opened)

    def on_open(self, callback: Callable[[Optional[BaseException]], None]) -> None:
        """Run ``callback`` once the open is answered, at once if it already was:
        with ``None`` on success, else with the open error."""

        def done(fut: Future) -> None:
            if fut.cancelled():
                callback(RuntimeError("open cancelled"))
            else:
                callback(fut.exception())

        self._opened.add_done_callback(done)

    @property
    def failure(self) -> Optional[BaseException]:
        """The sticky worker failure, if the worker group is gone."""
        close = self._closed
        if close is not None and close.done() and not close.cancelled():
            return close.exception()
        return None

    @property
    def lease_lost(self) -> Optional[str]:
        """Why views over the slots are no longer usable; ``None`` while they are.
        Set as soon as a close has begun, requested or worker-initiated."""
        close = self._closed
        if close is None:
            return None
        failure = self.failure
        return str(failure) if failure is not None else f"session {self.session_id} is closed"

    def closing(self) -> Optional[Future]:
        """The close operation, if one has begun."""
        return self._closed

    def unread_media(self) -> bool:
        with self._lock:
            return any(m.type is ResultType.MEDIA for m in self._results)

    def lockstep_due(self) -> int:
        with self._lock:
            return self._lockstep_due

    def _queued_input(self) -> Optional[str]:
        caps = self.capabilities or {}
        if not caps.get("lockstep"):
            return None
        for spec in caps.get("inputs", []):
            if spec["policy"] == "queued":
                return spec["name"]
        return None

    # ---- opening (background thread)

    def open_succeeded(
        self, capabilities: Dict[str, Any], slots: List[Dict[str, torch.Tensor]]
    ) -> None:
        with self._lock:
            self._slots = slots
            if capabilities.get("lockstep") and capabilities.get("initial_result"):
                self._lockstep_due = 1
        self._complete(self._opened, capabilities)

    def open_failed(self, error: Exception) -> None:
        self._complete(self._opened, error=error)
        with self._lock:
            close = self._closed
        if close is not None:
            self._complete(close, None)  # a close requested while opening: nothing to send

    # ---- results

    def deliver(self, message: SessionResultMessage) -> None:
        with self._lock:
            close = self._closed
            if (close is not None and close.done()) or (
                self._results and self._results[-1].type is ResultType.CLOSED
            ):
                return  # after the end: a stray
            self._results.append(message)
            reader = self._reader
        self._notify(reader)

    def reserve_read(self) -> None:
        with self._lock:
            if self._reader is not None:
                raise SessionError("a result request is already in progress on this session")
            self._reader = Future()

    def release_read(self) -> None:
        with self._lock:
            self._reader = None

    def take(
        self,
        build: Callable[[SessionResultMessage], Any],
        expected_index: Any = None,
    ) -> Any:
        """One step of a reserved read. Returns the built result,
        ``ResultType.ENDED`` when the stream has ended, or ``None`` when nothing
        is readable yet; in that
        case the wakeup is armed in the same critical section, so a delivery,
        close or failure landing afterwards is never missed by the following
        ``notification()``. Raises the sticky failure or ``SessionEnded`` after an
        explicit close. With ``expected_index`` (an advance), media at the head
        carrying another advance index raises ``SessionError`` before anything is
        taken, so the head stays readable. Builds under the lock, so the mapping
        cannot be torn down meanwhile."""
        with self._lock:
            failure = self.failure
            if failure is not None:
                raise failure
            if not self._results:
                if self._closed is not None:
                    raise SessionEnded(f"session {self.session_id} is closed")
                if self._reader is not None and self._reader.done():
                    self._reader = Future()  # re-armed with the predicate still false
                return None
            head = self._results[0]
            if head.type is ResultType.CLOSED:
                raise SessionEnded(f"session {self.session_id} is closed")
            if head.type is ResultType.ENDED:
                return ResultType.ENDED
            if (
                expected_index is not None
                and head.type is ResultType.MEDIA
                and head.advance_index != expected_index
            ):
                raise SessionError(
                    f"advance() admitted index {expected_index} but the next result is advance "
                    f"{head.advance_index}; the worker broke result ordering"
                )
            result = build(head)
            self._results.popleft()
            if head.type is ResultType.TERMINAL:
                self._results.appendleft(
                    SessionResultMessage.end(self.session_id, ResultType.ENDED)
                )
            elif head.type is ResultType.MEDIA and self._lockstep_due > 0:
                self._lockstep_due -= 1
            return result

    def notification(self) -> Future:
        """The future to await before the next ``take``: whatever ``take`` armed
        when it found nothing readable. Never re-armed here: a completed one
        means something changed since, and the next ``take`` sees it."""
        with self._lock:
            if self._reader is None:
                raise SessionError("no read is reserved")
            return self._reader

    def _notify(self, reader: Optional[Future]) -> None:
        if reader is not None:
            self._complete(reader, None)

    # ---- admission

    def reserve_submit(self, name: str, for_advance: bool = False) -> Future:
        """Register the one submit in progress; returns its admission future. For
        an advance, also refuse while an earlier result is still owed; the check
        and the reservation are one step."""
        with self._lock:
            failure = self.failure
            if failure is not None:
                raise failure
            if self._closed is not None:
                raise SessionEnded(f"session {self.session_id} is closed")
            if self._submit is not None:
                raise SessionError("a submit is already in progress on this session")
            if for_advance and self._lockstep_due:
                raise SessionError(
                    f"{self._lockstep_due} earlier result(s) are unread; take them through results() "
                    "before calling advance()"
                )
            fut: Future = Future()
            self._submit = PendingSubmit(name, fut)
            return fut

    def plan_advance(self, name: str) -> Optional[Future]:
        """Start an advance under a reserved read: ``None`` when the head of the
        stream is already a terminal (take it, submit nothing), else the
        admission future of the reserved submit. Sticky failure first; an ended
        or explicitly closed stream ends the call; a retained terminal is
        claimable even after a graceful worker close; only then does a close
        prevent a new submit."""
        with self._lock:
            failure = self.failure
            if failure is not None:
                raise failure
            head = self._results[0] if self._results else None
            if head is not None and head.type in (ResultType.CLOSED, ResultType.ENDED):
                raise SessionEnded(f"session {self.session_id} has ended")
            if head is not None and head.type is ResultType.TERMINAL:
                return None
            if self._closed is not None:
                raise SessionEnded(f"session {self.session_id} is closed")
            if self._submit is not None:
                raise SessionError("a submit is already in progress on this session")
            if self._lockstep_due:
                raise SessionError(
                    f"{self._lockstep_due} earlier result(s) are unread; take them through results() "
                    "before calling advance()"
                )
            fut: Future = Future()
            self._submit = PendingSubmit(name, fut)
            return fut

    def admitted(self, index: Optional[int], error: Optional[Exception]) -> None:
        """The worker's answer to the one pending submit (background thread)."""
        with self._lock:
            pending = self._submit
            if pending is None:
                logger.warning(
                    f"DiffusionClient: admission for session {self.session_id} with no submit pending"
                )
                return
            if error is None and pending.name == self._queued_input():
                self._lockstep_due += 1  # owed from now on, whether or not the caller still waits
            # Completed under the lock: either the caller still owns the future
            # and retires the record when done, or it had cancelled and the
            # record retires here, with the reply. No window between the two.
            if not self._complete(pending.admission, index, error):
                self._submit = None

    def withdraw_submit(self, fut: Future) -> None:
        """A reserved submit that was never sent (refused locally): the record
        goes, nothing is owed, no index moved."""
        with self._lock:
            pending = self._submit
            if pending is not None and pending.admission is fut:
                self._submit = None

    def release_submit(self, fut: Future) -> None:
        """The caller of the submit that owns ``fut`` is done with it. A reply
        already received retires the record; a caller leaving before the reply
        marks it cancelled and the reply retires it."""
        with self._lock:
            pending = self._submit
            if pending is None or pending.admission is not fut:
                return
            if fut.done() and not fut.cancelled():
                self._submit = None
                return
            fut.cancel()

    # ---- close and failure

    def request_close(self) -> Tuple[Future, bool]:
        """The client closes: discard the unread stream, settle the submit in
        progress, expire the leases. Returns the close operation and whether this
        call began it (the caller then arranges the one close command)."""
        with self._lock:
            if self._closed is not None:
                # Already closing or closed (a worker-initiated close may have
                # retained a terminal): the client's close still discards the stream.
                if self.failure is None and not (
                    self._results and self._results[-1].type is ResultType.CLOSED
                ):
                    self._results.clear()
                    self._results.append(
                        SessionResultMessage.end(self.session_id, ResultType.CLOSED)
                    )
                return self._closed, False
            self._closed = Future()
            self._results.clear()
            self._results.append(SessionResultMessage.end(self.session_id, ResultType.CLOSED))
            self._lockstep_due = 0
            pending, self._submit = self._submit, None
            reader = self._reader
        if pending is not None:
            self._complete(pending.admission, error=SessionEnded("session closed"))
        self._notify(reader)
        return self._closed, True

    def worker_closed(self) -> None:
        """The worker confirmed (or initiated) the close. Media still unread is
        dropped with its mappings; a terminal is kept unless the client had
        closed; the close operation completes."""
        with self._lock:
            if self._closed is None:
                self._closed = Future()
                kept = [
                    m for m in self._results if m.type in (ResultType.ENDED, ResultType.TERMINAL)
                ]
                self._results.clear()
                self._results.extend(kept)
            self._lockstep_due = 0
            pending, self._submit = self._submit, None
            reader = self._reader
            self._slots = None
            close = self._closed
        if pending is not None:
            self._complete(pending.admission, error=SessionEnded("session closed"))
        self._complete(close, None)
        self._notify(reader)

    def fail(self, error: Exception) -> None:
        """The worker group is gone. Unless the session already ended gracefully,
        the close operation fails with ``error`` (sticky), every waiter is woken
        with it and the mappings are released. Never raises."""
        with self._lock:
            close = self._closed
            if close is not None and close.done():
                return  # already over
            if close is None:
                self._closed = close = Future()
            # Published inside the critical section: from here every reader and
            # every caller sees the failure, never a graceful close in progress.
            self._complete(close, error=error)
            self._results.clear()
            self._lockstep_due = 0
            pending, self._submit = self._submit, None
            reader = self._reader
            self._slots = None
        # The opening completes after the close outcome: a close callback that
        # fires on it finds the failure already recorded and cannot soften it.
        self._complete(self._opened, error=error)
        if pending is not None:
            self._complete(pending.admission, error=error)
        self._notify(reader)

    @staticmethod
    def _complete(fut: Future, value: Any = None, error: Optional[BaseException] = None) -> bool:
        """Complete a future the other side may have cancelled or completed."""
        try:
            if error is not None:
                fut.set_exception(error)
            else:
                fut.set_result(value)
            return True
        except InvalidStateError:
            return False


class DiffusionRemoteClient:
    """Client proxy for remote DiffusionExecutor in worker processes.

    Internal coordinator-side counterpart to :class:`DiffusionExecutor`. Not
    part of the public ``tensorrt_llm.visual_gen`` API; the user-facing
    entry point is :class:`tensorrt_llm.visual_gen.VisualGen`, which resolves
    every request's seed before reaching :meth:`enqueue_requests`.

    Supports two launch modes:

    **Single-node (default)**
        ``VisualGen`` is called from an ordinary Python script.
        ``DiffusionRemoteClient`` spawns all worker processes locally via
        ``mp.Process`` with ``master_addr=127.0.0.1``. Each worker starts a
        native coordinator watchdog before model initialization and exits if
        this coordinator process dies. The watchdog uses a pidfd where
        available and falls back to low-frequency parent-PID polling. The
        coordinator also monitors worker liveness and terminates the remaining
        local ranks after one exits.

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

        The external launcher owns sibling-rank cleanup. ``torchrun`` monitors
        and terminates its worker group. ``srun`` deployments must enable
        ``KillOnBadExit`` or ``--kill-on-bad-exit`` to provide that guarantee.
    """

    def __init__(
        self,
        args: VisualGenArgs,
    ):
        self.args = args
        self.n_workers = args.parallel_config.n_workers

        # --- Detect external launcher (torchrun / srun) ---
        ext = _detect_external_launch()

        if ext is None:
            # Single-node: coordinator spawns all workers locally
            # Setup distributed env
            self.master_addr = "127.0.0.1"
            self.master_port = find_free_port()

            # Setup IPC addresses
            self.host_ip = get_ip_address()
            req_port, resp_port, ctl_port = find_free_port(), find_free_port(), find_free_port()

            self.request_queue_addr = f"tcp://0.0.0.0:{req_port}"
            self.response_queue_addr = f"tcp://0.0.0.0:{resp_port}"
            self.control_queue_addr = f"tcp://0.0.0.0:{ctl_port}"
            self.req_addr_connect = f"tcp://{self.host_ip}:{req_port}"
            self.resp_addr_connect = f"tcp://{self.host_ip}:{resp_port}"
            self.ctl_addr_connect = f"tcp://{self.host_ip}:{ctl_port}"

        else:
            # rank == 0 guaranteed — ranks 1..N-1 exited in VisualGen.__init__
            rank, local_rank, world_size, master_addr, master_port = ext
            req_port = find_free_port()
            resp_port = find_free_port()
            ctl_port = find_free_port()
            self.master_addr = master_addr
            self.master_port = master_port
            self.request_queue_addr = f"tcp://0.0.0.0:{req_port}"
            self.response_queue_addr = f"tcp://0.0.0.0:{resp_port}"
            self.control_queue_addr = f"tcp://0.0.0.0:{ctl_port}"
            self.req_addr_connect = f"tcp://{master_addr}:{req_port}"
            self.resp_addr_connect = f"tcp://{master_addr}:{resp_port}"
            self.ctl_addr_connect = f"tcp://{master_addr}:{ctl_port}"

        # Generate shared HMAC keys for IPC authentication
        self.req_hmac_key = os.urandom(32)
        self.resp_hmac_key = os.urandom(32)
        self.ctl_hmac_key = os.urandom(32)

        # IPC setup
        self.requests_ipc = None
        self.responses_ipc = None
        self.controls_ipc = None
        self.pending_requests = queue.Queue()
        # Session control commands (slot returns, close, keep-alive): their own
        # outbox and socket, so a request parked on a non-writable request socket
        # never holds them up. Keyed by (session_id, type, slot_id), so repeats
        # coalesce; an entry stays until its send succeeded, so it also stands
        # for a command being retried.
        self._controls: "OrderedDict[Tuple[int, ControlType, Optional[int]], SessionControl]" = (
            OrderedDict()
        )
        self._controls_lock = threading.Lock()
        self.completed_responses: Dict[int, DiffusionResponse] = {}
        self._sessions: Dict[int, SessionChannel] = {}
        self._sessions_lock = threading.Lock()
        # Request ids the caller has given up on (e.g., aresult timed out).
        # _store_response drops late-arriving responses for these ids so a
        # full PipelineOutput tensor does not pin in completed_responses for
        # the process lifetime.
        self._abandoned_request_ids: Set[int] = set()
        # Iteration-stats tracker — populated on lifecycle events (enqueue,
        # request started, response received) and drained by
        # ``get_iteration_stats`` for the /metrics HTTP endpoint.  Mirrors
        # the LLM iteration-stats producer but with a visual-gen-shaped
        # payload.
        self._iter_stats = _IterationStatsTracker()

        # We'll create asyncio primitives in the background thread's event loop
        self._event_loop = None
        self.response_event = None
        self.lock = None
        self.shutdown_event = _Event()
        self.event_loop_ready = _Event()
        self._shutdown_lock = threading.Lock()
        self._shutdown_started = False
        self._shutdown_complete = _Event()
        self._shutdown_error: Optional[BaseException] = None
        self._shutdown_thread: Optional[threading.Thread] = None
        self.worker_processes: List[mp.Process] = []
        self._worker_spawner: Optional[_WorkerProcessSpawner] = None
        self._ext_worker_thread: Optional[threading.Thread] = None
        self._monitor_worker_liveness = False
        self._worker_failure: Optional[str] = None
        self._request_to_send: Optional[DiffusionRequest] = None

        # Start background thread (it will create its own event loop)
        self.background_thread = _Thread(target=self._serve_forever_thread, daemon=True)
        self.background_thread.start()

        # Wait for the background thread to initialize the event loop
        self.event_loop_ready.wait()

        # Pipeline metadata — populated by _wait_ready from the READY signal.
        self.default_generation_params: Dict = {}
        self.extra_param_specs: Dict = {}
        self.supports_image_edit: bool = False
        self.ref_slot_specs: Dict = {}

        # --- Launch workers ---
        # multiprocessing installs its own timeout-less child joins at import
        # time. atexit is LIFO, so this callback must be registered afterward
        # and before startup can block or fail.
        _register_atexit(DiffusionRemoteClient._atexit_shutdown, weakref.ref(self))

        try:
            if ext is None:
                logger.info(f"DiffusionClient: Launching {self.n_workers} workers")
                ctx = _get_mp_context("spawn")
                parent_pid = _get_process_id()
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
                            "visual_gen_args": self.args,
                            "req_hmac_key": self.req_hmac_key,
                            "resp_hmac_key": self.resp_hmac_key,
                            "control_queue_addr": self.ctl_addr_connect,
                            "ctl_hmac_key": self.ctl_hmac_key,
                            "log_level": logger.level,
                            "local_rank": rank,
                            "parent_pid": parent_pid,
                        },
                    )
                    self.worker_processes.append(p)

                # Process.start() can block during spawn bootstrap. Run the
                # finite spawn batch away from Python's signal-handling thread
                # so shutdown can remain bounded. This thread exits as soon as
                # spawning finishes; worker lifetime follows the coordinator
                # process through the native watchdog instead.
                self._worker_spawner = _WorkerProcessSpawner(self.worker_processes)
                self._worker_spawner.start()
                self._worker_spawner.wait_for_spawn()
                self._monitor_worker_liveness = True
            else:
                # External launch: rank 0 runs its own worker in a background thread.
                # Other nodes' workers are already running (they were launched by the
                # external launcher and will connect to our ZMQ server once it binds).
                self._ext_worker_thread = _Thread(
                    target=run_diffusion_worker,
                    kwargs={
                        "rank": rank,
                        "world_size": self.n_workers,
                        "master_addr": master_addr,
                        "master_port": master_port,
                        "request_queue_addr": self.req_addr_connect,
                        "response_queue_addr": self.resp_addr_connect,
                        "visual_gen_args": self.args,
                        "req_hmac_key": self.req_hmac_key,
                        "resp_hmac_key": self.resp_hmac_key,
                        "control_queue_addr": self.ctl_addr_connect,
                        "ctl_hmac_key": self.ctl_hmac_key,
                        "log_level": logger.level,
                        "local_rank": local_rank,
                        "in_client_process": True,
                    },
                    daemon=True,
                )
                self._ext_worker_thread.start()
                self._monitor_worker_liveness = True

            self._wait_ready()
        except BaseException:
            self.shutdown()
            raise

    @staticmethod
    def _atexit_shutdown(
        self_ref: "weakref.ReferenceType[DiffusionRemoteClient]",
    ) -> None:
        instance = self_ref()
        if instance is not None:
            instance.shutdown()

    @staticmethod
    def _close_socket(ipc_queue):
        if ipc_queue and ipc_queue.socket:
            ipc_queue.socket.setsockopt(zmq.LINGER, 0)
            ipc_queue.close()

    def enqueue_requests(self, requests: List[DiffusionRequest]) -> List[int]:
        """Enqueue requests and return their IDs."""
        if self._worker_failure is not None:
            raise RuntimeError(self._worker_failure)

        req_ids = []
        for req in requests:
            self.pending_requests.put(req)
            req_ids.append(req.request_id)
        # Record one snapshot per enqueue so a /metrics consumer sees the
        # queued-request transitions even if the dispatcher drains the queue
        # before the next poll.
        if req_ids:
            self._iter_stats.record_enqueue(self.pending_requests.qsize())
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

        if self._worker_failure is not None:
            responses = [
                DiffusionResponse(request_id=req_id, error_msg=self._worker_failure)
                for req_id in ids
            ]
            return responses[0] if is_single else responses

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

            if self._worker_failure is not None:
                for req_id in ids:
                    if req_id not in results:
                        results[req_id] = DiffusionResponse(
                            request_id=req_id,
                            error_msg=self._worker_failure,
                        )
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
        if self._worker_failure is not None:
            is_single = isinstance(request_ids, int)
            ids = [request_ids] if is_single else request_ids
            responses = [
                DiffusionResponse(request_id=req_id, error_msg=self._worker_failure)
                for req_id in ids
            ]
            return responses[0] if is_single else responses

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
            self.controls_ipc = ZeroMqQueue(
                (self.control_queue_addr, self.ctl_hmac_key),
                is_server=True,
                socket_type=zmq.PUSH,
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
            try:
                self.requests_ipc.put_nowait(None)
            except zmq.Again:
                logger.info("DiffusionClient: Worker request socket is no longer writable")
            finally:
                self._close_socket(self.requests_ipc)

    @staticmethod
    def _discard_unsent_request(request: Optional[Any]) -> None:
        if isinstance(request, SessionOpen):
            request = request.request
        if request is None or not hasattr(request, "refs_from_shm"):
            return  # session commands carry no shared-memory payload
        try:
            # Rebuilding an unsent shared-memory handle releases its producer
            # reference. The restored request is discarded immediately.
            request.refs_from_shm()
        except RuntimeError as e:
            logger.warning(f"DiffusionClient: Failed to release request {request.request_id}: {e}")

    def _process_requests(self):
        """Process pending requests."""
        req = None
        try:
            if self._request_to_send is None:
                req = self.pending_requests.get(timeout=POLL_TIMEOUT)
                if req is None:
                    try:
                        self._send_shutdown()
                    finally:
                        self.shutdown_event.set()
                    return
                self._request_to_send = req

            req = self._request_to_send
            if isinstance(req, DiffusionRequest):
                logger.info(f"DiffusionClient: Sending request {req.request_id}")
            self.requests_ipc.put_nowait(req)
            self._request_to_send = None
            # Once the request has been handed to the workers it becomes the
            # in-flight ("active") request from the client's perspective.
            if isinstance(req, DiffusionRequest):
                self._iter_stats.record_request_started(
                    req.request_id, self.pending_requests.qsize()
                )
        except queue.Empty:
            pass
        except zmq.Again:
            # A PUSH socket becomes non-writable when its worker peer exits.
            # Keep the request at the head of the coordinator's dispatch path
            # and recheck worker state before the next retry.
            worker_failure = self._check_worker_liveness()
            if worker_failure is not None:
                self._abort_worker_group(worker_failure)
        except Exception as e:
            self._request_to_send = None
            logger.error(f"DiffusionClient: Error sending request: {e}")
            logger.error(traceback.format_exc())
            self._session_send_failed(req, e)

    def _retire_unsent(self, session_id: int) -> None:
        """The session is closed on the worker: a submit of it still unsent (parked
        on the request socket, or queued behind other requests) is dropped, so it
        neither holds its payload nor reaches a session that no longer exists;
        its control commands still in the outbox go too. Runs on the background
        thread, which owns the send slot."""
        parked = self._request_to_send
        if isinstance(parked, SessionSubmit) and parked.session_id == session_id:
            self._request_to_send = None
        with self._sessions_lock:  # serialized against a submit being published
            kept = []
            while True:
                try:
                    item = self.pending_requests.get_nowait()
                except queue.Empty:
                    break
                if not (isinstance(item, SessionSubmit) and item.session_id == session_id):
                    kept.append(item)
            for item in kept:
                self.pending_requests.put(item)
        with self._controls_lock:
            for key in [k for k in self._controls if k[0] == session_id]:
                del self._controls[key]

    def _enqueue_control(self, command: SessionControl) -> None:
        key = (command.session_id, command.type, command.slot_id)
        with self._controls_lock:
            if key not in self._controls:
                self._controls[key] = command

    def _process_controls(self):
        """Send the first pending session control command, if any: the control
        socket's counterpart of ``_process_requests``, with the same ``Again``
        retry and the same containment on any other failure. The entry is
        removed by its key after the send, never by position: a close or a
        failure may have retired it meanwhile."""
        with self._controls_lock:
            if not self._controls:
                return
            key, command = next(iter(self._controls.items()))
        try:
            self.controls_ipc.put_nowait(command)
        except zmq.Again:
            worker_failure = self._check_worker_liveness()
            if worker_failure is not None:
                self._abort_worker_group(worker_failure)
            return
        except Exception as e:
            logger.error(f"DiffusionClient: Error sending session control: {e}")
            logger.error(traceback.format_exc())
            with self._controls_lock:
                if self._controls.get(key) is command:
                    del self._controls[key]
            self._session_send_failed(command, e)
            return
        with self._controls_lock:
            if self._controls.get(key) is command:
                del self._controls[key]

    _PAYLOAD_ERRORS = (TypeError, pickle.PicklingError, AttributeError, ValueError)

    def _session_send_failed(self, req: Any, exc: Exception) -> None:
        """A session command did not leave the process. A payload that cannot be
        serialized (nothing was sent, the transport is intact) fails just the
        call that made it: the pending submit gets ``ValueError`` and consumes no
        index; an open gets ``ValueError``, its reservation and its request
        references are released. Any other failure means control messages may be
        lost on a transport we can no longer trust, so the worker group is
        aborted and every session call raises that failure."""
        if not isinstance(req, SessionCommand):
            return
        payload_error = isinstance(exc, self._PAYLOAD_ERRORS) and not isinstance(exc, OSError)
        with self._sessions_lock:
            channel = self._sessions.get(req.session_id)
        if not payload_error or isinstance(req, SessionControl):
            # A failing data request goes back into the send slot so the abort
            # releases its references with everything else, once; a control
            # command owns nothing and must not displace a parked request.
            if isinstance(req, (SessionOpen, SessionSubmit)) and self._request_to_send is None:
                self._request_to_send = req
            self._abort_worker_group(f"DiffusionClient: could not send {type(req).__name__}: {exc}")
            return
        if isinstance(req, SessionSubmit):
            if channel is not None:
                channel.admitted(
                    None, ValueError(f"input {req.name!r} cannot be serialized: {exc}")
                )
            return
        with self._sessions_lock:
            self._sessions.pop(req.session_id, None)
        self._discard_unsent_request(req)
        if channel is not None:
            channel.open_failed(
                ValueError(f"Session open failed: request cannot be serialized: {exc}")
            )

    def _process_responses(self):
        """Poll and process responses."""
        try:
            if self.responses_ipc.poll(timeout=POLL_TIMEOUT):
                response = self.responses_ipc.get()
                if isinstance(response, SessionReply):
                    self._route_session_reply(response)
                elif isinstance(response, DiffusionResponse):
                    if response.request_id == -1:
                        logger.info("DiffusionClient: Received READY signal")

                    if isinstance(response.output, PipelineOutput):
                        response.output.to_tensor()

                    # Schedule the lock acquisition and event setting in the event loop
                    asyncio.run_coroutine_threadsafe(
                        self._store_response(response), self._event_loop
                    )
        except Exception as e:
            logger.error(f"DiffusionClient: Error processing response: {e}")

    async def _store_response(self, response: DiffusionResponse):
        """Store response in the completed_responses dict (async helper).

        Drops the response if the request id has been abandoned so that
        late-arriving responses for timed-out requests do not leak into
        ``completed_responses`` for the process lifetime.
        """
        async with self.lock:
            if response.request_id in self._abandoned_request_ids:
                self._abandoned_request_ids.discard(response.request_id)
                # The request was abandoned — still mark it complete in the
                # iteration-stats so ``numActiveRequests`` decrements.
                if response.request_id != -1:
                    self._iter_stats.record_request_completed(
                        response.request_id, self.pending_requests.qsize()
                    )
                return
            self.completed_responses[response.request_id] = response
        # Record completion outside the asyncio lock to avoid blocking the
        # event loop on the (uncontended) tracker mutex.  The READY signal
        # uses request_id == -1 and is not tracked as a real request.
        if response.request_id != -1:
            self._iter_stats.record_request_completed(
                response.request_id, self.pending_requests.qsize()
            )
        self.response_event.set()

    # ------------------------------------------------------------------ sessions

    def open_session(self, request: DiffusionRequest) -> SessionChannel:
        """Send ``SessionOpen``; the caller waits through ``channel.wait_open`` (or
        its async form) until the worker has opened it.

        One session per model instance: while another session is opening or open
        (its entry stays in ``_sessions`` until its close is confirmed or its open
        failed), this raises ``MemoryError`` at once, without a round trip to a
        worker that may be busy, and releases the request's shared-memory
        payload."""
        channel = SessionChannel(request.request_id)
        # Failure check, reservation and enqueue under the one lock the abort path
        # takes to record the failure and snapshot the channels: an open either
        # lands in that snapshot with its message already queued (both are then
        # failed and released by the abort) or sees the failure and refuses.
        with self._sessions_lock:
            if self._worker_failure is not None:
                self._discard_unsent_request(request)
                raise RuntimeError(self._worker_failure)
            if self._sessions:
                open_id = next(iter(self._sessions))
                self._discard_unsent_request(request)
                raise MemoryError(
                    f"session {open_id} is open on this model instance; close it before "
                    "opening another"
                )
            self._sessions[request.request_id] = channel
            self.pending_requests.put(SessionOpen(request))
        return channel

    async def open_session_async(self, request: DiffusionRequest) -> SessionChannel:
        """``open_session`` plus the wait, safe against the caller's cancellation.

        The opening future is shielded: cancelling the caller never cancels it,
        so the worker's reply always completes it. On cancellation the session is
        closed (the one close operation: sent once the open is acknowledged,
        settled if the open fails) and the ``CancelledError`` propagates."""
        channel = self.open_session(request)
        opening = channel.wait_open_async()
        try:
            await asyncio.shield(opening)
        except asyncio.CancelledError:
            # Nobody awaits the wrapper any more; its eventual exception is the
            # channel's business (recorded there), not an unretrieved one.
            opening.add_done_callback(_consume_exception)
            self.session_close(channel)
            raise
        return channel

    def session_submit(
        self, channel: SessionChannel, name: str, value: Any, for_advance: bool = False
    ) -> Future:
        """Send one value; the future resolves to the advance index or raises
        ``ValueError`` / ``SessionQueueFull`` / ``SessionEnded``. The caller must
        hand the future to ``channel.release_submit`` when done with it."""
        if self._worker_failure is not None:
            raise RuntimeError(self._worker_failure)
        inputs = (channel.capabilities or {}).get("inputs", [])
        if not any(s["name"] == name for s in inputs):
            raise ValueError(f"unknown input {name!r}; declared: {[s['name'] for s in inputs]}")
        if for_advance:
            # The stream decides first: a terminal at the head means nothing is
            # submitted, so the unused value is not even looked at.
            fut = channel.plan_advance(name)
            if fut is None:
                return None
        else:
            fut = channel.reserve_submit(name)
        # Reserved, not yet sent: until the command is in the queue any exit,
        # an interrupt included, withdraws the reservation.
        try:
            # Publication is conditional on the session still being live and
            # happens under the lock that retirement (close, abort) takes, so a
            # command is either seen by the retirement that follows or never
            # queued behind a close that came first.
            with self._sessions_lock:
                if self._worker_failure is not None:
                    raise RuntimeError(self._worker_failure)
                if channel.closing() is not None:
                    raise SessionEnded(f"session {channel.session_id} is closed")
                self.pending_requests.put(SessionSubmit(channel.session_id, name, value))
        except BaseException:
            channel.withdraw_submit(fut)
            raise
        return fut

    def session_return_slot(self, channel: SessionChannel, slot_id: int) -> None:
        if self._worker_failure is None:
            self._enqueue_control(
                SessionControl(channel.session_id, ControlType.SLOT_RETURN, slot_id)
            )

    def session_keep_alive(self, channel: SessionChannel) -> None:
        """Tell the worker the client is still there (restarts its idle clock);
        repeats coalesce in the outbox while one is pending."""
        if self._worker_failure is None and channel.lease_lost is None:
            self._enqueue_control(SessionControl(channel.session_id, ControlType.KEEP_ALIVE))

    def session_close(self, channel: SessionChannel) -> Future:
        """Close the session; the future completes on the worker's confirmation.
        The first call begins the one close operation and arranges the one close
        command: sent once the open is acknowledged (so the worker has the
        session before its close can reach it on the other socket), or nothing
        if the open fails. Later calls return the same operation."""
        close, first = channel.request_close()
        if first:

            def after_open(error: Optional[BaseException]) -> None:
                if error is not None:
                    channel.open_failed(error)
                    return
                if self._worker_failure is None:
                    self._enqueue_control(SessionControl(channel.session_id, ControlType.CLOSE))

            channel.on_open(after_open)
        return close

    def _route_session_reply(self, reply: Any) -> None:
        with self._sessions_lock:
            channel = self._sessions.get(reply.session_id)
        if channel is None:
            # Normal after a close: a reply to a command that crossed it in flight.
            logger.debug(f"DiffusionClient: reply for closed session {reply.session_id}")
            return
        if isinstance(reply, SessionOpened):
            if reply.error is not None:
                with self._sessions_lock:
                    self._sessions.pop(reply.session_id, None)
                error_type = reply.error_type
                message = f"Session open failed: {reply.error}"
                exc: Exception = (
                    ValueError(message)
                    if error_type == "client"
                    else MemoryError(message)
                    if error_type == "capacity"
                    else RuntimeError(message)
                )
                channel.open_failed(exc)
                return
            channel.open_succeeded(reply.capabilities or {}, map_slots(reply.slots or []))
        elif isinstance(reply, SessionAdmitted):
            error: Optional[Exception] = None
            if reply.error is not None:
                error = {
                    AdmissionFailure.QUEUE_FULL: SessionQueueFull,
                    AdmissionFailure.ENDED: SessionEnded,
                }.get(reply.failure, ValueError)(reply.error)
            channel.admitted(reply.advance_index, error)
        elif isinstance(reply, SessionResultMessage):
            channel.deliver(reply)
        elif isinstance(reply, SessionClosed):
            with self._sessions_lock:
                self._sessions.pop(reply.session_id, None)
            channel.worker_closed()
            self._retire_unsent(reply.session_id)

    def get_iteration_stats(self) -> List[Dict]:
        """Return all buffered iteration-stats snapshots and clear the buffer.

        Each dict matches the shape documented for visual-gen ``/metrics``:
        ``iter``, ``timestamp``, ``numQueuedRequests``, ``numActiveRequests``,
        ``currentStepsProcessed``, ``currentRequestId``, ``currentRequestStepIdx``.
        Snapshots are appended on lifecycle events (enqueue, request started,
        response received) so the buffer is non-empty even between calls
        unless the executor has been completely idle.
        """
        return self._iter_stats.drain()

    def get_current_iteration_snapshot(self) -> Dict:
        """Return a single snapshot of the current state (no buffering)."""
        return self._iter_stats.current_snapshot(self.pending_requests.qsize())

    async def abandon_request_id(self, request_id: int):
        """Mark a request id as abandoned and drop any cached response.

        Called from the result handle's timeout branch to prevent the
        executor from holding a full ``PipelineOutput`` for a request whose
        caller has stopped waiting. Handles both orderings:

        - Response already arrived between the timeout firing and the
          abandon call → ``pop`` releases it here.
        - Response arrives after the abandon call → ``_store_response``
          checks the abandoned set and drops it on arrival.
        """
        async with self.lock:
            self.completed_responses.pop(request_id, None)
            self._abandoned_request_ids.add(request_id)

    def _cleanup_ipc(self):
        """Cleanup IPC."""
        logger.info("DiffusionClient: Cleaning up IPC")
        self._close_socket(self.requests_ipc)
        self._close_socket(self.responses_ipc)
        self._close_socket(self.controls_ipc)

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
            worker_failure = self._check_worker_liveness()
            if worker_failure is not None:
                self._abort_worker_group(worker_failure)
            if self._worker_failure is None:
                # Controls first: they never wait behind a request. The one
                # request serialized per pass bounds how long they wait by the
                # declared input size, nothing more.
                self._process_controls()
                self._process_requests()
                self._process_responses()
            await asyncio.sleep(0.001)  # Yield control to allow other coroutines to run

        self._cleanup_ipc()

        # Keep the event loop available long enough for outstanding result
        # handles to retrieve the worker-failure response. Explicit or atexit
        # shutdown flips _shutdown_started, after which this thread exits
        # within one polling interval and remains bounded.
        while self._worker_failure is not None and not self._shutdown_started:
            await asyncio.sleep(POLL_TIMEOUT)

    def _check_worker_liveness(self) -> Optional[str]:
        if (
            not self._monitor_worker_liveness
            or self._worker_failure is not None
            or self._shutdown_started
        ):
            return None

        dead_workers = []
        for process in self.worker_processes:
            if not process.is_alive():
                dead_workers.append((process.pid, process.exitcode))
        external_worker_dead = (
            self._ext_worker_thread is not None and not self._ext_worker_thread.is_alive()
        )
        if not dead_workers and not external_worker_dead:
            return None

        if dead_workers:
            statuses = ", ".join(
                f"pid={worker_pid}, exitcode={exitcode}" for worker_pid, exitcode in dead_workers
            )
            detail = f"local worker processes exited: {statuses}"
        else:
            detail = "external-launch worker thread exited"
        return f"DiffusionClient: {detail}"

    def _abort_worker_group(self, worker_failure: str) -> None:
        """Record a terminal worker failure, then kill and reap the worker group."""
        if self._worker_failure is not None or self._shutdown_started:
            return

        # Lifecycle tests build partial clients without session state; nothing to fail then.
        sessions = getattr(self, "_sessions", None)
        sessions_lock = getattr(self, "_sessions_lock", None) or contextlib.nullcontext()
        with sessions_lock:
            # Recorded under the same lock open_session reserves under, so no
            # open can register after this snapshot (see open_session).
            self._worker_failure = worker_failure
            channels = list(sessions.values()) if sessions else []
            if sessions:
                sessions.clear()
        request_to_discard = self._request_to_send
        self._request_to_send = None
        logger.error(self._worker_failure)
        self.shutdown_event.set()
        self.response_event.set()
        try:
            # Nothing queued behind the failure will be sent; release the
            # shared-memory payloads parked there (an unsent open, a generate).
            pending = getattr(self, "pending_requests", None)
            if pending is not None:
                with sessions_lock:  # serialized against a submit being published
                    while True:
                        try:
                            self._discard_unsent_request(pending.get_nowait())
                        except queue.Empty:
                            break
            controls_lock = getattr(self, "_controls_lock", None)
            if controls_lock is not None:
                with controls_lock:
                    self._controls.clear()
            if channels:
                for channel in channels:
                    try:
                        channel.fail(RuntimeError(worker_failure))
                    except Exception as e:
                        logger.error(f"DiffusionClient: failing session {channel.session_id}: {e}")
        finally:
            # Whatever happened above, the survivors are killed: a surviving rank
            # may be blocked in a collective that can never complete after
            # another rank exits, so containment is immediate.
            try:
                for process in self.worker_processes:
                    if process.is_alive():
                        try:
                            process.kill()
                        except ProcessLookupError:
                            pass

                worker_spawner = getattr(self, "_worker_spawner", None)
                if worker_spawner is not None:
                    worker_spawner.reap_started_processes()
                else:
                    for process in self.worker_processes:
                        _reap_worker_process(process)
            finally:
                self._discard_unsent_request(request_to_discard)

    def shutdown(self):
        """Shutdown client and workers."""
        start_cleanup = False
        try:
            with self._shutdown_lock:
                if not self._shutdown_started:
                    self._shutdown_started = True
                    self._shutdown_error = None
                    self._shutdown_thread = _Thread(
                        target=self._perform_shutdown,
                        name="visualgen-shutdown",
                        daemon=True,
                    )
                    start_cleanup = True

            if start_cleanup:
                self._shutdown_thread.start()
        except BaseException:
            # Thread.start() may raise after the OS thread has begun running.
            # If it did not, allow a later explicit or atexit call to retry.
            if self._shutdown_thread is None or not self._shutdown_thread.is_alive():
                with self._shutdown_lock:
                    self._shutdown_started = False
            raise

        pending_error = None
        while not self._shutdown_complete.is_set():
            try:
                self._shutdown_complete.wait()
            except BaseException as e:
                # Python signal handlers always run on the main thread, even
                # when the kernel delivered the signal to another thread.
                # Keep waiting while the cleanup thread reaps non-daemon
                # multiprocessing children, then deliver the interruption.
                if pending_error is None:
                    pending_error = e

        if pending_error is not None:
            raise pending_error
        if self._shutdown_error is not None:
            raise self._shutdown_error

    def _perform_shutdown(self) -> None:
        """Run bounded shutdown work outside Python's signal-handling thread."""
        shutdown_error = None
        try:
            logger.info("DiffusionClient: Shutting down")

            worker_spawner = getattr(self, "_worker_spawner", None)
            if worker_spawner is not None:
                worker_spawner.cancel_spawn()
                # p.start() runs on the spawner thread and cannot be interrupted
                # by Python's main-thread signal handlers. Give the current
                # start a short bounded chance to finish, then reap every
                # registered process whose pid has been published.
                spawn_complete = worker_spawner.wait_for_spawn(
                    timeout=WORKER_SPAWN_SHUTDOWN_TIMEOUT,
                    raise_error=False,
                )
                if not spawn_complete:
                    logger.error(
                        "VisualGen worker spawn batch did not complete within "
                        f"{WORKER_SPAWN_SHUTDOWN_TIMEOUT:.0f}s during shutdown; "
                        "continuing to reap every worker with a published pid"
                    )

            self.pending_requests.put(None)

            self.background_thread.join(timeout=THREAD_TIMEOUT)
            if self.background_thread.is_alive():
                logger.warning("DiffusionClient: Force stopping background thread")
                self.shutdown_event.set()
                self.background_thread.join(timeout=1.0)
        except BaseException as e:
            shutdown_error = e
            logger.error(f"DiffusionClient: Error stopping coordinator thread: {e}")

        try:
            logger.info("DiffusionClient: Stopping workers")
            worker_spawner = getattr(self, "_worker_spawner", None)
            if worker_spawner is not None:
                worker_spawner.reap_started_processes()
            else:
                for process in self.worker_processes:
                    _reap_worker_process(process)

            # External-launch mode: the other ranks' destroy_process_group waits for rank 0's,
            # which runs in this daemon thread and dies with the interpreter if not joined.
            if self._ext_worker_thread is not None and self._ext_worker_thread.is_alive():
                self._ext_worker_thread.join(timeout=EXT_WORKER_TEARDOWN_TIMEOUT)
                if self._ext_worker_thread.is_alive():
                    raise RuntimeError(
                        "Rank 0 worker did not finish its teardown within "
                        f"{EXT_WORKER_TEARDOWN_TIMEOUT:.0f}s; the other ranks may hang in "
                        "destroy_process_group"
                    )
        except BaseException as e:
            if shutdown_error is None:
                shutdown_error = e
            logger.error(f"DiffusionClient: Error stopping workers: {e}")
        finally:
            self._shutdown_error = shutdown_error
            self._shutdown_complete.set()

    def _wait_ready(self):
        """Wait for workers to be ready (sync wrapper for async operation)."""
        logger.info("DiffusionClient: Waiting for workers")

        future = asyncio.run_coroutine_threadsafe(self._wait_ready_async(), self._event_loop)
        try:
            future.result()
        except BaseException:
            future.cancel()
            self.shutdown()
            raise

    async def _wait_ready_async(self):
        """Wait for workers to be ready (async version).

        Polls indefinitely for the ready signal. Raises if any worker process
        dies during initialization.
        """
        if self._worker_failure is not None:
            raise RuntimeError(self._worker_failure)

        start_time = time.time()
        last_log_time = start_time
        log_interval = 300

        while True:
            if self._worker_failure is not None:
                raise RuntimeError(self._worker_failure)

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
                        self.supports_image_edit = bool(payload.get("supports_image_edit", False))
                        self.ref_slot_specs = payload.get("ref_slot_specs", {})
                    if self._worker_failure is not None:
                        raise RuntimeError(self._worker_failure)
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
