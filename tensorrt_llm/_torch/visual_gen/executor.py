import asyncio
import atexit
import os
import queue
import socket
import threading
import time
import traceback
import weakref
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import zmq

from tensorrt_llm._torch.shared_tensor import SharedTensorContainer
from tensorrt_llm._torch.visual_gen.output import PipelineOutput
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
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
    ):
        self.request_queue_addr = request_queue_addr
        self.response_queue_addr = response_queue_addr
        self.device_id = device_id
        self.visual_gen_args = visual_gen_args
        self.resp_hmac_key = resp_hmac_key
        self.in_client_process = in_client_process

        self.pipeline = None  # initialized in _load_pipeline
        self.requests_ipc = None
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.response_queue = queue.Queue()
        self.sender_thread = None

        # Only rank 0 handles IPC
        if self.rank == 0:
            logger.info(f"Worker {device_id}: Connecting to request queue")
            self.requests_ipc = ZeroMqQueue(
                (request_queue_addr, req_hmac_key),
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
        if self.rank == 0 and req is not None:
            # Take the payloads out before the object is pickled, and leave their
            # sizes behind so the peers can size their receive buffers.
            for slot in ("image_reference", "video_reference", "audio_reference"):
                for ref in getattr(req.params, slot, None) or []:
                    payloads.append(ref.content)
                    ref.content = b""
            req.ref_sizes = [len(p) for p in payloads]

        obj_list = [req]
        dist.broadcast_object_list(obj_list, src=0)
        req = obj_list[0]
        if req is None:
            return None

        if self.rank == 0:
            # Read-only views: the src rank only reads its buffer.
            buffers = [torch.frombuffer(p, dtype=torch.uint8) for p in payloads]
        else:
            buffers = [torch.empty(n, dtype=torch.uint8) for n in req.ref_sizes or []]
        for buffer in buffers:
            dist.broadcast(buffer, src=0)

        if self.rank != 0:
            payloads = [b.numpy().tobytes() for b in buffers]

        refs = [
            ref
            for slot in ("image_reference", "video_reference", "audio_reference")
            for ref in getattr(req.params, slot, None) or []
        ]
        if len(refs) != len(payloads):
            # zip() would silently leave the tail of either side behind, and
            # clearing ref_sizes below would erase the evidence.
            raise ValueError(f"expected {len(refs)} reference payloads, got {len(payloads)}.")
        for ref, payload in zip(refs, payloads):
            ref.content = payload
        req.ref_sizes = None
        return req

    def serve_forever(self):
        """Main execution loop."""
        while True:
            req = None
            if self.rank == 0:
                req = self.requests_ipc.get()
                if req is not None:
                    req.refs_from_shm()
                logger.info(f"Worker {self.device_id}: Request available")

            # Skipped at world_size 1: with no peer, the object broadcast would
            # still serialize the whole request to a tensor before finding out.
            if self.world_size > 1:
                req = self._broadcast_request(req)

            if req is None:
                logger.info(f"Worker {self.device_id}: Shutdown signal received")
                if self.rank == 0 and self.sender_thread:
                    self.response_queue.put(None)
                    self.sender_thread.join()
                break

            logger.info(f"Worker {self.device_id}: Processing request {req.request_id}")
            self.process_request(req)

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
            try:
                configure_cpu_affinity(device_id)
            except Exception as e:
                logger.warning(
                    f"[rank {rank}] NUMA-aware CPU affinity setup failed: {e}. "
                    f"The worker will run without NUMA pinning, which may impact "
                    f"performance."
                )

        dist.init_process_group(
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
        )
        executor.serve_forever()
        if executor.pipeline is not None:
            executor.pipeline.cleanup()
        dist.destroy_process_group()

    except Exception as e:
        logger.error(f"Worker failed: {e}")
        traceback.print_exc()
        raise


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
            logger.info(f"DiffusionClient: Sending request {req.request_id}")
            self.requests_ipc.put_nowait(req)
            self._request_to_send = None
            # Once the request has been handed to the workers it becomes the
            # in-flight ("active") request from the client's perspective.
            self._iter_stats.record_request_started(req.request_id, self.pending_requests.qsize())
        except queue.Empty:
            pass
        except zmq.Again:
            # A PUSH socket becomes non-writable when its worker peer exits.
            # Keep the request at the head of the coordinator's dispatch path
            # and recheck worker state before the next retry.
            self._check_worker_liveness()
        except Exception as e:
            self._request_to_send = None
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
            self._check_worker_liveness()
            if self._worker_failure is None:
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

    def _check_worker_liveness(self) -> None:
        if (
            not self._monitor_worker_liveness
            or self._worker_failure is not None
            or self._shutdown_started
        ):
            return

        dead_workers = []
        live_workers = []
        for process in self.worker_processes:
            if process.is_alive():
                live_workers.append(process)
            else:
                dead_workers.append((process.pid, process.exitcode))
        external_worker_dead = (
            self._ext_worker_thread is not None and not self._ext_worker_thread.is_alive()
        )
        if not dead_workers and not external_worker_dead:
            return

        if dead_workers:
            statuses = ", ".join(
                f"pid={worker_pid}, exitcode={exitcode}" for worker_pid, exitcode in dead_workers
            )
            detail = f"local worker processes exited: {statuses}"
        else:
            detail = "external-launch worker thread exited"
        self._worker_failure = f"DiffusionClient: {detail}"
        logger.error(self._worker_failure)
        self.shutdown_event.set()
        self.response_event.set()

        # A distributed worker group cannot continue after losing a rank.
        # SIGKILL the remaining local ranks immediately so they cannot retain
        # model weights while callers observe the failure through result().
        for process in live_workers:
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

            # External-launch mode: join rank-0 worker thread.
            if self._ext_worker_thread is not None and self._ext_worker_thread.is_alive():
                self._ext_worker_thread.join(timeout=WORKER_TIMEOUT)
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
