import asyncio
import os
import queue
import socket
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import zmq

from tensorrt_llm._torch.visual_gen.output import PipelineOutput
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
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

# Default cap on the size of the iteration-stats snapshot buffer used by the
# /metrics endpoint.  Mirrors the LLM ``iter_stats_max_iterations`` default.
_DEFAULT_ITER_STATS_MAX = 1000


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


@dataclass
class DiffusionRequest:
    """Request for diffusion inference.

    Generation parameters live in the optional ``params`` object
    (a :class:`~tensorrt_llm.visual_gen.params.VisualGenParams` instance).
    When ``params`` is ``None`` (the default), the executor creates a
    ``VisualGenParams()`` and fills it with pipeline-specific defaults
    before calling ``pipeline.infer()``.
    """

    request_id: int
    prompt: List[str]
    params: Optional["VisualGenParams"] = None


@dataclass
class DiffusionResponse:
    """Response with model-specific output.

    Attributes:
        request_id: Unique identifier for the request.
        output: Generated media as :class:`PipelineOutput` with the
            model-specific fields populated. Set to ``None`` on the error
            path; on the READY signal it carries a ``dict`` instead.
        error_msg: Error message if generation failed.
        generation: Wall-clock time the executor measured around the
            engine's inference call (host ``time.perf_counter()``), in
            seconds. Default ``0.0`` so the dataclass round-trips through
            pickling across worker/client; the error path leaves it at
            ``0.0``.
    """

    request_id: int
    output: Optional[PipelineOutput] = None
    error_msg: Optional[str] = None
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
    ):
        self.request_queue_addr = request_queue_addr
        self.response_queue_addr = response_queue_addr
        self.device_id = device_id
        self.visual_gen_args = visual_gen_args
        self.resp_hmac_key = resp_hmac_key

        self.pipeline = None  # initialized in _load_pipeline
        self.requests_ipc = None
        self.rank = dist.get_rank()
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
                    },
                )
            )

    def serve_forever(self):
        """Main execution loop."""
        while True:
            req = None
            if self.rank == 0:
                req = self.requests_ipc.get()
                logger.info(f"Worker {self.device_id}: Request available")

            # Broadcast to all ranks. ``req.params.seed`` is already a
            # concrete int — resolved once on the coordinator process at
            # :meth:`VisualGen.generate_async` entry — so the broadcast
            # propagates the same value to every rank.
            obj_list = [req]
            dist.broadcast_object_list(obj_list, src=0)
            req = obj_list[0]

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
                setattr(params, field_name, default_value)

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
        try:
            self._merge_defaults(req)
            cache_key = self.pipeline.warmup_cache_key(
                req.params.height, req.params.width, num_frames=req.params.num_frames
            )
            if self.pipeline._warmed_up_shapes and cache_key not in self.pipeline._warmed_up_shapes:
                logger.warning(
                    f"Requested shape {cache_key} was not warmed up. "
                    f"First request with this shape will be slower due to "
                    f"torch.compile recompilation or CUDA graph capture. "
                    f"Warmed-up shapes: {self.pipeline._warmed_up_shapes}"
                )
            # Host wall-clock around pipeline.infer(). The pipeline already
            # syncs at the end (decode_latents path), so this captures the
            # full executor-side envelope including any pre/post-pipeline work
            # that the per-phase CUDA-event timings on PipelineOutput do not.
            generation_start = time.perf_counter()
            output = self.pipeline.infer(req)
            generation = time.perf_counter() - generation_start  # seconds
            if self.rank == 0:
                # External launch co-locates this worker with the coordinator in one
                # process; to_handle(local=True) hands the tensor over in-process
                # instead of cross-process CUDA IPC (invalid same-process).
                output.to_handle(local=_detect_external_launch() is not None)
                self.response_queue.put(
                    DiffusionResponse(
                        request_id=req.request_id,
                        output=output,
                        generation=generation,
                    )
                )
        except Exception as e:
            logger.error(f"Worker {self.device_id}: Error: {e}")
            logger.error(traceback.format_exc())
            if self.rank == 0:
                self.response_queue.put(
                    DiffusionResponse(request_id=req.request_id, error_msg=str(e))
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
):
    """Entry point for worker process."""
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

        # NCCL_NVLS_ENABLE=0 is required to prevent a hang on Blackwell when
        # VSA (CuTeDSL) + Ulysses is active
        if torch.cuda.is_available() and visual_gen_args is not None:
            _attn = visual_gen_args.attention_config
            _sa = getattr(_attn, "sparse_attention_config", None)
            _is_vsa = (
                getattr(_attn, "backend", "") == "CUTEDSL"
                and getattr(_sa, "algorithm", "") == "vsa"
            )
            _has_ulysses = getattr(visual_gen_args.parallel_config, "ulysses_size", 1) > 1
            if _is_vsa and _has_ulysses:
                os.environ.setdefault("NCCL_NVLS_ENABLE", "0")

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
        )
        executor.serve_forever()
        if executor.pipeline is not None:
            executor.pipeline.cleanup()
        dist.destroy_process_group()

    except Exception as e:
        logger.error(f"Worker failed: {e}")
        traceback.print_exc()


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
                        "visual_gen_args": self.args,
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
                    "visual_gen_args": self.args,
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
            # Once the request has been handed to the workers it becomes the
            # in-flight ("active") request from the client's perspective.
            self._iter_stats.record_request_started(req.request_id, self.pending_requests.qsize())
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
