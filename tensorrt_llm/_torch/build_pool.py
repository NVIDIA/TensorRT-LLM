# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``BuildPool``: worker processes that build this rank's kernels in parallel.

KernelBuilder's build mode and the autotuner's tune mode build an op's kernels on these processes
(:meth:`BuildPool.build`) and go on once the builds have ended. CUTLASS DSL keeps one compile
session per process, so CuTe DSL kernels compile in parallel only in separate processes; the other
compilers run there too, so every op builds the same way.

* A build is a module-level function called with picklable positional arguments (:class:`Build`).
  Its kernel reaches this process the way its compiler caches it: in the compiler's disk cache
  (DeepGEMM, FlashInfer, Triton, fmha_sm100, PyTorch extensions), or, for a CuTe DSL kernel, as an
  object file in the pool's store, which ``process_built`` functions and the runners' shared
  ``kernel_cache`` dicts (``tensorrt_llm._jit_share``) load.
* :func:`process_built` memoizes a CuTe DSL compile function whose kernels the workers build
  several at a time (:meth:`BuildPool.build_kernels`, which ``KernelBuilder.build_kernels`` calls);
  ``fn(*args)`` then loads a kernel a worker built.
* The workers live inside a pass (:meth:`BuildPool.pass_scope`, which KernelBuilder's build pass
  opens, in the memory-profiling warmup only): they start with the pass's first builds, while the
  GPU keeps ``_MIN_FREE_GPU_BYTES`` free, and stop when the outermost pass ends. Each holds a CUDA
  context while it runs, which must be gone before the KV cache estimation measures free memory; a
  warmup with the final KV cache starts none. Outside a pass no worker takes a build, and the
  caller builds in this process.
* The ranks split a build that has a :class:`Claim`: the worker that starts it first claims it and
  makes it, and the other ranks' workers find it claimed and skip it.
* When the ranks share JIT builds (``TLLM_JIT_SHARE_DIR``, or by default the launch's directory on
  each node; see ``tensorrt_llm._jit_share``), the store is ``<that directory>/cute``, and the ranks
  split the CuTe DSL kernels built into it. Otherwise every rank builds each kernel into a store of
  its own, removed when the process exits.
* A worker takes a build only when builds of its kind take long enough to repay a worker
  (:meth:`BuildPool.worth_a_worker`): ``TLLM_KERNEL_BUILDER_MIN_BUILD_S`` seconds (default 5), as
  measured so far in this process and its workers, or as the caller expects until one is measured.
  Shorter builds are made in this process.
* A build that runs a kernel to compile it allocates the kernel's inputs on the rank's GPU. A
  worker frees what a build allocated once the build ends, and starts a build only while the GPU's
  free memory covers it beside the builds running, plus ``_MIN_FREE_GPU_BYTES``: how much a build
  takes is the most a build of its kind took so far in a worker, and a kind not measured yet builds
  one at a time. A build that does not fit with no other build running is left to the caller.
* The workers start with ``subprocess`` in an environment of their own: without the launcher's
  MPI, PMIx and Slurm variables, since importing TensorRT-LLM initializes MPI, and with only this
  rank's GPU visible.
* :meth:`BuildPool.submit` queues builds and returns at once; its :class:`BuildBatch` waits for
  them. A build's function may return a value, which its result carries.
* In a pass that runs beside other work of the rank (``pass_scope(overlapped=True)``, a rehearsal
  beside the checkpoint load), a worker's cost is hidden behind that work: a build that cannot run
  on the pass's threads goes to a worker whatever its time, and nothing is measured in this
  process. A build waits for the GPU's memory there rather than being left to the caller at once,
  since the rank's other work frees memory too.
* The pass's in-process build threads (:attr:`BuildPool.threads`) take builds whose compilers run
  outside the GIL and that do not repay a worker. Each runs on a CUDA stream of its own, and
  starts only while the GPU has room for the bytes it says it allocates, beside the workers'.

``TLLM_KERNEL_BUILDER=0`` starts no worker: every kernel builds where it is first needed.
``TLLM_KERNEL_BUILDER_WORKERS`` caps the workers of a rank (default: this rank's share of the
CPUs, up to 8).
"""

from __future__ import annotations

import atexit
import collections
import contextlib
import dataclasses
import enum
import faulthandler
import functools
import hashlib
import importlib
import os
import pickle  # nosec B403 - messages between this process and its own workers
import queue
import shutil
import struct
import subprocess  # nosec B404 - starts its workers: sys.executable, a fixed argv, no shell
import sys
import tempfile
import threading
import time
import traceback
from typing import IO, Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Sequence, Tuple

from tensorrt_llm.logger import logger

__all__ = [
    "Build",
    "BuildBatch",
    "BuildPool",
    "BuildResult",
    "BuildStrategy",
    "Claim",
    "process_built",
]

ENABLE_ENV = "TLLM_KERNEL_BUILDER"
_WORKERS_ENV = "TLLM_KERNEL_BUILDER_WORKERS"
_MIN_BUILD_ENV = "TLLM_KERNEL_BUILDER_MIN_BUILD_S"
# A build goes to a worker only when builds of its kind take at least this long: a worker
# spends seconds starting and loading its compiler, which a short build does not repay.
_MIN_BUILD_S = 5.0
# Each worker holds a CUDA context (about a GiB) on the rank's GPU while the pass lasts.
_MAX_DEFAULT_WORKERS = 8
# Set in a worker: the store its CuTe DSL kernels go to, and whether the ranks share it.
STORE_ENV = "TLLM_BUILD_POOL_STORE"
_SHARED_ENV = "TLLM_BUILD_POOL_SHARED"
# A worker's environment drops these: importing TensorRT-LLM initializes MPI, which must not
# join the launcher's job.
_LAUNCHER_ENV_PREFIXES = ("OMPI_", "PMIX_", "PMI_", "SLURM_", "MPI_", "I_MPI_", "HYDRA_")
# How much of a worker's stderr a failure report quotes.
_STDERR_TAIL = 2000
# Seconds a stop waits for idle workers to exit before it kills them.
_STOP_TIMEOUT_S = 60.0
# A build whose pickled request is larger (a runner that holds weights) is built here.
_MAX_MESSAGE_BYTES = 64 << 20
# Ends a failed build's report when the build cost the worker its CUDA context: the worker ends.
_WORKER_ENDS = " (the worker's CUDA context is lost: the worker ends)"
# A worker starts only while the rank's GPU has this much memory free: its CUDA context and
# builds take about a GiB of it, and the rank's own warmup needs the rest.
_MIN_FREE_GPU_BYTES = 8 << 30
# In an overlapped pass, a build that does not fit with no other build running looks at the
# GPU's free memory again after this many seconds, until the pass ends.
_PATIENT_POLL_S = 1.0

# How a build ended (BuildResult.status): a worker built it; it was built before; another rank's
# worker claimed it, and makes it; it failed; no worker took it.
BUILT, PRESENT, CLAIMED, FAILED, REFUSED = "built", "present", "claimed", "failed", "refused"

_CacheInfo = collections.namedtuple("CacheInfo", ["hits", "misses", "maxsize", "currsize"])


class BuildStrategy(enum.Enum):
    """How the processes of a launch split an op's kernel builds (the autotuner's
    ``TuningConfig.build_strategy``, KernelBuilder's ``BuildConfig.strategy``).

    * ``INDEPENDENT``: each process builds every kernel it needs.
    * ``PARALLEL``: the processes that share their JIT builds (``TLLM_JIT_SHARE_DIR``, or by
      default the launch's directory on each node; see ``tensorrt_llm._jit_share``) split the
      builds: the process that claims a kernel builds it, and the others load it. Only for
      kernels that the other processes can load from where the builder leaves them.
    """

    INDEPENDENT = "independent"
    PARALLEL = "parallel"


class Claim(NamedTuple):
    """The files through which the ranks that split a build learn which of them makes it, and
    when it has ended: ``claim``, which the worker that starts it creates; ``done``, which the
    build leaves (its kernel's object file, or a mark once it returns); ``failed``."""

    claim: str
    done: str
    failed: str


class Build(NamedTuple):
    """One build for a worker process: ``function`` (``"module:qualname"`` of a module-level
    function), called with ``args``, which must pickle. ``family`` names the kind of build, whose
    seconds the pool measures (:meth:`BuildPool.worth_a_worker`). With a ``claim``, the ranks
    split the build: the worker that starts it first makes it."""

    function: str
    args: tuple = ()
    family: Optional[str] = None
    claim: Optional[Claim] = None


class BuildResult(NamedTuple):
    """How a build ended: ``status`` ``"built"``, ``"present"`` (built before), ``"claimed"``
    (another rank's worker makes it), ``"failed"``, or ``"refused"`` (no worker took it: its
    caller builds it); the seconds the worker took; the error of a failed build; the most GPU
    memory the build allocated at once, in bytes; what the build's function returned."""

    status: str
    seconds: float = 0.0
    error: str = ""
    peak_bytes: int = 0
    value: Any = None


class _Compiler(NamedTuple):
    """A ``process_built`` function's name for its kernels, its ``"module:qualname"``, how long
    a compile takes until one is measured, and its kernels compiled so far, by arguments."""

    name: str
    where: str
    build_seconds: Optional[float]
    memo: Dict[tuple, Any]


def process_built(
    name: str, build_seconds: Optional[float] = None
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Memoize a CuTe DSL kernel compile function whose kernels worker processes can build.

    Use it in place of ``functools.cache`` on a function that compiles one kernel from hashable,
    picklable positional arguments and returns the compiled kernel (what ``cute.compile``
    returns). The function keeps ``cache_info()`` and ``cache_clear()``;
    ``KernelBuilder.build_kernels`` compiles several of its kernels at a time. A call loads the
    kernel from the store when a worker built it, and compiles it here otherwise. In a worker, a
    call compiles the kernel and exports it to the store, so that its rank loads it. ``raw`` is
    the undecorated function.

    Args:
        name: Names the function's kernels in the store; unique per compile function.
        build_seconds: How long one compile takes, until compiles of it are measured; it
            decides whether the workers take them (:meth:`BuildPool.worth_a_worker`).
    """

    def decorate(raw: Callable[..., Any]) -> Callable[..., Any]:
        memo: Dict[tuple, Any] = {}
        lock = threading.Lock()
        counts = {"hits": 0, "misses": 0}

        @functools.wraps(raw)
        def compiled(*args):
            kernel = memo.get(args)
            if kernel is None:
                with lock:
                    kernel = memo.get(args)
                    if kernel is None:
                        counts["misses"] += 1
                        pool = BuildPool.get()
                        kernel = pool.load(name, args)
                        if kernel is None:
                            t0 = time.perf_counter()
                            kernel = raw(*args)
                            pool.record_build(name, time.perf_counter() - t0)
                            pool.store(name, args, kernel)
                        memo[args] = kernel
                        return kernel
            counts["hits"] += 1
            return kernel

        def cache_info() -> _CacheInfo:
            return _CacheInfo(counts["hits"], counts["misses"], None, len(memo))

        def cache_clear() -> None:
            with lock:
                memo.clear()
                counts["hits"] = counts["misses"] = 0

        compiled.raw = raw
        compiled.cache_info = cache_info
        compiled.cache_clear = cache_clear
        compiled._compiler = _Compiler(
            name, f"{raw.__module__}:{raw.__qualname__}", build_seconds, memo
        )
        return compiled

    return decorate


class _Paths(NamedTuple):
    symbol: str
    obj: str
    meta: str
    claim: str
    failed: str


class BuildBatch:
    """The builds of one :meth:`BuildPool.submit` call: :meth:`wait` returns their results, in
    order, once every one has ended."""

    def __init__(self, size: int, on_done: Optional[Callable[[int, BuildResult], None]] = None):
        self.results: List[Optional[BuildResult]] = [None] * size
        self._left = size
        self._on_done = on_done
        self._ended = threading.Condition()

    def settle(self, index: int, result: BuildResult) -> None:
        if self._on_done is not None:
            try:
                self._on_done(index, result)
            except Exception as e:  # noqa: BLE001 - the other builds of the batch still end
                logger.warning(f"[BuildPool] a build's callback raised {type(e).__name__}: {e}")
        with self._ended:
            self.results[index] = result
            self._left -= 1
            if not self._left:
                self._ended.notify_all()

    def done(self) -> bool:
        """Whether every build of the batch has ended."""
        with self._ended:
            return not self._left

    def wait(self, timeout: Optional[float] = None) -> Optional[List[BuildResult]]:
        """The results once every build has ended; None when they have not within ``timeout``
        seconds."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._ended:
            while self._left:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return None
                self._ended.wait(remaining)
        return list(self.results)


@dataclasses.dataclass(eq=False)
class _Task:
    # The pickled (function, args, claim).
    message: bytes
    family: Optional[str]
    batch: BuildBatch
    index: int
    settled: bool = False


def _gpu_has_room() -> bool:
    """Whether the rank's GPU has room for one more worker (``_MIN_FREE_GPU_BYTES`` free)."""
    import torch

    if not torch.cuda.is_available() or not torch.cuda.is_initialized():
        return True
    free, _ = torch.cuda.mem_get_info()
    return free >= _MIN_FREE_GPU_BYTES


def _current_device() -> Optional[int]:
    """This thread's GPU, or None when this process does not use CUDA."""
    import torch

    if not torch.cuda.is_available() or not torch.cuda.is_initialized():
        return None
    return torch.cuda.current_device()


def _free_gpu_bytes(device: Optional[int]) -> Optional[int]:
    """The free memory of GPU ``device``, or None for no GPU."""
    if device is None:
        return None
    import torch

    free, _ = torch.cuda.mem_get_info(device)
    return free


def _cached_gpu_bytes(device: Optional[int]) -> int:
    """The memory this process's allocator holds on GPU ``device`` without using it, which a build
    in this process allocates from before it asks the GPU for more."""
    import torch

    if device is None or not torch.cuda.is_initialized():
        return 0
    return max(0, torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device))


class _Admission:
    """The GPU memory a pass's builds may take, in the workers and on the pass's threads alike: a
    build starts only while the GPU's free memory covers it beside the builds running, plus
    ``_MIN_FREE_GPU_BYTES``. A build's need is the bytes it says it takes, or else the most a build
    of its kind took so far in a worker; a kind not measured yet builds one at a time. A build that
    does not fit with no other build running is refused, or, in an overlapped pass, waits for the
    GPU's memory until the pass ends."""

    def __init__(self, owner: "BuildPool", device: Optional[int]):
        self._owner = owner
        self.device = device
        self._memory = threading.Condition()
        self._stopping = False
        # The builds running: their families and the memory they may take.
        self._running: Dict[int, Tuple[Optional[str], int]] = {}

    def admit(
        self,
        key: int,
        family: Optional[str],
        need: Optional[int] = None,
        in_process: bool = False,
    ) -> bool:
        """Wait until the GPU has room for the build ``key`` (``need`` bytes, or its family's
        measured peak), and count it running; False when it cannot start. ``in_process``: the
        build allocates from this process's allocator, whose cached memory it reuses."""
        with self._memory:
            while not self._stopping:
                free = _free_gpu_bytes(self.device)
                if free is None:
                    return True  # No GPU memory to keep.
                if in_process:
                    free += _cached_gpu_bytes(self.device)
                if need is None:
                    need = self._owner.peak_bytes(family)
                    measured = need is not None
                else:
                    measured = True
                if (
                    not measured
                    and family is not None
                    and any(f == family for f, _ in self._running.values())
                ):
                    fits = False  # A kind of build not measured yet builds one at a time.
                else:
                    taken = sum(nbytes for _, nbytes in self._running.values())
                    fits = free - taken - (need or 0) >= _MIN_FREE_GPU_BYTES
                if fits:
                    self._running[key] = (family, need or 0)
                    return True
                patient = self._owner.overlapping()
                if not self._running and not patient:
                    return False
                self._memory.wait(_PATIENT_POLL_S if patient else None)
                if not measured:
                    need = None  # Measured meanwhile, perhaps.
            return False

    def release(self, key: int) -> None:
        with self._memory:
            self._running.pop(key, None)
            self._memory.notify_all()

    def stop(self) -> None:
        """Refuse the builds waiting for memory, and every build asked for later."""
        with self._memory:
            self._stopping = True
            self._memory.notify_all()


def _default_workers() -> int:
    from .autotuner import _default_parallel_jit_warmup_threads

    return min(_MAX_DEFAULT_WORKERS, _default_parallel_jit_warmup_threads())


class BuildPool:
    """This rank's worker processes, and the store they build CuTe DSL kernels into.

    Inside a pass (:meth:`pass_scope`), :meth:`build` runs builds on the workers and returns once
    they have ended; :meth:`load` loads a CuTe DSL kernel a worker built. The workers start with
    the builds, up to ``TLLM_KERNEL_BUILDER_WORKERS`` and while the GPU has room, and stop when
    the outermost pass ends.
    """

    _instance: Optional["BuildPool"] = None
    _instance_lock = threading.Lock()

    def __init__(
        self,
        workers: Optional[int] = None,
        enabled: Optional[bool] = None,
        store: Optional[str] = None,
    ):
        """``store``: the directory to keep CuTe DSL kernels in, of this rank alone; by default
        ``<TLLM_JIT_SHARE_DIR>/cute`` or a temporary one."""
        self.enabled = os.environ.get(ENABLE_ENV, "1") != "0" if enabled is None else enabled
        if workers is None:
            workers = (
                int(os.environ[_WORKERS_ENV]) if _WORKERS_ENV in os.environ else _default_workers()
            )
        self.max_workers = max(1, workers)
        # In a worker: the store of the rank that started it.
        self.in_worker = STORE_ENV in os.environ
        self._store: Optional[str] = os.environ.get(STORE_ENV) if self.in_worker else store
        self._shared = self.in_worker and os.environ.get(_SHARED_ENV) == "1"
        self._store_ready = False
        self._store_lock = threading.Lock()
        self._lock = threading.Lock()
        self._depth = 0
        self._overlapped = 0
        self._pool: Optional[_Pool] = None
        self._admission: Optional[_Admission] = None
        self._lane: Optional[_ThreadLane] = None
        self._counts: collections.Counter = collections.Counter()
        self._build_s = 0.0
        self._workers_started = 0
        self.min_build_s = float(os.environ.get(_MIN_BUILD_ENV, _MIN_BUILD_S))
        # family -> [builds measured, their seconds]
        self._build_times: Dict[str, List[float]] = {}
        # family -> the most GPU memory a build of it allocated at once in a worker
        self._peak_bytes: Dict[str, int] = {}

    @classmethod
    def get(cls) -> "BuildPool":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # Passes.

    @contextlib.contextmanager
    def pass_scope(self, overlapped: bool = False) -> Iterator[None]:
        """A build pass: workers take the builds asked for inside it, and stop when the
        outermost pass ends. Once a worker takes a build, the rank's shared kernel caches
        (``tensorrt_llm._jit_share``) keep CuTe DSL kernels in the store the workers build into,
        until the pass ends. ``overlapped``: the pass runs beside other work of the rank, so
        nothing is measured in this process and builds wait for memory (see the module)."""
        from tensorrt_llm import _jit_share

        with self._lock:
            self._depth += 1
            self._overlapped += int(overlapped)
        try:
            yield
        finally:
            with self._lock:
                self._depth -= 1
                self._overlapped -= int(overlapped)
                last = self._depth == 0
            if last:
                self.stop()
                if not self.in_worker:
                    _jit_share.set_pool_store(None)

    def accepting(self) -> bool:
        """Whether a worker takes a build asked for now."""
        return self.enabled and not self.in_worker and self._depth > 0

    def overlapping(self) -> bool:
        """Whether an overlapped pass is open (see :meth:`pass_scope`)."""
        return self._overlapped > 0

    @property
    def threads(self) -> Optional["_ThreadLane"]:
        """The pass's in-process build threads; None outside a pass and with workers off, when
        the caller builds where it is."""
        if not self.accepting():
            return None
        with self._lock:
            if self._lane is None:
                from .autotuner import _default_parallel_jit_warmup_threads

                self._lane = _ThreadLane(
                    self, self._ensure_admission(), _default_parallel_jit_warmup_threads()
                )
            return self._lane

    def takes(self, args: tuple) -> bool:
        """Whether a worker takes a build asked for now with ``args`` (see :meth:`build`)."""
        if not self.accepting():
            return False
        try:
            message = pickle.dumps(args, protocol=pickle.HIGHEST_PROTOCOL)
        except (pickle.PicklingError, TypeError, AttributeError):
            return False
        return len(message) <= _MAX_MESSAGE_BYTES

    def worth_a_worker(
        self, family: str, expected_s: Optional[float] = None, thread_safe: bool = False
    ) -> bool:
        """Whether a worker should take a build of ``family`` now: while workers take builds,
        when builds of the family take at least ``TLLM_KERNEL_BUILDER_MIN_BUILD_S`` seconds
        (default 5; 0 sends every build to the workers). How long they take is their mean so far,
        here or in a worker, or ``expected_s`` until one is measured; a family that has neither
        builds in this process. In an overlapped pass, a build that cannot run on the pass's
        threads (not ``thread_safe``) goes to a worker whatever its time."""
        if not self.accepting():
            return False
        if self.min_build_s <= 0 or (self.overlapping() and not thread_safe):
            return True
        estimate = self.build_seconds(family)
        if estimate is None:
            estimate = expected_s
        return estimate is not None and estimate >= self.min_build_s

    def unmeasured(self, family: str, expected_s: Optional[float] = None) -> bool:
        """Whether :meth:`worth_a_worker` cannot decide for ``family`` yet: while workers take
        builds and the rule is on, before a build of the family is measured, without
        ``expected_s``, outside an overlapped pass. The caller then makes one build here, timed
        (:meth:`record_build`)."""
        return (
            self.accepting()
            and not self.overlapping()
            and self.min_build_s > 0
            and expected_s is None
            and self.build_seconds(family) is None
        )

    def build_seconds(self, family: str) -> Optional[float]:
        """The mean of the builds of ``family`` measured so far, or None."""
        with self._lock:
            measured = self._build_times.get(family)
        return measured[1] / measured[0] if measured else None

    def record_build(self, family: str, seconds: float) -> None:
        """Measure a build of ``family`` that took ``seconds`` here."""
        with self._lock:
            self._record_build(family, seconds)

    def _record_build(self, family: str, seconds: float) -> None:
        measured = self._build_times.setdefault(family, [0, 0.0])
        measured[0] += 1
        measured[1] += seconds

    def peak_bytes(self, family: Optional[str]) -> Optional[int]:
        """The most GPU memory a build of ``family`` allocated at once in a worker so far, or
        None before one is measured."""
        if family is None:
            return None
        with self._lock:
            return self._peak_bytes.get(family)

    def record_peak(self, family: str, nbytes: int) -> None:
        """Measure a build of ``family`` that allocated ``nbytes`` of GPU memory at once."""
        with self._lock:
            self._record_peak(family, nbytes)

    def _record_peak(self, family: str, nbytes: int) -> None:
        self._peak_bytes[family] = max(nbytes, self._peak_bytes.get(family, 0))

    # Builds.

    def build(
        self,
        builds: Sequence[Build],
        on_done: Optional[Callable[[int, BuildResult], None]] = None,
    ) -> List[BuildResult]:
        """Run ``builds`` on the workers, in parallel, and return their results, in order, once
        every one has ended (:meth:`submit`, then :meth:`BuildBatch.wait`)."""
        return self.submit(builds, on_done).wait()

    def submit(
        self,
        builds: Sequence[Build],
        on_done: Optional[Callable[[int, BuildResult], None]] = None,
    ) -> BuildBatch:
        """Queue ``builds`` on the workers and return at once; the batch waits for them.
        ``on_done(index, result)`` is called as each ends, on a thread of the pool. A build no
        worker takes is ``"refused"``, and its caller builds it: outside a pass, with workers off,
        with arguments that do not pickle or are too large to send, when the store cannot be
        made, and when no worker can start."""
        batch = BuildBatch(len(builds), on_done)
        self._submit_into(batch, list(enumerate(builds)))
        return batch

    def _submit_into(self, batch: BuildBatch, builds: Sequence[Tuple[int, Build]]) -> None:
        """Queue ``builds``, each settling the batch's result at its index."""
        pool = None
        if self.accepting() and builds:
            with self._lock:
                pool = self._ensure_pool()
        for index, build in builds:
            message = self._message(build) if pool is not None else None
            task = _Task(message, build.family, batch, index)
            if message is None:
                self._settle(task, BuildResult(REFUSED))
            else:
                pool.put(task)

    def _message(self, build: Build) -> Optional[bytes]:
        try:
            message = pickle.dumps(
                (build.function, build.args, build.claim), protocol=pickle.HIGHEST_PROTOCOL
            )
        except (pickle.PicklingError, TypeError, AttributeError) as e:
            logger.debug(f"[BuildPool] {build.function} is built here: {type(e).__name__}: {e}")
            return None
        if len(message) > _MAX_MESSAGE_BYTES:
            logger.debug(f"[BuildPool] {build.function} is built here: {len(message)} bytes")
            return None
        return message

    def build_kernels(
        self, kernels: Sequence[Tuple[Callable[..., Any], tuple]]
    ) -> List[BuildResult]:
        """Build the kernels of ``(function, args)`` pairs of ``process_built`` functions into the
        store on the workers, and return each one's result once all have ended (see
        :meth:`submit_kernels`)."""
        return self.submit_kernels(kernels).wait()

    def submit_kernels(self, kernels: Sequence[Tuple[Callable[..., Any], tuple]]) -> BuildBatch:
        """Queue the kernels of ``(function, args)`` pairs of ``process_built`` functions for the
        workers to build into the store, and return at once; the batch's results are each
        kernel's, in order: ``"present"`` for a kernel in the store already. With a shared store,
        the ranks split them. The calls ``function(*args)`` then load them."""
        names = [fn._compiler.name for fn, _ in kernels]

        def warn(index: int, result: BuildResult) -> None:
            if result.status == FAILED:
                error = result.error.splitlines()[0] if result.error else ""
                logger.warning_once(
                    f"[BuildPool] {names[index]}: a kernel a worker did not build is compiled "
                    f"here ({error})",
                    key=("build_pool", names[index], error),
                )

        batch = BuildBatch(len(kernels), warn)
        if not kernels:
            return batch
        if not self.accepting() or self._usable_store() is None:
            for index in range(len(kernels)):
                batch.settle(index, BuildResult(REFUSED))
            return batch
        builds = []
        for index, (fn, args) in enumerate(kernels):
            compiler = fn._compiler
            _, paths = self._paths(compiler.name, args)
            if os.path.exists(paths.obj):
                batch.settle(index, BuildResult(PRESENT))
                continue
            claim = Claim(paths.claim, paths.obj, paths.failed) if self._shared else None
            builds.append(
                (
                    index,
                    Build(
                        f"{__name__}:_build_kernel",
                        (compiler.where, args, paths),
                        compiler.name,
                        claim,
                    ),
                )
            )
        self._submit_into(batch, builds)
        return batch

    def load(self, name: str, args: tuple) -> Any:
        """The kernel of ``name(*args)`` in the store, or None. With a store the ranks share,
        waits for a kernel another rank claimed (at most ``TLLM_JIT_SHARE_WAIT_S`` seconds)."""
        if self._existing_store() is None:
            return None
        from tensorrt_llm import _jit_share

        _, paths = self._paths(name, args)
        if not os.path.exists(paths.obj) and not (
            self._shared
            and os.path.exists(paths.claim)
            and _jit_share._wait(paths.obj, paths.failed)
        ):
            return None
        try:
            return _jit_share._load_kernel(paths.obj, paths.symbol, paths.meta)
        except Exception as e:  # noqa: BLE001 - a kernel that does not load is compiled here
            logger.debug(f"[BuildPool] {name}{args}: loading it failed: {type(e).__name__}: {e}")
            return None

    def store(self, name: str, args: tuple, kernel: Any) -> None:
        """Export a kernel compiled here to the store when another process loads it from there:
        in a worker, for its rank; with a shared store, for the other ranks."""
        if not (self.in_worker or (self._shared and self._store is not None)):
            return
        from tensorrt_llm import _jit_share

        _, paths = self._paths(name, args)
        if os.path.exists(paths.obj):
            return
        try:
            _jit_share._export_kernel(kernel, paths.obj, paths.symbol, paths.meta)
        except Exception as e:  # noqa: BLE001 - the kernel works here; the others compile it
            logger.debug(f"[BuildPool] {name}{args}: exporting it failed: {e}")

    def stop(self) -> None:
        """Stop the workers and the threads, which frees the workers' GPU memory: the builds
        running end, and those still waiting are refused."""
        with self._lock:
            pool, self._pool = self._pool, None
            lane, self._lane = self._lane, None
            admission, self._admission = self._admission, None
        if admission is not None:
            admission.stop()
        if lane is not None:
            lane.stop()
        if pool is not None:
            pool.stop(_STOP_TIMEOUT_S)
        if pool is not None or lane is not None:
            threads = f"; {lane.report()}" if lane is not None else ""
            logger.info(f"[BuildPool] {self.report()}{threads}")

    def join_threads(self) -> Dict[str, int]:
        """Wait for the builds queued on the pass's threads; how many ended each way."""
        with self._lock:
            lane = self._lane
        return lane.join() if lane is not None else {}

    def counts(self) -> Dict[str, int]:
        """How many builds ended each way (:class:`BuildResult`'s statuses)."""
        with self._lock:
            return dict(self._counts)

    def report(self) -> str:
        counts = self.counts()
        with self._lock:
            seconds, started = self._build_s, self._workers_started
        return (
            f"{started} worker processes: {counts.get(BUILT, 0)} builds in {seconds:.1f}s, "
            f"{counts.get(PRESENT, 0)} built before, {counts.get(CLAIMED, 0)} claimed by other "
            f"ranks, {counts.get(FAILED, 0)} failed, {counts.get(REFUSED, 0)} left to the caller"
        )

    def _settle(self, task: _Task, result: BuildResult) -> None:
        with self._lock:
            if task.settled:
                return
            task.settled = True
            self._counts[result.status] += 1
            if result.status == BUILT:
                self._build_s += result.seconds
                if task.family is not None:
                    self._record_build(task.family, result.seconds)
                    self._record_peak(task.family, result.peak_bytes)
        task.batch.settle(task.index, result)

    def _ensure_pool(self) -> Optional["_Pool"]:
        """The pool of this pass, or None when the store cannot be made; called holding
        ``self._lock``."""
        if self._pool is None:
            store = self._usable_store()
            if store is None:
                return None
            from tensorrt_llm import _jit_share

            _jit_share.set_pool_store(store)
            env = {STORE_ENV: store, _SHARED_ENV: "1" if self._shared else "0"}
            self._pool = _Pool(self, self.max_workers, store, env, self._ensure_admission())
        return self._pool

    def _ensure_admission(self) -> "_Admission":
        """The GPU memory admission of this pass; called holding ``self._lock``."""
        if self._admission is None:
            self._admission = _Admission(self, _current_device())
        return self._admission

    def _worker_started(self) -> None:
        with self._lock:
            self._workers_started += 1

    # The store.

    def _existing_store(self) -> Optional[str]:
        """The store, when this process has one: a lookup creates none, except of the store
        the ranks of a launch share."""
        from tensorrt_llm import _jit_share

        if self._store_ready or self.in_worker:
            return self._store
        if _jit_share.enabled():
            return self._usable_store()
        return None

    def _usable_store(self) -> Optional[str]:
        """The store, made if needed; None when it cannot be made, which turns the workers off:
        every kernel then builds in this process."""
        try:
            return self._store_dir()
        except OSError as e:
            self.enabled = False
            logger.warning_once(
                f"[BuildPool] no worker process builds kernels: the store cannot be made: "
                f"{type(e).__name__}: {e}",
                key=("build_pool", "store"),
            )
            return None

    def _store_dir(self) -> str:
        with self._store_lock:
            if not self._store_ready:
                from tensorrt_llm import _jit_share

                store, shared = self._store, self._shared
                if store is None:
                    if _jit_share.enabled():
                        store = os.path.join(os.environ[_jit_share.SHARE_DIR_ENV], "cute")
                        shared = True
                    else:
                        store = tempfile.mkdtemp(prefix="tllm-build-pool-")
                        atexit.register(shutil.rmtree, store, True)
                os.makedirs(store, exist_ok=True)
                self._store, self._shared, self._store_ready = store, shared, True
            return self._store

    def _paths(self, name: str, args: tuple) -> Tuple[str, _Paths]:
        from tensorrt_llm import _jit_share

        store = self._store_dir()
        text = f"{_jit_share._cute_identity()}|{name}|{args!r}"
        symbol = "p" + hashlib.sha256(text.encode()).hexdigest()[:32]
        base = os.path.join(store, symbol)
        return symbol, _Paths(
            symbol, base + ".o", base + ".json", base + ".claim", base + ".failed"
        )


class _Pool:
    """The workers of one pass, which serve one queue of builds and start as builds come in."""

    def __init__(
        self,
        owner: BuildPool,
        max_workers: int,
        cwd: str,
        extra_env: Dict[str, str],
        admission: _Admission,
    ):
        self.queue: "queue.Queue[Optional[_Task]]" = queue.Queue()
        self._owner = owner
        self._max = max_workers
        self._cwd = cwd
        self._env = {**_worker_env(), **extra_env}
        self._lock = threading.Lock()
        self._idle = 0
        self._live = 0
        self._stopping = False
        self._processes: List[subprocess.Popen] = []
        self._threads: List[threading.Thread] = []
        # Which builds start, by the GPU memory they take beside those running.
        self._admission = admission

    def put(self, task: _Task) -> None:
        """Queue ``task``, starting a worker when none is idle and the cap and the GPU's free
        memory allow; with no worker able to take it, refuse it at once."""
        with self._lock:
            if not self._stopping:
                if self._idle == 0 and len(self._processes) < self._max and _gpu_has_room():
                    try:
                        self._start_worker()
                    except OSError as e:
                        logger.warning(f"[BuildPool] a worker process did not start: {e}")
                if self._live:
                    self._idle = max(0, self._idle - 1)
                    self.queue.put(task)
                    return
        self._owner._settle(task, BuildResult(REFUSED, 0.0, "no worker process is running"))

    def stop(self, timeout: float) -> None:
        with self._lock:
            self._stopping = True
            for _ in self._threads:
                self.queue.put(None)
        self._admission.stop()
        deadline = time.monotonic() + timeout
        for thread in self._threads:
            thread.join(max(0.0, deadline - time.monotonic()))
        for process in self._processes:
            try:
                process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    def _start_worker(self) -> None:
        """Start one worker and the thread that serves it; called holding ``self._lock``."""
        index = len(self._processes)
        task_read, task_write = os.pipe()
        result_read, result_write = os.pipe()
        stderr = tempfile.TemporaryFile()
        try:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    "from tensorrt_llm._torch.build_pool import _worker_main; "
                    f"_worker_main({task_read}, {result_write})",
                ],
                cwd=self._cwd,
                env=self._env,
                pass_fds=(task_read, result_write),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=stderr,
            )
        except OSError:
            for fd in (task_read, task_write, result_read, result_write):
                os.close(fd)
            stderr.close()
            raise
        os.close(task_read)
        os.close(result_write)
        thread = threading.Thread(
            target=self._serve,
            args=(task_write, result_read, stderr),
            name=f"tllm-build-pool-{index}",
            daemon=True,
        )
        self._processes.append(process)
        self._threads.append(thread)
        self._live += 1
        self._idle += 1
        self._owner._worker_started()
        thread.start()

    def _serve(self, task_write: int, result_read: int, stderr: IO[bytes]) -> None:
        with (
            stderr,
            os.fdopen(task_write, "wb") as to_worker,
            os.fdopen(result_read, "rb") as from_worker,
        ):
            while True:
                task = self.queue.get()
                if task is None:
                    try:
                        _write(to_worker, pickle.dumps(None))
                    except OSError:
                        pass
                    return
                if not self._admission.admit(id(task), task.family):
                    self._owner._settle(
                        task, BuildResult(REFUSED, 0.0, "the GPU has no room for the build")
                    )
                    with self._lock:
                        self._idle += 1
                    continue
                try:
                    _write(to_worker, task.message)
                    result = BuildResult(*_recv(from_worker))
                except (OSError, EOFError, pickle.UnpicklingError) as e:
                    self._owner._settle(
                        task,
                        BuildResult(
                            FAILED, 0.0, f"the worker process ended ({e!r}): {_tail(stderr)}"
                        ),
                    )
                    self._admission.release(id(task))
                    self._ended()
                    return
                # The result records the memory the build took, before the builds waiting for
                # room look again.
                self._owner._settle(task, result)
                self._admission.release(id(task))
                if result.status == FAILED and result.error.endswith(_WORKER_ENDS):
                    self._ended()
                    return
                with self._lock:
                    self._idle += 1

    def _ended(self) -> None:
        """A worker ended: with none left, refuse the builds still queued."""
        stranded = []
        with self._lock:
            self._live -= 1
            if self._live:
                return
            while True:
                try:
                    task = self.queue.get_nowait()
                except queue.Empty:
                    break
                if task is not None:
                    stranded.append(task)
        for task in stranded:
            self._owner._settle(task, BuildResult(REFUSED, 0.0, "no worker process is left"))


class _LaneBuild(NamedTuple):
    function: Callable[[], Any]
    family: Optional[str]
    need_bytes: int


class _ThreadLane:
    """A pass's in-process build threads (:attr:`BuildPool.threads`), for builds whose compilers
    run outside the GIL (nvcc, ninja, Triton) and that do not repay a worker process. A thread
    starts with the builds, up to ``max_threads``; each runs its builds on a CUDA stream of its own
    and synchronizes that stream alone, so that they go on beside the rank's other work. A build
    starts only while the GPU has room for the ``need_bytes`` it says it allocates (the pass's
    :class:`_Admission`); the threads never empty the allocator's cache, which synchronizes the
    device: the pass's end does. A build that raises fails alone, and its caller builds it."""

    def __init__(self, owner: BuildPool, admission: _Admission, max_threads: int):
        self._owner = owner
        self._admission = admission
        self._max = max(1, max_threads)
        self._queue: "queue.Queue[Optional[_LaneBuild]]" = queue.Queue()
        self._cond = threading.Condition()
        self._threads: List[threading.Thread] = []
        self._pending = 0
        self._stopping = False
        self._counts: collections.Counter = collections.Counter()
        self._seconds = 0.0

    def submit(
        self, function: Callable[[], Any], family: Optional[str] = None, need_bytes: int = 0
    ) -> bool:
        """Queue ``function()``, which allocates at most ``need_bytes`` of GPU memory at once;
        False when the threads have stopped, and the caller builds it."""
        with self._cond:
            if self._stopping:
                self._counts[REFUSED] += 1
                return False
            self._pending += 1
            if len(self._threads) < min(self._max, self._pending):
                thread = threading.Thread(
                    target=self._serve,
                    name=f"tllm-build-thread-{len(self._threads)}",
                    daemon=True,
                )
                self._threads.append(thread)
                thread.start()
        self._queue.put(_LaneBuild(function, family, need_bytes))
        return True

    def join(self) -> Dict[str, int]:
        """Wait until every build queued has ended; how many ended each way."""
        with self._cond:
            while self._pending:
                self._cond.wait()
            return dict(self._counts)

    def counts(self) -> Dict[str, int]:
        with self._cond:
            return dict(self._counts)

    def report(self) -> str:
        with self._cond:
            counts, seconds, threads = dict(self._counts), self._seconds, len(self._threads)
        return (
            f"{threads} build threads: {counts.get(BUILT, 0)} builds in {seconds:.1f}s, "
            f"{counts.get(FAILED, 0)} failed, {counts.get(REFUSED, 0)} left to the caller"
        )

    def stop(self) -> None:
        """Refuse the builds still queued, and wait for those running to end."""
        stranded = 0
        with self._cond:
            self._stopping = True
            while True:
                try:
                    item = self._queue.get_nowait()
                except queue.Empty:
                    break
                if item is not None:
                    stranded += 1
            self._counts[REFUSED] += stranded
            self._pending -= stranded
            for _ in self._threads:
                self._queue.put(None)
            self._cond.notify_all()
        for thread in self._threads:
            thread.join()

    def _serve(self) -> None:
        stream = None
        if self._admission.device is not None:
            import torch

            torch.cuda.set_device(self._admission.device)
            stream = torch.cuda.Stream()
        while True:
            item = self._queue.get()
            if item is None:
                return
            key = id(item)
            status, seconds = REFUSED, 0.0
            if self._admission.admit(key, item.family, item.need_bytes, in_process=True):
                t0 = time.perf_counter()
                try:
                    if stream is None:
                        item.function()
                    else:
                        import torch

                        with torch.cuda.stream(stream):
                            item.function()
                        stream.synchronize()
                    status = BUILT
                except Exception as e:  # noqa: BLE001 - a build that fails here fails alone
                    status = FAILED
                    logger.debug(
                        f"[BuildPool] {item.family}: a build on a thread failed: "
                        f"{type(e).__name__}: {e}"
                    )
                finally:
                    self._admission.release(key)
                seconds = time.perf_counter() - t0
                if status == BUILT and item.family is not None:
                    self._owner.record_build(item.family, seconds)
            with self._cond:
                self._counts[status] += 1
                self._seconds += seconds
                self._pending -= 1
                self._cond.notify_all()


def _tail(stream: IO[bytes]) -> str:
    try:
        stream.seek(0)
        return stream.read().decode(errors="replace")[-_STDERR_TAIL:].strip()
    except (OSError, ValueError):
        return ""


def _worker_env() -> Dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(_LAUNCHER_ENV_PREFIXES)}
    # The worker runs in the store, which holds no modules; its directory stays off sys.path.
    env["PYTHONSAFEPATH"] = "1"
    import torch

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        visible = [d.strip() for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")]
        visible = [d for d in visible if d]
        env["CUDA_VISIBLE_DEVICES"] = visible[device] if device < len(visible) else str(device)
    return env


def _write(stream: IO[bytes], data: bytes) -> None:
    stream.write(struct.pack("<Q", len(data)))
    stream.write(data)
    stream.flush()


def _read(stream: IO[bytes]) -> bytes:
    header = stream.read(8)
    if len(header) < 8:
        raise EOFError("the pipe closed")
    (size,) = struct.unpack("<Q", header)
    data = stream.read(size)
    if len(data) < size:
        raise EOFError("the pipe closed")
    return data


def _recv(stream: IO[bytes]) -> Any:
    return pickle.loads(_read(stream))  # nosec B301 - from this process's own worker


def _worker_main(task_read: int, result_write: int) -> None:
    """A worker: make each build asked for, until asked to stop or the rank ends."""
    # A crash leaves its stack in the stderr the rank quotes.
    faulthandler.enable()
    with os.fdopen(task_read, "rb") as tasks, os.fdopen(result_write, "wb") as results:
        while True:
            try:
                data = _read(tasks)
            except EOFError:
                return
            try:
                message = pickle.loads(data)  # nosec B301 - from this worker's parent
            except Exception as e:  # noqa: BLE001 - a build this process cannot load fails alone
                result = (FAILED, 0.0, f"the worker could not load it: {type(e).__name__}: {e}")
            else:
                if message is None:
                    return
                result = _run(*message)
            ends = result[0] == FAILED and _cuda_context_broken()
            if ends:
                result = (result[0], result[1], result[2] + _WORKER_ENDS, *result[3:])
            try:
                data = pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
            except (pickle.PicklingError, TypeError, AttributeError) as e:
                error = f"the build's value does not pickle: {type(e).__name__}: {e}"
                data = pickle.dumps((FAILED, result[1], error), protocol=pickle.HIGHEST_PROTOCOL)
            try:
                _write(results, data)
            except BrokenPipeError:
                return
            if ends:
                return


def _cuda_context_broken() -> bool:
    """Whether a build left this worker's CUDA context with a sticky error (a kernel fault),
    after which every CUDA call fails: the worker then ends, so that the builds queued behind
    it go to the other workers."""
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_initialized():
        return False
    try:
        torch.cuda.synchronize()
    except Exception:  # noqa: BLE001 - any error a synchronize raises is the context's
        return True
    return False


def _resolve(function: str) -> Any:
    module, qualname = function.split(":", 1)
    target: Any = importlib.import_module(module)
    for part in qualname.split("."):
        target = getattr(target, part)
    return target


def _touch(path: str) -> None:
    try:
        os.close(os.open(path, os.O_CREAT | os.O_WRONLY, 0o644))
    except OSError:
        pass  # The other ranks then wait for the build until their wait runs out.


def _run(function: str, args: tuple, claim: Optional[Claim]) -> tuple:
    """In a worker process: make one build; with a ``claim``, only when no other rank's worker
    claimed it first."""
    from tensorrt_llm import _jit_share

    if claim is not None:
        if os.path.exists(claim.done):
            return (PRESENT, 0.0, "")
        try:
            claimed = _jit_share._claim(claim.claim)
        except OSError:
            claimed = True  # The directory takes no claims: build it here.
        if not claimed:
            return (PRESENT if os.path.exists(claim.done) else CLAIMED, 0.0, "")
    t0 = time.perf_counter()
    try:
        value = _resolve(function)(*args)
    except Exception as e:  # noqa: BLE001 - reported to the rank, which builds it itself
        if claim is not None:
            _jit_share._mark_failed(claim.failed, f"{type(e).__name__}: {e}")
        detail = traceback.format_exc()[-_STDERR_TAIL:]
        seconds = time.perf_counter() - t0
        return (FAILED, seconds, f"{type(e).__name__}: {e}\n{detail}", _release_gpu_memory())
    seconds = time.perf_counter() - t0
    peak = _release_gpu_memory()
    if claim is not None and not os.path.exists(claim.done):
        _touch(claim.done)
    return (BUILT, seconds, "", peak, value)


def _release_gpu_memory() -> int:
    """In a worker process: free the GPU memory the build that just ended allocated, and return
    the most it allocated at once (PyTorch's allocator; 0 for a build that did not use CUDA)."""
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_initialized():
        return 0
    try:
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    except Exception:  # noqa: BLE001 - a lost CUDA context ends the worker after its report
        return 0
    return int(peak)


def _build_kernel(function: str, args: tuple, paths: _Paths) -> None:
    """In a worker process: compile the kernel of the ``process_built`` function ``function``
    for ``args`` into the store."""
    from tensorrt_llm import _jit_share

    kernel = _resolve(function).raw(*args)
    _jit_share._export_kernel(kernel, paths.obj, paths.symbol, paths.meta)
