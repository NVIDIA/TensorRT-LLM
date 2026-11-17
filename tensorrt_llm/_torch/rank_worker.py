# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``RankWorker``: worker processes that build this rank's kernels in parallel.

KernelBuilder's build mode and the autotuner's tune mode hand their kernel builds to these
processes. CUTLASS DSL keeps one compile session per process, so CuTe DSL kernels compile in
parallel only in separate processes; the other compilers run there too, so every op builds the
same way.

* A build is a module-level function called with picklable positional arguments
  (:meth:`RankWorker.run`). Its kernel reaches this process the way its compiler caches it: in
  the compiler's disk cache (DeepGEMM, FlashInfer, Triton, fmha_sm100, PyTorch extensions), or,
  for a CuTe DSL kernel, as an object file in this rank's store, which ``process_built``
  functions and the runners' shared ``kernel_cache`` dicts (``tensorrt_llm._jit_share``) load.
* :func:`process_built` memoizes a CuTe DSL compile function: ``fn.submit(*args)`` has a worker
  build that kernel into the store, and ``fn(*args)`` loads it from there.
* The workers live inside a pass (:meth:`RankWorker.pass_scope`, which KernelBuilder's build
  pass opens, in the memory-profiling warmup only): they start with the pass's builds, while the
  GPU keeps ``_MIN_FREE_GPU_BYTES`` free, and stop when the outermost pass ends, after those
  builds. Each holds a CUDA context while it runs, which must be gone before the KV cache
  estimation measures free memory; a warmup with the final KV cache starts none. Outside a pass
  no worker takes a build, and the caller builds in this process.
* When the ranks share JIT builds (``TLLM_JIT_SHARE_DIR``, or by default the launch's directory
  on each node; see ``tensorrt_llm._jit_share``), the store is ``<that directory>/cute`` and the
  ranks claim each kernel there: the rank that claims a kernel first builds it, and the others
  load it. Otherwise every rank builds each kernel into a store of its own, removed when the
  process exits.
* A worker takes a build only when builds of its kind take long enough to repay a worker
  (:meth:`RankWorker.worth_a_worker`): ``TLLM_KERNEL_BUILDER_MIN_BUILD_S`` seconds (default 5),
  as measured so far in this process and its workers, or as the caller expects until one is
  measured. Shorter builds are made where they are first needed.
* The workers start with ``subprocess`` in an environment of their own: without the launcher's
  MPI, PMIx and Slurm variables, since importing TensorRT-LLM initializes MPI, and with only this
  rank's GPU visible.

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
import pickle
import queue
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from concurrent.futures import Future
from typing import IO, Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Tuple

from tensorrt_llm.logger import logger

__all__ = ["BuildStrategy", "RankWorker", "process_built"]

ENABLE_ENV = "TLLM_KERNEL_BUILDER"
_WORKERS_ENV = "TLLM_KERNEL_BUILDER_WORKERS"
_MIN_BUILD_ENV = "TLLM_KERNEL_BUILDER_MIN_BUILD_S"
# A build goes to a worker only when builds of its kind take at least this long: a worker
# spends seconds starting and loading its compiler, which a short build does not repay.
_MIN_BUILD_S = 5.0
# Each worker holds a CUDA context (about a GiB) on the rank's GPU while the pass lasts.
_MAX_DEFAULT_WORKERS = 8
# Set in a worker: the store its CuTe DSL kernels go to, and whether the ranks share it.
STORE_ENV = "TLLM_RANK_WORKER_STORE"
_SHARED_ENV = "TLLM_RANK_WORKER_SHARED"
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

# A build's outcome: the worker built the kernel, it was in the store already, another rank
# claimed it (and builds it), the build failed, or a run ended.
_BUILT, _PRESENT, _CLAIMED, _FAILED, _DONE = "built", "present", "claimed", "failed", "done"
# A build no worker had started when this process needed its kernel, which it then made itself.
_TAKEN = "taken"

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


def process_built(
    name: str, build_seconds: Optional[float] = None
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Memoize a CuTe DSL kernel compile function whose kernels worker processes can build.

    Use it in place of ``functools.cache`` on a function that compiles one kernel from hashable,
    picklable positional arguments and returns the compiled kernel (what ``cute.compile``
    returns). The function keeps ``cache_info()`` and ``cache_clear()`` and gains
    ``submit(*args)``: have a worker build the kernel of these arguments into the store
    (:meth:`RankWorker.submit`); True when a worker took it. A call loads the kernel from the
    store when a worker built it, and compiles it here otherwise. In a worker, a call compiles the
    kernel and exports it to the store, so that this rank loads it. ``raw`` is the undecorated
    function.

    Args:
        name: Names the function's kernels in the store; unique per compile function.
        build_seconds: How long one compile takes, until compiles of it are measured; ``submit``
            leaves shorter ones to the caller (:meth:`RankWorker.worth_a_worker`).
    """

    def decorate(raw: Callable[..., Any]) -> Callable[..., Any]:
        memo: Dict[tuple, Any] = {}
        lock = threading.Lock()
        counts = {"hits": 0, "misses": 0}
        where = f"{raw.__module__}:{raw.__qualname__}"

        @functools.wraps(raw)
        def compiled(*args):
            kernel = memo.get(args)
            if kernel is None:
                with lock:
                    kernel = memo.get(args)
                    if kernel is None:
                        counts["misses"] += 1
                        worker = RankWorker.get()
                        kernel = worker.take(name, args)
                        if kernel is None:
                            t0 = time.perf_counter()
                            kernel = raw(*args)
                            worker.record_build(name, time.perf_counter() - t0)
                            worker.store(name, args, kernel)
                        memo[args] = kernel
                        return kernel
            counts["hits"] += 1
            return kernel

        def submit(*args) -> bool:
            if args in memo:
                return False
            return RankWorker.get().submit(name, where, args, build_seconds)

        def cache_info() -> _CacheInfo:
            return _CacheInfo(counts["hits"], counts["misses"], None, len(memo))

        def cache_clear() -> None:
            with lock:
                memo.clear()
                counts["hits"] = counts["misses"] = 0

        compiled.raw = raw
        compiled.submit = submit
        compiled.cache_info = cache_info
        compiled.cache_clear = cache_clear
        return compiled

    return decorate


class _Paths(NamedTuple):
    symbol: str
    obj: str
    meta: str
    claim: str
    failed: str


@dataclasses.dataclass(eq=False)
class _Task:
    key: str
    # The pickled ("build", (function, args, paths, shared)) or ("run", (function, args)).
    message: bytes
    future: Future
    is_build: bool
    # Set holding the rank's lock: a worker took the task, the rank took it back, it was settled.
    started: bool = False
    taken: bool = False
    settled: bool = False
    # What kind of build it is, for the rank's measure of how long such builds take.
    family: Optional[str] = None


def _gpu_has_room() -> bool:
    """Whether the rank's GPU has room for one more worker (``_MIN_FREE_GPU_BYTES`` free)."""
    import torch

    if not torch.cuda.is_available() or not torch.cuda.is_initialized():
        return True
    free, _ = torch.cuda.mem_get_info()
    return free >= _MIN_FREE_GPU_BYTES


def _default_workers() -> int:
    from .autotuner import _default_parallel_jit_warmup_threads

    return min(_MAX_DEFAULT_WORKERS, _default_parallel_jit_warmup_threads())


class RankWorker:
    """This rank's worker processes, and the store they build CuTe DSL kernels into.

    Inside a pass (:meth:`pass_scope`), builds are asked for with :meth:`run` (any build) or
    :meth:`submit` (through ``process_built``'s ``submit``); :meth:`take` loads a kernel a worker
    built; :meth:`wait` and :meth:`join` wait for builds. The workers start as builds come in, up
    to ``TLLM_KERNEL_BUILDER_WORKERS`` and while the GPU has room, and stop when the outermost
    pass ends.
    """

    _instance: Optional["RankWorker"] = None
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
        self._pool: Optional[_Pool] = None
        self._futures: Dict[str, Future] = {}
        # The builds not settled yet, by key.
        self._tasks: Dict[str, _Task] = {}
        self._counts: collections.Counter = collections.Counter()
        self._build_s = 0.0
        self._workers_started = 0
        self.min_build_s = float(os.environ.get(_MIN_BUILD_ENV, _MIN_BUILD_S))
        # family -> [builds measured, their seconds]
        self._build_times: Dict[str, List[float]] = {}

    @classmethod
    def get(cls) -> "RankWorker":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # Passes.

    @contextlib.contextmanager
    def pass_scope(self) -> Iterator[None]:
        """A build pass: workers take the builds asked for inside it, and stop when the
        outermost pass ends, after its builds. Once a worker takes a build, the rank's shared
        kernel caches (``tensorrt_llm._jit_share``) keep CuTe DSL kernels in the store the
        workers build into, until the pass ends."""
        from tensorrt_llm import _jit_share

        with self._lock:
            self._depth += 1
        try:
            yield
        finally:
            with self._lock:
                self._depth -= 1
                last = self._depth == 0
            if last:
                self.join()
                self.stop()
                if not self.in_worker:
                    _jit_share.set_rank_store(None)

    def accepting(self) -> bool:
        """Whether a worker takes a build asked for now."""
        return self.enabled and not self.in_worker and self._depth > 0

    def takes(self, args: tuple) -> bool:
        """Whether a worker takes a build asked for now with ``args`` (see :meth:`run`)."""
        if not self.accepting():
            return False
        try:
            message = pickle.dumps(args, protocol=pickle.HIGHEST_PROTOCOL)
        except (pickle.PicklingError, TypeError, AttributeError):
            return False
        return len(message) <= _MAX_MESSAGE_BYTES

    def worth_a_worker(self, family: str, expected_s: Optional[float] = None) -> bool:
        """Whether a worker should take a build of ``family`` now: while workers take builds,
        when builds of the family take at least ``TLLM_KERNEL_BUILDER_MIN_BUILD_S`` seconds
        (default 5; 0 sends every build to the workers). How long they take is their mean so far,
        here or in a worker, or ``expected_s`` until one is measured; a family that has neither
        builds in this process."""
        if not self.accepting():
            return False
        if self.min_build_s <= 0:
            return True
        estimate = self.build_seconds(family)
        if estimate is None:
            estimate = expected_s
        return estimate is not None and estimate >= self.min_build_s

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

    # Builds.

    def run(
        self, key: str, function: str, args: tuple, family: Optional[str] = None
    ) -> Optional[Future]:
        """Have a worker call ``function`` (``"module:qualname"`` of a module-level function)
        with ``args``; the build's kernel reaches this process through its compiler's cache or
        the store. Returns a future of ``(status, seconds, error)`` (status ``"done"`` or
        ``"failed"``), the earlier one for a ``key`` asked for before, or None when no worker
        takes it (outside a pass, with workers off, ``args`` that do not pickle, or a store that
        cannot be made): the caller then builds in this process."""
        if not self.accepting():
            return None
        with self._lock:
            future = self._futures.get(key)
            if future is not None:
                return future
            try:
                message = pickle.dumps(("run", (function, args)), protocol=pickle.HIGHEST_PROTOCOL)
            except (pickle.PicklingError, TypeError, AttributeError) as e:
                logger.debug(f"[RankWorker] {key} is built here: {type(e).__name__}: {e}")
                return None
            if len(message) > _MAX_MESSAGE_BYTES:
                logger.debug(f"[RankWorker] {key} is built here: {len(message)} bytes to send")
                return None
            pool = self._ensure_pool()
            if pool is None:
                return None
            future = Future()
            self._futures[key] = future
            task = self._tasks[key] = _Task(key, message, future, True, family=family)
        pool.put(task)
        return future

    def submit(
        self, name: str, function: str, args: tuple, build_seconds: Optional[float] = None
    ) -> bool:
        """Have a worker build the kernel of ``function`` (``"module:qualname"`` of a
        ``process_built`` function) for ``args`` into the store.

        Returns False when the kernel was asked for before or is in the store already, when its
        builds are too short for a worker (:meth:`worth_a_worker`; ``build_seconds`` is how long
        one takes until one is measured), and when no worker takes it (see :meth:`run`).
        """
        if not self.worth_a_worker(name, build_seconds) or self._usable_store() is None:
            return False
        key, paths = self._paths(name, args)
        with self._lock:
            if key in self._futures:
                return False
            try:
                message = pickle.dumps(
                    ("build", (function, args, paths, self._shared)),
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            except (pickle.PicklingError, TypeError, AttributeError) as e:
                logger.debug(f"[RankWorker] {name}{args} is built here: {type(e).__name__}: {e}")
                return False
            pool = self._ensure_pool()
            if pool is None:
                return False
            future = Future()
            self._futures[key] = future
            if os.path.exists(paths.obj):
                future.set_result((_PRESENT, 0.0, ""))
                self._counts[_PRESENT] += 1
                return False
            task = self._tasks[key] = _Task(key, message, future, True, family=name)
        return pool.put(task)

    def take(self, name: str, args: tuple) -> Any:
        """The kernel of ``name(*args)`` in the store, or None. Waits for a build of it a worker
        is making, and for one another rank claimed (at most ``TLLM_JIT_SHARE_WAIT_S`` seconds);
        takes back one no worker has started, which the caller then makes."""
        store = self._existing_store()
        if store is None:
            return None
        from tensorrt_llm import _jit_share

        key, paths = self._paths(name, args)
        future = self._futures.get(key)
        error = "not in the store"
        if future is not None:
            if not future.done() and self._take_back(key):
                return None
            status, _, error = future.result()
            if status == _TAKEN:
                return None
            if status == _CLAIMED and not _jit_share._wait(paths.obj, paths.failed):
                status, error = _FAILED, "the rank that claimed it did not build it"
            if status == _FAILED:
                logger.warning_once(
                    f"[RankWorker] {name}: a kernel a worker did not build is compiled here "
                    f"({error})",
                    key=("rank_worker", name, error),
                )
                return None
        elif not os.path.exists(paths.obj) and not (
            self._shared
            and os.path.exists(paths.claim)
            and _jit_share._wait(paths.obj, paths.failed)
        ):
            return None
        try:
            return _jit_share._load_kernel(paths.obj, paths.symbol, paths.meta)
        except Exception as e:  # noqa: BLE001 - a kernel that does not load is compiled here
            logger.debug(f"[RankWorker] {name}{args}: loading it failed: {type(e).__name__}: {e}")
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
            logger.debug(f"[RankWorker] {name}{args}: exporting it failed: {e}")

    def wait(self, key: str) -> float:
        """Wait for the build of ``key`` (:meth:`run`) when a worker is making it; take it back
        when none has started it, for the caller to make. Returns the seconds waited."""
        future = self._futures.get(key)
        if future is None or (not future.done() and self._take_back(key)):
            return 0.0
        t0 = time.perf_counter()
        future.result()
        return time.perf_counter() - t0

    def state(self, key: str) -> Optional[str]:
        """``"queued"`` while the build of ``key`` runs, its status once it ended, None when
        none was asked for."""
        future = self._futures.get(key)
        if future is None:
            return None
        return future.result()[0] if future.done() else "queued"

    def join(self, keys: Optional[List[str]] = None) -> float:
        """Wait for the builds of ``keys``, or for every build asked for; returns the seconds
        waited."""
        t0 = time.perf_counter()
        with self._lock:
            futures = (
                list(self._futures.values())
                if keys is None
                else [self._futures[k] for k in keys if k in self._futures]
            )
        for future in futures:
            future.result()
        return time.perf_counter() - t0

    def stop(self) -> None:
        """Stop the workers after the builds already queued, which frees their GPU memory."""
        with self._lock:
            pool, self._pool = self._pool, None
        if pool is not None:
            pool.stop(_STOP_TIMEOUT_S)
            logger.info(f"[RankWorker] {self.report()}")

    def counts(self) -> Dict[str, int]:
        """How many builds ended each way: ``built``, ``present`` (in the store already),
        ``claimed`` (by another rank, which builds it), ``failed``, ``done`` (runs)."""
        with self._lock:
            return dict(self._counts)

    def report(self) -> str:
        counts = self.counts()
        with self._lock:
            seconds, started = self._build_s, self._workers_started
        return (
            f"{started} worker processes: {counts.get(_DONE, 0) + counts.get(_BUILT, 0)} builds "
            f"in {seconds:.1f}s, {counts.get(_PRESENT, 0)} in the store already, "
            f"{counts.get(_CLAIMED, 0)} claimed by other ranks, {counts.get(_TAKEN, 0)} taken back, "
            f"{counts.get(_FAILED, 0)} failed"
        )

    def _take_back(self, key: str) -> bool:
        """Take back the build of ``key`` when no worker has started it: this process makes the
        kernel now instead of waiting behind the builds queued before it."""
        with self._lock:
            task = self._tasks.get(key)
            if task is None or task.started or task.taken:
                return False
            task.taken = True
        self._settle(task, (_TAKEN, 0.0, ""))
        return True

    def _settle(self, task: _Task, result: tuple) -> None:
        with self._lock:
            if task.settled:
                return
            task.settled = True
            self._tasks.pop(task.key, None)
            if task.is_build:
                self._counts[result[0]] += 1
                if result[0] in (_BUILT, _DONE):
                    self._build_s += result[1]
                    if task.family is not None:
                        self._record_build(task.family, result[1])
        task.future.set_result(result)

    def _ensure_pool(self) -> Optional["_Pool"]:
        """The pool of this pass, or None when the store cannot be made; called holding
        ``self._lock``."""
        if self._pool is None:
            store = self._usable_store()
            if store is None:
                return None
            from tensorrt_llm import _jit_share

            _jit_share.set_rank_store(store)
            env = {STORE_ENV: store, _SHARED_ENV: "1" if self._shared else "0"}
            self._pool = _Pool(self, self.max_workers, store, env)
        return self._pool

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
                f"[RankWorker] no worker process builds kernels: the store cannot be made: "
                f"{type(e).__name__}: {e}",
                key=("rank_worker", "store"),
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
                        store = tempfile.mkdtemp(prefix="tllm-rank-worker-")
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

    def __init__(self, owner: RankWorker, max_workers: int, cwd: str, extra_env: Dict[str, str]):
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

    def put(self, task: _Task) -> bool:
        """Queue ``task``, starting a worker when none is idle and the cap and the GPU's free
        memory allow; with no worker able to take it, fail it at once. Returns whether it was
        queued."""
        with self._lock:
            if not self._stopping:
                if self._idle == 0 and len(self._processes) < self._max and _gpu_has_room():
                    try:
                        self._start_worker()
                    except OSError as e:
                        logger.warning(f"[RankWorker] a worker process did not start: {e}")
                if self._live:
                    self._idle = max(0, self._idle - 1)
                    self.queue.put(task)
                    return True
        self._owner._settle(task, (_FAILED, 0.0, "no worker process is running"))
        return False

    def stop(self, timeout: float) -> None:
        with self._lock:
            self._stopping = True
            for _ in self._threads:
                self.queue.put(None)
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
                    "from tensorrt_llm._torch.rank_worker import _worker_main; "
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
            name=f"tllm-rank-worker-{index}",
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
                with self._owner._lock:
                    skip = task.taken
                    task.started = not skip
                if skip:
                    with self._lock:
                        self._idle += 1
                    continue
                try:
                    _write(to_worker, task.message)
                    result = _recv(from_worker)
                except (OSError, EOFError, pickle.UnpicklingError) as e:
                    self._owner._settle(
                        task,
                        (_FAILED, 0.0, f"the worker process ended ({e!r}): {_tail(stderr)}"),
                    )
                    self._ended()
                    return
                self._owner._settle(task, result)
                if result[0] == _FAILED and result[2].endswith(_WORKER_ENDS):
                    self._ended()
                    return
                with self._lock:
                    self._idle += 1

    def _ended(self) -> None:
        """A worker ended: with none left, fail the builds still queued."""
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
            self._owner._settle(task, (_FAILED, 0.0, "no worker process is left"))


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
    return pickle.loads(_read(stream))


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
                message = pickle.loads(data)
            except Exception as e:  # noqa: BLE001 - a build this process cannot load fails alone
                result = (_FAILED, 0.0, f"the worker could not load it: {type(e).__name__}: {e}")
            else:
                if message is None:
                    return
                kind, payload = message
                result = _build(*payload) if kind == "build" else _run(*payload)
            ends = result[0] == _FAILED and _cuda_context_broken()
            if ends:
                result = (result[0], result[1], result[2] + _WORKER_ENDS)
            try:
                _write(results, pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL))
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


def _run(function: str, args: tuple) -> tuple:
    t0 = time.perf_counter()
    try:
        _resolve(function)(*args)
    except Exception as e:  # noqa: BLE001 - reported to the rank, which builds it itself
        detail = traceback.format_exc()[-_STDERR_TAIL:]
        return (_FAILED, time.perf_counter() - t0, f"{type(e).__name__}: {e}\n{detail}")
    return (_DONE, time.perf_counter() - t0, "")


def _build(function: str, args: tuple, paths: _Paths, shared: bool) -> tuple:
    from tensorrt_llm import _jit_share

    t0 = time.perf_counter()
    if shared:
        try:
            claimed = _jit_share._claim(paths.claim)
        except OSError:
            claimed = True  # The directory takes no claims: build it here.
        if not claimed:
            return (_PRESENT if os.path.exists(paths.obj) else _CLAIMED, 0.0, "")
    try:
        kernel = _resolve(function).raw(*args)
        _jit_share._export_kernel(kernel, paths.obj, paths.symbol, paths.meta)
    except Exception as e:  # noqa: BLE001 - reported to the rank, which compiles it itself
        if shared:
            _jit_share._mark_failed(paths.failed, f"{type(e).__name__}: {e}")
        return (_FAILED, time.perf_counter() - t0, f"{type(e).__name__}: {e}")
    return (_BUILT, time.perf_counter() - t0, "")
