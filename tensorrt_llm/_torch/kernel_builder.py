# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``KernelBuilder``: kernels built on worker processes before the calls that need them.

For ops whose kernels compile on first use outside the autotuner. Three modes:

* Inference (outside a build pass): an op compiles a kernel it lacks on first use.
* Build (inside :meth:`KernelBuilder.build_pass`, which the memory-profiling warmup opens): for
  an op that sets ``BuildConfig.use_build_pool_in_warmup``, the first call of an op signature
  lists the kernels of every shape its ``BuildConfig`` sweeps, as tuning buckets do, builds them
  on the rank's workers (``tensorrt_llm._torch.build_pool``), and goes on once they are built.
  The other ops compile their kernels on first use, as in inference mode.
* Rehearsal (inside :meth:`KernelBuilder.rehearsal`: the warmup's forwards on dummy weights before
  the checkpoint load): for an op that sets ``BuildConfig.use_build_pool_in_rehearsal``, a call
  queues its builds and goes on at once. They build beside the load, on the pass's threads when
  ``BuildConfig.thread_safe``, on workers otherwise, and :meth:`KernelBuilder.join` waits for
  them. The other ops compile their kernels on first use, as in inference mode.

A build is a :class:`BuildTask`. ``BuildConfig.strategy=SHARED`` splits the builds across the
ranks that share JIT builds and ``BuildConfig.share_scope``. :meth:`KernelBuilder.build_kernels`
builds a list of ``process_built`` CuTe DSL compile calls, such as a model's prewarm list.
``TLLM_KERNEL_BUILDER=0`` turns KernelBuilder off; ``TLLM_KERNEL_BUILDER_REHEARSAL=0`` turns the
rehearsal off.
"""

from __future__ import annotations

import collections
import contextlib
import dataclasses
import functools
import hashlib
import itertools
import os
import threading
import time
from typing import (
    Any,
    Callable,
    ContextManager,
    Dict,
    Hashable,
    Iterable,
    Iterator,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch

from tensorrt_llm import _jit_share
from tensorrt_llm.logger import logger

from .build_pool import (
    BUILT,
    CLAIMED,
    ENABLE_ENV,
    FAILED,
    PRESENT,
    REFUSED,
    Build,
    BuildBatch,
    BuildPool,
    BuildResult,
    BuildStrategy,
    Claim,
    ThreadBuild,
    _resolve,
)

__all__ = ["BuildConfig", "BuildStrategy", "BuildTask", "DynamicSpec", "KernelBuilder"]

REHEARSAL_ENV = "TLLM_KERNEL_BUILDER_REHEARSAL"


class BuildTask(NamedTuple):
    """One kernel build: ``function`` (``"module:qualname"`` of a module-level function), called
    in a worker process with ``args``, which must pickle."""

    function: str
    args: tuple = ()


@dataclasses.dataclass(frozen=True)
class DynamicSpec:
    """A dimension an op's kernels depend on, swept as the autotuner sweeps a
    ``DynamicTensorSpec``.

    Args:
        name: The dimension's key in the shape a call has and in the shapes swept.
        gen_buckets: The values to build for: a tuple, or a function of the value the
            call has (the largest a forward supplies) returning them. The call's own
            value is always one of them.
    """

    name: str
    gen_buckets: Union[Tuple[int, ...], Callable[[int], Iterable[int]]]


@dataclasses.dataclass(frozen=True)
class BuildConfig:
    """The dimensions an op's kernels depend on, as a ``TuningConfig`` declares its own, and how
    the ranks build them.

    Args:
        dynamic_specs: The dimensions, swept as their ``DynamicSpec`` says.
        strategy: ``BuildStrategy.SHARED`` splits the builds with the other ranks that share JIT
            builds and ``share_scope``; ``PER_RANK`` builds every kernel on this rank.
        share_scope: For SHARED: names where the op's kernels land, such as its compiler's disk
            cache on this host. Only ranks with the same scope split the builds: each loads the
            kernels the others built from there.
        use_build_pool_in_warmup: In build mode, the op's calls build the kernels of the shapes
            they sweep on the build pool's workers before they go on. For ops whose builds are
            long enough to repay a worker; without it, the op compiles on first use.
        use_build_pool_in_rehearsal: In a rehearsal, the op's calls queue the builds of the
            shapes they sweep on the build pool, which builds them beside the checkpoint load.
        thread_safe: The op's builds run outside the GIL and may run on several threads of this
            process at once, allocating no GPU memory: in a rehearsal they run on the pass's
            threads (``BuildPool.threads``) instead of workers, split with the other ranks as
            ``strategy`` says.
    """

    dynamic_specs: Tuple[DynamicSpec, ...] = ()
    strategy: BuildStrategy = BuildStrategy.PER_RANK
    share_scope: Optional[Callable[[], str]] = None
    use_build_pool_in_warmup: bool = False
    use_build_pool_in_rehearsal: bool = False
    thread_safe: bool = False

    def shapes(self, observed: Mapping[str, int]) -> List[Dict[str, int]]:
        """Every combination of the dimensions' buckets for a call that has ``observed``."""
        axes = []
        for spec in self.dynamic_specs:
            value = observed[spec.name]
            buckets = spec.gen_buckets(value) if callable(spec.gen_buckets) else spec.gen_buckets
            axes.append(sorted(set(buckets) | {value}))
        names = [spec.name for spec in self.dynamic_specs]
        return [dict(zip(names, point)) for point in itertools.product(*axes)]


@dataclasses.dataclass
class _OpStats:
    signatures: int = 0
    list_s: float = 0.0
    # Builds made by this rank's workers, failed there, made by other ranks, left to the op,
    # queued on the pass's threads (less those other ranks made).
    built: int = 0
    failed: int = 0
    elsewhere: int = 0
    inline: int = 0
    threads: int = 0
    # Seconds the op's calls spent building, waits for other ranks included.
    build_s: float = 0.0
    late: int = 0
    late_s: float = 0.0


def _claim_files(op: str, key: Hashable, scope: str) -> Optional[Claim]:
    """The claim of ``(op, key)`` among the ranks of ``scope``, in the directory through which
    they share JIT builds; None when they do not share one."""
    share_dir = os.environ.get(_jit_share.SHARE_DIR_ENV)
    if not share_dir or not _jit_share.enabled():
        return None
    name = hashlib.sha256(f"{op}|{key!r}|{scope}".encode()).hexdigest()[:32]
    base = os.path.join(share_dir, "kernel_builder", name)
    return Claim(base + ".claim", base + ".done", base + ".failed")


class KernelBuilder:
    """The rank's kernel builds of ops outside the autotuner.

    ``prebuild`` is called at an op's entry point; outside a build pass it returns after one
    check. ``build`` builds kernels that need only the configuration, ``build_kernels`` the CuTe
    DSL kernels an op lists itself, and ``op_compiled`` records a kernel the op compiled itself.
    """

    _instance: Optional["KernelBuilder"] = None
    _instance_lock = threading.Lock()

    def __init__(self, enabled: Optional[bool] = None, rehearse: Optional[bool] = None):
        self.enabled = os.environ.get(ENABLE_ENV, "1") != "0" if enabled is None else enabled
        # Whether a rehearsal runs: the model loader defers the checkpoint for one.
        self.rehearse = self.enabled and (
            os.environ.get(REHEARSAL_ENV, "1") != "0" if rehearse is None else rehearse
        )
        self._lock = threading.Lock()
        self._passes = 0
        self._rehearsals = 0
        # What join() does once a rehearsal's builds end: one entry per queued batch.
        self._pending: List[Callable[[], None]] = []
        # The shapes listed so far for each (op, call signature).
        self._listed: Dict[Tuple[str, Hashable], set] = {}
        # The (op, key) of every build asked for so far.
        self._requested: set = set()
        self._stats: Dict[str, _OpStats] = {}

    @classmethod
    def get(cls) -> "KernelBuilder":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # Build passes.

    @contextlib.contextmanager
    def build_pass(self, name: str) -> Iterator[None]:
        """Build mode: inside, ops build the kernels of the shapes they sweep on worker
        processes. The outermost pass stops the workers at its end and logs what was built."""
        if not self.enabled:
            yield
            return
        with self._lock:
            self._passes += 1
        t0 = time.perf_counter()
        try:
            with BuildPool.get().pass_scope():
                yield
        finally:
            with self._lock:
                self._passes -= 1
                last = self._passes == 0
            if last:
                for line in self.report(name, time.perf_counter() - t0):
                    logger.info(line)

    def warmup_pass(self, purpose: str) -> ContextManager[None]:
        """The build pass of a warmup of ``purpose``: only the memory-profiling warmup's, whose KV
        cache is not at its final size yet and is sized after the workers stop. A warmup with the
        final KV cache builds in this process: nothing budgets the workers' GPU memory there."""
        if purpose == "memory_profiling":
            return self.build_pass("warmup")
        return contextlib.nullcontext()

    @contextlib.contextmanager
    def rehearsal(self) -> Iterator[None]:
        """Rehearsal mode (see the module): inside, ops queue the builds of the shapes they sweep
        and go on. The caller holds ``BuildPool.pass_scope(overlapped=True)`` open from before the
        rehearsal until after :meth:`join`, so that the builds outlive the rehearsal's forwards."""
        if not self.rehearse:
            yield
            return
        with self._lock:
            self._passes += 1
            self._rehearsals += 1
        try:
            yield
        finally:
            with self._lock:
                self._passes -= 1
                self._rehearsals -= 1

    def active(self) -> bool:
        """Whether an op builds its kernels ahead now (build or rehearsal mode)."""
        return self.enabled and self._passes > 0

    def rehearsing(self) -> bool:
        """Whether a rehearsal is running: an op queues its builds and does not wait for them."""
        return self.enabled and self._rehearsals > 0

    def _uses_pool(self, config: Any) -> bool:
        """Whether the builds of ``config`` (a ``BuildConfig``, or a ``process_built`` function's
        compiler) go to the build pool now: in a rehearsal, as its ``use_build_pool_in_rehearsal``
        says; in build mode, as its ``use_build_pool_in_warmup`` says."""
        if self.rehearsing():
            return config.use_build_pool_in_rehearsal
        return config.use_build_pool_in_warmup

    def join(self, wait: bool = True) -> None:
        """Wait for the builds the rehearsal queued, and log how they ended; with ``wait``
        False, forget them (a rehearsal that failed)."""
        with self._lock:
            pending, self._pending = self._pending, []
        if wait:
            for settle in pending:
                settle()

    # Registration.

    def prebuild(
        self,
        op: str,
        signature: Hashable,
        config: BuildConfig,
        observed: Mapping[str, int],
        list_builds: Callable[[List[Dict[str, int]]], Iterable[tuple]],
    ) -> None:
        """In build mode, for a ``config`` that sets ``use_build_pool_in_warmup``, build the
        kernels ``list_builds(shapes)`` returns (:meth:`build`), and return once they are built;
        in a rehearsal, for one that sets ``use_build_pool_in_rehearsal``, queue them and return.
        ``shapes`` are those ``config`` sweeps for a call that has ``observed`` and that no earlier
        call of ``(op, signature)`` swept. ``list_builds`` returns ``(key, BuildTask)`` pairs, or
        ``(key, BuildTask, BuildConfig)`` for a build whose configuration is not ``config``.

        A failure to list leaves the kernels to the op.
        """
        if not (self.active() and self._uses_pool(config) and BuildPool.get().accepting()):
            return
        swept = config.shapes(observed)
        with self._lock:
            listed = self._listed.setdefault((op, signature), set())
            new_signature = not listed
            shapes = [shape for shape in swept if tuple(sorted(shape.items())) not in listed]
            if not shapes:
                return
            listed.update(tuple(sorted(shape.items())) for shape in shapes)
        t0 = time.perf_counter()
        try:
            builds = {}
            for item in list_builds(shapes):
                key, task = item[0], item[1]
                builds[key] = (key, task, item[2] if len(item) > 2 else config)
        except Exception as e:  # noqa: BLE001 - the op still builds what it needs on first use
            logger.warning(
                f"[KernelBuilder] {op}: listing the kernels of a call signature failed; the op "
                f"builds them on first use. {type(e).__name__}: {e}"
            )
            return
        list_s = time.perf_counter() - t0
        with self._lock:
            stats = self._op_stats(op)
            stats.signatures += new_signature
            stats.list_s += list_s
        logger.info(
            f"[KernelBuilder] {op}: {len(builds)} kernels over {len(shapes)} new shapes of "
            f"{'a new' if new_signature else 'a'} call signature ({list_s:.1f}s to list)"
        )
        self.build(op, list(builds.values()))

    def build(self, op: str, builds: Sequence[Tuple[Hashable, BuildTask, BuildConfig]]) -> None:
        """In build mode, build the kernels of ``builds`` (``(key, BuildTask, BuildConfig)``)
        not asked for before on the rank's worker processes, in parallel, and return once they
        are built; in a rehearsal, queue them and return at once (:meth:`join` waits for them).

        With the SHARED strategy, a build another rank started first, on a worker or a thread, is
        that rank's: this one waits for the builds it left to other ranks, at most
        ``TLLM_JIT_SHARE_WAIT_S`` seconds in all. A build whose configuration does not use the
        build pool now (``use_build_pool_in_warmup``; in a rehearsal,
        ``use_build_pool_in_rehearsal``), one no worker takes, and one that fails are left to the
        op, which builds them on first use.
        """
        if not self.active():
            return
        with self._lock:
            new = []
            for key, task, config in builds:
                if (op, key) not in self._requested:
                    self._requested.add((op, key))
                    new.append((key, task, config))
        if not new:
            return
        pool = BuildPool.get()
        rehearsing = self.rehearsing()
        to_pool = [b for b in new if self._uses_pool(b[2])] if pool.accepting() else []
        # In a rehearsal, a thread-safe op's builds run on the pass's threads.
        on_threads = [b for b in to_pool if rehearsing and b[2].thread_safe]
        to_workers = [b for b in to_pool if not (rehearsing and b[2].thread_safe)]
        if len(to_pool) < len(new):
            with self._lock:
                self._op_stats(op).inline += len(new) - len(to_pool)
        if on_threads:
            self._to_threads(op, on_threads)
        if not to_workers:
            return
        jobs = [
            Build(task.function, task.args, op, self._claim(op, key, config))
            for key, task, config in self._rotated(to_workers)
        ]
        t0 = time.perf_counter()
        batch = pool.submit(jobs)
        if self.rehearsing():
            with self._lock:
                self._pending.append(functools.partial(self._settle, op, jobs, batch, t0))
            return
        self._settle(op, jobs, batch, t0)

    def _to_threads(
        self, op: str, builds: Sequence[Tuple[Hashable, BuildTask, BuildConfig]]
    ) -> None:
        """Queue a thread-safe op's builds on the pass's threads, which split the SHARED ones
        with the other ranks as workers do; a build the threads refuse is the op's."""
        lane = BuildPool.get().threads
        queued, shared = 0, []
        if lane is not None:
            for key, task, config in self._rotated(list(builds)):
                claim = self._claim(op, key, config)
                build = lane.submit(functools.partial(_run_task, task), op, claim=claim)
                if build is not None:
                    queued += 1
                    if claim is not None:
                        shared.append(build)
        with self._lock:
            stats = self._op_stats(op)
            stats.threads += queued
            stats.inline += len(builds) - queued
            if shared:
                self._pending.append(functools.partial(self._settle_threads, op, shared))

    def _settle_threads(self, op: str, builds: List[ThreadBuild]) -> None:
        """Once the pass's threads have run ``builds`` (SHARED ones), wait for those that other
        ranks are making, and count the ones other ranks made."""
        BuildPool.get().join_threads()
        claimed = [build.claim for build in builds if build.status == CLAIMED]
        if claimed:
            with _jit_share.wait_until(time.time() + _jit_share.wait_seconds()):
                late = sum(not _jit_share._wait(c.done, c.failed) for c in claimed)
        else:
            late = 0
        elsewhere = sum(build.status in (PRESENT, CLAIMED) for build in builds)
        with self._lock:
            stats = self._op_stats(op)
            stats.threads -= elsewhere
            stats.elsewhere += elsewhere
        if elsewhere:
            logger.info(
                f"[KernelBuilder] {op}: {elsewhere} of {len(builds)} kernels queued on build "
                f"threads built by other ranks ({late} not in time)"
            )

    def _settle(self, op: str, jobs: List[Build], batch: BuildBatch, t0: float) -> None:
        """Wait for ``batch`` and for the builds it found other ranks making, and count them."""
        results = batch.wait()
        counts = collections.Counter(result.status for result in results)
        claimed = [job.claim for job, result in zip(jobs, results) if result.status == CLAIMED]
        if claimed:
            with _jit_share.wait_until(time.time() + _jit_share.wait_seconds()):
                late = sum(not _jit_share._wait(c.done, c.failed) for c in claimed)
        else:
            late = 0
        seconds = time.perf_counter() - t0
        with self._lock:
            stats = self._op_stats(op)
            stats.built += counts[BUILT]
            stats.failed += counts[FAILED]
            stats.elsewhere += counts[PRESENT] + counts[CLAIMED]
            stats.inline += counts[REFUSED]
            stats.build_s += seconds
        logger.info(
            f"[KernelBuilder] {op}: {counts[BUILT]} of {len(jobs)} kernels built in worker "
            f"processes, {counts[PRESENT] + counts[CLAIMED]} by other ranks ({late} not in time), "
            f"{counts[FAILED]} failed, {counts[REFUSED]} left to the op, in {seconds:.1f}s"
        )

    def build_kernels(self, kernels: Iterable[Tuple[Callable[..., Any], tuple]]) -> int:
        """Compile the kernels of ``(function, args)`` pairs, calling each ``function(*args)``,
        and return the number of pairs: kernels an op knows it needs, such as a model's prewarm
        list. Each ``process_built`` CuTe DSL compile function (``tensorrt_llm._torch.build_pool``)
        is an op here, under its name.

        In build mode, the kernels of a function that sets ``use_build_pool_in_warmup``
        (``process_built``) build on the rank's worker processes first, all together, and the
        calls load them. In a rehearsal, the kernels of a function that sets
        ``use_build_pool_in_rehearsal`` are queued for the workers (:meth:`join` waits for them)
        and do not compile here: the op's calls after the rehearsal load or compile them. The
        other kernels, any a worker did not build, and those of any other function compile in
        the calls, as in inference mode.
        """
        kernels = list(dict.fromkeys(kernels))
        if self.active():
            pool = BuildPool.get()
            counts: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
            to_workers = []
            for fn, args in kernels:
                compiler = getattr(fn, "_compiler", None)
                if compiler is None or args in compiler.memo:
                    continue
                if pool.accepting() and self._uses_pool(compiler):
                    to_workers.append((fn, args))
                else:
                    counts[compiler.name][REFUSED] += 1
            t0 = time.perf_counter()
            batch = pool.submit_kernels(to_workers)
            settle = functools.partial(self._settle_kernels, to_workers, batch, counts, t0)
            if self.rehearsing():
                # The queued kernels do not compile here: the calls after the rehearsal load them.
                with self._lock:
                    self._pending.append(settle)
                queued = set(to_workers)
            else:
                settle()
                queued = set()
        else:
            queued = set()
        for fn, args in kernels:
            if (fn, args) not in queued:
                fn(*args)
        return len(kernels)

    def _settle_kernels(
        self,
        to_workers: List[Tuple[Callable[..., Any], tuple]],
        batch: BuildBatch,
        counts: Dict[str, collections.Counter],
        t0: float,
    ) -> None:
        results: List[BuildResult] = batch.wait()
        seconds = time.perf_counter() - t0
        for (fn, _), result in zip(to_workers, results):
            counts[fn._compiler.name][result.status] += 1
        with self._lock:
            for op, count in counts.items():
                stats = self._op_stats(op)
                stats.built += count[BUILT]
                stats.failed += count[FAILED]
                stats.elsewhere += count[PRESENT] + count[CLAIMED]
                stats.inline += count[REFUSED]
                if count[BUILT] + count[FAILED] + count[CLAIMED]:
                    stats.build_s += seconds
        if to_workers:
            done = sum(result.status == BUILT for result in results)
            logger.info(
                f"[KernelBuilder] {len(to_workers)} listed kernels: {done} built in worker "
                f"processes in {seconds:.1f}s"
            )

    def _claim(self, op: str, key: Hashable, config: BuildConfig) -> Optional[Claim]:
        """The claim of a SHARED build in the directory the ranks share; None for a build that
        is this rank's alone."""
        if config.strategy is not BuildStrategy.SHARED:
            return None
        try:
            scope = config.share_scope() if config.share_scope is not None else ""
            claim = _claim_files(op, key, scope)
            if claim is not None:
                os.makedirs(os.path.dirname(claim.claim), exist_ok=True)
            return claim
        except Exception as e:  # noqa: BLE001 - the op's scope or the directory failed: build here
            logger.debug(f"[KernelBuilder] {op} {key}: built on this rank. {type(e).__name__}: {e}")
            return None

    def stats(self, op: str) -> _OpStats:
        """A copy of ``op``'s counts so far."""
        with self._lock:
            return dataclasses.replace(self._op_stats(op))

    def op_compiled(self, op: str, key: Hashable, seconds: float) -> None:
        """Record a kernel the op's own call had to compile."""
        if not self.enabled:
            return
        with self._lock:
            stats = self._op_stats(op)
            stats.late += 1
            stats.late_s += seconds
        logger.info(
            f"[KernelBuilder] {op} {key}: not built ahead, the op compiled it in {seconds:.1f}s"
        )

    # Reporting.

    def report(self, name: str, seconds: float = 0.0) -> List[str]:
        with self._lock:
            ops = sorted(self._stats.items())
        lines = []
        for op, stats in ops:
            parts = []
            if stats.signatures:
                parts.append(f"{stats.signatures} call signatures listed in {stats.list_s:.1f}s")
            if stats.built or stats.failed:
                parts.append(
                    f"{stats.built} kernels built in worker processes ({stats.failed} failed) "
                    f"in {stats.build_s:.1f}s"
                )
            if stats.elsewhere:
                parts.append(f"{stats.elsewhere} built by other ranks")
            if stats.threads:
                parts.append(f"{stats.threads} queued on build threads")
            if stats.inline:
                parts.append(f"{stats.inline} left to the op")
            if stats.late:
                parts.append(f"{stats.late} compiled by the op itself in {stats.late_s:.1f}s")
            if parts:
                lines.append(f"[KernelBuilder] {name}: {op}: " + "; ".join(parts))
        if lines:
            lines.append(f"[KernelBuilder] {name}: the pass took {seconds:.1f}s")
        return lines

    def _op_stats(self, op: str) -> _OpStats:
        return self._stats.setdefault(op, _OpStats())

    def _rotated(self, builds: List[tuple]) -> List[tuple]:
        """Start each GPU's list at a different place: ranks sharing a compiler cache then
        build different kernels first instead of waiting on each other's locks."""
        if not builds or not torch.cuda.is_available():
            return builds
        offset = torch.cuda.current_device() % len(builds)
        return builds[offset:] + builds[:offset]


def _run_task(task: BuildTask) -> None:
    """Make ``task`` in this process, on a thread of the pass."""
    _resolve(task.function)(*task.args)
