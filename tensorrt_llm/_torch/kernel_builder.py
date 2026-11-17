# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``KernelBuilder``: kernels built on worker processes before the calls that need them.

An op whose kernels compile on first use, and that does not go through the autotuner,
registers them here, the way an op that does registers its tactics with the autotuner.
KernelBuilder has two modes:

* Inference, outside any build pass: an op runs as it is and compiles a kernel it lacks.
* Build, inside a build pass (:meth:`KernelBuilder.build_pass`, which the memory-profiling
  warmup opens, see :meth:`KernelBuilder.warmup_pass`): the first call of an op signature lists
  the kernels of every shape the op's ``BuildConfig`` sweeps from the call's, as a
  ``TuningConfig`` turns a call into tuning buckets, and builds them on the rank's worker
  processes (``tensorrt_llm._torch.build_pool``), in parallel. The call goes on once they are
  built, and the op's calls find them built.

Workers take only builds long enough to repay them (``BuildPool.worth_a_worker``): an op's builds
go to workers once they are measured, or expected (``BuildConfig.build_seconds``), to take
``TLLM_KERNEL_BUILDER_MIN_BUILD_S`` seconds; shorter ones are the op's own. With the PARALLEL
strategy (``BuildConfig.strategy``), the ranks that share JIT builds (``tensorrt_llm._jit_share``)
and the place the op's kernels land (``BuildConfig.share_scope``) split the builds: the worker
that starts a build first makes it, and the other ranks wait for it.

A build is a module-level function and picklable arguments (:class:`BuildTask`). The worker
calls it; the kernel reaches this process through its compiler's disk cache or, for a CuTe DSL
kernel, through the rank's store (``process_built``, whose functions build their kernels with
``build_kernels``).

``TLLM_KERNEL_BUILDER=0`` turns KernelBuilder and the worker processes off: every kernel compiles
on first use.
"""

from __future__ import annotations

import collections
import contextlib
import dataclasses
import hashlib
import itertools
import os
import threading
import time
from typing import (
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
    BuildPool,
    BuildStrategy,
    Claim,
)

__all__ = ["BuildConfig", "BuildStrategy", "BuildTask", "DynamicSpec", "KernelBuilder"]


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
        strategy: ``BuildStrategy.PARALLEL`` splits the builds with the other ranks that share JIT
            builds and ``share_scope``; ``INDEPENDENT`` builds every kernel on this rank.
        share_scope: For PARALLEL: names where the op's kernels land, such as its compiler's disk
            cache on this host. Only ranks with the same scope split the builds: each loads the
            kernels the others built from there.
        build_seconds: How long one build of the op takes, until builds of it are measured; it
            decides whether workers take them (``BuildPool.worth_a_worker``).
    """

    dynamic_specs: Tuple[DynamicSpec, ...] = ()
    strategy: BuildStrategy = BuildStrategy.INDEPENDENT
    share_scope: Optional[Callable[[], str]] = None
    build_seconds: Optional[float] = None

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
    # Builds made by this rank's workers, failed there, made by other ranks, left to the op.
    built: int = 0
    failed: int = 0
    elsewhere: int = 0
    inline: int = 0
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
    check. ``build`` builds kernels that need only the configuration, and ``op_compiled`` records
    a kernel the op compiled itself.
    """

    _instance: Optional["KernelBuilder"] = None
    _instance_lock = threading.Lock()

    def __init__(self, enabled: Optional[bool] = None):
        self.enabled = os.environ.get(ENABLE_ENV, "1") != "0" if enabled is None else enabled
        self._lock = threading.Lock()
        self._passes = 0
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

    def active(self) -> bool:
        """Whether an op builds its kernels ahead now (build mode)."""
        return self.enabled and self._passes > 0

    # Registration.

    def prebuild(
        self,
        op: str,
        signature: Hashable,
        config: BuildConfig,
        observed: Mapping[str, int],
        list_builds: Callable[[List[Dict[str, int]]], Iterable[tuple]],
    ) -> None:
        """In build mode, build the kernels ``list_builds(shapes)`` returns (:meth:`build`), and
        return once they are built; ``shapes`` are those ``config`` sweeps for a call that has
        ``observed`` and that no earlier call of ``(op, signature)`` swept. ``list_builds``
        returns ``(key, BuildTask)`` pairs, or ``(key, BuildTask, BuildConfig)`` for a build whose
        configuration is not ``config``.

        A failure to list leaves the kernels to the op, and so do builds of ``op`` too short for
        a worker (``BuildPool.worth_a_worker``), which are not listed.
        """
        if not self.active() or not BuildPool.get().worth_a_worker(op, config.build_seconds):
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
        are built.

        With the PARALLEL strategy, a build another rank's worker started first is that rank's:
        this one waits for the builds it left to other ranks, at most ``TLLM_JIT_SHARE_WAIT_S``
        seconds in all. A build no worker takes, one too short for a worker
        (``BuildPool.worth_a_worker``), and one that fails are left to the op, which builds them
        on first use.
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
        expected = next((c.build_seconds for _, _, c in new if c.build_seconds is not None), None)
        if not pool.worth_a_worker(op, expected):
            with self._lock:
                self._op_stats(op).inline += len(new)
            return
        jobs = [
            Build(task.function, task.args, op, self._claim(op, key, config))
            for key, task, config in self._rotated(new)
        ]
        t0 = time.perf_counter()
        results = pool.build(jobs)
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

    def _claim(self, op: str, key: Hashable, config: BuildConfig) -> Optional[Claim]:
        """The claim of a PARALLEL build in the directory the ranks share; None for a build that
        is this rank's alone."""
        if config.strategy is not BuildStrategy.PARALLEL:
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
        """Record a kernel the op's own call had to compile: it measures the op's builds."""
        if not self.enabled:
            return
        BuildPool.get().record_build(op, seconds)
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
            parts = [
                f"{stats.signatures} call signatures listed in {stats.list_s:.1f}s",
                f"{stats.built} kernels built in worker processes ({stats.failed} failed) in "
                f"{stats.build_s:.1f}s",
            ]
            if stats.elsewhere:
                parts.append(f"{stats.elsewhere} built by other ranks")
            if stats.inline:
                parts.append(f"{stats.inline} left to the op")
            if stats.late:
                parts.append(f"{stats.late} compiled by the op itself in {stats.late_s:.1f}s")
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
