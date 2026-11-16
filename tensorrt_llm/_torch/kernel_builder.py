# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``KernelBuilder``: kernels built in worker processes ahead of the calls that need them.

An op whose kernels compile on first use, and that does not go through the autotuner,
registers them here, the way an op that does registers its tactics with the autotuner.
KernelBuilder has two modes:

* Inference, outside any build pass: an op runs as it is and compiles a kernel it lacks.
* Build, inside a build pass (:meth:`KernelBuilder.build_pass`, which the memory-profiling
  warmup opens, see :meth:`KernelBuilder.warmup_pass`): the
  first call of an op signature lists the kernels of every shape the op's ``BuildConfig`` sweeps
  from the call's, as a ``TuningConfig`` turns a call into tuning buckets, and KernelBuilder hands
  each to a worker process of the rank (``tensorrt_llm._torch.rank_worker``). They build in
  parallel while the warmup goes on; the op's later calls find a kernel built, or wait for its
  build (:meth:`KernelBuilder.wait`).

A build is a module-level function and picklable arguments (:class:`BuildTask`). The worker
calls it; the kernel reaches this process through its compiler's disk cache or, for a CuTe DSL
kernel, through the rank's store (``process_built``). A CuTe DSL compile function decorated with
``process_built`` is submitted with its own ``submit``.

``TLLM_KERNEL_BUILDER=0`` turns KernelBuilder and the worker processes off: every kernel compiles
on first use.
"""

from __future__ import annotations

import contextlib
import dataclasses
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
    Tuple,
    Union,
)

import torch

from tensorrt_llm.logger import logger

from .rank_worker import ENABLE_ENV, RankWorker

__all__ = ["BuildConfig", "BuildTask", "DynamicSpec", "KernelBuilder"]

# A wait for a build shorter than this is a lookup, not a wait.
_WAIT_REPORT_S = 0.1


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
    """The dimensions an op's kernels depend on, as a ``TuningConfig`` declares its own."""

    dynamic_specs: Tuple[DynamicSpec, ...] = ()

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
    submitted: int = 0
    inline: int = 0
    waited: int = 0
    waited_s: float = 0.0
    late: int = 0
    late_s: float = 0.0


class KernelBuilder:
    """The rank's kernel builds of ops outside the autotuner.

    ``prebuild`` is called at an op's entry point; outside a build pass it returns after one
    check. ``submit`` hands one build to a worker directly, for builds that need only the
    configuration. ``wait`` waits for a build before the op compiles the kernel itself, and
    ``op_compiled`` records a kernel the op compiled itself.
    """

    _instance: Optional["KernelBuilder"] = None
    _instance_lock = threading.Lock()

    def __init__(self, enabled: Optional[bool] = None):
        self.enabled = os.environ.get(ENABLE_ENV, "1") != "0" if enabled is None else enabled
        self._lock = threading.Lock()
        self._passes = 0
        # The shapes listed so far for each (op, call signature).
        self._listed: Dict[Tuple[str, Hashable], set] = {}
        # The worker task key of each (op, key) submitted.
        self._tasks: Dict[Tuple[str, Hashable], str] = {}
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
        """Build mode: inside, ops list the kernels of the shapes they sweep, and worker
        processes build them. The outermost pass waits for the builds at its end, stops the
        workers and logs what was built."""
        if not self.enabled:
            yield
            return
        with self._lock:
            self._passes += 1
        t0 = time.perf_counter()
        try:
            with RankWorker.get().pass_scope():
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
        """Whether an op lists its kernels now (build mode)."""
        return self.enabled and self._passes > 0

    # Registration.

    def prebuild(
        self,
        op: str,
        signature: Hashable,
        config: BuildConfig,
        observed: Mapping[str, int],
        list_builds: Callable[[List[Dict[str, int]]], Iterable[Tuple[Hashable, BuildTask]]],
    ) -> None:
        """In build mode, submit every build ``list_builds(shapes)`` returns as
        ``(key, BuildTask)``, where ``shapes`` are those ``config`` sweeps for a call that has
        ``observed`` and that no earlier call of ``(op, signature)`` swept.

        A failure to list leaves the kernels to the op.
        """
        if not self.active():
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
            builds = list(dict(list_builds(shapes)).items())
        except Exception as e:  # noqa: BLE001 - the op still builds what it needs on first use
            logger.warning(
                f"[KernelBuilder] {op}: listing the kernels of a call signature failed; the op "
                f"builds them on first use. {type(e).__name__}: {e}"
            )
            return
        list_s = time.perf_counter() - t0
        queued = sum(self.submit(op, key, task) for key, task in self._rotated(builds))
        with self._lock:
            stats = self._op_stats(op)
            stats.signatures += new_signature
            stats.list_s += list_s
        logger.info(
            f"[KernelBuilder] {op}: {len(builds)} kernels over {len(shapes)} new shapes of "
            f"{'a new' if new_signature else 'a'} call signature, {queued} queued "
            f"({list_s:.1f}s to list)"
        )

    def submit(self, op: str, key: Hashable, task: BuildTask) -> bool:
        """Hand one build to a worker; False when ``(op, key)`` was submitted before, outside
        build mode, and when no worker takes it (the op then builds it on first use)."""
        if not self.active():
            return False
        task_key = f"kernel_builder|{op}|{key!r}"
        with self._lock:
            if (op, key) in self._tasks:
                return False
            self._tasks[(op, key)] = task_key
        taken = RankWorker.get().run(task_key, task.function, task.args) is not None
        with self._lock:
            stats = self._op_stats(op)
            if taken:
                stats.submitted += 1
            else:
                stats.inline += 1
                del self._tasks[(op, key)]
        return taken

    def wait(self, op: str, key: Hashable) -> float:
        """Wait for the build of ``(op, key)`` when a worker is making it, so that the op's own
        lookup finds the kernel instead of compiling it again; when no worker has started it, take
        it back for the op to make now. Returns the seconds waited."""
        with self._lock:
            task_key = self._tasks.get((op, key))
        if task_key is None:
            return 0.0
        seconds = RankWorker.get().wait(task_key)
        if seconds >= _WAIT_REPORT_S:
            with self._lock:
                stats = self._op_stats(op)
                stats.waited += 1
                stats.waited_s += seconds
        return seconds

    def state(self, op: str, key: Hashable) -> Optional[str]:
        """``"queued"``, the build's status once it ended (``"done"``, ``"failed"``, or
        ``"taken"`` when the op made it itself), or None for a kernel never submitted."""
        with self._lock:
            task_key = self._tasks.get((op, key))
        return None if task_key is None else RankWorker.get().state(task_key)

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
        worker = RankWorker.get()
        with self._lock:
            ops = sorted(self._stats.items())
            tasks = dict(self._tasks)
        lines = []
        for op, stats in ops:
            states = [worker.state(task_key) for (o, _), task_key in tasks.items() if o == op]
            failed = sum(state == "failed" for state in states)
            parts = [
                f"{stats.signatures} call signatures listed in {stats.list_s:.1f}s",
                f"{stats.submitted} builds in worker processes ({failed} failed)",
            ]
            if stats.inline:
                parts.append(f"{stats.inline} left to the op")
            if stats.waited:
                parts.append(f"the op waited {stats.waited_s:.1f}s for {stats.waited} of them")
            if stats.late:
                parts.append(f"{stats.late} compiled by the op itself in {stats.late_s:.1f}s")
            lines.append(f"[KernelBuilder] {name}: {op}: " + "; ".join(parts))
        if lines:
            lines.append(f"[KernelBuilder] {name}: the pass took {seconds:.1f}s")
        return lines

    def _op_stats(self, op: str) -> _OpStats:
        return self._stats.setdefault(op, _OpStats())

    def _rotated(
        self, builds: List[Tuple[Hashable, BuildTask]]
    ) -> List[Tuple[Hashable, BuildTask]]:
        """Start each GPU's list at a different place: ranks sharing a compiler cache then
        build different kernels first instead of waiting on each other's locks."""
        if not builds or not torch.cuda.is_available():
            return builds
        offset = torch.cuda.current_device() % len(builds)
        return builds[offset:] + builds[:offset]
