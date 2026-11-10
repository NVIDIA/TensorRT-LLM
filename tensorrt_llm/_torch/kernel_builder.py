# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``KernelBuilder``: kernels built on worker threads before the calls that need them.

An op whose kernels compile on first use, and that does not go through the autotuner,
registers them here. On the first call of each call signature inside a build pass, the
op lists every kernel that signature reaches over its runtime shapes; worker threads
then build them in parallel. The op's own later call finds a kernel built, or waits on
the op's own lock for a build that is still running.

A rehearsal is a build pass over forwards on dummy weights, run before the checkpoint
loads: the ops list their kernels without building or launching them, and the pass does
not wait, so the builds run while the weights load. It runs for a model that one of the
op families in ``_FAMILIES`` reaches.

``TLLM_KERNEL_BUILDER=0`` turns it off: no pass is active and nothing is submitted.
``TLLM_KERNEL_BUILDER_REHEARSAL=0`` keeps the builds but not the rehearsal: they start
at the warmup. ``TLLM_KERNEL_BUILDER_THREADS`` sets the number of worker threads
(default 8).
"""

from __future__ import annotations

import contextlib
import dataclasses
import importlib
import os
import queue
import threading
import time
from typing import Any, Callable, Dict, Hashable, Iterable, Iterator, List, Optional, Tuple

import torch

from tensorrt_llm.logger import logger

__all__ = ["KernelBuilder"]

_ENV = "TLLM_KERNEL_BUILDER"
_REHEARSAL_ENV = "TLLM_KERNEL_BUILDER_REHEARSAL"
_THREADS_ENV = "TLLM_KERNEL_BUILDER_THREADS"
_DEFAULT_THREADS = 8
_PROGRESS_EVERY_S = 60.0

# The op families a rehearsal can reach: name -> "module:function". The function takes the
# model config, returns True when the model reaches the op, and then queues the builds the
# configuration alone determines. A module is imported only when asked.
_FAMILIES = {
    "fmha_sm100": (
        "tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.msa_kernel_builds:reaches"
    ),
}

# A build returns True when it compiled the kernel, False when the kernel was already there.
Build = Callable[[], bool]

_QUEUED, _DONE, _FAILED = "queued", "done", "failed"


@dataclasses.dataclass
class _OpStats:
    signatures: int = 0
    list_s: float = 0.0
    submitted: int = 0
    built: int = 0
    present: int = 0
    failed: int = 0
    build_s: float = 0.0
    late: int = 0
    late_s: float = 0.0
    waited: int = 0
    waited_s: float = 0.0

    def summary(self) -> str:
        parts = [
            f"{self.signatures} call signatures listed in {self.list_s:.1f}s",
            f"{self.submitted} kernels submitted: {self.built} built in {self.build_s:.1f}s, "
            f"{self.present} already built, {self.failed} failed",
        ]
        if self.waited:
            parts.append(f"the op waited {self.waited_s:.1f}s for {self.waited} of them")
        if self.late:
            parts.append(f"{self.late} compiled by the op itself in {self.late_s:.1f}s")
        return "; ".join(parts)


class KernelBuilder:
    """Process-wide queue of kernel builds, run on worker threads.

    ``prebuild`` is called at an op's entry point; outside a build pass it returns after
    one check. ``submit`` queues one build directly, for builds that need only the
    configuration. ``op_compiled`` records a kernel the op compiled itself.
    """

    _instance: Optional["KernelBuilder"] = None
    _instance_lock = threading.Lock()

    def __init__(
        self,
        threads: Optional[int] = None,
        enabled: Optional[bool] = None,
        rehearse: Optional[bool] = None,
    ):
        self.enabled = os.environ.get(_ENV, "1") != "0" if enabled is None else enabled
        if rehearse is None:
            rehearse = os.environ.get(_REHEARSAL_ENV, "1") != "0"
        self.rehearse = self.enabled and rehearse
        if threads is None:
            threads = int(os.environ.get(_THREADS_ENV, str(_DEFAULT_THREADS)))
        self.threads = max(1, threads)
        self._lock = threading.Lock()
        self._idle = threading.Condition(self._lock)
        self._queue: "queue.Queue[Tuple[str, Hashable, Build]]" = queue.Queue()
        self._workers: List[threading.Thread] = []
        self._device: Optional[int] = None
        self._passes = 0
        self._rehearsals = 0
        self._pending = 0
        self._signatures: set = set()
        self._states: Dict[Tuple[str, Hashable], str] = {}
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
    def build_pass(self, name: str, wait: bool = True) -> Iterator[None]:
        """Ops list their kernels on first sight inside. The outermost pass logs what was
        built at its end, after waiting for the builds unless ``wait`` is False."""
        if not self.enabled:
            yield
            return
        with self._lock:
            self._passes += 1
        try:
            yield
        except BaseException:
            with self._lock:
                self._passes -= 1
            raise
        with self._lock:
            self._passes -= 1
            last = self._passes == 0
        if last:
            waited = self.join() if wait else 0.0
            for line in self.report(name, waited):
                logger.info(line)

    @contextlib.contextmanager
    def rehearsal(self) -> Iterator[None]:
        """A build pass over forwards on dummy weights. Ops list their kernels without
        building or launching them, and the pass does not wait for the builds."""
        if not self.enabled:
            yield
            return
        with self._lock:
            self._rehearsals += 1
        try:
            with self.build_pass("rehearsal", wait=False):
                yield
        finally:
            with self._lock:
                self._rehearsals -= 1

    def active(self) -> bool:
        return self.enabled and self._passes > 0

    def rehearsing(self) -> bool:
        """True inside a rehearsal: an op lists its kernels, then skips its own launch."""
        return self.enabled and self._rehearsals > 0

    def wants_rehearsal(self, model_config: Any) -> bool:
        """True when the rehearsal is on and an op family in ``_FAMILIES`` reaches the
        model. Each family that does queues the builds its configuration determines."""
        if not self.rehearse:
            return False
        wanted = False
        for name, target in _FAMILIES.items():
            module_name, function = target.split(":")
            try:
                reaches = getattr(importlib.import_module(module_name), function)
                wanted = bool(reaches(model_config)) or wanted
            except Exception as e:  # noqa: BLE001 - the family's kernels build at the warmup
                logger.warning(
                    f"[KernelBuilder] {name}: could not tell whether the model reaches it; "
                    f"its kernels build at the warmup. {type(e).__name__}: {e}"
                )
        return wanted

    # Registration.

    def prebuild(
        self,
        op: str,
        signature: Hashable,
        list_builds: Callable[[], Iterable[Tuple[Hashable, Build]]],
    ) -> None:
        """On the first sight of ``(op, signature)`` in a build pass, submit every build
        ``list_builds()`` returns as ``(key, build)``.

        A failure to list leaves the kernels to the op.
        """
        if not self.active():
            return
        with self._lock:
            if (op, signature) in self._signatures:
                return
            self._signatures.add((op, signature))
        t0 = time.perf_counter()
        try:
            builds = list(list_builds())
        except Exception as e:  # noqa: BLE001 - the op still builds what it needs on first use
            logger.warning(
                f"[KernelBuilder] {op}: listing the kernels of a call signature failed; the op "
                f"builds them on first use. {type(e).__name__}: {e}"
            )
            return
        list_s = time.perf_counter() - t0
        queued = sum(self.submit(op, key, build) for key, build in self._rotated(builds))
        with self._lock:
            stats = self._op_stats(op)
            stats.signatures += 1
            stats.list_s += list_s
        logger.info(
            f"[KernelBuilder] {op}: {len(builds)} kernels for a new call signature, "
            f"{queued} queued ({list_s:.1f}s to list)"
        )

    def submit(self, op: str, key: Hashable, build: Build) -> bool:
        """Queue one build; False when ``(op, key)`` was already queued or built."""
        if not self.enabled:
            return False
        with self._lock:
            if (op, key) in self._states:
                return False
            self._states[(op, key)] = _QUEUED
            self._op_stats(op).submitted += 1
            self._pending += 1
            self._start_workers()
        self._queue.put((op, key, build))
        return True

    def state(self, op: str, key: Hashable) -> Optional[str]:
        """``"queued"``, ``"done"``, ``"failed"``, or None for a kernel never submitted."""
        with self._lock:
            return self._states.get((op, key))

    def stats(self, op: str) -> _OpStats:
        """A copy of ``op``'s counts so far."""
        with self._lock:
            return dataclasses.replace(self._op_stats(op))

    def op_compiled(self, op: str, key: Hashable, seconds: float) -> None:
        """Record a kernel the op's own call had to build or wait for."""
        if not self.enabled:
            return
        with self._lock:
            stats = self._op_stats(op)
            waited = self._states.get((op, key)) in (_QUEUED, _DONE)
            if waited:
                stats.waited += 1
                stats.waited_s += seconds
            else:
                stats.late += 1
                stats.late_s += seconds
        if waited:
            logger.info(f"[KernelBuilder] {op} {key}: the op waited {seconds:.1f}s for its build")
        else:
            logger.info(
                f"[KernelBuilder] {op} {key}: not built ahead, the op compiled it in {seconds:.1f}s"
            )

    # Waiting and reporting.

    def join(self, timeout: Optional[float] = None) -> float:
        """Wait for every queued build; returns the seconds waited."""
        t0 = last = time.perf_counter()
        with self._idle:
            while self._pending > 0:
                now = time.perf_counter()
                if timeout is not None and now - t0 >= timeout:
                    break
                if now - last >= _PROGRESS_EVERY_S:
                    logger.info(
                        f"[KernelBuilder] waiting {now - t0:.0f}s for {self._pending} builds"
                    )
                    last = now
                self._idle.wait(timeout=1.0)
        return time.perf_counter() - t0

    def report(self, name: str, waited: float = 0.0) -> List[str]:
        with self._lock:
            lines = [
                f"[KernelBuilder] {name}: {op}: {stats.summary()}"
                for op, stats in sorted(self._stats.items())
            ]
            pending = self._pending
        if lines and waited >= 0.1:
            lines.append(f"[KernelBuilder] {name}: waited {waited:.1f}s for the builds at its end")
        if pending:
            lines.append(f"[KernelBuilder] {name}: {pending} builds continue after it")
        return lines

    # Workers.

    def _op_stats(self, op: str) -> _OpStats:
        return self._stats.setdefault(op, _OpStats())

    def _rotated(self, builds: List[Tuple[Hashable, Build]]) -> List[Tuple[Hashable, Build]]:
        """Start each GPU's list at a different place: ranks sharing a cache directory then
        build different kernels first instead of waiting on each other's locks."""
        if not builds or not torch.cuda.is_available():
            return builds
        offset = torch.cuda.current_device() % len(builds)
        return builds[offset:] + builds[:offset]

    def _start_workers(self) -> None:
        if self._device is None and torch.cuda.is_available():
            self._device = torch.cuda.current_device()
        while len(self._workers) < min(self.threads, self._pending):
            worker = threading.Thread(
                target=self._work, name=f"tllm-kernel-builder-{len(self._workers)}", daemon=True
            )
            worker.start()
            self._workers.append(worker)

    def _work(self) -> None:
        device_set = False
        while True:
            op, key, build = self._queue.get()
            t0 = time.perf_counter()
            state = _DONE
            # Every item is settled, whatever it raises: an unsettled one would hang join().
            try:
                if not device_set and self._device is not None:
                    torch.cuda.set_device(self._device)
                    device_set = True
                built = build()
            except BaseException as e:  # noqa: BLE001 - the op builds a failed kernel on first use
                state, built = _FAILED, False
                logger.warning(
                    f"[KernelBuilder] {op} {key}: build failed; the op builds it on first use. "
                    f"{type(e).__name__}: {e}"
                )
            seconds = time.perf_counter() - t0
            with self._idle:
                self._states[(op, key)] = state
                stats = self._op_stats(op)
                if state == _FAILED:
                    stats.failed += 1
                elif built:
                    stats.built += 1
                    stats.build_s += seconds
                else:
                    stats.present += 1
                self._pending -= 1
                if self._pending == 0:
                    self._idle.notify_all()
