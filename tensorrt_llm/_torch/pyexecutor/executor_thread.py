# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Host-side placement of the executor loop thread.

* ``TLLM_NUMA_AWARE_WORKER_AFFINITY=pin_executor``: the NUMA-aware worker affinity of ``1``,
  and the executor loop (overlap or not) pins its thread to one physical core of that set (one
  core per GPU of the node, never CPU 0's core) while every other thread of the process stays
  off that core and its SMT siblings (:class:`ExecutorThreadPin`).
* ``enable_min_latency``: the interpreter's thread switch interval drops from 5 ms to
  :data:`MIN_LATENCY_SWITCH_INTERVAL_S` while the executor loop runs
  (:func:`executor_switch_interval`), so a CPU-bound background thread holds the GIL against
  the executor for at most that long.
"""

import contextlib
import os
import sys
import threading
from typing import Iterator, Optional, Set

from tensorrt_llm.llmapi.utils import NUMA_AFFINITY_PIN_EXECUTOR as PIN_EXECUTOR_AFFINITY
from tensorrt_llm.logger import logger

NUMA_AFFINITY_ENV = "TLLM_NUMA_AWARE_WORKER_AFFINITY"
MIN_LATENCY_SWITCH_INTERVAL_S = 200e-6


def executor_pin_requested() -> bool:
    """Whether the worker affinity policy pins the executor loop thread."""
    return os.environ.get(NUMA_AFFINITY_ENV, "").strip() == PIN_EXECUTOR_AFFINITY


@contextlib.contextmanager
def executor_switch_interval(min_latency: bool) -> Iterator[None]:
    """Shorten ``sys.setswitchinterval`` for the block under ``enable_min_latency``."""
    if not min_latency:
        yield
        return
    previous = sys.getswitchinterval()
    sys.setswitchinterval(MIN_LATENCY_SWITCH_INTERVAL_S)
    logger.info(
        f"enable_min_latency: thread switch interval {sys.getswitchinterval() * 1e6:.0f} us"
    )
    try:
        yield
    finally:
        sys.setswitchinterval(previous)


def _physical_gpu_index() -> int:
    """The node-local index of this process's GPU (0 without CUDA)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return 0
        device = torch.cuda.current_device()
    except (ImportError, RuntimeError):
        return 0
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if device < len(visible) and visible[device].strip().isdigit():
        return int(visible[device])
    return device


def _smt_siblings(cpu: int) -> Set[int]:
    """``cpu`` and its SMT siblings (just ``cpu`` when sysfs does not say)."""
    path = f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
    try:
        with open(path) as f:
            text = f.read().strip()
    except OSError:
        return {cpu}
    siblings: Set[int] = set()
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            siblings.update(range(int(a), int(b) + 1))
        elif part:
            siblings.add(int(part))
    return siblings | {cpu}


def pin_executor_cpu(available: Set[int], gpu_index: Optional[int] = None) -> Optional[int]:
    """The CPU the executor thread is pinned to, or None for an empty set.

    One physical core per GPU: the ranks of a node often share one NUMA-wide set, so the core
    is the ``gpu_index``-th of the set's physical cores (lowest CPU of each SMT group),
    skipping the core of CPU 0 (housekeeping interrupts).

    Args:
        available: The process's affinity set.
        gpu_index: The node-local index of this process's GPU (default: from the current
            CUDA device and ``CUDA_VISIBLE_DEVICES``).
    """
    cpus = sorted(available)
    if not cpus:
        return None
    cores, seen = [], set()
    for cpu in cpus:
        if cpu in seen:
            continue
        siblings = _smt_siblings(cpu)
        seen |= siblings
        if 0 not in siblings:
            cores.append(cpu)
    if not cores:
        return cpus[0]
    if gpu_index is None:
        gpu_index = _physical_gpu_index()
    return cores[gpu_index % len(cores)]


class ExecutorThreadPin:
    """Keeps the executor loop thread alone on one CPU (``pin_executor`` affinity policy).

    The calling thread gets ``{cpu}``; every other thread of the process gets the process's
    CPU set without ``cpu`` and its SMT siblings. Threads started later inherit the affinity
    of the thread that starts them, so :meth:`maybe_refresh` moves new threads off the CPU
    every ``refresh_iters`` iterations (a scan of ``/proc/self/task``).

    Args:
        cpu: The CPU the calling thread is pinned to.
        available: The process's affinity set.
        refresh_iters: Iterations between two scans for new threads.
    """

    def __init__(self, cpu: int, available: Set[int], refresh_iters: int = 512):
        self.cpu = cpu
        self.tid = threading.get_native_id()
        reserved = _smt_siblings(cpu)
        self.others = set(available) - reserved
        if not self.others:
            # A one-core set: nothing to move the other threads to.
            self.others = set(available) - {cpu} or set(available)
        self.refresh_iters = max(1, refresh_iters)
        self._known: Set[int] = set()

    @classmethod
    def create(cls) -> Optional["ExecutorThreadPin"]:
        """Pin the calling thread under the ``pin_executor`` policy; None otherwise or on failure."""
        if not executor_pin_requested():
            return None
        try:
            available = os.sched_getaffinity(0)
            cpu = pin_executor_cpu(available)
            if cpu is None:
                return None
            pin = cls(cpu, available)
            os.sched_setaffinity(0, {cpu})
            pin.refresh()
        except OSError as error:
            logger.warning(f"{NUMA_AFFINITY_ENV}={PIN_EXECUTOR_AFFINITY}: not pinned ({error})")
            return None
        logger.info(
            f"{NUMA_AFFINITY_ENV}={PIN_EXECUTOR_AFFINITY}: executor thread {pin.tid} pinned to "
            f"CPU {cpu}; {len(pin.others)} CPUs left for the other threads"
        )
        return pin

    def refresh(self) -> int:
        """Move the threads not seen before off the executor's CPU; returns how many moved."""
        moved = 0
        try:
            tids = [int(t) for t in os.listdir("/proc/self/task")]
        except OSError:
            return 0
        for tid in tids:
            if tid == self.tid or tid in self._known:
                continue
            try:
                os.sched_setaffinity(tid, self.others)
                moved += 1
            except OSError:
                # The thread exited, or its affinity is not ours to change.
                pass
            self._known.add(tid)
        self._known.intersection_update(tids)
        return moved

    def maybe_refresh(self, iteration: int) -> None:
        """Run :meth:`refresh` every ``refresh_iters`` iterations."""
        if iteration % self.refresh_iters == 0:
            self.refresh()
