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
"""The low-latency decode-step shell (everything around the model's layers).

A model asks for it with a class attribute ``ull_step_shell`` (a :class:`StepShell`; the Kimi-K3
helix16 ULL target and a modeling_v2 ULL generation target set one). The decoder runner
adopts it when it is built (:func:`adopt`), once per process, before warm-up. A model without the
attribute leaves the shell to the environment switches below (each read once per process, all
off by default):

* ``step_prep`` (``TRTLLM_K3_ULL_STEP_PREP=1``): one-launch step preparation. The batched input
  commit (``input_commit.py``) runs for CUDA-graph generation steps, also on engines whose MLA
  layers use the FP4 KV cache (``TrtllmAttentionMetadata.fp4_mla_state``, the Kimi-K3 helix path)
  and on a DSA metadata subclass that stages its uploads, which ``TRTLLM_BATCHED_INPUT_COMMIT=1`` alone leaves on the
  copy-by-copy path.
* ``pin_executor`` (``TRTLLM_K3_ULL_PIN_EXECUTOR=auto``): pin the executor loop thread to one
  physical core of the process's affinity set (one core per GPU of the node, never CPU 0's core)
  and keep the process's other threads off that core and its SMT siblings
  (:class:`ExecutorThreadPin`).
* ``gil_switch_us`` (``TRTLLM_K3_ULL_GIL_SWITCH_US=<us>``): the interpreter's thread switch
  interval (``sys.setswitchinterval``, default 5000 us; 0 keeps it). A thread that wants the GIL
  asks its holder to drop it only after this interval, so a CPU-bound background thread can hold
  the executor for up to the interval.
* ``step_fold`` (``TRTLLM_K3_ULL_STEP_FOLD=1``): the device work between two decode graph replays
  stays on the execution stream as back-to-back kernels, without copy-engine or cross-stream
  hand-offs: the DFlash draft-history snapshot is one kernel and its readback runs on a side
  stream, the mamba state-index upload of a generation-only step runs on the execution stream,
  and the KV block-offset copy of a committed step runs inside the input commit kernel
  (:class:`StepFold`). ``TRTLLM_K3_ULL_STEP_FOLD_AB_PERIOD=<n>`` folds only every other block of
  ``n`` iterations (A/B arms inside one run).
* ``spec_head_fusion``: the overlap-scheduler position / KV-length correction and the Helix
  verify-group buffers of ``preprocess_inputs`` in one kernel (``TLLM_SPEC_HEAD_FUSION=1``,
  :func:`spec_head_fusion_enabled`).
* ``verify_commits``: how many input commits (and folded snapshots / block-offset copies) are
  checked against the unfolded copies (synchronizing; a mismatch raises).
  ``TRTLLM_BATCHED_INPUT_COMMIT_VERIFY`` and ``TRTLLM_K3_ULL_STEP_FOLD_VERIFY`` override it.

The decode CUDA graphs of every one-model speculative engine without MRoPE read their input ids
and positions straight from the engine's buffers (``DecoderRunner._cuda_graph_runner_config``), and
its sampler runs on the execution stream (``PyExecutor._executor_loop_overlap``); the shell has
nothing to add there (:func:`report_default_on_switches`).
"""

import dataclasses
import os
import sys
import threading
from typing import Optional, Set

from tensorrt_llm.logger import logger

_LOG = "ULL step shell"

STEP_PREP_ENV = "TRTLLM_K3_ULL_STEP_PREP"
GRAPH_INPUT_ALIAS_ENV = "TRTLLM_K3_ULL_GRAPH_INPUT_ALIAS"
SAMPLE_EXEC_STREAM_ENV = "TRTLLM_K3_ULL_SAMPLE_EXEC_STREAM"
PIN_EXECUTOR_ENV = "TRTLLM_K3_ULL_PIN_EXECUTOR"
GIL_SWITCH_US_ENV = "TRTLLM_K3_ULL_GIL_SWITCH_US"
STEP_FOLD_ENV = "TRTLLM_K3_ULL_STEP_FOLD"
STEP_FOLD_AB_PERIOD_ENV = "TRTLLM_K3_ULL_STEP_FOLD_AB_PERIOD"
STEP_FOLD_VERIFY_ENV = "TRTLLM_K3_ULL_STEP_FOLD_VERIFY"

_STEP_PREP_FROM_ENV = os.environ.get(STEP_PREP_ENV, "0") == "1"


@dataclasses.dataclass(frozen=True)
class StepShell:
    """The decode-step shell a model asks for (its ``ull_step_shell``; module docstring)."""

    step_prep: bool = False
    step_fold: bool = False
    pin_executor: bool = False
    gil_switch_us: int = 0
    # Not part of the shell's log line, which stays identical across targets.
    verify_commits: int = dataclasses.field(default=0, repr=False)
    # The decode-step head in one kernel (``speculative/spec_head_kernels.py``), as with
    # ``TLLM_SPEC_HEAD_FUSION=1``. Not part of the log line either.
    spec_head_fusion: bool = dataclasses.field(default=False, repr=False)

    def __post_init__(self):
        if self.gil_switch_us < 0:
            raise ValueError(f"gil_switch_us must be >= 0, got {self.gil_switch_us}")
        if self.verify_commits < -1:
            raise ValueError(f"verify_commits must be >= -1, got {self.verify_commits}")


# The shell a model asked for (adopt); None leaves every piece to the environment switches.
_SHELL: Optional[StepShell] = None


def shell() -> Optional[StepShell]:
    """The step shell a model of this process asked for, or None (environment switches)."""
    return _SHELL


def _requested(model) -> Optional[StepShell]:
    modules = getattr(model, "modules", None)
    candidates = modules() if callable(modules) else (model,)
    for module in candidates:
        found = getattr(type(module), "ull_step_shell", None)
        if isinstance(found, StepShell):
            return found
    return None


def adopt(model) -> Optional[StepShell]:
    """Take the step shell ``model`` (or one of its modules) asks for, if any; a model that asks
    for none leaves the process's shell as it is. Call before the model's warm-up."""
    global _SHELL
    found = _requested(model)
    if found is None or found == _SHELL:
        return _SHELL
    _SHELL = found
    STEP_FOLD.enable(found.step_fold)
    if STEP_FOLD_VERIFY_ENV not in os.environ:
        STEP_FOLD.verify = max(found.verify_commits, 0)
    logger.info(f"{_LOG}: {found}")
    return _SHELL


def step_prep_enabled() -> bool:
    """Whether the batched input commit runs, also on the FP4 MLA KV and staged DSA metadata paths."""
    return _SHELL.step_prep if _SHELL is not None else _STEP_PREP_FROM_ENV


def spec_head_fusion_requested() -> bool:
    """Whether the model's shell asks for the fused decode-step head."""
    return _SHELL is not None and _SHELL.spec_head_fusion


def default_verify_commits() -> int:
    """Input commits to verify when ``TRTLLM_BATCHED_INPUT_COMMIT_VERIFY`` is unset."""
    return _SHELL.verify_commits if _SHELL is not None else 0


def report_default_on_switches() -> None:
    """Log the shell switches that select this executor's default behavior, when set."""
    defaults = {
        GRAPH_INPUT_ALIAS_ENV: "decode CUDA graphs of one-model speculative engines read "
        "the engine's input id and position buffers",
        SAMPLE_EXEC_STREAM_ENV: "the one-model speculative sampler runs on the execution stream",
    }
    for name, behavior in defaults.items():
        value = os.environ.get(name)
        if value is not None:
            logger.info(f"{name}={value}: always on here ({behavior}); the switch has no effect")


class StepFold:
    """State of the shell's ``step_fold`` (one per process, :data:`STEP_FOLD`).

    ``active`` says whether the current iteration folds; it is False whenever the fold is off, so
    call sites test it alone. The executor loop registers its execution stream (:meth:`start`) and
    reports each iteration (:meth:`on_iteration`). ``count`` records how often each folded site ran
    and fell back (logged every ``LOG_EVERY`` iterations).

    Args:
        enabled: Whether the fold is on (:func:`adopt` sets it from the model's shell).
        ab_period: Iterations per A/B block, 0 for none. With a period, iteration ``i``
            folds only when ``(i // ab_period) % 2 == 1`` (the even blocks are the A arm).
        verify: How many folded snapshots to check against the unfolded ops.
    """

    LOG_EVERY = 2048

    def __init__(self, enabled: bool, ab_period: int = 0, verify: int = 0):
        if ab_period < 0:
            raise ValueError(f"StepFold ab_period must be >= 0, got {ab_period}")
        self.enabled = enabled
        self.ab_period = ab_period
        self.verify = verify
        # Development A/B in one process: folding held off.
        self.suppressed = False
        self.active = enabled and ab_period == 0
        self.stream = None
        self.iteration = 0
        self.counts = {}
        self.num_block_checks = 0

    @classmethod
    def from_env(cls) -> "StepFold":
        return cls(
            os.environ.get(STEP_FOLD_ENV, "0") == "1",
            int(os.environ.get(STEP_FOLD_AB_PERIOD_ENV, "0") or "0"),
            int(os.environ.get(STEP_FOLD_VERIFY_ENV, "0") or "0"),
        )

    def _update(self) -> None:
        on = self.enabled and not self.suppressed
        if self.ab_period:
            on = on and (self.iteration // self.ab_period) % 2 == 1
        self.active = on

    def enable(self, on: bool) -> None:
        """Turn the fold on or off (the model's shell; before the objects that fold are built)."""
        self.enabled = bool(on)
        self._update()

    def suppress(self, off: bool) -> None:
        """Development A/B: hold the fold off (``off``) or let it run where it is enabled."""
        self.suppressed = bool(off)
        self._update()

    def start(self, execution_stream) -> bool:
        """Register the executor's execution stream; returns whether the fold is on."""
        if not self.enabled:
            return False
        self.stream = execution_stream
        logger.info(
            f"{_LOG}: step prep folded onto the execution stream"
            + (f", A/B blocks of {self.ab_period} iterations" if self.ab_period else "")
        )
        return True

    def on_iteration(self, iteration: int) -> None:
        """Select the arm of ``iteration`` and log the counts every ``LOG_EVERY`` iterations."""
        self.iteration = iteration
        if self.ab_period:
            self._update()
        if iteration and iteration % self.LOG_EVERY == 0 and self.counts:
            counts = ", ".join(f"{k} {v}" for k, v in sorted(self.counts.items()))
            logger.info(f"{_LOG}: step fold iter {iteration}: {counts}")

    def take_block_check(self) -> bool:
        """Whether to check this folded block-offset copy (the first ``verify`` ones)."""
        if self.num_block_checks >= self.verify:
            return False
        self.num_block_checks += 1
        if self.num_block_checks == self.verify:
            logger.info(
                f"{_LOG}: step fold verify: checking the last of {self.verify} block copies"
            )
        return True

    def count(self, what: str) -> None:
        first = what not in self.counts
        self.counts[what] = self.counts.get(what, 0) + 1
        if first:
            logger.info(f"{_LOG}: step fold: first {what} at iteration {self.iteration}")


STEP_FOLD = StepFold.from_env()


def _gil_switch_us_from_env() -> Optional[float]:
    value = os.environ.get(GIL_SWITCH_US_ENV, "").strip()
    if not value or value == "0":
        return None
    us = float(value)
    if us <= 0:
        raise ValueError(f"{GIL_SWITCH_US_ENV} must be > 0, got {value!r}")
    return us


def apply_gil_switch_interval() -> Optional[float]:
    """Set ``sys.setswitchinterval`` from the shell's ``gil_switch_us``.

    Returns:
        The interval set, in seconds, or None when the shell keeps the interpreter's.

    Raises:
        ValueError: ``TRTLLM_K3_ULL_GIL_SWITCH_US`` (no model shell) is negative.
    """
    if _SHELL is not None:
        us, tag = _SHELL.gil_switch_us, _LOG
    else:
        us, tag = _gil_switch_us_from_env(), GIL_SWITCH_US_ENV
    if not us:
        return None
    sys.setswitchinterval(us * 1e-6)
    logger.info(f"{tag}: thread switch interval {sys.getswitchinterval() * 1e6:.0f} us")
    return sys.getswitchinterval()


def _pin_executor_requested() -> bool:
    """Whether the executor thread is pinned (the model's shell, else the environment).

    Raises:
        ValueError: ``TRTLLM_K3_ULL_PIN_EXECUTOR`` (no model shell) is set to something other
            than ``auto``, ``0`` or the empty string.
    """
    if _SHELL is not None:
        return _SHELL.pin_executor
    value = os.environ.get(PIN_EXECUTOR_ENV, "").strip()
    if not value or value == "0":
        return False
    if value != "auto":
        raise ValueError(f"{PIN_EXECUTOR_ENV} must be auto, got {value!r}")
    return True


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


def pin_executor_cpu(
    available: Optional[Set[int]] = None, gpu_index: Optional[int] = None
) -> Optional[int]:
    """The CPU the executor thread is pinned to, or None when the shell does not pin it.

    One physical core per GPU: the ranks of a node often share one NUMA-wide set, so the core
    is the ``gpu_index``-th of the set's physical cores (lowest CPU of each SMT group),
    skipping the core of CPU 0 (housekeeping interrupts).

    Args:
        available: The process's affinity set (``os.sched_getaffinity(0)`` by default).
        gpu_index: The node-local index of this process's GPU (default: from the current
            CUDA device and ``CUDA_VISIBLE_DEVICES``).

    Raises:
        ValueError: ``TRTLLM_K3_ULL_PIN_EXECUTOR`` (no model shell) has another value than
            ``auto``, ``0`` or the empty string.
    """
    if not _pin_executor_requested():
        return None
    if available is None:
        available = os.sched_getaffinity(0)
    if not available:
        return None
    if gpu_index is None:
        gpu_index = _physical_gpu_index()
    return _executor_core(available, gpu_index)


def _executor_core(available: Set[int], gpu_index: int) -> int:
    """The executor core of the rank driving node-local GPU ``gpu_index`` in ``available``.

    The ``gpu_index``-th physical core of the set (lowest CPU of each SMT group), skipping
    the core of CPU 0; every rank sharing the set computes the same answer for a given GPU.
    """
    cpus = sorted(available)
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
    return cores[gpu_index % len(cores)]


def _node_gpu_count() -> int:
    """The number of GPUs on the node (the driver's view, independent of CUDA_VISIBLE_DEVICES)."""
    try:
        count = len(os.listdir("/proc/driver/nvidia/gpus"))
    except OSError:
        count = 0
    if count:
        return count
    try:
        import torch

        return torch.cuda.device_count() if torch.cuda.is_available() else 1
    except (ImportError, RuntimeError):
        return 1


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


class ExecutorThreadPin:
    """Keeps the executor loop thread alone on one CPU (the shell's ``pin_executor``).

    The calling thread gets ``{cpu}``; every other thread of the process gets the process's
    CPU set without the executor cores of all node-local GPUs and their SMT siblings. The
    ranks of a node usually share one affinity set, so leaving out only this rank's core
    would let another rank's busy threads (for example a transfer-library progress thread)
    run on this rank's executor core, where the pinned thread cannot move away. Threads
    started later inherit the affinity of the thread that starts them, so
    :meth:`maybe_refresh` moves new threads off the reserved cores every ``refresh_iters``
    iterations (a scan of ``/proc/self/task``).

    Args:
        cpu: The CPU the calling thread is pinned to.
        available: The process's affinity set.
        refresh_iters: Iterations between two scans for new threads.
        local_gpus: The number of GPUs on the node whose executor cores are reserved
            (default: :func:`_node_gpu_count`).
    """

    def __init__(
        self,
        cpu: int,
        available: Set[int],
        refresh_iters: int = 512,
        local_gpus: Optional[int] = None,
    ):
        self.cpu = cpu
        self.tid = threading.get_native_id()
        reserved = _smt_siblings(cpu)
        if local_gpus is None:
            local_gpus = _node_gpu_count()
        for gpu_index in range(local_gpus):
            reserved |= _smt_siblings(_executor_core(available, gpu_index))
        self.others = set(available) - reserved
        if not self.others:
            # Every core is reserved: keep only this rank's core free.
            self.others = set(available) - _smt_siblings(cpu)
        if not self.others:
            # A one-core set: nothing to move the other threads to.
            self.others = set(available) - {cpu} or set(available)
        self.refresh_iters = max(1, refresh_iters)
        self._known: Set[int] = set()

    @classmethod
    def create(cls) -> Optional["ExecutorThreadPin"]:
        """Pin the calling thread when the shell asks for it; None otherwise or on failure."""
        tag = _LOG if _SHELL is not None else PIN_EXECUTOR_ENV
        if _SHELL is not None:
            if not _SHELL.pin_executor:
                return None
        elif os.environ.get(PIN_EXECUTOR_ENV, "").strip() in ("", "0"):
            return None
        try:
            available = os.sched_getaffinity(0)
            cpu = pin_executor_cpu(available)
            if cpu is None:
                return None
            pin = cls(cpu, available)
            os.sched_setaffinity(0, {cpu})
            pin.refresh()
        except (OSError, ValueError) as error:
            logger.warning(f"{tag}: executor thread not pinned ({error})")
            return None
        logger.info(
            f"{tag}: executor thread {pin.tid} pinned to CPU {cpu}; "
            f"{len(pin.others)} CPUs left for the other threads"
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
