# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The GEN worker's step shell of this target, declared to the engine as ``RuntimeHooks``.

The measured deployment ran the decode-step shell ``StepShell(step_prep, step_fold, pin_executor,
gil_switch_us=200)``. Each piece maps onto one engine hook (``make_runtime_hooks``):

* ``pin_executor`` and ``gil_switch_us``: :class:`StepShellEvents`. Before the overlap loop starts,
  the executor thread is pinned to one physical core of the process's CPU set, and the other threads
  of the process are moved off that core and its SMT siblings when their affinity permits it.
  New threads are checked every
  ``PIN_REFRESH_ITERS`` iterations (``on_iteration``). The interpreter's thread switch interval
  is set to ``GIL_SWITCH_US``.
* ``step_prep``: :class:`InputCommitPack`, the engine's step-input pack
  (``step_input_pack_factory``). The decode step's small uploads are staged into a pinned arena.
  One commit kernel writes them right after the input preparation (``issue_after_prepare``), reading
  the arena in place.
  The uploads are the positions, KV and prompt lengths, the Helix parameters, the sampling seeds and
  offsets, and the first-step input and draft ids.
* ``step_fold``: :class:`OverlapFixupFold` (``spec_step_fold``) folds the overlap scheduler's
  previous-step gathers into the same commit launch. :func:`fold_block_offsets_copy` does the same
  for the KV cache manager's block-offset copy; the manager calls it.

The kernels are the served ones (``kernels/step/``, ``kernels/SOURCE.json``). The executor pin
reads ``CUDA_VISIBLE_DEVICES`` to identify the node-local GPU.
"""

from __future__ import annotations

import os
import sys
import threading
from dataclasses import replace
from typing import Optional, Set

import torch

from tensorrt_llm._torch.pyexecutor import step_inputs
from tensorrt_llm._torch.pyexecutor.runtime_hooks import DEFAULT_RUNTIME_HOOKS, RuntimeHooks
from tensorrt_llm._torch.pyexecutor.step_inputs import StepInputPack
from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.logger import logger

_LOG = "K3 ULL step shell"

# The interpreter's thread switch interval while serving (sys.setswitchinterval; Python's default is 5000 us).
GIL_SWITCH_US = 200
# Iterations between two scans that move new threads off the executor's CPU.
PIN_REFRESH_ITERS = 512


def smt_siblings(cpu: int) -> Set[int]:
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


def physical_gpu_index() -> Optional[int]:
    """Return the node-local GPU index when it can be determined."""
    try:
        if not torch.cuda.is_available():
            return None
        device = torch.cuda.current_device()
    except RuntimeError:
        return None
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return device
    devices = visible.split(",")
    if device < len(devices):
        identifier = devices[device].strip()
        if identifier.isdecimal():
            return int(identifier)
    # A UUID mask may give each rank the same logical device ordinal.
    return None


def pin_executor_cpu(available: Set[int], gpu_index: Optional[int] = None) -> Optional[int]:
    """The CPU the executor thread is pinned to: the ``gpu_index``-th physical core of ``available``.

    A physical core is the lowest CPU of an SMT group; the core of CPU 0 (housekeeping
    interrupts) is skipped. Ranks that share one NUMA-wide set thus get one core per GPU.
    """
    cpus = sorted(available)
    if not cpus:
        return None
    if gpu_index is None:
        gpu_index = physical_gpu_index()
        if gpu_index is None:
            return None
    cores, seen = [], set()
    for cpu in cpus:
        if cpu in seen:
            continue
        siblings = smt_siblings(cpu)
        seen |= siblings
        if 0 not in siblings:
            cores.append(cpu)
    if not cores:
        return cpus[0]
    return cores[gpu_index % len(cores)]


class ExecutorThreadPin:
    """Reserves one CPU for the executor loop where other threads' affinity permits it.

    The calling thread gets ``{cpu}``. Other threads keep their own affinity, minus the reserved
    core's CPUs when that leaves an allowed CPU. Threads started later inherit their creator's
    affinity, so :meth:`maybe_refresh` checks new threads every ``refresh_iters`` iterations.
    """

    def __init__(self, cpu: int, refresh_iters: int = PIN_REFRESH_ITERS):
        self.cpu = cpu
        self.tid = threading.get_native_id()
        self._reserved_cpus = smt_siblings(cpu)
        self.refresh_iters = max(1, refresh_iters)
        self._known: Set[int] = set()

    @classmethod
    def create(cls) -> Optional["ExecutorThreadPin"]:
        """Pin the calling thread; None when no CPU can be chosen or the kernel refuses."""
        try:
            available = os.sched_getaffinity(0)
            cpu = pin_executor_cpu(available)
            if cpu is None:
                return None
            pin = cls(cpu)
            os.sched_setaffinity(0, {cpu})
            pin.refresh()
        except (OSError, ValueError) as error:
            logger.warning(f"{_LOG}: executor thread not pinned ({error})")
            return None
        logger.info(f"{_LOG}: executor thread {pin.tid} pinned to CPU {cpu}")
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
                current = os.sched_getaffinity(tid)
                remaining = current - self._reserved_cpus
                if remaining and remaining != current:
                    os.sched_setaffinity(tid, remaining)
                    moved += 1
            except OSError:
                # The thread exited, or its affinity is not ours to change.
                pass
            self._known.add(tid)
        self._known.intersection_update(tids)
        return moved

    def maybe_refresh(self, iteration: int) -> None:
        if iteration % self.refresh_iters == 0:
            self.refresh()


class StepFold:
    """Counts of the folded step-prep sites, logged every ``LOG_EVERY`` iterations.

    ``active`` is always ``enabled`` (the measured shell folds every step). ``verify``: how many
    folded block-offset copies to check against the standalone copy (0 while serving).
    """

    LOG_EVERY = 2048

    def __init__(self, enabled: bool, verify: int = 0):
        self.enabled = enabled
        self.active = enabled
        self.verify = verify
        self.iteration = 0
        self.counts: dict[str, int] = {}
        self.num_block_checks = 0

    def on_iteration(self, iteration: int) -> None:
        self.iteration = iteration
        if iteration and iteration % self.LOG_EVERY == 0 and self.counts:
            counts = ", ".join(f"{k} {v}" for k, v in sorted(self.counts.items()))
            logger.info(f"{_LOG}: step fold iter {iteration}: {counts}")

    def take_block_check(self) -> bool:
        """Whether to check this folded block-offset copy (the first ``verify`` ones)."""
        if self.num_block_checks >= self.verify:
            return False
        self.num_block_checks += 1
        return True

    def count(self, what: str) -> None:
        first = what not in self.counts
        self.counts[what] = self.counts.get(what, 0) + 1
        if first:
            logger.info(f"{_LOG}: step fold: first {what} at iteration {self.iteration}")


# Read by kernels/step/input_commit.py (the served module's STEP_FOLD).
STEP_FOLD = StepFold(enabled=True)


class StepShellEvents:
    """Executor events of the step shell (``ExecutorEvents``): executor pinning and the GIL switch
    interval before the overlap loop; :meth:`on_iteration`, the events' optional per-iteration call,
    rescans the threads and logs the fold counters."""

    def __init__(self) -> None:
        self.pin: Optional[ExecutorThreadPin] = None

    def after_warmup(self) -> None:
        return

    def before_overlap_loop(self) -> None:
        # Runs on the executor thread. ExecutorThreadPin.create logs and returns None on an OS refusal.
        self.pin = ExecutorThreadPin.create()
        sys.setswitchinterval(GIL_SWITCH_US * 1e-6)
        logger.info(f"{_LOG}: thread switch interval {sys.getswitchinterval() * 1e6:.0f} us")

    def at_shutdown(self) -> None:
        return

    def on_iteration(self, iteration: int) -> None:
        if self.pin is not None:
            self.pin.maybe_refresh(iteration)
        STEP_FOLD.on_iteration(iteration)


class InputCommitPack(StepInputPack):
    """The engine's step-input pack as the served input commit (``kernels/step/input_commit.py``).

    ``stage`` copies the host values into the commit's pinned arena; ``flush`` launches the one
    commit kernel on the current stream. The engine issues it right after the input preparation
    (``issue_after_prepare``), where served committed, and routes the metadata uploads
    (``step_inputs.copy_metadata_``) into it. Copies the commit cannot take (device sources,
    non-contiguous destinations, other dtypes) are declined, and ``step_inputs.copy_`` then issues
    what was staged and copies directly, in order.
    """

    issue_after_prepare = True

    def __init__(self, device: torch.device) -> None:
        from ...kernels.k3r_step.input_commit import (
            _TORCH_TO_NUMPY,
            InputCommit,
            warmup_input_commit_kernels,
        )

        # None of the stock pack's buffers or graph options: the commit owns its arenas.
        self.parallel_scatter = False
        self.in_graph = False
        self.early_copy = False
        self.prologue_op = None
        self.epoch = None
        self.open = False
        self.device = device
        self._dtypes = frozenset(_TORCH_TO_NUMPY)
        self.commit = InputCommit()
        # Compiled here: warmup and capture forwards never open the pack.
        warmup_input_commit_kernels(self.commit)

    def begin(self) -> None:
        if self.open:
            self.flush()
        self.commit.begin()
        self.open = True

    def stage(self, dst: torch.Tensor, src: torch.Tensor) -> bool:
        if (
            src.is_cuda
            or not dst.is_cuda
            or not dst.is_contiguous()
            or dst.numel() != src.numel()
            or dst.dtype not in self._dtypes
        ):
            return False
        self.commit.stage(dst, src)
        return True

    def set_overlap_fixup(self, previous_slots: torch.Tensor, **kwargs) -> None:
        """The commit's overlap fixup with the previous step's slots read from ``previous_slots``
        (the engine's int32 device buffer) instead of the arena."""
        n = previous_slots.numel()
        self.commit.set_overlap_fixup(previous_slots=[0] * n, **kwargs)
        self.commit._fixup["slots_device"] = previous_slots

    def flush(self, in_graph: bool = False) -> None:
        if not self.open:
            return
        self.open = False
        self.commit.commit()

    def discard(self) -> None:
        if self.open:
            self.commit.abort()
        self.open = False

    def capture_prologue(self) -> bool:
        return False

    def quiesce_graph_window(self) -> None:
        return

    def on_graph_replay(self) -> None:
        return


class OverlapFixupFold:
    """``SpecStepFold`` whose only step is the overlap scheduler's previous-step gathers, done by the
    open :class:`InputCommitPack`'s commit kernel. The forward stages nothing ahead and the sampler
    keeps its own stores (``begin_step``, ``sampler_rows_stored`` and ``scatter`` decline)."""

    def __init__(self) -> None:
        self.pack: Optional[InputCommitPack] = None
        self._buffers = None

    def bind_engine_buffers(
        self,
        input_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        position_id_offsets: torch.Tensor,
        kv_lens_offsets: torch.Tensor,
        max_batch_size: int,
    ) -> None:
        self._buffers = (input_ids, draft_tokens, position_id_offsets, kv_lens_offsets)

    def bind_sampler_store(self, store: object) -> None:
        return

    def begin_step(self, seq_slots: Optional[list[int]], num_rows: int) -> bool:
        return False

    def gather(
        self,
        new_tokens: torch.Tensor,
        next_draft_tokens: torch.Tensor,
        new_tokens_lens: torch.Tensor,
        previous_slots: torch.Tensor,
        previous_pos_indices: torch.Tensor,
        previous_batch_len: int,
        tokens_per_gen_step: int,
        draft_token_buffer_width: int,
        num_tokens: int,
        num_draft_tokens: int,
        first_previous_row: int,
    ) -> bool:
        from ...kernels.k3r_step.input_commit import overlap_fixup_supported

        pack = self.pack
        if (
            pack is None
            or not pack.open
            or not step_inputs.is_active(pack)
            or self._buffers is None
            or pack.commit._fixup is not None
        ):
            return False
        input_ids, draft_tokens, pos_offsets, kv_offsets = self._buffers
        slots = previous_slots[:previous_batch_len]
        if not (
            overlap_fixup_supported(new_tokens, new_tokens_lens, next_draft_tokens)
            and slots.is_cuda
            and slots.dtype == torch.int32
            and slots.is_contiguous()
            and all(
                t.is_cuda and t.dtype == torch.int32 and t.is_contiguous()
                for t in (input_ids, draft_tokens, pos_offsets, kv_offsets)
            )
            and previous_batch_len * tokens_per_gen_step <= input_ids.numel() - num_tokens
            and previous_batch_len * draft_token_buffer_width
            <= draft_tokens.numel() - num_draft_tokens
        ):
            return False
        pack.set_overlap_fixup(
            slots,
            tokens_per_req=tokens_per_gen_step,
            draft_width=draft_token_buffer_width,
            new_tokens=new_tokens,
            new_tokens_lens=new_tokens_lens,
            next_draft_tokens=next_draft_tokens,
            input_ids=input_ids,
            input_ids_start=num_tokens,
            draft_tokens=draft_tokens,
            draft_tokens_start=num_draft_tokens,
            pos_offsets=pos_offsets,
            pos_offsets_start=first_previous_row * tokens_per_gen_step,
            kv_offsets=kv_offsets,
            kv_offsets_start=first_previous_row,
        )
        return True

    def sampler_rows_stored(self, outputs: dict, num_skip: int, seq_slots: list[int]) -> bool:
        return False

    def scatter(
        self,
        seq_slots: torch.Tensor,
        new_tokens: torch.Tensor,
        next_new_tokens: torch.Tensor,
        new_tokens_lens: torch.Tensor,
        next_draft_tokens: torch.Tensor,
    ) -> bool:
        return False


# The pack of this process's GEN engine (one decoder runner per worker), for fold_block_offsets_copy.
_PACK: Optional[InputCommitPack] = None


def fold_block_offsets_copy(
    host_offsets: torch.Tensor,
    dst: torch.Tensor,
    copy_idx: torch.Tensor,
    index_scales: torch.Tensor,
    kv_offset: torch.Tensor,
    stream: Optional[torch.cuda.Stream] = None,
) -> bool:
    """Hand the KV cache manager's block-offset copy (``copy_batch_block_offsets_to_device``'s
    arguments) to the open commit, whose kernel then does it. Returns False, having done nothing,
    when no commit of this engine is open, the copy would run on another stream, its layout does not
    fit, or the commit already holds one; the caller then copies itself."""
    from ...kernels.k3r_step.step_upload_kernels import block_offsets_copy_job

    pack = _PACK
    if pack is None or not pack.open or not step_inputs.is_active(pack):
        return False
    if stream is not None and stream != torch.cuda.current_stream():
        STEP_FOLD.count("block_offsets_other_stream")
        return False
    job = block_offsets_copy_job(host_offsets, dst, copy_idx, index_scales, kv_offset)
    if job is None or not pack.commit.add_block_offsets(job):
        STEP_FOLD.count("block_offsets_not_folded")
        return False
    STEP_FOLD.count("block_offsets_folded")
    return True


def make_runtime_hooks(base: Optional[RuntimeHooks] = None) -> RuntimeHooks:
    """``base`` (the defaults when None) plus the step shell. Build it once per model: the engine
    binds its buffers to the returned fold and its pack to the module."""
    fold = OverlapFixupFold()
    events = StepShellEvents()

    def make_pack(device: torch.device) -> StepInputPack:
        global _PACK
        if not prefer_pinned():
            # The commit kernel reads its arena over the host link, which needs pinned memory.
            return StepInputPack(device)
        pack = InputCommitPack(device)
        fold.pack = pack
        _PACK = pack
        logger.info(f"{_LOG}: step prep is one input commit per decode step")
        return pack

    return replace(
        base or DEFAULT_RUNTIME_HOOKS,
        pack_step_inputs=True,
        pack_spec_rng_uploads=True,
        step_input_pack_factory=make_pack,
        spec_step_fold=fold,
        executor_events=events,
    )


__all__ = [
    "GIL_SWITCH_US",
    "PIN_REFRESH_ITERS",
    "STEP_FOLD",
    "ExecutorThreadPin",
    "InputCommitPack",
    "OverlapFixupFold",
    "StepFold",
    "StepShellEvents",
    "fold_block_offsets_copy",
    "make_runtime_hooks",
    "pin_executor_cpu",
    "smt_siblings",
]
