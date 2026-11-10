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
"""Attributing torch-visible memory to the code that asked for it.

There are two ways in, and they suit different memory.

**Claiming a module.** :func:`claim_module_memory` records a weak reference
to a module; when a report is built, the module's current parameters and
buffers are mapped onto the allocator's live blocks. This is how weights
are attributed. It needs no history recording, costs nothing until a report
is built, and follows the module through anything that replaces its
storage -- fused or quantised weights, a weight update, sleep and wake.

**Tagging allocations.** :func:`tag` labels allocations made inside a
block, for memory that no module owns. ``_set_memory_metadata`` does not put
the label on the block: snapshot blocks carry only ``address``, ``size``,
``requested_size``, ``state`` and ``frames``, and the label lives on the
*allocation event* in ``device_traces`` as ``user_metadata``. Attributing
live memory therefore means replaying the trace, and two properties of the
trace shape how:

* The trace is a ring buffer: once it wraps, the earliest events are gone
  and the blocks they allocated carry no label. The address-to-tag map is
  therefore **frozen once, while the trace is still intact**, and reused
  afterwards. Each frozen owner keeps the size of the block it labelled, and
  a live block is labelled only while address and size both still match, so
  a block allocated after the freeze at a freed block's address is not
  mistaken for it.
* Recording stack-free events is cheap; recording stacks is not. Recording
  is confined to startup and stopped before the startup report is built.

The label is per thread: an allocation made by a worker thread while the
main thread holds a tag carries none. Startup steps hand work to thread
pools -- weight loading, for one -- so each tag's time window is kept too,
and an allocation event that carries no label but falls inside a window
takes the innermost window's label. Trace events carry wall-clock
``time_us``, the same clock as ``time.time()``. Memory a background thread
allocates while a step runs is therefore named after that step; an explicit
label and a claimed module still take precedence.

Tags survive a ``torch.cuda.MemPool``, including the pluggable-allocator
pool behind ``virtual_memory.scope``. They do not survive a process-wide
non-native backend such as ``cudaMallocAsync``, where the native caching
allocator and its trace are not in use at all; :func:`tagging_supported`
refuses up front there rather than produce a small, plausible, wrong number.

A history recording that someone else started -- a user or a framework
calling ``_record_memory_history`` before building the executor -- is left
alone: tagging is skipped for that startup rather than replacing it.

Whatever is live but carries no label is always reported as ``untagged``
rather than dropped. A breakdown that omits it cannot be checked against
``torch_live`` and quietly reads as complete.
"""

from __future__ import annotations

import bisect
import contextlib
import itertools
import time
from dataclasses import dataclass, field
from typing import Callable, Iterator

import torch

from tensorrt_llm.logger import logger

from . import providers, startup
from .capability import tagging_supported
from .claims import claim_module_memory, claimed_modules, claimed_storages, clear_claims
from .names import CUDA_GRAPHS, UNTAGGED
from .pools import classify_all, is_released
from .settings import full_mode
from .stacks import site_of, stacks_enabled, start_stack_recording

__all__ = [
    "CUDA_GRAPHS",
    "UNTAGGED",
    "claim_module_memory",
    "claimed_modules",
    "clear_claims",
    "forget_startup_of",
    "TagSnapshot",
    "freeze_tag_map",
    "get_frozen_map",
    "history_recording_active",
    "live_attribution",
    "live_bytes_by_tag",
    "set_frozen_map",
    "start_tagging_window",
    "startup_window",
    "stop_tagging_window",
    "tag",
    "tagging_supported",
]

_FREE_ACTIONS = ("free_completed", "segment_free")

#: Generous enough for a typical startup; a wrap is detected and reported.
#: The buffer is discarded as soon as the map is frozen, so the cost is
#: bounded and temporary.
_DEFAULT_MAX_ENTRIES = 1_000_000

#: Ring size of the recording in progress, so a wrap is judged against it.
_max_entries: int = _DEFAULT_MAX_ENTRIES


def history_recording_active() -> bool:
    """Whether PyTorch is recording allocator history right now, by anyone.

    Asks only once CUDA is initialised: the query reads allocator state that
    does not exist before then, and nothing can be recording before it.
    """
    probe = getattr(torch._C, "_cuda_isHistoryEnabled", None)
    if probe is None or not torch.cuda.is_initialized():
        return False
    return bool(probe())


@contextlib.contextmanager
def tag(name: str) -> Iterator[None]:
    """Label allocations made inside this block.

    Restores the previous label rather than clearing it, so nesting a
    narrower tag inside a broader one leaves the outer one in effect
    afterwards.

    A no-op when the monitor is off or tagging is unsupported, so call
    sites need no guard.
    """
    if not full_mode() or not tagging_supported():
        yield
        return

    memory = torch.cuda.memory
    try:
        previous = memory._get_memory_metadata()
    except RuntimeError:
        previous = ""

    try:
        memory._set_memory_metadata(name)
        applied = True
    except RuntimeError:
        logger.debug(f"memory monitor: could not set tag {name!r}")
        applied = False
    # Yielded outside the handler, so an exception from the tagged code does
    # not carry the monitor's error as its context.
    if not applied:
        yield
        return

    start_us = time.time() * 1e6
    try:
        yield
    finally:
        _windows.append((start_us, time.time() * 1e6, name))
        try:
            memory._set_memory_metadata(previous)
        except RuntimeError:
            logger.debug("memory monitor: could not restore the previous tag")


#: ``(start_us, end_us, label)`` of every tag applied in the current window.
_windows: list[tuple[float, float, str]] = []


def _window_lookup(
    windows: list[tuple[float, float, str]],
) -> Callable[[float], str | None]:
    """Map a timestamp to the innermost window containing it, or ``None``.

    Flattened once into sorted boundaries, so each lookup is a bisection.
    Windows nest (a tag inside a tag), never partially overlap, so the
    innermost is the one that started last.
    """
    if not windows:
        return lambda _t: None
    bounds = sorted({t for start, end, _ in windows for t in (start, end)})
    labels = []
    for left, right in itertools.pairwise(bounds):
        middle = (left + right) / 2
        inside = [w for w in windows if w[0] <= middle < w[1]]
        labels.append(max(inside)[2] if inside else None)

    def lookup(t: float) -> str | None:
        position = bisect.bisect_right(bounds, t) - 1
        if 0 <= position < len(labels):
            return labels[position]
        return None

    return lookup


def start_tagging_window(max_entries: int = _DEFAULT_MAX_ENTRIES) -> bool:
    """Begin recording allocation history so tags land in the trace.

    Returns whether recording started. Keep the window to startup.

    ``context=None`` records the allocate/free events and their
    ``user_metadata`` without a stack per event. The tag survives either
    way; per-event stacks are what make recording and snapshots expensive.

    Does nothing, and returns False, while a recording someone else started
    is active: replacing it would discard their trace.
    """
    global _max_entries
    if not tagging_supported():
        return False
    if history_recording_active():
        logger.info(
            "memory monitor: allocator history is already being recorded; leaving it alone "
            "and skipping startup tagging, so PyTorch rows are not split by startup step"
        )
        return False
    try:
        torch.cuda.memory._record_memory_history(context=None, max_entries=max_entries)
    except RuntimeError as exc:
        logger.debug(f"memory monitor: could not start history recording ({exc})")
        return False
    _windows.clear()
    _max_entries = max_entries
    return True


def stop_tagging_window() -> None:
    """Stop recording and discard the trace. Safe to call when it was never started."""
    try:
        torch.cuda.memory._record_memory_history(None)
    except RuntimeError as exc:
        logger.debug(f"memory monitor: could not stop history recording ({exc})")


@dataclass(frozen=True)
class TagSnapshot:
    """A frozen address-to-tag map, taken while the trace was intact.

    ``supported`` distinguishes "nothing was tagged" from "tagging could not
    work here"; collapsing the two would make an unsupported allocator look
    like a process that allocates nothing.
    """

    supported: bool
    owners: dict[int, str] = field(default_factory=dict)
    #: Requested size of the allocation each owner labelled. A live block at
    #: an owner's address takes the label only if its requested size matches;
    #: an owner without a recorded size matches any size.
    sizes: dict[int, int] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.supported

    def owner_of(self, address: int, size: int) -> str | None:
        """The label of the block at ``address``, if it is still that block."""
        label = self.owners.get(address)
        if label is None:
            return None
        expected = self.sizes.get(address)
        if expected is not None and expected != size:
            return None
        return label


def freeze_tag_map() -> TagSnapshot:
    """Capture address-to-tag ownership from the current trace.

    Call once, at the end of the tagging window and before the ring buffer
    can wrap. Replays allocate/free events in order so that an address which
    was reused carries its most recent tag rather than the first one.
    """
    if not tagging_supported():
        return TagSnapshot(supported=False)

    try:
        snapshot = torch.cuda.memory._snapshot()
    except RuntimeError as exc:
        logger.debug(f"memory monitor: snapshot failed while freezing tags ({exc})")
        return TagSnapshot(supported=False)

    owners: dict[int, str] = {}
    sizes: dict[int, int] = {}
    window_label = _window_lookup(list(_windows))
    for device_trace in snapshot.get("device_traces") or []:
        if len(device_trace) >= _max_entries:
            # The ring buffer kept only the latest events; blocks allocated
            # before them carry no tag and are reported as unattributed.
            logger.warning(
                f"memory monitor: startup made more than {_max_entries} allocation "
                "events; the earliest are not attributed to a startup step"
            )
        for event in device_trace:
            action = event.get("action")
            address = event.get("addr")
            if address is None:
                continue
            if action == "alloc":
                label = event.get("user_metadata") or ""
                if not label and event.get("time_us") is not None:
                    label = window_label(event["time_us"]) or ""
                if label:
                    owners[address] = label
                    sizes[address] = event.get("size")
                else:
                    # An untagged reuse must clear the previous owner, or the
                    # block would keep a tag it no longer belongs to.
                    owners.pop(address, None)
                    sizes.pop(address, None)
            elif action in _FREE_ACTIONS:
                owners.pop(address, None)
                sizes.pop(address, None)

    return TagSnapshot(
        supported=True,
        owners=owners,
        sizes={address: size for address, size in sizes.items() if size is not None},
    )


def _live_blocks(
    segments: list[dict] | None = None,
) -> list[tuple[int, int, bool, list, int]] | None:
    """Sorted ``(address, size, in_graph_pool, frames, requested)`` of every live block.

    ``None`` if the allocator could not be read. ``frames`` is the allocating
    stack when stack recording is on, else empty. ``requested`` is the size
    the allocation asked for, which is what trace events record.
    ``segments`` is an allocator snapshot already taken; one is taken if not.
    """
    if segments is None:
        try:
            segments = torch.cuda.memory._snapshot()["segments"]
        except RuntimeError as exc:
            logger.debug(f"memory monitor: snapshot failed while reading tags ({exc})")
            return None
    kinds = classify_all(segment.get("segment_pool_id", (0, 0)) for segment in segments)
    blocks = []
    for segment in segments:
        pool_id = tuple(segment.get("segment_pool_id", (0, 0)))
        if is_released(pool_id):
            continue  # asleep: not resident, so not attributed
        in_graph = kinds[pool_id][0] == "graph"
        for block in segment["blocks"]:
            if block["state"] == "active_allocated":
                blocks.append(
                    (
                        block["address"],
                        block["size"],
                        in_graph,
                        block.get("frames") or [],
                        block.get("requested_size", block["size"]),
                    )
                )
    blocks.sort()
    return blocks


def live_bytes_by_tag(snapshot: TagSnapshot) -> dict[str, int]:
    """Split currently live torch memory across tags and claimed modules."""
    return live_attribution(snapshot)[0]


def live_attribution(
    snapshot: TagSnapshot, segments: list[dict] | None = None
) -> tuple[dict[str, int], dict[str, int]]:
    """Live torch memory by label, and the untagged part by allocating code.

    Labels come from tags and claimed modules.

    A block held by a claimed module is labelled from the module; otherwise
    the frozen trace map applies. The ``untagged`` entry is always present
    when anything is live, even at zero, so a caller cannot mistake a
    partial breakdown for a complete one. Returns an empty mapping when
    neither source has anything -- not a dict of zeros, which would read as
    a measurement. ``segments`` is an allocator snapshot the caller already
    holds, so the tags describe the same moment as its other figures.
    """
    claims = claimed_storages()
    blocks = _live_blocks(segments)
    if blocks is None:
        return {}, {}
    in_graphs = any(block[2] for block in blocks)
    with_stacks = any(block[3] for block in blocks)
    if not snapshot.supported and not claims and not in_graphs and not with_stacks:
        return {}, {}
    addresses = [block[0] for block in blocks]

    # Each snapshot block is one allocation, so a storage lies inside exactly
    # one block, and storages sharing a block (tied weights) claim it once.
    claimed: dict[int, str] = {}
    for pointer, label in claims:
        position = bisect.bisect_right(addresses, pointer) - 1
        if position >= 0:
            address, size = blocks[position][:2]
            if pointer < address + size:
                claimed.setdefault(address, label)

    # Precedence: a claimed module, then the graph pool a block sits in, then
    # the creation step's tag.
    totals: dict[str, int] = {UNTAGGED: 0}
    # What is left untagged, by the code that allocated it -- populated only
    # when stack recording is on, since blocks carry frames only then.
    untagged_sites: dict[str, int] = {}
    for address, size, in_graph, frames, requested in blocks:
        label = claimed.get(address)
        if label is None:
            label = CUDA_GRAPHS if in_graph else (snapshot.owner_of(address, requested) or UNTAGGED)
        totals[label] = totals.get(label, 0) + size
        if label == UNTAGGED:
            site = site_of(frames)
            if site is not None:
                untagged_sites[site] = untagged_sites.get(site, 0) + size
    return totals, untagged_sites


_frozen: TagSnapshot = TagSnapshot(supported=False)


def set_frozen_map(snapshot: TagSnapshot) -> None:
    """Publish the map captured at the end of the startup window."""
    global _frozen
    _frozen = snapshot


def get_frozen_map() -> TagSnapshot:
    """The published map, or an unsupported one if startup never ran."""
    return _frozen


#: The executor whose startup the process-wide startup records describe.
_startup_executor: object | None = None


def forget_startup_of(token: object) -> None:
    """Drop the startup records if they describe the executor ``token``.

    Called when that executor shuts down, so a later report does not
    describe the startup of an executor that no longer exists.
    """
    global _startup_executor
    if token is None or token is not _startup_executor:
        return
    _startup_executor = None
    startup.clear()
    set_frozen_map(TagSnapshot(supported=False))


@contextlib.contextmanager
def startup_window(rank: int | None = None) -> Iterator[bool]:
    """Record tags for the duration of startup, then freeze, stop and report.

    The report is emitted whenever the monitor is on, whether or not tagging
    works here: on an allocator that drops tags the torch-side rows are
    missing, but the KV cache and every other component still belong in the
    startup breakdown.

    Recording stops before the report is built, so the snapshots the report
    takes are not paying for history. Freezing happens in ``finally``: a
    startup that raises part way through still publishes what it managed to
    tag and still reports, which is exactly when a breakdown is worth having.
    """
    global _startup_executor
    if not full_mode(rank):
        yield False
        return
    if providers.other_live_executors():
        # Startup records -- steps, regions, the warmup baseline and the
        # frozen tags -- are per process. Recording this executor's startup
        # would overwrite those of the executor still serving, whose reports
        # would then compare against the wrong warmup and lose the names of
        # its buffers.
        logger.warning(
            "memory monitor: another executor is running in this process; this "
            "executor's startup is not recorded, and reports keep describing the "
            "running executor's startup"
        )
        yield False
        return

    # A process can build more than one executor in turn; what the previous
    # one recorded would otherwise be reported as this one's.
    startup.clear()
    _startup_executor = providers.current_executor()
    recording = start_tagging_window()
    failed = interrupted = out_of_memory = False
    try:
        yield recording
    except Exception as exc:
        failed = True
        out_of_memory = _is_out_of_memory(exc)
        raise
    except BaseException:
        # Ctrl-C or exit: nobody is waiting for a memory report.
        interrupted = True
        raise
    finally:
        snapshot = TagSnapshot(supported=False)
        try:
            if recording:
                snapshot = freeze_tag_map()
        except Exception as exc:  # diagnostic boundary: must not replace startup's own error
            logger.warning(
                f"memory monitor: could not read startup tags ({type(exc).__name__}: {exc})"
            )
        finally:
            if recording:
                # Either way the event history ends here; stack mode keeps
                # only each live block's allocating stack from now on.
                if not (stacks_enabled() and start_stack_recording()):
                    stop_tagging_window()
        set_frozen_map(snapshot)
        if not interrupted:
            _log_window_result(snapshot, rank, failed=failed, out_of_memory=out_of_memory)


#: Messages of failures that mean the device ran out of memory: CUDA's and
#: PyTorch's, the executor's own explanation of a failed creation step, and
#: the NCCL form a failed communicator allocation takes.
_OOM_MESSAGES = ("out of memory", "insufficient gpu memory", "unhandled cuda error")


def _is_out_of_memory(exc: BaseException) -> bool:
    """Whether ``exc``, or an exception it was raised from, is a device OOM."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, torch.OutOfMemoryError):
            return True
        message = str(current).lower()
        if any(text in message for text in _OOM_MESSAGES):
            return True
        current = current.__cause__ or current.__context__
    return False


def _log_window_result(
    snapshot: TagSnapshot,
    rank: int | None = None,
    failed: bool = False,
    out_of_memory: bool = False,
) -> None:
    """State what startup attributed, then emit the full startup breakdown.

    The closing line names the totals -- or says that nothing was
    attributed -- rather than leaving the first sign of trouble to be a
    missing row in a report read much later.
    """
    try:
        totals = live_bytes_by_tag(snapshot)
        tagged = {k: v for k, v in totals.items() if k != UNTAGGED and v}
        if tagged:
            summary = ", ".join(
                f"{name}={size / (1 << 20):.1f} MiB" for name, size in sorted(tagged.items())
            )
            logger.info(f"memory monitor: startup attribution -- {summary}")
        elif not snapshot.supported:
            logger.info("memory monitor: tagging unsupported here; torch-side rows will be absent")
        else:
            logger.warning(
                "memory monitor: startup attributed no bytes; torch-side rows "
                "will be absent rather than zero"
            )
    except Exception as exc:  # diagnostic boundary: never fail startup over a log line
        logger.debug(f"memory monitor: startup summary failed ({type(exc).__name__}: {exc})")

    # Imported here: ``triggers`` builds reports through ``capture``, which
    # reads this module's tags.
    from .triggers import report_now

    tag_name = "startup_failed" if failed else "startup_complete"
    headline, suggestion = "", ""
    step = startup.failed_stage() if failed else None
    if step is not None:
        headline = (
            f"STARTUP FAILED while creating: {step.name} "
            f"(free device memory before it: {step.free_before / (1 << 30):.2f} GiB)."
        )
        # A step's tuning advice is about its memory; any other failure
        # (a missing checkpoint key, say) gets none.
        suggestion = (startup.advice_for_stage(step.stage) or "") if out_of_memory else ""
    elif failed:
        headline = "STARTUP FAILED."
    # A completed startup, warmup included, is the baseline later reports
    # compare against: every static allocation exists by now, and warmup
    # has already run the largest shapes, so later growth is the workload's.
    report = report_now(
        tag=tag_name,
        rank=rank,
        record_baseline=not failed,
        headline=headline,
        suggestion=suggestion,
        startup_steps=True,
    )
    if report is not None:
        log = logger.warning if failed else logger.info
        log(f"memory monitor: {tag_name.replace('_', ' ')} breakdown\n{report}")
