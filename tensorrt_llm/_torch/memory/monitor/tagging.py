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
live memory therefore means replaying the trace. Two measured properties
shape how:

* The trace is a ring buffer. With ``max_entries=200`` and 2000 later
  allocations, a 128 MiB tagged block dropped out of the trace entirely and
  reappeared as untraced memory, with nothing announcing it. The
  address-to-tag map is therefore **frozen once, while the trace is still
  intact**, and reused afterwards.
* Recording is only affordable without stacks. See
  :func:`start_tagging_window` for the numbers; recording is confined to
  startup and stopped before the startup report is built.

The label is per thread (measured: an allocation made by a worker thread
while the main thread held a tag carried none). Startup steps hand work to
thread pools -- weight loading, for one -- so each tag's time window is kept
too, and an allocation event that carries no label but falls inside a window
takes the innermost window's label. Trace events carry wall-clock
``time_us``, the same clock as ``time.time()`` (measured). Memory a
background thread allocates while a step runs is therefore named after that
step; an explicit label and a claimed module still take precedence.

Tags survive a ``torch.cuda.MemPool``, including the pluggable-allocator
pool behind ``virtual_memory.scope`` (measured). They do not survive a
process-wide non-native backend such as ``cudaMallocAsync``, where the
native caching allocator and its trace are not in use at all;
:func:`tagging_supported` refuses up front there rather than produce a
small, plausible, wrong number.

Whatever is live but carries no label is always reported as ``untagged``
rather than dropped. A breakdown that omits it cannot be checked against
``torch_live`` and quietly reads as complete.
"""

from __future__ import annotations

import bisect
import contextlib
import itertools
import time
import weakref
from dataclasses import dataclass, field

import torch

from tensorrt_llm.logger import logger

__all__ = [
    "CUDA_GRAPHS",
    "UNTAGGED",
    "claim_module_memory",
    "claimed_modules",
    "clear_claims",
    "TagSnapshot",
    "freeze_tag_map",
    "get_frozen_map",
    "live_attribution",
    "live_bytes_by_tag",
    "set_frozen_map",
    "start_tagging_window",
    "startup_window",
    "stop_tagging_window",
    "tag",
    "tagging_supported",
]

#: Label for live memory that no frozen tag claims. Always reported.
UNTAGGED = "untagged"

_ALLOC_ACTIONS = ("alloc", "segment_alloc")
_FREE_ACTIONS = ("free_completed", "segment_free")

#: Generous enough for a typical startup; a wrap is detected and reported.
#: The buffer is discarded as soon as the map is frozen, so the cost is
#: bounded and temporary.
_DEFAULT_MAX_ENTRIES = 1_000_000


def tagging_supported() -> bool:
    """Whether a tag set here would actually be recorded.

    Returns False rather than raising, but never reports a tag total of zero
    on an allocator that drops tags -- the caller is expected to render the
    whole section as unsupported.
    """
    memory = torch.cuda.memory
    probe = getattr(memory, "_memory_metadata_supported", None)
    if probe is None:
        return False
    try:
        if not probe():
            return False
    except RuntimeError:
        # A process-wide pluggable allocator answers by raising.
        return False
    if not hasattr(memory, "_set_memory_metadata"):
        return False
    if not hasattr(memory, "_record_memory_history"):
        return False
    # The docstring's own caveat: anything but the native caching allocator
    # accepts the call and discards the tag.
    return torch.cuda.get_allocator_backend() == "native"


@contextlib.contextmanager
def tag(name: str):
    """Label allocations made inside this block.

    Restores the previous label rather than clearing it, so nesting a
    narrower tag inside a broader one leaves the outer one in effect
    afterwards.

    A no-op when the monitor is off or tagging is unsupported, so call
    sites need no guard.
    """
    from . import enabled

    if not enabled() or not tagging_supported():
        yield
        return

    memory = torch.cuda.memory
    try:
        previous = memory._get_memory_metadata()
    except Exception:  # noqa: BLE001 - never break the annotated code
        previous = ""

    try:
        memory._set_memory_metadata(name)
    except Exception:  # noqa: BLE001
        logger.debug(f"memory monitor: could not set tag {name!r}")
        yield
        return

    start_us = time.time() * 1e6
    try:
        yield
    finally:
        _windows.append((start_us, time.time() * 1e6, name))
        try:
            memory._set_memory_metadata(previous)
        except Exception:  # noqa: BLE001
            logger.debug("memory monitor: could not restore the previous tag")


#: ``(start_us, end_us, label)`` of every tag applied in the current window.
_windows: list[tuple[float, float, str]] = []


def _window_lookup(windows):
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

    def lookup(t):
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
    way; the stacks are what make it expensive. Measured on 8000 events:
    a snapshot took 2643 ms with the defaults and 4.8 ms without stacks,
    and 4000 allocations took 16.3 ms against 9.2 ms (7.8 ms unrecorded).
    """
    if not tagging_supported():
        return False
    try:
        torch.cuda.memory._record_memory_history(context=None, max_entries=max_entries)
        _windows.clear()
    except Exception:  # noqa: BLE001
        logger.debug("memory monitor: could not start history recording")
        return False
    return True


def stop_tagging_window() -> None:
    """Stop recording. Safe to call when it was never started."""
    try:
        torch.cuda.memory._record_memory_history(None)
    except Exception:  # noqa: BLE001
        logger.debug("memory monitor: could not stop history recording")


@dataclass(frozen=True)
class TagSnapshot:
    """A frozen address-to-tag map, taken while the trace was intact.

    ``supported`` distinguishes "nothing was tagged" from "tagging could not
    work here"; collapsing the two would make an unsupported allocator look
    like a process that allocates nothing.
    """

    supported: bool
    owners: dict[int, str] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.supported


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
    except Exception:  # noqa: BLE001
        logger.debug("memory monitor: snapshot failed while freezing tags")
        return TagSnapshot(supported=False)

    owners: dict[int, str] = {}
    window_label = _window_lookup(list(_windows))
    for device_trace in snapshot.get("device_traces") or []:
        if len(device_trace) >= _DEFAULT_MAX_ENTRIES:
            # The ring buffer kept only the latest events; blocks allocated
            # before them carry no tag and are reported as unattributed.
            logger.warning(
                f"memory monitor: startup made more than {_DEFAULT_MAX_ENTRIES} allocation "
                "events; the earliest are not attributed to a startup step"
            )
        for event in device_trace:
            action = event.get("action")
            address = event.get("addr")
            if address is None:
                continue
            if action in _ALLOC_ACTIONS:
                label = event.get("user_metadata") or ""
                if not label and event.get("time_us") is not None:
                    label = window_label(event["time_us"]) or ""
                if label:
                    owners[address] = label
                else:
                    # An untagged reuse must clear the previous owner, or the
                    # block would keep a tag it no longer belongs to.
                    owners.pop(address, None)
            elif action in _FREE_ACTIONS:
                owners.pop(address, None)

    return TagSnapshot(supported=True, owners=owners)


#: Label for live blocks in a CUDA graph's private pool. Applied from the
#: pool, not from a tag: capture happens inside a creation step whose tag
#: would otherwise claim the graph's memory as that step's.
CUDA_GRAPHS = "CUDA_GRAPHS"


def _live_blocks(segments: list | None = None) -> list[tuple[int, int, bool, list]] | None:
    """Sorted ``(address, size, in_graph_pool, frames)`` of every live block, or ``None``.

    ``frames`` is the allocating stack when stack recording is on, else empty.
    ``segments`` is an allocator snapshot already taken; one is taken if not.
    """
    from .pools import classify_all

    if segments is None:
        try:
            segments = torch.cuda.memory._snapshot()["segments"]
        except Exception:  # noqa: BLE001
            logger.debug("memory monitor: snapshot failed while reading tags")
            return None
    kinds = classify_all(segment.get("segment_pool_id", (0, 0)) for segment in segments)
    blocks = []
    for segment in segments:
        in_graph = kinds[tuple(segment.get("segment_pool_id", (0, 0)))][0] == "graph"
        for block in segment["blocks"]:
            if block["state"] == "active_allocated":
                blocks.append(
                    (block["address"], block["size"], in_graph, block.get("frames") or [])
                )
    blocks.sort()
    return blocks


def live_bytes_by_tag(snapshot: TagSnapshot) -> dict[str, int]:
    """Split currently live torch memory across tags and claimed modules."""
    return live_attribution(snapshot)[0]


def live_attribution(
    snapshot: TagSnapshot, segments: list | None = None
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
    claims = _claimed_storages()
    blocks = _live_blocks(segments)
    if blocks is None:
        return {}, {}
    in_graphs = any(graph for _, _, graph, _ in blocks)
    with_stacks = any(frames for _, _, _, frames in blocks)
    if not snapshot.supported and not claims and not in_graphs and not with_stacks:
        return {}, {}
    addresses = [address for address, _, _, _ in blocks]

    # Each snapshot block is one allocation, so a storage lies inside exactly
    # one block, and storages sharing a block (tied weights) claim it once.
    claimed: dict[int, str] = {}
    for pointer, label in claims:
        position = bisect.bisect_right(addresses, pointer) - 1
        if position >= 0:
            address, size, _, _ = blocks[position]
            if pointer < address + size:
                claimed.setdefault(address, label)

    # Precedence: a claimed module, then the graph pool a block sits in, then
    # the creation step's tag.
    from .stacks import site_of

    totals: dict[str, int] = {UNTAGGED: 0}
    # What is left untagged, by the code that allocated it -- populated only
    # when stack recording is on, since blocks carry frames only then.
    untagged_sites: dict[str, int] = {}
    for address, size, in_graph, frames in blocks:
        label = claimed.get(address)
        if label is None:
            label = CUDA_GRAPHS if in_graph else snapshot.owners.get(address, UNTAGGED)
        totals[label] = totals.get(label, 0) + size
        if label == UNTAGGED:
            site = site_of(frames)
            if site is not None:
                untagged_sites[site] = untagged_sites.get(site, 0) + size
    return totals, untagged_sites


_frozen: TagSnapshot = TagSnapshot(supported=False)

#: Modules whose parameters and buffers belong to a label, held weakly.
#: Resolved to storages each time a report is built rather than once at
#: load: an address recorded once keeps its label after the block is freed
#: and reused by an unrelated allocation, while walking the module sees only
#: the storage it holds now -- including after sleep/wake or a weight
#: update has replaced it.
_claimed_modules: list[tuple["weakref.ref", str]] = []


def set_frozen_map(snapshot: TagSnapshot) -> None:
    """Publish the map captured at the end of the startup window."""
    global _frozen
    _frozen = snapshot


def get_frozen_map() -> TagSnapshot:
    """The published map, or an unsupported one if startup never ran."""
    return _frozen


def claim_module_memory(module, label: str) -> bool:
    """Attribute ``module``'s parameters and buffers to ``label``.

    Needs no history recording and does no work beyond taking a weak
    reference; the storages are resolved when a report is built. Recorded
    whether or not the monitor is on, so a report asked for later
    (``LLM.get_memory_report``) still names the weights; holding a weak
    reference costs nothing. Returns whether the claim was recorded.
    """
    try:
        reference = weakref.ref(module)
    except TypeError:
        logger.debug(f"memory monitor: cannot weakly reference {type(module).__name__}")
        return False
    _claimed_modules[:] = [
        (ref, name) for ref, name in _claimed_modules if ref() is not None and ref() is not module
    ]
    _claimed_modules.append((reference, label))
    return True


def claimed_modules() -> list:
    """Claimed modules that are still alive.

    Lets a provider whose objects hang off the model -- with no registry of
    their own -- walk it without holding it.
    """
    return [module for module in (ref() for ref, _ in list(_claimed_modules)) if module is not None]


def clear_claims() -> None:
    """Forget every claimed module. For tests."""
    _claimed_modules.clear()


def _claimed_storages() -> list[tuple[int, str]]:
    """``(storage pointer, label)`` for every CUDA tensor a claimed module holds now."""
    found = []
    for reference, label in list(_claimed_modules):
        module = reference()
        if module is None:
            continue
        try:
            tensors = itertools.chain(module.parameters(), module.buffers())
            for tensor in tensors:
                if tensor.is_cuda:
                    found.append((tensor.untyped_storage().data_ptr(), label))
        except Exception:  # noqa: BLE001 - a report must not break on one module
            logger.debug(f"memory monitor: could not walk the {label!r} module")
    return found


@contextlib.contextmanager
def startup_window(rank: int | None = None):
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
    from . import clear_baseline, enabled, startup

    if not enabled(rank):
        yield False
        return

    # A process can build more than one executor; what the previous one
    # recorded would otherwise be reported as this one's.
    startup.clear()
    clear_baseline()
    recording = start_tagging_window()
    failed = interrupted = False
    try:
        yield recording
    except Exception:
        failed = True
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
        except Exception as exc:  # noqa: BLE001 - must not replace startup's own error
            logger.warning(
                f"memory monitor: could not read startup tags ({type(exc).__name__}: {exc})"
            )
        finally:
            if recording:
                from .stacks import stacks_enabled, start_stack_recording

                # Either way the event history ends here; stack mode keeps
                # only each live block's allocating stack from now on.
                if not (stacks_enabled() and start_stack_recording()):
                    stop_tagging_window()
        set_frozen_map(snapshot)
        if not interrupted:
            _log_window_result(snapshot, rank, failed=failed)


def _log_window_result(
    snapshot: TagSnapshot, rank: int | None = None, failed: bool = False
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
    except Exception as exc:  # noqa: BLE001 - never fail startup over a log line
        logger.debug(f"memory monitor: startup summary failed ({type(exc).__name__}: {exc})")

    from . import startup
    from .triggers import report_now

    tag_name = "startup_failed" if failed else "startup_complete"
    headline, suggestion = "", ""
    step = startup.failed_stage() if failed else None
    if step is not None:
        headline = (
            f"STARTUP FAILED while creating: {step.name} "
            f"(free device memory before it: {step.free_before / (1 << 30):.2f} GiB)."
        )
        suggestion = startup.advice_for_stage(step.stage) or ""
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
