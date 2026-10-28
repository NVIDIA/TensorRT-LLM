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

``_set_memory_metadata`` labels subsequent allocations, but the label does
not live on the block. Snapshot blocks carry only ``address``, ``size``,
``requested_size``, ``state`` and ``frames``; the label is recorded on the
*allocation event* in ``device_traces`` as ``user_metadata``. Attributing
live memory therefore means replaying the trace to learn which address
belongs to which tag.

Three measured properties shape how that is done.

**The trace is a ring buffer, and weights are allocated first.** With
``max_entries=200`` and 2000 later allocations, a 128 MiB block tagged
``EARLY_WEIGHTS`` dropped out of the trace entirely and reappeared as 240 MiB
of untraced memory. The largest category is the one most likely to be
evicted, and nothing in the API announces it -- the tag simply stops
existing. So the address-to-tag map is **frozen once, while the trace is
still intact**, and reused afterwards.

**Recording costs about 2x per allocation.** 4000 allocate/free pairs took
6.8 ms with history off and 13.3 ms with it on. Since every tagged category
(weights, sampler, logits) is allocated during startup, recording is enabled
only for that window and switched off afterwards; measured steady-state cost
returns to baseline (6.7 ms vs 6.8 ms).

**Tags are silently dropped on non-native allocators.** ``_annotate_tensor``
documents that it "is only supported by the native caching allocator; with
other backends (e.g. ``cudaMallocAsync`` or a pluggable allocator) it is
silently ignored" -- which includes the pluggable allocator behind the sleep
feature's ``virtual_memory.scope``. Silence is the problem: the sums would
come out small and plausible. :func:`tagging_supported` refuses up front
instead.

Whatever is live but carries no tag is always reported as ``untagged``
rather than dropped. A tag breakdown that omits it cannot be checked against
``torch_live`` and quietly reads as complete.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field

import torch

from tensorrt_llm.logger import logger

__all__ = [
    "UNTAGGED",
    "claim_module_memory",
    "TagSnapshot",
    "freeze_tag_map",
    "get_frozen_map",
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

#: Generous enough that startup cannot wrap it; the buffer is discarded as
#: soon as the map is frozen, so the cost is bounded and temporary.
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

    A no-op when tagging is unsupported, so call sites need no guard.
    """
    if not tagging_supported():
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
        logger.debug(f"memory ledger: could not set tag {name!r}")
        yield
        return

    try:
        yield
    finally:
        try:
            memory._set_memory_metadata(previous)
        except Exception:  # noqa: BLE001
            logger.debug("memory ledger: could not restore the previous tag")


def start_tagging_window(max_entries: int = _DEFAULT_MAX_ENTRIES) -> bool:
    """Begin recording allocation history so tags land in the trace.

    Returns whether recording started. Keep the window to startup: this is
    what costs roughly 2x per allocation.
    """
    if not tagging_supported():
        return False
    try:
        torch.cuda.memory._record_memory_history(max_entries=max_entries)
    except Exception:  # noqa: BLE001
        logger.debug("memory ledger: could not start history recording")
        return False
    return True


def stop_tagging_window() -> None:
    """Stop recording. Safe to call when it was never started."""
    try:
        torch.cuda.memory._record_memory_history(None)
    except Exception:  # noqa: BLE001
        logger.debug("memory ledger: could not stop history recording")


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
        logger.debug("memory ledger: snapshot failed while freezing tags")
        return TagSnapshot(supported=False)

    owners: dict[int, str] = {}
    for device_trace in snapshot.get("device_traces") or []:
        for event in device_trace:
            action = event.get("action")
            address = event.get("addr")
            if address is None:
                continue
            if action in _ALLOC_ACTIONS:
                label = event.get("user_metadata") or ""
                if label:
                    owners[address] = label
                else:
                    # An untagged reuse must clear the previous owner, or the
                    # block would keep a tag it no longer belongs to.
                    owners.pop(address, None)
            elif action in _FREE_ACTIONS:
                owners.pop(address, None)

    return TagSnapshot(supported=True, owners=owners)


def live_bytes_by_tag(snapshot: TagSnapshot) -> dict[str, int]:
    """Split currently live torch memory across the frozen tags.

    The ``untagged`` entry is always present when anything is live, even at
    zero, so a caller cannot mistake a partial breakdown for a complete one.
    Returns an empty mapping when tagging is unsupported -- not a dict of
    zeros, which would read as a measurement.
    """
    if not snapshot.supported and not _claims:
        return {}

    try:
        segments = torch.cuda.memory._snapshot()["segments"]
    except Exception:  # noqa: BLE001
        logger.debug("memory ledger: snapshot failed while reading tags")
        return {}

    totals: dict[str, int] = {UNTAGGED: 0}
    for segment in segments:
        for block in segment["blocks"]:
            if block["state"] != "active_allocated":
                continue
            address = block["address"]
            label = _claims.get(address) or snapshot.owners.get(address, UNTAGGED)
            totals[label] = totals.get(label, 0) + block["size"]
    return totals


_frozen: TagSnapshot = TagSnapshot(supported=False)

#: Addresses claimed from a module's final parameters. Kept apart from the
#: trace-derived map because the window's closing freeze rebuilds that map
#: wholesale; merging them there would discard every claim made during
#: loading. Claims win on read: they describe the state the model ended up
#: in, while the trace describes what was asked for on the way there.
_claims: dict[int, str] = {}


def set_frozen_map(snapshot: TagSnapshot) -> None:
    """Publish the map captured at the end of the startup window."""
    global _frozen
    _frozen = snapshot


def get_frozen_map() -> TagSnapshot:
    """The published map, or an unsupported one if startup never ran."""
    return _frozen


@contextlib.contextmanager
def startup_window(rank: int | None = None):
    """Record tags for the duration of startup, then freeze and stop.

    Costs roughly 2x per allocation while open, so it is gated on the ledger
    being enabled and closes before steady state begins. Tagged categories
    are all allocated during startup, so nothing is lost by closing it.

    Freezing happens in ``finally``: a startup that raises part way through
    still publishes what it managed to tag, which is exactly the case where
    a memory report is worth having.
    """
    from . import enabled

    if not enabled(rank) or not start_tagging_window():
        yield False
        return

    try:
        yield True
    finally:
        try:
            snapshot = freeze_tag_map()
            set_frozen_map(snapshot)
            _log_window_result(snapshot)
        finally:
            stop_tagging_window()


def _log_window_result(snapshot: TagSnapshot) -> None:
    """Report what the window captured, including when it captured nothing.

    A startup tag that silently fails to record is the failure mode this
    whole module is built around, so the closing line states the totals
    rather than leaving the first sign of trouble to be a missing row in a
    report read much later.
    """
    if not snapshot.supported:
        logger.info("memory ledger: tagging unsupported; no startup tags")
        return

    totals = live_bytes_by_tag(snapshot)
    tagged = {k: v for k, v in totals.items() if k != UNTAGGED and v}
    if not tagged:
        logger.warning(
            "memory ledger: startup window closed with no tagged bytes; "
            "torch-side rows will be absent rather than zero"
        )
        return

    summary = ", ".join(
        f"{name}={size / (1 << 20):.1f} MiB" for name, size in sorted(tagged.items())
    )
    logger.info(f"memory ledger: startup tags captured -- {summary}")

    # The executor's process is the one holding the memory, and this is the
    # first moment every provider has something to report, so emit the full
    # breakdown here rather than leaving the startup tags as the only output
    # a run ever produces.
    from .triggers import report_now

    report = report_now(tag="startup_complete")
    if report is not None:
        logger.info(f"memory ledger: startup breakdown\n{report}")


def _block_index() -> list[tuple[int, int]]:
    """Sorted ``(address, size)`` of every live block, for containment search."""
    try:
        segments = torch.cuda.memory._snapshot()["segments"]
    except Exception:  # noqa: BLE001
        return []
    blocks = [
        (block["address"], block["size"])
        for segment in segments
        for block in segment["blocks"]
        if block["state"] == "active_allocated"
    ]
    blocks.sort()
    return blocks


def claim_module_memory(module, label: str) -> int:
    """Attribute a module's *current* parameters and buffers to ``label``.

    Tagging at the allocation site under-reports anything the loader later
    replaces. Measured on Qwen2-0.5B: the weight allocation region tagged
    944 MiB while the model held 1.17 GiB of parameters, because fusing and
    quantising weights frees the originally tagged blocks and allocates new,
    untagged ones. Reading the final parameters instead is immune to that,
    and needs no history recording.

    Each block in a snapshot is one allocation, so a storage maps to exactly
    one block and claiming is not an over-estimate. Returns the bytes
    claimed.
    """
    import bisect

    blocks = _block_index()
    if not blocks:
        return 0
    addresses = [address for address, _ in blocks]

    claimed: dict[int, int] = {}
    tensors = []
    get_params = getattr(module, "parameters", None)
    get_buffers = getattr(module, "buffers", None)
    if get_params is not None:
        tensors.extend(get_params())
    if get_buffers is not None:
        tensors.extend(get_buffers())

    for tensor in tensors:
        if not getattr(tensor, "is_cuda", False):
            continue
        try:
            pointer = tensor.untyped_storage().data_ptr()
        except Exception:  # noqa: BLE001
            continue
        position = bisect.bisect_right(addresses, pointer) - 1
        if position < 0:
            continue
        address, size = blocks[position]
        if pointer < address + size:
            claimed[address] = size

    if claimed:
        _claims.update({address: label for address in claimed})

    return sum(claimed.values())
