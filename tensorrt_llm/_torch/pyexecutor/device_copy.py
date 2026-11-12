# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A step's small copies into graph-stable device buffers, merged as the model's copy plan says.

Between two decode CUDA graph replays, the next step's inputs and metadata reach their device buffers as a chain of tiny
copies (positions, KV lengths, page tables, the graph runner's static inputs). Each one is a separate operation on the
stream the next graph waits for, so the GPU idles for the whole chain.

Every such copy in the runtime names its call site with a ``CopySite`` and goes through ``copy`` (or ``fill_tail``).
Two questions decide what happens to it, and two different owners answer them:

* **When** a copy must land is the copy plan's business. It depends on the destination's readers, which the model
  chose, so the model declares a ``CopyPlan`` (``copy_plan`` on the model):

  - ``merge``: the sites whose copies are staged into the step's pack. A host-to-device copy snapshots its host bytes
    into pinned staging and a device-to-device copy records its (dst, src); a flush issues one upload of
    [descriptors | data] and one ``trtllm::device_copy_scatter`` launch, which writes every staged host copy and then
    every device copy. A merged copy lands before the next barrier, and at the latest at the graph replay, the step's
    final barrier. A pack is opened only on a step that replays an already captured graph, so it always reaches it.
  - ``flush_after`` (a subset of ``merge``): barrier sites. When a call through one of them stages its copy, the pack
    is flushed right after it, so that copy has landed before anything that runs next. Only the anchor's own write
    is guaranteed: a reader that needs other merged data anchors on those sites too.

  - ``parallel_scatter`` and ``final_flush=InGraph(...)``: how a segment is issued. With ``parallel_scatter`` an
    order-free segment (see ``_parallel_ok``) is scattered with one program per copy. With an ``InGraph`` final flush
    every decode graph is captured with a prologue that issues a pinned window, and the final flush before a replay
    only fills that window, so no operation is left between the decode graphs. Barrier segments stay eager: only the
    last segment rides the window, and only when it fits and is order-free.

  Listing a site in ``merge`` asserts that between the call and its barrier no device work reads or writes the
  destination, nor, for a device-to-device copy, writes the source (which is read at the flush). A site in neither
  set, and every site of an engine whose model declares no plan, is ``dst.copy_(src, non_blocking=True)``. An eager
  copy or fill issued through this module while copies are staged issues them first, so the copies land in call order
  whichever sites the plan merges; only device work outside this module needs a ``flush_after`` anchor.
* **Whether** a copy is needed at all is the buffer owner's business. It depends on the values and on who else writes
  the buffer, which only the owner knows: the owner passes ``unchanged=True`` when the destination already holds what
  the call would write, and the call is not issued. ``DeviceCopyBuffer`` is that owner-side record for one buffer and
  one site: ``update`` compares the host values with what its last issued copy wrote, issues the copy through its site
  otherwise, records only a copy that was issued, and passes ``on_drop`` so a merged copy that never lands (its step
  failed and the pack was discarded) invalidates the record. ``unchanged=True`` is refused during CUDA graph capture:
  a captured copy reruns on every replay.

Model-specific backends name their own sites with their own ``CopySiteEnum``. Adding a model: declare the plan from
the readers of each destination (merge what no device work touches before the replay, and anchor a barrier on each
copy that has an eager reader), and let each buffer owner decide on its own whether its uploads may be skipped (a
``DeviceCopyBuffer`` with the copy's site). Worked example: MiniMax-M3. Its MSA attention backend names its sites
(``MsaCopySite`` in ``attention/backends/sparse/minimax_m3/common.py``, each with the readers of its destination) and
skips its unchanged uploads through ``DeviceCopyBuffer``; each modeling_v2 target declares ``_COPY_PLAN`` in its
``modeling.py`` (``minimax_m3_nvfp4__sm_103__tp4`` merges plain copies; ``minimax_m3_nvfp4__sm_107__tp8`` adds
``changed_span``, ``parallel_scatter`` and an ``InGraph`` final flush) and sets it as ``copy_plan``.

``CHECK_MODE`` verifies the assertions at runtime, at the cost of a device synchronization per check: a write of a
merged destination before its flush, and an ``unchanged`` copy or fill (including every copy a ``DeviceCopyBuffer``
skips) whose destination does not hold the values. A read of a merged destination before its barrier is caught only
where the reader asserts it with ``expect_landed`` under ``if device_copy.CHECK_MODE:``; elsewhere an early reader sees
the poison bytes instead and fails downstream. It is read once at import from ``TLLM_DEVICE_COPY_CHECK=1``, or from
``TLLM_DEBUG_MODE=1``, which turned on the verification of ``DeviceCopyBuffer``'s skipped copies before the two were
one switch.

Each engine with a plan owns one ``DeviceCopier`` and makes it the active one for its forward (``active``); the module
functions act on the calling context's copier, so engines on other threads or devices never share one. The attention
and speculative metadata that copy have no engine to ask, which is why the copier travels in the context.
"""

from __future__ import annotations

import contextlib
import os
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterator, Optional, Protocol

import torch
import triton
import triton.language as tl

from tensorrt_llm._utils import prefer_pinned

CHECK_ENV = "TLLM_DEVICE_COPY_CHECK"
#: Whether the copies' assertions are verified at runtime (``TLLM_DEVICE_COPY_CHECK=1``, or ``TLLM_DEBUG_MODE=1``), read
#: once at import.
CHECK_MODE: bool = (
    os.environ.get(CHECK_ENV, "0") == "1" or os.environ.get("TLLM_DEBUG_MODE", "")[0:1] == "1"
)
# Byte written into a merged copy's destination in check mode, so an early reader sees garbage and an early writer
# is caught at the flush.
_POISON = 0xA5


class Direction(Enum):
    H2D = "h2d"
    D2D = "d2d"
    # A device-side fill of a buffer's unused tail.
    FILL = "fill"


class CopySiteEnum(Enum):
    """Base of the copy-site enums: each member is ``(key, direction)``, the key unique across all enums."""

    def __init__(self, key: str, direction: Direction) -> None:
        self.key = key
        self.direction = direction


class CopySite(CopySiteEnum):
    """Copy sites in shared runtime code."""

    # CUDAGraphRunner.replay: the step's input ids / position ids into the graph's static inputs.
    GRAPH_INPUT_IDS = ("graph.input_ids", Direction.D2D)
    GRAPH_POSITION_IDS = ("graph.position_ids", Direction.D2D)
    # Decoder inputs: host position ids.
    INPUT_POSITION_IDS = ("inputs.position_ids", Direction.H2D)
    # The DeviceCopyBuffer uploads below skip unchanged values for every model; no plan merges them today. Decoder
    # inputs: the speculative gather ids. (The previous-batch tables skip through a host-side comparison.)
    INPUT_GATHER_IDS = ("inputs.gather_ids", Direction.H2D)
    # TrtllmAttentionMetadata.
    ATTN_KV_LENS = ("attn.kv_lens", Direction.H2D)
    ATTN_PROMPT_LENS = ("attn.prompt_lens", Direction.H2D)
    # Speculative metadata and samplers (DeviceCopyBuffer): Eagle3's batch indices and slot ids, DFlash's batch
    # indices and batch-to-slot map (also DSpark's), the speculative sampler's slot table.
    EAGLE3_BATCH_INDICES = ("eagle3.batch_indices", Direction.H2D)
    EAGLE3_SLOT_IDS = ("eagle3.slot_ids", Direction.H2D)
    DFLASH_BATCH_INDICES = ("dflash.batch_indices", Direction.H2D)
    DFLASH_BATCH_TO_SLOT = ("dflash.batch_to_slot", Direction.H2D)
    SPEC_SAMPLER_SLOT_TABLE = ("spec_sampler.slot_table", Direction.H2D)
    SPEC_SAMPLER_SLOT_IDS = ("spec_sampler.slot_ids", Direction.H2D)
    # Mamba cache managers and Mamba2 metadata (DeviceCopyBuffer): per-request state slots and the dummy-request mask.
    MAMBA_STATE_INDICES = ("mamba.state_indices", Direction.H2D)
    MAMBA_DUMMY_REQUEST_MASK = ("mamba.dummy_request_mask", Direction.H2D)
    MAMBA2_METADATA_STATE_INDICES = ("mamba2_metadata.state_indices", Direction.H2D)
    # SpecMetadata: the one-model speculative sampler's per-step Philox seeds and offsets (first read by the forward).
    SPEC_RNG_REQUEST_SEEDS = ("spec.rng_request_seeds", Direction.H2D)
    SPEC_RNG_REQUEST_OFFSETS = ("spec.rng_request_offsets", Direction.H2D)
    SPEC_RNG_SEEDS = ("spec.rng_seeds", Direction.H2D)
    SPEC_RNG_OFFSETS = ("spec.rng_offsets", Direction.H2D)


class PrologueOp(Protocol):
    """A model's one-launch in-graph prologue (``InGraph.prologue_op``).

    Called inside the capture as ``op(window, epoch_or_None, max_desc)``: it writes the window's host copies, then its
    device copies (at most ``max_desc`` of each), zeroes the window's header and, with an epoch, increments
    ``epoch[0]`` once. It writes exactly what the default prologue (two scatter launches and a header fill) writes.
    """

    def __call__(
        self, window: torch.Tensor, epoch: Optional[torch.Tensor], max_desc: int
    ) -> None: ...


@dataclass(frozen=True)
class InGraph:
    """The step's final barrier issued by the replayed decode graph itself (``CopyPlan.final_flush``).

    Every decode graph is captured with a prologue that scatters a fixed window, and the final flush before a replay
    only fills that window on the host: no operation is left between the decode graphs. Only the step's last segment
    rides the window, and only when it fits and is order-free (see ``_parallel_ok``); otherwise it is issued eagerly
    and the window is left empty. A graph whose runner has pre-replay hooks is captured without a prologue, since the
    hooks read the static inputs before the replay.

    ``early_copy``: the window is uploaded on a side copy stream into a device buffer of the prologue's own while the
    previous graph may still run; the prologue then waits for that upload and scatters device to device.
    ``prologue_op``: the model's one-launch prologue (needs ``early_copy``). ``step_epoch``: the prologue increments
    one int64 device counter once per replay (``step_epoch()``), never reset, so kernels later in the graph can tag
    per-step device flags with it instead of clearing them between steps.
    """

    early_copy: bool = False
    prologue_op: Optional[PrologueOp] = None
    step_epoch: bool = False

    def __post_init__(self) -> None:
        if self.prologue_op is not None and not self.early_copy:
            raise ValueError(
                "InGraph.prologue_op scatters the early-copied window, so it needs early_copy=True"
            )


@dataclass(frozen=True)
class CopyPlan:
    """When a model's per-step copies land. Empty by default: every copy is plain.

    ``merge``: sites whose copies are staged into the step's pack and land before the next barrier (see the module
    docstring). ``flush_after``: barrier sites, a subset of ``merge``. A call through one that stages its copy flushes
    the pack right after it; a call whose copy is unchanged, or runs eagerly, has nothing of its own to land and does
    not flush. ``changed_span`` (a subset of ``merge``): sites whose ``DeviceCopyBuffer.update(..., changed_span=True)``
    uploads may stage only each row's changed span (see there). ``parallel_scatter``: a segment whose copies are
    order-free is scattered with one program per copy instead of the serial one-CTA scatter. ``final_flush``: ``None``
    issues the step's last segment eagerly before the replay; ``InGraph(...)`` lets the replayed graph issue it (needs
    ``parallel_scatter``).
    """

    merge: frozenset[CopySiteEnum] = frozenset()
    flush_after: frozenset[CopySiteEnum] = frozenset()
    changed_span: frozenset[CopySiteEnum] = frozenset()
    parallel_scatter: bool = False
    final_flush: Optional[InGraph] = None

    def __post_init__(self) -> None:
        for name in ("merge", "flush_after", "changed_span"):
            object.__setattr__(self, name, frozenset(getattr(self, name)))
        for site in self.merge | self.flush_after | self.changed_span:
            if not isinstance(site, CopySiteEnum):
                raise TypeError(f"CopyPlan sites must be CopySiteEnum members, got {site!r}")
        fills = [site for site in self.merge if site.direction is Direction.FILL]
        if fills:
            raise ValueError(f"fill sites run on the device and cannot be merged: {fills}")
        unmerged = self.flush_after - self.merge
        if unmerged:
            raise ValueError(
                "a flush_after barrier fires only when its own copy was merged, so it must be in merge: "
                f"{sorted(unmerged, key=str)}"
            )
        unmerged_spans = self.changed_span - self.merge
        if unmerged_spans:
            raise ValueError(
                "a changed-span upload stages several copies into the pack, so its site must be in merge: "
                f"{sorted(unmerged_spans, key=str)}"
            )
        if self.final_flush is not None:
            if not isinstance(self.final_flush, InGraph):
                raise TypeError(
                    f"CopyPlan.final_flush must be None or InGraph, got {self.final_flush!r}"
                )
            if not self.parallel_scatter:
                raise ValueError(
                    "an InGraph final flush scatters in parallel, so it needs parallel_scatter=True"
                )
            if not self.merge:
                raise ValueError(
                    "an InGraph final flush issues merged copies, so the plan must merge some"
                )

    def __bool__(self) -> bool:
        return bool(self.merge)


def copy_plan_of(model: object) -> CopyPlan:
    """The plan ``model`` declares as ``copy_plan``, or the empty plan."""
    plan = getattr(model, "copy_plan", None)
    if plan is None:
        return CopyPlan()
    if not isinstance(plan, CopyPlan):
        raise TypeError(
            f"{type(model).__name__}.copy_plan must be a CopyPlan, got {type(plan).__name__}"
        )
    return plan


def _byte_range(t: torch.Tensor) -> tuple[int, int]:
    """[first, last + 1) device bytes ``t`` covers."""
    if t.numel() == 0:
        return (t.data_ptr(), t.data_ptr())
    if t.is_contiguous():
        return (t.data_ptr(), t.data_ptr() + t.numel() * t.element_size())
    span = sum((size - 1) * stride for size, stride in zip(t.shape, t.stride())) + 1
    return (t.data_ptr(), t.data_ptr() + span * t.element_size())


def _overlaps(r: tuple[int, int], ranges: list[tuple[int, int]]) -> bool:
    return any(r[0] < hi and lo < r[1] for lo, hi in ranges)


def _capturing(t: torch.Tensor) -> bool:
    """Whether a copy into ``t`` would be captured; a host tensor never is, and never initializes CUDA."""
    return t.is_cuda and torch.cuda.is_current_stream_capturing()


def _site_name(site: CopySiteEnum) -> str:
    return f"{type(site).__name__}.{site.name}"


# One compiled variant for every descriptor offset: specializing ``first`` (== 1, divisible by 16) would compile a
# new kernel when a step's segment first holds a new number of copies, after the warm-up.
@triton.jit(do_not_specialize=["first"])
def _parallel_scatter_kernel(
    stage_u8, stage_i64, first, DEVICE_SRC: tl.constexpr, BLOCK: tl.constexpr
):
    """Program i writes copy ``first + i`` ([dst, src, bytes] descriptors after the 2-word header). ``src`` is an offset
    into the staged segment for a host copy, a device address for a device copy (``DEVICE_SRC``)."""
    d = 2 + 3 * (first + tl.program_id(0))
    dst = tl.load(stage_i64 + d).to(tl.pointer_type(tl.uint8))
    src_word = tl.load(stage_i64 + d + 1)
    if DEVICE_SRC:
        src = src_word.to(tl.pointer_type(tl.uint8))
    else:
        src = stage_u8 + src_word
    nbytes = tl.load(stage_i64 + d + 2)
    for start in range(0, nbytes, BLOCK):
        o = start + tl.arange(0, BLOCK)
        tl.store(dst + o, tl.load(src + o, mask=o < nbytes), mask=o < nbytes)


@triton.jit
def _graph_scatter_kernel(
    stage_u8,
    stage_i64,
    epoch,
    DEVICE_SRC: tl.constexpr,
    BLOCK: tl.constexpr,
    BUMP_EPOCH: tl.constexpr,
):
    """Descriptor slot i of the host copies (``DEVICE_SRC`` False) or of the device copies, with the counts read from
    the staged header; a slot past its count writes nothing. ``BUMP_EPOCH``: program 0 also increments ``epoch[0]``."""
    i = tl.program_id(0)
    if BUMP_EPOCH:
        if i == 0:
            tl.store(epoch, tl.load(epoch) + 1)
    num_h2d = tl.load(stage_i64)
    if DEVICE_SRC:
        count = tl.load(stage_i64 + 1)
        d = 2 + 3 * (num_h2d + i)
    else:
        count = num_h2d
        d = 2 + 3 * i
    if i < count:
        dst = tl.load(stage_i64 + d).to(tl.pointer_type(tl.uint8))
        src_word = tl.load(stage_i64 + d + 1)
        if DEVICE_SRC:
            src = src_word.to(tl.pointer_type(tl.uint8))
        else:
            src = stage_u8 + src_word
        nbytes = tl.load(stage_i64 + d + 2)
        for start in range(0, nbytes, BLOCK):
            o = start + tl.arange(0, BLOCK)
            tl.store(dst + o, tl.load(src + o, mask=o < nbytes), mask=o < nbytes)


def _disjoint(ranges: list[tuple[int, int]]) -> bool:
    """Whether the [start, start + bytes) ranges are pairwise disjoint."""
    end = -1
    for start, nbytes in sorted(ranges):
        if start < end:
            return False
        end = start + nbytes
    return True


def _parallel_ok(h2d: list[tuple[int, int, int]], d2d: list[tuple[int, int, int]]) -> bool:
    """Whether issuing the host copies in parallel and then the device copies in parallel writes what the serial
    scatter (every host copy in order, then every device copy in order) writes: the host copies' destinations are
    disjoint, and the device copies' destinations are disjoint and none of them is a device copy's source (a host copy
    may still feed a device copy, as the launches run in order)."""
    if not _disjoint([(dst, n) for dst, _, n in h2d]):
        return False
    if not d2d:
        return True
    dsts = [(dst, n) for dst, _, n in d2d]
    if not _disjoint(dsts):
        return False
    return all(_disjoint(dsts + [(src, n)]) for _, src, n in d2d)


class _Pack:
    """One engine's pack of merged copies (see the module docstring).

    A step's pack lives in one slot of a ring of pinned staging buffers and in one device staging buffer. Each barrier
    issues the copies staged since the previous one as a segment ([header | descriptors | data], aligned) appended
    after the previous segment in the same slot, with its own upload and scatter launch. The ring advances once per
    step, at ``begin``, which waits only for the upload that last used the slot RING steps ago.

    With an ``InGraph`` final flush, the step's last segment is written instead into a pinned window (``ghost``) that
    the prologue of every captured decode graph issues (``capture_prologue``): a host copy of the window into the
    device staging buffer and the graph scatter, or, with ``early_copy``, a wait on the window's side-stream upload
    into the prologue's own device buffer (``gdev``) and the scatter from there. A segment's descriptors address its
    data relative to its own start, so the window is the segment's bytes as they are.
    """

    CAPACITY = 1 << 16
    MAX_DESC = 256
    # Pinned staging buffers in rotation, one per step: a step stages without waiting on the previous steps' uploads.
    RING = 4
    HEADER_BYTES = 16
    DESC_BYTES = 24
    ALIGN = 16

    # The pinned window a graph's prologue issues (InGraph): the header, the descriptors and up to ~10 KB of data.
    GRAPH_WINDOW = 1 << 14

    def __init__(
        self,
        device: torch.device,
        check: bool,
        parallel_scatter: bool = False,
        in_graph: Optional[InGraph] = None,
    ) -> None:
        self.host = [
            torch.empty(self.CAPACITY, dtype=torch.uint8, pin_memory=prefer_pinned())
            for _ in range(self.RING)
        ]
        self.events: list[Optional[torch.cuda.Event]] = [None] * self.RING
        self.dev = torch.empty(self.CAPACITY, dtype=torch.uint8, device=device)
        # A segment's data starts after its header and descriptor area.
        self.data_start = self.HEADER_BYTES + self.DESC_BYTES * self.MAX_DESC
        self.slot = self.RING - 1
        self.open = False
        self.check = check
        # First byte of the current segment in the slot.
        self.seg = 0
        self.h2d: list[tuple[int, int, int]] = []
        self.d2d: list[tuple[int, int, int]] = []
        # Bytes every staged device copy of the segment reads or writes: a later host copy into them would land out
        # of order (the scatter writes the host copies first).
        self.d2d_ranges: list[tuple[int, int]] = []
        # Check mode: bytes every staged copy of the segment writes.
        self.dst_ranges: list[tuple[int, int]] = []
        # Every tensor a staged copy reads or writes, kept alive until its segment is issued.
        self.staged: list[torch.Tensor] = []
        # The tensors of the segment in the graph window, kept alive until the replay that issues them is queued.
        self.graph_staged: list[torch.Tensor] = []
        # The callers' on_drop callbacks of the segment's copies, called if the pack is discarded.
        self.drops: list[Callable[[], None]] = []
        # Check mode: (site, destination) of every staged copy, and (site, source, snapshot) of every device copy.
        self.poisoned: list[tuple[CopySiteEnum, torch.Tensor]] = []
        self.sources: list[tuple[CopySiteEnum, torch.Tensor, torch.Tensor]] = []
        self.used = self.seg + self.data_start
        self.parallel_scatter = parallel_scatter
        self.in_graph = in_graph
        self.early_copy = in_graph is not None and in_graph.early_copy
        self.prologue_op = in_graph.prologue_op if in_graph is not None else None
        # The step epoch (InGraph.step_epoch); None without it.
        self.epoch: Optional[torch.Tensor] = (
            torch.zeros(1, dtype=torch.int64, device=device)
            if in_graph is not None and in_graph.step_epoch
            else None
        )
        if self.early_copy:
            # The prologue's own device window: an eager segment (self.dev) can never leave a header in it.
            self.gdev = torch.zeros(self.GRAPH_WINDOW, dtype=torch.uint8, device=device)
            self.copy_stream = torch.cuda.Stream(device=device)
            # Recorded by the side upload once it landed (the prologue waits on it) ...
            self.staged_ready = torch.cuda.Event(external=True)
            # ... and by every replayed prologue after its scatter (the next upload waits on it).
            self.prologue_done = torch.cuda.Event(external=True)
            # A CUDA event is created on its first record, and a wait on one never created is skipped: a capture
            # would then leave the prologue's wait out of the graph. Create both now.
            self.staged_ready.record(self.copy_stream)
            self.prologue_done.record(self.copy_stream)
        if in_graph is not None:
            self.ghost = torch.zeros(
                self.GRAPH_WINDOW, dtype=torch.uint8, pin_memory=prefer_pinned()
            )
            # Recorded by every replayed prologue right after it copied the window (an event node).
            self.ghost_free = torch.cuda.Event(external=True)
        if parallel_scatter:
            self._warm_up(device)
            if in_graph is not None:
                self._warm_up_graph_kernel()

    def _warm_up(self, device: torch.device) -> None:
        """Compile the parallel scatter now, not on a serving step's flush."""
        dst = torch.zeros(8, dtype=torch.int32, device=device)
        check, self.check = self.check, False
        try:
            self.begin()
            self.stage(None, dst[:4], torch.arange(4, dtype=torch.int32))
            self.stage(None, dst[4:], dst[:4])
            self.flush()
        finally:
            self.check = check
        torch.cuda.synchronize(device)
        # The next step starts in a fresh slot and segment.
        self.seg = 0

    def _warm_up_graph_kernel(self) -> None:
        """Compile the prologue's scatter now (a zero-count header: it writes nothing)."""
        self.dev[: self.HEADER_BYTES].zero_()
        if self.prologue_op is not None:
            # The window it is captured with; its header is zero until the first upload.
            self.prologue_op(self.gdev, self.epoch, self.MAX_DESC)
        else:
            self._launch_graph_scatter(self.dev)
        torch.cuda.synchronize(self.dev.device)

    def _launch_graph_scatter(self, window: torch.Tensor) -> None:
        stage_i64 = window.view(torch.int64)
        for device_src in (False, True):
            bump = self.epoch is not None and not device_src
            _graph_scatter_kernel[(self.MAX_DESC,)](
                window,
                stage_i64,
                self.epoch if bump else stage_i64,
                DEVICE_SRC=device_src,
                BLOCK=256,
                BUMP_EPOCH=bump,
                num_warps=1,
            )

    def capture_prologue(self) -> bool:
        """Inside a graph capture: the prologue that issues the window ``flush(in_graph=True)`` fills. Returns whether
        the graph has it."""
        if self.in_graph is None:
            return False
        if self.early_copy:
            torch.cuda.current_stream().wait_event(self.staged_ready)
            if self.prologue_op is not None:
                self.prologue_op(self.gdev, self.epoch, self.MAX_DESC)
            else:
                self._launch_graph_scatter(self.gdev)
                # Consumed: a replay with no new upload scatters nothing.
                self.gdev[: self.HEADER_BYTES].zero_()
            self.prologue_done.record()
            return True
        self.dev[: self.GRAPH_WINDOW].copy_(self.ghost, non_blocking=True)
        self.ghost_free.record()
        self._launch_graph_scatter(self.dev)
        return True

    def _upload_window(self, nbytes: int) -> None:
        """early_copy: upload the filled pinned window into gdev on the side stream, once the last replayed prologue
        has consumed gdev; the window is fully written on the host before this is enqueued."""
        stream = self.copy_stream
        stream.wait_event(self.prologue_done)
        with torch.cuda.stream(stream):
            self.gdev[:nbytes].copy_(self.ghost[:nbytes], non_blocking=True)
        self.ghost_free.record(stream)
        self.staged_ready.record(stream)

    def _ghost_writable(self) -> None:
        # The last replayed prologue (or side upload) has copied the window (a no-op before the first replay).
        self.ghost_free.synchronize()

    def quiesce_graph_window(self) -> None:
        """Before replaying a graph with the prologue without filling the window: a zero-count window, so it writes
        nothing."""
        if self.in_graph is not None and self.ghost[: self.HEADER_BYTES].any():
            self._ghost_writable()
            self.ghost[: self.HEADER_BYTES].zero_()
            if self.early_copy:
                # A window uploaded but never replayed would still sit in gdev: overwrite its header.
                self._upload_window(self.HEADER_BYTES)

    def _fill_graph_window(self) -> bool:
        """Write the open segment into the pinned window the replay's prologue issues; False when it does not fit or
        is not order-free (the caller issues it eagerly)."""
        nbytes = self.used - self.seg
        if nbytes > self.GRAPH_WINDOW or not _parallel_ok(self.h2d, self.d2d):
            return False
        desc = [len(self.h2d), len(self.d2d)] + [
            value for copy in self.h2d + self.d2d for value in copy
        ]
        self._ghost_writable()
        self.ghost[:nbytes].copy_(self.host[self.slot][self.seg : self.used])
        self.ghost[: 8 * len(desc)].view(torch.int64).copy_(torch.tensor(desc, dtype=torch.int64))
        if self.early_copy:
            self._upload_window(nbytes)
        return True

    def on_graph_replay(self) -> None:
        """Release the window's operands once the graph reading their addresses has been queued."""
        if self.graph_staged:
            stream = torch.cuda.current_stream(self.dev.device)
            for tensor in self.graph_staged:
                tensor.record_stream(stream)
            self.graph_staged.clear()

    def begin(self) -> None:
        if self.open:
            self.flush()
        self.slot = (self.slot + 1) % self.RING
        event = self.events[self.slot]
        if event is not None:
            # The uploads that last read this pinned buffer have run.
            event.synchronize()
        self.seg = 0
        self._clear()
        self.open = True

    def _clear(self) -> None:
        self.h2d.clear()
        self.d2d.clear()
        self.d2d_ranges.clear()
        self.dst_ranges.clear()
        self.staged.clear()
        self.drops.clear()
        self.poisoned.clear()
        self.sources.clear()
        self.used = self.seg + self.data_start

    def writes(self, t: torch.Tensor) -> bool:
        """Whether a staged copy writes any byte of ``t`` (check mode only)."""
        return _overlaps(_byte_range(t), self.dst_ranges)

    def pending_sites(self, t: torch.Tensor) -> list[CopySiteEnum]:
        """The sites of staged, not yet issued copies that write any byte of ``t`` (check mode only)."""
        r = _byte_range(t)
        return [site for site, dst in self.poisoned if _overlaps(r, [_byte_range(dst)])]

    def stage(
        self,
        site: CopySiteEnum,
        dst: torch.Tensor,
        src: torch.Tensor,
        on_drop: Optional[Callable[[], None]] = None,
    ) -> bool:
        if not dst.is_cuda or dst.numel() != src.numel():
            return False
        nbytes = dst.numel() * dst.element_size()
        # Byte ranges are needed only against staged device copies, and in check mode.
        dst_range = _byte_range(dst) if self.d2d_ranges or self.check or src.is_cuda else None
        if not src.is_cuda:
            if self.d2d_ranges and _overlaps(dst_range, self.d2d_ranges):
                return False
            # A contiguous destination is one descriptor; a 2-D slice with contiguous rows (a page table's leading
            # rows and columns) is one per row.
            if dst.is_contiguous():
                rows, row_bytes, row_stride = 1, nbytes, 0
            elif dst.dim() == 2 and dst.stride(1) == 1:
                rows = dst.shape[0]
                row_bytes = dst.shape[1] * dst.element_size()
                row_stride = dst.stride(0) * dst.element_size()
            else:
                return False
            offset = (self.used + self.ALIGN - 1) // self.ALIGN * self.ALIGN
            if (
                offset + nbytes > self.CAPACITY
                or len(self.h2d) + len(self.d2d) + rows > self.MAX_DESC
            ):
                return False
            # A dtype change converts on the host, as copy_ would on the device.
            src = src.to(dst.dtype).contiguous()
            self.host[self.slot][offset : offset + nbytes].copy_(src.view(-1).view(torch.uint8))
            base = dst.data_ptr()
            # The scatter reads a host copy's bytes at an offset from its segment's start.
            rel = offset - self.seg
            for row in range(rows):
                self.h2d.append((base + row * row_stride, rel + row * row_bytes, row_bytes))
            self.staged.append(dst)
            self.used = offset + nbytes
        else:
            if self.used > self.CAPACITY or len(self.h2d) + len(self.d2d) >= self.MAX_DESC:
                return False
            if not (
                src.is_cuda
                and src.dtype == dst.dtype
                and src.is_contiguous()
                and dst.is_contiguous()
            ):
                return False
            src_range = _byte_range(src)
            if self.check and not _overlaps(src_range, self.dst_ranges):
                # A source no staged copy fills must not change before the flush reads it.
                self.sources.append((site, src, src.clone()))
            self.d2d.append((dst.data_ptr(), src.data_ptr(), nbytes))
            self.d2d_ranges += (dst_range, src_range)
            self.staged += (dst, src)
        if on_drop is not None:
            self.drops.append(on_drop)
        if self.check:
            self.dst_ranges.append(dst_range)
            dst.view(torch.uint8).fill_(_POISON)
            self.poisoned.append((site, dst))
        return True

    def _verify(self) -> None:
        for site, dst in self.poisoned:
            if not bool(dst.view(torch.uint8).eq(_POISON).all()):
                raise RuntimeError(
                    f"{CHECK_ENV}: the destination of merged copy {_site_name(site)} was written between the "
                    "copy and its flush; the plan must not merge this site, or needs a barrier before that writer"
                )
        for site, src, snapshot in self.sources:
            if not torch.equal(src, snapshot):
                raise RuntimeError(
                    f"{CHECK_ENV}: the source of merged device copy {_site_name(site)} changed between the copy "
                    "and its flush; the plan must not merge this site, or needs a barrier before that writer"
                )

    def _issue(self) -> None:
        """Issue the segment staged since the last barrier and start the next one after it in the same slot."""
        if not self.h2d and not self.d2d:
            return
        if self.check:
            self._verify()
        host, seg = self.host[self.slot], self.seg
        desc = [len(self.h2d), len(self.d2d)] + [
            value for copy in self.h2d + self.d2d for value in copy
        ]
        host[seg : seg + 8 * len(desc)].view(torch.int64).copy_(
            torch.tensor(desc, dtype=torch.int64)
        )
        # Descriptors and data in one upload; the gap between them is unused padding.
        self.dev[seg : self.used].copy_(host[seg : self.used], non_blocking=True)
        if self.parallel_scatter and _parallel_ok(self.h2d, self.d2d):
            stage = self.dev[seg:]
            stage_i64 = stage.view(torch.int64)
            if self.h2d:
                _parallel_scatter_kernel[(len(self.h2d),)](
                    stage, stage_i64, 0, DEVICE_SRC=False, BLOCK=256, num_warps=2
                )
            if self.d2d:
                _parallel_scatter_kernel[(len(self.d2d),)](
                    stage, stage_i64, len(self.h2d), DEVICE_SRC=True, BLOCK=256, num_warps=2
                )
        else:
            torch.ops.trtllm.device_copy_scatter(self.dev[seg:])
        event = self.events[self.slot]
        if event is None:
            event = self.events[self.slot] = torch.cuda.Event()
        event.record()
        # The scatter is queued on the current stream: the caching allocator keeps the staged tensors' memory for it.
        self.seg = (self.used + self.ALIGN - 1) // self.ALIGN * self.ALIGN
        self._clear()

    def barrier(self) -> None:
        """Issue what is staged; the pack stays open for the rest of the step."""
        if self.open:
            self._issue()

    def flush(self, in_graph: bool = False) -> None:
        """Issue what is staged and close the pack (the step's final barrier).

        ``in_graph``: the next replay is of a graph captured with this pack's prologue, which issues the segment if it
        rides the window; otherwise the segment is issued eagerly and the window left empty.
        """
        if not self.open:
            if in_graph:
                self.quiesce_graph_window()
            return
        self.open = False
        if in_graph and self.in_graph is not None:
            if self.check:
                self._verify()
            if self._fill_graph_window():
                self.graph_staged.extend(self.staged)
                self._clear()
                return
            self.quiesce_graph_window()
        self._issue()

    def discard(self) -> None:
        """Close the pack without issuing it (its step failed); the unissued copies' on_drop callbacks run."""
        drops = list(self.drops)
        self.open = False
        self.seg = 0
        self._clear()
        for on_drop in drops:
            on_drop()


def _view_key(t: torch.Tensor) -> tuple:
    return (t.storage_offset(), tuple(t.shape), tuple(t.stride()), t.dtype)


class DeviceCopier:
    """One engine's merge pack, applied per site as its ``CopyPlan`` says.

    In check mode (``CHECK_MODE`` when the copier is built), every merged copy's destination is poisoned at the call and
    verified untouched when its segment is issued, every merged device copy's source is verified unchanged then, and
    every ``unchanged`` copy or fill is verified against the device contents. A failed check raises and names the site.
    A device read of a merged destination before its barrier is detected only by a reader that calls
    ``expect_landed``; any other early reader reads the poison.
    """

    def __init__(self, plan: CopyPlan, device: torch.device) -> None:
        self.plan = plan
        self.check = CHECK_MODE
        self._pack = (
            _Pack(device, self.check, plan.parallel_scatter, plan.final_flush)
            if plan.merge
            else None
        )
        # Site membership by identity: an Enum member hashes through a Python-level __hash__.
        self._merge_ids = frozenset(map(id, plan.merge))
        self._barrier_ids = frozenset(map(id, plan.flush_after))
        self._changed_span_ids = frozenset(map(id, plan.changed_span))

    @property
    def merges(self) -> bool:
        return self._pack is not None

    def pending_sites(self, t: torch.Tensor) -> list[CopySiteEnum]:
        """Check mode: the merged sites whose staged, not yet issued copies write any byte of ``t``."""
        if self._pack is None or not self._pack.open:
            return []
        return self._pack.pending_sites(t)

    def writes_pending(self, t: torch.Tensor) -> bool:
        """Check mode: whether a staged, not yet issued copy writes any byte of ``t``."""
        return self._pack is not None and self._pack.open and self._pack.writes(t)

    @property
    def is_open(self) -> bool:
        return self._pack is not None and self._pack.open

    def merges_now(self, site: CopySiteEnum) -> bool:
        """Whether a copy through ``site`` would be staged into the open pack now."""
        return (
            self._pack is not None
            and self._pack.open
            and id(site) in self._merge_ids
            and not torch.cuda.is_current_stream_capturing()
        )

    def stages_changed_span(self, site: CopySiteEnum) -> bool:
        """Whether a changed-span upload through ``site`` may stage per-row spans now (``CopyPlan.changed_span``). Not
        through a barrier site: each span would flush on its own, so the whole view is staged with one barrier."""
        return (
            id(site) in self._changed_span_ids
            and id(site) not in self._barrier_ids
            and self.merges_now(site)
        )

    def begin(self) -> None:
        if self._pack is not None:
            self._pack.begin()

    def flush(self, in_graph: bool = False) -> None:
        if self._pack is not None:
            self._pack.flush(in_graph)

    @property
    def epoch(self) -> Optional[torch.Tensor]:
        """The step epoch (``InGraph.step_epoch``), or None."""
        return self._pack.epoch if self._pack is not None else None

    def capture_prologue(self) -> bool:
        """Inside a graph capture: add the in-graph final flush's prologue; whether the graph has it."""
        return self._pack is not None and self._pack.capture_prologue()

    def quiesce_graph_window(self) -> None:
        """Before replaying a graph with this copier's prologue outside its context: an empty window."""
        if self._pack is not None:
            self._pack.quiesce_graph_window()

    def on_graph_replay(self) -> None:
        """After queueing the replay of a graph with this copier's prologue: release the window's operands."""
        if self._pack is not None:
            self._pack.on_graph_replay()

    def discard(self) -> None:
        """Drop the open pack: the step failed, so its staged copies never land (their on_drop callbacks run)."""
        if self._pack is not None:
            self._pack.discard()

    def copy(
        self,
        site: CopySiteEnum,
        dst: torch.Tensor,
        src: torch.Tensor,
        unchanged: bool = False,
        on_drop: Optional[Callable[[], None]] = None,
        snapshot: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> bool:
        if site.direction is Direction.FILL:
            raise ValueError(f"{_site_name(site)} is a fill site; use fill_tail")
        if src.is_cuda == (site.direction is Direction.H2D):
            raise ValueError(
                f"{_site_name(site)} is declared {site.direction.value}, but its source is on {src.device}"
            )
        if unchanged:
            _check_unchanged_copy(site, dst, src, self)
            return False
        pack = self._pack
        if (
            pack is not None
            and pack.open
            and id(site) in self._merge_ids
            and not torch.cuda.is_current_stream_capturing()
        ):
            if pack.stage(site, dst, src, on_drop):
                if id(site) in self._barrier_ids:
                    pack.barrier()
                return True
            # Cannot be staged: what is staged lands first, then this copy (which keeps the order).
            pack.barrier()
        else:
            self._land_staged()
        dst.copy_(src if snapshot is None else snapshot(src), non_blocking=True)
        return True

    def fill_tail(
        self, site: CopySiteEnum, buf: torch.Tensor, start: int, value: int, unchanged: bool = False
    ) -> None:
        if site.direction is not Direction.FILL:
            raise ValueError(f"{_site_name(site)} is not a fill site")
        if not unchanged and start < buf.shape[0]:
            self._land_staged()
        _fill_tail(site, buf, start, value, unchanged, self)

    def _land_staged(self) -> None:
        """Before an eager copy or fill: issue what is staged, so the eager one lands after every earlier copy."""
        pack = self._pack
        if (
            pack is not None
            and pack.open
            and (pack.h2d or pack.d2d)
            and not torch.cuda.is_current_stream_capturing()
        ):
            pack.barrier()


def _check_unchanged_copy(
    site: CopySiteEnum, dst: torch.Tensor, src: torch.Tensor, copier: Optional[DeviceCopier]
) -> None:
    """An ``unchanged`` copy is refused during capture and, in check mode, verified against the device."""
    if _capturing(dst):
        raise ValueError(
            f"{_site_name(site)}: unchanged=True during CUDA graph capture; a captured copy reruns on every "
            "replay, so it must be issued (and not recorded as skippable)"
        )
    if CHECK_MODE and not (copier is not None and copier.writes_pending(dst)):
        if not torch.equal(dst.cpu(), src.to(device="cpu", dtype=dst.dtype)):
            raise RuntimeError(
                f"{CHECK_ENV}: copy {_site_name(site)} was passed unchanged=True, but its destination holds other "
                "values; something else writes this destination, or the owner's record is stale"
            )


def _fill_tail(
    site: CopySiteEnum,
    buf: torch.Tensor,
    start: int,
    value: int,
    unchanged: bool,
    copier: Optional[DeviceCopier],
) -> None:
    if start >= buf.shape[0]:
        return
    if not unchanged:
        buf[start:].fill_(value)
        return
    if _capturing(buf):
        raise ValueError(
            f"{_site_name(site)}: unchanged=True during CUDA graph capture; a captured fill reruns on every replay"
        )
    tail = buf[start:]
    if CHECK_MODE and not (copier is not None and copier.writes_pending(tail)):
        if not bool(tail.eq(value).all()):
            raise RuntimeError(
                f"{CHECK_ENV}: tail fill {_site_name(site)} was passed unchanged=True, but the tail holds other "
                "values; something else writes this tail, or the owner's record is stale"
            )


_active: ContextVar[Optional[DeviceCopier]] = ContextVar("trtllm_device_copier", default=None)


@contextlib.contextmanager
def active(copier: Optional[DeviceCopier]) -> Iterator[None]:
    """Make ``copier`` the calling context's copier (None: plain copies) for the duration."""
    token = _active.set(copier)
    try:
        yield
    finally:
        _active.reset(token)


def current() -> Optional[DeviceCopier]:
    return _active.get()


def copy(
    site: CopySiteEnum,
    dst: torch.Tensor,
    src: torch.Tensor,
    *,
    unchanged: bool = False,
    on_drop: Optional[Callable[[], None]] = None,
    snapshot: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> bool:
    """``dst.copy_(src, non_blocking=True)``, merged as the active copier's plan says; True if issued.

    ``unchanged``: the caller's assertion that ``dst`` already holds ``src``; the copy is not issued and False is
    returned. Refused during CUDA graph capture, and verified in check mode. ``on_drop``: called if the copy was merged
    and its pack is then discarded without issuing it (the caller's record of the copy is then stale). ``snapshot``:
    applied to ``src`` when the copy is issued eagerly, for a caller that writes ``src`` again before the copy runs (a
    merged copy is snapshotted into the pack already).
    """
    copier = _active.get()
    if copier is not None:
        return copier.copy(site, dst, src, unchanged, on_drop, snapshot)
    if unchanged:
        _check_unchanged_copy(site, dst, src, None)
        return False
    dst.copy_(src if snapshot is None else snapshot(src), non_blocking=True)
    return True


def fill_tail(
    site: CopySiteEnum, buf: torch.Tensor, start: int, value: int, *, unchanged: bool = False
) -> None:
    """``buf[start:].fill_(value)``; not issued when the caller asserts ``unchanged`` (the tail already holds it).

    ``unchanged`` is refused during CUDA graph capture and verified in check mode.
    """
    copier = _active.get()
    if copier is not None:
        copier.fill_tail(site, buf, start, value, unchanged)
        return
    _fill_tail(site, buf, start, value, unchanged, None)


def expect_landed(site: CopySiteEnum, t: torch.Tensor) -> None:
    """Check mode: assert that no merged copy into ``t`` is still staged, i.e. a barrier precedes this reader.

    ``site`` names the reader's anchor (the site whose ``flush_after`` barrier should have issued the copies) in the
    error. Call it only under ``if device_copy.CHECK_MODE:``; it raises when check mode is off, so an unguarded call
    cannot add work to the hot path unnoticed. Without an active copier nothing is staged and it passes.
    """
    if not CHECK_MODE:
        raise RuntimeError(
            f"expect_landed({_site_name(site)}) is a {CHECK_ENV} assertion; call it under `if device_copy.CHECK_MODE:`"
        )
    copier = _active.get()
    if copier is None:
        return
    pending = copier.pending_sites(t)
    if pending:
        raise RuntimeError(
            f"{CHECK_ENV}: a reader anchored at {_site_name(site)} reads bytes that merged copies "
            f"{sorted({_site_name(p) for p in pending})} have not written yet; the plan needs {_site_name(site)} "
            "in flush_after, and its call must precede this reader"
        )


def begin() -> None:
    """Open the active copier's pack. Call only when the step will reach ``flush`` (a captured graph's replay)."""
    copier = _active.get()
    if copier is not None:
        copier.begin()


def is_open() -> bool:
    copier = _active.get()
    return copier is not None and copier.is_open


def stages_changed_span(site: CopySiteEnum) -> bool:
    """Whether a changed-span upload through ``site`` may stage per-row spans now (see ``DeviceCopyBuffer.update``)."""
    copier = _active.get()
    return copier is not None and copier.stages_changed_span(site)


def flush(in_graph: bool = False) -> None:
    """Issue the active open pack's last segment (one upload and one scatter launch on the current stream) and close it.

    The step's final barrier, at the graph replay. ``in_graph``: the next replay is of a graph captured with the active
    copier's prologue (``capture_prologue``), which issues the segment itself when it rides the window.
    """
    copier = _active.get()
    if copier is not None:
        copier.flush(in_graph)


def is_active(copier: DeviceCopier) -> bool:
    """Whether ``copier`` is the calling context's copier."""
    return _active.get() is copier


def capture_prologue() -> Optional[DeviceCopier]:
    """Inside a graph capture: add the active copier's in-graph final flush prologue (``InGraph``); the copier whose
    window the graph issues, or None when the graph has no prologue."""
    copier = _active.get()
    return copier if copier is not None and copier.capture_prologue() else None


def step_epoch() -> Optional[torch.Tensor]:
    """The active copier's step epoch (an int64 [1] device tensor, ``InGraph.step_epoch``), or None."""
    copier = _active.get()
    return copier.epoch if copier is not None else None


def discard() -> None:
    """Drop the active copier's open pack (its step failed); the staged copies' on_drop callbacks run."""
    copier = _active.get()
    if copier is not None:
        copier.discard()


class DeviceCopyBuffer:
    """A device buffer's record of what its last issued copy wrote, so its owner skips uploads of values it holds.

    ``site`` names the buffer's uploads (a host-to-device ``CopySiteEnum`` member): ``update`` issues them through
    ``copy``, so a model's plan may merge them, and check mode verifies every skipped one. ``site=None`` makes a
    record-only buffer for an owner that writes the buffer through its own path and keeps the record with ``matches`` /
    ``record`` / ``invalidate``.

    The record follows the copy: it is dropped before a copy is issued (a copy that fails leaves none), written only
    after the copy was issued, dropped again if a merged copy never lands (``on_drop``, the pack was discarded), and
    neither consulted nor written during CUDA graph capture, where a copy is always issued: a captured copy reruns on
    every replay. An owner that writes the buffer by other means calls ``invalidate``.

    ``update(host)`` writes the leading ``host.numel()`` elements, and a recorded longer run whose prefix matches still
    holds them, so it is skipped. ``update(host, view=v)`` writes ``host`` into the view ``v`` of the buffer's storage
    instead and is skipped only for the same view geometry and host shape. The host values are snapshotted when the
    copy runs eagerly (into pinned memory with ``pin_memory``), since owners reuse their host buffers; a merged copy is
    snapshotted into the pack instead. ``snapshot=False`` skips the eager snapshot for an owner that passes a host
    tensor it does not write again.

    ``fill_tail`` fills ``tensor[start:]`` and records the fill (start, value), so an identical fill is skipped. Call it
    after every write of the live rows ``tensor[:start]``, including one that covers the whole buffer (``start ==
    len(tensor)``, which fills nothing): a live-row write past the recorded start overwrites the tail.
    """

    def __init__(
        self,
        tensor: torch.Tensor,
        *,
        site: Optional[CopySiteEnum],
        pin_memory: bool = False,
        snapshot: bool = True,
    ) -> None:
        if site is not None and site.direction is not Direction.H2D:
            raise ValueError(
                f"DeviceCopyBuffer uploads from the host; {_site_name(site)} is not an h2d site"
            )
        self._tensor = tensor
        self._site = site
        self._pin_memory = pin_memory
        self._snapshot = self._stage if snapshot else None
        self._last_host_values = None
        self._last_host_shape: torch.Size | None = None
        self._copy_key: object = None
        # (view, start, value) of the last tail fill.
        self._tail: Optional[tuple] = None

    @property
    def tensor(self) -> torch.Tensor:
        return self._tensor

    @property
    def site(self) -> Optional[CopySiteEnum]:
        return self._site

    @property
    def host_values(self) -> Optional[torch.Tensor]:
        """Last recorded host values, reshaped to the submitted view; callers must not mutate them."""
        if self._last_host_values is None:
            return None
        count = self._last_host_shape.numel()
        return self._last_host_values[:count].view(self._last_host_shape)

    def invalidate(self) -> None:
        """Forget a pending or externally overwritten copy (and the tail fill)."""
        self._last_host_values = None
        self._last_host_shape = None
        self._copy_key = None
        self._tail = None

    def matches(self, host: torch.Tensor, *, key: object = None, exact: bool = False) -> bool:
        """Whether the recorded host inputs match, including any copy-specific layout key.

        ``exact`` requires the same host shape instead of accepting a matching prefix.
        Callers that transform or defer their copies must record only successful submissions
        and invalidate the record if the submission is discarded or another writer intervenes.
        """
        last = self._last_host_values
        flat = host.reshape(-1)
        return (
            last is not None
            and key == self._copy_key
            and last.dtype == host.dtype
            and (not exact or self._last_host_shape == host.shape)
            and last.numel() >= flat.numel()
            and torch.equal(last[: flat.numel()], flat)
        )

    def record(self, host: torch.Tensor, *, key: object = None) -> None:
        """Remember the inputs of a copy issued by this buffer or a specialized copy path."""
        flat = host.reshape(-1)
        last = self._last_host_values
        if (
            last is None
            or key != self._copy_key
            or last.dtype != host.dtype
            or last.numel() <= flat.numel()
        ):
            self._last_host_values = flat.clone()
        else:
            last = last.clone()
            last[: flat.numel()] = flat
            self._last_host_values = last
        self._last_host_shape = host.shape
        self._copy_key = key

    def _stage(self, src: torch.Tensor) -> torch.Tensor:
        staging = torch.empty_like(src, device="cpu", pin_memory=self._pin_memory)
        staging.copy_(src)
        return staging

    # A changed-span upload is used only while it moves at most this fraction of the view's elements.
    CHANGED_SPAN_MAX_FRACTION = 0.5

    def update(
        self,
        host: torch.Tensor,
        force: bool = False,
        *,
        view: Optional[torch.Tensor] = None,
        changed_span: bool = False,
    ) -> bool:
        """Upload ``host`` through the buffer's site unless the record says the buffer holds it; True if issued.

        ``force`` issues the copy whatever the record says. ``view``: the destination, a view of the buffer's storage
        (default: the buffer's leading ``host.numel()`` elements).

        ``changed_span``: a changed upload into a 2-D ``view`` with contiguous rows, merged into the open pack through a
        site in the plan's ``changed_span``, whose record covers the same rows (same first element, row count and row
        stride) issues, per row, only the span from its first to its last changed column, plus any columns the record
        does not cover, each through the buffer's site. A page table that gained one page then stages a few bytes
        instead of every padded row. It falls back to the whole view when the spans would move more than
        ``CHANGED_SPAN_MAX_FRACTION`` of it, or when the copy is not merged (per-row eager copies would cost more than
        one).
        """
        site = self._site
        if site is None:
            raise ValueError("a record-only DeviceCopyBuffer (site=None) does not issue copies")
        if view is None:
            src = host.reshape(-1)
            dst = self._tensor.view(-1)[: src.numel()]
            key, exact = None, False
            if self._tail is not None and src.numel() > self._tail[1]:
                self._tail = None
        else:
            src, dst = host, view
            key, exact = _view_key(view), True
            self._tail = None
        if _capturing(dst):
            self.invalidate()
            return copy(site, dst, src, snapshot=self._snapshot)
        if not force and self.matches(host, key=key, exact=exact):
            return copy(site, dst, src, unchanged=True)
        # Dropped while the copy is issued, then recorded on top of what the buffer held (a longer recorded run
        # keeps its tail).
        previous = (self._last_host_values, self._last_host_shape, self._copy_key)
        old = (
            (self._copy_key, self.host_values)
            if changed_span and view is not None and self._last_host_values is not None
            else None
        )
        self._last_host_values = self._last_host_shape = self._copy_key = None
        if old is not None and self._copy_changed_span(site, dst, src, *old):
            issued = True
        else:
            issued = copy(site, dst, src, on_drop=self.invalidate, snapshot=self._snapshot)
        self._last_host_values, self._last_host_shape, self._copy_key = previous
        self.record(host, key=key)
        return issued

    def _copy_changed_span(
        self,
        site: CopySiteEnum,
        dst: torch.Tensor,
        src: torch.Tensor,
        old_key: object,
        old: torch.Tensor,
    ) -> bool:
        """Stage only the changed span of each row of ``dst`` (see ``update``); False if not applicable."""
        if not isinstance(old_key, tuple) or not stages_changed_span(site):
            return False
        offset, shape, stride, dtype = old_key
        if (
            dst.dim() != 2
            or dst.stride(1) != 1
            or offset != dst.storage_offset()
            or len(shape) != 2
            or shape[0] != dst.shape[0]
            or stride != tuple(dst.stride())
            or dtype != dst.dtype
            or old.dtype != src.dtype
            or tuple(src.shape) != tuple(dst.shape)
        ):
            return False
        cols = dst.shape[1]
        known = min(shape[1], cols)
        changed = src[:, :known] != old[:, :known]
        if cols > known:
            # Columns the record does not cover hold unknown device values.
            changed = torch.cat([changed, changed.new_ones(dst.shape[0], cols - known)], dim=1)
        rows = changed.any(dim=1).nonzero().flatten().tolist()
        spans = []
        for row in rows:
            columns = changed[row].nonzero().flatten()
            spans.append((row, int(columns[0]), int(columns[-1]) + 1))
        if sum(hi - lo for _, lo, hi in spans) > self.CHANGED_SPAN_MAX_FRACTION * dst.numel():
            return False
        for row, lo, hi in spans:
            copy(site, dst[row, lo:hi], src[row, lo:hi], on_drop=self.invalidate)
        return True

    def fill_tail(self, site: CopySiteEnum, start: int, value: int) -> None:
        """``fill_tail(site, tensor, start, value)``, passing unchanged=True when the last fill was the same."""
        buf = self._tensor
        if start >= buf.shape[0] or _capturing(buf):
            self._tail = None
            fill_tail(site, buf, start, value)
            return
        key = (_view_key(buf), start, value)
        unchanged = self._tail == key
        if not unchanged:
            self._tail = None
        fill_tail(site, buf, start, value, unchanged=unchanged)
        self._tail = key
