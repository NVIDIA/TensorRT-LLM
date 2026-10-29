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
"""Dense DWDP runtime: slot rings, events and the per-layer hooks.

Per covered layer ``l`` (``s_A`` and ``s_D`` are its slots in the attention and MoE-dense rings):

=====  ==========================================  ==========================================
Hook   Call site                                   Action
=====  ==========================================  ==========================================
H0     ``KimiLinearModel.forward`` entry           ``DwdpManager.ensure_primed()``: barrier
                                                   after an abort, then ``L(1), E(1), L(2),
                                                   E(2)`` unless already primed
H1     decoder layer entry                         compute waits ``ready_A[s_A]`` and
                                                   ``ready_D[s_D]``
H2     after the attention call                    record ``consume_A[s_A]``, then enqueue
                                                   ``L(l + 2)``
H3     after the MoE block                         record ``consume_D[s_D]``
=====  ==========================================  ==========================================

All reads of ``A(l)`` happen between H1 and H2; all reads of ``D(l)`` between H1 and H3.  Copies
run on the expert DWDP copy stream, so the copy FIFO is ``... E(l+1), L(l+2), E(l+2), L(l+3)
...``: enqueue order equals deadline order.  Between commit and release the runtime never
synchronizes the host; every dependency is a stream wait on an event.  The one exception is the
slot watchdog of ``TRTLLM_DWDP_DENSE_DEBUG=2`` (``watchdog.SlotWatchdog``), whose counters H3 of
the last covered layer reads once per forward.
"""

from __future__ import annotations

import enum
import operator
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING

from tensorrt_llm.logger import logger

from .types import (
    EMPTY_TAG,
    LOG_PREFIX,
    RING_DEPTH,
    SEGMENT_ORDER,
    DenseProtocolError,
    DeviceOps,
    EventLike,
    LayerLauncher,
    SegmentKind,
    SlotBuffer,
    StreamLike,
    TorchCudaOps,
    ring_slot,
)

if TYPE_CHECKING:
    from .repoint import DenseBindings
    from .store import DenseShardStore
    from .watchdog import Finding, SlotWatchdog

#: One slot: ``(segment, index in the segment's ring)``.
_SlotKey = tuple[SegmentKind, int]


class _Phase(enum.Enum):
    """Lifecycle of :class:`DenseDWDP`."""

    SETUP = "setup"
    """Constructed; setup calls (P3-P5) are valid."""

    ARMED = "armed"
    """Committed (P6); the hooks, priming and recovery are valid."""

    RELEASED = "released"
    """Released; only :meth:`DenseDWDP.release` is valid."""


class DenseDWDP:
    """Slot rings, events and per-layer hooks of dense DWDP; owned by ``DwdpManager``.

    State, per enabled segment and ring slot ``s``: ``ready[seg][s]`` (recorded on
    the copy stream after a gather, waited by compute: RAW), ``consume[seg][s]`` (recorded on
    the compute stream at the end of the read window, waited by the copy stream: WAR),
    ``tag[seg][s]`` (layer last enqueued into the slot, ``EMPTY_TAG`` if none),
    ``opened[seg][s]`` (whether H1 made compute wait for the slot's current holder, which opens a
    read window) and ``consumed[seg][s]`` (whether a consume record closed the holder's latest
    read window).  One ``fence`` event serves demand fetches.  ``primed`` and ``forward_open``
    drive priming.  All events come from ``ops.new_event()`` and are never shared
    with the expert DWDP events.

    Invariants, checked on the host:

    * I1 (RAW): at H1 ``tag[seg][s] == l`` for every segment, else a fenced demand fetch of
      ``L(l)`` runs first; compute then waits ``ready[seg][s]``, whose latest record follows the
      copy of ``L(l)``.
    * I2 (WAR): enqueueing ``L(n)`` into slot ``s`` waits ``consume[seg][s]``, whose latest record
      follows every read of the slot enqueued so far unless the current holder's read window is
      open (``opened`` and not ``consumed``); then the copy also waits the fence (``strict``:
      :class:`~.types.DenseProtocolError`).  A holder whose window never opened has not been
      read, so it needs no fence.
    * I3 (window): consume events are recorded only at H2 (attention) and H3 (MoE-dense), after
      the window's reads are enqueued, and only to close a window that H1 opened: H2 and H3 raise
      :class:`~.types.DenseProtocolError` when the slot holds another layer or its holder's read
      window is not open.
    * I4 (EDF): ``L(n)`` is enqueued at H2 of the layer two positions before ``n``; with the
      expert enqueues the host order is the deadline order.

    Layers are addressed by position ``p`` among the covered layers: the slot is
    ``ring_slot(seg, p)`` and H2 enqueues ``layers[p + 2]`` if it exists.  For Kimi-K3 (layers
    1..92) that is ``(l - 1) % 2``, ``(l - 1) % 3`` and ``l + 2``.

    Lifecycle: constructed in setup P2 (records every consume event once on the current stream),
    :meth:`set_launcher` in P3, :meth:`fetch_blocking` in P4 and P5, :meth:`commit` in P6.  The
    hooks, :meth:`prefetch`, :meth:`mark_primed` and :meth:`war_barrier` are valid only after
    :meth:`commit`; :meth:`set_launcher` and :meth:`fetch_blocking` only before it.
    :meth:`release` is valid at any time; after it, every call named above raises
    :class:`~.types.DenseProtocolError`, while the introspection properties and methods keep
    reporting the last state (:attr:`slots` is empty).  The forward needs eager execution on one
    stream: H0 and H1 refuse a CUDA graph capture and multi-stream execution
    (:meth:`require_eager`) before they record or enqueue anything.

    With a ``watchdog`` (``TRTLLM_DWDP_DENSE_DEBUG=2``), H1 also enqueues on the
    compute stream a check of each slot it waited for against the layer's P1 checksum; H3 of the
    last layer reads the mismatch counters, which synchronizes the host once per forward, and
    logs every ``(layer, segment)`` whose slot held other bytes (``strict``: raises).

    Args:
        layers: Covered layers, ascending; the same list as the expert DWDP MoE layers.
        segments: Enabled segment kinds in canonical order (``ATTN`` first).
        slots: Ring per enabled segment, ``len(slots[seg]) == RING_DEPTH[seg]``; owned by the
            runtime once the constructor returns (if it raises, the caller keeps them).
        copy_stream: The expert DWDP copy stream (``DWDPWeightManager.copy_stream``).
        ops: CUDA seam; ``TorchCudaOps()`` if None.
        strict: ``TRTLLM_DWDP_DENSE_STRICT``: demand fetches, fence fallbacks and watchdog
            mismatches raise.
        store: Row store the launcher reads from; owned and released by the runtime once the
            constructor returns.
        watchdog: Slot checksum watchdog over the same layers and segments; owned and released by
            the runtime once the constructor returns.  None (the default) checks nothing.

    Raises:
        ValueError: Empty, unsorted, duplicate or negative ``layers``; non-canonical
            ``segments``; rings that do not match the enabled segments, or a ring of the wrong
            depth or order; no ``copy_stream``; a ``watchdog`` of other layers or segments.
    """

    def __init__(
        self,
        layers: Sequence[int],
        segments: Sequence[SegmentKind],
        slots: Mapping[SegmentKind, Sequence[SlotBuffer]],
        copy_stream: StreamLike,
        *,
        ops: DeviceOps | None = None,
        strict: bool = False,
        store: DenseShardStore | None = None,
        watchdog: SlotWatchdog | None = None,
    ) -> None:
        covered = _covered_layers(layers)
        kinds = _segment_kinds(segments)
        rings = _slot_rings(kinds, slots)
        if copy_stream is None:
            raise ValueError("dense DWDP needs the expert DWDP copy stream")
        if watchdog is not None and (watchdog.layers, watchdog.segments) != (covered, kinds):
            raise ValueError(
                f"the slot watchdog covers layers {list(watchdog.layers)} and segments "
                f"{[kind.value for kind in watchdog.segments]}, the runtime {list(covered)} and "
                f"{[kind.value for kind in kinds]}"
            )

        self._ops: DeviceOps = TorchCudaOps() if ops is None else ops
        self._layers = covered
        self._positions = {layer: position for position, layer in enumerate(covered)}
        self._segments = kinds
        self._has_moe = SegmentKind.MOE in kinds
        self._targets: tuple[tuple[_SlotKey, ...], ...] = tuple(
            tuple((kind, ring_slot(kind, position)) for kind in kinds)
            for position in range(len(covered))
        )
        self._copy_stream = copy_stream
        self._strict = bool(strict)
        self._slots: dict[SegmentKind, tuple[SlotBuffer, ...]] = rings
        self._ready: dict[SegmentKind, tuple[EventLike, ...]] = {
            kind: tuple(self._ops.new_event() for _ in rings[kind]) for kind in kinds
        }
        self._consume: dict[SegmentKind, tuple[EventLike, ...]] = {
            kind: tuple(self._ops.new_event() for _ in rings[kind]) for kind in kinds
        }
        self._fence: EventLike = self._ops.new_event()
        self._tag: dict[SegmentKind, list[int]] = {
            kind: [EMPTY_TAG] * len(rings[kind]) for kind in kinds
        }
        self._opened: dict[SegmentKind, list[bool]] = {
            kind: [False] * len(rings[kind]) for kind in kinds
        }
        self._consumed: dict[SegmentKind, list[bool]] = {
            kind: [True] * len(rings[kind]) for kind in kinds
        }
        self._primed = False
        self._forward_open = False
        self._demand_fetches = 0
        self._war_fences = 0
        self._launcher: LayerLauncher | None = None
        self._bindings: DenseBindings | None = None
        self._store = store
        self._watchdog = watchdog
        self._record_consume_events()
        # Set last: ``__del__`` releases nothing of an object whose constructor raised.
        self._phase = _Phase.SETUP
        logger.debug(
            f"{LOG_PREFIX} runtime: {len(covered)} layers {covered[0]}..{covered[-1]}, "
            f"segments {','.join(kind.value for kind in kinds)}, strict={int(self._strict)}, "
            f"slot watchdog {'on' if watchdog is not None else 'off'}"
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def layers(self) -> tuple[int, ...]:
        """Covered layers, ascending."""
        return self._layers

    @property
    def segments(self) -> tuple[SegmentKind, ...]:
        """Enabled segment kinds in canonical order."""
        return self._segments

    @property
    def first_layer(self) -> int:
        """First covered layer; H1 of it opens a forward."""
        return self._layers[0]

    @property
    def last_layer(self) -> int:
        """Last covered layer; H3 of it closes a forward."""
        return self._layers[-1]

    @property
    def slots(self) -> Mapping[SegmentKind, tuple[SlotBuffer, ...]]:
        """The slot rings (read-only); empty after :meth:`release`."""
        return MappingProxyType(self._slots)

    @property
    def primed(self) -> bool:
        """Whether ``L(first)`` and ``L(second)`` are enqueued for a forward that has not started."""
        return self._primed

    @property
    def forward_open(self) -> bool:
        """Whether a forward passed H1 of the first layer and has not reached H3 of the last."""
        return self._forward_open

    @property
    def demand_fetches(self) -> int:
        """Fenced demand fetches at H1 so far; 0 after warmup in a healthy run."""
        return self._demand_fetches

    @property
    def war_fences(self) -> int:
        """Prefetches that waited on the fence because a slot holder's read window was open."""
        return self._war_fences

    @property
    def watchdog_checks(self) -> int:
        """Slot checks the watchdog enqueued at H1 so far; 0 without a watchdog."""
        return 0 if self._watchdog is None else self._watchdog.checks

    @property
    def watchdog_mismatches(self) -> int:
        """Slot checks that found other bytes than their layer's, as of the watchdog's latest read.

        The counters are read at H3 of the last covered layer and at :meth:`release`; 0 without a
        watchdog.
        """
        return 0 if self._watchdog is None else self._watchdog.mismatches

    def covers(self, layer: int) -> bool:
        """Whether ``layer`` is covered; the model calls the hooks only for covered layers."""
        return layer in self._positions

    def slot_index(self, segment: SegmentKind, layer: int) -> int:
        """Ring slot of ``layer`` in ``segment``.

        Raises:
            DenseProtocolError: ``layer`` is not covered, or ``segment`` is not enabled.
            ValueError: ``segment`` is not a segment kind.
        """
        kind = self._enabled(segment)
        return ring_slot(kind, self._position(layer))

    def slot_tag(self, segment: SegmentKind, index: int) -> int:
        """Layer last enqueued into slot ``index`` of ``segment``, or ``EMPTY_TAG``.

        Raises:
            DenseProtocolError: ``segment`` is not enabled.
            ValueError: ``segment`` is not a segment kind.
            IndexError: ``index`` is outside the ring.
        """
        kind = self._enabled(segment)
        return self._tag[kind][self._ring_index(kind, index)]

    def slot_consumed(self, segment: SegmentKind, index: int) -> bool:
        """Whether a consume record closed the latest read window of ``segment``'s slot ``index``.

        False from the copy that installs the holder, and again from each H1 that reads it, until
        H2 (attention) or H3 (MoE-dense) records the slot's consume event; True for an empty slot.

        Raises:
            DenseProtocolError: ``segment`` is not enabled.
            ValueError: ``segment`` is not a segment kind.
            IndexError: ``index`` is outside the ring.
        """
        kind = self._enabled(segment)
        return self._consumed[kind][self._ring_index(kind, index)]

    # ------------------------------------------------------------------
    # Setup (P3-P6)
    # ------------------------------------------------------------------

    def set_launcher(self, launcher: LayerLauncher) -> None:
        """Install the copy launcher (P3); the runtime calls it with the copy stream current.

        The runtime drops the launcher at :meth:`release` without calling anything on it.

        Raises:
            DenseProtocolError: Called after :meth:`commit`.
            TypeError: ``launcher`` is not callable.
        """
        self._require_setup("set_launcher")
        if not callable(launcher):
            raise TypeError(f"the dense DWDP launcher must be callable, got {launcher!r}")
        self._launcher = launcher

    def fetch_blocking(self, layer: int) -> None:
        """Gather ``L(layer)`` outside the event protocol, then synchronize (P4 Phase V, P5).

        Launches on the copy stream exactly as the runtime does and calls ``ops.synchronize()``;
        tags, ``consumed`` flags and events are untouched.

        Raises:
            DenseProtocolError: No launcher, ``layer`` not covered, or called after
                :meth:`commit`.
        """
        self._require_setup("fetch_blocking")
        self._position(layer)
        launcher = self._launcher
        if launcher is None:
            raise DenseProtocolError(f"fetch_blocking(L{layer}) before set_launcher")
        with self._ops.stream(self._copy_stream):
            launcher(layer)
        self._ops.synchronize()

    def commit(self, bindings: DenseBindings | None, *, fill_byte: int) -> None:
        """Arm the runtime at the end of setup P6, in this fixed order.

        Phase V and the P5 gathers are finished and synchronized by the caller.  Then: fill every
        slot with ``fill_byte`` (0, or ``DEBUG_FILL_BYTE`` for NaN under debug) -> every tag
        ``EMPTY_TAG``, no read window open and every ``consumed`` True -> record every consume
        event on the current stream -> ``primed`` and ``forward_open`` False.  The fill is
        enqueued on the current stream before the consume records, so the first copy into any
        slot waits for it.  Takes ownership of ``bindings``, which :meth:`release` detaches (None
        when nothing is repointed, as in unit tests).

        Raises:
            DenseProtocolError: No launcher, or called twice.
            ValueError: ``fill_byte`` is not in ``[0, 255]``.
        """
        self._require_setup("commit")
        if self._launcher is None:
            raise DenseProtocolError("commit before set_launcher")
        if not 0 <= fill_byte <= 0xFF:
            raise ValueError(f"the slot fill byte must be in [0, 255], got {fill_byte}")
        for kind in self._segments:
            for slot in self._slots[kind]:
                slot.storage.fill_(fill_byte)
        self._reset_slot_state()
        self._record_consume_events()
        self._primed = False
        self._forward_open = False
        self._bindings = bindings
        self._phase = _Phase.ARMED
        logger.debug(f"{LOG_PREFIX} runtime armed; slots filled with {fill_byte:#04x}")
        if self._watchdog is not None:
            logger.info(
                f"{LOG_PREFIX} slot watchdog on (TRTLLM_DWDP_DENSE_DEBUG=2): every H1 checks its "
                f"slots against the layer's checksum; the counters are read once per forward"
            )

    # ------------------------------------------------------------------
    # Per-layer hooks (compute stream)
    # ------------------------------------------------------------------

    def require_eager(self, hook: str) -> None:
        """Raise unless the current stream runs eagerly with multi-stream execution off.

        The hooks keep host-side slot state and order the copies and the weight reads with
        events they record in every forward.  Inside a CUDA graph capture those records and
        copies would go into the graph instead of running, and the host state would describe
        work that never ran; with multi-stream execution, reads would run on streams that no
        event covers.  H0 (``DwdpManager.ensure_primed``) calls this before it records or
        enqueues anything, and H1 (:meth:`wait_layer`) before it waits: a refused hook records,
        enqueues and changes nothing, so the next eager forward finds the state the refused hook
        found.  Valid in every phase.

        Args:
            hook: The refused call, for the error message.

        Raises:
            DenseProtocolError: The current stream is capturing a CUDA graph, or
                ``ops.do_multi_stream()`` is on.
        """
        if self._ops.is_current_stream_capturing():
            raise DenseProtocolError(
                f"{hook} while the current stream is capturing a CUDA graph; dense DWDP needs "
                f"eager execution"
            )
        if self._ops.do_multi_stream():
            raise DenseProtocolError(
                f"{hook} with multi-stream execution on; dense DWDP covers reads on the compute "
                f"stream only"
            )

    def wait_layer(self, layer: int) -> None:
        """H1, before the layer's first statement: make compute wait for ``L(layer)``.

        Raises :class:`~.types.DenseProtocolError` if the current stream is capturing a CUDA graph
        or ``ops.do_multi_stream()`` is on (:meth:`require_eager`).  If any enabled segment's
        slot is not tagged ``layer`` (not primed, aborted forward, protocol bug), one fenced
        demand fetch of ``L(layer)`` runs first and ``demand_fetches`` grows by one (``strict``:
        raises instead, before anything is enqueued); a warning is logged at the 1st, 2nd, 4th,
        8th, ... demand fetch.  Then compute waits every enabled segment's ``ready`` event, which
        opens the read windows of the slots' holders (``opened`` True, ``consumed`` False until
        H2 and H3; a holder that an earlier forward already read gets a new window).  For the
        first layer it sets ``primed`` False and ``forward_open`` True.  Wrapped in the NVTX
        range ``dwdp.dense.wait/L{layer}``.  With a watchdog, the check of every slot it waited
        for is then enqueued on the current stream, inside the read windows, in the NVTX range
        ``dwdp.dense.watchdog/L{layer}``.

        Raises:
            DenseProtocolError: See above; also before :meth:`commit` or for an uncovered layer.
        """
        self._require_armed("wait_layer")
        position = self._position(layer)
        self.require_eager(f"wait_layer(L{layer})")
        targets = self._targets[position]
        with self._ops.nvtx_range(f"dwdp.dense.wait/L{layer}"):
            stale = [(kind, slot) for kind, slot in targets if self._tag[kind][slot] != layer]
            if stale:
                self._demand_fetch(layer, stale)
            compute = self._ops.current_stream()
            for kind, slot in targets:
                compute.wait_event(self._ready[kind][slot])
            for kind, slot in targets:
                self._opened[kind][slot] = True
                self._consumed[kind][slot] = False
        if position == 0:
            self._primed = False
            self._forward_open = True
        watchdog = self._watchdog
        if watchdog is not None:
            with self._ops.nvtx_range(f"dwdp.dense.watchdog/L{layer}"):
                for kind, slot in targets:
                    watchdog.check(layer, kind, self._slots[kind][slot])

    def attention_done(self, layer: int) -> None:
        """H2, after the attention call: release ``A(layer)``, then enqueue ``L(layer + 2)``.

        Closes the read window that H1 opened for ``A(layer)``: records ``consume[ATTN][s]`` on
        the current stream and marks the slot consumed.  Then, if a covered layer two positions
        ahead exists, :meth:`prefetch`-es it, so the copy stream's wait binds to the record just
        made.

        Raises:
            DenseProtocolError: The attention slot is not tagged ``layer``, or its read window is
                not open because :meth:`wait_layer` (H1) did not open it since ``L(layer)`` was
                enqueued or this hook already closed it (always a protocol bug, regardless of
                ``strict``; raised before anything is recorded); before :meth:`commit`; an
                uncovered layer.
        """
        self._require_armed("attention_done")
        position = self._position(layer)
        slot = ring_slot(SegmentKind.ATTN, position)
        self._close_window(SegmentKind.ATTN, slot, layer, "attention_done")
        ahead = position + 2
        if ahead < len(self._layers):
            self._prefetch(self._layers[ahead], fence=False)

    def layer_done(self, layer: int) -> None:
        """H3, after the MoE block: release ``D(layer)``.

        With the MoE-dense segment enabled, closes the read window that H1 opened for
        ``D(layer)``: records ``consume[MOE][s]`` on the current stream and marks the slot
        consumed.  For the last layer it sets ``forward_open`` False; then, with a watchdog, it
        reads the watchdog's counters (a host synchronization with the current stream) and logs
        an error naming every ``(layer, segment)`` whose slot held other bytes at H1 since the
        previous read.

        Raises:
            DenseProtocolError: The MoE-dense slot is not tagged ``layer``, or its read window is
                not open (as for :meth:`attention_done`); before :meth:`commit`; an uncovered
                layer; ``strict`` and the watchdog found mismatches (raised once the forward is
                closed).
        """
        self._require_armed("layer_done")
        position = self._position(layer)
        if self._has_moe:
            slot = ring_slot(SegmentKind.MOE, position)
            self._close_window(SegmentKind.MOE, slot, layer, "layer_done")
        if position == len(self._layers) - 1:
            self._forward_open = False
            if self._watchdog is not None:
                self._report_watchdog(self._watchdog.poll(), strict=self._strict)

    # ------------------------------------------------------------------
    # Priming and recovery (driven by DwdpManager)
    # ------------------------------------------------------------------

    def prefetch(self, layer: int) -> None:
        """Enqueue ``L(layer)`` (every enabled segment, one launch) on the copy stream.

        Used by H2 and by ``DwdpManager`` priming (``L(first), E(first), L(second), E(second)``).
        Same as ``_prefetch(layer, fence=False)``: a target slot whose holder's read window is
        open gets the fence, recorded on the current stream.  The model runs H1-H3 of every
        covered layer in order, so no window is open while ``forward_open`` is False: every
        window closes by H3 of the last layer, and :meth:`war_barrier` resets them all.  Priming
        then needs no fence, whichever stream it runs on; slots left by an interrupted priming
        hold layers that nobody has read and are refilled without one.

        Raises:
            DenseProtocolError: Before :meth:`commit`; an uncovered layer; ``strict`` and a target
                slot's holder has an open read window.
        """
        self._require_armed("prefetch")
        self._prefetch(layer, fence=False)

    def mark_primed(self) -> None:
        """Record that ``DwdpManager`` enqueued the first two layers (dense and expert).

        Raises:
            DenseProtocolError: Before :meth:`commit`; a forward is open (run
                :meth:`war_barrier` first); the slots of the first two covered layers are not
                tagged with them, i.e. :meth:`prefetch` did not enqueue them or a forward has
                consumed them since.
        """
        self._require_armed("mark_primed")
        if self._forward_open:
            raise DenseProtocolError(
                "mark_primed while a forward is open; war_barrier must run before re-priming"
            )
        missing = [
            f"L({layer}) {kind.value}[{slot}] holds {self._holder(kind, slot)}"
            for position, layer in enumerate(self._layers[:2])
            for kind, slot in self._targets[position]
            if self._tag[kind][slot] != layer
        ]
        if missing:
            raise DenseProtocolError(
                f"mark_primed but the first layers are not enqueued: {'; '.join(missing)}"
            )
        self._primed = True

    def war_barrier(self) -> None:
        """Recover after a forward aborted between H1 of the first and H3 of the last layer.

        Records every consume event on the current stream (the execution stream, inside the
        model's H0), so later copies wait for every read the dead forward enqueued; then every tag
        ``EMPTY_TAG``, no read window open, every ``consumed`` True, ``primed`` and
        ``forward_open`` False.  Copies the dead forward left on the copy stream still run first
        (FIFO), and the next ready records follow them.  ``DwdpManager`` re-records the two expert
        consume events alongside.

        Raises:
            DenseProtocolError: Before :meth:`commit`.
        """
        self._require_armed("war_barrier")
        self._record_consume_events()
        self._reset_slot_state()
        self._primed = False
        self._forward_open = False
        logger.info(f"{LOG_PREFIX} WAR barrier after an aborted forward; slots reset")

    def _prefetch(self, layer: int, *, fence: bool) -> None:
        """Enqueue ``L(layer)``: the one place that launches a copy under the event protocol.

        With ``compute = ops.current_stream()`` taken first and the copy stream current: for each
        enabled segment's target slot, fall back to the fence if its holder's read window is open
        (``war_fences`` grows; ``strict``: raise before anything is enqueued), and make the copy
        stream wait its consume event; if fencing, record the fence on ``compute`` and make the
        copy stream wait it; call the launcher once; record each target slot's ready event on the
        copy stream, set its tag to ``layer`` and mark the new holder neither opened nor consumed.
        If the launcher raises, the target slots are tagged ``EMPTY_TAG`` (their bytes are
        unknown).  Wrapped in the NVTX range ``dwdp.dense.prefetch/L{layer}``.
        """
        targets = self._targets[self._position(layer)]
        reading = [
            (kind, slot)
            for kind, slot in targets
            if self._opened[kind][slot] and not self._consumed[kind][slot]
        ]
        if reading and not fence:
            self._fall_back_to_fence(layer, reading)
            fence = True
        launcher = self._launcher
        if launcher is None:
            raise DenseProtocolError(f"prefetch(L{layer}) without a launcher")
        compute = self._ops.current_stream()
        copy_stream = self._copy_stream
        with self._ops.nvtx_range(f"dwdp.dense.prefetch/L{layer}"):
            with self._ops.stream(copy_stream):
                for kind, slot in targets:
                    copy_stream.wait_event(self._consume[kind][slot])
                if fence:
                    self._fence.record(compute)
                    copy_stream.wait_event(self._fence)
                try:
                    launcher(layer)
                except BaseException:
                    for kind, slot in targets:
                        self._tag[kind][slot] = EMPTY_TAG
                    raise
                for kind, slot in targets:
                    self._ready[kind][slot].record(copy_stream)
                    self._tag[kind][slot] = layer
                    self._opened[kind][slot] = False
                    self._consumed[kind][slot] = False

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def release(self) -> None:
        """Release everything the runtime owns; idempotent.

        Order: ``ops.synchronize()`` -> the watchdog's last read (its mismatches are logged, never
        raised) and its release -> detach the bindings (zero-size tensors) -> drop the launcher
        and its plans -> drop the slots -> ``store.release()`` (peer mappings and imports, then the
        local store).  Every step runs even if an earlier one fails; the first error is raised at
        the end and later ones are logged.  ``DwdpManager.cleanup`` calls it before the expert
        teardown.
        """
        if self._phase is _Phase.RELEASED:
            return
        armed = self._phase is _Phase.ARMED
        self._phase = _Phase.RELEASED
        bindings, self._bindings = self._bindings, None
        store, self._store = self._store, None
        failures: list[tuple[str, Exception]] = []
        try:
            self._ops.synchronize()
        except Exception as exc:  # every teardown step runs; the first failure is raised below
            failures.append(("synchronize", exc))
        watchdog = self._watchdog
        if watchdog is not None:
            try:
                self._report_watchdog(watchdog.poll(), strict=False)
                if armed:
                    logger.info(
                        f"{LOG_PREFIX} slot watchdog: {watchdog.checks} slot checks, "
                        f"{watchdog.mismatches} mismatches"
                    )
            except Exception as exc:  # every teardown step runs; the first failure is raised below
                failures.append(("read the slot watchdog", exc))
            watchdog.release()
        if bindings is not None:
            try:
                bindings.detach()
            except Exception as exc:  # every teardown step runs; the first failure is raised below
                failures.append(("detach the module bindings", exc))
        self._launcher = None
        self._slots = {}
        if store is not None:
            try:
                store.release()
            except Exception as exc:  # every teardown step runs; the first failure is raised below
                failures.append(("release the row store", exc))
        if failures:
            for step, exc in failures[1:]:
                logger.error(f"{LOG_PREFIX} release: {step} also failed: {exc!r}")
            raise failures[0][1]
        logger.debug(f"{LOG_PREFIX} runtime released")

    def __del__(self) -> None:
        """Best-effort :meth:`release` for owners that miss the explicit teardown.

        Must tolerate an object whose ``__init__`` raised part way, and must never raise, also
        at interpreter exit, when this module's globals may already be None: the released check
        reaches the phase enum through the phase itself, and the failure log is skipped once the
        logger is gone.
        """
        phase = getattr(self, "_phase", None)
        if phase is None or phase is type(phase).RELEASED:
            return
        try:
            self.release()
        except Exception as exc:  # a finalizer must not raise
            if logger is not None:
                logger.debug(f"{LOG_PREFIX} release from __del__ failed: {exc!r}")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _require_setup(self, action: str) -> None:
        if self._phase is not _Phase.SETUP:
            raise DenseProtocolError(f"dense DWDP {action} called after {self._phase_text()}")

    def _require_armed(self, action: str) -> None:
        if self._phase is _Phase.ARMED:
            return
        if self._phase is _Phase.SETUP:
            raise DenseProtocolError(f"dense DWDP {action} called before commit")
        raise DenseProtocolError(f"dense DWDP {action} called after release")

    def _phase_text(self) -> str:
        return "commit" if self._phase is _Phase.ARMED else "release"

    def _position(self, layer: int) -> int:
        position = self._positions.get(layer)
        if position is None:
            raise DenseProtocolError(
                f"layer {layer} is not covered by dense DWDP ({len(self._layers)} covered layers, "
                f"{self._layers[0]}..{self._layers[-1]})"
            )
        return position

    def _enabled(self, segment: SegmentKind) -> SegmentKind:
        kind = SegmentKind(segment)
        if kind not in self._tag:
            raise DenseProtocolError(f"segment {kind.value} is not enabled")
        return kind

    def _ring_index(self, kind: SegmentKind, index: int) -> int:
        if not 0 <= index < len(self._tag[kind]):
            raise IndexError(f"the {kind.value} ring has {len(self._tag[kind])} slots, got {index}")
        return index

    def _holder(self, kind: SegmentKind, slot: int) -> str:
        tag = self._tag[kind][slot]
        return "nothing" if tag == EMPTY_TAG else f"layer {tag}"

    def _close_window(self, kind: SegmentKind, slot: int, layer: int, hook: str) -> None:
        """End the read window of ``layer`` in ``kind``'s slot ``slot``; ``hook`` is the caller.

        Records the slot's consume event on the current stream and marks the slot consumed, after
        checking that the slot holds ``layer`` and that the holder's read window is open: opened
        by H1 and not closed since.  A failed check raises before anything is recorded.
        """
        if self._tag[kind][slot] != layer:
            raise DenseProtocolError(
                f"{hook}(L{layer}): {kind.value} slot {slot} holds {self._holder(kind, slot)}: "
                f"the slot was refilled inside the read window of layer {layer}, or H1 "
                f"(wait_layer) of layer {layer} did not run"
            )
        if not self._opened[kind][slot]:
            raise DenseProtocolError(
                f"{hook}(L{layer}): H1 (wait_layer) of layer {layer} did not open the read window "
                f"of {kind.value} slot {slot} since L({layer}) was enqueued, so the layer's reads "
                f"did not wait for its copy"
            )
        if self._consumed[kind][slot]:
            raise DenseProtocolError(
                f"{hook}(L{layer}): the read window of layer {layer} in {kind.value} slot {slot} "
                f"is already closed; {hook} ran again without H1 (wait_layer) of layer {layer}"
            )
        self._consume[kind][slot].record(self._ops.current_stream())
        self._consumed[kind][slot] = True

    def _report_watchdog(self, findings: Sequence[Finding], *, strict: bool) -> None:
        """Log the watchdog's new mismatches as an error; ``strict``: raise them instead."""
        if not findings:
            return
        total = sum(count for _, _, count in findings)
        message = (
            f"slot watchdog: {total} slot checks at H1 found other bytes than their layer's "
            f"checksum ({self._watchdog.describe(findings)}; {self._watchdog.mismatches} so far): "
            f"a copy or a stream wait of dense DWDP is broken, or its row store changed"
        )
        if strict:
            raise DenseProtocolError(message)
        logger.error(f"{LOG_PREFIX} {message}")

    def _record_consume_events(self) -> None:
        stream = self._ops.current_stream()
        for kind in self._segments:
            for event in self._consume[kind]:
                event.record(stream)

    def _reset_slot_state(self) -> None:
        for kind in self._segments:
            depth = len(self._tag[kind])
            self._tag[kind][:] = [EMPTY_TAG] * depth
            self._opened[kind][:] = [False] * depth
            self._consumed[kind][:] = [True] * depth

    def _demand_fetch(self, layer: int, stale: Sequence[_SlotKey]) -> None:
        held = ", ".join(
            f"{kind.value}[{slot}] holds {self._holder(kind, slot)}" for kind, slot in stale
        )
        if self._strict:
            raise DenseProtocolError(
                f"wait_layer(L{layer}): {held}; L({layer}) was not enqueued ahead and "
                f"TRTLLM_DWDP_DENSE_STRICT forbids demand fetches"
            )
        self._demand_fetches += 1
        _warn_sparsely(
            f"wait_layer(L{layer}): {held}; running a fenced demand fetch "
            f"(demand fetch #{self._demand_fetches})",
            self._demand_fetches,
        )
        self._prefetch(layer, fence=True)

    def _fall_back_to_fence(self, layer: int, reading: Sequence[_SlotKey]) -> None:
        held = ", ".join(
            f"{kind.value}[{slot}] holds {self._holder(kind, slot)}" for kind, slot in reading
        )
        if self._strict:
            raise DenseProtocolError(
                f"prefetch(L{layer}): {held}, whose read window opened at H1 and has no consume "
                f"record; TRTLLM_DWDP_DENSE_STRICT forbids the fence fallback"
            )
        self._war_fences += 1
        _warn_sparsely(
            f"prefetch(L{layer}): {held}, whose read window opened at H1 and has no consume "
            f"record; the copy waits for all work on the current stream "
            f"(fence fallback #{self._war_fences})",
            self._war_fences,
        )


def _warn_sparsely(message: str, count: int) -> None:
    """Log the 1st, 2nd, 4th, 8th, ... occurrence, so a systematic fault cannot flood the log."""
    if count & (count - 1) == 0:
        logger.warning(f"{LOG_PREFIX} {message}")


def _covered_layers(layers: Sequence[int]) -> tuple[int, ...]:
    covered = tuple(operator.index(layer) for layer in layers)
    if not covered:
        raise ValueError("dense DWDP needs at least one covered layer")
    if any(later <= earlier for earlier, later in zip(covered, covered[1:])):
        raise ValueError(f"covered layers must be strictly ascending, got {list(covered)}")
    if covered[0] < 0:
        raise ValueError(f"covered layers must be non-negative, got {list(covered)}")
    return covered


def _segment_kinds(segments: Sequence[SegmentKind]) -> tuple[SegmentKind, ...]:
    kinds = tuple(SegmentKind(kind) for kind in segments)
    canonical = tuple(kind for kind in SEGMENT_ORDER if kind in kinds)
    if not kinds or kinds != canonical or kinds[0] is not SegmentKind.ATTN:
        raise ValueError(
            f"segments must be attn, optionally followed by moe, got "
            f"{[kind.value for kind in kinds]}"
        )
    return kinds


def _slot_rings(
    kinds: Sequence[SegmentKind], slots: Mapping[SegmentKind, Sequence[SlotBuffer]]
) -> dict[SegmentKind, tuple[SlotBuffer, ...]]:
    rings = {SegmentKind(kind): tuple(ring) for kind, ring in slots.items()}
    if set(rings) != set(kinds):
        raise ValueError(
            f"slot rings must match the enabled segments {[kind.value for kind in kinds]}, "
            f"got rings for {sorted(kind.value for kind in rings)}"
        )
    for kind in kinds:
        ring = rings[kind]
        if len(ring) != RING_DEPTH[kind]:
            raise ValueError(
                f"the {kind.value} ring needs {RING_DEPTH[kind]} slots, got {len(ring)}"
            )
        for index, slot in enumerate(ring):
            if slot.segment != kind or slot.index != index:
                raise ValueError(
                    f"position {index} of the {kind.value} ring holds slot "
                    f"{SegmentKind(slot.segment).value}[{slot.index}]"
                )
    return {kind: rings[kind] for kind in kinds}
