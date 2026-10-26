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
"""CPU tests of the dense DWDP runtime protocol (``runtime.DenseDWDP``) beside expert DWDP.

A recording fake of the CUDA seam logs every event record, stream wait, copy launch and device
synchronization in host order; a wait binds to the latest record of its event before it, as
``cudaStreamWaitEvent`` does, and a device synchronization orders everything after it behind
everything before it.  The runtime runs on the fake through its ``ops`` seam; the real
``DWDPWeightManager``, with its ``PrefetchSchedule``, runs on it through patched ``torch.cuda``
calls and shares the copy stream.  A driver calls them the way the executor loop
(``DwdpManager.start_forward`` on the default stream), ``KimiLinearModel`` (H0-H3, on the
execution stream) and ``ConfigurableMoE`` (``wait_and_bind``, ``finish_layer``) do, and logs the
model's weight reads.  The tests check where the runtime records its events, the copy order on the
shared copy stream, that the host synchronizes only to recover a forward abandoned inside a MoE
layer and, with vector clocks over the log, that no copy into a dense or expert slot can overlap a
read of that slot: a missing consume wait, fence, WAR barrier or synchronization fails them.
"""

from __future__ import annotations

import contextlib
import gc
import random
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from unittest.mock import patch

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.modules.dwdp.dense.runtime import DenseDWDP
from tensorrt_llm._torch.modules.dwdp.dense.types import (
    EMPTY_TAG,
    RING_DEPTH,
    ROW_ALIGN,
    SegmentKind,
    SlotSpec,
    allocate_slot_ring,
)
from tensorrt_llm._torch.modules.dwdp.weight_manager import DWDPWeightManager
from tensorrt_llm._torch.pyexecutor.dwdp import DwdpManager

pytestmark = pytest.mark.cpu_only

ATTN = SegmentKind.ATTN
MOE = SegmentKind.MOE
BOTH = (ATTN, MOE)
SEGMENT_SETS = pytest.mark.parametrize("segments", [BOTH, (ATTN,)], ids=["attn+moe", "attn"])
K3_LAYERS = tuple(range(1, 93))
#: What a launch or a read names: the dense segments, and the expert weights.
EXPERT = "E"
_WEIGHT_MANAGER = "tensorrt_llm._torch.modules.dwdp.weight_manager"

# ---------------------------------------------------------------------------
# Recording fake of the CUDA seam
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class Entry:
    """One log entry, in host order.

    ``kind`` is ``record``, ``wait``, ``launch`` or ``read`` (stream operations), ``sync`` (a
    device synchronization) or ``mark`` (host-side).  ``bound`` is the log index of the record a
    ``wait`` binds to (None: the event was never recorded).  ``what`` is ``L``/``E`` for a
    launch, the segment or ``E`` for a read and the hook for a mark.
    """

    kind: str
    stream: FakeStream | None = None
    event: FakeEvent | None = None
    bound: int | None = None
    what: str = ""
    layer: int | None = None


class FakeStream:
    def __init__(self, log: list[Entry], name: str) -> None:
        self.log = log
        self.name = name

    def wait_event(self, event: FakeEvent) -> None:
        self.log.append(Entry("wait", stream=self, event=event, bound=event.last_record))


class FakeEvent:
    """An event; ``side`` tells the dense runtime's events from the weight manager's."""

    def __init__(self, ops: FakeOps, side: str) -> None:
        self.ops = ops
        self.side = side
        self.last_record: int | None = None

    def record(self, stream: FakeStream | None = None) -> None:
        self.last_record = len(self.ops.log)
        stream = self.ops.current if stream is None else stream
        self.ops.log.append(Entry("record", stream=stream, event=self))


class FakeOps:
    """``DeviceOps`` that logs every stream operation in host order.

    Three streams: ``default`` (the executor loop and setup), ``compute`` (the execution stream
    the model runs on) and ``copy``.
    """

    def __init__(self) -> None:
        self.log: list[Entry] = []
        self.default = FakeStream(self.log, "default")
        self.compute = FakeStream(self.log, "compute")
        self.copy = FakeStream(self.log, "copy")
        self.current = self.default

    def new_event(self) -> FakeEvent:
        return FakeEvent(self, "dense")

    def current_stream(self) -> FakeStream:
        return self.current

    @contextlib.contextmanager
    def stream(self, stream: FakeStream) -> Iterator[FakeStream]:
        previous, self.current = self.current, stream
        try:
            yield stream
        finally:
            self.current = previous

    def synchronize(self) -> None:
        self.log.append(Entry("sync"))

    def is_current_stream_capturing(self) -> bool:
        return False

    def do_multi_stream(self) -> bool:
        return False

    def nvtx_range(self, message: str) -> contextlib.AbstractContextManager:
        return contextlib.nullcontext()


class _SlotMap:
    """The ``WeightBuffer`` calls ``DWDPWeightManager`` makes: MoE order ``o`` uses slot ``o % k``."""

    device_id = 0
    local_start = 0
    local_end = 1

    def __init__(self, layers: tuple[int, ...], num_slots: int) -> None:
        self._layers = layers
        self.num_slots = num_slots

    def buffer_index_for_layer(self, layer: int) -> int:
        return self._layers.index(layer) % self.num_slots

    def get_full_tensor(self, layer: int, name: str) -> torch.Tensor:
        return torch.empty(0)


# ---------------------------------------------------------------------------
# Happens-before over the log
# ---------------------------------------------------------------------------


def vector_clocks(log: list[Entry]) -> dict[int, dict[FakeStream, int]]:
    """Vector clock of every stream operation: ``clock[s]`` = operations of ``s`` it follows.

    Streams run their operations in enqueue order; a wait makes the waiting stream follow every
    operation its bound record follows, the record included; a device synchronization makes every
    later operation follow every earlier one.
    """
    stream_clocks: dict[FakeStream, dict[FakeStream, int]] = {}
    clocks: dict[int, dict[FakeStream, int]] = {}
    floor: dict[FakeStream, int] = {}
    for index, entry in enumerate(log):
        if entry.kind == "sync":
            for clock in stream_clocks.values():
                for stream, count in clock.items():
                    floor[stream] = max(floor.get(stream, 0), count)
            for stream in stream_clocks:
                stream_clocks[stream] = dict(floor)
            continue
        if entry.stream is None:
            continue
        clock = stream_clocks.setdefault(entry.stream, dict(floor))
        if entry.kind == "wait":
            assert entry.bound is not None, f"wait #{index} binds to an event never recorded"
            for stream, count in clocks[entry.bound].items():
                clock[stream] = max(clock.get(stream, 0), count)
        clock[entry.stream] = clock.get(entry.stream, 0) + 1
        clocks[index] = dict(clock)
    return clocks


def happens_before(
    log: list[Entry], clocks: dict[int, dict[FakeStream, int]], first: int, second: int
) -> bool:
    """Whether stream operation ``first`` completes before stream operation ``second`` starts."""
    stream = log[first].stream
    return first != second and clocks[second].get(stream, 0) >= clocks[first][stream]


# ---------------------------------------------------------------------------
# Driver: the executor loop, the model, ConfigurableMoE and DwdpManager around the runtime
# ---------------------------------------------------------------------------


class Abort(Exception):
    """A forward that stops midway, as one whose kernel launch or collective raised."""


class LaunchError(Exception):
    """A dense copy launch that raised."""


class Driver:
    """Calls the runtime and the weight manager in program order; logs reads and copy launches.

    Every call is wrapped in a span keyed ``(forward, hook, layer)``; a ``mark`` entry opens it, so
    each log entry can be attributed to the call that made it.

    Args:
        segments: Enabled dense segments.
        layers: Covered layers, which are also the expert DWDP MoE layers.
        num_slots: Expert prefetch slots (``DwdpConfig.num_prefetch_slots``).
    """

    def __init__(
        self,
        segments: tuple[SegmentKind, ...] = BOTH,
        *,
        layers: tuple[int, ...] = K3_LAYERS,
        num_slots: int = 2,
    ) -> None:
        self.ops = FakeOps()
        self.log = self.ops.log
        self.segments = segments
        self.layers = tuple(layers)
        self.num_slots = num_slots
        self.positions = {layer: position for position, layer in enumerate(self.layers)}
        self.forward_index = 0
        self.spans: dict[tuple[int, str, int | None], tuple[int, int]] = {}
        # Set to make the next dense launch raise, as a failing copy launch does.
        self.fail_dense_launch = False
        spec = {kind: SlotSpec(kind, RING_DEPTH[kind], ROW_ALIGN, 2) for kind in segments}
        slots = {kind: allocate_slot_ring(spec[kind], "cpu") for kind in segments}
        self.mark("construct", None)
        self.runtime = DenseDWDP(self.layers, segments, slots, self.ops.copy, ops=self.ops)
        self.runtime.set_launcher(self.launch_dense)
        self.span("commit", None, lambda: self.runtime.commit(None))
        with self.cuda():
            self.weight_manager = DWDPWeightManager(
                weight_buffer=_SlotMap(self.layers, num_slots),
                peer_views={},
                peer_ranges=[(0, 1), (1, 2)],
                moe_layer_indices=list(self.layers),
                weight_names=["w"],
                dwdp_rank=0,
                dwdp_size=2,
            )
        # Nothing to release: no weight buffer, transport or device behind the fake.
        self.weight_manager._released = True
        self.manager = object.__new__(DwdpManager)
        self.manager._weight_manager = self.weight_manager
        self.manager._dense = self.runtime
        self.backend = nn.Module()
        self.backend.w = nn.Parameter(torch.empty(0), requires_grad=False)
        self.step_start = FakeEvent(self.ops, "executor")
        self.step_end = FakeEvent(self.ops, "executor")

    # -- logging -------------------------------------------------------------

    def mark(self, hook: str, layer: int | None) -> None:
        self.log.append(Entry("mark", what=hook, layer=layer))

    def span(self, hook: str, layer: int | None, action: Callable[[], object]) -> None:
        start = len(self.log)
        self.mark(hook, layer)
        try:
            action()
        finally:
            self.spans[(self.forward_index, hook, layer)] = (start, len(self.log))

    def launch(self, what: str, layer: int) -> None:
        self.log.append(Entry("launch", stream=self.ops.current, what=what, layer=layer))

    def launch_dense(self, layer: int) -> None:
        """The dense ``LayerLauncher``: one launch on the current stream."""
        if self.fail_dense_launch:
            self.fail_dense_launch = False
            raise LaunchError(f"L({layer})")
        self.launch("L", layer)

    def read(self, what: str, layer: int) -> None:
        """A kernel that reads ``what``'s weights of ``layer`` on the current stream."""
        self.log.append(Entry("read", stream=self.ops.current, what=what, layer=layer))

    @contextlib.contextmanager
    def cuda(self) -> Iterator[None]:
        """``torch.cuda`` as the weight manager sees it: the fake seam, and its copies logged.

        The patches are process-wide, so the cyclic garbage collector is paused meanwhile: the
        finalizer of another test's weight manager or runtime would call them and log a device
        synchronization here.
        """
        ops = self.ops
        fakes = {
            "Stream": lambda device=None: ops.copy,
            "Event": lambda: FakeEvent(ops, "expert"),
            "current_stream": lambda device=None: ops.current,
            "stream": ops.stream,
            "synchronize": lambda device=None: ops.synchronize(),
        }
        with contextlib.ExitStack() as stack:
            if gc.isenabled():
                gc.disable()
                stack.callback(gc.enable)
            for name, fake in fakes.items():
                stack.enter_context(patch(f"{_WEIGHT_MANAGER}.torch.cuda.{name}", fake))
            stack.enter_context(
                patch.object(
                    DWDPWeightManager,
                    "_prefetch_layer_per_slice",
                    lambda _manager, layer: self.launch(EXPERT, layer),
                )
            )
            yield

    # -- executor loop, model and ConfigurableMoE ---------------------------------

    def start(self) -> None:
        """The executor loop before a forward: ``DwdpManager.start_forward``, default stream."""
        with self.cuda(), self.ops.stream(self.ops.default):
            self.span("start", None, self.manager.start_forward)

    def h0(self) -> None:
        """The model's forward entry: ``DwdpManager.ensure_primed``."""
        self.manager.ensure_primed()

    def invalidate(self) -> None:
        """The executor after a control action: ``DwdpManager.invalidate_prefetch``."""
        with self.cuda(), self.ops.stream(self.ops.default):
            self.span("invalidate", None, self.manager.invalidate_prefetch)

    def layer(self, layer: int, stop: str | None = None) -> None:
        """One decoder layer in program order; raises :class:`Abort` right before ``stop``.

        ``stop`` is ``H1``, ``H2``, ``finish`` or ``H3``; ``fill`` makes the first dense copy
        launch of the layer's hooks raise instead.
        """
        runtime = self.runtime
        if stop == "fill":
            self.fail_dense_launch = True
        try:
            if stop == "H1":
                raise Abort
            self.span("H1", layer, lambda: runtime.wait_layer(layer))
            self.read(ATTN.value, layer)
            if stop == "H2":
                raise Abort
            self.span("H2", layer, lambda: runtime.attention_done(layer))
            if MOE in self.segments:
                self.read(MOE.value, layer)  # the router and the latent down-projection
            self.span("bind", layer, lambda: self.manager.wait_and_bind(self.backend, layer))
            self.read(EXPERT, layer)
            if stop == "finish":
                raise Abort
            self.span("finish", layer, lambda: self.manager.finish_layer(layer))
            if MOE in self.segments:
                self.read(MOE.value, layer)  # the latent up-projection and the shared expert
            if stop == "H3":
                raise Abort
            self.span("H3", layer, lambda: runtime.layer_done(layer))
        finally:
            self.fail_dense_launch = False

    def forward(
        self,
        *,
        start: bool = True,
        h0: bool = True,
        stop_at: tuple[str, int] | None = None,
        fail_h0: bool = False,
    ) -> bool:
        """One executor step; False if the forward stopped midway.

        ``start``: the executor loop starts the forward (the non-overlap loop; warmup and the
        KV-cache estimation do not).  ``h0``: the model's forward entry runs H0.
        ``stop_at=(stop, layer)``: the forward stops in ``layer`` (see :meth:`layer`).
        ``fail_h0``: the first dense copy launch of H0 raises.  The execution stream joins the
        default stream after a forward that completes, as ``PyExecutor._forward_step`` does.
        """
        if start:
            self.start()
        ops = self.ops
        try:
            with self.cuda():
                self.step_start.record(ops.default)
                ops.compute.wait_event(self.step_start)
                with ops.stream(ops.compute):
                    if h0:
                        self.fail_dense_launch = fail_h0
                        try:
                            self.span("H0", None, self.h0)
                        finally:
                            self.fail_dense_launch = False
                    for layer in self.layers:
                        stop = stop_at[0] if stop_at is not None and stop_at[1] == layer else None
                        self.layer(layer, stop)
                    self.step_end.record(ops.compute)
                ops.default.wait_event(self.step_end)
        except (Abort, LaunchError):
            return False
        finally:
            self.forward_index += 1
        return True

    # -- analysis --------------------------------------------------------------

    def slots_of(self, what: str, layer: int) -> list[tuple[str, int]]:
        """The slots a launch of ``what`` writes, or a read of ``what`` reads, for ``layer``."""
        position = self.positions[layer]
        if what == EXPERT:
            return [(EXPERT, position % self.num_slots)]
        kinds = self.segments if what == "L" else (SegmentKind(what),)
        return [(kind.value, position % RING_DEPTH[kind]) for kind in kinds]

    def assert_race_free(self) -> int:
        """Check every logged read against every copy into its slot; returns the reads checked.

        RAW: the latest copy of the read layer into the slot completes before the read.  WAR: the
        next copy into the slot starts after the read completes; copies into one slot run on one
        stream in enqueue order, so every later copy does too.
        """
        log = self.log
        clocks = vector_clocks(log)
        writes: dict[tuple[str, int], list[tuple[int, int]]] = {}
        for index, entry in enumerate(log):
            if entry.kind == "launch":
                for key in self.slots_of(entry.what, entry.layer):
                    writes.setdefault(key, []).append((index, entry.layer))
        for key, slot_writes in writes.items():
            assert len({log[index].stream for index, _ in slot_writes}) == 1, key
        reads = 0
        for index, entry in enumerate(log):
            if entry.kind != "read":
                continue
            (key,) = self.slots_of(entry.what, entry.layer)
            slot_writes = writes.get(key, [])
            earlier = [(w, layer) for w, layer in slot_writes if w < index]
            sources = [w for w, layer in earlier if layer == entry.layer]
            assert sources, f"read #{index} of {entry.what}({entry.layer}) precedes its copy"
            source = sources[-1]
            assert happens_before(log, clocks, source, index), (
                f"RAW: read #{index} of {entry.what}({entry.layer}) may start before copy #{source}"
            )
            later = [(w, layer) for w, layer in slot_writes if w > source]
            if later:
                w, layer = later[0]
                assert happens_before(log, clocks, index, w), (
                    f"WAR: copy #{w} of {layer} into {key} may start before read #{index} of "
                    f"{entry.what}({entry.layer}) completes"
                )
            reads += 1
        return reads

    def syncs(self, start: int = 0, end: int | None = None) -> list[int]:
        """Log indices of the device synchronizations in ``[start, end)``."""
        stop = len(self.log) if end is None else end
        return [i for i in range(start, stop) if self.log[i].kind == "sync"]

    def assert_no_host_sync(self) -> None:
        """Check that the host never synchronized after commit, so it can run ahead of the GPU."""
        committed = self.spans[(0, "commit", None)][1]
        assert self.syncs(committed) == []

    def records(self, start: int, end: int, side: str, stream: FakeStream | None = None) -> list:
        """Log indices of the records of ``side``'s events in ``[start, end)``, on ``stream``."""
        return [
            index
            for index in range(start, end)
            if self.log[index].kind == "record"
            and self.log[index].event.side == side
            and (stream is None or self.log[index].stream is stream)
        ]

    def launches(
        self, start: int = 0, end: int | None = None, what: str | None = None
    ) -> list[tuple[str, int]]:
        """``(what, layer)`` of the copy launches in ``[start, end)``."""
        return [
            (entry.what, entry.layer)
            for entry in self.log[start:end]
            if entry.kind == "launch" and (what is None or entry.what == what)
        ]

    def launches_in(self, key: tuple[int, str, int | None]) -> list[tuple[str, int]]:
        """The copy launches made inside the span ``key``."""
        start, end = self.spans[key]
        return self.launches(start, end)

    def forward_bounds(self, forward: int) -> tuple[int, int]:
        """Log range of forward ``forward``: from its first span to the next forward's first."""
        starts = {}
        for (index, _hook, _layer), (start, _end) in self.spans.items():
            starts[index] = min(start, starts.get(index, start))
        end = starts.get(forward + 1, len(self.log))
        return starts[forward], end

    def stream_ops(self, start: int, end: int | None = None) -> list[Entry]:
        return [e for e in self.log[start:end] if e.kind in ("record", "wait", "launch")]


def _check_steady(driver: Driver) -> None:
    """No fence, demand fetch or host synchronization so far, and every read race-free."""
    assert driver.runtime.war_fences == 0
    assert driver.runtime.demand_fetches == 0
    driver.assert_no_host_sync()
    driver.assert_race_free()


# ---------------------------------------------------------------------------
# Two complete Kimi-K3 forwards
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", params=[BOTH, (ATTN,)], ids=["attn+moe", "attn"])
def two_forwards(request: pytest.FixtureRequest) -> Driver:
    """Two complete K3 forwards (92 layers each), started by the executor; tests only read it."""
    driver = Driver(segments=request.param)
    assert driver.forward() and driver.forward()
    return driver


def test_read_windows_end_at_h2_and_h3_whose_fill_runs_into_the_next_forward(
    two_forwards: Driver,
) -> None:
    driver = two_forwards
    log = driver.log
    has_moe = MOE in driver.segments
    window_ends = ("H2", "H3") if has_moe else ("H2",)
    # Where the fill enqueues the next forward's first layers: H3 of the last two layers, whose
    # MoE-dense windows hold those layers' slots until then; H2 with the attention segment alone.
    boundary = "H3" if has_moe else "H2"
    for forward in range(2):
        for layer in driver.layers:
            for hook in window_ends:
                start, end = driver.spans[(forward, hook, layer)]
                ops = driver.stream_ops(start, end)
                # The hook ends the window first, so each copy it enqueues binds to that record.
                assert ops[0].kind == "record" and ops[0].stream is driver.ops.compute
                consume = driver.records(start, end, "dense", driver.ops.compute)
                assert len(consume) == 1, (forward, hook, layer)
            expected = {"H2": [], "H3": []}
            if layer + 2 <= 92:
                expected["H2"] = [("L", layer + 2)]
            else:
                expected[boundary] = [("L", layer - 90)]
            for hook in window_ends:
                assert driver.launches_in((forward, hook, layer)) == expected[hook], (
                    forward,
                    hook,
                    layer,
                )
    # Those are the only dense compute records after commit: no other hook ends a read window.
    committed = driver.spans[(0, "commit", None)][1]
    records = driver.records(committed, len(log), "dense", driver.ops.compute)
    assert len(records) == 2 * len(driver.layers) * len(window_ends)
    # The expert slots are released right after each MoE, on the execution stream.
    finishes = driver.records(committed, len(log), "expert", driver.ops.compute)
    assert len(finishes) == 2 * len(driver.layers)
    # The first two layers: primed by the first forward's start, then by the end of each forward.
    for what in ("L", EXPERT):
        copies = Counter(layer for kind, layer in driver.launches() if kind == what)
        assert copies == Counter({layer: 3 if layer <= 2 else 2 for layer in driver.layers})
    assert not driver.runtime.forward_open and driver.runtime.primed
    reads_per_layer = (3 if has_moe else 1) + 1
    assert driver.assert_race_free() == 2 * len(driver.layers) * reads_per_layer
    _check_steady(driver)


# ---------------------------------------------------------------------------
# Copy order on the shared copy stream
# ---------------------------------------------------------------------------


def copy_order_problems(driver: Driver) -> list[str]:
    """How two complete K3 forwards with two expert slots break the expected copy order.

    * Cold start: the executor's start enqueues ``L(1), L(2), E(1), E(2)``.
    * Within a forward: each ``L(l)`` of layers 3..92 right before ``E(l)``.
    * At the end of a forward: the next one's ``E(1), L(1), E(2), L(2)`` (``L(1), E(1), L(2),
      E(2)`` with the attention segment alone).
    * The second forward's start and H0 enqueue nothing; it copies every layer once.
    """
    problems = []
    cold = driver.launches_in((0, "start", None))
    if cold != [("L", 1), ("L", 2), (EXPERT, 1), (EXPERT, 2)]:
        problems.append(f"cold start {cold}")
    for forward in range(2):
        first, end = driver.spans[(forward, "H1", 1)][0], driver.forward_bounds(forward)[1]
        sequence = driver.launches(first, end)
        for layer in driver.layers[2:]:
            at = sequence.index((EXPERT, layer))
            if at == 0 or sequence[at - 1] != ("L", layer):
                problems.append(f"forward {forward}: E({layer}) not right after L({layer})")
        tail = sequence[-4:]
        if MOE in driver.segments:
            expected_tail = [(EXPERT, 1), ("L", 1), (EXPERT, 2), ("L", 2)]
        else:
            expected_tail = [("L", 1), (EXPERT, 1), ("L", 2), (EXPERT, 2)]
        if tail != expected_tail:
            problems.append(f"forward {forward} ends with {tail}")
    for hook in ("start", "H0"):
        if driver.launches_in((1, hook, None)):
            problems.append(
                f"the second forward's {hook} enqueued {driver.launches_in((1, hook, None))}"
            )
    second = Counter(driver.launches(*driver.forward_bounds(1)))
    if second != Counter((what, layer) for what in ("L", EXPERT) for layer in driver.layers):
        problems.append("the second forward does not copy every layer once")
    return problems


@SEGMENT_SETS
def test_the_attached_copy_order(segments: tuple[SegmentKind, ...]) -> None:
    driver = Driver(segments)
    assert driver.forward() and driver.forward()
    assert copy_order_problems(driver) == []
    _check_steady(driver)


@SEGMENT_SETS
def test_the_order_check_fails_if_h2_does_not_fill(
    monkeypatch: pytest.MonkeyPatch, segments: tuple[SegmentKind, ...]
) -> None:
    """Sensitivity: with the fill at H3 only, ``L(l)`` follows ``E(l)``, issued at the finish."""
    attention_done = DenseDWDP.attention_done

    def attention_done_without_fill(runtime: DenseDWDP, layer: int) -> None:
        runtime._fill = lambda: None
        try:
            attention_done(runtime, layer)
        finally:
            del runtime._fill

    monkeypatch.setattr(DenseDWDP, "attention_done", attention_done_without_fill)
    driver = Driver(segments)
    assert driver.forward() and driver.forward()
    problems = copy_order_problems(driver)
    assert any("not right after" in problem for problem in problems), problems


def test_the_order_check_fails_if_the_expert_prefetch_starts_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sensitivity: an expert start ahead of the dense priming breaks the cold start."""
    driver = Driver()
    prefetch = DenseDWDP.prefetch

    def prefetch_after_the_expert_start(runtime: DenseDWDP, layer: int) -> None:
        driver.weight_manager.start_forward()
        prefetch(runtime, layer)

    monkeypatch.setattr(DenseDWDP, "prefetch", prefetch_after_the_expert_start)
    assert driver.forward() and driver.forward()
    problems = copy_order_problems(driver)
    assert any(problem.startswith("cold start") for problem in problems), problems


# ---------------------------------------------------------------------------
# Expert prefetch slots and model sizes
# ---------------------------------------------------------------------------


@SEGMENT_SETS
@pytest.mark.parametrize("num_layers", [1, 2, 3, 4, 5, 92])
@pytest.mark.parametrize("num_slots", [2, 3, 4])
def test_attached_forwards_are_race_free(
    segments: tuple[SegmentKind, ...], num_layers: int, num_slots: int
) -> None:
    """A warmup forward the executor did not start, then three it did: every read is ordered."""
    layers = tuple(range(1, num_layers + 1))
    driver = Driver(segments, layers=layers, num_slots=num_slots)
    assert driver.forward(start=False)
    assert driver.runtime.primed
    for _ in range(3):
        assert driver.forward()
        assert driver.runtime.primed and not driver.runtime.forward_open
    # The last forward copies every layer once, dense and expert.
    last = Counter(driver.launches(*driver.forward_bounds(3)))
    assert last == Counter((what, layer) for what in ("L", EXPERT) for layer in layers)
    _check_steady(driver)


# ---------------------------------------------------------------------------
# Fences, demand fetches, the WAR barrier and recovery
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("prime", [True, False], ids=["primed", "demand-fetched"])
def test_fence_when_a_holder_has_an_open_read_window(prime: bool) -> None:
    """A forward dies inside attention(1); a prefetch that skips the WAR barrier gets the fence.

    H1(1) opened the read windows of ``A(1)`` and ``D(1)``, whether priming or, in a forward that
    skipped priming, H1's demand fetch enqueued ``L(1)``, and no consume record closed them: only
    the fence, recorded on compute after the reads, orders the copy after them.
    """
    driver = Driver()
    runtime = driver.runtime
    assert not driver.forward(start=prime, h0=prime, stop_at=("H2", 1))
    assert runtime.forward_open and runtime.demand_fetches == int(not prime)
    with driver.ops.stream(driver.ops.compute):
        driver.span("again", None, lambda: runtime.prefetch(1))
    assert runtime.war_fences == 1  # one per prefetch, not one per slot
    assert driver.launches_in((1, "again", None)) == [("L", 1)]
    assert driver.forward()  # H0 runs the barrier, then primes
    assert runtime.war_fences == 1 and runtime.demand_fetches == int(not prime)
    driver.assert_race_free()
    driver.assert_no_host_sync()


def test_war_barrier_then_reprime_after_an_abort() -> None:
    driver = Driver()
    runtime = driver.runtime
    assert driver.forward()
    assert not driver.forward(stop_at=("H3", 41))  # dies inside the MoE block of layer 41
    assert runtime.forward_open and not runtime.primed
    start = len(driver.log)
    with driver.ops.stream(driver.ops.compute):
        driver.span("barrier", None, runtime.war_barrier)
    records = driver.stream_ops(start)
    assert [entry.kind for entry in records] == ["record"] * (RING_DEPTH[ATTN] + RING_DEPTH[MOE])
    assert all(entry.stream is driver.ops.compute for entry in records)
    assert len({id(entry.event) for entry in records}) == len(records)
    for kind in BOTH:
        for index in range(RING_DEPTH[kind]):
            assert runtime.slot_tag(kind, index) == EMPTY_TAG
            assert runtime.slot_consumed(kind, index)
    assert not runtime.primed and not runtime.forward_open
    assert driver.forward()  # the start primes; the expert prefetch restarts its sequence
    assert driver.launches_in((2, "start", None)) == [
        ("L", 1),
        ("L", 2),
        (EXPERT, 1),
        (EXPERT, 2),
    ]
    assert driver.forward()
    _check_steady(driver)


ABORT_POINTS = ["H1", "H2", "finish", "H3"]


@SEGMENT_SETS
@pytest.mark.parametrize("recovery", ["executor", "model"])
@pytest.mark.parametrize("position", [0, 1, 46, 90, 91])
@pytest.mark.parametrize("stop", ABORT_POINTS)
def test_an_aborted_forward_recovers_both_sides(
    segments: tuple[SegmentKind, ...], recovery: str, position: int, stop: str
) -> None:
    """A K3 forward stops at ``stop`` of the layer at ``position``; the next forward recovers.

    The executor's start leaves an open forward to H0, which runs the dense WAR barrier on the
    execution stream, primes ``L(1), L(2)`` and then starts the expert prefetch, which restarts
    its sequence, synchronizing the device only if the stopped forward left a MoE layer bound.
    """
    driver = Driver(segments)
    runtime = driver.runtime
    layer = driver.layers[position]
    assert driver.forward() and driver.forward()
    aborted_at = len(driver.log)
    assert not driver.forward(stop_at=(stop, layer))
    was_open = runtime.forward_open
    assert was_open == (stop != "H1" or position > 0)
    bound = stop == "finish"
    # The expert sequence restarts unless the stopped forward bound no MoE layer or finished all.
    last = len(driver.layers) - 1
    restarts = bound or (position > 0 if stop in ("H1", "H2") else position != last)
    assert driver.forward(start=recovery == "executor")
    if recovery == "executor":
        # An open forward is left to H0; otherwise the forward is primed already.
        assert driver.launches_in((3, "start", None)) == []
    h0_start, h0_end = driver.spans[(3, "H0", None)]
    barrier = driver.records(h0_start, h0_end, "dense", driver.ops.compute)
    assert len(barrier) == (sum(RING_DEPTH[kind] for kind in segments) if was_open else 0)
    assert driver.records(h0_start, h0_end, "expert", driver.ops.compute) == []
    expected = [("L", 1), ("L", 2)] * was_open + [(EXPERT, 1), (EXPERT, 2)] * restarts
    assert driver.launches(h0_start, h0_end) == expected
    assert driver.syncs(aborted_at) == driver.syncs(h0_start, h0_end)
    assert len(driver.syncs(aborted_at)) == int(bound)
    assert runtime.primed and not runtime.forward_open
    assert driver.forward()
    assert runtime.war_fences == 0 and runtime.demand_fetches == 0
    driver.assert_race_free()


def test_invalidate_prefetch_keeps_the_dense_copies() -> None:
    """New expert weights regather ``E(1), E(2)``; the dense weights cannot change."""
    driver = Driver()
    assert driver.forward() and driver.forward()
    driver.invalidate()
    assert driver.launches_in((2, "invalidate", None)) == []
    assert driver.forward()
    assert driver.launches_in((2, "start", None)) == [(EXPERT, 1), (EXPERT, 2)]
    assert driver.launches_in((2, "H0", None)) == []
    assert driver.runtime.primed
    _check_steady(driver)


# ---------------------------------------------------------------------------
# Random sequences
# ---------------------------------------------------------------------------


def _random_sequence(seed: int) -> str | None:
    """One random sequence of forwards; what went wrong, or None."""
    rng = random.Random(seed)
    num_layers = rng.choice([1, 2, 3, 5, 8])
    segments = rng.choice([BOTH, (ATTN,)])
    driver = Driver(
        segments, layers=tuple(range(1, num_layers + 1)), num_slots=rng.choice([2, 3, 4])
    )
    runtime = driver.runtime
    steps = []
    assert driver.forward(start=False)
    for _ in range(12):
        roll = rng.random()
        start = rng.random() < 0.7
        if roll < 0.35:
            stop = rng.choice(["H1", "H2", "finish", "H3", "fill"])
            layer = rng.choice(driver.layers)
            steps.append(f"stop at {stop} of L{layer}, start={start}")
            driver.forward(start=start, stop_at=(stop, layer))
        elif roll < 0.45:
            steps.append("start, then the forward fails before the model")
            driver.start()
        elif roll < 0.55:
            steps.append("invalidate")
            driver.invalidate()
        elif roll < 0.62:
            steps.append(f"a launch of H0 fails, start={start}")
            driver.forward(start=start, fail_h0=True)
        else:
            steps.append(f"full forward, start={start}")
            demand_fetches = runtime.demand_fetches
            if not driver.forward(start=start):
                return f"{steps}: the forward stopped"
            if runtime.demand_fetches != demand_fetches or not runtime.primed:
                return f"{steps}: demand fetch, or not primed after a complete forward"
    if runtime.war_fences:
        return f"{steps}: {runtime.war_fences} fences"
    try:
        driver.assert_race_free()
    except AssertionError as exc:
        return f"{steps}: {exc}"
    return None


def test_random_sequences_are_race_free() -> None:
    """Aborts at every point (also inside a fill or H0's priming), starts whose forward fails
    before the model and invalidations, in random order: no race, fence or window error."""
    problems = {seed: problem for seed in range(200) if (problem := _random_sequence(seed))}
    assert problems == {}
