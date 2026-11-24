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
"""CPU tests of the standalone dense DWDP runtime protocol (``runtime.DenseDWDP``).

A recording fake of the CUDA seam logs every event record, stream wait, copy launch and device
synchronization in host order; a wait binds to the latest record of its event before it, as
``cudaStreamWaitEvent`` does, and a device synchronization orders everything after it behind
everything before it.  The runtime runs on the fake through its ``ops`` seam, and the real
standalone ``DenseDwdpManager`` holds it.  A driver calls them the way the executor loop
(``DenseDwdpManager.prefetch_first_layers`` on the default stream) and ``KimiLinearModel`` (H0-H3,
on the execution stream) do, and logs the model's weight reads.  The tests check where the runtime
records its events, the copy order on the copy stream, that the host never synchronizes after the
commit and, with vector clocks over the log, that no copy into a dense slot can overlap a read of
that slot: a missing consume wait, fence or WAR barrier fails them.
"""

from __future__ import annotations

import contextlib
import random
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import pytest

from tensorrt_llm._torch.modules.dwdp.dense.manager import DenseDwdpManager
from tensorrt_llm._torch.modules.dwdp.dense.runtime import DenseDWDP
from tensorrt_llm._torch.modules.dwdp.dense.types import (
    EMPTY_TAG,
    RING_DEPTH,
    ROW_ALIGN,
    DenseMode,
    SegmentKind,
    SlotSpec,
    allocate_slot_ring,
)
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only

ATTN = SegmentKind.ATTN
MOE = SegmentKind.MOE
BOTH = (ATTN, MOE)
SEGMENT_SETS = pytest.mark.parametrize("segments", [BOTH, (ATTN,)], ids=["attn+moe", "attn"])
#: The covered layers of Kimi K3.1: every MoE layer, 1..92 (layer 0 has the dense MLP).
K3_LAYERS = tuple(range(1, 93))

# ---------------------------------------------------------------------------
# Recording fake of the CUDA seam
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class Entry:
    """One log entry, in host order.

    ``kind`` is ``record``, ``wait``, ``launch`` or ``read`` (stream operations), ``sync`` (a
    device synchronization) or ``mark`` (host-side).  ``bound`` is the log index of the record a
    ``wait`` binds to (None: the event was never recorded).  ``what`` is ``L`` for a launch, the
    segment for a read and the hook for a mark.
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
    """An event; ``side`` tells the dense runtime's events from the executor's."""

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
# Driver: the executor loop, the model and the standalone manager around the runtime
# ---------------------------------------------------------------------------


class Abort(Exception):
    """A forward that stops midway, as one whose kernel launch or collective raised."""


class LaunchError(Exception):
    """A dense copy launch that raised."""


class Driver:
    """Calls the standalone manager and the runtime in program order; logs reads and launches.

    Every call is wrapped in a span keyed ``(forward, hook, layer)``; a ``mark`` entry opens it, so
    each log entry can be attributed to the call that made it.

    Args:
        segments: Enabled dense segments.
        layers: Covered layers.
    """

    def __init__(
        self,
        segments: tuple[SegmentKind, ...] = BOTH,
        *,
        layers: tuple[int, ...] = K3_LAYERS,
    ) -> None:
        self.ops = FakeOps()
        self.log = self.ops.log
        self.segments = segments
        self.layers = tuple(layers)
        self.positions = {layer: position for position, layer in enumerate(self.layers)}
        self.forward_index = 0
        self.spans: dict[tuple[int, str, int | None], tuple[int, int]] = {}
        # Set to make the next dense launch raise, as a failing copy launch does.
        self.fail_dense_launch = False
        spec = {kind: SlotSpec(kind, RING_DEPTH[kind], ROW_ALIGN, 2) for kind in segments}
        slots = {kind: allocate_slot_ring(spec[kind], "cpu") for kind in segments}
        self.mark("construct", None)
        self.runtime = DenseDWDP(
            self.layers,
            segments,
            slots,
            self.ops.copy,
            prefetch_next_pass=False,
            ops=self.ops,
        )
        self.runtime.set_launcher(self.launch_dense)
        self.span("commit", None, lambda: self.runtime.commit(None))
        mapping = Mapping(world_size=2, rank=0, tp_size=2, enable_attention_dp=True)
        self.dense_manager = DenseDwdpManager(DenseMode.STANDALONE, mapping=mapping)
        # What DenseDwdpManager.setup stores once setup_dense_dwdp commits.
        self.dense_manager._dense = self.runtime
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

    def launch_dense(self, layer: int) -> None:
        """The dense ``LayerLauncher``: one launch on the current stream."""
        if self.fail_dense_launch:
            self.fail_dense_launch = False
            raise LaunchError(f"L({layer})")
        self.log.append(Entry("launch", stream=self.ops.current, what="L", layer=layer))

    def read(self, what: str, layer: int) -> None:
        """A kernel that reads ``what``'s weights of ``layer`` on the current stream."""
        self.log.append(Entry("read", stream=self.ops.current, what=what, layer=layer))

    # -- executor loop and model ----------------------------------------------

    def start(self) -> None:
        """The non-overlap executor loop before a forward, on the default stream."""
        with self.ops.stream(self.ops.default):
            self.span("start", None, self.dense_manager.prefetch_first_layers)

    def h0(self) -> None:
        """The model's forward entry: ``DenseDwdpManager.ensure_primed``."""
        self.dense_manager.ensure_primed()

    def layer(self, layer: int, stop: str | None = None) -> None:
        """One decoder layer in program order; raises :class:`Abort` right before ``stop``.

        ``stop`` is ``H1``, ``H2`` or ``H3``; ``fill`` makes the first dense copy launch of the
        layer's hooks raise instead.  The attention reads stand for the attention modules'
        forward and, under sequence-expert parallelism, for ``forward_kda_sep`` /
        ``forward_dsa_sep``, which run at the same place.
        """
        runtime = self.runtime
        if stop == "fill":
            self.fail_dense_launch = True
        try:
            if stop == "H1":
                raise Abort
            self.span("H1", layer, lambda: runtime.wait_layer(layer))
            self.read(ATTN.value, layer)  # the projections and the core
            self.read(ATTN.value, layer)  # the output gate and o_proj
            if stop == "H2":
                raise Abort
            self.span("H2", layer, lambda: runtime.attention_done(layer))
            if MOE in self.segments:
                self.read(MOE.value, layer)  # the router and the latent down-projection
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
        default stream after a forward that completes.
        """
        if start:
            self.start()
        ops = self.ops
        try:
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

    def launches(self, start: int = 0, end: int | None = None) -> list[tuple[str, int]]:
        """``(what, layer)`` of the copy launches in ``[start, end)``."""
        return [
            (entry.what, entry.layer) for entry in self.log[start:end] if entry.kind == "launch"
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
# Two complete Kimi K3.1 forwards
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", params=[BOTH, (ATTN,)], ids=["attn+moe", "attn"])
def two_forwards(request: pytest.FixtureRequest) -> Driver:
    """Two complete K3.1 forwards (92 covered layers each), started by the executor; tests only
    read it."""
    driver = Driver(segments=request.param)
    assert driver.forward() and driver.forward()
    return driver


def test_read_windows_end_at_h2_and_h3_and_h2_fills_two_layers_ahead(two_forwards: Driver) -> None:
    driver = two_forwards
    log = driver.log
    has_moe = MOE in driver.segments
    window_ends = ("H2", "H3") if has_moe else ("H2",)
    for forward in range(2):
        # The executor's start primes the first two layers of every forward.
        assert driver.launches_in((forward, "start", None)) == [("L", 1), ("L", 2)]
        assert driver.launches_in((forward, "H0", None)) == []
        for layer in driver.layers:
            for hook in window_ends:
                start, end = driver.spans[(forward, hook, layer)]
                ops = driver.stream_ops(start, end)
                # The hook ends the window first, so each copy it enqueues binds to that record.
                assert ops[0].kind == "record" and ops[0].stream is driver.ops.compute
                consume = driver.records(start, end, "dense", driver.ops.compute)
                assert len(consume) == 1, (forward, hook, layer)
            expected = [("L", layer + 2)] if layer + 2 <= driver.layers[-1] else []
            assert driver.launches_in((forward, "H2", layer)) == expected, (forward, layer)
            if has_moe:
                assert driver.launches_in((forward, "H3", layer)) == [], (forward, layer)
    # Those are the only dense compute records after commit: no other hook ends a read window.
    committed = driver.spans[(0, "commit", None)][1]
    records = driver.records(committed, len(log), "dense", driver.ops.compute)
    assert len(records) == 2 * len(driver.layers) * len(window_ends)
    copies = Counter(layer for _what, layer in driver.launches())
    assert copies == Counter({layer: 2 for layer in driver.layers})
    assert not driver.runtime.forward_open and not driver.runtime.primed
    reads_per_layer = 4 if has_moe else 2
    assert driver.assert_race_free() == 2 * len(driver.layers) * reads_per_layer
    _check_steady(driver)


@SEGMENT_SETS
@pytest.mark.parametrize("num_layers", [1, 2, 3, 4, 5, 92])
def test_standalone_forwards_are_race_free(
    segments: tuple[SegmentKind, ...], num_layers: int
) -> None:
    """A warmup forward the executor did not start, then three it did: every read is ordered."""
    layers = tuple(range(1, num_layers + 1))
    driver = Driver(segments, layers=layers)
    assert driver.forward(start=False)
    # H0 primed the warmup forward.
    assert driver.launches_in((0, "H0", None)) == [("L", layer) for layer in layers[:2]]
    for _ in range(3):
        assert driver.forward()
        assert not driver.runtime.primed and not driver.runtime.forward_open
    # The last forward copies every layer once.
    last = Counter(driver.launches(*driver.forward_bounds(3)))
    assert last == Counter(("L", layer) for layer in layers)
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
    assert driver.forward()  # the start primes again
    assert driver.launches_in((2, "start", None)) == [("L", 1), ("L", 2)]
    assert driver.forward()
    _check_steady(driver)


@SEGMENT_SETS
@pytest.mark.parametrize("recovery", ["executor", "model"])
@pytest.mark.parametrize("position", [0, 1, 46, 90, 91])
@pytest.mark.parametrize("stop", ["H1", "H2", "H3"])
def test_a_forward_recovers_after_an_abort(
    segments: tuple[SegmentKind, ...], recovery: str, position: int, stop: str
) -> None:
    """H0 runs the WAR barrier and primes again; the executor's start leaves an open forward."""
    driver = Driver(segments)
    runtime = driver.runtime
    assert driver.forward()
    assert not driver.forward(stop_at=(stop, driver.layers[position]))
    was_open = runtime.forward_open
    assert was_open == (stop != "H1" or position > 0)
    assert driver.forward(start=recovery == "executor")
    if recovery == "executor" and was_open:
        # An open forward is left to H0.
        assert driver.launches_in((2, "start", None)) == []
    start, end = driver.spans[(2, "H0", None)]
    if was_open:
        assert driver.launches(start, end) == [("L", 1), ("L", 2)]
        assert len(driver.records(start, end, "dense", driver.ops.compute)) == sum(
            RING_DEPTH[kind] for kind in segments
        )
    assert not runtime.primed and not runtime.forward_open
    assert driver.forward()
    _check_steady(driver)


# ---------------------------------------------------------------------------
# Random sequences
# ---------------------------------------------------------------------------


def _random_sequence(seed: int) -> str | None:
    """One random sequence of forwards; what went wrong, or None."""
    rng = random.Random(seed)
    num_layers = rng.choice([1, 2, 3, 5, 8])
    segments = rng.choice([BOTH, (ATTN,)])
    driver = Driver(segments, layers=tuple(range(1, num_layers + 1)))
    runtime = driver.runtime
    steps = []
    assert driver.forward(start=False)
    for _ in range(12):
        roll = rng.random()
        start = rng.random() < 0.7
        if roll < 0.35:
            stop = rng.choice(["H1", "H2", "H3", "fill"])
            layer = rng.choice(driver.layers)
            steps.append(f"stop at {stop} of L{layer}, start={start}")
            driver.forward(start=start, stop_at=(stop, layer))
        elif roll < 0.5:
            steps.append("start, then the forward fails before the model")
            driver.start()
        elif roll < 0.6:
            steps.append(f"a launch of H0 fails, start={start}")
            driver.forward(start=start, fail_h0=True)
        else:
            steps.append(f"full forward, start={start}")
            demand_fetches = runtime.demand_fetches
            if not driver.forward(start=start):
                return f"{steps}: the forward stopped"
            if runtime.demand_fetches != demand_fetches or runtime.primed:
                return f"{steps}: demand fetch, or primed state wrong after a complete forward"
    if runtime.war_fences:
        return f"{steps}: {runtime.war_fences} fences"
    try:
        driver.assert_race_free()
    except AssertionError as exc:
        return f"{steps}: {exc}"
    return None


def test_random_sequences_are_race_free() -> None:
    """Aborts at every point (also inside a fill or H0's priming) and starts whose forward fails
    before the model, in random order: no race, fence or window error."""
    problems = {seed: problem for seed in range(200) if (problem := _random_sequence(seed))}
    assert problems == {}
