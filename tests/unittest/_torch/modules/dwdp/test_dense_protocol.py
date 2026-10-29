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
"""CPU tests of the dense DWDP runtime protocol.

A recording fake of the runtime's CUDA seam logs every event record, stream wait and copy launch
in host order; a wait binds to the latest record of its event before it, as
``cudaStreamWaitEvent`` does.  A driver calls the runtime the way ``KimiLinearModel`` (H0-H3),
``ConfigurableMoE`` (the expert copies ``E(l)``, on the same copy stream) and ``DwdpManager``
(priming, WAR barrier) do, and logs the model's weight reads on the compute stream.  The tests
check the bindings the protocol lists (invariants I1-I4), that the host never
synchronizes after commit and, with vector clocks over the log, that no copy into a slot can
overlap a read of that slot.  They also check that H0 and H1 refuse a CUDA graph capture and
multi-stream execution before anything reaches a stream, and drive the slot watchdog of
``TRTLLM_DWDP_DENSE_DEBUG=2`` over CPU slots that the fake copies fill.
"""

from __future__ import annotations

import contextlib
import gc
import re
import weakref
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import pytest
import torch

from tensorrt_llm._torch.modules.dwdp.dense import runtime as runtime_module
from tensorrt_llm._torch.modules.dwdp.dense.layout import tensor_checksum
from tensorrt_llm._torch.modules.dwdp.dense.runtime import DenseDWDP
from tensorrt_llm._torch.modules.dwdp.dense.types import (
    EMPTY_TAG,
    LOG_PREFIX,
    RING_DEPTH,
    ROW_ALIGN,
    DenseProtocolError,
    SegmentKind,
    SlotBuffer,
    SlotSpec,
    allocate_slot_ring,
)
from tensorrt_llm._torch.modules.dwdp.dense.watchdog import SlotWatchdog

pytestmark = pytest.mark.cpu_only

ATTN = SegmentKind.ATTN
MOE = SegmentKind.MOE
BOTH = (ATTN, MOE)
ATTN_ONLY = (ATTN,)
K3_LAYERS = tuple(range(1, 93))
#: Hook that closes each segment's read window.
WINDOW_END = {ATTN: "H2", MOE: "H3"}
#: ``(forward, hook, layer)`` of the hook during which a log entry was made.
Owner = tuple[int, str, int | None]
#: How ``DwdpManager.ensure_primed`` names itself when ``require_eager`` refuses it.
H0_HOOK = "DwdpManager.ensure_primed (H0)"

# ---------------------------------------------------------------------------
# Recording fake of the CUDA seam
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class Entry:
    """One log entry, in host order.

    ``kind`` is ``record``, ``wait``, ``launch`` or ``read`` (stream operations), or ``mark``,
    ``nvtx``, ``sync``, ``detach``, ``store`` (host-side).  ``bound`` is the log index of the
    record a ``wait`` binds to (None: the event was never recorded).  ``what`` is ``L``/``E`` for
    a launch, the segment for a read, the hook for a mark and the range name for NVTX.
    """

    kind: str
    stream: FakeStream | None = None
    event: FakeEvent | None = None
    bound: int | None = None
    what: str = ""
    layer: int | None = None
    forward: int | None = None


class FakeStream:
    def __init__(self, log: list[Entry], name: str) -> None:
        self.log = log
        self.name = name

    def wait_event(self, event: FakeEvent) -> None:
        self.log.append(Entry("wait", stream=self, event=event, bound=event.last_record))

    def __repr__(self) -> str:
        return f"<stream {self.name}>"


class FakeEvent:
    def __init__(self, ops: FakeOps) -> None:
        self.ops = ops
        self.last_record: int | None = None

    def record(self, stream: FakeStream | None = None) -> None:
        stream = self.ops.current if stream is None else stream
        for observer in self.ops.record_observers:
            observer(stream)
        self.last_record = len(self.ops.log)
        self.ops.log.append(Entry("record", stream=stream, event=self))


class FakeOps:
    """``DeviceOps`` that logs every stream operation in host order."""

    def __init__(self) -> None:
        self.log: list[Entry] = []
        self.compute = FakeStream(self.log, "compute")
        self.copy = FakeStream(self.log, "copy")
        self.current = self.compute
        self.capturing = False
        self.multi_stream = False
        self.sync_error: Exception | None = None
        self.record_observers: list[Callable[[FakeStream], None]] = []

    def new_event(self) -> FakeEvent:
        return FakeEvent(self)

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
        if self.sync_error is not None:
            raise self.sync_error

    def is_current_stream_capturing(self) -> bool:
        return self.capturing

    def do_multi_stream(self) -> bool:
        return self.multi_stream

    @contextlib.contextmanager
    def nvtx_range(self, message: str) -> Iterator[None]:
        self.log.append(Entry("nvtx", what=message))
        yield


class FakeBindings:
    """The part of ``DenseBindings`` the runtime calls."""

    def __init__(self, log: list[Entry], error: Exception | None = None) -> None:
        self.log = log
        self.error = error

    def detach(self) -> None:
        self.log.append(Entry("detach"))
        if self.error is not None:
            raise self.error


class FakeStore:
    """The part of ``DenseShardStore`` the runtime calls."""

    def __init__(self, log: list[Entry], error: Exception | None = None) -> None:
        self.log = log
        self.error = error
        self.releases = 0

    def release(self) -> None:
        self.releases += 1
        self.log.append(Entry("store"))
        if self.error is not None:
            raise self.error


class CaptureLogger:
    """Stands in for the runtime module's ``logger``: keeps the warnings, drops the rest."""

    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warning(self, message: str) -> None:
        self.warnings.append(message)

    def debug(self, message: str) -> None:
        pass

    info = error = debug


class CallableLauncher:
    """A launcher object, so tests can check the runtime drops it."""

    def __init__(self, ops: FakeOps) -> None:
        self.ops = ops

    def __call__(self, layer: int) -> None:
        self.ops.log.append(Entry("launch", stream=self.ops.current, what="L", layer=layer))


def make_ring(kind: SegmentKind, depth: int | None = None) -> tuple[SlotBuffer, ...]:
    depth = RING_DEPTH[kind] if depth is None else depth
    return allocate_slot_ring(SlotSpec(kind, depth, ROW_ALIGN, 2), "cpu")


def make_rings(segments: tuple[SegmentKind, ...] = BOTH) -> dict[SegmentKind, tuple]:
    return {kind: make_ring(kind) for kind in segments}


# ---------------------------------------------------------------------------
# Happens-before over the log
# ---------------------------------------------------------------------------


def vector_clocks(log: list[Entry]) -> dict[int, dict[FakeStream, int]]:
    """Vector clock of every stream operation: ``clock[s]`` = operations of ``s`` it follows.

    Streams run their operations in enqueue order; a wait makes the waiting stream follow every
    operation its bound record follows, the record included.
    """
    stream_clocks: dict[FakeStream, dict[FakeStream, int]] = {}
    clocks: dict[int, dict[FakeStream, int]] = {}
    for index, entry in enumerate(log):
        if entry.stream is None:
            continue
        clock = stream_clocks.setdefault(entry.stream, {})
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
# Driver: the model, ConfigurableMoE and DwdpManager around the runtime
# ---------------------------------------------------------------------------


class Driver:
    """Calls the runtime in the model's program order and logs reads and expert copies.

    Every runtime call is wrapped in a span keyed ``(forward, hook, layer)``; a ``mark`` entry
    opens it, so each log entry can be attributed to the hook that made it.
    """

    def __init__(
        self,
        *,
        segments: tuple[SegmentKind, ...] = BOTH,
        layers: tuple[int, ...] = K3_LAYERS,
        strict: bool = False,
        commit: bool = True,
        watchdog: SlotWatchdog | None = None,
    ) -> None:
        self.ops = FakeOps()
        self.log = self.ops.log
        self.segments = tuple(segments)
        self.layers = tuple(layers)
        self.slots = make_rings(self.segments)
        self.forward_index = 0
        self.fail_launch = False
        self.spans: dict[tuple[int, str, int | None], tuple[int, int]] = {}
        self.mark("construct", None)
        self.runtime = DenseDWDP(
            self.layers,
            self.segments,
            self.slots,
            self.ops.copy,
            ops=self.ops,
            strict=strict,
            watchdog=watchdog,
        )
        self.runtime.set_launcher(self.launch_dense)
        if commit:
            self.span("commit", None, lambda: self.runtime.commit(None, fill_byte=0))

    # -- logging -------------------------------------------------------------

    def mark(self, hook: str, layer: int | None) -> None:
        self.log.append(Entry("mark", what=hook, layer=layer, forward=self.forward_index))

    def span(self, hook: str, layer: int | None, action: Callable[[], object]) -> None:
        start = len(self.log)
        self.mark(hook, layer)
        action()
        self.spans[(self.forward_index, hook, layer)] = (start, len(self.log))

    def launch_dense(self, layer: int) -> None:
        """The dense ``LayerLauncher``: one launch on the current stream."""
        if self.fail_launch:
            self.fail_launch = False
            raise RuntimeError(f"launch of L({layer}) failed")
        self.log.append(Entry("launch", stream=self.ops.current, what="L", layer=layer))

    def launch_expert(self, layer: int) -> None:
        """``DWDPWeightManager.prefetch_layer``: ``E(layer)`` on the shared copy stream."""
        with self.ops.stream(self.ops.copy):
            self.log.append(Entry("launch", stream=self.ops.current, what="E", layer=layer))

    def read(self, kind: SegmentKind, layer: int) -> None:
        """A kernel that reads ``kind``'s weights of ``layer`` on the current stream."""
        self.log.append(Entry("read", stream=self.ops.current, what=kind.value, layer=layer))

    # -- DwdpManager -----------------------------------------------------------

    def prime_all(self) -> None:
        """``DwdpManager._prime_all``: ``L(first), E(first), L(second), E(second)``."""
        for layer in self.layers[:2]:
            self.runtime.prefetch(layer)
            self.launch_expert(layer)
        self.runtime.mark_primed()

    def ensure_primed(self) -> None:
        """``DwdpManager.ensure_primed`` at the model's H0."""
        self.runtime.require_eager(H0_HOOK)
        if self.runtime.forward_open:
            self.runtime.war_barrier()
        if not self.runtime.primed:
            self.prime_all()

    # -- model -----------------------------------------------------------------

    def layer(self, layer: int, *, stop_before: str | None = None) -> bool:
        """One decoder layer in program order; False if it stops before a hook."""
        runtime = self.runtime
        if stop_before == "H1":
            return False
        self.span("H1", layer, lambda: runtime.wait_layer(layer))
        self.read(ATTN, layer)  # attention(l): MKL:2053-2056
        if stop_before == "H2":
            return False
        self.span("H2", layer, lambda: runtime.attention_done(layer))
        if MOE in self.segments:
            self.read(MOE, layer)  # router and W-down
        self.span("experts", layer, lambda: None)  # CMOE:690, the first read of E(l)
        ahead = self.layers.index(layer) + 2
        if ahead < len(self.layers):
            self.span("C_E", layer, lambda: self.launch_expert(self.layers[ahead]))  # CMOE:709
        if MOE in self.segments:
            self.read(MOE, layer)  # W-up and the shared expert
        if stop_before == "H3":
            return False
        self.span("H3", layer, lambda: runtime.layer_done(layer))
        return True

    def forward(self, *, stop_at: tuple[str, int] | None = None, prime: bool = True) -> None:
        """One model forward; ``stop_at=(hook, layer)`` aborts it right before that hook."""
        if prime:
            self.span("H0", None, self.ensure_primed)
        for layer in self.layers:
            stop = stop_at[0] if stop_at is not None and stop_at[1] == layer else None
            if not self.layer(layer, stop_before=stop):
                break
        self.forward_index += 1

    # -- analysis --------------------------------------------------------------

    def owners(self) -> list[Owner | None]:
        """For every log index, the span (``(forward, hook, layer)``) that made the entry."""
        current = None
        result = []
        for entry in self.log:
            if entry.kind == "mark":
                current = (entry.forward, entry.what, entry.layer)
            result.append(current)
        return result

    def launch_index(self, what: str, layer: int, forward: int) -> int:
        owners = self.owners()
        found = [
            index
            for index, entry in enumerate(self.log)
            if entry.kind == "launch"
            and (entry.what, entry.layer) == (what, layer)
            and owners[index][0] == forward
        ]
        assert len(found) == 1, f"{what}({layer}) of forward {forward} launched at {found}"
        return found[0]

    def launches_in(self, key: tuple[int, str, int | None], what: str = "L") -> list[int]:
        """Log indices of the ``what`` launches made inside the span ``key``."""
        start, end = self.spans[key]
        return [
            index
            for index in range(start, end)
            if self.log[index].kind == "launch" and self.log[index].what == what
        ]

    def war_binding(self, launch: int) -> dict[str, list[tuple[int, int | None]]]:
        """``{hook: [(forward, layer)]}`` of the records that the copy waits before ``launch`` bind to.

        A consume wait binds to the latest consume record of its slot, a fence wait to the fence
        that the enqueueing hook records; both are compute-stream records.
        """
        owners = self.owners()
        result: dict[str, list[tuple[int, int | None]]] = {}
        for wait in self.copy_waits_before(launch):
            bound = self.log[wait].bound
            assert bound is not None and self.log[bound].stream is self.ops.compute
            forward, hook, layer = owners[bound]
            result.setdefault(hook, []).append((forward, layer))
        return result

    def copy_waits_before(self, launch: int) -> list[int]:
        """Copy-stream waits that the hook enqueueing ``launch`` made before it.

        They follow the hook's span mark and the copy-stream launch before ``launch``; waits that
        a failed launch left on the copy stream in an earlier span are not counted.
        """
        waits = []
        for index in range(launch - 1, -1, -1):
            entry = self.log[index]
            if entry.kind == "mark":
                break
            if entry.stream is not self.ops.copy:
                continue
            if entry.kind == "launch":
                break
            if entry.kind == "wait":
                waits.append(index)
        return waits[::-1]

    def last_copy_launch_before(self, index: int) -> Entry:
        for earlier in range(index - 1, -1, -1):
            entry = self.log[earlier]
            if entry.kind == "launch" and entry.stream is self.ops.copy:
                return entry
        raise AssertionError(f"no copy launch before #{index}")

    def expected_war_owner(self, kind: SegmentKind, layer: int, forward: int) -> Owner:
        """Whose consume record the copy of ``L(layer)`` into ``kind``'s ring waits for.

        The slot's previous holder: the layer one ring depth earlier in the same forward, else
        the last layer of the previous forward that used the slot, else the records of P6.
        """
        depth = RING_DEPTH[kind]
        position = self.layers.index(layer)
        if position >= depth:
            return (forward, WINDOW_END[kind], self.layers[position - depth])
        if forward == 0:
            return (0, "commit", None)
        previous = max(p for p in range(len(self.layers)) if p % depth == position % depth)
        return (forward - 1, WINDOW_END[kind], self.layers[previous])

    def assert_war_bindings(self) -> int:
        """Check that every dense copy waits for exactly the previous holders of its slots."""
        owners = self.owners()
        checked = 0
        for index, entry in enumerate(self.log):
            if entry.kind != "launch" or entry.what != "L":
                continue
            forward = owners[index][0]
            waits = self.copy_waits_before(index)
            assert all(self.log[self.log[w].bound].stream is self.ops.compute for w in waits)
            bound = Counter(owners[self.log[w].bound] for w in waits)
            expected = Counter(
                self.expected_war_owner(kind, entry.layer, forward) for kind in self.segments
            )
            assert bound == expected, f"L({entry.layer}) of forward {forward}"
            checked += 1
        return checked

    def assert_race_free(self) -> int:
        """Check every logged read against every copy into its slot; returns the reads checked.

        RAW: the latest copy of the read layer into the slot completes before the read.  WAR:
        every later copy into the slot starts after the read completes.
        """
        log = self.log
        clocks = vector_clocks(log)
        position = {layer: index for index, layer in enumerate(self.layers)}
        writes: dict[tuple[SegmentKind, int], list[tuple[int, int]]] = {}
        for index, entry in enumerate(log):
            if entry.kind == "launch" and entry.what == "L":
                for kind in self.segments:
                    key = (kind, position[entry.layer] % RING_DEPTH[kind])
                    writes.setdefault(key, []).append((index, entry.layer))
        reads = 0
        for index, entry in enumerate(log):
            if entry.kind != "read":
                continue
            kind = SegmentKind(entry.what)
            key = (kind, position[entry.layer] % RING_DEPTH[kind])
            slot_writes = writes.get(key, [])
            sources = [w for w, layer in slot_writes if w < index and layer == entry.layer]
            assert sources, f"read #{index} of {kind.value}({entry.layer}) precedes its copy"
            source = sources[-1]
            assert happens_before(log, clocks, source, index), (
                f"RAW: read #{index} of {kind.value}({entry.layer}) may start before copy #{source}"
            )
            for later, layer in slot_writes:
                if later > source:
                    assert happens_before(log, clocks, index, later), (
                        f"WAR: copy #{later} of L({layer}) into {kind.value}[{key[1]}] may start "
                        f"before read #{index} of {kind.value}({entry.layer}) completes"
                    )
            reads += 1
        return reads

    def assert_no_host_sync(self) -> None:
        """Check that the host never synchronized after commit, so it can run ahead of the GPU."""
        log = self.log
        committed = self.spans[(0, "commit", None)][1]
        syncs = [index for index in range(committed, len(log)) if log[index].kind == "sync"]
        assert syncs == [], f"host synchronize at log entries {syncs}"

    def compute_records(self, start: int, end: int) -> list[int]:
        return [
            index
            for index in range(start, end)
            if self.log[index].kind == "record" and self.log[index].stream is self.ops.compute
        ]

    def stream_ops(self, start: int, end: int | None = None) -> list[Entry]:
        return [e for e in self.log[start:end] if e.kind in ("record", "wait", "launch")]


def all_slots(segments: tuple[SegmentKind, ...] = BOTH) -> list[tuple[SegmentKind, int]]:
    return [(kind, index) for kind in segments for index in range(RING_DEPTH[kind])]


@pytest.fixture(scope="module", params=[BOTH, ATTN_ONLY], ids=["attn+moe", "attn"])
def two_forwards(request: pytest.FixtureRequest) -> Driver:
    """Two complete K3 forwards (92 layers each), primed at H0; tests only read it."""
    driver = Driver(segments=request.param)
    driver.forward()
    driver.forward()
    return driver


# ---------------------------------------------------------------------------
# Construction, introspection and lifecycle
# ---------------------------------------------------------------------------


def test_constructor_records_every_consume_event_once():
    driver = Driver(commit=False)
    runtime = driver.runtime
    records = [entry for entry in driver.log if entry.kind == "record"]
    assert len(records) == RING_DEPTH[ATTN] + RING_DEPTH[MOE]
    assert len({id(entry.event) for entry in records}) == len(records)
    assert all(entry.stream is driver.ops.compute for entry in records)
    assert not [entry for entry in driver.log if entry.kind in ("wait", "launch", "sync")]
    assert not runtime.primed and not runtime.forward_open
    assert runtime.demand_fetches == 0 and runtime.war_fences == 0
    for kind, index in all_slots():
        assert runtime.slot_tag(kind, index) == EMPTY_TAG
        assert runtime.slot_consumed(kind, index)


def test_introspection_and_the_slot_mapping_of_table_2_4():
    driver = Driver()
    runtime = driver.runtime
    assert runtime.layers == K3_LAYERS
    assert (runtime.first_layer, runtime.last_layer) == (1, 92)
    assert runtime.segments == BOTH
    assert dict(runtime.slots) == {kind: tuple(ring) for kind, ring in driver.slots.items()}
    with pytest.raises(TypeError):
        runtime.slots[ATTN] = ()
    assert runtime.covers(1) and runtime.covers(92)
    assert not runtime.covers(0) and not runtime.covers(93)
    table = (1, 2, 3, 4, 5, 6, 89, 90, 91, 92)
    assert [runtime.slot_index(ATTN, layer) for layer in table] == [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    assert [runtime.slot_index(MOE, layer) for layer in table] == [0, 1, 2, 0, 1, 2, 1, 2, 0, 1]
    assert runtime.slot_index("moe", 4) == 0
    with pytest.raises(DenseProtocolError, match="layer 0 is not covered"):
        runtime.slot_index(ATTN, 0)
    with pytest.raises(IndexError):
        runtime.slot_tag(MOE, 3)
    with pytest.raises(IndexError):
        runtime.slot_consumed(ATTN, -1)
    for call in (
        lambda: runtime.slot_index("mlp", 1),
        lambda: runtime.slot_tag("mlp", 0),
        lambda: runtime.slot_consumed("mlp", 0),
    ):
        with pytest.raises(ValueError, match="'mlp' is not a valid SegmentKind"):
            call()


@pytest.mark.parametrize(
    "layers, segments, rings, match",
    [
        ((), BOTH, None, "at least one covered layer"),
        ((2, 1), BOTH, None, "strictly ascending"),
        ((1, 1, 2), BOTH, None, "strictly ascending"),
        ((-1, 0), BOTH, None, "non-negative"),
        (K3_LAYERS, (), None, "segments must be attn"),
        (K3_LAYERS, (MOE, ATTN), None, "segments must be attn"),
        (K3_LAYERS, (MOE,), None, "segments must be attn"),
        (K3_LAYERS, (ATTN, ATTN), None, "segments must be attn"),
        (K3_LAYERS, BOTH, lambda: {ATTN: make_ring(ATTN)}, "rings must match"),
        (K3_LAYERS, ATTN_ONLY, make_rings, "rings must match"),
        (K3_LAYERS, BOTH, lambda: {ATTN: make_ring(ATTN, 3), MOE: make_ring(MOE)}, "needs 2"),
        (K3_LAYERS, BOTH, lambda: {ATTN: make_ring(ATTN), MOE: make_ring(MOE)[::-1]}, "holds"),
        (K3_LAYERS, BOTH, lambda: {ATTN: make_ring(MOE)[:2], MOE: make_ring(MOE)}, "holds"),
    ],
    ids=[
        "no-layers",
        "unsorted",
        "duplicate",
        "negative",
        "no-segments",
        "moe-first",
        "moe-only",
        "repeated",
        "missing-ring",
        "extra-ring",
        "deep-attn-ring",
        "reordered-ring",
        "foreign-slots",
    ],
)
def test_constructor_rejects_bad_arguments(layers, segments, rings, match):
    ops = FakeOps()
    slots = make_rings() if rings is None else rings()
    with pytest.raises(ValueError, match=match):
        DenseDWDP(layers, segments, slots, ops.copy, ops=ops)


def test_constructor_needs_a_copy_stream_and_accepts_segment_names():
    ops = FakeOps()
    with pytest.raises(ValueError, match="copy stream"):
        DenseDWDP(K3_LAYERS, BOTH, make_rings(), None, ops=ops)
    rings = make_rings()
    runtime = DenseDWDP(
        K3_LAYERS, ("attn", "moe"), {"attn": rings[ATTN], "moe": rings[MOE]}, ops.copy, ops=ops
    )
    assert runtime.segments == BOTH


def test_hooks_raise_before_commit():
    driver = Driver(commit=False)
    runtime = driver.runtime
    calls = {
        "wait_layer": lambda: runtime.wait_layer(1),
        "attention_done": lambda: runtime.attention_done(1),
        "layer_done": lambda: runtime.layer_done(1),
        "prefetch": lambda: runtime.prefetch(1),
        "mark_primed": runtime.mark_primed,
        "war_barrier": runtime.war_barrier,
    }
    start = len(driver.log)
    for name, call in calls.items():
        with pytest.raises(DenseProtocolError, match=f"{name} called before commit"):
            call()
    assert driver.stream_ops(start) == []


def test_setup_calls_raise_after_commit():
    driver = Driver()
    runtime = driver.runtime
    calls = {
        "set_launcher": lambda: runtime.set_launcher(driver.launch_dense),
        "fetch_blocking": lambda: runtime.fetch_blocking(1),
        "commit": lambda: runtime.commit(None, fill_byte=0),
    }
    for name, call in calls.items():
        with pytest.raises(DenseProtocolError, match=f"{name} called after commit"):
            call()


def test_commit_needs_a_launcher_and_a_byte_fill_value():
    ops = FakeOps()
    runtime = DenseDWDP(K3_LAYERS, BOTH, make_rings(), ops.copy, ops=ops)
    with pytest.raises(DenseProtocolError, match="before set_launcher"):
        runtime.commit(None, fill_byte=0)
    with pytest.raises(DenseProtocolError, match="before set_launcher"):
        runtime.fetch_blocking(1)
    with pytest.raises(TypeError, match="callable"):
        runtime.set_launcher(None)
    runtime.set_launcher(CallableLauncher(ops))
    for fill_byte in (-1, 256):
        with pytest.raises(ValueError, match="fill byte"):
            runtime.commit(None, fill_byte=fill_byte)
    runtime.commit(None, fill_byte=0xFF)


def test_uncovered_layers_raise():
    driver = Driver()
    runtime = driver.runtime
    calls = (
        lambda: runtime.wait_layer(0),
        lambda: runtime.attention_done(93),
        lambda: runtime.layer_done(0),
        lambda: runtime.prefetch(93),
    )
    for call in calls:
        with pytest.raises(DenseProtocolError, match=r"layer (0|93) is not covered"):
            call()


def test_fetch_blocking_launches_on_the_copy_stream_then_synchronizes():
    driver = Driver(commit=False)
    runtime = driver.runtime
    start = len(driver.log)
    runtime.fetch_blocking(5)
    host_order = [
        (entry.kind, entry.what, entry.layer, entry.stream)
        for entry in driver.log[start:]
        if entry.kind in ("launch", "record", "wait", "sync")
    ]
    assert host_order == [("launch", "L", 5, driver.ops.copy), ("sync", "", None, None)]
    assert driver.ops.current is driver.ops.compute
    for kind, index in all_slots():
        assert runtime.slot_tag(kind, index) == EMPTY_TAG
        assert runtime.slot_consumed(kind, index)
    with pytest.raises(DenseProtocolError, match="layer 93 is not covered"):
        runtime.fetch_blocking(93)


def test_commit_fills_the_slots_then_resets_them_then_records_consume_events():
    """The fixed P6 order, seen from each consume record that commit makes.

    Fill -> tags ``EMPTY_TAG`` and ``consumed`` True -> consume records -> ``primed`` and
    ``forward_open`` False.  The fill is enqueued before the records on the same stream, so the
    first copy into any slot waits for it.  No public call before commit changes the tags or the
    flags, so the test seeds stale values directly to make their reset order observable.
    """
    driver = Driver(commit=False)
    runtime = driver.runtime
    for ring in driver.slots.values():
        for slot in ring:
            slot.storage.fill_(0x5A)  # what Phase V and the P5 gathers left in the slots
    runtime._tag[ATTN][1] = 7
    runtime._consumed[MOE][2] = False
    runtime._primed = runtime._forward_open = True
    seen = []

    def observe(stream: FakeStream) -> None:
        filled = all(
            bool((slot.storage == 0xFF).all()) for ring in driver.slots.values() for slot in ring
        )
        reset = all(
            runtime.slot_tag(kind, index) == EMPTY_TAG and runtime.slot_consumed(kind, index)
            for kind, index in all_slots()
        )
        seen.append((stream, filled, reset, runtime.primed and runtime.forward_open))

    constructed = {id(entry.event) for entry in driver.log if entry.kind == "record"}
    driver.ops.record_observers.append(observe)
    start = len(driver.log)
    runtime.commit(None, fill_byte=0xFF)
    driver.ops.record_observers.clear()
    records = [entry for entry in driver.log[start:] if entry.kind == "record"]
    assert {id(entry.event) for entry in records} == constructed
    assert len(records) == len(constructed) == RING_DEPTH[ATTN] + RING_DEPTH[MOE]
    assert seen == [(driver.ops.compute, True, True, True)] * len(records)
    assert [entry.kind for entry in driver.stream_ops(start)] == ["record"] * len(records)
    assert not runtime.primed and not runtime.forward_open


# ---------------------------------------------------------------------------
# Two forwards x 92 layers: I1-I4, table 2.4, the last layers
# ---------------------------------------------------------------------------


def test_i1_every_h1_wait_binds_to_the_copy_of_its_layer(two_forwards):
    driver = two_forwards
    log = driver.log
    owners = driver.owners()
    for forward in range(2):
        for layer in driver.layers:
            start, end = driver.spans[(forward, "H1", layer)]
            waits = [index for index in range(start, end) if log[index].kind == "wait"]
            assert len(waits) == len(driver.segments)
            assert len({id(log[index].event) for index in waits}) == len(waits)
            for index in waits:
                wait = log[index]
                assert wait.stream is driver.ops.compute and wait.bound is not None
                record = log[wait.bound]
                assert record.kind == "record" and record.stream is driver.ops.copy
                launch = driver.last_copy_launch_before(wait.bound)
                assert (launch.what, launch.layer) == ("L", layer)
                assert owners[wait.bound][0] == forward
    assert driver.runtime.demand_fetches == 0


def test_i2_every_copy_waits_for_the_previous_holders_of_its_slots(two_forwards):
    driver = two_forwards
    assert driver.assert_war_bindings() == 2 * len(driver.layers)
    assert driver.runtime.war_fences == 0


def test_i3_consume_events_are_recorded_only_where_read_windows_end(two_forwards):
    driver = two_forwards
    log = driver.log
    owners = driver.owners()
    committed = driver.spans[(0, "commit", None)][1]
    per_hook = Counter(owners[index][1] for index in driver.compute_records(committed, len(log)))
    expected = {"H2": 2 * len(driver.layers)}
    if MOE in driver.segments:
        expected["H3"] = 2 * len(driver.layers)
    assert per_hook == expected
    for forward in range(2):
        for layer in driver.layers:
            # H2 records first, so the copy of L(layer + 2) it enqueues binds to that record.
            start, end = driver.spans[(forward, "H2", layer)]
            ops = driver.stream_ops(start, end)
            assert ops[0].kind == "record" and ops[0].stream is driver.ops.compute
    # Every read of a window completes before any later copy into its slot starts.
    reads_per_layer = 3 if MOE in driver.segments else 1
    assert driver.assert_race_free() == 2 * len(driver.layers) * reads_per_layer


def test_i4_copy_fifo_is_in_deadline_order(two_forwards):
    driver = two_forwards
    log = driver.log
    owners = driver.owners()
    fifo = [index for index, entry in enumerate(log) if entry.kind == "launch"]
    assert all(log[index].stream is driver.ops.copy for index in fifo)
    # Deadline: the first read of the copied weights, H1(n) for L(n) and CMOE:690 of n for E(n).
    deadlines = []
    for index in fifo:
        entry = log[index]
        hook = "H1" if entry.what == "L" else "experts"
        deadlines.append(driver.spans[(owners[index][0], hook, entry.layer)][0])
    assert deadlines == sorted(set(deadlines))
    for forward in range(2):
        order = [
            (log[index].what, log[index].layer) for index in fifo if owners[index][0] == forward
        ]
        assert order == [(what, layer) for layer in driver.layers for what in ("L", "E")]
    assert driver.ops.current is driver.ops.compute


def test_one_layer_event_sequence_of_figure_6(two_forwards):
    """Stream operations of each hook of a layer whose H2 enqueues.

    H1: compute waits each ready event.  H2: record the attention consume event, then on the copy
    stream wait each target slot's consume event, launch ``L(l + 2)`` once and record each ready
    event.  H3: record the MoE-dense consume event.  The last two layers are checked by
    ``test_last_layers_enqueue_nothing``.
    """
    driver = two_forwards
    compute, copy = driver.ops.compute, driver.ops.copy
    count = len(driver.segments)
    h1 = [("wait", compute)] * count
    h2 = [("record", compute)] + [("wait", copy)] * count + [("launch", copy)]
    h2 += [("record", copy)] * count
    h3 = [("record", compute)] * (count - 1)
    for forward in range(2):
        for layer in driver.layers[:-2]:
            sequence = {
                hook: [
                    (entry.kind, entry.stream)
                    for entry in driver.stream_ops(*driver.spans[(forward, hook, layer)])
                ]
                for hook in ("H1", "H2", "H3")
            }
            assert sequence == {"H1": h1, "H2": h2, "H3": h3}, f"layer {layer}"


def test_the_host_never_synchronizes_after_commit(two_forwards):
    """Every dependency is a stream wait on an event, so the host runs ahead of the GPU (2.3)."""
    two_forwards.assert_no_host_sync()


def test_last_layers_enqueue_nothing(two_forwards):
    driver = two_forwards
    log = driver.log
    for forward in range(2):
        for layer in driver.layers:
            start, end = driver.spans[(forward, "H2", layer)]
            launched = [
                (log[i].what, log[i].layer) for i in range(start, end) if log[i].kind == "launch"
            ]
            assert launched == ([("L", layer + 2)] if layer + 2 <= 92 else [])
            assert len(driver.compute_records(start, end)) == 1
    dense = Counter(entry.layer for entry in log if entry.kind == "launch" and entry.what == "L")
    assert dense == Counter({layer: 2 for layer in driver.layers})
    assert not driver.runtime.forward_open and not driver.runtime.primed


def test_iteration_boundary_bindings_of_table_2_4():
    driver = Driver()
    driver.forward()
    driver.forward()
    owners = driver.owners()

    def war_binding(layer: int, forward: int) -> dict[str, list[tuple[int, int | None]]]:
        """``{hook: [(forward, layer)]}`` whose consume records the copy of ``L(layer)`` waits."""
        return driver.war_binding(driver.launch_index("L", layer, forward))

    first, second = 0, 1
    # A slots 0/1 were last read by layers 91/92; D slots 0/1/2 by layers 91/92/90.
    assert war_binding(1, second) == {"H2": [(first, 91)], "H3": [(first, 91)]}
    assert war_binding(2, second) == {"H2": [(first, 92)], "H3": [(first, 92)]}
    assert war_binding(3, second) == {"H2": [(second, 1)], "H3": [(first, 90)]}
    assert war_binding(4, second) == {"H2": [(second, 2)], "H3": [(second, 1)]}
    assert war_binding(92, second) == {"H2": [(second, 90)], "H3": [(second, 89)]}
    # The first forward binds to the consume events recorded at the end of setup (P6).
    assert war_binding(1, first) == {"commit": [(first, None)] * 2}
    assert war_binding(2, first) == {"commit": [(first, None)] * 2}
    assert war_binding(3, first) == {"H2": [(first, 1)], "commit": [(first, None)]}
    assert war_binding(4, first) == {"H2": [(first, 2)], "H3": [(first, 1)]}
    # Enqueue points: prime, prime, H2(1), H2(2), ..., H2(90).
    for forward in (first, second):
        for layer in (1, 2):
            assert owners[driver.launch_index("L", layer, forward)][1:] == ("H0", None)
        for layer in range(3, 93):
            assert owners[driver.launch_index("L", layer, forward)][1:] == ("H2", layer - 2)


@pytest.mark.parametrize("layers", [(3, 5, 8, 13, 21, 34), (7,), (4, 9), (0, 1, 2, 3)])
def test_slots_and_enqueue_points_follow_layer_positions(layers):
    driver = Driver(layers=layers)
    runtime = driver.runtime
    for position, layer in enumerate(layers):
        assert runtime.slot_index(ATTN, layer) == position % 2
        assert runtime.slot_index(MOE, layer) == position % 3
    driver.forward()
    driver.forward()
    log = driver.log
    owners = driver.owners()
    for forward in range(2):
        for position, layer in enumerate(layers):
            start, end = driver.spans[(forward, "H2", layer)]
            launched = [log[i].layer for i in range(start, end) if log[i].kind == "launch"]
            assert launched == list(layers[position + 2 : position + 3])
        fifo = [
            (entry.what, entry.layer)
            for index, entry in enumerate(log)
            if entry.kind == "launch" and owners[index][0] == forward
        ]
        assert fifo == [(what, layer) for layer in layers for what in ("L", "E")]
    assert driver.assert_war_bindings() == 2 * len(layers)
    assert driver.assert_race_free() == 2 * len(layers) * 3
    assert runtime.demand_fetches == 0 and runtime.war_fences == 0


# ---------------------------------------------------------------------------
# Demand fetches and fences
# ---------------------------------------------------------------------------


def test_tag_mismatch_takes_one_fenced_demand_fetch():
    """A forward that skipped priming finds L(1) and L(2) missing and fetches each at H1."""
    driver = Driver()
    driver.forward(prime=False)
    runtime = driver.runtime
    assert runtime.demand_fetches == 2 and runtime.war_fences == 0
    log = driver.log
    for layer in (1, 2):
        start, end = driver.spans[(0, "H1", layer)]
        launches = [i for i in range(start, end) if log[i].kind == "launch"]
        assert [(log[i].what, log[i].layer) for i in launches] == [("L", layer)]
        # The copy waits for its slots' consume records (made at the end of setup) and for the
        # fence, recorded on compute at H1.
        assert driver.war_binding(launches[0]) == {"commit": [(0, None)] * 2, "H1": [(0, layer)]}
        # Compute then waits for the ready records that follow the demand copy.
        compute_waits = [
            i
            for i in range(launches[0], end)
            if log[i].kind == "wait" and log[i].stream is driver.ops.compute
        ]
        assert len(compute_waits) == 2 and all(log[i].bound > launches[0] for i in compute_waits)
        names = [entry.what for entry in log[start:end] if entry.kind == "nvtx"]
        assert names == [f"dwdp.dense.wait/L{layer}", f"dwdp.dense.prefetch/L{layer}"]
    for layer in driver.layers[2:]:
        start, end = driver.spans[(0, "H1", layer)]
        assert not [entry for entry in log[start:end] if entry.kind == "launch"]
    driver.forward()
    assert runtime.demand_fetches == 2
    driver.assert_race_free()
    driver.assert_no_host_sync()


def test_one_stale_segment_refetches_the_whole_layer():
    driver = Driver()
    runtime = driver.runtime
    driver.span("H0", None, driver.ensure_primed)
    runtime.prefetch(4)  # out of protocol: L(4) lands in A slot 1 and D slot 0 ahead of time
    assert runtime.war_fences == 0  # their holders, L(2) and L(1), have not been read
    assert (runtime.slot_tag(ATTN, 0), runtime.slot_tag(MOE, 0)) == (1, 4)
    start = len(driver.log)
    runtime.wait_layer(1)
    assert runtime.demand_fetches == 1
    launched = [(entry.what, entry.layer) for entry in driver.log[start:] if entry.kind == "launch"]
    assert launched == [("L", 1)]
    assert (runtime.slot_tag(ATTN, 0), runtime.slot_tag(MOE, 0)) == (1, 1)


def test_tag_mismatch_raises_under_strict():
    driver = Driver(strict=True)
    runtime = driver.runtime
    start = len(driver.log)
    with pytest.raises(DenseProtocolError, match="STRICT forbids demand fetches"):
        runtime.wait_layer(1)
    assert driver.stream_ops(start) == []
    assert runtime.demand_fetches == 0 and not runtime.forward_open
    assert runtime.slot_tag(ATTN, 0) == EMPTY_TAG
    driver.forward()
    driver.forward()
    assert runtime.demand_fetches == 0 and runtime.war_fences == 0
    driver.assert_race_free()
    driver.assert_no_host_sync()


def test_h2_and_h3_tag_mismatch_always_raise():
    driver = Driver()
    runtime = driver.runtime
    with pytest.raises(DenseProtocolError, match="attn slot 0 holds nothing"):
        runtime.attention_done(1)
    with pytest.raises(DenseProtocolError, match="moe slot 0 holds nothing"):
        runtime.layer_done(1)
    driver.span("H0", None, driver.ensure_primed)
    driver.layer(1)
    start = len(driver.log)
    with pytest.raises(DenseProtocolError, match="attn slot 0 holds layer 3"):
        runtime.attention_done(1)  # a second H2(1): L(3) already refilled the slot
    with pytest.raises(DenseProtocolError, match="moe slot 0 holds layer 1"):
        runtime.layer_done(4)  # H3 of a layer whose H1 never ran
    assert driver.stream_ops(start) == []


@pytest.mark.parametrize("strict", [False, True], ids=["default", "strict"])
@pytest.mark.parametrize("segments", [BOTH, ATTN_ONLY], ids=["attn+moe", "attn"])
def test_h2_and_h3_raise_when_h1_did_not_open_the_read_window(segments, strict):
    """Primed ``L(1)`` holds its slots, but no H1 waited for its copy: its reads may race it.

    H2, and with the MoE-dense segment H3, of layer 1 raise whatever ``strict`` says, before they
    record or enqueue anything, and leave the slots as they were: H1 then runs the layer as usual.
    """
    driver = Driver(segments=segments, strict=strict)
    runtime = driver.runtime
    driver.span("H0", None, driver.ensure_primed)
    start = len(driver.log)
    with pytest.raises(
        DenseProtocolError,
        match=r"^attention_done\(L1\): H1 \(wait_layer\) of layer 1 did not open the read window "
        r"of attn slot 0 since L\(1\) was enqueued",
    ):
        runtime.attention_done(1)
    if MOE in segments:
        with pytest.raises(
            DenseProtocolError,
            match=r"^layer_done\(L1\): H1 \(wait_layer\) of layer 1 did not open the read window "
            r"of moe slot 0",
        ):
            runtime.layer_done(1)
    else:
        runtime.layer_done(1)  # nothing to close: the attention ring is the only one
    assert driver.stream_ops(start) == []
    assert (runtime.slot_tag(ATTN, 0), runtime.slot_tag(ATTN, 1)) == (1, 2)  # no L(3)
    for kind in segments:
        assert not runtime.slot_consumed(kind, 0)
    assert runtime.primed and not runtime.forward_open
    driver.forward(prime=False)
    driver.forward()
    assert runtime.demand_fetches == 0 and runtime.war_fences == 0
    driver.assert_race_free()
    driver.assert_no_host_sync()


@pytest.mark.parametrize("hook", ["attention_done", "layer_done"])
def test_a_layer_without_h1_aborts_its_forward_and_the_next_h0_recovers(hook):
    """Layer 5 runs without H1, so its first closing hook raises; the next forward recovers.

    ``L(5)`` was enqueued at H2(3) and holds its slots, but no read window of it is open.  The
    forward is open, so the next H0 runs the WAR barrier and primes again.
    """
    driver = Driver()
    runtime = driver.runtime
    driver.span("H0", None, driver.ensure_primed)
    for layer in driver.layers[:4]:
        driver.layer(layer)
    assert (runtime.slot_tag(ATTN, 0), runtime.slot_tag(MOE, 1)) == (5, 5)
    start = len(driver.log)
    with pytest.raises(DenseProtocolError, match=rf"^{hook}\(L5\): H1 \(wait_layer\) of layer 5"):
        getattr(runtime, hook)(5)
    assert driver.stream_ops(start) == []
    assert runtime.forward_open
    driver.forward_index += 1
    driver.forward()
    driver.forward()
    assert runtime.demand_fetches == 0 and runtime.war_fences == 0
    driver.assert_race_free()
    driver.assert_no_host_sync()


def test_h2_and_h3_raise_when_the_read_window_is_already_closed():
    """A second H3(1), or a second H2 of the last layer, finds a window its H1 opened closed.

    ``D(1)`` keeps its slot until H2(2) enqueues ``L(4)``, and nothing refills the attention slot
    of the last layer; the repeated hook raises before it records anything.
    """
    driver = Driver()
    runtime = driver.runtime
    driver.span("H0", None, driver.ensure_primed)
    driver.layer(1)
    start = len(driver.log)
    with pytest.raises(
        DenseProtocolError,
        match=r"^layer_done\(L1\): the read window of layer 1 in moe slot 0 is already closed; "
        r"layer_done ran again without H1 \(wait_layer\) of layer 1",
    ):
        runtime.layer_done(1)
    assert driver.stream_ops(start) == []
    for layer in driver.layers[1:]:
        driver.layer(layer)
    last = driver.layers[-1]
    start = len(driver.log)
    with pytest.raises(
        DenseProtocolError,
        match=rf"^attention_done\(L{last}\): the read window of layer {last} in attn slot 1 is "
        r"already closed",
    ):
        runtime.attention_done(last)
    assert driver.stream_ops(start) == []
    assert not runtime.forward_open
    driver.forward_index += 1
    driver.forward()
    assert runtime.demand_fetches == 0 and runtime.war_fences == 0
    driver.assert_race_free()
    driver.assert_no_host_sync()


@pytest.mark.parametrize("prime", [True, False], ids=["primed", "demand-fetched"])
def test_fence_when_a_holder_has_an_open_read_window(prime):
    """A forward dies inside attention(1); a prefetch that skips the WAR barrier gets the fence.

    H1(1) opened the read windows of ``A(1)`` and ``D(1)``, whether priming or H1's demand fetch
    enqueued ``L(1)``, and no consume record closed them: the consume events' latest records (from
    setup) precede the dead forward's reads.  The copy waits for those records and for the fence,
    recorded on compute after the reads.
    """
    driver = Driver()
    runtime = driver.runtime
    driver.forward(stop_at=("H2", 1), prime=prime)
    assert runtime.forward_open and runtime.demand_fetches == int(not prime)
    driver.span("again", None, lambda: runtime.prefetch(1))
    assert runtime.war_fences == 1  # one per prefetch, not one per slot
    (launch,) = driver.launches_in((1, "again", None))
    assert driver.war_binding(launch) == {"commit": [(0, None)] * 2, "again": [(1, None)]}
    driver.forward()  # H0 runs the barrier, then primes
    assert runtime.war_fences == 1 and runtime.demand_fetches == int(not prime)
    driver.assert_race_free()
    driver.assert_no_host_sync()


def test_fence_when_a_later_forward_reopens_a_window():
    """H1 reopens the windows of a holder that an earlier forward already read and consumed.

    With one covered layer, ``L(7)`` stays in its slots after a forward, so a forward without
    priming reads it again with no new copy, then dies inside attention.  The slots' consume
    records come from the first forward and precede these reads, so a prefetch that skips the
    WAR barrier still needs the fence.
    """
    driver = Driver(layers=(7,))
    runtime = driver.runtime
    driver.forward()
    driver.forward(stop_at=("H2", 7), prime=False)
    assert runtime.demand_fetches == 0 and not runtime.slot_consumed(ATTN, 0)
    driver.span("again", None, lambda: runtime.prefetch(7))
    assert runtime.war_fences == 1
    (launch,) = driver.launches_in((2, "again", None))
    assert driver.war_binding(launch) == {"H2": [(0, 7)], "H3": [(0, 7)], "again": [(2, None)]}
    driver.assert_race_free()
    driver.assert_no_host_sync()


def test_fence_fallback_raises_under_strict():
    driver = Driver(strict=True)
    runtime = driver.runtime
    driver.forward(stop_at=("H2", 1))  # H1(1) opened the read windows of A(1) and D(1)
    start = len(driver.log)
    with pytest.raises(DenseProtocolError, match="STRICT forbids the fence fallback"):
        runtime.prefetch(1)
    assert driver.stream_ops(start) == []
    assert runtime.war_fences == 0
    assert runtime.slot_tag(ATTN, 0) == 1 and not runtime.slot_consumed(ATTN, 0)
    # The next H0 resets the slots with the WAR barrier, so priming needs no fence.
    driver.forward()
    driver.forward()
    assert runtime.war_fences == 0 and runtime.demand_fetches == 0
    driver.assert_race_free()
    driver.assert_no_host_sync()


@pytest.mark.parametrize("strict", [False, True], ids=["default", "strict"])
def test_a_holder_nobody_has_read_needs_no_fence(strict):
    """Priming twice refills slots whose holders no H1 has opened: no fence, no strict error.

    The second copy of each layer waits for the same consume records as the first, which follow
    every read of the slots, and runs after the first on the copy stream.
    """
    driver = Driver(strict=strict)
    runtime = driver.runtime
    driver.span("H0", None, driver.ensure_primed)
    driver.span("again", None, driver.prime_all)
    assert runtime.war_fences == 0 and runtime.primed
    primed = driver.launches_in((0, "H0", None))
    again = driver.launches_in((0, "again", None))
    assert len(primed) == len(again) == 2
    setup_records = {"commit": [(0, None)] * 2}
    for first, second in zip(primed, again):
        assert driver.war_binding(first) == driver.war_binding(second) == setup_records
    driver.forward()
    assert driver.stream_ops(*driver.spans[(0, "H0", None)]) == []  # primed: H0 enqueues nothing
    assert runtime.war_fences == 0 and runtime.demand_fetches == 0
    driver.assert_race_free()
    driver.assert_no_host_sync()


@pytest.mark.parametrize("strict", [False, True], ids=["default", "strict"])
def test_an_interrupted_priming_is_retried_without_a_fence(strict):
    """Priming stops after ``L(1), E(1)`` because the launch of ``L(2)`` fails; H0 primes again.

    Nobody read ``L(1)`` and the failed ``L(2)`` left its slots empty, so the retry needs no fence
    and also works under ``strict``.
    """
    driver = Driver(strict=strict)
    runtime = driver.runtime
    driver.forward()

    def interrupted_priming() -> None:
        runtime.prefetch(1)
        driver.launch_expert(1)
        driver.fail_launch = True
        runtime.prefetch(2)

    with pytest.raises(RuntimeError, match=r"launch of L\(2\) failed"):
        driver.span("prime", None, interrupted_priming)
    assert not runtime.primed and not runtime.forward_open
    assert (runtime.slot_tag(ATTN, 0), runtime.slot_tag(MOE, 0)) == (1, 1)
    assert (runtime.slot_tag(ATTN, 1), runtime.slot_tag(MOE, 1)) == (EMPTY_TAG, EMPTY_TAG)
    driver.forward()
    # The retried copies wait for the last reads of their slots, in the forward before.
    retry_1, retry_2 = driver.launches_in((1, "H0", None))
    assert driver.war_binding(retry_1) == {"H2": [(0, 91)], "H3": [(0, 91)]}
    assert driver.war_binding(retry_2) == {"H2": [(0, 92)], "H3": [(0, 92)]}
    driver.forward()
    assert runtime.war_fences == 0 and runtime.demand_fetches == 0
    driver.assert_race_free()
    driver.assert_no_host_sync()


def test_fence_covers_the_reads_of_a_forward_that_died_without_a_barrier():
    driver = Driver()
    runtime = driver.runtime
    driver.forward()
    driver.forward(stop_at=("H2", 5))  # dies inside attention(5)
    # A path that skips the barrier and primes directly.  H1(5) opened the windows of A slot 0
    # and D slot 1 (layer 5); A slot 1 and D slot 2 hold layer 6, which no H1 opened; D slot 0
    # (layer 4) was consumed at H3(4).
    driver.span("prime", None, lambda: (runtime.prefetch(1), runtime.prefetch(2)))
    assert runtime.war_fences == 2
    # Each copy waits for its slots' latest consume records and for the fence.
    launch_1, launch_2 = driver.launches_in((2, "prime", None))
    assert driver.war_binding(launch_1) == {"H2": [(1, 3)], "H3": [(1, 4)], "prime": [(2, None)]}
    assert driver.war_binding(launch_2) == {"H2": [(1, 4)], "H3": [(1, 2)], "prime": [(2, None)]}
    driver.forward(prime=False)
    # L(3), enqueued at H2(1), refills D slot 2, whose holder (layer 6) was never read: the slot's
    # latest consume record, H3(3) of the dead forward, follows its last reads.
    assert driver.war_binding(driver.launch_index("L", 3, 2)) == {"H2": [(2, 1)], "H3": [(1, 3)]}
    assert runtime.war_fences == 2 and runtime.demand_fetches == 0
    driver.assert_race_free()
    driver.assert_no_host_sync()


def test_demand_fetches_and_fence_fallbacks_warn_at_powers_of_two(monkeypatch):
    """Each warning names the layer and the running count; counts 1, 2, 4, 8, ... are logged."""
    capture = CaptureLogger()
    monkeypatch.setattr(runtime_module, "logger", capture)
    driver = Driver()
    runtime = driver.runtime

    def logged(pattern: str) -> list[tuple[int, int]]:
        found = [re.search(pattern, warning) for warning in capture.warnings]
        return [(int(match[1]), int(match[2])) for match in found if match is not None]

    driver.forward(prime=False)  # demand fetches #1 and #2, at H1(1) and H1(2)
    driver.forward(prime=False)  # #3 and #4
    assert runtime.demand_fetches == 4
    assert logged(r"wait_layer\(L(\d+)\).*demand fetch #(\d+)") == [(1, 1), (2, 2), (2, 4)]
    driver.span("H0", None, driver.ensure_primed)
    for _ in range(4):
        driver.forward(stop_at=("H2", 1), prime=False)  # H1(1) opens L(1)'s windows, then dies
        runtime.prefetch(1)  # refills both slots of L(1) without the WAR barrier: fence fallback
    assert runtime.war_fences == 4 and runtime.demand_fetches == 4
    assert logged(r"prefetch\(L(\d+)\).*fence fallback #(\d+)") == [(1, 1), (1, 2), (1, 4)]
    assert len(capture.warnings) == 6
    assert all(warning.startswith(f"{LOG_PREFIX} ") for warning in capture.warnings)


def test_a_failed_launch_leaves_its_slots_empty():
    driver = Driver()
    runtime = driver.runtime
    driver.span("H0", None, driver.ensure_primed)
    driver.layer(1)
    driver.span("H1", 2, lambda: runtime.wait_layer(2))
    driver.read(ATTN, 2)
    driver.fail_launch = True
    with pytest.raises(RuntimeError, match=r"launch of L\(4\) failed"):
        runtime.attention_done(2)
    # L(4) targets A slot 1 and D slot 0; their contents are unknown, so nothing trusts them.
    assert runtime.slot_tag(ATTN, 1) == EMPTY_TAG and runtime.slot_tag(MOE, 0) == EMPTY_TAG
    start = len(driver.log)
    runtime.wait_layer(4)
    assert runtime.demand_fetches == 1
    assert [(e.what, e.layer) for e in driver.log[start:] if e.kind == "launch"] == [("L", 4)]
    driver.assert_no_host_sync()


def test_wait_layer_requires_eager_single_stream_execution():
    for flag, match in (("capturing", "capturing a CUDA graph"), ("multi_stream", "multi-stream")):
        driver = Driver()
        driver.span("H0", None, driver.ensure_primed)
        setattr(driver.ops, flag, True)
        start = len(driver.log)
        with pytest.raises(DenseProtocolError, match=match):
            driver.runtime.wait_layer(1)
        assert driver.log[start:] == []
        assert driver.runtime.primed and not driver.runtime.forward_open


@pytest.mark.parametrize(
    "flag,match",
    [
        ("capturing", "while the current stream is capturing a CUDA graph"),
        ("multi_stream", "with multi-stream execution on"),
    ],
    ids=["capture", "multi-stream"],
)
def test_h0_refuses_before_it_records_or_enqueues_anything(flag, match):
    """After an abort H0 owes a WAR barrier and a priming; refused, it makes neither.

    Under a CUDA graph capture the barrier's records and the priming copies would go into the
    graph instead of running.  H0 raises before both, so nothing reaches any stream, the runtime
    still owes the barrier, and the next eager forward recovers as after any abort.
    """
    driver = Driver()
    runtime = driver.runtime
    driver.forward()
    driver.forward(stop_at=("H2", 41))  # dies with the attention and MoE-dense windows of 41 open
    assert runtime.forward_open and not runtime.primed

    def slot_state():
        return [
            (runtime.slot_tag(kind, i), runtime.slot_consumed(kind, i)) for kind, i in all_slots()
        ]

    before = slot_state()
    setattr(driver.ops, flag, True)
    start = len(driver.log)
    with pytest.raises(DenseProtocolError, match=re.escape(f"{H0_HOOK} ") + match):
        driver.ensure_primed()
    assert driver.log[start:] == []
    assert runtime.forward_open and not runtime.primed and slot_state() == before
    setattr(driver.ops, flag, False)
    driver.forward()
    driver.forward()
    log = driver.log
    h0_start, h0_end = driver.spans[(2, "H0", None)]
    assert len(driver.compute_records(h0_start, h0_end)) == sum(RING_DEPTH.values())
    launched = [
        (log[i].what, log[i].layer) for i in range(h0_start, h0_end) if log[i].kind == "launch"
    ]
    assert launched == [("L", 1), ("E", 1), ("L", 2), ("E", 2)]
    assert runtime.demand_fetches == 0 and runtime.war_fences == 0
    driver.assert_race_free()
    driver.assert_no_host_sync()


def test_require_eager_checks_the_execution_mode_in_every_phase():
    ops = FakeOps()
    runtime = DenseDWDP(K3_LAYERS, BOTH, make_rings(), ops.copy, ops=ops)
    runtime.set_launcher(CallableLauncher(ops))
    phases = {
        "setup": lambda: None,
        "armed": lambda: runtime.commit(None, fill_byte=0),
        "released": runtime.release,
    }
    for phase, enter in phases.items():
        enter()
        start = len(ops.log)
        runtime.require_eager("probe")
        for flag, match in (
            ("capturing", "^probe while the current stream is capturing a CUDA graph"),
            ("multi_stream", "^probe with multi-stream execution on"),
        ):
            setattr(ops, flag, True)
            with pytest.raises(DenseProtocolError, match=match):
                runtime.require_eager("probe")
            setattr(ops, flag, False)
        assert ops.log[start:] == [], phase


def test_nvtx_ranges_name_each_layer():
    driver = Driver()
    driver.forward()
    names = Counter(entry.what for entry in driver.log if entry.kind == "nvtx")
    expected = Counter(f"dwdp.dense.wait/L{layer}" for layer in K3_LAYERS)
    expected.update(f"dwdp.dense.prefetch/L{layer}" for layer in K3_LAYERS)
    assert names == expected


# ---------------------------------------------------------------------------
# Priming, aborts and the WAR barrier
# ---------------------------------------------------------------------------


def test_priming_flags_follow_the_forward():
    driver = Driver()
    runtime = driver.runtime
    with pytest.raises(DenseProtocolError, match=r"L\(1\) attn\[0\] holds nothing"):
        runtime.mark_primed()
    runtime.prefetch(1)
    with pytest.raises(DenseProtocolError, match=r"L\(2\) attn\[1\] holds nothing"):
        runtime.mark_primed()
    runtime.prefetch(2)
    runtime.mark_primed()
    assert runtime.primed and not runtime.forward_open
    driver.layer(1)
    assert not runtime.primed and runtime.forward_open
    with pytest.raises(DenseProtocolError, match="forward is open"):
        runtime.mark_primed()
    for layer in driver.layers[1:-1]:
        driver.layer(layer)
    assert runtime.forward_open
    driver.layer(driver.layers[-1])
    assert not runtime.primed and not runtime.forward_open


def test_war_barrier_then_reprime_after_an_abort():
    driver = Driver()
    runtime = driver.runtime
    driver.forward()
    driver.forward(stop_at=("H3", 41))  # dies inside the MoE block of layer 41
    assert runtime.forward_open and not runtime.primed
    start = len(driver.log)
    driver.span("barrier", None, runtime.war_barrier)
    records = driver.stream_ops(start)
    assert [entry.kind for entry in records] == ["record"] * (RING_DEPTH[ATTN] + RING_DEPTH[MOE])
    assert all(entry.stream is driver.ops.compute for entry in records)
    assert len({id(entry.event) for entry in records}) == len(records)
    for kind, index in all_slots():
        assert runtime.slot_tag(kind, index) == EMPTY_TAG and runtime.slot_consumed(kind, index)
    assert not runtime.primed and not runtime.forward_open
    driver.span("prime", None, driver.prime_all)
    owners = driver.owners()
    for layer in (1, 2):
        waits = driver.copy_waits_before(driver.launch_index("L", layer, driver.forward_index))
        assert Counter(owners[driver.log[w].bound][1] for w in waits) == Counter({"barrier": 2})
    assert runtime.primed and runtime.war_fences == 0
    driver.forward()
    assert runtime.demand_fetches == 0 and runtime.war_fences == 0
    driver.assert_race_free()
    driver.assert_no_host_sync()


@pytest.mark.parametrize("segments", [BOTH, ATTN_ONLY], ids=["attn+moe", "attn"])
@pytest.mark.parametrize(
    "stop_at", [("H1", 1), ("H2", 1), ("H3", 1), ("H2", 41), ("H3", 41), ("H1", 60), ("H3", 92)]
)
def test_next_h0_recovers_from_an_aborted_forward(segments, stop_at):
    driver = Driver(segments=segments)
    runtime = driver.runtime
    driver.forward()
    driver.forward(stop_at=stop_at)
    opened = stop_at != ("H1", 1)
    assert runtime.forward_open is opened and runtime.primed is not opened
    driver.forward()
    driver.forward()
    log = driver.log
    start, end = driver.spans[(2, "H0", None)]
    barrier = driver.compute_records(start, end)
    launched = [(log[i].what, log[i].layer) for i in range(start, end) if log[i].kind == "launch"]
    if opened:
        # The barrier records every consume event; L(1) and L(2) wait for those records.
        assert len(barrier) == sum(RING_DEPTH[kind] for kind in segments)
        assert launched == [("L", 1), ("E", 1), ("L", 2), ("E", 2)]
        for layer in (1, 2):
            waits = driver.copy_waits_before(driver.launch_index("L", layer, 2))
            assert waits and all(log[w].bound in barrier for w in waits)
    else:
        # Died after priming, before H1 of the first layer: the primed slots are used as they are.
        assert barrier == [] and launched == []
    assert runtime.demand_fetches == 0 and runtime.war_fences == 0
    driver.assert_race_free()
    driver.assert_no_host_sync()


def test_attention_only_handles_the_attention_ring_only():
    driver = Driver(segments=ATTN_ONLY)
    runtime = driver.runtime
    log = driver.log
    assert runtime.segments == ATTN_ONLY and set(runtime.slots) == {ATTN}
    construct_end = driver.spans[(0, "commit", None)][0]
    assert len(driver.compute_records(0, construct_end)) == RING_DEPTH[ATTN]
    assert len(driver.compute_records(*driver.spans[(0, "commit", None)])) == RING_DEPTH[ATTN]
    driver.forward()
    driver.forward()
    for forward in range(2):
        for layer in driver.layers:
            h1_ops = driver.stream_ops(*driver.spans[(forward, "H1", layer)])
            assert [entry.kind for entry in h1_ops] == ["wait"]
            assert len(driver.compute_records(*driver.spans[(forward, "H2", layer)])) == 1
            assert driver.stream_ops(*driver.spans[(forward, "H3", layer)]) == []
    for index, entry in enumerate(log):
        if entry.kind == "launch" and entry.what == "L":
            assert len(driver.copy_waits_before(index)) == 1
    for call in (
        lambda: runtime.slot_tag(MOE, 0),
        lambda: runtime.slot_consumed(MOE, 0),
        lambda: runtime.slot_index(MOE, 1),
    ):
        with pytest.raises(DenseProtocolError, match="moe is not enabled"):
            call()
    assert not runtime.forward_open
    assert driver.assert_race_free() == 2 * len(driver.layers)


# ---------------------------------------------------------------------------
# The slot watchdog of TRTLLM_DWDP_DENSE_DEBUG=2
# ---------------------------------------------------------------------------

WATCHED_LAYERS = tuple(range(1, 9))
#: Blob bytes of every watched ``(layer, segment)``: not whole words, and inside one ring slot.
WATCHED_BLOB = 2 * ROW_ALIGN - 3


def blob_of(layer: int, kind: SegmentKind) -> torch.Tensor:
    """The bytes a correct gather of ``(layer, kind)`` leaves in its slot."""
    first = 2 * layer + (kind is MOE)
    return (torch.arange(WATCHED_BLOB) * 7 + first).remainder(251).to(torch.uint8)


class WatchedDriver(Driver):
    """A driver whose dense copies write each layer's blob into its slots, under a CPU watchdog.

    Every watchdog check and read is logged (``check``, ``poll``), so the tests can tell which
    hook made it.  ``corrupt`` names the ``(layer, segment)`` blobs the copies get wrong.
    """

    def __init__(self, *, strict: bool = False) -> None:
        self.corrupt: set[tuple[int, SegmentKind]] = set()
        expected = {
            (layer, kind): (WATCHED_BLOB, tensor_checksum(blob_of(layer, kind)))
            for layer in WATCHED_LAYERS
            for kind in BOTH
        }
        self.watchdog = SlotWatchdog(WATCHED_LAYERS, BOTH, expected, "cpu")
        check, poll = self.watchdog.check, self.watchdog.poll

        def logged_check(layer: int, segment: SegmentKind, slot: SlotBuffer) -> None:
            self.log.append(Entry("check", what=SegmentKind(segment).value, layer=layer))
            check(layer, segment, slot)

        def logged_poll() -> list:
            self.log.append(Entry("poll"))
            return poll()

        self.watchdog.check = logged_check
        self.watchdog.poll = logged_poll
        super().__init__(layers=WATCHED_LAYERS, strict=strict, watchdog=self.watchdog)

    def launch_dense(self, layer: int) -> None:
        super().launch_dense(layer)
        for kind in self.segments:
            data = blob_of(layer, kind)
            if (layer, kind) in self.corrupt:
                data[WATCHED_BLOB // 2] ^= 0x5A
            self.slots[kind][self.runtime.slot_index(kind, layer)].blob_view(WATCHED_BLOB).copy_(
                data
            )


@pytest.mark.parametrize("strict", [False, True], ids=["lenient", "strict"])
def test_the_slot_watchdog_checks_every_h1_and_reads_its_counters_at_the_last_h3(
    monkeypatch, strict
):
    """``DEBUG=2``: each H1 checks the slots it waited for; H3 of the last layer reads the result.

    Clean forwards check every ``(layer, segment)`` once, inside that layer's H1, and find
    nothing; the counters are read once per forward, inside H3 of the last layer.  A copy that
    brings other bytes is named after the forward: logged, or (``STRICT``) raised once the
    forward is closed.  The forwards after it are clean again.
    """
    errors: list[str] = []
    capture = CaptureLogger()
    capture.error = errors.append
    monkeypatch.setattr(runtime_module, "logger", capture)
    driver = WatchedDriver(strict=strict)
    runtime = driver.runtime
    last = WATCHED_LAYERS[-1]
    for _ in range(2):
        driver.forward()
    per_forward = len(WATCHED_LAYERS) * len(BOTH)
    assert (runtime.watchdog_checks, runtime.watchdog_mismatches) == (2 * per_forward, 0)
    owners = driver.owners()
    checks = Counter(
        (owners[index], entry.layer, entry.what)
        for index, entry in enumerate(driver.log)
        if entry.kind == "check"
    )
    assert checks == Counter(
        ((forward, "H1", layer), layer, kind.value)
        for forward in range(2)
        for layer in WATCHED_LAYERS
        for kind in BOTH
    )
    polls = [owners[index] for index, entry in enumerate(driver.log) if entry.kind == "poll"]
    assert polls == [(0, "H3", last), (1, "H3", last)]
    assert errors == []

    driver.corrupt.add((5, MOE))
    pattern = (
        r"slot watchdog: 1 slot checks at H1 found other bytes .*\(layers\.5 moe x1; 1 so far\)"
    )
    if strict:
        with pytest.raises(DenseProtocolError, match=pattern):
            driver.forward()
        driver.forward_index += 1
        assert errors == []
    else:
        driver.forward()
        assert len(errors) == 1 and re.search(pattern, errors[0]), errors
    assert not runtime.forward_open and not runtime.primed
    assert (runtime.watchdog_checks, runtime.watchdog_mismatches) == (3 * per_forward, 1)
    driver.corrupt.clear()
    driver.forward()
    assert (runtime.watchdog_checks, runtime.watchdog_mismatches) == (4 * per_forward, 1)
    assert len(errors) == (0 if strict else 1)
    assert runtime.demand_fetches == 0 and runtime.war_fences == 0
    driver.assert_race_free()


# ---------------------------------------------------------------------------
# Teardown
# ---------------------------------------------------------------------------


def test_release_order_and_idempotence():
    ops = FakeOps()
    store = FakeStore(ops.log)
    runtime = DenseDWDP(K3_LAYERS, BOTH, make_rings(), ops.copy, ops=ops, store=store)
    launcher = CallableLauncher(ops)
    launcher_ref = weakref.ref(launcher)
    runtime.set_launcher(launcher)
    del launcher
    runtime.commit(FakeBindings(ops.log), fill_byte=0)
    start = len(ops.log)
    runtime.release()
    assert [entry.kind for entry in ops.log[start:]] == ["sync", "detach", "store"]
    assert launcher_ref() is None
    assert dict(runtime.slots) == {}
    runtime.release()
    assert len(ops.log) == start + 3 and store.releases == 1
    # Introspection keeps reporting the last state; every other call raises.
    assert runtime.covers(1) and runtime.layers == K3_LAYERS and runtime.segments == BOTH
    assert runtime.slot_tag(ATTN, 0) == EMPTY_TAG and runtime.slot_consumed(MOE, 2)
    assert not runtime.primed and not runtime.forward_open
    calls = {
        "wait_layer": lambda: runtime.wait_layer(1),
        "attention_done": lambda: runtime.attention_done(1),
        "layer_done": lambda: runtime.layer_done(1),
        "prefetch": lambda: runtime.prefetch(1),
        "mark_primed": runtime.mark_primed,
        "war_barrier": runtime.war_barrier,
        "set_launcher": lambda: runtime.set_launcher(CallableLauncher(ops)),
        "fetch_blocking": lambda: runtime.fetch_blocking(1),
        "commit": lambda: runtime.commit(None, fill_byte=0),
    }
    for name, call in calls.items():
        with pytest.raises(DenseProtocolError, match=f"{name} called after release"):
            call()


def test_release_before_commit_releases_the_store():
    ops = FakeOps()
    store = FakeStore(ops.log)
    runtime = DenseDWDP(K3_LAYERS, BOTH, make_rings(), ops.copy, ops=ops, store=store)
    start = len(ops.log)
    runtime.release()
    assert [entry.kind for entry in ops.log[start:]] == ["sync", "store"]


@pytest.mark.parametrize("failing", ["sync", "detach"])
def test_release_runs_every_step_and_raises_the_first_error(failing):
    ops = FakeOps()
    first = RuntimeError(f"{failing} failed")
    ops.sync_error = first if failing == "sync" else None
    bindings = FakeBindings(ops.log, error=first if failing == "detach" else None)
    store = FakeStore(ops.log, error=ValueError("store failed"))
    runtime = DenseDWDP(K3_LAYERS, BOTH, make_rings(), ops.copy, ops=ops, store=store)
    runtime.set_launcher(CallableLauncher(ops))
    runtime.commit(bindings, fill_byte=0)
    start = len(ops.log)
    with pytest.raises(RuntimeError, match=f"{failing} failed"):
        runtime.release()
    assert [entry.kind for entry in ops.log[start:]] == ["sync", "detach", "store"]
    runtime.release()
    assert len(ops.log) == start + 3


def test_del_tolerates_a_constructor_that_raised():
    DenseDWDP.__new__(DenseDWDP).__del__()
    ops = FakeOps()
    store = FakeStore(ops.log)
    with pytest.raises(ValueError):
        DenseDWDP((), BOTH, make_rings(), ops.copy, ops=ops, store=store)
    gc.collect()
    assert store.releases == 0  # the caller keeps the store when the constructor raises


def test_del_never_raises_once_the_module_globals_are_gone():
    """At interpreter exit a runtime can be collected after its module's globals became None."""
    ops = FakeOps()
    released = DenseDWDP(K3_LAYERS, BOTH, make_rings(), ops.copy, ops=ops)
    released.release()
    store = FakeStore(ops.log)
    unreleased = DenseDWDP(K3_LAYERS, BOTH, make_rings(), ops.copy, ops=ops, store=store)
    with pytest.MonkeyPatch.context() as patch:
        for name in ("_Phase", "logger", "LOG_PREFIX", "DenseProtocolError"):
            patch.setattr(runtime_module, name, None)
        released.__del__()
        unreleased.__del__()  # its release fails without the globals; the finalizer drops that
    assert store.releases == 0
    unreleased.release()
    assert store.releases == 1


def test_del_releases_an_unreleased_runtime_and_never_raises():
    ops = FakeOps()
    store = FakeStore(ops.log, error=RuntimeError("store failed"))
    runtime = DenseDWDP(K3_LAYERS, BOTH, make_rings(), ops.copy, ops=ops, store=store)
    runtime.__del__()
    assert store.releases == 1
    assert [entry.kind for entry in ops.log if entry.kind in ("sync", "store")] == ["sync", "store"]
    runtime.__del__()
    assert store.releases == 1
    other_store = FakeStore(ops.log)
    other = DenseDWDP(K3_LAYERS, BOTH, make_rings(), ops.copy, ops=ops, store=other_store)
    del other
    gc.collect()
    assert other_store.releases == 1
