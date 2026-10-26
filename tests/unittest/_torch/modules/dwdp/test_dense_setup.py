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
"""CPU tests of the dense DWDP setup.

Every rank of an ``N``-rank dense group is a thread that runs ``setup_dense_dwdp`` on its own copy
of the toy Kimi-K3 decoder of ``test_dense_layout.py``.  The threads gather over
``StoreDwdpGroup`` on one in-process ``HashStore`` with every wait bounded, so a rank that skips or
adds a collective fails the test instead of hanging it.  Each rank keeps its rows in the real
``LoopbackShardStore``; fakes stand in for the device: ``FakeOps`` of ``test_dense_protocol.py``
for the stream and event seam and ``FakeLauncher``, which runs the real copy plans as host
memmoves, each checked to read a store chunk and write a live slot.  Faults are injected in each
phase.
"""

from __future__ import annotations

import ctypes
import gc
import re
import sys
import threading
import traceback
import weakref
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest
import torch
from test_dense_layout import COVERED, LAYER_KINDS, TOY_MIN_BYTES, ToyModel, collect
from test_dense_protocol import FakeOps
from torch import nn

from tensorrt_llm._torch.modules.dwdp.dense import repoint as dense_repoint
from tensorrt_llm._torch.modules.dwdp.dense import setup as dense_setup
from tensorrt_llm._torch.modules.dwdp.dense.config import DenseConfig
from tensorrt_llm._torch.modules.dwdp.dense.runtime import DenseDWDP
from tensorrt_llm._torch.modules.dwdp.dense.setup import PhaseReport, setup_dense_dwdp
from tensorrt_llm._torch.modules.dwdp.dense.store import LoopbackShardStore, StorePayload
from tensorrt_llm._torch.modules.dwdp.dense.types import (
    DenseLayout,
    DenseRepointError,
    DenseSetupError,
    LayerCopyPlan,
    SegmentKind,
)
from tensorrt_llm._torch.modules.dwdp.rendezvous import StoreDwdpGroup

pytestmark = pytest.mark.cpu_only

#: Seconds one collective waits for the slowest rank; a rank that skips a collective times out.
COLLECTIVE_TIMEOUT_S = 20
#: Seconds a test waits for its rank threads.
JOIN_TIMEOUT_S = 90
#: Free device memory every rank reports.
PLENTY_FREE_BYTES = 64 << 30
#: The last MoE-dense anchor of the last covered layer; the corruption faults flip its last byte.
LAST_MOE_ANCHOR = f"layers.{COVERED[-1]}.block_sparse_moe.routed_expert_up_proj.weight"


# ---------------------------------------------------------------------------
# Module tensor snapshots
# ---------------------------------------------------------------------------


def module_tensors(decoder: nn.Module) -> dict[str, torch.Tensor]:
    """Every tensor the decoder layers hold, by ``layers.<l>.<path>``."""
    found = {}
    for index, layer in enumerate(decoder.layers):
        for path, module in layer.named_modules(prefix=f"layers.{index}"):
            for name, value in [*module._parameters.items(), *module._buffers.items()]:
                if value is not None:
                    found[f"{path}.{name}"] = value
            for name, value in vars(module).items():
                if isinstance(value, torch.Tensor):
                    found[f"{path}.{name}"] = value
                elif isinstance(value, (tuple, list)) and value:
                    if all(isinstance(item, torch.Tensor) for item in value):
                        found.update({f"{path}.{name}[{i}]": item for i, item in enumerate(value)})
    return found


def placement(tensor: torch.Tensor) -> tuple:
    """Where ``tensor`` lives: storage, offset, shape, stride and dtype."""
    storage = tensor.untyped_storage().data_ptr()
    return (
        storage,
        tensor.storage_offset(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
    )


def raw_bytes(tensor: torch.Tensor) -> torch.Tensor:
    """A copy of ``tensor``'s bytes."""
    return tensor.detach().contiguous().reshape(-1).view(torch.uint8).clone()


class Snapshot:
    """A decoder's tensors before setup: placements, bytes and Parameter objects.

    It holds no strong reference to an original storage, so the originals can die at commit.
    """

    def __init__(self, decoder: nn.Module) -> None:
        tensors = module_tensors(decoder)
        self.placements = {path: placement(tensor) for path, tensor in tensors.items()}
        self.values = {path: raw_bytes(tensor) for path, tensor in tensors.items()}
        self.parameters = {p: t for p, t in tensors.items() if isinstance(t, nn.Parameter)}

    def assert_restored(self, decoder: nn.Module) -> None:
        """Every tensor is back: same storage, view, dtype and bytes; same Parameter objects."""
        now = module_tensors(decoder)
        assert now.keys() == self.placements.keys()
        changed = [
            path
            for path, tensor in now.items()
            if placement(tensor) != self.placements[path]
            or not torch.equal(raw_bytes(tensor), self.values[path])
        ]
        assert changed == []
        assert [path for path, param in self.parameters.items() if now[path] is not param] == []


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class HostMemory:
    """The store chunks and slots of a test's ranks; every fake copy must stay inside them.

    Slots are tracked by weak reference.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.chunks = []
        self.slots = []

    def add_chunk(self, start: int, end: int) -> None:
        with self._lock:
            self.chunks.append((start, end))

    def add_slot(self, storage: torch.Tensor) -> None:
        entry = (threading.current_thread(), storage.data_ptr(), storage.numel())
        with self._lock:
            self.slots.append((*entry, weakref.ref(storage.untyped_storage())))

    def live_slots(self, thread: threading.Thread | None = None) -> list[str]:
        """Owners of slot storages that are still alive; only ``thread``'s if given."""
        with self._lock:
            return [s[0].name for s in self.slots if s[3]() is not None and thread in (None, s[0])]

    def check_copy(self, dst: int, src: int, nbytes: int) -> None:
        with self._lock:
            chunks = list(self.chunks)
            slots = [(start, start + size) for _, start, size, ref in self.slots if ref()]
        if not any(start <= src and src + nbytes <= end for start, end in chunks):
            raise AssertionError(f"copy source {src:#x}+{nbytes} is outside every store chunk")
        if not any(start <= dst and dst + nbytes <= end for start, end in slots):
            raise AssertionError(f"copy destination {dst:#x}+{nbytes} is outside every slot")


def _called_from_finalizer() -> bool:
    """Whether a ``__del__`` is on the stack: garbage collection, not setup, releases."""
    frame = sys._getframe(1)
    while frame is not None:
        if frame.f_code.co_name == "__del__":
            return True
        frame = frame.f_back
    return False


class LoopbackStore(LoopbackShardStore):
    """The real loopback store, every rank's rows in this process, with its calls counted.

    ``spec.fail`` makes the step it names raise before it runs.
    """

    def __init__(
        self, layout: DenseLayout, rank: int, device: torch.device, *, owner: Rank
    ) -> None:
        super().__init__(layout, rank, device)
        self.owner = owner
        self.calls = Counter()
        self.finalizer_releases = 0

    def _step(self, name: str, step: Callable[..., object], *args: object) -> object:
        self.owner.log.append(name)
        self.calls[name] += 1
        if self.owner.spec.fail == name:
            raise RuntimeError(f"injected: store {name} failed")
        return step(*args)

    def allocate(self) -> None:
        self._step("allocate", super().allocate)

    def write_local_rows(self) -> None:
        self._step("write", super().write_local_rows)

    def export(self) -> StorePayload:
        return self._step("export", super().export)

    def import_peers(self, payloads: Sequence[StorePayload]) -> None:
        self._step("import", super().import_peers, payloads)
        for bases in self.chunk_bases():
            for chunk, base in zip(self._layout.chunks, bases):
                self.owner.memory.add_chunk(base, base + chunk.handle_bytes)

    def release(self) -> None:
        self.finalizer_releases += _called_from_finalizer()
        self._step("release", super().release)


class FakeLauncher:
    """Runs a rank's copy plans as host memmoves; round 0 is Phase V (P4), round 1 the P5 check.

    With ``spec.corrupt`` (4 or 5) it flips the last byte of ``LAST_MOE_ANCHOR`` in its slot right
    after it copies the last layer in that phase.
    """

    def __init__(self, plans: Mapping[int, LayerCopyPlan], rank: Rank) -> None:
        self.plans = dict(plans)
        self.rank = rank
        self.launches = 0
        self.releases = 0

    def __call__(self, layer: int) -> None:
        rank = self.rank
        phase = 4 + self.launches // len(self.plans)
        self.launches += 1
        rank.log.append(f"launch L{layer}")
        for dst, src, nbytes in self.plans[layer].descriptors:
            rank.memory.check_copy(dst, src, nbytes)
            ctypes.memmove(dst, src, nbytes)
        if (rank.spec.corrupt, layer) == (phase, COVERED[-1]):
            ctypes.c_uint8.from_address(rank.corrupt_address).value ^= 0xFF

    def release(self) -> None:
        self.releases += 1


def corruption_address(layout: DenseLayout, slot_bases: Mapping[SegmentKind, list[int]]) -> int:
    """Address of the last byte of ``LAST_MOE_ANCHOR`` in the slot its layer is gathered into."""
    layer = COVERED[-1]
    anchor = layout.segment(layer, SegmentKind.MOE).anchors[-1]
    base = slot_bases[SegmentKind.MOE][layout.slot_index(SegmentKind.MOE, layer)]
    return base + anchor.blob_offset + anchor.nbytes - 1


class CountingComm:
    """The rank's ``GroupComm``: a ``StoreDwdpGroup`` whose collectives are recorded."""

    def __init__(self, group: StoreDwdpGroup) -> None:
        self.group = group
        self.sent = []

    def Get_rank(self) -> int:
        return self.group.Get_rank()

    def Get_size(self) -> int:
        return self.group.Get_size()

    def allgather(self, obj: object) -> list[object]:
        self.sent.append(obj)
        return self.group.allgather(obj)


class CaptureLogger:
    """Stands in for the setup and repoint modules' ``logger``; keeps each message's thread."""

    def __init__(self) -> None:
        self.records = []

    def __getattr__(self, level: str) -> Callable[[str], None]:
        return lambda message: self.records.append((threading.current_thread(), level, message))


# ---------------------------------------------------------------------------
# Rank threads
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RankSpec:
    """One rank's settings, model, free device memory and injected fault.

    ``fail`` names a store step that raises, or ``"commit"`` for the P6 memory reading after the
    originals are dropped; ``corrupt`` (4 or 5) flips a slot byte in that phase (see
    ``FakeLauncher``); ``perturb`` adds 1 to the first element of a decoder parameter.
    ``attached`` passes the expert DWDP MoE layers (``moe_layers``); False sets up standalone.
    """

    requested: bool = True
    strict: bool = False
    kinds: tuple[str, ...] = LAYER_KINDS
    free_bytes: int = PLENTY_FREE_BYTES
    fail: str | None = None
    corrupt: int | None = None
    perturb: str | None = None
    attached: bool = True


class Rank:
    """Everything one rank thread owns: model, fakes, its log and the outcome of setup."""

    def __init__(self, index: int, spec: RankSpec, memory: HostMemory) -> None:
        self.index = index
        self.spec = spec
        self.memory = memory
        self.log = []
        # Like the deployed VL wrapper: ``model.model`` is the decoder that holds ``layers``.
        self.model = nn.Module()
        self.model.model = self.decoder = ToyModel(kinds=spec.kinds)
        if spec.perturb is not None:
            with torch.no_grad():
                self.decoder.get_parameter(spec.perturb).view(-1)[0] += 1
        self.snapshot = Snapshot(self.decoder)
        self.ops = FakeOps()
        self.transfer = SimpleNamespace(
            copy_stream=self.ops.copy, use_gather=False, use_batched_prefetch=False
        )
        self.stores = []
        self.launchers = []
        #: ``(stores released, live slots of this rank)`` at every ``torch.cuda.empty_cache``.
        self.cache_states = []
        self.corrupt_address = None
        self.thread = self.comm = self.result = self.error = None
        self.trace = ""
        self.mem_get_info_calls = 0

    def store_factory(self, layout: DenseLayout, rank: int, device: torch.device) -> LoopbackStore:
        store = LoopbackStore(layout, rank, device, owner=self)
        self.stores.append(store)
        return store

    def launcher_factory(
        self, plans: Mapping[int, LayerCopyPlan], transfer: SimpleNamespace, device: torch.device
    ) -> FakeLauncher:
        assert transfer is self.transfer and device == torch.device("cpu")
        assert self.ops.current is self.ops.copy, "the launcher is built on the copy stream"
        self.launchers.append(FakeLauncher(plans, self))
        return self.launchers[-1]

    def on_empty_cache(self) -> None:
        self.log.append("empty_cache")
        released = all(store.calls["release"] for store in self.stores)
        self.cache_states.append((released, self.memory.live_slots(self.thread)))

    def mem_get_info(self) -> tuple[int, int]:
        """The rank's free and total device memory; the second reading is the one in P6."""
        self.mem_get_info_calls += 1
        if self.spec.fail == "commit" and self.mem_get_info_calls == 2:
            raise RuntimeError("injected: P6 memory reading failed")
        return self.spec.free_bytes, 4 * PLENTY_FREE_BYTES

    def run(self, store: object, size: int) -> None:
        group = StoreDwdpGroup(
            global_rank=self.index,
            dwdp_size=size,
            master_addr="unused",
            timeout_s=COLLECTIVE_TIMEOUT_S,
            store=store,
        )
        self.comm = CountingComm(group)
        config = DenseConfig(
            enabled=self.spec.requested, min_bytes=TOY_MIN_BYTES, strict=self.spec.strict
        )
        try:
            self.result = setup_dense_dwdp(
                self.model,
                self.transfer,
                self.comm,
                dwdp_rank=self.index,
                dwdp_size=size,
                device_id=0,
                moe_layers=list(COVERED) if self.spec.attached else None,
                config=config,
                store_factory=self.store_factory,
                ops=self.ops,
                device=torch.device("cpu"),
                launcher_factory=self.launcher_factory,
                mem_get_info=self.mem_get_info,
            )
        except Exception as exc:  # the test inspects every rank's outcome
            self.error = (type(exc), str(exc))
            self.trace = traceback.format_exc()


class Group:
    """``N`` rank threads that run dense setup together over one ``HashStore``."""

    def __init__(
        self, specs: Sequence[RankSpec], memory: HostMemory, capture: CaptureLogger, store: object
    ) -> None:
        self.ranks = [Rank(index, spec, memory) for index, spec in enumerate(specs)]
        self.memory = memory
        self.capture = capture
        self.store = store

    def run(self) -> Group:
        for rank in self.ranks:
            rank.thread = threading.Thread(
                target=rank.run,
                args=(self.store, len(self.ranks)),
                name=f"dense-setup-rank{rank.index}",
                daemon=True,
            )
        for rank in self.ranks:
            rank.thread.start()
        for rank in self.ranks:
            rank.thread.join(JOIN_TIMEOUT_S)
        stuck = [rank.index for rank in self.ranks if rank.thread.is_alive()]
        assert stuck == [], f"dense setup did not finish on ranks {stuck}"
        return self

    def messages(self, rank: Rank, level: str) -> list[str]:
        records = self.capture.records
        return [text for thread, kind, text in records if thread is rank.thread and kind == level]

    def assert_collectives(self, count: int) -> list[PhaseReport]:
        """Every rank made ``count`` collectives of ``PhaseReport``s; returns the last gathered."""
        for rank in self.ranks:
            assert len(rank.comm.sent) == count, rank.index
            assert all(isinstance(report, PhaseReport) for report in rank.comm.sent)
        return [rank.comm.sent[-1] for rank in self.ranks]

    def assert_abandoned(self, pattern: str) -> None:
        """Every rank raised ``DenseSetupError`` matching ``pattern``, and its model is intact."""
        for rank in self.ranks:
            assert rank.error is not None and rank.error[0] is DenseSetupError, rank.trace
            assert rank.result is None
            message = rank.error[1]
            assert re.search(pattern, message), message
            rank.snapshot.assert_restored(rank.decoder)
            assert not hasattr(rank.model, "_dense_dwdp_active")
            assert not hasattr(rank.decoder, "_dense_dwdp_active")

    def assert_released(self) -> None:
        """Each store and launcher was released once, by setup; the last step, ``empty_cache``,
        came after the store release and the death of the slots."""
        gc.collect()
        for rank in self.ranks:
            released = [(store.calls["release"], store.finalizer_releases) for store in rank.stores]
            assert released == [(1, 0)] * len(rank.stores), rank.index
            assert [launcher.releases for launcher in rank.launchers] == [1] * len(rank.launchers)
            assert rank.log[-1] == "empty_cache", rank.log
            assert rank.cache_states[-1] == (True, []), rank.index
        assert self.memory.live_slots() == []


@pytest.fixture
def make_group(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[[list[RankSpec]], Group]]:
    """Builds rank groups; routes setup's slots, plans, cache release and logs to the test."""
    torch_distributed = pytest.importorskip("torch.distributed")
    memory = HostMemory()
    capture = CaptureLogger()
    groups = []
    allocate_slot_ring = dense_setup.allocate_slot_ring
    build_copy_plans = dense_setup.build_copy_plans

    def current_rank() -> Rank | None:
        thread = threading.current_thread()
        ranks = [rank for group in groups for rank in group.ranks if rank.thread is thread]
        return ranks[0] if ranks else None

    def recording_allocate_slot_ring(spec: object, device: torch.device) -> tuple:
        ring = allocate_slot_ring(spec, device)
        for slot in ring:
            memory.add_slot(slot.storage)
        return ring

    def recording_build_copy_plans(
        layout: DenseLayout,
        *,
        rank: int,
        chunk_bases: Sequence[Sequence[int]],
        slot_bases: Mapping[SegmentKind, list[int]],
    ) -> dict[int, LayerCopyPlan]:
        current_rank().corrupt_address = corruption_address(layout, slot_bases)
        return build_copy_plans(layout, rank=rank, chunk_bases=chunk_bases, slot_bases=slot_bases)

    def recording_empty_cache() -> None:
        rank = current_rank()
        if rank is not None:
            rank.on_empty_cache()

    monkeypatch.setattr(dense_setup, "allocate_slot_ring", recording_allocate_slot_ring)
    monkeypatch.setattr(dense_setup, "build_copy_plans", recording_build_copy_plans)
    monkeypatch.setattr(dense_setup, "logger", capture)
    monkeypatch.setattr(dense_repoint, "logger", capture)
    monkeypatch.setattr(torch.cuda, "empty_cache", recording_empty_cache)

    def make(specs: list[RankSpec]) -> Group:
        groups.append(Group(specs, memory, capture, torch_distributed.HashStore()))
        return groups[-1]

    yield make
    for group in groups:
        for rank in group.ranks:
            if isinstance(rank.result, DenseDWDP):
                rank.result.release()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("attached", [True, False], ids=["attached", "standalone"])
def test_only_attached_dense_dwdp_prefetches_the_next_forward(
    make_group: Callable, attached: bool
) -> None:
    """With expert DWDP (``moe_layers``) the runtime gathers the next forward's first layers at
    the end of a forward, beside the expert prefetch; standalone the dense manager primes them."""
    group = make_group([RankSpec(attached=attached)] * 2).run()
    group.assert_collectives(6)
    for rank in group.ranks:
        assert isinstance(rank.result, DenseDWDP), rank.trace
        assert rank.result.layers == COVERED
        assert rank.result.prefetch_next_pass is attached


def _moved_paths() -> set[str]:
    """The module tensors setup repoints, by ``layers.<l>.<path>``."""
    return {
        f"layers.{layer}.{binding.qualified_name}"
        for (layer, _), segment in collect(ToyModel()).segments.items()
        for anchor in segment.anchors
        for binding in anchor.bindings
    }


def test_a_leak_is_reported_and_committed_without_strict_mode(make_group: Callable) -> None:
    """Setup makes six collectives; a leaked original is a warning and a count, not an error."""
    group = make_group([RankSpec(), RankSpec()])
    leaker = group.ranks[1]
    # A holder the bindings do not know about keeps one original alive.
    hidden = leaker.decoder.layers[1].linear_attn.o_proj.weight
    group.run()
    reports = group.assert_collectives(6)
    assert [report.data.leaks for report in reports] == [0, 1]
    (warning,) = [m for m in group.messages(leaker, "warning") if "still alive after commit" in m]
    assert "storage of layers.1.linear_attn._o_dense " in warning
    for rank, leaks in zip(group.ranks, (0, 1)):
        assert isinstance(rank.result, DenseDWDP), rank.trace
        (line,) = [m for m in group.messages(rank, "info") if "committed:" in m]
        assert f" leaks={leaks} " in line
    expected = leaker.snapshot.values["layers.1.linear_attn.o_proj.weight"]
    assert torch.equal(raw_bytes(hidden), expected)


def test_a_leak_in_strict_mode_raises_on_every_rank_after_the_commit(make_group: Callable) -> None:
    """Collective #6 gives every rank the leak, so every rank raises, not the leaking one alone.

    Past the commit point the originals are gone: every rank detaches its bindings (zero-size
    tensors), releases its store and launcher once and flags its model.
    """
    group = make_group([RankSpec(strict=True)] * 3)
    leaker = group.ranks[1]
    hidden = leaker.decoder.layers[1].linear_attn.o_proj.weight
    group.run()
    group.assert_collectives(6)
    moved = _moved_paths()
    for rank in group.ranks:
        assert rank.result is None
        error_type, message = rank.error
        assert error_type is DenseRepointError, rank.trace
        assert "still alive after commit on rank 1, which strict mode forbids" in message
        assert "layers.1.linear_attn._o_dense" in message
        tensors = module_tensors(rank.decoder)
        assert all(tensors[path].numel() == 0 for path in moved)
        assert rank.model._dense_dwdp_active is True
    group.assert_released()
    expected = leaker.snapshot.values["layers.1.linear_attn.o_proj.weight"]
    assert torch.equal(raw_bytes(hidden), expected)


def test_a_failed_commit_raises_on_every_rank(make_group: Callable) -> None:
    """A failure in P6 on one rank, past the commit point, makes every rank raise after #6."""
    specs = [RankSpec()] * 3
    specs[2] = replace(specs[2], fail="commit")
    group = make_group(specs).run()
    reports = group.assert_collectives(6)
    assert [report.ok for report in reports] == [True, True, False]
    moved = _moved_paths()
    for rank in group.ranks:
        assert rank.result is None
        error_type, message = rank.error
        assert error_type is DenseSetupError, rank.trace
        assert re.search(
            r"abandoned at collective #6 \(P6 commit\): rank 2: RuntimeError: injected: P6 "
            r"memory reading failed; past the commit point the original dense weights are "
            r"dropped",
            message,
        ), message
        tensors = module_tensors(rank.decoder)
        assert all(tensors[path].numel() == 0 for path in moved)
        assert rank.model._dense_dwdp_active is True
    group.assert_released()


#: ``(change on one rank, what every rank raises)``.
DISAGREEMENTS = {
    "layouts": (
        {"kinds": ("dense",) + ("kda",) * 5},
        r"abandoned at collective #1 .*the dense layouts differ across ranks \(digest .* on rank 2\)",
    ),
    "weights": (
        {"perturb": "layers.4.block_sparse_moe.gate.weight"},
        r"#1 .*the dense weights differ across ranks: layers\.4 moe: ranks \[0, 1, 3\] vs rank 2$",
    ),
    "enable_dense": (
        {"requested": False},
        r"dense DWDP is enabled on ranks \[0, 1, 3\] and off on rank 2 of the DWDP group",
    ),
}


@pytest.mark.parametrize("change,pattern", DISAGREEMENTS.values(), ids=DISAGREEMENTS)
def test_ranks_that_disagree_stop_at_the_first_collective(
    make_group: Callable, change: dict, pattern: str
) -> None:
    specs = [RankSpec()] * 4
    specs[2] = replace(specs[2], **change)
    group = make_group(specs).run()
    group.assert_collectives(1)
    group.assert_abandoned(pattern)
    assert all(rank.stores == [] for rank in group.ranks)


@pytest.mark.parametrize(
    "phase,num_ranks,failing,fault,report",
    [
        (1, 4, 3, {"free_bytes": 1 << 20}, r"DenseSetupError: 0\.00 GB of device memory"),
        (2, 2, 0, {"fail": "allocate"}, r"RuntimeError: injected: store allocate failed"),
        (3, 4, 1, {"fail": "import"}, r"RuntimeError: injected: store import failed"),
        (4, 2, 1, {"corrupt": 4}, r"Phase V: 1 gathered anchors differ from their"),
        (5, 4, 2, {"corrupt": 5}, r"P5 self-check: 1 module tensors are not their slot"),
    ],
    ids=["p1-memory", "p2-allocate", "p3-import", "p4-phase-v", "p5-self-check"],
)
def test_a_failure_in_any_phase_abandons_everywhere(
    make_group: Callable, phase: int, num_ranks: int, failing: int, fault: dict, report: str
) -> None:
    """Every rank raises after the same collective, with its original tensors back."""
    specs = [RankSpec()] * num_ranks
    specs[failing] = replace(specs[failing], **fault)
    group = make_group(specs).run()

    reports = group.assert_collectives(phase)
    assert [report.ok for report in reports] == [rank != failing for rank in range(num_ranks)]
    if phase >= 4:
        # The failing rank names exactly the corrupted tensor; the others found nothing.
        assert [report.data for report in reports] == [
            [LAST_MOE_ANCHOR] if rank == failing else [] for rank in range(num_ranks)
        ]
    group.assert_abandoned(rf"abandoned at collective #{phase} .*: rank {failing}: {report}")
    group.assert_released()
    assert [len(rank.stores) for rank in group.ranks] == [int(phase >= 2)] * num_ranks
    expected_launchers = [
        int(phase >= 4 or (phase == 3 and r != failing)) for r in range(num_ranks)
    ]
    assert [len(rank.launchers) for rank in group.ranks] == expected_launchers
