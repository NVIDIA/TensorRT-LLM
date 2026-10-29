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
"""Tests for the dense DWDP row store.

The layouts come from toy decoder layers through ``layout.collect_dense_layout``: each layer's
attention and MoE-dense roots hold parameters of the given dtypes and shapes, filled with seeded
random bytes.  The odd layers cover a replicated layer, a gap in the layer indices, a segment
with nothing to move (``B = 0``, once with a tensor below the threshold and once with no tensor),
blobs smaller than one page and smaller than ``15 * 4096`` bytes, and blob sizes that are not
multiples of 16.  The big layers have rows of about 1.25 MiB (``N = 4``), so
``STORE_CHUNK_MB`` splits them.

The CPU tests (``cpu_only``) check that:

* the loopback store puts every rank's row of every ``(layer, segment)`` at
  ``chunk_bases[p][chunk] + chunk_offset`` (``Q_p + off`` with one chunk), zeros elsewhere, and
  that applying the copy plans with a host ``memmove`` reassembles every blob exactly as
  ``layout.write_blob`` writes it;
* ``STORE_CHUNK_MB > 0`` puts whole layers in each chunk, every handle is a multiple of 2 MiB and
  the rows are the same as with one chunk;
* every base is ``ROW_ALIGN``-aligned and no two chunks overlap, also for a chunk without rows
  (trailing empty layers), which still gets an allocation of its own, and also when the
  allocator returns buffers off a ``ROW_ALIGN`` boundary;
* each step runs once and in order: ``chunk_bases`` raises before ``import_peers``, payloads are
  checked (refused payloads use up ``import_peers``), and ``release`` is idempotent and ends the
  store;
* the constructors reject a rank outside ``[0, N)`` and rows that do not fit their chunk at a
  ``ROW_ALIGN``-aligned offset, and ``write_local_rows`` rejects an anchor source that was
  dropped or does not match its anchor;
* the FABRIC store, over a recording fake of the CUDA VMM calls, maps its chunks read-write and
  zero-filled, writes this rank's rows, exports one handle per chunk, imports and maps every
  peer's handles read-only in rank order, and releases peer mappings, imported handles, local
  mappings and local handles in that order -- also after a failure at any step of
  ``allocate`` or ``import_peers``, which then does not run again.  Every handle is whole granules
  of the device's granularity, at least one (a chunk without rows gets one granule) and more than
  ``handle_bytes`` when the granularity exceeds 2 MiB, which ``allocate`` logs; every reservation
  and mapping covers one whole handle; mappings off a ``ROW_ALIGN`` boundary fail
  ``import_peers``; a local view that copies instead of aliasing its mapping fails ``allocate``;
  ``release`` raises the first of several errors and logs the others;
* ``__del__`` releases a store dropped without ``release``, never raises, and leaves a store
  whose constructor raised alone.

The GPU tests run the loopback store of a 16-rank group with ``DenseCopyLauncher`` on one GPU, in
each transfer mode, and compare every slot with ``write_blob``; and check that the loopback store
synchronizes its device before it drops its buffers.  ``test_dense_store_multiproc.py`` covers
real FABRIC handles across processes.
"""

import ctypes
import gc
import math
import re
import sys
from collections import Counter
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.modules.dwdp import fabric_gather, transport, vmm
from tensorrt_llm._torch.modules.dwdp.dense import store as dense_store
from tensorrt_llm._torch.modules.dwdp.dense.config import DenseConfig
from tensorrt_llm._torch.modules.dwdp.dense.layout import collect_dense_layout, write_blob
from tensorrt_llm._torch.modules.dwdp.dense.plan import DenseCopyLauncher, build_copy_plans
from tensorrt_llm._torch.modules.dwdp.dense.store import (
    DenseShardStore,
    FabricShardStore,
    LoopbackShardStore,
)
from tensorrt_llm._torch.modules.dwdp.dense.types import (
    ROW_ALIGN,
    STORE_ALIGN,
    CopyMode,
    DenseLayout,
    DenseSetupError,
    SegmentKind,
    SlotBuffer,
    allocate_slot_ring,
    build_dense_layout,
)

ATTN = SegmentKind.ATTN
MOE = SegmentKind.MOE
CPU = torch.device("cpu")
BF16 = torch.bfloat16
F16 = torch.float16
F32 = torch.float32
FP8 = torch.float8_e4m3fn
I32 = torch.int32
U8 = torch.uint8

FABRIC = vmm.cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
POSIX_FD = vmm.cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
PROT_READ = vmm.cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READ
PROT_READWRITE = vmm.cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE

#: Storages below this stay replicated in the toy layouts.
TOY_MIN_BYTES = 64

#: Decoder layer -> (attention tensors, MoE-dense tensors) as ``(dtype, shape)``; None: the layer
#: stays replicated.  Blob sizes: layer 1 A 614457 (two anchors), D 308000; layer 2 A 777, D 0
#: (a 5-byte tensor stays); layer 3 A 61438, D 132000 (a 21-byte tensor stays); layer 5 A 81920,
#: D 0 (no tensor).
ODD_LAYERS = {
    0: None,
    1: ([(BF16, (300, 1000)), (U8, (12345,))], [(F32, (1000, 77))]),
    2: ([(U8, (777,))], [(U8, (5,))]),
    3: ([(F16, (30719,))], [(I32, (1000, 33)), (FP8, (3, 7))]),
    5: ([(FP8, (2048, 40))], []),
}
ODD_BLOB_BYTES = {
    (1, ATTN): 614457,
    (1, MOE): 308000,
    (2, ATTN): 777,
    (2, MOE): 0,
    (3, ATTN): 61438,
    (3, MOE): 132000,
    (5, ATTN): 81920,
    (5, MOE): 0,
}

#: Four layers whose rows take about 1.25 MiB each with N = 4 (A 3 MiB + 5 * l bytes, D 2 MiB).
BIG_LAYERS = {
    layer: ([(U8, ((3 << 20) + 5 * layer,))], [(BF16, (1 << 20,))]) for layer in range(1, 5)
}
#: ``TRTLLM_DWDP_DENSE_STORE_CHUNK_MB`` -> the layers of each chunk of the big layout.
BIG_CHUNKS = {1: [(1,), (2,), (3,), (4,)], 3: [(1, 2), (3, 4)]}

POISON = 0x5A  # Slot bytes before a copy.
GARBAGE = 0xEE  # Fake device memory before the store zero-fills it.


# ---------------------------------------------------------------------------------------------
# Toy layers and helpers
# ---------------------------------------------------------------------------------------------


class ToyRoot(nn.Module):
    """A dense root holding one Parameter per ``(dtype, shape)``, filled with random bytes."""

    def __init__(self, tensors, generator: torch.Generator, device: torch.device) -> None:
        super().__init__()
        for index, (dtype, shape) in enumerate(tensors):
            nbytes = math.prod(shape) * dtype.itemsize
            data = torch.randint(0, 256, (nbytes,), generator=generator, dtype=torch.uint8)
            weight = data.view(dtype).reshape(shape).to(device)
            self.register_parameter(f"w{index}", nn.Parameter(weight, requires_grad=False))


class ToyLayer(nn.Module):
    """A decoder layer with the ``dense_dwdp_roots`` seam; ``spec`` None keeps it replicated."""

    def __init__(self, spec, generator: torch.Generator, device: torch.device) -> None:
        super().__init__()
        self.covered = spec is not None
        attn, moe = spec if spec is not None else ([], [])
        self.attn = ToyRoot(attn, generator, device)
        self.moe = ToyRoot(moe, generator, device)

    def dense_dwdp_roots(self, segment: str) -> list[tuple[str, nn.Module]]:
        return [(segment, getattr(self, segment))] if self.covered else []


def collect(
    spec, num_ranks: int, device: torch.device = CPU, *, segments=("attn", "moe"), chunk_mb=0
) -> tuple[DenseLayout, list[ToyLayer]]:
    """The layout of toy layers built from ``spec`` (seeded, so every call gives equal bytes)."""
    generator = torch.Generator().manual_seed(1234)
    layers = [(index, ToyLayer(spec[index], generator, device)) for index in sorted(spec)]
    config = DenseConfig(
        enabled=True, segments=segments, min_bytes=TOY_MIN_BYTES, store_chunk_mb=chunk_mb
    )
    layout = collect_dense_layout(
        layers,
        config,
        num_ranks=num_ranks,
        device=device,
        attr_allow=frozenset(),
        exclude=frozenset(),
    )
    return layout, [layer for _, layer in layers]


def blob(segment, num_ranks: int, device: torch.device = CPU) -> torch.Tensor:
    """The full ``N * S`` blob of a segment, as a gather leaves it in a slot."""
    out = torch.empty(num_ranks * segment.row_bytes, dtype=torch.uint8, device=device)
    write_blob(segment, out, num_ranks=num_ranks)
    return out


def host_bytes(address: int, nbytes: int) -> torch.Tensor:
    """A copy of ``nbytes`` of host memory at ``address``."""
    if nbytes == 0:
        return torch.empty(0, dtype=torch.uint8)
    view = torch.frombuffer((ctypes.c_ubyte * nbytes).from_address(address), dtype=torch.uint8)
    return view.clone()


def host_memmove(plan) -> None:
    """Run a copy plan on host memory: ``memmove`` every descriptor."""
    for descriptor in plan.descriptors:
        ctypes.memmove(descriptor.dst, descriptor.src, descriptor.nbytes)


def loopback(layout: DenseLayout, rank: int = 0, device: torch.device = CPU) -> LoopbackShardStore:
    """A loopback store through the setup call order: allocate, write, export, import."""
    store = LoopbackShardStore(layout, rank, device)
    store.allocate()
    store.write_local_rows()
    payload = store.export()
    store.import_peers([payload] * layout.num_ranks)
    return store


def slot_rings(layout: DenseLayout, device: torch.device = CPU) -> dict:
    return {kind: allocate_slot_ring(spec, device) for kind, spec in layout.slot_specs.items()}


def slot_bases(rings) -> dict:
    return {kind: [slot.base for slot in ring] for kind, ring in rings.items()}


def holding_slot(layout: DenseLayout, rings, kind: SegmentKind, layer: int) -> SlotBuffer:
    return rings[kind][layout.slot_index(kind, layer)]


def poison(layout: DenseLayout, rings, layer: int) -> None:
    for kind in layout.segment_kinds:
        holding_slot(layout, rings, kind, layer).storage.fill_(POISON)


def slot_mismatches(layout: DenseLayout, rings, layer: int, expected, label: str) -> list[str]:
    """Why the slots of ``layer`` do not hold its blobs from their bases, poison elsewhere."""
    failures = []
    for segment in layout.layer_segments(layer):
        slot = holding_slot(layout, rings, segment.segment, layer)
        nbytes = layout.num_ranks * segment.row_bytes
        got = slot.blob_view(nbytes)
        want = expected[segment.key]
        name = f"{label}/layer {layer}/{segment.segment.value}"
        if not torch.equal(got, want):
            wrong = torch.nonzero(got != want).flatten()
            row, offset = divmod(int(wrong[0]), segment.row_bytes)
            failures.append(f"{name}: {wrong.numel()} wrong bytes, first in row {row} at {offset}")
        outside = (slot.storage[: slot.pad], slot.storage[slot.pad + nbytes :])
        if not all(bool((part == POISON).all()) for part in outside):
            failures.append(f"{name}: slot bytes outside the blob were written")
    return failures


def chunk_spans(layout: DenseLayout, bases) -> list[tuple[int, int, str]]:
    return [
        (base, base + chunk.handle_bytes, f"rank {rank} chunk {chunk.index}")
        for rank, row in enumerate(bases)
        for chunk, base in zip(layout.chunks, row)
    ]


class RecordingLogger:
    """Stands in for the store module's logger and keeps every ``(level, message)``."""

    def __init__(self) -> None:
        self.records: list[tuple[str, str]] = []

    def debug(self, message: str) -> None:
        self.records.append(("debug", message))

    def info(self, message: str) -> None:
        self.records.append(("info", message))

    def warning(self, message: str) -> None:
        self.records.append(("warning", message))

    def error(self, message: str) -> None:
        self.records.append(("error", message))

    def messages(self, level: str) -> list[str]:
        return [message for name, message in self.records if name == level]


@pytest.fixture
def store_log(monkeypatch) -> RecordingLogger:
    """What the store logs during the test."""
    log = RecordingLogger()
    monkeypatch.setattr(dense_store, "logger", log)
    return log


@pytest.fixture
def unraisable(monkeypatch) -> list[BaseException]:
    """Exceptions that escape a finalizer (``sys.unraisablehook``) during the test."""
    gc.collect()  # Garbage of earlier tests is finalized before the hook is installed.
    escaped: list[BaseException] = []
    monkeypatch.setattr(sys, "unraisablehook", lambda args: escaped.append(args.exc_value))
    return escaped


# ---------------------------------------------------------------------------------------------
# CPU: the toy layouts
# ---------------------------------------------------------------------------------------------


@pytest.mark.cpu_only
def test_toy_layouts_cover_the_edge_cases():
    """The odd layout has the blob sizes the tests rely on; the big layout splits into chunks."""
    layout, _ = collect(ODD_LAYERS, 16)
    assert layout.layers == (1, 2, 3, 5)
    assert {key: segment.blob_bytes for key, segment in layout.segments.items()} == ODD_BLOB_BYTES
    sizes = ODD_BLOB_BYTES.values()
    assert 0 in sizes
    assert any(0 < size < 15 * ROW_ALIGN for size in sizes)
    assert any(size % 16 for size in sizes)
    assert len(layout.segment(1, ATTN).anchors) == 2
    assert len(layout.chunks) == 1

    for chunk_mb, groups in BIG_CHUNKS.items():
        big, _ = collect(BIG_LAYERS, 4, chunk_mb=chunk_mb)
        assert [chunk.layers for chunk in big.chunks] == groups


# ---------------------------------------------------------------------------------------------
# CPU: loopback rows and addressing
# ---------------------------------------------------------------------------------------------


@pytest.mark.cpu_only
@pytest.mark.parametrize("segments", [("attn", "moe"), ("attn",)], ids=["attn+moe", "attn"])
@pytest.mark.parametrize("num_ranks", [4, 16])
def test_loopback_rows_land_at_their_store_address(num_ranks, segments):
    """Rank ``p``'s row of ``(l, seg)`` is at ``Q_p + off(l, seg)``; every other byte is zero."""
    layout, _ = collect(ODD_LAYERS, num_ranks, segments=segments)
    store = loopback(layout, rank=num_ranks - 1)
    bases = store.chunk_bases()
    assert len(bases) == num_ranks
    assert all(len(row) == len(layout.chunks) == 1 for row in bases)
    for segment in layout.segments.values():
        assert segment.chunk_offset == segment.store_offset
        full = blob(segment, num_ranks)
        width = segment.row_bytes
        for rank in range(num_ranks):
            got = host_bytes(bases[rank][0] + segment.store_offset, width)
            assert torch.equal(got, full[rank * width : (rank + 1) * width]), (
                f"rank {rank}'s row of {segment.key}"
            )
    chunk = layout.chunks[0]
    for rank in range(num_ranks):
        tail = host_bytes(bases[rank][0] + chunk.nbytes, chunk.handle_bytes - chunk.nbytes)
        assert not tail.any(), f"rank {rank}'s chunk has bytes past its rows"
    store.release()


@pytest.mark.cpu_only
@pytest.mark.parametrize("chunk_mb", [0, 1], ids=["one-chunk", "chunked"])
@pytest.mark.parametrize("segments", [("attn", "moe"), ("attn",)], ids=["attn+moe", "attn"])
@pytest.mark.parametrize("num_ranks,rank", [(4, 0), (4, 3), (16, 0), (16, 9), (16, 15)])
def test_loopback_plans_reassemble_every_blob_on_host(num_ranks, rank, segments, chunk_mb):
    """The copy plans over the loopback bases, run with ``memmove``, rebuild every blob."""
    layout, _ = collect(ODD_LAYERS, num_ranks, segments=segments, chunk_mb=chunk_mb)
    store = loopback(layout, rank=rank)
    rings = slot_rings(layout)
    plans = build_copy_plans(
        layout, rank=rank, chunk_bases=store.chunk_bases(), slot_bases=slot_bases(rings)
    )
    expected = {key: blob(segment, num_ranks) for key, segment in layout.segments.items()}
    failures = []
    for layer in layout.layers:
        poison(layout, rings, layer)
        host_memmove(plans[layer])
        failures += slot_mismatches(layout, rings, layer, expected, f"rank {rank}")
    assert not failures, "\n".join(failures)
    store.release()


@pytest.mark.cpu_only
@pytest.mark.parametrize("chunk_mb", sorted(BIG_CHUNKS))
def test_store_chunks_hold_whole_layers_and_the_same_rows(chunk_mb):
    """With ``STORE_CHUNK_MB`` each chunk holds whole layers in whole 2 MiB handles; rows match."""
    num_ranks = 4
    single, _ = collect(BIG_LAYERS, num_ranks)
    chunked, _ = collect(BIG_LAYERS, num_ranks, chunk_mb=chunk_mb)
    assert len(single.chunks) == 1
    assert [chunk.layers for chunk in chunked.chunks] == BIG_CHUNKS[chunk_mb]
    for chunk in chunked.chunks:
        assert chunk.handle_bytes % STORE_ALIGN == 0
        assert chunk.nbytes <= chunk.handle_bytes < chunk.nbytes + STORE_ALIGN
        assert chunk.nbytes <= chunk_mb << 20 or len(chunk.layers) == 1
        for layer in chunk.layers:
            assert {segment.chunk for segment in chunked.layer_segments(layer)} == {chunk.index}

    one = loopback(single)
    many = loopback(chunked)
    for key, segment in chunked.segments.items():
        reference = single.segments[key]
        assert segment.row_bytes == reference.row_bytes
        full = blob(segment, num_ranks)
        width = segment.row_bytes
        for rank in range(num_ranks):
            row = full[rank * width : (rank + 1) * width]
            at = many.chunk_bases()[rank][segment.chunk] + segment.chunk_offset
            assert torch.equal(host_bytes(at, width), row), f"chunked row {rank} of {key}"
            at = one.chunk_bases()[rank][0] + reference.store_offset
            assert torch.equal(host_bytes(at, width), row), f"single-chunk row {rank} of {key}"

    rings = slot_rings(chunked)
    plans = build_copy_plans(
        chunked, rank=2, chunk_bases=many.chunk_bases(), slot_bases=slot_bases(rings)
    )
    expected = {key: blob(segment, num_ranks) for key, segment in chunked.segments.items()}
    failures = []
    for layer in chunked.layers:
        poison(chunked, rings, layer)
        host_memmove(plans[layer])
        failures += slot_mismatches(chunked, rings, layer, expected, "chunked rank 2")
    assert not failures, "\n".join(failures)
    one.release()
    many.release()


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "spec,num_ranks,chunk_mb", [(ODD_LAYERS, 16, 0), (BIG_LAYERS, 4, 1), (BIG_LAYERS, 4, 3)]
)
def test_every_base_is_row_aligned_and_chunks_never_overlap(spec, num_ranks, chunk_mb):
    """Every rank's chunks are distinct ``ROW_ALIGN``-aligned allocations of ``handle_bytes``."""
    layout, _ = collect(spec, num_ranks, chunk_mb=chunk_mb)
    store = loopback(layout, rank=1)
    bases = store.chunk_bases()
    assert [len(row) for row in bases] == [len(layout.chunks)] * num_ranks
    assert all(base % ROW_ALIGN == 0 and base > 0 for row in bases for base in row)
    spans = sorted(chunk_spans(layout, bases))
    overlaps = [(a[2], b[2]) for a, b in zip(spans, spans[1:]) if b[0] < a[1]]
    assert not overlaps
    store.release()


@pytest.mark.cpu_only
def test_loopback_aligns_buffers_the_allocator_does_not(monkeypatch):
    """Buffers that start 64 bytes past a ``ROW_ALIGN`` boundary still give aligned bases."""
    num_ranks = 4
    layout, _ = collect(BIG_LAYERS, num_ranks, chunk_mb=3)
    zeros = torch.zeros
    skews = []

    def misaligned_zeros(nbytes, **kwargs):
        storage = zeros(nbytes + 2 * ROW_ALIGN, **kwargs)
        start = (64 - storage.data_ptr()) % ROW_ALIGN
        buffer = storage[start : start + nbytes]
        skews.append(buffer.data_ptr() % ROW_ALIGN)
        return buffer

    store = LoopbackShardStore(layout, 1, CPU)
    with monkeypatch.context() as patch:
        patch.setattr(torch, "zeros", misaligned_zeros)
        store.allocate()
    assert skews == [64] * (num_ranks * len(layout.chunks))
    store.write_local_rows()
    store.import_peers([None] * num_ranks)
    bases = store.chunk_bases()
    assert all(base % ROW_ALIGN == 0 for row in bases for base in row)
    for key, segment in layout.segments.items():
        full = blob(segment, num_ranks)
        width = segment.row_bytes
        for rank in range(num_ranks):
            got = host_bytes(bases[rank][segment.chunk] + segment.chunk_offset, width)
            assert torch.equal(got, full[rank * width : (rank + 1) * width]), f"{rank}, {key}"
    store.release()


# ---------------------------------------------------------------------------------------------
# CPU: call order, payloads, release
# ---------------------------------------------------------------------------------------------


@pytest.mark.cpu_only
def test_chunk_bases_raises_before_import_peers():
    layout, _ = collect(ODD_LAYERS, 4)
    store = LoopbackShardStore(layout, 0, CPU)
    with pytest.raises(RuntimeError, match="before import_peers"):
        store.chunk_bases()
    store.allocate()
    with pytest.raises(RuntimeError, match="before import_peers"):
        store.chunk_bases()
    store.write_local_rows()
    assert store.export() is None
    with pytest.raises(RuntimeError, match="before import_peers"):
        store.chunk_bases()
    store.import_peers([None] * 4)
    assert len(store.chunk_bases()) == 4
    store.release()

    fabric = FabricShardStore(layout, 0, torch.device("cuda", 0))
    with pytest.raises(RuntimeError, match="before import_peers"):
        fabric.chunk_bases()
    fabric.release()


@pytest.mark.cpu_only
def test_steps_run_once_and_in_order():
    layout, _ = collect(ODD_LAYERS, 4)
    store = LoopbackShardStore(layout, 0, CPU)
    for step in (store.write_local_rows, store.export):
        with pytest.raises(RuntimeError, match="before a successful allocate"):
            step()
    with pytest.raises(RuntimeError, match="before a successful allocate"):
        store.import_peers([None] * 4)
    store.allocate()
    with pytest.raises(RuntimeError, match="allocate already ran"):
        store.allocate()
    store.write_local_rows()
    store.write_local_rows()  # The same bytes again.
    store.import_peers([store.export()] * 4)
    with pytest.raises(RuntimeError, match="import_peers already ran"):
        store.import_peers([None] * 4)
    store.release()


@pytest.mark.cpu_only
def test_release_is_idempotent_and_ends_the_store(monkeypatch):
    syncs = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: syncs.append(device))
    layout, _ = collect(ODD_LAYERS, 4)
    fresh = LoopbackShardStore(layout, 0, CPU)
    fresh.release()
    fresh.release()

    store = loopback(layout)
    store.release()
    store.release()
    assert syncs == []  # A store on the CPU has no device to synchronize.
    for step in (store.allocate, store.write_local_rows, store.export, store.chunk_bases):
        with pytest.raises(RuntimeError, match="after release"):
            step()
    with pytest.raises(RuntimeError, match="after release"):
        store.import_peers([None] * 4)

    # A FABRIC store that never allocated releases without any CUDA call.
    fabric = FabricShardStore(layout, 3, torch.device("cuda", 0))
    fabric.release()
    fabric.release()
    with pytest.raises(RuntimeError, match="after release"):
        fabric.allocate()


@pytest.mark.cpu_only
def test_loopback_import_checks_the_payloads():
    """Payloads it refuses use up ``import_peers``: a second call raises."""
    layout, _ = collect(ODD_LAYERS, 4)
    store = LoopbackShardStore(layout, 0, CPU)
    store.allocate()
    with pytest.raises(ValueError, match="one payload per rank"):
        store.import_peers([None] * 3)
    other = LoopbackShardStore(layout, 0, CPU)
    other.allocate()
    with pytest.raises(ValueError, match=r"ranks \[2\] sent store handles"):
        other.import_peers([None, None, [b"handle"], None])
    for refused in (store, other):
        with pytest.raises(RuntimeError, match="import_peers already ran"):
            refused.import_peers([None] * 4)
        with pytest.raises(RuntimeError, match="before import_peers"):
            refused.chunk_bases()
    store.release()
    other.release()


@pytest.mark.cpu_only
def test_write_local_rows_needs_the_anchor_sources():
    layout, _ = collect(ODD_LAYERS, 4)
    store = LoopbackShardStore(layout, 0, CPU)
    store.allocate()
    layout.segment(3, ATTN).anchors[0].drop_originals()
    with pytest.raises(DenseSetupError, match="was dropped"):
        store.write_local_rows()
    store.release()


#: Anchor sources that do not match their anchor: a byte short, another dtype, not flat.
SOURCE_DEFECTS = {
    "short": lambda source: source[:-1],
    "int8": lambda source: source.view(torch.int8),
    "2d": lambda source: source.view(1, -1),
}


@pytest.mark.cpu_only
@pytest.mark.parametrize("defect", sorted(SOURCE_DEFECTS))
def test_write_local_rows_rejects_a_source_that_does_not_match_its_anchor(monkeypatch, defect):
    """Both stores check every anchor's source: a flat ``uint8`` tensor of the anchor's size."""
    layout, _ = collect(ODD_LAYERS, 4)
    anchor = layout.segment(1, ATTN).anchors[1]
    anchor.source = SOURCE_DEFECTS[defect](anchor.source)
    FakeVmm(monkeypatch)
    for store in (
        LoopbackShardStore(layout, 0, CPU),
        FabricShardStore(layout, 0, torch.device("cuda", 0)),
    ):
        store.allocate()
        with pytest.raises(
            DenseSetupError,
            match=rf"source of {re.escape(anchor.path)} is .*; expected a flat uint8 tensor of "
            rf"{anchor.nbytes} bytes",
        ):
            store.write_local_rows()
        store.release()


@pytest.mark.cpu_only
@pytest.mark.parametrize("rank", [-1, 4])
def test_constructors_check_their_arguments(rank):
    layout, _ = collect(ODD_LAYERS, 4)
    for store_type in (LoopbackShardStore, FabricShardStore):
        with pytest.raises(ValueError, match=r"rank must be in \[0, 4\)"):
            store_type(layout, rank, torch.device("cuda", 0))
    with pytest.raises(ValueError, match="needs a CUDA device"):
        FabricShardStore(layout, 0, CPU)
    assert issubclass(FabricShardStore, DenseShardStore)
    assert issubclass(LoopbackShardStore, DenseShardStore)


#: Placement defects, each breaking exactly one rule: ``defect -> (layer, field, new value)`` for
#: the layer's attention rows, given the field's value.  Layer 1's rows have room after them in
#: the single chunk; layer 5's end exactly at its end.
ROW_DEFECTS = {
    "unaligned offset": (1, "chunk_offset", lambda offset: offset + 64),
    "negative offset": (1, "chunk_offset", lambda offset: -ROW_ALIGN),
    "past the chunk end": (5, "chunk_offset", lambda offset: offset + ROW_ALIGN),
    "chunk out of range": (1, "chunk", lambda chunk: chunk + 1),
    "negative chunk": (1, "chunk", lambda chunk: -1),
}


@pytest.mark.cpu_only
@pytest.mark.parametrize("defect", list(ROW_DEFECTS))
def test_constructors_reject_rows_outside_their_chunk(defect):
    """Rows must lie inside an existing chunk at a ``ROW_ALIGN``-aligned offset."""
    layout, _ = collect(ODD_LAYERS, 4)
    assert len(layout.chunks) == 1
    last = layout.segment(5, ATTN)
    assert last.row_bytes > 0 and last.chunk_offset + last.row_bytes == layout.chunks[0].nbytes
    layer, name, change = ROW_DEFECTS[defect]
    segment = layout.segment(layer, ATTN)
    setattr(segment, name, change(getattr(segment, name)))
    for store_type in (LoopbackShardStore, FabricShardStore):
        with pytest.raises(ValueError, match=rf"rows of \({layer}, attn\) do not fit store chunk"):
            store_type(layout, 0, torch.device("cuda", 0))


# ---------------------------------------------------------------------------------------------
# CPU: the FABRIC store over a recording fake of the CUDA VMM calls
# ---------------------------------------------------------------------------------------------


class FakeVmm:
    """The VMM, export/import and synchronize calls of the FABRIC store, over host memory.

    Every call is recorded as ``(op, value)``; ``cuMemSetAccess`` is faked at the driver binding,
    so the access flags the store asks for are what the ``access_rw`` / ``access_ro`` records
    say.  Reserved address ranges are real host buffers filled with garbage (as fresh device
    memory may be), so the store's views alias them and its writes land there; with
    ``aliasing_views=False`` a view is a copy instead.

    Like the driver, the fake knows the size of every handle: created, or imported from a
    payload whose size ``export`` (or ``peer_payloads``, for a peer's) registered.  Handles and
    reservations are whole granules of ``granularity``, at least one, and a reservation must be
    aligned to it; a mapping covers one whole handle (a FABRIC handle cannot be mapped in part)
    and exactly its reservation; access, unmap and free name a whole reservation.  Reservations
    start ``reserve_skew`` bytes past a ``ROW_ALIGN`` boundary.  ``fail=(op, n)`` makes the
    ``n``-th call of ``op`` raise before it takes effect, ``fail_every`` every call of the ops it
    names.
    """

    GRANULARITY = 2 << 20

    def __init__(
        self,
        monkeypatch,
        *,
        fail=None,
        fail_every=frozenset(),
        handle_type=FABRIC,
        aliasing_views=True,
        granularity=GRANULARITY,
        reserve_skew=0,
    ) -> None:
        self.calls: list[tuple[str, object]] = []
        self.fail = fail
        self.fail_every = fail_every
        self.aliasing_views = aliasing_views
        self.granularity = granularity
        self.reserve_skew = reserve_skew
        self.counts: Counter = Counter()
        self.memory: dict[int, torch.Tensor] = {}
        self.owner: dict[int, int] = {}  # va -> the handle mapped (or about to be) there
        self.imported: set[int] = set()
        self.handle_sizes: dict[int, int] = {}
        self.payload_sizes: dict[bytes, int] = {}
        self.va_sizes: dict[int, int] = {}
        self._next_handle = 1000
        self._last_handle = None
        patches = {
            (vmm, "peer_handle_type"): lambda: handle_type,
            (vmm, "get_allocation_granularity"): lambda device_id: self.granularity,
            (vmm, "create_fabric_handle"): self.create,
            (vmm, "reserve_va"): self.reserve,
            (vmm, "map_handle"): self.map,
            (vmm.cuda, "cuMemSetAccess"): self.set_access,
            (vmm, "tensor_from_ptr"): self.view,
            (vmm, "unmap_va"): self.unmap,
            (vmm, "free_va"): self.free,
            (vmm, "release_handle"): self.release,
            (transport, "_export_handle"): self.export,
            (transport, "_import_handle"): self.import_handle,
            (torch.cuda, "synchronize"): self.synchronize,
            (torch.cuda, "current_stream"): self.current_stream,
        }
        for (owner, name), value in patches.items():
            monkeypatch.setattr(owner, name, value)

    def _record(self, op: str, value: object) -> None:
        self.counts[op] += 1
        if self.fail == (op, self.counts[op]) or op in self.fail_every:
            raise RuntimeError(f"injected failure of {op} #{self.counts[op]}")
        self.calls.append((op, value))

    def _new_handle(self) -> int:
        self._next_handle += 1
        return self._next_handle

    def _check_whole_granules(self, what: str, size: int) -> None:
        assert size > 0 and size % self.granularity == 0, f"{what} of {size} bytes"

    def _check_reservation(self, what: str, va: int, size: int) -> None:
        assert va in self.va_sizes, f"{what} at {va:#x}, which is not reserved"
        assert size == self.va_sizes[va], (
            f"{what} of {size} bytes at {va:#x}, whose reservation has {self.va_sizes[va]}"
        )

    def create(self, size, device_id):
        self._check_whole_granules("cuMemCreate", size)
        handle = self._new_handle()
        self._record("create", (handle, size))
        self.handle_sizes[handle] = size
        self._last_handle = handle
        return handle

    def import_handle(self, payload, handle_type):
        assert handle_type == FABRIC and isinstance(payload, bytes)
        assert payload in self.payload_sizes, f"import of an unknown payload {payload!r}"
        handle = self._new_handle()
        self._record("import", (handle, payload))
        self.handle_sizes[handle] = self.payload_sizes[payload]
        self.imported.add(handle)
        self._last_handle = handle
        return handle

    def export(self, handle, handle_type):
        assert handle_type == FABRIC
        self._record("export", handle)
        payload = f"fabric-handle-{handle}".encode()
        self.payload_sizes[payload] = self.handle_sizes[handle]
        return payload

    def reserve(self, size, alignment):
        assert alignment == self.granularity
        self._check_whole_granules("cuMemAddressReserve", size)
        self._record("reserve", size)
        buffer = torch.full((size + 2 * ROW_ALIGN,), GARBAGE, dtype=torch.uint8)
        va = buffer.data_ptr() + (-buffer.data_ptr()) % ROW_ALIGN + self.reserve_skew
        self.memory[va] = buffer
        self.va_sizes[va] = size
        self.owner[va] = self._last_handle
        self.calls[-1] = ("reserve", (va, size))
        return va

    def map(self, va, size, handle, offset=0):
        assert self.owner[va] == handle and offset == 0
        self._check_reservation("cuMemMap", va, size)
        assert size == self.handle_sizes[handle], (
            f"cuMemMap of {size} bytes of handle {handle}, which has {self.handle_sizes[handle]}"
        )
        self._record("map", (va, handle))

    def set_access(self, va, size, descriptors, count):
        """``cuMemSetAccess``: records ``access_rw`` or ``access_ro`` by the requested flags."""
        assert count == len(descriptors) == 1 and descriptors[0].location.id == 0
        self._check_reservation("cuMemSetAccess", va, size)
        flags = int(descriptors[0].flags)
        op = {int(PROT_READWRITE): "access_rw", int(PROT_READ): "access_ro"}.get(flags)
        assert op is not None, f"unexpected access flags {flags}"
        self._record(op, va)
        return (vmm.cuda.CUresult.CUDA_SUCCESS,)

    def view(self, ptr, shape, dtype, device_id):
        self._record("view", ptr)
        if not self.aliasing_views:
            return torch.zeros(shape, dtype=dtype)
        nbytes = math.prod(shape) * dtype.itemsize
        return host_bytes_view(ptr, nbytes).view(dtype).reshape(shape)

    def unmap(self, va, size):
        self._check_reservation("cuMemUnmap", va, size)
        self._record("unmap", va)

    def free(self, va, size):
        self._check_reservation("cuMemAddressFree", va, size)
        self._record("free", va)
        del self.memory[va]
        del self.va_sizes[va]

    def release(self, handle):
        self._record("release", handle)

    def synchronize(self, device=None):
        self._record("sync", device)

    def current_stream(self, device=None):
        return SimpleNamespace(synchronize=lambda: self._record("stream_sync", device))

    # Views of what happened ------------------------------------------------------------------

    def ops(self, start: int = 0) -> list[str]:
        return [op for op, _ in self.calls[start:]]

    def values(self, op: str, start: int = 0, stop: int | None = None) -> list:
        return [value for name, value in self.calls[start:stop] if name == op]

    def is_peer(self, value) -> bool:
        """Whether a recorded handle or address belongs to an imported (peer) handle."""
        return self.owner.get(value, value) in self.imported


def host_bytes_view(address: int, nbytes: int) -> torch.Tensor:
    """A ``uint8`` tensor that aliases ``nbytes`` of host memory at ``address``."""
    return torch.frombuffer((ctypes.c_ubyte * nbytes).from_address(address), dtype=torch.uint8)


def peer_payloads(fake: FakeVmm, store: FabricShardStore, layout: DenseLayout, rank: int):
    """Collective #2 as this rank sees it: its own export plus a distinct export per peer.

    Every rank allocates the same chunks, so a peer's handle of a chunk has the size of this
    rank's handle of it; ``fake`` learns that size for each fabricated payload.
    """
    own = store.export()
    payloads = []
    for peer in range(layout.num_ranks):
        if peer == rank:
            payloads.append(own)
            continue
        blobs = [f"peer-{peer}-chunk-{chunk.index}".encode() for chunk in layout.chunks]
        for blob, own_blob in zip(blobs, own):
            fake.payload_sizes[blob] = fake.payload_sizes[own_blob]
        payloads.append(blobs)
    return payloads


def check_release(fake: FakeVmm, start: int) -> None:
    """The calls since ``start`` undo every handle and mapping created before it, once each.

    Order: a device synchronize first; then the peer mappings (unmap, free) and imported
    handles, then the local mappings and handles; a mapping is unmapped before its VA is freed
    and both happen before its handle is released.
    """
    created = [handle for handle, _ in fake.values("create", 0, start)]
    imported = [handle for handle, _ in fake.values("import", 0, start)]
    reserved = [va for va, _ in fake.values("reserve", 0, start)]
    mapped = [va for va, _ in fake.values("map", 0, start)]
    tail = fake.calls[start:]
    if not (created or imported):
        assert tail == []
        return
    assert tail[0][0] == "sync"
    released = fake.values("release", start)
    assert sorted(released) == sorted(created + imported) and len(set(released)) == len(released)
    assert sorted(fake.values("free", start)) == sorted(reserved)
    assert sorted(fake.values("unmap", start)) == sorted(mapped)
    position = {(op, value): index for index, (op, value) in enumerate(tail)}
    for va in reserved:
        handle_released = position[("release", fake.owner[va])]
        assert position[("free", va)] < handle_released
        if va in mapped:
            assert position[("unmap", va)] < position[("free", va)]
    peer_steps = [i for i, (op, value) in enumerate(tail[1:]) if fake.is_peer(value)]
    local_steps = [i for i, (op, value) in enumerate(tail[1:]) if not fake.is_peer(value)]
    if peer_steps and local_steps:
        assert max(peer_steps) < min(local_steps), "peer resources must go before local ones"
    released_peers = [handle for handle in released if handle in fake.imported]
    peer_frees = [position[("free", va)] for va in reserved if fake.is_peer(va)]
    if released_peers and peer_frees:
        first_release = min(position[("release", handle)] for handle in released_peers)
        assert max(peer_frees) < first_release, (
            "peer VA ranges are freed before the imports are released"
        )


@pytest.mark.cpu_only
@pytest.mark.parametrize("chunk_mb", [0, 1], ids=["one-chunk", "chunked"])
def test_fabric_store_maps_writes_exports_and_imports(monkeypatch, store_log, chunk_mb):
    """One handle per chunk mapped read-write and zeroed, this rank's rows, peers read-only."""
    num_ranks, rank = 4, 2
    layout, _ = collect(BIG_LAYERS, num_ranks, chunk_mb=chunk_mb)
    num_chunks = len(layout.chunks)
    fake = FakeVmm(monkeypatch)
    store = FabricShardStore(layout, rank, torch.device("cuda", 0))

    store.allocate()
    per_chunk = ["create", "reserve", "map", "access_rw", "view"]
    assert fake.ops() == per_chunk * num_chunks + ["stream_sync"]
    assert [size for _, size in fake.values("create")] == [c.handle_bytes for c in layout.chunks]
    assert store_log.messages("info") == []  # It allocated exactly ``layout.store_bytes``.
    local_vas = [va for va, _ in fake.values("reserve")]
    for chunk, va in zip(layout.chunks, local_vas):
        assert not host_bytes(va, chunk.handle_bytes).any(), f"chunk {chunk.index} not zeroed"

    store.write_local_rows()
    for chunk, va in zip(layout.chunks, local_vas):
        local = host_bytes(va, chunk.handle_bytes)
        written = torch.zeros_like(local)
        for layer in chunk.layers:
            for segment in layout.layer_segments(layer):
                width = segment.row_bytes
                row = blob(segment, num_ranks)[rank * width : (rank + 1) * width]
                written[segment.chunk_offset : segment.chunk_offset + width] = row
        assert torch.equal(local, written), f"chunk {chunk.index}"

    mark = len(fake.calls)
    payloads = peer_payloads(fake, store, layout, rank)
    assert payloads[rank] == [f"fabric-handle-{h}".encode() for h, _ in fake.values("create")]
    assert store.export() == payloads[rank]  # Exported once, returned again.
    assert fake.ops(mark) == ["export"] * num_chunks

    mark = len(fake.calls)
    store.import_peers(payloads)
    imports = [payload for _, payload in fake.values("import", mark)]
    assert imports == [blob for peer in (0, 1, 3) for blob in payloads[peer]]
    assert fake.ops(mark) == ["import", "reserve", "map", "access_ro"] * (3 * num_chunks)
    peer_sizes = [size for _, size in fake.values("reserve", mark)]
    assert peer_sizes == [chunk.handle_bytes for chunk in layout.chunks] * 3
    peer_vas = iter(va for va, _ in fake.values("reserve", mark))
    expected_bases = tuple(
        tuple(local_vas) if peer == rank else tuple(next(peer_vas) for _ in layout.chunks)
        for peer in range(num_ranks)
    )
    assert store.chunk_bases() == expected_bases

    mark = len(fake.calls)
    store.release()
    check_release(fake, mark)
    mark = len(fake.calls)
    store.release()
    assert fake.calls[mark:] == []
    assert fake.memory == {}


#: ``(op, n)`` failure points of ``allocate`` and ``import_peers`` on rank 1 of 3 with four chunks:
#: allocate makes four creates, reserves, maps, read-write accesses and views; import_peers makes
#: eight imports, reserves (calls 5-12 overall), maps and read-only accesses.
FAILURE_POINTS = [
    ("create", 1),
    ("create", 3),
    ("reserve", 2),
    ("map", 3),
    ("access_rw", 2),
    ("view", 4),
    ("stream_sync", 1),
    ("export", 2),
    ("import", 1),
    ("import", 6),
    ("reserve", 5),
    ("reserve", 11),
    ("map", 8),
    ("access_ro", 1),
    ("access_ro", 7),
]


@pytest.mark.cpu_only
@pytest.mark.parametrize("fail", FAILURE_POINTS, ids=[f"{op}-{n}" for op, n in FAILURE_POINTS])
def test_fabric_release_after_a_partial_allocate_or_import(monkeypatch, fail):
    """A step that fails part way does not run again and leaves what it created to ``release``."""
    num_ranks, rank = 3, 1
    layout, _ = collect(BIG_LAYERS, num_ranks, chunk_mb=1)
    assert len(layout.chunks) == 4
    fake = FakeVmm(monkeypatch, fail=fail)
    store = FabricShardStore(layout, rank, torch.device("cuda", 0))
    failed = "allocate"
    with pytest.raises(RuntimeError, match="injected failure"):
        store.allocate()
        failed = "write_local_rows"
        store.write_local_rows()
        failed = "export"
        payloads = peer_payloads(fake, store, layout, rank)
        failed = "import_peers"
        store.import_peers(payloads)
    with pytest.raises(RuntimeError):
        store.chunk_bases()
    if failed == "allocate":
        with pytest.raises(RuntimeError, match="allocate already ran"):
            store.allocate()
        for step in (store.write_local_rows, store.export):
            with pytest.raises(RuntimeError, match="before a successful allocate"):
                step()
    elif failed == "import_peers":
        with pytest.raises(RuntimeError, match="import_peers already ran"):
            store.import_peers(payloads)
    mark = len(fake.calls)
    store.release()
    check_release(fake, mark)
    assert fake.memory == {}
    mark = len(fake.calls)
    store.release()
    assert fake.calls[mark:] == []


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "chunk_mb, failing, layers", [(0, 1, "layers 1..4"), (3, 2, "layers 3..4"), (1, 3, "layer 3")]
)
def test_a_refused_store_handle_names_its_chunk_and_the_chunking_switch(
    monkeypatch, chunk_mb, failing, layers
):
    """A refused ``cuMemCreate`` names the chunk, its size and the switch that splits the store."""
    num_ranks, rank = 4, 2
    layout, _ = collect(BIG_LAYERS, num_ranks, chunk_mb=chunk_mb)
    chunk = layout.chunks[failing - 1]
    fake = FakeVmm(monkeypatch, fail=("create", failing))
    store = FabricShardStore(layout, rank, torch.device("cuda", 0))
    with pytest.raises(RuntimeError, match="injected failure of create") as error:
        store.allocate()
    size = dense_store._handle_size(chunk.handle_bytes, FakeVmm.GRANULARITY)
    assert (
        f"cuMemCreate of the FABRIC handle of store chunk {chunk.index} of {len(layout.chunks)} "
        f"({size} bytes, {layers}) on cuda:0 failed: injected failure of create #{failing}. "
        f"TRTLLM_DWDP_DENSE_STORE_CHUNK_MB (now {chunk_mb}) splits the store into handles of "
        f"whole layers of about that many MiB"
    ) in str(error.value)
    assert isinstance(error.value.__cause__, RuntimeError)
    mark = len(fake.calls)
    store.release()
    check_release(fake, mark)
    assert fake.memory == {}


@pytest.mark.cpu_only
def test_fabric_release_runs_every_step_and_raises_the_first_error(monkeypatch):
    num_ranks, rank = 3, 0
    layout, _ = collect(BIG_LAYERS, num_ranks, chunk_mb=3)
    fake = FakeVmm(monkeypatch)
    store = FabricShardStore(layout, rank, torch.device("cuda", 0))
    store.allocate()
    store.import_peers(peer_payloads(fake, store, layout, rank))
    fake.fail = ("unmap", 1)
    mark = len(fake.calls)
    with pytest.raises(RuntimeError, match="injected failure of unmap #1"):
        store.release()
    released = fake.values("release", mark)
    assert len(released) == len(fake.values("create")) + len(fake.values("import"))
    assert len(fake.values("free", mark)) == len(fake.values("reserve"))
    mark = len(fake.calls)
    store.release()
    assert fake.calls[mark:] == []


@pytest.mark.cpu_only
def test_fabric_release_raises_the_first_of_several_errors(monkeypatch, store_log):
    """With every unmap failing, release still frees everything, raises the first, logs the rest."""
    num_ranks, rank = 3, 0
    layout, _ = collect(BIG_LAYERS, num_ranks, chunk_mb=3)
    fake = FakeVmm(monkeypatch)
    store = FabricShardStore(layout, rank, torch.device("cuda", 0))
    store.allocate()
    store.import_peers(peer_payloads(fake, store, layout, rank))
    mapped = len(fake.values("map"))
    handles = [handle for handle, _ in fake.values("create") + fake.values("import")]
    fake.fail_every = {"unmap"}
    mark = len(fake.calls)
    with pytest.raises(RuntimeError, match=r"^injected failure of unmap #1$"):
        store.release()
    assert mapped == num_ranks * len(layout.chunks) and fake.counts["unmap"] == mapped
    assert sorted(fake.values("release", mark)) == sorted(handles)
    assert fake.memory == {}
    later = [message for message in store_log.messages("error") if "also failed" in message]
    assert len(later) == mapped - 1
    assert [re.search(r"unmap #(\d+)", message).group(1) for message in later] == [
        str(number) for number in range(2, mapped + 1)
    ]


@pytest.mark.cpu_only
def test_fabric_allocate_rejects_a_view_that_copies(monkeypatch):
    """A local view must alias its mapping; after that failure the store is not allocated."""
    num_ranks, rank = 3, 1
    layout, _ = collect(BIG_LAYERS, num_ranks, chunk_mb=1)
    fake = FakeVmm(monkeypatch, aliasing_views=False)
    store = FabricShardStore(layout, rank, torch.device("cuda", 0))
    with pytest.raises(RuntimeError, match="view of store chunk 0 at 0x[0-9a-f]+ is a copy"):
        store.allocate()
    assert fake.ops().count("view") == 1
    for step in (store.write_local_rows, store.export):
        with pytest.raises(RuntimeError, match="before a successful allocate"):
            step()
    with pytest.raises(RuntimeError, match="before a successful allocate"):
        store.import_peers([[b"handle"] * len(layout.chunks)] * num_ranks)
    mark = len(fake.calls)
    store.release()
    check_release(fake, mark)
    assert fake.memory == {}


@pytest.mark.cpu_only
def test_del_releases_a_forgotten_store(monkeypatch, unraisable):
    """A store dropped without ``release`` is released by ``__del__``, in the same order."""
    num_ranks, rank = 3, 1
    layout, _ = collect(BIG_LAYERS, num_ranks, chunk_mb=1)
    fake = FakeVmm(monkeypatch)
    store = FabricShardStore(layout, rank, torch.device("cuda", 0))
    store.allocate()
    store.write_local_rows()
    store.import_peers(peer_payloads(fake, store, layout, rank))
    mark = len(fake.calls)
    del store
    gc.collect()
    check_release(fake, mark)
    assert fake.memory == {}
    assert unraisable == []


@pytest.mark.cpu_only
def test_del_never_raises(monkeypatch, store_log, unraisable):
    """``__del__`` logs a failing release instead of raising; the release still ran every step."""
    num_ranks, rank = 3, 0
    layout, _ = collect(BIG_LAYERS, num_ranks, chunk_mb=3)
    fake = FakeVmm(monkeypatch)
    store = FabricShardStore(layout, rank, torch.device("cuda", 0))
    store.allocate()
    store.import_peers(peer_payloads(fake, store, layout, rank))
    handles = [handle for handle, _ in fake.values("create") + fake.values("import")]
    fake.fail_every = {"unmap"}
    mark = len(fake.calls)
    del store
    gc.collect()
    assert unraisable == []
    assert sorted(fake.values("release", mark)) == sorted(handles)
    assert fake.memory == {}
    assert any(
        "release from __del__ failed" in message and "injected failure of unmap #1" in message
        for message in store_log.messages("debug")
    )


@pytest.mark.cpu_only
def test_del_leaves_a_store_whose_constructor_raised_alone(monkeypatch, unraisable):
    """Finalizing a store whose constructor raised part way makes no call and raises nothing."""
    layout, _ = collect(ODD_LAYERS, 4)
    fake = FakeVmm(monkeypatch)
    finalized = []
    finalize = DenseShardStore.__del__

    def recording_del(store) -> None:
        finalized.append(type(store).__name__)
        finalize(store)

    monkeypatch.setattr(DenseShardStore, "__del__", recording_del)
    cuda = torch.device("cuda", 0)
    for make in (
        lambda: LoopbackShardStore(layout, -1, CPU),
        lambda: FabricShardStore(layout, 4, cuda),
        lambda: FabricShardStore(layout, 0, CPU),
    ):
        with pytest.raises(ValueError):
            make()
    gc.collect()
    assert sorted(finalized) == ["FabricShardStore", "FabricShardStore", "LoopbackShardStore"]
    assert fake.calls == []
    assert unraisable == []


@pytest.mark.cpu_only
def test_fabric_import_checks_the_payloads(monkeypatch):
    num_ranks, rank = 3, 1
    layout, _ = collect(BIG_LAYERS, num_ranks, chunk_mb=4)
    fake = FakeVmm(monkeypatch)

    def fresh() -> FabricShardStore:
        store = FabricShardStore(layout, rank, torch.device("cuda", 0))
        store.allocate()
        return store

    store = fresh()
    with pytest.raises(RuntimeError, match="before export"):
        store.import_peers([None] * num_ranks)
    good = peer_payloads(fake, store, layout, rank)
    store.release()
    num_chunks = len(layout.chunks)
    assert num_chunks == 2
    bad_payloads = {
        "one payload per rank": good[:2],
        rf"rank 0's store payload must be {num_chunks} exported FABRIC .*got NoneType": [
            None,
            *good[1:],
        ],
        r"rank 2's store payload .*got a list of 1 \(bytes\)": [*good[:2], good[2][:1]],
        r"rank 0's store payload .*got a list of 2 \(bytes, str\)": [[b"x", "y"], *good[1:]],
        r"rank 2's store payload .*\(non-empty bytes\), got a list of 2 \(bytes\)": [
            *good[:2],
            [good[2][0], b""],
        ],
        r"rank 0's store payload .*got a list of 0 \(nothing\)": [[], *good[1:]],
        "not this rank's export": [good[0], good[2], good[1]],
    }
    for message, payloads in bad_payloads.items():
        store = fresh()
        store.export()
        with pytest.raises(ValueError, match=message):
            store.import_peers(payloads)
        with pytest.raises(RuntimeError, match="import_peers already ran"):
            store.import_peers(payloads)
        assert fake.values("import") == []
        store.release()


@pytest.mark.cpu_only
def test_fabric_store_needs_fabric_handles(monkeypatch):
    layout, _ = collect(ODD_LAYERS, 4)
    fake = FakeVmm(monkeypatch, handle_type=POSIX_FD)
    store = FabricShardStore(layout, 0, torch.device("cuda", 0))
    with pytest.raises(RuntimeError, match="needs FABRIC peer handles"):
        store.allocate()
    assert fake.calls == []
    store.release()
    assert fake.calls == []


@pytest.mark.cpu_only
def test_a_chunk_without_rows_still_gets_a_base(monkeypatch, store_log):
    """Trailing empty layers in a chunk of their own: one granule, a distinct aligned base.

    The FABRIC store creates, exports, imports and maps that granule like any other handle, and
    logs that it allocated one granule more than ``layout.store_bytes``.
    """
    switches = DenseConfig(store_chunk_mb=1).digest_switches(4)
    sizes = {(1, ATTN): 8 << 20, (1, MOE): 12345, (2, ATTN): 0, (2, MOE): 0}
    layout = build_dense_layout(sizes, switches)
    assert [(chunk.layers, chunk.handle_bytes) for chunk in layout.chunks] == [
        ((1,), 4 << 20),
        ((2,), 0),
    ]
    store = loopback(layout)
    bases = store.chunk_bases()
    assert all(base % ROW_ALIGN == 0 for row in bases for base in row)
    assert len({base for row in bases for base in row}) == 8
    store.release()

    fake = FakeVmm(monkeypatch)
    fabric = FabricShardStore(layout, 0, torch.device("cuda", 0))
    fabric.allocate()
    fabric.write_local_rows()  # A layout from blob sizes alone has no anchors: rows stay zero.
    handle_sizes = [4 << 20, FakeVmm.GRANULARITY]
    assert [size for _, size in fake.values("create")] == handle_sizes
    assert layout.store_bytes == 4 << 20
    (info,) = store_log.messages("info")
    assert f"{FakeVmm.GRANULARITY} more than the layout's store_bytes ({4 << 20})" in info
    mark = len(fake.calls)
    fabric.import_peers(peer_payloads(fake, fabric, layout, 0))
    assert [size for _, size in fake.values("reserve", mark)] == handle_sizes * 3
    assert all(base % ROW_ALIGN == 0 for row in fabric.chunk_bases() for base in row)
    mark = len(fake.calls)
    fabric.release()
    check_release(fake, mark)


@pytest.mark.cpu_only
def test_fabric_handles_are_whole_granules_of_the_device(monkeypatch, store_log):
    """With a granularity above ``STORE_ALIGN`` each handle rounds up to whole granules of it.

    A chunk of three granules of ``STORE_ALIGN`` takes two of 4 MiB, a chunk of one takes one.
    ``allocate`` logs how much more than ``layout.store_bytes`` it allocated, and the peers'
    handles are reserved and mapped whole, at the device's granularity.
    """
    num_ranks, rank = 3, 2
    layout, _ = collect(BIG_LAYERS, num_ranks, chunk_mb=6)
    granularity = 2 * STORE_ALIGN
    assert [chunk.handle_bytes for chunk in layout.chunks] == [3 * STORE_ALIGN, STORE_ALIGN]
    sizes = [2 * granularity, granularity]
    fake = FakeVmm(monkeypatch, granularity=granularity)
    store = FabricShardStore(layout, rank, torch.device("cuda", 0))
    store.allocate()
    assert [size for _, size in fake.values("create")] == sizes
    (info,) = store_log.messages("info")
    extra = sum(sizes) - layout.store_bytes
    assert f"{extra} more than the layout's store_bytes ({layout.store_bytes})" in info
    store.write_local_rows()
    mark = len(fake.calls)
    store.import_peers(peer_payloads(fake, store, layout, rank))
    assert [size for _, size in fake.values("reserve", mark)] == sizes * (num_ranks - 1)
    mark = len(fake.calls)
    store.release()
    check_release(fake, mark)
    assert fake.memory == {}


@pytest.mark.cpu_only
def test_fabric_import_rejects_unaligned_mappings(monkeypatch):
    """Mappings off a ``ROW_ALIGN`` boundary fail ``import_peers``; ``release`` frees them all."""
    num_ranks, rank = 3, 0
    layout, _ = collect(BIG_LAYERS, num_ranks, chunk_mb=4)
    fake = FakeVmm(monkeypatch, reserve_skew=512)
    store = FabricShardStore(layout, rank, torch.device("cuda", 0))
    store.allocate()
    store.write_local_rows()
    payloads = peer_payloads(fake, store, layout, rank)
    with pytest.raises(RuntimeError, match=rf"store mappings .* are not {ROW_ALIGN}-aligned"):
        store.import_peers(payloads)
    assert len(fake.values("map")) == num_ranks * len(layout.chunks)
    with pytest.raises(RuntimeError, match="before import_peers"):
        store.chunk_bases()
    with pytest.raises(RuntimeError, match="import_peers already ran"):
        store.import_peers(payloads)
    mark = len(fake.calls)
    store.release()
    check_release(fake, mark)
    assert fake.memory == {}


# ---------------------------------------------------------------------------------------------
# GPU: loopback store and copy launcher
# ---------------------------------------------------------------------------------------------

GPU_RANKS = (0, 5, 15)
GPU_NUM_RANKS = 16


def _cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    return torch.device("cuda", torch.cuda.current_device())


def _require_gather(device: torch.device) -> None:
    """Skip unless the gather op is registered and has a kernel for this GPU."""
    op = fabric_gather.gather_op()
    if op is None:
        pytest.skip("trtllm::dwdp_fabric_gather is not registered by this TensorRT-LLM build")
    src = torch.arange(fabric_gather.LINE_BYTES, dtype=torch.uint8, device=device)
    dst = torch.zeros_like(src)

    def int64(values):
        return torch.tensor(values, dtype=torch.int64, device=device)

    try:
        op(int64([dst.data_ptr()]), int64([src.data_ptr()]), int64([1]), int64([0, 1]), 1, 32)
        torch.cuda.synchronize(device)
    except RuntimeError as exc:
        if "no kernel image" in str(exc):
            pytest.skip(f"this build has no gather kernel for this GPU: {exc}")
        raise
    assert torch.equal(dst, src), "a one-line probe gather did not copy its 16 bytes"


@pytest.mark.parametrize("mode", list(CopyMode), ids=[mode.value for mode in CopyMode])
def test_loopback_launcher_reassembles_every_blob_on_gpu(mode):
    """Loopback store + ``DenseCopyLauncher`` rebuild every blob byte-exact on one GPU.

    A 16-rank group (15 gather rows) over the odd layers; the plans of ranks 0, 5 and 15 run on a
    copy stream, as setup builds and launches them.  Each slot must hold the blob that
    ``write_blob`` writes and keep its poison elsewhere.
    """
    device = _cuda_device()
    if mode is CopyMode.GATHER:
        _require_gather(device)
    layout, layers = collect(ODD_LAYERS, GPU_NUM_RANKS, device)
    expected = {
        key: blob(segment, GPU_NUM_RANKS, device) for key, segment in layout.segments.items()
    }
    compute_stream = torch.cuda.current_stream(device)
    copy_stream = torch.cuda.Stream(device)
    failures = []
    compared = 0
    for rank in GPU_RANKS:
        store = loopback(layout, rank, device)
        rings = slot_rings(layout, device)
        plans = build_copy_plans(
            layout, rank=rank, chunk_bases=store.chunk_bases(), slot_bases=slot_bases(rings)
        )
        with torch.cuda.stream(copy_stream):
            launcher = DenseCopyLauncher(plans, mode, device=device)
        try:
            for layer in layout.layers:
                poison(layout, rings, layer)
                copy_stream.wait_stream(compute_stream)
                with torch.cuda.stream(copy_stream):
                    launcher(layer)
                compute_stream.wait_stream(copy_stream)
                torch.cuda.synchronize(device)
                failures += slot_mismatches(layout, rings, layer, expected, f"rank {rank}")
                compared += len(layout.layer_segments(layer))
        finally:
            launcher.release()
            store.release()
    assert not failures, "\n".join(failures)
    assert compared == len(GPU_RANKS) * len(layout.segments)
    del layers


def test_loopback_release_synchronizes_the_device(monkeypatch):
    """The loopback store synchronizes its device once, while its buffers are still allocated."""
    device = _cuda_device()
    layout, layers = collect(ODD_LAYERS, 4, device)
    store = loopback(layout, 0, device)
    synchronize = torch.cuda.synchronize
    allocated_at_sync = []

    def recording_synchronize(target=None) -> None:
        allocated_at_sync.append((torch.device(target), torch.cuda.memory_allocated(device)))
        synchronize(target)

    synchronize(device)
    allocated = torch.cuda.memory_allocated(device)
    monkeypatch.setattr(torch.cuda, "synchronize", recording_synchronize)
    store.release()
    store.release()
    assert allocated_at_sync == [(device, allocated)]
    assert torch.cuda.memory_allocated(device) < allocated
    del layers
