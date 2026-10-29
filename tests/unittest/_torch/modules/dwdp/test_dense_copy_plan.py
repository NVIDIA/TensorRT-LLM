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
"""Tests for the dense DWDP copy plans and their launcher.

The CPU tests (``cpu_only``) build plans over layouts made from blob sizes alone -- Kimi-K3's 92
layers, and a small layout of odd sizes -- at fake addresses or at the real addresses of host
buffers, and check that:

* the plan, its batched pieces and its gather arguments conserve bytes;
* destinations never overlap and cover every row of every slot exactly once, the local row
  included;
* every descriptor reads rank ``p``'s row at ``chunk_bases[p][chunk] + chunk_offset`` and writes
  it to ``base + p * S`` of the slot holding the layer, in the plan's slice order;
* the plan does not depend on the rank: the same row shapes and the same byte mapping;
* 15 rows give ``blocks_per_row == 2``; a plan is never None and never falls back to the copy
  engine;
* a bad layer, rank, address table or segment is rejected by the check that names it;
* applied to host buffers, every plan reassembles every blob byte-exact;
* each transfer of the launcher enqueues exactly the work it prepared, on the current stream; the
  gather passes ``fabric_gather.THREADS`` and the per-slice views use the launcher's device.

The odd layout has an empty segment, rows of one 4 KiB page (a single non-empty local slice),
blob sizes that are not multiples of 16, a row cut into 15 one-page slices, rows whose last slice
is shorter, rows with trailing empty slices, a layer with nothing to move, and covered layers that
are not consecutive.

``test_plan_reassembles_rows_on_gpu`` runs the real ``trtllm::dwdp_fabric_gather`` and both
copy-engine transfers on one GPU, with local buffers standing in for the 16 ranks' stores.  The op
latches the gather kernel (``TRTLLM_DWDP_GATHER_KERNEL``) per process, so the test re-runs this
file as a child process (``--kernel``, ``--out``) per kernel; the child writes a JSON report that
the parent turns into a pass, a failure listing every mismatch, or a skip.
"""

import argparse
import bisect
import itertools
import json
import math
import os
import re
import subprocess
import sys
import traceback
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modules.dwdp import fabric_gather, vmm
from tensorrt_llm._torch.modules.dwdp.dense import plan as dense_plan
from tensorrt_llm._torch.modules.dwdp.dense.config import DenseConfig
from tensorrt_llm._torch.modules.dwdp.dense.plan import (
    DenseCopyLauncher,
    batched_copies,
    build_copy_plans,
    build_layer_plan,
    gather_launch_args,
    local_slices,
    select_copy_mode,
)
from tensorrt_llm._torch.modules.dwdp.dense.types import (
    BATCH_SLICE_BYTES,
    RING_DEPTH,
    ROW_ALIGN,
    SEGMENT_ORDER,
    CopyDescriptor,
    CopyMode,
    DenseLayout,
    DenseSetupError,
    LayerCopyPlan,
    SegmentKind,
    SlotBuffer,
    allocate_slot_ring,
    build_dense_layout,
)

NUM_RANKS = 16
NUM_ROWS = NUM_RANKS - 1
ATTN = SegmentKind.ATTN
MOE = SegmentKind.MOE
CPU = torch.device("cpu")
_BLOCKS_ENV = "TRTLLM_DWDP_GATHER_BLOCKS"

# Kimi-K3 blob sizes: KDA attention, MLA attention (24 layers), MoE-dense.
K3_KDA_ATTN = 471_500_000
K3_MLA_ATTN = 275_500_000
K3_MOE = 213_300_000
K3_MLA_LAYERS = frozenset(range(3, 93, 4)) | {92}
K3_BLOBS = {
    layer: (K3_MLA_ATTN if layer in K3_MLA_LAYERS else K3_KDA_ATTN, K3_MOE)
    for layer in range(1, 93)
}

# Covered layer -> (attention, MoE-dense) blob bytes.  Row widths S and slice widths q in pages:
#   1: A S=458, q=31, the last of 15 slices 24 pages | D S=16, q=2: 8 slices, then 7 empty ones
#   2: A and D S=1: only slice 0 is non-empty (D is 7 bytes)
#   4: A S=15: 15 one-page slices | D empty
#   5: A empty | D S=321, q=22, the last slice 13 pages
#   7: nothing moves: an empty plan
#   9: A S=16: 8 slices | D S=1
#  10: A S=1 (16 bytes) | D S=29, q=2, the last slice 1 page
ODD_BLOBS = {
    1: (29_999_989, 1_000_003),
    2: (40_000, 7),
    4: (983_040, 0),
    5: (0, 20_971_537),
    7: (0, 0),
    9: (1_048_576, 12_345),
    10: (16, 1_899_997),
}

# name -> (blob sizes, segments, TRTLLM_DWDP_DENSE_STORE_CHUNK_MB)
LAYOUT_CASES = {
    "k3": (K3_BLOBS, ("attn", "moe"), 0),
    "k3-chunked": (K3_BLOBS, ("attn", "moe"), 1024),
    "k3-attn": (K3_BLOBS, ("attn",), 0),
    "odd": (ODD_BLOBS, ("attn", "moe"), 0),
    "odd-chunked": (ODD_BLOBS, ("attn", "moe"), 1),
    "odd-attn": (ODD_BLOBS, ("attn",), 0),
}
ALL_LAYOUTS = pytest.mark.parametrize("case", list(LAYOUT_CASES))
HOST_LAYOUTS = pytest.mark.parametrize("case", ["odd", "odd-chunked", "odd-attn"])

CANARY = 0xA5  # Store bytes outside every row.
POISON = 0x5A  # Slot bytes before a copy.


def _layout(blobs, *, segments=("attn", "moe"), store_chunk_mb=0) -> DenseLayout:
    config = DenseConfig(segments=segments, store_chunk_mb=store_chunk_mb)
    sizes = {}
    for layer, (attn_bytes, moe_bytes) in blobs.items():
        sizes[(layer, ATTN)] = attn_bytes
        if config.has_moe:
            sizes[(layer, MOE)] = moe_bytes
    return build_dense_layout(sizes, config.digest_switches(NUM_RANKS))


def _case_layout(case: str) -> DenseLayout:
    blobs, segments, store_chunk_mb = LAYOUT_CASES[case]
    return _layout(blobs, segments=segments, store_chunk_mb=store_chunk_mb)


class _FakeTables:
    """Fake, widely spaced, ``ROW_ALIGN``-aligned addresses of every store chunk and ring slot.

    Every address decodes back to what it names, so a descriptor's source and destination can be
    checked without trusting the plan.
    """

    STORE = 1 << 46
    RANK_STRIDE = 1 << 41
    CHUNK_STRIDE = 1 << 35  # Larger than any store chunk.
    SLOTS = 1 << 44
    SEGMENT_STRIDE = 1 << 40
    SLOT_STRIDE = 1 << 34  # Larger than any slot.

    def __init__(self, layout: DenseLayout) -> None:
        self.layout = layout
        self.chunk_bases = [
            [
                self.STORE + rank * self.RANK_STRIDE + chunk * self.CHUNK_STRIDE
                for chunk in range(len(layout.chunks))
            ]
            for rank in range(layout.num_ranks)
        ]
        self.slot_bases = {
            kind: [
                self.SLOTS + SEGMENT_ORDER.index(kind) * self.SEGMENT_STRIDE + i * self.SLOT_STRIDE
                for i in range(spec.depth)
            ]
            for kind, spec in layout.slot_specs.items()
        }

    def plans(self, rank: int) -> dict[int, LayerCopyPlan]:
        return build_copy_plans(
            self.layout, rank=rank, chunk_bases=self.chunk_bases, slot_bases=self.slot_bases
        )

    def source(self, descriptor: CopyDescriptor) -> tuple[int, int, int]:
        """``(rank, chunk, offset in the chunk)`` of the store bytes a descriptor reads."""
        rank, rest = divmod(descriptor.src - self.STORE, self.RANK_STRIDE)
        chunk, offset = divmod(rest, self.CHUNK_STRIDE)
        assert 0 <= rank < self.layout.num_ranks and 0 <= chunk < len(self.layout.chunks)
        assert offset + descriptor.nbytes <= self.layout.chunks[chunk].nbytes, (
            f"{descriptor} reads past the rows of chunk {chunk}"
        )
        return rank, chunk, offset

    def slot_of(self, descriptor: CopyDescriptor) -> tuple[SegmentKind, int, int]:
        """``(segment, ring slot, offset from the slot base)`` of what a descriptor writes."""
        kind_index, rest = divmod(descriptor.dst - self.SLOTS, self.SEGMENT_STRIDE)
        slot, offset = divmod(rest, self.SLOT_STRIDE)
        kind = SEGMENT_ORDER[kind_index]
        assert kind in self.layout.segment_kinds and 0 <= slot < RING_DEPTH[kind]
        return kind, slot, offset

    def destination(self, layer: int, descriptor: CopyDescriptor) -> tuple[SegmentKind, int, int]:
        """``(segment, row, offset in the row)`` of what a descriptor writes for ``layer``."""
        kind, slot, offset = self.slot_of(descriptor)
        assert slot == self.layout.slot_index(kind, layer), (
            f"{descriptor} writes {kind.value} slot {slot}, which does not hold layer {layer}"
        )
        width = self.layout.segment(layer, kind).row_bytes
        assert width > 0, f"{descriptor} writes the empty {kind.value} blob of layer {layer}"
        row, offset = divmod(offset, width)
        assert row < self.layout.num_ranks and offset + descriptor.nbytes <= width, (
            f"{descriptor} leaves the blob or straddles two rows"
        )
        return kind, row, offset


def _design_slices(width: int) -> list[tuple[int, int]]:
    """The slicing rule: ``q = align_up(ceil(S / 15), 4096)``, slice ``k = (k q, min(q, S - k q))``."""
    q = -(-width // NUM_ROWS)
    q = -(-q // ROW_ALIGN) * ROW_ALIGN
    return [(k * q, max(0, min(q, width - k * q))) for k in range(NUM_ROWS)]


def _byte_mapping(copies) -> list[tuple[int, int, int]]:
    """The minimal sorted ``(dst, src, nbytes)`` runs of the bytes that ``copies`` move."""
    runs = []
    for dst, src, nbytes in sorted(copies):
        if runs and runs[-1][0] + runs[-1][2] == dst and runs[-1][1] + runs[-1][2] == src:
            runs[-1] = (runs[-1][0], runs[-1][1], runs[-1][2] + nbytes)
        else:
            runs.append((dst, src, nbytes))
    return runs


def _overlaps(intervals) -> list:
    """Neighbouring ``(start, nbytes)`` intervals that overlap, after sorting."""
    ordered = sorted(intervals)
    return [(a, b) for a, b in zip(ordered, ordered[1:]) if a[0] + a[1] > b[0]]


def _tiling_errors(intervals, start: int, end: int) -> list[str]:
    """Why ``(start, nbytes)`` intervals do not cover ``[start, end)`` exactly once."""
    errors = []
    cursor = start
    for first, nbytes in sorted(intervals):
        if first != cursor:
            errors.append(f"{'gap' if first > cursor else 'overlap'} at {cursor - start:#x}")
        cursor = max(cursor, first + nbytes)
    if cursor != end:
        errors.append(f"covers up to {cursor - start:#x}, not {end - start:#x}")
    return errors


class _Cluster:
    """The 16 ranks' row stores and one set of slot rings, in the memory of one device.

    Rank ``p``'s chunk ``c`` holds row ``p`` of every ``(layer, segment)`` of the chunk at its
    ``chunk_offset`` -- zero past the blob, as the store's P2 writes it -- and canary bytes
    everywhere else.  The addresses are real, so the tables drive the real transfers.
    """

    def __init__(self, layout: DenseLayout, device: torch.device, seed: int) -> None:
        self.layout = layout
        self._regions: list[tuple[int, torch.Tensor]] = []
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        # (layer, segment) -> the blob zero-padded to N * S: what a gather leaves in the slot.
        self.expected = {}
        for key, segment in layout.segments.items():
            padded = torch.zeros(NUM_RANKS * segment.row_bytes, dtype=torch.uint8, device=device)
            padded[: segment.blob_bytes].random_(0, 256, generator=generator)
            self.expected[key] = padded
        self.chunk_bases = []
        for rank in range(NUM_RANKS):
            bases = []
            for chunk in layout.chunks:
                address, store = self._allocate(chunk.handle_bytes, device)
                for layer in chunk.layers:
                    for segment in layout.layer_segments(layer):
                        width = segment.row_bytes
                        row = self.expected[segment.key][rank * width : (rank + 1) * width]
                        store[segment.chunk_offset : segment.chunk_offset + width] = row
                bases.append(address)
            self.chunk_bases.append(bases)
        self.slots = {
            kind: allocate_slot_ring(spec, device) for kind, spec in layout.slot_specs.items()
        }
        for ring in self.slots.values():
            for slot in ring:
                self._regions.append((slot.storage.data_ptr(), slot.storage))
        self.slot_bases = {kind: [slot.base for slot in ring] for kind, ring in self.slots.items()}

    def _allocate(self, nbytes: int, device: torch.device) -> tuple[int, torch.Tensor]:
        raw = torch.full((nbytes + ROW_ALIGN,), CANARY, dtype=torch.uint8, device=device)
        pad = (-raw.data_ptr()) % ROW_ALIGN
        self._regions.append((raw.data_ptr(), raw))
        return raw.data_ptr() + pad, raw[pad : pad + nbytes]

    def plans(self, rank: int) -> dict[int, LayerCopyPlan]:
        return build_copy_plans(
            self.layout, rank=rank, chunk_bases=self.chunk_bases, slot_bases=self.slot_bases
        )

    def view(self, address: int, nbytes: int) -> torch.Tensor:
        """The ``uint8 [nbytes]`` at ``address``, inside one of the cluster's buffers."""
        for start, buffer in self._regions:
            if start <= address and address + nbytes <= start + buffer.numel():
                return buffer[address - start : address - start + nbytes]
        raise AssertionError(f"[{address:#x}, +{nbytes}) is outside every buffer of the cluster")

    def tensor_from_ptr(self, ptr, shape, dtype, device_id) -> torch.Tensor:
        """``vmm.tensor_from_ptr`` over the cluster's buffers: a view that aliases ``ptr``."""
        return self.view(ptr, math.prod(shape) * dtype.itemsize).view(dtype).reshape(shape)

    def apply(self, plan: LayerCopyPlan) -> None:
        """Run a plan by the descriptor semantics: copy ``nbytes`` from ``src`` to ``dst``."""
        for descriptor in plan.descriptors:
            src = self.view(descriptor.src, descriptor.nbytes)
            self.view(descriptor.dst, descriptor.nbytes).copy_(src)

    def slot(self, kind: SegmentKind, layer: int) -> SlotBuffer:
        return self.slots[kind][self.layout.slot_index(kind, layer)]

    def poison(self, layer: int) -> None:
        for kind in self.layout.segment_kinds:
            self.slot(kind, layer).storage.fill_(POISON)

    def check(self, layer: int, rank: int, label: str) -> tuple[list[str], int]:
        """Mismatches of ``layer``'s slots, and the number of segments compared.

        Each slot must hold the zero-padded blob from its base and keep its poison elsewhere.
        """
        failures = []
        segments = self.layout.layer_segments(layer)
        for segment in segments:
            slot = self.slot(segment.segment, layer)
            width = segment.row_bytes
            nbytes = NUM_RANKS * width
            got = slot.blob_view(nbytes)
            want = self.expected[segment.key]
            if not torch.equal(got, want):
                wrong = torch.nonzero(got != want).flatten()
                row, offset = divmod(int(wrong[0]), width)
                where = "the local row" if row == rank else f"row {row}"
                failures.append(
                    f"{label}/{segment.segment.value}: {wrong.numel()} wrong bytes, the first in "
                    f"{where} at byte {offset}"
                )
            outside = (slot.storage[: slot.pad], slot.storage[slot.pad + nbytes :])
            if not all(bool((part == POISON).all()) for part in outside):
                failures.append(
                    f"{label}/{segment.segment.value}: slot bytes outside the blob were written"
                )
        return failures, len(segments)


class _FakeCudart:
    """The part of ``cuda.bindings.runtime`` the batched transfer uses; records every batch."""

    class cudaMemcpySrcAccessOrder:
        cudaMemcpySrcAccessOrderStream = "stream order"

    class cudaError_t:
        cudaSuccess = 0

    class cudaMemcpyAttributes:
        srcAccessOrder = None

    def __init__(self) -> None:
        self.calls = []
        self.result = self.cudaError_t.cudaSuccess

    def cudaMemcpyBatchAsync(self, *args):
        self.calls.append(args)
        return (self.result,)


# ---------------------------------------------------------------------------------------------
# CPU: slicing, addressing and coverage
# ---------------------------------------------------------------------------------------------


@pytest.mark.cpu_only
@pytest.mark.parametrize("pages", [0, 1, 2, 14, 15, 16, 29, 321, 458, 3255, 4204, 7195])
@pytest.mark.parametrize("num_slices", [1, 2, 15])
def test_local_slices_tile_the_row(pages, num_slices):
    """``num_slices`` aligned slices of width ``q`` tile ``[0, S)``; empty ones come last."""
    width = pages * ROW_ALIGN
    slices = local_slices(width, num_slices)
    q = -(-width // num_slices)
    q = -(-q // ROW_ALIGN) * ROW_ALIGN
    assert [offset for offset, _ in slices] == [k * q for k in range(num_slices)]
    lengths = [length for _, length in slices]
    assert sum(lengths) == width
    assert all(length % ROW_ALIGN == 0 for length in lengths)
    live = [length for length in lengths if length]
    assert lengths == live + [0] * (num_slices - len(live))
    assert all(length == q for length in live[:-1])
    assert not live or 0 < live[-1] <= q


@pytest.mark.cpu_only
def test_local_slices_of_k3_rows():
    """K3's KDA attention row has a shorter last slice; its MoE-dense row cuts into 15 equal ones."""
    kda = local_slices(29_470_720, NUM_ROWS)
    assert kda == [(k * 1_966_080, 1_966_080) for k in range(14)] + [(14 * 1_966_080, 1_945_600)]
    assert local_slices(13_332_480, NUM_ROWS) == [(k * 888_832, 888_832) for k in range(15)]
    assert local_slices(65_536, NUM_ROWS) == [(k * 8192, 8192 if k < 8 else 0) for k in range(15)]
    assert local_slices(0, NUM_ROWS) == [(0, 0)] * NUM_ROWS


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "width, num_slices", [(-ROW_ALIGN, 15), (ROW_ALIGN - 1, 15), (ROW_ALIGN + 16, 15), (0, 0)]
)
def test_local_slices_reject_bad_arguments(width, num_slices):
    with pytest.raises(ValueError):
        local_slices(width, num_slices)


@pytest.mark.cpu_only
def test_layouts_exercise_the_edge_cases():
    """The odd layout hits every slicing edge case; the chunked layouts use several chunks."""
    odd = _case_layout("odd")
    pages = {key: segment.row_bytes // ROW_ALIGN for key, segment in odd.segments.items()}
    assert pages == {
        (1, ATTN): 458,
        (1, MOE): 16,
        (2, ATTN): 1,
        (2, MOE): 1,
        (4, ATTN): 15,
        (4, MOE): 0,
        (5, ATTN): 0,
        (5, MOE): 321,
        (7, ATTN): 0,
        (7, MOE): 0,
        (9, ATTN): 16,
        (9, MOE): 1,
        (10, ATTN): 1,
        (10, MOE): 29,
    }
    slices = {
        key: [length // ROW_ALIGN for _, length in _design_slices(segment.row_bytes) if length]
        for key, segment in odd.segments.items()
    }
    assert slices[(1, ATTN)] == [31] * 14 + [24]
    assert slices[(1, MOE)] == [2] * 8
    assert slices[(2, ATTN)] == slices[(2, MOE)] == [1]
    assert slices[(4, ATTN)] == [1] * 15
    assert slices[(5, MOE)] == [22] * 14 + [13]
    assert slices[(10, MOE)] == [2] * 14 + [1]
    assert any(segment.blob_bytes % 16 for segment in odd.segments.values())
    chunked = _case_layout("odd-chunked")
    assert [chunk.layers for chunk in chunked.chunks] == [(1,), (2, 4), (5,), (7, 9, 10)]
    for layout in (chunked, _case_layout("k3-chunked")):
        assert len(layout.chunks) > 1
        assert any(seg.chunk > 0 and seg.chunk_offset > 0 for seg in layout.segments.values())


@pytest.mark.cpu_only
@ALL_LAYOUTS
def test_plan_conserves_bytes(case, monkeypatch):
    """Per segment, 15 peer rows and the own row arrive: ``N * S`` bytes, in every form."""
    monkeypatch.delenv(_BLOCKS_ENV, raising=False)
    layout = _case_layout(case)
    tables = _FakeTables(layout)
    for rank in range(NUM_RANKS):
        for layer, plan in tables.plans(rank).items():
            moved = {}
            for descriptor in plan.descriptors:
                owner = tables.source(descriptor)[0]
                kind = tables.destination(layer, descriptor)[0]
                key = (kind, owner == rank)
                moved[key] = moved.get(key, 0) + descriptor.nbytes
            for segment in layout.layer_segments(layer):
                width = segment.row_bytes
                assert moved.get((segment.segment, False), 0) == NUM_ROWS * width
                assert moved.get((segment.segment, True), 0) == width
                assert NUM_RANKS * width >= segment.blob_bytes
            total = sum(NUM_RANKS * segment.row_bytes for segment in layout.layer_segments(layer))
            assert plan.nbytes == total
            assert sum(batched_copies(plan)[2]) == total
            if not plan.is_empty:
                lines = gather_launch_args(plan, CPU)[2]
                assert int(lines.sum()) * fabric_gather.LINE_BYTES == total


@pytest.mark.cpu_only
@ALL_LAYOUTS
def test_destinations_never_overlap(case):
    """No two descriptors of a plan, or pieces of its batch, write the same byte."""
    layout = _case_layout(case)
    tables = _FakeTables(layout)
    for rank in range(NUM_RANKS):
        for layer, plan in tables.plans(rank).items():
            assert all(descriptor.nbytes > 0 for descriptor in plan.descriptors)
            intervals = [(descriptor.dst, descriptor.nbytes) for descriptor in plan.descriptors]
            assert _overlaps(intervals) == [], f"rank {rank}, layer {layer}"
            dst_ptrs, _, sizes = batched_copies(plan)
            assert _overlaps(list(zip(dst_ptrs, sizes))) == [], f"rank {rank}, layer {layer}"


@pytest.mark.cpu_only
@ALL_LAYOUTS
def test_destinations_cover_every_row_exactly_once(case):
    """``[base, base + N * S)`` of each segment is written exactly once, the local row by slices."""
    layout = _case_layout(case)
    tables = _FakeTables(layout)
    for rank in range(NUM_RANKS):
        for layer, plan in tables.plans(rank).items():
            written = {kind: [] for kind in layout.segment_kinds}
            local = {kind: [] for kind in layout.segment_kinds}
            for descriptor in plan.descriptors:
                kind, row, _ = tables.destination(layer, descriptor)
                written[kind].append((descriptor.dst, descriptor.nbytes))
                if row == rank:
                    local[kind].append((descriptor.dst, descriptor.nbytes))
            for segment in layout.layer_segments(layer):
                kind = segment.segment
                width = segment.row_bytes
                base = tables.slot_bases[kind][layout.slot_index(kind, layer)]
                label = f"rank {rank}, layer {layer}, {kind.value}"
                assert _tiling_errors(written[kind], base, base + NUM_RANKS * width) == [], label
                local_row = (base + rank * width, base + (rank + 1) * width)
                assert _tiling_errors(local[kind], *local_row) == [], label
                live_slices = sum(1 for _, length in _design_slices(width) if length)
                assert len(local[kind]) == live_slices, label


@pytest.mark.cpu_only
@ALL_LAYOUTS
def test_descriptors_follow_the_store_addressing(case):
    """Rank ``p``'s row is read at its chunk offset and written to row ``p``."""
    layout = _case_layout(case)
    tables = _FakeTables(layout)
    for rank in range(NUM_RANKS):
        for layer, plan in tables.plans(rank).items():
            for peer, row in zip(plan.peers, plan.rows):
                for descriptor in row:
                    assert all(value % ROW_ALIGN == 0 for value in descriptor), descriptor
                    owner, chunk, chunk_offset = tables.source(descriptor)
                    kind, dst_row, offset = tables.destination(layer, descriptor)
                    segment = layout.segment(layer, kind)
                    assert owner in (peer, rank)
                    assert dst_row == owner
                    assert chunk == segment.chunk
                    assert chunk_offset == segment.chunk_offset + offset


@pytest.mark.cpu_only
@ALL_LAYOUTS
def test_rows_hold_the_peer_rows_then_the_local_slices(case):
    """Row ``k``: A and D rows of peer ``p_k``, then slice ``k`` of the local A and D rows."""
    layout = _case_layout(case)
    tables = _FakeTables(layout)
    for rank in range(NUM_RANKS):
        peers = tuple(peer for peer in range(NUM_RANKS) if peer != rank)
        for layer, plan in tables.plans(rank).items():
            assert (plan.layer, plan.rank, plan.peers) == (layer, rank, peers)
            assert len(plan.rows) == NUM_ROWS
            segments = layout.layer_segments(layer)
            for k, (peer, row) in enumerate(zip(peers, plan.rows)):
                expected = [(seg.segment, peer, 0, seg.row_bytes) for seg in segments]
                expected += [
                    (seg.segment, rank, *_design_slices(seg.row_bytes)[k]) for seg in segments
                ]
                expected = [entry for entry in expected if entry[3]]
                got = [
                    (*tables.destination(layer, descriptor), descriptor.nbytes)
                    for descriptor in row
                ]
                assert got == expected, f"rank {rank}, layer {layer}, row {k}"


@pytest.mark.cpu_only
@pytest.mark.parametrize("case", ["k3", "odd"])
def test_layers_use_the_slot_of_their_position(case):
    """``A(n)`` goes to A slot ``pos % 2`` and ``D(n)`` to D slot ``pos % 3``."""
    layout = _case_layout(case)
    tables = _FakeTables(layout)
    for position, (layer, plan) in enumerate(tables.plans(5).items()):
        written = {tables.slot_of(descriptor)[:2] for descriptor in plan.descriptors}
        expected = {(ATTN, position % 2), (MOE, position % 3)}
        assert written == {
            (kind, slot) for kind, slot in expected if layout.segment(layer, kind).row_bytes
        }
        if case == "k3":
            assert written == {(ATTN, (layer - 1) % 2), (MOE, (layer - 1) % 3)}


@pytest.mark.cpu_only
@ALL_LAYOUTS
def test_plan_does_not_depend_on_the_rank(case):
    """Every rank's plan has the same row shapes and moves the same bytes to the same places."""
    layout = _case_layout(case)
    tables = _FakeTables(layout)
    reference_shape = reference_mapping = None
    for rank in range(NUM_RANKS):
        plans = tables.plans(rank)
        shape = {
            layer: [[descriptor.nbytes for descriptor in row] for row in plan.rows]
            for layer, plan in plans.items()
        }
        mapping = {layer: _byte_mapping(plan.descriptors) for layer, plan in plans.items()}
        if reference_shape is None:
            reference_shape, reference_mapping = shape, mapping
            continue
        assert shape == reference_shape, f"rank {rank} cuts the rows differently from rank 0"
        assert mapping == reference_mapping, f"rank {rank} moves other bytes than rank 0"


@pytest.mark.cpu_only
@pytest.mark.parametrize("case", ["k3", "k3-attn", "odd"])
def test_fifteen_rows_give_two_blocks_per_row(case, monkeypatch):
    """The gather arguments carry the plan rows unchanged: 15 rows, 2 blocks each (cluster-2)."""
    monkeypatch.delenv(_BLOCKS_ENV, raising=False)
    assert fabric_gather.blocks_per_row(NUM_ROWS) == 2
    layout = _case_layout(case)
    tables = _FakeTables(layout)
    for rank in (0, 9, 15):
        for plan in tables.plans(rank).values():
            if plan.is_empty:
                continue
            dst_ptrs, src_ptrs, lines, row_starts, blocks = gather_launch_args(plan, CPU)
            assert blocks == 2
            assert row_starts.tolist() == [0, *itertools.accumulate(map(len, plan.rows))]
            assert dst_ptrs.tolist() == [descriptor.dst for descriptor in plan.descriptors]
            assert src_ptrs.tolist() == [descriptor.src for descriptor in plan.descriptors]
            lines_bytes = (lines * fabric_gather.LINE_BYTES).tolist()
            assert lines_bytes == [descriptor.nbytes for descriptor in plan.descriptors]
            tensors = (dst_ptrs, src_ptrs, lines, row_starts)
            assert all(t.dtype == torch.int64 and t.device == CPU for t in tensors)


@pytest.mark.cpu_only
@ALL_LAYOUTS
def test_plan_is_never_none(case):
    """Every covered layer gets an aligned plan on every rank; non-empty ones gather."""
    layout = _case_layout(case)
    tables = _FakeTables(layout)
    for rank in range(NUM_RANKS):
        plans = tables.plans(rank)
        assert list(plans) == list(layout.layers)
        for layer in layout.layers:
            plan = build_layer_plan(
                layout,
                layer,
                rank=rank,
                chunk_bases=tables.chunk_bases,
                slot_bases=tables.slot_bases,
            )
            assert isinstance(plan, LayerCopyPlan) and plan == plans[layer]
            assert all(value % ROW_ALIGN == 0 for d in plan.descriptors for value in d)
            empty = all(segment.blob_bytes == 0 for segment in layout.layer_segments(layer))
            assert plan.is_empty == empty
            if empty:
                with pytest.raises(DenseSetupError, match="empty"):
                    gather_launch_args(plan, CPU)
            else:
                assert len(gather_launch_args(plan, CPU)) == 5


@pytest.mark.cpu_only
def test_gather_args_never_fall_back_to_the_copy_engine():
    """A plan ``build_plan`` rejects is a setup error, never a silent copy-engine fallback."""
    misaligned = LayerCopyPlan(
        layer=3,
        rank=0,
        peers=(1,),
        rows=((CopyDescriptor(dst=1 << 20, src=(2 << 20) + 8, nbytes=ROW_ALIGN),),),
    )
    with pytest.raises(DenseSetupError, match="no copy-engine fallback"):
        gather_launch_args(misaligned, CPU)
    with pytest.raises(DenseSetupError, match="empty"):
        gather_launch_args(LayerCopyPlan(layer=3, rank=0, peers=(1,), rows=((),)), CPU)


@pytest.mark.cpu_only
@pytest.mark.parametrize("case", ["k3-attn", "odd-attn"])
def test_attention_only_plans_have_no_moe_descriptors(case):
    """With ``SEGMENTS=attn`` a row holds the peer's A row and one local A slice at most."""
    layout = _case_layout(case)
    tables = _FakeTables(layout)
    for layer, plan in tables.plans(3).items():
        assert all(len(row) <= 2 for row in plan.rows)
        assert {tables.slot_of(descriptor)[0] for descriptor in plan.descriptors} <= {ATTN}
    moe_slots = [tables.SLOTS + tables.SEGMENT_STRIDE + i * tables.SLOT_STRIDE for i in range(3)]
    with pytest.raises(ValueError, match="enabled segments"):
        build_copy_plans(
            layout,
            rank=3,
            chunk_bases=tables.chunk_bases,
            slot_bases={**tables.slot_bases, MOE: moe_slots},
        )


def _replace(rows, index, value):
    return [*rows[:index], value, *rows[index + 1 :]]


# change -> (text the error message contains, keyword arguments of build_layer_plan to replace
# given (chunk_bases, slot_bases)).  In the odd layout a slot spans N * S_max bytes, far more than
# a page.
_BAD_ARGUMENTS = {
    "uncovered layer": ("layer 3 is not covered", lambda c, s: {"layer": 3}),
    "rank -1": ("rank must be in [0, 16), got -1", lambda c, s: {"rank": -1}),
    "rank N": ("rank must be in [0, 16), got 16", lambda c, s: {"rank": NUM_RANKS}),
    "15 ranks of chunks": (
        "chunk_bases needs one entry per rank (16), got 15",
        lambda c, s: {"chunk_bases": c[:-1]},
    ),
    "two chunks": (
        "chunk_bases[3] needs one address per store chunk (1), got 2",
        lambda c, s: {"chunk_bases": _replace(c, 3, [*c[3], c[3][0] + (1 << 30)])},
    ),
    "misaligned chunk": (
        "chunk_bases[5][0] must be a non-negative multiple",
        lambda c, s: {"chunk_bases": _replace(c, 5, [c[5][0] + 16])},
    ),
    "negative chunk": (
        "chunk_bases[5][0] must be a non-negative multiple",
        lambda c, s: {"chunk_bases": _replace(c, 5, [-ROW_ALIGN])},
    ),
    "float chunk": (
        "chunk_bases[5][0] must be an integer address",
        lambda c, s: {"chunk_bases": _replace(c, 5, [float(c[5][0])])},
    ),
    "no MoE slots": (
        "slot_bases must hold exactly the enabled segments",
        lambda c, s: {"slot_bases": {ATTN: s[ATTN]}},
    ),
    "unknown segment": (
        "'mlp' is not a valid SegmentKind",
        lambda c, s: {"slot_bases": {**s, "mlp": s[MOE]}},
    ),
    "three A slots": (
        "slot_bases[attn] needs one base per ring slot (2), got 3",
        lambda c, s: {"slot_bases": {**s, ATTN: [*s[ATTN], s[ATTN][1] + (1 << 34)]}},
    ),
    "misaligned slot": (
        "slot_bases[moe][0] must be a non-negative multiple",
        lambda c, s: {"slot_bases": {**s, MOE: [s[MOE][0] + 512, *s[MOE][1:]]}},
    ),
    "D slot inside an A slot": (
        "slots attn[1] and moe[0] overlap",
        lambda c, s: {"slot_bases": {**s, MOE: [s[ATTN][1] + ROW_ALIGN, *s[MOE][1:]]}},
    ),
    "duplicate A slots": (
        "slots attn[0] and attn[1] overlap",
        lambda c, s: {"slot_bases": {**s, ATTN: [s[ATTN][0], s[ATTN][0]]}},
    ),
    "overlapping D slots": (
        "slots moe[0] and moe[1] overlap",
        lambda c, s: {"slot_bases": {**s, MOE: [s[MOE][0], s[MOE][0] + ROW_ALIGN, s[MOE][2]]}},
    ),
}


@pytest.mark.cpu_only
@pytest.mark.parametrize("change", list(_BAD_ARGUMENTS))
def test_bad_arguments_are_rejected(change):
    """A bad layer, rank or address table raises the error of the check that guards it."""
    layout = _case_layout("odd")
    tables = _FakeTables(layout)
    kwargs = {
        "layer": 1,
        "rank": 4,
        "chunk_bases": tables.chunk_bases,
        "slot_bases": tables.slot_bases,
    }
    message, replace = _BAD_ARGUMENTS[change]
    kwargs.update(replace(tables.chunk_bases, tables.slot_bases))
    layer = kwargs.pop("layer")
    with pytest.raises(ValueError, match=re.escape(message)):
        build_layer_plan(layout, layer, **kwargs)
    if layout.covers(layer):
        with pytest.raises(ValueError, match=re.escape(message)):
            build_copy_plans(layout, **kwargs)


# change -> (SegmentLayout field, its bad value given the layout and the segment)
_BAD_SEGMENTS = {
    "negative S": ("row_bytes", lambda layout, segment: -ROW_ALIGN),
    "unaligned S": ("row_bytes", lambda layout, segment: ROW_ALIGN + 16),
    "S too wide for the slot": (
        "row_bytes",
        lambda layout, segment: layout.slot_specs[segment.segment].row_bytes + ROW_ALIGN,
    ),
    "negative chunk offset": ("chunk_offset", lambda layout, segment: -ROW_ALIGN),
    "unaligned chunk offset": ("chunk_offset", lambda layout, segment: segment.chunk_offset + 16),
    "chunk -1": ("chunk", lambda layout, segment: -1),
    "chunk past the store": ("chunk", lambda layout, segment: len(layout.chunks)),
}


@pytest.mark.cpu_only
@pytest.mark.parametrize("kind", [ATTN, MOE], ids=lambda kind: kind.value)
@pytest.mark.parametrize("change", list(_BAD_SEGMENTS))
def test_malformed_segments_are_rejected(change, kind):
    """A segment whose ``S``, chunk or chunk offset would misplace a descriptor is rejected.

    ``build_dense_layout`` never makes one, so one field of layer 1's segment is edited by hand;
    each edit breaks exactly one of the segment check's conditions.
    """
    layout = _case_layout("odd")
    tables = _FakeTables(layout)
    segment = layout.segment(1, kind)
    field, bad_value = _BAD_SEGMENTS[change]
    setattr(segment, field, bad_value(layout, segment))
    kwargs = {"rank": 4, "chunk_bases": tables.chunk_bases, "slot_bases": tables.slot_bases}
    message = re.escape(f"segment (1, {kind.value}) has ")
    with pytest.raises(ValueError, match=message):
        build_layer_plan(layout, 1, **kwargs)
    with pytest.raises(ValueError, match=message):
        build_copy_plans(layout, **kwargs)


def _piece_rows(plan: LayerCopyPlan, dst_ptrs) -> list[int]:
    """The plan row each batched piece belongs to, by the descriptor its destination lies in."""
    spans = sorted(
        (descriptor.dst, descriptor.nbytes, k)
        for k, row in enumerate(plan.rows)
        for descriptor in row
    )
    starts = [start for start, _, _ in spans]
    rows = []
    for dst in dst_ptrs:
        start, nbytes, k = spans[bisect.bisect_right(starts, dst) - 1]
        assert start <= dst < start + nbytes
        rows.append(k)
    return rows


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "case, slice_bytes",
    [
        ("k3", BATCH_SLICE_BYTES),
        ("odd", BATCH_SLICE_BYTES),
        ("odd", 3 * ROW_ALIGN),
        ("odd", 1000),
        ("odd-attn", 3 * ROW_ALIGN),
    ],
)
def test_batched_copies_slice_the_plan_and_rotate_over_rows(case, slice_bytes):
    """Pieces of at most ``slice_bytes``, round-robin over the rows, moving the plan's bytes."""
    layout = _case_layout(case)
    tables = _FakeTables(layout)
    for rank in (0, 7, 15):
        for layer, plan in tables.plans(rank).items():
            label = f"rank {rank}, layer {layer}"
            if slice_bytes == BATCH_SLICE_BYTES:
                assert batched_copies(plan) == batched_copies(plan, slice_bytes=slice_bytes)
            dst_ptrs, src_ptrs, sizes = batched_copies(plan, slice_bytes=slice_bytes)
            assert len(dst_ptrs) == len(src_ptrs) == len(sizes), label
            assert all(0 < size <= slice_bytes for size in sizes), label
            pieces = list(zip(dst_ptrs, src_ptrs, sizes))
            assert _byte_mapping(pieces) == _byte_mapping(plan.descriptors), label
            # Each descriptor in order, cut into slice_bytes pieces; one piece per row per round.
            row_pieces = [
                [
                    (dst + offset, src + offset, min(slice_bytes, nbytes - offset))
                    for dst, src, nbytes in row
                    for offset in range(0, nbytes, slice_bytes)
                ]
                for row in plan.rows
            ]
            rounds = max(map(len, row_pieces))
            expected_rows = [
                k for j in range(rounds) for k, cut in enumerate(row_pieces) if len(cut) > j
            ]
            rows = _piece_rows(plan, dst_ptrs) if pieces else []
            assert rows == expected_rows, label
            for k, cut in enumerate(row_pieces):
                assert [piece for piece, row in zip(pieces, rows) if row == k] == cut, label


@pytest.mark.cpu_only
@pytest.mark.parametrize("slice_bytes", [0, -1])
def test_batched_copies_reject_bad_slices(slice_bytes):
    plan = _FakeTables(_case_layout("odd")).plans(0)[1]
    with pytest.raises(ValueError):
        batched_copies(plan, slice_bytes=slice_bytes)


@pytest.mark.cpu_only
@HOST_LAYOUTS
def test_plans_reassemble_every_blob_on_host(case):
    """Applied to host buffers at their real addresses, every plan rebuilds every blob."""
    layout = _case_layout(case)
    cluster = _Cluster(layout, CPU, seed=11)
    for rank in (0, 6, 15):
        for layer, plan in cluster.plans(rank).items():
            cluster.poison(layer)
            cluster.apply(plan)
            failures, _ = cluster.check(layer, rank, f"{case}/rank {rank}/layer {layer}")
            assert not failures, "\n".join(failures)


# ---------------------------------------------------------------------------------------------
# CPU: transfer selection and the launcher
# ---------------------------------------------------------------------------------------------


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "use_gather, use_batched, expected",
    [
        (True, True, CopyMode.GATHER),
        (True, False, CopyMode.GATHER),
        (False, True, CopyMode.BATCHED),
        (False, False, CopyMode.PER_SLICE),
    ],
)
def test_select_copy_mode_follows_the_expert_precedence(use_gather, use_batched, expected):
    assert select_copy_mode(use_gather=use_gather, use_batched=use_batched) is expected


@pytest.mark.cpu_only
def test_gather_launcher_launches_each_prepared_plan(monkeypatch):
    """One gather per non-empty layer with the arguments built once; nothing for empty layers.

    Each gather runs ``fabric_gather.THREADS`` threads per block (``TRTLLM_DWDP_GATHER_THREADS``),
    set here to 128 rather than the default 1024.
    """
    layout = _case_layout("odd")
    plans = _FakeTables(layout).plans(3)
    calls = []
    threads = 128
    monkeypatch.setattr(fabric_gather, "THREADS", threads)
    monkeypatch.setattr(fabric_gather, "gather_op", lambda: lambda *args: calls.append(args))
    launcher = DenseCopyLauncher(plans, CopyMode.GATHER, device=CPU)
    assert launcher.mode is CopyMode.GATHER and calls == []
    live = [layer for layer, plan in plans.items() if not plan.is_empty]
    assert len(live) == len(plans) - 1
    for _ in range(2):
        for layer in layout.layers:
            launcher(layer)
    assert len(calls) == 2 * len(live)
    for layer, first, second in zip(live, calls, calls[len(live) :]):
        assert all(a is b for a, b in zip(first, second)), f"layer {layer} was rebuilt"
        expected = gather_launch_args(plans[layer], CPU)
        assert all(torch.equal(got, want) for got, want in zip(first[:4], expected[:4]))
        assert first[4:] == (expected[4], threads)
    with pytest.raises(KeyError):
        launcher(3)
    launcher.release()
    launcher.release()
    with pytest.raises(RuntimeError, match="released"):
        launcher(live[0])
    assert len(calls) == 2 * len(live)


@pytest.mark.cpu_only
def test_gather_launcher_needs_the_op_and_aligned_plans(monkeypatch):
    plans = _FakeTables(_case_layout("odd")).plans(0)
    monkeypatch.setattr(fabric_gather, "gather_op", lambda: None)
    with pytest.raises(DenseSetupError, match="dwdp_fabric_gather"):
        DenseCopyLauncher(plans, CopyMode.GATHER, device=CPU)
    monkeypatch.setattr(fabric_gather, "gather_op", lambda: lambda *args: None)
    misaligned = LayerCopyPlan(
        layer=1, rank=0, peers=(1,), rows=((CopyDescriptor(ROW_ALIGN, ROW_ALIGN + 8, 4096),),)
    )
    with pytest.raises(DenseSetupError, match="no copy-engine fallback"):
        DenseCopyLauncher({**plans, 1: misaligned}, CopyMode.GATHER, device=CPU)


@pytest.mark.cpu_only
def test_launcher_rejects_misfiled_plans_and_unknown_modes():
    plans = _FakeTables(_case_layout("odd")).plans(0)
    with pytest.raises(ValueError, match="keyed by layer 2"):
        DenseCopyLauncher({2: plans[1]}, CopyMode.BATCHED, device=CPU)
    with pytest.raises(ValueError):
        DenseCopyLauncher(plans, "dma", device=CPU)


@pytest.mark.cpu_only
def test_batched_launcher_copies_on_the_current_stream(monkeypatch):
    """One ``cudaMemcpyBatchAsync`` of the prepared pieces, on the stream current at launch."""
    plans = _FakeTables(_case_layout("odd")).plans(11)
    cudart = _FakeCudart()
    monkeypatch.setattr(dense_plan, "_load_cudart", lambda: cudart)
    current = SimpleNamespace(cuda_stream=0x1111)
    devices = []
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda device=None: devices.append(device) or current
    )
    launcher = DenseCopyLauncher(plans, CopyMode.BATCHED, device=CPU)
    assert launcher.mode is CopyMode.BATCHED and cudart.calls == []
    launcher(1)
    current.cuda_stream = 0x2222
    launcher(1)
    launcher(7)
    assert len(cudart.calls) == 2 and devices == [CPU, CPU]
    copies = batched_copies(plans[1])
    order = _FakeCudart.cudaMemcpySrcAccessOrder.cudaMemcpySrcAccessOrderStream
    for call, handle in zip(cudart.calls, (0x1111, 0x2222)):
        dst_ptrs, src_ptrs, sizes, count, attrs, attr_indices, num_attrs, stream = call
        assert (dst_ptrs, src_ptrs, sizes) == copies and count == len(copies[0])
        assert len(attrs) == 1 and attrs[0].srcAccessOrder == order
        assert (attr_indices, num_attrs, stream) == ([0], 1, handle)
    cudart.result = 700
    with pytest.raises(RuntimeError, match="cudaMemcpyBatchAsync failed"):
        launcher(1)


@pytest.mark.cpu_only
def test_batched_launcher_needs_the_runtime_bindings(monkeypatch):
    def missing():
        raise ImportError("No module named 'cuda.bindings'")

    monkeypatch.setattr(dense_plan, "_load_cudart", missing)
    with pytest.raises(DenseSetupError, match="cuda.bindings.runtime"):
        DenseCopyLauncher(_FakeTables(_case_layout("odd")).plans(0), CopyMode.BATCHED, device=CPU)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "device, index",
    [(torch.device("cuda", 3), 3), (torch.device("cuda"), 2)],
    ids=["cuda:3", "cuda"],
)
def test_per_slice_launcher_reassembles_host_rows(device, index, monkeypatch):
    """Views on the launcher's device, made once; each launch copies every descriptor of the layer.

    A CUDA device without an index is the current device (2 here).  The views alias host buffers,
    so no GPU is needed.
    """
    layout = _case_layout("odd-chunked")
    cluster = _Cluster(layout, CPU, seed=5)
    made = []

    def tensor_from_ptr(ptr, shape, dtype, device_id):
        made.append((dtype, device_id))
        return cluster.tensor_from_ptr(ptr, shape, dtype, device_id)

    monkeypatch.setattr(vmm, "tensor_from_ptr", tensor_from_ptr)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    rank = 6
    plans = cluster.plans(rank)
    launcher = DenseCopyLauncher(plans, CopyMode.PER_SLICE, device=device)
    views = 2 * sum(len(plan.descriptors) for plan in plans.values())
    assert launcher.mode is CopyMode.PER_SLICE
    assert made == [(torch.uint8, index)] * views
    for layer in layout.layers:
        cluster.poison(layer)
        launcher(layer)
        failures, _ = cluster.check(layer, rank, f"per-slice/layer {layer}")
        assert not failures, "\n".join(failures)
    assert len(made) == views


@pytest.mark.cpu_only
def test_per_slice_launcher_needs_views_that_alias_the_memory(monkeypatch):
    plans = _FakeTables(_case_layout("odd")).plans(0)
    monkeypatch.setattr(
        vmm, "tensor_from_ptr", lambda ptr, shape, dtype, device_id: torch.zeros(shape, dtype=dtype)
    )
    with pytest.raises(DenseSetupError, match="alias"):
        DenseCopyLauncher(plans, CopyMode.PER_SLICE, device=CPU)

    def refuse(ptr, shape, dtype, device_id):
        raise ValueError("Cannot create tensor from null pointer")

    monkeypatch.setattr(vmm, "tensor_from_ptr", refuse)
    with pytest.raises(DenseSetupError, match="cannot view"):
        DenseCopyLauncher(plans, CopyMode.PER_SLICE, device=CPU)


# ---------------------------------------------------------------------------------------------
# GPU: the real transfers on one GPU (parent side)
# ---------------------------------------------------------------------------------------------

_KERNEL_ENV = "TRTLLM_DWDP_GATHER_KERNEL"
# Gather knobs and the compute-side defaults; children start without any of them.
_PROCESS_KNOBS = (
    _KERNEL_ENV,
    "TRTLLM_DWDP_GATHER_CLUSTER",
    "TRTLLM_DWDP_PREFETCH_GATHER",
    "TRTLLM_DWDP_GATHER_THREADS",
    _BLOCKS_ENV,
    "TRTLLM_DWDP_GATHER_LIB",
    "TRTLLM_CUTEDSL_SM_RESERVE",
    "TRTLLM_CUTEDSL_DENSE_GEMM_SCHEDULER",
)
# Importing TensorRT-LLM initializes MPI: a child that kept an enclosing launcher's rank variables
# would join that job, so every child runs as a standalone MPI singleton.
_LAUNCHER_ENV_PREFIXES = ("SLURM_", "PMIX_", "PMI_", "OMPI_", "PRTE_", "I_MPI_", "HYDRA_", "MPI_")
KERNELS = ("register", "tma")
_GATHER_KERNELS = {"register": "dwdpFabricGatherKernel", "tma": "dwdpFabricGatherTmaKernel"}
_GATHER_PREFIX = "dwdpFabricGather"
_WORKER_TIMEOUT_S = 600.0

# GPU layouts: K3's KDA and MLA layers at full size, then the odd layers, in one store chunk; the
# odd layers alone in 1 MiB store chunks.
GPU_CASES = {
    "k3": (
        {
            1: (K3_KDA_ATTN, K3_MOE),
            2: (K3_MLA_ATTN, K3_MOE),
            **{layer + 2: blobs for layer, blobs in ODD_BLOBS.items()},
        },
        0,
    ),
    "odd-chunked": (ODD_BLOBS, 1),
}
GPU_RANKS = (0, 5, 15)
# Whose K3 gather is profiled: one gather kernel for the KDA layer, none for the empty layer.
GPU_PROFILED_RANK = 5


def _gpu_comparisons() -> int:
    """Segments the GPU worker compares: layers x segments x ranks x transfers."""
    layers = sum(len(blobs) for blobs, _ in GPU_CASES.values())
    return layers * len(SEGMENT_ORDER) * len(GPU_RANKS) * len(CopyMode)


def _require_gpu() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")


def _require_gather_op() -> None:
    try:
        registered = torch.ops.trtllm.dwdp_fabric_gather is not None
    except (AttributeError, RuntimeError):
        registered = False
    if not registered:
        pytest.skip("trtllm::dwdp_fabric_gather is not registered by this TensorRT-LLM build")


def _child_env(kernel: str) -> dict:
    env = {
        name: value
        for name, value in os.environ.items()
        if name not in _PROCESS_KNOBS and not name.startswith(_LAUNCHER_ENV_PREFIXES)
    }
    if kernel == "tma":
        env[_KERNEL_ENV] = "tma"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _tail(text, limit: int = 4000) -> str:
    if text is None:
        return ""
    if isinstance(text, bytes):
        text = text.decode(errors="replace")
    return text[-limit:]


def _run_worker(tmp_path: Path, kernel: str) -> dict:
    """Run the GPU worker for ``kernel`` in a child process and return its report."""
    out = tmp_path / f"dense_plan_{kernel}.json"
    cmd = [sys.executable, os.path.abspath(__file__), "--kernel", kernel, "--out", str(out)]
    name = f"dense copy plan worker [{kernel}]"
    try:
        proc = subprocess.run(
            cmd,
            env=_child_env(kernel),
            cwd=str(tmp_path),
            capture_output=True,
            text=True,
            errors="replace",
            timeout=_WORKER_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(
            f"{name}: still running after {_WORKER_TIMEOUT_S:.0f} s and killed\n"
            f"--- stdout ---\n{_tail(exc.stdout)}\n--- stderr ---\n{_tail(exc.stderr)}"
        )
    if not out.is_file():
        pytest.fail(
            f"{name}: exited with code {proc.returncode} without a report\n"
            f"--- stdout ---\n{_tail(proc.stdout)}\n--- stderr ---\n{_tail(proc.stderr)}"
        )
    report = json.loads(out.read_text())
    for note in report["notes"]:
        warnings.warn(f"{name}: {note}", UserWarning, stacklevel=2)
    failures = report["failures"]
    if failures:
        shown = "\n".join(f"- {failure}" for failure in failures[:40])
        more = f"\n... and {len(failures) - 40} more" if len(failures) > 40 else ""
        pytest.fail(
            f"{name}: {len(failures)} failure(s)\n{shown}{more}\n"
            f"--- worker stderr ---\n{_tail(proc.stderr)}"
        )
    if report["skip"]:
        pytest.skip(f"{name}: {report['skip']}")
    if proc.returncode != 0:
        pytest.fail(f"{name}: exit code {proc.returncode}\n{_tail(proc.stderr)}")
    return report


@pytest.mark.parametrize("kernel", KERNELS)
def test_plan_reassembles_rows_on_gpu(kernel, tmp_path):
    """Every transfer of every plan reproduces every rank's rows byte-exact on one GPU.

    The 16 ranks' stores are local buffers.  The plans of ranks 0, 5 and 15 run through
    ``DenseCopyLauncher`` with the real ``trtllm::dwdp_fabric_gather`` (15 rows, 2 blocks per
    row), ``cudaMemcpyBatchAsync`` and per-slice copies, over K3-sized layers in one store chunk
    and the odd layers in 1 MiB chunks.  Each slot must hold the zero-padded blob and keep its
    poison elsewhere; a profiled K3 layer launches one gather kernel and no copy-engine work.
    """
    _require_gpu()
    _require_gather_op()
    report = _run_worker(tmp_path, kernel)
    info = report["info"]
    assert info["compared"] == _gpu_comparisons()
    assert info["gather_geometry"] == [[NUM_ROWS, 2]]


# ---------------------------------------------------------------------------------------------
# GPU: the worker (child side)
# ---------------------------------------------------------------------------------------------


class _Skip(Exception):
    """Raised in the worker when this machine or build cannot run the test."""


class _Report:
    """Failures, notes and facts the worker hands back to the parent."""

    def __init__(self) -> None:
        self.failures = []
        self.notes = []
        self.info = {}
        self.skip = None

    def check(self, ok: bool, message: str) -> bool:
        if not ok:
            self.failures.append(message)
        return ok

    def to_json(self) -> dict:
        return {
            "failures": self.failures,
            "notes": self.notes,
            "info": self.info,
            "skip": self.skip,
        }


def _int64(values, device: torch.device) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int64, device=device)


def _probe_launch(op, device: torch.device) -> None:
    """Gather one 16-byte line; skip when the build has no gather kernel for this GPU."""
    src = torch.arange(fabric_gather.LINE_BYTES, dtype=torch.uint8, device=device)
    dst = torch.zeros_like(src)
    try:
        op(
            _int64([dst.data_ptr()], device),
            _int64([src.data_ptr()], device),
            _int64([1], device),
            _int64([0, 1], device),
            1,
            32,
        )
        torch.cuda.synchronize(device)
    except RuntimeError as exc:
        if "no kernel image" in str(exc):
            raise _Skip(f"this build has no gather kernel for this GPU: {exc}") from exc
        raise
    if not torch.equal(dst, src):
        raise RuntimeError("a one-line probe gather did not copy its 16 bytes")


def _worker_setup(report: _Report, kernel: str) -> torch.device:
    """Check the device and the latched kernel, probe the op; return the device."""
    if not torch.cuda.is_available():
        raise _Skip("CUDA is not available")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    capability = torch.cuda.get_device_capability(device)
    report.info["device"] = torch.cuda.get_device_name(device)
    report.info["knobs"] = {name: os.environ.get(name) for name in _PROCESS_KNOBS}
    wanted = "tma" if kernel == "tma" else None
    if os.environ.get(_KERNEL_ENV) != wanted:
        raise RuntimeError(
            f"the worker for the {kernel} kernel started with "
            f"{_KERNEL_ENV}={os.environ.get(_KERNEL_ENV)!r}"
        )
    if capability < (9, 0):
        raise _Skip(f"the TMA kernel and cluster launches need sm_90+, found {capability}")
    op = fabric_gather.gather_op()
    if op is None:
        raise _Skip("trtllm::dwdp_fabric_gather is not registered by this TensorRT-LLM build")
    _probe_launch(op, device)
    return device


def _check_one_gather_per_layer(
    report: _Report, kernel: str, launcher, plans, copy_stream, device
) -> None:
    """Profile the K3 KDA layer and the empty layer: one gather kernel, no copy engine."""
    from torch.profiler import ProfilerActivity, profile

    empty = [layer for layer, plan in plans.items() if plan.is_empty]
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with torch.cuda.stream(copy_stream):
            launcher(1)
            for layer in empty:
                launcher(layer)
        torch.cuda.synchronize(device)
    names = [
        event.name for event in prof.events() if event.device_type == torch.autograd.DeviceType.CUDA
    ]
    report.info["profiled_device_events"] = names
    if not names:
        report.notes.append("torch.profiler recorded no device activity; launches unverified")
        return
    gathers = [name for name in names if _GATHER_PREFIX in name]
    expected = _GATHER_KERNELS[kernel]
    report.check(
        len(gathers) == 1 and expected in gathers[0],
        f"k3/rank {GPU_PROFILED_RANK}: layer 1 and the empty layers {empty} should launch one "
        f"{expected}; the profile has {names}",
    )
    copies = [name for name in names if "memcpy" in name.lower()]
    report.check(not copies, f"k3/rank {GPU_PROFILED_RANK}: the gather used the copy engine")


def _worker(report: _Report, kernel: str) -> None:
    device = _worker_setup(report, kernel)
    compute_stream = torch.cuda.current_stream(device)
    copy_stream = torch.cuda.Stream(device)
    compared = 0
    geometry = set()
    for seed, (case, (blobs, store_chunk_mb)) in enumerate(GPU_CASES.items()):
        layout = _layout(blobs, store_chunk_mb=store_chunk_mb)
        cluster = _Cluster(layout, device, seed=100 + seed)
        for rank in GPU_RANKS:
            plans = cluster.plans(rank)
            for plan in plans.values():
                if not plan.is_empty:
                    row_starts, blocks = gather_launch_args(plan, device)[3:]
                    geometry.add((row_starts.numel() - 1, blocks))
            for mode in CopyMode:
                # Launch arguments are built on the stream that launches them, as in setup P3.
                with torch.cuda.stream(copy_stream):
                    launcher = DenseCopyLauncher(plans, mode, device=device)
                for layer in layout.layers:
                    cluster.poison(layer)
                    copy_stream.wait_stream(compute_stream)
                    with torch.cuda.stream(copy_stream):
                        launcher(layer)
                    compute_stream.wait_stream(copy_stream)
                    torch.cuda.synchronize(device)
                    label = f"{case}/rank {rank}/{mode.value}/layer {layer}"
                    failures, count = cluster.check(layer, rank, label)
                    report.failures.extend(failures)
                    compared += count
                if case == "k3" and rank == GPU_PROFILED_RANK and mode is CopyMode.GATHER:
                    _check_one_gather_per_layer(
                        report, kernel, launcher, plans, copy_stream, device
                    )
                launcher.release()
        del cluster
        torch.cuda.empty_cache()
    report.info["compared"] = compared
    report.info["gather_geometry"] = sorted(geometry)


def _worker_main(argv) -> int:
    parser = argparse.ArgumentParser(description="dense DWDP copy plan GPU worker")
    parser.add_argument("--kernel", required=True, choices=KERNELS)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    report = _Report()
    try:
        _worker(report, args.kernel)
    except _Skip as exc:
        report.skip = str(exc)
    except Exception:
        # The worker's top level: any error becomes a reported failure with its traceback.
        report.failures.append(f"the worker raised:\n{traceback.format_exc()}")
    finally:
        out = Path(args.out)
        partial = out.with_suffix(".partial")
        partial.write_text(json.dumps(report.to_json(), indent=2, default=str))
        os.replace(partial, out)
    return 0


if __name__ == "__main__":
    sys.exit(_worker_main(sys.argv[1:]))
