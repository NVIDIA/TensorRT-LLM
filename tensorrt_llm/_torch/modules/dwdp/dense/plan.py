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
"""Per-layer copy plans of dense DWDP and the launcher that runs them.

Addressing: rank ``p``'s row of ``(layer, segment)`` is at
``chunk_bases[p][chunk] + chunk_offset`` (``p == rank`` is this rank's own store) and lands in the
slot holding the layer at ``base + p * S``.  One plan per layer merges both segments: one row per
peer, each carrying the peer's ``A`` and ``D`` rows plus one slice of this rank's own ``A`` and
``D`` rows, so the whole layer is one gather launch with the same 15-row, 2-blocks-per-row,
cluster-2 geometry as the expert gather.

The builders are pure host logic over the layout and plain integer addresses (CPU-testable); only
:class:`DenseCopyLauncher` touches CUDA.
"""

from __future__ import annotations

import functools
import itertools
import operator
from collections.abc import Callable, Mapping, Sequence
from types import ModuleType
from typing import NamedTuple

import torch

from tensorrt_llm.logger import logger

from .. import fabric_gather
from .types import (
    BATCH_SLICE_BYTES,
    LOG_PREFIX,
    ROW_ALIGN,
    CopyDescriptor,
    CopyMode,
    DenseLayout,
    DenseSetupError,
    LayerCopyPlan,
    SegmentKind,
    SegmentLayout,
    align_up,
    ceil_div,
)

#: ``trtllm::dwdp_fabric_gather`` arguments of one layer, as ``fabric_gather.build_plan`` returns
#: them: device ``int64`` destination pointers, source pointers, 16-byte line counts and row starts,
#: plus the blocks per row.
GatherArgs = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]

#: One ``cudaMemcpyBatchAsync``: parallel destination, source and size lists of equal length.
BatchedCopies = tuple[list[int], list[int], list[int]]


def local_slices(row_bytes: int, num_slices: int) -> list[tuple[int, int]]:
    """Cut this rank's row into one slice per peer row.

    With ``q = align_up(ceil(S / num_slices), ROW_ALIGN)``, slice ``k`` is
    ``(k * q, max(0, min(q, S - k * q)))``; slices past the row end have length 0 and are skipped
    by the plan.

    Args:
        row_bytes: ``S``, a multiple of ``ROW_ALIGN``.
        num_slices: ``N - 1``, positive.

    Returns:
        ``num_slices`` ``(offset, length)`` pairs; the lengths sum to ``S``.

    Raises:
        ValueError: ``row_bytes`` is negative or not aligned, or ``num_slices`` is not positive.
    """
    if num_slices <= 0:
        raise ValueError(f"num_slices must be positive, got {num_slices}")
    if row_bytes < 0 or row_bytes % ROW_ALIGN:
        raise ValueError(
            f"row_bytes must be a non-negative multiple of {ROW_ALIGN}, got {row_bytes}"
        )
    width = align_up(ceil_div(row_bytes, num_slices), ROW_ALIGN)
    return [(k * width, max(0, min(width, row_bytes - k * width))) for k in range(num_slices)]


class _AddressTables(NamedTuple):
    """Validated address tables of one rank."""

    chunk_bases: tuple[tuple[int, ...], ...]
    """``[rank][chunk]`` store chunk addresses."""

    slot_bases: dict[SegmentKind, tuple[int, ...]]
    """``[segment][ring slot]`` slot base addresses, for the enabled segments."""


def _checked_address(value: int, name: str) -> int:
    """``value`` as an address: an integer, non-negative and ``ROW_ALIGN``-aligned."""
    try:
        address = operator.index(value)
    except TypeError:
        raise ValueError(f"{name} must be an integer address, got {value!r}") from None
    if address < 0 or address % ROW_ALIGN:
        raise ValueError(f"{name} must be a non-negative multiple of {ROW_ALIGN}, got {address:#x}")
    return address


def _check_tables(
    layout: DenseLayout,
    rank: int,
    chunk_bases: Sequence[Sequence[int]],
    slot_bases: Mapping[SegmentKind, Sequence[int]],
) -> _AddressTables:
    """Validate ``rank`` and the address tables against ``layout``.

    Raises:
        ValueError: ``rank`` outside ``[0, N)``; a table with the wrong number of ranks, chunks,
            segments or ring slots; an address that is not a non-negative multiple of
            ``ROW_ALIGN``; or two slots whose ``N * S_max`` byte ranges overlap.
    """
    num_ranks = layout.num_ranks
    if not 0 <= rank < num_ranks:
        raise ValueError(f"rank must be in [0, {num_ranks}), got {rank}")

    num_chunks = len(layout.chunks)
    if len(chunk_bases) != num_ranks:
        raise ValueError(
            f"chunk_bases needs one entry per rank ({num_ranks}), got {len(chunk_bases)}"
        )
    chunks = []
    for peer, bases in enumerate(chunk_bases):
        if len(bases) != num_chunks:
            raise ValueError(
                f"chunk_bases[{peer}] needs one address per store chunk ({num_chunks}), "
                f"got {len(bases)}"
            )
        chunks.append(
            tuple(
                _checked_address(base, f"chunk_bases[{peer}][{chunk}]")
                for chunk, base in enumerate(bases)
            )
        )

    by_kind = {SegmentKind(kind): bases for kind, bases in slot_bases.items()}
    if set(by_kind) != set(layout.segment_kinds):
        raise ValueError(
            f"slot_bases must hold exactly the enabled segments "
            f"{[kind.value for kind in layout.segment_kinds]}, got "
            f"{sorted(kind.value for kind in by_kind)}"
        )
    slots: dict[SegmentKind, tuple[int, ...]] = {}
    spans: list[tuple[int, int, str]] = []
    for kind in layout.segment_kinds:
        spec = layout.slot_specs[kind]
        bases = by_kind[kind]
        if len(bases) != spec.depth:
            raise ValueError(
                f"slot_bases[{kind.value}] needs one base per ring slot ({spec.depth}), "
                f"got {len(bases)}"
            )
        slots[kind] = tuple(
            _checked_address(base, f"slot_bases[{kind.value}][{index}]")
            for index, base in enumerate(bases)
        )
        spans.extend(
            (base, base + spec.valid_bytes, f"{kind.value}[{index}]")
            for index, base in enumerate(slots[kind])
        )
    # Sorted by start, two ranges overlap iff some neighbouring pair does.
    spans = sorted(span for span in spans if span[0] < span[1])
    for (_, end, first), (start, _, second) in zip(spans, spans[1:]):
        if start < end:
            raise ValueError(f"slots {first} and {second} overlap")
    return _AddressTables(chunk_bases=tuple(chunks), slot_bases=slots)


def _check_segment(layout: DenseLayout, segment: SegmentLayout) -> None:
    """The row width, chunk and chunk offset of a segment keep every plan address aligned.

    Raises:
        ValueError: ``S`` or ``chunk_offset`` is negative or not ``ROW_ALIGN``-aligned, the chunk
            does not exist, or ``N * S`` exceeds the segment's slot.
    """
    if (
        segment.row_bytes < 0
        or segment.row_bytes % ROW_ALIGN
        or segment.chunk_offset < 0
        or segment.chunk_offset % ROW_ALIGN
        or not 0 <= segment.chunk < len(layout.chunks)
        or layout.num_ranks * segment.row_bytes > layout.slot_specs[segment.segment].valid_bytes
    ):
        raise ValueError(
            f"segment ({segment.layer}, {segment.segment.value}) has S={segment.row_bytes}, "
            f"chunk {segment.chunk}, chunk_offset={segment.chunk_offset}; S and the offset must "
            f"be non-negative multiples of {ROW_ALIGN}, the chunk must exist and N * S must fit "
            f"the slot"
        )


def _layer_plan(
    layout: DenseLayout, layer: int, rank: int, tables: _AddressTables
) -> LayerCopyPlan:
    """:func:`build_layer_plan` of a covered layer over tables :func:`_check_tables` accepted."""
    peers = tuple(peer for peer in range(layout.num_ranks) if peer != rank)
    # Per enabled segment: its layout, the base of the slot holding ``layer`` and the local slices.
    targets = []
    for segment in layout.layer_segments(layer):
        _check_segment(layout, segment)
        base = tables.slot_bases[segment.segment][layout.slot_index(segment.segment, layer)]
        targets.append((segment, base, local_slices(segment.row_bytes, len(peers))))

    rows = []
    for k, peer in enumerate(peers):
        row = []
        for segment, base, _ in targets:
            if segment.row_bytes:
                row.append(
                    CopyDescriptor(
                        dst=base + peer * segment.row_bytes,
                        src=tables.chunk_bases[peer][segment.chunk] + segment.chunk_offset,
                        nbytes=segment.row_bytes,
                    )
                )
        for segment, base, slices in targets:
            offset, length = slices[k]
            if length:
                row.append(
                    CopyDescriptor(
                        dst=base + rank * segment.row_bytes + offset,
                        src=tables.chunk_bases[rank][segment.chunk] + segment.chunk_offset + offset,
                        nbytes=length,
                    )
                )
        rows.append(tuple(row))
    return LayerCopyPlan(layer=layer, rank=rank, peers=peers, rows=tuple(rows))


def build_layer_plan(
    layout: DenseLayout,
    layer: int,
    *,
    rank: int,
    chunk_bases: Sequence[Sequence[int]],
    slot_bases: Mapping[SegmentKind, Sequence[int]],
) -> LayerCopyPlan:
    """The gather of ``L(layer)`` on ``rank``.

    Peers ``p_0 < ... < p_{N-2}`` are the ranks other than ``rank``.  Row ``k`` holds, in this
    order and skipping zero-length ones:

    * ``A`` row of ``p_k``: ``base_A + p_k * S_A`` <- ``chunk_bases[p_k][c_A] + o_A``, ``S_A`` bytes;
    * ``D`` row of ``p_k``: ``base_D + p_k * S_D`` <- ``chunk_bases[p_k][c_D] + o_D``, ``S_D`` bytes;
    * slice ``k`` of the local ``A`` row: ``base_A + rank * S_A + off_k`` <-
      ``chunk_bases[rank][c_A] + o_A + off_k``, ``len_k`` bytes (:func:`local_slices`);
    * slice ``k`` of the local ``D`` row, likewise;

    where ``base_X = slot_bases[X][layout.slot_index(X, layer)]`` and ``c_X``/``o_X`` are the
    segment's ``chunk``/``chunk_offset``.  ``D`` entries exist only with the MoE-dense segment
    enabled.  Every address and length is a multiple of ``ROW_ALIGN``, the destinations cover
    ``[base_X, base_X + N * S_X)`` exactly once per segment, and the plan's shape does not depend
    on ``rank``.

    Args:
        layout: The layout every rank shares.
        layer: Covered layer ``n``.
        rank: This rank ``r``.
        chunk_bases: ``chunk_bases[p][c]``: address of chunk ``c`` of rank ``p``'s store as mapped
            on this rank; ``N`` entries of ``len(layout.chunks)`` addresses, each a multiple of
            ``ROW_ALIGN``.
        slot_bases: ``slot_bases[segment][i]``: ``SlotBuffer.base`` of ring slot ``i``, for every
            enabled segment.  The slots' ``N * S_max`` byte ranges must not overlap.

    Returns:
        The plan (never None; empty only if every enabled segment of ``layer`` has ``B == 0``).

    Raises:
        ValueError: ``layer`` not covered; ``rank`` outside ``[0, N)``; address tables of the
            wrong shape or alignment, or with overlapping slots; or a segment of ``layer`` whose
            ``S`` or chunk offset is negative or not ``ROW_ALIGN``-aligned, whose chunk does not
            exist, or whose ``N * S`` exceeds its slot.
    """
    if not layout.covers(layer):
        raise ValueError(f"layer {layer} is not covered by the dense layout")
    return _layer_plan(layout, layer, rank, _check_tables(layout, rank, chunk_bases, slot_bases))


def build_copy_plans(
    layout: DenseLayout,
    *,
    rank: int,
    chunk_bases: Sequence[Sequence[int]],
    slot_bases: Mapping[SegmentKind, Sequence[int]],
) -> dict[int, LayerCopyPlan]:
    """:func:`build_layer_plan` for every covered layer, keyed by layer.

    Raises:
        ValueError: As :func:`build_layer_plan`, for any covered layer.
    """
    tables = _check_tables(layout, rank, chunk_bases, slot_bases)
    return {layer: _layer_plan(layout, layer, rank, tables) for layer in layout.layers}


def gather_launch_args(plan: LayerCopyPlan, device: torch.device) -> GatherArgs:
    """``fabric_gather.build_plan`` of the plan's rows, uploaded to ``device``.

    One gather row per plan row (empty rows allowed), so ``N - 1 = 15`` rows give
    ``blocks_per_row == 2``.  The plan is never silently dropped: alignment that
    ``build_plan`` rejects is an error here, not a copy-engine fallback.  Build
    under the stream that later launches the gather (the copy stream), as the expert plans are.

    Raises:
        DenseSetupError: The plan is empty, or ``build_plan`` returned None.
    """
    if plan.is_empty:
        raise DenseSetupError(
            f"the dense copy plan of layer {plan.layer} is empty; nothing to gather"
        )
    args = fabric_gather.build_plan([list(row) for row in plan.rows], device)
    if args is None:
        raise DenseSetupError(
            f"fabric_gather.build_plan rejected the dense copy plan of layer {plan.layer}: a "
            f"descriptor is not {fabric_gather.LINE_BYTES}-byte aligned (dense DWDP has no "
            f"copy-engine fallback)"
        )
    return args


def batched_copies(plan: LayerCopyPlan, *, slice_bytes: int = BATCH_SLICE_BYTES) -> BatchedCopies:
    """The plan as one ``cudaMemcpyBatchAsync``.

    Each descriptor is cut into ``slice_bytes`` pieces; the pieces are then taken round-robin over
    the plan rows (one piece from every row that still has one, per round), so consecutive
    copies read from different sources.  The pieces cover the plan's bytes exactly.

    Raises:
        ValueError: ``slice_bytes`` is not positive.
    """
    if slice_bytes <= 0:
        raise ValueError(f"slice_bytes must be positive, got {slice_bytes}")
    row_pieces = [
        [
            (dst + offset, src + offset, min(slice_bytes, nbytes - offset))
            for dst, src, nbytes in row
            for offset in range(0, nbytes, slice_bytes)
        ]
        for row in plan.rows
    ]
    dst_ptrs: list[int] = []
    src_ptrs: list[int] = []
    sizes: list[int] = []
    for round_pieces in itertools.zip_longest(*row_pieces):
        for piece in round_pieces:
            if piece is not None:
                dst_ptrs.append(piece[0])
                src_ptrs.append(piece[1])
                sizes.append(piece[2])
    return dst_ptrs, src_ptrs, sizes


def select_copy_mode(*, use_gather: bool, use_batched: bool) -> CopyMode:
    """The dense transfer that follows the expert one: gather, else batched, else per-slice."""
    if use_gather:
        return CopyMode.GATHER
    if use_batched:
        return CopyMode.BATCHED
    return CopyMode.PER_SLICE


def _load_cudart() -> ModuleType:
    """``cuda.bindings.runtime``; imported on first use, as only the batched transfer needs it."""
    from cuda.bindings import runtime as cudart

    return cudart


def _memcpy_batch(
    cudart: ModuleType,
    attrs: list[object],
    device: torch.device,
    layer: int,
    copies: BatchedCopies,
) -> None:
    """One ``cudaMemcpyBatchAsync`` of ``copies`` on the current stream of ``device``."""
    dst_ptrs, src_ptrs, sizes = copies
    stream = torch.cuda.current_stream(device)
    (err,) = cudart.cudaMemcpyBatchAsync(
        dst_ptrs, src_ptrs, sizes, len(dst_ptrs), attrs, [0], 1, stream.cuda_stream
    )
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(
            f"cudaMemcpyBatchAsync failed for the dense DWDP copy of layer {layer} "
            f"({len(dst_ptrs)} copies): {err}"
        )


def _descriptor_views(
    plan: LayerCopyPlan, device_index: int
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """``(dst, src)`` ``uint8`` views of every descriptor, aliasing the plan's addresses.

    Raises:
        DenseSetupError: An address cannot be viewed, or its view does not alias it.
    """
    from .. import vmm

    views = []
    for descriptor in plan.descriptors:
        pair = []
        for address in (descriptor.dst, descriptor.src):
            try:
                view = vmm.tensor_from_ptr(address, (descriptor.nbytes,), torch.uint8, device_index)
            except ValueError as exc:
                raise DenseSetupError(
                    f"layer {plan.layer}: cannot view {descriptor.nbytes} bytes at {address:#x}: "
                    f"{exc}"
                ) from exc
            if view.data_ptr() != address:
                raise DenseSetupError(
                    f"layer {plan.layer}: the view of {address:#x} is a copy at "
                    f"{view.data_ptr():#x}; a per-slice copy needs views that alias the memory"
                )
            pair.append(view)
        views.append((pair[0], pair[1]))
    return tuple(views)


def _copy_views(views: tuple[tuple[torch.Tensor, torch.Tensor], ...]) -> None:
    """One ``copy_`` per ``(dst, src)`` pair, on the current stream."""
    for dst, src in views:
        dst.copy_(src)


class DenseCopyLauncher:
    """Runs the per-layer plans; the runtime's ``LayerLauncher``.

    Construction prepares everything a launch needs, so a launch only enqueues work on the
    current stream (the runtime makes the copy stream current):

    * ``GATHER``: :func:`gather_launch_args` per layer, built on the current stream (construct it
      under the copy stream); a launch is one ``trtllm::dwdp_fabric_gather`` with
      ``fabric_gather.THREADS``.
    * ``BATCHED``: :func:`batched_copies` per layer; a launch is one ``cudaMemcpyBatchAsync`` on
      the current stream.
    * ``PER_SLICE``: one ``uint8`` destination and source view per descriptor on ``device``
      (``vmm.tensor_from_ptr``); a launch is one ``copy_`` per descriptor.

    Layers whose plan is empty launch nothing.

    Args:
        plans: Plan per covered layer, from :func:`build_copy_plans`.
        mode: Transfer, from :func:`select_copy_mode`.
        device: The dense CUDA device; a CUDA device without an index is the current device.

    Raises:
        DenseSetupError: ``GATHER`` without a registered ``trtllm::dwdp_fabric_gather``, or a plan
            the chosen transfer cannot run.
        ValueError: ``mode`` is not a :class:`~.types.CopyMode`, or a plan is keyed by a layer
            other than its own.
    """

    def __init__(
        self, plans: Mapping[int, LayerCopyPlan], mode: CopyMode, *, device: torch.device
    ) -> None:
        self._mode = CopyMode(mode)
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self._device = device
        self._released = False
        for layer, plan in plans.items():
            if plan.layer != layer:
                raise ValueError(
                    f"the plan keyed by layer {layer} is the plan of layer {plan.layer}"
                )

        if self._mode is CopyMode.GATHER:
            op = fabric_gather.gather_op()
            if op is None:
                raise DenseSetupError(
                    "the dense gather needs trtllm::dwdp_fabric_gather, which this build does "
                    "not register"
                )
            threads = fabric_gather.THREADS

            def prepare(plan: LayerCopyPlan) -> Callable[[], None]:
                return functools.partial(op, *gather_launch_args(plan, device), threads)

        elif self._mode is CopyMode.BATCHED:
            try:
                cudart = _load_cudart()
            except ImportError as exc:
                raise DenseSetupError(
                    f"the batched dense copy needs cuda.bindings.runtime: {exc}"
                ) from exc
            attrs = cudart.cudaMemcpyAttributes()
            attrs.srcAccessOrder = cudart.cudaMemcpySrcAccessOrder.cudaMemcpySrcAccessOrderStream

            def prepare(plan: LayerCopyPlan) -> Callable[[], None]:
                return functools.partial(
                    _memcpy_batch, cudart, [attrs], device, plan.layer, batched_copies(plan)
                )

        else:
            device_index = device.index or 0

            def prepare(plan: LayerCopyPlan) -> Callable[[], None]:
                return functools.partial(_copy_views, _descriptor_views(plan, device_index))

        self._launches: dict[int, Callable[[], None] | None] = {
            layer: None if plan.is_empty else prepare(plan) for layer, plan in plans.items()
        }
        logger.debug(
            f"{LOG_PREFIX} copy launcher: mode={self._mode.value}, layers={len(plans)}, "
            f"descriptors={sum(len(plan.descriptors) for plan in plans.values())}, "
            f"bytes={sum(plan.nbytes for plan in plans.values())}"
        )

    @property
    def mode(self) -> CopyMode:
        """The transfer in use."""
        return self._mode

    def __call__(self, layer: int) -> None:
        """Enqueue the copy of ``L(layer)`` on the current stream.

        Raises:
            KeyError: ``layer`` has no plan.
            RuntimeError: The launcher was released, or ``cudaMemcpyBatchAsync`` failed.
        """
        if self._released:
            raise RuntimeError("the dense copy launcher was released")
        try:
            launch = self._launches[layer]
        except KeyError:
            raise KeyError(f"no dense copy plan for layer {layer}") from None
        if launch is not None:
            launch()

    def release(self) -> None:
        """Drop the prepared launch arguments and views; idempotent."""
        self._launches = {}
        self._released = True
