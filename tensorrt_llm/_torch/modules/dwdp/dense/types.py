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
"""Shared data model of dense DWDP.

Dense DWDP keeps 1/N of each MoE decoder layer's replicated dense weights on every rank of an
N-rank DWDP group and gathers the layer back into a small staging slot two layers ahead of its use.
Each layer contributes one blob per segment kind: attention ``A(l)`` and MoE-dense ``D(l)``.  A
blob is the concatenation of the layer's moved storages ("anchors"), viewed as ``uint8[N, S]``;
rank ``r`` stores row ``r``.

This module holds what more than one part of the implementation reads or writes:

* segment kinds and their slot rings (``A``: depth 2, ``D``: depth 3), copy modes and run modes,
* the per-tensor (:class:`TensorBinding`) and per-storage (:class:`AnchorRecord`) records,
* the per-(layer, segment) row layout, the store offsets and chunks (:class:`DenseLayout`),
* slot specifications and slot buffers,
* copy descriptors and per-layer copy plans,
* the CUDA seams the runtime calls, with their torch implementation (:class:`TorchCudaOps`),
* alignment constants, checksums and exceptions.

Nothing here calls CUDA at import time; only :class:`TorchCudaOps` and
:func:`allocate_slot_ring` with a CUDA device do, when called.
"""

from __future__ import annotations

import enum
import hashlib
import weakref
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import NamedTuple, Protocol

import torch
from torch import nn

from tensorrt_llm.logger import logger

from ...multi_stream_utils import do_multi_stream

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Alignment of anchor offsets inside a blob, of row widths ``S``, of the local-row slices of a
#: copy plan and of slot bases.  Covers the gather's 16-byte lines, every itemsize of a moved
#: dtype, the CuTe DSL operand alignment and the caching allocator's 512 bytes.
ROW_ALIGN = 4096

#: Size granularity of each FABRIC store handle.
STORE_ALIGN = 2 << 20

#: Slice size of the batched copy-engine transfer (one ``cudaMemcpyBatchAsync`` per layer).
BATCH_SLICE_BYTES = 2 << 20

#: A candidate storage larger than the union of its views' byte ranges by more than this is
#: treated as pooled and stays replicated.
POOLED_SLACK_BYTES = 1 << 20

#: Default of ``DenseConfig.min_bytes``: smaller storages stay replicated.
DEFAULT_MIN_BYTES = 1 << 20

#: Tag of a slot that holds no layer.
EMPTY_TAG = -1

#: Odd multiplier of the word-position hash of the weighted checksum.
CHECKSUM_MULTIPLIER = 2654435761

#: Dtypes a moved tensor may have.
ALLOWED_DTYPES = frozenset(
    {
        torch.float8_e4m3fn,
        torch.uint8,
        torch.bfloat16,
        torch.float16,
        torch.float32,
        torch.int32,
    }
)

#: Prefix of every dense DWDP log line.
LOG_PREFIX = "[DWDP Dense]"

_U64_MASK = (1 << 64) - 1

# ---------------------------------------------------------------------------
# Segment kinds, slot rings, binding kinds, copy modes
# ---------------------------------------------------------------------------


class SegmentKind(str, enum.Enum):
    """The part of a decoder layer a blob holds.

    The values are the strings the model's ``dense_dwdp_roots(segment)`` and
    ``DenseConfig.segments`` use.
    """

    ATTN = "attn"
    """Attention weights ``A(l)``, read between H1 ``wait_layer`` and H2 ``attention_done``."""

    MOE = "moe"
    """MoE-dense weights ``D(l)`` (router, shared expert, latent projections), read between H1 and
    H3 ``layer_done``."""


#: Canonical segment order: blob order inside a layer, store order, digest order.
SEGMENT_ORDER: tuple[SegmentKind, ...] = (SegmentKind.ATTN, SegmentKind.MOE)

#: Slot ring depth per segment.  ``A(l)`` is consumed at H2(l) and its slot is refilled with
#: ``A(l+2)`` right after, so two slots suffice; ``D(l)`` is consumed only at H3(l) while ``D(l+2)``
#: is enqueued at H2(l), so it needs three.
RING_DEPTH: Mapping[SegmentKind, int] = MappingProxyType({SegmentKind.ATTN: 2, SegmentKind.MOE: 3})


def ring_slot(segment: SegmentKind, position: int) -> int:
    """Slot of a layer in ``segment``'s ring.

    Args:
        segment: Segment kind.
        position: Index of the layer among the covered layers (ascending).  For Kimi-K3 the
            covered layers are 1..92, so ``position = l - 1``: layer ``l`` takes slot
            ``(l - 1) % 2`` of the attention ring and slot ``(l - 1) % 3`` of the MoE-dense ring.

    Returns:
        ``position % RING_DEPTH[segment]``.

    Raises:
        ValueError: ``position`` is negative.
    """
    if position < 0:
        raise ValueError(f"layer position must be non-negative, got {position}")
    return position % RING_DEPTH[segment]


class BindingKind(str, enum.Enum):
    """How a module holds a tensor; decides how repointing installs the slot view."""

    PARAMETER = "param"
    """An ``nn.Parameter`` in ``module._parameters``; repointed with ``param.data = view``."""

    BUFFER = "buffer"
    """A registered buffer; repointed with ``module._buffers[name] = view``."""

    ATTRIBUTE = "attr"
    """A plain tensor attribute; repointed with ``object.__setattr__(module, name, view)``."""

    SEQUENCE_ITEM = "tuple"
    """Element ``index`` of a tuple or list attribute whose elements are all tensors; repointed by
    rebuilding the container with the view in place and setting it again."""


class CopyMode(str, enum.Enum):
    """Transfer used to gather a layer into its slots.

    With expert DWDP it follows the expert DWDP transfer; standalone, ``DwdpConfig.copy_mode`` picks
    it, batched when unset.
    """

    GATHER = "gather"
    """One ``trtllm::dwdp_fabric_gather`` launch per layer (expert ``use_gather``)."""

    BATCHED = "batched"
    """One ``cudaMemcpyBatchAsync`` of ``BATCH_SLICE_BYTES`` slices per layer
    (expert ``use_batched_prefetch``)."""

    PER_SLICE = "per_slice"
    """One ``Tensor.copy_`` per copy descriptor."""


class DenseMode(str, enum.Enum):
    """How dense DWDP runs; ``gate.validate_dense_gate`` decides it from the launch."""

    ATTACHED = "attached"
    """With expert DWDP (``DwdpConfig.enable_experts``): the DWDP group, the expert DWDP copy stream
    and its transfer."""

    STANDALONE = "standalone"
    """On an attention-DP instance with ``tp_size > 1``, without expert DWDP: the tp group, the
    dense manager's own copy stream and ``DwdpConfig.copy_mode``."""


#: Key of one blob: ``(layer index, segment kind)``.
SegmentKey = tuple[int, SegmentKind]

# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------


def align_up(value: int, alignment: int) -> int:
    """Round a non-negative ``value`` up to a multiple of the power-of-two ``alignment``.

    Raises:
        ValueError: ``alignment`` is not a positive power of two, or ``value`` is negative.
    """
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError(f"alignment must be a positive power of 2, got {alignment}")
    if value < 0:
        raise ValueError(f"cannot align a negative value ({value})")
    return (value + alignment - 1) // alignment * alignment


def ceil_div(numerator: int, denominator: int) -> int:
    """``ceil(numerator / denominator)`` for a non-negative numerator and positive denominator."""
    if denominator <= 0 or numerator < 0:
        raise ValueError(
            f"ceil_div needs numerator >= 0 and denominator > 0, got {numerator}, {denominator}"
        )
    return -(-numerator // denominator)


def row_bytes_for(blob_bytes: int, num_ranks: int) -> int:
    """Row width of a blob: ``S = align_up(ceil(B / N), ROW_ALIGN)``."""
    return align_up(ceil_div(blob_bytes, num_ranks), ROW_ALIGN)


# ---------------------------------------------------------------------------
# Checksums
# ---------------------------------------------------------------------------


class Checksum(NamedTuple):
    """Two wrap-around sums over a byte range read as int32 words.

    With ``w_i`` the i-th word read as a signed little-endian int32 and ``h_i = (i *
    CHECKSUM_MULTIPLIER) mod 2**32``: ``word_sum = sum(w_i)`` and ``weighted_sum = sum(w_i * h_i)``,
    both modulo ``2**64`` and stored as unsigned ints.  ``i`` counts words from an origin the caller
    names (the blob start for per-segment sums, the tensor start for per-tensor sums), so sums of
    disjoint ranges combine with :meth:`merge` and do not depend on the reduction order.  A range
    that does not end on a word boundary is zero-padded to one.
    """

    word_sum: int
    weighted_sum: int

    def merge(self, other: Checksum) -> Checksum:
        """Checksum of the union of two disjoint ranges with the same word origin."""
        return Checksum(
            (self.word_sum + other.word_sum) & _U64_MASK,
            (self.weighted_sum + other.weighted_sum) & _U64_MASK,
        )


#: Checksum of an empty (or all-zero) range.
ZERO_CHECKSUM = Checksum(0, 0)

# ---------------------------------------------------------------------------
# Collection records
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class TensorBinding:
    """One reference through which a decoder layer reaches an anchor storage.

    Every reference under the layer's roots for the anchor's segment is recorded, including names
    in the model's exclude table, so repointing preserves every alias.  A reference is unique by
    ``(owner, kind, name, index)``.

    Attributes:
        module_path: Path of the owning module relative to the decoder layer, as produced by
            ``root.named_modules(prefix=root_name)``; e.g. ``"linear_attn.qkvg_proj"``.
        kind: How the owner holds the tensor.
        name: ``_parameters`` / ``_buffers`` key or attribute name on the owner.
        index: Element index for ``SEQUENCE_ITEM``; None for the other kinds.
        dtype: Dtype of the view the owner exposes.
        size: Size of that view.
        stride: Stride of that view, in elements.
        byte_offset: ``storage_offset() * element_size()``: where the view starts inside the anchor
            storage.
        owner_ref: Weak reference to the owning module.
        original: The tensor the owner exposed at collection (for ``PARAMETER``, ``param.data``).
            Held until commit for rollback and for the P2 to P5 checks; None after commit.
        checksum: Per-tensor checksum of ``original`` (made contiguous), from P1, for the P5
            self-check through the module attribute.
    """

    module_path: str
    kind: BindingKind
    name: str
    index: int | None
    dtype: torch.dtype
    size: tuple[int, ...]
    stride: tuple[int, ...]
    byte_offset: int
    owner_ref: weakref.ReferenceType[nn.Module]
    original: torch.Tensor | None = None
    checksum: Checksum | None = None

    @property
    def qualified_name(self) -> str:
        """``module_path.name`` plus ``[index]`` for sequence items; unique within a layer."""
        suffix = "" if self.index is None else f"[{self.index}]"
        return f"{self.module_path}.{self.name}{suffix}"

    @property
    def span_bytes(self) -> int:
        """Bytes from the view start to one past its last element; 0 for an empty view."""
        if any(dim == 0 for dim in self.size):
            return 0
        span = 1 + sum((dim - 1) * step for dim, step in zip(self.size, self.stride))
        return span * self.dtype.itemsize

    def owner(self) -> nn.Module:
        """The owning module.

        Raises:
            DenseDWDPError: The owner was garbage collected.
        """
        module = self.owner_ref()
        if module is None:
            raise DenseDWDPError(f"the module owning {self.qualified_name} was garbage collected")
        return module

    def digest_tuple(self) -> tuple[object, ...]:
        """``(path, kind, name, index, dtype, size, stride, byte_offset)`` in primitive types."""
        return (
            self.module_path,
            self.kind.value,
            self.name,
            self.index,
            str(self.dtype),
            tuple(self.size),
            tuple(self.stride),
            self.byte_offset,
        )


@dataclass(eq=False)
class AnchorRecord:
    """One moved storage and every binding of it.

    Attributes:
        layer: Decoder layer index.
        segment: Segment kind; an anchor belongs to exactly one ``(layer, segment)``.
        index: Position among the segment's anchors, in first-encounter order of the walk.
        nbytes: ``untyped_storage().nbytes()``; the whole storage moves, including bytes no view
            covers (e.g. the pad rows of ``_bfa_proj_weight``).
        blob_offset: ``O_X``, the anchor's byte offset in its blob; a multiple of ``ROW_ALIGN``.
        bindings: Every reference to the storage under the segment's roots, in walk order.
        source: Flat ``uint8 [nbytes]`` view of the original storage; the bytes P2 writes into the
            store rows and P4 compares against.  None after commit.
    """

    layer: int
    segment: SegmentKind
    index: int
    nbytes: int
    blob_offset: int
    bindings: tuple[TensorBinding, ...]
    source: torch.Tensor | None = None

    @property
    def blob_end(self) -> int:
        """One past the anchor's last blob byte."""
        return self.blob_offset + self.nbytes

    @property
    def path(self) -> str:
        """``layers.<layer>.<first binding>``: how logs and leak reports name the anchor."""
        first = self.bindings[0].qualified_name if self.bindings else "<unbound>"
        return f"layers.{self.layer}.{first}"

    def drop_originals(self) -> None:
        """Drop this record's strong references to the original storage (``source``, originals)."""
        self.source = None
        for binding in self.bindings:
            binding.original = None

    def digest_tuple(self) -> tuple[object, ...]:
        """``(idx, nbytes, O_X, bindings)`` in primitive types."""
        return (
            self.index,
            self.nbytes,
            self.blob_offset,
            tuple(binding.digest_tuple() for binding in self.bindings),
        )


class RowPiece(NamedTuple):
    """One contiguous run of a rank's row that comes from one anchor.

    The rest of the row is zero.
    """

    anchor_index: int
    """Index into ``SegmentLayout.anchors``."""

    anchor_offset: int
    """Byte offset inside the anchor storage (and ``AnchorRecord.source``)."""

    row_offset: int
    """Byte offset inside the row, ``0 <= row_offset < S``."""

    nbytes: int
    """Length of the run."""


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class SegmentLayout:
    """Blob, rows and store placement of one ``(layer, segment)``.

    Attributes:
        layer: Decoder layer index.
        segment: Segment kind.
        blob_bytes: ``B``: end of the last anchor; 0 if nothing of this segment moves.
        row_bytes: ``S = align_up(ceil(B / N), ROW_ALIGN)``; row ``p`` is blob bytes
            ``[p * S, (p + 1) * S)``, zero-padded past ``B``.
        store_offset: ``off``: offset of the rows in the logical store, the running sum of ``S``
            over ``(layer ascending, attn before moe)``; identical on every rank.
        chunk: Index of the store chunk holding the rows.
        chunk_offset: ``store_offset`` relative to the start of that chunk; rank ``p``'s row is at
            ``chunk_bases[p][chunk] + chunk_offset``.
        anchors: The anchors in blob order; empty for layouts built from blob sizes alone.
        checksum: Per-segment content checksum over the blob (word origin: blob start), from P1.
    """

    layer: int
    segment: SegmentKind
    blob_bytes: int
    row_bytes: int
    store_offset: int
    chunk: int
    chunk_offset: int
    anchors: tuple[AnchorRecord, ...] = ()
    checksum: Checksum | None = None

    @property
    def key(self) -> SegmentKey:
        """``(layer, segment)``."""
        return (self.layer, self.segment)

    @property
    def moved_bytes(self) -> int:
        """Sum of the anchors' storage sizes."""
        return sum(anchor.nbytes for anchor in self.anchors)

    def row_range(self, rank: int) -> tuple[int, int]:
        """Blob byte range ``[rank * S, (rank + 1) * S)`` of ``rank``'s row."""
        return rank * self.row_bytes, (rank + 1) * self.row_bytes

    def digest_tuple(self) -> tuple[object, ...]:
        """``(l, seg, B, S, off, chunk, anchors)`` in primitive types."""
        return (
            self.layer,
            self.segment.value,
            self.blob_bytes,
            self.row_bytes,
            self.store_offset,
            self.chunk,
            tuple(anchor.digest_tuple() for anchor in self.anchors),
        )


@dataclass(frozen=True)
class StoreChunk:
    """A run of whole layers whose rows share one store handle.

    ``DenseConfig.store_chunk_mb == 0`` gives a single chunk.

    Attributes:
        index: Chunk index, ascending with the layers.
        layers: The layers whose rows (all segments) live in this chunk.
        offset: Logical store offset of the chunk's first row.
        nbytes: Sum of the chunk's row widths.
    """

    index: int
    layers: tuple[int, ...]
    offset: int
    nbytes: int

    @property
    def handle_bytes(self) -> int:
        """Size of the chunk's handle on every rank: ``align_up(nbytes, STORE_ALIGN)``."""
        return align_up(self.nbytes, STORE_ALIGN)


@dataclass(frozen=True)
class SlotSpec:
    """Shape of one segment's slot ring.

    Attributes:
        segment: Segment kind.
        depth: Ring depth, ``RING_DEPTH[segment]``.
        row_bytes: ``S_max``: the largest row width of the segment over the covered layers.
        num_ranks: ``N``.
    """

    segment: SegmentKind
    depth: int
    row_bytes: int
    num_ranks: int

    @property
    def valid_bytes(self) -> int:
        """``N * S_max``: the largest blob a slot holds, starting at its aligned base."""
        return self.num_ranks * self.row_bytes

    @property
    def alloc_bytes(self) -> int:
        """Allocation size of one slot: ``valid_bytes`` plus ``ROW_ALIGN`` for base alignment."""
        return self.valid_bytes + ROW_ALIGN


@dataclass(frozen=True)
class DigestSwitches:
    """The switches the layout depends on; part of the cross-rank digest.

    Attributes:
        segments: Enabled segment kinds in canonical order; always starts with ``ATTN``.
        min_bytes: ``DenseConfig.min_bytes``.
        store_chunk_mb: ``DenseConfig.store_chunk_mb``.
        num_ranks: ``N``, the DWDP group size; at least 2.
        row_align: ``ROW_ALIGN``.
        store_align: ``STORE_ALIGN``.
        ring_depths: Ring depths of ``SEGMENT_ORDER``.
    """

    segments: tuple[SegmentKind, ...]
    min_bytes: int
    store_chunk_mb: int
    num_ranks: int
    row_align: int = ROW_ALIGN
    store_align: int = STORE_ALIGN
    ring_depths: tuple[int, ...] = tuple(RING_DEPTH[kind] for kind in SEGMENT_ORDER)

    def __post_init__(self) -> None:
        canonical = tuple(kind for kind in SEGMENT_ORDER if kind in self.segments)
        if not self.segments or tuple(self.segments) != canonical:
            raise ValueError(
                f"segments must be a non-empty subset of {[k.value for k in SEGMENT_ORDER]} "
                f"in that order, got {self.segments}"
            )
        if self.segments[0] is not SegmentKind.ATTN:
            raise ValueError("dense DWDP always includes the attention segment")
        if self.num_ranks < 2:
            raise ValueError(f"dense DWDP needs at least 2 ranks, got {self.num_ranks}")
        if self.min_bytes < 0 or self.store_chunk_mb < 0:
            raise ValueError(
                f"min_bytes and store_chunk_mb must be non-negative, got "
                f"{self.min_bytes} and {self.store_chunk_mb}"
            )

    def as_tuple(self) -> tuple[object, ...]:
        """The switch tuple in primitive types, in field order."""
        return (
            tuple(kind.value for kind in self.segments),
            self.min_bytes,
            self.store_chunk_mb,
            self.num_ranks,
            self.row_align,
            self.store_align,
            tuple(self.ring_depths),
        )


@dataclass(eq=False)
class DenseLayout:
    """Everything every rank must agree on, plus this rank's collection records.

    Built by :func:`build_dense_layout`; the collector (``layout.collect_dense_layout``) adds the
    anchors, tests can build it from blob sizes alone.  The plan, store, slots and repointing all
    read it; only the checksum fields and the anchors' originals change after it is built.

    Attributes:
        switches: Layout switches (segments, threshold, chunking, N, alignments, ring depths).
        layers: Covered decoder layers, ascending.  Every covered layer has an entry for every
            enabled segment.
        segments: ``(layer, segment) -> SegmentLayout`` in store order.
        chunks: Store chunks in store order.
        slot_specs: Ring shape per enabled segment.
        digest: sha256 hex digest of :meth:`digest_input`; equal on two ranks iff their layouts
            are.
        warnings: Non-fatal findings of the collector (storages left in place), in walk order.
    """

    switches: DigestSwitches
    layers: tuple[int, ...]
    segments: dict[SegmentKey, SegmentLayout]
    chunks: tuple[StoreChunk, ...]
    slot_specs: dict[SegmentKind, SlotSpec]
    digest: str = ""
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._positions = {layer: position for position, layer in enumerate(self.layers)}

    @property
    def num_ranks(self) -> int:
        """``N``."""
        return self.switches.num_ranks

    @property
    def segment_kinds(self) -> tuple[SegmentKind, ...]:
        """Enabled segment kinds in canonical order."""
        return self.switches.segments

    @property
    def moved_bytes(self) -> int:
        """Bytes of replicated storage this layout moves."""
        return sum(segment.moved_bytes for segment in self.segments.values())

    @property
    def store_bytes(self) -> int:
        """Bytes of store handles each rank allocates."""
        return sum(chunk.handle_bytes for chunk in self.chunks)

    @property
    def slot_bytes(self) -> int:
        """Bytes of slot allocations each rank makes."""
        return sum(spec.depth * spec.alloc_bytes for spec in self.slot_specs.values())

    def covers(self, layer: int) -> bool:
        """Whether ``layer`` is a covered layer."""
        return layer in self._positions

    def position(self, layer: int) -> int:
        """Index of ``layer`` among the covered layers.

        Raises:
            KeyError: ``layer`` is not covered.
        """
        try:
            return self._positions[layer]
        except KeyError:
            raise KeyError(f"layer {layer} is not covered by dense DWDP") from None

    def slot_index(self, segment: SegmentKind, layer: int) -> int:
        """Ring slot holding ``layer``'s ``segment`` blob."""
        return ring_slot(segment, self.position(layer))

    def segment(self, layer: int, segment: SegmentKind) -> SegmentLayout:
        """The ``(layer, segment)`` entry.

        Raises:
            KeyError: No such entry.
        """
        return self.segments[(layer, segment)]

    def layer_segments(self, layer: int) -> tuple[SegmentLayout, ...]:
        """``layer``'s entries in canonical segment order."""
        return tuple(self.segments[(layer, kind)] for kind in self.segment_kinds)

    def anchors(self) -> Iterator[AnchorRecord]:
        """Every anchor, in store order."""
        for segment in self.segments.values():
            yield from segment.anchors

    def checksums(self) -> tuple[Checksum, ...]:
        """Per-segment content checksums in store order: the #1 payload.

        Raises:
            DenseLayoutError: Checksums were not computed yet.
        """
        missing = [key for key, segment in self.segments.items() if segment.checksum is None]
        if missing:
            raise DenseLayoutError(f"segment checksums not computed for {missing[:4]}...")
        return tuple(segment.checksum for segment in self.segments.values())

    def digest_input(self) -> tuple[object, ...]:
        """What the digest covers: switches, every segment with its anchors and bindings, chunks."""
        return (
            self.switches.as_tuple(),
            tuple(segment.digest_tuple() for segment in self.segments.values()),
            tuple((chunk.index, chunk.layers, chunk.offset, chunk.nbytes) for chunk in self.chunks),
        )


def compute_layout_digest(layout: DenseLayout) -> str:
    """The sha256 hex digest of ``repr(layout.digest_input())``; stable, as it holds primitives."""
    return hashlib.sha256(repr(layout.digest_input()).encode("utf-8")).hexdigest()


def build_dense_layout(
    blob_bytes: Mapping[SegmentKey, int],
    switches: DigestSwitches,
    *,
    anchors: Mapping[SegmentKey, Sequence[AnchorRecord]] | None = None,
    warnings: Sequence[str] = (),
) -> DenseLayout:
    """Compute rows, store offsets, chunks, slot specs and the digest.

    ``S`` and ``off`` are as :class:`SegmentLayout` defines them.  Chunks hold whole layers: with
    ``store_chunk_mb == 0`` there is one chunk; otherwise layers are added in ascending order and
    a new chunk starts when the next layer's rows would push a non-empty chunk past
    ``store_chunk_mb`` MiB (a single larger layer gets a chunk of its own).

    Args:
        blob_bytes: ``B`` for every ``(covered layer, enabled segment)``; the covered layers are
            the layers that appear in the keys.
        switches: Layout switches.
        anchors: The collector's anchors per key, in blob order.  When given, every key's ``B``
            must equal its last anchor's end (0 without anchors) and the offsets must be
            ``ROW_ALIGN``-aligned, ascending and non-overlapping.  Omit it to build a layout from
            blob sizes alone (copy-plan and runtime tests).
        warnings: Collector findings to carry on the layout.

    Returns:
        The layout, with ``digest`` set and checksums unset.

    Raises:
        ValueError: Keys do not cover every enabled segment of every covered layer, a size is
            negative, or the anchors are inconsistent with ``blob_bytes``.
    """
    kinds = switches.segments
    sizes = {(layer, SegmentKind(kind)): size for (layer, kind), size in blob_bytes.items()}
    layers = tuple(sorted({layer for layer, _ in sizes}))
    if not layers:
        raise ValueError("a dense layout needs at least one covered layer")
    expected = {(layer, kind) for layer in layers for kind in kinds}
    if set(sizes) != expected:
        raise ValueError(
            f"blob sizes must cover every enabled segment of every covered layer; "
            f"missing {sorted(expected - set(sizes))[:4]}, "
            f"unexpected {sorted(set(sizes) - expected)[:4]}"
        )
    anchor_map = {
        (layer, SegmentKind(kind)): tuple(value) for (layer, kind), value in (anchors or {}).items()
    }
    unknown = set(anchor_map) - expected
    if unknown:
        raise ValueError(f"anchors for keys outside the layout: {sorted(unknown)[:4]}")

    num_ranks = switches.num_ranks
    rows: list[tuple[SegmentKey, int, int, int, tuple[AnchorRecord, ...]]] = []
    layer_bytes: dict[int, int] = {}
    offset = 0
    for layer in layers:
        for kind in kinds:
            key = (layer, kind)
            size = sizes[key]
            if size < 0:
                raise ValueError(f"blob size of {key} is negative ({size})")
            segment_anchors = anchor_map.get(key, ())
            if anchors is not None:
                _check_anchor_placement(key, size, segment_anchors)
            row = row_bytes_for(size, num_ranks)
            rows.append((key, size, row, offset, segment_anchors))
            layer_bytes[layer] = layer_bytes.get(layer, 0) + row
            offset += row

    chunk_layers = _assign_chunks(layers, layer_bytes, switches.store_chunk_mb << 20)
    chunk_of_layer = {layer: index for index, group in enumerate(chunk_layers) for layer in group}
    first_offset = {}
    for key, _, _, row_offset, _ in rows:
        first_offset.setdefault(chunk_of_layer[key[0]], row_offset)
    chunks = tuple(
        StoreChunk(
            index=index,
            layers=tuple(group),
            offset=first_offset[index],
            nbytes=sum(layer_bytes[layer] for layer in group),
        )
        for index, group in enumerate(chunk_layers)
    )

    segments: dict[SegmentKey, SegmentLayout] = {}
    for key, size, row, row_offset, segment_anchors in rows:
        chunk = chunk_of_layer[key[0]]
        segments[key] = SegmentLayout(
            layer=key[0],
            segment=key[1],
            blob_bytes=size,
            row_bytes=row,
            store_offset=row_offset,
            chunk=chunk,
            chunk_offset=row_offset - chunks[chunk].offset,
            anchors=segment_anchors,
        )

    slot_specs = {
        kind: SlotSpec(
            segment=kind,
            depth=RING_DEPTH[kind],
            row_bytes=max(segments[(layer, kind)].row_bytes for layer in layers),
            num_ranks=num_ranks,
        )
        for kind in kinds
    }
    layout = DenseLayout(
        switches=switches,
        layers=layers,
        segments=segments,
        chunks=chunks,
        slot_specs=slot_specs,
        warnings=list(warnings),
    )
    layout.digest = compute_layout_digest(layout)
    return layout


def _check_anchor_placement(
    key: SegmentKey, blob_bytes: int, anchors: Sequence[AnchorRecord]
) -> None:
    end = 0
    for position, anchor in enumerate(anchors):
        if (anchor.layer, anchor.segment) != key or anchor.index != position:
            raise ValueError(
                f"anchor {anchor.path} is recorded as ({anchor.layer}, {anchor.segment.value}) "
                f"#{anchor.index} but placed in {key} at #{position}"
            )
        if anchor.blob_offset % ROW_ALIGN or anchor.blob_offset < end or anchor.nbytes <= 0:
            raise ValueError(
                f"anchor {anchor.path} has offset {anchor.blob_offset} and {anchor.nbytes} bytes; "
                f"offsets must be {ROW_ALIGN}-aligned, ascending, non-overlapping and sizes positive"
            )
        end = anchor.blob_end
    if end != blob_bytes:
        raise ValueError(f"blob size of {key} is {blob_bytes} but its anchors end at {end}")


def _assign_chunks(
    layers: Sequence[int], layer_bytes: Mapping[int, int], limit: int
) -> list[list[int]]:
    groups: list[list[int]] = [[]]
    used = 0
    for layer in layers:
        size = layer_bytes[layer]
        if limit > 0 and used > 0 and used + size > limit:
            groups.append([])
            used = 0
        groups[-1].append(layer)
        used += size
    return groups


# ---------------------------------------------------------------------------
# Slots
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class SlotBuffer:
    """One staging slot of a segment ring.

    Blob byte ``b`` of the layer the slot currently holds lives at ``storage[pad + b]``; rows use
    that layer's own ``S``.  Module views made by repointing are ``as_strided`` views of
    ``storage``, so the views keep the allocation alive.

    Attributes:
        segment: Ring the slot belongs to.
        index: Position in the ring, ``0 .. RING_DEPTH[segment] - 1``.
        storage: Flat ``uint8 [SlotSpec.alloc_bytes]`` tensor with storage offset 0.
        pad: ``(-storage.data_ptr()) % ROW_ALIGN``.
    """

    segment: SegmentKind
    index: int
    storage: torch.Tensor
    pad: int

    @property
    def base(self) -> int:
        """``ROW_ALIGN``-aligned device address of blob byte 0."""
        return self.storage.data_ptr() + self.pad

    @property
    def capacity(self) -> int:
        """Bytes available from ``base`` to the end of the allocation."""
        return self.storage.numel() - self.pad

    def blob_view(self, nbytes: int) -> torch.Tensor:
        """``uint8 [nbytes]`` view of the first ``nbytes`` blob bytes.

        Raises:
            ValueError: ``nbytes`` exceeds ``capacity``.
        """
        if nbytes < 0 or nbytes > self.capacity:
            raise ValueError(
                f"slot {self.segment.value}[{self.index}] holds {self.capacity} bytes, asked {nbytes}"
            )
        return self.storage[self.pad : self.pad + nbytes]


def allocate_slot_ring(spec: SlotSpec, device: torch.device | str) -> tuple[SlotBuffer, ...]:
    """Allocate ``spec.depth`` zero-filled slots of ``spec.alloc_bytes`` bytes on ``device``.

    Slots are ordinary torch allocations (not cuMem), so the module views repointed into them can
    never dangle.  CPU devices work for unit tests.
    """
    slots = []
    for index in range(spec.depth):
        storage = torch.zeros(spec.alloc_bytes, dtype=torch.uint8, device=device)
        slots.append(
            SlotBuffer(
                segment=spec.segment,
                index=index,
                storage=storage,
                pad=(-storage.data_ptr()) % ROW_ALIGN,
            )
        )
    return tuple(slots)


# ---------------------------------------------------------------------------
# Copy plans
# ---------------------------------------------------------------------------


class CopyDescriptor(NamedTuple):
    """One copy of ``nbytes`` from device address ``src`` to ``dst``.

    A plain tuple, so lists of descriptors are valid ``fabric_gather.build_plan`` groups.
    """

    dst: int
    src: int
    nbytes: int


@dataclass(frozen=True)
class LayerCopyPlan:
    """Everything one gather of ``L(layer)`` copies on one rank.

    Row ``k`` serves peer ``peers[k]`` (ascending, this rank excluded) and holds, in this order and
    skipping empty ones: the ``A`` row of that peer, the ``D`` row of that peer, slice ``k`` of
    this rank's ``A`` row and slice ``k`` of this rank's ``D`` row.  Together the rows write every
    byte of ``[base_A, base_A + N * S_A)`` and ``[base_D, base_D + N * S_D)`` exactly once.

    Attributes:
        layer: Target layer ``n``.
        rank: This rank ``r``.
        peers: ``p_0 < ... < p_{N-2}``.
        rows: One descriptor tuple per peer, aligned with ``peers``.
    """

    layer: int
    rank: int
    peers: tuple[int, ...]
    rows: tuple[tuple[CopyDescriptor, ...], ...]

    @property
    def descriptors(self) -> tuple[CopyDescriptor, ...]:
        """All descriptors in row order."""
        return tuple(descriptor for row in self.rows for descriptor in row)

    @property
    def nbytes(self) -> int:
        """Bytes the plan copies."""
        return sum(descriptor.nbytes for descriptor in self.descriptors)

    @property
    def is_empty(self) -> bool:
        """True if the plan copies nothing (every enabled segment of the layer has ``B == 0``)."""
        return self.nbytes == 0


#: Enqueues the copy of ``L(layer)`` (all enabled segments) on the current CUDA stream.  The
#: runtime makes the copy stream current around the call.
LayerLauncher = Callable[[int], None]

# ---------------------------------------------------------------------------
# CUDA seams of the runtime
# ---------------------------------------------------------------------------


class EventLike(Protocol):
    """The part of ``torch.cuda.Event`` the runtime uses."""

    def record(self, stream: StreamLike | None = None) -> None:
        """Record the event on ``stream`` (the current stream if None)."""
        ...


class StreamLike(Protocol):
    """The part of ``torch.cuda.Stream`` the runtime uses."""

    def wait_event(self, event: EventLike) -> None:
        """Make future work on this stream wait for the event's most recent record."""
        ...


class DeviceOps(Protocol):
    """Every CUDA call the dense runtime makes; unit tests pass a recording fake."""

    def new_event(self) -> EventLike:
        """A new non-timing event."""
        ...

    def current_stream(self) -> StreamLike:
        """The current stream of the dense device."""
        ...

    def stream(self, stream: StreamLike) -> AbstractContextManager[object]:
        """Context that makes ``stream`` current (``torch.cuda.stream``)."""
        ...

    def synchronize(self) -> None:
        """Wait for all work on the dense device."""
        ...

    def is_current_stream_capturing(self) -> bool:
        """Whether the current stream is capturing a CUDA graph."""
        ...

    def do_multi_stream(self) -> bool:
        """Whether modules may fork work onto auxiliary streams (``multi_stream_utils``)."""
        ...

    def nvtx_range(self, message: str) -> AbstractContextManager[object]:
        """Context that wraps an NVTX range named ``message``."""
        ...


class TorchCudaOps:
    """:class:`DeviceOps` backed by ``torch.cuda``.

    Args:
        device: The dense device; None means the current device.
    """

    def __init__(self, device: torch.device | int | None = None) -> None:
        self._device = device

    def new_event(self) -> torch.cuda.Event:
        """A new non-timing ``torch.cuda.Event``."""
        return torch.cuda.Event()

    def current_stream(self) -> torch.cuda.Stream:
        """``torch.cuda.current_stream(device)``."""
        return torch.cuda.current_stream(self._device)

    def stream(self, stream: StreamLike) -> AbstractContextManager[object]:
        """``torch.cuda.stream(stream)``."""
        return torch.cuda.stream(stream)

    def synchronize(self) -> None:
        """``torch.cuda.synchronize(device)``."""
        torch.cuda.synchronize(self._device)

    def is_current_stream_capturing(self) -> bool:
        """``torch.cuda.is_current_stream_capturing()``."""
        return torch.cuda.is_current_stream_capturing()

    def do_multi_stream(self) -> bool:
        """``multi_stream_utils.do_multi_stream()``."""
        return do_multi_stream()

    def nvtx_range(self, message: str) -> AbstractContextManager[object]:
        """``torch.cuda.nvtx.range(message)``."""
        return torch.cuda.nvtx.range(message)


# ---------------------------------------------------------------------------
# Model seam
# ---------------------------------------------------------------------------


class DenseRootsProvider(Protocol):
    """A decoder layer that names its dense roots.

    ``modules/dwdp`` does not import model code; a layer opts in by implementing this method.
    """

    def dense_dwdp_roots(self, segment: str) -> list[tuple[str, nn.Module]]:
        """The ``(path, module)`` roots of ``segment``.

        Args:
            segment: ``"attn"`` or ``"moe"``.

        Returns:
            The roots in walk order; [] if the layer stays replicated.
        """
        ...


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DenseDWDPError(RuntimeError):
    """Base class of dense DWDP errors."""


class DenseLayoutError(DenseDWDPError):
    """Collection, its checks or the layout rejected the model."""


class DenseRepointError(DenseDWDPError):
    """Repointing, the self-check, rollback or the release check failed."""


class DenseProtocolError(DenseDWDPError):
    """A runtime hook ran out of protocol."""


class DenseSetupError(DenseDWDPError):
    """A setup phase failed, or the group decided to abandon dense DWDP."""


def warn_or_raise(
    message: str, *, strict: bool, error: type[DenseDWDPError] = DenseDWDPError
) -> str:
    """Report a finding that strict mode (``DenseConfig.strict``) turns into an error.

    Args:
        message: What happened, without the log prefix.
        strict: Raise instead of warning.
        error: Exception type to raise under ``strict``.

    Returns:
        ``message``, after logging it as a warning.

    Raises:
        DenseDWDPError: ``error(message)`` when ``strict``.
    """
    if strict:
        raise error(message)
    logger.warning(f"{LOG_PREFIX} {message}")
    return message
