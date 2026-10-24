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
"""Collection, checks and blob layout of the dense weights.

The collector walks the dense roots each decoder layer names, decides which storages move
(anchors), records every reference to them (bindings), places the anchors in one blob per
``(layer, segment)`` and hands the result to :func:`~.types.build_dense_layout` for rows, store
offsets and the digest.  Everything here is host logic over tensor metadata; the checksum
reductions run on whatever device the tensors live on (CPU in unit tests), and no function needs
CUDA.
"""

from __future__ import annotations

import weakref
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import nn

from .config import DenseConfig
from .types import (
    ALLOWED_DTYPES,
    CHECKSUM_MULTIPLIER,
    POOLED_SLACK_BYTES,
    ROW_ALIGN,
    ZERO_CHECKSUM,
    AnchorRecord,
    BindingKind,
    Checksum,
    DenseLayout,
    DenseLayoutError,
    DenseRootsProvider,
    RowPiece,
    SegmentKey,
    SegmentKind,
    SegmentLayout,
    TensorBinding,
    align_up,
    build_dense_layout,
    warn_or_raise,
)

#: Words one pass of :func:`tensor_checksum` reduces; bounds its int64 temporaries (32 MiB each).
_CHUNK_WORDS = 1 << 22

#: Words per partial sum.  A word times a 16-bit half of its position hash is below ``2**47`` in
#: magnitude, so a partial sum of ``2**15`` such products stays below ``2**62``: every sum the
#: checksum computes in int64 is exact, and the wrap-around happens only in Python integers.
_PARTIAL_WORDS = 1 << 15

_WORD_BYTES = 4
_HASH_MODULUS = 1 << 32
_HALF_BITS = 16
_HALF_MASK = (1 << _HALF_BITS) - 1
_U64_MASK = (1 << 64) - 1

#: Most findings one :class:`~.types.DenseLayoutError` lists.
_MAX_LISTED = 8

#: Tensor types a moved reference may have; any other subclass is rejected.
_MOVABLE_TYPES = (torch.Tensor, nn.Parameter)

_MODULE_KINDS = (BindingKind.PARAMETER, BindingKind.BUFFER)

_RootList = list[tuple[str, nn.Module]]


@dataclass(eq=False)
class _Reference:
    """A tensor reference on the dense device met by the walk of one ``(layer, segment)``."""

    layer: int
    module_path: str
    kind: BindingKind
    name: str
    index: int | None
    owner: nn.Module
    tensor: torch.Tensor
    storage: torch.UntypedStorage
    storage_key: int
    nbytes: int
    eligible: bool
    unknown: bool

    @property
    def path(self) -> str:
        return _reference_path(self.layer, self.module_path, self.name, self.index)

    def byte_range(self) -> tuple[int, int]:
        start = self.tensor.storage_offset() * self.tensor.element_size()
        return start, start + _span_bytes(self.tensor)


def collect_dense_layout(
    layers: Sequence[tuple[int, DenseRootsProvider]],
    config: DenseConfig,
    *,
    num_ranks: int,
    device: torch.device,
    attr_allow: frozenset[str],
    exclude: frozenset[str],
) -> DenseLayout:
    """Collect the anchors and bindings of every covered layer and lay them out (P1, no copies).

    Walk.  Layers in ascending index order; per layer the enabled segments in
    canonical order; per segment the roots ``layer.dense_dwdp_roots(segment.value)`` in the order
    returned; per root ``root.named_modules(prefix=root_path)`` in registration order, visiting a
    module reached twice within one ``(layer, segment)`` once.  Per module, in this order:
    ``_parameters`` in insertion order (``PARAMETER``), ``_buffers`` in insertion order
    (``BUFFER``), then ``vars(module)`` sorted by name: tensor values (``ATTRIBUTE``) and non-empty
    tuples or lists whose elements are all tensors (``SEQUENCE_ITEM``, one per element).  None
    entries are skipped.  Only tensors on ``device`` with a non-empty storage are references
    (a ``device`` without an index matches every index of its type); tensors on another device of
    the same type whose storage reaches the threshold are an error, and tensors on other device
    types are ignored.

    Anchors.  A storage becomes an anchor of ``(layer, segment)`` when it is
    reached through an eligible reference and ``0 < untyped_storage().nbytes()`` and
    ``nbytes >= config.min_bytes``.  Parameters and buffers are eligible unless their name is in
    ``exclude``; attributes and sequence items are eligible only if their name is in
    ``attr_allow`` and not in ``exclude``.  An attribute or sequence item named in neither table
    whose storage reaches the threshold but does not become an anchor through another reference
    stays in place with a warning (``config.strict``: :class:`~.types.DenseLayoutError`).

    Bindings.  Every reference under the segment's roots to an anchor storage becomes a
    :class:`~.types.TensorBinding` of that anchor, including names in ``exclude`` and attributes
    outside ``attr_allow``; bindings are unique by ``(owner, kind, name, index)``.  ``original``
    is the referenced tensor (``param.data`` for parameters) and ``source`` a flat ``uint8`` view
    of the whole storage.

    Left in place with a warning (``config.strict``: :class:`~.types.DenseLayoutError`): a
    storage referenced, through any reference, from more than one ``(layer, segment)``;
    a pooled storage, whose ``nbytes`` exceeds the union of its references' byte ranges by more
    than ``POOLED_SLACK_BYTES``.

    Errors regardless of ``config.strict`` (:class:`~.types.DenseLayoutError`): a
    binding whose owner has ``_weights_transformed`` False; a binding dtype outside
    ``ALLOWED_DTYPES``; a binding that is a tensor subclass other than ``nn.Parameter``; a
    binding whose dry-run repoint fails, i.e. ``(O_X + byte_offset)`` is not a multiple of its
    itemsize or ``byte_offset + span_bytes > nbytes``.  Also an error: a layer without
    ``dense_dwdp_roots``, a layer with roots for some enabled segments but not all, no covered
    layer, nothing moved.

    Layout.  Per ``(layer, segment)``, anchors in the order their first eligible
    reference is met, at offsets from :func:`assign_blob_offsets`; covered layers are those whose
    roots are non-empty.  Rows, store offsets, chunks, slot specs and the digest come from
    :func:`~.types.build_dense_layout` with ``config.digest_switches(num_ranks)``.  Checksums are
    not computed here (see :func:`compute_checksums`).

    Args:
        layers: ``(layer index, decoder layer)`` for every decoder layer, ascending; layers whose
            roots are [] (layer 0 of Kimi-K3) are not covered.
        config: Dense switches; ``segments``, ``min_bytes``, ``store_chunk_mb`` and ``strict``
            are used.
        num_ranks: DWDP group size ``N``.
        device: The device whose tensors move (``cuda:<id>`` in production, CPU in tests).
        attr_allow: Plain attribute names that may anchor a storage
            (the model's ``DWDP_DENSE_ATTR_ALLOW``).
        exclude: Names that never anchor a storage (the model's ``DWDP_DENSE_EXCLUDE``).

    Returns:
        The layout with anchors, bindings, ``warnings`` and ``digest``; checksums unset.

    Raises:
        DenseLayoutError: See above.
        ValueError: ``layers`` is not strictly ascending, or ``num_ranks`` is below 2.
    """
    switches = config.digest_switches(num_ranks)
    device = torch.device(device)
    kinds = switches.segments

    walks: dict[SegmentKey, list[_Reference]] = {}
    for layer, roots in _covered_roots(layers, kinds):
        for kind in kinds:
            walks[(layer, kind)] = _walk_segment(
                layer,
                roots[kind],
                device=device,
                min_bytes=config.min_bytes,
                attr_allow=attr_allow,
                exclude=exclude,
            )

    selected, warnings = _select_anchors(walks, min_bytes=config.min_bytes, strict=config.strict)

    blob_bytes: dict[SegmentKey, int] = {}
    anchors: dict[SegmentKey, list[AnchorRecord]] = {}
    errors: list[str] = []
    for key, groups in selected.items():
        anchors[key], blob_bytes[key] = _place_anchors(key, groups, errors)
    if errors:
        raise DenseLayoutError(_listing("dense weights cannot move", errors))
    if not any(anchors.values()):
        covered = sorted({layer for layer, _ in walks})
        raise DenseLayoutError(
            f"nothing to move: no storage of at least {config.min_bytes} bytes under the dense "
            f"roots of layers {covered} can move"
        )
    try:
        return build_dense_layout(blob_bytes, switches, anchors=anchors, warnings=warnings)
    except ValueError as exc:
        raise DenseLayoutError(f"inconsistent dense layout: {exc}") from exc


def assign_blob_offsets(nbytes: Sequence[int]) -> tuple[list[int], int]:
    """Place storages of the given sizes back to back in a blob.

    Args:
        nbytes: Storage sizes in blob order, each positive.

    Returns:
        ``(offsets, blob_bytes)``: offset ``i`` is ``align_up(end of storage i - 1, ROW_ALIGN)``
        (0 for the first); ``blob_bytes`` is the end of the last storage (0 for none).

    Raises:
        ValueError: A size is not positive.
    """
    offsets: list[int] = []
    end = 0
    for size in nbytes:
        if size <= 0:
            raise ValueError(f"blob members must have a positive size, got {size}")
        offset = align_up(end, ROW_ALIGN)
        offsets.append(offset)
        end = offset + size
    return offsets, end


def tensor_checksum(tensor: torch.Tensor, *, word_offset: int = 0) -> Checksum:
    """Checksum of a contiguous tensor's bytes.

    The bytes are zero-padded to whole int32 words; word ``i`` of the tensor is word
    ``word_offset + i`` of the checksum's origin.  The reduction runs on the tensor's device, in
    bounded chunks, and the result is copied to the host.

    Args:
        tensor: Contiguous tensor of any dtype; callers pass ``t.contiguous()`` for views.
        word_offset: Word index of the tensor's first byte relative to the origin (``O_X // 4``
            for an anchor inside its blob).

    Returns:
        The two sums, unsigned modulo ``2**64``.

    Raises:
        ValueError: ``tensor`` is not contiguous or ``word_offset`` is negative.
    """
    if word_offset < 0:
        raise ValueError(f"word_offset must be non-negative, got {word_offset}")
    if not tensor.is_contiguous():
        raise ValueError("tensor_checksum needs a contiguous tensor; pass t.contiguous()")
    flat = tensor.detach().reshape(-1)
    if flat.numel() == 0:
        return ZERO_CHECKSUM
    data = flat.view(torch.uint8)
    chunk_bytes = _CHUNK_WORDS * _WORD_BYTES
    word_sum = 0
    weighted_sum = 0
    for start in range(0, data.numel(), chunk_bytes):
        words, weighted = _chunk_sums(
            data[start : start + chunk_bytes], word_offset + start // _WORD_BYTES
        )
        word_sum += words
        weighted_sum += weighted
    return Checksum(word_sum & _U64_MASK, weighted_sum & _U64_MASK)


def device_checksum(tensor: torch.Tensor, *, word_offset: int = 0) -> torch.Tensor:
    """:func:`tensor_checksum` of a contiguous tensor, left on the tensor's device.

    The same two sums over the same bounded chunks, but nothing is copied to the host: the
    reductions are enqueued on the current stream and the host is never synchronized.  Device
    integer arithmetic wraps modulo ``2**64``, so each sum is the int64 with the 64 bits of the
    unsigned sum whatever the reduction order; :func:`checksum_bits` gives the same pair for a
    :class:`~.types.Checksum`.

    Args:
        tensor: Contiguous tensor of any dtype.
        word_offset: Word index of the tensor's first byte relative to the origin.

    Returns:
        ``int64 [2]`` on the tensor's device: ``word_sum`` and ``weighted_sum``.

    Raises:
        ValueError: ``tensor`` is not contiguous or ``word_offset`` is negative.
    """
    if word_offset < 0:
        raise ValueError(f"word_offset must be non-negative, got {word_offset}")
    if not tensor.is_contiguous():
        raise ValueError("device_checksum needs a contiguous tensor; pass t.contiguous()")
    data = tensor.detach().reshape(-1).view(torch.uint8)
    sums = torch.zeros(2, dtype=torch.int64, device=data.device)
    chunk_bytes = _CHUNK_WORDS * _WORD_BYTES
    for start in range(0, data.numel(), chunk_bytes):
        words = _words(data[start : start + chunk_bytes])
        first_word = word_offset + start // _WORD_BYTES
        # h_i = ((first_word + i) * M) mod 2**32; i * M stays below 2**55, so it is exact, and
        # w_i * h_i stays below 2**63 in magnitude.
        hashes = torch.arange(words.numel(), dtype=torch.int64, device=words.device)
        hashes.mul_(CHECKSUM_MULTIPLIER).add_(first_word * CHECKSUM_MULTIPLIER % _HASH_MODULUS)
        hashes.bitwise_and_(_HASH_MODULUS - 1)
        sums[0].add_(words.sum(dtype=torch.int64))
        sums[1].add_(hashes.mul_(words).sum())
    return sums


def checksum_bits(checksum: Checksum) -> tuple[int, int]:
    """The sums of ``checksum`` as the int64 values with the same 64 bits (:func:`device_checksum`)."""
    return _signed_64(checksum.word_sum), _signed_64(checksum.weighted_sum)


def compute_checksums(layout: DenseLayout) -> None:
    """Fill the per-segment and per-binding checksums of a collected layout (P1).

    ``SegmentLayout.checksum`` is the merge of :func:`tensor_checksum` of every anchor's
    ``source`` at word offset ``blob_offset // 4``, i.e. the checksum of the zero-padded blob;
    ``TensorBinding.checksum`` is :func:`tensor_checksum` of ``original.contiguous()``.

    Raises:
        DenseLayoutError: An anchor's ``source`` or a binding's ``original`` was already dropped.
    """
    with torch.no_grad():
        for segment in layout.segments.values():
            total = ZERO_CHECKSUM
            for anchor in segment.anchors:
                if anchor.source is None:
                    raise DenseLayoutError(f"the source of {anchor.path} was already dropped")
                total = total.merge(
                    tensor_checksum(anchor.source, word_offset=anchor.blob_offset // _WORD_BYTES)
                )
                for binding in anchor.bindings:
                    if binding.original is None:
                        raise DenseLayoutError(
                            f"the original of layers.{anchor.layer}.{binding.qualified_name} "
                            f"was already dropped"
                        )
                    binding.checksum = tensor_checksum(binding.original.contiguous())
            segment.checksum = total


def row_pieces(segment: SegmentLayout, rank: int, *, num_ranks: int) -> list[RowPiece]:
    """The runs of ``rank``'s row that come from anchors.

    Row ``rank`` is blob bytes ``[rank * S, (rank + 1) * S)``; each anchor overlapping it gives
    one piece, in ascending ``row_offset``.  Bytes no piece covers are zero.

    Args:
        segment: A segment of a collected layout.
        rank: Row index.
        num_ranks: ``N``.

    Raises:
        ValueError: ``rank`` is outside ``[0, num_ranks)``, or ``N`` rows of the segment's width
            do not hold its blob.
    """
    if not 0 <= rank < num_ranks:
        raise ValueError(f"rank {rank} is outside [0, {num_ranks})")
    _check_rows(segment, num_ranks)
    row_start, row_end = segment.row_range(rank)
    pieces = []
    for index, anchor in enumerate(segment.anchors):
        start = max(row_start, anchor.blob_offset)
        end = min(row_end, anchor.blob_end)
        if start < end:
            pieces.append(
                RowPiece(
                    anchor_index=index,
                    anchor_offset=start - anchor.blob_offset,
                    row_offset=start - row_start,
                    nbytes=end - start,
                )
            )
    return pieces


def write_blob(segment: SegmentLayout, out: torch.Tensor, *, num_ranks: int) -> None:
    """Write the full blob (all ``N`` rows) of a collected segment into ``out``.

    ``out[:N * S]`` receives each anchor's ``source`` at its ``blob_offset`` and zeros elsewhere,
    which is exactly what a gather of ``L(layer)`` leaves in a slot; ``out[N * S:]`` is not
    touched.  Used by the loopback store, Phase V references and unit tests.

    Args:
        segment: A segment of a collected layout (anchors with ``source``).
        out: Flat ``uint8`` tensor with at least ``N * S`` elements, on any device.
        num_ranks: ``N``.

    Raises:
        DenseLayoutError: An anchor's ``source`` was already dropped.
        ValueError: ``out`` is not a flat ``uint8`` tensor or is too small, or ``N`` rows of the
            segment's width do not hold its blob.
    """
    if out.dtype != torch.uint8 or out.dim() != 1:
        raise ValueError(f"out must be a flat uint8 tensor, got {out.dtype} {tuple(out.shape)}")
    _check_rows(segment, num_ranks)
    total = num_ranks * segment.row_bytes
    if out.numel() < total:
        raise ValueError(f"out holds {out.numel()} bytes; the blob needs {total}")
    for anchor in segment.anchors:
        if anchor.source is None:
            raise DenseLayoutError(f"the source of {anchor.path} was already dropped")
    cursor = 0
    for anchor in segment.anchors:
        if anchor.blob_offset > cursor:
            out[cursor : anchor.blob_offset].zero_()
        out[anchor.blob_offset : anchor.blob_end].copy_(anchor.source)
        cursor = anchor.blob_end
    if total > cursor:
        out[cursor:total].zero_()


# ---------------------------------------------------------------------------
# Walk
# ---------------------------------------------------------------------------


def _covered_roots(
    layers: Sequence[tuple[int, DenseRootsProvider]], kinds: Sequence[SegmentKind]
) -> list[tuple[int, dict[SegmentKind, _RootList]]]:
    covered = []
    previous: int | None = None
    for layer, module in layers:
        if previous is not None and layer <= previous:
            raise ValueError(
                f"decoder layers must be strictly ascending; layer {layer} follows {previous}"
            )
        previous = layer
        roots_of = getattr(module, "dense_dwdp_roots", None)
        if not callable(roots_of):
            raise DenseLayoutError(
                f"decoder layer {layer} ({type(module).__name__}) does not provide dense_dwdp_roots"
            )
        roots = {kind: _checked_roots(layer, kind, roots_of(kind.value)) for kind in kinds}
        present = [kind.value for kind in kinds if roots[kind]]
        if not present:
            continue
        if len(present) != len(kinds):
            missing = [kind.value for kind in kinds if not roots[kind]]
            raise DenseLayoutError(
                f"decoder layer {layer} has dense roots for {present} but none for {missing}"
            )
        covered.append((layer, roots))
    if not covered:
        raise DenseLayoutError("no decoder layer has dense roots, so nothing can move")
    return covered


def _checked_roots(layer: int, kind: SegmentKind, roots: Sequence[object]) -> _RootList:
    checked = []
    for entry in roots:
        try:
            path, module = entry
        except (TypeError, ValueError):
            raise DenseLayoutError(
                f"decoder layer {layer} returned {entry!r} as a {kind.value} root; "
                f"expected a (path, nn.Module) pair"
            ) from None
        if not isinstance(path, str) or not isinstance(module, nn.Module):
            raise DenseLayoutError(
                f"decoder layer {layer} returned ({path!r}, {type(module).__name__}) as a "
                f"{kind.value} root; expected a (path, nn.Module) pair"
            )
        checked.append((path, module))
    return checked


def _walk_segment(
    layer: int,
    roots: _RootList,
    *,
    device: torch.device,
    min_bytes: int,
    attr_allow: frozenset[str],
    exclude: frozenset[str],
) -> list[_Reference]:
    references = []
    visited: set[int] = set()
    for root_path, root in roots:
        for module_path, module in root.named_modules(prefix=root_path):
            if id(module) in visited:
                continue
            visited.add(id(module))
            for kind, name, index, tensor in _module_tensors(module):
                if tensor.device.type != device.type:
                    continue
                storage = tensor.untyped_storage()
                nbytes = storage.nbytes()
                if device.index is not None and tensor.device.index != device.index:
                    if 0 < nbytes and nbytes >= min_bytes:
                        raise DenseLayoutError(
                            f"{_reference_path(layer, module_path, name, index)} ({nbytes}-byte "
                            f"storage) is on {tensor.device}, not on the dense device {device}"
                        )
                    continue
                if nbytes == 0:
                    continue
                module_kind = kind in _MODULE_KINDS
                references.append(
                    _Reference(
                        layer=layer,
                        module_path=module_path,
                        kind=kind,
                        name=name,
                        index=index,
                        owner=module,
                        tensor=tensor,
                        storage=storage,
                        storage_key=storage.data_ptr(),
                        nbytes=nbytes,
                        eligible=name not in exclude and (module_kind or name in attr_allow),
                        unknown=not module_kind and name not in attr_allow and name not in exclude,
                    )
                )
    return references


def _reference_path(layer: int, module_path: str, name: str, index: int | None) -> str:
    suffix = "" if index is None else f"[{index}]"
    return f"layers.{layer}.{module_path}.{name}{suffix}"


def _module_tensors(
    module: nn.Module,
) -> Iterator[tuple[BindingKind, str, int | None, torch.Tensor]]:
    for name, param in module._parameters.items():
        if param is not None:
            yield BindingKind.PARAMETER, name, None, param
    for name, buffer in module._buffers.items():
        if buffer is not None:
            yield BindingKind.BUFFER, name, None, buffer
    attributes = vars(module)
    for name in sorted(attributes):
        value = attributes[name]
        if isinstance(value, torch.Tensor):
            yield BindingKind.ATTRIBUTE, name, None, value
        elif (
            isinstance(value, (tuple, list))
            and value
            and all(isinstance(item, torch.Tensor) for item in value)
        ):
            for index, item in enumerate(value):
                yield BindingKind.SEQUENCE_ITEM, name, index, item


# ---------------------------------------------------------------------------
# Anchor selection and placement
# ---------------------------------------------------------------------------


def _select_anchors(
    walks: Mapping[SegmentKey, list[_Reference]], *, min_bytes: int, strict: bool
) -> tuple[dict[SegmentKey, list[list[_Reference]]], list[str]]:
    holders: dict[int, list[SegmentKey]] = {}
    for key, references in walks.items():
        for reference in references:
            keys = holders.setdefault(reference.storage_key, [])
            if key not in keys:
                keys.append(key)

    warnings: list[str] = []
    reported_shared: set[int] = set()
    selected: dict[SegmentKey, list[list[_Reference]]] = {}
    for key, references in walks.items():
        groups: dict[int, list[_Reference]] = {}
        candidates: dict[int, _Reference] = {}
        for reference in references:
            groups.setdefault(reference.storage_key, []).append(reference)
            if reference.eligible and reference.nbytes >= min_bytes:
                candidates.setdefault(reference.storage_key, reference)

        chosen = []
        for storage_key, first in candidates.items():
            sharing = holders[storage_key]
            if len(sharing) > 1:
                if storage_key not in reported_shared:
                    reported_shared.add(storage_key)
                    owners = ", ".join(f"layers.{layer}/{kind.value}" for layer, kind in sharing)
                    warnings.append(
                        warn_or_raise(
                            f"storage of {first.path} ({first.nbytes} bytes) is referenced from "
                            f"{owners}; it stays replicated",
                            strict=strict,
                            error=DenseLayoutError,
                        )
                    )
                continue
            covered = _covered_bytes(groups[storage_key])
            if first.nbytes - covered > POOLED_SLACK_BYTES:
                warnings.append(
                    warn_or_raise(
                        f"storage of {first.path} ({first.nbytes} bytes) looks pooled: its "
                        f"references cover only {covered} bytes; it stays replicated",
                        strict=strict,
                        error=DenseLayoutError,
                    )
                )
                continue
            chosen.append(groups[storage_key])

        anchored = {group[0].storage_key for group in chosen}
        for reference in references:
            if (
                reference.unknown
                and reference.nbytes >= min_bytes
                and reference.storage_key not in anchored
            ):
                warnings.append(
                    warn_or_raise(
                        f"{reference.path} ({reference.nbytes}-byte storage) is a plain attribute "
                        f"in neither the allow nor the exclude table; it stays in place",
                        strict=strict,
                        error=DenseLayoutError,
                    )
                )
        selected[key] = chosen
    return selected, warnings


def _place_anchors(
    key: SegmentKey, groups: list[list[_Reference]], errors: list[str]
) -> tuple[list[AnchorRecord], int]:
    layer, kind = key
    offsets, blob_bytes = assign_blob_offsets([group[0].nbytes for group in groups])
    anchors = []
    for index, (group, offset) in enumerate(zip(groups, offsets)):
        bindings = tuple(_binding(reference) for reference in group)
        for reference, binding in zip(group, bindings):
            errors.extend(_binding_errors(reference, binding, offset))
        anchors.append(
            AnchorRecord(
                layer=layer,
                segment=kind,
                index=index,
                nbytes=group[0].nbytes,
                blob_offset=offset,
                bindings=bindings,
                source=_storage_bytes(group[0].storage),
            )
        )
    return anchors, blob_bytes


def _binding(reference: _Reference) -> TensorBinding:
    tensor = reference.tensor
    return TensorBinding(
        module_path=reference.module_path,
        kind=reference.kind,
        name=reference.name,
        index=reference.index,
        dtype=tensor.dtype,
        size=tuple(tensor.shape),
        stride=tuple(tensor.stride()),
        byte_offset=tensor.storage_offset() * tensor.element_size(),
        owner_ref=weakref.ref(reference.owner),
        original=tensor.data if reference.kind is BindingKind.PARAMETER else tensor,
    )


def _binding_errors(reference: _Reference, binding: TensorBinding, blob_offset: int) -> list[str]:
    errors = []
    if type(reference.tensor) not in _MOVABLE_TYPES:
        errors.append(
            f"{reference.path} is a {type(reference.tensor).__name__}; only plain tensors and "
            f"nn.Parameter can move"
        )
    if binding.dtype not in ALLOWED_DTYPES:
        errors.append(f"{reference.path} has dtype {binding.dtype}, which dense DWDP cannot move")
    if not getattr(reference.owner, "_weights_transformed", True):
        errors.append(
            f"{reference.path}: its {type(reference.owner).__name__} has "
            f"_weights_transformed=False, so a later weight transform would run on slot memory"
        )
    start = blob_offset + binding.byte_offset
    if (
        start % binding.dtype.itemsize
        or binding.byte_offset + binding.span_bytes > reference.nbytes
    ):
        errors.append(
            f"{reference.path} cannot be repointed: a {binding.span_bytes}-byte view at blob "
            f"offset {start} of a {reference.nbytes}-byte storage"
        )
    return errors


def _storage_bytes(storage: torch.UntypedStorage) -> torch.Tensor:
    """Flat ``uint8 [nbytes]`` tensor over the whole of ``storage``."""
    return torch.empty(0, dtype=torch.uint8, device=storage.device).set_(storage)


def _span_bytes(tensor: torch.Tensor) -> int:
    if tensor.numel() == 0:
        return 0
    span = 1 + sum((dim - 1) * step for dim, step in zip(tensor.shape, tensor.stride()))
    return span * tensor.element_size()


def _covered_bytes(references: Sequence[_Reference]) -> int:
    covered = 0
    run_start = run_end = 0
    for start, end in sorted(reference.byte_range() for reference in references):
        if start >= end:
            continue
        if start > run_end:
            covered += run_end - run_start
            run_start, run_end = start, end
        else:
            run_end = max(run_end, end)
    return covered + run_end - run_start


def _check_rows(segment: SegmentLayout, num_ranks: int) -> None:
    if num_ranks <= 0 or num_ranks * segment.row_bytes < segment.blob_bytes:
        raise ValueError(
            f"{num_ranks} rows of {segment.row_bytes} bytes cannot hold the "
            f"{segment.blob_bytes}-byte blob of layer {segment.layer} ({segment.segment.value})"
        )


def _listing(headline: str, findings: Sequence[str]) -> str:
    listed = "; ".join(findings[:_MAX_LISTED])
    more = len(findings) - _MAX_LISTED
    return f"{headline}: {listed}" + (f"; and {more} more" if more > 0 else "")


# ---------------------------------------------------------------------------
# Checksums
# ---------------------------------------------------------------------------


def _signed_64(value: int) -> int:
    """The int64 whose 64 bits are those of the unsigned ``value`` (below ``2**64``)."""
    return value - (1 << 64) if value >= 1 << 63 else value


def _words(data: torch.Tensor) -> torch.Tensor:
    """A ``uint8`` run as int32 words, zero-padded to a whole word (copied only if it must be)."""
    nbytes = data.numel()
    if nbytes % _WORD_BYTES or data.storage_offset() % _WORD_BYTES:
        aligned = data.new_zeros(align_up(nbytes, _WORD_BYTES))
        aligned[:nbytes].copy_(data)
        data = aligned
    return data.view(torch.int32)


def _chunk_sums(data: torch.Tensor, first_word: int) -> tuple[int, int]:
    """Exact ``(sum w_i, sum w_i * h_i)`` of a ``uint8`` run whose first word is ``first_word``."""
    words = _words(data).to(torch.int64)
    count = words.numel()
    rows = -(-count // _PARTIAL_WORDS)
    if rows * _PARTIAL_WORDS != count:
        words = torch.cat((words, words.new_zeros(rows * _PARTIAL_WORDS - count)))
    # h_i = ((first_word + i) * M) mod 2**32; positions * M stays below 2**55, so it is exact.
    base = first_word * CHECKSUM_MULTIPLIER % _HASH_MODULUS
    positions = torch.arange(rows * _PARTIAL_WORDS, dtype=torch.int64, device=words.device)
    hashes = ((positions * CHECKSUM_MULTIPLIER + base) % _HASH_MODULUS).view(rows, _PARTIAL_WORDS)
    words = words.view(rows, _PARTIAL_WORDS)
    low = (words * (hashes & _HALF_MASK)).sum(dim=1)
    high = (words * (hashes >> _HALF_BITS)).sum(dim=1)
    sums = torch.cat((words.sum().reshape(1), low, high)).tolist()
    weighted = sum(sums[1 : 1 + rows]) + (sum(sums[1 + rows :]) << _HALF_BITS)
    return sums[0], weighted
