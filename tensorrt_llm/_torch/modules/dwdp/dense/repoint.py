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
"""Repointing module tensors into the staging slots.

After repointing, every binding of an anchor is a view of the slot that holds its layer: layer
``l`` and the layers that share its slot (``l + 2`` for attention, ``l + 3`` for MoE-dense) see
the same addresses, and the runtime protocol guarantees the slot holds a layer's bytes whenever
that layer reads them.  Each binding is rebuilt from its own dtype, size, stride and byte offset,
so aliases stay aliases.  The original storages are released only by dropping references; nothing
here resizes a storage.
"""

from __future__ import annotations

import enum
import gc
import weakref
from collections.abc import Callable, Mapping, Sequence
from typing import NamedTuple

import torch
from torch import nn

from tensorrt_llm.logger import logger

from .layout import tensor_checksum
from .types import (
    LOG_PREFIX,
    RING_DEPTH,
    AnchorRecord,
    BindingKind,
    DenseLayout,
    DenseRepointError,
    SegmentKind,
    SlotBuffer,
    TensorBinding,
)

#: Most failed bindings one error message lists; also the leaked anchors that ``commit`` warns
#: about one by one.
_MAX_LISTED = 8


class BindingState(enum.Enum):
    """Where the modules' dense bindings point."""

    PENDING = "pending"
    """Collected; modules hold the original tensors."""

    REPOINTED = "repointed"
    """Modules hold slot views; the originals are still alive (P5)."""

    ROLLED_BACK = "rolled_back"
    """Modules hold the original tensors again."""

    COMMITTED = "committed"
    """Modules hold slot views; the references to the originals are dropped (P6)."""

    DETACHED = "detached"
    """Modules hold zero-size tensors (teardown)."""


def slot_view(binding: TensorBinding, anchor: AnchorRecord, slot: SlotBuffer) -> torch.Tensor:
    """The view of ``slot`` that replaces ``binding``.

    ``slot.storage.view(dtype).as_strided(size, stride, (pad + O_X + byte_offset) // itemsize)``,
    where ``pad`` is the slot's, ``O_X`` the anchor's blob offset and the rest the binding's.

    Raises:
        DenseRepointError: The byte offset is not a multiple of the itemsize or the view leaves
            the slot; the layout's dry run rules both out.
    """
    storage = slot.storage
    if storage.dtype != torch.uint8 or storage.dim() != 1 or storage.storage_offset() != 0:
        raise DenseRepointError(
            f"slot {slot.segment.value}[{slot.index}] must be a flat uint8 tensor at storage "
            f"offset 0, got {storage.dtype} {tuple(storage.shape)} at {storage.storage_offset()}"
        )
    where = f"layers.{anchor.layer}.{binding.qualified_name}"
    itemsize = binding.dtype.itemsize
    start = slot.pad + anchor.blob_offset + binding.byte_offset
    span = binding.span_bytes
    if start % itemsize:
        raise DenseRepointError(
            f"{where} would start at slot byte {start}, not a multiple of its itemsize {itemsize}"
        )
    if binding.byte_offset + span > anchor.nbytes or start + span > storage.numel():
        raise DenseRepointError(
            f"{where} ({span} bytes at anchor offset {binding.byte_offset}) leaves its "
            f"{anchor.nbytes}-byte anchor or the {storage.numel()}-byte slot "
            f"{slot.segment.value}[{slot.index}]"
        )
    usable = storage.numel() - storage.numel() % itemsize
    return (
        storage[:usable]
        .view(binding.dtype)
        .as_strided(binding.size, binding.stride, start // itemsize)
    )


def read_binding(binding: TensorBinding) -> torch.Tensor:
    """The tensor the owner exposes for ``binding`` now.

    ``PARAMETER``: ``owner._parameters[name]``; ``BUFFER``: ``owner._buffers[name]``;
    ``ATTRIBUTE``: ``vars(owner)[name]``; ``SEQUENCE_ITEM``: ``vars(owner)[name][index]``.

    Raises:
        DenseRepointError: The owner is gone or no longer holds the name.
    """
    owner = _owner(binding)
    if binding.kind is BindingKind.PARAMETER:
        value = owner._parameters.get(binding.name)
    elif binding.kind is BindingKind.BUFFER:
        value = owner._buffers.get(binding.name)
    else:
        value = vars(owner).get(binding.name)
        if binding.kind is BindingKind.SEQUENCE_ITEM:
            value = _item(value, binding.index)
    if not isinstance(value, torch.Tensor):
        raise _missing(owner, binding)
    return value


def install_binding(binding: TensorBinding, tensor: torch.Tensor) -> None:
    """Make the owner expose ``tensor`` for ``binding``.

    ``PARAMETER``: ``param.data = tensor`` (the ``nn.Parameter`` object stays);
    ``BUFFER``: ``owner._buffers[name] = tensor`` (stays registered, persistence unchanged);
    ``ATTRIBUTE``: ``object.__setattr__(owner, name, tensor)``;
    ``SEQUENCE_ITEM``: rebuild the tuple or list with ``tensor`` at ``index`` (same container
    type, other elements unchanged) and set it with ``object.__setattr__``.

    Raises:
        DenseRepointError: ``tensor.dtype`` differs from ``binding.dtype``, or the owner is gone
            or no longer holds the name.
    """
    if tensor.dtype != binding.dtype:
        raise DenseRepointError(
            f"cannot install a {tensor.dtype} tensor for {binding.qualified_name} ({binding.dtype})"
        )
    owner = _owner(binding)
    name = binding.name
    if binding.kind is BindingKind.PARAMETER:
        param = owner._parameters.get(name)
        if param is None:
            raise _missing(owner, binding)
        param.data = tensor
    elif binding.kind is BindingKind.BUFFER:
        if name not in owner._buffers:
            raise _missing(owner, binding)
        owner._buffers[name] = tensor
    elif binding.kind is BindingKind.ATTRIBUTE:
        if name not in vars(owner):
            raise _missing(owner, binding)
        object.__setattr__(owner, name, tensor)
    else:
        container = vars(owner).get(name)
        if _item(container, binding.index) is None:
            raise _missing(owner, binding)
        items = list(container)
        items[binding.index] = tensor
        object.__setattr__(owner, name, _rebuild(container, items))


class _Entry(NamedTuple):
    """One binding with the anchor it views and the slot that holds its layer."""

    binding: TensorBinding
    anchor: AnchorRecord
    slot: SlotBuffer


class DenseBindings:
    """Every dense binding of a layout and where it points: slots, originals or nothing.

    The state machine (:class:`BindingState`) is ``PENDING -> REPOINTED -> COMMITTED -> DETACHED``,
    with ``REPOINTED -> ROLLED_BACK`` (and back to ``REPOINTED``) before the commit point.  The
    layout keeps the originals alive until :meth:`commit`.

    Args:
        layout: A collected layout (anchors with ``source``, bindings with ``original`` and
            ``checksum``).
        slots: The slot rings, ``slots[segment][i]`` for ``i < RING_DEPTH[segment]``; a binding of
            layer ``l`` goes to ``slots[segment][layout.slot_index(segment, l)]``.

    Raises:
        ValueError: A ring of an enabled segment is missing, has the wrong depth, or holds a slot
            of another segment or position.
        DenseRepointError: A binding's ``original`` was already dropped.
    """

    def __init__(
        self, layout: DenseLayout, slots: Mapping[SegmentKind, Sequence[SlotBuffer]]
    ) -> None:
        rings = {SegmentKind(kind): tuple(ring) for kind, ring in slots.items()}
        for kind in layout.segment_kinds:
            ring = rings.get(kind, ())
            if len(ring) != RING_DEPTH[kind]:
                raise ValueError(
                    f"the {kind.value} ring needs {RING_DEPTH[kind]} slots, got {len(ring)}"
                )
            for position, slot in enumerate(ring):
                if slot.segment is not kind or slot.index != position:
                    raise ValueError(
                        f"slot {position} of the {kind.value} ring is "
                        f"{slot.segment.value}[{slot.index}]"
                    )
        entries = []
        for anchor in layout.anchors():
            slot = rings[anchor.segment][layout.slot_index(anchor.segment, anchor.layer)]
            for binding in anchor.bindings:
                if binding.original is None:
                    raise DenseRepointError(
                        f"the original of layers.{anchor.layer}.{binding.qualified_name} was "
                        f"already dropped"
                    )
                entries.append(_Entry(binding, anchor, slot))
        self._layout = layout
        self._entries = tuple(entries)
        self._by_layer = {layer: [] for layer in layout.layers}
        for entry in self._entries:
            self._by_layer[entry.anchor.layer].append(entry)
        self._state = BindingState.PENDING

    @property
    def state(self) -> BindingState:
        """Where the bindings point."""
        return self._state

    def repoint(self) -> None:
        """Install :func:`slot_view` for every binding (P5).

        Valid in ``PENDING`` and ``ROLLED_BACK``; leaves ``REPOINTED``.  Reads nothing from the
        slots, so it is valid before any gather.  If installing fails midway, the bindings already
        repointed are restored before the error propagates.

        Raises:
            DenseRepointError: Called in another state, or a binding could not be installed.
        """
        self._require("repoint", BindingState.PENDING, BindingState.ROLLED_BACK)
        views = [slot_view(entry.binding, entry.anchor, entry.slot) for entry in self._entries]
        installed = 0
        try:
            for entry, view in zip(self._entries, views):
                install_binding(entry.binding, view)
                installed += 1
        finally:
            if installed < len(self._entries):
                for entry in reversed(self._entries[:installed]):
                    install_binding(entry.binding, entry.binding.original)
        self._state = BindingState.REPOINTED

    def verify_layer(self, layer: int) -> list[str]:
        """P5 self-check of one layer, through the module attributes.

        The caller has gathered ``L(layer)`` into the layer's slots and synchronized.  For every
        binding of ``layer``, ``read_binding(binding)`` must be the binding's slot view (same
        storage, offset, size, stride and dtype) and :func:`~.layout.tensor_checksum` of
        ``read_binding(binding).contiguous()`` must equal the P1 ``binding.checksum``.

        Returns:
            ``layers.<layer>.<qualified name>`` of every mismatch; [] if all match.

        Raises:
            DenseRepointError: Not ``REPOINTED``, ``layer`` is not covered, a binding has no P1
                checksum, or an owner no longer holds its binding.
        """
        self._require("verify_layer", BindingState.REPOINTED)
        entries = self._by_layer.get(layer)
        if entries is None:
            raise DenseRepointError(f"layer {layer} is not covered by dense DWDP")
        mismatches = []
        with torch.no_grad():
            for entry in entries:
                binding = entry.binding
                if binding.checksum is None:
                    raise DenseRepointError(
                        f"layers.{layer}.{binding.qualified_name} has no P1 checksum"
                    )
                tensor = read_binding(binding)
                if (
                    not _is_slot_view(tensor, entry)
                    or tensor_checksum(tensor.contiguous()) != binding.checksum
                ):
                    mismatches.append(f"layers.{layer}.{binding.qualified_name}")
        return mismatches

    def rollback(self) -> None:
        """Make every owner expose its original tensor again (failure before the commit point).

        Valid in ``REPOINTED``; a no-op in ``PENDING`` and ``ROLLED_BACK``.  Afterwards every
        attribute, parameter and buffer is as collected: same ``nn.Parameter`` objects with their
        original ``.data`` views, the original buffer and attribute tensors, containers rebuilt
        with the original elements.  Every binding is attempted; the state is ``ROLLED_BACK`` even
        if some could not be restored.

        Raises:
            DenseRepointError: Called after :meth:`commit`, or an owner no longer holds its
                binding.
        """
        self._require(
            "rollback", BindingState.PENDING, BindingState.REPOINTED, BindingState.ROLLED_BACK
        )
        if self._state is not BindingState.REPOINTED:
            return
        failures = self._install_all(lambda entry: entry.binding.original)
        self._state = BindingState.ROLLED_BACK
        if failures:
            raise DenseRepointError(_listing("rollback could not restore every binding", failures))

    def commit(self) -> list[str]:
        """Drop the references to the originals and check that their storages die (P6).

        Valid in ``REPOINTED``; leaves ``COMMITTED``.  Layer by layer: take a weak reference to
        each anchor's storage, call ``AnchorRecord.drop_originals()``, and check the weak
        references are dead; after the last layer run ``gc.collect()`` once and check the survivors
        again.  A surviving storage keeps an old, byte-identical copy alive (memory, not numerics).
        The first ``_MAX_LISTED`` survivors (in layer order) get one warning each, naming the
        anchor path and its bindings; more survivors get one summary warning with their count,
        bytes and layers, so a leak that repeats in every layer does not flood the log.  Whether a
        leak fails setup is the caller's decision, which every rank of the group makes together.

        Returns:
            Anchor paths whose storage survived, all of them; [] if none.

        Raises:
            DenseRepointError: Called in another state.
        """
        self._require("commit", BindingState.REPOINTED)
        self._state = BindingState.COMMITTED
        survivors: list[tuple[AnchorRecord, weakref.ReferenceType[torch.UntypedStorage]]] = []
        for layer in self._layout.layers:
            layer_refs = []
            for segment in self._layout.layer_segments(layer):
                for anchor in segment.anchors:
                    storage_ref = _storage_ref(anchor)
                    anchor.drop_originals()
                    if storage_ref is not None:
                        layer_refs.append((anchor, storage_ref))
            survivors.extend(
                (anchor, storage_ref)
                for anchor, storage_ref in layer_refs
                if storage_ref() is not None
            )
        if survivors:
            gc.collect()
            survivors = [(anchor, ref) for anchor, ref in survivors if ref() is not None]
        for anchor, _ in survivors[:_MAX_LISTED]:
            bindings = ", ".join(binding.qualified_name for binding in anchor.bindings)
            logger.warning(
                f"{LOG_PREFIX} storage of {anchor.path} ({anchor.nbytes} bytes) is still alive "
                f"after commit: a holder outside its bindings ({bindings}) keeps the old copy"
            )
        if len(survivors) > _MAX_LISTED:
            rest = [anchor for anchor, _ in survivors[_MAX_LISTED:]]
            kinds = {anchor.segment for anchor in rest}
            total_bytes = sum(anchor.nbytes for anchor, _ in survivors)
            logger.warning(
                f"{LOG_PREFIX} {len(rest)} more moved storages "
                f"({sum(anchor.nbytes for anchor in rest)} bytes) are still alive after commit, "
                f"in segments {','.join(kind.value for kind in SegmentKind if kind in kinds)} of "
                f"layers {_layer_ranges(sorted({anchor.layer for anchor in rest}))}: "
                f"{len(survivors)} storages, {total_bytes} bytes in all; the first {_MAX_LISTED} "
                f"are named above"
            )
        return [anchor.path for anchor, _ in survivors]

    def detach(self) -> None:
        """Point every binding at a zero-size tensor of its dtype on the slot device (teardown).

        From ``COMMITTED``: leaves ``DETACHED``, so a later stray read fails on its shape instead of
        reading freed slot memory.  A binding whose owner was garbage collected counts as detached:
        no module can read it any more, so a model collected before teardown detaches cleanly.
        From ``REPOINTED``: rolls back instead (the originals are still alive).  A no-op in
        ``PENDING``, ``ROLLED_BACK`` and ``DETACHED``; idempotent.  Every binding is attempted.

        Raises:
            DenseRepointError: A live owner no longer holds its binding.
        """
        if self._state is BindingState.REPOINTED:
            self.rollback()
            return
        if self._state is not BindingState.COMMITTED:
            return
        empties: dict[tuple[torch.dtype, torch.device], torch.Tensor] = {}

        def empty_like(entry: _Entry) -> torch.Tensor:
            key = (entry.binding.dtype, entry.slot.storage.device)
            if key not in empties:
                empties[key] = torch.empty(0, dtype=key[0], device=key[1])
            return empties[key]

        failures = self._install_all(empty_like, skip_collected=True)
        self._state = BindingState.DETACHED
        if failures:
            raise DenseRepointError(_listing("detach could not reach every binding", failures))

    def _require(self, action: str, *states: BindingState) -> None:
        if self._state not in states:
            raise DenseRepointError(
                f"{action}() is not valid in state {self._state.value}; it needs "
                f"{' or '.join(state.value for state in states)}"
            )

    def _install_all(
        self,
        tensor_for: Callable[[_Entry], torch.Tensor | None],
        *,
        skip_collected: bool = False,
    ) -> list[str]:
        """Install ``tensor_for(entry)`` for every binding, last first; returns the failures.

        With ``skip_collected``, a binding whose owner was garbage collected is skipped instead of
        reported.
        """
        failures = []
        collected = 0
        for entry in reversed(self._entries):
            # Held until the install, so a live owner cannot be collected in between.
            owner = entry.binding.owner_ref()
            if owner is None and skip_collected:
                collected += 1
                continue
            tensor = tensor_for(entry)
            path = f"layers.{entry.anchor.layer}.{entry.binding.qualified_name}"
            if tensor is None:
                failures.append(f"{path}: no tensor to install")
                continue
            try:
                install_binding(entry.binding, tensor)
            except DenseRepointError as exc:
                failures.append(f"{path}: {exc}")
        if collected:
            logger.debug(
                f"{LOG_PREFIX} {collected} of {len(self._entries)} bindings belong to "
                f"garbage-collected modules; nothing reads them"
            )
        return failures


def _owner(binding: TensorBinding) -> nn.Module:
    owner = binding.owner_ref()
    if owner is None:
        raise DenseRepointError(f"the module owning {binding.qualified_name} was garbage collected")
    return owner


def _missing(owner: nn.Module, binding: TensorBinding) -> DenseRepointError:
    return DenseRepointError(
        f"{type(owner).__name__} at {binding.module_path} no longer holds "
        f"{binding.qualified_name} as a {binding.kind.value}"
    )


def _item(container: object, index: int | None) -> torch.Tensor | None:
    if not isinstance(container, (tuple, list)) or index is None or not 0 <= index < len(container):
        return None
    item = container[index]
    return item if isinstance(item, torch.Tensor) else None


def _rebuild(container: tuple | list, items: list[torch.Tensor]) -> tuple | list:
    cls = type(container)
    if cls is tuple or cls is list:
        return cls(items)
    make = getattr(cls, "_make", None)
    return make(items) if make is not None else cls(items)


def _is_slot_view(tensor: torch.Tensor, entry: _Entry) -> bool:
    """Whether ``tensor`` is the view :func:`slot_view` makes for ``entry``.

    The placement is compared as storage plus storage offset, not ``data_ptr()``, which is 0 for
    a tensor without elements wherever it points.
    """
    binding, anchor, slot = entry
    start = slot.storage.storage_offset() + slot.pad + anchor.blob_offset + binding.byte_offset
    return (
        tensor.dtype == binding.dtype
        and tuple(tensor.shape) == binding.size
        and tuple(tensor.stride()) == binding.stride
        and tensor.untyped_storage().data_ptr() == slot.storage.untyped_storage().data_ptr()
        and tensor.storage_offset() * tensor.element_size() == start
    )


def _storage_ref(anchor: AnchorRecord) -> weakref.ReferenceType[torch.UntypedStorage] | None:
    source = anchor.source
    if source is None:
        return None
    return weakref.ref(source.untyped_storage())


def _layer_ranges(layers: Sequence[int]) -> str:
    """Ascending layers as runs, e.g. ``[1, 2, 3, 5, 7, 8]`` as ``1..3, 5, 7..8``."""
    runs: list[list[int]] = []
    for layer in layers:
        if runs and layer == runs[-1][1] + 1:
            runs[-1][1] = layer
        else:
            runs.append([layer, layer])
    return ", ".join(str(first) if first == last else f"{first}..{last}" for first, last in runs)


def _listing(headline: str, findings: Sequence[str]) -> str:
    listed = "; ".join(findings[:_MAX_LISTED])
    more = len(findings) - _MAX_LISTED
    return f"{headline}: {listed}" + (f"; and {more} more" if more > 0 else "")
