"""Bring one streamed unit from where its weights are to active tensors in a device buffer.
A unit's ``Readiness`` names what its data already is: nothing read yet, checkpoint tensors exactly as stored,
clean SDOps outputs, or active tensors at the requested dtype with every LoRA applied. A retained state decides
where a unit starts: an exact ``FusedKey`` entry is active, a ``SourceKey`` entry is clean, and every other unit
is read from its checkpoint.
A unit's ``transfer_at`` is the readiness at which its bytes cross to the device: the steps before it run on
the host, the steps after it on the device. ``CHECKPOINT`` runs the SDOps on the device, when the mapping requires
it; ``CLEAN`` runs them on the host and fuses LoRAs on the device; ``ACTIVE`` also fuses in host memory, for weights
the fuse rule cannot fuse in the device's memory, so only finished bytes are copied. A unit that starts at or past
its ``transfer_at`` is copied as it is.
The functions here never see the scheduler or the transport type. The read worker calls ``read_unit`` into a
host buffer; ``deliver`` runs on the transport's queue and leaves active tensors in a device buffer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import IntEnum

import torch

from tensorrt_llm._torch.visual_gen.models.ltx_core.loader.buffers import aligned_region_bytes, layout_bytes, unit_views
from tensorrt_llm._torch.visual_gen.models.ltx_core.loader.checkpoint import UnitSource
from tensorrt_llm._torch.visual_gen.models.ltx_core.loader.lora import LoRAFuser
from tensorrt_llm._torch.visual_gen.models.ltx_core.loader.mapping import ExecutionCache, file_read_parts, fill_from_source, transform
from tensorrt_llm._torch.visual_gen.models.ltx_core.loader.partitioners import stage_inputs
from tensorrt_llm._torch.visual_gen.models.ltx_core.loader.state_entry import StateEntry
from tensorrt_llm._torch.visual_gen.models.ltx_core.loader.streaming_transport import Transfer
from tensorrt_llm._torch.visual_gen.models.ltx_core.loader.units import PatchedUnitSpec, StateDictMetadata, TensorsUnit


class Readiness(IntEnum):
    """What a streamed unit's data already is, ordered from least to most finished."""

    UNREAD = 0
    """Nothing is read: the unit comes from its checkpoint."""
    CHECKPOINT = 1
    """Checkpoint tensors exactly as stored, before the SDOps."""
    CLEAN = 2
    """SDOps outputs, as a ``SourceKey`` entry holds them; LoRAs are not applied yet."""
    ACTIVE = 3
    """Requested dtype and every LoRA applied, as a ``FusedKey`` entry holds them. Says nothing about location."""


def readiness_of(entry: StateEntry | None) -> Readiness:
    """Where a unit starts: its retained state's spec tells an active entry from a clean one."""
    if entry is None:
        return Readiness.UNREAD
    return Readiness.ACTIVE if isinstance(entry.spec, PatchedUnitSpec) else Readiness.CLEAN


@dataclass(frozen=True, slots=True)
class Staged:
    """A unit's data on the host, ready for transfer.
    ``payload`` is a byte range copied to the front of the device buffer: a host buffer region or a retained byte
    owner already laid out for the device. ``tensors`` are retained tensors copied one by one instead.
    ``host_index`` is the host buffer still leased for this unit.
    """

    uid: str
    readiness: Readiness
    payload: torch.Tensor | None = None
    tensors: Mapping[str, torch.Tensor] | None = None
    host_index: int | None = None


def checkpoint_metadata(spec: PatchedUnitSpec) -> StateDictMetadata:
    """The checkpoint tensors a unit's SDOps read, in a stable order."""
    tensors = spec.clean.load_mapping.tensor_mapping
    keys = sorted({key for name in spec.clean.unit.metadata for key in tensors.dependencies[name]})
    return {key: tensors.source_metadata[key] for key in keys}


def active_nbytes(spec: PatchedUnitSpec) -> int:
    """Bytes of a unit's active tensors with their projected strides."""
    clean = spec.clean
    return layout_bytes(
        clean.unit.metadata, clean.load_mapping.tensor_mapping.destination_strides, spec.requested_dtype
    )


def host_nbytes(spec: PatchedUnitSpec, readiness: Readiness, transfer_at: Readiness) -> int:
    """Host buffer bytes a unit needs before its transfer; zero for a unit copied as it starts."""
    if readiness >= transfer_at:
        return 0
    return layout_bytes(checkpoint_metadata(spec)) if transfer_at is Readiness.CHECKPOINT else active_nbytes(spec)


def staged_input_metadata(spec: PatchedUnitSpec) -> StateDictMetadata:
    """The raw inputs a unit's value operations read when its SDOps run on the host, in a stable order."""
    tensors = spec.clean.load_mapping.tensor_mapping
    transformed = [name for name in spec.clean.unit.metadata if name in tensors.transformed_value_destinations]
    return {key: tensors.source_metadata[key] for key in stage_inputs(tensors, transformed)}


def host_buffer_nbytes(spec: PatchedUnitSpec, readiness: Readiness, transfer_at: Readiness) -> int:
    """Host buffer bytes: what a unit transfers, then an aligned region for the raw inputs it stages on the host.
    Only a unit cleaned from its checkpoint on the host stages inputs, for its value operations.
    """
    nbytes = host_nbytes(spec, readiness, transfer_at)
    staged = staged_input_metadata(spec) if readiness is Readiness.UNREAD and transfer_at > Readiness.CHECKPOINT else {}
    return aligned_region_bytes(nbytes) + layout_bytes(staged) if staged else nbytes


def device_nbytes(spec: PatchedUnitSpec, readiness: Readiness, transfer_at: Readiness) -> int:
    """Device buffer bytes: active tensors, after an aligned region of checkpoint tensors when SDOps run there."""
    if transfer_at is Readiness.CHECKPOINT and readiness < transfer_at:
        return aligned_region_bytes(layout_bytes(checkpoint_metadata(spec))) + active_nbytes(spec)
    return active_nbytes(spec)


def active_views(region: torch.Tensor, spec: PatchedUnitSpec) -> dict[str, torch.Tensor]:
    """Typed views of the unit's active tensors, with their projected strides, at the front of a byte region."""
    clean = spec.clean
    strides = clean.load_mapping.tensor_mapping.destination_strides
    return unit_views(region, clean.unit.metadata, strides, spec.requested_dtype)


class UnitViews:
    """Typed views of units in reused byte buffers, made once per buffer region, unit, and readiness."""

    def __init__(self) -> None:
        self._views: dict[tuple[torch.device, int, int, str, Readiness], dict[str, torch.Tensor]] = {}

    def active(self, region: torch.Tensor, spec: PatchedUnitSpec) -> dict[str, torch.Tensor]:
        """Views of the unit's active tensors at the front of ``region``.
        Callers fill the returned tensors in place and must not replace its entries.
        """
        key = (region.device, region.data_ptr(), region.numel(), spec.clean.unit.unit_id, Readiness.ACTIVE)
        views = self._views.get(key)
        if views is None:
            views = self._views[key] = active_views(region, spec)
        return views

    def checkpoint(self, region: torch.Tensor, spec: PatchedUnitSpec) -> dict[str, torch.Tensor]:
        """Views of the checkpoint tensors the unit's SDOps read, at the front of ``region``."""
        key = (region.device, region.data_ptr(), region.numel(), spec.clean.unit.unit_id, Readiness.CHECKPOINT)
        views = self._views.get(key)
        if views is None:
            views = self._views[key] = unit_views(region, checkpoint_metadata(spec))
        return views

    def clear(self) -> None:
        self._views.clear()


def owner_matches(entry: StateEntry, spec: PatchedUnitSpec) -> bool:
    """Whether a retained byte owner already has the unit's active device layout, so it moves as one copy."""
    owner = entry.buffer
    clean = spec.clean
    if owner is None or owner.numel() != active_nbytes(spec) or entry.tensors.keys() != clean.unit.metadata.keys():
        return False
    expected = active_views(owner, spec)
    return all(
        (value.shape, value.dtype, value.data_ptr(), value.stride())
        == (view.shape, view.dtype, view.data_ptr(), view.stride())
        for value, view in ((entry.tensors[name], view) for name, view in expected.items())
    )


def read_unit(
    spec: PatchedUnitSpec,
    source: UnitSource,
    host: torch.Tensor,
    views: UnitViews,
    *,
    transfer_at: Readiness,
    cache: ExecutionCache | None = None,
    parts: Sequence[PatchedUnitSpec] | None = None,
    staging: torch.Tensor | None = None,
) -> Readiness:
    """Fill a host buffer region from the checkpoint and return what it now holds.
    A unit transferred as checkpoint tensors is filled with its raw inputs as stored; its SDOps run in ``deliver``.
    Every other unit has its SDOps run here, filling the active layout at the requested dtype through
    ``fill_from_source`` with ``parts``, its ``file_read_parts`` (split here when absent), ``staging`` bytes for the
    raw inputs of its value operations, and ``cache`` for its shared groups.
    """
    if transfer_at is Readiness.CHECKPOINT:
        source.read_into(spec.clean.load_mapping.tensor_mapping, views.checkpoint(host, spec))
        return Readiness.CHECKPOINT
    parts = file_read_parts(spec) if parts is None else parts
    fill_from_source(parts, source, destination=views.active(host, spec), cache=cache, staging=staging)
    return Readiness.CLEAN


def fill_device(
    spec: PatchedUnitSpec, staged: Staged, buffer: torch.Tensor, transfer: Transfer, views: UnitViews
) -> dict[str, torch.Tensor]:
    """Copy a staged unit into a device buffer and run any SDOps still due; return the active views.
    Every returned tensor is a view into ``buffer``: a temporary allocated here belongs to the transfer queue.
    """
    if staged.readiness is Readiness.CHECKPOINT:
        assert staged.payload is not None
        transfer.copy(buffer.narrow(0, 0, staged.payload.numel()), staged.payload)
        checkpoint = views.checkpoint(buffer, spec)
        start = aligned_region_bytes(staged.payload.numel())
        active = views.active(buffer.narrow(0, start, buffer.numel() - start), spec)
        transform(spec.clean, checkpoint, destination=active, requested_dtype=spec.requested_dtype)
        return active
    active = views.active(buffer, spec)
    if staged.payload is not None:
        transfer.copy(buffer.narrow(0, 0, staged.payload.numel()), staged.payload)
    else:
        assert staged.tensors is not None
        for name, view in active.items():
            transfer.copy(view, staged.tensors[name])
    return active


def deliver(
    spec: PatchedUnitSpec,
    staged: Staged,
    transfer_at: Readiness,
    buffer: torch.Tensor,
    transfer: Transfer,
    views: UnitViews,
    fuser: LoRAFuser | None,
    factors: Sequence[TensorsUnit],
) -> Mapping[str, torch.Tensor]:
    """Leave a staged unit's active tensors in a device buffer, running on the transport's queue.
    Steps the unit still needs before ``transfer_at`` run on the host, the rest after the copy, on the device.
    """
    readiness = staged.readiness
    if readiness < transfer_at:
        # Fuse in host memory; deltas are still computed on the device.
        assert readiness is Readiness.CLEAN
        assert staged.payload is not None
        assert fuser is not None
        host = views.active(staged.payload, spec)
        replace(fuser, apply_device=fuser.application_device(buffer.device)).patch_into(spec, host, factors, host)
        readiness = Readiness.ACTIVE
    active = fill_device(spec, staged, buffer, transfer, views)
    if readiness < Readiness.ACTIVE and spec.loras:
        assert fuser is not None
        fuser.patch_into(spec, active, factors, active)
    return active
