"""Short state keys and retained tensor entries for unit loading."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import torch

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.placement import Placement
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.units import PatchedUnitSpec, UnitSpec

SourcePrefix = tuple[tuple[Path, ...], tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class SourceKey:
    """Identify selected tensors without storing their metadata in the key.
    ``paths`` holds resolved checkpoint files in configured order. ``sd_ops_names`` retains
    transformation order. ``unit_id`` identifies the selection within that
    checkpoint recipe and must remain stable across equivalent builds.
    """

    paths: tuple[Path, ...]
    sd_ops_names: tuple[str, ...]
    unit_id: str

    def __post_init__(self) -> None:
        """Resolve paths and require a nonempty unit ID for every construction."""
        if not self.unit_id:
            raise ValueError("source identity needs a unit ID")
        object.__setattr__(self, "paths", tuple(Path(path).resolve() for path in self.paths))
        object.__setattr__(self, "sd_ops_names", tuple(self.sd_ops_names))

    @property
    def prefix(self) -> SourcePrefix:
        """Return checkpoint paths and SDOps names without the unit ID."""
        return self.paths, self.sd_ops_names

    @classmethod
    def from_unit_spec(cls, spec: UnitSpec) -> SourceKey:
        """Build a source key from the unit ID and its shared load mapping."""
        mapping = spec.load_mapping
        return cls(mapping.paths, tuple(operation.name for operation in mapping.sd_ops), spec.unit.unit_id)


@dataclass(frozen=True, slots=True)
class FusedKey:
    """Identify an active state by its clean source and ordered LoRA selections.
    ``adapters`` pairs each LoRA factor-unit key with its strength. The fusion
    rule name and requested dtype distinguish results from the same sources.
    A spec without LoRAs uses ``clean`` as its rule name.
    """

    clean: SourceKey
    adapters: tuple[tuple[SourceKey, float], ...]
    rule_name: str
    requested_dtype: torch.dtype | None

    @classmethod
    def from_patched_unit_spec(cls, spec: PatchedUnitSpec) -> FusedKey:
        """Identify the active state selected by a patched unit spec."""
        return cls(
            SourceKey.from_unit_spec(spec.clean),
            tuple((SourceKey.from_unit_spec(lora), lora.strength) for lora in spec.loras),
            spec.fuse_rule.name if spec.loras else "clean",
            spec.requested_dtype,
        )


StateKey = SourceKey | FusedKey


def selected_metadata(spec: UnitSpec | PatchedUnitSpec) -> Mapping[str, tuple[torch.Size, torch.dtype]]:
    """Return the model or factor metadata selected by this spec."""
    return spec.clean.unit.metadata if isinstance(spec, PatchedUnitSpec) else spec.unit.metadata


@dataclass(frozen=True, slots=True, eq=False)
class StateEntry:
    """One retained selection of model tensors or LoRA factors.
    ``spec`` retains the selected unit and its full checkpoint mapping,
    including ordered LoRAs and the fusion rule for a patched state. The
    registry derives and stores the short logical key from that spec.
    ``tensors`` holds the selected model weights or LoRA factors by name. Each
    tensor may have its own storage; when ``buffer`` is present, all tensors
    are views into that shared byte allocation. ``placement`` records the
    storage device and whether CPU storage is pinned. The caller must supply
    the views into ``buffer``; an entry with a buffer checks that allocation
    instead of validating every tensor separately.
    """

    spec: UnitSpec | PatchedUnitSpec
    tensors: Mapping[str, torch.Tensor]
    placement: Placement
    buffer: torch.Tensor | None = None

    def __post_init__(self) -> None:
        """Validate values against the selected unit and freeze the key-to-view map."""
        if self.placement.device is None:
            raise ValueError("an unplaced state cannot be retained")
        if self.buffer is not None:
            if (
                self.buffer.device != self.placement.device
                or self.buffer.dtype != torch.uint8
                or self.buffer.ndim != 1
                or not self.buffer.is_contiguous()
            ):
                raise ValueError("state buffer must be contiguous uint8 on its retention device")
            object.__setattr__(self, "tensors", MappingProxyType(dict(self.tensors)))
            return
        expected = selected_metadata(self.spec)
        if tuple(self.tensors) != tuple(expected):
            raise ValueError("state entry keys disagree with selected unit")
        for name, (shape, dtype) in expected.items():
            value = self.tensors[name]
            if value.shape != shape or (isinstance(self.spec, UnitSpec) and value.dtype != dtype):
                raise ValueError(f"state entry {name!r} disagrees with selected unit")
            if value.device != self.placement.device:
                raise ValueError(f"state entry {name!r} is on {value.device}, expected {self.placement.device}")
        object.__setattr__(self, "tensors", MappingProxyType(dict(self.tensors)))

    def to(self, device: torch.device, *, non_blocking: bool = False) -> dict[str, torch.Tensor]:
        """Return its tensors on ``device``, copying the shared buffer once when present.
        A same-device request returns the retained tensor views without copying.
        """
        device = torch.device(device)
        if device == self.placement.device:
            return dict(self.tensors)
        if self.buffer is None:
            return {name: value.to(device, non_blocking=non_blocking) for name, value in self.tensors.items()}

        transferred = self.buffer.to(device, non_blocking=non_blocking)
        source_start = self.buffer.data_ptr()
        result: dict[str, torch.Tensor] = {}
        for name, value in self.tensors.items():
            if not value.numel():
                result[name] = torch.empty(value.shape, dtype=value.dtype, device=device)
                continue
            byte_offset = value.data_ptr() - source_start
            if byte_offset < 0 or byte_offset % value.element_size():
                raise ValueError(f"retained tensor {name!r} is outside its buffer")
            element_offset = byte_offset // value.element_size()
            typed = transferred.view(value.dtype)
            result[name] = torch.as_strided(typed, value.shape, value.stride(), storage_offset=element_offset)
        return result
