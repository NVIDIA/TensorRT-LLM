"""Load clean model units, LoRA factors, and active states through the registry."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import cast

import torch

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.buffers import host_bytes
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.checkpoint import CheckpointReader, UnitSource
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.lora import FLOAT_FUSE_RULE, LoRAFuser, patched_keys
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.mapping import ExecutionCache, file_read_parts, fill_from_source, validate_destination
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.placement import Placement, PlacementPolicy, packs_storage
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.registry import ModelRegistry, RegistryKeys
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.state_entry import FusedKey, SourceKey, StateEntry, StateKey
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.units import (
    UNIT_ALIGN,
    LoRAUnitSpec,
    PatchedUnitSpec,
    StateDictMetadata,
    TensorsUnit,
    UnitSpec,
    active_dtype,
)


@dataclass(frozen=True, slots=True)
class StateLoader:
    """Build selected active units from checkpoint data and retained states.
    The registry owns read-only clean and LoRA states. An active state may be
    retained independently; a fused cache hit skips its clean and factor reads.
    """

    fuser: LoRAFuser
    registry: ModelRegistry
    placement_policy: PlacementPolicy

    def load(
        self,
        units: Sequence[PatchedUnitSpec],
        device: torch.device,
        dst: MutableMapping[str, torch.Tensor],
        held: Sequence[PatchedUnitSpec] | None = None,
    ) -> MutableMapping[str, torch.Tensor]:
        """Fill ``dst`` with all selected units using one reader and shared-group cache.
        ``held`` lists, per unit, the patched spec whose active state ``dst`` already holds.
        """
        if not units:
            return dst
        previous = (None,) * len(units) if held is None else tuple(held)
        if len(previous) != len(units):
            raise ValueError("held states and units differ in count")
        paths = tuple(
            dict.fromkeys(
                path for unit in units for source in (unit.clean, *unit.loras) for path in source.load_mapping.paths
            )
        )
        context = torch.cuda.device(device) if device.type == "cuda" else nullcontext()
        with context, CheckpointReader.open(paths, stagers=self.registry.device_stager) as reader:
            pending = tuple(
                bool(unit.loras) and self.registry.get_state(FusedKey.from_patched_unit_spec(unit)) is None
                for unit in units
            )
            selected = {
                SourceKey.from_unit_spec(adapter): adapter
                for unit, needs_factors in zip(units, pending, strict=True)
                if needs_factors
                for adapter in unit.loras
            }
            loaded = dict(zip(selected, self.load_factors(tuple(selected.values()), device, reader), strict=True))
            reading = tuple(
                unit.clean
                for unit, held_unit in zip(units, previous, strict=True)
                if self.reads_checkpoint(unit, held=held_unit)
            )
            cache = ExecutionCache.if_shared(reading)
            for unit, needs_factors, held_unit in zip(units, pending, previous, strict=True):
                factors = (
                    tuple(loaded[SourceKey.from_unit_spec(adapter)] for adapter in unit.loras) if needs_factors else ()
                )
                self.load_unit(
                    unit,
                    device,
                    dst,
                    factors,
                    source=reader,
                    cache=cache,
                    held=held_unit,
                )
        return dst

    def load_unit(  # noqa: PLR0912, PLR0915
        self,
        patched: PatchedUnitSpec,
        device: torch.device,
        dst: MutableMapping[str, torch.Tensor],
        factors: Sequence[TensorsUnit],
        *,
        source: UnitSource,
        cache: ExecutionCache | None = None,
        held: PatchedUnitSpec | None = None,
    ) -> MutableMapping[str, torch.Tensor]:
        """Fill selected keys in ``dst``, reusing a fused entry before reading inputs from ``source``.
        ``held`` is the patched spec of the same unit whose active state ``dst`` already holds;
        then only the keys either spec patches are rewritten, and the rest stay clean.
        """
        spec = patched.clean
        if cache is not None and not cache.tracks(spec):
            cache = None  # the cache serves only units sharing one of its groups; this one reads on its own
        # A retained state of this unit may be dst's storage: it keeps its own copy before dst is rewritten.
        existing = {name: dst[name] for name in spec.unit.metadata if name in dst}
        states = self.registry.unit_states(SourceKey.from_unit_spec(spec)) if existing else {}
        for key, entry in states.items():
            if _shares_storage(entry.tensors, existing):
                self.registry.pop_state(key)
                copied = {name: value.clone() for name, value in entry.tensors.items()}
                self.registry.adopt_entry(StateEntry(entry.spec, copied, entry.placement))
        if held is not None:
            return self._swap_unit(patched, held, device, dst, factors, source=source, cache=cache)
        aliases = spec.load_mapping.tensor_mapping.destination_aliases
        if aliases and patched.loras and dst:
            targets = set().union(*(adapter.bindings for adapter in patched.loras))
            for alias, root in aliases.items():
                if alias not in targets and root not in targets:
                    continue
                alias_storage, root_storage = dst.get(alias), dst.get(root)
                if (
                    alias_storage is not None
                    and root_storage is not None
                    and alias_storage.data_ptr() == root_storage.data_ptr()
                ):
                    raise ValueError(f"separately patched destinations {alias!r} and {root!r} share storage")
        active_placement = self.placement_policy.placement(patched)
        clean_placement = self.placement_policy.placement(spec)
        reuse_clean = not patched.loras and patched.requested_dtype is None and clean_placement == active_placement
        hit = None if reuse_clean else self.registry.get_state(FusedKey.from_patched_unit_spec(patched))
        if hit is not None:
            delivered = hit.to(device, non_blocking=device.type == "cuda")
            active = (
                delivered
                if patched.requested_dtype is None
                else _active_values(delivered, spec.unit.metadata, patched.requested_dtype, device, aliases)
            )
            if active_placement.device is not None and hit.placement != active_placement:
                self.registry.retain_state(patched, active, active_placement)
        elif not patched.loras and clean_placement.device is None and patched.requested_dtype is None:
            validate_destination(spec, dst, None)
            fill_from_source(file_read_parts(patched), source, destination=dst, device=device, cache=cache)
            self._retain_active(patched, dst, active_placement)
            return dst
        else:
            clean, private = self._clean(spec, device, source, cache)
            active_clean = (
                clean
                if patched.requested_dtype is None
                else _active_values(clean, spec.unit.metadata, patched.requested_dtype, device, aliases)
            )
            complete_dst = bool(dst) and all(name in dst for name in spec.unit.metadata)
            if not patched.loras:
                active = active_clean
            elif aliases or not (complete_dst or private):
                active = {**active_clean, **self.fuser.patch(patched, active_clean, factors)}
            elif complete_dst:
                written = self.fuser.patch_into(patched, active_clean, factors, dst)
                with torch.no_grad():
                    for name, value in active_clean.items():
                        if name not in written and dst[name] is not value:
                            dst[name].copy_(value)
                active = dst
            else:
                self.fuser.patch_into(patched, active_clean, factors, active_clean)
                active = active_clean
            if not reuse_clean:
                self._retain_active(patched, active, active_placement)
        if active is dst:
            return dst
        if not dst or dst.keys().isdisjoint(active):
            dst.update(active)
            return dst
        _copy_into_destination(dst, active)
        return dst

    def reads_checkpoint(
        self,
        patched: PatchedUnitSpec,
        *,
        held: PatchedUnitSpec | None = None,
    ) -> bool:
        """Whether ``load_unit`` would read this unit from its checkpoint instead of taking a registry state.
        A shared-read ``ExecutionCache`` registers only these units, so it counts exactly their uses.
        """
        spec = patched.clean
        if held is not None:
            if not patched_keys(held) | patched_keys(patched):
                return False
            if patched.loras and self.registry.get_state(FusedKey.from_patched_unit_spec(patched)) is not None:
                return False
            return self.registry.get_state(SourceKey.from_unit_spec(spec)) is None
        active_placement = self.placement_policy.placement(patched)
        clean_placement = self.placement_policy.placement(spec)
        reuse_clean = not patched.loras and patched.requested_dtype is None and clean_placement == active_placement
        if not reuse_clean and self.registry.get_state(FusedKey.from_patched_unit_spec(patched)) is not None:
            return False
        if not patched.loras and clean_placement.device is None and patched.requested_dtype is None:
            return True
        return self.registry.get_state(SourceKey.from_unit_spec(spec)) is None

    def load_factors(
        self, loras: Sequence[LoRAUnitSpec], device: torch.device, source: UnitSource
    ) -> tuple[TensorsUnit, ...]:
        """Load each source once and return factor units in request order."""
        supplied = []
        loaded: dict[SourceKey, TensorsUnit] = {}
        for adapter in loras:
            key = SourceKey.from_unit_spec(adapter)
            if key in loaded:
                supplied.append(loaded[key])
                continue
            placement = self.placement_policy.placement(adapter)
            hit = self.registry.get_state(key)
            if hit is not None:
                if placement.device is not None and hit.placement != placement:
                    hit = self.registry.retain_state(adapter, hit.tensors, placement) or hit
                values = hit.tensors
            else:
                loading = placement if placement.device is not None else self.fuser.factor_loading_placement(device)
                owner, values, actual = _unit_storage(
                    adapter.unit.metadata,
                    loading,
                    adapter.load_mapping.tensor_mapping.destination_strides,
                    adapter.load_mapping.tensor_mapping.destination_aliases,
                )
                fill_from_source(
                    file_read_parts(PatchedUnitSpec(adapter, FLOAT_FUSE_RULE)),
                    source,
                    destination=values,
                    device=loading.device,
                )
                if placement.device is not None:
                    values = self.registry.adopt_entry(StateEntry(adapter, values, actual, owner)).tensors
            unit = TensorsUnit(adapter.unit.unit_id, values)
            loaded[key] = unit
            supplied.append(unit)
        return tuple(supplied)

    def registry_keys(self, units: Sequence[PatchedUnitSpec], *, factors: bool = True) -> RegistryKeys:
        """Return the registry keys ``load`` queries and retains for these units.
        With ``factors=False`` the caller loads LoRA factors itself and the units' adapter keys are left out.
        """
        lookup: set[StateKey] = set()
        retained: set[StateKey] = set()
        for patched in units:
            spec = patched.clean
            active_placement = self.placement_policy.placement(patched)
            clean_placement = self.placement_policy.placement(spec)
            reuse_clean = not patched.loras and patched.requested_dtype is None and clean_placement == active_placement
            if not reuse_clean:
                fused = FusedKey.from_patched_unit_spec(patched)
                lookup.add(fused)
                if active_placement.device is not None:
                    retained.add(fused)
            if patched.loras or clean_placement.device is not None or patched.requested_dtype is not None:
                source = SourceKey.from_unit_spec(spec)
                lookup.add(source)
                if clean_placement.device is not None:
                    retained.add(source)
        keys = RegistryKeys(frozenset(lookup), frozenset(retained))
        if factors:
            keys |= self.factor_keys(tuple(adapter for unit in units for adapter in unit.loras))
        return keys

    def factor_keys(self, loras: Sequence[LoRAUnitSpec]) -> RegistryKeys:
        """Return the registry keys ``load_factors`` queries and retains for these adapters."""
        lookup = frozenset(SourceKey.from_unit_spec(adapter) for adapter in loras)
        retained = frozenset(
            SourceKey.from_unit_spec(adapter)
            for adapter in loras
            if self.placement_policy.placement(adapter).device is not None
        )
        return RegistryKeys(lookup, retained)

    def _clean(
        self, spec: UnitSpec, device: torch.device, source: UnitSource, cache: ExecutionCache | None
    ) -> tuple[dict[str, torch.Tensor], bool]:
        placement = self.placement_policy.placement(spec)
        hit = self.registry.get_state(SourceKey.from_unit_spec(spec))
        if hit is not None:
            if placement.device is not None and hit.placement != placement:
                retained = self.registry.retain_state(cast(UnitSpec, hit.spec), hit.tensors, placement)
                if retained is not None and retained.placement.device == device:
                    hit = retained
            return hit.to(device, non_blocking=device.type == "cuda"), False
        loading = placement if placement.device is not None else Placement(device)
        owner, values, actual = _unit_storage(
            spec.unit.metadata,
            loading,
            spec.load_mapping.tensor_mapping.destination_strides,
            spec.load_mapping.tensor_mapping.destination_aliases,
        )
        patched = PatchedUnitSpec(spec, FLOAT_FUSE_RULE)
        fill_from_source(file_read_parts(patched), source, destination=values, device=loading.device, cache=cache)
        if placement.device is not None:
            retained = self.registry.adopt_entry(StateEntry(spec, values, actual, owner))
            values = dict(retained.tensors)
            return values if actual.device == device else retained.to(device), False
        if loading.device == device:
            return values, True
        temporary = StateEntry(spec, values, actual, owner)
        return temporary.to(device, non_blocking=device.type == "cuda"), True

    def _retain_active(
        self, patched: PatchedUnitSpec, values: Mapping[str, torch.Tensor], placement: Placement
    ) -> None:
        """Retain an active state where ``placement`` says.
        Device storage on the values' own device keeps them as they are; any other placement keeps a copy.
        """
        if placement.device is None:
            return
        selected = {name: values[name] for name in patched.clean.unit.metadata}
        if not packs_storage(placement.device) and all(value.device == placement.device for value in selected.values()):
            self.registry.adopt_entry(StateEntry(patched, selected, placement))
        else:
            self.registry.retain_state(patched, selected, placement)

    def _swap_unit(
        self,
        patched: PatchedUnitSpec,
        held: PatchedUnitSpec,
        device: torch.device,
        dst: MutableMapping[str, torch.Tensor],
        factors: Sequence[TensorsUnit],
        *,
        source: UnitSource,
        cache: ExecutionCache | None,
    ) -> MutableMapping[str, torch.Tensor]:
        """Turn ``held``'s active state in ``dst`` into ``patched``'s, rewriting only keys either patches."""
        spec = patched.clean
        if held.clean.unit.unit_id != spec.unit.unit_id:
            raise ValueError(f"{spec.unit.unit_id!r} cannot replace the held state of {held.clean.unit.unit_id!r}")
        written = patched_keys(patched)
        touched = patched_keys(held) | written
        hit = self.registry.get_state(FusedKey.from_patched_unit_spec(patched)) if touched and patched.loras else None
        if not touched:
            return dst
        if hit is not None:
            # The retained active state already carries the patch.
            current, written = hit.tensors, frozenset()
        else:
            current, _ = self._clean(spec, device, source, cache)
        aliases = spec.load_mapping.tensor_mapping.destination_aliases
        active = _active_values(
            {name: current[name] for name in touched},
            spec.unit.metadata,
            patched.requested_dtype,
            device,
            aliases,
        )
        with torch.no_grad():
            for name in touched - written:
                dst[name].copy_(active[name])
        if written:
            self.fuser.patch_into(patched, active, factors, dst)
            self._retain_active(patched, dst, self.placement_policy.placement(patched))
        return dst


def _unit_storage(
    metadata: StateDictMetadata,
    placement: Placement,
    strides: Mapping[str, tuple[int, ...]],
    aliases: Mapping[str, str],
) -> tuple[torch.Tensor | None, dict[str, torch.Tensor], Placement]:
    """Allocate a unit's storage, sharing identical projected outputs.
    Host storage is one aligned owner with tensor views. Device storage gets one
    allocation per distinct output and no owner (see ``packs_storage``).
    """
    device = placement.device
    if device is None:
        raise ValueError("unit storage needs a device")
    if not packs_storage(device):
        separate: dict[str, torch.Tensor] = {}
        for name, (shape, dtype) in metadata.items():
            root = aliases.get(name, name)
            if root not in separate:
                stride = strides.get(name)
                separate[root] = (
                    torch.empty(shape, dtype=dtype, device=device)
                    if stride is None
                    else torch.empty_strided(shape, stride, dtype=dtype, device=device)
                )
        return None, {name: separate[aliases.get(name, name)] for name in metadata}, Placement(device)
    regions: dict[str, tuple[int, int]] = {}
    first_selected: dict[str, str] = {}
    total = 0
    for name, (shape, dtype) in metadata.items():
        root = aliases.get(name, name)
        first = first_selected.setdefault(root, name)
        if first != name:
            regions[name] = regions[first]
            continue
        total = (total + UNIT_ALIGN - 1) // UNIT_ALIGN * UNIT_ALIGN
        stride = strides.get(name)
        span = (
            1 + sum((size - 1) * step for size, step in zip(shape, stride, strict=True))
            if stride is not None and shape.numel()
            else shape.numel()
        )
        regions[name] = total, span
        total += span * dtype.itemsize
    owner = host_bytes(total, pin=placement.pin_memory and torch.cuda.is_available())
    views: dict[str, torch.Tensor] = {}
    for name, (shape, dtype) in metadata.items():
        offset, span = regions[name]
        raw = owner.narrow(0, offset, span * dtype.itemsize).view(dtype)
        stride = strides.get(name)
        views[name] = raw.reshape(shape) if stride is None else torch.as_strided(raw, shape, stride)
    return owner, views, Placement(device, pin_memory=owner.is_pinned())


def _active_values(
    values: Mapping[str, torch.Tensor],
    metadata: StateDictMetadata,
    requested_dtype: torch.dtype | None,
    device: torch.device,
    aliases: Mapping[str, str],
) -> dict[str, torch.Tensor]:
    """Convert values of the clean ``metadata`` keys to their active dtypes on ``device``, keeping their layout."""
    result = {}
    converted: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name, value in values.items():
        root = aliases.get(name, name)
        earlier = converted.get(root)
        dtype = active_dtype(metadata[name], requested_dtype)
        if (
            earlier is not None
            and value.data_ptr() == earlier[0].data_ptr()
            and value.shape == earlier[0].shape
            and value.stride() == earlier[0].stride()
            and dtype == earlier[1].dtype
        ):
            result[name] = earlier[1]
            continue
        result[name] = value.to(device=device, dtype=dtype)
        converted[root] = value, result[name]
    return result


def _shares_storage(values: Mapping[str, torch.Tensor], others: Mapping[str, torch.Tensor]) -> bool:
    """Whether any of ``values`` uses the storage of one of ``others``."""
    storages = {other.untyped_storage().data_ptr() for other in others.values()}
    return any(value.untyped_storage().data_ptr() in storages for value in values.values())


def _copy_into_destination(dst: MutableMapping[str, torch.Tensor], active: Mapping[str, torch.Tensor]) -> None:
    """Copy overlapping values into caller storage and add missing values."""
    with torch.no_grad():
        for name, value in active.items():
            target = dst.get(name)
            if target is None:
                dst[name] = value
            else:
                target.copy_(value)
