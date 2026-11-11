"""Partition patched model specs while preserving fusion-required model-key groups."""

from __future__ import annotations

import hashlib
from collections import Counter, OrderedDict
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import Protocol

import torch

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.units import (
    FileReadUnit,
    LoRAUnitSpec,
    MetadataUnit,
    ModelKeyGroup,
    PatchedUnitSpec,
    RankUnit,
    StreamedUnit,
    TensorMapping,
    TensorMetadata,
    TensorsUnit,
)


class LoRAPartitioning(str, Enum):
    """Keep each LoRA whole or select its factors with model-weight ownership."""

    WHOLE = "whole"
    FOLLOW_MODEL = "follow_model"


class Partitioner(Protocol):
    """Split patched units, including their LoRAs, into new patched units."""

    def __call__(self, specs: Sequence[PatchedUnitSpec]) -> tuple[PatchedUnitSpec, ...]: ...


def _members(group: ModelKeyGroup) -> tuple[str, ...]:
    return (group,) if isinstance(group, str) else group


def _model_key_groups(spec: PatchedUnitSpec) -> Iterable[ModelKeyGroup]:
    return spec.clean.unit.metadata if spec.model_key_groups is None else spec.model_key_groups


def _select_loras(
    spec: PatchedUnitSpec, partitioned_unit: MetadataUnit, behavior: LoRAPartitioning
) -> tuple[LoRAUnitSpec, ...]:
    """Keep whole LoRAs or select their factors for this partitioned model unit."""
    if behavior is LoRAPartitioning.WHOLE:
        return spec.loras
    selected: list[LoRAUnitSpec] = []
    for lora in spec.loras:
        factors = {}
        for model_key in partitioned_unit.metadata:
            if pair := lora.bindings.get(model_key):
                factors[pair.a] = lora.unit.metadata[pair.a]
                factors[pair.b] = lora.unit.metadata[pair.b]
        if factors:
            selected.append(
                replace(
                    lora,
                    unit=MetadataUnit(f"{lora.unit.unit_id}@{partitioned_unit.unit_id}", factors),
                )
            )
    return tuple(selected)


def _child(
    spec: PatchedUnitSpec,
    partitioned_unit: MetadataUnit,
    groups: list[ModelKeyGroup] | None,
    lora_partitioning: LoRAPartitioning,
) -> PatchedUnitSpec:
    """Build a child spec from the partitioned model unit and LoRA partitioning."""
    return replace(
        spec,
        clean=replace(spec.clean, unit=partitioned_unit),
        loras=_select_loras(spec, partitioned_unit, lora_partitioning),
        model_key_groups=tuple(groups) if groups is not None else None,
    )


def _validate_model_groups(previous: Sequence[PatchedUnitSpec], current: Sequence[PatchedUnitSpec]) -> None:
    """Require child selections to keep each prepared model-key group intact."""
    group_by_key: dict[str, tuple[str, ...]] = {}
    for spec in previous:
        if spec.model_key_groups is None:
            continue
        for group in spec.model_key_groups:
            if isinstance(group, tuple):
                group_by_key.update((key, group) for key in group)
    for child in current:
        keys = child.clean.unit.metadata
        for group in {group_by_key[key] for key in keys if key in group_by_key}:
            if any(key not in keys for key in group):
                raise ValueError(f"partition separates fusion group {group!r}")
        if child.model_key_groups is not None and Counter(
            key for group in child.model_key_groups for key in _members(group)
        ) != Counter(keys.keys()):
            raise ValueError("partition groups disagree with selected model keys")


def validate_partition(
    previous: Sequence[PatchedUnitSpec],
    current: Sequence[PatchedUnitSpec],
    *,
    shared: Collection[str] = (),
) -> None:
    """Audit selected keys, mapping identity, shared uses, and intact fusion groups."""
    if not previous or not current:
        raise ValueError("a partition must contain units")
    ids = [spec.clean.unit.unit_id for spec in current]
    if len(ids) != len(set(ids)):
        raise ValueError("a partition has duplicate unit ids")
    mapping = previous[0].clean.load_mapping
    if any(spec.clean.load_mapping is not mapping for spec in (*previous, *current)):
        raise ValueError("a partition must share its original load mapping")
    before = Counter(key for spec in previous for key in spec.clean.unit.metadata)
    after = Counter(key for spec in current for key in spec.clean.unit.metadata)
    shared_keys = set(shared)
    if unknown := shared_keys - before.keys():
        raise ValueError(f"shared uses name unknown keys: {sorted(unknown)}")
    if before.keys() != after.keys() or any(
        after[key] < before[key] if key in shared_keys else after[key] != before[key] for key in before
    ):
        raise ValueError("partition changes destination use counts")
    metadata = {key: value for spec in previous for key, value in spec.clean.unit.metadata.items()}
    if any(metadata[key] != value for spec in current for key, value in spec.clean.unit.metadata.items()):
        raise ValueError("partition changes tensor metadata")
    _validate_model_groups(previous, current)


def apply_partitioners(root: PatchedUnitSpec, partitioners: Sequence[Partitioner]) -> tuple[PatchedUnitSpec, ...]:
    """Apply partitioner objects in order to the same patched-spec structure."""
    specs = (root,)
    for partitioner in partitioners:
        specs = partitioner(specs)
    return specs


def _block_index(group: ModelKeyGroup, prefix: str) -> int | None:
    """Read a canonical block index without splitting a model-key group."""
    members = _members(group)
    key = members[0]
    if not key.startswith(prefix):
        return None
    index_text, separator, _ = key[len(prefix) :].partition(".")
    if not separator or not index_text.isascii() or not index_text.isdecimal():
        return None
    index = int(index_text)
    if index_text != str(index):
        raise ValueError(f"{key!r} uses a non-canonical block index")
    if any(not member.startswith(f"{prefix}{index}.") for member in members):
        raise ValueError(f"fusion group crosses blocks: {group!r}")
    return index


@dataclass(frozen=True, slots=True)
class DenseBlockPartitioner(Partitioner):
    """Split numbered model blocks; keep each attached LoRA whole by default."""

    blocks_prefix: str
    lora_partitioning: LoRAPartitioning = LoRAPartitioning.WHOLE

    def __call__(self, specs: Sequence[PatchedUnitSpec]) -> tuple[PatchedUnitSpec, ...]:
        if not self.blocks_prefix:
            raise ValueError("blocks_prefix must be non-empty")
        prefix = f"{self.blocks_prefix}."
        result: list[PatchedUnitSpec] = []
        for spec in specs:
            unit = spec.clean.unit
            blocks: OrderedDict[int, dict[str, tuple[torch.Size, torch.dtype]]] = OrderedDict()
            block_groups: dict[int, list[ModelKeyGroup]] = {}
            remainder: dict[str, tuple[torch.Size, torch.dtype]] = {}
            remainder_groups: list[ModelKeyGroup] | None = [] if spec.model_key_groups is not None else None
            for group in _model_key_groups(spec):
                members = _members(group)
                index = _block_index(group, prefix)
                if index is None:
                    target = remainder
                    if remainder_groups is not None:
                        remainder_groups.append(group)
                else:
                    target = blocks.setdefault(index, {})
                    if spec.model_key_groups is not None:
                        block_groups.setdefault(index, []).append(group)
                target.update((member, unit.metadata[member]) for member in members)
            stem = "" if unit.unit_id == "model" else f"{unit.unit_id}:"
            for index, values in sorted(blocks.items()):
                part = StreamedUnit(f"{stem}block:{index}", values, trigger_path=f"{self.blocks_prefix}.{index}")
                result.append(
                    _child(
                        spec,
                        part,
                        block_groups.get(index) if spec.model_key_groups is not None else None,
                        self.lora_partitioning,
                    )
                )
            if remainder:
                result.append(
                    _child(spec, MetadataUnit(f"{stem}non-block", remainder), remainder_groups, self.lora_partitioning)
                )
        return tuple(result)


@dataclass(frozen=True, slots=True)
class BlockPartPartitioner(Partitioner):
    """Split each block by predicates while carrying whole LoRAs by default."""

    blocks_prefix: str
    parts: Mapping[str, Callable[[str], bool]]
    shared: Collection[str] = ()
    triggers: Mapping[str, str] | None = None
    lora_partitioning: LoRAPartitioning = LoRAPartitioning.WHOLE

    def __call__(self, specs: Sequence[PatchedUnitSpec]) -> tuple[PatchedUnitSpec, ...]:
        if len(self.parts) < 2 or any(not name for name in self.parts):
            raise ValueError("parts must contain at least two names")
        prefix = f"{self.blocks_prefix}."
        result: list[PatchedUnitSpec] = []
        for spec in specs:
            unit = spec.clean.unit
            if not unit.unit_id.startswith("block:"):
                result.append(spec)
                continue
            selected: dict[str, dict[str, tuple[torch.Size, torch.dtype]]] = {name: {} for name in self.parts}
            selected_groups: dict[str, list[ModelKeyGroup]] | None = (
                {name: [] for name in self.parts} if spec.model_key_groups is not None else None
            )
            for group in _model_key_groups(spec):
                members = _members(group)
                if any(not key.startswith(prefix) for key in members):
                    raise ValueError(f"{unit.unit_id!r} contains a non-block key")
                relative = [key[len(prefix) :].partition(".")[2] for key in members]
                owners = [name for name, predicate in self.parts.items() if all(predicate(key) for key in relative)]
                if not owners or (len(owners) > 1 and any(key not in self.shared for key in members)):
                    raise ValueError(f"fusion group {group!r} has {len(owners)} part owners")
                for name in owners:
                    selected[name].update((key, unit.metadata[key]) for key in members)
                    if selected_groups is not None:
                        selected_groups[name].append(group)
            populated = [(name, values) for name, values in selected.items() if values]
            if len(populated) == 1:
                result.append(spec)
                continue
            for name, values in populated:
                path = (
                    self.triggers[name]
                    if self.triggers is not None
                    else (unit.trigger_path if isinstance(unit, StreamedUnit) else None)
                )
                part = (
                    StreamedUnit(f"{unit.unit_id}:{name}", values, trigger_path=path)
                    if path is not None
                    else MetadataUnit(f"{unit.unit_id}:{name}", values)
                )
                result.append(
                    _child(
                        spec,
                        part,
                        selected_groups[name] if selected_groups is not None else None,
                        self.lora_partitioning,
                    )
                )
        return tuple(result)


def _rank_groups(spec: PatchedUnitSpec) -> Iterable[tuple[ModelKeyGroup, ...]]:
    """Coalesce model-key groups where checkpoint results share inputs."""
    connected = spec.clean.load_mapping.tensor_mapping.connected_destinations
    if not connected:
        for group in _model_key_groups(spec):
            yield (group,)
        return
    model_groups = tuple(_model_key_groups(spec))
    key_to_group = {key: index for index, group in enumerate(model_groups) for key in _members(group)}
    parent = list(range(len(model_groups)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for destinations in connected:
        indices = [key_to_group[key] for key in destinations if key in key_to_group]
        if indices:
            first = root(indices[0])
            for index in indices[1:]:
                parent[root(index)] = first
    merged: dict[int, list[ModelKeyGroup]] = {}
    for index, group in enumerate(model_groups):
        merged.setdefault(root(index), []).append(group)
    for group in merged.values():
        yield tuple(group)


@dataclass(frozen=True, slots=True)
class RankPartitioner(Partitioner):
    """Assign checkpoint-connected model-key groups to ranks and split LoRAs with them."""

    world_size: int
    lora_partitioning: LoRAPartitioning = LoRAPartitioning.FOLLOW_MODEL

    def __call__(self, specs: Sequence[PatchedUnitSpec]) -> tuple[PatchedUnitSpec, ...]:
        if self.world_size <= 0:
            raise ValueError("world_size must be positive")
        result: list[PatchedUnitSpec] = []
        for spec in specs:
            metadata = spec.clean.unit.metadata
            by_rank: dict[int, dict[str, tuple[torch.Size, torch.dtype]]] = {}
            rank_groups: dict[int, list[ModelKeyGroup]] = {}
            ownership_groups = sorted(
                _rank_groups(spec),
                key=lambda connected_groups: sorted(key for group in connected_groups for key in _members(group))[0],
            )
            for connected_groups in ownership_groups:
                keys = sorted(key for group in connected_groups for key in _members(group))
                owner = int.from_bytes(hashlib.md5("\0".join(keys).encode()).digest()[:8], "little") % self.world_size
                by_rank.setdefault(owner, {}).update((key, metadata[key]) for key in keys)
                if spec.model_key_groups is not None:
                    rank_groups.setdefault(owner, []).extend(connected_groups)
            digest = hashlib.sha256("\0".join(sorted(set(spec.fuse_rule.requires_companions))).encode()).hexdigest()[
                :16
            ]
            stem = "" if spec.clean.unit.unit_id == "model" else f"{spec.clean.unit.unit_id}:"
            for owner, values in sorted(by_rank.items()):
                unit = RankUnit(f"{stem}rank:{owner}of{self.world_size}:{digest}", values, rank_id=owner)
                result.append(
                    _child(
                        spec,
                        unit,
                        rank_groups.get(owner) if spec.model_key_groups is not None else None,
                        self.lora_partitioning,
                    )
                )
        return tuple(result)


def fit_bytes(units: Sequence[TensorsUnit], limit: int) -> tuple[TensorsUnit, ...]:
    """Group small tensors and split large contiguous tensors into transfer views."""
    if limit <= 0:
        raise ValueError("byte limit must be positive")
    batches: list[TensorsUnit] = []
    current: dict[str, torch.Tensor] = {}
    size = 0
    seen: set[str] = set()

    def flush() -> None:
        nonlocal current, size
        if current:
            batches.append(TensorsUnit(f"batch:{len(batches)}", current))
            current, size = {}, 0

    for unit in units:
        for key, tensor in unit.tensors.items():
            if key in seen:
                raise ValueError(f"byte fitting has duplicate tensor key {key!r}")
            seen.add(key)
            if not tensor.is_contiguous():
                raise ValueError(f"{key!r} must be contiguous for byte fitting")
            if not tensor.numel():
                flush()
                batches.append(TensorsUnit(f"batch:{len(batches)}", {key: tensor}))
                continue
            byte_view = tensor.reshape(-1).view(torch.uint8)
            offset = 0
            while offset < byte_view.numel():
                if size == limit:
                    flush()
                take = min(limit - size, byte_view.numel() - offset)
                fragment = byte_view.narrow(0, offset, take)
                name = key if offset == 0 and take == byte_view.numel() else f"{key}@{offset}:{offset + take}"
                if name in current:
                    raise ValueError(f"byte fitting has duplicate fragment key {name!r}")
                current[name] = fragment
                size += take
                offset += take
    flush()
    return tuple(batches)


@dataclass(frozen=True, slots=True)
class FileReadPartitioner(Partitioner):
    """Split units by how their destinations are read from checkpoint files.
    Destinations outside the mapping's ``transformed_value_destinations`` form one ``FileReadUnit``: each holds the
    bytes of the single raw key it renames. The others keep their read groups (outputs sharing raw inputs) whole and
    are packed in metadata order into ``FileReadUnit``s with ``staging_required`` whose raw inputs take at most
    ``stage_bytes``; a larger group forms a unit of its own. Children keep the unit's LoRAs whole and its requested
    dtype.
    """

    stage_bytes: int = 1 << 30

    def __call__(self, specs: Sequence[PatchedUnitSpec]) -> tuple[PatchedUnitSpec, ...]:
        result: list[PatchedUnitSpec] = []
        for spec in specs:
            tensors = spec.clean.load_mapping.tensor_mapping
            uid = spec.clean.unit.unit_id
            copied: dict[str, TensorMetadata] = {}
            groups: dict[int | str, dict[str, TensorMetadata]] = {}
            for name, value in spec.clean.unit.metadata.items():
                if name not in tensors.transformed_value_destinations:
                    copied[name] = value
                else:
                    index = tensors._group_index.get(name)
                    groups.setdefault(name if index is None else index, {})[name] = value
            if copied:
                result.append(_child(spec, FileReadUnit(f"{uid}:copy", copied), None, LoRAPartitioning.WHOLE))
            batches: list[dict[str, TensorMetadata]] = []
            inputs: set[str] = set()
            for group in groups.values():
                group_inputs = _group_inputs(tensors, next(iter(group)))
                if not batches or _source_bytes(tensors, inputs | group_inputs) > self.stage_bytes:
                    batches.append({})
                    inputs = set()
                batches[-1].update(group)
                inputs |= group_inputs
            result.extend(
                _child(
                    spec, FileReadUnit(f"{uid}:stage:{n}", batch, staging_required=True), None, LoRAPartitioning.WHOLE
                )
                for n, batch in enumerate(batches)
            )
        return tuple(result)


def _read_group(tensors: TensorMapping, name: str) -> tuple[str, ...]:
    """The destinations sharing raw inputs with ``name``, which value operations produce together."""
    index = tensors._group_index.get(name)
    return (name,) if index is None else tensors.connected_destinations[index]


def _group_inputs(tensors: TensorMapping, name: str) -> set[str]:
    """The raw keys the value operations producing ``name``'s read group read."""
    return set().union(*(tensors.dependencies[member] for member in _read_group(tensors, name)))


def stage_inputs(tensors: TensorMapping, names: Iterable[str]) -> tuple[str, ...]:
    """The raw keys the value operations producing ``names`` read, sorted; whole read groups need all their inputs."""
    return tuple(sorted(set().union(*(_group_inputs(tensors, name) for name in names))))


def _source_bytes(tensors: TensorMapping, keys: set[str]) -> int:
    return sum(tensors.source_metadata[key][0].numel() * tensors.source_metadata[key][1].itemsize for key in keys)
