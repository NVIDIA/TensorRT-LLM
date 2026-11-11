"""Prepare checkpoint provenance and fill selected model or LoRA tensors."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, MutableMapping, Sequence
from dataclasses import replace
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import TypeVar

import torch

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.buffers import unit_views
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.checkpoint import CheckpointReader, UnitSource
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.partitioners import FileReadPartitioner, apply_partitioners, stage_inputs
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.sd_ops import SDKeyValueOperation, SDOps
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.units import (
    FileReadUnit,
    LoadMapping,
    PatchedUnitSpec,
    StateDictMetadata,
    TensorMapping,
    TensorMetadata,
    UnitSpec,
    active_dtype,
)

_Source = TypeVar("_Source")
_CASTABLE_DTYPES = frozenset({torch.float16, torch.bfloat16, torch.float32, torch.float64})


_GroupId = tuple[LoadMapping, int | str]


def build_mapping(paths: Sequence[Path], sd_ops: Sequence[SDOps]) -> LoadMapping:
    """Build one full checkpoint recipe from configured shards and SDOps.
    Resolve paths in caller order, including shards with no final outputs.
    Scan raw tensor keys, shapes and dtypes and first-shard configuration
    without reading payloads. Abstract value operations determine destination
    keys, shapes and dtypes; dependencies and read groups are built once.
    """
    configured_paths = tuple(path.resolve() for path in paths)
    with CheckpointReader.open(configured_paths) as reader:
        checkpoint_metadata = reader.read_checkpoint_metadata()
        sources, source_metadata = reader.scan()
        offsets = reader.source_offsets()

    metadata: StateDictMetadata = dict(source_metadata)
    dependencies = {key: frozenset((key,)) for key in sources}
    strides: dict[str, tuple[int, ...]] = {}
    aliases: dict[str, str] = {}
    transformed: frozenset[str] = frozenset()
    operations = tuple(sd_ops)
    for operation in operations:
        metadata, dependencies, strides, aliases, transformed = _project_stage(
            metadata, dependencies, strides, aliases, transformed, operation
        )

    used_sources = {source for inputs in dependencies.values() for source in inputs}
    retained_sources: dict[str, Path] = {}
    retained_metadata: dict[str, TensorMetadata] = {}
    retained_offsets: dict[str, int] = {}
    for key, path in sources.items():
        if key in used_sources:
            retained_sources[key] = path
            retained_metadata[key] = source_metadata[key]
            retained_offsets[key] = offsets[key]
    groups, group_index = _connected_destinations(dependencies) if _has_value_operations(operations) else ((), {})
    tensors = TensorMapping(
        MappingProxyType(retained_sources),
        MappingProxyType(retained_metadata),
        MappingProxyType(dict(metadata)),
        MappingProxyType(dict(dependencies)),
        groups,
        MappingProxyType(group_index),
        MappingProxyType(strides),
        MappingProxyType(aliases),
        source_offsets=MappingProxyType(retained_offsets),
        transformed_value_destinations=transformed,
    )
    return LoadMapping(configured_paths, operations, MappingProxyType(checkpoint_metadata), tensors)


def validate_mapping(mapping: LoadMapping, expected_metadata: Mapping[str, TensorMetadata]) -> None:
    """Require final mapping keys to equal model persistent state keys.
    Reject missing and orphan keys together. Shapes must match; supported
    floating dtype differences are cast when filling destination storage.
    """
    destinations = mapping.tensor_mapping.destination_metadata
    missing = sorted(expected_metadata.keys() - destinations.keys())
    orphan = sorted(destinations.keys() - expected_metadata.keys())
    if missing or orphan:
        raise ValueError(f"checkpoint/model destination keys disagree; missing: {missing}, orphan: {orphan}")
    for key, (shape, dtype) in expected_metadata.items():
        destination_shape, destination_dtype = destinations[key]
        if destination_shape != shape:
            raise ValueError(
                f"mapping destination {key!r} has shape {tuple(destination_shape)}, expected {tuple(shape)}"
            )
        if not _can_copy_as(shape, destination_dtype, dtype):
            raise ValueError(
                f"mapping destination {key!r} has dtype {destination_dtype}, which cannot be copied as {dtype}"
            )


class ExecutionCache:
    """Keep shared read-group outputs until their registered unit uses finish.
    Only transformed groups crossing unit boundaries get a cache record, and only units with such a group are
    registered: ``tracks`` tells them apart, and loaders hand the cache to those units alone. The cache reads
    nothing: the first unit to fill a shared group produces its outputs and keeps them here, later units take them
    from here, and each use is released once its unit is filled. Pass only units that read their checkpoint: a unit
    taken from the registry would hold its uses open. The cache belongs to one build or streaming pass.
    """

    @classmethod
    def if_shared(cls, specs: Sequence[UnitSpec]) -> ExecutionCache | None:
        """Create a cache only when selected units share a value-operation output."""
        cache = cls(specs)
        return cache if cache._tracked else None

    def __init__(self, specs: Sequence[UnitSpec]) -> None:
        first_owner: dict[_GroupId, UnitSpec] = {}
        shared: set[_GroupId] = set()
        for spec in specs:
            seen: set[_GroupId] = set()
            for key in spec.unit.destinations:
                identity = _shareable_group_identity(spec.load_mapping, key)
                if identity is None or identity in seen:
                    continue
                seen.add(identity)
                if first_owner.setdefault(identity, spec) is not spec:
                    shared.add(identity)
        self._remaining = {identity: Counter[str]() for identity in shared}
        self._outputs: dict[_GroupId, dict[str, torch.Tensor]] = {}
        self._tracked: set[UnitSpec] = set()
        for spec in specs:
            for key in spec.unit.destinations:
                identity = _shareable_group_identity(spec.load_mapping, key)
                if identity in self._remaining:
                    self._remaining[identity][key] += 1
                    self._tracked.add(spec)

    def tracks(self, spec: UnitSpec) -> bool:
        """Whether ``spec`` has a destination in a group shared with another unit of this pass."""
        return spec in self._tracked

    def _identity(self, mapping: LoadMapping, key: str) -> _GroupId | None:
        identity = _shareable_group_identity(mapping, key)
        return identity if identity in self._remaining else None

    def holds(self, mapping: LoadMapping, key: str) -> bool:
        """Whether another unit already produced the outputs of ``key``'s shared group."""
        identity = self._identity(mapping, key)
        return identity is not None and identity in self._outputs

    def _keep(self, identity: _GroupId, outputs: Mapping[str, torch.Tensor]) -> None:
        self._outputs[identity] = {name: value for name, value in outputs.items() if name in self._remaining[identity]}

    def _consume(self, mapping: LoadMapping, key: str) -> None:
        """Release one use and discard an output after its final consumer."""
        identity = self._identity(mapping, key)
        if identity is None or self._remaining[identity].get(key, 0) <= 0:
            raise ValueError(f"unregistered or repeated destination use: {key!r}")
        self._remaining[identity][key] -= 1
        if self._remaining[identity][key] == 0:
            del self._remaining[identity][key]
            if identity in self._outputs:
                del self._outputs[identity][key]
                if not self._outputs[identity]:
                    del self._outputs[identity]


def project(  # noqa: PLR0912
    produce: Callable[[LoadMapping, tuple[str, ...], _Source, torch.device | None], Mapping[str, torch.Tensor]],
    spec: UnitSpec,
    source: _Source,
    *,
    destination: MutableMapping[str, torch.Tensor],
    device: torch.device | None = None,
    requested_dtype: torch.dtype | None = None,
    cache: ExecutionCache | None = None,
) -> None:
    """Fill selected tensors read group by read group with ``produce(mapping, group, source, device)``.
    ``produce`` returns a read group's outputs, on ``device`` when it is CUDA. A group shared through ``cache`` is
    taken from it once another unit produced it, and kept there for the others otherwise. Existing selected tensors
    are validated and copied into; missing ones are allocated on ``device`` when supplied, otherwise on the produced
    tensor's device. The dictionary may contain unrelated entries. With ``requested_dtype`` each tensor is filled at
    its ``active_dtype``; the spec keeps describing the clean projected tensors.
    """
    validate_destination(spec, destination, requested_dtype)
    mapping = spec.load_mapping
    tensors = mapping.tensor_mapping
    selected = spec.unit.metadata
    seen: set[int] = set()
    consumed: list[str] = []
    supplied = frozenset(destination)
    with torch.no_grad():
        for key in selected:
            group_index = tensors._group_index.get(key)
            if group_index is not None:
                if group_index in seen:
                    continue
                seen.add(group_index)
            group = (key,) if group_index is None else tensors.connected_destinations[group_index]
            identity = cache._identity(mapping, key) if cache is not None else None
            outputs = cache._outputs.get(identity) if identity is not None else None
            if outputs is None:
                group_device = device
                if group_device is None:
                    group_device = next((destination[name].device for name in group if name in destination), None)
                outputs = produce(mapping, group, source, group_device)
                if identity is not None:
                    cache._keep(identity, outputs)
            for name, value in outputs.items():
                if name not in selected:
                    continue
                target_device = device if device is not None else value.device
                target_dtype = active_dtype(selected[name], requested_dtype)
                alias = tensors.destination_aliases.get(name)
                if alias in outputs and outputs[alias] is not value:
                    raise ValueError(f"destination {name!r} disagrees with projected alias {alias!r}")
                if (
                    alias is not None
                    and alias in outputs
                    and name not in destination
                    and alias not in supplied
                    and alias in destination
                    and destination[alias].device == target_device
                    and destination[alias].dtype == target_dtype
                ):
                    destination[name] = destination[alias]
                else:
                    _fill_output(destination, name, value, target_dtype, target_device)
                if identity is not None:
                    consumed.append(name)
    if cache is not None:
        for name in consumed:
            cache._consume(mapping, name)


def transform_group(
    mapping: LoadMapping, group: tuple[str, ...], inputs: Mapping[str, torch.Tensor], device: torch.device | None = None
) -> dict[str, torch.Tensor]:
    """Run the SDOps on a read group's raw inputs, already in memory, on ``device`` when it is CUDA.
    ``inputs`` holds at least the group's raw keys. They move to the GPU before the first operation, so transforms
    such as dtype casts never run on the host.
    """
    tensors = mapping.tensor_mapping
    needed = set().union(*(tensors.dependencies[key] for key in group))
    transform_device = _transform_device(device)
    values: dict[str, torch.Tensor] = {}
    for key in sorted(needed):
        value = inputs[key]
        _validate_metadata("checkpoint source", key, value, tensors.source_metadata[key])
        values[key] = value if transform_device is None else value.to(transform_device)
    for operation in mapping.sd_ops:
        values = _materialize_stage(values, operation)
    missing = set(group) - values.keys()
    unexpected = values.keys() - set(group)
    if missing or unexpected:
        raise ValueError(
            f"materialized destinations disagree with projection; missing: {sorted(missing)}, "
            f"unexpected: {sorted(unexpected)}"
        )
    for key in group:
        _validate_metadata("destination", key, values[key], tensors.destination_metadata[key])
    return values


transform = partial(project, transform_group)
"""Fill selected tensors by running the SDOps on raw inputs already in memory: ``transform(spec, inputs, ...)``.
``inputs`` holds, by raw key, every checkpoint tensor the selection's read groups depend on, except those of groups
``cache`` already holds; see ``project``.
"""


def file_read_parts(spec: PatchedUnitSpec) -> tuple[PatchedUnitSpec, ...]:
    """Split a unit's clean state, without its LoRAs, by how ``FileReadPartitioner`` reads it from files."""
    return apply_partitioners(replace(spec, loras=()) if spec.loras else spec, (FileReadPartitioner(),))


def fill_from_source(
    parts: Sequence[PatchedUnitSpec],
    source: UnitSource,
    *,
    destination: MutableMapping[str, torch.Tensor],
    device: torch.device | None = None,
    cache: ExecutionCache | None = None,
    staging: torch.Tensor | None = None,
) -> None:
    """Fill one unit's destinations at their requested dtype from its file-read parts (``file_read_parts``).
    Callers split the unit and validate storage they did not allocate from its spec (``validate_destination``).
    A copy part's destination tensors are filled by raw key through ``source.read_into``; missing ones are first
    allocated on ``device`` (default CPU) at their active dtype. A stage part reads its raw inputs the same way, onto
    a CUDA ``device``, into ``staging`` (host bytes the caller reuses for every unit it fills) or into new host
    memory, and ``transform`` runs its value operations on them; groups another unit already produced come from
    ``cache`` without reads.
    """
    for part in parts:
        clean = part.clean
        tensors = clean.load_mapping.tensor_mapping
        if isinstance(clean.unit, FileReadUnit) and not clean.unit.staging_required:
            targets = {}
            for name, metadata in clean.unit.metadata.items():
                if name not in destination:
                    dtype = active_dtype(metadata, part.requested_dtype)
                    destination[name] = torch.empty(metadata[0], dtype=dtype, device=device or "cpu")
                (key,) = tensors.dependencies[name]
                targets[key] = destination[name]
            source.read_into(tensors, targets)
            continue
        names = [name for name in clean.unit.metadata if cache is None or not cache.holds(clean.load_mapping, name)]
        stored = {key: tensors.source_metadata[key] for key in stage_inputs(tensors, names)}
        on_cuda = device if device is not None and device.type == "cuda" else None
        if staging is not None and on_cuda is None and cache is None:
            inputs = unit_views(staging, stored)
        else:  # kept shared outputs may alias their inputs, which therefore never live in reused staging bytes
            inputs = {key: torch.empty(shape, dtype=dtype, device=on_cuda) for key, (shape, dtype) in stored.items()}
        if inputs:
            source.read_into(tensors, inputs)
        transform(
            clean, inputs, destination=destination, device=device, requested_dtype=part.requested_dtype, cache=cache
        )


def validate_destination(
    spec: UnitSpec, destination: Mapping[str, torch.Tensor], requested_dtype: torch.dtype | None
) -> None:
    """Check a unit's selection against the mapping and the supplied storage of its destinations."""
    tensors = spec.load_mapping.tensor_mapping
    selected = spec.unit.metadata
    unknown = selected.keys() - tensors.destination_metadata.keys()
    if unknown:
        raise ValueError(f"unit selects unknown destinations: {sorted(unknown)}")
    for key, (shape, dtype) in selected.items():
        if shape != tensors.destination_metadata[key][0]:
            raise ValueError(f"unit destination {key!r} disagrees with checkpoint shape")
        if key not in destination:
            continue
        target = destination[key]
        if target.shape != shape:
            raise ValueError(f"destination storage {key!r} has shape {tuple(target.shape)}, expected {tuple(shape)}")
        source_dtype = tensors.destination_metadata[key][1]
        if not _can_copy_as(shape, source_dtype, target.dtype) or target.dtype != active_dtype(
            (shape, dtype), requested_dtype
        ):
            raise ValueError(
                f"destination storage {key!r} has dtype {target.dtype}, which cannot receive projected {source_dtype}"
            )


def _can_copy_as(shape: torch.Size, source: torch.dtype, destination: torch.dtype) -> bool:
    if source == destination:
        return True
    if not shape and source == torch.float32:
        return False
    return source in _CASTABLE_DTYPES and destination in _CASTABLE_DTYPES


def _connected_destinations(
    dependencies: Mapping[str, frozenset[str]],
) -> tuple[tuple[tuple[str, ...], ...], dict[str, int]]:
    """Index only destination groups that share at least one raw checkpoint key."""
    parent = {key: key for key in dependencies}

    def root(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    first_by_source: dict[str, str] = {}
    for destination, sources in dependencies.items():
        for source in sources:
            first = first_by_source.setdefault(source, destination)
            parent[root(destination)] = root(first)

    members: dict[str, list[str]] = defaultdict(list)
    for key in dependencies:
        members[root(key)].append(key)
    groups = tuple(tuple(keys) for keys in members.values() if len(keys) > 1)
    index = {key: number for number, group in enumerate(groups) for key in group}
    return groups, index


def _project_stage(  # noqa: PLR0912, PLR0915
    metadata: StateDictMetadata,
    dependencies: dict[str, frozenset[str]],
    strides: dict[str, tuple[int, ...]],
    aliases: dict[str, str],
    transformed: frozenset[str],
    operation: SDOps,
) -> tuple[
    dict[str, TensorMetadata],
    dict[str, frozenset[str]],
    dict[str, tuple[int, ...]],
    dict[str, str],
    frozenset[str],
]:
    renamed = _renamed_keys(metadata, operation)
    renamed_transformed = frozenset(new_key for key, new_key in renamed.items() if key in transformed)
    renamed_metadata = {new_key: metadata[key] for key, new_key in renamed.items()}
    renamed_dependencies = {new_key: dependencies[key] for key, new_key in renamed.items()}
    renamed_strides = {new_key: strides[key] for key, new_key in renamed.items() if key in strides}
    renamed_roots = {new_key: aliases.get(key, key) for key, new_key in renamed.items()}
    first_renamed: dict[str, str] = {}
    renamed_aliases: dict[str, str] = {}
    for name, root in renamed_roots.items():
        first = first_renamed.setdefault(root, name)
        if first != name:
            renamed_aliases[name] = first
    if not _has_value_operations((operation,)):
        return renamed_metadata, renamed_dependencies, renamed_strides, renamed_aliases, renamed_transformed
    projected_transformed: set[str] = set()
    projected_metadata: dict[str, TensorMetadata] = {}
    projected_dependencies: dict[str, frozenset[str]] = {}
    projected_strides: dict[str, tuple[int, ...]] = {}
    projected_aliases: dict[str, str] = {}
    first_output: dict[tuple[str, str | int], str] = {}
    meta_inputs: dict[str, torch.Tensor] = {}
    input_roots_by_id: dict[int, str] = {}
    output_references: list[torch.Tensor] = []
    for keys in operation.grouped(renamed_metadata):
        if isinstance(keys, str):
            primary = keys
            if primary in projected_metadata:
                raise ValueError(f"{operation.name}: several operations emit {primary!r}")
            projected_metadata[primary] = renamed_metadata[primary]
            projected_dependencies[primary] = renamed_dependencies[primary]
            if primary in renamed_transformed:
                projected_transformed.add(primary)
            if primary in renamed_strides:
                projected_strides[primary] = renamed_strides[primary]
            identity = ("input", renamed_roots[primary])
            first = first_output.setdefault(identity, primary)
            if first != primary:
                projected_aliases[primary] = first
            continue
        primary = keys[0]
        inputs: dict[str, torch.Tensor] = {}
        for key in keys:
            root = renamed_roots[key]
            value = meta_inputs.get(root)
            if value is None:
                shape, dtype = renamed_metadata[key]
                stride = renamed_strides.get(key)
                value = (
                    torch.empty(shape, dtype=dtype, device="meta")
                    if stride is None
                    else torch.empty_strided(shape, stride, dtype=dtype, device="meta")
                )
                meta_inputs[root] = value
                input_roots_by_id[id(value)] = root
            inputs[key] = value
        sources = frozenset().union(*(renamed_dependencies[key] for key in keys))
        for name, value in operation.apply_to_key_value_abstract(primary, inputs).items():
            if name in projected_metadata:
                raise ValueError(f"{operation.name}: several operations emit {name!r}")
            projected_metadata[name] = (value.shape, value.dtype)
            projected_dependencies[name] = sources
            projected_transformed.add(name)
            if not value.is_contiguous():
                projected_strides[name] = tuple(value.stride())
            output_references.append(value)
            identity = (
                ("input", input_roots_by_id[id(value)]) if id(value) in input_roots_by_id else ("output", id(value))
            )
            first = first_output.setdefault(identity, name)
            if first != name:
                projected_aliases[name] = first
    return (
        projected_metadata,
        projected_dependencies,
        projected_strides,
        projected_aliases,
        frozenset(projected_transformed),
    )


def _renamed_keys(keys: Iterable[str], operation: SDOps) -> dict[str, str]:
    renamed: dict[str, str] = {}
    used: set[str] = set()
    for key in keys:
        new_key = operation.apply_to_key(key)
        if new_key is None:
            continue
        if new_key in used:
            raise ValueError(f"{operation.name}: several keys are renamed to {new_key!r}")
        renamed[key] = new_key
        used.add(new_key)
    return renamed


def _materialize_stage(values: dict[str, torch.Tensor], operation: SDOps) -> dict[str, torch.Tensor]:
    renamed = _renamed_keys(values, operation)
    renamed_values = {new_key: values[key] for key, new_key in renamed.items()}
    if not _has_value_operations((operation,)):
        return renamed_values
    projected: dict[str, torch.Tensor] = {}
    for keys in operation.grouped(renamed_values):
        if isinstance(keys, str):
            primary = keys
            if primary in projected:
                raise ValueError(f"{operation.name}: several operations emit {primary!r}")
            projected[primary] = renamed_values[primary]
            continue
        primary = keys[0]
        inputs = {key: renamed_values[key] for key in keys}
        for name, value in operation.apply_to_key_value(primary, inputs).items():
            if name in projected:
                raise ValueError(f"{operation.name}: several operations emit {name!r}")
            projected[name] = value
    return projected


def _transform_device(target: torch.device | None) -> torch.device | None:
    """Run value operations on a CUDA target; other targets transform where their inputs are."""
    return target if target is not None and target.type == "cuda" else None


def _validate_metadata(kind: str, key: str, value: torch.Tensor, expected: TensorMetadata) -> None:
    shape, dtype = expected
    if value.shape != shape:
        raise ValueError(f"{kind} {key!r} has shape {tuple(value.shape)}, expected {tuple(shape)}")
    if value.dtype != dtype:
        raise ValueError(f"{kind} {key!r} has dtype {value.dtype}, expected {dtype}")


def _fill_output(
    destination: MutableMapping[str, torch.Tensor],
    name: str,
    source: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    """Copy into supplied storage or create an independent tensor with the source layout."""
    target = destination.get(name)
    if target is not None:
        target.copy_(source)
        return
    if not source.is_contiguous():
        target = torch.empty_strided(source.shape, source.stride(), dtype=dtype, device=device)
        target.copy_(source)
    else:
        target = source.to(device=device, dtype=dtype, copy=True, memory_format=torch.preserve_format)
    destination[name] = target


def _has_value_operations(operations: Sequence[SDOps]) -> bool:
    return any(isinstance(item, SDKeyValueOperation) for operation in operations for item in operation.mapping)


def _shareable_group_identity(mapping: LoadMapping, key: str) -> _GroupId | None:
    """A transformed destination's read group; other destinations hold their raw key's bytes, read where needed."""
    tensors = mapping.tensor_mapping
    if key not in tensors.transformed_value_destinations:
        return None
    index = tensors._group_index.get(key)
    return mapping, key if index is None else index
