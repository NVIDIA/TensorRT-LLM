"""Load and exchange rank-owned active model weights.
Ownership is fixed after checkpoint transforms, then agreed across ranks before
bucketed all-gather rounds copy owners' values into each rank's existing tensors.
"""

from __future__ import annotations

import hashlib
from collections.abc import Collection, Mapping, MutableMapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import TypeVar

import torch
import torch.distributed as dist

from tensorrt_llm._torch.visual_gen.models.ltx_core.loader.buffers import layout_bytes, unit_offsets
from tensorrt_llm._torch.visual_gen.models.ltx_core.loader.helpers import assign_state_dict
from tensorrt_llm._torch.visual_gen.models.ltx_core.loader.lora import patched_keys
from tensorrt_llm._torch.visual_gen.models.ltx_core.loader.partitioners import LoRAPartitioning, RankPartitioner, _select_loras, apply_partitioners
from tensorrt_llm._torch.visual_gen.models.ltx_core.loader.registry import ModelRegistry, RegistryKeys
from tensorrt_llm._torch.visual_gen.models.ltx_core.loader.shell import ShellKey, materialize_nonpersistent_buffers, validate_bound_model
from tensorrt_llm._torch.visual_gen.models.ltx_core.loader.simple_builder import (
    SimpleBuilder,
    _allocate_active,
    _check_destination,
    _fused,
    _on,
    _same_source,
    _target_device,
)
from tensorrt_llm._torch.visual_gen.models.ltx_core.loader.state_loader import StateLoader
from tensorrt_llm._torch.visual_gen.models.ltx_core.loader.units import LoadMapping, MetadataUnit, PatchedUnitSpec, RankUnit, UnitSpec, active_dtype

_Model = TypeVar("_Model", bound=torch.nn.Module)

# Staging segments start on a multiple of this many bytes, which keeps the multi-tensor copies' loads aligned.
_SEGMENT_ALIGN = 256


def build_distributed(  # noqa: PLR0912, PLR0915
    builder: SimpleBuilder[_Model],
    group: dist.ProcessGroup,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    *,
    dst: MutableMapping[str, torch.Tensor] | None = None,
    bucket_bytes: int = 256 << 20,
) -> _Model:
    """Build a shell over the whole active model, each rank loading only the units it owns.
    The ranks exchange their units so every rank holds the whole model. An exact repeat returns the shell as the
    registry recorded it. A filled ``dst``, which builders of several shells may share, is rewritten in place: an
    adapter switch rewrites and exchanges only the keys the held or the requested selection patches. Models bound
    to one ``dst`` must not run concurrently.
    """
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")
    if bucket_bytes <= 0:
        raise ValueError("bucket_bytes must be positive")
    target = _target_device(device)
    registry = builder.registry
    error = None
    try:
        builder._check_module_ops(target)
        load_mapping = registry.mapping(builder.paths, builder.sd_ops)
        metamodel = builder._metamodel(load_mapping)
        whole = builder._whole_model_unit(load_mapping, dtype)
        rank_units = _rank_units(builder, load_mapping, dtype, group)
        specs = _selected_specs(builder, rank_units, whole)
        ranks = tuple(spec.clean.unit for spec in specs)
        metamodel_key = builder._metamodel_key(load_mapping)
        state_dict = dst if dst is not None else {}
        if dst is None:
            active = registry.active_model(metamodel_key)
            current = None if active is None else active[1]
            on_target = active is not None and _on(active[0].parameters(), target)
        else:
            current = registry.held_spec(dst) if dst else None
            on_target = _on(dst.values(), target)
        ready = current is not None and _fused(current) == _fused(whole) and on_target
        held = (
            _selected_specs(builder, rank_units, current)
            if current is not None and state_dict and _same_source(current, whole)
            else None
        )
        metadata = load_mapping.tensor_mapping.destination_metadata
        strides = load_mapping.tensor_mapping.destination_strides
        layout = sorted(
            (key, tuple(clean[0]), str(active_dtype(clean, dtype)), strides.get(key)) for key, clean in metadata.items()
        )
        topology = (ownership_signature(ranks), tuple(layout))  # type: ignore[arg-type]
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    request = (
        tuple(str(Path(path).resolve()) for path in builder.paths),
        tuple(operation.name for operation in builder.sd_ops),
        tuple(operation.name for operation in builder.module_ops),
        target.type,
        str(dtype),
        builder.fuse_rule.name,
        tuple(
            (
                tuple(str(Path(path).resolve()) for path in lora.paths),
                lora.strength,
                tuple(operation.name for operation in lora.sd_ops),
            )
            for lora in builder.loras
        ),
        bucket_bytes,
        None if error is not None else (topology, bool(state_dict), held is not None),
    )
    ready_everywhere = _agree(group, "distributed build", request, error, ready=error is None and ready)
    shell = metamodel.shell  # every rank prepared without error
    if ready_everywhere:
        return shell if dst is None else _bind(registry, metamodel_key, whole, shell, dst, target)
    touched = (
        None
        if held is None
        else frozenset().union(*(patched_keys(old) | patched_keys(new) for old, new in zip(held, specs, strict=True)))
    )
    error = None
    shards: tuple[tuple[int, int, int], ...] = ()
    owner = staging = None
    try:
        if state_dict:
            _check_destination(state_dict, whole.clean.unit.metadata, dtype, target)
            registry.forget_active_models(state_dict)
        else:
            owner = _allocate_active(whole, target, state_dict)
        # Whole shards move only through an owner allocated here, whose layout every rank planned alike.
        shards = _shard_ranges(specs, whole) if touched is None and owner is not None else ()
        if not shards and (touched is None or touched):
            staging = registry.staging(target, bucket_bytes)
        rank = dist.get_rank(group)
        owned = [index for index, spec in enumerate(specs) if spec.clean.unit.rank_id == rank]
        if owned:
            StateLoader(builder.fuser, registry, builder.resolved_placement_policy).load(
                tuple(specs[index] for index in owned),
                target,
                state_dict,
                held=None if held is None else tuple(held[index] for index in owned),
            )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    _agree(group, "rank-local weights", request, error)
    if shards:
        assert owner is not None
        for source_rank, start, end in shards:
            dist.broadcast(
                owner.narrow(0, start, end - start), src=dist.get_global_rank(group, source_rank), group=group
            )
    elif touched is None or touched:
        exchange_active(
            ranks,
            state_dict,
            bucket_bytes,
            group=group,
            staging=staging,
            requested_dtype=dtype,
            keys=touched,
        )
    return _bind(
        registry, metamodel_key, whole, shell, dst if dst is not None else state_dict, target, fresh=dst is None
    )


def distributed_registry_keys(
    builder: SimpleBuilder[_Model], group: dist.ProcessGroup, dtype: torch.dtype | None = None
) -> RegistryKeys:
    """Return the registry keys this rank's ``build_distributed`` at ``dtype`` queries and retains."""
    load_mapping = builder.registry.mapping(builder.paths, builder.sd_ops)
    whole = builder._whole_model_unit(load_mapping, dtype)
    specs = _selected_specs(builder, _rank_units(builder, load_mapping, dtype, group), whole)
    rank = dist.get_rank(group)
    owned = tuple(spec for spec in specs if spec.clean.unit.rank_id == rank)
    state_loader = StateLoader(builder.fuser, builder.registry, builder.resolved_placement_policy)
    return (
        builder._metamodel_keys(load_mapping)
        | state_loader.registry_keys(owned)
        | builder._active_model_keys(load_mapping, dtype)
    )


def ownership_signature(ranks: Sequence[RankUnit]) -> tuple:
    """Compact ordered content for equality checks before collectives."""
    return tuple(
        (
            item.unit_id,
            item.rank_id,
            tuple((key, tuple(shape), str(dtype)) for key, (shape, dtype) in item.metadata.items()),
        )
        for item in ranks
    )


def _plan_rounds(keys: Sequence[str], sizes: Mapping[str, int], capacity: int) -> list[list[tuple[str, int, int]]]:
    """Pack ``keys`` into rounds of ``capacity`` bytes as ``(key, byte offset, byte count)`` chunks."""
    rounds: list[list[tuple[str, int, int]]] = []
    current: list[tuple[str, int, int]] = []
    used = 0
    for key in keys:
        offset, remaining = 0, sizes[key]
        while remaining:
            if used == capacity:
                rounds.append(current)
                current, used = [], 0
            chunk = min(remaining, capacity - used)
            current.append((key, offset, chunk))
            used += chunk
            offset += chunk
            remaining -= chunk
    if current:
        rounds.append(current)
    return rounds


def _bytes(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(-1).view(torch.uint8)


def _gather_rounds(
    plans: Sequence[list[list[tuple[str, int, int]]]],
    active: Mapping[str, torch.Tensor],
    staging: torch.Tensor,
    segment: int,
    group: dist.ProcessGroup | None,
) -> None:
    """Run every rank's planned rounds; a rank with fewer rounds sends an unused segment."""
    local_rank = dist.get_rank(group)
    send = staging[:segment]
    receive = staging[segment : segment * (len(plans) + 1)]
    for index in range(max(map(len, plans), default=0)):
        mine = plans[local_rank][index] if index < len(plans[local_rank]) else []
        if mine:
            packed, filled = [], 0
            for *_, chunk in mine:
                packed.append(send[filled : filled + chunk])
                filled += chunk
            torch._foreach_copy_(packed, [_bytes(active[key])[offset : offset + chunk] for key, offset, chunk in mine])
        dist.all_gather_into_tensor(receive, send, group=group)
        targets: list[torch.Tensor] = []
        sources: list[torch.Tensor] = []
        for owner, rounds in enumerate(plans):
            if owner == local_rank or index >= len(rounds):
                continue
            cursor = owner * segment
            for key, offset, chunk in rounds[index]:
                targets.append(_bytes(active[key])[offset : offset + chunk])
                sources.append(receive[cursor : cursor + chunk])
                cursor += chunk
        if targets:
            torch._foreach_copy_(targets, sources)


def _agree_on_exchange(  # noqa: PLR0912
    ranks: Sequence[RankUnit],
    active: Mapping[str, torch.Tensor],
    bucket_bytes: int,
    group: dist.ProcessGroup | None,
    staging: torch.Tensor | None,
    requested_dtype: torch.dtype | None,
    keys: Collection[str] | None,
) -> tuple[int, torch.Tensor]:
    """Validate this rank's exchange inputs and agree with every rank before any collective copy.
    Returns the segment size and the staging bucket. Raises on every rank when any rank's inputs are invalid or
    the ranks disagree about ownership, bucket size, active layout, or the exchanged keys.
    """
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")
    world = dist.get_world_size(group)
    if bucket_bytes <= 0 or any(item.rank_id >= world for item in ranks):
        raise ValueError("invalid bucket size or owner rank")
    metadata = {key: clean for item in ranks for key, clean in item.metadata.items()}
    invalid: list[str] = []
    if keys is not None and not set(keys) <= metadata.keys():
        invalid.append("exchanged keys are not active keys")
    for key, clean in metadata.items():
        tensor = active.get(key)
        if (
            tensor is None
            or tensor.shape != clean[0]
            or tensor.dtype != active_dtype(clean, requested_dtype)
            or not tensor.is_contiguous()
        ):
            invalid.append(key)
    first_device = next(iter(active.values())).device if active else None
    if any(tensor.device != first_device for tensor in active.values()):
        invalid.append("active tensors must share one device")
    segment = bucket_bytes // (world + 1)
    if segment >= _SEGMENT_ALIGN:
        segment = segment // _SEGMENT_ALIGN * _SEGMENT_ALIGN
    if segment == 0:
        invalid.append(f"a {bucket_bytes}-byte bucket has no segment for {world} ranks")
    if staging is not None and (
        staging.dtype != torch.uint8 or staging.numel() < bucket_bytes or staging.device != first_device
    ):
        invalid.append("exchange staging must be a device-matched uint8 bucket")
    if staging is None:
        try:
            staging = torch.empty(bucket_bytes, dtype=torch.uint8, device=first_device)
        except Exception as exc:
            invalid.append(f"exchange staging allocation failed: {type(exc).__name__}: {exc}")
    active_signature = tuple(
        sorted((key, tuple(clean[0]), str(active_dtype(clean, requested_dtype))) for key, clean in metadata.items())
    )
    selection = None if keys is None else hashlib.sha256("\0".join(sorted(keys)).encode()).hexdigest()
    signature = (ownership_signature(ranks), bucket_bytes, active_signature, selection, tuple(invalid))
    signatures: list[object] = [None] * world
    dist.all_gather_object(signatures, signature, group=group)
    if any(candidate[:4] != signature[:4] for candidate in signatures):
        raise ValueError("ranks disagree about ownership or collective order")
    if any(candidate[4] for candidate in signatures):
        raise ValueError(f"one or more ranks have incompatible active destinations: {signatures}")
    assert staging is not None
    return segment, staging


def exchange_active(
    ranks: Sequence[RankUnit],
    active: Mapping[str, torch.Tensor],
    bucket_bytes: int,
    *,
    group: dist.ProcessGroup | None = None,
    staging: torch.Tensor | None = None,
    requested_dtype: torch.dtype | None = None,
    keys: Collection[str] | None = None,
) -> None:
    """Copy owner data into each rank's existing full active allocation.
    ``bucket_bytes`` of staging hold one send segment and one receive segment per rank. In round
    ``i`` every rank packs its ``i``-th segment of owned bytes and one ``all_gather_into_tensor``
    ships all of them, so every link direction carries data at once; a rank with fewer
    rounds sends an unused segment. Tensors larger than a segment split across rounds. ``keys``
    restricts the exchange to a subset every rank passes identically. Active tensors have the
    ``active_dtype`` of the rank units' clean metadata at ``requested_dtype``.
    """
    segment, staging = _agree_on_exchange(ranks, active, bucket_bytes, group, staging, requested_dtype, keys)
    metadata = {key: clean for item in ranks for key, clean in item.metadata.items()}
    wanted = None if keys is None else frozenset(keys)
    sizes = {key: clean[0].numel() * active_dtype(clean, requested_dtype).itemsize for key, clean in metadata.items()}
    plans = [
        _plan_rounds(
            sorted(
                key
                for item in ranks
                if item.rank_id == owner
                for key in item.destinations
                if wanted is None or key in wanted
            ),
            sizes,
            segment,
        )
        for owner in range(dist.get_world_size(group))
    ]
    _gather_rounds(plans, active, staging, segment, group)


def _bind(
    registry: ModelRegistry,
    metamodel_key: ShellKey,
    whole_model_unit: PatchedUnitSpec,
    shell: _Model,
    state_dict: Mapping[str, torch.Tensor],
    device: torch.device,
    *,
    fresh: bool = False,
) -> _Model:
    """Bind the shell to the active weights unless the registry records it bound to them, and record it."""
    dst = None if fresh else state_dict
    if fresh or registry.active_model(metamodel_key, dst) is None:
        assign_state_dict(shell, state_dict)
        materialize_nonpersistent_buffers(shell, device)
        validate_bound_model(shell)
    registry.set_active_model(metamodel_key, whole_model_unit, dst)
    return shell


def _rank_units(
    builder: SimpleBuilder[_Model], load_mapping: LoadMapping, dtype: torch.dtype | None, group: dist.ProcessGroup
) -> tuple[PatchedUnitSpec, ...]:
    """Partition the whole model, without adapters, into units each owned by one rank."""
    whole_model_unit = builder._whole_model_unit(load_mapping, dtype, include_loras=False)
    units = apply_partitioners(whole_model_unit, (*builder.partitioners, RankPartitioner(dist.get_world_size(group))))
    if any(not isinstance(spec.clean.unit, RankUnit) for spec in units):
        raise ValueError("rank partition left a unit without an owner")
    return units


def _selected_specs(
    builder: SimpleBuilder[_Model], specs: tuple[PatchedUnitSpec, ...], whole_model_unit: PatchedUnitSpec
) -> tuple[PatchedUnitSpec, ...]:
    """Attach each rank unit's factors of the selected adapters without recomputing model ownership."""
    if not whole_model_unit.loras:
        return specs
    source = replace(specs[0], loras=whole_model_unit.loras)
    if builder.validate_binding:
        root = UnitSpec(MetadataUnit("model", whole_model_unit.clean.unit.metadata), specs[0].clean.load_mapping)
        PatchedUnitSpec(root, builder.fuse_rule, whole_model_unit.loras, validate_binding=True)
    return tuple(
        replace(spec, loras=_select_loras(source, spec.clean.unit, LoRAPartitioning.FOLLOW_MODEL)) for spec in specs
    )


def _shard_ranges(
    specs: tuple[PatchedUnitSpec, ...], whole_model_unit: PatchedUnitSpec
) -> tuple[tuple[int, int, int], ...]:
    """Find rank units whose active bytes have their complete local layout in one range of a host owner."""
    model_metadata = whole_model_unit.clean.unit.metadata
    strides = whole_model_unit.clean.load_mapping.tensor_mapping.destination_strides
    requested_dtype = whole_model_unit.requested_dtype
    offsets = unit_offsets(model_metadata, strides, requested_dtype)
    ranges = []
    for spec in specs:
        metadata = spec.clean.unit.metadata
        if any(model_metadata[key] != value for key, value in metadata.items()):
            return ()
        local = unit_offsets(metadata, strides, requested_dtype)
        size = layout_bytes(metadata, strides, requested_dtype)
        start = offsets[spec.clean.unit.destinations[0]]
        if any(offsets[key] - start != local[key] for key in metadata):
            return ()
        ranges.append((spec.clean.unit.rank_id, start, start + size))
    return tuple(ranges)


def _agree(
    group: dist.ProcessGroup, subject: str, value: object, error: str | None = None, *, ready: bool = True
) -> bool:
    """Make a rank-local preparation failure visible before the next collective; return whether all are ready."""
    gathered: list[tuple[object, str | None, bool] | None] = [None] * dist.get_world_size(group)
    dist.all_gather_object(gathered, (value, error, ready), group=group)
    if any(report is None or report[1] is not None for report in gathered):
        raise RuntimeError(
            f"ranks failed to prepare {subject}: {[report[1] if report else None for report in gathered]}"
        )
    if any(report is not None and report[0] != value for report in gathered):
        raise ValueError(f"ranks disagree about {subject}")
    return all(report is not None and report[2] for report in gathered)
