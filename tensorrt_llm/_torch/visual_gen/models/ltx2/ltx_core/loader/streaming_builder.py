"""Build streamed models from the same checkpoint recipe as ordinary builds."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.buffers import host_bytes
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.checkpoint import CheckpointReader
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.lora import LoRAFuser, patched_keys
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.mapping import ExecutionCache, file_read_parts, fill_from_source
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.partitioners import DenseBlockPartitioner, Partitioner, apply_partitioners
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.placement import Placement
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.registry import ModelRegistry, RegistryKeys
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.sd_ops import SDKeyValueOperation, SDOps
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.shell import (
    create_shell,
    materialize_nonpersistent_buffers,
    validate_bound_model,
    validate_shell_projection,
)
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.simple_builder import _target_device
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.state_entry import FusedKey, SourceKey, StateEntry
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.state_loader import StateLoader
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.streaming_delivery import (
    Readiness,
    active_nbytes,
    active_views,
    device_nbytes,
    host_buffer_nbytes,
    readiness_of,
)
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.streaming_model import StreamingModel
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.streaming_scheduler import BufferPool, StreamingScheduler
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.streaming_transport import make_transport
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.units import (
    LoadMapping,
    LoRAUnitSpec,
    MetadataUnit,
    PatchedUnitSpec,
    StreamedUnit,
    TensorsUnit,
    active_dtype,
)

if TYPE_CHECKING:
    from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.simple_builder import SimpleBuilder


@dataclass(frozen=True, slots=True)
class StreamingBuilder:
    """Compose a simple builder's checkpoint and adapter selection with streamed delivery.
    The streamed model owns its RAM-streamed host units and frees them on close; states
    outlive it only where the base builder's placement policy retains them.
    """

    base: SimpleBuilder
    config: StreamingConfig

    def build(
        self, device: torch.device | None = None, dtype: torch.dtype | None = None, **_kwargs: object
    ) -> StreamingModel:
        """Build resident weights once and choose registry, RAM, or disk delivery for each streamed unit.
        Each build has a fresh hooked shell; weights come from the base builder's registry and policy.
        """
        if dtype is None:
            raise ValueError("streaming requires an explicit dtype")
        base, config = self.base, self.config
        device = _streaming_device(device)
        base._check_module_ops(device)
        device_projection = config.mode == "disk" and _projects_on_device(base.sd_ops)
        if device_projection and device.type != "cuda":
            raise ValueError("CUDA checkpoint projection needs a CUDA streaming target")
        load_mapping = base.registry.mapping(base.paths, base.sd_ops)
        metamodel = create_shell(base.configurator, dict(load_mapping.checkpoint_metadata), base.module_ops)
        validate_shell_projection(metamodel, load_mapping)
        whole_model_unit = base._whole_model_unit(load_mapping, dtype)
        units = apply_partitioners(whole_model_unit, _partitioners(base, config))
        model = metamodel.shell
        streamed_units = _validate_units(model, units, config.blocks_prefix, config.device_buffers)
        resident_units = tuple(spec for spec in units if not isinstance(spec.clean.unit, StreamedUnit))
        retained = _registry_states(base.registry, streamed_units)
        state_loader = StateLoader(base.fuser, base.registry, base.resolved_placement_policy)
        host_placement = Placement(torch.device("cpu"), pin_memory=device.type == "cuda")
        loras = whole_model_unit.loras
        paths = tuple(
            dict.fromkeys(path for source in (whole_model_unit.clean, *loras) for path in source.load_mapping.paths)
        )
        with (
            torch.inference_mode(),
            torch.cuda.device(device) if device.type == "cuda" else nullcontext(),
            CheckpointReader.open(paths, stagers=base.registry.device_stager) as reader,
        ):
            needs_factors = any(
                spec.loras and base.registry.get_state(FusedKey.from_patched_unit_spec(spec)) is None for spec in units
            )
            factors = state_loader.load_factors(loras, device, reader) if loras and needs_factors else ()
            ram_units = _ram_fill_order(load_mapping, streamed_units) if config.mode == "cpu" else ()
            ram_units = tuple(
                spec for spec in ram_units if not _is_host_active(retained.get(_uid(spec)), host_placement)
            )
            cache_specs = (
                *(
                    spec.clean
                    for spec in ram_units
                    if _uid(spec) not in retained or state_loader.reads_checkpoint(spec)
                ),
                *(spec.clean for spec in resident_units if state_loader.reads_checkpoint(spec)),
            )
            cache = ExecutionCache.if_shared(cache_specs)
            # Fill RAM units before loading resident ones. Copying a checkpoint view (an mmap
            # of the file) straight to MPS leaves a hidden copy of the source pages, about the
            # size of the resident weights, alive until this reader closes; loading residents
            # last keeps that memory out of the RAM fill.
            for spec in ram_units:
                retained[_uid(spec)] = _fill_ram_unit(
                    spec,
                    retained.get(_uid(spec)),
                    _selected_factors(loras, factors, spec.loras),
                    reader,
                    cache,
                    state_loader,
                    device,
                    host_placement,
                )
            _load_residents(model, resident_units, loras, factors, reader, cache, state_loader, device)
            materialize_nonpersistent_buffers(model, device)
            validate_bound_model(
                model, allowed_unbound={key for spec in streamed_units for key in spec.clean.unit.metadata}
            )
        return _streaming_model(
            model, streamed_units, retained, loras, factors, base.fuser, device_projection, config, device
        )

    def registry_keys(self, dtype: torch.dtype | None = None) -> RegistryKeys:
        """Return the registry keys a build at ``dtype`` queries and retains.
        The streamed shell is built outside the registry. Every streamed unit's active and clean
        states are queried; RAM streaming retains active states where the placement policy keeps them.
        """
        if dtype is None:
            raise ValueError("streaming requires an explicit dtype")
        builder = self.base
        load_mapping = builder.registry.mapping(builder.paths, builder.sd_ops)
        whole_model_unit = builder._whole_model_unit(load_mapping, dtype)
        units = apply_partitioners(whole_model_unit, _partitioners(builder, self.config))
        placement_policy = builder.resolved_placement_policy
        state_loader = StateLoader(builder.fuser, builder.registry, placement_policy)
        resident_units = tuple(spec for spec in units if not isinstance(spec.clean.unit, StreamedUnit))
        keys = state_loader.registry_keys(resident_units, factors=False)
        if any(spec.loras for spec in units):
            keys |= state_loader.factor_keys(whole_model_unit.loras)
        streamed_units = tuple(spec for spec in units if isinstance(spec.clean.unit, StreamedUnit))
        lookup = frozenset(
            key
            for spec in streamed_units
            for key in (FusedKey.from_patched_unit_spec(spec), SourceKey.from_unit_spec(spec.clean))
        )
        retained = frozenset(
            FusedKey.from_patched_unit_spec(spec)
            for spec in streamed_units
            if self.config.mode == "cpu" and placement_policy.placement(spec).device is not None
        )
        return keys | RegistryKeys(lookup, retained)

    def model_config(self) -> dict[str, object]:
        """Read checkpoint configuration through the base registry's metadata cache."""
        return self.base.model_config()

    @property
    def keeps_gpu_resident_weights(self) -> bool:
        """Streamed weights move through device slots allocated per build."""
        return False

    @property
    def builds_per_shape(self) -> bool:
        """A built model serves any latent shape."""
        return False


@dataclass(frozen=True, slots=True)
class StreamingConfig:
    """Choose the streamed block path, backing mode, read-ahead depth, and buffer counts.
    Read-ahead may exceed the number of device buffers. Workers wait for a free
    device buffer while keeping later units queued for host reads.
    """

    blocks_prefix: str
    mode: str
    prefetch_depth: int = 2
    device_buffers: int = 2
    host_buffers: int = 2

    def __post_init__(self) -> None:
        if not self.blocks_prefix or self.mode not in {"cpu", "disk"}:
            raise ValueError("streaming needs a blocks_prefix and mode 'cpu' or 'disk'")
        if self.prefetch_depth < 0 or self.device_buffers < 1 or self.host_buffers < 1:
            raise ValueError("streaming buffer counts must be positive and prefetch depth nonnegative")


def _partitioners(builder: SimpleBuilder, config: StreamingConfig) -> tuple[Partitioner, ...]:
    return builder.partitioners or (DenseBlockPartitioner(config.blocks_prefix),)


def _validate_units(
    model: torch.nn.Module, specs: Sequence[PatchedUnitSpec], prefix: str, slots: int
) -> tuple[PatchedUnitSpec, ...]:
    """Check that streamed units cover every ordered block and fit nested triggers."""
    try:
        blocks = model.get_submodule(prefix)
    except AttributeError as error:
        raise ValueError(f"streaming block path {prefix!r} does not exist") from error
    if not isinstance(blocks, (torch.nn.ModuleList, torch.nn.Sequential)):
        raise ValueError(f"streaming block path {prefix!r} must be an ordered module list")
    transient = tuple(spec for spec in specs if isinstance(spec.clean.unit, StreamedUnit))
    if not transient:
        raise ValueError("streaming partition has no triggered units")
    ids = [spec.clean.unit.unit_id for spec in specs]
    if len(ids) != len(set(ids)):
        raise ValueError("streaming unit IDs must be unique")
    counts = Counter(key for spec in specs for key in spec.clean.unit.metadata)
    if any(count != 1 for count in counts.values()):
        raise ValueError("streaming units must own each destination exactly once")
    covered: set[int] = set()
    paths: list[str] = []
    for spec in transient:
        path = spec.clean.unit.trigger_path
        try:
            model.get_submodule(path)
        except AttributeError as error:
            raise ValueError(f"trigger path {path!r} does not exist") from error
        if not path.startswith(f"{prefix}."):
            raise ValueError(f"trigger path {path!r} is outside {prefix!r}")
        block_text = path[len(prefix) + 1 :].partition(".")[0]
        if not block_text.isdecimal() or int(block_text) >= len(blocks):
            raise ValueError(f"trigger path {path!r} is not in a valid block")
        covered.add(int(block_text))
        paths.append(path)
        if any(not key.startswith(f"{path}.") for key in spec.clean.unit.metadata):
            raise ValueError(f"unit {spec.clean.unit.unit_id!r} owns keys outside trigger {path!r}")
    if covered != set(range(len(blocks))):
        raise ValueError(f"streaming partition misses blocks {sorted(set(range(len(blocks))) - covered)}")
    nesting = max((sum(other == path or path.startswith(f"{other}.") for other in paths) for path in paths), default=1)
    if slots < nesting:
        raise ValueError(f"nested triggers need at least {nesting} device slots")
    return transient


def _allocate_buffers(
    units: Sequence[PatchedUnitSpec],
    retained: Mapping[str, StateEntry],
    transfer_at: Mapping[str, Readiness],
    config: StreamingConfig,
    prefetch_depth: int,
    device: torch.device,
) -> tuple[BufferPool, BufferPool]:
    """Size host and device buffers from what each unit's delivery needs."""
    readiness = [readiness_of(retained.get(_uid(spec))) for spec in units]
    staged = [
        host_buffer_nbytes(spec, state, transfer_at[_uid(spec)]) for spec, state in zip(units, readiness, strict=True)
    ]
    host_count = config.host_buffers + prefetch_depth if any(staged) else 0
    host = BufferPool(tuple(host_bytes(max(staged), pin=device.type == "cuda") for _ in range(host_count)))
    needed = max(
        device_nbytes(spec, state, transfer_at[_uid(spec)]) for spec, state in zip(units, readiness, strict=True)
    )
    buffers = tuple(torch.empty(needed, dtype=torch.uint8, device=device) for _ in range(config.device_buffers))
    return host, BufferPool(buffers)


def _streaming_device(device: torch.device | None) -> torch.device:
    """Select an available accelerator, falling back to CPU for a default build."""
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    return _target_device(device)


def _selected_factors(
    adapter_roots: Sequence[LoRAUnitSpec], loaded: Sequence[TensorsUnit], selected: Sequence[LoRAUnitSpec]
) -> tuple[TensorsUnit, ...]:
    """Match partitioned adapter selections to the root factor units in request order."""
    if not loaded:
        return ()
    result: list[TensorsUnit] = []
    start = 0
    for adapter in selected:
        for index in range(start, len(adapter_roots)):
            root = adapter_roots[index]
            if root.load_mapping is adapter.load_mapping and root.strength == adapter.strength:
                result.append(loaded[index])
                start = index + 1
                break
        else:
            raise ValueError("partitioned adapter is not in the ordered root selection")
    return tuple(result)


def _selection(spec: PatchedUnitSpec, keys: Sequence[str], suffix: str) -> PatchedUnitSpec:
    """The clean keys ``keys`` of ``spec`` as a unit of their own, without LoRAs, at the same requested dtype."""
    metadata = spec.clean.unit.metadata
    unit = MetadataUnit(f"{spec.clean.unit.unit_id}:{suffix}", {key: metadata[key] for key in keys})
    return replace(spec, clean=replace(spec.clean, unit=unit), loras=(), model_key_groups=None)


def _fill_patched_host_unit(
    spec: PatchedUnitSpec,
    values: Mapping[str, torch.Tensor],
    factors: Sequence[TensorsUnit],
    reader: CheckpointReader,
    cache: ExecutionCache | None,
    device: torch.device,
    fuser: LoRAFuser,
) -> None:
    """Fill a RAM-streamed unit's host storage, patching the weights its adapters bind on ``device``.
    Only bound weights, with the outputs sharing their checkpoint reads, visit the device. CUDA
    reads them straight from the checkpoint; elsewhere the unit is read into host storage first,
    since a checkpoint view copied to MPS keeps a hidden copy of its source pages. A weight the
    fuse rule cannot fuse in device memory stays on the host: its delta is computed on the device.
    """
    clean = spec.clean
    metadata = clean.unit.metadata
    strides = clean.load_mapping.tensor_mapping.destination_strides
    bound = set(patched_keys(spec))
    for group in clean.load_mapping.tensor_mapping.connected_destinations:
        if bound.intersection(group):
            bound.update(key for key in group if key in metadata)
    if not bound:
        fill_from_source(file_read_parts(spec), reader, destination=values, cache=cache)
        return
    if device.type == "cuda" and cache is None:
        unbound = [key for key in metadata if key not in bound]
        if unbound:  # adapters may bind every tensor of a unit
            fill_from_source(file_read_parts(_selection(spec, unbound, "unbound")), reader, destination=values)
        weights: dict[str, torch.Tensor] = {}
        for key, clean_metadata in metadata.items():
            if key not in bound:
                continue
            shape, dtype = clean_metadata[0], active_dtype(clean_metadata, spec.requested_dtype)
            weights[key] = (
                torch.empty_strided(shape, strides[key], dtype=dtype, device=device)
                if key in strides
                else torch.empty(shape, dtype=dtype, device=device)
            )
        fill_from_source(
            file_read_parts(_selection(spec, tuple(weights), "bound")), reader, destination=weights, device=device
        )
    else:
        fill_from_source(file_read_parts(spec), reader, destination=values, cache=cache)
        weights = {
            key: values[key].to(device) if spec.fuse_rule.fuses_on(device.type, values[key].dtype) else values[key]
            for key in bound
        }
        if fuser.apply_device is None:
            fuser = replace(fuser, apply_device=device)
    fused = fuser.patch(spec, weights, factors)
    with torch.no_grad():
        for key in bound:
            value = fused.get(key, weights[key])
            if value is not values[key]:
                values[key].copy_(value)


def _uid(spec: PatchedUnitSpec) -> str:
    return spec.clean.unit.unit_id


def _projects_on_device(sd_ops: Sequence[SDOps]) -> bool:
    """Whether a checkpoint operation keeps its outputs on the device its inputs are on."""
    return any(
        isinstance(operation, SDKeyValueOperation) and operation.kv_operation.preserve_output_device
        for stage in sd_ops
        for operation in stage.mapping
    )


def _registry_states(registry: ModelRegistry, units: Sequence[PatchedUnitSpec]) -> dict[str, StateEntry]:
    """Each unit's exact active state when the registry holds one, otherwise its clean state, if any."""
    states: dict[str, StateEntry] = {}
    for spec in units:
        entry = registry.get_state(FusedKey.from_patched_unit_spec(spec)) or registry.get_state(
            SourceKey.from_unit_spec(spec.clean)
        )
        if entry is not None:
            states[_uid(spec)] = entry
    return states


def _is_host_active(entry: StateEntry | None, host_placement: Placement) -> bool:
    return entry is not None and readiness_of(entry) is Readiness.ACTIVE and entry.placement == host_placement


def _ram_fill_order(load_mapping: LoadMapping, units: Sequence[PatchedUnitSpec]) -> tuple[PatchedUnitSpec, ...]:
    """Order units by their first checkpoint source, so a RAM fill reads the file front to back."""
    tensors = load_mapping.tensor_mapping
    position = {key: index for index, key in enumerate(tensors.sources)}
    return tuple(
        sorted(
            units,
            key=lambda spec: min(
                position[source] for name in spec.clean.unit.metadata for source in tensors.dependencies[name]
            ),
        )
    )


def _fill_ram_unit(
    spec: PatchedUnitSpec,
    entry: StateEntry | None,
    factors: Sequence[TensorsUnit],
    reader: CheckpointReader,
    cache: ExecutionCache | None,
    state_loader: StateLoader,
    device: torch.device,
    host_placement: Placement,
) -> StateEntry:
    """Fill one RAM-streamed unit's active tensors into an exact host buffer and keep it where the policy says.
    A unit without a registry state is projected from the checkpoint, patching bound weights on ``device``;
    a clean state, or an active one held elsewhere, goes through the state loader.
    """
    if cache is not None and not cache.tracks(spec.clean):
        cache = None  # the cache serves only units sharing one of its groups; this one reads on its own
    owner = host_bytes(active_nbytes(spec), pin=host_placement.pin_memory)
    values = active_views(owner, spec)
    if entry is None and not spec.loras:
        fill_from_source(file_read_parts(spec), reader, destination=values, cache=cache)
    elif entry is None:
        _fill_patched_host_unit(spec, values, factors, reader, cache, device, state_loader.fuser)
    else:
        # Patch LoRAs on the compute device, then retain the finished unit in host storage for repeated forwards.
        active: dict[str, torch.Tensor] = {}
        application_device = device if spec.loras else host_placement.device
        state_loader.load_unit(spec, application_device, active, factors, source=reader, cache=cache)
        for name, value in active.items():
            values[name].copy_(value)
    filled = StateEntry(spec, values, host_placement, owner)
    if state_loader.placement_policy.placement(spec).device is not None:
        # The policy keeps active states across builds; RAM streaming keeps them on the host.
        filled = state_loader.registry.adopt_entry(filled)
    return filled


def _load_residents(
    model: torch.nn.Module,
    units: Sequence[PatchedUnitSpec],
    loras: Sequence[LoRAUnitSpec],
    factors: Sequence[TensorsUnit],
    reader: CheckpointReader,
    cache: ExecutionCache | None,
    state_loader: StateLoader,
    device: torch.device,
) -> None:
    for spec in units:
        values: dict[str, torch.Tensor] = {}
        selected = _selected_factors(loras, factors, spec.loras)
        state_loader.load_unit(spec, device, values, selected, source=reader, cache=cache)
        # Residents are a small subset of a large streamed shell: bind them with a partial,
        # assigning load_state_dict and leave streamed keys to their delivery hooks.
        model.load_state_dict(values, strict=False, assign=True)


def _streaming_model(
    model: torch.nn.Module,
    units: Sequence[PatchedUnitSpec],
    retained: Mapping[str, StateEntry],
    loras: Sequence[LoRAUnitSpec],
    factors: Sequence[TensorsUnit],
    fuser: LoRAFuser,
    device_projection: bool,
    config: StreamingConfig,
    device: torch.device,
) -> StreamingModel:
    """Hook a scheduler over the streamed units into the model, routing each unit from where it starts."""
    # A unit crosses as checkpoint tensors when its SDOps run on the device, and as active tensors when the fuse
    # rule cannot fuse one of its weights in device memory.
    transfer_at: dict[str, Readiness] = {}
    for spec in units:
        metadata = spec.clean.unit.metadata
        host_fused = any(
            not spec.fuse_rule.fuses_on(device.type, active_dtype(metadata[key], spec.requested_dtype))
            for key in patched_keys(spec)
        )
        if host_fused and device_projection:
            raise ValueError(f"fuse rule {spec.fuse_rule.name!r} fuses on the host, but SDOps run on {device.type}")
        transfer_at[_uid(spec)] = (
            Readiness.ACTIVE if host_fused else Readiness.CHECKPOINT if device_projection else Readiness.CLEAN
        )
    unit_factors = {
        _uid(spec): _selected_factors(loras, factors, spec.loras)
        for spec in units
        if spec.loras and readiness_of(retained.get(_uid(spec))) is not Readiness.ACTIVE
    }
    prefetch_depth = min(config.prefetch_depth, len(units) - 1)
    host_pool, device_pool = _allocate_buffers(units, retained, transfer_at, config, prefetch_depth, device)
    # The build's reader closes when StreamingBuilder.build returns. The scheduler reads for the model's
    # lifetime through a second session, which the returned model closes after its scheduler.
    stack = ExitStack()
    try:
        scheduler = StreamingScheduler(
            units,
            CheckpointReader(
                tuple(dict.fromkeys(path for spec in units for path in spec.clean.load_mapping.paths)), stack
            ),
            make_transport(device),
            host_pool,
            device_pool,
            retained=retained,
            transfer_at=transfer_at,
            fuser=fuser if unit_factors else None,
            factors=unit_factors,
            prefetch_depth=prefetch_depth,
        )
        return StreamingModel(model, scheduler, stack)
    except BaseException:
        stack.close()
        raise
