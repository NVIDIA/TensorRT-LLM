"""Build and rebind a model shell from checkpoint-selected units."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, MutableMapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Generic, TypeVar

import torch

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.buffers import layout_bytes, unit_views
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.helpers import assign_state_dict
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.lora import FLOAT_FUSE_RULE, LoRAFuser, bind_lora_factors, whole_adapter_spec
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.module_ops import ModuleOps
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.partitioners import Partitioner, apply_partitioners
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.placement import PlacementPolicy, packs_storage
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.registry import ActiveModelKey, ModelRegistry, RegistryKeys
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.sd_ops import SDOps
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.shell import (
    ShellEntry,
    ShellKey,
    create_shell,
    materialize_nonpersistent_buffers,
    shell_key,
    validate_shell_projection,
)
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.state_entry import FusedKey
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.state_loader import StateLoader
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.units import (
    FuseRule,
    LoadMapping,
    LoRAUnitSpec,
    MetadataUnit,
    PatchedUnitSpec,
    StateDictMetadata,
    UnitSpec,
    active_dtype,
)
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.model_protocol import ModelConfigurator

_Model = TypeVar("_Model", bound=torch.nn.Module)


@dataclass(frozen=True, slots=True)
class LoRARequest:
    """Checkpoint shards, strength, and SDOps for one ordered adapter request."""

    paths: tuple[Path, ...]
    strength: float
    sd_ops: tuple[SDOps, ...] = ()


def _target_device(device: torch.device | None) -> torch.device:
    """Use the requested device, or the available CUDA, MPS, then CPU backend."""
    if device is None:
        if torch.cuda.is_available():
            target = torch.device("cuda")
        elif torch.backends.mps.is_available():
            target = torch.device("mps")
        else:
            target = torch.device("cpu")
    else:
        target = torch.device(device)
    if target.type == "cuda" and target.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    if target.type == "mps" and target.index is None:
        return torch.device("mps", 0)
    return target


def _check_destination(
    destination: Mapping[str, torch.Tensor],
    metadata: StateDictMetadata,
    requested_dtype: torch.dtype | None,
    device: torch.device,
) -> None:
    """Require caller storage for every clean ``metadata`` key at its active dtype on ``device``."""
    for key, clean in metadata.items():
        value = destination.get(key)
        if (
            value is None
            or value.shape != clean[0]
            or value.dtype != active_dtype(clean, requested_dtype)
            or value.device != device
        ):
            raise ValueError(f"active destination {key!r} disagrees with planned shape, dtype, or device")


def _on(tensors: Iterable[torch.Tensor], device: torch.device) -> bool:
    """Whether the first of ``tensors``, and so the allocation they belong to, is on ``device``."""
    first = next(iter(tensors), None)
    return first is not None and first.device == device


def _same_source(held: PatchedUnitSpec, requested: PatchedUnitSpec) -> bool:
    """Whether two selections patch the same clean model at the same requested dtype with the same fuse rule."""
    held_key, requested_key = _fused(held), _fused(requested)
    return (held_key.clean, held_key.requested_dtype, held.fuse_rule.name) == (
        requested_key.clean,
        requested_key.requested_dtype,
        requested.fuse_rule.name,
    )


def _allocate_active(
    whole_model_unit: PatchedUnitSpec, device: torch.device, state_dict: MutableMapping[str, torch.Tensor]
) -> torch.Tensor | None:
    """Fill ``state_dict`` with private storage for the active tensors and return the host byte owner, if any.
    Host storage is one owner with tensor views; device storage gets one block per tensor (``packs_storage``).
    """
    metadata = whole_model_unit.clean.unit.metadata
    strides = whole_model_unit.clean.load_mapping.tensor_mapping.destination_strides
    requested_dtype = whole_model_unit.requested_dtype
    if packs_storage(device):
        owner = torch.empty(layout_bytes(metadata, strides, requested_dtype), dtype=torch.uint8, device=device)
        state_dict.update(unit_views(owner, metadata, strides, requested_dtype))
        return owner
    for key, clean in metadata.items():
        dtype = active_dtype(clean, requested_dtype)
        state_dict[key] = (
            torch.empty_strided(clean[0], strides[key], dtype=dtype, device=device)
            if key in strides
            else torch.empty(clean[0], dtype=dtype, device=device)
        )
    return None


def _fused(spec: PatchedUnitSpec) -> FusedKey:
    return FusedKey.from_patched_unit_spec(spec)


@dataclass(frozen=True, slots=True)
class SimpleBuilder(Generic[_Model]):
    """Construct and rebind one cached shell from model and ordered LoRA checkpoints.
    A registry-cached shell is the same model object on later builds. Concurrent
    builds and two live versions of that shell are unsupported. An explicit
    placement policy overrides the registry's default retention policy.
    """

    configurator: type[ModelConfigurator[_Model]]
    paths: tuple[Path, ...]
    sd_ops: tuple[SDOps, ...] = ()
    module_ops: tuple[ModuleOps, ...] = ()
    partitioners: tuple[Partitioner, ...] = ()
    loras: tuple[LoRARequest, ...] = ()
    fuse_rule: FuseRule = FLOAT_FUSE_RULE
    fuser: LoRAFuser = field(default_factory=LoRAFuser)
    registry: ModelRegistry = field(default_factory=ModelRegistry)
    placement_policy: PlacementPolicy | None = None
    validate_binding: bool = False

    def build(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        dst: MutableMapping[str, torch.Tensor] | None = None,
        **_kwargs: object,
    ) -> _Model:
        """Load selected weights and ordered adapters into a cached or new shell.
        The registry records the selection each cached shell holds, so an exact repeat returns the shell as it is.
        An empty ``dst`` is filled; a filled one, which builds of several shells may share, is rewritten in place,
        only in the keys the held or the requested selection patches.
        """
        target_device = _target_device(device)
        self._check_module_ops(target_device)
        load_mapping = self.registry.mapping(self.paths, self.sd_ops)
        metamodel = self._metamodel(load_mapping)
        whole_model_unit = self._whole_model_unit(load_mapping, dtype)
        metamodel_key = self._metamodel_key(load_mapping)
        fused = _fused(whole_model_unit)
        if dst is None:
            active = self.registry.active_model(metamodel_key)
            if active is not None and _fused(active[1]) == fused and _on(active[0].parameters(), target_device):
                return active[0]  # type: ignore[return-value]
        state_dict = dst if dst is not None else {}
        held_spec = self.registry.held_spec(state_dict) if state_dict else None
        if held_spec is None or _fused(held_spec) != fused or not _on(state_dict.values(), target_device):
            if state_dict:
                _check_destination(state_dict, whole_model_unit.clean.unit.metadata, dtype, target_device)
                self.registry.forget_active_models(state_dict)
            elif dst is not None:
                # A destination later builds rewrite in place gets storage no retained state shares.
                _allocate_active(whole_model_unit, target_device, dst)
            held = (
                apply_partitioners(held_spec, self.partitioners)
                if held_spec is not None and _same_source(held_spec, whole_model_unit)
                else None
            )
            units = apply_partitioners(whole_model_unit, self.partitioners)
            StateLoader(self.fuser, self.registry, self.resolved_placement_policy).load(
                units, target_device, state_dict, held=held
            )
        if dst is None or self.registry.active_model(metamodel_key, dst) is None:
            assign_state_dict(metamodel.shell, state_dict)
            materialize_nonpersistent_buffers(metamodel.shell, target_device)
        self.registry.set_active_model(metamodel_key, whole_model_unit, dst)
        return metamodel.shell

    def registry_keys(self, dtype: torch.dtype | None = None) -> RegistryKeys:
        """Return the registry keys a build at ``dtype`` queries and retains."""
        load_mapping = self.registry.mapping(self.paths, self.sd_ops)
        units = apply_partitioners(self._whole_model_unit(load_mapping, dtype), self.partitioners)
        state_loader = StateLoader(self.fuser, self.registry, self.resolved_placement_policy)
        return (
            self._metamodel_keys(load_mapping)
            | state_loader.registry_keys(units)
            | self._active_model_keys(load_mapping, dtype)
        )

    def model_config(self) -> dict[str, object]:
        """Read checkpoint configuration through the registry's metadata cache."""
        return self.registry.checkpoint_metadata(self.paths).get("config", {})

    def with_sd_ops(self, operations: tuple[SDOps, ...]) -> SimpleBuilder[_Model]:
        """Return a builder with the given checkpoint operation chain."""
        return replace(self, sd_ops=operations)

    def with_module_ops(self, operations: tuple[ModuleOps, ...]) -> SimpleBuilder[_Model]:
        """Return a builder with the given shell operations."""
        return replace(self, module_ops=operations)

    def with_partitioner(self, partitioner: Partitioner) -> SimpleBuilder[_Model]:
        """Append a partitioner after the existing partitioners."""
        return replace(self, partitioners=(*self.partitioners, partitioner))

    def with_loras(self, requests: tuple[LoRARequest, ...]) -> SimpleBuilder[_Model]:
        """Replace the ordered adapter requests."""
        return replace(self, loras=requests)

    def lora(self, paths: tuple[Path, ...], strength: float, sd_ops: tuple[SDOps, ...] = ()) -> SimpleBuilder[_Model]:
        """Append one adapter request."""
        return replace(self, loras=(*self.loras, LoRARequest(paths, strength, sd_ops)))

    def with_fuser(self, fuser: LoRAFuser) -> SimpleBuilder[_Model]:
        """Return a builder using the supplied LoRA application and loading placement."""
        return replace(self, fuser=fuser)

    def with_fuse_rule(self, rule: FuseRule) -> SimpleBuilder[_Model]:
        """Return a builder grouping and patching model weights with this rule."""
        return replace(self, fuse_rule=rule)

    def with_registry(self, registry: ModelRegistry) -> SimpleBuilder[_Model]:
        """Return a builder sharing shell and weight retention through this registry."""
        if registry is None:
            raise ValueError("a builder needs a registry")
        return replace(self, registry=registry)

    def with_placement_policy(self, policy: PlacementPolicy) -> SimpleBuilder[_Model]:
        """Return a builder choosing retention placement through this policy."""
        return replace(self, placement_policy=policy)

    @property
    def keeps_gpu_resident_weights(self) -> bool:
        """Whether this builder guarantees stable GPU weight storage across builds."""
        return False

    @property
    def builds_per_shape(self) -> bool:
        """Whether a built model serves only the latent shape it was built for."""
        return False

    @property
    def resolved_placement_policy(self) -> PlacementPolicy:
        """The builder's placement policy, or its registry's default."""
        return self.placement_policy or self.registry.placement_policy

    def __post_init__(self) -> None:
        """Normalize checkpoint paths and require a registry."""
        object.__setattr__(self, "paths", tuple(Path(path) for path in self.paths))
        if not self.paths:
            raise ValueError("a model needs at least one checkpoint shard")
        if self.registry is None:
            raise ValueError("a builder needs a registry")

    def _adapters(self, model_metadata: StateDictMetadata) -> tuple[LoRAUnitSpec, ...]:
        adapters = []
        mappings: dict[tuple[tuple[Path, ...], tuple[str, ...]], LoadMapping] = {}
        for request in (request for request in self.loras if request.strength):
            prefix = (
                tuple(path.resolve() for path in request.paths),
                tuple(operation.name for operation in request.sd_ops),
            )
            mapping = mappings.get(prefix)
            if mapping is None:
                mapping = self.registry.mapping(request.paths, request.sd_ops)
                mappings[prefix] = mapping
            bindings = bind_lora_factors(mapping, model_metadata)
            selection_id = hashlib.sha256("\0".join(bindings).encode()).hexdigest()[:12]
            adapters.append(whole_adapter_spec(mapping, bindings, f"adapter:{selection_id}", strength=request.strength))
        return tuple(adapters)

    def _check_module_ops(self, device: torch.device) -> None:
        """Reject a target device that one of the shell operations cannot run on."""
        for operation in self.module_ops:
            if operation.required_device_type is not None and device.type != operation.required_device_type:
                raise ValueError(f"{operation.name} requires {operation.required_device_type.upper()}-resident weights")

    def _metamodel_key(self, load_mapping: LoadMapping) -> ShellKey:
        return shell_key(self.configurator, dict(load_mapping.checkpoint_metadata), self.module_ops)

    def _metamodel(self, load_mapping: LoadMapping) -> ShellEntry:
        """Return the registry's shell for this checkpoint and these shell operations, creating it once."""
        metamodel_key = self._metamodel_key(load_mapping)
        metamodel = self.registry.get_shell(metamodel_key)
        if metamodel is None:
            metamodel = self.registry.add_shell(
                metamodel_key, create_shell(self.configurator, dict(load_mapping.checkpoint_metadata), self.module_ops)
            )
        validate_shell_projection(metamodel, load_mapping)
        return metamodel

    def _metamodel_keys(self, load_mapping: LoadMapping) -> RegistryKeys:
        metamodel_key = self._metamodel_key(load_mapping)
        return RegistryKeys(
            frozenset({metamodel_key}), frozenset({metamodel_key}) if self.registry.cache_models else frozenset()
        )

    def _active_model_keys(self, load_mapping: LoadMapping, dtype: torch.dtype | None) -> RegistryKeys:
        """The active model a build looks up, and records when the registry caches its shell."""
        fused = FusedKey.from_patched_unit_spec(self._whole_model_unit(load_mapping, dtype))
        key = ActiveModelKey(fused, self._metamodel_key(load_mapping))
        return RegistryKeys(frozenset({key}), frozenset({key}) if self.registry.cache_models else frozenset())

    def _whole_model_unit(
        self, load_mapping: LoadMapping, dtype: torch.dtype | None, *, include_loras: bool = True
    ) -> PatchedUnitSpec:
        """Select every model tensor of the checkpoint, with the ordered LoRAs unless ``include_loras`` is off."""
        clean_model_unit = UnitSpec(
            MetadataUnit("model", load_mapping.tensor_mapping.destination_metadata), load_mapping
        )
        return PatchedUnitSpec(
            clean_model_unit,
            self.fuse_rule,
            self._adapters(clean_model_unit.unit.metadata) if include_loras else (),
            requested_dtype=dtype,
            validate_binding=self.validate_binding,
        )
