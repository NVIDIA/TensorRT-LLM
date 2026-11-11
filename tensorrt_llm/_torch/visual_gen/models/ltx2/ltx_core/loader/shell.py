"""Meta model shells and their immutable initial persistent-state layouts."""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import torch

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.helpers import module_tensors
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.mapping import validate_mapping
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.module_ops import ModuleOps, apply_module_ops
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.units import LoadMapping, TensorMetadata
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.model_protocol import ModelConfigurator


@dataclass(frozen=True, slots=True)
class ShellKey:
    """Identity of a cached shell's constructor, checkpoint configuration, and transforms.
    ``configurator`` is the actual class object, including any path-bound
    subclass. ``metadata_json`` is sorted checkpoint configuration JSON.
    ``module_ops_names`` preserves operation order; names must distinguish
    transforms that produce different shell structures.
    """

    configurator: type
    metadata_json: str
    module_ops_names: tuple[str, ...]


def shell_key(configurator: type, metadata: dict, operations: Sequence[ModuleOps]) -> ShellKey:
    """Build a cache key from configurator identity, metadata, and operation names."""
    return ShellKey(
        configurator,
        json.dumps(metadata, sort_keys=True, separators=(",", ":")),
        tuple(operation.name for operation in operations),
    )


@dataclass(frozen=True, slots=True)
class ShellEntry:
    """A meta shell and the persistent tensor layout it had at construction.
    ``shell`` may have a different Python class from the configurator's result
    after a module operation replaces the root. ``persistent_metadata`` is a
    read-only snapshot made before any later weight binding changes the shell.
    ``shared_names`` groups persistent names that reach one tensor slot of one
    module registered under several paths; binding any one name fills them all.
    """

    shell: torch.nn.Module
    persistent_metadata: Mapping[str, TensorMetadata]
    shared_names: tuple[tuple[str, ...], ...] = ()


def create_shell(
    configurator: type[ModelConfigurator[torch.nn.Module]],
    metadata: dict,
    operations: Sequence[ModuleOps] = (),
) -> ShellEntry:
    """Apply structural operations on a meta model, then snapshot its state layout."""
    with torch.device("meta"):
        shell = configurator.from_metadata(metadata)
        shell = apply_module_ops(shell, operations)
    persistent = {
        name: (torch.Size(value.shape), value.dtype) for name, value in shell.state_dict(keep_vars=True).items()
    }
    return ShellEntry(shell, MappingProxyType(persistent), _shared_names(shell, persistent))


def _shared_names(shell: torch.nn.Module, persistent: Collection[str]) -> tuple[tuple[str, ...], ...]:
    """Group persistent names by the (module object, slot) they bind, keeping groups of more than one."""
    slots: dict[tuple[int, str], list[str]] = {}
    for prefix, module in shell.named_modules(remove_duplicate=False):
        for slot in (*module._parameters, *module._buffers):
            name = f"{prefix}.{slot}" if prefix else slot
            if name in persistent:
                slots.setdefault((id(module), slot), []).append(name)
    return tuple(tuple(names) for names in slots.values() if len(names) > 1)


def validate_shell_projection(entry: ShellEntry, projection: LoadMapping) -> None:
    """Require checkpoint destinations to equal the shell's persistent state.
    A shared group needs one destination: binding it fills the slot every other name of the group reaches.
    """
    destinations = projection.tensor_mapping.destination_metadata
    expected = dict(entry.persistent_metadata)
    for names in entry.shared_names:
        if any(name in destinations for name in names):
            for name in names:
                if name not in destinations:
                    del expected[name]
    validate_mapping(projection, expected)


def materialize_nonpersistent_buffers(model: torch.nn.Module, device: torch.device) -> None:
    """Move generated nonpersistent buffers and reject ones left on meta."""
    for module in model.modules():
        for name in module._non_persistent_buffers_set:
            value = module._buffers.get(name)
            if value is None:
                continue
            if value.device.type == "meta":
                raise ValueError(f"nonpersistent buffer {name!r} remains on meta")
            if value.device != device:
                module._buffers[name] = value.to(device)


def validate_bound_model(model: torch.nn.Module, *, allowed_unbound: Collection[str] = ()) -> None:
    """Reject parameters or buffers left on meta unless their names are allowed."""
    if all(value.device.type != "meta" for value in module_tensors(model)):
        return
    permitted = set(allowed_unbound)
    uninitialized = [
        name
        for name, value in (*model.named_parameters(), *model.named_buffers())
        if value.device.type == "meta" and name not in permitted
    ]
    if uninitialized:
        raise ValueError(f"model has uninitialized meta parameters or buffers: {uninitialized}")
