"""Retain unit states and model shells by their short keys."""

from __future__ import annotations

import copy
import threading
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from tensorrt_llm._torch.visual_gen.models.ltx_core.loader.buffers import host_bytes
from tensorrt_llm._torch.visual_gen.models.ltx_core.loader.checkpoint import CheckpointReader
from tensorrt_llm._torch.visual_gen.models.ltx_core.loader.mapping import build_mapping
from tensorrt_llm._torch.visual_gen.models.ltx_core.loader.placement import NO_RETENTION_POLICY, Placement, PlacementPolicy
from tensorrt_llm._torch.visual_gen.models.ltx_core.loader.raw import DeviceStager
from tensorrt_llm._torch.visual_gen.models.ltx_core.loader.sd_ops import SDOps
from tensorrt_llm._torch.visual_gen.models.ltx_core.loader.shell import ShellEntry, ShellKey
from tensorrt_llm._torch.visual_gen.models.ltx_core.loader.state_entry import (
    FusedKey,
    SourceKey,
    SourcePrefix,
    StateEntry,
    StateKey,
    selected_metadata,
)
from tensorrt_llm._torch.visual_gen.models.ltx_core.loader.units import UNIT_ALIGN, LoadMapping, PatchedUnitSpec, UnitSpec
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.disposable import Disposable


class ModelRegistry:
    """Cache model shells, the active models bound to them, checkpoint metadata, and retained unit states.
    ``cache_models`` keeps meta shells by structure. ``cache_metadata`` keeps parsed checkpoint
    metadata by resolved shard path and checkpoint projections by resolved paths and SDOps names.
    ``placement_policy`` is the default retention policy of builders using this registry; a
    builder may override it. States are retained only where a policy places them.
    """

    def __init__(
        self,
        *,
        cache_models: bool = True,
        cache_metadata: bool = True,
        placement_policy: PlacementPolicy = NO_RETENTION_POLICY,
    ) -> None:
        self.cache_models = cache_models
        self.cache_metadata = cache_metadata
        self.placement_policy = placement_policy
        self._states: dict[StateKey, StateEntry] = {}
        self._source_index: dict[SourcePrefix, dict[StateKey, None]] = {}
        self._shells: dict[ShellKey, ShellEntry] = {}
        self._active_models: dict[ShellKey, tuple[PatchedUnitSpec, Mapping[str, torch.Tensor] | None]] = {}
        self._staging: dict[torch.device, torch.Tensor] = {}
        self._mappings: dict[tuple[tuple[Path, ...], tuple[str, ...]], LoadMapping] = {}
        self._checkpoint_metadata: dict[Path, dict[str, object]] = {}
        self._stagers: dict[torch.device, DeviceStager] = {}
        self._lock = threading.RLock()

    def get_state(self, key: StateKey) -> StateEntry | None:
        """Return the retained entry for a logical state, if present."""
        with self._lock:
            return self._states.get(key)

    def retain_state(
        self, spec: UnitSpec | PatchedUnitSpec, tensors: Mapping[str, torch.Tensor], placement: Placement
    ) -> StateEntry | None:
        """Retain a copied unit state under the key derived from its spec.
        Reuse an entry with the same placement and layout; otherwise replace
        it. Return ``None`` for a placement without a device.
        """
        if placement.device is None:
            return None
        key = _state_key(spec)
        if placement.pin_memory and not torch.cuda.is_available():
            placement = Placement(torch.device("cpu"))
        expected = selected_metadata(spec)
        if tuple(tensors) != tuple(expected) or any(
            tensors[name].shape != shape or (isinstance(spec, UnitSpec) and tensors[name].dtype != dtype)
            for name, (shape, dtype) in expected.items()
        ):
            raise ValueError("state entry values disagree with selected unit")
        with self._lock:
            existing = self._states.get(key)
            if _matches_entry(existing, tensors, placement):
                return existing
        entry = (
            pack_units_cpu(spec, tensors, placement)
            if placement.device.type == "cpu"
            else StateEntry(
                spec,
                {name: tensors[name].to(placement.device, copy=True) for name in expected},
                placement,
            )
        )
        with self._lock:
            existing = self._states.get(key)
            if _matches_entry(existing, entry.tensors, entry.placement):
                return existing
            self._states[key] = entry
            if existing is None:
                for prefix in _source_prefixes(key):
                    self._source_index.setdefault(prefix, {})[key] = None
            return entry

    def adopt_entry(self, entry: StateEntry) -> StateEntry:
        """Retain an already placed entry without copying its tensor storage."""
        key = _state_key(entry.spec)
        with self._lock:
            existing = self._states.get(key)
            if _matches_entry(existing, entry.tensors, entry.placement):
                return existing
            self._states[key] = entry
            if existing is None:
                for prefix in _source_prefixes(key):
                    self._source_index.setdefault(prefix, {})[key] = None
            return entry

    def unit_states(self, source: SourceKey) -> dict[StateKey, StateEntry]:
        """Return the retained clean state of ``source`` and every retained active state built from it."""
        with self._lock:
            return {
                key: entry
                for key, entry in self._states.items()
                if key == source or (isinstance(key, FusedKey) and key.clean == source)
            }

    def pop_state(self, key: StateKey) -> StateEntry | None:
        """Remove and return the retained entry for this logical state."""
        with self._lock:
            entry = self._states.pop(key, None)
            for prefix in _source_prefixes(key) if entry is not None else ():
                bucket = self._source_index[prefix]
                bucket.pop(key, None)
                if not bucket:
                    del self._source_index[prefix]
            return entry

    def get_shell(self, key: ShellKey) -> ShellEntry | None:
        """Return the cached shell for ``key``, if present."""
        with self._lock:
            return self._shells.get(key)

    def add_shell(self, key: ShellKey, entry: ShellEntry) -> ShellEntry:
        """Retain a shell unless model caching is disabled."""
        if not self.cache_models:
            return entry
        with self._lock:
            if key in self._shells:
                raise ValueError("shell is already retained; get it before adding")
            self._shells[key] = entry
        return entry

    def pop_shell(self, key: ShellKey) -> ShellEntry | None:
        """Remove and return a retained shell."""
        with self._lock:
            return self._shells.pop(key, None)

    def active_model(
        self, shell: ShellKey, dst: Mapping[str, torch.Tensor] | None = None
    ) -> tuple[torch.nn.Module, PatchedUnitSpec] | None:
        """Return the cached shell under ``shell`` and the whole-model spec it holds, if it is bound to ``dst``.
        ``dst`` is matched by identity; ``None`` means storage the shell's last build allocated for itself.
        """
        with self._lock:
            record = self._active_models.get(shell)
            entry = self._shells.get(shell)
        if record is None or entry is None or record[1] is not dst or _disposed(entry.shell):
            return None
        return entry.shell, record[0]

    def held_spec(self, dst: Mapping[str, torch.Tensor]) -> PatchedUnitSpec | None:
        """Return the whole-model spec whose active state ``dst`` holds, as recorded for a shell bound to it."""
        with self._lock:
            records = [
                (self._shells.get(shell), spec) for shell, (spec, bound) in self._active_models.items() if bound is dst
            ]
        return next((spec for entry, spec in records if entry is not None and not _disposed(entry.shell)), None)

    def set_active_model(
        self, shell: ShellKey, spec: PatchedUnitSpec, dst: Mapping[str, torch.Tensor] | None = None
    ) -> None:
        """Record that the cached shell holds ``spec``'s active state in ``dst``, as does every shell bound to it."""
        with self._lock:
            if shell not in self._shells:
                return
            self._active_models[shell] = spec, dst
            if dst is not None:
                for other, (_, bound) in self._active_models.items():
                    if bound is dst:
                        self._active_models[other] = spec, dst

    def forget_active_models(self, dst: Mapping[str, torch.Tensor]) -> None:
        """Drop the records of the shells bound to ``dst``, before ``dst`` is rewritten."""
        with self._lock:
            for shell in [shell for shell, (_, bound) in self._active_models.items() if bound is dst]:
                del self._active_models[shell]

    def release_active_models(self, dst: MutableMapping[str, torch.Tensor]) -> None:
        """Dispose every shell bound to ``dst``, drop their records, and empty ``dst`` so its storage is freed."""
        with self._lock:
            shells = [shell for shell, (_, bound) in self._active_models.items() if bound is dst]
            for shell in shells:
                del self._active_models[shell]
            bound_models = [self._shells[shell].shell for shell in shells if shell in self._shells]
        for model in bound_models:
            Disposable.dispose(model)
        dst.clear()

    def staging(self, device: torch.device, nbytes: int) -> torch.Tensor:
        """Return ``nbytes`` of this device's exchange scratch buffer, reused by every distributed build."""
        with self._lock:
            buffer = self._staging.get(device)
            if buffer is None or buffer.numel() < nbytes:
                self._staging.pop(device, None)
                buffer = self._staging[device] = torch.empty(nbytes, dtype=torch.uint8, device=device)
            return buffer.narrow(0, 0, nbytes)

    def mapping(self, paths: Sequence[Path], operations: Sequence[SDOps]) -> LoadMapping:
        """Project checkpoint shards through ``operations``, reusing a cached or live entry's projection."""
        prefix = tuple(Path(path).resolve() for path in paths), tuple(operation.name for operation in operations)
        with self._lock:
            cached = self._mappings.get(prefix)
        if cached is not None:
            return cached
        mapping = self._live_mapping(prefix) or build_mapping(tuple(paths), tuple(operations))
        if self.cache_metadata:
            with self._lock:
                mapping = self._mappings.setdefault(prefix, mapping)
        return mapping

    def checkpoint_metadata(self, paths: Sequence[Path]) -> dict[str, object]:
        """Return a copy of the first shard's parsed ``__metadata__``."""
        path = Path(paths[0]).resolve()
        with self._lock:
            parsed = self._checkpoint_metadata.get(path)
        if parsed is None:
            with CheckpointReader.open((path,)) as reader:
                parsed = reader.read_checkpoint_metadata()
            if self.cache_metadata:
                with self._lock:
                    parsed = self._checkpoint_metadata.setdefault(path, parsed)
        return copy.deepcopy(parsed)

    def device_stager(self, device: torch.device) -> DeviceStager:
        """Return the reusable host buckets that stage checkpoint bytes onto ``device``."""
        with self._lock:
            stager = self._stagers.get(device)
            if stager is None:
                stager = self._stagers[device] = DeviceStager(device)
            return stager

    def find_source(self, paths: Sequence[Path], sd_ops_names: Sequence[str]) -> StateEntry | None:
        """Find a live entry referring to these checkpoint paths and SDOps names.
        A patched entry can supply a clean or LoRA mapping through its spec;
        this lookup does not imply that the corresponding input tensors are retained.
        """
        prefix = tuple(Path(path).resolve() for path in paths), tuple(sd_ops_names)
        with self._lock:
            for key in self._source_index.get(prefix, ()):
                entry = self._states.get(key)
                if entry is not None:
                    return entry
        return None

    def clear(self) -> None:
        """Drop retained states, model shells, checkpoint metadata, device stagers, and source indexes."""
        with self._lock:
            self._states.clear()
            self._source_index.clear()
            self._shells.clear()
            self._active_models.clear()
            self._staging.clear()
            self._mappings.clear()
            self._checkpoint_metadata.clear()
            self._stagers.clear()

    def _live_mapping(self, prefix: tuple[tuple[Path, ...], tuple[str, ...]]) -> LoadMapping | None:
        entry = self.find_source(*prefix)
        if entry is None:
            return None
        sources = (entry.spec.clean, *entry.spec.loras) if isinstance(entry.spec, PatchedUnitSpec) else (entry.spec,)
        for source in sources:
            mapping = source.load_mapping
            if (mapping.paths, tuple(operation.name for operation in mapping.sd_ops)) == prefix:
                return mapping
        return None


class DummyRegistry(ModelRegistry):
    """A registry that caches no model shells or checkpoint metadata and retains no states."""

    def __init__(self) -> None:
        super().__init__(cache_models=False, cache_metadata=False)


@dataclass(frozen=True, slots=True)
class RegistryKeys:
    """Registry keys a build queries and the entries it leaves in the registry.
    ``lookup`` lists every key the build may query, whatever the registry holds: a hit can make
    later queries unnecessary. ``retained`` lists the entries the build stores under its placement
    policy, and its shell when the registry caches models.
    """

    lookup: frozenset[ShellKey | StateKey | ActiveModelKey] = frozenset()
    retained: frozenset[ShellKey | StateKey | ActiveModelKey] = frozenset()

    def __or__(self, other: RegistryKeys) -> RegistryKeys:
        return RegistryKeys(self.lookup | other.lookup, self.retained | other.retained)


@dataclass(frozen=True, slots=True)
class ActiveModelKey:
    """A cached shell, under ``shell``, bound to the active state of ``fused``."""

    fused: FusedKey
    shell: ShellKey


def pack_units_cpu(
    spec: UnitSpec | PatchedUnitSpec, tensors: Mapping[str, torch.Tensor], placement: Placement
) -> StateEntry:
    """Copy selected tensors to CPU, packing dense layouts into one aligned buffer."""
    if placement.device != torch.device("cpu"):
        raise ValueError("host packing needs CPU placement")
    ordered = {name: tensors[name].to("cpu") for name in selected_metadata(spec)}
    if any(_span_elements(value) != value.numel() for value in ordered.values()):
        retained = {name: value.clone(memory_format=torch.preserve_format) for name, value in ordered.items()}
        return StateEntry(spec, retained, Placement(torch.device("cpu")))

    offsets: dict[str, int] = {}
    total = 0
    for name, value in ordered.items():
        total = _align(total)
        offsets[name] = total
        total += _span_elements(value) * value.element_size()
    pin = placement.pin_memory and torch.cuda.is_available()
    try:
        buffer = host_bytes(total, pin=pin)
    except RuntimeError:
        if not placement.pin_memory:
            raise
        buffer = torch.empty(total, dtype=torch.uint8)
    retained: dict[str, torch.Tensor] = {}
    for name, value in ordered.items():
        span = _span_elements(value) * value.element_size()
        raw = buffer.narrow(0, offsets[name], span).view(value.dtype)
        view = torch.as_strided(raw, value.shape, value.stride())
        view.copy_(value)
        retained[name] = view
    actual = Placement(torch.device("cpu"), pin_memory=buffer.is_pinned())
    return StateEntry(spec, retained, actual, buffer)


def _disposed(model: torch.nn.Module) -> bool:
    """Whether a model's weights were released to meta, or it has none."""
    parameter = next(model.parameters(), None)
    return parameter is None or parameter.is_meta


def _source_prefixes(key: StateKey) -> tuple[SourcePrefix, ...]:
    sources = (key,) if isinstance(key, SourceKey) else (key.clean, *(source for source, _ in key.adapters))
    return tuple(dict.fromkeys(source.prefix for source in sources))


def _state_key(spec: UnitSpec | PatchedUnitSpec) -> StateKey:
    return (
        FusedKey.from_patched_unit_spec(spec) if isinstance(spec, PatchedUnitSpec) else SourceKey.from_unit_spec(spec)
    )


def _matches_entry(entry: StateEntry | None, tensors: Mapping[str, torch.Tensor], placement: Placement) -> bool:
    return (
        entry is not None
        and entry.placement == placement
        and tuple(entry.tensors) == tuple(tensors)
        and all(
            (entry.tensors[name].shape, entry.tensors[name].dtype, entry.tensors[name].stride())
            == (value.shape, value.dtype, value.stride())
            for name, value in tensors.items()
        )
    )


def _span_elements(tensor: torch.Tensor) -> int:
    if not tensor.numel():
        return 0
    return 1 + sum((size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride(), strict=True))


def _align(offset: int) -> int:
    return (offset + UNIT_ALIGN - 1) // UNIT_ALIGN * UNIT_ALIGN
