"""Checkpoint mappings for model tensors and LoRA factors, and execution units.
The types here describe source metadata and the tensor groups consumed or
produced during loading. They keep checkpoint identity separate from where a
unit is retained or executed.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import InitVar, dataclass, field
from functools import cached_property
from pathlib import Path
from types import MappingProxyType
from typing import NamedTuple, Protocol

import torch

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.sd_ops import ModelKeyGroup, SDOps

UNIT_ALIGN = 512
"""Byte alignment of every tensor in a packed unit layout: host owners and streaming buffers.
It matches the CUDA caching allocator's block granularity, so a tensor in a device buffer starts on the same boundary
as a separately allocated one; kernels run slower on weights off that alignment. Host owners use the same layout as
the device buffers they are copied into, so a whole unit moves from host to device as one byte range.
"""

TensorMetadata = tuple[torch.Size, torch.dtype]  # Shape and dtype of one tensor.
StateDictMetadata = Mapping[str, TensorMetadata]  # Ordered tensor keys to their shapes and dtypes.
_FLOAT_DTYPES = frozenset({torch.float16, torch.bfloat16, torch.float32, torch.float64})
_FP8_DTYPES = frozenset({torch.float8_e4m3fn, torch.float8_e5m2})
MPS_HOST_FUSED_FP8: Mapping[str, frozenset[torch.dtype]] = MappingProxyType({"mps": _FP8_DTYPES})
"""``FuseRule.host_fused_dtypes`` for rules whose FP8 arithmetic MPS does not run."""
_LORA_A = ".lora_A.weight"
_LORA_B = ".lora_B.weight"
PatchRule = Callable[[str, torch.Tensor, torch.Tensor, Mapping[str, torch.Tensor]], Mapping[str, torch.Tensor]]


def active_dtype(metadata: TensorMetadata, requested_dtype: torch.dtype | None) -> torch.dtype:
    """Return the dtype of one clean tensor in an active state requested at ``requested_dtype``.
    Floating tensors take the requested dtype. FP32 scalars and non-floating tensors, such as
    quantized weights, keep their checkpoint dtype, as does every tensor when nothing is requested.
    """
    shape, dtype = metadata
    if requested_dtype is None or dtype not in _FLOAT_DTYPES or (not shape and dtype == torch.float32):
        return dtype
    return requested_dtype


class LoRAFactorKeys(NamedTuple):
    """LoRA A and B factor keys identifying the model weight they modify.
    ``a`` and ``b`` name the LoRA factors after the adapter's state-dict operations.
    """

    a: str
    b: str


def lora_target(key: str) -> str | None:
    """Return the model-weight key identified by a LoRA factor key.
    Return ``None`` when the key has neither supported A nor B factor suffix.
    """
    for suffix in (_LORA_A, _LORA_B):
        if key.endswith(suffix):
            return f"{key.removesuffix(suffix)}.weight"
    return None


def factor_pairs(keys: Iterable[str]) -> dict[str, LoRAFactorKeys]:
    """Index selected LoRA A/B factor pairs by the model-weight keys they patch.
    Ignore keys without a LoRA factor suffix and reject incomplete pairs.
    """
    pairs: dict[str, dict[str, str]] = {}
    for key in keys:
        target = lora_target(key)
        if target is not None:
            pairs.setdefault(target, {})["a" if key.endswith(_LORA_A) else "b"] = key
    result = {}
    for target, pair in pairs.items():
        try:
            result[target] = LoRAFactorKeys(pair["a"], pair["b"])
        except KeyError as exc:
            raise ValueError(f"adapter has incomplete A/B pair for {target!r}") from exc
    return result


@dataclass(frozen=True, slots=True, eq=False)
class TensorMapping:
    """Full checkpoint provenance for model tensors or applicable LoRA factors.
    Mapping preparation validates dependencies and supplies read-only fields.
    Units select tensors while sharing this complete record.
    """

    sources: Mapping[str, Path]
    """Raw checkpoint keys to the files that hold them."""
    source_metadata: StateDictMetadata
    """Stored shapes and dtypes of the raw checkpoint keys."""
    destination_metadata: StateDictMetadata
    """Shapes and dtypes after the state-dict operations, under model parameter or buffer keys, or under LoRA
    factor keys identifying the model weights they modify."""
    dependencies: Mapping[str, frozenset[str]]
    """Each destination key to its contributing raw keys; a weight and its scale may come from different files."""
    connected_destinations: tuple[tuple[str, ...], ...]
    """Ordered groups of at least two outputs connected by shared checkpoint inputs. These are checkpoint read
    relationships, not fusion-rule requirements."""
    _group_index: Mapping[str, int] = field(repr=False)
    """Keys of ``connected_destinations`` to their group positions; absent keys are independent single outputs."""
    destination_strides: Mapping[str, tuple[int, ...]] = field(default_factory=lambda: MappingProxyType({}), repr=False)
    """Non-contiguous output layouts produced by the state-dict operations, and only those. The loader uses them
    when it allocates a selected tensor before reading its checkpoint inputs."""
    destination_aliases: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}), repr=False)
    """A later output key to the first output key whose abstract operation returned the exact same tensor object.
    Loader-owned storage can share those outputs; caller-supplied destinations keep their separate storage."""
    source_offsets: Mapping[str, int] = field(default_factory=lambda: MappingProxyType({}), repr=False)
    """Each raw key's absolute payload offset in its file, where the checkpoint format exposes one; the payload is
    ``numel * itemsize`` bytes of ``source_metadata``. Readers that copy file bytes straight into destination
    storage use it; other consumers ignore it."""
    transformed_value_destinations: frozenset[str] = field(default=frozenset(), repr=False)
    """Destinations a value operation produced from their ``dependencies``. Every other destination renames the
    single raw key in its ``dependencies`` and holds that key's stored bytes unchanged, so a reader can copy them
    straight from the file."""


@dataclass(frozen=True, slots=True, eq=False)
class LoadMapping:
    """One shared checkpoint recipe for model tensors or applicable LoRA factors.
    ``paths`` contains the configured, resolved checkpoint files in caller order,
    including files whose tensors are removed by state-dict operations.
    ``sd_ops`` contains the executable operation chain. ``checkpoint_metadata``
    is the read-only checkpoint configuration used to construct a model shell,
    distinct from tensor shapes and dtypes in ``tensor_mapping``.
    """

    paths: tuple[Path, ...]
    sd_ops: tuple[SDOps, ...]
    checkpoint_metadata: Mapping[str, object]
    tensor_mapping: TensorMapping


class Unit(Protocol):
    """Common interface for a tensor group with an ID, ordered keys, and sizes.
    ``unit_id`` identifies the group for partitioning and placement policies.
    ``metadata`` maps its keys to shapes and dtypes in execution order.
    """

    __slots__ = ()

    @property
    def unit_id(self) -> str:
        """Return the stable ID of this tensor group."""
        ...

    @property
    def metadata(self) -> StateDictMetadata:
        """Read the group's ordered tensor keys, shapes and dtypes."""
        ...

    @property
    def destinations(self) -> tuple[str, ...]:
        """Return this unit's tensor keys in metadata order."""
        ...

    @property
    def nbytes(self) -> int:
        """Return total tensor payload bytes, excluding storage padding and stride gaps."""
        ...


@dataclass(frozen=True, slots=True)
class MetadataUnit(Unit):
    """An identified tensor group described by ordered shapes and dtypes.
    ``unit_id`` names the group. ``metadata`` maps its tensor keys to shape
    and dtype pairs; the group holds no materialized tensors.
    """

    unit_id: str = field()
    metadata: StateDictMetadata = field()

    def __post_init__(self) -> None:
        """Require an ID and tensor keys, preserving a prepared read-only selection."""
        if not self.unit_id or not self.metadata:
            raise ValueError("a unit needs an id and tensor metadata")
        if not isinstance(self.metadata, MappingProxyType):
            object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def destinations(self) -> tuple[str, ...]:
        """Return the unit's tensor keys in metadata insertion order."""
        return tuple(self.metadata)

    @property
    def nbytes(self) -> int:
        """Return tensor payload bytes implied by the group's shapes and dtypes."""
        return sum(shape.numel() * dtype.itemsize for shape, dtype in self.metadata.values())


@dataclass(slots=True)
class TensorsUnit(Unit):
    """An identified group of live tensors implementing the unit interface.
    ``unit_id`` names the group. ``tensors`` maps its tensor keys to live
    values; their shapes and dtypes provide the unit's metadata.
    """

    unit_id: str = field()
    tensors: Mapping[str, torch.Tensor]

    def __post_init__(self) -> None:
        """Require a nonempty ID and at least one live tensor."""
        if not self.unit_id or not self.tensors:
            raise ValueError("a tensors unit needs an id and tensors")

    @property
    def metadata(self) -> StateDictMetadata:
        """Derive shape and dtype metadata from the live tensors."""
        return {key: (value.shape, value.dtype) for key, value in self.tensors.items()}

    @property
    def destinations(self) -> tuple[str, ...]:
        """Return tensor keys in the live tensor mapping's insertion order."""
        return tuple(self.tensors)

    @property
    def nbytes(self) -> int:
        """Return the live tensor storage size in bytes, excluding stride gaps."""
        return sum(value.numel() * value.element_size() for value in self.tensors.values())


@dataclass(frozen=True, slots=True, kw_only=True)
class RankUnit(MetadataUnit):
    """A metadata unit owned by ``rank_id`` within its process group."""

    rank_id: int

    def __post_init__(self) -> None:
        """Validate the unit's ID and keys and require a nonnegative owner rank."""
        MetadataUnit.__post_init__(self)
        if self.rank_id < 0:
            raise ValueError("rank_id must be nonnegative")


@dataclass(frozen=True, slots=True, kw_only=True)
class StreamedUnit(MetadataUnit):
    """A metadata unit loaded when ``trigger_path`` is about to execute.
    ``trigger_path`` names the owning submodule relative to the model root.
    """

    trigger_path: str

    def __post_init__(self) -> None:
        """Validate the unit's ID and keys and require a nonempty trigger path."""
        MetadataUnit.__post_init__(self)
        if not self.trigger_path:
            raise ValueError("streamed unit needs a trigger path")


@dataclass(frozen=True, slots=True, kw_only=True)
class FileReadUnit(MetadataUnit):
    """Destinations of one unit that are read from checkpoint files the same way.
    Without ``staging_required`` each destination renames one raw key, whose bytes are read straight into the
    destination tensor (cast where its dtype differs). With it, the destinations are whole read groups that value
    operations produce: their raw inputs are staged first. A file-read unit only describes reading: the unit it comes
    from keeps its own storage, retention and placement.
    """

    staging_required: bool = False


@dataclass(frozen=True, slots=True)
class FuseRule:
    """Named LoRA arithmetic and companion keys needed for quantized weights.
    ``host_fused_dtypes`` maps a device type to the weight dtypes the rule cannot fuse in that device's memory:
    such weights are fused in host memory, with their deltas still computed on the device.
    """

    name: str
    aggregation_dtype: torch.dtype
    patch: PatchRule
    requires_companions: tuple[str, ...] = ()
    logical_weight_shape: Callable[[torch.Size, torch.dtype], torch.Size] | None = None
    host_fused_dtypes: Mapping[str, frozenset[torch.dtype]] = field(
        default_factory=lambda: MappingProxyType({}), compare=False
    )

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("a fuse rule needs a stable name")

    def fuses_on(self, device_type: str, dtype: torch.dtype) -> bool:
        """Whether a weight of ``dtype`` can be fused in the memory of a ``device_type`` device."""
        return dtype not in self.host_fused_dtypes.get(device_type, frozenset())

    def dtype_for(self, device: torch.device) -> torch.dtype:
        """Use FP32 instead of BF16 aggregation on MPS."""
        return (
            torch.float32
            if device.type == "mps" and self.aggregation_dtype == torch.bfloat16
            else self.aggregation_dtype
        )

    def __call__(
        self, key: str, weight: torch.Tensor, delta: torch.Tensor, clean: Mapping[str, torch.Tensor]
    ) -> Mapping[str, torch.Tensor]:
        return self.patch(key, weight, delta, clean)


@dataclass(frozen=True, slots=True, eq=False)
class UnitSpec:
    """A tensor selection and its shared, complete checkpoint recipe.
    ``unit`` identifies selected tensor keys, shapes and dtypes.
    ``load_mapping`` describes the original checkpoint transformation, including
    tensors outside this selection. Root validation and partition preparation
    check selections; constructing a child spec does not scan the mapping.
    """

    unit: Unit
    load_mapping: LoadMapping


@dataclass(frozen=True, eq=False)
class LoRAUnitSpec(UnitSpec):
    """Selected LoRA factors and their strength for modifying model weights.
    ``unit`` selects LoRA factor keys. ``load_mapping`` describes their full
    checkpoint transformation into factors applicable to model weights.
    ``strength`` scales this LoRA adapter's contribution.
    """

    strength: float

    @cached_property
    def bindings(self) -> dict[str, LoRAFactorKeys]:
        """Map each targeted model-weight key to its selected LoRA A/B factor keys."""
        return factor_pairs(self.unit.destinations)


@dataclass(frozen=True, slots=True)
class PatchedUnitSpec:
    """A model unit, its fusion rule, ordered adapters, and model-key groups.
    ``clean`` describes its unpatched checkpoint state. ``fuse_rule`` also
    governs partition grouping when ``loras`` is empty. ``loras`` lists selected
    adapters in application order. ``requested_dtype`` distinguishes active
    states requested at different output dtypes.
    ``model_key_groups`` is prepared once from the root unit and the rule.
    A string is one independently partitionable model key; a tuple contains
    existing model keys that must travel together, such as an FP8 weight and
    its weight scale. Every tuple member remains in ``clean.unit.metadata``
    and in loaded state; grouping does not consume or transform a tensor.
    Partitioners pass selected whole groups into child specs. ``None`` means
    the rule has no companion keys and the unit's metadata keys can be used
    directly, without building a second key sequence. To change the rule,
    construct a new root spec or pass ``model_key_groups=None`` to ``replace``;
    existing groups belong to the old rule.
    Set ``validate_binding=True`` only when explicitly checking the adapter-to-model
    binding. It is an initialization argument, not part of the resulting spec,
    so partitioned children do not repeat the check.
    """

    clean: UnitSpec
    fuse_rule: FuseRule
    loras: tuple[LoRAUnitSpec, ...] = ()
    requested_dtype: torch.dtype | None = None
    model_key_groups: tuple[ModelKeyGroup, ...] | None = field(default=None, repr=False, compare=False, kw_only=True)
    validate_binding: InitVar[bool] = field(default=False, kw_only=True)

    def __post_init__(self, validate_binding: bool) -> None:
        """Prepare root groups and optionally check the adapter-to-model binding once."""
        if self.requested_dtype is not None and self.requested_dtype not in _FLOAT_DTYPES:
            raise ValueError(f"requested dtype {self.requested_dtype} is not a supported floating dtype")
        if not self.fuse_rule.requires_companions:
            object.__setattr__(self, "model_key_groups", None)
        elif self.model_key_groups is None:
            object.__setattr__(
                self,
                "model_key_groups",
                _group_model_keys(self.clean.unit.metadata, self.fuse_rule.requires_companions),
            )
        if validate_binding:
            model_metadata = self.clean.unit.metadata
            for adapter in self.loras:
                checkpoint_metadata = adapter.load_mapping.tensor_mapping.destination_metadata
                for name, metadata in adapter.unit.metadata.items():
                    if checkpoint_metadata.get(name) != metadata:
                        raise ValueError(f"LoRA factor {name!r} disagrees with checkpoint metadata")
                for key, pair in adapter.bindings.items():
                    if key not in model_metadata:
                        raise ValueError(f"LoRA target {key!r} is absent from the model unit")
                    a_shape, _ = adapter.unit.metadata[pair.a]
                    b_shape, _ = adapter.unit.metadata[pair.b]
                    weight_shape, weight_dtype = model_metadata[key]
                    logical_shape = (
                        self.fuse_rule.logical_weight_shape(weight_shape, weight_dtype)
                        if self.fuse_rule.logical_weight_shape is not None
                        else weight_shape
                    )
                    if (
                        len(a_shape) != 2
                        or len(b_shape) != 2
                        or b_shape[1] != a_shape[0]
                        or torch.Size((b_shape[0], a_shape[1])) != logical_shape
                    ):
                        raise ValueError(f"LoRA factors for {key!r} do not match the model weight")
                    if weight_dtype in (torch.uint8, torch.float8_e4m3fn, torch.float8_e5m2):
                        prefix = key.removesuffix(".weight")
                        missing = [
                            f"{prefix}{suffix}"
                            for suffix in self.fuse_rule.requires_companions
                            if f"{prefix}{suffix}" not in model_metadata
                        ]
                        if missing:
                            raise ValueError(f"{key!r} needs companion tensors: {missing}")


def _group_model_keys(metadata: StateDictMetadata, companion_suffixes: tuple[str, ...]) -> tuple[ModelKeyGroup, ...]:
    """Put each existing model key in one ordered model-key group."""
    groups: dict[str, tuple[str, ...]] = {}
    companions: set[str] = set()
    for key in metadata:
        if not key.endswith(".weight"):
            continue
        prefix = key.removesuffix(".weight")
        present = tuple(name for suffix in companion_suffixes if (name := f"{prefix}{suffix}") in metadata)
        if present:
            groups[key] = (key, *present)
            companions.update(present)
    return tuple(groups.get(key, key) for key in metadata if key not in companions)
