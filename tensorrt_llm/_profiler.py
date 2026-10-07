# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared, observational PyTorch profiler scopes and literal metadata."""

import math
import os
from collections.abc import Callable, Generator, Iterable, Mapping
from contextlib import AbstractContextManager, contextmanager
from enum import Enum
from functools import wraps
from types import TracebackType
from typing import Literal, ParamSpec, TypeAlias, TypeVar, cast

import numpy as np
import torch

ProfilerRegion: TypeAlias = Literal["workflow", "mpi", "nccl", "nanobind", "kv", "moe"]
MpiOperationKind: TypeAlias = Literal["collective", "p2p"]
MpiDirection: TypeAlias = Literal["send", "recv"]
MetadataValue: TypeAlias = (
    str
    | int
    | float
    | bool
    | None
    | list["MetadataValue"]
    | tuple["MetadataValue", ...]
    | dict[str, "MetadataValue"]
)

__all__ = [
    "MetadataValue",
    "MpiDirection",
    "MpiOperationKind",
    "ProfilerRegion",
    "custom_region_enabled",
    "record_function_metadata_scope",
    "record_function_with_metadata",
    "record_moe_routing",
    "record_mpi_function",
]

_CUSTOM_REGIONS_ENV = "TLLM_PROFILE_CUSTOM_REGIONS"
_CUSTOM_REGION_NAMES: frozenset[ProfilerRegion] = frozenset(
    {"workflow", "mpi", "nccl", "nanobind", "kv", "moe"}
)
_CUSTOM_REGION_DEFAULTS = _CUSTOM_REGION_NAMES - {"kv", "moe"}
_METADATA_PREFIX = "TRTLLM1"
_P = ParamSpec("_P")
_R = TypeVar("_R")
_MISSING = object()


def _parse_custom_regions(value: str | None) -> frozenset[ProfilerRegion]:
    if value is None:
        return _CUSTOM_REGION_DEFAULTS
    regions = [region.strip(" \t\r\n") for region in value.split(",")]
    if regions == ["none"]:
        return frozenset()
    if (
        not all(regions)
        or any(region not in _CUSTOM_REGION_NAMES for region in regions)
        or len(regions) != len(set(regions))
    ):
        raise ValueError(
            f"Invalid {_CUSTOM_REGIONS_ENV}: {value!r}; expected a comma-separated subset of "
            "workflow,mpi,nccl,nanobind,kv,moe or none"
        )
    return frozenset(cast(ProfilerRegion, region) for region in regions)


_CUSTOM_REGIONS = _parse_custom_regions(os.environ.get(_CUSTOM_REGIONS_ENV))


def custom_region_enabled(region: ProfilerRegion) -> bool:
    """Return whether process-start configuration selects a custom region.

    Args:
        region: One of the six supported region categories.

    Returns:
        Whether the cached selector includes the region; this does not enable the profiler.
    """
    return region in _CUSTOM_REGIONS


def _validate_metadata_value(value: MetadataValue) -> None:
    if type(value) not in (type(None), str, bool, int, float, list, tuple, dict):
        raise TypeError(f"Unsupported profiler metadata value: {type(value).__name__}")
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Profiler metadata floats must be finite")
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _validate_metadata_value(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError("Profiler metadata keys must be strings")
            _validate_metadata_value(item)
        return


def _encode_metadata(region: ProfilerRegion, metadata: Mapping[str, MetadataValue]) -> str:
    """Encode a UTF-8 byte-prefixed region and a supported Python literal dictionary."""
    fields = dict(metadata)
    _validate_metadata_value(fields)
    return _METADATA_PREFIX + f"{len(region.encode('utf-8'))}:{region}" + repr(fields)


@contextmanager
def record_function_metadata_scope(
    name: str,
    region: ProfilerRegion,
    metadata_factory: Callable[[], str | Mapping[str, MetadataValue]],
) -> Generator[None, None, None]:
    """Record a selected active scope without changing the body's outcome.

    Args:
        name: The PyTorch user-scope name.
        region: The category controlled by process-start selection.
        metadata_factory: A lazy producer of a complete envelope or typed fields.

    Yields:
        None while the wrapped operation executes. ``contextmanager`` converts
        this generator into a context manager, which can also be used as a decorator.

    Metadata and profiler entry/cleanup failures are observational and discarded.
    Exceptions from the body propagate unchanged, including BaseException subclasses.
    """
    if not custom_region_enabled(region) or not torch.autograd._profiler_enabled():
        yield
        return

    scope: torch.profiler.record_function | None = None
    try:
        metadata = metadata_factory()
        if not isinstance(metadata, str):
            metadata = _encode_metadata(region, metadata)
        scope = torch.profiler.record_function(name, args=metadata)
        scope.__enter__()
    except Exception:
        # This boundary covers only observational setup, never the operation body.
        scope = None
    try:
        yield
    finally:
        if scope is not None:
            try:
                scope.__exit__(None, None, None)
            except Exception:
                # Cleanup must not replace an exception raised by the operation body.
                pass


def _tensor_metadata(tensor: torch.Tensor) -> dict[str, MetadataValue]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "stride": list(tensor.stride()),
        "device": str(tensor.device),
    }


class _RecordFunctionWithMetadata:
    """Context manager/decorator for lazily inspected direct tensor arguments."""

    def __init__(
        self,
        name: str,
        *values: object,
        metadata_factory: Callable[..., Mapping[str, MetadataValue]] | None = None,
    ) -> None:
        self._name = name
        self._values = values
        self._metadata_factory = metadata_factory
        self._scope: AbstractContextManager[None] | None = None

    def _metadata(self) -> dict[str, MetadataValue]:
        metadata: dict[str, MetadataValue] = {}
        tensor_index = 0
        for value in self._values:
            if isinstance(value, torch.Tensor):
                prefix = f"workflow.tensor.{tensor_index}"
                metadata.update(
                    {f"{prefix}.{key}": item for key, item in _tensor_metadata(value).items()}
                )
                tensor_index += 1
        metadata["workflow.tensor_count"] = tensor_index
        if self._metadata_factory is not None:
            metadata.update(self._metadata_factory())
        return metadata

    def __enter__(self) -> "_RecordFunctionWithMetadata":
        self._scope = record_function_metadata_scope(self._name, "workflow", self._metadata)
        self._scope.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        assert self._scope is not None
        return self._scope.__exit__(exc_type, exc_value, traceback)

    def __call__(self, func: Callable[_P, _R]) -> Callable[_P, _R]:
        @wraps(func)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            factory = self._metadata_factory
            with _RecordFunctionWithMetadata(
                self._name,
                *self._values,
                *args,
                *kwargs.values(),
                metadata_factory=(lambda: factory(*args, **kwargs))
                if factory is not None
                else None,
            ):
                return func(*args, **kwargs)

        return wrapper


def record_function_with_metadata(
    name: str,
    *values: object,
    metadata_factory: Callable[..., Mapping[str, MetadataValue]] | None = None,
) -> _RecordFunctionWithMetadata:
    """Record a workflow boundary as a context manager or typed decorator.

    Args:
        name: The PyTorch user-scope name.
        values: Objects to inspect lazily. Direct tensors of any shape/dtype
            contribute shape, dtype, stride, and device; containers are not
            traversed and tensor contents are never copied. A decorated call
            also inspects its direct positional and keyword arguments.
        metadata_factory: Optional lazy typed fields, merged after tensor fields.
            For a context it takes no arguments; for a decorator it receives
            the wrapped call's arguments. It runs inside observational setup.

    Returns:
        A scope that preserves arguments, return identity, and body exceptions
        through the shared observational scope API. Metadata is constructed only
        when workflow is selected and a profiler is active. CUDA graph replay
        does not re-execute Python decorators inside the captured model.
    """
    return _RecordFunctionWithMetadata(name, *values, metadata_factory=metadata_factory)


def _mpi_metadata_value(value: object) -> MetadataValue:
    if isinstance(value, Enum):
        value = value.value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return cast(MetadataValue, value)


def _mpi_payload_metadata(payload: object) -> dict[str, MetadataValue]:
    fields: dict[str, MetadataValue] = {"mpi.payload.type": type(payload).__name__}
    if isinstance(payload, torch.Tensor):
        numel = payload.numel()
        fields.update(
            {
                "mpi.payload.shape": list(payload.shape),
                "mpi.payload.dtype": str(payload.dtype),
                "mpi.payload.stride": list(payload.stride()),
                "mpi.payload.device": str(payload.device),
                "mpi.payload.numel": numel,
                "mpi.payload.bytes": numel * payload.element_size(),
            }
        )
    elif isinstance(payload, np.ndarray):
        fields.update(
            {
                "mpi.payload.shape": list(payload.shape),
                "mpi.payload.dtype": str(payload.dtype),
                "mpi.payload.numel": int(payload.size),
                "mpi.payload.bytes": int(payload.nbytes),
            }
        )
    elif isinstance(payload, memoryview):
        fields["mpi.payload.bytes"] = payload.nbytes
    elif isinstance(payload, (bytes, bytearray)):
        fields["mpi.payload.bytes"] = len(payload)
    elif isinstance(payload, (str, list, tuple, dict, set)):
        fields["mpi.payload.length"] = len(payload)
    return fields


@contextmanager
def record_mpi_function(
    operation: str,
    *,
    communicator: str,
    group: Callable[[], Iterable[int]],
    rank: int | Callable[[], int],
    kind: MpiOperationKind = "collective",
    direction: MpiDirection | None = None,
    payload: object = _MISSING,
    **fields: MetadataValue,
) -> Generator[None, None, None]:
    """Record shallow MPI metadata through the shared observational scope.

    Args:
        operation: Wrapped MPI method name, without the ``mpi_`` prefix.
        communicator: Logical world, TP, PP, or CP communicator label.
        group: Lazy ordered members in the active parent MPI communicator's rank space.
        rank: Rank in that same space, or a lazy rank query.
        kind: Collective or point-to-point operation.
        direction: Send or receive direction for point-to-point operations.
        payload: Optional object described without serialization or recursive traversal.
            Tensors/NumPy arrays of any shape/dtype provide only shape, dtype and byte
            counts; tensors also provide stride/device. No tensor values are read or
            copied, including during CUDA capture. Other objects provide type and,
            for built-in containers, length; byte buffers provide byte size.
        **fields: Additional direct fields, such as communicator-local root/peer, tag,
            reduction enum or chunk size. Direct NumPy scalars/enums become literals.

    Yields:
        None while the operation executes. Group, rank and payload inspection are
        skipped for disabled/inactive scopes. Metadata/setup/cleanup failures are
        observational and operation exceptions propagate unchanged.
    """

    def metadata() -> dict[str, MetadataValue]:
        values: dict[str, MetadataValue] = {
            "mpi.operation": operation,
            "mpi.communicator": communicator,
            "mpi.group": ",".join(str(int(member)) for member in group()),
            "mpi.kind": kind,
            "mpi.rank": _mpi_metadata_value(rank() if callable(rank) else rank),
        }
        if direction is not None:
            values["mpi.direction"] = direction
        values.update(
            {
                f"mpi.{name}": _mpi_metadata_value(value)
                for name, value in fields.items()
                if value is not None
            }
        )
        if payload is not _MISSING:
            values.update(_mpi_payload_metadata(payload))
        return values

    with record_function_metadata_scope(f"trtllm::mpi_{operation}", "mpi", metadata):
        yield


def _moe_routing_metadata(
    layer_idx: int | None,
    num_experts: int,
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor | None,
) -> dict[str, MetadataValue]:
    selected_experts = cast(list[MetadataValue], token_selected_experts.detach().cpu().tolist())
    routing_weights = (
        None
        if token_final_scales is None
        else cast(list[MetadataValue], token_final_scales.detach().cpu().tolist())
    )
    return {
        "moe.layer": layer_idx,
        "moe.num_tokens": token_selected_experts.shape[0],
        "moe.num_experts": num_experts,
        "moe.experts_per_token": token_selected_experts.shape[1],
        "moe.selected_expert_ids": selected_experts,
        "moe.routing_weights": routing_weights,
    }


def record_moe_routing(
    layer_idx: int | None,
    num_experts: int,
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor | None,
) -> None:
    """Record already-materialized logical routing without changing its tensors.

    Args:
        layer_idx: Model layer index, or None when unavailable.
        num_experts: Logical expert count, not the EPLB slot count.
        token_selected_experts: Integer expert IDs [num_tokens, top_k] in routing order.
        token_final_scales: Floating routing weights of the same shape, or None
            when unavailable. This does not relax a scheduler's weight contract.

    Disabled or inactive calls do not inspect tensors or construct metadata. Active
    eager calls copy tensors to CPU and serialize their values, which synchronizes
    CUDA work and has measurable cost. CUDA current-stream capture skips only this
    observation; existing RouterReplay device capture remains independent. Python
    events do not recur on graph replay. Metadata/setup/cleanup failures are
    discarded at the shared observational boundary; tensor values are not modified.
    """
    if not custom_region_enabled("moe") or not torch.autograd._profiler_enabled():
        return
    try:
        uses_cuda = token_selected_experts.is_cuda or (
            token_final_scales is not None and token_final_scales.is_cuda
        )
        if uses_cuda and torch.cuda.is_current_stream_capturing():
            return
    except RuntimeError:
        # An unavailable CUDA capture-state query must not affect computation.
        return
    with record_function_metadata_scope(
        "trtllm::moe_routing",
        "moe",
        lambda: _moe_routing_metadata(
            layer_idx, num_experts, token_selected_experts, token_final_scales
        ),
    ):
        pass
