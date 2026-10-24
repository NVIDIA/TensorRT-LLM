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
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from typing import Literal, TypeAlias, cast

import torch

ProfilerRegion: TypeAlias = Literal["workflow", "mpi", "nccl", "nanobind", "kv", "moe"]
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
    "ProfilerRegion",
    "custom_region_enabled",
    "record_function_metadata_scope",
]

_CUSTOM_REGIONS_ENV = "TLLM_PROFILE_CUSTOM_REGIONS"
_CUSTOM_REGION_NAMES: frozenset[ProfilerRegion] = frozenset(
    {"workflow", "mpi", "nccl", "nanobind", "kv", "moe"}
)
_CUSTOM_REGION_DEFAULTS = _CUSTOM_REGION_NAMES - {"kv", "moe"}
_METADATA_PREFIX = "TRTLLM1"


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
