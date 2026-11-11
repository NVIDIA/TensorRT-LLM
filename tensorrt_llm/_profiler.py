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

"""Shared, observational PyTorch profiler scopes and JSON metadata."""

import json
import logging
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
    "custom_region_active",
    "custom_region_enabled",
    "encode_metadata",
    "record_function_metadata_scope",
    "warn_profiler_failure",
]

_CUSTOM_REGIONS_ENV = "TLLM_PROFILE_CUSTOM_REGIONS"
_CUSTOM_REGION_NAMES: frozenset[ProfilerRegion] = frozenset(
    {"workflow", "mpi", "nccl", "nanobind", "kv", "moe"}
)
_CUSTOM_REGION_DEFAULTS = _CUSTOM_REGION_NAMES - {"kv", "moe"}
_METADATA_PREFIX = "TRTLLM1:"


def _parse_custom_regions(value: str | None) -> frozenset[ProfilerRegion]:
    """Parse the startup selector, rejecting unknown or repeated region names."""
    if value is None or not value.strip(" \t\r\n"):
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
            f"{','.join(sorted(_CUSTOM_REGION_NAMES))} or none"
        )
    return frozenset(cast(ProfilerRegion, region) for region in regions)


def _load_custom_regions() -> frozenset[ProfilerRegion]:
    """Load the startup selector and warn before falling back to defaults."""
    try:
        return _parse_custom_regions(os.environ.get(_CUSTOM_REGIONS_ENV))
    except ValueError as error:
        logging.getLogger(__name__).warning("%s; using default custom regions", error)
        return _CUSTOM_REGION_DEFAULTS


_CUSTOM_REGIONS = _load_custom_regions()
# Metadata access, literal validation, tensor conversion, and profiler operations
# can fail independently of the observed workload.
_OBSERVATION_ERRORS = (AttributeError, LookupError, TypeError, ValueError, RuntimeError, OSError)


def custom_region_enabled(region: ProfilerRegion) -> bool:
    """Return whether process-start configuration selects a custom region.

    Args:
        region: One of the six supported region categories.

    Returns:
        Whether the cached selector includes the region; this does not enable the profiler.
    """
    return region in _CUSTOM_REGIONS


def custom_region_active(region: ProfilerRegion) -> bool:
    """Check selection and active profiling without adding graph breaks."""
    if torch.compiler.is_compiling():
        return False
    return custom_region_enabled(region) and torch.autograd._profiler_enabled()


def warn_profiler_failure(name: str, region: ProfilerRegion, phase: str, error: Exception) -> None:
    """Warn once per failed observation boundary without affecting the workload."""
    try:
        from .logger import logger

        logger.warning_once(
            f"Skipping {region} profiler {phase} for {name!r}: {type(error).__name__}: "
            f"{error}. Check profiler metadata and configuration if events are missing.",
            key=("profiler_observation", region, name, phase),
        )
    except (ImportError, AttributeError, TypeError, ValueError, RuntimeError, OSError):
        # Diagnostic delivery is observational too, including during imports.
        pass


def _validate_metadata_value(value: MetadataValue, ancestors: set[int] | None = None) -> None:
    """Reject nonfinite values, nonstring keys, unsupported types, and cycles."""
    value_type = type(value)
    # Scalar fields dominate workflow/MPI payloads. Exact types retain the
    # strict built-in contract without repeatedly building a type tuple.
    if value_type is str or value_type is int or value_type is bool or value is None:
        return
    if value_type is float:
        if not math.isfinite(value):
            raise ValueError("Profiler metadata floats must be finite")
        return
    if value_type not in (list, tuple, dict):
        raise TypeError(f"Unsupported profiler metadata value: {value_type.__name__}")
    if ancestors is None:
        ancestors = set()
    identity = id(value)
    if identity in ancestors:
        raise ValueError("Profiler metadata cannot contain cycles")
    ancestors.add(identity)
    try:
        if isinstance(value, dict):
            for key, item in value.items():
                if type(key) is not str:
                    raise TypeError("Profiler metadata keys must be strings")
                _validate_metadata_value(item, ancestors)
        else:
            for item in value:
                _validate_metadata_value(item, ancestors)
    finally:
        ancestors.remove(identity)


def encode_metadata(region: str, metadata: Mapping[str, MetadataValue]) -> str:
    """Encode ``TRTLLM1:<UTF-8 region bytes>:<region><finite JSON object>``.

    Dictionary keys must be strings at every depth. Tuples become JSON arrays;
    unsupported objects, nonfinite numbers, and cycles raise TypeError/ValueError.
    """
    fields = dict(metadata)
    _validate_metadata_value(fields)
    encoded = json.dumps(fields, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return _METADATA_PREFIX + f"{len(region.encode('utf-8'))}:{region}" + encoded


@contextmanager
def record_function_metadata_scope(
    name: str,
    region: ProfilerRegion,
    metadata_factory: Callable[[], str | Mapping[str, MetadataValue]] | None = None,
) -> Generator[None, None, None]:
    """Record a selected active scope without changing the body's outcome.

    Args:
        name: The PyTorch user-scope name.
        region: The category controlled by process-start selection.
        metadata_factory: Optional lazy envelope or typed fields; omit it for a
            metadata-free observational scope.

    Yields:
        None while the wrapped operation executes. ``contextmanager`` converts
        this generator into a context manager, which can also be used as a decorator.

    Attribute, lookup, type/value, runtime, and OS errors from metadata or
    profiler entry/cleanup are warned once per scope and phase, then discarded.
    Other exception types indicate an unsupported metadata-factory failure.
    Exceptions from the body propagate unchanged, including BaseException subclasses.
    """
    if not custom_region_active(region):
        yield
        return

    scope: torch.profiler.record_function | None = None
    try:
        metadata = metadata_factory() if metadata_factory is not None else None
        if metadata is not None and not isinstance(metadata, str):
            metadata = encode_metadata(region, metadata)
    except _OBSERVATION_ERRORS as error:
        warn_profiler_failure(name, region, "metadata", error)
        metadata = None
    try:
        scope = torch.profiler.record_function(name, args=metadata)
        scope.__enter__()
    except _OBSERVATION_ERRORS as error:
        warn_profiler_failure(name, region, "entry", error)
        scope = None
    try:
        yield
    finally:
        if scope is not None:
            try:
                scope.__exit__(None, None, None)
            except _OBSERVATION_ERRORS as error:
                warn_profiler_failure(name, region, "cleanup", error)
