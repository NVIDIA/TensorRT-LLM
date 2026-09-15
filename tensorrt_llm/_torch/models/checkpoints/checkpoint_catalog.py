# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-neutral logical and physical checkpoint metadata.

Schema v1 can represent both name-only sources and byte-addressable sources.
Consumers must check :attr:`CheckpointCatalog.has_complete_byte_ranges`
before scheduling range reads.
"""

import hashlib
import json
from bisect import bisect_left
from dataclasses import dataclass, field
from typing import Collection

CHECKPOINT_CATALOG_SCHEMA_VERSION = 1


def _metadata_id(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class CheckpointObject:
    """One logical object in a checkpoint source.

    ``object_id`` is resolved by the source adapter. It is not a filesystem
    path or an object-store URI. ``version_token`` is an optional immutable
    provider version such as an object generation or ETag.
    """

    object_id: str
    size_bytes: int
    version_token: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.object_id, str) or not self.object_id:
            raise ValueError("checkpoint object_id must be a nonempty string")
        if not isinstance(self.size_bytes, int) or isinstance(self.size_bytes, bool):
            raise ValueError("checkpoint object size_bytes must be an integer")
        if self.size_bytes < 0:
            raise ValueError("checkpoint object size_bytes must be nonnegative")
        if self.version_token is not None and (
            not isinstance(self.version_token, str) or not self.version_token
        ):
            raise ValueError("checkpoint object version_token must be a nonempty string or None")


@dataclass(frozen=True, slots=True)
class CheckpointExtent:
    """One exact byte range backing part or all of a logical tensor."""

    object_id: str
    offset_bytes: int
    length_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.object_id, str) or not self.object_id:
            raise ValueError("checkpoint extent object_id must be a nonempty string")
        for field_name, value in (
            ("offset_bytes", self.offset_bytes),
            ("length_bytes", self.length_bytes),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"checkpoint extent {field_name} must be an integer")
            if value < 0:
                raise ValueError(f"checkpoint extent {field_name} must be nonnegative")

    @property
    def end_offset_bytes(self) -> int:
        return self.offset_bytes + self.length_bytes


@dataclass(frozen=True, slots=True)
class CheckpointTensor:
    """One logical source tensor with optional metadata and physical ranges."""

    name: str
    dtype: str | None = None
    shape: tuple[int, ...] | None = None
    extents: tuple[CheckpointExtent, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("checkpoint tensor name must be a nonempty string")
        if self.dtype is not None and (not isinstance(self.dtype, str) or not self.dtype):
            raise ValueError("checkpoint tensor dtype must be a nonempty string or None")
        if self.shape is not None:
            if not isinstance(self.shape, tuple):
                raise ValueError("checkpoint tensor shape must be a tuple or None")
            if any(
                not isinstance(dim, int) or isinstance(dim, bool) or dim < 0 for dim in self.shape
            ):
                raise ValueError("checkpoint tensor shape dimensions must be nonnegative integers")
        if not isinstance(self.extents, tuple) or not all(
            isinstance(extent, CheckpointExtent) for extent in self.extents
        ):
            raise ValueError("checkpoint tensor extents must be a tuple of CheckpointExtent")


@dataclass(frozen=True, slots=True)
class CheckpointCatalog:
    """Validated, path-free checkpoint metadata shared by loader policies.

    ``catalog_id`` hashes only logical object and tensor metadata; constructing
    a catalog never hashes checkpoint payload bytes.
    """

    objects: tuple[CheckpointObject, ...]
    tensors: tuple[CheckpointTensor, ...]
    schema_version: int = CHECKPOINT_CATALOG_SCHEMA_VERSION
    catalog_id: str = field(init=False)
    _tensor_names: frozenset[str] = field(init=False, repr=False, compare=False)
    _sorted_tensor_names: tuple[str, ...] = field(init=False, repr=False, compare=False)
    _tensors_by_sorted_name: tuple[CheckpointTensor, ...] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.schema_version, int) or isinstance(self.schema_version, bool):
            raise ValueError("checkpoint catalog schema_version must be an integer")
        if self.schema_version != CHECKPOINT_CATALOG_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported checkpoint catalog schema_version: {self.schema_version}"
            )
        if not isinstance(self.objects, tuple) or not all(
            isinstance(obj, CheckpointObject) for obj in self.objects
        ):
            raise ValueError("checkpoint catalog objects must be a tuple of CheckpointObject")
        if not isinstance(self.tensors, tuple) or not all(
            isinstance(tensor, CheckpointTensor) for tensor in self.tensors
        ):
            raise ValueError("checkpoint catalog tensors must be a tuple of CheckpointTensor")
        if not self.tensors:
            raise ValueError("checkpoint catalog must contain at least one tensor")

        object_by_id = {obj.object_id: obj for obj in self.objects}
        if len(object_by_id) != len(self.objects):
            raise ValueError("checkpoint catalog object IDs must be unique")
        tensor_names = frozenset(tensor.name for tensor in self.tensors)
        if len(tensor_names) != len(self.tensors):
            raise ValueError("checkpoint catalog tensor names must be unique")

        for tensor in self.tensors:
            for extent in tensor.extents:
                checkpoint_object = object_by_id.get(extent.object_id)
                if checkpoint_object is None:
                    raise ValueError(
                        f"checkpoint tensor {tensor.name!r} references unknown object "
                        f"{extent.object_id!r}"
                    )
                if extent.end_offset_bytes > checkpoint_object.size_bytes:
                    raise ValueError(
                        f"checkpoint tensor {tensor.name!r} extends beyond object "
                        f"{extent.object_id!r}"
                    )

        # Shared or view-backed tensors may intentionally overlap in a
        # source-neutral catalog. Format adapters can impose stricter layout
        # rules, while range readers must union physical intervals.
        tensors_by_name = tuple(sorted(self.tensors, key=lambda tensor: tensor.name))
        object.__setattr__(self, "_tensor_names", tensor_names)
        object.__setattr__(
            self, "_sorted_tensor_names", tuple(tensor.name for tensor in tensors_by_name)
        )
        object.__setattr__(self, "_tensors_by_sorted_name", tensors_by_name)

        object_payload = [
            {
                "object_id": obj.object_id,
                "size_bytes": obj.size_bytes,
                "version_token": obj.version_token,
            }
            for obj in sorted(self.objects, key=lambda obj: obj.object_id)
        ]
        tensor_payload = [
            {
                "name": tensor.name,
                "dtype": tensor.dtype,
                "shape": tensor.shape,
                "extents": [
                    {
                        "object_id": extent.object_id,
                        "offset_bytes": extent.offset_bytes,
                        "length_bytes": extent.length_bytes,
                    }
                    for extent in tensor.extents
                ],
            }
            for tensor in sorted(self.tensors, key=lambda tensor: tensor.name)
        ]
        object.__setattr__(
            self,
            "catalog_id",
            _metadata_id(
                {
                    "schema_version": self.schema_version,
                    "objects": object_payload,
                    "tensors": tensor_payload,
                }
            ),
        )

    @property
    def tensor_names(self) -> frozenset[str]:
        """Return the stable set of tensor names exposed by this checkpoint."""
        return self._tensor_names

    @property
    def has_complete_byte_ranges(self) -> bool:
        """Whether every logical tensor has at least one validated byte range."""
        return all(tensor.extents for tensor in self.tensors)

    def get_tensor(self, name: str) -> CheckpointTensor:
        """Resolve a source tensor name to its logical metadata and byte ranges."""
        index = bisect_left(self._sorted_tensor_names, name)
        if index == len(self._sorted_tensor_names) or self._sorted_tensor_names[index] != name:
            raise KeyError(name)
        return self._tensors_by_sorted_name[index]

    def select_tensors(self, names: Collection[str]) -> tuple[CheckpointTensor, ...]:
        """Resolve names in deterministic catalog order."""
        requested = frozenset(names)
        unknown = requested - self.tensor_names
        if unknown:
            raise KeyError(f"unknown checkpoint tensors: {sorted(unknown)}")
        return tuple(tensor for tensor in self.tensors if tensor.name in requested)
