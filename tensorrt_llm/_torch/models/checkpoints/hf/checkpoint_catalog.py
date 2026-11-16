# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SafeTensors metadata adapter for :mod:`checkpoint_catalog`."""

import json
import os
import struct
from collections.abc import Sequence

from tensorrt_llm._torch.models.checkpoints.checkpoint_catalog import (
    CheckpointCatalog,
    CheckpointExtent,
    CheckpointObject,
    CheckpointTensor,
)

_HEADER_PREFIX_BYTES = 8
_MAX_HEADER_BYTES = 100_000_000


def _read_safetensors_header(path: str) -> tuple[int, int, dict[str, object]]:
    try:
        with open(path, "rb") as checkpoint_file:
            prefix = checkpoint_file.read(_HEADER_PREFIX_BYTES)
            if len(prefix) != _HEADER_PREFIX_BYTES:
                raise ValueError("SafeTensors file is shorter than its header prefix")
            header_bytes = struct.unpack("<Q", prefix)[0]
            file_size = os.fstat(checkpoint_file.fileno()).st_size
            if header_bytes > _MAX_HEADER_BYTES:
                raise ValueError(
                    f"SafeTensors header exceeds {_MAX_HEADER_BYTES} bytes: {header_bytes}"
                )
            if header_bytes > file_size - _HEADER_PREFIX_BYTES:
                raise ValueError("SafeTensors header extends beyond the file")
            encoded_header = checkpoint_file.read(header_bytes)
    except OSError as error:
        raise ValueError(f"failed to read SafeTensors header for {path!r}") from error

    try:
        header = json.loads(encoded_header.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid SafeTensors JSON header for {path!r}") from error
    if not isinstance(header, dict):
        raise ValueError(f"SafeTensors header for {path!r} must be a JSON object")
    return file_size, header_bytes, header


def build_safetensors_checkpoint_catalog(weight_files: Sequence[str]) -> CheckpointCatalog:
    """Build a metadata-only catalog over an already-selected file sequence."""
    if not weight_files:
        raise ValueError("SafeTensors checkpoint catalog requires at least one file")

    objects = []
    tensors = []
    for path in sorted(weight_files, key=os.path.basename):
        object_id = os.path.basename(path)
        size_bytes, header_bytes, header = _read_safetensors_header(path)
        objects.append(CheckpointObject(object_id=object_id, size_bytes=size_bytes))
        data_start = _HEADER_PREFIX_BYTES + header_bytes
        for name, tensor_info in header.items():
            if name == "__metadata__":
                continue
            if not isinstance(name, str) or not isinstance(tensor_info, dict):
                raise ValueError(f"invalid tensor entry in SafeTensors object {object_id!r}")
            try:
                dtype = tensor_info["dtype"]
                shape = tensor_info["shape"]
                data_offsets = tensor_info["data_offsets"]
            except KeyError as error:
                raise ValueError(
                    f"incomplete tensor entry {name!r} in SafeTensors object {object_id!r}"
                ) from error
            if not isinstance(dtype, str):
                raise ValueError(f"invalid dtype for SafeTensors tensor {name!r}")
            if not isinstance(shape, list):
                raise ValueError(f"invalid shape for SafeTensors tensor {name!r}")
            if not isinstance(data_offsets, list) or len(data_offsets) != 2:
                raise ValueError(f"invalid data_offsets for SafeTensors tensor {name!r}")
            start, end = data_offsets
            if (
                not isinstance(start, int)
                or isinstance(start, bool)
                or not isinstance(end, int)
                or isinstance(end, bool)
                or start < 0
                or end < start
            ):
                raise ValueError(f"invalid data_offsets for SafeTensors tensor {name!r}")
            tensors.append(
                CheckpointTensor(
                    name=name,
                    dtype=dtype,
                    shape=tuple(shape),
                    extents=(
                        CheckpointExtent(
                            object_id=object_id,
                            offset_bytes=data_start + start,
                            length_bytes=end - start,
                        ),
                    ),
                )
            )

    extents_by_object: dict[str, list[tuple[str, CheckpointExtent]]] = {
        obj.object_id: [] for obj in objects
    }
    for tensor in tensors:
        for extent in tensor.extents:
            extents_by_object[extent.object_id].append((tensor.name, extent))
    for object_id, object_extents in extents_by_object.items():
        ordered_extents = sorted(
            object_extents,
            key=lambda item: (item[1].offset_bytes, item[0]),
        )
        for previous, current in zip(ordered_extents, ordered_extents[1:]):
            if previous[1].end_offset_bytes > current[1].offset_bytes:
                raise ValueError(
                    f"SafeTensors entries {previous[0]!r} and "
                    f"{current[0]!r} overlap in object {object_id!r}"
                )

    object_order = {obj.object_id: index for index, obj in enumerate(objects)}
    ordered_tensors = tuple(
        sorted(
            tensors,
            key=lambda tensor: (
                object_order[tensor.extents[0].object_id],
                tensor.extents[0].offset_bytes,
                tensor.name,
            ),
        )
    )
    return CheckpointCatalog(objects=tuple(objects), tensors=ordered_tensors)
