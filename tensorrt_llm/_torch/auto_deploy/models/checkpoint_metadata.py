# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""Safetensors checkpoint metadata helpers."""

from __future__ import annotations

import json
import struct
from collections.abc import Mapping
from pathlib import Path

from .quant_checkpoint_layout import QuantizedCheckpointLayoutError


def has_safetensors_metadata(ckpt_dir: str | Path) -> bool:
    ckpt_path = Path(ckpt_dir)
    index_path = ckpt_path / "model.safetensors.index.json"
    if index_path.exists():
        index = _read_safetensors_index(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, Mapping) or not weight_map:
            raise QuantizedCheckpointLayoutError(
                f"safetensors index is missing a non-empty weight_map: {index_path}"
            )
        return any((ckpt_path / str(filename)).exists() for filename in set(weight_map.values()))
    return any(path.name.endswith(".safetensors") for path in ckpt_path.iterdir())


def read_safetensors_metadata(ckpt_dir: str | Path) -> dict[str, dict[str, object]]:
    ckpt_path = Path(ckpt_dir)
    index_path = ckpt_path / "model.safetensors.index.json"
    if index_path.exists():
        index = _read_safetensors_index(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, Mapping) or not weight_map:
            raise QuantizedCheckpointLayoutError(
                f"safetensors index is missing a non-empty weight_map: {index_path}"
            )
        safetensors_files = sorted({str(filename) for filename in weight_map.values()})
    else:
        safetensors_files = sorted(path.name for path in ckpt_path.glob("*.safetensors"))

    if not safetensors_files:
        raise QuantizedCheckpointLayoutError(
            f"Quantized checkpoint layout requires safetensors metadata in {ckpt_path}"
        )

    tensor_metadata: dict[str, dict[str, object]] = {}
    for filename in safetensors_files:
        tensor_metadata.update(_read_safetensors_header(ckpt_path / filename))
    return tensor_metadata


def _read_safetensors_index(path: Path) -> Mapping[str, object]:
    try:
        with path.open("r", encoding="utf-8") as f:
            index = json.load(f)
    except json.JSONDecodeError as error:
        raise QuantizedCheckpointLayoutError(f"Invalid safetensors index JSON: {path}") from error
    if not isinstance(index, Mapping):
        raise QuantizedCheckpointLayoutError(f"Invalid safetensors index JSON: {path}")
    return index


def _read_safetensors_header(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        raise QuantizedCheckpointLayoutError(f"safetensors file not found: {path}")

    with path.open("rb") as f:
        header_size_bytes = f.read(8)
        if len(header_size_bytes) != 8:
            raise QuantizedCheckpointLayoutError(f"Invalid safetensors header in {path}")
        header_size = struct.unpack("<Q", header_size_bytes)[0]
        header_bytes = f.read(header_size)
        if len(header_bytes) != header_size:
            raise QuantizedCheckpointLayoutError(f"Truncated safetensors header in {path}")
        try:
            header = json.loads(header_bytes)
        except json.JSONDecodeError as error:
            raise QuantizedCheckpointLayoutError(
                f"Invalid safetensors header JSON: {path}"
            ) from error

    if not isinstance(header, Mapping):
        raise QuantizedCheckpointLayoutError(f"Invalid safetensors header in {path}")

    tensor_metadata: dict[str, dict[str, object]] = {}
    for name, metadata in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(metadata, Mapping):
            raise QuantizedCheckpointLayoutError(f"Invalid metadata for tensor {name} in {path}")
        tensor_metadata[name] = {"dtype": metadata.get("dtype"), "shape": metadata.get("shape")}
    return tensor_metadata
