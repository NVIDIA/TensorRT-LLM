# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read Comfy Kitchen NVFP4 tensors without dequantizing their weights.

The ``nvfp4`` Comfy Kitchen layout stores even elements in the high nibble
and E4M3 block scales in cuBLAS 128-row/four-column tiles. ModelOpt-style
checkpoints consumed by Linear use low-nibble-first weights and linear scales.
The global scale has the same dequantization convention in both formats.

Layout conversion is adapted from comfy-kitchen's float_utils.py; metadata
handling follows SGLang's checkpoint contract. See COMFY_IMPORT_NOTICE for
pinned sources, attribution, and the scope of the adaptations.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
from safetensors import safe_open

_METADATA_SUFFIX = ".comfy_quant"
_MAX_METADATA_BYTES = 4096


def comfy_metadata_from_header(metadata: dict[str, str] | None) -> dict[str, torch.Tensor]:
    """Translate the Comfy version-1.0 header into current per-layer metadata.

    The header's layers object and tensor markers describe the same recipes.
    Only metadata is normalized here; tensor payloads are left untouched.
    """
    if not metadata or "_quantization_metadata" not in metadata:
        return {}
    raw = metadata["_quantization_metadata"]
    try:
        config = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid Comfy quantization metadata header") from exc
    # The legacy header is shared with existing FP8 loaders. Claim only
    # explicitly tagged NVFP4 recipes, then validate the entire recipe.
    layers = config.get("layers", {}) if isinstance(config, dict) else {}
    if not isinstance(layers, dict) or not any(
        isinstance(recipe, dict) and recipe.get("format") == "nvfp4" for recipe in layers.values()
    ):
        return {}
    if len(raw) > 1024 * 1024:
        raise ValueError("Comfy quantization metadata header is too large")
    if config.get("format_version") != "1.0":
        raise ValueError("Unsupported Comfy quantization metadata header version")
    if config.keys() - {"format_version", "layers"}:
        raise ValueError("Unsupported Comfy quantization metadata header fields")
    layers = config.get("layers")
    if not isinstance(layers, dict) or not layers:
        raise ValueError("Comfy quantization metadata must declare layers")
    result = {}
    for name, recipe in layers.items():
        if not name or not isinstance(recipe, dict):
            raise ValueError("Invalid layer in Comfy quantization metadata")
        tensor = torch.tensor(list(json.dumps(recipe).encode()), dtype=torch.uint8)
        _validate_metadata(tensor, name)
        result[name + _METADATA_SUFFIX] = tensor
    return result


@dataclass(frozen=True)
class ComfyQuantization:
    """Quantized source modules and their logical, uncompressed matrix shapes."""

    nvfp4_layers: dict[str, tuple[int, int]]
    weight_shapes: dict[str, tuple[int, ...]]
    adaln_curve_shape: tuple[int, int] | None = None


def _validate_metadata(metadata: torch.Tensor, name: str) -> dict:
    if metadata.dtype != torch.uint8 or metadata.ndim != 1:
        raise ValueError(f"{name}: comfy_quant must be a one-dimensional uint8 JSON tensor")
    if not 0 < metadata.numel() <= _MAX_METADATA_BYTES:
        raise ValueError(f"{name}: invalid comfy_quant metadata size {metadata.numel()}")
    try:
        config = json.loads(bytes(metadata.cpu().tolist()))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name}: invalid comfy_quant JSON") from exc
    if not isinstance(config, dict):
        raise ValueError(f"{name}: comfy_quant must contain a JSON object")
    if config.get("format") != "nvfp4":
        raise ValueError(f"{name}: unsupported Comfy quantization format {config.get('format')!r}")
    unknown = config.keys() - {"format", "full_precision_matrix_mult"}
    if unknown:
        raise ValueError(f"{name}: unsupported comfy_quant fields {sorted(unknown)}")
    if config.get("full_precision_matrix_mult", False) is not False:
        raise ValueError(f"{name}: full_precision_matrix_mult is not an NVFP4 GEMM recipe")
    config.pop("full_precision_matrix_mult", None)
    return config


def _matrix_shape(shape: tuple[int, ...], dtype: torch.dtype, name: str) -> tuple[int, int]:
    if dtype != torch.uint8 or len(shape) != 2 or min(shape) <= 0:
        raise ValueError(
            f"{name}: Comfy NVFP4 weight must be a nonempty two-dimensional uint8 tensor"
        )
    rows, packed_cols = shape
    cols = packed_cols * 2
    if cols % 16:
        raise ValueError(f"{name}: NVFP4 input dimension {cols} must be divisible by 16")
    return rows, cols


def inspect_comfy_quantization(paths: list[Path]) -> ComfyQuantization | None:
    """Detect explicit Comfy metadata while reading only headers and small tensors.

    All shards of one component must be supplied. A directory of alternative
    model variants is not a sharded checkpoint; duplicate tensor names fail.
    This function does not infer quantization from a filename or uint8 dtype.
    """
    metadata: dict[str, torch.Tensor] = {}
    shapes: dict[str, tuple[int, ...]] = {}
    dtypes: dict[str, str] = {}
    duplicates = set()
    for path in paths:
        with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
            for key, value in comfy_metadata_from_header(checkpoint.metadata()).items():
                if key in metadata and _validate_metadata(metadata[key], key) != _validate_metadata(
                    value, key
                ):
                    raise ValueError(f"Conflicting Comfy metadata for {key}")
                metadata[key] = value
            for key in checkpoint.keys():
                if key in shapes:
                    duplicates.add(key)
                tensor = checkpoint.get_slice(key)
                shapes[key] = tuple(tensor.get_shape())
                dtypes[key] = tensor.get_dtype()
                if key.endswith(_METADATA_SUFFIX):
                    if len(shapes[key]) != 1 or not 0 < shapes[key][0] <= _MAX_METADATA_BYTES:
                        raise ValueError(f"{key}: invalid comfy_quant metadata shape {shapes[key]}")
                    value = checkpoint.get_tensor(key)
                    _validate_metadata(value, key)
                    if key in metadata and _validate_metadata(
                        metadata[key], key
                    ) != _validate_metadata(value, key):
                        raise ValueError(f"Conflicting Comfy metadata for {key}")
                    metadata[key] = value
    if not metadata:
        return None
    if duplicates:
        raise ValueError(
            f"Duplicate checkpoint tensor {min(duplicates)!r}; select one model variant"
        )
    layers = {}
    for key, value in metadata.items():
        name = key.removesuffix(_METADATA_SUFFIX)
        _validate_metadata(value, name)
        weight_key = f"{name}.weight"
        if weight_key not in shapes or dtypes[weight_key] != "U8":
            raise ValueError(f"{name}: missing packed uint8 NVFP4 weight")
        rows, cols = _matrix_shape(shapes[weight_key], torch.uint8, name)
        scale_key, global_key = f"{name}.weight_scale", f"{name}.weight_scale_2"
        padded_shape = ((rows + 127) // 128 * 128, (cols // 16 + 3) // 4 * 4)
        if scale_key not in shapes or shapes[scale_key] != padded_shape:
            raise ValueError(f"{name}: expected tiled NVFP4 block scales with shape {padded_shape}")
        if dtypes[scale_key] not in ("F8_E4M3", "U8"):
            raise ValueError(f"{name}: block scales must contain E4M3 bytes")
        if global_key not in shapes or shapes[global_key] not in ((), (1,)):
            raise ValueError(f"{name}: missing scalar NVFP4 global scale")
        if dtypes[global_key] != "F32":
            raise ValueError(f"{name}: NVFP4 global scale must be float32")
        layers[name] = (rows, cols)
    curve_shape = None
    for key in ("adaln_t_table", "model.diffusion_model.adaln_t_table"):
        if key in shapes:
            if (
                len(shapes[key]) != 2
                or shapes[key][0] < 2
                or shapes[key][1] < 1
                or dtypes[key] != "F32"
            ):
                raise ValueError(f"{key}: expected a float32 AdaLN curve with at least two rows")
            curve_shape = shapes[key]
    return ComfyQuantization(
        layers,
        {key: shape for key, shape in shapes.items() if key.endswith(".weight")},
        curve_shape,
    )


def _linear_block_scales(scales: torch.Tensor, rows: int, cols: int, name: str) -> torch.Tensor:
    scale_cols = cols // 16
    row_tiles, col_tiles = (rows + 127) // 128, (scale_cols + 3) // 4
    expected = (row_tiles * 128, col_tiles * 4)
    if scales.dtype not in (torch.uint8, torch.float8_e4m3fn) or tuple(scales.shape) != expected:
        raise ValueError(f"{name}: expected E4M3 tiled block scales with shape {expected}")
    # Adapted from comfy-kitchen float_utils.from_blocked. Operate on bytes
    # because float8 indexing/transpose support differs between devices.
    blocked = scales.contiguous().view(torch.uint8).reshape(-1, 32, 16)
    groups = blocked.reshape(-1, 32, 4, 4).transpose(1, 2)
    tiles = groups.reshape(row_tiles, col_tiles, 128, 4)
    linear = tiles.permute(0, 2, 1, 3).reshape(expected)
    linear = linear[:rows, :scale_cols].contiguous().view(torch.float8_e4m3fn)
    values = linear.float()
    if not torch.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"{name}: NVFP4 block scales must be finite and nonnegative")
    return linear


def convert_comfy_nvfp4(weights: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Normalize explicitly tagged NVFP4 layers; leave other tensors unchanged.

    Keys retain the source model's names. Model adapters own renaming and
    fused-projection changes. The input mapping and tensor storage are not
    modified. Metadata is consumed so the conversion is idempotent.
    """
    converted = dict(weights)
    for key, metadata in weights.items():
        if not key.endswith(_METADATA_SUFFIX):
            continue
        name = key.removesuffix(_METADATA_SUFFIX)
        _validate_metadata(metadata, name)
        weight_key = f"{name}.weight"
        scale_key, global_key = f"{name}.weight_scale", f"{name}.weight_scale_2"
        missing = {weight_key, scale_key, global_key} - weights.keys()
        if missing:
            raise ValueError(f"{name}: missing Comfy NVFP4 tensors {sorted(missing)}")
        weight = weights[weight_key]
        rows, cols = _matrix_shape(tuple(weight.shape), weight.dtype, name)
        global_scale = weights[global_key]
        if global_scale.dtype != torch.float32 or tuple(global_scale.shape) not in ((), (1,)):
            raise ValueError(f"{name}: NVFP4 global scale must be a float32 scalar")
        if not torch.isfinite(global_scale).all() or (global_scale <= 0).any():
            raise ValueError(f"{name}: NVFP4 global scale must be finite and positive")
        converted[weight_key] = (weight << 4) | (weight >> 4)
        converted[scale_key] = _linear_block_scales(weights[scale_key], rows, cols, name)
        # Imported checkpoints use dynamic activation quantization. A source
        # calibration tensor must not be interpreted as a TRT-LLM input scale.
        converted.pop(f"{name}.input_scale", None)
        del converted[key]
    return converted
