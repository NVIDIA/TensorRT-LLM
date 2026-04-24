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

"""DeepSeek V4 checkpoint quantization metadata helpers."""

import json
import re
import struct
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypeAlias

DEEPSEEK_V4_QUANT_METHOD = "deepseek_v4_fp8"
DEEPSEEK_V4_HF_QUANT_METHOD = "fp8"
DEEPSEEK_V4_LINEAR_QUANT_METHOD = "finegrained_fp8"
DEEPSEEK_V4_EXPERT_QUANT_METHOD = "mxfp4"
DEEPSEEK_V4_SCALE_FMT = "ue8m0"
DEEPSEEK_V4_WEIGHT_BLOCK_SIZE = [128, 128]
DEEPSEEK_V4_EXPERT_BLOCK_SIZE = 32

FINEGRAINED_FP8_LINEAR = "finegrained_fp8_linear"
PACKED_MXFP4_EXPERT = "packed_mxfp4_expert"
BF16_OR_F32 = "bf16_or_f32"
INTEGER_METADATA = "integer_metadata"
SKIPPED_MTP = "skipped_mtp"
UNKNOWN = "unknown"

CLASSIFIER_CATEGORIES = (
    FINEGRAINED_FP8_LINEAR,
    PACKED_MXFP4_EXPERT,
    BF16_OR_F32,
    INTEGER_METADATA,
    SKIPPED_MTP,
    UNKNOWN,
)

DEEPSEEK_V4_EXCLUDED_MODULES = (
    "embed",
    "head",
    "*.ffn.gate",
    "*.attn.compressor",
    "*.attn.indexer.compressor",
    "*.attn.indexer.weights_proj",
    "*.norm",
    "*.hc_*",
    "*.attn_sink",
    "mtp.*",
)

TensorMetadata: TypeAlias = Mapping[str, object]
CheckpointMetadata: TypeAlias = Mapping[str, TensorMetadata]


class DeepSeekV4QuantConfigError(ValueError):
    """Raised when DeepSeek V4 quantization metadata is missing or inconsistent."""


_FINEGRAINED_FP8_LINEAR_RE = re.compile(
    r"^layers\.\d+\.(?:"
    r"attn\.(?:wq_a|wq_b|wkv|wo_a|wo_b)|"
    r"attn\.indexer\.wq_b|"
    r"ffn\.shared_experts\.w[123]"
    r")\.(?:weight|scale)$"
)
_PACKED_MXFP4_EXPERT_RE = re.compile(r"^layers\.\d+\.ffn\.experts\.\d+\.w[123]\.(?:weight|scale)$")
_INTEGER_METADATA_RE = re.compile(r"^layers\.\d+\.ffn\.gate\.tid2eid$")
_BF16_F32_PATTERNS = (
    re.compile(r"^(?:embed|head|norm)\.weight$"),
    re.compile(r"^hc_head_(?:base|fn|scale)$"),
    re.compile(r"^layers\.\d+\.attn\.attn_sink$"),
    re.compile(r"^layers\.\d+\.attn\.(?:q_norm|kv_norm)\.weight$"),
    re.compile(r"^layers\.\d+\.(?:attn_norm|ffn_norm)\.weight$"),
    re.compile(r"^layers\.\d+\.ffn\.gate\.(?:weight|bias)$"),
    re.compile(r"^layers\.\d+\.hc_(?:attn|ffn)_(?:base|fn|scale)$"),
    re.compile(
        r"^layers\.\d+\.attn\.compressor\."
        r"(?:ape|norm\.weight|wgate\.weight|wkv\.weight)$"
    ),
    re.compile(
        r"^layers\.\d+\.attn\.indexer\.(?:"
        r"compressor\.(?:ape|norm\.weight|wgate\.weight|wkv\.weight)|"
        r"weights_proj\.weight"
        r")$"
    ),
)


def is_deepseek_v4_fp8_config(config: Mapping[str, object]) -> bool:
    """Return whether an HF config describes a DeepSeek V4 FP8 checkpoint."""
    qconf = config.get("quantization_config")
    if not isinstance(qconf, Mapping):
        return False
    return (
        config.get("model_type") == "deepseek_v4"
        and str(qconf.get("quant_method", "")).lower() == DEEPSEEK_V4_HF_QUANT_METHOD
    )


def read_deepseek_v4_safetensors_metadata(ckpt_dir: str | Path) -> dict[str, dict[str, object]]:
    """Read tensor dtype and shape metadata from safetensors headers only."""
    ckpt_path = Path(ckpt_dir)
    index_path = ckpt_path / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as f:
            index = json.load(f)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, Mapping) or not weight_map:
            raise DeepSeekV4QuantConfigError(
                f"DeepSeek V4 safetensors index is missing a non-empty weight_map: {index_path}"
            )
        safetensors_files = sorted({str(filename) for filename in weight_map.values()})
    else:
        safetensors_files = sorted(path.name for path in ckpt_path.glob("*.safetensors"))

    if not safetensors_files:
        raise DeepSeekV4QuantConfigError(
            f"DeepSeek V4 quantization requires safetensors metadata in {ckpt_path}"
        )

    tensor_metadata: dict[str, dict[str, object]] = {}
    for filename in safetensors_files:
        tensor_metadata.update(_read_safetensors_header(ckpt_path / filename))

    return tensor_metadata


def classify_deepseek_v4_checkpoint(
    config: Mapping[str, object],
    tensor_metadata: CheckpointMetadata,
    *,
    allow_unknown: bool = False,
) -> dict[str, object]:
    """Classify DeepSeek V4 checkpoint tensors by quantization family."""
    if config.get("model_type") != "deepseek_v4":
        raise DeepSeekV4QuantConfigError(
            "DeepSeek V4 classifier requires model_type='deepseek_v4'."
        )
    if not tensor_metadata:
        raise DeepSeekV4QuantConfigError("DeepSeek V4 classifier requires tensor metadata.")

    qconf = _get_quantization_config(config)
    weight_block_size = _get_weight_block_size(qconf)
    _validate_scale_fmt(qconf)

    categories = {category: [] for category in CLASSIFIER_CATEGORIES}
    tensors: dict[str, dict[str, object]] = {}

    for name in sorted(tensor_metadata):
        metadata = tensor_metadata[name]
        dtype = _get_dtype(name, metadata)
        shape = _get_shape(name, metadata)
        category = _classify_tensor_name(name)
        _validate_tensor(name, dtype, shape, category, tensor_metadata, weight_block_size)

        categories[category].append(name)
        tensors[name] = {
            "category": category,
            "dtype": dtype,
            "shape": shape,
        }

    unknown_tensors = categories[UNKNOWN]
    if unknown_tensors and not allow_unknown:
        sample = ", ".join(unknown_tensors[:10])
        raise DeepSeekV4QuantConfigError(f"Unknown DeepSeek V4 checkpoint tensor(s): {sample}")

    _validate_required_quantized_families(categories)
    return {
        "categories": categories,
        "counts": {category: len(names) for category, names in categories.items()},
        "tensors": tensors,
    }


def build_deepseek_v4_quant_config(
    config: Mapping[str, object],
    tensor_metadata: CheckpointMetadata,
    *,
    allow_unknown: bool = False,
) -> dict[str, object]:
    """Build the normalized AutoDeploy quantization config for DeepSeek V4."""
    if not is_deepseek_v4_fp8_config(config):
        raise DeepSeekV4QuantConfigError("Expected DeepSeek V4 HF fp8 quantization config.")

    qconf = _get_quantization_config(config)
    _validate_scale_fmt(qconf)
    weight_block_size = _get_weight_block_size(qconf)
    classifier = classify_deepseek_v4_checkpoint(
        config, tensor_metadata, allow_unknown=allow_unknown
    )

    normalized = dict(qconf)
    normalized.update(
        {
            "quant_method": DEEPSEEK_V4_QUANT_METHOD,
            "hf_quant_method": DEEPSEEK_V4_HF_QUANT_METHOD,
            "linear_quant_method": DEEPSEEK_V4_LINEAR_QUANT_METHOD,
            "expert_quant_method": DEEPSEEK_V4_EXPERT_QUANT_METHOD,
            "scale_fmt": DEEPSEEK_V4_SCALE_FMT,
            "weight_block_size": weight_block_size,
            "expert_block_size": DEEPSEEK_V4_EXPERT_BLOCK_SIZE,
            "exclude_modules": list(DEEPSEEK_V4_EXCLUDED_MODULES),
            "checkpoint_classification": classifier,
        }
    )
    return normalized


def _read_safetensors_header(path: Path) -> dict[str, dict[str, object]]:
    with path.open("rb") as f:
        header_size_bytes = f.read(8)
        if len(header_size_bytes) != 8:
            raise DeepSeekV4QuantConfigError(f"Invalid safetensors header in {path}")
        header_size = struct.unpack("<Q", header_size_bytes)[0]
        header = json.loads(f.read(header_size))

    tensor_metadata: dict[str, dict[str, object]] = {}
    for name, metadata in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(metadata, Mapping):
            raise DeepSeekV4QuantConfigError(f"Invalid metadata for tensor {name} in {path}")
        tensor_metadata[name] = {
            "dtype": metadata.get("dtype"),
            "shape": metadata.get("shape"),
        }
    return tensor_metadata


def _get_quantization_config(config: Mapping[str, object]) -> Mapping[str, object]:
    qconf = config.get("quantization_config")
    if not isinstance(qconf, Mapping):
        raise DeepSeekV4QuantConfigError("DeepSeek V4 config is missing quantization_config.")
    return qconf


def _validate_scale_fmt(qconf: Mapping[str, object]) -> None:
    scale_fmt = str(qconf.get("scale_fmt", "")).lower()
    if scale_fmt != DEEPSEEK_V4_SCALE_FMT:
        raise DeepSeekV4QuantConfigError(
            f"DeepSeek V4 requires scale_fmt='{DEEPSEEK_V4_SCALE_FMT}', got '{scale_fmt}'."
        )


def _get_weight_block_size(qconf: Mapping[str, object]) -> list[int]:
    block_size = qconf.get("weight_block_size")
    if not isinstance(block_size, Sequence) or isinstance(block_size, str):
        raise DeepSeekV4QuantConfigError("DeepSeek V4 requires weight_block_size=[128, 128].")
    normalized = [int(dim) for dim in block_size]
    if normalized != DEEPSEEK_V4_WEIGHT_BLOCK_SIZE:
        raise DeepSeekV4QuantConfigError(
            "DeepSeek V4 requires weight_block_size="
            f"{DEEPSEEK_V4_WEIGHT_BLOCK_SIZE}, got {normalized}."
        )
    return normalized


def _classify_tensor_name(name: str) -> str:
    if name.startswith("mtp."):
        return SKIPPED_MTP
    if _FINEGRAINED_FP8_LINEAR_RE.match(name):
        return FINEGRAINED_FP8_LINEAR
    if _PACKED_MXFP4_EXPERT_RE.match(name):
        return PACKED_MXFP4_EXPERT
    if _INTEGER_METADATA_RE.match(name):
        return INTEGER_METADATA
    if any(pattern.match(name) for pattern in _BF16_F32_PATTERNS):
        return BF16_OR_F32
    return UNKNOWN


def _validate_tensor(
    name: str,
    dtype: str,
    shape: list[int],
    category: str,
    tensor_metadata: CheckpointMetadata,
    weight_block_size: Sequence[int],
) -> None:
    if category == FINEGRAINED_FP8_LINEAR:
        _validate_finegrained_fp8_tensor(name, dtype, shape, tensor_metadata, weight_block_size)
    elif category == PACKED_MXFP4_EXPERT:
        _validate_packed_mxfp4_tensor(name, dtype, shape, tensor_metadata)
    elif category == BF16_OR_F32 and dtype not in {"BF16", "F32"}:
        raise DeepSeekV4QuantConfigError(f"{name} should be BF16/F32, got {dtype}.")
    elif category == INTEGER_METADATA and dtype != "I64":
        raise DeepSeekV4QuantConfigError(f"{name} should be I64 metadata, got {dtype}.")


def _validate_finegrained_fp8_tensor(
    name: str,
    dtype: str,
    shape: list[int],
    tensor_metadata: CheckpointMetadata,
    weight_block_size: Sequence[int],
) -> None:
    _require_2d(name, shape)
    if name.endswith(".weight"):
        if dtype != "F8_E4M3":
            raise DeepSeekV4QuantConfigError(f"{name} should be F8_E4M3, got {dtype}.")
        return

    if dtype != "F8_E8M0":
        raise DeepSeekV4QuantConfigError(f"{name} should be F8_E8M0 scale, got {dtype}.")
    weight_name = name.removesuffix(".scale") + ".weight"
    weight_metadata = tensor_metadata.get(weight_name)
    if weight_metadata is None:
        return
    weight_shape = _get_shape(weight_name, weight_metadata)
    _require_2d(weight_name, weight_shape)
    expected = [
        _ceil_div(weight_shape[0], int(weight_block_size[0])),
        _ceil_div(weight_shape[1], int(weight_block_size[1])),
    ]
    if shape != expected:
        raise DeepSeekV4QuantConfigError(
            f"{name} has shape {shape}, expected {expected} for {weight_name}."
        )


def _validate_packed_mxfp4_tensor(
    name: str,
    dtype: str,
    shape: list[int],
    tensor_metadata: CheckpointMetadata,
) -> None:
    _require_2d(name, shape)
    if name.endswith(".weight"):
        if dtype != "I8":
            raise DeepSeekV4QuantConfigError(f"{name} should be packed I8 FP4, got {dtype}.")
        return

    if dtype != "F8_E8M0":
        raise DeepSeekV4QuantConfigError(f"{name} should be F8_E8M0 scale, got {dtype}.")
    weight_name = name.removesuffix(".scale") + ".weight"
    weight_metadata = tensor_metadata.get(weight_name)
    if weight_metadata is None:
        return
    weight_shape = _get_shape(weight_name, weight_metadata)
    _require_2d(weight_name, weight_shape)
    expected = [
        weight_shape[0],
        _ceil_div(weight_shape[1] * 2, DEEPSEEK_V4_EXPERT_BLOCK_SIZE),
    ]
    if shape != expected:
        raise DeepSeekV4QuantConfigError(
            f"{name} has shape {shape}, expected {expected} for packed FP4 {weight_name}."
        )


def _validate_required_quantized_families(categories: Mapping[str, Sequence[str]]) -> None:
    if not categories[FINEGRAINED_FP8_LINEAR]:
        raise DeepSeekV4QuantConfigError("DeepSeek V4 metadata has no FP8 linear tensors.")
    if not categories[PACKED_MXFP4_EXPERT]:
        raise DeepSeekV4QuantConfigError("DeepSeek V4 metadata has no packed MXFP4 expert tensors.")


def _get_dtype(name: str, metadata: TensorMetadata) -> str:
    dtype = metadata.get("dtype")
    if not isinstance(dtype, str) or not dtype:
        raise DeepSeekV4QuantConfigError(f"Tensor {name} is missing dtype metadata.")
    return dtype.upper()


def _get_shape(name: str, metadata: TensorMetadata) -> list[int]:
    shape = metadata.get("shape")
    if not isinstance(shape, Sequence) or isinstance(shape, str):
        raise DeepSeekV4QuantConfigError(f"Tensor {name} is missing shape metadata.")
    return [int(dim) for dim in shape]


def _require_2d(name: str, shape: Sequence[int]) -> None:
    if len(shape) != 2:
        raise DeepSeekV4QuantConfigError(f"Tensor {name} should be 2D, got shape {list(shape)}.")


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)
