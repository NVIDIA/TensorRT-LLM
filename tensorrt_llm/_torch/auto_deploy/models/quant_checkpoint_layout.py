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

"""Generic checkpoint-layout helpers for pre-quantized HF checkpoints."""

from __future__ import annotations

import fnmatch
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, TypeAlias, runtime_checkable

import torch

TensorMetadata: TypeAlias = Mapping[str, object]
CheckpointMetadata: TypeAlias = Mapping[str, TensorMetadata]

_E8M0_DTYPE = getattr(torch, "float8_e8m0fnu", None)


class QuantizedCheckpointLayoutError(ValueError):
    """Raised when a quantized checkpoint layout cannot consume checkpoint tensors."""


@runtime_checkable
class PackedMXFP4ExpertCheckpointLayout(Protocol):
    """Consumer contract for packed MXFP4 expert tensors in checkpoint layouts."""

    quant_method: str

    def layer_from_runtime_name(self, name: str) -> int | None:
        """Return the checkpoint layer index encoded in a runtime buffer name, if present."""
        ...

    def load_runtime_buffers(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        *,
        layer: int,
        hidden_size: int,
        intermediate_size: int,
        target_gate_up_blocks: str,
        target_gate_up_scales: str,
        target_down_blocks: str,
        target_down_scales: str,
        expert_indices: Sequence[int] | None = None,
        num_experts: int | None = None,
    ) -> None:
        """Pack checkpoint tensors into runtime MXFP4 expert buffers."""
        ...


class QuantCheckpointLayoutRegistry:
    """Registry for model-owned quantized checkpoint layout builders."""

    _registry: dict[str, Callable[[Mapping[str, object]], "QuantizedCheckpointLayout | None"]] = {}

    @classmethod
    def register(
        cls, model_type: str
    ) -> Callable[
        [Callable[[Mapping[str, object]], "QuantizedCheckpointLayout | None"]],
        Callable[[Mapping[str, object]], "QuantizedCheckpointLayout | None"],
    ]:
        def inner(
            builder: Callable[[Mapping[str, object]], "QuantizedCheckpointLayout | None"],
        ) -> Callable[[Mapping[str, object]], "QuantizedCheckpointLayout | None"]:
            cls._registry[model_type] = builder
            return builder

        return inner

    @classmethod
    def build_from_config(cls, config: Mapping[str, object]) -> "QuantizedCheckpointLayout | None":
        model_type = config.get("model_type")
        if not isinstance(model_type, str):
            return None
        builder = cls._registry.get(model_type)
        if builder is None:
            return None
        return builder(config)


@dataclass(frozen=True)
class FineGrainedFP8CheckpointLayout:
    """Checkpoint layout for block-wise FP8 linear weights."""

    weight_name_patterns: tuple[str, ...]
    weight_block_size: tuple[int, int]
    scale_suffix: str = "scale"
    runtime_scale_name: str = "weight_scale_inv"
    scale_fmt: str = "ue8m0"
    weight_dtype: str = "F8_E4M3"
    scale_dtype: str = "F8_E8M0"
    exclude_patterns: tuple[str, ...] = ()
    quant_method: str = "finegrained_fp8"

    def is_weight_targeted(
        self,
        weight_name: str,
        extra_exclude_patterns: Sequence[str] | None = None,
    ) -> bool:
        if not any(re.match(pattern, weight_name) for pattern in self.weight_name_patterns):
            return False
        return not self.is_weight_excluded(weight_name, extra_exclude_patterns)

    def is_weight_excluded(
        self,
        weight_name: str,
        extra_exclude_patterns: Sequence[str] | None = None,
    ) -> bool:
        module_name = weight_name.removesuffix(".weight")
        for pattern in (*self.exclude_patterns, *(extra_exclude_patterns or ())):
            if fnmatch.fnmatchcase(weight_name, pattern) or fnmatch.fnmatchcase(
                module_name, pattern
            ):
                return True
        return False

    def scale_name_for_weight(self, weight_name: str) -> str:
        module_name = weight_name.removesuffix(".weight")
        return f"{module_name}.{self.scale_suffix}"

    def runtime_scale_name_for_weight(self, weight_name: str) -> str:
        module_name = weight_name.rsplit(".", 1)[0]
        return f"{module_name}.{self.runtime_scale_name}"

    def default_scales(self, original_weight_shape: Sequence[int]) -> dict[str, torch.Tensor]:
        if len(original_weight_shape) != 2:
            raise QuantizedCheckpointLayoutError(
                f"FineGrained FP8 weight should be 2D, got {tuple(original_weight_shape)}."
            )
        n, k = (int(dim) for dim in original_weight_shape)
        block_n, block_k = self.weight_block_size
        scale_shape = (math.ceil(n / block_n), math.ceil(k / block_k))
        return {self.runtime_scale_name: torch.ones(scale_shape, dtype=torch.float32)}

    def decode_scale(self, scale: torch.Tensor) -> torch.Tensor:
        if self.scale_fmt.lower() == "ue8m0" and (
            scale.dtype == torch.uint8 or (_E8M0_DTYPE is not None and scale.dtype == _E8M0_DTYPE)
        ):
            raw_exponents = scale.contiguous().view(torch.uint8)
            float_bits = raw_exponents.to(torch.int32) << 23
            float_bits = torch.where(
                raw_exponents == 0,
                torch.full_like(float_bits, 1 << 22),
                float_bits,
            )
            float_bits = torch.where(
                raw_exponents == 255,
                torch.full_like(float_bits, 0x7FC00000),
                float_bits,
            )
            return float_bits.view(torch.float32).reshape(scale.shape)
        return scale.to(torch.float32)

    def validate_checkpoint_metadata(self, tensor_metadata: CheckpointMetadata) -> int:
        count = 0
        for name, metadata in tensor_metadata.items():
            if not isinstance(metadata, Mapping):
                raise QuantizedCheckpointLayoutError(f"Invalid metadata for tensor {name}.")
            if name.endswith(".weight") and self.is_weight_targeted(name):
                self._validate_weight_metadata(name, metadata)
                scale_name = self.scale_name_for_weight(name)
                scale_metadata = tensor_metadata.get(scale_name)
                if not isinstance(scale_metadata, Mapping):
                    raise QuantizedCheckpointLayoutError(
                        f"{name} is missing companion scale {scale_name}."
                    )
                self._validate_scale_metadata(name, metadata, scale_name, scale_metadata)
                count += 1
            elif name.endswith(f".{self.scale_suffix}"):
                weight_name = name.removesuffix(f".{self.scale_suffix}") + ".weight"
                if self.is_weight_targeted(weight_name):
                    weight_metadata = tensor_metadata.get(weight_name)
                    if not isinstance(weight_metadata, Mapping):
                        raise QuantizedCheckpointLayoutError(
                            f"{name} is missing companion weight {weight_name}."
                        )
                    self._validate_scale_metadata(weight_name, weight_metadata, name, metadata)
        if count == 0:
            raise QuantizedCheckpointLayoutError(
                "Checkpoint metadata has no FineGrained FP8 tensors."
            )
        return count

    def validate_scale_shape(
        self,
        weight_name: str,
        weight_shape: Sequence[int] | torch.Tensor,
        scale_name: str,
        scale_shape: Sequence[int] | torch.Tensor,
    ) -> None:
        if isinstance(weight_shape, torch.Tensor):
            weight_shape = weight_shape.shape
        if isinstance(scale_shape, torch.Tensor):
            scale_shape = scale_shape.shape
        _require_2d(weight_name, weight_shape)
        block_n, block_k = self.weight_block_size
        expected = (
            math.ceil(int(weight_shape[0]) / block_n),
            math.ceil(int(weight_shape[1]) / block_k),
        )
        actual = tuple(int(dim) for dim in scale_shape)
        if actual != expected:
            raise QuantizedCheckpointLayoutError(
                f"{scale_name} shape {actual} does not match expected {expected} "
                f"(expected {list(expected)}) for {weight_name}."
            )

    def _validate_weight_metadata(self, name: str, metadata: TensorMetadata) -> None:
        dtype = _get_dtype(name, metadata)
        if dtype != self.weight_dtype:
            raise QuantizedCheckpointLayoutError(
                f"{name} should be {self.weight_dtype}, got {dtype}."
            )
        _require_2d(name, _get_shape(name, metadata))

    def _validate_scale_metadata(
        self,
        weight_name: str,
        weight_metadata: TensorMetadata,
        scale_name: str,
        scale_metadata: TensorMetadata,
    ) -> None:
        dtype = _get_dtype(scale_name, scale_metadata)
        if dtype != self.scale_dtype:
            raise QuantizedCheckpointLayoutError(
                f"{scale_name} should be {self.scale_dtype} scale, got {dtype}."
            )
        self.validate_scale_shape(
            weight_name,
            _get_shape(weight_name, weight_metadata),
            scale_name,
            _get_shape(scale_name, scale_metadata),
        )


@dataclass(frozen=True)
class QuantizedCheckpointLayout:
    """Quantized checkpoint layout selected by a model-owned layout builder."""

    finegrained_fp8: FineGrainedFP8CheckpointLayout | None = None
    checkpoint_consumers: tuple[object, ...] = ()
    extra_model_kwargs: Mapping[str, object] = field(default_factory=dict)
    extra_quant_config: Mapping[str, object] = field(default_factory=dict)

    def validate_checkpoint_metadata(self, tensor_metadata: CheckpointMetadata) -> None:
        if self.finegrained_fp8 is not None:
            self.finegrained_fp8.validate_checkpoint_metadata(tensor_metadata)
        for consumer in self.checkpoint_consumers:
            validate = getattr(consumer, "validate_checkpoint_metadata", None)
            if validate is None:
                raise QuantizedCheckpointLayoutError(
                    f"Checkpoint consumer {type(consumer).__name__} cannot validate metadata."
                )
            validate(tensor_metadata)

    def apply_to_quant_config(self, quant_config: Mapping[str, object]) -> dict[str, object]:
        normalized = dict(quant_config)
        normalized["checkpoint_layout"] = self
        if self.finegrained_fp8 is not None:
            scale_fmt = normalized.get("scale_fmt")
            if scale_fmt is not None and str(scale_fmt).lower() != self.finegrained_fp8.scale_fmt:
                raise QuantizedCheckpointLayoutError(
                    "Quantized checkpoint layout requires "
                    f"scale_fmt='{self.finegrained_fp8.scale_fmt}', got '{scale_fmt}'."
                )
            block_size = normalized.get("weight_block_size")
            if block_size is not None:
                parsed_block_size = _normalize_block_size("weight_block_size", block_size)
                if parsed_block_size != self.finegrained_fp8.weight_block_size:
                    raise QuantizedCheckpointLayoutError(
                        "Quantized checkpoint layout requires "
                        f"weight_block_size={list(self.finegrained_fp8.weight_block_size)}, "
                        f"got {list(parsed_block_size)}."
                    )
            normalized["linear_quant_method"] = self.finegrained_fp8.quant_method
            normalized["scale_fmt"] = self.finegrained_fp8.scale_fmt
            normalized["weight_block_size"] = list(self.finegrained_fp8.weight_block_size)
            excludes = list(normalized.get("exclude_modules") or [])
            for pattern in self.finegrained_fp8.exclude_patterns:
                if pattern not in excludes:
                    excludes.append(pattern)
            normalized["exclude_modules"] = excludes
        normalized.update(self.extra_quant_config)
        return normalized


def _get_dtype(name: str, metadata: TensorMetadata) -> str:
    dtype = metadata.get("dtype")
    if not isinstance(dtype, str) or not dtype:
        raise QuantizedCheckpointLayoutError(f"Tensor {name} is missing dtype metadata.")
    return dtype.upper()


def _get_shape(name: str, metadata: TensorMetadata) -> list[int]:
    shape = metadata.get("shape")
    if not isinstance(shape, Sequence) or isinstance(shape, str):
        raise QuantizedCheckpointLayoutError(f"Tensor {name} is missing shape metadata.")
    try:
        return [int(dim) for dim in shape]
    except (TypeError, ValueError) as error:
        raise QuantizedCheckpointLayoutError(
            f"Tensor {name} has invalid shape metadata {shape}."
        ) from error


def _require_2d(name: str, shape: Sequence[int]) -> None:
    if len(shape) != 2:
        raise QuantizedCheckpointLayoutError(
            f"Tensor {name} should be 2D, got shape {list(shape)}."
        )


def _normalize_block_size(name: str, value: object) -> tuple[int, int]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise QuantizedCheckpointLayoutError(f"{name} should be a two-element integer sequence.")
    try:
        normalized = tuple(int(dim) for dim in value)
    except (TypeError, ValueError) as error:
        raise QuantizedCheckpointLayoutError(
            f"{name} should contain integer values, got {value}."
        ) from error
    if len(normalized) != 2 or any(dim <= 0 for dim in normalized):
        raise QuantizedCheckpointLayoutError(
            f"{name} should be a two-element positive integer sequence, got {value}."
        )
    return normalized
