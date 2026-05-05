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
import json
import math
import operator
import re
import struct
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import torch

TensorMetadata: TypeAlias = Mapping[str, object]
CheckpointMetadata: TypeAlias = Mapping[str, TensorMetadata]

_E8M0_DTYPE = getattr(torch, "float8_e8m0fnu", None)


class QuantizedCheckpointLayoutError(ValueError):
    """Raised when a quantized checkpoint layout cannot consume checkpoint tensors."""


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
class PackedExpertTensorKey:
    """Parsed packed expert tensor key."""

    layer: int
    expert: int
    projection: str
    tensor_kind: str


@dataclass(frozen=True)
class PackedMxfp4Experts:
    """Packed MXFP4 expert tensors in runtime layout."""

    gate_up_blocks: torch.Tensor
    gate_up_scales: torch.Tensor
    down_blocks: torch.Tensor
    down_scales: torch.Tensor
    expert_indices: tuple[int, ...]
    hidden_size: int
    intermediate_size: int


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

    def is_target_weight_name(self, weight_name: str) -> bool:
        return self.is_weight_targeted(weight_name)

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

    def is_excluded_weight_name(self, weight_name: str) -> bool:
        return self.is_weight_excluded(weight_name)

    def scale_name_for_weight(self, weight_name: str) -> str:
        module_name = weight_name.removesuffix(".weight")
        return f"{module_name}.{self.scale_suffix}"

    def checkpoint_scale_name(self, weight_name: str) -> str:
        return self.scale_name_for_weight(weight_name)

    def runtime_scale_name_for_weight(self, weight_name: str) -> str:
        module_name = weight_name.rsplit(".", 1)[0]
        return f"{module_name}.{self.runtime_scale_name}"

    def scale_target_name(self, weight_name: str) -> str:
        return self.runtime_scale_name_for_weight(weight_name)

    def default_scales(self, original_weight_shape: Sequence[int]) -> dict[str, torch.Tensor]:
        if len(original_weight_shape) != 2:
            raise QuantizedCheckpointLayoutError(
                f"FineGrained FP8 weight should be 2D, got {tuple(original_weight_shape)}."
            )
        n, k = (int(dim) for dim in original_weight_shape)
        block_n, block_k = self.weight_block_size
        scale_shape = (math.ceil(n / block_n), math.ceil(k / block_k))
        return {self.runtime_scale_name: torch.ones(scale_shape, dtype=torch.float32)}

    def load_weight_scale(
        self,
        state_dict: dict[str, torch.Tensor],
        weight_key: str,
        *,
        extra_exclude_patterns: Sequence[str] | None = None,
    ) -> bool:
        if weight_key not in state_dict or not self.is_weight_targeted(
            weight_key, extra_exclude_patterns
        ):
            return False

        weight = state_dict[weight_key]
        if weight.dtype != torch.float8_e4m3fn:
            return False

        scale_name = self.scale_name_for_weight(weight_key)
        if scale_name not in state_dict:
            return False

        scale = state_dict[scale_name]
        self.validate_scale_shape(weight_key, weight.shape, scale_name, scale.shape)
        state_dict.pop(scale_name)
        state_dict[self.runtime_scale_name_for_weight(weight_key)] = self.decode_scale(scale)
        return True

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

    def validate_weight_scale_shape(
        self,
        weight_name: str,
        weight: torch.Tensor,
        scale_name: str,
        scale: torch.Tensor,
    ) -> None:
        self.validate_scale_shape(weight_name, weight.shape, scale_name, scale.shape)

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
class PackedMxfp4ExpertsCheckpointLayout:
    """Checkpoint layout for packed MXFP4 routed expert tensors."""

    expert_key_pattern: str
    runtime_gate_up_order: tuple[str, str]
    runtime_down_projection: str
    key_template: str = "layers.{layer}.ffn.experts.{expert}.{projection}.{tensor_kind}"
    projection_names: tuple[str, ...] = ("w1", "w2", "w3")
    expert_block_size: int = 32
    weight_dtypes: tuple[str, ...] = ("I8",)
    scale_dtype: str = "F8_E8M0"
    scale_fmt: str = "ue8m0"
    layer_name_pattern: str = r"(?:^|\.)layers\.(?P<layer>\d+)\.ffn\."
    quant_method: str = "mxfp4"

    def parse_key(self, name: str) -> PackedExpertTensorKey | None:
        match = re.fullmatch(self.expert_key_pattern, name)
        if match is None:
            return None
        projection = match.group("projection")
        groups = match.groupdict()
        tensor_kind = groups.get("kind") or groups.get("tensor_kind")
        if projection not in self.projection_names or tensor_kind not in ("weight", "scale"):
            return None
        return PackedExpertTensorKey(
            layer=int(match.group("layer")),
            expert=int(match.group("expert")),
            projection=projection,
            tensor_kind=tensor_kind,
        )

    def layer_from_runtime_name(self, name: str) -> int | None:
        match = re.search(self.layer_name_pattern, name)
        if match is None:
            return None
        return int(match.group("layer"))

    def validate_checkpoint_metadata(self, tensor_metadata: CheckpointMetadata) -> int:
        parsed_metadata: dict[tuple[int, int, str], dict[str, TensorMetadata]] = {}
        expert_projections: dict[tuple[int, int], set[str]] = {}
        for name, metadata in tensor_metadata.items():
            parsed = self.parse_key(name)
            if parsed is None:
                continue
            if not isinstance(metadata, Mapping):
                raise QuantizedCheckpointLayoutError(f"Invalid metadata for tensor {name}.")
            key = (parsed.layer, parsed.expert, parsed.projection)
            parsed_metadata.setdefault(key, {})[parsed.tensor_kind] = metadata
            expert_projections.setdefault((parsed.layer, parsed.expert), set()).add(
                parsed.projection
            )

        for (layer, expert, projection), tensors in parsed_metadata.items():
            weight_name = self.format_key(layer, expert, projection, "weight")
            scale_name = self.format_key(layer, expert, projection, "scale")
            weight_metadata = tensors.get("weight")
            scale_metadata = tensors.get("scale")
            if not isinstance(weight_metadata, Mapping):
                raise QuantizedCheckpointLayoutError(
                    f"{scale_name} is missing companion weight {weight_name}."
                )
            if not isinstance(scale_metadata, Mapping):
                raise QuantizedCheckpointLayoutError(
                    f"{weight_name} is missing companion scale {scale_name}."
                )
            self._validate_weight_metadata(weight_name, weight_metadata)
            self._validate_scale_metadata(weight_name, weight_metadata, scale_name, scale_metadata)

        for (layer, expert), projections in expert_projections.items():
            for projection in self._required_projection_names():
                if projection in projections:
                    continue
                weight_name = self.format_key(layer, expert, projection, "weight")
                scale_name = self.format_key(layer, expert, projection, "scale")
                raise QuantizedCheckpointLayoutError(
                    "Packed MXFP4 expert metadata is missing required projection "
                    f"{projection} for layer {layer}, expert {expert}: "
                    f"{weight_name} and {scale_name}."
                )

        if not parsed_metadata:
            raise QuantizedCheckpointLayoutError(
                "Checkpoint metadata has no packed MXFP4 expert tensors."
            )
        return len(parsed_metadata)

    def format_key(self, layer: int, expert: int, projection: str, tensor_kind: str) -> str:
        return self.key_template.format(
            layer=layer,
            expert=expert,
            projection=projection,
            kind=tensor_kind,
            tensor_kind=tensor_kind,
        )

    def pack_experts(
        self,
        state_dict: Mapping[str, torch.Tensor],
        *,
        layer: int,
        hidden_size: int,
        intermediate_size: int,
        expert_indices: Sequence[int] | None = None,
        num_experts: int | None = None,
    ) -> PackedMxfp4Experts:
        layer = _validate_nonnegative_int("layer", layer)
        hidden_size = _validate_positive_divisible_by(
            "hidden_size", hidden_size, self.expert_block_size
        )
        intermediate_size = _validate_positive_divisible_by(
            "intermediate_size", intermediate_size, self.expert_block_size
        )

        expert_tensors = self._collect_expert_tensors(state_dict, layer)
        expert_order = _normalize_expert_order(
            expert_tensors,
            expert_indices=expert_indices,
            num_experts=num_experts,
        )

        gate_up_blocks = []
        gate_up_scales = []
        down_blocks = []
        down_scales = []
        for expert in expert_order:
            expert_map = self._get_required_expert_map(expert_tensors, layer, expert)
            gate_up_blocks.append(
                torch.cat(
                    [
                        self._load_weight_blocks(
                            expert_map,
                            layer,
                            expert,
                            projection,
                            expected_shape=(intermediate_size, hidden_size // 2),
                            packed_shape=self._packed_shape(intermediate_size, hidden_size),
                        )
                        for projection in self.runtime_gate_up_order
                    ],
                    dim=0,
                )
            )
            gate_up_scales.append(
                torch.cat(
                    [
                        self._load_scales(
                            expert_map,
                            layer,
                            expert,
                            projection,
                            expected_shape=(
                                intermediate_size,
                                hidden_size // self.expert_block_size,
                            ),
                        )
                        for projection in self.runtime_gate_up_order
                    ],
                    dim=0,
                )
            )
            down_blocks.append(
                self._load_weight_blocks(
                    expert_map,
                    layer,
                    expert,
                    self.runtime_down_projection,
                    expected_shape=(hidden_size, intermediate_size // 2),
                    packed_shape=self._packed_shape(hidden_size, intermediate_size),
                )
            )
            down_scales.append(
                self._load_scales(
                    expert_map,
                    layer,
                    expert,
                    self.runtime_down_projection,
                    expected_shape=(hidden_size, intermediate_size // self.expert_block_size),
                )
            )

        return PackedMxfp4Experts(
            gate_up_blocks=torch.stack(gate_up_blocks, dim=0),
            gate_up_scales=torch.stack(gate_up_scales, dim=0),
            down_blocks=torch.stack(down_blocks, dim=0),
            down_scales=torch.stack(down_scales, dim=0),
            expert_indices=expert_order,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )

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
        source_state = _strip_prefix_from_state_dict(state_dict, prefix)
        packed = self.pack_experts(
            source_state,
            layer=layer,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            expert_indices=expert_indices,
            num_experts=num_experts,
        )
        for source_key in self._source_keys_for_packed_experts(layer, packed.expert_indices):
            state_dict.pop(prefix + source_key, None)
        state_dict[prefix + target_gate_up_blocks] = packed.gate_up_blocks
        state_dict[prefix + target_gate_up_scales] = packed.gate_up_scales
        state_dict[prefix + target_down_blocks] = packed.down_blocks
        state_dict[prefix + target_down_scales] = packed.down_scales

    def _collect_expert_tensors(
        self,
        state_dict: Mapping[str, torch.Tensor],
        layer: int,
    ) -> dict[int, dict[str, dict[str, torch.Tensor]]]:
        expert_tensors: dict[int, dict[str, dict[str, torch.Tensor]]] = {}
        for name, tensor in state_dict.items():
            parsed = self.parse_key(name)
            if parsed is None or parsed.layer != layer:
                continue
            if not isinstance(tensor, torch.Tensor):
                raise QuantizedCheckpointLayoutError(f"{name} should be a torch.Tensor.")
            weight_tensors = expert_tensors.setdefault(parsed.expert, {})
            tensor_kinds = weight_tensors.setdefault(parsed.projection, {})
            tensor_kinds[parsed.tensor_kind] = tensor
        return expert_tensors

    def _get_required_expert_map(
        self,
        expert_tensors: Mapping[int, dict[str, dict[str, torch.Tensor]]],
        layer: int,
        expert: int,
    ) -> dict[str, dict[str, torch.Tensor]]:
        expert_map = expert_tensors.get(expert)
        if expert_map is None:
            first_missing = self.format_key(layer, expert, self.runtime_gate_up_order[0], "weight")
            raise QuantizedCheckpointLayoutError(f"Missing packed MXFP4 tensor {first_missing}.")
        return expert_map

    def _load_weight_blocks(
        self,
        expert_map: Mapping[str, Mapping[str, torch.Tensor]],
        layer: int,
        expert: int,
        projection: str,
        *,
        expected_shape: tuple[int, int],
        packed_shape: tuple[int, int, int],
    ) -> torch.Tensor:
        name = self.format_key(layer, expert, projection, "weight")
        tensor = self._get_required_tensor(expert_map, layer, expert, projection, "weight")
        _validate_tensor_shape(name, tensor, expected_shape)
        return self._reinterpret_weight_as_uint8(name, tensor).view(packed_shape)

    def _load_scales(
        self,
        expert_map: Mapping[str, Mapping[str, torch.Tensor]],
        layer: int,
        expert: int,
        projection: str,
        *,
        expected_shape: tuple[int, int],
    ) -> torch.Tensor:
        name = self.format_key(layer, expert, projection, "scale")
        tensor = self._get_required_tensor(expert_map, layer, expert, projection, "scale")
        _validate_tensor_shape(name, tensor, expected_shape)
        return self._reinterpret_scale_as_uint8(name, tensor).view(expected_shape)

    def _get_required_tensor(
        self,
        expert_map: Mapping[str, Mapping[str, torch.Tensor]],
        layer: int,
        expert: int,
        projection: str,
        tensor_kind: str,
    ) -> torch.Tensor:
        tensor = expert_map.get(projection, {}).get(tensor_kind)
        if tensor is None:
            name = self.format_key(layer, expert, projection, tensor_kind)
            raise QuantizedCheckpointLayoutError(f"Missing packed MXFP4 tensor {name}.")
        return tensor

    def _packed_shape(self, rows: int, logical_cols: int) -> tuple[int, int, int]:
        return (rows, logical_cols // self.expert_block_size, self.expert_block_size // 2)

    def _required_projection_names(self) -> tuple[str, ...]:
        required = tuple(dict.fromkeys((*self.runtime_gate_up_order, self.runtime_down_projection)))
        invalid = [projection for projection in required if projection not in self.projection_names]
        if invalid:
            raise QuantizedCheckpointLayoutError(
                "Packed MXFP4 runtime projections should be declared in projection_names, "
                f"got invalid projection(s): {invalid}."
            )
        return required

    def _source_keys_for_packed_experts(
        self, layer: int, expert_indices: Sequence[int]
    ) -> tuple[str, ...]:
        return tuple(
            self.format_key(layer, expert, projection, tensor_kind)
            for expert in expert_indices
            for projection in self._required_projection_names()
            for tensor_kind in ("weight", "scale")
        )

    def _validate_weight_metadata(self, name: str, metadata: TensorMetadata) -> None:
        dtype = _get_dtype(name, metadata)
        if dtype not in self.weight_dtypes:
            raise QuantizedCheckpointLayoutError(
                f"{name} should be packed FP4 with dtype {self.weight_dtypes}, got {dtype}."
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
        weight_shape = _get_shape(weight_name, weight_metadata)
        _require_2d(weight_name, weight_shape)
        expected = [weight_shape[0], math.ceil(weight_shape[1] * 2 / self.expert_block_size)]
        shape = _get_shape(scale_name, scale_metadata)
        if shape != expected:
            raise QuantizedCheckpointLayoutError(
                f"{scale_name} has shape {shape}, expected {expected} for packed FP4 {weight_name}."
            )

    def _reinterpret_weight_as_uint8(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dtype not in (torch.int8, torch.uint8):
            raise QuantizedCheckpointLayoutError(
                f"{name} should be packed FP4 bytes with dtype torch.int8 or torch.uint8, got {tensor.dtype}."
            )
        return _view_as_uint8(tensor)

    def _reinterpret_scale_as_uint8(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dtype == torch.uint8 or (_E8M0_DTYPE is not None and tensor.dtype == _E8M0_DTYPE):
            return _view_as_uint8(tensor)
        expected = "torch.uint8"
        if _E8M0_DTYPE is not None:
            expected += " or torch.float8_e8m0fnu"
        raise QuantizedCheckpointLayoutError(
            f"{name} should contain raw E8M0 scale bytes with dtype {expected}, got {tensor.dtype}."
        )


@dataclass(frozen=True)
class QuantizedCheckpointLayout:
    """Quantized checkpoint layout selected by a model-owned layout builder."""

    finegrained_fp8: FineGrainedFP8CheckpointLayout | None = None
    packed_mxfp4_experts: PackedMxfp4ExpertsCheckpointLayout | None = None

    def validate_checkpoint_metadata(self, tensor_metadata: CheckpointMetadata) -> None:
        if self.finegrained_fp8 is not None:
            self.finegrained_fp8.validate_checkpoint_metadata(tensor_metadata)
        if self.packed_mxfp4_experts is not None:
            self.packed_mxfp4_experts.validate_checkpoint_metadata(tensor_metadata)

    def validate_consumed_metadata(self, tensor_metadata: CheckpointMetadata) -> None:
        self.validate_checkpoint_metadata(tensor_metadata)

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
        if self.packed_mxfp4_experts is not None:
            normalized["expert_quant_method"] = self.packed_mxfp4_experts.quant_method
            normalized["expert_block_size"] = self.packed_mxfp4_experts.expert_block_size
        return normalized


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


def _strip_prefix_from_state_dict(
    state_dict: Mapping[str, torch.Tensor], prefix: str
) -> dict[str, torch.Tensor]:
    if not prefix:
        return dict(state_dict)
    return {
        key.removeprefix(prefix): tensor
        for key, tensor in state_dict.items()
        if key.startswith(prefix)
    }


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


def _validate_tensor_shape(
    name: str, tensor: torch.Tensor, expected_shape: tuple[int, int]
) -> None:
    actual_shape = tuple(tensor.shape)
    if actual_shape != expected_shape:
        raise QuantizedCheckpointLayoutError(
            f"{name} has shape {list(actual_shape)}, expected {list(expected_shape)}."
        )


def _view_as_uint8(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == torch.uint8 and tensor.is_contiguous():
        return tensor
    return tensor.contiguous().view(torch.uint8)


def _normalize_expert_order(
    expert_tensors: Mapping[int, object],
    *,
    expert_indices: Sequence[int] | None,
    num_experts: int | None,
) -> tuple[int, ...]:
    if num_experts is not None:
        num_experts = _validate_positive_int("num_experts", num_experts)

    if expert_indices is None:
        if num_experts is not None:
            expert_order = tuple(range(num_experts))
        else:
            expert_order = tuple(sorted(expert_tensors))
    else:
        expert_order = tuple(
            _validate_nonnegative_int("expert_indices", idx) for idx in expert_indices
        )

    if not expert_tensors:
        raise QuantizedCheckpointLayoutError("No packed MXFP4 routed expert tensors were found.")
    if not expert_order:
        raise QuantizedCheckpointLayoutError("No packed MXFP4 expert indices were selected.")
    if len(set(expert_order)) != len(expert_order):
        raise QuantizedCheckpointLayoutError(
            f"expert_indices should not contain duplicates, got {expert_order}."
        )
    if num_experts is not None:
        invalid_indices = [expert for expert in expert_order if expert >= num_experts]
        if invalid_indices:
            raise QuantizedCheckpointLayoutError(
                f"expert_indices should be less than num_experts={num_experts}, got {invalid_indices}."
            )
    return expert_order


def _validate_positive_divisible_by(name: str, value: int, divisor: int) -> int:
    value = _validate_positive_int(name, value)
    if value % divisor != 0:
        raise QuantizedCheckpointLayoutError(
            f"{name} should be divisible by {divisor}, got {value}."
        )
    return value


def _validate_positive_int(name: str, value: int) -> int:
    value = _validate_int(name, value)
    if value <= 0:
        raise QuantizedCheckpointLayoutError(f"{name} should be positive, got {value}.")
    return value


def _validate_nonnegative_int(name: str, value: int) -> int:
    value = _validate_int(name, value)
    if value < 0:
        raise QuantizedCheckpointLayoutError(f"{name} should be non-negative, got {value}.")
    return value


def _validate_int(name: str, value: int) -> int:
    if isinstance(value, bool):
        raise QuantizedCheckpointLayoutError(f"{name} should be an integer, got {value}.")
    try:
        return operator.index(value)
    except TypeError as error:
        raise QuantizedCheckpointLayoutError(
            f"{name} should be an integer, got {type(value).__name__}."
        ) from error
