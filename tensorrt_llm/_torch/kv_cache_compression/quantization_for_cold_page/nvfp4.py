# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""NVFP4 policy and cold-page layout construction."""

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from tensorrt_llm.quantization.modelopt_config import (
    is_modelopt_quant_config,
    read_modelopt_quant_config,
)

from ...pyexecutor.resource_manager import DataType
from .quantization_for_cold_page import ColdPageCodecPolicy, ColdPageQuantizationMethod

ScalePair = tuple[float, float]
LayerScales = tuple[ScalePair, ScalePair]

_IDENTITY_NVFP4_SCALES: LayerScales = ((1.0, 1.0), (1.0, 1.0))
_MODEL_OPT_LANGUAGE_KV_SCALE_KEY = re.compile(
    r"^model(?:\.language_model)?\.layers\.(?P<layer_id>\d+)\.self_attn\."
    r"(?P<kind>[kv])_proj\.(?P=kind)_scale$"
)
_COLD_PAGE_ALIGNMENT = 16
_ELEMENTS_PER_BYTE = 2
_ELEMENTS_PER_SCALE = 16
_ELEMENTS_PER_HALF_GROUP = 8
_MAX_HALF_GROUPS_PER_TILE = 2048
_MAX_BUFFERS_PER_LAUNCH = 256
_WIDE_FIELDS = 6
_INTEGER_FIELDS = 5
_SCALE_FIELDS = 4
_NVFP4_TRANSFORM = 0
_LOSSLESS_TRANSFORM = 1


@dataclass(frozen=True)
class _Nvfp4Scales:
    nvfp4_orig_quant: float
    nvfp4_quant_orig: float
    fp8_orig_quant: float = 1.0
    fp8_quant_orig: float = 1.0


@dataclass(frozen=True)
class _Nvfp4BufferLayout:
    role: str
    scales: _Nvfp4Scales | None = None


@dataclass(frozen=True)
class _Nvfp4LayerLayout:
    layer_id: int
    num_kv_heads: int
    tokens_per_page: int
    head_dim: int
    buffers: tuple[_Nvfp4BufferLayout, ...]


@dataclass(frozen=True)
class _Nvfp4ColdPageMetadata:
    """Python-owned launch metadata for one KVCM lifecycle."""

    wide: torch.Tensor
    integers: torch.Tensor
    scales: torch.Tensor
    num_buffers: int
    max_half_groups_per_tile: int
    cold_page_bytes: int
    runtime_type: int


def _load_modelopt_nvfp4_scales(
    checkpoint_path: str | None,
) -> dict[int, LayerScales]:
    """Load optional ModelOpt NVFP4 K/V global scales by model layer."""

    if checkpoint_path is None or os.environ.get("TRTLLM_LOAD_KV_SCALES", "1") != "1":
        return {}

    checkpoint_dir = Path(checkpoint_path)
    weight_files = sorted(checkpoint_dir.glob("*.safetensors"))
    ordinary_files = [path for path in weight_files if "consolidated" not in path.name]
    weight_files = ordinary_files or weight_files
    if not weight_files:
        raise FileNotFoundError(
            f"No safetensors files in ModelOpt scale checkpoint {checkpoint_dir}"
        )

    metadata_path = checkpoint_dir / "hf_quant_config.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
    else:
        config_path = checkpoint_dir / "config.json"
        metadata = (
            json.loads(config_path.read_text()).get("quantization_config")
            if config_path.exists()
            else None
        )
    if not is_modelopt_quant_config(metadata):
        return {}
    if read_modelopt_quant_config(metadata).get("kv_cache_quant_algo") != "NVFP4":
        return {}

    from safetensors import safe_open

    values: dict[int, dict[str, list[float]]] = {}
    for file_path in weight_files:
        with safe_open(str(file_path), framework="pt", device="cpu") as checkpoint:
            for tensor_name in checkpoint.keys():
                match = _MODEL_OPT_LANGUAGE_KV_SCALE_KEY.fullmatch(tensor_name)
                if match is None:
                    continue
                value = float(checkpoint.get_tensor(tensor_name).reshape([]).item())
                if not math.isfinite(value) or value <= 0.0:
                    raise ValueError(
                        f"ModelOpt KV scale {file_path}:{tensor_name} must be finite and positive"
                    )
                layer_values = values.setdefault(int(match.group("layer_id")), {"k": [], "v": []})
                layer_values[match.group("kind")].append(value)

    result: dict[int, LayerScales] = {}
    for layer_id, layer_values in values.items():
        k_values, v_values = layer_values["k"], layer_values["v"]
        if not k_values or not v_values:
            raise ValueError(f"ModelOpt KV scales for layer {layer_id} must contain both K and V")
        quant_orig = (max(k_values), max(v_values))
        orig_quant = (1.0 / quant_orig[0], 1.0 / quant_orig[1])
        stored_scales = torch.tensor(
            (*orig_quant, *quant_orig), dtype=torch.float32, device="cpu"
        ).tolist()
        if any(not math.isfinite(value) or value <= 0.0 for value in stored_scales):
            raise ValueError(
                f"ModelOpt KV scales for layer {layer_id} are not representable as float32"
            )
        result[layer_id] = (
            (stored_scales[0], stored_scales[1]),
            (stored_scales[2], stored_scales[3]),
        )
    return result


def _align_up(value: int, alignment: int = _COLD_PAGE_ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


class Nvfp4ColdPagePolicy(ColdPageCodecPolicy):
    """Resolve NVFP4 metadata and submit one CUDA op per codec batch."""

    def __init__(self, layer_layouts: Sequence[_Nvfp4LayerLayout], runtime_type: int) -> None:
        self._layer_layouts = {layout.layer_id: layout for layout in layer_layouts}
        self._runtime_type = runtime_type
        self._lifecycle_metadata: list[_Nvfp4ColdPageMetadata] = []

    @property
    def layer_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._layer_layouts))

    def configure(self, lifecycles: Sequence[object]) -> Sequence[object]:
        """Resolve hot buffers into immutable Python-owned launch metadata."""

        from tensorrt_llm.bindings.internal import kv_cache_compression as native

        lifecycle_metadata = []
        properties = []
        for lifecycle in lifecycles:
            wide_rows: list[list[int]] = []
            integer_rows: list[list[int]] = []
            scale_rows: list[list[float]] = []
            cold_page_bytes = 0
            max_half_groups_per_tile = 0

            for layer_id, hot_buffers in lifecycle.layers.items():
                layout = self._layer_layouts[int(layer_id)]
                expected_roles = {buffer.role for buffer in layout.buffers}
                if set(hot_buffers) != expected_roles:
                    raise ValueError(
                        f"Cold-page layer {layer_id} roles do not match its KVCM layout"
                    )
                compressed = [buffer for buffer in layout.buffers if buffer.scales]
                packed_bytes = (
                    layout.num_kv_heads * layout.tokens_per_page * layout.head_dim
                ) // _ELEMENTS_PER_BYTE
                scale_bytes = (
                    layout.num_kv_heads * layout.tokens_per_page * layout.head_dim
                ) // _ELEMENTS_PER_SCALE
                layer_start = cold_page_bytes
                scale_start = layer_start + len(compressed) * packed_bytes
                cursor = scale_start + len(compressed) * scale_bytes

                compressed_index = 0
                for buffer in layout.buffers:
                    hot = hot_buffers[buffer.role]
                    raw_base = int(hot.raw_base)
                    raw_slot_bytes = int(hot.raw_slot_bytes)
                    raw_bytes = int(hot.raw_bytes)
                    if raw_base <= 0 or raw_bytes <= 0 or raw_bytes > raw_slot_bytes:
                        raise ValueError("Cold-page hot buffer has invalid address or size")

                    if buffer.scales:
                        data_offset = layer_start + compressed_index * packed_bytes
                        scale_offset = scale_start + compressed_index * scale_bytes
                        compressed_index += 1
                        expected_raw_bytes = (
                            layout.num_kv_heads
                            * layout.tokens_per_page
                            * layout.head_dim
                            * (1 if self._runtime_type == 2 else 2)
                        )
                        if raw_bytes != expected_raw_bytes:
                            raise ValueError("Hot buffer size does not match NVFP4 geometry")
                        if raw_base % 16 or raw_slot_bytes % 16:
                            raise ValueError(
                                "NVFP4 hot address and Slot stride must be 16-byte aligned"
                            )
                        half_groups = (
                            expected_raw_bytes
                            // (1 if self._runtime_type == 2 else 2)
                            // _ELEMENTS_PER_HALF_GROUP
                        )
                        max_half_groups_per_tile = max(
                            max_half_groups_per_tile,
                            min(half_groups, _MAX_HALF_GROUPS_PER_TILE),
                        )
                    else:
                        data_offset = cursor
                        scale_offset = 0
                        cursor += raw_bytes

                    wide_rows.append(
                        [
                            raw_base,
                            raw_slot_bytes,
                            raw_bytes,
                            data_offset,
                            scale_offset,
                            0,
                        ]
                    )
                    integer_rows.append(
                        [
                            0,
                            _NVFP4_TRANSFORM if buffer.scales else _LOSSLESS_TRANSFORM,
                            layout.num_kv_heads if buffer.scales else 0,
                            layout.tokens_per_page if buffer.scales else 0,
                            layout.head_dim if buffer.scales else 0,
                        ]
                    )
                    buffer_scales = buffer.scales or _Nvfp4Scales(1.0, 1.0)
                    scale_rows.append(
                        [
                            buffer_scales.nvfp4_orig_quant,
                            buffer_scales.nvfp4_quant_orig,
                            buffer_scales.fp8_orig_quant,
                            buffer_scales.fp8_quant_orig,
                        ]
                    )
                layer_end = _align_up(cursor)
                wide_rows[-1][5] = cursor
                integer_rows[-1][0] = layer_end - cursor
                cold_page_bytes = layer_end

            num_buffers = len(wide_rows)
            if not 0 < num_buffers <= _MAX_BUFFERS_PER_LAUNCH:
                raise ValueError(
                    f"NVFP4 cold-page lifecycle has {num_buffers} buffers; "
                    f"the maximum is {_MAX_BUFFERS_PER_LAUNCH}"
                )
            padding = _MAX_BUFFERS_PER_LAUNCH - num_buffers
            lifecycle_metadata.append(
                _Nvfp4ColdPageMetadata(
                    wide=torch.tensor(
                        wide_rows + [[0] * _WIDE_FIELDS for _ in range(padding)],
                        dtype=torch.int64,
                        device="cpu",
                    ),
                    integers=torch.tensor(
                        integer_rows + [[0] * _INTEGER_FIELDS for _ in range(padding)],
                        dtype=torch.int32,
                        device="cpu",
                    ),
                    scales=torch.tensor(
                        scale_rows + [[0.0] * _SCALE_FIELDS for _ in range(padding)],
                        dtype=torch.float32,
                        device="cpu",
                    ),
                    num_buffers=num_buffers,
                    max_half_groups_per_tile=max_half_groups_per_tile,
                    cold_page_bytes=cold_page_bytes,
                    runtime_type=self._runtime_type,
                )
            )
            lifecycle_properties = native.ColdPageLifecycleProperties()
            lifecycle_properties.cold_page_bytes = cold_page_bytes
            lifecycle_properties.page_index_location = native.ColdPageIndexLocation.HOST
            properties.append(lifecycle_properties)

        self._lifecycle_metadata = lifecycle_metadata
        return properties

    def encode(
        self,
        lifecycle_index: int,
        cold_base: int,
        page_indices: int,
        num_pages: int,
        stream: int,
    ) -> None:
        metadata = self._lifecycle_metadata[lifecycle_index]
        torch.ops.trtllm.nvfp4_cold_page_encode(
            metadata.wide,
            metadata.integers,
            metadata.scales,
            metadata.num_buffers,
            metadata.max_half_groups_per_tile,
            metadata.cold_page_bytes,
            metadata.runtime_type,
            cold_base,
            page_indices,
            num_pages,
            stream,
        )

    def decode(
        self,
        lifecycle_index: int,
        cold_base: int,
        page_indices: int,
        num_pages: int,
        stream: int,
    ) -> None:
        metadata = self._lifecycle_metadata[lifecycle_index]
        torch.ops.trtllm.nvfp4_cold_page_decode(
            metadata.wide,
            metadata.integers,
            metadata.scales,
            metadata.num_buffers,
            metadata.max_half_groups_per_tile,
            metadata.cold_page_bytes,
            metadata.runtime_type,
            cold_base,
            page_indices,
            num_pages,
            stream,
        )


class Nvfp4ColdPageQuantization(ColdPageQuantizationMethod):
    """Build one fresh NVFP4 callback policy for each KVCM construction."""

    def __init__(self, checkpoint_path: str | None) -> None:
        self._model_scales = _load_modelopt_nvfp4_scales(checkpoint_path)

    def create_cold_page_codec(
        self,
        cache_config: object,
        *,
        runtime_dtype: DataType,
        pp_layers: Sequence[int],
        num_kv_heads_per_layer: Sequence[int],
        head_dim_per_layer: Sequence[int],
        is_draft: bool = False,
    ) -> object:
        from tensorrt_llm.bindings.internal import kv_cache_compression as native
        from tensorrt_llm.runtime.kv_cache_manager_v2 import AttentionLayerConfig

        runtime_type = {
            DataType.HALF: 0,
            DataType.BF16: 1,
            DataType.FP8: 2,
        }.get(runtime_dtype)
        attention_layers = [
            layer for layer in cache_config.layers if isinstance(layer, AttentionLayerConfig)
        ]
        if attention_layers and runtime_type is None:
            raise RuntimeError(
                "NVFP4 cold-page compression supports FP16, BF16, or FP8 "
                f"Attention KV, not {runtime_dtype}"
            )

        layer_layouts = []
        for layer in attention_layers:
            layer_id = int(layer.layer_id)
            buffers_by_role = {str(buffer.role): buffer for buffer in layer.buffers}
            if "key" not in buffers_by_role:
                raise NotImplementedError(
                    "NVFP4 cold-page compression requires an Attention key buffer"
                )

            compressed_roles = ("key", "value") if "value" in buffers_by_role else ("key",)
            if len(compressed_roles) == 2 and not is_draft:
                orig_quant, quant_orig = self._model_scales.get(
                    int(pp_layers[layer_id]), _IDENTITY_NVFP4_SCALES
                )
            else:
                orig_quant, quant_orig = _IDENTITY_NVFP4_SCALES

            num_kv_heads = int(num_kv_heads_per_layer[layer_id])
            tokens_per_page = int(cache_config.tokens_per_block)
            head_dim = int(head_dim_per_layer[layer_id])
            if head_dim <= 0 or head_dim % _ELEMENTS_PER_SCALE != 0:
                raise ValueError(
                    f"NVFP4 cold pages require head_dim divisible by 16, got {head_dim}"
                )
            buffer_layouts = [
                _Nvfp4BufferLayout(
                    role=role,
                    scales=_Nvfp4Scales(orig_quant[index], quant_orig[index]),
                )
                for index, role in enumerate(compressed_roles)
            ]
            for buffer in layer.buffers:
                role = str(buffer.role)
                if role not in compressed_roles:
                    buffer_layouts.append(_Nvfp4BufferLayout(role=role))

            layer_layouts.append(
                _Nvfp4LayerLayout(
                    layer_id=layer_id,
                    num_kv_heads=num_kv_heads,
                    tokens_per_page=tokens_per_page,
                    head_dim=head_dim,
                    buffers=tuple(buffer_layouts),
                )
            )

        policy = Nvfp4ColdPagePolicy(layer_layouts, runtime_type or 0)
        return native.create_python_cold_page_codec(policy)
