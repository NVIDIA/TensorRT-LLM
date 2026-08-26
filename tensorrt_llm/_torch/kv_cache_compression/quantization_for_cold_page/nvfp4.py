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
    data_offset: int
    scale_offset: int = 0
    scales: _Nvfp4Scales | None = None


@dataclass(frozen=True)
class _Nvfp4LayerLayout:
    layer_id: int
    runtime_type: int
    num_kv_heads: int
    tokens_per_page: int
    head_dim: int
    cold_page_bytes: int
    padding_offset: int
    buffers: tuple[_Nvfp4BufferLayout, ...]


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
        result[layer_id] = (
            (1.0 / quant_orig[0], 1.0 / quant_orig[1]),
            quant_orig,
        )
    return result


def _align_up(value: int, alignment: int = _COLD_PAGE_ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def _buffer_bytes(buffer: object, tokens_per_page: int) -> int:
    buffer_tokens = buffer.tokens_per_block_override or tokens_per_page
    return int(buffer.size) * (tokens_per_page // buffer_tokens)


class Nvfp4ColdPagePolicy(ColdPageCodecPolicy):
    """Own NVFP4 lifecycle programs and dispatch one operation per codec batch."""

    def __init__(self, layer_layouts: Sequence[_Nvfp4LayerLayout]) -> None:
        self._layer_layouts = {layout.layer_id: layout for layout in layer_layouts}
        self._programs: list[object] = []

    @property
    def layer_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._layer_layouts))

    def configure(self, lifecycles: Sequence[object]) -> Sequence[object]:
        """Resolve Python layouts against hot buffers and prepare method programs."""

        from tensorrt_llm.bindings.internal import kv_cache_compression as native

        programs = []
        properties = []
        for lifecycle in lifecycles:
            metadata: list[list[int]] = []
            scales: list[list[float]] = []
            cold_page_bytes = 0
            runtime_type = None

            for layer_id, hot_buffers in lifecycle.layers.items():
                layout = self._layer_layouts[int(layer_id)]
                if runtime_type is not None and runtime_type != layout.runtime_type:
                    raise ValueError("One cold-page lifecycle must use one runtime dtype")
                runtime_type = layout.runtime_type

                for index, buffer in enumerate(layout.buffers):
                    hot = hot_buffers[buffer.role]
                    padding_offset = 0
                    padding_bytes = 0
                    if index + 1 == len(layout.buffers):
                        padding_offset = cold_page_bytes + layout.padding_offset
                        padding_bytes = layout.cold_page_bytes - layout.padding_offset
                    metadata.append(
                        [
                            int(hot.raw_base),
                            int(hot.raw_slot_bytes),
                            int(hot.raw_bytes),
                            cold_page_bytes + buffer.data_offset,
                            cold_page_bytes + buffer.scale_offset if buffer.scales else 0,
                            padding_offset,
                            padding_bytes,
                            _NVFP4_TRANSFORM if buffer.scales else _LOSSLESS_TRANSFORM,
                            layout.num_kv_heads if buffer.scales else 0,
                            layout.tokens_per_page if buffer.scales else 0,
                            layout.head_dim if buffer.scales else 0,
                        ]
                    )
                    buffer_scales = buffer.scales or _Nvfp4Scales(1.0, 1.0)
                    scales.append(
                        [
                            buffer_scales.nvfp4_orig_quant,
                            buffer_scales.nvfp4_quant_orig,
                            buffer_scales.fp8_orig_quant,
                            buffer_scales.fp8_quant_orig,
                        ]
                    )
                cold_page_bytes += layout.cold_page_bytes

            if runtime_type is None:
                raise ValueError("NVFP4 received an empty cold-page lifecycle")
            programs.append(
                torch.ops.trtllm.prepare_nvfp4_cold_page_program(
                    metadata, scales, cold_page_bytes, runtime_type
                )
            )
            lifecycle_properties = native.ColdPageLifecycleProperties()
            lifecycle_properties.cold_page_bytes = cold_page_bytes
            lifecycle_properties.page_index_location = native.ColdPageIndexLocation.HOST
            properties.append(lifecycle_properties)

        self._programs = programs
        return properties

    def encode(
        self,
        program_index: int,
        cold_base: int,
        page_indices: int,
        num_pages: int,
        stream: int,
    ) -> None:
        torch.ops.trtllm.nvfp4_cold_page_encode(
            self._programs[program_index], cold_base, page_indices, num_pages, stream
        )

    def decode(
        self,
        program_index: int,
        cold_base: int,
        page_indices: int,
        num_pages: int,
        stream: int,
    ) -> None:
        torch.ops.trtllm.nvfp4_cold_page_decode(
            self._programs[program_index], cold_base, page_indices, num_pages, stream
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
            elements = num_kv_heads * tokens_per_page * head_dim
            packed_bytes = elements // _ELEMENTS_PER_BYTE
            scale_bytes = elements // _ELEMENTS_PER_SCALE
            data_offsets = {
                role: index * packed_bytes for index, role in enumerate(compressed_roles)
            }
            scale_base = len(compressed_roles) * packed_bytes
            scale_offsets = {
                role: scale_base + index * scale_bytes
                for index, role in enumerate(compressed_roles)
            }
            cursor = scale_base + len(compressed_roles) * scale_bytes

            buffer_layouts = [
                _Nvfp4BufferLayout(
                    role=role,
                    data_offset=data_offsets[role],
                    scale_offset=scale_offsets[role],
                    scales=_Nvfp4Scales(orig_quant[index], quant_orig[index]),
                )
                for index, role in enumerate(compressed_roles)
            ]
            for buffer in layer.buffers:
                role = str(buffer.role)
                if role not in compressed_roles:
                    buffer_layouts.append(_Nvfp4BufferLayout(role=role, data_offset=cursor))
                    cursor += _buffer_bytes(buffer, tokens_per_page)

            layer_layouts.append(
                _Nvfp4LayerLayout(
                    layer_id=layer_id,
                    runtime_type=runtime_type,
                    num_kv_heads=num_kv_heads,
                    tokens_per_page=tokens_per_page,
                    head_dim=head_dim,
                    cold_page_bytes=_align_up(cursor),
                    padding_offset=cursor,
                    buffers=tuple(buffer_layouts),
                )
            )

        policy = Nvfp4ColdPagePolicy(layer_layouts)
        return native.create_python_cold_page_codec(policy.layer_ids, policy)
