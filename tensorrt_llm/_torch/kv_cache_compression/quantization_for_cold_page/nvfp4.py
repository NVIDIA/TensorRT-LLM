# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""NVFP4 policy and cold-page layout construction."""

import json
import math
import os
import re
from pathlib import Path
from typing import Sequence

from tensorrt_llm.quantization.modelopt_config import (
    is_modelopt_quant_config,
    read_modelopt_quant_config,
)

from ...pyexecutor.resource_manager import DataType

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
    if buffer_tokens <= 0 or tokens_per_page % buffer_tokens != 0:
        raise ValueError("tokens_per_block_override must be a positive divisor of tokens_per_block")
    return int(buffer.size) * (tokens_per_page // buffer_tokens)


class Nvfp4ColdPagePolicy:
    """Build native NVFP4 plans without retaining Python in the data path."""

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
        """Create the native generic codec from explicit per-buffer plans."""

        from tensorrt_llm.bindings.internal import kv_cache_compression as native
        from tensorrt_llm.runtime.kv_cache_manager_v2 import AttentionLayerConfig

        attention_layers = [
            layer for layer in cache_config.layers if isinstance(layer, AttentionLayerConfig)
        ]
        if not attention_layers:
            return native.create_nvfp4_cold_page_codec([])

        runtime_type = {
            DataType.HALF: native.Nvfp4ColdPageRuntimeType.FLOAT16,
            DataType.BF16: native.Nvfp4ColdPageRuntimeType.BFLOAT16,
            DataType.FP8: native.Nvfp4ColdPageRuntimeType.FP8_E4M3,
        }.get(runtime_dtype)
        if runtime_type is None:
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
                # Target projection scales describe neither MLA latents nor a
                # separately numbered draft model.
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

            buffer_layouts = []
            for scale_index, role in enumerate(compressed_roles):
                scales = native.Nvfp4ColdPageScales()
                scales.nvfp4_scale_orig_quant = orig_quant[scale_index]
                scales.nvfp4_scale_quant_orig = quant_orig[scale_index]
                scales.fp8_scale_orig_quant = 1.0
                scales.fp8_scale_quant_orig = 1.0

                buffer_layout = native.Nvfp4ColdPageBufferLayout()
                buffer_layout.role = role
                buffer_layout.cold_data_offset = data_offsets[role]
                buffer_layout.cold_scale_offset = scale_offsets[role]
                buffer_layout.scales = scales
                buffer_layouts.append(buffer_layout)

            for buffer in layer.buffers:
                role = str(buffer.role)
                if role in compressed_roles:
                    continue
                buffer_layout = native.Nvfp4ColdPageBufferLayout()
                buffer_layout.role = role
                buffer_layout.cold_data_offset = cursor
                buffer_layouts.append(buffer_layout)
                cursor += _buffer_bytes(buffer, tokens_per_page)

            cold_page_bytes = _align_up(cursor)
            layer_layout = native.Nvfp4ColdPageLayerLayout()
            layer_layout.layer_id = layer_id
            layer_layout.runtime_type = runtime_type
            layer_layout.num_kv_heads = num_kv_heads
            layer_layout.tokens_per_page = tokens_per_page
            layer_layout.head_dim = head_dim
            layer_layout.cold_page_bytes = cold_page_bytes
            layer_layout.cold_padding_offset = cursor
            layer_layout.buffers = buffer_layouts
            layer_layouts.append(layer_layout)

        return native.create_nvfp4_cold_page_codec(layer_layouts)
