# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""NVFP4 compression for KVCM V2 cold pages.

The compression manager owns optional ModelOpt K/V global scales and creates
the native storage-boundary codec before KVCM allocates cold Slots. Attention,
the active KV-cache format, and the normal model-loading path remain unaware of
the cold representation.
"""

import json
import math
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from tensorrt_llm.quantization.modelopt_config import (
    is_modelopt_quant_config,
    read_modelopt_quant_config,
)

from ...pyexecutor.resource_manager import DataType, KVCacheCompressionManager

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import ColdPageQuantizationCompressionConfig

ScalePair = tuple[float, float]
LayerScales = tuple[ScalePair, ScalePair]

_IDENTITY_NVFP4_SCALES: LayerScales = ((1.0, 1.0), (1.0, 1.0))
_MODEL_OPT_KV_SCALE_KEY = re.compile(
    r"(?:^|\.)layers\.(?P<layer_id>\d+)\.self_attn\."
    r"(?P<kind>[kv])_proj\.(?P=kind)_scale$"
)


def _load_modelopt_nvfp4_scales(
    checkpoint_path: str | None,
) -> dict[int, LayerScales]:
    """Load optional ModelOpt NVFP4 K/V global scales by model layer.

    This mirrors the native NVFP4 KV loader's contract: no checkpoint means
    identity scales, ``TRTLLM_LOAD_KV_SCALES`` can disable loading, and K/V
    scalars from standard safetensors shards are reduced with ``max``.
    """

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
                match = _MODEL_OPT_KV_SCALE_KEY.search(tensor_name)
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


class ColdPageQuantizationCompression(KVCacheCompressionManager):
    """NVFP4 cold-page manager and owner of its optional model scales."""

    uses_iteration_lifecycle = False
    provides_cold_page_codec = True

    def __init__(self, config: "ColdPageQuantizationCompressionConfig") -> None:
        super().__init__(config)
        self._model_nvfp4_scales = _load_modelopt_nvfp4_scales(config.scale_checkpoint_path)

    def create_cold_page_codec(
        self,
        cache_config: object,
        *,
        runtime_dtype: DataType,
        pp_layers: Sequence[int],
        num_kv_heads_per_layer: Sequence[int],
        head_dim_per_layer: Sequence[int],
    ) -> object:
        """Create the native codec that KVCM consumes exactly once."""

        from tensorrt_llm.bindings.internal import kv_cache_compression as native
        from tensorrt_llm.runtime.kv_cache_manager_v2 import SsmLayerConfig

        attention_layers = []
        for layer in cache_config.layers:
            if isinstance(layer, SsmLayerConfig):
                continue
            roles = {buffer.role for buffer in layer.buffers}
            if "key" not in roles:
                raise NotImplementedError(
                    "NVFP4 cold-page compression requires an Attention key buffer"
                )
            has_value = "value" in roles
            attention_layers.append((layer, has_value))
        if not attention_layers:
            return native.create_nvfp4_cold_page_codec([])

        runtime_type = {
            DataType.HALF: native.Nvfp4BoundaryRuntimeType.FLOAT16,
            DataType.BF16: native.Nvfp4BoundaryRuntimeType.BFLOAT16,
            DataType.FP8: native.Nvfp4BoundaryRuntimeType.FP8_E4M3,
        }.get(runtime_dtype)
        if runtime_type is None:
            raise RuntimeError(
                "NVFP4 cold-page compression supports FP16, BF16, or FP8 "
                f"Attention KV, not {runtime_dtype}"
            )

        native_configs = []
        for layer, has_value in attention_layers:
            layer_id = int(layer.layer_id)
            if has_value:
                orig_quant, quant_orig = self._model_nvfp4_scales.get(
                    int(pp_layers[layer_id]), _IDENTITY_NVFP4_SCALES
                )
            else:
                # ModelOpt K/V projection scales do not apply to MLA latent buffers.
                orig_quant, quant_orig = _IDENTITY_NVFP4_SCALES
            native_config = native.Nvfp4ColdPageLayerConfig()
            native_config.layer_id = layer_id
            native_config.runtime_type = runtime_type
            native_config.num_kv_heads = int(num_kv_heads_per_layer[layer_id])
            native_config.tokens_per_page = int(cache_config.tokens_per_block)
            native_config.head_dim = int(head_dim_per_layer[layer_id])
            native_config.nvfp4_scale_orig_quant = orig_quant
            native_config.nvfp4_scale_quant_orig = quant_orig
            native_configs.append(native_config)

        # The C++ factory is required for unique_ptr ownership transfer into KVCM.
        return native.create_nvfp4_cold_page_codec(native_configs)
