# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""NVFP4 cold-page layout, scales, metadata, and kernel dispatch."""

import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import torch

from tensorrt_llm.logger import logger
from tensorrt_llm.quantization.modelopt_config import (
    is_modelopt_quant_config,
    read_modelopt_quant_config,
)

from ...attention.backends.interface import RopeParams
from ...pyexecutor.resource_manager import DataType
from .quantization_for_cold_page import ColdPageQuantizationCompression

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from tensorrt_llm.llmapi.llm_args import ColdPageQuantizationCompressionConfig
    from tensorrt_llm.runtime.kv_cache_manager_v2 import AttentionLayerConfig

_ScalePair = tuple[float, float]
_LayerScales = dict[str, _ScalePair]

_IDENTITY_NVFP4_SCALE: _ScalePair = (1.0, 1.0)
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
_INTEGER_FIELDS = 7
_SCALE_FIELDS = 4
_NVFP4_TRANSFORM = 0
_LOSSLESS_TRANSFORM = 1

# Models whose K vectors have a known position-free (NoPE) part and whose accuracy
# with skip_rope_quantization has been checked. Other models ignore the switch.
_SKIP_ROPE_QUANTIZATION_MODEL_TYPES = frozenset(
    {"deepseek_v4", "glm_moe_dsa", "qwen3_5", "qwen3_5_moe", "qwen3_5_text", "qwen3_5_moe_text"}
)

_DEEPSEEK_V4_PREFIX = "deepseek_v4_"
_DEEPSEEK_V4_SWA = f"{_DEEPSEEK_V4_PREFIX}swa"
_DEEPSEEK_V4_COMPRESS = f"{_DEEPSEEK_V4_PREFIX}compress"
_DEEPSEEK_V4_INDEXER_COMPRESS = f"{_DEEPSEEK_V4_PREFIX}indexer_compress"
_DEEPSEEK_V4_CSA_ROLES = frozenset({_DEEPSEEK_V4_COMPRESS, _DEEPSEEK_V4_INDEXER_COMPRESS})
_DEEPSEEK_V4_HCA_ROLES = frozenset({_DEEPSEEK_V4_COMPRESS})
_DEEPSEEK_V4_ROLES = frozenset(
    {
        _DEEPSEEK_V4_SWA,
        _DEEPSEEK_V4_COMPRESS,
        _DEEPSEEK_V4_INDEXER_COMPRESS,
        f"{_DEEPSEEK_V4_PREFIX}compressor_kv",
        f"{_DEEPSEEK_V4_PREFIX}compressor_score",
        f"{_DEEPSEEK_V4_PREFIX}indexer_compressor_kv",
        f"{_DEEPSEEK_V4_PREFIX}indexer_compressor_score",
    }
)
_DEEPSEEK_V4_NOPE_DIM = 448
_DEEPSEEK_V4_ROW_STRIDE = 512
_DEEPSEEK_V4_FOOTER_SCALE_ROW_BYTES = 584


@dataclass(frozen=True)
class _Nvfp4Scales:
    nvfp4_orig_quant: float
    nvfp4_quant_orig: float
    fp8_orig_quant: float = 1.0
    fp8_quant_orig: float = 1.0


@dataclass(frozen=True)
class _Nvfp4BufferLayout:
    """One hot buffer; ``quantized_range_*`` picks which numbers of each K/V vector become NVFP4."""

    role: str
    scales: _Nvfp4Scales | None = None
    quantized_range_start: int = 0
    quantized_range_elements: int = 0


@dataclass(frozen=True)
class _Nvfp4LayerLayout:
    layer_id: int
    num_kv_heads: int
    tokens_per_page: int
    raw_row_stride_elements: int
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


@dataclass
class _Nvfp4ColdPageCodecState:
    """NVFP4 state owned by one target, draft, or retry codec."""

    layer_layouts: dict[int, _Nvfp4LayerLayout]
    layer_ids: tuple[int, ...]
    runtime_type: int
    lifecycle_metadata: tuple[_Nvfp4ColdPageMetadata, ...] = field(init=False)


def _load_modelopt_nvfp4_scales(
    checkpoint_path: str | None,
) -> dict[int, _LayerScales]:
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

    result: dict[int, _LayerScales] = {}
    for layer_id, layer_values in values.items():
        stored: _LayerScales = {}
        for role in ("k", "v"):
            if not layer_values[role]:
                continue
            quant_orig = max(layer_values[role])
            stored_scales = torch.tensor(
                (1.0 / quant_orig, quant_orig), dtype=torch.float32, device="cpu"
            ).tolist()
            if any(not math.isfinite(value) or value <= 0.0 for value in stored_scales):
                raise ValueError(
                    f"ModelOpt {role.upper()} scale for layer {layer_id} "
                    "is not representable as float32"
                )
            stored[role] = (stored_scales[0], stored_scales[1])
        result[layer_id] = stored
    return result


class Nvfp4ColdPageQuantizationCompression(ColdPageQuantizationCompression):
    """NVFP4 layout, calibration metadata, and CUDA dispatch."""

    def __init__(
        self,
        config: "ColdPageQuantizationCompressionConfig",
        *,
        pretrained_config: "PretrainedConfig",
    ) -> None:
        super().__init__(config, pretrained_config=pretrained_config)
        self._model_scales = _load_modelopt_nvfp4_scales(config.scale_checkpoint_path)
        self._skip_rope_quantization = bool(config.skip_rope_quantization)
        model_type = getattr(pretrained_config, "model_type", None)
        if self._skip_rope_quantization and model_type not in _SKIP_ROPE_QUANTIZATION_MODEL_TYPES:
            logger.warning(
                "skip_rope_quantization is validated for model types "
                f"{sorted(_SKIP_ROPE_QUANTIZATION_MODEL_TYPES)} only; ignoring it for "
                f"{model_type!r} and turning whole K and V vectors into NVFP4."
            )
            self._skip_rope_quantization = False

    def _calculate_quantized_range(
        self,
        row_elements: int,
        *,
        skip_rope: bool,
        rope: tuple[int, int] | None = None,
        key_only: bool = False,
        buffer_name: str,
    ) -> tuple[int, int]:
        """Return (start, count) of the numbers of each K or V vector that become NVFP4."""

        if not skip_rope:
            return 0, row_elements
        if rope is None:
            # The same RoPE width the attention layers use.
            text_config = self.pretrained_config.get_text_config()
            rope_dim = RopeParams.from_config(text_config).dim
            if key_only:  # MLA latent vector: kv_lora_rank NoPE numbers, then the RoPE numbers.
                kv_lora_rank = getattr(text_config, "kv_lora_rank", None)
                if not isinstance(kv_lora_rank, int) or kv_lora_rank + rope_dim != row_elements:
                    raise NotImplementedError(
                        f"{buffer_name}: head_dim {row_elements} is not kv_lora_rank + "
                        f"qk_rope_head_dim ({kv_lora_rank} + {rope_dim}) of an MLA latent vector, "
                        "so its RoPE part cannot be located; skip_rope_quantization is unsupported here"
                    )
                rope = (kv_lora_rank, rope_dim)
            else:  # GQA head: the leading numbers carry RoPE.
                rope = (0, rope_dim)
        rope_start, rope_elements = rope
        if rope_start < 0 or rope_elements < 0 or rope_start + rope_elements > row_elements:
            raise ValueError(
                f"{buffer_name}: RoPE range [{rope_start}, {rope_start + rope_elements}) lies "
                f"outside the {row_elements}-element row; check partial_rotary_factor and "
                "qk_rope_head_dim in the model config"
            )
        if rope_elements == 0:
            return 0, row_elements
        if rope_elements >= row_elements:
            raise ValueError(
                f"{buffer_name}: every element is position-encoded, so skip_rope_quantization would "
                "leave nothing to quantize; the K vectors of this model are entirely RoPE"
            )
        if rope_start == 0:  # RoPE leads the row (partial-rotary GQA heads).
            start, elements = rope_elements, row_elements - rope_elements
        elif rope_start + rope_elements == row_elements:  # RoPE trails the row (MLA, DeepSeek-V4).
            start, elements = 0, rope_start
        else:
            raise ValueError(
                f"{buffer_name}: RoPE elements [{rope_start}, {rope_start + rope_elements}) sit "
                "inside the row; the kernel quantizes one contiguous range per row"
            )
        if start % _ELEMENTS_PER_SCALE or elements % _ELEMENTS_PER_SCALE:
            raise ValueError(
                f"{buffer_name}: the quantized range [{start}, {start + elements}) must start and "
                f"end on {_ELEMENTS_PER_SCALE}-element NVFP4 scale groups"
            )
        return start, elements

    def _build_deepseek_v4_layer_layouts(
        self,
        cache_config: object,
        *,
        attention_layers: Sequence["AttentionLayerConfig"],
        pp_layers: Sequence[int],
        runtime_type: int,
        is_draft: bool,
        skip_rope: bool,
    ) -> dict[int, _Nvfp4LayerLayout | None]:
        """Build layouts for DeepSeek-V4 lifecycles; None selects lossless fallback."""

        roles_by_layer: dict[int, set[str]] = {}
        for layer in attention_layers:
            layer_id = int(layer.layer_id)
            buffer_roles = {str(buffer.role) for buffer in layer.buffers}
            deepseek_v4_roles = {
                role for role in buffer_roles if role.startswith(_DEEPSEEK_V4_PREFIX)
            }
            if deepseek_v4_roles:
                if deepseek_v4_roles != buffer_roles or not buffer_roles <= _DEEPSEEK_V4_ROLES:
                    raise NotImplementedError(
                        f"Unsupported DeepSeek-V4 cold-page roles: {sorted(buffer_roles)}"
                    )
                roles_by_layer[layer_id] = buffer_roles

        has_csa_cache = any(roles == _DEEPSEEK_V4_CSA_ROLES for roles in roles_by_layer.values())
        model_layers: dict[int, int] = {}
        if has_csa_cache:
            model_layer = None
            num_model_layers = 0
            for layer in attention_layers:
                layer_id = int(layer.layer_id)
                roles = roles_by_layer.get(layer_id)
                if roles is None:
                    continue
                if _DEEPSEEK_V4_SWA in roles:
                    if num_model_layers == len(pp_layers):
                        raise ValueError(
                            "DeepSeek-V4 KVCM layout has more model-layer anchors than pp_layers"
                        )
                    model_layer = int(pp_layers[num_model_layers])
                    num_model_layers += 1
                elif model_layer is None:
                    raise ValueError(
                        "DeepSeek-V4 KVCM layout must begin each model layer with an SWA Page"
                    )
                model_layers[layer_id] = model_layer
            if num_model_layers != len(pp_layers):
                raise ValueError(
                    "DeepSeek-V4 KVCM layout and pp_layers have different model-layer counts"
                )

        layouts: dict[int, _Nvfp4LayerLayout | None] = {}
        for layer in attention_layers:
            layer_id = int(layer.layer_id)
            buffer_roles = roles_by_layer.get(layer_id)
            if buffer_roles is None:
                continue
            if buffer_roles != _DEEPSEEK_V4_CSA_ROLES:
                layout = None
                if has_csa_cache and buffer_roles == _DEEPSEEK_V4_HCA_ROLES:
                    layout = _Nvfp4LayerLayout(
                        layer_id=layer_id,
                        num_kv_heads=0,
                        tokens_per_page=0,
                        raw_row_stride_elements=0,
                        buffers=tuple(
                            _Nvfp4BufferLayout(role=str(buffer.role)) for buffer in layer.buffers
                        ),
                    )
                layouts[layer_id] = layout
                continue

            tokens_per_page = int(cache_config.tokens_per_block)
            if tokens_per_page % 4 != 0:
                raise ValueError("DeepSeek-V4 CSA Page geometry must divide tokens_per_block by 4")
            tokens_per_page //= 4

            element_bytes = 1 if runtime_type == 2 else 2
            raw_bytes = tokens_per_page * _DEEPSEEK_V4_ROW_STRIDE * element_bytes
            configured_bytes = next(
                int(buffer.size)
                for buffer in layer.buffers
                if str(buffer.role) == _DEEPSEEK_V4_COMPRESS
            )
            footer_bytes = tokens_per_page * _DEEPSEEK_V4_FOOTER_SCALE_ROW_BYTES
            if runtime_type == 2 and configured_bytes == footer_bytes:
                raise NotImplementedError(
                    "NVFP4 cold-page compression does not support DeepSeek-V4 "
                    "fp8_ds_mla footer-scale Pages"
                )
            if configured_bytes != raw_bytes:
                raise ValueError(
                    f"DeepSeek-V4 {_DEEPSEEK_V4_COMPRESS} buffer has "
                    f"{configured_bytes} bytes; expected {raw_bytes} for an ordinary runtime Page"
                )

            scale = _IDENTITY_NVFP4_SCALE
            if not is_draft:
                model_scales = self._model_scales.get(model_layers[layer_id])
                if model_scales:
                    if "k" not in model_scales:
                        raise ValueError(
                            "DeepSeek-V4 NVFP4 cold pages require a K scale when "
                            "model-layer scale metadata is present"
                        )
                    scale = model_scales["k"]

            # A compressed row is 448 NoPE elements followed by 64 RoPE elements.
            range_start, range_elements = self._calculate_quantized_range(
                _DEEPSEEK_V4_ROW_STRIDE,
                skip_rope=skip_rope,
                rope=(_DEEPSEEK_V4_NOPE_DIM, _DEEPSEEK_V4_ROW_STRIDE - _DEEPSEEK_V4_NOPE_DIM),
                buffer_name=f"cold-page layer {layer_id} {_DEEPSEEK_V4_COMPRESS}",
            )
            layouts[layer_id] = _Nvfp4LayerLayout(
                layer_id=layer_id,
                num_kv_heads=1,
                tokens_per_page=tokens_per_page,
                raw_row_stride_elements=_DEEPSEEK_V4_ROW_STRIDE,
                buffers=tuple(
                    _Nvfp4BufferLayout(
                        role=str(buffer.role),
                        scales=_Nvfp4Scales(*scale),
                        quantized_range_start=range_start,
                        quantized_range_elements=range_elements,
                    )
                    if str(buffer.role) == _DEEPSEEK_V4_COMPRESS
                    else _Nvfp4BufferLayout(role=str(buffer.role))
                    for buffer in layer.buffers
                ),
            )

        return layouts

    def build_codec_state(
        self,
        cache_config: object,
        *,
        runtime_dtype: DataType,
        pp_layers: Sequence[int],
        num_kv_heads_per_layer: Sequence[int],
        head_dim_per_layer: Sequence[int],
        is_draft: bool = False,
    ) -> _Nvfp4ColdPageCodecState:
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

        # The codec holds only the target model's config, so it cannot locate RoPE
        # in a draft model's K vectors: draft KVCMs always quantize whole vectors.
        skip_rope = self._skip_rope_quantization
        if skip_rope and is_draft:
            logger.warning(
                "skip_rope_quantization: draft-model K and V vectors become NVFP4 in full."
            )
            skip_rope = False

        deepseek_v4_layouts = {}
        if self.pretrained_config.model_type == "deepseek_v4":
            deepseek_v4_layouts = self._build_deepseek_v4_layer_layouts(
                cache_config,
                attention_layers=attention_layers,
                pp_layers=pp_layers,
                runtime_type=runtime_type if runtime_type is not None else 0,
                is_draft=is_draft,
                skip_rope=skip_rope,
            )

        layer_layouts = []
        for layer in attention_layers:
            layer_id = int(layer.layer_id)
            if layer_id in deepseek_v4_layouts:
                layout = deepseek_v4_layouts[layer_id]
                if layout is not None:
                    layer_layouts.append(layout)
                continue

            buffer_roles = {str(buffer.role) for buffer in layer.buffers}
            if "key" not in buffer_roles:
                raise NotImplementedError(
                    "NVFP4 cold-page compression requires an Attention key buffer"
                )

            compressed_roles = ("key", "value") if "value" in buffer_roles else ("key",)
            model_scales = None if is_draft else self._model_scales.get(int(pp_layers[layer_id]))
            if model_scales and set(model_scales) != {"k", "v"}:
                raise ValueError(
                    f"ModelOpt KV scales for layer {pp_layers[layer_id]} must contain both K and V"
                )
            if len(compressed_roles) == 2 and model_scales:
                scales = tuple(model_scales[role] for role in ("k", "v"))
            else:
                scales = (_IDENTITY_NVFP4_SCALE,) * len(compressed_roles)

            num_kv_heads = int(num_kv_heads_per_layer[layer_id])
            tokens_per_page = int(cache_config.tokens_per_block)
            head_dim = int(head_dim_per_layer[layer_id])
            if head_dim <= 0 or head_dim % _ELEMENTS_PER_SCALE != 0:
                raise ValueError(
                    f"NVFP4 cold pages require head_dim divisible by 16, got {head_dim}"
                )
            key_range = self._calculate_quantized_range(
                head_dim,
                skip_rope=skip_rope,
                key_only=compressed_roles == ("key",),
                buffer_name=f"cold-page layer {layer_id} key",
            )
            buffer_layouts = [
                _Nvfp4BufferLayout(
                    role=role,
                    scales=_Nvfp4Scales(*scales[index]),
                    quantized_range_start=key_range[0] if role == "key" else 0,
                    quantized_range_elements=key_range[1] if role == "key" else head_dim,
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
                    raw_row_stride_elements=head_dim,
                    buffers=tuple(buffer_layouts),
                )
            )

        layouts_by_layer = {layout.layer_id: layout for layout in layer_layouts}
        return _Nvfp4ColdPageCodecState(
            layer_layouts=layouts_by_layer,
            layer_ids=tuple(sorted(layouts_by_layer)),
            runtime_type=runtime_type if runtime_type is not None else 0,
        )

    def build_lifecycle_metadata(
        self, codec_state: _Nvfp4ColdPageCodecState, lifecycle: object
    ) -> _Nvfp4ColdPageMetadata:
        wide_rows: list[list[int]] = []
        integer_rows: list[list[int]] = []
        scale_rows: list[list[float]] = []
        cold_page_bytes = 0
        max_half_groups_per_tile = 0

        for layer_id, hot_buffers in lifecycle.layers.items():
            layout = codec_state.layer_layouts[int(layer_id)]
            expected_roles = {buffer.role for buffer in layout.buffers}
            if set(hot_buffers) != expected_roles:
                raise ValueError(f"Cold-page layer {layer_id} roles do not match its KVCM layout")
            element_bytes = 1 if codec_state.runtime_type == 2 else 2
            rows = layout.num_kv_heads * layout.tokens_per_page
            stride = layout.raw_row_stride_elements
            expected_raw_bytes = rows * stride * element_bytes

            # Cold layer: [NVFP4 data per compressed buffer][scales then copied bytes
            # per compressed buffer][buffers copied whole][16-byte padding].
            compressed = [buffer for buffer in layout.buffers if buffer.scales is not None]
            packed_bytes = {
                buffer.role: rows * buffer.quantized_range_elements // _ELEMENTS_PER_BYTE
                for buffer in compressed
            }
            scale_and_lossless_bytes = {
                buffer.role: rows * buffer.quantized_range_elements // _ELEMENTS_PER_SCALE
                + rows * (stride - buffer.quantized_range_elements) * element_bytes
                for buffer in compressed
            }
            layer_start = cold_page_bytes
            data_cursor = layer_start
            scale_cursor = layer_start + sum(packed_bytes.values())
            cursor = scale_cursor + sum(scale_and_lossless_bytes.values())

            for buffer in layout.buffers:
                is_compressed = buffer.scales is not None
                hot = hot_buffers[buffer.role]
                raw_base = int(hot.raw_base)
                raw_slot_bytes = int(hot.raw_slot_bytes)
                raw_bytes = int(hot.raw_bytes)
                if raw_base <= 0 or raw_bytes <= 0 or raw_bytes > raw_slot_bytes:
                    raise ValueError("Cold-page hot buffer has invalid address or size")

                if is_compressed:
                    data_offset, scale_offset = data_cursor, scale_cursor
                    data_cursor += packed_bytes[buffer.role]
                    scale_cursor += scale_and_lossless_bytes[buffer.role]
                    if raw_bytes != expected_raw_bytes:
                        raise ValueError("Hot buffer size does not match NVFP4 geometry")
                    if raw_base % 16 or raw_slot_bytes % 16:
                        raise ValueError(
                            "NVFP4 hot address and Slot stride must be 16-byte aligned"
                        )
                    half_groups = rows * buffer.quantized_range_elements // _ELEMENTS_PER_HALF_GROUP
                    max_half_groups_per_tile = max(
                        max_half_groups_per_tile,
                        min(half_groups, _MAX_HALF_GROUPS_PER_TILE),
                    )
                else:
                    data_offset = cursor
                    scale_offset = 0
                    cursor += raw_bytes

                transform = _NVFP4_TRANSFORM if is_compressed else _LOSSLESS_TRANSFORM

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
                        transform,
                        layout.num_kv_heads if is_compressed else 0,
                        layout.tokens_per_page if is_compressed else 0,
                        buffer.quantized_range_elements if is_compressed else 0,
                        stride if is_compressed else 0,
                        buffer.quantized_range_start if is_compressed else 0,
                    ]
                )
                buffer_scales = buffer.scales if is_compressed else _Nvfp4Scales(1.0, 1.0)
                scale_rows.append(
                    [
                        buffer_scales.nvfp4_orig_quant,
                        buffer_scales.nvfp4_quant_orig,
                        buffer_scales.fp8_orig_quant,
                        buffer_scales.fp8_quant_orig,
                    ]
                )
            layer_end = (
                (cursor + _COLD_PAGE_ALIGNMENT - 1) // _COLD_PAGE_ALIGNMENT * _COLD_PAGE_ALIGNMENT
            )
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
        return _Nvfp4ColdPageMetadata(
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
        )

    def encode_cold_pages(
        self,
        codec_state: _Nvfp4ColdPageCodecState,
        lifecycle_index: int,
        cold_base: int,
        page_indices: int,
        num_pages: int,
        stream: int,
    ) -> None:
        from tensorrt_llm.bindings.internal import kv_cache_compression as native

        metadata = codec_state.lifecycle_metadata[lifecycle_index]
        native.nvfp4_cold_page_encode(
            page_indices,
            num_pages,
            metadata.wide.data_ptr(),
            metadata.integers.data_ptr(),
            metadata.scales.data_ptr(),
            metadata.num_buffers,
            metadata.max_half_groups_per_tile,
            metadata.cold_page_bytes,
            codec_state.runtime_type,
            cold_base,
            stream,
        )

    def decode_cold_pages(
        self,
        codec_state: _Nvfp4ColdPageCodecState,
        lifecycle_index: int,
        cold_base: int,
        page_indices: int,
        num_pages: int,
        stream: int,
    ) -> None:
        from tensorrt_llm.bindings.internal import kv_cache_compression as native

        metadata = codec_state.lifecycle_metadata[lifecycle_index]
        native.nvfp4_cold_page_decode(
            page_indices,
            num_pages,
            metadata.wide.data_ptr(),
            metadata.integers.data_ptr(),
            metadata.scales.data_ptr(),
            metadata.num_buffers,
            metadata.max_half_groups_per_tile,
            metadata.cold_page_bytes,
            codec_state.runtime_type,
            cold_base,
            stream,
        )
