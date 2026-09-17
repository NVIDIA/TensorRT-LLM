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

from tensorrt_llm.quantization.modelopt_config import (
    is_modelopt_quant_config,
    read_modelopt_quant_config,
)

from ...pyexecutor.resource_manager import DataType
from .quantization_for_cold_page import ColdPageQuantizationCompression
from .row_schema import EXCLUDE, ColdPagePolicy, LayerSchema, ResolverContext, resolve_layer_schemas

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from tensorrt_llm.llmapi.llm_args import ColdPageQuantizationCompressionConfig

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
_INTEGER_FIELDS = 8
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
    """One hot buffer. ``scales`` is None for a byte-exact lossless copy; a
    quantized buffer encodes ``quantized_elements`` of each
    ``raw_row_stride_elements``-element row as NVFP4, starting at element
    ``quantized_row_offset_elements``, and preserves the elements before and
    after that run byte-for-byte."""

    role: str
    scales: _Nvfp4Scales | None = None
    quantized_row_offset_elements: int = 0
    quantized_elements: int = 0
    raw_row_stride_elements: int = 0

    @property
    def lossless_prefix_elements(self) -> int:
        return self.quantized_row_offset_elements

    @property
    def lossless_suffix_elements(self) -> int:
        return (
            self.raw_row_stride_elements
            - self.quantized_row_offset_elements
            - self.quantized_elements
        )


@dataclass(frozen=True)
class _Nvfp4LayerLayout:
    layer_id: int
    num_kv_heads: int
    tokens_per_page: int
    buffers: tuple[_Nvfp4BufferLayout, ...]

    @property
    def rows(self) -> int:
        return self.num_kv_heads * self.tokens_per_page


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
        self._policy = ColdPagePolicy(position=config.rope_precision)

    @property
    def policy(self) -> ColdPagePolicy:
        return self._policy

    def _layer_scales(self, schema: LayerSchema, is_draft: bool) -> dict[str, _ScalePair]:
        """ModelOpt global scales by ``scale_key`` for one layer, or identity.

        Regular K/V layers use checkpoint scales only when both K and V are
        present and both buffers are quantized. Key-only MLA rows stay at
        identity. DeepSeek-V4 compressed rows use the model layer's K scale.
        """

        compressed = [buffer for buffer in schema.buffers if not buffer.is_opaque]
        keyed = {buffer.scale_key for buffer in compressed if buffer.scale_key is not None}
        model_scales = (
            None
            if is_draft or schema.model_layer is None
            else self._model_scales.get(int(schema.model_layer))
        )
        if schema.family == "deepseek_v4":
            if model_scales and "k" not in model_scales:
                raise ValueError(
                    "DeepSeek-V4 NVFP4 cold pages require a K scale when "
                    "model-layer scale metadata is present"
                )
            return {"k": model_scales["k"]} if model_scales else {}
        if model_scales and set(model_scales) != {"k", "v"}:
            raise ValueError(
                f"ModelOpt KV scales for layer {schema.model_layer} must contain both K and V"
            )
        if model_scales and keyed == {"k", "v"}:
            return dict(model_scales)
        return {}

    def _layout_from_schema(self, schema: LayerSchema, is_draft: bool) -> _Nvfp4LayerLayout:
        """Turn one resolved layer schema into the kernel-facing layout.

        Quantized buffers come first in schema order, then opaque buffers, so
        every family shares one cold-page byte order.
        """

        scales = self._layer_scales(schema, is_draft)
        quantized: list[_Nvfp4BufferLayout] = []
        opaque: list[_Nvfp4BufferLayout] = []
        for buffer in schema.buffers:
            if buffer.is_opaque:
                opaque.append(_Nvfp4BufferLayout(role=buffer.role))
                continue
            where = f"cold-page layer {schema.layer_id} buffer {buffer.role!r}"
            geometry = self._policy.row_geometry(buffer, schema.family, where=where)
            scale_pair = scales.get(buffer.scale_key, _IDENTITY_NVFP4_SCALE)
            quantized.append(
                _Nvfp4BufferLayout(
                    role=buffer.role,
                    scales=_Nvfp4Scales(*scale_pair),
                    quantized_row_offset_elements=geometry.quantized_row_offset_elements,
                    quantized_elements=geometry.quantized_elements,
                    raw_row_stride_elements=geometry.raw_row_stride_elements,
                )
            )
        return _Nvfp4LayerLayout(
            layer_id=schema.layer_id,
            num_kv_heads=schema.num_kv_heads,
            tokens_per_page=schema.tokens_per_page,
            buffers=tuple(quantized + opaque),
        )

    def build_codec_state(
        self,
        cache_config: object,
        *,
        runtime_dtype: DataType,
        pp_layers: Sequence[int],
        num_kv_heads_per_layer: Sequence[int],
        head_dim_per_layer: Sequence[int],
        is_draft: bool = False,
        pretrained_config: object | None = None,
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

        context = ResolverContext(
            pretrained_config=(
                pretrained_config if pretrained_config is not None else self.pretrained_config
            ),
            policy=self._policy,
            pp_layers=tuple(int(layer) for layer in pp_layers),
            tokens_per_block=int(cache_config.tokens_per_block),
            runtime_type=runtime_type if runtime_type is not None else 0,
            is_draft=is_draft,
            num_kv_heads_per_layer=tuple(int(heads) for heads in num_kv_heads_per_layer),
            head_dim_per_layer=tuple(int(dim) for dim in head_dim_per_layer),
        )
        layouts_by_layer: dict[int, _Nvfp4LayerLayout] = {}
        for layer_id, schema in resolve_layer_schemas(attention_layers, context).items():
            if schema is EXCLUDE:
                continue
            layouts_by_layer[layer_id] = self._layout_from_schema(schema, is_draft)

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
        element_bytes = 1 if codec_state.runtime_type == 2 else 2

        for layer_id, hot_buffers in lifecycle.layers.items():
            layout = codec_state.layer_layouts[int(layer_id)]
            expected_roles = {buffer.role for buffer in layout.buffers}
            if set(hot_buffers) != expected_roles:
                raise ValueError(f"Cold-page layer {layer_id} roles do not match its KVCM layout")
            rows = layout.rows

            # Cold layer: every quantized buffer's packed data, then each
            # quantized buffer's scales followed by its lossless row bytes,
            # then the opaque buffers, then alignment padding.
            def packed_bytes(buffer: _Nvfp4BufferLayout) -> int:
                return rows * buffer.quantized_elements // _ELEMENTS_PER_BYTE

            def scale_and_lossless_bytes(buffer: _Nvfp4BufferLayout) -> int:
                scale_bytes = rows * buffer.quantized_elements // _ELEMENTS_PER_SCALE
                lossless_bytes = (
                    rows
                    * (buffer.lossless_prefix_elements + buffer.lossless_suffix_elements)
                    * element_bytes
                )
                return scale_bytes + lossless_bytes

            quantized_buffers = [buffer for buffer in layout.buffers if buffer.scales is not None]
            layer_start = cold_page_bytes
            scale_start = layer_start + sum(packed_bytes(buffer) for buffer in quantized_buffers)
            cursor = scale_start + sum(
                scale_and_lossless_bytes(buffer) for buffer in quantized_buffers
            )
            data_cursor = layer_start
            scale_cursor = scale_start

            for buffer in layout.buffers:
                is_compressed = buffer.scales is not None
                hot = hot_buffers[buffer.role]
                raw_base = int(hot.raw_base)
                raw_slot_bytes = int(hot.raw_slot_bytes)
                raw_bytes = int(hot.raw_bytes)
                if raw_base <= 0 or raw_bytes <= 0 or raw_bytes > raw_slot_bytes:
                    raise ValueError("Cold-page hot buffer has invalid address or size")

                suffix_bytes_per_row = 0
                if is_compressed:
                    expected_raw_bytes = rows * buffer.raw_row_stride_elements * element_bytes
                    if raw_bytes != expected_raw_bytes:
                        raise ValueError("Hot buffer size does not match NVFP4 geometry")
                    if raw_base % 16 or raw_slot_bytes % 16:
                        raise ValueError(
                            "NVFP4 hot address and Slot stride must be 16-byte aligned"
                        )
                    data_offset = data_cursor
                    scale_offset = scale_cursor
                    data_cursor += packed_bytes(buffer)
                    scale_cursor += scale_and_lossless_bytes(buffer)
                    half_groups = rows * buffer.quantized_elements // _ELEMENTS_PER_HALF_GROUP
                    max_half_groups_per_tile = max(
                        max_half_groups_per_tile,
                        min(half_groups, _MAX_HALF_GROUPS_PER_TILE),
                    )
                    suffix_bytes_per_row = buffer.lossless_suffix_elements * element_bytes
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
                        buffer.quantized_elements if is_compressed else 0,
                        buffer.raw_row_stride_elements if is_compressed else 0,
                        suffix_bytes_per_row,
                        buffer.quantized_row_offset_elements if is_compressed else 0,
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
            metadata.wide.shape[1],
            metadata.integers.shape[1],
            metadata.scales.shape[1],
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
            metadata.wide.shape[1],
            metadata.integers.shape[1],
            metadata.scales.shape[1],
        )
